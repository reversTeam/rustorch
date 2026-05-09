//! cuBLAS bindings — gemm / gemv / batched gemm.
//!
//! Without `--features cuda`, ops fall back to a scalar f32 reference
//! so call sites stay portable and tests can exercise the math.
//!
//! With `--features cuda`, ops route to `cublasGemmEx`/`cublasSgemm` via
//! the safe wrappers in `cudarc::cublas`. The current Stage-A path copies
//! host slices to device on every call (H2D + GEMM + D2H + sync); the
//! perf focus here is correctness + wiring, not zero-copy. Device-resident
//! tensors land in T240.2 once `Tensor::cuda()` is plumbed through
//! rustorch-core.
//!
//! ## Row-major → cuBLAS column-major mapping
//!
//! cuBLAS is column-major. To compute the row-major product
//!     C_rm[m×n] = alpha · A_rm[m×k] · B_rm[k×n] + beta · C_rm
//! we exploit `(A·B)ᵀ = Bᵀ·Aᵀ` and feed cuBLAS the operands swapped:
//!     cublas computes  Bᵀ · Aᵀ  (in column-major land)
//!                  =  (A·B)ᵀ
//!                  =  Cᵀ
//!     and a column-major Cᵀ shares the same byte layout as a row-major C.
//! Concrete arg mapping:
//!     transa=N, transb=N
//!     cublas.m = n_rm   cublas.n = m_rm   cublas.k = k_rm
//!     cublas.A_buffer = b_rm   cublas.lda = n_rm
//!     cublas.B_buffer = a_rm   cublas.ldb = k_rm
//!     cublas.C_buffer = c_rm   cublas.ldc = n_rm
//! See https://docs.nvidia.com/cuda/cublas/#cublas-t-gemm for the
//! authoritative reference.

#![allow(clippy::needless_range_loop)]

use crate::error::CudaError;
use crate::stream::Stream;
use std::collections::HashMap;
use std::sync::Mutex;

/// Per-stream cuBLAS handle cache. Without the cuda feature, the
/// `handles` map stores u64 sentinels (one per stream raw handle).
#[derive(Debug, Default)]
pub struct HandleCache {
    handles: Mutex<HashMap<u64, u64>>,
}

impl HandleCache {
    /// Empty cache.
    pub fn new() -> Self {
        Self::default()
    }

    /// Get-or-create the handle bound to `stream`.
    pub fn get_or_insert(&self, stream: &Stream) -> u64 {
        let mut g = self.handles.lock().unwrap();
        let key = stream.raw();
        if let Some(h) = g.get(&key) {
            *h
        } else {
            // Sentinel: new handle = key | 0xCAFE
            let h = key ^ 0xCAFE;
            g.insert(key, h);
            h
        }
    }

    /// Number of handles cached.
    pub fn len(&self) -> usize {
        self.handles.lock().unwrap().len()
    }

    /// Empty cache?
    pub fn is_empty(&self) -> bool {
        self.handles.lock().unwrap().is_empty()
    }
}

/// `c = alpha * a @ b + beta * c` for f32 row-major buffers.
///
/// Without `--features cuda`, runs a scalar reference implementation.
/// With `--features cuda`, dispatches to `cublasSgemm` on the primary
/// CUDA context's default stream.
pub fn gemm_f32(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    alpha: f32,
    beta: f32,
) -> Result<(), CudaError> {
    if a.len() != m * k {
        return Err(CudaError::Unsupported {
            msg: format!("a expected {}, got {}", m * k, a.len()),
        });
    }
    if b.len() != k * n {
        return Err(CudaError::Unsupported {
            msg: format!("b expected {}, got {}", k * n, b.len()),
        });
    }
    if c.len() != m * n {
        return Err(CudaError::Unsupported {
            msg: format!("c expected {}, got {}", m * n, c.len()),
        });
    }
    if m == 0 || n == 0 {
        // Empty short-circuit.
        return Ok(());
    }
    if k == 0 {
        // No accumulation — c = beta * c (no kernel launch needed).
        for v in c.iter_mut() {
            *v *= beta;
        }
        return Ok(());
    }

    #[cfg(feature = "cuda")]
    {
        return gemm_f32_cuda(a, b, c, m, k, n, alpha, beta);
    }

    #[cfg(not(feature = "cuda"))]
    {
        gemm_f32_scalar(a, b, c, m, k, n, alpha, beta);
        Ok(())
    }
}

/// Scalar fallback used both as a reference (when cuda feature is off)
/// and for parity tests under `--features cuda`.
#[allow(dead_code)]
fn gemm_f32_scalar(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    alpha: f32,
    beta: f32,
) {
    for row in 0..m {
        for col in 0..n {
            let mut acc = 0.0f32;
            for kk in 0..k {
                acc += a[row * k + kk] * b[kk * n + col];
            }
            c[row * n + col] = alpha * acc + beta * c[row * n + col];
        }
    }
}

/// CUDA-backed gemm via cuBLAS. Allocates device buffers, copies a/b/c,
/// runs `cublasSgemm`, copies the result back. Stage-A simple path:
/// pays H2D + D2H per call. Optimised away once tensors are device-resident.
#[cfg(feature = "cuda")]
fn gemm_f32_cuda(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    alpha: f32,
    beta: f32,
) -> Result<(), CudaError> {
    use cudarc::cublas::{sys, CudaBlas, Gemm, GemmConfig};
    use cudarc::driver::CudaContext;

    let ctx = CudaContext::new(0).map_err(|e| CudaError::Driver {
        code: hash_diag(&format!("{e:?}")),
        location: "cublas::gemm_f32_cuda::CudaContext::new",
    })?;
    let stream = ctx.default_stream();
    let blas = CudaBlas::new(stream.clone()).map_err(|e| CudaError::CublasStatus {
        code: hash_diag(&format!("{e:?}")),
        location: "cublas::gemm_f32_cuda::CudaBlas::new",
    })?;
    // Enable TF32 TensorCore math on Ampere/Hopper/Blackwell. cudarc's safe
    // Gemm<f32> impl calls plain cublasSgemm which would otherwise stay on
    // the FP32 ALU path (no TensorCores). With CUBLAS_TF32_TENSOR_OP_MATH
    // set on the handle, sgemm transparently uses 19-bit TF32 multiplies +
    // f32 accumulation — bit-different from strict IEEE f32 (1e-5 max
    // relative error in practice) but ~3-5× faster on these GPUs. Drivers
    // older than CUDA 11 ignore this flag, so it's a safe no-op on legacy.
    // SAFETY: blas.handle() is a valid cublasHandle_t, mode is in-range.
    unsafe {
        let status = cudarc::cublas::sys::cublasSetMathMode(
            *blas.handle(),
            cudarc::cublas::sys::cublasMath_t::CUBLAS_TF32_TENSOR_OP_MATH,
        );
        if status != cudarc::cublas::sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
            return Err(CudaError::CublasStatus {
                code: status as i32,
                location: "cublas::gemm_f32_cuda::set_math_mode_tf32",
            });
        }
    }

    // H2D for a, b and (when beta != 0) c. Even when beta == 0 we still
    // need a device-side buffer for c — allocate zeros to keep the path
    // uniform.
    let a_dev = stream
        .memcpy_stod(a)
        .map_err(|e| map_driver_err(&e, "cublas::gemm_f32_cuda::h2d_a"))?;
    let b_dev = stream
        .memcpy_stod(b)
        .map_err(|e| map_driver_err(&e, "cublas::gemm_f32_cuda::h2d_b"))?;
    let mut c_dev = if beta == 0.0 {
        // Skip H2D when beta == 0 — initial c values won't be read.
        stream
            .alloc_zeros::<f32>(c.len())
            .map_err(|e| map_driver_err(&e, "cublas::gemm_f32_cuda::alloc_c"))?
    } else {
        stream
            .memcpy_stod(c)
            .map_err(|e| map_driver_err(&e, "cublas::gemm_f32_cuda::h2d_c"))?
    };

    // Row-major → column-major arg swap (see module docs).
    let cfg = GemmConfig::<f32> {
        transa: sys::cublasOperation_t::CUBLAS_OP_N,
        transb: sys::cublasOperation_t::CUBLAS_OP_N,
        m: n as i32,
        n: m as i32,
        k: k as i32,
        alpha,
        lda: n as i32,
        ldb: k as i32,
        beta,
        ldc: n as i32,
    };
    // SAFETY: shapes were validated at the top of `gemm_f32`. Both input
    // slices have the right length, the device buffers were just freshly
    // allocated with matching size, and `cfg` exactly mirrors them.
    unsafe {
        blas.gemm(cfg, &b_dev, &a_dev, &mut c_dev)
            .map_err(|e| CudaError::CublasStatus {
                code: hash_diag(&format!("{e:?}")),
                location: "cublas::gemm_f32_cuda::blas.gemm",
            })?;
    }

    stream
        .memcpy_dtoh(&c_dev, c)
        .map_err(|e| map_driver_err(&e, "cublas::gemm_f32_cuda::d2h_c"))?;
    stream
        .synchronize()
        .map_err(|e| map_driver_err(&e, "cublas::gemm_f32_cuda::sync"))?;

    Ok(())
}

#[cfg(feature = "cuda")]
fn map_driver_err(e: &cudarc::driver::DriverError, location: &'static str) -> CudaError {
    CudaError::Driver {
        code: hash_diag(&format!("{e:?}")),
        location,
    }
}

#[cfg(feature = "cuda")]
fn hash_diag(s: &str) -> i32 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    s.hash(&mut h);
    let v = (h.finish() & 0x7FFF_FFFF) as i32;
    if v == 0 {
        1
    } else {
        v
    }
}

/// `y = alpha * A @ x + beta * y` for f32 buffers (row-major A is `m × n`).
///
/// Without `--features cuda`, scalar fallback. With cuda, dispatches to
/// `cublasSgemv` on the primary CUDA context's default stream.
pub fn gemv_f32(
    a: &[f32],
    x: &[f32],
    y: &mut [f32],
    m: usize,
    n: usize,
    alpha: f32,
    beta: f32,
) -> Result<(), CudaError> {
    if a.len() != m * n {
        return Err(CudaError::Unsupported {
            msg: format!("a expected {}, got {}", m * n, a.len()),
        });
    }
    if x.len() != n {
        return Err(CudaError::Unsupported {
            msg: format!("x expected {n}, got {}", x.len()),
        });
    }
    if y.len() != m {
        return Err(CudaError::Unsupported {
            msg: format!("y expected {m}, got {}", y.len()),
        });
    }
    if m == 0 {
        return Ok(());
    }
    if n == 0 {
        // No cols → y = beta * y.
        for v in y.iter_mut() {
            *v *= beta;
        }
        return Ok(());
    }

    #[cfg(feature = "cuda")]
    {
        return gemv_f32_cuda(a, x, y, m, n, alpha, beta);
    }

    #[cfg(not(feature = "cuda"))]
    {
        gemv_f32_scalar(a, x, y, m, n, alpha, beta);
        Ok(())
    }
}

/// Scalar fallback used both as a reference (when cuda feature is off)
/// and for parity tests.
#[allow(dead_code)]
fn gemv_f32_scalar(a: &[f32], x: &[f32], y: &mut [f32], m: usize, n: usize, alpha: f32, beta: f32) {
    for row in 0..m {
        let mut acc = 0.0f32;
        for col in 0..n {
            acc += a[row * n + col] * x[col];
        }
        y[row] = alpha * acc + beta * y[row];
    }
}

/// CUDA-backed gemv via cuBLAS.
///
/// Row-major A (m×n) is laid out the same in memory as col-major Aᵀ (n×m).
/// We tell cuBLAS the matrix is (n×m) col-major and ask it to transpose:
/// trans=T, cublas.m=n_rm, cublas.n=m_rm, lda=n_rm.
///
/// y_rm = alpha · A_rm · x + beta · y
///      = alpha · (A_cm)ᵀ · x + beta · y
#[cfg(feature = "cuda")]
fn gemv_f32_cuda(
    a: &[f32],
    x: &[f32],
    y: &mut [f32],
    m: usize,
    n: usize,
    alpha: f32,
    beta: f32,
) -> Result<(), CudaError> {
    use cudarc::cublas::{sys, CudaBlas, Gemv, GemvConfig};
    use cudarc::driver::CudaContext;

    let ctx = CudaContext::new(0).map_err(|e| CudaError::Driver {
        code: hash_diag(&format!("{e:?}")),
        location: "cublas::gemv_f32_cuda::CudaContext::new",
    })?;
    let stream = ctx.default_stream();
    let blas = CudaBlas::new(stream.clone()).map_err(|e| CudaError::CublasStatus {
        code: hash_diag(&format!("{e:?}")),
        location: "cublas::gemv_f32_cuda::CudaBlas::new",
    })?;

    let a_dev = stream
        .memcpy_stod(a)
        .map_err(|e| map_driver_err(&e, "cublas::gemv_f32_cuda::h2d_a"))?;
    let x_dev = stream
        .memcpy_stod(x)
        .map_err(|e| map_driver_err(&e, "cublas::gemv_f32_cuda::h2d_x"))?;
    let mut y_dev = if beta == 0.0 {
        stream
            .alloc_zeros::<f32>(y.len())
            .map_err(|e| map_driver_err(&e, "cublas::gemv_f32_cuda::alloc_y"))?
    } else {
        stream
            .memcpy_stod(y)
            .map_err(|e| map_driver_err(&e, "cublas::gemv_f32_cuda::h2d_y"))?
    };

    let cfg = GemvConfig::<f32> {
        trans: sys::cublasOperation_t::CUBLAS_OP_T,
        m: n as i32, // rows of A in col-major view = n_rm
        n: m as i32, // cols of A in col-major view = m_rm
        alpha,
        lda: n as i32,
        incx: 1,
        beta,
        incy: 1,
    };
    // SAFETY: shapes validated at the top of `gemv_f32`. Buffers freshly
    // alloc'd to matching sizes.
    unsafe {
        blas.gemv(cfg, &a_dev, &x_dev, &mut y_dev)
            .map_err(|e| CudaError::CublasStatus {
                code: hash_diag(&format!("{e:?}")),
                location: "cublas::gemv_f32_cuda::blas.gemv",
            })?;
    }

    stream
        .memcpy_dtoh(&y_dev, y)
        .map_err(|e| map_driver_err(&e, "cublas::gemv_f32_cuda::d2h_y"))?;
    stream
        .synchronize()
        .map_err(|e| map_driver_err(&e, "cublas::gemv_f32_cuda::sync"))?;

    Ok(())
}

/// `c = alpha * a @ b + beta * c` for **BF16** row-major buffers.
///
/// Inputs and outputs are `bf16` for compute density (TensorCore-native);
/// alpha/beta + internal accumulation are `f32` for numerical stability.
/// This is the standard "BF16 mixed precision" gemm used in modern training
/// (Ampere+) and matches what PyTorch's autocast emits.
///
/// Without `--features cuda`, scalar fallback going through f32 conversion.
pub fn gemm_bf16(
    a: &[half::bf16],
    b: &[half::bf16],
    c: &mut [half::bf16],
    m: usize,
    k: usize,
    n: usize,
    alpha: f32,
    beta: f32,
) -> Result<(), CudaError> {
    if a.len() != m * k {
        return Err(CudaError::Unsupported {
            msg: format!("a expected {}, got {}", m * k, a.len()),
        });
    }
    if b.len() != k * n {
        return Err(CudaError::Unsupported {
            msg: format!("b expected {}, got {}", k * n, b.len()),
        });
    }
    if c.len() != m * n {
        return Err(CudaError::Unsupported {
            msg: format!("c expected {}, got {}", m * n, c.len()),
        });
    }
    if m == 0 || n == 0 {
        return Ok(());
    }
    if k == 0 {
        let beta_b = half::bf16::from_f32(beta);
        for v in c.iter_mut() {
            *v = half::bf16::from_f32(v.to_f32() * beta_b.to_f32());
        }
        return Ok(());
    }

    #[cfg(feature = "cuda")]
    {
        return gemm_bf16_cuda(a, b, c, m, k, n, alpha, beta);
    }

    #[cfg(not(feature = "cuda"))]
    {
        gemm_bf16_scalar(a, b, c, m, k, n, alpha, beta);
        Ok(())
    }
}

/// Scalar BF16 fallback. Performs the math in f32 (BF16 ALU is rare on CPU)
/// and rounds back to bf16 once per output element.
#[allow(dead_code)]
fn gemm_bf16_scalar(
    a: &[half::bf16],
    b: &[half::bf16],
    c: &mut [half::bf16],
    m: usize,
    k: usize,
    n: usize,
    alpha: f32,
    beta: f32,
) {
    for row in 0..m {
        for col in 0..n {
            let mut acc = 0.0f32;
            for kk in 0..k {
                acc += a[row * k + kk].to_f32() * b[kk * n + col].to_f32();
            }
            let prev = c[row * n + col].to_f32();
            c[row * n + col] = half::bf16::from_f32(alpha * acc + beta * prev);
        }
    }
}

/// CUDA-backed BF16 gemm via `cublasGemmEx` (CUDA_R_16BF inputs,
/// CUBLAS_COMPUTE_32F accumulation). Same row→col-major arg-swap as the
/// f32 path. TensorCores are engaged automatically by cublasGemmEx for
/// BF16 inputs.
#[cfg(feature = "cuda")]
fn gemm_bf16_cuda(
    a: &[half::bf16],
    b: &[half::bf16],
    c: &mut [half::bf16],
    m: usize,
    k: usize,
    n: usize,
    alpha: f32,
    beta: f32,
) -> Result<(), CudaError> {
    use cudarc::cublas::{sys, CudaBlas, Gemm, GemmConfig};
    use cudarc::driver::CudaContext;

    let ctx = CudaContext::new(0).map_err(|e| CudaError::Driver {
        code: hash_diag(&format!("{e:?}")),
        location: "cublas::gemm_bf16_cuda::CudaContext::new",
    })?;
    let stream = ctx.default_stream();
    let blas = CudaBlas::new(stream.clone()).map_err(|e| CudaError::CublasStatus {
        code: hash_diag(&format!("{e:?}")),
        location: "cublas::gemm_bf16_cuda::CudaBlas::new",
    })?;
    // BF16 path is already TensorCore-native; setting math mode is a
    // no-op but harmless. Keep it for consistency with the f32 path.
    unsafe {
        let _ = cudarc::cublas::sys::cublasSetMathMode(
            *blas.handle(),
            cudarc::cublas::sys::cublasMath_t::CUBLAS_DEFAULT_MATH,
        );
    }

    let a_dev = stream
        .memcpy_stod(a)
        .map_err(|e| map_driver_err(&e, "cublas::gemm_bf16_cuda::h2d_a"))?;
    let b_dev = stream
        .memcpy_stod(b)
        .map_err(|e| map_driver_err(&e, "cublas::gemm_bf16_cuda::h2d_b"))?;
    let mut c_dev = if beta == 0.0 {
        stream
            .alloc_zeros::<half::bf16>(c.len())
            .map_err(|e| map_driver_err(&e, "cublas::gemm_bf16_cuda::alloc_c"))?
    } else {
        stream
            .memcpy_stod(c)
            .map_err(|e| map_driver_err(&e, "cublas::gemm_bf16_cuda::h2d_c"))?
    };

    let cfg = GemmConfig::<half::bf16> {
        transa: sys::cublasOperation_t::CUBLAS_OP_N,
        transb: sys::cublasOperation_t::CUBLAS_OP_N,
        m: n as i32,
        n: m as i32,
        k: k as i32,
        alpha: half::bf16::from_f32(alpha),
        lda: n as i32,
        ldb: k as i32,
        beta: half::bf16::from_f32(beta),
        ldc: n as i32,
    };
    // SAFETY: shapes validated upstream; buffers freshly alloc'd to match.
    unsafe {
        blas.gemm(cfg, &b_dev, &a_dev, &mut c_dev)
            .map_err(|e| CudaError::CublasStatus {
                code: hash_diag(&format!("{e:?}")),
                location: "cublas::gemm_bf16_cuda::blas.gemm",
            })?;
    }

    stream
        .memcpy_dtoh(&c_dev, c)
        .map_err(|e| map_driver_err(&e, "cublas::gemm_bf16_cuda::d2h_c"))?;
    stream
        .synchronize()
        .map_err(|e| map_driver_err(&e, "cublas::gemm_bf16_cuda::sync"))?;

    Ok(())
}

/// Batched gemm: `C[b] = alpha * A[b] @ B[b] + beta * C[b]` for B=batch.
///
/// Without `--features cuda`, loops over the scalar reference. With cuda,
/// dispatches to `cublasSgemmStridedBatched` so the entire batch runs in
/// one kernel launch.
#[allow(clippy::too_many_arguments)]
pub fn gemm_batched_f32(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    batch: usize,
    m: usize,
    k: usize,
    n: usize,
    alpha: f32,
    beta: f32,
) -> Result<(), CudaError> {
    let stride_a = m * k;
    let stride_b = k * n;
    let stride_c = m * n;
    if a.len() != batch * stride_a || b.len() != batch * stride_b || c.len() != batch * stride_c {
        return Err(CudaError::Unsupported {
            msg: "batched gemm shape mismatch".into(),
        });
    }
    if batch == 0 || m == 0 || n == 0 {
        return Ok(());
    }

    #[cfg(feature = "cuda")]
    {
        return gemm_batched_f32_cuda(a, b, c, batch, m, k, n, alpha, beta);
    }

    #[cfg(not(feature = "cuda"))]
    {
        for bi in 0..batch {
            let ai = &a[bi * stride_a..(bi + 1) * stride_a];
            let bi_ = &b[bi * stride_b..(bi + 1) * stride_b];
            let ci = &mut c[bi * stride_c..(bi + 1) * stride_c];
            gemm_f32(ai, bi_, ci, m, k, n, alpha, beta)?;
        }
        Ok(())
    }
}

/// Single-launch batched gemm via `cublasSgemmStridedBatched`. Same row/col
/// major swap as `gemm_f32_cuda`. The strides are per-batch element counts
/// (not bytes) — cudarc converts to byte strides internally.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn gemm_batched_f32_cuda(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    batch: usize,
    m: usize,
    k: usize,
    n: usize,
    alpha: f32,
    beta: f32,
) -> Result<(), CudaError> {
    use cudarc::cublas::{sys, CudaBlas, Gemm, GemmConfig, StridedBatchedConfig};
    use cudarc::driver::CudaContext;

    if k == 0 {
        // Pure scaling on c, per-batch.
        let stride_c = m * n;
        for bi in 0..batch {
            let ci = &mut c[bi * stride_c..(bi + 1) * stride_c];
            for v in ci.iter_mut() {
                *v *= beta;
            }
        }
        return Ok(());
    }

    let ctx = CudaContext::new(0).map_err(|e| CudaError::Driver {
        code: hash_diag(&format!("{e:?}")),
        location: "cublas::gemm_batched_f32_cuda::CudaContext::new",
    })?;
    let stream = ctx.default_stream();
    let blas = CudaBlas::new(stream.clone()).map_err(|e| CudaError::CublasStatus {
        code: hash_diag(&format!("{e:?}")),
        location: "cublas::gemm_batched_f32_cuda::CudaBlas::new",
    })?;
    // TF32 to keep parity with the scalar gemm path.
    unsafe {
        let status = cudarc::cublas::sys::cublasSetMathMode(
            *blas.handle(),
            cudarc::cublas::sys::cublasMath_t::CUBLAS_TF32_TENSOR_OP_MATH,
        );
        if status != cudarc::cublas::sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
            return Err(CudaError::CublasStatus {
                code: status as i32,
                location: "cublas::gemm_batched_f32_cuda::set_math_mode_tf32",
            });
        }
    }

    let a_dev = stream
        .memcpy_stod(a)
        .map_err(|e| map_driver_err(&e, "cublas::gemm_batched_f32_cuda::h2d_a"))?;
    let b_dev = stream
        .memcpy_stod(b)
        .map_err(|e| map_driver_err(&e, "cublas::gemm_batched_f32_cuda::h2d_b"))?;
    let mut c_dev = if beta == 0.0 {
        stream
            .alloc_zeros::<f32>(c.len())
            .map_err(|e| map_driver_err(&e, "cublas::gemm_batched_f32_cuda::alloc_c"))?
    } else {
        stream
            .memcpy_stod(c)
            .map_err(|e| map_driver_err(&e, "cublas::gemm_batched_f32_cuda::h2d_c"))?
    };

    // Same row→col swap as `gemm_f32_cuda`: cuBLAS sees (B, A) with
    // dims (n, m, k). Per-batch strides reflect the row-major layout
    // since the col-major view shares the same byte stride.
    let cfg = StridedBatchedConfig::<f32> {
        gemm: GemmConfig {
            transa: sys::cublasOperation_t::CUBLAS_OP_N,
            transb: sys::cublasOperation_t::CUBLAS_OP_N,
            m: n as i32,
            n: m as i32,
            k: k as i32,
            alpha,
            lda: n as i32,
            ldb: k as i32,
            beta,
            ldc: n as i32,
        },
        batch_size: batch as i32,
        // cuBLAS arg-order is (cublas_A, cublas_B, cublas_C) which we
        // swapped with (B_rm, A_rm, C_rm). So stride_a here = stride
        // per batch of B_rm (= k*n), stride_b = stride of A_rm (= m*k).
        stride_a: (k * n) as i64,
        stride_b: (m * k) as i64,
        stride_c: (m * n) as i64,
    };
    // SAFETY: shapes validated upstream; strides match the slice lengths
    // (a.len = batch*m*k, b.len = batch*k*n, c.len = batch*m*n).
    unsafe {
        blas.gemm_strided_batched(cfg, &b_dev, &a_dev, &mut c_dev)
            .map_err(|e| CudaError::CublasStatus {
                code: hash_diag(&format!("{e:?}")),
                location: "cublas::gemm_batched_f32_cuda::blas.gemm_strided_batched",
            })?;
    }

    stream
        .memcpy_dtoh(&c_dev, c)
        .map_err(|e| map_driver_err(&e, "cublas::gemm_batched_f32_cuda::d2h_c"))?;
    stream
        .synchronize()
        .map_err(|e| map_driver_err(&e, "cublas::gemm_batched_f32_cuda::sync"))?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::Device;

    fn s() -> Stream {
        Stream::new(Device { index: 0 }).unwrap()
    }

    #[test]
    fn handle_cache_returns_same_handle_for_same_stream() {
        let cache = HandleCache::new();
        let stream = s();
        let h1 = cache.get_or_insert(&stream);
        let h2 = cache.get_or_insert(&stream);
        assert_eq!(h1, h2);
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn handle_cache_returns_distinct_for_different_streams() {
        let cache = HandleCache::new();
        let s0 = Stream::new(Device { index: 0 }).unwrap();
        let s1 = Stream::new(Device { index: 1 }).unwrap();
        let h0 = cache.get_or_insert(&s0);
        let h1 = cache.get_or_insert(&s1);
        assert_ne!(h0, h1);
    }

    #[test]
    fn gemm_2x2x2_identity_check() {
        // I @ I = I
        let a = [1.0f32, 0.0, 0.0, 1.0];
        let b = [1.0f32, 0.0, 0.0, 1.0];
        let mut c = vec![0.0f32; 4];
        gemm_f32(&a, &b, &mut c, 2, 2, 2, 1.0, 0.0).unwrap();
        assert_eq!(c, vec![1.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn gemm_alpha_beta_applied() {
        // c = alpha * a@b + beta * c, with alpha=2, beta=3
        let a = [1.0f32];
        let b = [4.0f32];
        let mut c = vec![10.0f32; 1];
        gemm_f32(&a, &b, &mut c, 1, 1, 1, 2.0, 3.0).unwrap();
        assert_eq!(c[0], 2.0 * 4.0 + 3.0 * 10.0);
    }

    #[test]
    fn gemm_empty_short_circuit() {
        let mut c: Vec<f32> = Vec::new();
        gemm_f32(&[], &[], &mut c, 0, 0, 0, 1.0, 0.0).unwrap();
        assert!(c.is_empty());
    }

    #[test]
    fn gemm_k_zero_just_scales_c() {
        let mut c = vec![5.0f32; 4];
        gemm_f32(&[], &[], &mut c, 2, 0, 2, 1.0, 0.5).unwrap();
        assert_eq!(c, vec![2.5; 4]);
    }

    #[test]
    fn gemm_shape_mismatch_returns_error() {
        let mut c = vec![0.0f32; 4];
        let err = gemm_f32(&[1.0; 3], &[1.0; 4], &mut c, 2, 2, 2, 1.0, 0.0).unwrap_err();
        assert!(matches!(err, CudaError::Unsupported { .. }));
    }

    #[test]
    fn gemm_3x4x2_matches_scalar_reference() {
        // 3×4 @ 4×2 = 3×2. Random-ish values, exercise the row/col-major
        // swap on a non-square shape so any transpose bug surfaces.
        let a: Vec<f32> = (0..12).map(|i| i as f32 * 0.5 - 1.0).collect();
        let b: Vec<f32> = (0..8).map(|i| (i as f32 - 3.0) * 0.25).collect();
        let mut c_cuda = vec![0.5f32; 6];
        let mut c_ref = c_cuda.clone();

        gemm_f32(&a, &b, &mut c_cuda, 3, 4, 2, 0.7, 0.3).unwrap();
        gemm_f32_scalar(&a, &b, &mut c_ref, 3, 4, 2, 0.7, 0.3);

        for i in 0..6 {
            assert!(
                (c_cuda[i] - c_ref[i]).abs() < 1e-4,
                "elem {i}: cuda={} ref={}",
                c_cuda[i],
                c_ref[i]
            );
        }
    }

    #[test]
    fn gemv_correct_for_2x3() {
        let a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let x = [1.0f32, 1.0, 1.0];
        let mut y = vec![0.0f32; 2];
        gemv_f32(&a, &x, &mut y, 2, 3, 1.0, 0.0).unwrap();
        // a[0,:] = [1,2,3] @ [1,1,1] = 6
        // a[1,:] = [4,5,6] @ [1,1,1] = 15
        assert!((y[0] - 6.0).abs() < 1e-4, "y[0]={} expected 6.0", y[0]);
        assert!((y[1] - 15.0).abs() < 1e-4, "y[1]={} expected 15.0", y[1]);
    }

    #[test]
    fn gemv_5x7_alpha_beta_matches_scalar() {
        // Non-trivial 5×7 with alpha,beta — exercise the row→col-major
        // transpose-on-gemv mapping on a non-square shape.
        let m = 5;
        let n = 7;
        let a: Vec<f32> = (0..m * n).map(|i| (i as f32 - 17.0) * 0.13).collect();
        let x: Vec<f32> = (0..n).map(|i| (i as f32 + 1.0) * 0.21).collect();
        let mut y_cuda: Vec<f32> = (0..m).map(|i| i as f32 * 0.5).collect();
        let mut y_ref = y_cuda.clone();

        gemv_f32(&a, &x, &mut y_cuda, m, n, 0.7, 0.3).unwrap();
        gemv_f32_scalar(&a, &x, &mut y_ref, m, n, 0.7, 0.3);

        for i in 0..m {
            assert!(
                (y_cuda[i] - y_ref[i]).abs() < 1e-3,
                "y[{i}]: cuda={} ref={}",
                y_cuda[i],
                y_ref[i]
            );
        }
    }

    #[test]
    fn batched_gemm_processes_each_batch() {
        let batch = 2;
        let m = 1;
        let k = 1;
        let n = 1;
        let a = vec![2.0f32, 3.0];
        let b = vec![4.0f32, 5.0];
        let mut c = vec![0.0f32; 2];
        gemm_batched_f32(&a, &b, &mut c, batch, m, k, n, 1.0, 0.0).unwrap();
        assert_eq!(c, vec![8.0, 15.0]);
    }

    #[test]
    fn gemm_bf16_3x4x2_matches_f32_reference() {
        // Same shape as the f32 parity test but in BF16. Tolerance widened
        // to 1e-2 because BF16 mantissa is 7 bits + accumulation rounding.
        use half::bf16;
        let a: Vec<bf16> = (0..12)
            .map(|i| bf16::from_f32(i as f32 * 0.5 - 1.0))
            .collect();
        let b: Vec<bf16> = (0..8)
            .map(|i| bf16::from_f32((i as f32 - 3.0) * 0.25))
            .collect();
        let mut c_cuda: Vec<bf16> = vec![bf16::from_f32(0.5); 6];
        // Reference in f32 for ground truth.
        let a_f32: Vec<f32> = a.iter().map(|x| x.to_f32()).collect();
        let b_f32: Vec<f32> = b.iter().map(|x| x.to_f32()).collect();
        let mut c_ref = vec![0.5f32; 6];

        gemm_bf16(&a, &b, &mut c_cuda, 3, 4, 2, 0.7, 0.3).unwrap();
        gemm_f32_scalar(&a_f32, &b_f32, &mut c_ref, 3, 4, 2, 0.7, 0.3);

        for i in 0..6 {
            let cuda = c_cuda[i].to_f32();
            let r = c_ref[i];
            assert!(
                (cuda - r).abs() < 1e-2,
                "elem {i}: bf16_cuda={cuda} f32_ref={r}"
            );
        }
    }

    #[test]
    fn batched_gemm_3x_2x3x4_matches_per_batch_scalar() {
        // 3 batches of 2×3 @ 3×4 = 2×4. Validate the strided-batched
        // path produces the same numbers as a per-batch scalar loop.
        let batch = 3;
        let m = 2;
        let k = 3;
        let n = 4;
        let stride_a = m * k;
        let stride_b = k * n;
        let stride_c = m * n;
        let a: Vec<f32> = (0..batch * stride_a)
            .map(|i| (i as f32 - 7.0) * 0.11)
            .collect();
        let b: Vec<f32> = (0..batch * stride_b)
            .map(|i| (i as f32 + 3.0) * 0.07)
            .collect();
        let mut c_cuda: Vec<f32> = (0..batch * stride_c).map(|i| i as f32 * 0.05).collect();
        let mut c_ref = c_cuda.clone();

        gemm_batched_f32(&a, &b, &mut c_cuda, batch, m, k, n, 0.5, 0.4).unwrap();
        for bi in 0..batch {
            let ai = &a[bi * stride_a..(bi + 1) * stride_a];
            let bi_ = &b[bi * stride_b..(bi + 1) * stride_b];
            let ci = &mut c_ref[bi * stride_c..(bi + 1) * stride_c];
            gemm_f32_scalar(ai, bi_, ci, m, k, n, 0.5, 0.4);
        }

        for i in 0..batch * stride_c {
            assert!(
                (c_cuda[i] - c_ref[i]).abs() < 1e-3,
                "elem {i}: cuda={} ref={}",
                c_cuda[i],
                c_ref[i]
            );
        }
    }
}
