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

/// `y = alpha * A @ x + beta * y` for f32 buffers.
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
    // T240.1 keeps gemv on the scalar fallback. cuBLAS gemv lands in T240.2
    // when device-resident vectors are wired.
    for row in 0..m {
        let mut acc = 0.0f32;
        for col in 0..n {
            acc += a[row * n + col] * x[col];
        }
        y[row] = alpha * acc + beta * y[row];
    }
    Ok(())
}

/// Batched gemm: `C[b] = alpha * A[b] @ B[b] + beta * C[b]` for B=batch.
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
    for bi in 0..batch {
        let ai = &a[bi * stride_a..(bi + 1) * stride_a];
        let bi_ = &b[bi * stride_b..(bi + 1) * stride_b];
        let ci = &mut c[bi * stride_c..(bi + 1) * stride_c];
        gemm_f32(ai, bi_, ci, m, k, n, alpha, beta)?;
    }
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
        assert_eq!(y, vec![6.0, 15.0]);
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
}
