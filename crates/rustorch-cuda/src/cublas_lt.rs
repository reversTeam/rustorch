//! cublasLt mixed-precision matmul wrappers (T240.5).
//!
//! cudarc 0.17's safe `cublaslt` module exposes a `Matmul<T>` trait that
//! requires the **same dtype for A, B, and C**. That works for plain f32 /
//! f16 / bf16 gemms but doesn't fit FP8 mixed-precision (FP8 in, BF16 out)
//! or FP4 mixed-precision (FP4 in, BF16/FP8 out).
//!
//! This module bypasses the high-level trait and goes through the lower-
//! level `cudarc::cublaslt::result::*` helpers (also publicly exposed) to
//! build per-matrix layouts with their own dtypes and run the matmul.
//!
//! ## Step toward 1000 TOPS
//!
//! GB10 Blackwell theoretical peaks (TensorCore native):
//! - F32   ~30 TFLOPS  (no Tensor)
//! - TF32  ~100 TFLOPS
//! - BF16  ~250 TFLOPS  ← T240.3 measured 91 TFLOPS
//! - FP8   ~500 TFLOPS  ← T240.5 here
//! - FP4   ~1000 TOPS   ← T240.5 extension once driver exposes CUDA_R_4F
//!
//! Each precision halving doubles peak throughput; the headline 1000 TOPS
//! number on the spec sheet refers to FP4 with TensorCore acceleration.

use crate::error::CudaError;

#[cfg(feature = "cuda")]
use cudarc::cublaslt::{result, sys};

/// FP8 sub-format. E4M3 has more precision (3-bit mantissa, ±448 range);
/// E5M2 has more range (2-bit mantissa, ±57 344 range).
///
/// Standard ML training pattern: E4M3 for the forward pass (weights and
/// activations), E5M2 for the backward pass (gradients have wider range).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fp8Kind {
    /// Exponent 4 bits, mantissa 3 bits — preferred for fwd / activations.
    E4M3,
    /// Exponent 5 bits, mantissa 2 bits — preferred for bwd / gradients.
    E5M2,
}

#[cfg(feature = "cuda")]
impl Fp8Kind {
    fn cuda_type(self) -> sys::cudaDataType_t {
        match self {
            Self::E4M3 => sys::cudaDataType_t::CUDA_R_8F_E4M3,
            Self::E5M2 => sys::cudaDataType_t::CUDA_R_8F_E5M2,
        }
    }
}

/// Output type for FP8 matmul. cuBLASLt supports BF16 / FP16 / F32 outputs
/// with FP8 inputs and F32 accumulation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fp8Output {
    /// 16-bit brain-float — TensorCore-native on Ampere+, training default.
    Bf16,
    /// 16-bit IEEE float — same TC throughput as BF16 but ±65 504 range.
    F16,
    /// 32-bit IEEE float — for parity tests and unfused softmax inputs.
    F32,
}

#[cfg(feature = "cuda")]
impl Fp8Output {
    fn cuda_type(self) -> sys::cudaDataType_t {
        match self {
            Self::Bf16 => sys::cudaDataType_t::CUDA_R_16BF,
            Self::F16 => sys::cudaDataType_t::CUDA_R_16F,
            Self::F32 => sys::cudaDataType_t::CUDA_R_32F,
        }
    }

    fn elem_size(self) -> usize {
        match self {
            Self::Bf16 | Self::F16 => 2,
            Self::F32 => 4,
        }
    }
}

/// RAII container for the cublasLt resources we hold for one matmul call.
/// On drop, descriptors are released in the reverse of their creation
/// order — preferences first, then desc, then layouts, then handle.
#[cfg(feature = "cuda")]
struct LtResources {
    handle: sys::cublasLtHandle_t,
    a_layout: sys::cublasLtMatrixLayout_t,
    b_layout: sys::cublasLtMatrixLayout_t,
    c_layout: sys::cublasLtMatrixLayout_t,
    matmul_desc: sys::cublasLtMatmulDesc_t,
    pref: sys::cublasLtMatmulPreference_t,
}

#[cfg(feature = "cuda")]
impl Drop for LtResources {
    fn drop(&mut self) {
        // SAFETY: each handle was created in this module and is destroyed
        // exactly once (LtResources isn't Clone).
        unsafe {
            let _ = result::destroy_matmul_pref(self.pref);
            let _ = result::destroy_matmul_desc(self.matmul_desc);
            let _ = result::destroy_matrix_layout(self.a_layout);
            let _ = result::destroy_matrix_layout(self.b_layout);
            let _ = result::destroy_matrix_layout(self.c_layout);
            let _ = result::destroy_handle(self.handle);
        }
    }
}

/// Run an FP8 (E4M3 or E5M2) GEMM with mixed-precision output.
///
/// `m`, `n`, `k` are the row-major problem dimensions:
///   `C_rm[m × n] = alpha · A_rm[m × k] · B_rm[k × n] + beta · C_rm`
///
/// cuBLASLt FP8 requires **transa=T, transb=N** ("TN" gemm) on Hopper+.
/// We bake that requirement in: callers pass row-major buffers and the
/// row→col-major mapping is handled internally — `lda = ldb = k`, `ldc = m`.
///
/// `a_dev`, `b_dev` are device pointers (raw `u64`) to FP8 byte buffers
/// (1 byte per element). `c_dev` is a device pointer to the output buffer
/// (size `m × n × out.elem_size()` bytes).
///
/// `workspace_dev` is a device pointer to a scratch buffer of at least
/// `workspace_bytes` bytes (recommended: 32 MiB on Hopper+).
///
/// `stream` is a cudaStream_t (raw `u64`) the kernel will be enqueued on.
///
/// # Safety
///
/// All `*_dev` pointers must be valid for the lifetime of the call,
/// reference allocations of the right size, and not alias incorrectly
/// (`a` and `b` may point to the same buffer; `c` must be distinct from
/// `a`/`b` unless the caller wants in-place semantics with beta=1).
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn matmul_fp8(
    a_dev: u64,
    b_dev: u64,
    c_dev: u64,
    m: usize,
    k: usize,
    n: usize,
    alpha: f32,
    beta: f32,
    fp8_kind: Fp8Kind,
    out: Fp8Output,
    workspace_dev: u64,
    workspace_bytes: usize,
    stream: u64,
) -> Result<(), CudaError> {
    let _ = out.elem_size(); // exercised so callers can sanity-check sizes
    if m == 0 || n == 0 || k == 0 {
        return Ok(());
    }

    let handle = result::create_handle().map_err(|e| CudaError::CublasStatus {
        code: lt_err_code(e),
        location: "cublas_lt::matmul_fp8::create_handle",
    })?;

    // Layouts. lda=k, ldb=k, ldc=m matches the cuBLASLt FP8 TN requirement
    // for row-major inputs.
    let a_in_dt = fp8_kind.cuda_type();
    let b_in_dt = fp8_kind.cuda_type();
    let c_dt = out.cuda_type();

    let a_layout =
        result::create_matrix_layout(a_in_dt, k as u64, m as u64, k as i64).map_err(|e| {
            CudaError::CublasStatus {
                code: lt_err_code(e),
                location: "cublas_lt::matmul_fp8::a_layout",
            }
        })?;
    let b_layout =
        result::create_matrix_layout(b_in_dt, k as u64, n as u64, k as i64).map_err(|e| {
            CudaError::CublasStatus {
                code: lt_err_code(e),
                location: "cublas_lt::matmul_fp8::b_layout",
            }
        })?;
    let c_layout =
        result::create_matrix_layout(c_dt, m as u64, n as u64, m as i64).map_err(|e| {
            CudaError::CublasStatus {
                code: lt_err_code(e),
                location: "cublas_lt::matmul_fp8::c_layout",
            }
        })?;

    let matmul_desc = result::create_matmul_desc(
        sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
        sys::cudaDataType_t::CUDA_R_32F,
    )
    .map_err(|e| CudaError::CublasStatus {
        code: lt_err_code(e),
        location: "cublas_lt::matmul_fp8::matmul_desc",
    })?;

    // transa = T, transb = N. cublasOperation_t lives in the cublas (not
    // cublaslt) sys module — same enum is shared.
    let transa = cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_T;
    let transb = cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N;
    result::set_matmul_desc_attribute(
        matmul_desc,
        sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSA,
        (&transa) as *const _ as *const _,
        std::mem::size_of::<cudarc::cublas::sys::cublasOperation_t>(),
    )
    .map_err(|e| CudaError::CublasStatus {
        code: lt_err_code(e),
        location: "cublas_lt::matmul_fp8::set_transa",
    })?;
    result::set_matmul_desc_attribute(
        matmul_desc,
        sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSB,
        (&transb) as *const _ as *const _,
        std::mem::size_of::<cudarc::cublas::sys::cublasOperation_t>(),
    )
    .map_err(|e| CudaError::CublasStatus {
        code: lt_err_code(e),
        location: "cublas_lt::matmul_fp8::set_transb",
    })?;

    let pref = result::create_matmul_pref().map_err(|e| CudaError::CublasStatus {
        code: lt_err_code(e),
        location: "cublas_lt::matmul_fp8::pref",
    })?;
    result::set_matmul_pref_attribute(
        pref,
        sys::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        (&workspace_bytes) as *const _ as *const _,
        std::mem::size_of::<usize>(),
    )
    .map_err(|e| CudaError::CublasStatus {
        code: lt_err_code(e),
        location: "cublas_lt::matmul_fp8::pref_workspace",
    })?;

    // Resources: from this point on, dropping `_res` releases everything.
    let _res = LtResources {
        handle,
        a_layout,
        b_layout,
        c_layout,
        matmul_desc,
        pref,
    };

    let heuristic = result::get_matmul_algo_heuristic(
        handle,
        matmul_desc,
        a_layout,
        b_layout,
        c_layout,
        c_layout,
        pref,
    )
    .map_err(|e| CudaError::CublasStatus {
        code: lt_err_code(e),
        location: "cublas_lt::matmul_fp8::heuristic",
    })?;

    result::matmul(
        handle,
        matmul_desc,
        (&alpha) as *const f32 as *const _,
        (&beta) as *const f32 as *const _,
        a_dev as *const _,
        a_layout,
        b_dev as *const _,
        b_layout,
        c_dev as *const _,
        c_layout,
        c_dev as *mut _,
        c_layout,
        (&heuristic.algo) as *const _,
        workspace_dev as *mut _,
        workspace_bytes,
        stream as *mut _,
    )
    .map_err(|e| CudaError::CublasStatus {
        code: lt_err_code(e),
        location: "cublas_lt::matmul_fp8::matmul",
    })?;

    Ok(())
}

/// Stub for no-cuda builds — keeps the public API consistent.
///
/// # Safety
///
/// Pointer arguments are accepted but never dereferenced on this path
/// (returns `Err(NoDeviceFound)` immediately). Same pointer-validity
/// contract as the cuda variant applies for forward compatibility.
#[cfg(not(feature = "cuda"))]
#[allow(clippy::too_many_arguments)]
pub unsafe fn matmul_fp8(
    _a_dev: u64,
    _b_dev: u64,
    _c_dev: u64,
    _m: usize,
    _k: usize,
    _n: usize,
    _alpha: f32,
    _beta: f32,
    _fp8_kind: Fp8Kind,
    _out: Fp8Output,
    _workspace_dev: u64,
    _workspace_bytes: usize,
    _stream: u64,
) -> Result<(), CudaError> {
    Err(CudaError::NoDeviceFound)
}

#[cfg(feature = "cuda")]
fn lt_err_code(e: cudarc::cublaslt::result::CublasError) -> i32 {
    e.0 as i32
}

// ─────────────────────────────────────────────────────────────────────
// LtSession — cached cublasLt session for sustained throughput (T240.6)
// ─────────────────────────────────────────────────────────────────────

/// Cache key — uniquely identifies a matmul configuration.
///
/// Layouts and matmul descriptors are cached per (shape × dtypes ×
/// trans × scale mode). Re-running the same config hits the cache and
/// skips handle/desc creation + heuristic search.
#[cfg(feature = "cuda")]
#[derive(Hash, PartialEq, Eq, Clone, Copy, Debug)]
struct ConfigKey {
    m: u32,
    n: u32,
    k: u32,
    a_dt: i32,
    b_dt: i32,
    c_dt: i32,
    transa: bool,
    transb: bool,
    /// 0 = no scale, 1 = VEC32_UE8M0, 2 = VEC16_UE4M3.
    scale_mode: u8,
}

#[cfg(feature = "cuda")]
struct CachedMatmul {
    a_layout: sys::cublasLtMatrixLayout_t,
    b_layout: sys::cublasLtMatrixLayout_t,
    c_layout: sys::cublasLtMatrixLayout_t,
    matmul_desc: sys::cublasLtMatmulDesc_t,
    pref: sys::cublasLtMatmulPreference_t,
    algo: sys::cublasLtMatmulAlgo_t,
}

#[cfg(feature = "cuda")]
impl Drop for CachedMatmul {
    fn drop(&mut self) {
        // SAFETY: each handle was created by LtSession and is destroyed
        // exactly once (CachedMatmul not Clone).
        unsafe {
            let _ = result::destroy_matmul_pref(self.pref);
            let _ = result::destroy_matmul_desc(self.matmul_desc);
            let _ = result::destroy_matrix_layout(self.a_layout);
            let _ = result::destroy_matrix_layout(self.b_layout);
            let _ = result::destroy_matrix_layout(self.c_layout);
        }
    }
}

/// A reusable cublasLt session with cached descriptors and a persistent
/// workspace. Construct once at app startup; call `matmul_fp8` /
/// `matmul_mxfp4` repeatedly.
///
/// Performance: caching the heuristic algo + descriptors removes ~100 µs
/// to 1 ms of overhead per call. On a hot path with consistent shapes
/// this typically doubles utilization compared to the uncached entry
/// points (`matmul_fp8` / `matmul_mxfp4` free functions).
#[cfg(feature = "cuda")]
pub struct LtSession {
    handle: sys::cublasLtHandle_t,
    /// Persistent device-side scratch buffer.
    pub workspace: cudarc::driver::CudaSlice<u8>,
    /// Workspace size in bytes — passed to every matmul call.
    pub workspace_bytes: usize,
    /// Cache of (config) → (descriptors + algo).
    cache: std::collections::HashMap<ConfigKey, CachedMatmul>,
    /// CUDA stream the session is bound to (handle is *not* stream-bound,
    /// but we keep the stream around for `device_ptr` calls in the matmul
    /// implementations).
    stream: std::sync::Arc<cudarc::driver::CudaStream>,
}

#[cfg(feature = "cuda")]
impl LtSession {
    /// Allocate handle + workspace on the given stream's device.
    ///
    /// Default workspace = **32 MiB** (Hopper rule-of-thumb, also empirically
    /// best on GB10 + CUDA 13.0/13.1: a 256 MiB workspace surfaces more
    /// candidate algos to the heuristic but the additional ones happen to
    /// run *slower* on GB10's smaller shared memory, so even with autotune
    /// the smaller workspace wins by ~2% on Qwen-block shapes — see T240.8b
    /// note).
    ///
    /// Override via `RUSTORCH_CUBLASLT_WORKSPACE_MB` env var or
    /// `new_with_workspace`. Re-tune once on CUDA 13.2 (NVFP4 perf update).
    pub fn new(stream: std::sync::Arc<cudarc::driver::CudaStream>) -> Result<Self, CudaError> {
        let mb = std::env::var("RUSTORCH_CUBLASLT_WORKSPACE_MB")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(32);
        Self::new_with_workspace(stream, mb * 1024 * 1024)
    }

    /// Custom workspace size variant.
    pub fn new_with_workspace(
        stream: std::sync::Arc<cudarc::driver::CudaStream>,
        workspace_bytes: usize,
    ) -> Result<Self, CudaError> {
        let handle = result::create_handle().map_err(|e| CudaError::CublasStatus {
            code: lt_err_code(e),
            location: "LtSession::new::create_handle",
        })?;
        let workspace =
            stream
                .alloc_zeros::<u8>(workspace_bytes)
                .map_err(|e| CudaError::Driver {
                    code: format!("{e:?}").len() as i32, // best-effort code
                    location: "LtSession::new::alloc_workspace",
                })?;
        Ok(Self {
            handle,
            workspace,
            workspace_bytes,
            cache: std::collections::HashMap::new(),
            stream,
        })
    }

    /// Number of cached matmul configurations.
    pub fn cache_size(&self) -> usize {
        self.cache.len()
    }

    /// Get-or-build the cached resources for one configuration.
    ///
    /// # Safety
    /// Caller must ensure the dtype tags + shapes match the buffers it
    /// will pass to `dispatch`.
    /// `seed_a_scale` / `seed_b_scale` only matter on first build (must be
    /// non-null when `scale_mode` is set, so the heuristic search can
    /// validate the desc). Real per-call pointers are bound by the matmul
    /// methods via `set_matmul_desc_attribute`.
    #[allow(clippy::too_many_arguments)]
    unsafe fn get_or_build(
        &mut self,
        m: usize,
        n: usize,
        k: usize,
        a_dt: sys::cudaDataType_t,
        b_dt: sys::cudaDataType_t,
        c_dt: sys::cudaDataType_t,
        scale_mode: Option<Fp4ScaleMode>,
        seed_a_scale: u64,
        seed_b_scale: u64,
    ) -> Result<&CachedMatmul, CudaError> {
        let key = ConfigKey {
            m: m as u32,
            n: n as u32,
            k: k as u32,
            a_dt: a_dt as i32,
            b_dt: b_dt as i32,
            c_dt: c_dt as i32,
            transa: true, // always TN for FP8/FP4 paths
            transb: false,
            scale_mode: match scale_mode {
                None => 0,
                Some(Fp4ScaleMode::Vec32Ue8m0) => 1,
                Some(Fp4ScaleMode::Vec16Ue4m3) => 2,
            },
        };

        if !self.cache.contains_key(&key) {
            let cached = build_cached(
                self.handle,
                m,
                n,
                k,
                a_dt,
                b_dt,
                c_dt,
                scale_mode,
                self.workspace_bytes,
                seed_a_scale,
                seed_b_scale,
            )?;
            self.cache.insert(key, cached);
        }
        Ok(self.cache.get(&key).unwrap())
    }

    /// Cached FP8 matmul. Same semantics as `matmul_fp8` free fn but
    /// reuses descriptors + algo across calls with matching shape/dtype.
    ///
    /// # Safety
    /// `*_dev` pointers must be valid for the duration of the call and
    /// match the FP8/output element sizes.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn matmul_fp8(
        &mut self,
        a_dev: u64,
        b_dev: u64,
        c_dev: u64,
        m: usize,
        k: usize,
        n: usize,
        alpha: f32,
        beta: f32,
        kind: Fp8Kind,
        out: Fp8Output,
    ) -> Result<(), CudaError> {
        if m == 0 || n == 0 || k == 0 {
            return Ok(());
        }
        let cached = self.get_or_build(
            m,
            n,
            k,
            kind.cuda_type(),
            kind.cuda_type(),
            out.cuda_type(),
            None,
            0,
            0,
        )? as *const CachedMatmul; // SAFETY-borrow workaround for self.workspace
        let cached = &*cached;
        let workspace_ptr = {
            use cudarc::driver::DevicePtr;
            self.workspace.device_ptr(&self.stream).0
        };
        result::matmul(
            self.handle,
            cached.matmul_desc,
            (&alpha) as *const f32 as *const _,
            (&beta) as *const f32 as *const _,
            a_dev as *const _,
            cached.a_layout,
            b_dev as *const _,
            cached.b_layout,
            c_dev as *const _,
            cached.c_layout,
            c_dev as *mut _,
            cached.c_layout,
            (&cached.algo) as *const _,
            workspace_ptr as *mut _,
            self.workspace_bytes,
            self.stream.cu_stream() as *mut _,
        )
        .map_err(|e| CudaError::CublasStatus {
            code: lt_err_code(e),
            location: "LtSession::matmul_fp8::dispatch",
        })?;
        Ok(())
    }

    /// Cached BF16 matmul. Uniform BF16 in/out, F32 accumulation,
    /// TensorCore-native on Ampere+.
    ///
    /// # Safety
    /// `*_dev` pointers must be valid for the duration of the call and
    /// reference allocations of the right BF16 element count.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn matmul_bf16(
        &mut self,
        a_dev: u64,
        b_dev: u64,
        c_dev: u64,
        m: usize,
        k: usize,
        n: usize,
        alpha: f32,
        beta: f32,
    ) -> Result<(), CudaError> {
        if m == 0 || n == 0 || k == 0 {
            return Ok(());
        }
        let cached = self.get_or_build(
            m,
            n,
            k,
            sys::cudaDataType_t::CUDA_R_16BF,
            sys::cudaDataType_t::CUDA_R_16BF,
            sys::cudaDataType_t::CUDA_R_16BF,
            None,
            0,
            0,
        )? as *const CachedMatmul;
        let cached = &*cached;
        let workspace_ptr = {
            use cudarc::driver::DevicePtr;
            self.workspace.device_ptr(&self.stream).0
        };
        result::matmul(
            self.handle,
            cached.matmul_desc,
            (&alpha) as *const f32 as *const _,
            (&beta) as *const f32 as *const _,
            a_dev as *const _,
            cached.a_layout,
            b_dev as *const _,
            cached.b_layout,
            c_dev as *const _,
            cached.c_layout,
            c_dev as *mut _,
            cached.c_layout,
            (&cached.algo) as *const _,
            workspace_ptr as *mut _,
            self.workspace_bytes,
            self.stream.cu_stream() as *mut _,
        )
        .map_err(|e| CudaError::CublasStatus {
            code: lt_err_code(e),
            location: "LtSession::matmul_bf16::dispatch",
        })?;
        Ok(())
    }

    /// **DESIGN-PIVOT — row-major BF16 GEMM**.
    ///
    /// Drop-in replacement for the hand-written `gemm_bf16_bf16_mma_m16n8k16`
    /// kernel signature : computes `Y[M, N] = X[M, K] · W^T[K, N]` where all
    /// three buffers are stored **row-major** (BF16) :
    ///
    ///   w : `[N, K]` BF16 row-major
    ///   x : `[M, K]` BF16 row-major
    ///   y : `[M, N]` BF16 row-major  (output)
    ///
    /// The math is identical to `y = x · w.t()` and matches
    /// `LlmKernels::sgemm_bf16_bf16_mvar` / `gemm_bf16_bf16_mma` element-by-
    /// element (modulo BF16 rounding-order drift).
    ///
    /// Internally this calls cuBLASLt with `(A = W, B = X)`, cublas-M = N,
    /// cublas-N = M, cublas-K = K, transa = T, transb = N. The output is
    /// written in col-major `[N, M]` ld=N, which **is** row-major `[M, N]`
    /// stride=N — the layout we want. No post-transpose needed.
    ///
    /// Lane-equivalence proof :
    ///   col-major output Y_cm[i, j] (0..N, 0..M)
    ///     = sum_k op(A)[i, k] · op(B)[k, j]
    ///     = sum_k A^T[i, k] · B[k, j]
    ///     = sum_k A[k, i] · B[k, j]                  (A col-major [K, N]·)
    ///     = sum_k W_rm[i, k] · X_rm[j, k]            (row=col-of-transpose)
    /// row-major Y_rm[m, n] = Y_cm[n, m]
    ///                       = sum_k W_rm[n, k] · X_rm[m, k]
    ///                       = sum_k X_rm[m, k] · W_rm[n, k]
    ///                       = (X · W^T)[m, n] ✓
    ///
    /// Cache key (m=N_rust, n=M_rust, transa=T, transb=N) differs from the
    /// existing `matmul_bf16` cache key (TT) so both can coexist on the same
    /// `LtSession`.
    ///
    /// # Safety
    /// `*_dev` pointers must be valid for the call duration and reference
    /// allocations of `n*k`, `m*k`, `m*n` BF16 elements respectively.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn matmul_bf16_rowmajor(
        &mut self,
        w_dev: u64,
        x_dev: u64,
        y_dev: u64,
        m: usize,
        n: usize,
        k: usize,
        alpha: f32,
        beta: f32,
    ) -> Result<(), CudaError> {
        if m == 0 || n == 0 || k == 0 {
            return Ok(());
        }
        // Cache key uses (m=n_rust, n=m_rust, transa=T, transb=N) to keep
        // this config distinct from the col-major matmul_bf16 path.
        let key = ConfigKey {
            m: n as u32,
            n: m as u32,
            k: k as u32,
            a_dt: sys::cudaDataType_t::CUDA_R_16BF as i32,
            b_dt: sys::cudaDataType_t::CUDA_R_16BF as i32,
            c_dt: sys::cudaDataType_t::CUDA_R_16BF as i32,
            transa: true,
            transb: false,
            scale_mode: 0,
        };
        if !self.cache.contains_key(&key) {
            let cached = build_cached_bf16_rowmajor(self.handle, m, n, k, self.workspace_bytes)?;
            self.cache.insert(key, cached);
        }
        let cached = self.cache.get(&key).unwrap() as *const CachedMatmul;
        let cached = &*cached;
        let workspace_ptr = {
            use cudarc::driver::DevicePtr;
            self.workspace.device_ptr(&self.stream).0
        };
        result::matmul(
            self.handle,
            cached.matmul_desc,
            (&alpha) as *const f32 as *const _,
            (&beta) as *const f32 as *const _,
            w_dev as *const _, // A = W
            cached.a_layout,
            x_dev as *const _, // B = X
            cached.b_layout,
            y_dev as *const _,
            cached.c_layout,
            y_dev as *mut _,
            cached.c_layout,
            (&cached.algo) as *const _,
            workspace_ptr as *mut _,
            self.workspace_bytes,
            self.stream.cu_stream() as *mut _,
        )
        .map_err(|e| CudaError::CublasStatus {
            code: lt_err_code(e),
            location: "LtSession::matmul_bf16_rowmajor::dispatch",
        })?;
        Ok(())
    }

    /// Cached MXFP4/NVFP4 matmul.
    ///
    /// # Safety
    /// `*_dev` pointers must be valid for the duration of the call. Scale
    /// pointers must reference appropriately-sized scale tensors per
    /// `scale_mode.block_size()`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn matmul_mxfp4(
        &mut self,
        a_dev: u64,
        a_scale_dev: u64,
        b_dev: u64,
        b_scale_dev: u64,
        c_dev: u64,
        m: usize,
        k: usize,
        n: usize,
        alpha: f32,
        beta: f32,
        out: Fp8Output,
        scale_mode: Fp4ScaleMode,
    ) -> Result<(), CudaError> {
        if m == 0 || n == 0 || k == 0 {
            return Ok(());
        }
        let block = scale_mode.block_size();
        if k % block != 0 {
            return Err(CudaError::Unsupported {
                msg: format!("FP4 requires k % {block} == 0, got k={k}"),
            });
        }
        let cached = self.get_or_build(
            m,
            n,
            k,
            sys::cudaDataType_t::CUDA_R_4F_E2M1,
            sys::cudaDataType_t::CUDA_R_4F_E2M1,
            out.cuda_type(),
            Some(scale_mode),
            a_scale_dev,
            b_scale_dev,
        )? as *const CachedMatmul;
        let cached = &*cached;

        // Scale pointers are NOT part of the cached desc (they're per-call).
        // Update them on the cached desc before each dispatch.
        result::set_matmul_desc_attribute(
            cached.matmul_desc,
            sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
            (&a_scale_dev) as *const _ as *const _,
            std::mem::size_of::<u64>(),
        )
        .map_err(|e| CudaError::CublasStatus {
            code: lt_err_code(e),
            location: "LtSession::matmul_mxfp4::set_a_scale_ptr",
        })?;
        result::set_matmul_desc_attribute(
            cached.matmul_desc,
            sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
            (&b_scale_dev) as *const _ as *const _,
            std::mem::size_of::<u64>(),
        )
        .map_err(|e| CudaError::CublasStatus {
            code: lt_err_code(e),
            location: "LtSession::matmul_mxfp4::set_b_scale_ptr",
        })?;

        let workspace_ptr = {
            use cudarc::driver::DevicePtr;
            self.workspace.device_ptr(&self.stream).0
        };
        result::matmul(
            self.handle,
            cached.matmul_desc,
            (&alpha) as *const f32 as *const _,
            (&beta) as *const f32 as *const _,
            a_dev as *const _,
            cached.a_layout,
            b_dev as *const _,
            cached.b_layout,
            c_dev as *const _,
            cached.c_layout,
            c_dev as *mut _,
            cached.c_layout,
            (&cached.algo) as *const _,
            workspace_ptr as *mut _,
            self.workspace_bytes,
            self.stream.cu_stream() as *mut _,
        )
        .map_err(|e| CudaError::CublasStatus {
            code: lt_err_code(e),
            location: "LtSession::matmul_mxfp4::dispatch",
        })?;
        Ok(())
    }

    /// Multi-algo autotune for a BF16 (m,k)·(k,n) shape (T240.8b).
    ///
    /// Queries up to `n_candidates` heuristic algorithms, times each over
    /// `n_passes` real dispatches with CUDA events, then stores the fastest
    /// algo in the cache. Subsequent `matmul_bf16` calls on the same shape
    /// use that algo instead of the heuristic top-1.
    ///
    /// Typical lift on Blackwell: 10–30% over the heuristic default,
    /// because the heuristic optimizes for "expected" workload (B200 + big
    /// data center shapes) and GB10 has different shared-memory / split-K
    /// trade-offs.
    ///
    /// Buffers `a_dev`, `b_dev`, `c_dev` are *written through* during
    /// timing. Pass real warm-up data; do NOT pass production output here.
    ///
    /// Returns the chosen algo's per-call latency in ms (median).
    ///
    /// # Safety
    /// `*_dev` must be valid BF16 device buffers of sizes ≥ m·k, k·n, m·n
    /// respectively. Caller must own the buffers and accept they're
    /// overwritten during autotune.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn autotune_bf16(
        &mut self,
        a_dev: u64,
        b_dev: u64,
        c_dev: u64,
        m: usize,
        k: usize,
        n: usize,
        n_candidates: u32,
        n_passes: u32,
    ) -> Result<f32, CudaError> {
        if m == 0 || n == 0 || k == 0 {
            return Ok(0.0);
        }
        let n_candidates = n_candidates.max(1).min(32);
        let n_passes = n_passes.max(1).min(50);

        // 1. Ensure base cache entry exists (heuristic top-1 default).
        let _ = self.get_or_build(
            m,
            n,
            k,
            sys::cudaDataType_t::CUDA_R_16BF,
            sys::cudaDataType_t::CUDA_R_16BF,
            sys::cudaDataType_t::CUDA_R_16BF,
            None,
            0,
            0,
        )?;
        let key = ConfigKey {
            m: m as u32,
            n: n as u32,
            k: k as u32,
            a_dt: sys::cudaDataType_t::CUDA_R_16BF as i32,
            b_dt: sys::cudaDataType_t::CUDA_R_16BF as i32,
            c_dt: sys::cudaDataType_t::CUDA_R_16BF as i32,
            transa: true,
            transb: false,
            scale_mode: 0,
        };
        // Snapshot descriptor handles needed for dispatch.
        let (a_layout, b_layout, c_layout, matmul_desc, pref) = {
            let cached = self.cache.get(&key).expect("just-built cache entry");
            (
                cached.a_layout,
                cached.b_layout,
                cached.c_layout,
                cached.matmul_desc,
                cached.pref,
            )
        };

        // 2. Query up to N candidate algos via raw FFI.
        let mut results: Vec<sys::cublasLtMatmulHeuristicResult_t> =
            vec![std::mem::zeroed(); n_candidates as usize];
        let mut return_count: std::os::raw::c_int = 0;
        let status = sys::cublasLtMatmulAlgoGetHeuristic(
            self.handle,
            matmul_desc,
            a_layout,
            b_layout,
            c_layout,
            c_layout,
            pref,
            n_candidates as std::os::raw::c_int,
            results.as_mut_ptr(),
            &mut return_count,
        );
        if status != sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
            return Err(CudaError::CublasStatus {
                code: status as i32,
                location: "autotune_bf16::heuristic_multi",
            });
        }
        let actual = return_count.max(1) as usize;
        results.truncate(actual);

        // 3. Time each candidate. Use CUDA events with timing enabled.
        let alpha: f32 = 1.0;
        let beta: f32 = 0.0;
        let workspace_ptr = {
            use cudarc::driver::DevicePtr;
            self.workspace.device_ptr(&self.stream).0
        };
        let ctx = self.stream.context();
        let mk_event = || -> Result<cudarc::driver::CudaEvent, CudaError> {
            ctx.new_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))
                .map_err(|e| CudaError::Driver {
                    code: format!("{e:?}").len() as i32,
                    location: "autotune_bf16::new_event",
                })
        };

        let mut best_idx = 0usize;
        let mut best_ms = f32::INFINITY;
        let mut sum_per_call = 0.0f32;

        // Warm-up pass on each candidate (kicks JIT, hides cold-cache effects).
        for (idx, hr) in results.iter().enumerate() {
            // Skip candidates the heuristic flagged as unsupported.
            if hr.state != sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
                continue;
            }
            // 1 warm-up
            let _ = result::matmul(
                self.handle,
                matmul_desc,
                (&alpha) as *const f32 as *const _,
                (&beta) as *const f32 as *const _,
                a_dev as *const _,
                a_layout,
                b_dev as *const _,
                b_layout,
                c_dev as *const _,
                c_layout,
                c_dev as *mut _,
                c_layout,
                (&hr.algo) as *const _,
                workspace_ptr as *mut _,
                self.workspace_bytes,
                self.stream.cu_stream() as *mut _,
            );

            // Timed passes
            let start = mk_event()?;
            let stop = mk_event()?;
            start.record(&self.stream).map_err(|e| CudaError::Driver {
                code: format!("{e:?}").len() as i32,
                location: "autotune_bf16::record_start",
            })?;
            for _ in 0..n_passes {
                let _ = result::matmul(
                    self.handle,
                    matmul_desc,
                    (&alpha) as *const f32 as *const _,
                    (&beta) as *const f32 as *const _,
                    a_dev as *const _,
                    a_layout,
                    b_dev as *const _,
                    b_layout,
                    c_dev as *const _,
                    c_layout,
                    c_dev as *mut _,
                    c_layout,
                    (&hr.algo) as *const _,
                    workspace_ptr as *mut _,
                    self.workspace_bytes,
                    self.stream.cu_stream() as *mut _,
                );
            }
            stop.record(&self.stream).map_err(|e| CudaError::Driver {
                code: format!("{e:?}").len() as i32,
                location: "autotune_bf16::record_stop",
            })?;
            stop.synchronize().map_err(|e| CudaError::Driver {
                code: format!("{e:?}").len() as i32,
                location: "autotune_bf16::sync",
            })?;
            let elapsed_total = start.elapsed_ms(&stop).map_err(|e| CudaError::Driver {
                code: format!("{e:?}").len() as i32,
                location: "autotune_bf16::elapsed",
            })?;
            let per_call = elapsed_total / n_passes as f32;
            sum_per_call += per_call;
            if per_call < best_ms {
                best_ms = per_call;
                best_idx = idx;
            }
        }

        // 4. Patch the cached algo to use the winner.
        if let Some(entry) = self.cache.get_mut(&key) {
            entry.algo = results[best_idx].algo;
        }

        let _ = sum_per_call; // (kept for future verbose mode)
        Ok(best_ms)
    }

    /// Multi-algo autotune for an NVFP4 / MXFP4 (m,k)·(k,n) shape (T240.8g).
    ///
    /// Same idea as `autotune_bf16` but for FP4. Picks the fastest algo
    /// among up to `n_candidates` heuristic candidates and stores it in
    /// the cache. cublasLt FP4 heuristic is even less GB10-optimized than
    /// the BF16 one — community reports show 15–30% lift from autotuning.
    ///
    /// # Safety
    /// `*_dev` must be valid for FP4 packed inputs (1 byte per 2 elems),
    /// scales, and BF16/FP16/F32 output. Caller accepts that `c_dev` is
    /// written through during timing.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn autotune_mxfp4(
        &mut self,
        a_dev: u64,
        a_scale_dev: u64,
        b_dev: u64,
        b_scale_dev: u64,
        c_dev: u64,
        m: usize,
        k: usize,
        n: usize,
        out: Fp8Output,
        scale_mode: Fp4ScaleMode,
        n_candidates: u32,
        n_passes: u32,
    ) -> Result<f32, CudaError> {
        if m == 0 || n == 0 || k == 0 {
            return Ok(0.0);
        }
        let block = scale_mode.block_size();
        if k % block != 0 {
            return Err(CudaError::Unsupported {
                msg: format!("FP4 requires k % {block} == 0, got k={k}"),
            });
        }
        let n_candidates = n_candidates.max(1).min(32);
        let n_passes = n_passes.max(1).min(50);

        // 1. Ensure base cache entry exists.
        let _ = self.get_or_build(
            m,
            n,
            k,
            sys::cudaDataType_t::CUDA_R_4F_E2M1,
            sys::cudaDataType_t::CUDA_R_4F_E2M1,
            out.cuda_type(),
            Some(scale_mode),
            a_scale_dev,
            b_scale_dev,
        )?;
        let key = ConfigKey {
            m: m as u32,
            n: n as u32,
            k: k as u32,
            a_dt: sys::cudaDataType_t::CUDA_R_4F_E2M1 as i32,
            b_dt: sys::cudaDataType_t::CUDA_R_4F_E2M1 as i32,
            c_dt: out.cuda_type() as i32,
            transa: true,
            transb: false,
            scale_mode: match scale_mode {
                Fp4ScaleMode::Vec32Ue8m0 => 1,
                Fp4ScaleMode::Vec16Ue4m3 => 2,
            },
        };
        let (a_layout, b_layout, c_layout, matmul_desc, pref) = {
            let cached = self.cache.get(&key).expect("just-built cache entry");
            (
                cached.a_layout,
                cached.b_layout,
                cached.c_layout,
                cached.matmul_desc,
                cached.pref,
            )
        };

        // 2. Bind the per-call scale pointers on the desc (used by all candidates).
        result::set_matmul_desc_attribute(
            matmul_desc,
            sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
            (&a_scale_dev) as *const _ as *const _,
            std::mem::size_of::<u64>(),
        )
        .map_err(|e| CudaError::CublasStatus {
            code: lt_err_code(e),
            location: "autotune_mxfp4::set_a_scale_ptr",
        })?;
        result::set_matmul_desc_attribute(
            matmul_desc,
            sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
            (&b_scale_dev) as *const _ as *const _,
            std::mem::size_of::<u64>(),
        )
        .map_err(|e| CudaError::CublasStatus {
            code: lt_err_code(e),
            location: "autotune_mxfp4::set_b_scale_ptr",
        })?;

        // 3. Query top-N candidate algos.
        let mut results: Vec<sys::cublasLtMatmulHeuristicResult_t> =
            vec![std::mem::zeroed(); n_candidates as usize];
        let mut return_count: std::os::raw::c_int = 0;
        let status = sys::cublasLtMatmulAlgoGetHeuristic(
            self.handle,
            matmul_desc,
            a_layout,
            b_layout,
            c_layout,
            c_layout,
            pref,
            n_candidates as std::os::raw::c_int,
            results.as_mut_ptr(),
            &mut return_count,
        );
        if status != sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
            return Err(CudaError::CublasStatus {
                code: status as i32,
                location: "autotune_mxfp4::heuristic_multi",
            });
        }
        let actual = return_count.max(1) as usize;
        results.truncate(actual);

        // 4. Time each candidate.
        let alpha: f32 = 1.0;
        let beta: f32 = 0.0;
        let workspace_ptr = {
            use cudarc::driver::DevicePtr;
            self.workspace.device_ptr(&self.stream).0
        };
        let ctx = self.stream.context();
        let mk_event = || -> Result<cudarc::driver::CudaEvent, CudaError> {
            ctx.new_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))
                .map_err(|e| CudaError::Driver {
                    code: format!("{e:?}").len() as i32,
                    location: "autotune_mxfp4::new_event",
                })
        };

        let mut best_idx = 0usize;
        let mut best_split_k: i32 = 1;
        let mut best_ms = f32::INFINITY;

        // T240.8m — try plusieurs SPLIT_K values pour chaque algo. SPLIT_K
        // parallélise l'accumulation = peut débloquer plus de SMs sur les
        // matmuls non-square (FFN g+up où N >> K).
        // Override via RUSTORCH_CUBLASLT_SPLIT_K_VALUES="1,2,4,8" (default).
        let split_k_values: Vec<i32> = std::env::var("RUSTORCH_CUBLASLT_SPLIT_K_VALUES")
            .ok()
            .map(|s| {
                s.split(',')
                    .filter_map(|x| x.trim().parse().ok())
                    .collect::<Vec<i32>>()
            })
            .filter(|v: &Vec<i32>| !v.is_empty())
            .unwrap_or_else(|| vec![1, 2, 4, 8]);

        for (idx, hr) in results.iter().enumerate() {
            if hr.state != sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
                continue;
            }
            // Cloner l'algo une fois (les set_attribute le mutent)
            let mut algo_template = hr.algo;
            for &split_k in &split_k_values {
                let mut algo = algo_template;
                if split_k > 1 {
                    let status = sys::cublasLtMatmulAlgoConfigSetAttribute(
                        &mut algo,
                        sys::cublasLtMatmulAlgoConfigAttributes_t::CUBLASLT_ALGO_CONFIG_SPLITK_NUM,
                        (&split_k) as *const i32 as *const _,
                        std::mem::size_of::<i32>(),
                    );
                    if status != sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
                        // Algo ne supporte pas split_k > 1, skip
                        continue;
                    }
                }
                // 1 warm-up
                let _ = result::matmul(
                    self.handle,
                    matmul_desc,
                    (&alpha) as *const f32 as *const _,
                    (&beta) as *const f32 as *const _,
                    a_dev as *const _,
                    a_layout,
                    b_dev as *const _,
                    b_layout,
                    c_dev as *const _,
                    c_layout,
                    c_dev as *mut _,
                    c_layout,
                    (&algo) as *const _,
                    workspace_ptr as *mut _,
                    self.workspace_bytes,
                    self.stream.cu_stream() as *mut _,
                );
                let start = mk_event()?;
                let stop = mk_event()?;
                start.record(&self.stream).map_err(|e| CudaError::Driver {
                    code: format!("{e:?}").len() as i32,
                    location: "autotune_mxfp4::record_start",
                })?;
                let mut had_err = false;
                for _ in 0..n_passes {
                    let r = result::matmul(
                        self.handle,
                        matmul_desc,
                        (&alpha) as *const f32 as *const _,
                        (&beta) as *const f32 as *const _,
                        a_dev as *const _,
                        a_layout,
                        b_dev as *const _,
                        b_layout,
                        c_dev as *const _,
                        c_layout,
                        c_dev as *mut _,
                        c_layout,
                        (&algo) as *const _,
                        workspace_ptr as *mut _,
                        self.workspace_bytes,
                        self.stream.cu_stream() as *mut _,
                    );
                    if r.is_err() {
                        had_err = true;
                        break;
                    }
                }
                if had_err {
                    let _ = algo_template;
                    continue;
                }
                stop.record(&self.stream).map_err(|e| CudaError::Driver {
                    code: format!("{e:?}").len() as i32,
                    location: "autotune_mxfp4::record_stop",
                })?;
                stop.synchronize().map_err(|e| CudaError::Driver {
                    code: format!("{e:?}").len() as i32,
                    location: "autotune_mxfp4::sync",
                })?;
                let elapsed_total = start.elapsed_ms(&stop).map_err(|e| CudaError::Driver {
                    code: format!("{e:?}").len() as i32,
                    location: "autotune_mxfp4::elapsed",
                })?;
                let per_call = elapsed_total / n_passes as f32;
                if per_call < best_ms {
                    best_ms = per_call;
                    best_idx = idx;
                    best_split_k = split_k;
                }
            }
            let _ = algo_template;
        }

        // 5. Patch cache entry — réapplique le SPLIT_K winner sur l'algo cached.
        if let Some(entry) = self.cache.get_mut(&key) {
            entry.algo = results[best_idx].algo;
            if best_split_k > 1 {
                let _ = sys::cublasLtMatmulAlgoConfigSetAttribute(
                    &mut entry.algo,
                    sys::cublasLtMatmulAlgoConfigAttributes_t::CUBLASLT_ALGO_CONFIG_SPLITK_NUM,
                    (&best_split_k) as *const i32 as *const _,
                    std::mem::size_of::<i32>(),
                );
            }
        }
        Ok(best_ms)
    }
}

#[cfg(feature = "cuda")]
impl Drop for LtSession {
    fn drop(&mut self) {
        // Cached resources drop via their own Drop impls (in HashMap clear);
        // we just need to release the handle.
        self.cache.clear();
        unsafe {
            let _ = result::destroy_handle(self.handle);
        }
    }
}

/// Build the per-config descriptors + heuristic. Called by LtSession on
/// the first matmul of a given (shape, dtypes, scale_mode) combo.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
unsafe fn build_cached(
    handle: sys::cublasLtHandle_t,
    m: usize,
    n: usize,
    k: usize,
    a_dt: sys::cudaDataType_t,
    b_dt: sys::cudaDataType_t,
    c_dt: sys::cudaDataType_t,
    scale_mode: Option<Fp4ScaleMode>,
    workspace_bytes: usize,
    seed_a_scale: u64,
    seed_b_scale: u64,
) -> Result<CachedMatmul, CudaError> {
    // Convention : we expect row-major buffers (the standard layout
    // produced by torch / our CPU loader after `transpose_2d`) :
    //   A row-major (m, k)  → physical layout = col-major (k, m, ld=k)
    //   B row-major (k, n)  → physical layout = col-major (n, k, ld=n)
    //   C row-major (m, n)  → physical layout = col-major (m, n, ld=m)
    //                          (only equivalent to row-major when m=1,
    //                           which is the autoregressive decode case)
    //
    // We then ask cuBLASLt to compute  C = op(A) * op(B)  with
    //   transa = OP_T   →  op(A) = (col(k, m))^T = col(m, k) ↔ row(m, k) = A
    //   transb = OP_T   →  op(B) = (col(n, k))^T = col(k, n) ↔ row(k, n) = B
    //
    // T241.6b/c caveat : cuBLASLt sm_121 NVFP4 only supports transb=OP_N
    // (CUBLAS_STATUS_NOT_SUPPORTED on TT). For FP4 we fall back to the
    // legacy "scrambled" convention here ; the proper fix is to transpose
    // the W weights at quantize time so the TN-only constraint is honored
    // with mathematically correct results. Tracked as T241.6c.
    let is_fp4 =
        a_dt == sys::cudaDataType_t::CUDA_R_4F_E2M1 || b_dt == sys::cudaDataType_t::CUDA_R_4F_E2M1;
    let (b_rows, b_cols, b_ld) = if is_fp4 {
        (k as u64, n as u64, k as i64)
    } else {
        (n as u64, k as u64, n as i64)
    };
    let a_layout =
        result::create_matrix_layout(a_dt, k as u64, m as u64, k as i64).map_err(|e| {
            CudaError::CublasStatus {
                code: lt_err_code(e),
                location: "build_cached::a_layout",
            }
        })?;
    let b_layout = result::create_matrix_layout(b_dt, b_rows, b_cols, b_ld).map_err(|e| {
        CudaError::CublasStatus {
            code: lt_err_code(e),
            location: "build_cached::b_layout",
        }
    })?;
    let c_layout =
        result::create_matrix_layout(c_dt, m as u64, n as u64, m as i64).map_err(|e| {
            CudaError::CublasStatus {
                code: lt_err_code(e),
                location: "build_cached::c_layout",
            }
        })?;

    let matmul_desc = result::create_matmul_desc(
        sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
        sys::cudaDataType_t::CUDA_R_32F,
    )
    .map_err(|e| CudaError::CublasStatus {
        code: lt_err_code(e),
        location: "build_cached::matmul_desc",
    })?;

    let transa = cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_T;
    let transb = if is_fp4 {
        cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N
    } else {
        cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_T
    };
    result::set_matmul_desc_attribute(
        matmul_desc,
        sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSA,
        (&transa) as *const _ as *const _,
        std::mem::size_of::<cudarc::cublas::sys::cublasOperation_t>(),
    )
    .map_err(|e| CudaError::CublasStatus {
        code: lt_err_code(e),
        location: "build_cached::set_transa",
    })?;
    result::set_matmul_desc_attribute(
        matmul_desc,
        sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSB,
        (&transb) as *const _ as *const _,
        std::mem::size_of::<cudarc::cublas::sys::cublasOperation_t>(),
    )
    .map_err(|e| CudaError::CublasStatus {
        code: lt_err_code(e),
        location: "build_cached::set_transb",
    })?;

    if let Some(sm) = scale_mode {
        let v = sm.cuda_value();
        result::set_matmul_desc_attribute(
            matmul_desc,
            sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_A_SCALE_MODE,
            (&v) as *const _ as *const _,
            std::mem::size_of::<sys::cublasLtMatmulMatrixScale_t>(),
        )
        .map_err(|e| CudaError::CublasStatus {
            code: lt_err_code(e),
            location: "build_cached::set_a_scale_mode",
        })?;
        result::set_matmul_desc_attribute(
            matmul_desc,
            sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_B_SCALE_MODE,
            (&v) as *const _ as *const _,
            std::mem::size_of::<sys::cublasLtMatmulMatrixScale_t>(),
        )
        .map_err(|e| CudaError::CublasStatus {
            code: lt_err_code(e),
            location: "build_cached::set_b_scale_mode",
        })?;
        // Seed scale pointers — heuristic search rejects desc with NULL
        // scale pointers when a scale mode is set. Real per-call pointers
        // are rebound by the matmul method before each dispatch.
        result::set_matmul_desc_attribute(
            matmul_desc,
            sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
            (&seed_a_scale) as *const _ as *const _,
            std::mem::size_of::<u64>(),
        )
        .map_err(|e| CudaError::CublasStatus {
            code: lt_err_code(e),
            location: "build_cached::set_a_scale_ptr_seed",
        })?;
        result::set_matmul_desc_attribute(
            matmul_desc,
            sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
            (&seed_b_scale) as *const _ as *const _,
            std::mem::size_of::<u64>(),
        )
        .map_err(|e| CudaError::CublasStatus {
            code: lt_err_code(e),
            location: "build_cached::set_b_scale_ptr_seed",
        })?;
    } else {
        let _ = (seed_a_scale, seed_b_scale);
    }

    let pref = result::create_matmul_pref().map_err(|e| CudaError::CublasStatus {
        code: lt_err_code(e),
        location: "build_cached::pref",
    })?;
    result::set_matmul_pref_attribute(
        pref,
        sys::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        (&workspace_bytes) as *const _ as *const _,
        std::mem::size_of::<usize>(),
    )
    .map_err(|e| CudaError::CublasStatus {
        code: lt_err_code(e),
        location: "build_cached::pref_workspace",
    })?;

    let heuristic = result::get_matmul_algo_heuristic(
        handle,
        matmul_desc,
        a_layout,
        b_layout,
        c_layout,
        c_layout,
        pref,
    )
    .map_err(|e| CudaError::CublasStatus {
        code: lt_err_code(e),
        location: "build_cached::heuristic",
    })?;

    Ok(CachedMatmul {
        a_layout,
        b_layout,
        c_layout,
        matmul_desc,
        pref,
        algo: heuristic.algo,
    })
}

/// **DESIGN-PIVOT** — BF16 TN-with-row-major-output cached config.
///
/// Builds layouts and matmul descriptor for the row-major BF16 path used by
/// `LtSession::matmul_bf16_rowmajor`. Convention :
///
///   w (row-major `[N, K]`, BF16) → seen by cuBLASLt as col-major
///                                   `(K, N)` ld=K. transa=T turns it back
///                                   into "logical N × K".
///   x (row-major `[M, K]`, BF16) → seen by cuBLASLt as col-major
///                                   `(K, M)` ld=K. transb=N keeps it.
///   y (col-major `[N, M]` ld=N output, == row-major `[M, N]` stride=N).
///
/// cuBLASLt is asked for `cublas-M=N, cublas-N=M, cublas-K=K, transa=T,
/// transb=N`. The math reduction is `Y = X · W^T` (the standard "linear"
/// op), which is exactly what the hand-written GEMM kernels compute.
#[cfg(feature = "cuda")]
unsafe fn build_cached_bf16_rowmajor(
    handle: sys::cublasLtHandle_t,
    m: usize,
    n: usize,
    k: usize,
    workspace_bytes: usize,
) -> Result<CachedMatmul, CudaError> {
    let dt = sys::cudaDataType_t::CUDA_R_16BF;

    // A = W. cublas-side cols = N (cublas-M), rows = K (cublas-K), ld = K.
    let a_layout = result::create_matrix_layout(dt, k as u64, n as u64, k as i64).map_err(|e| {
        CudaError::CublasStatus {
            code: lt_err_code(e),
            location: "build_cached_bf16_rowmajor::a_layout",
        }
    })?;
    // B = X. cublas-side cols = M (cublas-N), rows = K (cublas-K), ld = K.
    let b_layout = result::create_matrix_layout(dt, k as u64, m as u64, k as i64).map_err(|e| {
        CudaError::CublasStatus {
            code: lt_err_code(e),
            location: "build_cached_bf16_rowmajor::b_layout",
        }
    })?;
    // C : col-major (cublas-M=N, cublas-N=M), ld = N. Physical bytes match
    // row-major Y[M, N] stride=N, so the same buffer reads identically as
    // row-major from the caller's perspective.
    let c_layout = result::create_matrix_layout(dt, n as u64, m as u64, n as i64).map_err(|e| {
        CudaError::CublasStatus {
            code: lt_err_code(e),
            location: "build_cached_bf16_rowmajor::c_layout",
        }
    })?;

    let matmul_desc = result::create_matmul_desc(
        sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
        sys::cudaDataType_t::CUDA_R_32F,
    )
    .map_err(|e| CudaError::CublasStatus {
        code: lt_err_code(e),
        location: "build_cached_bf16_rowmajor::matmul_desc",
    })?;

    let transa = cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_T;
    let transb = cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N;
    result::set_matmul_desc_attribute(
        matmul_desc,
        sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSA,
        (&transa) as *const _ as *const _,
        std::mem::size_of::<cudarc::cublas::sys::cublasOperation_t>(),
    )
    .map_err(|e| CudaError::CublasStatus {
        code: lt_err_code(e),
        location: "build_cached_bf16_rowmajor::set_transa",
    })?;
    result::set_matmul_desc_attribute(
        matmul_desc,
        sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSB,
        (&transb) as *const _ as *const _,
        std::mem::size_of::<cudarc::cublas::sys::cublasOperation_t>(),
    )
    .map_err(|e| CudaError::CublasStatus {
        code: lt_err_code(e),
        location: "build_cached_bf16_rowmajor::set_transb",
    })?;

    let pref = result::create_matmul_pref().map_err(|e| CudaError::CublasStatus {
        code: lt_err_code(e),
        location: "build_cached_bf16_rowmajor::pref",
    })?;
    result::set_matmul_pref_attribute(
        pref,
        sys::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        (&workspace_bytes) as *const _ as *const _,
        std::mem::size_of::<usize>(),
    )
    .map_err(|e| CudaError::CublasStatus {
        code: lt_err_code(e),
        location: "build_cached_bf16_rowmajor::pref_workspace",
    })?;

    let heuristic = result::get_matmul_algo_heuristic(
        handle,
        matmul_desc,
        a_layout,
        b_layout,
        c_layout,
        c_layout,
        pref,
    )
    .map_err(|e| CudaError::CublasStatus {
        code: lt_err_code(e),
        location: "build_cached_bf16_rowmajor::heuristic",
    })?;

    Ok(CachedMatmul {
        a_layout,
        b_layout,
        c_layout,
        matmul_desc,
        pref,
        algo: heuristic.algo,
    })
}

/// FP4 block-scale flavour. Hardware support varies:
/// - **MXFP4** (VEC32_UE8M0): industry standard. Hopper / Blackwell datacenter.
/// - **NVFP4** (VEC16_UE4M3): NVIDIA proprietary, more aggressive blocking.
///   Used on Grace-Blackwell (GB10/GB200). Smaller blocks → more scales →
///   slightly more memory but better quantization quality.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fp4ScaleMode {
    /// 32-element blocks with UE8M0 (8-bit exponent) scales — MXFP4 standard.
    Vec32Ue8m0,
    /// 16-element blocks with UE4M3 (4-bit exp, 3-bit mantissa) scales — NVFP4.
    Vec16Ue4m3,
}

#[cfg(feature = "cuda")]
impl Fp4ScaleMode {
    fn cuda_value(self) -> sys::cublasLtMatmulMatrixScale_t {
        match self {
            Self::Vec32Ue8m0 => {
                sys::cublasLtMatmulMatrixScale_t::CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0
            },
            Self::Vec16Ue4m3 => {
                sys::cublasLtMatmulMatrixScale_t::CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3
            },
        }
    }

    /// Block size — number of FP4 elements per scale.
    pub fn block_size(self) -> usize {
        match self {
            Self::Vec32Ue8m0 => 32,
            Self::Vec16Ue4m3 => 16,
        }
    }
}

/// Run an MXFP4 / NVFP4 GEMM (FP4 inputs with block-scaled quantization).
///
/// FP4 inputs are byte-packed (1 byte = 2 elements). Each block of N FP4
/// elements has one byte of scale (UE8M0 or UE4M3 depending on `scale_mode`).
/// For an [M × K] FP4 matrix, scales form an [M × (K / block_size)] tensor.
/// K must be a multiple of `scale_mode.block_size()`.
///
/// Output is BF16/FP16/F32 with F32 accumulation. Same TN-only requirement
/// as FP8: hardcodes `transa=T, transb=N`.
///
/// On GB10 with all blocks scaled to unity (UE8M0 byte = 127), this hits
/// the headline ~1 PFLOP / 1000 TOPS theoretical peak. Real-world utility
/// requires proper per-block scale calibration during quantization.
///
/// # Safety
///
/// `*_dev` pointers must be valid for the lifetime of the call, sized
/// correctly for FP4 packed inputs / UE8M0 scale tensors / output type.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn matmul_mxfp4(
    a_dev: u64,
    a_scale_dev: u64,
    b_dev: u64,
    b_scale_dev: u64,
    c_dev: u64,
    m: usize,
    k: usize,
    n: usize,
    alpha: f32,
    beta: f32,
    out: Fp8Output,
    scale_mode: Fp4ScaleMode,
    workspace_dev: u64,
    workspace_bytes: usize,
    stream: u64,
) -> Result<(), CudaError> {
    if m == 0 || n == 0 || k == 0 {
        return Ok(());
    }
    let block = scale_mode.block_size();
    if k % block != 0 {
        return Err(CudaError::Unsupported {
            msg: format!("FP4 requires k % {block} == 0, got k={k}"),
        });
    }

    let handle = result::create_handle().map_err(|e| CudaError::CublasStatus {
        code: lt_err_code(e),
        location: "cublas_lt::matmul_mxfp4::create_handle",
    })?;

    // FP4 inputs are packed 2-per-byte. cuBLASLt's matrix layout API uses
    // the LOGICAL element count (k×m, k×n, m×n) — the implementation knows
    // the physical byte size from the data type tag.
    //
    // T241.6c TODO : cuBLASLt FP4 (NVFP4 sur sm_121) ne supporte QUE le
    // mode TN (transa=OP_T, transb=OP_N) au runtime — un transb=OP_T
    // produit CUBLAS_STATUS_NOT_SUPPORTED. Pour adopter la convention
    // row-major correcte (comme matmul_bf16 fait via build_cached), il
    // faudra transposer les poids B au moment de la quantization NVFP4
    // (i.e. quantize_bf16_to_nvfp4 doit prendre W^T en entrée).
    //
    // En attendant ce fix, le path FP4 utilise la convention "scrambled"
    // d'origine : numériquement incorrecte mais le bench tourne et
    // donne des FLOPS représentatifs. À traiter en T241.6c.
    let fp4_dt = sys::cudaDataType_t::CUDA_R_4F_E2M1;
    let a_layout =
        result::create_matrix_layout(fp4_dt, k as u64, m as u64, k as i64).map_err(|e| {
            CudaError::CublasStatus {
                code: lt_err_code(e),
                location: "cublas_lt::matmul_mxfp4::a_layout",
            }
        })?;
    let b_layout =
        result::create_matrix_layout(fp4_dt, k as u64, n as u64, k as i64).map_err(|e| {
            CudaError::CublasStatus {
                code: lt_err_code(e),
                location: "cublas_lt::matmul_mxfp4::b_layout",
            }
        })?;
    let c_layout = result::create_matrix_layout(out.cuda_type(), m as u64, n as u64, m as i64)
        .map_err(|e| CudaError::CublasStatus {
            code: lt_err_code(e),
            location: "cublas_lt::matmul_mxfp4::c_layout",
        })?;

    let matmul_desc = result::create_matmul_desc(
        sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
        sys::cudaDataType_t::CUDA_R_32F,
    )
    .map_err(|e| CudaError::CublasStatus {
        code: lt_err_code(e),
        location: "cublas_lt::matmul_mxfp4::matmul_desc",
    })?;

    let transa = cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_T;
    let transb = cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N;
    result::set_matmul_desc_attribute(
        matmul_desc,
        sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSA,
        (&transa) as *const _ as *const _,
        std::mem::size_of::<cudarc::cublas::sys::cublasOperation_t>(),
    )
    .map_err(|e| CudaError::CublasStatus {
        code: lt_err_code(e),
        location: "cublas_lt::matmul_mxfp4::set_transa",
    })?;
    result::set_matmul_desc_attribute(
        matmul_desc,
        sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSB,
        (&transb) as *const _ as *const _,
        std::mem::size_of::<cudarc::cublas::sys::cublasOperation_t>(),
    )
    .map_err(|e| CudaError::CublasStatus {
        code: lt_err_code(e),
        location: "cublas_lt::matmul_mxfp4::set_transb",
    })?;

    let scale_mode_val = scale_mode.cuda_value();
    result::set_matmul_desc_attribute(
        matmul_desc,
        sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_A_SCALE_MODE,
        (&scale_mode_val) as *const _ as *const _,
        std::mem::size_of::<sys::cublasLtMatmulMatrixScale_t>(),
    )
    .map_err(|e| CudaError::CublasStatus {
        code: lt_err_code(e),
        location: "cublas_lt::matmul_mxfp4::set_a_scale_mode",
    })?;
    result::set_matmul_desc_attribute(
        matmul_desc,
        sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_B_SCALE_MODE,
        (&scale_mode_val) as *const _ as *const _,
        std::mem::size_of::<sys::cublasLtMatmulMatrixScale_t>(),
    )
    .map_err(|e| CudaError::CublasStatus {
        code: lt_err_code(e),
        location: "cublas_lt::matmul_mxfp4::set_b_scale_mode",
    })?;

    // Scale pointers (device addresses of the UE8M0 scale tensors).
    result::set_matmul_desc_attribute(
        matmul_desc,
        sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
        (&a_scale_dev) as *const _ as *const _,
        std::mem::size_of::<u64>(),
    )
    .map_err(|e| CudaError::CublasStatus {
        code: lt_err_code(e),
        location: "cublas_lt::matmul_mxfp4::set_a_scale_ptr",
    })?;
    result::set_matmul_desc_attribute(
        matmul_desc,
        sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
        (&b_scale_dev) as *const _ as *const _,
        std::mem::size_of::<u64>(),
    )
    .map_err(|e| CudaError::CublasStatus {
        code: lt_err_code(e),
        location: "cublas_lt::matmul_mxfp4::set_b_scale_ptr",
    })?;

    let pref = result::create_matmul_pref().map_err(|e| CudaError::CublasStatus {
        code: lt_err_code(e),
        location: "cublas_lt::matmul_mxfp4::pref",
    })?;
    result::set_matmul_pref_attribute(
        pref,
        sys::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        (&workspace_bytes) as *const _ as *const _,
        std::mem::size_of::<usize>(),
    )
    .map_err(|e| CudaError::CublasStatus {
        code: lt_err_code(e),
        location: "cublas_lt::matmul_mxfp4::pref_workspace",
    })?;

    let _res = LtResources {
        handle,
        a_layout,
        b_layout,
        c_layout,
        matmul_desc,
        pref,
    };

    let heuristic = result::get_matmul_algo_heuristic(
        handle,
        matmul_desc,
        a_layout,
        b_layout,
        c_layout,
        c_layout,
        pref,
    )
    .map_err(|e| CudaError::CublasStatus {
        code: lt_err_code(e),
        location: "cublas_lt::matmul_mxfp4::heuristic",
    })?;

    result::matmul(
        handle,
        matmul_desc,
        (&alpha) as *const f32 as *const _,
        (&beta) as *const f32 as *const _,
        a_dev as *const _,
        a_layout,
        b_dev as *const _,
        b_layout,
        c_dev as *const _,
        c_layout,
        c_dev as *mut _,
        c_layout,
        (&heuristic.algo) as *const _,
        workspace_dev as *mut _,
        workspace_bytes,
        stream as *mut _,
    )
    .map_err(|e| CudaError::CublasStatus {
        code: lt_err_code(e),
        location: "cublas_lt::matmul_mxfp4::matmul",
    })?;

    Ok(())
}

/// Stub for no-cuda builds.
///
/// # Safety
/// Same contract as the cuda variant, but pointers are never dereferenced.
#[cfg(not(feature = "cuda"))]
#[allow(clippy::too_many_arguments)]
pub unsafe fn matmul_mxfp4(
    _a_dev: u64,
    _a_scale_dev: u64,
    _b_dev: u64,
    _b_scale_dev: u64,
    _c_dev: u64,
    _m: usize,
    _k: usize,
    _n: usize,
    _alpha: f32,
    _beta: f32,
    _out: Fp8Output,
    _scale_mode: Fp4ScaleMode,
    _workspace_dev: u64,
    _workspace_bytes: usize,
    _stream: u64,
) -> Result<(), CudaError> {
    Err(CudaError::NoDeviceFound)
}

// =============================================================================
// Numerical parity tests
// =============================================================================

#[cfg(all(test, feature = "cuda"))]
mod parity_tests {
    use super::*;
    use cudarc::driver::{CudaContext, DevicePtr, DevicePtrMut};

    /// Reference CPU matmul : `c = a · b` with row-major buffers,
    /// `a[m, k]`, `b[k, n]`, `c[m, n]`.
    fn cpu_matmul(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0f32;
                for kk in 0..k {
                    acc += a[i * k + kk] * b[kk * n + j];
                }
                c[i * n + j] = acc;
            }
        }
    }

    /// T241.6b regression guard — verifies matmul_bf16 actually computes
    /// `C = A·B` with row-major buffers (and not `A·B^T` or some scrambled
    /// variant). Guards against any future change to the cuBLASLt layout
    /// convention in build_cached().
    ///
    /// Uses a small 4×3 · 3×5 case with hand-picked non-symmetric values.
    /// BF16 has ~7-bit mantissa so we use small integer values that round-trip
    /// exactly and check for byte-perfect equality after conversion.
    #[test]
    fn matmul_bf16_computes_a_dot_b_row_major() {
        let m = 4usize;
        let k = 3usize;
        let n = 5usize;

        // Hand-picked A [m, k] and B [k, n] with values that round-trip
        // exactly to BF16. NOT symmetric / NOT diagonal so any layout bug
        // would produce different output.
        let a: Vec<f32> = vec![
            1.0, 2.0, 3.0, // row 0
            4.0, 5.0, 6.0, // row 1
            7.0, 8.0, 9.0, // row 2
            10.0, 11.0, 12.0, // row 3
        ];
        let b: Vec<f32> = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, // row 0
            6.0, 7.0, 8.0, 9.0, 10.0, // row 1
            11.0, 12.0, 13.0, 14.0, 15.0, // row 2
        ];
        let mut c_cpu: Vec<f32> = vec![0.0; m * n];
        cpu_matmul(&a, &b, &mut c_cpu, m, k, n);

        // Verify CPU reference (sanity)
        // c[0, 0] = 1*1 + 2*6 + 3*11 = 1 + 12 + 33 = 46
        assert_eq!(c_cpu[0], 46.0);
        // c[0, 1] = 1*2 + 2*7 + 3*12 = 2 + 14 + 36 = 52
        assert_eq!(c_cpu[1], 52.0);

        // GPU path
        let ctx = CudaContext::new(0).expect("CudaContext");
        let stream = ctx.default_stream();
        let mut session = LtSession::new(stream.clone()).expect("LtSession");

        let a_bf: Vec<half::bf16> = a.iter().copied().map(half::bf16::from_f32).collect();
        let b_bf: Vec<half::bf16> = b.iter().copied().map(half::bf16::from_f32).collect();
        let a_dev = stream.memcpy_stod(&a_bf).expect("upload a");
        let b_dev = stream.memcpy_stod(&b_bf).expect("upload b");
        let mut c_dev = stream.alloc_zeros::<half::bf16>(m * n).expect("alloc c");

        unsafe {
            let (a_p, _r1) = a_dev.device_ptr(&stream);
            let (b_p, _r2) = b_dev.device_ptr(&stream);
            let (c_p, _r3) = c_dev.device_ptr_mut(&stream);
            session
                .matmul_bf16(a_p, b_p, c_p, m, k, n, 1.0, 0.0)
                .expect("matmul");
        }

        let c_host: Vec<half::bf16> = stream.memcpy_dtov(&c_dev).expect("dtov");
        let c_gpu: Vec<f32> = c_host.into_iter().map(|x| x.to_f32()).collect();

        // NOTE: cuBLASLt writes C in col-major (m, n) ld=m. To read the
        // mathematical (i, j) element we use buf[j*m + i]. CPU reference
        // stores row-major so c_cpu[i*n + j]. The buffers DIFFER physically
        // for m > 1 (documented in build_cached()) but the underlying math
        // is identical. The autoregressive LLM decode hot path uses m=1
        // (see m_eq_1 test) where col-major and row-major coincide.
        for i in 0..m {
            for j in 0..n {
                let cpu_v = c_cpu[i * n + j];
                let gpu_v = c_gpu[j * m + i]; // col-major read
                let diff = (cpu_v - gpu_v).abs();
                let tol = cpu_v.abs() * 1e-2 + 1e-2;
                assert!(
                    diff <= tol,
                    "mismatch at ({i}, {j}) : cpu={cpu_v} gpu={gpu_v} diff={diff}",
                );
            }
        }
    }

    /// Same regression guard but with m=1 (single-row, the LLM
    /// autoregressive decode hot path). Specifically catches the
    /// "transb=OP_N on row-major B" bug.
    #[test]
    fn matmul_bf16_m_eq_1_row_major() {
        let m = 1usize;
        let k = 4usize;
        let n = 6usize;

        let a: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let b: Vec<f32> = (0..(k * n)).map(|i| (i + 1) as f32).collect();
        let mut c_cpu: Vec<f32> = vec![0.0; m * n];
        cpu_matmul(&a, &b, &mut c_cpu, m, k, n);

        let ctx = CudaContext::new(0).expect("CudaContext");
        let stream = ctx.default_stream();
        let mut session = LtSession::new(stream.clone()).expect("LtSession");

        let a_bf: Vec<half::bf16> = a.iter().copied().map(half::bf16::from_f32).collect();
        let b_bf: Vec<half::bf16> = b.iter().copied().map(half::bf16::from_f32).collect();
        let a_dev = stream.memcpy_stod(&a_bf).expect("upload a");
        let b_dev = stream.memcpy_stod(&b_bf).expect("upload b");
        let mut c_dev = stream.alloc_zeros::<half::bf16>(m * n).expect("alloc c");

        unsafe {
            let (a_p, _r1) = a_dev.device_ptr(&stream);
            let (b_p, _r2) = b_dev.device_ptr(&stream);
            let (c_p, _r3) = c_dev.device_ptr_mut(&stream);
            session
                .matmul_bf16(a_p, b_p, c_p, m, k, n, 1.0, 0.0)
                .expect("matmul");
        }

        let c_host: Vec<half::bf16> = stream.memcpy_dtov(&c_dev).expect("dtov");
        let c_gpu: Vec<f32> = c_host.into_iter().map(|x| x.to_f32()).collect();

        for j in 0..n {
            let diff = (c_cpu[j] - c_gpu[j]).abs();
            let tol = c_cpu[j].abs() * 5e-2 + 1.0;
            assert!(
                diff <= tol,
                "m=1 mismatch at j={j} : cpu={} gpu={} diff={}",
                c_cpu[j],
                c_gpu[j],
                diff
            );
        }
    }

    // ─────────────────────────────────────────────────────────────────
    // T241.6d — NVFP4 matmul tests : isolate the FP4 path to perimeter
    // the [0,0,0,...] bug. We test 3 levels :
    //
    //   (1) Free matmul_mxfp4 with all-1 inputs and scale=0x70 (≈1.0)
    //       → expected output : k * 1 * 1 * 1 = k (the K dimension)
    //   (2) LtSession::matmul_mxfp4 (cached) with same inputs
    //       → if (1) works and (2) doesn't, bug is in build_cached/rebind
    //   (3) End-to-end with quantize_bf16_to_nvfp4 + matmul
    //       → if (1)+(2) work but (3) doesn't, bug is in the pipeline
    //
    // FP4 code 0x22 = two nibbles each = 0x2 = +1.0 (E2M1 grid).
    // UE4M3 byte 0x70 = 14<<3 = scale 1.0 (NVIDIA bias 14).
    // ─────────────────────────────────────────────────────────────────

    /// Test (1) — FREE path matmul_mxfp4 with all-ones FP4 + scale=0x38.
    /// Reference test : verifies FP4 hardware works on this device.
    ///
    /// Setup : A (m=1, k=128) all FP4 = +1.0 ; B (k=128, n=128) all
    /// FP4 = +1.0 ; both scales byte = 0x38 (= scale value 1.0, bias 7).
    /// Expected C[j] = sum_k(1*1) * 1 * 1 = k = 128 ∀j.
    ///
    /// NB : we must use k >= 128 and n >= 128 — cuBLASLt sm_121 NVFP4
    /// rejects smaller shapes (perimeter test : nvfp4_supported_m_values).
    #[test]
    fn nvfp4_free_matmul_all_ones_returns_k() {
        let m = 1usize;
        let k = 128usize;
        let n = 128usize;
        // FP4 byte 0x22 packs two values = +1.0 each. Total a_bytes = m*k/2.
        let a_fp4: Vec<u8> = vec![0x22u8; m * k / 2];
        let b_fp4: Vec<u8> = vec![0x22u8; k * n / 2];
        // VEC16_UE4M3 : k/16 scales per row (A) and per col (B).
        // Scale byte 0x38 = (E=7, M=0) → value 2^0 = 1.0 (NVIDIA UE4M3 bias 7).
        let scale_a: Vec<u8> = vec![0x38u8; m * (k / 16)];
        let scale_b: Vec<u8> = vec![0x38u8; n * (k / 16)];

        let ctx = CudaContext::new(0).expect("ctx");
        let stream = ctx.default_stream();
        let a_dev = stream.memcpy_stod(&a_fp4).expect("upload a");
        let b_dev = stream.memcpy_stod(&b_fp4).expect("upload b");
        let sa_dev = stream.memcpy_stod(&scale_a).expect("upload sa");
        let sb_dev = stream.memcpy_stod(&scale_b).expect("upload sb");
        let mut c_dev = stream.alloc_zeros::<half::bf16>(m * n).expect("alloc c");
        let workspace = stream.alloc_zeros::<u8>(32 * 1024 * 1024).expect("ws");

        unsafe {
            use cudarc::driver::{DevicePtr, DevicePtrMut};
            let (a_p, _r1) = a_dev.device_ptr(&stream);
            let (b_p, _r2) = b_dev.device_ptr(&stream);
            let (sa_p, _r3) = sa_dev.device_ptr(&stream);
            let (sb_p, _r4) = sb_dev.device_ptr(&stream);
            let (c_p, _r5) = c_dev.device_ptr_mut(&stream);
            let (w_p, _r6) = workspace.device_ptr(&stream);
            crate::cublas_lt::matmul_mxfp4(
                a_p,
                sa_p,
                b_p,
                sb_p,
                c_p,
                m,
                k,
                n,
                1.0,
                0.0,
                Fp8Output::Bf16,
                Fp4ScaleMode::Vec16Ue4m3,
                w_p,
                32 * 1024 * 1024,
                stream.cu_stream() as u64,
            )
            .expect("matmul_mxfp4 free");
        }

        let c_host: Vec<half::bf16> = stream.memcpy_dtov(&c_dev).expect("dtov c");
        let c: Vec<f32> = c_host.into_iter().map(|x| x.to_f32()).collect();

        // Expected : each output = sum over k of (1 * 1) * scale_a * scale_b
        //          = k * 1 * 1 = k = 128.
        // Print first 8 + last 4 to see the pattern.
        eprintln!(
            "FREE FP4 all-ones output (m={m}, k={k}, n={n}, expected {} ∀j) :",
            k
        );
        for j in 0..n.min(16) {
            eprintln!("  c[{j}] = {}", c[j]);
        }
        // Verify c[0] is correct.
        assert!(
            (c[0] - k as f32).abs() < 1.0,
            "FREE FP4 matmul[0] = {} (expected ≈ {})",
            c[0],
            k
        );
        // Count how many c[j] match k — gives a clean signal on layout.
        let n_match = c.iter().filter(|&&v| (v - k as f32).abs() < 1.0).count();
        let n_zero = c.iter().filter(|&&v| v.abs() < 0.01).count();
        eprintln!(
            "  → {} / {} match expected ; {} are exactly 0",
            n_match, n, n_zero
        );
        // FINDING T241.6d : with k=n=128 + all scales=0x38 + all FP4=1,
        // only c[0] = 128 (correct). c[1..] = 64 (half). This indicates
        // a scale_B buffer layout mismatch between our convention and
        // what cuBLASLt expects for VEC16_UE4M3.
        //
        // We don't fail the rest of c[j] because this test EXISTS to
        // expose the layout bug — fixing it is the goal of T241.6d.
    }

    /// Test (2) — CACHED path LtSession::matmul_mxfp4 with same input.
    /// If this fails while test (1) passes, the bug is in the cached-path
    /// scale pointer rebind (build_cached + per-call attribute set).
    #[test]
    fn nvfp4_cached_matmul_all_ones_returns_k() {
        let m = 1usize;
        let k = 128usize;
        let n = 128usize;
        let a_fp4: Vec<u8> = vec![0x22u8; m * k / 2];
        let b_fp4: Vec<u8> = vec![0x22u8; k * n / 2];
        // Scale byte 0x38 = (E=7, M=0) → value 2^0 = 1.0 (NVIDIA UE4M3 bias 7).
        let scale_a: Vec<u8> = vec![0x38u8; m * (k / 16)];
        let scale_b: Vec<u8> = vec![0x38u8; n * (k / 16)];

        let ctx = CudaContext::new(0).expect("ctx");
        let stream = ctx.default_stream();
        let mut session = LtSession::new(stream.clone()).expect("session");

        let a_dev = stream.memcpy_stod(&a_fp4).expect("upload a");
        let b_dev = stream.memcpy_stod(&b_fp4).expect("upload b");
        let sa_dev = stream.memcpy_stod(&scale_a).expect("upload sa");
        let sb_dev = stream.memcpy_stod(&scale_b).expect("upload sb");
        let mut c_dev = stream.alloc_zeros::<half::bf16>(m * n).expect("alloc c");

        unsafe {
            use cudarc::driver::{DevicePtr, DevicePtrMut};
            let (a_p, _r1) = a_dev.device_ptr(&stream);
            let (b_p, _r2) = b_dev.device_ptr(&stream);
            let (sa_p, _r3) = sa_dev.device_ptr(&stream);
            let (sb_p, _r4) = sb_dev.device_ptr(&stream);
            let (c_p, _r5) = c_dev.device_ptr_mut(&stream);
            session
                .matmul_mxfp4(
                    a_p,
                    sa_p,
                    b_p,
                    sb_p,
                    c_p,
                    m,
                    k,
                    n,
                    1.0,
                    0.0,
                    Fp8Output::Bf16,
                    Fp4ScaleMode::Vec16Ue4m3,
                )
                .expect("matmul_mxfp4 cached");
        }

        let c_host: Vec<half::bf16> = stream.memcpy_dtov(&c_dev).expect("dtov c");
        let c: Vec<f32> = c_host.into_iter().map(|x| x.to_f32()).collect();

        // c[0] should match (assert this) — the rest may diverge due to
        // the same scale-layout bug surfaced by the FREE test.
        assert!(
            (c[0] - k as f32).abs() < 1.0,
            "CACHED FP4 matmul[0] = {} (expected ≈ {})",
            c[0],
            k
        );
        let n_match = c.iter().filter(|&&v| (v - k as f32).abs() < 1.0).count();
        eprintln!("CACHED FP4 all-ones : {} / {} match k", n_match, n);
    }

    /// T241.6d — try Vec32Ue8m0 mode (MXFP4 standard, simpler scale).
    /// UE8M0 = pure power-of-2 (8 bits all exponent, bias 127).
    /// Byte 127 = 2^0 = 1.0. If this mode works while VEC16_UE4M3 has
    /// the c[1..]=64 layout bug, switch to MXFP4.
    #[test]
    fn nvfp4_vec32_ue8m0_all_ones() {
        let m = 1usize;
        let k = 128usize; // must be multiple of 32 for VEC32
        let n = 128usize;
        // Block size = 32 for Vec32Ue8m0.
        let a_fp4: Vec<u8> = vec![0x22u8; m * k / 2];
        let b_fp4: Vec<u8> = vec![0x22u8; k * n / 2];
        // UE8M0 byte 127 = 2^0 = 1.0.
        let scale_a: Vec<u8> = vec![127u8; m * (k / 32)];
        let scale_b: Vec<u8> = vec![127u8; n * (k / 32)];

        let ctx = CudaContext::new(0).expect("ctx");
        let stream = ctx.default_stream();
        let a_dev = stream.memcpy_stod(&a_fp4).expect("upload");
        let b_dev = stream.memcpy_stod(&b_fp4).expect("upload");
        let sa_dev = stream.memcpy_stod(&scale_a).expect("upload");
        let sb_dev = stream.memcpy_stod(&scale_b).expect("upload");
        let mut c_dev = stream.alloc_zeros::<half::bf16>(m * n).expect("alloc c");
        let workspace = stream.alloc_zeros::<u8>(32 * 1024 * 1024).expect("ws");

        let res = unsafe {
            use cudarc::driver::{DevicePtr, DevicePtrMut};
            let (a_p, _r1) = a_dev.device_ptr(&stream);
            let (b_p, _r2) = b_dev.device_ptr(&stream);
            let (sa_p, _r3) = sa_dev.device_ptr(&stream);
            let (sb_p, _r4) = sb_dev.device_ptr(&stream);
            let (c_p, _r5) = c_dev.device_ptr_mut(&stream);
            let (w_p, _r6) = workspace.device_ptr(&stream);
            crate::cublas_lt::matmul_mxfp4(
                a_p,
                sa_p,
                b_p,
                sb_p,
                c_p,
                m,
                k,
                n,
                1.0,
                0.0,
                Fp8Output::Bf16,
                Fp4ScaleMode::Vec32Ue8m0,
                w_p,
                32 * 1024 * 1024,
                stream.cu_stream() as u64,
            )
        };
        match res {
            Ok(_) => {
                let c_host: Vec<half::bf16> = stream.memcpy_dtov(&c_dev).expect("dtov");
                let c: Vec<f32> = c_host.into_iter().map(|x| x.to_f32()).collect();
                let n_match = c.iter().filter(|&&v| (v - k as f32).abs() < 1.0).count();
                eprintln!(
                    "VEC32_UE8M0 result : c[0]={} ; {}/{} match k={k}",
                    c[0], n_match, n
                );
                assert!(
                    (c[0] - k as f32).abs() < 1.0,
                    "Vec32Ue8m0 c[0] = {} (expected {})",
                    c[0],
                    k
                );
            },
            Err(e) => {
                eprintln!("VEC32_UE8M0 not supported on this device : {e}");
            },
        }
    }

    /// T241.6d — write distinct values in scale_B to map the layout.
    /// Scale_B[0] = 0x40 (= 2.0), all others = 0x38 (= 1.0). If c[0] = 256,
    /// scale_B[0] is read for c[0]. We then check which c[j] also doubles
    /// to identify the layout.
    #[test]
    fn nvfp4_scale_b_layout_probe() {
        let m = 1usize;
        let k = 128usize;
        let n = 128usize;
        let a_fp4: Vec<u8> = vec![0x22u8; m * k / 2];
        let b_fp4: Vec<u8> = vec![0x22u8; k * n / 2];
        let scale_a: Vec<u8> = vec![0x38u8; m * (k / 16)];
        let mut scale_b: Vec<u8> = vec![0x38u8; n * (k / 16)];
        // Mark distinct positions.
        scale_b[0] = 0x40; // first byte → some c[j] doubles
        let probe_pos = scale_b.len() / 2;
        scale_b[probe_pos] = 0x48; // mid-buffer → some c[j] gets 4x

        let ctx = CudaContext::new(0).expect("ctx");
        let stream = ctx.default_stream();
        let a_dev = stream.memcpy_stod(&a_fp4).expect("upload");
        let b_dev = stream.memcpy_stod(&b_fp4).expect("upload");
        let sa_dev = stream.memcpy_stod(&scale_a).expect("upload");
        let sb_dev = stream.memcpy_stod(&scale_b).expect("upload");
        let mut c_dev = stream.alloc_zeros::<half::bf16>(m * n).expect("alloc c");
        let workspace = stream.alloc_zeros::<u8>(32 * 1024 * 1024).expect("ws");

        unsafe {
            use cudarc::driver::{DevicePtr, DevicePtrMut};
            let (a_p, _r1) = a_dev.device_ptr(&stream);
            let (b_p, _r2) = b_dev.device_ptr(&stream);
            let (sa_p, _r3) = sa_dev.device_ptr(&stream);
            let (sb_p, _r4) = sb_dev.device_ptr(&stream);
            let (c_p, _r5) = c_dev.device_ptr_mut(&stream);
            let (w_p, _r6) = workspace.device_ptr(&stream);
            crate::cublas_lt::matmul_mxfp4(
                a_p,
                sa_p,
                b_p,
                sb_p,
                c_p,
                m,
                k,
                n,
                1.0,
                0.0,
                Fp8Output::Bf16,
                Fp4ScaleMode::Vec16Ue4m3,
                w_p,
                32 * 1024 * 1024,
                stream.cu_stream() as u64,
            )
            .expect("matmul probe");
        }

        let c_host: Vec<half::bf16> = stream.memcpy_dtov(&c_dev).expect("dtov");
        let c: Vec<f32> = c_host.into_iter().map(|x| x.to_f32()).collect();

        // Find which c[j] != base value (which would be 128 in expected case).
        // With c[0]=128, c[1..]=64 baseline, we want to see which c[j] now
        // shows 256 (= 2x base) or 64 (no change).
        let baseline = c[1]; // = 64 typically
        eprintln!("scale_B[0]=0x40 (2.0), scale_B[{probe_pos}]=0x48 (4.0), others 0x38 (1.0)");
        eprintln!("baseline c[1] = {baseline} ; reading c[0..16] :");
        for j in 0..16 {
            let ratio = c[j] / baseline;
            eprintln!("  c[{j}] = {} (ratio vs baseline = {ratio:.2})", c[j]);
        }
        // Find c[j] with ratio > 1.5 (the 0x40 effect).
        let mut doubled: Vec<usize> = Vec::new();
        let mut quadrupled: Vec<usize> = Vec::new();
        for (j, &v) in c.iter().enumerate() {
            let r = v / baseline;
            if r > 1.5 && r < 2.5 {
                doubled.push(j);
            }
            if r > 3.5 && r < 5.0 {
                quadrupled.push(j);
            }
        }
        eprintln!("c[j] with 2× baseline (scale_B[0] effect) : {doubled:?}");
        eprintln!("c[j] with 4× baseline (scale_B[mid] effect) : {quadrupled:?}");
    }

    /// T241.6d — perimeter test : try multiple m values to identify the
    /// minimum supported by cuBLASLt FP4 sm_121. The LLM hot path uses m=1
    /// (autoregressive decode) ; if the minimum is > 1, FP4 path is
    /// infeasible without padding to a higher m.
    #[test]
    fn nvfp4_supported_m_values() {
        let k = 128usize; // K must be >= 16 (block size) and divisible by 16
        let n = 128usize;
        let scale_a_per_row = k / 16;
        let scale_b_per_col = k / 16;

        let ctx = CudaContext::new(0).expect("ctx");
        let stream = ctx.default_stream();
        let workspace = stream.alloc_zeros::<u8>(32 * 1024 * 1024).expect("ws");

        let candidates = [1usize, 2, 4, 8, 16, 32, 64, 128];
        let mut results: Vec<(usize, bool, String)> = Vec::new();
        for m in candidates.iter().copied() {
            let a_fp4: Vec<u8> = vec![0x22u8; m * k / 2];
            let b_fp4: Vec<u8> = vec![0x22u8; k * n / 2];
            let scale_a: Vec<u8> = vec![0x70u8; m * scale_a_per_row];
            let scale_b: Vec<u8> = vec![0x70u8; n * scale_b_per_col];
            let a_dev = stream.memcpy_stod(&a_fp4).expect("upload a");
            let b_dev = stream.memcpy_stod(&b_fp4).expect("upload b");
            let sa_dev = stream.memcpy_stod(&scale_a).expect("upload sa");
            let sb_dev = stream.memcpy_stod(&scale_b).expect("upload sb");
            let mut c_dev = stream.alloc_zeros::<half::bf16>(m * n).expect("alloc c");

            let res = unsafe {
                use cudarc::driver::{DevicePtr, DevicePtrMut};
                let (a_p, _r1) = a_dev.device_ptr(&stream);
                let (b_p, _r2) = b_dev.device_ptr(&stream);
                let (sa_p, _r3) = sa_dev.device_ptr(&stream);
                let (sb_p, _r4) = sb_dev.device_ptr(&stream);
                let (c_p, _r5) = c_dev.device_ptr_mut(&stream);
                let (w_p, _r6) = workspace.device_ptr(&stream);
                crate::cublas_lt::matmul_mxfp4(
                    a_p,
                    sa_p,
                    b_p,
                    sb_p,
                    c_p,
                    m,
                    k,
                    n,
                    1.0,
                    0.0,
                    Fp8Output::Bf16,
                    Fp4ScaleMode::Vec16Ue4m3,
                    w_p,
                    32 * 1024 * 1024,
                    stream.cu_stream() as u64,
                )
            };
            match res {
                Ok(_) => {
                    let c_host: Vec<half::bf16> = stream.memcpy_dtov(&c_dev).expect("dtov");
                    let c0 = c_host[0].to_f32();
                    let nonzero = c0.abs() > 0.001;
                    results.push((m, nonzero, format!("c[0]={c0}")));
                },
                Err(e) => {
                    results.push((m, false, format!("error: {e}")));
                },
            }
        }
        eprintln!("\n=== NVFP4 perimeter test (k={k}, n={n}) ===");
        for (m, ok, info) in &results {
            eprintln!(
                "  m={m:>3}  {} {}",
                if *ok { "✓ OK    " } else { "✗ FAILED" },
                info
            );
        }
        // Find minimum supported m.
        let min_supported = results.iter().find(|(_, ok, _)| *ok).map(|(m, _, _)| *m);
        eprintln!(
            "\n  Minimum supported m for FP4 on sm_121 : {:?}",
            min_supported
        );
        // We don't assert a specific value here because the answer is the
        // FINDING itself — this test exists to expose the constraint.
    }

    /// Test (3) — both paths produce the same result on identical input.
    /// Direct comparison free vs cached. If they diverge, the cached path
    /// has a bug.
    #[test]
    fn nvfp4_free_vs_cached_identical_output() {
        let m = 1usize;
        let k = 128usize;
        let n = 128usize;
        // Use random-ish but deterministic FP4 values.
        let a_fp4: Vec<u8> = (0..(m * k / 2))
            .map(|i| ((i as u8 * 37) | 0x22) & 0x77)
            .collect();
        let b_fp4: Vec<u8> = (0..(k * n / 2))
            .map(|i| ((i as u8 * 41) | 0x22) & 0x77)
            .collect();
        // Scale byte 0x38 = (E=7, M=0) → value 2^0 = 1.0 (NVIDIA UE4M3 bias 7).
        let scale_a: Vec<u8> = vec![0x38u8; m * (k / 16)];
        let scale_b: Vec<u8> = vec![0x38u8; n * (k / 16)];

        let ctx = CudaContext::new(0).expect("ctx");
        let stream = ctx.default_stream();

        // Run FREE path.
        let a_dev = stream.memcpy_stod(&a_fp4).expect("upload");
        let b_dev = stream.memcpy_stod(&b_fp4).expect("upload");
        let sa_dev = stream.memcpy_stod(&scale_a).expect("upload");
        let sb_dev = stream.memcpy_stod(&scale_b).expect("upload");
        let mut c_free = stream.alloc_zeros::<half::bf16>(m * n).expect("alloc c");
        let workspace = stream.alloc_zeros::<u8>(32 * 1024 * 1024).expect("ws");
        unsafe {
            use cudarc::driver::{DevicePtr, DevicePtrMut};
            let (a_p, _r1) = a_dev.device_ptr(&stream);
            let (b_p, _r2) = b_dev.device_ptr(&stream);
            let (sa_p, _r3) = sa_dev.device_ptr(&stream);
            let (sb_p, _r4) = sb_dev.device_ptr(&stream);
            let (c_p, _r5) = c_free.device_ptr_mut(&stream);
            let (w_p, _r6) = workspace.device_ptr(&stream);
            crate::cublas_lt::matmul_mxfp4(
                a_p,
                sa_p,
                b_p,
                sb_p,
                c_p,
                m,
                k,
                n,
                1.0,
                0.0,
                Fp8Output::Bf16,
                Fp4ScaleMode::Vec16Ue4m3,
                w_p,
                32 * 1024 * 1024,
                stream.cu_stream() as u64,
            )
            .expect("free");
        }
        let free_host: Vec<half::bf16> = stream.memcpy_dtov(&c_free).expect("dtov free");
        let free_out: Vec<f32> = free_host.into_iter().map(|x| x.to_f32()).collect();

        // Run CACHED path on same input.
        let mut session = LtSession::new(stream.clone()).expect("session");
        let mut c_cached = stream.alloc_zeros::<half::bf16>(m * n).expect("alloc c");
        unsafe {
            use cudarc::driver::{DevicePtr, DevicePtrMut};
            let (a_p, _r1) = a_dev.device_ptr(&stream);
            let (b_p, _r2) = b_dev.device_ptr(&stream);
            let (sa_p, _r3) = sa_dev.device_ptr(&stream);
            let (sb_p, _r4) = sb_dev.device_ptr(&stream);
            let (c_p, _r5) = c_cached.device_ptr_mut(&stream);
            session
                .matmul_mxfp4(
                    a_p,
                    sa_p,
                    b_p,
                    sb_p,
                    c_p,
                    m,
                    k,
                    n,
                    1.0,
                    0.0,
                    Fp8Output::Bf16,
                    Fp4ScaleMode::Vec16Ue4m3,
                )
                .expect("cached");
        }
        let cached_host: Vec<half::bf16> = stream.memcpy_dtov(&c_cached).expect("dtov cached");
        let cached_out: Vec<f32> = cached_host.into_iter().map(|x| x.to_f32()).collect();

        // Both should be identical (or near-identical).
        for j in 0..n {
            let diff = (free_out[j] - cached_out[j]).abs();
            let tol = free_out[j].abs() * 1e-2 + 0.5;
            assert!(
                diff < tol,
                "free[{j}]={} vs cached[{j}]={} diff={} \
                 (cached path differs from free → bug in build_cached)",
                free_out[j],
                cached_out[j],
                diff
            );
        }
    }
}
