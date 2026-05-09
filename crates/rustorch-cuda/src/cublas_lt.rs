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
    /// Default workspace = **256 MiB** (Blackwell-tuned, T240.8b). Hopper's
    /// rule of thumb was 32 MiB, but on GB10 the cuBLASLt heuristic needs
    /// more scratch to pick split-K and pipelined kernels for the larger
    /// shapes used in LLM prefill (FFN_up: M=2048, N=11008, K=2048 →
    /// ~90 MiB just for the algo's persistent state). Override via
    /// `RUSTORCH_CUBLASLT_WORKSPACE_MB` env var or `new_with_workspace`.
    pub fn new(stream: std::sync::Arc<cudarc::driver::CudaStream>) -> Result<Self, CudaError> {
        let mb = std::env::var("RUSTORCH_CUBLASLT_WORKSPACE_MB")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(256);
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
    let a_layout =
        result::create_matrix_layout(a_dt, k as u64, m as u64, k as i64).map_err(|e| {
            CudaError::CublasStatus {
                code: lt_err_code(e),
                location: "build_cached::a_layout",
            }
        })?;
    let b_layout =
        result::create_matrix_layout(b_dt, k as u64, n as u64, k as i64).map_err(|e| {
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
    let transb = cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N;
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
