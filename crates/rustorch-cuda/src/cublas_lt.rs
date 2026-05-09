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

/// Run an MXFP4 GEMM (FP4 inputs with VEC32_UE8M0 block scaling).
///
/// FP4 inputs are byte-packed (1 byte = 2 elements). Each block of 32 FP4
/// elements has one UE8M0 scale (1 byte: 8-bit unsigned exponent, no
/// mantissa). For an [M × K] FP4 matrix, scales form an [M × (K/32)] UE8M0
/// matrix. K must be a multiple of 32.
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
    workspace_dev: u64,
    workspace_bytes: usize,
    stream: u64,
) -> Result<(), CudaError> {
    if m == 0 || n == 0 || k == 0 {
        return Ok(());
    }
    if k % 32 != 0 {
        return Err(CudaError::Unsupported {
            msg: format!("MXFP4 requires k % 32 == 0, got k={k}"),
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

    // Scale modes: VEC32_UE8M0 means "one UE8M0 scale per 32 FP4 elements".
    // This is the MXFP4 standard.
    let scale_mode = sys::cublasLtMatmulMatrixScale_t::CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0;
    result::set_matmul_desc_attribute(
        matmul_desc,
        sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_A_SCALE_MODE,
        (&scale_mode) as *const _ as *const _,
        std::mem::size_of::<sys::cublasLtMatmulMatrixScale_t>(),
    )
    .map_err(|e| CudaError::CublasStatus {
        code: lt_err_code(e),
        location: "cublas_lt::matmul_mxfp4::set_a_scale_mode",
    })?;
    result::set_matmul_desc_attribute(
        matmul_desc,
        sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_B_SCALE_MODE,
        (&scale_mode) as *const _ as *const _,
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
    _workspace_dev: u64,
    _workspace_bytes: usize,
    _stream: u64,
) -> Result<(), CudaError> {
    Err(CudaError::NoDeviceFound)
}
