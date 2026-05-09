//! cuSPARSELt 2:4 structured-sparsity matmul wrappers (T240.8c).
//!
//! cuSPARSELt is NVIDIA's library for 2:4 structured sparse matmuls,
//! introduced with Ampere (sm_80). On GB10 (sm_121) the library reports
//! 1.79× speedup at large M over dense — community-measured. Stacked with
//! NVFP4 (T240.8e) we project ~490 effective TFLOPS on Qwen prefill.
//!
//! cudarc 0.17 doesn't yet wrap cuSPARSELt, so we declare the minimal raw
//! FFI we need (handle/desc opaques + ~12 functions) and run them via
//! `extern "C"`. The build script (`build.rs`) tells rustc to link
//! against `libcusparseLt`; runtime resolution uses `LD_LIBRARY_PATH`
//! (typically `/usr/lib/aarch64-linux-gnu/libcusparseLt/13/`).
//!
//! ## Pipeline (per weight tensor, one-time setup)
//!
//! ```ignore
//! 1. cusparseLtInit(&handle)
//! 2. cusparseLtStructuredDescriptorInit(handle, &spA_desc, m, k, ld, ...)
//!    cusparseLtDenseDescriptorInit(handle, &B_desc, ...)        // activation
//!    cusparseLtDenseDescriptorInit(handle, &C_desc, ...)        // output
//! 3. cusparseLtMatmulDescriptorInit(handle, &matmul, op, op, &spA, &B, &C, &C, COMPUTE_32F)
//! 4. cusparseLtMatmulAlgSelectionInit(handle, &alg, &matmul, ALG_DEFAULT)
//! 5. cusparseLtMatmulPlanInit(handle, &plan, &matmul, &alg)
//! 6. cusparseLtSpMMAPrune(handle, &matmul, dense_in, dense_out_2_4, TILE)
//! 7. cusparseLtSpMMACompressedSize(handle, &plan, &compressed_size, &compressed_buf_size)
//! 8. cusparseLtSpMMACompress(handle, &plan, dense_2_4, compressed_buf, ...)
//! 9. (optional) cusparseLtMatmulSearch(handle, &plan, ...)  — autotune
//! ```
//!
//! Steps 1-9 happen once per weight (offline / model-load). Then per call:
//!
//! ```ignore
//! cusparseLtMatmul(handle, plan, &alpha, compressed, b_dev, &beta, c_dev, c_dev, workspace, ...)
//! ```

use crate::error::CudaError;

#[cfg(feature = "cuda")]
use cudarc::cublaslt::sys::cudaDataType_t;

// ─────────────────────────────────────────────────────────────────────────
// Opaque types (size taken from cusparseLt.h: alignas(16) uint8_t data[512])
// ─────────────────────────────────────────────────────────────────────────

#[cfg(feature = "cuda")]
#[repr(C, align(16))]
#[derive(Copy, Clone)]
struct CusparseLtHandle {
    _data: [u8; 512],
}

#[cfg(feature = "cuda")]
#[repr(C, align(16))]
#[derive(Copy, Clone)]
struct CusparseLtMatDescriptor {
    _data: [u8; 512],
}

#[cfg(feature = "cuda")]
#[repr(C, align(16))]
#[derive(Copy, Clone)]
struct CusparseLtMatmulDescriptor {
    _data: [u8; 512],
}

#[cfg(feature = "cuda")]
#[repr(C, align(16))]
#[derive(Copy, Clone)]
struct CusparseLtMatmulAlgSelection {
    _data: [u8; 512],
}

#[cfg(feature = "cuda")]
#[repr(C, align(16))]
#[derive(Copy, Clone)]
struct CusparseLtMatmulPlan {
    _data: [u8; 512],
}

// cuSPARSELt enums (matching /usr/include/libcusparseLt/13/cusparseLt.h)
#[cfg(feature = "cuda")]
#[repr(i32)]
#[allow(dead_code, non_camel_case_types)]
#[derive(Copy, Clone, Debug)]
enum CusparseStatus {
    Success = 0,
}

#[cfg(feature = "cuda")]
#[repr(i32)]
#[allow(dead_code, non_camel_case_types)]
#[derive(Copy, Clone, Debug)]
enum CusparseOperation {
    NonTranspose = 0,
    Transpose = 1,
}

#[cfg(feature = "cuda")]
#[repr(i32)]
#[allow(dead_code, non_camel_case_types)]
#[derive(Copy, Clone, Debug)]
enum CusparseOrder {
    Col = 1,
    Row = 2,
}

#[cfg(feature = "cuda")]
#[repr(i32)]
#[allow(dead_code, non_camel_case_types)]
#[derive(Copy, Clone, Debug)]
enum CusparseLtSparsity {
    Sparsity50Percent = 0,
}

#[cfg(feature = "cuda")]
#[repr(i32)]
#[allow(dead_code, non_camel_case_types)]
#[derive(Copy, Clone, Debug)]
enum CusparseComputeType {
    Compute32I = 0,
    Compute16F = 1,
    Compute32F = 2,
}

#[cfg(feature = "cuda")]
#[repr(i32)]
#[allow(dead_code, non_camel_case_types)]
#[derive(Copy, Clone, Debug)]
enum CusparseLtMatmulAlg {
    AlgDefault = 0,
}

#[cfg(feature = "cuda")]
#[repr(i32)]
#[allow(dead_code, non_camel_case_types)]
#[derive(Copy, Clone, Debug)]
enum CusparseLtPruneAlg {
    SpmmaTile = 0,
    SpmmaStrip = 1,
}

// ─────────────────────────────────────────────────────────────────────────
// Raw FFI
// ─────────────────────────────────────────────────────────────────────────

#[cfg(feature = "cuda")]
#[link(name = "cusparseLt", kind = "dylib")]
extern "C" {
    fn cusparseLtInit(handle: *mut CusparseLtHandle) -> i32;

    fn cusparseLtDestroy(handle: *const CusparseLtHandle) -> i32;

    #[allow(clippy::too_many_arguments)]
    fn cusparseLtDenseDescriptorInit(
        handle: *const CusparseLtHandle,
        mat_descr: *mut CusparseLtMatDescriptor,
        rows: i64,
        cols: i64,
        ld: i64,
        alignment: u32,
        value_type: cudaDataType_t,
        order: CusparseOrder,
    ) -> i32;

    #[allow(clippy::too_many_arguments)]
    fn cusparseLtStructuredDescriptorInit(
        handle: *const CusparseLtHandle,
        mat_descr: *mut CusparseLtMatDescriptor,
        rows: i64,
        cols: i64,
        ld: i64,
        alignment: u32,
        value_type: cudaDataType_t,
        order: CusparseOrder,
        sparsity: CusparseLtSparsity,
    ) -> i32;

    fn cusparseLtMatDescriptorDestroy(mat_descr: *const CusparseLtMatDescriptor) -> i32;

    #[allow(clippy::too_many_arguments)]
    fn cusparseLtMatmulDescriptorInit(
        handle: *const CusparseLtHandle,
        matmul_descr: *mut CusparseLtMatmulDescriptor,
        op_a: CusparseOperation,
        op_b: CusparseOperation,
        mat_a: *const CusparseLtMatDescriptor,
        mat_b: *const CusparseLtMatDescriptor,
        mat_c: *const CusparseLtMatDescriptor,
        mat_d: *const CusparseLtMatDescriptor,
        compute_type: CusparseComputeType,
    ) -> i32;

    fn cusparseLtMatmulAlgSelectionInit(
        handle: *const CusparseLtHandle,
        alg_selection: *mut CusparseLtMatmulAlgSelection,
        matmul_descr: *const CusparseLtMatmulDescriptor,
        alg: CusparseLtMatmulAlg,
    ) -> i32;

    fn cusparseLtMatmulPlanInit(
        handle: *const CusparseLtHandle,
        plan: *mut CusparseLtMatmulPlan,
        matmul_descr: *const CusparseLtMatmulDescriptor,
        alg_selection: *const CusparseLtMatmulAlgSelection,
    ) -> i32;

    fn cusparseLtMatmulPlanDestroy(plan: *const CusparseLtMatmulPlan) -> i32;

    fn cusparseLtMatmulGetWorkspace(
        handle: *const CusparseLtHandle,
        plan: *const CusparseLtMatmulPlan,
        workspace_size: *mut usize,
    ) -> i32;

    fn cusparseLtSpMMAPrune(
        handle: *const CusparseLtHandle,
        matmul_descr: *const CusparseLtMatmulDescriptor,
        d_in: *const std::ffi::c_void,
        d_out: *mut std::ffi::c_void,
        prune_alg: CusparseLtPruneAlg,
        stream: *mut std::ffi::c_void, // cudaStream_t
    ) -> i32;

    fn cusparseLtSpMMACompressedSize(
        handle: *const CusparseLtHandle,
        plan: *const CusparseLtMatmulPlan,
        compressed_size: *mut usize,
        compressed_buffer_size: *mut usize,
    ) -> i32;

    fn cusparseLtSpMMACompress(
        handle: *const CusparseLtHandle,
        plan: *const CusparseLtMatmulPlan,
        d_dense: *const std::ffi::c_void,
        d_compressed: *mut std::ffi::c_void,
        d_compressed_buffer: *mut std::ffi::c_void,
        stream: *mut std::ffi::c_void,
    ) -> i32;

    #[allow(clippy::too_many_arguments)]
    fn cusparseLtMatmul(
        handle: *const CusparseLtHandle,
        plan: *const CusparseLtMatmulPlan,
        alpha: *const std::ffi::c_void,
        d_a: *const std::ffi::c_void,
        d_b: *const std::ffi::c_void,
        beta: *const std::ffi::c_void,
        d_c: *const std::ffi::c_void,
        d_d: *mut std::ffi::c_void,
        workspace: *mut std::ffi::c_void,
        streams: *mut *mut std::ffi::c_void,
        num_streams: i32,
    ) -> i32;
}

// ─────────────────────────────────────────────────────────────────────────
// Safe wrappers
// ─────────────────────────────────────────────────────────────────────────

/// One precomputed compressed sparse weight + matmul plan, ready for
/// `SparseLtSession::matmul_bf16`.
#[cfg(feature = "cuda")]
pub struct SparseWeight {
    a_desc: CusparseLtMatDescriptor,
    b_desc: CusparseLtMatDescriptor,
    c_desc: CusparseLtMatDescriptor,
    matmul_desc: CusparseLtMatmulDescriptor,
    _alg_sel: CusparseLtMatmulAlgSelection,
    plan: CusparseLtMatmulPlan,
    /// Compressed sparse weight on device (owned).
    compressed: cudarc::driver::CudaSlice<u8>,
    /// Workspace required by this plan.
    workspace_size: usize,
    workspace: cudarc::driver::CudaSlice<u8>,
}

#[cfg(feature = "cuda")]
impl Drop for SparseWeight {
    fn drop(&mut self) {
        unsafe {
            let _ = cusparseLtMatmulPlanDestroy(&self.plan);
            let _ = cusparseLtMatDescriptorDestroy(&self.a_desc);
            let _ = cusparseLtMatDescriptorDestroy(&self.b_desc);
            let _ = cusparseLtMatDescriptorDestroy(&self.c_desc);
        }
    }
}

/// Reusable cuSPARSELt session. Holds the library handle and the bound
/// stream. Use `prune_compress_bf16` once per weight (offline / model-load
/// time), then `matmul_bf16` on the resulting `SparseWeight`.
#[cfg(feature = "cuda")]
pub struct SparseLtSession {
    handle: CusparseLtHandle,
    stream: std::sync::Arc<cudarc::driver::CudaStream>,
}

#[cfg(feature = "cuda")]
impl SparseLtSession {
    /// Create a new session bound to the given stream.
    pub fn new(stream: std::sync::Arc<cudarc::driver::CudaStream>) -> Result<Self, CudaError> {
        let mut handle = CusparseLtHandle { _data: [0u8; 512] };
        let status = unsafe { cusparseLtInit(&mut handle) };
        if status != CusparseStatus::Success as i32 {
            return Err(CudaError::CublasStatus {
                code: status,
                location: "SparseLtSession::new::cusparseLtInit",
            });
        }
        Ok(Self { handle, stream })
    }

    /// Prune a dense BF16 weight to 2:4 structured pattern, compress it,
    /// and build the matmul plan for shape (m, n, k). The compressed
    /// weight + plan is wrapped in `SparseWeight` for repeated use.
    ///
    /// Layout convention: weight is (k × n) row-major (B layout in the
    /// matmul A · B = C). For SwiGLU FFN on Qwen, prepass the gate+up
    /// fused weight (k=hidden, n=2*ffn).
    ///
    /// Pruning algorithm: `Tile` (default — best for inference accuracy).
    ///
    /// # Safety
    /// `dense_weight_dev` must be a valid BF16 buffer of `k * n` elements
    /// (= `k * n * 2` bytes). The buffer is read but not modified — the
    /// pruned + compressed copy lives inside the returned SparseWeight.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn prune_compress_bf16(
        &self,
        dense_weight_dev: u64,
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<SparseWeight, CudaError> {
        let order = CusparseOrder::Col;
        let value = cudaDataType_t::CUDA_R_16BF;
        let alignment: u32 = 16;
        let stream_ptr = self.stream.cu_stream() as *mut std::ffi::c_void;

        // Descriptors. We use:
        //   A = sparse weight  (m × k)  — but in our matmul we want
        //   sparse-on-B (the weight), so for cuBLAS-style A·B we set
        //   A=activation (dense), B=weight (sparse). cuSPARSELt API
        //   currently supports sparse-A only with structured-on-A. We
        //   model the matmul as B=weight·A (i.e., result is weight·act).
        //   For a weight of shape (k_w × n_w) being applied to an
        //   activation of shape (k_a × m_a) we have:
        //     m = n_w     (output rows = weight cols)
        //     k = k_w     (inner = weight rows)
        //     n = m_a     (output cols = activation cols)
        //   In cuSPARSELt's convention "sparse A" means the sparse op is
        //   the one with structured rows. We thus call it sparse-A with
        //   shape (m, k).

        let mut a_desc = CusparseLtMatDescriptor { _data: [0u8; 512] };
        let mut b_desc = CusparseLtMatDescriptor { _data: [0u8; 512] };
        let mut c_desc = CusparseLtMatDescriptor { _data: [0u8; 512] };
        let mut matmul_desc = CusparseLtMatmulDescriptor { _data: [0u8; 512] };
        let mut alg_sel = CusparseLtMatmulAlgSelection { _data: [0u8; 512] };
        let mut plan = CusparseLtMatmulPlan { _data: [0u8; 512] };

        // Sparse A: (m × k), col-major, ld=m
        check(
            cusparseLtStructuredDescriptorInit(
                &self.handle,
                &mut a_desc,
                m as i64,
                k as i64,
                m as i64,
                alignment,
                value,
                order,
                CusparseLtSparsity::Sparsity50Percent,
            ),
            "structured_desc_a",
        )?;
        // Dense B: (k × n), col-major, ld=k
        check(
            cusparseLtDenseDescriptorInit(
                &self.handle,
                &mut b_desc,
                k as i64,
                n as i64,
                k as i64,
                alignment,
                value,
                order,
            ),
            "dense_desc_b",
        )?;
        // Dense C: (m × n), col-major, ld=m
        check(
            cusparseLtDenseDescriptorInit(
                &self.handle,
                &mut c_desc,
                m as i64,
                n as i64,
                m as i64,
                alignment,
                value,
                order,
            ),
            "dense_desc_c",
        )?;

        check(
            cusparseLtMatmulDescriptorInit(
                &self.handle,
                &mut matmul_desc,
                CusparseOperation::NonTranspose,
                CusparseOperation::NonTranspose,
                &a_desc,
                &b_desc,
                &c_desc,
                &c_desc,
                CusparseComputeType::Compute32F,
            ),
            "matmul_desc_init",
        )?;

        check(
            cusparseLtMatmulAlgSelectionInit(
                &self.handle,
                &mut alg_sel,
                &matmul_desc,
                CusparseLtMatmulAlg::AlgDefault,
            ),
            "alg_sel_init",
        )?;

        check(
            cusparseLtMatmulPlanInit(&self.handle, &mut plan, &matmul_desc, &alg_sel),
            "plan_init",
        )?;

        // Workspace
        let mut workspace_size: usize = 0;
        check(
            cusparseLtMatmulGetWorkspace(&self.handle, &plan, &mut workspace_size),
            "get_workspace",
        )?;
        let workspace = self
            .stream
            .alloc_zeros::<u8>(workspace_size.max(1))
            .map_err(|e| CudaError::Driver {
                code: format!("{e:?}").len() as i32,
                location: "SparseLtSession::workspace_alloc",
            })?;

        // Prune dense_weight in-place into a new buffer
        let pruned_size = m * k * 2; // BF16 bytes
        let mut pruned: cudarc::driver::CudaSlice<u8> = self
            .stream
            .alloc_zeros::<u8>(pruned_size)
            .map_err(|e| CudaError::Driver {
                code: format!("{e:?}").len() as i32,
                location: "SparseLtSession::pruned_alloc",
            })?;
        let pruned_ptr = {
            use cudarc::driver::DevicePtrMut;
            pruned.device_ptr_mut(&self.stream).0 as *mut std::ffi::c_void
        };
        check(
            cusparseLtSpMMAPrune(
                &self.handle,
                &matmul_desc,
                dense_weight_dev as *const std::ffi::c_void,
                pruned_ptr,
                CusparseLtPruneAlg::SpmmaTile,
                stream_ptr,
            ),
            "prune",
        )?;

        // Compressed sizes
        let mut compressed_size: usize = 0;
        let mut compressed_buffer_size: usize = 0;
        check(
            cusparseLtSpMMACompressedSize(
                &self.handle,
                &plan,
                &mut compressed_size,
                &mut compressed_buffer_size,
            ),
            "compressed_size",
        )?;

        let compressed = self
            .stream
            .alloc_zeros::<u8>(compressed_size)
            .map_err(|e| CudaError::Driver {
                code: format!("{e:?}").len() as i32,
                location: "SparseLtSession::compressed_alloc",
            })?;
        let mut compressed_buffer = self
            .stream
            .alloc_zeros::<u8>(compressed_buffer_size.max(1))
            .map_err(|e| CudaError::Driver {
                code: format!("{e:?}").len() as i32,
                location: "SparseLtSession::compressed_buffer_alloc",
            })?;

        // Compress
        let pruned_const_ptr = {
            use cudarc::driver::DevicePtr;
            pruned.device_ptr(&self.stream).0 as *const std::ffi::c_void
        };
        let compressed_ptr = {
            use cudarc::driver::DevicePtr;
            compressed.device_ptr(&self.stream).0 as *mut std::ffi::c_void
        };
        let compressed_buffer_ptr = {
            use cudarc::driver::DevicePtrMut;
            compressed_buffer.device_ptr_mut(&self.stream).0 as *mut std::ffi::c_void
        };
        check(
            cusparseLtSpMMACompress(
                &self.handle,
                &plan,
                pruned_const_ptr,
                compressed_ptr,
                compressed_buffer_ptr,
                stream_ptr,
            ),
            "compress",
        )?;

        // Synchronize before pruned + compressed_buffer drop
        self.stream.synchronize().map_err(|e| CudaError::Driver {
            code: format!("{e:?}").len() as i32,
            location: "SparseLtSession::sync_after_compress",
        })?;
        // pruned and compressed_buffer are dropped here, freeing their memory.

        Ok(SparseWeight {
            a_desc,
            b_desc,
            c_desc,
            matmul_desc,
            _alg_sel: alg_sel,
            plan,
            compressed,
            workspace_size,
            workspace,
        })
    }

    /// Sparse · dense BF16 matmul.
    ///
    /// Computes  C = α · (sparse_weight · B) + β · C
    /// where the sparse weight is the precomputed `SparseWeight` and
    /// B / C are dense BF16 buffers on device.
    ///
    /// # Safety
    /// `b_dev` and `c_dev` must reference BF16 buffers of the right sizes
    /// (k · n and m · n elements respectively, matching the plan).
    pub unsafe fn matmul_bf16(
        &self,
        sparse: &mut SparseWeight,
        b_dev: u64,
        c_dev: u64,
        alpha: f32,
        beta: f32,
    ) -> Result<(), CudaError> {
        let stream_ptr = self.stream.cu_stream() as *mut std::ffi::c_void;
        let mut streams_array = [stream_ptr];
        let workspace_ptr = {
            use cudarc::driver::DevicePtrMut;
            sparse.workspace.device_ptr_mut(&self.stream).0 as *mut std::ffi::c_void
        };
        let compressed_ptr = {
            use cudarc::driver::DevicePtr;
            sparse.compressed.device_ptr(&self.stream).0 as *const std::ffi::c_void
        };
        check(
            cusparseLtMatmul(
                &self.handle,
                &sparse.plan,
                (&alpha) as *const f32 as *const _,
                compressed_ptr,
                b_dev as *const std::ffi::c_void,
                (&beta) as *const f32 as *const _,
                c_dev as *const std::ffi::c_void,
                c_dev as *mut std::ffi::c_void,
                workspace_ptr,
                streams_array.as_mut_ptr(),
                1,
            ),
            "matmul",
        )?;
        // Quiet unused-field warnings (we hold these for liveness/RAII)
        let _ = (
            sparse.a_desc._data[0],
            sparse.b_desc._data[0],
            sparse.c_desc._data[0],
            sparse.matmul_desc._data[0],
            sparse.workspace_size,
        );
        Ok(())
    }
}

#[cfg(feature = "cuda")]
impl Drop for SparseLtSession {
    fn drop(&mut self) {
        unsafe {
            let _ = cusparseLtDestroy(&self.handle);
        }
    }
}

#[cfg(feature = "cuda")]
fn check(status: i32, location: &'static str) -> Result<(), CudaError> {
    if status == CusparseStatus::Success as i32 {
        Ok(())
    } else {
        Err(CudaError::CublasStatus {
            code: status,
            location,
        })
    }
}

// ─────────────────────────────────────────────────────────────────────────
// Stubs without cuda feature
// ─────────────────────────────────────────────────────────────────────────

#[cfg(not(feature = "cuda"))]
/// Stub when `cuda` feature is off — all methods return Unsupported.
pub struct SparseLtSession;

#[cfg(not(feature = "cuda"))]
/// Stub.
pub struct SparseWeight;

#[cfg(not(feature = "cuda"))]
impl SparseLtSession {
    /// Stub.
    pub fn new<T>(_: T) -> Result<Self, CudaError> {
        Err(CudaError::Unsupported {
            msg: "cuSPARSELt requires --features cuda".into(),
        })
    }
}
