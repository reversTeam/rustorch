//! # rustorch-cuda
//!
//! Native CUDA backend foundation for rustorch. The public API is
//! always present (so portable call sites compile cleanly); the
//! actual cudarc FFI lights up under the `cuda` feature flag.
//!
//! Without `--features cuda`:
//! - `Context::init()` returns `Err(CudaError::NoDeviceFound)`
//! - `Device::all()` returns an empty Vec
//! - All compute paths return `Err(CudaError::Unsupported)`
//!
//! With `--features cuda`, the same APIs route to `cudarc` (driver
//! API) + cuBLAS / cuDNN / cuSPARSE / cuRAND.
//!
//! ## Modules
//! - [`error`] — `CudaError` enum + `check_*!` macros
//! - [`context`] — `Context::init()` + global handle
//! - [`device`] — `Device` + `Device::all()`
//! - [`stream`] — `Stream` + `StreamPool`
//! - [`event`] — `Event` + `EventStatus`
//! - [`cublas`] — gemm / gemv / batched gemm
//! - [`cudnn`] — conv / batchnorm / RNN descriptors
//! - [`cusparse`] — CSR/COO sparse-dense gemm
//! - [`curand`] — Philox / MTGP32 generators
//! - [`allocator`] — caching device allocator (size bins + coalescing)
//! - [`ops`] — high-level op dispatch routing rustorch-core ops to cuBLAS / cuDNN
//! - [`kernels`] — custom CUDA kernels (PTX templates + scalar fallback)
//! - [`nccl`] — NCCL collectives (all_reduce / all_gather / broadcast / reduce_scatter)

#![cfg_attr(docsrs, feature(doc_cfg))]
#![warn(missing_docs)]
#![warn(rust_2018_idioms)]

pub mod allocator;
pub mod context;
pub mod cublas;
pub mod cublas_lt;
pub mod cudnn;
pub mod curand;
pub mod cusparse;
pub mod cusparse_lt;
pub mod device;
pub mod error;
pub mod event;
pub mod kernels;
pub mod nccl;
pub mod ops;
pub mod stream;

pub use context::Context;
pub use device::{Device, DeviceProperties};
pub use error::CudaError;
pub use event::{Event, EventStatus};
pub use stream::{Stream, StreamPool};

/// Crate version.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// True iff this build was compiled with `--features cuda` (i.e. real
/// cudarc backing). False otherwise — the public API still works but
/// returns `NoDeviceFound` / `Unsupported` from compute paths.
pub const fn cuda_feature_enabled() -> bool {
    cfg!(feature = "cuda")
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn smoke_version_present() {
        assert!(!VERSION.is_empty());
    }
    #[test]
    fn feature_flag_is_known_at_compile_time() {
        // Just exercise the function so coverage counts it.
        let _ = cuda_feature_enabled();
    }
}
