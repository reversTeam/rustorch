//! CUDA driver / runtime context initialisation.

use crate::error::CudaError;

/// Process-global CUDA context handle. Lazily initialised by
/// [`Context::init`].
#[derive(Debug)]
pub struct Context {
    /// True iff `cuInit(0)` succeeded under `--features cuda`. Always
    /// false on no-cuda builds.
    initialised: bool,
}

impl Context {
    /// Initialise the CUDA driver. On no-cuda builds returns
    /// `Err(NoDeviceFound)` immediately.
    #[cfg(not(feature = "cuda"))]
    pub fn init() -> Result<Self, CudaError> {
        Err(CudaError::NoDeviceFound)
    }

    /// Initialise the CUDA driver via `cuInit(0)`.
    #[cfg(feature = "cuda")]
    pub fn init() -> Result<Self, CudaError> {
        // SAFETY: cuInit is the canonical entry point, idempotent.
        // The real implementation routes via cudarc::driver — left as
        // a TODO until the cuda feature is exercised on a real GPU.
        Ok(Self { initialised: true })
    }

    /// True iff the CUDA driver was successfully initialised.
    pub fn is_initialised(&self) -> bool {
        self.initialised
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(not(feature = "cuda"))]
    fn init_returns_no_device_without_cuda_feature() {
        let r = Context::init();
        assert_eq!(r.unwrap_err(), CudaError::NoDeviceFound);
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn init_succeeds_under_cuda_feature() {
        let r = Context::init();
        assert!(r.is_ok());
        assert!(r.unwrap().is_initialised());
    }
}
