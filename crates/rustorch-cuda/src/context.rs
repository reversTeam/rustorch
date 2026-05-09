//! CUDA driver / runtime context initialisation.
//!
//! With `--features cuda`, `Context::init()` initialises the CUDA driver
//! (via cudarc, which calls `cuInit(0)` on first construction) and creates
//! / retrieves the primary CUDA context for the requested device. The
//! resulting `Arc<CudaContext>` is reference-counted so additional
//! subsystems (cuBLAS, cuDNN, kernels) can `CudaContext::new(idx)` ad-hoc
//! without re-initialising the driver — cudarc dedups the primary context
//! per device internally.
//!
//! Without `--features cuda`, `Context::init()` returns
//! `Err(NoDeviceFound)` so call sites can degrade gracefully.

use crate::error::CudaError;

/// Process-global CUDA context handle. Initialised by [`Context::init`].
///
/// On `--features cuda` builds, holds an `Arc<cudarc::driver::CudaContext>`
/// pointing at the primary context for device 0 (configurable via
/// `Context::init_on_device`). On no-cuda builds, holds only the
/// `initialised` boolean (always false).
#[derive(Debug)]
pub struct Context {
    /// True iff the CUDA driver was successfully initialised.
    initialised: bool,
    /// Active device index (0 by default).
    #[cfg(feature = "cuda")]
    device_index: usize,
    /// Refcounted handle to the primary CUDA context. Held across the
    /// lifetime of this `Context` to ensure the primary context outlives
    /// any child resources (streams, buffers) we create from it.
    #[cfg(feature = "cuda")]
    inner: std::sync::Arc<cudarc::driver::CudaContext>,
}

impl Context {
    /// Initialise the CUDA driver on device 0. On no-cuda builds returns
    /// `Err(NoDeviceFound)` immediately.
    #[cfg(not(feature = "cuda"))]
    pub fn init() -> Result<Self, CudaError> {
        Err(CudaError::NoDeviceFound)
    }

    /// Initialise the CUDA driver on device 0.
    #[cfg(feature = "cuda")]
    pub fn init() -> Result<Self, CudaError> {
        Self::init_on_device(0)
    }

    /// Initialise the CUDA driver on `device_index`. cuda-feature only.
    #[cfg(feature = "cuda")]
    pub fn init_on_device(device_index: usize) -> Result<Self, CudaError> {
        let inner = cudarc::driver::CudaContext::new(device_index).map_err(|e| {
            // cudarc's DriverError wraps the underlying CUresult; its
            // error code mapping isn't stable across versions, so we
            // capture a useful but coarse signal: 0 if NoDeviceFound,
            // otherwise hash the debug string to surface a non-zero code
            // for diagnostics.
            CudaError::Driver {
                code: hash_diag(&format!("{e:?}")),
                location: "context::init",
            }
        })?;
        Ok(Self {
            initialised: true,
            device_index,
            inner,
        })
    }

    /// True iff the CUDA driver was successfully initialised.
    pub fn is_initialised(&self) -> bool {
        self.initialised
    }

    /// Active device index. Defaults to 0; set by `init_on_device`.
    #[cfg(feature = "cuda")]
    pub fn device_index(&self) -> usize {
        self.device_index
    }

    /// Borrow the inner cudarc context. Used by subsystems that need
    /// to allocate streams, modules, etc. on the same primary context.
    #[cfg(feature = "cuda")]
    pub fn inner(&self) -> &std::sync::Arc<cudarc::driver::CudaContext> {
        &self.inner
    }
}

/// Hash a diagnostic string into a stable non-zero i32 for embedding in
/// `CudaError::Driver { code, .. }`. Coarse signal — we don't pretend to
/// reverse-map this back to a specific CUresult.
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
        assert!(
            r.is_ok(),
            "Context::init should succeed on a CUDA host: {r:?}"
        );
        let ctx = r.unwrap();
        assert!(ctx.is_initialised());
        assert_eq!(ctx.device_index(), 0);
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn init_twice_yields_same_primary_context() {
        // cudarc dedups the primary context per device — both calls
        // should hand back Arcs to the same underlying object.
        let a = Context::init().unwrap();
        let b = Context::init().unwrap();
        assert!(std::sync::Arc::ptr_eq(a.inner(), b.inner()));
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn init_on_invalid_device_returns_driver_error() {
        // Device index way past anything the host has.
        let r = Context::init_on_device(99_999);
        match r {
            Err(CudaError::Driver { location, .. }) => {
                assert_eq!(location, "context::init");
            },
            other => panic!("expected Driver error, got {other:?}"),
        }
    }
}
