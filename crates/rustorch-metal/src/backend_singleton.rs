//! Lazy global `MetalBackend` singleton.
//!
//! Mirrors `rustorch_wgpu::backend_singleton::wgpu_backend()` so the
//! autograd dispatcher can pick a Metal handle from a `Device::Metal`
//! marker at op time without threading the backend through every
//! call site.

use crate::backend::MetalBackend;
use crate::error::MetalError;
use std::sync::OnceLock;

static SINGLETON: OnceLock<Result<MetalBackend, MetalError>> = OnceLock::new();

/// Try to access the lazily-initialised global `MetalBackend`.
/// Cached `Err` on initialisation failure (no retries).
pub fn try_metal_backend() -> Result<&'static MetalBackend, &'static MetalError> {
    SINGLETON.get_or_init(MetalBackend::new).as_ref()
}

/// Same as [`try_metal_backend`] but panics with a clear message
/// when no adapter is available.
pub fn metal_backend() -> &'static MetalBackend {
    try_metal_backend().unwrap_or_else(|e| {
        panic!(
            "rustorch-metal: no Metal device ({e}).\n\
             - macOS only — this backend is conditionally compiled out\n\
               on Linux/Windows/wasm targets.\n\
             - Make sure you're on a Mac with a Metal-capable GPU\n\
               (any Apple Silicon, or x86 Mac with discrete GPU)."
        )
    })
}
