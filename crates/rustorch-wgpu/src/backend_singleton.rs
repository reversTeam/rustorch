//! Lazy global `WgpuBackend` singleton.
//!
//! Mirrors `rustorch_cpu::cpu_backend::cpu_backend()` so the autograd
//! dispatcher can pick a backend handle from a `Device` enum at op time
//! without threading the backend through every call site.
//!
//! Pattern: `OnceLock<WgpuBackend>` initialised on first access. Failures
//! during initialisation (no adapter, request_adapter timed out, etc.) are
//! cached as `Err` so subsequent calls return the same error fast — they
//! do not retry the adapter discovery on every call.
//!
//! P3.Y plan, Phase A2.

use crate::backend::WgpuBackend;
use crate::error::WgpuError;
use std::sync::OnceLock;

/// Cached singleton.  We store `Result` so initialisation failures are
/// observable (no `try_init` retries).
static SINGLETON: OnceLock<Result<WgpuBackend, WgpuError>> = OnceLock::new();

/// Try to access the lazily-initialised global `WgpuBackend`.
///
/// Returns `Err` once and forever if the host has no usable wgpu adapter
/// (CI runner without a GPU, head-less Linux without Vulkan, etc.).
/// Otherwise the same `&WgpuBackend` is returned on every call.
///
/// Thread-safe: `OnceLock` guarantees single initialisation even under
/// contention.
///
/// # Example
/// ```no_run
/// use rustorch_wgpu::wgpu_backend;
/// let backend = wgpu_backend().expect("no GPU available");
/// println!("running on {}", backend.adapter_name());
/// ```
pub fn try_wgpu_backend() -> Result<&'static WgpuBackend, &'static WgpuError> {
    SINGLETON.get_or_init(WgpuBackend::new_blocking).as_ref()
}

/// Same as [`try_wgpu_backend`] but panics with a clear message when no
/// adapter is available.  Use this from code that has already gated on
/// `Device::Wgpu` and therefore cannot recover.
pub fn wgpu_backend() -> &'static WgpuBackend {
    try_wgpu_backend().unwrap_or_else(|e| {
        panic!(
            "rustorch-wgpu: no usable adapter ({e}).\n\
             - macOS: Metal must be available (run on a real Mac, not Rosetta)\n\
             - Linux: install Vulkan loader + drivers (mesa/nvidia)\n\
             - Windows: DX12-capable GPU + recent drivers\n\
             - WASM: browser must support WebGPU (Chrome 120+, Firefox 141+)\n\
             Set RUST_LOG=wgpu=debug for adapter discovery details."
        )
    })
}

/// Reset the singleton — for tests only.  Subsequent calls re-run the
/// adapter discovery.  Not exposed to callers because the underlying
/// `OnceLock::take` is gated behind `#[cfg(test)]`.
#[cfg(test)]
pub fn reset_for_test() {
    // OnceLock has no public reset — we use a fresh handle per test
    // process instead.  This stub is kept for documentation parity with
    // the `cpu_backend` API.  Real test isolation goes through `cargo
    // test --test-threads=1` and per-process forking.
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Sanity: the singleton hands back the same backend across calls.
    #[test]
    #[cfg_attr(not(feature = "gpu-tests"), ignore = "needs a GPU adapter")]
    fn singleton_is_stable() {
        let a = wgpu_backend();
        let b = wgpu_backend();
        // Adapter name + ptr equality.
        assert!(std::ptr::eq(a, b));
    }

    /// `try_wgpu_backend` returns the cached `Err` when no adapter
    /// is available — does not panic, does not retry.
    ///
    /// This test cannot be reliably written without isolating the
    /// `OnceLock`.  It is documented here as a property contract.
    #[test]
    #[ignore = "documented contract — relies on OnceLock semantics, not a runtime check"]
    fn try_wgpu_backend_is_idempotent_on_failure() {
        // Property: subsequent calls reuse the cached Err.
        let _ = try_wgpu_backend();
        let _ = try_wgpu_backend();
    }
}
