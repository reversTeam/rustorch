//! Cross-target async runtime helpers.
//!
//! On native, `pollster::block_on(future)` drives a future to
//! completion on the current thread. On `wasm32-unknown-unknown`,
//! the browser already runs the JS event loop so blocking is not an
//! option — callers must `.await` the future from JS / web-bindings.
//!
//! [`block_on`] hides this difference: it accepts any `Future` and
//! returns its `Output` on native; on wasm32 it panics (the function
//! is not callable from sync context). Use it where the codebase
//! needs a single name regardless of target.
//!
//! [`submit_and_wait`] flushes the queue and blocks until the GPU
//! has acknowledged the submission — useful for benches and tests
//! where deterministic timing matters.

use crate::backend::WgpuBackend;
use std::future::Future;

/// Drive a future to completion. On native, uses `pollster`; on wasm32,
/// panics — the browser must `.await` the future from JS.
#[cfg(not(target_arch = "wasm32"))]
pub fn block_on<F: Future>(fut: F) -> F::Output {
    pollster::block_on(fut)
}

/// `wasm32` shim: blocking sync runtime is unavailable in the browser.
/// Callers must use `wasm-bindgen-futures` and `await` the future
/// from JS instead. Calling this at runtime panics.
#[cfg(target_arch = "wasm32")]
pub fn block_on<F: Future>(_fut: F) -> F::Output {
    panic!("rustorch_wgpu::runtime::block_on is unavailable on wasm32 — use .await from JS")
}

/// Submit `command_buffers` and block until the GPU has finished
/// executing them. Returns the `SubmissionIndex` so callers can hand
/// it to a [`crate::staging::StagingRing`] for fence tracking.
///
/// On wasm32, blocks on `device.poll(Wait)` is not a real wait
/// because the browser can't yield from sync code; callers should
/// instead use `to_cpu_async` and await it.
pub fn submit_and_wait<I>(backend: &WgpuBackend, command_buffers: I) -> wgpu::SubmissionIndex
where
    I: IntoIterator<Item = wgpu::CommandBuffer>,
{
    let idx = backend.queue.submit(command_buffers);
    backend.device.poll(wgpu::Maintain::Wait);
    idx
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn block_on_runs_a_simple_future() {
        let v = block_on(async { 1 + 2 });
        assert_eq!(v, 3);
    }
}

#[cfg(all(test, feature = "gpu-tests", not(target_arch = "wasm32")))]
mod gpu_tests {
    use super::*;

    #[test]
    fn submit_and_wait_returns_a_submission_index() {
        let backend = WgpuBackend::new_blocking().expect("init");
        // Empty submit — just verify the helper threads through.
        let _ = submit_and_wait(&backend, std::iter::empty::<wgpu::CommandBuffer>());
    }
}
