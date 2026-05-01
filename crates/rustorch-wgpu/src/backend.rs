//! `WgpuBackend` — the central handle holding the wgpu device, queue,
//! and a pipeline cache.

use crate::cache::PipelineCache;
use crate::error::WgpuError;
use std::sync::Arc;

/// The GPU backend handle. Holds an [`wgpu::Instance`], its negotiated
/// [`wgpu::Adapter`] / [`wgpu::Device`] / [`wgpu::Queue`], and the
/// pipeline cache.
///
/// Construct via [`WgpuBackend::new_blocking`] which performs the
/// async dance synchronously (uses `pollster`).
pub struct WgpuBackend {
    /// Underlying device (handle to the GPU).
    pub device: Arc<wgpu::Device>,
    /// Submit queue for compute work.
    pub queue: Arc<wgpu::Queue>,
    /// Compiled-pipeline cache, keyed by op signature.
    pub cache: PipelineCache,
    /// Limits negotiated with the adapter (workgroup sizes, max
    /// storage buffer size, …).
    pub limits: wgpu::Limits,
    /// Adapter-level features actually granted (subset of requested).
    pub features: wgpu::Features,
}

impl WgpuBackend {
    /// Synchronously initialise a `WgpuBackend` using the default
    /// adapter discovery path.
    ///
    /// Returns an error if no adapter is available (e.g. CI without
    /// GPU) so that callers can fall back to CPU.
    pub fn new_blocking() -> Result<Self, WgpuError> {
        pollster::block_on(Self::new())
    }

    /// Async constructor — call from a tokio/futures runtime when one
    /// is already in scope.
    pub async fn new() -> Result<Self, WgpuError> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            ..Default::default()
        });
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
            })
            .await
            .ok_or(WgpuError::NoAdapter)?;
        let features_req = wgpu::Features::empty();
        let limits = adapter.limits();
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("rustorch-wgpu device"),
                    required_features: features_req,
                    required_limits: limits.clone(),
                    memory_hints: wgpu::MemoryHints::Performance,
                },
                None,
            )
            .await
            .map_err(|e| WgpuError::DeviceRequest(e.to_string()))?;
        let device = Arc::new(device);
        let queue = Arc::new(queue);
        Ok(WgpuBackend {
            features: adapter.features(),
            device,
            queue,
            cache: PipelineCache::default(),
            limits,
        })
    }

    /// Adapter info for diagnostics / logging.
    pub fn name(&self) -> &'static str {
        "wgpu"
    }
}

#[cfg(all(test, feature = "gpu-tests"))]
mod tests {
    use super::*;

    #[test]
    fn device_init_smoke() {
        // Only runs when the gpu-tests feature is enabled, since CI
        // without a GPU will fail this.
        let backend = WgpuBackend::new_blocking().expect("init wgpu backend");
        assert!(backend.limits.max_compute_workgroup_size_x >= 64);
    }
}
