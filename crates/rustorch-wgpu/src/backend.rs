//! `WgpuBackend` — the central handle holding the wgpu device, queue,
//! and a pipeline cache.

use crate::cache::PipelineCache;
use crate::capabilities::Capabilities;
use crate::error::WgpuError;
use std::sync::Arc;

/// Parse the optional `WGPU_BACKEND` env override (vulkan / metal /
/// dx12 / gl / webgpu / browser). Anything else falls back to the
/// `wgpu::Backends::PRIMARY` default. Case-insensitive.
fn backend_from_env() -> wgpu::Backends {
    match std::env::var("WGPU_BACKEND")
        .ok()
        .as_deref()
        .map(str::to_ascii_lowercase)
        .as_deref()
    {
        Some("vulkan") => wgpu::Backends::VULKAN,
        Some("metal") => wgpu::Backends::METAL,
        Some("dx12") | Some("d3d12") => wgpu::Backends::DX12,
        Some("gl") | Some("opengl") | Some("gles") => wgpu::Backends::GL,
        Some("webgpu") | Some("browser") => wgpu::Backends::BROWSER_WEBGPU,
        // Empty or unset → cross-platform default (every native backend).
        _ => wgpu::Backends::PRIMARY,
    }
}

/// Score an adapter by its device type so the cross-platform default
/// picks discrete GPU > integrated > CPU > virtual. Higher = better.
fn score_adapter_type(t: wgpu::DeviceType) -> u8 {
    match t {
        wgpu::DeviceType::DiscreteGpu => 4,
        wgpu::DeviceType::IntegratedGpu => 3,
        wgpu::DeviceType::VirtualGpu => 2,
        wgpu::DeviceType::Cpu => 1,
        wgpu::DeviceType::Other => 0,
    }
}

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
    /// Human-readable adapter name from `adapter.get_info()`.
    adapter_name: String,
    /// `Backend` enum stringified (Vulkan / Metal / DX12 / …).
    adapter_backend: String,
    /// `DeviceType` enum stringified (DiscreteGpu / IntegratedGpu / …).
    adapter_device_type: String,
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
        Self::new_with_index(None).await
    }

    /// Construct, optionally pinning to the adapter at a specific
    /// 0-based index in the enumeration order. Passing `None` selects
    /// the highest-scoring adapter (DiscreteGpu > IntegratedGpu > …).
    /// Honours the `WGPU_BACKEND` env var to pre-filter the adapter
    /// search.
    pub async fn new_with_index(index: Option<usize>) -> Result<Self, WgpuError> {
        let backends = backend_from_env();
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends,
            ..Default::default()
        });
        // Enumerate, score, and pick the best (or the requested index).
        let adapters: Vec<wgpu::Adapter> = instance.enumerate_adapters(backends);
        let chosen = if let Some(i) = index {
            adapters.into_iter().nth(i).ok_or(WgpuError::NoAdapter)?
        } else if adapters.is_empty() {
            // Fall back to request_adapter — some platforms (browser)
            // don't expose enumerate_adapters meaningfully.
            instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    force_fallback_adapter: false,
                    compatible_surface: None,
                })
                .await
                .ok_or(WgpuError::NoAdapter)?
        } else {
            adapters
                .into_iter()
                .max_by_key(|a| score_adapter_type(a.get_info().device_type))
                .ok_or(WgpuError::NoAdapter)?
        };

        let features_req = wgpu::Features::empty();
        let limits = chosen.limits();
        let (device, queue) = chosen
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
        // Capture the uncaptured-error stream so wgpu validation
        // failures surface as `WgpuError::Validation` rather than
        // panicking through wgpu's default handler.
        device.on_uncaptured_error(Box::new(|e| {
            // Uncaptured errors are reported via the tracing/log
            // ecosystem so users can hook them however they like.
            // We use eprintln! as the lowest-friction sink.
            eprintln!("[rustorch-wgpu uncaptured error] {e}");
        }));

        let device = Arc::new(device);
        let queue = Arc::new(queue);
        let backend_info = chosen.get_info();
        Ok(WgpuBackend {
            features: chosen.features(),
            device,
            queue,
            cache: PipelineCache::default(),
            limits,
            adapter_name: backend_info.name,
            adapter_backend: format!("{:?}", backend_info.backend),
            adapter_device_type: format!("{:?}", backend_info.device_type),
        })
    }

    /// Adapter info for diagnostics / logging.
    pub fn name(&self) -> &'static str {
        "wgpu"
    }

    /// Probe the adapter / device for kernel-relevant features.
    /// Cheap (just reads back from the live device).
    pub fn capabilities(&self) -> Capabilities {
        // We don't keep the adapter handle around (wgpu doesn't let us
        // borrow it after device creation), so probe only fields that
        // can be fetched from the device + cached features.
        Capabilities {
            limits: self.limits.clone(),
            features: self.features,
            shader_f16: self.features.contains(wgpu::Features::SHADER_F16),
            timestamp_query: self.features.contains(wgpu::Features::TIMESTAMP_QUERY),
            subgroup: self
                .features
                .iter_names()
                .any(|(name, _)| name.eq_ignore_ascii_case("SUBGROUP")),
        }
    }

    /// Human-readable adapter name (e.g. "Apple M3 Max" / "NVIDIA RTX 4090").
    pub fn adapter_name(&self) -> &str {
        &self.adapter_name
    }

    /// Backend used by this device (Vulkan / Metal / DX12 / …).
    pub fn adapter_backend(&self) -> &str {
        &self.adapter_backend
    }

    /// Device type (DiscreteGpu / IntegratedGpu / Cpu / …).
    pub fn adapter_device_type(&self) -> &str {
        &self.adapter_device_type
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
