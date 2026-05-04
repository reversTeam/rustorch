//! `MetalBackend` — owns the `MTLDevice`, `MTLCommandQueue`, and the
//! pipeline-state cache.

use crate::error::MetalError;
use metal::{CompileOptions, ComputePipelineState, Device, MTLResourceOptions};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Apple Metal backend handle. Holds the [`Device`] and
/// [`CommandQueue`](metal::CommandQueue) used to allocate buffers
/// and dispatch compute work.
///
/// Cloning is cheap — both `Device` and `CommandQueue` are
/// reference-counted Apple objects (the `metal` crate's Rust types
/// wrap them in `foreign_types::ForeignType` smart pointers). The
/// `Arc` here gives us a Send+Sync handle suitable for the global
/// singleton (mirrors the `WgpuBackend` pattern in `rustorch-wgpu`).
pub struct MetalBackend {
    /// Underlying `MTLDevice` — handle to the GPU.
    pub device: Arc<Device>,
    /// Submit queue for compute work. Multi-buffered command buffers
    /// flow through this single queue (commitAndContinue pattern
    /// will multiplex up to N in-flight buffers in a follow-up).
    pub queue: Arc<metal::CommandQueue>,
    /// Pipeline-state cache keyed by kernel name. Avoids the ~0.3 ms
    /// MSL compile cost on every dispatch — first call to a kernel
    /// pays the compile, subsequent calls hit the cache. Same pattern
    /// as `rustorch_wgpu::cache::PipelineCache` but for Metal pipelines.
    pipeline_cache: Arc<Mutex<HashMap<&'static str, Arc<ComputePipelineState>>>>,
    /// Adapter name from `device.name()` (e.g. "Apple M4 Max").
    adapter_name: String,
    /// `true` if the device reports `supportsFamily(MTLGPUFamilyMetal3)`
    /// — gates the `simdgroup_matrix<bfloat,8,8>` kernel path. M3+ /
    /// M4 / Vision Pro / A17 Pro+ qualify; M1 / M2 fall through to
    /// the f16 + GradScaler path.
    supports_metal3: bool,
}

impl MetalBackend {
    /// Initialise the default-system Metal backend.
    ///
    /// Returns [`MetalError::NoDevice`] when running outside macOS or
    /// when no Metal-capable adapter is present (rare on Apple Silicon
    /// — typically only x86 Macs without integrated GPU surface this).
    pub fn new() -> Result<Self, MetalError> {
        let device = Device::system_default().ok_or_else(|| {
            MetalError::NoDevice("MTLCreateSystemDefaultDevice returned nil".to_string())
        })?;
        let queue = device.new_command_queue();
        let adapter_name = device.name().to_string();
        // `MTLGPUFamilyMetal3` is the gate for simdgroup_matrix on
        // Apple Silicon. M3+, M4, A17 Pro, Vision Pro qualify.
        let supports_metal3 = device.supports_family(metal::MTLGPUFamily::Metal3);
        Ok(MetalBackend {
            device: Arc::new(device),
            queue: Arc::new(queue),
            pipeline_cache: Arc::new(Mutex::new(HashMap::new())),
            adapter_name,
            supports_metal3,
        })
    }

    /// Compile or look up a compute pipeline state by kernel name.
    /// First call compiles `source` (paying the ~0.3 ms MSL compile
    /// cost); subsequent calls return the cached pipeline.
    ///
    /// `name` is the cache key — passing the same `name` with
    /// different `source` values returns the cached pipeline of the
    /// first call (kernels SHOULD pick a unique name per shader).
    pub fn pipeline(
        &self,
        name: &'static str,
        source: &str,
        entry: &str,
    ) -> Result<Arc<ComputePipelineState>, MetalError> {
        // Fast path: cache hit. Acquire the lock, look up, drop it.
        {
            let guard = self
                .pipeline_cache
                .lock()
                .expect("metal pipeline cache lock");
            if let Some(p) = guard.get(name) {
                return Ok(p.clone());
            }
        }
        // Slow path: compile + insert. Two threads racing on the same
        // name will both compile but only the first insert wins; the
        // second's compile is wasted GPU work but the pipeline graph
        // stays consistent.
        let library = self
            .device
            .new_library_with_source(source, &CompileOptions::new())
            .map_err(MetalError::ShaderCompile)?;
        let function = library
            .get_function(entry, None)
            .map_err(|e| MetalError::PipelineState(format!("get_function {entry}: {e}")))?;
        let pipeline = self
            .device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(MetalError::PipelineState)?;
        let pipeline = Arc::new(pipeline);
        let mut guard = self
            .pipeline_cache
            .lock()
            .expect("metal pipeline cache lock");
        Ok(guard
            .entry(name)
            .or_insert_with(|| pipeline.clone())
            .clone())
    }

    /// Adapter name reported by `MTLDevice::name()`.
    pub fn adapter_name(&self) -> &str {
        &self.adapter_name
    }

    /// `true` if the device supports `simdgroup_matrix` (Metal 3 family).
    pub fn supports_metal3(&self) -> bool {
        self.supports_metal3
    }

    /// Allocate a fresh GPU buffer of `byte_len` bytes with the
    /// `MTLStorageModeShared` storage mode — unified memory on Apple
    /// Silicon, so the buffer is mapped to host address space without
    /// an explicit copy. Used for parameters / activations / gradients
    /// where CPU peeks are common (e.g. inspection, serialisation).
    pub fn alloc_shared(&self, byte_len: usize) -> Result<metal::Buffer, MetalError> {
        if byte_len == 0 {
            return Err(MetalError::ShapeMismatch(
                "alloc_shared: byte_len must be > 0".to_string(),
            ));
        }
        Ok(self
            .device
            .new_buffer(byte_len as u64, MTLResourceOptions::StorageModeShared))
    }

    /// Allocate a `MTLStorageModePrivate` buffer — GPU-only, no host
    /// access without an explicit copy via blit encoder. Use for
    /// internal scratchpads / activations that never need CPU read.
    pub fn alloc_private(&self, byte_len: usize) -> Result<metal::Buffer, MetalError> {
        if byte_len == 0 {
            return Err(MetalError::ShapeMismatch(
                "alloc_private: byte_len must be > 0".to_string(),
            ));
        }
        Ok(self
            .device
            .new_buffer(byte_len as u64, MTLResourceOptions::StorageModePrivate))
    }
}

impl std::fmt::Debug for MetalBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MetalBackend")
            .field("adapter_name", &self.adapter_name)
            .field("supports_metal3", &self.supports_metal3)
            .finish()
    }
}

#[cfg(all(test, feature = "gpu-tests"))]
mod tests {
    use super::*;

    #[test]
    fn smoke_init_default_device() {
        let b = MetalBackend::new().expect("init Metal");
        assert!(!b.adapter_name().is_empty());
    }

    #[test]
    fn alloc_shared_returns_writable_buffer() {
        let b = MetalBackend::new().expect("init Metal");
        let buf = b.alloc_shared(1024).expect("alloc");
        assert!(buf.length() >= 1024);
        // SAFETY: shared-storage buffer has CPU-readable contents on
        // Apple Silicon. We read-back via the `contents()` raw pointer.
        let ptr = buf.contents() as *const u8;
        assert!(!ptr.is_null());
    }
}
