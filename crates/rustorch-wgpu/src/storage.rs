//! `WgpuStorage` — refcounted handle to a `wgpu::Buffer`.

use crate::error::WgpuError;
use rustorch_core::tensor::dtype::Dtype;
use std::sync::Arc;

/// A refcounted GPU buffer + dtype. The buffer is shared via `Arc`
/// so that views/slices can clone cheaply.
#[derive(Clone)]
pub struct WgpuStorage {
    /// Underlying wgpu buffer.
    pub buffer: Arc<wgpu::Buffer>,
    /// Element dtype (kept here because wgpu::Buffer is just bytes).
    pub dtype: Dtype,
    /// Number of elements (`bytes / dtype.byte_size()`).
    pub numel: usize,
}

impl WgpuStorage {
    /// Allocate a fresh GPU buffer of the requested `numel` elements
    /// of `dtype`. Buffer is created with STORAGE | COPY_SRC | COPY_DST
    /// usage so it can participate in compute shaders + transfers.
    pub fn allocate(device: &wgpu::Device, numel: usize, dtype: Dtype) -> Result<Self, WgpuError> {
        let n_bytes = numel * dtype.byte_size();
        // wgpu requires buffers to be at least 1 byte; clamp.
        let size = (n_bytes as u64).max(4);
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rustorch-wgpu storage"),
            size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Ok(WgpuStorage {
            buffer: Arc::new(buffer),
            dtype,
            numel,
        })
    }

    /// Total byte size of the underlying buffer (`numel * dtype.byte_size()`,
    /// possibly padded to wgpu minimum).
    pub fn byte_size(&self) -> usize {
        self.numel * self.dtype.byte_size()
    }
}
