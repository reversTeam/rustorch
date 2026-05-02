//! Host ↔ GPU transfer helpers.
//!
//! `to_gpu(backend, &Tensor) -> WgpuStorage` uploads bytes via the
//! queue. `to_cpu(backend, &WgpuStorage, shape) -> Tensor` reads the
//! buffer back through a staging buffer with `MAP_READ` usage.

use crate::backend::WgpuBackend;
use crate::error::WgpuError;
use crate::storage::WgpuStorage;
use rustorch_core::tensor::dtype::Dtype;
use rustorch_core::tensor::tensor_impl::Tensor;
use std::sync::Arc;

/// Upload a CPU `Tensor` to GPU memory. Currently F32 only.
pub fn to_gpu(backend: &WgpuBackend, t: &Tensor) -> Result<WgpuStorage, WgpuError> {
    if t.dtype() != Dtype::F32 {
        return Err(WgpuError::UnsupportedDtype(t.dtype()));
    }
    let data = t
        .as_slice::<f32>()
        .ok_or_else(|| WgpuError::ShapeMismatch("tensor must be contiguous F32".to_string()))?;
    let n_bytes = (data.len() * 4) as u64;
    // Allocate a buffer with COPY_DST | STORAGE | COPY_SRC so it can both
    // receive uploads and serve as a kernel input.
    let buffer = backend.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("rustorch-wgpu upload buffer"),
        size: n_bytes.max(4),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    backend
        .queue
        .write_buffer(&buffer, 0, bytemuck::cast_slice(data));
    backend
        .queue
        .submit(std::iter::empty::<wgpu::CommandBuffer>());
    Ok(WgpuStorage {
        buffer: Arc::new(crate::pooled::PooledBuffer::standalone(buffer)),
        dtype: Dtype::F32,
        numel: data.len(),
    })
}

/// Read a GPU buffer back into a CPU `Tensor`. The caller supplies the
/// target `shape` (must match `storage.numel`).
pub fn to_cpu(
    backend: &WgpuBackend,
    storage: &WgpuStorage,
    shape: Vec<usize>,
) -> Result<Tensor, WgpuError> {
    if storage.dtype != Dtype::F32 {
        return Err(WgpuError::UnsupportedDtype(storage.dtype));
    }
    let numel: usize = shape.iter().product();
    if numel != storage.numel {
        return Err(WgpuError::ShapeMismatch(format!(
            "shape numel {numel} != storage numel {}",
            storage.numel
        )));
    }
    let n_bytes = (numel * 4) as u64;
    let staging = backend.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("rustorch-wgpu readback staging"),
        size: n_bytes.max(4),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = backend
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("rustorch-wgpu readback encoder"),
        });
    encoder.copy_buffer_to_buffer(&storage.buffer, 0, &staging, 0, n_bytes.max(4));
    backend.queue.submit(Some(encoder.finish()));

    let buffer_slice = staging.slice(..);
    let (sender, receiver) = std::sync::mpsc::channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    backend.device.poll(wgpu::Maintain::Wait);
    receiver
        .recv()
        .map_err(|e| WgpuError::MapFailure(e.to_string()))?
        .map_err(|e| WgpuError::MapFailure(e.to_string()))?;

    let data = buffer_slice.get_mapped_range();
    let bytes: &[u8] = &data[..n_bytes as usize];
    let v: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    drop(data);
    staging.unmap();

    Tensor::from_vec(shape, v).map_err(|e| WgpuError::ShapeMismatch(format!("{e}")))
}

#[cfg(all(test, feature = "gpu-tests"))]
mod tests {
    use super::*;

    #[test]
    fn round_trip_f32_vector() {
        let backend = WgpuBackend::new_blocking().expect("init wgpu");
        let cpu = Tensor::from_vec([4_usize], vec![1.0_f32, 2.0, 3.0, 4.0]).unwrap();
        let gpu = to_gpu(&backend, &cpu).unwrap();
        let back = to_cpu(&backend, &gpu, vec![4]).unwrap();
        assert_eq!(back.as_slice::<f32>().unwrap(), &[1.0_f32, 2.0, 3.0, 4.0]);
    }
}
