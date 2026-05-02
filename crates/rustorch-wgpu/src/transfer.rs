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

/// Async variant of [`to_cpu`]. Returns a `Future` that resolves once
/// the GPU has finished writing and the staging buffer has been mapped
/// back to host memory. On native, drive completion with
/// `pollster::block_on(future)` (which is what [`to_cpu`] does
/// internally) or by calling `backend.device.poll(Wait)` from another
/// thread. On wasm, the browser drives the poll loop automatically.
///
/// This is the path used by browser demos that don't want to block
/// the main JS thread on a GPU readback.
pub async fn to_cpu_async(
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
        label: Some("rustorch-wgpu readback staging (async)"),
        size: n_bytes.max(4),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = backend
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("rustorch-wgpu readback encoder (async)"),
        });
    encoder.copy_buffer_to_buffer(&storage.buffer, 0, &staging, 0, n_bytes.max(4));
    backend.queue.submit(Some(encoder.finish()));

    let buffer_slice = staging.slice(..);
    // We need a oneshot-style channel that works in both std and wasm.
    // `std::sync::mpsc` is fine on native; on wasm32 we'd use
    // `futures::channel::oneshot`. Wrap in cfg.
    #[cfg(not(target_arch = "wasm32"))]
    let map_result = {
        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        // Drive the poll loop until the callback fires. On native, the
        // caller can run this future on its own runtime; we need at
        // least one poll to make progress.
        backend.device.poll(wgpu::Maintain::Wait);
        rx.recv()
            .map_err(|e| WgpuError::MapFailure(e.to_string()))?
    };
    #[cfg(target_arch = "wasm32")]
    let map_result = {
        // On wasm32 we use a poll-based oneshot built on a shared
        // Cell<Option<Result>>. Avoids pulling `futures-channel` into
        // the dep tree just for this one path. The browser drives
        // the wgpu poll loop on its own.
        use std::cell::RefCell;
        use std::rc::Rc;
        use std::task::{Context, Poll, Waker};
        let slot: Rc<RefCell<Option<Result<(), wgpu::BufferAsyncError>>>> =
            Rc::new(RefCell::new(None));
        let waker_cell: Rc<RefCell<Option<Waker>>> = Rc::new(RefCell::new(None));
        {
            let slot = slot.clone();
            let waker_cell = waker_cell.clone();
            buffer_slice.map_async(wgpu::MapMode::Read, move |r| {
                *slot.borrow_mut() = Some(r);
                if let Some(w) = waker_cell.borrow_mut().take() {
                    w.wake();
                }
            });
        }
        std::future::poll_fn(|cx: &mut Context<'_>| -> Poll<Result<(), WgpuError>> {
            if let Some(r) = slot.borrow_mut().take() {
                Poll::Ready(r.map_err(|e| WgpuError::MapFailure(e.to_string())))
            } else {
                *waker_cell.borrow_mut() = Some(cx.waker().clone());
                Poll::Pending
            }
        })
        .await?;
        Ok::<(), wgpu::BufferAsyncError>(())
    };
    map_result.map_err(|e| WgpuError::MapFailure(e.to_string()))?;

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

/// Read a GPU buffer back into a CPU `Tensor`. The caller supplies the
/// target `shape` (must match `storage.numel`).
///
/// Blocking — on native this drives the wgpu poll loop with
/// `pollster::block_on`. For non-blocking readback (browser demos,
/// async pipelines), call [`to_cpu_async`] directly and `.await` it
/// from your runtime.
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

    #[test]
    fn round_trip_f32_async_native() {
        // pollster::block_on drives the future on native; in the
        // browser the same code path is awaited directly from JS.
        let backend = WgpuBackend::new_blocking().expect("init wgpu");
        let cpu = Tensor::from_vec([3_usize], vec![10.0_f32, 20.0, 30.0]).unwrap();
        let gpu = to_gpu(&backend, &cpu).unwrap();
        let back =
            pollster::block_on(to_cpu_async(&backend, &gpu, vec![3])).expect("async readback");
        assert_eq!(back.as_slice::<f32>().unwrap(), &[10.0_f32, 20.0, 30.0]);
    }
}
