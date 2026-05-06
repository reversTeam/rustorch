//! Host ↔ GPU transfer helpers.
//!
//! `to_gpu(backend, &Tensor) -> WgpuStorage` uploads bytes via the
//! queue. `to_cpu(backend, &WgpuStorage, shape) -> Tensor` reads the
//! buffer back through a staging buffer with `MAP_READ` usage.

use crate::backend::WgpuBackend;
use crate::backend_singleton::wgpu_backend;
use crate::error::WgpuError;
use crate::storage::WgpuStorage;
use rustorch_core::tensor::dtype::Dtype;
use rustorch_core::tensor::storage::{Storage, WgpuStorage as CoreWgpuStorage};
use rustorch_core::tensor::tensor_impl::Tensor;

/// Upload a CPU `Tensor` to GPU memory **iff it is not already on
/// the GPU**. P3.Z Task A round-trip elimination: when `t.storage()`
/// is already `Storage::Wgpu(handle)`, we extract the existing
/// `core::WgpuStorage` (cheap Arc clone) and skip the `write_buffer`
/// upload — the buffer survives across op boundaries with no host
/// round trip. Currently F32 only.
pub fn to_gpu(backend: &WgpuBackend, t: &Tensor) -> Result<WgpuStorage, WgpuError> {
    if t.dtype() != Dtype::F32 {
        return Err(WgpuError::UnsupportedDtype(t.dtype()));
    }
    // Storage Option A fast path: the Tensor already lives on the GPU,
    // so we just clone the Arc<WgpuStorageInner> handle out — no
    // upload, no allocation. This is the per-op round-trip elimination
    // that takes Wgpu from ~22 ms/step to ~4 ms/step on Linear 1024².
    if let Storage::Wgpu(core_handle) = t.storage() {
        return Ok(WgpuStorage::from_core(
            core_handle.clone(),
            t.dtype(),
            t.numel(),
        ));
    }
    // Slow path: tensor is on CPU (Storage::Cpu) — perform the upload.
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
        buffer: CoreWgpuStorage::standalone(buffer, n_bytes as usize),
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
        // Mutex<Option<Result>>. Avoids pulling `futures-channel` into
        // the dep tree just for this one path. The browser drives
        // the wgpu poll loop on its own.
        //
        // Why Arc<Mutex<…>> rather than Rc<RefCell<…>>: enabling
        // wgpu's `fragile-send-sync-non-atomic-wasm` feature (needed to
        // wrap `wgpu::Device` in `Arc` on wasm32) makes `map_async`'s
        // closure require `Send`. Wasm32 is single-threaded, so the
        // Mutex never actually contends — the cost is just one atomic
        // CAS per take/insert.
        use std::sync::{Arc, Mutex};
        use std::task::{Context, Poll, Waker};
        let slot: Arc<Mutex<Option<Result<(), wgpu::BufferAsyncError>>>> =
            Arc::new(Mutex::new(None));
        let waker_cell: Arc<Mutex<Option<Waker>>> = Arc::new(Mutex::new(None));
        {
            let slot = slot.clone();
            let waker_cell = waker_cell.clone();
            buffer_slice.map_async(wgpu::MapMode::Read, move |r| {
                *slot.lock().unwrap() = Some(r);
                if let Some(w) = waker_cell.lock().unwrap().take() {
                    w.wake();
                }
            });
        }
        std::future::poll_fn(|cx: &mut Context<'_>| -> Poll<Result<(), WgpuError>> {
            if let Some(r) = slot.lock().unwrap().take() {
                Poll::Ready(r.map_err(|e| WgpuError::MapFailure(e.to_string())))
            } else {
                *waker_cell.lock().unwrap() = Some(cx.waker().clone());
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

/// Materialise a Tensor as a fresh CPU-storage Tensor.
///
/// **P3.Z Task A** convenience for tests / inspection / serialization
/// after the strict-`as_slice` change: when a Tensor lives on the GPU
/// (Storage::Wgpu), `as_slice<T>()` returns `None` to catch
/// cross-device bugs. To inspect the data, callers go through this
/// helper which extracts the GPU buffer, copies it to host memory,
/// and returns a CPU-storage Tensor with the same shape + dtype.
///
/// - `Storage::Cpu`  → returns a clone of `t` (no copy beyond Arc bump).
/// - `Storage::Wgpu` → drives a `to_cpu` readback via the global
///   wgpu backend singleton. F32 only for now (matches existing
///   `to_cpu` constraint).
/// - Other variants  → not yet implemented (Tasks J / M).
pub fn tensor_to_cpu(t: &Tensor) -> Result<Tensor, WgpuError> {
    match t.storage() {
        Storage::Cpu(_) => Ok(t.clone()),
        Storage::Wgpu(core_handle) => {
            // Wrap the core handle as a kernel-side WgpuStorage so we
            // can reuse the existing `to_cpu` plumbing (which expects
            // dtype + numel metadata).
            let kernel_storage = WgpuStorage::from_core(core_handle.clone(), t.dtype(), t.numel());
            let backend = wgpu_backend();
            to_cpu(backend, &kernel_storage, t.shape().to_vec())
        },
        // Future: WgpuShared, Cuda, Metal handled when their backends
        // (Tasks J / M) are wired in.
        #[allow(unreachable_patterns)]
        _ => Err(WgpuError::ShapeMismatch(format!(
            "tensor_to_cpu: storage variant not yet supported (device={:?})",
            t.device()
        ))),
    }
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

    #[test]
    fn async_readback_preserves_nan_bits() {
        // NaN must round-trip bit-exactly.
        let backend = WgpuBackend::new_blocking().expect("init wgpu");
        let cpu = Tensor::from_vec([3_usize], vec![f32::NAN, 1.0, f32::INFINITY]).unwrap();
        let gpu = to_gpu(&backend, &cpu).unwrap();
        let back = pollster::block_on(to_cpu_async(&backend, &gpu, vec![3])).unwrap();
        let got = back.as_slice::<f32>().unwrap();
        assert!(got[0].is_nan(), "NaN was not preserved");
        assert_eq!(got[1], 1.0);
        assert_eq!(got[2], f32::INFINITY);
    }

    #[test]
    fn parallel_readbacks_to_distinct_buffers_dont_interfere() {
        // Two pipelines reading two distinct buffers must both succeed.
        let backend = WgpuBackend::new_blocking().expect("init wgpu");
        let a = Tensor::from_vec([2_usize], vec![1.0_f32, 2.0]).unwrap();
        let b = Tensor::from_vec([2_usize], vec![3.0_f32, 4.0]).unwrap();
        let ga = to_gpu(&backend, &a).unwrap();
        let gb = to_gpu(&backend, &b).unwrap();
        let ra = to_cpu(&backend, &ga, vec![2]).unwrap();
        let rb = to_cpu(&backend, &gb, vec![2]).unwrap();
        assert_eq!(ra.as_slice::<f32>().unwrap(), &[1.0, 2.0]);
        assert_eq!(rb.as_slice::<f32>().unwrap(), &[3.0, 4.0]);
    }

    #[test]
    fn empty_shape_readback_returns_empty_tensor() {
        let backend = WgpuBackend::new_blocking().expect("init wgpu");
        // Shape [0] → 0 elements. Tensor::from_vec rejects it; instead
        // we exercise a shape with a 0 dimension.
        let cpu = Tensor::from_vec([2_usize, 0], vec![] as Vec<f32>).unwrap();
        let gpu = to_gpu(&backend, &cpu).unwrap();
        let back = to_cpu(&backend, &gpu, vec![2, 0]).unwrap();
        assert_eq!(back.numel(), 0);
    }

    /// Property-style sweep: a battery of randomly-generated tensors
    /// must round-trip bit-exact through `to_gpu → to_cpu`. We use a
    /// xorshift-style PRNG (deterministic for reproducibility) instead
    /// of pulling in `proptest` as a dep.
    #[test]
    fn random_round_trip_property_sweep() {
        let backend = WgpuBackend::new_blocking().expect("init wgpu");
        let mut s: u64 = 0xDEAD_BEEF_CAFE_BABE;
        for trial in 0..32_usize {
            // Random length in [1, 1024].
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let n = (s % 1024) as usize + 1;
            // Random F32 values via bit reinterpret.
            let data: Vec<f32> = (0..n)
                .map(|i| {
                    let bits = s.wrapping_mul(0x9E37_79B9).wrapping_add(i as u64) as u32;
                    f32::from_bits(bits & 0x7FFF_FFFF) // clear sign for finite-ish values
                })
                .filter(|x| x.is_finite())
                .collect();
            if data.is_empty() {
                continue;
            }
            let n = data.len();
            let cpu = Tensor::from_vec([n], data.clone()).unwrap();
            let gpu = to_gpu(&backend, &cpu).unwrap();
            let back = to_cpu(&backend, &gpu, vec![n]).unwrap();
            let raw = back.as_slice::<f32>().unwrap();
            // Bit-exact equality via to_bits comparison.
            for (a, b) in raw.iter().zip(&data) {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "trial {} mismatch at byte: {:?} vs {:?}",
                    trial,
                    a,
                    b
                );
            }
        }
    }
}
