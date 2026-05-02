//! Direct large-payload upload via `mapped_at_creation`.
//!
//! For payloads that exceed the [`crate::staging::StagingRing`] slot
//! size, `queue.write_buffer` performs an internal copy. Allocating
//! the destination buffer with `mapped_at_creation = true` bypasses
//! that copy entirely: the GPU driver maps the buffer directly into
//! host-visible memory at allocation time, the host writes the
//! payload, and `unmap()` flushes the bytes — no intermediate
//! staging buffer at all. Used for weight loading and one-shot big
//! tensors.
//!
//! Trade-off: forces the buffer to be **freshly allocated** (no
//! pool reuse) and requires the size to be known up-front.

use crate::error::WgpuError;
use crate::pooled::PooledBuffer;
use crate::storage::WgpuStorage;
use rustorch_core::tensor::dtype::Dtype;
use std::sync::Arc;

/// Allocate a `WgpuStorage` of `bytes` size and write `data` into it
/// in one shot via `mapped_at_creation`. The buffer is created with
/// `STORAGE | COPY_SRC | COPY_DST` usage so it can participate in
/// kernels and transfers immediately.
///
/// Returns `Err(ShapeMismatch)` if `data.len() != numel * dtype.byte_size()`.
/// The buffer is wrapped as a [`PooledBuffer::standalone`] — it will
/// not return to any pool on drop (`mapped_at_creation` is incompatible
/// with the size-bin pool because the destination size is exact, not
/// bucketed).
pub fn upload_mapped_at_creation(
    device: &wgpu::Device,
    numel: usize,
    dtype: Dtype,
    data: &[u8],
) -> Result<WgpuStorage, WgpuError> {
    let expected_bytes = numel * dtype.byte_size();
    if data.len() != expected_bytes {
        return Err(WgpuError::ShapeMismatch(format!(
            "upload_mapped_at_creation: data {} bytes != numel*dtype {} bytes",
            data.len(),
            expected_bytes
        )));
    }
    if numel == 0 {
        return Err(WgpuError::ShapeMismatch(
            "upload_mapped_at_creation: cannot map empty buffer".to_string(),
        ));
    }
    // Buffer sizes must be 4-byte aligned for some platforms (Metal/DX12).
    let size = (data.len() as u64).max(4);
    if size % 4 != 0 {
        return Err(WgpuError::ShapeMismatch(format!(
            "upload_mapped_at_creation: size {} must be 4-byte aligned",
            size
        )));
    }
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("rustorch-wgpu mapped-at-creation upload"),
        size,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: true,
    });
    // Slice the mapped range, copy, unmap.
    {
        let mut view = buffer.slice(..size).get_mapped_range_mut();
        view[..data.len()].copy_from_slice(data);
        // If size > data.len() because of the 4-byte clamp, the tail
        // is left uninitialized — irrelevant because the kernel only
        // reads `numel` elements.
    }
    buffer.unmap();
    Ok(WgpuStorage {
        buffer: Arc::new(PooledBuffer::standalone(buffer)),
        dtype,
        numel,
    })
}

#[cfg(all(test, feature = "gpu-tests"))]
mod gpu_tests {
    use super::*;
    use crate::backend::WgpuBackend;
    use crate::transfer::to_cpu;

    #[test]
    fn mapped_round_trip_f32_vector() {
        let backend = WgpuBackend::new_blocking().expect("init");
        let data: Vec<f32> = (0..128).map(|i| (i as f32) * 0.5 - 1.0).collect();
        let bytes: &[u8] = bytemuck::cast_slice(&data);
        let storage =
            upload_mapped_at_creation(&backend.device, data.len(), Dtype::F32, bytes).unwrap();
        let t = to_cpu(&backend, &storage, vec![data.len()]).unwrap();
        assert_eq!(t.as_slice::<f32>().unwrap(), data.as_slice());
    }

    #[test]
    fn mapped_rejects_size_mismatch() {
        let backend = WgpuBackend::new_blocking().expect("init");
        // numel=10 → expects 40 bytes but we pass only 32.
        let bytes = vec![0_u8; 32];
        let err = match upload_mapped_at_creation(&backend.device, 10, Dtype::F32, &bytes) {
            Ok(_) => panic!("should have rejected size mismatch"),
            Err(e) => e,
        };
        assert!(format!("{err}").contains("32 bytes"));
    }

    #[test]
    fn mapped_rejects_zero_numel() {
        let backend = WgpuBackend::new_blocking().expect("init");
        let err = match upload_mapped_at_creation(&backend.device, 0, Dtype::F32, &[]) {
            Ok(_) => panic!("should have rejected empty"),
            Err(e) => e,
        };
        assert!(format!("{err}").contains("empty"));
    }

    #[test]
    fn mapped_preserves_nan_bits() {
        let backend = WgpuBackend::new_blocking().expect("init");
        // NaN bit-pattern must survive the round-trip.
        let data = vec![f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 0.0];
        let bytes: &[u8] = bytemuck::cast_slice(&data);
        let storage =
            upload_mapped_at_creation(&backend.device, data.len(), Dtype::F32, bytes).unwrap();
        let t = to_cpu(&backend, &storage, vec![data.len()]).unwrap();
        let got = t.as_slice::<f32>().unwrap();
        assert!(got[0].is_nan());
        assert_eq!(got[1], f32::INFINITY);
        assert_eq!(got[2], f32::NEG_INFINITY);
        assert_eq!(got[3], 0.0);
    }
}
