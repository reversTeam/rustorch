//! Host ↔ Metal transfer helpers (P3.Z Task J).

use crate::backend_singleton::metal_backend;
use crate::error::MetalError;
use rustorch_core::tensor::storage::Storage;
use rustorch_core::tensor::tensor_impl::Tensor;

/// Materialise a Tensor as a fresh CPU-storage Tensor.
///
/// `Storage::Metal` → reads via `metal::Buffer::contents()` (unified
/// memory, no blit needed on Apple Silicon).
/// `Storage::Cpu`   → returns a clone of `t`.
/// Other variants  → error (CUDA / Wgpu live elsewhere).
pub fn tensor_to_cpu(t: &Tensor) -> Result<Tensor, MetalError> {
    match t.storage() {
        Storage::Cpu(_) => Ok(t.clone()),
        Storage::Metal(metal_storage) => {
            let backend = metal_backend();
            let n = t.numel();
            // Drain the **pending shared command buffer**: every kernel
            // dispatch built up via `with_encoder` is queued on a single
            // command buffer that is only committed when explicitly
            // drained. Without this commit the host read sees whatever
            // was in the buffer before the kernel write (typically 0
            // for a freshly-allocated MTLStorageModeShared buffer or
            // stale contents for a pool-recycled one).
            //
            // Earlier this routine submitted an empty command buffer to
            // the queue and waited — that flushed the *queue* but
            // never committed the pending buffer holding the kernel
            // work, so reads of any tensor whose producing kernel was
            // still pending returned garbage.
            backend.drain();
            // SAFETY: shared-storage buffer; `contents()` is host-mapped.
            let data: Vec<f32> = unsafe {
                let p = metal_storage.contents() as *const f32;
                std::slice::from_raw_parts(p, n).to_vec()
            };
            Tensor::from_vec(t.shape().to_vec(), data)
                .map_err(|e| MetalError::ShapeMismatch(format!("tensor_to_cpu: {e}")))
        },
        _ => Err(MetalError::ShapeMismatch(format!(
            "tensor_to_cpu: storage variant not supported by rustorch-metal (device={:?})",
            t.device()
        ))),
    }
}
