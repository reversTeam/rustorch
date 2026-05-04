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
            let _backend = metal_backend();
            let n = t.numel();
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
