//! Browser-side safetensors model loader.
//!
//! Takes a byte slice (typically from a `fetch().arrayBuffer()` call
//! in JS), parses it as a HuggingFace safetensors archive, and
//! uploads every tensor to the GPU backend. Returns a map keyed by
//! tensor name so model code can look up `"weight"`, `"bias"`, etc.
//!
//! Designed to be JS-friendly: every error path returns a `JsValue`
//! with a descriptive message rather than panicking.

use rustorch_core::tensor::tensor_impl::Tensor;
use rustorch_wgpu::{to_gpu, WgpuBackend, WgpuError, WgpuStorage};
use std::collections::BTreeMap;

/// Result of parsing + uploading a safetensors archive.
pub struct LoadedModel {
    /// Tensor name → GPU storage. Order matches the on-disk layout
    /// (BTreeMap's lexicographic key order).
    pub tensors: BTreeMap<String, WgpuStorage>,
}

impl LoadedModel {
    /// Look up a tensor by name. Returns `None` if absent.
    pub fn get(&self, name: &str) -> Option<&WgpuStorage> {
        self.tensors.get(name)
    }

    /// Number of distinct tensors in the archive.
    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    /// Empty archive?
    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }
}

/// Errors specific to the model loader.
#[derive(Debug)]
pub enum LoadError {
    /// Empty payload — caller should check the `Content-Length`
    /// header before passing bytes here.
    Empty,
    /// Safetensors parse error (malformed header, bad JSON, etc.).
    Parse(String),
    /// GPU upload error.
    Gpu(WgpuError),
}

impl core::fmt::Display for LoadError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            LoadError::Empty => write!(f, "empty safetensors archive"),
            LoadError::Parse(e) => write!(f, "safetensors parse error: {e}"),
            LoadError::Gpu(e) => write!(f, "GPU upload error: {e}"),
        }
    }
}

impl std::error::Error for LoadError {}

/// Parse `data` as a safetensors archive and upload every tensor to
/// the supplied GPU backend.
///
/// Reports progress via the optional callback (`(loaded, total)` in
/// bytes-of-source-payload units, called after each tensor upload).
/// Pass `None` for headless callers; the demo HTML wires a
/// `progress` event listener on the JS side.
pub fn load_safetensors(
    backend: &WgpuBackend,
    data: &[u8],
    mut on_progress: Option<&mut dyn FnMut(u64, u64)>,
) -> Result<LoadedModel, LoadError> {
    if data.is_empty() {
        return Err(LoadError::Empty);
    }
    // Parse via the existing rustorch-serde safetensors reader.
    let mut cursor = std::io::Cursor::new(data);
    let cpu_tensors: BTreeMap<String, Tensor> = rustorch_serde::safetensors::read_from(&mut cursor)
        .map_err(|e| LoadError::Parse(format!("{e:?}")))?;

    if cpu_tensors.is_empty() {
        return Err(LoadError::Empty);
    }

    let total = data.len() as u64;
    let mut loaded: u64 = 0;
    let mut gpu_tensors: BTreeMap<String, WgpuStorage> = BTreeMap::new();
    for (name, t) in cpu_tensors {
        // Approximate per-tensor progress as `numel * dtype.byte_size`.
        // Conservative — doesn't include the safetensors header — so
        // the final value may slightly exceed `total`, but we clamp.
        let bytes = t.numel() * t.dtype().byte_size();
        loaded += bytes as u64;
        let gpu = to_gpu(backend, &t).map_err(LoadError::Gpu)?;
        gpu_tensors.insert(name, gpu);
        if let Some(cb) = on_progress.as_mut() {
            cb(loaded.min(total), total);
        }
    }
    Ok(LoadedModel {
        tensors: gpu_tensors,
    })
}

#[cfg(target_arch = "wasm32")]
mod web {
    use super::*;
    use wasm_bindgen::prelude::*;

    /// JS-callable wrapper. Takes a `Uint8Array` containing the
    /// safetensors archive bytes, initialises a `WgpuBackend`, and
    /// returns a JS `Map<string, Float32Array>` with each tensor's
    /// **CPU-side** values for inspection (the GPU storage cannot be
    /// directly handed back to JS).
    ///
    /// Heavy: large models trigger a host round-trip per tensor. For
    /// production demos, prefer keeping tensors on the GPU and only
    /// reading back the final output.
    #[wasm_bindgen]
    pub async fn load_safetensors_to_js(data: &[u8]) -> Result<js_sys::Map, JsValue> {
        let backend = WgpuBackend::new()
            .await
            .map_err(|e| JsValue::from_str(&format!("backend init: {e}")))?;
        let loaded = load_safetensors(&backend, data, None)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        let map = js_sys::Map::new();
        for (name, storage) in &loaded.tensors {
            // Read back to CPU as f32 for JS inspection.
            // Only F32 tensors are exposed; others are skipped.
            if storage.dtype != rustorch_core::tensor::dtype::Dtype::F32 {
                continue;
            }
            let shape: Vec<usize> = vec![storage.numel];
            let cpu = rustorch_wgpu::to_cpu(&backend, storage, shape)
                .map_err(|e| JsValue::from_str(&format!("readback: {e}")))?;
            let slice = cpu
                .as_slice::<f32>()
                .ok_or_else(|| JsValue::from_str("expected f32"))?;
            map.set(
                &JsValue::from_str(name),
                &js_sys::Float32Array::from(slice).into(),
            );
        }
        Ok(map)
    }
}

#[cfg(target_arch = "wasm32")]
pub use web::*;

#[cfg(all(test, not(target_arch = "wasm32"), feature = "gpu-tests"))]
mod gpu_tests {
    use super::*;
    use rustorch_serde::safetensors::write_to;
    use rustorch_wgpu::to_cpu;

    fn build_archive(tensors: &[(&str, Tensor)]) -> Vec<u8> {
        // Serialize to in-memory safetensors bytes.
        let map: BTreeMap<String, Tensor> = tensors
            .iter()
            .map(|(n, t)| (n.to_string(), t.clone()))
            .collect();
        let mut buf: Vec<u8> = Vec::new();
        write_to(&mut buf, &map).unwrap();
        buf
    }

    #[test]
    fn empty_archive_errors() {
        let backend = WgpuBackend::new_blocking().expect("init");
        let err = match load_safetensors(&backend, &[], None) {
            Ok(_) => panic!("expected Empty"),
            Err(e) => e,
        };
        assert!(matches!(err, LoadError::Empty), "got {err}");
    }

    #[test]
    fn round_trip_small_archive() {
        let backend = WgpuBackend::new_blocking().expect("init");
        let weight =
            Tensor::from_vec([2_usize, 3], vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let bias = Tensor::from_vec([3_usize], vec![0.1_f32, 0.2, 0.3]).unwrap();
        let bytes = build_archive(&[("weight", weight.clone()), ("bias", bias.clone())]);
        let loaded = load_safetensors(&backend, &bytes, None).unwrap();
        assert_eq!(loaded.len(), 2);

        // Read back the weight tensor and check element-wise.
        let w_gpu = loaded.get("weight").expect("weight present");
        let w_back = to_cpu(&backend, w_gpu, vec![6]).unwrap();
        assert_eq!(
            w_back.as_slice::<f32>().unwrap(),
            &[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]
        );

        let b_gpu = loaded.get("bias").expect("bias present");
        let b_back = to_cpu(&backend, b_gpu, vec![3]).unwrap();
        assert_eq!(b_back.as_slice::<f32>().unwrap(), &[0.1_f32, 0.2, 0.3]);
    }

    #[test]
    fn progress_callback_fires_per_tensor() {
        let backend = WgpuBackend::new_blocking().expect("init");
        let a = Tensor::from_vec([4_usize], vec![1.0_f32, 2.0, 3.0, 4.0]).unwrap();
        let b = Tensor::from_vec([4_usize], vec![5.0_f32, 6.0, 7.0, 8.0]).unwrap();
        let c = Tensor::from_vec([4_usize], vec![9.0_f32, 10.0, 11.0, 12.0]).unwrap();
        let bytes = build_archive(&[("a", a), ("b", b), ("c", c)]);
        let mut events: Vec<(u64, u64)> = Vec::new();
        load_safetensors(
            &backend,
            &bytes,
            Some(&mut |loaded, total| events.push((loaded, total))),
        )
        .unwrap();
        // 3 tensors → 3 progress events.
        assert_eq!(events.len(), 3);
        // Loaded must be monotonically increasing and the final
        // value must be ≤ total.
        for w in events.windows(2) {
            assert!(w[0].0 <= w[1].0);
        }
        assert!(events.last().unwrap().0 <= events.last().unwrap().1);
    }

    #[test]
    fn malformed_archive_rejects_with_parse_error() {
        let backend = WgpuBackend::new_blocking().expect("init");
        // 8 bytes that cannot possibly form a valid safetensors header.
        let bytes = vec![0xFF_u8; 8];
        let err = match load_safetensors(&backend, &bytes, None) {
            Ok(_) => panic!("expected parse error"),
            Err(e) => e,
        };
        assert!(matches!(err, LoadError::Parse(_)), "got {err}");
    }
}
