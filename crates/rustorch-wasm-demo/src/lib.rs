//! # rustorch-wasm-demo
//!
//! Tiny browser-WebGPU demo of rustorch. Exposes a single
//! `run_demo()` entry-point callable from JS that:
//!   1. Initialises a `WgpuBackend` against the browser WebGPU adapter.
//!   2. Uploads two small F32 tensors.
//!   3. Runs `add` then `relu` then `softmax` on the GPU.
//!   4. Returns the final values back as a `Float32Array`.
//!
//! This crate is intentionally minimal — its purpose is to prove the
//! end-to-end pipeline works in Chrome 120+ with WebGPU enabled. Real
//! demos (ResNet-18 inference, GPT-2 token streaming) live behind
//! feature flags in follow-up slices.

#![cfg_attr(docsrs, feature(doc_cfg))]
#![warn(missing_docs)]
#![warn(rust_2018_idioms)]

#[cfg(target_arch = "wasm32")]
mod web {
    use rustorch_core::tensor::dtype::Dtype;
    use rustorch_wgpu::{
        dispatch_binary, dispatch_unary, softmax_rows, to_cpu, to_gpu, WgpuBackend,
    };
    use wasm_bindgen::prelude::*;

    /// Install a panic hook that prints to the JS console.
    #[wasm_bindgen(start)]
    pub fn boot() {
        console_error_panic_hook::set_once();
    }

    /// Run the demo pipeline. Returns `[B, K]` softmax probabilities as
    /// a `Float32Array`. Resolves the JS Promise with the final buffer.
    #[wasm_bindgen]
    pub async fn run_demo(b: u32, k: u32) -> Result<js_sys::Float32Array, JsValue> {
        let b = b as usize;
        let k = k as usize;
        let backend = WgpuBackend::new()
            .await
            .map_err(|e| JsValue::from_str(&format!("backend init: {e}")))?;
        let lhs: Vec<f32> = (0..(b * k)).map(|i| (i as f32) * 0.01).collect();
        let rhs: Vec<f32> = (0..(b * k)).map(|i| (i as f32).sin()).collect();
        let tlhs = rustorch_core::tensor::tensor_impl::Tensor::from_vec([b, k], lhs)
            .map_err(|e| JsValue::from_str(&format!("lhs tensor: {e}")))?;
        let trhs = rustorch_core::tensor::tensor_impl::Tensor::from_vec([b, k], rhs)
            .map_err(|e| JsValue::from_str(&format!("rhs tensor: {e}")))?;
        let glhs =
            to_gpu(&backend, &tlhs).map_err(|e| JsValue::from_str(&format!("upload lhs: {e}")))?;
        let grhs =
            to_gpu(&backend, &trhs).map_err(|e| JsValue::from_str(&format!("upload rhs: {e}")))?;

        let summed = dispatch_binary(&backend, "add", &glhs, &grhs)
            .map_err(|e| JsValue::from_str(&format!("add: {e}")))?;
        let activated = dispatch_unary(&backend, "relu", &summed)
            .map_err(|e| JsValue::from_str(&format!("relu: {e}")))?;
        let probs = softmax_rows(&backend, &activated, b, k)
            .map_err(|e| JsValue::from_str(&format!("softmax: {e}")))?;

        let out = to_cpu(&backend, &probs, vec![b, k])
            .map_err(|e| JsValue::from_str(&format!("readback: {e}")))?;
        let slice = out
            .as_slice::<f32>()
            .ok_or_else(|| JsValue::from_str("output not f32"))?;
        // SAFETY: Float32Array::from copies the data; ownership remains in Rust.
        let _ = Dtype::F32; // suppress unused-import warning when feature off
        Ok(js_sys::Float32Array::from(slice))
    }
}

#[cfg(target_arch = "wasm32")]
pub use web::*;

/// Native build (no WebGPU in browser): expose a pure-Rust function so
/// `cargo build` / `cargo test` keep working on Linux/macOS/Windows.
#[cfg(not(target_arch = "wasm32"))]
pub mod native {
    /// Pure-Rust placeholder — the actual demo runs in the browser.
    pub fn run_demo_native(b: usize, k: usize) -> Vec<f32> {
        // Trivial CPU softmax of zeros so unit tests have something to
        // verify without requiring a GPU adapter.
        vec![1.0_f32 / k as f32; b * k]
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::native::run_demo_native;

    #[test]
    fn native_softmax_uniform_distribution() {
        let v = run_demo_native(2, 4);
        for x in &v {
            assert!((x - 0.25).abs() < 1e-6);
        }
        assert_eq!(v.len(), 8);
    }
}
