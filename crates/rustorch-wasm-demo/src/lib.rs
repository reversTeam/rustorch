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

pub mod gpt_block;
pub mod model_loader;
pub mod resnet_block;
pub mod transformer_block;

#[cfg(target_arch = "wasm32")]
mod web {
    use rustorch_core::tensor::dtype::Dtype;
    use rustorch_core::tensor::tensor_impl::Tensor;
    use rustorch_wgpu::{
        attention_naive, dispatch_binary, dispatch_unary, layernorm_rows, matmul, softmax_rows,
        to_cpu, to_gpu, WgpuBackend,
    };
    use wasm_bindgen::prelude::*;

    /// Install a panic hook that prints to the JS console.
    #[wasm_bindgen(start)]
    pub fn boot() {
        console_error_panic_hook::set_once();
    }

    /// Performance report from one demo run. Returned to JS as a
    /// plain object so the page can render a small benchmark table.
    #[wasm_bindgen]
    pub struct DemoStats {
        upload_ms: f64,
        compute_ms: f64,
        readback_ms: f64,
        total_ms: f64,
        first_eight: js_sys::Float32Array,
    }

    #[wasm_bindgen]
    impl DemoStats {
        /// Time spent uploading the input tensors to the GPU.
        #[wasm_bindgen(getter)]
        pub fn upload_ms(&self) -> f64 {
            self.upload_ms
        }
        /// Time spent inside the GPU kernel chain.
        #[wasm_bindgen(getter)]
        pub fn compute_ms(&self) -> f64 {
            self.compute_ms
        }
        /// Time spent mapping the output buffer back to host memory.
        #[wasm_bindgen(getter)]
        pub fn readback_ms(&self) -> f64 {
            self.readback_ms
        }
        /// Sum of the three phases above.
        #[wasm_bindgen(getter)]
        pub fn total_ms(&self) -> f64 {
            self.total_ms
        }
        /// First eight values of the final output for visual inspection.
        #[wasm_bindgen(getter)]
        pub fn first_eight(&self) -> js_sys::Float32Array {
            self.first_eight.clone()
        }
    }

    /// Read `performance.now()` from the `Window` object — the only
    /// reasonably-precise wall clock available in browser WASM.
    fn now_ms() -> f64 {
        js_sys::Reflect::get(&js_sys::global(), &"performance".into())
            .ok()
            .and_then(|p| js_sys::Reflect::get(&p, &"now".into()).ok())
            .and_then(|f| f.dyn_into::<js_sys::Function>().ok())
            .and_then(|f| f.call0(&js_sys::global()).ok())
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0)
    }

    /// The original tiny demo: `add → relu → softmax` on `[B, K]` input.
    /// Kept for backwards compatibility with the headless smoke test.
    #[wasm_bindgen]
    pub async fn run_demo(b: u32, k: u32) -> Result<js_sys::Float32Array, JsValue> {
        let b = b as usize;
        let k = k as usize;
        let backend = WgpuBackend::new()
            .await
            .map_err(|e| JsValue::from_str(&format!("backend init: {e}")))?;
        let lhs: Vec<f32> = (0..(b * k)).map(|i| (i as f32) * 0.01).collect();
        let rhs: Vec<f32> = (0..(b * k)).map(|i| (i as f32).sin()).collect();
        let tlhs = Tensor::from_vec([b, k], lhs)
            .map_err(|e| JsValue::from_str(&format!("lhs tensor: {e}")))?;
        let trhs = Tensor::from_vec([b, k], rhs)
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
        let _ = Dtype::F32;
        Ok(js_sys::Float32Array::from(slice))
    }

    /// Larger demo pipeline that exercises the full P2 kernel surface
    /// in a transformer-style block:
    ///
    ///   x ∈ [S, D]  --(matmul Wq, Wk, Wv)-->  Q, K, V ∈ [S, D]
    ///   --(attention_naive)-->  attn_out ∈ [S, D]
    ///   --(layernorm)-->         normed ∈ [S, D]
    ///
    /// `S` (sequence length) and `D` (embedding dim) are passed from
    /// JS so the demo page can sweep across sizes for a benchmark.
    /// Returns timing breakdown + the first 8 output values.
    #[wasm_bindgen]
    pub async fn run_attention_block(s: u32, d: u32) -> Result<DemoStats, JsValue> {
        let s = s as usize;
        let d = d as usize;
        let backend = WgpuBackend::new()
            .await
            .map_err(|e| JsValue::from_str(&format!("backend init: {e}")))?;

        // Synthetic inputs — a real demo would feed pre-tokenized text.
        // `mk(seed, n)` generates `n` deterministic f32 values in [-0.5, 0.5].
        let mk = |seed: u64, n: usize| -> Vec<f32> {
            (0..n)
                .map(|i| {
                    let bits = seed
                        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                        .wrapping_add(i as u64);
                    let unit = ((bits >> 33) as u32 as f32) / (u32::MAX as f32);
                    unit - 0.5
                })
                .collect()
        };
        // `x` has shape [S, D] (S*D elements); the projection matrices
        // have shape [D, D] (D*D elements each).
        let x_data = mk(0xA1, s * d);
        let wq_data = mk(0xB2, d * d);
        let wk_data = mk(0xC3, d * d);
        let wv_data = mk(0xD4, d * d);
        let gamma_data = vec![1.0_f32; d];
        let beta_data = vec![0.0_f32; d];

        // ---- Upload phase ------------------------------------------------
        let t_up = now_ms();
        let tx = Tensor::from_vec([s, d], x_data)
            .map_err(|e| JsValue::from_str(&format!("x tensor: {e}")))?;
        let twq = Tensor::from_vec([d, d], wq_data)
            .map_err(|e| JsValue::from_str(&format!("Wq tensor: {e}")))?;
        let twk = Tensor::from_vec([d, d], wk_data)
            .map_err(|e| JsValue::from_str(&format!("Wk tensor: {e}")))?;
        let twv = Tensor::from_vec([d, d], wv_data)
            .map_err(|e| JsValue::from_str(&format!("Wv tensor: {e}")))?;
        let tg = Tensor::from_vec([d], gamma_data)
            .map_err(|e| JsValue::from_str(&format!("gamma tensor: {e}")))?;
        let tbb = Tensor::from_vec([d], beta_data)
            .map_err(|e| JsValue::from_str(&format!("beta tensor: {e}")))?;
        let gx = to_gpu(&backend, &tx).map_err(|e| JsValue::from_str(&format!("upload x: {e}")))?;
        let gwq =
            to_gpu(&backend, &twq).map_err(|e| JsValue::from_str(&format!("upload Wq: {e}")))?;
        let gwk =
            to_gpu(&backend, &twk).map_err(|e| JsValue::from_str(&format!("upload Wk: {e}")))?;
        let gwv =
            to_gpu(&backend, &twv).map_err(|e| JsValue::from_str(&format!("upload Wv: {e}")))?;
        let ggamma =
            to_gpu(&backend, &tg).map_err(|e| JsValue::from_str(&format!("upload γ: {e}")))?;
        let gbeta =
            to_gpu(&backend, &tbb).map_err(|e| JsValue::from_str(&format!("upload β: {e}")))?;
        let upload_ms = now_ms() - t_up;

        // ---- Compute phase -----------------------------------------------
        let t_c = now_ms();
        let q = matmul(&backend, &gx, &gwq, s, d, d)
            .map_err(|e| JsValue::from_str(&format!("Q = X@Wq: {e}")))?;
        let k = matmul(&backend, &gx, &gwk, s, d, d)
            .map_err(|e| JsValue::from_str(&format!("K = X@Wk: {e}")))?;
        let v = matmul(&backend, &gx, &gwv, s, d, d)
            .map_err(|e| JsValue::from_str(&format!("V = X@Wv: {e}")))?;
        let attn = attention_naive(&backend, &q, &k, &v, s, d)
            .map_err(|e| JsValue::from_str(&format!("attention: {e}")))?;
        let normed = layernorm_rows(&backend, &attn, &ggamma, &gbeta, s, d, 1e-5)
            .map_err(|e| JsValue::from_str(&format!("layernorm: {e}")))?;
        let compute_ms = now_ms() - t_c;

        // ---- Readback phase ----------------------------------------------
        let t_r = now_ms();
        let out = to_cpu(&backend, &normed, vec![s, d])
            .map_err(|e| JsValue::from_str(&format!("readback: {e}")))?;
        let readback_ms = now_ms() - t_r;
        let total_ms = upload_ms + compute_ms + readback_ms;

        let slice = out
            .as_slice::<f32>()
            .ok_or_else(|| JsValue::from_str("output not f32"))?;
        let head_len = slice.len().min(8);
        Ok(DemoStats {
            upload_ms,
            compute_ms,
            readback_ms,
            total_ms,
            first_eight: js_sys::Float32Array::from(&slice[..head_len]),
        })
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
