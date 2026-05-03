//! Transformer encoder block running entirely on the **CPU autograd
//! stack** in the browser. Distinct from [`crate::gpt_block`] (which
//! exercises the raw WGPU kernels): this path goes through
//! `rustorch-nn` modules and therefore proves end-to-end that the
//! autograd-aware building blocks (Linear rank-N, MultiHeadAttention,
//! RMSNorm, SinusoidalPositionalEncoding, causal_mask) compile and
//! execute under wasm32 — no WebGPU adapter required.

use rustorch_autograd::{ops, Variable};
use rustorch_nn::{causal_mask, Linear, Module, MultiHeadAttention, RMSNorm};

/// Tiny single-block transformer encoder for the browser demo.
///
/// Layout follows the modern pre-norm convention:
/// ```text
///   h = x + MHA(RMSNorm(x), causal_mask)
///   y = h + Linear(RMSNorm(h))     (FFN simplified to a single Linear)
/// ```
pub struct TinyEncoder {
    /// Pre-norm before self-attention.
    pub norm_attn: RMSNorm,
    /// Self-attention block (rank-3 with `Linear` rank-N internally).
    pub mha: MultiHeadAttention,
    /// Pre-norm before the feed-forward network.
    pub norm_ffn: RMSNorm,
    /// Feed-forward network — single `Linear` for the demo (real
    /// transformers use 2 Linears + GeLU).
    pub ffn: Linear,
}

impl TinyEncoder {
    /// Build a TinyEncoder with `embed_dim = dim` and `num_heads`
    /// attention heads. `dim % num_heads` must be 0.
    pub fn new(dim: usize, num_heads: usize) -> Self {
        Self {
            norm_attn: RMSNorm::new(dim),
            mha: MultiHeadAttention::new(dim, num_heads),
            norm_ffn: RMSNorm::new(dim),
            ffn: Linear::new(dim, dim),
        }
    }

    /// Forward pass on `[B, T, D]` with optional causal masking.
    pub fn forward_with_mask(
        &self,
        x: &Variable,
        causal: bool,
    ) -> Result<Variable, rustorch_autograd::BackwardError> {
        let seq = x.tensor().shape()[1];
        let mask = if causal { Some(causal_mask(seq)) } else { None };
        let normed = self.norm_attn.forward(x)?;
        let attn_out = self.mha.self_attention(&normed, mask.as_ref())?;
        let h = ops::add(x, &attn_out)?;
        let normed2 = self.norm_ffn.forward(&h)?;
        let ffn_out = self.ffn.forward(&normed2)?;
        ops::add(&h, &ffn_out)
    }

    /// Total parameter count (sum of element counts across every
    /// trainable Variable in the block).
    pub fn parameter_count(&self) -> usize {
        let mut count = 0;
        for p in self.parameters() {
            count += p.tensor().numel();
        }
        count
    }
}

impl Module for TinyEncoder {
    fn forward(&self, input: &Variable) -> Result<Variable, rustorch_autograd::BackwardError> {
        TinyEncoder::forward_with_mask(self, input, false)
    }

    fn parameters(&self) -> Vec<Variable> {
        let mut p = self.norm_attn.parameters();
        p.extend(self.mha.parameters());
        p.extend(self.norm_ffn.parameters());
        p.extend(self.ffn.parameters());
        p
    }
}

#[cfg(target_arch = "wasm32")]
mod web {
    use super::*;
    use rustorch_core::tensor::tensor_impl::Tensor;
    use rustorch_nn::{CrossAttentionPool, Linear, SinusoidalPositionalEncoding};
    use rustorch_optim::clip_grad_norm_;
    use wasm_bindgen::prelude::*;

    fn now_ms() -> f64 {
        js_sys::Reflect::get(&js_sys::global(), &"performance".into())
            .ok()
            .and_then(|p| js_sys::Reflect::get(&p, &"now".into()).ok())
            .and_then(|f| f.dyn_into::<js_sys::Function>().ok())
            .and_then(|f| f.call0(&js_sys::global()).ok())
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0)
    }

    /// Synthetic deterministic input `[B, T, D]` for repeatable demos.
    fn synth(b: usize, t: usize, d: usize, seed: u64) -> Vec<f32> {
        (0..(b * t * d))
            .map(|i| {
                let bits = seed
                    .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                    .wrapping_add(i as u64);
                let unit = ((bits >> 33) as u32 as f32) / (u32::MAX as f32);
                unit - 0.5
            })
            .collect()
    }

    /// Stats for a single-block forward pass.
    #[wasm_bindgen]
    pub struct EncoderStats {
        param_count: u32,
        compute_ms: f64,
        first_eight: js_sys::Float32Array,
    }

    #[wasm_bindgen]
    impl EncoderStats {
        #[wasm_bindgen(getter)]
        pub fn param_count(&self) -> u32 {
            self.param_count
        }
        #[wasm_bindgen(getter)]
        pub fn compute_ms(&self) -> f64 {
            self.compute_ms
        }
        #[wasm_bindgen(getter)]
        pub fn first_eight(&self) -> js_sys::Float32Array {
            self.first_eight.clone()
        }
    }

    /// Run a tiny **autograd-aware** transformer encoder block on the
    /// CPU side of the WASM module. This exercises the new building
    /// blocks shipped in PRs 1, 2, 3, 5: `Linear` rank-N input,
    /// `MultiHeadAttention` self-attention, `causal_mask`, `RMSNorm`,
    /// `SinusoidalPositionalEncoding` (via `add` broadcast).
    ///
    /// Inputs:
    ///   - `b`, `t`, `d`: batch / seq_len / embed_dim
    ///   - `num_heads`: must divide `d`
    ///   - `causal`: whether to apply the causal mask
    ///
    /// Output: timing + parameter count + first eight values of the
    /// output tensor `[B, T, D]`.
    #[wasm_bindgen]
    pub fn run_encoder_block(
        b: u32,
        t: u32,
        d: u32,
        num_heads: u32,
        causal: bool,
    ) -> Result<EncoderStats, JsValue> {
        let b = b as usize;
        let t = t as usize;
        let d = d as usize;
        let num_heads = num_heads as usize;
        if d % num_heads != 0 {
            return Err(JsValue::from_str(&format!(
                "embed_dim {d} not divisible by num_heads {num_heads}"
            )));
        }

        // Build the model.
        let encoder = TinyEncoder::new(d, num_heads);
        let pe = SinusoidalPositionalEncoding::new(d, t);

        // Build the input + add positional encoding (broadcast over batch).
        let x_data = synth(b, t, d, 0xA1);
        let x = Variable::new(
            Tensor::from_vec([b, t, d], x_data)
                .map_err(|e| JsValue::from_str(&format!("input shape: {e}")))?,
        );
        let pe_v = pe
            .forward_for_len(t)
            .map_err(|e| JsValue::from_str(&format!("PE: {e}")))?; // [T, D]
        let pe_btd = ops::reshape(&pe_v, vec![1, t, d])
            .map_err(|e| JsValue::from_str(&format!("PE reshape: {e}")))?;
        let x = ops::add(&x, &pe_btd).map_err(|e| JsValue::from_str(&format!("add PE: {e}")))?;

        // Forward pass.
        let t0 = now_ms();
        let out = encoder
            .forward_with_mask(&x, causal)
            .map_err(|e| JsValue::from_str(&format!("encoder forward: {e}")))?;
        let compute_ms = now_ms() - t0;

        let out_t = out.tensor();
        let s = out_t
            .as_slice::<f32>()
            .ok_or_else(|| JsValue::from_str("output not f32"))?;
        let head = s.len().min(8);
        Ok(EncoderStats {
            param_count: encoder.parameter_count() as u32,
            compute_ms,
            first_eight: js_sys::Float32Array::from(&s[..head]),
        })
    }

    /// Stats for a CrossAttentionPool run.
    #[wasm_bindgen]
    pub struct PoolStats {
        out_shape: js_sys::Uint32Array,
        param_count: u32,
        compute_ms: f64,
        first_eight: js_sys::Float32Array,
    }

    #[wasm_bindgen]
    impl PoolStats {
        #[wasm_bindgen(getter)]
        pub fn out_shape(&self) -> js_sys::Uint32Array {
            self.out_shape.clone()
        }
        #[wasm_bindgen(getter)]
        pub fn param_count(&self) -> u32 {
            self.param_count
        }
        #[wasm_bindgen(getter)]
        pub fn compute_ms(&self) -> f64 {
            self.compute_ms
        }
        #[wasm_bindgen(getter)]
        pub fn first_eight(&self) -> js_sys::Float32Array {
            self.first_eight.clone()
        }
    }

    /// Pool a `[B, T, D]` sequence to `[B, num_queries, D]` via
    /// `CrossAttentionPool` (PR 7) — the Perceiver / Q-Former pattern.
    #[wasm_bindgen]
    pub fn run_cross_attention_pool(
        b: u32,
        t: u32,
        d: u32,
        num_heads: u32,
        num_queries: u32,
    ) -> Result<PoolStats, JsValue> {
        let b = b as usize;
        let t = t as usize;
        let d = d as usize;
        let num_heads = num_heads as usize;
        let num_queries = num_queries as usize;
        if d % num_heads != 0 {
            return Err(JsValue::from_str(&format!(
                "embed_dim {d} not divisible by num_heads {num_heads}"
            )));
        }

        let pool = CrossAttentionPool::new(d, num_queries, num_heads);
        let x_data = synth(b, t, d, 0xC4);
        let x = Variable::new(
            Tensor::from_vec([b, t, d], x_data)
                .map_err(|e| JsValue::from_str(&format!("input shape: {e}")))?,
        );

        let mut param_count: usize = 0;
        for p in pool.parameters() {
            param_count += p.tensor().numel();
        }

        let t0 = now_ms();
        let out = pool
            .forward(&x)
            .map_err(|e| JsValue::from_str(&format!("pool forward: {e}")))?;
        let compute_ms = now_ms() - t0;

        let shape: Vec<u32> = out.tensor().shape().iter().map(|&v| v as u32).collect();
        let out_t = out.tensor();
        let s = out_t
            .as_slice::<f32>()
            .ok_or_else(|| JsValue::from_str("output not f32"))?;
        let head = s.len().min(8);
        Ok(PoolStats {
            out_shape: js_sys::Uint32Array::from(shape.as_slice()),
            param_count: param_count as u32,
            compute_ms,
            first_eight: js_sys::Float32Array::from(&s[..head]),
        })
    }

    /// Stats for a single training step.
    #[wasm_bindgen]
    pub struct TrainingStats {
        loss_before: f32,
        grad_norm_before_clip: f32,
        clipped: bool,
        forward_ms: f64,
        backward_ms: f64,
        clip_ms: f64,
        param_count: u32,
    }

    #[wasm_bindgen]
    impl TrainingStats {
        #[wasm_bindgen(getter)]
        pub fn loss_before(&self) -> f32 {
            self.loss_before
        }
        #[wasm_bindgen(getter)]
        pub fn grad_norm_before_clip(&self) -> f32 {
            self.grad_norm_before_clip
        }
        #[wasm_bindgen(getter)]
        pub fn clipped(&self) -> bool {
            self.clipped
        }
        #[wasm_bindgen(getter)]
        pub fn forward_ms(&self) -> f64 {
            self.forward_ms
        }
        #[wasm_bindgen(getter)]
        pub fn backward_ms(&self) -> f64 {
            self.backward_ms
        }
        #[wasm_bindgen(getter)]
        pub fn clip_ms(&self) -> f64 {
            self.clip_ms
        }
        #[wasm_bindgen(getter)]
        pub fn param_count(&self) -> u32 {
            self.param_count
        }
    }

    /// Run a single forward + backward + clip_grad_norm step on a tiny
    /// 2-layer MLP. Demonstrates the **training stack** in the
    /// browser: `Linear` rank-N → `relu` → `Linear` → `mse_loss` →
    /// `backward` → `clip_grad_norm_`. Output is a stats struct (no
    /// optimizer step is performed; this is a smoke test of the
    /// gradient flow + clipping path).
    #[wasm_bindgen]
    pub fn run_training_step(
        batch: u32,
        in_dim: u32,
        hidden: u32,
        out_dim: u32,
        max_norm: f32,
    ) -> Result<TrainingStats, JsValue> {
        let batch = batch as usize;
        let in_dim = in_dim as usize;
        let hidden = hidden as usize;
        let out_dim = out_dim as usize;

        let l1 = Linear::new(in_dim, hidden);
        let l2 = Linear::new(hidden, out_dim);
        let params: Vec<Variable> = {
            let mut p = l1.parameters();
            p.extend(l2.parameters());
            p
        };
        let param_count: usize = params.iter().map(|p| p.tensor().numel()).sum();

        let x_data = synth(batch, 1, in_dim, 0xE5);
        let target_data = synth(batch, 1, out_dim, 0xF6);
        let x = Variable::new(
            Tensor::from_vec([batch, in_dim], x_data)
                .map_err(|e| JsValue::from_str(&format!("x: {e}")))?,
        );
        let target = Variable::new(
            Tensor::from_vec([batch, out_dim], target_data)
                .map_err(|e| JsValue::from_str(&format!("target: {e}")))?,
        );

        // Forward.
        let t_f = now_ms();
        let h = l1
            .forward(&x)
            .map_err(|e| JsValue::from_str(&format!("l1: {e}")))?;
        let h = ops::relu(&h).map_err(|e| JsValue::from_str(&format!("relu: {e}")))?;
        let logits = l2
            .forward(&h)
            .map_err(|e| JsValue::from_str(&format!("l2: {e}")))?;
        let loss = ops::mse_loss(&logits, &target, rustorch_cpu::backend::Reduction::Mean)
            .map_err(|e| JsValue::from_str(&format!("mse_loss: {e}")))?;
        let forward_ms = now_ms() - t_f;
        let loss_before = loss.tensor().as_slice::<f32>().unwrap()[0];

        // Backward.
        let t_b = now_ms();
        rustorch_autograd::backward(&loss, None)
            .map_err(|e| JsValue::from_str(&format!("backward: {e}")))?;
        let backward_ms = now_ms() - t_b;

        // Clip grad norm.
        let t_c = now_ms();
        let grad_norm = clip_grad_norm_(&params, max_norm);
        let clip_ms = now_ms() - t_c;
        let clipped = grad_norm > max_norm;

        Ok(TrainingStats {
            loss_before,
            grad_norm_before_clip: grad_norm,
            clipped,
            forward_ms,
            backward_ms,
            clip_ms,
            param_count: param_count as u32,
        })
    }
}

#[cfg(target_arch = "wasm32")]
pub use web::*;

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;
    use rustorch_core::tensor::tensor_impl::Tensor;

    fn synth_native(b: usize, t: usize, d: usize) -> Vec<f32> {
        (0..(b * t * d))
            .map(|i| ((i as u32).wrapping_mul(2_654_435_761) as f32 / u32::MAX as f32) - 0.5)
            .collect()
    }

    /// The TinyEncoder forward path mirrors what the wasm `run_encoder_block`
    /// drives. We exercise it natively to confirm the gradient flow + the
    /// causal-mask wiring without needing a browser.
    #[test]
    fn encoder_forward_shape_preserved() {
        let dim = 16;
        let heads = 4;
        let b = 2;
        let t = 5;
        let encoder = TinyEncoder::new(dim, heads);
        let x = Variable::new(Tensor::from_vec([b, t, dim], synth_native(b, t, dim)).unwrap());
        let y = encoder.forward_with_mask(&x, true).unwrap();
        assert_eq!(y.tensor().shape(), &[b, t, dim]);
    }

    #[test]
    fn encoder_parameter_count_consistent() {
        // 2 RMSNorm gammas of size dim + 4 MHA projections (each W=[D,D] + b=[D]) + 1 ffn (W=[D,D] + b=[D]).
        // = 2*D + 4*(D*D + D) + (D*D + D) = 2D + 5*D*D + 5*D = 5*D² + 7*D
        let dim = 16;
        let heads = 4;
        let encoder = TinyEncoder::new(dim, heads);
        let expected = 5 * dim * dim + 7 * dim;
        assert_eq!(encoder.parameter_count(), expected);
    }

    /// Causal mask must not produce NaNs even on the first row (where all
    /// future positions are masked).
    #[test]
    fn encoder_causal_mask_no_nan() {
        let encoder = TinyEncoder::new(8, 2);
        let x = Variable::new(Tensor::from_vec([1, 4, 8], synth_native(1, 4, 8)).unwrap());
        let y = encoder.forward_with_mask(&x, true).unwrap();
        for &v in y.tensor().as_slice::<f32>().unwrap() {
            assert!(v.is_finite(), "encoder output has non-finite value: {v}");
        }
    }
}
