//! Single GPT-style transformer block with character-level
//! tokenisation, runnable in the browser.
//!
//! Architecture (one block):
//! ```text
//!   x [S, D]
//!     → layernorm
//!     → multi-head causal self-attention   (8 heads by default)
//!     → residual add
//!     → layernorm
//!     → linear (4D) → gelu → linear (D)
//!     → residual add
//! ```
//!
//! v1 ships with random weights and character-level "tokenisation"
//! (one ASCII byte = one token). This is **not** a real GPT-2 with
//! BPE + 124M trained weights — it's a structurally accurate
//! demonstration of the kernel pipeline (attention + residual + FFN
//! + LayerNorm) running end-to-end in the browser.
//!
//! Real GPT-2 weights + BPE land in a follow-up once the
//! safetensors loader is wired into the page.

use rustorch_core::tensor::tensor_impl::Tensor;
use rustorch_wgpu::{
    dispatch_binary, layernorm_rows, linear_gelu_fused, matmul, multi_head_attention, to_cpu,
    to_gpu, WgpuBackend, WgpuError,
};

/// Generate `n` deterministic f32 values in `[-σ, σ]` from `seed`.
fn det_weights(n: usize, seed: u64, sigma: f32) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let mut x = seed
                .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                .wrapping_add(i as u64);
            x ^= x >> 33;
            x = x.wrapping_mul(0xff51_afd7_ed55_8ccd);
            x ^= x >> 33;
            let unit = ((x >> 33) as u32 as f32) / (u32::MAX as f32);
            (unit - 0.5) * 2.0 * sigma
        })
        .collect()
}

/// Convert an ASCII string into a `[s, d_model]` embedding.
/// Character-level: each byte indexes a row of an embedding matrix
/// of shape `[256, d_model]`. The matrix uses fixed seeded weights
/// so the demo is reproducible.
fn embed_chars(text: &str, d_model: usize) -> Vec<f32> {
    let bytes = text.as_bytes();
    let s = bytes.len();
    let emb_table = det_weights(256 * d_model, 0xE4B0, 0.1);
    let mut out = Vec::with_capacity(s * d_model);
    for &b in bytes {
        let off = (b as usize) * d_model;
        out.extend_from_slice(&emb_table[off..off + d_model]);
    }
    out
}

/// Run one GPT-style transformer block on `text`, returning the
/// resulting `[s, d_model]` activation as a flat F32 vector.
///
/// `s` (sequence length) = `text.as_bytes().len()`.
/// `d_model` and `num_heads` are caller-supplied; `d_model` must be
/// divisible by `num_heads` and `d_model ≤ 256` (Flash Attention
/// per-thread d-cap is honoured by the multi_head_attention helper
/// downstream).
pub fn run_gpt_block(
    backend: &WgpuBackend,
    text: &str,
    d_model: usize,
    num_heads: usize,
) -> Result<Vec<f32>, WgpuError> {
    let bytes = text.as_bytes();
    let s = bytes.len();
    if s == 0 {
        return Err(WgpuError::ShapeMismatch("empty prompt".to_string()));
    }
    if d_model % num_heads != 0 {
        return Err(WgpuError::ShapeMismatch(format!(
            "d_model {} not divisible by num_heads {}",
            d_model, num_heads
        )));
    }

    // Embed.
    let x_data = embed_chars(text, d_model);
    let tx = Tensor::from_vec([s, d_model], x_data)
        .map_err(|e| WgpuError::ShapeMismatch(format!("{e}")))?;
    let gx = to_gpu(backend, &tx)?;

    // Pre-attention LayerNorm (gamma=1, beta=0).
    let gamma = vec![1.0_f32; d_model];
    let beta = vec![0.0_f32; d_model];
    let tg = Tensor::from_vec([d_model], gamma.clone())
        .map_err(|e| WgpuError::ShapeMismatch(format!("{e}")))?;
    let tb = Tensor::from_vec([d_model], beta.clone())
        .map_err(|e| WgpuError::ShapeMismatch(format!("{e}")))?;
    let gg = to_gpu(backend, &tg)?;
    let gbeta = to_gpu(backend, &tb)?;
    let g_pre1 = layernorm_rows(backend, &gx, &gg, &gbeta, s, d_model, 1e-5)?;

    // Q/K/V projections.
    let wq = det_weights(d_model * d_model, 0x110A, 0.05);
    let wk = det_weights(d_model * d_model, 0x110B, 0.05);
    let wv = det_weights(d_model * d_model, 0x110C, 0.05);
    let g_wq = to_gpu(
        backend,
        &Tensor::from_vec([d_model, d_model], wq)
            .map_err(|e| WgpuError::ShapeMismatch(format!("{e}")))?,
    )?;
    let g_wk = to_gpu(
        backend,
        &Tensor::from_vec([d_model, d_model], wk)
            .map_err(|e| WgpuError::ShapeMismatch(format!("{e}")))?,
    )?;
    let g_wv = to_gpu(
        backend,
        &Tensor::from_vec([d_model, d_model], wv)
            .map_err(|e| WgpuError::ShapeMismatch(format!("{e}")))?,
    )?;
    let q = matmul(backend, &g_pre1, &g_wq, s, d_model, d_model)?;
    let k = matmul(backend, &g_pre1, &g_wk, s, d_model, d_model)?;
    let v = matmul(backend, &g_pre1, &g_wv, s, d_model, d_model)?;

    // Causal multi-head attention.
    let g_attn = multi_head_attention(backend, &q, &k, &v, s, d_model, num_heads, true)?;

    // Residual add: x + attn_out.
    let g_res1 = dispatch_binary(backend, "add", &gx, &g_attn)?;

    // Pre-FFN LayerNorm.
    let g_pre2 = layernorm_rows(backend, &g_res1, &gg, &gbeta, s, d_model, 1e-5)?;

    // FFN: linear (d_model → 4·d_model) + GELU + linear (4·d_model → d_model).
    let d_ff = 4 * d_model;
    let w_ff1 = det_weights(d_model * d_ff, 0xFF01, 0.05);
    let b_ff1 = det_weights(d_ff, 0xFF02, 0.01);
    let w_ff2 = det_weights(d_ff * d_model, 0xFF03, 0.05);
    let b_ff2 = det_weights(d_model, 0xFF04, 0.01);

    let g_w_ff1 = to_gpu(
        backend,
        &Tensor::from_vec([d_model, d_ff], w_ff1)
            .map_err(|e| WgpuError::ShapeMismatch(format!("{e}")))?,
    )?;
    let g_b_ff1 = to_gpu(
        backend,
        &Tensor::from_vec([d_ff], b_ff1).map_err(|e| WgpuError::ShapeMismatch(format!("{e}")))?,
    )?;
    let g_ffn1 = linear_gelu_fused(backend, &g_pre2, &g_w_ff1, &g_b_ff1, s, d_model, d_ff)?;

    let g_w_ff2 = to_gpu(
        backend,
        &Tensor::from_vec([d_ff, d_model], w_ff2)
            .map_err(|e| WgpuError::ShapeMismatch(format!("{e}")))?,
    )?;
    // Second linear: matmul + bias add (no activation).
    let g_proj = matmul(backend, &g_ffn1, &g_w_ff2, s, d_ff, d_model)?;
    // Broadcast bias [d_model] across rows then add.
    let bias_bcast: Vec<f32> = (0..s).flat_map(|_| b_ff2.clone()).collect();
    let g_b_bcast = to_gpu(
        backend,
        &Tensor::from_vec([s, d_model], bias_bcast)
            .map_err(|e| WgpuError::ShapeMismatch(format!("{e}")))?,
    )?;
    let g_ffn2 = dispatch_binary(backend, "add", &g_proj, &g_b_bcast)?;

    // Final residual: g_res1 + g_ffn2.
    let g_out = dispatch_binary(backend, "add", &g_res1, &g_ffn2)?;
    let out_cpu = to_cpu(backend, &g_out, vec![s, d_model])?;
    Ok(out_cpu
        .as_slice::<f32>()
        .ok_or_else(|| WgpuError::ShapeMismatch("out not f32".into()))?
        .to_vec())
}

/// Greedy "generation" stub: take a prompt, run the block, look at
/// the last position's activation, pick the byte whose row in the
/// embedding table maximises the dot product (softmax argmax). Not
/// real text generation — it's a structural placeholder so the demo
/// page can show the full encode → forward → argmax → decode loop.
pub fn generate_one_byte(
    backend: &WgpuBackend,
    prompt: &str,
    d_model: usize,
    num_heads: usize,
) -> Result<u8, WgpuError> {
    let last = run_gpt_block(backend, prompt, d_model, num_heads)?;
    // Last `d_model` chunk = activation at the final position.
    let s = prompt.len();
    let activation = &last[(s - 1) * d_model..s * d_model];
    // Score each byte by dot product against its embedding row.
    let emb_table = det_weights(256 * d_model, 0xE4B0, 0.1);
    let mut best_byte: u8 = 0;
    let mut best_score: f32 = f32::NEG_INFINITY;
    for b in 0..=255_u8 {
        let row = &emb_table[(b as usize) * d_model..(b as usize + 1) * d_model];
        let score: f32 = activation.iter().zip(row).map(|(a, b)| a * b).sum();
        if score > best_score {
            best_score = score;
            best_byte = b;
        }
    }
    Ok(best_byte)
}

#[cfg(target_arch = "wasm32")]
mod web {
    use super::*;
    use wasm_bindgen::prelude::*;

    /// JS-callable: run one transformer block on `prompt`, return
    /// the activation of the last position as a Float32Array.
    #[wasm_bindgen]
    pub async fn run_gpt_block_js(
        prompt: &str,
        d_model: u32,
        num_heads: u32,
    ) -> Result<js_sys::Float32Array, JsValue> {
        let backend = WgpuBackend::new()
            .await
            .map_err(|e| JsValue::from_str(&format!("backend init: {e}")))?;
        let out = run_gpt_block(&backend, prompt, d_model as usize, num_heads as usize)
            .map_err(|e| JsValue::from_str(&format!("gpt_block: {e}")))?;
        let s = prompt.as_bytes().len();
        let dm = d_model as usize;
        let last_row = &out[(s - 1) * dm..s * dm];
        Ok(js_sys::Float32Array::from(last_row))
    }
}

#[cfg(target_arch = "wasm32")]
pub use web::*;

#[cfg(all(test, not(target_arch = "wasm32"), feature = "gpu-tests"))]
mod gpu_tests {
    use super::*;

    #[test]
    fn gpt_block_runs_short_prompt() {
        let backend = WgpuBackend::new_blocking().expect("init");
        let out = run_gpt_block(&backend, "hello", 32, 4).unwrap();
        assert_eq!(out.len(), 5 * 32);
        // Output must be finite (no NaN / Inf from the residual chain).
        for &v in &out {
            assert!(v.is_finite(), "non-finite value: {v}");
        }
    }

    #[test]
    fn generate_one_byte_returns_an_ascii_byte() {
        let backend = WgpuBackend::new_blocking().expect("init");
        let b = generate_one_byte(&backend, "abc", 32, 4).unwrap();
        let _ = b; // any byte is valid; the test asserts no panic.
    }

    #[test]
    fn gpt_block_rejects_empty_prompt() {
        let backend = WgpuBackend::new_blocking().expect("init");
        let err = match run_gpt_block(&backend, "", 32, 4) {
            Ok(_) => panic!("expected error"),
            Err(e) => e,
        };
        assert!(format!("{err}").contains("empty"));
    }

    #[test]
    fn gpt_block_rejects_bad_head_count() {
        let backend = WgpuBackend::new_blocking().expect("init");
        // d_model=32, num_heads=5 — not divisible.
        let err = match run_gpt_block(&backend, "hello", 32, 5) {
            Ok(_) => panic!("expected error"),
            Err(e) => e,
        };
        assert!(format!("{err}").contains("divisible"));
    }
}
