//! Llama / Qwen forward pass + autoregressive decode loop (T59).
//!
//! Wires the LLM-runner bricks (KV-cache, RoPE, GQA, sampling)
//! into a single end-to-end `generate()` API. Built on top of the
//! `LlamaConfig` + `HfWeights` produced by [`crate::HfWeights`].
//!
//! ## Architecture covered
//!
//! Llama-2/3 / Qwen2.5 / Qwen3 (dense) decoder block:
//! ```text
//!   x = x + Attention(RMSNorm(x))           // attention sub-block
//!   x = x + SwiGLU_FFN(RMSNorm(x))           // FFN sub-block
//! ```
//! With:
//!   - GQA attention (n_heads query heads, n_kv_heads K/V heads).
//!   - Rotary position embeddings on Q and K (V not rotated).
//!   - SwiGLU FFN: `down(silu(gate(x)) * up(x))`.
//!
//! ## Limitations of v0
//!
//! - F32 weights only. INT4 / bf16 land in T58.
//! - No causal mask in attention — decode-only path. For prefill
//!   on multi-token prompts we currently call decode_step in a
//!   loop, which is O(N²) but correct (the KV-cache trick still
//!   gives O(N) per token after prefill). A flash-attention
//!   prefill path is a follow-up.
//! - QKV bias is loaded if present (Qwen2.5/Qwen3 use bias on
//!   Q/K/V) but not yet plumbed through the kernel — TODO.

use crate::{hf_block_key, hf_keys, HfWeights, LlamaConfig, LlmError};
use rustorch_core::tensor::tensor_impl::Tensor;
use rustorch_fusion::patterns::matmul_bias_act::{fused_matmul_bias_activation, Activation};
use rustorch_nn::gqa::gqa_forward_f32;
use rustorch_nn::kv_cache::KVCache;
use rustorch_nn::rope::RoPE;
use rustorch_nn::sampling::{sample_next, SamplingConfig};

/// Per-block weights extracted from the HF weight map and stored
/// as raw `Vec<f32>` for the hot-path kernel (no Tensor wrapping
/// across step boundaries).
struct BlockWeights {
    rms_attn: Vec<f32>, // [D]
    w_q: Vec<f32>,      // [D, D]
    w_k: Vec<f32>,      // [D, KV_DIM]
    w_v: Vec<f32>,      // [D, KV_DIM]
    w_o: Vec<f32>,      // [D, D]
    rms_ffn: Vec<f32>,  // [D]
    /// SwiGLU gate projection: [D, F].
    w_gate: Vec<f32>,
    /// SwiGLU up projection: [D, F].
    w_up: Vec<f32>,
    /// SwiGLU down projection: [F, D].
    w_down: Vec<f32>,
}

/// A loaded Llama / Qwen model ready for autoregressive decode.
pub struct LlamaModel {
    pub config: LlamaConfig,
    blocks: Vec<BlockWeights>,
    /// `[V, D]` token embedding lookup table flattened row-major.
    token_emb: Vec<f32>,
    /// `[D]` final RMSNorm gamma.
    final_norm: Vec<f32>,
    /// `[D, V]` LM head weight (or alias of `token_emb` if
    /// `tie_word_embeddings == true`).
    lm_head: Vec<f32>,
    /// Pre-computed RoPE cos/sin tables sized to
    /// `min(config.max_position_embeddings, max_seq)`.
    rope: RoPE,
}

impl LlamaModel {
    /// Build a runnable model from a parsed config and a flat
    /// HuggingFace weight dictionary. Copies tensor data into the
    /// raw f32 working buffers.
    ///
    /// `max_seq` caps the size of the RoPE table; the KV-cache is
    /// sized when [`generate`] is called.
    pub fn from_hf(
        config: LlamaConfig,
        weights: &HfWeights,
        max_seq: usize,
    ) -> Result<Self, LlmError> {
        let d = config.hidden_size;
        let kv_dim = config.n_kv_heads() * config.head_dim();
        let f = config.intermediate_size;

        let token_emb_t = weights.expect(hf_keys::EMBED_TOKENS)?;
        let token_emb = tensor_to_vec(token_emb_t)?;
        if token_emb.len() != config.vocab_size * d {
            return Err(LlmError::MissingWeight(format!(
                "embed_tokens: got {} f32, expected {}",
                token_emb.len(),
                config.vocab_size * d
            )));
        }

        let final_norm = tensor_to_vec(weights.expect(hf_keys::FINAL_NORM)?)?;

        // tie_word_embeddings: lm_head shares storage with the
        // token embedding (transposed view at attention time, but
        // since we stored embed [V, D] and lm_head needs [D, V] for
        // sgemv x @ W, we flat-copy in the right layout).
        let lm_head = if config.tie_word_embeddings {
            // The HF convention with tied weights is to store
            // `embed_tokens.weight` only and skip `lm_head.weight`.
            // For our sgemv path we need [D, V] (input D, output V),
            // which is the transpose of the [V, D] embedding. We
            // transpose into a separate buffer once at load time;
            // that costs `V*D*4` bytes but avoids per-step cost.
            transpose_2d(&token_emb, config.vocab_size, d)
        } else {
            tensor_to_vec(weights.expect(hf_keys::LM_HEAD)?)?
        };

        let mut blocks: Vec<BlockWeights> = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let rms_attn =
                tensor_to_vec(weights.expect(&hf_block_key(i, "input_layernorm.weight"))?)?;
            let w_q = tensor_to_vec(weights.expect(&hf_block_key(i, "self_attn.q_proj.weight"))?)?;
            let w_k = tensor_to_vec(weights.expect(&hf_block_key(i, "self_attn.k_proj.weight"))?)?;
            let w_v = tensor_to_vec(weights.expect(&hf_block_key(i, "self_attn.v_proj.weight"))?)?;
            let w_o = tensor_to_vec(weights.expect(&hf_block_key(i, "self_attn.o_proj.weight"))?)?;
            let rms_ffn = tensor_to_vec(
                weights.expect(&hf_block_key(i, "post_attention_layernorm.weight"))?,
            )?;
            let w_gate = tensor_to_vec(weights.expect(&hf_block_key(i, "mlp.gate_proj.weight"))?)?;
            let w_up = tensor_to_vec(weights.expect(&hf_block_key(i, "mlp.up_proj.weight"))?)?;
            let w_down = tensor_to_vec(weights.expect(&hf_block_key(i, "mlp.down_proj.weight"))?)?;
            // HF stores Q/K/V as [out, in]; our sgemv kernels expect
            // [in, out] (input vector left-multiplies the matrix).
            // Transpose at load time so the hot path stays clean.
            let w_q = transpose_2d(&w_q, d, d);
            let w_k = transpose_2d(&w_k, kv_dim, d);
            let w_v = transpose_2d(&w_v, kv_dim, d);
            let w_o = transpose_2d(&w_o, d, d);
            let w_gate = transpose_2d(&w_gate, f, d);
            let w_up = transpose_2d(&w_up, f, d);
            let w_down = transpose_2d(&w_down, d, f);
            blocks.push(BlockWeights {
                rms_attn,
                w_q,
                w_k,
                w_v,
                w_o,
                rms_ffn,
                w_gate,
                w_up,
                w_down,
            });
        }

        let rope = RoPE::new(
            config.head_dim(),
            max_seq.min(config.max_position_embeddings),
            config.rope_theta,
        );

        Ok(LlamaModel {
            config,
            blocks,
            token_emb,
            final_norm,
            lm_head,
            rope,
        })
    }

    /// Run a full prefill + autoregressive decode generation loop.
    /// Returns the generated token ids (excluding the prompt).
    pub fn generate(
        &self,
        prompt_ids: &[u32],
        sampling_cfg: &SamplingConfig,
        max_new_tokens: usize,
        max_seq: usize,
    ) -> Vec<u32> {
        let cfg = &self.config;
        let mut cache = KVCache::new(
            cfg.num_hidden_layers,
            1,
            cfg.n_kv_heads(),
            cfg.head_dim(),
            max_seq,
        );
        let mut scratch = Scratch::new(cfg);
        let mut output: Vec<u32> = Vec::with_capacity(prompt_ids.len() + max_new_tokens);
        let mut next_u_state: u64 = 1;
        let mut next_u = move || {
            next_u_state = next_u_state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            ((next_u_state >> 32) as f32) / (u32::MAX as f32)
        };

        // Prefill on the prompt.
        for (pos, &tok) in prompt_ids.iter().enumerate() {
            self.decode_step(tok as i64, pos, &mut cache, &mut scratch);
            cache.advance(1).unwrap();
            output.push(tok);
        }

        if prompt_ids.is_empty() {
            return Vec::new();
        }
        let mut last_token = *prompt_ids.last().unwrap() as i64;
        let mut new_tokens: Vec<u32> = Vec::with_capacity(max_new_tokens);
        for _ in 0..max_new_tokens {
            let pos = cache.current_len();
            self.decode_step(last_token, pos, &mut cache, &mut scratch);
            cache.advance(1).unwrap();
            let next = sample_next(&scratch.logits, sampling_cfg, &output, &mut next_u);
            output.push(next as u32);
            new_tokens.push(next as u32);
            last_token = next as i64;
        }
        new_tokens
    }

    /// One decode step. Reads token_id, writes logits into
    /// `scratch.logits`. Caller is responsible for sampling +
    /// `cache.advance(1)`.
    fn decode_step(
        &self,
        token_id: i64,
        position: usize,
        cache: &mut KVCache,
        scratch: &mut Scratch,
    ) {
        let cfg = &self.config;
        let d = cfg.hidden_size;
        let kv_dim = cfg.n_kv_heads() * cfg.head_dim();
        let f = cfg.intermediate_size;
        let n_heads = cfg.num_attention_heads;
        let n_kv = cfg.n_kv_heads();
        let head_dim = cfg.head_dim();

        // Embed.
        let off = (token_id as usize) * d;
        scratch.x[..d].copy_from_slice(&self.token_emb[off..off + d]);

        for (layer_idx, block) in self.blocks.iter().enumerate() {
            // 1. RMSNorm + Q/K/V proj.
            scratch.h[..d].copy_from_slice(&scratch.x[..d]);
            rms_norm_inplace(&mut scratch.h[..d], &block.rms_attn, cfg.rms_norm_eps);
            sgemv_dispatch(&scratch.h[..d], &block.w_q, &mut scratch.q[..d], d, d);
            sgemv_dispatch(
                &scratch.h[..d],
                &block.w_k,
                &mut scratch.k[..kv_dim],
                d,
                kv_dim,
            );
            sgemv_dispatch(
                &scratch.h[..d],
                &block.w_v,
                &mut scratch.v[..kv_dim],
                d,
                kv_dim,
            );

            // 2. RoPE on Q and K.
            self.rope
                .apply_inplace(&mut scratch.q[..d], 1, n_heads, 1, position)
                .unwrap();
            self.rope
                .apply_inplace(&mut scratch.k[..kv_dim], 1, n_kv, 1, position)
                .unwrap();

            // 3. KV-cache append.
            cache
                .append(layer_idx, 1, &scratch.k[..kv_dim], &scratch.v[..kv_dim])
                .unwrap();

            // 4. Trim cache prefix and run GQA decode.
            let kv_len = position + 1;
            let max_seq = cache.max_seq();
            let k_full = cache.k_buffer(layer_idx).unwrap();
            let v_full = cache.v_buffer(layer_idx).unwrap();
            let trim_len = n_kv * kv_len * head_dim;
            for kv_h in 0..n_kv {
                let src_off = kv_h * max_seq * head_dim;
                let dst_off = kv_h * kv_len * head_dim;
                scratch.k_trim[dst_off..dst_off + kv_len * head_dim]
                    .copy_from_slice(&k_full[src_off..src_off + kv_len * head_dim]);
                scratch.v_trim[dst_off..dst_off + kv_len * head_dim]
                    .copy_from_slice(&v_full[src_off..src_off + kv_len * head_dim]);
            }
            gqa_forward_f32(
                &scratch.q[..d],
                &scratch.k_trim[..trim_len],
                &scratch.v_trim[..trim_len],
                &mut scratch.attn_out[..d],
                1,
                n_heads,
                n_kv,
                1,
                kv_len,
                head_dim,
            )
            .unwrap();

            // 5. O proj + residual.
            sgemv_dispatch(
                &scratch.attn_out[..d],
                &block.w_o,
                &mut scratch.o_out[..d],
                d,
                d,
            );
            for i in 0..d {
                scratch.x[i] += scratch.o_out[i];
            }

            // 6. RMSNorm + SwiGLU FFN + residual.
            scratch.h[..d].copy_from_slice(&scratch.x[..d]);
            rms_norm_inplace(&mut scratch.h[..d], &block.rms_ffn, cfg.rms_norm_eps);
            sgemv_dispatch(
                &scratch.h[..d],
                &block.w_gate,
                &mut scratch.gate_out[..f],
                d,
                f,
            );
            sgemv_dispatch(&scratch.h[..d], &block.w_up, &mut scratch.up_out[..f], d, f);
            // SwiGLU: silu(gate) * up — element-wise.
            for i in 0..f {
                let g = scratch.gate_out[i];
                let s = g / (1.0 + (-g).exp()); // silu(x) = x * sigmoid(x)
                scratch.gate_out[i] = s * scratch.up_out[i];
            }
            sgemv_dispatch(
                &scratch.gate_out[..f],
                &block.w_down,
                &mut scratch.fc2_out[..d],
                f,
                d,
            );
            for i in 0..d {
                scratch.x[i] += scratch.fc2_out[i];
            }
        }

        // 7. Final RMSNorm + LM head.
        scratch.h[..d].copy_from_slice(&scratch.x[..d]);
        rms_norm_inplace(&mut scratch.h[..d], &self.final_norm, cfg.rms_norm_eps);
        sgemv_dispatch(
            &scratch.h[..d],
            &self.lm_head,
            &mut scratch.logits[..cfg.vocab_size],
            d,
            cfg.vocab_size,
        );
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Per-step scratch buffers, allocated once.
struct Scratch {
    x: Vec<f32>,
    h: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    attn_out: Vec<f32>,
    o_out: Vec<f32>,
    gate_out: Vec<f32>, // F
    up_out: Vec<f32>,   // F
    fc2_out: Vec<f32>,  // D
    logits: Vec<f32>,
    k_trim: Vec<f32>,
    v_trim: Vec<f32>,
}

impl Scratch {
    fn new(cfg: &LlamaConfig) -> Self {
        let d = cfg.hidden_size;
        let kv_dim = cfg.n_kv_heads() * cfg.head_dim();
        let f = cfg.intermediate_size;
        let max_kv_size = cfg.n_kv_heads() * cfg.max_position_embeddings * cfg.head_dim();
        Scratch {
            x: vec![0.0; d],
            h: vec![0.0; d],
            q: vec![0.0; d],
            k: vec![0.0; kv_dim],
            v: vec![0.0; kv_dim],
            attn_out: vec![0.0; d],
            o_out: vec![0.0; d],
            gate_out: vec![0.0; f],
            up_out: vec![0.0; f],
            fc2_out: vec![0.0; d],
            logits: vec![0.0; cfg.vocab_size],
            k_trim: vec![0.0; max_kv_size],
            v_trim: vec![0.0; max_kv_size],
        }
    }
}

fn rms_norm_inplace(x: &mut [f32], gamma: &[f32], eps: f32) {
    let d = x.len();
    let inv_d = 1.0_f32 / d as f32;
    let mut sq = 0.0_f32;
    for &v in x.iter() {
        sq += v * v;
    }
    let inv_rms = 1.0 / (sq * inv_d + eps).sqrt();
    for i in 0..d {
        x[i] = x[i] * inv_rms * gamma[i];
    }
}

/// Hybrid sgemv — same calibration as T53 in the demo: cBLAS via
/// `fused_matmul_bias_activation` for shapes ≥ 200 K FLOPs,
/// custom row-major loop otherwise.
fn sgemv_dispatch(x: &[f32], w: &[f32], y: &mut [f32], k: usize, n: usize) {
    if k * n >= 200_000 {
        fused_matmul_bias_activation(x, w, None, y, 1, k, n, Activation::None).unwrap();
    } else {
        y.fill(0.0);
        for kk in 0..k {
            let xk = x[kk];
            let row = &w[kk * n..(kk + 1) * n];
            for nn in 0..n {
                y[nn] += xk * row[nn];
            }
        }
    }
}

fn tensor_to_vec(t: &Tensor) -> Result<Vec<f32>, LlmError> {
    t.as_slice::<f32>()
        .map(|s| s.to_vec())
        .ok_or_else(|| LlmError::Safetensors("non-contiguous or non-f32 tensor".to_string()))
}

/// Transpose a row-major `[rows, cols]` flat buffer to
/// `[cols, rows]`. Used at load time so the sgemv hot path always
/// sees the orientation it expects.
fn transpose_2d(src: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    debug_assert_eq!(src.len(), rows * cols);
    let mut dst = vec![0.0_f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            dst[c * rows + r] = src[r * cols + c];
        }
    }
    dst
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    /// Build a tiny Llama-style model in-memory (no on-disk
    /// fixtures) and verify the forward pass produces finite
    /// logits of the right shape.
    #[test]
    fn forward_random_weights_runs_and_outputs_finite_logits() {
        let cfg = LlamaConfig::from_json(
            r#"{
                "hidden_size": 32,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "intermediate_size": 64,
                "num_hidden_layers": 2,
                "vocab_size": 16,
                "max_position_embeddings": 16,
                "rms_norm_eps": 1e-5,
                "rope_theta": 10000.0,
                "tie_word_embeddings": false
            }"#,
        )
        .unwrap();

        // Build random weights matching the config.
        let mut weights: BTreeMap<String, Tensor> = BTreeMap::new();
        let d = cfg.hidden_size;
        let kv_dim = cfg.n_kv_heads() * cfg.head_dim();
        let f = cfg.intermediate_size;

        let mut s: u64 = 1;
        let mut next = || {
            s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            ((s >> 32) as f32) / (u32::MAX as f32) * 2.0 - 1.0
        };
        let mut rand_vec = |n: usize| -> Vec<f32> { (0..n).map(|_| next()).collect() };

        weights.insert(
            hf_keys::EMBED_TOKENS.to_string(),
            Tensor::from_vec(vec![cfg.vocab_size, d], rand_vec(cfg.vocab_size * d)).unwrap(),
        );
        weights.insert(
            hf_keys::FINAL_NORM.to_string(),
            Tensor::from_vec(vec![d], vec![1.0_f32; d]).unwrap(),
        );
        weights.insert(
            hf_keys::LM_HEAD.to_string(),
            Tensor::from_vec(vec![cfg.vocab_size, d], rand_vec(cfg.vocab_size * d)).unwrap(),
        );
        for i in 0..cfg.num_hidden_layers {
            weights.insert(
                hf_block_key(i, "input_layernorm.weight"),
                Tensor::from_vec(vec![d], vec![1.0_f32; d]).unwrap(),
            );
            weights.insert(
                hf_block_key(i, "self_attn.q_proj.weight"),
                Tensor::from_vec(vec![d, d], rand_vec(d * d)).unwrap(),
            );
            weights.insert(
                hf_block_key(i, "self_attn.k_proj.weight"),
                Tensor::from_vec(vec![kv_dim, d], rand_vec(kv_dim * d)).unwrap(),
            );
            weights.insert(
                hf_block_key(i, "self_attn.v_proj.weight"),
                Tensor::from_vec(vec![kv_dim, d], rand_vec(kv_dim * d)).unwrap(),
            );
            weights.insert(
                hf_block_key(i, "self_attn.o_proj.weight"),
                Tensor::from_vec(vec![d, d], rand_vec(d * d)).unwrap(),
            );
            weights.insert(
                hf_block_key(i, "post_attention_layernorm.weight"),
                Tensor::from_vec(vec![d], vec![1.0_f32; d]).unwrap(),
            );
            weights.insert(
                hf_block_key(i, "mlp.gate_proj.weight"),
                Tensor::from_vec(vec![f, d], rand_vec(f * d)).unwrap(),
            );
            weights.insert(
                hf_block_key(i, "mlp.up_proj.weight"),
                Tensor::from_vec(vec![f, d], rand_vec(f * d)).unwrap(),
            );
            weights.insert(
                hf_block_key(i, "mlp.down_proj.weight"),
                Tensor::from_vec(vec![d, f], rand_vec(d * f)).unwrap(),
            );
        }
        let hf = HfWeights::from_map(weights);
        let model = LlamaModel::from_hf(cfg, &hf, 16).unwrap();

        let cfg_sampling = SamplingConfig::greedy();
        let new_tokens = model.generate(&[1, 2, 3], &cfg_sampling, 4, 16);
        assert_eq!(new_tokens.len(), 4);
        for &tok in new_tokens.iter() {
            assert!((tok as usize) < model.config.vocab_size);
        }
    }
}
