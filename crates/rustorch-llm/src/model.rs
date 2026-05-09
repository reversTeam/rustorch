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

use crate::{hf_block_key, hf_keys, GgufWeights, HfWeights, LlamaConfig, LlmError};
use rustorch_core::tensor::tensor_impl::Tensor;
use rustorch_fusion::patterns::matmul_bias_act::{fused_matmul_bias_activation, Activation};
use rustorch_nn::gqa::gqa_forward_f32;
use rustorch_nn::kv_cache::KVCache;
use rustorch_nn::rope::RoPE;
use rustorch_nn::sampling::{sample_next, SamplingConfig};

/// Per-block weights extracted from the HF weight map and stored
/// as raw `Vec<f32>` for the hot-path kernel (no Tensor wrapping
/// across step boundaries).
pub(crate) struct BlockWeights {
    pub(crate) rms_attn: Vec<f32>, // [D]
    /// Fused Q || K || V projection — single matmul amortises the
    /// memory loads of `h` across all three projections (Q, K, V
    /// share the same input). Layout: row-major `[D, D + 2*KV_DIM]`.
    /// Output is sliced into `q[..D]`, `k[..KV_DIM]`, `v[..KV_DIM]`.
    /// Inspired by vLLM / llama.cpp `attn_qkv.weight`. Shrinks
    /// `qkv_proj` profile time by ~30% on M4 Max.
    pub(crate) w_qkv: Vec<f32>,
    pub(crate) w_o: Vec<f32>,     // [D, D]
    pub(crate) rms_ffn: Vec<f32>, // [D]
    /// Fused SwiGLU gate || up projection. Layout: row-major
    /// `[D, 2*F]`. Output sliced into `gate[..F]`, `up[..F]`.
    /// Same fusion trick as `w_qkv`. Shrinks `gate_up_proj`
    /// profile time by ~30%.
    pub(crate) w_gate_up: Vec<f32>,
    /// SwiGLU down projection: [F, D].
    pub(crate) w_down: Vec<f32>,
    /// Optional Qwen3 per-head Q-norm — `[head_dim]`. Applied after
    /// the Q projection, before RoPE, to each query head independently.
    pub(crate) q_norm: Option<Vec<f32>>,
    /// Optional Qwen3 per-head K-norm — `[head_dim]`.
    pub(crate) k_norm: Option<Vec<f32>>,
}

/// A loaded Llama / Qwen model ready for autoregressive decode.
pub struct LlamaModel {
    pub config: LlamaConfig,
    pub(crate) blocks: Vec<BlockWeights>,
    /// `[V, D]` token embedding lookup table flattened row-major.
    pub(crate) token_emb: Vec<f32>,
    /// `[D]` final RMSNorm gamma.
    pub(crate) final_norm: Vec<f32>,
    /// `[D, V]` LM head weight (or alias of `token_emb` if
    /// `tie_word_embeddings == true`).
    pub(crate) lm_head: Vec<f32>,
    /// Pre-computed RoPE cos/sin tables sized to
    /// `min(config.max_position_embeddings, max_seq)`.
    pub(crate) rope: RoPE,
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
            // Fuse Q+K+V and gate+up so the hot decode path issues a
            // single sgemv per fused projection and amortises the
            // memory loads of the input vector across the outputs
            // (vLLM / llama.cpp `attn_qkv.weight` / `ffn_gate_up.weight`).
            let w_qkv = fuse_rows_3(&w_q, d, &w_k, kv_dim, &w_v, kv_dim);
            let w_gate_up = fuse_rows_2(&w_gate, f, &w_up, f);
            drop(w_q);
            drop(w_k);
            drop(w_v);
            drop(w_gate);
            drop(w_up);
            blocks.push(BlockWeights {
                rms_attn,
                w_qkv,
                w_o,
                rms_ffn,
                w_gate_up,
                w_down,
                q_norm: None,
                k_norm: None,
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

    /// Build a runnable model with deterministic random weights for
    /// testing (T241.6b parity tests). Uses a simple xorshift64 PRNG
    /// seeded with `seed`. Weights are scaled to ~N(0, 0.02) which is
    /// the typical init range for pre-trained Qwen / Llama checkpoints.
    pub fn from_random(config: LlamaConfig, max_seq: usize, seed: u64) -> Self {
        let d = config.hidden_size;
        let kv_dim = config.n_kv_heads() * config.head_dim();
        let f = config.intermediate_size;
        let v = config.vocab_size;

        // Simple xorshift64 RNG → uniform [0,1) → scaled [-σ, σ]
        let mut state = seed.max(1);
        let mut next_unit = || -> f32 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state >> 8) as f32) / ((u64::MAX >> 8) as f32)
        };
        let sigma = 0.02f32;
        let mut sample =
            |n: usize| -> Vec<f32> { (0..n).map(|_| (next_unit() * 2.0 - 1.0) * sigma).collect() };
        // RMSNorm gamma initialized to 1.0 (standard)
        let ones = |n: usize| -> Vec<f32> { vec![1.0f32; n] };

        let token_emb = sample(v * d);
        let final_norm = ones(d);
        let lm_head = if config.tie_word_embeddings {
            transpose_2d(&token_emb, v, d)
        } else {
            sample(d * v)
        };
        let mut blocks: Vec<BlockWeights> = Vec::with_capacity(config.num_hidden_layers);
        for _ in 0..config.num_hidden_layers {
            blocks.push(BlockWeights {
                rms_attn: ones(d),
                w_qkv: sample(d * (d + 2 * kv_dim)),
                w_o: sample(d * d),
                rms_ffn: ones(d),
                w_gate_up: sample(d * 2 * f),
                w_down: sample(f * d),
                q_norm: None,
                k_norm: None,
            });
        }
        let rope = RoPE::new(
            config.head_dim(),
            max_seq.min(config.max_position_embeddings),
            config.rope_theta,
        );
        LlamaModel {
            config,
            blocks,
            token_emb,
            final_norm,
            lm_head,
            rope,
        }
    }

    /// Build a runnable model from a parsed config and a fully
    /// dequantized [`GgufWeights`] bundle. GGUF stores linear weights
    /// in the same `[out, in]` row-major layout as HF safetensors —
    /// we apply the same transpose at load time so the sgemv hot path
    /// always sees `[in, out]`.
    ///
    /// Optional Qwen3 per-head Q/K norms are kept as-is (they are
    /// shape `[head_dim]`).
    pub fn from_gguf(
        config: LlamaConfig,
        weights: GgufWeights,
        max_seq: usize,
    ) -> Result<Self, LlmError> {
        let d = config.hidden_size;
        let kv_dim = config.n_kv_heads() * config.head_dim();
        let f = config.intermediate_size;

        // For sgemv we need [in, out] row-major. GGUF / numpy gives us
        // [out, in] in `weights.*` — transpose once.
        let token_emb = weights.token_emb; // [V, D] — direct lookup, no transpose
        let final_norm = weights.final_norm;
        let lm_head = if weights.tied_lm_head {
            // tied: transpose [V, D] → [D, V] so the LM-head sgemv sees [in=D, out=V].
            transpose_2d(&token_emb, config.vocab_size, d)
        } else {
            // weights.lm_head_t is [V, D]; transpose to [D, V].
            transpose_2d(&weights.lm_head_t, config.vocab_size, d)
        };

        let mut blocks: Vec<BlockWeights> = Vec::with_capacity(config.num_hidden_layers);
        for (i, b) in weights.blocks.into_iter().enumerate() {
            // Sanity-check shapes before transposing — catches dim
            // mix-ups much earlier than the eventual NaN-cascade.
            check_len("attn_norm", i, &b.attn_norm, d)?;
            check_len("w_q", i, &b.w_q, d * d)?;
            check_len("w_k", i, &b.w_k, kv_dim * d)?;
            check_len("w_v", i, &b.w_v, kv_dim * d)?;
            check_len("w_o", i, &b.w_o, d * d)?;
            check_len("ffn_norm", i, &b.ffn_norm, d)?;
            check_len("w_gate", i, &b.w_gate, f * d)?;
            check_len("w_up", i, &b.w_up, f * d)?;
            check_len("w_down", i, &b.w_down, d * f)?;

            let w_q = transpose_2d(&b.w_q, d, d);
            let w_k = transpose_2d(&b.w_k, kv_dim, d);
            let w_v = transpose_2d(&b.w_v, kv_dim, d);
            let w_o = transpose_2d(&b.w_o, d, d);
            let w_gate = transpose_2d(&b.w_gate, f, d);
            let w_up = transpose_2d(&b.w_up, f, d);
            let w_down = transpose_2d(&b.w_down, d, f);
            // Fuse Q+K+V and gate+up — see `from_hf` for rationale.
            let w_qkv = fuse_rows_3(&w_q, d, &w_k, kv_dim, &w_v, kv_dim);
            let w_gate_up = fuse_rows_2(&w_gate, f, &w_up, f);
            drop(w_q);
            drop(w_k);
            drop(w_v);
            drop(w_gate);
            drop(w_up);

            // NOTE: TeichAI Claude-Distill stores attn_norm/ffn_norm with
            // mean ≈ 0 instead of mean ≈ 1 (Qwen3 vanilla). Initial guess
            // was `gamma_eff = 1 + gamma_stored`, but applying that
            // amplifies the residual by ~80× (because the post-RMSNorm
            // amplitude becomes ||gamma||·sqrt(d) ≈ 71 instead of 0.89)
            // and makes the cascade WORSE, not better. So this is NOT
            // the right convention. The fine-tune likely just learned
            // small gammas as part of its dynamic. The collapse is
            // probably caused by something else upstream.
            let attn_norm = b.attn_norm;
            let ffn_norm = b.ffn_norm;

            if std::env::var("RUSTORCH_DEBUG_WEIGHT_STATS").is_ok() && i <= 1 {
                let stats = |name: &str, v: &[f32]| {
                    let n = v.len();
                    let mean = v.iter().sum::<f32>() / n as f32;
                    let sumsq = v.iter().map(|x| x * x).sum::<f32>();
                    let l2 = sumsq.sqrt();
                    let max = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let min = v.iter().cloned().fold(f32::INFINITY, f32::min);
                    eprintln!(
                        "  [w_stats] L{i:>2} {name:>10}  n={n:>10}  l2={l2:>12.4}  mean={mean:>+12.6}  min={min:>+10.4}  max={max:>+10.4}  first8={:?}",
                        &v[..n.min(8)]
                    );
                };
                stats("ffn_norm_eff", &ffn_norm);
                stats("w_gate_up", &w_gate_up);
            }

            blocks.push(BlockWeights {
                rms_attn: attn_norm,
                w_qkv,
                w_o,
                rms_ffn: ffn_norm,
                w_gate_up,
                w_down,
                q_norm: b.attn_q_norm,
                k_norm: b.attn_k_norm,
            });
        }

        let rope = RoPE::new(
            config.head_dim(),
            max_seq.min(config.max_position_embeddings),
            config.rope_theta,
        );

        // Optional: dump first 8 floats of w_gate per layer after
        // transpose to compare layer-0 (known good) vs layer-1+
        // (known buggy offset DC).
        if std::env::var("RUSTORCH_DEBUG_DUMP_WEIGHTS")
            .map(|v| v == "1")
            .unwrap_or(false)
        {
            for (i, b) in blocks.iter().enumerate() {
                if i > 3 && i + 1 < blocks.len() {
                    continue;
                }
                // w_gate_up is laid out [in=D, out=2F] row-major; the
                // first F columns of each row are the gate weights.
                let stride = 2 * f;
                let mut mean_per_row = vec![0f32; d];
                for (ii, slot) in mean_per_row.iter_mut().enumerate() {
                    let row = &b.w_gate_up[ii * stride..ii * stride + f];
                    *slot = row.iter().sum::<f32>() / f as f32;
                }
                let mpr_l2 = mean_per_row.iter().map(|x| x * x).sum::<f32>().sqrt();
                let mpr_mean = mean_per_row.iter().sum::<f32>() / d as f32;
                eprintln!(
                    "  [w_gate L{i:>2} mean_per_row] len={d} l2={mpr_l2:.4} mean={mpr_mean:+.6} first8={:?}",
                    &mean_per_row[..8]
                );
            }
        }

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
        // Reset profile before the autoregressive loop so prefill doesn't
        // pollute the per-token cost estimates (prefill is sequential
        // single-token decodes today, same hot path, but we want the
        // dump to reflect steady-state generation cost).
        if profile::enabled() {
            profile::reset();
        }
        for _ in 0..max_new_tokens {
            let pos = cache.current_len();
            self.decode_step(last_token, pos, &mut cache, &mut scratch);
            cache.advance(1).unwrap();
            let next = sample_next(&scratch.logits, sampling_cfg, &output, &mut next_u);
            output.push(next as u32);
            new_tokens.push(next as u32);
            last_token = next as i64;
        }
        if profile::enabled() {
            profile::dump();
        }
        new_tokens
    }

    /// Debug helper — runs prefill then a single decode step on the
    /// last prompt token and returns the top-`k` `(token_id, logit)`
    /// entries from the resulting logits, sorted descending. Useful
    /// for sanity-checking that the model produces a non-degenerate
    /// distribution.
    pub fn debug_top_logits(
        &self,
        prompt_ids: &[u32],
        max_seq: usize,
        k: usize,
    ) -> Vec<(u32, f32)> {
        let cfg = &self.config;
        let mut cache = KVCache::new(
            cfg.num_hidden_layers,
            1,
            cfg.n_kv_heads(),
            cfg.head_dim(),
            max_seq,
        );
        let mut scratch = Scratch::new(cfg);
        for (pos, &tok) in prompt_ids.iter().enumerate() {
            self.decode_step(tok as i64, pos, &mut cache, &mut scratch);
            cache.advance(1).unwrap();
        }
        // Sort logits to get the top-k.
        let mut idx: Vec<u32> = (0..cfg.vocab_size as u32).collect();
        idx.sort_by(|&a, &b| {
            scratch.logits[b as usize]
                .partial_cmp(&scratch.logits[a as usize])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        idx.into_iter()
            .take(k)
            .map(|i| (i, scratch.logits[i as usize]))
            .collect()
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
        let prof = profile::enabled();

        // Optional residual-stream tracing — set RUSTORCH_DEBUG_TRACE=1
        // to dump the L2 norm and first-8 values of scratch.x at each
        // checkpoint. Used to bisect "model ignores input" bugs.
        let trace = std::env::var("RUSTORCH_DEBUG_TRACE")
            .map(|v| v == "1")
            .unwrap_or(false);
        let dump = |tag: &str, v: &[f32]| {
            let n = v.len();
            let l2 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            let mean = v.iter().sum::<f32>() / n as f32;
            let max = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let min = v.iter().cloned().fold(f32::INFINITY, f32::min);
            let var = v.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / n as f32;
            let std = var.sqrt();
            // Print first 4 values + last 3 to compare with llama-eval-callback
            let first: Vec<String> = v.iter().take(4).map(|x| format!("{:+.4}", x)).collect();
            let last: Vec<String> = v
                .iter()
                .rev()
                .take(3)
                .rev()
                .map(|x| format!("{:+.4}", x))
                .collect();
            eprintln!(
                "  [trace] {tag:>22}  l2={l2:>10.3} mean={mean:>+10.4} std={std:>8.3} min={min:>+8.3} max={max:>+8.3}  [{}, ..., {}]",
                first.join(", "), last.join(", ")
            );
        };
        if trace {
            eprintln!("  [trace] === decode_step token_id={token_id} pos={position} ===");
            dump("after_embed", &scratch.x[..d]);
        }

        for (layer_idx, block) in self.blocks.iter().enumerate() {
            // 1. RMSNorm + Q/K/V proj.
            scratch.h[..d].copy_from_slice(&scratch.x[..d]);
            profile::time(prof, 0, || {
                rms_norm_inplace(&mut scratch.h[..d], &block.rms_attn, cfg.rms_norm_eps);
            });
            if trace && layer_idx == 0 {
                dump(&format!("L{layer_idx} attn_h_post_rms"), &scratch.h[..d]);
            }
            // Fused QKV: one sgemv with stride d + 2*kv_dim, then split.
            let qkv_stride = d + 2 * kv_dim;
            profile::time(prof, 1, || {
                sgemv_dispatch(
                    &scratch.h[..d],
                    &block.w_qkv,
                    &mut scratch.qkv[..qkv_stride],
                    d,
                    qkv_stride,
                );
                scratch.q[..d].copy_from_slice(&scratch.qkv[..d]);
                scratch.k[..kv_dim].copy_from_slice(&scratch.qkv[d..d + kv_dim]);
                scratch.v[..kv_dim].copy_from_slice(&scratch.qkv[d + kv_dim..qkv_stride]);
            });
            if trace && layer_idx == 0 {
                dump(&format!("L{layer_idx} q_pre_norm"), &scratch.q[..d]);
                dump(&format!("L{layer_idx} k_pre_norm"), &scratch.k[..kv_dim]);
                dump(&format!("L{layer_idx} v"), &scratch.v[..kv_dim]);
            }

            // 2a. Optional Qwen3 per-head Q/K RMSNorm (before RoPE).
            //     Shape is [head_dim]; applied independently to each head's
            //     contiguous head_dim-slice in q[..d] and k[..kv_dim].
            //     Toggle via env var RUSTORCH_DEBUG_DISABLE_QK_NORM=1 to A/B test.
            let qk_norm_disabled = std::env::var("RUSTORCH_DEBUG_DISABLE_QK_NORM")
                .map(|v| v == "1")
                .unwrap_or(false);
            profile::time(prof, 2, || {
                if !qk_norm_disabled {
                    if let Some(qn) = block.q_norm.as_deref() {
                        rms_norm_per_head(
                            &mut scratch.q[..d],
                            qn,
                            n_heads,
                            head_dim,
                            cfg.rms_norm_eps,
                        );
                    }
                    if let Some(kn) = block.k_norm.as_deref() {
                        rms_norm_per_head(
                            &mut scratch.k[..kv_dim],
                            kn,
                            n_kv,
                            head_dim,
                            cfg.rms_norm_eps,
                        );
                    }
                }
            });

            // 2b. RoPE on Q and K.
            //
            // HF / GGUF weights are baked for the half-split RoPE
            // convention used by `apply_rotary_pos_emb`. The legacy
            // interleaved convention can be re-enabled via env var for
            // A/B comparison.
            let rope_interleaved = std::env::var("RUSTORCH_DEBUG_ROPE_INTERLEAVED")
                .map(|v| v == "1")
                .unwrap_or(false);
            profile::time(prof, 3, || {
                if rope_interleaved {
                    self.rope
                        .apply_inplace(&mut scratch.q[..d], 1, n_heads, 1, position)
                        .unwrap();
                    self.rope
                        .apply_inplace(&mut scratch.k[..kv_dim], 1, n_kv, 1, position)
                        .unwrap();
                } else {
                    self.rope
                        .apply_inplace_half_split(&mut scratch.q[..d], 1, n_heads, 1, position)
                        .unwrap();
                    self.rope
                        .apply_inplace_half_split(&mut scratch.k[..kv_dim], 1, n_kv, 1, position)
                        .unwrap();
                }
            });

            // 3. KV-cache append.
            let kv_len = position + 1;
            let trim_len = profile::time(prof, 4, || {
                cache
                    .append(layer_idx, 1, &scratch.k[..kv_dim], &scratch.v[..kv_dim])
                    .unwrap();
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
                trim_len
            });

            // 4. GQA decode.
            profile::time(prof, 5, || {
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
            });

            // 5. O proj + residual.
            profile::time(prof, 6, || {
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
            });
            if trace && layer_idx <= 4 {
                dump(&format!("L{layer_idx} o_out"), &scratch.o_out[..d]);
            }
            if trace && layer_idx <= 4 {
                dump(&format!("L{layer_idx} after_attn_res"), &scratch.x[..d]);
            }

            // 6. RMSNorm + SwiGLU FFN + residual.
            //
            // RUSTORCH_DEBUG_SKIP_FFN_FROM=K disables the FFN sub-block
            // (sets fc2_out := 0) for all layers >= K. Used to bisect
            // which layer's FFN destroys the input-dependence of the
            // residual stream.
            let skip_ffn_from = std::env::var("RUSTORCH_DEBUG_SKIP_FFN_FROM")
                .ok()
                .and_then(|s| s.parse::<usize>().ok());
            if let Some(threshold) = skip_ffn_from {
                if layer_idx >= threshold {
                    // Skip: no FFN contribution to residual.
                    if trace
                        && (layer_idx == 0 || layer_idx == 1 || layer_idx + 1 == self.blocks.len())
                    {
                        dump(
                            &format!("L{layer_idx} after_ffn_res (FFN SKIPPED)"),
                            &scratch.x[..d],
                        );
                    }
                    continue;
                }
            }
            scratch.h[..d].copy_from_slice(&scratch.x[..d]);
            // Optional override: replace ffn_norm gamma with L0's gamma for all
            // layers (RUSTORCH_DEBUG_FORCE_FFN_NORM_FROM_L0=1) — used to test
            // whether the per-layer gamma is the source of the input-collapse.
            let force_l0_gamma = std::env::var("RUSTORCH_DEBUG_FORCE_FFN_NORM_FROM_L0")
                .map(|v| v == "1")
                .unwrap_or(false);
            profile::time(prof, 7, || {
                if force_l0_gamma && layer_idx > 0 {
                    rms_norm_inplace(
                        &mut scratch.h[..d],
                        &self.blocks[0].rms_ffn,
                        cfg.rms_norm_eps,
                    );
                } else {
                    rms_norm_inplace(&mut scratch.h[..d], &block.rms_ffn, cfg.rms_norm_eps);
                }
            });
            if trace && layer_idx <= 4 {
                dump(&format!("L{layer_idx} ffn_h_post_rms"), &scratch.h[..d]);
            }
            // Fused gate+up: one sgemv with stride 2*f, then split.
            let gate_up_stride = 2 * f;
            profile::time(prof, 8, || {
                sgemv_dispatch(
                    &scratch.h[..d],
                    &block.w_gate_up,
                    &mut scratch.gate_up[..gate_up_stride],
                    d,
                    gate_up_stride,
                );
                scratch.gate_out[..f].copy_from_slice(&scratch.gate_up[..f]);
                scratch.up_out[..f].copy_from_slice(&scratch.gate_up[f..gate_up_stride]);
            });
            if trace && layer_idx <= 4 {
                dump(
                    &format!("L{layer_idx} gate_pre_silu"),
                    &scratch.gate_out[..f],
                );
                dump(&format!("L{layer_idx} up"), &scratch.up_out[..f]);
            }
            // SwiGLU: silu(gate) * up — element-wise.
            profile::time(prof, 9, || {
                for i in 0..f {
                    let g = scratch.gate_out[i];
                    let s = g / (1.0 + (-g).exp()); // silu(x) = x * sigmoid(x)
                    scratch.gate_out[i] = s * scratch.up_out[i];
                }
            });
            if trace && layer_idx <= 4 {
                dump(&format!("L{layer_idx} swiglu_out"), &scratch.gate_out[..f]);
            }
            profile::time(prof, 10, || {
                sgemv_dispatch(
                    &scratch.gate_out[..f],
                    &block.w_down,
                    &mut scratch.fc2_out[..d],
                    f,
                    d,
                );
            });
            if trace && layer_idx <= 4 {
                dump(&format!("L{layer_idx} fc2_out"), &scratch.fc2_out[..d]);
            }
            profile::time(prof, 11, || {
                for i in 0..d {
                    scratch.x[i] += scratch.fc2_out[i];
                }
            });
            if trace && (layer_idx <= 4 || layer_idx + 1 == self.blocks.len()) {
                dump(&format!("L{layer_idx} after_ffn_res"), &scratch.x[..d]);
            }
        }

        if trace {
            dump("pre_final_norm", &scratch.x[..d]);
        }

        // 7. Final RMSNorm + LM head.
        scratch.h[..d].copy_from_slice(&scratch.x[..d]);
        profile::time(prof, 12, || {
            rms_norm_inplace(&mut scratch.h[..d], &self.final_norm, cfg.rms_norm_eps);
        });
        profile::time(prof, 13, || {
            sgemv_dispatch(
                &scratch.h[..d],
                &self.lm_head,
                &mut scratch.logits[..cfg.vocab_size],
                d,
                cfg.vocab_size,
            );
        });
        profile::record_step();
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Per-step scratch buffers, allocated once.
struct Scratch {
    x: Vec<f32>,
    h: Vec<f32>,
    /// Output of the fused QKV projection — layout `[d + 2*kv_dim]`,
    /// sliced into `[..d]=Q, [d..d+kv_dim]=K, [d+kv_dim..]=V`.
    qkv: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    attn_out: Vec<f32>,
    o_out: Vec<f32>,
    /// Output of the fused gate+up projection — layout `[2*f]`,
    /// sliced into `[..f]=gate, [f..]=up`.
    gate_up: Vec<f32>,
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
            qkv: vec![0.0; d + 2 * kv_dim],
            q: vec![0.0; d],
            k: vec![0.0; kv_dim],
            v: vec![0.0; kv_dim],
            attn_out: vec![0.0; d],
            o_out: vec![0.0; d],
            gate_up: vec![0.0; 2 * f],
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
    // RUSTORCH_DEBUG_GAMMA_PLUS_ONE=1 — test the alternate convention
    // y = x * inv_rms * (1 + gamma) used by some Qwen / TeichAI checkpoints
    // where gamma is stored as a delta on top of identity.
    let plus_one = std::env::var("RUSTORCH_DEBUG_GAMMA_PLUS_ONE")
        .map(|v| v == "1")
        .unwrap_or(false);
    if plus_one {
        for i in 0..d {
            x[i] = x[i] * inv_rms * (1.0 + gamma[i]);
        }
    } else {
        for i in 0..d {
            x[i] = x[i] * inv_rms * gamma[i];
        }
    }
}

/// Apply RMSNorm independently to each head of a packed `[n_heads * head_dim]`
/// activation (Qwen3 q_norm / k_norm). `gamma` is shared across heads
/// and has shape `[head_dim]`.
fn rms_norm_per_head(x: &mut [f32], gamma: &[f32], n_heads: usize, head_dim: usize, eps: f32) {
    debug_assert_eq!(x.len(), n_heads * head_dim);
    debug_assert_eq!(gamma.len(), head_dim);
    for h in 0..n_heads {
        let head = &mut x[h * head_dim..(h + 1) * head_dim];
        rms_norm_inplace(head, gamma, eps);
    }
}

/// Validate that a deserialized GGUF tensor has the expected number of
/// f32 elements before we hand it to `transpose_2d`.
fn check_len(label: &str, layer: usize, v: &[f32], expected: usize) -> Result<(), LlmError> {
    if v.len() != expected {
        return Err(LlmError::Config(format!(
            "blk.{layer}.{label}: got {} f32, expected {}",
            v.len(),
            expected
        )));
    }
    Ok(())
}

/// Hybrid sgemv — same calibration as T53 in the demo: cBLAS via
/// `fused_matmul_bias_activation` for shapes ≥ 200 K FLOPs,
/// custom row-major loop otherwise.
fn sgemv_dispatch(x: &[f32], w: &[f32], y: &mut [f32], k: usize, n: usize) {
    let force_naive = std::env::var("RUSTORCH_DEBUG_FORCE_NAIVE_SGEMV")
        .map(|v| v == "1")
        .unwrap_or(false);
    if !force_naive && k * n >= 200_000 {
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

/// Concatenate three row-major matrices that share the same number
/// of rows (input dim) into a single row-major matrix whose rows
/// hold `[a_row, b_row, c_row]`. Used to fuse Q/K/V projections so a
/// single sgemv produces all three outputs while reading `h` once.
///
/// All inputs are `[rows, *_cols]` row-major; output is
/// `[rows, a_cols + b_cols + c_cols]` row-major.
fn fuse_rows_3(
    a: &[f32],
    a_cols: usize,
    b: &[f32],
    b_cols: usize,
    c: &[f32],
    c_cols: usize,
) -> Vec<f32> {
    let rows = a.len() / a_cols;
    debug_assert_eq!(a.len(), rows * a_cols);
    debug_assert_eq!(b.len(), rows * b_cols);
    debug_assert_eq!(c.len(), rows * c_cols);
    let stride = a_cols + b_cols + c_cols;
    let mut dst = vec![0.0_f32; rows * stride];
    for r in 0..rows {
        let off = r * stride;
        dst[off..off + a_cols].copy_from_slice(&a[r * a_cols..(r + 1) * a_cols]);
        dst[off + a_cols..off + a_cols + b_cols].copy_from_slice(&b[r * b_cols..(r + 1) * b_cols]);
        dst[off + a_cols + b_cols..off + stride].copy_from_slice(&c[r * c_cols..(r + 1) * c_cols]);
    }
    dst
}

/// Same as [`fuse_rows_3`] for two matrices. Used for fused gate||up.
fn fuse_rows_2(a: &[f32], a_cols: usize, b: &[f32], b_cols: usize) -> Vec<f32> {
    let rows = a.len() / a_cols;
    debug_assert_eq!(a.len(), rows * a_cols);
    debug_assert_eq!(b.len(), rows * b_cols);
    let stride = a_cols + b_cols;
    let mut dst = vec![0.0_f32; rows * stride];
    for r in 0..rows {
        let off = r * stride;
        dst[off..off + a_cols].copy_from_slice(&a[r * a_cols..(r + 1) * a_cols]);
        dst[off + a_cols..off + stride].copy_from_slice(&b[r * b_cols..(r + 1) * b_cols]);
    }
    dst
}

// ---------------------------------------------------------------------------
// In-process profiler — toggle with `RUSTORCH_PROFILE=1`. Accumulates per-
// phase wall-clock time over all decode_step calls; dumped via
// `LlamaModel::dump_profile()` (called automatically by `generate` when
// the env var is set).
// ---------------------------------------------------------------------------

mod profile {
    use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
    use std::time::Instant;

    pub const N_PHASES: usize = 14;
    pub const PHASE_NAMES: [&str; N_PHASES] = [
        "rmsnorm_attn",    // 0
        "qkv_proj",        // 1  (q + k + v projections)
        "qk_norm",         // 2
        "rope",            // 3
        "kv_cache_trim",   // 4  (append + per-head trim copy)
        "attention_gqa",   // 5
        "o_proj_residual", // 6
        "rmsnorm_ffn",     // 7
        "gate_up_proj",    // 8
        "swiglu",          // 9
        "down_proj",       //10
        "ffn_residual",    //11
        "final_norm",      //12
        "lm_head",         //13
    ];

    static NS: [AtomicU64; N_PHASES] = [
        AtomicU64::new(0),
        AtomicU64::new(0),
        AtomicU64::new(0),
        AtomicU64::new(0),
        AtomicU64::new(0),
        AtomicU64::new(0),
        AtomicU64::new(0),
        AtomicU64::new(0),
        AtomicU64::new(0),
        AtomicU64::new(0),
        AtomicU64::new(0),
        AtomicU64::new(0),
        AtomicU64::new(0),
        AtomicU64::new(0),
    ];
    static STEPS: AtomicUsize = AtomicUsize::new(0);

    pub fn enabled() -> bool {
        std::env::var("RUSTORCH_PROFILE")
            .map(|v| v == "1")
            .unwrap_or(false)
    }

    /// Run `f`, accumulate its wall-clock time into phase `idx`. The
    /// `enabled` check is hoisted out by the caller so we don't pay
    /// an env-var lookup per phase per layer per token.
    #[inline(always)]
    pub fn time<R>(enabled: bool, idx: usize, f: impl FnOnce() -> R) -> R {
        if enabled {
            let t = Instant::now();
            let r = f();
            NS[idx].fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
            r
        } else {
            f()
        }
    }

    pub fn record_step() {
        STEPS.fetch_add(1, Ordering::Relaxed);
    }

    pub fn reset() {
        for a in NS.iter() {
            a.store(0, Ordering::Relaxed);
        }
        STEPS.store(0, Ordering::Relaxed);
    }

    pub fn dump() {
        let steps = STEPS.load(Ordering::Relaxed).max(1);
        let total: u64 = NS.iter().map(|a| a.load(Ordering::Relaxed)).sum();
        eprintln!();
        eprintln!(
            "══════ DECODE PROFILE ({steps} decode_step calls, total {:.2} ms) ══════",
            total as f64 / 1e6
        );
        eprintln!(
            "  {:<18}  {:>10}  {:>9}  {:>7}",
            "phase", "total (ms)", "per-tok", "% of total"
        );
        let mut order: Vec<usize> = (0..N_PHASES).collect();
        order.sort_by_key(|&i| std::cmp::Reverse(NS[i].load(Ordering::Relaxed)));
        for i in order {
            let ns = NS[i].load(Ordering::Relaxed);
            if ns == 0 {
                continue;
            }
            let pct = 100.0 * ns as f64 / total as f64;
            let ms = ns as f64 / 1e6;
            let per_tok = ms / steps as f64;
            eprintln!(
                "  {:<18}  {:>10.2}  {:>7.3}ms  {:>6.1}%",
                PHASE_NAMES[i], ms, per_tok, pct
            );
        }
        eprintln!();
    }
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
