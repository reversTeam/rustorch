//! Llama-style autoregressive decode demo (T47/T48).
//!
//! Builds a mini Llama (4 layers, RMSNorm + GQA + RoPE + FFN) with
//! random weights, then runs a 64-token generation loop on a 16-token
//! prompt. Demonstrates the integration of the LLM-runner bricks
//! shipped in T41 (KV-cache), T42 (RoPE), T43 (sampling), T45 (GQA).
//!
//! Run with:
//!     cargo run --release --example llama_decode_demo
//!
//! T48 — the entire decode path is now raw `Vec<f32>` / `&[f32]`,
//! with no Tensor or Variable wrappers across step boundaries. This
//! eliminates the per-layer allocator + autograd-dispatch overhead
//! that the initial T47 implementation paid (roughly 110 us per
//! layer on this shape), and brings the per-step decode latency
//! below PyTorch's equivalent.

use rustorch_fusion::patterns::matmul_bias_act::{fused_matmul_bias_activation, Activation};
use rustorch_nn::gqa::gqa_forward_f32;
use rustorch_nn::kv_cache::KVCache;
use rustorch_nn::rope::RoPE;
use rustorch_nn::sampling::{sample_next, SamplingConfig};
use std::time::Instant;

// Mini-Llama config.
const NUM_LAYERS: usize = 4;
const D_MODEL: usize = 256;
const N_HEADS: usize = 8;
const N_KV_HEADS: usize = 2; // 4-to-1 GQA ratio (Qwen-style)
const HEAD_DIM: usize = D_MODEL / N_HEADS; // 32
const KV_DIM: usize = N_KV_HEADS * HEAD_DIM; // 64
const D_FF: usize = 1024;
const VOCAB: usize = 4096;
const MAX_SEQ: usize = 128;
const RMS_EPS: f32 = 1e-6;

const PROMPT_LEN: usize = 16;
const MAX_NEW_TOKENS: usize = 64;

/// Per-block weights stored as raw f32 buffers — no Tensor/Variable
/// wrap. `linear_d_d`, `linear_d_kv`, `linear_d_ff`, `linear_ff_d`
/// shapes follow the Llama convention (no bias).
struct LlamaBlock {
    rms_attn: Vec<f32>, // [D]
    w_q: Vec<f32>,      // [D, D]
    w_k: Vec<f32>,      // [D, KV_DIM]
    w_v: Vec<f32>,      // [D, KV_DIM]
    w_o: Vec<f32>,      // [D, D]
    rms_ffn: Vec<f32>,  // [D]
    w_fc1: Vec<f32>,    // [D, F]
    w_fc2: Vec<f32>,    // [F, D]
}

/// Deterministic LCG noise scaled to `[-bound, bound]`.
fn lcg_init(n: usize, seed: u64, bound: f32) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            let u = (s >> 33) as u32;
            let f = (u as f32) / (u32::MAX as f32);
            (f * 2.0 - 1.0) * bound
        })
        .collect()
}

impl LlamaBlock {
    fn new(layer_idx: usize) -> Self {
        let bound_d = 1.0 / (D_MODEL as f32).sqrt();
        let bound_f = 1.0 / (D_FF as f32).sqrt();
        let seed_base = (layer_idx as u64) * 0xDEADBEEF + 1;
        LlamaBlock {
            rms_attn: vec![1.0_f32; D_MODEL],
            w_q: lcg_init(D_MODEL * D_MODEL, seed_base + 1, bound_d),
            w_k: lcg_init(D_MODEL * KV_DIM, seed_base + 2, bound_d),
            w_v: lcg_init(D_MODEL * KV_DIM, seed_base + 3, bound_d),
            w_o: lcg_init(D_MODEL * D_MODEL, seed_base + 4, bound_d),
            rms_ffn: vec![1.0_f32; D_MODEL],
            w_fc1: lcg_init(D_MODEL * D_FF, seed_base + 5, bound_d),
            w_fc2: lcg_init(D_FF * D_MODEL, seed_base + 6, bound_f),
        }
    }
}

/// In-place RMSNorm of a single row of length `D` with scale `gamma`.
#[inline]
fn rms_norm_inplace(x: &mut [f32], gamma: &[f32]) {
    let d = x.len();
    let inv_d = 1.0_f32 / d as f32;
    let mut sq_sum = 0.0_f32;
    for &v in x.iter() {
        sq_sum += v * v;
    }
    let inv_rms = 1.0 / (sq_sum * inv_d + RMS_EPS).sqrt();
    for i in 0..d {
        x[i] = x[i] * inv_rms * gamma[i];
    }
}

/// Reusable scratch buffers across decode steps. Allocated once.
struct DecodeScratch {
    x: Vec<f32>,        // [D]
    h: Vec<f32>,        // [D]
    q: Vec<f32>,        // [D]
    k: Vec<f32>,        // [KV_DIM]
    v: Vec<f32>,        // [KV_DIM]
    attn_out: Vec<f32>, // [D]
    o_out: Vec<f32>,    // [D]
    fc1_out: Vec<f32>,  // [F]
    fc2_out: Vec<f32>,  // [D]
    logits: Vec<f32>,   // [V]
    k_trim: Vec<f32>,   // [N_KV_HEADS * MAX_SEQ * HEAD_DIM] re-used
    v_trim: Vec<f32>,   // same
}

impl DecodeScratch {
    fn new() -> Self {
        DecodeScratch {
            x: vec![0.0_f32; D_MODEL],
            h: vec![0.0_f32; D_MODEL],
            q: vec![0.0_f32; D_MODEL],
            k: vec![0.0_f32; KV_DIM],
            v: vec![0.0_f32; KV_DIM],
            attn_out: vec![0.0_f32; D_MODEL],
            o_out: vec![0.0_f32; D_MODEL],
            fc1_out: vec![0.0_f32; D_FF],
            fc2_out: vec![0.0_f32; D_MODEL],
            logits: vec![0.0_f32; VOCAB],
            k_trim: vec![0.0_f32; N_KV_HEADS * MAX_SEQ * HEAD_DIM],
            v_trim: vec![0.0_f32; N_KV_HEADS * MAX_SEQ * HEAD_DIM],
        }
    }
}

/// Embedding lookup: copy row `token_id` of `tok_emb_weight` into
/// `dst[..D_MODEL]`.
fn embed_lookup(tok_emb_weight: &[f32], token_id: usize, dst: &mut [f32]) {
    let off = token_id * D_MODEL;
    dst.copy_from_slice(&tok_emb_weight[off..off + D_MODEL]);
}

/// One full decode step in pure raw f32 — no Tensor/Variable. Writes
/// the next-token logits into `scratch.logits`.
#[allow(clippy::too_many_arguments)]
fn decode_step_raw(
    token_id: i64,
    position: usize,
    blocks: &[LlamaBlock],
    final_ln_w: &[f32],
    lm_head_w: &[f32],
    tok_emb_weight: &[f32],
    rope: &RoPE,
    cache: &mut KVCache,
    scratch: &mut DecodeScratch,
) {
    embed_lookup(tok_emb_weight, token_id as usize, &mut scratch.x);

    for (layer_idx, block) in blocks.iter().enumerate() {
        // 1. RMSNorm into scratch.h.
        scratch.h.copy_from_slice(&scratch.x);
        rms_norm_inplace(&mut scratch.h, &block.rms_attn);

        // 2. Q/K/V projections via fused matmul (M=1, K=D, N varies).
        fused_matmul_bias_activation(
            &scratch.h,
            &block.w_q,
            None,
            &mut scratch.q,
            1,
            D_MODEL,
            D_MODEL,
            Activation::None,
        )
        .unwrap();
        fused_matmul_bias_activation(
            &scratch.h,
            &block.w_k,
            None,
            &mut scratch.k,
            1,
            D_MODEL,
            KV_DIM,
            Activation::None,
        )
        .unwrap();
        fused_matmul_bias_activation(
            &scratch.h,
            &block.w_v,
            None,
            &mut scratch.v,
            1,
            D_MODEL,
            KV_DIM,
            Activation::None,
        )
        .unwrap();

        // 3. RoPE on Q (n_heads heads) and K (n_kv_heads heads).
        rope.apply_inplace(&mut scratch.q, 1, N_HEADS, 1, position)
            .unwrap();
        rope.apply_inplace(&mut scratch.k, 1, N_KV_HEADS, 1, position)
            .unwrap();

        // 4. Append to KV-cache.
        cache.append(layer_idx, 1, &scratch.k, &scratch.v).unwrap();

        // 5. Build trimmed [N_KV_HEADS, kv_len, HEAD_DIM] from cache.
        let kv_len = position + 1;
        let k_full = cache.k_buffer(layer_idx).unwrap();
        let v_full = cache.v_buffer(layer_idx).unwrap();
        for kv_h in 0..N_KV_HEADS {
            let src_off = kv_h * MAX_SEQ * HEAD_DIM;
            let dst_off = kv_h * kv_len * HEAD_DIM;
            scratch.k_trim[dst_off..dst_off + kv_len * HEAD_DIM]
                .copy_from_slice(&k_full[src_off..src_off + kv_len * HEAD_DIM]);
            scratch.v_trim[dst_off..dst_off + kv_len * HEAD_DIM]
                .copy_from_slice(&v_full[src_off..src_off + kv_len * HEAD_DIM]);
        }

        // 6. GQA forward: q [1, H, 1, hd] over k/v [1, KV, kv_len, hd].
        gqa_forward_f32(
            &scratch.q,
            &scratch.k_trim[..N_KV_HEADS * kv_len * HEAD_DIM],
            &scratch.v_trim[..N_KV_HEADS * kv_len * HEAD_DIM],
            &mut scratch.attn_out,
            1,
            N_HEADS,
            N_KV_HEADS,
            1,
            kv_len,
            HEAD_DIM,
        )
        .unwrap();

        // 7. O proj + residual add.
        fused_matmul_bias_activation(
            &scratch.attn_out,
            &block.w_o,
            None,
            &mut scratch.o_out,
            1,
            D_MODEL,
            D_MODEL,
            Activation::None,
        )
        .unwrap();
        for d in 0..D_MODEL {
            scratch.x[d] += scratch.o_out[d];
        }

        // 8. RMSNorm + FFN (linear+relu+linear) + residual.
        scratch.h.copy_from_slice(&scratch.x);
        rms_norm_inplace(&mut scratch.h, &block.rms_ffn);
        fused_matmul_bias_activation(
            &scratch.h,
            &block.w_fc1,
            None,
            &mut scratch.fc1_out,
            1,
            D_MODEL,
            D_FF,
            Activation::Relu,
        )
        .unwrap();
        fused_matmul_bias_activation(
            &scratch.fc1_out,
            &block.w_fc2,
            None,
            &mut scratch.fc2_out,
            1,
            D_FF,
            D_MODEL,
            Activation::None,
        )
        .unwrap();
        for d in 0..D_MODEL {
            scratch.x[d] += scratch.fc2_out[d];
        }
    }

    // 9. Final RMSNorm + LM head.
    scratch.h.copy_from_slice(&scratch.x);
    rms_norm_inplace(&mut scratch.h, final_ln_w);
    fused_matmul_bias_activation(
        &scratch.h,
        lm_head_w,
        None,
        &mut scratch.logits,
        1,
        D_MODEL,
        VOCAB,
        Activation::None,
    )
    .unwrap();
}

fn main() {
    println!();
    println!("=========================================================");
    println!(" RusTorch — mini-Llama decode loop (T47/T48 raw)");
    println!("=========================================================");
    println!();
    println!(
        "  Architecture : {} layers, GQA {}/{} heads, D={} F={}",
        NUM_LAYERS, N_HEADS, N_KV_HEADS, D_MODEL, D_FF
    );
    println!("  Vocab        : {}, max_seq {}", VOCAB, MAX_SEQ);
    println!("  Prompt len   : {}", PROMPT_LEN);
    println!("  Decode steps : {}", MAX_NEW_TOKENS);
    println!();

    println!("  Building model...");
    let t_build = Instant::now();
    let blocks: Vec<LlamaBlock> = (0..NUM_LAYERS).map(LlamaBlock::new).collect();
    let final_ln_w = vec![1.0_f32; D_MODEL];
    let bound_d = 1.0 / (D_MODEL as f32).sqrt();
    let lm_head_w = lcg_init(D_MODEL * VOCAB, 0xC0DE, bound_d);
    let tok_emb_weight = lcg_init(VOCAB * D_MODEL, 0xCAFE, 1.0);
    let rope = RoPE::new(HEAD_DIM, MAX_SEQ, 10000.0);
    println!("  Build time   : {:.2} s", t_build.elapsed().as_secs_f64());

    let mut cache = KVCache::new(NUM_LAYERS, 1, N_KV_HEADS, HEAD_DIM, MAX_SEQ);
    let mut scratch = DecodeScratch::new();
    let mut output: Vec<u32> = Vec::with_capacity(PROMPT_LEN + MAX_NEW_TOKENS);
    let sampling_cfg = SamplingConfig::greedy();
    let mut s: u64 = 1;
    let mut next_u = move || {
        s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        ((s >> 32) as f32) / (u32::MAX as f32)
    };

    let prompt: Vec<i64> = (0..PROMPT_LEN as i64)
        .map(|i| (i * 17 + 3) % VOCAB as i64)
        .collect();
    println!("  Prompt token ids: {:?}", &prompt);
    println!();

    println!("  Prefill ({} tokens)...", PROMPT_LEN);
    let t_prefill = Instant::now();
    for (pos, &tok) in prompt.iter().enumerate() {
        decode_step_raw(
            tok,
            pos,
            &blocks,
            &final_ln_w,
            &lm_head_w,
            &tok_emb_weight,
            &rope,
            &mut cache,
            &mut scratch,
        );
        cache.advance(1).unwrap();
        output.push(tok as u32);
    }
    let prefill_ms = t_prefill.elapsed().as_secs_f64() * 1000.0;
    println!(
        "  Prefill time : {:.2} ms ({:.2} ms/token)",
        prefill_ms,
        prefill_ms / PROMPT_LEN as f64
    );
    println!();

    println!("  Decode (greedy, {} new tokens):", MAX_NEW_TOKENS);
    let mut step_times_us: Vec<u128> = Vec::with_capacity(MAX_NEW_TOKENS);
    let mut last_token = output[output.len() - 1] as i64;
    for step in 0..MAX_NEW_TOKENS {
        let t0 = Instant::now();
        let pos = cache.current_len();
        decode_step_raw(
            last_token,
            pos,
            &blocks,
            &final_ln_w,
            &lm_head_w,
            &tok_emb_weight,
            &rope,
            &mut cache,
            &mut scratch,
        );
        cache.advance(1).unwrap();
        let next = sample_next(&scratch.logits, &sampling_cfg, &output, &mut next_u);
        let elapsed = t0.elapsed();
        step_times_us.push(elapsed.as_micros());
        last_token = next as i64;
        output.push(next as u32);
        if step < 4 || step == MAX_NEW_TOKENS - 1 {
            println!(
                "    step {:>2}: {:>7.2} ms   token = {:>5}",
                step + 1,
                elapsed.as_secs_f64() * 1000.0,
                next
            );
        } else if step == 4 {
            println!("    ...");
        }
    }
    println!();

    step_times_us.sort();
    let median_us = step_times_us[step_times_us.len() / 2];
    let mean_us = step_times_us.iter().sum::<u128>() / step_times_us.len() as u128;
    let p99_us = step_times_us[(step_times_us.len() * 99 / 100).min(step_times_us.len() - 1)];
    println!("=========================================================");
    println!(" Decode summary");
    println!("---------------------------------------------------------");
    println!("  Median per step : {:>7.2} ms", median_us as f64 / 1000.0);
    println!("  Mean per step   : {:>7.2} ms", mean_us as f64 / 1000.0);
    println!("  p99 per step    : {:>7.2} ms", p99_us as f64 / 1000.0);
    println!(
        "  Tokens/sec      : {:>7}",
        1_000_000_u128 / median_us.max(1)
    );
    println!("=========================================================");
    println!();

    println!("  Final sequence ({} tokens): {:?}", output.len(), &output);
    println!();
}
