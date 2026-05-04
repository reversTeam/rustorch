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

// T51 NOTE: tested a direct cblas_sgemm wrapper to skip the
// fused_matmul_bias_activation shape-check + epilogue path.
// Result: SLOWER on M=1 K=N=256 (Q/K/V/O proj) because the
// fused dispatcher routes those tiny shapes (98K FLOPs) to a
// scalar 3-loop kernel at L1d-resident speed (~13 µs), while
// cblas_sgemv takes ~20 µs (FFI dispatch overhead). Reverted.

/// T52 — custom row-major friendly sgemv for M=1.
///
/// The default `fused_matmul_bias_activation` scalar kernel is
///   for col in 0..N: for k in 0..K: acc += x[k] * W[k*N + col]
/// — which strides W by `N` on every inner iteration, defeating
/// the L1d prefetcher (256 KB weight, 256 cols × 256 k accesses
/// each ~1 cache line apart in DRAM).
///
/// This kernel rotates the loop order to outer-k / inner-n:
///   y.fill(0); for k in 0..K: for n in 0..N: y[n] += x[k] * W[k*N + n]
/// Each k iteration scans `W[k*N..(k+1)*N]` *contiguously*,
/// streaming the weight matrix exactly once and letting LLVM
/// emit `fmla.4s` over the inner `n` loop. Brings DRAM bandwidth
/// utilisation close to peak.
#[inline(always)]
fn sgemv_m1_rowmajor(x: &[f32], w: &[f32], y: &mut [f32], k: usize, n: usize) {
    debug_assert_eq!(x.len(), k);
    debug_assert_eq!(w.len(), k * n);
    debug_assert_eq!(y.len(), n);
    y.fill(0.0);
    for kk in 0..k {
        let xk = x[kk];
        let w_row = &w[kk * n..(kk + 1) * n];
        for nn in 0..n {
            y[nn] += xk * w_row[nn];
        }
    }
}

/// T52 — sgemv with fused ReLU epilogue (for FC1).
#[inline(always)]
fn sgemv_m1_rowmajor_relu(x: &[f32], w: &[f32], y: &mut [f32], k: usize, n: usize) {
    sgemv_m1_rowmajor(x, w, y, k, n);
    for v in y.iter_mut() {
        if *v < 0.0 {
            *v = 0.0;
        }
    }
}

/// T53 — hybrid sgemv dispatcher. The custom row-major kernel is
/// optimal for small shapes (Q/K/V/O proj where K=N=256 fits L2),
/// but loses to cBLAS sgemv on larger shapes (FFN K=256 N=1024,
/// LM head K=256 N=4096) where the BLAS micro-kernel is ~10×
/// faster. Threshold 200 K FLOPs (= roughly the FFN crossover on
/// M-series).
#[inline]
fn sgemv_m1_dispatch(
    x: &[f32],
    w: &[f32],
    y: &mut [f32],
    k: usize,
    n: usize,
    activation: Activation,
) {
    let flops = k * n;
    if flops >= 200_000 {
        // Large path: route to cBLAS via the fused dispatcher.
        fused_matmul_bias_activation(x, w, None, y, 1, k, n, activation).unwrap();
    } else {
        // Small path: row-major sgemv kernel.
        match activation {
            Activation::Relu => sgemv_m1_rowmajor_relu(x, w, y, k, n),
            _ => sgemv_m1_rowmajor(x, w, y, k, n),
        }
    }
}

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
/// wrap. T49: Q/K/V projections are merged into a single
/// `w_qkv` matrix [D, D + 2*KV_DIM] so one sgemm produces the
/// concatenated output, saving 2 cBLAS FFI calls per layer plus
/// improving cache locality.
struct LlamaBlock {
    rms_attn: Vec<f32>, // [D]
    /// Concatenated [W_Q | W_K | W_V] of shape [D, D + 2*KV_DIM].
    w_qkv: Vec<f32>,
    w_o: Vec<f32>,     // [D, D]
    rms_ffn: Vec<f32>, // [D]
    w_fc1: Vec<f32>,   // [D, F]
    w_fc2: Vec<f32>,   // [F, D]
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

/// Combined Q/K/V output dim. T49 fused-QKV layout.
const QKV_DIM: usize = D_MODEL + 2 * KV_DIM; // 256 + 64 + 64 = 384

impl LlamaBlock {
    fn new(layer_idx: usize) -> Self {
        let bound_d = 1.0 / (D_MODEL as f32).sqrt();
        let bound_f = 1.0 / (D_FF as f32).sqrt();
        let seed_base = (layer_idx as u64) * 0xDEADBEEF + 1;
        // T49: build W_QKV as a single contiguous [D, QKV_DIM]
        // matrix so one fused matmul produces concatenated outputs.
        // The layout (row-major) is: for each input row d in 0..D,
        //   first D output cols hold W_Q row,
        //   next KV_DIM cols hold W_K row,
        //   last KV_DIM cols hold W_V row.
        let w_q_init = lcg_init(D_MODEL * D_MODEL, seed_base + 1, bound_d);
        let w_k_init = lcg_init(D_MODEL * KV_DIM, seed_base + 2, bound_d);
        let w_v_init = lcg_init(D_MODEL * KV_DIM, seed_base + 3, bound_d);
        let mut w_qkv = vec![0.0_f32; D_MODEL * QKV_DIM];
        for d in 0..D_MODEL {
            let dst_row = d * QKV_DIM;
            // W_Q [D, D]: row d → cols [0, D)
            w_qkv[dst_row..dst_row + D_MODEL]
                .copy_from_slice(&w_q_init[d * D_MODEL..(d + 1) * D_MODEL]);
            // W_K [D, KV_DIM]: row d → cols [D, D + KV_DIM)
            w_qkv[dst_row + D_MODEL..dst_row + D_MODEL + KV_DIM]
                .copy_from_slice(&w_k_init[d * KV_DIM..(d + 1) * KV_DIM]);
            // W_V [D, KV_DIM]: row d → cols [D + KV_DIM, QKV_DIM)
            w_qkv[dst_row + D_MODEL + KV_DIM..dst_row + QKV_DIM]
                .copy_from_slice(&w_v_init[d * KV_DIM..(d + 1) * KV_DIM]);
        }
        LlamaBlock {
            rms_attn: vec![1.0_f32; D_MODEL],
            w_qkv,
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
    qkv: Vec<f32>,      // [QKV_DIM] — fused Q|K|V output (T49)
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
            qkv: vec![0.0_f32; QKV_DIM],
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

        // 2. T52 — fused Q/K/V projection via the row-major sgemv
        //    kernel. One contiguous scan over W_QKV (D × QKV_DIM
        //    = 384 KB) at peak DRAM bandwidth.
        sgemv_m1_dispatch(
            &scratch.h,
            &block.w_qkv,
            &mut scratch.qkv,
            D_MODEL,
            QKV_DIM,
            Activation::None,
        );
        // Split scratch.qkv into Q [D], K [KV_DIM], V [KV_DIM].
        let (q_slice, kv_slice) = scratch.qkv.split_at_mut(D_MODEL);
        let (k_slice, v_slice) = kv_slice.split_at_mut(KV_DIM);

        // 3. RoPE on Q (n_heads heads) and K (n_kv_heads heads).
        rope.apply_inplace(q_slice, 1, N_HEADS, 1, position)
            .unwrap();
        rope.apply_inplace(k_slice, 1, N_KV_HEADS, 1, position)
            .unwrap();

        // 4. Append to KV-cache.
        cache.append(layer_idx, 1, k_slice, v_slice).unwrap();

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
        // We borrow Q from the QKV split slice (still in `scratch.qkv`).
        let q_slice_ro = &scratch.qkv[..D_MODEL];
        gqa_forward_f32(
            q_slice_ro,
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

        // 7. O proj + residual add (T53 hybrid sgemv).
        sgemv_m1_dispatch(
            &scratch.attn_out,
            &block.w_o,
            &mut scratch.o_out,
            D_MODEL,
            D_MODEL,
            Activation::None,
        );
        for d in 0..D_MODEL {
            scratch.x[d] += scratch.o_out[d];
        }

        // 8. RMSNorm + FFN (linear+relu+linear) + residual.
        scratch.h.copy_from_slice(&scratch.x);
        rms_norm_inplace(&mut scratch.h, &block.rms_ffn);
        sgemv_m1_dispatch(
            &scratch.h,
            &block.w_fc1,
            &mut scratch.fc1_out,
            D_MODEL,
            D_FF,
            Activation::Relu,
        );
        sgemv_m1_dispatch(
            &scratch.fc1_out,
            &block.w_fc2,
            &mut scratch.fc2_out,
            D_FF,
            D_MODEL,
            Activation::None,
        );
        for d in 0..D_MODEL {
            scratch.x[d] += scratch.fc2_out[d];
        }
    }

    // 9. Final RMSNorm + LM head.
    scratch.h.copy_from_slice(&scratch.x);
    rms_norm_inplace(&mut scratch.h, final_ln_w);
    sgemv_m1_dispatch(
        &scratch.h,
        lm_head_w,
        &mut scratch.logits,
        D_MODEL,
        VOCAB,
        Activation::None,
    );
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

    // ------------------ Per-stage profile ---------------------------
    // Profile a single decode step at a fixed position to identify
    // the dominant cost (matches the PyTorch bench's single-step
    // measurement at context=32).
    println!("=========================================================");
    println!(" Per-stage profile (single step, context=32)");
    println!("---------------------------------------------------------");
    let mut profile_cache = KVCache::new(NUM_LAYERS, 1, N_KV_HEADS, HEAD_DIM, MAX_SEQ);
    let mut profile_scratch = DecodeScratch::new();
    // Pre-warm cache to position 32.
    for pos in 0..32 {
        decode_step_raw(
            42,
            pos,
            &blocks,
            &final_ln_w,
            &lm_head_w,
            &tok_emb_weight,
            &rope,
            &mut profile_cache,
            &mut profile_scratch,
        );
        profile_cache.advance(1).unwrap();
    }
    // Measure one step many times for stable timings.
    const PROFILE_ITERS: usize = 50;
    let mut total_us = 0_u128;
    let mut stage_us = [0_u128; 7]; // [embed, qkv, rope, gqa+trim, oproj+resid, ffn, lmhead]
    for _ in 0..PROFILE_ITERS {
        let t_total = Instant::now();
        // Inline copy of decode_step_raw with per-stage timers.
        embed_lookup(&tok_emb_weight, 42_usize, &mut profile_scratch.x);
        for (layer_idx, block) in blocks.iter().enumerate() {
            let t = Instant::now();
            profile_scratch.h.copy_from_slice(&profile_scratch.x);
            rms_norm_inplace(&mut profile_scratch.h, &block.rms_attn);
            sgemv_m1_dispatch(
                &profile_scratch.h,
                &block.w_qkv,
                &mut profile_scratch.qkv,
                D_MODEL,
                QKV_DIM,
                Activation::None,
            );
            stage_us[1] += t.elapsed().as_nanos();
            let t = Instant::now();
            let (q_slice, kv_slice) = profile_scratch.qkv.split_at_mut(D_MODEL);
            let (k_slice, v_slice) = kv_slice.split_at_mut(KV_DIM);
            rope.apply_inplace(q_slice, 1, N_HEADS, 1, 32).unwrap();
            rope.apply_inplace(k_slice, 1, N_KV_HEADS, 1, 32).unwrap();
            stage_us[2] += t.elapsed().as_nanos();
            let t = Instant::now();
            profile_cache
                .append(layer_idx, 1, k_slice, v_slice)
                .unwrap();
            let kv_len = 33;
            let k_full = profile_cache.k_buffer(layer_idx).unwrap();
            let v_full = profile_cache.v_buffer(layer_idx).unwrap();
            for kv_h in 0..N_KV_HEADS {
                let src_off = kv_h * MAX_SEQ * HEAD_DIM;
                let dst_off = kv_h * kv_len * HEAD_DIM;
                profile_scratch.k_trim[dst_off..dst_off + kv_len * HEAD_DIM]
                    .copy_from_slice(&k_full[src_off..src_off + kv_len * HEAD_DIM]);
                profile_scratch.v_trim[dst_off..dst_off + kv_len * HEAD_DIM]
                    .copy_from_slice(&v_full[src_off..src_off + kv_len * HEAD_DIM]);
            }
            let q_slice_ro = &profile_scratch.qkv[..D_MODEL];
            gqa_forward_f32(
                q_slice_ro,
                &profile_scratch.k_trim[..N_KV_HEADS * kv_len * HEAD_DIM],
                &profile_scratch.v_trim[..N_KV_HEADS * kv_len * HEAD_DIM],
                &mut profile_scratch.attn_out,
                1,
                N_HEADS,
                N_KV_HEADS,
                1,
                kv_len,
                HEAD_DIM,
            )
            .unwrap();
            stage_us[3] += t.elapsed().as_nanos();
            let t = Instant::now();
            sgemv_m1_dispatch(
                &profile_scratch.attn_out,
                &block.w_o,
                &mut profile_scratch.o_out,
                D_MODEL,
                D_MODEL,
                Activation::None,
            );
            for d in 0..D_MODEL {
                profile_scratch.x[d] += profile_scratch.o_out[d];
            }
            stage_us[4] += t.elapsed().as_nanos();
            let t = Instant::now();
            profile_scratch.h.copy_from_slice(&profile_scratch.x);
            rms_norm_inplace(&mut profile_scratch.h, &block.rms_ffn);
            sgemv_m1_dispatch(
                &profile_scratch.h,
                &block.w_fc1,
                &mut profile_scratch.fc1_out,
                D_MODEL,
                D_FF,
                Activation::Relu,
            );
            sgemv_m1_dispatch(
                &profile_scratch.fc1_out,
                &block.w_fc2,
                &mut profile_scratch.fc2_out,
                D_FF,
                D_MODEL,
                Activation::None,
            );
            for d in 0..D_MODEL {
                profile_scratch.x[d] += profile_scratch.fc2_out[d];
            }
            stage_us[5] += t.elapsed().as_nanos();
        }
        let t = Instant::now();
        profile_scratch.h.copy_from_slice(&profile_scratch.x);
        rms_norm_inplace(&mut profile_scratch.h, &final_ln_w);
        sgemv_m1_dispatch(
            &profile_scratch.h,
            &lm_head_w,
            &mut profile_scratch.logits,
            D_MODEL,
            VOCAB,
            Activation::None,
        );
        stage_us[6] += t.elapsed().as_nanos();
        total_us += t_total.elapsed().as_micros();
    }
    let avg_us = total_us / PROFILE_ITERS as u128;
    println!(
        "  Single step at ctx=32 : {:>7.2} ms (avg over {} iters)",
        avg_us as f64 / 1000.0,
        PROFILE_ITERS
    );
    println!("  Per-stage breakdown (avg per step, ns):");
    let stages = [
        "",
        "QKV proj (×4 layers)",
        "RoPE (×4)",
        "KV trim+GQA (×4)",
        "O proj+residual (×4)",
        "FFN (×4)",
        "Final LN+LM head",
    ];
    for i in 1..7 {
        let avg_ns = stage_us[i] / PROFILE_ITERS as u128;
        println!(
            "    {:<28}: {:>8} ns ({:.2}% of step)",
            stages[i],
            avg_ns,
            100.0 * avg_ns as f64 / (avg_us * 1000) as f64
        );
    }
    println!("=========================================================");
    println!();
}
