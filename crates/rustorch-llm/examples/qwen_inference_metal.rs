//! `qwen_inference_metal` — end-to-end Qwen3 decode using our
//! T71/T72 Metal Q4_K / Q6_K matmul kernels for the heavy
//! projections, plus the existing CPU rustorch-nn primitives for
//! everything else (RMSNorm, RoPE, GQA, SwiGLU). The point is to
//! measure end-to-end tok/s with our **own** kernels (no candle,
//! no llama.cpp, no ggml) on the way to beating MLX.
//!
//! Status: MVP. Each Metal sgemv is dispatched + drained synchronously
//! per call to keep the CPU-side scratch logic simple. A follow-up
//! commit will batch the GPU commands across a layer and skip the
//! drain so we don't pay 6 sync points per layer × 40 layers.
//!
//! Usage:
//!   cargo run --release -p rustorch-llm --example qwen_inference_metal -- \
//!       --model ~/models/Qwen3-14B-Claude-4.5-Opus-Distill.q4_k_m.gguf \
//!       --prompt-ids 12522,5193,264,882,11,1052,572,264,2613,25105,879 \
//!       --n 50

#![cfg(target_os = "macos")]
#![allow(dead_code)] // some scratch buffers are kept around for the
                     // CPU fallback path that's swapped in/out across
                     // commits as we port more ops to Metal kernels.

use std::env;
use std::process::ExitCode;
use std::time::{Duration, Instant};

use rustorch_gguf::{GgmlType, GgufFile};
use rustorch_metal::backend::MetalBackend;
use rustorch_metal::backend_singleton::metal_backend;
use rustorch_metal::kernels::{
    add_inplace_f32, gqa_decode_f32, kv_append_f32, rms_norm_f32, rms_norm_per_head_f32,
    rope_half_split_f32, sgemv_q4_k_f32_into, sgemv_q4_k_f32_pair_into,
    sgemv_q4_k_f32_pair_quadcoop_into, sgemv_q4_k_f32_simdcoop_nr2_into,
    sgemv_q4_k_f32_triple_quadcoop_into, sgemv_q6_k_f32_into, sgemv_q6_k_f32_simdcoop_into,
    swiglu_f32,
};

use metal::Buffer;

use rustorch_nn::rope::RoPE;

/// One projection's worth of weights, kept GPU-resident.
struct MetalWeight {
    buffer: Buffer,
    k: usize,
    n: usize,
    dtype: GgmlType,
}

impl MetalWeight {
    fn matmul_into(&self, backend: &MetalBackend, x_buf: &Buffer, out_buf: &Buffer) {
        // Three-way kernel selection:
        //   - blocks_per_row ≥ 32 (K ≥ 8192): simdcoop saturates the simd-
        //     group nicely (32 threads, 32+ blocks of useful work each).
        //   - blocks_per_row ∈ [16, 32) and Q4_K and n > 500: quadcoop
        //     (T84 — 4 outputs per simdgroup, 8 threads K-coop per output)
        //     beats simdcoop because plain simdcoop wastes 32 - bpr threads.
        //   - else: 1-thread-per-output simple kernel.
        let blocks_per_row = self.k / 256;
        match (self.dtype, blocks_per_row, self.n) {
            // T89 — multi-row simdcoop (2 rows per simdgroup, port of llama.cpp's
            // N_R0_Q4_K=2). For Q4_K with bpr ≥ 16 and large N (sgemv shapes
            // where x bytes >> W bytes per row), sharing 1 x-tile across 2
            // rows halves the x memory bandwidth pressure.
            (GgmlType::Q4_K, bpr, n) if bpr >= 16 && n > 500 => sgemv_q4_k_f32_simdcoop_nr2_into(
                backend,
                x_buf,
                &self.buffer,
                out_buf,
                self.k,
                self.n,
            )
            .unwrap(),
            (GgmlType::Q4_K, _, _) => {
                sgemv_q4_k_f32_into(backend, x_buf, &self.buffer, out_buf, self.k, self.n).unwrap()
            },
            // T90 reverted: Q6_K simdcoop_nr2 regressed -23% on Qwen3-14B
            // (likely register spill — too many per-row scales held). Stay
            // on single-row simdcoop for W_down.
            (GgmlType::Q6_K, bpr, n) if bpr >= 16 && n > 500 => {
                sgemv_q6_k_f32_simdcoop_into(backend, x_buf, &self.buffer, out_buf, self.k, self.n)
                    .unwrap()
            },
            (GgmlType::Q6_K, _, _) => {
                sgemv_q6_k_f32_into(backend, x_buf, &self.buffer, out_buf, self.k, self.n).unwrap()
            },
            _ => panic!("unsupported dtype: {:?}", self.dtype),
        }
    }
}

struct LayerWeights {
    attn_norm: Vec<f32>,   // [d] CPU copy (for QK norm fallback)
    attn_norm_buf: Buffer, // GPU copy
    w_q: MetalWeight,
    w_k: MetalWeight,
    w_v: MetalWeight,
    w_o: MetalWeight,
    ffn_norm: Vec<f32>,
    ffn_norm_buf: Buffer,
    w_gate: MetalWeight,
    w_up: MetalWeight,
    w_down: MetalWeight,
    attn_q_norm: Option<Vec<f32>>,
    attn_k_norm: Option<Vec<f32>>,
    attn_q_norm_buf: Option<Buffer>,
    attn_k_norm_buf: Option<Buffer>,
}

struct ModelMetal {
    cfg: ModelCfg,
    layers: Vec<LayerWeights>,
    token_emb: Vec<f32>, // [V, D] — kept f32 for cheap lookup
    final_norm: Vec<f32>,
    final_norm_buf: Buffer,
    lm_head: MetalWeight, // Q6_K usually
    rope: RoPE,
    rope_cos_buf: Buffer, // [max_seq, head_dim/2] f32
    rope_sin_buf: Buffer,
}

#[derive(Clone, Copy)]
struct ModelCfg {
    d: usize,
    f: usize,
    n_layers: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    vocab: usize,
    rms_eps: f32,
    rope_theta: f32,
    max_seq: usize,
}

fn alloc_metal_from_bytes(backend: &MetalBackend, bytes: &[u8]) -> Buffer {
    let buf = backend.alloc_shared(bytes.len()).unwrap();
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), buf.contents() as *mut u8, bytes.len());
    }
    buf
}

fn load_f32(file: &GgufFile, name: &str) -> Vec<f32> {
    let t = file
        .tensor(name)
        .unwrap_or_else(|| panic!("missing tensor: {name}"));
    let bytes = file.tensor_bytes(t);
    rustorch_gguf::dequant_to_f32(t, bytes).unwrap()
}

fn load_metal_weight(backend: &MetalBackend, file: &GgufFile, name: &str) -> MetalWeight {
    let t = file
        .tensor(name)
        .unwrap_or_else(|| panic!("missing tensor: {name}"));
    let bytes = file.tensor_bytes(t);
    // GGUF native layout [N, K/256, blk_bytes] row-major matches our
    // kernel's expected layout — each thread (= one output column)
    // walks K/256 contiguous super-blocks for its row. Tested
    // alternative: pre-transpose to [K/256, N, blk_bytes] (block-major)
    // — turned out to REGRESS perf by ~6% because the per-thread
    // 144-byte chunks now jump N*144 (~5MB) bytes between blocks,
    // killing the GPU prefetcher. Sticking with native layout.
    let k = t.shape[0] as usize;
    let n = t.shape[1] as usize;
    MetalWeight {
        buffer: alloc_metal_from_bytes(backend, bytes),
        k,
        n,
        dtype: t.dtype,
    }
}

fn rms_norm(x: &mut [f32], gamma: &[f32], eps: f32) {
    let d = x.len();
    let inv_d = 1.0_f32 / d as f32;
    let sq = x.iter().map(|v| v * v).sum::<f32>();
    let inv_rms = 1.0 / (sq * inv_d + eps).sqrt();
    for i in 0..d {
        x[i] = x[i] * inv_rms * gamma[i];
    }
}

fn rms_norm_per_head(x: &mut [f32], gamma: &[f32], n_heads: usize, head_dim: usize, eps: f32) {
    for h in 0..n_heads {
        let head = &mut x[h * head_dim..(h + 1) * head_dim];
        rms_norm(head, gamma, eps);
    }
}

fn argmax(logits: &[f32]) -> u32 {
    let (i, _) =
        logits
            .iter()
            .enumerate()
            .fold((0usize, f32::NEG_INFINITY), |(bi, bv), (i, &v)| {
                if v > bv {
                    (i, v)
                } else {
                    (bi, bv)
                }
            });
    i as u32
}

// T85 — Per-stage profiling accumulator. Each field accumulates the wall
// time spent in that stage across all calls (typically all layers × all
// tokens). Each stage drains the Metal command buffer to force GPU sync
// before reading the clock — this serializes execution and inflates
// total time vs the chained path, but the relative breakdown is accurate.
#[derive(Default, Debug)]
struct Stages {
    embed: Duration,           // CPU memcpy of token embedding into xd_buf
    attn_norm: Duration,       // RMSNorm pre-attn
    qkv: Duration,             // Q/K/V matmul (triple_quadcoop or pair+single)
    qknorm_rope: Duration,     // QK-norm + RoPE on Q & K
    kv_append: Duration,       // KV cache append (K + V)
    attention: Duration,       // GQA decode
    o_residual: Duration,      // W_O matmul + residual add
    ffn_norm: Duration,        // RMSNorm pre-ffn
    gate_up: Duration,         // pair_into gate+up
    swiglu: Duration,          // SwiGLU activation
    down_residual: Duration,   // W_down matmul + residual add
    final_norm: Duration,      // Final RMSNorm
    lm_head: Duration,         // lm_head matmul
    logits_readback: Duration, // GPU→CPU logits copy + argmax
}

impl Stages {
    fn total(&self) -> Duration {
        self.embed
            + self.attn_norm
            + self.qkv
            + self.qknorm_rope
            + self.kv_append
            + self.attention
            + self.o_residual
            + self.ffn_norm
            + self.gate_up
            + self.swiglu
            + self.down_residual
            + self.final_norm
            + self.lm_head
            + self.logits_readback
    }

    fn print_breakdown(&self, n_tokens: usize) {
        let total = self.total();
        let total_ms = total.as_secs_f64() * 1000.0;
        let per_token_ms = total_ms / n_tokens as f64;
        let row = |label: &str, d: Duration| {
            let ms = d.as_secs_f64() * 1000.0;
            let pct = if total_ms > 0.0 {
                100.0 * ms / total_ms
            } else {
                0.0
            };
            let per_tok = ms / n_tokens as f64;
            println!(
                "  {:<18} {:>9.2} ms total | {:>7.3} ms/tok | {:>5.1}%",
                label, ms, per_tok, pct
            );
        };
        println!(
            "\n=== T85 stage breakdown over {} tokens (total {:.2} ms = {:.3} ms/tok) ===",
            n_tokens, total_ms, per_token_ms
        );
        row("embed", self.embed);
        row("attn_norm", self.attn_norm);
        row("qkv", self.qkv);
        row("qknorm_rope", self.qknorm_rope);
        row("kv_append", self.kv_append);
        row("attention", self.attention);
        row("o_residual", self.o_residual);
        row("ffn_norm", self.ffn_norm);
        row("gate_up", self.gate_up);
        row("swiglu", self.swiglu);
        row("down_residual", self.down_residual);
        row("final_norm", self.final_norm);
        row("lm_head", self.lm_head);
        row("logits_readback", self.logits_readback);
        println!(
            "  {:<18} {:>9.2} ms total | {:>7.3} ms/tok | {:>5.1}%",
            "TOTAL", total_ms, per_token_ms, 100.0
        );
        let proxy_tps = 1000.0 / per_token_ms;
        println!(
            "  (profiled tok/s = {:.2}, ~{:.1}× slower than chained path due to per-stage drains)",
            proxy_tps,
            proxy_tps.recip() / (1.0 / 29.27)
        );
    }
}

// T87 — Per-layer FFN sparsity statistics. After each SwiGLU dispatch we
// drain the GPU and read fd_buf (length f=17408 in Qwen3-14B), counting
// how many elements are |x| < threshold for thresholds 1e-3, 1e-2, 1e-1.
// Aggregated over all tokens × all layers to validate the DejaVu
// hypothesis that 70-90% of FFN neurons are quasi-zero.
#[derive(Default, Debug)]
struct SparsityStats {
    /// Per-layer counts indexed by layer_idx. Each tuple is
    /// (count_lt_1e3, count_lt_1e2, count_lt_1e1, total_samples_for_this_layer).
    /// total_samples = f × n_tokens_observed.
    per_layer: Vec<(u64, u64, u64, u64)>,
    f: usize,
}

impl SparsityStats {
    fn new(n_layers: usize, f: usize) -> Self {
        Self {
            per_layer: vec![(0, 0, 0, 0); n_layers],
            f,
        }
    }

    fn record(&mut self, layer_idx: usize, fd_data: &[f32]) {
        debug_assert_eq!(fd_data.len(), self.f);
        let mut c1 = 0u64;
        let mut c2 = 0u64;
        let mut c3 = 0u64;
        for &v in fd_data {
            let av = v.abs();
            if av < 1e-3 {
                c1 += 1;
            }
            if av < 1e-2 {
                c2 += 1;
            }
            if av < 1e-1 {
                c3 += 1;
            }
        }
        let entry = &mut self.per_layer[layer_idx];
        entry.0 += c1;
        entry.1 += c2;
        entry.2 += c3;
        entry.3 += self.f as u64;
    }

    fn print_breakdown(&self) {
        println!(
            "\n=== T87 FFN sparsity breakdown ({} layers, f={}) ===",
            self.per_layer.len(),
            self.f
        );
        println!(
            "  {:<6} {:>12} {:>12} {:>12}",
            "layer", "%<1e-3", "%<1e-2", "%<1e-1"
        );
        let mut sum1 = 0.0f64;
        let mut sum2 = 0.0f64;
        let mut sum3 = 0.0f64;
        let mut max1 = 0.0f64;
        let mut max2 = 0.0f64;
        let mut max3 = 0.0f64;
        let mut min1 = 100.0f64;
        let mut min2 = 100.0f64;
        let mut min3 = 100.0f64;
        for (li, &(c1, c2, c3, total)) in self.per_layer.iter().enumerate() {
            if total == 0 {
                continue;
            }
            let p1 = 100.0 * c1 as f64 / total as f64;
            let p2 = 100.0 * c2 as f64 / total as f64;
            let p3 = 100.0 * c3 as f64 / total as f64;
            sum1 += p1;
            sum2 += p2;
            sum3 += p3;
            max1 = max1.max(p1);
            max2 = max2.max(p2);
            max3 = max3.max(p3);
            min1 = min1.min(p1);
            min2 = min2.min(p2);
            min3 = min3.min(p3);
            println!("  {:<6} {:>11.2}% {:>11.2}% {:>11.2}%", li, p1, p2, p3);
        }
        let n = self.per_layer.len() as f64;
        println!(
            "  {:<6} {:>11.2}% {:>11.2}% {:>11.2}% (mean)",
            "----",
            sum1 / n,
            sum2 / n,
            sum3 / n
        );
        println!(
            "  {:<6} {:>11.2}% {:>11.2}% {:>11.2}% (min)",
            "min", min1, min2, min3
        );
        println!(
            "  {:<6} {:>11.2}% {:>11.2}% {:>11.2}% (max)",
            "max", max1, max2, max3
        );
        println!("\nDejaVu hypothesis check (threshold 1e-2):");
        let mean_p2 = sum2 / n;
        if mean_p2 >= 70.0 {
            println!(
                "  ✓ CONFIRMED — mean sparsity {:.1}% ≥ 70%. Proceed to T88 sparse FFN kernel.",
                mean_p2
            );
        } else if mean_p2 >= 50.0 {
            println!("  ~ MARGINAL — mean sparsity {:.1}%. Sparse FFN may pay off but proceed cautiously.", mean_p2);
        } else {
            println!("  ✗ FAILED — mean sparsity {:.1}% < 50%. DejaVu approach unlikely to help; pivot to lookahead.", mean_p2);
        }
    }
}

// T88a — N-gram hit rate analysis on a generated sequence. Replays the
// decode step by step, maintaining a trigram → continuations cache built
// from the tokens generated so far. At each step ≥ 3 we look up whether
// the cache (using the trigram of the 3 previous tokens) would have
// predicted the next K tokens correctly, simulating Lookahead Jacobi
// without actually running batched forwards.
//
// Pure CPU offline analysis on a Vec<u32> — adds no runtime cost to the
// decode itself. The result tells us empirically what acceptance rate
// Lookahead Jacobi would deliver on this exact workload.
fn analyze_ngram_hit_rate(generated: &[u32], max_n: usize) {
    use std::collections::HashMap;
    if generated.len() < 4 + max_n {
        println!(
            "\n=== T88a n-gram hit rate ===\n  Sequence too short ({} tokens), need ≥ {} for max_n={}",
            generated.len(),
            4 + max_n,
            max_n
        );
        return;
    }

    // Cache: trigram (a, b, c) → HashMap<next_token, count>.
    // Top-1 prediction is the argmax of the inner counts.
    let mut cache: HashMap<(u32, u32, u32), HashMap<u32, u32>> = HashMap::new();
    let mut hits_at_n = vec![0u64; max_n];
    let mut coverage = 0u64;
    let mut total_lookups = 0u64;
    let mut accepted_lengths: Vec<u32> = Vec::with_capacity(generated.len());

    for i in 3..generated.len().saturating_sub(max_n) {
        let trigram = (generated[i - 3], generated[i - 2], generated[i - 1]);
        total_lookups += 1;
        if let Some(continuations) = cache.get(&trigram) {
            coverage += 1;
            // Top-1 next prediction
            let next_pred = continuations
                .iter()
                .max_by_key(|(_, &c)| c)
                .map(|(&tok, _)| tok);
            let mut accepted = 0u32;
            if let Some(p0) = next_pred {
                if p0 == generated[i] {
                    accepted = 1;
                    hits_at_n[0] += 1;
                    // Continue speculative chain: at each step, build trigram from
                    // (prev_2, prev_1, accepted_token) and look up next.
                    let mut t1 = generated[i - 2];
                    let mut t2 = generated[i - 1];
                    let mut t3 = generated[i];
                    for n in 1..max_n {
                        let next_trigram = (t1, t2, t3);
                        let Some(conts) = cache.get(&next_trigram) else {
                            break;
                        };
                        let Some((&p, _)) = conts.iter().max_by_key(|(_, &c)| c) else {
                            break;
                        };
                        if p == generated[i + n] {
                            accepted += 1;
                            hits_at_n[n] += 1;
                            t1 = t2;
                            t2 = t3;
                            t3 = p;
                        } else {
                            break;
                        }
                    }
                }
            }
            accepted_lengths.push(accepted);
        } else {
            accepted_lengths.push(0);
        }

        // Update cache with the actual observed continuation.
        let entry = cache.entry(trigram).or_default();
        *entry.entry(generated[i]).or_insert(0) += 1;
    }

    let n = total_lookups.max(1) as f64;
    let cov_pct = 100.0 * coverage as f64 / n;
    let mean_accept = accepted_lengths.iter().map(|&v| v as f64).sum::<f64>()
        / accepted_lengths.len().max(1) as f64;
    let max_accept = accepted_lengths.iter().copied().max().unwrap_or(0);
    println!(
        "\n=== T88a n-gram hit rate ({} steps analysed, max_n={}) ===",
        total_lookups, max_n
    );
    println!("  coverage (cached trigram)  : {:>6.2}%", cov_pct);
    println!("  mean accepted length       : {:>6.2}", mean_accept);
    println!("  max accepted length        : {:>6}", max_accept);
    println!("  hit@N (% of all lookups, cumulative as accept length grows):");
    for (n_idx, hits) in hits_at_n.iter().enumerate().take(max_n) {
        let pct = 100.0 * *hits as f64 / total_lookups.max(1) as f64;
        println!("    accepted ≥ {:<3} : {:>6.2}%", n_idx + 1, pct);
    }
    println!("\nDecision criteria for Lookahead Jacobi (T88):");
    let hit_4 = if max_n >= 4 {
        100.0 * hits_at_n[3] as f64 / total_lookups.max(1) as f64
    } else {
        0.0
    };
    if hit_4 >= 40.0 {
        println!(
            "  ✓ CONFIRMED — hit@4 = {:.1}% ≥ 40%. Lookahead Jacobi paie. Procéder T88 full sprint.",
            hit_4
        );
    } else if hit_4 >= 20.0 {
        println!(
            "  ~ MARGINAL — hit@4 = {:.1}%. Plain trigram cache trop simple. Considérer REST (chunk-level retrieval).",
            hit_4
        );
    } else {
        println!(
            "  ✗ FAILED — hit@4 = {:.1}% < 20%. Lookahead Jacobi écarté. Pivot direct vers HSTC ou external chunk store.",
            hit_4
        );
    }
}

fn forward_token(
    backend: &MetalBackend,
    model: &ModelMetal,
    token_id: u32,
    position: usize,
    scratch: &mut Scratch,
) -> u32 {
    let cfg = model.cfg;
    let d = cfg.d;
    let f = cfg.f;
    let _kv_dim = cfg.n_kv_heads * cfg.head_dim;
    let n_heads = cfg.n_heads;
    let n_kv = cfg.n_kv_heads;
    let head_dim = cfg.head_dim;
    let max_seq = cfg.max_seq;

    // Embed (CPU lookup, written into xd_buf as residual stream).
    let off = (token_id as usize) * d;
    scratch.x.copy_from_slice(&model.token_emb[off..off + d]);
    unsafe {
        std::ptr::copy_nonoverlapping(scratch.x.as_ptr(), scratch.xd_buf.contents() as *mut f32, d);
    }

    for (li, layer) in model.layers.iter().enumerate() {
        // 1. RMSNorm GPU.
        rms_norm_f32(
            backend,
            &scratch.xd_buf,
            &layer.attn_norm_buf,
            &scratch.h_buf,
            d,
            cfg.rms_eps,
        )
        .unwrap();

        // 2. Q / K / V matmul GPU.
        //
        // In Qwen3 Q4_K_M the dtype mix is asymmetric:
        //   attn_q, attn_k -> Q4_K  (144 B / super-block)
        //   attn_v         -> Q6_K  (210 B / super-block)
        //
        // The Q4_K triple kernel can't fuse a Q6_K row (different
        // block layout) — feeding V's Q6_K bytes through the Q4_K
        // shader scrambles V completely (T80 collapse seen in tests:
        // tokens go to 0). So we fuse Q+K via pair_into (one
        // dispatch) and dispatch V separately. Net: 2 kernels per
        // attention layer instead of 3 — still a clear win over the
        // pre-T80 split path.
        // T84: quadcoop-fused QKV. With Q4_K K=5120 (bpr=20),
        // 4-output-per-simdgroup beats both simple (under-saturates GPU)
        // and simdcoop (wastes 32-bpr=12 threads per group).
        if matches!(layer.w_v.dtype, GgmlType::Q4_K) {
            sgemv_q4_k_f32_triple_quadcoop_into(
                backend,
                &scratch.h_buf,
                &layer.w_q.buffer,
                &layer.w_k.buffer,
                &layer.w_v.buffer,
                &scratch.q_buf,
                &scratch.k_buf,
                &scratch.v_buf,
                layer.w_q.k,
                layer.w_q.n,
                layer.w_k.n,
                layer.w_v.n,
            )
            .unwrap();
        } else {
            sgemv_q4_k_f32_pair_quadcoop_into(
                backend,
                &scratch.h_buf,
                &layer.w_q.buffer,
                &layer.w_k.buffer,
                &scratch.q_buf,
                &scratch.k_buf,
                layer.w_q.k,
                layer.w_q.n,
                layer.w_k.n,
            )
            .unwrap();
            layer
                .w_v
                .matmul_into(backend, &scratch.h_buf, &scratch.v_buf);
        }

        // 3. QK-norm BEFORE RoPE (Qwen3 HF reference order). All GPU.
        if let Some(qn_buf) = layer.attn_q_norm_buf.as_ref() {
            rms_norm_per_head_f32(
                backend,
                &scratch.q_buf,
                qn_buf,
                n_heads,
                head_dim,
                cfg.rms_eps,
            )
            .unwrap();
        }
        if let Some(kn_buf) = layer.attn_k_norm_buf.as_ref() {
            rms_norm_per_head_f32(backend, &scratch.k_buf, kn_buf, n_kv, head_dim, cfg.rms_eps)
                .unwrap();
        }

        // 4. RoPE GPU on Q and K (after QK norm).
        rope_half_split_f32(
            backend,
            &scratch.q_buf,
            &model.rope_cos_buf,
            &model.rope_sin_buf,
            n_heads,
            head_dim,
            position,
        )
        .unwrap();
        rope_half_split_f32(
            backend,
            &scratch.k_buf,
            &model.rope_cos_buf,
            &model.rope_sin_buf,
            n_kv,
            head_dim,
            position,
        )
        .unwrap();

        // 4. KV cache append — pure GPU kernel, no drain needed.
        kv_append_f32(
            backend,
            &scratch.k_buf,
            &scratch.k_caches[li],
            n_kv,
            head_dim,
            position,
            max_seq,
        )
        .unwrap();
        kv_append_f32(
            backend,
            &scratch.v_buf,
            &scratch.v_caches[li],
            n_kv,
            head_dim,
            position,
            max_seq,
        )
        .unwrap();

        // 5. GQA decode GPU.
        let kv_len = position + 1;
        gqa_decode_f32(
            backend,
            &scratch.q_buf,
            &scratch.k_caches[li],
            &scratch.v_caches[li],
            &scratch.attn_buf,
            n_heads,
            n_kv,
            head_dim,
            kv_len,
            max_seq,
        )
        .unwrap();

        // 6+7+8+9: O proj → residual_add → RMSNorm → gate/up → SwiGLU →
        // down → residual_add. ALL GPU, chained without drain.
        // attn_buf comes straight from the GQA kernel; no upload needed.
        layer
            .w_o
            .matmul_into(backend, &scratch.attn_buf, &scratch.o_buf);
        add_inplace_f32(backend, &scratch.xd_buf, &scratch.o_buf, d).unwrap();
        rms_norm_f32(
            backend,
            &scratch.xd_buf,
            &layer.ffn_norm_buf,
            &scratch.h_buf,
            d,
            cfg.rms_eps,
        )
        .unwrap();
        // gate + up fused into one dispatch (non-simdcoop wins at this scale).
        sgemv_q4_k_f32_pair_into(
            backend,
            &scratch.h_buf,
            &layer.w_gate.buffer,
            &layer.w_up.buffer,
            &scratch.gate_buf,
            &scratch.up_buf,
            layer.w_gate.k,
            layer.w_gate.n,
            layer.w_up.n,
        )
        .unwrap();
        swiglu_f32(
            backend,
            &scratch.gate_buf,
            &scratch.up_buf,
            &scratch.fd_buf,
            f,
        )
        .unwrap();
        layer
            .w_down
            .matmul_into(backend, &scratch.fd_buf, &scratch.fc2_buf);
        add_inplace_f32(backend, &scratch.xd_buf, &scratch.fc2_buf, d).unwrap();
    }

    // Final RMSNorm GPU + LM head GPU. Residual stream is in xd_buf;
    // we run RMSNorm into h_buf, then lm_head into logits_buf.
    rms_norm_f32(
        backend,
        &scratch.xd_buf,
        &model.final_norm_buf,
        &scratch.h_buf,
        cfg.d,
        cfg.rms_eps,
    )
    .unwrap();
    model
        .lm_head
        .matmul_into(backend, &scratch.h_buf, &scratch.logits_buf);
    backend.drain();
    let mut logits = vec![0.0_f32; cfg.vocab];
    unsafe {
        std::ptr::copy_nonoverlapping(
            scratch.logits_buf.contents() as *const f32,
            logits.as_mut_ptr(),
            cfg.vocab,
        );
    }
    argmax(&logits)
}

// T85 — Stage-instrumented forward_token. Drains the Metal command buffer
// after each named stage and accumulates the wall time into Stages. The
// per-stage drains serialize execution and slow the overall path, but
// give an accurate breakdown of where GPU time actually goes.
fn forward_token_profiled(
    backend: &MetalBackend,
    model: &ModelMetal,
    token_id: u32,
    position: usize,
    scratch: &mut Scratch,
    stages: &mut Stages,
) -> u32 {
    let cfg = model.cfg;
    let d = cfg.d;
    let f = cfg.f;
    let n_heads = cfg.n_heads;
    let n_kv = cfg.n_kv_heads;
    let head_dim = cfg.head_dim;
    let max_seq = cfg.max_seq;

    // 0. Embed (CPU memcpy).
    let t = Instant::now();
    let off = (token_id as usize) * d;
    scratch.x.copy_from_slice(&model.token_emb[off..off + d]);
    unsafe {
        std::ptr::copy_nonoverlapping(scratch.x.as_ptr(), scratch.xd_buf.contents() as *mut f32, d);
    }
    stages.embed += t.elapsed();

    for (li, layer) in model.layers.iter().enumerate() {
        // 1. attn_norm.
        let t = Instant::now();
        rms_norm_f32(
            backend,
            &scratch.xd_buf,
            &layer.attn_norm_buf,
            &scratch.h_buf,
            d,
            cfg.rms_eps,
        )
        .unwrap();
        backend.drain();
        stages.attn_norm += t.elapsed();

        // 2. QKV.
        let t = Instant::now();
        if matches!(layer.w_v.dtype, GgmlType::Q4_K) {
            sgemv_q4_k_f32_triple_quadcoop_into(
                backend,
                &scratch.h_buf,
                &layer.w_q.buffer,
                &layer.w_k.buffer,
                &layer.w_v.buffer,
                &scratch.q_buf,
                &scratch.k_buf,
                &scratch.v_buf,
                layer.w_q.k,
                layer.w_q.n,
                layer.w_k.n,
                layer.w_v.n,
            )
            .unwrap();
        } else {
            sgemv_q4_k_f32_pair_quadcoop_into(
                backend,
                &scratch.h_buf,
                &layer.w_q.buffer,
                &layer.w_k.buffer,
                &scratch.q_buf,
                &scratch.k_buf,
                layer.w_q.k,
                layer.w_q.n,
                layer.w_k.n,
            )
            .unwrap();
            layer
                .w_v
                .matmul_into(backend, &scratch.h_buf, &scratch.v_buf);
        }
        backend.drain();
        stages.qkv += t.elapsed();

        // 3. QK-norm + RoPE.
        let t = Instant::now();
        if let Some(qn_buf) = layer.attn_q_norm_buf.as_ref() {
            rms_norm_per_head_f32(
                backend,
                &scratch.q_buf,
                qn_buf,
                n_heads,
                head_dim,
                cfg.rms_eps,
            )
            .unwrap();
        }
        if let Some(kn_buf) = layer.attn_k_norm_buf.as_ref() {
            rms_norm_per_head_f32(backend, &scratch.k_buf, kn_buf, n_kv, head_dim, cfg.rms_eps)
                .unwrap();
        }
        rope_half_split_f32(
            backend,
            &scratch.q_buf,
            &model.rope_cos_buf,
            &model.rope_sin_buf,
            n_heads,
            head_dim,
            position,
        )
        .unwrap();
        rope_half_split_f32(
            backend,
            &scratch.k_buf,
            &model.rope_cos_buf,
            &model.rope_sin_buf,
            n_kv,
            head_dim,
            position,
        )
        .unwrap();
        backend.drain();
        stages.qknorm_rope += t.elapsed();

        // 4. KV append.
        let t = Instant::now();
        kv_append_f32(
            backend,
            &scratch.k_buf,
            &scratch.k_caches[li],
            n_kv,
            head_dim,
            position,
            max_seq,
        )
        .unwrap();
        kv_append_f32(
            backend,
            &scratch.v_buf,
            &scratch.v_caches[li],
            n_kv,
            head_dim,
            position,
            max_seq,
        )
        .unwrap();
        backend.drain();
        stages.kv_append += t.elapsed();

        // 5. Attention (GQA decode).
        let t = Instant::now();
        let kv_len = position + 1;
        gqa_decode_f32(
            backend,
            &scratch.q_buf,
            &scratch.k_caches[li],
            &scratch.v_caches[li],
            &scratch.attn_buf,
            n_heads,
            n_kv,
            head_dim,
            kv_len,
            max_seq,
        )
        .unwrap();
        backend.drain();
        stages.attention += t.elapsed();

        // 6. W_O + residual #1.
        let t = Instant::now();
        layer
            .w_o
            .matmul_into(backend, &scratch.attn_buf, &scratch.o_buf);
        add_inplace_f32(backend, &scratch.xd_buf, &scratch.o_buf, d).unwrap();
        backend.drain();
        stages.o_residual += t.elapsed();

        // 7. ffn_norm.
        let t = Instant::now();
        rms_norm_f32(
            backend,
            &scratch.xd_buf,
            &layer.ffn_norm_buf,
            &scratch.h_buf,
            d,
            cfg.rms_eps,
        )
        .unwrap();
        backend.drain();
        stages.ffn_norm += t.elapsed();

        // 8. gate + up (fused pair, non-simdcoop).
        let t = Instant::now();
        sgemv_q4_k_f32_pair_into(
            backend,
            &scratch.h_buf,
            &layer.w_gate.buffer,
            &layer.w_up.buffer,
            &scratch.gate_buf,
            &scratch.up_buf,
            layer.w_gate.k,
            layer.w_gate.n,
            layer.w_up.n,
        )
        .unwrap();
        backend.drain();
        stages.gate_up += t.elapsed();

        // 9. SwiGLU.
        let t = Instant::now();
        swiglu_f32(
            backend,
            &scratch.gate_buf,
            &scratch.up_buf,
            &scratch.fd_buf,
            f,
        )
        .unwrap();
        backend.drain();
        stages.swiglu += t.elapsed();

        // 10. W_down + residual #2.
        let t = Instant::now();
        layer
            .w_down
            .matmul_into(backend, &scratch.fd_buf, &scratch.fc2_buf);
        add_inplace_f32(backend, &scratch.xd_buf, &scratch.fc2_buf, d).unwrap();
        backend.drain();
        stages.down_residual += t.elapsed();
    }

    // 11. Final RMSNorm.
    let t = Instant::now();
    rms_norm_f32(
        backend,
        &scratch.xd_buf,
        &model.final_norm_buf,
        &scratch.h_buf,
        cfg.d,
        cfg.rms_eps,
    )
    .unwrap();
    backend.drain();
    stages.final_norm += t.elapsed();

    // 12. lm_head.
    let t = Instant::now();
    model
        .lm_head
        .matmul_into(backend, &scratch.h_buf, &scratch.logits_buf);
    backend.drain();
    stages.lm_head += t.elapsed();

    // 13. Logits readback + argmax.
    let t = Instant::now();
    let mut logits = vec![0.0_f32; cfg.vocab];
    unsafe {
        std::ptr::copy_nonoverlapping(
            scratch.logits_buf.contents() as *const f32,
            logits.as_mut_ptr(),
            cfg.vocab,
        );
    }
    let next = argmax(&logits);
    stages.logits_readback += t.elapsed();
    next
}

// T87 — Sparsity-instrumented forward. Identical to forward_token except
// after each SwiGLU we drain the GPU and read fd_buf to record the FFN
// sparsity at this layer. Drains are expensive (~100 µs each, 40 per
// token) — total inference ~3-4× slower than chained, fine for offline
// profiling. Only the SwiGLU output is sampled; the rest of the forward
// uses the chained path.
fn forward_token_sparsity(
    backend: &MetalBackend,
    model: &ModelMetal,
    token_id: u32,
    position: usize,
    scratch: &mut Scratch,
    stats: &mut SparsityStats,
) -> u32 {
    let cfg = model.cfg;
    let d = cfg.d;
    let f = cfg.f;
    let n_heads = cfg.n_heads;
    let n_kv = cfg.n_kv_heads;
    let head_dim = cfg.head_dim;
    let max_seq = cfg.max_seq;

    let off = (token_id as usize) * d;
    scratch.x.copy_from_slice(&model.token_emb[off..off + d]);
    unsafe {
        std::ptr::copy_nonoverlapping(scratch.x.as_ptr(), scratch.xd_buf.contents() as *mut f32, d);
    }

    let mut fd_scratch = vec![0.0_f32; f];

    for (li, layer) in model.layers.iter().enumerate() {
        rms_norm_f32(
            backend,
            &scratch.xd_buf,
            &layer.attn_norm_buf,
            &scratch.h_buf,
            d,
            cfg.rms_eps,
        )
        .unwrap();
        if matches!(layer.w_v.dtype, GgmlType::Q4_K) {
            sgemv_q4_k_f32_triple_quadcoop_into(
                backend,
                &scratch.h_buf,
                &layer.w_q.buffer,
                &layer.w_k.buffer,
                &layer.w_v.buffer,
                &scratch.q_buf,
                &scratch.k_buf,
                &scratch.v_buf,
                layer.w_q.k,
                layer.w_q.n,
                layer.w_k.n,
                layer.w_v.n,
            )
            .unwrap();
        } else {
            sgemv_q4_k_f32_pair_quadcoop_into(
                backend,
                &scratch.h_buf,
                &layer.w_q.buffer,
                &layer.w_k.buffer,
                &scratch.q_buf,
                &scratch.k_buf,
                layer.w_q.k,
                layer.w_q.n,
                layer.w_k.n,
            )
            .unwrap();
            layer
                .w_v
                .matmul_into(backend, &scratch.h_buf, &scratch.v_buf);
        }

        if let Some(qn_buf) = layer.attn_q_norm_buf.as_ref() {
            rms_norm_per_head_f32(
                backend,
                &scratch.q_buf,
                qn_buf,
                n_heads,
                head_dim,
                cfg.rms_eps,
            )
            .unwrap();
        }
        if let Some(kn_buf) = layer.attn_k_norm_buf.as_ref() {
            rms_norm_per_head_f32(backend, &scratch.k_buf, kn_buf, n_kv, head_dim, cfg.rms_eps)
                .unwrap();
        }
        rope_half_split_f32(
            backend,
            &scratch.q_buf,
            &model.rope_cos_buf,
            &model.rope_sin_buf,
            n_heads,
            head_dim,
            position,
        )
        .unwrap();
        rope_half_split_f32(
            backend,
            &scratch.k_buf,
            &model.rope_cos_buf,
            &model.rope_sin_buf,
            n_kv,
            head_dim,
            position,
        )
        .unwrap();
        kv_append_f32(
            backend,
            &scratch.k_buf,
            &scratch.k_caches[li],
            n_kv,
            head_dim,
            position,
            max_seq,
        )
        .unwrap();
        kv_append_f32(
            backend,
            &scratch.v_buf,
            &scratch.v_caches[li],
            n_kv,
            head_dim,
            position,
            max_seq,
        )
        .unwrap();
        let kv_len = position + 1;
        gqa_decode_f32(
            backend,
            &scratch.q_buf,
            &scratch.k_caches[li],
            &scratch.v_caches[li],
            &scratch.attn_buf,
            n_heads,
            n_kv,
            head_dim,
            kv_len,
            max_seq,
        )
        .unwrap();
        layer
            .w_o
            .matmul_into(backend, &scratch.attn_buf, &scratch.o_buf);
        add_inplace_f32(backend, &scratch.xd_buf, &scratch.o_buf, d).unwrap();
        rms_norm_f32(
            backend,
            &scratch.xd_buf,
            &layer.ffn_norm_buf,
            &scratch.h_buf,
            d,
            cfg.rms_eps,
        )
        .unwrap();
        sgemv_q4_k_f32_pair_into(
            backend,
            &scratch.h_buf,
            &layer.w_gate.buffer,
            &layer.w_up.buffer,
            &scratch.gate_buf,
            &scratch.up_buf,
            layer.w_gate.k,
            layer.w_gate.n,
            layer.w_up.n,
        )
        .unwrap();
        swiglu_f32(
            backend,
            &scratch.gate_buf,
            &scratch.up_buf,
            &scratch.fd_buf,
            f,
        )
        .unwrap();
        // T87 — drain & sample fd_buf for sparsity stats.
        backend.drain();
        unsafe {
            std::ptr::copy_nonoverlapping(
                scratch.fd_buf.contents() as *const f32,
                fd_scratch.as_mut_ptr(),
                f,
            );
        }
        stats.record(li, &fd_scratch);

        layer
            .w_down
            .matmul_into(backend, &scratch.fd_buf, &scratch.fc2_buf);
        add_inplace_f32(backend, &scratch.xd_buf, &scratch.fc2_buf, d).unwrap();
    }

    rms_norm_f32(
        backend,
        &scratch.xd_buf,
        &model.final_norm_buf,
        &scratch.h_buf,
        cfg.d,
        cfg.rms_eps,
    )
    .unwrap();
    model
        .lm_head
        .matmul_into(backend, &scratch.h_buf, &scratch.logits_buf);
    backend.drain();
    let mut logits = vec![0.0_f32; cfg.vocab];
    unsafe {
        std::ptr::copy_nonoverlapping(
            scratch.logits_buf.contents() as *const f32,
            logits.as_mut_ptr(),
            cfg.vocab,
        );
    }
    argmax(&logits)
}

struct Scratch {
    x: Vec<f32>,
    h: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    attn_out: Vec<f32>,
    o_out: Vec<f32>,
    gate_out: Vec<f32>,
    up_out: Vec<f32>,
    fc2_out: Vec<f32>,
    k_trim: Vec<f32>,
    v_trim: Vec<f32>,
    // Persistent GPU buffers — pre-allocated once, reused every layer
    // every token. Eliminates the per-call MTLBuffer alloc cost
    // (~10-50µs each) which was 6 allocs × 40 layers = 240/token.
    xd_buf: Buffer,     // d-sized residual stream (lives across all layers)
    h_buf: Buffer,      // d-sized norm output / matmul input
    fd_buf: Buffer,     // f-sized input (swiglu output → down)
    q_buf: Buffer,      // d-sized Q output
    k_buf: Buffer,      // kv_dim-sized K output
    v_buf: Buffer,      // kv_dim-sized V output
    o_buf: Buffer,      // d-sized O proj output
    attn_buf: Buffer,   // d-sized GQA output (input to O proj)
    gate_buf: Buffer,   // f-sized gate output
    up_buf: Buffer,     // f-sized up output
    fc2_buf: Buffer,    // d-sized down output
    logits_buf: Buffer, // vocab-sized lm_head output
    /// Per-layer KV cache, GPU-resident. Each one is `[n_kv * max_seq * head_dim]`
    /// f32. We append into specific positions via direct host-mapped writes
    /// (alloc_shared makes them visible from CPU too — no kernel needed).
    k_caches: Vec<Buffer>,
    v_caches: Vec<Buffer>,
}

impl Scratch {
    fn new(backend: &MetalBackend, cfg: &ModelCfg) -> Self {
        let d = cfg.d;
        let kv_dim = cfg.n_kv_heads * cfg.head_dim;
        let f = cfg.f;
        let max_kv = cfg.n_kv_heads * cfg.max_seq * cfg.head_dim;
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
            k_trim: vec![0.0; max_kv],
            v_trim: vec![0.0; max_kv],
            xd_buf: backend.alloc_shared(d * 4).unwrap(),
            h_buf: backend.alloc_shared(d * 4).unwrap(),
            fd_buf: backend.alloc_shared(f * 4).unwrap(),
            q_buf: backend.alloc_shared(d * 4).unwrap(),
            k_buf: backend.alloc_shared(kv_dim * 4).unwrap(),
            v_buf: backend.alloc_shared(kv_dim * 4).unwrap(),
            o_buf: backend.alloc_shared(d * 4).unwrap(),
            attn_buf: backend.alloc_shared(d * 4).unwrap(),
            gate_buf: backend.alloc_shared(f * 4).unwrap(),
            up_buf: backend.alloc_shared(f * 4).unwrap(),
            fc2_buf: backend.alloc_shared(d * 4).unwrap(),
            logits_buf: backend.alloc_shared(cfg.vocab * 4).unwrap(),
            k_caches: (0..cfg.n_layers)
                .map(|_| backend.alloc_shared(max_kv * 4).unwrap())
                .collect(),
            v_caches: (0..cfg.n_layers)
                .map(|_| backend.alloc_shared(max_kv * 4).unwrap())
                .collect(),
        }
    }
}

fn load_model(backend: &MetalBackend, path: &str, max_seq: usize) -> ModelMetal {
    println!("→ opening {path}");
    let file = GgufFile::open(path).expect("open gguf");
    let arch = file
        .metadata()
        .get("general.architecture")
        .and_then(|v| v.as_str())
        .unwrap_or("?");
    let p = |k: &str| {
        file.metadata()
            .get(&format!("{arch}.{k}"))
            .and_then(|v| v.as_u32())
            .unwrap() as usize
    };
    let pf = |k: &str| {
        file.metadata()
            .get(&format!("{arch}.{k}"))
            .and_then(|v| v.as_f32())
            .unwrap()
    };
    let cfg = ModelCfg {
        d: p("embedding_length"),
        f: p("feed_forward_length"),
        n_layers: p("block_count"),
        n_heads: p("attention.head_count"),
        n_kv_heads: p("attention.head_count_kv"),
        head_dim: p("attention.key_length"),
        vocab: file.tensor("token_embd.weight").unwrap().shape[1] as usize,
        rms_eps: pf("attention.layer_norm_rms_epsilon"),
        rope_theta: pf("rope.freq_base"),
        max_seq,
    };
    println!(
        "  arch={arch} d={} f={} layers={} heads={}/{} head_dim={} vocab={}",
        cfg.d, cfg.f, cfg.n_layers, cfg.n_heads, cfg.n_kv_heads, cfg.head_dim, cfg.vocab
    );

    let token_emb = load_f32(&file, "token_embd.weight");
    let final_norm = load_f32(&file, "output_norm.weight");
    let final_norm_buf = alloc_metal_from_bytes(backend, bytemuck_cast(&final_norm));
    let lm_head_name = if file.tensor("output.weight").is_some() {
        "output.weight"
    } else {
        "token_embd.weight" // tied
    };
    let lm_head = load_metal_weight(backend, &file, lm_head_name);

    let t_load = Instant::now();
    let mut layers = Vec::with_capacity(cfg.n_layers);
    for i in 0..cfg.n_layers {
        let attn_norm = load_f32(&file, &format!("blk.{i}.attn_norm.weight"));
        let ffn_norm = load_f32(&file, &format!("blk.{i}.ffn_norm.weight"));
        let attn_norm_buf = alloc_metal_from_bytes(backend, bytemuck_cast(&attn_norm));
        let ffn_norm_buf = alloc_metal_from_bytes(backend, bytemuck_cast(&ffn_norm));
        let layer = LayerWeights {
            attn_norm,
            attn_norm_buf,
            w_q: load_metal_weight(backend, &file, &format!("blk.{i}.attn_q.weight")),
            w_k: load_metal_weight(backend, &file, &format!("blk.{i}.attn_k.weight")),
            w_v: load_metal_weight(backend, &file, &format!("blk.{i}.attn_v.weight")),
            w_o: load_metal_weight(backend, &file, &format!("blk.{i}.attn_output.weight")),
            ffn_norm,
            ffn_norm_buf,
            w_gate: load_metal_weight(backend, &file, &format!("blk.{i}.ffn_gate.weight")),
            w_up: load_metal_weight(backend, &file, &format!("blk.{i}.ffn_up.weight")),
            w_down: load_metal_weight(backend, &file, &format!("blk.{i}.ffn_down.weight")),
            attn_q_norm: file
                .tensor(&format!("blk.{i}.attn_q_norm.weight"))
                .map(|t| rustorch_gguf::dequant_to_f32(t, file.tensor_bytes(t)).unwrap()),
            attn_k_norm: file
                .tensor(&format!("blk.{i}.attn_k_norm.weight"))
                .map(|t| rustorch_gguf::dequant_to_f32(t, file.tensor_bytes(t)).unwrap()),
            attn_q_norm_buf: file
                .tensor(&format!("blk.{i}.attn_q_norm.weight"))
                .map(|t| {
                    let v = rustorch_gguf::dequant_to_f32(t, file.tensor_bytes(t)).unwrap();
                    alloc_metal_from_bytes(backend, bytemuck_cast(&v))
                }),
            attn_k_norm_buf: file
                .tensor(&format!("blk.{i}.attn_k_norm.weight"))
                .map(|t| {
                    let v = rustorch_gguf::dequant_to_f32(t, file.tensor_bytes(t)).unwrap();
                    alloc_metal_from_bytes(backend, bytemuck_cast(&v))
                }),
        };
        layers.push(layer);
    }
    println!(
        "  loaded {} layers into Metal buffers in {:.2}s",
        cfg.n_layers,
        t_load.elapsed().as_secs_f32()
    );

    let rope = RoPE::new(cfg.head_dim, cfg.max_seq, cfg.rope_theta);
    let rope_cos_buf = alloc_metal_from_bytes(backend, bytemuck_cast(rope.cos_table()));
    let rope_sin_buf = alloc_metal_from_bytes(backend, bytemuck_cast(rope.sin_table()));
    ModelMetal {
        cfg,
        layers,
        token_emb,
        final_norm,
        final_norm_buf,
        lm_head,
        rope,
        rope_cos_buf,
        rope_sin_buf,
    }
}

/// Reinterpret a `&[f32]` as a `&[u8]` for upload into a shared MTLBuffer.
fn bytemuck_cast(v: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

fn main() -> ExitCode {
    let mut args: Vec<String> = env::args().skip(1).collect();
    let mut model_path: Option<String> = None;
    let mut prompt_ids: Vec<u32> = vec![1];
    let mut n: usize = 50;
    let mut max_seq: usize = 256;
    let mut profile = false;
    let mut sparsity_profile = false;
    let mut ngram_profile = false;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--model" => {
                model_path = Some(args[i + 1].clone());
                args.remove(i + 1);
                args.remove(i);
                continue;
            },
            "--prompt-ids" => {
                prompt_ids = args[i + 1]
                    .split(',')
                    .filter_map(|s| s.trim().parse::<u32>().ok())
                    .collect();
                args.remove(i + 1);
                args.remove(i);
                continue;
            },
            "--n" => {
                n = args[i + 1].parse().unwrap_or(n);
                args.remove(i + 1);
                args.remove(i);
                continue;
            },
            "--max-seq" => {
                max_seq = args[i + 1].parse().unwrap_or(max_seq);
                args.remove(i + 1);
                args.remove(i);
                continue;
            },
            "--profile" => {
                profile = true;
                args.remove(i);
                continue;
            },
            "--sparsity-profile" => {
                sparsity_profile = true;
                args.remove(i);
                continue;
            },
            "--ngram-profile" => {
                ngram_profile = true;
                args.remove(i);
                continue;
            },
            _ => {},
        }
        i += 1;
    }
    let path = model_path.expect("--model required");
    let _ = SamplingPlaceholder; // keep workspace happy
    let _ = LlamaConfigUnused;

    let backend = metal_backend();
    println!(
        "device: {} (Metal3: {})",
        backend.adapter_name(),
        backend.supports_metal3()
    );

    let model = load_model(backend, &path, max_seq);
    let mut scratch = Scratch::new(backend, &model.cfg);

    let mut stages = Stages::default();
    let mut sparsity = SparsityStats::new(model.cfg.n_layers, model.cfg.f);
    println!("\n→ prefill {} tokens", prompt_ids.len());
    let t_pre = Instant::now();
    let mut last = 0u32;
    let mut cur_pos = 0usize;
    for &tok in prompt_ids.iter() {
        last = if profile {
            forward_token_profiled(backend, &model, tok, cur_pos, &mut scratch, &mut stages)
        } else if sparsity_profile {
            forward_token_sparsity(backend, &model, tok, cur_pos, &mut scratch, &mut sparsity)
        } else {
            forward_token(backend, &model, tok, cur_pos, &mut scratch)
        };
        cur_pos += 1;
    }
    let prefill_d = t_pre.elapsed();
    println!(
        "  prefill: {} tokens in {:.3}s = {:.2} tok/s",
        prompt_ids.len(),
        prefill_d.as_secs_f64(),
        prompt_ids.len() as f64 / prefill_d.as_secs_f64()
    );

    let mut generated = vec![last];
    // T85 — when profiling, reset accumulator after warmup so prefill
    // costs (different shapes) don't pollute the decode breakdown.
    if profile {
        stages = Stages::default();
    }
    if sparsity_profile {
        sparsity = SparsityStats::new(model.cfg.n_layers, model.cfg.f);
    }
    let t_dec = Instant::now();
    for _ in 1..n {
        last = if profile {
            forward_token_profiled(backend, &model, last, cur_pos, &mut scratch, &mut stages)
        } else if sparsity_profile {
            forward_token_sparsity(backend, &model, last, cur_pos, &mut scratch, &mut sparsity)
        } else {
            forward_token(backend, &model, last, cur_pos, &mut scratch)
        };
        cur_pos += 1;
        generated.push(last);
    }
    let decode_d = t_dec.elapsed();
    println!(
        "  decode : {} tokens in {:.3}s = {:.2} tok/s",
        n - 1,
        decode_d.as_secs_f64(),
        (n - 1) as f64 / decode_d.as_secs_f64()
    );
    if profile {
        stages.print_breakdown(n - 1);
    }
    if sparsity_profile {
        sparsity.print_breakdown();
    }
    if ngram_profile {
        analyze_ngram_hit_rate(&generated, 16);
    }
    println!("\ngenerated: {:?}", generated);
    ExitCode::SUCCESS
}

// Keep workspace deps happy.
#[allow(dead_code)]
struct SamplingPlaceholder;
#[allow(dead_code)]
struct LlamaConfigUnused;
