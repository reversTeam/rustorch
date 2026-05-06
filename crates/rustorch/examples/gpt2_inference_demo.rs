//! GPT-2-small inference demo — concrete RusTorch perf showcase.
//!
//! Run with:
//!     cargo run --release --example gpt2_inference_demo
//!
//! Builds a 12-layer transformer with GPT-2-small dimensions
//! (D=768, H=12, F=3072, vocab=50257) and runs a forward pass on
//! 128-token input sequences. Reports per-iteration timings,
//! median, p50/p99, and the throughput in tokens/sec.
//!
//! Compare against the equivalent Python script (using
//! `bench-vs-pytorch/pytorch_bench.py::bench_gpt2_full_stack`) to
//! see RusTorch's CPU performance head-to-head with PyTorch.

use rustorch_autograd::{no_grad, ops};
use rustorch_core::tensor::tensor_impl::Tensor;
use rustorch_fusion::patterns::matmul_bias_act::Activation;
use rustorch_nn::{Embedding, LayerNorm, Linear, Module, MultiHeadAttention};
use std::time::Instant;

const NUM_LAYERS: usize = 12;
const BATCH: usize = 1;
const SEQ: usize = 128;
const D_MODEL: usize = 768;
const N_HEADS: usize = 12;
const D_FF: usize = 3072;
const VOCAB: usize = 50257;
const WARMUP: usize = 3;
const TIMED_ITERS: usize = 10;

fn main() {
    println!();
    println!("=========================================================");
    println!(" RusTorch — GPT-2 small forward inference demo");
    println!("=========================================================");
    println!();
    println!(
        "  Architecture : {} layers × {{LayerNorm, MHA(H={}, D={}), LayerNorm, FFN(F={})}}",
        NUM_LAYERS, N_HEADS, D_MODEL, D_FF
    );
    println!(
        "  Embedding    : token({} × {})  +  position({} × {})",
        VOCAB, D_MODEL, SEQ, D_MODEL
    );
    println!("  LM head      : Linear({} → {})", D_MODEL, VOCAB);
    println!("  Input shape  : [B={}, S={}]", BATCH, SEQ);

    // Estimate parameter count (rough: emb + 12*(LN + 4*Linear D² + LN +
    // Linear D*F + Linear F*D) + final LN + LM head D*V).
    let param_count = VOCAB * D_MODEL                           // tok_emb
        + SEQ * D_MODEL                                          // pos_emb
        + NUM_LAYERS * (2 * D_MODEL                              // 2 LayerNorms gamma+beta combined  (approx 2*D each)
            + 4 * D_MODEL * D_MODEL                              // Q/K/V/O projections
            + D_MODEL * D_FF                                     // FFN.fc1
            + D_FF * D_MODEL)                                    // FFN.fc2
        + 2 * D_MODEL                                            // final LN
        + D_MODEL * VOCAB; // LM head
    println!("  Parameters   : ~{:.1} M", param_count as f64 / 1e6);
    println!();
    println!("  Building model...");
    let t_build_start = Instant::now();

    // ---------- model construction --------------------------------------
    let token_emb = Embedding::with_seed(VOCAB, D_MODEL, 0xC1A4);
    let pos_emb = Embedding::with_seed(SEQ, D_MODEL, 0xC1A5);

    let mut layers: Vec<(LayerNorm, MultiHeadAttention, LayerNorm, Linear, Linear)> =
        Vec::with_capacity(NUM_LAYERS);
    for _ in 0..NUM_LAYERS {
        layers.push((
            LayerNorm::new(D_MODEL),
            MultiHeadAttention::new(D_MODEL, N_HEADS),
            LayerNorm::new(D_MODEL),
            Linear::new(D_MODEL, D_FF),
            Linear::new(D_FF, D_MODEL),
        ));
    }
    let final_ln = LayerNorm::new(D_MODEL);
    let lm_head = Linear::new(D_MODEL, VOCAB);
    let t_build = t_build_start.elapsed();
    println!("  Build time   : {:.2} s", t_build.as_secs_f64());
    println!();

    // ---------- inputs --------------------------------------------------
    let mut s: u64 = 1;
    let mut ids: Vec<i64> = Vec::with_capacity(BATCH * SEQ);
    for _ in 0..BATCH * SEQ {
        s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        ids.push(((s >> 32) as i64).rem_euclid(VOCAB as i64));
    }
    let ids_t = Tensor::from_vec_typed::<i64, _>([BATCH * SEQ], ids).unwrap();
    let pos_ids: Vec<i64> = (0..SEQ as i64).collect();
    let pos_t = Tensor::from_vec_typed::<i64, _>([SEQ], pos_ids).unwrap();

    // ---------- one forward pass closure -------------------------------
    let forward = || {
        no_grad(|| {
            let tok = token_emb.forward_indices(&ids_t).unwrap();
            let pos = pos_emb.forward_indices(&pos_t).unwrap();
            let mut x = ops::add(&tok, &pos).unwrap();
            x = ops::reshape(&x, vec![BATCH, SEQ, D_MODEL]).unwrap();

            for (ln1, mha, ln2, fc1, fc2) in layers.iter() {
                let h = ln1.forward(&x).unwrap();
                let h = mha.forward(&h, &h, &h, None).unwrap();
                x = ops::add(&x, &h).unwrap();
                let h = ln2.forward(&x).unwrap();
                let h = fc1.forward_with_activation(&h, Activation::Relu).unwrap();
                let h = fc2.forward(&h).unwrap();
                x = ops::add(&x, &h).unwrap();
            }
            let h = final_ln.forward(&x).unwrap();
            lm_head.forward(&h).unwrap()
        })
    };

    // ---------- warmup --------------------------------------------------
    println!("  Warming up ({} iters)...", WARMUP);
    for _ in 0..WARMUP {
        let _ = forward();
    }

    // ---------- timed run -----------------------------------------------
    println!("  Timed run    ({} iters):", TIMED_ITERS);
    let mut samples: Vec<u128> = Vec::with_capacity(TIMED_ITERS);
    let mut last_logits = None;
    for i in 0..TIMED_ITERS {
        let t0 = Instant::now();
        let logits = forward();
        let elapsed = t0.elapsed();
        samples.push(elapsed.as_micros());
        println!(
            "    iter {:>2}: {:>7.2} ms",
            i + 1,
            elapsed.as_secs_f64() * 1000.0
        );
        last_logits = Some(logits);
    }
    println!();

    // ---------- summary -------------------------------------------------
    samples.sort();
    let median = samples[samples.len() / 2];
    let p99 = samples[((samples.len() as f64) * 0.99) as usize];
    let mean = samples.iter().sum::<u128>() / samples.len() as u128;
    println!("=========================================================");
    println!(" Forward pass summary (B={}, S={})", BATCH, SEQ);
    println!("---------------------------------------------------------");
    println!("  Median       : {:>7.2} ms", median as f64 / 1000.0);
    println!("  Mean         : {:>7.2} ms", mean as f64 / 1000.0);
    println!("  p99          : {:>7.2} ms", p99 as f64 / 1000.0);
    println!(
        "  Tokens/sec   : {:>7} ({}-token sequences)",
        ((BATCH * SEQ) as u128 * 1_000_000 / median),
        BATCH * SEQ
    );
    println!("=========================================================");
    println!();

    // ---------- output sanity -------------------------------------------
    let logits = last_logits.unwrap();
    let logits_t = logits.tensor();
    println!("  Output logits shape : {:?}", logits_t.shape());
    let logits_v = logits_t.as_slice::<f32>().unwrap();
    let pos = 0_usize;
    let row_off = pos * VOCAB;
    let mut top: Vec<(usize, f32)> = (0..VOCAB).map(|i| (i, logits_v[row_off + i])).collect();
    top.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    println!("  Top-5 logits at position 0:");
    for (rank, (idx, score)) in top.iter().take(5).enumerate() {
        println!(
            "    {}. token_id = {:>5}   logit = {:>+8.4}",
            rank + 1,
            idx,
            score
        );
    }
    println!();
    println!("  Compare with PyTorch via:  python3 bench-vs-pytorch/pytorch_bench.py 1 \\");
    println!("                             | jq -r '.results[] | select(.op == \"gpt2_full_stack_forward\")'");
    println!();
}
