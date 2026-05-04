//! Llama-style autoregressive decode demo (T47).
//!
//! Builds a mini Llama (4 layers, RMSNorm + GQA + RoPE + FFN) with
//! random weights, then runs a 64-token generation loop on a 16-token
//! prompt. Demonstrates the integration of the LLM-runner bricks
//! shipped in T41 (KV-cache), T42 (RoPE), T43 (sampling), T45 (GQA).
//!
//! Run with:
//!     cargo run --release --example llama_decode_demo
//!
//! Output: per-step decode latency + tokens/sec, plus the (random)
//! generated token sequence. With random weights the actual tokens
//! are meaningless; the metric that matters is per-token latency.

use rustorch_autograd::{no_grad, Variable};
use rustorch_core::tensor::tensor_impl::Tensor;
use rustorch_fusion::patterns::matmul_bias_act::Activation;
use rustorch_nn::gqa::gqa_forward_f32;
use rustorch_nn::kv_cache::KVCache;
use rustorch_nn::rope::RoPE;
use rustorch_nn::sampling::{sample_next, SamplingConfig};
use rustorch_nn::{Embedding, Linear, Module, RMSNorm};
use std::time::Instant;

// Mini-Llama config.
const NUM_LAYERS: usize = 4;
const D_MODEL: usize = 256;
const N_HEADS: usize = 8;
const N_KV_HEADS: usize = 2; // 4-to-1 GQA ratio (Qwen-style)
const HEAD_DIM: usize = D_MODEL / N_HEADS; // 32
const D_FF: usize = 1024;
const VOCAB: usize = 4096;
const MAX_SEQ: usize = 128;

const PROMPT_LEN: usize = 16;
const MAX_NEW_TOKENS: usize = 64;

/// One transformer block weights (RMSNorm + Q/K/V/O linears + RMSNorm + FFN).
struct LlamaBlock {
    ln_attn: RMSNorm,
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    ln_ffn: RMSNorm,
    fc1: Linear,
    fc2: Linear,
}

impl LlamaBlock {
    fn new() -> Self {
        let kv_dim = N_KV_HEADS * HEAD_DIM;
        LlamaBlock {
            ln_attn: RMSNorm::new(D_MODEL),
            q_proj: Linear::new(D_MODEL, D_MODEL).no_bias_self(),
            k_proj: Linear::new(D_MODEL, kv_dim).no_bias_self(),
            v_proj: Linear::new(D_MODEL, kv_dim).no_bias_self(),
            o_proj: Linear::new(D_MODEL, D_MODEL).no_bias_self(),
            ln_ffn: RMSNorm::new(D_MODEL),
            fc1: Linear::new(D_MODEL, D_FF).no_bias_self(),
            fc2: Linear::new(D_FF, D_MODEL).no_bias_self(),
        }
    }
}

trait LinearExt {
    fn no_bias_self(self) -> Self;
}
impl LinearExt for Linear {
    fn no_bias_self(mut self) -> Self {
        self.bias = None;
        self
    }
}

fn linear_forward(layer: &Linear, x: &Variable) -> Variable {
    layer.forward(x).unwrap()
}

/// One decode step: forward the model for a single new token at
/// `position`, append K/V to the cache, return the logits over vocab.
#[allow(clippy::too_many_arguments)]
fn decode_step(
    token_id: i64,
    position: usize,
    blocks: &[LlamaBlock],
    final_ln: &RMSNorm,
    lm_head: &Linear,
    token_emb: &Embedding,
    rope: &RoPE,
    cache: &mut KVCache,
) -> Vec<f32> {
    no_grad(|| {
        // 1. Embed the single new token. Output [1, D_MODEL].
        let id_t = Tensor::from_vec_typed::<i64, _>([1usize], vec![token_id]).unwrap();
        let mut x = token_emb.forward_indices(&id_t).unwrap();
        // Reshape to [B=1, S=1, D].
        x = rustorch_autograd::ops::reshape(&x, vec![1usize, 1, D_MODEL]).unwrap();

        for (layer_idx, block) in blocks.iter().enumerate() {
            // 2. RMSNorm + project Q/K/V (each Linear is a [B, S, D]
            //    matmul; under no_grad it dispatches to the fused
            //    sgemv path for S=1).
            let h = block.ln_attn.forward(&x).unwrap();
            let q = linear_forward(&block.q_proj, &h);
            let k = linear_forward(&block.k_proj, &h);
            let v = linear_forward(&block.v_proj, &h);

            // 3. Apply RoPE to Q and K (V is not rotated).
            let q_t = q.tensor();
            let k_t = k.tensor();
            let q_slice = q_t.as_slice::<f32>().unwrap();
            let k_slice = k_t.as_slice::<f32>().unwrap();
            let mut q_buf = q_slice.to_vec(); // [1, 1, D] = D floats
            let mut k_buf = k_slice.to_vec(); // [1, 1, KV_DIM]
                                              // Reinterpret as [B=1, n_heads, S=1, head_dim] for RoPE.
            rope.apply_inplace(&mut q_buf, 1, N_HEADS, 1, position)
                .unwrap();
            rope.apply_inplace(&mut k_buf, 1, N_KV_HEADS, 1, position)
                .unwrap();

            // 4. V projection slice as raw f32.
            let v_t = v.tensor();
            let v_slice = v_t.as_slice::<f32>().unwrap();

            // 5. Append the new K, V to the cache.
            cache.append(layer_idx, 1, &k_buf, v_slice).unwrap();

            // 6. GQA over the cached prefix: q [1, H, 1, hd] attends
            //    to k/v [1, KV, current_len+1, hd].
            let kv_len = position + 1;
            // Build trimmed k/v slices viewing only the first kv_len
            // positions of the cache buffer (which is [1, KV, max_seq, hd]).
            let k_buf_full = cache.k_buffer(layer_idx).unwrap();
            let v_buf_full = cache.v_buffer(layer_idx).unwrap();
            let mut k_trim = vec![0.0_f32; N_KV_HEADS * kv_len * HEAD_DIM];
            let mut v_trim = vec![0.0_f32; N_KV_HEADS * kv_len * HEAD_DIM];
            for kv_h in 0..N_KV_HEADS {
                let src_off = kv_h * MAX_SEQ * HEAD_DIM;
                let dst_off = kv_h * kv_len * HEAD_DIM;
                k_trim[dst_off..dst_off + kv_len * HEAD_DIM]
                    .copy_from_slice(&k_buf_full[src_off..src_off + kv_len * HEAD_DIM]);
                v_trim[dst_off..dst_off + kv_len * HEAD_DIM]
                    .copy_from_slice(&v_buf_full[src_off..src_off + kv_len * HEAD_DIM]);
            }
            let mut attn_out = vec![0.0_f32; N_HEADS * HEAD_DIM];
            gqa_forward_f32(
                &q_buf,
                &k_trim,
                &v_trim,
                &mut attn_out,
                1,
                N_HEADS,
                N_KV_HEADS,
                1,
                kv_len,
                HEAD_DIM,
            )
            .unwrap();

            // 7. O proj + residual.
            let attn_t = Tensor::from_vec(vec![1usize, 1, D_MODEL], attn_out).unwrap();
            let attn_var = Variable::new(attn_t);
            let attn_proj = linear_forward(&block.o_proj, &attn_var);
            x = rustorch_autograd::ops::add(&x, &attn_proj).unwrap();

            // 8. RMSNorm + FFN (linear + relu + linear) + residual.
            let h = block.ln_ffn.forward(&x).unwrap();
            let h = block
                .fc1
                .forward_with_activation(&h, Activation::Relu)
                .unwrap();
            let h = linear_forward(&block.fc2, &h);
            x = rustorch_autograd::ops::add(&x, &h).unwrap();
        }

        // 9. Final RMSNorm + LM head.
        let h = final_ln.forward(&x).unwrap();
        let logits = linear_forward(lm_head, &h);
        // logits shape [1, 1, VOCAB] -> flatten to Vec<f32>.
        logits.tensor().as_slice::<f32>().unwrap().to_vec()
    })
}

#[allow(unused_variables)]
fn main() {
    println!();
    println!("=========================================================");
    println!(" RusTorch — mini-Llama decode loop (T47)");
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
    let blocks: Vec<LlamaBlock> = (0..NUM_LAYERS).map(|_| LlamaBlock::new()).collect();
    let final_ln = RMSNorm::new(D_MODEL);
    let lm_head = Linear::new(D_MODEL, VOCAB).no_bias_self();
    let token_emb = Embedding::with_seed(VOCAB, D_MODEL, 0xCAFE);
    let rope = RoPE::new(HEAD_DIM, MAX_SEQ, 10000.0);
    println!("  Build time   : {:.2} s", t_build.elapsed().as_secs_f64());

    let mut cache = KVCache::new(NUM_LAYERS, 1, N_KV_HEADS, HEAD_DIM, MAX_SEQ);
    let mut output: Vec<u32> = Vec::with_capacity(PROMPT_LEN + MAX_NEW_TOKENS);
    let sampling_cfg = SamplingConfig::greedy();

    // Deterministic LCG for sampling fallback (greedy doesn't use it).
    let mut s: u64 = 1;
    let mut next_u = move || {
        s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        ((s >> 32) as f32) / (u32::MAX as f32)
    };

    // Build a deterministic prompt.
    let prompt: Vec<i64> = (0..PROMPT_LEN as i64)
        .map(|i| (i * 17 + 3) % VOCAB as i64)
        .collect();
    println!("  Prompt token ids: {:?}", &prompt);
    println!();

    // ------------------ Prefill -------------------------------------
    println!("  Prefill ({} tokens)...", PROMPT_LEN);
    let t_prefill = Instant::now();
    for (pos, &tok) in prompt.iter().enumerate() {
        let _ = decode_step(
            tok, pos, &blocks, &final_ln, &lm_head, &token_emb, &rope, &mut cache,
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

    // ------------------ Decode loop ---------------------------------
    println!("  Decode (greedy, {} new tokens):", MAX_NEW_TOKENS);
    let mut step_times_us: Vec<u128> = Vec::with_capacity(MAX_NEW_TOKENS);
    let mut last_token = output[output.len() - 1] as i64;
    for step in 0..MAX_NEW_TOKENS {
        let t0 = Instant::now();
        let pos = cache.current_len();
        let logits = decode_step(
            last_token, pos, &blocks, &final_ln, &lm_head, &token_emb, &rope, &mut cache,
        );
        cache.advance(1).unwrap();
        let next = sample_next(&logits, &sampling_cfg, &output, &mut next_u);
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

    // ------------------ Summary -------------------------------------
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
