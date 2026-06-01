//! T246.11 TRAINING-BENCH.1 — rustorch CUDA training bench.
//!
//! Goal : measure rustorch CUDA training throughput on a synthetic
//! transformer-style block (RMSNorm + Linear + SwiGLU + Linear + residual)
//! so we can compare it against PyTorch CUDA on the same hardware.
//!
//! Scope notes (HONEST) :
//!   - Only the autograd-wired CUDA ops are used here :
//!       * `rms_norm_cuda`     (T247.1)
//!       * `matmul_bf16_cuda`  (T246.11 new)
//!       * `swiglu_cuda`       (T246.11 new)
//!     Each op goes through `apply_custom`, which currently stages tensors
//!     CPU↔CUDA per call (no device-resident `Storage::Cuda` yet — P3.Z
//!     Task M is not done). Expect the round-trip to dominate wall-clock.
//!   - Residual add + parameter update happen on CPU via standard autograd
//!     ops (gradient unbroadcast / AdamW step) — both are tiny compared
//!     to the CUDA matmul/SwiGLU work, so this represents the rustorch
//!     "best path available today" for end-to-end training.
//!   - This is intentionally a **floor measurement** for the obrain
//!     integration go/no-go : if rustorch is faster than PyTorch *here*,
//!     we're done. If it loses, the per-op CPU staging is the prime
//!     suspect and the report identifies it explicitly.
//!
//! Workload :
//!   batch=8, seq_len=512 (flattened to 4096 tokens)
//!   hidden=2048, ffn=4096
//!   100 steps of (fwd + MSE-style loss + bwd + opt step)
//!   BF16 activations on-device, F32 master weights / grads on-host
//!
//! Run with :
//! ```sh
//! cargo run -p rustorch-llm --release --features cuda --example training_bench_cuda
//! ```

#![cfg(feature = "cuda")]
#![allow(missing_docs)]

use rustorch_autograd::{backward, ops, Variable};
use rustorch_core::tensor::tensor_impl::Tensor;
use rustorch_llm::cuda_train::{matmul_bf16_cuda, rms_norm_cuda, swiglu_cuda};
use rustorch_optim::{AdamW, Optimizer};
use std::time::Instant;

/// Deterministic small-amplitude tensor (avoid NaNs on BF16 round-trip).
fn det_tensor(shape: &[usize], seed: f32) -> Tensor {
    let n: usize = shape.iter().product();
    let data: Vec<f32> = (0..n)
        .map(|i| ((i as f32 + 1.0) * seed * 0.0001).sin() * 0.1)
        .collect();
    Tensor::from_vec(shape.to_vec(), data).expect("det_tensor")
}

fn ones(shape: &[usize]) -> Tensor {
    let n: usize = shape.iter().product();
    Tensor::from_vec(shape.to_vec(), vec![0.5_f32; n]).expect("ones")
}

fn main() {
    // ---- Workload shape — fits comfortably on GB10 ----
    let batch = 8usize;
    let seq_len = 512usize;
    let m = batch * seq_len; // 4096 tokens
    let hidden = 2048usize;
    let ffn = 4096usize;
    let n_steps = 100usize;
    let eps = 1e-6_f32;

    println!("=== rustorch CUDA training bench (T246.11 TRAINING-BENCH.1) ===");
    println!(
        "shape : batch={batch} seq_len={seq_len} tokens/step={m} hidden={hidden} ffn={ffn} steps={n_steps}"
    );
    println!("dtype : BF16 activations / matmul, F32 master weights");
    println!();

    // ---- Synthetic inputs and weight params (CPU, will be staged per op) ----
    let x_data = det_tensor(&[m, hidden], 1.0);
    let target = ones(&[m, hidden]);

    // Pre-norm gamma : [hidden]
    let gamma = Variable::leaf(det_tensor(&[hidden], 0.7)).requires_grad(true);
    // Gate proj : [hidden, ffn]
    let w_gate = Variable::leaf(det_tensor(&[hidden, ffn], 0.5)).requires_grad(true);
    // Up proj   : [hidden, ffn]
    let w_up = Variable::leaf(det_tensor(&[hidden, ffn], 0.3)).requires_grad(true);
    // Down proj : [ffn, hidden]
    let w_down = Variable::leaf(det_tensor(&[ffn, hidden], 0.2)).requires_grad(true);

    let x = Variable::new(x_data);
    let t = Variable::new(target);

    let mut opt = AdamW::new(
        vec![gamma.clone(), w_gate.clone(), w_up.clone(), w_down.clone()],
        1e-4,
    );

    // ---- Warm-up (NVRTC kernel compile + LtSession plan cache prime) ----
    println!("warm-up : 1 step...");
    let warm = Instant::now();
    {
        opt.zero_grad();
        let n = rms_norm_cuda(&x, &gamma, eps).expect("rms_norm_cuda warm");
        let gate = matmul_bf16_cuda(&n, &w_gate).expect("matmul gate warm");
        let upv = matmul_bf16_cuda(&n, &w_up).expect("matmul up warm");
        let act = swiglu_cuda(&gate, &upv).expect("swiglu warm");
        let h = matmul_bf16_cuda(&act, &w_down).expect("matmul down warm");
        let y = ops::add(&h, &x).expect("residual warm");
        let diff = ops::sub(&y, &t).expect("diff warm");
        let sq = ops::mul(&diff, &diff).expect("sq warm");
        let loss = ops::sum(&sq).expect("loss warm");
        backward(&loss, None).expect("bwd warm");
        opt.step();
    }
    let warm_t = warm.elapsed();
    println!("warm-up done : {warm_t:?}");
    println!();

    // ---- Measured loop ----
    println!("measuring {n_steps} steps...");
    let t0 = Instant::now();
    let mut last_loss = 0.0_f32;
    for step in 0..n_steps {
        opt.zero_grad();

        // Pre-norm.
        let n = rms_norm_cuda(&x, &gamma, eps).expect("rms_norm_cuda");
        // FFN : (silu(N @ W_gate) * (N @ W_up)) @ W_down + residual.
        let gate = matmul_bf16_cuda(&n, &w_gate).expect("matmul gate");
        let upv = matmul_bf16_cuda(&n, &w_up).expect("matmul up");
        let act = swiglu_cuda(&gate, &upv).expect("swiglu");
        let h = matmul_bf16_cuda(&act, &w_down).expect("matmul down");
        let y = ops::add(&h, &x).expect("residual");

        // Simple MSE-style loss : sum((y - t)^2).
        let diff = ops::sub(&y, &t).expect("diff");
        let sq = ops::mul(&diff, &diff).expect("sq");
        let loss = ops::sum(&sq).expect("loss");
        last_loss = loss.tensor().as_slice::<f32>().unwrap()[0];

        backward(&loss, None).expect("backward");
        opt.step();

        if step == 0 || step == n_steps - 1 {
            println!("  step {step} loss={last_loss:.5e}");
        }
    }
    let total = t0.elapsed();

    // ---- Report ----
    let total_s = total.as_secs_f64();
    let tokens = (m * n_steps) as f64;
    let tok_per_s = tokens / total_s;
    let ms_per_step = total_s * 1000.0 / (n_steps as f64);

    println!();
    println!("---- results ----");
    println!("  total wall-clock : {total:?}");
    println!("  per-step         : {ms_per_step:.2} ms");
    println!("  tokens/sec       : {tok_per_s:.1}");
    println!("  final loss       : {last_loss:.5e}");
    println!();
    println!("=== done ===");

    // Emit a one-line summary the parser can grep.
    println!(
        "RESULT_JSON {{\"path\":\"rustorch_cuda\",\"steps\":{n_steps},\"tokens_per_step\":{m},\"wall_clock_s\":{total_s:.4},\"ms_per_step\":{ms_per_step:.4},\"tokens_per_sec\":{tok_per_s:.2}}}"
    );
}
