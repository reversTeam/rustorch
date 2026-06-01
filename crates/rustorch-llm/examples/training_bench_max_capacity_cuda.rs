//! T246.12 MAX-CAP-TRAIN.1 — rustorch CUDA training bench at MAX-CAPACITY shapes.
//!
//! Companion to `training_bench_cuda.rs` (T246.11) — same op set, much
//! larger shapes designed to **saturate** GB10 Blackwell BF16 tensor
//! cores (250 TFLOPS peak).
//!
//! ## Workload — Option C (synthetic transformer block, max shapes)
//!
//! Default (overridable via env) :
//!   batch        = 32        (env `RUSTORCH_BENCH_BATCH`)
//!   seq_len      = 2048      (env `RUSTORCH_BENCH_SEQ`)
//!   hidden       = 4096      (env `RUSTORCH_BENCH_HIDDEN`)
//!   ffn          = 14336     (env `RUSTORCH_BENCH_FFN`)
//!   layers       = 1         (env `RUSTORCH_BENCH_LAYERS`)
//!   steps        = 20        (env `RUSTORCH_BENCH_STEPS`)
//!
//! Tokens / step = batch * seq_len = 65 536 at default.
//!
//! ## Honest scope caveats
//!
//! - Every CUDA op in `cuda_train` stages tensors CPU↔CUDA per call
//!   (no `Storage::Cuda` yet — P3.Z Task M is not done). At max-capacity
//!   shapes the round-trip dominates wall-clock.
//! - Residual add + loss + AdamW step run on CPU autograd, since
//!   `Device::Cuda` is not yet dispatched. These are small relative to
//!   the matmul work BUT at max shapes the residual/sum on CPU is ~MB-class
//!   and may itself stand out — captured per-step.
//! - This bench reports the **measured rustorch CUDA training ceiling
//!   today** ; it is NOT an extrapolation of what's possible with proper
//!   device-resident tensors. The PyTorch comparison runs at the EXACT
//!   same shape so the comparison is fair.
//!
//! ## Workload note (FFN-only, no attention)
//!
//! Only the autograd-wired CUDA ops are exercised : RMSNorm → gate/up
//! GEMM → SwiGLU → down GEMM → residual. No attention (no GQA backward
//! is wired through autograd yet). This is the rustorch-best-path for
//! today's wiring ; PyTorch runs the identical block for fairness.
//!
//! ## FLOP accounting (per step, one layer)
//!
//! Forward GEMMs : 2 × (M·K·N) for gate + up + 1 × (M·K·N) for down.
//!   gate / up : M=tokens, K=hidden, N=ffn  → 2·M·hidden·ffn   FMA  ⇒ 4·M·hidden·ffn flops
//!   down      : M=tokens, K=ffn, N=hidden  → 2·M·ffn·hidden  FMA   ⇒ 4·M·ffn·hidden flops
//! Backward GEMMs : two more matmuls per forward GEMM ⇒ 3× factor on bwd over fwd.
//! Total ≈ 4 × (fwd GEMMs).
//!
//! For tokens=65 536, hidden=4096, ffn=14336, layers=1, the per-step
//! GEMM flops are :
//!   fwd  = 4 × (2 × tokens × hidden × ffn + tokens × ffn × hidden)
//!        = 4 × 3 × tokens × hidden × ffn
//!        = 4 × 3 × 65 536 × 4096 × 14336
//!        ≈ 46 TFLOP
//!   total step (fwd + bwd ≈ 4× fwd flops actually 2.5×) ≈ 115 TFLOP
//!
//! Run :
//! ```sh
//! cargo run -p rustorch-llm --release --features cuda --example training_bench_max_capacity_cuda
//! ```

#![cfg(feature = "cuda")]
#![allow(missing_docs)]

use rustorch_autograd::{backward, ops, Variable};
use rustorch_core::tensor::tensor_impl::Tensor;
use rustorch_llm::cuda_train::{matmul_bf16_cuda, rms_norm_cuda, swiglu_cuda};
use rustorch_optim::{AdamW, Optimizer};
use std::env;
use std::time::Instant;

fn env_usize(name: &str, default: usize) -> usize {
    env::var(name)
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(default)
}

/// Deterministic small-amplitude tensor (avoid NaNs on BF16 round-trip).
fn det_tensor(shape: &[usize], seed: f32) -> Tensor {
    let n: usize = shape.iter().product();
    let data: Vec<f32> = (0..n)
        .map(|i| ((i as f32 + 1.0) * seed * 0.0001).sin() * 0.1)
        .collect();
    Tensor::from_vec(shape.to_vec(), data).expect("det_tensor")
}

fn target_tensor(shape: &[usize]) -> Tensor {
    let n: usize = shape.iter().product();
    Tensor::from_vec(shape.to_vec(), vec![0.5_f32; n]).expect("target")
}

struct LayerParams {
    gamma: Variable,
    w_gate: Variable,
    w_up: Variable,
    w_down: Variable,
}

fn build_layer(hidden: usize, ffn: usize, layer_idx: usize) -> LayerParams {
    let s = (layer_idx as f32) * 0.13 + 0.7;
    LayerParams {
        gamma: Variable::leaf(det_tensor(&[hidden], s)).requires_grad(true),
        w_gate: Variable::leaf(det_tensor(&[hidden, ffn], s + 0.5)).requires_grad(true),
        w_up: Variable::leaf(det_tensor(&[hidden, ffn], s + 0.3)).requires_grad(true),
        w_down: Variable::leaf(det_tensor(&[ffn, hidden], s + 0.2)).requires_grad(true),
    }
}

fn forward_layer(x: &Variable, p: &LayerParams, eps: f32) -> Variable {
    let n = rms_norm_cuda(x, &p.gamma, eps).expect("rms_norm_cuda");
    let gate = matmul_bf16_cuda(&n, &p.w_gate).expect("matmul gate");
    let upv = matmul_bf16_cuda(&n, &p.w_up).expect("matmul up");
    let act = swiglu_cuda(&gate, &upv).expect("swiglu");
    let h = matmul_bf16_cuda(&act, &p.w_down).expect("matmul down");
    ops::add(&h, x).expect("residual")
}

fn main() {
    // ---- Workload shape — overridable via env ----
    let batch = env_usize("RUSTORCH_BENCH_BATCH", 32);
    let seq_len = env_usize("RUSTORCH_BENCH_SEQ", 2048);
    let hidden = env_usize("RUSTORCH_BENCH_HIDDEN", 4096);
    let ffn = env_usize("RUSTORCH_BENCH_FFN", 14336);
    let layers = env_usize("RUSTORCH_BENCH_LAYERS", 1);
    let n_steps = env_usize("RUSTORCH_BENCH_STEPS", 20);
    let eps = 1e-6_f32;
    let tokens = batch * seq_len;

    println!("=== rustorch CUDA MAX-CAPACITY training bench (T246.12 MAX-CAP-TRAIN.1) ===");
    println!(
        "shape : batch={batch} seq_len={seq_len} tokens/step={tokens} hidden={hidden} ffn={ffn} layers={layers} steps={n_steps}"
    );
    println!("dtype : BF16 activations / matmul, F32 master weights");
    println!(
        "memory hint : input ~{:.2} GB, gate-act ~{:.2} GB, w_gate ~{:.2} GB",
        (tokens * hidden * 4) as f64 / 1e9,
        (tokens * ffn * 4) as f64 / 1e9,
        (hidden * ffn * 4) as f64 / 1e9
    );
    println!();

    // ---- Inputs / targets ----
    let x_data = det_tensor(&[tokens, hidden], 1.0);
    let target = target_tensor(&[tokens, hidden]);
    let x = Variable::new(x_data);
    let t = Variable::new(target);

    // ---- Build layers (params + AdamW) ----
    let mut all_params: Vec<Variable> = Vec::new();
    let mut layer_params: Vec<LayerParams> = Vec::with_capacity(layers);
    for li in 0..layers {
        let lp = build_layer(hidden, ffn, li);
        all_params.push(lp.gamma.clone());
        all_params.push(lp.w_gate.clone());
        all_params.push(lp.w_up.clone());
        all_params.push(lp.w_down.clone());
        layer_params.push(lp);
    }
    let mut opt = AdamW::new(all_params, 1e-4);

    // ---- Warm-up : 1 step ----
    println!("warm-up : 1 step (cuBLASLt plan cache + NVRTC) ...");
    let warm_start = Instant::now();
    {
        opt.zero_grad();
        let mut h = x.clone();
        for lp in &layer_params {
            h = forward_layer(&h, lp, eps);
        }
        let diff = ops::sub(&h, &t).expect("diff warm");
        let sq = ops::mul(&diff, &diff).expect("sq warm");
        let loss = ops::sum(&sq).expect("loss warm");
        backward(&loss, None).expect("bwd warm");
        opt.step();
    }
    let warm = warm_start.elapsed();
    println!("warm-up done : {warm:?}");
    println!();

    // ---- Measured loop ----
    println!("measuring {n_steps} steps ...");
    let t0 = Instant::now();
    let mut last_loss = 0.0_f32;
    let mut step_times: Vec<f64> = Vec::with_capacity(n_steps);
    for step in 0..n_steps {
        let t_step = Instant::now();
        opt.zero_grad();
        let mut h = x.clone();
        for lp in &layer_params {
            h = forward_layer(&h, lp, eps);
        }
        let diff = ops::sub(&h, &t).expect("diff");
        let sq = ops::mul(&diff, &diff).expect("sq");
        let loss = ops::sum(&sq).expect("loss");
        last_loss = loss.tensor().as_slice::<f32>().unwrap()[0];
        backward(&loss, None).expect("backward");
        opt.step();
        let dt = t_step.elapsed().as_secs_f64();
        step_times.push(dt);
        if step == 0 || step == n_steps - 1 || step % 5 == 0 {
            println!("  step {step:>3} loss={last_loss:.5e} step_time={dt:.3}s");
        }
    }
    let total = t0.elapsed();

    // ---- FLOP / TFLOPS estimate ----
    // Forward GEMM flops per layer per step :
    //   gate : 2·tokens·hidden·ffn
    //   up   : 2·tokens·hidden·ffn
    //   down : 2·tokens·ffn·hidden
    // Backward ≈ 2× forward in terms of GEMM flops (one matmul becomes two).
    // Conservative end-to-end multiplier : fwd + bwd ≈ 3× the fwd GEMM flops.
    let fwd_gemm_flops = (tokens * hidden * ffn * 2 * 3) as f64; // gate+up+down, each 2·M·K·N
    let total_flops_per_step = fwd_gemm_flops * 3.0 * (layers as f64); // fwd+bwd ~3× fwd
    let total_flops = total_flops_per_step * (n_steps as f64);
    let tflops = total_flops / total.as_secs_f64() / 1e12;
    let pct_peak = tflops / 250.0 * 100.0;

    // ---- Report ----
    let total_s = total.as_secs_f64();
    let tokens_total = (tokens * n_steps) as f64;
    let tok_per_s = tokens_total / total_s;
    let ms_per_step = total_s * 1000.0 / (n_steps as f64);

    // Step-time distribution.
    let mut sorted = step_times.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p50 = sorted[sorted.len() / 2] * 1000.0;
    let p95 = sorted[(sorted.len() * 95 / 100).min(sorted.len() - 1)] * 1000.0;

    println!();
    println!("---- results ----");
    println!("  total wall-clock : {total:?}");
    println!("  per-step (mean)  : {ms_per_step:.2} ms");
    println!("  per-step (p50)   : {p50:.2} ms");
    println!("  per-step (p95)   : {p95:.2} ms");
    println!("  tokens/sec       : {tok_per_s:.1}");
    println!("  TFLOPS achieved  : {tflops:.2}");
    println!("  % of 250 TF peak : {pct_peak:.2}%");
    println!("  final loss       : {last_loss:.5e}");
    println!();
    println!("=== done ===");

    println!(
        "RESULT_JSON {{\"path\":\"rustorch_cuda_maxcap\",\"steps\":{n_steps},\"tokens_per_step\":{tokens},\"layers\":{layers},\"hidden\":{hidden},\"ffn\":{ffn},\"wall_clock_s\":{total_s:.4},\"ms_per_step\":{ms_per_step:.4},\"tokens_per_sec\":{tok_per_s:.2},\"tflops\":{tflops:.4},\"pct_peak\":{pct_peak:.4},\"p50_ms\":{p50:.4},\"p95_ms\":{p95:.4}}}"
    );
}
