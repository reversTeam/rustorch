//! T246.11 TRAINING-BENCH.3 — per-op breakdown of rustorch CUDA training.
//!
//! Identifies which op is the bottleneck for the slowness measured by
//! `training_bench_cuda` vs PyTorch. Times each forward + backward op
//! individually so we can rank them.
//!
//! Run with :
//! ```sh
//! cargo run -p rustorch-llm --release --features cuda --example training_bench_cuda_breakdown
//! ```

#![cfg(feature = "cuda")]
#![allow(missing_docs)]

use rustorch_autograd::{backward, ops, Variable};
use rustorch_core::tensor::tensor_impl::Tensor;
use rustorch_llm::cuda_train::{matmul_bf16_cuda, rms_norm_cuda, swiglu_cuda};
use std::time::Instant;

fn det_tensor(shape: &[usize], seed: f32) -> Tensor {
    let n: usize = shape.iter().product();
    let data: Vec<f32> = (0..n)
        .map(|i| ((i as f32 + 1.0) * seed * 0.0001).sin() * 0.1)
        .collect();
    Tensor::from_vec(shape.to_vec(), data).expect("det_tensor")
}

fn time_op<F: FnMut() -> ()>(name: &str, iters: u32, mut f: F) {
    // Warm up once.
    f();
    let t0 = Instant::now();
    for _ in 0..iters {
        f();
    }
    let total = t0.elapsed();
    let per = total / iters;
    println!("  {name:40} : {per:?} / iter (total {total:?} over {iters} iters)");
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

fn main() {
    let batch = env_usize("RUSTORCH_BENCH_BATCH", 8);
    let seq_len = env_usize("RUSTORCH_BENCH_SEQ", 512);
    let m = batch * seq_len;
    let hidden = env_usize("RUSTORCH_BENCH_HIDDEN", 2048);
    let ffn = env_usize("RUSTORCH_BENCH_FFN", 4096);
    let eps = 1e-6;
    let iters = env_usize("RUSTORCH_BENCH_STEPS", 5) as u32;

    println!("=== rustorch CUDA per-op breakdown (T246.11 TRAINING-BENCH.3) ===");
    println!("shape : tokens={m} hidden={hidden} ffn={ffn} iters={iters}");
    println!();

    let x = Variable::leaf(det_tensor(&[m, hidden], 1.0));
    let gamma = Variable::leaf(det_tensor(&[hidden], 0.7)).requires_grad(true);
    let w_gate = Variable::leaf(det_tensor(&[hidden, ffn], 0.5)).requires_grad(true);
    let w_up = Variable::leaf(det_tensor(&[hidden, ffn], 0.3)).requires_grad(true);
    let w_down = Variable::leaf(det_tensor(&[ffn, hidden], 0.2)).requires_grad(true);

    println!("--- forward only ---");

    time_op("rms_norm_cuda([m, hidden])", iters, || {
        let _ = rms_norm_cuda(&x, &gamma, eps).expect("rms");
    });

    let n = rms_norm_cuda(&x, &gamma, eps).expect("rms");

    time_op("matmul_bf16_cuda([m, h] @ [h, ffn])", iters, || {
        let _ = matmul_bf16_cuda(&n, &w_gate).expect("mm");
    });

    let gate = matmul_bf16_cuda(&n, &w_gate).expect("mm gate");
    let up = matmul_bf16_cuda(&n, &w_up).expect("mm up");

    time_op("swiglu_cuda([m, ffn])", iters, || {
        let _ = swiglu_cuda(&gate, &up).expect("swiglu");
    });

    let act = swiglu_cuda(&gate, &up).expect("swiglu");

    time_op("matmul_bf16_cuda([m, ffn] @ [ffn, h])", iters, || {
        let _ = matmul_bf16_cuda(&act, &w_down).expect("mm down");
    });

    println!();
    println!("--- full fwd + bwd ---");

    let total_t0 = Instant::now();
    for _ in 0..iters {
        // Reset grads on each iter.
        gamma.zero_grad();
        w_gate.zero_grad();
        w_up.zero_grad();
        w_down.zero_grad();

        let n = rms_norm_cuda(&x, &gamma, eps).expect("rms");
        let gate = matmul_bf16_cuda(&n, &w_gate).expect("mm gate");
        let up = matmul_bf16_cuda(&n, &w_up).expect("mm up");
        let act = swiglu_cuda(&gate, &up).expect("swiglu");
        let h = matmul_bf16_cuda(&act, &w_down).expect("mm down");
        let y = ops::add(&h, &x).expect("res");
        let loss = ops::sum(&y).expect("loss");
        backward(&loss, None).expect("bwd");
    }
    let total = total_t0.elapsed();
    println!("  full fwd+bwd / iter : {:?}", total / iters);
    println!();
    println!("=== done ===");
}
