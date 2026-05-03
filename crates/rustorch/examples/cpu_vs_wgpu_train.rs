//! CPU vs Wgpu training-step bench (P3.Y plan, Phase F step 28c18803).
//!
//! Times one full training step (forward + loss + backward + AdamW)
//! on CPU vs Wgpu for a Linear 1024×1024 layer. Reports the ratio.
//!
//! Not a Criterion bench (which would add a heavy dev-dep) but a simple
//! `Instant::now()` timing — sufficient to surface the order-of-magnitude
//! comparison. Real perf comes when Storage Option A lands and removes
//! the per-op host↔device round-trip.
//!
//! Run with:
//! ```sh
//! cargo run -p rustorch --release --example cpu_vs_wgpu_train --features wgpu
//! ```

#![allow(missing_docs)]

use rustorch_autograd::{backward, ops, Variable};
use rustorch_core::tensor::device::Device;
use rustorch_core::tensor::tensor_impl::Tensor;
use rustorch_cpu::backend::Reduction;
use rustorch_optim::{AdamW, Optimizer};
use std::time::Instant;

fn det_tensor(shape: &[usize], seed: f32) -> Tensor {
    let n: usize = shape.iter().product();
    let data: Vec<f32> = (0..n)
        .map(|i| ((i as f32 + 1.0) * seed * 0.001).sin())
        .collect();
    Tensor::from_vec(shape.to_vec(), data).expect("det_tensor")
}

/// Run `n_steps` of (forward + MSE loss + backward + AdamW.step) for a
/// Linear layer X[B, M] @ W[M, N] = Y[B, N], targets ones-like(Y), all
/// on the given device. Returns total wall-clock time.
fn time_train_steps(
    device: Device,
    b: usize,
    m: usize,
    n: usize,
    n_steps: usize,
) -> std::time::Duration {
    let xs = det_tensor(&[b, m], 1.0).with_device(device);
    let ys = det_tensor(&[b, n], 0.5).with_device(device);
    let w = Variable::leaf(det_tensor(&[m, n], 0.1).with_device(device)).requires_grad(true);
    let bias = Variable::leaf(det_tensor(&[n], 0.05).with_device(device)).requires_grad(true);

    let xs_var = Variable::new(xs);
    let ys_var = Variable::new(ys);

    let mut opt = AdamW::new(vec![w.clone(), bias.clone()], 0.001);

    let t0 = Instant::now();
    for _ in 0..n_steps {
        opt.zero_grad();
        let pred = ops::matmul(&xs_var, &w).expect("matmul");
        let pred = ops::add_bias(&pred, &bias).expect("add_bias");
        let loss = ops::mse_loss(&pred, &ys_var, Reduction::Mean).expect("mse");
        backward(&loss, None).expect("backward");
        opt.step();
    }
    t0.elapsed()
}

fn main() {
    let b = 64;
    let m = 1024;
    let n = 1024;
    let n_steps = 5;

    println!(
        "=== Training-step bench — Linear [B={b}, M={m}] @ [M={m}, N={n}], MSE+AdamW, {n_steps} steps ==="
    );

    // Warm up each backend (first kernel launch is slower due to pipeline build).
    let _ = time_train_steps(Device::Cpu, b, m, n, 1);
    let _ = time_train_steps(Device::Wgpu, b, m, n, 1);

    let cpu_time = time_train_steps(Device::Cpu, b, m, n, n_steps);
    let gpu_time = time_train_steps(Device::Wgpu, b, m, n, n_steps);

    let cpu_per = cpu_time / (n_steps as u32);
    let gpu_per = gpu_time / (n_steps as u32);

    println!("CPU  : {n_steps} steps in {cpu_time:?}, {cpu_per:?}/step");
    println!("Wgpu : {n_steps} steps in {gpu_time:?}, {gpu_per:?}/step");

    let ratio = cpu_time.as_secs_f32() / gpu_time.as_secs_f32();
    if ratio > 1.0 {
        println!("→ Wgpu is {ratio:.2}× faster than CPU on this workload");
    } else {
        println!(
            "→ Wgpu is {:.2}× SLOWER than CPU here (Storage Option B's per-op round-trip dominates)",
            1.0 / ratio
        );
    }
    println!("=== done ===");
}
