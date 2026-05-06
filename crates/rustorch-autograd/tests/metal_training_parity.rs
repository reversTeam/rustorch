//! End-to-end CPU vs Metal training-loop parity.
//!
//! Validates that the Metal-direct backend produces the **same training
//! trajectory** as the CPU reference for the canonical
//! `Linear 1024² + MSE + AdamW` workload — same input, same init,
//! same hyperparameters, N steps. The two paths share the autograd
//! graph but dispatch through different backends; if any kernel
//! (linear forward, MSE loss, MatMulBackward, AddBiasBackward,
//! sum_dim, FusedAdamW in-place, …) drifts, the per-step loss
//! sequences will diverge.
//!
//! Tolerance: cosine similarity ≥ 0.9999 on the loss sequence + the
//! final weight tensor. Strict element-wise equality is unrealistic
//! because (a) GPU FMA reordering, (b) Metal's f32 simdgroup_matrix
//! aggregates 8×K multiply-add in a different order than the CPU
//! reference, and (c) bias-corrected AdamW amplifies tiny drift over
//! steps.
//!
//! Run with:
//! ```sh
//! cargo test -p rustorch-autograd --features metal,gpu-tests --test metal_training_parity
//! ```

#![cfg(all(feature = "metal", target_os = "macos"))]

use rustorch_autograd::{backward, ops, Variable};
use rustorch_core::tensor::device::Device;
use rustorch_core::tensor::tensor_impl::Tensor;
use rustorch_cpu::backend::Reduction;
use rustorch_metal::transfer::tensor_to_cpu;
use rustorch_optim::{AdamW, Optimizer};

/// Build a deterministic F32 tensor (sin-based seeding so values
/// span (-1, 1) without all-zero patches).
fn det_tensor(shape: &[usize], seed: f32) -> Tensor {
    let n: usize = shape.iter().product();
    let data: Vec<f32> = (0..n)
        .map(|i| ((i as f32 + 1.0) * seed * 0.001).sin())
        .collect();
    Tensor::from_vec(shape.to_vec(), data).expect("det_tensor")
}

/// Materialise a Tensor on the host and return a fresh `Vec<f32>`.
fn cpu_data(t: &Tensor) -> Vec<f32> {
    if t.device() == Device::Metal {
        let host = tensor_to_cpu(t).expect("metal tensor_to_cpu");
        host.as_slice::<f32>()
            .expect("after tensor_to_cpu, storage must be CPU")
            .to_vec()
    } else {
        t.as_slice::<f32>()
            .expect("CPU tensor must have f32 slice")
            .to_vec()
    }
}

/// Cosine similarity between two flat F32 buffers.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len(), "cosine_similarity: length mismatch");
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for (&x, &y) in a.iter().zip(b.iter()) {
        let xf = x as f64;
        let yf = y as f64;
        dot += xf * yf;
        na += xf * xf;
        nb += yf * yf;
    }
    if na == 0.0 || nb == 0.0 {
        return 1.0;
    }
    dot / (na.sqrt() * nb.sqrt())
}

/// Run `n_steps` of the canonical Linear+MSE+AdamW training loop on
/// `device`. Returns (per-step loss sequence, final weight values).
fn train_steps(
    device: Device,
    b: usize,
    m: usize,
    n: usize,
    n_steps: usize,
) -> (Vec<f32>, Vec<f32>) {
    let xs = det_tensor(&[b, m], 1.0).with_device(device);
    let ys = det_tensor(&[b, n], 0.5).with_device(device);
    let w = Variable::leaf(det_tensor(&[m, n], 0.1).with_device(device)).requires_grad(true);
    let bias = Variable::leaf(det_tensor(&[n], 0.05).with_device(device)).requires_grad(true);

    let xs_var = Variable::new(xs);
    let ys_var = Variable::new(ys);

    let mut opt = AdamW::new(vec![w.clone(), bias.clone()], 0.001);

    let mut losses = Vec::with_capacity(n_steps);
    for _ in 0..n_steps {
        opt.zero_grad();
        let pred = ops::linear(&xs_var, &w, &bias).expect("linear");
        let loss = ops::mse_loss(&pred, &ys_var, Reduction::Mean).expect("mse_loss");
        let loss_val = cpu_data(&loss.tensor())[0];
        losses.push(loss_val);
        backward(&loss, None).expect("backward");
        opt.step();
    }
    let final_w = cpu_data(&w.tensor());
    (losses, final_w)
}

#[test]
#[cfg_attr(not(feature = "gpu-tests"), ignore = "needs a GPU adapter")]
fn training_loop_cpu_vs_metal_loss_and_weights_match() {
    // Smaller bench than the perf one — we want a quick test, but the
    // shapes still hit the m%16==0 / n%256==0 fast path so the
    // coarsened-wide / b_t / a_t kernels run.
    let b = 32;
    let m = 256;
    let n = 256;
    let n_steps = 5;

    let (cpu_losses, cpu_w) = train_steps(Device::Cpu, b, m, n, n_steps);
    let (metal_losses, metal_w) = train_steps(Device::Metal, b, m, n, n_steps);

    // Loss sequences must match closely. We compare both element-wise
    // (each step's loss) and via cosine similarity on the sequence.
    assert_eq!(cpu_losses.len(), metal_losses.len());
    let loss_cs = cosine_similarity(&cpu_losses, &metal_losses);
    assert!(
        loss_cs > 0.9999,
        "loss-sequence cosine similarity too low: {loss_cs:.6}\n  CPU: {cpu_losses:?}\n  Metal: {metal_losses:?}"
    );

    // Per-step relative error tolerance (FMA reordering accumulates
    // across the AdamW step). 1% over 5 steps is comfortable; tighter
    // checks happen at the kernel-parity level.
    for (i, (c, g)) in cpu_losses.iter().zip(metal_losses.iter()).enumerate() {
        let rel = ((c - g).abs() / c.abs().max(1e-6)) as f64;
        assert!(
            rel < 1e-2,
            "step {i}: loss diverged — CPU={c:.6} Metal={g:.6} rel_err={rel:.4e}"
        );
    }

    // Final weights: the trained model should converge to the same
    // point on both backends.
    assert_eq!(cpu_w.len(), metal_w.len());
    let w_cs = cosine_similarity(&cpu_w, &metal_w);
    assert!(
        w_cs > 0.9999,
        "final-weight cosine similarity too low: {w_cs:.6}"
    );
}

#[test]
#[cfg_attr(not(feature = "gpu-tests"), ignore = "needs a GPU adapter")]
fn training_loop_cpu_vs_metal_bench_shape() {
    // Same shape as the perf bench (1024² Linear, B=64). Confirms the
    // numerical correctness of every kernel exercised in the public
    // BEATS-MPS result.
    let b = 64;
    let m = 1024;
    let n = 1024;
    let n_steps = 5;

    let (cpu_losses, _) = train_steps(Device::Cpu, b, m, n, n_steps);
    let (metal_losses, _) = train_steps(Device::Metal, b, m, n, n_steps);

    let loss_cs = cosine_similarity(&cpu_losses, &metal_losses);
    assert!(
        loss_cs > 0.9999,
        "bench-shape loss-sequence cosine similarity too low: {loss_cs:.6}\n  CPU: {cpu_losses:?}\n  Metal: {metal_losses:?}"
    );
    for (i, (c, g)) in cpu_losses.iter().zip(metal_losses.iter()).enumerate() {
        let rel = ((c - g).abs() / c.abs().max(1e-6)) as f64;
        assert!(
            rel < 1e-2,
            "bench-shape step {i}: CPU={c:.6} Metal={g:.6} rel_err={rel:.4e}"
        );
    }
}
