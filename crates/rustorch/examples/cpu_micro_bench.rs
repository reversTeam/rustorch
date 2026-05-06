//! CPU micro-bench: time the individual phases of one training step
//! at the bench shape (Linear 1024² + MSE + AdamW). Pinpoints the
//! remaining bottleneck after the Accelerate / parallel AdamW push.

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

fn main() {
    let b = 64;
    let m = 1024;
    let n = 1024;
    let n_iter = 100;

    let xs = det_tensor(&[b, m], 1.0).with_device(Device::Cpu);
    let ys = det_tensor(&[b, n], 0.5).with_device(Device::Cpu);
    let w = Variable::leaf(det_tensor(&[m, n], 0.1)).requires_grad(true);
    let bias = Variable::leaf(det_tensor(&[n], 0.05)).requires_grad(true);

    let xs_var = Variable::new(xs.clone());
    let ys_var = Variable::new(ys);

    let mut opt = AdamW::new(vec![w.clone(), bias.clone()], 0.001);

    // Warmup.
    for _ in 0..3 {
        opt.zero_grad();
        let pred = ops::linear(&xs_var, &w, &bias).expect("linear");
        let loss = ops::mse_loss(&pred, &ys_var, Reduction::Mean).expect("mse");
        backward(&loss, None).expect("bw");
        opt.step();
    }

    println!(
        "\n=== CPU step phase breakdown (Linear 1024² + MSE + AdamW, B={b}, {n_iter} iters) ===\n"
    );

    // Phase 1: linear forward (matmul + bias).
    let t = Instant::now();
    let mut last_pred = None;
    for _ in 0..n_iter {
        last_pred = Some(ops::linear(&xs_var, &w, &bias).expect("linear"));
    }
    let dt = t.elapsed();
    println!(
        "  linear forward (matmul+bias):     {:>8.1?} per iter",
        dt / n_iter
    );
    let pred = last_pred.unwrap();

    // Phase 2: mse_loss forward (allocates fresh diff every iter).
    let t = Instant::now();
    let mut last_loss = None;
    for _ in 0..n_iter {
        last_loss = Some(ops::mse_loss(&pred, &ys_var, Reduction::Mean).expect("mse"));
    }
    let dt = t.elapsed();
    println!(
        "  mse_loss forward (sub+mul+mean):  {:>8.1?} per iter",
        dt / n_iter
    );
    let _loss = last_loss.unwrap();

    // Phase 3: full forward+backward+optim (no slicing).
    let mut t_opt_total = std::time::Duration::ZERO;
    let mut t_bw_total = std::time::Duration::ZERO;
    for _ in 0..n_iter {
        opt.zero_grad();
        let pred = ops::linear(&xs_var, &w, &bias).expect("linear");
        let loss = ops::mse_loss(&pred, &ys_var, Reduction::Mean).expect("mse");
        let t = Instant::now();
        backward(&loss, None).expect("bw");
        t_bw_total += t.elapsed();
        let t = Instant::now();
        opt.step();
        t_opt_total += t.elapsed();
    }
    println!(
        "  backward (3 matmuls + sum_dim):   {:>8.1?} per iter",
        t_bw_total / n_iter
    );
    println!(
        "  AdamW step (2 params):            {:>8.1?} per iter",
        t_opt_total / n_iter
    );

    // Direct kernel timing — bypass autograd entirely. The Backend
    // trait import is needed for the `.matmul`/`.sum_dim`/etc. method
    // resolution; clippy's unused-import lint can't see through that.
    use rustorch_cpu::cpu_backend;
    #[allow(unused_imports)]
    use rustorch_cpu::Backend;
    let cpu = cpu_backend();
    let xs_t = det_tensor(&[b, m], 1.0);
    let ys_t = det_tensor(&[b, n], 0.5);
    let w_t = det_tensor(&[m, n], 0.1);
    let grad_t = det_tensor(&[b, n], 0.7); // proxy upstream grad

    let t = Instant::now();
    for _ in 0..n_iter {
        let _ = cpu.matmul(&xs_t, &w_t).unwrap();
    }
    println!(
        "\n  raw matmul (Accelerate):          {:>8.1?} per iter",
        t.elapsed() / n_iter
    );

    let t = Instant::now();
    for _ in 0..n_iter {
        let _ = cpu
            .matmul_with_transposes(&grad_t, &w_t, false, true)
            .unwrap();
    }
    println!(
        "  raw matmul_with_transposes(B=T):  {:>8.1?} per iter",
        t.elapsed() / n_iter
    );

    let t = Instant::now();
    for _ in 0..n_iter {
        let _ = cpu
            .matmul_with_transposes(&xs_t, &grad_t, true, false)
            .unwrap();
    }
    println!(
        "  raw matmul_with_transposes(A=T):  {:>8.1?} per iter",
        t.elapsed() / n_iter
    );

    let t = Instant::now();
    for _ in 0..n_iter {
        let _ = cpu.sum_dim(&grad_t, &[0], false).unwrap();
    }
    println!(
        "  raw sum_dim axis=0:               {:>8.1?} per iter",
        t.elapsed() / n_iter
    );

    let t = Instant::now();
    for _ in 0..n_iter {
        let diff = cpu.sub(&grad_t, &ys_t).unwrap();
        let _sq = cpu.mul(&diff, &diff).unwrap();
    }
    println!(
        "  raw sub+mul (mse_loss core):      {:>8.1?} per iter",
        t.elapsed() / n_iter
    );

    // Phase 4: full step.
    let t = Instant::now();
    for _ in 0..n_iter {
        opt.zero_grad();
        let pred = ops::linear(&xs_var, &w, &bias).expect("linear");
        let loss = ops::mse_loss(&pred, &ys_var, Reduction::Mean).expect("mse");
        backward(&loss, None).expect("bw");
        opt.step();
    }
    let dt = t.elapsed();
    println!(
        "\n  TOTAL step:                       {:>8.1?} per iter",
        dt / n_iter
    );
}
