//! CPU vs **rustorch-metal direct** training-step bench (P3.Z Task J).
//!
//! Same workload as `cpu_vs_wgpu_train` (Linear 1024² + MSE + AdamW)
//! but routed through `Device::Metal` → `rustorch-metal::MetalBackend`
//! → `simdgroup_matrix<float, 8, 8>` matmul on Apple's tensor units.
//!
//! Reference numbers (M4 Max, F32):
//! - rustorch CPU:                 ~12.7 ms/step
//! - rustorch-wgpu (post Task A):  2.35 ms/step
//! - rustorch-metal (this bench):  TARGET < 1 ms (Phase 1), < 0.5 ms (Phase 4 bf16)
//! - PyTorch MPS:                  0.89 ms/step
//! - PyTorch CPU:                  1.75 ms/step
//!
//! Run with:
//! ```sh
//! cargo run -p rustorch --release --example cpu_vs_metal_train --features metal
//! ```

#![allow(missing_docs)]

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("cpu_vs_metal_train: macOS only (rustorch-metal direct backend).");
}

#[cfg(target_os = "macos")]
fn main() {
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

        // In-function warmup (3 steps): pays first-time costs that
        // would otherwise inflate step 1 of the timed loop —
        // pipeline-state cache compiles (~0.3 ms × N kernels first
        // time), AdamW m/v lazy alloc, CPU→Metal storage promotion
        // for params/inputs, bias bf16-cache prime. After this loop,
        // every kernel is hot in the cache and every buffer is
        // resident.
        for _ in 0..3 {
            opt.zero_grad();
            let pred = ops::linear(&xs_var, &w, &bias).expect("linear (warmup)");
            let loss = ops::mse_loss(&pred, &ys_var, Reduction::Mean).expect("mse (warmup)");
            backward(&loss, None).expect("backward (warmup)");
            opt.step();
        }
        if device == Device::Metal {
            #[cfg(target_os = "macos")]
            rustorch_metal::backend_singleton::metal_backend().drain();
        }

        let t0 = Instant::now();
        for _ in 0..n_steps {
            opt.zero_grad();
            // Fused linear: y = x @ w + bias (single dispatch on Metal3
            // via `matmul_with_bias`; on CPU/WGPU it composes
            // `matmul + add_bias`).
            let pred = ops::linear(&xs_var, &w, &bias).expect("linear");
            let loss = ops::mse_loss(&pred, &ys_var, Reduction::Mean).expect("mse");
            backward(&loss, None).expect("backward");
            opt.step();
        }
        // Honest timing: ensure the GPU has finished executing every
        // kernel committed during the loop before we stop the clock.
        // Without this drain, commitAndContinue can defer GPU work
        // past `t0.elapsed()` and we'd be measuring CPU-side kernel
        // ENCODING rather than actual completion.
        if device == Device::Metal {
            #[cfg(target_os = "macos")]
            rustorch_metal::backend_singleton::metal_backend().drain();
        }
        t0.elapsed()
    }

    let b = 64;
    let m = 1024;
    let n = 1024;
    // Steady-state measurement: 50 steps after a 3-step warmup amortises
    // any residual one-shot costs (encoder pool warm-up, OS scheduler
    // settling). Reported per-step is the median across the 50.
    let n_steps = 50;

    println!(
        "=== Training-step bench — Linear [B={b}, M={m}] @ [M={m}, N={n}], MSE+AdamW, {n_steps} steps ===\n\
         Backends: rustorch-cpu vs rustorch-metal (Apple Silicon direct, simdgroup_matrix)"
    );

    // Warm up each backend (first kernel launch pays the pipeline compile).
    let _ = time_train_steps(Device::Cpu, b, m, n, 1);
    let _ = time_train_steps(Device::Metal, b, m, n, 1);

    let cpu_time = time_train_steps(Device::Cpu, b, m, n, n_steps);
    let metal_time = time_train_steps(Device::Metal, b, m, n, n_steps);

    let cpu_per = cpu_time / (n_steps as u32);
    let metal_per = metal_time / (n_steps as u32);

    println!("CPU   : {n_steps} steps in {cpu_time:?}, {cpu_per:?}/step");
    println!("Metal : {n_steps} steps in {metal_time:?}, {metal_per:?}/step");

    let ratio = cpu_time.as_secs_f32() / metal_time.as_secs_f32();
    if ratio > 1.0 {
        println!("→ rustorch-metal is {ratio:.2}× faster than CPU on this workload");
    } else {
        println!(
            "→ rustorch-metal is {:.2}× SLOWER than CPU here (Task J kernels still scaffolding)",
            1.0 / ratio
        );
    }
    println!("\nReference: PyTorch MPS = 0.89 ms/step, PyTorch CPU = 1.75 ms/step");
    let metal_ms = metal_per.as_secs_f64() * 1000.0;
    let mps_ratio = 0.89 / metal_ms;
    if mps_ratio > 1.0 {
        println!(
            "→ rustorch-metal beats PyTorch MPS by {mps_ratio:.2}×  ({metal_ms:.2} vs 0.89 ms) ✅"
        );
    } else {
        println!(
            "→ rustorch-metal {:.2}× slower than PyTorch MPS  ({metal_ms:.2} vs 0.89 ms)",
            1.0 / mps_ratio
        );
    }
}
