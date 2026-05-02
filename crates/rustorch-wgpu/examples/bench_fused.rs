//! Fused-vs-unfused linear+activation benchmark.
//!
//! Run with `cargo run --release -p rustorch-wgpu --example bench_fused`.

use rustorch_core::tensor::tensor_impl::Tensor;
use rustorch_wgpu::{
    dispatch_binary, dispatch_unary, linear_gelu_fused, linear_relu_fused, matmul, to_cpu, to_gpu,
    WgpuBackend,
};
use std::time::{Duration, Instant};

fn time_it<F: FnMut()>(repeat: usize, mut f: F) -> Duration {
    let t0 = Instant::now();
    for _ in 0..repeat {
        f();
    }
    t0.elapsed() / repeat as u32
}

fn bench_one(backend: &WgpuBackend, m: usize, k: usize, n: usize, repeat: usize) {
    let x: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.001 - 0.5).collect();
    let w: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.0005 - 0.25).collect();
    let bias: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01).collect();
    let bias_bcast: Vec<f32> = (0..m).flat_map(|_| bias.clone()).collect();

    let tx = Tensor::from_vec([m, k], x).unwrap();
    let tw = Tensor::from_vec([k, n], w).unwrap();
    let tb = Tensor::from_vec([n], bias).unwrap();
    let tb_bcast = Tensor::from_vec([m, n], bias_bcast).unwrap();
    let gx = to_gpu(backend, &tx).unwrap();
    let gw = to_gpu(backend, &tw).unwrap();
    let gb = to_gpu(backend, &tb).unwrap();
    let gb_bcast = to_gpu(backend, &tb_bcast).unwrap();

    // Warm-up.
    let _ = linear_relu_fused(backend, &gx, &gw, &gb, m, k, n).unwrap();

    // ----- Unfused: matmul → add bias → relu (3 dispatches, 2 readback writes)
    let unfused_per = time_it(repeat, || {
        let mm = matmul(backend, &gx, &gw, m, k, n).unwrap();
        let pre = dispatch_binary(backend, "add", &mm, &gb_bcast).unwrap();
        let _ = dispatch_unary(backend, "relu", &pre).unwrap();
    });

    // ----- Fused: 1 dispatch
    let fused_per = time_it(repeat, || {
        let _ = linear_relu_fused(backend, &gx, &gw, &gb, m, k, n).unwrap();
    });

    // ----- Same for GELU
    let gelu_per = time_it(repeat, || {
        let _ = linear_gelu_fused(backend, &gx, &gw, &gb, m, k, n).unwrap();
    });

    // Read back the fused result so we know the GPU is actually done
    // before the next iteration.
    let g_check = linear_relu_fused(backend, &gx, &gw, &gb, m, k, n).unwrap();
    let _ = to_cpu(backend, &g_check, vec![m, n]).unwrap();

    let speedup = unfused_per.as_secs_f64() / fused_per.as_secs_f64();
    println!(
        "  [M={:>4}, K={:>4}, N={:>4}]  unfused {:>7.3} ms  | fused-relu {:>7.3} ms ({:>4.2}x)  | fused-gelu {:>7.3} ms",
        m, k, n,
        unfused_per.as_secs_f64() * 1000.0,
        fused_per.as_secs_f64() * 1000.0,
        speedup,
        gelu_per.as_secs_f64() * 1000.0
    );
}

fn main() {
    let backend = WgpuBackend::new_blocking().expect("init wgpu");
    println!("Fused linear+activation benchmark");
    println!("  unfused = matmul → add bias → relu  (3 dispatches)");
    println!("  fused   = linear_relu_fused          (1 dispatch)");
    println!("---------------------------------------------------------");
    bench_one(&backend, 64, 64, 64, 30);
    bench_one(&backend, 128, 128, 128, 20);
    bench_one(&backend, 256, 256, 256, 10);
    bench_one(&backend, 512, 512, 512, 5);
    bench_one(&backend, 1024, 1024, 1024, 3);
}
