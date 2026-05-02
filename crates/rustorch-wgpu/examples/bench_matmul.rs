//! Quick CPU vs GPU matmul benchmark for the rustorch-wgpu crate.
//!
//! Run with `cargo run --release -p rustorch-wgpu --example bench_matmul`.

use rustorch_core::tensor::tensor_impl::Tensor;
use rustorch_wgpu::{matmul, to_cpu, to_gpu, WgpuBackend};
use std::time::Instant;

fn cpu_matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0.0_f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut s = 0.0_f32;
            for kk in 0..k {
                s += a[i * k + kk] * b[kk * n + j];
            }
            out[i * n + j] = s;
        }
    }
    out
}

fn bench_one(backend: &WgpuBackend, m: usize, k: usize, n: usize, repeat: usize) {
    let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.001).collect();
    let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.0005).collect();

    // CPU
    let t0 = Instant::now();
    let mut cpu_out = vec![];
    for _ in 0..repeat {
        cpu_out = cpu_matmul(&a, &b, m, k, n);
    }
    let cpu_total = t0.elapsed();
    let cpu_per = cpu_total / repeat as u32;

    // GPU
    let ta = Tensor::from_vec([m, k], a.clone()).unwrap();
    let tb = Tensor::from_vec([k, n], b.clone()).unwrap();
    let ga = to_gpu(backend, &ta).unwrap();
    let gb = to_gpu(backend, &tb).unwrap();
    // Warm-up
    let _ = matmul(backend, &ga, &gb, m, k, n).unwrap();

    let t0 = Instant::now();
    let mut gpu_out = vec![];
    for _ in 0..repeat {
        let gc = matmul(backend, &ga, &gb, m, k, n).unwrap();
        let c = to_cpu(backend, &gc, vec![m, n]).unwrap();
        gpu_out = c.as_slice::<f32>().unwrap().to_vec();
    }
    let gpu_total = t0.elapsed();
    let gpu_per = gpu_total / repeat as u32;

    // Parity check
    let mut max_err = 0.0_f32;
    for (g, c) in gpu_out.iter().zip(&cpu_out) {
        max_err = max_err.max((g - c).abs());
    }

    let flops = 2.0 * m as f64 * k as f64 * n as f64;
    let cpu_gflops = flops / cpu_per.as_secs_f64() / 1e9;
    let gpu_gflops = flops / gpu_per.as_secs_f64() / 1e9;
    let speedup = cpu_per.as_secs_f64() / gpu_per.as_secs_f64();

    println!(
        "  [M={:>4}, K={:>4}, N={:>4}]  CPU {:>8.2} ms ({:>5.1} GF/s)  | GPU {:>8.2} ms ({:>5.1} GF/s)  | speedup {:>5.2}x  | max_err {:.2e}",
        m,
        k,
        n,
        cpu_per.as_secs_f64() * 1000.0,
        cpu_gflops,
        gpu_per.as_secs_f64() * 1000.0,
        gpu_gflops,
        speedup,
        max_err
    );
}

fn main() {
    println!("Initializing wgpu backend…");
    let backend = WgpuBackend::new_blocking().expect("init wgpu");
    println!("Backend OK.");
    println!();
    println!("Matmul C = A @ B   (A: [M,K], B: [K,N], C: [M,N])");
    println!("---------------------------------------------------");
    bench_one(&backend, 64, 64, 64, 30);
    bench_one(&backend, 128, 128, 128, 20);
    bench_one(&backend, 256, 256, 256, 10);
    bench_one(&backend, 512, 512, 512, 5);
    bench_one(&backend, 1024, 1024, 1024, 3);
}
