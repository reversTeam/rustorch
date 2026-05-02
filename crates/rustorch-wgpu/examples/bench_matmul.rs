//! CPU (sequential + parallel) vs GPU matmul benchmark.
//!
//! Run with `cargo run --release -p rustorch-wgpu --example bench_matmul`.

use rayon::prelude::*;
use rustorch_core::tensor::tensor_impl::Tensor;
use rustorch_wgpu::{matmul, to_cpu, to_gpu, WgpuBackend};
use std::time::{Duration, Instant};

/// Naive single-threaded matmul. The triple-loop reference everyone
/// has in their toolbox.
fn cpu_naive(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
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

/// Parallel matmul: 1 row block per rayon thread, blocked-K with
/// register-tile-friendly inner loop. Roughly representative of what a
/// half-decent multi-threaded CPU baseline would do without dragging
/// in a real BLAS.
fn cpu_rayon(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0.0_f32; m * n];
    out.par_chunks_mut(n).enumerate().for_each(|(i, row_out)| {
        let a_row = &a[i * k..(i + 1) * k];
        for j in 0..n {
            let mut s = 0.0_f32;
            // Pre-fetch the j-th column of B once per row to help the
            // compiler schedule the multiply-adds.
            for kk in 0..k {
                s += a_row[kk] * b[kk * n + j];
            }
            row_out[j] = s;
        }
    });
    out
}

fn time_it<F: FnMut()>(repeat: usize, mut f: F) -> Duration {
    let t0 = Instant::now();
    for _ in 0..repeat {
        f();
    }
    t0.elapsed() / repeat as u32
}

fn bench_one(backend: &WgpuBackend, m: usize, k: usize, n: usize, repeat: usize) {
    let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.001).collect();
    let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.0005).collect();

    let mut cpu_out = vec![];
    let cpu_naive_per = time_it(repeat, || cpu_out = cpu_naive(&a, &b, m, k, n));
    let cpu_rayon_per = time_it(repeat.max(2), || {
        cpu_out = cpu_rayon(&a, &b, m, k, n);
    });

    let ta = Tensor::from_vec([m, k], a.clone()).unwrap();
    let tb = Tensor::from_vec([k, n], b.clone()).unwrap();
    let ga = to_gpu(backend, &ta).unwrap();
    let gb = to_gpu(backend, &tb).unwrap();
    let _ = matmul(backend, &ga, &gb, m, k, n).unwrap(); // warm-up

    let mut gpu_out = vec![];
    let gpu_per = time_it(repeat, || {
        let gc = matmul(backend, &ga, &gb, m, k, n).unwrap();
        let c = to_cpu(backend, &gc, vec![m, n]).unwrap();
        gpu_out = c.as_slice::<f32>().unwrap().to_vec();
    });

    // Relative-error parity check (matmul accumulation order differs
    // between tiled GPU and sequential CPU; absolute error grows with
    // K but the relative error stays small on values >> 1).
    let mut max_rel_err = 0.0_f32;
    for (g, c) in gpu_out.iter().zip(&cpu_out) {
        let scale = c.abs().max(1e-3);
        max_rel_err = max_rel_err.max((g - c).abs() / scale);
    }

    let flops = 2.0 * m as f64 * k as f64 * n as f64;
    let g_gflops = |d: Duration| flops / d.as_secs_f64() / 1e9;
    let speedup_naive = cpu_naive_per.as_secs_f64() / gpu_per.as_secs_f64();
    let speedup_rayon = cpu_rayon_per.as_secs_f64() / gpu_per.as_secs_f64();

    println!(
        "  [M={:>4}, K={:>4}, N={:>4}]  naive {:>7.2} ms ({:>5.1} GF/s)  | rayon {:>7.2} ms ({:>5.1} GF/s)  | GPU {:>7.2} ms ({:>5.1} GF/s)  | speedup vs naive {:>6.2}x | vs rayon {:>5.2}x | max_rel_err {:.2e}",
        m, k, n,
        cpu_naive_per.as_secs_f64() * 1000.0, g_gflops(cpu_naive_per),
        cpu_rayon_per.as_secs_f64() * 1000.0, g_gflops(cpu_rayon_per),
        gpu_per.as_secs_f64() * 1000.0, g_gflops(gpu_per),
        speedup_naive, speedup_rayon, max_rel_err
    );
}

fn main() {
    println!("Initializing wgpu backend…");
    let backend = WgpuBackend::new_blocking().expect("init wgpu");
    println!(
        "Backend OK. rayon threads = {}",
        rayon::current_num_threads()
    );
    println!();
    println!("Matmul C = A @ B   (A: [M,K], B: [K,N], C: [M,N])");
    println!("------------------------------------------------------------------------");
    bench_one(&backend, 64, 64, 64, 30);
    bench_one(&backend, 128, 128, 128, 20);
    bench_one(&backend, 256, 256, 256, 10);
    bench_one(&backend, 512, 512, 512, 5);
    bench_one(&backend, 1024, 1024, 1024, 3);
}
