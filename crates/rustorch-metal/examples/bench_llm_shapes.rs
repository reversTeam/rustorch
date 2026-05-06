//! Compare Metal `simdgroup_matrix<f32>` vs Apple Accelerate AMX
//! `cblas_sgemm` on the exact shapes used by Qwen3-14B FFN/attention
//! projections. Drives the decision of whether to wire Metal into the
//! `rustorch-llm` decode hot path (Phase 2 of the LLM perf push).
//!
//! All Metal timings are bracketed by `backend.drain()` so we measure
//! committed compute time, not just the dispatch enqueue.
//!
//! Run with:
//! ```sh
//! cargo run -p rustorch-metal --release --example bench_llm_shapes
//! ```

#![cfg(target_os = "macos")]
#![allow(missing_docs)]

use rustorch_fusion::patterns::matmul_bias_act::{fused_matmul_bias_activation, Activation};
use rustorch_metal::backend_singleton::metal_backend;
use rustorch_metal::kernels::{
    matmul_simdgroup_f32, matmul_simdgroup_f32_coarsened, matmul_simdgroup_f32_coarsened_wide,
    matmul_simdgroup_f32_multisg, sgemv_f32_simd,
};
use std::time::Instant;

fn det_vec(n: usize, seed: f32) -> Vec<f32> {
    (0..n)
        .map(|i| ((i as f32 + 1.0) * seed * 0.001).sin())
        .collect()
}

/// Native M=1 sgemv via the new `sgemv_f32_simd` kernel.
fn time_metal_sgemv(k: usize, n: usize, n_iter: usize) -> std::time::Duration {
    let backend = metal_backend();
    let x_buf = backend.alloc_shared(k * 4).expect("x alloc");
    let w_buf = backend.alloc_shared(k * n * 4).expect("w alloc");
    unsafe {
        let px = x_buf.contents() as *mut f32;
        for (i, val) in det_vec(k, 1.0).iter().enumerate() {
            *px.add(i) = *val;
        }
        let pw = w_buf.contents() as *mut f32;
        for (i, val) in det_vec(k * n, 0.5).iter().enumerate() {
            *pw.add(i) = *val;
        }
    }
    // Warmup.
    let _ = sgemv_f32_simd(backend, &x_buf, &w_buf, k, n).unwrap();
    backend.drain();
    let t0 = Instant::now();
    for _ in 0..n_iter {
        let _ = sgemv_f32_simd(backend, &x_buf, &w_buf, k, n).unwrap();
    }
    backend.drain();
    t0.elapsed()
}

/// CPU AMX path with M=1 (the real shape during decode).
fn time_cpu_m1(k: usize, n: usize, n_iter: usize) -> std::time::Duration {
    let a = det_vec(k, 1.0);
    let b = det_vec(k * n, 0.5);
    let mut c = vec![0.0_f32; n];
    fused_matmul_bias_activation(&a, &b, None, &mut c, 1, k, n, Activation::None).unwrap();
    let t0 = Instant::now();
    for _ in 0..n_iter {
        fused_matmul_bias_activation(&a, &b, None, &mut c, 1, k, n, Activation::None).unwrap();
    }
    t0.elapsed()
}

fn time_metal(m: usize, k: usize, n: usize, n_iter: usize) -> std::time::Duration {
    let backend = metal_backend();
    let a_buf = backend.alloc_shared(m * k * 4).expect("a alloc");
    let b_buf = backend.alloc_shared(k * n * 4).expect("b alloc");
    unsafe {
        let pa = a_buf.contents() as *mut f32;
        for (i, val) in det_vec(m * k, 1.0).iter().enumerate() {
            *pa.add(i) = *val;
        }
        let pb = b_buf.contents() as *mut f32;
        for (i, val) in det_vec(k * n, 0.5).iter().enumerate() {
            *pb.add(i) = *val;
        }
    }

    // Pick the fastest variant for the shape.
    let dispatch = |a: &metal::Buffer, b: &metal::Buffer| -> metal::Buffer {
        if m % 16 == 0 && n % 256 == 0 {
            matmul_simdgroup_f32_coarsened_wide(backend, a, b, m, k, n).unwrap()
        } else if n % 256 == 0 {
            matmul_simdgroup_f32_coarsened(backend, a, b, m, k, n).unwrap()
        } else if n % 64 == 0 {
            matmul_simdgroup_f32_multisg(backend, a, b, m, k, n).unwrap()
        } else {
            matmul_simdgroup_f32(backend, a, b, m, k, n).unwrap()
        }
    };

    // Warmup (compile pipeline + page in buffers + measure dispatch only).
    let _ = dispatch(&a_buf, &b_buf);
    backend.drain();

    let t0 = Instant::now();
    for _ in 0..n_iter {
        let _ = dispatch(&a_buf, &b_buf);
    }
    // Single drain at the end → average dispatch + compute time.
    backend.drain();
    t0.elapsed()
}

fn time_cpu(m: usize, k: usize, n: usize, n_iter: usize) -> std::time::Duration {
    let a = det_vec(m * k, 1.0);
    let b = det_vec(k * n, 0.5);
    let mut c = vec![0.0_f32; m * n];

    // Warmup.
    fused_matmul_bias_activation(&a, &b, None, &mut c, m, k, n, Activation::None).unwrap();

    let t0 = Instant::now();
    for _ in 0..n_iter {
        fused_matmul_bias_activation(&a, &b, None, &mut c, m, k, n, Activation::None).unwrap();
    }
    t0.elapsed()
}

fn flops(m: usize, k: usize, n: usize) -> f64 {
    2.0 * m as f64 * k as f64 * n as f64
}

fn run(label: &str, m: usize, k: usize, n: usize, n_iter: usize) {
    let metal_d = time_metal(m, k, n, n_iter);
    let cpu_d = time_cpu(m, k, n, n_iter);
    let metal_per = metal_d / n_iter as u32;
    let cpu_per = cpu_d / n_iter as u32;
    let metal_gflops = flops(m, k, n) / metal_per.as_secs_f64() / 1e9;
    let cpu_gflops = flops(m, k, n) / cpu_per.as_secs_f64() / 1e9;
    let speedup = cpu_per.as_secs_f64() / metal_per.as_secs_f64();
    println!(
        "{label:<28} [{m:>4}×{k:>5}] @ [{k:>5}×{n:>6}]  Metal {:>8.3}ms ({:>5.0} GFLOPS)  CPU AMX {:>8.3}ms ({:>5.0} GFLOPS)  speedup {:>4.1}×",
        metal_per.as_secs_f64() * 1e3,
        metal_gflops,
        cpu_per.as_secs_f64() * 1e3,
        cpu_gflops,
        speedup,
    );
}

fn run_sgemv(label: &str, k: usize, n: usize, n_iter: usize) {
    let metal_d = time_metal_sgemv(k, n, n_iter);
    let cpu_d = time_cpu_m1(k, n, n_iter);
    let metal_per = metal_d / n_iter as u32;
    let cpu_per = cpu_d / n_iter as u32;
    let bytes_w = k as f64 * n as f64 * 4.0;
    let metal_gbps = bytes_w / metal_per.as_secs_f64() / 1e9;
    let cpu_gbps = bytes_w / cpu_per.as_secs_f64() / 1e9;
    let speedup = cpu_per.as_secs_f64() / metal_per.as_secs_f64();
    println!(
        "{label:<32} K={k:>5} N={n:>6}  Metal sgemv {:>7.3}ms ({:>5.0} GB/s)  CPU AMX {:>7.3}ms ({:>5.0} GB/s)  speedup {:>4.2}×",
        metal_per.as_secs_f64() * 1e3,
        metal_gbps,
        cpu_per.as_secs_f64() * 1e3,
        cpu_gbps,
        speedup,
    );
}

fn main() {
    let backend = metal_backend();
    println!(
        "=== rustorch-metal vs rustorch-fusion (CBLAS AMX) — Qwen3-14B FFN/attn shapes ===\n\
         Adapter: {} | Metal3: {}\n",
        backend.adapter_name(),
        backend.supports_metal3(),
    );

    let n_iter = 100;
    let d = 5120;
    let f = 17408;
    let kv_dim = 1024;
    let vocab = 151_936;

    println!("--- M=1 native sgemv (the real decode shape) ---");
    println!(
        "{:<32} {:>16}  {:>40}  {:>34}  {:>10}",
        "shape (label)",
        "K, N",
        "Metal sgemv_f32_simd (NATIVE M=1)",
        "CPU AMX cblas_sgemm M=1",
        "speedup"
    );
    println!("{:-<160}", "");
    run_sgemv("qkv_proj (fused)", d, d + 2 * kv_dim, n_iter);
    run_sgemv("gate_up_proj (fused)", d, 2 * f, n_iter);
    run_sgemv("down_proj", f, d, n_iter);
    run_sgemv("o_proj", d, d, n_iter);
    run_sgemv("lm_head", d, vocab, n_iter);

    println!();
    println!("--- For comparison: M=8 padded (matmul_simdgroup_f32_*) ---");
    println!(
        "{:<32} {:>20}  {:>34}  {:>34}  {:>10}",
        "shape (label)", "shape", "Metal", "CPU AMX (cblas_sgemm)", "speedup"
    );
    println!("{:-<160}", "");
    let n_iter_big = 30;
    run("qkv_proj (M=8 padded)", 8, d, d + 2 * kv_dim, n_iter_big);
    run("gate_up_proj (M=8 padded)", 8, d, 2 * f, n_iter_big);
    run("down_proj (M=8 padded)", 8, f, d, n_iter_big);
    run("o_proj (M=8 padded)", 8, d, d, n_iter_big);
    run("lm_head (M=8 padded)", 8, d, vocab, n_iter_big);
}
