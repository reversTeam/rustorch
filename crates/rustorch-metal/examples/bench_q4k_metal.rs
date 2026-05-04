//! Benchmark + correctness for Metal Q4_K direct sgemv kernel
//! (`sgemv_q4_k_f32`) against:
//!  - the scalar `rustorch-gguf::sgemv_q4_k` CPU reference (correctness)
//!  - Apple Accelerate `cblas_sgemm` f32 cached AMX (performance baseline)
//!
//! This is the kernel we need to wire into `rustorch-llm` to break past
//! the 41 tok/s candle ceiling on Qwen3-14B M4 Max.

#![cfg(target_os = "macos")]
#![allow(missing_docs)]

use half::f16;
use rustorch_fusion::patterns::matmul_bias_act::{fused_matmul_bias_activation, Activation};
use rustorch_gguf::dequant::{Q4_K_BYTES, QK_K};
use rustorch_gguf::sgemv_q4_k as sgemv_q4_k_cpu;
use rustorch_metal::backend_singleton::metal_backend;
use rustorch_metal::kernels::sgemv_q4_k_f32;
use std::time::Instant;

fn build_random_q4k_matrix(n: usize, k: usize, seed: u32) -> Vec<u8> {
    let blocks_per_row = k / QK_K;
    let total = n * blocks_per_row * Q4_K_BYTES;
    let mut buf = vec![0u8; total];
    let mut s = seed;
    let mut rand_byte = || {
        s = s.wrapping_mul(1103515245).wrapping_add(12345);
        (s >> 8) as u8
    };
    let dh = f16::from_f32(0.05).to_le_bytes();
    let dminh = f16::from_f32(0.01).to_le_bytes();
    for blk in 0..(n * blocks_per_row) {
        let off = blk * Q4_K_BYTES;
        buf[off] = dh[0];
        buf[off + 1] = dh[1];
        buf[off + 2] = dminh[0];
        buf[off + 3] = dminh[1];
        for i in 4..16 {
            buf[off + i] = rand_byte() & 0x3F;
        }
        for i in 0..128 {
            buf[off + 16 + i] = rand_byte();
        }
    }
    buf
}

fn time_metal_q4k(k: usize, n: usize, n_iter: usize) -> std::time::Duration {
    let backend = metal_backend();
    let w_bytes = build_random_q4k_matrix(n, k, 1);
    let x: Vec<f32> = (0..k).map(|i| ((i as f32 + 1.0) * 0.001).sin()).collect();

    let x_buf = backend.alloc_shared(k * 4).unwrap();
    let w_buf = backend.alloc_shared(w_bytes.len()).unwrap();
    unsafe {
        std::ptr::copy_nonoverlapping(x.as_ptr(), x_buf.contents() as *mut f32, k);
        std::ptr::copy_nonoverlapping(w_bytes.as_ptr(), w_buf.contents() as *mut u8, w_bytes.len());
    }

    // Warmup.
    let _ = sgemv_q4_k_f32(backend, &x_buf, &w_buf, k, n).unwrap();
    backend.drain();

    let t0 = Instant::now();
    for _ in 0..n_iter {
        let _ = sgemv_q4_k_f32(backend, &x_buf, &w_buf, k, n).unwrap();
    }
    backend.drain();
    t0.elapsed()
}

fn time_cpu_amx_f32(k: usize, n: usize, n_iter: usize) -> std::time::Duration {
    // Build a random f32 weight matrix as a fair AMX baseline (the
    // shape rustorch-llm currently uses after dequant-to-f32).
    let a: Vec<f32> = (0..k).map(|i| ((i as f32 + 1.0) * 0.001).sin()).collect();
    let b: Vec<f32> = (0..k * n)
        .map(|i| ((i as f32 + 1.0) * 0.013).cos() * 0.5)
        .collect();
    let mut c = vec![0.0_f32; n];
    fused_matmul_bias_activation(&a, &b, None, &mut c, 1, k, n, Activation::None).unwrap();
    let t0 = Instant::now();
    for _ in 0..n_iter {
        fused_matmul_bias_activation(&a, &b, None, &mut c, 1, k, n, Activation::None).unwrap();
    }
    t0.elapsed()
}

fn correctness_check(k: usize, n: usize) {
    let backend = metal_backend();
    let w_bytes = build_random_q4k_matrix(n, k, 42);
    let x: Vec<f32> = (0..k).map(|i| ((i as f32 + 1.0) * 0.001).sin()).collect();

    // CPU reference (validated by rustorch-gguf tests).
    let mut y_cpu = vec![0.0_f32; n];
    sgemv_q4_k_cpu(&x, &w_bytes, &mut y_cpu, k, n).unwrap();

    let x_buf = backend.alloc_shared(k * 4).unwrap();
    let w_buf = backend.alloc_shared(w_bytes.len()).unwrap();
    unsafe {
        std::ptr::copy_nonoverlapping(x.as_ptr(), x_buf.contents() as *mut f32, k);
        std::ptr::copy_nonoverlapping(w_bytes.as_ptr(), w_buf.contents() as *mut u8, w_bytes.len());
    }
    let out = sgemv_q4_k_f32(backend, &x_buf, &w_buf, k, n).unwrap();
    backend.drain();
    let mut y_metal = vec![0.0_f32; n];
    unsafe {
        std::ptr::copy_nonoverlapping(out.contents() as *const f32, y_metal.as_mut_ptr(), n);
    }

    // f16 super-block scales mean ~1e-3 absolute drift between CPU
    // and GPU paths is expected.
    let l2_ref: f32 = y_cpu.iter().map(|v| v * v).sum::<f32>().sqrt();
    let l2_err: f32 = y_cpu
        .iter()
        .zip(y_metal.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f32>()
        .sqrt();
    let rel = l2_err / l2_ref.max(1e-6);
    println!("correctness K={k} N={n}: l2_ref={l2_ref:.3} l2_err={l2_err:.3e} rel={rel:.3e}");
    assert!(
        rel < 1e-3,
        "GPU Q4_K kernel drifted from CPU reference (rel L2 = {rel})"
    );
    println!(
        "    sample y[0..3] cpu={:?}  metal={:?}",
        &y_cpu[..3],
        &y_metal[..3]
    );
}

fn main() {
    let backend = metal_backend();
    println!(
        "=== sgemv_q4_k_f32 (Metal direct Q4_K) — Qwen3-14B FFN/attn shapes ===\n\
         Adapter: {} | Metal3: {}\n",
        backend.adapter_name(),
        backend.supports_metal3(),
    );

    println!("--- Correctness vs CPU reference ---");
    correctness_check(256, 8);
    correctness_check(1024, 16);
    println!();

    println!("--- Performance: Metal Q4_K vs CPU AMX f32 (M=1 sgemv) ---");
    let n_iter = 30;
    let d = 5120;
    let f = 17408;
    let kv_dim = 1024;
    let vocab = 151_936;

    for &(label, k, n) in &[
        ("qkv_proj  (fused)", d, d + 2 * kv_dim),
        ("gate_up   (fused)", d, 2 * f),
        ("down_proj", f, d),
        ("o_proj", d, d),
        ("lm_head", d, vocab),
    ] {
        let metal_d = time_metal_q4k(k, n, n_iter);
        let cpu_d = time_cpu_amx_f32(k, n, n_iter);
        let metal_per = metal_d / n_iter as u32;
        let cpu_per = cpu_d / n_iter as u32;
        let speedup = cpu_per.as_secs_f64() / metal_per.as_secs_f64();
        let bytes_q4k = (n * (k / QK_K) * Q4_K_BYTES) as f64;
        let bytes_f32 = (n * k * 4) as f64;
        let metal_gbps = bytes_q4k / metal_per.as_secs_f64() / 1e9;
        let cpu_gbps = bytes_f32 / cpu_per.as_secs_f64() / 1e9;
        println!(
            "{label:<22} K={k:>5} N={n:>6}  Metal-Q4K {:>7.3}ms ({:>5.0} GB/s)  CPU-AMX-f32 {:>7.3}ms ({:>5.0} GB/s)  speedup {:>5.2}×",
            metal_per.as_secs_f64() * 1e3,
            metal_gbps,
            cpu_per.as_secs_f64() * 1e3,
            cpu_gbps,
            speedup,
        );
    }
}
