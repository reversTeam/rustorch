//! Bench focal — compare SGEMM Q4_K kernels sur les shapes 14B prefill.
//!
//! Mesure la latence de :
//!  - `sgemm_q4_k_f32_simdgroup_matrix_64`  (BM×BN×BK = 64×64×32, half MMA, 4 SG)
//!  - `sgemm_q4_k_f32_lcpp_ported`          (BM×BN×BK = 32×64×32, swizzled SHM, half MMA, 4 SG)
//!
//! Shapes (M, N, K) :
//!   - Q proj    : (256, 5120,  5120)
//!   - K/V proj  : (256, 1024,  5120)   — GQA factor 5
//!   - Out proj  : (256, 5120,  5120)
//!   - Gate/Up   : (256, 17408, 5120)
//!   - Down (Q6) : (256, 5120, 17408)   — non testé ici (Q6_K)
//!
//! Cible : closer le gap 14B Q4 prefill 233 t/s → 454+ t/s (llama.cpp).

#![cfg(target_os = "macos")]
#![allow(missing_docs)]

use half::f16;
use rustorch_gguf::dequant::{Q4_K_BYTES, QK_K};
use rustorch_metal::backend_singleton::metal_backend;
use rustorch_metal::kernels::{
    sgemm_q4_k_f32_lcpp_ported_into, sgemm_q4_k_f32_simdgroup_matrix_64_into,
};
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

fn time_kernel<F>(
    label: &str,
    m: usize,
    n: usize,
    k: usize,
    n_iter: usize,
    mut run: F,
) -> (f64, f64)
// (ms_per_iter, gflops)
where
    F: FnMut(usize, usize, usize),
{
    // Warmup
    run(m, n, k);
    let backend = metal_backend();
    backend.drain();

    let t0 = Instant::now();
    for _ in 0..n_iter {
        run(m, n, k);
    }
    backend.drain();
    let dt = t0.elapsed();
    let ms_per_iter = dt.as_secs_f64() * 1000.0 / n_iter as f64;
    let flops = 2.0 * m as f64 * n as f64 * k as f64;
    let gflops = flops / (ms_per_iter * 1e6);
    println!(
        "  {:<24} M={:<4} N={:<6} K={:<6}  {:>7.3} ms/iter  {:>7.1} GFLOPS",
        label, m, n, k, ms_per_iter, gflops
    );
    (ms_per_iter, gflops)
}

fn run_shape(m: usize, n: usize, k: usize, n_iter: usize) {
    let backend = metal_backend();
    let w_bytes = build_random_q4k_matrix(n, k, 1);
    let x: Vec<f32> = (0..(m * k))
        .map(|i| ((i as f32 + 1.0) * 0.001).sin())
        .collect();

    let x_buf = backend.alloc_shared(m * k * 4).unwrap();
    let w_buf = backend.alloc_shared(w_bytes.len()).unwrap();
    let c_buf = backend.alloc_shared(m * n * 4).unwrap();
    unsafe {
        std::ptr::copy_nonoverlapping(x.as_ptr(), x_buf.contents() as *mut f32, m * k);
        std::ptr::copy_nonoverlapping(w_bytes.as_ptr(), w_buf.contents() as *mut u8, w_bytes.len());
    }

    println!("\n--- Shape M={m} N={n} K={k} ---");

    let aligned_64 = m >= 64 && m % 64 == 0 && n % 64 == 0 && k % 256 == 0;
    let aligned_lcpp = m >= 32 && m % 32 == 0 && n % 64 == 0 && k % 256 == 0;

    let mut t_64 = 0.0;
    let mut t_lcpp = 0.0;

    if aligned_64 {
        let backend2 = metal_backend();
        let xb = x_buf.clone();
        let wb = w_buf.clone();
        let cb = c_buf.clone();
        let (ms, _g) = time_kernel("simdgroup_matrix_64", m, n, k, n_iter, move |m, n, k| {
            sgemm_q4_k_f32_simdgroup_matrix_64_into(backend2, &xb, &wb, &cb, m, n, k).unwrap();
        });
        t_64 = ms;
    } else {
        println!("  simdgroup_matrix_64    M={m} N={n} K={k}  (skip - not aligned 64)");
    }

    if aligned_lcpp {
        let backend2 = metal_backend();
        let xb = x_buf.clone();
        let wb = w_buf.clone();
        let cb = c_buf.clone();
        let (ms, _g) = time_kernel("lcpp_ported (32×64)", m, n, k, n_iter, move |m, n, k| {
            sgemm_q4_k_f32_lcpp_ported_into(backend2, &xb, &wb, &cb, m, n, k).unwrap();
        });
        t_lcpp = ms;
    } else {
        println!("  lcpp_ported            M={m} N={n} K={k}  (skip - not aligned 32×64)");
    }

    if t_64 > 0.0 && t_lcpp > 0.0 {
        let speedup = t_64 / t_lcpp;
        println!(
            "  → lcpp_ported {:.2}× {} simdgroup_matrix_64",
            speedup,
            if speedup > 1.0 { "vs" } else { "(slower than)" }
        );
    }
}

fn main() {
    println!("=== SGEMM Q4_K kernel comparison — Qwen3-14B prefill shapes ===");
    println!("(N_iter=20 par mesure ; warmup 1 iter avant timing)");

    let n_iter = 20;

    // Qwen3-14B : hidden=5120, q_heads=40, kv_heads=8 → q_proj N=5120 K=5120 ;
    // k/v_proj N=1024 K=5120 ; out_proj N=5120 K=5120.
    // FFN intermediate=17408, gate/up N=17408 K=5120 ; down (Q6_K) N=5120 K=17408.
    //
    // M = batch (max 256 in qwen_inference_metal::B_MAX).
    for m in &[64, 128, 256] {
        run_shape(*m, 5120, 5120, n_iter); // Q/Out proj
        run_shape(*m, 1024, 5120, n_iter); // K/V proj (note: 1024 not %64 — non aligned, will skip)
        run_shape(*m, 17408, 5120, n_iter); // FFN gate/up (non %256 K — but 5120%256==0 ✓)
    }
}
