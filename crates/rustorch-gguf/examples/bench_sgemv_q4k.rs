//! Benchmark `sgemv_q4_k` (direct on Q4_K bytes) against the
//! "dequant-then-sgemm" path. The goal is to confirm whether
//! reading 4× less DRAM (50 MB Q4_K vs 712 MB f32) actually
//! translates into a wall-clock speedup once the inline
//! dequantisation overhead is paid per block.
//!
//! Run with:
//! ```sh
//! cargo run -p rustorch-gguf --release --example bench_sgemv_q4k
//! ```

#![allow(missing_docs)]

use half::f16;
use rustorch_gguf::dequant::{Q4_K_BYTES, QK_K};
use rustorch_gguf::sgemv_q4_k;
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

fn time_q4k(k: usize, n: usize, n_iter: usize) -> std::time::Duration {
    let w = build_random_q4k_matrix(n, k, 1);
    let x: Vec<f32> = (0..k).map(|i| ((i as f32 + 1.0) * 0.001).sin()).collect();
    let mut y = vec![0.0_f32; n];
    // Warmup.
    sgemv_q4_k(&x, &w, &mut y, k, n).unwrap();
    let t0 = Instant::now();
    for _ in 0..n_iter {
        sgemv_q4_k(&x, &w, &mut y, k, n).unwrap();
    }
    t0.elapsed()
}

fn main() {
    println!("=== sgemv_q4_k (direct Q4_K, scalar PoC) — Qwen3-14B FFN/attn shapes ===\n");
    let n_iter = 5; // PoC scalar — keep iter count low

    let d = 5120;
    let f = 17408;
    let kv_dim = 1024;
    let vocab = 151_936;

    for &(label, k, n) in &[
        ("qkv_proj (fused)", d, d + 2 * kv_dim),
        ("gate_up_proj (fused)", d, 2 * f),
        ("down_proj", f, d),
        ("o_proj", d, d),
        ("lm_head", d, vocab),
    ] {
        let dur = time_q4k(k, n, n_iter);
        let per = dur / n_iter as u32;
        let bytes_w = (n * (k / QK_K) * Q4_K_BYTES) as f64;
        let gbps = bytes_w / per.as_secs_f64() / 1e9;
        let mb = bytes_w / 1e6;
        println!(
            "{label:<32} K={k:>5} N={n:>6}  Q4_K bytes={mb:>6.1}MB  per={:>7.2}ms  effective={:>5.1} GB/s",
            per.as_secs_f64() * 1e3,
            gbps,
        );
    }
    println!();
    println!("(Reference: CPU AMX cblas_sgemm f32 cached ≈ 0.09 ms/call for gate_up_proj)");
    println!("(Goal: bring per-call time below ~0.10 ms via NEON intrinsics in next iter)");
}
