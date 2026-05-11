//! T246.8 A5 — Parity tests for `sgemv_q6k_bf16_split_k` (split-K Q6_K
//! SGEMV for lm_head) vs the V2 single-pass reference.
//!
//! Split-K design : block grid `(N, k_chunks)` × 64 threads/block. Each
//! (row, chunk) block computes a FP32 partial sum over a contiguous range
//! of `blocks_per_chunk = (K/256) / k_chunks` super-blocks ; a reduction
//! kernel then sums dim 0 of the `[k_chunks, N]` partial buffer to produce
//! the final BF16 `[N]` output.
//!
//! FP non-associativity caveat : V2's reduction order is
//! `warp_reduce(sum_{b in K} per-thread)`, while split-K is
//! `sum_{c} warp_reduce(sum_{b in c} per-thread)`. For `blocks_per_chunk
//! = 1` (the lm_head case : K=2048 → blocks_per_row=8 → k_chunks=8), the
//! warp_reduce inputs are the SAME single-block per-thread values in both
//! kernels — only the final chunk-summation order differs. We allow up to
//! 4 BF16 ULP drift but expect the majority of outputs to be bit-exact.
//!
//! Run on GB10 (DGX Spark) :
//!   PATH=/usr/local/cuda-13.0/bin:$PATH \
//!     LD_LIBRARY_PATH=/usr/local/cuda-13.0/targets/sbsa-linux/lib:$LD_LIBRARY_PATH \
//!     cargo test --release --features cuda -p rustorch-cuda \
//!       --test cuda_sgemv_q6k_split_k_parity -- --nocapture --test-threads=1

#![cfg(feature = "cuda")]

use cudarc::driver::{CudaContext, DevicePtr, DevicePtrMut};
use rustorch_cuda::llm_kernels::LlmKernels;

const Q6K_BLOCK_SIZE_BYTES: usize = 210;
const Q6K_GROUP_SIZE: usize = 256;

/// Build a synthetic Q6_K weight tensor of shape `[n, k]` with bit
/// patterns that resemble real lm_head weights. Each super-block has :
/// - 128 bytes ql (4-bit low nibbles)
/// - 32 bytes qh (2-bit high bits)
/// - 16 bytes scales (signed)
/// - 2 bytes d (half scale)
fn synth_q6k(n: usize, k: usize, seed: u64) -> Vec<u8> {
    assert!(k % Q6K_GROUP_SIZE == 0);
    let blocks_per_row = k / Q6K_GROUP_SIZE;
    let mut out = vec![0u8; n * blocks_per_row * Q6K_BLOCK_SIZE_BYTES];
    let mut s = seed;
    for row in 0..n {
        for b in 0..blocks_per_row {
            let off = (row * blocks_per_row + b) * Q6K_BLOCK_SIZE_BYTES;
            // ql : 128 bytes
            for i in 0..128 {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                out[off + i] = (s >> 33) as u8;
            }
            // qh : 32 bytes
            for i in 0..32 {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                out[off + 128 + i] = (s >> 33) as u8;
            }
            // scales : 16 signed bytes, biased small to keep magnitudes sane
            for i in 0..16 {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let v = ((s >> 33) as i8) % 8; // -8..7 signed
                out[off + 192 + i] = v as u8;
            }
            // d : half-precision scale, encode a small positive value
            let d_f32: f32 = 0.001 + ((row + b) as f32 * 1e-5);
            let d_half = half::f16::from_f32(d_f32);
            let d_bits: u16 = d_half.to_bits();
            out[off + 208] = (d_bits & 0xFF) as u8;
            out[off + 209] = (d_bits >> 8) as u8;
        }
    }
    out
}

fn synth_x_bf16(k: usize, seed: f32) -> Vec<half::bf16> {
    (0..k)
        .map(|i| half::bf16::from_f32(((i as f32 + 1.0) * 0.0023 + seed).sin() * 0.3))
        .collect()
}

fn assert_close(label: &str, ref_v2: &[half::bf16], split_k: &[half::bf16], max_bad_ulp: u32) {
    assert_eq!(ref_v2.len(), split_k.len(), "{label}: length mismatch");
    let mut bit_exact = 0usize;
    let mut within_1_ulp = 0usize;
    let mut within_4_ulp = 0usize;
    let mut bad = 0usize;
    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;
    let mut first_bad: Option<(usize, half::bf16, half::bf16, u32)> = None;
    for (i, (a, b)) in ref_v2.iter().zip(split_k.iter()).enumerate() {
        if a.to_bits() == b.to_bits() {
            bit_exact += 1;
            within_1_ulp += 1;
            within_4_ulp += 1;
            continue;
        }
        let av = a.to_f32();
        let bv = b.to_f32();
        let abs = (av - bv).abs();
        let rel = if av.abs() > 1e-3 { abs / av.abs() } else { 0.0 };
        max_abs = max_abs.max(abs);
        max_rel = max_rel.max(rel);
        let ulp = (a.to_bits() as i32 - b.to_bits() as i32).unsigned_abs();
        if ulp <= 1 {
            within_1_ulp += 1;
            within_4_ulp += 1;
        } else if ulp <= 4 {
            within_4_ulp += 1;
        } else if rel > 0.01 {
            bad += 1;
            if first_bad.is_none() {
                first_bad = Some((i, *a, *b, ulp));
            }
        }
    }
    let n = ref_v2.len();
    let bad_pct = (bad as f32) / (n as f32) * 100.0;
    eprintln!(
        "{label}: N={} bit_exact={} ({}%) within_1_ulp={} ({}%) within_4_ulp={} ({}%) bad>{}ULP&>1%rel={} ({:.3}%) max_abs={} max_rel={}",
        n,
        bit_exact,
        (bit_exact * 100) / n,
        within_1_ulp,
        (within_1_ulp * 100) / n,
        within_4_ulp,
        (within_4_ulp * 100) / n,
        max_bad_ulp,
        bad,
        bad_pct,
        max_abs,
        max_rel
    );
    if bad_pct > 1.0 {
        let (idx, a, b, ulp) = first_bad.unwrap();
        panic!(
            "{label}: too many drift > {max_bad_ulp} ULP & > 1% rel ({bad_pct:.3}% > 1%). \
             First bad at {idx}: v2={} (0x{:04x}) split_k={} (0x{:04x}) ulp={}",
            a.to_f32(),
            a.to_bits(),
            b.to_f32(),
            b.to_bits(),
            ulp
        );
    }
}

fn run_parity_case(label: &str, n: usize, k: usize, k_chunks: i32) {
    let w = synth_q6k(n, k, 0xA5_5A_C0_DE_DEAD_BEEF);
    let x = synth_x_bf16(k, 0.07);

    let ctx = CudaContext::new(0).expect("ctx");
    let stream = ctx.default_stream();
    let kernels = LlmKernels::new(ctx);

    let w_dev = stream.memcpy_stod(&w).expect("w");
    let x_dev = stream.memcpy_stod(&x).expect("x");
    let mut y_v2 = stream
        .memcpy_stod(&vec![half::bf16::from_f32(0.0); n])
        .expect("y_v2");
    let mut y_split = stream
        .memcpy_stod(&vec![half::bf16::from_f32(0.0); n])
        .expect("y_split");
    let mut partial = stream
        .alloc_zeros::<f32>((k_chunks as usize) * n)
        .expect("partial");

    unsafe {
        let (wp, _g0) = w_dev.device_ptr(&stream);
        let (xp, _g1) = x_dev.device_ptr(&stream);
        let (yp2, _g2) = y_v2.device_ptr_mut(&stream);
        let (yps, _g3) = y_split.device_ptr_mut(&stream);
        let (pp, _g4) = partial.device_ptr_mut(&stream);

        kernels
            .sgemv_q6k_bf16_v2(&stream, wp, xp, yp2, n as i32, k as i32)
            .expect("v2 launch");
        kernels
            .sgemv_q6k_bf16_split_k(&stream, wp, xp, pp, yps, n as i32, k as i32, k_chunks)
            .expect("split_k launch");
    }

    let out_v2 = stream.memcpy_dtov(&y_v2).expect("dtov v2");
    let out_split = stream.memcpy_dtov(&y_split).expect("dtov split");

    // Allow up to 4 BF16 ULP drift on a few outputs (FP non-associativity of
    // the reordered chunk-summation in the reduction kernel).
    assert_close(label, &out_v2, &out_split, 4);
}

#[test]
fn sgemv_q6k_split_k_matches_v2_lm_head_subsample() {
    // Qwen3.6-35B-A3B lm_head shape : N=152064, K=2048. Subsample N=4096
    // to keep test fast while exercising K=2048 / k_chunks=8.
    run_parity_case("split_k_lm_head_4096x2048_kc8", 4096, 2048, 8);
}

#[test]
fn sgemv_q6k_split_k_matches_v2_lm_head_full() {
    // Full lm_head shape on Qwen3.6-35B-A3B : N=152064, K=2048.
    // Q6_K weights = ~30 MB, partial buffer = ~4.8 MB FP32. Test fits easily.
    run_parity_case("split_k_lm_head_full_152064x2048_kc8", 152064, 2048, 8);
}

#[test]
fn sgemv_q6k_split_k_matches_v2_small_qkv() {
    // Smaller shape for sanity : QKV-like N=10240, K=2048.
    run_parity_case("split_k_qkv_10240x2048_kc8", 10240, 2048, 8);
}

#[test]
fn sgemv_q6k_split_k_matches_v2_kc4() {
    // Same N=2048, K=2048 but k_chunks=4 (blocks_per_chunk=2 → tests
    // multi-block-per-chunk accumulation order).
    run_parity_case("split_k_2048x2048_kc4", 2048, 2048, 4);
}

#[test]
fn sgemv_q6k_split_k_matches_v2_kc2() {
    // k_chunks=2 → blocks_per_chunk=4 (less SM split, larger per-chunk
    // accumulation). Tests that multi-block chunks still match V2 within
    // tolerance.
    run_parity_case("split_k_2048x2048_kc2", 2048, 2048, 2);
}

#[test]
fn sgemv_q6k_split_k_matches_v2_kc1_trivial() {
    // k_chunks=1 → single chunk = whole K. This should be bit-exact with V2
    // since per-thread accumulation order is identical (no inter-chunk
    // reduction).
    run_parity_case("split_k_2048x2048_kc1_trivial", 2048, 2048, 1);
}

#[test]
fn sgemv_q6k_split_k_matches_v2_large_k() {
    // Larger K (K=4096 = 16 super-blocks) — exercises 8 chunks of 2 blocks.
    run_parity_case("split_k_2048x4096_kc8", 2048, 4096, 8);
}
