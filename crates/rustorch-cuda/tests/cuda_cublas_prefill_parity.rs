//! DESIGN-PIVOT — parity tests for `LtSession::matmul_bf16_rowmajor`
//!
//! Validates that the cuBLASLt-backed row-major BF16 GEMM produces the same
//! result (within BF16 ULP tolerance) as the hand-written
//! `sgemm_bf16_bf16_mvar` baseline kernel on the prefill shapes that matter
//! for Qwen3.6 (M ∈ {16, 64, 128, 512}, K = 2048, N = 4096).
//!
//! Both kernels compute `Y[M, N] = X[M, K] · W^T[K, N]` with all three
//! buffers row-major BF16. They accumulate in FP32 and down-cast to BF16,
//! but differ in reduction order (mvar = warp-shuffle 256-K block ; cuBLASLt
//! = vendor heuristic kernel), so we don't require bit-exact match.
//!
//! Run on GB10 :
//!   PATH=/usr/local/cuda-13.0/bin:$PATH \
//!     LD_LIBRARY_PATH=/usr/local/cuda-13.0/targets/sbsa-linux/lib:$LD_LIBRARY_PATH \
//!     cargo test --release --features cuda -p rustorch-cuda \
//!       --test cuda_cublas_prefill_parity -- --nocapture --test-threads=1

#![cfg(feature = "cuda")]

use cudarc::driver::{CudaContext, DevicePtr, DevicePtrMut};
use rustorch_cuda::cublas_lt::LtSession;
use rustorch_cuda::llm_kernels::LlmKernels;

fn xorshift(seed: u64) -> impl FnMut() -> u32 {
    let mut s = seed;
    move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s as u32
    }
}

fn fill_bf16(n: usize, scale: f32, seed: u64) -> Vec<half::bf16> {
    let mut next = xorshift(seed);
    (0..n)
        .map(|_| {
            let r = (next() % 4096) as f32 - 2048.0;
            half::bf16::from_f32(r * scale)
        })
        .collect()
}

fn run_case(m: usize, n: usize, k: usize) {
    assert!(
        k % 256 == 0,
        "K must be multiple of 256 for the mvar baseline"
    );

    let ctx = CudaContext::new(0).expect("ctx");
    let stream = ctx.default_stream();
    let kernels = LlmKernels::new(ctx.clone());
    let mut session = LtSession::new(stream.clone()).expect("LtSession");

    let scale_w = 0.001f32;
    let scale_x = 0.001f32;
    let w = fill_bf16(
        n * k,
        scale_w,
        0xc8a5_a6b_u64 ^ (m as u64) ^ ((n as u64) << 16) ^ ((k as u64) << 32),
    );
    let x = fill_bf16(
        m * k,
        scale_x,
        0xfeed_d00d_u64 ^ (m as u64) ^ ((n as u64) << 16) ^ ((k as u64) << 32),
    );

    let w_dev = stream.memcpy_stod(&w).expect("w");
    let x_dev = stream.memcpy_stod(&x).expect("x");

    let mut y_mvar_dev = stream.alloc_zeros::<half::bf16>(m * n).expect("y_mvar");
    let mut y_cublas_dev = stream.alloc_zeros::<half::bf16>(m * n).expect("y_cublas");

    unsafe {
        let (w_p, _gw) = w_dev.device_ptr(&stream);
        let (x_p, _gx) = x_dev.device_ptr(&stream);
        let (y_p, _gy) = y_mvar_dev.device_ptr_mut(&stream);
        kernels
            .sgemm_bf16_bf16_mvar(&stream, w_p, x_p, y_p, m as i32, n as i32, k as i32)
            .expect("mvar");
    }
    unsafe {
        let (w_p, _gw) = w_dev.device_ptr(&stream);
        let (x_p, _gx) = x_dev.device_ptr(&stream);
        let (y_p, _gy) = y_cublas_dev.device_ptr_mut(&stream);
        session
            .matmul_bf16_rowmajor(w_p, x_p, y_p, m, n, k, 1.0, 0.0)
            .expect("cublas");
    }

    let y_mvar: Vec<half::bf16> = stream.memcpy_dtov(&y_mvar_dev).expect("dl mvar");
    let y_cublas: Vec<half::bf16> = stream.memcpy_dtov(&y_cublas_dev).expect("dl cublas");

    let mut max_abs = 0.0f32;
    let mut bit_exact = 0usize;
    let mut within_4_ulp = 0usize;
    let mut bad_abs = 0usize;
    let mut first_bad: Option<(usize, f32, f32)> = None;
    // BF16 result magnitudes ≤ ~ K * (2e-3)^2 = ~16e-3 at K=4096 ; absolute
    // tol of 2.0 is generous (catches order-of-magnitude permutation / layout
    // bugs without flagging FP rounding drift).
    let abs_tol_global = 2.0f32;
    for (idx, (a, b)) in y_mvar.iter().zip(y_cublas.iter()).enumerate() {
        if a.to_bits() == b.to_bits() {
            bit_exact += 1;
            within_4_ulp += 1;
            continue;
        }
        let av = a.to_f32();
        let bv = b.to_f32();
        let abs = (av - bv).abs();
        max_abs = max_abs.max(abs);
        let ulp = (a.to_bits() as i32 - b.to_bits() as i32).unsigned_abs();
        if ulp <= 4 {
            within_4_ulp += 1;
        }
        if abs > abs_tol_global {
            bad_abs += 1;
            if first_bad.is_none() {
                first_bad = Some((idx, av, bv));
            }
        }
    }
    let total = y_mvar.len();
    let bit_exact_pct = (bit_exact * 100) / total;
    let within_4_pct = (within_4_ulp * 100) / total;
    eprintln!(
        "M={m} N={n} K={k}: bit_exact={} ({}%) within_4_ulp={} ({}%) bad_abs>tol={} max_abs={:.5}",
        bit_exact, bit_exact_pct, within_4_ulp, within_4_pct, bad_abs, max_abs,
    );
    if let Some((i, a, b)) = first_bad {
        eprintln!("  first bad idx={i} mvar={a} cublas={b}");
    }
    // Different kernel implementations + different K-reduction order →
    // bit-exact rate is lower than within-kernel-family comparison. We
    // require ≥ 95 % bit-exact and ≥ 99 % within 4 ULP. Zero outliers >
    // absolute tolerance.
    assert!(
        bit_exact_pct >= 95,
        "M={m} N={n} K={k} : bit_exact rate {bit_exact_pct}% < 95%"
    );
    assert!(
        within_4_ulp * 100 >= total * 99,
        "M={m} N={n} K={k} : within_4_ulp rate {within_4_ulp}/{total} < 99%"
    );
    assert!(
        bad_abs == 0,
        "M={m} N={n} K={k} : {bad_abs} elements drift > {abs_tol_global} abs (max_abs={max_abs})"
    );
}

#[test]
fn cublas_bf16_rowmajor_matches_mvar_prefill_shapes() {
    // K=2048 N=4096 is the dense attention W_q / W_k / W_v / W_o block on
    // Qwen3.6 with hidden_size = 2048.
    run_case(16, 4096, 2048);
    run_case(64, 4096, 2048);
    run_case(128, 4096, 2048);
    run_case(512, 4096, 2048);
}

#[test]
fn cublas_bf16_rowmajor_matches_mvar_ssm_shapes() {
    // SSM in_proj : K=2048, N≈3072. SSM out_proj : K=18944, N=2048.
    run_case(16, 2048, 2048);
    run_case(128, 2048, 2048);
    run_case(128, 4096, 2048);
}

#[test]
fn cublas_bf16_rowmajor_matches_mvar_small_shapes() {
    // Cross-validation against the gemm_bf16_mma test grid.
    run_case(16, 512, 512);
    run_case(32, 512, 512);
}
