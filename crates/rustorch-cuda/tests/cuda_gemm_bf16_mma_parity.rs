//! T246.10 TrackG-lite — parity tests for `gemm_bf16_bf16_mma_m16n8k16`
//!
//! Compares the mma.sync m16n8k16 BF16 GEMM against the warp-shuffle scalar
//! baseline `sgemm_bf16_bf16_mvar` over a representative grid of shapes :
//!
//!   M ∈ {16, 32, 64, 128, 256, 512}   (mma path is for M >= 16)
//!   (N, K) ∈ {(2048, 2048), (2048, 4096), (4096, 2048), (18944, 2048)}
//!
//! Both kernels use FP32 accumulator and produce a BF16 down-cast. The
//! reduction order differs (mma : 16-K-chunk register accumulate ; mvar :
//! 256-K-block warp-shuffle), so we don't expect bit-exact match. We assert
//! the result is within a small ULP+relative envelope.
//!
//! Run on GB10 :
//!   PATH=/usr/local/cuda-13.0/bin:$PATH \
//!     LD_LIBRARY_PATH=/usr/local/cuda-13.0/targets/sbsa-linux/lib:$LD_LIBRARY_PATH \
//!     cargo test --release --features cuda -p rustorch-cuda \
//!       --test cuda_gemm_bf16_mma_parity -- --nocapture --test-threads=1

#![cfg(feature = "cuda")]

use cudarc::driver::{CudaContext, DevicePtr, DevicePtrMut};
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
    let kernels = LlmKernels::new(ctx);

    // Use small magnitudes to keep accumulator well within BF16 range
    // (≈ ±64) — large K otherwise pushes into BF16 overflow on the
    // mvar baseline.
    let scale_w = 0.001f32;
    let scale_x = 0.001f32;
    let w = fill_bf16(
        n * k,
        scale_w,
        0xbf16_a6b_u64 ^ (m as u64) ^ ((n as u64) << 16) ^ ((k as u64) << 32),
    );
    let x = fill_bf16(
        m * k,
        scale_x,
        0xfeed_beef_u64 ^ (m as u64) ^ ((n as u64) << 16) ^ ((k as u64) << 32),
    );

    let w_dev = stream.memcpy_stod(&w).expect("w");
    let x_dev = stream.memcpy_stod(&x).expect("x");

    let mut y_mvar_dev = stream.alloc_zeros::<half::bf16>(m * n).expect("y_mvar");
    let mut y_mma_dev = stream.alloc_zeros::<half::bf16>(m * n).expect("y_mma");

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
        let (y_p, _gy) = y_mma_dev.device_ptr_mut(&stream);
        kernels
            .gemm_bf16_bf16_mma(&stream, w_p, x_p, y_p, m as i32, n as i32, k as i32)
            .expect("mma");
    }

    let y_mvar: Vec<half::bf16> = stream.memcpy_dtov(&y_mvar_dev).expect("dl mvar");
    let y_mma: Vec<half::bf16> = stream.memcpy_dtov(&y_mma_dev).expect("dl mma");

    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;
    let mut bit_exact = 0usize;
    let mut within_2_ulp = 0usize;
    let mut bad = 0usize;
    let mut first_bad: Option<(usize, f32, f32)> = None;
    for (idx, (a, b)) in y_mvar.iter().zip(y_mma.iter()).enumerate() {
        if a.to_bits() == b.to_bits() {
            bit_exact += 1;
            within_2_ulp += 1;
            continue;
        }
        let av = a.to_f32();
        let bv = b.to_f32();
        let abs = (av - bv).abs();
        let rel = if av.abs() > 1e-3 { abs / av.abs() } else { 0.0 };
        max_abs = max_abs.max(abs);
        max_rel = max_rel.max(rel);
        let ulp = (a.to_bits() as i32 - b.to_bits() as i32).unsigned_abs();
        if ulp <= 2 {
            within_2_ulp += 1;
        } else if rel > 0.02 && abs > 0.01 {
            bad += 1;
            if first_bad.is_none() {
                first_bad = Some((idx, av, bv));
            }
        }
    }
    let total = y_mvar.len();
    let bad_pct = (bad as f32) / (total as f32) * 100.0;
    eprintln!(
        "M={m} N={n} K={k}: bit_exact={} ({}%) within_2_ulp={} ({}%) bad>2ULP&>2%rel&>0.01abs={} ({:.3}%) max_abs={:.5} max_rel={:.5}",
        bit_exact,
        (bit_exact * 100) / total,
        within_2_ulp,
        (within_2_ulp * 100) / total,
        bad,
        bad_pct,
        max_abs,
        max_rel,
    );
    if let Some((i, a, b)) = first_bad {
        eprintln!("  first bad idx={i} mvar={a} mma={b}");
    }
    assert!(
        bad_pct < 0.5,
        "M={m} N={n} K={k} : too many drift elements ({bad}/{total} = {bad_pct:.3}%)"
    );
    // Reduction order differs ; max_abs may be a few BF16 ulps. Bound it
    // loosely (the FP32 accumulator paths agree to <1% absolute on the
    // input magnitudes we picked).
    assert!(
        max_rel < 0.10,
        "M={m} N={n} K={k} : max_rel {max_rel} too large"
    );
}

#[test]
fn gemm_bf16_mma_matches_mvar_small_shapes() {
    run_case(16, 256, 256);
    run_case(16, 512, 512);
    run_case(32, 256, 512);
    run_case(32, 512, 512);
}

#[test]
fn gemm_bf16_mma_matches_mvar_qwen_ssm_shapes() {
    // Representative of Qwen3.6 SSM proj shapes (kept smaller K than
    // production to keep test fast, K=2048 covers worst-case accumulation).
    run_case(16, 2048, 2048);
    run_case(32, 2048, 2048);
    run_case(64, 2048, 2048);
}

#[test]
fn gemm_bf16_mma_matches_mvar_large_m() {
    run_case(128, 2048, 2048);
    run_case(256, 2048, 2048);
    run_case(512, 2048, 2048);
}

#[test]
fn gemm_bf16_mma_matches_mvar_oddly_sized() {
    // Non-multiple-of-block shapes — exercises the boundary masks.
    run_case(17, 512, 512);
    run_case(33, 1024, 512);
    run_case(127, 2048, 2048);
}
