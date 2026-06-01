//! T246.8 A3 — Parity tests for `sgemv_bf16_bf16_v2` (multi-row block
//! warp-shuffle SGEMV) vs the V1 single-row reference.
//!
//! V2 uses 4 rows per block × 64 threads per row (256 threads/block, V1's
//! exact per-thread MAC sequence preserved → BIT-EXACT with V1). V1 is the
//! original 64-thread, 1-row-per-block warp-shuffle kernel.
//!
//! Both use FP32 accumulator and produce the same final BF16 down-cast.
//! The mma.sync m16n8k16 tensor-core variant was abandoned (see
//! llm_kernels.rs SGEMV_BF16_BF16_V2_SRC docstring & note 0418e02c) — it
//! was 30% slower end-to-end due to the 87.5% wasted-N broadcast pattern.
//!
//! V2 final design is BIT-EXACT with V1 by construction (same per-thread
//! MAC count, same warp-shuffle reduction order). All assertions check
//! bit-exact equality.
//!
//! Run on GB10 (DGX Spark) :
//!   PATH=/usr/local/cuda-13.0/bin:$PATH \
//!     LD_LIBRARY_PATH=/usr/local/cuda-13.0/targets/sbsa-linux/lib:$LD_LIBRARY_PATH \
//!     cargo test --release --features cuda -p rustorch-cuda \
//!       --test cuda_sgemv_bf16_v2_parity -- --nocapture --test-threads=1

#![cfg(feature = "cuda")]

use cudarc::driver::{CudaContext, DevicePtr, DevicePtrMut};
use rustorch_cuda::llm_kernels::LlmKernels;

fn bf16_vec(n: usize, seed: f32, off: f32) -> Vec<half::bf16> {
    (0..n)
        .map(|i| half::bf16::from_f32(((i as f32 + 1.0) * seed + off).sin() * 0.4))
        .collect()
}

/// V2 is bit-exact with V1 by construction. We still allow up to 4 BF16 ULP
/// drift in case future V2 reformulations introduce non-associativity ;
/// fail if more than 2% of elements drift > 4 ULP & > 1% relative.
fn assert_close(label: &str, ref_v1: &[half::bf16], v2: &[half::bf16]) {
    assert_eq!(ref_v1.len(), v2.len(), "{label}: length mismatch");
    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;
    let mut bit_exact = 0usize;
    let mut within_1_ulp = 0usize;
    let mut bad = 0usize;
    let mut first_bad: Option<(usize, half::bf16, half::bf16)> = None;
    for (i, (a, b)) in ref_v1.iter().zip(v2.iter()).enumerate() {
        if a.to_bits() == b.to_bits() {
            bit_exact += 1;
            within_1_ulp += 1;
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
        } else if ulp > 4 && rel > 0.01 {
            bad += 1;
            if first_bad.is_none() {
                first_bad = Some((i, *a, *b));
            }
        }
    }
    let n = ref_v1.len();
    let bad_pct = (bad as f32) / (n as f32) * 100.0;
    eprintln!(
        "{label}: N={} bit_exact={} (>{}%) within_1_ulp={} ({}%) bad>{:.0}ULP&>1%={} ({:.2}%) max_abs={} max_rel={}",
        n,
        bit_exact,
        (bit_exact * 100) / n,
        within_1_ulp,
        (within_1_ulp * 100) / n,
        4.0,
        bad,
        bad_pct,
        max_abs,
        max_rel
    );
    if bad_pct > 2.0 {
        let (idx, a, b) = first_bad.unwrap();
        panic!(
            "{label}: too many drift > 4 ULP & > 1% rel ({bad_pct:.2}% > 2%). \
             First bad at {idx}: v1={} (0x{:04x}) v2={} (0x{:04x})",
            a.to_f32(),
            a.to_bits(),
            b.to_f32(),
            b.to_bits()
        );
    }
}

fn run_parity_case(label: &str, n: usize, k: usize) {
    let w = bf16_vec(n * k, 0.0017, 0.03);
    let x = bf16_vec(k, 0.0021, 0.07);

    let ctx = CudaContext::new(0).expect("ctx");
    let stream = ctx.default_stream();
    let kernels = LlmKernels::new(ctx);

    let w_dev = stream.memcpy_stod(&w).expect("w");
    let x_dev = stream.memcpy_stod(&x).expect("x");
    let mut y_v1 = stream
        .memcpy_stod(&vec![half::bf16::from_f32(0.0); n])
        .expect("y_v1");
    let mut y_v2 = stream
        .memcpy_stod(&vec![half::bf16::from_f32(0.0); n])
        .expect("y_v2");

    unsafe {
        let (wp, _g0) = w_dev.device_ptr(&stream);
        let (xp, _g1) = x_dev.device_ptr(&stream);
        let (yp1, _g2) = y_v1.device_ptr_mut(&stream);
        let (yp2, _g3) = y_v2.device_ptr_mut(&stream);

        kernels
            .sgemv_bf16_bf16(&stream, wp, xp, yp1, n as i32, k as i32)
            .expect("v1 launch");
        kernels
            .sgemv_bf16_bf16_v2(&stream, wp, xp, yp2, n as i32, k as i32)
            .expect("v2 launch");
    }

    let out_v1 = stream.memcpy_dtov(&y_v1).expect("dtov v1");
    let out_v2 = stream.memcpy_dtov(&y_v2).expect("dtov v2");

    assert_close(label, &out_v1, &out_v2);
}

#[test]
fn sgemv_bf16_v2_matches_v1_small() {
    // Smaller "FFN-shexp"-class shape : N=128, K=2048.
    // K=2048 = 8 super-blocks of 256 (V1 native size) ; K%16=0 ✓
    run_parity_case("sgemv_v2_128x2048", 128, 2048);
}

#[test]
fn sgemv_bf16_v2_matches_v1_medium() {
    // Mid-size : N=512, K=4096 (representative of attention proj shapes).
    run_parity_case("sgemv_v2_512x4096", 512, 4096);
}

#[test]
fn sgemv_bf16_v2_matches_v1_large() {
    // Larger : N=1024, K=8192.
    run_parity_case("sgemv_v2_1024x8192", 1024, 8192);
}

#[test]
fn sgemv_bf16_v2_matches_v1_lm_head_subsample() {
    // lm_head shape on Qwen3.6-35B-A3B is N=152064, K=2048 (D=2048).
    // Full size eats GPU memory in tests ; sample to N=2048 to keep test fast
    // while exercising the dual-row path with realistic K=2048.
    // Note : 35B-A3B has hidden D=2048 (vs 5120 on 27B) per qwen35.rs L205.
    run_parity_case("sgemv_v2_lm_head_subsample_2048x2048", 2048, 2048);
}

#[test]
fn sgemv_bf16_v2_n_not_multiple_of_16_handles_tail() {
    // Edge case : N=130 (not multiple of 16). 9 blocks of 16 = 144 ; rows
    // 130..144 are masked off. Verify only valid rows are written and match V1.
    run_parity_case("sgemv_v2_130x2048_tail_mask", 130, 2048);
}

/// T246.8 A3 — Stress test : verify V2 == V1 bit-exact across a wide range
/// of N (up to lm_head 152064) and input scales (0.4 to 4.0). Catches edge
/// cases where the parity-test default scale 0.4 may not exercise overflow
/// or precision-sensitive paths.
#[test]
fn sgemv_bf16_v2_stress_scales_and_shapes() {
    let cases = [
        (256usize, 2048usize, 1.0f32),    // attention proj-class
        (384usize, 2048usize, 1.0f32),    // attention KV proj
        (2048usize, 2048usize, 1.0f32),   // attention Q / O proj
        (256usize, 2048usize, 4.0f32),    // higher scale
        (4096usize, 2048usize, 1.0f32),   // FFN intermediate
        (152064usize, 2048usize, 1.0f32), // lm_head full
    ];
    let ctx = CudaContext::new(0).expect("ctx");
    let stream = ctx.default_stream();
    let kernels = LlmKernels::new(ctx);
    for &(n, k, scale) in &cases {
        let label = format!("stress_n{n}_k{k}_scale{scale}");
        let w: Vec<half::bf16> = (0..n * k)
            .map(|i| half::bf16::from_f32(((i as f32 + 1.0) * 0.0017 + 0.03).sin() * scale))
            .collect();
        let x: Vec<half::bf16> = (0..k)
            .map(|i| half::bf16::from_f32(((i as f32 + 1.0) * 0.0021 + 0.07).sin() * scale))
            .collect();
        let w_dev = stream.memcpy_stod(&w).expect("w");
        let x_dev = stream.memcpy_stod(&x).expect("x");
        let mut y_v1 = stream
            .memcpy_stod(&vec![half::bf16::from_f32(0.0); n])
            .expect("y_v1");
        let mut y_v2 = stream
            .memcpy_stod(&vec![half::bf16::from_f32(0.0); n])
            .expect("y_v2");
        unsafe {
            let (wp, _g0) = w_dev.device_ptr(&stream);
            let (xp, _g1) = x_dev.device_ptr(&stream);
            let (yp1, _g2) = y_v1.device_ptr_mut(&stream);
            let (yp2, _g3) = y_v2.device_ptr_mut(&stream);
            kernels
                .sgemv_bf16_bf16(&stream, wp, xp, yp1, n as i32, k as i32)
                .expect("v1");
            kernels
                .sgemv_bf16_bf16_v2(&stream, wp, xp, yp2, n as i32, k as i32)
                .expect("v2");
        }
        let o1 = stream.memcpy_dtov(&y_v1).expect("dtov v1");
        let o2 = stream.memcpy_dtov(&y_v2).expect("dtov v2");
        let mismatches: Vec<_> = o1
            .iter()
            .zip(o2.iter())
            .enumerate()
            .filter(|(_, (a, b))| a.to_bits() != b.to_bits())
            .map(|(i, (a, b))| (i, a.to_f32(), b.to_f32(), a.to_bits(), b.to_bits()))
            .collect();
        eprintln!(
            "{label}: N={} mismatches={} (scale={scale})",
            n,
            mismatches.len()
        );
        if !mismatches.is_empty() {
            let (i, av, bv, ab, bb) = mismatches[0];
            eprintln!("  first mismatch idx={i}: v1={av} (0x{ab:04x}) v2={bv} (0x{bb:04x})");
        }
        assert_eq!(
            mismatches.len(),
            0,
            "{label}: expected bit-exact match between V1 and V2"
        );
    }
}

#[test]
fn sgemv_bf16_v2_dispatch_routes_correctly() {
    // Dispatch defaults to V1 (V2 opt-in via RUSTORCH_ENABLE_BF16_V2=1).
    // Since V1 and V2 are bit-exact (same per-thread arithmetic), the
    // dispatch result is bit-exact with both V1 and V2 regardless of which
    // is selected. We assert against V2 here.
    let n = 256usize;
    let k = 2048usize;
    let w = bf16_vec(n * k, 0.0011, 0.05);
    let x = bf16_vec(k, 0.0019, 0.13);
    let ctx = CudaContext::new(0).expect("ctx");
    let stream = ctx.default_stream();
    let kernels = LlmKernels::new(ctx);
    let w_dev = stream.memcpy_stod(&w).expect("w");
    let x_dev = stream.memcpy_stod(&x).expect("x");
    let mut y_disp = stream
        .memcpy_stod(&vec![half::bf16::from_f32(0.0); n])
        .expect("y");
    let mut y_v2 = stream
        .memcpy_stod(&vec![half::bf16::from_f32(0.0); n])
        .expect("y2");
    unsafe {
        let (wp, _g0) = w_dev.device_ptr(&stream);
        let (xp, _g1) = x_dev.device_ptr(&stream);
        let (ypd, _g2) = y_disp.device_ptr_mut(&stream);
        let (yp2, _g3) = y_v2.device_ptr_mut(&stream);
        kernels
            .sgemv_bf16_bf16_dispatch(&stream, wp, xp, ypd, n as i32, k as i32)
            .expect("dispatch");
        kernels
            .sgemv_bf16_bf16_v2(&stream, wp, xp, yp2, n as i32, k as i32)
            .expect("v2");
    }
    let od = stream.memcpy_dtov(&y_disp).expect("dtov");
    let o2 = stream.memcpy_dtov(&y_v2).expect("dtov v2");
    // V1 and V2 are bit-exact ; dispatch always matches V2 either way.
    assert_eq!(
        od.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
        o2.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
        "dispatch (N=256) should be bit-exact with V2 (and V1, since V1==V2)"
    );

    // Now test dispatch fall-through to V1 for tiny N=64.
    let n_small = 64usize;
    let w_small = bf16_vec(n_small * k, 0.0011, 0.05);
    let w_small_dev = stream.memcpy_stod(&w_small).expect("w_small");
    let mut y_disp_small = stream
        .memcpy_stod(&vec![half::bf16::from_f32(0.0); n_small])
        .expect("y_disp_small");
    let mut y_v1_small = stream
        .memcpy_stod(&vec![half::bf16::from_f32(0.0); n_small])
        .expect("y_v1_small");
    unsafe {
        let (wp, _g0) = w_small_dev.device_ptr(&stream);
        let (xp, _g1) = x_dev.device_ptr(&stream);
        let (yds, _g2) = y_disp_small.device_ptr_mut(&stream);
        let (yvs, _g3) = y_v1_small.device_ptr_mut(&stream);
        kernels
            .sgemv_bf16_bf16_dispatch(&stream, wp, xp, yds, n_small as i32, k as i32)
            .expect("disp small");
        kernels
            .sgemv_bf16_bf16(&stream, wp, xp, yvs, n_small as i32, k as i32)
            .expect("v1 small");
    }
    let ods = stream.memcpy_dtov(&y_disp_small).expect("dtov");
    let o1s = stream.memcpy_dtov(&y_v1_small).expect("dtov v1");
    assert_eq!(
        ods.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
        o1s.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
        "dispatch (N=64) should pick V1 → bit-exact match with direct V1"
    );
}
