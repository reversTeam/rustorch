//! T246.9 NVFP4.2 — Parity tests for the NVFP4 SGEMV kernels :
//! `sgemv_nvfp4_bf16` (single-Linear) and `sgemv_nvfp4_bf16_indexed`
//! (MoE, with device-side expert pointer dispatch).
//!
//! ## Reference
//! The CPU reference dequantizes the NVFP4 weights to fp32 (per-block
//! UE4M3 scale + per-tensor `weight_global * input_global` alpha at
//! the end), runs an exact dot product, and BF16 down-casts.
//!
//! The kernel uses the same fp32 arithmetic ; we expect bit-equality
//! on simple inputs (where the reduction order matches) and ≤ 1 BF16
//! ULP drift on randomized inputs.
//!
//! ## Run on GB10 (DGX Spark)
//! ```
//! PATH=/usr/local/cuda-13.0/bin:$PATH \
//!   LD_LIBRARY_PATH=/usr/local/cuda-13.0/targets/sbsa-linux/lib:$LD_LIBRARY_PATH \
//!   cargo test --release --features cuda -p rustorch-cuda \
//!     --test cuda_nvfp4_indexed_parity -- --nocapture --test-threads=1
//! ```

#![cfg(feature = "cuda")]

use cudarc::driver::{CudaContext, DevicePtr, DevicePtrMut};
use rustorch_cuda::llm_kernels::LlmKernels;

// ---------------------------------------------------------------------------
// CPU reference : dequant FP4 → fp32, matmul, BF16 down-cast
// ---------------------------------------------------------------------------

const FP4_MAG: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];

fn fp4_decode(code: u8) -> f32 {
    let m = FP4_MAG[(code & 7) as usize];
    if (code & 8) != 0 {
        -m
    } else {
        m
    }
}

fn ue4m3_decode(b: u8) -> f32 {
    let e = ((b >> 3) & 0xF) as i32;
    let m = (b & 0x7) as f32;
    if e == 0 {
        m * (1.0 / 1024.0)
    } else {
        let mantissa = 1.0 + m * 0.125;
        // 2^(e - 7)
        let exp = e - 7;
        if exp >= 0 {
            mantissa * ((1u32 << exp as u32) as f32)
        } else {
            mantissa / ((1u32 << (-exp) as u32) as f32)
        }
    }
}

/// CPU NVFP4 SGEMV reference. `packed` is `[N, K/2]`, `scale` is
/// `[N, K/16]` (UE4M3), `alpha = 1 / (w_g * in_g)`. Returns BF16 [N].
fn cpu_nvfp4_sgemv(
    packed: &[u8],
    scale: &[u8],
    alpha: f32,
    x: &[half::bf16],
    n: usize,
    k: usize,
) -> Vec<half::bf16> {
    assert_eq!(packed.len(), n * k / 2);
    assert_eq!(scale.len(), n * k / 16);
    assert_eq!(x.len(), k);
    let mut out = Vec::with_capacity(n);
    let n_blocks = k / 16;
    for row in 0..n {
        let mut acc: f32 = 0.0;
        for b in 0..n_blocks {
            let p_off = row * (k / 2) + b * 8;
            let s_off = row * n_blocks + b;
            let s = ue4m3_decode(scale[s_off]);
            for i in 0..8 {
                let by = packed[p_off + i];
                let lo = by & 0xF;
                let hi = (by >> 4) & 0xF;
                let w0 = fp4_decode(lo) * s;
                let w1 = fp4_decode(hi) * s;
                let x0 = x[b * 16 + 2 * i].to_f32();
                let x1 = x[b * 16 + 2 * i + 1].to_f32();
                acc += w0 * x0 + w1 * x1;
            }
        }
        out.push(half::bf16::from_f32(acc * alpha));
    }
    out
}

// ---------------------------------------------------------------------------
// Synthetic data builders
// ---------------------------------------------------------------------------

/// Build a deterministic synthetic NVFP4 weight matrix. Packs simple
/// codes that exercise both signs and several magnitudes.
fn build_synthetic_nvfp4(n: usize, k: usize, seed: u32) -> (Vec<u8>, Vec<u8>) {
    assert!(k % 16 == 0);
    let mut packed = vec![0u8; n * k / 2];
    let mut scale = vec![0u8; n * k / 16];
    let mut s = seed;
    for byte in packed.iter_mut() {
        // xorshift32
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        // Each byte = 2 FP4 codes. Mask to 4-bit.
        *byte = ((s >> 16) & 0xFF) as u8;
    }
    // Scales : pick a mix of UE4M3 codes that decode to ~ {0.5, 1.0, 2.0}
    // (E={6,7,8}, M=0). Avoid subnormals.
    let scale_choices: [u8; 4] = [
        (6 << 3),
        (7 << 3),
        (8 << 3),
        (7 << 3) | 0x3, // 1.0 * (1 + 3/8) = 1.375
    ];
    for (i, b) in scale.iter_mut().enumerate() {
        *b = scale_choices[i % 4];
    }
    (packed, scale)
}

fn bf16_vec(n: usize, scale: f32, seed: f32) -> Vec<half::bf16> {
    (0..n)
        .map(|i| half::bf16::from_f32(((i as f32 + 1.0) * scale + seed).sin() * 0.4))
        .collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

fn run_nvfp4_solo_case(label: &str, n: usize, k: usize, alpha: f32, seed: u32) {
    let (packed, scale) = build_synthetic_nvfp4(n, k, seed);
    let x = bf16_vec(k, 0.0021, 0.07);
    let cpu_y = cpu_nvfp4_sgemv(&packed, &scale, alpha, &x, n, k);

    let ctx = CudaContext::new(0).expect("ctx");
    let stream = ctx.default_stream();
    let kernels = LlmKernels::new(ctx);

    let packed_dev = stream.memcpy_stod(&packed).expect("packed");
    let scale_dev = stream.memcpy_stod(&scale).expect("scale");
    let x_dev = stream.memcpy_stod(&x).expect("x");
    let mut y_dev = stream
        .memcpy_stod(&vec![half::bf16::from_f32(0.0); n])
        .expect("y");

    unsafe {
        let (pp, _g0) = packed_dev.device_ptr(&stream);
        let (sp, _g1) = scale_dev.device_ptr(&stream);
        let (xp, _g2) = x_dev.device_ptr(&stream);
        let (yp, _g3) = y_dev.device_ptr_mut(&stream);
        kernels
            .sgemv_nvfp4_bf16(&stream, pp, sp, alpha, xp, yp, n as i32, k as i32)
            .expect("kernel launch");
    }
    let gpu_y: Vec<half::bf16> = stream.memcpy_dtov(&y_dev).expect("dl");

    assert_eq!(gpu_y.len(), cpu_y.len());

    let mut max_abs = 0.0f32;
    let mut bad: usize = 0;
    let mut first_bad: Option<(usize, half::bf16, half::bf16)> = None;
    for (i, (a, b)) in cpu_y.iter().zip(gpu_y.iter()).enumerate() {
        if a.to_bits() == b.to_bits() {
            continue;
        }
        let av = a.to_f32();
        let bv = b.to_f32();
        let abs = (av - bv).abs();
        max_abs = max_abs.max(abs);
        let ulp = (a.to_bits() as i32 - b.to_bits() as i32).unsigned_abs();
        if ulp > 4 && (av.abs() < 1e-3 || abs / av.abs() > 0.05) {
            bad += 1;
            if first_bad.is_none() {
                first_bad = Some((i, *a, *b));
            }
        }
    }
    let bad_pct = (bad as f32) / (n as f32) * 100.0;
    eprintln!("{label}: N={n} K={k} alpha={alpha} max_abs={max_abs} bad={bad} ({bad_pct:.2}%)");
    if bad_pct > 2.0 {
        let (i, a, b) = first_bad.unwrap();
        panic!(
            "{label}: too many drift > 4 ULP & > 5% rel ({bad_pct:.2}% > 2%). \
             First bad at {i}: cpu={} (0x{:04x}) gpu={} (0x{:04x})",
            a.to_f32(),
            a.to_bits(),
            b.to_f32(),
            b.to_bits()
        );
    }
}

#[test]
fn nvfp4_solo_n_k_minimum() {
    // K=16 → 1 micro-block per row, sanity check on smallest valid K.
    run_nvfp4_solo_case("solo n=8 k=16 a=1.0", 8, 16, 1.0, 0xCAFE);
}

#[test]
fn nvfp4_solo_qwen36_attn_shape() {
    // Q projection on Qwen3.6-A3B : K=2048, N=4096 (16 q heads × 256 head_dim).
    run_nvfp4_solo_case("solo qwen attn q-proj", 4096, 2048, 0.5, 0x12345);
}

#[test]
fn nvfp4_solo_qwen36_moe_shape() {
    // Per-expert gate_proj on Qwen3.6-A3B : K=2048, N=512 (expert_f).
    run_nvfp4_solo_case("solo qwen moe gate", 512, 2048, 0.25, 0xABCDE);
}

#[test]
fn nvfp4_solo_alpha_zero_yields_zero() {
    run_nvfp4_solo_case("solo alpha=0", 64, 64, 0.0, 0x99);
    // Sanity : the CPU reference returns 0 for alpha=0, so the test
    // would fail on any non-zero kernel output.
}

// ---------------------------------------------------------------------------
// Indexed kernel test : 2-expert dispatch via topk_indices[slot]
// ---------------------------------------------------------------------------

fn run_nvfp4_indexed_case(label: &str, n: usize, k: usize, n_experts: usize, slot: usize) {
    assert!(slot < n_experts);
    // Build n_experts distinct weight matrices.
    let mut packs: Vec<Vec<u8>> = Vec::with_capacity(n_experts);
    let mut scales: Vec<Vec<u8>> = Vec::with_capacity(n_experts);
    let mut alphas: Vec<f32> = Vec::with_capacity(n_experts);
    for e in 0..n_experts {
        let (p, s) = build_synthetic_nvfp4(n, k, 0xC0FFEE_u32 + e as u32 * 7);
        packs.push(p);
        scales.push(s);
        alphas.push(0.5 + (e as f32) * 0.25);
    }
    let x = bf16_vec(k, 0.003, 0.11);

    // CPU reference using only the slot-selected expert.
    let e_sel = slot; // topk_indices = [0, 1, ..., n_experts-1] ; slot picks e_sel
    let cpu_y = cpu_nvfp4_sgemv(&packs[e_sel], &scales[e_sel], alphas[e_sel], &x, n, k);

    let ctx = CudaContext::new(0).expect("ctx");
    let stream = ctx.default_stream();
    let kernels = LlmKernels::new(ctx);

    // Upload each expert's packed/scale to device, collect base pointers.
    let packed_devs: Vec<_> = packs
        .iter()
        .map(|p| stream.memcpy_stod(p).expect("pack"))
        .collect();
    let scale_devs: Vec<_> = scales
        .iter()
        .map(|s| stream.memcpy_stod(s).expect("scale"))
        .collect();
    let mut packed_ptrs_h: Vec<u64> = Vec::with_capacity(n_experts);
    let mut scale_ptrs_h: Vec<u64> = Vec::with_capacity(n_experts);
    for (pd, sd) in packed_devs.iter().zip(scale_devs.iter()) {
        let (p, _g) = pd.device_ptr(&stream);
        packed_ptrs_h.push(p);
        let (s, _g2) = sd.device_ptr(&stream);
        scale_ptrs_h.push(s);
    }
    let pack_ptrs_dev = stream.memcpy_stod(&packed_ptrs_h).expect("pack_ptrs");
    let scale_ptrs_dev = stream.memcpy_stod(&scale_ptrs_h).expect("scale_ptrs");
    let alphas_dev = stream.memcpy_stod(&alphas).expect("alphas");
    let topk_idx: Vec<i32> = (0..n_experts as i32).collect();
    let topk_idx_dev = stream.memcpy_stod(&topk_idx).expect("topk");

    let x_dev = stream.memcpy_stod(&x).expect("x");
    let mut y_dev = stream
        .memcpy_stod(&vec![half::bf16::from_f32(0.0); n])
        .expect("y");

    unsafe {
        let (pp, _g0) = pack_ptrs_dev.device_ptr(&stream);
        let (sp, _g1) = scale_ptrs_dev.device_ptr(&stream);
        let (ap, _g2) = alphas_dev.device_ptr(&stream);
        let (tp, _g3) = topk_idx_dev.device_ptr(&stream);
        let (xp, _g4) = x_dev.device_ptr(&stream);
        let (yp, _g5) = y_dev.device_ptr_mut(&stream);
        kernels
            .sgemv_nvfp4_bf16_indexed(
                &stream,
                pp,
                sp,
                ap,
                tp,
                slot as i32,
                xp,
                yp,
                n as i32,
                k as i32,
            )
            .expect("indexed launch");
    }
    let gpu_y: Vec<half::bf16> = stream.memcpy_dtov(&y_dev).expect("dl");

    let mut max_abs = 0.0f32;
    let mut bad: usize = 0;
    let mut first_bad: Option<(usize, half::bf16, half::bf16)> = None;
    for (i, (a, b)) in cpu_y.iter().zip(gpu_y.iter()).enumerate() {
        if a.to_bits() == b.to_bits() {
            continue;
        }
        let av = a.to_f32();
        let bv = b.to_f32();
        let abs = (av - bv).abs();
        max_abs = max_abs.max(abs);
        let ulp = (a.to_bits() as i32 - b.to_bits() as i32).unsigned_abs();
        if ulp > 4 && (av.abs() < 1e-3 || abs / av.abs() > 0.05) {
            bad += 1;
            if first_bad.is_none() {
                first_bad = Some((i, *a, *b));
            }
        }
    }
    let bad_pct = (bad as f32) / (n as f32) * 100.0;
    eprintln!(
        "{label}: N={n} K={k} n_experts={n_experts} slot={slot} max_abs={max_abs} bad={bad} ({bad_pct:.2}%)"
    );
    if bad_pct > 2.0 {
        let (i, a, b) = first_bad.unwrap();
        panic!(
            "{label}: too many drift > 4 ULP & > 5% rel ({bad_pct:.2}% > 2%). \
             First bad at {i}: cpu={} (0x{:04x}) gpu={} (0x{:04x})",
            a.to_f32(),
            a.to_bits(),
            b.to_f32(),
            b.to_bits()
        );
    }
}

#[test]
fn nvfp4_indexed_2_experts_slot_0() {
    run_nvfp4_indexed_case("idx 2-exp slot 0", 64, 64, 2, 0);
}

#[test]
fn nvfp4_indexed_2_experts_slot_1() {
    run_nvfp4_indexed_case("idx 2-exp slot 1", 64, 64, 2, 1);
}

#[test]
fn nvfp4_indexed_8_experts_qwen_moe_shape() {
    // 8 experts, slot=3 → expert 3 selected ; shape = Qwen3.6-A3B per-expert
    // gate_proj ([512, 2048]).
    run_nvfp4_indexed_case("idx 8-exp qwen moe slot 3", 512, 2048, 8, 3);
}

// ---------------------------------------------------------------------------
// Sanity test : indexed-kernel with slot=N picks expert N (no aliasing)
// ---------------------------------------------------------------------------

#[test]
fn nvfp4_indexed_slot_picks_correct_expert() {
    // 2 experts ; slot=0 should produce a different output than slot=1.
    let (n, k, n_experts) = (32, 32, 2);
    let mut packs = Vec::new();
    let mut scales = Vec::new();
    let mut alphas = Vec::new();
    for e in 0..n_experts {
        let (p, s) = build_synthetic_nvfp4(n, k, 0xDEADu32 + e as u32);
        packs.push(p);
        scales.push(s);
        alphas.push(1.0);
    }
    let x = bf16_vec(k, 0.0021, 0.0);

    let ctx = CudaContext::new(0).expect("ctx");
    let stream = ctx.default_stream();
    let kernels = LlmKernels::new(ctx);

    let packed_devs: Vec<_> = packs
        .iter()
        .map(|p| stream.memcpy_stod(p).expect("p"))
        .collect();
    let scale_devs: Vec<_> = scales
        .iter()
        .map(|s| stream.memcpy_stod(s).expect("s"))
        .collect();
    let mut pack_ptrs_h: Vec<u64> = Vec::with_capacity(n_experts);
    let mut scale_ptrs_h: Vec<u64> = Vec::with_capacity(n_experts);
    for (pd, sd) in packed_devs.iter().zip(scale_devs.iter()) {
        let (p, _g) = pd.device_ptr(&stream);
        pack_ptrs_h.push(p);
        let (s, _g2) = sd.device_ptr(&stream);
        scale_ptrs_h.push(s);
    }
    let pack_ptrs_dev = stream.memcpy_stod(&pack_ptrs_h).expect("pp");
    let scale_ptrs_dev = stream.memcpy_stod(&scale_ptrs_h).expect("sp");
    let alphas_dev = stream.memcpy_stod(&alphas).expect("a");
    let topk = vec![0i32, 1];
    let topk_dev = stream.memcpy_stod(&topk).expect("topk");

    let x_dev = stream.memcpy_stod(&x).expect("x");
    let mut y_slot0 = stream
        .memcpy_stod(&vec![half::bf16::from_f32(0.0); n])
        .expect("y0");
    let mut y_slot1 = stream
        .memcpy_stod(&vec![half::bf16::from_f32(0.0); n])
        .expect("y1");

    unsafe {
        let (pp, _g0) = pack_ptrs_dev.device_ptr(&stream);
        let (sp, _g1) = scale_ptrs_dev.device_ptr(&stream);
        let (ap, _g2) = alphas_dev.device_ptr(&stream);
        let (tp, _g3) = topk_dev.device_ptr(&stream);
        let (xp, _g4) = x_dev.device_ptr(&stream);
        let (y0, _g5) = y_slot0.device_ptr_mut(&stream);
        kernels
            .sgemv_nvfp4_bf16_indexed(&stream, pp, sp, ap, tp, 0, xp, y0, n as i32, k as i32)
            .expect("slot 0");
        let (y1, _g6) = y_slot1.device_ptr_mut(&stream);
        kernels
            .sgemv_nvfp4_bf16_indexed(&stream, pp, sp, ap, tp, 1, xp, y1, n as i32, k as i32)
            .expect("slot 1");
    }
    let v0: Vec<half::bf16> = stream.memcpy_dtov(&y_slot0).expect("dl0");
    let v1: Vec<half::bf16> = stream.memcpy_dtov(&y_slot1).expect("dl1");
    let same = v0
        .iter()
        .zip(v1.iter())
        .filter(|(a, b)| a.to_bits() == b.to_bits())
        .count();
    assert!(
        same < n / 2,
        "slot 0 and slot 1 yielded same output {same}/{n} elements — not enough divergence"
    );
}
