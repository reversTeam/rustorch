//! T246.8 A1 — Bit-exact parity tests for the fused SSM mega-kernels.
//!
//! Two fused kernels replace per-SSM-layer chains of small launches :
//!
//! * `ssm_pre_step_bf16(alpha, beta, dt_bias, ssm_a, n)` replaces
//!   sigmoid_inplace(beta) → add_inplace(alpha, dt_bias) →
//!   softplus_inplace(alpha) → mul_inplace(alpha, ssm_a).
//!
//! * `ssm_post_step_bf16(out, z, gamma, eps, n_v, head_kv)` replaces
//!   rms_norm_bf16(out, gamma, eps, head_kv, n_v) → silu_bf16(z) →
//!   mul_inplace(out, z).
//!
//! Both fused kernels must produce results bit-identical to the unfused
//! chain (no FP-order changes within a thread). This file gates that.
//!
//! Running on GB10 (DGX Spark) :
//!   PATH=/usr/local/cuda-13.0/bin:$PATH \
//!     cargo test --release --features cuda -p rustorch-cuda \
//!       --test cuda_ssm_fusion_parity -- --nocapture --test-threads=1

#![cfg(feature = "cuda")]

use cudarc::driver::{CudaContext, DevicePtr, DevicePtrMut};
use rustorch_cuda::llm_kernels::LlmKernels;

fn bf16_vec(n: usize, seed: f32, off: f32) -> Vec<half::bf16> {
    (0..n)
        .map(|i| half::bf16::from_f32(((i as f32 + 1.0) * seed + off).sin() * 0.4))
        .collect()
}

fn assert_bit_exact(label: &str, a: &[half::bf16], b: &[half::bf16]) {
    assert_eq!(a.len(), b.len(), "{label}: length mismatch");
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        if x.to_bits() != y.to_bits() {
            panic!(
                "{label}: bit-mismatch at {i}: unfused=0x{:04x} ({}) fused=0x{:04x} ({})",
                x.to_bits(),
                x.to_f32(),
                y.to_bits(),
                y.to_f32()
            );
        }
    }
}

#[test]
fn ssm_pre_step_bf16_matches_unfused_chain() {
    let n: usize = 1024; // matches Qwen3.6 SSM n_v dimension footprint
    let alpha_init = bf16_vec(n, 0.013, 0.07);
    let beta_init = bf16_vec(n, 0.017, 0.11);
    let dt_bias = bf16_vec(n, 0.019, 0.05);
    let ssm_a = bf16_vec(n, 0.023, 0.03);

    let ctx = CudaContext::new(0).expect("ctx");
    let stream = ctx.default_stream();
    let kernels = LlmKernels::new(ctx);

    // ── Reference : run the four unfused kernels in sequence.
    let mut alpha_ref = stream.memcpy_stod(&alpha_init).expect("alpha_ref");
    let mut beta_ref = stream.memcpy_stod(&beta_init).expect("beta_ref");
    let dt_bias_dev = stream.memcpy_stod(&dt_bias).expect("dt_bias");
    let ssm_a_dev = stream.memcpy_stod(&ssm_a).expect("ssm_a");

    unsafe {
        let (ap, _g0) = alpha_ref.device_ptr_mut(&stream);
        let (bp, _g1) = beta_ref.device_ptr_mut(&stream);
        let (db, _g2) = dt_bias_dev.device_ptr(&stream);
        let (sa, _g3) = ssm_a_dev.device_ptr(&stream);

        kernels
            .sigmoid_inplace_bf16(&stream, bp, n as i32)
            .expect("sigmoid");
        kernels
            .add_inplace_bf16(&stream, ap, db, n as i32)
            .expect("add_dt");
        kernels
            .softplus_inplace_bf16(&stream, ap, n as i32)
            .expect("softplus");
        kernels
            .mul_inplace_bf16(&stream, ap, sa, n as i32)
            .expect("mul_a");
    }
    let alpha_ref_h: Vec<half::bf16> = stream.memcpy_dtov(&alpha_ref).expect("alpha_ref dtoh");
    let beta_ref_h: Vec<half::bf16> = stream.memcpy_dtov(&beta_ref).expect("beta_ref dtoh");

    // ── Fused : single kernel.
    let mut alpha_f = stream.memcpy_stod(&alpha_init).expect("alpha_f");
    let mut beta_f = stream.memcpy_stod(&beta_init).expect("beta_f");
    unsafe {
        let (ap, _g0) = alpha_f.device_ptr_mut(&stream);
        let (bp, _g1) = beta_f.device_ptr_mut(&stream);
        let (db, _g2) = dt_bias_dev.device_ptr(&stream);
        let (sa, _g3) = ssm_a_dev.device_ptr(&stream);
        kernels
            .ssm_pre_step_bf16(&stream, ap, bp, db, sa, n as i32)
            .expect("ssm_pre_step");
    }
    let alpha_f_h: Vec<half::bf16> = stream.memcpy_dtov(&alpha_f).expect("alpha_f dtoh");
    let beta_f_h: Vec<half::bf16> = stream.memcpy_dtov(&beta_f).expect("beta_f dtoh");

    assert_bit_exact("alpha", &alpha_ref_h, &alpha_f_h);
    assert_bit_exact("beta", &beta_ref_h, &beta_f_h);
}

#[test]
fn ssm_post_step_bf16_matches_unfused_chain() {
    // Qwen3.6 shape : n_v=32 heads × head_kv=128 channels.
    let n_v: usize = 32;
    let head_kv: usize = 128;
    let total = n_v * head_kv;
    let eps = 1e-6f32;

    let out_init = bf16_vec(total, 0.013, 0.07);
    let z_init = bf16_vec(total, 0.017, 0.11);
    let gamma = bf16_vec(head_kv, 0.019, 0.05);

    let ctx = CudaContext::new(0).expect("ctx");
    let stream = ctx.default_stream();
    let kernels = LlmKernels::new(ctx);

    // ── Reference : 3 unfused kernels.
    let mut out_ref = stream.memcpy_stod(&out_init).expect("out_ref");
    let mut z_ref = stream.memcpy_stod(&z_init).expect("z_ref");
    let gamma_dev = stream.memcpy_stod(&gamma).expect("gamma");
    unsafe {
        let (op, _g0) = out_ref.device_ptr_mut(&stream);
        let (zp, _g1) = z_ref.device_ptr_mut(&stream);
        let (gp, _g2) = gamma_dev.device_ptr(&stream);
        // rms_norm_bf16(out, gamma, eps, head_kv, n_v) — n_v batches of head_kv
        kernels
            .rms_norm_bf16(&stream, op, gp, eps, head_kv as i32, n_v as i32)
            .expect("rms_norm");
        kernels.silu_bf16(&stream, zp, total as i32).expect("silu");
        kernels
            .mul_inplace_bf16(&stream, op, zp, total as i32)
            .expect("mul");
    }
    let out_ref_h: Vec<half::bf16> = stream.memcpy_dtov(&out_ref).expect("out_ref dtoh");
    let z_ref_h: Vec<half::bf16> = stream.memcpy_dtov(&z_ref).expect("z_ref dtoh");

    // ── Fused.
    let mut out_f = stream.memcpy_stod(&out_init).expect("out_f");
    let mut z_f = stream.memcpy_stod(&z_init).expect("z_f");
    unsafe {
        let (op, _g0) = out_f.device_ptr_mut(&stream);
        let (zp, _g1) = z_f.device_ptr_mut(&stream);
        let (gp, _g2) = gamma_dev.device_ptr(&stream);
        kernels
            .ssm_post_step_bf16(&stream, op, zp, gp, eps, n_v as i32, head_kv as i32)
            .expect("ssm_post_step");
    }
    let out_f_h: Vec<half::bf16> = stream.memcpy_dtov(&out_f).expect("out_f dtoh");
    let z_f_h: Vec<half::bf16> = stream.memcpy_dtov(&z_f).expect("z_f dtoh");

    assert_bit_exact("out", &out_ref_h, &out_f_h);
    assert_bit_exact("z", &z_ref_h, &z_f_h);
}
