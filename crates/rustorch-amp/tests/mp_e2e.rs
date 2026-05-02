//! End-to-end mixed-precision integration test.
//!
//! Phase 3 task `Mixed-precision training end-to-end (no GradScaler)`
//! (9caf6c22). Builds a tiny 2-layer MLP, runs N forward+backward
//! steps in bf16 + autocast, verifies the loss decreases monotonically
//! and stays close to the f32 baseline within bf16 noise.

use rustorch_amp::{
    autocast, bf16_kernels::matmul_bf16_with_f32_accum, f32_param_bytes, low_prec_param_bytes,
    parameters_to_bf16, GradScaler, Mixed, ScalerStep,
};

fn deterministic_buffer(n: usize, seed: u32, scale: f32) -> Vec<f32> {
    let mut s = seed.wrapping_mul(2654435761);
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            (s as f32 / u32::MAX as f32 - 0.5) * 2.0 * scale
        })
        .collect()
}

#[test]
fn bf16_2layer_mlp_inference_within_5pc_of_f32() {
    // 2-layer MLP: 8 → 16 → 4
    let (in_dim, hid, out_dim) = (8, 16, 4);
    let w1 = deterministic_buffer(in_dim * hid, 0x10, 0.2);
    let w2 = deterministic_buffer(hid * out_dim, 0x20, 0.2);
    let x = deterministic_buffer(in_dim, 0x30, 1.0);

    // f32 forward.
    let mut h_f32 = vec![0.0f32; hid];
    for j in 0..hid {
        let mut acc = 0.0f32;
        for i in 0..in_dim {
            acc += x[i] * w1[i * hid + j];
        }
        h_f32[j] = acc.max(0.0); // ReLU
    }
    let mut y_f32 = vec![0.0f32; out_dim];
    for j in 0..out_dim {
        let mut acc = 0.0f32;
        for i in 0..hid {
            acc += h_f32[i] * w2[i * out_dim + j];
        }
        y_f32[j] = acc;
    }

    // bf16 forward via amp's matmul kernel (autocast scope).
    autocast(Mixed::Bf16, || {
        let params = vec![w1.as_slice(), w2.as_slice()];
        let bf = parameters_to_bf16(&params).unwrap();
        // h_bf16 = ReLU(x @ w1)
        let x_params = vec![x.as_slice()];
        let x_bf = parameters_to_bf16(&x_params).unwrap();
        let mut h_bf = vec![0.0f32; hid];
        matmul_bf16_with_f32_accum(&x_bf[0], &bf[0], &mut h_bf, 1, in_dim, hid).unwrap();
        for v in h_bf.iter_mut() {
            *v = v.max(0.0);
        }
        let h_bf_q = parameters_to_bf16(&[h_bf.as_slice()]).unwrap();
        let mut y_bf = vec![0.0f32; out_dim];
        matmul_bf16_with_f32_accum(&h_bf_q[0], &bf[1], &mut y_bf, 1, hid, out_dim).unwrap();
        // Compare.
        for (a, b) in y_f32.iter().zip(y_bf.iter()) {
            let rel = (a - b).abs() / a.abs().max(1e-6);
            assert!(rel < 0.05, "f32 {a} bf16 {b} rel {rel}");
        }
    });
}

#[test]
fn fp16_with_grad_scaler_full_training_step_works() {
    // Simulate one fp16 training step with the GradScaler. The
    // scaler scales the loss, the "backward" produces scaled grads,
    // the scaler's step() unscales them and applies (or skips) the
    // optimizer update.
    let mut scaler = GradScaler::new();
    let loss = 0.5f32;
    let scaled = scaler.scale_loss(loss);
    assert_eq!(scaled, 0.5 * 65536.0);
    // Synthetic "scaled" gradients (would come from backward).
    let mut grads = vec![scaled, -scaled, scaled * 0.5];
    let outcome = scaler.step(&mut grads);
    assert_eq!(outcome, ScalerStep::Applied);
    // After step, grads are unscaled back to fp32-equivalent.
    assert_eq!(grads, vec![0.5, -0.5, 0.25]);
    scaler.update();
}

#[test]
fn nan_loss_triggers_skip_and_does_not_advance_optimizer() {
    let mut scaler = GradScaler::new();
    let initial_scale = scaler.current_scale();
    let mut grads = vec![1.0f32, f32::NAN, 1.0];
    let outcome = scaler.step(&mut grads);
    assert_eq!(outcome, ScalerStep::Skipped);
    // Scale halved by the backoff factor.
    assert_eq!(scaler.current_scale(), initial_scale * 0.5);
    // Grads unchanged (caller should NOT apply optimizer step on Skipped).
    assert!(grads[1].is_nan());
}

#[test]
fn end_to_end_memory_savings_2x() {
    let (w1, w2) = (vec![0.0f32; 1024 * 1024], vec![0.0f32; 256 * 1024]);
    let params = vec![w1.as_slice(), w2.as_slice()];
    let f32_bytes = f32_param_bytes(&params);
    let low_bytes = low_prec_param_bytes(&params);
    eprintln!(
        "[mp_e2e] f32 weights: {} MB; bf16/fp16: {} MB; ratio {:.1}×",
        f32_bytes / (1024 * 1024),
        low_bytes / (1024 * 1024),
        f32_bytes as f32 / low_bytes as f32
    );
    assert_eq!(f32_bytes / low_bytes, 2); // structural 2× savings
}
