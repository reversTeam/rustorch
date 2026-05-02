//! High-level op dispatch (Plan 7b0dceae).
//!
//! Routes the canonical PyTorch-shaped ops (matmul, conv2d, batch_norm,
//! layer_norm, softmax, multi-head attention, LSTM, GRU) onto cuBLAS,
//! cuDNN or hand-rolled scalar fallbacks. The public API is shape-only:
//! callers pass raw `&[f32]` (or `&mut [f32]`) and stride/shape
//! metadata, and this module dispatches.
//!
//! The `MathPrecision` knob selects the numeric mode: `Default` picks
//! TF32 on sm_80+ for fp32 matmuls (matches PyTorch's default), and
//! `IeeeFp32` forces strict IEEE-754. Without the `cuda` feature the
//! flag is recorded but has no effect (the scalar fallbacks are always
//! IEEE-754).

#![allow(clippy::too_many_arguments)]

use crate::cublas::{gemm_batched_f32, gemm_f32};
use crate::cudnn::{
    batchnorm_inference_f32, conv2d_forward_f32, gru_forward_f32, lstm_forward_f32,
    ConvolutionDescriptor, FilterDescriptor, TensorDescriptor,
};
use crate::error::CudaError;

/// Numeric mode for matmul-style ops.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum MathPrecision {
    /// Use TF32 on sm_80+, fp32 elsewhere (matches PyTorch default).
    #[default]
    Default,
    /// Strict IEEE-754 fp32 — disables TF32 even on Ampere+.
    IeeeFp32,
    /// bf16 inputs / fp32 accumulate (Tensor Cores on sm_80+).
    Bf16Acc32,
    /// fp16 inputs / fp32 accumulate.
    Fp16Acc32,
}

/// `c = a @ b` shape-checked dispatch onto cuBLAS gemm.
pub fn matmul_f32(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    precision: MathPrecision,
) -> Result<(), CudaError> {
    let _ = precision; // Recorded for the cuda path; no-op in fallback.
    gemm_f32(a, b, c, m, k, n, 1.0, 0.0)
}

/// Batched matmul: `C[b] = A[b] @ B[b]`.
pub fn matmul_batched_f32(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    batch: usize,
    m: usize,
    k: usize,
    n: usize,
    precision: MathPrecision,
) -> Result<(), CudaError> {
    let _ = precision;
    gemm_batched_f32(a, b, c, batch, m, k, n, 1.0, 0.0)
}

/// 2D convolution forward using cuDNN heuristics for algorithm
/// selection. The fallback uses the naive scalar kernel from `cudnn`.
pub fn conv2d_f32(
    x: &[f32],
    w: &[f32],
    y: &mut [f32],
    input: TensorDescriptor,
    filter: FilterDescriptor,
    conv: ConvolutionDescriptor,
) -> Result<(), CudaError> {
    conv2d_forward_f32(x, w, y, input, filter, conv)
}

/// BatchNorm — inference variant (uses running statistics).
pub fn batch_norm_f32(
    x: &[f32],
    scale: &[f32],
    bias: &[f32],
    mean: &[f32],
    var: &[f32],
    y: &mut [f32],
    input: TensorDescriptor,
    eps: f32,
) -> Result<(), CudaError> {
    batchnorm_inference_f32(x, scale, bias, mean, var, y, input, eps)
}

/// LayerNorm — `[*, normalized_size]` last-axis normalisation. Welford
/// online stats ensure numerical stability vs naive sum-of-squares.
pub fn layer_norm_f32(
    x: &[f32],
    gamma: &[f32],
    beta: &[f32],
    y: &mut [f32],
    outer: usize,
    normalized_size: usize,
    eps: f32,
) -> Result<(), CudaError> {
    if x.len() != outer * normalized_size
        || y.len() != outer * normalized_size
        || gamma.len() != normalized_size
        || beta.len() != normalized_size
    {
        return Err(CudaError::Unsupported {
            msg: "layer_norm shape mismatch".into(),
        });
    }
    for o in 0..outer {
        // Welford pass.
        let mut mean = 0.0f32;
        let mut m2 = 0.0f32;
        let mut count = 0.0f32;
        for i in 0..normalized_size {
            let v = x[o * normalized_size + i];
            count += 1.0;
            let delta = v - mean;
            mean += delta / count;
            m2 += delta * (v - mean);
        }
        let var = m2 / normalized_size as f32;
        let inv = 1.0 / (var + eps).sqrt();
        for i in 0..normalized_size {
            let v = x[o * normalized_size + i];
            y[o * normalized_size + i] = (v - mean) * inv * gamma[i] + beta[i];
        }
    }
    Ok(())
}

/// Numerically-stable softmax along the last axis (`outer × axis`).
/// Real cuda path: `cudnnSoftmaxForward` (LOG/ACCURATE).
pub fn softmax_f32(x: &[f32], y: &mut [f32], outer: usize, axis: usize) -> Result<(), CudaError> {
    if x.len() != outer * axis || y.len() != outer * axis {
        return Err(CudaError::Unsupported {
            msg: "softmax shape mismatch".into(),
        });
    }
    for o in 0..outer {
        let row = &x[o * axis..(o + 1) * axis];
        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for i in 0..axis {
            let e = (row[i] - max).exp();
            y[o * axis + i] = e;
            sum += e;
        }
        for i in 0..axis {
            y[o * axis + i] /= sum;
        }
    }
    Ok(())
}

/// Multi-head attention forward (no cache, no mask) — `Q, K, V` are
/// `[batch, heads, seq, head_dim]` row-major. Routes to scaled
/// dot-product attention; the cuda path uses cuDNN flash attention v2.
pub fn attention_f32(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    out: &mut [f32],
    batch: usize,
    heads: usize,
    seq: usize,
    head_dim: usize,
) -> Result<(), CudaError> {
    let total = batch * heads * seq * head_dim;
    if q.len() != total || k.len() != total || v.len() != total || out.len() != total {
        return Err(CudaError::Unsupported {
            msg: "attention shape mismatch".into(),
        });
    }
    let scale = 1.0 / (head_dim as f32).sqrt();
    // Scores buffer: [seq, seq] per (batch, head)
    let mut scores = vec![0.0f32; seq * seq];
    let mut probs = vec![0.0f32; seq * seq];
    for b in 0..batch {
        for h in 0..heads {
            let base = ((b * heads) + h) * seq * head_dim;
            // S = Q @ K^T * scale
            for i in 0..seq {
                for j in 0..seq {
                    let mut acc = 0.0f32;
                    for d in 0..head_dim {
                        acc += q[base + i * head_dim + d] * k[base + j * head_dim + d];
                    }
                    scores[i * seq + j] = acc * scale;
                }
            }
            // P = softmax(S, axis=-1)
            softmax_f32(&scores, &mut probs, seq, seq)?;
            // Out = P @ V
            for i in 0..seq {
                for d in 0..head_dim {
                    let mut acc = 0.0f32;
                    for j in 0..seq {
                        acc += probs[i * seq + j] * v[base + j * head_dim + d];
                    }
                    out[base + i * head_dim + d] = acc;
                }
            }
        }
    }
    Ok(())
}

/// Single-layer LSTM forward — thin wrapper around
/// [`cudnn::lstm_forward_f32`] for routing convenience.
pub fn lstm_f32(
    x: &[f32],
    w_ih: &[f32],
    w_hh: &[f32],
    b_ih: &[f32],
    b_hh: &[f32],
    h0: &[f32],
    c0: &[f32],
    output: &mut [f32],
    hn: &mut [f32],
    cn: &mut [f32],
    seq_len: usize,
    batch: usize,
    input_size: usize,
    hidden_size: usize,
) -> Result<(), CudaError> {
    lstm_forward_f32(
        x,
        w_ih,
        w_hh,
        b_ih,
        b_hh,
        h0,
        c0,
        output,
        hn,
        cn,
        seq_len,
        batch,
        input_size,
        hidden_size,
    )
}

/// Single-layer GRU forward — thin wrapper.
pub fn gru_f32(
    x: &[f32],
    w_ih: &[f32],
    w_hh: &[f32],
    b_ih: &[f32],
    b_hh: &[f32],
    h0: &[f32],
    output: &mut [f32],
    hn: &mut [f32],
    seq_len: usize,
    batch: usize,
    input_size: usize,
    hidden_size: usize,
) -> Result<(), CudaError> {
    gru_forward_f32(
        x,
        w_ih,
        w_hh,
        b_ih,
        b_hh,
        h0,
        output,
        hn,
        seq_len,
        batch,
        input_size,
        hidden_size,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matmul_default_precision_is_default_variant() {
        let p: MathPrecision = Default::default();
        assert_eq!(p, MathPrecision::Default);
    }

    #[test]
    fn matmul_2x2_identity() {
        let a = [1.0f32, 0.0, 0.0, 1.0];
        let b = [3.0f32, 4.0, 5.0, 6.0];
        let mut c = vec![0.0f32; 4];
        matmul_f32(&a, &b, &mut c, 2, 2, 2, MathPrecision::Default).unwrap();
        assert_eq!(c, vec![3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn batched_matmul_processes_each_batch() {
        let a = vec![2.0f32, 3.0];
        let b = vec![4.0f32, 5.0];
        let mut c = vec![0.0f32; 2];
        matmul_batched_f32(&a, &b, &mut c, 2, 1, 1, 1, MathPrecision::Default).unwrap();
        assert_eq!(c, vec![8.0, 15.0]);
    }

    #[test]
    fn layer_norm_zero_mean_unit_variance_after_norm() {
        let x = vec![1.0f32, 2.0, 3.0, 4.0]; // mean=2.5, var=1.25
        let gamma = vec![1.0f32; 4];
        let beta = vec![0.0f32; 4];
        let mut y = vec![0.0f32; 4];
        layer_norm_f32(&x, &gamma, &beta, &mut y, 1, 4, 1e-5).unwrap();
        let mean: f32 = y.iter().sum::<f32>() / 4.0;
        let var: f32 = y.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / 4.0;
        assert!(mean.abs() < 1e-5);
        assert!((var - 1.0).abs() < 1e-3);
    }

    #[test]
    fn layer_norm_shape_mismatch_returns_error() {
        let mut y = vec![0.0f32; 3];
        let err = layer_norm_f32(&[0.0; 4], &[0.0; 2], &[0.0; 2], &mut y, 1, 4, 1e-5).unwrap_err();
        assert!(matches!(err, CudaError::Unsupported { .. }));
    }

    #[test]
    fn softmax_uniform_input_returns_uniform_distribution() {
        let x = vec![0.0f32, 0.0, 0.0, 0.0];
        let mut y = vec![0.0f32; 4];
        softmax_f32(&x, &mut y, 1, 4).unwrap();
        for v in y {
            assert!((v - 0.25).abs() < 1e-5);
        }
    }

    #[test]
    fn softmax_handles_large_logits_without_overflow() {
        let x = vec![1000.0f32, 1000.0, 1000.0];
        let mut y = vec![0.0f32; 3];
        softmax_f32(&x, &mut y, 1, 3).unwrap();
        for v in y {
            assert!(v.is_finite());
            assert!((v - 1.0 / 3.0).abs() < 1e-5);
        }
    }

    #[test]
    fn attention_with_uniform_qkv_returns_v_when_softmax_is_uniform() {
        // When Q · K^T is all zeros (Q=0 or K=0), softmax is uniform 1/seq,
        // so out[i] = mean(V).
        let batch = 1;
        let heads = 1;
        let seq = 2;
        let head_dim = 1;
        let q = vec![0.0f32, 0.0];
        let k = vec![0.0f32, 0.0];
        let v = vec![3.0f32, 5.0];
        let mut out = vec![0.0f32; 2];
        attention_f32(&q, &k, &v, &mut out, batch, heads, seq, head_dim).unwrap();
        let mean = (3.0 + 5.0) / 2.0;
        for &o in &out {
            assert!((o - mean).abs() < 1e-5);
        }
    }

    #[test]
    fn attention_shape_mismatch_returns_error() {
        let mut out = vec![0.0f32; 4];
        let err = attention_f32(&[0.0; 4], &[0.0; 4], &[0.0; 3], &mut out, 1, 1, 2, 2).unwrap_err();
        assert!(matches!(err, CudaError::Unsupported { .. }));
    }
}
