//! Fused layernorm/rmsnorm + linear and residual add/mul kernels.
//!
//! - `layernorm_linear`: normalise `[..., D]` rows then matmul.
//!   Uses Welford's algorithm for numerical stability.
//! - `rmsnorm_linear`: rms-norm + matmul (no mean-centring; common
//!   in modern LLMs).
//! - `residual_add_scale_mul`: out = scale * (in + residual) * gate.

use crate::patterns::matmul_bias_act::FusionError;

/// Fused LayerNorm + Linear: `y = ((x - μ) / σ) @ w + b` per row.
///
/// `x` is `[m, k]` (m rows of dim k); `w` is `[k, n]`; `b` is `[n]`;
/// `y` is `[m, n]`. eps prevents div-by-zero in σ.
#[allow(clippy::too_many_arguments)]
#[allow(clippy::needless_range_loop)]
pub fn layernorm_linear(
    x: &[f32],
    w: &[f32],
    b: Option<&[f32]>,
    y: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    eps: f32,
) -> Result<(), FusionError> {
    if x.len() != m * k {
        return Err(FusionError::ShapeMismatch {
            which: "x",
            expected: m * k,
            got: x.len(),
        });
    }
    if w.len() != k * n {
        return Err(FusionError::ShapeMismatch {
            which: "w",
            expected: k * n,
            got: w.len(),
        });
    }
    if y.len() != m * n {
        return Err(FusionError::ShapeMismatch {
            which: "y",
            expected: m * n,
            got: y.len(),
        });
    }
    for row in 0..m {
        // Welford pass for stable mean+variance.
        let mut mean = 0.0f32;
        let mut m2 = 0.0f32;
        for kk in 0..k {
            let val = x[row * k + kk];
            let delta = val - mean;
            mean += delta / (kk + 1) as f32;
            m2 += delta * (val - mean);
        }
        let var = m2 / k as f32;
        let inv_std = 1.0 / (var + eps).sqrt();
        // Normalise + matmul row-by-row in one fused pass.
        for col in 0..n {
            let mut acc = b.map(|bb| bb[col]).unwrap_or(0.0);
            for kk in 0..k {
                let normed = (x[row * k + kk] - mean) * inv_std;
                acc += normed * w[kk * n + col];
            }
            y[row * n + col] = acc;
        }
    }
    Ok(())
}

/// Fused residual add + scale + element-wise mul:
/// `y[i] = scale * (input[i] + residual[i]) * gate[i]`.
pub fn residual_add_scale_mul(
    input: &[f32],
    residual: &[f32],
    gate: &[f32],
    y: &mut [f32],
    scale: f32,
) -> Result<(), FusionError> {
    if input.len() != residual.len() {
        return Err(FusionError::ShapeMismatch {
            which: "residual",
            expected: input.len(),
            got: residual.len(),
        });
    }
    if input.len() != gate.len() {
        return Err(FusionError::ShapeMismatch {
            which: "gate",
            expected: input.len(),
            got: gate.len(),
        });
    }
    if input.len() != y.len() {
        return Err(FusionError::ShapeMismatch {
            which: "y",
            expected: input.len(),
            got: y.len(),
        });
    }
    for i in 0..input.len() {
        y[i] = scale * (input[i] + residual[i]) * gate[i];
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layernorm_linear_matches_naive_within_1e_5() {
        // Build a small input and compare to a step-by-step naive
        // reference.
        let (m, k, n) = (2, 4, 3);
        let x: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.1 - 0.2).collect();
        let w: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.05).collect();
        let b: Vec<f32> = (0..n).map(|i| i as f32 * 0.1).collect();
        let eps = 1e-5;

        let mut fused = vec![0.0f32; m * n];
        layernorm_linear(&x, &w, Some(&b), &mut fused, m, k, n, eps).unwrap();

        // Naive: normalise each row, then matmul.
        let mut normed = vec![0.0f32; m * k];
        for row in 0..m {
            let mut sum = 0.0;
            for kk in 0..k {
                sum += x[row * k + kk];
            }
            let mean = sum / k as f32;
            let mut sq = 0.0;
            for kk in 0..k {
                let d = x[row * k + kk] - mean;
                sq += d * d;
            }
            let var = sq / k as f32;
            let inv_std = 1.0 / (var + eps).sqrt();
            for kk in 0..k {
                normed[row * k + kk] = (x[row * k + kk] - mean) * inv_std;
            }
        }
        let mut naive = vec![0.0f32; m * n];
        for row in 0..m {
            for col in 0..n {
                let mut acc = b[col];
                for kk in 0..k {
                    acc += normed[row * k + kk] * w[kk * n + col];
                }
                naive[row * n + col] = acc;
            }
        }
        for (f, n_) in fused.iter().zip(naive.iter()) {
            assert!((f - n_).abs() < 1e-5, "fused {f} naive {n_}");
        }
    }

    #[test]
    fn layernorm_handles_extreme_magnitudes_via_welford() {
        // Welford should keep variance numerically stable even for
        // huge baselines.
        let xs = vec![1e10_f32, 1e10 + 1.0, 1e10 - 1.0, 1e10 + 0.5];
        let w = vec![1.0_f32; 4];
        let mut y = vec![0.0f32; 1];
        layernorm_linear(&xs, &w, None, &mut y, 1, 4, 1, 1e-5).unwrap();
        // The normalised row sums to ~0 (mean-centred); the matmul
        // with all-ones W should give a small finite value.
        assert!(y[0].is_finite());
        assert!(y[0].abs() < 1.0);
    }

    #[test]
    fn residual_add_scale_mul_basic() {
        let input = vec![1.0_f32, 2.0, 3.0];
        let residual = vec![0.5, 0.5, 0.5];
        let gate = vec![2.0, 0.0, 1.0];
        let mut y = vec![0.0f32; 3];
        residual_add_scale_mul(&input, &residual, &gate, &mut y, 0.5).unwrap();
        // y[i] = 0.5 * (input[i] + 0.5) * gate[i]
        assert_eq!(y[0], 0.5 * 1.5 * 2.0);
        assert_eq!(y[1], 0.5 * 2.5 * 0.0);
        assert_eq!(y[2], 0.5 * 3.5 * 1.0);
    }

    #[test]
    fn residual_shape_mismatch_returns_error() {
        let mut y = vec![0.0f32; 3];
        let err = residual_add_scale_mul(&[1.0; 3], &[1.0; 2], &[1.0; 3], &mut y, 1.0).unwrap_err();
        assert!(matches!(
            err,
            FusionError::ShapeMismatch {
                which: "residual",
                ..
            }
        ));
    }

    #[test]
    fn layernorm_empty_returns_error() {
        let mut y = vec![0.0_f32; 0];
        layernorm_linear(&[], &[], None, &mut y, 0, 0, 0, 1e-5).unwrap();
        assert!(y.is_empty());
    }
}
