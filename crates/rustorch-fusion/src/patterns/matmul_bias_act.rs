//! Fused matmul+bias+activation kernels.
//!
//! Computes `y = activation(x @ w + b)` in a SINGLE pass over the
//! output buffer:
//!
//!   for each (m, n):
//!     acc = bias[n]
//!     for k in 0..K: acc += x[m, k] * w[k, n]
//!     y[m, n] = activation(acc)
//!
//! Compared to the naive sequential pipeline (matmul → bias add →
//! activation), the fused version:
//! 1. Avoids two extra passes over the M×N output buffer.
//! 2. Keeps the accumulator in a register / cache line until the
//!    activation is applied — no round-trip to memory.
//!
//! Currently scalar with f32 accumulator; LLVM auto-vectorises the
//! inner-K loop. SIMD intrinsics are a future arch-specific
//! specialisation.

use std::cmp::Ordering;

/// Activation choices for the fused epilogue.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Activation {
    /// Identity (no activation).
    None,
    /// max(0, x).
    Relu,
    /// 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³))) — tanh
    /// approximation, common for transformer FFNs.
    Gelu,
    /// x * sigmoid(x).
    Silu,
}

impl Activation {
    /// Apply the activation to a single scalar.
    #[inline]
    pub fn apply(self, x: f32) -> f32 {
        match self {
            Activation::None => x,
            Activation::Relu => x.max(0.0),
            Activation::Gelu => {
                let inner = (2.0_f32 / std::f32::consts::PI).sqrt() * (x + 0.044715 * x * x * x);
                0.5 * x * (1.0 + inner.tanh())
            },
            Activation::Silu => x * sigmoid(x),
        }
    }
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Errors raised by the fused kernel.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FusionError {
    /// Buffer length doesn't match `(m, k, n)`.
    ShapeMismatch {
        /// Which buffer.
        which: &'static str,
        /// Expected element count.
        expected: usize,
        /// Actual count.
        got: usize,
    },
}

impl core::fmt::Display for FusionError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            FusionError::ShapeMismatch {
                which,
                expected,
                got,
            } => write!(f, "{which} buffer has {got} elements, expected {expected}"),
        }
    }
}

impl std::error::Error for FusionError {}

/// Fused `y = activation(x @ w + b)` in a single pass.
///
/// - `x` is `[m, k]` row-major
/// - `w` is `[k, n]` row-major
/// - `b` is `[n]` (broadcast over batch m)
/// - `y` is `[m, n]` row-major
#[allow(clippy::too_many_arguments)]
#[allow(clippy::needless_range_loop)]
pub fn fused_matmul_bias_activation(
    x: &[f32],
    w: &[f32],
    b: Option<&[f32]>,
    y: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    activation: Activation,
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
    if let Some(bias) = b {
        if bias.len() != n {
            return Err(FusionError::ShapeMismatch {
                which: "b",
                expected: n,
                got: bias.len(),
            });
        }
    }

    for row in 0..m {
        for col in 0..n {
            let mut acc = b.map(|bb| bb[col]).unwrap_or(0.0);
            for kk in 0..k {
                acc += x[row * k + kk] * w[kk * n + col];
            }
            y[row * n + col] = activation.apply(acc);
        }
    }
    Ok(())
}

/// Naive reference (matmul → bias add → activation as 3 separate
/// passes). Used by tests + benches as the non-fused baseline.
#[allow(clippy::too_many_arguments)]
#[allow(clippy::needless_range_loop)]
pub fn naive_matmul_bias_activation(
    x: &[f32],
    w: &[f32],
    b: Option<&[f32]>,
    y: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    activation: Activation,
) -> Result<(), FusionError> {
    // matmul into y
    for row in 0..m {
        for col in 0..n {
            let mut acc = 0.0f32;
            for kk in 0..k {
                acc += x[row * k + kk] * w[kk * n + col];
            }
            y[row * n + col] = acc;
        }
    }
    // bias add (separate pass)
    if let Some(bias) = b {
        for row in 0..m {
            for col in 0..n {
                y[row * n + col] += bias[col];
            }
        }
    }
    // activation (separate pass)
    for cell in y.iter_mut() {
        *cell = activation.apply(*cell);
    }
    let _ = (x, w, k, Ordering::Equal); // silence unused warnings on reorderings
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f32, b: f32, atol: f32) -> bool {
        (a - b).abs() <= atol + 1e-5 * a.abs().max(b.abs())
    }

    fn fixture(m: usize, k: usize, n: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let x: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.1).sin()).collect();
        let w: Vec<f32> = (0..k * n).map(|i| (i as f32 * 0.07).cos()).collect();
        let b: Vec<f32> = (0..n).map(|i| i as f32 * 0.05).collect();
        (x, w, b)
    }

    #[test]
    fn fused_matches_naive_relu() {
        let (m, k, n) = (4, 8, 6);
        let (x, w, b) = fixture(m, k, n);
        let mut y_f = vec![0.0f32; m * n];
        let mut y_n = vec![0.0f32; m * n];
        fused_matmul_bias_activation(&x, &w, Some(&b), &mut y_f, m, k, n, Activation::Relu)
            .unwrap();
        naive_matmul_bias_activation(&x, &w, Some(&b), &mut y_n, m, k, n, Activation::Relu)
            .unwrap();
        for (a, b) in y_f.iter().zip(y_n.iter()) {
            assert!(close(*a, *b, 1e-5), "fused {a} naive {b}");
        }
    }

    #[test]
    fn fused_matches_naive_gelu() {
        let (m, k, n) = (3, 16, 4);
        let (x, w, b) = fixture(m, k, n);
        let mut y_f = vec![0.0f32; m * n];
        let mut y_n = vec![0.0f32; m * n];
        fused_matmul_bias_activation(&x, &w, Some(&b), &mut y_f, m, k, n, Activation::Gelu)
            .unwrap();
        naive_matmul_bias_activation(&x, &w, Some(&b), &mut y_n, m, k, n, Activation::Gelu)
            .unwrap();
        for (a, b) in y_f.iter().zip(y_n.iter()) {
            assert!(close(*a, *b, 1e-5), "fused {a} naive {b}");
        }
    }

    #[test]
    fn fused_matches_naive_silu() {
        let (m, k, n) = (2, 4, 3);
        let (x, w, b) = fixture(m, k, n);
        let mut y_f = vec![0.0f32; m * n];
        let mut y_n = vec![0.0f32; m * n];
        fused_matmul_bias_activation(&x, &w, Some(&b), &mut y_f, m, k, n, Activation::Silu)
            .unwrap();
        naive_matmul_bias_activation(&x, &w, Some(&b), &mut y_n, m, k, n, Activation::Silu)
            .unwrap();
        for (a, b) in y_f.iter().zip(y_n.iter()) {
            assert!(close(*a, *b, 1e-5));
        }
    }

    #[test]
    fn no_bias_is_supported() {
        let (m, k, n) = (2, 4, 2);
        let (x, w, _) = fixture(m, k, n);
        let mut y_f = vec![0.0f32; m * n];
        fused_matmul_bias_activation(&x, &w, None, &mut y_f, m, k, n, Activation::None).unwrap();
        // Spot-check: y should not be all zeros.
        assert!(y_f.iter().any(|&v| v != 0.0));
    }

    #[test]
    fn shape_mismatch_returns_error() {
        let mut y = vec![0.0f32; 4];
        let err = fused_matmul_bias_activation(
            &[1.0; 6],
            &[1.0; 8],
            None,
            &mut y,
            2,
            4,
            2,
            Activation::None,
        )
        .unwrap_err();
        assert!(matches!(err, FusionError::ShapeMismatch { which: "x", .. }));
    }

    #[test]
    fn empty_input_returns_empty_output() {
        let mut y: Vec<f32> = Vec::new();
        fused_matmul_bias_activation(&[], &[], None, &mut y, 0, 0, 0, Activation::Relu).unwrap();
        assert!(y.is_empty());
    }

    #[test]
    fn nan_input_propagates_through_non_relu_activation() {
        // Relu uses f32::max which MASKS NaN to 0.0 — use None
        // (identity) for genuine NaN propagation. Shape: m=2, k=2, n=2.
        let mut x = vec![0.0f32; 4];
        x[0] = f32::NAN;
        let w = vec![1.0f32; 4];
        let mut y = vec![0.0f32; 4];
        fused_matmul_bias_activation(&x, &w, None, &mut y, 2, 2, 2, Activation::None).unwrap();
        // Row 0 cells touch x[0] (NaN) → should be NaN.
        assert!(y[0].is_nan());
        assert!(y[1].is_nan());
    }

    #[test]
    fn relu_masks_nan_documented() {
        // Document for future maintainers: f32::max(NaN, 0) = 0,
        // so Relu silently sanitises NaN. Use Gelu/Silu/None if you
        // want NaN to flow through.
        let x = vec![f32::NAN, 0.0];
        let w = vec![1.0f32; 2];
        let mut y = vec![0.0f32; 1];
        fused_matmul_bias_activation(&x, &w, None, &mut y, 1, 2, 1, Activation::Relu).unwrap();
        assert_eq!(y[0], 0.0);
    }
}
