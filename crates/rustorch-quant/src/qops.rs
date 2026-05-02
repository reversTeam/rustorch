//! Quantize / dequantize ops.
//!
//! `quantize(x, qp)` rounds each element to the nearest int8 and
//! shifts by `zero_point`. The rounding mode is **HALF_TO_EVEN**
//! (banker's rounding), matching IEEE-754 default — bit-exact
//! reproducibility across platforms and parity with PyTorch's
//! `torch.quantize_per_tensor`.
//!
//! `dequantize(q, qp)` inverts: `f = (q - zero_point) * scale`.
//!
//! Round-trip property: for any `x` in `[zp_min, zp_max] * scale`,
//! `dequantize(quantize(x, qp), qp)` is within `scale / 2` of `x`.

use crate::dtype::QParams;

/// Errors raised by quantize / dequantize.
#[derive(Debug, Clone, PartialEq)]
pub enum QError {
    /// Output buffer length doesn't match input.
    LengthMismatch {
        /// Input length.
        expected: usize,
        /// Output length.
        got: usize,
    },
    /// `qp.scale` is non-positive — the scale must be > 0 to map
    /// int8 ↔ float meaningfully.
    NonPositiveScale {
        /// Offending scale value.
        scale: f32,
    },
}

impl core::fmt::Display for QError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            QError::LengthMismatch { expected, got } => {
                write!(f, "buffer length mismatch: expected {expected}, got {got}")
            },
            QError::NonPositiveScale { scale } => {
                write!(f, "qparams.scale must be > 0, got {scale}")
            },
        }
    }
}

impl std::error::Error for QError {}

/// HALF_TO_EVEN rounding for f32 → i32. Matches IEEE-754 round-to-
/// nearest-even (Banker's rounding).
#[inline]
fn round_half_to_even(x: f32) -> i32 {
    // Rust's f32::round_ties_even is unstable, so do it manually.
    let floor = x.floor();
    let diff = x - floor;
    if diff < 0.5 {
        floor as i32
    } else if diff > 0.5 {
        (floor + 1.0) as i32
    } else {
        // Exactly halfway → round to nearest even.
        let f_int = floor as i32;
        if f_int % 2 == 0 {
            f_int
        } else {
            f_int + 1
        }
    }
}

/// Quantize a float buffer into int8 according to `qp`. Output is
/// clamped to `[-128, 127]`. NaN inputs yield 0 (centre of the int8
/// range — least information loss for an unknown).
pub fn quantize(input: &[f32], output: &mut [i8], qp: QParams) -> Result<(), QError> {
    if input.len() != output.len() {
        return Err(QError::LengthMismatch {
            expected: input.len(),
            got: output.len(),
        });
    }
    if qp.scale <= 0.0 || !qp.scale.is_finite() {
        return Err(QError::NonPositiveScale { scale: qp.scale });
    }
    for (i, &x) in input.iter().enumerate() {
        if !x.is_finite() {
            output[i] = 0;
            continue;
        }
        let scaled = x / qp.scale + qp.zero_point as f32;
        let rounded = round_half_to_even(scaled);
        output[i] = rounded.clamp(-128, 127) as i8;
    }
    Ok(())
}

/// Dequantize an int8 buffer back to float using `qp`.
pub fn dequantize(input: &[i8], output: &mut [f32], qp: QParams) -> Result<(), QError> {
    if input.len() != output.len() {
        return Err(QError::LengthMismatch {
            expected: input.len(),
            got: output.len(),
        });
    }
    if qp.scale <= 0.0 || !qp.scale.is_finite() {
        return Err(QError::NonPositiveScale { scale: qp.scale });
    }
    for (i, &q) in input.iter().enumerate() {
        output[i] = (q as i32 - qp.zero_point) as f32 * qp.scale;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_half_to_even_works() {
        assert_eq!(round_half_to_even(0.5), 0);
        assert_eq!(round_half_to_even(1.5), 2);
        assert_eq!(round_half_to_even(2.5), 2);
        assert_eq!(round_half_to_even(3.5), 4);
        assert_eq!(round_half_to_even(-0.5), 0);
        assert_eq!(round_half_to_even(-1.5), -2);
        assert_eq!(round_half_to_even(0.4999), 0);
        assert_eq!(round_half_to_even(0.5001), 1);
    }

    #[test]
    fn round_trip_within_scale() {
        let qp = QParams::from_min_max(-2.0, 2.0).unwrap();
        let xs: Vec<f32> = (-50..=50).map(|i| i as f32 * 0.04).collect();
        let mut q = vec![0i8; xs.len()];
        let mut back = vec![0.0f32; xs.len()];
        quantize(&xs, &mut q, qp).unwrap();
        dequantize(&q, &mut back, qp).unwrap();
        for (x, r) in xs.iter().zip(back.iter()) {
            assert!(
                (x - r).abs() <= qp.scale,
                "x={} r={} diff={} scale={}",
                x,
                r,
                (x - r).abs(),
                qp.scale
            );
        }
    }

    #[test]
    fn quantize_clamps_out_of_range() {
        let qp = QParams {
            scale: 1.0,
            zero_point: 0,
        };
        let xs = [-1000.0, -129.0, -128.0, 127.0, 128.0, 1000.0];
        let mut q = vec![0i8; xs.len()];
        quantize(&xs, &mut q, qp).unwrap();
        assert_eq!(q, vec![-128, -128, -128, 127, 127, 127]);
    }

    #[test]
    fn quantize_nan_yields_zero() {
        let qp = QParams::IDENTITY;
        let xs = [1.0, f32::NAN, 2.0, f32::INFINITY, -1.0];
        let mut q = vec![0i8; xs.len()];
        quantize(&xs, &mut q, qp).unwrap();
        assert_eq!(q[1], 0);
        assert_eq!(q[3], 0);
        assert_eq!(q[0], 1);
        assert_eq!(q[4], -1);
    }

    #[test]
    fn empty_buffer_is_no_op() {
        let qp = QParams::IDENTITY;
        let xs: Vec<f32> = Vec::new();
        let mut q: Vec<i8> = Vec::new();
        quantize(&xs, &mut q, qp).unwrap();
        let mut back: Vec<f32> = Vec::new();
        dequantize(&q, &mut back, qp).unwrap();
        assert!(back.is_empty());
    }

    #[test]
    fn length_mismatch_returns_error() {
        let qp = QParams::IDENTITY;
        let xs = [1.0; 4];
        let mut q = vec![0i8; 3];
        assert_eq!(
            quantize(&xs, &mut q, qp).unwrap_err(),
            QError::LengthMismatch {
                expected: 4,
                got: 3,
            }
        );
    }

    #[test]
    fn non_positive_scale_returns_error() {
        let xs = [1.0];
        let mut q = vec![0i8; 1];
        let bad = QParams {
            scale: 0.0,
            zero_point: 0,
        };
        assert!(matches!(
            quantize(&xs, &mut q, bad).unwrap_err(),
            QError::NonPositiveScale { .. }
        ));
        let neg = QParams {
            scale: -1.0,
            zero_point: 0,
        };
        assert!(matches!(
            quantize(&xs, &mut q, neg).unwrap_err(),
            QError::NonPositiveScale { .. }
        ));
    }

    #[test]
    fn dequantize_with_zero_point_offset() {
        let qp = QParams {
            scale: 0.5,
            zero_point: -64,
        };
        // q = 0 → (0 - (-64)) * 0.5 = 32.0
        let q = [0i8, -64, 63];
        let mut back = vec![0.0f32; q.len()];
        dequantize(&q, &mut back, qp).unwrap();
        assert_eq!(back[0], 32.0);
        assert_eq!(back[1], 0.0);
        assert_eq!(back[2], 63.5);
    }

    #[test]
    fn quantize_preserves_zero_exactly() {
        let qp = QParams {
            scale: 0.123456,
            zero_point: 0,
        };
        let xs = [0.0f32];
        let mut q = vec![0i8; 1];
        quantize(&xs, &mut q, qp).unwrap();
        assert_eq!(q[0], 0);
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig {
            cases: 1000,
            .. ProptestConfig::default()
        })]

        /// `dequantize(quantize(x, qp), qp)` is within `scale` of `x`
        /// for any `x` in the representable range.
        #[test]
        fn round_trip_within_scale_random(
            values in prop::collection::vec(-100.0f32..100.0, 1..200)
        ) {
            // Choose qp from the values' range.
            let mn = values.iter().copied().fold(f32::INFINITY, f32::min);
            let mx = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            if let Some(qp) = QParams::from_min_max(mn, mx) {
                let mut q = vec![0i8; values.len()];
                let mut back = vec![0.0f32; values.len()];
                quantize(&values, &mut q, qp).unwrap();
                dequantize(&q, &mut back, qp).unwrap();
                for (x, r) in values.iter().zip(back.iter()) {
                    prop_assert!(
                        (x - r).abs() <= qp.scale + 1e-6,
                        "x={} r={} diff={} scale={}",
                        x, r, (x - r).abs(), qp.scale
                    );
                }
            }
        }

        /// Quantize is idempotent on its output range: re-quantising
        /// the dequantised values yields the same int8 buffer.
        #[test]
        fn quantize_idempotent_on_dequantised(
            values in prop::collection::vec(-100.0f32..100.0, 1..200)
        ) {
            let mn = values.iter().copied().fold(f32::INFINITY, f32::min);
            let mx = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            if let Some(qp) = QParams::from_min_max(mn, mx) {
                let mut q1 = vec![0i8; values.len()];
                let mut back = vec![0.0f32; values.len()];
                let mut q2 = vec![0i8; values.len()];
                quantize(&values, &mut q1, qp).unwrap();
                dequantize(&q1, &mut back, qp).unwrap();
                quantize(&back, &mut q2, qp).unwrap();
                prop_assert_eq!(q1, q2);
            }
        }
    }
}
