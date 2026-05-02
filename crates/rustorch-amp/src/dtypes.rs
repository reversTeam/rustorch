//! f32 ↔ bf16 / fp16 conversions via the `half` crate.
//!
//! `half::bf16` and `half::f16` already implement IEEE-754 round-to-
//! nearest-even truncation and preserve NaN/Inf/-0. This module wraps
//! batch conversions for vectors of f32 ↔ low-precision, which is
//! the operation actually wanted by the autocast layer.

use half::{bf16, f16};

/// Convert an f32 slice to bf16 (round-to-nearest-even). Output buffer
/// must have the same length as input.
pub fn f32_to_bf16(input: &[f32], output: &mut [bf16]) -> Result<(), DtypeError> {
    if input.len() != output.len() {
        return Err(DtypeError::LengthMismatch {
            expected: input.len(),
            got: output.len(),
        });
    }
    for (i, &x) in input.iter().enumerate() {
        output[i] = bf16::from_f32(x);
    }
    Ok(())
}

/// Convert a bf16 slice back to f32. Lossless (bf16 → f32 just zero-
/// extends the mantissa).
pub fn bf16_to_f32(input: &[bf16], output: &mut [f32]) -> Result<(), DtypeError> {
    if input.len() != output.len() {
        return Err(DtypeError::LengthMismatch {
            expected: input.len(),
            got: output.len(),
        });
    }
    for (i, &x) in input.iter().enumerate() {
        output[i] = x.to_f32();
    }
    Ok(())
}

/// Convert an f32 slice to fp16 (IEEE 754 binary16, round-to-nearest-
/// even). Saturates to ±∞ on overflow (range ±65504).
pub fn f32_to_fp16(input: &[f32], output: &mut [f16]) -> Result<(), DtypeError> {
    if input.len() != output.len() {
        return Err(DtypeError::LengthMismatch {
            expected: input.len(),
            got: output.len(),
        });
    }
    for (i, &x) in input.iter().enumerate() {
        output[i] = f16::from_f32(x);
    }
    Ok(())
}

/// Convert an fp16 slice back to f32. Lossless within fp16's range.
pub fn fp16_to_f32(input: &[f16], output: &mut [f32]) -> Result<(), DtypeError> {
    if input.len() != output.len() {
        return Err(DtypeError::LengthMismatch {
            expected: input.len(),
            got: output.len(),
        });
    }
    for (i, &x) in input.iter().enumerate() {
        output[i] = x.to_f32();
    }
    Ok(())
}

/// Errors raised by dtype conversions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DtypeError {
    /// Input and output buffer lengths differ.
    LengthMismatch {
        /// Input length.
        expected: usize,
        /// Output length.
        got: usize,
    },
}

impl core::fmt::Display for DtypeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            DtypeError::LengthMismatch { expected, got } => {
                write!(f, "length mismatch: expected {expected}, got {got}")
            },
        }
    }
}

impl std::error::Error for DtypeError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bf16_round_trip_preserves_top_8_mantissa_bits() {
        let xs = vec![1.0f32, -2.5, 0.125, 1.5e-3];
        let mut bf = vec![bf16::ZERO; xs.len()];
        let mut back = vec![0.0f32; xs.len()];
        f32_to_bf16(&xs, &mut bf).unwrap();
        bf16_to_f32(&bf, &mut back).unwrap();
        for (x, r) in xs.iter().zip(back.iter()) {
            // bf16 has 8 mantissa bits → relative error ≤ 2^-8 ≈ 0.4%
            let rel_err = (x - r).abs() / x.abs().max(1e-6);
            assert!(rel_err < 1.0 / 256.0, "x={x} r={r}");
        }
    }

    #[test]
    fn bf16_preserves_nan_inf_neg_zero() {
        let xs = vec![f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -0.0f32];
        let mut bf = vec![bf16::ZERO; xs.len()];
        let mut back = vec![0.0f32; xs.len()];
        f32_to_bf16(&xs, &mut bf).unwrap();
        bf16_to_f32(&bf, &mut back).unwrap();
        assert!(back[0].is_nan());
        assert_eq!(back[1], f32::INFINITY);
        assert_eq!(back[2], f32::NEG_INFINITY);
        assert_eq!(back[3].to_bits(), (-0.0f32).to_bits());
    }

    #[test]
    fn fp16_round_trip_preserves_11_bit_mantissa() {
        let xs = vec![1.0f32, -2.5, 0.125, 1.5e-3];
        let mut fp = vec![f16::ZERO; xs.len()];
        let mut back = vec![0.0f32; xs.len()];
        f32_to_fp16(&xs, &mut fp).unwrap();
        fp16_to_f32(&fp, &mut back).unwrap();
        for (x, r) in xs.iter().zip(back.iter()) {
            // fp16 has 11 mantissa bits → relative error ≤ 2^-11 ≈ 0.05%
            let rel_err = (x - r).abs() / x.abs().max(1e-6);
            assert!(rel_err < 1.0 / 2048.0, "x={x} r={r}");
        }
    }

    #[test]
    fn fp16_overflow_saturates_to_inf() {
        let xs = vec![70_000.0f32]; // > fp16 max 65504
        let mut fp = vec![f16::ZERO; 1];
        f32_to_fp16(&xs, &mut fp).unwrap();
        assert_eq!(fp[0], f16::INFINITY);
    }

    #[test]
    fn empty_buffer_is_no_op() {
        let mut bf: Vec<bf16> = Vec::new();
        f32_to_bf16(&[], &mut bf).unwrap();
        assert!(bf.is_empty());
    }

    #[test]
    fn length_mismatch_returns_error() {
        let mut bf = vec![bf16::ZERO; 3];
        let err = f32_to_bf16(&[1.0; 4], &mut bf).unwrap_err();
        assert!(matches!(
            err,
            DtypeError::LengthMismatch {
                expected: 4,
                got: 3
            }
        ));
    }
}
