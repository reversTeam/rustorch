//! Module-level cast helpers: convert all parameters in a parameter
//! list from f32 to bf16/fp16 (and back).
//!
//! This module operates on a flat `Vec<&mut [f32]>` view of a model's
//! parameters — the bridge from `rustorch_nn::Module::parameters()`
//! to this layer is a 3-line adapter at the call site (extract
//! `.tensor()` data slices). Keeps rustorch-amp decoupled from
//! rustorch-nn's autograd-aware Variable wrapping.

use crate::dtypes::{bf16_to_f32, f32_to_bf16, f32_to_fp16, fp16_to_f32, DtypeError};
use half::{bf16, f16};

/// Convert every parameter slice from f32 to bf16. Returns the bf16
/// buffers paralleling the input slice list. The original f32
/// buffers are NOT mutated — this is a clone-cast.
pub fn parameters_to_bf16(params: &[&[f32]]) -> Result<Vec<Vec<bf16>>, DtypeError> {
    params
        .iter()
        .map(|p| {
            let mut out = vec![bf16::ZERO; p.len()];
            f32_to_bf16(p, &mut out)?;
            Ok(out)
        })
        .collect()
}

/// Convert every parameter slice from f32 to fp16.
pub fn parameters_to_fp16(params: &[&[f32]]) -> Result<Vec<Vec<f16>>, DtypeError> {
    params
        .iter()
        .map(|p| {
            let mut out = vec![f16::ZERO; p.len()];
            f32_to_fp16(p, &mut out)?;
            Ok(out)
        })
        .collect()
}

/// Inverse of [`parameters_to_bf16`].
pub fn parameters_from_bf16(params: &[&[bf16]]) -> Result<Vec<Vec<f32>>, DtypeError> {
    params
        .iter()
        .map(|p| {
            let mut out = vec![0.0f32; p.len()];
            bf16_to_f32(p, &mut out)?;
            Ok(out)
        })
        .collect()
}

/// Inverse of [`parameters_to_fp16`].
pub fn parameters_from_fp16(params: &[&[f16]]) -> Result<Vec<Vec<f32>>, DtypeError> {
    params
        .iter()
        .map(|p| {
            let mut out = vec![0.0f32; p.len()];
            fp16_to_f32(p, &mut out)?;
            Ok(out)
        })
        .collect()
}

/// Total bytes the f32 parameter list occupies.
pub fn f32_param_bytes(params: &[&[f32]]) -> usize {
    params.iter().map(|p| p.len() * 4).sum()
}

/// Total bytes the equivalent bf16 / fp16 parameter list occupies
/// (both are 2 bytes per element).
pub fn low_prec_param_bytes(params: &[&[f32]]) -> usize {
    params.iter().map(|p| p.len() * 2).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> (Vec<f32>, Vec<f32>) {
        let p1: Vec<f32> = (0..16).map(|i| i as f32 * 0.1).collect();
        let p2: Vec<f32> = (0..32).map(|i| -(i as f32) * 0.05).collect();
        (p1, p2)
    }

    #[test]
    fn parameters_to_bf16_preserves_count_and_lengths() {
        let (p1, p2) = fixture();
        let params = vec![p1.as_slice(), p2.as_slice()];
        let bf = parameters_to_bf16(&params).unwrap();
        assert_eq!(bf.len(), 2);
        assert_eq!(bf[0].len(), 16);
        assert_eq!(bf[1].len(), 32);
    }

    #[test]
    fn bf16_round_trip_within_bf16_epsilon() {
        let (p1, _) = fixture();
        let params = vec![p1.as_slice()];
        let bf = parameters_to_bf16(&params).unwrap();
        let bf_refs: Vec<&[bf16]> = bf.iter().map(|v| v.as_slice()).collect();
        let back = parameters_from_bf16(&bf_refs).unwrap();
        for (orig, recovered) in p1.iter().zip(back[0].iter()) {
            let rel = (orig - recovered).abs() / orig.abs().max(1e-6);
            assert!(rel < 1.0 / 256.0, "orig {orig} got {recovered}");
        }
    }

    #[test]
    fn fp16_round_trip_within_fp16_epsilon() {
        let (p1, _) = fixture();
        let params = vec![p1.as_slice()];
        let fp = parameters_to_fp16(&params).unwrap();
        let fp_refs: Vec<&[f16]> = fp.iter().map(|v| v.as_slice()).collect();
        let back = parameters_from_fp16(&fp_refs).unwrap();
        for (orig, recovered) in p1.iter().zip(back[0].iter()) {
            let rel = (orig - recovered).abs() / orig.abs().max(1e-6);
            assert!(rel < 1.0 / 2048.0, "orig {orig} got {recovered}");
        }
    }

    #[test]
    fn byte_counts_show_2x_savings() {
        let (p1, p2) = fixture();
        let params = vec![p1.as_slice(), p2.as_slice()];
        let f32_bytes = f32_param_bytes(&params);
        let low_bytes = low_prec_param_bytes(&params);
        assert_eq!(f32_bytes, (16 + 32) * 4);
        assert_eq!(low_bytes, (16 + 32) * 2);
        assert_eq!(f32_bytes / low_bytes, 2);
    }

    #[test]
    fn empty_parameter_list_works() {
        let params: Vec<&[f32]> = Vec::new();
        let bf = parameters_to_bf16(&params).unwrap();
        assert!(bf.is_empty());
    }
}
