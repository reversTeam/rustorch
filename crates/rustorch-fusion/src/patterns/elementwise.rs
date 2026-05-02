//! Elementwise epilogue chains — fuse N elementwise unary/binary ops
//! into a single SIMD loop over the output buffer.

use crate::patterns::matmul_bias_act::FusionError;

/// One elementwise step in a chain.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EwOp {
    /// out = out + scalar.
    AddScalar(f32),
    /// out = out * scalar.
    MulScalar(f32),
    /// out = max(0, out).
    Relu,
    /// out = sigmoid(out).
    Sigmoid,
    /// out = tanh(out).
    Tanh,
    /// out = out.abs().
    Abs,
    /// out = sqrt(out).
    Sqrt,
}

impl EwOp {
    /// Apply this op to a single scalar.
    #[inline]
    pub fn apply(self, x: f32) -> f32 {
        match self {
            EwOp::AddScalar(c) => x + c,
            EwOp::MulScalar(c) => x * c,
            EwOp::Relu => x.max(0.0),
            EwOp::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            EwOp::Tanh => x.tanh(),
            EwOp::Abs => x.abs(),
            EwOp::Sqrt => x.sqrt(),
        }
    }
}

/// Apply a chain of elementwise ops to `input`, writing to `output`
/// in a SINGLE pass. The naive implementation would do one full
/// pass per op (allocating intermediate buffers); fusing into a
/// single loop keeps the active values in registers and amortises
/// the memory traffic.
pub fn elementwise_chain(
    input: &[f32],
    output: &mut [f32],
    chain: &[EwOp],
) -> Result<(), FusionError> {
    if input.len() != output.len() {
        return Err(FusionError::ShapeMismatch {
            which: "output",
            expected: input.len(),
            got: output.len(),
        });
    }
    for (i, &x) in input.iter().enumerate() {
        let mut v = x;
        for op in chain {
            v = op.apply(v);
        }
        output[i] = v;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chain_of_2_ops_matches_sequential() {
        let xs = vec![-2.0f32, -1.0, 0.0, 1.0, 2.0];
        let mut fused = vec![0.0f32; xs.len()];
        elementwise_chain(&xs, &mut fused, &[EwOp::AddScalar(1.0), EwOp::Relu]).unwrap();
        let expected: Vec<f32> = xs.iter().map(|x| (x + 1.0).max(0.0)).collect();
        for (f, e) in fused.iter().zip(expected.iter()) {
            assert_eq!(*f, *e);
        }
    }

    #[test]
    fn chain_of_4_ops_matches_naive() {
        let xs = vec![1.0f32, 4.0, 9.0, 16.0];
        let chain = [
            EwOp::Sqrt,
            EwOp::AddScalar(1.0),
            EwOp::MulScalar(2.0),
            EwOp::Relu,
        ];
        let mut fused = vec![0.0f32; xs.len()];
        elementwise_chain(&xs, &mut fused, &chain).unwrap();
        let expected: Vec<f32> = xs
            .iter()
            .map(|x| ((x.sqrt() + 1.0) * 2.0).max(0.0))
            .collect();
        for (f, e) in fused.iter().zip(expected.iter()) {
            assert!((f - e).abs() < 1e-6);
        }
    }

    #[test]
    fn empty_input_returns_empty_output() {
        let mut out: Vec<f32> = Vec::new();
        elementwise_chain(&[], &mut out, &[EwOp::Relu]).unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn empty_chain_is_identity() {
        let xs = vec![1.0f32, -2.0, 3.0];
        let mut out = vec![0.0f32; xs.len()];
        elementwise_chain(&xs, &mut out, &[]).unwrap();
        assert_eq!(out, xs);
    }

    #[test]
    fn shape_mismatch_returns_error() {
        let mut out = vec![0.0f32; 3];
        let err = elementwise_chain(&[1.0; 4], &mut out, &[EwOp::Relu]).unwrap_err();
        assert!(matches!(err, FusionError::ShapeMismatch { .. }));
    }

    #[test]
    fn nan_propagates_through_chain_when_op_propagates_nan() {
        // GOTCHA: Rust's `f32::max(NaN, 0.0)` returns 0.0 (the non-NaN
        // arg), so a Relu(NaN) yields 0.0 — NaN gets MASKED. Use an
        // op that propagates (Sigmoid: exp(-NaN) = NaN, 1/(1+NaN) = NaN).
        let xs = vec![1.0f32, f32::NAN, -1.0];
        let mut out = vec![0.0f32; 3];
        elementwise_chain(&xs, &mut out, &[EwOp::Sigmoid, EwOp::AddScalar(1.0)]).unwrap();
        assert!(out[1].is_nan());
    }

    #[test]
    fn relu_masks_nan_to_zero_documented() {
        // Document the f32::max behaviour for future maintainers.
        let xs = vec![f32::NAN];
        let mut out = vec![0.0f32; 1];
        elementwise_chain(&xs, &mut out, &[EwOp::Relu]).unwrap();
        assert_eq!(out[0], 0.0);
    }

    #[test]
    fn negative_sqrt_yields_nan() {
        let xs = vec![-1.0f32];
        let mut out = vec![0.0f32; 1];
        elementwise_chain(&xs, &mut out, &[EwOp::Sqrt]).unwrap();
        assert!(out[0].is_nan());
    }
}
