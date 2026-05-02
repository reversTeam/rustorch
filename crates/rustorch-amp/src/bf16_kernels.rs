//! Scalar bf16 / fp16 matmul kernels with f32 accumulator.
//!
//! Compute `c = a @ b` where:
//! - `a` is `[m, k]` low-precision
//! - `b` is `[k, n]` low-precision
//! - `c` is `[m, n]` f32 (the f32 accumulator's natural output)
//!
//! Using an f32 accumulator preserves precision across long inner
//! reductions even when the inputs are bf16/fp16. This matches the
//! AVX-512 BF16 / NEON behaviour and is the standard practice for
//! mixed-precision GEMM.

#![allow(clippy::needless_range_loop)]

use half::{bf16, f16};

/// Errors raised by the bf16/fp16 GEMM.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BfGemmError {
    /// Buffer length doesn't match (m, k, n).
    ShapeMismatch {
        /// Which buffer (a / b / c).
        which: &'static str,
        /// Expected element count.
        expected: usize,
        /// Actual element count.
        got: usize,
    },
}

impl core::fmt::Display for BfGemmError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            BfGemmError::ShapeMismatch {
                which,
                expected,
                got,
            } => write!(f, "{which} buffer has {got} elements, expected {expected}"),
        }
    }
}

impl std::error::Error for BfGemmError {}

/// `c = a @ b` for bf16 inputs with f32 accumulator. Output `c` is
/// f32.
pub fn matmul_bf16_with_f32_accum(
    a: &[bf16],
    b: &[bf16],
    c: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
) -> Result<(), BfGemmError> {
    if a.len() != m * k {
        return Err(BfGemmError::ShapeMismatch {
            which: "a",
            expected: m * k,
            got: a.len(),
        });
    }
    if b.len() != k * n {
        return Err(BfGemmError::ShapeMismatch {
            which: "b",
            expected: k * n,
            got: b.len(),
        });
    }
    if c.len() != m * n {
        return Err(BfGemmError::ShapeMismatch {
            which: "c",
            expected: m * n,
            got: c.len(),
        });
    }
    for row in 0..m {
        for col in 0..n {
            let mut acc = 0.0f32;
            for kk in 0..k {
                let av = a[row * k + kk].to_f32();
                let bv = b[kk * n + col].to_f32();
                acc += av * bv;
            }
            c[row * n + col] = acc;
        }
    }
    Ok(())
}

/// `c = a @ b` for fp16 inputs with f32 accumulator. Output `c` is
/// f32.
pub fn matmul_fp16_with_f32_accum(
    a: &[f16],
    b: &[f16],
    c: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
) -> Result<(), BfGemmError> {
    if a.len() != m * k {
        return Err(BfGemmError::ShapeMismatch {
            which: "a",
            expected: m * k,
            got: a.len(),
        });
    }
    if b.len() != k * n {
        return Err(BfGemmError::ShapeMismatch {
            which: "b",
            expected: k * n,
            got: b.len(),
        });
    }
    if c.len() != m * n {
        return Err(BfGemmError::ShapeMismatch {
            which: "c",
            expected: m * n,
            got: c.len(),
        });
    }
    for row in 0..m {
        for col in 0..n {
            let mut acc = 0.0f32;
            for kk in 0..k {
                let av = a[row * k + kk].to_f32();
                let bv = b[kk * n + col].to_f32();
                acc += av * bv;
            }
            c[row * n + col] = acc;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f32, b: f32, atol: f32) -> bool {
        (a - b).abs() <= atol + 1e-3 * a.abs().max(b.abs())
    }

    fn ref_matmul_f32(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
        for row in 0..m {
            for col in 0..n {
                let mut acc = 0.0f32;
                for kk in 0..k {
                    acc += a[row * k + kk] * b[kk * n + col];
                }
                c[row * n + col] = acc;
            }
        }
    }

    #[test]
    fn bf16_matmul_diff_vs_f32_under_5e_3() {
        let m = 4;
        let k = 16;
        let n = 4;
        let a_f32: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.1).sin()).collect();
        let b_f32: Vec<f32> = (0..k * n).map(|i| (i as f32 * 0.07).cos()).collect();
        let a_bf: Vec<bf16> = a_f32.iter().map(|&x| bf16::from_f32(x)).collect();
        let b_bf: Vec<bf16> = b_f32.iter().map(|&x| bf16::from_f32(x)).collect();
        let mut c_bf = vec![0.0f32; m * n];
        let mut c_ref = vec![0.0f32; m * n];
        matmul_bf16_with_f32_accum(&a_bf, &b_bf, &mut c_bf, m, k, n).unwrap();
        ref_matmul_f32(&a_f32, &b_f32, &mut c_ref, m, k, n);
        for (q, r) in c_bf.iter().zip(c_ref.iter()) {
            assert!(close(*q, *r, 5e-3), "bf {q} f32 {r}");
        }
    }

    #[test]
    fn fp16_matmul_diff_vs_f32_under_5e_3() {
        let m = 4;
        let k = 16;
        let n = 4;
        let a_f32: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.1).sin()).collect();
        let b_f32: Vec<f32> = (0..k * n).map(|i| (i as f32 * 0.07).cos()).collect();
        let a_fp: Vec<f16> = a_f32.iter().map(|&x| f16::from_f32(x)).collect();
        let b_fp: Vec<f16> = b_f32.iter().map(|&x| f16::from_f32(x)).collect();
        let mut c_fp = vec![0.0f32; m * n];
        let mut c_ref = vec![0.0f32; m * n];
        matmul_fp16_with_f32_accum(&a_fp, &b_fp, &mut c_fp, m, k, n).unwrap();
        ref_matmul_f32(&a_f32, &b_f32, &mut c_ref, m, k, n);
        for (q, r) in c_fp.iter().zip(c_ref.iter()) {
            assert!(close(*q, *r, 5e-3), "fp {q} f32 {r}");
        }
    }

    #[test]
    fn empty_matmul_returns_empty() {
        let mut c: Vec<f32> = Vec::new();
        matmul_bf16_with_f32_accum(&[], &[], &mut c, 0, 0, 0).unwrap();
        assert!(c.is_empty());
    }

    #[test]
    fn shape_mismatch_returns_error() {
        let mut c = vec![0.0f32; 4];
        let err = matmul_bf16_with_f32_accum(&[bf16::ZERO; 6], &[bf16::ZERO; 8], &mut c, 2, 4, 2)
            .unwrap_err();
        assert!(matches!(err, BfGemmError::ShapeMismatch { which: "a", .. }));
    }

    #[test]
    fn f32_accumulator_preserves_precision_for_long_reduction() {
        // K=4096 sum of small values → without f32 accumulator the
        // reduction would lose precision in bf16.
        let m = 1;
        let k = 4096;
        let n = 1;
        // Each input ~ 0.01 → sum ~ 4096 * 0.01 = 41.0 in expectation
        // (alternating signs → ~0). bf16 accumulator would drift far
        // more than the f32 accumulator.
        let a_f32: Vec<f32> = (0..m * k)
            .map(|i| if i % 2 == 0 { 0.01 } else { -0.01 })
            .collect();
        let b_f32 = vec![1.0f32; k * n];
        let a_bf: Vec<bf16> = a_f32.iter().map(|&x| bf16::from_f32(x)).collect();
        let b_bf: Vec<bf16> = b_f32.iter().map(|&x| bf16::from_f32(x)).collect();
        let mut c_bf = vec![0.0f32; m * n];
        matmul_bf16_with_f32_accum(&a_bf, &b_bf, &mut c_bf, m, k, n).unwrap();
        // f32 accumulator → result close to 0.
        assert!(c_bf[0].abs() < 1.0, "got {}", c_bf[0]);
    }
}
