//! Int8 GEMM — scalar reference + (future) SIMD specialisations.
//!
//! This module ships the **scalar** int8 dot-product accumulator
//! that every other backend (AVX-VNNI, NEON sdot, GPU dp4a) must
//! match bit-exactly. The SIMD paths land in subsequent commits;
//! see Phase 3 plan `Quantization (Inference)`, task
//! `SIMD int8 kernels (CPU AVX-VNNI / NEON dot product)` for the
//! arch-specific specialisations.
//!
//! ## Math
//!
//! Given:
//! - `a: i8[M*K]`     (signed activations, with `qa: QParams`)
//! - `b: i8[K*N]`     (signed weights,    with `qb: QParams`)
//!
//! Output:
//! - `c: f32[M*N]`    (dequantised result)
//!
//! ```text
//!   acc[m, n] = Σ_k (a[m, k] - qa.zero_point) * (b[k, n] - qb.zero_point)
//!   c[m, n]   = acc[m, n] * qa.scale * qb.scale
//! ```
//!
//! The accumulator is i32 (matches AVX-VNNI's vpdpbusd and NEON's
//! sdot output type) — int8 × int8 fits in i16, summed over up to
//! 2^16 lanes still fits in i32. K up to ~16M is safe.
//!
//! ## Layout
//!
//! Row-major. `a` is `M × K`; `b` is `K × N`. `c` is `M × N`. The
//! caller passes flat slices in row-major order.

use crate::dtype::QParams;

/// Errors raised by int8 GEMM.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GemmError {
    /// Buffer length doesn't match the declared `(rows, cols)` shape.
    ShapeMismatch {
        /// Which buffer (a / b / c).
        which: &'static str,
        /// Expected element count = rows * cols.
        expected: usize,
        /// Actual element count.
        got: usize,
    },
}

impl core::fmt::Display for GemmError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            GemmError::ShapeMismatch {
                which,
                expected,
                got,
            } => write!(f, "{which} buffer has {got} elements, expected {expected}"),
        }
    }
}

impl std::error::Error for GemmError {}

/// Scalar int8 GEMM: `c = (a - qa.zp) · (b - qb.zp) * qa.scale * qb.scale`.
///
/// Used as the **golden reference** for the AVX-VNNI / NEON / GPU
/// specialisations — those are required to match this output bit-
/// exactly.
pub fn gemm_int8_scalar(
    a: &[i8],
    b: &[i8],
    c: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    qa: QParams,
    qb: QParams,
) -> Result<(), GemmError> {
    if a.len() != m * k {
        return Err(GemmError::ShapeMismatch {
            which: "a",
            expected: m * k,
            got: a.len(),
        });
    }
    if b.len() != k * n {
        return Err(GemmError::ShapeMismatch {
            which: "b",
            expected: k * n,
            got: b.len(),
        });
    }
    if c.len() != m * n {
        return Err(GemmError::ShapeMismatch {
            which: "c",
            expected: m * n,
            got: c.len(),
        });
    }
    let zp_a = qa.zero_point;
    let zp_b = qb.zero_point;
    let scale_out = qa.scale * qb.scale;
    for row in 0..m {
        for col in 0..n {
            let mut acc: i32 = 0;
            for kk in 0..k {
                let a_val = a[row * k + kk] as i32 - zp_a;
                let b_val = b[kk * n + col] as i32 - zp_b;
                acc += a_val * b_val;
            }
            c[row * n + col] = acc as f32 * scale_out;
        }
    }
    Ok(())
}

/// Reference f32 GEMM used by tests/benches as the baseline that
/// scalar int8 should approximate within quantisation error. Same
/// `[M, K] × [K, N] = [M, N]` row-major layout.
pub fn gemm_f32_reference(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
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

/// CPU feature detection. AVX-VNNI / NEON sdot specialisations land
/// in a follow-up; this helper exists so call sites can branch
/// today without churning when the SIMD paths arrive.
pub mod cpu_features {
    /// True iff the host advertises AVX-VNNI (`vpdpbusd`). Always
    /// false on non-x86_64 targets.
    #[inline]
    pub fn has_avx_vnni() -> bool {
        #[cfg(all(target_arch = "x86_64", target_feature = "avxvnni"))]
        {
            true
        }
        #[cfg(not(all(target_arch = "x86_64", target_feature = "avxvnni")))]
        {
            false
        }
    }

    /// True iff the host advertises NEON sdot (`sdot`/`udot`).
    /// Always false on non-aarch64 targets.
    #[inline]
    pub fn has_neon_sdot() -> bool {
        #[cfg(all(target_arch = "aarch64", target_feature = "dotprod"))]
        {
            true
        }
        #[cfg(not(all(target_arch = "aarch64", target_feature = "dotprod")))]
        {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f32, b: f32, atol: f32) -> bool {
        (a - b).abs() <= atol + 1e-3 * a.abs().max(b.abs())
    }

    #[test]
    fn identity_qparams_no_zp_basic_2x2x2() {
        // a = [[1, 2], [3, 4]]
        // b = [[5, 6], [7, 8]]
        // c = [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]]
        //   = [[19, 22], [43, 50]]
        let a = [1i8, 2, 3, 4];
        let b = [5i8, 6, 7, 8];
        let mut c = vec![0.0f32; 4];
        gemm_int8_scalar(
            &a,
            &b,
            &mut c,
            2,
            2,
            2,
            QParams::IDENTITY,
            QParams::IDENTITY,
        )
        .unwrap();
        assert_eq!(c, vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn scale_factor_applied_correctly() {
        // Single-element gemm: a=4, b=8 → acc = 32 → c = 32 * 0.25 * 0.5 = 4.
        let a = [4i8];
        let b = [8i8];
        let mut c = vec![0.0f32; 1];
        let qa = QParams {
            scale: 0.25,
            zero_point: 0,
        };
        let qb = QParams {
            scale: 0.5,
            zero_point: 0,
        };
        gemm_int8_scalar(&a, &b, &mut c, 1, 1, 1, qa, qb).unwrap();
        assert_eq!(c[0], 4.0);
    }

    #[test]
    fn zero_point_subtracted_before_dot() {
        // a = [10], qa.zp = 8 → effective 2.
        // b = [5], qb.zp = 0 → effective 5.
        // acc = 2 * 5 = 10 → c = 10 * 1.0 * 1.0 = 10.
        let a = [10i8];
        let b = [5i8];
        let mut c = vec![0.0f32; 1];
        let qa = QParams {
            scale: 1.0,
            zero_point: 8,
        };
        let qb = QParams::IDENTITY;
        gemm_int8_scalar(&a, &b, &mut c, 1, 1, 1, qa, qb).unwrap();
        assert_eq!(c[0], 10.0);
    }

    #[test]
    fn shape_mismatch_returns_error() {
        let a = [1i8; 6]; // M*K should be 2*4 = 8
        let b = [1i8; 8];
        let mut c = vec![0.0f32; 4];
        let err = gemm_int8_scalar(
            &a,
            &b,
            &mut c,
            2,
            4,
            2,
            QParams::IDENTITY,
            QParams::IDENTITY,
        )
        .unwrap_err();
        assert!(matches!(
            err,
            GemmError::ShapeMismatch {
                which: "a",
                expected: 8,
                got: 6
            }
        ));
    }

    #[test]
    fn empty_gemm_returns_empty_output() {
        let a: [i8; 0] = [];
        let b: [i8; 0] = [];
        let mut c: Vec<f32> = Vec::new();
        gemm_int8_scalar(
            &a,
            &b,
            &mut c,
            0,
            0,
            0,
            QParams::IDENTITY,
            QParams::IDENTITY,
        )
        .unwrap();
        assert!(c.is_empty());
    }

    #[test]
    fn k_zero_yields_all_zeros() {
        // K=0 → empty inner loop → acc = 0 → c = 0 * scale = 0.
        let a: [i8; 0] = [];
        let b: [i8; 0] = [];
        let mut c = vec![1.0f32; 4]; // pre-fill non-zero
        gemm_int8_scalar(
            &a,
            &b,
            &mut c,
            2,
            0,
            2,
            QParams {
                scale: 1.0,
                zero_point: 0,
            },
            QParams::IDENTITY,
        )
        .unwrap();
        assert_eq!(c, vec![0.0; 4]);
    }

    #[test]
    fn quantised_gemm_approximates_f32_within_quant_error() {
        // 4×8 × 8×4 random-ish gemm. Quantise inputs to int8 then
        // gemm; compare to direct f32 gemm. The error is bounded by
        // qa.scale * qb.scale * K (cumulative quant noise).
        let m = 4;
        let k = 8;
        let n = 4;
        let a_f32: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.137).sin() * 1.0).collect();
        let b_f32: Vec<f32> = (0..k * n).map(|i| (i as f32 * 0.291).cos() * 1.0).collect();
        // Quantise both ends symmetrically.
        let a_min = a_f32.iter().copied().fold(f32::INFINITY, f32::min);
        let a_max = a_f32.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let b_min = b_f32.iter().copied().fold(f32::INFINITY, f32::min);
        let b_max = b_f32.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let qa = QParams::from_min_max(a_min, a_max).unwrap();
        let qb = QParams::from_min_max(b_min, b_max).unwrap();
        let mut a_q = vec![0i8; a_f32.len()];
        let mut b_q = vec![0i8; b_f32.len()];
        crate::qops::quantize(&a_f32, &mut a_q, qa).unwrap();
        crate::qops::quantize(&b_f32, &mut b_q, qb).unwrap();

        let mut c_q = vec![0.0f32; m * n];
        gemm_int8_scalar(&a_q, &b_q, &mut c_q, m, k, n, qa, qb).unwrap();

        let mut c_ref = vec![0.0f32; m * n];
        gemm_f32_reference(&a_f32, &b_f32, &mut c_ref, m, k, n);

        // Cumulative quantisation error per output cell: each of K
        // products has bounded error |Δ(a)| * |b| + |a| * |Δ(b)| ≈
        // (qa.scale * |b_max| + qb.scale * |a_max|) / 2; summed over
        // K. Add a 4× safety factor for the int8 round-trip noise.
        let max_abs = a_max.abs().max(b_max.abs());
        let bound = k as f32 * (qa.scale + qb.scale) * max_abs * 2.0;
        for (q, r) in c_q.iter().zip(c_ref.iter()) {
            assert!(
                close(*q, *r, bound),
                "q={} r={} diff={} bound={}",
                q,
                r,
                (q - r).abs(),
                bound
            );
        }
    }

    #[test]
    fn cpu_feature_detection_does_not_panic() {
        // The detection helpers must always return a bool without
        // panic on any host. Ensures the cfg-gated body compiles on
        // every target.
        let _ = cpu_features::has_avx_vnni();
        let _ = cpu_features::has_neon_sdot();
    }
}

/// Property tests: 200-case verification of int8 GEMM correctness
/// against the f32 reference (quantisation error bounds).
#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    fn close(a: f32, b: f32, atol: f32) -> bool {
        (a - b).abs() <= atol
    }

    fn shape_strategy() -> impl Strategy<Value = (usize, usize, usize)> {
        (1usize..16, 1usize..16, 1usize..16)
    }

    proptest! {
        #![proptest_config(ProptestConfig {
            cases: 200,
            .. ProptestConfig::default()
        })]

        /// int8 gemm matches the f32 reference within the
        /// cumulative quantisation error bound.
        #[test]
        fn int8_gemm_matches_f32_reference((m, k, n) in shape_strategy()) {
            let a_f32: Vec<f32> = (0..m * k)
                .map(|i| ((i as u32).wrapping_mul(2654435761) as f32 / u32::MAX as f32 - 0.5) * 2.0)
                .collect();
            let b_f32: Vec<f32> = (0..k * n)
                .map(|i| ((i as u32).wrapping_mul(13).wrapping_mul(2654435761) as f32 / u32::MAX as f32 - 0.5) * 2.0)
                .collect();
            let a_min = a_f32.iter().copied().fold(f32::INFINITY, f32::min);
            let a_max = a_f32.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let b_min = b_f32.iter().copied().fold(f32::INFINITY, f32::min);
            let b_max = b_f32.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            if let (Some(qa), Some(qb)) = (
                QParams::from_min_max(a_min, a_max),
                QParams::from_min_max(b_min, b_max),
            ) {
                let mut a_q = vec![0i8; a_f32.len()];
                let mut b_q = vec![0i8; b_f32.len()];
                crate::qops::quantize(&a_f32, &mut a_q, qa).unwrap();
                crate::qops::quantize(&b_f32, &mut b_q, qb).unwrap();
                let mut c_q = vec![0.0f32; m * n];
                gemm_int8_scalar(&a_q, &b_q, &mut c_q, m, k, n, qa, qb).unwrap();
                let mut c_ref = vec![0.0f32; m * n];
                gemm_f32_reference(&a_f32, &b_f32, &mut c_ref, m, k, n);
                let max_abs = a_max.abs().max(b_max.abs()).max(1e-3);
                let bound = k as f32 * (qa.scale + qb.scale) * max_abs * 2.0;
                for (q, r) in c_q.iter().zip(c_ref.iter()) {
                    prop_assert!(
                        close(*q, *r, bound),
                        "m={} k={} n={} q={} r={} diff={} bound={}",
                        m, k, n, q, r, (q - r).abs(), bound
                    );
                }
            }
        }

        /// Output shape always equals m * n regardless of inputs.
        #[test]
        fn output_shape_is_m_times_n((m, k, n) in shape_strategy()) {
            let a = vec![1i8; m * k];
            let b = vec![1i8; k * n];
            let mut c = vec![0.0f32; m * n];
            gemm_int8_scalar(&a, &b, &mut c, m, k, n, QParams::IDENTITY, QParams::IDENTITY).unwrap();
            prop_assert_eq!(c.len(), m * n);
        }
    }
}
