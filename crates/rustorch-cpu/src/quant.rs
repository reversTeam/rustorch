//! Affine int8 quantization primitives — calibration + quant/dequant
//! for the inference path.
//!
//! Uses the standard symmetric-or-asymmetric affine scheme:
//!
//! ```text
//! q(x) = clamp(round(x / scale) + zero_point, -128, 127)
//! x ≈ scale * (q(x) - zero_point)
//! ```
//!
//! Calibration: run an [`Observer`] over a small dataset of activations
//! (typical: 64-256 batches) so it accumulates the running `min` /
//! `max`. Then [`Observer::compute_qparams`] produces `(scale,
//! zero_point)` such that `min` maps to `-128` and `max` to `127`.
//!
//! Compression: 4× smaller weights on disk + ~2× faster matmul on
//! SIMD-capable CPUs because each load reads 4× as many elements per
//! cache line.
//!
//! v1 surface (CPU only):
//! - [`Observer`] — running min/max tracker
//! - [`QParams`] — `{scale, zero_point}`
//! - [`quantize_tensor_i8`] / [`dequantize_tensor_i8`] —
//!   F32 ↔ I8 conversion via affine scheme
//! - [`int8_matmul`] — naive int8 GEMM with f32 accumulator (parity
//!   reference; SIMD variant lands in a follow-up)

use rustorch_core::tensor::dtype::Dtype;
use rustorch_core::tensor::tensor_impl::Tensor;

/// Quantization parameters produced by an [`Observer`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QParams {
    /// Scale factor (always positive). `q = round(x / scale) + zp`.
    pub scale: f32,
    /// Offset that maps the float zero. For symmetric quantization
    /// this is 0; for asymmetric it shifts so `min ↦ -128`.
    pub zero_point: i8,
}

impl QParams {
    /// Symmetric int8 qparams from an absolute-max bound. `zp = 0`,
    /// `scale = absmax / 127`. Used for weights (typically zero-mean).
    pub fn symmetric(absmax: f32) -> Self {
        let scale = (absmax / 127.0).max(1e-12);
        QParams {
            scale,
            zero_point: 0,
        }
    }

    /// Asymmetric int8 qparams from `[min, max]`. `min ↦ -128`,
    /// `max ↦ 127`. Used for activations (typically post-ReLU).
    pub fn asymmetric(min: f32, max: f32) -> Self {
        let lo = min.min(0.0);
        let hi = max.max(0.0);
        let scale = ((hi - lo) / 255.0).max(1e-12);
        // zero_point chosen so that lo ↦ -128 (clamped to i8 range).
        let zp = (-128.0 - lo / scale).round().clamp(-128.0, 127.0) as i8;
        QParams {
            scale,
            zero_point: zp,
        }
    }

    /// Round `x` to the nearest representable int8 quantum.
    #[inline]
    pub fn quantize(&self, x: f32) -> i8 {
        let q = (x / self.scale).round() as i32 + self.zero_point as i32;
        q.clamp(-128, 127) as i8
    }

    /// Reconstruct the float value of an int8 quantum.
    #[inline]
    pub fn dequantize(&self, q: i8) -> f32 {
        self.scale * (q as i32 - self.zero_point as i32) as f32
    }
}

/// Running observer over activations / weights. Each `record(x)`
/// extends `(min, max)` to include every element of `x`. Once
/// calibration is finished, [`Observer::compute_qparams`] returns the
/// affine qparams in either symmetric or asymmetric mode.
#[derive(Debug, Clone)]
pub struct Observer {
    /// Running minimum over every recorded element.
    pub min: f32,
    /// Running maximum over every recorded element.
    pub max: f32,
    /// Number of elements seen so far. Useful for asserting
    /// calibration is non-empty before producing qparams.
    pub count: u64,
}

impl Default for Observer {
    fn default() -> Self {
        Self::new()
    }
}

impl Observer {
    /// Empty observer.
    pub fn new() -> Self {
        Observer {
            min: f32::INFINITY,
            max: f32::NEG_INFINITY,
            count: 0,
        }
    }

    /// Extend the running `(min, max)` with every element of `slice`.
    pub fn record(&mut self, slice: &[f32]) {
        for &x in slice {
            if x < self.min {
                self.min = x;
            }
            if x > self.max {
                self.max = x;
            }
        }
        self.count += slice.len() as u64;
    }

    /// Convenience: feed a whole F32 [`Tensor`] (it must be
    /// contiguous — we read raw storage).
    pub fn record_tensor(&mut self, t: &Tensor) {
        if t.dtype() != Dtype::F32 {
            return;
        }
        if let Some(s) = t.as_slice::<f32>() {
            self.record(s);
        }
    }

    /// Build symmetric qparams from `max(|min|, |max|)`. Sensible
    /// default for weights. Empty observer (no calibration data) →
    /// fallback `scale = 1.0, zp = 0` (identity quantization).
    pub fn compute_qparams_symmetric(&self) -> QParams {
        if self.count == 0 {
            return QParams {
                scale: 1.0,
                zero_point: 0,
            };
        }
        let absmax = self.min.abs().max(self.max.abs()).max(1e-12);
        QParams::symmetric(absmax)
    }

    /// Build asymmetric qparams from `(min, max)`. Sensible default
    /// for activations. Empty observer → fallback `scale = 1.0, zp = 0`.
    pub fn compute_qparams_asymmetric(&self) -> QParams {
        if self.count == 0 {
            return QParams {
                scale: 1.0,
                zero_point: 0,
            };
        }
        QParams::asymmetric(self.min, self.max)
    }
}

/// Quantize a contiguous F32 [`Tensor`] to I8 using the affine scheme
/// described by `qp`.
pub fn quantize_tensor_i8(t: &Tensor, qp: QParams) -> Tensor {
    assert_eq!(t.dtype(), Dtype::F32, "quantize_tensor_i8 expects F32");
    let src = t
        .as_slice::<f32>()
        .expect("quantize_tensor_i8: input must be contiguous F32");
    let q: Vec<i8> = src.iter().map(|&x| qp.quantize(x)).collect();
    Tensor::from_vec_typed::<i8, _>(t.shape().to_vec(), q).expect("shape numel matches")
}

/// Reverse of [`quantize_tensor_i8`] — rebuild an approximate F32
/// [`Tensor`] from I8 quanta + qparams.
pub fn dequantize_tensor_i8(t: &Tensor, qp: QParams) -> Tensor {
    assert_eq!(t.dtype(), Dtype::I8, "dequantize_tensor_i8 expects I8");
    let src = t
        .as_slice::<i8>()
        .expect("dequantize_tensor_i8: input must be contiguous I8");
    let v: Vec<f32> = src.iter().map(|&q| qp.dequantize(q)).collect();
    Tensor::from_vec_typed::<f32, _>(t.shape().to_vec(), v).expect("shape numel matches")
}

/// Naive `[m, k] @ [k, n]` matmul on quantized I8 tensors. Returns a
/// dequantized F32 tensor of shape `[m, n]`.
///
/// Reference implementation — accumulates in `f32` after pairwise
/// dequantization. Kept simple on purpose: the SIMD-fast int8 GEMM
/// (using i32 accumulators + dp4a/sve.dot) is a follow-up slice.
///
/// Mathematically this is equivalent to
/// `dequant(a) @ dequant(b)` to within rounding.
pub fn int8_matmul(a: &Tensor, a_qp: QParams, b: &Tensor, b_qp: QParams) -> Tensor {
    assert_eq!(a.dtype(), Dtype::I8, "int8_matmul: lhs must be I8");
    assert_eq!(b.dtype(), Dtype::I8, "int8_matmul: rhs must be I8");
    assert_eq!(a.ndim(), 2, "int8_matmul: lhs must be 2D");
    assert_eq!(b.ndim(), 2, "int8_matmul: rhs must be 2D");
    let (m, k1) = (a.shape()[0], a.shape()[1]);
    let (k2, n) = (b.shape()[0], b.shape()[1]);
    assert_eq!(k1, k2, "int8_matmul: K mismatch");
    let k = k1;
    let a_raw = a.as_slice::<i8>().expect("lhs not contiguous I8");
    let b_raw = b.as_slice::<i8>().expect("rhs not contiguous I8");
    let mut out = vec![0.0_f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0_f32;
            for kk in 0..k {
                let a_val = a_qp.dequantize(a_raw[i * k + kk]);
                let b_val = b_qp.dequantize(b_raw[kk * n + j]);
                acc += a_val * b_val;
            }
            out[i * n + j] = acc;
        }
    }
    Tensor::from_vec_typed::<f32, _>(vec![m, n], out).expect("shape numel matches")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn observer_records_min_and_max() {
        let mut obs = Observer::new();
        obs.record(&[1.0, -2.0, 3.5, 0.5]);
        obs.record(&[-3.0, 4.0]);
        assert!((obs.min - (-3.0)).abs() < 1e-9);
        assert!((obs.max - 4.0).abs() < 1e-9);
        assert_eq!(obs.count, 6);
    }

    #[test]
    fn symmetric_qparams_roundtrip() {
        let qp = QParams::symmetric(2.0);
        // 0 maps exactly to 0
        let q0 = qp.quantize(0.0);
        assert_eq!(q0, 0);
        // 2.0 maps to ~127, dequant should be ~2.0
        let q_max = qp.quantize(2.0);
        assert_eq!(q_max, 127);
        let back = qp.dequantize(127);
        assert!((back - 2.0).abs() < 0.02, "got {back}");
    }

    #[test]
    fn asymmetric_qparams_roundtrip_min_max() {
        let qp = QParams::asymmetric(-1.0, 3.0);
        // min should map close to -128
        let q_min = qp.quantize(-1.0);
        assert!(q_min <= -127, "got {q_min}");
        // max should map close to +127
        let q_max = qp.quantize(3.0);
        assert!(q_max >= 126, "got {q_max}");
        // round-trip preserves to within scale/2
        let mid = qp.dequantize(qp.quantize(1.5));
        assert!((mid - 1.5).abs() < qp.scale * 0.51, "got {mid}");
    }

    #[test]
    fn quantize_tensor_then_dequantize_within_scale() {
        let t = Tensor::from_vec([4_usize], vec![-1.0_f32, 0.0, 0.5, 1.0]).unwrap();
        let mut obs = Observer::new();
        obs.record_tensor(&t);
        let qp = obs.compute_qparams_symmetric();
        let q = quantize_tensor_i8(&t, qp);
        assert_eq!(q.dtype(), Dtype::I8);
        let back = dequantize_tensor_i8(&q, qp);
        let orig = t.as_slice::<f32>().unwrap();
        let recon = back.as_slice::<f32>().unwrap();
        for (a, b) in orig.iter().zip(recon) {
            assert!((a - b).abs() < qp.scale, "{a} vs {b}");
        }
    }

    #[test]
    fn int8_matmul_matches_f32_within_quant_error() {
        // 4x4 @ 4x4 — small enough that the dequant noise stays bounded.
        let a_f32 = Tensor::from_vec(
            [4_usize, 4],
            vec![
                0.5, -0.25, 1.0, -0.5, 0.0, 0.75, -1.0, 0.25, 1.0, -0.5, 0.5, 0.0, -0.25, 0.5,
                -0.5, 1.0,
            ],
        )
        .unwrap();
        let b_f32 = Tensor::from_vec(
            [4_usize, 4],
            vec![
                1.0, 0.0, -0.5, 0.25, 0.5, -1.0, 0.0, 0.5, -0.25, 0.5, 1.0, -0.75, 0.0, 0.25, -0.5,
                0.5,
            ],
        )
        .unwrap();
        // Calibrate.
        let mut obs_a = Observer::new();
        obs_a.record_tensor(&a_f32);
        let qp_a = obs_a.compute_qparams_symmetric();
        let mut obs_b = Observer::new();
        obs_b.record_tensor(&b_f32);
        let qp_b = obs_b.compute_qparams_symmetric();
        // Quantize.
        let a_q = quantize_tensor_i8(&a_f32, qp_a);
        let b_q = quantize_tensor_i8(&b_f32, qp_b);
        // Int8 matmul → F32 reference matmul.
        let c_q = int8_matmul(&a_q, qp_a, &b_q, qp_b);
        // F32 reference.
        let mut c_ref = vec![0.0_f32; 16];
        let a_raw = a_f32.as_slice::<f32>().unwrap();
        let b_raw = b_f32.as_slice::<f32>().unwrap();
        for i in 0..4 {
            for j in 0..4 {
                let mut s = 0.0_f32;
                for kk in 0..4 {
                    s += a_raw[i * 4 + kk] * b_raw[kk * 4 + j];
                }
                c_ref[i * 4 + j] = s;
            }
        }
        // Tolerance ≈ k * (qp_a.scale + qp_b.scale) — 4 * 0.008 + 4 * 0.008.
        let tol = 4.0 * (qp_a.scale + qp_b.scale);
        for (got, exp) in c_q.as_slice::<f32>().unwrap().iter().zip(&c_ref) {
            assert!((got - exp).abs() < tol, "got {got} vs {exp} (tol {tol})");
        }
    }

    #[test]
    fn empty_observer_yields_safe_qparams() {
        // Even with no calibration data, qparams should not produce
        // NaN / inf — they fall back to a tiny scale and zp=0.
        let obs = Observer::new();
        let qp = obs.compute_qparams_symmetric();
        assert!(qp.scale.is_finite());
        assert!(qp.scale > 0.0);
    }
}
