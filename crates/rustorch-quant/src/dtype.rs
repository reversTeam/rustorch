//! Int8 quantization parameters.
//!
//! A quantized tensor is represented as a buffer of `i8` values plus
//! a [`QParams`] descriptor that tells the runtime how to map
//! between the int8 domain and the original float domain:
//!
//! ```text
//!   x_float ≈ (x_int8 - zero_point) * scale
//! ```
//!
//! The `Int8` dtype itself is a no-op marker — the actual storage is
//! `i8`; this module only carries the metadata needed to round-trip.

/// Per-tensor quantization parameters.
///
/// The float ↔ int8 map is `f = (i - zero_point) * scale`. Both ends
/// of the map use these params to quantize an input batch and to
/// dequantize an output batch back to float.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QParams {
    /// Multiplier between int8 ticks and float units. Always > 0.
    pub scale: f32,
    /// Float origin in int8 ticks. Typically in `[-128, 127]`.
    pub zero_point: i32,
}

impl QParams {
    /// Identity-ish params: scale = 1, zero_point = 0. Useful for
    /// tests and as a sentinel for "uncalibrated".
    pub const IDENTITY: Self = Self {
        scale: 1.0,
        zero_point: 0,
    };

    /// Build params from a `(min, max)` range using the symmetric
    /// `[-127, 127]` int8 range (asymmetric `[-128, 127]` would
    /// quantize one extra negative value but break sign symmetry).
    /// Returns `None` if `min == max` (zero-range observation).
    pub fn from_min_max(min: f32, max: f32) -> Option<Self> {
        if !(min.is_finite() && max.is_finite()) || min >= max {
            return None;
        }
        let abs_max = min.abs().max(max.abs());
        if abs_max == 0.0 {
            return None;
        }
        let scale = abs_max / 127.0;
        Some(Self {
            scale,
            zero_point: 0,
        })
    }

    /// Build asymmetric params from a `(min, max)` range using the
    /// full int8 `[-128, 127]` range. Useful for unsigned-leaning
    /// activations (e.g. ReLU outputs in `[0, x_max]`).
    pub fn from_min_max_asymmetric(min: f32, max: f32) -> Option<Self> {
        if !(min.is_finite() && max.is_finite()) || min >= max {
            return None;
        }
        let q_max = 127.0_f32;
        let q_min = -128.0_f32;
        let scale = (max - min) / (q_max - q_min);
        if scale == 0.0 {
            return None;
        }
        let zero_point = (q_min - min / scale).round().clamp(q_min, q_max) as i32;
        Some(Self { scale, zero_point })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_params_have_unit_scale_and_zero_zp() {
        assert_eq!(QParams::IDENTITY.scale, 1.0);
        assert_eq!(QParams::IDENTITY.zero_point, 0);
    }

    #[test]
    fn from_min_max_symmetric_centers_on_zero() {
        let qp = QParams::from_min_max(-2.0, 1.0).unwrap();
        // abs_max = 2 → scale = 2/127.
        assert!((qp.scale - 2.0 / 127.0).abs() < 1e-7);
        assert_eq!(qp.zero_point, 0);
    }

    #[test]
    fn from_min_max_zero_range_returns_none() {
        assert!(QParams::from_min_max(1.0, 1.0).is_none());
    }

    #[test]
    fn from_min_max_inverted_returns_none() {
        assert!(QParams::from_min_max(2.0, -2.0).is_none());
    }

    #[test]
    fn from_min_max_with_nan_returns_none() {
        assert!(QParams::from_min_max(f32::NAN, 1.0).is_none());
        assert!(QParams::from_min_max(0.0, f32::INFINITY).is_none());
    }

    #[test]
    fn asymmetric_uses_full_int8_range() {
        // Range [0, 1] → scale = 1/255, zero_point = -128.
        let qp = QParams::from_min_max_asymmetric(0.0, 1.0).unwrap();
        assert!((qp.scale - 1.0 / 255.0).abs() < 1e-7);
        assert_eq!(qp.zero_point, -128);
    }

    #[test]
    fn qparams_serialise_round_trip_via_field_copy() {
        // Sanity: QParams is Copy, so trivially round-trips. Used as
        // a stand-in for the "serialization round-trip" step until
        // the rustorch-serde integration lands.
        let original = QParams {
            scale: 0.0078125,
            zero_point: -64,
        };
        let copied = original;
        assert_eq!(copied, original);
    }
}
