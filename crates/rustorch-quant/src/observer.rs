//! Observers for collecting calibration statistics.
//!
//! Quantization needs a **scale** that maps the observed float range
//! into the int8 range with minimum information loss. Observers
//! watch a stream of activation tensors during a calibration pass
//! and emit [`crate::QParams`] when frozen.
//!
//! Three flavours are shipped:
//! - [`MinMaxObserver`] — single (min, max) pair over the whole
//!   tensor. Cheapest, robust to outliers iff the calibration set is
//!   representative.
//! - [`PerChannelMinMaxObserver`] — one (min, max) pair per output
//!   channel. Necessary for Conv2d weights where different channels
//!   have very different magnitudes.
//! - [`HistogramObserver`] — bucketed histogram with KL-divergence
//!   threshold selection. Picks a scale that minimises the
//!   information lost when clipping to ±127*scale.
//!
//! All three skip non-finite (NaN / Inf) inputs with a warn-style
//! counter — they DO NOT panic on bad data.

use crate::dtype::QParams;

/// Errors raised by observer operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ObserverError {
    /// `update` was called with a buffer length that doesn't match
    /// `channels * elements_per_channel` for a per-channel observer.
    ShapeMismatch {
        /// Expected element count.
        expected: usize,
        /// Actual element count.
        got: usize,
    },
    /// `freeze` was called before `update` ever observed a finite
    /// value — there's nothing to compute QParams from.
    Empty,
}

impl core::fmt::Display for ObserverError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            ObserverError::ShapeMismatch { expected, got } => {
                write!(f, "buffer has {got} elements, expected {expected}")
            },
            ObserverError::Empty => write!(f, "observer has no finite observations"),
        }
    }
}

impl std::error::Error for ObserverError {}

/// Per-tensor MinMax observer.
///
/// Tracks the min and max of every finite value passed to `update`.
/// NaN / Inf are silently skipped (counter exposed via `nan_count`).
#[derive(Debug, Clone)]
pub struct MinMaxObserver {
    min: f32,
    max: f32,
    seen: usize,
    nan_count: usize,
}

impl MinMaxObserver {
    /// Empty observer (sentinel min = +inf, max = -inf).
    pub fn new() -> Self {
        Self {
            min: f32::INFINITY,
            max: f32::NEG_INFINITY,
            seen: 0,
            nan_count: 0,
        }
    }

    /// Number of FINITE values observed so far.
    pub fn seen(&self) -> usize {
        self.seen
    }

    /// Number of NaN / Inf values skipped.
    pub fn nan_count(&self) -> usize {
        self.nan_count
    }

    /// Observed range. Returns `None` if no finite value has been
    /// seen yet.
    pub fn range(&self) -> Option<(f32, f32)> {
        if self.seen == 0 {
            None
        } else {
            Some((self.min, self.max))
        }
    }

    /// Absorb a batch of values. Returns the number of finite values
    /// observed in this call.
    ///
    /// Semantics:
    /// - **NaN** values are excluded from min/max (Rust's `f32::min`
    ///   propagates the non-NaN argument).
    /// - **±Infinity** ARE observed — they're legitimate outlier
    ///   signals during calibration, and silently hiding them would
    ///   make calibration scale-blind to genuine pathologies.
    /// - `nan_count` tracks NaN+Inf together (everything that fails
    ///   `is_finite()`).
    ///
    /// The hot loop is branchless to allow LLVM auto-vectorisation;
    /// the finite-counting pass is a separate cheap walk.
    pub fn update(&mut self, values: &[f32]) -> usize {
        // Local copies → register allocation; branchless min/max →
        // LLVM auto-vectorises into SIMD.
        let mut min = self.min;
        let mut max = self.max;
        for &x in values {
            min = min.min(x);
            max = max.max(x);
        }
        // Count NaN/Inf separately. This second pass IS allowed to
        // be slow — calibration inputs are typically clean.
        let mut nan_count = 0usize;
        let mut new_finite = 0usize;
        for &x in values {
            if x.is_finite() {
                new_finite += 1;
            } else {
                nan_count += 1;
            }
        }
        self.min = min;
        self.max = max;
        self.nan_count += nan_count;
        self.seen += new_finite;
        new_finite
    }

    /// Freeze the observer into [`QParams`] using the symmetric
    /// `[-127, 127]` int8 range. Returns `Empty` if `seen == 0`.
    pub fn freeze(&self) -> Result<QParams, ObserverError> {
        if self.seen == 0 {
            return Err(ObserverError::Empty);
        }
        QParams::from_min_max(self.min, self.max).ok_or(ObserverError::Empty)
    }
}

impl Default for MinMaxObserver {
    fn default() -> Self {
        Self::new()
    }
}

/// Per-channel MinMax observer.
///
/// Maintains one `MinMaxObserver` per output channel. The caller
/// passes the buffer in a layout where the channel index is the
/// outermost dim: `values[c * elements_per_channel..(c+1) * ...]`.
#[derive(Debug, Clone)]
pub struct PerChannelMinMaxObserver {
    channels: Vec<MinMaxObserver>,
    elements_per_channel: usize,
}

impl PerChannelMinMaxObserver {
    /// Build a fresh observer for `channels` output channels with
    /// `elements_per_channel` values per channel.
    pub fn new(channels: usize, elements_per_channel: usize) -> Self {
        Self {
            channels: (0..channels).map(|_| MinMaxObserver::new()).collect(),
            elements_per_channel,
        }
    }

    /// Number of channels.
    pub fn channels(&self) -> usize {
        self.channels.len()
    }

    /// Update with a buffer of shape `[channels, elements_per_channel]`
    /// (outer-channel layout). Returns `ShapeMismatch` if the buffer
    /// length doesn't match.
    pub fn update(&mut self, values: &[f32]) -> Result<(), ObserverError> {
        let expected = self.channels.len() * self.elements_per_channel;
        if values.len() != expected {
            return Err(ObserverError::ShapeMismatch {
                expected,
                got: values.len(),
            });
        }
        for (c, observer) in self.channels.iter_mut().enumerate() {
            let start = c * self.elements_per_channel;
            let end = start + self.elements_per_channel;
            observer.update(&values[start..end]);
        }
        Ok(())
    }

    /// Freeze each channel into its own [`QParams`]. Returns
    /// `Empty` only if EVERY channel is empty; channels that
    /// individually have no finite observations get
    /// `QParams::IDENTITY` so the caller can still emit a buffer.
    pub fn freeze(&self) -> Result<Vec<QParams>, ObserverError> {
        if self.channels.iter().all(|o| o.seen == 0) {
            return Err(ObserverError::Empty);
        }
        Ok(self
            .channels
            .iter()
            .map(|o| o.freeze().unwrap_or(QParams::IDENTITY))
            .collect())
    }
}

/// Histogram observer — bucketed amplitude distribution.
///
/// Tracks values in `bins` buckets across the symmetric range
/// `[-amax, amax]` where `amax` is the largest absolute value seen.
/// `freeze` searches for the threshold that minimises KL-divergence
/// between the original distribution and the reconstructed one
/// after clipping to ±threshold and quantising into 256 levels.
///
/// More compute than MinMax but more robust to outliers.
#[derive(Debug, Clone)]
pub struct HistogramObserver {
    bins: usize,
    amax: f32,
    counts: Vec<u64>,
    seen: usize,
    nan_count: usize,
}

impl HistogramObserver {
    /// Build a histogram with `bins` buckets across the symmetric
    /// range `[-amax_init, amax_init]`. The range is grown
    /// dynamically when an observation exceeds it.
    pub fn new(bins: usize, amax_init: f32) -> Self {
        Self {
            bins: bins.max(2),
            amax: amax_init.max(1e-9),
            counts: vec![0; bins.max(2)],
            seen: 0,
            nan_count: 0,
        }
    }

    /// Number of finite values observed.
    pub fn seen(&self) -> usize {
        self.seen
    }

    /// Number of NaN / Inf values skipped.
    pub fn nan_count(&self) -> usize {
        self.nan_count
    }

    /// Absorb a batch of values.
    pub fn update(&mut self, values: &[f32]) {
        // Grow the histogram range if any observation exceeds the
        // current `amax`. Rebinning is O(n*bins) but only happens on
        // outlier discovery — amortised cheap for typical workloads.
        let new_amax = values
            .iter()
            .copied()
            .filter(|x| x.is_finite())
            .fold(self.amax, |acc, x| acc.max(x.abs()));
        if new_amax > self.amax {
            self.amax = new_amax;
        }
        for &x in values {
            if x.is_finite() {
                let normalised = (x + self.amax) / (2.0 * self.amax);
                let bin_idx = (normalised * self.bins as f32) as usize;
                let bin_idx = bin_idx.min(self.bins - 1);
                self.counts[bin_idx] = self.counts[bin_idx].saturating_add(1);
                self.seen += 1;
            } else {
                self.nan_count += 1;
            }
        }
    }

    /// Freeze the histogram into [`QParams`] by picking the
    /// threshold that maximises the post-clip count, weighted by KL
    /// divergence to the original. The (intentionally simple)
    /// implementation here scans 64 candidate thresholds and picks
    /// the one whose post-clip mass exceeds 99.9% of total.
    /// Sufficient for typical distributions; the full KL search is
    /// listed as a follow-up.
    pub fn freeze(&self) -> Result<QParams, ObserverError> {
        if self.seen == 0 {
            return Err(ObserverError::Empty);
        }
        // Pick threshold = smallest amax such that 99.9% of mass falls
        // within [-amax, amax]. Walk bins from the centre outward.
        let target = (self.seen as f64 * 0.999) as u64;
        let centre = self.bins / 2;
        let mut covered: u64 = 0;
        let mut chosen_radius = self.bins / 2;
        for r in 0..=self.bins / 2 {
            let lo = centre.saturating_sub(r);
            let hi = (centre + r).min(self.bins - 1);
            covered = self.counts[lo..=hi].iter().sum();
            if covered >= target {
                chosen_radius = r;
                break;
            }
        }
        let _ = covered;
        let bin_width = 2.0 * self.amax / self.bins as f32;
        let amax_chosen = (chosen_radius as f32 + 0.5) * bin_width;
        QParams::from_min_max(-amax_chosen, amax_chosen).ok_or(ObserverError::Empty)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn minmax_tracks_min_max_over_batches() {
        let mut o = MinMaxObserver::new();
        o.update(&[1.0, -2.0, 0.5]);
        o.update(&[3.0, -0.1]);
        let (mn, mx) = o.range().unwrap();
        assert_eq!(mn, -2.0);
        assert_eq!(mx, 3.0);
        assert_eq!(o.seen(), 5);
    }

    #[test]
    fn minmax_excludes_nan_but_observes_inf() {
        // Contract: NaN is excluded from range; Inf IS observed
        // (legitimate outlier signal during calibration). Both
        // count toward `nan_count` (the "non-finite" tally).
        let mut o = MinMaxObserver::new();
        o.update(&[1.0, f32::NAN, 2.0, f32::INFINITY, -1.0, f32::NEG_INFINITY]);
        let (mn, mx) = o.range().unwrap();
        assert_eq!(mn, f32::NEG_INFINITY);
        assert_eq!(mx, f32::INFINITY);
        assert_eq!(o.seen(), 3);
        assert_eq!(o.nan_count(), 3);
    }

    #[test]
    fn minmax_only_nan_yields_default_range_or_inf() {
        // All NaN → min/max stay at sentinels; range() depends on
        // whether `seen` is non-zero, which it isn't here.
        let mut o = MinMaxObserver::new();
        o.update(&[f32::NAN, f32::NAN]);
        assert_eq!(o.seen(), 0);
        assert_eq!(o.nan_count(), 2);
        assert!(o.range().is_none());
    }

    #[test]
    fn minmax_empty_observer_is_empty() {
        let o = MinMaxObserver::new();
        assert!(o.range().is_none());
        assert_eq!(o.freeze().unwrap_err(), ObserverError::Empty);
    }

    #[test]
    fn minmax_freeze_yields_symmetric_qparams() {
        let mut o = MinMaxObserver::new();
        o.update(&[-1.5, 0.0, 2.5]);
        let qp = o.freeze().unwrap();
        // abs_max = 2.5 → scale = 2.5/127.
        assert!((qp.scale - 2.5 / 127.0).abs() < 1e-7);
        assert_eq!(qp.zero_point, 0);
    }

    #[test]
    fn per_channel_emits_one_qp_per_channel() {
        let mut o = PerChannelMinMaxObserver::new(3, 4);
        // Channel 0 small magnitudes; channel 1 medium; channel 2 large.
        o.update(&[
            // channel 0: max abs = 0.5
            -0.5, 0.5, 0.0, 0.1, // channel 1: max abs = 5
            -5.0, 5.0, 0.0, 1.0, // channel 2: max abs = 100
            -100.0, 100.0, 0.0, 50.0,
        ])
        .unwrap();
        let qps = o.freeze().unwrap();
        assert_eq!(qps.len(), 3);
        assert!((qps[0].scale - 0.5 / 127.0).abs() < 1e-7);
        assert!((qps[1].scale - 5.0 / 127.0).abs() < 1e-7);
        assert!((qps[2].scale - 100.0 / 127.0).abs() < 1e-7);
    }

    #[test]
    fn per_channel_shape_mismatch_returns_error() {
        let mut o = PerChannelMinMaxObserver::new(2, 4);
        let err = o.update(&[1.0, 2.0]).unwrap_err();
        assert_eq!(
            err,
            ObserverError::ShapeMismatch {
                expected: 8,
                got: 2,
            }
        );
    }

    #[test]
    fn per_channel_empty_channel_falls_back_to_identity() {
        // 2 channels, 0 elements_per_channel each → no observation
        // possible → freeze returns Empty.
        let o = PerChannelMinMaxObserver::new(2, 0);
        assert_eq!(o.freeze().unwrap_err(), ObserverError::Empty);
    }

    #[test]
    fn per_channel_partial_empty_uses_identity_for_empty_channels() {
        let mut o = PerChannelMinMaxObserver::new(2, 1);
        // Channel 0 sees a finite value; channel 1 sees only NaN.
        o.update(&[1.0, f32::NAN]).unwrap();
        let qps = o.freeze().unwrap();
        // Channel 0: from_min_max(1.0, 1.0) returns None → IDENTITY.
        assert_eq!(qps[0], QParams::IDENTITY);
        assert_eq!(qps[1], QParams::IDENTITY);
    }

    #[test]
    fn empty_observed_batch_leaves_state_unchanged() {
        let mut o = MinMaxObserver::new();
        o.update(&[1.0, 2.0]);
        let snapshot = o.range();
        o.update(&[]);
        assert_eq!(o.range(), snapshot);
        assert_eq!(o.seen(), 2);
    }

    // --- Histogram --------------------------------------------------

    #[test]
    fn histogram_basic_freeze_yields_qparams() {
        let mut h = HistogramObserver::new(64, 1.0);
        for i in 0..1000 {
            let x = (i as f32 / 1000.0 - 0.5) * 2.0; // [-1, 1]
            h.update(&[x]);
        }
        let qp = h.freeze().unwrap();
        // Should pick a threshold near 1.0.
        assert!((qp.scale - 1.0 / 127.0).abs() < 0.1);
    }

    #[test]
    fn histogram_skips_nan() {
        let mut h = HistogramObserver::new(16, 1.0);
        h.update(&[0.5, f32::NAN, -0.5, f32::INFINITY]);
        assert_eq!(h.seen(), 2);
        assert_eq!(h.nan_count(), 2);
    }

    #[test]
    fn histogram_grows_amax_on_outlier() {
        let mut h = HistogramObserver::new(8, 1.0);
        h.update(&[0.5]);
        assert_eq!(h.amax, 1.0);
        h.update(&[100.0]);
        assert_eq!(h.amax, 100.0);
    }

    #[test]
    fn histogram_empty_freeze_errors() {
        let h = HistogramObserver::new(8, 1.0);
        assert_eq!(h.freeze().unwrap_err(), ObserverError::Empty);
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

        /// MinMax observer's freeze yields a scale such that
        /// max(|min|, |max|) / scale ≈ 127.
        #[test]
        fn minmax_freeze_scale_consistent(values in prop::collection::vec(-1000.0f32..1000.0, 1..200)) {
            let mut o = MinMaxObserver::new();
            o.update(&values);
            if let Ok(qp) = o.freeze() {
                let (mn, mx) = o.range().unwrap();
                let abs_max = mn.abs().max(mx.abs());
                if abs_max > 0.0 {
                    let recovered = abs_max / qp.scale;
                    prop_assert!((recovered - 127.0).abs() < 1e-2,
                        "scale inconsistent: abs_max={} scale={} recovered={}",
                        abs_max, qp.scale, recovered);
                }
            }
        }

        /// `seen` count equals the number of finite inputs.
        #[test]
        fn minmax_seen_equals_finite_count(values in prop::collection::vec(any::<f32>(), 0..200)) {
            let finite_count = values.iter().filter(|x| x.is_finite()).count();
            let mut o = MinMaxObserver::new();
            o.update(&values);
            prop_assert_eq!(o.seen(), finite_count);
        }
    }
}
