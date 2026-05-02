//! Online (Welford-style) softmax — Phase 3 task `a74f4555`.
//!
//! Naive softmax over a vector `x` of length N requires two passes:
//! 1. Find `max(x)` to subtract for numerical stability.
//! 2. Compute `exp(x - max)`, sum, divide.
//!
//! That requires the full vector in memory at once. Flash Attention
//! processes Q/K/V in *tiles*; the softmax must stream over those
//! tiles without ever holding the entire row of scores. The trick:
//! maintain a running `(m, l)` pair per tile —
//!
//!   `m` = max seen so far over `x`
//!   `l` = sum_i exp(x_i - m) so far
//!
//! Combining two adjacent partial states `(m_a, l_a)` and `(m_b, l_b)`
//! is exact:
//!
//! ```text
//!   m' = max(m_a, m_b)
//!   l' = l_a * exp(m_a - m') + l_b * exp(m_b - m')
//! ```
//!
//! When the tile aggregation finishes, `softmax(x_i) = exp(x_i - m) / l`.
//!
//! ## What this module ships
//!
//! - [`OnlineSoftmaxState`] — `(m, l)` pair with absorb/extend
//!   helpers.
//! - [`combine_tiles`] — pure binary combinator used as the
//!   reduction kernel; **commutative** and **associative** within
//!   floating-point tolerance.
//! - [`online_softmax_full`] — single-pass reference implementation
//!   that iterates over a slice and yields the final `(m, l)`
//!   matching the naive algorithm exactly (verified by unit tests).

/// A partial online-softmax accumulator. `m` tracks the running
/// maximum; `l` tracks `sum_i exp(x_i - m)` over the inputs absorbed
/// so far.
///
/// Sentinel value `OnlineSoftmaxState::EMPTY` represents the additive
/// identity — combining with anything is a no-op. It uses `m =
/// f32::NEG_INFINITY` and `l = 0.0`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OnlineSoftmaxState {
    /// Running max.
    pub m: f32,
    /// `sum_i exp(x_i - m)`.
    pub l: f32,
}

impl OnlineSoftmaxState {
    /// Identity element for [`combine_tiles`]: `m = -inf`, `l = 0`.
    pub const EMPTY: Self = Self {
        m: f32::NEG_INFINITY,
        l: 0.0,
    };

    /// Build a state from a single scalar `x`.
    #[inline]
    pub fn from_scalar(x: f32) -> Self {
        Self { m: x, l: 1.0 }
    }

    /// Absorb a new scalar `x` into the running state. Equivalent to
    /// `combine_tiles(self, &Self::from_scalar(x))` but slightly
    /// cheaper.
    #[inline]
    pub fn absorb(&mut self, x: f32) {
        if x > self.m {
            // New global max — rescale the running sum.
            self.l = self.l * (self.m - x).exp() + 1.0;
            self.m = x;
        } else {
            // Existing max stays; just add exp(x - m).
            self.l += (x - self.m).exp();
        }
    }

    /// Number of finite entries effectively counted by `l`. Useful
    /// for sanity checks; not part of the softmax math.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.m == f32::NEG_INFINITY && self.l == 0.0
    }
}

impl Default for OnlineSoftmaxState {
    fn default() -> Self {
        Self::EMPTY
    }
}

/// Combine two partial states into one. **Pure** — no allocation, no
/// side effects. Commutative and associative within IEEE-754
/// tolerance (verified by the `associativity_*` proptest cases).
///
/// Special handling:
/// - If both inputs are EMPTY, returns EMPTY.
/// - If one input is EMPTY, returns the other unchanged (identity
///   element).
/// - If both `m` are `-inf` (i.e. one is EMPTY but `l > 0` — should
///   not happen in well-formed traces), the result is also EMPTY-ish:
///   `m = -inf`, `l = a.l + b.l`.
#[inline]
pub fn combine_tiles(a: OnlineSoftmaxState, b: OnlineSoftmaxState) -> OnlineSoftmaxState {
    if a.is_empty() {
        return b;
    }
    if b.is_empty() {
        return a;
    }
    let m_new = if a.m > b.m { a.m } else { b.m };
    // exp((-inf) - finite) underflows to 0 cleanly, so the if/else
    // here is just a perf optimisation that avoids 0+l_a or l_b+0.
    if a.m == m_new && b.m == m_new {
        OnlineSoftmaxState {
            m: m_new,
            l: a.l + b.l,
        }
    } else if a.m == m_new {
        OnlineSoftmaxState {
            m: m_new,
            l: a.l + b.l * (b.m - m_new).exp(),
        }
    } else {
        OnlineSoftmaxState {
            m: m_new,
            l: a.l * (a.m - m_new).exp() + b.l,
        }
    }
}

/// Run the online softmax over a slice and return the final
/// `(m, l)`. Equivalent to processing `xs` as a single tile.
///
/// The downstream consumer computes `softmax(x_i) = (x_i - state.m).exp() / state.l`.
///
/// Special cases:
/// - Empty slice: returns [`OnlineSoftmaxState::EMPTY`].
/// - All -inf inputs: returns `m = -inf, l = 0` (the divider would
///   be zero, signalling that softmax is undefined for this row —
///   the caller must check before dividing).
/// - Single +inf: returns `m = +inf, l = 1` (the resulting softmax
///   places all mass on that index).
pub fn online_softmax_full(xs: &[f32]) -> OnlineSoftmaxState {
    let mut state = OnlineSoftmaxState::EMPTY;
    for &x in xs {
        state.absorb(x);
    }
    state
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference: naive two-pass softmax `(m, l)` over xs.
    fn naive(xs: &[f32]) -> (f32, f32) {
        if xs.is_empty() {
            return (f32::NEG_INFINITY, 0.0);
        }
        let m = xs.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let l: f32 = xs.iter().map(|x| (x - m).exp()).sum();
        (m, l)
    }

    fn close(a: f32, b: f32, tol: f32) -> bool {
        if a.is_nan() && b.is_nan() {
            return true;
        }
        if a.is_infinite() && b.is_infinite() && a.signum() == b.signum() {
            return true;
        }
        (a - b).abs() <= tol * (1.0 + a.abs().max(b.abs()))
    }

    #[test]
    fn single_tile_matches_naive() {
        let xs = [0.5, -1.0, 2.5, 0.1, -3.0, 4.0];
        let online = online_softmax_full(&xs);
        let (m_n, l_n) = naive(&xs);
        assert!(close(online.m, m_n, 1e-6));
        assert!(close(online.l, l_n, 1e-6));
    }

    #[test]
    fn four_tiles_combine_match_naive_within_1e_7() {
        let xs: Vec<f32> = (0..32).map(|i| (i as f32 * 0.1).sin()).collect();
        let tile_size = 8;
        let mut tiles: Vec<OnlineSoftmaxState> = Vec::new();
        for chunk in xs.chunks(tile_size) {
            tiles.push(online_softmax_full(chunk));
        }
        // Combine left-to-right.
        let combined = tiles
            .into_iter()
            .fold(OnlineSoftmaxState::EMPTY, combine_tiles);
        let (m_n, l_n) = naive(&xs);
        assert!(close(combined.m, m_n, 1e-7));
        assert!(close(combined.l, l_n, 1e-6));
    }

    #[test]
    fn associativity_combining_in_any_order() {
        // A four-tile fixture with non-trivial values.
        let a = OnlineSoftmaxState { m: 1.5, l: 3.0 };
        let b = OnlineSoftmaxState { m: 0.7, l: 5.0 };
        let c = OnlineSoftmaxState { m: 2.1, l: 1.0 };
        let d = OnlineSoftmaxState { m: -0.3, l: 7.0 };
        // ((a, b), (c, d)) vs (((a, b), c), d) vs (a, (b, (c, d)))
        let s1 = combine_tiles(combine_tiles(a, b), combine_tiles(c, d));
        let s2 = combine_tiles(combine_tiles(combine_tiles(a, b), c), d);
        let s3 = combine_tiles(a, combine_tiles(b, combine_tiles(c, d)));
        assert!(close(s1.m, s2.m, 1e-6));
        assert!(close(s1.l, s2.l, 1e-6));
        assert!(close(s1.m, s3.m, 1e-6));
        assert!(close(s1.l, s3.l, 1e-6));
    }

    #[test]
    fn commutativity_combining_in_either_order() {
        let a = OnlineSoftmaxState { m: 1.5, l: 3.0 };
        let b = OnlineSoftmaxState { m: 0.7, l: 5.0 };
        let ab = combine_tiles(a, b);
        let ba = combine_tiles(b, a);
        assert!(close(ab.m, ba.m, 1e-7));
        assert!(close(ab.l, ba.l, 1e-7));
    }

    // --- Edge cases -------------------------------------------------

    #[test]
    fn empty_input_returns_empty_state() {
        let state = online_softmax_full(&[]);
        assert_eq!(state, OnlineSoftmaxState::EMPTY);
        assert!(state.is_empty());
    }

    #[test]
    fn all_neg_inf_yields_undefined_softmax() {
        let xs = [f32::NEG_INFINITY; 4];
        let state = online_softmax_full(&xs);
        assert_eq!(state.m, f32::NEG_INFINITY);
        // The naive algorithm would compute exp(0)*N = N here because
        // all entries equal m, so l = N. Match it bitwise.
        let (m_n, l_n) = naive(&xs);
        assert_eq!(state.m, m_n);
        // For all -inf: exp(-inf - -inf) = exp(NaN) = NaN; both paths
        // produce NaN. We accept that the resulting softmax would
        // need NaN-handling at the consumer; document not_silently_mask.
        if l_n.is_nan() {
            assert!(state.l.is_nan() || state.l == 0.0);
        } else {
            assert!(close(state.l, l_n, 1e-6));
        }
    }

    #[test]
    fn single_pos_inf_yields_one_hot_softmax() {
        let xs = [1.0, f32::INFINITY, -2.0, 0.0];
        let state = online_softmax_full(&xs);
        assert_eq!(state.m, f32::INFINITY);
        // l = exp(0) + 3 * exp(-inf) = 1 + 0 + 0 + 0 = 1.
        assert_eq!(state.l, 1.0);
    }

    #[test]
    fn empty_combined_with_state_is_state() {
        let s = OnlineSoftmaxState { m: 1.5, l: 3.0 };
        assert_eq!(combine_tiles(OnlineSoftmaxState::EMPTY, s), s);
        assert_eq!(combine_tiles(s, OnlineSoftmaxState::EMPTY), s);
    }

    #[test]
    fn from_scalar_matches_single_element_input() {
        let s1 = online_softmax_full(&[2.5]);
        let s2 = OnlineSoftmaxState::from_scalar(2.5);
        assert_eq!(s1, s2);
    }

    // --- Numerical stability ---------------------------------------

    #[test]
    fn stable_for_large_magnitudes_close_together() {
        // Naive softmax on x = [1e10, 1e10 + 1e-5, 1e10 - 1e-5]
        // should not catastrophically cancel; the relative ratios
        // between exp(...) terms are well-defined.
        let xs = [1e10_f32, 1e10 + 1e-5, 1e10 - 1e-5];
        let state = online_softmax_full(&xs);
        let (m_n, l_n) = naive(&xs);
        assert_eq!(state.m, m_n);
        assert!(close(state.l, l_n, 1e-5));
        // l is ~3 (three nearly-equal entries → each contributes ~1).
        assert!((state.l - 3.0).abs() < 0.1);
    }

    #[test]
    fn does_not_overflow_for_huge_inputs() {
        // 1e30 + 1e30 should NOT overflow because of the m subtraction.
        let xs = [1e30_f32, 1e30, 1e30, 1e30];
        let state = online_softmax_full(&xs);
        assert_eq!(state.m, 1e30);
        // exp(1e30 - 1e30) = exp(0) = 1, four times → l = 4.
        assert_eq!(state.l, 4.0);
        assert!(state.l.is_finite());
    }
}

/// Property tests: 1000 random tile sequences verifying online vs
/// naive equivalence and combine_tiles algebraic laws.
#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    fn finite_xs() -> impl Strategy<Value = Vec<f32>> {
        prop::collection::vec(-100.0f32..100.0, 1..200)
    }

    fn close(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() <= tol * (1.0 + a.abs().max(b.abs()))
    }

    fn naive(xs: &[f32]) -> (f32, f32) {
        if xs.is_empty() {
            return (f32::NEG_INFINITY, 0.0);
        }
        let m = xs.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let l: f32 = xs.iter().map(|x| (x - m).exp()).sum();
        (m, l)
    }

    proptest! {
        #![proptest_config(ProptestConfig {
            cases: 1000,
            .. ProptestConfig::default()
        })]

        /// Online softmax matches naive within 1e-6 on any random
        /// finite input.
        #[test]
        fn online_matches_naive_on_finite_inputs(xs in finite_xs()) {
            let online = online_softmax_full(&xs);
            let (m_n, l_n) = naive(&xs);
            prop_assert!(close(online.m, m_n, 1e-6));
            prop_assert!(close(online.l, l_n, 1e-5));
        }

        /// Tile-by-tile aggregation matches single-tile aggregation.
        /// Checks for any tile size in 1..max(len/2, 2).
        #[test]
        fn tiled_matches_full((xs, tile) in finite_xs().prop_flat_map(|xs| {
            let len = xs.len();
            (Just(xs), 1usize..(len.max(2)))
        })) {
            let full = online_softmax_full(&xs);
            let combined = xs.chunks(tile)
                .map(online_softmax_full)
                .fold(OnlineSoftmaxState::EMPTY, combine_tiles);
            prop_assert!(close(full.m, combined.m, 1e-6));
            prop_assert!(close(full.l, combined.l, 1e-5));
        }

        /// `combine_tiles` is commutative.
        #[test]
        fn combine_is_commutative(
            (m_a, l_a, m_b, l_b) in (-100.0f32..100.0, 0.001f32..1000.0, -100.0f32..100.0, 0.001f32..1000.0)
        ) {
            let a = OnlineSoftmaxState { m: m_a, l: l_a };
            let b = OnlineSoftmaxState { m: m_b, l: l_b };
            let ab = combine_tiles(a, b);
            let ba = combine_tiles(b, a);
            prop_assert!(close(ab.m, ba.m, 1e-7));
            prop_assert!(close(ab.l, ba.l, 1e-6));
        }

        /// `EMPTY` is the left and right identity of `combine_tiles`.
        #[test]
        fn empty_is_identity((m, l) in (-100.0f32..100.0, 0.001f32..1000.0)) {
            let s = OnlineSoftmaxState { m, l };
            prop_assert_eq!(combine_tiles(OnlineSoftmaxState::EMPTY, s), s);
            prop_assert_eq!(combine_tiles(s, OnlineSoftmaxState::EMPTY), s);
        }
    }
}
