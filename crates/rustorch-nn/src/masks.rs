//! Attention mask helpers — autograd-aware Variable masks suitable for
//! adding directly to attention scores before softmax.
//!
//! These are complementary to the low-level GPU/CPU mask kernels (e.g.
//! `rustorch_wgpu::apply_causal_mask`, `rustorch_attention::Mask`):
//! those operate on raw buffers, while these wrap a Variable for use
//! inside autograd-aware modules.
//!
//! Convention: every mask returned here is an **additive** bias — `0.0`
//! on positions to keep, `-1e4` on positions to mask. We use `-1e4`
//! instead of `f32::NEG_INFINITY` to avoid NaN propagation through
//! softmax when an entire row is masked: `softmax([-1e4, -1e4]) =
//! [0.5, 0.5]` (numerically), whereas `softmax([-inf, -inf])` produces
//! NaN. The chosen value is safely below f32 precision so masked
//! positions exponentiate to ~0.

use rustorch_autograd::Variable;
use rustorch_core::tensor::tensor_impl::Tensor;

/// Large negative value used to flag masked positions in additive
/// attention bias. See module-level documentation for the rationale.
pub const MASK_NEG: f32 = -1.0e4;

/// Build a causal (lower-triangular) additive mask of shape `[seq_len,
/// seq_len]`. `mask[i, j] = 0` if `j <= i` else `MASK_NEG`.
///
/// After adding to scaled-dot-product scores and softmaxing, position
/// `i` only attends to positions `0..=i`.
pub fn causal_mask(seq_len: usize) -> Variable {
    let mut data = vec![0.0_f32; seq_len * seq_len];
    for i in 0..seq_len {
        for j in 0..seq_len {
            if j > i {
                data[i * seq_len + j] = MASK_NEG;
            }
        }
    }
    Variable::new(Tensor::from_vec([seq_len, seq_len], data).expect("causal mask shape"))
}

/// Build a sliding-window additive mask of shape `[seq_len, seq_len]`.
///
/// `mask[i, j] = 0` iff `j <= i AND (i - j) < window`; otherwise
/// `MASK_NEG`. Equivalent to a causal mask intersected with a
/// `window`-length lookback.
///
/// `window` must be `>= 1`. A `window` of `1` keeps only the diagonal
/// (each position attends to itself only). A `window >= seq_len` is
/// equivalent to a plain causal mask.
pub fn sliding_window_mask(seq_len: usize, window: usize) -> Variable {
    assert!(window >= 1, "sliding_window_mask: window must be >= 1");
    let mut data = vec![0.0_f32; seq_len * seq_len];
    for i in 0..seq_len {
        for j in 0..seq_len {
            let inside_causal = j <= i;
            let inside_window = i.saturating_sub(j) < window;
            if !(inside_causal && inside_window) {
                data[i * seq_len + j] = MASK_NEG;
            }
        }
    }
    Variable::new(Tensor::from_vec([seq_len, seq_len], data).expect("sliding mask shape"))
}

/// Convert a boolean keep-mask (where `1.0` means "keep" and `0.0`
/// means "mask") to an additive bias (`0.0` for keep, `MASK_NEG` for
/// mask). Useful when callers have a precomputed binary mask (e.g.
/// padding masks).
///
/// Input must be F32 with values in `{0.0, 1.0}`.
pub fn bool_to_additive(mask: &Variable) -> Variable {
    let t = mask.tensor();
    let s = t.as_slice::<f32>().expect("bool_to_additive: f32 mask");
    let out: Vec<f32> = s
        .iter()
        .map(|&v| if v >= 0.5 { 0.0_f32 } else { MASK_NEG })
        .collect();
    let shape = t.shape().to_vec();
    Variable::new(Tensor::from_vec(shape, out).expect("additive mask shape"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustorch_autograd::ops::softmax;

    /// `causal_mask(4)` produces zeros on/below the diagonal, MASK_NEG
    /// strictly above.
    #[test]
    fn causal_mask_4_is_lower_triangular() {
        let m = causal_mask(4);
        let t = m.tensor();
        let s = t.as_slice::<f32>().unwrap();
        for i in 0..4 {
            for j in 0..4 {
                let v = s[i * 4 + j];
                if j <= i {
                    assert!((v - 0.0).abs() < 1e-6, "expected 0 at ({i},{j}), got {v}");
                } else {
                    assert!(
                        (v - MASK_NEG).abs() < 1e-3,
                        "expected MASK_NEG at ({i},{j}), got {v}"
                    );
                }
            }
        }
    }

    /// Hand-computed truth table for `sliding_window_mask(8, 3)`.
    ///
    /// At row `i`, the kept columns are `max(0, i-2)..=i` (window=3 means
    /// i, i-1, i-2 are visible).
    #[test]
    fn sliding_window_8_3_truth_table() {
        let m = sliding_window_mask(8, 3);
        let t = m.tensor();
        let s = t.as_slice::<f32>().unwrap();
        for i in 0..8 {
            for j in 0..8 {
                let v = s[i * 8 + j];
                let lo = i.saturating_sub(2);
                let in_window = j >= lo && j <= i;
                if in_window {
                    assert!((v - 0.0).abs() < 1e-6, "expected 0 at ({i},{j})");
                } else {
                    assert!(
                        (v - MASK_NEG).abs() < 1e-3,
                        "expected MASK_NEG at ({i},{j})"
                    );
                }
            }
        }
    }

    /// `window >= seq_len` collapses sliding-window to plain causal.
    #[test]
    fn sliding_window_full_equals_causal() {
        let s = sliding_window_mask(5, 100);
        let c = causal_mask(5);
        let s_v = s.tensor();
        let c_v = c.tensor();
        assert_eq!(
            s_v.as_slice::<f32>().unwrap(),
            c_v.as_slice::<f32>().unwrap()
        );
    }

    /// Adding the causal mask to constant scores then softmaxing leaves
    /// non-zero probability mass only on positions `j <= i`. Hand-check
    /// the structure on a small `seq_len = 3` case with constant scores.
    #[test]
    fn causal_mask_zeroes_softmax_above_diagonal() {
        let len = 3;
        let mask = causal_mask(len);
        // Zero scores so the only structure is the mask.
        let scores = Variable::new(Tensor::from_vec([len, len], vec![0.0_f32; len * len]).unwrap());
        let masked = rustorch_autograd::ops::add(&scores, &mask).unwrap();
        let probs = softmax(&masked, 1).unwrap();
        let p = probs.tensor();
        let v = p.as_slice::<f32>().unwrap();
        // Row 0: only column 0 is allowed → prob ≈ 1.0 there, ~0 elsewhere
        assert!((v[0] - 1.0).abs() < 1e-3);
        assert!(v[1].abs() < 1e-3);
        assert!(v[2].abs() < 1e-3);
        // Row 2: columns 0, 1, 2 allowed → roughly uniform 1/3
        for j in 0..len {
            assert!(
                (v[2 * len + j] - 1.0 / len as f32).abs() < 1e-3,
                "row 2 col {j}: {}",
                v[2 * len + j]
            );
        }
    }

    /// Boolean mask (1=keep, 0=mask) round-trips to additive bias.
    #[test]
    fn bool_to_additive_round_trip() {
        let bool_data = vec![1.0_f32, 0.0, 1.0, 1.0];
        let m = Variable::new(Tensor::from_vec([2usize, 2], bool_data).unwrap());
        let add = bool_to_additive(&m);
        let t = add.tensor();
        let s = t.as_slice::<f32>().unwrap();
        assert_eq!(s[0], 0.0);
        assert!((s[1] - MASK_NEG).abs() < 1e-3);
        assert_eq!(s[2], 0.0);
        assert_eq!(s[3], 0.0);
    }

    /// Sliding window size 1 → diagonal only.
    #[test]
    fn sliding_window_1_is_diagonal() {
        let m = sliding_window_mask(4, 1);
        let t = m.tensor();
        let s = t.as_slice::<f32>().unwrap();
        for i in 0..4 {
            for j in 0..4 {
                let v = s[i * 4 + j];
                if i == j {
                    assert!((v - 0.0).abs() < 1e-6);
                } else {
                    assert!((v - MASK_NEG).abs() < 1e-3);
                }
            }
        }
    }
}
