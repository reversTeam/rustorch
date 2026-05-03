//! Gradient unbroadcast helper.
//!
//! Element-wise ops (`add`, `sub`, `mul`, `div`, …) support PyTorch-style
//! broadcast on the forward path: `[B, T, 2, 1] * [B, T, 2, 256]` → output
//! `[B, T, 2, 256]`. The chain rule then sends a gradient of the **output**
//! shape back to each input. For the small input that was broadcast, that
//! gradient must be **reduced** (summed) along the axes where the small
//! input had size 1, otherwise the upstream `AccumulateGrad` slot rejects
//! it for shape mismatch (or accumulates the wrong shape silently).
//!
//! This is the canonical PyTorch `Tensor::sum_to_size(target_shape)`
//! algorithm, adapted to the rustorch CPU backend.
//!
//! Discovered while implementing `l2_normalize` (PR 4 of the transformer
//! building blocks plan). Tracked in note `7a32584a` (gotcha) and
//! addressed here as the generic fix referenced by that note.

use rustorch_core::tensor::tensor_impl::Tensor;
use rustorch_cpu::cpu_backend::cpu_backend;

/// Sum the upstream gradient down to `target_shape` so it matches the
/// shape of the input that was broadcast on the forward path.
///
/// Implements PyTorch's right-aligned broadcast rules:
/// 1. Strip leading dims when `grad` has more dims than `target_shape`
///    (sum them with `keepdim=false`).
/// 2. For each remaining axis, sum (with `keepdim=true`) when
///    `target_shape[i] == 1` and `grad.shape()[aligned_i] > 1`.
///
/// Returns `grad` unchanged when no reduction is needed (the common case
/// where shapes already match).
///
/// # Panics
///
/// Panics if the shapes are not broadcast-compatible — the forward path
/// would have rejected them in that case, so this is an invariant.
pub fn unbroadcast_to(grad: &Tensor, target_shape: &[usize]) -> Tensor {
    let g_shape = grad.shape();

    // Fast path: shapes already match.
    if g_shape == target_shape {
        return grad.clone();
    }

    // Step 1 — Strip leading dims when grad has more rank than target.
    let leading = g_shape.len().saturating_sub(target_shape.len());
    let mut current = if leading > 0 {
        let dims: Vec<usize> = (0..leading).collect();
        cpu_backend()
            .sum_dim(grad, &dims, false)
            .expect("unbroadcast_to: sum_dim leading axes")
    } else {
        grad.clone()
    };

    // Step 2 — Sum axes where target had size 1 but current is larger.
    let cur_shape = current.shape().to_vec();
    debug_assert_eq!(
        cur_shape.len(),
        target_shape.len(),
        "rank mismatch after leading reduction"
    );
    let mut axes_to_sum: Vec<usize> = Vec::new();
    for (i, (&c, &t)) in cur_shape.iter().zip(target_shape.iter()).enumerate() {
        if t == 1 && c > 1 {
            axes_to_sum.push(i);
        } else if t != 1 && t != c {
            panic!(
                "unbroadcast_to: incompatible shapes grad={:?} target={:?} at axis {}",
                g_shape, target_shape, i
            );
        }
    }
    if !axes_to_sum.is_empty() {
        current = cpu_backend()
            .sum_dim(&current, &axes_to_sum, true)
            .expect("unbroadcast_to: sum_dim broadcast axes");
    }

    debug_assert_eq!(
        current.shape(),
        target_shape,
        "unbroadcast_to: result shape mismatch"
    );
    current
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_reduction_when_shapes_match() {
        let g = Tensor::from_vec([2usize, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let r = unbroadcast_to(&g, &[2, 3]);
        assert_eq!(r.shape(), &[2, 3]);
        assert_eq!(
            r.as_slice::<f32>().unwrap(),
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        );
    }

    #[test]
    fn sum_trailing_size_one_axis_with_keepdim() {
        // grad [2, 3] -> target [2, 1] : sum over axis 1 keepdim.
        let g = Tensor::from_vec([2usize, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let r = unbroadcast_to(&g, &[2, 1]);
        assert_eq!(r.shape(), &[2, 1]);
        assert_eq!(r.as_slice::<f32>().unwrap(), &[6.0, 15.0]);
    }

    #[test]
    fn sum_leading_extra_dim() {
        // grad [4, 3] -> target [3] : sum over axis 0 with keepdim=false.
        let g = Tensor::from_vec(
            [4usize, 3],
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
        )
        .unwrap();
        let r = unbroadcast_to(&g, &[3]);
        assert_eq!(r.shape(), &[3]);
        // sum of columns: 1+4+7+10=22, 2+5+8+11=26, 3+6+9+12=30
        assert_eq!(r.as_slice::<f32>().unwrap(), &[22.0, 26.0, 30.0]);
    }

    #[test]
    fn user_reported_bug_shape_b_t_2_1_vs_b_t_2_256() {
        // The exact case the user reported:
        //   mul([B, T, 2, 1], [B, T, 2, 256]) -> [B, T, 2, 256]
        // Backward upstream grad has shape [B, T, 2, 256].
        // Path to lhs needs to be reduced to [B, T, 2, 1] (sum over axis 3 keepdim).
        let b = 2usize;
        let t = 3usize;
        let last = 8usize; // small for the test, real case is 256
        let total = b * t * 2 * last;
        let data: Vec<f32> = (0..total).map(|i| (i + 1) as f32).collect();
        let g = Tensor::from_vec([b, t, 2, last], data.clone()).unwrap();
        let r = unbroadcast_to(&g, &[b, t, 2, 1]);
        assert_eq!(r.shape(), &[b, t, 2, 1]);
        // For each [b, t, 2] slot, expect sum of `last` consecutive entries.
        let r_data = r.as_slice::<f32>().unwrap();
        for (i, &got) in r_data.iter().enumerate() {
            let expected: f32 = (0..last).map(|j| (i * last + j + 1) as f32).sum();
            assert_eq!(got, expected, "slot {i} expected sum {expected} got {got}");
        }
    }

    #[test]
    fn combined_leading_and_size_one_reductions() {
        // grad [4, 2, 3] -> target [2, 1]:
        //   1) drop leading axis 0 (sum_dim 0 keepdim=false) -> [2, 3]
        //   2) sum axis 1 keepdim=true -> [2, 1]
        let g = Tensor::from_vec(
            [4usize, 2, 3],
            (1..=24).map(|x| x as f32).collect::<Vec<_>>(),
        )
        .unwrap();
        let r = unbroadcast_to(&g, &[2, 1]);
        assert_eq!(r.shape(), &[2, 1]);
    }
}
