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
//! algorithm.
//!
//! P3.Y plan, Phase C1: dispatches via `pick_backend(device)` so the
//! reduction runs on the right backend (Cpu or Wgpu) — keyed on the
//! `device` field stored in each `*Backward` node.
//!
//! Discovered while implementing `l2_normalize` (PR 4 of the transformer
//! building blocks plan). Tracked in note `7a32584a` (gotcha) and
//! addressed here as the generic fix referenced by that note.

use crate::dispatch::pick_backend;
use rustorch_core::tensor::device::Device;
use rustorch_core::tensor::tensor_impl::Tensor;

/// Sum the upstream gradient down to `target_shape` so it matches the
/// shape of the input that was broadcast on the forward path.
///
/// The reduction runs on the backend selected by `device`, so a backward
/// over a Wgpu Variable stays GPU-resident (once the `wgpu` feature is on
/// and Wgpu kernels for `sum_dim`/`reshape` exist; otherwise this falls
/// through to whatever the Backend trait's `unbroadcast_to` returns —
/// `Unsupported` until those land).
///
/// Implements PyTorch's right-aligned broadcast rules via the
/// `Backend::unbroadcast_to` trait method (see `rustorch-cpu/src/backend.rs`):
/// 1. Pad `target_shape` on the left with 1s to match `grad.ndim()`.
/// 2. Sum every axis where the padded target is 1 but `grad.shape[axis]` > 1.
/// 3. Reshape back to `target_shape`.
///
/// # Panics
///
/// Panics if the shapes are not broadcast-compatible — the forward path
/// would have rejected them in that case, so this is an invariant.
pub fn unbroadcast_to(device: Device, grad: &Tensor, target_shape: &[usize]) -> Tensor {
    pick_backend(device)
        .unbroadcast_to(grad, target_shape)
        .expect("unbroadcast_to never fails on broadcast-compatible shapes")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_reduction_when_shapes_match() {
        let g = Tensor::from_vec([2usize, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let r = unbroadcast_to(Device::Cpu, &g, &[2, 3]);
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
        let r = unbroadcast_to(Device::Cpu, &g, &[2, 1]);
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
        let r = unbroadcast_to(Device::Cpu, &g, &[3]);
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
        let r = unbroadcast_to(Device::Cpu, &g, &[b, t, 2, 1]);
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
        let r = unbroadcast_to(Device::Cpu, &g, &[2, 1]);
        assert_eq!(r.shape(), &[2, 1]);
    }
}
