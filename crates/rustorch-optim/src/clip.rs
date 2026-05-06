//! Gradient clipping helpers — analogous to
//! `torch.nn.utils.clip_grad_norm_` and friends.
//!
//! Used in trainers to prevent gradient explosion. Call between
//! `backward()` and `optimizer.step()`.

use rustorch_autograd::Variable;
use rustorch_core::tensor::tensor_impl::Tensor;

/// Clip the **global** L2 norm of gradients across `parameters` to
/// `max_norm`. Mutates each parameter's grad slot in place by uniform
/// scaling.
///
/// Returns the original (pre-clip) global L2 norm — useful for
/// logging.
///
/// Equivalent to `torch.nn.utils.clip_grad_norm_(parameters, max_norm)`.
///
/// Behaviour:
/// - If `global_norm <= max_norm`, gradients are left untouched.
/// - If `global_norm > max_norm`, every grad is multiplied by
///   `max_norm / global_norm`.
/// - Parameters whose grad slot is `None` are skipped without panic
///   (they don't contribute to the global norm).
/// - The original global norm is always returned, regardless of
///   whether clipping fired.
pub fn clip_grad_norm_(parameters: &[Variable], max_norm: f32) -> f32 {
    // 1) Accumulate sum of squares across all parameter grads.
    let mut total_sq: f64 = 0.0;
    for p in parameters {
        if let Some(g) = p.grad() {
            let s = g.as_slice::<f32>().expect("grad must be f32");
            for &v in s {
                total_sq += (v as f64) * (v as f64);
            }
        }
    }
    let global_norm = total_sq.sqrt() as f32;

    // 2) If clipping needed, scale every grad in place.
    // P3.Y plan, Phase D: preserve the grad's device tag so subsequent
    // ops (optimiser.step) keep dispatching to the correct backend.
    if global_norm > max_norm && global_norm > 0.0 {
        let scale = max_norm / global_norm;
        for p in parameters {
            if let Some(g) = p.grad() {
                let scaled: Vec<f32> = g
                    .as_slice::<f32>()
                    .expect("grad must be f32")
                    .iter()
                    .map(|v| v * scale)
                    .collect();
                let shape = g.shape().to_vec();
                let new_g = Tensor::from_vec(shape, scaled)
                    .expect("clipped grad shape")
                    .with_device(g.device());
                p.set_grad(Some(new_g));
            }
        }
    }
    global_norm
}

/// Per-parameter L2 clip — each grad is independently clipped to
/// `max_norm`. Returns the per-parameter pre-clip norms in the same
/// order as `parameters` (`0.0` if a grad is `None`).
///
/// Use [`clip_grad_norm_`] for the standard global-norm path.
pub fn clip_grad_norm_per_param_(parameters: &[Variable], max_norm: f32) -> Vec<f32> {
    let mut norms = Vec::with_capacity(parameters.len());
    for p in parameters {
        let Some(g) = p.grad() else {
            norms.push(0.0);
            continue;
        };
        let s = g.as_slice::<f32>().expect("grad must be f32");
        let n_sq: f64 = s.iter().map(|&v| (v as f64) * (v as f64)).sum();
        let n = n_sq.sqrt() as f32;
        norms.push(n);
        if n > max_norm && n > 0.0 {
            let scale = max_norm / n;
            let scaled: Vec<f32> = s.iter().map(|v| v * scale).collect();
            let shape = g.shape().to_vec();
            // P3.Y plan, Phase D: preserve grad device tag.
            let new_g = Tensor::from_vec(shape, scaled)
                .expect("clipped grad shape")
                .with_device(g.device());
            p.set_grad(Some(new_g));
        }
    }
    norms
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fresh_param_with_grad(shape: Vec<usize>, data: Vec<f32>) -> Variable {
        let p = Variable::leaf(Tensor::from_vec(shape.clone(), vec![0.0_f32; data.len()]).unwrap());
        let g = Tensor::from_vec(shape, data).unwrap();
        p.set_grad(Some(g));
        p
    }

    /// Single param: grad of L2 norm 10 clipped to 1.0 → final norm 1.0,
    /// returned norm equals 10.0.
    #[test]
    fn single_param_clips_to_max_norm() {
        let p = fresh_param_with_grad(vec![2], vec![6.0, 8.0]); // norm = 10
        let original = clip_grad_norm_(std::slice::from_ref(&p), 1.0);
        assert!((original - 10.0).abs() < 1e-5);
        let g = p.grad().unwrap();
        let s = g.as_slice::<f32>().unwrap();
        let new_norm = (s[0].powi(2) + s[1].powi(2)).sqrt();
        assert!((new_norm - 1.0).abs() < 1e-5);
        // Direction preserved: ratio s[1]/s[0] should still be 8/6.
        assert!((s[1] / s[0] - 8.0 / 6.0).abs() < 1e-5);
    }

    /// Multi-param: scaling is uniform — sum of squares of clipped
    /// grads should equal `max_norm²`.
    #[test]
    fn multi_param_global_norm_matches_max() {
        let p1 = fresh_param_with_grad(vec![2], vec![3.0, 0.0]); // norm 3
        let p2 = fresh_param_with_grad(vec![2], vec![0.0, 4.0]); // norm 4
                                                                 // Global norm = sqrt(9 + 16) = 5
        let original = clip_grad_norm_(&[p1.clone(), p2.clone()], 1.0);
        assert!((original - 5.0).abs() < 1e-5);

        let g1 = p1.grad().unwrap();
        let g2 = p2.grad().unwrap();
        let s1 = g1.as_slice::<f32>().unwrap();
        let s2 = g2.as_slice::<f32>().unwrap();
        let total_sq = s1[0].powi(2) + s1[1].powi(2) + s2[0].powi(2) + s2[1].powi(2);
        assert!(
            (total_sq - 1.0).abs() < 1e-5,
            "total² should be 1.0, got {}",
            total_sq
        );
    }

    /// No-op when global norm <= max_norm: grads must remain identical
    /// and the returned norm is the unchanged original.
    #[test]
    fn noop_when_under_max_norm() {
        let p = fresh_param_with_grad(vec![2], vec![0.3, 0.4]); // norm 0.5
        let before = p.grad().unwrap().as_slice::<f32>().unwrap().to_vec();
        let original = clip_grad_norm_(std::slice::from_ref(&p), 1.0);
        assert!((original - 0.5).abs() < 1e-5);
        let after = p.grad().unwrap().as_slice::<f32>().unwrap().to_vec();
        assert_eq!(before, after);
    }

    /// Parameters with no grad must be skipped silently.
    #[test]
    fn none_grad_params_are_skipped() {
        let p_with = fresh_param_with_grad(vec![2], vec![3.0, 4.0]); // norm 5
        let p_without = Variable::leaf(Tensor::from_vec([3], vec![0.0_f32; 3]).unwrap());
        // Should not panic. Global norm only counts the one param with a grad.
        let original = clip_grad_norm_(&[p_with.clone(), p_without.clone()], 1.0);
        assert!((original - 5.0).abs() < 1e-5);
        // The None-grad param's grad slot stays None.
        assert!(p_without.grad().is_none());
    }

    /// Per-param variant: each grad is clipped against `max_norm`
    /// independently, and the returned norms preserve order.
    #[test]
    fn per_param_clips_each_independently() {
        let p_big = fresh_param_with_grad(vec![2], vec![6.0, 8.0]); // norm 10
        let p_small = fresh_param_with_grad(vec![2], vec![0.3, 0.4]); // norm 0.5
        let norms = clip_grad_norm_per_param_(&[p_big.clone(), p_small.clone()], 1.0);
        assert!((norms[0] - 10.0).abs() < 1e-5);
        assert!((norms[1] - 0.5).abs() < 1e-5);
        // Big one is clipped to norm 1; small one is untouched.
        let g1 = p_big.grad().unwrap();
        let n1: f32 = g1
            .as_slice::<f32>()
            .unwrap()
            .iter()
            .map(|v| v.powi(2))
            .sum::<f32>()
            .sqrt();
        assert!((n1 - 1.0).abs() < 1e-5);
        let g2 = p_small.grad().unwrap();
        let n2: f32 = g2
            .as_slice::<f32>()
            .unwrap()
            .iter()
            .map(|v| v.powi(2))
            .sum::<f32>()
            .sqrt();
        assert!((n2 - 0.5).abs() < 1e-5);
    }
}
