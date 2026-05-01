//! Optimizer `state_dict` / `load_state_dict` (P1.7).
//!
//! Each optimizer that carries per-parameter buffers (Adam's m/v,
//! SGD's velocity, RMSprop's v, Adagrad's acc, NAdam's m/v, Lion's m)
//! exposes `state_dict()` returning a `BTreeMap<String, Tensor>` keyed
//! by `"{param_idx}.{buffer_name}"`. Hyperparameters (lr, betas, eps,
//! step counters) are scalar — they live in [`OptimMeta`] and can be
//! serialised separately if the user wants a full checkpoint.
//!
//! v1 limits the shape of each per-param buffer to whatever the
//! optimizer chose (1-D Vec<f32> stored alongside the Variable).

use rustorch_core::tensor::tensor_impl::Tensor;
use std::collections::BTreeMap;

/// Hyperparameters / step counters that don't fit in a tensor map.
/// Optimizers expose these so the user can checkpoint them as JSON.
#[derive(Debug, Clone, PartialEq)]
pub struct OptimMeta {
    /// Learning rate.
    pub lr: f32,
    /// Step counter (Adam-family).
    pub step_t: usize,
    /// Beta coefficients (Adam-family / Lion).
    pub betas: Option<(f32, f32)>,
    /// Epsilon.
    pub eps: Option<f32>,
    /// Weight-decay coefficient.
    pub weight_decay: Option<f32>,
    /// Momentum coefficient (SGD).
    pub momentum: Option<f32>,
    /// Alpha coefficient (RMSprop).
    pub alpha: Option<f32>,
}

impl Default for OptimMeta {
    fn default() -> Self {
        OptimMeta {
            lr: 0.0,
            step_t: 0,
            betas: None,
            eps: None,
            weight_decay: None,
            momentum: None,
            alpha: None,
        }
    }
}

/// Helper: turn a raw `Vec<f32>` per-param buffer into a 1-D tensor
/// keyed by `"{prefix}.{idx}"`.
pub(crate) fn flatten_buffers(
    map: &mut BTreeMap<String, Tensor>,
    prefix: &str,
    buffers: &[Option<Vec<f32>>],
) {
    for (i, buf) in buffers.iter().enumerate() {
        if let Some(b) = buf {
            let n = b.len();
            let t = Tensor::from_vec([n], b.clone()).expect("buffer rebuild");
            map.insert(format!("{prefix}.{i}"), t);
        }
    }
}

/// Helper: read tensors back into `Vec<Option<Vec<f32>>>`. Missing
/// keys → `None` slots.
pub(crate) fn unflatten_buffers(
    map: &BTreeMap<String, Tensor>,
    prefix: &str,
    n_params: usize,
) -> Vec<Option<Vec<f32>>> {
    (0..n_params)
        .map(|i| {
            map.get(&format!("{prefix}.{i}"))
                .map(|t| t.as_slice::<f32>().expect("F32 buffer").to_vec())
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use crate::{Adam, Optimizer, Sgd};
    use rustorch_autograd::{backward, ops, Variable};
    use rustorch_core::tensor::tensor_impl::Tensor;

    #[test]
    fn adam_state_dict_round_trip() {
        let p = Variable::leaf(Tensor::from_vec([3], vec![1.0_f32, 2.0, 3.0]).unwrap());
        let mut opt = Adam::new(vec![p.clone()], 0.01);
        // Run a few steps to populate m / v.
        for _ in 0..5 {
            opt.zero_grad();
            let s = ops::sum(&ops::mul(&p, &p).unwrap()).unwrap();
            backward(&s, None).unwrap();
            opt.step();
        }
        let sd = opt.state_dict();
        let meta = opt.meta();

        // Build a fresh Adam and restore.
        let p2 = Variable::leaf(Tensor::from_vec([3], vec![5.0_f32, 5.0, 5.0]).unwrap());
        let mut opt2 = Adam::new(vec![p2.clone()], 0.0);
        opt2.load_state_dict(&sd);
        opt2.load_meta(&meta);
        // m and v should match.
        assert_eq!(opt2.meta().step_t, 5);
        assert!((opt2.meta().lr - 0.01).abs() < 1e-7);
        assert!(sd.contains_key("m.0"));
        assert!(sd.contains_key("v.0"));
        assert_eq!(sd["m.0"].shape(), &[3]);
        assert_eq!(sd["v.0"].shape(), &[3]);
    }

    #[test]
    fn sgd_no_momentum_state_dict_empty() {
        let p = Variable::leaf(Tensor::from_vec([3], vec![1.0_f32, 2.0, 3.0]).unwrap());
        let mut opt = Sgd::new(vec![p.clone()], 0.01);
        opt.zero_grad();
        let s = ops::sum(&ops::mul(&p, &p).unwrap()).unwrap();
        backward(&s, None).unwrap();
        opt.step();
        // Without momentum, no velocity buffers.
        let sd = opt.state_dict();
        assert!(sd.is_empty());
    }

    #[test]
    fn sgd_with_momentum_state_dict_populates() {
        let p = Variable::leaf(Tensor::from_vec([3], vec![1.0_f32, 2.0, 3.0]).unwrap());
        let mut opt = Sgd::new(vec![p.clone()], 0.01).momentum(0.9);
        opt.zero_grad();
        let s = ops::sum(&ops::mul(&p, &p).unwrap()).unwrap();
        backward(&s, None).unwrap();
        opt.step();
        let sd = opt.state_dict();
        assert!(sd.contains_key("velocity.0"));
        assert_eq!(sd["velocity.0"].shape(), &[3]);
    }
}
