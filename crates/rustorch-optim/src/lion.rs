//! Lion optimizer (Chen et al. 2023, "Symbolic Discovery of Optimization
//! Algorithms"). Sign-of-momentum descent with two betas. Single state
//! tensor per parameter — about half the memory of Adam.
//!
//! Update rule:
//! ```text
//!   c_t = β₁ * m_{t-1} + (1 - β₁) * g       // interim direction
//!   θ_t = θ_{t-1} - lr * (sign(c_t) + wd * θ_{t-1})
//!   m_t = β₂ * m_{t-1} + (1 - β₂) * g       // momentum update AFTER param
//! ```
//! Note: weight decay is decoupled (acts directly on θ, AdamW-style).
//!
//! Recommended defaults (from the paper):
//! - β₁ = 0.9, β₂ = 0.99
//! - lr = 1e-4 (Adam-style lr ÷ ~10 typically works)
//! - wd = same scale as Adam wd

use crate::{write_param_data, Optimizer};
use rustorch_autograd::Variable;

/// Lion optimiser (sign-of-momentum, lower memory than Adam).
pub struct Lion {
    params: Vec<Variable>,
    lr: f32,
    betas: (f32, f32),
    weight_decay: f32,
    /// Per-parameter momentum buffer (None until first step touches it).
    m: Vec<Option<Vec<f32>>>,
}

impl Lion {
    /// Build a Lion optimiser with the given learning rate and the
    /// paper's default betas (0.9, 0.99) and zero weight decay.
    pub fn new(params: Vec<Variable>, lr: f32) -> Self {
        let n = params.len();
        Lion {
            params,
            lr,
            betas: (0.9, 0.99),
            weight_decay: 0.0,
            m: vec![None; n],
        }
    }

    /// Override `(β₁, β₂)`. Defaults are (0.9, 0.99).
    #[must_use]
    pub fn betas(mut self, b1: f32, b2: f32) -> Self {
        self.betas = (b1, b2);
        self
    }

    /// Set decoupled weight decay coefficient.
    #[must_use]
    pub fn weight_decay(mut self, wd: f32) -> Self {
        self.weight_decay = wd;
        self
    }

    /// Borrow the parameter list (read-only).
    pub fn parameters(&self) -> &[Variable] {
        &self.params
    }
}

impl Optimizer for Lion {
    fn step(&mut self) {
        let (b1, b2) = self.betas;
        for (i, param) in self.params.iter().enumerate() {
            let grad = match param.grad() {
                Some(g) => g,
                None => continue,
            };
            let snapshot = param.data_snapshot();
            let p_data: &[f32] = snapshot.as_slice::<f32>().expect("Lion: F32 only");
            let g_data: &[f32] = grad.as_slice::<f32>().expect("Lion: F32 grads only");

            let mut m_buf = self.m[i]
                .take()
                .unwrap_or_else(|| vec![0.0_f32; p_data.len()]);

            let mut new = Vec::with_capacity(p_data.len());
            for k in 0..p_data.len() {
                // c = β1 * m + (1-β1) * g
                let c = b1 * m_buf[k] + (1.0 - b1) * g_data[k];
                let sign_c = if c > 0.0 {
                    1.0_f32
                } else if c < 0.0 {
                    -1.0_f32
                } else {
                    0.0_f32
                };
                // θ -= lr * (sign(c) + wd * θ)
                let mut p_new = p_data[k] - self.lr * sign_c;
                if self.weight_decay > 0.0 {
                    p_new -= self.lr * self.weight_decay * p_data[k];
                }
                new.push(p_new);

                // m = β2 * m + (1-β2) * g (post-update state evolution)
                m_buf[k] = b2 * m_buf[k] + (1.0 - b2) * g_data[k];
            }

            write_param_data(param, new);
            self.m[i] = Some(m_buf);
        }
    }

    fn zero_grad(&mut self) {
        for param in &self.params {
            param.zero_grad();
        }
    }

    fn lr(&self) -> f32 {
        self.lr
    }

    fn set_lr(&mut self, lr: f32) {
        self.lr = lr;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustorch_autograd::{backward, ops, Variable};
    use rustorch_core::tensor::tensor_impl::Tensor;

    #[test]
    fn lion_lr_zero_leaves_params_unchanged() {
        let p = Variable::leaf(Tensor::from_vec([3], vec![1.0_f32, 2.0, 3.0]).unwrap());
        // Manually fake a grad via a forward+backward.
        let s = ops::sum(&ops::mul(&p, &p).unwrap()).unwrap();
        backward(&s, None).unwrap();
        let before: Vec<f32> = p.tensor().as_slice::<f32>().unwrap().to_vec();

        let mut opt = Lion::new(vec![p.clone()], 0.0);
        opt.step();

        let after: Vec<f32> = p.tensor().as_slice::<f32>().unwrap().to_vec();
        assert_eq!(before, after);
    }

    #[test]
    fn lion_one_step_takes_a_unit_step_via_sign() {
        // y = (x - target)² ; grad = 2(x - target)
        // For x=[1,2,3], target=[1,1,1] → grad ∝ (0, 2, 4)
        // Lion uses sign(grad) so update = -lr * (0, 1, 1) (modulo first-step
        // momentum mixing). With β1=0.9, c = 0.1 * g; sign(c) = sign(g).
        let x = Variable::leaf(Tensor::from_vec([3], vec![1.0_f32, 2.0, 3.0]).unwrap());
        let target = Variable::new(Tensor::from_vec([3], vec![1.0_f32, 1.0, 1.0]).unwrap());
        let diff = ops::sub(&x, &target).unwrap();
        let s = ops::sum(&ops::mul(&diff, &diff).unwrap()).unwrap();
        backward(&s, None).unwrap();

        let mut opt = Lion::new(vec![x.clone()], 0.1);
        opt.step();

        // Expected update on first step: lr * sign of (β1 * 0 + (1-β1) * grad)
        // = lr * sign(grad). grad on lane 0 is 0 → no update. Lanes 1,2 grad>0 → -lr.
        let after = x.tensor().as_slice::<f32>().unwrap().to_vec();
        assert!(
            (after[0] - 1.0).abs() < 1e-6,
            "lane 0 (zero grad) unchanged, got {}",
            after[0]
        );
        assert!(
            (after[1] - 1.9).abs() < 1e-6,
            "lane 1: 2.0 - 0.1 * sign(grad)=1 = 1.9, got {}",
            after[1]
        );
        assert!(
            (after[2] - 2.9).abs() < 1e-6,
            "lane 2: 3.0 - 0.1 * sign(grad)=1 = 2.9, got {}",
            after[2]
        );
    }

    #[test]
    fn lion_state_dict_uses_one_buffer_per_param() {
        // Indirectly: Lion only allocates `m`, no `v`. Verify by inspecting
        // the `m` field length after a step.
        let p = Variable::leaf(Tensor::from_vec([3], vec![1.0_f32, 2.0, 3.0]).unwrap());
        let s = ops::sum(&ops::mul(&p, &p).unwrap()).unwrap();
        backward(&s, None).unwrap();
        let mut opt = Lion::new(vec![p.clone()], 0.01);
        opt.step();
        assert_eq!(opt.m.len(), 1);
        assert!(opt.m[0].is_some(), "m buffer initialised after step");
    }

    #[test]
    fn lion_weight_decay_pulls_toward_zero() {
        // With wd>0 and zero grad, params decay toward 0 each step.
        let p = Variable::leaf(Tensor::from_vec([1], vec![1.0_f32]).unwrap());
        // Force a grad without affecting it: build a no-op forward/backward.
        let s = ops::sum(&ops::mul(&p, &p).unwrap()).unwrap();
        backward(&s, None).unwrap();
        let mut opt = Lion::new(vec![p.clone()], 0.1).weight_decay(0.5);
        let before = p.tensor().as_slice::<f32>().unwrap()[0];
        opt.step();
        let after = p.tensor().as_slice::<f32>().unwrap()[0];
        // grad is 2*p = 2.0 > 0 → sign=1 → step = -lr*(1 + wd*p) = -0.1*(1 + 0.5*1) = -0.15
        // expected after = 1.0 - 0.15 = 0.85
        assert!(
            (after - 0.85).abs() < 1e-5,
            "before={before}, after={after}"
        );
    }

    #[test]
    fn lion_set_lr_changes_step_size() {
        let p = Variable::leaf(Tensor::from_vec([1], vec![5.0_f32]).unwrap());
        let s = ops::sum(&ops::mul(&p, &p).unwrap()).unwrap();
        backward(&s, None).unwrap();
        let mut opt = Lion::new(vec![p.clone()], 0.01);
        assert!((opt.lr() - 0.01).abs() < 1e-7);
        opt.set_lr(0.5);
        assert!((opt.lr() - 0.5).abs() < 1e-7);
    }
}
