//! RMSprop optimizer (P1.7).
//!
//! Update rule:
//! ```text
//!   v_t = α * v_{t-1} + (1 - α) * g²
//!   θ_t = θ_{t-1} - lr * g / (sqrt(v_t) + ε)
//! ```
//! where `α` (default 0.99) is the running-average coefficient.

use crate::{write_param_data, Optimizer};
use rustorch_autograd::Variable;

/// RMSprop (Hinton 2012).
pub struct RMSprop {
    params: Vec<Variable>,
    lr: f32,
    alpha: f32,
    eps: f32,
    weight_decay: f32,
    v: Vec<Option<Vec<f32>>>,
}

impl RMSprop {
    /// Build with the given learning rate (defaults: α=0.99, ε=1e-8).
    pub fn new(params: Vec<Variable>, lr: f32) -> Self {
        let n = params.len();
        RMSprop {
            params,
            lr,
            alpha: 0.99,
            eps: 1e-8,
            weight_decay: 0.0,
            v: vec![None; n],
        }
    }

    /// Override α.
    #[must_use]
    pub fn alpha(mut self, alpha: f32) -> Self {
        self.alpha = alpha;
        self
    }

    /// Override ε.
    #[must_use]
    pub fn eps(mut self, eps: f32) -> Self {
        self.eps = eps;
        self
    }

    /// Set L2 weight decay (added to gradient pre-update).
    #[must_use]
    pub fn weight_decay(mut self, wd: f32) -> Self {
        self.weight_decay = wd;
        self
    }

    /// Borrow the parameter list.
    pub fn parameters(&self) -> &[Variable] {
        &self.params
    }
}

impl Optimizer for RMSprop {
    fn step(&mut self) {
        for (i, param) in self.params.iter().enumerate() {
            let grad = match param.grad() {
                Some(g) => g,
                None => continue,
            };
            let snapshot = param.data_snapshot();
            let p_data: &[f32] = snapshot.as_slice::<f32>().expect("RMSprop: F32 only");
            let g_data: &[f32] = grad.as_slice::<f32>().expect("RMSprop: F32 only");

            let mut v_buf = self.v[i]
                .take()
                .unwrap_or_else(|| vec![0.0_f32; p_data.len()]);

            let mut new = Vec::with_capacity(p_data.len());
            for k in 0..p_data.len() {
                let g = if self.weight_decay > 0.0 {
                    g_data[k] + self.weight_decay * p_data[k]
                } else {
                    g_data[k]
                };
                v_buf[k] = self.alpha * v_buf[k] + (1.0 - self.alpha) * g * g;
                let p_new = p_data[k] - self.lr * g / (v_buf[k].sqrt() + self.eps);
                new.push(p_new);
            }
            write_param_data(param, new);
            self.v[i] = Some(v_buf);
        }
    }
    fn zero_grad(&mut self) {
        for p in &self.params {
            p.zero_grad();
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
    fn rmsprop_lr_zero_no_change() {
        let p = Variable::leaf(Tensor::from_vec([3], vec![1.0_f32, 2.0, 3.0]).unwrap());
        let s = ops::sum(&ops::mul(&p, &p).unwrap()).unwrap();
        backward(&s, None).unwrap();
        let before: Vec<f32> = p.tensor().as_slice::<f32>().unwrap().to_vec();
        let mut opt = RMSprop::new(vec![p.clone()], 0.0);
        opt.step();
        let after: Vec<f32> = p.tensor().as_slice::<f32>().unwrap().to_vec();
        assert_eq!(before, after);
    }

    #[test]
    fn rmsprop_descends_quadratic_loss() {
        // f(x) = sum(x²) ; minimum at 0. RMSprop is conservative; needs
        // more steps than Adam at the same lr.
        let p = Variable::leaf(Tensor::from_vec([3], vec![5.0_f32, -3.0, 2.0]).unwrap());
        let mut opt = RMSprop::new(vec![p.clone()], 0.1);
        for _ in 0..1000 {
            opt.zero_grad();
            let s = ops::sum(&ops::mul(&p, &p).unwrap()).unwrap();
            backward(&s, None).unwrap();
            opt.step();
        }
        let v = p.tensor().as_slice::<f32>().unwrap().to_vec();
        for &x in &v {
            assert!(x.abs() < 0.5, "got {x} after 1000 steps");
        }
    }
}
