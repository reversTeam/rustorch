//! Adagrad optimizer (P1.7).
//!
//! Update rule:
//! ```text
//!   acc_t = acc_{t-1} + g²
//!   θ_t = θ_{t-1} - lr * g / (sqrt(acc_t) + ε)
//! ```
//! Adapts the learning rate per-parameter using the accumulated
//! squared gradient. Pioneered by Duchi et al. (2011).

use crate::{write_param_data, Optimizer};
use rustorch_autograd::Variable;

/// Adagrad (Duchi et al. 2011).
pub struct Adagrad {
    params: Vec<Variable>,
    lr: f32,
    eps: f32,
    weight_decay: f32,
    initial_accumulator: f32,
    acc: Vec<Option<Vec<f32>>>,
}

impl Adagrad {
    /// Build with the given learning rate. Defaults: ε=1e-10,
    /// initial_accumulator=0.
    pub fn new(params: Vec<Variable>, lr: f32) -> Self {
        let n = params.len();
        Adagrad {
            params,
            lr,
            eps: 1e-10,
            weight_decay: 0.0,
            initial_accumulator: 0.0,
            acc: vec![None; n],
        }
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

    /// Override the initial value of the accumulator (PyTorch defaults
    /// this to 0; some setups use a small positive number).
    #[must_use]
    pub fn initial_accumulator(mut self, c: f32) -> Self {
        self.initial_accumulator = c;
        self
    }

    /// Borrow the parameter list.
    pub fn parameters(&self) -> &[Variable] {
        &self.params
    }
}

impl Optimizer for Adagrad {
    fn step(&mut self) {
        for (i, param) in self.params.iter().enumerate() {
            let grad = match param.grad() {
                Some(g) => g,
                None => continue,
            };
            let snapshot = param.data_snapshot();
            let p_data: &[f32] = snapshot.as_slice::<f32>().expect("Adagrad: F32 only");
            let g_data: &[f32] = grad.as_slice::<f32>().expect("Adagrad: F32 only");

            let mut acc_buf = self.acc[i]
                .take()
                .unwrap_or_else(|| vec![self.initial_accumulator; p_data.len()]);

            let mut new = Vec::with_capacity(p_data.len());
            for k in 0..p_data.len() {
                let g = if self.weight_decay > 0.0 {
                    g_data[k] + self.weight_decay * p_data[k]
                } else {
                    g_data[k]
                };
                acc_buf[k] += g * g;
                let p_new = p_data[k] - self.lr * g / (acc_buf[k].sqrt() + self.eps);
                new.push(p_new);
            }
            write_param_data(param, new);
            self.acc[i] = Some(acc_buf);
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
    fn adagrad_descends_quadratic_loss() {
        let p = Variable::leaf(Tensor::from_vec([3], vec![5.0_f32, -3.0, 2.0]).unwrap());
        let mut opt = Adagrad::new(vec![p.clone()], 1.0);
        for _ in 0..500 {
            opt.zero_grad();
            let s = ops::sum(&ops::mul(&p, &p).unwrap()).unwrap();
            backward(&s, None).unwrap();
            opt.step();
        }
        let v = p.tensor().as_slice::<f32>().unwrap().to_vec();
        for &x in &v {
            assert!(x.abs() < 0.5, "got {x} after 500 steps");
        }
    }

    #[test]
    fn adagrad_accumulator_monotonically_increases() {
        let p = Variable::leaf(Tensor::from_vec([1], vec![1.0_f32]).unwrap());
        let mut opt = Adagrad::new(vec![p.clone()], 0.01);

        // Two steps should increase accumulator monotonically.
        opt.zero_grad();
        let s = ops::sum(&ops::mul(&p, &p).unwrap()).unwrap();
        backward(&s, None).unwrap();
        opt.step();
        let acc1 = opt.acc[0].as_ref().unwrap()[0];

        opt.zero_grad();
        let s = ops::sum(&ops::mul(&p, &p).unwrap()).unwrap();
        backward(&s, None).unwrap();
        opt.step();
        let acc2 = opt.acc[0].as_ref().unwrap()[0];

        assert!(acc2 > acc1, "accumulator should increase: {acc1} -> {acc2}");
    }
}
