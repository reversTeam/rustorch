//! NAdam optimizer (P1.7).
//!
//! Adam with Nesterov-style momentum (Dozat 2016).
//! Update rule:
//! ```text
//!   m_t = β1 * m_{t-1} + (1 - β1) * g
//!   v_t = β2 * v_{t-1} + (1 - β2) * g²
//!   m̂_t = m_t / (1 - β1^t)
//!   v̂_t = v_t / (1 - β2^t)
//!   m̂'_t = β1 * m̂_t + (1 - β1) * g / (1 - β1^t)   ← Nesterov correction
//!   θ_t = θ_{t-1} - lr * m̂'_t / (sqrt(v̂_t) + ε)
//! ```

use crate::{write_param_data, Optimizer};
use rustorch_autograd::Variable;

/// NAdam (Adam + Nesterov, Dozat 2016).
pub struct NAdam {
    params: Vec<Variable>,
    lr: f32,
    betas: (f32, f32),
    eps: f32,
    weight_decay: f32,
    step_t: usize,
    m: Vec<Option<Vec<f32>>>,
    v: Vec<Option<Vec<f32>>>,
}

impl NAdam {
    /// Build with the given learning rate.
    pub fn new(params: Vec<Variable>, lr: f32) -> Self {
        let n = params.len();
        NAdam {
            params,
            lr,
            betas: (0.9, 0.999),
            eps: 1e-8,
            weight_decay: 0.0,
            step_t: 0,
            m: vec![None; n],
            v: vec![None; n],
        }
    }

    /// Override (β1, β2). Defaults are (0.9, 0.999).
    #[must_use]
    pub fn betas(mut self, b1: f32, b2: f32) -> Self {
        self.betas = (b1, b2);
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

impl Optimizer for NAdam {
    fn step(&mut self) {
        self.step_t += 1;
        let t = self.step_t as f32;
        let (b1, b2) = self.betas;
        let bc1 = 1.0 - b1.powf(t);
        let bc2 = 1.0 - b2.powf(t);
        for (i, param) in self.params.iter().enumerate() {
            let grad = match param.grad() {
                Some(g) => g,
                None => continue,
            };
            let snapshot = param.data_snapshot();
            let p_data: &[f32] = snapshot.as_slice::<f32>().expect("NAdam: F32 only");
            let g_data: &[f32] = grad.as_slice::<f32>().expect("NAdam: F32 only");

            let mut m_buf = self.m[i]
                .take()
                .unwrap_or_else(|| vec![0.0_f32; p_data.len()]);
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
                m_buf[k] = b1 * m_buf[k] + (1.0 - b1) * g;
                v_buf[k] = b2 * v_buf[k] + (1.0 - b2) * g * g;
                let m_hat = m_buf[k] / bc1;
                let v_hat = v_buf[k] / bc2;
                // Nesterov-corrected momentum
                let m_hat_prime = b1 * m_hat + (1.0 - b1) * g / bc1;
                let p_new = p_data[k] - self.lr * m_hat_prime / (v_hat.sqrt() + self.eps);
                new.push(p_new);
            }
            write_param_data(param, new);
            self.m[i] = Some(m_buf);
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
    fn nadam_descends_quadratic_loss() {
        let p = Variable::leaf(Tensor::from_vec([3], vec![5.0_f32, -3.0, 2.0]).unwrap());
        let mut opt = NAdam::new(vec![p.clone()], 0.05);
        for _ in 0..200 {
            opt.zero_grad();
            let s = ops::sum(&ops::mul(&p, &p).unwrap()).unwrap();
            backward(&s, None).unwrap();
            opt.step();
        }
        let v = p.tensor().as_slice::<f32>().unwrap().to_vec();
        for &x in &v {
            assert!(x.abs() < 0.1, "got {x}");
        }
    }
}
