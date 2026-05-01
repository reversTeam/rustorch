//! Adamax optimizer (P1.7) — Adam with infinity norm.
//!
//! ```text
//!   m_t = β₁*m_{t-1} + (1-β₁)*g
//!   u_t = max(β₂*u_{t-1}, |g|)
//!   m̂_t = m_t / (1 - β₁^t)
//!   θ_t = θ_{t-1} - lr * m̂_t / (u_t + ε)
//! ```

use crate::{write_param_data, Optimizer};
use rustorch_autograd::Variable;

/// Adamax (Kingma & Ba 2014).
pub struct Adamax {
    params: Vec<Variable>,
    lr: f32,
    betas: (f32, f32),
    eps: f32,
    weight_decay: f32,
    step_t: usize,
    m: Vec<Option<Vec<f32>>>,
    u: Vec<Option<Vec<f32>>>,
}

impl Adamax {
    /// Build with the given learning rate.
    pub fn new(params: Vec<Variable>, lr: f32) -> Self {
        let n = params.len();
        Adamax {
            params,
            lr,
            betas: (0.9, 0.999),
            eps: 1e-8,
            weight_decay: 0.0,
            step_t: 0,
            m: vec![None; n],
            u: vec![None; n],
        }
    }

    /// Override (β₁, β₂).
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
    /// Set L2 weight decay.
    #[must_use]
    pub fn weight_decay(mut self, wd: f32) -> Self {
        self.weight_decay = wd;
        self
    }
    /// Borrow parameter list.
    pub fn parameters(&self) -> &[Variable] {
        &self.params
    }
}

impl Optimizer for Adamax {
    fn step(&mut self) {
        self.step_t += 1;
        let t = self.step_t as f32;
        let (b1, b2) = self.betas;
        let bc1 = 1.0 - b1.powf(t);
        for (i, param) in self.params.iter().enumerate() {
            let grad = match param.grad() {
                Some(g) => g,
                None => continue,
            };
            let snapshot = param.data_snapshot();
            let p_data: &[f32] = snapshot.as_slice::<f32>().expect("Adamax: F32 only");
            let g_data: &[f32] = grad.as_slice::<f32>().expect("Adamax: F32 only");
            let mut m_buf = self.m[i]
                .take()
                .unwrap_or_else(|| vec![0.0_f32; p_data.len()]);
            let mut u_buf = self.u[i]
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
                u_buf[k] = (b2 * u_buf[k]).max(g.abs());
                let m_hat = m_buf[k] / bc1;
                new.push(p_data[k] - self.lr * m_hat / (u_buf[k] + self.eps));
            }
            write_param_data(param, new);
            self.m[i] = Some(m_buf);
            self.u[i] = Some(u_buf);
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
    fn adamax_descends_quadratic_loss() {
        let p = Variable::leaf(Tensor::from_vec([3], vec![5.0_f32, -3.0, 2.0]).unwrap());
        let mut opt = Adamax::new(vec![p.clone()], 0.05);
        for _ in 0..200 {
            opt.zero_grad();
            let s = ops::sum(&ops::mul(&p, &p).unwrap()).unwrap();
            backward(&s, None).unwrap();
            opt.step();
        }
        for &x in p.tensor().as_slice::<f32>().unwrap() {
            assert!(x.abs() < 0.5, "got {x}");
        }
    }
}
