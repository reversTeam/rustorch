//! RAdam optimizer (Liu et al. 2020) — Rectified Adam.
//!
//! Rectifies the variance term during early training to avoid the
//! "warmup" needed by vanilla Adam.

use crate::{write_param_data, Optimizer};
use rustorch_autograd::Variable;

/// RAdam (Rectified Adam, Liu et al. 2020).
pub struct RAdam {
    params: Vec<Variable>,
    lr: f32,
    betas: (f32, f32),
    eps: f32,
    weight_decay: f32,
    step_t: usize,
    m: Vec<Option<Vec<f32>>>,
    v: Vec<Option<Vec<f32>>>,
}

impl RAdam {
    /// Build with the given learning rate.
    pub fn new(params: Vec<Variable>, lr: f32) -> Self {
        let n = params.len();
        RAdam {
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
    /// Override (β1, β2).
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
    /// Borrow params.
    pub fn parameters(&self) -> &[Variable] {
        &self.params
    }
}

impl Optimizer for RAdam {
    fn step(&mut self) {
        self.step_t += 1;
        let t = self.step_t as f32;
        let (b1, b2) = self.betas;
        let bc1 = 1.0 - b1.powf(t);
        let bc2 = 1.0 - b2.powf(t);
        // Length of the approximated SMA at infinity.
        let rho_inf = 2.0 / (1.0 - b2) - 1.0;
        // SMA at this step.
        let rho_t = rho_inf - 2.0 * t * b2.powf(t) / bc2;

        for (i, param) in self.params.iter().enumerate() {
            let grad = match param.grad() {
                Some(g) => g,
                None => continue,
            };
            let snapshot = param.data_snapshot();
            let p_data: &[f32] = snapshot.as_slice::<f32>().expect("RAdam F32");
            let g_data: &[f32] = grad.as_slice::<f32>().expect("RAdam F32");
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
                if rho_t > 4.0 {
                    let v_hat = (v_buf[k] / bc2).sqrt();
                    let r = ((rho_t - 4.0) * (rho_t - 2.0) * rho_inf
                        / ((rho_inf - 4.0) * (rho_inf - 2.0) * rho_t))
                        .sqrt();
                    new.push(p_data[k] - self.lr * r * m_hat / (v_hat + self.eps));
                } else {
                    // Variance not yet reliable — fall back to plain momentum step.
                    new.push(p_data[k] - self.lr * m_hat);
                }
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
    fn radam_descends_quadratic_loss() {
        let p = Variable::leaf(Tensor::from_vec([3], vec![5.0_f32, -3.0, 2.0]).unwrap());
        let mut opt = RAdam::new(vec![p.clone()], 0.1);
        for _ in 0..400 {
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
