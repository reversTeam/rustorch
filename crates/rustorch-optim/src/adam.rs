//! Adam / AdamW optimisers (P1.7).
//!
//! Adam (Kingma & Ba 2015):
//! ```text
//!   m_t = β₁ * m_{t-1} + (1 - β₁) * g
//!   v_t = β₂ * v_{t-1} + (1 - β₂) * g²
//!   m̂_t = m_t / (1 - β₁^t)
//!   v̂_t = v_t / (1 - β₂^t)
//!   θ_t = θ_{t-1} - lr * m̂_t / (sqrt(v̂_t) + ε)
//! ```
//!
//! AdamW: same update with **decoupled** weight decay applied directly
//! on the parameter (not via the gradient).

use crate::{write_param_data, Optimizer};
use rustorch_autograd::Variable;

/// Adam optimiser with bias correction.
pub struct Adam {
    params: Vec<Variable>,
    lr: f32,
    betas: (f32, f32),
    eps: f32,
    weight_decay: f32,
    decoupled_wd: bool,
    step_t: usize,
    m: Vec<Option<Vec<f32>>>,
    v: Vec<Option<Vec<f32>>>,
}

impl Adam {
    /// Build an Adam optimiser with the given learning rate.
    pub fn new(params: Vec<Variable>, lr: f32) -> Self {
        let n = params.len();
        Adam {
            params,
            lr,
            betas: (0.9, 0.999),
            eps: 1e-8,
            weight_decay: 0.0,
            decoupled_wd: false,
            step_t: 0,
            m: vec![None; n],
            v: vec![None; n],
        }
    }

    /// Override `(β₁, β₂)`. Defaults are (0.9, 0.999).
    #[must_use]
    pub fn betas(mut self, b1: f32, b2: f32) -> Self {
        self.betas = (b1, b2);
        self
    }

    /// Override `ε`. Default is 1e-8.
    #[must_use]
    pub fn eps(mut self, e: f32) -> Self {
        self.eps = e;
        self
    }

    /// Set L2 weight decay coefficient (added to the gradient).
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

/// AdamW = Adam with decoupled weight decay (Loshchilov & Hutter 2019).
pub struct AdamW(Adam);

impl AdamW {
    /// Build an AdamW optimiser. Default weight_decay = 0.01.
    pub fn new(params: Vec<Variable>, lr: f32) -> Self {
        let mut inner = Adam::new(params, lr);
        inner.weight_decay = 0.01;
        inner.decoupled_wd = true;
        AdamW(inner)
    }

    /// Override betas (passes through to inner Adam).
    #[must_use]
    pub fn betas(mut self, b1: f32, b2: f32) -> Self {
        self.0.betas = (b1, b2);
        self
    }

    /// Override weight decay (decoupled).
    #[must_use]
    pub fn weight_decay(mut self, wd: f32) -> Self {
        self.0.weight_decay = wd;
        self
    }

    /// Override eps.
    #[must_use]
    pub fn eps(mut self, e: f32) -> Self {
        self.0.eps = e;
        self
    }

    /// Borrow the parameter list.
    pub fn parameters(&self) -> &[Variable] {
        self.0.parameters()
    }
}

impl Optimizer for Adam {
    fn step(&mut self) {
        self.step_t += 1;
        let t = self.step_t as f32;
        let bc1 = 1.0 - self.betas.0.powf(t);
        let bc2 = 1.0 - self.betas.1.powf(t);
        for (i, param) in self.params.iter().enumerate() {
            let grad = match param.grad() {
                Some(g) => g,
                None => continue,
            };
            let snapshot = param.data_snapshot();
            let p_data: &[f32] = snapshot.as_slice::<f32>().expect("Adam: F32 only");
            let g_data: &[f32] = grad.as_slice::<f32>().expect("Adam: F32 only");

            // Effective gradient: classical Adam couples weight decay
            // into the gradient; AdamW applies it directly to the param.
            let g_eff: Vec<f32> = if !self.decoupled_wd && self.weight_decay > 0.0 {
                p_data
                    .iter()
                    .zip(g_data.iter())
                    .map(|(&p, &g)| g + self.weight_decay * p)
                    .collect()
            } else {
                g_data.to_vec()
            };

            // m and v buffers
            let mut m_buf = self.m[i]
                .take()
                .unwrap_or_else(|| vec![0.0_f32; p_data.len()]);
            let mut v_buf = self.v[i]
                .take()
                .unwrap_or_else(|| vec![0.0_f32; p_data.len()]);
            for k in 0..p_data.len() {
                m_buf[k] = self.betas.0 * m_buf[k] + (1.0 - self.betas.0) * g_eff[k];
                v_buf[k] = self.betas.1 * v_buf[k] + (1.0 - self.betas.1) * g_eff[k] * g_eff[k];
            }

            let mut new = Vec::with_capacity(p_data.len());
            for k in 0..p_data.len() {
                let m_hat = m_buf[k] / bc1;
                let v_hat = v_buf[k] / bc2;
                let mut p_new = p_data[k] - self.lr * m_hat / (v_hat.sqrt() + self.eps);
                if self.decoupled_wd && self.weight_decay > 0.0 {
                    p_new -= self.lr * self.weight_decay * p_data[k];
                }
                new.push(p_new);
            }
            write_param_data(param, new);
            self.m[i] = Some(m_buf);
            self.v[i] = Some(v_buf);
        }
        // No reload needed — Variable::tensor() reads fresh from `data`.
    }

    fn zero_grad(&mut self) {
        for param in &self.params {
            param.zero_grad();
        }
    }
}

impl Optimizer for AdamW {
    fn step(&mut self) {
        self.0.step();
    }
    fn zero_grad(&mut self) {
        self.0.zero_grad();
    }
}
