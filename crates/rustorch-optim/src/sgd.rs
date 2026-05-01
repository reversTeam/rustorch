//! Stochastic Gradient Descent (SGD) optimiser (P1.7).
//!
//! Supports vanilla SGD, momentum (Polyak), Nesterov accelerated
//! gradient, and L2 weight decay (decoupled from the gradient).

use crate::{write_param_data, Optimizer};
use rustorch_autograd::Variable;

/// SGD optimiser with optional momentum + Nesterov + weight decay.
pub struct Sgd {
    params: Vec<Variable>,
    lr: f32,
    momentum: f32,
    weight_decay: f32,
    nesterov: bool,
    /// Per-parameter momentum buffer (None until the first step that
    /// uses momentum > 0).
    velocity: Vec<Option<Vec<f32>>>,
}

impl Sgd {
    /// Build a vanilla SGD optimiser with the given learning rate.
    pub fn new(params: Vec<Variable>, lr: f32) -> Self {
        let n = params.len();
        Sgd {
            params,
            lr,
            momentum: 0.0,
            weight_decay: 0.0,
            nesterov: false,
            velocity: vec![None; n],
        }
    }

    /// Set momentum coefficient (Polyak). Builder-style.
    #[must_use]
    pub fn momentum(mut self, m: f32) -> Self {
        self.momentum = m;
        self
    }

    /// Enable Nesterov accelerated gradient. Builder-style.
    #[must_use]
    pub fn nesterov(mut self, n: bool) -> Self {
        self.nesterov = n;
        self
    }

    /// Set L2 weight decay (decoupled from grad — added per param
    /// before the velocity update).
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

impl Optimizer for Sgd {
    fn step(&mut self) {
        for (i, param) in self.params.iter().enumerate() {
            let grad = match param.grad() {
                Some(g) => g,
                None => continue,
            };
            let snapshot = param.data_snapshot();
            let p_data: &[f32] = snapshot.as_slice::<f32>().expect("SGD: F32 params only");
            let g_data: &[f32] = grad.as_slice::<f32>().expect("SGD: F32 grads only");
            let mut new = Vec::with_capacity(p_data.len());

            // Compute effective gradient (with weight decay).
            let effective_grad: Vec<f32> = if self.weight_decay > 0.0 {
                p_data
                    .iter()
                    .zip(g_data.iter())
                    .map(|(&p, &g)| g + self.weight_decay * p)
                    .collect()
            } else {
                g_data.to_vec()
            };

            if self.momentum > 0.0 {
                // velocity_t = momentum * velocity_{t-1} + effective_grad
                let v_prev = self.velocity[i]
                    .take()
                    .unwrap_or_else(|| vec![0.0_f32; p_data.len()]);
                let v_new: Vec<f32> = v_prev
                    .iter()
                    .zip(effective_grad.iter())
                    .map(|(&v, &g)| self.momentum * v + g)
                    .collect();
                let update: Vec<f32> = if self.nesterov {
                    // Nesterov: x_{t+1} = x_t - lr * (g + momentum * v_new)
                    effective_grad
                        .iter()
                        .zip(v_new.iter())
                        .map(|(&g, &v)| g + self.momentum * v)
                        .collect()
                } else {
                    v_new.clone()
                };
                for (p, u) in p_data.iter().zip(update.iter()) {
                    new.push(p - self.lr * u);
                }
                self.velocity[i] = Some(v_new);
            } else {
                // Vanilla SGD
                for (p, g) in p_data.iter().zip(effective_grad.iter()) {
                    new.push(p - self.lr * g);
                }
            }
            write_param_data(param, new);
        }
        // No reload needed — Variable::tensor() now reads fresh from the
        // shared `data` mutex on every call.
    }

    fn zero_grad(&mut self) {
        for param in &self.params {
            param.zero_grad();
        }
    }
}
