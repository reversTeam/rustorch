//! Adafactor (Shazeer & Stern 2018) — memory-efficient Adam.
//!
//! For 2-D weights, factors the second moment as outer product of two
//! 1-D vectors (R, C) — `O(M+N)` state instead of `O(M*N)`. For
//! 1-D / scalar params, falls back to standard Adam-style state.
//!
//! v1 simplification: vanilla beta_2 schedule (no relative-step or
//! warmup-init); decoupled weight decay; no learning-rate scaling
//! by RMS of params (paper's "scale_parameter" option).

use crate::{write_param_data, Optimizer};
use rustorch_autograd::Variable;

/// Adafactor (Shazeer & Stern 2018).
pub struct Adafactor {
    params: Vec<Variable>,
    lr: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    /// For 2-D params: row factor R (length = rows). Otherwise None.
    r: Vec<Option<Vec<f32>>>,
    /// For 2-D params: col factor C (length = cols). Otherwise None.
    c: Vec<Option<Vec<f32>>>,
    /// For non-factored params: full v buffer.
    v: Vec<Option<Vec<f32>>>,
}

impl Adafactor {
    /// Build with the given learning rate.
    pub fn new(params: Vec<Variable>, lr: f32) -> Self {
        let n = params.len();
        Adafactor {
            params,
            lr,
            beta2: 0.999,
            eps: 1e-30,
            weight_decay: 0.0,
            r: vec![None; n],
            c: vec![None; n],
            v: vec![None; n],
        }
    }

    /// Override β₂.
    #[must_use]
    pub fn beta2(mut self, b2: f32) -> Self {
        self.beta2 = b2;
        self
    }
    /// Decoupled weight decay.
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

impl Optimizer for Adafactor {
    fn step(&mut self) {
        for (i, param) in self.params.iter().enumerate() {
            let grad = match param.grad() {
                Some(g) => g,
                None => continue,
            };
            let snapshot = param.data_snapshot();
            let p_data: &[f32] = snapshot.as_slice::<f32>().expect("Adafactor F32");
            let g_data: &[f32] = grad.as_slice::<f32>().expect("Adafactor F32");
            let shape = param.tensor().shape().to_vec();
            let factored = shape.len() == 2;

            let mut new = vec![0.0_f32; p_data.len()];
            if factored {
                let rows = shape[0];
                let cols = shape[1];
                let mut r = self.r[i].take().unwrap_or_else(|| vec![0.0_f32; rows]);
                let mut c = self.c[i].take().unwrap_or_else(|| vec![0.0_f32; cols]);
                // Compute row/column means of g²
                let mut r_new = vec![0.0_f32; rows];
                let mut c_new = vec![0.0_f32; cols];
                for ri in 0..rows {
                    let mut s = 0.0_f32;
                    for cj in 0..cols {
                        let g = g_data[ri * cols + cj];
                        s += g * g;
                    }
                    r_new[ri] = s / cols as f32;
                }
                for cj in 0..cols {
                    let mut s = 0.0_f32;
                    for ri in 0..rows {
                        let g = g_data[ri * cols + cj];
                        s += g * g;
                    }
                    c_new[cj] = s / rows as f32;
                }
                // Update factored second moment.
                for ri in 0..rows {
                    r[ri] = self.beta2 * r[ri] + (1.0 - self.beta2) * r_new[ri];
                }
                for cj in 0..cols {
                    c[cj] = self.beta2 * c[cj] + (1.0 - self.beta2) * c_new[cj];
                }
                // Reconstruct V approximately as outer(R, C / mean(R)).
                let r_mean = r.iter().sum::<f32>() / rows as f32;
                let r_mean = r_mean.max(self.eps);
                for ri in 0..rows {
                    for cj in 0..cols {
                        let v_approx = r[ri] * c[cj] / r_mean;
                        let g = g_data[ri * cols + cj];
                        let g_eff = if self.weight_decay > 0.0 {
                            g + self.weight_decay * p_data[ri * cols + cj]
                        } else {
                            g
                        };
                        new[ri * cols + cj] =
                            p_data[ri * cols + cj] - self.lr * g_eff / (v_approx.sqrt() + self.eps);
                    }
                }
                self.r[i] = Some(r);
                self.c[i] = Some(c);
            } else {
                // Non-factored fall-back: standard Adam-style v buffer.
                let mut v_buf = self.v[i]
                    .take()
                    .unwrap_or_else(|| vec![0.0_f32; p_data.len()]);
                for k in 0..p_data.len() {
                    let g = g_data[k];
                    let g_eff = if self.weight_decay > 0.0 {
                        g + self.weight_decay * p_data[k]
                    } else {
                        g
                    };
                    v_buf[k] = self.beta2 * v_buf[k] + (1.0 - self.beta2) * g * g;
                    new[k] = p_data[k] - self.lr * g_eff / (v_buf[k].sqrt() + self.eps);
                }
                self.v[i] = Some(v_buf);
            }
            write_param_data(param, new);
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
    fn adafactor_factored_state_for_2d_param() {
        let p = Variable::leaf(Tensor::from_vec([3usize, 4], vec![1.0_f32; 12]).unwrap());
        let mut opt = Adafactor::new(vec![p.clone()], 0.01);
        let s = ops::sum(&ops::mul(&p, &p).unwrap()).unwrap();
        backward(&s, None).unwrap();
        opt.step();
        // Factored state: R length 3, C length 4 (vs. full 12-element v).
        assert_eq!(opt.r[0].as_ref().unwrap().len(), 3);
        assert_eq!(opt.c[0].as_ref().unwrap().len(), 4);
        assert!(opt.v[0].is_none());
    }

    #[test]
    fn adafactor_descends_2d_quadratic_loss() {
        let p = Variable::leaf(Tensor::from_vec([2usize, 3], vec![5.0_f32; 6]).unwrap());
        let mut opt = Adafactor::new(vec![p.clone()], 1.0);
        for _ in 0..600 {
            opt.zero_grad();
            let s = ops::sum(&ops::mul(&p, &p).unwrap()).unwrap();
            backward(&s, None).unwrap();
            opt.step();
        }
        for &x in p.tensor().as_slice::<f32>().unwrap() {
            assert!(x.abs() < 1.0, "got {x}");
        }
    }
}
