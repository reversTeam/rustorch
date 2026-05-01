//! Normalization modules (P1.6).
//!
//! v1 ships [`RMSNorm`] (Zhang & Sennrich 2019), the simplest variant
//! used by Llama, Mistral, T5-derived architectures. LayerNorm /
//! GroupNorm / BatchNorm follow once their dedicated kernels land
//! in P1.4.
//!
//! `RMSNorm` is built by composing existing autograd ops (mul,
//! mean_dim, sqrt, div, add) — backward falls out automatically from
//! the dynamic graph. No hand-coded backward formula required.

use crate::module::{Module, ModuleError};
use rustorch_autograd::{ops, Variable};
use rustorch_core::tensor::tensor_impl::Tensor;

/// Root-Mean-Square LayerNorm: `y = x / sqrt(mean(x², last) + eps) * gamma`.
///
/// `gamma` is a learnable scale of shape `[normalized_size]`. There's
/// no shift parameter (this is the key difference vs LayerNorm).
pub struct RMSNorm {
    /// Learnable per-channel scale of shape `[normalized_size]`.
    pub gamma: Variable,
    eps: f32,
    normalized_size: usize,
}

impl RMSNorm {
    /// Build with the given last-axis size and default eps = 1e-6.
    pub fn new(normalized_size: usize) -> Self {
        Self::with_eps(normalized_size, 1e-6)
    }

    /// Build with explicit epsilon.
    pub fn with_eps(normalized_size: usize, eps: f32) -> Self {
        // Initialise gamma to ones (no-op at init).
        let gamma_data = vec![1.0_f32; normalized_size];
        let gamma =
            Variable::leaf(Tensor::from_vec([normalized_size], gamma_data).expect("gamma shape"));
        RMSNorm {
            gamma,
            eps,
            normalized_size,
        }
    }

    /// Last-axis size this norm operates on.
    pub fn normalized_size(&self) -> usize {
        self.normalized_size
    }
}

impl Module for RMSNorm {
    fn forward(&self, input: &Variable) -> Result<Variable, ModuleError> {
        // Reduce dim = last axis of input.
        let last_dim = input.tensor().ndim() - 1;
        // x²
        let x_sq = ops::mul(input, input)?;
        // mean over last axis (keepdim=true)
        let mean_x_sq = ops::mean_dim(&x_sq, &[last_dim])?;
        // mean + eps  (eps as broadcastable scalar Variable)
        let eps_var = Variable::new(Tensor::scalar(self.eps));
        let rms_sq = ops::add(&mean_x_sq, &eps_var)?;
        // sqrt
        let rms = ops::sqrt(&rms_sq)?;
        // x / rms (broadcast last axis)
        let normed = ops::div(input, &rms)?;
        // * gamma (gamma [D] broadcasts across leading dims of normed)
        ops::mul(&normed, &self.gamma)
    }

    fn parameters(&self) -> Vec<Variable> {
        vec![self.gamma.clone()]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustorch_autograd::{backward, Variable};
    use rustorch_core::tensor::tensor_impl::Tensor;

    fn var(shape: impl Into<Vec<usize>>, data: Vec<f32>) -> Variable {
        Variable::leaf(Tensor::from_vec(shape.into(), data).unwrap())
    }

    #[test]
    fn rms_norm_unit_input_returns_unit_output() {
        // For x = ones and gamma = ones, RMS = sqrt(1 + eps) ≈ 1, so y ≈ 1.
        let norm = RMSNorm::with_eps(4, 0.0);
        let x = var(vec![1usize, 4], vec![1.0_f32; 4]);
        let y = norm.forward(&x).unwrap();
        let y_t = y.tensor();
        let y_v = y_t.as_slice::<f32>().unwrap();
        for &v in y_v {
            assert!((v - 1.0).abs() < 1e-5, "got {v}");
        }
    }

    #[test]
    fn rms_norm_preserves_shape() {
        let norm = RMSNorm::new(8);
        let x = var(vec![3usize, 5, 8], vec![0.5_f32; 3 * 5 * 8]);
        let y = norm.forward(&x).unwrap();
        assert_eq!(y.tensor().shape(), &[3, 5, 8]);
    }

    #[test]
    fn rms_norm_gamma_scales_output_uniformly() {
        // If gamma = c * ones, output should be c * normed.
        let mut norm = RMSNorm::with_eps(4, 0.0);
        // Override gamma.
        let g = Variable::leaf(Tensor::from_vec([4usize], vec![3.0_f32; 4]).unwrap());
        norm.gamma = g;
        let x = var(vec![1usize, 4], vec![1.0_f32, 1.0, 1.0, 1.0]);
        let y = norm.forward(&x).unwrap();
        let y_t = y.tensor();
        let y_v = y_t.as_slice::<f32>().unwrap();
        // RMS=1, gamma=3 → y = 3
        for &v in y_v {
            assert!((v - 3.0).abs() < 1e-5, "got {v}");
        }
    }

    #[test]
    fn rms_norm_backward_runs_end_to_end() {
        // Basic gradient flow: train objective y.sum() against random input.
        let norm = RMSNorm::new(4);
        let x = var(
            vec![2usize, 4],
            vec![0.5_f32, 1.0, 1.5, 2.0, 0.3, 0.6, 0.9, 1.2],
        );
        let y = norm.forward(&x).unwrap();
        let s = rustorch_autograd::ops::sum(&y).unwrap();
        backward(&s, None).unwrap();
        // Gamma should have a non-zero grad after backward.
        let g_grad = norm.gamma.grad().unwrap();
        let g_grad_v = g_grad.as_slice::<f32>().unwrap();
        assert!(
            g_grad_v.iter().any(|&v| v.abs() > 1e-6),
            "gamma should accumulate non-zero grad, got {:?}",
            g_grad_v
        );
        // x should also have a grad.
        let x_grad = x.grad().unwrap();
        let x_grad_v = x_grad.as_slice::<f32>().unwrap();
        assert!(
            x_grad_v.iter().any(|&v| v.abs() > 1e-6),
            "x should accumulate non-zero grad, got {:?}",
            x_grad_v
        );
    }

    #[test]
    fn rms_norm_parameters_returns_gamma_only() {
        let norm = RMSNorm::new(8);
        let p = norm.parameters();
        assert_eq!(p.len(), 1);
        assert_eq!(p[0].tensor().shape(), &[8]);
    }
}
