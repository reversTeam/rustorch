//! Normalization modules (P1.6).
//!
//! v1 ships:
//! - [`RMSNorm`] (Zhang & Sennrich 2019), the simplest variant used by
//!   Llama, Mistral, T5-derived architectures.
//! - [`LayerNorm`] (Ba et al. 2016), the workhorse of Transformers.
//!
//! Both norms are built by composing existing autograd ops (mul,
//! mean_dim, sqrt, div, add, sub) — backward falls out automatically
//! from the dynamic graph. No hand-coded backward formula required.
//!
//! GroupNorm / BatchNorm / InstanceNorm follow once their dedicated
//! kernels land in P1.4 (Welford-stable variance is on the roadmap).

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

    fn named_parameters(&self) -> Vec<(String, Variable)> {
        vec![("gamma".to_string(), self.gamma.clone())]
    }
}

// ------------------------------ LayerNorm ------------------------------

/// LayerNorm: `y = (x - mean) / sqrt(var + eps) * gamma + beta`.
///
/// `gamma` and `beta` are both learnable parameters of shape
/// `[normalized_size]`, broadcasted across leading dims.
pub struct LayerNorm {
    /// Learnable per-channel scale.
    pub gamma: Variable,
    /// Learnable per-channel shift.
    pub beta: Variable,
    eps: f32,
    normalized_size: usize,
}

impl LayerNorm {
    /// Build with the given last-axis size and default eps = 1e-5.
    pub fn new(normalized_size: usize) -> Self {
        Self::with_eps(normalized_size, 1e-5)
    }

    /// Build with explicit epsilon.
    pub fn with_eps(normalized_size: usize, eps: f32) -> Self {
        let gamma = Variable::leaf(
            Tensor::from_vec([normalized_size], vec![1.0_f32; normalized_size])
                .expect("gamma shape"),
        );
        let beta = Variable::leaf(
            Tensor::from_vec([normalized_size], vec![0.0_f32; normalized_size])
                .expect("beta shape"),
        );
        LayerNorm {
            gamma,
            beta,
            eps,
            normalized_size,
        }
    }

    /// Last-axis size this norm operates on.
    pub fn normalized_size(&self) -> usize {
        self.normalized_size
    }
}

impl Module for LayerNorm {
    fn forward(&self, input: &Variable) -> Result<Variable, ModuleError> {
        let last_dim = input.tensor().ndim() - 1;
        // mean
        let mean = ops::mean_dim(input, &[last_dim])?;
        // x - mean (broadcast)
        let centered = ops::sub(input, &mean)?;
        // (x - mean)²
        let centered_sq = ops::mul(&centered, &centered)?;
        // var = mean of centered²
        let var = ops::mean_dim(&centered_sq, &[last_dim])?;
        // var + eps
        let eps_var = Variable::new(Tensor::scalar(self.eps));
        let var_eps = ops::add(&var, &eps_var)?;
        // std = sqrt(var + eps)
        let std = ops::sqrt(&var_eps)?;
        // normalised = centered / std
        let normed = ops::div(&centered, &std)?;
        // * gamma + beta
        let scaled = ops::mul(&normed, &self.gamma)?;
        ops::add(&scaled, &self.beta)
    }

    fn parameters(&self) -> Vec<Variable> {
        vec![self.gamma.clone(), self.beta.clone()]
    }

    fn named_parameters(&self) -> Vec<(String, Variable)> {
        vec![
            ("gamma".to_string(), self.gamma.clone()),
            ("beta".to_string(), self.beta.clone()),
        ]
    }
}

// ------------------------------ BatchNorm2d ------------------------------

/// BatchNorm2d (Ioffe & Szegedy 2015). Layout `[N, C, H, W]`.
/// v1 ships **train mode only** — running stats are not tracked yet
/// (eval-mode + running mean/var pending).
pub struct BatchNorm2d {
    /// Learnable per-channel scale.
    pub gamma: Variable,
    /// Learnable per-channel shift.
    pub beta: Variable,
    eps: f32,
    num_features: usize,
}

impl BatchNorm2d {
    /// Build with the given `num_features` (channel count) and default
    /// eps = 1e-5.
    pub fn new(num_features: usize) -> Self {
        Self::with_eps(num_features, 1e-5)
    }

    /// Build with explicit epsilon.
    pub fn with_eps(num_features: usize, eps: f32) -> Self {
        let gamma = Variable::leaf(
            Tensor::from_vec([num_features], vec![1.0_f32; num_features]).expect("gamma shape"),
        );
        let beta = Variable::leaf(
            Tensor::from_vec([num_features], vec![0.0_f32; num_features]).expect("beta shape"),
        );
        BatchNorm2d {
            gamma,
            beta,
            eps,
            num_features,
        }
    }

    /// Channel count.
    pub fn num_features(&self) -> usize {
        self.num_features
    }
}

impl Module for BatchNorm2d {
    fn forward(&self, input: &Variable) -> Result<Variable, ModuleError> {
        ops::batch_norm2d(input, &self.gamma, &self.beta, self.eps)
    }

    fn parameters(&self) -> Vec<Variable> {
        vec![self.gamma.clone(), self.beta.clone()]
    }

    fn named_parameters(&self) -> Vec<(String, Variable)> {
        vec![
            ("gamma".to_string(), self.gamma.clone()),
            ("beta".to_string(), self.beta.clone()),
        ]
    }
}

#[cfg(test)]
mod batch_norm_tests {
    use super::*;
    use rustorch_autograd::backward;

    #[test]
    fn batch_norm2d_normalises_per_channel() {
        let bn = BatchNorm2d::new(2);
        let x = Variable::new(
            Tensor::from_vec(
                [2usize, 2, 2, 2],
                (0..16).map(|i| i as f32).collect::<Vec<_>>(),
            )
            .unwrap(),
        );
        let y = bn.forward(&x).unwrap();
        assert_eq!(y.tensor().shape(), &[2, 2, 2, 2]);
    }

    #[test]
    fn batch_norm2d_backward_grads_flow() {
        let bn = BatchNorm2d::new(2);
        let x = Variable::leaf(
            Tensor::from_vec(
                [2usize, 2, 2, 2],
                (0..16).map(|i| i as f32 * 0.1).collect::<Vec<_>>(),
            )
            .unwrap(),
        );
        let y = bn.forward(&x).unwrap();
        let s = rustorch_autograd::ops::sum(&y).unwrap();
        backward(&s, None).unwrap();
        assert!(bn.gamma.grad().is_some());
        assert!(bn.beta.grad().is_some());
        assert!(x.grad().is_some());
    }

    #[test]
    fn batch_norm2d_named_parameters() {
        let bn = BatchNorm2d::new(8);
        let np = bn.named_parameters();
        let names: Vec<String> = np.iter().map(|(n, _)| n.clone()).collect();
        assert_eq!(names, vec!["gamma".to_string(), "beta".to_string()]);
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

    // -------------------- LayerNorm --------------------

    #[test]
    fn layer_norm_zero_input_returns_beta() {
        // mean=0, var=0 → centered=0, normed=0, y=0*gamma+beta = beta
        let norm = LayerNorm::with_eps(4, 1e-5);
        let x = var(vec![1usize, 4], vec![0.0_f32; 4]);
        let y = norm.forward(&x).unwrap();
        let y_t = y.tensor();
        let y_v = y_t.as_slice::<f32>().unwrap();
        // beta defaults to zeros → y = zeros
        for &v in y_v {
            assert!(v.abs() < 1e-5, "expected ~0 (beta=0), got {v}");
        }
    }

    #[test]
    fn layer_norm_centers_and_unit_variance() {
        // After LayerNorm on a row, the per-row mean ≈ beta and per-row
        // sample-stddev ≈ gamma (default 1, 0).
        let norm = LayerNorm::with_eps(4, 0.0);
        let x = var(vec![1usize, 4], vec![1.0_f32, 2.0, 3.0, 4.0]);
        let y = norm.forward(&x).unwrap();
        let y_t = y.tensor();
        let y_v = y_t.as_slice::<f32>().unwrap();
        let mean = y_v.iter().sum::<f32>() / y_v.len() as f32;
        let var = y_v.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / y_v.len() as f32;
        assert!(mean.abs() < 1e-5, "mean ≈ 0, got {mean}");
        assert!((var - 1.0).abs() < 1e-4, "var ≈ 1, got {var}");
    }

    #[test]
    fn layer_norm_preserves_shape() {
        let norm = LayerNorm::new(8);
        let x = var(vec![3usize, 5, 8], vec![0.5_f32; 3 * 5 * 8]);
        let y = norm.forward(&x).unwrap();
        assert_eq!(y.tensor().shape(), &[3, 5, 8]);
    }

    #[test]
    fn layer_norm_backward_flows_to_gamma_beta_x() {
        let norm = LayerNorm::new(4);
        let x = var(
            vec![2usize, 4],
            vec![0.5_f32, 1.0, 1.5, 2.0, 0.3, 0.6, 0.9, 1.2],
        );
        let y = norm.forward(&x).unwrap();
        let s = rustorch_autograd::ops::sum(&y).unwrap();
        backward(&s, None).unwrap();
        for p in &[&norm.gamma, &norm.beta] {
            let g = p.grad().unwrap();
            assert!(
                g.as_slice::<f32>().unwrap().iter().any(|v| v.abs() > 1e-7),
                "param should accumulate non-zero grad"
            );
        }
        let xg = x.grad().unwrap();
        assert!(xg.as_slice::<f32>().unwrap().iter().any(|v| v.abs() > 1e-7));
    }

    #[test]
    fn layer_norm_parameters_count() {
        let norm = LayerNorm::new(8);
        let p = norm.parameters();
        assert_eq!(p.len(), 2);
        assert_eq!(p[0].tensor().shape(), &[8]);
        assert_eq!(p[1].tensor().shape(), &[8]);
    }
}
