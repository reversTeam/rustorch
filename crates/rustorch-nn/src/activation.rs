//! Module-able activations. Each unit struct implements [`Module`]
//! and forwards to the corresponding autograd op.
//!
//! Functional variants are also re-exported for ergonomic standalone
//! use without going through a Module.

use crate::module::{Module, ModuleError};
use rustorch_autograd::ops;
use rustorch_autograd::Variable;

// ------------------------------ ReLU ------------------------------

/// ReLU activation (`max(0, x)`).
#[derive(Default)]
pub struct Relu;

impl Module for Relu {
    fn forward(&self, input: &Variable) -> Result<Variable, ModuleError> {
        ops::relu(input)
    }
}

/// Functional ReLU.
pub fn relu(x: &Variable) -> Result<Variable, ModuleError> {
    ops::relu(x)
}

// ------------------------------ Sigmoid ------------------------------

/// Sigmoid activation.
#[derive(Default)]
pub struct Sigmoid;

impl Module for Sigmoid {
    fn forward(&self, input: &Variable) -> Result<Variable, ModuleError> {
        ops::sigmoid(input)
    }
}

/// Functional Sigmoid.
pub fn sigmoid(x: &Variable) -> Result<Variable, ModuleError> {
    ops::sigmoid(x)
}

// ------------------------------ Tanh ------------------------------

/// Tanh activation.
#[derive(Default)]
pub struct Tanh;

impl Module for Tanh {
    fn forward(&self, input: &Variable) -> Result<Variable, ModuleError> {
        ops::tanh(input)
    }
}

/// Functional Tanh.
pub fn tanh(x: &Variable) -> Result<Variable, ModuleError> {
    ops::tanh(x)
}

// ------------------------------ SiLU / Swish ------------------------------

/// SiLU / Swish activation (`x * sigmoid(x)`).
#[derive(Default)]
pub struct Silu;

impl Module for Silu {
    fn forward(&self, input: &Variable) -> Result<Variable, ModuleError> {
        ops::silu(input)
    }
}

/// Functional SiLU.
pub fn silu(x: &Variable) -> Result<Variable, ModuleError> {
    ops::silu(x)
}

// ------------------------------ LeakyReLU ------------------------------

/// LeakyReLU: `x` if `x > 0` else `slope * x`.
pub struct LeakyRelu {
    slope: f64,
}

impl LeakyRelu {
    /// Build a LeakyReLU with the given negative-slope.
    pub fn new(slope: f64) -> Self {
        LeakyRelu { slope }
    }
}

impl Default for LeakyRelu {
    /// PyTorch default of 0.01.
    fn default() -> Self {
        LeakyRelu { slope: 0.01 }
    }
}

impl Module for LeakyRelu {
    fn forward(&self, input: &Variable) -> Result<Variable, ModuleError> {
        ops::leaky_relu(input, self.slope)
    }
}

/// Functional LeakyReLU.
pub fn leaky_relu(x: &Variable, slope: f64) -> Result<Variable, ModuleError> {
    ops::leaky_relu(x, slope)
}

// ------------------------------ Softmax ------------------------------

/// Softmax along a fixed axis.
pub struct Softmax {
    dim: usize,
}

impl Softmax {
    /// Softmax along the given dimension.
    pub fn new(dim: usize) -> Self {
        Softmax { dim }
    }
}

impl Module for Softmax {
    fn forward(&self, input: &Variable) -> Result<Variable, ModuleError> {
        ops::softmax(input, self.dim)
    }
}

/// Functional Softmax.
pub fn softmax(x: &Variable, dim: usize) -> Result<Variable, ModuleError> {
    ops::softmax(x, dim)
}

// ------------------------------ LogSoftmax ------------------------------

/// LogSoftmax along a fixed axis.
pub struct LogSoftmax {
    dim: usize,
}

impl LogSoftmax {
    /// LogSoftmax along the given dimension.
    pub fn new(dim: usize) -> Self {
        LogSoftmax { dim }
    }
}

impl Module for LogSoftmax {
    fn forward(&self, input: &Variable) -> Result<Variable, ModuleError> {
        ops::log_softmax(input, self.dim)
    }
}

/// Functional LogSoftmax.
pub fn log_softmax(x: &Variable, dim: usize) -> Result<Variable, ModuleError> {
    ops::log_softmax(x, dim)
}

// ------------------------------ Mish ------------------------------

/// Mish activation (Misra 2019): `y = x * tanh(softplus(x))`.
///
/// Built via composition of existing autograd-aware ops
/// (exp/add/log/tanh/mul). Softplus is computed as `log(1 + exp(x))`.
/// Numerically stable for `x in [-50, 50]` (typical activation range);
/// extremely large `|x|` may overflow exp — pending kernel-level
/// stabilisation.
#[derive(Default)]
pub struct Mish;

impl Module for Mish {
    fn forward(&self, input: &Variable) -> Result<Variable, ModuleError> {
        mish(input)
    }
}

/// Functional Mish: `y = x * tanh(softplus(x))`.
pub fn mish(x: &Variable) -> Result<Variable, ModuleError> {
    use rustorch_core::tensor::tensor_impl::Tensor;
    let exp_x = ops::exp(x)?;
    let one = Variable::new(Tensor::scalar(1.0_f32));
    let one_plus_exp = ops::add(&one, &exp_x)?;
    let softplus_x = ops::log(&one_plus_exp)?;
    let tanh_sp = ops::tanh(&softplus_x)?;
    ops::mul(x, &tanh_sp)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustorch_autograd::{backward, Variable};
    use rustorch_core::tensor::tensor_impl::Tensor;

    #[test]
    fn mish_at_zero_is_zero() {
        // Mish(0) = 0 * tanh(softplus(0)) = 0
        let x = Variable::new(Tensor::from_vec([1usize], vec![0.0_f32]).unwrap());
        let y = mish(&x).unwrap();
        let y_t = y.tensor();
        let v = y_t.as_slice::<f32>().unwrap();
        assert!(v[0].abs() < 1e-6, "got {}", v[0]);
    }

    #[test]
    fn mish_large_x_approaches_x() {
        // For large x, softplus(x) ≈ x, tanh(softplus(x)) ≈ 1 → Mish(x) ≈ x
        let x = Variable::new(Tensor::from_vec([1usize], vec![10.0_f32]).unwrap());
        let y = mish(&x).unwrap();
        let y_t = y.tensor();
        let v = y_t.as_slice::<f32>().unwrap();
        assert!((v[0] - 10.0).abs() < 0.01, "got {}", v[0]);
    }

    #[test]
    fn mish_backward_runs_and_is_finite() {
        let x = Variable::leaf(Tensor::from_vec([4usize], vec![-2.0_f32, -0.5, 0.5, 2.0]).unwrap());
        let y = mish(&x).unwrap();
        let s = rustorch_autograd::ops::sum(&y).unwrap();
        backward(&s, None).unwrap();
        let g = x.grad().unwrap();
        let g_v = g.as_slice::<f32>().unwrap();
        for &gi in g_v {
            assert!(gi.is_finite(), "got non-finite grad {gi}");
        }
    }

    #[test]
    fn mish_module_matches_functional() {
        let m = Mish;
        let x = Variable::new(Tensor::from_vec([3usize], vec![0.5_f32, 1.0, 1.5]).unwrap());
        let y_mod = m.forward(&x).unwrap();
        let y_fn = mish(&x).unwrap();
        let mt = y_mod.tensor();
        let ft = y_fn.tensor();
        assert_eq!(mt.as_slice::<f32>().unwrap(), ft.as_slice::<f32>().unwrap());
    }
}
