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
