//! Module-able activations. Each unit struct implements [`Module`]
//! and forwards to the corresponding autograd op.

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
