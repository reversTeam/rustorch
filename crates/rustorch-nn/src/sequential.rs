//! `Sequential` — chained module composition.
//!
//! ```ignore
//! let net = Sequential::new()
//!     .add(Linear::new(784, 256))
//!     .add(Relu)
//!     .add(Linear::new(256, 10));
//! ```

use crate::module::{Module, ModuleError};
use rustorch_autograd::Variable;

/// Sequential composition of [`Module`]s.
///
/// Forward = run inputs through every child in order. Parameters =
/// concatenation of every child's `parameters()`.
#[derive(Default)]
pub struct Sequential {
    modules: Vec<Box<dyn Module>>,
}

impl Sequential {
    /// Empty `Sequential`.
    pub fn new() -> Self {
        Sequential {
            modules: Vec::new(),
        }
    }

    /// Push a child module (builder-style).
    #[must_use]
    #[allow(clippy::should_implement_trait)]
    pub fn add<M: Module + 'static>(mut self, m: M) -> Self {
        self.modules.push(Box::new(m));
        self
    }

    /// Number of child modules.
    pub fn len(&self) -> usize {
        self.modules.len()
    }

    /// Empty?
    pub fn is_empty(&self) -> bool {
        self.modules.is_empty()
    }
}

impl Module for Sequential {
    fn forward(&self, input: &Variable) -> Result<Variable, ModuleError> {
        let mut x = input.clone();
        for m in &self.modules {
            x = m.forward(&x)?;
        }
        Ok(x)
    }

    fn parameters(&self) -> Vec<Variable> {
        let mut params = Vec::new();
        for m in &self.modules {
            params.extend(m.parameters());
        }
        params
    }

    fn train(&mut self) {
        for m in &mut self.modules {
            m.train();
        }
    }

    fn eval(&mut self) {
        for m in &mut self.modules {
            m.eval();
        }
    }
}
