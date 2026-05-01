//! `Module` trait — the common API for every nn building block
//! (P1.6 task `Module trait`).
//!
//! v1 ships the minimum surface needed to assemble + train a Linear+
//! ReLU MLP:
//! - `forward(input) -> Result<output>` for the actual computation
//! - `parameters()` returning every trainable [`Variable`] inside the
//!   module (recursively into children)
//! - `train()` / `eval()` toggling — drives Dropout and BatchNorm in
//!   later iterations.
//!
//! Helper [`ModuleError`] wraps autograd errors uniformly.

use rustorch_autograd::{BackwardError, Variable};

/// Errors returned by [`Module::forward`].
pub type ModuleError = BackwardError;

/// Common trait for every neural-network module.
pub trait Module: Send + Sync {
    /// Compute the forward pass.
    fn forward(&self, input: &Variable) -> Result<Variable, ModuleError>;

    /// Trainable parameters owned by this module + every child module.
    /// Order is stable across calls (used by optimisers to map slot →
    /// gradient).
    fn parameters(&self) -> Vec<Variable> {
        Vec::new()
    }

    /// Switch to training mode (Dropout active, BatchNorm updates running
    /// stats). Default is no-op for stateless modules.
    fn train(&mut self) {}

    /// Switch to eval mode.
    fn eval(&mut self) {}
}
