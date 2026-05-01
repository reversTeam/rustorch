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

    /// Trainable parameters paired with their local names. Default impl
    /// derives names from index (`"0"`, `"1"`, …); composite modules
    /// override to expose meaningful names like `"weight"` / `"bias"` /
    /// child paths (e.g. `"0.weight"` for a Sequential).
    ///
    /// Used by [`crate::state_dict::state_dict`] to build a flat
    /// HashMap keyed by dotted path.
    fn named_parameters(&self) -> Vec<(String, Variable)> {
        self.parameters()
            .into_iter()
            .enumerate()
            .map(|(i, v)| (i.to_string(), v))
            .collect()
    }

    /// Switch to training mode (Dropout active, BatchNorm updates running
    /// stats). Default is no-op for stateless modules.
    fn train(&mut self) {}

    /// Switch to eval mode.
    fn eval(&mut self) {}
}
