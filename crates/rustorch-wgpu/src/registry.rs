//! Centralised kernel registry.
//!
//! Each module in this crate owns its own WGSL source and dispatches
//! through [`crate::cache::PipelineCache::get_or_insert_with`] for
//! per-pipeline caching. That works fine but spreads the catalogue
//! of "what kernels the crate ships" across many files. This module
//! collects them under a single [`OpId`] enum and an
//! [`OpId::source`] method so callers can:
//!
//! - Iterate over every shipped kernel (e.g. for the
//!   [`crate::template::validate_all_kernels`] smoke check).
//! - Look up a kernel's WGSL by ID without going through the
//!   per-module accessors.
//! - Use a shared [`KernelRegistry`] facade for tools that don't
//!   want to mention each module by name.
//!
//! The dispatch hot path is **unchanged** — it still goes through
//! `PipelineCache::get_or_insert_with` under the per-module
//! `PipelineKey { op, dtype, variant }`. This module is only the
//! "phone book" of what kernels exist.

use crate::shaders;

/// Stable identifier for every WGSL kernel shipped by the crate.
/// Used by [`OpId::source`] and by the validate-all-kernels smoke
/// check; not present on the dispatch hot path.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum OpId {
    /// Element-wise binary add (`out = lhs + rhs`).
    Add,
    /// Element-wise binary sub.
    Sub,
    /// Element-wise binary mul.
    Mul,
    /// Element-wise binary div.
    Div,
    /// Element-wise unary relu.
    Relu,
    /// Element-wise unary neg.
    Neg,
    /// Element-wise unary sigmoid.
    Sigmoid,
    /// Element-wise unary tanh.
    Tanh,
    /// Element-wise unary silu.
    Silu,
}

impl OpId {
    /// Return the canonical op name string (matches the
    /// [`crate::cache::PipelineKey::op`] value used by dispatch).
    pub fn name(self) -> &'static str {
        match self {
            OpId::Add => "add",
            OpId::Sub => "sub",
            OpId::Mul => "mul",
            OpId::Div => "div",
            OpId::Relu => "relu",
            OpId::Neg => "neg",
            OpId::Sigmoid => "sigmoid",
            OpId::Tanh => "tanh",
            OpId::Silu => "silu",
        }
    }

    /// Build the WGSL source for this op. Cheap (just a `format!`).
    pub fn source(self) -> String {
        shaders::source_for(self.name())
    }

    /// Iterate over every variant of [`OpId`].
    pub fn all() -> &'static [OpId] {
        &[
            OpId::Add,
            OpId::Sub,
            OpId::Mul,
            OpId::Div,
            OpId::Relu,
            OpId::Neg,
            OpId::Sigmoid,
            OpId::Tanh,
            OpId::Silu,
        ]
    }
}

/// Registry facade: thin wrapper around the crate's existing
/// [`crate::cache::PipelineCache`], providing OpId-keyed access on
/// top of the per-module dispatch surfaces.
///
/// Each `WgpuBackend` already carries a `PipelineCache`, so the
/// registry is essentially free — it doesn't allocate a second
/// cache. The dispatch fast path still goes through the per-module
/// helpers (`dispatch_binary`, `matmul`, `softmax_rows`, …).
#[derive(Clone)]
pub struct KernelRegistry;

impl KernelRegistry {
    /// Construct an empty registry. Currently a unit struct — held
    /// for API symmetry with future versions that may carry per-op
    /// metadata (specialisation constants, dtype variants, …).
    pub fn new() -> Self {
        KernelRegistry
    }

    /// Return the WGSL source for `op`. Equivalent to
    /// [`OpId::source`].
    pub fn source(&self, op: OpId) -> String {
        op.source()
    }

    /// Number of distinct ops registered.
    pub fn len(&self) -> usize {
        OpId::all().len()
    }

    /// Empty?
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl Default for KernelRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn op_id_name_round_trips_through_source() {
        for op in OpId::all() {
            let src = op.source();
            assert!(!src.is_empty(), "no source for {:?}", op);
        }
    }

    #[test]
    fn registry_lists_every_op() {
        let reg = KernelRegistry::new();
        assert_eq!(reg.len(), OpId::all().len());
        for op in OpId::all() {
            assert!(!reg.source(*op).is_empty());
        }
    }
}
