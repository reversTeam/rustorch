//! Eager dispatcher integration — runtime selection of fused
//! kernels via a global flag + a `FusionRegistry` that maps op
//! patterns to fused implementations.
//!
//! This is the host-side glue that the autograd would call when it
//! sees a recognised chain in the trace. The actual autograd custom-
//! Function wiring lives in `rustorch-fusion-autograd` (separate
//! sub-crate that depends on rustorch-autograd) — kept out of THIS
//! crate so rustorch-fusion stays autograd-free.

use crate::pattern::Pattern;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

/// Process-wide flag — when `false`, the dispatcher skips fusion
/// entirely (useful for debugging numerical mismatches).
pub static NO_FUSE: AtomicBool = AtomicBool::new(false);

/// Read the current `no_fuse` flag.
pub fn no_fuse() -> bool {
    NO_FUSE.load(Ordering::Relaxed)
}

/// Set the process-wide `no_fuse` flag.
pub fn set_no_fuse(disabled: bool) {
    NO_FUSE.store(disabled, Ordering::Relaxed);
}

/// RAII guard that locally disables fusion within a scope (and
/// restores the prior state on drop).
pub struct NoFuseGuard {
    prev: bool,
}

impl NoFuseGuard {
    /// Enter a no-fuse scope.
    pub fn enter() -> Self {
        let prev = no_fuse();
        set_no_fuse(true);
        Self { prev }
    }
}

impl Drop for NoFuseGuard {
    fn drop(&mut self) {
        set_no_fuse(self.prev);
    }
}

/// Decision about whether to fuse a particular op chain.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FuseDecision {
    /// Apply the fused kernel.
    Apply,
    /// Skip — fall back to the un-fused chain.
    Skip,
}

/// Registry of fusion patterns. Lookup is `O(N)` over registered
/// patterns; for the 15-pattern default library this is negligible.
#[derive(Default, Clone)]
pub struct FusionRegistry {
    patterns: Vec<Arc<Pattern>>,
}

impl FusionRegistry {
    /// Empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a pattern. The order of registration is preserved;
    /// callers typically register longer patterns first so the
    /// dispatcher prefers them on overlap.
    pub fn register(&mut self, pattern: Pattern) {
        self.patterns.push(Arc::new(pattern));
    }

    /// Number of registered patterns.
    pub fn len(&self) -> usize {
        self.patterns.len()
    }

    /// Empty registry?
    pub fn is_empty(&self) -> bool {
        self.patterns.is_empty()
    }

    /// Iterate over the registered patterns.
    pub fn iter(&self) -> impl Iterator<Item = &Pattern> {
        self.patterns.iter().map(|p| p.as_ref())
    }

    /// Decide whether to fuse `pattern_label` — returns Skip if
    /// `NO_FUSE` is set globally OR the pattern isn't registered.
    pub fn decide(&self, pattern_label: &str) -> FuseDecision {
        if no_fuse() {
            return FuseDecision::Skip;
        }
        if self.patterns.iter().any(|p| p.label == pattern_label) {
            FuseDecision::Apply
        } else {
            FuseDecision::Skip
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pattern::{default_library, OpKind};

    #[test]
    fn empty_registry_skips_everything() {
        let r = FusionRegistry::new();
        assert_eq!(r.decide("matmul+bias+relu"), FuseDecision::Skip);
    }

    #[test]
    fn registered_pattern_is_applied() {
        let mut r = FusionRegistry::new();
        for p in default_library() {
            r.register(p);
        }
        assert_eq!(r.decide("matmul+bias+relu"), FuseDecision::Apply);
        assert_eq!(r.decide("layernorm+linear"), FuseDecision::Apply);
        assert_eq!(r.decide("never+seen"), FuseDecision::Skip);
    }

    #[test]
    fn no_fuse_flag_disables_dispatcher() {
        let mut r = FusionRegistry::new();
        r.register(Pattern::new("test", vec![OpKind::Matmul]));
        assert_eq!(r.decide("test"), FuseDecision::Apply);
        let _g = NoFuseGuard::enter();
        assert_eq!(r.decide("test"), FuseDecision::Skip);
        // _g dropped at end of scope → flag restored.
    }

    #[test]
    fn no_fuse_guard_restores_prior_state_on_drop() {
        set_no_fuse(false);
        {
            let _g = NoFuseGuard::enter();
            assert!(no_fuse());
        }
        // After drop, flag is back to false.
        assert!(!no_fuse());
    }

    #[test]
    fn no_fuse_guard_nested_state_handled_correctly() {
        set_no_fuse(false);
        {
            let _g1 = NoFuseGuard::enter();
            assert!(no_fuse());
            {
                let _g2 = NoFuseGuard::enter();
                assert!(no_fuse());
            }
            // g2 drop restores prev state (which was true from g1).
            assert!(no_fuse());
        }
        assert!(!no_fuse());
    }
}
