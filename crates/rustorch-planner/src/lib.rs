//! # rustorch-planner
//!
//! Memory planner for rustorch — analyses the lifetime of every
//! intermediate tensor in an autograd-style execution trace and packs
//! them into a small set of physical buffers via interval-disjoint
//! union-find.
//!
//! Inspired by torch.compile/Inductor's "memory planning" pass:
//! given an ordered list of `(def, last_use)` intervals, two tensors
//! can share the same physical slot iff their intervals don't
//! overlap. The planner builds that mapping ahead of execution so the
//! runtime can pre-allocate exactly `peak_concurrent_bytes` and reuse
//! freed slots. Typical savings on a Transformer block: 30-50% peak
//! activation memory vs. the naive "every tensor gets its own
//! buffer" baseline.
//!
//! Public surface:
//! - [`lifetime`] — [`Interval`], [`LifetimeTable`], + a builder
//! - [`allocator`] — [`PlanResult`], [`Slot`], + the planner itself
//! - [`plan`] — one-shot convenience entry-point
//!
//! The planner is **pure**: it operates on integer tensor IDs and
//! byte sizes; it does not allocate any GPU/CPU memory itself.

#![cfg_attr(docsrs, feature(doc_cfg))]
#![warn(missing_docs)]
#![warn(rust_2018_idioms)]

pub mod allocator;
pub mod inplace;
pub mod lifetime;

pub use allocator::{plan, plan_with_budget, PlanError, PlanResult, Slot};
pub use inplace::{
    evaluate_hint, plan_with_inplace, InPlaceBlockers, InPlaceDecision, InPlaceHint,
};
pub use lifetime::{Interval, LifetimeError, LifetimeTable, TensorId};

/// Crate version reported at runtime.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smoke_version_present() {
        assert!(!VERSION.is_empty());
    }
}
