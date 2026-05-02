//! Hyperparameter sweep planner. Aligned with doc v0.7.2 §
//! `/Recipes/Hyperparameter sweeps`. Pure-data crate — sweep specs
//! turn into concrete `Trial`s; the actual run launching is the
//! Console backend's job (P2.2 + P2.1).
//!
//! Strategies shipped:
//! * `Grid`   — full Cartesian product (cap ≤ ~30 combos in practice).
//! * `Random` — uniform sampling, seeded for reproducibility.
//! * `Asha`   — asynchronous successive halving — emit a series of
//!   trials at increasing resource budgets, prune the bottom 1/η at
//!   each rung. v1 returns the rung schedule; the actual prune
//!   decisions are wired in by the orchestrator.
//! * `Bayes`  — interface only. The GP + acquisition implementation
//!   is deferred (multi-week project); v1 falls back to Random with
//!   a warning so callers can still ship.
//!
//! API shape mirrors the doc:
//!
//! ```rust,ignore
//! let spec = SweepSpec::grid()
//!     .add("lr", &json!([1e-4, 3e-4, 1e-3]))
//!     .add("batch", &json!([64, 128]))
//!     .build();
//! let trials = spec.plan(&base_cfg)?;
//! ```

#![allow(missing_docs)]

mod range;
mod spec;
pub use range::{linspace, logspace};
pub use spec::{
    AshaConfig, BayesConfig, RandomConfig, Strategy, SweepBuilder, SweepError, SweepResult,
    SweepSpec, Trial,
};
