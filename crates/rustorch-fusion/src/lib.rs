//! # rustorch-fusion
//!
//! Eager-mode op fusion for rustorch — combines short chains of ops
//! (matmul+bias+activation, layernorm+linear, residual add+mul,
//! elementwise epilogue chains) into single SIMD-friendly passes
//! over the output buffer.
//!
//! ## Modules
//! - [`pattern`] — Pattern DSL + match results
//! - [`matcher`] — DAG matcher engine (post-order traversal)
//! - [`patterns`] — hand-written fused kernel library
//!
//! ## What's deferred
//! - The `fuse!` proc-macro (compile-time codegen) lands in a
//!   sibling `rustorch-fusion-macros` crate later.
//! - Eager dispatcher integration + autograd backward registry land
//!   when rustorch-autograd grows a stable custom-Function API
//!   (Phase 3 plan `Gradient Checkpointing` task `Autograd tape
//!   integration`).

#![cfg_attr(docsrs, feature(doc_cfg))]
#![warn(missing_docs)]
#![warn(rust_2018_idioms)]

pub mod dispatcher;
pub mod matcher;
pub mod pattern;
pub mod patterns;

pub use dispatcher::{no_fuse, set_no_fuse, FuseDecision, FusionRegistry, NoFuseGuard};
pub use matcher::{find_matches, Match};
pub use pattern::{OpKind, Pattern};
pub use patterns::{
    elementwise_chain, fused_matmul_bias_activation, naive_matmul_bias_activation, Activation,
    FusionError,
};

/// Crate version reported at runtime.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
