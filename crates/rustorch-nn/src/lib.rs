//! # rustorch-nn — Neural network building blocks (P1.6).
//!
//! Public surface:
//! - [`Module`] — common trait (forward, parameters, train/eval).
//! - [`Linear`] — y = x @ W + b (rustorch convention: weight is [in, out]).
//! - [`relu`] / [`sigmoid`] / [`tanh`] — functional activations.
//! - [`Relu`] / [`Sigmoid`] / [`Tanh`] — Module-able variants for use in
//!   [`Sequential`].
//! - [`Sequential`] — chained Module composition.

#![cfg_attr(docsrs, feature(doc_cfg))]
#![warn(missing_docs)]
#![warn(rust_2018_idioms)]

pub mod activation;
pub mod linear;
pub mod module;
pub mod sequential;

pub use activation::{relu, sigmoid, tanh, Relu, Sigmoid, Tanh};
pub use linear::Linear;
pub use module::Module;
pub use sequential::Sequential;

/// Crate version reported at runtime.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
