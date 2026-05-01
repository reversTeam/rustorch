//! # rustorch-nn — Neural network building blocks (P1.6).
//!
//! Public surface:
//! - [`Module`] — common trait (forward, parameters, train/eval).
//! - [`Linear`] — y = x @ W + b (rustorch convention: weight is [in, out]).
//! - Activation functions and Module wrappers: [`relu`] / [`sigmoid`] /
//!   [`tanh`] / [`silu`] / [`leaky_relu`] / [`softmax`] / [`log_softmax`]
//!   and [`Relu`] / [`Sigmoid`] / [`Tanh`] / [`Silu`] / [`LeakyRelu`] /
//!   [`Softmax`] / [`LogSoftmax`].
//! - Loss modules via [`Criterion`]: [`MseLoss`], [`CrossEntropyLoss`].
//! - [`Sequential`] — chained Module composition.

#![cfg_attr(docsrs, feature(doc_cfg))]
#![warn(missing_docs)]
#![warn(rust_2018_idioms)]

pub mod activation;
pub mod container;
pub mod linear;
pub mod loss;
pub mod module;
pub mod norm;
pub mod sequential;

pub use activation::{
    leaky_relu, log_softmax, relu, sigmoid, silu, softmax, tanh, LeakyRelu, LogSoftmax, Relu,
    Sigmoid, Silu, Softmax, Tanh,
};
pub use container::{ModuleDict, ModuleList};
pub use linear::Linear;
pub use loss::{Criterion, CrossEntropyLoss, MseLoss};
pub use module::Module;
pub use norm::{LayerNorm, RMSNorm};
pub use sequential::Sequential;

/// Crate version reported at runtime.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
