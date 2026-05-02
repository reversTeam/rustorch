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
pub mod attention;
pub mod checkpointed;
pub mod container;
pub mod conv;
pub mod embedding;
pub mod hooks;
pub mod init;
pub mod linear;
pub mod loss;
pub mod module;
pub mod norm;
pub mod rnn;
pub mod sequential;
pub mod state_dict;

pub use activation::{
    leaky_relu, log_softmax, mish, relu, sigmoid, silu, softmax, tanh, LeakyRelu, LogSoftmax, Mish,
    Relu, Sigmoid, Silu, Softmax, Tanh,
};
pub use attention::{scaled_dot_product_attention, SingleHeadAttention};
pub use checkpointed::Checkpointed;
pub use container::{ModuleDict, ModuleList};
pub use conv::{Conv1d, Conv2d, MaxPool2d};
pub use embedding::Embedding;
pub use hooks::{HookHandle, HookedModule};
pub use init::{calculate_fan, init, init_with_seed, FanMode, Init, Nonlinearity};
pub use linear::{Bilinear, Linear};
pub use loss::{CTCLoss, Criterion, CrossEntropyLoss, MseLoss};
pub use module::{Buffer, Module, Parameter};
pub use norm::{BatchNorm2d, LayerNorm, RMSNorm};
pub use rnn::{LstmCell, RnnCell};
pub use sequential::Sequential;
pub use state_dict::{load_state_dict, state_dict, LoadReport, StateDictError};

/// Crate version reported at runtime.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
