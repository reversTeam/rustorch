//! # rustorch-nn — Neural network building blocks (P1.6).
//!
//! ## Public surface
//!
//! ### Core
//! - [`Module`] — common trait (forward, parameters, train/eval, to_dtype).
//! - [`Linear`] — `y = x @ W + b` with PyTorch parity for arbitrary
//!   leading batch dims (`[*, in_features] -> [*, out_features]`).
//! - [`Sequential`] — chained Module composition.
//!
//! ### Activations & losses
//! - Activation functions and Module wrappers: [`relu`] / [`sigmoid`] /
//!   [`tanh`] / [`silu`] / [`leaky_relu`] / [`softmax`] / [`log_softmax`]
//!   and [`Relu`] / [`Sigmoid`] / [`Tanh`] / [`Silu`] / [`LeakyRelu`] /
//!   [`Softmax`] / [`LogSoftmax`].
//! - Loss modules via [`Criterion`]: [`MseLoss`], [`CrossEntropyLoss`],
//!   [`CTCLoss`].
//!
//! ### Normalisation
//! - [`LayerNorm`], [`RMSNorm`], [`BatchNorm2d`].
//!
//! ### Convolution & RNN
//! - [`Conv1d`], [`Conv2d`], [`MaxPool2d`].
//! - [`RnnCell`], [`LstmCell`].
//!
//! ### Embedding & positional
//! - [`Embedding`] — vocabulary lookup.
//! - [`SinusoidalPositionalEncoding`] (Vaswani et al. 2017, non-trainable),
//!   [`LearnedPositionalEncoding`] (trainable `[max_len, dim]` table).
//!
//! ### Attention
//! - [`scaled_dot_product_attention`] — stateless rank-3 SDP.
//! - [`SingleHeadAttention`] — Q/K/V/O projections, single head.
//! - [`MultiHeadAttention`] — `torch.nn.MultiheadAttention` parity, with
//!   self- and cross-attention. Rank-3 fold-batch path internally.
//! - [`causal_mask`], [`sliding_window_mask`], [`bool_to_additive`] —
//!   autograd-aware Variable masks (additive bias, [`MASK_NEG`] for
//!   masked positions).
//! - [`CrossAttentionPool`] — Perceiver / Q-Former / BLIP-2 style pooling
//!   from `[B, T, D]` to `[B, num_queries, D]`.
//!
//! ### Containers
//! - [`ModuleDict`], [`ModuleList`].
//! - [`Checkpointed`] — gradient-checkpointing wrapper.
//! - [`HookedModule`], [`HookHandle`] — pre/post forward hooks.
//!
//! ### State dict
//! - [`state_dict`], [`load_state_dict`], [`LoadReport`], [`StateDictError`].

#![cfg_attr(docsrs, feature(doc_cfg))]
#![warn(missing_docs)]
#![warn(rust_2018_idioms)]

pub mod activation;
pub mod attention;
pub mod checkpointed;
pub mod container;
pub mod conv;
pub mod embedding;
pub mod gqa;
pub mod hooks;
pub mod init;
pub mod kv_cache;
pub mod linear;
pub mod loss;
pub mod masks;
pub mod module;
pub mod norm;
pub mod pool;
pub mod positional;
pub mod rnn;
pub mod rope;
pub mod sampling;
pub mod sequential;
pub mod state_dict;

pub use activation::{
    leaky_relu, log_softmax, mish, relu, sigmoid, silu, softmax, tanh, LeakyRelu, LogSoftmax, Mish,
    Relu, Sigmoid, Silu, Softmax, Tanh,
};
pub use attention::{scaled_dot_product_attention, MultiHeadAttention, SingleHeadAttention};
pub use checkpointed::Checkpointed;
pub use container::{ModuleDict, ModuleList};
pub use conv::{Conv1d, Conv2d, MaxPool2d};
pub use embedding::Embedding;
pub use hooks::{HookHandle, HookedModule};
pub use init::{calculate_fan, init, init_with_seed, FanMode, Init, Nonlinearity};
pub use linear::{Bilinear, Linear};
pub use loss::{CTCLoss, Criterion, CrossEntropyLoss, MseLoss};
pub use masks::{bool_to_additive, causal_mask, sliding_window_mask, MASK_NEG};
pub use module::{Buffer, Module, Parameter};
pub use norm::{BatchNorm2d, LayerNorm, RMSNorm};
pub use pool::CrossAttentionPool;
pub use positional::{LearnedPositionalEncoding, SinusoidalPositionalEncoding};
pub use rnn::{LstmCell, RnnCell};
pub use sequential::Sequential;
pub use state_dict::{load_state_dict, state_dict, LoadReport, StateDictError};

/// Crate version reported at runtime.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
