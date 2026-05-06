//! Sequence-pooling building blocks. Currently:
//! - [`CrossAttentionPool`] — Perceiver / Q-Former / BLIP-2 style
//!   pooling of `[B, T, D]` to `[B, num_queries, D]` via cross-attention
//!   from a learnable query bank to the input.

pub mod cross_attention;

pub use cross_attention::CrossAttentionPool;
