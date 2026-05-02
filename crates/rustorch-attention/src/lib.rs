//! # rustorch-attention
//!
//! Flash-Attention-style tiled attention for rustorch — pure-Rust
//! CPU implementation built around an online (Welford-style) softmax
//! that aggregates `(max, sum)` per tile so the full N×N score
//! matrix never materialises in memory.
//!
//! Currently exposed:
//! - [`online_softmax`] — the streaming softmax primitive that lets
//!   the rest of the pipeline process Q/K/V in O(N) memory.
//!
//! The forward / backward / GPU paths land in subsequent commits
//! (see Phase 3 plan `Flash Attention (CPU + GPU)`).

#![cfg_attr(docsrs, feature(doc_cfg))]
#![warn(missing_docs)]
#![warn(rust_2018_idioms)]

pub mod cpu_forward;
pub mod online_softmax;

pub use cpu_forward::{flash_forward, naive_forward, AttentionError, AttentionShape};
pub use online_softmax::{combine_tiles, online_softmax_full, OnlineSoftmaxState};

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
