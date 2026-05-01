//! # rustorch-core
//!
//! The zero-ML foundation of rustorch: `Tensor`, `Storage`, `Layout`, `Dtype`.
//!
//! This crate is intentionally backend-agnostic. Concrete computation lives in
//! `rustorch-cpu`, `rustorch-cuda`, `rustorch-wgpu`. The autograd engine lives
//! in `rustorch-autograd`.
//!
//! Per RFC-0002, the public API is stable from v0.7.2 onward.

#![cfg_attr(docsrs, feature(doc_cfg))]
#![warn(missing_docs)]
#![warn(rust_2018_idioms)]

// Public modules will be filled in by P1.1 (Tensor core). Stubs for now.
pub mod dtype;
pub mod layout;
pub mod shape;
pub mod storage;
pub mod tensor;

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
