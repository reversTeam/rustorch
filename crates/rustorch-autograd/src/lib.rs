//! # rustorch-autograd
//!
//! Reverse-mode automatic differentiation. Tape-based, thread-local.
//! Per RFC-0003 (P0.1).
//!
//! Concrete impl lands in P1.5 (Autograd engine).

#![cfg_attr(docsrs, feature(doc_cfg))]
#![warn(rust_2018_idioms)]

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
