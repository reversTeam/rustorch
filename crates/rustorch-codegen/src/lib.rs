//! # rustorch-codegen
//!
//! Build-time helpers for rustorch. Parses `ops.yaml` (the canonical op schema)
//! and emits Rust source via `quote!`. Used by build.rs in `rustorch-core` and
//! downstream backends to generate dispatch boilerplate from a single source
//! of truth.
//!
//! Concrete impl lands in P0.1 (RFC-0005 Codegen strategy).

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
