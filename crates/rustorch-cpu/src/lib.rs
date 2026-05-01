//! # rustorch-cpu
//!
//! CPU backend for rustorch. Aligned 64B allocator, rayon parallelism, SIMD
//! abstraction, full implementation of the `Backend` trait.
//!
//! Concrete impl lands in P1.2 (CPU backend infrastructure).

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
