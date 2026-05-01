//! # rustorch-data
//!
//! `Dataset` and `IterableDataset` traits + `DataLoader` with workers (rayon)
//! + `Sampler` impls (Sequential, Random, WeightedRandom, BucketBy, Distributed).
//!
//! Concrete impl lands in P1.8 (DataLoader). Augmentations live in P1.9.

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
