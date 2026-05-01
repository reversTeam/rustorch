//! # rustorch-data
//!
//! `Dataset` and `IterableDataset` traits + `DataLoader` with batched
//! iteration + `Sampler` impls (Sequential, Random — others pending).
//!
//! v1 is single-threaded. Workers (`num_workers` via rayon) land in a
//! follow-up; the iterator API is stable. Augmentations live in P1.9.

#![cfg_attr(docsrs, feature(doc_cfg))]
#![warn(missing_docs)]
#![warn(rust_2018_idioms)]

pub mod augment;
pub mod bundled;
pub mod dataloader;
pub mod dataset;
pub mod sampler;

pub use augment::{Augment, AugmentError, HorizontalFlip, Mixup, Normalize, RandomHorizontalFlip};
pub use bundled::{synthetic_cifar10, synthetic_mnist};
pub use dataloader::{DataLoader, DataLoaderError};
pub use dataset::{Dataset, DatasetError, IterableDataset, TensorDataset};
pub use sampler::{
    DistributedSampler, RandomSampler, Sampler, SequentialSampler, WeightedRandomSampler,
};

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
