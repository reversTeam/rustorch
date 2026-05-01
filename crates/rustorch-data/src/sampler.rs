//! `Sampler` trait + impls (P1.8).
//!
//! Samplers produce indices into a [`Dataset`]. The DataLoader pulls
//! indices from the sampler then fetches items via `dataset.get(idx)`.
//!
//! v1 ships:
//! - [`SequentialSampler`] — `0..len` in order.
//! - [`RandomSampler`] — Fisher-Yates shuffle, seedable.
//!
//! WeightedRandomSampler / BucketBy / Distributed land in follow-ups
//! per the project plan.

use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;

/// Sampler produces a fresh sequence of indices each time `iter()` is
/// called. Iteration order may differ across calls (e.g. shuffled).
pub trait Sampler {
    /// Return the indices for one full epoch.
    fn iter(&mut self) -> Vec<usize>;
    /// Number of indices per epoch.
    fn len(&self) -> usize;
    /// True iff `len() == 0`.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

// ------------------------------ SequentialSampler ------------------------------

/// Iterate `0..n` in order.
pub struct SequentialSampler {
    n: usize,
}

impl SequentialSampler {
    /// Build over a dataset of `n` items.
    pub fn new(n: usize) -> Self {
        SequentialSampler { n }
    }
}

impl Sampler for SequentialSampler {
    fn iter(&mut self) -> Vec<usize> {
        (0..self.n).collect()
    }
    fn len(&self) -> usize {
        self.n
    }
}

// ------------------------------ RandomSampler ------------------------------

/// Fisher-Yates shuffle of `0..n` per epoch. Uses a re-seedable RNG
/// for reproducibility.
pub struct RandomSampler {
    n: usize,
    rng: StdRng,
}

impl RandomSampler {
    /// Build with the given size and a time-derived seed (call
    /// [`Self::with_seed`] for reproducible shuffles).
    pub fn new(n: usize) -> Self {
        let seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0);
        RandomSampler {
            n,
            rng: StdRng::seed_from_u64(seed),
        }
    }

    /// Build with a fixed seed (reproducible shuffle order).
    pub fn with_seed(n: usize, seed: u64) -> Self {
        RandomSampler {
            n,
            rng: StdRng::seed_from_u64(seed),
        }
    }
}

impl Sampler for RandomSampler {
    fn iter(&mut self) -> Vec<usize> {
        let mut idx: Vec<usize> = (0..self.n).collect();
        idx.shuffle(&mut self.rng);
        idx
    }
    fn len(&self) -> usize {
        self.n
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sequential_returns_in_order() {
        let mut s = SequentialSampler::new(5);
        assert_eq!(s.iter(), vec![0, 1, 2, 3, 4]);
        // Calling iter again returns the same order.
        assert_eq!(s.iter(), vec![0, 1, 2, 3, 4]);
        assert_eq!(s.len(), 5);
    }

    #[test]
    fn random_seeded_reproducible() {
        let mut s1 = RandomSampler::with_seed(10, 42);
        let mut s2 = RandomSampler::with_seed(10, 42);
        assert_eq!(s1.iter(), s2.iter());
    }

    #[test]
    fn random_shuffles_off_identity() {
        let mut s = RandomSampler::with_seed(20, 7);
        let order = s.iter();
        // It's astronomically unlikely a 20-elem shuffle equals identity.
        assert_ne!(order, (0..20).collect::<Vec<_>>());
    }

    #[test]
    fn random_yields_a_permutation() {
        let mut s = RandomSampler::with_seed(30, 0xC0FFEE);
        let order = s.iter();
        let mut sorted = order.clone();
        sorted.sort();
        assert_eq!(sorted, (0..30).collect::<Vec<_>>());
    }

    #[test]
    fn empty_samplers_yield_empty() {
        let mut s = SequentialSampler::new(0);
        assert!(s.iter().is_empty());
        assert!(s.is_empty());
        let mut r = RandomSampler::with_seed(0, 0);
        assert!(r.iter().is_empty());
        assert!(r.is_empty());
    }
}
