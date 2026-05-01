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

// ------------------------------ WeightedRandomSampler ------------------------------

use rand::Rng;

/// Sample indices with replacement, drawn proportionally to per-index
/// weights. Useful for class-imbalanced training.
pub struct WeightedRandomSampler {
    weights: Vec<f32>,
    /// Cumulative-distribution function of weights (sorted prefix sums).
    cdf: Vec<f32>,
    n_samples: usize,
    rng: StdRng,
}

impl WeightedRandomSampler {
    /// Build with per-index `weights` (must be non-negative and not all
    /// zero) and `n_samples` to draw per epoch.
    pub fn new(weights: Vec<f32>, n_samples: usize) -> Self {
        Self::with_seed(weights, n_samples, 0)
    }

    /// Build with a fixed seed for reproducible draws.
    pub fn with_seed(weights: Vec<f32>, n_samples: usize, seed: u64) -> Self {
        let total: f32 = weights.iter().sum();
        assert!(total > 0.0, "weights must have positive sum");
        let mut cdf = Vec::with_capacity(weights.len());
        let mut acc = 0.0_f32;
        for &w in &weights {
            assert!(w >= 0.0, "weights must be non-negative");
            acc += w / total;
            cdf.push(acc);
        }
        WeightedRandomSampler {
            weights,
            cdf,
            n_samples,
            rng: StdRng::seed_from_u64(seed),
        }
    }
}

impl Sampler for WeightedRandomSampler {
    fn iter(&mut self) -> Vec<usize> {
        let mut out = Vec::with_capacity(self.n_samples);
        for _ in 0..self.n_samples {
            let u: f32 = self.rng.gen_range(0.0_f32..1.0);
            // Binary search for the smallest index with cdf > u.
            let idx = self
                .cdf
                .partition_point(|&c| c <= u)
                .min(self.weights.len() - 1);
            out.push(idx);
        }
        out
    }
    fn len(&self) -> usize {
        self.n_samples
    }
}

// ------------------------------ DistributedSampler ------------------------------

/// Subsample indices `0..n` for one rank in a `world_size`-way data
/// parallel split. Indices are partitioned by `i % world_size == rank`.
///
/// Reseed per epoch via [`Self::set_epoch`] for shuffle-friendly ranks.
pub struct DistributedSampler {
    n: usize,
    rank: usize,
    world_size: usize,
    epoch: u64,
}

impl DistributedSampler {
    /// Build with the given total `n`, `rank`, and `world_size`.
    pub fn new(n: usize, rank: usize, world_size: usize) -> Self {
        assert!(world_size > 0);
        assert!(rank < world_size);
        DistributedSampler {
            n,
            rank,
            world_size,
            epoch: 0,
        }
    }

    /// Set the epoch counter (used as part of the shuffle seed).
    pub fn set_epoch(&mut self, epoch: u64) {
        self.epoch = epoch;
    }
}

impl Sampler for DistributedSampler {
    fn iter(&mut self) -> Vec<usize> {
        // Stride partition: rank takes indices [rank, rank+W, rank+2W, ...].
        // Apply a deterministic per-epoch shuffle to the source range first
        // so each rank sees fresh ordering across epochs.
        let mut all: Vec<usize> = (0..self.n).collect();
        let mut rng = StdRng::seed_from_u64(self.epoch);
        all.shuffle(&mut rng);
        all.into_iter()
            .enumerate()
            .filter_map(|(i, v)| {
                if i % self.world_size == self.rank {
                    Some(v)
                } else {
                    None
                }
            })
            .collect()
    }
    fn len(&self) -> usize {
        // Per-rank slice size: ceil(n / world_size) when rank < n%W else floor.
        let base = self.n / self.world_size;
        let extra = if self.rank < self.n % self.world_size {
            1
        } else {
            0
        };
        base + extra
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

    // -------------------- WeightedRandomSampler --------------------

    #[test]
    fn weighted_sampler_picks_high_weight_more_often() {
        // 3 classes, weight on class 2 is 100x the others. Over 10k draws,
        // class 2 should dominate.
        let mut s = WeightedRandomSampler::with_seed(vec![1.0, 1.0, 100.0], 10_000, 42);
        let idx = s.iter();
        let class2_count = idx.iter().filter(|&&i| i == 2).count();
        assert!(
            class2_count > 9_000,
            "class 2 should dominate, got {class2_count}/10000"
        );
    }

    #[test]
    fn weighted_sampler_with_uniform_weights_is_roughly_uniform() {
        let mut s = WeightedRandomSampler::with_seed(vec![1.0; 4], 4_000, 7);
        let idx = s.iter();
        for c in 0..4 {
            let count = idx.iter().filter(|&&i| i == c).count();
            // Expect 1000 ± 100 (3σ for a binomial with p=0.25, n=4000).
            assert!(
                count > 850 && count < 1150,
                "class {c} count {count} outside 850..1150"
            );
        }
    }

    // -------------------- DistributedSampler --------------------

    #[test]
    fn distributed_sampler_partitions_indices() {
        let mut s0 = DistributedSampler::new(8, 0, 4);
        let mut s1 = DistributedSampler::new(8, 1, 4);
        let mut s2 = DistributedSampler::new(8, 2, 4);
        let mut s3 = DistributedSampler::new(8, 3, 4);
        let i0 = s0.iter();
        let i1 = s1.iter();
        let i2 = s2.iter();
        let i3 = s3.iter();
        // Together they should cover all 0..8 exactly once.
        let mut all: Vec<usize> = i0
            .iter()
            .chain(&i1)
            .chain(&i2)
            .chain(&i3)
            .copied()
            .collect();
        all.sort();
        assert_eq!(all, (0..8).collect::<Vec<_>>());
    }

    #[test]
    fn distributed_sampler_reshuffles_per_epoch() {
        let mut s = DistributedSampler::new(20, 0, 2);
        let e0 = s.iter();
        s.set_epoch(1);
        let e1 = s.iter();
        // High probability the orderings differ.
        assert_ne!(e0, e1);
    }

    #[test]
    fn distributed_sampler_len_matches_partition() {
        let s = DistributedSampler::new(10, 0, 4);
        // 10 / 4 = 2 base; rank 0 < 10 % 4 = 2 → +1 → 3 items.
        assert_eq!(s.len(), 3);
        let s2 = DistributedSampler::new(10, 3, 4);
        // rank 3 not < 2 → 2 items.
        assert_eq!(s2.len(), 2);
    }
}
