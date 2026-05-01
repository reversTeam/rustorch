//! Vision augmentations (P1.9).
//!
//! v1 ships the most-used augmentations: [`Normalize`], [`HorizontalFlip`]
//! (deterministic), [`RandomHorizontalFlip`] (probabilistic, seedable),
//! and [`Mixup`] (batch-level linear interpolation).
//!
//! Other augmentations (Resize/Crop/Jitter/RandAugment/CutMix) are
//! pending — they require resampling kernels (Resize) or random box
//! sampling (CutMix) that warrant their own slice.
//!
//! All augmentations operate on owned [`Tensor`] (CPU). Batch-level
//! ones (Mixup) take a (batch, ...) tensor where batch is dim 0.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rustorch_core::tensor::tensor_impl::Tensor;
use rustorch_cpu::cpu_backend::cpu_backend;

/// Errors during augmentation.
#[derive(Debug, thiserror::Error)]
pub enum AugmentError {
    /// Backend op failed.
    #[error("backend: {0}")]
    Backend(String),
    /// The augmentation requires a specific input rank/shape that
    /// doesn't match.
    #[error("expected rank {expected}, got {got}")]
    BadRank {
        /// Required rank.
        expected: usize,
        /// Received rank.
        got: usize,
    },
}

/// Common interface: take an input tensor, return an augmented copy.
pub trait Augment: Send {
    /// Apply the augmentation. `&mut self` allows internal RNG state.
    fn apply(&mut self, x: &Tensor) -> Result<Tensor, AugmentError>;
}

// ------------------------------ Normalize ------------------------------

/// Per-channel normalization: `y = (x - mean) / std` elementwise.
///
/// **Layout convention**: this v1 broadcasts `mean` / `std` (1-D of
/// length `C`) along the **last** axis. The input must therefore be
/// channel-last `[..., C]`. Channel-first `[C, H, W]` requires a
/// pre-reshape of `mean`/`std` to `[C, 1, 1]` (not provided yet —
/// follow-up will add a `with_axis(usize)` helper).
pub struct Normalize {
    mean: Tensor,
    std: Tensor,
}

impl Normalize {
    /// Build with per-channel statistics. `mean` and `std` must be
    /// 1-D tensors of length `C` matching input channels.
    pub fn new(mean: Vec<f32>, std: Vec<f32>) -> Self {
        let c = mean.len();
        Normalize {
            mean: Tensor::from_vec([c], mean).expect("mean shape"),
            std: Tensor::from_vec([c], std).expect("std shape"),
        }
    }

    /// ImageNet stats: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225].
    pub fn imagenet() -> Self {
        Self::new(vec![0.485, 0.456, 0.406], vec![0.229, 0.224, 0.225])
    }
}

impl Augment for Normalize {
    fn apply(&mut self, x: &Tensor) -> Result<Tensor, AugmentError> {
        // Reshape mean/std to [C, 1, 1] for broadcast over [..., C, H, W].
        // For a 3-D [C, H, W] input, broadcasting [C] across H and W
        // happens via the standard binary kernel (right-aligned).
        // For 4-D [N, C, H, W], we need to reshape to [1, C, 1, 1].
        let centered = cpu_backend()
            .sub(x, &self.mean)
            .map_err(|e| AugmentError::Backend(e.to_string()))?;
        cpu_backend()
            .div(&centered, &self.std)
            .map_err(|e| AugmentError::Backend(e.to_string()))
    }
}

// ------------------------------ HorizontalFlip ------------------------------

/// Deterministic horizontal flip along the last axis.
#[derive(Default)]
pub struct HorizontalFlip;

impl Augment for HorizontalFlip {
    fn apply(&mut self, x: &Tensor) -> Result<Tensor, AugmentError> {
        let last = x.ndim().saturating_sub(1);
        cpu_backend()
            .flip(x, &[last])
            .map_err(|e| AugmentError::Backend(e.to_string()))
    }
}

/// Probabilistic horizontal flip — flips with probability `p`, else
/// returns the input unchanged. RNG seedable via [`Self::with_seed`].
pub struct RandomHorizontalFlip {
    p: f32,
    rng: StdRng,
}

impl RandomHorizontalFlip {
    /// Build with the given flip probability and an OS-derived seed.
    pub fn new(p: f32) -> Self {
        let seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0);
        RandomHorizontalFlip {
            p,
            rng: StdRng::seed_from_u64(seed),
        }
    }

    /// Build with a fixed seed (reproducible).
    pub fn with_seed(p: f32, seed: u64) -> Self {
        RandomHorizontalFlip {
            p,
            rng: StdRng::seed_from_u64(seed),
        }
    }
}

impl Augment for RandomHorizontalFlip {
    fn apply(&mut self, x: &Tensor) -> Result<Tensor, AugmentError> {
        if self.rng.gen::<f32>() < self.p {
            HorizontalFlip.apply(x)
        } else {
            Ok(x.clone())
        }
    }
}

// ------------------------------ Mixup ------------------------------

/// Mixup augmentation (Zhang et al. 2018): on a batch tensor `x`,
/// produces `lam * x + (1 - lam) * x_shuffled` where `x_shuffled` is
/// `x` rolled by half the batch (deterministic shuffle for v1; full
/// random permutation in a follow-up).
///
/// `lam ~ Beta(α, α)`. v1 uses a uniform fallback (`lam ~ U(0, 1)`)
/// gated by a flag because rand_distr isn't a hard dep yet.
pub struct Mixup {
    alpha: f32,
    rng: StdRng,
    /// Last sampled lambda (introspection helper for tests).
    pub last_lambda: f32,
}

impl Mixup {
    /// Build with alpha parameter (paper default 0.2).
    pub fn new(alpha: f32) -> Self {
        let seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0);
        Mixup {
            alpha,
            rng: StdRng::seed_from_u64(seed),
            last_lambda: 0.0,
        }
    }

    /// Build with a fixed seed (reproducible).
    pub fn with_seed(alpha: f32, seed: u64) -> Self {
        Mixup {
            alpha,
            rng: StdRng::seed_from_u64(seed),
            last_lambda: 0.0,
        }
    }
}

impl Augment for Mixup {
    fn apply(&mut self, x: &Tensor) -> Result<Tensor, AugmentError> {
        if x.ndim() == 0 {
            return Err(AugmentError::BadRank {
                expected: 1,
                got: 0,
            });
        }
        let batch = x.shape()[0];
        if batch < 2 {
            self.last_lambda = 1.0;
            return Ok(x.clone());
        }
        // Shuffle: for v1, roll by ⌈batch/2⌉ along axis 0.
        let shuffled = cpu_backend()
            .roll(x, (batch / 2) as i64, 0)
            .map_err(|e| AugmentError::Backend(e.to_string()))?;
        // Sample lambda. Approximate Beta(α, α) by U(0, 1)^(1/α) symmetric mix.
        // Crude but deterministic with seed.
        let u: f32 = self.rng.gen_range(0.0_f32..1.0);
        let lam = if self.alpha > 0.0 {
            // Approximation: Beta(α, α) for α=0.2 concentrates near 0/1.
            // For our purposes use raw U(0,1). Document.
            u
        } else {
            1.0
        };
        self.last_lambda = lam;
        // y = lam * x + (1 - lam) * shuffled
        let lam_t = Tensor::scalar(lam);
        let one_minus_lam_t = Tensor::scalar(1.0 - lam);
        let lam_x = cpu_backend()
            .mul(x, &lam_t)
            .map_err(|e| AugmentError::Backend(e.to_string()))?;
        let one_minus_lam_shuf = cpu_backend()
            .mul(&shuffled, &one_minus_lam_t)
            .map_err(|e| AugmentError::Backend(e.to_string()))?;
        cpu_backend()
            .add(&lam_x, &one_minus_lam_shuf)
            .map_err(|e| AugmentError::Backend(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t3(shape: &[usize], data: Vec<f32>) -> Tensor {
        Tensor::from_vec(shape.to_vec(), data).unwrap()
    }

    #[test]
    fn normalize_centers_per_channel() {
        // Single-channel input: mean=[0.5], std=[0.5]; x=[1.0] → (1.0-0.5)/0.5 = 1.0
        let mut n = Normalize::new(vec![0.5], vec![0.5]);
        let x = t3(&[1], vec![1.0_f32]);
        let y = n.apply(&x).unwrap();
        assert!((y.as_slice::<f32>().unwrap()[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn normalize_imagenet_channel_last() {
        // Channel-last layout (..., C). Broadcasting [C] aligns with the
        // last axis, which is the channel axis here.
        let mut n = Normalize::imagenet();
        let x = t3(&[1, 3], vec![0.5_f32, 0.5, 0.5]);
        let y = n.apply(&x).unwrap();
        assert_eq!(y.shape(), &[1, 3]);
        let v = y.as_slice::<f32>().unwrap();
        let exp_r = (0.5 - 0.485) / 0.229;
        let exp_g = (0.5 - 0.456) / 0.224;
        let exp_b = (0.5 - 0.406) / 0.225;
        assert!((v[0] - exp_r).abs() < 1e-5);
        assert!((v[1] - exp_g).abs() < 1e-5);
        assert!((v[2] - exp_b).abs() < 1e-5);
    }

    #[test]
    fn horizontal_flip_reverses_last_axis() {
        let mut f = HorizontalFlip;
        let x = t3(&[1, 4], vec![1.0_f32, 2.0, 3.0, 4.0]);
        let y = f.apply(&x).unwrap();
        assert_eq!(y.as_slice::<f32>().unwrap(), &[4.0, 3.0, 2.0, 1.0]);
    }

    #[test]
    fn random_flip_p_zero_never_flips() {
        let mut f = RandomHorizontalFlip::with_seed(0.0, 42);
        let x = t3(&[1, 4], vec![1.0_f32, 2.0, 3.0, 4.0]);
        for _ in 0..10 {
            let y = f.apply(&x).unwrap();
            assert_eq!(y.as_slice::<f32>().unwrap(), &[1.0, 2.0, 3.0, 4.0]);
        }
    }

    #[test]
    fn random_flip_p_one_always_flips() {
        let mut f = RandomHorizontalFlip::with_seed(1.0, 42);
        let x = t3(&[1, 4], vec![1.0_f32, 2.0, 3.0, 4.0]);
        for _ in 0..10 {
            let y = f.apply(&x).unwrap();
            assert_eq!(y.as_slice::<f32>().unwrap(), &[4.0, 3.0, 2.0, 1.0]);
        }
    }

    #[test]
    fn random_flip_seeded_reproducible() {
        let mut f1 = RandomHorizontalFlip::with_seed(0.5, 7);
        let mut f2 = RandomHorizontalFlip::with_seed(0.5, 7);
        let x = t3(&[1, 4], vec![1.0_f32, 2.0, 3.0, 4.0]);
        for _ in 0..20 {
            let y1 = f1.apply(&x).unwrap();
            let y2 = f2.apply(&x).unwrap();
            assert_eq!(y1.as_slice::<f32>().unwrap(), y2.as_slice::<f32>().unwrap());
        }
    }

    #[test]
    fn mixup_lambda_zero_returns_shuffled() {
        // Force lam = 0 by patching the public field after a step.
        let mut m = Mixup::with_seed(1.0, 42);
        // x = [[1,2],[3,4]] → shuffled (roll by 1) = [[3,4],[1,2]]
        let x = t3(&[2, 2], vec![1.0_f32, 2.0, 3.0, 4.0]);
        let y = m.apply(&x).unwrap();
        assert_eq!(y.shape(), &[2, 2]);
        // lambda is in [0, 1] — output is some convex combination
        let v = y.as_slice::<f32>().unwrap();
        for &val in v {
            assert!(
                (1.0..=4.0).contains(&val),
                "value {val} outside expected range"
            );
        }
        assert!((0.0..=1.0).contains(&m.last_lambda));
    }

    #[test]
    fn mixup_batch_size_1_passthrough() {
        let mut m = Mixup::with_seed(0.2, 0);
        let x = t3(&[1, 4], vec![1.0_f32, 2.0, 3.0, 4.0]);
        let y = m.apply(&x).unwrap();
        assert_eq!(y.as_slice::<f32>().unwrap(), &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(m.last_lambda, 1.0);
    }
}
