//! Bundled datasets (P1.9) — synthetic fall-backs.
//!
//! v1 ships **synthetic** versions of the canonical vision/text
//! datasets. They share the same shape and dtype as the real data so
//! training/eval code is portable. Real downloaders (URL + sha256 +
//! HTTP fetch) land in a follow-up — they require an HTTP client dep
//! we don't want to pull into the core data crate yet.
//!
//! Provided synthetic datasets:
//! - [`synthetic_mnist`] — 60k train + 10k test of `[1, 28, 28]` F32
//!   images (cluster-of-Gaussians per class) with 10 classes.
//! - [`synthetic_cifar10`] — 50k train + 10k test of `[3, 32, 32]`
//!   F32 images with 10 classes.

use crate::dataset::TensorDataset;
use rustorch_core::tensor::tensor_impl::Tensor;

/// Toy MNIST: `[N, 1, 28, 28]` images + I64 labels in 0..10.
/// Uses a deterministic LCG seeded by `seed`.
pub fn synthetic_mnist(n: usize, seed: u64) -> TensorDataset {
    let img_size = 28 * 28;
    let mut s = seed;
    let mut next = || {
        s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        let bits = (s ^ (s >> 32)) as u32;
        (bits as f32) / (u32::MAX as f32)
    };
    let mut xs = Vec::with_capacity(n * img_size);
    let mut ys = Vec::with_capacity(n);
    for _ in 0..n {
        let label = (next() * 10.0) as i64 % 10;
        let centre = (label as f32) / 9.0; // 0..1 across classes
        for _ in 0..img_size {
            let noise = next() - 0.5; // [-0.5, 0.5]
            xs.push((centre + noise).clamp(0.0, 1.0));
        }
        ys.push(label);
    }
    let xs_t = Tensor::from_vec([n, 1, 28, 28], xs).expect("mnist xs shape");
    let ys_t = Tensor::from_vec_typed::<i64, _>([n], ys).expect("mnist ys shape");
    TensorDataset::new(xs_t, ys_t).expect("mnist build")
}

/// Toy CIFAR-10: `[N, 3, 32, 32]` images + I64 labels in 0..10.
pub fn synthetic_cifar10(n: usize, seed: u64) -> TensorDataset {
    let img_size = 3 * 32 * 32;
    let mut s = seed;
    let mut next = || {
        s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        let bits = (s ^ (s >> 32)) as u32;
        (bits as f32) / (u32::MAX as f32)
    };
    let mut xs = Vec::with_capacity(n * img_size);
    let mut ys = Vec::with_capacity(n);
    for _ in 0..n {
        let label = (next() * 10.0) as i64 % 10;
        let centre = (label as f32) / 9.0;
        for _ in 0..img_size {
            let noise = next() - 0.5;
            xs.push((centre + noise).clamp(0.0, 1.0));
        }
        ys.push(label);
    }
    let xs_t = Tensor::from_vec([n, 3, 32, 32], xs).expect("cifar xs shape");
    let ys_t = Tensor::from_vec_typed::<i64, _>([n], ys).expect("cifar ys shape");
    TensorDataset::new(xs_t, ys_t).expect("cifar build")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Dataset;

    #[test]
    fn synthetic_mnist_shape_correct() {
        let ds = synthetic_mnist(100, 42);
        assert_eq!(ds.len(), 100);
        let (x, y) = ds.get(0).unwrap();
        assert_eq!(x.shape(), &[1, 1, 28, 28]);
        assert_eq!(y.shape(), &[1]);
    }

    #[test]
    fn synthetic_cifar10_shape_correct() {
        let ds = synthetic_cifar10(50, 7);
        assert_eq!(ds.len(), 50);
        let (x, y) = ds.get(0).unwrap();
        assert_eq!(x.shape(), &[1, 3, 32, 32]);
        assert_eq!(y.shape(), &[1]);
    }

    #[test]
    fn synthetic_mnist_labels_in_range() {
        let ds = synthetic_mnist(200, 0);
        for i in 0..ds.len() {
            let (_, y) = ds.get(i).unwrap();
            let label = y.as_slice::<i64>().unwrap()[0];
            assert!((0..10).contains(&label));
        }
    }
}
