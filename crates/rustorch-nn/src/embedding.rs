//! Embedding lookup module (P1.6).
//!
//! Standard `nn.Embedding`: learnable lookup table of shape
//! `[vocab_size, embed_dim]`. Forward takes a 1-D I64 index tensor and
//! returns the corresponding rows.
//!
//! Backward via autograd-aware [`rustorch_autograd::ops::index_select`]
//! — the embedding rows that are looked up receive grad accumulation;
//! unused rows stay at zero.

use crate::init::{init_with_seed, Init};
use crate::module::{Module, ModuleError};
use rustorch_autograd::{ops, Variable};
use rustorch_core::tensor::tensor_impl::Tensor;

/// `nn.Embedding(vocab_size, embed_dim)`.
pub struct Embedding {
    /// Trainable lookup table of shape `[vocab_size, embed_dim]`.
    pub weight: Variable,
    vocab_size: usize,
    embed_dim: usize,
}

impl Embedding {
    /// Build with the given vocab + embedding sizes. Weight is sampled
    /// from `N(0, 1)` by default (matches `torch.nn.Embedding`).
    pub fn new(vocab_size: usize, embed_dim: usize) -> Self {
        Self::with_seed(vocab_size, embed_dim, 0)
    }

    /// Build with a fixed seed for reproducible init.
    pub fn with_seed(vocab_size: usize, embed_dim: usize, seed: u64) -> Self {
        let weight_t = init_with_seed(
            Init::Normal {
                mean: 0.0,
                std: 1.0,
            },
            &[vocab_size, embed_dim],
            seed,
        );
        Embedding {
            weight: Variable::leaf(weight_t),
            vocab_size,
            embed_dim,
        }
    }

    /// Vocabulary size (rows of the weight matrix).
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Embedding dimension (cols of the weight matrix).
    pub fn embed_dim(&self) -> usize {
        self.embed_dim
    }

    /// Forward with a 1-D I64 index tensor — returns `[len(indices), embed_dim]`.
    pub fn forward_indices(&self, indices: &Tensor) -> Result<Variable, ModuleError> {
        ops::index_select(&self.weight, indices)
    }
}

impl Module for Embedding {
    /// `forward(indices)` — input is a Variable holding a 1-D I64 index
    /// tensor (autograd doesn't track grad through indices since they
    /// aren't continuous).
    fn forward(&self, input: &Variable) -> Result<Variable, ModuleError> {
        ops::index_select(&self.weight, &input.tensor())
    }

    fn parameters(&self) -> Vec<Variable> {
        vec![self.weight.clone()]
    }

    fn named_parameters(&self) -> Vec<(String, Variable)> {
        vec![("weight".to_string(), self.weight.clone())]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustorch_autograd::backward;

    #[test]
    fn embedding_lookup_returns_correct_rows() {
        let mut e = Embedding::new(5, 3);
        // Override weight with known values for a deterministic test.
        let known = Tensor::from_vec(
            [5usize, 3],
            vec![
                0.0_f32, 0.1, 0.2, // row 0
                1.0, 1.1, 1.2, // row 1
                2.0, 2.1, 2.2, // row 2
                3.0, 3.1, 3.2, // row 3
                4.0, 4.1, 4.2, // row 4
            ],
        )
        .unwrap();
        e.weight = Variable::leaf(known);

        let idx = Tensor::from_vec_typed::<i64, _>([3usize], vec![1_i64, 3, 0]).unwrap();
        let out = e.forward_indices(&idx).unwrap();
        let out_t = out.tensor();
        let v = out_t.as_slice::<f32>().unwrap();
        assert_eq!(out_t.shape(), &[3, 3]);
        assert_eq!(v, &[1.0, 1.1, 1.2, 3.0, 3.1, 3.2, 0.0, 0.1, 0.2]);
    }

    #[test]
    fn embedding_backward_accumulates_in_used_rows() {
        let e = Embedding::with_seed(4, 2, 42);
        // Look up indices [0, 2, 0] — backward should accumulate at
        // rows 0 (twice) and 2 (once); rows 1 and 3 unchanged.
        let idx = Tensor::from_vec_typed::<i64, _>([3usize], vec![0_i64, 2, 0]).unwrap();
        let out = e.forward_indices(&idx).unwrap();
        let s = rustorch_autograd::ops::sum(&out).unwrap();
        backward(&s, None).unwrap();
        let g = e.weight.grad().unwrap();
        let g_v = g.as_slice::<f32>().unwrap();
        // Each lookup adds ones[2] to its row. Row 0 picked twice → grad row 0 = [2, 2].
        // Row 2 picked once → grad row 2 = [1, 1]. Rows 1 and 3 should be zeros.
        assert_eq!(&g_v[0..2], &[2.0_f32, 2.0]);
        assert_eq!(&g_v[2..4], &[0.0_f32, 0.0]);
        assert_eq!(&g_v[4..6], &[1.0_f32, 1.0]);
        assert_eq!(&g_v[6..8], &[0.0_f32, 0.0]);
    }

    #[test]
    fn embedding_module_forward_via_variable() {
        let e = Embedding::new(10, 4);
        let idx = Variable::new(
            Tensor::from_vec_typed::<i64, _>([5usize], vec![0_i64, 1, 2, 3, 4]).unwrap(),
        );
        let out = e.forward(&idx).unwrap();
        assert_eq!(out.tensor().shape(), &[5, 4]);
    }

    #[test]
    fn embedding_parameters_returns_weight() {
        let e = Embedding::new(8, 3);
        let p = e.parameters();
        assert_eq!(p.len(), 1);
        assert_eq!(p[0].tensor().shape(), &[8, 3]);
        let np = e.named_parameters();
        assert_eq!(np.len(), 1);
        assert_eq!(np[0].0, "weight");
    }

    #[test]
    fn embedding_init_is_normal() {
        let e = Embedding::with_seed(100, 50, 0);
        let w = e.weight.tensor();
        let v = w.as_slice::<f32>().unwrap();
        let mean = v.iter().sum::<f32>() / v.len() as f32;
        let var = v.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / v.len() as f32;
        assert!(mean.abs() < 0.1, "init mean ≈ 0, got {mean}");
        assert!(
            (var.sqrt() - 1.0).abs() < 0.1,
            "init std ≈ 1, got {}",
            var.sqrt()
        );
    }
}
