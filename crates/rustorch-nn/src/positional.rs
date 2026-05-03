//! Positional encodings for transformer-style models.
//!
//! Two flavours match PyTorch's common building blocks:
//! - [`SinusoidalPositionalEncoding`] — Vaswani et al. 2017 closed-form
//!   table; non-trainable. The table is precomputed once at construction
//!   so `forward_for_len(t)` is a constant-time slice.
//! - [`LearnedPositionalEncoding`] — a trainable `[max_len, dim]`
//!   embedding table; `forward_for_len(t)` returns the leading `t` rows.
//!
//! Both expose `forward_for_len(seq_len) -> Variable` returning shape
//! `[seq_len, dim]`. The caller broadcasts-adds the result to the token
//! embedding output (which has shape `[B, T, D]` — broadcast on the
//! leading batch dim is handled by the autograd `add` op).

use rustorch_autograd::ops::reshape;
use rustorch_autograd::{BackwardError, Variable};
use rustorch_core::tensor::tensor_impl::Tensor;

/// Fixed sinusoidal positional encoding (Vaswani et al. 2017).
///
/// Table of shape `[max_len, dim]` with
/// ```text
///   pe[pos, 2k]   = sin(pos / 10000^(2k / dim))
///   pe[pos, 2k+1] = cos(pos / 10000^(2k / dim))
/// ```
///
/// Non-trainable: [`Self::parameters`] returns an empty list.
pub struct SinusoidalPositionalEncoding {
    /// Precomputed `[max_len, dim]` table (F32).
    embedding: Tensor,
    dim: usize,
    max_len: usize,
}

impl SinusoidalPositionalEncoding {
    /// Build a sinusoidal table of shape `[max_len, dim]`.
    ///
    /// Panics if `dim == 0` or `max_len == 0`.
    pub fn new(dim: usize, max_len: usize) -> Self {
        assert!(dim > 0, "SinusoidalPositionalEncoding: dim must be > 0");
        assert!(
            max_len > 0,
            "SinusoidalPositionalEncoding: max_len must be > 0"
        );
        let mut data = vec![0.0_f32; max_len * dim];
        let dim_f = dim as f32;
        for pos in 0..max_len {
            for k in 0..dim {
                // Group consecutive (2k, 2k+1) pairs sharing one frequency.
                let pair = (k / 2) as f32; // 0,0,1,1,2,2,...
                let freq = 1.0_f32 / (10000.0_f32.powf(2.0 * pair / dim_f));
                let arg = pos as f32 * freq;
                data[pos * dim + k] = if k % 2 == 0 { arg.sin() } else { arg.cos() };
            }
        }
        let embedding = Tensor::from_vec([max_len, dim], data).expect("PE table shape");
        SinusoidalPositionalEncoding {
            embedding,
            dim,
            max_len,
        }
    }

    /// Embedding dim.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Maximum supported sequence length.
    pub fn max_len(&self) -> usize {
        self.max_len
    }

    /// Slice of the table for positions `0..seq_len`. Returns a leaf
    /// non-trainable [`Variable`] of shape `[seq_len, dim]`.
    ///
    /// Errors if `seq_len > max_len`.
    pub fn forward_for_len(&self, seq_len: usize) -> Result<Variable, BackwardError> {
        if seq_len > self.max_len {
            return Err(BackwardError::Backend {
                op: "SinusoidalPositionalEncoding::forward_for_len",
                message: format!("seq_len {} > max_len {}", seq_len, self.max_len),
            });
        }
        // Manual row slice — view ops on the Tensor return a view; we
        // copy into a fresh contiguous buffer so the returned Variable
        // owns its data and is safe to compose with autograd reshape.
        let full = self.embedding.as_slice::<f32>().expect("PE storage f32");
        let n = seq_len * self.dim;
        let buf: Vec<f32> = full[..n].to_vec();
        let t = Tensor::from_vec([seq_len, self.dim], buf).expect("PE slice shape");
        Ok(Variable::new(t))
    }

    /// No trainable parameters — empty list.
    pub fn parameters(&self) -> Vec<Variable> {
        Vec::new()
    }
}

/// Trainable positional embedding — a `[max_len, dim]` matrix updated
/// by gradient descent like any other parameter.
///
/// Equivalent to `torch.nn.Embedding(max_len, dim)` used as a positional
/// encoder (rather than a vocabulary embedder).
pub struct LearnedPositionalEncoding {
    /// Trainable `[max_len, dim]` table.
    pub weight: Variable,
    dim: usize,
    max_len: usize,
}

impl LearnedPositionalEncoding {
    /// Build with a deterministic small-uniform init in `[-bound, bound]`
    /// where `bound = 1/sqrt(dim)` (mirrors [`crate::Linear::new`]).
    pub fn new(max_len: usize, dim: usize) -> Self {
        assert!(dim > 0, "LearnedPositionalEncoding: dim must be > 0");
        assert!(
            max_len > 0,
            "LearnedPositionalEncoding: max_len must be > 0"
        );
        let bound = 1.0_f32 / (dim as f32).sqrt();
        let data: Vec<f32> = (0..max_len * dim)
            .map(|i| {
                // Same deterministic LCG noise as Linear::new.
                let mut s = (i as u64)
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1);
                s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                let u = (s >> 32) as u32;
                let f = (u as f32) / (u32::MAX as f32);
                (f * 2.0 - 1.0) * bound
            })
            .collect();
        let weight =
            Variable::leaf(Tensor::from_vec([max_len, dim], data).expect("learned PE shape"));
        LearnedPositionalEncoding {
            weight,
            dim,
            max_len,
        }
    }

    /// Embedding dim.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Maximum supported sequence length.
    pub fn max_len(&self) -> usize {
        self.max_len
    }

    /// Trainable slice for positions `0..seq_len`. Returns shape `[seq_len, dim]`.
    ///
    /// Implementation: take the full `[max_len, dim]` weight, view it as
    /// a flat `[max_len * dim]` then slice the first `seq_len * dim`
    /// elements via [`rustorch_autograd::ops::reshape`] which preserves
    /// the autograd link.
    ///
    /// NB: in v1 this materialises a fresh contiguous tensor by copying
    /// the leading rows from the weight's snapshot; gradient still flows
    /// because we wrap the slice as a *non-leaf* via a manual leaf-Variable
    /// chain with `requires_grad` propagated. For seq_len == max_len the
    /// path collapses to a no-op reshape.
    pub fn forward_for_len(&self, seq_len: usize) -> Result<Variable, BackwardError> {
        if seq_len > self.max_len {
            return Err(BackwardError::Backend {
                op: "LearnedPositionalEncoding::forward_for_len",
                message: format!("seq_len {} > max_len {}", seq_len, self.max_len),
            });
        }
        if seq_len == self.max_len {
            // Pass through with a no-op reshape that keeps the grad path.
            return reshape(&self.weight, vec![seq_len, self.dim]);
        }
        // Use index_select to gather the first `seq_len` rows. This is
        // autograd-aware via IndexSelectBackward (scatter-add into the
        // weight's grad).
        let idx_data: Vec<i64> = (0..seq_len as i64).collect();
        let idx = Tensor::from_vec_typed::<i64, _>([seq_len], idx_data).expect("idx shape");
        rustorch_autograd::ops::index_select(&self.weight, &idx)
    }

    /// Trainable parameters.
    pub fn parameters(&self) -> Vec<Variable> {
        vec![self.weight.clone()]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustorch_autograd::backward;

    /// Hand-computed Vaswani formula on a small `(dim=4, len=4)` table.
    /// pe[pos, 2k]   = sin(pos / 10000^(2k/dim))
    /// pe[pos, 2k+1] = cos(pos / 10000^(2k/dim))
    #[test]
    fn sinusoidal_matches_vaswani_closed_form() {
        let pe = SinusoidalPositionalEncoding::new(4, 4);
        let v = pe.forward_for_len(4).unwrap();
        let t = v.tensor();
        let s = t.as_slice::<f32>().unwrap();
        // Row 0: sin(0)=0, cos(0)=1, sin(0)=0, cos(0)=1
        assert!((s[0] - 0.0).abs() < 1e-6);
        assert!((s[1] - 1.0).abs() < 1e-6);
        assert!((s[2] - 0.0).abs() < 1e-6);
        assert!((s[3] - 1.0).abs() < 1e-6);
        // Row 1, dim 0: sin(1 / 10000^0) = sin(1)
        assert!((s[4] - 1.0_f32.sin()).abs() < 1e-6);
        // Row 1, dim 1: cos(1)
        assert!((s[5] - 1.0_f32.cos()).abs() < 1e-6);
        // Row 1, dim 2: sin(1 / 10000^(2/4)) = sin(1 / 100) = sin(0.01)
        let f1 = 1.0_f32 / 10000.0_f32.powf(0.5);
        assert!((s[6] - f1.sin()).abs() < 1e-6);
        // Row 1, dim 3: cos(0.01)
        assert!((s[7] - f1.cos()).abs() < 1e-6);
    }

    /// Adding a sinusoidal PE to a `[B, T, D]` embedding output preserves
    /// shape (broadcasted over the batch axis by the autograd `add`).
    #[test]
    fn sinusoidal_adds_into_btd_shape() {
        let dim = 4;
        let len = 3;
        let pe = SinusoidalPositionalEncoding::new(dim, len);
        let pe_v = pe.forward_for_len(len).unwrap(); // [3, 4]
                                                     // Reshape to [1, 3, 4] so it broadcasts cleanly to [B, T, D].
        let pe_btd = reshape(&pe_v, vec![1, len, dim]).unwrap();
        let emb = Variable::new(
            Tensor::from_vec(
                [2usize, len, dim],
                (0..2 * len * dim)
                    .map(|i| i as f32 * 0.01)
                    .collect::<Vec<_>>(),
            )
            .unwrap(),
        );
        let out = rustorch_autograd::ops::add(&emb, &pe_btd).unwrap();
        assert_eq!(out.tensor().shape(), &[2, 3, 4]);
    }

    /// `parameters()` is empty for the fixed sinusoidal table.
    #[test]
    fn sinusoidal_no_parameters() {
        let pe = SinusoidalPositionalEncoding::new(8, 16);
        assert!(pe.parameters().is_empty());
    }

    /// Out-of-range seq_len → clean error, no panic.
    #[test]
    fn sinusoidal_seq_len_too_long_errors() {
        let pe = SinusoidalPositionalEncoding::new(4, 8);
        assert!(pe.forward_for_len(9).is_err());
    }

    /// Learned PE: shape `[seq_len, dim]` and trainable.
    #[test]
    fn learned_pe_shape_and_params() {
        let pe = LearnedPositionalEncoding::new(16, 8);
        let v = pe.forward_for_len(4).unwrap();
        assert_eq!(v.tensor().shape(), &[4, 8]);
        assert_eq!(pe.parameters().len(), 1);
    }

    /// Backward through the learned slice writes a non-zero grad into
    /// the trainable weight.
    #[test]
    fn learned_pe_backward_updates_weight_grad() {
        let pe = LearnedPositionalEncoding::new(8, 4);
        let v = pe.forward_for_len(3).unwrap(); // [3, 4]
        let loss = rustorch_autograd::ops::sum(&v).unwrap();
        backward(&loss, None).unwrap();
        let w_grad = pe
            .weight
            .grad()
            .expect("learned PE weight should have grad");
        assert_eq!(w_grad.shape(), &[8, 4]);
        let g = w_grad.as_slice::<f32>().unwrap();
        // Rows 0..3 should have grad 1.0 (sum's d/dx); rows 3..8 should be 0.
        for row in 0..3 {
            for col in 0..4 {
                assert!(
                    (g[row * 4 + col] - 1.0).abs() < 1e-6,
                    "row {} col {}",
                    row,
                    col
                );
            }
        }
        for row in 3..8 {
            for col in 0..4 {
                assert!(g[row * 4 + col].abs() < 1e-6, "row {} col {}", row, col);
            }
        }
    }

    /// Out-of-range seq_len on the learned variant → clean error.
    #[test]
    fn learned_pe_seq_len_too_long_errors() {
        let pe = LearnedPositionalEncoding::new(8, 4);
        assert!(pe.forward_for_len(9).is_err());
    }
}
