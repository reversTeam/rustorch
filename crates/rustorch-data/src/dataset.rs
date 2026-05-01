//! `Dataset` and `IterableDataset` traits + common impls (P1.8).
//!
//! Two paradigms aligned with PyTorch:
//! - Map-style ([`Dataset`]): random access via `len()` + `get(i)`.
//!   The default for in-memory or indexed-on-disk datasets.
//! - Stream-style ([`IterableDataset`]): sequential `next()` only.
//!   The default for true streams (network, generators).
//!
//! Concrete impls in v1: [`TensorDataset`] (in-memory zip of tensors).

use rustorch_core::tensor::tensor_impl::Tensor;

/// Errors when constructing or querying a [`Dataset`].
#[derive(Debug, thiserror::Error)]
pub enum DatasetError {
    /// A dataset was built with rows of inconsistent shape.
    #[error("shape mismatch in dataset: row {row} has shape {got:?}, expected {expected:?}")]
    ShapeMismatch {
        /// Row index that failed.
        row: usize,
        /// Actual shape.
        got: Vec<usize>,
        /// First-row shape (taken as canonical).
        expected: Vec<usize>,
    },
    /// `get(i)` was called with `i >= len()`.
    #[error("index {idx} out of bounds for dataset of len {len}")]
    IndexOutOfBounds {
        /// Index requested.
        idx: usize,
        /// Dataset length.
        len: usize,
    },
}

/// Map-style dataset: random access by index. Items are typed via the
/// associated `Item` so concrete impls can return tuples / structs.
pub trait Dataset {
    /// Item produced by `get`.
    type Item;
    /// Number of items.
    fn len(&self) -> usize;
    /// True iff `len() == 0`.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Fetch the i-th item.
    fn get(&self, idx: usize) -> Result<Self::Item, DatasetError>;
}

/// Streaming dataset: sequential `next()`. Equivalent to PyTorch's
/// `IterableDataset`.
pub trait IterableDataset {
    /// Item produced by `next`.
    type Item;
    /// Yield the next item, or `None` if the stream is exhausted.
    fn next(&mut self) -> Option<Self::Item>;
}

// ------------------------------ TensorDataset ------------------------------

/// In-memory zip of (input, target) tensors. Each row is one item;
/// `get(i)` clones the i-th slice along the first dimension.
///
/// Both inputs and targets must have the same first-dim length.
pub struct TensorDataset {
    inputs: Tensor,
    targets: Tensor,
    len: usize,
}

impl TensorDataset {
    /// Build from two tensors that share their leading dimension.
    pub fn new(inputs: Tensor, targets: Tensor) -> Result<Self, DatasetError> {
        let n_in = if inputs.shape().is_empty() {
            0
        } else {
            inputs.shape()[0]
        };
        let n_tgt = if targets.shape().is_empty() {
            0
        } else {
            targets.shape()[0]
        };
        if n_in != n_tgt {
            return Err(DatasetError::ShapeMismatch {
                row: 0,
                got: targets.shape().to_vec(),
                expected: inputs.shape().to_vec(),
            });
        }
        Ok(TensorDataset {
            inputs,
            targets,
            len: n_in,
        })
    }

    /// Borrow the underlying input tensor.
    pub fn inputs(&self) -> &Tensor {
        &self.inputs
    }

    /// Borrow the underlying target tensor.
    pub fn targets(&self) -> &Tensor {
        &self.targets
    }
}

impl Dataset for TensorDataset {
    type Item = (Tensor, Tensor);

    fn len(&self) -> usize {
        self.len
    }

    fn get(&self, idx: usize) -> Result<Self::Item, DatasetError> {
        if idx >= self.len {
            return Err(DatasetError::IndexOutOfBounds { idx, len: self.len });
        }
        // Slice along axis 0: take row `idx` of inputs and targets.
        // Use index_select with a 1-element [idx] tensor, then squeeze
        // dim 0 by reshaping to the per-row shape.
        let idx_t = Tensor::from_vec_typed::<i64, _>([1usize], vec![idx as i64])
            .expect("scalar index tensor");

        let in_row = rustorch_cpu::cpu_backend::cpu_backend()
            .index_select(&self.inputs, 0, &idx_t)
            .expect("index_select inputs");
        let tgt_row = rustorch_cpu::cpu_backend::cpu_backend()
            .index_select(&self.targets, 0, &idx_t)
            .expect("index_select targets");
        Ok((in_row, tgt_row))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn xs() -> Tensor {
        Tensor::from_vec(
            [4usize, 3],
            vec![
                1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
        )
        .unwrap()
    }

    fn ys() -> Tensor {
        Tensor::from_vec_typed::<i64, _>([4usize], vec![0_i64, 1, 0, 1]).unwrap()
    }

    #[test]
    fn tensor_dataset_len_matches_first_dim() {
        let ds = TensorDataset::new(xs(), ys()).unwrap();
        assert_eq!(ds.len(), 4);
        assert!(!ds.is_empty());
    }

    #[test]
    fn tensor_dataset_get_first_row() {
        let ds = TensorDataset::new(xs(), ys()).unwrap();
        let (x, y) = ds.get(0).unwrap();
        assert_eq!(x.shape(), &[1, 3]);
        assert_eq!(x.as_slice::<f32>().unwrap(), &[1.0, 2.0, 3.0]);
        assert_eq!(y.as_slice::<i64>().unwrap(), &[0]);
    }

    #[test]
    fn tensor_dataset_get_last_row() {
        let ds = TensorDataset::new(xs(), ys()).unwrap();
        let (x, y) = ds.get(3).unwrap();
        assert_eq!(x.as_slice::<f32>().unwrap(), &[10.0, 11.0, 12.0]);
        assert_eq!(y.as_slice::<i64>().unwrap(), &[1]);
    }

    #[test]
    fn tensor_dataset_oob_returns_err() {
        let ds = TensorDataset::new(xs(), ys()).unwrap();
        assert!(matches!(
            ds.get(4),
            Err(DatasetError::IndexOutOfBounds { idx: 4, len: 4 })
        ));
    }

    #[test]
    fn tensor_dataset_mismatched_lengths_err() {
        let xs = Tensor::from_vec([4usize, 3], vec![0.0_f32; 12]).unwrap();
        let ys = Tensor::from_vec_typed::<i64, _>([3usize], vec![0_i64, 1, 0]).unwrap();
        assert!(TensorDataset::new(xs, ys).is_err());
    }

    #[test]
    fn empty_dataset_has_len_zero() {
        let xs = Tensor::from_vec([0usize, 3], vec![0.0_f32; 0]).unwrap();
        let ys = Tensor::from_vec_typed::<i64, _>([0usize], Vec::<i64>::new()).unwrap();
        let ds = TensorDataset::new(xs, ys).unwrap();
        assert_eq!(ds.len(), 0);
        assert!(ds.is_empty());
    }
}
