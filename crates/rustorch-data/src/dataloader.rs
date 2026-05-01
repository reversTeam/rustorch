//! `DataLoader` (P1.8) — single-threaded reference impl.
//!
//! Pulls indices from a [`Sampler`], fetches items via
//! [`Dataset::get`], and stacks them into batched tensors via
//! `cpu_backend::stack`.
//!
//! v1 is **single-threaded**. Workers (`num_workers`) using rayon will
//! be wired in a follow-up — the iterator API and batching semantics
//! are stable now.

use crate::dataset::{Dataset, DatasetError};
use crate::sampler::Sampler;
use rustorch_core::tensor::tensor_impl::Tensor;
use rustorch_cpu::cpu_backend::cpu_backend;

/// Errors during DataLoader iteration.
#[derive(Debug, thiserror::Error)]
pub enum DataLoaderError {
    /// A `Dataset::get` call failed.
    #[error("dataset error: {0}")]
    Dataset(#[from] DatasetError),
    /// A backend op (cat / stack) failed during batch collation.
    #[error("backend error: {0}")]
    Backend(String),
}

/// DataLoader for `(input, target)` map-style datasets.
pub struct DataLoader<D, S>
where
    D: Dataset<Item = (Tensor, Tensor)>,
    S: Sampler,
{
    dataset: D,
    sampler: S,
    batch_size: usize,
    drop_last: bool,
}

impl<D, S> DataLoader<D, S>
where
    D: Dataset<Item = (Tensor, Tensor)>,
    S: Sampler,
{
    /// Build a DataLoader. `drop_last` defaults to `false`.
    pub fn new(dataset: D, sampler: S, batch_size: usize) -> Self {
        DataLoader {
            dataset,
            sampler,
            batch_size,
            drop_last: false,
        }
    }

    /// Drop the last partial batch if it doesn't fill `batch_size`.
    #[must_use]
    pub fn drop_last(mut self, drop_last: bool) -> Self {
        self.drop_last = drop_last;
        self
    }

    /// Number of batches in one epoch.
    pub fn num_batches(&self) -> usize {
        let n = self.sampler.len();
        if self.drop_last {
            n / self.batch_size
        } else {
            n.div_ceil(self.batch_size)
        }
    }

    /// Borrow the dataset (read-only).
    pub fn dataset(&self) -> &D {
        &self.dataset
    }

    /// Iterate all batches in one epoch. Returns `Result` per batch.
    pub fn iter_epoch(&mut self) -> Vec<Result<(Tensor, Tensor), DataLoaderError>> {
        let indices = self.sampler.iter();
        let mut batches = Vec::with_capacity(indices.len().div_ceil(self.batch_size));

        for chunk in indices.chunks(self.batch_size) {
            if self.drop_last && chunk.len() < self.batch_size {
                break;
            }
            let mut x_rows = Vec::with_capacity(chunk.len());
            let mut y_rows = Vec::with_capacity(chunk.len());
            let mut chunk_err: Option<DataLoaderError> = None;
            for &i in chunk {
                match self.dataset.get(i) {
                    Ok((x, y)) => {
                        x_rows.push(x);
                        y_rows.push(y);
                    },
                    Err(e) => {
                        chunk_err = Some(e.into());
                        break;
                    },
                }
            }
            if let Some(e) = chunk_err {
                batches.push(Err(e));
                continue;
            }
            // Collate by concat along axis 0 (rows are already [1, ...]).
            let x_refs: Vec<&Tensor> = x_rows.iter().collect();
            let y_refs: Vec<&Tensor> = y_rows.iter().collect();
            let xb = cpu_backend()
                .cat(&x_refs, 0)
                .map_err(|e| DataLoaderError::Backend(e.to_string()));
            let yb = cpu_backend()
                .cat(&y_refs, 0)
                .map_err(|e| DataLoaderError::Backend(e.to_string()));
            match (xb, yb) {
                (Ok(x), Ok(y)) => batches.push(Ok((x, y))),
                (Err(e), _) | (_, Err(e)) => batches.push(Err(e)),
            }
        }
        batches
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::TensorDataset;
    use crate::sampler::{RandomSampler, SequentialSampler};

    fn dataset(n: usize) -> TensorDataset {
        let xs =
            Tensor::from_vec([n, 2], (0..n * 2).map(|i| i as f32).collect::<Vec<_>>()).unwrap();
        let ys = Tensor::from_vec_typed::<i64, _>([n], (0..n as i64).collect::<Vec<_>>()).unwrap();
        TensorDataset::new(xs, ys).unwrap()
    }

    #[test]
    fn dataloader_yields_ceil_n_over_batch_size() {
        let ds = dataset(10);
        let s = SequentialSampler::new(10);
        let mut dl = DataLoader::new(ds, s, 3);
        // ceil(10/3) = 4 batches
        assert_eq!(dl.num_batches(), 4);
        let batches = dl.iter_epoch();
        assert_eq!(batches.len(), 4);
        let (x, y) = batches[0].as_ref().unwrap();
        assert_eq!(x.shape(), &[3, 2]);
        assert_eq!(y.shape(), &[3]);
        // Last batch shape: 10 - 9 = 1
        let (x_last, y_last) = batches[3].as_ref().unwrap();
        assert_eq!(x_last.shape(), &[1, 2]);
        assert_eq!(y_last.shape(), &[1]);
    }

    #[test]
    fn dataloader_drop_last_skips_partial_batch() {
        let ds = dataset(10);
        let s = SequentialSampler::new(10);
        let mut dl = DataLoader::new(ds, s, 3).drop_last(true);
        // floor(10/3) = 3 batches
        assert_eq!(dl.num_batches(), 3);
        assert_eq!(dl.iter_epoch().len(), 3);
    }

    #[test]
    fn dataloader_sequential_preserves_order() {
        let ds = dataset(6);
        let s = SequentialSampler::new(6);
        let mut dl = DataLoader::new(ds, s, 2);
        let batches = dl.iter_epoch();
        // Batch 0 should be [[0,1],[2,3]] with ys [0,1]
        let (_, y0) = batches[0].as_ref().unwrap();
        assert_eq!(y0.as_slice::<i64>().unwrap(), &[0_i64, 1]);
        let (_, y1) = batches[1].as_ref().unwrap();
        assert_eq!(y1.as_slice::<i64>().unwrap(), &[2_i64, 3]);
        let (_, y2) = batches[2].as_ref().unwrap();
        assert_eq!(y2.as_slice::<i64>().unwrap(), &[4_i64, 5]);
    }

    #[test]
    fn dataloader_random_changes_order() {
        let ds = dataset(20);
        let s = RandomSampler::with_seed(20, 42);
        let mut dl = DataLoader::new(ds, s, 4);
        let batches = dl.iter_epoch();
        // Concat all ys in order; should NOT equal 0..20 (very high prob).
        let mut all_y = Vec::new();
        for b in batches.iter() {
            let (_, y) = b.as_ref().unwrap();
            all_y.extend_from_slice(y.as_slice::<i64>().unwrap());
        }
        assert_ne!(all_y, (0_i64..20).collect::<Vec<_>>());
        // But total count must match.
        assert_eq!(all_y.len(), 20);
    }

    #[test]
    fn dataloader_total_items_consumed_matches_dataset_len() {
        let ds = dataset(7);
        let s = SequentialSampler::new(7);
        let mut dl = DataLoader::new(ds, s, 3);
        let batches = dl.iter_epoch();
        let total: usize = batches
            .iter()
            .map(|b| b.as_ref().unwrap().0.shape()[0])
            .sum();
        assert_eq!(total, 7);
    }

    #[test]
    fn dataloader_empty_dataset_yields_zero_batches() {
        let xs = Tensor::from_vec([0usize, 2], Vec::<f32>::new()).unwrap();
        let ys = Tensor::from_vec_typed::<i64, _>([0usize], Vec::<i64>::new()).unwrap();
        let ds = TensorDataset::new(xs, ys).unwrap();
        let s = SequentialSampler::new(0);
        let mut dl = DataLoader::new(ds, s, 4);
        assert_eq!(dl.num_batches(), 0);
        assert!(dl.iter_epoch().is_empty());
    }
}
