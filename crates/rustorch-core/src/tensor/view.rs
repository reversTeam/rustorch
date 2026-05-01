//! View operations — zero-copy reshapes (RFC-0002, P1.1 task `Views`).
//!
//! Every method here returns a new [`Tensor`] that shares the same
//! [`Storage`](super::storage::Storage) and [`VersionCounter`] as the
//! source — only the [`Layout`](super::layout::Layout) changes.
//!
//! - [`Tensor::view`] / [`Tensor::reshape`] — total-shape change.
//! - [`Tensor::transpose`] / [`Tensor::permute`] — axis reorder.
//! - [`Tensor::squeeze`] / [`Tensor::unsqueeze`] — size-1 axis manipulation.
//! - [`Tensor::narrow`] / [`Tensor::slice`] — axis-range subset.
//!
//! ```
//! use rustorch_core::Tensor;
//!
//! let t = Tensor::from_vec([2usize, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
//! let v = t.view([6usize]).unwrap();
//! assert_eq!(v.shape(), &[6]);
//! assert_eq!(v.as_slice::<f32>().unwrap(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
//!
//! let t2 = t.transpose(0, 1).unwrap();
//! assert_eq!(t2.shape(), &[3, 2]);
//! assert_eq!(t2.strides(), &[1, 3]);
//! assert!(!t2.is_contiguous());
//! ```

use super::layout::Layout;
use super::shape::Shape;
use super::tensor_impl::Tensor;

/// Errors returned by view operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ViewError {
    /// `view(new)` was called on a non-contiguous source.
    NonContiguous,
    /// New shape's `numel` differs from source's `numel`.
    NumelMismatch {
        /// numel of source layout.
        src_numel: usize,
        /// numel of requested shape.
        dst_numel: usize,
    },
    /// Permutation array length != source rank.
    PermuteRankMismatch {
        /// Source rank.
        src_ndim: usize,
        /// Permutation length.
        perm_len: usize,
    },
    /// Permutation array is not a valid permutation of `[0..ndim)`.
    InvalidPermutation {
        /// The bad permutation.
        perm: Vec<usize>,
    },
    /// `transpose(d0, d1)` referenced a dim out of range.
    DimOutOfRange {
        /// The offending dim index.
        dim: usize,
        /// Source rank.
        ndim: usize,
    },
    /// `narrow` / `slice` parameters out of range.
    NarrowOutOfRange {
        /// The dim being narrowed.
        dim: usize,
        /// Start element index.
        start: usize,
        /// End or length argument.
        end_or_len: usize,
        /// Size of the dim being narrowed.
        dim_size: usize,
    },
    /// `squeeze(dim)` was called on a dim whose size is not 1.
    NotSqueezable {
        /// The dim requested.
        dim: usize,
        /// Actual size of that dim.
        size: usize,
    },
}

impl core::fmt::Display for ViewError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            ViewError::NonContiguous => f.write_str(
                "Tensor::view requires the source to be contiguous; \
                 call .reshape() (with a possible copy) or .contiguous() first",
            ),
            ViewError::NumelMismatch {
                src_numel,
                dst_numel,
            } => write!(
                f,
                "view shape numel ({dst_numel}) does not match source numel ({src_numel})"
            ),
            ViewError::PermuteRankMismatch { src_ndim, perm_len } => write!(
                f,
                "permute: expected permutation of length {src_ndim}, got {perm_len}"
            ),
            ViewError::InvalidPermutation { perm } => {
                write!(
                    f,
                    "permute: {perm:?} is not a valid permutation of [0..ndim)"
                )
            },
            ViewError::DimOutOfRange { dim, ndim } => {
                write!(f, "dim {dim} out of range for tensor of rank {ndim}")
            },
            ViewError::NarrowOutOfRange {
                dim,
                start,
                end_or_len,
                dim_size,
            } => write!(
                f,
                "narrow/slice: range [{start}, {end_or_len}) on dim {dim} \
                 (size {dim_size}) is out of bounds"
            ),
            ViewError::NotSqueezable { dim, size } => write!(
                f,
                "squeeze: dim {dim} has size {size} (only size-1 dims can be squeezed)"
            ),
        }
    }
}

impl std::error::Error for ViewError {}

// --------------------------------------------------------------------------
// Tensor view methods
// --------------------------------------------------------------------------

impl Tensor {
    /// Strict, zero-copy reshape: requires source to be contiguous *and*
    /// `new_shape.numel() == self.numel()`.
    ///
    /// Use [`Tensor::reshape`] for the smart fallback that copies when
    /// the source is non-contiguous.
    pub fn view<S: Into<Shape>>(&self, new_shape: S) -> Result<Tensor, ViewError> {
        let new_shape = new_shape.into();
        if !self.is_contiguous() {
            return Err(ViewError::NonContiguous);
        }
        if new_shape.numel() != self.numel() {
            return Err(ViewError::NumelMismatch {
                src_numel: self.numel(),
                dst_numel: new_shape.numel(),
            });
        }
        let new_layout = Layout::contiguous(new_shape, self.dtype());
        Ok(Tensor::from_parts(
            self.storage().clone(),
            new_layout,
            self.version().clone(),
            self.requires_grad(),
        ))
    }

    /// Reshape with smart fallback. If the source is contiguous, this
    /// is a zero-copy view; if not, a copy is materialised first.
    ///
    /// In v1 the non-contiguous branch returns `ViewError::NonContiguous`
    /// — the contiguous-via-copy fallback lands together with
    /// [`Tensor::contiguous`] in the Conversions task.
    pub fn reshape<S: Into<Shape>>(&self, new_shape: S) -> Result<Tensor, ViewError> {
        if self.is_contiguous() {
            return self.view(new_shape);
        }
        Err(ViewError::NonContiguous)
    }

    /// Swap two axes (zero-copy). The result is non-contiguous unless
    /// one of the axes had size 1.
    ///
    /// ```
    /// # use rustorch_core::Tensor;
    /// let t = Tensor::from_vec([2usize, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    /// let t2 = t.transpose(0, 1).unwrap();
    /// assert_eq!(t2.shape(), &[3, 2]);
    /// assert_eq!(t2.strides(), &[1, 3]);
    /// ```
    pub fn transpose(&self, d0: usize, d1: usize) -> Result<Tensor, ViewError> {
        let ndim = self.ndim();
        if d0 >= ndim {
            return Err(ViewError::DimOutOfRange { dim: d0, ndim });
        }
        if d1 >= ndim {
            return Err(ViewError::DimOutOfRange { dim: d1, ndim });
        }
        let mut shape: Vec<usize> = self.shape().to_vec();
        let mut strides: Vec<isize> = self.strides().to_vec();
        shape.swap(d0, d1);
        strides.swap(d0, d1);
        let new_layout = Layout::from_strides(shape, strides, self.storage_offset(), self.dtype())
            .expect("transpose preserves shape/strides rank");
        Ok(Tensor::from_parts(
            self.storage().clone(),
            new_layout,
            self.version().clone(),
            self.requires_grad(),
        ))
    }

    /// Reorder axes by an explicit permutation (zero-copy).
    ///
    /// `perm` must be a permutation of `[0..ndim)`.
    pub fn permute(&self, perm: &[usize]) -> Result<Tensor, ViewError> {
        let ndim = self.ndim();
        if perm.len() != ndim {
            return Err(ViewError::PermuteRankMismatch {
                src_ndim: ndim,
                perm_len: perm.len(),
            });
        }
        // Check perm is a valid permutation of [0..ndim)
        let mut seen = vec![false; ndim];
        for &p in perm {
            if p >= ndim || seen[p] {
                return Err(ViewError::InvalidPermutation {
                    perm: perm.to_vec(),
                });
            }
            seen[p] = true;
        }
        let src_shape = self.shape();
        let src_strides = self.strides();
        let mut new_shape = Vec::with_capacity(ndim);
        let mut new_strides = Vec::with_capacity(ndim);
        for &p in perm {
            new_shape.push(src_shape[p]);
            new_strides.push(src_strides[p]);
        }
        let new_layout =
            Layout::from_strides(new_shape, new_strides, self.storage_offset(), self.dtype())
                .expect("permute preserves shape/strides rank");
        Ok(Tensor::from_parts(
            self.storage().clone(),
            new_layout,
            self.version().clone(),
            self.requires_grad(),
        ))
    }

    /// Insert a new size-1 axis at position `dim` (zero-copy).
    ///
    /// `dim` may equal `ndim` (push to the end).
    pub fn unsqueeze(&self, dim: usize) -> Result<Tensor, ViewError> {
        let ndim = self.ndim();
        if dim > ndim {
            return Err(ViewError::DimOutOfRange { dim, ndim });
        }
        let mut new_shape: Vec<usize> = self.shape().to_vec();
        let mut new_strides: Vec<isize> = self.strides().to_vec();
        new_shape.insert(dim, 1);
        // Stride for a size-1 axis is irrelevant; pick the next axis's
        // stride so it composes well with subsequent ops.
        let inserted_stride = if dim < ndim { new_strides[dim] } else { 1 };
        new_strides.insert(dim, inserted_stride);
        let new_layout =
            Layout::from_strides(new_shape, new_strides, self.storage_offset(), self.dtype())
                .expect("unsqueeze preserves shape/strides rank");
        Ok(Tensor::from_parts(
            self.storage().clone(),
            new_layout,
            self.version().clone(),
            self.requires_grad(),
        ))
    }

    /// Remove a size-1 axis at position `dim` (zero-copy).
    ///
    /// `Err` if the dim's size is not 1.
    pub fn squeeze(&self, dim: usize) -> Result<Tensor, ViewError> {
        let ndim = self.ndim();
        if dim >= ndim {
            return Err(ViewError::DimOutOfRange { dim, ndim });
        }
        let size = self.shape()[dim];
        if size != 1 {
            return Err(ViewError::NotSqueezable { dim, size });
        }
        let mut new_shape: Vec<usize> = self.shape().to_vec();
        let mut new_strides: Vec<isize> = self.strides().to_vec();
        new_shape.remove(dim);
        new_strides.remove(dim);
        let new_layout =
            Layout::from_strides(new_shape, new_strides, self.storage_offset(), self.dtype())
                .expect("squeeze preserves shape/strides rank");
        Ok(Tensor::from_parts(
            self.storage().clone(),
            new_layout,
            self.version().clone(),
            self.requires_grad(),
        ))
    }

    /// Take a contiguous range `[start, start+length)` along `dim`
    /// (zero-copy). Same as `t[:, start:start+length, :]` in PyTorch
    /// when `dim == 1`.
    pub fn narrow(&self, dim: usize, start: usize, length: usize) -> Result<Tensor, ViewError> {
        let ndim = self.ndim();
        if dim >= ndim {
            return Err(ViewError::DimOutOfRange { dim, ndim });
        }
        let dim_size = self.shape()[dim];
        let end = start
            .checked_add(length)
            .ok_or(ViewError::NarrowOutOfRange {
                dim,
                start,
                end_or_len: length,
                dim_size,
            })?;
        if end > dim_size {
            return Err(ViewError::NarrowOutOfRange {
                dim,
                start,
                end_or_len: length,
                dim_size,
            });
        }
        let mut new_shape: Vec<usize> = self.shape().to_vec();
        new_shape[dim] = length;
        let new_offset = self
            .storage_offset()
            .saturating_add((start as isize * self.strides()[dim]).max(0) as usize);
        let new_layout =
            Layout::from_strides(new_shape, self.strides().to_vec(), new_offset, self.dtype())
                .expect("narrow preserves shape/strides rank");
        Ok(Tensor::from_parts(
            self.storage().clone(),
            new_layout,
            self.version().clone(),
            self.requires_grad(),
        ))
    }

    /// Strided slice along `dim`: takes every `step`-th element in
    /// `[start, end)`. Step must be `>= 1`.
    pub fn slice(
        &self,
        dim: usize,
        start: usize,
        end: usize,
        step: usize,
    ) -> Result<Tensor, ViewError> {
        let ndim = self.ndim();
        if dim >= ndim {
            return Err(ViewError::DimOutOfRange { dim, ndim });
        }
        let dim_size = self.shape()[dim];
        if start > end || end > dim_size || step == 0 {
            return Err(ViewError::NarrowOutOfRange {
                dim,
                start,
                end_or_len: end,
                dim_size,
            });
        }
        let span = end - start;
        let new_dim = if span == 0 { 0 } else { (span - 1) / step + 1 };
        let mut new_shape: Vec<usize> = self.shape().to_vec();
        new_shape[dim] = new_dim;
        let mut new_strides: Vec<isize> = self.strides().to_vec();
        new_strides[dim] *= step as isize;
        let new_offset = self
            .storage_offset()
            .saturating_add((start as isize * self.strides()[dim]).max(0) as usize);
        let new_layout = Layout::from_strides(new_shape, new_strides, new_offset, self.dtype())
            .expect("slice preserves shape/strides rank");
        Ok(Tensor::from_parts(
            self.storage().clone(),
            new_layout,
            self.version().clone(),
            self.requires_grad(),
        ))
    }
}

// --------------------------------------------------------------------------
// Tests
// --------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn t_2x3() -> Tensor {
        Tensor::from_vec([2usize, 3], vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap()
    }

    #[test]
    fn view_contiguous_only() {
        let t = t_2x3();
        let v = t.view([6usize]).unwrap();
        assert_eq!(v.shape(), &[6]);
        // Same buffer
        assert_eq!(
            t.as_slice::<f32>().unwrap().as_ptr(),
            v.as_slice::<f32>().unwrap().as_ptr(),
        );
        // Storage refcount went from 1 to 2
        assert_eq!(t.storage().strong_count(), 2);
    }

    #[test]
    fn view_on_non_contiguous_returns_err() {
        let t = t_2x3().transpose(0, 1).unwrap();
        assert!(!t.is_contiguous());
        match t.view([6usize]) {
            Err(ViewError::NonContiguous) => {},
            other => panic!("expected NonContiguous, got {other:?}"),
        }
    }

    #[test]
    fn view_numel_mismatch() {
        let t = t_2x3();
        let err = t.view([7usize]).unwrap_err();
        assert_eq!(
            err,
            ViewError::NumelMismatch {
                src_numel: 6,
                dst_numel: 7,
            }
        );
    }

    #[test]
    fn reshape_falls_back_to_view_on_contiguous() {
        let t = t_2x3();
        let r = t.reshape([3usize, 2]).unwrap();
        assert_eq!(r.shape(), &[3, 2]);
    }

    #[test]
    fn reshape_on_non_contiguous_v1_returns_err() {
        let t = t_2x3().transpose(0, 1).unwrap();
        // v1: no fallback to copy yet (lands with .contiguous() in conversions task)
        assert!(matches!(t.reshape([6usize]), Err(ViewError::NonContiguous)));
    }

    #[test]
    fn transpose_swaps_strides() {
        let t = t_2x3();
        let t2 = t.transpose(0, 1).unwrap();
        assert_eq!(t2.shape(), &[3, 2]);
        assert_eq!(t2.strides(), &[1, 3]);
        assert!(!t2.is_contiguous());
    }

    #[test]
    fn transpose_involution() {
        let t = t_2x3();
        let t2 = t.transpose(0, 1).unwrap();
        let t3 = t2.transpose(0, 1).unwrap();
        assert_eq!(t3.shape(), t.shape());
        assert_eq!(t3.strides(), t.strides());
    }

    #[test]
    fn transpose_dim_out_of_range() {
        let t = t_2x3();
        match t.transpose(0, 5) {
            Err(ViewError::DimOutOfRange { dim, ndim }) => {
                assert_eq!(dim, 5);
                assert_eq!(ndim, 2);
            },
            other => panic!("expected DimOutOfRange, got {other:?}"),
        }
    }

    #[test]
    fn permute_reorders_dims_and_strides() {
        let t = Tensor::zeros([2usize, 3, 4]);
        let p = t.permute(&[2, 0, 1]).unwrap();
        assert_eq!(p.shape(), &[4, 2, 3]);
        // Source contiguous strides: [12, 4, 1]
        // After permute [2, 0, 1] → [1, 12, 4]
        assert_eq!(p.strides(), &[1, 12, 4]);
    }

    #[test]
    fn permute_inverse_identity() {
        let t = Tensor::zeros([2usize, 3, 4]);
        let p = t.permute(&[2, 0, 1]).unwrap();
        // Inverse of [2, 0, 1] is [1, 2, 0]
        let pp = p.permute(&[1, 2, 0]).unwrap();
        assert_eq!(pp.shape(), t.shape());
        assert_eq!(pp.strides(), t.strides());
    }

    #[test]
    fn permute_invalid_returns_err() {
        let t = Tensor::zeros([2usize, 3, 4]);
        // Duplicate index
        assert!(matches!(
            t.permute(&[0, 0, 2]),
            Err(ViewError::InvalidPermutation { .. }),
        ));
        // Out of range
        assert!(matches!(
            t.permute(&[0, 1, 5]),
            Err(ViewError::InvalidPermutation { .. }),
        ));
        // Wrong length
        assert!(matches!(
            t.permute(&[0, 1]),
            Err(ViewError::PermuteRankMismatch { .. }),
        ));
    }

    #[test]
    fn unsqueeze_inserts_size_1() {
        let t = Tensor::zeros([2usize, 3]);
        let u = t.unsqueeze(0).unwrap();
        assert_eq!(u.shape(), &[1, 2, 3]);
        let u_end = t.unsqueeze(2).unwrap();
        assert_eq!(u_end.shape(), &[2, 3, 1]);
    }

    #[test]
    fn squeeze_removes_size_1() {
        let t = Tensor::zeros([1usize, 2, 3]);
        let s = t.squeeze(0).unwrap();
        assert_eq!(s.shape(), &[2, 3]);
    }

    #[test]
    fn squeeze_non_unit_returns_err() {
        let t = Tensor::zeros([2usize, 3]);
        match t.squeeze(0) {
            Err(ViewError::NotSqueezable { dim, size }) => {
                assert_eq!(dim, 0);
                assert_eq!(size, 2);
            },
            other => panic!("expected NotSqueezable, got {other:?}"),
        }
    }

    #[test]
    fn narrow_takes_range() {
        // [4] → take [1, 1+2) = [1, 3) → length 2
        let t = Tensor::from_vec([4usize], vec![10.0_f32, 20.0, 30.0, 40.0]).unwrap();
        let n = t.narrow(0, 1, 2).unwrap();
        assert_eq!(n.shape(), &[2]);
        // Note: as_slice requires offset=0; n has offset=1 → returns None.
        assert!(n.as_slice::<f32>().is_none());
        assert_eq!(n.storage_offset(), 1);
        assert_eq!(n.strides(), &[1]);
    }

    #[test]
    fn narrow_out_of_range() {
        let t = Tensor::zeros([4usize]);
        // start + length > dim_size
        match t.narrow(0, 2, 5) {
            Err(ViewError::NarrowOutOfRange { dim_size, .. }) => assert_eq!(dim_size, 4),
            other => panic!("expected NarrowOutOfRange, got {other:?}"),
        }
        // dim out of range
        match t.narrow(2, 0, 1) {
            Err(ViewError::DimOutOfRange { ndim, .. }) => assert_eq!(ndim, 1),
            other => panic!("expected DimOutOfRange, got {other:?}"),
        }
    }

    #[test]
    fn slice_with_step() {
        // [6] with start=0, end=6, step=2 → 3 elements
        let t = Tensor::from_vec([6usize], (0..6).map(|i| i as f32).collect()).unwrap();
        let s = t.slice(0, 0, 6, 2).unwrap();
        assert_eq!(s.shape(), &[3]);
        assert_eq!(s.strides(), &[2]);
    }

    #[test]
    fn slice_step_one_equivalent_to_narrow() {
        let t = Tensor::zeros([4usize]);
        let n = t.narrow(0, 1, 2).unwrap();
        let s = t.slice(0, 1, 3, 1).unwrap();
        assert_eq!(s.shape(), n.shape());
        assert_eq!(s.strides(), n.strides());
        assert_eq!(s.storage_offset(), n.storage_offset());
    }

    #[test]
    fn slice_step_zero_returns_err() {
        let t = Tensor::zeros([4usize]);
        assert!(t.slice(0, 0, 4, 0).is_err());
    }

    #[test]
    fn slice_empty_range() {
        let t = Tensor::zeros([4usize]);
        let s = t.slice(0, 2, 2, 1).unwrap();
        assert_eq!(s.shape(), &[0]);
        assert_eq!(s.numel(), 0);
    }

    #[test]
    fn views_share_version_counter() {
        let t = t_2x3();
        let v = t.view([6usize]).unwrap();
        // Bumping through the original is observed by the view.
        t.version().bump();
        assert_eq!(v.version().current(), 1);
        // And vice versa.
        v.version().bump();
        assert_eq!(t.version().current(), 2);
    }

    #[test]
    fn view_error_display_smoke() {
        let e = ViewError::NumelMismatch {
            src_numel: 6,
            dst_numel: 7,
        };
        assert!(e.to_string().contains("6"));
        assert!(e.to_string().contains("7"));
    }
}
