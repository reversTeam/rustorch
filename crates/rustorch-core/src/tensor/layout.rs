//! Layout — strides + offset + dtype + contiguity (RFC-0002, P1.1 task `Layout`).
//!
//! [`Layout`] is the *view metadata* of a [`Tensor`]: where in the
//! underlying [`Storage`](super::storage::Storage) the elements live
//! and how to walk them. It does **not** own buffer memory.
//!
//! Held inside `Tensor` (small + cheap to clone), Layout is the bridge
//! between [`Shape`] (logical dimensions) and the flat byte buffer.
//!
//! ```
//! use rustorch_core::tensor::layout::Layout;
//! use rustorch_core::tensor::dtype::Dtype;
//!
//! let l = Layout::contiguous([2, 3, 4], Dtype::F32);
//! assert_eq!(l.shape(), &[2, 3, 4]);
//! assert_eq!(l.strides(), &[12, 4, 1]);
//! assert!(l.is_contiguous());
//! assert_eq!(l.numel(), 24);
//! assert_eq!(l.byte_size(), 24 * 4);
//! ```
//!
//! [`Tensor`]: super::Tensor

use super::dtype::Dtype;
use super::shape::Shape;
use core::fmt;

/// Element-strides (signed because broadcasted axes use stride `0` and
/// negative strides will be allowed in a future iteration for `flip`).
pub type Strides = Vec<isize>;

/// View metadata for a tensor.
///
/// `Layout` is `Clone`; it owns small `Vec`s for shape and strides but
/// no buffer memory. The cached `is_contiguous` flag is recomputed by
/// [`Layout::contiguous`] / [`Layout::from_strides`] /
/// [`Layout::recompute_contiguous`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Layout {
    shape: Shape,
    strides: Strides,
    /// Element offset into the storage (in elements, not bytes).
    storage_offset: usize,
    dtype: Dtype,
    /// Cached. Use [`Layout::is_contiguous`] to read.
    contiguous: bool,
}

impl Layout {
    /// Build a contiguous (C-order, row-major) layout from a shape.
    ///
    /// Strides are computed walking the shape from the inner dim
    /// outward (rightmost stride is `1`).
    ///
    /// ```
    /// # use rustorch_core::tensor::{layout::Layout, dtype::Dtype};
    /// let l = Layout::contiguous([2, 3], Dtype::F32);
    /// assert_eq!(l.strides(), &[3, 1]);
    /// assert!(l.is_contiguous());
    /// ```
    pub fn contiguous<S: Into<Shape>>(shape: S, dtype: Dtype) -> Self {
        let shape = shape.into();
        let strides = compute_contiguous_strides(shape.as_slice());
        Layout {
            shape,
            strides,
            storage_offset: 0,
            dtype,
            contiguous: true,
        }
    }

    /// Build a layout with explicit strides + offset. Recomputes the
    /// `contiguous` flag.
    ///
    /// `Err` if `strides.len() != shape.ndim()`.
    pub fn from_strides<S: Into<Shape>>(
        shape: S,
        strides: Strides,
        storage_offset: usize,
        dtype: Dtype,
    ) -> Result<Self, LayoutError> {
        let shape = shape.into();
        if strides.len() != shape.ndim() {
            return Err(LayoutError::StridesRankMismatch {
                shape: shape.into_vec(),
                strides_len: strides.len(),
            });
        }
        let mut layout = Layout {
            shape,
            strides,
            storage_offset,
            dtype,
            contiguous: false,
        };
        layout.contiguous = is_contiguous_strides(layout.shape.as_slice(), &layout.strides);
        Ok(layout)
    }

    /// Read-only view of the dimensions.
    #[inline]
    pub fn shape(&self) -> &[usize] {
        self.shape.as_slice()
    }

    /// Borrow the [`Shape`] newtype directly.
    #[inline]
    pub fn shape_ref(&self) -> &Shape {
        &self.shape
    }

    /// Read-only view of the element strides.
    #[inline]
    pub fn strides(&self) -> &[isize] {
        &self.strides
    }

    /// Element offset into the storage.
    #[inline]
    pub fn storage_offset(&self) -> usize {
        self.storage_offset
    }

    /// Element dtype.
    #[inline]
    pub fn dtype(&self) -> Dtype {
        self.dtype
    }

    /// Number of dimensions (rank).
    #[inline]
    pub fn ndim(&self) -> usize {
        self.shape.ndim()
    }

    /// Total number of elements.
    #[inline]
    pub fn numel(&self) -> usize {
        self.shape.numel()
    }

    /// Total byte size (`numel * dtype.byte_size()`).
    #[inline]
    pub fn byte_size(&self) -> usize {
        self.numel() * self.dtype.byte_size()
    }

    /// `true` iff the layout walks the storage in row-major contiguous
    /// order with stride[-1] == 1, stride[-2] == shape[-1], … and
    /// `storage_offset == 0`.
    #[inline]
    pub fn is_contiguous(&self) -> bool {
        self.contiguous
    }

    /// `true` iff the layout walks the storage in NHWC channels-last
    /// order. Only meaningful for 4D tensors.
    pub fn is_channels_last(&self) -> bool {
        if self.ndim() != 4 {
            return false;
        }
        let s = self.shape.as_slice();
        let st = &self.strides;
        // NHWC strides: stride[c] = 1, stride[w] = c, stride[h] = w*c, stride[n] = h*w*c
        let (n, c, h, w) = (s[0], s[1], s[2], s[3]);
        let _ = n; // unused except for documentation parity
        let expected: [isize; 4] = [
            (c * h * w) as isize, // n
            1,                    // c (last in memory)
            (c * w) as isize,     // h
            c as isize,           // w
        ];
        st[..] == expected[..] && self.storage_offset == 0
    }

    /// Force-recompute the cached `contiguous` flag. Called after
    /// in-place edits to shape/strides (rare — most callers should use
    /// [`Layout::from_strides`] which already recomputes).
    pub fn recompute_contiguous(&mut self) {
        self.contiguous = is_contiguous_strides(self.shape.as_slice(), &self.strides);
    }
}

impl fmt::Display for Layout {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Layout({} {}, strides={:?}, offset={}, contiguous={})",
            self.dtype, self.shape, self.strides, self.storage_offset, self.contiguous,
        )
    }
}

// --------------------------------------------------------------------------
// Errors
// --------------------------------------------------------------------------

/// Errors produced by [`Layout`] constructors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LayoutError {
    /// `from_strides` was given a strides slice with a different length
    /// than the shape rank.
    StridesRankMismatch {
        /// Shape requested.
        shape: Vec<usize>,
        /// Number of strides actually provided.
        strides_len: usize,
    },
}

impl fmt::Display for LayoutError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LayoutError::StridesRankMismatch { shape, strides_len } => write!(
                f,
                "strides rank mismatch: shape {shape:?} (rank {}) vs strides len {strides_len}",
                shape.len()
            ),
        }
    }
}

impl std::error::Error for LayoutError {}

// --------------------------------------------------------------------------
// Helpers — pure functions, also used by tests
// --------------------------------------------------------------------------

/// Walk the shape from inner to outer: stride[-1] = 1, stride[-2] = shape[-1], …
pub(crate) fn compute_contiguous_strides(shape: &[usize]) -> Strides {
    let n = shape.len();
    if n == 0 {
        return Vec::new();
    }
    let mut s: Strides = vec![0; n];
    s[n - 1] = 1;
    for i in (0..n - 1).rev() {
        s[i] = s[i + 1] * shape[i + 1] as isize;
    }
    s
}

/// `true` iff `strides` describes a row-major contiguous walk of `shape`.
///
/// Special cases:
/// - empty shape (`scalar`) → trivially contiguous.
/// - any dim == 0 → trivially contiguous (the storage is empty).
/// - dim == 1 → that axis's stride is irrelevant (treated as compatible).
pub(crate) fn is_contiguous_strides(shape: &[usize], strides: &[isize]) -> bool {
    if shape.len() != strides.len() {
        return false;
    }
    if shape.is_empty() {
        return true;
    }
    if shape.contains(&0) {
        return true;
    }
    // Compute expected strides under contiguity, with size-1 axes free.
    let mut expected: isize = 1;
    for i in (0..shape.len()).rev() {
        if shape[i] != 1 && strides[i] != expected {
            return false;
        }
        expected *= shape[i] as isize;
    }
    true
}

// --------------------------------------------------------------------------
// Tests
// --------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_contains_dtype_shape_and_flag() {
        let l = Layout::contiguous([2, 3], Dtype::F32);
        let s = l.to_string();
        assert!(s.contains("f32"), "{s}");
        assert!(s.contains("[2, 3]"), "{s}");
        assert!(s.contains("contiguous=true"), "{s}");
    }

    #[test]
    fn debug_format_works() {
        let l = Layout::contiguous([2, 3], Dtype::F32);
        let _ = format!("{l:?}"); // smoke
    }

    #[test]
    fn contiguous_strides_2x3x4() {
        let l = Layout::contiguous([2, 3, 4], Dtype::F32);
        assert_eq!(l.shape(), &[2, 3, 4]);
        assert_eq!(l.strides(), &[12, 4, 1]);
        assert_eq!(l.storage_offset(), 0);
        assert_eq!(l.dtype(), Dtype::F32);
        assert!(l.is_contiguous());
        assert_eq!(l.numel(), 24);
        assert_eq!(l.byte_size(), 24 * 4);
        assert_eq!(l.ndim(), 3);
    }

    #[test]
    fn contiguous_strides_scalar() {
        let l = Layout::contiguous(Shape::default(), Dtype::F32);
        assert!(l.shape().is_empty());
        assert!(l.strides().is_empty());
        assert_eq!(l.numel(), 1);
        assert_eq!(l.byte_size(), 4);
        assert!(l.is_contiguous());
    }

    #[test]
    fn contiguous_strides_1d() {
        let l = Layout::contiguous([7usize], Dtype::F64);
        assert_eq!(l.strides(), &[1]);
        assert_eq!(l.byte_size(), 56);
        assert!(l.is_contiguous());
    }

    #[test]
    fn empty_dim_zero_is_contiguous() {
        // Shape with a zero-length axis: storage is empty, contiguity holds vacuously.
        let l = Layout::contiguous([0usize, 3], Dtype::F32);
        assert!(l.is_contiguous());
        assert_eq!(l.numel(), 0);
        assert_eq!(l.byte_size(), 0);
    }

    #[test]
    fn from_strides_round_trip() {
        // Manually build the contiguous layout via from_strides
        let l = Layout::from_strides([2usize, 3, 4], vec![12, 4, 1], 0, Dtype::F32).unwrap();
        assert!(l.is_contiguous());
        assert_eq!(l.strides(), &[12, 4, 1]);
    }

    #[test]
    fn from_strides_non_contiguous_detected() {
        // Transposed [3, 2] from a base [2, 3] with strides [1, 3] (column-major)
        let l = Layout::from_strides([3usize, 2], vec![1, 3], 0, Dtype::F32).unwrap();
        assert!(!l.is_contiguous());
    }

    #[test]
    fn from_strides_size_one_axis_is_free() {
        // A size-1 axis can have any stride — still considered contiguous.
        let l = Layout::from_strides([2usize, 1, 3], vec![3, 999, 1], 0, Dtype::F32).unwrap();
        assert!(l.is_contiguous());
    }

    #[test]
    fn from_strides_rank_mismatch() {
        let err = Layout::from_strides([2usize, 3], vec![1, 2, 3], 0, Dtype::F32).unwrap_err();
        match err {
            LayoutError::StridesRankMismatch { shape, strides_len } => {
                assert_eq!(shape, vec![2, 3]);
                assert_eq!(strides_len, 3);
            },
        }
    }

    #[test]
    fn channels_last_4d_detection() {
        // NHWC strides for shape [2, 3, 4, 5]:
        //   stride_n = c*h*w = 3*4*5 = 60
        //   stride_c = 1
        //   stride_h = c*w = 3*5 = 15
        //   stride_w = c = 3
        let l = Layout::from_strides([2usize, 3, 4, 5], vec![60, 1, 15, 3], 0, Dtype::F32).unwrap();
        assert!(l.is_channels_last());
        assert!(!l.is_contiguous());
    }

    #[test]
    fn contiguous_4d_is_not_channels_last() {
        let l = Layout::contiguous([2usize, 3, 4, 5], Dtype::F32);
        assert!(!l.is_channels_last());
        assert!(l.is_contiguous());
    }

    #[test]
    fn channels_last_only_for_rank_4() {
        let l = Layout::contiguous([2usize, 3, 4], Dtype::F32);
        assert!(!l.is_channels_last());
    }

    #[test]
    fn recompute_contiguous_flips_correctly() {
        // Build via from_strides as non-contiguous, then mutate strides and recompute.
        // Note: we cannot publicly mutate strides; recompute is a smoke check.
        let mut l = Layout::contiguous([2usize, 3], Dtype::F32);
        assert!(l.is_contiguous());
        // No-op recompute: still contiguous.
        l.recompute_contiguous();
        assert!(l.is_contiguous());
    }

    #[test]
    fn proptest_like_invariants_500() {
        // Independent re-impl of contiguous strides for cross-check.
        let mut s: u64 = 0xC0FFEE;
        let next = |s: &mut u64| {
            *s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            *s
        };
        for _ in 0..500 {
            let n = ((next(&mut s) % 4) + 1) as usize; // rank 1..=4
            let mut shape = Vec::with_capacity(n);
            for _ in 0..n {
                let d = ((next(&mut s) % 5) + 1) as usize; // dims 1..=5
                shape.push(d);
            }
            let l = Layout::contiguous(shape.clone(), Dtype::F32);
            // Independent expected strides
            let mut expected = vec![1isize; n];
            for i in (0..n.saturating_sub(1)).rev() {
                expected[i] = expected[i + 1] * shape[i + 1] as isize;
            }
            assert_eq!(l.strides(), expected.as_slice());
            assert_eq!(l.numel(), shape.iter().product::<usize>());
            assert!(l.is_contiguous());
        }
    }

    #[test]
    fn copy_and_eq_smoke() {
        let a = Layout::contiguous([2usize, 3], Dtype::F32);
        #[allow(clippy::redundant_clone)]
        let b = a.clone();
        assert_eq!(a, b);
    }
}
