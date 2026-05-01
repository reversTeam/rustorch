//! `Tensor` struct — RFC-0002 design (P1.1).
//!
//! [`Tensor`] is the user-facing handle to a multi-dimensional array.
//! It composes:
//!
//! - [`Storage`] — refcounted buffer (Cpu in v1).
//! - [`Layout`] — view metadata (shape, strides, offset, dtype,
//!   contiguity flag).
//! - [`VersionCounter`] — atomic counter shared with views, used by
//!   autograd to detect in-place mutation.
//! - `requires_grad: bool` — placeholder for autograd metadata.
//!   Full grad/grad_fn slot lands in P1.5.
//!
//! Cloning is cheap (Arc bumps on Storage + Version, shallow Vec clone
//! on Layout). Two clones share the same buffer and the same version
//! counter; this is what lets views observe in-place mutations on the
//! base tensor.
//!
//! ```
//! use rustorch_core::Tensor;
//! use rustorch_core::tensor::Dtype;
//!
//! let t = Tensor::from_vec([2usize, 3], vec![1.0_f32; 6]).unwrap();
//! assert_eq!(t.shape(), &[2, 3]);
//! assert_eq!(t.dtype(), Dtype::F32);
//! assert_eq!(t.numel(), 6);
//! assert!(t.is_contiguous());
//! ```

use super::dtype::{Dtype, Element};
use super::layout::Layout;
use super::shape::Shape;
use super::storage::{Storage, StorageError};
use super::version::VersionCounter;

/// Multi-dimensional array — the user-facing tensor.
///
/// Cloning is cheap: Storage and VersionCounter are `Arc`-bumped; the
/// Layout is a shallow `Vec` clone (typically <= 6 dims). Two clones
/// **share** the same buffer and version counter, so in-place ops on
/// one are visible to the other.
///
/// `Debug` is provided by [`crate::format`] (PyTorch-style); not derived.
#[derive(Clone)]
pub struct Tensor {
    storage: Storage,
    layout: Layout,
    version: VersionCounter,
    requires_grad: bool,
}

// --------------------------------------------------------------------------
// Construction
// --------------------------------------------------------------------------

impl Tensor {
    /// Build an `F32` tensor from a `Vec<f32>`.
    ///
    /// This is the v1 ergonomic constructor — most user code starts
    /// here. For other dtypes, use [`Tensor::from_vec_typed`].
    ///
    /// `Err(TensorError::ShapeDataMismatch)` if `data.len() != numel(shape)`.
    pub fn from_vec<S: Into<Shape>>(shape: S, data: Vec<f32>) -> Result<Tensor, TensorError> {
        Tensor::from_vec_typed::<f32, _>(shape, data)
    }

    /// Build a tensor of any [`Element`] type from a typed vec.
    ///
    /// ```
    /// # use rustorch_core::Tensor;
    /// # use rustorch_core::tensor::Dtype;
    /// let t = Tensor::from_vec_typed::<i32, _>([3usize], vec![1i32, 2, 3]).unwrap();
    /// assert_eq!(t.dtype(), Dtype::I32);
    /// ```
    pub fn from_vec_typed<T: Element, S: Into<Shape>>(
        shape: S,
        data: Vec<T>,
    ) -> Result<Tensor, TensorError> {
        let shape = shape.into();
        let expected = shape.numel();
        if data.len() != expected {
            return Err(TensorError::ShapeDataMismatch {
                expected,
                got: data.len(),
                shape: shape.into_vec(),
            });
        }
        let elem_size = core::mem::size_of::<T>();
        let byte_len = expected * elem_size;
        let mut storage = Storage::cpu_zeroed(byte_len)?;
        if byte_len > 0 {
            // SAFETY: storage was just allocated; we are unique owner.
            let dst = storage
                .as_bytes_mut()
                .expect("freshly allocated storage is unique");
            // SAFETY: T: Copy + 'static (Element bound). We bytewise copy
            // a Vec<T> into the byte buffer; alignment is satisfied
            // (CPU_ALIGN >= align_of::<T>() for all 8 v1 element types).
            unsafe {
                core::ptr::copy_nonoverlapping(
                    data.as_ptr() as *const u8,
                    dst.as_mut_ptr(),
                    byte_len,
                );
            }
        }
        let layout = Layout::contiguous(shape, T::DTYPE);
        Ok(Tensor {
            storage,
            layout,
            version: VersionCounter::new(),
            requires_grad: false,
        })
    }

    /// `F32` zeros of the given shape. Convenience for the common case.
    pub fn zeros<S: Into<Shape>>(shape: S) -> Tensor {
        Tensor::zeros_dtype::<S>(shape, Dtype::F32)
    }

    /// Zeros of the given shape and dtype (heap-allocated).
    pub fn zeros_dtype<S: Into<Shape>>(shape: S, dtype: Dtype) -> Tensor {
        let shape = shape.into();
        let byte_len = shape.numel() * dtype.byte_size();
        let storage = Storage::cpu_zeroed(byte_len)
            .expect("zeros: cpu allocation cannot fail for representable shapes");
        let layout = Layout::contiguous(shape, dtype);
        Tensor {
            storage,
            layout,
            version: VersionCounter::new(),
            requires_grad: false,
        }
    }

    /// `F32` ones of the given shape.
    pub fn ones<S: Into<Shape>>(shape: S) -> Tensor {
        let shape = shape.into();
        let n = shape.numel();
        Tensor::from_vec_typed::<f32, _>(shape, vec![1.0_f32; n])
            .expect("ones: shape numel matches Vec length by construction")
    }

    /// 0-dimensional `F32` scalar tensor.
    pub fn scalar(v: f32) -> Tensor {
        Tensor::from_vec_typed::<f32, _>(Shape::default(), vec![v])
            .expect("scalar: shape [] has numel 1")
    }

    /// Build a tensor from explicit `Storage`, `Layout`, and a fresh
    /// version counter. Used by view ops in `view.rs`.
    #[allow(dead_code)]
    pub(crate) fn from_parts(
        storage: Storage,
        layout: Layout,
        version: VersionCounter,
        requires_grad: bool,
    ) -> Tensor {
        Tensor {
            storage,
            layout,
            version,
            requires_grad,
        }
    }
}

// --------------------------------------------------------------------------
// Accessors
// --------------------------------------------------------------------------

impl Tensor {
    /// Read-only access to the dimensions.
    #[inline]
    pub fn shape(&self) -> &[usize] {
        self.layout.shape()
    }

    /// Read-only access to the [`Shape`] newtype.
    #[inline]
    pub fn shape_ref(&self) -> &Shape {
        self.layout.shape_ref()
    }

    /// Read-only access to the element strides.
    #[inline]
    pub fn strides(&self) -> &[isize] {
        self.layout.strides()
    }

    /// Element offset into the storage (in elements).
    #[inline]
    pub fn storage_offset(&self) -> usize {
        self.layout.storage_offset()
    }

    /// Element dtype.
    #[inline]
    pub fn dtype(&self) -> Dtype {
        self.layout.dtype()
    }

    /// Number of dimensions (rank).
    #[inline]
    pub fn ndim(&self) -> usize {
        self.layout.ndim()
    }

    /// Total number of elements (PyTorch's `numel`).
    #[inline]
    pub fn numel(&self) -> usize {
        self.layout.numel()
    }

    /// Alias for [`Tensor::numel`] for `f32` parity with the P0.3
    /// prototype API. Returns `0` for tensors with any zero-size axis.
    #[inline]
    pub fn len(&self) -> usize {
        self.layout.numel()
    }

    /// `true` iff `numel() == 0`.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.layout.numel() == 0
    }

    /// `true` iff the layout walks the storage in row-major contiguous
    /// order with `storage_offset == 0` (and stride[-1] == 1).
    #[inline]
    pub fn is_contiguous(&self) -> bool {
        self.layout.is_contiguous()
    }

    /// Borrow the underlying [`Layout`].
    #[inline]
    pub fn layout(&self) -> &Layout {
        &self.layout
    }

    /// Borrow the underlying [`Storage`].
    #[inline]
    pub fn storage(&self) -> &Storage {
        &self.storage
    }

    /// Borrow the [`VersionCounter`] shared with views.
    #[inline]
    pub fn version(&self) -> &VersionCounter {
        &self.version
    }

    /// `true` iff this tensor was marked as a leaf parameter for
    /// autograd. The flag is sticky on clone.
    #[inline]
    pub fn requires_grad(&self) -> bool {
        self.requires_grad
    }

    /// Set the `requires_grad` flag and return `self` (builder-style).
    /// PyTorch-faithful — name matches `torch.Tensor.requires_grad_`.
    #[must_use = "requires_grad_ returns the modified tensor; consume it"]
    pub fn requires_grad_(mut self, flag: bool) -> Tensor {
        self.requires_grad = flag;
        self
    }
}

// --------------------------------------------------------------------------
// Typed slice access
// --------------------------------------------------------------------------

impl Tensor {
    /// Borrow the buffer as `&[f32]` for the common P0.3-prototype path.
    ///
    /// **Panics** if the tensor is not contiguous F32 with
    /// `storage_offset == 0`. Use [`Tensor::as_slice`] for the typed,
    /// fallible form.
    #[inline]
    pub fn data(&self) -> &[f32] {
        match self.as_slice::<f32>() {
            Some(s) => s,
            None => panic!(
                "Tensor::data() requires contiguous F32 storage with offset 0; \
                 got dtype={:?}, contiguous={}, offset={}",
                self.dtype(),
                self.is_contiguous(),
                self.storage_offset(),
            ),
        }
    }

    /// Borrow the buffer as `&[T]` if the tensor is contiguous, has
    /// dtype matching `T`, and `storage_offset == 0`. Returns `None`
    /// otherwise — caller must `.contiguous()` first (P1.1 task
    /// `Conversions`).
    pub fn as_slice<T: Element>(&self) -> Option<&[T]> {
        if self.dtype() != T::DTYPE || !self.is_contiguous() || self.storage_offset() != 0 {
            return None;
        }
        // SAFETY: dtype matches T; the storage holds `numel * size_of::<T>()`
        // valid bytes by construction; the layout is contiguous starting
        // at offset 0.
        let slice = unsafe { self.storage.as_slice::<T>() };
        // The storage may have been allocated for a different numel
        // (rare but possible after slicing reattach in future tasks);
        // clamp to numel to match callers' expectations.
        Some(&slice[..self.numel()])
    }

    /// Convenience: extract a `Vec<T>` (allocates + copies). Useful for
    /// tests and for the safetensors writer.
    pub fn to_vec<T: Element>(&self) -> Option<Vec<T>> {
        self.as_slice::<T>().map(<[T]>::to_vec)
    }
}

// --------------------------------------------------------------------------
// Equality — element-wise on contiguous F32 tensors only (back-compat with
// the P0.3 prototype). General-purpose Eq comes in a later task.
// --------------------------------------------------------------------------

impl PartialEq for Tensor {
    fn eq(&self, other: &Self) -> bool {
        if self.shape() != other.shape() || self.dtype() != other.dtype() {
            return false;
        }
        match (self.as_slice::<f32>(), other.as_slice::<f32>()) {
            (Some(a), Some(b)) => a == b,
            _ => false, // non-F32 or non-contiguous — caller must use a typed compare
        }
    }
}

// --------------------------------------------------------------------------
// Errors
// --------------------------------------------------------------------------

/// Errors returned by [`Tensor`] constructors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TensorError {
    /// Constructor was given a `data` of wrong length for the requested
    /// shape.
    ShapeDataMismatch {
        /// Expected length = product of the requested shape.
        expected: usize,
        /// Actual length of the data passed in.
        got: usize,
        /// Shape requested (for diagnostic display).
        shape: Vec<usize>,
    },
    /// Underlying storage allocation failed.
    Storage(StorageError),
}

impl From<StorageError> for TensorError {
    fn from(e: StorageError) -> Self {
        TensorError::Storage(e)
    }
}

impl core::fmt::Display for TensorError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            TensorError::ShapeDataMismatch {
                expected,
                got,
                shape,
            } => write!(f, "shape {shape:?} requires {expected} elements, got {got}"),
            TensorError::Storage(e) => write!(f, "storage error: {e}"),
        }
    }
}

impl std::error::Error for TensorError {}

// --------------------------------------------------------------------------
// Tests
// --------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_vec_f32_basic() {
        let t = Tensor::from_vec([2usize, 3], vec![1.0_f32; 6]).unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.numel(), 6);
        assert_eq!(t.ndim(), 2);
        assert_eq!(t.dtype(), Dtype::F32);
        assert!(t.is_contiguous());
        assert_eq!(t.storage_offset(), 0);
        assert_eq!(t.strides(), &[3, 1]);
        assert_eq!(t.as_slice::<f32>().unwrap(), &[1.0_f32; 6]);
    }

    #[test]
    fn from_vec_typed_i64() {
        let t = Tensor::from_vec_typed::<i64, _>([3usize], vec![1_i64, 2, 3]).unwrap();
        assert_eq!(t.dtype(), Dtype::I64);
        assert_eq!(t.as_slice::<i64>().unwrap(), &[1, 2, 3]);
        // Mismatched dtype access returns None
        assert!(t.as_slice::<f32>().is_none());
    }

    #[test]
    fn from_vec_typed_bool() {
        let t =
            Tensor::from_vec_typed::<bool, _>([4usize], vec![true, false, true, false]).unwrap();
        assert_eq!(t.dtype(), Dtype::Bool);
        assert_eq!(t.as_slice::<bool>().unwrap(), &[true, false, true, false]);
    }

    #[test]
    fn shape_data_mismatch_returns_err() {
        let err = Tensor::from_vec([2usize, 3], vec![1.0_f32; 5]).unwrap_err();
        match err {
            TensorError::ShapeDataMismatch {
                expected,
                got,
                shape,
            } => {
                assert_eq!(expected, 6);
                assert_eq!(got, 5);
                assert_eq!(shape, vec![2, 3]);
            },
            _ => panic!("expected ShapeDataMismatch"),
        }
    }

    #[test]
    fn zeros_and_ones() {
        let z = Tensor::zeros([2usize, 3]);
        assert_eq!(z.as_slice::<f32>().unwrap(), &[0.0_f32; 6]);
        let o = Tensor::ones([2usize, 3]);
        assert_eq!(o.as_slice::<f32>().unwrap(), &[1.0_f32; 6]);
    }

    #[test]
    fn zeros_dtype_path() {
        let t = Tensor::zeros_dtype([3usize], Dtype::I64);
        assert_eq!(t.dtype(), Dtype::I64);
        assert_eq!(t.as_slice::<i64>().unwrap(), &[0_i64; 3]);
    }

    #[test]
    fn scalar_is_0d_with_numel_1() {
        let t = Tensor::scalar(2.71);
        assert_eq!(t.shape(), &[] as &[usize]);
        assert_eq!(t.numel(), 1);
        assert_eq!(t.as_slice::<f32>().unwrap(), &[2.71]);
    }

    #[test]
    fn empty_tensor_with_zero_dim() {
        let t = Tensor::zeros([0usize]);
        assert_eq!(t.shape(), &[0]);
        assert_eq!(t.numel(), 0);
        assert!(t.is_empty());
        assert!(t.as_slice::<f32>().unwrap().is_empty());
    }

    #[test]
    fn data_back_compat_returns_f32_slice() {
        let t = Tensor::from_vec([2usize, 2], vec![1.0_f32, 2.0, 3.0, 4.0]).unwrap();
        assert_eq!(t.data(), &[1.0_f32, 2.0, 3.0, 4.0]);
    }

    #[test]
    #[should_panic(expected = "Tensor::data() requires contiguous F32")]
    fn data_panics_on_non_f32() {
        let t = Tensor::from_vec_typed::<i32, _>([3usize], vec![1, 2, 3]).unwrap();
        let _ = t.data();
    }

    #[test]
    fn clone_shares_storage_and_version() {
        let a = Tensor::from_vec([3usize], vec![1.0_f32, 2.0, 3.0]).unwrap();
        let b = a.clone();
        // Both views see the same buffer pointer.
        assert_eq!(
            a.as_slice::<f32>().unwrap().as_ptr(),
            b.as_slice::<f32>().unwrap().as_ptr(),
        );
        // Storage refcount is 2.
        assert_eq!(a.storage().strong_count(), 2);
        // Versions share the underlying Arc.
        assert!(a.version().shares_with(b.version()));
        // Bumping through one is observed by the other.
        a.version().bump();
        assert_eq!(b.version().current(), 1);
    }

    #[test]
    fn requires_grad_default_false_and_setter() {
        let a = Tensor::scalar(1.0);
        assert!(!a.requires_grad());
        let a = a.requires_grad_(true);
        assert!(a.requires_grad());
    }

    #[test]
    fn partial_eq_compares_shape_and_data() {
        let a = Tensor::from_vec([2usize], vec![1.0_f32, 2.0]).unwrap();
        let b = Tensor::from_vec([2usize], vec![1.0_f32, 2.0]).unwrap();
        assert_eq!(a, b);
        let c = Tensor::from_vec([2usize], vec![1.0_f32, 3.0]).unwrap();
        assert_ne!(a, c);
        let d = Tensor::from_vec([1usize, 2], vec![1.0_f32, 2.0]).unwrap();
        assert_ne!(a, d); // different shape
    }

    #[test]
    fn to_vec_round_trip() {
        let t = Tensor::from_vec([3usize], vec![1.0_f32, 2.0, 3.0]).unwrap();
        assert_eq!(t.to_vec::<f32>().unwrap(), vec![1.0_f32, 2.0, 3.0]);
        assert!(t.to_vec::<i32>().is_none());
    }

    // (Debug formatting is tested in `crate::format::tests`.)

    #[test]
    fn empty_tensor_storage_byte_len_zero() {
        let t = Tensor::zeros([0usize]);
        assert_eq!(t.storage().byte_len(), 0);
        assert_eq!(t.numel(), 0);
        assert!(t.is_empty());
    }
}
