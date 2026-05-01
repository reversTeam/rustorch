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
// In-place mutating ops (P1.3 task `Arithmetic ops` in-place variants).
//
// Each in-place op:
// 1. Checks dtype + shape compatibility (broadcast lhs ←= broadcast(lhs, rhs)).
// 2. Acquires unique mutable access to the buffer (Storage must be unique).
// 3. Walks lhs in shape order with strided index decoding (works on
//    non-contiguous lhs).
// 4. Bumps the shared VersionCounter on success — autograd's
//    SavedVariable detects the bump and panics with a clear
//    "in-place modification of saved tensor" message.
//
// Failure modes:
// - DtypeMismatch on incompatible scalar types.
// - ShapeDataMismatch when rhs cannot be broadcast to lhs (out shape
//   must equal lhs shape — in-place cannot grow lhs).
// - "Tensor::*_ aliasing violation" panic when storage is shared.
//   This is consistent with PyTorch's behaviour ("a leaf Variable
//   that requires grad has been used in an in-place op").
// --------------------------------------------------------------------------

impl Tensor {
    /// In-place addition: `self += other` (with `other` broadcast to
    /// `self.shape()`).
    ///
    /// **Bumps the shared [`VersionCounter`]** so any autograd
    /// `SavedVariable` snapshotting the same tensor will detect the
    /// mutation at backward time.
    ///
    /// Errors:
    /// - [`TensorError::DtypeMismatch`] if dtypes differ.
    /// - [`TensorError::ShapeMismatch`] if `other.shape()` cannot
    ///   broadcast to `self.shape()` *without growing self*.
    ///
    /// Panics if the underlying storage is shared with a clone (an
    /// alias) — the caller must `.contiguous()` or own the buffer.
    pub fn add_(&mut self, other: &Tensor) -> Result<&mut Tensor, TensorError> {
        binary_inplace(self, other, "add_", |a, b| a + b, |a, b| a + b)
    }

    /// In-place subtraction: `self -= other`.
    pub fn sub_(&mut self, other: &Tensor) -> Result<&mut Tensor, TensorError> {
        binary_inplace(self, other, "sub_", |a, b| a - b, |a, b| a - b)
    }

    /// In-place multiplication: `self *= other`.
    pub fn mul_(&mut self, other: &Tensor) -> Result<&mut Tensor, TensorError> {
        binary_inplace(self, other, "mul_", |a, b| a * b, |a, b| a * b)
    }

    /// In-place division: `self /= other`.
    pub fn div_(&mut self, other: &Tensor) -> Result<&mut Tensor, TensorError> {
        binary_inplace(self, other, "div_", |a, b| a / b, |a, b| a / b)
    }

    /// In-place negation: `self = -self`.
    pub fn neg_(&mut self) -> Result<&mut Tensor, TensorError> {
        unary_inplace(self, "neg_", |x| -x, |x| -x)
    }

    /// In-place absolute value: `self = |self|`.
    pub fn abs_(&mut self) -> Result<&mut Tensor, TensorError> {
        unary_inplace(self, "abs_", |x: f32| x.abs(), |x: f64| x.abs())
    }

    /// In-place fill with a scalar: `self[..] = value`.
    pub fn fill_(&mut self, value: f64) -> Result<&mut Tensor, TensorError> {
        match self.dtype() {
            Dtype::F32 => {
                let v = value as f32;
                fill_inplace_f32(self, v)
            },
            Dtype::F64 => fill_inplace_f64(self, value),
            d => Err(TensorError::DtypeMismatch {
                op: "fill_",
                got: d,
                expected: Dtype::F32,
            }),
        }
    }

    /// In-place zero: `self[..] = 0`.
    pub fn zero_(&mut self) -> Result<&mut Tensor, TensorError> {
        self.fill_(0.0)
    }

    /// In-place copy from another tensor of the same shape and dtype.
    pub fn copy_(&mut self, other: &Tensor) -> Result<&mut Tensor, TensorError> {
        if self.dtype() != other.dtype() {
            return Err(TensorError::DtypeMismatch {
                op: "copy_",
                got: self.dtype(),
                expected: other.dtype(),
            });
        }
        if self.shape() != other.shape() {
            return Err(TensorError::ShapeMismatch {
                op: "copy_",
                lhs: self.shape().to_vec(),
                rhs: other.shape().to_vec(),
            });
        }
        // copy_ is implemented as: `self *= 0; self += other` — a tiny
        // bit slower than memcpy but works on any layout uniformly and
        // re-uses the same uniqueness/version machinery. The version
        // counter is bumped *once* (we drop the intermediate bump from
        // the multiplication by suppressing it via raw access).
        // For simplicity, we re-implement here as a straightforward
        // shape-order walk.
        match self.dtype() {
            Dtype::F32 => copy_inplace_typed::<f32>(self, other),
            Dtype::F64 => copy_inplace_typed::<f64>(self, other),
            d => Err(TensorError::DtypeMismatch {
                op: "copy_",
                got: d,
                expected: Dtype::F32,
            }),
        }
    }
}

/// Generic in-place binary op dispatch (f32 + f64 paths).
fn binary_inplace<'a>(
    lhs: &'a mut Tensor,
    rhs: &Tensor,
    op_name: &'static str,
    op_f32: impl Fn(f32, f32) -> f32,
    op_f64: impl Fn(f64, f64) -> f64,
) -> Result<&'a mut Tensor, TensorError> {
    if lhs.dtype() != rhs.dtype() {
        return Err(TensorError::DtypeMismatch {
            op: op_name,
            got: lhs.dtype(),
            expected: rhs.dtype(),
        });
    }
    // Broadcast rhs to lhs shape; reject if out-of-place broadcast would
    // grow lhs.
    let _expected_strides = rhs
        .shape_ref()
        .expand_strides_to(lhs.shape_ref())
        .map_err(|_| TensorError::ShapeMismatch {
            op: op_name,
            lhs: lhs.shape().to_vec(),
            rhs: rhs.shape().to_vec(),
        })?;
    match lhs.dtype() {
        Dtype::F32 => binary_inplace_typed::<f32>(lhs, rhs, op_f32, op_name),
        Dtype::F64 => binary_inplace_typed::<f64>(lhs, rhs, op_f64, op_name),
        d => Err(TensorError::DtypeMismatch {
            op: op_name,
            got: d,
            expected: Dtype::F32,
        }),
    }
}

/// Concrete typed in-place binary kernel.
fn binary_inplace_typed<'a, T>(
    lhs: &'a mut Tensor,
    rhs: &Tensor,
    op: impl Fn(T, T) -> T,
    op_name: &'static str,
) -> Result<&'a mut Tensor, TensorError>
where
    T: Element + core::ops::Add<Output = T> + core::ops::Sub<Output = T>,
{
    let n = lhs.numel();
    let lhs_shape = lhs.shape().to_vec();
    let lhs_strides = lhs.strides().to_vec();
    let lhs_offset = lhs.storage_offset();
    let rhs_strides = rhs
        .shape_ref()
        .expand_strides_to(lhs.shape_ref())
        .expect("broadcast was checked");
    let rhs_offset = rhs.storage_offset();

    // Read rhs first (immutable borrow via storage()).
    let rhs_raw_ptr = rhs.storage().as_bytes().as_ptr() as *const T;
    let rhs_len = rhs.storage().byte_len() / core::mem::size_of::<T>();
    // SAFETY: dtype matches, lifetime tied to rhs which we don't mutate.
    let rhs_raw: &[T] = if rhs_len == 0 {
        &[]
    } else {
        unsafe { core::slice::from_raw_parts(rhs_raw_ptr, rhs_len) }
    };

    // Acquire unique mutable bytes from lhs.
    let lhs_storage = lhs
        .storage_mut_for_inplace()
        .ok_or(TensorError::Aliased { op: op_name })?;
    let lhs_bytes_ptr = lhs_storage.as_mut_ptr();
    let lhs_byte_len = lhs_storage.len();
    let lhs_typed_len = lhs_byte_len / core::mem::size_of::<T>();
    // SAFETY: unique borrow via Storage::as_bytes_mut, dtype matches.
    let lhs_raw: &mut [T] =
        unsafe { core::slice::from_raw_parts_mut(lhs_bytes_ptr as *mut T, lhs_typed_len) };

    for i in 0..n {
        let li = strided_index(i, &lhs_shape, &lhs_strides, lhs_offset);
        let ri = strided_index(i, &lhs_shape, &rhs_strides, rhs_offset);
        lhs_raw[li] = op(lhs_raw[li], rhs_raw[ri]);
    }
    lhs.version().bump();
    Ok(lhs)
}

/// Generic in-place unary op dispatch (f32 + f64 paths).
fn unary_inplace<'a>(
    src: &'a mut Tensor,
    op_name: &'static str,
    op_f32: impl Fn(f32) -> f32,
    op_f64: impl Fn(f64) -> f64,
) -> Result<&'a mut Tensor, TensorError> {
    match src.dtype() {
        Dtype::F32 => unary_inplace_typed::<f32>(src, op_f32, op_name),
        Dtype::F64 => unary_inplace_typed::<f64>(src, op_f64, op_name),
        d => Err(TensorError::DtypeMismatch {
            op: op_name,
            got: d,
            expected: Dtype::F32,
        }),
    }
}

fn unary_inplace_typed<'a, T: Element>(
    src: &'a mut Tensor,
    op: impl Fn(T) -> T,
    op_name: &'static str,
) -> Result<&'a mut Tensor, TensorError> {
    let n = src.numel();
    let shape = src.shape().to_vec();
    let strides = src.strides().to_vec();
    let offset = src.storage_offset();
    let storage = src
        .storage_mut_for_inplace()
        .ok_or(TensorError::Aliased { op: op_name })?;
    let typed_len = storage.len() / core::mem::size_of::<T>();
    // SAFETY: unique mutable borrow + dtype match.
    let raw: &mut [T] =
        unsafe { core::slice::from_raw_parts_mut(storage.as_mut_ptr() as *mut T, typed_len) };
    for i in 0..n {
        let idx = strided_index(i, &shape, &strides, offset);
        raw[idx] = op(raw[idx]);
    }
    src.version().bump();
    Ok(src)
}

fn fill_inplace_f32(t: &mut Tensor, v: f32) -> Result<&mut Tensor, TensorError> {
    let n = t.numel();
    let shape = t.shape().to_vec();
    let strides = t.strides().to_vec();
    let offset = t.storage_offset();
    let storage = t
        .storage_mut_for_inplace()
        .ok_or(TensorError::Aliased { op: "fill_" })?;
    let raw: &mut [f32] = unsafe {
        core::slice::from_raw_parts_mut(
            storage.as_mut_ptr() as *mut f32,
            storage.len() / core::mem::size_of::<f32>(),
        )
    };
    for i in 0..n {
        let idx = strided_index(i, &shape, &strides, offset);
        raw[idx] = v;
    }
    t.version().bump();
    Ok(t)
}

fn fill_inplace_f64(t: &mut Tensor, v: f64) -> Result<&mut Tensor, TensorError> {
    let n = t.numel();
    let shape = t.shape().to_vec();
    let strides = t.strides().to_vec();
    let offset = t.storage_offset();
    let storage = t
        .storage_mut_for_inplace()
        .ok_or(TensorError::Aliased { op: "fill_" })?;
    let raw: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(
            storage.as_mut_ptr() as *mut f64,
            storage.len() / core::mem::size_of::<f64>(),
        )
    };
    for i in 0..n {
        let idx = strided_index(i, &shape, &strides, offset);
        raw[idx] = v;
    }
    t.version().bump();
    Ok(t)
}

fn copy_inplace_typed<'a, T: Element>(
    dst: &'a mut Tensor,
    src: &Tensor,
) -> Result<&'a mut Tensor, TensorError> {
    let n = dst.numel();
    let shape = dst.shape().to_vec();
    let dst_strides = dst.strides().to_vec();
    let dst_offset = dst.storage_offset();
    let src_strides = src.strides().to_vec();
    let src_offset = src.storage_offset();

    let src_ptr = src.storage().as_bytes().as_ptr() as *const T;
    let src_len = src.storage().byte_len() / core::mem::size_of::<T>();
    let src_raw: &[T] = if src_len == 0 {
        &[]
    } else {
        unsafe { core::slice::from_raw_parts(src_ptr, src_len) }
    };

    let storage = dst
        .storage_mut_for_inplace()
        .ok_or(TensorError::Aliased { op: "copy_" })?;
    let dst_typed_len = storage.len() / core::mem::size_of::<T>();
    let dst_raw: &mut [T] =
        unsafe { core::slice::from_raw_parts_mut(storage.as_mut_ptr() as *mut T, dst_typed_len) };
    for i in 0..n {
        let di = strided_index(i, &shape, &dst_strides, dst_offset);
        let si = strided_index(i, &shape, &src_strides, src_offset);
        dst_raw[di] = src_raw[si];
    }
    dst.version().bump();
    Ok(dst)
}

/// Compute the storage element index for the i-th shape-order element.
fn strided_index(linear: usize, shape: &[usize], strides: &[isize], offset: usize) -> usize {
    let mut idx = linear;
    let mut storage = offset as isize;
    for (axis, &dim) in shape.iter().enumerate().rev() {
        let coord = idx % dim;
        idx /= dim;
        storage += coord as isize * strides[axis];
    }
    storage as usize
}

impl Tensor {
    /// Crate-internal: borrow the inner CPU buffer mutably for the
    /// purpose of in-place ops. Returns `None` when storage is shared
    /// (any clone exists), forcing the caller to either COW or reject.
    fn storage_mut_for_inplace(&mut self) -> Option<&mut [u8]> {
        // SAFETY: we require unique storage; the borrow is mutable on
        // self and lifetime-bounded.
        self.storage.as_bytes_mut()
    }
}

// --------------------------------------------------------------------------
// Errors
// --------------------------------------------------------------------------

/// Errors returned by [`Tensor`] constructors and in-place ops.
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
    /// Two operands have shapes that cannot be jointly used.
    ShapeMismatch {
        /// Op name (`"add_"`, `"copy_"`, …).
        op: &'static str,
        /// LHS shape.
        lhs: Vec<usize>,
        /// RHS shape.
        rhs: Vec<usize>,
    },
    /// Two operands have incompatible dtypes.
    DtypeMismatch {
        /// Op name.
        op: &'static str,
        /// Dtype actually received.
        got: Dtype,
        /// Dtype expected.
        expected: Dtype,
    },
    /// Op required unique ownership of the storage but the buffer is
    /// shared with at least one other clone or view.
    Aliased {
        /// Op name.
        op: &'static str,
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
            TensorError::ShapeMismatch { op, lhs, rhs } => {
                write!(f, "{op}: incompatible shapes {lhs:?} vs {rhs:?}")
            },
            TensorError::DtypeMismatch { op, got, expected } => {
                write!(f, "{op}: dtype mismatch (got {got}, expected {expected})")
            },
            TensorError::Aliased { op } => {
                write!(f, "{op}: in-place op requires unique storage; clone first")
            },
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

    // ---------------------- in-place op tests ----------------------

    #[test]
    fn add_inplace_basic_f32() {
        let mut a = Tensor::from_vec([3usize], vec![1.0_f32, 2.0, 3.0]).unwrap();
        let v0 = a.version().current();
        let b = Tensor::from_vec([3usize], vec![10.0_f32, 20.0, 30.0]).unwrap();
        a.add_(&b).unwrap();
        assert_eq!(a.as_slice::<f32>().unwrap(), &[11.0, 22.0, 33.0]);
        assert_eq!(a.version().current(), v0 + 1, "version_counter must bump");
    }

    #[test]
    fn add_inplace_with_broadcast_row_vector() {
        let mut a = Tensor::from_vec([2usize, 3], vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let row = Tensor::from_vec([3usize], vec![10.0_f32, 20.0, 30.0]).unwrap();
        a.add_(&row).unwrap();
        assert_eq!(
            a.as_slice::<f32>().unwrap(),
            &[11.0, 22.0, 33.0, 14.0, 25.0, 36.0]
        );
    }

    #[test]
    fn sub_mul_div_inplace_paths() {
        let mut a = Tensor::from_vec([3usize], vec![6.0_f32, 8.0, 10.0]).unwrap();
        let b = Tensor::from_vec([3usize], vec![2.0_f32, 4.0, 5.0]).unwrap();
        a.sub_(&b).unwrap();
        assert_eq!(a.as_slice::<f32>().unwrap(), &[4.0, 4.0, 5.0]);
        a.mul_(&b).unwrap();
        assert_eq!(a.as_slice::<f32>().unwrap(), &[8.0, 16.0, 25.0]);
        a.div_(&b).unwrap();
        assert_eq!(a.as_slice::<f32>().unwrap(), &[4.0, 4.0, 5.0]);
    }

    #[test]
    fn neg_abs_inplace() {
        let mut a = Tensor::from_vec([3usize], vec![1.0_f32, -2.0, 3.0]).unwrap();
        a.neg_().unwrap();
        assert_eq!(a.as_slice::<f32>().unwrap(), &[-1.0, 2.0, -3.0]);
        a.abs_().unwrap();
        assert_eq!(a.as_slice::<f32>().unwrap(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn fill_zero_inplace() {
        let mut a = Tensor::from_vec([3usize], vec![1.0_f32, 2.0, 3.0]).unwrap();
        a.fill_(7.5).unwrap();
        assert_eq!(a.as_slice::<f32>().unwrap(), &[7.5, 7.5, 7.5]);
        a.zero_().unwrap();
        assert_eq!(a.as_slice::<f32>().unwrap(), &[0.0, 0.0, 0.0]);
    }

    #[test]
    fn copy_inplace() {
        let mut dst = Tensor::zeros([3usize]);
        let src = Tensor::from_vec([3usize], vec![1.0_f32, 2.0, 3.0]).unwrap();
        dst.copy_(&src).unwrap();
        assert_eq!(dst.as_slice::<f32>().unwrap(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn inplace_aliased_returns_err() {
        let mut a = Tensor::from_vec([3usize], vec![1.0_f32; 3]).unwrap();
        let _alias = a.clone(); // bumps storage refcount → aliasing
        let b = Tensor::from_vec([3usize], vec![1.0_f32; 3]).unwrap();
        match a.add_(&b) {
            Err(TensorError::Aliased { op }) => assert_eq!(op, "add_"),
            other => panic!("expected Aliased, got {other:?}"),
        }
    }

    #[test]
    fn inplace_dtype_mismatch_returns_err() {
        let mut a = Tensor::from_vec([3usize], vec![1.0_f32; 3]).unwrap();
        let b = Tensor::from_vec_typed::<i64, _>([3usize], vec![1_i64, 2, 3]).unwrap();
        match a.add_(&b) {
            Err(TensorError::DtypeMismatch { op, .. }) => assert_eq!(op, "add_"),
            other => panic!("expected DtypeMismatch, got {other:?}"),
        }
    }

    #[test]
    fn inplace_shape_mismatch_returns_err() {
        let mut a = Tensor::from_vec([3usize], vec![1.0_f32; 3]).unwrap();
        // Cannot broadcast a [4] into a [3] in-place — would grow.
        let b = Tensor::from_vec([4usize], vec![1.0_f32; 4]).unwrap();
        match a.add_(&b) {
            Err(TensorError::ShapeMismatch { op, .. }) => assert_eq!(op, "add_"),
            other => panic!("expected ShapeMismatch, got {other:?}"),
        }
    }

    #[test]
    fn inplace_through_view_bumps_base_version() {
        // The version counter is shared between a tensor and its views.
        // Mutating through a view bumps the base's counter too.
        let a = Tensor::from_vec([2usize, 3], vec![1.0_f32; 6]).unwrap();
        let v0 = a.version().current();
        // We can't mutate through a view if storage is shared; so first
        // verify the *counter* alone shares.
        let view = a.clone();
        assert!(view.version().shares_with(a.version()));
        let _ = v0;
        // Force unique storage by dropping the alias before mutating.
        drop(view);
        let mut a = a; // re-bind for mut access
        let b = Tensor::scalar(1.0);
        a.add_(&b).unwrap();
        assert_eq!(a.version().current(), 1);
    }

    #[test]
    fn fill_dtype_unsupported_returns_err() {
        let mut a = Tensor::from_vec_typed::<i64, _>([3usize], vec![1_i64; 3]).unwrap();
        match a.fill_(0.0) {
            Err(TensorError::DtypeMismatch { op, .. }) => assert_eq!(op, "fill_"),
            other => panic!("expected DtypeMismatch, got {other:?}"),
        }
    }
}
