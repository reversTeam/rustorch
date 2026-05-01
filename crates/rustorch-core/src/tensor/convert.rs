//! Conversions — `.contiguous()`, `.to(dtype)`, `from_vec`, `to_vec`
//! (RFC-0002, P1.1 task `Conversions`).
//!
//! These are the cross-cutting paths that move data between layouts,
//! dtypes, and Rust collections. v1 ships:
//!
//! - [`Tensor::contiguous`] — materialise a row-major contiguous copy
//!   (no-op if already contiguous).
//! - [`Tensor::to_dtype`] — element-wise cast across the 8 v1 dtypes.
//! - [`Tensor::from_slice`] — typed slice → owning tensor (mirror of
//!   [`Tensor::from_vec_typed`] without the move).
//! - [`Tensor::iter_elements`] — iterate over the elements of a non-
//!   contiguous tensor in shape order (used by `.contiguous()` and
//!   tests).
//!
//! Device transfer (`.to(device)`) is a Phase 2 plan once we have a
//! second backend; the v1 surface is CPU-only.

use super::dtype::{Dtype, Element};
use super::layout::Layout;
use super::shape::Shape;
use super::storage::Storage;
use super::tensor_impl::{Tensor, TensorError};
use super::version::VersionCounter;

/// Errors specific to conversion ops.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConvertError {
    /// `to_dtype` was called with a target equal to the source dtype.
    /// Not exposed publicly — collapsed into a no-op via the early
    /// return in [`Tensor::to_dtype`]. Reserved for future signed
    /// conversions where lossy casts must be opt-in.
    Lossy {
        /// Source dtype.
        from: Dtype,
        /// Target dtype.
        to: Dtype,
    },
}

impl core::fmt::Display for ConvertError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            ConvertError::Lossy { from, to } => write!(
                f,
                "potentially lossy dtype conversion {from} -> {to} requires explicit opt-in",
            ),
        }
    }
}

impl std::error::Error for ConvertError {}

// --------------------------------------------------------------------------
// Iterator over a (possibly non-contiguous) F32-typed tensor.
// --------------------------------------------------------------------------

impl Tensor {
    /// Iterate over the typed elements of the tensor in shape order
    /// (works for any contiguity).
    ///
    /// Returns `None` if `T` does not match the tensor's dtype.
    pub fn iter_elements<T: Element>(&self) -> Option<ShapeIter<'_, T>> {
        if self.dtype() != T::DTYPE {
            return None;
        }
        // SAFETY: dtype matches, storage holds at least
        // `numel * size_of::<T>()` valid bytes.
        let raw = unsafe { self.storage().as_slice::<T>() };
        Some(ShapeIter {
            data: raw,
            shape: self.shape().to_vec(),
            strides: self.strides().to_vec(),
            offset: self.storage_offset(),
            index: 0,
            numel: self.numel(),
        })
    }
}

/// Iterator over the elements of a tensor in shape order.
pub struct ShapeIter<'a, T> {
    data: &'a [T],
    shape: Vec<usize>,
    strides: Vec<isize>,
    offset: usize,
    index: usize,
    numel: usize,
}

impl<T: Copy> Iterator for ShapeIter<'_, T> {
    type Item = T;
    fn next(&mut self) -> Option<T> {
        if self.index >= self.numel {
            return None;
        }
        // Decode the linear index into multi-dim coords (row-major over `shape`).
        let mut linear = self.index;
        let mut storage_idx: isize = self.offset as isize;
        for (axis, &dim) in self.shape.iter().enumerate().rev() {
            let coord = linear % dim;
            linear /= dim;
            storage_idx += coord as isize * self.strides[axis];
        }
        let v = self.data[storage_idx as usize];
        self.index += 1;
        Some(v)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.numel.saturating_sub(self.index);
        (remaining, Some(remaining))
    }
}

impl<T: Copy> ExactSizeIterator for ShapeIter<'_, T> {}

// --------------------------------------------------------------------------
// .contiguous() — materialise a row-major contiguous copy
// --------------------------------------------------------------------------

impl Tensor {
    /// Return a contiguous version of this tensor. Zero-copy fast path
    /// when already contiguous; otherwise allocates a new buffer and
    /// re-orders elements in shape order.
    ///
    /// Always allocates a fresh [`VersionCounter`] (the contiguous
    /// copy is a new logical tensor — not a view of `self`).
    ///
    /// ```
    /// # use rustorch_core::Tensor;
    /// let t = Tensor::from_vec([2usize, 3], vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    /// let same = t.contiguous();
    /// assert!(same.is_contiguous());
    ///
    /// let tr = t.transpose(0, 1).unwrap();      // non-contig, [3, 2]
    /// assert!(!tr.is_contiguous());
    /// let copied = tr.contiguous();
    /// assert!(copied.is_contiguous());
    /// assert_eq!(copied.as_slice::<f32>().unwrap(), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    /// ```
    #[must_use = "contiguous() returns a (possibly new) tensor"]
    pub fn contiguous(&self) -> Tensor {
        if self.is_contiguous() && self.storage_offset() == 0 {
            return self.clone();
        }
        // Materialise — copy element-by-element into a fresh contiguous buffer.
        match self.dtype() {
            Dtype::F32 => copy_to_contiguous::<f32>(self),
            Dtype::F64 => copy_to_contiguous::<f64>(self),
            Dtype::F16 => copy_to_contiguous::<half::f16>(self),
            Dtype::BF16 => copy_to_contiguous::<half::bf16>(self),
            Dtype::I64 => copy_to_contiguous::<i64>(self),
            Dtype::I32 => copy_to_contiguous::<i32>(self),
            Dtype::I8 => copy_to_contiguous::<i8>(self),
            Dtype::Bool => copy_to_contiguous::<bool>(self),
        }
    }
}

fn copy_to_contiguous<T: Element>(src: &Tensor) -> Tensor {
    let collected: Vec<T> = src
        .iter_elements::<T>()
        .expect("dtype matches by exhaustive match on Tensor::contiguous")
        .collect();
    Tensor::from_vec_typed::<T, _>(src.shape().to_vec(), collected)
        .expect("contiguous copy preserves numel")
}

// --------------------------------------------------------------------------
// .to_dtype() — element-wise cast
// --------------------------------------------------------------------------

impl Tensor {
    /// Convert the tensor to a different element dtype, allocating a
    /// new buffer. No-op (cheap clone) if `target == self.dtype()`.
    ///
    /// Lossy conversions (e.g. `F32 -> I8`) are performed using Rust's
    /// `as`-cast semantics (saturating for float → int, truncating for
    /// integer narrowing). Same as `torch.Tensor.to(dtype)`.
    ///
    /// ```
    /// # use rustorch_core::Tensor;
    /// # use rustorch_core::tensor::Dtype;
    /// let t = Tensor::from_vec([3usize], vec![1.5_f32, 2.5, 3.5]).unwrap();
    /// let i = t.to_dtype(Dtype::I32);
    /// assert_eq!(i.dtype(), Dtype::I32);
    /// assert_eq!(i.as_slice::<i32>().unwrap(), &[1, 2, 3]);
    /// ```
    pub fn to_dtype(&self, target: Dtype) -> Tensor {
        if self.dtype() == target {
            return self.clone();
        }
        // Cast element-by-element through f64 (the v1 widest float)
        // for the float path, or i64 for integer paths. This keeps
        // the kernel surface tiny — perf-tuned dtype-pair kernels are
        // a separate codegen track (RFC-0005).
        match self.dtype() {
            Dtype::F32 => cast_via_f64::<f32>(self, target),
            Dtype::F64 => cast_via_f64::<f64>(self, target),
            Dtype::F16 => cast_via_f64::<half::f16>(self, target),
            Dtype::BF16 => cast_via_f64::<half::bf16>(self, target),
            Dtype::I64 => cast_via_f64::<i64>(self, target),
            Dtype::I32 => cast_via_f64::<i32>(self, target),
            Dtype::I8 => cast_via_f64::<i8>(self, target),
            Dtype::Bool => cast_from_bool(self, target),
        }
    }
}

trait ToF64 {
    fn to_f64(self) -> f64;
}
macro_rules! impl_to_f64 {
    ($($t:ty),*) => {$(
        impl ToF64 for $t {
            #[inline]
            fn to_f64(self) -> f64 { self as f64 }
        }
    )*};
}
impl_to_f64!(f32, f64, i64, i32, i8);

impl ToF64 for half::f16 {
    #[inline]
    fn to_f64(self) -> f64 {
        self.to_f64()
    }
}
impl ToF64 for half::bf16 {
    #[inline]
    fn to_f64(self) -> f64 {
        self.to_f64()
    }
}

fn cast_via_f64<T>(src: &Tensor, target: Dtype) -> Tensor
where
    T: Element + ToF64,
{
    let iter = src
        .iter_elements::<T>()
        .expect("Tensor::to_dtype dispatches on src.dtype, so T matches");
    let f64s: Vec<f64> = iter.map(<T as ToF64>::to_f64).collect();
    cast_from_f64(src.shape(), &f64s, target)
}

fn cast_from_f64(shape: &[usize], src: &[f64], target: Dtype) -> Tensor {
    match target {
        Dtype::F32 => {
            let v: Vec<f32> = src.iter().map(|&x| x as f32).collect();
            Tensor::from_vec_typed::<f32, _>(shape.to_vec(), v).expect("shape numel matches")
        },
        Dtype::F64 => {
            let v: Vec<f64> = src.to_vec();
            Tensor::from_vec_typed::<f64, _>(shape.to_vec(), v).expect("shape numel matches")
        },
        Dtype::F16 => {
            let v: Vec<half::f16> = src.iter().map(|&x| half::f16::from_f64(x)).collect();
            Tensor::from_vec_typed::<half::f16, _>(shape.to_vec(), v).expect("shape numel matches")
        },
        Dtype::BF16 => {
            let v: Vec<half::bf16> = src.iter().map(|&x| half::bf16::from_f64(x)).collect();
            Tensor::from_vec_typed::<half::bf16, _>(shape.to_vec(), v).expect("shape numel matches")
        },
        Dtype::I64 => {
            let v: Vec<i64> = src.iter().map(|&x| x as i64).collect();
            Tensor::from_vec_typed::<i64, _>(shape.to_vec(), v).expect("shape numel matches")
        },
        Dtype::I32 => {
            let v: Vec<i32> = src.iter().map(|&x| x as i32).collect();
            Tensor::from_vec_typed::<i32, _>(shape.to_vec(), v).expect("shape numel matches")
        },
        Dtype::I8 => {
            let v: Vec<i8> = src.iter().map(|&x| x as i8).collect();
            Tensor::from_vec_typed::<i8, _>(shape.to_vec(), v).expect("shape numel matches")
        },
        Dtype::Bool => {
            let v: Vec<bool> = src.iter().map(|&x| x != 0.0).collect();
            Tensor::from_vec_typed::<bool, _>(shape.to_vec(), v).expect("shape numel matches")
        },
    }
}

fn cast_from_bool(src: &Tensor, target: Dtype) -> Tensor {
    let bools: Vec<bool> = src
        .iter_elements::<bool>()
        .expect("dispatched from src.dtype == Bool")
        .collect();
    let f64s: Vec<f64> = bools.iter().map(|&b| if b { 1.0 } else { 0.0 }).collect();
    cast_from_f64(src.shape(), &f64s, target)
}

// --------------------------------------------------------------------------
// from_slice — non-consuming alternative to from_vec
// --------------------------------------------------------------------------

impl Tensor {
    /// Build a tensor by copying from a typed slice. Mirrors
    /// [`Tensor::from_vec_typed`] without consuming the input.
    pub fn from_slice<T: Element, S: Into<Shape>>(
        shape: S,
        data: &[T],
    ) -> Result<Tensor, TensorError> {
        Tensor::from_vec_typed::<T, _>(shape, data.to_vec())
    }
}

// --------------------------------------------------------------------------
// Re-export: Tensor::from_parts kept; expose a public construction
// helper for callers that already have a Storage + Layout pair (used by
// the safetensors loader in P1.8).
// --------------------------------------------------------------------------

/// Build a tensor from explicit `Storage` + `Layout` with a fresh
/// version counter and `requires_grad: false`. Public counterpart of
/// the crate-internal [`Tensor::from_parts`] used by view ops.
pub fn tensor_from_storage(storage: Storage, layout: Layout) -> Tensor {
    Tensor::from_parts(storage, layout, VersionCounter::new(), false)
}

// --------------------------------------------------------------------------
// Tests
// --------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn contiguous_no_op_on_contiguous() {
        let t = Tensor::from_vec([2usize, 3], vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let c = t.contiguous();
        // Same buffer pointer (.clone() shares storage).
        assert_eq!(
            t.as_slice::<f32>().unwrap().as_ptr(),
            c.as_slice::<f32>().unwrap().as_ptr(),
        );
    }

    #[test]
    fn contiguous_materialises_after_transpose() {
        let t = Tensor::from_vec([2usize, 3], vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let tr = t.transpose(0, 1).unwrap();
        assert!(!tr.is_contiguous());
        let c = tr.contiguous();
        assert!(c.is_contiguous());
        // Contiguous view of [3, 2] from [2, 3]: rows of original
        // become columns → shape order is (1,1) (1,2) (1,3) (2,1)
        // (2,2) (2,3) → (1,4) (2,5) (3,6) → flat = [1, 4, 2, 5, 3, 6].
        assert_eq!(c.shape(), &[3, 2]);
        assert_eq!(
            c.as_slice::<f32>().unwrap(),
            &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
        );
    }

    #[test]
    fn contiguous_after_narrow_drops_offset() {
        let t = Tensor::from_vec([5usize], vec![10.0_f32, 20.0, 30.0, 40.0, 50.0]).unwrap();
        let n = t.narrow(0, 1, 3).unwrap();
        assert_eq!(n.storage_offset(), 1);
        let c = n.contiguous();
        assert_eq!(c.storage_offset(), 0);
        assert_eq!(c.as_slice::<f32>().unwrap(), &[20.0, 30.0, 40.0]);
    }

    #[test]
    fn iter_elements_in_shape_order_for_transposed() {
        let t = Tensor::from_vec([2usize, 3], vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let tr = t.transpose(0, 1).unwrap();
        let v: Vec<f32> = tr.iter_elements::<f32>().unwrap().collect();
        // Transposed [3, 2] visits (0,0)=1 (0,1)=4 (1,0)=2 (1,1)=5 (2,0)=3 (2,1)=6
        assert_eq!(v, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn iter_elements_dtype_mismatch_returns_none() {
        let t = Tensor::from_vec_typed::<i32, _>([3usize], vec![1, 2, 3]).unwrap();
        assert!(t.iter_elements::<f32>().is_none());
    }

    #[test]
    fn iter_elements_size_hint() {
        let t = Tensor::from_vec([4usize], vec![1.0_f32; 4]).unwrap();
        let it = t.iter_elements::<f32>().unwrap();
        assert_eq!(it.size_hint(), (4, Some(4)));
    }

    #[test]
    fn to_dtype_no_op_when_same() {
        let t = Tensor::from_vec([3usize], vec![1.0_f32, 2.0, 3.0]).unwrap();
        let s = t.to_dtype(Dtype::F32);
        // Cheap clone (same buffer pointer).
        assert_eq!(
            t.as_slice::<f32>().unwrap().as_ptr(),
            s.as_slice::<f32>().unwrap().as_ptr(),
        );
    }

    #[test]
    fn to_dtype_f32_to_i32_truncates() {
        let t = Tensor::from_vec([3usize], vec![1.5_f32, 2.7, -3.4]).unwrap();
        let i = t.to_dtype(Dtype::I32);
        assert_eq!(i.dtype(), Dtype::I32);
        assert_eq!(i.as_slice::<i32>().unwrap(), &[1, 2, -3]);
    }

    #[test]
    fn to_dtype_i64_to_f64() {
        let t = Tensor::from_vec_typed::<i64, _>([3usize], vec![1_i64, 2, 3]).unwrap();
        let f = t.to_dtype(Dtype::F64);
        assert_eq!(f.dtype(), Dtype::F64);
        assert_eq!(f.as_slice::<f64>().unwrap(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn to_dtype_f16_round_trip() {
        let src: Vec<half::f16> = [1.0_f32, 2.0, 3.0]
            .iter()
            .map(|&x| half::f16::from_f32(x))
            .collect();
        let t = Tensor::from_vec_typed::<half::f16, _>([3usize], src).unwrap();
        let f = t.to_dtype(Dtype::F32);
        assert_eq!(f.as_slice::<f32>().unwrap(), &[1.0_f32, 2.0, 3.0]);
        let back = f.to_dtype(Dtype::F16);
        let got: Vec<f32> = back
            .as_slice::<half::f16>()
            .unwrap()
            .iter()
            .map(|h| h.to_f32())
            .collect();
        assert_eq!(got, vec![1.0_f32, 2.0, 3.0]);
    }

    #[test]
    fn to_dtype_f32_to_bool_zero_means_false() {
        let t = Tensor::from_vec([4usize], vec![0.0_f32, 1.0, -2.0, 0.0]).unwrap();
        let b = t.to_dtype(Dtype::Bool);
        assert_eq!(b.as_slice::<bool>().unwrap(), &[false, true, true, false]);
    }

    #[test]
    fn to_dtype_bool_to_i32() {
        let t = Tensor::from_vec_typed::<bool, _>([3usize], vec![true, false, true]).unwrap();
        let i = t.to_dtype(Dtype::I32);
        assert_eq!(i.as_slice::<i32>().unwrap(), &[1, 0, 1]);
    }

    #[test]
    fn from_slice_copies_data() {
        let data = [1.0_f32, 2.0, 3.0];
        let t = Tensor::from_slice::<f32, _>([3usize], &data).unwrap();
        assert_eq!(t.as_slice::<f32>().unwrap(), &data);
        // Original slice still usable (no consume).
        assert_eq!(data[0], 1.0);
    }

    #[test]
    fn tensor_from_storage_smoke() {
        // Build a Storage manually with 3 f32 zeros, wrap as Tensor.
        use crate::tensor::storage::Storage;
        let s = Storage::cpu_zeroed(12).unwrap();
        let l = Layout::contiguous([3usize], Dtype::F32);
        let t = tensor_from_storage(s, l);
        assert_eq!(t.shape(), &[3]);
        assert_eq!(t.dtype(), Dtype::F32);
        assert_eq!(t.as_slice::<f32>().unwrap(), &[0.0_f32; 3]);
    }

    #[test]
    fn round_trip_through_contiguous_via_permute() {
        let t = Tensor::from_vec([2usize, 3, 4], (0..24).map(|i| i as f32).collect()).unwrap();
        let p = t.permute(&[2, 0, 1]).unwrap();
        let c = p.contiguous();
        assert!(c.is_contiguous());
        // Re-permute back via inverse [1, 2, 0]
        let pp = c.permute(&[1, 2, 0]).unwrap().contiguous();
        assert_eq!(pp.as_slice::<f32>().unwrap(), t.as_slice::<f32>().unwrap());
    }

    #[test]
    fn convert_error_display_smoke() {
        let e = ConvertError::Lossy {
            from: Dtype::F32,
            to: Dtype::I8,
        };
        let s = e.to_string();
        assert!(s.contains("f32"));
        assert!(s.contains("i8"));
    }
}
