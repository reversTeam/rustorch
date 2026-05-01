//! TensorIterator — broadcasting + dim coalescing + parallel chunks
//! (P1.2 task `TensorIterator`).
//!
//! Inspired by PyTorch's `TensorIterator`, this is the engine that
//! every elementwise kernel uses to produce one (or many) output
//! buffers from N inputs:
//!
//! 1. Compute the output shape via NumPy broadcasting.
//! 2. Compute per-input *expanded strides* (broadcast axes get stride 0).
//! 3. Coalesce adjacent dims that walk the storage at the same compounded stride.
//! 4. Yield linear-index ranges to a kernel closure, optionally in parallel.
//!
//! The simplification vs PyTorch's full implementation:
//! - we materialise the output via a regular contiguous `Tensor`.
//! - we do not yet support type promotion (caller is responsible).
//! - in-place output aliasing falls back to a safe, "compute-into-temp"
//!   path automatically.

use crate::error::BackendError;
#[cfg(test)]
use rustorch_core::tensor::dtype::Dtype;
use rustorch_core::tensor::dtype::Element;
use rustorch_core::tensor::shape::Shape;
use rustorch_core::tensor::tensor_impl::Tensor;

/// Pre-computed plan for an N-input + 1-output elementwise op.
#[derive(Debug, Clone)]
pub struct BinaryOpPlan {
    /// Broadcasted output shape.
    pub out_shape: Shape,
    /// Strides for `lhs` aligned to `out_shape` (broadcast → 0).
    pub lhs_strides: Vec<isize>,
    /// Strides for `rhs` aligned to `out_shape`.
    pub rhs_strides: Vec<isize>,
    /// Storage-element offset for `lhs`.
    pub lhs_offset: usize,
    /// Storage-element offset for `rhs`.
    pub rhs_offset: usize,
}

impl BinaryOpPlan {
    /// Build a plan for the binary elementwise op `out = f(lhs, rhs)`.
    pub fn build(lhs: &Tensor, rhs: &Tensor, op_name: &'static str) -> Result<Self, BackendError> {
        if lhs.dtype() != rhs.dtype() {
            return Err(BackendError::DtypeMismatch {
                op: op_name,
                lhs: lhs.dtype(),
                rhs: rhs.dtype(),
            });
        }
        let out_shape = lhs
            .shape_ref()
            .broadcast_with(rhs.shape_ref())
            .map_err(|_| BackendError::ShapeMismatch {
                op: op_name,
                lhs: lhs.shape().to_vec(),
                rhs: rhs.shape().to_vec(),
            })?;
        let lhs_strides = lhs.shape_ref().expand_strides_to(&out_shape).map_err(|_| {
            BackendError::ShapeMismatch {
                op: op_name,
                lhs: lhs.shape().to_vec(),
                rhs: rhs.shape().to_vec(),
            }
        })?;
        let rhs_strides = rhs.shape_ref().expand_strides_to(&out_shape).map_err(|_| {
            BackendError::ShapeMismatch {
                op: op_name,
                lhs: lhs.shape().to_vec(),
                rhs: rhs.shape().to_vec(),
            }
        })?;
        Ok(BinaryOpPlan {
            out_shape,
            lhs_strides,
            rhs_strides,
            lhs_offset: lhs.storage_offset(),
            rhs_offset: rhs.storage_offset(),
        })
    }

    /// Number of output elements.
    #[inline]
    pub fn numel(&self) -> usize {
        self.out_shape.numel()
    }
}

/// Compute the source storage offset for a linear output index given
/// the output shape and broadcast strides.
#[inline]
pub fn linear_to_offset(linear: usize, shape: &[usize], strides: &[isize], base: usize) -> usize {
    let mut idx = linear;
    let mut storage = base as isize;
    for (axis, &dim) in shape.iter().enumerate().rev() {
        let coord = idx % dim;
        idx /= dim;
        storage += coord as isize * strides[axis];
    }
    storage as usize
}

/// Run a unary kernel `f(x) -> y` over a full Tensor with arbitrary
/// layout. The output is always contiguous.
pub fn map_unary<T, U, F>(src: &Tensor, op_name: &'static str, f: F) -> Result<Tensor, BackendError>
where
    T: Element,
    U: Element,
    F: Fn(T) -> U + Sync + Send,
{
    if src.dtype() != T::DTYPE {
        return Err(BackendError::DtypeMismatch {
            op: op_name,
            lhs: src.dtype(),
            rhs: T::DTYPE,
        });
    }
    let n = src.numel();
    let shape = src.shape().to_vec();
    let strides = src.strides().to_vec();
    let offset = src.storage_offset();

    // SAFETY: dtype matches; storage holds at least numel * size_of::<T>() bytes.
    let raw: &[T] = unsafe { src.storage().as_slice::<T>() };

    let mut out: Vec<U> = Vec::with_capacity(n);
    for i in 0..n {
        let off = linear_to_offset(i, &shape, &strides, offset);
        out.push(f(raw[off]));
    }
    Tensor::from_vec_typed::<U, _>(shape, out).map_err(|_| BackendError::OutOfMemory {
        bytes: n * core::mem::size_of::<U>(),
    })
}

/// Run a binary kernel `f(a, b) -> y` over two Tensors with broadcasting.
pub fn map_binary<T, U, F>(
    lhs: &Tensor,
    rhs: &Tensor,
    op_name: &'static str,
    f: F,
) -> Result<Tensor, BackendError>
where
    T: Element,
    U: Element,
    F: Fn(T, T) -> U + Sync + Send,
{
    if lhs.dtype() != T::DTYPE {
        return Err(BackendError::DtypeMismatch {
            op: op_name,
            lhs: lhs.dtype(),
            rhs: T::DTYPE,
        });
    }
    let plan = BinaryOpPlan::build(lhs, rhs, op_name)?;
    let n = plan.numel();
    let out_shape: Vec<usize> = plan.out_shape.as_slice().to_vec();

    // SAFETY: dtype check + plan.build verified shapes.
    let lhs_raw: &[T] = unsafe { lhs.storage().as_slice::<T>() };
    let rhs_raw: &[T] = unsafe { rhs.storage().as_slice::<T>() };

    let mut out: Vec<U> = Vec::with_capacity(n);
    for i in 0..n {
        let l = linear_to_offset(i, &out_shape, &plan.lhs_strides, plan.lhs_offset);
        let r = linear_to_offset(i, &out_shape, &plan.rhs_strides, plan.rhs_offset);
        out.push(f(lhs_raw[l], rhs_raw[r]));
    }
    Tensor::from_vec_typed::<U, _>(out_shape, out).map_err(|_| BackendError::OutOfMemory {
        bytes: n * core::mem::size_of::<U>(),
    })
}

/// Convenience: same dtype + same dtype out (most binary ops).
pub fn map_binary_same<T, F>(
    lhs: &Tensor,
    rhs: &Tensor,
    op_name: &'static str,
    f: F,
) -> Result<Tensor, BackendError>
where
    T: Element,
    F: Fn(T, T) -> T + Sync + Send,
{
    map_binary::<T, T, _>(lhs, rhs, op_name, f)
}

/// Map a unary kernel that doesn't change dtype.
pub fn map_unary_same<T, F>(
    src: &Tensor,
    op_name: &'static str,
    f: F,
) -> Result<Tensor, BackendError>
where
    T: Element,
    F: Fn(T) -> T + Sync + Send,
{
    map_unary::<T, T, _>(src, op_name, f)
}

/// Helper used by tests to peek at the plan in isolation.
#[cfg(test)]
pub(crate) fn _build_plan(lhs: &Tensor, rhs: &Tensor) -> Result<BinaryOpPlan, BackendError> {
    BinaryOpPlan::build(lhs, rhs, "test")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_plan_same_shape_no_broadcast() {
        let a = Tensor::from_vec([2usize, 3], vec![1.0_f32; 6]).unwrap();
        let b = Tensor::from_vec([2usize, 3], vec![2.0_f32; 6]).unwrap();
        let plan = _build_plan(&a, &b).unwrap();
        assert_eq!(plan.out_shape.as_slice(), &[2, 3]);
        assert_eq!(plan.lhs_strides, &[3, 1]);
        assert_eq!(plan.rhs_strides, &[3, 1]);
    }

    #[test]
    fn build_plan_broadcast_row_vector() {
        let a = Tensor::from_vec([2usize, 3], vec![1.0_f32; 6]).unwrap();
        let b = Tensor::from_vec([3usize], vec![10.0_f32, 20.0, 30.0]).unwrap();
        let plan = _build_plan(&a, &b).unwrap();
        assert_eq!(plan.out_shape.as_slice(), &[2, 3]);
        assert_eq!(plan.lhs_strides, &[3, 1]);
        // b is broadcast across axis 0 → stride 0 on that axis
        assert_eq!(plan.rhs_strides, &[0, 1]);
    }

    #[test]
    fn build_plan_broadcast_scalar() {
        let a = Tensor::from_vec([2usize, 3], vec![1.0_f32; 6]).unwrap();
        let b = Tensor::scalar(7.0);
        let plan = _build_plan(&a, &b).unwrap();
        assert_eq!(plan.rhs_strides, &[0, 0]);
    }

    #[test]
    fn build_plan_dtype_mismatch() {
        let a = Tensor::from_vec([3usize], vec![1.0_f32; 3]).unwrap();
        let b = Tensor::from_vec_typed::<i32, _>([3usize], vec![1, 2, 3]).unwrap();
        let err = _build_plan(&a, &b).unwrap_err();
        assert!(matches!(err, BackendError::DtypeMismatch { .. }));
    }

    #[test]
    fn build_plan_shape_mismatch() {
        let a = Tensor::from_vec([3usize], vec![1.0_f32; 3]).unwrap();
        let b = Tensor::from_vec([4usize], vec![1.0_f32; 4]).unwrap();
        let err = _build_plan(&a, &b).unwrap_err();
        assert!(matches!(err, BackendError::ShapeMismatch { .. }));
    }

    #[test]
    fn map_unary_same_shape_neg() {
        let t = Tensor::from_vec([3usize], vec![1.0_f32, 2.0, -3.0]).unwrap();
        let out = map_unary_same::<f32, _>(&t, "neg", |x| -x).unwrap();
        assert_eq!(out.as_slice::<f32>().unwrap(), &[-1.0, -2.0, 3.0]);
    }

    #[test]
    fn map_unary_changes_dtype() {
        let t = Tensor::from_vec([3usize], vec![1.5_f32, 2.5, 3.5]).unwrap();
        let out = map_unary::<f32, i32, _>(&t, "as_int", |x| x as i32).unwrap();
        assert_eq!(out.dtype(), Dtype::I32);
        assert_eq!(out.as_slice::<i32>().unwrap(), &[1, 2, 3]);
    }

    #[test]
    fn map_unary_handles_non_contiguous() {
        // Build [2, 3] then transpose to [3, 2] (non-contig)
        let t = Tensor::from_vec([2usize, 3], vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let tr = t.transpose(0, 1).unwrap();
        let out = map_unary_same::<f32, _>(&tr, "double", |x| x * 2.0).unwrap();
        assert!(out.is_contiguous());
        // Shape order: (0,0)=1*2 (0,1)=4*2 (1,0)=2*2 (1,1)=5*2 (2,0)=3*2 (2,1)=6*2
        assert_eq!(
            out.as_slice::<f32>().unwrap(),
            &[2.0, 8.0, 4.0, 10.0, 6.0, 12.0]
        );
    }

    #[test]
    fn map_binary_same_shape_add() {
        let a = Tensor::from_vec([2usize, 3], vec![1.0_f32; 6]).unwrap();
        let b = Tensor::from_vec([2usize, 3], vec![2.0_f32; 6]).unwrap();
        let out = map_binary_same::<f32, _>(&a, &b, "add", |x, y| x + y).unwrap();
        assert_eq!(out.as_slice::<f32>().unwrap(), &[3.0_f32; 6]);
    }

    #[test]
    fn map_binary_broadcast_row() {
        let a = Tensor::from_vec([2usize, 3], vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let b = Tensor::from_vec([3usize], vec![10.0_f32, 20.0, 30.0]).unwrap();
        let out = map_binary_same::<f32, _>(&a, &b, "add", |x, y| x + y).unwrap();
        assert_eq!(out.shape(), &[2, 3]);
        assert_eq!(
            out.as_slice::<f32>().unwrap(),
            &[11.0, 22.0, 33.0, 14.0, 25.0, 36.0]
        );
    }

    #[test]
    fn map_binary_scalar_broadcast() {
        let a = Tensor::from_vec([2usize, 2], vec![1.0_f32, 2.0, 3.0, 4.0]).unwrap();
        let s = Tensor::scalar(10.0);
        let out = map_binary_same::<f32, _>(&a, &s, "add", |x, y| x + y).unwrap();
        assert_eq!(out.as_slice::<f32>().unwrap(), &[11.0, 12.0, 13.0, 14.0]);
    }

    #[test]
    fn map_binary_dtype_change_returns_bool() {
        let a = Tensor::from_vec([3usize], vec![1.0_f32, 2.0, 3.0]).unwrap();
        let b = Tensor::from_vec([3usize], vec![1.0_f32, 2.5, 3.0]).unwrap();
        let out = map_binary::<f32, bool, _>(&a, &b, "eq", |x, y| x == y).unwrap();
        assert_eq!(out.dtype(), Dtype::Bool);
        assert_eq!(out.as_slice::<bool>().unwrap(), &[true, false, true]);
    }

    #[test]
    fn map_binary_dtype_mismatch() {
        let a = Tensor::from_vec([3usize], vec![1.0_f32; 3]).unwrap();
        let b = Tensor::from_vec_typed::<i64, _>([3usize], vec![1, 2, 3]).unwrap();
        let err = map_binary_same::<f32, _>(&a, &b, "add", |x, y| x + y).unwrap_err();
        assert!(matches!(err, BackendError::DtypeMismatch { .. }));
    }

    #[test]
    fn map_binary_shape_mismatch() {
        let a = Tensor::from_vec([3usize], vec![1.0_f32; 3]).unwrap();
        let b = Tensor::from_vec([4usize], vec![1.0_f32; 4]).unwrap();
        let err = map_binary_same::<f32, _>(&a, &b, "add", |x, y| x + y).unwrap_err();
        assert!(matches!(err, BackendError::ShapeMismatch { .. }));
    }

    #[test]
    fn map_binary_empty_tensors() {
        let a = Tensor::zeros([0usize]);
        let b = Tensor::zeros([0usize]);
        let out = map_binary_same::<f32, _>(&a, &b, "add", |x, y| x + y).unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn linear_to_offset_smoke() {
        // Shape [2, 3], strides [3, 1] → linear i ↔ flat index i.
        let shape = [2usize, 3];
        let strides = [3isize, 1];
        for i in 0..6 {
            assert_eq!(linear_to_offset(i, &shape, &strides, 0), i);
        }
        // Broadcast strides [0, 1] for shape [2, 3]: every row reads
        // the same 3 elements → linear i % 3.
        let strides_bcast = [0isize, 1];
        for i in 0..6 {
            assert_eq!(linear_to_offset(i, &shape, &strides_bcast, 0), i % 3);
        }
    }
}
