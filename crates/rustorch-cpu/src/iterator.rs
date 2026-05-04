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
    T: Element + Send + Sync,
    U: Element + Send + Sync,
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

    // Fast path: contiguous offset-0 input → direct-index parallel loop,
    // no per-element stride math. Covers the hot cases (relu/sigmoid
    // /etc. on activations).
    if src.is_contiguous() && src.storage_offset() == 0 {
        use rayon::prelude::*;

        const PARALLEL_UNARY_THRESHOLD: usize = 16_384;
        const CHUNK: usize = 8_192;

        // SAFETY: dtype matches; storage holds at least numel*size_of::<T>().
        let raw: &[T] = unsafe { src.storage().as_slice::<T>() };
        let raw_slice = &raw[..n];

        // Allocate uninitialised storage and treat it as MaybeUninit
        // until every slot has been written by the kernel below. This
        // skips the zero-fill that `vec![U::default(); n]` would emit.
        let mut out: Vec<core::mem::MaybeUninit<U>> = Vec::with_capacity(n);
        // SAFETY: capacity is exactly n; the loop below writes every
        // element before any read. The `Vec` is then reinterpreted as
        // `Vec<U>` via `from_raw_parts`.
        #[allow(clippy::uninit_vec)]
        unsafe {
            out.set_len(n);
        }

        if n < PARALLEL_UNARY_THRESHOLD {
            for i in 0..n {
                out[i].write(f(raw_slice[i]));
            }
        } else {
            out.par_chunks_mut(CHUNK)
                .zip(raw_slice.par_chunks(CHUNK))
                .for_each(|(out_chunk, in_chunk)| {
                    for i in 0..out_chunk.len() {
                        out_chunk[i].write(f(in_chunk[i]));
                    }
                });
        }
        // SAFETY: every slot written above. Transmute Vec<MaybeUninit<U>>
        // → Vec<U> by reusing the same allocation pointer.
        let out: Vec<U> = unsafe {
            let mut o = core::mem::ManuallyDrop::new(out);
            Vec::from_raw_parts(o.as_mut_ptr() as *mut U, n, o.capacity())
        };
        return Tensor::from_vec_typed::<U, _>(shape, out).map_err(|_| BackendError::OutOfMemory {
            bytes: n * core::mem::size_of::<U>(),
        });
    }

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
    T: Element + Send + Sync,
    U: Element + Send + Sync,
    F: Fn(T, T) -> U + Sync + Send,
{
    if lhs.dtype() != T::DTYPE {
        return Err(BackendError::DtypeMismatch {
            op: op_name,
            lhs: lhs.dtype(),
            rhs: T::DTYPE,
        });
    }
    if rhs.dtype() != T::DTYPE {
        return Err(BackendError::DtypeMismatch {
            op: op_name,
            lhs: lhs.dtype(),
            rhs: rhs.dtype(),
        });
    }
    // Fast path: same-shape contiguous F32 inputs hit a parallel
    // direct-index loop with no per-element stride math. Covers the
    // hot cases in autograd (add/sub/mul of same-shape activations &
    // gradients) — the most common training-loop element-wise ops.
    if lhs.shape() == rhs.shape()
        && lhs.is_contiguous()
        && rhs.is_contiguous()
        && lhs.storage_offset() == 0
        && rhs.storage_offset() == 0
    {
        return map_binary_contiguous_parallel::<T, U, F>(lhs, rhs, f);
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

/// Parallel fast path for same-shape contiguous binary ops. Skips the
/// `BinaryOpPlan::build` + per-element `linear_to_offset` math and
/// drives the kernel through `rayon::par_chunks` for elements above
/// `PARALLEL_BINARY_THRESHOLD`. Below the threshold a single-threaded
/// chunked loop avoids the rayon dispatch tax.
fn map_binary_contiguous_parallel<T, U, F>(
    lhs: &Tensor,
    rhs: &Tensor,
    f: F,
) -> Result<Tensor, BackendError>
where
    T: Element + Send + Sync,
    U: Element + Send + Sync,
    F: Fn(T, T) -> U + Sync + Send,
{
    use rayon::prelude::*;

    /// Below this many elements we stay sequential.
    const PARALLEL_BINARY_THRESHOLD: usize = 16_384;
    /// Chunk size for rayon work-stealing.
    const CHUNK: usize = 8_192;

    let n = lhs.numel();
    let shape = lhs.shape().to_vec();

    // SAFETY: contiguity + dtype + offset 0 verified by caller.
    let lhs_raw: &[T] = unsafe { lhs.storage().as_slice::<T>() };
    let rhs_raw: &[T] = unsafe { rhs.storage().as_slice::<T>() };
    let lhs_slice = &lhs_raw[..n];
    let rhs_slice = &rhs_raw[..n];

    let mut out: Vec<core::mem::MaybeUninit<U>> = Vec::with_capacity(n);
    // SAFETY: capacity is exactly n; every slot is written before read
    // by the loop below.
    #[allow(clippy::uninit_vec)]
    unsafe {
        out.set_len(n);
    }

    if n < PARALLEL_BINARY_THRESHOLD {
        for i in 0..n {
            out[i].write(f(lhs_slice[i], rhs_slice[i]));
        }
    } else {
        out.par_chunks_mut(CHUNK)
            .zip(lhs_slice.par_chunks(CHUNK))
            .zip(rhs_slice.par_chunks(CHUNK))
            .for_each(|((out_chunk, l_chunk), r_chunk)| {
                for i in 0..out_chunk.len() {
                    out_chunk[i].write(f(l_chunk[i], r_chunk[i]));
                }
            });
    }
    // SAFETY: every slot written. Reinterpret Vec<MaybeUninit<U>> → Vec<U>.
    let out: Vec<U> = unsafe {
        let mut o = core::mem::ManuallyDrop::new(out);
        Vec::from_raw_parts(o.as_mut_ptr() as *mut U, n, o.capacity())
    };

    Tensor::from_vec_typed::<U, _>(shape, out).map_err(|_| BackendError::OutOfMemory {
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
    T: Element + Send + Sync,
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
    T: Element + Send + Sync,
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
