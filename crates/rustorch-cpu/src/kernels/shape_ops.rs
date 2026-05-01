//! Shape-manipulation kernels (P1.3 task `Shape ops`).
//!
//! Implements `cat / stack / split / chunk / repeat / flip / roll` over
//! the v1 dtype set (F32 / F64 / I64 / I32 / I8 / Bool). View ops
//! (`view / reshape / transpose / permute / squeeze / unsqueeze /
//! narrow`) live in `rustorch_core::tensor::view` — this module only
//! ships the kernels that need to materialise a fresh contiguous
//! buffer.

use crate::error::BackendError;
use rustorch_core::tensor::dtype::{Dtype, Element};
use rustorch_core::tensor::tensor_impl::Tensor;

// --------------------------------------------------------------------------
// cat — concatenate along `dim`
// --------------------------------------------------------------------------

/// Concatenate `tensors` along axis `dim`. All inputs must agree on
/// dtype and on every dim except `dim`.
pub fn cat(tensors: &[&Tensor], dim: usize) -> Result<Tensor, BackendError> {
    if tensors.is_empty() {
        return Err(BackendError::ShapeMismatch {
            op: "cat",
            lhs: vec![],
            rhs: vec![],
        });
    }
    let first = tensors[0];
    let dtype = first.dtype();
    let ndim = first.ndim();
    if dim >= ndim {
        return Err(BackendError::IndexOutOfBounds {
            op: "cat",
            index: dim as i64,
            bound: ndim,
        });
    }
    // Validate every input.
    for t in tensors.iter().skip(1) {
        if t.dtype() != dtype {
            return Err(BackendError::DtypeMismatch {
                op: "cat",
                lhs: dtype,
                rhs: t.dtype(),
            });
        }
        if t.ndim() != ndim {
            return Err(BackendError::ShapeMismatch {
                op: "cat",
                lhs: first.shape().to_vec(),
                rhs: t.shape().to_vec(),
            });
        }
        for (axis, (&a, &b)) in first.shape().iter().zip(t.shape().iter()).enumerate() {
            if axis != dim && a != b {
                return Err(BackendError::ShapeMismatch {
                    op: "cat",
                    lhs: first.shape().to_vec(),
                    rhs: t.shape().to_vec(),
                });
            }
        }
    }
    let total_dim_size: usize = tensors.iter().map(|t| t.shape()[dim]).sum();
    let mut out_shape = first.shape().to_vec();
    out_shape[dim] = total_dim_size;

    match dtype {
        Dtype::F32 => cat_typed::<f32>(tensors, dim, out_shape),
        Dtype::F64 => cat_typed::<f64>(tensors, dim, out_shape),
        Dtype::I64 => cat_typed::<i64>(tensors, dim, out_shape),
        Dtype::I32 => cat_typed::<i32>(tensors, dim, out_shape),
        Dtype::I8 => cat_typed::<i8>(tensors, dim, out_shape),
        Dtype::Bool => cat_typed::<bool>(tensors, dim, out_shape),
        d => Err(BackendError::DtypeMismatch {
            op: "cat",
            lhs: d,
            rhs: d,
        }),
    }
}

fn cat_typed<T: Element>(
    tensors: &[&Tensor],
    dim: usize,
    out_shape: Vec<usize>,
) -> Result<Tensor, BackendError> {
    let n_out: usize = out_shape.iter().product();
    let mut out = Vec::<T>::with_capacity(n_out);
    let out_stride_inner: usize = out_shape[dim + 1..].iter().product::<usize>().max(1);
    let outer: usize = out_shape[..dim].iter().product::<usize>().max(1);

    // For each outer position, walk every input contributing its dim slice
    // before moving on to the next outer position.
    for outer_idx in 0..outer {
        for t in tensors {
            let dim_size = t.shape()[dim];
            let buf: Vec<T> = t.iter_elements::<T>().expect("dtype").collect();
            let inner: usize = t.shape()[dim + 1..].iter().product::<usize>().max(1);
            let in_offset = outer_idx * dim_size * inner;
            let chunk_len = dim_size * inner;
            out.extend_from_slice(&buf[in_offset..in_offset + chunk_len]);
        }
    }
    let _ = out_stride_inner; // kept for clarity; not used because out is built sequentially
    Tensor::from_vec_typed::<T, _>(out_shape, out).map_err(|_| BackendError::OutOfMemory {
        bytes: n_out * core::mem::size_of::<T>(),
    })
}

// --------------------------------------------------------------------------
// stack — concatenate along a new axis
// --------------------------------------------------------------------------

/// Stack `tensors` along a new axis at position `dim`. Inputs must
/// all share dtype and shape.
pub fn stack(tensors: &[&Tensor], dim: usize) -> Result<Tensor, BackendError> {
    if tensors.is_empty() {
        return Err(BackendError::ShapeMismatch {
            op: "stack",
            lhs: vec![],
            rhs: vec![],
        });
    }
    let first = tensors[0];
    let dtype = first.dtype();
    let shape = first.shape().to_vec();
    if dim > shape.len() {
        return Err(BackendError::IndexOutOfBounds {
            op: "stack",
            index: dim as i64,
            bound: shape.len() + 1,
        });
    }
    for t in tensors.iter().skip(1) {
        if t.dtype() != dtype {
            return Err(BackendError::DtypeMismatch {
                op: "stack",
                lhs: dtype,
                rhs: t.dtype(),
            });
        }
        if t.shape() != shape.as_slice() {
            return Err(BackendError::ShapeMismatch {
                op: "stack",
                lhs: shape.clone(),
                rhs: t.shape().to_vec(),
            });
        }
    }
    // Each input is "unsqueezed" along `dim`, then we cat them along that dim.
    // Equivalently: view each input as shape `[..., 1, ...]` inserted at dim.
    let mut new_shape = shape.clone();
    new_shape.insert(dim, 1);

    // Materialise each input contiguously then build the cat output.
    let unsqueezed: Vec<Tensor> = tensors
        .iter()
        .map(|t| {
            t.contiguous()
                .view(new_shape.clone())
                .expect("compatible shape")
        })
        .collect();
    let unsqueezed_refs: Vec<&Tensor> = unsqueezed.iter().collect();
    cat(&unsqueezed_refs, dim)
}

// --------------------------------------------------------------------------
// split / chunk — break into pieces along `dim`
// --------------------------------------------------------------------------

/// Split a tensor into pieces of `split_size` along `dim`. Last piece
/// may be smaller if `dim_size % split_size != 0`.
pub fn split(src: &Tensor, split_size: usize, dim: usize) -> Result<Vec<Tensor>, BackendError> {
    if dim >= src.ndim() {
        return Err(BackendError::IndexOutOfBounds {
            op: "split",
            index: dim as i64,
            bound: src.ndim(),
        });
    }
    if split_size == 0 {
        return Err(BackendError::ShapeMismatch {
            op: "split",
            lhs: src.shape().to_vec(),
            rhs: vec![split_size],
        });
    }
    let dim_size = src.shape()[dim];
    let mut pieces = Vec::new();
    let mut start = 0usize;
    while start < dim_size {
        let len = (dim_size - start).min(split_size);
        let piece = src
            .narrow(dim, start, len)
            .map_err(|_| BackendError::IndexOutOfBounds {
                op: "split",
                index: start as i64,
                bound: dim_size,
            })?;
        pieces.push(piece.contiguous());
        start += len;
    }
    Ok(pieces)
}

/// Split into approximately equal `n_chunks` along `dim`.
pub fn chunk(src: &Tensor, n_chunks: usize, dim: usize) -> Result<Vec<Tensor>, BackendError> {
    if n_chunks == 0 {
        return Err(BackendError::ShapeMismatch {
            op: "chunk",
            lhs: src.shape().to_vec(),
            rhs: vec![n_chunks],
        });
    }
    if dim >= src.ndim() {
        return Err(BackendError::IndexOutOfBounds {
            op: "chunk",
            index: dim as i64,
            bound: src.ndim(),
        });
    }
    let dim_size = src.shape()[dim];
    // PyTorch's chunk: split_size = ceil(dim_size / n_chunks); last chunk
    // may be smaller (or absent) if dim_size doesn't divide evenly.
    let split_size = dim_size.div_ceil(n_chunks).max(1);
    split(src, split_size, dim)
}

// --------------------------------------------------------------------------
// repeat — replicate along axes
// --------------------------------------------------------------------------

/// Repeat the tensor along the given `repeats` factors. `repeats.len()`
/// must equal `src.ndim()`. Each axis size is multiplied by the
/// corresponding `repeats[i]`.
pub fn repeat(src: &Tensor, repeats: &[usize]) -> Result<Tensor, BackendError> {
    if repeats.len() != src.ndim() {
        return Err(BackendError::ShapeMismatch {
            op: "repeat",
            lhs: src.shape().to_vec(),
            rhs: repeats.to_vec(),
        });
    }
    let new_shape: Vec<usize> = src
        .shape()
        .iter()
        .zip(repeats.iter())
        .map(|(d, r)| d * r)
        .collect();

    match src.dtype() {
        Dtype::F32 => repeat_typed::<f32>(src, repeats, new_shape),
        Dtype::F64 => repeat_typed::<f64>(src, repeats, new_shape),
        Dtype::I64 => repeat_typed::<i64>(src, repeats, new_shape),
        Dtype::I32 => repeat_typed::<i32>(src, repeats, new_shape),
        Dtype::I8 => repeat_typed::<i8>(src, repeats, new_shape),
        Dtype::Bool => repeat_typed::<bool>(src, repeats, new_shape),
        d => Err(BackendError::DtypeMismatch {
            op: "repeat",
            lhs: d,
            rhs: d,
        }),
    }
}

fn repeat_typed<T: Element>(
    src: &Tensor,
    repeats: &[usize],
    new_shape: Vec<usize>,
) -> Result<Tensor, BackendError> {
    let src_buf: Vec<T> = src.iter_elements::<T>().expect("dtype").collect();
    let src_shape = src.shape();
    let n_out: usize = new_shape.iter().product();
    let mut out: Vec<T> = Vec::with_capacity(n_out);
    let out_strides = contiguous_strides(&new_shape);
    let src_strides = contiguous_strides(src_shape);
    for linear in 0..n_out {
        let coords = decode_coords(linear, &new_shape);
        // Map output coord back to input coord by mod with src dim.
        let src_linear: usize = coords
            .iter()
            .zip(src_shape.iter())
            .zip(src_strides.iter())
            .map(|((c, &d), s)| (c % d) * s)
            .sum();
        out.push(src_buf[src_linear]);
    }
    let _ = repeats; // not directly used (encoded in new_shape)
    let _ = out_strides;
    Tensor::from_vec_typed::<T, _>(new_shape, out).map_err(|_| BackendError::OutOfMemory {
        bytes: n_out * core::mem::size_of::<T>(),
    })
}

// --------------------------------------------------------------------------
// flip — reverse along given axes
// --------------------------------------------------------------------------

/// Reverse the tensor along every axis listed in `dims`.
pub fn flip(src: &Tensor, dims: &[usize]) -> Result<Tensor, BackendError> {
    for &d in dims {
        if d >= src.ndim() {
            return Err(BackendError::IndexOutOfBounds {
                op: "flip",
                index: d as i64,
                bound: src.ndim(),
            });
        }
    }
    match src.dtype() {
        Dtype::F32 => flip_typed::<f32>(src, dims),
        Dtype::F64 => flip_typed::<f64>(src, dims),
        Dtype::I64 => flip_typed::<i64>(src, dims),
        Dtype::I32 => flip_typed::<i32>(src, dims),
        Dtype::I8 => flip_typed::<i8>(src, dims),
        Dtype::Bool => flip_typed::<bool>(src, dims),
        d => Err(BackendError::DtypeMismatch {
            op: "flip",
            lhs: d,
            rhs: d,
        }),
    }
}

fn flip_typed<T: Element>(src: &Tensor, dims: &[usize]) -> Result<Tensor, BackendError> {
    let buf: Vec<T> = src.iter_elements::<T>().expect("dtype").collect();
    let shape = src.shape();
    let n = src.numel();
    let strides = contiguous_strides(shape);
    let mut out: Vec<T> = Vec::with_capacity(n);
    for linear in 0..n {
        let mut coords = decode_coords(linear, shape);
        for &d in dims {
            coords[d] = shape[d] - 1 - coords[d];
        }
        let src_linear: usize = coords.iter().zip(strides.iter()).map(|(c, s)| c * s).sum();
        out.push(buf[src_linear]);
    }
    Tensor::from_vec_typed::<T, _>(shape.to_vec(), out).map_err(|_| BackendError::OutOfMemory {
        bytes: n * core::mem::size_of::<T>(),
    })
}

// --------------------------------------------------------------------------
// roll — circular shift along an axis
// --------------------------------------------------------------------------

/// Circular shift along `dim` by `shifts` positions. Positive `shifts`
/// move values toward higher indices (wrapping the trailing values to
/// the front); negative shifts the other way.
pub fn roll(src: &Tensor, shifts: i64, dim: usize) -> Result<Tensor, BackendError> {
    if dim >= src.ndim() {
        return Err(BackendError::IndexOutOfBounds {
            op: "roll",
            index: dim as i64,
            bound: src.ndim(),
        });
    }
    let n = src.shape()[dim] as i64;
    if n == 0 {
        return Ok(src.contiguous());
    }
    let s = ((shifts % n) + n) % n; // normalise to [0, n)
    match src.dtype() {
        Dtype::F32 => roll_typed::<f32>(src, s as usize, dim),
        Dtype::F64 => roll_typed::<f64>(src, s as usize, dim),
        Dtype::I64 => roll_typed::<i64>(src, s as usize, dim),
        Dtype::I32 => roll_typed::<i32>(src, s as usize, dim),
        Dtype::I8 => roll_typed::<i8>(src, s as usize, dim),
        Dtype::Bool => roll_typed::<bool>(src, s as usize, dim),
        d => Err(BackendError::DtypeMismatch {
            op: "roll",
            lhs: d,
            rhs: d,
        }),
    }
}

fn roll_typed<T: Element>(src: &Tensor, s: usize, dim: usize) -> Result<Tensor, BackendError> {
    let buf: Vec<T> = src.iter_elements::<T>().expect("dtype").collect();
    let shape = src.shape();
    let n = src.numel();
    let strides = contiguous_strides(shape);
    let dim_size = shape[dim];
    let mut out: Vec<T> = Vec::with_capacity(n);
    for linear in 0..n {
        let mut coords = decode_coords(linear, shape);
        coords[dim] = (coords[dim] + dim_size - s) % dim_size;
        let src_linear: usize = coords.iter().zip(strides.iter()).map(|(c, s)| c * s).sum();
        out.push(buf[src_linear]);
    }
    Tensor::from_vec_typed::<T, _>(shape.to_vec(), out).map_err(|_| BackendError::OutOfMemory {
        bytes: n * core::mem::size_of::<T>(),
    })
}

// --------------------------------------------------------------------------
// helpers (duplicated from cpu_backend.rs to keep this module self-contained)
// --------------------------------------------------------------------------

fn contiguous_strides(shape: &[usize]) -> Vec<usize> {
    if shape.is_empty() {
        return vec![];
    }
    let mut s = vec![1usize; shape.len()];
    for i in (0..shape.len() - 1).rev() {
        s[i] = s[i + 1] * shape[i + 1];
    }
    s
}

fn decode_coords(linear: usize, shape: &[usize]) -> Vec<usize> {
    if shape.is_empty() {
        return vec![];
    }
    let mut idx = linear;
    let mut coords = vec![0usize; shape.len()];
    for axis in (0..shape.len()).rev() {
        coords[axis] = idx % shape[axis];
        idx /= shape[axis];
    }
    coords
}
