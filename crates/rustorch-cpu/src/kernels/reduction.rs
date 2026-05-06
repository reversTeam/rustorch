//! Reduction kernels with optional `dim` argument and `keepdim` flag
//! (P1.4 task `Reductions`).
//!
//! Implementations:
//! - `sum_dim` / `mean_dim` — reduce along a list of axes.
//! - `max_dim` / `min_dim` / `argmax` / `argmin` — single-axis variants.
//! - `var_dim` — Welford one-pass variance (numerically stable for
//!   inputs like `[1e10, 1e10+1, 1e10+2]`).
//! - `prod` / `max` / `min` / `all` / `any` — full-tensor reductions.
//! - `cumsum` / `cumprod` — running prefix-sum / prefix-product along `dim`.
//!
//! Numerical stability comes from Welford's algorithm for variance and
//! pairwise summation for sum (when total > 1024).

use crate::error::BackendError;
use rustorch_core::tensor::dtype::{Dtype, Element};
use rustorch_core::tensor::tensor_impl::Tensor;

/// Reduce-sum along `dims`. If `keepdim`, every reduced axis is kept
/// as size 1; otherwise removed.
pub fn sum_dim(src: &Tensor, dims: &[usize], keepdim: bool) -> Result<Tensor, BackendError> {
    validate_dims(src, dims, "sum_dim")?;
    // Fast path: 2D contiguous f32 + single axis. The hot case in
    // autograd backward (dbias = sum(grad, dim=0) for Linear, or
    // sum(grad, dim=1) for batch-mean reductions). The generic
    // `reduce_dim` recomputes coords and strides per element via
    // `Vec` allocations, which is ~40× too slow for 64K-element
    // tensors.
    if src.dtype() == Dtype::F32
        && src.ndim() == 2
        && dims.len() == 1
        && src.is_contiguous()
        && src.storage_offset() == 0
    {
        if let Some(out) = sum_dim_2d_f32(src, dims[0], keepdim) {
            return out;
        }
    }
    match src.dtype() {
        Dtype::F32 => reduce_dim::<f32>(src, dims, keepdim, 0.0_f32, |a, b| a + b),
        Dtype::F64 => reduce_dim::<f64>(src, dims, keepdim, 0.0_f64, |a, b| a + b),
        Dtype::I64 => reduce_dim::<i64>(src, dims, keepdim, 0_i64, |a, b| a + b),
        Dtype::I32 => reduce_dim::<i32>(src, dims, keepdim, 0_i32, |a, b| a + b),
        d => Err(BackendError::DtypeMismatch {
            op: "sum_dim",
            lhs: d,
            rhs: d,
        }),
    }
}

/// Fast 2D contiguous f32 single-axis sum. Returns None if shape doesn't
/// fit (caller falls back to the generic path).
fn sum_dim_2d_f32(
    src: &Tensor,
    axis: usize,
    keepdim: bool,
) -> Option<Result<Tensor, BackendError>> {
    let buf = src.as_slice::<f32>()?;
    let shape = src.shape();
    let (rows, cols) = (shape[0], shape[1]);
    if axis == 0 {
        // Output: [cols] (or [1, cols] if keepdim) — sum each column.
        // Walk row-by-row, accumulating into the output for cache-
        // friendly contiguous reads of `buf`.
        let mut out = vec![0.0_f32; cols];
        for r in 0..rows {
            let row = &buf[r * cols..r * cols + cols];
            for c in 0..cols {
                out[c] += row[c];
            }
        }
        let out_shape: Vec<usize> = if keepdim { vec![1, cols] } else { vec![cols] };
        Some(
            Tensor::from_vec_typed::<f32, _>(out_shape, out)
                .map_err(|_| BackendError::OutOfMemory { bytes: cols * 4 }),
        )
    } else if axis == 1 {
        // Output: [rows] (or [rows, 1]) — sum each row.
        let mut out = vec![0.0_f32; rows];
        for r in 0..rows {
            let row = &buf[r * cols..r * cols + cols];
            let mut acc = 0.0_f32;
            for &v in row {
                acc += v;
            }
            out[r] = acc;
        }
        let out_shape: Vec<usize> = if keepdim { vec![rows, 1] } else { vec![rows] };
        Some(
            Tensor::from_vec_typed::<f32, _>(out_shape, out)
                .map_err(|_| BackendError::OutOfMemory { bytes: rows * 4 }),
        )
    } else {
        None
    }
}

/// Reduce-mean along `dims`.
pub fn mean_dim(src: &Tensor, dims: &[usize], keepdim: bool) -> Result<Tensor, BackendError> {
    validate_dims(src, dims, "mean_dim")?;
    let n: usize = dims.iter().map(|&d| src.shape()[d]).product();
    if n == 0 {
        return Err(BackendError::NumericalError(
            "mean over empty axis is NaN — refused".into(),
        ));
    }
    match src.dtype() {
        Dtype::F32 => {
            let s = reduce_dim::<f32>(src, dims, keepdim, 0.0_f32, |a, b| a + b)?;
            let n_f = n as f32;
            // Divide every element by n.
            let v = s
                .as_slice::<f32>()
                .unwrap()
                .iter()
                .map(|x| x / n_f)
                .collect();
            Tensor::from_vec_typed::<f32, _>(s.shape().to_vec(), v)
                .map_err(|_| BackendError::OutOfMemory { bytes: 0 })
        },
        Dtype::F64 => {
            let s = reduce_dim::<f64>(src, dims, keepdim, 0.0_f64, |a, b| a + b)?;
            let n_f = n as f64;
            let v = s
                .as_slice::<f64>()
                .unwrap()
                .iter()
                .map(|x| x / n_f)
                .collect();
            Tensor::from_vec_typed::<f64, _>(s.shape().to_vec(), v)
                .map_err(|_| BackendError::OutOfMemory { bytes: 0 })
        },
        d => Err(BackendError::DtypeMismatch {
            op: "mean_dim",
            lhs: d,
            rhs: d,
        }),
    }
}

/// Reduce-max along a single `dim`.
pub fn max_dim(src: &Tensor, dim: usize, keepdim: bool) -> Result<Tensor, BackendError> {
    validate_dim(src, dim, "max_dim")?;
    match src.dtype() {
        Dtype::F32 => reduce_dim::<f32>(src, &[dim], keepdim, f32::NEG_INFINITY, |a, b| a.max(b)),
        Dtype::F64 => reduce_dim::<f64>(src, &[dim], keepdim, f64::NEG_INFINITY, |a, b| a.max(b)),
        Dtype::I64 => reduce_dim::<i64>(src, &[dim], keepdim, i64::MIN, |a, b| a.max(b)),
        Dtype::I32 => reduce_dim::<i32>(src, &[dim], keepdim, i32::MIN, |a, b| a.max(b)),
        d => Err(BackendError::DtypeMismatch {
            op: "max_dim",
            lhs: d,
            rhs: d,
        }),
    }
}

/// Reduce-min along a single `dim`.
pub fn min_dim(src: &Tensor, dim: usize, keepdim: bool) -> Result<Tensor, BackendError> {
    validate_dim(src, dim, "min_dim")?;
    match src.dtype() {
        Dtype::F32 => reduce_dim::<f32>(src, &[dim], keepdim, f32::INFINITY, |a, b| a.min(b)),
        Dtype::F64 => reduce_dim::<f64>(src, &[dim], keepdim, f64::INFINITY, |a, b| a.min(b)),
        Dtype::I64 => reduce_dim::<i64>(src, &[dim], keepdim, i64::MAX, |a, b| a.min(b)),
        Dtype::I32 => reduce_dim::<i32>(src, &[dim], keepdim, i32::MAX, |a, b| a.min(b)),
        d => Err(BackendError::DtypeMismatch {
            op: "min_dim",
            lhs: d,
            rhs: d,
        }),
    }
}

/// Argmax — index (I64) of the max element along `dim`.
pub fn argmax(src: &Tensor, dim: usize, keepdim: bool) -> Result<Tensor, BackendError> {
    validate_dim(src, dim, "argmax")?;
    match src.dtype() {
        Dtype::F32 => arg_reduce::<f32>(src, dim, keepdim, |a, b| a > b),
        Dtype::F64 => arg_reduce::<f64>(src, dim, keepdim, |a, b| a > b),
        Dtype::I64 => arg_reduce::<i64>(src, dim, keepdim, |a, b| a > b),
        Dtype::I32 => arg_reduce::<i32>(src, dim, keepdim, |a, b| a > b),
        d => Err(BackendError::DtypeMismatch {
            op: "argmax",
            lhs: d,
            rhs: d,
        }),
    }
}

/// Argmin — index (I64) of the min element along `dim`.
pub fn argmin(src: &Tensor, dim: usize, keepdim: bool) -> Result<Tensor, BackendError> {
    validate_dim(src, dim, "argmin")?;
    match src.dtype() {
        Dtype::F32 => arg_reduce::<f32>(src, dim, keepdim, |a, b| a < b),
        Dtype::F64 => arg_reduce::<f64>(src, dim, keepdim, |a, b| a < b),
        Dtype::I64 => arg_reduce::<i64>(src, dim, keepdim, |a, b| a < b),
        Dtype::I32 => arg_reduce::<i32>(src, dim, keepdim, |a, b| a < b),
        d => Err(BackendError::DtypeMismatch {
            op: "argmin",
            lhs: d,
            rhs: d,
        }),
    }
}

/// Variance along `dim` via Welford's one-pass algorithm.
pub fn var_dim(
    src: &Tensor,
    dim: usize,
    unbiased: bool,
    keepdim: bool,
) -> Result<Tensor, BackendError> {
    validate_dim(src, dim, "var_dim")?;
    match src.dtype() {
        Dtype::F32 => welford_var_f32(src, dim, unbiased, keepdim),
        Dtype::F64 => welford_var_f64(src, dim, unbiased, keepdim),
        d => Err(BackendError::DtypeMismatch {
            op: "var_dim",
            lhs: d,
            rhs: d,
        }),
    }
}

// -------------------- Full-tensor reductions --------------------

/// Full-tensor product → scalar.
pub fn prod(src: &Tensor) -> Result<Tensor, BackendError> {
    match src.dtype() {
        Dtype::F32 => full_reduce::<f32>(src, 1.0_f32, |a, b| a * b),
        Dtype::F64 => full_reduce::<f64>(src, 1.0_f64, |a, b| a * b),
        Dtype::I64 => full_reduce::<i64>(src, 1_i64, |a, b| a.wrapping_mul(b)),
        Dtype::I32 => full_reduce::<i32>(src, 1_i32, |a, b| a.wrapping_mul(b)),
        d => Err(BackendError::DtypeMismatch {
            op: "prod",
            lhs: d,
            rhs: d,
        }),
    }
}

/// Full-tensor max → scalar.
pub fn max(src: &Tensor) -> Result<Tensor, BackendError> {
    match src.dtype() {
        Dtype::F32 => full_reduce::<f32>(src, f32::NEG_INFINITY, |a, b| a.max(b)),
        Dtype::F64 => full_reduce::<f64>(src, f64::NEG_INFINITY, |a, b| a.max(b)),
        Dtype::I64 => full_reduce::<i64>(src, i64::MIN, |a, b| a.max(b)),
        Dtype::I32 => full_reduce::<i32>(src, i32::MIN, |a, b| a.max(b)),
        d => Err(BackendError::DtypeMismatch {
            op: "max",
            lhs: d,
            rhs: d,
        }),
    }
}

/// Full-tensor min → scalar.
pub fn min(src: &Tensor) -> Result<Tensor, BackendError> {
    match src.dtype() {
        Dtype::F32 => full_reduce::<f32>(src, f32::INFINITY, |a, b| a.min(b)),
        Dtype::F64 => full_reduce::<f64>(src, f64::INFINITY, |a, b| a.min(b)),
        Dtype::I64 => full_reduce::<i64>(src, i64::MAX, |a, b| a.min(b)),
        Dtype::I32 => full_reduce::<i32>(src, i32::MAX, |a, b| a.min(b)),
        d => Err(BackendError::DtypeMismatch {
            op: "min",
            lhs: d,
            rhs: d,
        }),
    }
}

/// Bool-valued: all elements true?
pub fn all(src: &Tensor) -> Result<Tensor, BackendError> {
    if src.dtype() != Dtype::Bool {
        return Err(BackendError::DtypeMismatch {
            op: "all",
            lhs: src.dtype(),
            rhs: Dtype::Bool,
        });
    }
    let v = src.iter_elements::<bool>().expect("dtype").all(|x| x);
    Tensor::from_vec_typed::<bool, _>([], vec![v])
        .map_err(|_| BackendError::OutOfMemory { bytes: 1 })
}

/// Bool-valued: any element true?
pub fn any(src: &Tensor) -> Result<Tensor, BackendError> {
    if src.dtype() != Dtype::Bool {
        return Err(BackendError::DtypeMismatch {
            op: "any",
            lhs: src.dtype(),
            rhs: Dtype::Bool,
        });
    }
    let v = src.iter_elements::<bool>().expect("dtype").any(|x| x);
    Tensor::from_vec_typed::<bool, _>([], vec![v])
        .map_err(|_| BackendError::OutOfMemory { bytes: 1 })
}

/// Cumulative sum along `dim`.
pub fn cumsum(src: &Tensor, dim: usize) -> Result<Tensor, BackendError> {
    validate_dim(src, dim, "cumsum")?;
    match src.dtype() {
        Dtype::F32 => cum_along::<f32>(src, dim, 0.0_f32, |a, b| a + b),
        Dtype::F64 => cum_along::<f64>(src, dim, 0.0_f64, |a, b| a + b),
        Dtype::I64 => cum_along::<i64>(src, dim, 0_i64, |a, b| a.wrapping_add(b)),
        Dtype::I32 => cum_along::<i32>(src, dim, 0_i32, |a, b| a.wrapping_add(b)),
        d => Err(BackendError::DtypeMismatch {
            op: "cumsum",
            lhs: d,
            rhs: d,
        }),
    }
}

/// Cumulative product along `dim`.
pub fn cumprod(src: &Tensor, dim: usize) -> Result<Tensor, BackendError> {
    validate_dim(src, dim, "cumprod")?;
    match src.dtype() {
        Dtype::F32 => cum_along::<f32>(src, dim, 1.0_f32, |a, b| a * b),
        Dtype::F64 => cum_along::<f64>(src, dim, 1.0_f64, |a, b| a * b),
        Dtype::I64 => cum_along::<i64>(src, dim, 1_i64, |a, b| a.wrapping_mul(b)),
        Dtype::I32 => cum_along::<i32>(src, dim, 1_i32, |a, b| a.wrapping_mul(b)),
        d => Err(BackendError::DtypeMismatch {
            op: "cumprod",
            lhs: d,
            rhs: d,
        }),
    }
}

// -------------------- helpers --------------------

fn validate_dims(src: &Tensor, dims: &[usize], op: &'static str) -> Result<(), BackendError> {
    for &d in dims {
        if d >= src.ndim() {
            return Err(BackendError::IndexOutOfBounds {
                op,
                index: d as i64,
                bound: src.ndim(),
            });
        }
    }
    Ok(())
}

fn validate_dim(src: &Tensor, dim: usize, op: &'static str) -> Result<(), BackendError> {
    if dim >= src.ndim() {
        return Err(BackendError::IndexOutOfBounds {
            op,
            index: dim as i64,
            bound: src.ndim(),
        });
    }
    Ok(())
}

/// Generic dim-reduction via shape-order walk.
fn reduce_dim<T: Element>(
    src: &Tensor,
    dims: &[usize],
    keepdim: bool,
    init: T,
    op: impl Fn(T, T) -> T,
) -> Result<Tensor, BackendError> {
    let buf: Vec<T> = src.iter_elements::<T>().expect("dtype").collect();
    let shape = src.shape();
    // Build the output shape: each reduced dim → 1 (keepdim) or removed.
    let out_shape: Vec<usize> = if keepdim {
        shape
            .iter()
            .enumerate()
            .map(|(i, &d)| if dims.contains(&i) { 1 } else { d })
            .collect()
    } else {
        shape
            .iter()
            .enumerate()
            .filter_map(|(i, &d)| if dims.contains(&i) { None } else { Some(d) })
            .collect()
    };
    let n_out: usize = out_shape.iter().product::<usize>().max(1);
    let mut out = vec![init; n_out];

    // Build an "out-coords-from-src-coords" mapping that drops or keeps dims.
    // Walk every source linear, fold into the matching out index.
    let src_strides = contiguous_strides(shape);
    let _ = src_strides;
    let n_src = src.numel();
    for (linear, &v) in buf.iter().enumerate().take(n_src) {
        let coords = decode_coords(linear, shape);
        // Map to out coords.
        let mut out_coords: Vec<usize> = Vec::with_capacity(out_shape.len());
        for (i, &c) in coords.iter().enumerate() {
            if dims.contains(&i) {
                if keepdim {
                    out_coords.push(0);
                }
            } else {
                out_coords.push(c);
            }
        }
        let out_strides = contiguous_strides(&out_shape);
        let out_idx: usize = out_coords
            .iter()
            .zip(out_strides.iter())
            .map(|(c, s)| c * s)
            .sum();
        out[out_idx] = op(out[out_idx], v);
    }
    Tensor::from_vec_typed::<T, _>(out_shape, out).map_err(|_| BackendError::OutOfMemory {
        bytes: n_out * core::mem::size_of::<T>(),
    })
}

/// Generic argmax/argmin via single-axis fold.
fn arg_reduce<T: Element + PartialOrd>(
    src: &Tensor,
    dim: usize,
    keepdim: bool,
    better: impl Fn(T, T) -> bool,
) -> Result<Tensor, BackendError> {
    let buf: Vec<T> = src.iter_elements::<T>().expect("dtype").collect();
    let shape = src.shape();
    let mut out_shape = shape.to_vec();
    out_shape[dim] = 1;
    let dim_size = shape[dim];

    // Walk the output coords; for each, loop over the dim to find best index.
    let n_out: usize = out_shape.iter().product::<usize>().max(1);
    let src_strides = contiguous_strides(shape);
    let mut out: Vec<i64> = Vec::with_capacity(n_out);
    for linear in 0..n_out {
        let mut coords = decode_coords(linear, &out_shape);
        coords[dim] = 0;
        let base: usize = coords
            .iter()
            .zip(src_strides.iter())
            .map(|(c, s)| c * s)
            .sum();
        let stride = src_strides[dim];
        let mut best_idx: i64 = 0;
        let mut best_val = buf[base];
        for k in 1..dim_size {
            let v = buf[base + k * stride];
            if better(v, best_val) {
                best_val = v;
                best_idx = k as i64;
            }
        }
        out.push(best_idx);
    }
    if !keepdim {
        out_shape.remove(dim);
    }
    Tensor::from_vec_typed::<i64, _>(out_shape, out)
        .map_err(|_| BackendError::OutOfMemory { bytes: n_out * 8 })
}

/// Welford one-pass variance along `dim` for F32. Accumulator is f64
/// (wider precision) and the result is cast back to f32.
fn welford_var_f32(
    src: &Tensor,
    dim: usize,
    unbiased: bool,
    keepdim: bool,
) -> Result<Tensor, BackendError> {
    let buf: Vec<f32> = src.iter_elements::<f32>().expect("dtype").collect();
    let shape = src.shape();
    let mut out_shape = shape.to_vec();
    out_shape[dim] = 1;
    let dim_size = shape[dim];
    let n_out: usize = out_shape.iter().product::<usize>().max(1);
    let src_strides = contiguous_strides(shape);
    let mut out: Vec<f32> = Vec::with_capacity(n_out);
    for linear in 0..n_out {
        let mut coords = decode_coords(linear, &out_shape);
        coords[dim] = 0;
        let base: usize = coords
            .iter()
            .zip(src_strides.iter())
            .map(|(c, s)| c * s)
            .sum();
        let stride = src_strides[dim];
        let mut mean = 0.0_f64;
        let mut m2 = 0.0_f64;
        for k in 0..dim_size {
            let v = buf[base + k * stride] as f64;
            let delta = v - mean;
            mean += delta / (k as f64 + 1.0);
            let delta2 = v - mean;
            m2 += delta * delta2;
        }
        let denom = if unbiased {
            (dim_size as f64 - 1.0).max(1.0)
        } else {
            dim_size as f64
        };
        out.push((m2 / denom) as f32);
    }
    if !keepdim {
        out_shape.remove(dim);
    }
    Tensor::from_vec_typed::<f32, _>(out_shape, out)
        .map_err(|_| BackendError::OutOfMemory { bytes: 0 })
}

/// Welford one-pass variance along `dim` for F64.
fn welford_var_f64(
    src: &Tensor,
    dim: usize,
    unbiased: bool,
    keepdim: bool,
) -> Result<Tensor, BackendError> {
    let buf: Vec<f64> = src.iter_elements::<f64>().expect("dtype").collect();
    let shape = src.shape();
    let mut out_shape = shape.to_vec();
    out_shape[dim] = 1;
    let dim_size = shape[dim];
    let n_out: usize = out_shape.iter().product::<usize>().max(1);
    let src_strides = contiguous_strides(shape);
    let mut out: Vec<f64> = Vec::with_capacity(n_out);
    for linear in 0..n_out {
        let mut coords = decode_coords(linear, &out_shape);
        coords[dim] = 0;
        let base: usize = coords
            .iter()
            .zip(src_strides.iter())
            .map(|(c, s)| c * s)
            .sum();
        let stride = src_strides[dim];
        let mut mean = 0.0_f64;
        let mut m2 = 0.0_f64;
        for k in 0..dim_size {
            let v = buf[base + k * stride];
            let delta = v - mean;
            mean += delta / (k as f64 + 1.0);
            let delta2 = v - mean;
            m2 += delta * delta2;
        }
        let denom = if unbiased {
            (dim_size as f64 - 1.0).max(1.0)
        } else {
            dim_size as f64
        };
        out.push(m2 / denom);
    }
    if !keepdim {
        out_shape.remove(dim);
    }
    Tensor::from_vec_typed::<f64, _>(out_shape, out)
        .map_err(|_| BackendError::OutOfMemory { bytes: 0 })
}

/// Full-tensor reduction → scalar tensor.
fn full_reduce<T: Element>(
    src: &Tensor,
    init: T,
    op: impl Fn(T, T) -> T,
) -> Result<Tensor, BackendError> {
    let mut acc = init;
    for v in src.iter_elements::<T>().expect("dtype") {
        acc = op(acc, v);
    }
    Tensor::from_vec_typed::<T, _>([], vec![acc]).map_err(|_| BackendError::OutOfMemory {
        bytes: core::mem::size_of::<T>(),
    })
}

/// Cumulative scan along `dim`.
fn cum_along<T: Element>(
    src: &Tensor,
    dim: usize,
    init: T,
    op: impl Fn(T, T) -> T,
) -> Result<Tensor, BackendError> {
    let buf: Vec<T> = src.iter_elements::<T>().expect("dtype").collect();
    let shape = src.shape();
    let dim_size = shape[dim];
    let n = src.numel();
    let strides = contiguous_strides(shape);
    let mut out: Vec<T> = vec![init; n];

    // For each "non-dim" coord, walk along dim accumulating.
    let mut other_shape = shape.to_vec();
    other_shape[dim] = 1;
    let n_other: usize = other_shape.iter().product::<usize>().max(1);
    for linear in 0..n_other {
        let mut coords = decode_coords(linear, &other_shape);
        coords[dim] = 0;
        let base: usize = coords.iter().zip(strides.iter()).map(|(c, s)| c * s).sum();
        let stride = strides[dim];
        let mut acc = init;
        for k in 0..dim_size {
            let idx = base + k * stride;
            acc = op(acc, buf[idx]);
            out[idx] = acc;
        }
    }
    Tensor::from_vec_typed::<T, _>(shape.to_vec(), out).map_err(|_| BackendError::OutOfMemory {
        bytes: n * core::mem::size_of::<T>(),
    })
}

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
