//! Numerically stable softmax / log_softmax (P1.4 task `Softmax`).
//!
//! Both ops use the max-subtraction trick to avoid overflow on extreme
//! inputs:
//!
//! ```text
//! softmax(x)_i     = exp(x_i - max(x)) / sum_j exp(x_j - max(x))
//! log_softmax(x)_i = (x_i - max(x)) - log(sum_j exp(x_j - max(x)))
//! ```

use crate::error::BackendError;
use rustorch_core::tensor::dtype::Dtype;
use rustorch_core::tensor::tensor_impl::Tensor;

/// Numerically stable softmax along `dim`.
pub fn softmax(src: &Tensor, dim: usize) -> Result<Tensor, BackendError> {
    if dim >= src.ndim() {
        return Err(BackendError::IndexOutOfBounds {
            op: "softmax",
            index: dim as i64,
            bound: src.ndim(),
        });
    }
    match src.dtype() {
        Dtype::F32 => softmax_f32(src, dim, false),
        Dtype::F64 => softmax_f64(src, dim, false),
        d => Err(BackendError::DtypeMismatch {
            op: "softmax",
            lhs: d,
            rhs: d,
        }),
    }
}

/// Numerically stable log_softmax along `dim`.
pub fn log_softmax(src: &Tensor, dim: usize) -> Result<Tensor, BackendError> {
    if dim >= src.ndim() {
        return Err(BackendError::IndexOutOfBounds {
            op: "log_softmax",
            index: dim as i64,
            bound: src.ndim(),
        });
    }
    match src.dtype() {
        Dtype::F32 => softmax_f32(src, dim, true),
        Dtype::F64 => softmax_f64(src, dim, true),
        d => Err(BackendError::DtypeMismatch {
            op: "log_softmax",
            lhs: d,
            rhs: d,
        }),
    }
}

fn softmax_f32(src: &Tensor, dim: usize, log_form: bool) -> Result<Tensor, BackendError> {
    let buf: Vec<f32> = src.iter_elements::<f32>().expect("dtype").collect();
    let shape = src.shape();
    let strides = contiguous_strides(shape);
    let dim_size = shape[dim];
    let mut out = vec![0.0_f32; src.numel()];
    let n_other: usize = src.numel() / dim_size.max(1);

    // Walk every "non-dim" position (encoded by removing `dim` from shape).
    let mut other_shape = shape.to_vec();
    other_shape[dim] = 1;
    for linear in 0..n_other {
        let mut coords = decode_coords(linear, &other_shape);
        coords[dim] = 0;
        let base: usize = coords.iter().zip(strides.iter()).map(|(c, s)| c * s).sum();
        let stride = strides[dim];
        // Pass 1: max
        let mut mx = f32::NEG_INFINITY;
        for k in 0..dim_size {
            let v = buf[base + k * stride];
            if v > mx {
                mx = v;
            }
        }
        // Pass 2: sum of exp(x - max)
        let mut sumexp = 0.0_f32;
        for k in 0..dim_size {
            sumexp += (buf[base + k * stride] - mx).exp();
        }
        let log_sumexp = sumexp.ln();
        // Pass 3: write softmax / log_softmax
        for k in 0..dim_size {
            let v = buf[base + k * stride];
            let shifted = v - mx;
            out[base + k * stride] = if log_form {
                shifted - log_sumexp
            } else {
                shifted.exp() / sumexp
            };
        }
    }
    Tensor::from_vec_typed::<f32, _>(shape.to_vec(), out)
        .map_err(|_| BackendError::OutOfMemory { bytes: 0 })
}

fn softmax_f64(src: &Tensor, dim: usize, log_form: bool) -> Result<Tensor, BackendError> {
    let buf: Vec<f64> = src.iter_elements::<f64>().expect("dtype").collect();
    let shape = src.shape();
    let strides = contiguous_strides(shape);
    let dim_size = shape[dim];
    let mut out = vec![0.0_f64; src.numel()];
    let n_other: usize = src.numel() / dim_size.max(1);
    let mut other_shape = shape.to_vec();
    other_shape[dim] = 1;
    for linear in 0..n_other {
        let mut coords = decode_coords(linear, &other_shape);
        coords[dim] = 0;
        let base: usize = coords.iter().zip(strides.iter()).map(|(c, s)| c * s).sum();
        let stride = strides[dim];
        let mut mx = f64::NEG_INFINITY;
        for k in 0..dim_size {
            let v = buf[base + k * stride];
            if v > mx {
                mx = v;
            }
        }
        let mut sumexp = 0.0_f64;
        for k in 0..dim_size {
            sumexp += (buf[base + k * stride] - mx).exp();
        }
        let log_sumexp = sumexp.ln();
        for k in 0..dim_size {
            let v = buf[base + k * stride];
            let shifted = v - mx;
            out[base + k * stride] = if log_form {
                shifted - log_sumexp
            } else {
                shifted.exp() / sumexp
            };
        }
    }
    Tensor::from_vec_typed::<f64, _>(shape.to_vec(), out)
        .map_err(|_| BackendError::OutOfMemory { bytes: 0 })
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
