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
    let shape = src.shape().to_vec();
    let dim_size = shape[dim];
    let n_total = src.numel();
    let mut out = vec![0.0_f32; n_total];

    // Hot path: last-dim softmax over a contiguous tensor (the
    // dominant shape for transformer logits / attention scores).
    // The row-major layout makes every "row" of `dim_size` elements
    // contiguous, so we can:
    // 1. Borrow `&[f32]` directly (no `.iter_elements().collect()`,
    //    saves one full input allocation + copy).
    // 2. Drop `decode_coords` Vec<usize> allocation per row.
    // 3. Use rayon to parallelise across independent rows (T9-new).
    // 4. Use online softmax (Milakov & Gimelshein 2018) — one pass
    //    over the row computes both `max` and `sum_exp` together, vs
    //    two passes in the strided fallback. Then a second pass
    //    writes the output. Total: 2 passes vs 3 (≈ 33% bandwidth
    //    saving on memory-bound shapes like 1024×50257).
    if dim + 1 == shape.len() && src.is_contiguous() && src.storage_offset() == 0 {
        if let Some(input) = src.as_slice::<f32>() {
            let n_rows = if dim_size == 0 { 0 } else { n_total / dim_size };
            #[cfg(not(target_arch = "wasm32"))]
            {
                use rayon::prelude::*;
                input
                    .par_chunks_exact(dim_size.max(1))
                    .zip(out.par_chunks_exact_mut(dim_size.max(1)))
                    .for_each(|(row_in, row_out)| {
                        softmax_row_online_f32(row_in, row_out, log_form)
                    });
            }
            #[cfg(target_arch = "wasm32")]
            {
                for i in 0..n_rows {
                    let base = i * dim_size;
                    let row_in = &input[base..base + dim_size];
                    let row_out = &mut out[base..base + dim_size];
                    softmax_row_online_f32(row_in, row_out, log_form);
                }
            }
            let _ = n_rows;
            return Tensor::from_vec_typed::<f32, _>(shape, out)
                .map_err(|_| BackendError::OutOfMemory { bytes: 0 });
        }
    }

    // Fallback: strided / non-contiguous / non-last-dim — 3-pass loop.
    let buf: Vec<f32> = src.iter_elements::<f32>().expect("dtype").collect();
    let strides = contiguous_strides(&shape);
    let n_other: usize = n_total / dim_size.max(1);
    let mut other_shape = shape.clone();
    other_shape[dim] = 1;
    for linear in 0..n_other {
        let mut coords = decode_coords(linear, &other_shape);
        coords[dim] = 0;
        let base: usize = coords.iter().zip(strides.iter()).map(|(c, s)| c * s).sum();
        let stride = strides[dim];
        let mut mx = f32::NEG_INFINITY;
        for k in 0..dim_size {
            let v = buf[base + k * stride];
            if v > mx {
                mx = v;
            }
        }
        let mut sumexp = 0.0_f32;
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
    Tensor::from_vec_typed::<f32, _>(shape, out).map_err(|_| BackendError::OutOfMemory { bytes: 0 })
}

/// Compute softmax (or log-softmax) over a single contiguous row using
/// the online algorithm of Milakov & Gimelshein 2018:
///
///   m_i = max(m_{i-1}, x_i)
///   s_i = s_{i-1} * exp(m_{i-1} - m_i) + exp(x_i - m_i)
///
/// One pass yields both `max` and `sum exp(x - max)`. A second pass
/// writes the normalised output. LLVM auto-vectorises the inner loop
/// on `target-cpu=native` (NEON FMA for the multiply-adds; `expf`
/// stays scalar but is amortised over the bandwidth-bound writes).
#[inline]
fn softmax_row_online_f32(row_in: &[f32], row_out: &mut [f32], log_form: bool) {
    let n = row_in.len();
    if n == 0 {
        return;
    }
    let mut mx = f32::NEG_INFINITY;
    let mut sum = 0.0_f32;
    for &x in row_in {
        if x > mx {
            // shift previous accumulator into the new max frame
            sum *= (mx - x).exp();
            mx = x;
        }
        sum += (x - mx).exp();
    }
    if log_form {
        let log_sum = sum.ln();
        for (out, &x) in row_out.iter_mut().zip(row_in.iter()) {
            *out = (x - mx) - log_sum;
        }
    } else {
        let inv_sum = 1.0_f32 / sum;
        for (out, &x) in row_out.iter_mut().zip(row_in.iter()) {
            *out = (x - mx).exp() * inv_sum;
        }
    }
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
