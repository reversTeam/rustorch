//! Loss functions (P1.4 task `Loss functions`).
//!
//! Implements:
//! - `mse_loss(input, target, reduction)` — `mean((x - y)²)` etc.
//! - `cross_entropy(input, target, reduction)` — `log_softmax(input)`
//!   then `nll_loss` against I64 class targets.
//! - `nll_loss(log_probs, target, reduction)` — `-log_probs[i, target[i]]`.
//! - `bce_with_logits(input, target, reduction)` — numerically stable
//!   binary cross-entropy with logits via softplus form.
//!
//! Reduction modes (per [`crate::backend::Reduction`]):
//! - `Mean` → divide sum by element count.
//! - `Sum` → return raw sum.
//! - `None` → keep per-element values.

use crate::backend::Reduction;
use crate::error::BackendError;
use crate::kernels::softmax::log_softmax;
use rustorch_core::tensor::dtype::Dtype;
use rustorch_core::tensor::tensor_impl::Tensor;

/// MSE loss: `((input - target)²)`, then reduce.
pub fn mse_loss(
    input: &Tensor,
    target: &Tensor,
    reduction: Reduction,
) -> Result<Tensor, BackendError> {
    if input.shape() != target.shape() {
        return Err(BackendError::ShapeMismatch {
            op: "mse_loss",
            lhs: input.shape().to_vec(),
            rhs: target.shape().to_vec(),
        });
    }
    if input.dtype() != target.dtype() {
        return Err(BackendError::DtypeMismatch {
            op: "mse_loss",
            lhs: input.dtype(),
            rhs: target.dtype(),
        });
    }
    match input.dtype() {
        Dtype::F32 => mse_f32(input, target, reduction),
        Dtype::F64 => mse_f64(input, target, reduction),
        d => Err(BackendError::DtypeMismatch {
            op: "mse_loss",
            lhs: d,
            rhs: d,
        }),
    }
}

/// Cross-entropy: combines log_softmax + NLL with class-index targets (I64).
pub fn cross_entropy(
    input: &Tensor,
    target: &Tensor,
    reduction: Reduction,
) -> Result<Tensor, BackendError> {
    if !matches!(input.dtype(), Dtype::F32 | Dtype::F64) {
        return Err(BackendError::DtypeMismatch {
            op: "cross_entropy",
            lhs: input.dtype(),
            rhs: Dtype::F32,
        });
    }
    if target.dtype() != Dtype::I64 {
        return Err(BackendError::DtypeMismatch {
            op: "cross_entropy",
            lhs: target.dtype(),
            rhs: Dtype::I64,
        });
    }
    if input.ndim() != 2 {
        return Err(BackendError::ShapeMismatch {
            op: "cross_entropy",
            lhs: input.shape().to_vec(),
            rhs: vec![],
        });
    }
    if target.ndim() != 1 || target.numel() != input.shape()[0] {
        return Err(BackendError::ShapeMismatch {
            op: "cross_entropy",
            lhs: input.shape().to_vec(),
            rhs: target.shape().to_vec(),
        });
    }
    let log_probs = log_softmax(input, 1)?;
    nll_loss(&log_probs, target, reduction)
}

/// NLL loss: `-log_probs[i, target[i]]`, then reduce.
pub fn nll_loss(
    log_probs: &Tensor,
    target: &Tensor,
    reduction: Reduction,
) -> Result<Tensor, BackendError> {
    if !matches!(log_probs.dtype(), Dtype::F32 | Dtype::F64) {
        return Err(BackendError::DtypeMismatch {
            op: "nll_loss",
            lhs: log_probs.dtype(),
            rhs: Dtype::F32,
        });
    }
    if target.dtype() != Dtype::I64 {
        return Err(BackendError::DtypeMismatch {
            op: "nll_loss",
            lhs: target.dtype(),
            rhs: Dtype::I64,
        });
    }
    if log_probs.ndim() != 2 {
        return Err(BackendError::ShapeMismatch {
            op: "nll_loss",
            lhs: log_probs.shape().to_vec(),
            rhs: vec![],
        });
    }
    let n_samples = log_probs.shape()[0];
    let n_classes = log_probs.shape()[1];
    if target.numel() != n_samples {
        return Err(BackendError::ShapeMismatch {
            op: "nll_loss",
            lhs: log_probs.shape().to_vec(),
            rhs: target.shape().to_vec(),
        });
    }
    let target_v: Vec<i64> = target.iter_elements::<i64>().expect("i64").collect();
    match log_probs.dtype() {
        Dtype::F32 => {
            let lp: Vec<f32> = log_probs.iter_elements::<f32>().expect("dtype").collect();
            let mut per_sample = Vec::with_capacity(n_samples);
            for i in 0..n_samples {
                let t = target_v[i];
                if t < 0 || (t as usize) >= n_classes {
                    return Err(BackendError::IndexOutOfBounds {
                        op: "nll_loss",
                        index: t,
                        bound: n_classes,
                    });
                }
                per_sample.push(-lp[i * n_classes + t as usize]);
            }
            reduce_per_sample_f32(per_sample, reduction)
        },
        Dtype::F64 => {
            let lp: Vec<f64> = log_probs.iter_elements::<f64>().expect("dtype").collect();
            let mut per_sample = Vec::with_capacity(n_samples);
            for i in 0..n_samples {
                let t = target_v[i];
                if t < 0 || (t as usize) >= n_classes {
                    return Err(BackendError::IndexOutOfBounds {
                        op: "nll_loss",
                        index: t,
                        bound: n_classes,
                    });
                }
                per_sample.push(-lp[i * n_classes + t as usize]);
            }
            reduce_per_sample_f64(per_sample, reduction)
        },
        d => Err(BackendError::DtypeMismatch {
            op: "nll_loss",
            lhs: d,
            rhs: d,
        }),
    }
}

/// Binary cross-entropy with logits, numerically stable form:
///
/// ```text
///   bce_with_logits(x, t) = max(x, 0) - x*t + log(1 + exp(-|x|))
/// ```
pub fn bce_with_logits(
    input: &Tensor,
    target: &Tensor,
    reduction: Reduction,
) -> Result<Tensor, BackendError> {
    if input.shape() != target.shape() {
        return Err(BackendError::ShapeMismatch {
            op: "bce_with_logits",
            lhs: input.shape().to_vec(),
            rhs: target.shape().to_vec(),
        });
    }
    if input.dtype() != target.dtype() {
        return Err(BackendError::DtypeMismatch {
            op: "bce_with_logits",
            lhs: input.dtype(),
            rhs: target.dtype(),
        });
    }
    match input.dtype() {
        Dtype::F32 => {
            let xs: Vec<f32> = input.iter_elements::<f32>().expect("dtype").collect();
            let ts: Vec<f32> = target.iter_elements::<f32>().expect("dtype").collect();
            let per: Vec<f32> = xs
                .iter()
                .zip(ts.iter())
                .map(|(&x, &t)| x.max(0.0) - x * t + (1.0 + (-x.abs()).exp()).ln())
                .collect();
            match reduction {
                Reduction::None => Tensor::from_vec_typed::<f32, _>(input.shape().to_vec(), per)
                    .map_err(|_| BackendError::OutOfMemory { bytes: 0 }),
                _ => reduce_per_sample_f32(per, reduction),
            }
        },
        Dtype::F64 => {
            let xs: Vec<f64> = input.iter_elements::<f64>().expect("dtype").collect();
            let ts: Vec<f64> = target.iter_elements::<f64>().expect("dtype").collect();
            let per: Vec<f64> = xs
                .iter()
                .zip(ts.iter())
                .map(|(&x, &t)| x.max(0.0) - x * t + (1.0 + (-x.abs()).exp()).ln())
                .collect();
            match reduction {
                Reduction::None => Tensor::from_vec_typed::<f64, _>(input.shape().to_vec(), per)
                    .map_err(|_| BackendError::OutOfMemory { bytes: 0 }),
                _ => reduce_per_sample_f64(per, reduction),
            }
        },
        d => Err(BackendError::DtypeMismatch {
            op: "bce_with_logits",
            lhs: d,
            rhs: d,
        }),
    }
}

// -------------------- helpers --------------------

fn mse_f32(input: &Tensor, target: &Tensor, reduction: Reduction) -> Result<Tensor, BackendError> {
    let xs: Vec<f32> = input.iter_elements::<f32>().expect("dtype").collect();
    let ts: Vec<f32> = target.iter_elements::<f32>().expect("dtype").collect();
    let per: Vec<f32> = xs
        .iter()
        .zip(ts.iter())
        .map(|(&x, &t)| (x - t) * (x - t))
        .collect();
    match reduction {
        Reduction::None => Tensor::from_vec_typed::<f32, _>(input.shape().to_vec(), per)
            .map_err(|_| BackendError::OutOfMemory { bytes: 0 }),
        _ => reduce_per_sample_f32(per, reduction),
    }
}

fn mse_f64(input: &Tensor, target: &Tensor, reduction: Reduction) -> Result<Tensor, BackendError> {
    let xs: Vec<f64> = input.iter_elements::<f64>().expect("dtype").collect();
    let ts: Vec<f64> = target.iter_elements::<f64>().expect("dtype").collect();
    let per: Vec<f64> = xs
        .iter()
        .zip(ts.iter())
        .map(|(&x, &t)| (x - t) * (x - t))
        .collect();
    match reduction {
        Reduction::None => Tensor::from_vec_typed::<f64, _>(input.shape().to_vec(), per)
            .map_err(|_| BackendError::OutOfMemory { bytes: 0 }),
        _ => reduce_per_sample_f64(per, reduction),
    }
}

fn reduce_per_sample_f32(per: Vec<f32>, reduction: Reduction) -> Result<Tensor, BackendError> {
    let n = per.len() as f32;
    let s: f32 = per.iter().sum();
    let val = match reduction {
        Reduction::Mean => {
            if n == 0.0 {
                f32::NAN
            } else {
                s / n
            }
        },
        Reduction::Sum => s,
        Reduction::None => unreachable!(),
    };
    Tensor::from_vec_typed::<f32, _>([], vec![val])
        .map_err(|_| BackendError::OutOfMemory { bytes: 4 })
}

fn reduce_per_sample_f64(per: Vec<f64>, reduction: Reduction) -> Result<Tensor, BackendError> {
    let n = per.len() as f64;
    let s: f64 = per.iter().sum();
    let val = match reduction {
        Reduction::Mean => {
            if n == 0.0 {
                f64::NAN
            } else {
                s / n
            }
        },
        Reduction::Sum => s,
        Reduction::None => unreachable!(),
    };
    Tensor::from_vec_typed::<f64, _>([], vec![val])
        .map_err(|_| BackendError::OutOfMemory { bytes: 8 })
}
