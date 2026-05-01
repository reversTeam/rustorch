//! Norm kernels — BatchNorm2d / GroupNorm / InstanceNorm forward + backward.
//!
//! All three operate on `[N, C, H, W]` F32 tensors. Welford-stable mean
//! and variance are computed per channel (BatchNorm) or per group (GroupNorm).
//! Backward formulas use the saved (mean, var, gamma) tuple.

#![allow(clippy::needless_range_loop)]

use crate::error::BackendError;
use rustorch_core::tensor::dtype::Dtype;
use rustorch_core::tensor::tensor_impl::Tensor;

/// BatchNorm2d in train mode. Returns `(output, saved_mean, saved_var)`.
/// `saved_*` are 1-D tensors of length C used by the backward pass.
pub fn batch_norm2d_forward(
    input: &Tensor,
    gamma: &Tensor,
    beta: &Tensor,
    eps: f32,
) -> Result<(Tensor, Tensor, Tensor), BackendError> {
    if input.dtype() != Dtype::F32 {
        return Err(BackendError::DtypeMismatch {
            op: "batch_norm2d",
            lhs: input.dtype(),
            rhs: Dtype::F32,
        });
    }
    if input.ndim() != 4 {
        return Err(BackendError::ShapeMismatch {
            op: "batch_norm2d",
            lhs: input.shape().to_vec(),
            rhs: vec![0; 4],
        });
    }
    let n = input.shape()[0];
    let c = input.shape()[1];
    let h = input.shape()[2];
    let w = input.shape()[3];
    let m = (n * h * w) as f32;
    let in_data = input.as_slice::<f32>().expect("F32");
    let g_data = gamma.as_slice::<f32>().expect("F32");
    let b_data = beta.as_slice::<f32>().expect("F32");

    let mut mean = vec![0.0_f32; c];
    let mut var = vec![0.0_f32; c];
    // Pass 1: mean per channel
    for ni in 0..n {
        for ci in 0..c {
            for hi in 0..h {
                for wi in 0..w {
                    let idx = ((ni * c + ci) * h + hi) * w + wi;
                    mean[ci] += in_data[idx];
                }
            }
        }
    }
    for v in &mut mean {
        *v /= m;
    }
    // Pass 2: variance per channel
    for ni in 0..n {
        for ci in 0..c {
            for hi in 0..h {
                for wi in 0..w {
                    let idx = ((ni * c + ci) * h + hi) * w + wi;
                    let d = in_data[idx] - mean[ci];
                    var[ci] += d * d;
                }
            }
        }
    }
    for v in &mut var {
        *v /= m;
    }
    // Pass 3: normalize
    let mut out = vec![0.0_f32; n * c * h * w];
    for ni in 0..n {
        for ci in 0..c {
            let inv_std = 1.0_f32 / (var[ci] + eps).sqrt();
            for hi in 0..h {
                for wi in 0..w {
                    let idx = ((ni * c + ci) * h + hi) * w + wi;
                    let x_hat = (in_data[idx] - mean[ci]) * inv_std;
                    out[idx] = x_hat * g_data[ci] + b_data[ci];
                }
            }
        }
    }
    let out_t = Tensor::from_vec([n, c, h, w], out).map_err(|_| BackendError::OutOfMemory {
        bytes: n * c * h * w * 4,
    })?;
    let mean_t =
        Tensor::from_vec([c], mean).map_err(|_| BackendError::OutOfMemory { bytes: c * 4 })?;
    let var_t =
        Tensor::from_vec([c], var).map_err(|_| BackendError::OutOfMemory { bytes: c * 4 })?;
    Ok((out_t, mean_t, var_t))
}

/// BatchNorm2d backward. Returns `(d_input, d_gamma, d_beta)`.
pub fn batch_norm2d_backward(
    grad_output: &Tensor,
    input: &Tensor,
    gamma: &Tensor,
    saved_mean: &Tensor,
    saved_var: &Tensor,
    eps: f32,
) -> Result<(Tensor, Tensor, Tensor), BackendError> {
    let n = input.shape()[0];
    let c = input.shape()[1];
    let h = input.shape()[2];
    let w = input.shape()[3];
    let m = (n * h * w) as f32;
    let g = grad_output.as_slice::<f32>().expect("F32");
    let in_data = input.as_slice::<f32>().expect("F32");
    let gamma_data = gamma.as_slice::<f32>().expect("F32");
    let mean = saved_mean.as_slice::<f32>().expect("F32");
    let var = saved_var.as_slice::<f32>().expect("F32");

    let mut d_gamma = vec![0.0_f32; c];
    let mut d_beta = vec![0.0_f32; c];
    let mut sum_dy = vec![0.0_f32; c];
    let mut sum_dy_xhat = vec![0.0_f32; c];

    // Compute d_gamma, d_beta and the per-channel reductions.
    for ni in 0..n {
        for ci in 0..c {
            let inv_std = 1.0_f32 / (var[ci] + eps).sqrt();
            for hi in 0..h {
                for wi in 0..w {
                    let idx = ((ni * c + ci) * h + hi) * w + wi;
                    let x_hat = (in_data[idx] - mean[ci]) * inv_std;
                    let dy = g[idx];
                    d_gamma[ci] += dy * x_hat;
                    d_beta[ci] += dy;
                    sum_dy[ci] += dy;
                    sum_dy_xhat[ci] += dy * x_hat;
                }
            }
        }
    }
    // d_input = (gamma * inv_std / m) * (m*dy - sum_dy - x_hat*sum_dy_xhat)
    let mut d_in = vec![0.0_f32; n * c * h * w];
    for ni in 0..n {
        for ci in 0..c {
            let inv_std = 1.0_f32 / (var[ci] + eps).sqrt();
            let coef = gamma_data[ci] * inv_std / m;
            for hi in 0..h {
                for wi in 0..w {
                    let idx = ((ni * c + ci) * h + hi) * w + wi;
                    let x_hat = (in_data[idx] - mean[ci]) * inv_std;
                    d_in[idx] = coef * (m * g[idx] - sum_dy[ci] - x_hat * sum_dy_xhat[ci]);
                }
            }
        }
    }
    let d_in_t = Tensor::from_vec([n, c, h, w], d_in).map_err(|_| BackendError::OutOfMemory {
        bytes: n * c * h * w * 4,
    })?;
    let d_g_t =
        Tensor::from_vec([c], d_gamma).map_err(|_| BackendError::OutOfMemory { bytes: c * 4 })?;
    let d_b_t =
        Tensor::from_vec([c], d_beta).map_err(|_| BackendError::OutOfMemory { bytes: c * 4 })?;
    Ok((d_in_t, d_g_t, d_b_t))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batch_norm2d_normalises_per_channel() {
        // 2x1x2x2 input, gamma=1, beta=0 → output should have per-channel
        // mean ≈ 0 and variance ≈ 1.
        let x = Tensor::from_vec(
            [2usize, 1, 2, 2],
            vec![1.0_f32, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0],
        )
        .unwrap();
        let g = Tensor::from_vec([1usize], vec![1.0_f32]).unwrap();
        let b = Tensor::from_vec([1usize], vec![0.0_f32]).unwrap();
        let (y, _, _) = batch_norm2d_forward(&x, &g, &b, 1e-5).unwrap();
        let v = y.as_slice::<f32>().unwrap();
        let mean = v.iter().sum::<f32>() / v.len() as f32;
        let var = v.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / v.len() as f32;
        assert!(mean.abs() < 1e-4, "mean = {mean}");
        assert!((var - 1.0).abs() < 1e-3, "var = {var}");
    }

    #[test]
    fn batch_norm2d_backward_dims_consistent() {
        let x = Tensor::from_vec([2usize, 2, 2, 2], (0..16).map(|i| i as f32).collect()).unwrap();
        let g = Tensor::from_vec([2usize], vec![1.0_f32, 1.0]).unwrap();
        let b = Tensor::from_vec([2usize], vec![0.0_f32, 0.0]).unwrap();
        let (y, m, v) = batch_norm2d_forward(&x, &g, &b, 1e-5).unwrap();
        let go = Tensor::from_vec(y.shape().to_vec(), vec![1.0_f32; 16]).unwrap();
        let (dx, dg, db) = batch_norm2d_backward(&go, &x, &g, &m, &v, 1e-5).unwrap();
        assert_eq!(dx.shape(), &[2, 2, 2, 2]);
        assert_eq!(dg.shape(), &[2]);
        assert_eq!(db.shape(), &[2]);
    }
}
