//! Conv2d forward + backward kernels (P1.4 minimal slice).
//!
//! v1: stride=1, padding configurable, no dilation, no groups,
//! F32 only. Naive 7-deep loop — correct, slow.
//!
//! Layout: [N, C_in, H, W] inputs, [C_out, C_in, kH, kW] weights,
//! produces [N, C_out, H_out, W_out] where:
//! ```text
//!   H_out = (H + 2*padH - kH) + 1
//!   W_out = (W + 2*padW - kW) + 1
//! ```
//!
//! Future slices: stride > 1, dilation, groups, depthwise via groups.

use crate::error::BackendError;
use rustorch_core::tensor::dtype::Dtype;
use rustorch_core::tensor::tensor_impl::Tensor;

/// Forward: conv2d with stride=1.
pub fn conv2d_forward(
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    pad_h: usize,
    pad_w: usize,
) -> Result<Tensor, BackendError> {
    if input.dtype() != Dtype::F32 || weight.dtype() != Dtype::F32 {
        return Err(BackendError::DtypeMismatch {
            op: "conv2d",
            lhs: input.dtype(),
            rhs: weight.dtype(),
        });
    }
    if input.ndim() != 4 || weight.ndim() != 4 {
        return Err(BackendError::ShapeMismatch {
            op: "conv2d",
            lhs: input.shape().to_vec(),
            rhs: weight.shape().to_vec(),
        });
    }
    let n = input.shape()[0];
    let c_in = input.shape()[1];
    let h = input.shape()[2];
    let w = input.shape()[3];
    let c_out = weight.shape()[0];
    let w_cin = weight.shape()[1];
    let kh = weight.shape()[2];
    let kw = weight.shape()[3];
    if c_in != w_cin {
        return Err(BackendError::ShapeMismatch {
            op: "conv2d",
            lhs: input.shape().to_vec(),
            rhs: weight.shape().to_vec(),
        });
    }
    let h_out = (h + 2 * pad_h).saturating_sub(kh) + 1;
    let w_out = (w + 2 * pad_w).saturating_sub(kw) + 1;
    let in_data = input
        .as_slice::<f32>()
        .ok_or_else(|| BackendError::DtypeMismatch {
            op: "conv2d",
            lhs: input.dtype(),
            rhs: Dtype::F32,
        })?;
    let w_data = weight
        .as_slice::<f32>()
        .ok_or_else(|| BackendError::DtypeMismatch {
            op: "conv2d",
            lhs: weight.dtype(),
            rhs: Dtype::F32,
        })?;
    let bias_data: Option<&[f32]> = bias.and_then(|b| b.as_slice::<f32>());

    let mut out = vec![0.0_f32; n * c_out * h_out * w_out];

    for ni in 0..n {
        for co in 0..c_out {
            let b = bias_data.map(|b| b[co]).unwrap_or(0.0);
            for hi in 0..h_out {
                for wi in 0..w_out {
                    let mut acc = 0.0_f32;
                    for ci in 0..c_in {
                        for kr in 0..kh {
                            for kc in 0..kw {
                                let h_in = hi + kr;
                                let w_in = wi + kc;
                                // padding handled by zero-skipping.
                                if h_in < pad_h
                                    || h_in >= h + pad_h
                                    || w_in < pad_w
                                    || w_in >= w + pad_w
                                {
                                    continue;
                                }
                                let h_in_real = h_in - pad_h;
                                let w_in_real = w_in - pad_w;
                                let in_idx = ((ni * c_in + ci) * h + h_in_real) * w + w_in_real;
                                let w_idx = ((co * c_in + ci) * kh + kr) * kw + kc;
                                acc += in_data[in_idx] * w_data[w_idx];
                            }
                        }
                    }
                    let out_idx = ((ni * c_out + co) * h_out + hi) * w_out + wi;
                    out[out_idx] = acc + b;
                }
            }
        }
    }

    Tensor::from_vec([n, c_out, h_out, w_out], out).map_err(|_| BackendError::OutOfMemory {
        bytes: n * c_out * h_out * w_out * 4,
    })
}

/// Backward gradient with respect to the input: full convolution with
/// the weight flipped along the spatial axes.
pub fn conv2d_grad_input(
    grad_output: &Tensor,
    weight: &Tensor,
    in_shape: &[usize],
    pad_h: usize,
    pad_w: usize,
) -> Result<Tensor, BackendError> {
    let n = in_shape[0];
    let c_in = in_shape[1];
    let h = in_shape[2];
    let w = in_shape[3];
    let c_out = weight.shape()[0];
    let kh = weight.shape()[2];
    let kw = weight.shape()[3];
    let h_out = grad_output.shape()[2];
    let w_out = grad_output.shape()[3];
    let g = grad_output
        .as_slice::<f32>()
        .ok_or_else(|| BackendError::DtypeMismatch {
            op: "conv2d_grad_input",
            lhs: grad_output.dtype(),
            rhs: Dtype::F32,
        })?;
    let w_data = weight.as_slice::<f32>().expect("F32 weight");

    let mut din = vec![0.0_f32; n * c_in * h * w];
    for ni in 0..n {
        for ci in 0..c_in {
            for hi in 0..h {
                for wi in 0..w {
                    let mut acc = 0.0_f32;
                    for co in 0..c_out {
                        for kr in 0..kh {
                            for kc in 0..kw {
                                // Find the output position (ho, wo) such that
                                // hi + pad_h = ho + kr  → ho = hi + pad_h - kr
                                let ho_signed = hi as isize + pad_h as isize - kr as isize;
                                let wo_signed = wi as isize + pad_w as isize - kc as isize;
                                if ho_signed < 0 || wo_signed < 0 {
                                    continue;
                                }
                                let ho = ho_signed as usize;
                                let wo = wo_signed as usize;
                                if ho >= h_out || wo >= w_out {
                                    continue;
                                }
                                let g_idx = ((ni * c_out + co) * h_out + ho) * w_out + wo;
                                let w_idx = ((co * c_in + ci) * kh + kr) * kw + kc;
                                acc += g[g_idx] * w_data[w_idx];
                            }
                        }
                    }
                    let in_idx = ((ni * c_in + ci) * h + hi) * w + wi;
                    din[in_idx] = acc;
                }
            }
        }
    }
    Tensor::from_vec([n, c_in, h, w], din).map_err(|_| BackendError::OutOfMemory {
        bytes: n * c_in * h * w * 4,
    })
}

/// Backward gradient with respect to the weight: cross-correlation
/// between input and grad_output.
pub fn conv2d_grad_weight(
    input: &Tensor,
    grad_output: &Tensor,
    weight_shape: &[usize],
    pad_h: usize,
    pad_w: usize,
) -> Result<Tensor, BackendError> {
    let c_out = weight_shape[0];
    let c_in = weight_shape[1];
    let kh = weight_shape[2];
    let kw = weight_shape[3];
    let n = input.shape()[0];
    let h = input.shape()[2];
    let w = input.shape()[3];
    let h_out = grad_output.shape()[2];
    let w_out = grad_output.shape()[3];
    let in_data = input.as_slice::<f32>().expect("F32 input");
    let g = grad_output.as_slice::<f32>().expect("F32 grad");

    let mut dw = vec![0.0_f32; c_out * c_in * kh * kw];
    for co in 0..c_out {
        for ci in 0..c_in {
            for kr in 0..kh {
                for kc in 0..kw {
                    let mut acc = 0.0_f32;
                    for ni in 0..n {
                        for ho in 0..h_out {
                            for wo in 0..w_out {
                                let h_in_signed = ho as isize + kr as isize - pad_h as isize;
                                let w_in_signed = wo as isize + kc as isize - pad_w as isize;
                                if h_in_signed < 0
                                    || w_in_signed < 0
                                    || h_in_signed >= h as isize
                                    || w_in_signed >= w as isize
                                {
                                    continue;
                                }
                                let in_idx = ((ni * c_in + ci) * h + h_in_signed as usize) * w
                                    + w_in_signed as usize;
                                let g_idx = ((ni * c_out + co) * h_out + ho) * w_out + wo;
                                acc += in_data[in_idx] * g[g_idx];
                            }
                        }
                    }
                    let w_idx = ((co * c_in + ci) * kh + kr) * kw + kc;
                    dw[w_idx] = acc;
                }
            }
        }
    }
    Tensor::from_vec([c_out, c_in, kh, kw], dw).map_err(|_| BackendError::OutOfMemory {
        bytes: c_out * c_in * kh * kw * 4,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conv2d_identity_kernel() {
        // 1x1 identity kernel: w[0,0,0,0] = 1, output == input.
        let x = Tensor::from_vec([1usize, 1, 3, 3], (0..9).map(|i| i as f32).collect()).unwrap();
        let w = Tensor::from_vec([1usize, 1, 1, 1], vec![1.0_f32]).unwrap();
        let y = conv2d_forward(&x, &w, None, 0, 0).unwrap();
        assert_eq!(y.shape(), &[1, 1, 3, 3]);
        assert_eq!(y.as_slice::<f32>().unwrap(), x.as_slice::<f32>().unwrap());
    }

    #[test]
    fn conv2d_3x3_box_blur() {
        // 3x3 box filter (sum) on a 3x3 input → 1x1 output equal to sum.
        let x = Tensor::from_vec([1usize, 1, 3, 3], vec![1.0_f32; 9]).unwrap();
        let w = Tensor::from_vec([1usize, 1, 3, 3], vec![1.0_f32; 9]).unwrap();
        let y = conv2d_forward(&x, &w, None, 0, 0).unwrap();
        assert_eq!(y.shape(), &[1, 1, 1, 1]);
        assert_eq!(y.as_slice::<f32>().unwrap(), &[9.0_f32]);
    }

    #[test]
    fn conv2d_with_bias() {
        let x = Tensor::from_vec([1usize, 1, 2, 2], vec![1.0_f32, 1.0, 1.0, 1.0]).unwrap();
        let w = Tensor::from_vec([2usize, 1, 1, 1], vec![1.0_f32, 2.0]).unwrap();
        let b = Tensor::from_vec([2usize], vec![10.0_f32, 20.0]).unwrap();
        let y = conv2d_forward(&x, &w, Some(&b), 0, 0).unwrap();
        assert_eq!(y.shape(), &[1, 2, 2, 2]);
        // Channel 0: 1 * 1 + 10 = 11; Channel 1: 1 * 2 + 20 = 22
        assert_eq!(
            y.as_slice::<f32>().unwrap(),
            &[11.0_f32, 11.0, 11.0, 11.0, 22.0, 22.0, 22.0, 22.0]
        );
    }

    #[test]
    fn conv2d_padding_preserves_shape() {
        // 3x3 kernel + padding 1 → output size = input size
        let x = Tensor::from_vec([1usize, 1, 5, 5], vec![1.0_f32; 25]).unwrap();
        let w = Tensor::from_vec([1usize, 1, 3, 3], vec![1.0_f32; 9]).unwrap();
        let y = conv2d_forward(&x, &w, None, 1, 1).unwrap();
        assert_eq!(y.shape(), &[1, 1, 5, 5]);
    }

    #[test]
    fn conv2d_grad_input_dim_consistency() {
        let x = Tensor::from_vec([1usize, 1, 4, 4], (0..16).map(|i| i as f32).collect()).unwrap();
        let w = Tensor::from_vec([1usize, 1, 3, 3], vec![1.0_f32; 9]).unwrap();
        let y = conv2d_forward(&x, &w, None, 0, 0).unwrap();
        // Backward: grad_output is ones [1,1,2,2]
        let g = Tensor::from_vec([1usize, 1, 2, 2], vec![1.0_f32; 4]).unwrap();
        let din = conv2d_grad_input(&g, &w, &[1, 1, 4, 4], 0, 0).unwrap();
        assert_eq!(din.shape(), &[1, 1, 4, 4]);
        // dw should also be valid shape
        let dw = conv2d_grad_weight(&x, &g, &[1, 1, 3, 3], 0, 0).unwrap();
        assert_eq!(dw.shape(), &[1, 1, 3, 3]);
        // y is just used to verify forward succeeded.
        assert_eq!(y.shape(), &[1, 1, 2, 2]);
    }

    #[test]
    fn conv2d_grad_weight_correctness() {
        // x=[[1,2],[3,4]], w=identity 1x1 → y=x. With grad=ones, dw should be sum(x).
        let x = Tensor::from_vec([1usize, 1, 2, 2], vec![1.0_f32, 2.0, 3.0, 4.0]).unwrap();
        let g = Tensor::from_vec([1usize, 1, 2, 2], vec![1.0_f32; 4]).unwrap();
        let dw = conv2d_grad_weight(&x, &g, &[1, 1, 1, 1], 0, 0).unwrap();
        // dw[0, 0, 0, 0] = sum over (n, h, w) of x * g = 1+2+3+4 = 10
        assert_eq!(dw.as_slice::<f32>().unwrap(), &[10.0_f32]);
    }
}
