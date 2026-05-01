//! Pooling kernels (P1.4 / P1.6 minimal slice).
//!
//! v1 ships [`maxpool2d_forward`] + matching backward. Layout
//! `[N, C, H, W]`, square or rectangular kernel, configurable stride,
//! padding=0, F32 only.

use crate::error::BackendError;
use rustorch_core::tensor::dtype::Dtype;
use rustorch_core::tensor::tensor_impl::Tensor;

/// MaxPool2d forward — returns the pooled output and a flat index map
/// (`[N, C, H_out, W_out]` of i64) recording the argmax position
/// inside each pooling window. The index map is used by
/// [`maxpool2d_backward`] to scatter gradient back.
pub fn maxpool2d_forward(
    input: &Tensor,
    kernel_size: (usize, usize),
    stride: (usize, usize),
) -> Result<(Tensor, Tensor), BackendError> {
    if input.dtype() != Dtype::F32 {
        return Err(BackendError::DtypeMismatch {
            op: "maxpool2d",
            lhs: input.dtype(),
            rhs: Dtype::F32,
        });
    }
    if input.ndim() != 4 {
        return Err(BackendError::ShapeMismatch {
            op: "maxpool2d",
            lhs: input.shape().to_vec(),
            rhs: vec![0; 4],
        });
    }
    let n = input.shape()[0];
    let c = input.shape()[1];
    let h = input.shape()[2];
    let w = input.shape()[3];
    let (kh, kw) = kernel_size;
    let (sh, sw) = stride;
    if h < kh || w < kw {
        return Err(BackendError::ShapeMismatch {
            op: "maxpool2d",
            lhs: input.shape().to_vec(),
            rhs: vec![n, c, kh, kw],
        });
    }
    let h_out = (h - kh) / sh + 1;
    let w_out = (w - kw) / sw + 1;
    let in_data = input.as_slice::<f32>().expect("F32 input");

    let mut out = vec![f32::NEG_INFINITY; n * c * h_out * w_out];
    let mut idx = vec![0_i64; n * c * h_out * w_out];

    for ni in 0..n {
        for ci in 0..c {
            for ho in 0..h_out {
                for wo in 0..w_out {
                    let h_start = ho * sh;
                    let w_start = wo * sw;
                    let mut best = f32::NEG_INFINITY;
                    let mut best_idx: i64 = 0;
                    for kr in 0..kh {
                        for kc in 0..kw {
                            let h_in = h_start + kr;
                            let w_in = w_start + kc;
                            let in_idx = ((ni * c + ci) * h + h_in) * w + w_in;
                            let v = in_data[in_idx];
                            if v > best {
                                best = v;
                                best_idx = in_idx as i64;
                            }
                        }
                    }
                    let out_idx = ((ni * c + ci) * h_out + ho) * w_out + wo;
                    out[out_idx] = best;
                    idx[out_idx] = best_idx;
                }
            }
        }
    }

    let out_t =
        Tensor::from_vec([n, c, h_out, w_out], out).map_err(|_| BackendError::OutOfMemory {
            bytes: n * c * h_out * w_out * 4,
        })?;
    let idx_t = Tensor::from_vec_typed::<i64, _>([n, c, h_out, w_out], idx).map_err(|_| {
        BackendError::OutOfMemory {
            bytes: n * c * h_out * w_out * 8,
        }
    })?;
    Ok((out_t, idx_t))
}

/// MaxPool2d backward — scatter `grad_output` into a zero tensor of
/// `in_shape`, accumulating at positions recorded in `argmax_idx`.
pub fn maxpool2d_backward(
    grad_output: &Tensor,
    argmax_idx: &Tensor,
    in_shape: &[usize],
) -> Result<Tensor, BackendError> {
    let g = grad_output.as_slice::<f32>().expect("F32 grad");
    let idx = argmax_idx.as_slice::<i64>().expect("I64 idx");
    let n_total: usize = in_shape.iter().product();
    let mut din = vec![0.0_f32; n_total];
    for (linear, &flat_idx) in idx.iter().enumerate() {
        let i = flat_idx as usize;
        if i < n_total {
            din[i] += g[linear];
        }
    }
    Tensor::from_vec(in_shape.to_vec(), din)
        .map_err(|_| BackendError::OutOfMemory { bytes: n_total * 4 })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maxpool2d_2x2_stride2_correct() {
        // 1x1x4x4 input, kernel 2x2, stride 2 → 1x1x2x2 output
        let x = Tensor::from_vec(
            [1usize, 1, 4, 4],
            vec![
                1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0,
                15.0, 16.0,
            ],
        )
        .unwrap();
        let (y, _idx) = maxpool2d_forward(&x, (2, 2), (2, 2)).unwrap();
        // Top-left window max = 6 (positions 0,1,4,5)
        // Top-right: max = 8 (2,3,6,7)
        // Bot-left: max = 14 (8,9,12,13)
        // Bot-right: max = 16 (10,11,14,15)
        assert_eq!(y.shape(), &[1, 1, 2, 2]);
        assert_eq!(y.as_slice::<f32>().unwrap(), &[6.0_f32, 8.0, 14.0, 16.0]);
    }

    #[test]
    fn maxpool2d_index_records_argmax_position() {
        let x = Tensor::from_vec([1usize, 1, 2, 2], vec![1.0_f32, 5.0, 3.0, 2.0]).unwrap();
        let (y, idx) = maxpool2d_forward(&x, (2, 2), (1, 1)).unwrap();
        // 2x2 window → single output max=5, argmax at flat idx 1
        assert_eq!(y.as_slice::<f32>().unwrap(), &[5.0_f32]);
        assert_eq!(idx.as_slice::<i64>().unwrap(), &[1_i64]);
    }

    #[test]
    fn maxpool2d_backward_scatters_to_argmax() {
        let x = Tensor::from_vec([1usize, 1, 2, 2], vec![1.0_f32, 5.0, 3.0, 2.0]).unwrap();
        let (_, idx) = maxpool2d_forward(&x, (2, 2), (1, 1)).unwrap();
        let grad = Tensor::from_vec([1usize, 1, 1, 1], vec![10.0_f32]).unwrap();
        let din = maxpool2d_backward(&grad, &idx, &[1, 1, 2, 2]).unwrap();
        // Grad should flow only to position 1
        assert_eq!(din.shape(), &[1, 1, 2, 2]);
        assert_eq!(din.as_slice::<f32>().unwrap(), &[0.0_f32, 10.0, 0.0, 0.0]);
    }

    #[test]
    fn maxpool2d_stride_1_overlapping_windows() {
        let x = Tensor::from_vec(
            [1usize, 1, 3, 3],
            vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        )
        .unwrap();
        let (y, _) = maxpool2d_forward(&x, (2, 2), (1, 1)).unwrap();
        // 4 overlapping windows: max values are 5, 6, 8, 9
        assert_eq!(y.shape(), &[1, 1, 2, 2]);
        assert_eq!(y.as_slice::<f32>().unwrap(), &[5.0_f32, 6.0, 8.0, 9.0]);
    }
}
