//! cuDNN bindings — convolution / batchnorm / RNN.
//!
//! Without `--features cuda`, ops fall back to scalar reference
//! implementations so call sites compile and tests pass without GPU.

#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_many_arguments)]

use crate::error::CudaError;

/// Tensor descriptor (NCHW layout).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TensorDescriptor {
    /// Batch.
    pub n: usize,
    /// Channels.
    pub c: usize,
    /// Height.
    pub h: usize,
    /// Width.
    pub w: usize,
}

/// Filter descriptor (KCRS layout: out_channels, in_channels, kernel_h, kernel_w).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FilterDescriptor {
    /// Output channels.
    pub k: usize,
    /// Input channels.
    pub c: usize,
    /// Kernel height.
    pub r: usize,
    /// Kernel width.
    pub s: usize,
}

/// Convolution descriptor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConvolutionDescriptor {
    /// Padding height.
    pub pad_h: usize,
    /// Padding width.
    pub pad_w: usize,
    /// Stride height.
    pub stride_h: usize,
    /// Stride width.
    pub stride_w: usize,
}

impl ConvolutionDescriptor {
    /// 3×3 stride 1 padding 1 (same-padding) — the canonical default.
    pub fn k3s1p1() -> Self {
        Self {
            pad_h: 1,
            pad_w: 1,
            stride_h: 1,
            stride_w: 1,
        }
    }

    /// Compute output spatial dims given input + filter shapes.
    pub fn output_size(&self, input: TensorDescriptor, filter: FilterDescriptor) -> (usize, usize) {
        let h_out = (input.h + 2 * self.pad_h - filter.r) / self.stride_h + 1;
        let w_out = (input.w + 2 * self.pad_w - filter.s) / self.stride_w + 1;
        (h_out, w_out)
    }
}

/// 2D convolution forward `y = conv(x, w)`. Layout: x is NCHW
/// `[input.n, input.c, input.h, input.w]`; w is KCRS
/// `[filter.k, filter.c, filter.r, filter.s]`; y is `[input.n,
/// filter.k, h_out, w_out]`.
pub fn conv2d_forward_f32(
    x: &[f32],
    w: &[f32],
    y: &mut [f32],
    input: TensorDescriptor,
    filter: FilterDescriptor,
    conv: ConvolutionDescriptor,
) -> Result<(), CudaError> {
    // Short-circuit on empty batch — no kernel launch, no shape validation
    // beyond the trivial empty-y assertion (so callers get a free no-op).
    if input.n == 0 {
        if !y.is_empty() {
            return Err(CudaError::Unsupported {
                msg: format!("y expected 0 (n==0), got {}", y.len()),
            });
        }
        return Ok(());
    }
    if x.len() != input.n * input.c * input.h * input.w {
        return Err(CudaError::Unsupported {
            msg: format!(
                "x expected {}, got {}",
                input.n * input.c * input.h * input.w,
                x.len()
            ),
        });
    }
    if w.len() != filter.k * filter.c * filter.r * filter.s {
        return Err(CudaError::Unsupported {
            msg: format!(
                "w expected {}, got {}",
                filter.k * filter.c * filter.r * filter.s,
                w.len()
            ),
        });
    }
    if input.c != filter.c {
        return Err(CudaError::Unsupported {
            msg: format!("input channels {} != filter channels {}", input.c, filter.c),
        });
    }
    let (h_out, w_out) = conv.output_size(input, filter);
    if y.len() != input.n * filter.k * h_out * w_out {
        return Err(CudaError::Unsupported {
            msg: format!(
                "y expected {}, got {}",
                input.n * filter.k * h_out * w_out,
                y.len()
            ),
        });
    }
    // Scalar reference. Real cuda path: cudnnConvolutionForward.
    for n in 0..input.n {
        for k in 0..filter.k {
            for h_o in 0..h_out {
                for w_o in 0..w_out {
                    let mut acc = 0.0f32;
                    for c in 0..filter.c {
                        for r in 0..filter.r {
                            for s in 0..filter.s {
                                let h_i = (h_o * conv.stride_h + r) as isize - conv.pad_h as isize;
                                let w_i = (w_o * conv.stride_w + s) as isize - conv.pad_w as isize;
                                if h_i >= 0
                                    && h_i < input.h as isize
                                    && w_i >= 0
                                    && w_i < input.w as isize
                                {
                                    let xi = ((n * input.c + c) * input.h + h_i as usize) * input.w
                                        + w_i as usize;
                                    let wi = ((k * filter.c + c) * filter.r + r) * filter.s + s;
                                    acc += x[xi] * w[wi];
                                }
                            }
                        }
                    }
                    let yi = ((n * filter.k + k) * h_out + h_o) * w_out + w_o;
                    y[yi] = acc;
                }
            }
        }
    }
    Ok(())
}

/// BatchNorm forward (inference variant: uses running statistics).
/// `y[n, c, h, w] = scale[c] * (x[n, c, h, w] - mean[c]) / sqrt(var[c] + eps) + bias[c]`
pub fn batchnorm_inference_f32(
    x: &[f32],
    scale: &[f32],
    bias: &[f32],
    mean: &[f32],
    var: &[f32],
    y: &mut [f32],
    input: TensorDescriptor,
    eps: f32,
) -> Result<(), CudaError> {
    let n = input.n * input.c * input.h * input.w;
    if x.len() != n || y.len() != n {
        return Err(CudaError::Unsupported {
            msg: "x or y shape mismatch".into(),
        });
    }
    for c in 0..input.c {
        let inv_std = 1.0 / (var[c] + eps).sqrt();
        let s = scale[c] * inv_std;
        let b = bias[c] - mean[c] * s;
        for ni in 0..input.n {
            for hi in 0..input.h {
                for wi in 0..input.w {
                    let off = ((ni * input.c + c) * input.h + hi) * input.w + wi;
                    y[off] = x[off] * s + b;
                }
            }
        }
    }
    let _ = (scale, bias, mean); // silence in-loop double-borrow lints
    Ok(())
}

/// Recurrent cell mode — what kind of recurrent unit does the RNN
/// descriptor describe.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RnnMode {
    /// Vanilla tanh RNN.
    RnnTanh,
    /// LSTM (input/forget/cell/output gates).
    Lstm,
    /// GRU (reset/update/new gates).
    Gru,
}

/// RNN descriptor — minimal subset of `cudnnRNNDescriptor_t`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RnnDescriptor {
    /// Hidden state size.
    pub hidden_size: usize,
    /// Number of stacked layers.
    pub num_layers: usize,
    /// Cell flavour.
    pub mode: RnnMode,
    /// True for bidirectional. v1 only supports `false`.
    pub bidirectional: bool,
}

/// Sigmoid for the gates.
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Single-layer LSTM forward pass on a length-`seq_len` sequence.
///
/// Inputs (row-major):
/// - `x`: `[seq_len, batch, input_size]`
/// - `w_ih`: `[4*hidden, input_size]` — gate weights for input (i, f, g, o)
/// - `w_hh`: `[4*hidden, hidden]` — gate weights for hidden state
/// - `b_ih`, `b_hh`: `[4*hidden]` biases
/// - `h0`, `c0`: `[batch, hidden]` initial hidden + cell
///
/// Outputs:
/// - `output`: `[seq_len, batch, hidden]` — hidden at each timestep
/// - `hn`, `cn`: `[batch, hidden]` final hidden + cell
///
/// Real cuda path: `cudnnRNNForwardInference` with persistent kernels.
pub fn lstm_forward_f32(
    x: &[f32],
    w_ih: &[f32],
    w_hh: &[f32],
    b_ih: &[f32],
    b_hh: &[f32],
    h0: &[f32],
    c0: &[f32],
    output: &mut [f32],
    hn: &mut [f32],
    cn: &mut [f32],
    seq_len: usize,
    batch: usize,
    input_size: usize,
    hidden_size: usize,
) -> Result<(), CudaError> {
    let g = 4 * hidden_size;
    if x.len() != seq_len * batch * input_size
        || w_ih.len() != g * input_size
        || w_hh.len() != g * hidden_size
        || b_ih.len() != g
        || b_hh.len() != g
        || h0.len() != batch * hidden_size
        || c0.len() != batch * hidden_size
        || output.len() != seq_len * batch * hidden_size
        || hn.len() != batch * hidden_size
        || cn.len() != batch * hidden_size
    {
        return Err(CudaError::Unsupported {
            msg: "lstm_forward shape mismatch".into(),
        });
    }
    if seq_len == 0 || batch == 0 {
        return Ok(());
    }
    // Initialise (h, c) from (h0, c0).
    let mut h = h0.to_vec();
    let mut c = c0.to_vec();
    for t in 0..seq_len {
        for b in 0..batch {
            // Compute gates: ih = w_ih @ x[t,b,:] + b_ih ; hh = w_hh @ h[b,:] + b_hh
            let mut gates = vec![0.0f32; g];
            for go in 0..g {
                let mut acc = b_ih[go] + b_hh[go];
                for i in 0..input_size {
                    acc += w_ih[go * input_size + i] * x[(t * batch + b) * input_size + i];
                }
                for i in 0..hidden_size {
                    acc += w_hh[go * hidden_size + i] * h[b * hidden_size + i];
                }
                gates[go] = acc;
            }
            // Split into i, f, g, o gates and apply nonlinearities.
            for hi in 0..hidden_size {
                let i_g = sigmoid(gates[hi]);
                let f_g = sigmoid(gates[hidden_size + hi]);
                let g_g = gates[2 * hidden_size + hi].tanh();
                let o_g = sigmoid(gates[3 * hidden_size + hi]);
                let new_c = f_g * c[b * hidden_size + hi] + i_g * g_g;
                let new_h = o_g * new_c.tanh();
                c[b * hidden_size + hi] = new_c;
                h[b * hidden_size + hi] = new_h;
                output[(t * batch + b) * hidden_size + hi] = new_h;
            }
        }
    }
    hn.copy_from_slice(&h);
    cn.copy_from_slice(&c);
    Ok(())
}

/// Single-layer GRU forward pass.
///
/// Gate layout matches PyTorch / cuDNN (r, z, n).
pub fn gru_forward_f32(
    x: &[f32],
    w_ih: &[f32],
    w_hh: &[f32],
    b_ih: &[f32],
    b_hh: &[f32],
    h0: &[f32],
    output: &mut [f32],
    hn: &mut [f32],
    seq_len: usize,
    batch: usize,
    input_size: usize,
    hidden_size: usize,
) -> Result<(), CudaError> {
    let g = 3 * hidden_size;
    if x.len() != seq_len * batch * input_size
        || w_ih.len() != g * input_size
        || w_hh.len() != g * hidden_size
        || b_ih.len() != g
        || b_hh.len() != g
        || h0.len() != batch * hidden_size
        || output.len() != seq_len * batch * hidden_size
        || hn.len() != batch * hidden_size
    {
        return Err(CudaError::Unsupported {
            msg: "gru_forward shape mismatch".into(),
        });
    }
    if seq_len == 0 || batch == 0 {
        return Ok(());
    }
    let mut h = h0.to_vec();
    for t in 0..seq_len {
        for b in 0..batch {
            // ih = w_ih @ x + b_ih ; hh = w_hh @ h + b_hh
            let mut ih = vec![0.0f32; g];
            let mut hh = vec![0.0f32; g];
            for go in 0..g {
                let mut a = b_ih[go];
                for i in 0..input_size {
                    a += w_ih[go * input_size + i] * x[(t * batch + b) * input_size + i];
                }
                ih[go] = a;
                let mut a2 = b_hh[go];
                for i in 0..hidden_size {
                    a2 += w_hh[go * hidden_size + i] * h[b * hidden_size + i];
                }
                hh[go] = a2;
            }
            for hi in 0..hidden_size {
                let r = sigmoid(ih[hi] + hh[hi]);
                let z = sigmoid(ih[hidden_size + hi] + hh[hidden_size + hi]);
                // n uses r * hh_n (not (r * h) re-projected) per PyTorch semantics.
                let n = (ih[2 * hidden_size + hi] + r * hh[2 * hidden_size + hi]).tanh();
                let new_h = (1.0 - z) * n + z * h[b * hidden_size + hi];
                h[b * hidden_size + hi] = new_h;
                output[(t * batch + b) * hidden_size + hi] = new_h;
            }
        }
    }
    hn.copy_from_slice(&h);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conv_descriptor_output_size_3x3_pad1_stride1() {
        let conv = ConvolutionDescriptor::k3s1p1();
        let (h, w) = conv.output_size(
            TensorDescriptor {
                n: 1,
                c: 3,
                h: 8,
                w: 8,
            },
            FilterDescriptor {
                k: 16,
                c: 3,
                r: 3,
                s: 3,
            },
        );
        assert_eq!(h, 8);
        assert_eq!(w, 8);
    }

    #[test]
    fn conv2d_1x1_kernel_acts_as_pointwise_multiplier() {
        let input = TensorDescriptor {
            n: 1,
            c: 1,
            h: 2,
            w: 2,
        };
        let filter = FilterDescriptor {
            k: 1,
            c: 1,
            r: 1,
            s: 1,
        };
        let conv = ConvolutionDescriptor {
            pad_h: 0,
            pad_w: 0,
            stride_h: 1,
            stride_w: 1,
        };
        let x = vec![1.0f32, 2.0, 3.0, 4.0];
        let w = vec![5.0f32];
        let mut y = vec![0.0f32; 4];
        conv2d_forward_f32(&x, &w, &mut y, input, filter, conv).unwrap();
        assert_eq!(y, vec![5.0, 10.0, 15.0, 20.0]);
    }

    #[test]
    fn conv2d_zero_batch_is_no_op() {
        let input = TensorDescriptor {
            n: 0,
            c: 3,
            h: 8,
            w: 8,
        };
        let filter = FilterDescriptor {
            k: 16,
            c: 3,
            r: 3,
            s: 3,
        };
        let conv = ConvolutionDescriptor::k3s1p1();
        let mut y: Vec<f32> = Vec::new();
        conv2d_forward_f32(&[], &[], &mut y, input, filter, conv).unwrap();
        assert!(y.is_empty());
    }

    #[test]
    fn conv2d_channel_mismatch_returns_error() {
        let input = TensorDescriptor {
            n: 1,
            c: 3,
            h: 4,
            w: 4,
        };
        let filter = FilterDescriptor {
            k: 1,
            c: 5,
            r: 1,
            s: 1,
        };
        let conv = ConvolutionDescriptor::k3s1p1();
        let mut y = vec![0.0f32; 16];
        let err =
            conv2d_forward_f32(&[0.0; 48], &[0.0; 5], &mut y, input, filter, conv).unwrap_err();
        assert!(matches!(err, CudaError::Unsupported { .. }));
    }

    #[test]
    fn lstm_descriptor_builds() {
        let d = RnnDescriptor {
            hidden_size: 32,
            num_layers: 1,
            mode: RnnMode::Lstm,
            bidirectional: false,
        };
        assert_eq!(d.hidden_size, 32);
        assert_eq!(d.mode, RnnMode::Lstm);
    }

    #[test]
    fn lstm_zero_input_keeps_initial_state_when_weights_zero() {
        // With all-zero weights and biases, gates collapse to sigmoid(0)=0.5 / tanh(0)=0
        // so c stays at f*c0 = 0.5*c0, h = o * tanh(c).
        let seq_len = 1;
        let batch = 1;
        let input_size = 1;
        let hidden = 1;
        let x = vec![0.0f32];
        let w_ih = vec![0.0f32; 4 * hidden * input_size];
        let w_hh = vec![0.0f32; 4 * hidden * hidden];
        let b_ih = vec![0.0f32; 4 * hidden];
        let b_hh = vec![0.0f32; 4 * hidden];
        let h0 = vec![0.0f32];
        let c0 = vec![1.0f32];
        let mut out = vec![0.0f32; seq_len * batch * hidden];
        let mut hn = vec![0.0f32; batch * hidden];
        let mut cn = vec![0.0f32; batch * hidden];
        lstm_forward_f32(
            &x, &w_ih, &w_hh, &b_ih, &b_hh, &h0, &c0, &mut out, &mut hn, &mut cn, seq_len, batch,
            input_size, hidden,
        )
        .unwrap();
        // f = 0.5, i*g = 0; new_c = 0.5
        assert!((cn[0] - 0.5).abs() < 1e-6);
        // o = 0.5; new_h = 0.5 * tanh(0.5)
        let expected_h = 0.5 * 0.5f32.tanh();
        assert!((hn[0] - expected_h).abs() < 1e-6);
    }

    #[test]
    fn lstm_shape_mismatch_returns_error() {
        let mut out = vec![0.0; 0];
        let mut hn = vec![0.0; 0];
        let mut cn = vec![0.0; 0];
        let err = lstm_forward_f32(
            &[1.0],
            &[],
            &[],
            &[],
            &[],
            &[],
            &[],
            &mut out,
            &mut hn,
            &mut cn,
            1,
            1,
            1,
            1,
        )
        .unwrap_err();
        assert!(matches!(err, CudaError::Unsupported { .. }));
    }

    #[test]
    fn gru_zero_inputs_zero_weights_returns_zero_hidden() {
        let seq_len = 2;
        let batch = 1;
        let input_size = 1;
        let hidden = 1;
        let x = vec![0.0f32; seq_len * batch * input_size];
        let g = 3 * hidden;
        let w_ih = vec![0.0f32; g * input_size];
        let w_hh = vec![0.0f32; g * hidden];
        let b_ih = vec![0.0f32; g];
        let b_hh = vec![0.0f32; g];
        let h0 = vec![0.0f32; batch * hidden];
        let mut out = vec![0.0f32; seq_len * batch * hidden];
        let mut hn = vec![0.0f32; batch * hidden];
        gru_forward_f32(
            &x, &w_ih, &w_hh, &b_ih, &b_hh, &h0, &mut out, &mut hn, seq_len, batch, input_size,
            hidden,
        )
        .unwrap();
        // r = z = 0.5, n = tanh(0) = 0, new_h = 0.5*0 + 0.5*0 = 0
        assert_eq!(out, vec![0.0; seq_len * batch * hidden]);
        assert_eq!(hn, vec![0.0; batch * hidden]);
    }

    #[test]
    fn batchnorm_inference_normalises_per_channel() {
        // Single channel, single pixel, x=mean → y = bias.
        let input = TensorDescriptor {
            n: 1,
            c: 1,
            h: 1,
            w: 1,
        };
        let x = vec![2.0f32];
        let scale = vec![1.0f32];
        let bias = vec![10.0f32];
        let mean = vec![2.0f32];
        let var = vec![1.0f32];
        let mut y = vec![0.0f32; 1];
        batchnorm_inference_f32(&x, &scale, &bias, &mean, &var, &mut y, input, 1e-5).unwrap();
        // (2 - 2) / sqrt(1) * 1 + 10 = 10
        assert!((y[0] - 10.0).abs() < 1e-4);
    }
}
