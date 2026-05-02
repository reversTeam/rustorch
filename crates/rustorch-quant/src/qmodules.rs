//! Quantised inference modules — `QuantLinear` and `QuantConv2d`.
//!
//! These are **inference-only** wrappers that hold pre-quantised
//! int8 weights plus a calibrated `QParams`, and dynamically
//! quantise their float input on each call. They DO NOT integrate
//! with the autograd tape — quantisation is inference territory.
//!
//! The bridge from `rustorch_nn::Linear` (a trained autograd module)
//! to `QuantLinear` (an int8 inference layer) is the
//! `quantize_model` calibration workflow, shipped in a separate
//! task. The data-plane API exposed here is what that workflow
//! produces.

use crate::dtype::QParams;
use crate::gemm_int8::{gemm_int8_scalar, GemmError};
use crate::observer::MinMaxObserver;
use crate::qops::quantize;

/// Errors raised by quantised modules.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QModuleError {
    /// GEMM-level failure (shape mismatch, etc).
    Gemm(GemmError),
    /// Input length doesn't match the layer's `in_features` * batch.
    InputShape {
        /// Expected input length = batch * in_features.
        expected: usize,
        /// Actual input length.
        got: usize,
    },
    /// Conv2d input dimensions don't fit the configured channels /
    /// height / width.
    ConvShape {
        /// What went wrong (free-form for now).
        msg: String,
    },
}

impl core::fmt::Display for QModuleError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            QModuleError::Gemm(e) => write!(f, "gemm error: {e}"),
            QModuleError::InputShape { expected, got } => {
                write!(f, "input length {got}, expected {expected}")
            },
            QModuleError::ConvShape { msg } => write!(f, "conv shape error: {msg}"),
        }
    }
}

impl std::error::Error for QModuleError {}

impl From<GemmError> for QModuleError {
    fn from(e: GemmError) -> Self {
        QModuleError::Gemm(e)
    }
}

/// Int8-quantised affine layer `y = x @ W + b`.
///
/// - Weight `[in_features, out_features]` stored as `i8` with
///   per-tensor `weight_qp`.
/// - Bias `[out_features]` stored as f32 (bias quantisation gives
///   marginal savings and risks accuracy loss — common practice).
/// - Input is quantised dynamically per call using the running
///   `input_observer`'s frozen QParams, OR from a caller-provided
///   `input_qp` for deterministic inference.
#[derive(Debug, Clone)]
pub struct QuantLinear {
    /// Quantised weights `[in_features, out_features]`.
    pub weight_q: Vec<i8>,
    /// Weight quantisation parameters (per-tensor symmetric).
    pub weight_qp: QParams,
    /// Bias `[out_features]` (f32).
    pub bias: Option<Vec<f32>>,
    /// Pre-frozen activation QParams (set after calibration).
    pub input_qp: Option<QParams>,
    /// In-flight observer used during calibration.
    pub input_observer: MinMaxObserver,
    in_features: usize,
    out_features: usize,
}

impl QuantLinear {
    /// Build a fresh QuantLinear from already-quantised weight bytes.
    /// Bias may be `None` for a no-bias layer.
    pub fn new(
        weight_q: Vec<i8>,
        weight_qp: QParams,
        bias: Option<Vec<f32>>,
        in_features: usize,
        out_features: usize,
    ) -> Self {
        assert_eq!(
            weight_q.len(),
            in_features * out_features,
            "weight_q.len() must equal in_features * out_features"
        );
        if let Some(b) = bias.as_ref() {
            assert_eq!(b.len(), out_features, "bias.len() must equal out_features");
        }
        Self {
            weight_q,
            weight_qp,
            bias,
            input_qp: None,
            input_observer: MinMaxObserver::new(),
            in_features,
            out_features,
        }
    }

    /// Build by quantising f32 weights with the provided QParams.
    pub fn from_f32_weights(
        weight_f32: &[f32],
        weight_qp: QParams,
        bias: Option<Vec<f32>>,
        in_features: usize,
        out_features: usize,
    ) -> Self {
        let mut weight_q = vec![0i8; weight_f32.len()];
        quantize(weight_f32, &mut weight_q, weight_qp).expect("quantise weights");
        Self::new(weight_q, weight_qp, bias, in_features, out_features)
    }

    /// Input feature count.
    pub fn in_features(&self) -> usize {
        self.in_features
    }

    /// Output feature count.
    pub fn out_features(&self) -> usize {
        self.out_features
    }

    /// Calibrate by observing one batch of activations. After enough
    /// batches, call `freeze_input_qp` to lock in the input scale.
    pub fn observe(&mut self, input: &[f32]) {
        self.input_observer.update(input);
    }

    /// Freeze the input QParams from the current observer state.
    /// Returns the chosen QParams.
    pub fn freeze_input_qp(&mut self) -> Result<QParams, QModuleError> {
        let qp = self
            .input_observer
            .freeze()
            .map_err(|e| QModuleError::ConvShape {
                msg: format!("input observer empty: {e}"),
            })?;
        self.input_qp = Some(qp);
        Ok(qp)
    }

    /// Forward pass. `input` is `[batch, in_features]` f32 (row-major);
    /// `output` is `[batch, out_features]` f32 (row-major). The
    /// caller passes `input_qp` (typically the value frozen at
    /// calibration time); if `None`, falls back to `self.input_qp`.
    pub fn forward(
        &self,
        input: &[f32],
        output: &mut [f32],
        batch: usize,
        input_qp: Option<QParams>,
    ) -> Result<(), QModuleError> {
        if input.len() != batch * self.in_features {
            return Err(QModuleError::InputShape {
                expected: batch * self.in_features,
                got: input.len(),
            });
        }
        if output.len() != batch * self.out_features {
            return Err(QModuleError::InputShape {
                expected: batch * self.out_features,
                got: output.len(),
            });
        }
        let qp = input_qp.or(self.input_qp).unwrap_or(QParams::IDENTITY);
        // Dynamically quantise input to int8.
        let mut input_q = vec![0i8; input.len()];
        quantize(input, &mut input_q, qp).map_err(|e| QModuleError::ConvShape {
            msg: format!("quantise input: {e}"),
        })?;
        // GEMM: input_q [batch, in] × weight_q [in, out] = output [batch, out].
        gemm_int8_scalar(
            &input_q,
            &self.weight_q,
            output,
            batch,
            self.in_features,
            self.out_features,
            qp,
            self.weight_qp,
        )?;
        // Add bias (f32) if present.
        if let Some(bias) = &self.bias {
            for b_idx in 0..batch {
                for o in 0..self.out_features {
                    output[b_idx * self.out_features + o] += bias[o];
                }
            }
        }
        Ok(())
    }
}

/// Int8-quantised 2D convolution via im2col + int8 GEMM.
///
/// Layout convention:
/// - Input  `[N, Cin, H, W]` f32 row-major
/// - Weight `[Cout, Cin, KH, KW]` int8 row-major
/// - Output `[N, Cout, H_out, W_out]` f32 row-major
///
/// Stride / padding / dilation default to (1, 1, 0, 1) in
/// `Conv2dParams::default` — typical 1×1 / 3×3 / 5×5 conv shapes
/// work out of the box.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Conv2dParams {
    /// Kernel height.
    pub kh: usize,
    /// Kernel width.
    pub kw: usize,
    /// Stride (h, w).
    pub stride: (usize, usize),
    /// Padding (h, w).
    pub padding: (usize, usize),
}

impl Conv2dParams {
    /// 3×3 stride 1 padding 1 (the most common conv).
    pub fn k3s1p1() -> Self {
        Self {
            kh: 3,
            kw: 3,
            stride: (1, 1),
            padding: (1, 1),
        }
    }
}

/// Compute conv2d output spatial dims.
pub fn conv2d_output_size(h_in: usize, w_in: usize, params: Conv2dParams) -> (usize, usize) {
    let h_out = (h_in + 2 * params.padding.0 - params.kh) / params.stride.0 + 1;
    let w_out = (w_in + 2 * params.padding.1 - params.kw) / params.stride.1 + 1;
    (h_out, w_out)
}

/// Quantised 2D convolution. Holds pre-quantised weights `[Cout,
/// Cin*KH*KW]` flattened (post-im2col layout) and the original
/// shapes for reference.
#[derive(Debug, Clone)]
pub struct QuantConv2d {
    /// Pre-quantised weights, `[Cout, Cin * KH * KW]` row-major.
    pub weight_q: Vec<i8>,
    /// Weight QParams.
    pub weight_qp: QParams,
    /// Optional bias `[Cout]` f32.
    pub bias: Option<Vec<f32>>,
    /// Channels-out.
    pub c_out: usize,
    /// Channels-in.
    pub c_in: usize,
    /// Convolution parameters.
    pub params: Conv2dParams,
}

impl QuantConv2d {
    /// Build from f32 weights `[Cout, Cin, KH, KW]` and a QParams.
    /// Internally quantises and flattens to `[Cout, Cin*KH*KW]` for
    /// GEMM consumption.
    pub fn from_f32_weights(
        weight_f32: &[f32],
        weight_qp: QParams,
        bias: Option<Vec<f32>>,
        c_out: usize,
        c_in: usize,
        params: Conv2dParams,
    ) -> Self {
        let kw = params.kw;
        let kh = params.kh;
        assert_eq!(weight_f32.len(), c_out * c_in * kh * kw);
        let mut weight_q = vec![0i8; weight_f32.len()];
        quantize(weight_f32, &mut weight_q, weight_qp).expect("quantise weights");
        Self {
            weight_q,
            weight_qp,
            bias,
            c_out,
            c_in,
            params,
        }
    }

    /// Forward pass. `input` is `[N, Cin, H, W]`; `output` is `[N,
    /// Cout, H_out, W_out]`.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        input: &[f32],
        output: &mut [f32],
        n: usize,
        h_in: usize,
        w_in: usize,
        input_qp: QParams,
    ) -> Result<(), QModuleError> {
        let expected_in = n * self.c_in * h_in * w_in;
        if input.len() != expected_in {
            return Err(QModuleError::InputShape {
                expected: expected_in,
                got: input.len(),
            });
        }
        let (h_out, w_out) = conv2d_output_size(h_in, w_in, self.params);
        let expected_out = n * self.c_out * h_out * w_out;
        if output.len() != expected_out {
            return Err(QModuleError::InputShape {
                expected: expected_out,
                got: output.len(),
            });
        }
        let kh = self.params.kh;
        let kw = self.params.kw;
        let cols_per_position = self.c_in * kh * kw;

        // Per-batch im2col + GEMM.
        for n_idx in 0..n {
            // im2col: build a [H_out*W_out, Cin*KH*KW] matrix in f32.
            let mut col = vec![0.0f32; h_out * w_out * cols_per_position];
            for h_o in 0..h_out {
                for w_o in 0..w_out {
                    for c in 0..self.c_in {
                        for r in 0..kh {
                            for s in 0..kw {
                                let h_i = (h_o * self.params.stride.0 + r) as isize
                                    - self.params.padding.0 as isize;
                                let w_i = (w_o * self.params.stride.1 + s) as isize
                                    - self.params.padding.1 as isize;
                                let val = if h_i >= 0
                                    && h_i < h_in as isize
                                    && w_i >= 0
                                    && w_i < w_in as isize
                                {
                                    let in_off = ((n_idx * self.c_in + c) * h_in + h_i as usize)
                                        * w_in
                                        + w_i as usize;
                                    input[in_off]
                                } else {
                                    0.0
                                };
                                let col_off = ((h_o * w_out + w_o) * cols_per_position)
                                    + (c * kh * kw + r * kw + s);
                                col[col_off] = val;
                            }
                        }
                    }
                }
            }
            // Quantise the col matrix.
            let mut col_q = vec![0i8; col.len()];
            quantize(&col, &mut col_q, input_qp).map_err(|e| QModuleError::ConvShape {
                msg: format!("quantise col: {e}"),
            })?;
            // GEMM: col_q [H_out*W_out, K] × weight_q^T [K, Cout]
            // We have weight_q laid out as [Cout, K]; need [K, Cout].
            // Transpose on the fly into a temporary buffer (small).
            let m = h_out * w_out;
            let k_dim = cols_per_position;
            let n_out = self.c_out;
            let mut weight_t = vec![0i8; k_dim * n_out];
            for c_o in 0..n_out {
                for kk in 0..k_dim {
                    weight_t[kk * n_out + c_o] = self.weight_q[c_o * k_dim + kk];
                }
            }
            // Output for this batch: [H_out*W_out, Cout] in m*n_out
            // layout, then transposed to [Cout, H_out*W_out] for the
            // (N, Cout, H_out, W_out) output buffer.
            let mut gemm_out = vec![0.0f32; m * n_out];
            gemm_int8_scalar(
                &col_q,
                &weight_t,
                &mut gemm_out,
                m,
                k_dim,
                n_out,
                input_qp,
                self.weight_qp,
            )?;
            // Transpose [m, n_out] → [n_out, m] into output slab
            // for batch n_idx + apply bias if present.
            for c_o in 0..n_out {
                for hw in 0..m {
                    let bias_v = self.bias.as_ref().map(|b| b[c_o]).unwrap_or(0.0);
                    let dst = ((n_idx * n_out + c_o) * h_out * w_out) + hw;
                    output[dst] = gemm_out[hw * n_out + c_o] + bias_v;
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f32, b: f32, atol: f32) -> bool {
        (a - b).abs() <= atol + 1e-3 * a.abs().max(b.abs())
    }

    // --- QuantLinear tests -----------------------------------------

    #[test]
    fn quant_linear_no_bias_matches_f32_within_quant_error() {
        // 2-batch, 4-in, 3-out
        let in_features = 4;
        let out_features = 3;
        let weight_f32: Vec<f32> = (0..in_features * out_features)
            .map(|i| (i as f32 * 0.1) - 0.5)
            .collect();
        let input_f32 = vec![0.5f32, -0.3, 0.1, 0.7, -0.2, 0.4, 0.0, -0.5];

        // f32 reference.
        let mut ref_out = vec![0.0f32; 2 * out_features];
        for b in 0..2 {
            for o in 0..out_features {
                let mut acc = 0.0f32;
                for i in 0..in_features {
                    acc += input_f32[b * in_features + i] * weight_f32[i * out_features + o];
                }
                ref_out[b * out_features + o] = acc;
            }
        }

        // Quantise weights.
        let w_min = weight_f32.iter().copied().fold(f32::INFINITY, f32::min);
        let w_max = weight_f32.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let w_qp = QParams::from_min_max(w_min, w_max).unwrap();
        let layer =
            QuantLinear::from_f32_weights(&weight_f32, w_qp, None, in_features, out_features);

        // Quantise inputs from observed range.
        let in_min = input_f32.iter().copied().fold(f32::INFINITY, f32::min);
        let in_max = input_f32.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let i_qp = QParams::from_min_max(in_min, in_max).unwrap();

        let mut q_out = vec![0.0f32; 2 * out_features];
        layer
            .forward(&input_f32, &mut q_out, 2, Some(i_qp))
            .unwrap();

        // Quantisation error bound.
        let max_abs = w_max.abs().max(in_max.abs());
        let bound = in_features as f32 * (w_qp.scale + i_qp.scale) * max_abs * 4.0;
        for (q, r) in q_out.iter().zip(ref_out.iter()) {
            assert!(
                close(*q, *r, bound),
                "q={q} r={r} diff={} bound={bound}",
                (q - r).abs()
            );
        }
    }

    #[test]
    fn quant_linear_with_bias_adds_after_gemm() {
        let in_features = 2;
        let out_features = 2;
        // Identity-like weights: W = [[1, 0], [0, 1]].
        let weight_f32 = vec![1.0_f32, 0.0, 0.0, 1.0];
        let bias = Some(vec![10.0_f32, 20.0]);
        let w_qp = QParams::from_min_max(0.0, 1.0).unwrap();
        let layer =
            QuantLinear::from_f32_weights(&weight_f32, w_qp, bias, in_features, out_features);
        let input = vec![1.0_f32, 2.0]; // batch 1
        let mut out = vec![0.0_f32; 2];
        let i_qp = QParams::from_min_max(0.0, 2.0).unwrap();
        layer.forward(&input, &mut out, 1, Some(i_qp)).unwrap();
        // Expected approximately [1+10, 2+20] = [11, 22], within
        // quant noise.
        assert!((out[0] - 11.0).abs() < 0.5, "out[0]={}", out[0]);
        assert!((out[1] - 22.0).abs() < 0.5, "out[1]={}", out[1]);
    }

    #[test]
    fn quant_linear_input_shape_mismatch() {
        let layer = QuantLinear::from_f32_weights(&[0.1f32; 12], QParams::IDENTITY, None, 4, 3);
        let input = vec![1.0_f32; 5]; // wrong: should be batch * 4
        let mut out = vec![0.0_f32; 3];
        let err = layer.forward(&input, &mut out, 1, None).unwrap_err();
        assert!(matches!(
            err,
            QModuleError::InputShape {
                expected: 4,
                got: 5
            }
        ));
    }

    #[test]
    fn quant_linear_observer_calibration_freezes_qp() {
        let mut layer = QuantLinear::from_f32_weights(&[0.1f32; 12], QParams::IDENTITY, None, 4, 3);
        layer.observe(&[1.0, -2.0, 0.5, 0.0]);
        layer.observe(&[0.7, -1.5, 2.5, 1.0]);
        let qp = layer.freeze_input_qp().unwrap();
        // abs_max = 2.5 → scale = 2.5/127.
        assert!((qp.scale - 2.5 / 127.0).abs() < 1e-7);
        assert!(layer.input_qp.is_some());
    }

    #[test]
    fn quant_linear_empty_observer_freeze_errors() {
        let mut layer = QuantLinear::from_f32_weights(&[0.1f32; 12], QParams::IDENTITY, None, 4, 3);
        let err = layer.freeze_input_qp().unwrap_err();
        assert!(matches!(err, QModuleError::ConvShape { .. }));
    }

    // --- QuantConv2d tests -----------------------------------------

    #[test]
    fn conv2d_output_size_3x3_pad1_stride1() {
        let (h, w) = conv2d_output_size(8, 8, Conv2dParams::k3s1p1());
        assert_eq!(h, 8);
        assert_eq!(w, 8);
    }

    #[test]
    fn conv2d_output_size_3x3_pad0_stride1() {
        let (h, w) = conv2d_output_size(
            8,
            8,
            Conv2dParams {
                kh: 3,
                kw: 3,
                stride: (1, 1),
                padding: (0, 0),
            },
        );
        assert_eq!(h, 6);
        assert_eq!(w, 6);
    }

    #[test]
    fn quant_conv2d_1x1_kernel_acts_like_pointwise_linear() {
        // 1×1 kernel = pointwise channel mixing. Test on a tiny
        // [1, 2, 2, 2] input with [2, 2, 1, 1] weights.
        let c_in = 2;
        let c_out = 2;
        let h_in = 2;
        let w_in = 2;
        let n = 1;
        let params = Conv2dParams {
            kh: 1,
            kw: 1,
            stride: (1, 1),
            padding: (0, 0),
        };
        // Identity-ish weights: out[0] = in[0], out[1] = in[1].
        let weight_f32 = vec![1.0_f32, 0.0, 0.0, 1.0];
        let w_qp = QParams::from_min_max(0.0, 1.0).unwrap();
        let layer = QuantConv2d::from_f32_weights(&weight_f32, w_qp, None, c_out, c_in, params);
        // Input: [1, 2, 2, 2] = 8 values. channel 0 = [10, 20, 30, 40], channel 1 = [-1, -2, -3, -4].
        let input = vec![10.0_f32, 20.0, 30.0, 40.0, -1.0, -2.0, -3.0, -4.0];
        let (h_out, w_out) = conv2d_output_size(h_in, w_in, params);
        assert_eq!(h_out, 2);
        assert_eq!(w_out, 2);
        let mut out = vec![0.0_f32; n * c_out * h_out * w_out];
        let i_qp = QParams::from_min_max(-4.0, 40.0).unwrap();
        layer
            .forward(&input, &mut out, n, h_in, w_in, i_qp)
            .unwrap();
        // Within quant error, out[channel 0] ≈ in[channel 0] = [10, 20, 30, 40].
        for (q, r) in out.iter().take(4).zip(&[10.0_f32, 20.0, 30.0, 40.0]) {
            assert!((q - r).abs() < 1.0, "q={q} r={r}");
        }
        for (q, r) in out.iter().skip(4).zip(&[-1.0_f32, -2.0, -3.0, -4.0]) {
            assert!((q - r).abs() < 1.0, "q={q} r={r}");
        }
    }

    #[test]
    fn quant_conv2d_input_shape_mismatch() {
        let params = Conv2dParams::k3s1p1();
        let weight_f32 = vec![0.1_f32; 2 * 2 * 3 * 3];
        let w_qp = QParams::from_min_max(0.0, 1.0).unwrap();
        let layer = QuantConv2d::from_f32_weights(&weight_f32, w_qp, None, 2, 2, params);
        let input = vec![0.0_f32; 5]; // wrong
        let mut out = vec![0.0_f32; 2 * 2 * 8 * 8];
        let err = layer
            .forward(&input, &mut out, 1, 8, 8, QParams::IDENTITY)
            .unwrap_err();
        assert!(matches!(err, QModuleError::InputShape { .. }));
    }
}
