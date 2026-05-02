//! Calibration workflow: walk a model, run calibration batches,
//! freeze QParams, swap layers to int8 variants.
//!
//! Phase 3 task `Calibration workflow + end-to-end inference
//! validation` (dbb57331). The workflow is layer-list-driven (not
//! Module-trait-driven) to keep rustorch-quant decoupled from
//! rustorch-nn — see decision attached to QuantLinear/Conv2d.

use crate::dtype::QParams;
use crate::observer::MinMaxObserver;
use crate::qmodules::{Conv2dParams, QModuleError, QuantConv2d, QuantLinear};

/// A layer spec the workflow knows how to quantise.
#[derive(Debug, Clone)]
pub enum LayerSpec {
    /// Linear: `[in_features, out_features]` weights + optional bias.
    Linear {
        /// f32 weights `[in_features * out_features]`.
        weight: Vec<f32>,
        /// Optional bias `[out_features]`.
        bias: Option<Vec<f32>>,
        /// in_features.
        in_features: usize,
        /// out_features.
        out_features: usize,
    },
    /// Conv2d: `[Cout, Cin, KH, KW]` weights + optional bias.
    Conv2d {
        /// f32 weights flattened in `[Cout, Cin, KH, KW]` row-major.
        weight: Vec<f32>,
        /// Optional bias `[Cout]`.
        bias: Option<Vec<f32>>,
        /// Cout.
        c_out: usize,
        /// Cin.
        c_in: usize,
        /// Conv params.
        params: Conv2dParams,
    },
}

/// Output of [`quantize_model`]: parallel list to the input layer
/// specs, each replaced with a quantised module.
#[derive(Debug, Clone)]
pub enum QuantLayer {
    /// Quantised linear layer.
    Linear(QuantLinear),
    /// Quantised conv2d layer.
    Conv2d(QuantConv2d),
}

/// Workflow errors.
#[derive(Debug, Clone, PartialEq)]
pub enum WorkflowError {
    /// Calibration loader yielded zero batches.
    EmptyCalibrationLoader,
    /// Layer count doesn't match calibration data shape.
    LayerCountMismatch {
        /// Expected count.
        expected: usize,
        /// Actual count.
        got: usize,
    },
    /// Per-layer module error.
    Module(QModuleError),
}

impl core::fmt::Display for WorkflowError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            WorkflowError::EmptyCalibrationLoader => {
                write!(f, "calibration loader yielded zero batches")
            },
            WorkflowError::LayerCountMismatch { expected, got } => {
                write!(f, "layer count mismatch: expected {expected}, got {got}")
            },
            WorkflowError::Module(e) => write!(f, "module error: {e}"),
        }
    }
}

impl std::error::Error for WorkflowError {}

impl From<QModuleError> for WorkflowError {
    fn from(e: QModuleError) -> Self {
        WorkflowError::Module(e)
    }
}

/// Quantise a list of f32 layers using the provided list of
/// per-layer activation observations (one observer per layer,
/// pre-fed by the caller through the calibration loader).
///
/// Returns parallel `Vec<QuantLayer>` + per-layer activation
/// QParams (vec length == layers.len()).
///
/// The caller is responsible for running activations through the
/// f32 model FIRST during calibration and feeding the input to each
/// layer into the corresponding observer. This keeps the workflow
/// independent of the autograd / Module machinery.
pub fn quantize_model(
    layers: &[LayerSpec],
    activation_observers: &[MinMaxObserver],
) -> Result<(Vec<QuantLayer>, Vec<QParams>), WorkflowError> {
    if layers.len() != activation_observers.len() {
        return Err(WorkflowError::LayerCountMismatch {
            expected: layers.len(),
            got: activation_observers.len(),
        });
    }
    let mut q_layers: Vec<QuantLayer> = Vec::with_capacity(layers.len());
    let mut activation_qps: Vec<QParams> = Vec::with_capacity(layers.len());
    for (layer, observer) in layers.iter().zip(activation_observers.iter()) {
        // Freeze the activation observer; if it saw no finite data,
        // fall back to IDENTITY (caller can detect via the seen()
        // count if they care).
        let act_qp = observer.freeze().unwrap_or(QParams::IDENTITY);
        activation_qps.push(act_qp);
        match layer {
            LayerSpec::Linear {
                weight,
                bias,
                in_features,
                out_features,
            } => {
                let mut w_obs = MinMaxObserver::new();
                w_obs.update(weight);
                let w_qp = w_obs.freeze().unwrap_or(QParams::IDENTITY);
                let mut q = QuantLinear::from_f32_weights(
                    weight,
                    w_qp,
                    bias.clone(),
                    *in_features,
                    *out_features,
                );
                q.input_qp = Some(act_qp);
                q_layers.push(QuantLayer::Linear(q));
            },
            LayerSpec::Conv2d {
                weight,
                bias,
                c_out,
                c_in,
                params,
            } => {
                let mut w_obs = MinMaxObserver::new();
                w_obs.update(weight);
                let w_qp = w_obs.freeze().unwrap_or(QParams::IDENTITY);
                let q = QuantConv2d::from_f32_weights(
                    weight,
                    w_qp,
                    bias.clone(),
                    *c_out,
                    *c_in,
                    *params,
                );
                q_layers.push(QuantLayer::Conv2d(q));
            },
        }
    }
    Ok((q_layers, activation_qps))
}

/// Total bytes the original f32 layer weights occupy.
pub fn f32_weight_bytes(layers: &[LayerSpec]) -> usize {
    layers
        .iter()
        .map(|l| match l {
            LayerSpec::Linear { weight, .. } => weight.len() * 4,
            LayerSpec::Conv2d { weight, .. } => weight.len() * 4,
        })
        .sum()
}

/// Total bytes the int8 quantised weights occupy.
pub fn int8_weight_bytes(q_layers: &[QuantLayer]) -> usize {
    q_layers
        .iter()
        .map(|l| match l {
            QuantLayer::Linear(q) => q.weight_q.len(),
            QuantLayer::Conv2d(q) => q.weight_q.len(),
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture_linear(seed: u32, in_f: usize, out_f: usize) -> LayerSpec {
        let mut s = seed.wrapping_mul(2654435761);
        let weight: Vec<f32> = (0..in_f * out_f)
            .map(|_| {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                (s as f32 / u32::MAX as f32 - 0.5) * 0.1
            })
            .collect();
        LayerSpec::Linear {
            weight,
            bias: Some(vec![0.0; out_f]),
            in_features: in_f,
            out_features: out_f,
        }
    }

    #[test]
    fn quantize_model_swaps_linear_to_quant_linear() {
        let layers = vec![fixture_linear(1, 8, 4), fixture_linear(2, 4, 2)];
        let mut obs0 = MinMaxObserver::new();
        obs0.update(&[1.0, -1.0, 0.5]);
        let mut obs1 = MinMaxObserver::new();
        obs1.update(&[2.0, -2.0]);
        let observers = vec![obs0, obs1];
        let (q_layers, act_qps) = quantize_model(&layers, &observers).unwrap();
        assert_eq!(q_layers.len(), 2);
        assert_eq!(act_qps.len(), 2);
        assert!(matches!(q_layers[0], QuantLayer::Linear(_)));
        assert!(matches!(q_layers[1], QuantLayer::Linear(_)));
    }

    #[test]
    fn quantize_model_layer_count_mismatch_returns_error() {
        let layers = vec![fixture_linear(1, 8, 4)];
        let observers = vec![MinMaxObserver::new(), MinMaxObserver::new()];
        let err = quantize_model(&layers, &observers).unwrap_err();
        assert!(matches!(err, WorkflowError::LayerCountMismatch { .. }));
    }

    #[test]
    fn empty_observer_falls_back_to_identity() {
        let layers = vec![fixture_linear(1, 8, 4)];
        let observers = vec![MinMaxObserver::new()]; // never .update'd
        let (q_layers, act_qps) = quantize_model(&layers, &observers).unwrap();
        assert_eq!(act_qps[0], QParams::IDENTITY);
        // Quantised layer still produced (so caller can branch).
        assert_eq!(q_layers.len(), 1);
    }

    #[test]
    fn weight_byte_accounting_4x_smaller() {
        // 100 KB f32 weights → 25 KB int8.
        let layers = vec![fixture_linear(1, 100, 256)]; // 25 600 weights
        let mut obs = MinMaxObserver::new();
        obs.update(&[0.1; 100]);
        let observers = vec![obs];
        let (q_layers, _) = quantize_model(&layers, &observers).unwrap();
        let f32_bytes = f32_weight_bytes(&layers);
        let int8_bytes = int8_weight_bytes(&q_layers);
        assert_eq!(f32_bytes, 25_600 * 4);
        assert_eq!(int8_bytes, 25_600);
        assert_eq!(f32_bytes / int8_bytes, 4);
    }

    #[test]
    fn quantize_model_supports_mixed_linear_and_conv2d() {
        let conv_weight = vec![0.1f32; 2 * 2 * 3 * 3]; // [Cout=2, Cin=2, KH=3, KW=3]
        let layers = vec![
            LayerSpec::Conv2d {
                weight: conv_weight,
                bias: Some(vec![0.0; 2]),
                c_out: 2,
                c_in: 2,
                params: Conv2dParams::k3s1p1(),
            },
            fixture_linear(1, 8, 4),
        ];
        let mut obs0 = MinMaxObserver::new();
        obs0.update(&[1.0, -0.5, 0.3]);
        let mut obs1 = MinMaxObserver::new();
        obs1.update(&[0.5, -0.5]);
        let (q_layers, _) = quantize_model(&layers, &[obs0, obs1]).unwrap();
        assert!(matches!(q_layers[0], QuantLayer::Conv2d(_)));
        assert!(matches!(q_layers[1], QuantLayer::Linear(_)));
    }

    #[test]
    fn nan_observations_skipped_silently() {
        let layers = vec![fixture_linear(1, 4, 2)];
        let mut obs = MinMaxObserver::new();
        obs.update(&[1.0, f32::NAN, -1.0]);
        let (_, act_qps) = quantize_model(&layers, &[obs]).unwrap();
        // QPs are derived from the FINITE values only; range is [-1, 1].
        assert!(act_qps[0].scale > 0.0 && act_qps[0].scale.is_finite());
    }
}
