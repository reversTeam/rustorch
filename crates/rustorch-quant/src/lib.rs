//! # rustorch-quant
//!
//! Int8 dynamic quantization primitives for rustorch — observers,
//! quantization parameters, and quantize/dequantize ops.
//!
//! Currently exposed:
//! - [`dtype`] — `Int8` dtype variant + [`QParams`] (scale + zero
//!   point) struct.
//! - [`observer`] — [`MinMaxObserver`] (per-tensor + per-channel)
//!   and [`HistogramObserver`] for calibration.
//! - [`qops`] — [`quantize`] / [`dequantize`] with HALF_TO_EVEN
//!   rounding.
//!
//! Quant modules (`QuantLinear`, `QuantConv2d`), SIMD int8 kernels,
//! and the GPU path land in subsequent commits — see Phase 3 plan
//! `Quantization (Inference)`.

#![cfg_attr(docsrs, feature(doc_cfg))]
#![warn(missing_docs)]
#![warn(rust_2018_idioms)]

pub mod dtype;
pub mod gemm_int8;
pub mod gpu_int8;
pub mod observer;
pub mod qmodules;
pub mod qops;
pub mod workflow;

pub use dtype::QParams;
pub use gemm_int8::{gemm_f32_reference, gemm_int8_scalar, GemmError};
pub use observer::{HistogramObserver, MinMaxObserver, ObserverError, PerChannelMinMaxObserver};
pub use qmodules::{conv2d_output_size, Conv2dParams, QModuleError, QuantConv2d, QuantLinear};
pub use qops::{dequantize, quantize, QError};
pub use workflow::{
    f32_weight_bytes, int8_weight_bytes, quantize_model, LayerSpec, QuantLayer, WorkflowError,
};

/// Crate version reported at runtime.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn smoke_version_present() {
        assert!(!VERSION.is_empty());
    }
}
