//! # rustorch-autograd
//!
//! Tape-based reverse-mode automatic differentiation (RFC-0003, P1.5).
//!
//! Public surface:
//! - [`Variable`] — autograd-aware tensor wrapper.
//! - [`backward`] — run reverse-mode autodiff from a scalar output.
//! - [`no_grad`] / [`with_grad`] — thread-local guards toggling gradient
//!   tracking.
//! - [`ops`] — autograd-aware ops (`add`, `sub`, `mul`, `neg`, `matmul`,
//!   `relu`, `sigmoid`, `tanh`, `sum`, `mean`, `cross_entropy`, `mse_loss`).

#![cfg_attr(docsrs, feature(doc_cfg))]
#![warn(missing_docs)]
#![warn(rust_2018_idioms)]

pub mod anomaly;
pub mod backward;
pub mod node;
pub mod ops;
pub mod tape;
pub mod variable;

pub use anomaly::{
    check_or_panic, first_bad_value, is_detect_anomaly, set_detect_anomaly, DetectAnomalyGuard,
};
pub use backward::{backward, BackwardError};
pub use node::{Edge, Node};
pub use tape::{is_grad_enabled, no_grad, with_grad, NoGradGuard, WithGradGuard};
pub use variable::Variable;

use rustorch_core::tensor::dtype::Dtype;
use rustorch_core::tensor::tensor_impl::Tensor;

/// Build a tensor of ones with the same shape and dtype as `t`. Used
/// as the default initial gradient by [`backward`].
pub fn ones_like(t: &Tensor) -> Tensor {
    let n = t.numel();
    match t.dtype() {
        Dtype::F32 => Tensor::from_vec_typed::<f32, _>(t.shape().to_vec(), vec![1.0_f32; n])
            .expect("ones_like"),
        Dtype::F64 => Tensor::from_vec_typed::<f64, _>(t.shape().to_vec(), vec![1.0_f64; n])
            .expect("ones_like"),
        _ => panic!("ones_like only supports F32/F64"),
    }
}

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
