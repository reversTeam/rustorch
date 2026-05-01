//! # rustorch-optim — Optimizers + LR schedulers (P1.7).
//!
//! v1 ships:
//! - [`Optimizer`] trait — `step`, `zero_grad`.
//! - [`Sgd`] — vanilla + momentum + Nesterov + weight_decay.
//! - [`Adam`] / [`AdamW`] — adaptive optimisers with bias correction.
//!
//! More optimisers (Lion, Adafactor, RMSprop, NAdam, RAdam, LBFGS,
//! Adadelta) and LR schedulers (StepLR, Cosine, WarmupCosine, OneCycle,
//! Plateau) plug into the same [`Optimizer`] trait — they remain pending
//! per the project plan.

#![cfg_attr(docsrs, feature(doc_cfg))]
#![warn(missing_docs)]
#![warn(rust_2018_idioms)]

pub mod adam;
pub mod sgd;

pub use adam::{Adam, AdamW};
pub use sgd::Sgd;

use rustorch_autograd::Variable;
use rustorch_core::tensor::tensor_impl::Tensor;

/// Common optimiser interface.
pub trait Optimizer {
    /// Apply one optimisation step using the gradients currently
    /// accumulated in each parameter's grad slot.
    fn step(&mut self);

    /// Zero out every parameter's gradient (set to None — same as
    /// PyTorch's `set_to_none=True`).
    fn zero_grad(&mut self);
}

/// Helper: write a fresh `Vec<f32>` into the parameter's shared
/// `data` slot. Used by SGD and the final update step of Adam/AdamW.
pub(crate) fn write_param_data(param: &Variable, new_data: Vec<f32>) {
    let shape = param.tensor().shape().to_vec();
    let new_t = Tensor::from_vec(shape, new_data).expect("optimiser write");
    param.set_data(new_t);
}
