//! Loss-function modules.
//!
//! Loss modules don't fit the single-input `Module::forward(input)`
//! signature (they take both `input` and `target`), so they implement a
//! sister trait [`Criterion`]. They're still building blocks of the
//! training loop, just outside [`Sequential`].
//!
//! Available criteria:
//! - [`MseLoss`] — mean-squared-error regression loss
//! - [`CrossEntropyLoss`] — softmax + NLL classification with class
//!   targets (I64)
//!
//! Numerical stability and reduction semantics are inherited from the
//! underlying autograd / cpu-backend kernels.
//!
//! `NllLoss` and `BceWithLogitsLoss` are pending an autograd-aware
//! kernel hook-up — see plan P1.5 follow-up.

use crate::module::ModuleError;
use rustorch_autograd::{ops, Variable};
use rustorch_cpu::backend::Reduction;

/// Common interface for loss modules. Subset of `nn.Module` semantics
/// but accepting `(input, target)` as separate arguments.
pub trait Criterion: Send + Sync {
    /// Compute the loss given prediction and target.
    fn forward(&self, input: &Variable, target: &Variable) -> Result<Variable, ModuleError>;
}

// ------------------------------ MseLoss ------------------------------

/// Mean Squared Error loss with configurable reduction.
pub struct MseLoss {
    reduction: Reduction,
}

impl MseLoss {
    /// Build with the given reduction.
    pub fn new(reduction: Reduction) -> Self {
        MseLoss { reduction }
    }
}

impl Default for MseLoss {
    fn default() -> Self {
        MseLoss {
            reduction: Reduction::Mean,
        }
    }
}

impl Criterion for MseLoss {
    fn forward(&self, input: &Variable, target: &Variable) -> Result<Variable, ModuleError> {
        ops::mse_loss(input, target, self.reduction)
    }
}

// ------------------------------ CrossEntropyLoss ------------------------------

/// Cross-Entropy loss = log_softmax + NLL with class targets (I64).
pub struct CrossEntropyLoss {
    reduction: Reduction,
}

impl CrossEntropyLoss {
    /// Build with the given reduction.
    pub fn new(reduction: Reduction) -> Self {
        CrossEntropyLoss { reduction }
    }
}

impl Default for CrossEntropyLoss {
    fn default() -> Self {
        CrossEntropyLoss {
            reduction: Reduction::Mean,
        }
    }
}

impl Criterion for CrossEntropyLoss {
    fn forward(&self, input: &Variable, target: &Variable) -> Result<Variable, ModuleError> {
        ops::cross_entropy(input, target, self.reduction)
    }
}

// ------------------------------ CTCLoss ------------------------------

/// CTC loss wrapper around the cpu kernel. v1 is **forward only** —
/// returns a Variable that does NOT track gradient. Useful for
/// metric reporting; differentiable training requires the alpha-beta
/// backward (pending follow-up).
pub struct CTCLoss {
    reduction: Reduction,
    /// Per-sample input lengths.
    pub input_lens: Vec<i64>,
    /// Per-sample target lengths.
    pub target_lens: Vec<i64>,
}

impl CTCLoss {
    /// Build with input + target lengths (len = batch size) and reduction.
    pub fn new(input_lens: Vec<i64>, target_lens: Vec<i64>, reduction: Reduction) -> Self {
        CTCLoss {
            reduction,
            input_lens,
            target_lens,
        }
    }
}

impl Criterion for CTCLoss {
    fn forward(&self, log_probs: &Variable, targets: &Variable) -> Result<Variable, ModuleError> {
        let loss = rustorch_cpu::kernels::ctc::ctc_loss_forward(
            &log_probs.tensor(),
            &targets.tensor(),
            &self.input_lens,
            &self.target_lens,
            self.reduction,
        )
        .map_err(|e| rustorch_autograd::BackwardError::Backend {
            op: "ctc_loss",
            message: e.to_string(),
        })?;
        Ok(Variable::new(loss))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustorch_core::tensor::tensor_impl::Tensor;

    fn var(shape: impl Into<Vec<usize>>, data: Vec<f32>) -> Variable {
        Variable::new(Tensor::from_vec(shape.into(), data).unwrap())
    }

    #[test]
    fn mse_default_is_mean() {
        let loss = MseLoss::default();
        let pred = var(vec![2usize, 2], vec![1.0, 2.0, 3.0, 4.0]);
        let tgt = var(vec![2usize, 2], vec![1.0, 2.0, 3.0, 4.0]);
        let l = loss.forward(&pred, &tgt).unwrap();
        let v = l.tensor().as_slice::<f32>().unwrap()[0];
        assert!(v.abs() < 1e-7, "perfect prediction → loss 0, got {v}");
    }

    #[test]
    fn mse_reduction_sum() {
        let loss = MseLoss::new(Reduction::Sum);
        let pred = var(vec![3usize], vec![0.0, 0.0, 0.0]);
        let tgt = var(vec![3usize], vec![1.0, 2.0, 3.0]);
        let l = loss.forward(&pred, &tgt).unwrap();
        // (1 + 4 + 9) = 14
        assert!((l.tensor().as_slice::<f32>().unwrap()[0] - 14.0).abs() < 1e-5);
    }

    #[test]
    fn cross_entropy_perfect_prediction_low_loss() {
        let loss = CrossEntropyLoss::default();
        // 2 classes; one-hot-like logits (large gap → near-perfect prob)
        let pred = var(vec![1usize, 2], vec![10.0_f32, -10.0]);
        let tgt =
            Variable::new(Tensor::from_vec_typed::<i64, _>(vec![1usize], vec![0_i64]).unwrap());
        let l = loss.forward(&pred, &tgt).unwrap();
        let v = l.tensor().as_slice::<f32>().unwrap()[0];
        assert!(v < 1e-3, "very confident correct → near-zero loss, got {v}");
    }
}
