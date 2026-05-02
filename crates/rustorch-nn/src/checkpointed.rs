//! [`Checkpointed`] — module wrapper that gradient-checkpoints
//! the inner module's forward pass.
//!
//! ```ignore
//! use rustorch_nn::{Checkpointed, Linear, Sequential};
//!
//! // Wrap each block of a deep stack so its forward activations are
//! // dropped after the forward pass and re-computed on backward.
//! let net = Sequential::new()
//!     .add(Checkpointed::new(TransformerBlock::new(...)))
//!     .add(Checkpointed::new(TransformerBlock::new(...)))
//!     .add(Checkpointed::new(TransformerBlock::new(...)));
//! ```
//!
//! Trade-off: ~+30% compute (one extra forward pass per block) for
//! roughly halving the activation memory of a deep stack. Practical
//! sweet-spot for fitting Llama-class models on a single 24 GB GPU.
//!
//! v1 caveats:
//! - The inner module must be stateless across forward calls
//!   (Dropout / BatchNorm in train mode would re-sample on the
//!   backward pass — same caveat as PyTorch's
//!   `torch.utils.checkpoint`).
//! - `train()` / `eval()` on the wrapper are no-ops; switch the inner
//!   module's mode *before* wrapping. Driven by the `Arc<M>` storage,
//!   which precludes `&mut self` access.

use crate::module::{Module, ModuleError};
use rustorch_autograd::Variable;
use std::sync::Arc;

/// Module wrapper that re-runs the inner forward pass during backward
/// instead of caching activations on the autograd tape.
pub struct Checkpointed<M: Module + 'static> {
    inner: Arc<M>,
}

impl<M: Module + 'static> Checkpointed<M> {
    /// Wrap `m`. The inner module is moved into an `Arc` so the
    /// re-compute closure can stay `'static + Send + Sync`.
    pub fn new(m: M) -> Self {
        Checkpointed { inner: Arc::new(m) }
    }

    /// Borrow the inner module. Useful for inspecting state outside
    /// the forward path (e.g. exporting a state_dict).
    pub fn inner(&self) -> &M {
        &self.inner
    }
}

impl<M: Module + Send + Sync + 'static> Module for Checkpointed<M> {
    fn forward(&self, input: &Variable) -> Result<Variable, ModuleError> {
        let inner = self.inner.clone();
        rustorch_autograd::checkpoint(move |x: &Variable| inner.forward(x), input)
    }

    fn parameters(&self) -> Vec<Variable> {
        self.inner.parameters()
    }

    fn named_parameters(&self) -> Vec<(String, Variable)> {
        // Prefix with "ckpt." so users can tell apart the checkpoint
        // wrapper in a state_dict listing.
        self.inner
            .named_parameters()
            .into_iter()
            .map(|(n, v)| (format!("ckpt.{n}"), v))
            .collect()
    }

    // train()/eval() left as no-op — see module docs.
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Linear;
    use rustorch_autograd::backward;
    use rustorch_autograd::ops;
    use rustorch_core::tensor::tensor_impl::Tensor;

    #[test]
    fn checkpointed_linear_forward_matches_uncheckpointed() {
        let inner = Linear::new(3, 2);
        // Capture the inner params so we can build a parallel un-wrapped
        // copy with bit-identical state.
        let raw_w = inner.parameters()[0].tensor().clone();
        let raw_b = inner.parameters()[1].tensor().clone();
        let plain = Linear::new(3, 2);
        plain.parameters()[0].set_data(raw_w);
        plain.parameters()[1].set_data(raw_b);

        let ckpt = Checkpointed::new(inner);

        let x = Variable::leaf(
            Tensor::from_vec([2_usize, 3], vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
        );
        let y_ck = ckpt.forward(&x).unwrap();
        let y_plain = plain.forward(&x).unwrap();
        assert_eq!(
            y_ck.tensor().as_slice::<f32>().unwrap(),
            y_plain.tensor().as_slice::<f32>().unwrap()
        );
    }

    #[test]
    fn checkpointed_propagates_grads_to_input() {
        let net = Checkpointed::new(Linear::new(3, 2));
        let x = Variable::leaf(
            Tensor::from_vec([2_usize, 3], vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
        );
        let y = net.forward(&x).unwrap();
        let s = ops::sum(&y).unwrap();
        backward(&s, None).unwrap();
        // Backward must populate grads on the wrapped Linear's params.
        for p in net.parameters() {
            assert!(p.grad().is_some(), "param missing gradient");
        }
    }

    #[test]
    fn checkpointed_named_params_are_prefixed() {
        let net = Checkpointed::new(Linear::new(2, 2));
        for (name, _) in net.named_parameters() {
            assert!(
                name.starts_with("ckpt."),
                "expected ckpt. prefix, got {name}"
            );
        }
    }

    #[test]
    fn deep_stack_with_checkpointed_blocks_recomputes_each() {
        // Stack of 4 Checkpointed Linear blocks → backward must trigger
        // 4 re-computes (one per block).
        use crate::Sequential;
        use rustorch_autograd::{gradient_checkpointing_count, ops, Variable};

        let net = Sequential::new()
            .add(Checkpointed::new(Linear::new(8, 8)))
            .add(Checkpointed::new(Linear::new(8, 8)))
            .add(Checkpointed::new(Linear::new(8, 8)))
            .add(Checkpointed::new(Linear::new(8, 8)));

        let x = Variable::leaf(
            rustorch_core::tensor::tensor_impl::Tensor::from_vec(
                [2_usize, 8],
                (0..16).map(|i| (i as f32) * 0.1).collect(),
            )
            .unwrap(),
        );
        let before = gradient_checkpointing_count();
        let y = net.forward(&x).unwrap();
        let s = ops::sum(&y).unwrap();
        rustorch_autograd::backward(&s, None).unwrap();
        let delta = gradient_checkpointing_count() - before;
        assert!(
            delta >= 4,
            "expected ≥ 4 re-computes (one per block), got {delta}"
        );
    }
}
