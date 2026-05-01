//! Gradient checkpointing (P1.5) — trade compute for memory.
//!
//! `checkpoint(f, x)` runs `f(x)` without storing intermediate
//! activations on the autograd tape. On the backward pass, the
//! function is re-executed with `requires_grad` re-enabled so the
//! gradient can flow through. The trade-off is roughly +30% compute
//! for ~50% peak-memory savings on transformer-style stacks.
//!
//! v1: single-input single-output, F functions are `Fn(&Variable) ->
//! Result<Variable, BackwardError>`. Multi-input / multi-output
//! checkpointing pending follow-up.

use crate::backward::BackwardError;
use crate::node::{Edge, Node};
use crate::tape::is_grad_enabled;
use crate::variable::Variable;
use rustorch_core::tensor::tensor_impl::Tensor;
use std::sync::Arc;

/// Type-erased checkpointable function.
type CheckpointFn = dyn Fn(&Variable) -> Result<Variable, BackwardError> + Send + Sync;

struct CheckpointBackward {
    f: Arc<CheckpointFn>,
    saved_input: Tensor,
    edges: [Edge; 1],
}

impl Node for CheckpointBackward {
    fn name(&self) -> &'static str {
        "CheckpointBackward"
    }
    fn apply(&self, grad: &Tensor) -> Vec<Option<Tensor>> {
        // Re-run f with grad enabled to rebuild the local tape, then
        // backward against the local output.
        let leaf = Variable::leaf(self.saved_input.clone());
        let out = (self.f)(&leaf).expect("checkpoint recompute");
        // Inject `grad` as the upstream gradient on the recomputed output.
        crate::backward::backward(&out, Some(grad.clone())).expect("checkpoint backward");
        let g = leaf.grad().expect("checkpoint leaf grad");
        vec![Some(g)]
    }
    fn next_edges(&self) -> &[Edge] {
        &self.edges
    }
}

/// Run `f(x)` with intermediate activations dropped after forward.
/// Backward re-runs `f` to produce gradients.
pub fn checkpoint<F>(f: F, x: &Variable) -> Result<Variable, BackwardError>
where
    F: Fn(&Variable) -> Result<Variable, BackwardError> + Send + Sync + 'static,
{
    // Forward without grad tracking; we'll rebuild the tape on backward.
    let out_no_grad = crate::tape::no_grad(|| f(x))?;

    let mut out_var = Variable::new(out_no_grad.tensor().clone());
    if is_grad_enabled() && x.requires_grad {
        let node = Arc::new(CheckpointBackward {
            f: Arc::new(f),
            saved_input: x.tensor().clone(),
            edges: [x.edge()],
        });
        out_var.grad_fn = Some(node);
        out_var.requires_grad = true;
    }
    Ok(out_var)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{backward, ops, Variable};
    use rustorch_core::tensor::tensor_impl::Tensor;

    #[test]
    fn checkpoint_forward_matches_direct_call() {
        let f = |x: &Variable| -> Result<Variable, BackwardError> {
            // y = x²
            ops::mul(x, x)
        };
        let x = Variable::new(Tensor::from_vec([3], vec![2.0_f32, 3.0, 4.0]).unwrap());
        let y_direct = f(&x).unwrap();
        let y_ckpt = checkpoint(f, &x).unwrap();
        assert_eq!(
            y_direct.tensor().as_slice::<f32>().unwrap(),
            y_ckpt.tensor().as_slice::<f32>().unwrap()
        );
    }

    #[test]
    fn checkpoint_backward_matches_non_checkpointed() {
        let f = |x: &Variable| -> Result<Variable, BackwardError> { ops::mul(x, x) };
        let x = Variable::leaf(Tensor::from_vec([3], vec![2.0_f32, 3.0, 4.0]).unwrap());
        let y = checkpoint(f, &x).unwrap();
        let s = ops::sum(&y).unwrap();
        backward(&s, None).unwrap();
        let g = x.grad().unwrap();
        // d/dx (x²).sum() = 2x → [4, 6, 8]
        assert_eq!(g.as_slice::<f32>().unwrap(), &[4.0_f32, 6.0, 8.0]);
    }

    #[test]
    fn checkpoint_no_grad_inputs_silent() {
        let f = |x: &Variable| -> Result<Variable, BackwardError> { ops::mul(x, x) };
        let x = Variable::new(Tensor::from_vec([3], vec![1.0_f32, 2.0, 3.0]).unwrap());
        let y = checkpoint(f, &x).unwrap();
        assert!(!y.requires_grad);
    }
}
