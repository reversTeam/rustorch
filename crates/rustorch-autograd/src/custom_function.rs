//! User-defined autograd functions (P1.5).
//!
//! Mirrors `torch.autograd.Function`. Users implement [`CustomFunction`]
//! with separate `forward` and `backward` methods, then call
//! [`apply_custom`] to splice the new op into the autograd tape.
//!
//! Example:
//! ```ignore
//! struct Square;
//! impl CustomFunction for Square {
//!     fn forward(ctx: &mut FwdCtx, inputs: &[Tensor]) -> Vec<Tensor> {
//!         ctx.save_for_backward(inputs[0].clone());
//!         vec![cpu_backend().mul(&inputs[0], &inputs[0]).unwrap()]
//!     }
//!     fn backward(ctx: &BwdCtx, grad_outputs: &[Tensor]) -> Vec<Tensor> {
//!         let x = &ctx.saved_tensors()[0];
//!         let two = Tensor::scalar(2.0);
//!         let g = cpu_backend().mul(&grad_outputs[0], x).unwrap();
//!         let g = cpu_backend().mul(&g, &two).unwrap();
//!         vec![g]
//!     }
//! }
//!
//! let y = apply_custom::<Square>(&[x])?;
//! ```

use crate::backward::BackwardError;
use crate::node::{Edge, Node};
use crate::tape::is_grad_enabled;
use crate::variable::Variable;
use rustorch_core::tensor::tensor_impl::Tensor;
use std::sync::Arc;

/// User-defined autograd function — implement this trait to add a new
/// op without modifying the autograd internals.
pub trait CustomFunction: Send + Sync + 'static {
    /// Compute outputs from inputs. The provided context can save
    /// intermediate tensors that `backward` will need.
    fn forward(ctx: &mut FwdCtx, inputs: &[Tensor]) -> Vec<Tensor>;

    /// Given the gradients on each output (in the same order returned
    /// by `forward`), produce gradients with respect to each input
    /// (same order as the original `inputs` slice).
    ///
    /// Return a vector of length `inputs.len()`. Use a zero-tensor if
    /// an input doesn't require grad.
    fn backward(ctx: &BwdCtx, grad_outputs: &[Tensor]) -> Vec<Tensor>;
}

/// Forward context — accumulator passed to `CustomFunction::forward`.
pub struct FwdCtx {
    saved: Vec<Tensor>,
}

impl FwdCtx {
    /// Build an empty context.
    pub fn new() -> Self {
        FwdCtx { saved: Vec::new() }
    }

    /// Save a tensor for the backward pass.
    pub fn save_for_backward(&mut self, t: Tensor) {
        self.saved.push(t);
    }

    /// Move out the saved tensors (consumed when wrapping into
    /// [`BwdCtx`]).
    pub fn into_saved(self) -> Vec<Tensor> {
        self.saved
    }
}

impl Default for FwdCtx {
    fn default() -> Self {
        Self::new()
    }
}

/// Backward context — read-only handle to tensors saved during
/// the forward pass.
pub struct BwdCtx {
    saved: Vec<Tensor>,
}

impl BwdCtx {
    /// Tensors saved by `forward` via `ctx.save_for_backward`.
    pub fn saved_tensors(&self) -> &[Tensor] {
        &self.saved
    }
}

struct CustomBackward<F: CustomFunction> {
    saved: Vec<Tensor>,
    edges: Vec<Edge>,
    _phantom: std::marker::PhantomData<F>,
}

impl<F: CustomFunction> Node for CustomBackward<F> {
    fn name(&self) -> &'static str {
        "CustomBackward"
    }
    fn apply(&self, grad: &Tensor) -> Vec<Option<Tensor>> {
        let ctx = BwdCtx {
            saved: self.saved.clone(),
        };
        let grad_inputs = F::backward(&ctx, std::slice::from_ref(grad));
        grad_inputs.into_iter().map(Some).collect()
    }
    fn next_edges(&self) -> &[Edge] {
        &self.edges
    }
}

/// Run a user-defined `CustomFunction` and splice the resulting
/// backward node into the autograd tape.
///
/// v1 supports single-output functions; multi-output is pending —
/// the returned `Vec<Variable>` always has length 1 in this slice.
pub fn apply_custom<F: CustomFunction>(
    inputs: &[Variable],
) -> Result<Vec<Variable>, BackwardError> {
    let raw_inputs: Vec<Tensor> = inputs.iter().map(|v| v.tensor().clone()).collect();
    let mut ctx = FwdCtx::new();
    let outputs = F::forward(&mut ctx, &raw_inputs);
    if outputs.is_empty() {
        return Err(BackwardError::Backend {
            op: "custom",
            message: "CustomFunction::forward returned 0 outputs".to_string(),
        });
    }
    let saved = ctx.into_saved();

    let any_requires_grad = inputs.iter().any(|v| v.requires_grad);
    let edges: Vec<Edge> = inputs.iter().map(|v| v.edge()).collect();

    let mut out_vars: Vec<Variable> = outputs.into_iter().map(Variable::new).collect();
    if is_grad_enabled() && any_requires_grad {
        let node = Arc::new(CustomBackward::<F> {
            saved,
            edges,
            _phantom: std::marker::PhantomData,
        });
        for v in &mut out_vars {
            v.grad_fn = Some(node.clone());
            v.requires_grad = true;
        }
    }
    Ok(out_vars)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{backward, ops, Variable};
    use rustorch_core::tensor::tensor_impl::Tensor;
    use rustorch_cpu::cpu_backend::cpu_backend;

    /// Square: y = x*x ; dy/dx = 2x.
    struct Square;
    impl CustomFunction for Square {
        fn forward(ctx: &mut FwdCtx, inputs: &[Tensor]) -> Vec<Tensor> {
            ctx.save_for_backward(inputs[0].clone());
            vec![cpu_backend().mul(&inputs[0], &inputs[0]).unwrap()]
        }
        fn backward(ctx: &BwdCtx, grad_outputs: &[Tensor]) -> Vec<Tensor> {
            let x = &ctx.saved_tensors()[0];
            let two = Tensor::scalar(2.0_f32);
            let xg = cpu_backend().mul(&grad_outputs[0], x).unwrap();
            vec![cpu_backend().mul(&xg, &two).unwrap()]
        }
    }

    #[test]
    fn custom_square_forward_correct() {
        let x = Variable::leaf(Tensor::from_vec([3], vec![2.0_f32, 3.0, 4.0]).unwrap());
        let y = apply_custom::<Square>(std::slice::from_ref(&x))
            .unwrap()
            .pop()
            .unwrap();
        let y_t = y.tensor();
        let v = y_t.as_slice::<f32>().unwrap();
        assert_eq!(v, &[4.0_f32, 9.0, 16.0]);
    }

    #[test]
    fn custom_square_backward_gives_2x() {
        let x = Variable::leaf(Tensor::from_vec([3], vec![2.0_f32, 3.0, 4.0]).unwrap());
        let y = apply_custom::<Square>(std::slice::from_ref(&x))
            .unwrap()
            .pop()
            .unwrap();
        let s = ops::sum(&y).unwrap();
        backward(&s, None).unwrap();
        let g = x.grad().unwrap();
        let g_v = g.as_slice::<f32>().unwrap();
        assert_eq!(g_v, &[4.0_f32, 6.0, 8.0]); // 2*x
    }

    #[test]
    fn custom_no_grad_path_silent() {
        let x = Variable::new(Tensor::from_vec([3], vec![1.0_f32, 2.0, 3.0]).unwrap());
        let y = apply_custom::<Square>(&[x]).unwrap().pop().unwrap();
        // No requires_grad on inputs → no grad_fn on output.
        assert!(!y.requires_grad);
    }

    #[test]
    fn custom_chain_with_other_ops() {
        // f(x) = (x²) * 3 = 3x². df/dx = 6x.
        let x = Variable::leaf(Tensor::from_vec([2], vec![1.0_f32, 2.0]).unwrap());
        let xsq = apply_custom::<Square>(std::slice::from_ref(&x))
            .unwrap()
            .pop()
            .unwrap();
        let three = Variable::new(Tensor::scalar(3.0_f32));
        let y = ops::mul(&xsq, &three).unwrap();
        let s = ops::sum(&y).unwrap();
        backward(&s, None).unwrap();
        // Expected: 6 * [1, 2] = [6, 12]
        let g = x.grad().unwrap();
        assert_eq!(g.as_slice::<f32>().unwrap(), &[6.0_f32, 12.0]);
    }
}
