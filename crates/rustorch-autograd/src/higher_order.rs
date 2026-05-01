//! Higher-order grads (P1.5) — `grad()` API + create_graph hook.
//!
//! v1 ships:
//! - [`grad`] — compute the gradient of `outputs` w.r.t. `inputs`,
//!   matching `torch.autograd.grad()` shape-for-shape.
//! - [`backward_create_graph`] — alias for the existing backward when
//!   `create_graph=true` is requested. (Currently the inner backward
//!   ops aren't tape-tracked — true higher-order needs that hook —
//!   but the API surface lets call sites compile and run.)
//!
//! True nested-tape recording is gated behind a follow-up that
//! refactors all backward Nodes to emit autograd-aware ops on the
//! tape. The shape compatibility on `grad()` is preserved so call
//! sites don't change when that lands.

use crate::backward::{backward, BackwardError};
use crate::variable::Variable;
use rustorch_core::tensor::tensor_impl::Tensor;

/// Compute `d outputs / d inputs[i]` for each input.
///
/// `grad_outputs` lets the caller seed the upstream gradient (defaults
/// to `ones_like(outputs)`). The function runs `backward(outputs)` and
/// then collects the `.grad()` of each input.
///
/// `create_graph=true` requests that the backward graph be itself
/// recorded so that the returned gradients can be differentiated again.
/// In v1 this argument is accepted but doesn't yet build a nested tape
/// — the returned gradients are leaf tensors.
pub fn grad(
    outputs: &Variable,
    inputs: &[Variable],
    grad_outputs: Option<Tensor>,
    _create_graph: bool,
) -> Result<Vec<Tensor>, BackwardError> {
    backward(outputs, grad_outputs)?;
    let mut out = Vec::with_capacity(inputs.len());
    for inp in inputs {
        // Inputs that didn't accumulate (e.g. unused) get a zero tensor
        // matching the input shape.
        match inp.grad() {
            Some(g) => out.push(g),
            None => {
                let shape = inp.tensor().shape().to_vec();
                let n: usize = shape.iter().product();
                let zeros = Tensor::from_vec(shape, vec![0.0_f32; n]).expect("zero grad shape");
                out.push(zeros);
            },
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ops, Variable};
    use rustorch_core::tensor::tensor_impl::Tensor;

    #[test]
    fn grad_simple_x_squared() {
        let x = Variable::leaf(Tensor::from_vec([3], vec![1.0_f32, 2.0, 3.0]).unwrap());
        let y = ops::sum(&ops::mul(&x, &x).unwrap()).unwrap();
        let g = grad(&y, std::slice::from_ref(&x), None, false).unwrap();
        assert_eq!(g.len(), 1);
        // ∂(x.x.sum)/∂x = 2x
        assert_eq!(g[0].as_slice::<f32>().unwrap(), &[2.0_f32, 4.0, 6.0]);
    }

    #[test]
    fn grad_unused_input_returns_zeros() {
        let x = Variable::leaf(Tensor::from_vec([3], vec![1.0_f32, 2.0, 3.0]).unwrap());
        let unused = Variable::leaf(Tensor::from_vec([2], vec![10.0_f32, 20.0]).unwrap());
        let y = ops::sum(&ops::mul(&x, &x).unwrap()).unwrap();
        let g = grad(&y, &[x.clone(), unused.clone()], None, false).unwrap();
        assert_eq!(g.len(), 2);
        assert_eq!(g[1].as_slice::<f32>().unwrap(), &[0.0_f32, 0.0]);
    }
}
