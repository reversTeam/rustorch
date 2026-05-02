//! Gradient checkpointing (P1.5 + P3.2) — trade compute for memory.
//!
//! `checkpoint(f, x)` runs `f(x)` without storing intermediate
//! activations on the autograd tape. On the backward pass, the
//! function is re-executed with `requires_grad` re-enabled so the
//! gradient can flow through. The trade-off is roughly +30% compute
//! for ~50% peak-memory savings on transformer-style stacks.
//!
//! Three flavours of the API:
//! - [`checkpoint`] — single input, single output (the original P1.5 form)
//! - [`checkpoint_n`] — N inputs → 1 output (P3.2)
//! - [`gradient_checkpointing_count`] — global atomic counter for tests
//!   asserting that `f` was actually re-invoked during backward.

use crate::backward::BackwardError;
use crate::node::{Edge, Node};
use crate::tape::is_grad_enabled;
use crate::variable::Variable;
use rustorch_core::tensor::tensor_impl::Tensor;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// Process-wide counter incremented each time a checkpointed
/// closure is re-invoked during a backward pass. Reset by tests
/// via [`reset_gradient_checkpointing_count`].
///
/// Useful as a cheap "did the recompute actually happen?" assertion:
/// in a non-checkpointed graph the closure runs once (forward); in a
/// checkpointed graph it runs twice (forward + backward).
pub static GC_RECOMPUTES: AtomicU64 = AtomicU64::new(0);

/// Read the current re-compute counter.
pub fn gradient_checkpointing_count() -> u64 {
    GC_RECOMPUTES.load(Ordering::Relaxed)
}

/// Reset the re-compute counter. Tests use this to scope assertions
/// to a single forward+backward window.
pub fn reset_gradient_checkpointing_count() {
    GC_RECOMPUTES.store(0, Ordering::Relaxed);
}

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
        GC_RECOMPUTES.fetch_add(1, Ordering::Relaxed);
        // Inject `grad` as the upstream gradient on the recomputed output.
        crate::backward::backward(&out, Some(grad.clone())).expect("checkpoint backward");
        let g = leaf.grad().expect("checkpoint leaf grad");
        vec![Some(g)]
    }
    fn next_edges(&self) -> &[Edge] {
        &self.edges
    }
}

// --------------------------------------------------------------------------
// checkpoint_n — multi-input, single output
// --------------------------------------------------------------------------

/// N-ary type-erased checkpointable function: takes a slice of
/// Variables and produces one Variable.
type CheckpointFnN = dyn Fn(&[Variable]) -> Result<Variable, BackwardError> + Send + Sync;

struct CheckpointNBackward {
    f: Arc<CheckpointFnN>,
    saved_inputs: Vec<Tensor>,
    edges: Vec<Edge>,
}

impl Node for CheckpointNBackward {
    fn name(&self) -> &'static str {
        "CheckpointNBackward"
    }
    fn apply(&self, grad: &Tensor) -> Vec<Option<Tensor>> {
        // Rebuild a fresh local tape: detach inputs, re-run f, backward.
        let leaves: Vec<Variable> = self
            .saved_inputs
            .iter()
            .cloned()
            .map(Variable::leaf)
            .collect();
        let out = (self.f)(&leaves).expect("checkpoint_n recompute");
        GC_RECOMPUTES.fetch_add(1, Ordering::Relaxed);
        crate::backward::backward(&out, Some(grad.clone())).expect("checkpoint_n backward");
        // Collect each leaf's grad. Inputs that did not flow into the
        // output (e.g. unused branches) get `None`.
        leaves.iter().map(|l| l.grad()).collect()
    }
    fn next_edges(&self) -> &[Edge] {
        &self.edges
    }
}

/// Multi-input variant of [`checkpoint`]. The closure receives a slice
/// of the inputs; the local tape is dropped after forward and rebuilt
/// from scratch on backward.
///
/// ```ignore
/// // Two-input MLP block: y = (x @ w + b).relu().sum()
/// let y = checkpoint_n(
///     |args| {
///         let pre = ops::matmul(&args[0], &args[1])?;
///         let act = ops::relu(&ops::add(&pre, &args[2])?)?;
///         ops::sum(&act)
///     },
///     &[x.clone(), w.clone(), b.clone()],
/// )?;
/// ```
pub fn checkpoint_n<F>(f: F, inputs: &[Variable]) -> Result<Variable, BackwardError>
where
    F: Fn(&[Variable]) -> Result<Variable, BackwardError> + Send + Sync + 'static,
{
    // Forward without grad tracking; we'll rebuild the tape on backward.
    let out_no_grad = crate::tape::no_grad(|| f(inputs))?;

    let mut out_var = Variable::new(out_no_grad.tensor().clone());
    if is_grad_enabled() && inputs.iter().any(|v| v.requires_grad) {
        let saved_inputs: Vec<Tensor> = inputs.iter().map(|v| v.tensor().clone()).collect();
        let edges: Vec<Edge> = inputs.iter().map(|v| v.edge()).collect();
        let node = Arc::new(CheckpointNBackward {
            f: Arc::new(f),
            saved_inputs,
            edges,
        });
        out_var.grad_fn = Some(node);
        out_var.requires_grad = true;
    }
    Ok(out_var)
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

    #[test]
    fn checkpoint_recompute_counter_increments_once() {
        // Single backward → at least one re-compute. We use `>=` and
        // a delta read because the global counter is shared across
        // the whole test process and `cargo test` runs tests
        // concurrently by default.
        let before = gradient_checkpointing_count();
        let f = |x: &Variable| -> Result<Variable, BackwardError> { ops::mul(x, x) };
        let x = Variable::leaf(Tensor::from_vec([2], vec![3.0_f32, 4.0]).unwrap());
        let y = checkpoint(f, &x).unwrap();
        let s = ops::sum(&y).unwrap();
        backward(&s, None).unwrap();
        let delta = gradient_checkpointing_count() - before;
        assert!(delta >= 1, "expected at least one re-compute, got {delta}");
    }

    #[test]
    fn checkpoint_n_two_inputs() {
        // y = (a + b)² on a=[1,2,3], b=[10,20,30] → [121, 484, 1089]
        // d/da = d/db = 2(a+b) → [22, 44, 66]
        let f = |args: &[Variable]| -> Result<Variable, BackwardError> {
            let s = ops::add(&args[0], &args[1])?;
            ops::mul(&s, &s)
        };
        let a = Variable::leaf(Tensor::from_vec([3], vec![1.0_f32, 2.0, 3.0]).unwrap());
        let b = Variable::leaf(Tensor::from_vec([3], vec![10.0_f32, 20.0, 30.0]).unwrap());
        let y = checkpoint_n(f, &[a.clone(), b.clone()]).unwrap();
        let s = ops::sum(&y).unwrap();
        backward(&s, None).unwrap();
        assert_eq!(
            a.grad().unwrap().as_slice::<f32>().unwrap(),
            &[22.0_f32, 44.0, 66.0]
        );
        assert_eq!(
            b.grad().unwrap().as_slice::<f32>().unwrap(),
            &[22.0_f32, 44.0, 66.0]
        );
    }

    #[test]
    fn checkpoint_n_matches_non_checkpointed_grads() {
        // Verify that the grads coming out of checkpoint_n are
        // bit-equal to those from a direct (un-checkpointed) call.
        let f = |args: &[Variable]| -> Result<Variable, BackwardError> {
            let p = ops::mul(&args[0], &args[1])?;
            ops::add(&p, &args[2])
        };
        let mk = || {
            (
                Variable::leaf(Tensor::from_vec([3], vec![1.0_f32, 2.0, 3.0]).unwrap()),
                Variable::leaf(Tensor::from_vec([3], vec![4.0_f32, 5.0, 6.0]).unwrap()),
                Variable::leaf(Tensor::from_vec([3], vec![7.0_f32, 8.0, 9.0]).unwrap()),
            )
        };

        // Non-checkpointed reference.
        let (a1, b1, c1) = mk();
        let y_ref = f(&[a1.clone(), b1.clone(), c1.clone()]).unwrap();
        backward(&ops::sum(&y_ref).unwrap(), None).unwrap();
        let g_ref_a = a1.grad().unwrap();
        let g_ref_b = b1.grad().unwrap();
        let g_ref_c = c1.grad().unwrap();

        // Checkpointed.
        let before = gradient_checkpointing_count();
        let (a2, b2, c2) = mk();
        let y_ck = checkpoint_n(f, &[a2.clone(), b2.clone(), c2.clone()]).unwrap();
        backward(&ops::sum(&y_ck).unwrap(), None).unwrap();
        let delta = gradient_checkpointing_count() - before;
        assert!(delta >= 1, "expected at least one re-compute, got {delta}");
        assert_eq!(
            a2.grad().unwrap().as_slice::<f32>(),
            g_ref_a.as_slice::<f32>()
        );
        assert_eq!(
            b2.grad().unwrap().as_slice::<f32>(),
            g_ref_b.as_slice::<f32>()
        );
        assert_eq!(
            c2.grad().unwrap().as_slice::<f32>(),
            g_ref_c.as_slice::<f32>()
        );
    }
}
