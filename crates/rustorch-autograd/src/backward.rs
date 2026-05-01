//! `backward()` — walk the autograd DAG in reverse topological order
//! and accumulate gradients into leaf variables' grad slots.
//!
//! Algorithm (per RFC-0003):
//! 1. Build a topological order of nodes reachable from the output's
//!    `grad_fn` via DFS.
//! 2. Initialise the output's gradient (default: ones-like).
//! 3. Walk nodes in reverse topo order; for each, call `apply(grad)`
//!    to compute input gradients and `+=` them into each edge's
//!    target (either an upstream node's accumulated grad, or a leaf
//!    grad slot).
//!
//! Multiple paths to the same node are handled by an in-process map
//! keyed on `Arc::as_ptr`.

use crate::node::{Edge, Node};
use crate::variable::Variable;
use rustorch_core::tensor::tensor_impl::Tensor;
use rustorch_cpu::cpu_backend::cpu_backend;
use std::collections::HashMap;
use std::sync::Arc;

/// Errors raised by [`backward`].
#[derive(Debug, thiserror::Error)]
pub enum BackwardError {
    /// `backward()` called on a tensor without a `grad_fn`.
    #[error("backward: tensor has no grad_fn (was it produced by an autograd op?)")]
    NoGradFn,
    /// A backend op inside backward returned an error.
    #[error("backward: backend error in {op}: {message}")]
    Backend {
        /// The op name.
        op: &'static str,
        /// The backend message.
        message: String,
    },
}

/// Run reverse-mode autograd starting from `output`. Accumulates
/// gradients into every leaf variable reachable through the graph.
///
/// `initial_grad`: if `None`, a ones-like gradient is used (the
/// canonical "loss with implicit dL/dL = 1" case). Otherwise the
/// caller-supplied gradient is used (e.g. for grad-of-grad).
pub fn backward(output: &Variable, initial_grad: Option<Tensor>) -> Result<(), BackwardError> {
    let root = match &output.grad_fn {
        Some(n) => n.clone(),
        None => return Err(BackwardError::NoGradFn),
    };
    // Default gradient: ones of the same shape and dtype as the output.
    let grad = match initial_grad {
        Some(g) => g,
        None => crate::ones_like(&output.tensor()),
    };

    // Build topological order via DFS (so that when we iterate in
    // reverse, every node's gradient is finalised before we apply it).
    let mut topo: Vec<Arc<dyn Node>> = Vec::new();
    let mut visited: HashMap<usize, ()> = HashMap::new();
    visit(&root, &mut topo, &mut visited);

    // Map from node ptr → accumulated incoming gradient.
    let mut grads: HashMap<usize, Tensor> = HashMap::new();
    grads.insert(Arc::as_ptr(&root) as *const () as usize, grad);

    // Process in reverse topo order.
    for node in topo.iter().rev() {
        let key = Arc::as_ptr(node) as *const () as usize;
        let upstream_grad = match grads.remove(&key) {
            Some(g) => g,
            None => continue, // no path reached here
        };
        let input_grads = node.apply(&upstream_grad);
        let edges = node.next_edges();
        if input_grads.len() != edges.len() {
            return Err(BackwardError::Backend {
                op: node.name(),
                message: format!(
                    "input_grads.len() ({}) != next_edges.len() ({})",
                    input_grads.len(),
                    edges.len()
                ),
            });
        }
        for (edge, ig) in edges.iter().zip(input_grads.into_iter()) {
            let g = match ig {
                Some(g) => g,
                None => continue,
            };
            accumulate_into_edge(edge, g, &mut grads)?;
        }
    }
    Ok(())
}

fn visit(node: &Arc<dyn Node>, topo: &mut Vec<Arc<dyn Node>>, visited: &mut HashMap<usize, ()>) {
    let key = Arc::as_ptr(node) as *const () as usize;
    if visited.contains_key(&key) {
        return;
    }
    visited.insert(key, ());
    for edge in node.next_edges() {
        if let Some(ref n) = edge.node {
            visit(n, topo, visited);
        }
    }
    topo.push(node.clone());
}

fn accumulate_into_edge(
    edge: &Edge,
    grad: Tensor,
    grads: &mut HashMap<usize, Tensor>,
) -> Result<(), BackwardError> {
    if let Some(ref upstream) = edge.node {
        let key = Arc::as_ptr(upstream) as *const () as usize;
        if let Some(prev) = grads.remove(&key) {
            let summed = cpu_backend()
                .add(&prev, &grad)
                .map_err(|e| BackwardError::Backend {
                    op: "accumulate",
                    message: e.to_string(),
                })?;
            grads.insert(key, summed);
        } else {
            grads.insert(key, grad);
        }
    } else if let Some(ref slot) = edge.grad_slot {
        let mut g = slot.lock().unwrap();
        let next = match g.take() {
            Some(prev) => cpu_backend()
                .add(&prev, &grad)
                .map_err(|e| BackwardError::Backend {
                    op: "accumulate_leaf",
                    message: e.to_string(),
                })?,
            None => grad,
        };
        *g = Some(next);
    }
    // detached edges: drop the gradient on the floor.
    Ok(())
}
