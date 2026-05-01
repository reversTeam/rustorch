//! `Node` trait + `Edge` struct — the spine of the autograd graph.
//!
//! Per RFC-0003: every op that produces a tensor with `requires_grad`
//! creates exactly one `Node` describing how to compute partial
//! derivatives w.r.t. each of its inputs. `Edge`s wire `Node`s
//! together, forming a DAG that `backward()` walks in reverse
//! topological order.

use rustorch_core::tensor::tensor_impl::Tensor;
use std::sync::{Arc, Mutex};

/// One incoming edge to an autograd `Node`. Each edge represents a
/// path along which a gradient should flow back.
#[derive(Clone)]
pub struct Edge {
    /// The upstream node (None for a leaf — gradient is accumulated
    /// directly into [`Edge::grad_slot`]).
    pub node: Option<Arc<dyn Node>>,
    /// For leaves only — the slot in the originating [`Variable`]
    /// where the accumulated gradient lives.
    pub grad_slot: Option<Arc<Mutex<Option<Tensor>>>>,
}

impl Edge {
    /// Build an edge to a leaf `Variable`'s grad slot.
    pub fn leaf(slot: Arc<Mutex<Option<Tensor>>>) -> Self {
        Edge {
            node: None,
            grad_slot: Some(slot),
        }
    }

    /// Build an edge to an upstream node.
    pub fn from_node(node: Arc<dyn Node>) -> Self {
        Edge {
            node: Some(node),
            grad_slot: None,
        }
    }

    /// Build a "no-op" edge that swallows incoming gradient. Used
    /// when an input did not require a gradient.
    pub fn detached() -> Self {
        Edge {
            node: None,
            grad_slot: None,
        }
    }
}

/// Backward-pass node — given the gradient w.r.t. *this* node's
/// output, return the gradient w.r.t. each of its inputs.
pub trait Node: Send + Sync {
    /// A short name (`"AddBackward"`, `"MatMulBackward"`, …) used in
    /// error messages and debug printing.
    fn name(&self) -> &'static str;

    /// Compute the input gradients given the upstream gradient. The
    /// returned `Vec` must be the same length as `next_edges`. An
    /// `Option<Tensor>::None` slot means "this input did not need a
    /// gradient" (skipped in backward).
    fn apply(&self, grad: &Tensor) -> Vec<Option<Tensor>>;

    /// The edges into this node — one per input.
    fn next_edges(&self) -> &[Edge];
}
