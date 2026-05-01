//! `Variable` — the user-facing autograd-aware tensor wrapper.
//!
//! Composes:
//! - the underlying [`Tensor`] (the actual data + layout)
//! - an optional `grad_fn`: the backward node that produced this
//!   variable (None for leaves)
//! - a shared `grad` slot (Arc<Mutex<Option<Tensor>>>) where
//!   `backward()` accumulates the gradient
//! - a `requires_grad` flag

use crate::node::{Edge, Node};
use rustorch_core::tensor::tensor_impl::Tensor;
use std::sync::{Arc, Mutex};

/// Autograd-aware tensor wrapper.
#[derive(Clone)]
pub struct Variable {
    /// Underlying tensor.
    pub tensor: Tensor,
    /// Backward node — present iff this variable was produced by an
    /// op on at least one variable with `requires_grad`.
    pub grad_fn: Option<Arc<dyn Node>>,
    /// Accumulated gradient slot. Shared so multiple paths backward
    /// can `+=` into it.
    pub grad: Arc<Mutex<Option<Tensor>>>,
    /// Whether this variable participates in autograd.
    pub requires_grad: bool,
}

impl Variable {
    /// Build a *leaf* variable from an owned [`Tensor`]. `requires_grad`
    /// defaults to `false`.
    pub fn new(tensor: Tensor) -> Self {
        Variable {
            tensor,
            grad_fn: None,
            grad: Arc::new(Mutex::new(None)),
            requires_grad: false,
        }
    }

    /// Build a leaf variable that participates in autograd. Equivalent
    /// to `Variable::new(t).requires_grad(true)`.
    pub fn leaf(tensor: Tensor) -> Self {
        Variable {
            tensor,
            grad_fn: None,
            grad: Arc::new(Mutex::new(None)),
            requires_grad: true,
        }
    }

    /// Set the `requires_grad` flag (builder-style).
    #[must_use]
    pub fn requires_grad(mut self, flag: bool) -> Self {
        self.requires_grad = flag;
        self
    }

    /// Borrow the underlying tensor.
    #[inline]
    pub fn tensor(&self) -> &Tensor {
        &self.tensor
    }

    /// Read the accumulated gradient (returns `None` until
    /// `backward()` has run on a downstream output).
    pub fn grad(&self) -> Option<Tensor> {
        self.grad.lock().unwrap().clone()
    }

    /// Reset the accumulated gradient to `None`. Mirrors PyTorch's
    /// `optimizer.zero_grad(set_to_none=True)`.
    pub fn zero_grad(&self) {
        *self.grad.lock().unwrap() = None;
    }

    /// Return a *detached* copy — same data, no grad_fn, requires_grad
    /// false. The detach breaks the autograd graph at this point.
    pub fn detach(&self) -> Variable {
        Variable {
            tensor: self.tensor.clone(),
            grad_fn: None,
            grad: Arc::new(Mutex::new(None)),
            requires_grad: false,
        }
    }

    /// Crate-internal: get the [`Edge`] this variable contributes when
    /// used as input to an op. If the variable doesn't require grad,
    /// returns a detached edge.
    pub(crate) fn edge(&self) -> Edge {
        if !self.requires_grad {
            return Edge::detached();
        }
        match &self.grad_fn {
            Some(node) => Edge::from_node(node.clone()),
            None => Edge::leaf(self.grad.clone()),
        }
    }
}

impl core::fmt::Debug for Variable {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Variable")
            .field("shape", &self.tensor.shape())
            .field("dtype", &self.tensor.dtype())
            .field("requires_grad", &self.requires_grad)
            .field("has_grad_fn", &self.grad_fn.is_some())
            .finish()
    }
}
