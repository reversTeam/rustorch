//! `Variable` — the user-facing autograd-aware tensor wrapper.
//!
//! Composes:
//! - `data: Arc<Mutex<Tensor>>` — the underlying tensor, **shared
//!   mutable** so the optimiser and the model see the same parameter
//!   updates after a `step()`. Tensor is cheap to clone (Arc bump on
//!   storage), so `tensor()` returns a fresh clone of the current
//!   contents on every call.
//! - an optional `grad_fn`: the backward node that produced this
//!   variable (None for leaves)
//! - a shared `grad` slot (`Arc<Mutex<Option<Tensor>>>`) where
//!   `backward()` accumulates the gradient
//! - a `requires_grad` flag
//!
//! `Clone` shares the data slot — a parameter cloned into the
//! optimiser's parameter list aliases the model's copy, so an
//! optimiser-side `set_data(new_tensor)` is observed by the model on
//! the next forward pass.

use crate::node::{Edge, Node};
use rustorch_core::tensor::tensor_impl::Tensor;
use std::sync::{Arc, Mutex};

/// Autograd-aware tensor wrapper.
#[derive(Clone)]
pub struct Variable {
    /// Shared mutable tensor data. The wrapping `Arc<Mutex<...>>`
    /// supports the model+optimiser parameter-update flow.
    pub data: Arc<Mutex<Tensor>>,
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
            data: Arc::new(Mutex::new(tensor)),
            grad_fn: None,
            grad: Arc::new(Mutex::new(None)),
            requires_grad: false,
        }
    }

    /// Build a leaf variable that participates in autograd.
    pub fn leaf(tensor: Tensor) -> Self {
        Variable {
            data: Arc::new(Mutex::new(tensor)),
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

    /// Read the current tensor data. Returns a fresh `Tensor` clone
    /// (cheap — Arc bump on storage). Used by ops to get the value
    /// at forward time, even after the optimiser has mutated the
    /// shared data slot.
    #[inline]
    pub fn tensor(&self) -> Tensor {
        self.data.lock().unwrap().clone()
    }

    /// Snapshot the underlying tensor as a **host-readable** Tensor.
    ///
    /// Used by optimisers (Adam, AdamW, SGD, ...) that need to read
    /// parameter bytes via `.as_slice::<f32>()`. With P3.Z Task A
    /// strict semantics, that call returns `None` for tensors whose
    /// storage lives on the GPU — so this method auto-materialises
    /// via `tensor_to_cpu` when the wgpu feature is enabled.
    ///
    /// Transitional bridge: P3.Z Task O FusedAdamW will operate on
    /// `Storage::Wgpu` natively (no host trip), removing this
    /// download from the optimiser hot path.
    pub fn data_snapshot(&self) -> Tensor {
        let t = self.tensor();
        #[cfg(feature = "wgpu")]
        {
            if !t.storage().is_cpu() {
                if let Ok(host) = rustorch_wgpu::transfer::tensor_to_cpu(&t) {
                    // Preserve the device tag so the next op still
                    // dispatches through wgpu_backend(); the storage
                    // is now CPU but the routing decision survives.
                    return host.with_device(t.device());
                }
            }
        }
        t
    }

    /// Device the underlying tensor lives on. Used by the autograd
    /// dispatcher to pick the right backend (CPU vs Wgpu) at op time.
    /// See P3.Y plan, Phase A.
    #[inline]
    pub fn device(&self) -> rustorch_core::tensor::device::Device {
        self.data.lock().unwrap().device()
    }

    /// Replace the shared `data` with a new tensor. Used by optimisers
    /// after computing the parameter update.
    pub fn set_data(&self, new: Tensor) {
        *self.data.lock().unwrap() = new;
    }

    /// Read the accumulated gradient (returns `None` until
    /// `backward()` has run on a downstream output).
    ///
    /// **P3.Z Task A**: when the wgpu feature is on and the gradient
    /// lives on the GPU, this auto-materialises to host so optimisers
    /// (Adam / AdamW / SGD / ...) can read its bytes via
    /// `.as_slice::<f32>()`. Removed once Task O FusedAdamW lands.
    pub fn grad(&self) -> Option<Tensor> {
        let g = self.grad.lock().unwrap().clone()?;
        #[cfg(feature = "wgpu")]
        {
            if !g.storage().is_cpu() {
                if let Ok(host) = rustorch_wgpu::transfer::tensor_to_cpu(&g) {
                    return Some(host.with_device(g.device()));
                }
            }
        }
        Some(g)
    }

    /// Reset the accumulated gradient to `None`.
    pub fn zero_grad(&self) {
        *self.grad.lock().unwrap() = None;
    }

    /// Replace the accumulated gradient with `new_grad`. Used by
    /// gradient post-processors (e.g. `clip_grad_norm_` in
    /// `rustorch-optim`) that need to scale or clip the grad in
    /// place after `backward()` has run but before `optimizer.step()`.
    ///
    /// Pass `None` to reset (equivalent to [`Self::zero_grad`]).
    pub fn set_grad(&self, new_grad: Option<Tensor>) {
        *self.grad.lock().unwrap() = new_grad;
    }

    /// Return a *detached* copy — same data snapshot, no grad_fn,
    /// requires_grad false. Detach breaks the autograd graph.
    pub fn detach(&self) -> Variable {
        let snapshot = self.tensor();
        Variable {
            data: Arc::new(Mutex::new(snapshot)),
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
        let t = self.tensor();
        f.debug_struct("Variable")
            .field("shape", &t.shape())
            .field("dtype", &t.dtype())
            .field("requires_grad", &self.requires_grad)
            .field("has_grad_fn", &self.grad_fn.is_some())
            .finish()
    }
}
