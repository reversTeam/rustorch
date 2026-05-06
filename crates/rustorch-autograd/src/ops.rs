//! Autograd-aware ops on [`Variable`] — forward pass calls into the
//! CPU backend, then records a backward `Node` if any input requires
//! grad and grad tracking is enabled.
//!
//! Each op follows the same pattern:
//! 1. Compute the forward via [`cpu_backend`].
//! 2. If any input requires grad and `is_grad_enabled()`, build a
//!    backward `Node` that closes over whatever inputs/intermediates
//!    are needed for the chain rule, and attach it as `grad_fn` of
//!    the output `Variable`.
//! 3. Return the output `Variable`.
//!
//! v1 ships the formulas required for the MNIST critical path
//! (`add / sub / mul / neg / matmul / relu / sigmoid / tanh / sum /
//! mean / log_softmax / nll_loss / cross_entropy / mse_loss`), which
//! covers Linear+ReLU MLPs and BCE classification. Additional formulas
//! (conv, batch_norm, embedding, attention, …) plug in via the same
//! pattern.

use crate::backward::BackwardError;
use crate::dispatch::{pick_backend, require_same_device_2, require_same_device_3};
use crate::node::{Edge, Node};
use crate::tape::is_grad_enabled;
use crate::variable::Variable;
use rustorch_core::tensor::device::Device;
use rustorch_core::tensor::tensor_impl::Tensor;
use rustorch_cpu::backend::Reduction;

/// Convenience to wrap a backend error into [`BackwardError`].
fn backend_err(op: &'static str, e: rustorch_cpu::error::BackendError) -> BackwardError {
    BackwardError::Backend {
        op,
        message: e.to_string(),
    }
}

// --------------------------------------------------------------------------
// add — d/dx (x+y) = 1, d/dy (x+y) = 1
// --------------------------------------------------------------------------

struct AddBackward {
    /// Original input shapes — used to unbroadcast the upstream gradient
    /// when forward broadcast was performed (e.g. `[B, T, 1] + [B, T, D]`).
    lhs_shape: Vec<usize>,
    rhs_shape: Vec<usize>,
    device: Device,
    edges: [Edge; 2],
}

impl Node for AddBackward {
    fn name(&self) -> &'static str {
        "AddBackward"
    }
    fn apply(&self, grad: &Tensor) -> Vec<Option<Tensor>> {
        // Both inputs receive the same upstream gradient, but each must be
        // reduced back to the input's original shape if forward broadcast.
        let g_lhs = crate::broadcast::unbroadcast_to(self.device, grad, &self.lhs_shape);
        let g_rhs = crate::broadcast::unbroadcast_to(self.device, grad, &self.rhs_shape);
        vec![Some(g_lhs), Some(g_rhs)]
    }
    fn next_edges(&self) -> &[Edge] {
        &self.edges
    }
}

/// `lhs + rhs` (autograd-aware).
pub fn add(lhs: &Variable, rhs: &Variable) -> Result<Variable, BackwardError> {
    let device = require_same_device_2("add", lhs, rhs)?;
    let out = pick_backend(device)
        .add(&lhs.tensor(), &rhs.tensor())
        .map_err(|e| backend_err("add", e))?;
    let mut out = Variable::new(out);
    if is_grad_enabled() && (lhs.requires_grad || rhs.requires_grad) {
        let node = std::sync::Arc::new(AddBackward {
            lhs_shape: lhs.tensor().shape().to_vec(),
            rhs_shape: rhs.tensor().shape().to_vec(),
            device,
            edges: [lhs.edge(), rhs.edge()],
        });
        out.grad_fn = Some(node);
        out.requires_grad = true;
    }
    Ok(out)
}

// --------------------------------------------------------------------------
// sub — d/dx (x-y) = 1, d/dy (x-y) = -1
// --------------------------------------------------------------------------

struct SubBackward {
    lhs_shape: Vec<usize>,
    rhs_shape: Vec<usize>,
    device: Device,
    edges: [Edge; 2],
}

impl Node for SubBackward {
    fn name(&self) -> &'static str {
        "SubBackward"
    }
    fn apply(&self, grad: &Tensor) -> Vec<Option<Tensor>> {
        let neg_grad = pick_backend(self.device)
            .neg(grad)
            .expect("neg never fails on f32/f64");
        let g_lhs = crate::broadcast::unbroadcast_to(self.device, grad, &self.lhs_shape);
        let g_rhs = crate::broadcast::unbroadcast_to(self.device, &neg_grad, &self.rhs_shape);
        vec![Some(g_lhs), Some(g_rhs)]
    }
    fn next_edges(&self) -> &[Edge] {
        &self.edges
    }
}

/// `lhs - rhs`.
pub fn sub(lhs: &Variable, rhs: &Variable) -> Result<Variable, BackwardError> {
    let device = require_same_device_2("sub", lhs, rhs)?;
    let out = pick_backend(device)
        .sub(&lhs.tensor(), &rhs.tensor())
        .map_err(|e| backend_err("sub", e))?;
    let mut out = Variable::new(out);
    if is_grad_enabled() && (lhs.requires_grad || rhs.requires_grad) {
        let node = std::sync::Arc::new(SubBackward {
            lhs_shape: lhs.tensor().shape().to_vec(),
            rhs_shape: rhs.tensor().shape().to_vec(),
            device,
            edges: [lhs.edge(), rhs.edge()],
        });
        out.grad_fn = Some(node);
        out.requires_grad = true;
    }
    Ok(out)
}

// --------------------------------------------------------------------------
// mul — d/dx (x*y) = y, d/dy (x*y) = x
// --------------------------------------------------------------------------

struct MulBackward {
    lhs_saved: Tensor,
    rhs_saved: Tensor,
    device: Device,
    edges: [Edge; 2],
}

impl Node for MulBackward {
    fn name(&self) -> &'static str {
        "MulBackward"
    }
    fn apply(&self, grad: &Tensor) -> Vec<Option<Tensor>> {
        // d(x*y)/dx = y, d(x*y)/dy = x. The mul operates with broadcast,
        // so the raw products take the broadcast output shape — they must
        // be reduced back to each input's original shape before being
        // accumulated into the input slots.
        let g_lhs_raw = pick_backend(self.device)
            .mul(grad, &self.rhs_saved)
            .expect("mul backward");
        let g_rhs_raw = pick_backend(self.device)
            .mul(grad, &self.lhs_saved)
            .expect("mul backward");
        let g_lhs =
            crate::broadcast::unbroadcast_to(self.device, &g_lhs_raw, self.lhs_saved.shape());
        let g_rhs =
            crate::broadcast::unbroadcast_to(self.device, &g_rhs_raw, self.rhs_saved.shape());
        vec![Some(g_lhs), Some(g_rhs)]
    }
    fn next_edges(&self) -> &[Edge] {
        &self.edges
    }
}

/// `lhs * rhs`.
pub fn mul(lhs: &Variable, rhs: &Variable) -> Result<Variable, BackwardError> {
    let device = require_same_device_2("mul", lhs, rhs)?;
    let out = pick_backend(device)
        .mul(&lhs.tensor(), &rhs.tensor())
        .map_err(|e| backend_err("mul", e))?;
    let mut out = Variable::new(out);
    if is_grad_enabled() && (lhs.requires_grad || rhs.requires_grad) {
        let node = std::sync::Arc::new(MulBackward {
            lhs_saved: lhs.tensor().clone(),
            rhs_saved: rhs.tensor().clone(),
            device,
            edges: [lhs.edge(), rhs.edge()],
        });
        out.grad_fn = Some(node);
        out.requires_grad = true;
    }
    Ok(out)
}

// --------------------------------------------------------------------------
// neg — d/dx (-x) = -1
// --------------------------------------------------------------------------

struct NegBackward {
    device: Device,
    edges: [Edge; 1],
}

impl Node for NegBackward {
    fn name(&self) -> &'static str {
        "NegBackward"
    }
    fn apply(&self, grad: &Tensor) -> Vec<Option<Tensor>> {
        vec![Some(
            pick_backend(self.device).neg(grad).expect("neg backward"),
        )]
    }
    fn next_edges(&self) -> &[Edge] {
        &self.edges
    }
}

/// `-src`.
pub fn neg(src: &Variable) -> Result<Variable, BackwardError> {
    let out = pick_backend(src.device())
        .neg(&src.tensor())
        .map_err(|e| backend_err("neg", e))?;
    let mut out = Variable::new(out);
    if is_grad_enabled() && src.requires_grad {
        let node = std::sync::Arc::new(NegBackward {
            device: src.device(),
            edges: [src.edge()],
        });
        out.grad_fn = Some(node);
        out.requires_grad = true;
    }
    Ok(out)
}

// --------------------------------------------------------------------------
// matmul — d/dx (x@y) = grad @ y.T;  d/dy (x@y) = x.T @ grad
// --------------------------------------------------------------------------

struct MatMulBackward {
    lhs_saved: Tensor,
    rhs_saved: Tensor,
    device: Device,
    edges: [Edge; 2],
}

impl Node for MatMulBackward {
    fn name(&self) -> &'static str {
        "MatMulBackward"
    }
    fn apply(&self, grad: &Tensor) -> Vec<Option<Tensor>> {
        // Route the transposed matmuls through `matmul_with_transposes`.
        // Backends that override the default impl (e.g. MetalBackend) fuse
        // the transpose into the matmul kernel via `simdgroup_load(...,
        // transpose=true)`, skipping the explicit transpose dispatch +
        // the intermediate `[N,K]` / `[K,M]` buffer per backward pass.
        // CPU + WGPU keep the default `transpose + matmul` semantics.
        // - dX = grad @ W^T  → matmul_with_transposes(grad, W, false, true)
        // - dW = X^T @ grad  → matmul_with_transposes(X, grad, true, false)
        let g_lhs = pick_backend(self.device)
            .matmul_with_transposes(grad, &self.rhs_saved, false, true)
            .expect("matmul lhs grad (dX = G @ W^T)");
        let g_rhs = pick_backend(self.device)
            .matmul_with_transposes(&self.lhs_saved, grad, true, false)
            .expect("matmul rhs grad (dW = X^T @ G)");
        vec![Some(g_lhs), Some(g_rhs)]
    }
    fn next_edges(&self) -> &[Edge] {
        &self.edges
    }
}

/// `lhs @ rhs` (rank-2 matmul).
pub fn matmul(lhs: &Variable, rhs: &Variable) -> Result<Variable, BackwardError> {
    let device = require_same_device_2("matmul", lhs, rhs)?;
    let out = pick_backend(device)
        .matmul(&lhs.tensor(), &rhs.tensor())
        .map_err(|e| backend_err("matmul", e))?;
    let mut out = Variable::new(out);
    if is_grad_enabled() && (lhs.requires_grad || rhs.requires_grad) {
        // matmul backward needs the contiguous transposes; clone the
        // inputs to keep them alive for backward.
        let node = std::sync::Arc::new(MatMulBackward {
            lhs_saved: lhs.tensor().clone(),
            rhs_saved: rhs.tensor().clone(),
            device,
            edges: [lhs.edge(), rhs.edge()],
        });
        out.grad_fn = Some(node);
        out.requires_grad = true;
    }
    Ok(out)
}

// --------------------------------------------------------------------------
// relu — d/dx relu(x) = (x > 0) ? 1 : 0
// --------------------------------------------------------------------------

struct ReluBackward {
    saved: Tensor,
    device: Device,
    edges: [Edge; 1],
}

impl Node for ReluBackward {
    fn name(&self) -> &'static str {
        "ReluBackward"
    }
    fn apply(&self, grad: &Tensor) -> Vec<Option<Tensor>> {
        // mask = (saved > 0); g = grad * mask (with mask cast to grad dtype)
        let zero = Tensor::scalar(0.0);
        let mask = pick_backend(self.device)
            .gt(&self.saved, &zero)
            .expect("gt for relu mask");
        let mask_f = mask.to_dtype(grad.dtype());
        let g = pick_backend(self.device)
            .mul(grad, &mask_f)
            .expect("relu mask mul");
        vec![Some(g)]
    }
    fn next_edges(&self) -> &[Edge] {
        &self.edges
    }
}

/// `relu(src)`.
pub fn relu(src: &Variable) -> Result<Variable, BackwardError> {
    let out = pick_backend(src.device())
        .relu(&src.tensor())
        .map_err(|e| backend_err("relu", e))?;
    let mut out = Variable::new(out);
    if is_grad_enabled() && src.requires_grad {
        let node = std::sync::Arc::new(ReluBackward {
            saved: src.tensor().clone(),
            device: src.device(),
            edges: [src.edge()],
        });
        out.grad_fn = Some(node);
        out.requires_grad = true;
    }
    Ok(out)
}

// --------------------------------------------------------------------------
// sigmoid — d/dx sigmoid(x) = sigmoid(x) * (1 - sigmoid(x))
// --------------------------------------------------------------------------

struct SigmoidBackward {
    saved_out: Tensor,
    device: Device,
    edges: [Edge; 1],
}

impl Node for SigmoidBackward {
    fn name(&self) -> &'static str {
        "SigmoidBackward"
    }
    fn apply(&self, grad: &Tensor) -> Vec<Option<Tensor>> {
        // g = grad * out * (1 - out)
        let one = Tensor::scalar(1.0);
        let one_minus = pick_backend(self.device)
            .sub(&one, &self.saved_out)
            .expect("1-out");
        let s_times_one_minus = pick_backend(self.device)
            .mul(&self.saved_out, &one_minus)
            .expect("s*(1-s)");
        let g = pick_backend(self.device)
            .mul(grad, &s_times_one_minus)
            .expect("sigmoid backward mul");
        vec![Some(g)]
    }
    fn next_edges(&self) -> &[Edge] {
        &self.edges
    }
}

/// `sigmoid(src)`.
pub fn sigmoid(src: &Variable) -> Result<Variable, BackwardError> {
    let out = pick_backend(src.device())
        .sigmoid(&src.tensor())
        .map_err(|e| backend_err("sigmoid", e))?;
    let mut out_var = Variable::new(out.clone());
    if is_grad_enabled() && src.requires_grad {
        let node = std::sync::Arc::new(SigmoidBackward {
            saved_out: out,
            device: src.device(),
            edges: [src.edge()],
        });
        out_var.grad_fn = Some(node);
        out_var.requires_grad = true;
    }
    Ok(out_var)
}

// --------------------------------------------------------------------------
// tanh — d/dx tanh(x) = 1 - tanh(x)^2
// --------------------------------------------------------------------------

struct TanhBackward {
    saved_out: Tensor,
    device: Device,
    edges: [Edge; 1],
}

impl Node for TanhBackward {
    fn name(&self) -> &'static str {
        "TanhBackward"
    }
    fn apply(&self, grad: &Tensor) -> Vec<Option<Tensor>> {
        let one = Tensor::scalar(1.0);
        let sq = pick_backend(self.device)
            .mul(&self.saved_out, &self.saved_out)
            .expect("tanh^2");
        let one_minus_sq = pick_backend(self.device).sub(&one, &sq).expect("1-tanh^2");
        let g = pick_backend(self.device)
            .mul(grad, &one_minus_sq)
            .expect("tanh backward mul");
        vec![Some(g)]
    }
    fn next_edges(&self) -> &[Edge] {
        &self.edges
    }
}

/// `tanh(src)`.
pub fn tanh(src: &Variable) -> Result<Variable, BackwardError> {
    let out = pick_backend(src.device())
        .tanh(&src.tensor())
        .map_err(|e| backend_err("tanh", e))?;
    let mut out_var = Variable::new(out.clone());
    if is_grad_enabled() && src.requires_grad {
        let node = std::sync::Arc::new(TanhBackward {
            saved_out: out,
            device: src.device(),
            edges: [src.edge()],
        });
        out_var.grad_fn = Some(node);
        out_var.requires_grad = true;
    }
    Ok(out_var)
}

// --------------------------------------------------------------------------
// sum — d/dx sum(x) = ones_like(x) * grad (scalar broadcast back)
// --------------------------------------------------------------------------

struct SumBackward {
    in_shape: Vec<usize>,
    #[allow(dead_code)] // captured for Phase C2-C9 backward dispatch refactor
    device: Device,
    edges: [Edge; 1],
}

impl Node for SumBackward {
    fn name(&self) -> &'static str {
        "SumBackward"
    }
    fn apply(&self, grad: &Tensor) -> Vec<Option<Tensor>> {
        // grad is scalar; broadcast it back to in_shape (i.e. fill).
        let n: usize = if self.in_shape.is_empty() {
            1
        } else {
            self.in_shape.iter().product()
        };
        let g_scalar = grad.as_slice::<f32>().expect("f32 grad")[0];
        let v = vec![g_scalar; n];
        let g_full = Tensor::from_vec(self.in_shape.clone(), v).expect("ones-like");
        vec![Some(g_full)]
    }
    fn next_edges(&self) -> &[Edge] {
        &self.edges
    }
}

/// Full-tensor sum.
pub fn sum(src: &Variable) -> Result<Variable, BackwardError> {
    let out = pick_backend(src.device())
        .sum(&src.tensor())
        .map_err(|e| backend_err("sum", e))?;
    let mut out_var = Variable::new(out);
    if is_grad_enabled() && src.requires_grad {
        let node = std::sync::Arc::new(SumBackward {
            in_shape: src.tensor().shape().to_vec(),
            device: src.device(),
            edges: [src.edge()],
        });
        out_var.grad_fn = Some(node);
        out_var.requires_grad = true;
    }
    Ok(out_var)
}

// --------------------------------------------------------------------------
// mean — d/dx mean(x) = ones_like(x) / numel * grad
// --------------------------------------------------------------------------

struct MeanBackward {
    in_shape: Vec<usize>,
    n: usize,
    #[allow(dead_code)] // captured for Phase C2-C9 backward dispatch refactor
    device: Device,
    edges: [Edge; 1],
}

impl Node for MeanBackward {
    fn name(&self) -> &'static str {
        "MeanBackward"
    }
    fn apply(&self, grad: &Tensor) -> Vec<Option<Tensor>> {
        let g_scalar = grad.as_slice::<f32>().expect("f32 grad")[0] / (self.n as f32);
        let v = vec![g_scalar; self.n];
        let g_full = Tensor::from_vec(self.in_shape.clone(), v).expect("mean grad");
        vec![Some(g_full)]
    }
    fn next_edges(&self) -> &[Edge] {
        &self.edges
    }
}

/// Full-tensor mean.
pub fn mean(src: &Variable) -> Result<Variable, BackwardError> {
    let out = pick_backend(src.device())
        .mean(&src.tensor())
        .map_err(|e| backend_err("mean", e))?;
    let mut out_var = Variable::new(out);
    if is_grad_enabled() && src.requires_grad {
        let node = std::sync::Arc::new(MeanBackward {
            in_shape: src.tensor().shape().to_vec(),
            n: src.tensor().numel(),
            device: src.device(),
            edges: [src.edge()],
        });
        out_var.grad_fn = Some(node);
        out_var.requires_grad = true;
    }
    Ok(out_var)
}

// --------------------------------------------------------------------------
// cross_entropy — fused log_softmax + nll
// d/dx cross_entropy(x, t) = (softmax(x) - one_hot(t)) / N (when reduction=mean)
// --------------------------------------------------------------------------

struct CrossEntropyBackward {
    softmax_saved: Tensor,
    target: Tensor,
    n_samples: usize,
    n_classes: usize,
    reduction: Reduction,
    #[allow(dead_code)] // captured for Phase C2-C9 backward dispatch refactor
    device: Device,
    edges: [Edge; 1],
}

impl Node for CrossEntropyBackward {
    fn name(&self) -> &'static str {
        "CrossEntropyBackward"
    }
    fn apply(&self, grad: &Tensor) -> Vec<Option<Tensor>> {
        // softmax - one_hot(target) gives the per-element gradient
        // of cross_entropy w.r.t. the input logits (when reduction=mean
        // we additionally divide by N).
        let mut grad_input: Vec<f32> = self.softmax_saved.as_slice::<f32>().expect("f32").to_vec();
        let target = self.target.as_slice::<i64>().expect("i64");
        for (i, &t) in target.iter().enumerate() {
            let idx = i * self.n_classes + (t as usize);
            grad_input[idx] -= 1.0;
        }
        // Scale by upstream grad (a scalar for reduce-mean/sum).
        let scale = match self.reduction {
            Reduction::Mean => grad.as_slice::<f32>().expect("f32")[0] / (self.n_samples as f32),
            Reduction::Sum => grad.as_slice::<f32>().expect("f32")[0],
            // For Reduction::None we'd have a per-sample grad; not
            // supported in v1 fused path.
            Reduction::None => grad.as_slice::<f32>().expect("f32")[0],
        };
        for v in grad_input.iter_mut() {
            *v *= scale;
        }
        let g = Tensor::from_vec([self.n_samples, self.n_classes], grad_input)
            .expect("cross_entropy backward shape");
        vec![Some(g)]
    }
    fn next_edges(&self) -> &[Edge] {
        &self.edges
    }
}

/// Cross-entropy with class targets (autograd-aware).
pub fn cross_entropy(
    input: &Variable,
    target: &Variable,
    reduction: Reduction,
) -> Result<Variable, BackwardError> {
    // Forward: cross_entropy via the backend (= log_softmax + nll).
    let device = require_same_device_2("cross_entropy", input, target)?;
    let loss = pick_backend(device)
        .cross_entropy(&input.tensor(), &target.tensor(), reduction)
        .map_err(|e| backend_err("cross_entropy", e))?;
    let mut out_var = Variable::new(loss);
    if is_grad_enabled() && input.requires_grad {
        // For backward we save the softmax (not log_softmax) so that
        // grad = (softmax - one_hot(target)) / N.
        let softmax = pick_backend(device)
            .softmax(&input.tensor(), 1)
            .map_err(|e| backend_err("cross_entropy:softmax", e))?;
        let n_samples = input.tensor().shape()[0];
        let n_classes = input.tensor().shape()[1];
        let node = std::sync::Arc::new(CrossEntropyBackward {
            softmax_saved: softmax,
            target: target.tensor().clone(),
            n_samples,
            n_classes,
            reduction,
            device,
            edges: [input.edge()],
        });
        out_var.grad_fn = Some(node);
        out_var.requires_grad = true;
    }
    Ok(out_var)
}

// --------------------------------------------------------------------------
// add_bias — y = x + bias (with bias broadcast across batch).
// d/dx = grad ; d/dbias = sum(grad, dim=0).
// --------------------------------------------------------------------------

struct AddBiasBackward {
    /// Original bias shape (e.g. [out_features]) — used to sum the
    /// upstream grad back to that shape.
    bias_shape: Vec<usize>,
    device: Device,
    edges: [Edge; 2],
}

impl Node for AddBiasBackward {
    fn name(&self) -> &'static str {
        "AddBiasBackward"
    }
    fn apply(&self, grad: &Tensor) -> Vec<Option<Tensor>> {
        // d/dx: grad passes through unchanged.
        // d/dbias: sum over the batch axis (axis 0) to collapse [B, …]
        // back to bias_shape.
        let g_bias = pick_backend(self.device)
            .sum_dim(grad, &[0], false)
            .expect("sum_dim across batch for bias grad");
        // sum_dim of [B, out] over dim 0 → [out] — exactly bias_shape.
        let _ = &self.bias_shape;
        vec![Some(grad.clone()), Some(g_bias)]
    }
    fn next_edges(&self) -> &[Edge] {
        &self.edges
    }
}

/// `x + bias` where `bias.shape == x.shape[1..]` (broadcast across batch).
pub fn add_bias(x: &Variable, bias: &Variable) -> Result<Variable, BackwardError> {
    let device = require_same_device_2("add_bias", x, bias)?;
    let x_t = x.tensor();
    let bias_t = bias.tensor();
    if x_t.ndim() != 2 || bias_t.ndim() != 1 {
        return Err(BackwardError::Backend {
            op: "add_bias",
            message: format!(
                "expected x: rank-2, bias: rank-1; got {:?} and {:?}",
                x_t.shape(),
                bias_t.shape()
            ),
        });
    }
    let batch = x_t.shape()[0];
    let n_out = x_t.shape()[1];
    if bias_t.shape() != [n_out] {
        return Err(BackwardError::Backend {
            op: "add_bias",
            message: format!(
                "bias shape {:?} does not match x.shape[1] = {}",
                bias_t.shape(),
                n_out
            ),
        });
    }
    // Forward: dispatch to the backend's native add_bias which knows
    // how to broadcast a [N] bias across [B, N] without leaving the
    // device. P3.Z Task A: the previous CPU-side `as_slice + manual
    // tile + backend.add` round-trip path is gone — that would have
    // panicked the moment `bias_t` lived on Storage::Wgpu.
    let _ = (batch, n_out);
    let out = pick_backend(device)
        .add_bias(&x_t, &bias_t)
        .map_err(|e| backend_err("add_bias", e))?;
    let mut out_var = Variable::new(out);
    if is_grad_enabled() && (x.requires_grad || bias.requires_grad) {
        let node = std::sync::Arc::new(AddBiasBackward {
            bias_shape: bias_t.shape().to_vec(),
            device,
            edges: [x.edge(), bias.edge()],
        });
        out_var.grad_fn = Some(node);
        out_var.requires_grad = true;
    }
    Ok(out_var)
}

// --------------------------------------------------------------------------
// linear — y = x @ w + bias.  Forward dispatches to the backend's
// `matmul_with_bias` (Metal fuses into a single kernel; CPU/WGPU
// compose). Backward is the chain rule of matmul + add_bias:
//   dx    = grad @ w^T
//   dw    = x^T @ grad
//   dbias = sum(grad, dim=0)
// --------------------------------------------------------------------------

struct LinearBackward {
    x_saved: Tensor,
    w_saved: Tensor,
    bias_shape: Vec<usize>,
    device: Device,
    edges: [Edge; 3],
}

impl Node for LinearBackward {
    fn name(&self) -> &'static str {
        "LinearBackward"
    }
    fn apply(&self, grad: &Tensor) -> Vec<Option<Tensor>> {
        let backend = pick_backend(self.device);
        // dx = grad @ W^T  (transpose-aware matmul on Metal)
        let dx = backend
            .matmul_with_transposes(grad, &self.w_saved, false, true)
            .expect("linear bw: dx = grad @ W^T");
        // dW = X^T @ grad
        let dw = backend
            .matmul_with_transposes(&self.x_saved, grad, true, false)
            .expect("linear bw: dW = X^T @ grad");
        // dbias = sum(grad, dim=0) — collapse the batch axis.
        let dbias = backend
            .sum_dim(grad, &[0], false)
            .expect("linear bw: dbias = sum(grad, dim=0)");
        let _ = &self.bias_shape; // shape implicit in sum_dim output
        vec![Some(dx), Some(dw), Some(dbias)]
    }
    fn next_edges(&self) -> &[Edge] {
        &self.edges
    }
}

/// Fused linear layer forward: `y = x @ w + bias`.
///
/// Single autograd op equivalent to `add_bias(matmul(x, w), bias)`,
/// but the forward goes through the backend's `matmul_with_bias` which
/// on Metal3-capable devices fuses into a single dispatch (no
/// intermediate `[B, N]` write-back). Backward decomposes into the
/// usual matmul + add_bias chain rule, with the matmul side using
/// `matmul_with_transposes` to skip explicit transpose dispatches.
pub fn linear(x: &Variable, w: &Variable, bias: &Variable) -> Result<Variable, BackwardError> {
    let device = require_same_device_3("linear", x, w, bias)?;
    let x_t = x.tensor();
    let w_t = w.tensor();
    let bias_t = bias.tensor();
    if x_t.ndim() != 2 || w_t.ndim() != 2 || bias_t.ndim() != 1 {
        return Err(BackwardError::Backend {
            op: "linear",
            message: format!(
                "expected x: rank-2, w: rank-2, bias: rank-1; got {:?}, {:?}, {:?}",
                x_t.shape(),
                w_t.shape(),
                bias_t.shape()
            ),
        });
    }
    let out = pick_backend(device)
        .matmul_with_bias(&x_t, &w_t, &bias_t)
        .map_err(|e| backend_err("linear", e))?;
    let mut out_var = Variable::new(out);
    if is_grad_enabled() && (x.requires_grad || w.requires_grad || bias.requires_grad) {
        let node = std::sync::Arc::new(LinearBackward {
            x_saved: x_t.clone(),
            w_saved: w_t.clone(),
            bias_shape: bias_t.shape().to_vec(),
            device,
            edges: [x.edge(), w.edge(), bias.edge()],
        });
        out_var.grad_fn = Some(node);
        out_var.requires_grad = true;
    }
    Ok(out_var)
}

// --------------------------------------------------------------------------
// mse_loss — d/dx mse(x, y) = 2 * (x - y) / n  (reduction=mean)
// --------------------------------------------------------------------------

struct MseBackward {
    diff: Tensor, // x - y
    n: usize,
    reduction: Reduction,
    device: Device,
    edges: [Edge; 2],
    /// Skip the `neg` dispatch when the target tensor doesn't require
    /// gradients (the common training case — labels are constants).
    /// Saves one GPU dispatch per loss invocation.
    input_requires_grad: bool,
    target_requires_grad: bool,
}

impl Node for MseBackward {
    fn name(&self) -> &'static str {
        "MseBackward"
    }
    fn apply(&self, grad: &Tensor) -> Vec<Option<Tensor>> {
        let scale = match self.reduction {
            Reduction::Mean => grad.as_slice::<f32>().expect("f32")[0] * 2.0 / (self.n as f32),
            Reduction::Sum => grad.as_slice::<f32>().expect("f32")[0] * 2.0,
            Reduction::None => 2.0,
        };
        let scale_t = Tensor::scalar(scale);
        // g_x = diff * scale ; g_y = -g_x. Fold the sign into the scalar
        // when only one side is needed so we avoid a redundant `mul +
        // neg` chain. When both sides are needed, the standard `mul +
        // neg` path stays.
        match (self.input_requires_grad, self.target_requires_grad) {
            (true, true) => {
                let g_x = pick_backend(self.device)
                    .mul(&self.diff, &scale_t)
                    .expect("mse grad mul");
                let g_y = pick_backend(self.device).neg(&g_x).expect("mse grad neg");
                vec![Some(g_x), Some(g_y)]
            },
            (true, false) => {
                // Common training case: target is a constant (labels).
                // Skip the `neg` dispatch entirely.
                let g_x = pick_backend(self.device)
                    .mul(&self.diff, &scale_t)
                    .expect("mse grad mul");
                vec![Some(g_x), None]
            },
            (false, true) => {
                // Symmetric: only target needs a grad. Fold the negation
                // into the scalar — single mul dispatch, no neg.
                let neg_scale = Tensor::scalar(-scale);
                let g_y = pick_backend(self.device)
                    .mul(&self.diff, &neg_scale)
                    .expect("mse grad mul (-)");
                vec![None, Some(g_y)]
            },
            (false, false) => vec![None, None],
        }
    }
    fn next_edges(&self) -> &[Edge] {
        &self.edges
    }
}

/// Mean Squared Error loss (autograd-aware).
pub fn mse_loss(
    input: &Variable,
    target: &Variable,
    reduction: Reduction,
) -> Result<Variable, BackwardError> {
    let device = require_same_device_2("mse_loss", input, target)?;
    let loss = pick_backend(device)
        .mse_loss(&input.tensor(), &target.tensor(), reduction)
        .map_err(|e| backend_err("mse_loss", e))?;
    let mut out_var = Variable::new(loss);
    if is_grad_enabled() && (input.requires_grad || target.requires_grad) {
        let diff = pick_backend(device)
            .sub(&input.tensor(), &target.tensor())
            .map_err(|e| backend_err("mse_loss:sub", e))?;
        let node = std::sync::Arc::new(MseBackward {
            diff,
            n: input.tensor().numel(),
            reduction,
            device,
            edges: [input.edge(), target.edge()],
            input_requires_grad: input.requires_grad,
            target_requires_grad: target.requires_grad,
        });
        out_var.grad_fn = Some(node);
        out_var.requires_grad = true;
    }
    Ok(out_var)
}

// --------------------------------------------------------------------------
// silu — y = x * sigmoid(x); dy/dx = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
// --------------------------------------------------------------------------

struct SiluBackward {
    saved_input: Tensor,
    device: Device,
    edges: [Edge; 1],
}

impl Node for SiluBackward {
    fn name(&self) -> &'static str {
        "SiluBackward"
    }
    fn apply(&self, grad: &Tensor) -> Vec<Option<Tensor>> {
        // s = sigmoid(x); local = s * (1 + x * (1 - s))
        let s = pick_backend(self.device)
            .sigmoid(&self.saved_input)
            .expect("silu bw: sigmoid");
        let one = Tensor::scalar(1.0);
        let one_minus_s = pick_backend(self.device)
            .sub(&one, &s)
            .expect("silu bw: 1-s");
        let x_one_minus_s = pick_backend(self.device)
            .mul(&self.saved_input, &one_minus_s)
            .expect("silu bw: x*(1-s)");
        let inner = pick_backend(self.device)
            .add(&one, &x_one_minus_s)
            .expect("silu bw: 1 + x*(1-s)");
        let local = pick_backend(self.device)
            .mul(&s, &inner)
            .expect("silu bw: local");
        let g = pick_backend(self.device)
            .mul(grad, &local)
            .expect("silu bw: grad*local");
        vec![Some(g)]
    }
    fn next_edges(&self) -> &[Edge] {
        &self.edges
    }
}

/// `silu(src)` = `src * sigmoid(src)` (a.k.a. swish, autograd-aware).
pub fn silu(src: &Variable) -> Result<Variable, BackwardError> {
    let out = pick_backend(src.device())
        .silu(&src.tensor())
        .map_err(|e| backend_err("silu", e))?;
    let mut out_var = Variable::new(out);
    if is_grad_enabled() && src.requires_grad {
        let node = std::sync::Arc::new(SiluBackward {
            saved_input: src.tensor().clone(),
            device: src.device(),
            edges: [src.edge()],
        });
        out_var.grad_fn = Some(node);
        out_var.requires_grad = true;
    }
    Ok(out_var)
}

// --------------------------------------------------------------------------
// leaky_relu(x, slope) — dy/dx = 1 if x>0 else slope
// --------------------------------------------------------------------------

struct LeakyReluBackward {
    saved_input: Tensor,
    slope: f64,
    #[allow(dead_code)] // captured for Phase C2-C9 backward dispatch refactor
    device: Device,
    edges: [Edge; 1],
}

impl Node for LeakyReluBackward {
    fn name(&self) -> &'static str {
        "LeakyReluBackward"
    }
    fn apply(&self, grad: &Tensor) -> Vec<Option<Tensor>> {
        // mask: 1 if x>0 else slope, computed via slope + (1 - slope) * (x > 0)
        // We materialise it as f32 elementwise on the saved input.
        let x = self.saved_input.as_slice::<f32>().expect("f32 input");
        let g = grad.as_slice::<f32>().expect("f32 grad");
        let s = self.slope as f32;
        let mut out = Vec::with_capacity(x.len());
        for i in 0..x.len() {
            let m = if x[i] > 0.0 { 1.0_f32 } else { s };
            out.push(g[i] * m);
        }
        let g_t =
            Tensor::from_vec(self.saved_input.shape().to_vec(), out).expect("leaky_relu bw shape");
        vec![Some(g_t)]
    }
    fn next_edges(&self) -> &[Edge] {
        &self.edges
    }
}

/// `leaky_relu(src, slope)` (autograd-aware).
pub fn leaky_relu(src: &Variable, slope: f64) -> Result<Variable, BackwardError> {
    let out = pick_backend(src.device())
        .leaky_relu(&src.tensor(), slope)
        .map_err(|e| backend_err("leaky_relu", e))?;
    let mut out_var = Variable::new(out);
    if is_grad_enabled() && src.requires_grad {
        let node = std::sync::Arc::new(LeakyReluBackward {
            saved_input: src.tensor().clone(),
            slope,
            device: src.device(),
            edges: [src.edge()],
        });
        out_var.grad_fn = Some(node);
        out_var.requires_grad = true;
    }
    Ok(out_var)
}

// --------------------------------------------------------------------------
// softmax(x, dim) — y = softmax(x, dim); grad_j = y_j * (g_j - sum_i(g_i * y_i))
// --------------------------------------------------------------------------

struct SoftmaxBackward {
    saved_out: Tensor,
    dim: usize,
    device: Device,
    edges: [Edge; 1],
}

impl Node for SoftmaxBackward {
    fn name(&self) -> &'static str {
        "SoftmaxBackward"
    }
    fn apply(&self, grad: &Tensor) -> Vec<Option<Tensor>> {
        // dot = sum_along_dim(grad * y, dim, keepdim=true)
        let gy = pick_backend(self.device)
            .mul(grad, &self.saved_out)
            .expect("softmax bw: g*y");
        let dot = pick_backend(self.device)
            .sum_dim(&gy, &[self.dim], true)
            .expect("softmax bw: sum_dim");
        // diff = grad - dot (broadcast)
        let diff = pick_backend(self.device)
            .sub(grad, &dot)
            .expect("softmax bw: g - dot");
        let g = pick_backend(self.device)
            .mul(&self.saved_out, &diff)
            .expect("softmax bw: y*diff");
        vec![Some(g)]
    }
    fn next_edges(&self) -> &[Edge] {
        &self.edges
    }
}

/// `softmax(src, dim)` (numerically stable, autograd-aware).
pub fn softmax(src: &Variable, dim: usize) -> Result<Variable, BackwardError> {
    let out = pick_backend(src.device())
        .softmax(&src.tensor(), dim)
        .map_err(|e| backend_err("softmax", e))?;
    let mut out_var = Variable::new(out.clone());
    if is_grad_enabled() && src.requires_grad {
        let node = std::sync::Arc::new(SoftmaxBackward {
            saved_out: out,
            dim,
            device: src.device(),
            edges: [src.edge()],
        });
        out_var.grad_fn = Some(node);
        out_var.requires_grad = true;
    }
    Ok(out_var)
}

// --------------------------------------------------------------------------
// log_softmax(x, dim) — y = log_softmax(x, dim); grad_j = g_j - softmax_j * sum_i(g_i)
// --------------------------------------------------------------------------

struct LogSoftmaxBackward {
    saved_out: Tensor, // log_softmax output
    dim: usize,
    device: Device,
    edges: [Edge; 1],
}

impl Node for LogSoftmaxBackward {
    fn name(&self) -> &'static str {
        "LogSoftmaxBackward"
    }
    fn apply(&self, grad: &Tensor) -> Vec<Option<Tensor>> {
        // softmax = exp(log_softmax); grad_x = grad - softmax * sum_along(grad, dim, keepdim)
        let s = pick_backend(self.device)
            .exp(&self.saved_out)
            .expect("log_softmax bw: exp");
        let sum_g = pick_backend(self.device)
            .sum_dim(grad, &[self.dim], true)
            .expect("log_softmax bw: sum_dim");
        let s_sum = pick_backend(self.device)
            .mul(&s, &sum_g)
            .expect("log_softmax bw: s*sum_g");
        let g = pick_backend(self.device)
            .sub(grad, &s_sum)
            .expect("log_softmax bw: grad - s*sum_g");
        vec![Some(g)]
    }
    fn next_edges(&self) -> &[Edge] {
        &self.edges
    }
}

/// `log_softmax(src, dim)` (autograd-aware).
pub fn log_softmax(src: &Variable, dim: usize) -> Result<Variable, BackwardError> {
    let out = pick_backend(src.device())
        .log_softmax(&src.tensor(), dim)
        .map_err(|e| backend_err("log_softmax", e))?;
    let mut out_var = Variable::new(out.clone());
    if is_grad_enabled() && src.requires_grad {
        let node = std::sync::Arc::new(LogSoftmaxBackward {
            saved_out: out,
            dim,
            device: src.device(),
            edges: [src.edge()],
        });
        out_var.grad_fn = Some(node);
        out_var.requires_grad = true;
    }
    Ok(out_var)
}

// --------------------------------------------------------------------------
// div — d/dx (x/y) = 1/y;  d/dy (x/y) = -x/y²
// --------------------------------------------------------------------------

struct DivBackward {
    lhs_saved: Tensor,
    rhs_saved: Tensor,
    device: Device,
    edges: [Edge; 2],
}

impl Node for DivBackward {
    fn name(&self) -> &'static str {
        "DivBackward"
    }
    fn apply(&self, grad: &Tensor) -> Vec<Option<Tensor>> {
        // d(x/y)/dx = 1/y ; d(x/y)/dy = -x/y²
        // Both raw products take the broadcast output shape — reduce back
        // to each input's original shape before accumulating.
        let g_lhs_raw = pick_backend(self.device)
            .div(grad, &self.rhs_saved)
            .expect("div bw: g/rhs");
        let rhs_sq = pick_backend(self.device)
            .mul(&self.rhs_saved, &self.rhs_saved)
            .expect("div bw: rhs²");
        let lhs_grad = pick_backend(self.device)
            .mul(&self.lhs_saved, grad)
            .expect("div bw: x*g");
        let div_term = pick_backend(self.device)
            .div(&lhs_grad, &rhs_sq)
            .expect("div bw: x*g/rhs²");
        let g_rhs_raw = pick_backend(self.device)
            .neg(&div_term)
            .expect("div bw: neg");
        let g_lhs =
            crate::broadcast::unbroadcast_to(self.device, &g_lhs_raw, self.lhs_saved.shape());
        let g_rhs =
            crate::broadcast::unbroadcast_to(self.device, &g_rhs_raw, self.rhs_saved.shape());
        vec![Some(g_lhs), Some(g_rhs)]
    }
    fn next_edges(&self) -> &[Edge] {
        &self.edges
    }
}

/// `lhs / rhs` (autograd-aware).
pub fn div(lhs: &Variable, rhs: &Variable) -> Result<Variable, BackwardError> {
    let device = require_same_device_2("div", lhs, rhs)?;
    let out = pick_backend(device)
        .div(&lhs.tensor(), &rhs.tensor())
        .map_err(|e| backend_err("div", e))?;
    let mut out_var = Variable::new(out);
    if is_grad_enabled() && (lhs.requires_grad || rhs.requires_grad) {
        let node = std::sync::Arc::new(DivBackward {
            lhs_saved: lhs.tensor().clone(),
            rhs_saved: rhs.tensor().clone(),
            device,
            edges: [lhs.edge(), rhs.edge()],
        });
        out_var.grad_fn = Some(node);
        out_var.requires_grad = true;
    }
    Ok(out_var)
}

// --------------------------------------------------------------------------
// exp — d/dx exp(x) = exp(x)  (use saved output)
// --------------------------------------------------------------------------

struct ExpBackward {
    saved_out: Tensor,
    device: Device,
    edges: [Edge; 1],
}

impl Node for ExpBackward {
    fn name(&self) -> &'static str {
        "ExpBackward"
    }
    fn apply(&self, grad: &Tensor) -> Vec<Option<Tensor>> {
        let g = pick_backend(self.device)
            .mul(grad, &self.saved_out)
            .expect("exp bw: grad*out");
        vec![Some(g)]
    }
    fn next_edges(&self) -> &[Edge] {
        &self.edges
    }
}

/// `exp(src)` (autograd-aware).
pub fn exp(src: &Variable) -> Result<Variable, BackwardError> {
    let out = pick_backend(src.device())
        .exp(&src.tensor())
        .map_err(|e| backend_err("exp", e))?;
    let mut out_var = Variable::new(out.clone());
    if is_grad_enabled() && src.requires_grad {
        let node = std::sync::Arc::new(ExpBackward {
            saved_out: out,
            device: src.device(),
            edges: [src.edge()],
        });
        out_var.grad_fn = Some(node);
        out_var.requires_grad = true;
    }
    Ok(out_var)
}

// --------------------------------------------------------------------------
// log — d/dx log(x) = 1/x
// --------------------------------------------------------------------------

struct LogBackward {
    saved_input: Tensor,
    device: Device,
    edges: [Edge; 1],
}

impl Node for LogBackward {
    fn name(&self) -> &'static str {
        "LogBackward"
    }
    fn apply(&self, grad: &Tensor) -> Vec<Option<Tensor>> {
        let g = pick_backend(self.device)
            .div(grad, &self.saved_input)
            .expect("log bw: grad/x");
        vec![Some(g)]
    }
    fn next_edges(&self) -> &[Edge] {
        &self.edges
    }
}

/// `log(src)` (natural log, autograd-aware).
pub fn log(src: &Variable) -> Result<Variable, BackwardError> {
    let out = pick_backend(src.device())
        .log(&src.tensor())
        .map_err(|e| backend_err("log", e))?;
    let mut out_var = Variable::new(out);
    if is_grad_enabled() && src.requires_grad {
        let node = std::sync::Arc::new(LogBackward {
            saved_input: src.tensor().clone(),
            device: src.device(),
            edges: [src.edge()],
        });
        out_var.grad_fn = Some(node);
        out_var.requires_grad = true;
    }
    Ok(out_var)
}

// --------------------------------------------------------------------------
// sqrt — d/dx sqrt(x) = 1 / (2 * sqrt(x)) = 0.5 / out
// --------------------------------------------------------------------------

struct SqrtBackward {
    saved_out: Tensor,
    device: Device,
    edges: [Edge; 1],
}

impl Node for SqrtBackward {
    fn name(&self) -> &'static str {
        "SqrtBackward"
    }
    fn apply(&self, grad: &Tensor) -> Vec<Option<Tensor>> {
        // grad / (2 * out)
        let two_out = {
            let two = Tensor::scalar(2.0);
            pick_backend(self.device)
                .mul(&two, &self.saved_out)
                .expect("sqrt bw: 2*out")
        };
        let g = pick_backend(self.device)
            .div(grad, &two_out)
            .expect("sqrt bw: grad/(2*out)");
        vec![Some(g)]
    }
    fn next_edges(&self) -> &[Edge] {
        &self.edges
    }
}

/// `sqrt(src)` (autograd-aware).
pub fn sqrt(src: &Variable) -> Result<Variable, BackwardError> {
    let out = pick_backend(src.device())
        .sqrt(&src.tensor())
        .map_err(|e| backend_err("sqrt", e))?;
    let mut out_var = Variable::new(out.clone());
    if is_grad_enabled() && src.requires_grad {
        let node = std::sync::Arc::new(SqrtBackward {
            saved_out: out,
            device: src.device(),
            edges: [src.edge()],
        });
        out_var.grad_fn = Some(node);
        out_var.requires_grad = true;
    }
    Ok(out_var)
}

// --------------------------------------------------------------------------
// abs — d/dx |x| = sign(x); zero at x=0 by convention
// --------------------------------------------------------------------------

struct AbsBackward {
    saved_input: Tensor,
    #[allow(dead_code)] // captured for Phase C2-C9 backward dispatch refactor
    device: Device,
    edges: [Edge; 1],
}

impl Node for AbsBackward {
    fn name(&self) -> &'static str {
        "AbsBackward"
    }
    fn apply(&self, grad: &Tensor) -> Vec<Option<Tensor>> {
        let x = self.saved_input.as_slice::<f32>().expect("f32");
        let g = grad.as_slice::<f32>().expect("f32");
        let mut out = Vec::with_capacity(x.len());
        for i in 0..x.len() {
            let s = if x[i] > 0.0 {
                1.0_f32
            } else if x[i] < 0.0 {
                -1.0_f32
            } else {
                0.0_f32
            };
            out.push(g[i] * s);
        }
        let g_t = Tensor::from_vec(self.saved_input.shape().to_vec(), out).expect("abs bw shape");
        vec![Some(g_t)]
    }
    fn next_edges(&self) -> &[Edge] {
        &self.edges
    }
}

/// `abs(src)` (autograd-aware).
pub fn abs(src: &Variable) -> Result<Variable, BackwardError> {
    let out = pick_backend(src.device())
        .abs(&src.tensor())
        .map_err(|e| backend_err("abs", e))?;
    let mut out_var = Variable::new(out);
    if is_grad_enabled() && src.requires_grad {
        let node = std::sync::Arc::new(AbsBackward {
            saved_input: src.tensor().clone(),
            device: src.device(),
            edges: [src.edge()],
        });
        out_var.grad_fn = Some(node);
        out_var.requires_grad = true;
    }
    Ok(out_var)
}

// --------------------------------------------------------------------------
// pow_scalar — d/dx x^n = n * x^(n-1)
// --------------------------------------------------------------------------

struct PowScalarBackward {
    saved_input: Tensor,
    exponent: f64,
    device: Device,
    edges: [Edge; 1],
}

impl Node for PowScalarBackward {
    fn name(&self) -> &'static str {
        "PowScalarBackward"
    }
    fn apply(&self, grad: &Tensor) -> Vec<Option<Tensor>> {
        // n * x^(n-1) * grad
        let xn1 = pick_backend(self.device)
            .pow_scalar(&self.saved_input, self.exponent - 1.0)
            .expect("pow bw: x^(n-1)");
        let n = Tensor::scalar(self.exponent as f32);
        let scaled = pick_backend(self.device)
            .mul(&n, &xn1)
            .expect("pow bw: n*x^(n-1)");
        let g = pick_backend(self.device)
            .mul(grad, &scaled)
            .expect("pow bw: grad*scaled");
        vec![Some(g)]
    }
    fn next_edges(&self) -> &[Edge] {
        &self.edges
    }
}

/// `src.pow(exponent)` (scalar exponent, autograd-aware).
pub fn pow_scalar(src: &Variable, exponent: f64) -> Result<Variable, BackwardError> {
    let out = pick_backend(src.device())
        .pow_scalar(&src.tensor(), exponent)
        .map_err(|e| backend_err("pow_scalar", e))?;
    let mut out_var = Variable::new(out);
    if is_grad_enabled() && src.requires_grad {
        let node = std::sync::Arc::new(PowScalarBackward {
            saved_input: src.tensor().clone(),
            exponent,
            device: src.device(),
            edges: [src.edge()],
        });
        out_var.grad_fn = Some(node);
        out_var.requires_grad = true;
    }
    Ok(out_var)
}

// --------------------------------------------------------------------------
// index_select(src, dim=0, idx) — backward via scatter_add along dim 0
// (specialised for 1-D index tensor over a 2-D `src` of shape [V, D])
// --------------------------------------------------------------------------

struct IndexSelectBackward {
    in_shape: Vec<usize>, // src shape (V, D, ...)
    indices: Tensor,      // 1-D I64 of length K
    #[allow(dead_code)] // captured for Phase C2-C9 backward dispatch refactor
    device: Device,
    edges: [Edge; 1],
}

impl Node for IndexSelectBackward {
    fn name(&self) -> &'static str {
        "IndexSelectBackward"
    }
    fn apply(&self, grad: &Tensor) -> Vec<Option<Tensor>> {
        // For src shape [V, D...] and idx shape [K], grad has shape
        // [K, D...]. Backward: src_grad[idx[k]] += grad[k].
        let v = self.in_shape[0];
        let row_size: usize = self.in_shape[1..].iter().product();
        let row_size = row_size.max(1); // 1-D src (D=1) edge case
        let k = self.indices.numel();
        let g_data = grad.as_slice::<f32>().expect("f32 grad");
        let idx_data = self.indices.as_slice::<i64>().expect("i64 idx");
        let mut src_grad = vec![0.0_f32; v * row_size];
        for (k_i, &idx_v) in idx_data.iter().enumerate().take(k) {
            let v_i = idx_v as usize;
            // Add g_data[k_i * row_size .. (k_i+1) * row_size] to
            // src_grad[v_i * row_size .. (v_i+1) * row_size]
            let g_off = k_i * row_size;
            let s_off = v_i * row_size;
            for j in 0..row_size {
                src_grad[s_off + j] += g_data[g_off + j];
            }
        }
        let g_t = Tensor::from_vec(self.in_shape.clone(), src_grad).expect("index_select bw shape");
        vec![Some(g_t)]
    }
    fn next_edges(&self) -> &[Edge] {
        &self.edges
    }
}

/// `index_select(src, 0, indices)` — gather rows of `src` by 1-D I64
/// indices (autograd-aware; backward via scatter-add).
pub fn index_select(src: &Variable, indices: &Tensor) -> Result<Variable, BackwardError> {
    let out = pick_backend(src.device())
        .index_select(&src.tensor(), 0, indices)
        .map_err(|e| backend_err("index_select", e))?;
    let mut out_var = Variable::new(out);
    if is_grad_enabled() && src.requires_grad {
        let node = std::sync::Arc::new(IndexSelectBackward {
            in_shape: src.tensor().shape().to_vec(),
            indices: indices.clone(),
            device: src.device(),
            edges: [src.edge()],
        });
        out_var.grad_fn = Some(node);
        out_var.requires_grad = true;
    }
    Ok(out_var)
}

// --------------------------------------------------------------------------
// bmm — batched matmul: [B, M, K] @ [B, K, N] = [B, M, N]
// Forward now goes through the `Backend::bmm` trait method (added in P3.Y
// Phase A3); this module keeps `bmm_grad_inputs` for the backward path
// until Phase C1 wires backward through the trait too.
// Backward: dA[bi] = grad[bi] @ B[bi].T;  dB[bi] = A[bi].T @ grad[bi]
// --------------------------------------------------------------------------

struct BmmBackward {
    lhs_saved: Tensor,
    rhs_saved: Tensor,
    #[allow(dead_code)] // captured for Phase C2-C9 backward dispatch refactor
    device: Device,
    edges: [Edge; 2],
}

fn bmm_grad_inputs(grad: &Tensor, lhs: &Tensor, rhs: &Tensor) -> (Tensor, Tensor) {
    // dA[bi] = grad[bi] @ B[bi].T  → shape [B, M, K]
    // dB[bi] = A[bi].T @ grad[bi]  → shape [B, K, N]
    let l_shape = lhs.shape();
    let r_shape = rhs.shape();
    let b = l_shape[0];
    let m = l_shape[1];
    let k = l_shape[2];
    let n = r_shape[2];
    let g = grad.as_slice::<f32>().expect("f32 grad");
    let l = lhs.as_slice::<f32>().expect("f32 lhs");
    let r = rhs.as_slice::<f32>().expect("f32 rhs");

    let mut da = vec![0.0_f32; b * m * k];
    let mut db = vec![0.0_f32; b * k * n];

    for bi in 0..b {
        let g_off = bi * m * n;
        let l_off = bi * m * k;
        let r_off = bi * k * n;
        // dA[bi] = grad[bi] @ B[bi].T: shape [M, K]
        for i in 0..m {
            for kk in 0..k {
                let mut acc = 0.0_f32;
                for j in 0..n {
                    acc += g[g_off + i * n + j] * r[r_off + kk * n + j];
                }
                da[l_off + i * k + kk] = acc;
            }
        }
        // dB[bi] = A[bi].T @ grad[bi]: shape [K, N]
        for kk in 0..k {
            for j in 0..n {
                let mut acc = 0.0_f32;
                for i in 0..m {
                    acc += l[l_off + i * k + kk] * g[g_off + i * n + j];
                }
                db[r_off + kk * n + j] = acc;
            }
        }
    }

    let da_t = Tensor::from_vec([b, m, k], da).expect("dA build");
    let db_t = Tensor::from_vec([b, k, n], db).expect("dB build");
    (da_t, db_t)
}

impl Node for BmmBackward {
    fn name(&self) -> &'static str {
        "BmmBackward"
    }
    fn apply(&self, grad: &Tensor) -> Vec<Option<Tensor>> {
        let (da, db) = bmm_grad_inputs(grad, &self.lhs_saved, &self.rhs_saved);
        vec![Some(da), Some(db)]
    }
    fn next_edges(&self) -> &[Edge] {
        &self.edges
    }
}

// --------------------------------------------------------------------------
// transpose(d0, d1) — swap two axes; backward is itself a transpose
// --------------------------------------------------------------------------

struct TransposeBackward {
    d0: usize,
    d1: usize,
    #[allow(dead_code)] // captured for Phase C2-C9 backward dispatch refactor
    device: Device,
    edges: [Edge; 1],
}

impl Node for TransposeBackward {
    fn name(&self) -> &'static str {
        "TransposeBackward"
    }
    fn apply(&self, grad: &Tensor) -> Vec<Option<Tensor>> {
        // Backward of transpose(d0, d1) is transpose(d0, d1) again on
        // the upstream gradient. Materialise contiguous so downstream
        // ops that require contiguous F32 don't break.
        let g = grad
            .transpose(self.d0, self.d1)
            .expect("transpose bw: dim valid")
            .contiguous();
        vec![Some(g)]
    }
    fn next_edges(&self) -> &[Edge] {
        &self.edges
    }
}

/// `src.transpose(d0, d1)` (autograd-aware, contiguous output).
pub fn transpose(src: &Variable, d0: usize, d1: usize) -> Result<Variable, BackwardError> {
    let dev = src.device();
    let out_view = src
        .tensor()
        .transpose(d0, d1)
        .map_err(|e| BackwardError::Backend {
            op: "transpose",
            message: format!("{e}"),
        })?;
    // `.contiguous()` may strip the device tag (it materialises through the
    // CPU layout). Preserve it explicitly so autograd dispatch stays consistent.
    let out = out_view.contiguous().with_device(dev);
    let mut out_var = Variable::new(out);
    if is_grad_enabled() && src.requires_grad {
        let node = std::sync::Arc::new(TransposeBackward {
            d0,
            d1,
            device: src.device(),
            edges: [src.edge()],
        });
        out_var.grad_fn = Some(node);
        out_var.requires_grad = true;
    }
    Ok(out_var)
}

// --------------------------------------------------------------------------
// reshape — view-only op; backward reshapes the gradient to in_shape.
// (Requires the input to be contiguous; non-contiguous inputs need an
// explicit `.contiguous()` first. Same constraint PyTorch's `view` has.)
// --------------------------------------------------------------------------

struct ReshapeBackward {
    in_shape: Vec<usize>,
    #[allow(dead_code)] // captured for Phase C2-C9 backward dispatch refactor
    device: Device,
    edges: [Edge; 1],
}

impl Node for ReshapeBackward {
    fn name(&self) -> &'static str {
        "ReshapeBackward"
    }
    fn apply(&self, grad: &Tensor) -> Vec<Option<Tensor>> {
        // Reshape grad back to input shape. Since reshape is a view op
        // we materialise the bytes via from_vec.
        let data = grad.as_slice::<f32>().expect("f32 grad").to_vec();
        let g = Tensor::from_vec(self.in_shape.clone(), data).expect("reshape bw shape");
        vec![Some(g)]
    }
    fn next_edges(&self) -> &[Edge] {
        &self.edges
    }
}

// --------------------------------------------------------------------------
// conv2d — stride=1, configurable padding, no dilation/groups
// Backward via dedicated cpu kernels (grad_input + grad_weight).
// --------------------------------------------------------------------------

struct Conv2dBackward {
    input_saved: Tensor,
    weight_saved: Tensor,
    pad_h: usize,
    pad_w: usize,
    has_bias: bool,
    #[allow(dead_code)] // captured for Phase C2-C9 backward dispatch refactor
    device: Device,
    edges: Vec<Edge>,
}

impl Node for Conv2dBackward {
    fn name(&self) -> &'static str {
        "Conv2dBackward"
    }
    fn apply(&self, grad: &Tensor) -> Vec<Option<Tensor>> {
        let din = rustorch_cpu::kernels::conv::conv2d_grad_input(
            grad,
            &self.weight_saved,
            self.input_saved.shape(),
            self.pad_h,
            self.pad_w,
        )
        .expect("conv2d grad_input");
        let dw = rustorch_cpu::kernels::conv::conv2d_grad_weight(
            &self.input_saved,
            grad,
            self.weight_saved.shape(),
            self.pad_h,
            self.pad_w,
        )
        .expect("conv2d grad_weight");
        let mut grads: Vec<Option<Tensor>> = vec![Some(din), Some(dw)];
        if self.has_bias {
            // Bias grad: sum over batch + spatial dims, leaving [C_out].
            let g_data = grad.as_slice::<f32>().expect("f32 grad");
            let n = grad.shape()[0];
            let c_out = grad.shape()[1];
            let h_out = grad.shape()[2];
            let w_out = grad.shape()[3];
            let mut db = vec![0.0_f32; c_out];
            for ni in 0..n {
                for (co, db_co) in db.iter_mut().enumerate() {
                    for hi in 0..h_out {
                        for wi in 0..w_out {
                            let idx = ((ni * c_out + co) * h_out + hi) * w_out + wi;
                            *db_co += g_data[idx];
                        }
                    }
                }
            }
            let db_t = Tensor::from_vec([c_out], db).expect("bias grad");
            grads.push(Some(db_t));
        }
        grads
    }
    fn next_edges(&self) -> &[Edge] {
        &self.edges
    }
}

// --------------------------------------------------------------------------
// batch_norm2d — train mode; running stats handled at the Module level.
// --------------------------------------------------------------------------

struct BatchNorm2dBackward {
    input_saved: Tensor,
    gamma_saved: Tensor,
    saved_mean: Tensor,
    saved_var: Tensor,
    eps: f32,
    #[allow(dead_code)] // captured for Phase C2-C9 backward dispatch refactor
    device: Device,
    edges: [Edge; 3],
}

impl Node for BatchNorm2dBackward {
    fn name(&self) -> &'static str {
        "BatchNorm2dBackward"
    }
    fn apply(&self, grad: &Tensor) -> Vec<Option<Tensor>> {
        let (dx, dg, db) = rustorch_cpu::kernels::norm::batch_norm2d_backward(
            grad,
            &self.input_saved,
            &self.gamma_saved,
            &self.saved_mean,
            &self.saved_var,
            self.eps,
        )
        .expect("bn2d backward");
        vec![Some(dx), Some(dg), Some(db)]
    }
    fn next_edges(&self) -> &[Edge] {
        &self.edges
    }
}

/// Autograd-aware BatchNorm2d (train mode). Returns the normalised
/// output; running-stats updates are handled at the Module layer.
pub fn batch_norm2d(
    input: &Variable,
    gamma: &Variable,
    beta: &Variable,
    eps: f32,
) -> Result<Variable, BackwardError> {
    let (out, mean, var) = rustorch_cpu::kernels::norm::batch_norm2d_forward(
        &input.tensor(),
        &gamma.tensor(),
        &beta.tensor(),
        eps,
    )
    .map_err(|e| backend_err("batch_norm2d", e))?;
    let mut out_var = Variable::new(out);
    let any_grad = input.requires_grad || gamma.requires_grad || beta.requires_grad;
    if is_grad_enabled() && any_grad {
        let node = std::sync::Arc::new(BatchNorm2dBackward {
            input_saved: input.tensor().clone(),
            gamma_saved: gamma.tensor().clone(),
            saved_mean: mean,
            saved_var: var,
            eps,
            device: input.device(),
            edges: [input.edge(), gamma.edge(), beta.edge()],
        });
        out_var.grad_fn = Some(node);
        out_var.requires_grad = true;
    }
    Ok(out_var)
}

// --------------------------------------------------------------------------
// maxpool2d — backward scatters grad to argmax positions
// --------------------------------------------------------------------------

struct MaxPool2dBackward {
    in_shape: Vec<usize>,
    argmax_idx: Tensor,
    #[allow(dead_code)] // captured for Phase C2-C9 backward dispatch refactor
    device: Device,
    edges: [Edge; 1],
}

impl Node for MaxPool2dBackward {
    fn name(&self) -> &'static str {
        "MaxPool2dBackward"
    }
    fn apply(&self, grad: &Tensor) -> Vec<Option<Tensor>> {
        let din =
            rustorch_cpu::kernels::pool::maxpool2d_backward(grad, &self.argmax_idx, &self.in_shape)
                .expect("maxpool2d backward");
        vec![Some(din)]
    }
    fn next_edges(&self) -> &[Edge] {
        &self.edges
    }
}

/// Autograd-aware MaxPool2d. Returns the pooled output; argmax indices
/// are saved internally for backward.
pub fn max_pool2d(
    input: &Variable,
    kernel_size: (usize, usize),
    stride: (usize, usize),
) -> Result<Variable, BackwardError> {
    let (out, idx) =
        rustorch_cpu::kernels::pool::maxpool2d_forward(&input.tensor(), kernel_size, stride)
            .map_err(|e| backend_err("max_pool2d", e))?;
    let mut out_var = Variable::new(out);
    if is_grad_enabled() && input.requires_grad {
        let node = std::sync::Arc::new(MaxPool2dBackward {
            in_shape: input.tensor().shape().to_vec(),
            argmax_idx: idx,
            device: input.device(),
            edges: [input.edge()],
        });
        out_var.grad_fn = Some(node);
        out_var.requires_grad = true;
    }
    Ok(out_var)
}

/// Autograd-aware conv2d (stride=1, configurable padding).
pub fn conv2d(
    input: &Variable,
    weight: &Variable,
    bias: Option<&Variable>,
    pad_h: usize,
    pad_w: usize,
) -> Result<Variable, BackwardError> {
    let bias_t = bias.map(|b| b.tensor());
    let bias_ref = bias_t.as_ref();
    let out = rustorch_cpu::kernels::conv::conv2d_forward(
        &input.tensor(),
        &weight.tensor(),
        bias_ref,
        pad_h,
        pad_w,
    )
    .map_err(|e| backend_err("conv2d", e))?;
    let mut out_var = Variable::new(out);
    let any_grad = input.requires_grad
        || weight.requires_grad
        || bias.map(|b| b.requires_grad).unwrap_or(false);
    if is_grad_enabled() && any_grad {
        let mut edges = vec![input.edge(), weight.edge()];
        if let Some(b) = bias {
            edges.push(b.edge());
        }
        let node = std::sync::Arc::new(Conv2dBackward {
            input_saved: input.tensor().clone(),
            weight_saved: weight.tensor().clone(),
            pad_h,
            pad_w,
            has_bias: bias.is_some(),
            device: input.device(),
            edges,
        });
        out_var.grad_fn = Some(node);
        out_var.requires_grad = true;
    }
    Ok(out_var)
}

/// Reshape `src` to the given shape (autograd-aware view; F32-only).
/// Requires `numel(src) == numel(shape)`.
pub fn reshape(src: &Variable, shape: Vec<usize>) -> Result<Variable, BackwardError> {
    let src_t = src.tensor();
    let numel: usize = shape.iter().product();
    if numel != src_t.numel() {
        return Err(BackwardError::Backend {
            op: "reshape",
            message: format!(
                "shape {:?} has {numel} elements, src has {} elements",
                shape,
                src_t.numel()
            ),
        });
    }
    let device = src.device();
    // Pull the data via a CPU view, build the reshaped tensor, then re-tag
    // it with the source device so autograd dispatch keeps routing to the
    // right backend.
    let src_cpu = src_t
        .clone()
        .with_device(rustorch_core::tensor::device::Device::Cpu);
    let data = src_cpu
        .as_slice::<f32>()
        .ok_or_else(|| BackwardError::Backend {
            op: "reshape",
            message: "expected contiguous F32".to_string(),
        })?;
    let out = Tensor::from_vec(shape, data.to_vec()).map_err(|e| BackwardError::Backend {
        op: "reshape",
        message: format!("{e}"),
    })?;
    let out = out.with_device(device);
    let mut out_var = Variable::new(out);
    if is_grad_enabled() && src.requires_grad {
        let node = std::sync::Arc::new(ReshapeBackward {
            in_shape: src_t.shape().to_vec(),
            device: src.device(),
            edges: [src.edge()],
        });
        out_var.grad_fn = Some(node);
        out_var.requires_grad = true;
    }
    Ok(out_var)
}

/// Batched matmul `[B, M, K] @ [B, K, N] = [B, M, N]` (autograd-aware).
pub fn bmm(lhs: &Variable, rhs: &Variable) -> Result<Variable, BackwardError> {
    let device = require_same_device_2("bmm", lhs, rhs)?;
    let out = pick_backend(device)
        .bmm(&lhs.tensor(), &rhs.tensor())
        .map_err(|e| backend_err("bmm", e))?;
    let mut out_var = Variable::new(out);
    if is_grad_enabled() && (lhs.requires_grad || rhs.requires_grad) {
        let node = std::sync::Arc::new(BmmBackward {
            lhs_saved: lhs.tensor().clone(),
            rhs_saved: rhs.tensor().clone(),
            device,
            edges: [lhs.edge(), rhs.edge()],
        });
        out_var.grad_fn = Some(node);
        out_var.requires_grad = true;
    }
    Ok(out_var)
}

// --------------------------------------------------------------------------
// mean_dim — d/dx mean(x, dim, keepdim=true) broadcasts grad/N back to x.shape
// --------------------------------------------------------------------------

struct MeanDimBackward {
    in_shape: Vec<usize>,
    n_reduced: usize,
    device: Device,
    edges: [Edge; 1],
}

impl Node for MeanDimBackward {
    fn name(&self) -> &'static str {
        "MeanDimBackward"
    }
    fn apply(&self, grad: &Tensor) -> Vec<Option<Tensor>> {
        // Broadcast grad up to in_shape and divide by N.
        // We do this by computing `mul(grad, ones_in/N)` — the ones tensor
        // has the input shape, so the broadcast multiplies grad (with its
        // size-1 reduced dims) up to the full shape.
        let inv_n = 1.0_f32 / self.n_reduced as f32;
        let numel: usize = self.in_shape.iter().product();
        let scaled_ones =
            Tensor::from_vec(self.in_shape.clone(), vec![inv_n; numel]).expect("scaled ones");
        let g = pick_backend(self.device)
            .mul(grad, &scaled_ones)
            .expect("mean_dim bw: grad * ones/N");
        vec![Some(g)]
    }
    fn next_edges(&self) -> &[Edge] {
        &self.edges
    }
}

/// `mean(src, dims, keepdim=true)` (autograd-aware; v1 requires keepdim=true).
pub fn mean_dim(src: &Variable, dims: &[usize]) -> Result<Variable, BackwardError> {
    let out = pick_backend(src.device())
        .mean_dim(&src.tensor(), dims, true)
        .map_err(|e| backend_err("mean_dim", e))?;
    let mut out_var = Variable::new(out);
    if is_grad_enabled() && src.requires_grad {
        let in_shape = src.tensor().shape().to_vec();
        // N = product of reduced dim sizes
        let n_reduced: usize = dims.iter().map(|&d| in_shape[d]).product();
        let node = std::sync::Arc::new(MeanDimBackward {
            in_shape,
            n_reduced,
            device: src.device(),
            edges: [src.edge()],
        });
        out_var.grad_fn = Some(node);
        out_var.requires_grad = true;
    }
    Ok(out_var)
}

// --------------------------------------------------------------------------
// l2_normalize — `x / max(eps, sqrt(sum(x², dim)))` along a single axis.
//
// Composed entirely from existing autograd-aware ops:
//   square     = x * x          (via mul)
//   mean_sq    = mean_dim(square, dim)        // shape: [..., 1, ...]
//   sum_sq     = mean_sq * N                  // restore sum from mean
//   norm_sq    = sum_sq + eps                 // numerical stability
//   norm       = sqrt(norm_sq)
//   out        = x / norm                     // broadcast div
//
// This avoids extending the autograd surface with `sum_dim`. The
// gradient flows through every step via the existing backward formulas.
// --------------------------------------------------------------------------

/// L2-normalise `src` along `dim`. Equivalent to PyTorch's
/// `F.normalize(x, dim=dim, p=2)`.
///
/// Returns `x / sqrt(sum(x², dim, keepdim=true) + eps)` where `eps`
/// defaults to `1e-12`. Use [`l2_normalize_with_eps`] to override.
///
/// Numerical safety: for an all-zero input vector along `dim`, the
/// result is `x / sqrt(eps) ≈ 0/sqrt(1e-12)` which is finite. The
/// epsilon prevents division-by-zero NaN.
pub fn l2_normalize(src: &Variable, dim: usize) -> Result<Variable, BackwardError> {
    l2_normalize_with_eps(src, dim, 1e-12)
}

/// L2-normalise along `dim` with a caller-provided epsilon. See
/// [`l2_normalize`] for the formula.
pub fn l2_normalize_with_eps(
    src: &Variable,
    dim: usize,
    eps: f32,
) -> Result<Variable, BackwardError> {
    let in_shape = src.tensor().shape().to_vec();
    if dim >= in_shape.len() {
        return Err(BackwardError::Backend {
            op: "l2_normalize",
            message: format!("dim {} out of range for shape {:?}", dim, in_shape),
        });
    }
    let n_reduced = in_shape[dim];
    // square = x * x
    let dev = src.tensor().device();
    let square = mul(src, src)?;
    // mean over the reduced dim (keepdim=true)
    let mean_sq = mean_dim(&square, &[dim])?;
    // sum_sq = mean_sq * N  (restore sum from mean; N is a constant scalar
    // placed on the same device so autograd dispatch matches).
    let n_scalar = Variable::new(Tensor::scalar(n_reduced as f32).with_device(dev));
    let sum_sq = mul(&mean_sq, &n_scalar)?;
    // norm_sq = sum_sq + eps  (eps on the same device too)
    let eps_scalar = Variable::new(Tensor::scalar(eps).with_device(dev));
    let norm_sq = add(&sum_sq, &eps_scalar)?;
    // norm = sqrt(norm_sq)
    let norm = sqrt(&norm_sq)?;
    // out = x / norm  (broadcast over the kept-1 dim)
    div(src, &norm)
}
