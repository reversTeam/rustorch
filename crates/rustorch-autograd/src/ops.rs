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
use crate::node::{Edge, Node};
use crate::tape::is_grad_enabled;
use crate::variable::Variable;
use rustorch_core::tensor::tensor_impl::Tensor;
use rustorch_cpu::backend::Reduction;
use rustorch_cpu::cpu_backend::cpu_backend;

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
    edges: [Edge; 2],
}

impl Node for AddBackward {
    fn name(&self) -> &'static str {
        "AddBackward"
    }
    fn apply(&self, grad: &Tensor) -> Vec<Option<Tensor>> {
        // Both inputs receive the same upstream gradient.
        vec![Some(grad.clone()), Some(grad.clone())]
    }
    fn next_edges(&self) -> &[Edge] {
        &self.edges
    }
}

/// `lhs + rhs` (autograd-aware).
pub fn add(lhs: &Variable, rhs: &Variable) -> Result<Variable, BackwardError> {
    let out = cpu_backend()
        .add(&lhs.tensor(), &rhs.tensor())
        .map_err(|e| backend_err("add", e))?;
    let mut out = Variable::new(out);
    if is_grad_enabled() && (lhs.requires_grad || rhs.requires_grad) {
        let node = std::sync::Arc::new(AddBackward {
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
    edges: [Edge; 2],
}

impl Node for SubBackward {
    fn name(&self) -> &'static str {
        "SubBackward"
    }
    fn apply(&self, grad: &Tensor) -> Vec<Option<Tensor>> {
        let neg_grad = cpu_backend().neg(grad).expect("neg never fails on f32/f64");
        vec![Some(grad.clone()), Some(neg_grad)]
    }
    fn next_edges(&self) -> &[Edge] {
        &self.edges
    }
}

/// `lhs - rhs`.
pub fn sub(lhs: &Variable, rhs: &Variable) -> Result<Variable, BackwardError> {
    let out = cpu_backend()
        .sub(&lhs.tensor(), &rhs.tensor())
        .map_err(|e| backend_err("sub", e))?;
    let mut out = Variable::new(out);
    if is_grad_enabled() && (lhs.requires_grad || rhs.requires_grad) {
        let node = std::sync::Arc::new(SubBackward {
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
    edges: [Edge; 2],
}

impl Node for MulBackward {
    fn name(&self) -> &'static str {
        "MulBackward"
    }
    fn apply(&self, grad: &Tensor) -> Vec<Option<Tensor>> {
        let g_lhs = cpu_backend()
            .mul(grad, &self.rhs_saved)
            .expect("mul backward");
        let g_rhs = cpu_backend()
            .mul(grad, &self.lhs_saved)
            .expect("mul backward");
        vec![Some(g_lhs), Some(g_rhs)]
    }
    fn next_edges(&self) -> &[Edge] {
        &self.edges
    }
}

/// `lhs * rhs`.
pub fn mul(lhs: &Variable, rhs: &Variable) -> Result<Variable, BackwardError> {
    let out = cpu_backend()
        .mul(&lhs.tensor(), &rhs.tensor())
        .map_err(|e| backend_err("mul", e))?;
    let mut out = Variable::new(out);
    if is_grad_enabled() && (lhs.requires_grad || rhs.requires_grad) {
        let node = std::sync::Arc::new(MulBackward {
            lhs_saved: lhs.tensor().clone(),
            rhs_saved: rhs.tensor().clone(),
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
    edges: [Edge; 1],
}

impl Node for NegBackward {
    fn name(&self) -> &'static str {
        "NegBackward"
    }
    fn apply(&self, grad: &Tensor) -> Vec<Option<Tensor>> {
        vec![Some(cpu_backend().neg(grad).expect("neg backward"))]
    }
    fn next_edges(&self) -> &[Edge] {
        &self.edges
    }
}

/// `-src`.
pub fn neg(src: &Variable) -> Result<Variable, BackwardError> {
    let out = cpu_backend()
        .neg(&src.tensor())
        .map_err(|e| backend_err("neg", e))?;
    let mut out = Variable::new(out);
    if is_grad_enabled() && src.requires_grad {
        let node = std::sync::Arc::new(NegBackward {
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
    edges: [Edge; 2],
}

impl Node for MatMulBackward {
    fn name(&self) -> &'static str {
        "MatMulBackward"
    }
    fn apply(&self, grad: &Tensor) -> Vec<Option<Tensor>> {
        let rhs_t = self.rhs_saved.transpose(0, 1).expect("transpose");
        let lhs_t = self.lhs_saved.transpose(0, 1).expect("transpose");
        let g_lhs = cpu_backend().matmul(grad, &rhs_t).expect("matmul lhs grad");
        let g_rhs = cpu_backend().matmul(&lhs_t, grad).expect("matmul rhs grad");
        vec![Some(g_lhs), Some(g_rhs)]
    }
    fn next_edges(&self) -> &[Edge] {
        &self.edges
    }
}

/// `lhs @ rhs` (rank-2 matmul).
pub fn matmul(lhs: &Variable, rhs: &Variable) -> Result<Variable, BackwardError> {
    let out = cpu_backend()
        .matmul(&lhs.tensor(), &rhs.tensor())
        .map_err(|e| backend_err("matmul", e))?;
    let mut out = Variable::new(out);
    if is_grad_enabled() && (lhs.requires_grad || rhs.requires_grad) {
        // matmul backward needs the contiguous transposes; clone the
        // inputs to keep them alive for backward.
        let node = std::sync::Arc::new(MatMulBackward {
            lhs_saved: lhs.tensor().clone(),
            rhs_saved: rhs.tensor().clone(),
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
    edges: [Edge; 1],
}

impl Node for ReluBackward {
    fn name(&self) -> &'static str {
        "ReluBackward"
    }
    fn apply(&self, grad: &Tensor) -> Vec<Option<Tensor>> {
        // mask = (saved > 0); g = grad * mask (with mask cast to grad dtype)
        let zero = Tensor::scalar(0.0);
        let mask = cpu_backend()
            .gt(&self.saved, &zero)
            .expect("gt for relu mask");
        let mask_f = mask.to_dtype(grad.dtype());
        let g = cpu_backend().mul(grad, &mask_f).expect("relu mask mul");
        vec![Some(g)]
    }
    fn next_edges(&self) -> &[Edge] {
        &self.edges
    }
}

/// `relu(src)`.
pub fn relu(src: &Variable) -> Result<Variable, BackwardError> {
    let out = cpu_backend()
        .relu(&src.tensor())
        .map_err(|e| backend_err("relu", e))?;
    let mut out = Variable::new(out);
    if is_grad_enabled() && src.requires_grad {
        let node = std::sync::Arc::new(ReluBackward {
            saved: src.tensor().clone(),
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
    edges: [Edge; 1],
}

impl Node for SigmoidBackward {
    fn name(&self) -> &'static str {
        "SigmoidBackward"
    }
    fn apply(&self, grad: &Tensor) -> Vec<Option<Tensor>> {
        // g = grad * out * (1 - out)
        let one = Tensor::scalar(1.0);
        let one_minus = cpu_backend().sub(&one, &self.saved_out).expect("1-out");
        let s_times_one_minus = cpu_backend()
            .mul(&self.saved_out, &one_minus)
            .expect("s*(1-s)");
        let g = cpu_backend()
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
    let out = cpu_backend()
        .sigmoid(&src.tensor())
        .map_err(|e| backend_err("sigmoid", e))?;
    let mut out_var = Variable::new(out.clone());
    if is_grad_enabled() && src.requires_grad {
        let node = std::sync::Arc::new(SigmoidBackward {
            saved_out: out,
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
    edges: [Edge; 1],
}

impl Node for TanhBackward {
    fn name(&self) -> &'static str {
        "TanhBackward"
    }
    fn apply(&self, grad: &Tensor) -> Vec<Option<Tensor>> {
        let one = Tensor::scalar(1.0);
        let sq = cpu_backend()
            .mul(&self.saved_out, &self.saved_out)
            .expect("tanh^2");
        let one_minus_sq = cpu_backend().sub(&one, &sq).expect("1-tanh^2");
        let g = cpu_backend()
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
    let out = cpu_backend()
        .tanh(&src.tensor())
        .map_err(|e| backend_err("tanh", e))?;
    let mut out_var = Variable::new(out.clone());
    if is_grad_enabled() && src.requires_grad {
        let node = std::sync::Arc::new(TanhBackward {
            saved_out: out,
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
    let out = cpu_backend()
        .sum(&src.tensor())
        .map_err(|e| backend_err("sum", e))?;
    let mut out_var = Variable::new(out);
    if is_grad_enabled() && src.requires_grad {
        let node = std::sync::Arc::new(SumBackward {
            in_shape: src.tensor().shape().to_vec(),
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
    let out = cpu_backend()
        .mean(&src.tensor())
        .map_err(|e| backend_err("mean", e))?;
    let mut out_var = Variable::new(out);
    if is_grad_enabled() && src.requires_grad {
        let node = std::sync::Arc::new(MeanBackward {
            in_shape: src.tensor().shape().to_vec(),
            n: src.tensor().numel(),
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
    let loss = cpu_backend()
        .cross_entropy(&input.tensor(), &target.tensor(), reduction)
        .map_err(|e| backend_err("cross_entropy", e))?;
    let mut out_var = Variable::new(loss);
    if is_grad_enabled() && input.requires_grad {
        // For backward we save the softmax (not log_softmax) so that
        // grad = (softmax - one_hot(target)) / N.
        let softmax = cpu_backend()
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
        let g_bias = cpu_backend()
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
    // Forward: broadcast bias along axis 0.
    let bias_buf: &[f32] = bias_t.as_slice::<f32>().expect("f32");
    let mut wide = Vec::with_capacity(batch * n_out);
    for _ in 0..batch {
        wide.extend_from_slice(bias_buf);
    }
    let bias_wide = Tensor::from_vec([batch, n_out], wide).expect("bias broadcast shape");
    let out = cpu_backend()
        .add(&x_t, &bias_wide)
        .map_err(|e| backend_err("add_bias", e))?;
    let mut out_var = Variable::new(out);
    if is_grad_enabled() && (x.requires_grad || bias.requires_grad) {
        let node = std::sync::Arc::new(AddBiasBackward {
            bias_shape: bias_t.shape().to_vec(),
            edges: [x.edge(), bias.edge()],
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
    edges: [Edge; 2],
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
        let g_x = cpu_backend()
            .mul(&self.diff, &scale_t)
            .expect("mse grad mul");
        let g_y = cpu_backend().neg(&g_x).expect("mse grad neg");
        vec![Some(g_x), Some(g_y)]
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
    let loss = cpu_backend()
        .mse_loss(&input.tensor(), &target.tensor(), reduction)
        .map_err(|e| backend_err("mse_loss", e))?;
    let mut out_var = Variable::new(loss);
    if is_grad_enabled() && (input.requires_grad || target.requires_grad) {
        let diff = cpu_backend()
            .sub(&input.tensor(), &target.tensor())
            .map_err(|e| backend_err("mse_loss:sub", e))?;
        let node = std::sync::Arc::new(MseBackward {
            diff,
            n: input.tensor().numel(),
            reduction,
            edges: [input.edge(), target.edge()],
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
    edges: [Edge; 1],
}

impl Node for SiluBackward {
    fn name(&self) -> &'static str {
        "SiluBackward"
    }
    fn apply(&self, grad: &Tensor) -> Vec<Option<Tensor>> {
        // s = sigmoid(x); local = s * (1 + x * (1 - s))
        let s = cpu_backend()
            .sigmoid(&self.saved_input)
            .expect("silu bw: sigmoid");
        let one = Tensor::scalar(1.0);
        let one_minus_s = cpu_backend().sub(&one, &s).expect("silu bw: 1-s");
        let x_one_minus_s = cpu_backend()
            .mul(&self.saved_input, &one_minus_s)
            .expect("silu bw: x*(1-s)");
        let inner = cpu_backend()
            .add(&one, &x_one_minus_s)
            .expect("silu bw: 1 + x*(1-s)");
        let local = cpu_backend().mul(&s, &inner).expect("silu bw: local");
        let g = cpu_backend()
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
    let out = cpu_backend()
        .silu(&src.tensor())
        .map_err(|e| backend_err("silu", e))?;
    let mut out_var = Variable::new(out);
    if is_grad_enabled() && src.requires_grad {
        let node = std::sync::Arc::new(SiluBackward {
            saved_input: src.tensor().clone(),
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
    let out = cpu_backend()
        .leaky_relu(&src.tensor(), slope)
        .map_err(|e| backend_err("leaky_relu", e))?;
    let mut out_var = Variable::new(out);
    if is_grad_enabled() && src.requires_grad {
        let node = std::sync::Arc::new(LeakyReluBackward {
            saved_input: src.tensor().clone(),
            slope,
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
    edges: [Edge; 1],
}

impl Node for SoftmaxBackward {
    fn name(&self) -> &'static str {
        "SoftmaxBackward"
    }
    fn apply(&self, grad: &Tensor) -> Vec<Option<Tensor>> {
        // dot = sum_along_dim(grad * y, dim, keepdim=true)
        let gy = cpu_backend()
            .mul(grad, &self.saved_out)
            .expect("softmax bw: g*y");
        let dot = cpu_backend()
            .sum_dim(&gy, &[self.dim], true)
            .expect("softmax bw: sum_dim");
        // diff = grad - dot (broadcast)
        let diff = cpu_backend().sub(grad, &dot).expect("softmax bw: g - dot");
        let g = cpu_backend()
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
    let out = cpu_backend()
        .softmax(&src.tensor(), dim)
        .map_err(|e| backend_err("softmax", e))?;
    let mut out_var = Variable::new(out.clone());
    if is_grad_enabled() && src.requires_grad {
        let node = std::sync::Arc::new(SoftmaxBackward {
            saved_out: out,
            dim,
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
    edges: [Edge; 1],
}

impl Node for LogSoftmaxBackward {
    fn name(&self) -> &'static str {
        "LogSoftmaxBackward"
    }
    fn apply(&self, grad: &Tensor) -> Vec<Option<Tensor>> {
        // softmax = exp(log_softmax); grad_x = grad - softmax * sum_along(grad, dim, keepdim)
        let s = cpu_backend()
            .exp(&self.saved_out)
            .expect("log_softmax bw: exp");
        let sum_g = cpu_backend()
            .sum_dim(grad, &[self.dim], true)
            .expect("log_softmax bw: sum_dim");
        let s_sum = cpu_backend()
            .mul(&s, &sum_g)
            .expect("log_softmax bw: s*sum_g");
        let g = cpu_backend()
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
    let out = cpu_backend()
        .log_softmax(&src.tensor(), dim)
        .map_err(|e| backend_err("log_softmax", e))?;
    let mut out_var = Variable::new(out.clone());
    if is_grad_enabled() && src.requires_grad {
        let node = std::sync::Arc::new(LogSoftmaxBackward {
            saved_out: out,
            dim,
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
    edges: [Edge; 2],
}

impl Node for DivBackward {
    fn name(&self) -> &'static str {
        "DivBackward"
    }
    fn apply(&self, grad: &Tensor) -> Vec<Option<Tensor>> {
        let g_lhs = cpu_backend()
            .div(grad, &self.rhs_saved)
            .expect("div bw: g/rhs");
        // dy = -x/y² * grad = -lhs * grad / (rhs*rhs)
        let rhs_sq = cpu_backend()
            .mul(&self.rhs_saved, &self.rhs_saved)
            .expect("div bw: rhs²");
        let lhs_grad = cpu_backend()
            .mul(&self.lhs_saved, grad)
            .expect("div bw: x*g");
        let div_term = cpu_backend()
            .div(&lhs_grad, &rhs_sq)
            .expect("div bw: x*g/rhs²");
        let g_rhs = cpu_backend().neg(&div_term).expect("div bw: neg");
        vec![Some(g_lhs), Some(g_rhs)]
    }
    fn next_edges(&self) -> &[Edge] {
        &self.edges
    }
}

/// `lhs / rhs` (autograd-aware).
pub fn div(lhs: &Variable, rhs: &Variable) -> Result<Variable, BackwardError> {
    let out = cpu_backend()
        .div(&lhs.tensor(), &rhs.tensor())
        .map_err(|e| backend_err("div", e))?;
    let mut out_var = Variable::new(out);
    if is_grad_enabled() && (lhs.requires_grad || rhs.requires_grad) {
        let node = std::sync::Arc::new(DivBackward {
            lhs_saved: lhs.tensor().clone(),
            rhs_saved: rhs.tensor().clone(),
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
    edges: [Edge; 1],
}

impl Node for ExpBackward {
    fn name(&self) -> &'static str {
        "ExpBackward"
    }
    fn apply(&self, grad: &Tensor) -> Vec<Option<Tensor>> {
        let g = cpu_backend()
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
    let out = cpu_backend()
        .exp(&src.tensor())
        .map_err(|e| backend_err("exp", e))?;
    let mut out_var = Variable::new(out.clone());
    if is_grad_enabled() && src.requires_grad {
        let node = std::sync::Arc::new(ExpBackward {
            saved_out: out,
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
    edges: [Edge; 1],
}

impl Node for LogBackward {
    fn name(&self) -> &'static str {
        "LogBackward"
    }
    fn apply(&self, grad: &Tensor) -> Vec<Option<Tensor>> {
        let g = cpu_backend()
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
    let out = cpu_backend()
        .log(&src.tensor())
        .map_err(|e| backend_err("log", e))?;
    let mut out_var = Variable::new(out);
    if is_grad_enabled() && src.requires_grad {
        let node = std::sync::Arc::new(LogBackward {
            saved_input: src.tensor().clone(),
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
            cpu_backend()
                .mul(&two, &self.saved_out)
                .expect("sqrt bw: 2*out")
        };
        let g = cpu_backend()
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
    let out = cpu_backend()
        .sqrt(&src.tensor())
        .map_err(|e| backend_err("sqrt", e))?;
    let mut out_var = Variable::new(out.clone());
    if is_grad_enabled() && src.requires_grad {
        let node = std::sync::Arc::new(SqrtBackward {
            saved_out: out,
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
    let out = cpu_backend()
        .abs(&src.tensor())
        .map_err(|e| backend_err("abs", e))?;
    let mut out_var = Variable::new(out);
    if is_grad_enabled() && src.requires_grad {
        let node = std::sync::Arc::new(AbsBackward {
            saved_input: src.tensor().clone(),
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
    edges: [Edge; 1],
}

impl Node for PowScalarBackward {
    fn name(&self) -> &'static str {
        "PowScalarBackward"
    }
    fn apply(&self, grad: &Tensor) -> Vec<Option<Tensor>> {
        // n * x^(n-1) * grad
        let xn1 = cpu_backend()
            .pow_scalar(&self.saved_input, self.exponent - 1.0)
            .expect("pow bw: x^(n-1)");
        let n = Tensor::scalar(self.exponent as f32);
        let scaled = cpu_backend().mul(&n, &xn1).expect("pow bw: n*x^(n-1)");
        let g = cpu_backend()
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
    let out = cpu_backend()
        .pow_scalar(&src.tensor(), exponent)
        .map_err(|e| backend_err("pow_scalar", e))?;
    let mut out_var = Variable::new(out);
    if is_grad_enabled() && src.requires_grad {
        let node = std::sync::Arc::new(PowScalarBackward {
            saved_input: src.tensor().clone(),
            exponent,
            edges: [src.edge()],
        });
        out_var.grad_fn = Some(node);
        out_var.requires_grad = true;
    }
    Ok(out_var)
}
