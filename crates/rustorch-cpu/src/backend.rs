//! `Backend` trait — the dispatch surface for tensor ops
//! (RFC-0004, P1.2 task `Backend trait definition`).
//!
//! v1 ships a *minimal* trait surface (~12 methods) covering the ops
//! that P1.3 (elementary), P1.4 (linalg/conv/reductions) and P1.6 (nn)
//! actually call. The trait is `dyn`-safe (`&dyn Backend`) and
//! `Send + Sync`; the registry returns a `&'static dyn Backend`
//! singleton per device.
//!
//! Adding a new method to this trait is a non-breaking change in the
//! Rust sense as long as the method comes with a default impl that
//! returns `BackendError::UnsupportedOp` — that lets out-of-tree GPU
//! backends roll out incrementally.

use crate::error::BackendError;
use rustorch_core::tensor::tensor_impl::Tensor;

/// Single dispatch surface for tensor ops on a given device.
pub trait Backend: Send + Sync {
    /// Stable name of the device — `"cpu"`, `"wgpu"`, `"cuda"`, …
    fn name(&self) -> &'static str;

    /// Element-wise add: `out = lhs + rhs` (broadcast).
    fn add(&self, lhs: &Tensor, rhs: &Tensor) -> Result<Tensor, BackendError>;

    /// Element-wise sub: `out = lhs - rhs` (broadcast).
    fn sub(&self, lhs: &Tensor, rhs: &Tensor) -> Result<Tensor, BackendError>;

    /// Element-wise mul: `out = lhs * rhs` (broadcast).
    fn mul(&self, lhs: &Tensor, rhs: &Tensor) -> Result<Tensor, BackendError>;

    /// Element-wise div: `out = lhs / rhs` (broadcast).
    fn div(&self, lhs: &Tensor, rhs: &Tensor) -> Result<Tensor, BackendError>;

    /// Element-wise unary negation.
    fn neg(&self, src: &Tensor) -> Result<Tensor, BackendError>;

    /// Matrix multiplication on rank-2 tensors.
    fn matmul(&self, lhs: &Tensor, rhs: &Tensor) -> Result<Tensor, BackendError>;

    /// Matmul with optional transpose flags on the operands.
    /// Default impl runs `transpose + matmul`; backends with
    /// transpose-aware kernels (e.g. Metal `simdgroup_load` with
    /// `transpose` flag) override to skip the explicit transpose
    /// dispatch and avoid materialising the intermediate.
    fn matmul_with_transposes(
        &self,
        lhs: &Tensor,
        rhs: &Tensor,
        transpose_a: bool,
        transpose_b: bool,
    ) -> Result<Tensor, BackendError> {
        let lhs_eff = if transpose_a {
            self.transpose(lhs, 0, 1)?
        } else {
            lhs.clone()
        };
        let rhs_eff = if transpose_b {
            self.transpose(rhs, 0, 1)?
        } else {
            rhs.clone()
        };
        self.matmul(&lhs_eff, &rhs_eff)
    }

    /// Linear forward: `C = A @ B + bias_broadcast(N)`. Default impl
    /// composes `matmul + add_bias` (two dispatches + one ~M×N
    /// intermediate buffer). Backends that support a single fused
    /// kernel (e.g. Metal `matmul_simdgroup_f32_coarsened_wide_bias`)
    /// override to collapse this into a single dispatch / write-back.
    ///
    /// Shapes: `A:[M,K]`, `B:[K,N]`, `bias:[N]`, output `[M,N]`.
    fn matmul_with_bias(
        &self,
        lhs: &Tensor,
        rhs: &Tensor,
        bias: &Tensor,
    ) -> Result<Tensor, BackendError> {
        let mm = self.matmul(lhs, rhs)?;
        self.add_bias(&mm, bias)
    }

    /// Sum over all elements.
    fn sum(&self, _src: &Tensor) -> Result<Tensor, BackendError> {
        Err(BackendError::UnsupportedOp {
            op: "sum",
            device: self.name(),
        })
    }

    /// Mean over all elements.
    fn mean(&self, _src: &Tensor) -> Result<Tensor, BackendError> {
        Err(BackendError::UnsupportedOp {
            op: "mean",
            device: self.name(),
        })
    }

    /// ReLU activation.
    fn relu(&self, src: &Tensor) -> Result<Tensor, BackendError>;

    /// Equality (element-wise) → Bool tensor.
    fn eq(&self, lhs: &Tensor, rhs: &Tensor) -> Result<Tensor, BackendError>;

    // -------------------- P1.3 — extra elementary ops --------------------
    //
    // Each extra method has a default Unsupported impl. Backends opt in by
    // overriding; downstream ops degrade gracefully via `Result<_, BackendError>`.

    /// Element-wise absolute value.
    fn abs(&self, _src: &Tensor) -> Result<Tensor, BackendError> {
        Err(unsupported("abs", self.name()))
    }
    /// Element-wise square root.
    fn sqrt(&self, _src: &Tensor) -> Result<Tensor, BackendError> {
        Err(unsupported("sqrt", self.name()))
    }
    /// Element-wise exponential.
    fn exp(&self, _src: &Tensor) -> Result<Tensor, BackendError> {
        Err(unsupported("exp", self.name()))
    }
    /// Element-wise natural logarithm.
    fn log(&self, _src: &Tensor) -> Result<Tensor, BackendError> {
        Err(unsupported("log", self.name()))
    }
    /// Element-wise sine.
    fn sin(&self, _src: &Tensor) -> Result<Tensor, BackendError> {
        Err(unsupported("sin", self.name()))
    }
    /// Element-wise cosine.
    fn cos(&self, _src: &Tensor) -> Result<Tensor, BackendError> {
        Err(unsupported("cos", self.name()))
    }
    /// Element-wise tangent.
    fn tan(&self, _src: &Tensor) -> Result<Tensor, BackendError> {
        Err(unsupported("tan", self.name()))
    }
    /// Element-wise scalar pow: `out = src^exponent`.
    fn pow_scalar(&self, _src: &Tensor, _exponent: f64) -> Result<Tensor, BackendError> {
        Err(unsupported("pow_scalar", self.name()))
    }
    /// Element-wise tensor-tensor pow: `out = lhs^rhs` (broadcast).
    fn pow(&self, _lhs: &Tensor, _rhs: &Tensor) -> Result<Tensor, BackendError> {
        Err(unsupported("pow", self.name()))
    }
    /// 1 / sqrt(x).
    fn rsqrt(&self, _src: &Tensor) -> Result<Tensor, BackendError> {
        Err(unsupported("rsqrt", self.name()))
    }
    /// exp(x) - 1, accurate near 0.
    fn expm1(&self, _src: &Tensor) -> Result<Tensor, BackendError> {
        Err(unsupported("expm1", self.name()))
    }
    /// log(1 + x), accurate near 0.
    fn log1p(&self, _src: &Tensor) -> Result<Tensor, BackendError> {
        Err(unsupported("log1p", self.name()))
    }
    /// Base-2 logarithm.
    fn log2(&self, _src: &Tensor) -> Result<Tensor, BackendError> {
        Err(unsupported("log2", self.name()))
    }
    /// Base-10 logarithm.
    fn log10(&self, _src: &Tensor) -> Result<Tensor, BackendError> {
        Err(unsupported("log10", self.name()))
    }
    /// Element-wise asin (radians).
    fn asin(&self, _src: &Tensor) -> Result<Tensor, BackendError> {
        Err(unsupported("asin", self.name()))
    }
    /// Element-wise acos (radians).
    fn acos(&self, _src: &Tensor) -> Result<Tensor, BackendError> {
        Err(unsupported("acos", self.name()))
    }
    /// Element-wise atan (radians).
    fn atan(&self, _src: &Tensor) -> Result<Tensor, BackendError> {
        Err(unsupported("atan", self.name()))
    }
    /// 2-argument atan: `atan2(y, x)`.
    fn atan2(&self, _y: &Tensor, _x: &Tensor) -> Result<Tensor, BackendError> {
        Err(unsupported("atan2", self.name()))
    }
    /// Hyperbolic sine.
    fn sinh(&self, _src: &Tensor) -> Result<Tensor, BackendError> {
        Err(unsupported("sinh", self.name()))
    }
    /// Hyperbolic cosine.
    fn cosh(&self, _src: &Tensor) -> Result<Tensor, BackendError> {
        Err(unsupported("cosh", self.name()))
    }
    /// ELU: `x if x > 0 else alpha * (exp(x) - 1)`.
    fn elu(&self, _src: &Tensor, _alpha: f64) -> Result<Tensor, BackendError> {
        Err(unsupported("elu", self.name()))
    }
    /// Softplus: `(1/beta) * log(1 + exp(beta * x))`, numerically stable.
    fn softplus(&self, _src: &Tensor, _beta: f64) -> Result<Tensor, BackendError> {
        Err(unsupported("softplus", self.name()))
    }
    /// HardSwish: `x * relu6(x + 3) / 6`.
    fn hardswish(&self, _src: &Tensor) -> Result<Tensor, BackendError> {
        Err(unsupported("hardswish", self.name()))
    }
    /// HardTanh: clamp(x, min, max).
    fn hardtanh(&self, _src: &Tensor, _min: f64, _max: f64) -> Result<Tensor, BackendError> {
        Err(unsupported("hardtanh", self.name()))
    }
    /// HardSigmoid: `clamp((x + 3) / 6, 0, 1)`.
    fn hardsigmoid(&self, _src: &Tensor) -> Result<Tensor, BackendError> {
        Err(unsupported("hardsigmoid", self.name()))
    }
    /// Sigmoid: `1 / (1 + exp(-x))`.
    fn sigmoid(&self, _src: &Tensor) -> Result<Tensor, BackendError> {
        Err(unsupported("sigmoid", self.name()))
    }
    /// Tanh.
    fn tanh(&self, _src: &Tensor) -> Result<Tensor, BackendError> {
        Err(unsupported("tanh", self.name()))
    }
    /// GELU (tanh approximation, matches `torch.nn.functional.gelu(approximate='tanh')`).
    fn gelu(&self, _src: &Tensor) -> Result<Tensor, BackendError> {
        Err(unsupported("gelu", self.name()))
    }
    /// GELU (exact, using erf — matches `torch.nn.functional.gelu(approximate='none')`).
    /// Uses the Abramowitz-Stegun erf approximation (max abs error ≤ 1.5e-7).
    fn gelu_exact(&self, _src: &Tensor) -> Result<Tensor, BackendError> {
        Err(unsupported("gelu_exact", self.name()))
    }
    /// LeakyReLU: x if x > 0 else slope*x.
    fn leaky_relu(&self, _src: &Tensor, _slope: f64) -> Result<Tensor, BackendError> {
        Err(unsupported("leaky_relu", self.name()))
    }
    /// SiLU / Swish: x * sigmoid(x).
    fn silu(&self, _src: &Tensor) -> Result<Tensor, BackendError> {
        Err(unsupported("silu", self.name()))
    }

    /// Element-wise not-equal.
    fn ne(&self, _lhs: &Tensor, _rhs: &Tensor) -> Result<Tensor, BackendError> {
        Err(unsupported("ne", self.name()))
    }
    /// Element-wise less-than.
    fn lt(&self, _lhs: &Tensor, _rhs: &Tensor) -> Result<Tensor, BackendError> {
        Err(unsupported("lt", self.name()))
    }
    /// Element-wise less-than-or-equal.
    fn le(&self, _lhs: &Tensor, _rhs: &Tensor) -> Result<Tensor, BackendError> {
        Err(unsupported("le", self.name()))
    }
    /// Element-wise greater-than.
    fn gt(&self, _lhs: &Tensor, _rhs: &Tensor) -> Result<Tensor, BackendError> {
        Err(unsupported("gt", self.name()))
    }
    /// Element-wise greater-than-or-equal.
    fn ge(&self, _lhs: &Tensor, _rhs: &Tensor) -> Result<Tensor, BackendError> {
        Err(unsupported("ge", self.name()))
    }
    /// Bool tensor: which lanes are NaN?
    fn isnan(&self, _src: &Tensor) -> Result<Tensor, BackendError> {
        Err(unsupported("isnan", self.name()))
    }
    /// Bool tensor: which lanes are ±Inf?
    fn isinf(&self, _src: &Tensor) -> Result<Tensor, BackendError> {
        Err(unsupported("isinf", self.name()))
    }
    /// Bool tensor: which lanes are finite (not NaN/Inf)?
    fn isfinite(&self, _src: &Tensor) -> Result<Tensor, BackendError> {
        Err(unsupported("isfinite", self.name()))
    }

    // -------------------- P1.3 — Indexing ops --------------------

    /// Gather: `out[i_0, …, i_{D-1}] = src[i_0, …, idx[i_0,…,i_{D-1}], …, i_{D-1}]`
    /// along `dim`. `idx` must be I64 with the same rank as `src`; every
    /// index must be in `[0, src.shape[dim])`.
    fn gather(&self, _src: &Tensor, _dim: usize, _idx: &Tensor) -> Result<Tensor, BackendError> {
        Err(unsupported("gather", self.name()))
    }
    /// Scatter: `out[i_0, …, idx[…], …] = src[…]` along `dim`. The
    /// returned tensor is `dst`'s shape; values not addressed by idx
    /// are copied from `dst`.
    fn scatter(
        &self,
        _dst: &Tensor,
        _dim: usize,
        _idx: &Tensor,
        _src: &Tensor,
    ) -> Result<Tensor, BackendError> {
        Err(unsupported("scatter", self.name()))
    }
    /// Scatter-add: same as scatter but accumulates instead of
    /// overwriting (atomic on parallel paths).
    fn scatter_add(
        &self,
        _dst: &Tensor,
        _dim: usize,
        _idx: &Tensor,
        _src: &Tensor,
    ) -> Result<Tensor, BackendError> {
        Err(unsupported("scatter_add", self.name()))
    }
    /// `index_select(src, dim, indices)` — pick the slices of `src`
    /// along `dim` matching the (rank-1) `indices` tensor.
    fn index_select(
        &self,
        _src: &Tensor,
        _dim: usize,
        _indices: &Tensor,
    ) -> Result<Tensor, BackendError> {
        Err(unsupported("index_select", self.name()))
    }
    /// `masked_select(src, mask)` — flatten the source into a 1D
    /// tensor containing every element where `mask` is true.
    fn masked_select(&self, _src: &Tensor, _mask: &Tensor) -> Result<Tensor, BackendError> {
        Err(unsupported("masked_select", self.name()))
    }
    /// `masked_fill(src, mask, value)` — return a copy of `src` with
    /// `value` written wherever `mask` is true.
    fn masked_fill(
        &self,
        _src: &Tensor,
        _mask: &Tensor,
        _value: f64,
    ) -> Result<Tensor, BackendError> {
        Err(unsupported("masked_fill", self.name()))
    }
    /// `where(cond, x, y)` — element-wise select: `cond ? x : y`.
    fn r#where(&self, _cond: &Tensor, _x: &Tensor, _y: &Tensor) -> Result<Tensor, BackendError> {
        Err(unsupported("where", self.name()))
    }
    /// Indices of nonzero elements (rank-2 I64 tensor of shape
    /// `[N, ndim]` where `N` is the count of nonzero entries).
    fn nonzero(&self, _src: &Tensor) -> Result<Tensor, BackendError> {
        Err(unsupported("nonzero", self.name()))
    }

    // -------------------- P1.4 — Reductions with dim --------------------

    /// Reduce-sum along `dims`. If `keepdim` is true, every reduced axis
    /// is kept as size 1 (else removed).
    fn sum_dim(
        &self,
        _src: &Tensor,
        _dims: &[usize],
        _keepdim: bool,
    ) -> Result<Tensor, BackendError> {
        Err(unsupported("sum_dim", self.name()))
    }
    /// Reduce-mean along `dims`.
    fn mean_dim(
        &self,
        _src: &Tensor,
        _dims: &[usize],
        _keepdim: bool,
    ) -> Result<Tensor, BackendError> {
        Err(unsupported("mean_dim", self.name()))
    }
    /// Reduce-max along `dims` (single axis at a time for argmax).
    fn max_dim(&self, _src: &Tensor, _dim: usize, _keepdim: bool) -> Result<Tensor, BackendError> {
        Err(unsupported("max_dim", self.name()))
    }
    /// Reduce-min along `dim`.
    fn min_dim(&self, _src: &Tensor, _dim: usize, _keepdim: bool) -> Result<Tensor, BackendError> {
        Err(unsupported("min_dim", self.name()))
    }
    /// Argmax — index (I64) of the maximum element along `dim`.
    fn argmax(&self, _src: &Tensor, _dim: usize, _keepdim: bool) -> Result<Tensor, BackendError> {
        Err(unsupported("argmax", self.name()))
    }
    /// Argmin — index (I64) of the minimum element along `dim`.
    fn argmin(&self, _src: &Tensor, _dim: usize, _keepdim: bool) -> Result<Tensor, BackendError> {
        Err(unsupported("argmin", self.name()))
    }
    /// Variance along `dim` (Welford one-pass).
    fn var_dim(
        &self,
        _src: &Tensor,
        _dim: usize,
        _unbiased: bool,
        _keepdim: bool,
    ) -> Result<Tensor, BackendError> {
        Err(unsupported("var_dim", self.name()))
    }
    /// All-reduce ops (full-tensor) — full sum / mean already exist; here
    /// we add prod, full max, full min, all/any over Bool, cumsum, cumprod.
    fn prod(&self, _src: &Tensor) -> Result<Tensor, BackendError> {
        Err(unsupported("prod", self.name()))
    }
    /// Full-tensor max → scalar.
    fn max(&self, _src: &Tensor) -> Result<Tensor, BackendError> {
        Err(unsupported("max", self.name()))
    }
    /// Full-tensor min → scalar.
    fn min(&self, _src: &Tensor) -> Result<Tensor, BackendError> {
        Err(unsupported("min", self.name()))
    }
    /// True iff every element is true (Bool tensor).
    fn all(&self, _src: &Tensor) -> Result<Tensor, BackendError> {
        Err(unsupported("all", self.name()))
    }
    /// True iff any element is true.
    fn any(&self, _src: &Tensor) -> Result<Tensor, BackendError> {
        Err(unsupported("any", self.name()))
    }
    /// Cumulative sum along `dim`.
    fn cumsum(&self, _src: &Tensor, _dim: usize) -> Result<Tensor, BackendError> {
        Err(unsupported("cumsum", self.name()))
    }
    /// Cumulative product along `dim`.
    fn cumprod(&self, _src: &Tensor, _dim: usize) -> Result<Tensor, BackendError> {
        Err(unsupported("cumprod", self.name()))
    }

    // -------------------- P1.4 — Softmax / log_softmax --------------------

    /// Numerically stable softmax along `dim` (uses max-subtraction + log-sum-exp).
    fn softmax(&self, _src: &Tensor, _dim: usize) -> Result<Tensor, BackendError> {
        Err(unsupported("softmax", self.name()))
    }
    /// Numerically stable log_softmax along `dim`.
    fn log_softmax(&self, _src: &Tensor, _dim: usize) -> Result<Tensor, BackendError> {
        Err(unsupported("log_softmax", self.name()))
    }

    // -------------------- P1.4 — Loss functions --------------------

    /// Mean Squared Error: `((x - y)^2).reduce(reduction)`.
    fn mse_loss(
        &self,
        _input: &Tensor,
        _target: &Tensor,
        _reduction: Reduction,
    ) -> Result<Tensor, BackendError> {
        Err(unsupported("mse_loss", self.name()))
    }
    /// Cross-entropy = log_softmax(input) → NLL with class targets (I64).
    fn cross_entropy(
        &self,
        _input: &Tensor,
        _target: &Tensor,
        _reduction: Reduction,
    ) -> Result<Tensor, BackendError> {
        Err(unsupported("cross_entropy", self.name()))
    }
    /// Negative Log Likelihood loss against pre-computed log-probabilities.
    fn nll_loss(
        &self,
        _log_probs: &Tensor,
        _target: &Tensor,
        _reduction: Reduction,
    ) -> Result<Tensor, BackendError> {
        Err(unsupported("nll_loss", self.name()))
    }
    /// Binary cross-entropy with logits (numerically stable).
    fn bce_with_logits(
        &self,
        _input: &Tensor,
        _target: &Tensor,
        _reduction: Reduction,
    ) -> Result<Tensor, BackendError> {
        Err(unsupported("bce_with_logits", self.name()))
    }

    // -------------------- P1.3 — Shape ops (kernel-side) --------------------

    /// Concatenate `tensors` along `dim`. All tensors must have the
    /// same dtype and the same shape except along `dim`.
    fn cat(&self, _tensors: &[&Tensor], _dim: usize) -> Result<Tensor, BackendError> {
        Err(unsupported("cat", self.name()))
    }
    /// Stack `tensors` along a new axis at position `dim`. Inputs must
    /// all have identical shape and dtype.
    fn stack(&self, _tensors: &[&Tensor], _dim: usize) -> Result<Tensor, BackendError> {
        Err(unsupported("stack", self.name()))
    }
    /// Split a tensor into chunks of `split_size` along `dim`. Last
    /// chunk may be smaller if dim size is not divisible.
    fn split(
        &self,
        _src: &Tensor,
        _split_size: usize,
        _dim: usize,
    ) -> Result<Vec<Tensor>, BackendError> {
        Err(unsupported("split", self.name()))
    }
    /// Split into roughly equal `n_chunks` along `dim`.
    fn chunk(
        &self,
        _src: &Tensor,
        _n_chunks: usize,
        _dim: usize,
    ) -> Result<Vec<Tensor>, BackendError> {
        Err(unsupported("chunk", self.name()))
    }
    /// Repeat tensor along given `repeats` count per axis.
    fn repeat(&self, _src: &Tensor, _repeats: &[usize]) -> Result<Tensor, BackendError> {
        Err(unsupported("repeat", self.name()))
    }
    /// Reverse the tensor along the given `dims`.
    fn flip(&self, _src: &Tensor, _dims: &[usize]) -> Result<Tensor, BackendError> {
        Err(unsupported("flip", self.name()))
    }
    /// Circular shift along `dim` by `shifts` positions (positive shifts
    /// values toward higher indices; negative shifts wrap around).
    fn roll(&self, _src: &Tensor, _shifts: i64, _dim: usize) -> Result<Tensor, BackendError> {
        Err(unsupported("roll", self.name()))
    }

    /// Convert a tensor from its current dtype to `target` dtype.
    ///
    /// **Saturating** float→int conversions (NaN → 0, +Inf → MAX,
    /// -Inf → MIN), matching `torch.Tensor.to(dtype)` semantics. Bool
    /// conversion follows: any nonzero → true; true → 1; false → 0.
    fn cast(
        &self,
        _src: &Tensor,
        _target: rustorch_core::tensor::dtype::Dtype,
    ) -> Result<Tensor, BackendError> {
        Err(unsupported("cast", self.name()))
    }

    // -------------------- A3 — autograd dispatch plumbing (P3.Y) --------------------
    //
    // Methods added to support the wgpu autograd wiring (RFC P3.Y / Phase A3).
    // All have default `Unsupported` impls so existing backends are unaffected;
    // CpuBackend overrides them, future WgpuBackend overrides them progressively.

    /// Batched matmul: `[B, M, K] @ [B, K, N] = [B, M, N]`.
    ///
    /// Used by `MultiHeadAttention` and any rank-3 matmul flow. The CPU
    /// implementation is a simple triple-loop per batch; production GPU
    /// kernels can fuse the batch dim into a single dispatch.
    fn bmm(&self, _lhs: &Tensor, _rhs: &Tensor) -> Result<Tensor, BackendError> {
        Err(unsupported("bmm", self.name()))
    }

    /// Swap two axes of `src` (rank-N), returning a contiguous tensor.
    ///
    /// Equivalent to `Tensor::transpose(d0, d1).contiguous()` but exposed
    /// on the Backend trait so device-specific backends can use a fused
    /// permute kernel rather than a view-then-copy.
    fn transpose(&self, _src: &Tensor, _d0: usize, _d1: usize) -> Result<Tensor, BackendError> {
        Err(unsupported("transpose", self.name()))
    }

    /// Reshape `src` to `shape`. Requires `numel(src) == numel(shape)` and
    /// a contiguous source. CPU impl materialises a fresh `from_vec`; GPU
    /// impls may share storage when alignment allows.
    fn reshape(&self, _src: &Tensor, _shape: &[usize]) -> Result<Tensor, BackendError> {
        Err(unsupported("reshape", self.name()))
    }

    /// `x + bias` where `bias.shape == x.shape[1..]` (broadcast across the
    /// leading batch dim). Equivalent to `add(x, bias.broadcast_to(x.shape))`
    /// but exposed as its own method so backends can fuse the broadcast +
    /// add into a single kernel.
    fn add_bias(&self, _x: &Tensor, _bias: &Tensor) -> Result<Tensor, BackendError> {
        Err(unsupported("add_bias", self.name()))
    }

    // ---- A3 backward helpers ----

    /// Sum `grad` across axes that were broadcasted up to produce its
    /// current shape, returning a tensor of `target_shape`.
    ///
    /// Used by every binary op backward (add/sub/mul/div) to undo
    /// broadcasting before passing the gradient to the corresponding
    /// input edge. Concretely: pads `target_shape` on the left with 1s
    /// to match `grad.ndim()`, then sums along every axis where the
    /// padded target is 1 but `grad.shape` is > 1, then squeezes the
    /// leading padded dims.
    fn unbroadcast_to(
        &self,
        _grad: &Tensor,
        _target_shape: &[usize],
    ) -> Result<Tensor, BackendError> {
        Err(unsupported("unbroadcast_to", self.name()))
    }

    /// Backward of `softmax(input, dim) = output`:
    /// `d_input = output * (grad - sum(grad * output, dim, keepdim=true))`.
    ///
    /// Composable from `mul + sum_dim + sub`, but exposed as its own
    /// method so backends can fuse the three reductions into a single
    /// pass — significant on GPU where each launch has fixed overhead.
    fn softmax_grad(
        &self,
        _grad: &Tensor,
        _output: &Tensor,
        _dim: usize,
    ) -> Result<Tensor, BackendError> {
        Err(unsupported("softmax_grad", self.name()))
    }
}

#[allow(unused)]
fn unsupported(op: &'static str, device: &'static str) -> BackendError {
    BackendError::UnsupportedOp { op, device }
}

/// Reduction mode for loss functions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Reduction {
    /// Reduce by mean.
    Mean,
    /// Reduce by sum.
    Sum,
    /// No reduction — keep per-sample values.
    None,
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::sync::atomic::AtomicUsize;

    /// A fixed-output stub that's never asked to compute anything beyond
    /// `name`. Used to verify the trait is dyn-safe and Send + Sync.
    struct StubBackend;

    impl Backend for StubBackend {
        fn name(&self) -> &'static str {
            "stub"
        }
        fn add(&self, _: &Tensor, _: &Tensor) -> Result<Tensor, BackendError> {
            Ok(Tensor::scalar(0.0))
        }
        fn sub(&self, _: &Tensor, _: &Tensor) -> Result<Tensor, BackendError> {
            Ok(Tensor::scalar(0.0))
        }
        fn mul(&self, _: &Tensor, _: &Tensor) -> Result<Tensor, BackendError> {
            Ok(Tensor::scalar(0.0))
        }
        fn div(&self, _: &Tensor, _: &Tensor) -> Result<Tensor, BackendError> {
            Ok(Tensor::scalar(0.0))
        }
        fn neg(&self, _: &Tensor) -> Result<Tensor, BackendError> {
            Ok(Tensor::scalar(0.0))
        }
        fn matmul(&self, _: &Tensor, _: &Tensor) -> Result<Tensor, BackendError> {
            Ok(Tensor::scalar(0.0))
        }
        fn relu(&self, _: &Tensor) -> Result<Tensor, BackendError> {
            Ok(Tensor::scalar(0.0))
        }
        fn eq(&self, _: &Tensor, _: &Tensor) -> Result<Tensor, BackendError> {
            Ok(Tensor::scalar(0.0))
        }
    }

    #[test]
    fn dyn_backend_is_object_safe() {
        let s: &dyn Backend = &StubBackend;
        assert_eq!(s.name(), "stub");
    }

    #[test]
    fn default_sum_returns_unsupported() {
        let s: &dyn Backend = &StubBackend;
        let t = Tensor::scalar(1.0);
        let err = s.sum(&t).unwrap_err();
        assert!(matches!(
            err,
            BackendError::UnsupportedOp {
                op: "sum",
                device: "stub"
            }
        ));
    }

    #[test]
    fn default_mean_returns_unsupported() {
        let s: &dyn Backend = &StubBackend;
        let t = Tensor::scalar(1.0);
        let err = s.mean(&t).unwrap_err();
        assert!(matches!(
            err,
            BackendError::UnsupportedOp {
                op: "mean",
                device: "stub"
            }
        ));
    }

    #[test]
    fn backend_send_sync_compiles() {
        // Compile-time check that any &'static dyn Backend can cross threads.
        fn assert_send<T: Send>() {}
        fn assert_sync<T: Sync>() {}
        assert_send::<&'static dyn Backend>();
        assert_sync::<&'static dyn Backend>();
    }

    #[test]
    fn shared_arc_dyn_backend_works() {
        // We expose backends via Arc<dyn Backend> in the registry — assert
        // that pattern compiles + behaves.
        use std::sync::Arc;
        let b: Arc<dyn Backend> = Arc::new(StubBackend);
        assert_eq!(b.name(), "stub");
        let _counter = AtomicUsize::new(0);
    }
}
