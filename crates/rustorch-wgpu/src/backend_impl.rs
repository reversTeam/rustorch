//! `impl Backend for WgpuBackend` — surface the wgpu kernels through the
//! `rustorch_cpu::backend::Backend` trait so the autograd dispatcher can
//! pick this backend at op time via `Variable::device()`.
//!
//! Design — round-trip Tensor↔WgpuStorage internally:
//!
//! ```text
//!   &Tensor (CPU shadow)
//!     ↓ to_gpu (host → device)
//!   WgpuStorage
//!     ↓ kernel (matmul / dispatch_binary / …)
//!   WgpuStorage
//!     ↓ to_cpu (device → host)
//!   Tensor (CPU shadow)
//! ```
//!
//! The round-trip cost is acceptable for v1 because the higher-level
//! autograd path (P3.Y plan, Phase B+) caches the canonical `WgpuStorage`
//! at the `Variable` level and bypasses this Tensor-shaped surface for
//! GPU-resident params. This impl is the **fallback path** that makes
//! `wgpu_backend()` a fully-functional `&dyn Backend` on its own.
//!
//! Coverage as of A4 (P3.Y plan, Phase A4):
//! - **Native kernels**: `add`, `sub`, `mul`, `div`, `neg`, `relu`, `sigmoid`,
//!   `tanh`, `silu`, `matmul` (rank-2).
//! - **CPU fallback** (no kernel yet): `eq` (required by trait surface).
//! - **Default Unsupported** (inherited from trait): everything else;
//!   future phases override progressively.
//!
//! The fallback path uses `to_cpu` + `cpu_backend()` + `to_gpu` so the
//! caller does not need to be aware of which ops are kernel-backed.

use crate::backend::WgpuBackend;
use crate::broadcast::dispatch_binary_broadcast;
use crate::elementwise::{dispatch_binary, dispatch_unary};
use crate::matmul::matmul;
use crate::reduce::{reduce_rows, ReduceKind};
use crate::storage::WgpuStorage;
use crate::transfer::to_gpu;
use rustorch_core::tensor::device::Device;
use rustorch_core::tensor::tensor_impl::Tensor;
use rustorch_cpu::backend::Backend;
use rustorch_cpu::cpu_backend::cpu_backend;
use rustorch_cpu::error::BackendError;

/// Tag a Tensor as living on the Wgpu device.
///
/// With Storage Option A (P3.Z), Tensors built via
/// `Tensor::from_wgpu_storage(...)` are already device-tagged
/// `Device::Wgpu` and carry `Storage::Wgpu(handle)` natively (no
/// host shadow). This helper remains for the rare CPU-fallback paths
/// (e.g. `eq` which has no native kernel yet) — those still produce
/// CPU-storage tensors that must be retagged so subsequent ops keep
/// routing through `wgpu_backend()`.
fn tag_wgpu(t: Tensor) -> Tensor {
    t.with_device(Device::Wgpu)
}

/// Wrap a kernel-side `WgpuStorage` in a Tensor with `Storage::Wgpu`
/// natively — no host round trip. P3.Z Task A round-trip elimination.
///
/// The Tensor's storage holds the same `core::WgpuStorage` Arc clone
/// that the kernel produced, so a follow-on op invoking `to_gpu` on
/// this Tensor takes the fast path and reuses the buffer.
fn finish_wgpu_op(out: WgpuStorage, shape: Vec<usize>) -> Tensor {
    let dtype = out.dtype;
    let core = out.buffer; // CoreWgpuStorage with Drop hook intact
    Tensor::from_wgpu_storage(core, shape, dtype)
}

/// Map `WgpuError` → `BackendError::NumericalError(...)`. The trait expects
/// the latter (string-based) so device-specific errors are surfaced
/// without leaking wgpu types into the trait surface.
fn wgpu_err(op: &'static str, err: crate::error::WgpuError) -> BackendError {
    BackendError::NumericalError(format!("{op}: wgpu: {err}"))
}

/// **No-round-trip** helper: upload (or reuse) a Tensor's GPU buffer,
/// run a unary kernel, wrap the kernel output as a fresh Tensor with
/// `Storage::Wgpu(...)` natively. P3.Z Task A round-trip elimination.
///
/// Used by the elementary unary ops (relu/sigmoid/tanh/silu/neg) which
/// share the dispatch_unary signature.
fn unary_op(
    backend: &WgpuBackend,
    op_name: &'static str,
    src: &Tensor,
) -> Result<Tensor, BackendError> {
    let inp = to_gpu(backend, src).map_err(|e| wgpu_err(op_name, e))?;
    let out = dispatch_unary(backend, op_name, &inp).map_err(|e| wgpu_err(op_name, e))?;
    Ok(finish_wgpu_op(out, src.shape().to_vec()))
}

/// **No-round-trip** helper: upload (or reuse) two Tensors' GPU
/// buffers, run a binary kernel, wrap the kernel output as a fresh
/// Tensor with `Storage::Wgpu(...)` natively. P3.Z Task A round-trip
/// elimination.
///
/// Uses `dispatch_binary_broadcast` so element-wise broadcasting (matching
/// the CPU semantics in `dispatch_binary` of `cpu_backend.rs`) works
/// out of the box.
fn binary_op(
    backend: &WgpuBackend,
    op_name: &'static str,
    lhs: &Tensor,
    rhs: &Tensor,
) -> Result<Tensor, BackendError> {
    let lhs_g = to_gpu(backend, lhs).map_err(|e| wgpu_err(op_name, e))?;
    let rhs_g = to_gpu(backend, rhs).map_err(|e| wgpu_err(op_name, e))?;
    // Use the broadcast variant so e.g. `add([B,N], [N])` works the same
    // way it does on the CPU backend.
    let (out, out_shape) = if lhs.shape() == rhs.shape() {
        // Fast path: identical shapes, no broadcasting work needed.
        let s =
            dispatch_binary(backend, op_name, &lhs_g, &rhs_g).map_err(|e| wgpu_err(op_name, e))?;
        (s, lhs.shape().to_vec())
    } else {
        // dispatch_binary_broadcast returns (storage, out_shape).
        dispatch_binary_broadcast(backend, op_name, &lhs_g, lhs.shape(), &rhs_g, rhs.shape())
            .map_err(|e| wgpu_err(op_name, e))?
    };
    drop((lhs_g, rhs_g)); // explicit drop after kernel done
    Ok(finish_wgpu_op(out, out_shape))
}

/// Materialise a Tensor to host (CPU storage) so `cpu_backend()` ops
/// can read its bytes via `.as_slice::<f32>()`. P3.Z Task A:
/// `Storage::Wgpu` returns an empty slice from `as_slice` so any
/// CPU-fallback path needs an explicit download first.
///
/// CPU-storage Tensors pass through unchanged (cheap clone).
fn host(t: &Tensor) -> Result<Tensor, BackendError> {
    crate::transfer::tensor_to_cpu(t).map_err(|e| wgpu_err("cpu_fallback", e))
}

/// Native GPU fast path for `sum_dim` / `mean_dim` on 2D inputs with
/// a single reduction axis. Returns `None` if the shape/axis pattern
/// is not yet handled (3D+ tensors, multi-axis reductions) so callers
/// can fall back to the CPU path.
///
/// Composition map:
/// - input `[d0, d1]`, axis = 1 (last) → `reduce_rows(d0, d1)` → output `[d0]`
/// - input `[d0, d1]`, axis = 0        → `transpose2d → reduce_rows(d1, d0)` → output `[d1]`
///
/// `keepdim = true` keeps the reduced axis as size-1 in the result
/// shape (`[d0, 1]` / `[1, d1]`); `keepdim = false` drops it.
fn sum_or_mean_dim_2d(
    backend: &WgpuBackend,
    src: &Tensor,
    dims: &[usize],
    keepdim: bool,
    kind: ReduceKind,
) -> Result<Option<Tensor>, BackendError> {
    let shape = src.shape();
    if shape.len() != 2 || dims.len() != 1 {
        return Ok(None);
    }
    let (d0, d1) = (shape[0], shape[1]);
    let axis = dims[0];
    let op_name = match kind {
        ReduceKind::Sum => "sum_dim",
        ReduceKind::Mean => "mean_dim",
        _ => return Ok(None),
    };
    let inp = to_gpu(backend, src).map_err(|e| wgpu_err(op_name, e))?;
    let (reduced, out_shape) = match axis {
        1 => {
            // Reduce along last axis: [d0, d1] → [d0].
            let r = reduce_rows(backend, &inp, d0, d1, kind).map_err(|e| wgpu_err(op_name, e))?;
            let out_shape = if keepdim { vec![d0, 1] } else { vec![d0] };
            (r, out_shape)
        },
        0 => {
            // Reduce along first axis: transpose [d0, d1] → [d1, d0],
            // then reduce_rows → [d1].
            let xt = crate::transpose::transpose2d(backend, &inp, d0, d1)
                .map_err(|e| wgpu_err(op_name, e))?;
            let r = reduce_rows(backend, &xt, d1, d0, kind).map_err(|e| wgpu_err(op_name, e))?;
            let out_shape = if keepdim { vec![1, d1] } else { vec![d1] };
            (r, out_shape)
        },
        _ => {
            return Err(BackendError::ShapeMismatch {
                op: op_name,
                lhs: shape.to_vec(),
                rhs: dims.to_vec(),
            });
        },
    };
    Ok(Some(finish_wgpu_op(reduced, out_shape)))
}

/// CPU fallback: download both operands, run on `cpu_backend()`, return.
///
/// Used by methods like `eq` for which the wgpu crate ships no kernel
/// yet. Behaves like a slow but correct stub so the trait surface is
/// fully populated.
fn cpu_fallback_binary<F>(
    op_name: &'static str,
    lhs: &Tensor,
    rhs: &Tensor,
    f: F,
) -> Result<Tensor, BackendError>
where
    F: Fn(&dyn Backend, &Tensor, &Tensor) -> Result<Tensor, BackendError>,
{
    let _ = op_name;
    let l = host(lhs)?;
    let r = host(rhs)?;
    f(cpu_backend(), &l, &r)
}

impl Backend for WgpuBackend {
    fn name(&self) -> &'static str {
        "wgpu"
    }

    // -------------------- Required methods --------------------

    fn add(&self, lhs: &Tensor, rhs: &Tensor) -> Result<Tensor, BackendError> {
        binary_op(self, "add", lhs, rhs)
    }

    fn sub(&self, lhs: &Tensor, rhs: &Tensor) -> Result<Tensor, BackendError> {
        binary_op(self, "sub", lhs, rhs)
    }

    fn mul(&self, lhs: &Tensor, rhs: &Tensor) -> Result<Tensor, BackendError> {
        binary_op(self, "mul", lhs, rhs)
    }

    fn div(&self, lhs: &Tensor, rhs: &Tensor) -> Result<Tensor, BackendError> {
        binary_op(self, "div", lhs, rhs)
    }

    fn neg(&self, src: &Tensor) -> Result<Tensor, BackendError> {
        unary_op(self, "neg", src)
    }

    fn matmul(&self, lhs: &Tensor, rhs: &Tensor) -> Result<Tensor, BackendError> {
        let l_shape = lhs.shape();
        let r_shape = rhs.shape();
        if l_shape.len() != 2 || r_shape.len() != 2 {
            return Err(BackendError::ShapeMismatch {
                op: "matmul",
                lhs: l_shape.to_vec(),
                rhs: r_shape.to_vec(),
            });
        }
        let (m, k1) = (l_shape[0], l_shape[1]);
        let (k2, n) = (r_shape[0], r_shape[1]);
        if k1 != k2 {
            return Err(BackendError::ShapeMismatch {
                op: "matmul",
                lhs: l_shape.to_vec(),
                rhs: r_shape.to_vec(),
            });
        }
        let lhs_g: WgpuStorage = to_gpu(self, lhs).map_err(|e| wgpu_err("matmul", e))?;
        let rhs_g: WgpuStorage = to_gpu(self, rhs).map_err(|e| wgpu_err("matmul", e))?;
        let out = matmul(self, &lhs_g, &rhs_g, m, k1, n).map_err(|e| wgpu_err("matmul", e))?;
        Ok(finish_wgpu_op(out, vec![m, n]))
    }

    fn relu(&self, src: &Tensor) -> Result<Tensor, BackendError> {
        unary_op(self, "relu", src)
    }

    fn eq(&self, lhs: &Tensor, rhs: &Tensor) -> Result<Tensor, BackendError> {
        // No wgpu kernel for comparison ops yet — fall through to CPU.
        // Acceptable since `eq` is rare in the autograd hot path; future
        // wgpu kernels can override this with a native dispatch.
        cpu_fallback_binary("eq", lhs, rhs, |b, l, r| b.eq(l, r))
    }

    // -------------------- Optional methods with native wgpu kernels --------------------

    fn sigmoid(&self, src: &Tensor) -> Result<Tensor, BackendError> {
        unary_op(self, "sigmoid", src)
    }

    fn tanh(&self, src: &Tensor) -> Result<Tensor, BackendError> {
        unary_op(self, "tanh", src)
    }

    fn silu(&self, src: &Tensor) -> Result<Tensor, BackendError> {
        unary_op(self, "silu", src)
    }

    /// Full-tensor sum reduction → scalar `[1]`. Composes via
    /// `reduce_rows(b=1, k=numel)` after a no-cost reshape.
    fn sum(&self, src: &Tensor) -> Result<Tensor, BackendError> {
        let numel = src.numel();
        if numel == 0 {
            return Err(BackendError::ShapeMismatch {
                op: "sum",
                lhs: src.shape().to_vec(),
                rhs: vec![],
            });
        }
        let inp = to_gpu(self, src).map_err(|e| wgpu_err("sum", e))?;
        let out =
            reduce_rows(self, &inp, 1, numel, ReduceKind::Sum).map_err(|e| wgpu_err("sum", e))?;
        Ok(finish_wgpu_op(out, vec![1]))
    }

    /// Full-tensor mean reduction → scalar `[1]`. Composes via
    /// `reduce_rows(ReduceKind::Mean)`.
    fn mean(&self, src: &Tensor) -> Result<Tensor, BackendError> {
        let numel = src.numel();
        if numel == 0 {
            return Err(BackendError::ShapeMismatch {
                op: "mean",
                lhs: src.shape().to_vec(),
                rhs: vec![],
            });
        }
        let inp = to_gpu(self, src).map_err(|e| wgpu_err("mean", e))?;
        let out =
            reduce_rows(self, &inp, 1, numel, ReduceKind::Mean).map_err(|e| wgpu_err("mean", e))?;
        Ok(finish_wgpu_op(out, vec![1]))
    }

    // -------------------- Composed methods (no native wgpu kernel) --------------------
    //
    // These methods don't have a dedicated WGSL kernel yet; they're
    // implemented by composing existing primitives. Progressive override
    // as Phase B+ writes more kernels.

    /// `x + bias` where `bias.shape == x.shape[1..]` (broadcast across the
    /// batch axis). Composes via `dispatch_binary_broadcast("add")` on the
    /// host-uploaded tensors so a Linear forward `xw + b` runs on the GPU
    /// without a CPU round-trip in the middle.
    fn add_bias(&self, x: &Tensor, bias: &Tensor) -> Result<Tensor, BackendError> {
        // Validate shapes the same way CpuBackend does.
        if x.ndim() != 2 || bias.ndim() != 1 {
            return Err(BackendError::ShapeMismatch {
                op: "add_bias",
                lhs: x.shape().to_vec(),
                rhs: bias.shape().to_vec(),
            });
        }
        let n_out = x.shape()[1];
        if bias.shape() != [n_out] {
            return Err(BackendError::ShapeMismatch {
                op: "add_bias",
                lhs: x.shape().to_vec(),
                rhs: bias.shape().to_vec(),
            });
        }
        // Native GPU broadcast (no round-trip): lhs[B, N] + rhs[N] → [B, N].
        let x_g = to_gpu(self, x).map_err(|e| wgpu_err("add_bias", e))?;
        let bias_g = to_gpu(self, bias).map_err(|e| wgpu_err("add_bias", e))?;
        let (out, out_shape) =
            dispatch_binary_broadcast(self, "add", &x_g, x.shape(), &bias_g, bias.shape())
                .map_err(|e| wgpu_err("add_bias", e))?;
        drop((x_g, bias_g));
        Ok(finish_wgpu_op(out, out_shape))
    }

    /// Reshape — pure metadata change (no kernel needed). With Storage
    /// Option B, the data lives in the CPU shadow, so we rebuild the
    /// Tensor from the contiguous F32 slice with the new shape.
    fn reshape(&self, src: &Tensor, shape: &[usize]) -> Result<Tensor, BackendError> {
        let numel: usize = shape.iter().product();
        if numel != src.numel() {
            return Err(BackendError::ShapeMismatch {
                op: "reshape",
                lhs: src.shape().to_vec(),
                rhs: shape.to_vec(),
            });
        }
        // Storage Option A: under the hood reshape is a metadata-only
        // op (same buffer, new layout). For now we keep the GPU buffer
        // alive by cloning the `Storage::Wgpu(handle)` and rebuilding
        // the Tensor with the new shape — no host trip. CPU storage
        // also takes the metadata-only path via `from_vec`.
        if let Some(wgpu_storage) = src.as_wgpu_storage() {
            return Ok(Tensor::from_wgpu_storage(
                wgpu_storage.clone(),
                shape.to_vec(),
                src.dtype(),
            ));
        }
        // CPU path: copy bytes (cheap, contiguous) and re-tag Wgpu so
        // the dispatch chain stays consistent (this branch is hit
        // only for Tensors that have not yet been migrated to
        // Storage::Wgpu — typically scalar broadcasts in autograd).
        let data = src.as_slice::<f32>().ok_or_else(|| {
            BackendError::NumericalError("reshape: expected contiguous F32".to_string())
        })?;
        Tensor::from_vec(shape.to_vec(), data.to_vec())
            .map(tag_wgpu)
            .map_err(|e| BackendError::NumericalError(format!("reshape build: {e}")))
    }

    /// Reduce-sum along `dims`. Native GPU fast path for 2D inputs
    /// with single-axis reduction — the common autograd case (e.g.
    /// AddBackward `unbroadcast_to` reducing axis 0 of `[B, N]` →
    /// `[N]`, axis 1 of `[B, N]` → `[B]`). Composed via existing
    /// `reduce_rows` (axis = last) and `transpose2d + reduce_rows`
    /// (axis = first). Other shapes fall back to CPU (3D+ tensors,
    /// multi-axis reductions) until generalised in Task N.
    fn sum_dim(&self, src: &Tensor, dims: &[usize], keepdim: bool) -> Result<Tensor, BackendError> {
        if let Some(out) = sum_or_mean_dim_2d(self, src, dims, keepdim, ReduceKind::Sum)? {
            return Ok(out);
        }
        let host_src = crate::transfer::tensor_to_cpu(src).map_err(|e| wgpu_err("sum_dim", e))?;
        let result = cpu_backend().sum_dim(&host_src, dims, keepdim)?;
        Ok(tag_wgpu(result))
    }

    /// Reduce-mean along `dims`. Same fast path as [`sum_dim`].
    fn mean_dim(
        &self,
        src: &Tensor,
        dims: &[usize],
        keepdim: bool,
    ) -> Result<Tensor, BackendError> {
        if let Some(out) = sum_or_mean_dim_2d(self, src, dims, keepdim, ReduceKind::Mean)? {
            return Ok(out);
        }
        let host_src = crate::transfer::tensor_to_cpu(src).map_err(|e| wgpu_err("mean_dim", e))?;
        let result = cpu_backend().mean_dim(&host_src, dims, keepdim)?;
        Ok(tag_wgpu(result))
    }

    /// Sum the upstream `grad` down to `target_shape` so it matches the
    /// shape of the input that was broadcast on the forward path.
    /// Composes via `self.sum_dim` + `self.reshape` — same algorithm as
    /// CpuBackend's `unbroadcast_to_impl` but expressed against the trait
    /// surface so it works for any backend with those two primitives.
    ///
    /// Unlocks Add/Sub/Mul/Div backward on Wgpu via the autograd
    /// `crate::broadcast::unbroadcast_to(device, grad, target_shape)`
    /// helper which dispatches here.
    fn unbroadcast_to(
        &self,
        grad: &Tensor,
        target_shape: &[usize],
    ) -> Result<Tensor, BackendError> {
        let grad_shape = grad.shape();
        if grad_shape == target_shape {
            return Ok(grad.clone());
        }
        let g_ndim = grad_shape.len();
        let t_ndim = target_shape.len();
        if t_ndim > g_ndim {
            return Err(BackendError::ShapeMismatch {
                op: "unbroadcast_to",
                lhs: grad_shape.to_vec(),
                rhs: target_shape.to_vec(),
            });
        }
        let pad = g_ndim - t_ndim;
        let mut padded = vec![1usize; pad];
        padded.extend_from_slice(target_shape);
        for (&g, &t) in grad_shape.iter().zip(padded.iter()) {
            if t != 1 && t != g {
                return Err(BackendError::ShapeMismatch {
                    op: "unbroadcast_to",
                    lhs: grad_shape.to_vec(),
                    rhs: target_shape.to_vec(),
                });
            }
        }
        let reduce_axes: Vec<usize> = padded
            .iter()
            .zip(grad_shape.iter())
            .enumerate()
            .filter_map(|(axis, (&p, &g))| if p == 1 && g > 1 { Some(axis) } else { None })
            .collect();
        let reduced = if reduce_axes.is_empty() {
            grad.clone()
        } else {
            self.sum_dim(grad, &reduce_axes, true)?
        };
        if reduced.shape() == target_shape {
            Ok(reduced)
        } else {
            self.reshape(&reduced, target_shape)
        }
    }

    // -------------------- CPU-fallback overrides (correctness-first) --------------------
    //
    // These methods proxy through CpuBackend and re-tag the result as Wgpu.
    // Round-trip cost is identical to the native Wgpu helpers since data
    // already lives in the CPU shadow under Storage Option B. Native WGSL
    // kernels are perf optimisation backlog — but for plan completeness
    // every method on the trait surface returns the correct result.

    fn abs(&self, src: &Tensor) -> Result<Tensor, BackendError> {
        cpu_backend().abs(&host(src)?).map(tag_wgpu)
    }
    fn sqrt(&self, src: &Tensor) -> Result<Tensor, BackendError> {
        cpu_backend().sqrt(&host(src)?).map(tag_wgpu)
    }
    fn exp(&self, src: &Tensor) -> Result<Tensor, BackendError> {
        cpu_backend().exp(&host(src)?).map(tag_wgpu)
    }
    fn log(&self, src: &Tensor) -> Result<Tensor, BackendError> {
        cpu_backend().log(&host(src)?).map(tag_wgpu)
    }
    fn pow_scalar(&self, src: &Tensor, exponent: f64) -> Result<Tensor, BackendError> {
        cpu_backend()
            .pow_scalar(&host(src)?, exponent)
            .map(tag_wgpu)
    }
    fn leaky_relu(&self, src: &Tensor, slope: f64) -> Result<Tensor, BackendError> {
        cpu_backend().leaky_relu(&host(src)?, slope).map(tag_wgpu)
    }
    fn softmax(&self, src: &Tensor, dim: usize) -> Result<Tensor, BackendError> {
        cpu_backend().softmax(&host(src)?, dim).map(tag_wgpu)
    }
    fn log_softmax(&self, src: &Tensor, dim: usize) -> Result<Tensor, BackendError> {
        cpu_backend().log_softmax(&host(src)?, dim).map(tag_wgpu)
    }
    fn transpose(&self, src: &Tensor, d0: usize, d1: usize) -> Result<Tensor, BackendError> {
        // Native WGSL fast path for 2D transpose with axes (0, 1) — by
        // far the most common case in autograd (matmul backward,
        // attention QKV permutation). Falls back to CPU for higher
        // ranks until the WGSL kernel is generalised.
        let shape = src.shape();
        if shape.len() == 2 && ((d0 == 0 && d1 == 1) || (d0 == 1 && d1 == 0)) {
            let m = shape[0];
            let n = shape[1];
            let inp = to_gpu(self, src).map_err(|e| wgpu_err("transpose", e))?;
            let out = crate::transpose::transpose2d(self, &inp, m, n)
                .map_err(|e| wgpu_err("transpose", e))?;
            return Ok(finish_wgpu_op(out, vec![n, m]));
        }
        cpu_backend().transpose(&host(src)?, d0, d1).map(tag_wgpu)
    }
    fn bmm(&self, lhs: &Tensor, rhs: &Tensor) -> Result<Tensor, BackendError> {
        cpu_backend().bmm(&host(lhs)?, &host(rhs)?).map(tag_wgpu)
    }
    fn index_select(
        &self,
        src: &Tensor,
        dim: usize,
        indices: &Tensor,
    ) -> Result<Tensor, BackendError> {
        cpu_backend()
            .index_select(&host(src)?, dim, &host(indices)?)
            .map(tag_wgpu)
    }
    fn scatter_add(
        &self,
        dst: &Tensor,
        dim: usize,
        idx: &Tensor,
        src: &Tensor,
    ) -> Result<Tensor, BackendError> {
        cpu_backend()
            .scatter_add(&host(dst)?, dim, &host(idx)?, &host(src)?)
            .map(tag_wgpu)
    }
    fn gather(&self, src: &Tensor, dim: usize, idx: &Tensor) -> Result<Tensor, BackendError> {
        cpu_backend()
            .gather(&host(src)?, dim, &host(idx)?)
            .map(tag_wgpu)
    }
    fn cross_entropy(
        &self,
        input: &Tensor,
        target: &Tensor,
        reduction: rustorch_cpu::backend::Reduction,
    ) -> Result<Tensor, BackendError> {
        cpu_backend()
            .cross_entropy(&host(input)?, &host(target)?, reduction)
            .map(tag_wgpu)
    }
    fn mse_loss(
        &self,
        input: &Tensor,
        target: &Tensor,
        reduction: rustorch_cpu::backend::Reduction,
    ) -> Result<Tensor, BackendError> {
        // Native composition stays on GPU throughout: diff → sq → reduce.
        // Uses the existing dispatch_binary kernels (sub, mul) and the
        // full-tensor reduce (sum / mean), all of which produce
        // Storage::Wgpu output. No host trip — major win on the
        // training-loop hot path.
        use rustorch_cpu::backend::Reduction;
        let diff = self.sub(input, target)?;
        let sq = self.mul(&diff, &diff)?;
        match reduction {
            Reduction::Mean => self.mean(&sq),
            Reduction::Sum => self.sum(&sq),
            Reduction::None => Ok(sq),
        }
    }
    fn nll_loss(
        &self,
        log_probs: &Tensor,
        target: &Tensor,
        reduction: rustorch_cpu::backend::Reduction,
    ) -> Result<Tensor, BackendError> {
        cpu_backend()
            .nll_loss(&host(log_probs)?, &host(target)?, reduction)
            .map(tag_wgpu)
    }
    fn softmax_grad(
        &self,
        grad: &Tensor,
        output: &Tensor,
        dim: usize,
    ) -> Result<Tensor, BackendError> {
        cpu_backend()
            .softmax_grad(&host(grad)?, &host(output)?, dim)
            .map(tag_wgpu)
    }
    fn argmax(&self, src: &Tensor, dim: usize, keepdim: bool) -> Result<Tensor, BackendError> {
        cpu_backend()
            .argmax(&host(src)?, dim, keepdim)
            .map(tag_wgpu)
    }

    // Comparison ops (CPU fallback) — used by ReluBackward (gt) and for
    // user-facing predicate ops. No native WGSL kernel yet.
    fn ne(&self, lhs: &Tensor, rhs: &Tensor) -> Result<Tensor, BackendError> {
        cpu_backend().ne(&host(lhs)?, &host(rhs)?).map(tag_wgpu)
    }
    fn lt(&self, lhs: &Tensor, rhs: &Tensor) -> Result<Tensor, BackendError> {
        cpu_backend().lt(&host(lhs)?, &host(rhs)?).map(tag_wgpu)
    }
    fn le(&self, lhs: &Tensor, rhs: &Tensor) -> Result<Tensor, BackendError> {
        cpu_backend().le(&host(lhs)?, &host(rhs)?).map(tag_wgpu)
    }
    fn gt(&self, lhs: &Tensor, rhs: &Tensor) -> Result<Tensor, BackendError> {
        cpu_backend().gt(&host(lhs)?, &host(rhs)?).map(tag_wgpu)
    }
    fn ge(&self, lhs: &Tensor, rhs: &Tensor) -> Result<Tensor, BackendError> {
        cpu_backend().ge(&host(lhs)?, &host(rhs)?).map(tag_wgpu)
    }

    // Cast — needed when ReluBackward converts the bool mask to f32 to
    // multiply against grad.
    fn cast(
        &self,
        src: &Tensor,
        target: rustorch_core::tensor::dtype::Dtype,
    ) -> Result<Tensor, BackendError> {
        cpu_backend().cast(src, target).map(tag_wgpu)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend_singleton::try_wgpu_backend;
    use rustorch_core::tensor::tensor_impl::Tensor;

    fn close(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() <= tol
    }

    fn assert_close_slice(actual: &[f32], expected: &[f32], tol: f32, label: &str) {
        assert_eq!(actual.len(), expected.len(), "{label}: len mismatch");
        for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                close(*a, *e, tol),
                "{label}: idx {i} actual {a} expected {e} (tol {tol})"
            );
        }
    }

    #[test]
    #[cfg_attr(not(feature = "gpu-tests"), ignore = "needs a GPU adapter")]
    fn impl_backend_name_is_wgpu() {
        let backend: &dyn Backend = try_wgpu_backend().expect("no GPU adapter for tests");
        assert_eq!(backend.name(), "wgpu");
    }

    #[test]
    #[cfg_attr(not(feature = "gpu-tests"), ignore = "needs a GPU adapter")]
    fn add_through_trait_matches_cpu() {
        let backend: &dyn Backend = try_wgpu_backend().expect("no GPU adapter for tests");
        let a = Tensor::from_vec([4usize], vec![1.0_f32, 2.0, 3.0, 4.0]).unwrap();
        let b = Tensor::from_vec([4usize], vec![10.0_f32, 20.0, 30.0, 40.0]).unwrap();
        let c = backend.add(&a, &b).unwrap();
        assert_eq!(c.shape(), [4]);
        assert_close_slice(
            c.as_slice::<f32>().unwrap(),
            &[11.0, 22.0, 33.0, 44.0],
            1e-5,
            "wgpu add",
        );
    }

    #[test]
    #[cfg_attr(not(feature = "gpu-tests"), ignore = "needs a GPU adapter")]
    fn relu_through_trait_matches_cpu() {
        let backend: &dyn Backend = try_wgpu_backend().expect("no GPU adapter for tests");
        let a = Tensor::from_vec([5usize], vec![-2.0_f32, -1.0, 0.0, 1.0, 2.0]).unwrap();
        let r = backend.relu(&a).unwrap();
        assert_close_slice(
            r.as_slice::<f32>().unwrap(),
            &[0.0, 0.0, 0.0, 1.0, 2.0],
            1e-5,
            "wgpu relu",
        );
    }

    #[test]
    #[cfg_attr(not(feature = "gpu-tests"), ignore = "needs a GPU adapter")]
    fn matmul_through_trait_matches_cpu() {
        let backend: &dyn Backend = try_wgpu_backend().expect("no GPU adapter for tests");
        // [2, 3] @ [3, 2] = [2, 2]
        let a = Tensor::from_vec([2usize, 3], vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let b = Tensor::from_vec([3usize, 2], vec![1.0_f32, 0.0, 0.0, 1.0, 1.0, 1.0]).unwrap();
        let c = backend.matmul(&a, &b).unwrap();
        // Row 0: [1+0+3, 0+2+3] = [4, 5]
        // Row 1: [4+0+6, 0+5+6] = [10, 11]
        assert_close_slice(
            c.as_slice::<f32>().unwrap(),
            &[4.0, 5.0, 10.0, 11.0],
            1e-5,
            "wgpu matmul",
        );
    }

    #[test]
    #[cfg_attr(not(feature = "gpu-tests"), ignore = "needs a GPU adapter")]
    fn eq_through_trait_uses_cpu_fallback() {
        // `eq` has no wgpu kernel yet — the impl falls through to
        // cpu_backend. Verify the result matches CPU semantics by
        // dispatching the same op on both backends and comparing.
        let backend: &dyn Backend = try_wgpu_backend().expect("no GPU adapter for tests");
        let a = Tensor::from_vec([4usize], vec![1.0_f32, 2.0, 3.0, 4.0]).unwrap();
        let b = Tensor::from_vec([4usize], vec![1.0_f32, 0.0, 3.0, 0.0]).unwrap();
        let out_wgpu = backend.eq(&a, &b).unwrap();
        let out_cpu = cpu_backend().eq(&a, &b).unwrap();
        assert_eq!(out_wgpu.shape(), out_cpu.shape());
        // Bool tensors have dtype Bool; compare as raw bytes.
        assert_eq!(
            out_wgpu.as_slice::<bool>().unwrap_or_default(),
            out_cpu.as_slice::<bool>().unwrap_or_default(),
            "eq fallback must match CPU semantics"
        );
    }
}
