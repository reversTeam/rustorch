//! `impl Backend for MetalBackend` — surface the Metal kernels
//! through the `rustorch_cpu::backend::Backend` trait so the
//! autograd dispatcher can pick this backend at op time via
//! `Variable::device() == Device::Metal`.
//!
//! ## P3.Z Task J phasing
//!
//! v1 (this commit): native `matmul_simdgroup_f32` + `add` + `mul`
//! kernel set, everything else CPU-fallback through `host()` (mirrors
//! the early WgpuBackend pattern). Each subsequent commit adds
//! native Metal kernels and removes a CPU-fallback override.
//!
//! Native kernels operate on `Storage::Metal(...)` directly — no
//! host round trip. CPU-fallback ops materialise via
//! [`crate::transfer::tensor_to_cpu`], run on `cpu_backend()`, and
//! re-tag the output as `Device::Metal`.

use crate::backend::MetalBackend;
use crate::error::MetalError;
use crate::kernels::{
    abs_f32, add_bias_f32, add_f32, add_scalar_f32, cast_bf16_to_f32_pub, div_f32, div_scalar_f32,
    exp_f32, log_f32, matmul_bf16_in_f32_out, matmul_simdgroup_bf16_pub, matmul_simdgroup_f32,
    matmul_simdgroup_f32_coarsened, matmul_simdgroup_f32_coarsened_wide,
    matmul_simdgroup_f32_multisg, matmul_simdgroup_f32_via_bf16, mean_dim_2d_f32, mean_f32,
    mul_f32, mul_scalar_f32, neg_f32, relu_f32, sigmoid_f32, silu_f32, sqrt_f32, sub_f32,
    sub_scalar_f32, sum_dim_2d_f32, sum_f32, tanh_f32, transpose2d_f32,
};
use crate::transfer::tensor_to_cpu;
use rustorch_core::tensor::device::Device;
use rustorch_core::tensor::tensor_impl::Tensor;
use rustorch_cpu::backend::Backend;
use rustorch_cpu::cpu_backend::cpu_backend;
use rustorch_cpu::error::BackendError;

fn metal_err(op: &'static str, err: MetalError) -> BackendError {
    BackendError::NumericalError(format!("{op}: metal: {err}"))
}

/// Materialise a Tensor to host (CPU storage) so `cpu_backend()` can
/// read its bytes. P3.Z Task J: `Storage::Metal` returns an empty
/// slice from `as_slice` so any CPU-fallback path needs an explicit
/// download first. Ops with native Metal kernels skip this entirely.
fn host(t: &Tensor) -> Result<Tensor, BackendError> {
    tensor_to_cpu(t).map_err(|e| metal_err("cpu_fallback", e))
}

/// Re-tag a Tensor as living on `Device::Metal` after a CPU-fallback
/// computation. Mirrors `tag_wgpu` in rustorch-wgpu.
fn tag_metal(t: Tensor) -> Tensor {
    t.with_device(Device::Metal)
}

/// Upload a Tensor's bytes to a fresh Metal buffer if it isn't
/// already on `Storage::Metal`. Returns the inner `core::MetalStorage`.
fn to_gpu(
    backend: &MetalBackend,
    t: &Tensor,
) -> Result<rustorch_core::tensor::storage::MetalStorage, MetalError> {
    use rustorch_core::tensor::storage::MetalStorage;
    if let Some(s) = t.as_metal_storage() {
        return Ok(s.clone());
    }
    // CPU-storage tensor: upload via shared-mode buffer (unified
    // memory on Apple Silicon makes the upload essentially a memcpy
    // into the same physical RAM).
    let bytes = t.numel() * 4; // F32 only for now
    let buffer = backend.alloc_shared(bytes)?;
    let data = t.as_slice::<f32>().ok_or_else(|| {
        MetalError::ShapeMismatch("to_gpu: source tensor must be contiguous F32 on CPU".to_string())
    })?;
    // SAFETY: shared-storage buffer; `contents()` is host-mapped.
    unsafe {
        let dst = buffer.contents() as *mut f32;
        for (i, &v) in data.iter().enumerate() {
            *dst.add(i) = v;
        }
    }
    Ok(MetalStorage::standalone(buffer, bytes))
}

/// Wrap a fresh Metal `Buffer` (returned by a kernel) into a Tensor
/// with `Storage::Metal` natively — no host trip. Mirrors
/// `finish_wgpu_op` in rustorch-wgpu.
fn finish_metal_op(
    out: metal::Buffer,
    shape: Vec<usize>,
    dtype: rustorch_core::tensor::dtype::Dtype,
) -> Tensor {
    let n_bytes = shape.iter().product::<usize>() * 4;
    let core = rustorch_core::tensor::storage::MetalStorage::standalone(out, n_bytes);
    Tensor::from_metal_storage(core, shape, dtype)
}

/// Run a native binary Metal kernel: upload (or reuse) lhs+rhs,
/// dispatch, return Tensor with Storage::Metal. **Caller MUST have
/// already verified `lhs.shape() == rhs.shape()`** — broadcasting
/// goes through the CPU fallback path inside each Backend impl.
fn binary_native<F>(
    backend: &MetalBackend,
    op_name: &'static str,
    lhs: &Tensor,
    rhs: &Tensor,
    kernel: F,
) -> Result<Tensor, BackendError>
where
    F: FnOnce(
        &MetalBackend,
        &metal::Buffer,
        &metal::Buffer,
        usize,
    ) -> Result<metal::Buffer, MetalError>,
{
    let l = to_gpu(backend, lhs).map_err(|e| metal_err(op_name, e))?;
    let r = to_gpu(backend, rhs).map_err(|e| metal_err(op_name, e))?;
    let out = kernel(backend, &l, &r, lhs.numel()).map_err(|e| metal_err(op_name, e))?;
    Ok(finish_metal_op(out, lhs.shape().to_vec(), lhs.dtype()))
}

/// Native Metal fast path for `sum_dim` / `mean_dim` on 2D inputs
/// with a single reduction axis. Returns `None` for shapes/axes the
/// 2D kernel doesn't yet handle (3D+ tensors, multi-axis reductions)
/// so the caller can CPU-fallback.
fn sum_or_mean_dim_2d(
    backend: &MetalBackend,
    src: &Tensor,
    dims: &[usize],
    keepdim: bool,
    is_mean: bool,
) -> Result<Option<Tensor>, BackendError> {
    let shape = src.shape();
    if shape.len() != 2 || dims.len() != 1 {
        return Ok(None);
    }
    let (d0, d1) = (shape[0], shape[1]);
    let axis = dims[0];
    if axis > 1 {
        return Ok(None);
    }
    let op_name = if is_mean { "mean_dim" } else { "sum_dim" };
    let s = to_gpu(backend, src).map_err(|e| metal_err(op_name, e))?;
    let out = if is_mean {
        mean_dim_2d_f32(backend, &s, d0, d1, axis).map_err(|e| metal_err(op_name, e))?
    } else {
        sum_dim_2d_f32(backend, &s, d0, d1, axis).map_err(|e| metal_err(op_name, e))?
    };
    let out_shape = if axis == 0 {
        if keepdim {
            vec![1, d1]
        } else {
            vec![d1]
        }
    } else if keepdim {
        vec![d0, 1]
    } else {
        vec![d0]
    };
    Ok(Some(finish_metal_op(out, out_shape, src.dtype())))
}

/// Detect the "rhs is a 0-dim or numel-1 scalar" pattern and dispatch
/// the corresponding `*_scalar_f32` kernel. Returns `None` when the
/// shapes don't match a scalar-broadcast (caller should CPU-fallback).
///
/// Eliminates the hidden round-trip in autograd backward paths that
/// produce `Tensor::scalar(...)` and call `mul(x_metal, scalar_cpu)`
/// (e.g. MseBackward computes `g = diff * (2 / n)` this way).
fn scalar_broadcast_native<F>(
    backend: &MetalBackend,
    lhs: &Tensor,
    rhs: &Tensor,
    op_name: &'static str,
    kernel: F,
) -> Result<Option<Tensor>, BackendError>
where
    F: FnOnce(&MetalBackend, &metal::Buffer, f32, usize) -> Result<metal::Buffer, MetalError>,
{
    // Only handle the "rhs is the scalar" case for now (most common
    // in autograd: `tensor * scale_constant`). Symmetric op (lhs
    // scalar) would need swapped operands, doable as follow-up.
    if rhs.numel() != 1 || lhs.numel() == 1 {
        return Ok(None);
    }
    let scalar = if rhs.storage().is_cpu() {
        rhs.as_slice::<f32>().ok_or_else(|| {
            metal_err(
                op_name,
                MetalError::ShapeMismatch("scalar must be contiguous F32".to_string()),
            )
        })?[0]
    } else {
        // Scalar lives on Metal — read its single value via tensor_to_cpu.
        // Cheap (4 bytes) but synchronous; only triggers if the autograd
        // path produces a Metal-resident scalar (rare in practice).
        let host_t = host(rhs)?;
        host_t.as_slice::<f32>().ok_or_else(|| {
            metal_err(
                op_name,
                MetalError::ShapeMismatch("scalar download produced non-CPU storage".to_string()),
            )
        })?[0]
    };
    let lhs_buf = to_gpu(backend, lhs).map_err(|e| metal_err(op_name, e))?;
    let out = kernel(backend, &lhs_buf, scalar, lhs.numel()).map_err(|e| metal_err(op_name, e))?;
    Ok(Some(finish_metal_op(
        out,
        lhs.shape().to_vec(),
        lhs.dtype(),
    )))
}

/// Run a native unary Metal kernel.
fn unary_native<F>(
    backend: &MetalBackend,
    op_name: &'static str,
    src: &Tensor,
    kernel: F,
) -> Result<Tensor, BackendError>
where
    F: FnOnce(&MetalBackend, &metal::Buffer, usize) -> Result<metal::Buffer, MetalError>,
{
    let s = to_gpu(backend, src).map_err(|e| metal_err(op_name, e))?;
    let out = kernel(backend, &s, src.numel()).map_err(|e| metal_err(op_name, e))?;
    Ok(finish_metal_op(out, src.shape().to_vec(), src.dtype()))
}

impl Backend for MetalBackend {
    fn name(&self) -> &'static str {
        "metal"
    }

    // -------- Native Metal kernels (Storage::Metal in/out) ----------

    /// Element-wise add via the native Metal `add_f32` kernel.
    fn add(&self, lhs: &Tensor, rhs: &Tensor) -> Result<Tensor, BackendError> {
        if lhs.shape() == rhs.shape() {
            return binary_native(self, "add", lhs, rhs, add_f32);
        }
        if let Some(tensor) = scalar_broadcast_native(self, lhs, rhs, "add", add_scalar_f32)? {
            return Ok(tensor);
        }
        cpu_backend().add(&host(lhs)?, &host(rhs)?).map(tag_metal)
    }

    /// `lhs @ rhs` via `simdgroup_matrix<float, 8, 8>` — the
    /// perf-critical kernel. Falls back to CPU for non-rank-2
    /// shapes or shapes not divisible by 8 (Task J Phase 2 lifts
    /// the 8-alignment constraint).
    fn matmul(&self, lhs: &Tensor, rhs: &Tensor) -> Result<Tensor, BackendError> {
        let l_shape = lhs.shape();
        let r_shape = rhs.shape();
        if l_shape.len() != 2 || r_shape.len() != 2 {
            return cpu_backend()
                .matmul(&host(lhs)?, &host(rhs)?)
                .map(tag_metal);
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
        // simdgroup_matrix v1 needs M, K, N divisible by 8.
        if m % 8 != 0 || k1 % 8 != 0 || n % 8 != 0 || !self.supports_metal3() {
            return cpu_backend()
                .matmul(&host(lhs)?, &host(rhs)?)
                .map(tag_metal);
        }
        let l = to_gpu(self, lhs).map_err(|e| metal_err("matmul", e))?;
        let r = to_gpu(self, rhs).map_err(|e| metal_err("matmul", e))?;
        // Pick the highest-throughput kernel that fits the shape:
        // - bf16-in/f32-out (cached casts): m%16==0 & n%256==0
        //   Mixed-precision matmul: f32 → bf16 inputs are CACHED in
        //   MetalBackend.bf16_cache so casts are amortised across
        //   multiple matmul calls on the same buffers (e.g. forward
        //   xs@w + backward dxs@w.T both reuse w's bf16 cache).
        //   Output stays f32 directly via implicit widening in the
        //   simdgroup_load — no output cast needed.
        // - coarsened_wide (16 sg, 16x256 output): m%16==0 & n%256==0
        // - coarsened     (8 sg, 8x256, 4-tile/sg): n%256==0
        // - multi-sg      (8 sg, 8x64, 1-tile/sg):  n%64==0
        // - single-sg     (1 sg, 8x8, 1-tile/sg):   n%8==0 fallback
        // Pick the highest-throughput kernel that fits the shape:
        // - coarsened_wide (16 sg, 16x256 output): m%16==0 & n%256==0
        // - coarsened     (8 sg, 8x256, 4-tile/sg): n%256==0
        // - multi-sg      (8 sg, 8x64, 1-tile/sg):  n%64==0
        // - single-sg     (1 sg, 8x8, 1-tile/sg):   n%8==0 fallback
        //
        // bf16 paths (matmul_simdgroup_bf16_pub + bf16_cache) are
        // implemented and available but NOT the default — the bench's
        // shape mix has a lot of fresh transpose outputs as matmul
        // inputs (no cache hit), so the cast overhead exceeds the bf16
        // speedup. Will be the default once we wire matmul_with_transpose
        // to skip the explicit transpose dispatch (Apple's simdgroup_load
        // supports a transpose flag).
        let _ = matmul_simdgroup_bf16_pub;
        let _ = cast_bf16_to_f32_pub;
        let out = if m % 16 == 0 && n % 256 == 0 {
            matmul_simdgroup_f32_coarsened_wide(self, &l, &r, m, k1, n)
                .map_err(|e| metal_err("matmul", e))?
        } else if n % 256 == 0 {
            matmul_simdgroup_f32_coarsened(self, &l, &r, m, k1, n)
                .map_err(|e| metal_err("matmul", e))?
        } else if n % 64 == 0 {
            matmul_simdgroup_f32_multisg(self, &l, &r, m, k1, n)
                .map_err(|e| metal_err("matmul", e))?
        } else {
            matmul_simdgroup_f32(self, &l, &r, m, k1, n).map_err(|e| metal_err("matmul", e))?
        };
        // Keep symbols live for future bf16 mixed-precision activation
        // (when params are stored bf16 across steps).
        let _ = matmul_simdgroup_f32_via_bf16;
        let _ = matmul_bf16_in_f32_out;
        let core = rustorch_core::tensor::storage::MetalStorage::standalone(out, m * n * 4);
        Ok(Tensor::from_metal_storage(core, vec![m, n], lhs.dtype()))
    }

    /// Fused `matmul + bias` — single dispatch that folds the bias
    /// broadcast into the matmul write-back. Saves one `add_bias`
    /// dispatch + one full `[M,N]` intermediate buffer per Linear
    /// forward (~4 MB at 1024² f32). Falls back to the default
    /// `matmul + add_bias` composition outside the fast-path tile
    /// constraints (`m%16==0`, `k%8==0`, `n%256==0`, rank-2 inputs,
    /// rank-1 bias matching `n`, Metal3-capable device).
    fn matmul_with_bias(
        &self,
        lhs: &Tensor,
        rhs: &Tensor,
        bias: &Tensor,
    ) -> Result<Tensor, BackendError> {
        let l_shape = lhs.shape();
        let r_shape = rhs.shape();
        let bias_shape = bias.shape();
        let fast_path_eligible = self.supports_metal3()
            && l_shape.len() == 2
            && r_shape.len() == 2
            && bias_shape.len() == 1
            && r_shape[1] == bias_shape[0]
            && l_shape[1] == r_shape[0]
            && l_shape[0] % 16 == 0
            && l_shape[1] % 8 == 0
            && r_shape[1] % 256 == 0;
        if !fast_path_eligible {
            // Default composition (matmul + add_bias).
            let mm = self.matmul(lhs, rhs)?;
            return self.add_bias(&mm, bias);
        }
        let (m, k) = (l_shape[0], l_shape[1]);
        let n = r_shape[1];
        let l = to_gpu(self, lhs).map_err(|e| metal_err("matmul_with_bias", e))?;
        let r = to_gpu(self, rhs).map_err(|e| metal_err("matmul_with_bias", e))?;
        let bb = to_gpu(self, bias).map_err(|e| metal_err("matmul_with_bias", e))?;
        let out =
            crate::kernels::matmul_simdgroup_f32_coarsened_wide_bias(self, &l, &r, &bb, m, k, n)
                .map_err(|e| metal_err("matmul_with_bias", e))?;
        let core = rustorch_core::tensor::storage::MetalStorage::standalone(out, m * n * 4);
        Ok(Tensor::from_metal_storage(core, vec![m, n], lhs.dtype()))
    }

    /// Transpose-aware matmul. Uses Apple's `simdgroup_load(...,
    /// transpose=true)` to fuse the transpose into the load, skipping
    /// the explicit transpose dispatch + intermediate buffer that the
    /// default trait impl would emit. In `MatMulBackward` this saves 2
    /// dispatches per training step (one for `dW = X^T @ dY`, one for
    /// `dX = dY @ W^T`).
    ///
    /// Falls back to the default `transpose + matmul` path when the
    /// shapes don't fit the 16×8×256 tile constraints, when neither
    /// transpose flag is set, or when both are set (the doubly-
    /// transposed case is rare enough we keep it on the slow path).
    fn matmul_with_transposes(
        &self,
        lhs: &Tensor,
        rhs: &Tensor,
        transpose_a: bool,
        transpose_b: bool,
    ) -> Result<Tensor, BackendError> {
        // Fast path requires Metal3 and exactly one transpose flag.
        if !self.supports_metal3() || transpose_a == transpose_b {
            // No transpose, or both transposed → defer to default impl.
            // (Default uses self.transpose + self.matmul, which still
            // routes through our coarsened_wide kernel.)
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
            return self.matmul(&lhs_eff, &rhs_eff);
        }
        let l_shape = lhs.shape();
        let r_shape = rhs.shape();
        if l_shape.len() != 2 || r_shape.len() != 2 {
            // Fall back via default for non-rank-2.
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
            return self.matmul(&lhs_eff, &rhs_eff);
        }
        // Resolve effective dimensions after the implicit transpose:
        // - transpose_a:  A is [K, M], output rows = M, common dim K = A.shape[0]
        // - transpose_b:  B is [N, K], output cols = N, common dim K = B.shape[1]
        let (m, k_lhs) = if transpose_a {
            (l_shape[1], l_shape[0])
        } else {
            (l_shape[0], l_shape[1])
        };
        let (k_rhs, n) = if transpose_b {
            (r_shape[1], r_shape[0])
        } else {
            (r_shape[0], r_shape[1])
        };
        if k_lhs != k_rhs {
            return Err(BackendError::ShapeMismatch {
                op: "matmul_with_transposes",
                lhs: l_shape.to_vec(),
                rhs: r_shape.to_vec(),
            });
        }
        let k = k_lhs;
        // Tile constraints for the coarsened-wide layout.
        let fits_wide = m % 16 == 0 && k % 8 == 0 && n % 256 == 0;
        if !fits_wide {
            // Smaller tiles: defer to default (transpose + matmul) which
            // picks coarsened / multisg / single inside `matmul`.
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
            return self.matmul(&lhs_eff, &rhs_eff);
        }

        let l = to_gpu(self, lhs).map_err(|e| metal_err("matmul_with_transposes", e))?;
        let r = to_gpu(self, rhs).map_err(|e| metal_err("matmul_with_transposes", e))?;
        let out = if transpose_a {
            // C = A^T @ B  | A:[K,M] B:[K,N] C:[M,N]
            crate::kernels::matmul_simdgroup_f32_a_t(self, &l, &r, m, k, n)
                .map_err(|e| metal_err("matmul_with_transposes", e))?
        } else {
            // transpose_b — C = A @ B^T | A:[M,K] B:[N,K] C:[M,N]
            crate::kernels::matmul_simdgroup_f32_b_t(self, &l, &r, m, k, n)
                .map_err(|e| metal_err("matmul_with_transposes", e))?
        };
        let core = rustorch_core::tensor::storage::MetalStorage::standalone(out, m * n * 4);
        Ok(Tensor::from_metal_storage(core, vec![m, n], lhs.dtype()))
    }

    // -------- CPU-fallback overrides for the remaining trait surface --------
    //
    // Each CPU fallback materialises the Metal tensor to host, runs
    // on `cpu_backend()`, and re-tags as `Device::Metal`. Subsequent
    // Task J commits replace these with native Metal kernels.

    fn sub(&self, lhs: &Tensor, rhs: &Tensor) -> Result<Tensor, BackendError> {
        if lhs.shape() == rhs.shape() {
            return binary_native(self, "sub", lhs, rhs, sub_f32);
        }
        if let Some(tensor) = scalar_broadcast_native(self, lhs, rhs, "sub", sub_scalar_f32)? {
            return Ok(tensor);
        }
        cpu_backend().sub(&host(lhs)?, &host(rhs)?).map(tag_metal)
    }
    fn mul(&self, lhs: &Tensor, rhs: &Tensor) -> Result<Tensor, BackendError> {
        if lhs.shape() == rhs.shape() {
            return binary_native(self, "mul", lhs, rhs, mul_f32);
        }
        if let Some(tensor) = scalar_broadcast_native(self, lhs, rhs, "mul", mul_scalar_f32)? {
            return Ok(tensor);
        }
        cpu_backend().mul(&host(lhs)?, &host(rhs)?).map(tag_metal)
    }
    fn div(&self, lhs: &Tensor, rhs: &Tensor) -> Result<Tensor, BackendError> {
        if lhs.shape() == rhs.shape() {
            return binary_native(self, "div", lhs, rhs, div_f32);
        }
        if let Some(tensor) = scalar_broadcast_native(self, lhs, rhs, "div", div_scalar_f32)? {
            return Ok(tensor);
        }
        cpu_backend().div(&host(lhs)?, &host(rhs)?).map(tag_metal)
    }
    fn neg(&self, src: &Tensor) -> Result<Tensor, BackendError> {
        unary_native(self, "neg", src, neg_f32)
    }
    fn relu(&self, src: &Tensor) -> Result<Tensor, BackendError> {
        unary_native(self, "relu", src, relu_f32)
    }
    fn eq(&self, lhs: &Tensor, rhs: &Tensor) -> Result<Tensor, BackendError> {
        cpu_backend().eq(&host(lhs)?, &host(rhs)?).map(tag_metal)
    }
    fn sigmoid(&self, src: &Tensor) -> Result<Tensor, BackendError> {
        unary_native(self, "sigmoid", src, sigmoid_f32)
    }
    fn tanh(&self, src: &Tensor) -> Result<Tensor, BackendError> {
        unary_native(self, "tanh", src, tanh_f32)
    }
    fn silu(&self, src: &Tensor) -> Result<Tensor, BackendError> {
        unary_native(self, "silu", src, silu_f32)
    }
    fn sum(&self, src: &Tensor) -> Result<Tensor, BackendError> {
        let s = to_gpu(self, src).map_err(|e| metal_err("sum", e))?;
        let out = sum_f32(self, &s, src.numel()).map_err(|e| metal_err("sum", e))?;
        Ok(finish_metal_op(out, vec![1], src.dtype()))
    }
    fn mean(&self, src: &Tensor) -> Result<Tensor, BackendError> {
        let s = to_gpu(self, src).map_err(|e| metal_err("mean", e))?;
        let out = mean_f32(self, &s, src.numel()).map_err(|e| metal_err("mean", e))?;
        Ok(finish_metal_op(out, vec![1], src.dtype()))
    }
    fn add_bias(&self, x: &Tensor, bias: &Tensor) -> Result<Tensor, BackendError> {
        // Native Metal add_bias: broadcasts a [N] bias across [B, N]
        // input. Stays on GPU end-to-end (no host trip). Matches the
        // wgpu native add_bias path.
        let xs = x.shape();
        let bs = bias.shape();
        if xs.len() != 2 || bs.len() != 1 || bs[0] != xs[1] {
            return cpu_backend()
                .add_bias(&host(x)?, &host(bias)?)
                .map(tag_metal);
        }
        let (b_dim, n_dim) = (xs[0], xs[1]);
        let xb = to_gpu(self, x).map_err(|e| metal_err("add_bias", e))?;
        let bb = to_gpu(self, bias).map_err(|e| metal_err("add_bias", e))?;
        let out =
            add_bias_f32(self, &xb, &bb, b_dim, n_dim).map_err(|e| metal_err("add_bias", e))?;
        Ok(finish_metal_op(out, vec![b_dim, n_dim], x.dtype()))
    }
    fn reshape(&self, src: &Tensor, shape: &[usize]) -> Result<Tensor, BackendError> {
        let numel: usize = shape.iter().product();
        if numel != src.numel() {
            return Err(BackendError::ShapeMismatch {
                op: "reshape",
                lhs: src.shape().to_vec(),
                rhs: shape.to_vec(),
            });
        }
        // Storage::Metal: metadata-only op, clone the GPU storage with
        // a new shape — no kernel needed, no host trip.
        if let Some(metal_storage) = src.as_metal_storage() {
            return Ok(Tensor::from_metal_storage(
                metal_storage.clone(),
                shape.to_vec(),
                src.dtype(),
            ));
        }
        // CPU-storage fallback (rare — typically scalar broadcasts).
        cpu_backend().reshape(&host(src)?, shape).map(tag_metal)
    }
    fn sum_dim(&self, src: &Tensor, dims: &[usize], keepdim: bool) -> Result<Tensor, BackendError> {
        // Native Metal fast path: 2D input + single axis (the common
        // unbroadcast_to / add_bias-grad case in the autograd backward
        // chain). Falls back to CPU for higher ranks / multi-axis.
        if let Some(out) = sum_or_mean_dim_2d(self, src, dims, keepdim, false)? {
            return Ok(out);
        }
        cpu_backend()
            .sum_dim(&host(src)?, dims, keepdim)
            .map(tag_metal)
    }
    fn mean_dim(
        &self,
        src: &Tensor,
        dims: &[usize],
        keepdim: bool,
    ) -> Result<Tensor, BackendError> {
        if let Some(out) = sum_or_mean_dim_2d(self, src, dims, keepdim, true)? {
            return Ok(out);
        }
        cpu_backend()
            .mean_dim(&host(src)?, dims, keepdim)
            .map(tag_metal)
    }
    fn unbroadcast_to(
        &self,
        grad: &Tensor,
        target_shape: &[usize],
    ) -> Result<Tensor, BackendError> {
        // Native composition via the backend's own sum_dim + reshape
        // (both Metal-native), keeping the gradient on GPU end-to-end.
        // Mirror of WgpuBackend::unbroadcast_to.
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
    fn abs(&self, src: &Tensor) -> Result<Tensor, BackendError> {
        unary_native(self, "abs", src, abs_f32)
    }
    fn sqrt(&self, src: &Tensor) -> Result<Tensor, BackendError> {
        unary_native(self, "sqrt", src, sqrt_f32)
    }
    fn exp(&self, src: &Tensor) -> Result<Tensor, BackendError> {
        unary_native(self, "exp", src, exp_f32)
    }
    fn log(&self, src: &Tensor) -> Result<Tensor, BackendError> {
        unary_native(self, "log", src, log_f32)
    }
    fn pow_scalar(&self, src: &Tensor, exponent: f64) -> Result<Tensor, BackendError> {
        cpu_backend()
            .pow_scalar(&host(src)?, exponent)
            .map(tag_metal)
    }
    fn leaky_relu(&self, src: &Tensor, slope: f64) -> Result<Tensor, BackendError> {
        cpu_backend().leaky_relu(&host(src)?, slope).map(tag_metal)
    }
    fn softmax(&self, src: &Tensor, dim: usize) -> Result<Tensor, BackendError> {
        cpu_backend().softmax(&host(src)?, dim).map(tag_metal)
    }
    fn log_softmax(&self, src: &Tensor, dim: usize) -> Result<Tensor, BackendError> {
        cpu_backend().log_softmax(&host(src)?, dim).map(tag_metal)
    }
    fn transpose(&self, src: &Tensor, d0: usize, d1: usize) -> Result<Tensor, BackendError> {
        // Native Metal 2D transpose for the (0, 1) / (1, 0) common
        // case (matmul backward, attention QKV permutation). Higher
        // ranks fall through to CPU until the kernel is generalised.
        let shape = src.shape();
        if shape.len() == 2 && ((d0 == 0 && d1 == 1) || (d0 == 1 && d1 == 0)) {
            let m = shape[0];
            let n = shape[1];
            let s = to_gpu(self, src).map_err(|e| metal_err("transpose", e))?;
            let out = transpose2d_f32(self, &s, m, n).map_err(|e| metal_err("transpose", e))?;
            return Ok(finish_metal_op(out, vec![n, m], src.dtype()));
        }
        cpu_backend().transpose(&host(src)?, d0, d1).map(tag_metal)
    }
    fn bmm(&self, lhs: &Tensor, rhs: &Tensor) -> Result<Tensor, BackendError> {
        cpu_backend().bmm(&host(lhs)?, &host(rhs)?).map(tag_metal)
    }
    fn mse_loss(
        &self,
        input: &Tensor,
        target: &Tensor,
        reduction: rustorch_cpu::backend::Reduction,
    ) -> Result<Tensor, BackendError> {
        // Fast path: Reduction::Mean / Reduction::Sum on rank-2+ same-shape
        // inputs use the fused `mse_reduce_f32` kernel (1 partial dispatch
        // + 1 finalise instead of sub + mul + reduce). Saves 2 dispatches
        // and 2 intermediate buffers per training step. None falls through
        // to (a-b)² composition which keeps the elementwise output.
        use rustorch_cpu::backend::Reduction;
        if input.shape() == target.shape() && matches!(reduction, Reduction::Mean | Reduction::Sum)
        {
            let n = input.numel();
            let a = to_gpu(self, input).map_err(|e| metal_err("mse_loss", e))?;
            let b = to_gpu(self, target).map_err(|e| metal_err("mse_loss", e))?;
            let kind = if matches!(reduction, Reduction::Mean) {
                1u32
            } else {
                0u32
            };
            let out = crate::kernels::mse_reduce_f32(self, &a, &b, n, kind)
                .map_err(|e| metal_err("mse_loss", e))?;
            return Ok(finish_metal_op(out, vec![1], input.dtype()));
        }
        // Fallback: composition for None / mismatched shapes.
        let diff = self.sub(input, target)?;
        let sq = self.mul(&diff, &diff)?;
        match reduction {
            Reduction::Mean => self.mean(&sq),
            Reduction::Sum => self.sum(&sq),
            Reduction::None => Ok(sq),
        }
    }
    fn cross_entropy(
        &self,
        input: &Tensor,
        target: &Tensor,
        reduction: rustorch_cpu::backend::Reduction,
    ) -> Result<Tensor, BackendError> {
        cpu_backend()
            .cross_entropy(&host(input)?, &host(target)?, reduction)
            .map(tag_metal)
    }
    fn nll_loss(
        &self,
        log_probs: &Tensor,
        target: &Tensor,
        reduction: rustorch_cpu::backend::Reduction,
    ) -> Result<Tensor, BackendError> {
        cpu_backend()
            .nll_loss(&host(log_probs)?, &host(target)?, reduction)
            .map(tag_metal)
    }
    fn ne(&self, lhs: &Tensor, rhs: &Tensor) -> Result<Tensor, BackendError> {
        cpu_backend().ne(&host(lhs)?, &host(rhs)?).map(tag_metal)
    }
    fn lt(&self, lhs: &Tensor, rhs: &Tensor) -> Result<Tensor, BackendError> {
        cpu_backend().lt(&host(lhs)?, &host(rhs)?).map(tag_metal)
    }
    fn le(&self, lhs: &Tensor, rhs: &Tensor) -> Result<Tensor, BackendError> {
        cpu_backend().le(&host(lhs)?, &host(rhs)?).map(tag_metal)
    }
    fn gt(&self, lhs: &Tensor, rhs: &Tensor) -> Result<Tensor, BackendError> {
        cpu_backend().gt(&host(lhs)?, &host(rhs)?).map(tag_metal)
    }
    fn ge(&self, lhs: &Tensor, rhs: &Tensor) -> Result<Tensor, BackendError> {
        cpu_backend().ge(&host(lhs)?, &host(rhs)?).map(tag_metal)
    }
    fn cast(
        &self,
        src: &Tensor,
        target: rustorch_core::tensor::dtype::Dtype,
    ) -> Result<Tensor, BackendError> {
        cpu_backend().cast(&host(src)?, target).map(tag_metal)
    }
}
