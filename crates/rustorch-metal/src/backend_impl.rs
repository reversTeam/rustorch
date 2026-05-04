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
    abs_f32, add_f32, div_f32, exp_f32, log_f32, matmul_simdgroup_f32, mean_f32, mul_f32, neg_f32,
    relu_f32, sigmoid_f32, silu_f32, sqrt_f32, sub_f32, sum_f32, tanh_f32,
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
        if lhs.shape() != rhs.shape() {
            return cpu_backend().add(&host(lhs)?, &host(rhs)?).map(tag_metal);
        }
        binary_native(self, "add", lhs, rhs, add_f32)
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
        let out =
            matmul_simdgroup_f32(self, &l, &r, m, k1, n).map_err(|e| metal_err("matmul", e))?;
        let core = rustorch_core::tensor::storage::MetalStorage::standalone(out, m * n * 4);
        Ok(Tensor::from_metal_storage(core, vec![m, n], lhs.dtype()))
    }

    // -------- CPU-fallback overrides for the remaining trait surface --------
    //
    // Each CPU fallback materialises the Metal tensor to host, runs
    // on `cpu_backend()`, and re-tags as `Device::Metal`. Subsequent
    // Task J commits replace these with native Metal kernels.

    fn sub(&self, lhs: &Tensor, rhs: &Tensor) -> Result<Tensor, BackendError> {
        if lhs.shape() != rhs.shape() {
            return cpu_backend().sub(&host(lhs)?, &host(rhs)?).map(tag_metal);
        }
        binary_native(self, "sub", lhs, rhs, sub_f32)
    }
    fn mul(&self, lhs: &Tensor, rhs: &Tensor) -> Result<Tensor, BackendError> {
        if lhs.shape() != rhs.shape() {
            return cpu_backend().mul(&host(lhs)?, &host(rhs)?).map(tag_metal);
        }
        binary_native(self, "mul", lhs, rhs, mul_f32)
    }
    fn div(&self, lhs: &Tensor, rhs: &Tensor) -> Result<Tensor, BackendError> {
        if lhs.shape() != rhs.shape() {
            return cpu_backend().div(&host(lhs)?, &host(rhs)?).map(tag_metal);
        }
        binary_native(self, "div", lhs, rhs, div_f32)
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
        cpu_backend()
            .add_bias(&host(x)?, &host(bias)?)
            .map(tag_metal)
    }
    fn reshape(&self, src: &Tensor, shape: &[usize]) -> Result<Tensor, BackendError> {
        cpu_backend().reshape(&host(src)?, shape).map(tag_metal)
    }
    fn sum_dim(&self, src: &Tensor, dims: &[usize], keepdim: bool) -> Result<Tensor, BackendError> {
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
        cpu_backend()
            .mean_dim(&host(src)?, dims, keepdim)
            .map(tag_metal)
    }
    fn unbroadcast_to(
        &self,
        grad: &Tensor,
        target_shape: &[usize],
    ) -> Result<Tensor, BackendError> {
        cpu_backend()
            .unbroadcast_to(&host(grad)?, target_shape)
            .map(tag_metal)
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
        // Native composition stays on GPU end-to-end: sub → mul → reduce.
        // Mirror of the WGSL mse_loss native composition.
        use rustorch_cpu::backend::Reduction;
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
