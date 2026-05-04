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
use crate::kernels::{add_f32, matmul_simdgroup_f32};
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

impl Backend for MetalBackend {
    fn name(&self) -> &'static str {
        "metal"
    }

    // -------- Native Metal kernels (Storage::Metal in/out) ----------

    /// Element-wise add via the native Metal `add_f32` kernel.
    fn add(&self, lhs: &Tensor, rhs: &Tensor) -> Result<Tensor, BackendError> {
        if lhs.shape() != rhs.shape() {
            // Shape broadcasting not yet implemented in the native
            // kernel — fall through to CPU which handles it.
            return cpu_backend().add(&host(lhs)?, &host(rhs)?).map(tag_metal);
        }
        let l = to_gpu(self, lhs).map_err(|e| metal_err("add", e))?;
        let r = to_gpu(self, rhs).map_err(|e| metal_err("add", e))?;
        let out = add_f32(self, &l, &r, lhs.numel()).map_err(|e| metal_err("add", e))?;
        let core = rustorch_core::tensor::storage::MetalStorage::standalone(out, lhs.numel() * 4);
        Ok(Tensor::from_metal_storage(
            core,
            lhs.shape().to_vec(),
            lhs.dtype(),
        ))
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
        cpu_backend().sub(&host(lhs)?, &host(rhs)?).map(tag_metal)
    }
    fn mul(&self, lhs: &Tensor, rhs: &Tensor) -> Result<Tensor, BackendError> {
        cpu_backend().mul(&host(lhs)?, &host(rhs)?).map(tag_metal)
    }
    fn div(&self, lhs: &Tensor, rhs: &Tensor) -> Result<Tensor, BackendError> {
        cpu_backend().div(&host(lhs)?, &host(rhs)?).map(tag_metal)
    }
    fn neg(&self, src: &Tensor) -> Result<Tensor, BackendError> {
        cpu_backend().neg(&host(src)?).map(tag_metal)
    }
    fn relu(&self, src: &Tensor) -> Result<Tensor, BackendError> {
        cpu_backend().relu(&host(src)?).map(tag_metal)
    }
    fn eq(&self, lhs: &Tensor, rhs: &Tensor) -> Result<Tensor, BackendError> {
        cpu_backend().eq(&host(lhs)?, &host(rhs)?).map(tag_metal)
    }
    fn sigmoid(&self, src: &Tensor) -> Result<Tensor, BackendError> {
        cpu_backend().sigmoid(&host(src)?).map(tag_metal)
    }
    fn tanh(&self, src: &Tensor) -> Result<Tensor, BackendError> {
        cpu_backend().tanh(&host(src)?).map(tag_metal)
    }
    fn silu(&self, src: &Tensor) -> Result<Tensor, BackendError> {
        cpu_backend().silu(&host(src)?).map(tag_metal)
    }
    fn sum(&self, src: &Tensor) -> Result<Tensor, BackendError> {
        cpu_backend().sum(&host(src)?).map(tag_metal)
    }
    fn mean(&self, src: &Tensor) -> Result<Tensor, BackendError> {
        cpu_backend().mean(&host(src)?).map(tag_metal)
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
        cpu_backend().abs(&host(src)?).map(tag_metal)
    }
    fn sqrt(&self, src: &Tensor) -> Result<Tensor, BackendError> {
        cpu_backend().sqrt(&host(src)?).map(tag_metal)
    }
    fn exp(&self, src: &Tensor) -> Result<Tensor, BackendError> {
        cpu_backend().exp(&host(src)?).map(tag_metal)
    }
    fn log(&self, src: &Tensor) -> Result<Tensor, BackendError> {
        cpu_backend().log(&host(src)?).map(tag_metal)
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
        cpu_backend()
            .mse_loss(&host(input)?, &host(target)?, reduction)
            .map(tag_metal)
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
