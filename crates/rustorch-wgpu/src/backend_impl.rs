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
use crate::transfer::{to_cpu, to_gpu};
use rustorch_core::tensor::device::Device;
use rustorch_core::tensor::tensor_impl::Tensor;
use rustorch_cpu::backend::Backend;
use rustorch_cpu::cpu_backend::cpu_backend;
use rustorch_cpu::error::BackendError;

/// Tag a Tensor as living on the Wgpu device.
///
/// With Storage Option B (transitional, P3.Y plan), the actual storage
/// is the CPU shadow. Tagging the device-of-record as Wgpu makes
/// follow-on autograd ops keep dispatching through `wgpu_backend()` —
/// without it, `to_cpu` returns a CPU-tagged tensor and the next op
/// would route to `cpu_backend()`, which then mismatches against any
/// Wgpu operands and trips `DeviceMismatch`. Future Storage Option A
/// (Tensor enum Cpu | Wgpu) eliminates this round-trip + retag pattern.
fn tag_wgpu(t: Tensor) -> Tensor {
    t.with_device(Device::Wgpu)
}

/// Map `WgpuError` → `BackendError::NumericalError(...)`. The trait expects
/// the latter (string-based) so device-specific errors are surfaced
/// without leaking wgpu types into the trait surface.
fn wgpu_err(op: &'static str, err: crate::error::WgpuError) -> BackendError {
    BackendError::NumericalError(format!("{op}: wgpu: {err}"))
}

/// Round-trip helper: upload a Tensor, run a unary kernel, download.
///
/// Used by the elementary unary ops (relu/sigmoid/tanh/silu/neg) which
/// share the dispatch_unary signature.
fn unary_roundtrip(
    backend: &WgpuBackend,
    op_name: &'static str,
    src: &Tensor,
) -> Result<Tensor, BackendError> {
    let inp = to_gpu(backend, src).map_err(|e| wgpu_err(op_name, e))?;
    let out = dispatch_unary(backend, op_name, &inp).map_err(|e| wgpu_err(op_name, e))?;
    let t = to_cpu(backend, &out, src.shape().to_vec()).map_err(|e| wgpu_err(op_name, e))?;
    Ok(tag_wgpu(t))
}

/// Round-trip helper: upload two Tensors, run a binary kernel, download.
///
/// Uses `dispatch_binary_broadcast` so element-wise broadcasting (matching
/// the CPU semantics in `dispatch_binary` of `cpu_backend.rs`) works
/// out of the box.
fn binary_roundtrip(
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
    let t = to_cpu(backend, &out, out_shape).map_err(|e| wgpu_err(op_name, e))?;
    Ok(tag_wgpu(t))
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
    f(cpu_backend(), lhs, rhs)
}

impl Backend for WgpuBackend {
    fn name(&self) -> &'static str {
        "wgpu"
    }

    // -------------------- Required methods --------------------

    fn add(&self, lhs: &Tensor, rhs: &Tensor) -> Result<Tensor, BackendError> {
        binary_roundtrip(self, "add", lhs, rhs)
    }

    fn sub(&self, lhs: &Tensor, rhs: &Tensor) -> Result<Tensor, BackendError> {
        binary_roundtrip(self, "sub", lhs, rhs)
    }

    fn mul(&self, lhs: &Tensor, rhs: &Tensor) -> Result<Tensor, BackendError> {
        binary_roundtrip(self, "mul", lhs, rhs)
    }

    fn div(&self, lhs: &Tensor, rhs: &Tensor) -> Result<Tensor, BackendError> {
        binary_roundtrip(self, "div", lhs, rhs)
    }

    fn neg(&self, src: &Tensor) -> Result<Tensor, BackendError> {
        unary_roundtrip(self, "neg", src)
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
        let t = to_cpu(self, &out, vec![m, n]).map_err(|e| wgpu_err("matmul", e))?;
        Ok(tag_wgpu(t))
    }

    fn relu(&self, src: &Tensor) -> Result<Tensor, BackendError> {
        unary_roundtrip(self, "relu", src)
    }

    fn eq(&self, lhs: &Tensor, rhs: &Tensor) -> Result<Tensor, BackendError> {
        // No wgpu kernel for comparison ops yet — fall through to CPU.
        // Acceptable since `eq` is rare in the autograd hot path; future
        // wgpu kernels can override this with a native dispatch.
        cpu_fallback_binary("eq", lhs, rhs, |b, l, r| b.eq(l, r))
    }

    // -------------------- Optional methods with native wgpu kernels --------------------

    fn sigmoid(&self, src: &Tensor) -> Result<Tensor, BackendError> {
        unary_roundtrip(self, "sigmoid", src)
    }

    fn tanh(&self, src: &Tensor) -> Result<Tensor, BackendError> {
        unary_roundtrip(self, "tanh", src)
    }

    fn silu(&self, src: &Tensor) -> Result<Tensor, BackendError> {
        unary_roundtrip(self, "silu", src)
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
        let t = to_cpu(self, &out, vec![1]).map_err(|e| wgpu_err("sum", e))?;
        Ok(tag_wgpu(t))
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
        let t = to_cpu(self, &out, vec![1]).map_err(|e| wgpu_err("mean", e))?;
        Ok(tag_wgpu(t))
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
        // Round-trip + dispatch_binary_broadcast: lhs[B, N] + rhs[N] → [B, N].
        let x_g = to_gpu(self, x).map_err(|e| wgpu_err("add_bias", e))?;
        let bias_g = to_gpu(self, bias).map_err(|e| wgpu_err("add_bias", e))?;
        let (out, out_shape) =
            dispatch_binary_broadcast(self, "add", &x_g, x.shape(), &bias_g, bias.shape())
                .map_err(|e| wgpu_err("add_bias", e))?;
        drop((x_g, bias_g));
        let t = to_cpu(self, &out, out_shape).map_err(|e| wgpu_err("add_bias", e))?;
        Ok(tag_wgpu(t))
    }

    // Other optional methods (abs, sqrt, exp, log, sum, mean, softmax,
    // log_softmax, transpose, reshape, bmm, unbroadcast_to, softmax_grad,
    // gather, scatter, conv2d, …) inherit the default `UnsupportedOp` impl
    // from the trait. Phases B2+ and C2+ override them progressively as
    // kernels are written.
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
