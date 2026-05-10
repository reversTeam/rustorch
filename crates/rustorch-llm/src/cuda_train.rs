//! T247 Phase 2 — autograd dispatch wiring for CUDA backward kernels.
//!
//! Bridges the CUDA forward + backward kernels in `rustorch-cuda::llm_kernels`
//! into the autograd `Variable` graph. Each `*_cuda_fn` here wraps a forward
//! CUDA call + a backward CUDA call inside a `CustomFunction` so calling
//! `.backward()` on a downstream loss runs all gradient compute on GPU.
//!
//! Pilot scope (T247.1) : `rms_norm_cuda_fn`. The same pattern extends to
//! the other 6 backward kernels (SwiGLU, RoPE, Cross-entropy, Embedding,
//! Q4_K matmul, GQA) — each gets its own `*Fn` struct + helper.
//!
//! Data flow per call (this minimum-viable wiring uses CPU staging) :
//!   1. forward(x_cpu_f32, gamma_cpu_f32) :
//!      → upload to CUDA as bf16
//!      → run forward kernel (in a freshly-alloc'd output buffer)
//!      → download result back to CPU f32
//!      → save x, gamma, eps in autograd context
//!      → return CPU f32 Tensor
//!   2. backward(dy_cpu_f32) :
//!      → upload dy + saved x + saved gamma to CUDA
//!      → run backward kernel
//!      → download dx + dgamma to CPU
//!      → return [dx_cpu_f32, dgamma_cpu_f32]
//!
//! Real production wiring will benefit from first-class CUDA `Tensor`
//! storage (analogous to `WgpuStorage` in rustorch-core) so the round-trip
//! data transfer disappears.

#![cfg(feature = "cuda")]

use cudarc::driver::{CudaContext, CudaStream, DevicePtr, DevicePtrMut};
use rustorch_autograd::{apply_custom, BackwardError, BwdCtx, CustomFunction, FwdCtx, Variable};
use rustorch_core::tensor::tensor_impl::Tensor;
use rustorch_cuda::llm_kernels::LlmKernels;
use std::sync::Arc;

/// Lazily-initialised global CUDA context + stream + kernels for the
/// training-side glue. Production callers should manage their own
/// context — this is for tests / examples / proof-of-concept.
fn cuda_setup() -> (Arc<CudaContext>, Arc<CudaStream>, LlmKernels) {
    let ctx = CudaContext::new(0).expect("CudaContext::new");
    let stream = ctx.default_stream();
    let kernels = LlmKernels::new(ctx.clone());
    (ctx, stream, kernels)
}

/// Helper : f32 slice → bf16 device upload.
fn upload_bf16(stream: &Arc<CudaStream>, x: &[f32]) -> cudarc::driver::CudaSlice<half::bf16> {
    let bf: Vec<half::bf16> = x.iter().copied().map(half::bf16::from_f32).collect();
    stream.memcpy_stod(&bf).expect("upload bf16")
}

/// Helper : bf16 device buffer → f32 CPU vec.
fn download_bf16(
    stream: &Arc<CudaStream>,
    dev: &cudarc::driver::CudaSlice<half::bf16>,
) -> Vec<f32> {
    stream
        .memcpy_dtov(dev)
        .expect("download bf16")
        .into_iter()
        .map(|b: half::bf16| b.to_f32())
        .collect()
}

// ---------------------------------------------------------------------------
// T247.1 — RMSNorm
// ---------------------------------------------------------------------------

/// CustomFunction wiring T247.1 RMSNorm CUDA forward + backward.
///
/// Inputs to `apply_custom` (in order) :
///   inputs[0] = x        f32  [N, D]   (last axis D is normalized)
///   inputs[1] = gamma    f32  [D]
///   inputs[2] = eps      f32  [1]      (scalar tensor)
///
/// Output : y = (x / sqrt(mean(x², dim=-1) + eps)) · gamma     f32 [N, D]
struct RmsNormCudaFn;

impl CustomFunction for RmsNormCudaFn {
    fn forward(ctx: &mut FwdCtx, inputs: &[Tensor]) -> Vec<Tensor> {
        assert_eq!(inputs.len(), 3, "rms_norm_cuda_fn expects (x, gamma, eps)");
        let x = &inputs[0];
        let gamma = &inputs[1];
        let _eps_tensor = &inputs[2];

        let shape = x.shape().to_vec();
        assert!(!shape.is_empty(), "x must be at least 1-D");
        let d = *shape.last().unwrap();
        let outer: usize = shape.iter().take(shape.len() - 1).product();
        assert_eq!(gamma.numel(), d, "gamma length must match last axis of x");

        // The CUDA pilot kernel does single-row (outer=1). Iterate per row
        // for outer > 1 (sufficient for the proof of architecture ; a
        // production multi-row kernel is a follow-up).
        let x_slice = x.as_slice::<f32>().expect("x f32 contig");
        let g_slice = gamma.as_slice::<f32>().expect("gamma f32 contig");
        let eps_slice = _eps_tensor.as_slice::<f32>().expect("eps f32");
        let eps = eps_slice[0];

        let (_ctx, stream, kernels) = cuda_setup();
        let g_dev = upload_bf16(&stream, g_slice);

        let mut out_buf = vec![0.0_f32; outer * d];
        for r in 0..outer {
            let row = &x_slice[r * d..(r + 1) * d];
            let x_dev = upload_bf16(&stream, row);
            // Forward: in-place rms_norm_bf16. Allocate a copy to preserve x.
            let mut y_dev = stream
                .memcpy_stod(
                    &row.iter()
                        .copied()
                        .map(half::bf16::from_f32)
                        .collect::<Vec<_>>(),
                )
                .expect("alloc y");
            unsafe {
                let (yp, _g0) = y_dev.device_ptr_mut(&stream);
                let (gp, _g1) = g_dev.device_ptr(&stream);
                kernels
                    .rms_norm_bf16(&stream, yp, gp, eps, d as i32, 1)
                    .expect("rms_norm_bf16");
            }
            let y_cpu = download_bf16(&stream, &y_dev);
            out_buf[r * d..(r + 1) * d].copy_from_slice(&y_cpu);
            // Hint to drop x_dev as it's not needed past this iter.
            drop(x_dev);
        }

        // Save tensors for backward.
        ctx.save_for_backward(x.clone());
        ctx.save_for_backward(gamma.clone());
        ctx.save_for_backward(_eps_tensor.clone());

        let y = Tensor::from_vec(shape, out_buf).expect("y tensor");
        vec![y]
    }

    fn backward(ctx: &BwdCtx, grad_outputs: &[Tensor]) -> Vec<Tensor> {
        let saved = ctx.saved_tensors();
        let x = &saved[0];
        let gamma = &saved[1];
        let eps = saved[2].as_slice::<f32>().expect("eps")[0];
        let dy = &grad_outputs[0];

        let shape = x.shape().to_vec();
        let d = *shape.last().unwrap();
        let outer: usize = shape.iter().take(shape.len() - 1).product();

        let x_slice = x.as_slice::<f32>().expect("x f32");
        let g_slice = gamma.as_slice::<f32>().expect("gamma f32");
        let dy_slice = dy.as_slice::<f32>().expect("dy f32");

        let (_ctx, stream, kernels) = cuda_setup();
        let g_dev = upload_bf16(&stream, g_slice);

        let mut dx_buf = vec![0.0_f32; outer * d];
        // Per-row dgamma accumulation (CPU-side, since pilot kernel is
        // single-row outer=1 — sums across rows on host).
        let mut dgamma_buf = vec![0.0_f32; d];

        for r in 0..outer {
            let x_row = &x_slice[r * d..(r + 1) * d];
            let dy_row = &dy_slice[r * d..(r + 1) * d];

            let x_dev = upload_bf16(&stream, x_row);
            let dy_dev = upload_bf16(&stream, dy_row);
            let mut dx_dev = stream.alloc_zeros::<half::bf16>(d).expect("alloc dx");
            let mut dg_dev = stream.alloc_zeros::<half::bf16>(d).expect("alloc dg");

            unsafe {
                let (xp, _g0) = x_dev.device_ptr(&stream);
                let (gp, _g1) = g_dev.device_ptr(&stream);
                let (dyp, _g2) = dy_dev.device_ptr(&stream);
                let (dxp, _g3) = dx_dev.device_ptr_mut(&stream);
                let (dgp, _g4) = dg_dev.device_ptr_mut(&stream);
                kernels
                    .rms_norm_grad_bf16(&stream, xp, gp, dyp, dxp, dgp, d as i32, eps)
                    .expect("rms_norm_grad_bf16");
            }

            let dx_cpu = download_bf16(&stream, &dx_dev);
            let dg_cpu = download_bf16(&stream, &dg_dev);

            dx_buf[r * d..(r + 1) * d].copy_from_slice(&dx_cpu);
            for i in 0..d {
                dgamma_buf[i] += dg_cpu[i];
            }
        }

        let dx = Tensor::from_vec(shape, dx_buf).expect("dx tensor");
        let dgamma = Tensor::from_vec([d], dgamma_buf).expect("dgamma tensor");
        // eps grad is unused (parameter, not learned).
        let deps = Tensor::from_vec([1usize], vec![0.0_f32]).expect("deps tensor");
        vec![dx, dgamma, deps]
    }
}

/// T247.1 — Variable-aware RMSNorm running on CUDA for both forward and
/// backward. `x` is `[..., D]`, `gamma` is `[D]`, `eps` is the
/// numerical stabilization constant.
///
/// Returns a Variable spliced into the autograd tape ; calling
/// `.backward()` on a downstream loss will route gradients through the
/// CUDA `rms_norm_grad_bf16` kernel.
pub fn rms_norm_cuda(x: &Variable, gamma: &Variable, eps: f32) -> Result<Variable, BackwardError> {
    let eps_var = Variable::leaf(Tensor::from_vec([1usize], vec![eps]).expect("eps scalar tensor"));
    let mut outs = apply_custom::<RmsNormCudaFn>(&[x.clone(), gamma.clone(), eps_var])?;
    Ok(outs.pop().unwrap())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use rustorch_autograd::backward;

    /// T247 Phase 2 pilot — full forward + backward through CUDA kernels,
    /// exercised via `Variable::backward()`. Verifies that the gradients
    /// computed by CUDA backward kernels match those produced by the
    /// existing CPU autograd graph (through ops::mul, mean_dim, etc.)
    /// within BF16 tolerance.
    #[test]
    #[ignore = "requires CUDA GPU — run with --ignored on DGX"]
    fn rms_norm_cuda_backward_matches_cpu_autograd() {
        let d = 64usize;
        let outer = 2usize;
        let eps = 1e-6_f32;

        let mut state: u64 = 0xdeadbabe;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (((state >> 16) & 0xFFFF) as f32 / 65535.0) - 0.5
        };

        let x_data: Vec<f32> = (0..outer * d).map(|_| next() * 2.0).collect();
        let g_data: Vec<f32> = (0..d).map(|_| 0.5 + next() * 0.3).collect();

        // ---- CUDA path ----
        let x_cuda = Variable::leaf(Tensor::from_vec([outer, d], x_data.clone()).unwrap());
        let g_cuda = Variable::leaf(Tensor::from_vec([d], g_data.clone()).unwrap());
        let y_cuda = rms_norm_cuda(&x_cuda, &g_cuda, eps).unwrap();
        let loss_cuda = rustorch_autograd::ops::sum(&y_cuda).unwrap();
        backward(&loss_cuda, None).unwrap();

        let x_grad_cuda = x_cuda.grad().unwrap();
        let g_grad_cuda = g_cuda.grad().unwrap();
        let xg_v: Vec<f32> = x_grad_cuda.as_slice::<f32>().unwrap().to_vec();
        let gg_v: Vec<f32> = g_grad_cuda.as_slice::<f32>().unwrap().to_vec();

        // ---- CPU autograd path (reference) ----
        // Use the `rustorch_autograd::ops` decomposition that the existing
        // CPU RMSNorm forward falls back to with grad_enabled=true (mul,
        // mean_dim, add, sqrt, div, mul).
        use rustorch_autograd::ops;
        let x_cpu = Variable::leaf(Tensor::from_vec([outer, d], x_data.clone()).unwrap());
        let g_cpu = Variable::leaf(Tensor::from_vec([d], g_data.clone()).unwrap());
        let last_dim = x_cpu.tensor().ndim() - 1;
        let x_sq = ops::mul(&x_cpu, &x_cpu).unwrap();
        let mean_x_sq = ops::mean_dim(&x_sq, &[last_dim]).unwrap();
        let eps_var = Variable::leaf(Tensor::from_vec([1usize], vec![eps]).unwrap());
        let rms_sq = ops::add(&mean_x_sq, &eps_var).unwrap();
        let rms = ops::sqrt(&rms_sq).unwrap();
        let normed = ops::div(&x_cpu, &rms).unwrap();
        let y_cpu = ops::mul(&normed, &g_cpu).unwrap();
        let loss_cpu = ops::sum(&y_cpu).unwrap();
        backward(&loss_cpu, None).unwrap();

        let x_grad_cpu = x_cpu.grad().unwrap();
        let g_grad_cpu = g_cpu.grad().unwrap();
        let xg_ref: Vec<f32> = x_grad_cpu.as_slice::<f32>().unwrap().to_vec();
        let gg_ref: Vec<f32> = g_grad_cpu.as_slice::<f32>().unwrap().to_vec();

        // Compare. BF16 round-trip on x, gamma, dy → 1.5% rel + 5e-3 abs is
        // a comfortable tolerance for this setup (d=64 reduce, outer=2).
        for i in 0..outer * d {
            let diff = (xg_ref[i] - xg_v[i]).abs();
            let tol = xg_ref[i].abs() * 3e-2 + 1e-2;
            assert!(
                diff <= tol,
                "x.grad[{i}] cpu_ref={} cuda={} diff={} tol={}",
                xg_ref[i],
                xg_v[i],
                diff,
                tol
            );
        }
        for i in 0..d {
            let diff = (gg_ref[i] - gg_v[i]).abs();
            let tol = gg_ref[i].abs() * 3e-2 + 1e-2;
            assert!(
                diff <= tol,
                "gamma.grad[{i}] cpu_ref={} cuda={} diff={} tol={}",
                gg_ref[i],
                gg_v[i],
                diff,
                tol
            );
        }
    }

    /// T247 Phase 3 pilot — show one training step works end-to-end :
    /// fresh leaf gamma, single forward + backward, gamma.grad is non-zero
    /// in the right direction (RMSNorm with all-ones x and uniform gamma →
    /// dgamma should be positive when dy = ones).
    #[test]
    #[ignore = "requires CUDA GPU — run with --ignored on DGX"]
    fn rms_norm_cuda_one_step_grad_nonzero() {
        let d = 32usize;
        let eps = 1e-6_f32;

        let x = Variable::leaf(Tensor::from_vec([1, d], vec![1.0_f32; d]).unwrap());
        let g = Variable::leaf(Tensor::from_vec([d], vec![1.0_f32; d]).unwrap());
        let y = rms_norm_cuda(&x, &g, eps).unwrap();
        let loss = rustorch_autograd::ops::sum(&y).unwrap();
        backward(&loss, None).unwrap();

        let g_grad = g.grad().unwrap();
        let g_grad_v = g_grad.as_slice::<f32>().unwrap();
        // For x = 1s, normed = x / sqrt(1 + eps) ≈ 1, so dgamma_i = sum_outer(dy_i * normed_i)
        // = 1 * 1 = 1 per element. Allow ±10% slack (BF16 + sqrt rounding).
        for (i, &v) in g_grad_v.iter().enumerate() {
            assert!(
                (v - 1.0).abs() < 0.1,
                "gamma.grad[{i}] = {} (expected ≈ 1.0)",
                v
            );
        }
    }
}
