//! Fused AdamW optimizer step on Apple Metal — single dispatch.
//!
//! P3.Z Task J Phase 3 — mirrors the WGSL `fused_adamw` win
//! (10.50 → 2.35 ms/step on rustorch-wgpu) but on Metal direct.
//! One MTLComputeCommandEncoder dispatch reads `param_in`, `grad`,
//! `m`, `v` buffers and writes new `param_out` (fresh) plus updated
//! `m` / `v` (in-place). Zero host trips.
//!
//! Math (decoupled-WD AdamW, matching `rustorch_optim::adam::AdamW`):
//!
//! ```text
//!   m  ← β₁ · m + (1 − β₁) · g
//!   v  ← β₂ · v + (1 − β₂) · g²
//!   m̂ = m / (1 − β₁ᵗ)
//!   v̂ = v / (1 − β₂ᵗ)
//!   p ← p − lr · m̂ / (√v̂ + ε)
//!   p ← p − lr · wd · p_old        // decoupled WD (AdamW)
//! ```

use crate::backend::MetalBackend;
use crate::error::MetalError;
use metal::{Buffer, MTLSize};

/// Threadgroup size — 256 threads × N workgroups covers any param
/// size up to Metal's per-dim limit (typically 1024 max threads
/// per threadgroup; we choose 256 to leave room for occupancy).
const TG_SIZE: u64 = 256;

/// Hyperparameters laid out for the Metal `Params` constant buffer.
/// `repr(C)` so `bytemuck` produces the exact 32-byte payload the
/// shader expects.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct AdamWUniform {
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    bc1: f32, // 1 − β₁ᵗ
    bc2: f32, // 1 − β₂ᵗ
    n: u32,
}

unsafe impl bytemuck::Zeroable for AdamWUniform {}
unsafe impl bytemuck::Pod for AdamWUniform {}

/// Public hyperparameters; the kernel needs the bias-corrected
/// denominators precomputed (`1 − β^t`), which we do CPU-side.
#[derive(Debug, Clone, Copy)]
pub struct AdamWStepParams {
    /// Learning rate.
    pub lr: f32,
    /// β₁ — first-moment decay.
    pub beta1: f32,
    /// β₂ — second-moment decay.
    pub beta2: f32,
    /// ε denominator stabiliser.
    pub eps: f32,
    /// Decoupled weight decay (AdamW).
    pub weight_decay: f32,
    /// Step counter `t` (one-based).
    pub t: u32,
}

impl AdamWStepParams {
    fn into_uniform(self, n: u32) -> AdamWUniform {
        let bc1 = 1.0 - self.beta1.powi(self.t as i32);
        let bc2 = 1.0 - self.beta2.powi(self.t as i32);
        AdamWUniform {
            lr: self.lr,
            beta1: self.beta1,
            beta2: self.beta2,
            eps: self.eps,
            weight_decay: self.weight_decay,
            bc1,
            bc2,
            n,
        }
    }
}

const SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct Params {
    float lr;
    float beta1;
    float beta2;
    float eps;
    float weight_decay;
    float bc1;
    float bc2;
    uint  n;
};

kernel void fused_adamw_f32(
    constant Params&    params    [[buffer(0)]],
    device const float* param_in  [[buffer(1)]],
    device const float* grad      [[buffer(2)]],
    device       float* m         [[buffer(3)]],
    device       float* v         [[buffer(4)]],
    device       float* param_out [[buffer(5)]],
    uint                gid       [[thread_position_in_grid]]
) {
    if (gid >= params.n) { return; }

    float p_old = param_in[gid];
    float g = grad[gid];

    float new_m = params.beta1 * m[gid] + (1.0f - params.beta1) * g;
    m[gid] = new_m;

    float new_v = params.beta2 * v[gid] + (1.0f - params.beta2) * g * g;
    v[gid] = new_v;

    float m_hat = new_m / params.bc1;
    float v_hat = new_v / params.bc2;

    float p_new = p_old - params.lr * m_hat / (sqrt(v_hat) + params.eps);
    if (params.weight_decay > 0.0f) {
        p_new = p_new - params.lr * params.weight_decay * p_old;
    }
    param_out[gid] = p_new;
}

// In-place variant: same math, single param buffer that is read then
// written. Each thread reads p_old once (line 1) and writes p_new
// once (last line) — no aliasing hazard for `gid != gid'`.
kernel void fused_adamw_f32_inplace(
    constant Params&    params [[buffer(0)]],
    device       float* param  [[buffer(1)]],
    device const float* grad   [[buffer(2)]],
    device       float* m      [[buffer(3)]],
    device       float* v      [[buffer(4)]],
    uint                gid    [[thread_position_in_grid]]
) {
    if (gid >= params.n) { return; }

    float p_old = param[gid];
    float g = grad[gid];

    float new_m = params.beta1 * m[gid] + (1.0f - params.beta1) * g;
    m[gid] = new_m;

    float new_v = params.beta2 * v[gid] + (1.0f - params.beta2) * g * g;
    v[gid] = new_v;

    float m_hat = new_m / params.bc1;
    float v_hat = new_v / params.bc2;

    float p_new = p_old - params.lr * m_hat / (sqrt(v_hat) + params.eps);
    if (params.weight_decay > 0.0f) {
        p_new = p_new - params.lr * params.weight_decay * p_old;
    }
    param[gid] = p_new;
}
"#;

/// Run one AdamW step on GPU memory — single MTLComputeCommandEncoder
/// dispatch. Returns a fresh `param_out` buffer; `m` / `v` are
/// mutated in-place (their underlying `metal::Buffer`s are updated
/// by the kernel and survive across step() calls).
pub fn fused_adamw_step(
    backend: &MetalBackend,
    param_in: &Buffer,
    grad: &Buffer,
    m: &Buffer,
    v: &Buffer,
    n: usize,
    params: AdamWStepParams,
) -> Result<Buffer, MetalError> {
    let n_bytes = n * 4;
    if param_in.length() < n_bytes as u64
        || grad.length() < n_bytes as u64
        || m.length() < n_bytes as u64
        || v.length() < n_bytes as u64
    {
        return Err(MetalError::ShapeMismatch(format!(
            "fused_adamw: buffer too small for n={n} (need {n_bytes} bytes each)"
        )));
    }

    let pipeline = backend.pipeline("fused_adamw_f32", SHADER, "fused_adamw_f32")?;

    let param_out = backend.alloc_shared(n_bytes)?;

    // Build the uniform on a shared-mode buffer.
    let uniform = params.into_uniform(n as u32);
    let uniform_buf = backend.alloc_shared(core::mem::size_of::<AdamWUniform>())?;
    // SAFETY: shared-storage buffer; `contents()` is host-mapped.
    unsafe {
        let dst = uniform_buf.contents() as *mut AdamWUniform;
        *dst = uniform;
    }

    backend.with_encoder(|encoder| {
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(&uniform_buf), 0);
        encoder.set_buffer(1, Some(param_in), 0);
        encoder.set_buffer(2, Some(grad), 0);
        encoder.set_buffer(3, Some(m), 0);
        encoder.set_buffer(4, Some(v), 0);
        encoder.set_buffer(5, Some(&param_out), 0);

        let max_threads = pipeline.max_total_threads_per_threadgroup();
        let tg = MTLSize::new(TG_SIZE.min(max_threads), 1, 1);
        let grid = MTLSize::new(n as u64, 1, 1);
        encoder.dispatch_threads(grid, tg);
    });
    // wait_until_completed removed — Metal handles inter-kernel sync via queue order. Only host reads (in transfer.rs::tensor_to_cpu) need an explicit wait.

    Ok(param_out)
}

/// In-place AdamW step. Mutates `param` directly instead of allocating
/// a fresh `param_out`. Saves one `alloc_shared(n*4)` per param per
/// step (≈4 MB for a 1024² Linear weight matrix), which adds up across
/// many params and many steps. Caller keeps the same `param` buffer
/// across steps — `param.set_data()` is no longer needed.
///
/// `m`, `v` are still updated in place (same as the non-inplace path).
pub fn fused_adamw_step_inplace(
    backend: &MetalBackend,
    param: &Buffer,
    grad: &Buffer,
    m: &Buffer,
    v: &Buffer,
    n: usize,
    params: AdamWStepParams,
) -> Result<(), MetalError> {
    let n_bytes = n * 4;
    if param.length() < n_bytes as u64
        || grad.length() < n_bytes as u64
        || m.length() < n_bytes as u64
        || v.length() < n_bytes as u64
    {
        return Err(MetalError::ShapeMismatch(format!(
            "fused_adamw_inplace: buffer too small for n={n} (need {n_bytes} bytes each)"
        )));
    }
    let pipeline =
        backend.pipeline("fused_adamw_f32_inplace", SHADER, "fused_adamw_f32_inplace")?;

    // Pass `Params` uniform inline via set_bytes — saves an alloc_shared
    // per param per step (≈5–10 µs × 2 params × N steps).
    let uniform = params.into_uniform(n as u32);
    backend.with_encoder(|encoder| {
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_bytes(
            0,
            core::mem::size_of::<AdamWUniform>() as u64,
            &uniform as *const AdamWUniform as *const std::ffi::c_void,
        );
        encoder.set_buffer(1, Some(param), 0);
        encoder.set_buffer(2, Some(grad), 0);
        encoder.set_buffer(3, Some(m), 0);
        encoder.set_buffer(4, Some(v), 0);

        let max_threads = pipeline.max_total_threads_per_threadgroup();
        let tg = MTLSize::new(TG_SIZE.min(max_threads), 1, 1);
        let grid = MTLSize::new(n as u64, 1, 1);
        encoder.dispatch_threads(grid, tg);
    });
    Ok(())
}

/// Allocate a zero-initialised shared GPU buffer for fresh `m` / `v`
/// state on the first GPU AdamW step.
pub fn allocate_zeros(backend: &MetalBackend, numel: usize) -> Result<Buffer, MetalError> {
    let bytes = numel * 4;
    let buf = backend.alloc_shared(bytes)?;
    // SAFETY: shared-storage buffer, host pointer valid for `bytes` bytes.
    unsafe {
        let p = buf.contents() as *mut u8;
        std::ptr::write_bytes(p, 0, bytes);
    }
    Ok(buf)
}
