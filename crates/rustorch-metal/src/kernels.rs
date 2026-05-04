//! Metal compute kernels — first concrete dispatch path for Task J.
//!
//! Source `.metal` shaders are inlined as Rust string constants and
//! compiled at runtime via `Device::new_library_with_source`. This
//! avoids a `build.rs` xcrun-metal step in the early scaffolding
//! phase — once the kernel set stabilises (matmul, fused, attention)
//! we'll switch to a `.metallib` ahead-of-time pipeline for faster
//! cold-start.

use crate::backend::MetalBackend;
use crate::error::MetalError;
use metal::{Buffer, MTLSize};

/// Element-wise add: `out[i] = lhs[i] + rhs[i]` for `i ∈ [0, n)`.
///
/// Smoke-test kernel — validates the full dispatch pipeline:
/// shader compile, pipeline state, command buffer, compute encoder,
/// threadgroup dispatch, command buffer commit + waitUntilCompleted.
/// Same shape constraints as the WGPU `dispatch_binary("add", ...)`.
const ADD_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void add_f32(
    device const float* lhs [[buffer(0)]],
    device const float* rhs [[buffer(1)]],
    device       float* out [[buffer(2)]],
    constant     uint&  n   [[buffer(3)]],
    uint                gid [[thread_position_in_grid]]
) {
    if (gid >= n) { return; }
    out[gid] = lhs[gid] + rhs[gid];
}
"#;

/// Element-wise add of two F32 buffers into a fresh output buffer.
///
/// Allocates the output via [`MetalBackend::alloc_shared`] so callers
/// can read it back from the host without an explicit blit. The
/// shader is compiled on first call and cached at the device level
/// — subsequent calls reuse the pipeline state.
pub fn add_f32(
    backend: &MetalBackend,
    lhs: &Buffer,
    rhs: &Buffer,
    n: usize,
) -> Result<Buffer, MetalError> {
    let n_bytes = n * core::mem::size_of::<f32>();
    if lhs.length() < n_bytes as u64 || rhs.length() < n_bytes as u64 {
        return Err(MetalError::ShapeMismatch(format!(
            "add_f32: lhs/rhs buffer length too small (n={n}, need {n_bytes} bytes)"
        )));
    }

    let pipeline = backend.pipeline("add_f32", ADD_SHADER, "add_f32")?;

    let out = backend.alloc_shared(n_bytes)?;

    // Stage `n` into a single-uint constant buffer.
    let n_buf = backend.alloc_shared(core::mem::size_of::<u32>())?;
    // SAFETY: shared-mode buffer; n_buf.contents() is host-mapped.
    unsafe {
        let p = n_buf.contents() as *mut u32;
        *p = n as u32;
    }

    backend.with_encoder(|encoder| {
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(lhs), 0);
        encoder.set_buffer(1, Some(rhs), 0);
        encoder.set_buffer(2, Some(&out), 0);
        encoder.set_buffer(3, Some(&n_buf), 0);

        // Threadgroup of 256 threads, grid sized to cover all N elements.
        let tg_size = MTLSize::new(256, 1, 1);
        let max_threads = pipeline.max_total_threads_per_threadgroup();
        let tg = MTLSize::new(tg_size.width.min(max_threads), 1, 1);
        let grid = MTLSize::new(n as u64, 1, 1);
        encoder.dispatch_threads(grid, tg);
    });
    // wait_until_completed removed — Metal handles inter-kernel sync via queue order. Only host reads (in transfer.rs::tensor_to_cpu) need an explicit wait.

    Ok(out)
}

// ----------------------------------------------------------------------
// Generic element-wise binary / unary / reduction kernels.
// Each picks an op via the `op_kind` constant rather than spinning up
// a separate compute pipeline per operation — keeps pipeline-cache
// pressure bounded.
// ----------------------------------------------------------------------

/// Element-wise binary kernel selector. Match the `BIN_*` constants
/// in [`BIN_SHADER`].
#[allow(dead_code)]
const BIN_KIND_ADD: u32 = 0;
const BIN_KIND_SUB: u32 = 1;
const BIN_KIND_MUL: u32 = 2;
const BIN_KIND_DIV: u32 = 3;

const BIN_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct BinParams {
    uint n;
    uint op_kind;
};

kernel void binary_f32(
    constant BinParams& params [[buffer(0)]],
    device const float* lhs    [[buffer(1)]],
    device const float* rhs    [[buffer(2)]],
    device       float* out    [[buffer(3)]],
    uint                gid    [[thread_position_in_grid]]
) {
    if (gid >= params.n) { return; }
    float a = lhs[gid];
    float b = rhs[gid];
    float r;
    if (params.op_kind == 0u)      { r = a + b; }
    else if (params.op_kind == 1u) { r = a - b; }
    else if (params.op_kind == 2u) { r = a * b; }
    else                            { r = a / b; }
    out[gid] = r;
}
"#;

/// Element-wise unary kernel selector. Match the `UN_*` constants
/// in [`UN_SHADER`].
const UN_KIND_NEG: u32 = 0;
const UN_KIND_RELU: u32 = 1;
const UN_KIND_SIGMOID: u32 = 2;
const UN_KIND_TANH: u32 = 3;
const UN_KIND_SILU: u32 = 4;
const UN_KIND_ABS: u32 = 5;
const UN_KIND_SQRT: u32 = 6;
const UN_KIND_EXP: u32 = 7;
const UN_KIND_LOG: u32 = 8;

const UN_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct UnParams {
    uint n;
    uint op_kind;
};

kernel void unary_f32(
    constant UnParams& params [[buffer(0)]],
    device const float* src    [[buffer(1)]],
    device       float* out    [[buffer(2)]],
    uint                gid    [[thread_position_in_grid]]
) {
    if (gid >= params.n) { return; }
    float x = src[gid];
    float r;
    if (params.op_kind == 0u)      { r = -x; }
    else if (params.op_kind == 1u) { r = max(0.0f, x); }
    else if (params.op_kind == 2u) { r = 1.0f / (1.0f + exp(-x)); }
    else if (params.op_kind == 3u) { r = tanh(x); }
    else if (params.op_kind == 4u) { r = x / (1.0f + exp(-x)); }      // SiLU = x * sigmoid(x)
    else if (params.op_kind == 5u) { r = fabs(x); }
    else if (params.op_kind == 6u) { r = sqrt(x); }
    else if (params.op_kind == 7u) { r = exp(x); }
    else                            { r = log(x); }
    out[gid] = r;
}
"#;

#[repr(C)]
#[derive(Copy, Clone, Default)]
struct BinUnParams {
    n: u32,
    op_kind: u32,
}
unsafe impl bytemuck::Zeroable for BinUnParams {}
unsafe impl bytemuck::Pod for BinUnParams {}

/// Dispatch a generic binary op. `op_kind` is one of the `BIN_KIND_*`
/// constants; `lhs` and `rhs` must be F32 buffers of `n` elements each.
fn dispatch_binary(
    backend: &MetalBackend,
    op_kind: u32,
    op_name: &'static str,
    lhs: &Buffer,
    rhs: &Buffer,
    n: usize,
) -> Result<Buffer, MetalError> {
    let n_bytes = n * 4;
    let pipeline = backend.pipeline(op_name, BIN_SHADER, "binary_f32")?;
    let out = backend.alloc_shared(n_bytes)?;
    let params = BinUnParams {
        n: n as u32,
        op_kind,
    };
    let params_buf = backend.alloc_shared(core::mem::size_of::<BinUnParams>())?;
    // SAFETY: shared-storage uniform.
    unsafe {
        let dst = params_buf.contents() as *mut BinUnParams;
        *dst = params;
    }
    backend.with_encoder(|encoder| {
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(&params_buf), 0);
        encoder.set_buffer(1, Some(lhs), 0);
        encoder.set_buffer(2, Some(rhs), 0);
        encoder.set_buffer(3, Some(&out), 0);
        let max_threads = pipeline.max_total_threads_per_threadgroup();
        let tg = MTLSize::new(256u64.min(max_threads), 1, 1);
        let grid = MTLSize::new(n as u64, 1, 1);
        encoder.dispatch_threads(grid, tg);
    });
    // wait_until_completed removed — Metal handles inter-kernel sync via queue order. Only host reads (in transfer.rs::tensor_to_cpu) need an explicit wait.
    Ok(out)
}

/// Dispatch a generic unary op.
fn dispatch_unary(
    backend: &MetalBackend,
    op_kind: u32,
    op_name: &'static str,
    src: &Buffer,
    n: usize,
) -> Result<Buffer, MetalError> {
    let n_bytes = n * 4;
    let pipeline = backend.pipeline(op_name, UN_SHADER, "unary_f32")?;
    let out = backend.alloc_shared(n_bytes)?;
    let params = BinUnParams {
        n: n as u32,
        op_kind,
    };
    let params_buf = backend.alloc_shared(core::mem::size_of::<BinUnParams>())?;
    unsafe {
        let dst = params_buf.contents() as *mut BinUnParams;
        *dst = params;
    }
    backend.with_encoder(|encoder| {
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(&params_buf), 0);
        encoder.set_buffer(1, Some(src), 0);
        encoder.set_buffer(2, Some(&out), 0);
        let max_threads = pipeline.max_total_threads_per_threadgroup();
        let tg = MTLSize::new(256u64.min(max_threads), 1, 1);
        let grid = MTLSize::new(n as u64, 1, 1);
        encoder.dispatch_threads(grid, tg);
    });
    // wait_until_completed removed — Metal handles inter-kernel sync via queue order. Only host reads (in transfer.rs::tensor_to_cpu) need an explicit wait.
    Ok(out)
}

/// Element-wise sub: `out = lhs - rhs`.
pub fn sub_f32(b: &MetalBackend, l: &Buffer, r: &Buffer, n: usize) -> Result<Buffer, MetalError> {
    dispatch_binary(b, BIN_KIND_SUB, "sub_f32", l, r, n)
}
/// Element-wise mul: `out = lhs * rhs`.
pub fn mul_f32(b: &MetalBackend, l: &Buffer, r: &Buffer, n: usize) -> Result<Buffer, MetalError> {
    dispatch_binary(b, BIN_KIND_MUL, "mul_f32", l, r, n)
}
/// Element-wise div: `out = lhs / rhs`.
pub fn div_f32(b: &MetalBackend, l: &Buffer, r: &Buffer, n: usize) -> Result<Buffer, MetalError> {
    dispatch_binary(b, BIN_KIND_DIV, "div_f32", l, r, n)
}
/// Element-wise neg: `out = -src`.
pub fn neg_f32(b: &MetalBackend, s: &Buffer, n: usize) -> Result<Buffer, MetalError> {
    dispatch_unary(b, UN_KIND_NEG, "neg_f32", s, n)
}
/// Element-wise relu: `out = max(0, src)`.
pub fn relu_f32(b: &MetalBackend, s: &Buffer, n: usize) -> Result<Buffer, MetalError> {
    dispatch_unary(b, UN_KIND_RELU, "relu_f32", s, n)
}
/// Element-wise sigmoid.
pub fn sigmoid_f32(b: &MetalBackend, s: &Buffer, n: usize) -> Result<Buffer, MetalError> {
    dispatch_unary(b, UN_KIND_SIGMOID, "sigmoid_f32", s, n)
}
/// Element-wise tanh.
pub fn tanh_f32(b: &MetalBackend, s: &Buffer, n: usize) -> Result<Buffer, MetalError> {
    dispatch_unary(b, UN_KIND_TANH, "tanh_f32", s, n)
}
/// Element-wise silu: `out = src / (1 + exp(-src))`.
pub fn silu_f32(b: &MetalBackend, s: &Buffer, n: usize) -> Result<Buffer, MetalError> {
    dispatch_unary(b, UN_KIND_SILU, "silu_f32", s, n)
}
/// Element-wise abs.
pub fn abs_f32(b: &MetalBackend, s: &Buffer, n: usize) -> Result<Buffer, MetalError> {
    dispatch_unary(b, UN_KIND_ABS, "abs_f32", s, n)
}
/// Element-wise sqrt.
pub fn sqrt_f32(b: &MetalBackend, s: &Buffer, n: usize) -> Result<Buffer, MetalError> {
    dispatch_unary(b, UN_KIND_SQRT, "sqrt_f32", s, n)
}
/// Element-wise exp.
pub fn exp_f32(b: &MetalBackend, s: &Buffer, n: usize) -> Result<Buffer, MetalError> {
    dispatch_unary(b, UN_KIND_EXP, "exp_f32", s, n)
}
/// Element-wise log.
pub fn log_f32(b: &MetalBackend, s: &Buffer, n: usize) -> Result<Buffer, MetalError> {
    dispatch_unary(b, UN_KIND_LOG, "log_f32", s, n)
}

// ----------------------------------------------------------------------
// Full-tensor reductions (sum / mean → scalar [1]).
// ----------------------------------------------------------------------

const REDUCE_KIND_SUM: u32 = 0;
const REDUCE_KIND_MEAN: u32 = 1;

/// Two-stage reduction shader. Stage 1: each threadgroup of 256
/// threads reduces a 256-stride chunk to a single partial via
/// `simd_sum` + threadgroup memory. Stage 2: 1-thread cleanup over
/// the partials. Numerical stability isn't paramount here — used
/// for MSE loss reduction where the magnitudes are bounded.
const REDUCE_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct ReduceParams {
    uint n;
    uint kind;       // 0 = sum, 1 = mean
};

kernel void reduce_partial_f32(
    constant ReduceParams& params [[buffer(0)]],
    device const float*    src    [[buffer(1)]],
    device       float*    out    [[buffer(2)]],
    uint  tg_pos [[threadgroup_position_in_grid]],
    uint  lid    [[thread_index_in_threadgroup]],
    uint  tg_size [[threads_per_threadgroup]]
) {
    threadgroup float scratch[256];
    uint stride = tg_size;
    uint base = tg_pos * stride * 1u;  // each TG covers `stride` elements

    float acc = 0.0f;
    uint idx = base + lid;
    if (idx < params.n) {
        acc = src[idx];
    }
    scratch[lid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Tree reduction within the threadgroup.
    for (uint s = stride / 2u; s > 0u; s >>= 1u) {
        if (lid < s) {
            scratch[lid] += scratch[lid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lid == 0u) {
        out[tg_pos] = scratch[0];
    }
}

kernel void reduce_finalise_f32(
    constant ReduceParams& params   [[buffer(0)]],
    device const float*    partials [[buffer(1)]],
    device       float*    out      [[buffer(2)]],
    constant     uint&     n_partials [[buffer(3)]],
    uint                   gid      [[thread_position_in_grid]]
) {
    if (gid != 0u) { return; }
    float acc = 0.0f;
    for (uint i = 0u; i < n_partials; i = i + 1u) {
        acc += partials[i];
    }
    if (params.kind == 1u) {  // mean
        acc = acc / float(params.n);
    }
    out[0] = acc;
}
"#;

#[repr(C)]
#[derive(Copy, Clone, Default)]
struct ReduceParams {
    n: u32,
    kind: u32,
}
unsafe impl bytemuck::Zeroable for ReduceParams {}
unsafe impl bytemuck::Pod for ReduceParams {}

/// Reduce `src` to a scalar `[1]` buffer. `kind` = 0 for sum,
/// 1 for mean. Two-pass: 256-stride partial reduction, then
/// single-thread cleanup.
fn reduce_full(
    backend: &MetalBackend,
    src: &Buffer,
    n: usize,
    kind: u32,
    op_name_partial: &'static str,
    op_name_final: &'static str,
) -> Result<Buffer, MetalError> {
    let pipeline_partial =
        backend.pipeline(op_name_partial, REDUCE_SHADER, "reduce_partial_f32")?;
    let pipeline_final = backend.pipeline(op_name_final, REDUCE_SHADER, "reduce_finalise_f32")?;

    let stride = 256u64;
    let n_partials = (n as u64).div_ceil(stride);
    let partials = backend.alloc_shared((n_partials as usize) * 4)?;

    let params = ReduceParams { n: n as u32, kind };
    let params_buf = backend.alloc_shared(core::mem::size_of::<ReduceParams>())?;
    unsafe {
        let dst = params_buf.contents() as *mut ReduceParams;
        *dst = params;
    }

    // Stage 1: reduce_partial into `partials`.
    backend.with_encoder(|encoder| {
        encoder.set_compute_pipeline_state(&pipeline_partial);
        encoder.set_buffer(0, Some(&params_buf), 0);
        encoder.set_buffer(1, Some(src), 0);
        encoder.set_buffer(2, Some(&partials), 0);
        let tg = MTLSize::new(stride, 1, 1);
        let grid = MTLSize::new(n_partials * stride, 1, 1);
        encoder.dispatch_threads(grid, tg);
    });
    // wait removed — see kernels.rs comment.

    // Stage 2: single-thread finalisation.
    let out = backend.alloc_shared(4)?;
    let n_partials_buf = backend.alloc_shared(4)?;
    unsafe {
        let p = n_partials_buf.contents() as *mut u32;
        *p = n_partials as u32;
    }
    backend.with_encoder(|encoder| {
        encoder.set_compute_pipeline_state(&pipeline_final);
        encoder.set_buffer(0, Some(&params_buf), 0);
        encoder.set_buffer(1, Some(&partials), 0);
        encoder.set_buffer(2, Some(&out), 0);
        encoder.set_buffer(3, Some(&n_partials_buf), 0);
        let tg2 = MTLSize::new(1, 1, 1);
        let grid2 = MTLSize::new(1, 1, 1);
        encoder.dispatch_threads(grid2, tg2);
    });
    // wait removed.

    Ok(out)
}

/// Full-tensor sum reduction `src → [1]`.
pub fn sum_f32(b: &MetalBackend, s: &Buffer, n: usize) -> Result<Buffer, MetalError> {
    reduce_full(b, s, n, REDUCE_KIND_SUM, "sum_partial", "sum_final")
}
/// Full-tensor mean reduction `src → [1]`.
pub fn mean_f32(b: &MetalBackend, s: &Buffer, n: usize) -> Result<Buffer, MetalError> {
    reduce_full(b, s, n, REDUCE_KIND_MEAN, "mean_partial", "mean_final")
}

// ----------------------------------------------------------------------
// Fused MSE-loss: `(a - b)^2 .sum() / n` (Reduction::Mean) or
// `(a - b)^2 .sum()` (Reduction::Sum) → scalar `[1]` buffer.
//
// Replaces the 3-dispatch path `sub → mul → reduce` with a single
// 2-stage reduce that fuses the elementwise (a-b)^2 into stage 1.
// Net savings per training step: 2 dispatches + 2 intermediate
// `[B,N]` buffers (256 KB each at the bench size).
// ----------------------------------------------------------------------

const MSE_REDUCE_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct ReduceParams {
    uint n;
    uint kind;       // 0 = sum, 1 = mean
};

// Stage 1: each threadgroup reads `a[i]` and `b[i]`, computes
// `(a-b)^2`, and reduces 256 such squares to one partial.
kernel void mse_reduce_partial_f32(
    constant ReduceParams& params [[buffer(0)]],
    device const float*    a      [[buffer(1)]],
    device const float*    b      [[buffer(2)]],
    device       float*    out    [[buffer(3)]],
    uint  tg_pos  [[threadgroup_position_in_grid]],
    uint  lid     [[thread_index_in_threadgroup]],
    uint  tg_size [[threads_per_threadgroup]]
) {
    threadgroup float scratch[256];
    uint stride = tg_size;
    uint base = tg_pos * stride;

    float acc = 0.0f;
    uint idx = base + lid;
    if (idx < params.n) {
        float d = a[idx] - b[idx];
        acc = d * d;
    }
    scratch[lid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Tree reduction within the threadgroup.
    for (uint s = stride / 2u; s > 0u; s >>= 1u) {
        if (lid < s) {
            scratch[lid] += scratch[lid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lid == 0u) {
        out[tg_pos] = scratch[0];
    }
}
"#;

/// Fused `MSE(a, b)` reduction kernel. Computes
/// `Σ_i (a[i] - b[i])²` (sum) or `(1/n) Σ_i (a[i] - b[i])²` (mean)
/// without materialising `a-b` or `(a-b)²` as intermediate buffers.
///
/// `kind`: 0 = sum, 1 = mean. Returns a `[1]`-element shared buffer.
pub fn mse_reduce_f32(
    backend: &MetalBackend,
    a: &Buffer,
    b: &Buffer,
    n: usize,
    kind: u32,
) -> Result<Buffer, MetalError> {
    let pipeline_partial = backend.pipeline(
        "mse_reduce_partial",
        MSE_REDUCE_SHADER,
        "mse_reduce_partial_f32",
    )?;
    // Reuse the existing single-thread finalise (sums partials and
    // optionally divides by n). Pipeline cache key is shared with the
    // generic reduce so we don't double-compile.
    let pipeline_final =
        backend.pipeline("mse_reduce_final", REDUCE_SHADER, "reduce_finalise_f32")?;

    let stride = 256u64;
    let n_partials = (n as u64).div_ceil(stride);
    let partials = backend.alloc_shared((n_partials as usize) * 4)?;

    // ReduceParams (8 bytes) + n_partials (4 bytes) are tiny constants —
    // pass inline via set_bytes to avoid 3 small alloc_shared per call.
    let params = ReduceParams { n: n as u32, kind };

    // Stage 1: fused (a-b)² reduce → partials.
    backend.with_encoder(|encoder| {
        encoder.set_compute_pipeline_state(&pipeline_partial);
        encoder.set_bytes(
            0,
            core::mem::size_of::<ReduceParams>() as u64,
            &params as *const ReduceParams as *const std::ffi::c_void,
        );
        encoder.set_buffer(1, Some(a), 0);
        encoder.set_buffer(2, Some(b), 0);
        encoder.set_buffer(3, Some(&partials), 0);
        let tg = MTLSize::new(stride, 1, 1);
        let grid = MTLSize::new(n_partials * stride, 1, 1);
        encoder.dispatch_threads(grid, tg);
    });

    // Stage 2: single-thread finalise (sum + optional /n).
    let out = backend.alloc_shared(4)?;
    let np = n_partials as u32;
    backend.with_encoder(|encoder| {
        encoder.set_compute_pipeline_state(&pipeline_final);
        encoder.set_bytes(
            0,
            core::mem::size_of::<ReduceParams>() as u64,
            &params as *const ReduceParams as *const std::ffi::c_void,
        );
        encoder.set_buffer(1, Some(&partials), 0);
        encoder.set_buffer(2, Some(&out), 0);
        encoder.set_bytes(3, 4, &np as *const u32 as *const std::ffi::c_void);
        let tg2 = MTLSize::new(1, 1, 1);
        let grid2 = MTLSize::new(1, 1, 1);
        encoder.dispatch_threads(grid2, tg2);
    });
    Ok(out)
}

// ----------------------------------------------------------------------
// add_bias: `out[b, n] = x[b, n] + bias[n]` — broadcasts bias across
// the batch dimension. Used by every Linear layer's forward pass.
// ----------------------------------------------------------------------

const ADD_BIAS_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct AddBiasParams {
    uint b;   // batch
    uint n;   // features
};

kernel void add_bias_f32(
    constant AddBiasParams& params [[buffer(0)]],
    device const float*     x      [[buffer(1)]],   // [B, N]
    device const float*     bias   [[buffer(2)]],   // [N]
    device       float*     out    [[buffer(3)]],   // [B, N]
    uint                    gid    [[thread_position_in_grid]]
) {
    uint B = params.b;
    uint N = params.n;
    uint total = B * N;
    if (gid >= total) { return; }
    uint col = gid % N;
    out[gid] = x[gid] + bias[col];
}
"#;

#[repr(C)]
#[derive(Copy, Clone, Default)]
struct AddBiasParams {
    b: u32,
    n: u32,
}
unsafe impl bytemuck::Zeroable for AddBiasParams {}
unsafe impl bytemuck::Pod for AddBiasParams {}

/// Native Metal add_bias: `out[b, n] = x[b, n] + bias[n]`.
/// `x` shape `[B, N]`, `bias` shape `[N]`, output shape `[B, N]`.
pub fn add_bias_f32(
    backend: &MetalBackend,
    x: &Buffer,
    bias: &Buffer,
    b_dim: usize,
    n_dim: usize,
) -> Result<Buffer, MetalError> {
    let pipeline = backend.pipeline("add_bias_f32", ADD_BIAS_SHADER, "add_bias_f32")?;
    let total = b_dim * n_dim;
    let out = backend.alloc_shared(total * 4)?;
    let params = AddBiasParams {
        b: b_dim as u32,
        n: n_dim as u32,
    };
    let params_buf = backend.alloc_shared(core::mem::size_of::<AddBiasParams>())?;
    // SAFETY: shared-storage uniform.
    unsafe {
        let dst = params_buf.contents() as *mut AddBiasParams;
        *dst = params;
    }
    backend.with_encoder(|encoder| {
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(&params_buf), 0);
        encoder.set_buffer(1, Some(x), 0);
        encoder.set_buffer(2, Some(bias), 0);
        encoder.set_buffer(3, Some(&out), 0);
        let max_threads = pipeline.max_total_threads_per_threadgroup();
        let tg = MTLSize::new(256u64.min(max_threads), 1, 1);
        let grid = MTLSize::new(total as u64, 1, 1);
        encoder.dispatch_threads(grid, tg);
    });
    Ok(out)
}

// ----------------------------------------------------------------------
// 2D axis reductions: `[d0, d1]` → `[d1]` (axis 0) or `[d0]` (axis 1).
//
// Used by `AddBackward::unbroadcast_to` for `add_bias([B,N], [N])`
// gradients (axis 0 reduction). Each output element is the sum/mean
// over a row or column of the input tile.
// ----------------------------------------------------------------------

/// Two-axis-aware reduction kernel. Each thread computes one output
/// scalar by walking the reduced axis. For typical autograd shapes
/// (e.g. [64, 1024] → [1024] reducing axis 0) this is bandwidth-bound
/// rather than compute-bound; trading thread-coarsening for code
/// simplicity is fine.
const REDUCE_DIM_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct ReduceDimParams {
    uint d0;
    uint d1;
    uint axis;       // 0 → reduce d0 → output [d1]; 1 → reduce d1 → output [d0]
    uint kind;       // 0 = sum, 1 = mean
};

kernel void reduce_dim_2d_f32(
    constant ReduceDimParams& params [[buffer(0)]],
    device const float*       src    [[buffer(1)]],
    device       float*       out    [[buffer(2)]],
    uint                      gid    [[thread_position_in_grid]]
) {
    if (params.axis == 0u) {
        // Reduce axis 0: each thread handles one column j ∈ [0, d1).
        if (gid >= params.d1) { return; }
        float acc = 0.0f;
        for (uint i = 0u; i < params.d0; i = i + 1u) {
            acc += src[i * params.d1 + gid];
        }
        if (params.kind == 1u) { acc = acc / float(params.d0); }
        out[gid] = acc;
    } else {
        // Reduce axis 1: each thread handles one row i ∈ [0, d0).
        if (gid >= params.d0) { return; }
        float acc = 0.0f;
        for (uint j = 0u; j < params.d1; j = j + 1u) {
            acc += src[gid * params.d1 + j];
        }
        if (params.kind == 1u) { acc = acc / float(params.d1); }
        out[gid] = acc;
    }
}
"#;

#[repr(C)]
#[derive(Copy, Clone, Default)]
struct ReduceDimParams {
    d0: u32,
    d1: u32,
    axis: u32,
    kind: u32,
}
unsafe impl bytemuck::Zeroable for ReduceDimParams {}
unsafe impl bytemuck::Pod for ReduceDimParams {}

fn reduce_dim_2d(
    backend: &MetalBackend,
    src: &Buffer,
    d0: usize,
    d1: usize,
    axis: usize,
    kind: u32,
    op_name: &'static str,
) -> Result<Buffer, MetalError> {
    if axis > 1 {
        return Err(MetalError::ShapeMismatch(format!(
            "reduce_dim_2d: axis {axis} out of range for 2D input"
        )));
    }
    let out_n = if axis == 0 { d1 } else { d0 };
    let pipeline = backend.pipeline(op_name, REDUCE_DIM_SHADER, "reduce_dim_2d_f32")?;
    let out = backend.alloc_shared(out_n * 4)?;
    let params = ReduceDimParams {
        d0: d0 as u32,
        d1: d1 as u32,
        axis: axis as u32,
        kind,
    };
    let params_buf = backend.alloc_shared(core::mem::size_of::<ReduceDimParams>())?;
    // SAFETY: shared-storage uniform.
    unsafe {
        let dst = params_buf.contents() as *mut ReduceDimParams;
        *dst = params;
    }
    backend.with_encoder(|encoder| {
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(&params_buf), 0);
        encoder.set_buffer(1, Some(src), 0);
        encoder.set_buffer(2, Some(&out), 0);
        let max_threads = pipeline.max_total_threads_per_threadgroup();
        let tg = MTLSize::new(256u64.min(max_threads), 1, 1);
        let grid = MTLSize::new(out_n as u64, 1, 1);
        encoder.dispatch_threads(grid, tg);
    });
    Ok(out)
}

/// Reduce-sum a 2D tensor along `axis` (0 or 1).
pub fn sum_dim_2d_f32(
    b: &MetalBackend,
    s: &Buffer,
    d0: usize,
    d1: usize,
    axis: usize,
) -> Result<Buffer, MetalError> {
    reduce_dim_2d(b, s, d0, d1, axis, REDUCE_KIND_SUM, "sum_dim_2d")
}

/// Reduce-mean a 2D tensor along `axis` (0 or 1).
pub fn mean_dim_2d_f32(
    b: &MetalBackend,
    s: &Buffer,
    d0: usize,
    d1: usize,
    axis: usize,
) -> Result<Buffer, MetalError> {
    reduce_dim_2d(b, s, d0, d1, axis, REDUCE_KIND_MEAN, "mean_dim_2d")
}

// ----------------------------------------------------------------------
// 2D transpose `[m, n]` → `[n, m]`.
// Used by MatMulBackward (gradient via X^T @ G) and attention QKV
// permutation. Tile-shared-memory pattern for coalesced reads + writes.
// ----------------------------------------------------------------------

const TRANSPOSE_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

constant uint TILE = 16u;

struct TransposeParams {
    uint m;
    uint n;
};

kernel void transpose2d_f32(
    constant TransposeParams& params [[buffer(0)]],
    device const float*       src    [[buffer(1)]],
    device       float*       out    [[buffer(2)]],
    uint2                     gid    [[thread_position_in_grid]],
    uint2                     lid    [[thread_position_in_threadgroup]]
) {
    threadgroup float tile[16][17];  // +1 padding to avoid bank conflicts

    uint m = params.m;
    uint n = params.n;

    // Read [m, n] → tile, write tile^T → [n, m].
    uint src_row = gid.y;
    uint src_col = gid.x;
    if (src_row < m && src_col < n) {
        tile[lid.y][lid.x] = src[src_row * n + src_col];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint dst_row = (gid.x / TILE) * TILE + lid.y;
    uint dst_col = (gid.y / TILE) * TILE + lid.x;
    if (dst_row < n && dst_col < m) {
        out[dst_row * m + dst_col] = tile[lid.x][lid.y];
    }
}
"#;

#[repr(C)]
#[derive(Copy, Clone, Default)]
struct TransposeParams {
    m: u32,
    n: u32,
}
unsafe impl bytemuck::Zeroable for TransposeParams {}
unsafe impl bytemuck::Pod for TransposeParams {}

/// 2D transpose `[m, n]` → `[n, m]`.
pub fn transpose2d_f32(
    backend: &MetalBackend,
    src: &Buffer,
    m: usize,
    n: usize,
) -> Result<Buffer, MetalError> {
    let pipeline = backend.pipeline("transpose2d_f32", TRANSPOSE_SHADER, "transpose2d_f32")?;
    let out = backend.alloc_shared(m * n * 4)?;
    let params = TransposeParams {
        m: m as u32,
        n: n as u32,
    };
    let params_buf = backend.alloc_shared(core::mem::size_of::<TransposeParams>())?;
    // SAFETY: shared-storage uniform.
    unsafe {
        let dst = params_buf.contents() as *mut TransposeParams;
        *dst = params;
    }
    backend.with_encoder(|encoder| {
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(&params_buf), 0);
        encoder.set_buffer(1, Some(src), 0);
        encoder.set_buffer(2, Some(&out), 0);
        let tg = MTLSize::new(16, 16, 1);
        // dispatch_threadgroups instead of dispatch_threads so we can pad
        // up to whole tiles cleanly.
        let groups_x = (n as u64).div_ceil(16);
        let groups_y = (m as u64).div_ceil(16);
        let grid = MTLSize::new(groups_x * 16, groups_y * 16, 1);
        encoder.dispatch_threads(grid, tg);
    });
    Ok(out)
}

// ----------------------------------------------------------------------
// Scalar-broadcast binary ops: `out = lhs op scalar` — each thread
// applies a single op_kind with a constant rhs scalar. Used by
// autograd backward paths that scale by a Tensor::scalar (e.g.
// MseBackward `g = diff * (2 / n)`). Without this, those calls
// would hit the CPU-fallback shape-mismatch path in MetalBackend's
// generic mul/add/sub/div — costing a download + CPU compute +
// upload per backward step.
// ----------------------------------------------------------------------

const BIN_SCALAR_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct BinScalarParams {
    uint  n;
    uint  op_kind;
    float scalar;
};

kernel void binary_scalar_f32(
    constant BinScalarParams& params [[buffer(0)]],
    device const float*       src    [[buffer(1)]],
    device       float*       out    [[buffer(2)]],
    uint                      gid    [[thread_position_in_grid]]
) {
    if (gid >= params.n) { return; }
    float a = src[gid];
    float b = params.scalar;
    float r;
    if (params.op_kind == 0u)      { r = a + b; }
    else if (params.op_kind == 1u) { r = a - b; }
    else if (params.op_kind == 2u) { r = a * b; }
    else                            { r = a / b; }
    out[gid] = r;
}
"#;

#[repr(C)]
#[derive(Copy, Clone, Default)]
struct BinScalarParams {
    n: u32,
    op_kind: u32,
    scalar: f32,
}
unsafe impl bytemuck::Zeroable for BinScalarParams {}
unsafe impl bytemuck::Pod for BinScalarParams {}

fn dispatch_binary_scalar(
    backend: &MetalBackend,
    op_kind: u32,
    op_name: &'static str,
    src: &Buffer,
    scalar: f32,
    n: usize,
) -> Result<Buffer, MetalError> {
    let pipeline = backend.pipeline(op_name, BIN_SCALAR_SHADER, "binary_scalar_f32")?;
    let out = backend.alloc_shared(n * 4)?;
    let params = BinScalarParams {
        n: n as u32,
        op_kind,
        scalar,
    };
    let params_buf = backend.alloc_shared(core::mem::size_of::<BinScalarParams>())?;
    // SAFETY: shared-storage uniform.
    unsafe {
        let dst = params_buf.contents() as *mut BinScalarParams;
        *dst = params;
    }
    backend.with_encoder(|encoder| {
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(&params_buf), 0);
        encoder.set_buffer(1, Some(src), 0);
        encoder.set_buffer(2, Some(&out), 0);
        let max_threads = pipeline.max_total_threads_per_threadgroup();
        let tg = MTLSize::new(256u64.min(max_threads), 1, 1);
        let grid = MTLSize::new(n as u64, 1, 1);
        encoder.dispatch_threads(grid, tg);
    });
    Ok(out)
}

/// Element-wise add by a scalar: `out = src + scalar`.
pub fn add_scalar_f32(
    b: &MetalBackend,
    s: &Buffer,
    scalar: f32,
    n: usize,
) -> Result<Buffer, MetalError> {
    dispatch_binary_scalar(b, 0, "add_scalar_f32", s, scalar, n)
}
/// Element-wise sub by a scalar: `out = src - scalar`.
pub fn sub_scalar_f32(
    b: &MetalBackend,
    s: &Buffer,
    scalar: f32,
    n: usize,
) -> Result<Buffer, MetalError> {
    dispatch_binary_scalar(b, 1, "sub_scalar_f32", s, scalar, n)
}
/// Element-wise mul by a scalar: `out = src * scalar`.
pub fn mul_scalar_f32(
    b: &MetalBackend,
    s: &Buffer,
    scalar: f32,
    n: usize,
) -> Result<Buffer, MetalError> {
    dispatch_binary_scalar(b, 2, "mul_scalar_f32", s, scalar, n)
}
/// Element-wise div by a scalar: `out = src / scalar`.
pub fn div_scalar_f32(
    b: &MetalBackend,
    s: &Buffer,
    scalar: f32,
    n: usize,
) -> Result<Buffer, MetalError> {
    dispatch_binary_scalar(b, 3, "div_scalar_f32", s, scalar, n)
}

// Fused linear (matmul + bias) was attempted but Apple's
// simdgroup_matrix API has no "add row vector" op so the fusion
// requires a per-thread post-process with simdgroup_barrier, which
// is non-trivial and didn't show meaningful gain over the separate
// matmul + add_bias_f32 path (already on-device, ~30-50 µs cost).
// Left as a follow-up if/when we replace simdgroup_matrix with a
// hand-written tile loop where bias add is trivial to inline.

/// 4-output-per-sg pattern as the 8-sg kernel below, but with 16
/// simdgroups arranged as 2 rows × 8 cols. Each "row" of 8 sg
/// covers 8 output rows × 256 cols (same as the 8-sg kernel); the
/// second row of sg covers the next 8 output rows. Workgroup output
/// tile: 16 × 256 = 4096 outputs / wg.
///
/// Doubles the work per workgroup → halves the workgroup count →
/// improves GPU occupancy on small-M shapes (e.g. the bench's M=64
/// drops from 32 to 16 workgroups, better filling Apple's ~40 cores).
const MATMUL_SIMDGROUP_F32_COARSENED_WIDE_SHADER: &str = r#"
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

kernel void matmul_simdgroup_f32_coarsened_wide(
    device const float* a       [[buffer(0)]],
    device const float* b       [[buffer(1)]],
    device       float* c       [[buffer(2)]],
    constant     uint3& dims    [[buffer(3)]],
    uint2 tg_pos                [[threadgroup_position_in_grid]],
    uint  sg_idx                [[simdgroup_index_in_threadgroup]]
) {
    uint M = dims.x;
    uint K = dims.y;
    uint N = dims.z;

    // 16 simdgroups arranged 2 rows × 8 cols.
    uint sg_row = sg_idx / 8u;
    uint sg_col = sg_idx % 8u;
    uint row_tile = tg_pos.y * 16u + sg_row * 8u;
    uint col_base = tg_pos.x * 256u + sg_col * 32u;
    if (row_tile >= M || col_base >= N) { return; }

    simdgroup_matrix<float, 8, 8> mat_a;
    simdgroup_matrix<float, 8, 8> mat_b0;
    simdgroup_matrix<float, 8, 8> mat_b1;
    simdgroup_matrix<float, 8, 8> mat_b2;
    simdgroup_matrix<float, 8, 8> mat_b3;
    simdgroup_matrix<float, 8, 8> mat_c0 = simdgroup_matrix<float, 8, 8>(0);
    simdgroup_matrix<float, 8, 8> mat_c1 = simdgroup_matrix<float, 8, 8>(0);
    simdgroup_matrix<float, 8, 8> mat_c2 = simdgroup_matrix<float, 8, 8>(0);
    simdgroup_matrix<float, 8, 8> mat_c3 = simdgroup_matrix<float, 8, 8>(0);

    for (uint k = 0u; k < K; k += 8u) {
        simdgroup_load(mat_a, a + row_tile * K + k, K);
        simdgroup_load(mat_b0, b + k * N + col_base + 0u, N);
        simdgroup_load(mat_b1, b + k * N + col_base + 8u, N);
        simdgroup_load(mat_b2, b + k * N + col_base + 16u, N);
        simdgroup_load(mat_b3, b + k * N + col_base + 24u, N);
        simdgroup_multiply_accumulate(mat_c0, mat_a, mat_b0, mat_c0);
        simdgroup_multiply_accumulate(mat_c1, mat_a, mat_b1, mat_c1);
        simdgroup_multiply_accumulate(mat_c2, mat_a, mat_b2, mat_c2);
        simdgroup_multiply_accumulate(mat_c3, mat_a, mat_b3, mat_c3);
    }

    simdgroup_store(mat_c0, c + row_tile * N + col_base + 0u, N);
    simdgroup_store(mat_c1, c + row_tile * N + col_base + 8u, N);
    simdgroup_store(mat_c2, c + row_tile * N + col_base + 16u, N);
    simdgroup_store(mat_c3, c + row_tile * N + col_base + 24u, N);
}
"#;

/// 16-simdgroup matmul. Requires `m%16==0`, `k%8==0`, `n%256==0`.
pub fn matmul_simdgroup_f32_coarsened_wide(
    backend: &MetalBackend,
    a: &Buffer,
    b: &Buffer,
    m: usize,
    k: usize,
    n: usize,
) -> Result<Buffer, MetalError> {
    if !backend.supports_metal3() {
        return Err(MetalError::Unsupported(
            "matmul_simdgroup_f32_coarsened_wide needs Metal3".to_string(),
        ));
    }
    if m % 16 != 0 || k % 8 != 0 || n % 256 != 0 {
        return Err(MetalError::ShapeMismatch(format!(
            "matmul_simdgroup_f32_coarsened_wide needs m%16==0, k%8==0, n%256==0: got m={m}, k={k}, n={n}"
        )));
    }
    let pipeline = backend.pipeline(
        "matmul_simdgroup_f32_coarsened_wide",
        MATMUL_SIMDGROUP_F32_COARSENED_WIDE_SHADER,
        "matmul_simdgroup_f32_coarsened_wide",
    )?;
    let out = backend.pool_get(m * n * 4)?;
    // Pass `dims` inline via set_bytes — a 16-byte tiny-uniform fits
    // Metal's "argument-buffer" fast path and avoids the 5–10 µs cost
    // of `alloc_shared(16)` per matmul dispatch (3× per training step).
    let dims = [m as u32, k as u32, n as u32, 0u32];
    backend.with_encoder(|encoder| {
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(a), 0);
        encoder.set_buffer(1, Some(b), 0);
        encoder.set_buffer(2, Some(&out), 0);
        encoder.set_bytes(
            3,
            (dims.len() * 4) as u64,
            dims.as_ptr() as *const std::ffi::c_void,
        );
        // 16 simdgroups × 32 threads = 512 threads.
        let tg = MTLSize::new(512, 1, 1);
        let n_tiles_x = (n / 256) as u64;
        let n_tiles_y = (m / 16) as u64;
        let grid = MTLSize::new(n_tiles_x * 512, n_tiles_y, 1);
        encoder.dispatch_threads(grid, tg);
    });
    Ok(out)
}

/// Fused matmul + per-column bias-add for the Linear forward pass:
/// `C = A @ B + bias_broadcast(N)`. Folds the bias add into the
/// matmul-result write-back, saving one full `add_bias` dispatch +
/// its `[M, N]` round-trip through global memory per training step.
///
/// Same shape constraints as the non-bias variant (`m%16==0`,
/// `k%8==0`, `n%256==0`) and same 16-simdgroup × 256-col tile layout.
const MATMUL_SIMDGROUP_F32_COARSENED_WIDE_BIAS_SHADER: &str = r#"
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

kernel void matmul_simdgroup_f32_coarsened_wide_bias(
    device const float* a        [[buffer(0)]],
    device const float* b        [[buffer(1)]],
    device const float* bias     [[buffer(2)]],
    device       float* c        [[buffer(3)]],
    constant     uint3& dims     [[buffer(4)]],
    uint2 tg_pos                 [[threadgroup_position_in_grid]],
    uint  sg_idx                 [[simdgroup_index_in_threadgroup]],
    uint  lane                   [[thread_index_in_simdgroup]]
) {
    uint M = dims.x;
    uint K = dims.y;
    uint N = dims.z;

    uint sg_row = sg_idx / 8u;
    uint sg_col = sg_idx % 8u;
    uint row_tile = tg_pos.y * 16u + sg_row * 8u;
    uint col_base = tg_pos.x * 256u + sg_col * 32u;
    if (row_tile >= M || col_base >= N) { return; }

    simdgroup_matrix<float, 8, 8> mat_a;
    simdgroup_matrix<float, 8, 8> mat_b0;
    simdgroup_matrix<float, 8, 8> mat_b1;
    simdgroup_matrix<float, 8, 8> mat_b2;
    simdgroup_matrix<float, 8, 8> mat_b3;
    simdgroup_matrix<float, 8, 8> mat_c0 = simdgroup_matrix<float, 8, 8>(0);
    simdgroup_matrix<float, 8, 8> mat_c1 = simdgroup_matrix<float, 8, 8>(0);
    simdgroup_matrix<float, 8, 8> mat_c2 = simdgroup_matrix<float, 8, 8>(0);
    simdgroup_matrix<float, 8, 8> mat_c3 = simdgroup_matrix<float, 8, 8>(0);

    for (uint k = 0u; k < K; k += 8u) {
        simdgroup_load(mat_a, a + row_tile * K + k, K);
        simdgroup_load(mat_b0, b + k * N + col_base + 0u,  N);
        simdgroup_load(mat_b1, b + k * N + col_base + 8u,  N);
        simdgroup_load(mat_b2, b + k * N + col_base + 16u, N);
        simdgroup_load(mat_b3, b + k * N + col_base + 24u, N);
        simdgroup_multiply_accumulate(mat_c0, mat_a, mat_b0, mat_c0);
        simdgroup_multiply_accumulate(mat_c1, mat_a, mat_b1, mat_c1);
        simdgroup_multiply_accumulate(mat_c2, mat_a, mat_b2, mat_c2);
        simdgroup_multiply_accumulate(mat_c3, mat_a, mat_b3, mat_c3);
    }

    // Store the simdgroup-matrix tiles into a threadgroup-private tile,
    // then a per-thread fixup adds the bias and writes to global memory.
    // Tile dimensions: 16 rows × 256 cols (4096 floats × 4B = 16 KB).
    threadgroup float tile[16 * 256];
    threadgroup float* base = &tile[(sg_row * 8u) * 256u + sg_col * 32u];
    simdgroup_store(mat_c0, base + 0u,  256u);
    simdgroup_store(mat_c1, base + 8u,  256u);
    simdgroup_store(mat_c2, base + 16u, 256u);
    simdgroup_store(mat_c3, base + 24u, 256u);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 16 sg × 32 threads = 512 threads / wg. Tile = 4096 floats.
    // Each thread writes 8 elements (4096 / 512 = 8).
    uint lid = sg_idx * 32u + lane;
    uint global_row_base = tg_pos.y * 16u;
    uint global_col_base = tg_pos.x * 256u;
    for (uint k = 0u; k < 8u; ++k) {
        uint flat = lid * 8u + k;
        uint r = flat / 256u;
        uint co = flat % 256u;
        uint gr = global_row_base + r;
        uint gc = global_col_base + co;
        if (gr < M && gc < N) {
            c[gr * N + gc] = tile[r * 256u + co] + bias[gc];
        }
    }
}
"#;

/// Fused matmul + per-column bias dispatcher. Same constraints as
/// `matmul_simdgroup_f32_coarsened_wide`. Skips the standalone
/// `add_bias` dispatch + a 4 MB intermediate write-back per Linear
/// forward.
pub fn matmul_simdgroup_f32_coarsened_wide_bias(
    backend: &MetalBackend,
    a: &Buffer,
    b: &Buffer,
    bias: &Buffer,
    m: usize,
    k: usize,
    n: usize,
) -> Result<Buffer, MetalError> {
    if !backend.supports_metal3() {
        return Err(MetalError::Unsupported(
            "matmul_simdgroup_f32_coarsened_wide_bias needs Metal3".to_string(),
        ));
    }
    if m % 16 != 0 || k % 8 != 0 || n % 256 != 0 {
        return Err(MetalError::ShapeMismatch(format!(
            "matmul_simdgroup_f32_coarsened_wide_bias needs m%16==0, k%8==0, n%256==0: got m={m}, k={k}, n={n}"
        )));
    }
    let pipeline = backend.pipeline(
        "matmul_simdgroup_f32_coarsened_wide_bias",
        MATMUL_SIMDGROUP_F32_COARSENED_WIDE_BIAS_SHADER,
        "matmul_simdgroup_f32_coarsened_wide_bias",
    )?;
    let out = backend.pool_get(m * n * 4)?;
    // Inline tiny-uniform via set_bytes — see matmul_simdgroup_f32_coarsened_wide
    // above for rationale.
    let dims = [m as u32, k as u32, n as u32, 0u32];
    backend.with_encoder(|encoder| {
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(a), 0);
        encoder.set_buffer(1, Some(b), 0);
        encoder.set_buffer(2, Some(bias), 0);
        encoder.set_buffer(3, Some(&out), 0);
        encoder.set_bytes(
            4,
            (dims.len() * 4) as u64,
            dims.as_ptr() as *const std::ffi::c_void,
        );
        let tg = MTLSize::new(512, 1, 1);
        let n_tiles_x = (n / 256) as u64;
        let n_tiles_y = (m / 16) as u64;
        let grid = MTLSize::new(n_tiles_x * 512, n_tiles_y, 1);
        encoder.dispatch_threads(grid, tg);
    });
    Ok(out)
}

/// Transpose-aware coarsened-wide matmuls — fuse the transpose into
/// `simdgroup_load(..., transpose = true)` so we skip the explicit
/// transpose kernel + intermediate buffer that backward passes would
/// otherwise dispatch. In `MatMulBackward` (Linear forward `y = x @ W`),
/// computing `dW = X^T @ dY` and `dX = dY @ W^T` saves 2 transpose
/// dispatches + their N×N intermediate buffers per training step.
///
/// Both kernels mirror `matmul_simdgroup_f32_coarsened_wide`: 16
/// simdgroups arranged 2 rows × 8 cols, each simdgroup producing 4
/// output 8×8 tiles in the N direction → 16 rows × 256 cols per
/// workgroup. Constraints: `m%16==0`, `k%8==0`, `n%256==0`.
const MATMUL_SIMDGROUP_F32_TRANSPOSED_SHADER: &str = r#"
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

// C = A @ B^T   |   A:[M,K]   B:[N,K]   C:[M,N]
// Loads B with simdgroup_load(transpose = true) so the kernel sees
// the logical B^T tile without any explicit transpose pass.
kernel void matmul_simdgroup_f32_b_t(
    device const float* a       [[buffer(0)]],
    device const float* b       [[buffer(1)]],
    device       float* c       [[buffer(2)]],
    constant     uint3& dims    [[buffer(3)]],
    uint2 tg_pos                [[threadgroup_position_in_grid]],
    uint  sg_idx                [[simdgroup_index_in_threadgroup]]
) {
    uint M = dims.x;
    uint K = dims.y;
    uint N = dims.z;

    uint sg_row = sg_idx / 8u;
    uint sg_col = sg_idx % 8u;
    uint row_tile = tg_pos.y * 16u + sg_row * 8u;
    uint col_base = tg_pos.x * 256u + sg_col * 32u;
    if (row_tile >= M || col_base >= N) { return; }

    simdgroup_matrix<float, 8, 8> mat_a;
    simdgroup_matrix<float, 8, 8> mat_b0;
    simdgroup_matrix<float, 8, 8> mat_b1;
    simdgroup_matrix<float, 8, 8> mat_b2;
    simdgroup_matrix<float, 8, 8> mat_b3;
    simdgroup_matrix<float, 8, 8> mat_c0 = simdgroup_matrix<float, 8, 8>(0);
    simdgroup_matrix<float, 8, 8> mat_c1 = simdgroup_matrix<float, 8, 8>(0);
    simdgroup_matrix<float, 8, 8> mat_c2 = simdgroup_matrix<float, 8, 8>(0);
    simdgroup_matrix<float, 8, 8> mat_c3 = simdgroup_matrix<float, 8, 8>(0);

    for (uint k = 0u; k < K; k += 8u) {
        // A: standard load (row-major).
        simdgroup_load(mat_a, a + row_tile * K + k, K);
        // B^T tile @ rows [k, k+8) cols [col_base+ofs, col_base+ofs+8)
        // = B tile @ rows [col_base+ofs, col_base+ofs+8) cols [k, k+8) loaded transposed.
        simdgroup_load(mat_b0, b + (col_base + 0u)  * K + k, K, ulong2(0), true);
        simdgroup_load(mat_b1, b + (col_base + 8u)  * K + k, K, ulong2(0), true);
        simdgroup_load(mat_b2, b + (col_base + 16u) * K + k, K, ulong2(0), true);
        simdgroup_load(mat_b3, b + (col_base + 24u) * K + k, K, ulong2(0), true);
        simdgroup_multiply_accumulate(mat_c0, mat_a, mat_b0, mat_c0);
        simdgroup_multiply_accumulate(mat_c1, mat_a, mat_b1, mat_c1);
        simdgroup_multiply_accumulate(mat_c2, mat_a, mat_b2, mat_c2);
        simdgroup_multiply_accumulate(mat_c3, mat_a, mat_b3, mat_c3);
    }

    simdgroup_store(mat_c0, c + row_tile * N + col_base + 0u,  N);
    simdgroup_store(mat_c1, c + row_tile * N + col_base + 8u,  N);
    simdgroup_store(mat_c2, c + row_tile * N + col_base + 16u, N);
    simdgroup_store(mat_c3, c + row_tile * N + col_base + 24u, N);
}

// C = A^T @ B   |   A:[K,M]   B:[K,N]   C:[M,N]
// Loads A with simdgroup_load(transpose = true) so the kernel sees
// the logical A^T tile without an explicit transpose pass.
kernel void matmul_simdgroup_f32_a_t(
    device const float* a       [[buffer(0)]],
    device const float* b       [[buffer(1)]],
    device       float* c       [[buffer(2)]],
    constant     uint3& dims    [[buffer(3)]],
    uint2 tg_pos                [[threadgroup_position_in_grid]],
    uint  sg_idx                [[simdgroup_index_in_threadgroup]]
) {
    uint M = dims.x;
    uint K = dims.y;
    uint N = dims.z;

    uint sg_row = sg_idx / 8u;
    uint sg_col = sg_idx % 8u;
    uint row_tile = tg_pos.y * 16u + sg_row * 8u;
    uint col_base = tg_pos.x * 256u + sg_col * 32u;
    if (row_tile >= M || col_base >= N) { return; }

    simdgroup_matrix<float, 8, 8> mat_a;
    simdgroup_matrix<float, 8, 8> mat_b0;
    simdgroup_matrix<float, 8, 8> mat_b1;
    simdgroup_matrix<float, 8, 8> mat_b2;
    simdgroup_matrix<float, 8, 8> mat_b3;
    simdgroup_matrix<float, 8, 8> mat_c0 = simdgroup_matrix<float, 8, 8>(0);
    simdgroup_matrix<float, 8, 8> mat_c1 = simdgroup_matrix<float, 8, 8>(0);
    simdgroup_matrix<float, 8, 8> mat_c2 = simdgroup_matrix<float, 8, 8>(0);
    simdgroup_matrix<float, 8, 8> mat_c3 = simdgroup_matrix<float, 8, 8>(0);

    for (uint k = 0u; k < K; k += 8u) {
        // A^T tile @ rows [row_tile, row_tile+8) cols [k, k+8)
        // = A tile @ rows [k, k+8) cols [row_tile, row_tile+8) loaded transposed.
        simdgroup_load(mat_a, a + k * M + row_tile, M, ulong2(0), true);
        simdgroup_load(mat_b0, b + k * N + col_base + 0u,  N);
        simdgroup_load(mat_b1, b + k * N + col_base + 8u,  N);
        simdgroup_load(mat_b2, b + k * N + col_base + 16u, N);
        simdgroup_load(mat_b3, b + k * N + col_base + 24u, N);
        simdgroup_multiply_accumulate(mat_c0, mat_a, mat_b0, mat_c0);
        simdgroup_multiply_accumulate(mat_c1, mat_a, mat_b1, mat_c1);
        simdgroup_multiply_accumulate(mat_c2, mat_a, mat_b2, mat_c2);
        simdgroup_multiply_accumulate(mat_c3, mat_a, mat_b3, mat_c3);
    }

    simdgroup_store(mat_c0, c + row_tile * N + col_base + 0u,  N);
    simdgroup_store(mat_c1, c + row_tile * N + col_base + 8u,  N);
    simdgroup_store(mat_c2, c + row_tile * N + col_base + 16u, N);
    simdgroup_store(mat_c3, c + row_tile * N + col_base + 24u, N);
}
"#;

/// Compute `C = A @ B^T` directly without materialising `B^T`.
///
/// `A:[M,K]`, `B:[N,K]` (the un-transposed layout), `C:[M,N]`.
/// Constraints: `m%16==0`, `k%8==0`, `n%256==0`. Saves one transpose
/// dispatch + intermediate `[N,K]` buffer per call vs `transpose +
/// matmul`. Used by `MatMulBackward` for `dX = dY @ W^T`.
pub fn matmul_simdgroup_f32_b_t(
    backend: &MetalBackend,
    a: &Buffer,
    b: &Buffer,
    m: usize,
    k: usize,
    n: usize,
) -> Result<Buffer, MetalError> {
    if !backend.supports_metal3() {
        return Err(MetalError::Unsupported(
            "matmul_simdgroup_f32_b_t needs Metal3".to_string(),
        ));
    }
    if m % 16 != 0 || k % 8 != 0 || n % 256 != 0 {
        return Err(MetalError::ShapeMismatch(format!(
            "matmul_simdgroup_f32_b_t needs m%16==0, k%8==0, n%256==0: got m={m}, k={k}, n={n}"
        )));
    }
    let pipeline = backend.pipeline(
        "matmul_simdgroup_f32_b_t",
        MATMUL_SIMDGROUP_F32_TRANSPOSED_SHADER,
        "matmul_simdgroup_f32_b_t",
    )?;
    let out = backend.pool_get(m * n * 4)?;
    let dims = [m as u32, k as u32, n as u32, 0u32];
    backend.with_encoder(|encoder| {
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(a), 0);
        encoder.set_buffer(1, Some(b), 0);
        encoder.set_buffer(2, Some(&out), 0);
        encoder.set_bytes(
            3,
            (dims.len() * 4) as u64,
            dims.as_ptr() as *const std::ffi::c_void,
        );
        let tg = MTLSize::new(512, 1, 1);
        let n_tiles_x = (n / 256) as u64;
        let n_tiles_y = (m / 16) as u64;
        let grid = MTLSize::new(n_tiles_x * 512, n_tiles_y, 1);
        encoder.dispatch_threads(grid, tg);
    });
    Ok(out)
}

/// Compute `C = A^T @ B` directly without materialising `A^T`.
///
/// `A:[K,M]` (the un-transposed layout), `B:[K,N]`, `C:[M,N]`.
/// Constraints: `m%16==0`, `k%8==0`, `n%256==0`. Saves one transpose
/// dispatch + intermediate `[M,K]` buffer per call vs `transpose +
/// matmul`. Used by `MatMulBackward` for `dW = X^T @ dY`.
pub fn matmul_simdgroup_f32_a_t(
    backend: &MetalBackend,
    a: &Buffer,
    b: &Buffer,
    m: usize,
    k: usize,
    n: usize,
) -> Result<Buffer, MetalError> {
    if !backend.supports_metal3() {
        return Err(MetalError::Unsupported(
            "matmul_simdgroup_f32_a_t needs Metal3".to_string(),
        ));
    }
    if m % 16 != 0 || k % 8 != 0 || n % 256 != 0 {
        return Err(MetalError::ShapeMismatch(format!(
            "matmul_simdgroup_f32_a_t needs m%16==0, k%8==0, n%256==0: got m={m}, k={k}, n={n}"
        )));
    }
    let pipeline = backend.pipeline(
        "matmul_simdgroup_f32_a_t",
        MATMUL_SIMDGROUP_F32_TRANSPOSED_SHADER,
        "matmul_simdgroup_f32_a_t",
    )?;
    let out = backend.pool_get(m * n * 4)?;
    let dims = [m as u32, k as u32, n as u32, 0u32];
    backend.with_encoder(|encoder| {
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(a), 0);
        encoder.set_buffer(1, Some(b), 0);
        encoder.set_buffer(2, Some(&out), 0);
        encoder.set_bytes(
            3,
            (dims.len() * 4) as u64,
            dims.as_ptr() as *const std::ffi::c_void,
        );
        let tg = MTLSize::new(512, 1, 1);
        let n_tiles_x = (n / 256) as u64;
        let n_tiles_y = (m / 16) as u64;
        let grid = MTLSize::new(n_tiles_x * 512, n_tiles_y, 1);
        encoder.dispatch_threads(grid, tg);
    });
    Ok(out)
}

/// **Thread-coarsened multi-simdgroup matmul** — each simdgroup
/// computes 4 output 8×8 tiles in the N direction, sharing the same
/// 8×K row of A across the 4 operations. The load-once / use-many
/// pattern increases compute-to-memory ratio by 4× vs the single-tile
/// kernel, making it the highest-throughput f32 matmul we have.
///
/// Layout
/// - Workgroup: 8 simdgroups × 32 threads = 256 threads
/// - Per simdgroup: 4 output tiles in N, each 8×8 → 8×32 region
/// - Workgroup output tile: 8 rows × (8 sg × 32) = 8×256
/// - Constraints: M divisible by 8, K divisible by 8, N divisible by 256
const MATMUL_SIMDGROUP_F32_COARSENED_SHADER: &str = r#"
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

constant uint COL_PER_SG = 4u;  // 4 output 8×8 tiles per simdgroup in N direction

kernel void matmul_simdgroup_f32_coarsened(
    device const float* a       [[buffer(0)]],
    device const float* b       [[buffer(1)]],
    device       float* c       [[buffer(2)]],
    constant     uint3& dims    [[buffer(3)]],
    uint2 tg_pos                [[threadgroup_position_in_grid]],
    uint  sg_idx                [[simdgroup_index_in_threadgroup]]
) {
    uint M = dims.x;
    uint K = dims.y;
    uint N = dims.z;

    // Workgroup arrangement: 8 sg in N direction, each sg covering
    // 4 × 8 = 32 cols. Workgroup col base = tg_pos.x * 256.
    uint row_tile = tg_pos.y * 8u;
    uint col_base = tg_pos.x * 256u + sg_idx * 32u;
    if (row_tile >= M || col_base >= N) { return; }

    simdgroup_matrix<float, 8, 8> mat_a;
    simdgroup_matrix<float, 8, 8> mat_b0;
    simdgroup_matrix<float, 8, 8> mat_b1;
    simdgroup_matrix<float, 8, 8> mat_b2;
    simdgroup_matrix<float, 8, 8> mat_b3;
    simdgroup_matrix<float, 8, 8> mat_c0 = simdgroup_matrix<float, 8, 8>(0);
    simdgroup_matrix<float, 8, 8> mat_c1 = simdgroup_matrix<float, 8, 8>(0);
    simdgroup_matrix<float, 8, 8> mat_c2 = simdgroup_matrix<float, 8, 8>(0);
    simdgroup_matrix<float, 8, 8> mat_c3 = simdgroup_matrix<float, 8, 8>(0);

    for (uint k = 0u; k < K; k += 8u) {
        // ONE load of mat_a per K-iter, reused 4× across the 4 fma's.
        simdgroup_load(mat_a, a + row_tile * K + k, K);
        // Four mat_b tiles spanning [col_base, col_base + 32).
        simdgroup_load(mat_b0, b + k * N + col_base + 0u, N);
        simdgroup_load(mat_b1, b + k * N + col_base + 8u, N);
        simdgroup_load(mat_b2, b + k * N + col_base + 16u, N);
        simdgroup_load(mat_b3, b + k * N + col_base + 24u, N);
        simdgroup_multiply_accumulate(mat_c0, mat_a, mat_b0, mat_c0);
        simdgroup_multiply_accumulate(mat_c1, mat_a, mat_b1, mat_c1);
        simdgroup_multiply_accumulate(mat_c2, mat_a, mat_b2, mat_c2);
        simdgroup_multiply_accumulate(mat_c3, mat_a, mat_b3, mat_c3);
    }

    simdgroup_store(mat_c0, c + row_tile * N + col_base + 0u, N);
    simdgroup_store(mat_c1, c + row_tile * N + col_base + 8u, N);
    simdgroup_store(mat_c2, c + row_tile * N + col_base + 16u, N);
    simdgroup_store(mat_c3, c + row_tile * N + col_base + 24u, N);
}
"#;

/// `C = A @ B` via the thread-coarsened matmul kernel. Highest f32
/// throughput; requires `m%8==0`, `k%8==0`, `n%256==0`.
pub fn matmul_simdgroup_f32_coarsened(
    backend: &MetalBackend,
    a: &Buffer,
    b: &Buffer,
    m: usize,
    k: usize,
    n: usize,
) -> Result<Buffer, MetalError> {
    if !backend.supports_metal3() {
        return Err(MetalError::Unsupported(
            "matmul_simdgroup_f32_coarsened needs Metal3".to_string(),
        ));
    }
    if m % 8 != 0 || k % 8 != 0 || n % 256 != 0 {
        return Err(MetalError::ShapeMismatch(format!(
            "matmul_simdgroup_f32_coarsened needs m%8==0, k%8==0, n%256==0: got m={m}, k={k}, n={n}"
        )));
    }
    let pipeline = backend.pipeline(
        "matmul_simdgroup_f32_coarsened",
        MATMUL_SIMDGROUP_F32_COARSENED_SHADER,
        "matmul_simdgroup_f32_coarsened",
    )?;
    let out = backend.alloc_shared(m * n * 4)?;
    let dims_buf = backend.alloc_shared(16)?;
    // SAFETY: shared-storage uniform.
    unsafe {
        let p = dims_buf.contents() as *mut u32;
        *p.add(0) = m as u32;
        *p.add(1) = k as u32;
        *p.add(2) = n as u32;
    }
    backend.with_encoder(|encoder| {
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(a), 0);
        encoder.set_buffer(1, Some(b), 0);
        encoder.set_buffer(2, Some(&out), 0);
        encoder.set_buffer(3, Some(&dims_buf), 0);
        let tg = MTLSize::new(256, 1, 1);
        let n_tiles_x = (n / 256) as u64;
        let n_tiles_y = (m / 8) as u64;
        let grid = MTLSize::new(n_tiles_x * 256, n_tiles_y, 1);
        encoder.dispatch_threads(grid, tg);
    });
    Ok(out)
}

/// **Mixed-precision matmul** — bf16 inputs, f32 output, f32
/// accumulator. Reads bf16 via implicit simdgroup_load widening,
/// keeps f32 accumulation precision, writes f32 out directly.
/// Eliminates the bf16→f32 OUTPUT cast.
const MATMUL_BF16_IN_F32_OUT_SHADER: &str = r#"
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

kernel void matmul_bf16_in_f32_out(
    device const bfloat* a       [[buffer(0)]],
    device const bfloat* b       [[buffer(1)]],
    device       float*  c       [[buffer(2)]],
    constant     uint3&  dims    [[buffer(3)]],
    uint2 tg_pos                  [[threadgroup_position_in_grid]],
    uint  sg_idx                  [[simdgroup_index_in_threadgroup]]
) {
    uint M = dims.x;
    uint K = dims.y;
    uint N = dims.z;

    uint sg_row = sg_idx / 8u;
    uint sg_col = sg_idx % 8u;
    uint row_tile = tg_pos.y * 16u + sg_row * 8u;
    uint col_base = tg_pos.x * 256u + sg_col * 32u;
    if (row_tile >= M || col_base >= N) { return; }

    simdgroup_matrix<float, 8, 8> mat_a;
    simdgroup_matrix<float, 8, 8> mat_b0;
    simdgroup_matrix<float, 8, 8> mat_b1;
    simdgroup_matrix<float, 8, 8> mat_b2;
    simdgroup_matrix<float, 8, 8> mat_b3;
    simdgroup_matrix<float, 8, 8> mat_c0 = simdgroup_matrix<float, 8, 8>(0);
    simdgroup_matrix<float, 8, 8> mat_c1 = simdgroup_matrix<float, 8, 8>(0);
    simdgroup_matrix<float, 8, 8> mat_c2 = simdgroup_matrix<float, 8, 8>(0);
    simdgroup_matrix<float, 8, 8> mat_c3 = simdgroup_matrix<float, 8, 8>(0);

    for (uint k = 0u; k < K; k += 8u) {
        simdgroup_load(mat_a, a + row_tile * K + k, K);
        simdgroup_load(mat_b0, b + k * N + col_base + 0u, N);
        simdgroup_load(mat_b1, b + k * N + col_base + 8u, N);
        simdgroup_load(mat_b2, b + k * N + col_base + 16u, N);
        simdgroup_load(mat_b3, b + k * N + col_base + 24u, N);
        simdgroup_multiply_accumulate(mat_c0, mat_a, mat_b0, mat_c0);
        simdgroup_multiply_accumulate(mat_c1, mat_a, mat_b1, mat_c1);
        simdgroup_multiply_accumulate(mat_c2, mat_a, mat_b2, mat_c2);
        simdgroup_multiply_accumulate(mat_c3, mat_a, mat_b3, mat_c3);
    }

    simdgroup_store(mat_c0, c + row_tile * N + col_base + 0u, N);
    simdgroup_store(mat_c1, c + row_tile * N + col_base + 8u, N);
    simdgroup_store(mat_c2, c + row_tile * N + col_base + 16u, N);
    simdgroup_store(mat_c3, c + row_tile * N + col_base + 24u, N);
}
"#;

/// Mixed-precision matmul: bf16 inputs, f32 output.
pub fn matmul_bf16_in_f32_out(
    backend: &MetalBackend,
    a_bf16: &Buffer,
    b_bf16: &Buffer,
    m: usize,
    k: usize,
    n: usize,
) -> Result<Buffer, MetalError> {
    if !backend.supports_metal3() {
        return Err(MetalError::Unsupported(
            "matmul_bf16_in_f32_out needs Metal3".to_string(),
        ));
    }
    if m % 16 != 0 || k % 8 != 0 || n % 256 != 0 {
        return Err(MetalError::ShapeMismatch(format!(
            "matmul_bf16_in_f32_out needs m%16==0, k%8==0, n%256==0: got m={m}, k={k}, n={n}"
        )));
    }
    let pipeline = backend.pipeline(
        "matmul_bf16_in_f32_out",
        MATMUL_BF16_IN_F32_OUT_SHADER,
        "matmul_bf16_in_f32_out",
    )?;
    let out = backend.alloc_shared(m * n * 4)?;
    let dims_buf = backend.alloc_shared(16)?;
    // SAFETY: shared-storage uniform.
    unsafe {
        let p = dims_buf.contents() as *mut u32;
        *p.add(0) = m as u32;
        *p.add(1) = k as u32;
        *p.add(2) = n as u32;
    }
    backend.with_encoder(|encoder| {
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(a_bf16), 0);
        encoder.set_buffer(1, Some(b_bf16), 0);
        encoder.set_buffer(2, Some(&out), 0);
        encoder.set_buffer(3, Some(&dims_buf), 0);
        let tg = MTLSize::new(512, 1, 1);
        let n_tiles_x = (n / 256) as u64;
        let n_tiles_y = (m / 16) as u64;
        let grid = MTLSize::new(n_tiles_x * 512, n_tiles_y, 1);
        encoder.dispatch_threads(grid, tg);
    });
    Ok(out)
}

/// Public f32→bf16 cast (used by callers that manage their own
/// bf16 cache, e.g. for amortising the cast across multiple matmuls).
pub fn cast_f32_to_bf16_pub(
    backend: &MetalBackend,
    src: &Buffer,
    n: usize,
) -> Result<Buffer, MetalError> {
    cast_f32_to_bf16_kernel(backend, src, n)
}

/// Public bf16→f32 cast.
pub fn cast_bf16_to_f32_pub(
    backend: &MetalBackend,
    src: &Buffer,
    n: usize,
) -> Result<Buffer, MetalError> {
    cast_bf16_to_f32_kernel(backend, src, n)
}

/// Public bf16-throughout matmul. Inputs and output all bf16. Caller
/// managers the input bf16 buffers (typically via the backend's
/// bf16_cache) and casts the output back to f32 if needed downstream.
pub fn matmul_simdgroup_bf16_pub(
    backend: &MetalBackend,
    a_bf16: &Buffer,
    b_bf16: &Buffer,
    m: usize,
    k: usize,
    n: usize,
) -> Result<Buffer, MetalError> {
    if !backend.supports_metal3() {
        return Err(MetalError::Unsupported(
            "matmul_simdgroup_bf16_pub needs Metal3".to_string(),
        ));
    }
    if m % 8 != 0 || k % 8 != 0 || n % 64 != 0 {
        return Err(MetalError::ShapeMismatch(format!(
            "matmul_simdgroup_bf16_pub needs m%8==0, k%8==0, n%64==0: got m={m}, k={k}, n={n}"
        )));
    }
    let pipeline = backend.pipeline(
        "matmul_simdgroup_bf16",
        MATMUL_SIMDGROUP_BF16_SHADER,
        "matmul_simdgroup_bf16",
    )?;
    let c_bf16 = backend.alloc_shared(m * n * 2)?;
    let dims_buf = backend.alloc_shared(16)?;
    // SAFETY: shared-storage uniform.
    unsafe {
        let p = dims_buf.contents() as *mut u32;
        *p.add(0) = m as u32;
        *p.add(1) = k as u32;
        *p.add(2) = n as u32;
    }
    backend.with_encoder(|encoder| {
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(a_bf16), 0);
        encoder.set_buffer(1, Some(b_bf16), 0);
        encoder.set_buffer(2, Some(&c_bf16), 0);
        encoder.set_buffer(3, Some(&dims_buf), 0);
        let tg = MTLSize::new(256, 1, 1);
        let n_tiles_x = (n / 64) as u64;
        let n_tiles_y = (m / 8) as u64;
        let grid = MTLSize::new(n_tiles_x * 256, n_tiles_y, 1);
        encoder.dispatch_threads(grid, tg);
    });
    Ok(c_bf16)
}

/// **bf16 multi-simdgroup matmul** — uses
/// `simdgroup_matrix<bfloat, 8, 8>` (Apple tensor cores at 1.5-2×
/// the throughput of `<float, 8, 8>` on M3+/M4). Inputs and outputs
/// are bf16; the f32 → bf16 conversion happens via a separate
/// `cast_f32_bf16` kernel before this one.
const MATMUL_SIMDGROUP_BF16_SHADER: &str = r#"
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

kernel void matmul_simdgroup_bf16(
    device const bfloat* a       [[buffer(0)]],
    device const bfloat* b       [[buffer(1)]],
    device       bfloat* c       [[buffer(2)]],
    constant     uint3&  dims    [[buffer(3)]],
    uint2 tg_pos                  [[threadgroup_position_in_grid]],
    uint  sg_idx                  [[simdgroup_index_in_threadgroup]]
) {
    uint M = dims.x;
    uint K = dims.y;
    uint N = dims.z;

    uint row_tile = tg_pos.y * 8u;
    uint col_tile = tg_pos.x * 64u + sg_idx * 8u;
    if (row_tile >= M || col_tile >= N) { return; }

    simdgroup_matrix<bfloat, 8, 8> mat_a;
    simdgroup_matrix<bfloat, 8, 8> mat_b;
    simdgroup_matrix<bfloat, 8, 8> mat_c = simdgroup_matrix<bfloat, 8, 8>(0);

    for (uint k = 0u; k < K; k += 8u) {
        simdgroup_load(mat_a, a + row_tile * K + k, K);
        simdgroup_load(mat_b, b + k * N + col_tile, N);
        simdgroup_multiply_accumulate(mat_c, mat_a, mat_b, mat_c);
    }

    simdgroup_store(mat_c, c + row_tile * N + col_tile, N);
}
"#;

/// Cast f32 → bf16 element-wise.
const CAST_F32_TO_BF16_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void cast_f32_to_bf16(
    device const float*  src [[buffer(0)]],
    device       bfloat* dst [[buffer(1)]],
    constant     uint&   n   [[buffer(2)]],
    uint                 gid [[thread_position_in_grid]]
) {
    if (gid >= n) { return; }
    dst[gid] = bfloat(src[gid]);
}
"#;

/// Cast bf16 → f32 element-wise.
const CAST_BF16_TO_F32_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void cast_bf16_to_f32(
    device const bfloat* src [[buffer(0)]],
    device       float*  dst [[buffer(1)]],
    constant     uint&   n   [[buffer(2)]],
    uint                 gid [[thread_position_in_grid]]
) {
    if (gid >= n) { return; }
    dst[gid] = float(src[gid]);
}
"#;

fn cast_f32_to_bf16_kernel(
    backend: &MetalBackend,
    src: &Buffer,
    n: usize,
) -> Result<Buffer, MetalError> {
    let pipeline = backend.pipeline(
        "cast_f32_to_bf16",
        CAST_F32_TO_BF16_SHADER,
        "cast_f32_to_bf16",
    )?;
    let dst = backend.alloc_shared(n * 2)?;
    let n_buf = backend.alloc_shared(4)?;
    // SAFETY: shared-storage uniform.
    unsafe {
        let p = n_buf.contents() as *mut u32;
        *p = n as u32;
    }
    backend.with_encoder(|encoder| {
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(src), 0);
        encoder.set_buffer(1, Some(&dst), 0);
        encoder.set_buffer(2, Some(&n_buf), 0);
        let max_threads = pipeline.max_total_threads_per_threadgroup();
        let tg = MTLSize::new(256u64.min(max_threads), 1, 1);
        let grid = MTLSize::new(n as u64, 1, 1);
        encoder.dispatch_threads(grid, tg);
    });
    Ok(dst)
}

fn cast_bf16_to_f32_kernel(
    backend: &MetalBackend,
    src: &Buffer,
    n: usize,
) -> Result<Buffer, MetalError> {
    let pipeline = backend.pipeline(
        "cast_bf16_to_f32",
        CAST_BF16_TO_F32_SHADER,
        "cast_bf16_to_f32",
    )?;
    let dst = backend.alloc_shared(n * 4)?;
    let n_buf = backend.alloc_shared(4)?;
    // SAFETY: shared-storage uniform.
    unsafe {
        let p = n_buf.contents() as *mut u32;
        *p = n as u32;
    }
    backend.with_encoder(|encoder| {
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(src), 0);
        encoder.set_buffer(1, Some(&dst), 0);
        encoder.set_buffer(2, Some(&n_buf), 0);
        let max_threads = pipeline.max_total_threads_per_threadgroup();
        let tg = MTLSize::new(256u64.min(max_threads), 1, 1);
        let grid = MTLSize::new(n as u64, 1, 1);
        encoder.dispatch_threads(grid, tg);
    });
    Ok(dst)
}

/// Run an f32 matmul using bf16 `simdgroup_matrix` internally for
/// the tensor-core throughput gain. Casts inputs to bf16, runs the
/// bf16 kernel, casts result back to f32. Net win on M3+/M4 when the
/// matmul is large enough to amortise the 3 cast kernels.
pub fn matmul_simdgroup_f32_via_bf16(
    backend: &MetalBackend,
    a: &Buffer,
    b: &Buffer,
    m: usize,
    k: usize,
    n: usize,
) -> Result<Buffer, MetalError> {
    if !backend.supports_metal3() {
        return Err(MetalError::Unsupported(
            "matmul_simdgroup_f32_via_bf16 needs MTLGPUFamily::Metal3".to_string(),
        ));
    }
    if m % 8 != 0 || k % 8 != 0 || n % 64 != 0 {
        return Err(MetalError::ShapeMismatch(format!(
            "matmul_simdgroup_f32_via_bf16 needs m%8==0, k%8==0, n%64==0: got m={m}, k={k}, n={n}"
        )));
    }
    let a_bf16 = cast_f32_to_bf16_kernel(backend, a, m * k)?;
    let b_bf16 = cast_f32_to_bf16_kernel(backend, b, k * n)?;
    let c_bf16 = backend.alloc_shared(m * n * 2)?;
    let pipeline = backend.pipeline(
        "matmul_simdgroup_bf16",
        MATMUL_SIMDGROUP_BF16_SHADER,
        "matmul_simdgroup_bf16",
    )?;
    let dims_buf = backend.alloc_shared(16)?;
    // SAFETY: shared-storage uniform.
    unsafe {
        let p = dims_buf.contents() as *mut u32;
        *p.add(0) = m as u32;
        *p.add(1) = k as u32;
        *p.add(2) = n as u32;
    }
    backend.with_encoder(|encoder| {
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(&a_bf16), 0);
        encoder.set_buffer(1, Some(&b_bf16), 0);
        encoder.set_buffer(2, Some(&c_bf16), 0);
        encoder.set_buffer(3, Some(&dims_buf), 0);
        let tg = MTLSize::new(256, 1, 1);
        let n_tiles_x = (n / 64) as u64;
        let n_tiles_y = (m / 8) as u64;
        let grid = MTLSize::new(n_tiles_x * 256, n_tiles_y, 1);
        encoder.dispatch_threads(grid, tg);
    });
    cast_bf16_to_f32_kernel(backend, &c_bf16, m * n)
}

/// **Multi-simdgroup** matmul kernel — one threadgroup contains 8
/// simdgroups (256 threads), each computing a separate 8×8 output
/// tile. Output tile per workgroup: 8×64 (one row of 8 tiles).
///
/// This dramatically reduces threadgroup-launch overhead vs the v1
/// kernel that runs 1 simdgroup per workgroup. On 1024×1024 matmul
/// the workgroup count drops 8× (from 16384 to 2048), letting the
/// GPU schedule the work with much better occupancy.
const MATMUL_SIMDGROUP_F32_MULTISG_SHADER: &str = r#"
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

constant uint SG_PER_TG = 8u;

kernel void matmul_simdgroup_f32_multisg(
    device const float* a       [[buffer(0)]],
    device const float* b       [[buffer(1)]],
    device       float* c       [[buffer(2)]],
    constant     uint3& dims    [[buffer(3)]],  // {M, K, N}
    uint2 tg_pos                [[threadgroup_position_in_grid]],
    uint  sg_idx                [[simdgroup_index_in_threadgroup]]
) {
    uint M = dims.x;
    uint K = dims.y;
    uint N = dims.z;

    // 8 simdgroups arranged in a single row direction, each
    // computing one 8x8 output tile. Workgroup output tile = 8x64.
    uint row_tile = tg_pos.y * 8u;
    uint col_tile = tg_pos.x * 64u + sg_idx * 8u;
    if (row_tile >= M || col_tile >= N) { return; }

    simdgroup_matrix<float, 8, 8> mat_a;
    simdgroup_matrix<float, 8, 8> mat_b;
    simdgroup_matrix<float, 8, 8> mat_c = simdgroup_matrix<float, 8, 8>(0);

    for (uint k = 0u; k < K; k += 8u) {
        simdgroup_load(mat_a, a + row_tile * K + k, K);
        simdgroup_load(mat_b, b + k * N + col_tile, N);
        simdgroup_multiply_accumulate(mat_c, mat_a, mat_b, mat_c);
    }

    simdgroup_store(mat_c, c + row_tile * N + col_tile, N);
}
"#;

/// Multi-simdgroup matmul on Metal — 8 simdgroups per threadgroup,
/// each doing one 8×8 output tile. Output tile per workgroup: 8×64.
/// Same 8-alignment + Metal3 constraints as the v1 single-simdgroup
/// kernel; specifically `n` must additionally be divisible by 64.
pub fn matmul_simdgroup_f32_multisg(
    backend: &MetalBackend,
    a: &Buffer,
    b: &Buffer,
    m: usize,
    k: usize,
    n: usize,
) -> Result<Buffer, MetalError> {
    if !backend.supports_metal3() {
        return Err(MetalError::Unsupported(
            "matmul_simdgroup_f32_multisg needs MTLGPUFamily::Metal3".to_string(),
        ));
    }
    if m % 8 != 0 || k % 8 != 0 || n % 64 != 0 {
        return Err(MetalError::ShapeMismatch(format!(
            "matmul_simdgroup_f32_multisg needs m%8==0, k%8==0, n%64==0: got m={m}, k={k}, n={n}"
        )));
    }
    let pipeline = backend.pipeline(
        "matmul_simdgroup_f32_multisg",
        MATMUL_SIMDGROUP_F32_MULTISG_SHADER,
        "matmul_simdgroup_f32_multisg",
    )?;
    let out = backend.alloc_shared(m * n * 4)?;
    let dims_buf = backend.alloc_shared(16)?;
    unsafe {
        let p = dims_buf.contents() as *mut u32;
        *p.add(0) = m as u32;
        *p.add(1) = k as u32;
        *p.add(2) = n as u32;
    }
    backend.with_encoder(|encoder| {
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(a), 0);
        encoder.set_buffer(1, Some(b), 0);
        encoder.set_buffer(2, Some(&out), 0);
        encoder.set_buffer(3, Some(&dims_buf), 0);
        // 256 threads per threadgroup = 8 simdgroups.
        let tg = MTLSize::new(256, 1, 1);
        let n_tiles_x = (n / 64) as u64;
        let n_tiles_y = (m / 8) as u64;
        let grid = MTLSize::new(n_tiles_x * 256, n_tiles_y, 1);
        encoder.dispatch_threads(grid, tg);
    });
    Ok(out)
}

/// `C = A @ B` matmul using Apple's `simdgroup_matrix<float, 8, 8>`
/// (Metal 3 family). One simdgroup (32 threads) computes one 8×8
/// output tile by iterating `K` in chunks of 8.
///
/// **Why this kernel matters** — the Apple GPU's `simdgroup_matrix`
/// hardware path delivers ~10× the throughput of the scalar f32
/// `for k { acc += a[k] * b[k] }` matmul we use in `rustorch-wgpu`
/// (which has no SUBGROUP_MATRIX path on Metal-via-wgpu). It's the
/// single biggest perf lever for closing the gap to PyTorch MPS
/// (which uses MPSMatrixMultiplication, also `simdgroup_matrix`-based).
///
/// v1: `float` accumulator (matches our F32 dtype). bf16 / fp16
/// variants land with Mixed Precision (Task J Phase 4).
const MATMUL_SIMDGROUP_F32_SHADER: &str = r#"
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

kernel void matmul_simdgroup_f32(
    device const float* a       [[buffer(0)]],
    device const float* b       [[buffer(1)]],
    device       float* c       [[buffer(2)]],
    constant     uint3& dims    [[buffer(3)]],  // {M, K, N}
    uint2 tg_pos                [[threadgroup_position_in_grid]],
    uint  sg_idx                [[simdgroup_index_in_threadgroup]],
    uint  lid                   [[thread_index_in_simdgroup]]
) {
    uint M = dims.x;
    uint K = dims.y;
    uint N = dims.z;

    // Each threadgroup contains 1 simdgroup → 1 output 8×8 tile.
    uint row_tile = tg_pos.y * 8;
    uint col_tile = tg_pos.x * 8;
    if (row_tile >= M || col_tile >= N) { return; }

    simdgroup_matrix<float, 8, 8> mat_a;
    simdgroup_matrix<float, 8, 8> mat_b;
    simdgroup_matrix<float, 8, 8> mat_c = simdgroup_matrix<float, 8, 8>(0);

    // K must be a multiple of 8 for the tile loop. Callers should
    // pad / route to the scalar fallback otherwise.
    for (uint k = 0; k < K; k += 8) {
        // simdgroup_load takes a (pointer, leading_dimension) pair.
        simdgroup_load(mat_a, a + row_tile * K + k, K);
        simdgroup_load(mat_b, b + k * N + col_tile, N);
        simdgroup_multiply_accumulate(mat_c, mat_a, mat_b, mat_c);
    }

    simdgroup_store(mat_c, c + row_tile * N + col_tile, N);
}
"#;

/// `C = A @ B` on Metal via `simdgroup_matrix<float, 8, 8>`. F32 only,
/// shapes must align to 8 (M, K, N all divisible by 8) — the only
/// constraint of the v1 kernel; padding / mixed-tile fallback comes
/// in a follow-up.
pub fn matmul_simdgroup_f32(
    backend: &MetalBackend,
    a: &Buffer,
    b: &Buffer,
    m: usize,
    k: usize,
    n: usize,
) -> Result<Buffer, MetalError> {
    if !backend.supports_metal3() {
        return Err(MetalError::Unsupported(
            "matmul_simdgroup_f32 needs MTLGPUFamily::Metal3 (M3+, A17 Pro+)".to_string(),
        ));
    }
    if m % 8 != 0 || k % 8 != 0 || n % 8 != 0 {
        return Err(MetalError::ShapeMismatch(format!(
            "matmul_simdgroup_f32 v1 needs shapes divisible by 8: got m={m}, k={k}, n={n}"
        )));
    }

    let pipeline = backend.pipeline(
        "matmul_simdgroup_f32",
        MATMUL_SIMDGROUP_F32_SHADER,
        "matmul_simdgroup_f32",
    )?;

    let out = backend.alloc_shared(m * n * 4)?;

    // dims = uint3 { M, K, N }
    let dims_buf = backend.alloc_shared(16)?; // padded to 16 for alignment
                                              // SAFETY: shared-storage buffer; pointer valid for write.
    unsafe {
        let p = dims_buf.contents() as *mut u32;
        *p.add(0) = m as u32;
        *p.add(1) = k as u32;
        *p.add(2) = n as u32;
    }

    backend.with_encoder(|encoder| {
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(a), 0);
        encoder.set_buffer(1, Some(b), 0);
        encoder.set_buffer(2, Some(&out), 0);
        encoder.set_buffer(3, Some(&dims_buf), 0);

        // Each threadgroup = 1 simdgroup = 32 threads → 1 output 8×8 tile.
        let threadgroup_size = MTLSize::new(32, 1, 1);
        let n_tiles_x = (n / 8) as u64;
        let n_tiles_y = (m / 8) as u64;
        let grid = MTLSize::new(n_tiles_x * 32, n_tiles_y, 1);
        encoder.dispatch_threads(grid, threadgroup_size);
    });
    // wait_until_completed removed — Metal handles inter-kernel sync via queue order. Only host reads (in transfer.rs::tensor_to_cpu) need an explicit wait.

    Ok(out)
}

// =============================================================================
// sgemv_f32_simd — native M=1 sgemv `y = x @ W` where W is `[K, N]` row-major.
//
// Strategy: ONE threadgroup = ONE simdgroup (32 threads) = ONE output column.
// The 32 threads stride over K (each handles K/32 elements), partial sums are
// reduced across the simdgroup with `simd_sum` in a single instruction. No
// shape constraints (works for any K, N — caller responsible for slice sizing).
//
// This is the kernel `rustorch-llm` uses for autoregressive decode where M=1.
// On M4 Max we measure ~0.05–0.20 ms per sgemv on Qwen3-14B FFN/attn shapes
// (vs ~0.09 ms via Apple Accelerate AMX cached on the same shape).
// =============================================================================

const SGEMV_F32_SIMD_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

// Coalesced sgemv: one thread = one output column. Threads in the same
// simdgroup read CONSECUTIVE columns of the same row of W (i.e. 32
// adjacent floats = one cache line) on every K iteration. x[k] is
// broadcast across the simdgroup. This pattern saturates DRAM bandwidth
// on Apple GPU; the previous "one threadgroup per output" pattern with
// strided W reads (jumping N floats per thread) was bandwidth-starved
// (30–150 GB/s vs 240+ achievable here).
kernel void sgemv_f32_simd(
    device const float* x  [[buffer(0)]],   // [K] activation
    device const float* w  [[buffer(1)]],   // [K, N] weights, row-major
    device float* y        [[buffer(2)]],   // [N] output
    constant uint2& dims   [[buffer(3)]],   // (K, N)
    uint gid               [[thread_position_in_grid]]
) {
    uint K = dims.x;
    uint N = dims.y;
    uint n_idx = gid;
    if (n_idx >= N) return;

    float sum = 0.0;
    for (uint k = 0; k < K; ++k) {
        sum += x[k] * w[k * N + n_idx];
    }
    y[n_idx] = sum;
}
"#;

/// Native M=1 sgemv on Metal: `y = x @ W` where `W` is `[K, N]` row-major.
/// One simdgroup per output column; no padding or shape constraints (just
/// requires `K, N >= 1`).
pub fn sgemv_f32_simd(
    backend: &MetalBackend,
    x: &Buffer,
    w: &Buffer,
    k: usize,
    n: usize,
) -> Result<Buffer, MetalError> {
    if !backend.supports_metal3() {
        return Err(MetalError::Unsupported(
            "sgemv_f32_simd needs MTLGPUFamily::Metal3 (M3+, A17 Pro+)".to_string(),
        ));
    }
    if k == 0 || n == 0 {
        return Err(MetalError::ShapeMismatch(format!(
            "sgemv_f32_simd needs K >= 1, N >= 1: got K={k}, N={n}"
        )));
    }
    let pipeline = backend.pipeline("sgemv_f32_simd", SGEMV_F32_SIMD_SHADER, "sgemv_f32_simd")?;
    let out = backend.alloc_shared(n * 4)?;
    let dims_buf = backend.alloc_shared(8)?; // 2 × u32
    unsafe {
        let p = dims_buf.contents() as *mut u32;
        *p.add(0) = k as u32;
        *p.add(1) = n as u32;
    }
    backend.with_encoder(|encoder| {
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(x), 0);
        encoder.set_buffer(1, Some(w), 0);
        encoder.set_buffer(2, Some(&out), 0);
        encoder.set_buffer(3, Some(&dims_buf), 0);
        // One thread per output column; threadgroup size of 64 is a
        // good Apple GPU sweet spot (2 simdgroups per TG).
        let threadgroup_size = MTLSize::new(64, 1, 1);
        let grid = MTLSize::new(n as u64, 1, 1);
        encoder.dispatch_threads(grid, threadgroup_size);
    });
    Ok(out)
}

// =============================================================================
// sgemv_q4_k_f32 — direct sgemv on Q4_K-quantised weights, no f32 expansion.
//
// Reads 144-byte Q4_K super-blocks straight from the GPU buffer, dequantises
// each block in registers (8 sub-blocks × 32 weights = 256 weights / block,
// per-sub-block 6-bit scale + 6-bit min, super-block f16 d + f16 dmin), and
// accumulates the dot product with the matching slice of `x` on the fly.
//
// One thread = one output column. The Q4_K weight matrix is laid out
// `[N, K]` row-major (each output column owns its K weights contiguously,
// `K / 256` super-blocks of 144 bytes each = `bytes_per_row`). This pattern
// strides per-row at `bytes_per_row`, which is fine on Apple GPU because
// each thread reads its own contiguous chunk and there's no cache contention
// between simdgroup lanes.
//
// The big DRAM win: Qwen3-14B `gate_up_proj` (K=5120, N=34816) reads 100 MB
// of Q4_K bytes vs 712 MB of f32 weights — 7× less bandwidth pressure.
// =============================================================================

const SGEMV_Q4_K_F32_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

constant uint BLOCK_BYTES = 144u;
constant uint BLOCK_WEIGHTS = 256u;

// Optimised v2: uchar4 vectorised nibble loads (4 bytes per memory
// access instead of 1) and float4 vectorised x reads (16 bytes per
// access). The inner loop processes 4 elements at a time, halving
// the issued load instructions. The dequant compute stays in
// registers — no threadgroup memory needed since x is already
// well-cached at the simdgroup level (consecutive threads of the
// same simdgroup read the same x[k] for different W rows).
kernel void sgemv_q4_k_f32(
    device const float* x       [[buffer(0)]],   // [K] activation
    device const uchar* w_q4k   [[buffer(1)]],   // [N, K] Q4_K row-major
    device float* y             [[buffer(2)]],   // [N]
    constant uint2& dims        [[buffer(3)]],   // (K, N)
    uint gid                    [[thread_position_in_grid]]
) {
    uint K = dims.x;
    uint N = dims.y;
    uint n_idx = gid;
    if (n_idx >= N) return;

    uint blocks_per_row = K / BLOCK_WEIGHTS;
    uint row_off = n_idx * blocks_per_row * BLOCK_BYTES;

    float acc = 0.0;

    for (uint blk = 0; blk < blocks_per_row; ++blk) {
        device const uchar* block = w_q4k + row_off + blk * BLOCK_BYTES;

        ushort d_bits = ((ushort)block[1] << 8) | (ushort)block[0];
        ushort dmin_bits = ((ushort)block[3] << 8) | (ushort)block[2];
        float d = float(as_type<half>(d_bits));
        float dmin = float(as_type<half>(dmin_bits));

        // Unpack scales/mins from 12 bytes into 8 sub-block (sc, m) pairs.
        uchar packed[12];
        for (uint i = 0; i < 12u; ++i) packed[i] = block[4 + i];
        uchar sc[8], m[8];
        for (uint i = 0; i < 4u; ++i) {
            sc[i]     = packed[i] & 0x3F;
            m[i]      = packed[i + 4] & 0x3F;
            sc[i + 4] = (packed[i + 8] & 0x0F) | ((packed[i] >> 6) << 4);
            m[i + 4]  = (packed[i + 8] >> 4)   | ((packed[i + 4] >> 6) << 4);
        }

        // qs as uchar4: 32 uchar = 8 uchar4 per pair, but we use 4-vec
        // sub-iteration to amortise loads.
        device const uchar4* qs4 = (device const uchar4*)(block + 16);

        for (uint jp = 0; jp < 4u; ++jp) {
            uint j0 = 2u * jp;
            uint j1 = 2u * jp + 1u;
            float scale0 = d * float(sc[j0]);
            float min0   = dmin * float(m[j0]);
            float scale1 = d * float(sc[j1]);
            float min1   = dmin * float(m[j1]);

            uint x_low_off  = blk * BLOCK_WEIGHTS + j0 * 32u;
            uint x_high_off = blk * BLOCK_WEIGHTS + j1 * 32u;

            // 32 nibble bytes per pair = 8 uchar4. Each uchar4 holds
            // 4 nibble bytes, each holding 2 weights (low + high
            // nibble) -> 8 weights per uchar4.
            uint qs_base = jp * 8u;  // 8 uchar4 per pair (jp * 32 / 4)
            for (uint kg = 0; kg < 8u; ++kg) {
                uchar4 nibs = qs4[qs_base + kg];
                uint k = kg * 4u;
                // 4 lanes inside uchar4
                float n0lo = scale0 * float(nibs.x & 0x0F) - min0;
                float n1lo = scale0 * float(nibs.y & 0x0F) - min0;
                float n2lo = scale0 * float(nibs.z & 0x0F) - min0;
                float n3lo = scale0 * float(nibs.w & 0x0F) - min0;
                float n0hi = scale1 * float(nibs.x >> 4)   - min1;
                float n1hi = scale1 * float(nibs.y >> 4)   - min1;
                float n2hi = scale1 * float(nibs.z >> 4)   - min1;
                float n3hi = scale1 * float(nibs.w >> 4)   - min1;
                // Vectorised x loads: 4 consecutive floats per access.
                float4 xlo = float4(x[x_low_off  + k    ],
                                    x[x_low_off  + k + 1u],
                                    x[x_low_off  + k + 2u],
                                    x[x_low_off  + k + 3u]);
                float4 xhi = float4(x[x_high_off + k    ],
                                    x[x_high_off + k + 1u],
                                    x[x_high_off + k + 2u],
                                    x[x_high_off + k + 3u]);
                acc += xlo.x * n0lo + xlo.y * n1lo + xlo.z * n2lo + xlo.w * n3lo;
                acc += xhi.x * n0hi + xhi.y * n1hi + xhi.z * n2hi + xhi.w * n3hi;
            }
        }
    }

    y[n_idx] = acc;
}
"#;

/// Direct Q4_K sgemv on Metal — no f32 dequantisation buffer in DRAM.
/// Variant that writes into a caller-provided output buffer to avoid
/// the per-call 140KB allocation in the hot path. The caller must
/// ensure `out_buf` has at least `n * 4` bytes capacity.
pub fn sgemv_q4_k_f32_into(
    backend: &MetalBackend,
    x_buf: &Buffer,
    w_q4k_buf: &Buffer,
    out_buf: &Buffer,
    k: usize,
    n: usize,
) -> Result<(), MetalError> {
    if !backend.supports_metal3() {
        return Err(MetalError::Unsupported(
            "sgemv_q4_k_f32 needs MTLGPUFamily::Metal3 (M3+, A17 Pro+)".to_string(),
        ));
    }
    if k == 0 || n == 0 || k % 256 != 0 {
        return Err(MetalError::ShapeMismatch(format!(
            "sgemv_q4_k_f32 needs K>=1, N>=1, K%256==0: got K={k}, N={n}"
        )));
    }
    let pipeline = backend.pipeline("sgemv_q4_k_f32", SGEMV_Q4_K_F32_SHADER, "sgemv_q4_k_f32")?;
    let dims_buf = backend.alloc_shared(8)?;
    unsafe {
        let p = dims_buf.contents() as *mut u32;
        *p.add(0) = k as u32;
        *p.add(1) = n as u32;
    }
    backend.with_encoder(|encoder| {
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(x_buf), 0);
        encoder.set_buffer(1, Some(w_q4k_buf), 0);
        encoder.set_buffer(2, Some(out_buf), 0);
        encoder.set_buffer(3, Some(&dims_buf), 0);
        let threadgroup_size = MTLSize::new(64, 1, 1);
        let grid = MTLSize::new(n as u64, 1, 1);
        encoder.dispatch_threads(grid, threadgroup_size);
    });
    Ok(())
}

/// Direct Q4_K sgemv on Metal — no f32 dequantisation buffer in DRAM.
///
/// `x_buf`: f32 activation buffer of length K elements (K * 4 bytes).
/// `w_q4k_buf`: raw Q4_K weight bytes laid out `[N, K]` row-major
///   (each row is `K / 256 * 144` bytes).
/// Returns a freshly-allocated f32 output buffer of length N.
///
/// Constraints: `K % 256 == 0` (Q4_K super-block size).
pub fn sgemv_q4_k_f32(
    backend: &MetalBackend,
    x_buf: &Buffer,
    w_q4k_buf: &Buffer,
    k: usize,
    n: usize,
) -> Result<Buffer, MetalError> {
    if !backend.supports_metal3() {
        return Err(MetalError::Unsupported(
            "sgemv_q4_k_f32 needs MTLGPUFamily::Metal3 (M3+, A17 Pro+)".to_string(),
        ));
    }
    if k == 0 || n == 0 || k % 256 != 0 {
        return Err(MetalError::ShapeMismatch(format!(
            "sgemv_q4_k_f32 needs K>=1, N>=1, K%256==0: got K={k}, N={n}"
        )));
    }
    let pipeline = backend.pipeline("sgemv_q4_k_f32", SGEMV_Q4_K_F32_SHADER, "sgemv_q4_k_f32")?;
    let out = backend.alloc_shared(n * 4)?;
    let dims_buf = backend.alloc_shared(8)?;
    unsafe {
        let p = dims_buf.contents() as *mut u32;
        *p.add(0) = k as u32;
        *p.add(1) = n as u32;
    }
    backend.with_encoder(|encoder| {
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(x_buf), 0);
        encoder.set_buffer(1, Some(w_q4k_buf), 0);
        encoder.set_buffer(2, Some(&out), 0);
        encoder.set_buffer(3, Some(&dims_buf), 0);
        let threadgroup_size = MTLSize::new(64, 1, 1);
        let grid = MTLSize::new(n as u64, 1, 1);
        encoder.dispatch_threads(grid, threadgroup_size);
    });
    Ok(out)
}

// =============================================================================
// LLM building-block kernels (RMSNorm, SwiGLU, in-place add) — small but
// reused 3-6 times per layer in `rustorch-llm`. Chaining these on the GPU
// instead of the CPU eliminates the CPU<->GPU sync points between matmul
// calls, which is the dominant overhead in T74's MVP wiring.
// =============================================================================

const RMS_NORM_F32_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

// One simdgroup (32 threads) cooperates on one row of x[d]. Each thread
// stride-loops over k=tid..d step 32, accumulates squared values into a
// per-thread partial sum, then we use simd_sum to reduce to a single
// scalar replicated across the simdgroup. Each thread then writes its
// stride of normalised+scaled values.
kernel void rms_norm_f32(
    device const float* x     [[buffer(0)]],
    device const float* gamma [[buffer(1)]],
    device float* y           [[buffer(2)]],
    constant uint& d          [[buffer(3)]],
    constant float& eps       [[buffer(4)]],
    uint tid                  [[thread_position_in_threadgroup]],
    uint sg_size              [[threads_per_simdgroup]]
) {
    float partial = 0.0;
    for (uint i = tid; i < d; i += sg_size) {
        float v = x[i];
        partial += v * v;
    }
    float total = simd_sum(partial);
    float inv_rms = 1.0 / sqrt(total / float(d) + eps);
    for (uint i = tid; i < d; i += sg_size) {
        y[i] = x[i] * inv_rms * gamma[i];
    }
}
"#;

/// Single-row RMSNorm on Metal. Dispatch is one simdgroup (32 threads),
/// stride-looped across `d`. Output written into `y_buf` (caller-allocated).
pub fn rms_norm_f32(
    backend: &MetalBackend,
    x_buf: &Buffer,
    gamma_buf: &Buffer,
    y_buf: &Buffer,
    d: usize,
    eps: f32,
) -> Result<(), MetalError> {
    let pipeline = backend.pipeline("rms_norm_f32", RMS_NORM_F32_SHADER, "rms_norm_f32")?;
    let d_buf = backend.alloc_shared(4)?;
    let eps_buf = backend.alloc_shared(4)?;
    unsafe {
        *(d_buf.contents() as *mut u32) = d as u32;
        *(eps_buf.contents() as *mut f32) = eps;
    }
    backend.with_encoder(|encoder| {
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(x_buf), 0);
        encoder.set_buffer(1, Some(gamma_buf), 0);
        encoder.set_buffer(2, Some(y_buf), 0);
        encoder.set_buffer(3, Some(&d_buf), 0);
        encoder.set_buffer(4, Some(&eps_buf), 0);
        let tg_size = MTLSize::new(32, 1, 1);
        let grid = MTLSize::new(32, 1, 1);
        encoder.dispatch_threads(grid, tg_size);
    });
    Ok(())
}

const SWIGLU_F32_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

// Element-wise SwiGLU: y[i] = silu(gate[i]) * up[i] where
// silu(x) = x * sigmoid(x) = x / (1 + exp(-x)).
kernel void swiglu_f32(
    device const float* gate [[buffer(0)]],
    device const float* up   [[buffer(1)]],
    device float* y          [[buffer(2)]],
    constant uint& f         [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= f) return;
    float g = gate[gid];
    float s = g / (1.0 + exp(-g));
    y[gid] = s * up[gid];
}
"#;

/// In-place SwiGLU: writes `silu(gate) * up` into `y_buf`.
pub fn swiglu_f32(
    backend: &MetalBackend,
    gate_buf: &Buffer,
    up_buf: &Buffer,
    y_buf: &Buffer,
    f: usize,
) -> Result<(), MetalError> {
    let pipeline = backend.pipeline("swiglu_f32", SWIGLU_F32_SHADER, "swiglu_f32")?;
    let f_buf = backend.alloc_shared(4)?;
    unsafe {
        *(f_buf.contents() as *mut u32) = f as u32;
    }
    backend.with_encoder(|encoder| {
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(gate_buf), 0);
        encoder.set_buffer(1, Some(up_buf), 0);
        encoder.set_buffer(2, Some(y_buf), 0);
        encoder.set_buffer(3, Some(&f_buf), 0);
        let tg_size = MTLSize::new(256, 1, 1);
        let grid = MTLSize::new(f as u64, 1, 1);
        encoder.dispatch_threads(grid, tg_size);
    });
    Ok(())
}

const ADD_INPLACE_F32_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

// In-place residual add: x[i] += y[i].
kernel void add_inplace_f32(
    device float* x       [[buffer(0)]],
    device const float* y [[buffer(1)]],
    constant uint& d      [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= d) return;
    x[gid] += y[gid];
}
"#;

/// `x_buf += y_buf` element-wise on Metal.
pub fn add_inplace_f32(
    backend: &MetalBackend,
    x_buf: &Buffer,
    y_buf: &Buffer,
    d: usize,
) -> Result<(), MetalError> {
    let pipeline =
        backend.pipeline("add_inplace_f32", ADD_INPLACE_F32_SHADER, "add_inplace_f32")?;
    let d_buf = backend.alloc_shared(4)?;
    unsafe {
        *(d_buf.contents() as *mut u32) = d as u32;
    }
    backend.with_encoder(|encoder| {
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(x_buf), 0);
        encoder.set_buffer(1, Some(y_buf), 0);
        encoder.set_buffer(2, Some(&d_buf), 0);
        let tg_size = MTLSize::new(256, 1, 1);
        let grid = MTLSize::new(d as u64, 1, 1);
        encoder.dispatch_threads(grid, tg_size);
    });
    Ok(())
}

// =============================================================================
// sgemv_q6_k_f32 — direct Q6_K sgemv on Metal. Same fused-dequant-on-the-fly
// pattern as sgemv_q4_k_f32 but for Q6_K weights (6-bit, 256 weights / 210
// bytes). Used by `rustorch-llm` for `down_proj` and `lm_head` in Q4_K_M
// Qwen3 GGUFs (Q4_K_M is a mixed-precision quant: Q4_K for most, Q6_K for
// the perplexity-critical projections).
//
// CRITICAL: Q6_K scales are stored as `int8_t` (signed). The shader must
// reinterpret them via `(device const char*)` to get sign-extension —
// reading them as `uchar` and casting to int gives values in 0..255 instead
// of -128..127 and silently corrupts ~50% of the dequantised weights (this
// is the same class of bug as T63 fix on the CPU dequant path).
// =============================================================================

const SGEMV_Q6_K_F32_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

constant uint Q6K_BYTES = 210u;
constant uint Q6K_WEIGHTS = 256u;

// Optimised v2: precompute the 4 sub-block scales once per half,
// reuse them across the 32 inner iterations (was redundantly
// computing them via float(sc_h[l/16]) at every step). Removes
// 4*32 = 128 redundant ALU ops per block per thread.
kernel void sgemv_q6_k_f32(
    device const float* x       [[buffer(0)]],   // [K]
    device const uchar* w_q6k   [[buffer(1)]],   // [N, K] Q6_K row-major
    device float* y             [[buffer(2)]],   // [N]
    constant uint2& dims        [[buffer(3)]],   // (K, N)
    uint gid                    [[thread_position_in_grid]]
) {
    uint K = dims.x;
    uint N = dims.y;
    uint n_idx = gid;
    if (n_idx >= N) return;

    uint blocks_per_row = K / Q6K_WEIGHTS;
    uint row_off = n_idx * blocks_per_row * Q6K_BYTES;

    float acc = 0.0;

    for (uint blk = 0; blk < blocks_per_row; ++blk) {
        device const uchar* block = w_q6k + row_off + blk * Q6K_BYTES;
        device const uchar* ql = block;
        device const uchar* qh = block + 128;
        device const char*  sc = (device const char*)(block + 192); // SIGNED!
        ushort d_bits = ((ushort)block[209] << 8) | (ushort)block[208];
        float d = float(as_type<half>(d_bits));

        for (uint half_idx = 0u; half_idx < 2u; ++half_idx) {
            device const uchar* ql_h = ql + half_idx * 64u;
            device const uchar* qh_h = qh + half_idx * 32u;
            device const char*  sc_h = sc + half_idx * 8;
            uint x_h_off = blk * Q6K_WEIGHTS + half_idx * 128u;

            // Precompute 8 scales (2 sub-block-of-16 × 4 quarter-positions).
            // sc_h[0..2] = sub-block 0 & 1 for q1
            // sc_h[2..4] = ditto for q2
            // sc_h[4..6] = ditto for q3
            // sc_h[6..8] = ditto for q4
            float s1_lo = d * float(sc_h[0]);
            float s1_hi = d * float(sc_h[1]);
            float s2_lo = d * float(sc_h[2]);
            float s2_hi = d * float(sc_h[3]);
            float s3_lo = d * float(sc_h[4]);
            float s3_hi = d * float(sc_h[5]);
            float s4_lo = d * float(sc_h[6]);
            float s4_hi = d * float(sc_h[7]);

            for (uint l = 0; l < 32u; ++l) {
                uchar qhh = qh_h[l];
                int q1 = (int)(ql_h[l]      & 0x0F) | ((int)((qhh >> 0) & 0x03) << 4);
                int q2 = (int)(ql_h[l + 32] & 0x0F) | ((int)((qhh >> 2) & 0x03) << 4);
                int q3 = (int)(ql_h[l]      >> 4)   | ((int)((qhh >> 4) & 0x03) << 4);
                int q4 = (int)(ql_h[l + 32] >> 4)   | ((int)((qhh >> 6) & 0x03) << 4);
                float s1 = (l < 16u) ? s1_lo : s1_hi;
                float s2 = (l < 16u) ? s2_lo : s2_hi;
                float s3 = (l < 16u) ? s3_lo : s3_hi;
                float s4 = (l < 16u) ? s4_lo : s4_hi;
                acc += x[x_h_off + l]      * (s1 * float(q1 - 32));
                acc += x[x_h_off + l + 32] * (s2 * float(q2 - 32));
                acc += x[x_h_off + l + 64] * (s3 * float(q3 - 32));
                acc += x[x_h_off + l + 96] * (s4 * float(q4 - 32));
            }
        }
    }

    y[n_idx] = acc;
}
"#;

/// Q6_K equivalent of [`sgemv_q4_k_f32_into`]. Writes into a caller-
/// provided buffer to avoid the per-call allocation in the hot path.
pub fn sgemv_q6_k_f32_into(
    backend: &MetalBackend,
    x_buf: &Buffer,
    w_q6k_buf: &Buffer,
    out_buf: &Buffer,
    k: usize,
    n: usize,
) -> Result<(), MetalError> {
    if !backend.supports_metal3() {
        return Err(MetalError::Unsupported(
            "sgemv_q6_k_f32 needs MTLGPUFamily::Metal3 (M3+, A17 Pro+)".to_string(),
        ));
    }
    if k == 0 || n == 0 || k % 256 != 0 {
        return Err(MetalError::ShapeMismatch(format!(
            "sgemv_q6_k_f32 needs K>=1, N>=1, K%256==0: got K={k}, N={n}"
        )));
    }
    let pipeline = backend.pipeline("sgemv_q6_k_f32", SGEMV_Q6_K_F32_SHADER, "sgemv_q6_k_f32")?;
    let dims_buf = backend.alloc_shared(8)?;
    unsafe {
        let p = dims_buf.contents() as *mut u32;
        *p.add(0) = k as u32;
        *p.add(1) = n as u32;
    }
    backend.with_encoder(|encoder| {
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(x_buf), 0);
        encoder.set_buffer(1, Some(w_q6k_buf), 0);
        encoder.set_buffer(2, Some(out_buf), 0);
        encoder.set_buffer(3, Some(&dims_buf), 0);
        let threadgroup_size = MTLSize::new(64, 1, 1);
        let grid = MTLSize::new(n as u64, 1, 1);
        encoder.dispatch_threads(grid, threadgroup_size);
    });
    Ok(())
}

/// Direct Q6_K sgemv on Metal — companion to [`sgemv_q4_k_f32`].
pub fn sgemv_q6_k_f32(
    backend: &MetalBackend,
    x_buf: &Buffer,
    w_q6k_buf: &Buffer,
    k: usize,
    n: usize,
) -> Result<Buffer, MetalError> {
    if !backend.supports_metal3() {
        return Err(MetalError::Unsupported(
            "sgemv_q6_k_f32 needs MTLGPUFamily::Metal3 (M3+, A17 Pro+)".to_string(),
        ));
    }
    if k == 0 || n == 0 || k % 256 != 0 {
        return Err(MetalError::ShapeMismatch(format!(
            "sgemv_q6_k_f32 needs K>=1, N>=1, K%256==0: got K={k}, N={n}"
        )));
    }
    let pipeline = backend.pipeline("sgemv_q6_k_f32", SGEMV_Q6_K_F32_SHADER, "sgemv_q6_k_f32")?;
    let out = backend.alloc_shared(n * 4)?;
    let dims_buf = backend.alloc_shared(8)?;
    unsafe {
        let p = dims_buf.contents() as *mut u32;
        *p.add(0) = k as u32;
        *p.add(1) = n as u32;
    }
    backend.with_encoder(|encoder| {
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(x_buf), 0);
        encoder.set_buffer(1, Some(w_q6k_buf), 0);
        encoder.set_buffer(2, Some(&out), 0);
        encoder.set_buffer(3, Some(&dims_buf), 0);
        let threadgroup_size = MTLSize::new(64, 1, 1);
        let grid = MTLSize::new(n as u64, 1, 1);
        encoder.dispatch_threads(grid, threadgroup_size);
    });
    Ok(out)
}

#[cfg(all(test, target_os = "macos"))]
mod tests {
    use super::*;
    use crate::backend_singleton::metal_backend;

    /// Build a deterministic F32 vector for tests (sin-based seeding).
    fn det_vec(n: usize, seed: f32) -> Vec<f32> {
        (0..n)
            .map(|i| ((i as f32 + 1.0) * seed * 0.001).sin())
            .collect()
    }

    #[test]
    #[cfg(feature = "gpu-tests")]
    fn sgemv_f32_simd_matches_cpu_reference() {
        let backend = metal_backend();
        let k = 64;
        let n = 256;
        let x = det_vec(k, 1.0);
        let w = det_vec(k * n, 0.5);

        // CPU reference: y[n_idx] = sum_k x[k] * w[k * n + n_idx]
        let mut y_ref = vec![0.0_f32; n];
        for n_idx in 0..n {
            let mut s = 0.0_f32;
            for k_idx in 0..k {
                s += x[k_idx] * w[k_idx * n + n_idx];
            }
            y_ref[n_idx] = s;
        }

        let x_buf = backend.alloc_shared(k * 4).unwrap();
        let w_buf = backend.alloc_shared(k * n * 4).unwrap();
        unsafe {
            std::ptr::copy_nonoverlapping(x.as_ptr(), x_buf.contents() as *mut f32, k);
            std::ptr::copy_nonoverlapping(w.as_ptr(), w_buf.contents() as *mut f32, k * n);
        }
        let out = sgemv_f32_simd(backend, &x_buf, &w_buf, k, n).unwrap();
        backend.drain();
        let mut y_metal = vec![0.0_f32; n];
        unsafe {
            std::ptr::copy_nonoverlapping(out.contents() as *const f32, y_metal.as_mut_ptr(), n);
        }
        for i in 0..n {
            let abs_err = (y_ref[i] - y_metal[i]).abs();
            let denom = y_ref[i].abs().max(1e-4);
            assert!(
                abs_err / denom < 1e-3,
                "sgemv mismatch at idx {i}: ref={} metal={} (rel err {})",
                y_ref[i],
                y_metal[i],
                abs_err / denom
            );
        }
    }

    /// Naive CPU matmul for parity reference.
    fn cpu_matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
        let mut c = vec![0.0_f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0_f32;
                for kk in 0..k {
                    acc += a[i * k + kk] * b[kk * n + j];
                }
                c[i * n + j] = acc;
            }
        }
        c
    }

    #[test]
    fn matmul_simdgroup_f32_parity_with_cpu_64x64() {
        let backend = metal_backend();
        if !backend.supports_metal3() {
            eprintln!(
                "[matmul_simdgroup_f32] skipping: device does not support Metal3 \
                 ({})",
                backend.adapter_name()
            );
            return;
        }
        let m = 64;
        let k = 64;
        let n = 64;
        let a = det_vec(m * k, 1.0);
        let b = det_vec(k * n, 0.5);
        let expected = cpu_matmul(&a, &b, m, k, n);

        let a_buf = backend.alloc_shared(m * k * 4).expect("a alloc");
        let b_buf = backend.alloc_shared(k * n * 4).expect("b alloc");
        // SAFETY: shared-storage buffers, pointers valid for n*4 bytes.
        unsafe {
            let pa = a_buf.contents() as *mut f32;
            for (i, val) in a.iter().enumerate().take(m * k) {
                *pa.add(i) = *val;
            }
            let pb = b_buf.contents() as *mut f32;
            for (i, val) in b.iter().enumerate().take(k * n) {
                *pb.add(i) = *val;
            }
        }

        let out = matmul_simdgroup_f32(backend, &a_buf, &b_buf, m, k, n).expect("dispatch");
        // Kernel dispatch is async post-commit; drain before host read.
        backend.drain();
        // SAFETY: shared-storage output buffer, valid for m*n*4 bytes.
        let got: Vec<f32> = unsafe {
            let p = out.contents() as *const f32;
            std::slice::from_raw_parts(p, m * n).to_vec()
        };

        // Cosine similarity ≥ 0.999 (parity tolerance — FMA reordering
        // produces tiny float drift between CPU and GPU).
        let mut dot = 0.0_f64;
        let mut na = 0.0_f64;
        let mut nb = 0.0_f64;
        for i in 0..(m * n) {
            dot += got[i] as f64 * expected[i] as f64;
            na += (got[i] as f64).powi(2);
            nb += (expected[i] as f64).powi(2);
        }
        let cs = dot / (na.sqrt() * nb.sqrt());
        assert!(
            cs > 0.999,
            "matmul_simdgroup_f32 cosine sim too low: {cs:.6}"
        );
    }

    #[test]
    fn add_f32_smoke_parity_with_cpu() {
        let backend = metal_backend();
        let n = 1024;
        let lhs_data: Vec<f32> = (0..n).map(|i| i as f32 * 0.5).collect();
        let rhs_data: Vec<f32> = (0..n).map(|i| (i as f32 + 1.0) * 0.25).collect();
        let expected: Vec<f32> = lhs_data
            .iter()
            .zip(rhs_data.iter())
            .map(|(&a, &b)| a + b)
            .collect();

        // Upload via MTLStorageModeShared so we can write directly.
        let lhs_buf = backend.alloc_shared(n * 4).expect("lhs alloc");
        let rhs_buf = backend.alloc_shared(n * 4).expect("rhs alloc");
        // SAFETY: shared-storage buffer, host pointer valid for n*4 bytes.
        unsafe {
            let p = lhs_buf.contents() as *mut f32;
            for (i, val) in lhs_data.iter().enumerate().take(n) {
                *p.add(i) = *val;
            }
            let p = rhs_buf.contents() as *mut f32;
            for (i, val) in rhs_data.iter().enumerate().take(n) {
                *p.add(i) = *val;
            }
        }

        let out = add_f32(backend, &lhs_buf, &rhs_buf, n).expect("dispatch");
        backend.drain();

        // SAFETY: shared-storage output buffer, valid for n*4 bytes.
        let got: Vec<f32> = unsafe {
            let p = out.contents() as *const f32;
            std::slice::from_raw_parts(p, n).to_vec()
        };

        for i in 0..n {
            let _ = lhs_data[i]; // shut up unused-binding for the symmetry of the loop
            assert!(
                (got[i] - expected[i]).abs() < 1e-6,
                "mismatch at i={i}: got {} vs expected {}",
                got[i],
                expected[i]
            );
        }
    }

    /// Naive CPU `C = A @ B^T` reference: A:[M,K], B:[N,K], C:[M,N].
    fn cpu_matmul_b_t(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
        let mut c = vec![0.0_f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0_f32;
                for kk in 0..k {
                    // A[i, kk] * B[j, kk]   (B^T[kk, j] = B[j, kk])
                    acc += a[i * k + kk] * b[j * k + kk];
                }
                c[i * n + j] = acc;
            }
        }
        c
    }

    /// Naive CPU `C = A^T @ B` reference: A:[K,M], B:[K,N], C:[M,N].
    fn cpu_matmul_a_t(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
        let mut c = vec![0.0_f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0_f32;
                for kk in 0..k {
                    // A^T[i, kk] * B[kk, j]  (A^T[i, kk] = A[kk, i])
                    acc += a[kk * m + i] * b[kk * n + j];
                }
                c[i * n + j] = acc;
            }
        }
        c
    }

    fn fill_buf(buf: &Buffer, data: &[f32]) {
        // SAFETY: shared-storage buffer, host pointer valid for data.len()*4 bytes.
        unsafe {
            let p = buf.contents() as *mut f32;
            for (i, val) in data.iter().enumerate() {
                *p.add(i) = *val;
            }
        }
    }

    fn cosine_sim(a: &[f32], b: &[f32]) -> f64 {
        let mut dot = 0.0_f64;
        let mut na = 0.0_f64;
        let mut nb = 0.0_f64;
        for i in 0..a.len() {
            dot += a[i] as f64 * b[i] as f64;
            na += (a[i] as f64).powi(2);
            nb += (b[i] as f64).powi(2);
        }
        dot / (na.sqrt() * nb.sqrt())
    }

    #[test]
    fn matmul_simdgroup_f32_b_t_parity_with_cpu_16x16x256() {
        let backend = metal_backend();
        if !backend.supports_metal3() {
            eprintln!("[matmul_simdgroup_f32_b_t] skipping: no Metal3");
            return;
        }
        let m = 16;
        let k = 16;
        let n = 256;
        let a = det_vec(m * k, 1.0);
        let b = det_vec(n * k, 0.5);
        let expected = cpu_matmul_b_t(&a, &b, m, k, n);

        let a_buf = backend.alloc_shared(m * k * 4).expect("a alloc");
        let b_buf = backend.alloc_shared(n * k * 4).expect("b alloc");
        fill_buf(&a_buf, &a);
        fill_buf(&b_buf, &b);

        let out = matmul_simdgroup_f32_b_t(backend, &a_buf, &b_buf, m, k, n).expect("dispatch");
        backend.drain();
        let got: Vec<f32> = unsafe {
            let p = out.contents() as *const f32;
            std::slice::from_raw_parts(p, m * n).to_vec()
        };

        let cs = cosine_sim(&got, &expected);
        assert!(cs > 0.999, "cosine_sim {cs:.6} too low for b_t kernel");
    }

    #[test]
    fn mse_reduce_f32_mean_parity_with_cpu_8192() {
        let backend = metal_backend();
        let n = 8192;
        let a: Vec<f32> = det_vec(n, 1.0);
        let b: Vec<f32> = det_vec(n, 0.5);
        let expected_mean: f32 = {
            let mut acc = 0.0f64;
            for i in 0..n {
                let d = (a[i] - b[i]) as f64;
                acc += d * d;
            }
            (acc / n as f64) as f32
        };
        let expected_sum: f32 = {
            let mut acc = 0.0f64;
            for i in 0..n {
                let d = (a[i] - b[i]) as f64;
                acc += d * d;
            }
            acc as f32
        };

        let a_buf = backend.alloc_shared(n * 4).expect("a alloc");
        let b_buf = backend.alloc_shared(n * 4).expect("b alloc");
        fill_buf(&a_buf, &a);
        fill_buf(&b_buf, &b);

        // mean
        let out_mean = mse_reduce_f32(backend, &a_buf, &b_buf, n, 1).expect("dispatch mean");
        backend.drain();
        let got_mean: f32 = unsafe { *(out_mean.contents() as *const f32) };
        let rel_err_mean =
            ((got_mean - expected_mean).abs() / expected_mean.abs().max(1e-6)) as f64;
        assert!(
            rel_err_mean < 1e-4,
            "mse_reduce mean mismatch: got {got_mean:.6e} expected {expected_mean:.6e} (rel {rel_err_mean:.2e})"
        );

        // sum
        let out_sum = mse_reduce_f32(backend, &a_buf, &b_buf, n, 0).expect("dispatch sum");
        backend.drain();
        let got_sum: f32 = unsafe { *(out_sum.contents() as *const f32) };
        let rel_err_sum = ((got_sum - expected_sum).abs() / expected_sum.abs().max(1e-6)) as f64;
        assert!(
            rel_err_sum < 1e-4,
            "mse_reduce sum mismatch: got {got_sum:.6e} expected {expected_sum:.6e} (rel {rel_err_sum:.2e})"
        );
    }

    #[test]
    fn matmul_simdgroup_f32_coarsened_wide_bias_parity_64x1024x1024() {
        let backend = metal_backend();
        if !backend.supports_metal3() {
            eprintln!("[matmul_simdgroup_f32_coarsened_wide_bias] skipping: no Metal3");
            return;
        }
        let m = 64;
        let k = 1024;
        let n = 1024;
        let a = det_vec(m * k, 1.0);
        let b = det_vec(k * n, 0.5);
        let bias = det_vec(n, 0.25);

        // Expected: standard matmul + bias broadcast across rows.
        let mut expected = cpu_matmul(&a, &b, m, k, n);
        for i in 0..m {
            for j in 0..n {
                expected[i * n + j] += bias[j];
            }
        }

        let a_buf = backend.alloc_shared(m * k * 4).expect("a alloc");
        let b_buf = backend.alloc_shared(k * n * 4).expect("b alloc");
        let bias_buf = backend.alloc_shared(n * 4).expect("bias alloc");
        fill_buf(&a_buf, &a);
        fill_buf(&b_buf, &b);
        fill_buf(&bias_buf, &bias);

        let out =
            matmul_simdgroup_f32_coarsened_wide_bias(backend, &a_buf, &b_buf, &bias_buf, m, k, n)
                .expect("dispatch");
        backend.drain();
        let got: Vec<f32> = unsafe {
            let p = out.contents() as *const f32;
            std::slice::from_raw_parts(p, m * n).to_vec()
        };

        let cs = cosine_sim(&got, &expected);
        assert!(
            cs > 0.999,
            "cosine_sim {cs:.6} too low for matmul+bias kernel"
        );
    }

    #[test]
    fn matmul_simdgroup_f32_a_t_parity_with_cpu_16x16x256() {
        let backend = metal_backend();
        if !backend.supports_metal3() {
            eprintln!("[matmul_simdgroup_f32_a_t] skipping: no Metal3");
            return;
        }
        let m = 16;
        let k = 16;
        let n = 256;
        // A:[K,M] in raw layout — caller passes a "transposed" buffer.
        let a = det_vec(k * m, 1.0);
        let b = det_vec(k * n, 0.5);
        let expected = cpu_matmul_a_t(&a, &b, m, k, n);

        let a_buf = backend.alloc_shared(k * m * 4).expect("a alloc");
        let b_buf = backend.alloc_shared(k * n * 4).expect("b alloc");
        fill_buf(&a_buf, &a);
        fill_buf(&b_buf, &b);

        let out = matmul_simdgroup_f32_a_t(backend, &a_buf, &b_buf, m, k, n).expect("dispatch");
        backend.drain();
        let got: Vec<f32> = unsafe {
            let p = out.contents() as *const f32;
            std::slice::from_raw_parts(p, m * n).to_vec()
        };

        let cs = cosine_sim(&got, &expected);
        assert!(cs > 0.999, "cosine_sim {cs:.6} too low for a_t kernel");
    }
}
