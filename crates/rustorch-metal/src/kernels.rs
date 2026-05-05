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

// Transposed-layout variant of the Q4_K sgemv kernel.
//
// Layout reshuffle (done once at load time):
//   original GGUF: row-major [N, K/256, 144] — row n_idx is at offset
//                  n_idx * blocks_per_row * 144 (contiguous across blocks
//                  of K, but jumps `bytes_per_row` between adjacent rows).
//   transposed:    [K/256, N, 144] — for a given block_idx, all N rows
//                  are contiguous (144 bytes each), so 32 threads in
//                  the same simdgroup reading the same block_idx hit
//                  one cache line per ~4 threads instead of N cache
//                  lines.
//
// Reads of `x[block_chunk]` are unchanged (broadcast across the
// simdgroup is already free). The big win is on the W reads.
const SGEMV_Q4_K_F32_T_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

constant uint BLOCK_BYTES = 144u;
constant uint BLOCK_WEIGHTS = 256u;

kernel void sgemv_q4_k_f32_transposed(
    device const float* x       [[buffer(0)]],   // [K]
    device const uchar* w_q4k   [[buffer(1)]],   // [K/256, N, 144] transposed
    device float* y             [[buffer(2)]],   // [N]
    constant uint2& dims        [[buffer(3)]],   // (K, N)
    uint gid                    [[thread_position_in_grid]]
) {
    uint K = dims.x;
    uint N = dims.y;
    uint n_idx = gid;
    if (n_idx >= N) return;

    uint blocks_per_row = K / BLOCK_WEIGHTS;
    float acc = 0.0;

    for (uint blk = 0; blk < blocks_per_row; ++blk) {
        // Transposed offset: block `blk` for row `n_idx` is at
        //   blk * N * 144 + n_idx * 144
        device const uchar* block = w_q4k + blk * N * BLOCK_BYTES + n_idx * BLOCK_BYTES;

        ushort d_bits = ((ushort)block[1] << 8) | (ushort)block[0];
        ushort dmin_bits = ((ushort)block[3] << 8) | (ushort)block[2];
        float d = float(as_type<half>(d_bits));
        float dmin = float(as_type<half>(dmin_bits));

        uchar packed[12];
        for (uint i = 0; i < 12u; ++i) packed[i] = block[4 + i];
        uchar sc[8], m[8];
        for (uint i = 0; i < 4u; ++i) {
            sc[i]     = packed[i] & 0x3F;
            m[i]      = packed[i + 4] & 0x3F;
            sc[i + 4] = (packed[i + 8] & 0x0F) | ((packed[i] >> 6) << 4);
            m[i + 4]  = (packed[i + 8] >> 4)   | ((packed[i + 4] >> 6) << 4);
        }

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

            uint qs_base = jp * 8u;
            for (uint kg = 0; kg < 8u; ++kg) {
                uchar4 nibs = qs4[qs_base + kg];
                uint k = kg * 4u;
                float n0lo = scale0 * float(nibs.x & 0x0F) - min0;
                float n1lo = scale0 * float(nibs.y & 0x0F) - min0;
                float n2lo = scale0 * float(nibs.z & 0x0F) - min0;
                float n3lo = scale0 * float(nibs.w & 0x0F) - min0;
                float n0hi = scale1 * float(nibs.x >> 4)   - min1;
                float n1hi = scale1 * float(nibs.y >> 4)   - min1;
                float n2hi = scale1 * float(nibs.z >> 4)   - min1;
                float n3hi = scale1 * float(nibs.w >> 4)   - min1;
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

/// Q4_K sgemv with transposed weight layout (block-major):
/// `w_q4k_buf` must be a one-shot repack of the GGUF `[N, K/256, 144]`
/// row-major bytes into `[K/256, N, 144]` block-major. Use
/// [`repack_q4_k_transposed`] for the repack helper.
pub fn sgemv_q4_k_f32_transposed_into(
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
            "sgemv_q4_k_f32_t: K%256==0 required (K={k}, N={n})"
        )));
    }
    let pipeline = backend.pipeline(
        "sgemv_q4_k_f32_transposed",
        SGEMV_Q4_K_F32_T_SHADER,
        "sgemv_q4_k_f32_transposed",
    )?;
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

/// Repack a Q4_K weight tensor from GGUF `[N, K/256, 144]` row-major
/// to `[K/256, N, 144]` block-major (one `144`-byte super-block per
/// (block_idx, row) cell). One-shot, host-side, ~50-700 MB depending
/// on layer.
pub fn repack_q4_k_transposed(src: &[u8], k: usize, n: usize) -> Vec<u8> {
    let blocks_per_row = k / 256;
    assert_eq!(src.len(), n * blocks_per_row * 144);
    let mut dst = vec![0u8; src.len()];
    for n_idx in 0..n {
        for blk in 0..blocks_per_row {
            let src_off = n_idx * blocks_per_row * 144 + blk * 144;
            let dst_off = blk * n * 144 + n_idx * 144;
            dst[dst_off..dst_off + 144].copy_from_slice(&src[src_off..src_off + 144]);
        }
    }
    dst
}

// Simdgroup-cooperative variant: 32 threads cooperate on ONE output column
// via K-reduction. Each thread handles a stride of super-blocks; partial
// dot products are reduced through `simd_sum`. Designed for the case
// where N is so large that 1-thread-per-output already saturates the GPU
// and the per-thread per-output strided loads dominate (e.g. lm_head
// N=151936). For K=5120, blocks_per_row=20 — under-utilises the
// simdgroup (most threads idle) so we use this only for huge-N cases.
const SGEMV_Q4_K_F32_SIMDCOOP_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

constant uint BLOCK_BYTES = 144u;
constant uint BLOCK_WEIGHTS = 256u;

kernel void sgemv_q4_k_f32_simdcoop(
    device const float* x       [[buffer(0)]],
    device const uchar* w_q4k   [[buffer(1)]],
    device float* y             [[buffer(2)]],
    constant uint2& dims        [[buffer(3)]],   // (K, N)
    uint tg_id                  [[threadgroup_position_in_grid]],
    uint tid                    [[thread_position_in_threadgroup]],
    uint sg_size                [[threads_per_simdgroup]]
) {
    uint K = dims.x;
    uint N = dims.y;
    uint n_idx = tg_id;
    if (n_idx >= N) return;

    uint blocks_per_row = K / BLOCK_WEIGHTS;
    uint row_off = n_idx * blocks_per_row * BLOCK_BYTES;

    float partial = 0.0;

    for (uint blk = tid; blk < blocks_per_row; blk += sg_size) {
        device const uchar* block = w_q4k + row_off + blk * BLOCK_BYTES;
        ushort d_bits = ((ushort)block[1] << 8) | (ushort)block[0];
        ushort dmin_bits = ((ushort)block[3] << 8) | (ushort)block[2];
        float d = float(as_type<half>(d_bits));
        float dmin = float(as_type<half>(dmin_bits));
        uchar packed[12];
        for (uint i = 0; i < 12u; ++i) packed[i] = block[4 + i];
        uchar sc[8], m[8];
        for (uint i = 0; i < 4u; ++i) {
            sc[i]     = packed[i] & 0x3F;
            m[i]      = packed[i + 4] & 0x3F;
            sc[i + 4] = (packed[i + 8] & 0x0F) | ((packed[i] >> 6) << 4);
            m[i + 4]  = (packed[i + 8] >> 4)   | ((packed[i + 4] >> 6) << 4);
        }
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
            uint qs_base = jp * 8u;
            for (uint kg = 0; kg < 8u; ++kg) {
                uchar4 nibs = qs4[qs_base + kg];
                uint kk = kg * 4u;
                float n0lo = scale0 * float(nibs.x & 0x0F) - min0;
                float n1lo = scale0 * float(nibs.y & 0x0F) - min0;
                float n2lo = scale0 * float(nibs.z & 0x0F) - min0;
                float n3lo = scale0 * float(nibs.w & 0x0F) - min0;
                float n0hi = scale1 * float(nibs.x >> 4)   - min1;
                float n1hi = scale1 * float(nibs.y >> 4)   - min1;
                float n2hi = scale1 * float(nibs.z >> 4)   - min1;
                float n3hi = scale1 * float(nibs.w >> 4)   - min1;
                partial += x[x_low_off  + kk    ] * n0lo;
                partial += x[x_low_off  + kk + 1] * n1lo;
                partial += x[x_low_off  + kk + 2] * n2lo;
                partial += x[x_low_off  + kk + 3] * n3lo;
                partial += x[x_high_off + kk    ] * n0hi;
                partial += x[x_high_off + kk + 1] * n1hi;
                partial += x[x_high_off + kk + 2] * n2hi;
                partial += x[x_high_off + kk + 3] * n3hi;
            }
        }
    }

    float total = simd_sum(partial);
    if (tid == 0) {
        y[n_idx] = total;
    }
}
"#;

// T89 — Multi-row Q4_K simdcoop sgemv. Port of llama.cpp's
// `kernel_mul_mv_q4_K_f32_impl` with N_R0_Q4_K = 2: process 2 output rows
// per simdgroup. The 32 threads K-cooperate identically to the single-row
// simdcoop, but maintain 2 partial accumulators (one per row) and read
// from 2 weight rows simultaneously. The shared x-tile is loaded once
// per block and reused for both rows — divides x memory traffic by 2.
//
// For Qwen3-14B Q4_K_M shapes (K=5120, x=20KB per row, W=2880 bytes per row),
// x reads dominate bandwidth so multi-row is a clear win.
const SGEMV_Q4_K_F32_SIMDCOOP_NR2_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

constant uint BLOCK_BYTES = 144u;
constant uint BLOCK_WEIGHTS = 256u;

kernel void sgemv_q4_k_f32_simdcoop_nr2(
    device const float* x       [[buffer(0)]],
    device const uchar* w_q4k   [[buffer(1)]],
    device float* y             [[buffer(2)]],
    constant uint2& dims        [[buffer(3)]],   // (K, N)
    uint tg_id                  [[threadgroup_position_in_grid]],
    uint tid                    [[thread_position_in_threadgroup]],
    uint sg_size                [[threads_per_simdgroup]]
) {
    uint K = dims.x;
    uint N = dims.y;
    uint n_idx_base = tg_id * 2u;
    if (n_idx_base >= N) return;
    bool has_row1 = (n_idx_base + 1u) < N;

    uint blocks_per_row = K / BLOCK_WEIGHTS;
    uint row_off_0 = n_idx_base * blocks_per_row * BLOCK_BYTES;
    uint row_off_1 = (n_idx_base + 1u) * blocks_per_row * BLOCK_BYTES;

    float partial0 = 0.0;
    float partial1 = 0.0;

    for (uint blk = tid; blk < blocks_per_row; blk += sg_size) {
        // Pre-load the x-tile once for this block (256 floats spread
        // across 8 sub-block slices of 32 each). We don't actually
        // pre-load in registers since each lane fetches its own
        // x[xx], but the shared block index ensures the GPU's L2 /
        // texture cache is warm for both rows' subsequent loads.
        uint x_blk_base = blk * BLOCK_WEIGHTS;

        // --- Row 0 ---
        device const uchar* block0 = w_q4k + row_off_0 + blk * BLOCK_BYTES;
        ushort d_bits0 = ((ushort)block0[1] << 8) | (ushort)block0[0];
        ushort dmin_bits0 = ((ushort)block0[3] << 8) | (ushort)block0[2];
        float d0 = float(as_type<half>(d_bits0));
        float dmin0 = float(as_type<half>(dmin_bits0));
        uchar packed0[12];
        for (uint i = 0; i < 12u; ++i) packed0[i] = block0[4 + i];
        uchar sc0[8], m0[8];
        for (uint i = 0; i < 4u; ++i) {
            sc0[i]     = packed0[i] & 0x3F;
            m0[i]      = packed0[i + 4] & 0x3F;
            sc0[i + 4] = (packed0[i + 8] & 0x0F) | ((packed0[i] >> 6) << 4);
            m0[i + 4]  = (packed0[i + 8] >> 4)   | ((packed0[i + 4] >> 6) << 4);
        }
        device const uchar4* qs4_0 = (device const uchar4*)(block0 + 16);

        // --- Row 1 ---
        device const uchar* block1 = w_q4k + row_off_1 + blk * BLOCK_BYTES;
        ushort d_bits1 = has_row1 ? (((ushort)block1[1] << 8) | (ushort)block1[0]) : 0u;
        ushort dmin_bits1 = has_row1 ? (((ushort)block1[3] << 8) | (ushort)block1[2]) : 0u;
        float d1 = has_row1 ? float(as_type<half>(d_bits1)) : 0.0f;
        float dmin1 = has_row1 ? float(as_type<half>(dmin_bits1)) : 0.0f;
        uchar packed1[12];
        if (has_row1) {
            for (uint i = 0; i < 12u; ++i) packed1[i] = block1[4 + i];
        } else {
            for (uint i = 0; i < 12u; ++i) packed1[i] = 0;
        }
        uchar sc1[8], m1[8];
        for (uint i = 0; i < 4u; ++i) {
            sc1[i]     = packed1[i] & 0x3F;
            m1[i]      = packed1[i + 4] & 0x3F;
            sc1[i + 4] = (packed1[i + 8] & 0x0F) | ((packed1[i] >> 6) << 4);
            m1[i + 4]  = (packed1[i + 8] >> 4)   | ((packed1[i + 4] >> 6) << 4);
        }
        device const uchar4* qs4_1 = (device const uchar4*)(block1 + 16);

        // Inner loop: walk the 4 (j0, j1) pairs × 8 nibble groups, fetching
        // x once per position and using it for both rows.
        for (uint jp = 0; jp < 4u; ++jp) {
            uint j0 = 2u * jp;
            uint j1 = 2u * jp + 1u;

            float scale00 = d0 * float(sc0[j0]);
            float min00   = dmin0 * float(m0[j0]);
            float scale01 = d0 * float(sc0[j1]);
            float min01   = dmin0 * float(m0[j1]);

            float scale10 = d1 * float(sc1[j0]);
            float min10   = dmin1 * float(m1[j0]);
            float scale11 = d1 * float(sc1[j1]);
            float min11   = dmin1 * float(m1[j1]);

            uint x_low_off  = x_blk_base + j0 * 32u;
            uint x_high_off = x_blk_base + j1 * 32u;
            uint qs_base = jp * 8u;

            for (uint kg = 0; kg < 8u; ++kg) {
                uchar4 nibs0 = qs4_0[qs_base + kg];
                uchar4 nibs1 = qs4_1[qs_base + kg];
                uint kk = kg * 4u;

                // Fetch x once, use for both rows
                float xl0 = x[x_low_off  + kk    ];
                float xl1 = x[x_low_off  + kk + 1];
                float xl2 = x[x_low_off  + kk + 2];
                float xl3 = x[x_low_off  + kk + 3];
                float xh0 = x[x_high_off + kk    ];
                float xh1 = x[x_high_off + kk + 1];
                float xh2 = x[x_high_off + kk + 2];
                float xh3 = x[x_high_off + kk + 3];

                // Row 0 partials
                partial0 += xl0 * (scale00 * float(nibs0.x & 0x0F) - min00);
                partial0 += xl1 * (scale00 * float(nibs0.y & 0x0F) - min00);
                partial0 += xl2 * (scale00 * float(nibs0.z & 0x0F) - min00);
                partial0 += xl3 * (scale00 * float(nibs0.w & 0x0F) - min00);
                partial0 += xh0 * (scale01 * float(nibs0.x >> 4)   - min01);
                partial0 += xh1 * (scale01 * float(nibs0.y >> 4)   - min01);
                partial0 += xh2 * (scale01 * float(nibs0.z >> 4)   - min01);
                partial0 += xh3 * (scale01 * float(nibs0.w >> 4)   - min01);

                // Row 1 partials
                partial1 += xl0 * (scale10 * float(nibs1.x & 0x0F) - min10);
                partial1 += xl1 * (scale10 * float(nibs1.y & 0x0F) - min10);
                partial1 += xl2 * (scale10 * float(nibs1.z & 0x0F) - min10);
                partial1 += xl3 * (scale10 * float(nibs1.w & 0x0F) - min10);
                partial1 += xh0 * (scale11 * float(nibs1.x >> 4)   - min11);
                partial1 += xh1 * (scale11 * float(nibs1.y >> 4)   - min11);
                partial1 += xh2 * (scale11 * float(nibs1.z >> 4)   - min11);
                partial1 += xh3 * (scale11 * float(nibs1.w >> 4)   - min11);
            }
        }
    }

    float total0 = simd_sum(partial0);
    float total1 = simd_sum(partial1);
    if (tid == 0) {
        y[n_idx_base] = total0;
        if (has_row1) {
            y[n_idx_base + 1u] = total1;
        }
    }
}
"#;

/// T89 — Multi-row Q4_K simdcoop sgemv (2 rows per simdgroup).
///
/// Same K-cooperation as `sgemv_q4_k_f32_simdcoop_into` but processes 2
/// output rows per simdgroup. Halves x-memory bandwidth pressure when
/// blocks_per_row × BLOCK_BYTES (W bytes per row) is small relative to
/// K × 4 (x bytes), which is the common case for Qwen3-14B (e.g., gate/up
/// at K=5120: W=2880B per row, x=20KB → x dominates 7×).
pub fn sgemv_q4_k_f32_simdcoop_nr2_into(
    backend: &MetalBackend,
    x_buf: &Buffer,
    w_q4k_buf: &Buffer,
    out_buf: &Buffer,
    k: usize,
    n: usize,
) -> Result<(), MetalError> {
    if !backend.supports_metal3() {
        return Err(MetalError::Unsupported(
            "sgemv_q4_k_f32_simdcoop_nr2 needs Metal3".to_string(),
        ));
    }
    if k == 0 || n == 0 || k % 256 != 0 {
        return Err(MetalError::ShapeMismatch(format!(
            "sgemv_q4_k_f32_simdcoop_nr2: K%256==0 required (K={k}, N={n})"
        )));
    }
    let pipeline = backend.pipeline(
        "sgemv_q4_k_f32_simdcoop_nr2",
        SGEMV_Q4_K_F32_SIMDCOOP_NR2_SHADER,
        "sgemv_q4_k_f32_simdcoop_nr2",
    )?;
    let dims = [k as u32, n as u32];
    backend.with_encoder(|encoder| {
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(x_buf), 0);
        encoder.set_buffer(1, Some(w_q4k_buf), 0);
        encoder.set_buffer(2, Some(out_buf), 0);
        encoder.set_bytes(3, 8, dims.as_ptr() as *const std::ffi::c_void);
        let tg_size = MTLSize::new(32, 1, 1);
        // n/2 simdgroups (round up) since each handles 2 rows.
        let n_sg = (n as u64).div_ceil(2);
        let grid = MTLSize::new(32 * n_sg, 1, 1);
        encoder.dispatch_threads(grid, tg_size);
    });
    Ok(())
}

// T92 — Batched Q4_K sgemv kernel (foundation for multi-token forward).
//
// Accepts B input vectors of length K (laid out contiguously: x[B*K]) and
// produces B output vectors of length N (out[B*N]). Each thread computes
// one output element (n_idx, batch_idx). The Q4_K weight buffer is shared
// across all B batches — read once per (n_idx, blk) pair, reused for all B.
//
// Thread/grid mapping:
//   - tg_id.x  : n_idx ∈ [0, N)
//   - tg_id.y  : batch_idx ∈ [0, B)
//   - 1 thread per output element (no simdgroup K-coop in this baseline
//     batch kernel; can be added in T93+ as needed)
//
// Memory analysis for K=5120, N=5120, B=4:
//   - W: K*N/2 bytes (Q4_K) = ~7.4 MB; read once, cached for all 4 batches
//   - x: B*K*4 bytes = 80 KB; read 4 separate vectors
//   - out: B*N*4 bytes = 80 KB
//   Per output: (W + B*x + B*out) / (B*N) ≈ K/(N) bytes = 1 byte/output
//
// vs B=1 (4 separate sgemv): 4 * (W + K*4 + N*4) / N = ~K * 4 / N bytes/output.
// Ratio: ~×4 less bandwidth per output for batched.
const SGEMV_Q4_K_F32_BATCH_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

constant uint BLOCK_BYTES = 144u;
constant uint BLOCK_WEIGHTS = 256u;

kernel void sgemv_q4_k_f32_batch(
    device const float* x       [[buffer(0)]],   // [B, K]
    device const uchar* w_q4k   [[buffer(1)]],   // [N, K] Q4_K row-major
    device float* y             [[buffer(2)]],   // [B, N]
    constant uint3& dims        [[buffer(3)]],   // (K, N, B)
    uint2 gid                   [[thread_position_in_grid]]
) {
    uint K = dims.x;
    uint N = dims.y;
    uint B = dims.z;
    uint n_idx = gid.x;
    uint batch_idx = gid.y;
    if (n_idx >= N || batch_idx >= B) return;

    uint blocks_per_row = K / BLOCK_WEIGHTS;
    uint row_off = n_idx * blocks_per_row * BLOCK_BYTES;
    // x base for this batch
    device const float* xb = x + batch_idx * K;

    float acc = 0.0;

    for (uint blk = 0; blk < blocks_per_row; ++blk) {
        device const uchar* block = w_q4k + row_off + blk * BLOCK_BYTES;
        ushort d_bits = ((ushort)block[1] << 8) | (ushort)block[0];
        ushort dmin_bits = ((ushort)block[3] << 8) | (ushort)block[2];
        float d = float(as_type<half>(d_bits));
        float dmin = float(as_type<half>(dmin_bits));
        uchar packed[12];
        for (uint i = 0; i < 12u; ++i) packed[i] = block[4 + i];
        uchar sc[8], m[8];
        for (uint i = 0; i < 4u; ++i) {
            sc[i]     = packed[i] & 0x3F;
            m[i]      = packed[i + 4] & 0x3F;
            sc[i + 4] = (packed[i + 8] & 0x0F) | ((packed[i] >> 6) << 4);
            m[i + 4]  = (packed[i + 8] >> 4)   | ((packed[i + 4] >> 6) << 4);
        }
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
            uint qs_base = jp * 8u;
            for (uint kg = 0; kg < 8u; ++kg) {
                uchar4 nibs = qs4[qs_base + kg];
                uint kk = kg * 4u;
                float n0lo = scale0 * float(nibs.x & 0x0F) - min0;
                float n1lo = scale0 * float(nibs.y & 0x0F) - min0;
                float n2lo = scale0 * float(nibs.z & 0x0F) - min0;
                float n3lo = scale0 * float(nibs.w & 0x0F) - min0;
                float n0hi = scale1 * float(nibs.x >> 4)   - min1;
                float n1hi = scale1 * float(nibs.y >> 4)   - min1;
                float n2hi = scale1 * float(nibs.z >> 4)   - min1;
                float n3hi = scale1 * float(nibs.w >> 4)   - min1;
                acc += xb[x_low_off  + kk    ] * n0lo;
                acc += xb[x_low_off  + kk + 1] * n1lo;
                acc += xb[x_low_off  + kk + 2] * n2lo;
                acc += xb[x_low_off  + kk + 3] * n3lo;
                acc += xb[x_high_off + kk    ] * n0hi;
                acc += xb[x_high_off + kk + 1] * n1hi;
                acc += xb[x_high_off + kk + 2] * n2hi;
                acc += xb[x_high_off + kk + 3] * n3hi;
            }
        }
    }

    y[batch_idx * N + n_idx] = acc;
}
"#;

/// T92 — Batched Q4_K sgemv: compute B independent (x_b @ W^T) in 1 dispatch.
///
/// Inputs:
///   `x_buf`: f32 [B, K] — B input vectors, contiguous batch-major
///   `w_q4k_buf`: Q4_K weight buffer [N, K/256, 144]
///   `out_buf`: f32 [B, N] — B output vectors
///   `k`, `n`, `b`: dimensions
///
/// Foundation for multi-token forward (T93+). The W bytes are shared
/// across all B batches via cache; only x reads scale with B.
pub fn sgemv_q4_k_f32_batch_into(
    backend: &MetalBackend,
    x_buf: &Buffer,
    w_q4k_buf: &Buffer,
    out_buf: &Buffer,
    k: usize,
    n: usize,
    b: usize,
) -> Result<(), MetalError> {
    if !backend.supports_metal3() {
        return Err(MetalError::Unsupported(
            "sgemv_q4_k_f32_batch needs Metal3".to_string(),
        ));
    }
    if k == 0 || n == 0 || b == 0 || k % 256 != 0 {
        return Err(MetalError::ShapeMismatch(format!(
            "sgemv_q4_k_f32_batch: K%256==0 required (K={k}, N={n}, B={b})"
        )));
    }
    let pipeline = backend.pipeline(
        "sgemv_q4_k_f32_batch",
        SGEMV_Q4_K_F32_BATCH_SHADER,
        "sgemv_q4_k_f32_batch",
    )?;
    let dims = [k as u32, n as u32, b as u32];
    backend.with_encoder(|encoder| {
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(x_buf), 0);
        encoder.set_buffer(1, Some(w_q4k_buf), 0);
        encoder.set_buffer(2, Some(out_buf), 0);
        encoder.set_bytes(3, 12, dims.as_ptr() as *const std::ffi::c_void);
        let tg_size = MTLSize::new(64, 1, 1);
        let grid = MTLSize::new(n as u64, b as u64, 1);
        encoder.dispatch_threads(grid, tg_size);
    });
    Ok(())
}

// T91 — Faithful port of llama.cpp's `kernel_mul_mv_q4_K_f32_impl`.
//
// Three optims combined that our previous nr2 missed:
//   1. **Preloaded x tile in registers** (yl[16] + yh[16] = 32 floats).
//      The 32 threads partition x into 4 groups (ix=0..3) that handle
//      separate K-blocks (ib += 4). Each thread loads its 32 floats of
//      x once per block, reuses for both rows.
//   2. **Factored Q4_K formula**: `sumf += d*combined_acc - dmin*(sumy*sc_odd)`.
//      Combines the multiply-accumulate with the bias subtraction into
//      a single FMA chain instead of subtracting per-nibble.
//   3. **nr0=2 multi-row** with shared sumy[4] (sum of x across 4 sub-blocks).
//      The same x tile + sumy serves both rows.
//
// 32 threads × 1 simdgroup = 1 threadgroup processes nr0=2 outputs.
// Inner loop: ib goes 0, 4, 8, ... so each of the 4 partitions ix=0..3
// covers a different stride.
const SGEMV_Q4_K_F32_LCPP_NR2_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

constant uint BLOCK_BYTES = 144u;
constant uint BLOCK_WEIGHTS = 256u;
constant short NR0 = 2;
constant ushort KMASK1 = 0x3f3f;
constant ushort KMASK2 = 0x0f0f;
constant ushort KMASK3 = 0xc0c0;

kernel void sgemv_q4_k_f32_lcpp_nr2(
    device const float*  x      [[buffer(0)]],
    device const uchar*  w_q4k  [[buffer(1)]],
    device float*        y      [[buffer(2)]],
    constant uint2&      dims   [[buffer(3)]],
    uint                 tg_id  [[threadgroup_position_in_grid]],
    ushort               tiisg  [[thread_index_in_simdgroup]]
) {
    uint K = dims.x;
    uint N = dims.y;
    uint blocks_per_row = K / BLOCK_WEIGHTS;

    uint first_row = tg_id * (uint)NR0;
    if (first_row >= N) return;

    short ix = (short)(tiisg / 8u);    // 0..3
    short it = (short)(tiisg % 8u);    // 0..7
    short iq = it / 4;                 // 0 or 1
    short ir = it % 4;                 // 0..3

    int nb = (int)blocks_per_row;

    // x partition: each thread starts at ix*256 and steps 4*256 per iter
    device const float* y4 = x + ix * 256 + 64 * iq + 8 * ir;

    float yl[16];
    float yh[16];
    float sumf[2] = {0.0, 0.0};

    ushort sc16[4];
    thread const uchar* sc8 = (thread const uchar*)sc16;

    // Row stride in bytes (full row of W in Q4_K layout)
    uint row_stride = blocks_per_row * BLOCK_BYTES;

    for (int ib = ix; ib < nb; ib += 4) {
        // Load 32 floats of x for this block; track sumy[k] for the dmin path
        float4 sumy = {0.0, 0.0, 0.0, 0.0};
        for (short i = 0; i < 8; ++i) {
            yl[i + 0] = y4[i + 0];   sumy[0] += yl[i + 0];
            yl[i + 8] = y4[i + 32];  sumy[1] += yl[i + 8];
            yh[i + 0] = y4[i + 128]; sumy[2] += yh[i + 0];
            yh[i + 8] = y4[i + 160]; sumy[3] += yh[i + 8];
        }

        for (short row = 0; row < NR0; ++row) {
            uint nrow = first_row + (uint)row;
            if (nrow >= N) continue;

            // Block pointer for this row & ib
            device const uchar* block = w_q4k + (uint64_t)nrow * row_stride + (uint)ib * BLOCK_BYTES;

            // d, dmin
            device const uint16_t* dh_ptr = (device const uint16_t*)(block);
            float d    = float(as_type<half>(dh_ptr[0]));
            float dmin = float(as_type<half>(dh_ptr[1]));

            // scales as 6 ushorts at offset 4
            device const uint16_t* sc = (device const uint16_t*)(block + 4) + iq;

            // qs as ushorts at offset 16
            device const uint16_t* q1 = (device const uint16_t*)(block + 16) + 16 * iq + 4 * ir;
            device const uint16_t* q2 = q1 + 32;

            // Unpack 4 scales/mins
            sc16[0] = sc[0] & KMASK1;
            sc16[1] = sc[2] & KMASK1;
            sc16[2] = ((sc[4] >> 0) & KMASK2) | ((sc[0] & KMASK3) >> 2);
            sc16[3] = ((sc[4] >> 4) & KMASK2) | ((sc[2] & KMASK3) >> 2);

            float4 acc1 = {0.0, 0.0, 0.0, 0.0};
            float4 acc2 = {0.0, 0.0, 0.0, 0.0};

            for (short i = 0; i < 4; ++i) {
                acc1[0] += yl[2 * i + 0] * float(q1[i] & 0x000F);
                acc1[1] += yl[2 * i + 1] * float(q1[i] & 0x0F00);
                acc1[2] += yl[2 * i + 8] * float(q1[i] & 0x00F0);
                acc1[3] += yl[2 * i + 9] * float(q1[i] & 0xF000);
                acc2[0] += yh[2 * i + 0] * float(q2[i] & 0x000F);
                acc2[1] += yh[2 * i + 1] * float(q2[i] & 0x0F00);
                acc2[2] += yh[2 * i + 8] * float(q2[i] & 0x00F0);
                acc2[3] += yh[2 * i + 9] * float(q2[i] & 0xF000);
            }

            sumf[row] += d * ((acc1[0] + 1.0f/256.0f * acc1[1]) * float(sc8[0]) +
                              (acc1[2] + 1.0f/256.0f * acc1[3]) * float(sc8[1]) * 1.0f/16.0f +
                              (acc2[0] + 1.0f/256.0f * acc2[1]) * float(sc8[4]) +
                              (acc2[2] + 1.0f/256.0f * acc2[3]) * float(sc8[5]) * 1.0f/16.0f) -
                       dmin * (sumy[0] * float(sc8[2]) + sumy[1] * float(sc8[3]) +
                               sumy[2] * float(sc8[6]) + sumy[3] * float(sc8[7]));
        }

        y4 += 4 * (int)BLOCK_WEIGHTS;
    }

    for (short row = 0; row < NR0; ++row) {
        uint nrow = first_row + (uint)row;
        if (nrow >= N) break;
        float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) {
            y[nrow] = sum_all;
        }
    }
}
"#;

/// T91 — Faithful port of llama.cpp's `kernel_mul_mv_q4_K_f32_impl` with
/// N_R0_Q4_K=2. Combines load-x-once (yl/yh registers), factored Q4_K
/// formula (d*combined - dmin*sumy*sc_odd), and 2-row simdgroup processing.
pub fn sgemv_q4_k_f32_lcpp_nr2_into(
    backend: &MetalBackend,
    x_buf: &Buffer,
    w_q4k_buf: &Buffer,
    out_buf: &Buffer,
    k: usize,
    n: usize,
) -> Result<(), MetalError> {
    if !backend.supports_metal3() {
        return Err(MetalError::Unsupported(
            "sgemv_q4_k_f32_lcpp_nr2 needs Metal3".to_string(),
        ));
    }
    if k == 0 || n == 0 || k % 256 != 0 {
        return Err(MetalError::ShapeMismatch(format!(
            "sgemv_q4_k_f32_lcpp_nr2: K%256==0 required (K={k}, N={n})"
        )));
    }
    let pipeline = backend.pipeline(
        "sgemv_q4_k_f32_lcpp_nr2",
        SGEMV_Q4_K_F32_LCPP_NR2_SHADER,
        "sgemv_q4_k_f32_lcpp_nr2",
    )?;
    let dims = [k as u32, n as u32];
    backend.with_encoder(|encoder| {
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(x_buf), 0);
        encoder.set_buffer(1, Some(w_q4k_buf), 0);
        encoder.set_buffer(2, Some(out_buf), 0);
        encoder.set_bytes(3, 8, dims.as_ptr() as *const std::ffi::c_void);
        let tg_size = MTLSize::new(32, 1, 1);
        let n_sg = (n as u64).div_ceil(2);
        let grid = MTLSize::new(32 * n_sg, 1, 1);
        encoder.dispatch_threads(grid, tg_size);
    });
    Ok(())
}

// T84 — quad-cooperative Q4_K sgemv. The 32 threads of a simdgroup are
// split into 4 "quarters" of 8 threads each; each quarter computes one
// output by K-cooperating across blocks_per_row blocks. Reduction stays
// inside the quarter via simd_shuffle_xor with masks 1, 2, 4 (which
// never cross the 8-lane boundary). Best for shapes where blocks_per_row
// < 32 (so the regular simdcoop kernel wastes 32 - blocks_per_row threads
// per simdgroup) — typical Qwen3-14B Q4_K rows have blocks_per_row=20.
const SGEMV_Q4_K_F32_QUADCOOP_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

constant uint BLOCK_BYTES = 144u;
constant uint BLOCK_WEIGHTS = 256u;

kernel void sgemv_q4_k_f32_quadcoop(
    device const float* x       [[buffer(0)]],
    device const uchar* w_q4k   [[buffer(1)]],
    device float* y             [[buffer(2)]],
    constant uint2& dims        [[buffer(3)]],   // (K, N)
    uint tg_id                  [[threadgroup_position_in_grid]],
    uint tid                    [[thread_position_in_threadgroup]]
) {
    uint K = dims.x;
    uint N = dims.y;
    // Each tg processes 4 outputs starting at base = tg_id * 4.
    uint base = tg_id * 4u;
    uint quarter = tid / 8u;       // 0..3
    uint lane_q  = tid % 8u;       // 0..7
    uint n_idx = base + quarter;

    // n_idx may run past N if N % 4 != 0 — handle later before the write,
    // but still join the reduction so simd_shuffle_xor stays well-defined.

    uint blocks_per_row = K / BLOCK_WEIGHTS;
    uint row_off = (n_idx < N) ? n_idx * blocks_per_row * BLOCK_BYTES : 0u;

    float partial = 0.0;
    if (n_idx < N) {
        for (uint blk = lane_q; blk < blocks_per_row; blk += 8u) {
            device const uchar* block = w_q4k + row_off + blk * BLOCK_BYTES;
            ushort d_bits = ((ushort)block[1] << 8) | (ushort)block[0];
            ushort dmin_bits = ((ushort)block[3] << 8) | (ushort)block[2];
            float d = float(as_type<half>(d_bits));
            float dmin = float(as_type<half>(dmin_bits));
            uchar packed[12];
            for (uint i = 0; i < 12u; ++i) packed[i] = block[4 + i];
            uchar sc[8], m[8];
            for (uint i = 0; i < 4u; ++i) {
                sc[i]     = packed[i] & 0x3F;
                m[i]      = packed[i + 4] & 0x3F;
                sc[i + 4] = (packed[i + 8] & 0x0F) | ((packed[i] >> 6) << 4);
                m[i + 4]  = (packed[i + 8] >> 4)   | ((packed[i + 4] >> 6) << 4);
            }
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
                uint qs_base = jp * 8u;
                for (uint kg = 0; kg < 8u; ++kg) {
                    uchar4 nibs = qs4[qs_base + kg];
                    uint kk = kg * 4u;
                    float n0lo = scale0 * float(nibs.x & 0x0F) - min0;
                    float n1lo = scale0 * float(nibs.y & 0x0F) - min0;
                    float n2lo = scale0 * float(nibs.z & 0x0F) - min0;
                    float n3lo = scale0 * float(nibs.w & 0x0F) - min0;
                    float n0hi = scale1 * float(nibs.x >> 4)   - min1;
                    float n1hi = scale1 * float(nibs.y >> 4)   - min1;
                    float n2hi = scale1 * float(nibs.z >> 4)   - min1;
                    float n3hi = scale1 * float(nibs.w >> 4)   - min1;
                    partial += x[x_low_off  + kk    ] * n0lo;
                    partial += x[x_low_off  + kk + 1] * n1lo;
                    partial += x[x_low_off  + kk + 2] * n2lo;
                    partial += x[x_low_off  + kk + 3] * n3lo;
                    partial += x[x_high_off + kk    ] * n0hi;
                    partial += x[x_high_off + kk + 1] * n1hi;
                    partial += x[x_high_off + kk + 2] * n2hi;
                    partial += x[x_high_off + kk + 3] * n3hi;
                }
            }
        }
    }

    // Reduce within 8-thread quarter. XOR masks 1, 2, 4 stay inside the
    // 8-lane group (max stride < 8) — quarters don't bleed into each other.
    float sum = partial;
    sum += simd_shuffle_xor(sum, 1);
    sum += simd_shuffle_xor(sum, 2);
    sum += simd_shuffle_xor(sum, 4);
    if (lane_q == 0u && n_idx < N) {
        y[n_idx] = sum;
    }
}
"#;

/// Quad-cooperative Q4_K sgemv (T84). 4 outputs per simdgroup,
/// 8 threads K-cooperate per output. Reduces the wasted-thread count
/// when blocks_per_row < 32 (plain simdcoop ties up 32 threads per
/// output with at most blocks_per_row of them doing useful work).
pub fn sgemv_q4_k_f32_quadcoop_into(
    backend: &MetalBackend,
    x_buf: &Buffer,
    w_q4k_buf: &Buffer,
    out_buf: &Buffer,
    k: usize,
    n: usize,
) -> Result<(), MetalError> {
    if !backend.supports_metal3() {
        return Err(MetalError::Unsupported(
            "sgemv_q4_k_f32_quadcoop needs Metal3".to_string(),
        ));
    }
    if k == 0 || n == 0 || k % 256 != 0 {
        return Err(MetalError::ShapeMismatch(format!(
            "sgemv_q4_k_f32_quadcoop: K%256==0 required (K={k}, N={n})"
        )));
    }
    let pipeline = backend.pipeline(
        "sgemv_q4_k_f32_quadcoop",
        SGEMV_Q4_K_F32_QUADCOOP_SHADER,
        "sgemv_q4_k_f32_quadcoop",
    )?;
    let dims = [k as u32, n as u32];
    // Number of simdgroups = ceil(N/4); one tg per simdgroup (32 threads).
    let n_groups = n.div_ceil(4) as u64;
    backend.with_encoder(|encoder| {
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(x_buf), 0);
        encoder.set_buffer(1, Some(w_q4k_buf), 0);
        encoder.set_buffer(2, Some(out_buf), 0);
        encoder.set_bytes(3, 8, dims.as_ptr() as *const std::ffi::c_void);
        let tg_size = MTLSize::new(32, 1, 1);
        let grid = MTLSize::new(32 * n_groups, 1, 1);
        encoder.dispatch_threads(grid, tg_size);
    });
    Ok(())
}

/// Simdgroup-cooperative Q4_K sgemv (K-reduction across 32 threads).
/// Best for shapes where 1-thread-per-output already saturates the
/// GPU but per-thread strided W loads are the bottleneck (huge N).
/// For small `blocks_per_row` (< 32) the simdgroup is under-utilised;
/// prefer [`sgemv_q4_k_f32_into`] then.
pub fn sgemv_q4_k_f32_simdcoop_into(
    backend: &MetalBackend,
    x_buf: &Buffer,
    w_q4k_buf: &Buffer,
    out_buf: &Buffer,
    k: usize,
    n: usize,
) -> Result<(), MetalError> {
    if !backend.supports_metal3() {
        return Err(MetalError::Unsupported(
            "sgemv_q4_k_f32_simdcoop needs Metal3".to_string(),
        ));
    }
    if k == 0 || n == 0 || k % 256 != 0 {
        return Err(MetalError::ShapeMismatch(format!(
            "sgemv_q4_k_f32_simdcoop: K%256==0 required (K={k}, N={n})"
        )));
    }
    let pipeline = backend.pipeline(
        "sgemv_q4_k_f32_simdcoop",
        SGEMV_Q4_K_F32_SIMDCOOP_SHADER,
        "sgemv_q4_k_f32_simdcoop",
    )?;
    let dims = [k as u32, n as u32]; // T82 push-constants
    backend.with_encoder(|encoder| {
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(x_buf), 0);
        encoder.set_buffer(1, Some(w_q4k_buf), 0);
        encoder.set_buffer(2, Some(out_buf), 0);
        encoder.set_bytes(3, 8, dims.as_ptr() as *const std::ffi::c_void);
        let tg_size = MTLSize::new(32, 1, 1);
        let grid = MTLSize::new(32 * n as u64, 1, 1);
        encoder.dispatch_threads(grid, tg_size);
    });
    Ok(())
}

// T86 — Fused W_down sgemv + residual_add Q4_K simdcoop sgemv.
//
// First fusion attempt also tried inlining SwiGLU (silu(gate)*up computed
// inside this kernel). That regressed by 33% on Qwen3-14B because each
// output recomputes silu for every K input → with N=5120 and K=17408 the
// silu+mul cost was multiplied by N (~89M extra silu calls per layer
// vs 17408 in the standalone swiglu_f32 kernel). The regression was much
// larger than the dispatch overhead saved by fusion.
//
// This version keeps SwiGLU as a separate pre-pass (writing h into an
// intermediate buffer) and only fuses W_down + residual_add. Replaces 2
// dispatches per layer (sgemv_q*_simdcoop_into + add_inplace_f32) with 1.
//
// Inputs:
//   `h_buf`:  f32 [K] post-SwiGLU activations (filled by swiglu_f32 first)
//   `w_q4k`:  Q4_K weights for W_down [N, K] row-major
//   `y_buf`:  f32 [N] residual stream (xd_buf). In/out: y += dot(h, W[n_idx]).
//
// 1 simdgroup per output ensures no two threads write the same y[n_idx],
// so no atomic needed for the residual add.
const SGEMV_Q4_K_F32_SIMDCOOP_RESIDUAL_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

constant uint BLOCK_BYTES = 144u;
constant uint BLOCK_WEIGHTS = 256u;

kernel void sgemv_q4_k_f32_simdcoop_residual(
    device const float* x       [[buffer(0)]],   // [K] post-SwiGLU h
    device const uchar* w_q4k   [[buffer(1)]],   // [N, K] W_down Q4_K row-major
    device float* y             [[buffer(2)]],   // [N] residual stream (in/out)
    constant uint2& dims        [[buffer(3)]],   // (K, N)
    uint tg_id                  [[threadgroup_position_in_grid]],
    uint tid                    [[thread_position_in_threadgroup]],
    uint sg_size                [[threads_per_simdgroup]]
) {
    uint K = dims.x;
    uint N = dims.y;
    uint n_idx = tg_id;
    if (n_idx >= N) return;

    uint blocks_per_row = K / BLOCK_WEIGHTS;
    uint row_off = n_idx * blocks_per_row * BLOCK_BYTES;

    float partial = 0.0;

    for (uint blk = tid; blk < blocks_per_row; blk += sg_size) {
        device const uchar* block = w_q4k + row_off + blk * BLOCK_BYTES;
        ushort d_bits = ((ushort)block[1] << 8) | (ushort)block[0];
        ushort dmin_bits = ((ushort)block[3] << 8) | (ushort)block[2];
        float d = float(as_type<half>(d_bits));
        float dmin = float(as_type<half>(dmin_bits));
        uchar packed[12];
        for (uint i = 0; i < 12u; ++i) packed[i] = block[4 + i];
        uchar sc[8], m[8];
        for (uint i = 0; i < 4u; ++i) {
            sc[i]     = packed[i] & 0x3F;
            m[i]      = packed[i + 4] & 0x3F;
            sc[i + 4] = (packed[i + 8] & 0x0F) | ((packed[i] >> 6) << 4);
            m[i + 4]  = (packed[i + 8] >> 4)   | ((packed[i + 4] >> 6) << 4);
        }
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
            uint qs_base = jp * 8u;
            for (uint kg = 0; kg < 8u; ++kg) {
                uchar4 nibs = qs4[qs_base + kg];
                uint kk = kg * 4u;
                float n0lo = scale0 * float(nibs.x & 0x0F) - min0;
                float n1lo = scale0 * float(nibs.y & 0x0F) - min0;
                float n2lo = scale0 * float(nibs.z & 0x0F) - min0;
                float n3lo = scale0 * float(nibs.w & 0x0F) - min0;
                float n0hi = scale1 * float(nibs.x >> 4)   - min1;
                float n1hi = scale1 * float(nibs.y >> 4)   - min1;
                float n2hi = scale1 * float(nibs.z >> 4)   - min1;
                float n3hi = scale1 * float(nibs.w >> 4)   - min1;
                partial += x[x_low_off  + kk    ] * n0lo;
                partial += x[x_low_off  + kk + 1] * n1lo;
                partial += x[x_low_off  + kk + 2] * n2lo;
                partial += x[x_low_off  + kk + 3] * n3lo;
                partial += x[x_high_off + kk    ] * n0hi;
                partial += x[x_high_off + kk + 1] * n1hi;
                partial += x[x_high_off + kk + 2] * n2hi;
                partial += x[x_high_off + kk + 3] * n3hi;
            }
        }
    }

    float total = simd_sum(partial);
    if (tid == 0) {
        // Fused residual add: y[n_idx] already holds x + W_O@attn from
        // earlier in the layer; we add the FFN tail contribution onto it.
        y[n_idx] = y[n_idx] + total;
    }
}
"#;

/// T86 — Fused W_down sgemv + residual add (Q4_K simdcoop). Replaces 2
/// dispatches (sgemv_q4_k_f32_simdcoop_into + add_inplace_f32) with 1.
/// SwiGLU still runs as a separate pre-pass into `h_buf`.
pub fn sgemv_q4_k_f32_simdcoop_residual_into(
    backend: &MetalBackend,
    h_buf: &Buffer,
    w_q4k_buf: &Buffer,
    y_buf: &Buffer,
    k: usize,
    n: usize,
) -> Result<(), MetalError> {
    if !backend.supports_metal3() {
        return Err(MetalError::Unsupported(
            "sgemv_q4_k_f32_simdcoop_residual needs Metal3".to_string(),
        ));
    }
    if k == 0 || n == 0 || k % 256 != 0 {
        return Err(MetalError::ShapeMismatch(format!(
            "sgemv_q4_k_f32_simdcoop_residual: K%256==0 required (K={k}, N={n})"
        )));
    }
    let pipeline = backend.pipeline(
        "sgemv_q4_k_f32_simdcoop_residual",
        SGEMV_Q4_K_F32_SIMDCOOP_RESIDUAL_SHADER,
        "sgemv_q4_k_f32_simdcoop_residual",
    )?;
    let dims = [k as u32, n as u32];
    backend.with_encoder(|encoder| {
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(h_buf), 0);
        encoder.set_buffer(1, Some(w_q4k_buf), 0);
        encoder.set_buffer(2, Some(y_buf), 0);
        encoder.set_bytes(3, 8, dims.as_ptr() as *const std::ffi::c_void);
        let tg_size = MTLSize::new(32, 1, 1);
        let grid = MTLSize::new(32 * n as u64, 1, 1);
        encoder.dispatch_threads(grid, tg_size);
    });
    Ok(())
}

// Same idea for Q6_K.
const SGEMV_Q6_K_F32_SIMDCOOP_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

constant uint Q6K_BYTES = 210u;
constant uint Q6K_WEIGHTS = 256u;

kernel void sgemv_q6_k_f32_simdcoop(
    device const float* x       [[buffer(0)]],
    device const uchar* w_q6k   [[buffer(1)]],
    device float* y             [[buffer(2)]],
    constant uint2& dims        [[buffer(3)]],
    uint tg_id                  [[threadgroup_position_in_grid]],
    uint tid                    [[thread_position_in_threadgroup]],
    uint sg_size                [[threads_per_simdgroup]]
) {
    uint K = dims.x;
    uint N = dims.y;
    uint n_idx = tg_id;
    if (n_idx >= N) return;

    uint blocks_per_row = K / Q6K_WEIGHTS;
    uint row_off = n_idx * blocks_per_row * Q6K_BYTES;

    float partial = 0.0;

    for (uint blk = tid; blk < blocks_per_row; blk += sg_size) {
        device const uchar* block = w_q6k + row_off + blk * Q6K_BYTES;
        device const uchar* ql = block;
        device const uchar* qh = block + 128;
        device const char*  sc = (device const char*)(block + 192);
        ushort d_bits = ((ushort)block[209] << 8) | (ushort)block[208];
        float d = float(as_type<half>(d_bits));

        for (uint half_idx = 0u; half_idx < 2u; ++half_idx) {
            device const uchar* ql_h = ql + half_idx * 64u;
            device const uchar* qh_h = qh + half_idx * 32u;
            device const char*  sc_h = sc + half_idx * 8;
            uint x_h_off = blk * Q6K_WEIGHTS + half_idx * 128u;

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
                partial += x[x_h_off + l]      * (s1 * float(q1 - 32));
                partial += x[x_h_off + l + 32] * (s2 * float(q2 - 32));
                partial += x[x_h_off + l + 64] * (s3 * float(q3 - 32));
                partial += x[x_h_off + l + 96] * (s4 * float(q4 - 32));
            }
        }
    }

    float total = simd_sum(partial);
    if (tid == 0) {
        y[n_idx] = total;
    }
}
"#;

// T93 — Faithful port of llama.cpp's kernel_mul_mv_q6_K_f32_impl with
// N_R0_Q6_K = 2. Differences from T90 (which regressed -23%):
//
//   1. Smaller per-row state (16 floats yl + 4 sums + 1 d_val per row)
//      vs T90 (8 scales × 2 rows + 4 quants × 2 rows + per-row partials).
//   2. Simpler per-row formula: dh[0] * (sums[0]*sc[0] + sums[1]*sc[2]
//      + sums[2]*sc[4] + sums[3]*sc[6]) — single FMA chain per row.
//   3. Different threading: tid=tiisg/2, ix=tiisg%2 → 16 (ip,il) pairs
//      cover the block, 2 ix partitions split K work.
//   4. Per-row pointer increments via direct (nrow,i) addressing, no
//      block-of-row state duplication.
//
// For Qwen3-14B W_down (K=17408, N=5120) — biggest decode stage (~15%).
const SGEMV_Q6_K_F32_LCPP_NR2_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

constant uint Q6K_BYTES = 210u;
constant uint Q6K_WEIGHTS = 256u;
constant short NR0_Q6 = 2;

kernel void sgemv_q6_k_f32_lcpp_nr2(
    device const float*  x      [[buffer(0)]],
    device const uchar*  w_q6k  [[buffer(1)]],
    device float*        y      [[buffer(2)]],
    constant uint2&      dims   [[buffer(3)]],
    uint                 tg_id  [[threadgroup_position_in_grid]],
    ushort               tiisg  [[thread_index_in_simdgroup]]
) {
    constexpr uchar KMASK1 = 0x03;
    constexpr uchar KMASK2 = 0x0C;
    constexpr uchar KMASK3 = 0x30;
    constexpr uchar KMASK4 = 0xC0;

    uint K = dims.x;
    uint N = dims.y;
    int nb = (int)(K / Q6K_WEIGHTS);

    uint first_row = tg_id * (uint)NR0_Q6;
    if (first_row >= N) return;

    short tid = (short)(tiisg / 2u);   // 0..15
    short ix  = (short)(tiisg % 2u);   // 0 or 1
    short ip  = tid / 8;                // 0 or 1
    short il  = tid % 8;                // 0..7
    short l0  = 4 * il;
    short is  = 8 * ip + l0 / 16;

    short y_offset   = 128 * ip + l0;
    short q_offset_l = 64 * ip + l0;
    short q_offset_h = 32 * ip + l0;

    float sumf[2] = {0.0, 0.0};
    float yl[16];

    uint row_stride = (uint)nb * Q6K_BYTES;

    for (int i = ix; i < nb; i += 2) {
        device const float* yptr = x + i * (int)Q6K_WEIGHTS + (int)y_offset;
        for (short l = 0; l < 4; ++l) {
            yl[4*l + 0] = yptr[l +  0];
            yl[4*l + 1] = yptr[l + 32];
            yl[4*l + 2] = yptr[l + 64];
            yl[4*l + 3] = yptr[l + 96];
        }

        for (short row = 0; row < NR0_Q6; ++row) {
            uint nrow = first_row + (uint)row;
            if (nrow >= N) continue;

            device const uchar* block = w_q6k + (uint64_t)nrow * row_stride + (uint)i * Q6K_BYTES;
            device const uchar* q1 = block + 0   + (uint)q_offset_l;
            device const uchar* q2 = q1 + 32;
            device const uchar* qh = block + 128 + (uint)q_offset_h;
            device const char*  sc = (device const char*)(block + 192) + (int)is;
            device const uint16_t* dh = (device const uint16_t*)(block + 208);

            float d_val = float(as_type<half>(dh[0]));

            float4 sums = {0.0, 0.0, 0.0, 0.0};
            for (short l = 0; l < 4; ++l) {
                sums[0] += yl[4*l + 0] * (float)((int)((q1[l] & 0xF) | ((qh[l] & KMASK1) << 4)) - 32);
                sums[1] += yl[4*l + 1] * (float)((int)((q2[l] & 0xF) | ((qh[l] & KMASK2) << 2)) - 32);
                sums[2] += yl[4*l + 2] * (float)((int)((q1[l]  >> 4) | ((qh[l] & KMASK3) << 0)) - 32);
                sums[3] += yl[4*l + 3] * (float)((int)((q2[l]  >> 4) | ((qh[l] & KMASK4) >> 2)) - 32);
            }

            sumf[row] += d_val * (sums[0] * float(sc[0]) + sums[1] * float(sc[2])
                                + sums[2] * float(sc[4]) + sums[3] * float(sc[6]));
        }
    }

    for (short row = 0; row < NR0_Q6; ++row) {
        uint nrow = first_row + (uint)row;
        if (nrow >= N) break;
        float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) {
            y[nrow] = sum_all;
        }
    }
}
"#;

/// T93 — Faithful port of llama.cpp's `kernel_mul_mv_q6_K_f32_impl` with
/// N_R0_Q6_K=2. Smaller per-row state than T90 (which regressed); 2 rows
/// per simdgroup with manageable register pressure.
pub fn sgemv_q6_k_f32_lcpp_nr2_into(
    backend: &MetalBackend,
    x_buf: &Buffer,
    w_q6k_buf: &Buffer,
    out_buf: &Buffer,
    k: usize,
    n: usize,
) -> Result<(), MetalError> {
    if !backend.supports_metal3() {
        return Err(MetalError::Unsupported(
            "sgemv_q6_k_f32_lcpp_nr2 needs Metal3".to_string(),
        ));
    }
    if k == 0 || n == 0 || k % 256 != 0 {
        return Err(MetalError::ShapeMismatch(format!(
            "sgemv_q6_k_f32_lcpp_nr2: K%256==0 required (K={k}, N={n})"
        )));
    }
    let pipeline = backend.pipeline(
        "sgemv_q6_k_f32_lcpp_nr2",
        SGEMV_Q6_K_F32_LCPP_NR2_SHADER,
        "sgemv_q6_k_f32_lcpp_nr2",
    )?;
    let dims = [k as u32, n as u32];
    backend.with_encoder(|encoder| {
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(x_buf), 0);
        encoder.set_buffer(1, Some(w_q6k_buf), 0);
        encoder.set_buffer(2, Some(out_buf), 0);
        encoder.set_bytes(3, 8, dims.as_ptr() as *const std::ffi::c_void);
        let tg_size = MTLSize::new(32, 1, 1);
        let n_sg = (n as u64).div_ceil(2);
        let grid = MTLSize::new(32 * n_sg, 1, 1);
        encoder.dispatch_threads(grid, tg_size);
    });
    Ok(())
}

// T90 (legacy / reverted) — Multi-row Q6_K simdcoop sgemv (2 rows per
// simdgroup). Same as the single-row Q6_K simdcoop but processes 2
// output rows per simdgroup, sharing the x-tile reads across them.
// REGRESSED -23% on Qwen3-14B (likely register spill). Kept for reference.
const SGEMV_Q6_K_F32_SIMDCOOP_NR2_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

constant uint Q6K_BYTES = 210u;
constant uint Q6K_WEIGHTS = 256u;

kernel void sgemv_q6_k_f32_simdcoop_nr2(
    device const float* x       [[buffer(0)]],
    device const uchar* w_q6k   [[buffer(1)]],
    device float* y             [[buffer(2)]],
    constant uint2& dims        [[buffer(3)]],
    uint tg_id                  [[threadgroup_position_in_grid]],
    uint tid                    [[thread_position_in_threadgroup]],
    uint sg_size                [[threads_per_simdgroup]]
) {
    uint K = dims.x;
    uint N = dims.y;
    uint n_idx_base = tg_id * 2u;
    if (n_idx_base >= N) return;
    bool has_row1 = (n_idx_base + 1u) < N;

    uint blocks_per_row = K / Q6K_WEIGHTS;
    uint row_off_0 = n_idx_base * blocks_per_row * Q6K_BYTES;
    uint row_off_1 = (n_idx_base + 1u) * blocks_per_row * Q6K_BYTES;

    float partial0 = 0.0;
    float partial1 = 0.0;

    for (uint blk = tid; blk < blocks_per_row; blk += sg_size) {
        // Row 0 block
        device const uchar* block0 = w_q6k + row_off_0 + blk * Q6K_BYTES;
        device const uchar* ql0 = block0;
        device const uchar* qh0 = block0 + 128;
        device const char*  sc0 = (device const char*)(block0 + 192);
        ushort d_bits0 = ((ushort)block0[209] << 8) | (ushort)block0[208];
        float d0 = float(as_type<half>(d_bits0));

        // Row 1 block (if exists)
        device const uchar* block1 = w_q6k + row_off_1 + blk * Q6K_BYTES;
        device const uchar* ql1 = block1;
        device const uchar* qh1 = block1 + 128;
        device const char*  sc1 = (device const char*)(block1 + 192);
        ushort d_bits1 = has_row1 ? (((ushort)block1[209] << 8) | (ushort)block1[208]) : 0u;
        float d1 = has_row1 ? float(as_type<half>(d_bits1)) : 0.0f;

        for (uint half_idx = 0u; half_idx < 2u; ++half_idx) {
            device const uchar* ql_h0 = ql0 + half_idx * 64u;
            device const uchar* qh_h0 = qh0 + half_idx * 32u;
            device const char*  sc_h0 = sc0 + half_idx * 8;

            device const uchar* ql_h1 = ql1 + half_idx * 64u;
            device const uchar* qh_h1 = qh1 + half_idx * 32u;
            device const char*  sc_h1 = sc1 + half_idx * 8;

            uint x_h_off = blk * Q6K_WEIGHTS + half_idx * 128u;

            float s1_lo_0 = d0 * float(sc_h0[0]);
            float s1_hi_0 = d0 * float(sc_h0[1]);
            float s2_lo_0 = d0 * float(sc_h0[2]);
            float s2_hi_0 = d0 * float(sc_h0[3]);
            float s3_lo_0 = d0 * float(sc_h0[4]);
            float s3_hi_0 = d0 * float(sc_h0[5]);
            float s4_lo_0 = d0 * float(sc_h0[6]);
            float s4_hi_0 = d0 * float(sc_h0[7]);

            float s1_lo_1 = d1 * float(sc_h1[0]);
            float s1_hi_1 = d1 * float(sc_h1[1]);
            float s2_lo_1 = d1 * float(sc_h1[2]);
            float s2_hi_1 = d1 * float(sc_h1[3]);
            float s3_lo_1 = d1 * float(sc_h1[4]);
            float s3_hi_1 = d1 * float(sc_h1[5]);
            float s4_lo_1 = d1 * float(sc_h1[6]);
            float s4_hi_1 = d1 * float(sc_h1[7]);

            for (uint l = 0; l < 32u; ++l) {
                // Row 0 quants
                uchar qhh0 = qh_h0[l];
                int q1_0 = (int)(ql_h0[l]      & 0x0F) | ((int)((qhh0 >> 0) & 0x03) << 4);
                int q2_0 = (int)(ql_h0[l + 32] & 0x0F) | ((int)((qhh0 >> 2) & 0x03) << 4);
                int q3_0 = (int)(ql_h0[l]      >> 4)   | ((int)((qhh0 >> 4) & 0x03) << 4);
                int q4_0 = (int)(ql_h0[l + 32] >> 4)   | ((int)((qhh0 >> 6) & 0x03) << 4);

                // Row 1 quants
                uchar qhh1 = qh_h1[l];
                int q1_1 = (int)(ql_h1[l]      & 0x0F) | ((int)((qhh1 >> 0) & 0x03) << 4);
                int q2_1 = (int)(ql_h1[l + 32] & 0x0F) | ((int)((qhh1 >> 2) & 0x03) << 4);
                int q3_1 = (int)(ql_h1[l]      >> 4)   | ((int)((qhh1 >> 4) & 0x03) << 4);
                int q4_1 = (int)(ql_h1[l + 32] >> 4)   | ((int)((qhh1 >> 6) & 0x03) << 4);

                float s1_0 = (l < 16u) ? s1_lo_0 : s1_hi_0;
                float s2_0 = (l < 16u) ? s2_lo_0 : s2_hi_0;
                float s3_0 = (l < 16u) ? s3_lo_0 : s3_hi_0;
                float s4_0 = (l < 16u) ? s4_lo_0 : s4_hi_0;

                float s1_1 = (l < 16u) ? s1_lo_1 : s1_hi_1;
                float s2_1 = (l < 16u) ? s2_lo_1 : s2_hi_1;
                float s3_1 = (l < 16u) ? s3_lo_1 : s3_hi_1;
                float s4_1 = (l < 16u) ? s4_lo_1 : s4_hi_1;

                // Shared x reads
                float xv1 = x[x_h_off + l];
                float xv2 = x[x_h_off + l + 32];
                float xv3 = x[x_h_off + l + 64];
                float xv4 = x[x_h_off + l + 96];

                partial0 += xv1 * (s1_0 * float(q1_0 - 32));
                partial0 += xv2 * (s2_0 * float(q2_0 - 32));
                partial0 += xv3 * (s3_0 * float(q3_0 - 32));
                partial0 += xv4 * (s4_0 * float(q4_0 - 32));

                partial1 += xv1 * (s1_1 * float(q1_1 - 32));
                partial1 += xv2 * (s2_1 * float(q2_1 - 32));
                partial1 += xv3 * (s3_1 * float(q3_1 - 32));
                partial1 += xv4 * (s4_1 * float(q4_1 - 32));
            }
        }
    }

    float total0 = simd_sum(partial0);
    float total1 = simd_sum(partial1);
    if (tid == 0) {
        y[n_idx_base] = total0;
        if (has_row1) {
            y[n_idx_base + 1u] = total1;
        }
    }
}
"#;

/// T90 — Multi-row Q6_K simdcoop sgemv (2 rows per simdgroup).
/// For Qwen3-14B W_down (K=17408, Q6_K) where x bandwidth dominates.
pub fn sgemv_q6_k_f32_simdcoop_nr2_into(
    backend: &MetalBackend,
    x_buf: &Buffer,
    w_q6k_buf: &Buffer,
    out_buf: &Buffer,
    k: usize,
    n: usize,
) -> Result<(), MetalError> {
    if !backend.supports_metal3() {
        return Err(MetalError::Unsupported(
            "sgemv_q6_k_f32_simdcoop_nr2 needs Metal3".to_string(),
        ));
    }
    if k == 0 || n == 0 || k % 256 != 0 {
        return Err(MetalError::ShapeMismatch(format!(
            "sgemv_q6_k_f32_simdcoop_nr2: K%256==0 required (K={k}, N={n})"
        )));
    }
    let pipeline = backend.pipeline(
        "sgemv_q6_k_f32_simdcoop_nr2",
        SGEMV_Q6_K_F32_SIMDCOOP_NR2_SHADER,
        "sgemv_q6_k_f32_simdcoop_nr2",
    )?;
    let dims = [k as u32, n as u32];
    backend.with_encoder(|encoder| {
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(x_buf), 0);
        encoder.set_buffer(1, Some(w_q6k_buf), 0);
        encoder.set_buffer(2, Some(out_buf), 0);
        encoder.set_bytes(3, 8, dims.as_ptr() as *const std::ffi::c_void);
        let tg_size = MTLSize::new(32, 1, 1);
        let n_sg = (n as u64).div_ceil(2);
        let grid = MTLSize::new(32 * n_sg, 1, 1);
        encoder.dispatch_threads(grid, tg_size);
    });
    Ok(())
}

/// Simdgroup-cooperative Q6_K sgemv. Companion to
/// [`sgemv_q4_k_f32_simdcoop_into`]. Best for huge N (lm_head).
pub fn sgemv_q6_k_f32_simdcoop_into(
    backend: &MetalBackend,
    x_buf: &Buffer,
    w_q6k_buf: &Buffer,
    out_buf: &Buffer,
    k: usize,
    n: usize,
) -> Result<(), MetalError> {
    if !backend.supports_metal3() {
        return Err(MetalError::Unsupported(
            "sgemv_q6_k_f32_simdcoop needs Metal3".to_string(),
        ));
    }
    if k == 0 || n == 0 || k % 256 != 0 {
        return Err(MetalError::ShapeMismatch(format!(
            "sgemv_q6_k_f32_simdcoop: K%256==0 required (K={k}, N={n})"
        )));
    }
    let pipeline = backend.pipeline(
        "sgemv_q6_k_f32_simdcoop",
        SGEMV_Q6_K_F32_SIMDCOOP_SHADER,
        "sgemv_q6_k_f32_simdcoop",
    )?;
    let dims = [k as u32, n as u32]; // T82
    backend.with_encoder(|encoder| {
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(x_buf), 0);
        encoder.set_buffer(1, Some(w_q6k_buf), 0);
        encoder.set_buffer(2, Some(out_buf), 0);
        encoder.set_bytes(3, 8, dims.as_ptr() as *const std::ffi::c_void);
        let tg_size = MTLSize::new(32, 1, 1);
        let grid = MTLSize::new(32 * n as u64, 1, 1);
        encoder.dispatch_threads(grid, tg_size);
    });
    Ok(())
}

// T86 — Fused W_down + residual_add Q6_K simdcoop sgemv. Companion to
// the Q4_K version above. W_down in Qwen3-14B Q4_K_M is systematically
// Q6_K (210 bytes/super-block). Same fused semantics: y[n_idx] += dot(h, W_q6k[n_idx]).
const SGEMV_Q6_K_F32_SIMDCOOP_RESIDUAL_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

constant uint Q6K_BYTES = 210u;
constant uint Q6K_WEIGHTS = 256u;

kernel void sgemv_q6_k_f32_simdcoop_residual(
    device const float* x       [[buffer(0)]],   // [K] post-SwiGLU h
    device const uchar* w_q6k   [[buffer(1)]],   // [N, K] W_down Q6_K row-major
    device float* y             [[buffer(2)]],   // [N] residual stream (in/out)
    constant uint2& dims        [[buffer(3)]],   // (K, N)
    uint tg_id                  [[threadgroup_position_in_grid]],
    uint tid                    [[thread_position_in_threadgroup]],
    uint sg_size                [[threads_per_simdgroup]]
) {
    uint K = dims.x;
    uint N = dims.y;
    uint n_idx = tg_id;
    if (n_idx >= N) return;

    uint blocks_per_row = K / Q6K_WEIGHTS;
    uint row_off = n_idx * blocks_per_row * Q6K_BYTES;

    float partial = 0.0;

    for (uint blk = tid; blk < blocks_per_row; blk += sg_size) {
        device const uchar* block = w_q6k + row_off + blk * Q6K_BYTES;
        device const uchar* ql = block;
        device const uchar* qh = block + 128;
        device const char*  sc = (device const char*)(block + 192);
        ushort d_bits = ((ushort)block[209] << 8) | (ushort)block[208];
        float d = float(as_type<half>(d_bits));

        for (uint half_idx = 0u; half_idx < 2u; ++half_idx) {
            device const uchar* ql_h = ql + half_idx * 64u;
            device const uchar* qh_h = qh + half_idx * 32u;
            device const char*  sc_h = sc + half_idx * 8;
            uint x_h_off = blk * Q6K_WEIGHTS + half_idx * 128u;

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
                partial += x[x_h_off + l]      * (s1 * float(q1 - 32));
                partial += x[x_h_off + l + 32] * (s2 * float(q2 - 32));
                partial += x[x_h_off + l + 64] * (s3 * float(q3 - 32));
                partial += x[x_h_off + l + 96] * (s4 * float(q4 - 32));
            }
        }
    }

    float total = simd_sum(partial);
    if (tid == 0) {
        y[n_idx] = y[n_idx] + total;
    }
}
"#;

/// T86 — Q6_K variant of the fused W_down sgemv + residual add.
/// Used when W_down is Q6_K (typical of Q4_K_M format).
pub fn sgemv_q6_k_f32_simdcoop_residual_into(
    backend: &MetalBackend,
    h_buf: &Buffer,
    w_q6k_buf: &Buffer,
    y_buf: &Buffer,
    k: usize,
    n: usize,
) -> Result<(), MetalError> {
    if !backend.supports_metal3() {
        return Err(MetalError::Unsupported(
            "sgemv_q6_k_f32_simdcoop_residual needs Metal3".to_string(),
        ));
    }
    if k == 0 || n == 0 || k % 256 != 0 {
        return Err(MetalError::ShapeMismatch(format!(
            "sgemv_q6_k_f32_simdcoop_residual: K%256==0 required (K={k}, N={n})"
        )));
    }
    let pipeline = backend.pipeline(
        "sgemv_q6_k_f32_simdcoop_residual",
        SGEMV_Q6_K_F32_SIMDCOOP_RESIDUAL_SHADER,
        "sgemv_q6_k_f32_simdcoop_residual",
    )?;
    let dims = [k as u32, n as u32];
    backend.with_encoder(|encoder| {
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(h_buf), 0);
        encoder.set_buffer(1, Some(w_q6k_buf), 0);
        encoder.set_buffer(2, Some(y_buf), 0);
        encoder.set_bytes(3, 8, dims.as_ptr() as *const std::ffi::c_void);
        let tg_size = MTLSize::new(32, 1, 1);
        let grid = MTLSize::new(32 * n as u64, 1, 1);
        encoder.dispatch_threads(grid, tg_size);
    });
    Ok(())
}

// Fused Q+K+V Q4_K sgemv into a single contiguous output buffer
// `[n_q + n_k + n_v]`. One kernel dispatch instead of 3 — eliminates
// 2 encoder creations per layer × 40 layers = 80 encoder ops per
// token. Each thread maps to one output column index in the combined
// `[Q | K | V]` output and selects the matching weight matrix
// (w_q | w_k | w_v) based on its position.
//
// The shared input `x` is read once per thread (vs 3× before),
// although the dominant reads are the per-thread W bytes which are
// unchanged.
const SGEMV_Q4_K_F32_TRIPLE_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

constant uint BLOCK_BYTES = 144u;
constant uint BLOCK_WEIGHTS = 256u;

kernel void sgemv_q4_k_f32_triple(
    device const float* x       [[buffer(0)]],
    device const uchar* w_q     [[buffer(1)]],
    device const uchar* w_k     [[buffer(2)]],
    device const uchar* w_v     [[buffer(3)]],
    device float* out_q         [[buffer(4)]],
    device float* out_k         [[buffer(5)]],
    device float* out_v         [[buffer(6)]],
    constant uint4& dims        [[buffer(7)]],   // (K, n_q, n_k, n_v)
    uint gid                    [[thread_position_in_grid]]
) {
    uint K   = dims.x;
    uint n_q = dims.y;
    uint n_k = dims.z;
    uint n_v = dims.w;
    uint total = n_q + n_k + n_v;
    if (gid >= total) return;

    device const uchar* w;
    device float* out;
    uint n_idx;
    uint n_dim;
    if (gid < n_q) {
        w = w_q;
        out = out_q;
        n_idx = gid;
        n_dim = n_q;
    } else if (gid < n_q + n_k) {
        w = w_k;
        out = out_k;
        n_idx = gid - n_q;
        n_dim = n_k;
    } else {
        w = w_v;
        out = out_v;
        n_idx = gid - n_q - n_k;
        n_dim = n_v;
    }
    (void)n_dim;

    uint blocks_per_row = K / BLOCK_WEIGHTS;
    uint row_off = n_idx * blocks_per_row * BLOCK_BYTES;

    float acc = 0.0;

    for (uint blk = 0; blk < blocks_per_row; ++blk) {
        device const uchar* block = w + row_off + blk * BLOCK_BYTES;
        ushort d_bits = ((ushort)block[1] << 8) | (ushort)block[0];
        ushort dmin_bits = ((ushort)block[3] << 8) | (ushort)block[2];
        float d = float(as_type<half>(d_bits));
        float dmin = float(as_type<half>(dmin_bits));
        uchar packed[12];
        for (uint i = 0; i < 12u; ++i) packed[i] = block[4 + i];
        uchar sc[8], m[8];
        for (uint i = 0; i < 4u; ++i) {
            sc[i]     = packed[i] & 0x3F;
            m[i]      = packed[i + 4] & 0x3F;
            sc[i + 4] = (packed[i + 8] & 0x0F) | ((packed[i] >> 6) << 4);
            m[i + 4]  = (packed[i + 8] >> 4)   | ((packed[i + 4] >> 6) << 4);
        }
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
            uint qs_base = jp * 8u;
            for (uint kg = 0; kg < 8u; ++kg) {
                uchar4 nibs = qs4[qs_base + kg];
                uint kk = kg * 4u;
                float n0lo = scale0 * float(nibs.x & 0x0F) - min0;
                float n1lo = scale0 * float(nibs.y & 0x0F) - min0;
                float n2lo = scale0 * float(nibs.z & 0x0F) - min0;
                float n3lo = scale0 * float(nibs.w & 0x0F) - min0;
                float n0hi = scale1 * float(nibs.x >> 4)   - min1;
                float n1hi = scale1 * float(nibs.y >> 4)   - min1;
                float n2hi = scale1 * float(nibs.z >> 4)   - min1;
                float n3hi = scale1 * float(nibs.w >> 4)   - min1;
                acc += x[x_low_off  + kk    ] * n0lo;
                acc += x[x_low_off  + kk + 1] * n1lo;
                acc += x[x_low_off  + kk + 2] * n2lo;
                acc += x[x_low_off  + kk + 3] * n3lo;
                acc += x[x_high_off + kk    ] * n0hi;
                acc += x[x_high_off + kk + 1] * n1hi;
                acc += x[x_high_off + kk + 2] * n2hi;
                acc += x[x_high_off + kk + 3] * n3hi;
            }
        }
    }

    out[n_idx] = acc;
}
"#;

/// Fused Q+K+V Q4_K sgemv: dispatch ONCE for all three projections
/// reading the same `x`. Eliminates 2 encoder/dispatch overheads per
/// attention layer (~5-10us each on Apple GPU).
#[allow(clippy::too_many_arguments)]
pub fn sgemv_q4_k_f32_triple_into(
    backend: &MetalBackend,
    x_buf: &Buffer,
    w_q_buf: &Buffer,
    w_k_buf: &Buffer,
    w_v_buf: &Buffer,
    out_q: &Buffer,
    out_k: &Buffer,
    out_v: &Buffer,
    k: usize,
    n_q: usize,
    n_k: usize,
    n_v: usize,
) -> Result<(), MetalError> {
    if !backend.supports_metal3() {
        return Err(MetalError::Unsupported(
            "sgemv_q4_k_f32_triple needs Metal3".to_string(),
        ));
    }
    if k % 256 != 0 {
        return Err(MetalError::ShapeMismatch(format!(
            "sgemv_q4_k_f32_triple: K%256==0 required (K={k})"
        )));
    }
    let pipeline = backend.pipeline(
        "sgemv_q4_k_f32_triple",
        SGEMV_Q4_K_F32_TRIPLE_SHADER,
        "sgemv_q4_k_f32_triple",
    )?;
    let dims = [k as u32, n_q as u32, n_k as u32, n_v as u32]; // T82
    let total = n_q + n_k + n_v;
    backend.with_encoder(|encoder| {
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(x_buf), 0);
        encoder.set_buffer(1, Some(w_q_buf), 0);
        encoder.set_buffer(2, Some(w_k_buf), 0);
        encoder.set_buffer(3, Some(w_v_buf), 0);
        encoder.set_buffer(4, Some(out_q), 0);
        encoder.set_buffer(5, Some(out_k), 0);
        encoder.set_buffer(6, Some(out_v), 0);
        encoder.set_bytes(7, 16, dims.as_ptr() as *const std::ffi::c_void);
        let tg_size = MTLSize::new(64, 1, 1);
        let grid = MTLSize::new(total as u64, 1, 1);
        encoder.dispatch_threads(grid, tg_size);
    });
    Ok(())
}

/// Fused gate+up Q4_K sgemv: same idea, dispatch ONCE for both
/// projections reading the same `x`.
#[allow(clippy::too_many_arguments)]
pub fn sgemv_q4_k_f32_pair_into(
    backend: &MetalBackend,
    x_buf: &Buffer,
    w_a_buf: &Buffer,
    w_b_buf: &Buffer,
    out_a: &Buffer,
    out_b: &Buffer,
    k: usize,
    n_a: usize,
    n_b: usize,
) -> Result<(), MetalError> {
    // Reuse the triple shader by passing w_b as both K and V slots,
    // dispatching n_a + n_b threads, but routing the V slot to a
    // dummy. Cleaner: fake third matrix by aliasing one of them.
    // Simpler still: just call triple with n_v=0 → it works because
    // total = n_q + n_k + 0. The shader's `gid >= total` skips the
    // V-index path entirely.
    sgemv_q4_k_f32_triple_into(
        backend, x_buf, w_a_buf, w_b_buf, w_b_buf, out_a, out_b, out_b, k, n_a, n_b, 0,
    )
}

// T84 — quad-cooperative fused triple. 4 outputs per simdgroup, each
// possibly drawn from a different (W, out) slot. Routing is done
// inside each quarter independently (8 threads pick the same target
// since they share the same n_idx). Reduces wasted threads when
// blocks_per_row is small (e.g. Qwen3-14B Q4_K with bpr=20).
const SGEMV_Q4_K_F32_TRIPLE_QUADCOOP_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

constant uint BLOCK_BYTES = 144u;
constant uint BLOCK_WEIGHTS = 256u;

kernel void sgemv_q4_k_f32_triple_quadcoop(
    device const float* x       [[buffer(0)]],
    device const uchar* w_q     [[buffer(1)]],
    device const uchar* w_k     [[buffer(2)]],
    device const uchar* w_v     [[buffer(3)]],
    device float* out_q         [[buffer(4)]],
    device float* out_k         [[buffer(5)]],
    device float* out_v         [[buffer(6)]],
    constant uint4& dims        [[buffer(7)]],   // (K, n_q, n_k, n_v)
    uint tg_id                  [[threadgroup_position_in_grid]],
    uint tid                    [[thread_position_in_threadgroup]]
) {
    uint K   = dims.x;
    uint n_q = dims.y;
    uint n_k = dims.z;
    uint n_v = dims.w;
    uint total = n_q + n_k + n_v;
    uint base = tg_id * 4u;
    uint quarter = tid / 8u;
    uint lane_q  = tid % 8u;
    uint global_idx = base + quarter;

    // Route this quarter to its (W, out, n_idx) tuple.
    device const uchar* w = w_q;
    device float* out = out_q;
    uint n_idx = 0u;
    bool active = global_idx < total;
    if (active) {
        if (global_idx < n_q) {
            w = w_q;
            out = out_q;
            n_idx = global_idx;
        } else if (global_idx < n_q + n_k) {
            w = w_k;
            out = out_k;
            n_idx = global_idx - n_q;
        } else {
            w = w_v;
            out = out_v;
            n_idx = global_idx - n_q - n_k;
        }
    }

    uint blocks_per_row = K / BLOCK_WEIGHTS;
    uint row_off = active ? n_idx * blocks_per_row * BLOCK_BYTES : 0u;

    float partial = 0.0;
    if (active) {
        for (uint blk = lane_q; blk < blocks_per_row; blk += 8u) {
            device const uchar* block = w + row_off + blk * BLOCK_BYTES;
            ushort d_bits = ((ushort)block[1] << 8) | (ushort)block[0];
            ushort dmin_bits = ((ushort)block[3] << 8) | (ushort)block[2];
            float d = float(as_type<half>(d_bits));
            float dmin = float(as_type<half>(dmin_bits));
            uchar packed[12];
            for (uint i = 0; i < 12u; ++i) packed[i] = block[4 + i];
            uchar sc[8], m[8];
            for (uint i = 0; i < 4u; ++i) {
                sc[i]     = packed[i] & 0x3F;
                m[i]      = packed[i + 4] & 0x3F;
                sc[i + 4] = (packed[i + 8] & 0x0F) | ((packed[i] >> 6) << 4);
                m[i + 4]  = (packed[i + 8] >> 4)   | ((packed[i + 4] >> 6) << 4);
            }
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
                uint qs_base = jp * 8u;
                for (uint kg = 0; kg < 8u; ++kg) {
                    uchar4 nibs = qs4[qs_base + kg];
                    uint kk = kg * 4u;
                    float n0lo = scale0 * float(nibs.x & 0x0F) - min0;
                    float n1lo = scale0 * float(nibs.y & 0x0F) - min0;
                    float n2lo = scale0 * float(nibs.z & 0x0F) - min0;
                    float n3lo = scale0 * float(nibs.w & 0x0F) - min0;
                    float n0hi = scale1 * float(nibs.x >> 4)   - min1;
                    float n1hi = scale1 * float(nibs.y >> 4)   - min1;
                    float n2hi = scale1 * float(nibs.z >> 4)   - min1;
                    float n3hi = scale1 * float(nibs.w >> 4)   - min1;
                    partial += x[x_low_off  + kk    ] * n0lo;
                    partial += x[x_low_off  + kk + 1] * n1lo;
                    partial += x[x_low_off  + kk + 2] * n2lo;
                    partial += x[x_low_off  + kk + 3] * n3lo;
                    partial += x[x_high_off + kk    ] * n0hi;
                    partial += x[x_high_off + kk + 1] * n1hi;
                    partial += x[x_high_off + kk + 2] * n2hi;
                    partial += x[x_high_off + kk + 3] * n3hi;
                }
            }
        }
    }

    // 8-thread quarter reduction via XOR shuffles (mask < 8).
    float sum = partial;
    sum += simd_shuffle_xor(sum, 1);
    sum += simd_shuffle_xor(sum, 2);
    sum += simd_shuffle_xor(sum, 4);
    if (lane_q == 0u && active) {
        out[n_idx] = sum;
    }
}
"#;

/// Fused QKV (or pair via n_v=0) Q4_K sgemv with 4-output-per-simdgroup
/// quad-cooperative reduction. Best when blocks_per_row ∈ [16, 32) so
/// the simdcoop variant would waste threads.
#[allow(clippy::too_many_arguments)]
pub fn sgemv_q4_k_f32_triple_quadcoop_into(
    backend: &MetalBackend,
    x_buf: &Buffer,
    w_q_buf: &Buffer,
    w_k_buf: &Buffer,
    w_v_buf: &Buffer,
    out_q: &Buffer,
    out_k: &Buffer,
    out_v: &Buffer,
    k: usize,
    n_q: usize,
    n_k: usize,
    n_v: usize,
) -> Result<(), MetalError> {
    if !backend.supports_metal3() {
        return Err(MetalError::Unsupported(
            "sgemv_q4_k_f32_triple_quadcoop needs Metal3".to_string(),
        ));
    }
    if k % 256 != 0 {
        return Err(MetalError::ShapeMismatch(format!(
            "sgemv_q4_k_f32_triple_quadcoop: K%256==0 required (K={k})"
        )));
    }
    let pipeline = backend.pipeline(
        "sgemv_q4_k_f32_triple_quadcoop",
        SGEMV_Q4_K_F32_TRIPLE_QUADCOOP_SHADER,
        "sgemv_q4_k_f32_triple_quadcoop",
    )?;
    let dims = [k as u32, n_q as u32, n_k as u32, n_v as u32];
    let total = (n_q + n_k + n_v) as u64;
    let n_groups = total.div_ceil(4);
    backend.with_encoder(|encoder| {
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(x_buf), 0);
        encoder.set_buffer(1, Some(w_q_buf), 0);
        encoder.set_buffer(2, Some(w_k_buf), 0);
        encoder.set_buffer(3, Some(w_v_buf), 0);
        encoder.set_buffer(4, Some(out_q), 0);
        encoder.set_buffer(5, Some(out_k), 0);
        encoder.set_buffer(6, Some(out_v), 0);
        encoder.set_bytes(7, 16, dims.as_ptr() as *const std::ffi::c_void);
        let tg_size = MTLSize::new(32, 1, 1);
        let grid = MTLSize::new(32 * n_groups, 1, 1);
        encoder.dispatch_threads(grid, tg_size);
    });
    Ok(())
}

/// Pair variant — calls triple_quadcoop with n_v=0.
#[allow(clippy::too_many_arguments)]
pub fn sgemv_q4_k_f32_pair_quadcoop_into(
    backend: &MetalBackend,
    x_buf: &Buffer,
    w_a_buf: &Buffer,
    w_b_buf: &Buffer,
    out_a: &Buffer,
    out_b: &Buffer,
    k: usize,
    n_a: usize,
    n_b: usize,
) -> Result<(), MetalError> {
    sgemv_q4_k_f32_triple_quadcoop_into(
        backend, x_buf, w_a_buf, w_b_buf, w_b_buf, out_a, out_b, out_b, k, n_a, n_b, 0,
    )
}

// T81 simdcoop variant of the fused triple. Each output gets a
// dedicated 32-thread simdgroup that K-cooperates over its row, then
// reduces via simd_sum. Routing of (W, out) is done at threadgroup
// granularity — all 32 threads in a simdgroup share the same tg_id
// so there's no intra-simdgroup divergence on the branch.
const SGEMV_Q4_K_F32_TRIPLE_SIMDCOOP_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

constant uint BLOCK_BYTES = 144u;
constant uint BLOCK_WEIGHTS = 256u;

kernel void sgemv_q4_k_f32_triple_simdcoop(
    device const float* x       [[buffer(0)]],
    device const uchar* w_q     [[buffer(1)]],
    device const uchar* w_k     [[buffer(2)]],
    device const uchar* w_v     [[buffer(3)]],
    device float* out_q         [[buffer(4)]],
    device float* out_k         [[buffer(5)]],
    device float* out_v         [[buffer(6)]],
    constant uint4& dims        [[buffer(7)]],   // (K, n_q, n_k, n_v)
    uint tg_id                  [[threadgroup_position_in_grid]],
    uint tid                    [[thread_position_in_threadgroup]],
    uint sg_size                [[threads_per_simdgroup]]
) {
    uint K   = dims.x;
    uint n_q = dims.y;
    uint n_k = dims.z;
    uint n_v = dims.w;
    uint total = n_q + n_k + n_v;
    if (tg_id >= total) return;

    device const uchar* w;
    device float* out;
    uint n_idx;
    if (tg_id < n_q) {
        w = w_q;
        out = out_q;
        n_idx = tg_id;
    } else if (tg_id < n_q + n_k) {
        w = w_k;
        out = out_k;
        n_idx = tg_id - n_q;
    } else {
        w = w_v;
        out = out_v;
        n_idx = tg_id - n_q - n_k;
    }

    uint blocks_per_row = K / BLOCK_WEIGHTS;
    uint row_off = n_idx * blocks_per_row * BLOCK_BYTES;

    float partial = 0.0;
    for (uint blk = tid; blk < blocks_per_row; blk += sg_size) {
        device const uchar* block = w + row_off + blk * BLOCK_BYTES;
        ushort d_bits = ((ushort)block[1] << 8) | (ushort)block[0];
        ushort dmin_bits = ((ushort)block[3] << 8) | (ushort)block[2];
        float d = float(as_type<half>(d_bits));
        float dmin = float(as_type<half>(dmin_bits));
        uchar packed[12];
        for (uint i = 0; i < 12u; ++i) packed[i] = block[4 + i];
        uchar sc[8], m[8];
        for (uint i = 0; i < 4u; ++i) {
            sc[i]     = packed[i] & 0x3F;
            m[i]      = packed[i + 4] & 0x3F;
            sc[i + 4] = (packed[i + 8] & 0x0F) | ((packed[i] >> 6) << 4);
            m[i + 4]  = (packed[i + 8] >> 4)   | ((packed[i + 4] >> 6) << 4);
        }
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
            uint qs_base = jp * 8u;
            for (uint kg = 0; kg < 8u; ++kg) {
                uchar4 nibs = qs4[qs_base + kg];
                uint kk = kg * 4u;
                float n0lo = scale0 * float(nibs.x & 0x0F) - min0;
                float n1lo = scale0 * float(nibs.y & 0x0F) - min0;
                float n2lo = scale0 * float(nibs.z & 0x0F) - min0;
                float n3lo = scale0 * float(nibs.w & 0x0F) - min0;
                float n0hi = scale1 * float(nibs.x >> 4)   - min1;
                float n1hi = scale1 * float(nibs.y >> 4)   - min1;
                float n2hi = scale1 * float(nibs.z >> 4)   - min1;
                float n3hi = scale1 * float(nibs.w >> 4)   - min1;
                partial += x[x_low_off  + kk    ] * n0lo;
                partial += x[x_low_off  + kk + 1] * n1lo;
                partial += x[x_low_off  + kk + 2] * n2lo;
                partial += x[x_low_off  + kk + 3] * n3lo;
                partial += x[x_high_off + kk    ] * n0hi;
                partial += x[x_high_off + kk + 1] * n1hi;
                partial += x[x_high_off + kk + 2] * n2hi;
                partial += x[x_high_off + kk + 3] * n3hi;
            }
        }
    }

    float total_sum = simd_sum(partial);
    if (tid == 0) {
        out[n_idx] = total_sum;
    }
}
"#;

/// Simdcoop variant of the fused QKV triple kernel. Each output gets
/// a dedicated 32-thread simdgroup; reduces dispatch count AND uses
/// simd-cooperative K-reduction. Best when blocks_per_row >= 16
/// (K >= 4096); otherwise the simdgroup is under-utilised.
#[allow(clippy::too_many_arguments)]
pub fn sgemv_q4_k_f32_triple_simdcoop_into(
    backend: &MetalBackend,
    x_buf: &Buffer,
    w_q_buf: &Buffer,
    w_k_buf: &Buffer,
    w_v_buf: &Buffer,
    out_q: &Buffer,
    out_k: &Buffer,
    out_v: &Buffer,
    k: usize,
    n_q: usize,
    n_k: usize,
    n_v: usize,
) -> Result<(), MetalError> {
    if !backend.supports_metal3() {
        return Err(MetalError::Unsupported(
            "sgemv_q4_k_f32_triple_simdcoop needs Metal3".to_string(),
        ));
    }
    if k % 256 != 0 {
        return Err(MetalError::ShapeMismatch(format!(
            "sgemv_q4_k_f32_triple_simdcoop: K%256==0 required (K={k})"
        )));
    }
    let pipeline = backend.pipeline(
        "sgemv_q4_k_f32_triple_simdcoop",
        SGEMV_Q4_K_F32_TRIPLE_SIMDCOOP_SHADER,
        "sgemv_q4_k_f32_triple_simdcoop",
    )?;
    let dims = [k as u32, n_q as u32, n_k as u32, n_v as u32]; // T82
    let total = n_q + n_k + n_v;
    backend.with_encoder(|encoder| {
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(x_buf), 0);
        encoder.set_buffer(1, Some(w_q_buf), 0);
        encoder.set_buffer(2, Some(w_k_buf), 0);
        encoder.set_buffer(3, Some(w_v_buf), 0);
        encoder.set_buffer(4, Some(out_q), 0);
        encoder.set_buffer(5, Some(out_k), 0);
        encoder.set_buffer(6, Some(out_v), 0);
        encoder.set_bytes(7, 16, dims.as_ptr() as *const std::ffi::c_void);
        let tg_size = MTLSize::new(32, 1, 1);
        let grid = MTLSize::new(32 * total as u64, 1, 1);
        encoder.dispatch_threads(grid, tg_size);
    });
    Ok(())
}

/// Simdcoop variant of pair_into — Q4_K only.
#[allow(clippy::too_many_arguments)]
pub fn sgemv_q4_k_f32_pair_simdcoop_into(
    backend: &MetalBackend,
    x_buf: &Buffer,
    w_a_buf: &Buffer,
    w_b_buf: &Buffer,
    out_a: &Buffer,
    out_b: &Buffer,
    k: usize,
    n_a: usize,
    n_b: usize,
) -> Result<(), MetalError> {
    sgemv_q4_k_f32_triple_simdcoop_into(
        backend, x_buf, w_a_buf, w_b_buf, w_b_buf, out_a, out_b, out_b, k, n_a, n_b, 0,
    )
}

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
    // T82: dims passed via set_bytes (push constants) instead of an
    // MTLBuffer — saves one alloc_shared per dispatch on the hot path.
    let dims = [k as u32, n as u32];
    backend.with_encoder(|encoder| {
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(x_buf), 0);
        encoder.set_buffer(1, Some(w_q4k_buf), 0);
        encoder.set_buffer(2, Some(out_buf), 0);
        encoder.set_bytes(3, 8, dims.as_ptr() as *const std::ffi::c_void);
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

// T97 — RMSNorm with float4 vectorized loads. One simdgroup (32 threads)
// cooperates on one row of x[d]. Each thread strides over k=tid..d/4 step
// 32, reading 4 floats per access. Halves the issued load count vs scalar.
// Falls back to scalar tail if d % 4 != 0.
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
    uint d4 = d / 4u;

    device const float4* x4 = (device const float4*)x;
    for (uint i = tid; i < d4; i += sg_size) {
        float4 v = x4[i];
        partial += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }
    uint tail_start = d4 * 4u;
    for (uint i = tail_start + tid; i < d; i += sg_size) {
        float v = x[i];
        partial += v * v;
    }

    float total = simd_sum(partial);
    float inv_rms = 1.0 / sqrt(total / float(d) + eps);

    device const float4* g4 = (device const float4*)gamma;
    device float4* y4 = (device float4*)y;
    for (uint i = tid; i < d4; i += sg_size) {
        float4 v = x4[i];
        float4 g = g4[i];
        y4[i] = float4(v.x * inv_rms * g.x,
                       v.y * inv_rms * g.y,
                       v.z * inv_rms * g.z,
                       v.w * inv_rms * g.w);
    }
    for (uint i = tail_start + tid; i < d; i += sg_size) {
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
    let d_u = d as u32; // T82
    backend.with_encoder(|encoder| {
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(x_buf), 0);
        encoder.set_buffer(1, Some(gamma_buf), 0);
        encoder.set_buffer(2, Some(y_buf), 0);
        encoder.set_bytes(3, 4, &d_u as *const u32 as *const std::ffi::c_void);
        encoder.set_bytes(4, 4, &eps as *const f32 as *const std::ffi::c_void);
        let tg_size = MTLSize::new(32, 1, 1);
        let grid = MTLSize::new(32, 1, 1);
        encoder.dispatch_threads(grid, tg_size);
    });
    Ok(())
}

// T106 — Batched RMSNorm. B rows of size d processed in parallel.
// Each threadgroup handles one row (B threadgroups total). Same gamma
// shared across all rows. Used by forward_batch (multi-token forward).
const RMS_NORM_BATCHED_F32_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void rms_norm_batched_f32(
    device const float* x     [[buffer(0)]],   // [B, d]
    device const float* gamma [[buffer(1)]],   // [d]
    device float* y           [[buffer(2)]],   // [B, d]
    constant uint2& dims      [[buffer(3)]],   // (d, B)
    constant float& eps       [[buffer(4)]],
    uint b                    [[threadgroup_position_in_grid]],
    uint tid                  [[thread_position_in_threadgroup]],
    uint sg_size              [[threads_per_simdgroup]]
) {
    uint d = dims.x;
    uint B = dims.y;
    if (b >= B) return;

    device const float* xb = x + b * d;
    device float* yb       = y + b * d;
    uint d4 = d / 4u;

    float partial = 0.0;
    device const float4* xb4 = (device const float4*)xb;
    for (uint i = tid; i < d4; i += sg_size) {
        float4 v = xb4[i];
        partial += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }
    uint tail_start = d4 * 4u;
    for (uint i = tail_start + tid; i < d; i += sg_size) {
        float v = xb[i];
        partial += v * v;
    }

    float total = simd_sum(partial);
    float inv_rms = 1.0 / sqrt(total / float(d) + eps);

    device const float4* g4 = (device const float4*)gamma;
    device float4* yb4 = (device float4*)yb;
    for (uint i = tid; i < d4; i += sg_size) {
        float4 v = xb4[i];
        float4 g = g4[i];
        yb4[i] = float4(v.x * inv_rms * g.x,
                        v.y * inv_rms * g.y,
                        v.z * inv_rms * g.z,
                        v.w * inv_rms * g.w);
    }
    for (uint i = tail_start + tid; i < d; i += sg_size) {
        yb[i] = xb[i] * inv_rms * gamma[i];
    }
}
"#;

/// T106 — Batched RMSNorm: process B rows of size d in parallel.
/// Used by multi-token forward (forward_batch). Same gamma shared across
/// all rows. Output identical to B sequential rms_norm_f32 calls.
pub fn rms_norm_batched_f32(
    backend: &MetalBackend,
    x_buf: &Buffer,
    gamma_buf: &Buffer,
    y_buf: &Buffer,
    d: usize,
    b: usize,
    eps: f32,
) -> Result<(), MetalError> {
    let pipeline = backend.pipeline(
        "rms_norm_batched_f32",
        RMS_NORM_BATCHED_F32_SHADER,
        "rms_norm_batched_f32",
    )?;
    let dims = [d as u32, b as u32];
    backend.with_encoder(|encoder| {
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(x_buf), 0);
        encoder.set_buffer(1, Some(gamma_buf), 0);
        encoder.set_buffer(2, Some(y_buf), 0);
        encoder.set_bytes(3, 8, dims.as_ptr() as *const std::ffi::c_void);
        encoder.set_bytes(4, 4, &eps as *const f32 as *const std::ffi::c_void);
        let tg_size = MTLSize::new(32, 1, 1);
        let grid = MTLSize::new(32 * b as u64, 1, 1);
        encoder.dispatch_threads(grid, tg_size);
    });
    Ok(())
}

const RMS_NORM_PER_HEAD_F32_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

// Per-head RMSNorm (Qwen3 q_norm / k_norm). One threadgroup = one head;
// 32 threads cooperate on head_dim. gamma is shared across heads.
// (T98 float4 attempt regressed -1% — head_dim=128 too small to amortize
// vector overhead; reverted to scalar.)
kernel void rms_norm_per_head_f32(
    device float* x           [[buffer(0)]],
    device const float* gamma [[buffer(1)]],
    constant uint2& dims      [[buffer(2)]],
    constant float& eps       [[buffer(3)]],
    uint h                    [[threadgroup_position_in_grid]],
    uint tid                  [[thread_position_in_threadgroup]],
    uint sg_size              [[threads_per_simdgroup]]
) {
    uint n_heads = dims.x;
    uint head_dim = dims.y;
    if (h >= n_heads) return;

    device float* head = x + h * head_dim;
    float partial = 0.0;
    for (uint i = tid; i < head_dim; i += sg_size) {
        float v = head[i];
        partial += v * v;
    }
    float total = simd_sum(partial);
    float inv_rms = 1.0 / sqrt(total / float(head_dim) + eps);
    for (uint i = tid; i < head_dim; i += sg_size) {
        head[i] = head[i] * inv_rms * gamma[i];
    }
}
"#;

/// Per-head in-place RMSNorm — Qwen3 q_norm / k_norm.
pub fn rms_norm_per_head_f32(
    backend: &MetalBackend,
    x_buf: &Buffer,
    gamma_buf: &Buffer,
    n_heads: usize,
    head_dim: usize,
    eps: f32,
) -> Result<(), MetalError> {
    let pipeline = backend.pipeline(
        "rms_norm_per_head_f32",
        RMS_NORM_PER_HEAD_F32_SHADER,
        "rms_norm_per_head_f32",
    )?;
    let dims = [n_heads as u32, head_dim as u32]; // T82
    backend.with_encoder(|encoder| {
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(x_buf), 0);
        encoder.set_buffer(1, Some(gamma_buf), 0);
        encoder.set_bytes(2, 8, dims.as_ptr() as *const std::ffi::c_void);
        encoder.set_bytes(3, 4, &eps as *const f32 as *const std::ffi::c_void);
        let tg_size = MTLSize::new(32, 1, 1);
        let grid = MTLSize::new(32 * n_heads as u64, 1, 1);
        encoder.dispatch_threads(grid, tg_size);
    });
    Ok(())
}

const SWIGLU_F32_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

// Element-wise SwiGLU: y[i] = silu(gate[i]) * up[i] where
// silu(x) = x / (1 + exp(-x)).
// (T99 float4 attempt regressed slightly — divergence between vec/scalar
// paths costs more than the saved load count at this dispatch granularity.)
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
    let f_u = f as u32; // T82
    backend.with_encoder(|encoder| {
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(gate_buf), 0);
        encoder.set_buffer(1, Some(up_buf), 0);
        encoder.set_buffer(2, Some(y_buf), 0);
        encoder.set_bytes(3, 4, &f_u as *const u32 as *const std::ffi::c_void);
        let tg_size = MTLSize::new(256, 1, 1);
        let grid = MTLSize::new(f as u64, 1, 1);
        encoder.dispatch_threads(grid, tg_size);
    });
    Ok(())
}

const KV_APPEND_F32_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

// Append a single decode-step's K (or V) slice into the layer's KV
// cache at the given absolute position. Layout of the cache is
// [n_kv, max_seq, head_dim] row-major. Layout of src is
// [n_kv, head_dim] (one row per kv head).
kernel void kv_append_f32(
    device const float* src  [[buffer(0)]],   // [n_kv * head_dim]
    device float* dst        [[buffer(1)]],   // [n_kv * max_seq * head_dim]
    constant uint3& dims     [[buffer(2)]],   // (n_kv, head_dim, position)
    constant uint& max_seq   [[buffer(3)]],
    uint gid                 [[thread_position_in_grid]]
) {
    uint n_kv     = dims.x;
    uint head_dim = dims.y;
    uint position = dims.z;
    uint total    = n_kv * head_dim;
    if (gid >= total) return;
    uint kvh = gid / head_dim;
    uint dd  = gid % head_dim;
    uint dst_off = kvh * max_seq * head_dim + position * head_dim + dd;
    dst[dst_off] = src[gid];
}
"#;

/// Copy a decode-step K (or V) slice into the layer's KV cache at the
/// given absolute sequence position. Pure GPU — no drain needed.
pub fn kv_append_f32(
    backend: &MetalBackend,
    src_buf: &Buffer,
    dst_cache_buf: &Buffer,
    n_kv: usize,
    head_dim: usize,
    position: usize,
    max_seq: usize,
) -> Result<(), MetalError> {
    let pipeline = backend.pipeline("kv_append_f32", KV_APPEND_F32_SHADER, "kv_append_f32")?;
    let dims = [n_kv as u32, head_dim as u32, position as u32]; // T82
    let ms = max_seq as u32;
    let total = n_kv * head_dim;
    backend.with_encoder(|encoder| {
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(src_buf), 0);
        encoder.set_buffer(1, Some(dst_cache_buf), 0);
        encoder.set_bytes(2, 12, dims.as_ptr() as *const std::ffi::c_void);
        encoder.set_bytes(3, 4, &ms as *const u32 as *const std::ffi::c_void);
        let tg_size = MTLSize::new(64, 1, 1);
        let grid = MTLSize::new(total as u64, 1, 1);
        encoder.dispatch_threads(grid, tg_size);
    });
    Ok(())
}

// T108 — Batched K-only or V-only append. Writes B sequential positions
// (pos_base, pos_base+1, ..., pos_base+B-1) into the cache from a packed
// source [B, n_kv * head_dim].
const KV_APPEND_BATCHED_F32_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void kv_append_batched_f32(
    device const float* src  [[buffer(0)]],   // [B, n_kv * head_dim]
    device float* dst        [[buffer(1)]],   // [n_kv, max_seq, head_dim]
    constant uint4& dims     [[buffer(2)]],   // (n_kv, head_dim, pos_base, B)
    constant uint& max_seq   [[buffer(3)]],
    uint gid                 [[thread_position_in_grid]]
) {
    uint n_kv     = dims.x;
    uint head_dim = dims.y;
    uint pos_base = dims.z;
    uint B        = dims.w;
    uint per_token = n_kv * head_dim;
    uint total = B * per_token;
    if (gid >= total) return;

    uint b   = gid / per_token;
    uint flat = gid % per_token;
    uint kvh = flat / head_dim;
    uint dd  = flat % head_dim;
    uint dst_off = kvh * max_seq * head_dim + (pos_base + b) * head_dim + dd;
    dst[dst_off] = src[gid];
}
"#;

/// T108 — Batched KV append. Writes B sequential positions into the cache.
#[allow(clippy::too_many_arguments)]
pub fn kv_append_batched_f32(
    backend: &MetalBackend,
    src_buf: &Buffer,
    dst_cache_buf: &Buffer,
    n_kv: usize,
    head_dim: usize,
    position_base: usize,
    b: usize,
    max_seq: usize,
) -> Result<(), MetalError> {
    let pipeline = backend.pipeline(
        "kv_append_batched_f32",
        KV_APPEND_BATCHED_F32_SHADER,
        "kv_append_batched_f32",
    )?;
    let dims = [n_kv as u32, head_dim as u32, position_base as u32, b as u32];
    let ms = max_seq as u32;
    let total = b * n_kv * head_dim;
    backend.with_encoder(|encoder| {
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(src_buf), 0);
        encoder.set_buffer(1, Some(dst_cache_buf), 0);
        encoder.set_bytes(2, 16, dims.as_ptr() as *const std::ffi::c_void);
        encoder.set_bytes(3, 4, &ms as *const u32 as *const std::ffi::c_void);
        let tg_size = MTLSize::new(64, 1, 1);
        let grid = MTLSize::new(total as u64, 1, 1);
        encoder.dispatch_threads(grid, tg_size);
    });
    Ok(())
}

// T96 — Fused K+V append. 1 dispatch instead of 2 per layer × 40 layers.
const KV_APPEND_KV_F32_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void kv_append_kv_f32(
    device const float* src_k  [[buffer(0)]],   // [n_kv * head_dim]
    device const float* src_v  [[buffer(1)]],
    device float* dst_k        [[buffer(2)]],
    device float* dst_v        [[buffer(3)]],
    constant uint3& dims       [[buffer(4)]],   // (n_kv, head_dim, position)
    constant uint& max_seq     [[buffer(5)]],
    uint gid                   [[thread_position_in_grid]]
) {
    uint n_kv     = dims.x;
    uint head_dim = dims.y;
    uint position = dims.z;
    uint total    = n_kv * head_dim;
    if (gid >= 2u * total) return;
    bool is_v = gid >= total;
    uint flat = is_v ? (gid - total) : gid;
    uint kvh = flat / head_dim;
    uint dd  = flat % head_dim;
    uint dst_off = kvh * max_seq * head_dim + position * head_dim + dd;
    if (is_v) {
        dst_v[dst_off] = src_v[flat];
    } else {
        dst_k[dst_off] = src_k[flat];
    }
}
"#;

/// T96 — Fused K and V cache append in 1 dispatch.
#[allow(clippy::too_many_arguments)]
pub fn kv_append_kv_f32(
    backend: &MetalBackend,
    src_k: &Buffer,
    src_v: &Buffer,
    dst_k: &Buffer,
    dst_v: &Buffer,
    n_kv: usize,
    head_dim: usize,
    position: usize,
    max_seq: usize,
) -> Result<(), MetalError> {
    let pipeline = backend.pipeline(
        "kv_append_kv_f32",
        KV_APPEND_KV_F32_SHADER,
        "kv_append_kv_f32",
    )?;
    let dims = [n_kv as u32, head_dim as u32, position as u32];
    let ms = max_seq as u32;
    let total = n_kv * head_dim;
    backend.with_encoder(|encoder| {
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(src_k), 0);
        encoder.set_buffer(1, Some(src_v), 0);
        encoder.set_buffer(2, Some(dst_k), 0);
        encoder.set_buffer(3, Some(dst_v), 0);
        encoder.set_bytes(4, 12, dims.as_ptr() as *const std::ffi::c_void);
        encoder.set_bytes(5, 4, &ms as *const u32 as *const std::ffi::c_void);
        let tg_size = MTLSize::new(64, 1, 1);
        let grid = MTLSize::new((2 * total) as u64, 1, 1);
        encoder.dispatch_threads(grid, tg_size);
    });
    Ok(())
}

const ROPE_HALF_SPLIT_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

// Apply RoPE half-split convention to one [n_heads * head_dim] row,
// in place. position is the row's absolute sequence index (used to
// index into the precomputed cos/sin tables of shape
// [max_seq, head_dim/2] row-major).
//
// Half-split: pair dim k with dim (k + head_dim/2). For k in 0..D/2:
//   x'[k]      = x[k]      * cos(angle) - x[k + D/2] * sin(angle)
//   x'[k+D/2]  = x[k+D/2]  * cos(angle) + x[k]       * sin(angle)
// where angle = position * theta_k.
kernel void rope_half_split_f32(
    device float* x              [[buffer(0)]],   // [n_heads * head_dim]
    device const float* cos_tab  [[buffer(1)]],   // [max_seq, head_dim/2]
    device const float* sin_tab  [[buffer(2)]],
    constant uint3& dims         [[buffer(3)]],   // (n_heads, head_dim, position)
    uint gid                     [[thread_position_in_grid]]
) {
    uint n_heads  = dims.x;
    uint head_dim = dims.y;
    uint position = dims.z;
    uint half_dim = head_dim / 2u;
    uint total    = n_heads * half_dim;
    if (gid >= total) return;

    uint h = gid / half_dim;
    uint k = gid % half_dim;
    uint i0 = h * head_dim + k;
    uint i1 = h * head_dim + k + half_dim;

    uint tab_off = position * half_dim + k;
    float c = cos_tab[tab_off];
    float s = sin_tab[tab_off];
    float x0 = x[i0];
    float x1 = x[i1];
    x[i0] = x0 * c - x1 * s;
    x[i1] = x1 * c + x0 * s;
}
"#;

/// Apply half-split RoPE in place on a single decode-step row of
/// `x[n_heads * head_dim]` at the given absolute sequence position.
/// The cos/sin tables are precomputed (see `rustorch_nn::rope::RoPE`).
pub fn rope_half_split_f32(
    backend: &MetalBackend,
    x_buf: &Buffer,
    cos_buf: &Buffer,
    sin_buf: &Buffer,
    n_heads: usize,
    head_dim: usize,
    position: usize,
) -> Result<(), MetalError> {
    let pipeline = backend.pipeline(
        "rope_half_split_f32",
        ROPE_HALF_SPLIT_SHADER,
        "rope_half_split_f32",
    )?;
    let dims = [n_heads as u32, head_dim as u32, position as u32]; // T82
    let total = n_heads * (head_dim / 2);
    backend.with_encoder(|encoder| {
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(x_buf), 0);
        encoder.set_buffer(1, Some(cos_buf), 0);
        encoder.set_buffer(2, Some(sin_buf), 0);
        encoder.set_bytes(3, 12, dims.as_ptr() as *const std::ffi::c_void);
        let tg_size = MTLSize::new(64, 1, 1);
        let grid = MTLSize::new(total as u64, 1, 1);
        encoder.dispatch_threads(grid, tg_size);
    });
    Ok(())
}

// T107 — Batched RoPE. Process B rows of [n_heads * head_dim] in parallel,
// each row at a distinct sequence position position_base + b. Same cos/sin
// tables shared. 1 thread per (batch, head, k_in_half_dim).
const ROPE_HALF_SPLIT_BATCHED_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void rope_half_split_batched_f32(
    device float* x              [[buffer(0)]],   // [B, n_heads * head_dim]
    device const float* cos_tab  [[buffer(1)]],
    device const float* sin_tab  [[buffer(2)]],
    constant uint4& dims         [[buffer(3)]],   // (n_heads, head_dim, position_base, B)
    uint2 gid                    [[thread_position_in_grid]]
) {
    uint n_heads  = dims.x;
    uint head_dim = dims.y;
    uint pos_base = dims.z;
    uint B        = dims.w;

    uint b = gid.y;
    if (b >= B) return;

    uint half_dim = head_dim / 2u;
    uint flat = gid.x;
    uint total = n_heads * half_dim;
    if (flat >= total) return;

    uint h = flat / half_dim;
    uint k = flat % half_dim;
    uint row_off = b * (n_heads * head_dim);
    uint i0 = row_off + h * head_dim + k;
    uint i1 = row_off + h * head_dim + k + half_dim;

    uint position = pos_base + b;
    uint tab_off = position * half_dim + k;
    float c = cos_tab[tab_off];
    float s = sin_tab[tab_off];
    float x0 = x[i0];
    float x1 = x[i1];
    x[i0] = x0 * c - x1 * s;
    x[i1] = x1 * c + x0 * s;
}
"#;

/// T107 — Batched RoPE. Apply half-split RoPE in place to B rows of
/// [n_heads * head_dim], each at sequence position `position_base + b`.
pub fn rope_half_split_batched_f32(
    backend: &MetalBackend,
    x_buf: &Buffer,
    cos_buf: &Buffer,
    sin_buf: &Buffer,
    n_heads: usize,
    head_dim: usize,
    position_base: usize,
    b: usize,
) -> Result<(), MetalError> {
    let pipeline = backend.pipeline(
        "rope_half_split_batched_f32",
        ROPE_HALF_SPLIT_BATCHED_SHADER,
        "rope_half_split_batched_f32",
    )?;
    let dims = [
        n_heads as u32,
        head_dim as u32,
        position_base as u32,
        b as u32,
    ];
    let total = n_heads * (head_dim / 2);
    backend.with_encoder(|encoder| {
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(x_buf), 0);
        encoder.set_buffer(1, Some(cos_buf), 0);
        encoder.set_buffer(2, Some(sin_buf), 0);
        encoder.set_bytes(3, 16, dims.as_ptr() as *const std::ffi::c_void);
        let tg_size = MTLSize::new(64, 1, 1);
        let grid = MTLSize::new(total as u64, b as u64, 1);
        encoder.dispatch_threads(grid, tg_size);
    });
    Ok(())
}

const GQA_DECODE_F32_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

// GQA decode attention for a single query position vs a kv-length cached
// prefix. One threadgroup = one query head; the 32 threads of the inner
// simdgroup cooperate on the kv_len reduction (softmax over scores +
// weighted sum of V vectors).
//
// Inputs:
//   q[n_heads, head_dim]
//   k_cache[n_kv, max_seq, head_dim]   (only the first kv_len are read)
//   v_cache[n_kv, max_seq, head_dim]
//   out[n_heads, head_dim]
//
// Each query head q_h reads from kv head q_h / (n_heads / n_kv).
//
// We softmax-stable: max-shift then exp+sum.
kernel void gqa_decode_f32(
    device const float* q       [[buffer(0)]],
    device const float* k_cache [[buffer(1)]],
    device const float* v_cache [[buffer(2)]],
    device float* out           [[buffer(3)]],
    constant uint4& dims        [[buffer(4)]],   // (n_heads, n_kv, head_dim, kv_len)
    constant uint& max_seq      [[buffer(5)]],
    constant float& inv_sqrt_d  [[buffer(6)]],
    threadgroup float* shared   [[threadgroup(0)]],
    uint q_h                    [[threadgroup_position_in_grid]],
    uint tid                    [[thread_position_in_threadgroup]],
    uint sg_size                [[threads_per_simdgroup]]
) {
    uint n_heads  = dims.x;
    uint n_kv     = dims.y;
    uint head_dim = dims.z;
    uint kv_len   = dims.w;
    if (q_h >= n_heads) return;
    uint group_size = n_heads / n_kv;
    uint kv_h = q_h / group_size;

    // 1. Compute attention scores: score[p] = (q[q_h] dot k_cache[kv_h][p]) * inv_sqrt_d
    //    Each thread handles a stride of positions p = tid, tid+32, ...
    //    Then we softmax (max-shift) across positions cooperatively.

    // Phase A: each thread computes its own scores into shared[].
    // For kv_len up to ~1024 we fit comfortably; beyond that we'd need
    // tiling. For Qwen3-14B context lengths in practice this is fine.
    //
    // Layout of shared: [kv_len] float scores.

    device const float* q_h_ptr = q + q_h * head_dim;
    device const float* k_h_base = k_cache + kv_h * max_seq * head_dim;

    // T100 — float4 vectorized dot product. head_dim must be %4.
    uint hd4 = head_dim / 4u;
    device const float4* q_h_ptr4 = (device const float4*)q_h_ptr;
    for (uint p = tid; p < kv_len; p += sg_size) {
        device const float4* k_p4 = (device const float4*)(k_h_base + p * head_dim);
        float4 acc4 = float4(0.0, 0.0, 0.0, 0.0);
        for (uint d4 = 0; d4 < hd4; ++d4) {
            acc4 += q_h_ptr4[d4] * k_p4[d4];
        }
        // Scalar tail
        float dot = acc4.x + acc4.y + acc4.z + acc4.w;
        for (uint d = hd4 * 4u; d < head_dim; ++d) {
            dot += q_h_ptr[d] * k_h_base[p * head_dim + d];
        }
        shared[p] = dot * inv_sqrt_d;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase B: find max for stable softmax.
    float local_max = -INFINITY;
    for (uint p = tid; p < kv_len; p += sg_size) {
        local_max = max(local_max, shared[p]);
    }
    float max_score = simd_max(local_max);

    // Phase C: exp + accumulate sum.
    float local_sum = 0.0;
    for (uint p = tid; p < kv_len; p += sg_size) {
        float e = exp(shared[p] - max_score);
        shared[p] = e;
        local_sum += e;
    }
    float sum = simd_sum(local_sum);
    float inv_sum = 1.0 / sum;

    // Phase D: weighted sum of V vectors. Each thread accumulates one
    // dim of out. With sg_size=32 and head_dim=128 each thread handles
    // 4 dims via stride loop.
    // T101 — pre-multiply shared[p] *= inv_sum cooperatively, eliminates
    // kv_len multiplies in the inner accumulation loop.
    for (uint p = tid; p < kv_len; p += sg_size) {
        shared[p] *= inv_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device const float* v_h_base = v_cache + kv_h * max_seq * head_dim;
    device float* out_h = out + q_h * head_dim;
    for (uint d = tid; d < head_dim; d += sg_size) {
        float acc = 0.0;
        for (uint p = 0; p < kv_len; ++p) {
            acc += shared[p] * v_h_base[p * head_dim + d];
        }
        out_h[d] = acc;
    }
}
"#;

/// GQA attention for a single decode-step query (`q_seq = 1`) against
/// the cached `kv_len` K/V positions. One threadgroup per query head;
/// 32 threads inside cooperate on the kv-length reduction.
///
/// `shared_bytes` must be at least `kv_len * 4` (f32 score scratch).
#[allow(clippy::too_many_arguments)]
pub fn gqa_decode_f32(
    backend: &MetalBackend,
    q_buf: &Buffer,
    k_cache: &Buffer,
    v_cache: &Buffer,
    out_buf: &Buffer,
    n_heads: usize,
    n_kv: usize,
    head_dim: usize,
    kv_len: usize,
    max_seq: usize,
) -> Result<(), MetalError> {
    let pipeline = backend.pipeline("gqa_decode_f32", GQA_DECODE_F32_SHADER, "gqa_decode_f32")?;
    let dims = [n_heads as u32, n_kv as u32, head_dim as u32, kv_len as u32]; // T82
    let ms = max_seq as u32;
    let inv_sqrt_d: f32 = 1.0 / (head_dim as f32).sqrt();
    let shared_bytes = (kv_len * 4) as u64;
    backend.with_encoder(|encoder| {
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(q_buf), 0);
        encoder.set_buffer(1, Some(k_cache), 0);
        encoder.set_buffer(2, Some(v_cache), 0);
        encoder.set_buffer(3, Some(out_buf), 0);
        encoder.set_bytes(4, 16, dims.as_ptr() as *const std::ffi::c_void);
        encoder.set_bytes(5, 4, &ms as *const u32 as *const std::ffi::c_void);
        encoder.set_bytes(6, 4, &inv_sqrt_d as *const f32 as *const std::ffi::c_void);
        encoder.set_threadgroup_memory_length(0, shared_bytes);
        let tg_size = MTLSize::new(32, 1, 1);
        let grid = MTLSize::new(32 * n_heads as u64, 1, 1);
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
    let d_u = d as u32; // T82
    backend.with_encoder(|encoder| {
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(x_buf), 0);
        encoder.set_buffer(1, Some(y_buf), 0);
        encoder.set_bytes(2, 4, &d_u as *const u32 as *const std::ffi::c_void);
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

// Transposed-layout variant of sgemv_q6_k_f32 (block-major).
const SGEMV_Q6_K_F32_T_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

constant uint Q6K_BYTES = 210u;
constant uint Q6K_WEIGHTS = 256u;

kernel void sgemv_q6_k_f32_transposed(
    device const float* x       [[buffer(0)]],   // [K]
    device const uchar* w_q6k   [[buffer(1)]],   // [K/256, N, 210] transposed
    device float* y             [[buffer(2)]],   // [N]
    constant uint2& dims        [[buffer(3)]],   // (K, N)
    uint gid                    [[thread_position_in_grid]]
) {
    uint K = dims.x;
    uint N = dims.y;
    uint n_idx = gid;
    if (n_idx >= N) return;

    uint blocks_per_row = K / Q6K_WEIGHTS;
    float acc = 0.0;

    for (uint blk = 0; blk < blocks_per_row; ++blk) {
        // Transposed: block `blk` for row `n_idx` at blk*N*210 + n_idx*210
        device const uchar* block = w_q6k + blk * N * Q6K_BYTES + n_idx * Q6K_BYTES;
        device const uchar* ql = block;
        device const uchar* qh = block + 128;
        device const char*  sc = (device const char*)(block + 192);
        ushort d_bits = ((ushort)block[209] << 8) | (ushort)block[208];
        float d = float(as_type<half>(d_bits));

        for (uint half_idx = 0u; half_idx < 2u; ++half_idx) {
            device const uchar* ql_h = ql + half_idx * 64u;
            device const uchar* qh_h = qh + half_idx * 32u;
            device const char*  sc_h = sc + half_idx * 8;
            uint x_h_off = blk * Q6K_WEIGHTS + half_idx * 128u;

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

/// Transposed-layout Q6_K sgemv (companion to [`sgemv_q4_k_f32_transposed_into`]).
pub fn sgemv_q6_k_f32_transposed_into(
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
            "sgemv_q6_k_f32_t: K%256==0 required (K={k}, N={n})"
        )));
    }
    let pipeline = backend.pipeline(
        "sgemv_q6_k_f32_transposed",
        SGEMV_Q6_K_F32_T_SHADER,
        "sgemv_q6_k_f32_transposed",
    )?;
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

/// Repack Q6_K from row-major to block-major. See [`repack_q4_k_transposed`].
pub fn repack_q6_k_transposed(src: &[u8], k: usize, n: usize) -> Vec<u8> {
    let blocks_per_row = k / 256;
    assert_eq!(src.len(), n * blocks_per_row * 210);
    let mut dst = vec![0u8; src.len()];
    for n_idx in 0..n {
        for blk in 0..blocks_per_row {
            let src_off = n_idx * blocks_per_row * 210 + blk * 210;
            let dst_off = blk * n * 210 + n_idx * 210;
            dst[dst_off..dst_off + 210].copy_from_slice(&src[src_off..src_off + 210]);
        }
    }
    dst
}

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
    let dims = [k as u32, n as u32]; // T82
    backend.with_encoder(|encoder| {
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(x_buf), 0);
        encoder.set_buffer(1, Some(w_q6k_buf), 0);
        encoder.set_buffer(2, Some(out_buf), 0);
        encoder.set_bytes(3, 8, dims.as_ptr() as *const std::ffi::c_void);
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

    /// Build a deterministic Q4_K matrix for tests. Matches the layout used
    /// by `bench_q4k_metal::build_random_q4k_matrix` so we exercise the same
    /// shader code path the example uses.
    fn build_test_q4k_matrix(n: usize, k: usize, seed: u32) -> Vec<u8> {
        const QK_K: usize = 256;
        const Q4_K_BYTES: usize = 144;
        let blocks_per_row = k / QK_K;
        let total = n * blocks_per_row * Q4_K_BYTES;
        let mut buf = vec![0u8; total];
        let mut s = seed;
        let mut rand_byte = || {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            (s >> 8) as u8
        };
        // f16 d=0.05, dmin=0.01.
        let dh = half::f16::from_f32(0.05).to_le_bytes();
        let dminh = half::f16::from_f32(0.01).to_le_bytes();
        for blk in 0..(n * blocks_per_row) {
            let off = blk * Q4_K_BYTES;
            buf[off] = dh[0];
            buf[off + 1] = dh[1];
            buf[off + 2] = dminh[0];
            buf[off + 3] = dminh[1];
            for i in 4..16 {
                buf[off + i] = rand_byte() & 0x3F;
            }
            for i in 0..128 {
                buf[off + 16 + i] = rand_byte();
            }
        }
        buf
    }

    /// T80 — verify that the fused QKV triple kernel produces the same
    /// outputs as three independent sgemv_q4_k_f32_into calls.
    #[test]
    fn sgemv_q4_k_f32_triple_matches_three_singles() {
        let backend = metal_backend();
        if !backend.supports_metal3() {
            eprintln!("[triple] skipping: no Metal3");
            return;
        }
        let k = 256;
        let n_q = 8;
        let n_k = 4;
        let n_v = 4;

        let w_q_bytes = build_test_q4k_matrix(n_q, k, 11);
        let w_k_bytes = build_test_q4k_matrix(n_k, k, 17);
        let w_v_bytes = build_test_q4k_matrix(n_v, k, 23);
        let x: Vec<f32> = (0..k).map(|i| ((i as f32 + 1.0) * 0.001).sin()).collect();

        let x_buf = backend.alloc_shared(k * 4).unwrap();
        let w_q_buf = backend.alloc_shared(w_q_bytes.len()).unwrap();
        let w_k_buf = backend.alloc_shared(w_k_bytes.len()).unwrap();
        let w_v_buf = backend.alloc_shared(w_v_bytes.len()).unwrap();
        let out_q_single = backend.alloc_shared(n_q * 4).unwrap();
        let out_k_single = backend.alloc_shared(n_k * 4).unwrap();
        let out_v_single = backend.alloc_shared(n_v * 4).unwrap();
        let out_q_triple = backend.alloc_shared(n_q * 4).unwrap();
        let out_k_triple = backend.alloc_shared(n_k * 4).unwrap();
        let out_v_triple = backend.alloc_shared(n_v * 4).unwrap();

        unsafe {
            std::ptr::copy_nonoverlapping(x.as_ptr(), x_buf.contents() as *mut f32, k);
            std::ptr::copy_nonoverlapping(
                w_q_bytes.as_ptr(),
                w_q_buf.contents() as *mut u8,
                w_q_bytes.len(),
            );
            std::ptr::copy_nonoverlapping(
                w_k_bytes.as_ptr(),
                w_k_buf.contents() as *mut u8,
                w_k_bytes.len(),
            );
            std::ptr::copy_nonoverlapping(
                w_v_bytes.as_ptr(),
                w_v_buf.contents() as *mut u8,
                w_v_bytes.len(),
            );
        }

        // Three single dispatches (reference).
        sgemv_q4_k_f32_into(backend, &x_buf, &w_q_buf, &out_q_single, k, n_q).unwrap();
        sgemv_q4_k_f32_into(backend, &x_buf, &w_k_buf, &out_k_single, k, n_k).unwrap();
        sgemv_q4_k_f32_into(backend, &x_buf, &w_v_buf, &out_v_single, k, n_v).unwrap();
        backend.drain();

        // One triple dispatch.
        sgemv_q4_k_f32_triple_into(
            backend,
            &x_buf,
            &w_q_buf,
            &w_k_buf,
            &w_v_buf,
            &out_q_triple,
            &out_k_triple,
            &out_v_triple,
            k,
            n_q,
            n_k,
            n_v,
        )
        .unwrap();
        backend.drain();

        let read = |b: &Buffer, n: usize| -> Vec<f32> {
            unsafe { std::slice::from_raw_parts(b.contents() as *const f32, n).to_vec() }
        };
        let q_s = read(&out_q_single, n_q);
        let q_t = read(&out_q_triple, n_q);
        let k_s = read(&out_k_single, n_k);
        let k_t = read(&out_k_triple, n_k);
        let v_s = read(&out_v_single, n_v);
        let v_t = read(&out_v_triple, n_v);

        eprintln!("Q single: {:?}", q_s);
        eprintln!("Q triple: {:?}", q_t);
        eprintln!("K single: {:?}", k_s);
        eprintln!("K triple: {:?}", k_t);
        eprintln!("V single: {:?}", v_s);
        eprintln!("V triple: {:?}", v_t);

        for (a, b) in q_s.iter().zip(q_t.iter()) {
            assert!((a - b).abs() < 1e-4, "Q mismatch: {a} vs {b}");
        }
        for (a, b) in k_s.iter().zip(k_t.iter()) {
            assert!((a - b).abs() < 1e-4, "K mismatch: {a} vs {b}");
        }
        for (a, b) in v_s.iter().zip(v_t.iter()) {
            assert!((a - b).abs() < 1e-4, "V mismatch: {a} vs {b}");
        }
    }

    /// T80 — verify that pair_into (= triple_into with n_v=0 and aliased
    /// buffers for the V slot) produces the same outputs as two independent
    /// sgemv_q4_k_f32_into calls.
    #[test]
    fn sgemv_q4_k_f32_pair_matches_two_singles() {
        let backend = metal_backend();
        if !backend.supports_metal3() {
            eprintln!("[pair] skipping: no Metal3");
            return;
        }
        let k = 256;
        let n_a = 8;
        let n_b = 6;

        let w_a_bytes = build_test_q4k_matrix(n_a, k, 31);
        let w_b_bytes = build_test_q4k_matrix(n_b, k, 37);
        let x: Vec<f32> = (0..k).map(|i| ((i as f32 + 1.0) * 0.001).sin()).collect();

        let x_buf = backend.alloc_shared(k * 4).unwrap();
        let w_a_buf = backend.alloc_shared(w_a_bytes.len()).unwrap();
        let w_b_buf = backend.alloc_shared(w_b_bytes.len()).unwrap();
        let out_a_single = backend.alloc_shared(n_a * 4).unwrap();
        let out_b_single = backend.alloc_shared(n_b * 4).unwrap();
        let out_a_pair = backend.alloc_shared(n_a * 4).unwrap();
        let out_b_pair = backend.alloc_shared(n_b * 4).unwrap();

        unsafe {
            std::ptr::copy_nonoverlapping(x.as_ptr(), x_buf.contents() as *mut f32, k);
            std::ptr::copy_nonoverlapping(
                w_a_bytes.as_ptr(),
                w_a_buf.contents() as *mut u8,
                w_a_bytes.len(),
            );
            std::ptr::copy_nonoverlapping(
                w_b_bytes.as_ptr(),
                w_b_buf.contents() as *mut u8,
                w_b_bytes.len(),
            );
        }

        sgemv_q4_k_f32_into(backend, &x_buf, &w_a_buf, &out_a_single, k, n_a).unwrap();
        sgemv_q4_k_f32_into(backend, &x_buf, &w_b_buf, &out_b_single, k, n_b).unwrap();
        backend.drain();

        sgemv_q4_k_f32_pair_into(
            backend,
            &x_buf,
            &w_a_buf,
            &w_b_buf,
            &out_a_pair,
            &out_b_pair,
            k,
            n_a,
            n_b,
        )
        .unwrap();
        backend.drain();

        let read = |b: &Buffer, n: usize| -> Vec<f32> {
            unsafe { std::slice::from_raw_parts(b.contents() as *const f32, n).to_vec() }
        };
        let a_s = read(&out_a_single, n_a);
        let a_p = read(&out_a_pair, n_a);
        let b_s = read(&out_b_single, n_b);
        let b_p = read(&out_b_pair, n_b);

        eprintln!("A single: {:?}", a_s);
        eprintln!("A pair:   {:?}", a_p);
        eprintln!("B single: {:?}", b_s);
        eprintln!("B pair:   {:?}", b_p);

        for (a, b) in a_s.iter().zip(a_p.iter()) {
            assert!((a - b).abs() < 1e-4, "A mismatch: {a} vs {b}");
        }
        for (a, b) in b_s.iter().zip(b_p.iter()) {
            assert!((a - b).abs() < 1e-4, "B mismatch: {a} vs {b}");
        }
    }

    /// T81 — simdcoop variant of triple matches three single sgemv calls.
    #[test]
    fn sgemv_q4_k_f32_triple_simdcoop_matches_three_singles() {
        let backend = metal_backend();
        if !backend.supports_metal3() {
            eprintln!("[triple_simdcoop] skipping: no Metal3");
            return;
        }
        // K = 4096 to match the >=16 blocks per row constraint.
        let k = 4096;
        let n_q = 64;
        let n_k = 16;
        let n_v = 16;

        let w_q_bytes = build_test_q4k_matrix(n_q, k, 71);
        let w_k_bytes = build_test_q4k_matrix(n_k, k, 73);
        let w_v_bytes = build_test_q4k_matrix(n_v, k, 79);
        let x: Vec<f32> = (0..k).map(|i| ((i as f32 + 1.0) * 0.001).sin()).collect();

        let x_buf = backend.alloc_shared(k * 4).unwrap();
        let w_q_buf = backend.alloc_shared(w_q_bytes.len()).unwrap();
        let w_k_buf = backend.alloc_shared(w_k_bytes.len()).unwrap();
        let w_v_buf = backend.alloc_shared(w_v_bytes.len()).unwrap();
        let out_q_single = backend.alloc_shared(n_q * 4).unwrap();
        let out_k_single = backend.alloc_shared(n_k * 4).unwrap();
        let out_v_single = backend.alloc_shared(n_v * 4).unwrap();
        let out_q_t = backend.alloc_shared(n_q * 4).unwrap();
        let out_k_t = backend.alloc_shared(n_k * 4).unwrap();
        let out_v_t = backend.alloc_shared(n_v * 4).unwrap();
        unsafe {
            std::ptr::copy_nonoverlapping(x.as_ptr(), x_buf.contents() as *mut f32, k);
            std::ptr::copy_nonoverlapping(
                w_q_bytes.as_ptr(),
                w_q_buf.contents() as *mut u8,
                w_q_bytes.len(),
            );
            std::ptr::copy_nonoverlapping(
                w_k_bytes.as_ptr(),
                w_k_buf.contents() as *mut u8,
                w_k_bytes.len(),
            );
            std::ptr::copy_nonoverlapping(
                w_v_bytes.as_ptr(),
                w_v_buf.contents() as *mut u8,
                w_v_bytes.len(),
            );
        }

        sgemv_q4_k_f32_into(backend, &x_buf, &w_q_buf, &out_q_single, k, n_q).unwrap();
        sgemv_q4_k_f32_into(backend, &x_buf, &w_k_buf, &out_k_single, k, n_k).unwrap();
        sgemv_q4_k_f32_into(backend, &x_buf, &w_v_buf, &out_v_single, k, n_v).unwrap();
        backend.drain();

        sgemv_q4_k_f32_triple_simdcoop_into(
            backend, &x_buf, &w_q_buf, &w_k_buf, &w_v_buf, &out_q_t, &out_k_t, &out_v_t, k, n_q,
            n_k, n_v,
        )
        .unwrap();
        backend.drain();

        let read = |b: &Buffer, n: usize| -> Vec<f32> {
            unsafe { std::slice::from_raw_parts(b.contents() as *const f32, n).to_vec() }
        };
        let q_s = read(&out_q_single, n_q);
        let q_t = read(&out_q_t, n_q);
        let k_s = read(&out_k_single, n_k);
        let k_t = read(&out_k_t, n_k);
        let v_s = read(&out_v_single, n_v);
        let v_t = read(&out_v_t, n_v);

        // simd_sum reorders FMAs vs scalar sequential — relax tolerance.
        for (a, b) in q_s.iter().zip(q_t.iter()) {
            let r = (a - b).abs() / a.abs().max(1e-3);
            assert!(r < 1e-3, "Q mismatch: {a} vs {b} (rel {r:.3e})");
        }
        for (a, b) in k_s.iter().zip(k_t.iter()) {
            let r = (a - b).abs() / a.abs().max(1e-3);
            assert!(r < 1e-3, "K mismatch: {a} vs {b} (rel {r:.3e})");
        }
        for (a, b) in v_s.iter().zip(v_t.iter()) {
            let r = (a - b).abs() / a.abs().max(1e-3);
            assert!(r < 1e-3, "V mismatch: {a} vs {b} (rel {r:.3e})");
        }
    }

    /// T84 — quadcoop matches single sgemv.
    #[test]
    fn sgemv_q4_k_f32_quadcoop_matches_single() {
        let backend = metal_backend();
        if !backend.supports_metal3() {
            eprintln!("[quadcoop] skipping: no Metal3");
            return;
        }
        // Test shapes covering: small N (W_K=1024), W_O-like (5120),
        // and N % 4 != 0 edge case.
        for &(k, n) in &[(5120usize, 5120usize), (5120, 1024), (5120, 27)] {
            let w_bytes = build_test_q4k_matrix(n, k, 91);
            let x: Vec<f32> = (0..k).map(|i| ((i as f32 + 1.0) * 0.001).sin()).collect();
            let x_buf = backend.alloc_shared(k * 4).unwrap();
            let w_buf = backend.alloc_shared(w_bytes.len()).unwrap();
            let out_single = backend.alloc_shared(n * 4).unwrap();
            let out_quad = backend.alloc_shared(n * 4).unwrap();
            unsafe {
                std::ptr::copy_nonoverlapping(x.as_ptr(), x_buf.contents() as *mut f32, k);
                std::ptr::copy_nonoverlapping(
                    w_bytes.as_ptr(),
                    w_buf.contents() as *mut u8,
                    w_bytes.len(),
                );
            }
            sgemv_q4_k_f32_into(backend, &x_buf, &w_buf, &out_single, k, n).unwrap();
            backend.drain();
            sgemv_q4_k_f32_quadcoop_into(backend, &x_buf, &w_buf, &out_quad, k, n).unwrap();
            backend.drain();

            let s = unsafe {
                std::slice::from_raw_parts(out_single.contents() as *const f32, n).to_vec()
            };
            let q = unsafe {
                std::slice::from_raw_parts(out_quad.contents() as *const f32, n).to_vec()
            };
            for (a, b) in s.iter().zip(q.iter()) {
                let r = (a - b).abs() / a.abs().max(1e-3);
                assert!(
                    r < 1e-3,
                    "K={k} N={n} mismatch: single={a} quad={b} (rel {r:.3e})"
                );
            }
        }
    }

    /// T92 — Batched Q4_K sgemv must match B independent single-vector
    /// sgemv calls within FMA reorder tolerance.
    #[test]
    #[cfg(feature = "gpu-tests")]
    fn sgemv_q4_k_f32_batch_matches_singles() {
        let backend = metal_backend();
        if !backend.supports_metal3() {
            eprintln!("[batch] skipping: no Metal3");
            return;
        }
        let k = 5120usize;
        let n = 5120usize;
        let b = 4usize;

        let w_bytes = build_test_q4k_matrix(n, k, 137);
        let w_buf = backend.alloc_shared(w_bytes.len()).unwrap();
        unsafe {
            std::ptr::copy_nonoverlapping(
                w_bytes.as_ptr(),
                w_buf.contents() as *mut u8,
                w_bytes.len(),
            );
        }

        // Build B different x vectors (different seeds) packed contiguous.
        let mut x_all = vec![0.0f32; b * k];
        for batch in 0..b {
            for i in 0..k {
                x_all[batch * k + i] = ((i as f32 + 1.0) * 0.001 * (batch as f32 + 1.0)).sin();
            }
        }
        let x_buf = backend.alloc_shared(b * k * 4).unwrap();
        unsafe {
            std::ptr::copy_nonoverlapping(x_all.as_ptr(), x_buf.contents() as *mut f32, b * k);
        }

        // Reference: B separate sgemv calls.
        let mut y_ref = vec![0.0f32; b * n];
        for batch in 0..b {
            let xb_buf = backend.alloc_shared(k * 4).unwrap();
            let yb_buf = backend.alloc_shared(n * 4).unwrap();
            unsafe {
                std::ptr::copy_nonoverlapping(
                    x_all[batch * k..(batch + 1) * k].as_ptr(),
                    xb_buf.contents() as *mut f32,
                    k,
                );
            }
            sgemv_q4_k_f32_into(backend, &xb_buf, &w_buf, &yb_buf, k, n).unwrap();
            backend.drain();
            unsafe {
                std::ptr::copy_nonoverlapping(
                    yb_buf.contents() as *const f32,
                    y_ref[batch * n..(batch + 1) * n].as_mut_ptr(),
                    n,
                );
            }
        }

        // Batched call.
        let y_batch_buf = backend.alloc_shared(b * n * 4).unwrap();
        sgemv_q4_k_f32_batch_into(backend, &x_buf, &w_buf, &y_batch_buf, k, n, b).unwrap();
        backend.drain();
        let y_batch = unsafe {
            std::slice::from_raw_parts(y_batch_buf.contents() as *const f32, b * n).to_vec()
        };

        for batch in 0..b {
            for j in 0..n {
                let a = y_ref[batch * n + j];
                let b_val = y_batch[batch * n + j];
                let r = (a - b_val).abs() / a.abs().max(1e-3);
                assert!(
                    r < 1e-3,
                    "batch={batch} j={j} mismatch: single={a} batch={b_val} (rel {r:.3e})"
                );
            }
        }
    }

    /// T106 — Batched RMSNorm must match B sequential rms_norm_f32 calls.
    #[test]
    #[cfg(feature = "gpu-tests")]
    fn rms_norm_batched_f32_matches_singles() {
        let backend = metal_backend();
        let d = 5120usize;
        let b = 4usize;
        let eps = 1e-6_f32;

        // Build B different x rows
        let mut x_all = vec![0.0f32; b * d];
        for batch in 0..b {
            for i in 0..d {
                x_all[batch * d + i] = ((i as f32 + 1.0) * 0.001 * (batch as f32 + 1.0)).sin();
            }
        }
        let gamma: Vec<f32> = (0..d)
            .map(|i| 0.5 + ((i as f32 * 0.001).cos()) * 0.5)
            .collect();

        let x_buf = backend.alloc_shared(b * d * 4).unwrap();
        let gamma_buf = backend.alloc_shared(d * 4).unwrap();
        let y_seq = backend.alloc_shared(b * d * 4).unwrap();
        let y_batch = backend.alloc_shared(b * d * 4).unwrap();

        unsafe {
            std::ptr::copy_nonoverlapping(x_all.as_ptr(), x_buf.contents() as *mut f32, b * d);
            std::ptr::copy_nonoverlapping(gamma.as_ptr(), gamma_buf.contents() as *mut f32, d);
        }

        // Reference: B sequential calls
        for batch in 0..b {
            let xb_buf = backend.alloc_shared(d * 4).unwrap();
            let yb_buf = backend.alloc_shared(d * 4).unwrap();
            unsafe {
                std::ptr::copy_nonoverlapping(
                    x_all[batch * d..(batch + 1) * d].as_ptr(),
                    xb_buf.contents() as *mut f32,
                    d,
                );
            }
            rms_norm_f32(backend, &xb_buf, &gamma_buf, &yb_buf, d, eps).unwrap();
            backend.drain();
            unsafe {
                let dst = (y_seq.contents() as *mut f32).add(batch * d);
                std::ptr::copy_nonoverlapping(yb_buf.contents() as *const f32, dst, d);
            }
        }

        // Batched call
        rms_norm_batched_f32(backend, &x_buf, &gamma_buf, &y_batch, d, b, eps).unwrap();
        backend.drain();

        let seq =
            unsafe { std::slice::from_raw_parts(y_seq.contents() as *const f32, b * d).to_vec() };
        let bat =
            unsafe { std::slice::from_raw_parts(y_batch.contents() as *const f32, b * d).to_vec() };
        for batch in 0..b {
            for i in 0..d {
                let a = seq[batch * d + i];
                let bv = bat[batch * d + i];
                let r = (a - bv).abs() / a.abs().max(1e-4);
                assert!(
                    r < 1e-3,
                    "batch={batch} i={i} mismatch: seq={a} batch={bv} (rel {r:.3e})"
                );
            }
        }
    }

    /// T107 — Batched RoPE must match B sequential rope_half_split calls
    /// at consecutive positions (pos_base + b for b in 0..B).
    #[test]
    #[cfg(feature = "gpu-tests")]
    fn rope_half_split_batched_f32_matches_singles() {
        let backend = metal_backend();
        let n_heads = 4usize;
        let head_dim = 64usize;
        let max_seq = 32usize;
        let pos_base = 5usize;
        let b = 4usize;
        let row_size = n_heads * head_dim;

        // Build B different rows
        let mut x_all = vec![0.0f32; b * row_size];
        for batch in 0..b {
            for i in 0..row_size {
                x_all[batch * row_size + i] = ((i as f32 + 1.0 + batch as f32) * 0.01).sin();
            }
        }
        // Build cos/sin tables [max_seq, head_dim/2]
        let half_dim = head_dim / 2;
        let mut cos_tab = vec![0.0f32; max_seq * half_dim];
        let mut sin_tab = vec![0.0f32; max_seq * half_dim];
        for p in 0..max_seq {
            for k in 0..half_dim {
                let theta = (p as f32) * 0.001 * ((k + 1) as f32);
                cos_tab[p * half_dim + k] = theta.cos();
                sin_tab[p * half_dim + k] = theta.sin();
            }
        }

        let cos_buf = backend.alloc_shared(max_seq * half_dim * 4).unwrap();
        let sin_buf = backend.alloc_shared(max_seq * half_dim * 4).unwrap();
        unsafe {
            std::ptr::copy_nonoverlapping(
                cos_tab.as_ptr(),
                cos_buf.contents() as *mut f32,
                max_seq * half_dim,
            );
            std::ptr::copy_nonoverlapping(
                sin_tab.as_ptr(),
                sin_buf.contents() as *mut f32,
                max_seq * half_dim,
            );
        }

        // Reference: B sequential calls
        let mut x_seq = vec![0.0f32; b * row_size];
        for batch in 0..b {
            let xb_buf = backend.alloc_shared(row_size * 4).unwrap();
            unsafe {
                std::ptr::copy_nonoverlapping(
                    x_all[batch * row_size..(batch + 1) * row_size].as_ptr(),
                    xb_buf.contents() as *mut f32,
                    row_size,
                );
            }
            rope_half_split_f32(
                backend,
                &xb_buf,
                &cos_buf,
                &sin_buf,
                n_heads,
                head_dim,
                pos_base + batch,
            )
            .unwrap();
            backend.drain();
            unsafe {
                std::ptr::copy_nonoverlapping(
                    xb_buf.contents() as *const f32,
                    x_seq[batch * row_size..(batch + 1) * row_size].as_mut_ptr(),
                    row_size,
                );
            }
        }

        // Batched call
        let x_batch_buf = backend.alloc_shared(b * row_size * 4).unwrap();
        unsafe {
            std::ptr::copy_nonoverlapping(
                x_all.as_ptr(),
                x_batch_buf.contents() as *mut f32,
                b * row_size,
            );
        }
        rope_half_split_batched_f32(
            backend,
            &x_batch_buf,
            &cos_buf,
            &sin_buf,
            n_heads,
            head_dim,
            pos_base,
            b,
        )
        .unwrap();
        backend.drain();
        let x_batch = unsafe {
            std::slice::from_raw_parts(x_batch_buf.contents() as *const f32, b * row_size).to_vec()
        };

        for batch in 0..b {
            for i in 0..row_size {
                let a = x_seq[batch * row_size + i];
                let bv = x_batch[batch * row_size + i];
                let r = (a - bv).abs() / a.abs().max(1e-4);
                assert!(
                    r < 1e-4,
                    "batch={batch} i={i} mismatch: seq={a} batch={bv} (rel {r:.3e})"
                );
            }
        }
    }

    /// T108 — Batched KV append must match B sequential kv_append calls.
    #[test]
    #[cfg(feature = "gpu-tests")]
    fn kv_append_batched_f32_matches_singles() {
        let backend = metal_backend();
        let n_kv = 4usize;
        let head_dim = 64usize;
        let max_seq = 32usize;
        let pos_base = 3usize;
        let b = 4usize;
        let per_token = n_kv * head_dim;

        // Source data
        let mut src_all = vec![0.0f32; b * per_token];
        for batch in 0..b {
            for i in 0..per_token {
                src_all[batch * per_token + i] =
                    ((i as f32 + 1.0 + batch as f32 * 7.0) * 0.01).sin();
            }
        }

        // Reference: B sequential calls
        let dst_cache_seq = backend.alloc_shared(n_kv * max_seq * head_dim * 4).unwrap();
        // Zero-fill the cache
        unsafe {
            let p = dst_cache_seq.contents() as *mut f32;
            for i in 0..(n_kv * max_seq * head_dim) {
                *p.add(i) = 0.0;
            }
        }
        for batch in 0..b {
            let src_b = backend.alloc_shared(per_token * 4).unwrap();
            unsafe {
                std::ptr::copy_nonoverlapping(
                    src_all[batch * per_token..(batch + 1) * per_token].as_ptr(),
                    src_b.contents() as *mut f32,
                    per_token,
                );
            }
            kv_append_f32(
                backend,
                &src_b,
                &dst_cache_seq,
                n_kv,
                head_dim,
                pos_base + batch,
                max_seq,
            )
            .unwrap();
            backend.drain();
        }

        // Batched call
        let dst_cache_batch = backend.alloc_shared(n_kv * max_seq * head_dim * 4).unwrap();
        unsafe {
            let p = dst_cache_batch.contents() as *mut f32;
            for i in 0..(n_kv * max_seq * head_dim) {
                *p.add(i) = 0.0;
            }
        }
        let src_buf = backend.alloc_shared(b * per_token * 4).unwrap();
        unsafe {
            std::ptr::copy_nonoverlapping(
                src_all.as_ptr(),
                src_buf.contents() as *mut f32,
                b * per_token,
            );
        }
        kv_append_batched_f32(
            backend,
            &src_buf,
            &dst_cache_batch,
            n_kv,
            head_dim,
            pos_base,
            b,
            max_seq,
        )
        .unwrap();
        backend.drain();

        let seq = unsafe {
            std::slice::from_raw_parts(
                dst_cache_seq.contents() as *const f32,
                n_kv * max_seq * head_dim,
            )
            .to_vec()
        };
        let bat = unsafe {
            std::slice::from_raw_parts(
                dst_cache_batch.contents() as *const f32,
                n_kv * max_seq * head_dim,
            )
            .to_vec()
        };
        for i in 0..(n_kv * max_seq * head_dim) {
            assert_eq!(
                seq[i], bat[i],
                "kv_append cache mismatch at flat idx {i}: seq={} batch={}",
                seq[i], bat[i]
            );
        }
    }
}
