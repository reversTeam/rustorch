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
// T151 — sgemv_f32_lcpp_simd_into — F32 sgemv `y = W @ x` for the GGUF
// Linear-weight layout (output rows of input columns, i.e. data is laid out
// as `w[n_idx * K + k]`). Used by `HybridMetalWeight::matmul_into` for the
// F32 fallback case (ssm_alpha / ssm_beta on Qwen3.5/3.6 are stored as F32
// because they are not large enough to be quantised). Replaces the previous
// `backend.drain() + CPU naive matmul` path which incurred ~250 µs per
// matmul × 64 calls/token = ~16 ms/token = ~23% of the 27B decode budget.
//
// One simdgroup (32 threads) per output. Each lane strides over K with step
// 32 and accumulates a partial sum, then `simd_sum` reduces across the lanes.
// W is read with row-of-output stride: `w[n_idx * K + k]` — same layout as
// the CPU fallback (and matches the GGUF Linear-weight convention where
// shape `[K_in, N_out]` stores data with output as the slow axis).
//
// No drain is required: caller passes a pre-allocated `out_buf`, and the
// kernel issues into the current command buffer like every other sgemv.
// =============================================================================

const SGEMV_F32_LCPP_SIMD_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void sgemv_f32_lcpp_simd(
    device const float*  x      [[buffer(0)]],   // [K] activation
    device const float*  w      [[buffer(1)]],   // [N, K] row-major (output is slow axis)
    device float*        y      [[buffer(2)]],   // [N] output
    constant uint2&      dims   [[buffer(3)]],   // (K, N)
    uint                 tg_id  [[threadgroup_position_in_grid]],
    ushort               lane   [[thread_index_in_simdgroup]]
) {
    uint K = dims.x;
    uint N = dims.y;
    uint n_idx = tg_id;
    if (n_idx >= N) return;

    uint base = n_idx * K;
    float partial = 0.0;
    for (uint k = lane; k < K; k += 32u) {
        partial += w[base + k] * x[k];
    }
    float sum = simd_sum(partial);
    if (lane == 0) {
        y[n_idx] = sum;
    }
}
"#;

/// T151 — F32 sgemv `y = W @ x` for GGUF Linear-weight layout (output rows
/// of input columns). One simdgroup per output column; no drain needed.
/// Used as the GPU replacement for the prior CPU F32 fallback in
/// `HybridMetalWeight::matmul_into`.
pub fn sgemv_f32_lcpp_simd_into(
    backend: &MetalBackend,
    x_buf: &Buffer,
    w_buf: &Buffer,
    out_buf: &Buffer,
    k: usize,
    n: usize,
) -> Result<(), MetalError> {
    if k == 0 || n == 0 {
        return Err(MetalError::ShapeMismatch(format!(
            "sgemv_f32_lcpp_simd_into: K, N must be > 0 (got K={k}, N={n})"
        )));
    }
    let pipeline = backend.pipeline(
        "sgemv_f32_lcpp_simd",
        SGEMV_F32_LCPP_SIMD_SHADER,
        "sgemv_f32_lcpp_simd",
    )?;
    let dims = [k as u32, n as u32];
    backend.with_encoder(|encoder| {
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(x_buf), 0);
        encoder.set_buffer(1, Some(w_buf), 0);
        encoder.set_buffer(2, Some(out_buf), 0);
        encoder.set_bytes(3, 8, dims.as_ptr() as *const std::ffi::c_void);
        let tg = MTLSize::new(32, 1, 1);
        let grid = MTLSize::new(32 * n as u64, 1, 1);
        encoder.dispatch_threads(grid, tg);
    });
    Ok(())
}

// =============================================================================
// T162 phase 1 — F32 SGEMM avec simdgroup_matrix Apple Matrix Engine.
//
// PROOF-OF-CONCEPT pour le path simdgroup_matrix (Apple AMX). Si gain confirmé
// vs sgemv-en-boucle, débloque T162 phase 2 (Q4_K simdgroup_matrix) qui est
// l'optim majeure pour matcher MLX en single-batch decode et exploser le
// prefill (rustorch actuellement 22 % llama.cpp prefill car pas de path SGEMM).
//
// Architecture :
// - 1 simdgroup (32 threads) = 1 TG, produit 1 tile output 8×8
// - Boucle sur K en chunks de 8 : load A_tile [8,8] et B_tile [8,8] depuis DRAM,
//   simdgroup_multiply_accumulate(C, A, B, C) en 1 instruction Apple AMX
// - Store C tile [8,8] à la fin
// - Pattern direct issu de MLX `mma.h::BaseMMAFrag<T,8,8>::mma`
//
// Pré-conditions : M, N, K multiples de 8 (sinon fallback path naïf).
//
// Performance attendue : `simdgroup_multiply_accumulate` exécute 8×8×8 = 512 FMA
// en 1 cycle (= 1024 FLOPS/cycle). À ~4 GHz/SM × 32 SMs M4 Max théorique ~13 TFLOPS
// fp32. Notre sgemv-en-boucle actuel fait ~32 FMA/cycle (1 simdgroup × 32 lanes
// × 1 FMA chacun) = 32× moins dense. Donc gain théorique 32× sur le compute pur,
// limité en pratique par la bande passante DRAM (400 GB/s sur M4 Max).
const SGEMM_F32_SIMDGROUP_MATRIX_SHADER: &str = r#"
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

kernel void sgemm_f32_simdgroup_matrix(
    device const float*  A      [[buffer(0)]],   // [M, K] row-major
    device const float*  B      [[buffer(1)]],   // [K, N] row-major
    device float*        C      [[buffer(2)]],   // [M, N] row-major
    constant uint3&      dims   [[buffer(3)]],   // (M, N, K)
    uint                 tg_id  [[threadgroup_position_in_grid]]
) {
    uint M = dims.x;
    uint N = dims.y;
    uint K = dims.z;

    uint n_tiles_n = N / 8u;
    uint m_tile = tg_id / n_tiles_n;
    uint n_tile = tg_id % n_tiles_n;

    if (m_tile * 8u >= M || n_tile * 8u >= N) return;

    simdgroup_matrix<float, 8, 8> A_frag;
    simdgroup_matrix<float, 8, 8> B_frag;
    simdgroup_matrix<float, 8, 8> C_frag = simdgroup_matrix<float, 8, 8>(0.0);

    uint n_k_tiles = K / 8u;
    for (uint k_tile = 0; k_tile < n_k_tiles; ++k_tile) {
        device const float* A_ptr =
            A + (uint64_t)m_tile * 8u * (uint64_t)K + (uint64_t)k_tile * 8u;
        device const float* B_ptr =
            B + (uint64_t)k_tile * 8u * (uint64_t)N + (uint64_t)n_tile * 8u;

        simdgroup_load(A_frag, A_ptr, K);
        simdgroup_load(B_frag, B_ptr, N);

        simdgroup_multiply_accumulate(C_frag, A_frag, B_frag, C_frag);
    }

    device float* C_ptr =
        C + (uint64_t)m_tile * 8u * (uint64_t)N + (uint64_t)n_tile * 8u;
    simdgroup_store(C_frag, C_ptr, N);
}
"#;

/// T162 phase 1 — F32 SGEMM `C = A @ B` avec simdgroup_matrix 8×8.
///
/// Pré-conditions : M, N, K multiples de 8. Erreur sinon.
/// Layout row-major pour A, B, C.
///
/// Cette fonction est un POC pour valider le path Apple AMX. Si le bench
/// montre un gain significatif vs sgemv-en-boucle (M=1 itéré), T162 phase 2
/// portera le pattern à Q4_K (`affine_qmm_t` MLX-style).
pub fn sgemm_f32_simdgroup_matrix_into(
    backend: &MetalBackend,
    a_buf: &Buffer,
    b_buf: &Buffer,
    c_buf: &Buffer,
    m: usize,
    n: usize,
    k: usize,
) -> Result<(), MetalError> {
    if !backend.supports_metal3() {
        return Err(MetalError::Unsupported(
            "sgemm_f32_simdgroup_matrix needs Metal3".to_string(),
        ));
    }
    if m == 0 || n == 0 || k == 0 || m % 8 != 0 || n % 8 != 0 || k % 8 != 0 {
        return Err(MetalError::ShapeMismatch(format!(
            "sgemm_f32_simdgroup_matrix: M, N, K must be > 0 and multiple of 8 (got M={m}, N={n}, K={k})"
        )));
    }
    let pipeline = backend.pipeline(
        "sgemm_f32_simdgroup_matrix",
        SGEMM_F32_SIMDGROUP_MATRIX_SHADER,
        "sgemm_f32_simdgroup_matrix",
    )?;
    let dims = [m as u32, n as u32, k as u32];
    backend.with_encoder(|encoder| {
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(a_buf), 0);
        encoder.set_buffer(1, Some(b_buf), 0);
        encoder.set_buffer(2, Some(c_buf), 0);
        encoder.set_bytes(3, 12, dims.as_ptr() as *const std::ffi::c_void);
        // 1 simdgroup per TG = 32 threads. Each TG produces 1 8x8 output tile.
        let tg_size = MTLSize::new(32, 1, 1);
        let n_tg = ((m / 8) * (n / 8)) as u64;
        let groups = MTLSize::new(n_tg, 1, 1);
        encoder.dispatch_thread_groups(groups, tg_size);
    });
    Ok(())
}

// =============================================================================
// T162 phase 2 — Q4_K SGEMM avec simdgroup_matrix + dequant inline.
//
// Premier kernel rustorch combinant Apple Matrix Engine + Q4_K. C'est le
// gros levier de la roadmap "fastest engine" : `simdgroup_multiply_accumulate`
// donne 8×8×8 = 512 FMA/cycle vs 32 FMA/cycle pour notre sgemv-en-boucle,
// et la lecture Q4_K (-7× DRAM vs F32) reste possible via dequant inline en
// threadgroup memory.
//
// Architecture :
// - 1 simdgroup (32 threads) = 1 TG, produit 1 tile C 8×8
// - Threadgroup mem (4 KB total) :
//     Xs[8 × 32] f32 = morceau de A[8 rows × 32 cols K]
//     Ws[8 × 32] f32 = morceau de W[8 rows N × 32 cols K] dequant
// - Boucle externe : K en chunks de BK=32 (1 sub-block Q4_K)
//   Pour chaque chunk :
//     1) 32 threads chargent coopérativement Xs depuis A (8 floats/thread)
//     2) 32 threads dequant coopérativement Ws depuis bytes Q4_K (8 weights/thread)
//     3) simdgroup_barrier
//     4) 4 MMAs (BK=32 = 4 fragments de 8 sur K) :
//          load A_frag[8x8] depuis Xs (row-major)
//          load B_frag[8x8] depuis Ws TRANSPOSE (Ws[n,k] → B[k,n])
//          C += A_frag @ B_frag
// - Store C 8×8 vers DRAM
//
// Pré-conditions :
// - M, N multiples de 8 ; K multiple de 256 (Q4_K super-block)
// - W layout : [N, K] row-major, K/256 super-blocks de 144 bytes par row
//
// Ce kernel s'utilise pour M ≥ 8 (prefill, batched, spec decoding verify).
// Pour M=1 (decode autoregressive), le path sgemv (T132 NSG=2 NR0=2) reste
// optimal.
const SGEMM_Q4_K_F32_SIMDGROUP_MATRIX_SHADER: &str = r#"
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

constant uint Q4K_BYTES = 144u;
constant uint Q4K_WEIGHTS = 256u;
constant uint BM = 8u;
constant uint BN = 8u;
constant uint BK = 32u;

kernel void sgemm_q4_k_f32_simdgroup_matrix(
    device const float*  A      [[buffer(0)]],   // [M, K] f32 row-major
    device const uchar*  W_q4k  [[buffer(1)]],   // [N, K] Q4_K
    device float*        C      [[buffer(2)]],   // [M, N] f32 row-major
    constant uint3&      dims   [[buffer(3)]],   // (M, N, K)
    uint                 tg_id  [[threadgroup_position_in_grid]],
    ushort               tiisg  [[thread_index_in_simdgroup]]
) {
    uint M = dims.x;
    uint N = dims.y;
    uint K = dims.z;

    uint n_tiles_n = N / BN;
    uint m_tile = tg_id / n_tiles_n;
    uint n_tile = tg_id % n_tiles_n;
    if (m_tile * BM >= M || n_tile * BN >= N) return;

    threadgroup float Xs[8 * 32];
    threadgroup float Ws[8 * 32];

    simdgroup_matrix<float, 8, 8> C_frag = simdgroup_matrix<float, 8, 8>(0.0);
    simdgroup_matrix<float, 8, 8> A_frag;
    simdgroup_matrix<float, 8, 8> B_frag;

    uint blocks_per_row = K / Q4K_WEIGHTS;
    uint row_stride_bytes = blocks_per_row * Q4K_BYTES;

    // Each thread's coordinates inside the [BM=8 rows, BK=32 cols] tile.
    ushort row       = tiisg / 4u;        // 0..7 — selects N row (and A row)
    ushort col_chunk = tiisg % 4u;        // 0..3 — chunk of 8 cols within BK

    for (uint k_offset = 0; k_offset < K; k_offset += BK) {
        // 1. Load Xs[8 × 32] from A[m_tile*8 .. +8, k_offset .. +32].
        uint a_row_base =
            (uint)(m_tile * BM + row) * K + k_offset + (uint)col_chunk * 8u;
        threadgroup float* xs_row = Xs + (uint)row * BK + (uint)col_chunk * 8u;
        for (ushort c = 0; c < 8u; ++c) {
            xs_row[c] = A[a_row_base + c];
        }

        // 2. Dequant Ws[8 × 32] from Q4_K bytes.
        uint super_block_idx = k_offset / Q4K_WEIGHTS;
        uint sb_in_super = (k_offset % Q4K_WEIGHTS) / 32u;  // 0..7
        uint pair_idx    = sb_in_super / 2u;                // 0..3
        bool is_high     = (sb_in_super & 1u) != 0u;

        uint n_actual = n_tile * BN + (uint)row;
        device const uchar* row_block = W_q4k
            + (uint64_t)n_actual * row_stride_bytes
            + (uint64_t)super_block_idx * Q4K_BYTES;
        device const half* d_ptr = (device const half*)(row_block);
        float d    = float(d_ptr[0]);
        float dmin = float(d_ptr[1]);

        // Unpack 6-bit scales/mins (cf rustorch-gguf::unpack_q4_k_sc_m).
        device const uchar* sc_raw = row_block + 4;
        uchar sc6, m6;
        if (sb_in_super < 4u) {
            sc6 = sc_raw[sb_in_super]      & 0x3Fu;
            m6  = sc_raw[sb_in_super + 4u] & 0x3Fu;
        } else {
            uint i = sb_in_super - 4u;
            sc6 = (sc_raw[i + 8u] & 0x0Fu) | ((sc_raw[i]      >> 6u) << 4u);
            m6  = (sc_raw[i + 8u] >> 4u)   | ((sc_raw[i + 4u] >> 6u) << 4u);
        }
        float scale   = d    * float(sc6);
        float min_val = dmin * float(m6);

        device const uchar* qs_ptr = row_block + 16u + pair_idx * 32u;
        threadgroup float* ws_row = Ws + (uint)row * BK + (uint)col_chunk * 8u;
        for (ushort c = 0; c < 8u; ++c) {
            ushort byte_pos = col_chunk * 8u + c;
            uchar byte_val  = qs_ptr[byte_pos];
            uchar nibble    = is_high ? (byte_val >> 4u) : (byte_val & 0x0Fu);
            ws_row[c] = scale * float(nibble) - min_val;
        }

        simdgroup_barrier(mem_flags::mem_threadgroup);

        // 3. 4 MMAs : BK=32 → 4 fragments de 8 le long de K.
        // T162 phase 2-bis : unroll explicite (apprentissage de phase 3-bis).
        #pragma clang loop unroll(full)
        for (ushort k_frag = 0; k_frag < 4u; ++k_frag) {
            simdgroup_load(A_frag, Xs + (uint)k_frag * 8u, BK);
            // B is W transposed : load with transpose=true. stride = BK
            // (Ws is stored [BN=8 rows, BK=32 cols]).
            simdgroup_load(
                B_frag,
                Ws + (uint)k_frag * 8u,
                BK,
                ulong2(0, 0),
                /* transpose */ true);
            simdgroup_multiply_accumulate(C_frag, A_frag, B_frag, C_frag);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }

    // 4. Store C tile.
    device float* C_ptr = C
        + (uint64_t)m_tile * BM * (uint64_t)N
        + (uint64_t)n_tile * BN;
    simdgroup_store(C_frag, C_ptr, N);
}
"#;

/// T162 phase 2 — Q4_K SGEMM via simdgroup_matrix avec dequant inline.
///
/// `C = A @ W^T` où A est `[M, K]` f32 row-major, W est `[N, K]` Q4_K
/// (row-major en super-blocks de 256 weights × 144 bytes), C est `[M, N]`
/// f32 row-major.
///
/// Pré-conditions :
/// - Metal3 (Apple7+)
/// - M, N multiples de 8 ; K multiple de 256
///
/// Performance attendue : à M=128, K=1024, N=1024 : 7-10× plus rapide que
/// sgemv_q4_k-en-boucle (path prefill rustorch actuel). Pour M=1, sgemv
/// reste optimal.
pub fn sgemm_q4_k_f32_simdgroup_matrix_into(
    backend: &MetalBackend,
    a_buf: &Buffer,
    w_q4k_buf: &Buffer,
    c_buf: &Buffer,
    m: usize,
    n: usize,
    k: usize,
) -> Result<(), MetalError> {
    if !backend.supports_metal3() {
        return Err(MetalError::Unsupported(
            "sgemm_q4_k_f32_simdgroup_matrix needs Metal3".to_string(),
        ));
    }
    if m == 0 || n == 0 || k == 0 || m % 8 != 0 || n % 8 != 0 || k % 256 != 0 {
        return Err(MetalError::ShapeMismatch(format!(
            "sgemm_q4_k_f32_simdgroup_matrix: M, N must be multiples of 8 and K multiple of 256 (got M={m}, N={n}, K={k})"
        )));
    }
    let pipeline = backend.pipeline(
        "sgemm_q4_k_f32_simdgroup_matrix",
        SGEMM_Q4_K_F32_SIMDGROUP_MATRIX_SHADER,
        "sgemm_q4_k_f32_simdgroup_matrix",
    )?;
    let dims = [m as u32, n as u32, k as u32];
    backend.with_encoder(|encoder| {
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(a_buf), 0);
        encoder.set_buffer(1, Some(w_q4k_buf), 0);
        encoder.set_buffer(2, Some(c_buf), 0);
        encoder.set_bytes(3, 12, dims.as_ptr() as *const std::ffi::c_void);
        let tg_size = MTLSize::new(32, 1, 1);
        let n_tg = ((m / 8) * (n / 8)) as u64;
        let groups = MTLSize::new(n_tg, 1, 1);
        encoder.dispatch_thread_groups(groups, tg_size);
    });
    Ok(())
}

// =============================================================================
// T163 phase 9f-ter — Q4_K SGEMM EXPERT-MAJOR pour MoE batched.
//
// Variante du SGEMM Q4_K 8×8 single-warp avec INDIRECTION par M-tile. Au lieu
// d'une seule weight matrix, chaque M-tile lit un buffer de tile_expert_ids
// et offset W vers W_stacked + expert_id * expert_stride_bytes.
//
// Pré-condition caller :
// - Rows triées par expert (rows consécutives utilisent même expert)
// - Padding à mult-8 par groupe expert (pour aligner avec BM=8)
// - tile_expert_ids[M/8] : un expert_id par M-tile (= 8 rows)
//
// Output : C[M, N] où chaque tile-row de 8 utilise un expert différent.
// Le caller scatter ensuite les rows valides vers moe_acc avec pondération.
//
// Avantages vs gather sgemv :
// - SGEMM tile 8×8 → throughput MMA simdgroup_matrix (≈4-8× sgemv)
// - Single dispatch pour tous les expert evals (vs N_dispatches × n_used)
// - GPU saturation contrôlée par padding (M total = nb_evals_padded)
const SGEMM_Q4_K_F32_EXPERT_MAJOR_8X8_SHADER: &str = r#"
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

constant uint Q4K_BYTES_EM = 144u;
constant uint Q4K_WEIGHTS_EM = 256u;
constant uint BM_EM = 8u;
constant uint BN_EM = 8u;
constant uint BK_EM = 32u;

kernel void sgemm_q4_k_f32_expert_major_8x8(
    device const float*  A         [[buffer(0)]],   // [M, K] f32 row-major (gathered/packed)
    device const uchar*  W_stacked [[buffer(1)]],   // [E, N, K] Q4_K stacked
    device const uint*   tile_expert_ids [[buffer(2)]], // [M/8] expert id par M-tile
    device float*        C         [[buffer(3)]],   // [M, N] f32 row-major
    constant uint3&      dims      [[buffer(4)]],   // (M, N, K)
    constant uint&       expert_stride [[buffer(5)]], // bytes per expert (= N*(K/256)*144)
    uint                 tg_id     [[threadgroup_position_in_grid]],
    ushort               tiisg     [[thread_index_in_simdgroup]]
) {
    uint M = dims.x;
    uint N = dims.y;
    uint K = dims.z;

    uint n_tiles_n = N / BN_EM;
    uint m_tile = tg_id / n_tiles_n;
    uint n_tile = tg_id % n_tiles_n;
    if (m_tile * BM_EM >= M || n_tile * BN_EM >= N) return;

    // Per-tile expert lookup : all 8 rows in this M-tile share the same expert.
    uint expert_id = tile_expert_ids[m_tile];
    device const uchar* W_q4k = W_stacked + (uint64_t)expert_id * (uint64_t)expert_stride;

    // Float SHM + float MMA : le tile 8×8 single-warp est sensible à l'accumulation
    // error half (rel diff > 10% sur K=512). Le 64×64 multi-warp tolère bien
    // half MMA (parité < 1e-2) probablement grâce au plus grand fanout.
    threadgroup float Xs[8 * 32];
    threadgroup float Ws[8 * 32];

    simdgroup_matrix<float, 8, 8> C_frag = simdgroup_matrix<float, 8, 8>(0.0);
    simdgroup_matrix<float, 8, 8> A_frag;
    simdgroup_matrix<float, 8, 8> B_frag;

    uint blocks_per_row = K / Q4K_WEIGHTS_EM;
    uint row_stride_bytes = blocks_per_row * Q4K_BYTES_EM;

    ushort row       = tiisg / 4u;
    ushort col_chunk = tiisg % 4u;

    for (uint k_offset = 0; k_offset < K; k_offset += BK_EM) {
        // 1. Load Xs from A.
        uint a_row_base =
            (uint)(m_tile * BM_EM + row) * K + k_offset + (uint)col_chunk * 8u;
        threadgroup float* xs_row = Xs + (uint)row * BK_EM + (uint)col_chunk * 8u;
        for (ushort c = 0; c < 8u; ++c) {
            xs_row[c] = A[a_row_base + c];
        }

        // 2. Dequant Ws from Q4_K bytes (using expert_id-offsetted W).
        uint super_block_idx = k_offset / Q4K_WEIGHTS_EM;
        uint sb_in_super = (k_offset % Q4K_WEIGHTS_EM) / 32u;
        uint pair_idx    = sb_in_super / 2u;
        bool is_high     = (sb_in_super & 1u) != 0u;

        uint n_actual = n_tile * BN_EM + (uint)row;
        device const uchar* row_block = W_q4k
            + (uint64_t)n_actual * row_stride_bytes
            + (uint64_t)super_block_idx * Q4K_BYTES_EM;
        device const half* d_ptr = (device const half*)(row_block);
        float d    = float(d_ptr[0]);
        float dmin = float(d_ptr[1]);

        device const uchar* sc_raw = row_block + 4;
        uchar sc6, m6;
        if (sb_in_super < 4u) {
            sc6 = sc_raw[sb_in_super]      & 0x3Fu;
            m6  = sc_raw[sb_in_super + 4u] & 0x3Fu;
        } else {
            uint i = sb_in_super - 4u;
            sc6 = (sc_raw[i + 8u] & 0x0Fu) | ((sc_raw[i]      >> 6u) << 4u);
            m6  = (sc_raw[i + 8u] >> 4u)   | ((sc_raw[i + 4u] >> 6u) << 4u);
        }
        float scale   = d    * float(sc6);
        float min_val = dmin * float(m6);

        device const uchar* qs_ptr = row_block + 16u + pair_idx * 32u;
        threadgroup float* ws_row = Ws + (uint)row * BK_EM + (uint)col_chunk * 8u;
        for (ushort c = 0; c < 8u; ++c) {
            ushort byte_pos = col_chunk * 8u + c;
            uchar byte_val  = qs_ptr[byte_pos];
            uchar nibble    = is_high ? (byte_val >> 4u) : (byte_val & 0x0Fu);
            ws_row[c] = scale * float(nibble) - min_val;
        }

        simdgroup_barrier(mem_flags::mem_threadgroup);

        #pragma clang loop unroll(full)
        for (ushort k_frag = 0; k_frag < 4u; ++k_frag) {
            simdgroup_load(A_frag, Xs + (uint)k_frag * 8u, BK_EM);
            simdgroup_load(
                B_frag,
                Ws + (uint)k_frag * 8u,
                BK_EM,
                ulong2(0, 0),
                /* transpose */ true);
            simdgroup_multiply_accumulate(C_frag, A_frag, B_frag, C_frag);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }

    device float* C_ptr = C
        + (uint64_t)m_tile * BM_EM * (uint64_t)N
        + (uint64_t)n_tile * BN_EM;
    simdgroup_store(C_frag, C_ptr, N);
}
"#;

/// T163 phase 9f-ter — Q4_K SGEMM EXPERT-MAJOR pour MoE batched.
///
/// Pour MoE prefill : caller permute les expert evaluations par expert_id et
/// pad à mult-8 par groupe expert. Le kernel dispatch comme un SGEMM standard
/// mais lit `tile_expert_ids[m_tile]` pour offsetter W_stacked.
///
/// Pré-conditions :
/// - Metal3 (Apple7+)
/// - M%8==0, N%8==0, K%256==0
/// - tile_expert_ids buffer : exactly M/8 entries u32
/// - Tous les rows d'un même 8-row M-tile partagent le même expert_id
#[allow(clippy::too_many_arguments)]
pub fn sgemm_q4_k_f32_expert_major_8x8_into(
    backend: &MetalBackend,
    a_buf: &Buffer,
    w_stacked_buf: &Buffer,
    tile_expert_ids_buf: &Buffer,
    c_buf: &Buffer,
    m: usize,
    n: usize,
    k: usize,
    expert_stride_bytes: usize,
) -> Result<(), MetalError> {
    if !backend.supports_metal3() {
        return Err(MetalError::Unsupported(
            "sgemm_q4_k_f32_expert_major needs Metal3".to_string(),
        ));
    }
    if m == 0 || n == 0 || k == 0 || m % 8 != 0 || n % 8 != 0 || k % 256 != 0 {
        return Err(MetalError::ShapeMismatch(format!(
            "sgemm_q4_k_f32_expert_major: M%8==0, N%8==0, K%256==0 (got M={m}, N={n}, K={k})"
        )));
    }
    let pipeline = backend.pipeline(
        "sgemm_q4_k_f32_expert_major_8x8",
        SGEMM_Q4_K_F32_EXPERT_MAJOR_8X8_SHADER,
        "sgemm_q4_k_f32_expert_major_8x8",
    )?;
    let dims = [m as u32, n as u32, k as u32];
    let stride_u32 = expert_stride_bytes as u32;
    backend.with_encoder(|encoder| {
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(a_buf), 0);
        encoder.set_buffer(1, Some(w_stacked_buf), 0);
        encoder.set_buffer(2, Some(tile_expert_ids_buf), 0);
        encoder.set_buffer(3, Some(c_buf), 0);
        encoder.set_bytes(4, 12, dims.as_ptr() as *const std::ffi::c_void);
        encoder.set_bytes(5, 4, &stride_u32 as *const u32 as *const std::ffi::c_void);
        let tg_size = MTLSize::new(32, 1, 1);
        let n_tg = ((m / 8) * (n / 8)) as u64;
        let groups = MTLSize::new(n_tg, 1, 1);
        encoder.dispatch_thread_groups(groups, tg_size);
    });
    Ok(())
}

// =============================================================================
// T163 phase 9f-quater — gather rows from x according to permutation.
//
// Pour expert-major MoE pipeline : pack les rows de x dans un layout permuté
// où les rows d'un même expert sont contiguës.
//
// Input  : x [B, K] activations (per-token), src_indices [M_padded] (= permutation
//           mais stocké directement comme src token row indices, sentinel 0xFFFFFFFF
//           pour padding zero-rows).
// Output : x_packed [M_padded, K]
//
// src_indices[i] doit être :
//   - VALID : token_row_idx = b ∈ [0, B), copies x[b, :] vers x_packed[i, :]
//   - PADDING : 0xFFFFFFFF, écrit zeros à x_packed[i, :]
const GATHER_PACK_ROWS_F32_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

constant uint SENTINEL_PAD = 0xFFFFFFFFu;

kernel void gather_pack_rows_f32(
    device const float* x           [[buffer(0)]],   // [B, K]
    device const uint*  src_indices [[buffer(1)]],   // [M_padded]
    device float*       x_packed    [[buffer(2)]],   // [M_padded, K]
    constant uint2&     dims        [[buffer(3)]],   // (K, M_padded)
    uint2               gid         [[thread_position_in_grid]]
) {
    uint K = dims.x;
    uint M_padded = dims.y;
    uint k = gid.x;
    uint i = gid.y;
    if (k >= K || i >= M_padded) return;

    uint src = src_indices[i];
    if (src == SENTINEL_PAD) {
        x_packed[i * K + k] = 0.0f;
    } else {
        x_packed[i * K + k] = x[src * K + k];
    }
}
"#;

/// T163 phase 9f-quater — gather x rows par permutation pour expert-major MoE.
///
/// `src_indices[i]` :
/// - valid : token row idx (∈ [0, B))  → x_packed[i, :] = x[src, :]
/// - 0xFFFFFFFF : padding zero-row     → x_packed[i, :] = 0
///
/// Économie vs CPU memcpy : 1 dispatch, GPU-side, pas de drain CPU.
pub fn gather_pack_rows_f32(
    backend: &MetalBackend,
    x_buf: &Buffer,
    src_indices_buf: &Buffer,
    x_packed_buf: &Buffer,
    k: usize,
    m_padded: usize,
) -> Result<(), MetalError> {
    if k == 0 || m_padded == 0 {
        return Err(MetalError::ShapeMismatch(format!(
            "gather_pack_rows_f32: K={k}, M_padded={m_padded}"
        )));
    }
    let pipeline = backend.pipeline(
        "gather_pack_rows_f32",
        GATHER_PACK_ROWS_F32_SHADER,
        "gather_pack_rows_f32",
    )?;
    let dims = [k as u32, m_padded as u32];
    backend.with_encoder(|encoder| {
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(x_buf), 0);
        encoder.set_buffer(1, Some(src_indices_buf), 0);
        encoder.set_buffer(2, Some(x_packed_buf), 0);
        encoder.set_bytes(3, 8, dims.as_ptr() as *const std::ffi::c_void);
        let tg_size = MTLSize::new(64, 1, 1);
        let grid = MTLSize::new(k as u64, m_padded as u64, 1);
        encoder.dispatch_threads(grid, tg_size);
    });
    Ok(())
}

// =============================================================================
// T163 phase 9f-quater — weighted scatter add pour expert-major MoE.
//
// Inverse de gather_pack_rows. Pour chaque row i ∈ [0, M_padded) qui est valide
// (src_indices[i] != SENTINEL), accumule out_packed[i] pondéré par topw[src]
// dans moe_acc[token_idx, :] où token_idx = src / n_used.
//
// Input  : out_packed [M_padded, D], src_indices [M_padded], topw [B, n_used]
// Output : moe_acc [B, D] in/out (caller doit zero ou pré-charger)
const WEIGHTED_SCATTER_ADD_F32_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

constant uint SCATTER_SENTINEL_PAD = 0xFFFFFFFFu;

kernel void weighted_scatter_add_f32(
    device const float* out_packed  [[buffer(0)]],   // [M_padded, D]
    device const uint*  src_indices [[buffer(1)]],   // [M_padded]
    device const float* topw        [[buffer(2)]],   // [B, n_used] (flat)
    device       atomic_float* moe_acc [[buffer(3)]], // [B, D]
    constant uint3&     dims        [[buffer(4)]],   // (D, M_padded, n_used)
    uint2               gid         [[thread_position_in_grid]]
) {
    uint D = dims.x;
    uint M_padded = dims.y;
    uint n_used = dims.z;
    uint d = gid.x;
    uint i = gid.y;
    if (d >= D || i >= M_padded) return;

    uint src = src_indices[i];
    if (src == SCATTER_SENTINEL_PAD) return;

    uint b = src / n_used;
    float w = topw[src];
    float v = w * out_packed[i * D + d];

    // Atomic add to moe_acc[b, d] : multiple i can map to same b (n_used different
    // experts per token) → need atomic.
    atomic_fetch_add_explicit(&moe_acc[b * D + d], v, memory_order_relaxed);
}
"#;

/// T163 phase 9f-quater — weighted scatter add pour expert-major MoE.
///
/// Pour chaque row valide i (src != sentinel) :
///   moe_acc[src/n_used, :] += topw[src] * out_packed[i, :]
///
/// Atomic add nécessaire : n_used différents `i` (= different experts per token)
/// peuvent mapper vers le même `b = src/n_used`.
///
/// Pré-condition : `moe_acc` doit être pré-zéroé par caller (zero_f32 ou eq.)
pub fn weighted_scatter_add_f32(
    backend: &MetalBackend,
    out_packed_buf: &Buffer,
    src_indices_buf: &Buffer,
    topw_buf: &Buffer,
    moe_acc_buf: &Buffer,
    d: usize,
    m_padded: usize,
    n_used: usize,
) -> Result<(), MetalError> {
    if d == 0 || m_padded == 0 || n_used == 0 {
        return Err(MetalError::ShapeMismatch(format!(
            "weighted_scatter_add_f32: D={d}, M_padded={m_padded}, n_used={n_used}"
        )));
    }
    let pipeline = backend.pipeline(
        "weighted_scatter_add_f32",
        WEIGHTED_SCATTER_ADD_F32_SHADER,
        "weighted_scatter_add_f32",
    )?;
    let dims = [d as u32, m_padded as u32, n_used as u32];
    backend.with_encoder(|encoder| {
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(out_packed_buf), 0);
        encoder.set_buffer(1, Some(src_indices_buf), 0);
        encoder.set_buffer(2, Some(topw_buf), 0);
        encoder.set_buffer(3, Some(moe_acc_buf), 0);
        encoder.set_bytes(4, 12, dims.as_ptr() as *const std::ffi::c_void);
        let tg_size = MTLSize::new(64, 1, 1);
        let grid = MTLSize::new(d as u64, m_padded as u64, 1);
        encoder.dispatch_threads(grid, tg_size);
    });
    Ok(())
}

// =============================================================================
// T162 phase 3 — Q4_K SGEMM avec tiles 64×64 multi-warp.
//
// Évolution de phase 2 (tile 8×8 single-simdgroup) : passage à BM=BN=64 avec
// 4 simdgroups par TG (WM=WN=2). Chaque simdgroup gère un quadrant 32×32 =
// 16 fragments 8×8. Réduit le nombre de TGs de 64× (64×64 tile vs 8×8) →
// kill le dispatch overhead qui plafonnait la phase 2 à 2× sur grandes shapes.
//
// Architecture (alignée sur MLX `qmm_t_impl` BlockMMA<WM=2, WN=2>) :
// - TG : 128 threads = 4 simdgroups × 32. Output tile BM×BN = 64×64.
// - Threadgroup mem (16 KB) :
//     Xs[BM=64, BK=32] f32 = 8 KB
//     Ws[BN=64, BK=32] f32 = 8 KB (dequant Q4_K)
// - Cooperative load : 128 threads × 16 elements = 2048 = BM × BK. Chaque
//   thread charge/dequant 16 elements par K-iteration.
// - Par K-iteration (BK=32 = 1 sub-block Q4_K) :
//     load Xs (cooperative)
//     dequant Ws (cooperative, formula scale × nibble - min)
//     threadgroup_barrier
//     chaque simdgroup : 4 K-fragments × 4×4 MMAs = 64 simdgroup MMAs
// - Output stage : 4 simdgroups écrivent leurs 16 C fragments chacun.
//
// Pré-conditions : M, N multiples de 64 ; K multiple de 256.
const SGEMM_Q4_K_F32_SIMDGROUP_MATRIX_64_SHADER: &str = r#"
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

constant uint Q4K_BYTES = 144u;
constant uint Q4K_WEIGHTS = 256u;
constant uint BM = 64u;
constant uint BN = 64u;
constant uint BK = 32u;
constant uint WM = 2u;
constant uint WN = 2u;
constant uint TM = 32u;        // BM / WM
constant uint TN = 32u;        // BN / WN
constant uint FM = 4u;         // TM / 8 : 4 fragments per simdgroup row
constant uint FN = 4u;         // TN / 8 : 4 fragments per simdgroup col

kernel void sgemm_q4_k_f32_simdgroup_matrix_64(
    device const float*  A      [[buffer(0)]],
    device const uchar*  W_q4k  [[buffer(1)]],
    device float*        C      [[buffer(2)]],
    constant uint3&      dims   [[buffer(3)]],
    uint                 tg_id  [[threadgroup_position_in_grid]],
    ushort               tiisg  [[thread_index_in_simdgroup]],
    ushort               sgitg  [[simdgroup_index_in_threadgroup]]
) {
    uint M = dims.x;
    uint N = dims.y;
    uint K = dims.z;

    uint n_tiles_n = N / BN;
    uint m_tile = tg_id / n_tiles_n;
    uint n_tile = tg_id % n_tiles_n;
    if (m_tile * BM >= M || n_tile * BN >= N) return;

    uint sgi = (uint)sgitg / WN;       // 0..WM-1 = 0..1
    uint sgj = (uint)sgitg % WN;       // 0..WN-1 = 0..1

    // T162 phase 9h-bis : SHM en HALF (vs float). MMAs simdgroup_half8x8 plus
    // rapides sur Apple GPU. Réduit aussi la SHM de 16KB → 8KB.
    threadgroup half Xs[64 * 32];     // BM × BK
    threadgroup half Ws[64 * 32];     // BN × BK

    // 16 C fragments per simdgroup, accumulators.
    simdgroup_matrix<float, 8, 8> C_frag[4][4];
    for (uint i = 0; i < FM; ++i) {
        for (uint j = 0; j < FN; ++j) {
            C_frag[i][j] = simdgroup_matrix<float, 8, 8>(0.0);
        }
    }

    uint blocks_per_row = K / Q4K_WEIGHTS;
    uint row_stride_bytes = blocks_per_row * Q4K_BYTES;

    // Cooperative load layout : 128 threads, each handles 16 elements.
    uint tid = (uint)sgitg * 32u + (uint)tiisg;   // 0..127
    uint load_row = tid / 2u;                      // 0..63
    uint load_chunk = tid % 2u;                    // 0..1 (cols 0..16 or 16..32)

    for (uint k_offset = 0; k_offset < K; k_offset += BK) {
        // Phase 1 : load Xs[BM=64, BK=32] = 2048 halves coopérativement.
        uint a_row_global = m_tile * BM + load_row;
        threadgroup half* xs_dst = Xs + load_row * BK + load_chunk * 16u;
        if (a_row_global < M) {
            uint a_base = a_row_global * K + k_offset + load_chunk * 16u;
            for (uint c = 0; c < 16u; ++c) {
                xs_dst[c] = (half)A[a_base + c];
            }
        } else {
            for (uint c = 0; c < 16u; ++c) {
                xs_dst[c] = (half)0.0;
            }
        }

        // Phase 2 : dequant Ws[BN=64, BK=32] coopérativement.
        uint super_block_idx = k_offset / Q4K_WEIGHTS;
        uint sb_in_super = (k_offset % Q4K_WEIGHTS) / 32u;
        uint pair_idx = sb_in_super / 2u;
        bool is_high = (sb_in_super & 1u) != 0u;

        uint w_row_global = n_tile * BN + load_row;
        threadgroup half* ws_dst = Ws + load_row * BK + load_chunk * 16u;
        if (w_row_global < N) {
            device const uchar* row_block = W_q4k
                + (uint64_t)w_row_global * row_stride_bytes
                + (uint64_t)super_block_idx * Q4K_BYTES;
            device const half* d_ptr = (device const half*)(row_block);
            float d    = float(d_ptr[0]);
            float dmin = float(d_ptr[1]);

            device const uchar* sc_raw = row_block + 4;
            uchar sc6, m6;
            if (sb_in_super < 4u) {
                sc6 = sc_raw[sb_in_super]      & 0x3Fu;
                m6  = sc_raw[sb_in_super + 4u] & 0x3Fu;
            } else {
                uint i = sb_in_super - 4u;
                sc6 = (sc_raw[i + 8u] & 0x0Fu) | ((sc_raw[i]      >> 6u) << 4u);
                m6  = (sc_raw[i + 8u] >> 4u)   | ((sc_raw[i + 4u] >> 6u) << 4u);
            }
            float scale   = d    * float(sc6);
            float min_val = dmin * float(m6);

            device const uchar* qs_ptr = row_block + 16u + pair_idx * 32u;
            for (uint c = 0; c < 16u; ++c) {
                uint byte_pos  = load_chunk * 16u + c;
                uchar byte_val = qs_ptr[byte_pos];
                uchar nibble   = is_high ? (byte_val >> 4u) : (byte_val & 0x0Fu);
                ws_dst[c] = (half)(scale * float(nibble) - min_val);
            }
        } else {
            for (uint c = 0; c < 16u; ++c) {
                ws_dst[c] = (half)0.0;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Phase 3 : 4 K-fragments × 4×4 MMAs per simdgroup.
        // T162 phase 3-bis : full unroll des boucles fragment pour permettre
        // au compilateur Apple Metal de pipeliner les MMAs.
        // T162 phase 9g : simdgroup_barrier hints entre phases (pattern llama.cpp)
        // — aide le compilateur à serialize les loads avant les MMAs et eviter
        // les dépendances false sur les registres simdgroup.
        #pragma clang loop unroll(full)
        for (uint k_frag = 0; k_frag < BK / 8u; ++k_frag) {
            simdgroup_matrix<half, 8, 8> A_frags[4];
            simdgroup_matrix<half, 8, 8> B_frags[4];

            simdgroup_barrier(mem_flags::mem_none);
            #pragma clang loop unroll(full)
            for (uint i = 0; i < FM; ++i) {
                uint a_row = sgi * TM + i * 8u;
                simdgroup_load(A_frags[i], Xs + a_row * BK + k_frag * 8u, BK);
            }

            simdgroup_barrier(mem_flags::mem_none);
            #pragma clang loop unroll(full)
            for (uint j = 0; j < FN; ++j) {
                uint w_row = sgj * TN + j * 8u;
                simdgroup_load(
                    B_frags[j],
                    Ws + w_row * BK + k_frag * 8u,
                    BK,
                    ulong2(0, 0),
                    /* transpose */ true);
            }

            simdgroup_barrier(mem_flags::mem_none);
            // Outer product : C[i][j] += A[i] @ B[j], full unroll.
            #pragma clang loop unroll(full)
            for (uint i = 0; i < FM; ++i) {
                #pragma clang loop unroll(full)
                for (uint j = 0; j < FN; ++j) {
                    simdgroup_multiply_accumulate(
                        C_frag[i][j], A_frags[i], B_frags[j], C_frag[i][j]);
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Store C tile : 16 fragments × 4 simdgroups = 64 fragments écrits.
    for (uint i = 0; i < FM; ++i) {
        for (uint j = 0; j < FN; ++j) {
            uint c_row = m_tile * BM + sgi * TM + i * 8u;
            uint c_col = n_tile * BN + sgj * TN + j * 8u;
            if (c_row + 7u < M && c_col + 7u < N) {
                device float* C_ptr = C + (uint64_t)c_row * N + (uint64_t)c_col;
                simdgroup_store(C_frag[i][j], C_ptr, N);
            }
        }
    }
}
"#;

/// T162 phase 3 — Q4_K SGEMM tiles 64×64 multi-warp (4 simdgroups par TG).
///
/// Pré-conditions :
/// - Metal3 (Apple7+)
/// - M, N multiples de 64 ; K multiple de 256
///
/// Performance attendue : sur shapes ≥ 64×N×K, 5-7× plus rapide que la
/// phase 2 (tile 8×8). Réduit le TG count de 64× → dispatch overhead
/// éliminé sur grandes matmuls FFN du Qwen3-14B (5120×5120, 14336×5120).
pub fn sgemm_q4_k_f32_simdgroup_matrix_64_into(
    backend: &MetalBackend,
    a_buf: &Buffer,
    w_q4k_buf: &Buffer,
    c_buf: &Buffer,
    m: usize,
    n: usize,
    k: usize,
) -> Result<(), MetalError> {
    if !backend.supports_metal3() {
        return Err(MetalError::Unsupported(
            "sgemm_q4_k_f32_simdgroup_matrix_64 needs Metal3".to_string(),
        ));
    }
    if m == 0 || n == 0 || k == 0 || m % 64 != 0 || n % 64 != 0 || k % 256 != 0 {
        return Err(MetalError::ShapeMismatch(format!(
            "sgemm_q4_k_f32_simdgroup_matrix_64: M, N must be multiples of 64 and K multiple of 256 (got M={m}, N={n}, K={k})"
        )));
    }
    let pipeline = backend.pipeline(
        "sgemm_q4_k_f32_simdgroup_matrix_64",
        SGEMM_Q4_K_F32_SIMDGROUP_MATRIX_64_SHADER,
        "sgemm_q4_k_f32_simdgroup_matrix_64",
    )?;
    let dims = [m as u32, n as u32, k as u32];
    backend.with_encoder(|encoder| {
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(a_buf), 0);
        encoder.set_buffer(1, Some(w_q4k_buf), 0);
        encoder.set_buffer(2, Some(c_buf), 0);
        encoder.set_bytes(3, 12, dims.as_ptr() as *const std::ffi::c_void);
        let tg_size = MTLSize::new(128, 1, 1); // 4 simdgroups × 32
        let n_tg = ((m / 64) * (n / 64)) as u64;
        let groups = MTLSize::new(n_tg, 1, 1);
        encoder.dispatch_thread_groups(groups, tg_size);
    });
    Ok(())
}

// =============================================================================
// T162 phase 9h — port complet du kernel llama.cpp::kernel_mul_mm fallback Q4_K.
//
// Port direct de ggml-metal.metal::kernel_mul_mm template fallback (pre-M5 path).
// Différences clés vs notre `sgemm_q4_k_f32_simdgroup_matrix_64` :
//
// 1. **Tile aspect 64×32 (vs 64×64)** : NRA=64 rows of A, NRB=NK1=32 rows of B.
//    Per SG : 8 fragments (4 along A, 2 along B) au lieu de 16. Moins de
//    register pressure.
//
// 2. **Swizzled SHM layout 8x8 blocks** : `sa[ib*64 + ly*8 + lx]` au lieu de
//    `Ws[row*BK + col]`. simdgroup_load avec stride=8, no transpose au lieu
//    de stride=BK=32 transpose=true. Hardware-friendly.
//
// 3. **Outer product mc[i] = mb[i/4] @ ma[i%4]** : 8 frags par SG, structuré
//    pour minimiser les dépendances entre MMAs successives.
//
// 4. **Conventions** : llama.cpp utilise A=quantized weight [M=N_out, K=K_in]
//    et B=activations [K=K_in, N=M_tokens]. Pour matcher notre kernel
//    (A=activations [M_tok, K], W=weights [N_out, K], C=A @ W^T = [M_tok, N_out]),
//    on swap : la dimension M de llama.cpp = N_out chez nous, leur N = M_tok.
//    Donc tile produit M_tok=32 × N_out=64 par TG.
//
// Pré-conditions :
// - Metal3 (Apple7+)
// - M (= M_tok) multiple de 32 ; N (= N_out) multiple de 64 ; K multiple de 256
//
// Cible : closer le gap 14B Q4 prefill avec llama.cpp 454 t/s (vs 211 actuels).
const SGEMM_Q4_K_F32_LCPP_PORTED_SHADER: &str = r#"
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

constant uint Q4K_BYTES = 144u;
constant uint Q4K_WEIGHTS = 256u;

// Tile constants — match llama.cpp NR0=64 NR1=32 NK=32, swapped pour notre conv.
constant uint NR_W = 64u;     // BN : output rows = N_out per TG (= leur NRA=NR0)
constant uint NR_A = 32u;     // BM : tokens per TG (= leur NRB=NR1)
constant uint NK   = 32u;     // K tile per loop iter
constant uint NL_Q4K = 8u;    // 256 weights / 16 / 2 = 8 (Q4_K nl pour dequant)
constant uint NL0 = NK / 16u; // = 2 (work items per row in cooperative load)
constant uint NL1 = NK / 8u;  // = 4 (used in B load layout)

kernel void sgemm_q4_k_f32_lcpp_ported(
    device const float*  A      [[buffer(0)]],   // activations [M, K]
    device const uchar*  W_q4k  [[buffer(1)]],   // weights [N, K] Q4_K
    device float*        C      [[buffer(2)]],   // output [M, N]
    constant uint3&      dims   [[buffer(3)]],   // (M, N, K)
    uint3                tgpig  [[threadgroup_position_in_grid]],
    ushort               tiitg  [[thread_index_in_threadgroup]],
    ushort               sgitg  [[simdgroup_index_in_threadgroup]]
) {
    uint M = dims.x;
    uint N = dims.y;
    uint K = dims.z;

    // T162 phase 9h-bis : SHM en HALF (vs float). Réduit la BP SHM 2× et active
    // les MMAs simdgroup_half8x8 qui sont 2× plus rapides que float8x8 sur Apple
    // GPU. C'est CE qui fait la différence entre llama.cpp's 454 t/s et notre
    // 211 t/s sur 14B Q4 prefill (les opérateurs MMA half ont throughput double).
    //
    // sa : NR_W × NK = 64×32 = 2048 halves = 4KB (vs 8KB en float)
    // sb : NR_A × NK = 32×32 = 1024 halves = 2KB (vs 4KB en float)
    threadgroup half sa[64 * 32];
    threadgroup half sb[32 * 32];

    // Tile offsets (M_tok = activations row, N_out = output col).
    // tgpig.y = m-block index (0..M/NR_A), tgpig.x = n-block index (0..N/NR_W)
    uint m_tile = tgpig.y * NR_A;       // M-position (token index)
    uint n_tile = tgpig.x * NR_W;       // N-position (output dim)

    // Cooperative load layout — 128 threads (4 SG × 32).
    uint blocks_per_row_q4k = K / Q4K_WEIGHTS;
    uint w_row_stride_bytes = blocks_per_row_q4k * Q4K_BYTES;

    // For sa load (W dequant) : 128 threads × 16 elements = 2048 = NR_W × NK.
    // tiitg/NL0 = row index in tile (0..NR_W=64), tiitg%NL0 = k_chunk (0..1).
    uint sa_row = tiitg / NL0;          // 0..63 (within tile, = w_row_within_tile)
    uint sa_chunk = tiitg % NL0;        // 0..1 (k_chunk = 16 K-positions)

    // For sb load (A direct read) : same 128 threads × 8 elements = 1024 = NR_A × NK.
    // tiitg/NL1 = row index (0..NR_A=32), tiitg%NL1 = k_chunk_a (0..3).
    uint sb_row = tiitg / NL1;          // 0..31 (m_tok within tile)
    uint sb_chunk = tiitg % NL1;        // 0..3 (k_chunk = 8 K-positions)

    // 8 C fragments per SG : 4 along NR_W (sga axis), 2 along NR_A (sgb axis).
    // sgitg layout: sgitg%2 = a_block_idx (0..1, selects M-half), sgitg/2 = b_block_idx (0..1, selects N-half)
    simdgroup_matrix<float, 8, 8> mc[8];
    for (uint i = 0; i < 8u; ++i) {
        mc[i] = simdgroup_matrix<float, 8, 8>(0.0);
    }

    // Bounds clamps.
    uint nr_w_eff = (n_tile + NR_W < N) ? NR_W : (N - n_tile);
    uint nr_a_eff = (m_tile + NR_A < M) ? NR_A : (M - m_tile);

    for (uint k_offset = 0; k_offset < K; k_offset += NK) {
        // ============ Phase 1 : load+dequant W → sa swizzled (block 8x8 contigus) ============
        // sa[ib * 64 + ly*8 + lx], ib = 8*sx + sy, sx ∈ [0, NK/8=4), sy ∈ [0, NR_W/8=8)
        // Each thread (sa_row, sa_chunk) writes 16 floats : the 16 K-positions
        // for output row sa_row, K-positions [sa_chunk*16 .. sa_chunk*16+16).
        if (sa_row < nr_w_eff) {
            uint w_row_global = n_tile + sa_row;
            uint k_pos_base = k_offset + sa_chunk * 16u;

            // Q4_K block math : super-block = 256 weights.
            uint super_block_idx = k_pos_base / Q4K_WEIGHTS;
            uint sb_in_super = (k_pos_base % Q4K_WEIGHTS) / 32u;  // 0..7
            uint pair_idx = sb_in_super / 2u;                      // 0..3
            bool is_high = (sb_in_super & 1u) != 0u;

            device const uchar* row_block = W_q4k
                + (uint64_t)w_row_global * w_row_stride_bytes
                + (uint64_t)super_block_idx * Q4K_BYTES;
            device const half* d_ptr = (device const half*)(row_block);
            float d    = float(d_ptr[0]);
            float dmin = float(d_ptr[1]);

            device const uchar* sc_raw = row_block + 4;
            uchar sc6, m6;
            if (sb_in_super < 4u) {
                sc6 = sc_raw[sb_in_super]      & 0x3Fu;
                m6  = sc_raw[sb_in_super + 4u] & 0x3Fu;
            } else {
                uint i = sb_in_super - 4u;
                sc6 = (sc_raw[i + 8u] & 0x0Fu) | ((sc_raw[i]      >> 6u) << 4u);
                m6  = (sc_raw[i + 8u] >> 4u)   | ((sc_raw[i + 4u] >> 6u) << 4u);
            }
            float scale   = d    * float(sc6);
            float min_val = dmin * float(m6);

            device const uchar* qs_ptr = row_block + 16u + pair_idx * 32u;
            // sa stored as W^T (K rows × N_out cols swizzled, matching llama.cpp pattern).
            // Block layout : sx = K-block (0..NK/8=4), sy = N_out-block (0..NR_W/8=8).
            // ib = 8*sx + sy (sx outer, sy inner). Within block : ly = K%8, lx = N_out%8.
            // Each iter of K-loop advances `lsma += 8*64` (8 blocks = 8 sy values = next sx).
            //
            // Per-thread (sa_row=N_out_row, sa_chunk=K_chunk) : 16 elements at varying K-pos.
            //   sy = sa_row / 8     (0..7)
            //   lx = sa_row % 8     (col within block)
            //   For c in 0..16 : k_pos = sa_chunk*16+c → sx = k_pos/8, ly = k_pos%8
            uint sy = sa_row / 8u;
            uint lx = sa_row % 8u;
            #pragma clang loop unroll(full)
            for (uint c = 0; c < 16u; ++c) {
                uint k_in_tile = sa_chunk * 16u + c;
                uint sx = k_in_tile / 8u;
                uint ly = k_in_tile % 8u;
                uint ib = 8u * sx + sy;

                uchar byte_val = qs_ptr[k_in_tile];
                uchar nibble = is_high ? (byte_val >> 4u) : (byte_val & 0x0Fu);
                sa[ib * 64u + ly * 8u + lx] = scale * float(nibble) - min_val;
            }
        } else {
            uint sy = sa_row / 8u;
            uint lx = sa_row % 8u;
            #pragma clang loop unroll(full)
            for (uint c = 0; c < 16u; ++c) {
                uint k_in_tile = sa_chunk * 16u + c;
                uint sx = k_in_tile / 8u;
                uint ly = k_in_tile % 8u;
                uint ib = 8u * sx + sy;
                sa[ib * 64u + ly * 8u + lx] = 0.0;
            }
        }

        // ============ Phase 2 : load A → sb swizzled (block 8x8 contigus) ============
        // sb[ib * 64 + ly*8 + lx], ib = 4*sx + sy, sx ∈ [0, NK/8=4), sy ∈ [0, NR_A/8=4)
        // Each thread (sb_row, sb_chunk) loads 8 floats (one row of an 8x8 block).
        if (sb_row < nr_a_eff) {
            uint a_row_global = m_tile + sb_row;
            uint k_pos = k_offset + sb_chunk * 8u;
            uint sx = sb_chunk;                  // 0..3
            uint sy = sb_row / 8u;                // 0..3
            uint ly = sb_row % 8u;
            uint ib = 4u * sx + sy;
            #pragma clang loop unroll(full)
            for (uint lx = 0; lx < 8u; ++lx) {
                if (k_pos + lx < K) {
                    sb[ib * 64u + ly * 8u + lx] = A[a_row_global * K + k_pos + lx];
                } else {
                    sb[ib * 64u + ly * 8u + lx] = 0.0;
                }
            }
        } else {
            uint sx = sb_chunk;
            uint sy = sb_row / 8u;
            uint ly = sb_row % 8u;
            uint ib = 4u * sx + sy;
            #pragma clang loop unroll(full)
            for (uint lx = 0; lx < 8u; ++lx) {
                sb[ib * 64u + ly * 8u + lx] = 0.0;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ============ Phase 3 : MMAs (half precision sur Apple GPU = 2× throughput) ============
        // ma et mb en simdgroup_matrix<half, 8, 8>. mc reste float pour precision
        // d'accumulation (multiply_accumulate accepte mixed half×half→float).
        threadgroup const half* lsma = sa + 4u * 64u * (sgitg % 2u);  // 0 ou 256
        threadgroup const half* lsmb = sb + 2u * 64u * (sgitg / 2u);  // 0 ou 128

        #pragma clang loop unroll(full)
        for (uint ik = 0; ik < NK / 8u; ++ik) {
            simdgroup_matrix<half, 8, 8> ma[4];
            simdgroup_matrix<half, 8, 8> mb[2];

            simdgroup_barrier(mem_flags::mem_none);
            #pragma clang loop unroll(full)
            for (uint i = 0; i < 4u; ++i) {
                simdgroup_load(ma[i], lsma + 64u * i, 8);
            }

            simdgroup_barrier(mem_flags::mem_none);
            #pragma clang loop unroll(full)
            for (uint i = 0; i < 2u; ++i) {
                simdgroup_load(mb[i], lsmb + 64u * i, 8);
            }

            simdgroup_barrier(mem_flags::mem_none);
            // Outer product : mc[i] = mb[i/4] @ ma[i%4] (mais avec 4*2=8 frags,
            // i/4 ∈ {0,1} sélectionne mb, i%4 ∈ {0..3} sélectionne ma).
            #pragma clang loop unroll(full)
            for (uint i = 0; i < 8u; ++i) {
                simdgroup_multiply_accumulate(mc[i], mb[i / 4u], ma[i % 4u], mc[i]);
            }

            // Advance to next K-block (sx+1) : 8 blocks for sa (8 sy values),
            // 4 blocks for sb (4 sy values).
            lsma += 8u * 64u;
            lsmb += 4u * 64u;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ============ Store mc to C ============
    // Per SG :
    //   a_block_idx = sgitg % 2 (selects 32-row M-half of W)
    //   b_block_idx = sgitg / 2 (selects 16-col M-half of activations)
    // Each frag mc[i] : i%4 = ma slot (0..3, W-row offset), i/4 = mb slot (0..1, M-token offset).
    //
    // Output row in C : c_row = m_tile + (sgitg / 2) * 16 + (i / 4) * 8
    // Output col in C : c_col = n_tile + (sgitg % 2) * 32 + (i % 4) * 8
    // Wait — the result of mb @ ma is 8x8 with rows = M_tokens, cols = W_rows.
    // C[m_tok, n_out] = Σ_k A[m_tok, k] * W[n_out, k] = mb[m_tok-frag] @ ma[n_out-frag]^T... hmm.
    //
    // Actually simdgroup_multiply_accumulate(mc, mb, ma, mc) computes mc = mb @ ma + mc.
    // mb is 8x8 (rows from sb = M tokens × cols from sb K-direction).
    // ma is 8x8 (rows from sa = W K-direction × cols from sa W-direction).
    // mb @ ma → (8 M_tok rows × 8 K cols) @ (8 K rows × 8 W cols) = 8 M_tok × 8 W cols.
    // So mc has rows = M tokens, cols = W output rows = N_out.
    //
    // C[m_tok, n_out] storage : standard row-major [M, N].
    //   c_row = m_tile + (sgitg / 2) * 16 + (i / 4) * 8     (M token coordinate)
    //   c_col = n_tile + (sgitg % 2) * 32 + (i % 4) * 8     (N output dim coordinate)
    #pragma clang loop unroll(full)
    for (uint i = 0; i < 8u; ++i) {
        uint c_row = m_tile + (uint)(sgitg / 2u) * 16u + (i / 4u) * 8u;
        uint c_col = n_tile + (uint)(sgitg % 2u) * 32u + (i % 4u) * 8u;
        if (c_row + 7u < M && c_col + 7u < N) {
            device float* C_ptr = C + (uint64_t)c_row * N + (uint64_t)c_col;
            simdgroup_store(mc[i], C_ptr, N);
        }
    }
}
"#;

/// T162 phase 9h — port complet du kernel llama.cpp::kernel_mul_mm fallback Q4_K.
///
/// Pré-conditions :
/// - Metal3 (Apple7+)
/// - M multiple de 32 ; N multiple de 64 ; K multiple de 256
///
/// Cible : 14B Q4 prefill 211 → 350+ t/s (gap llama.cpp 454).
pub fn sgemm_q4_k_f32_lcpp_ported_into(
    backend: &MetalBackend,
    a_buf: &Buffer,
    w_q4k_buf: &Buffer,
    c_buf: &Buffer,
    m: usize,
    n: usize,
    k: usize,
) -> Result<(), MetalError> {
    if !backend.supports_metal3() {
        return Err(MetalError::Unsupported(
            "sgemm_q4_k_f32_lcpp_ported needs Metal3".to_string(),
        ));
    }
    if m == 0 || n == 0 || k == 0 || m % 32 != 0 || n % 64 != 0 || k % 256 != 0 {
        return Err(MetalError::ShapeMismatch(format!(
            "sgemm_q4_k_f32_lcpp_ported: M%32==0, N%64==0, K%256==0 required (got M={m}, N={n}, K={k})"
        )));
    }
    let pipeline = backend.pipeline(
        "sgemm_q4_k_f32_lcpp_ported",
        SGEMM_Q4_K_F32_LCPP_PORTED_SHADER,
        "sgemm_q4_k_f32_lcpp_ported",
    )?;
    let dims = [m as u32, n as u32, k as u32];
    backend.with_encoder(|encoder| {
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(a_buf), 0);
        encoder.set_buffer(1, Some(w_q4k_buf), 0);
        encoder.set_buffer(2, Some(c_buf), 0);
        encoder.set_bytes(3, 12, dims.as_ptr() as *const std::ffi::c_void);
        let tg_size = MTLSize::new(128, 1, 1); // 4 SG × 32 threads
                                               // Grid : tgpig.x = n-block (N/64), tgpig.y = m-block (M/32).
        let n_tg_x = (n / 64) as u64;
        let n_tg_y = (m / 32) as u64;
        let groups = MTLSize::new(n_tg_x, n_tg_y, 1);
        encoder.dispatch_thread_groups(groups, tg_size);
    });
    Ok(())
}

// =============================================================================
// T162 phase 5 — Q6_K SGEMM avec simdgroup_matrix + dequant inline.
//
// Port du pattern phase 2 (Q4_K tile 8×8) vers Q6_K. Le format Q6_K stocke
// 256 weights / 210 bytes en `ql[128]` (4 low bits par weight) + `qh[64]`
// (2 high bits par weight) + `scales_i8[16]` (8-bit signed) + `d` (f16).
// Chaque weight = d × sc_i8[sb] × (((ql_nibble) | (qh_bits << 4)) - 32).
//
// Découpage : 256 weights = 2 halves × 4 "rows" × 32 weights. Chaque row de
// 32 weights utilise une combinaison spécifique (half, row_in_half) pour le
// décodage de ql/qh/scales. Tableau de mapping (row_in_half ∈ 0..3) :
//   row_in_half | ql_byte_offset | qh_shift | use_high_nibble
//        0      |       l        |     0    |     false
//        1      |     l + 32     |     2    |     false
//        2      |       l        |     4    |     true
//        3      |     l + 32     |     6    |     true
//
// Sur Qwen3-14B Q4_K_M : V proj et FFN down sont stockés en Q6_K (pas Q4_K),
// ce kernel les couvre maintenant côté SGEMM batched. Speedup attendu ~2-4×
// vs sgemv-loop pour M ∈ [8, 128].
const SGEMM_Q6_K_F32_SIMDGROUP_MATRIX_SHADER: &str = r#"
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

constant uint Q6K_BYTES = 210u;
constant uint Q6K_WEIGHTS = 256u;
constant uint BM_Q6K = 8u;
constant uint BN_Q6K = 8u;
constant uint BK_Q6K = 32u;

kernel void sgemm_q6_k_f32_simdgroup_matrix(
    device const float*  A      [[buffer(0)]],   // [M, K] f32 row-major
    device const uchar*  W_q6k  [[buffer(1)]],   // [N, K] Q6_K
    device float*        C      [[buffer(2)]],   // [M, N] f32 row-major
    constant uint3&      dims   [[buffer(3)]],   // (M, N, K)
    uint                 tg_id  [[threadgroup_position_in_grid]],
    ushort               tiisg  [[thread_index_in_simdgroup]]
) {
    uint M = dims.x;
    uint N = dims.y;
    uint K = dims.z;

    uint n_tiles_n = N / BN_Q6K;
    uint m_tile = tg_id / n_tiles_n;
    uint n_tile = tg_id % n_tiles_n;
    if (m_tile * BM_Q6K >= M || n_tile * BN_Q6K >= N) return;

    threadgroup float Xs[8 * 32];
    threadgroup float Ws[8 * 32];

    simdgroup_matrix<float, 8, 8> C_frag = simdgroup_matrix<float, 8, 8>(0.0);
    simdgroup_matrix<float, 8, 8> A_frag;
    simdgroup_matrix<float, 8, 8> B_frag;

    uint blocks_per_row = K / Q6K_WEIGHTS;
    uint row_stride_bytes = blocks_per_row * Q6K_BYTES;

    ushort row       = tiisg / 4u;
    ushort col_chunk = tiisg % 4u;

    for (uint k_offset = 0; k_offset < K; k_offset += BK_Q6K) {
        // 1. Load Xs[8 × 32] from A[m_tile*8..+8, k_offset..+32].
        uint a_row_base =
            (uint)(m_tile * BM_Q6K + row) * K + k_offset + (uint)col_chunk * 8u;
        threadgroup float* xs_row = Xs + (uint)row * BK_Q6K + (uint)col_chunk * 8u;
        for (ushort c = 0; c < 8u; ++c) {
            xs_row[c] = A[a_row_base + c];
        }

        // 2. Dequant Ws[8 × 32] from Q6_K bytes.
        uint super_block_idx = k_offset / Q6K_WEIGHTS;
        uint sb_row = (k_offset % Q6K_WEIGHTS) / 32u;  // 0..7
        uint half_idx    = sb_row / 4u;                 // 0 or 1
        uint row_in_half = sb_row % 4u;                 // 0..3

        uint n_actual = n_tile * BN_Q6K + (uint)row;
        device const uchar* row_block = W_q6k
            + (uint64_t)n_actual * row_stride_bytes
            + (uint64_t)super_block_idx * Q6K_BYTES;

        device const half* d_ptr = (device const half*)(row_block + 208);
        float d_all = float(*d_ptr);

        device const uchar* ql_h = row_block + half_idx * 64u;
        device const uchar* qh_h = row_block + 128u + half_idx * 32u;
        device const char*  sc_h = (device const char*)(row_block + 192u + half_idx * 8u);

        uint qh_shift = row_in_half * 2u;
        bool use_high_nibble = (row_in_half >= 2u);
        uint ql_extra_offset = ((row_in_half & 1u) == 1u) ? 32u : 0u;

        threadgroup float* ws_row = Ws + (uint)row * BK_Q6K + (uint)col_chunk * 8u;
        #pragma clang loop unroll(full)
        for (ushort c = 0; c < 8u; ++c) {
            uint l = (uint)col_chunk * 8u + (uint)c;
            uint scale_idx = 2u * row_in_half + l / 16u;
            int sc6 = (int)sc_h[scale_idx];
            float scale = d_all * (float)sc6;

            uchar qh_byte = qh_h[l];
            uint qh_bits = ((uint)qh_byte >> qh_shift) & 0x03u;

            uchar ql_byte = ql_h[l + ql_extra_offset];
            uint ql_nibble = use_high_nibble ? ((uint)ql_byte >> 4u) : ((uint)ql_byte & 0x0Fu);

            int q = (int)(ql_nibble | (qh_bits << 4u));
            ws_row[c] = scale * (float)(q - 32);
        }

        simdgroup_barrier(mem_flags::mem_threadgroup);

        // 3. 4 MMAs : BK=32 → 4 fragments de 8 le long de K.
        #pragma clang loop unroll(full)
        for (ushort k_frag = 0; k_frag < 4u; ++k_frag) {
            simdgroup_load(A_frag, Xs + (uint)k_frag * 8u, BK_Q6K);
            simdgroup_load(
                B_frag,
                Ws + (uint)k_frag * 8u,
                BK_Q6K,
                ulong2(0, 0),
                /* transpose */ true);
            simdgroup_multiply_accumulate(C_frag, A_frag, B_frag, C_frag);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }

    device float* C_ptr = C
        + (uint64_t)m_tile * BM_Q6K * (uint64_t)N
        + (uint64_t)n_tile * BN_Q6K;
    simdgroup_store(C_frag, C_ptr, N);
}
"#;

/// T162 phase 5 — Q6_K SGEMM via simdgroup_matrix avec dequant inline.
///
/// Cible : V proj et FFN down de Qwen3-14B Q4_K_M (stockés en Q6_K).
///
/// `C = A @ W^T` où A est `[M, K]` f32 row-major, W est `[N, K]` Q6_K
/// (row-major en super-blocks de 256 weights × 210 bytes).
///
/// Pré-conditions : Metal3, M et N multiples de 8, K multiple de 256.
pub fn sgemm_q6_k_f32_simdgroup_matrix_into(
    backend: &MetalBackend,
    a_buf: &Buffer,
    w_q6k_buf: &Buffer,
    c_buf: &Buffer,
    m: usize,
    n: usize,
    k: usize,
) -> Result<(), MetalError> {
    if !backend.supports_metal3() {
        return Err(MetalError::Unsupported(
            "sgemm_q6_k_f32_simdgroup_matrix needs Metal3".to_string(),
        ));
    }
    if m == 0 || n == 0 || k == 0 || m % 8 != 0 || n % 8 != 0 || k % 256 != 0 {
        return Err(MetalError::ShapeMismatch(format!(
            "sgemm_q6_k_f32_simdgroup_matrix: M, N must be multiples of 8 and K multiple of 256 (got M={m}, N={n}, K={k})"
        )));
    }
    let pipeline = backend.pipeline(
        "sgemm_q6_k_f32_simdgroup_matrix",
        SGEMM_Q6_K_F32_SIMDGROUP_MATRIX_SHADER,
        "sgemm_q6_k_f32_simdgroup_matrix",
    )?;
    let dims = [m as u32, n as u32, k as u32];
    backend.with_encoder(|encoder| {
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(a_buf), 0);
        encoder.set_buffer(1, Some(w_q6k_buf), 0);
        encoder.set_buffer(2, Some(c_buf), 0);
        encoder.set_bytes(3, 12, dims.as_ptr() as *const std::ffi::c_void);
        let tg_size = MTLSize::new(32, 1, 1);
        let n_tg = ((m / 8) * (n / 8)) as u64;
        let groups = MTLSize::new(n_tg, 1, 1);
        encoder.dispatch_thread_groups(groups, tg_size);
    });
    Ok(())
}

// =============================================================================
// T162 phase 7-bis — Q3_K SGEMM tiles 64×64 multi-warp (4 simdgroups par TG).
//
// Combine le pattern multi-warp de phase 3-bis (Q4_K 64×64) + phase 5-bis
// (Q6_K 64×64) avec le dequant Q3_K de phase 7. Cible : modèles Q3_K_M
// (Qwen3-14B Q3_K_M) où FFN gate/up sont K=5120 N=17408 et bénéficient
// du multi-warp pour passer de ×2.6 (phase 7 8×8) à ×4-5 (phase 7-bis).
//
// Architecture : 128 threads = 4 simdgroups, output tile 64×64, threadgroup
// mem 16 KB, 16 C fragments par simdgroup. #pragma clang loop unroll(full)
// CRITIQUE sur les boucles MMA (gotcha 75efa9a).
//
// Q3_K dequant per-thread : 16 weights d'1 sub-block (avec BK=32 = 2 sub-blocks,
// chaque thread couvre 16 cols = 1 sub-block consécutif via load_chunk ∈ {0,1}).
const SGEMM_Q3_K_F32_SIMDGROUP_MATRIX_64_SHADER: &str = r#"
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

constant uint Q3K_BYTES_64 = 110u;
constant uint Q3K_WEIGHTS_64 = 256u;
constant uint BM_Q3K_64 = 64u;
constant uint BN_Q3K_64 = 64u;
constant uint BK_Q3K_64 = 32u;
constant uint WM_Q3K_64 = 2u;
constant uint WN_Q3K_64 = 2u;
constant uint TM_Q3K_64 = 32u;
constant uint TN_Q3K_64 = 32u;
constant uint FM_Q3K_64 = 4u;
constant uint FN_Q3K_64 = 4u;

kernel void sgemm_q3_k_f32_simdgroup_matrix_64(
    device const float*  A      [[buffer(0)]],
    device const uchar*  W_q3k  [[buffer(1)]],
    device float*        C      [[buffer(2)]],
    constant uint3&      dims   [[buffer(3)]],
    uint                 tg_id  [[threadgroup_position_in_grid]],
    ushort               tiisg  [[thread_index_in_simdgroup]],
    ushort               sgitg  [[simdgroup_index_in_threadgroup]]
) {
    uint M = dims.x;
    uint N = dims.y;
    uint K = dims.z;

    uint n_tiles_n = N / BN_Q3K_64;
    uint m_tile = tg_id / n_tiles_n;
    uint n_tile = tg_id % n_tiles_n;
    if (m_tile * BM_Q3K_64 >= M || n_tile * BN_Q3K_64 >= N) return;

    uint sgi = (uint)sgitg / WN_Q3K_64;
    uint sgj = (uint)sgitg % WN_Q3K_64;

    // Q3_K reste en float (half precision accumule >5% rel error sur K=512).
    threadgroup float Xs[64 * 32];
    threadgroup float Ws[64 * 32];

    simdgroup_matrix<float, 8, 8> C_frag[4][4];
    for (uint i = 0; i < FM_Q3K_64; ++i) {
        for (uint j = 0; j < FN_Q3K_64; ++j) {
            C_frag[i][j] = simdgroup_matrix<float, 8, 8>(0.0);
        }
    }

    uint blocks_per_row = K / Q3K_WEIGHTS_64;
    uint row_stride_bytes = blocks_per_row * Q3K_BYTES_64;

    uint tid = (uint)sgitg * 32u + (uint)tiisg;   // 0..127
    uint load_row = tid / 2u;                      // 0..63
    uint load_chunk = tid % 2u;                    // 0..1 (cols 0..16 or 16..32)

    for (uint k_offset = 0; k_offset < K; k_offset += BK_Q3K_64) {
        // Phase 1 : load Xs[BM=64, BK=32] coopératif.
        uint a_row_global = m_tile * BM_Q3K_64 + load_row;
        threadgroup float* xs_dst = Xs + load_row * BK_Q3K_64 + load_chunk * 16u;
        if (a_row_global < M) {
            uint a_base = a_row_global * K + k_offset + load_chunk * 16u;
            for (uint c = 0; c < 16u; ++c) {
                xs_dst[c] = A[a_base + c];
            }
        } else {
            for (uint c = 0; c < 16u; ++c) {
                xs_dst[c] = 0.0;
            }
        }

        // Phase 2 : dequant Ws[BN=64, BK=32] coopératif (Q3_K).
        // BK=32 = 2 Q3_K sub-blocks. load_chunk sélectionne lequel des 2.
        uint super_block_idx = k_offset / Q3K_WEIGHTS_64;
        uint sb_base = (k_offset % Q3K_WEIGHTS_64) / 16u;
        uint sb_global = sb_base + load_chunk;          // 0..15

        uint half_idx     = sb_global / 8u;
        uint sb_in_half   = sb_global % 8u;
        uint shift        = (sb_in_half / 2u) * 2u;
        uint qs_byte_base = half_idx * 32u + (sb_in_half % 2u) * 16u;
        uint hmask_base   = (sb_in_half % 2u) * 16u;
        uint j_global     = half_idx * 4u + (sb_in_half / 2u);
        uint m_bit        = 1u << j_global;

        uint w_row_global = n_tile * BN_Q3K_64 + load_row;
        threadgroup float* ws_dst = Ws + load_row * BK_Q3K_64 + load_chunk * 16u;
        if (w_row_global < N) {
            device const uchar* row_block = W_q3k
                + (uint64_t)w_row_global * row_stride_bytes
                + (uint64_t)super_block_idx * Q3K_BYTES_64;
            device const uchar* hmask = row_block;
            device const uchar* qs    = row_block + 32;
            device const uchar* sc_raw = row_block + 96;
            device const half*  d_ptr = (device const half*)(row_block + 108);
            float d_all = float(*d_ptr);

            uchar scale_byte;
            if (sb_global < 4u) {
                scale_byte = (sc_raw[sb_global] & 0x0Fu)
                           | ((sc_raw[8u + sb_global] & 0x03u) << 4u);
            } else if (sb_global < 8u) {
                uint i = sb_global - 4u;
                scale_byte = (sc_raw[4u + i] & 0x0Fu)
                           | (((sc_raw[8u + i] >> 2u) & 0x03u) << 4u);
            } else if (sb_global < 12u) {
                uint i = sb_global - 8u;
                scale_byte = (sc_raw[i] >> 4u)
                           | (((sc_raw[8u + i] >> 4u) & 0x03u) << 4u);
            } else {
                uint i = sb_global - 12u;
                scale_byte = (sc_raw[4u + i] >> 4u)
                           | (((sc_raw[8u + i] >> 6u) & 0x03u) << 4u);
            }
            float dl = d_all * (float)((int)((char)scale_byte) - 32);

            #pragma clang loop unroll(full)
            for (uint c = 0; c < 16u; ++c) {
                uint l = c;  // 0..16 within the sub-block
                uchar qs_byte    = qs[qs_byte_base + l];
                uchar hmask_byte = hmask[hmask_base + l];
                int q_lo  = (int)(((uint)qs_byte >> shift) & 0x03u);
                int h_bit = ((uint)hmask_byte & m_bit) != 0u;
                int v     = q_lo - (h_bit ? 0 : 4);
                ws_dst[c] = dl * (float)v;
            }
        } else {
            for (uint c = 0; c < 16u; ++c) {
                ws_dst[c] = 0.0;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Phase 3 : 4 K-fragments × 4×4 MMAs per simdgroup. UNROLL CRITIQUE.
        // T162 phase 9g : simdgroup_barrier hints (pattern llama.cpp).
        #pragma clang loop unroll(full)
        for (uint k_frag = 0; k_frag < BK_Q3K_64 / 8u; ++k_frag) {
            simdgroup_matrix<float, 8, 8> A_frags[4];
            simdgroup_matrix<float, 8, 8> B_frags[4];

            simdgroup_barrier(mem_flags::mem_none);
            #pragma clang loop unroll(full)
            for (uint i = 0; i < FM_Q3K_64; ++i) {
                uint a_row = sgi * TM_Q3K_64 + i * 8u;
                simdgroup_load(A_frags[i], Xs + a_row * BK_Q3K_64 + k_frag * 8u, BK_Q3K_64);
            }

            simdgroup_barrier(mem_flags::mem_none);
            #pragma clang loop unroll(full)
            for (uint j = 0; j < FN_Q3K_64; ++j) {
                uint w_row = sgj * TN_Q3K_64 + j * 8u;
                simdgroup_load(
                    B_frags[j],
                    Ws + w_row * BK_Q3K_64 + k_frag * 8u,
                    BK_Q3K_64,
                    ulong2(0, 0),
                    /* transpose */ true);
            }

            simdgroup_barrier(mem_flags::mem_none);
            #pragma clang loop unroll(full)
            for (uint i = 0; i < FM_Q3K_64; ++i) {
                #pragma clang loop unroll(full)
                for (uint j = 0; j < FN_Q3K_64; ++j) {
                    simdgroup_multiply_accumulate(
                        C_frag[i][j], A_frags[i], B_frags[j], C_frag[i][j]);
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint i = 0; i < FM_Q3K_64; ++i) {
        for (uint j = 0; j < FN_Q3K_64; ++j) {
            uint c_row = m_tile * BM_Q3K_64 + sgi * TM_Q3K_64 + i * 8u;
            uint c_col = n_tile * BN_Q3K_64 + sgj * TN_Q3K_64 + j * 8u;
            if (c_row + 7u < M && c_col + 7u < N) {
                device float* C_ptr = C + (uint64_t)c_row * N + (uint64_t)c_col;
                simdgroup_store(C_frag[i][j], C_ptr, N);
            }
        }
    }
}
"#;

/// T162 phase 7-bis — Q3_K SGEMM tiles 64×64 multi-warp.
///
/// Cible : modèles Q3_K_M (Qwen3-14B Q3_K_M dont FFN gate/up et Q proj
/// sont en Q3_K). Phase 7 (8×8) plafonne à ×2.6-2.7 ; phase 7-bis vise
/// ×4-5 comme Q6_K phase 5-bis l'a fait pour FFN down.
///
/// Pré-conditions : Metal3, M et N multiples de 64 ; K multiple de 256.
pub fn sgemm_q3_k_f32_simdgroup_matrix_64_into(
    backend: &MetalBackend,
    a_buf: &Buffer,
    w_q3k_buf: &Buffer,
    c_buf: &Buffer,
    m: usize,
    n: usize,
    k: usize,
) -> Result<(), MetalError> {
    if !backend.supports_metal3() {
        return Err(MetalError::Unsupported(
            "sgemm_q3_k_f32_simdgroup_matrix_64 needs Metal3".to_string(),
        ));
    }
    if m == 0 || n == 0 || k == 0 || m % 64 != 0 || n % 64 != 0 || k % 256 != 0 {
        return Err(MetalError::ShapeMismatch(format!(
            "sgemm_q3_k_f32_simdgroup_matrix_64: M, N must be multiples of 64 and K multiple of 256 (got M={m}, N={n}, K={k})"
        )));
    }
    let pipeline = backend.pipeline(
        "sgemm_q3_k_f32_simdgroup_matrix_64",
        SGEMM_Q3_K_F32_SIMDGROUP_MATRIX_64_SHADER,
        "sgemm_q3_k_f32_simdgroup_matrix_64",
    )?;
    let dims = [m as u32, n as u32, k as u32];
    backend.with_encoder(|encoder| {
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(a_buf), 0);
        encoder.set_buffer(1, Some(w_q3k_buf), 0);
        encoder.set_buffer(2, Some(c_buf), 0);
        encoder.set_bytes(3, 12, dims.as_ptr() as *const std::ffi::c_void);
        let tg_size = MTLSize::new(128, 1, 1);
        let n_tg = ((m / 64) * (n / 64)) as u64;
        let groups = MTLSize::new(n_tg, 1, 1);
        encoder.dispatch_thread_groups(groups, tg_size);
    });
    Ok(())
}

// =============================================================================
// T162 phase 7 — Q3_K SGEMM avec simdgroup_matrix + dequant inline.
//
// Étend la couverture batched matmul à Q3_K (le 4e K-quant : on a maintenant
// Q3, Q4, Q5 partial via existing, Q6 full). Q3_K format : 110 bytes /
// super-block de 256 weights : hmask[32] + qs[64] + scales[12] + d (f16).
// 16 sub-blocks de 16 weights, scales 6-bit signed packés dans 12 bytes.
//
// Per-thread : 8 weights d'une sub-block. Layout col_chunk ∈ 0..3 :
//   col_chunk=0 : sb_in_chunk=0, l_in_sb=0  (premières 8 weights sub 0)
//   col_chunk=1 : sb_in_chunk=0, l_in_sb=8  (dernières 8 weights sub 0)
//   col_chunk=2 : sb_in_chunk=1, l_in_sb=0  (premières 8 weights sub 1)
//   col_chunk=3 : sb_in_chunk=1, l_in_sb=8  (dernières 8 weights sub 1)
//
// BK=32 = 2 Q3_K sub-blocks consécutifs. K-iter avance par 32.
const SGEMM_Q3_K_F32_SIMDGROUP_MATRIX_SHADER: &str = r#"
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

constant uint Q3K_BYTES_SGEMM = 110u;
constant uint Q3K_WEIGHTS_SGEMM = 256u;
constant uint BM_Q3K = 8u;
constant uint BN_Q3K = 8u;
constant uint BK_Q3K = 32u;

kernel void sgemm_q3_k_f32_simdgroup_matrix(
    device const float*  A      [[buffer(0)]],
    device const uchar*  W_q3k  [[buffer(1)]],
    device float*        C      [[buffer(2)]],
    constant uint3&      dims   [[buffer(3)]],
    uint                 tg_id  [[threadgroup_position_in_grid]],
    ushort               tiisg  [[thread_index_in_simdgroup]]
) {
    uint M = dims.x;
    uint N = dims.y;
    uint K = dims.z;

    uint n_tiles_n = N / BN_Q3K;
    uint m_tile = tg_id / n_tiles_n;
    uint n_tile = tg_id % n_tiles_n;
    if (m_tile * BM_Q3K >= M || n_tile * BN_Q3K >= N) return;

    threadgroup float Xs[8 * 32];
    threadgroup float Ws[8 * 32];

    simdgroup_matrix<float, 8, 8> C_frag = simdgroup_matrix<float, 8, 8>(0.0);
    simdgroup_matrix<float, 8, 8> A_frag;
    simdgroup_matrix<float, 8, 8> B_frag;

    uint blocks_per_row = K / Q3K_WEIGHTS_SGEMM;
    uint row_stride_bytes = blocks_per_row * Q3K_BYTES_SGEMM;

    ushort row       = tiisg / 4u;
    ushort col_chunk = tiisg % 4u;

    // Per-thread Q3_K constants (sub-block within 2-sub-block K-chunk).
    uint sb_in_chunk = (uint)col_chunk / 2u;     // 0 or 1
    uint l_in_sb_base = ((uint)col_chunk % 2u) * 8u;  // 0 or 8 within sub-block

    for (uint k_offset = 0; k_offset < K; k_offset += BK_Q3K) {
        // 1. Load Xs[8 × 32] from A.
        uint a_row_base =
            (uint)(m_tile * BM_Q3K + row) * K + k_offset + (uint)col_chunk * 8u;
        threadgroup float* xs_row = Xs + (uint)row * BK_Q3K + (uint)col_chunk * 8u;
        for (ushort c = 0; c < 8u; ++c) {
            xs_row[c] = A[a_row_base + c];
        }

        // 2. Dequant Ws[8 × 32] from Q3_K bytes.
        uint super_block_idx = k_offset / Q3K_WEIGHTS_SGEMM;
        // The 2 Q3_K sub-blocks covered : sb_base, sb_base+1.
        uint sb_base = (k_offset % Q3K_WEIGHTS_SGEMM) / 16u;
        uint sb_global = sb_base + sb_in_chunk;     // 0..15

        uint half_idx     = sb_global / 8u;          // 0 or 1
        uint sb_in_half   = sb_global % 8u;          // 0..7
        uint shift        = (sb_in_half / 2u) * 2u;  // 0,2,4,6
        uint qs_byte_base = half_idx * 32u + (sb_in_half % 2u) * 16u;
        uint hmask_base   = (sb_in_half % 2u) * 16u;
        uint j_global     = half_idx * 4u + (sb_in_half / 2u);
        uint m_bit        = 1u << j_global;

        uint n_actual = n_tile * BN_Q3K + (uint)row;
        device const uchar* row_block = W_q3k
            + (uint64_t)n_actual * row_stride_bytes
            + (uint64_t)super_block_idx * Q3K_BYTES_SGEMM;
        device const uchar* hmask = row_block;
        device const uchar* qs    = row_block + 32;
        device const uchar* sc_raw = row_block + 96;
        device const half*  d_ptr = (device const half*)(row_block + 108);
        float d_all = float(*d_ptr);

        // Compute scale_byte for sb_global (4-way cascade).
        uchar scale_byte;
        if (sb_global < 4u) {
            scale_byte = (sc_raw[sb_global] & 0x0Fu)
                       | ((sc_raw[8u + sb_global] & 0x03u) << 4u);
        } else if (sb_global < 8u) {
            uint i = sb_global - 4u;
            scale_byte = (sc_raw[4u + i] & 0x0Fu)
                       | (((sc_raw[8u + i] >> 2u) & 0x03u) << 4u);
        } else if (sb_global < 12u) {
            uint i = sb_global - 8u;
            scale_byte = (sc_raw[i] >> 4u)
                       | (((sc_raw[8u + i] >> 4u) & 0x03u) << 4u);
        } else {
            uint i = sb_global - 12u;
            scale_byte = (sc_raw[4u + i] >> 4u)
                       | (((sc_raw[8u + i] >> 6u) & 0x03u) << 4u);
        }
        float dl = d_all * (float)((int)((char)scale_byte) - 32);

        // 8 weights at (l_in_sb_base + c) for c ∈ 0..8.
        threadgroup float* ws_row = Ws + (uint)row * BK_Q3K + (uint)col_chunk * 8u;
        #pragma clang loop unroll(full)
        for (ushort c = 0; c < 8u; ++c) {
            uint l = l_in_sb_base + (uint)c;
            uchar qs_byte    = qs[qs_byte_base + l];
            uchar hmask_byte = hmask[hmask_base + l];
            int q_lo  = (int)(((uint)qs_byte >> shift) & 0x03u);
            int h_bit = ((uint)hmask_byte & m_bit) != 0u;
            int v     = q_lo - (h_bit ? 0 : 4);
            ws_row[c] = dl * (float)v;
        }

        simdgroup_barrier(mem_flags::mem_threadgroup);

        #pragma clang loop unroll(full)
        for (ushort k_frag = 0; k_frag < 4u; ++k_frag) {
            simdgroup_load(A_frag, Xs + (uint)k_frag * 8u, BK_Q3K);
            simdgroup_load(
                B_frag,
                Ws + (uint)k_frag * 8u,
                BK_Q3K,
                ulong2(0, 0),
                /* transpose */ true);
            simdgroup_multiply_accumulate(C_frag, A_frag, B_frag, C_frag);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }

    device float* C_ptr = C
        + (uint64_t)m_tile * BM_Q3K * (uint64_t)N
        + (uint64_t)n_tile * BN_Q3K;
    simdgroup_store(C_frag, C_ptr, N);
}
"#;

/// T162 phase 7 — Q3_K SGEMM via simdgroup_matrix avec dequant inline.
///
/// `C = A @ W^T` où A est `[M, K]` f32 row-major, W est `[N, K]` Q3_K
/// (row-major en super-blocks de 256 weights × 110 bytes).
///
/// Pré-conditions : Metal3, M et N multiples de 8, K multiple de 256.
pub fn sgemm_q3_k_f32_simdgroup_matrix_into(
    backend: &MetalBackend,
    a_buf: &Buffer,
    w_q3k_buf: &Buffer,
    c_buf: &Buffer,
    m: usize,
    n: usize,
    k: usize,
) -> Result<(), MetalError> {
    if !backend.supports_metal3() {
        return Err(MetalError::Unsupported(
            "sgemm_q3_k_f32_simdgroup_matrix needs Metal3".to_string(),
        ));
    }
    if m == 0 || n == 0 || k == 0 || m % 8 != 0 || n % 8 != 0 || k % 256 != 0 {
        return Err(MetalError::ShapeMismatch(format!(
            "sgemm_q3_k_f32_simdgroup_matrix: M, N must be multiples of 8 and K multiple of 256 (got M={m}, N={n}, K={k})"
        )));
    }
    let pipeline = backend.pipeline(
        "sgemm_q3_k_f32_simdgroup_matrix",
        SGEMM_Q3_K_F32_SIMDGROUP_MATRIX_SHADER,
        "sgemm_q3_k_f32_simdgroup_matrix",
    )?;
    let dims = [m as u32, n as u32, k as u32];
    backend.with_encoder(|encoder| {
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(a_buf), 0);
        encoder.set_buffer(1, Some(w_q3k_buf), 0);
        encoder.set_buffer(2, Some(c_buf), 0);
        encoder.set_bytes(3, 12, dims.as_ptr() as *const std::ffi::c_void);
        let tg_size = MTLSize::new(32, 1, 1);
        let n_tg = ((m / 8) * (n / 8)) as u64;
        let groups = MTLSize::new(n_tg, 1, 1);
        encoder.dispatch_thread_groups(groups, tg_size);
    });
    Ok(())
}

// =============================================================================
// T162 phase 5-bis — Q6_K SGEMM tiles 64×64 multi-warp (4 simdgroups par TG).
//
// Évolution de phase 5 (tile 8×8) : passage à BM=BN=64 avec 4 simdgroups
// par TG (WM=WN=2). Combine le pattern multi-warp de phase 3-bis (Q4_K
// 64×64) avec le dequant Q6_K de phase 5.
//
// Cible spécifique : FFN down du Qwen3-14B, qui est en Q6_K avec K=17408
// (very large). Le tile 8×8 fait 544 K-iter sub-blocks → sous-utilise les
// compute units. Le tile 64×64 multi-warp pousse vers ×4-5 (vs ×2.5 phase 5).
//
// IMPORTANT : `#pragma clang loop unroll(full)` requis sur les boucles
// MMA (cf gotcha critique commit `75efa9a`). Sans unroll, les 16 MMAs
// par K-fragment sont sérialisées → -50% perf.
const SGEMM_Q6_K_F32_SIMDGROUP_MATRIX_64_SHADER: &str = r#"
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

constant uint Q6K_BYTES_64 = 210u;
constant uint Q6K_WEIGHTS_64 = 256u;
constant uint BM_Q6K_64 = 64u;
constant uint BN_Q6K_64 = 64u;
constant uint BK_Q6K_64 = 32u;
constant uint WM_Q6K_64 = 2u;
constant uint WN_Q6K_64 = 2u;
constant uint TM_Q6K_64 = 32u;        // BM / WM
constant uint TN_Q6K_64 = 32u;        // BN / WN
constant uint FM_Q6K_64 = 4u;         // TM / 8
constant uint FN_Q6K_64 = 4u;         // TN / 8

kernel void sgemm_q6_k_f32_simdgroup_matrix_64(
    device const float*  A      [[buffer(0)]],
    device const uchar*  W_q6k  [[buffer(1)]],
    device float*        C      [[buffer(2)]],
    constant uint3&      dims   [[buffer(3)]],
    uint                 tg_id  [[threadgroup_position_in_grid]],
    ushort               tiisg  [[thread_index_in_simdgroup]],
    ushort               sgitg  [[simdgroup_index_in_threadgroup]]
) {
    uint M = dims.x;
    uint N = dims.y;
    uint K = dims.z;

    uint n_tiles_n = N / BN_Q6K_64;
    uint m_tile = tg_id / n_tiles_n;
    uint n_tile = tg_id % n_tiles_n;
    if (m_tile * BM_Q6K_64 >= M || n_tile * BN_Q6K_64 >= N) return;

    uint sgi = (uint)sgitg / WN_Q6K_64;
    uint sgj = (uint)sgitg % WN_Q6K_64;

    // Q6_K reste en float (half precision accumule >5% rel error sur K=512).
    threadgroup float Xs[64 * 32];
    threadgroup float Ws[64 * 32];

    simdgroup_matrix<float, 8, 8> C_frag[4][4];
    for (uint i = 0; i < FM_Q6K_64; ++i) {
        for (uint j = 0; j < FN_Q6K_64; ++j) {
            C_frag[i][j] = simdgroup_matrix<float, 8, 8>(0.0);
        }
    }

    uint blocks_per_row = K / Q6K_WEIGHTS_64;
    uint row_stride_bytes = blocks_per_row * Q6K_BYTES_64;

    uint tid = (uint)sgitg * 32u + (uint)tiisg;   // 0..127
    uint load_row = tid / 2u;                      // 0..63
    uint load_chunk = tid % 2u;                    // 0..1 (cols 0..16 or 16..32)

    for (uint k_offset = 0; k_offset < K; k_offset += BK_Q6K_64) {
        // Phase 1 : load Xs[BM=64, BK=32] coopératif.
        uint a_row_global = m_tile * BM_Q6K_64 + load_row;
        threadgroup float* xs_dst = Xs + load_row * BK_Q6K_64 + load_chunk * 16u;
        if (a_row_global < M) {
            uint a_base = a_row_global * K + k_offset + load_chunk * 16u;
            for (uint c = 0; c < 16u; ++c) {
                xs_dst[c] = A[a_base + c];
            }
        } else {
            for (uint c = 0; c < 16u; ++c) {
                xs_dst[c] = 0.0;
            }
        }

        // Phase 2 : dequant Ws[BN=64, BK=32] coopératif (Q6_K).
        uint super_block_idx = k_offset / Q6K_WEIGHTS_64;
        uint sb_row = (k_offset % Q6K_WEIGHTS_64) / 32u;  // 0..7
        uint half_idx = sb_row / 4u;                       // 0 or 1
        uint row_in_half = sb_row % 4u;                    // 0..3

        uint w_row_global = n_tile * BN_Q6K_64 + load_row;
        threadgroup float* ws_dst = Ws + load_row * BK_Q6K_64 + load_chunk * 16u;
        if (w_row_global < N) {
            device const uchar* row_block = W_q6k
                + (uint64_t)w_row_global * row_stride_bytes
                + (uint64_t)super_block_idx * Q6K_BYTES_64;

            device const half* d_ptr = (device const half*)(row_block + 208);
            float d_all = float(*d_ptr);

            device const uchar* ql_h = row_block + half_idx * 64u;
            device const uchar* qh_h = row_block + 128u + half_idx * 32u;
            device const char*  sc_h = (device const char*)(row_block + 192u + half_idx * 8u);

            uint qh_shift = row_in_half * 2u;
            bool use_high_nibble = (row_in_half >= 2u);
            uint ql_extra_offset = ((row_in_half & 1u) == 1u) ? 32u : 0u;

            #pragma clang loop unroll(full)
            for (uint c = 0; c < 16u; ++c) {
                uint l = load_chunk * 16u + c;     // l ∈ 0..32
                uint scale_idx = 2u * row_in_half + l / 16u;
                int sc6 = (int)sc_h[scale_idx];
                float scale = d_all * (float)sc6;

                uchar qh_byte = qh_h[l];
                uint qh_bits = ((uint)qh_byte >> qh_shift) & 0x03u;

                uchar ql_byte = ql_h[l + ql_extra_offset];
                uint ql_nibble = use_high_nibble ? ((uint)ql_byte >> 4u) : ((uint)ql_byte & 0x0Fu);

                int q = (int)(ql_nibble | (qh_bits << 4u));
                ws_dst[c] = scale * (float)(q - 32);
            }
        } else {
            for (uint c = 0; c < 16u; ++c) {
                ws_dst[c] = 0.0;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Phase 3 : 4 K-fragments × 4×4 MMAs per simdgroup. UNROLL CRITIQUE.
        // T162 phase 9g : simdgroup_barrier hints (pattern llama.cpp).
        #pragma clang loop unroll(full)
        for (uint k_frag = 0; k_frag < BK_Q6K_64 / 8u; ++k_frag) {
            simdgroup_matrix<float, 8, 8> A_frags[4];
            simdgroup_matrix<float, 8, 8> B_frags[4];

            simdgroup_barrier(mem_flags::mem_none);
            #pragma clang loop unroll(full)
            for (uint i = 0; i < FM_Q6K_64; ++i) {
                uint a_row = sgi * TM_Q6K_64 + i * 8u;
                simdgroup_load(A_frags[i], Xs + a_row * BK_Q6K_64 + k_frag * 8u, BK_Q6K_64);
            }

            simdgroup_barrier(mem_flags::mem_none);
            #pragma clang loop unroll(full)
            for (uint j = 0; j < FN_Q6K_64; ++j) {
                uint w_row = sgj * TN_Q6K_64 + j * 8u;
                simdgroup_load(
                    B_frags[j],
                    Ws + w_row * BK_Q6K_64 + k_frag * 8u,
                    BK_Q6K_64,
                    ulong2(0, 0),
                    /* transpose */ true);
            }

            simdgroup_barrier(mem_flags::mem_none);
            #pragma clang loop unroll(full)
            for (uint i = 0; i < FM_Q6K_64; ++i) {
                #pragma clang loop unroll(full)
                for (uint j = 0; j < FN_Q6K_64; ++j) {
                    simdgroup_multiply_accumulate(
                        C_frag[i][j], A_frags[i], B_frags[j], C_frag[i][j]);
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Store : 16 fragments × 4 simdgroups = 64 fragments écrits.
    for (uint i = 0; i < FM_Q6K_64; ++i) {
        for (uint j = 0; j < FN_Q6K_64; ++j) {
            uint c_row = m_tile * BM_Q6K_64 + sgi * TM_Q6K_64 + i * 8u;
            uint c_col = n_tile * BN_Q6K_64 + sgj * TN_Q6K_64 + j * 8u;
            if (c_row + 7u < M && c_col + 7u < N) {
                device float* C_ptr = C + (uint64_t)c_row * N + (uint64_t)c_col;
                simdgroup_store(C_frag[i][j], C_ptr, N);
            }
        }
    }
}
"#;

/// T162 phase 5-bis — Q6_K SGEMM tiles 64×64 multi-warp.
///
/// Cible : FFN down Qwen3-14B (Q6_K, K=17408 — TRÈS large). Réduit le TG count
/// de 64× vs phase 5 (8×8) et débloque le throughput multi-warp Apple.
///
/// Pré-conditions : Metal3, M et N multiples de 64 ; K multiple de 256.
pub fn sgemm_q6_k_f32_simdgroup_matrix_64_into(
    backend: &MetalBackend,
    a_buf: &Buffer,
    w_q6k_buf: &Buffer,
    c_buf: &Buffer,
    m: usize,
    n: usize,
    k: usize,
) -> Result<(), MetalError> {
    if !backend.supports_metal3() {
        return Err(MetalError::Unsupported(
            "sgemm_q6_k_f32_simdgroup_matrix_64 needs Metal3".to_string(),
        ));
    }
    if m == 0 || n == 0 || k == 0 || m % 64 != 0 || n % 64 != 0 || k % 256 != 0 {
        return Err(MetalError::ShapeMismatch(format!(
            "sgemm_q6_k_f32_simdgroup_matrix_64: M, N must be multiples of 64 and K multiple of 256 (got M={m}, N={n}, K={k})"
        )));
    }
    let pipeline = backend.pipeline(
        "sgemm_q6_k_f32_simdgroup_matrix_64",
        SGEMM_Q6_K_F32_SIMDGROUP_MATRIX_64_SHADER,
        "sgemm_q6_k_f32_simdgroup_matrix_64",
    )?;
    let dims = [m as u32, n as u32, k as u32];
    backend.with_encoder(|encoder| {
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(a_buf), 0);
        encoder.set_buffer(1, Some(w_q6k_buf), 0);
        encoder.set_buffer(2, Some(c_buf), 0);
        encoder.set_bytes(3, 12, dims.as_ptr() as *const std::ffi::c_void);
        let tg_size = MTLSize::new(128, 1, 1);
        let n_tg = ((m / 64) * (n / 64)) as u64;
        let groups = MTLSize::new(n_tg, 1, 1);
        encoder.dispatch_thread_groups(groups, tg_size);
    });
    Ok(())
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

// T132 — Same body as T91/lcpp_nr2 but with NSG=2 simdgroups per threadgroup
// (matching llama.cpp's `N_SG_Q4_K = 2`). Two simdgroups inside one
// threadgroup let the GPU's threadgroup scheduler interleave compute and
// memory accesses between them, hiding global-memory latency that a single
// simdgroup can't mask alone. Each threadgroup processes NSG*NR0 = 4 output
// rows. Dispatch grid = N/4 threadgroups × 64 threads each.
//
// Critical subtle point: `first_row = (tgpig * NSG + sgitg) * NR0`. Each
// simdgroup inside the threadgroup picks the correct 2-row band via
// `sgitg`. The two bands are independent — no threadgroup memory or
// barriers needed (sumf[] is in registers, simd_sum reduces inside one
// simdgroup, output writes to disjoint rows).
//
// Note: tested NSG=4 (128 threads/tg, 8 rows/tg) — gave equivalent chained
// tok/s to NSG=2 on M4 Max while increasing register pressure marginally.
// NSG=2 stays the reference value (matches llama.cpp upstream).
const SGEMV_Q4_K_F32_LCPP_NSG2_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

constant uint BLOCK_BYTES = 144u;
constant uint BLOCK_WEIGHTS = 256u;
constant short NR0 = 2;
constant short NSG = 2;
constant ushort KMASK1 = 0x3f3f;
constant ushort KMASK2 = 0x0f0f;
constant ushort KMASK3 = 0xc0c0;

kernel void sgemv_q4_k_f32_lcpp_nsg2(
    device const float*  x      [[buffer(0)]],
    device const uchar*  w_q4k  [[buffer(1)]],
    device float*        y      [[buffer(2)]],
    constant uint2&      dims   [[buffer(3)]],
    uint                 tg_id  [[threadgroup_position_in_grid]],
    ushort               tiisg  [[thread_index_in_simdgroup]],
    ushort               sgitg  [[simdgroup_index_in_threadgroup]]
) {
    uint K = dims.x;
    uint N = dims.y;
    uint blocks_per_row = K / BLOCK_WEIGHTS;

    // Threadgroup tg_id covers (NSG*NR0)=4 rows; this simdgroup handles 2 of them.
    uint first_row = (tg_id * (uint)NSG + (uint)sgitg) * (uint)NR0;
    if (first_row >= N) return;

    short ix = (short)(tiisg / 8u);
    short it = (short)(tiisg % 8u);
    short iq = it / 4;
    short ir = it % 4;

    int nb = (int)blocks_per_row;

    device const float* y4 = x + ix * 256 + 64 * iq + 8 * ir;

    float yl[16];
    float yh[16];
    float sumf[2] = {0.0, 0.0};

    ushort sc16[4];
    thread const uchar* sc8 = (thread const uchar*)sc16;

    uint row_stride = blocks_per_row * BLOCK_BYTES;

    for (int ib = ix; ib < nb; ib += 4) {
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

            device const uchar* block = w_q4k + (uint64_t)nrow * row_stride + (uint)ib * BLOCK_BYTES;
            device const uint16_t* dh_ptr = (device const uint16_t*)(block);
            float d    = float(as_type<half>(dh_ptr[0]));
            float dmin = float(as_type<half>(dh_ptr[1]));

            device const uint16_t* sc = (device const uint16_t*)(block + 4) + iq;
            device const uint16_t* q1 = (device const uint16_t*)(block + 16) + 16 * iq + 4 * ir;
            device const uint16_t* q2 = q1 + 32;

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

// T157 — Q4_K sgemv RES-ADD variant : `y[i] += W @ x` au lieu de `y[i] = W @ x`.
// Permet de fusionner le `add_inplace(xd, fc2)` final du FFN dense dans
// la matmul `w_down`. Une passe au lieu de deux. Saves 1 dispatch par FFN
// dense layer × 40 layers = 40 dispatches/token sur Qwen3-14B et Qwen3.6-27B.
//
// La structure du kernel est identique à `sgemv_q4_k_f32_lcpp_nsg2`, seule
// la dernière instruction passe de `y[nrow] = sum_all` à `y[nrow] += sum_all`.
const SGEMV_Q4_K_F32_LCPP_NSG2_RESADD_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

constant uint BLOCK_BYTES = 144u;
constant uint BLOCK_WEIGHTS = 256u;
constant short NR0 = 2;
constant short NSG = 2;
constant ushort KMASK1 = 0x3f3f;
constant ushort KMASK2 = 0x0f0f;
constant ushort KMASK3 = 0xc0c0;

kernel void sgemv_q4_k_f32_lcpp_nsg2_resadd(
    device const float*  x      [[buffer(0)]],
    device const uchar*  w_q4k  [[buffer(1)]],
    device float*        y      [[buffer(2)]],
    constant uint2&      dims   [[buffer(3)]],
    uint                 tg_id  [[threadgroup_position_in_grid]],
    ushort               tiisg  [[thread_index_in_simdgroup]],
    ushort               sgitg  [[simdgroup_index_in_threadgroup]]
) {
    uint K = dims.x;
    uint N = dims.y;
    uint blocks_per_row = K / BLOCK_WEIGHTS;

    uint first_row = (tg_id * (uint)NSG + (uint)sgitg) * (uint)NR0;
    if (first_row >= N) return;

    short ix = (short)(tiisg / 8u);
    short it = (short)(tiisg % 8u);
    short iq = it / 4;
    short ir = it % 4;

    int nb = (int)blocks_per_row;
    device const float* y4 = x + ix * 256 + 64 * iq + 8 * ir;

    float yl[16];
    float yh[16];
    float sumf[2] = {0.0, 0.0};
    ushort sc16[4];
    thread const uchar* sc8 = (thread const uchar*)sc16;
    uint row_stride = blocks_per_row * BLOCK_BYTES;

    for (int ib = ix; ib < nb; ib += 4) {
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
            device const uchar* block = w_q4k + (uint64_t)nrow * row_stride + (uint)ib * BLOCK_BYTES;
            device const uint16_t* dh_ptr = (device const uint16_t*)(block);
            float d    = float(as_type<half>(dh_ptr[0]));
            float dmin = float(as_type<half>(dh_ptr[1]));

            device const uint16_t* sc = (device const uint16_t*)(block + 4) + iq;
            device const uint16_t* q1 = (device const uint16_t*)(block + 16) + 16 * iq + 4 * ir;
            device const uint16_t* q2 = q1 + 32;

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
            // T157 — RES-ADD : ajoute au résidu existant au lieu d'écraser.
            y[nrow] += sum_all;
        }
    }
}
"#;

/// T157 — Q4_K matmul-vec qui ADDITIONNE au buffer de sortie au lieu d'écraser.
///
/// **Note** : testé sur Qwen3-14B Q4_K_M, **régresse de -2 %** (decode 42.94
/// → 42.07 t/s). Cause : `y[nrow] += sum_all` requiert un load DRAM
/// supplémentaire vs le store-only `y[nrow] = sum_all`. Sur path DRAM-bound
/// (= 14B FFN dense à 80 % peak DRAM), le coût du load supplémentaire
/// l'emporte sur l'économie d'1 dispatch + add_inplace.
///
/// Conservé pour expérimentation future (peut éventuellement être utile
/// quand on est dispatch-bound, p.ex. si on combinait avec d'autres
/// optimisations qui réduisent dramatiquement la pression DRAM).
#[allow(dead_code)]
pub fn sgemv_q4_k_f32_lcpp_nsg2_resadd_into(
    backend: &MetalBackend,
    x_buf: &Buffer,
    w_q4k_buf: &Buffer,
    out_buf: &Buffer,
    k: usize,
    n: usize,
) -> Result<(), MetalError> {
    if !backend.supports_metal3() {
        return Err(MetalError::Unsupported(
            "sgemv_q4_k_f32_lcpp_nsg2_resadd needs Metal3".to_string(),
        ));
    }
    if k == 0 || n == 0 || k % 256 != 0 || n % 4 != 0 {
        return Err(MetalError::ShapeMismatch(format!(
            "sgemv_q4_k_resadd: K%256==0 && N%4==0 required (K={k}, N={n})"
        )));
    }
    let pipeline = backend.pipeline(
        "sgemv_q4_k_f32_lcpp_nsg2_resadd",
        SGEMV_Q4_K_F32_LCPP_NSG2_RESADD_SHADER,
        "sgemv_q4_k_f32_lcpp_nsg2_resadd",
    )?;
    let dims = [k as u32, n as u32];
    backend.with_encoder(|encoder| {
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(x_buf), 0);
        encoder.set_buffer(1, Some(w_q4k_buf), 0);
        encoder.set_buffer(2, Some(out_buf), 0);
        encoder.set_bytes(3, 8, dims.as_ptr() as *const std::ffi::c_void);
        let tg_size = MTLSize::new(64, 1, 1);
        let n_tg = (n as u64).div_ceil(4);
        let groups = MTLSize::new(n_tg, 1, 1);
        encoder.dispatch_thread_groups(groups, tg_size);
    });
    Ok(())
}

// T156 — Q4_K sgemv variant with NR0=4 (8 rows per threadgroup, 4 par
// simdgroup). Réutilise les yl/yh chargés (32 floats de x) sur 4 rows de
// poids au lieu de 2, doublant l'intensité ALU/byte. Sur shape FFN dense
// 14B (K=5120, N=14336) on attend +5-15 % vs NR0=2 si on est ALU-bound.
//
// Register pressure : sumf[4] f32 + 4× les loops dequant. M4 Max simdgroup
// a 32 lanes × 64-128 regs, OK pour ce surcoût. Si spill, perf se dégrade.
const SGEMV_Q4_K_F32_LCPP_NSG2_NR4_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

constant uint BLOCK_BYTES = 144u;
constant uint BLOCK_WEIGHTS = 256u;
constant short NR0 = 4;
constant short NSG = 2;
constant ushort KMASK1 = 0x3f3f;
constant ushort KMASK2 = 0x0f0f;
constant ushort KMASK3 = 0xc0c0;

kernel void sgemv_q4_k_f32_lcpp_nsg2_nr4(
    device const float*  x      [[buffer(0)]],
    device const uchar*  w_q4k  [[buffer(1)]],
    device float*        y      [[buffer(2)]],
    constant uint2&      dims   [[buffer(3)]],
    uint                 tg_id  [[threadgroup_position_in_grid]],
    ushort               tiisg  [[thread_index_in_simdgroup]],
    ushort               sgitg  [[simdgroup_index_in_threadgroup]]
) {
    uint K = dims.x;
    uint N = dims.y;
    uint blocks_per_row = K / BLOCK_WEIGHTS;

    uint first_row = (tg_id * (uint)NSG + (uint)sgitg) * (uint)NR0;
    if (first_row >= N) return;

    short ix = (short)(tiisg / 8u);
    short it = (short)(tiisg % 8u);
    short iq = it / 4;
    short ir = it % 4;

    int nb = (int)blocks_per_row;
    device const float* y4 = x + ix * 256 + 64 * iq + 8 * ir;

    float yl[16];
    float yh[16];
    float sumf[4] = {0.0, 0.0, 0.0, 0.0};

    ushort sc16[4];
    thread const uchar* sc8 = (thread const uchar*)sc16;
    uint row_stride = blocks_per_row * BLOCK_BYTES;

    for (int ib = ix; ib < nb; ib += 4) {
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
            device const uchar* block = w_q4k + (uint64_t)nrow * row_stride + (uint)ib * BLOCK_BYTES;
            device const uint16_t* dh_ptr = (device const uint16_t*)(block);
            float d    = float(as_type<half>(dh_ptr[0]));
            float dmin = float(as_type<half>(dh_ptr[1]));

            device const uint16_t* sc = (device const uint16_t*)(block + 4) + iq;
            device const uint16_t* q1 = (device const uint16_t*)(block + 16) + 16 * iq + 4 * ir;
            device const uint16_t* q2 = q1 + 32;

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

/// T156 — Q4_K sgemv NSG=2 NR0=4 (8 rows / threadgroup).
/// Auto-falls back to NR0=2 (`sgemv_q4_k_f32_lcpp_nsg2_into`) si N%8 != 0.
///
/// Note : testé sur Qwen3-14B Q4_K_M, **régresse de -2.3 %** (decode 41.97
/// → 40.99 t/s). La pression registre due à `sumf[4]` et 4× les loops de
/// dequant l'emporte sur le gain ALU/byte. Conservé pour expérimentation
/// future (peut-être utile pour des shapes plus larges en prefill batched).
#[allow(dead_code)]
pub fn sgemv_q4_k_f32_lcpp_nsg2_nr4_into(
    backend: &MetalBackend,
    x_buf: &Buffer,
    w_q4k_buf: &Buffer,
    out_buf: &Buffer,
    k: usize,
    n: usize,
) -> Result<(), MetalError> {
    if !backend.supports_metal3() {
        return Err(MetalError::Unsupported(
            "sgemv_q4_k_f32_lcpp_nsg2_nr4 needs Metal3".to_string(),
        ));
    }
    if k == 0 || n == 0 || k % 256 != 0 {
        return Err(MetalError::ShapeMismatch(format!(
            "sgemv_q4_k_f32_lcpp_nsg2_nr4: K%256==0 required (K={k}, N={n})"
        )));
    }
    if n % 8 != 0 {
        return sgemv_q4_k_f32_lcpp_nsg2_into(backend, x_buf, w_q4k_buf, out_buf, k, n);
    }
    let pipeline = backend.pipeline(
        "sgemv_q4_k_f32_lcpp_nsg2_nr4",
        SGEMV_Q4_K_F32_LCPP_NSG2_NR4_SHADER,
        "sgemv_q4_k_f32_lcpp_nsg2_nr4",
    )?;
    let dims = [k as u32, n as u32];
    backend.with_encoder(|encoder| {
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(x_buf), 0);
        encoder.set_buffer(1, Some(w_q4k_buf), 0);
        encoder.set_buffer(2, Some(out_buf), 0);
        encoder.set_bytes(3, 8, dims.as_ptr() as *const std::ffi::c_void);
        let tg_size = MTLSize::new(64, 1, 1);
        // 8 rows per threadgroup
        let n_tg = (n as u64).div_ceil(8);
        let groups = MTLSize::new(n_tg, 1, 1);
        encoder.dispatch_thread_groups(groups, tg_size);
    });
    Ok(())
}

/// T132 — lcpp_nr2 with NSG=2 (two simdgroups per threadgroup). Matches
/// llama.cpp's `N_SG_Q4_K = 2` dispatch, which yields better memory latency
/// hiding by interleaving compute & memory ops across the two simdgroups
/// in a single threadgroup. Each threadgroup processes NSG*NR0 = 4 rows.
///
/// Auto-falls back to NSG=1 (`sgemv_q4_k_f32_lcpp_nr2_into`) when N is not
/// divisible by 4, so callers can use this unconditionally.
pub fn sgemv_q4_k_f32_lcpp_nsg2_into(
    backend: &MetalBackend,
    x_buf: &Buffer,
    w_q4k_buf: &Buffer,
    out_buf: &Buffer,
    k: usize,
    n: usize,
) -> Result<(), MetalError> {
    if !backend.supports_metal3() {
        return Err(MetalError::Unsupported(
            "sgemv_q4_k_f32_lcpp_nsg2 needs Metal3".to_string(),
        ));
    }
    if k == 0 || n == 0 || k % 256 != 0 {
        return Err(MetalError::ShapeMismatch(format!(
            "sgemv_q4_k_f32_lcpp_nsg2: K%256==0 required (K={k}, N={n})"
        )));
    }
    // NSG=2 requires N divisible by 4 (each threadgroup writes 4 rows).
    // For non-multiple-of-4 N, fall back to NSG=1 lcpp_nr2.
    if n % 4 != 0 {
        return sgemv_q4_k_f32_lcpp_nr2_into(backend, x_buf, w_q4k_buf, out_buf, k, n);
    }
    let pipeline = backend.pipeline(
        "sgemv_q4_k_f32_lcpp_nsg2",
        SGEMV_Q4_K_F32_LCPP_NSG2_SHADER,
        "sgemv_q4_k_f32_lcpp_nsg2",
    )?;
    let dims = [k as u32, n as u32];
    backend.with_encoder(|encoder| {
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(x_buf), 0);
        encoder.set_buffer(1, Some(w_q4k_buf), 0);
        encoder.set_buffer(2, Some(out_buf), 0);
        encoder.set_bytes(3, 8, dims.as_ptr() as *const std::ffi::c_void);
        // 64 threads/threadgroup = 2 simdgroups × 32. Each threadgroup
        // processes NSG*NR0 = 4 rows. Total threadgroups = ceil(N/4).
        // Use `dispatch_thread_groups` (direct group count) instead of
        // `dispatch_threads` (which pays a CPU-side divmod-and-pad cost
        // every call) — matches llama.cpp's dispatch path.
        let tg_size = MTLSize::new(64, 1, 1);
        let n_tg = (n as u64).div_ceil(4);
        let groups = MTLSize::new(n_tg, 1, 1);
        encoder.dispatch_thread_groups(groups, tg_size);
    });
    Ok(())
}

// =============================================================================
// T162 phase 5 — qmv_fast Q4_K (port MLX pattern pour decode M=1).
//
// Pattern MLX (quantized.h::qmv_fast_impl) adapté au layout Q4_K (super-blocks
// 256w/144B + sub-blocks 32w + sc/m packés). Diffère de notre NSG=2 NR0=2 par :
//
// - NR0=4 (4 rows par simdgroup, 8 rows par TG) — vs NR0=2
// - Threads INDÉPENDANTS sur K (pas de cross-thread cooperation)
// - block_size=512 = 2 super-blocks par K-iter (vs 1 dans NSG=2)
// - values_per_thread=16 (vs 8 yl + 8 yh) → meilleur pipelining
// - simd_sum UNE FOIS au final (pas dans la boucle)
//
// Layout par thread (tiisg ∈ [0..32)) pour 1 K-iter de 512 weights :
//   super_in_iter = tiisg / 16   (0 ou 1)
//   sub_in_super  = (tiisg % 16) / 2  (0..7)
//   half_in_sub   = tiisg % 2  (0 ou 1)
//
// Thread couvre K positions [tiisg*16, tiisg*16+16) = 16 weights de demi-sub-block.
//
// Q4_K dequant per thread (16 weights de demi-sub) :
//   pair_idx = sub_in_super / 2 (0..3)
//   pair_qs_offset = pair_idx * 32 (bytes dans qs[128])
//   sub_is_high = sub_in_super % 2 (false=low nibble, true=high nibble)
//   l_start = half_in_sub * 16
//   For l ∈ [0..16) :
//     byte = qs[pair_qs_offset + l_start + l]
//     nibble = sub_is_high ? (byte >> 4) : (byte & 0x0F)
//     scale, min via packed sc/m (KMASK1/2/3)
//     weight = d × sc6 × nibble - dmin × m6
//
// Cible : 14B Q4_K_M decode 44 → 56-65 t/s = bat MLX 56 t/s.
// Status : SKELETON committed cette branche, IMPL Q4_K dequant + test parité
// dans la session suivante.
//
// Pré-conditions : Metal3, K % 512 == 0 (mult de 2 super-blocks), N % 8 == 0
// (8 rows par TG).
const SGEMV_Q4_K_F32_QMV_FAST_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

constant uint Q4K_BYTES = 144u;
constant uint Q4K_WEIGHTS = 256u;
constant uint NSG_QF = 2u;
constant uint NR0_QF = 4u;
constant uint VALUES_PER_THREAD = 16u;
constant uint BLOCK_SIZE = 512u;  // VALUES_PER_THREAD × SIMD_SIZE
constant ushort KMASK1 = 0x3f3f;
constant ushort KMASK2 = 0x0f0f;
constant ushort KMASK3 = 0xc0c0;

kernel void sgemv_q4_k_f32_qmv_fast(
    device const float*  x      [[buffer(0)]],   // [K]
    device const uchar*  w_q4k  [[buffer(1)]],   // [N, K] Q4_K
    device float*        y      [[buffer(2)]],   // [N]
    constant uint2&      dims   [[buffer(3)]],   // (K, N)
    uint                 tg_id  [[threadgroup_position_in_grid]],
    ushort               tiisg  [[thread_index_in_simdgroup]],
    ushort               sgitg  [[simdgroup_index_in_threadgroup]]
) {
    uint K = dims.x;
    uint N = dims.y;

    // 8 rows par TG (NSG=2 simdgroups × NR0=4 rows chaque).
    uint first_row = (tg_id * NSG_QF + (uint)sgitg) * NR0_QF;
    if (first_row >= N) return;

    uint blocks_per_row = K / Q4K_WEIGHTS;
    uint row_stride = blocks_per_row * Q4K_BYTES;

    // Per-thread layout (constant across K-loop).
    uint super_in_iter = (uint)tiisg / 16u;
    uint sub_in_super = ((uint)tiisg % 16u) / 2u;  // 0..7
    uint half_in_sub  = (uint)tiisg % 2u;
    uint pair_idx = sub_in_super / 2u;             // 0..3
    uint pair_qs_offset = pair_idx * 32u;
    bool sub_is_high = (sub_in_super % 2u) == 1u;
    uint l_start = half_in_sub * 16u;

    // Pre-compute scale/min unpacking helpers.
    // For sub_in_super sb, the (sc6, m6) bytes come from the 12-byte packed
    // scales section (block + 4) using KMASK1/2/3 logic.
    uint iq = sub_in_super / 2u;       // 0..3 (pair index for sc/m pack)
    bool sb_is_pair_high = (sub_in_super % 2u) == 1u;

    // Per-thread accumulators for 4 rows.
    float result[4] = {0.0, 0.0, 0.0, 0.0};

    // Load 16 x values for this thread's K-slice (constant across rows).
    // K_thread_offset = K_iter_start + tiisg * 16
    // Note : K_iter_start advances by BLOCK_SIZE per K-iter.
    // Each thread loads its own 16 contiguous x values.
    float x_thread[16];

    uint k_iters = K / BLOCK_SIZE;

    for (uint k_iter = 0; k_iter < k_iters; ++k_iter) {
        uint k_thread_offset = k_iter * BLOCK_SIZE + (uint)tiisg * 16u;

        // Load 16 x values for this thread.
        #pragma clang loop unroll(full)
        for (uint i = 0; i < 16u; ++i) {
            x_thread[i] = x[k_thread_offset + i];
        }

        // Process 4 rows : for each row, dequant 16 weights and accumulate.
        // The weights layout in Q4_K depends on which super-block this thread
        // is in (super_in_iter ∈ {0, 1}) and which sub-block.
        // T162 phase 5-bis : avoir un sum_x_thread INDÉPENDANT par row (au lieu
        // de hoister) donne 4 réductions indépendantes que le compilo Apple
        // entrelace mieux → +13% vs hoist version.
        uint super_block_global = k_iter * 2u + super_in_iter;

        #pragma clang loop unroll(full)
        for (uint row_off = 0; row_off < NR0_QF; ++row_off) {
            uint nrow = first_row + row_off;
            if (nrow >= N) continue;

            device const uchar* block = w_q4k
                + (uint64_t)nrow * row_stride
                + (uint64_t)super_block_global * Q4K_BYTES;
            device const uint16_t* dh_ptr = (device const uint16_t*)(block);
            float d    = float(as_type<half>(dh_ptr[0]));
            float dmin = float(as_type<half>(dh_ptr[1]));

            // Unpack (sc6, m6) pour sub_in_super via formule explicite Q4_K
            // (cf rustorch-gguf::dequant_q4_k). Pour sb < 4 : direct read.
            // Pour sb >= 4 : combiné des bytes packed.
            device const uchar* scales_bytes = block + 4;  // 12 bytes
            uint sb = sub_in_super;
            uchar sc6_byte, m6_byte;
            if (sb < 4u) {
                sc6_byte = scales_bytes[sb]      & 0x3Fu;
                m6_byte  = scales_bytes[sb + 4u] & 0x3Fu;
            } else {
                uint i = sb - 4u;
                sc6_byte = (scales_bytes[i + 8u] & 0x0Fu)
                         | ((scales_bytes[i]      >> 6u) << 4u);
                m6_byte  = (scales_bytes[i + 8u] >> 4u)
                         | ((scales_bytes[i + 4u] >> 6u) << 4u);
            }
            float scale   = d    * (float)sc6_byte;
            float min_val = dmin * (float)m6_byte;

            // Dequant 16 weights and dot product with x_thread[].
            device const uchar* qs = block + 16 + pair_qs_offset + l_start;
            float sum_w_x = 0.0;
            float sum_x_thread = 0.0;
            #pragma clang loop unroll(full)
            for (uint l = 0; l < 16u; ++l) {
                uchar byte = qs[l];
                uint nibble = sub_is_high ? ((uint)byte >> 4u) : ((uint)byte & 0x0Fu);
                float w_val = (float)nibble;
                sum_w_x      += w_val * x_thread[l];
                sum_x_thread += x_thread[l];
            }
            // Q4_K dequant final : weight = scale * nibble - min_val
            //   sum = sum_k(weight_k * x_k) = scale * sum(nibble * x) - min_val * sum(x)
            result[row_off] += scale * sum_w_x - min_val * sum_x_thread;
        }
    }

    // simd_sum across 32 threads, write 4 rows.
    #pragma clang loop unroll(full)
    for (uint row_off = 0; row_off < NR0_QF; ++row_off) {
        uint nrow = first_row + row_off;
        if (nrow >= N) continue;
        float row_sum = simd_sum(result[row_off]);
        if (tiisg == 0) {
            y[nrow] = row_sum;
        }
    }
}
"#;

/// T162 phase 5 — qmv_fast Q4_K pour decode M=1 (skeleton, à tester+intégrer).
///
/// 64 threads/TG = 2 simdgroups × 32. NR0=4 rows par simdgroup → 8 rows/TG.
/// Threads indépendants sur K, simd_sum final unique. Cible : bat MLX decode.
///
/// Pré-conditions : Metal3, K % 512 == 0, N % 8 == 0.
#[allow(dead_code)]
pub fn sgemv_q4_k_f32_qmv_fast_into(
    backend: &MetalBackend,
    x_buf: &Buffer,
    w_q4k_buf: &Buffer,
    out_buf: &Buffer,
    k: usize,
    n: usize,
) -> Result<(), MetalError> {
    if !backend.supports_metal3() {
        return Err(MetalError::Unsupported(
            "sgemv_q4_k_f32_qmv_fast needs Metal3".to_string(),
        ));
    }
    if k == 0 || n == 0 || k % 512 != 0 {
        return Err(MetalError::ShapeMismatch(format!(
            "sgemv_q4_k_f32_qmv_fast: K%512==0 required (K={k}, N={n})"
        )));
    }
    if n % 8 != 0 {
        // Fallback NSG=2 NR0=2 pour les rows non-alignées.
        return sgemv_q4_k_f32_lcpp_nsg2_into(backend, x_buf, w_q4k_buf, out_buf, k, n);
    }
    let pipeline = backend.pipeline(
        "sgemv_q4_k_f32_qmv_fast",
        SGEMV_Q4_K_F32_QMV_FAST_SHADER,
        "sgemv_q4_k_f32_qmv_fast",
    )?;
    let dims = [k as u32, n as u32];
    backend.with_encoder(|encoder| {
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(x_buf), 0);
        encoder.set_buffer(1, Some(w_q4k_buf), 0);
        encoder.set_buffer(2, Some(out_buf), 0);
        encoder.set_bytes(3, 8, dims.as_ptr() as *const std::ffi::c_void);
        // 64 threads/TG = 2 simdgroups × 32. Each TG processes 8 rows (NSG×NR0).
        let tg_size = MTLSize::new(64, 1, 1);
        let n_tg = (n as u64).div_ceil(8);
        let groups = MTLSize::new(n_tg, 1, 1);
        encoder.dispatch_thread_groups(groups, tg_size);
    });
    Ok(())
}

// T152 — Q4_K gather sgemv : sgemv « avec indirection sur expert » pour MoE.
//
// Une variante du kernel `sgemv_q4_k_f32_lcpp_nsg2` qui lit un index d'expert
// par token d'output et offset le pointeur de poids en conséquence. Permet
// d'effectuer en UN SEUL dispatch ce que le forward MoE faisait en
// `top_k * 1` dispatchs (un par expert sélectionné), en s'inspirant du
// pattern `gather_qmm_rhs` de MLX (`mlx/backend/metal/kernels/quantized.h`).
//
// Layouts supportés :
//   - x_stride_floats == 0  : input partagé, x lu comme [K] (gate_proj, up_proj
//     en decode où la même `h` est routée vers chaque expert top-k).
//   - x_stride_floats == K  : input par-row, x lu comme [B, K] (down_proj
//     où chaque expert a son propre hidden state silu(gate)*up).
//
// Dispatch :
//   threadgroup = 64 threads (NSG=2 × 32) ; grid = (N/4 threadgroups, B, 1).
//   Une threadgroup couvre 4 lignes de sortie pour UN token d'output (b).
//
// Buffers :
//   buf 0 : x       — `[K]` ou `[B, K]` f32 selon x_stride_floats
//   buf 1 : w_q4k   — `[E, N, K_q4k_bytes]` Q4_K stacked, contiguous
//   buf 2 : indices — `[B]` u32, expert id par token d'output
//   buf 3 : y       — `[B, N]` f32
//   buf 4 : dims    — uint4 = (K, N, B, expert_stride_bytes)
//   buf 5 : opts    — uint  = x_stride_floats (0 ou K)
const SGEMV_Q4_K_GATHER_F32_LCPP_NSG2_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

constant uint BLOCK_BYTES = 144u;
constant uint BLOCK_WEIGHTS = 256u;
constant short NR0 = 2;
constant short NSG = 2;
constant ushort KMASK1 = 0x3f3f;
constant ushort KMASK2 = 0x0f0f;
constant ushort KMASK3 = 0xc0c0;

kernel void sgemv_q4_k_gather_f32_lcpp_nsg2(
    device const float*  x         [[buffer(0)]],
    device const uchar*  w_q4k     [[buffer(1)]],
    device const uint*   indices   [[buffer(2)]],
    device float*        y         [[buffer(3)]],
    constant uint4&      dims      [[buffer(4)]],
    constant uint&       x_stride  [[buffer(5)]],
    uint2                tg_id     [[threadgroup_position_in_grid]],
    ushort               tiisg     [[thread_index_in_simdgroup]],
    ushort               sgitg     [[simdgroup_index_in_threadgroup]]
) {
    uint K = dims.x;
    uint N = dims.y;
    uint B = dims.z;
    uint expert_stride = dims.w;
    uint b = tg_id.y;
    if (b >= B) return;

    uint expert = indices[b];

    // Per-expert base of weights, x and y.
    device const uchar* w_base = w_q4k + (uint64_t)expert * (uint64_t)expert_stride;
    // x_stride==0 → broadcast (read x[k] for every b) ; ==K → per-row.
    device const float* x_base = x + (uint64_t)b * (uint64_t)x_stride;
    device       float* y_base = y + (uint64_t)b * (uint64_t)N;

    uint blocks_per_row = K / BLOCK_WEIGHTS;
    uint first_row = (tg_id.x * (uint)NSG + (uint)sgitg) * (uint)NR0;
    if (first_row >= N) return;

    short ix = (short)(tiisg / 8u);
    short it = (short)(tiisg % 8u);
    short iq = it / 4;
    short ir = it % 4;

    int nb = (int)blocks_per_row;

    device const float* y4 = x_base + ix * 256 + 64 * iq + 8 * ir;

    float yl[16];
    float yh[16];
    float sumf[2] = {0.0, 0.0};

    ushort sc16[4];
    thread const uchar* sc8 = (thread const uchar*)sc16;

    uint row_stride = blocks_per_row * BLOCK_BYTES;

    for (int ib = ix; ib < nb; ib += 4) {
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

            device const uchar* block = w_base + (uint64_t)nrow * row_stride + (uint)ib * BLOCK_BYTES;
            device const uint16_t* dh_ptr = (device const uint16_t*)(block);
            float d    = float(as_type<half>(dh_ptr[0]));
            float dmin = float(as_type<half>(dh_ptr[1]));

            device const uint16_t* sc = (device const uint16_t*)(block + 4) + iq;
            device const uint16_t* q1 = (device const uint16_t*)(block + 16) + 16 * iq + 4 * ir;
            device const uint16_t* q2 = q1 + 32;

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
            y_base[nrow] = sum_all;
        }
    }
}
"#;

/// T152 — Q4_K gather sgemv : `y[b, :] = W[indices[b], :, :] @ x[b * x_stride .. + K]`.
/// Source d'inspiration : `affine_gather_qmm_rhs` de MLX. Permet d'évaluer en
/// UN SEUL dispatch les `top_k` experts sélectionnés au lieu de `top_k`
/// dispatchs séparés. `expert_stride_bytes` = `N * (K/256) * 144` bytes
/// (= taille en Q4_K d'une matrice `[N, K]` complète).
///
/// `x_stride_floats == 0` : input partagé broadcast (gate_proj, up_proj).
/// `x_stride_floats == K` : input par-row (down_proj).
///
/// T152.1b — Indices désormais passés via `set_buffer` (Metal Buffer
/// device-side) au lieu de `set_bytes` (CPU). Ça permet à un kernel amont
/// (`topk_softmax_norm_f32`) d'écrire les indices directement sur GPU
/// sans drain CPU intermédiaire.
#[allow(clippy::too_many_arguments)]
pub fn sgemv_q4_k_gather_f32_lcpp_nsg2_into(
    backend: &MetalBackend,
    x_buf: &Buffer,
    w_q4k_stacked_buf: &Buffer,
    indices_buf: &Buffer,
    b: usize,
    out_buf: &Buffer,
    k: usize,
    n: usize,
    expert_stride_bytes: usize,
    x_stride_floats: usize,
) -> Result<(), MetalError> {
    if !backend.supports_metal3() {
        return Err(MetalError::Unsupported(
            "sgemv_q4_k_gather_f32_lcpp_nsg2 needs Metal3".to_string(),
        ));
    }
    if k == 0 || n == 0 || k % 256 != 0 || n % 4 != 0 {
        return Err(MetalError::ShapeMismatch(format!(
            "sgemv_q4_k_gather: K%256==0 && N%4==0 required (K={k}, N={n})"
        )));
    }
    if b == 0 {
        return Ok(());
    }
    if x_stride_floats != 0 && x_stride_floats != k {
        return Err(MetalError::ShapeMismatch(format!(
            "sgemv_q4_k_gather: x_stride_floats must be 0 (broadcast) or K={k}, got {x_stride_floats}"
        )));
    }
    let pipeline = backend.pipeline(
        "sgemv_q4_k_gather_f32_lcpp_nsg2",
        SGEMV_Q4_K_GATHER_F32_LCPP_NSG2_SHADER,
        "sgemv_q4_k_gather_f32_lcpp_nsg2",
    )?;
    let dims = [k as u32, n as u32, b as u32, expert_stride_bytes as u32];
    let x_stride_u32 = x_stride_floats as u32;
    backend.with_encoder(|encoder| {
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(x_buf), 0);
        encoder.set_buffer(1, Some(w_q4k_stacked_buf), 0);
        encoder.set_buffer(2, Some(indices_buf), 0);
        encoder.set_buffer(3, Some(out_buf), 0);
        encoder.set_bytes(4, 16, dims.as_ptr() as *const std::ffi::c_void);
        encoder.set_bytes(5, 4, &x_stride_u32 as *const u32 as *const std::ffi::c_void);
        let tg_size = MTLSize::new(64, 1, 1);
        let n_tg = (n as u64).div_ceil(4);
        let groups = MTLSize::new(n_tg, b as u64, 1);
        encoder.dispatch_thread_groups(groups, tg_size);
    });
    Ok(())
}

// T162 phase 9f-bis — Q4_K gather PER-TOKEN sgemv : version optimisée pour
// le batched MoE prefill. Au lieu de replicater h en x_repl [B*n_used, K]
// (= 128MB/layer pour 35B-A3B B=32), le kernel calcule directement le token
// index : `b_token = b_row / n_used` et lit x[b_token * K + k]. Économie : 0
// memcpy CPU + accès x cache-friendly (consécutifs n_used rows partagent x).
//
// Différence avec sgemv_q4_k_gather_f32_lcpp_nsg2 : ajout de `n_used` dans
// dims, `x_stride` n'est plus utilisé (toujours K virtuel via b_token).
const SGEMV_Q4_K_GATHER_PER_TOKEN_F32_LCPP_NSG2_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

constant uint BLOCK_BYTES_PT = 144u;
constant uint BLOCK_WEIGHTS_PT = 256u;
constant short NR0_PT = 2;
constant short NSG_PT = 2;
constant ushort KMASK1_PT = 0x3f3f;
constant ushort KMASK2_PT = 0x0f0f;
constant ushort KMASK3_PT = 0xc0c0;

kernel void sgemv_q4_k_gather_per_token_f32_lcpp_nsg2(
    device const float*  x         [[buffer(0)]],   // [B_tokens, K]
    device const uchar*  w_q4k     [[buffer(1)]],   // [E, N, K] Q4_K stacked
    device const uint*   indices   [[buffer(2)]],   // [B_eff = B_tokens * n_used]
    device float*        y         [[buffer(3)]],   // [B_eff, N]
    constant uint4&      dims      [[buffer(4)]],   // (K, N, B_eff, expert_stride_bytes)
    constant uint&       n_used    [[buffer(5)]],   // routing top-k size
    uint2                tg_id     [[threadgroup_position_in_grid]],
    ushort               tiisg     [[thread_index_in_simdgroup]],
    ushort               sgitg     [[simdgroup_index_in_threadgroup]]
) {
    uint K = dims.x;
    uint N = dims.y;
    uint B_eff = dims.z;
    uint expert_stride = dims.w;
    uint b = tg_id.y;
    if (b >= B_eff) return;

    uint expert = indices[b];
    uint b_token = b / n_used;     // intra-kernel "replication" sans memcpy

    device const uchar* w_base = w_q4k + (uint64_t)expert * (uint64_t)expert_stride;
    device const float* x_base = x + (uint64_t)b_token * (uint64_t)K;
    device       float* y_base = y + (uint64_t)b * (uint64_t)N;

    uint blocks_per_row = K / BLOCK_WEIGHTS_PT;
    uint first_row = (tg_id.x * (uint)NSG_PT + (uint)sgitg) * (uint)NR0_PT;
    if (first_row >= N) return;

    short ix = (short)(tiisg / 8u);
    short it = (short)(tiisg % 8u);
    short iq = it / 4;
    short ir = it % 4;

    int nb = (int)blocks_per_row;
    device const float* y4 = x_base + ix * 256 + 64 * iq + 8 * ir;

    float yl[16];
    float yh[16];
    float sumf[2] = {0.0, 0.0};

    ushort sc16[4];
    thread const uchar* sc8 = (thread const uchar*)sc16;

    uint row_stride = blocks_per_row * BLOCK_BYTES_PT;

    for (int ib = ix; ib < nb; ib += 4) {
        float4 sumy = {0.0, 0.0, 0.0, 0.0};
        for (short i = 0; i < 8; ++i) {
            yl[i + 0] = y4[i + 0];   sumy[0] += yl[i + 0];
            yl[i + 8] = y4[i + 32];  sumy[1] += yl[i + 8];
            yh[i + 0] = y4[i + 128]; sumy[2] += yh[i + 0];
            yh[i + 8] = y4[i + 160]; sumy[3] += yh[i + 8];
        }

        for (short row = 0; row < NR0_PT; ++row) {
            uint nrow = first_row + (uint)row;
            if (nrow >= N) continue;

            device const uchar* block = w_base + (uint64_t)nrow * row_stride + (uint)ib * BLOCK_BYTES_PT;
            device const uint16_t* dh_ptr = (device const uint16_t*)(block);
            float d    = float(as_type<half>(dh_ptr[0]));
            float dmin = float(as_type<half>(dh_ptr[1]));

            device const uint16_t* sc = (device const uint16_t*)(block + 4) + iq;
            device const uint16_t* q1 = (device const uint16_t*)(block + 16) + 16 * iq + 4 * ir;
            device const uint16_t* q2 = q1 + 32;

            sc16[0] = sc[0] & KMASK1_PT;
            sc16[1] = sc[2] & KMASK1_PT;
            sc16[2] = ((sc[4] >> 0) & KMASK2_PT) | ((sc[0] & KMASK3_PT) >> 2);
            sc16[3] = ((sc[4] >> 4) & KMASK2_PT) | ((sc[2] & KMASK3_PT) >> 2);

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

        y4 += 4 * (int)BLOCK_WEIGHTS_PT;
    }

    for (short row = 0; row < NR0_PT; ++row) {
        uint nrow = first_row + (uint)row;
        if (nrow >= N) break;
        float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) {
            y_base[nrow] = sum_all;
        }
    }
}
"#;

/// T162 phase 9f-bis — Q4_K gather sgemv per-token (sans memcpy h_repl).
/// Pour B_eff = B_tokens × n_used : `y[b, :] = W[indices[b], :, :] @ x[b/n_used, :]`.
///
/// Économie vs `sgemv_q4_k_gather_f32_lcpp_nsg2_into` avec x_repl :
/// - Pas de memcpy CPU 128MB/layer (B_tokens=32 × n_used=8 × d=2048 × 4)
/// - Accès x cache-friendly (n_used rows consécutifs partagent x[b_token])
#[allow(clippy::too_many_arguments)]
pub fn sgemv_q4_k_gather_per_token_f32_lcpp_nsg2_into(
    backend: &MetalBackend,
    x_buf: &Buffer,
    w_q4k_stacked_buf: &Buffer,
    indices_buf: &Buffer,
    b_eff: usize,
    n_used: usize,
    out_buf: &Buffer,
    k: usize,
    n: usize,
    expert_stride_bytes: usize,
) -> Result<(), MetalError> {
    if !backend.supports_metal3() {
        return Err(MetalError::Unsupported(
            "sgemv_q4_k_gather_per_token needs Metal3".to_string(),
        ));
    }
    if k == 0 || n == 0 || k % 256 != 0 || n % 4 != 0 {
        return Err(MetalError::ShapeMismatch(format!(
            "sgemv_q4_k_gather_per_token: K%256==0 && N%4==0 required (K={k}, N={n})"
        )));
    }
    if b_eff == 0 || n_used == 0 {
        return Ok(());
    }
    if b_eff % n_used != 0 {
        return Err(MetalError::ShapeMismatch(format!(
            "sgemv_q4_k_gather_per_token: b_eff={b_eff} must be multiple of n_used={n_used}"
        )));
    }
    let pipeline = backend.pipeline(
        "sgemv_q4_k_gather_per_token_f32_lcpp_nsg2",
        SGEMV_Q4_K_GATHER_PER_TOKEN_F32_LCPP_NSG2_SHADER,
        "sgemv_q4_k_gather_per_token_f32_lcpp_nsg2",
    )?;
    let dims = [k as u32, n as u32, b_eff as u32, expert_stride_bytes as u32];
    let n_used_u32 = n_used as u32;
    backend.with_encoder(|encoder| {
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(x_buf), 0);
        encoder.set_buffer(1, Some(w_q4k_stacked_buf), 0);
        encoder.set_buffer(2, Some(indices_buf), 0);
        encoder.set_buffer(3, Some(out_buf), 0);
        encoder.set_bytes(4, 16, dims.as_ptr() as *const std::ffi::c_void);
        encoder.set_bytes(5, 4, &n_used_u32 as *const u32 as *const std::ffi::c_void);
        let tg_size = MTLSize::new(64, 1, 1);
        let n_tg = (n as u64).div_ceil(4);
        let groups = MTLSize::new(n_tg, b_eff as u64, 1);
        encoder.dispatch_thread_groups(groups, tg_size);
    });
    Ok(())
}

// T152 — Q5_K gather sgemv : variante gather du `sgemv_q5_k_f32_lcpp_nsg2`.
// Utilisé pour `down_exps` du Qwen3.6-35B-A3B (Q5_K, alors que gate/up
// sont en Q4_K). Même logique que la variante Q4_K : indirection sur expert
// id + broadcast/per-row sur l'input.
const SGEMV_Q5_K_GATHER_F32_LCPP_NSG2_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

constant uint Q5K_BYTES = 176u;
constant uint Q5K_WEIGHTS = 256u;
constant short NR0_Q5K = 1;
constant short NSG_Q5K = 2;
constant ushort KMASK1 = 0x3f3f;
constant ushort KMASK2 = 0x0f0f;
constant ushort KMASK3 = 0xc0c0;

kernel void sgemv_q5_k_gather_f32_lcpp_nsg2(
    device const float*  x         [[buffer(0)]],
    device const uchar*  w_q5k     [[buffer(1)]],
    device const uint*   indices   [[buffer(2)]],
    device float*        y         [[buffer(3)]],
    constant uint4&      dims      [[buffer(4)]],   // (K, N, B, expert_stride_bytes)
    constant uint&       x_stride  [[buffer(5)]],
    uint2                tg_id     [[threadgroup_position_in_grid]],
    ushort               tiisg     [[thread_index_in_simdgroup]],
    ushort               sgitg     [[simdgroup_index_in_threadgroup]]
) {
    uint K = dims.x;
    uint N = dims.y;
    uint B = dims.z;
    uint expert_stride = dims.w;
    uint b = tg_id.y;
    if (b >= B) return;

    uint expert = indices[b];
    device const uchar* w_base = w_q5k + (uint64_t)expert * (uint64_t)expert_stride;
    device const float* x_base = x + (uint64_t)b * (uint64_t)x_stride;
    device       float* y_base = y + (uint64_t)b * (uint64_t)N;

    uint blocks_per_row = K / Q5K_WEIGHTS;
    uint first_row = (tg_id.x * (uint)NSG_Q5K + (uint)sgitg) * (uint)NR0_Q5K;
    if (first_row >= N) return;

    short tid = (short)(tiisg / 4u);
    short ix  = (short)(tiisg % 4u);
    short iq  = tid / 4;
    short ir  = tid % 4;

    short l0 = 8 * ir;
    short q_offset = 32 * iq + l0;
    short y_offset = 64 * iq + l0;

    uchar hm1 = 1u << (2*iq);
    uchar hm2 = hm1 << 1;
    uchar hm3 = hm1 << 4;
    uchar hm4 = hm2 << 4;

    int nb = (int)blocks_per_row;
    uint row_stride = blocks_per_row * Q5K_BYTES;

    float sumf = 0.0;
    float yl[16];
    float yh[16];
    ushort sc16[4];
    thread const uchar* sc8 = (thread const uchar*)sc16;

    device const float* y1 = x_base + (uint)ix * Q5K_WEIGHTS + (uint)y_offset;

    for (int i = ix; i < nb; i += 4) {
        device const float* y2 = y1 + 128;
        float4 sumy = {0.0, 0.0, 0.0, 0.0};
        for (short l = 0; l < 8; ++l) {
            yl[l + 0] = y1[l + 0];   sumy[0] += yl[l + 0];
            yl[l + 8] = y1[l + 32];  sumy[1] += yl[l + 8];
            yh[l + 0] = y2[l + 0];   sumy[2] += yh[l + 0];
            yh[l + 8] = y2[l + 32];  sumy[3] += yh[l + 8];
        }

        device const uchar* block = w_base + (uint64_t)first_row * row_stride + (uint)i * Q5K_BYTES;
        device const uint16_t* dh_ptr = (device const uint16_t*)(block);
        float d    = float(as_type<half>(dh_ptr[0]));
        float dmin = float(as_type<half>(dh_ptr[1]));

        device const uint16_t* a = (device const uint16_t*)(block + 4) + iq;
        sc16[0] = a[0] & KMASK1;
        sc16[1] = a[2] & KMASK1;
        sc16[2] = ((a[4] >> 0) & KMASK2) | ((a[0] & KMASK3) >> 2);
        sc16[3] = ((a[4] >> 4) & KMASK2) | ((a[2] & KMASK3) >> 2);

        device const uchar* qh = (device const uchar*)(block + 16) + (uint)l0;
        device const uchar* q1 = (device const uchar*)(block + 48) + (uint)q_offset;
        device const uchar* q2 = q1 + 64;

        float4 acc1 = {0.0, 0.0, 0.0, 0.0};
        float4 acc2 = {0.0, 0.0, 0.0, 0.0};
        for (short l = 0; l < 8; ++l) {
            uchar h = qh[l];
            acc1[0] += yl[l + 0] * (float)(q1[l] & 0x0F);
            acc1[1] += yl[l + 8] * (float)(q1[l] & 0xF0);
            acc1[2] += yh[l + 0] * (float)(q2[l] & 0x0F);
            acc1[3] += yh[l + 8] * (float)(q2[l] & 0xF0);
            acc2[0] += (h & hm1) ? yl[l + 0] : 0.0;
            acc2[1] += (h & hm2) ? yl[l + 8] : 0.0;
            acc2[2] += (h & hm3) ? yh[l + 0] : 0.0;
            acc2[3] += (h & hm4) ? yh[l + 8] : 0.0;
        }

        sumf += d * (
            float(sc8[0]) * (acc1[0]        + 16.0 * acc2[0]) +
            float(sc8[1]) * (acc1[1]/16.0   + 16.0 * acc2[1]) +
            float(sc8[4]) * (acc1[2]        + 16.0 * acc2[2]) +
            float(sc8[5]) * (acc1[3]/16.0   + 16.0 * acc2[3])
        ) - dmin * (
            sumy[0] * float(sc8[2]) +
            sumy[1] * float(sc8[3]) +
            sumy[2] * float(sc8[6]) +
            sumy[3] * float(sc8[7])
        );

        y1 += 4 * (int)Q5K_WEIGHTS;
    }

    if (first_row < N) {
        float row_sum = simd_sum(sumf);
        if (tiisg == 0) {
            y_base[first_row] = row_sum;
        }
    }
}
"#;

/// T152 — Q5_K gather sgemv. Mêmes paramètres et sémantique que
/// `sgemv_q4_k_gather_f32_lcpp_nsg2_into`, juste un format de quantization
/// différent. Utilisé pour `down_exps` du Qwen3.6-35B-A3B.
#[allow(clippy::too_many_arguments)]
pub fn sgemv_q5_k_gather_f32_lcpp_nsg2_into(
    backend: &MetalBackend,
    x_buf: &Buffer,
    w_q5k_stacked_buf: &Buffer,
    indices_buf: &Buffer,
    b: usize,
    out_buf: &Buffer,
    k: usize,
    n: usize,
    expert_stride_bytes: usize,
    x_stride_floats: usize,
) -> Result<(), MetalError> {
    if !backend.supports_metal3() {
        return Err(MetalError::Unsupported(
            "sgemv_q5_k_gather_f32_lcpp_nsg2 needs Metal3".to_string(),
        ));
    }
    if k == 0 || n == 0 || k % 256 != 0 {
        return Err(MetalError::ShapeMismatch(format!(
            "sgemv_q5_k_gather: K%256==0 required (K={k}, N={n})"
        )));
    }
    if b == 0 {
        return Ok(());
    }
    if x_stride_floats != 0 && x_stride_floats != k {
        return Err(MetalError::ShapeMismatch(format!(
            "sgemv_q5_k_gather: x_stride_floats must be 0 or K={k}, got {x_stride_floats}"
        )));
    }
    let pipeline = backend.pipeline(
        "sgemv_q5_k_gather_f32_lcpp_nsg2",
        SGEMV_Q5_K_GATHER_F32_LCPP_NSG2_SHADER,
        "sgemv_q5_k_gather_f32_lcpp_nsg2",
    )?;
    let dims = [k as u32, n as u32, b as u32, expert_stride_bytes as u32];
    let x_stride_u32 = x_stride_floats as u32;
    backend.with_encoder(|encoder| {
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(x_buf), 0);
        encoder.set_buffer(1, Some(w_q5k_stacked_buf), 0);
        encoder.set_buffer(2, Some(indices_buf), 0);
        encoder.set_buffer(3, Some(out_buf), 0);
        encoder.set_bytes(4, 16, dims.as_ptr() as *const std::ffi::c_void);
        encoder.set_bytes(5, 4, &x_stride_u32 as *const u32 as *const std::ffi::c_void);
        let tg_size = MTLSize::new(64, 1, 1);
        let n_tg = (n as u64).div_ceil(2);
        let groups = MTLSize::new(n_tg, b as u64, 1);
        encoder.dispatch_thread_groups(groups, tg_size);
    });
    Ok(())
}

// T158 — Q3_K matmul-vec kernel (3.4375 bpw, -24 % DRAM vs Q4_K).
//
// Format Q3_K (110 bytes / super-block de 256 weights) :
//   - hmask[32]  : high bit (bit 2) de chaque weight, packed 8 weights/byte
//   - qs[64]     : low 2 bits par weight, packed 4 weights/byte
//   - scales[12] : 16 sub-block scales 6-bit signed, packing complexe
//   - d (f16)    : super-block scale
//
// Per-weight : `w = d × (sc[sb] - 32) × (q_low2 - (h_bit ? 0 : 4))`
// où sc[sb] est le scale signed 6-bit en [-32, 31].
//
// Cette première version vise la simplicité (1 simdgroup par row, chaque
// thread couvre 8 weights consécutifs par super-block, simd_sum à la fin).
// Optim future : pattern NSG=2 NR0=2 chunké comme Q4_K si bench < 80 %
// peak DRAM.
//
// Validé numériquement par CPU `dequant_q3_k` (rustorch-gguf) + tests Q4_K
// kernel infra.
const SGEMV_Q3_K_F32_LCPP_NSG1_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

constant uint Q3K_BYTES = 110u;
constant uint Q3K_WEIGHTS = 256u;

kernel void sgemv_q3_k_f32_lcpp_nsg1(
    device const float*  x      [[buffer(0)]],   // [K]
    device const uchar*  w_q3k  [[buffer(1)]],   // [N * blocks_per_row * 110]
    device float*        y      [[buffer(2)]],   // [N]
    constant uint2&      dims   [[buffer(3)]],   // (K, N)
    uint                 tg_id  [[threadgroup_position_in_grid]],
    ushort               tiisg  [[thread_index_in_simdgroup]]
) {
    uint K = dims.x;
    uint N = dims.y;
    int  nb = (int)(K / Q3K_WEIGHTS);

    // Une row par threadgroup (NSG=1 NR0=1).
    uint nrow = tg_id;
    if (nrow >= N) return;

    uint row_stride = (uint)nb * Q3K_BYTES;

    // Layout per-thread : tiisg ∈ [0, 32), couvre 8 weights consécutifs
    // par super-block aux positions [tiisg*8, tiisg*8 + 8).
    uint sb_global    = tiisg / 2u;          // 0..15 (sub-block global)
    uint l_offset     = (tiisg % 2u) * 8u;   // 0 ou 8 (offset dans sub-block)
    uint half_idx     = sb_global / 8u;       // 0 ou 1
    uint sb_in_half   = sb_global % 8u;       // 0..7
    uint shift        = (sb_in_half / 2u) * 2u;             // 0,2,4,6
    uint qs_byte_base = half_idx * 32u + (sb_in_half % 2u) * 16u;
    uint hmask_base   = (sb_in_half % 2u) * 16u;
    uint j_global     = half_idx * 4u + (sb_in_half / 2u);
    uint m_bit        = 1u << j_global;

    float sumf = 0.0;

    for (int ib = 0; ib < nb; ++ib) {
        device const uchar* block = w_q3k + (uint64_t)nrow * row_stride + (uint)ib * Q3K_BYTES;
        device const uchar* hmask = block;
        device const uchar* qs    = block + 32;
        device const uchar* sc_raw = block + 96;
        device const half*  d_ptr = (device const half*)(block + 108);
        float d_all = float(*d_ptr);

        // Compute le scale 6-bit signed pour ce sub-block (cf rustorch-gguf
        // dequant_q3_k pour le packing détaillé).
        uchar scale_byte;
        if (sb_global < 4u) {
            scale_byte = (sc_raw[sb_global] & 0x0Fu)
                       | ((sc_raw[8u + sb_global] & 0x03u) << 4u);
        } else if (sb_global < 8u) {
            uint i = sb_global - 4u;
            scale_byte = (sc_raw[4u + i] & 0x0Fu)
                       | (((sc_raw[8u + i] >> 2u) & 0x03u) << 4u);
        } else if (sb_global < 12u) {
            uint i = sb_global - 8u;
            scale_byte = (sc_raw[i] >> 4u)
                       | (((sc_raw[8u + i] >> 4u) & 0x03u) << 4u);
        } else {
            uint i = sb_global - 12u;
            scale_byte = (sc_raw[4u + i] >> 4u)
                       | (((sc_raw[8u + i] >> 6u) & 0x03u) << 4u);
        }
        // Sign-extend 8-bit. scale_byte ∈ [0, 63] représente signed [0..63]
        // post-cast i8 (high bit toujours 0 sur 6-bit, donc pas de besoin
        // particulier de sign-extend explicite).
        float dl = d_all * (float)((int)((char)scale_byte) - 32);

        // Process les 8 weights de ce thread.
        uint x_base = (uint)ib * Q3K_WEIGHTS + tiisg * 8u;
        for (uint k = 0; k < 8u; ++k) {
            uint l = l_offset + k;
            uint qs_byte    = qs[qs_byte_base + l];
            uint hmask_byte = hmask[hmask_base + l];
            int  q_lo  = (int)((qs_byte >> shift) & 0x03u);
            int  h_bit = (hmask_byte & m_bit) != 0u;
            int  v     = q_lo - (h_bit ? 0 : 4);
            sumf += x[x_base + k] * dl * (float)v;
        }
    }

    float row_sum = simd_sum(sumf);
    if (tiisg == 0) {
        y[nrow] = row_sum;
    }
}
"#;

/// T158 phase 1b — Q3_K sgemv Metal `y = W @ x` où W est `[N, K]` Q3_K.
/// Une row par threadgroup, 32 threads (1 simdgroup) par row, simd_sum.
/// Pré-conditions : `K % 256 == 0` (alignement super-block).
pub fn sgemv_q3_k_f32_lcpp_nsg1_into(
    backend: &MetalBackend,
    x_buf: &Buffer,
    w_q3k_buf: &Buffer,
    out_buf: &Buffer,
    k: usize,
    n: usize,
) -> Result<(), MetalError> {
    if !backend.supports_metal3() {
        return Err(MetalError::Unsupported(
            "sgemv_q3_k_f32_lcpp_nsg1 needs Metal3".to_string(),
        ));
    }
    if k == 0 || n == 0 || k % 256 != 0 {
        return Err(MetalError::ShapeMismatch(format!(
            "sgemv_q3_k_f32_lcpp_nsg1: K%256==0 required (K={k}, N={n})"
        )));
    }
    let pipeline = backend.pipeline(
        "sgemv_q3_k_f32_lcpp_nsg1",
        SGEMV_Q3_K_F32_LCPP_NSG1_SHADER,
        "sgemv_q3_k_f32_lcpp_nsg1",
    )?;
    let dims = [k as u32, n as u32];
    backend.with_encoder(|encoder| {
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(x_buf), 0);
        encoder.set_buffer(1, Some(w_q3k_buf), 0);
        encoder.set_buffer(2, Some(out_buf), 0);
        encoder.set_bytes(3, 8, dims.as_ptr() as *const std::ffi::c_void);
        let tg_size = MTLSize::new(32, 1, 1);
        let groups = MTLSize::new(n as u64, 1, 1);
        encoder.dispatch_thread_groups(groups, tg_size);
    });
    Ok(())
}

// T160 — Q3_K sgemv NSG=2 NR0=1 fast-path avec ix-stripping (TurboQuant Q3_K).
//
// Pattern Q5_K-style : 64 threads par TG = 2 simdgroups (NSG=2), chaque
// simdgroup possède 1 row (NR0=1). Au sein d'un simdgroup, les 32 threads
// sont split en `tid = tiisg/4` (0..7, sélectionne une PAIRE de sub-blocks
// consécutifs) × `ix = tiisg%4` (0..3, K-axis stripe : ¼ des super-blocks).
// Chaque thread couvre 32 weights (= 2 sub-blocks de 16) par super-block
// visité, sur nb/4 super-blocks → même travail total que NSG=1 (8×nb), mais
// inner loops 4× plus denses → meilleure ILP.
//
// Préserve l'unpacking Q3_K validé en T158 (hmask + qs + scales 6-bit signed).
// Optim : `shift`, `half_idx`, `m_bit` SHARED entre sb_first et sb_second
// (pairs (0,1), (2,3), ..., (14,15) ont sb_in_half pair pour sb_first donc
// shift identique pour la pair, m_bit identique).
//
// Gains attendus vs NSG=1 :
//   * Halved TG count (NSG=2) → -50 % dispatch overhead
//   * Inner loops 4× plus denses (yl[16]+yh[16] register-resident, 16 iter
//     consécutives par sub-block) → meilleur instruction scheduling Apple GPU
//   * Outer loop iterates ¼ × → fewer block pointer arithmetic
const SGEMV_Q3_K_F32_LCPP_NSG2_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

constant uint Q3K_BYTES = 110u;
constant uint Q3K_WEIGHTS = 256u;
constant short NSG_Q3K = 2;

kernel void sgemv_q3_k_f32_lcpp_nsg2(
    device const float*  x      [[buffer(0)]],   // [K]
    device const uchar*  w_q3k  [[buffer(1)]],   // [N * blocks_per_row * 110]
    device float*        y      [[buffer(2)]],   // [N]
    constant uint2&      dims   [[buffer(3)]],   // (K, N)
    uint                 tg_id  [[threadgroup_position_in_grid]],
    ushort               tiisg  [[thread_index_in_simdgroup]],
    ushort               sgitg  [[simdgroup_index_in_threadgroup]]
) {
    uint K = dims.x;
    uint N = dims.y;
    int  nb = (int)(K / Q3K_WEIGHTS);

    // 2 simdgroups par TG, chaque simdgroup possède 1 row.
    uint nrow = (uint)tg_id * (uint)NSG_Q3K + (uint)sgitg;
    if (nrow >= N) return;

    uint row_stride = (uint)nb * Q3K_BYTES;

    // Per-thread layout : ix-stripping.
    short tid = (short)(tiisg / 4u);   // 0..7 — pair of consecutive sub-blocks
    short ix  = (short)(tiisg % 4u);   // 0..3 — K-axis stripe

    // Pair de sub-blocks : sb_first = tid*2 (pair), sb_second = tid*2+1.
    // (0,1), (2,3), ..., (14,15) — pairs au sein de la même half (0..7 ou 8..15).
    short sb_first  = tid * 2;
    short sb_second = sb_first + 1;

    // Métadonnées de packing — pre-computed (constants across ib).
    // sb_first et sb_second partagent half_idx, shift, m_bit (cf analyse T160).
    short half_idx        = sb_first / 8;             // 0 si tid<4, 1 sinon
    short sb_in_half_f    = sb_first % 8;             // {0,2,4,6} ou {0,2,4,6}
    short shift           = (sb_in_half_f / 2) * 2;   // 0,2,4,6
    short qs_base_first   = half_idx * 32 + 0;        // sb_first→sb_in_half%2==0
    short qs_base_second  = half_idx * 32 + 16;       // sb_second→sb_in_half%2==1
    short hmask_base_first  = 0;                      // (sb_in_half%2==0)*16
    short hmask_base_second = 16;                     // (sb_in_half%2==1)*16
    short j_global        = half_idx * 4 + sb_in_half_f / 2;
    uint  m_bit           = 1u << j_global;

    // Position dans x : ce thread couvre [sb_first*16 .. sb_first*16+32) du
    // super-block courant, sur les ¼ de super-blocks indexés par ix.
    device const float* y_first = x + (uint)ix * Q3K_WEIGHTS + (uint)sb_first * 16u;

    float sumf = 0.0;

    for (int i = ix; i < nb; i += 4) {
        // Load 32 floats register-resident pour 1 super-block visité.
        float yl[16];
        float yh[16];
        for (short l = 0; l < 16; ++l) {
            yl[l] = y_first[l];
            yh[l] = y_first[l + 16];
        }

        device const uchar* block = w_q3k + (uint64_t)nrow * row_stride + (uint)i * Q3K_BYTES;
        device const uchar* hmask = block;
        device const uchar* qs    = block + 32;
        device const uchar* sc_raw = block + 96;
        device const half*  d_ptr = (device const half*)(block + 108);
        float d_all = float(*d_ptr);

        // Scale sb_first.
        uchar scale_byte_f;
        if (sb_first < 4) {
            scale_byte_f = (sc_raw[sb_first] & 0x0Fu)
                         | ((sc_raw[8 + sb_first] & 0x03u) << 4u);
        } else if (sb_first < 8) {
            short ii = sb_first - 4;
            scale_byte_f = (sc_raw[4 + ii] & 0x0Fu)
                         | (((sc_raw[8 + ii] >> 2u) & 0x03u) << 4u);
        } else if (sb_first < 12) {
            short ii = sb_first - 8;
            scale_byte_f = (sc_raw[ii] >> 4u)
                         | (((sc_raw[8 + ii] >> 4u) & 0x03u) << 4u);
        } else {
            short ii = sb_first - 12;
            scale_byte_f = (sc_raw[4 + ii] >> 4u)
                         | (((sc_raw[8 + ii] >> 6u) & 0x03u) << 4u);
        }
        float dl_f = d_all * (float)((int)((char)scale_byte_f) - 32);

        // Scale sb_second.
        uchar scale_byte_s;
        if (sb_second < 4) {
            scale_byte_s = (sc_raw[sb_second] & 0x0Fu)
                         | ((sc_raw[8 + sb_second] & 0x03u) << 4u);
        } else if (sb_second < 8) {
            short ii = sb_second - 4;
            scale_byte_s = (sc_raw[4 + ii] & 0x0Fu)
                         | (((sc_raw[8 + ii] >> 2u) & 0x03u) << 4u);
        } else if (sb_second < 12) {
            short ii = sb_second - 8;
            scale_byte_s = (sc_raw[ii] >> 4u)
                         | (((sc_raw[8 + ii] >> 4u) & 0x03u) << 4u);
        } else {
            short ii = sb_second - 12;
            scale_byte_s = (sc_raw[4 + ii] >> 4u)
                         | (((sc_raw[8 + ii] >> 6u) & 0x03u) << 4u);
        }
        float dl_s = d_all * (float)((int)((char)scale_byte_s) - 32);

        // Inner loop sb_first : 16 weights register-resident yl[].
        float acc_f = 0.0;
        for (uint l = 0; l < 16u; ++l) {
            uint qs_byte    = qs[(uint)qs_base_first + l];
            uint hmask_byte = hmask[(uint)hmask_base_first + l];
            int  q_lo  = (int)((qs_byte >> (uint)shift) & 0x03u);
            int  h_bit = (hmask_byte & m_bit) != 0u;
            int  v     = q_lo - (h_bit ? 0 : 4);
            acc_f += yl[l] * (float)v;
        }
        sumf += acc_f * dl_f;

        // Inner loop sb_second : 16 weights register-resident yh[].
        float acc_s = 0.0;
        for (uint l = 0; l < 16u; ++l) {
            uint qs_byte    = qs[(uint)qs_base_second + l];
            uint hmask_byte = hmask[(uint)hmask_base_second + l];
            int  q_lo  = (int)((qs_byte >> (uint)shift) & 0x03u);
            int  h_bit = (hmask_byte & m_bit) != 0u;
            int  v     = q_lo - (h_bit ? 0 : 4);
            acc_s += yh[l] * (float)v;
        }
        sumf += acc_s * dl_s;

        y_first += 4 * (int)Q3K_WEIGHTS;
    }

    float row_sum = simd_sum(sumf);
    if (tiisg == 0) {
        y[nrow] = row_sum;
    }
}
"#;

/// T160 — Q3_K sgemv NSG=2 NR0=1 ix-stripped fast-path (TurboQuant Q3_K).
///
/// 64 threads par TG = 2 simdgroups × 32. Chaque simdgroup possède 1 row
/// (NR0=1). Au sein d'un simdgroup, ix-stripping : 8 threads/stripe couvrent
/// 32 weights (= 2 sub-blocks de 16) par super-block, sur ¼ des super-blocks.
/// Inner loops 4× plus denses que NSG=1 → meilleure ILP Apple GPU.
///
/// Pré-conditions identiques à NSG=1 (Metal3 + K%256==0).
pub fn sgemv_q3_k_f32_lcpp_nsg2_into(
    backend: &MetalBackend,
    x_buf: &Buffer,
    w_q3k_buf: &Buffer,
    out_buf: &Buffer,
    k: usize,
    n: usize,
) -> Result<(), MetalError> {
    if !backend.supports_metal3() {
        return Err(MetalError::Unsupported(
            "sgemv_q3_k_f32_lcpp_nsg2 needs Metal3".to_string(),
        ));
    }
    if k == 0 || n == 0 || k % 256 != 0 {
        return Err(MetalError::ShapeMismatch(format!(
            "sgemv_q3_k_f32_lcpp_nsg2: K%256==0 required (K={k}, N={n})"
        )));
    }
    let pipeline = backend.pipeline(
        "sgemv_q3_k_f32_lcpp_nsg2",
        SGEMV_Q3_K_F32_LCPP_NSG2_SHADER,
        "sgemv_q3_k_f32_lcpp_nsg2",
    )?;
    let dims = [k as u32, n as u32];
    backend.with_encoder(|encoder| {
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(x_buf), 0);
        encoder.set_buffer(1, Some(w_q3k_buf), 0);
        encoder.set_buffer(2, Some(out_buf), 0);
        encoder.set_bytes(3, 8, dims.as_ptr() as *const std::ffi::c_void);
        // 64 threads/tg = 2 simdgroups × 32. Each tg processes 2 rows.
        let tg_size = MTLSize::new(64, 1, 1);
        let n_tg = (n as u64).div_ceil(2);
        let groups = MTLSize::new(n_tg, 1, 1);
        encoder.dispatch_thread_groups(groups, tg_size);
    });
    Ok(())
}

// T142 — Q5_K matmul-vec kernel.
//
// Q5_K format: 256 weights / 176-byte super-block.
//   2 bytes d (half) + 2 bytes dmin (half) + 12 packed scales/mins
//   + 32 bytes qh (high bit per weight) + 128 bytes qs (low 4 bits per weight)
//
// 8 sub-blocks of 32 weights each. Sub-block i has its own (sc, m) pair
// (encoded in the 12 scales bytes — same packing as Q4_K).
// Per-weight value: w = d * sc_i * (low4 + (qh & u_bit ? 16 : 0)) - dmin * m_i
//
// Dispatch matches llama.cpp's `N_R0_Q5_K = 1, N_SG_Q5_K = 2`:
//   - 64 threads/threadgroup (2 simdgroups × 32)
//   - Each simdgroup processes NR0_Q5K = 1 row
//   - Each threadgroup processes NSG_Q5K * NR0_Q5K = 2 rows
//   - first_row = (tg_id * NSG + sgitg) * NR0_Q5K
//
// We use NR0=1 (vs Q4_K's NR0=2) because Q5_K has more per-block state
// (qh + qs split + 8 sub-block scales) which raises register pressure.
const SGEMV_Q5_K_F32_LCPP_NSG2_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

constant uint Q5K_BYTES = 176u;
constant uint Q5K_WEIGHTS = 256u;
constant short NR0_Q5K = 1;
constant short NSG_Q5K = 2;
constant ushort KMASK1 = 0x3f3f;
constant ushort KMASK2 = 0x0f0f;
constant ushort KMASK3 = 0xc0c0;

kernel void sgemv_q5_k_f32_lcpp_nsg2(
    device const float*  x      [[buffer(0)]],
    device const uchar*  w_q5k  [[buffer(1)]],
    device float*        y      [[buffer(2)]],
    constant uint2&      dims   [[buffer(3)]],
    uint                 tg_id  [[threadgroup_position_in_grid]],
    ushort               tiisg  [[thread_index_in_simdgroup]],
    ushort               sgitg  [[simdgroup_index_in_threadgroup]]
) {
    uint K = dims.x;
    uint N = dims.y;
    uint blocks_per_row = K / Q5K_WEIGHTS;

    // tg_id covers NSG*NR0 rows; this simdgroup's first row.
    uint first_row = (tg_id * (uint)NSG_Q5K + (uint)sgitg) * (uint)NR0_Q5K;
    if (first_row >= N) return;

    short tid = (short)(tiisg / 4u);   // 0..7
    short ix  = (short)(tiisg % 4u);   // 0..3 — partition K-blocks across 4 stripes
    short iq  = tid / 4;                // 0 or 1 — pick low/high half of qs
    short ir  = tid % 4;                // 0..3 — pick 8-element stripe within half

    short l0 = 8 * ir;
    short q_offset = 32 * iq + l0;
    short y_offset = 64 * iq + l0;

    // qh bit-position selectors for the 4 32-element groups within a sub-block-pair.
    uchar hm1 = 1u << (2*iq);
    uchar hm2 = hm1 << 1;
    uchar hm3 = hm1 << 4;
    uchar hm4 = hm2 << 4;

    int nb = (int)blocks_per_row;
    uint row_stride = blocks_per_row * Q5K_BYTES;

    float sumf = 0.0;
    float yl[16];
    float yh[16];
    ushort sc16[4];
    thread const uchar* sc8 = (thread const uchar*)sc16;

    // Walk x in stripes of 4 super-blocks, each thread loading its 16 floats
    // (low half) + 16 floats (high half) into yl/yh registers. ix selects which
    // of the 4 stripes this thread participates in.
    device const float* y1 = x + (uint)ix * Q5K_WEIGHTS + (uint)y_offset;

    for (int i = ix; i < nb; i += 4) {
        device const float* y2 = y1 + 128;
        float4 sumy = {0.0, 0.0, 0.0, 0.0};
        for (short l = 0; l < 8; ++l) {
            yl[l + 0] = y1[l + 0];   sumy[0] += yl[l + 0];
            yl[l + 8] = y1[l + 32];  sumy[1] += yl[l + 8];
            yh[l + 0] = y2[l + 0];   sumy[2] += yh[l + 0];
            yh[l + 8] = y2[l + 32];  sumy[3] += yh[l + 8];
        }

        // Single-row inner: this simdgroup processes exactly NR0=1 row.
        device const uchar* block = w_q5k + (uint64_t)first_row * row_stride + (uint)i * Q5K_BYTES;
        device const uint16_t* dh_ptr = (device const uint16_t*)(block);
        float d    = float(as_type<half>(dh_ptr[0]));
        float dmin = float(as_type<half>(dh_ptr[1]));

        // Scales/mins follow the same packing as Q4_K (12 bytes at offset 4).
        device const uint16_t* a = (device const uint16_t*)(block + 4) + iq;
        sc16[0] = a[0] & KMASK1;
        sc16[1] = a[2] & KMASK1;
        sc16[2] = ((a[4] >> 0) & KMASK2) | ((a[0] & KMASK3) >> 2);
        sc16[3] = ((a[4] >> 4) & KMASK2) | ((a[2] & KMASK3) >> 2);

        // qh (high bit per weight) follows scales: 32 bytes at offset 16.
        device const uchar* qh = (device const uchar*)(block + 16) + (uint)l0;
        // qs (low 4 bits per weight): 128 bytes at offset 16+32=48.
        device const uchar* q1 = (device const uchar*)(block + 48) + (uint)q_offset;
        device const uchar* q2 = q1 + 64;

        float4 acc1 = {0.0, 0.0, 0.0, 0.0};
        float4 acc2 = {0.0, 0.0, 0.0, 0.0};
        for (short l = 0; l < 8; ++l) {
            uchar h = qh[l];
            acc1[0] += yl[l + 0] * (float)(q1[l] & 0x0F);
            acc1[1] += yl[l + 8] * (float)(q1[l] & 0xF0);
            acc1[2] += yh[l + 0] * (float)(q2[l] & 0x0F);
            acc1[3] += yh[l + 8] * (float)(q2[l] & 0xF0);
            acc2[0] += (h & hm1) ? yl[l + 0] : 0.0;
            acc2[1] += (h & hm2) ? yl[l + 8] : 0.0;
            acc2[2] += (h & hm3) ? yh[l + 0] : 0.0;
            acc2[3] += (h & hm4) ? yh[l + 8] : 0.0;
        }

        sumf += d * (
            float(sc8[0]) * (acc1[0]        + 16.0 * acc2[0]) +
            float(sc8[1]) * (acc1[1]/16.0   + 16.0 * acc2[1]) +
            float(sc8[4]) * (acc1[2]        + 16.0 * acc2[2]) +
            float(sc8[5]) * (acc1[3]/16.0   + 16.0 * acc2[3])
        ) - dmin * (
            sumy[0] * float(sc8[2]) +
            sumy[1] * float(sc8[3]) +
            sumy[2] * float(sc8[6]) +
            sumy[3] * float(sc8[7])
        );

        y1 += 4 * (int)Q5K_WEIGHTS;
    }

    // Reduce across the 32 simdgroup threads into the row output.
    if (first_row < N) {
        float row_sum = simd_sum(sumf);
        if (tiisg == 0) {
            y[first_row] = row_sum;
        }
    }
}
"#;

/// T142 — Q5_K matmul-vec via NSG=2 dispatch (one row per simdgroup,
/// two simdgroups per threadgroup, two rows per threadgroup). Mirrors
/// our Q4_K / Q6_K nsg2 helpers but with the Q5_K block layout (high
/// bit + low 4 bits split, 176-byte block).
///
/// Used by both Qwen3.6-27B (`ssm_out.weight` is Q5_K) and Qwen3.6-35B-A3B
/// (`ffn_down_exps.weight` per-expert blocks are Q5_K).
pub fn sgemv_q5_k_f32_lcpp_nsg2_into(
    backend: &MetalBackend,
    x_buf: &Buffer,
    w_q5k_buf: &Buffer,
    out_buf: &Buffer,
    k: usize,
    n: usize,
) -> Result<(), MetalError> {
    if !backend.supports_metal3() {
        return Err(MetalError::Unsupported(
            "sgemv_q5_k_f32_lcpp_nsg2 needs Metal3".to_string(),
        ));
    }
    if k == 0 || n == 0 || k % 256 != 0 {
        return Err(MetalError::ShapeMismatch(format!(
            "sgemv_q5_k_f32_lcpp_nsg2: K%256==0 required (K={k}, N={n})"
        )));
    }
    let pipeline = backend.pipeline(
        "sgemv_q5_k_f32_lcpp_nsg2",
        SGEMV_Q5_K_F32_LCPP_NSG2_SHADER,
        "sgemv_q5_k_f32_lcpp_nsg2",
    )?;
    let dims = [k as u32, n as u32];
    backend.with_encoder(|encoder| {
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(x_buf), 0);
        encoder.set_buffer(1, Some(w_q5k_buf), 0);
        encoder.set_buffer(2, Some(out_buf), 0);
        encoder.set_bytes(3, 8, dims.as_ptr() as *const std::ffi::c_void);
        // 64 threads/tg = 2 simdgroups × 32. Each tg processes NSG*NR0 = 2 rows.
        let tg_size = MTLSize::new(64, 1, 1);
        let n_tg = (n as u64).div_ceil(2);
        let groups = MTLSize::new(n_tg, 1, 1);
        encoder.dispatch_thread_groups(groups, tg_size);
    });
    Ok(())
}

// T142 — Q8_0 matmul-vec kernel.
//
// Q8_0 format: 32 weights / 34-byte block. 2 bytes d (half) + 32 bytes int8
// quants. Per-weight: w = d * (int8_value).
//
// Dispatch: NR0=2, NSG=2 (matches our existing lcpp_nsg2 pattern for Q4_K).
//   - 64 threads/threadgroup
//   - Each simdgroup processes 2 rows
//   - Each threadgroup processes 4 rows
//
// Each thread reads one int8 value per block stride (stride = 32) and one
// float of x. The 32 threads in a simdgroup collaboratively cover all 32
// weights of one block.
const SGEMV_Q8_0_F32_LCPP_NSG2_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

constant uint Q8_0_WEIGHTS = 32u;
constant uint Q8_0_BYTES = 34u;
constant short NR0_Q80 = 2;
constant short NSG_Q80 = 2;

kernel void sgemv_q8_0_f32_lcpp_nsg2(
    device const float*  x      [[buffer(0)]],
    device const uchar*  w_q8   [[buffer(1)]],
    device float*        y      [[buffer(2)]],
    constant uint2&      dims   [[buffer(3)]],
    uint                 tg_id  [[threadgroup_position_in_grid]],
    ushort               tiisg  [[thread_index_in_simdgroup]],
    ushort               sgitg  [[simdgroup_index_in_threadgroup]]
) {
    uint K = dims.x;
    uint N = dims.y;
    uint blocks_per_row = K / Q8_0_WEIGHTS;

    uint first_row = (tg_id * (uint)NSG_Q80 + (uint)sgitg) * (uint)NR0_Q80;
    if (first_row >= N) return;

    int nb = (int)blocks_per_row;
    uint row_stride = blocks_per_row * Q8_0_BYTES;

    // Each thread handles 1 weight position within a block, walking blocks in stride 32.
    short lane = (short)tiisg; // 0..31

    float sumf[2] = {0.0, 0.0};

    for (int ib = 0; ib < nb; ++ib) {
        // Load x for this block at this lane.
        float xv = x[ib * (int)Q8_0_WEIGHTS + (int)lane];

        for (short row = 0; row < NR0_Q80; ++row) {
            uint nrow = first_row + (uint)row;
            if (nrow >= N) continue;
            device const uchar* block = w_q8 + (uint64_t)nrow * row_stride + (uint)ib * Q8_0_BYTES;
            // d = half at offset 0
            device const uint16_t* dh = (device const uint16_t*)block;
            float d = float(as_type<half>(dh[0]));
            // quants: int8 starting at offset 2
            device const char* qs = (device const char*)(block + 2);
            int q = (int)qs[lane];
            sumf[row] += d * (float)q * xv;
        }
    }

    // Reduce across the 32 lanes into the per-row output.
    for (short row = 0; row < NR0_Q80; ++row) {
        uint nrow = first_row + (uint)row;
        if (nrow >= N) continue;
        float row_sum = simd_sum(sumf[row]);
        if (tiisg == 0) {
            y[nrow] = row_sum;
        }
    }
}
"#;

/// T142 — Q8_0 matmul-vec. Used by Qwen3.6-35B-A3B which stores
/// `token_embd.weight` and several attention-block tensors in Q8_0.
/// Same dispatch geometry as our Q4_K nsg2 kernel: NR0=2, NSG=2.
pub fn sgemv_q8_0_f32_lcpp_nsg2_into(
    backend: &MetalBackend,
    x_buf: &Buffer,
    w_q8_buf: &Buffer,
    out_buf: &Buffer,
    k: usize,
    n: usize,
) -> Result<(), MetalError> {
    if !backend.supports_metal3() {
        return Err(MetalError::Unsupported(
            "sgemv_q8_0_f32_lcpp_nsg2 needs Metal3".to_string(),
        ));
    }
    if k == 0 || n == 0 || k % 32 != 0 {
        return Err(MetalError::ShapeMismatch(format!(
            "sgemv_q8_0_f32_lcpp_nsg2: K%32==0 required (K={k}, N={n})"
        )));
    }
    let pipeline = backend.pipeline(
        "sgemv_q8_0_f32_lcpp_nsg2",
        SGEMV_Q8_0_F32_LCPP_NSG2_SHADER,
        "sgemv_q8_0_f32_lcpp_nsg2",
    )?;
    let dims = [k as u32, n as u32];
    backend.with_encoder(|encoder| {
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(x_buf), 0);
        encoder.set_buffer(1, Some(w_q8_buf), 0);
        encoder.set_buffer(2, Some(out_buf), 0);
        encoder.set_bytes(3, 8, dims.as_ptr() as *const std::ffi::c_void);
        let tg_size = MTLSize::new(64, 1, 1);
        let n_tg = (n as u64).div_ceil(4);
        let groups = MTLSize::new(n_tg, 1, 1);
        encoder.dispatch_thread_groups(groups, tg_size);
    });
    Ok(())
}

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

// T123 — Batched lcpp_nr2 Q4_K sgemv. Same pattern as T91 lcpp_nr2 (preload
// x in registers, factored Q4_K formula d*(...) - dmin*(sumy*sc_odd), 2
// rows per simdgroup) but with batch dimension B.
//
// Each threadgroup = 1 simdgroup × NR0=2 rows × 1 batch. Grid indexed
// by tg_flat = batch * n_sg_pairs + sg_pair (row-major batch-then-pairs).
// W weight buffer is shared across all B batches via cache.
//
// CRITICAL: this is the kernel that unlocks speculative decoding speedup.
// The simple T92 batched kernel was 1.8× slower than lcpp_nr2 single-token.
// This batched lcpp_nr2 should match single-token throughput per output.
const SGEMV_Q4_K_F32_LCPP_NR2_BATCH_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

constant uint BLOCK_BYTES = 144u;
constant uint BLOCK_WEIGHTS = 256u;
constant short NR0 = 2;
constant ushort KMASK1 = 0x3f3f;
constant ushort KMASK2 = 0x0f0f;
constant ushort KMASK3 = 0xc0c0;

kernel void sgemv_q4_k_f32_lcpp_nr2_batch(
    device const float*  x      [[buffer(0)]],   // [B, K]
    device const uchar*  w_q4k  [[buffer(1)]],   // [N, K] Q4_K row-major
    device float*        y      [[buffer(2)]],   // [B, N]
    constant uint3&      dims   [[buffer(3)]],   // (K, N, B)
    uint                 tg_flat [[threadgroup_position_in_grid]],
    ushort               tiisg  [[thread_index_in_simdgroup]]
) {
    uint K = dims.x;
    uint N = dims.y;
    uint B = dims.z;
    uint blocks_per_row = K / BLOCK_WEIGHTS;
    uint n_sg_pairs = (N + (uint)NR0 - 1u) / (uint)NR0;

    // Decompose flat tg_id into (batch, sg_pair)
    uint b = tg_flat / n_sg_pairs;
    uint sg_pair = tg_flat % n_sg_pairs;
    if (b >= B) return;

    uint first_row = sg_pair * (uint)NR0;
    if (first_row >= N) return;

    short ix = (short)(tiisg / 8u);    // 0..3
    short it = (short)(tiisg % 8u);    // 0..7
    short iq = it / 4;                 // 0 or 1
    short ir = it % 4;                 // 0..3

    int nb = (int)blocks_per_row;

    // x partition for this batch
    device const float* xb = x + b * K;
    device const float* y4 = xb + ix * 256 + 64 * iq + 8 * ir;

    float yl[16];
    float yh[16];
    float sumf[2] = {0.0, 0.0};

    ushort sc16[4];
    thread const uchar* sc8 = (thread const uchar*)sc16;

    uint row_stride = blocks_per_row * BLOCK_BYTES;

    for (int ib = ix; ib < nb; ib += 4) {
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

            device const uchar* block = w_q4k + (uint64_t)nrow * row_stride + (uint)ib * BLOCK_BYTES;
            device const uint16_t* dh_ptr = (device const uint16_t*)(block);
            float d    = float(as_type<half>(dh_ptr[0]));
            float dmin = float(as_type<half>(dh_ptr[1]));

            device const uint16_t* sc = (device const uint16_t*)(block + 4) + iq;
            device const uint16_t* q1 = (device const uint16_t*)(block + 16) + 16 * iq + 4 * ir;
            device const uint16_t* q2 = q1 + 32;

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
            y[b * N + nrow] = sum_all;
        }
    }
}
"#;

/// T123 — Batched lcpp_nr2 Q4_K sgemv. Critical kernel for speculative
/// decoding speedup — matches single-token lcpp_nr2 throughput per output.
pub fn sgemv_q4_k_f32_lcpp_nr2_batch_into(
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
            "sgemv_q4_k_f32_lcpp_nr2_batch needs Metal3".to_string(),
        ));
    }
    if k == 0 || n == 0 || b == 0 || k % 256 != 0 {
        return Err(MetalError::ShapeMismatch(format!(
            "sgemv_q4_k_f32_lcpp_nr2_batch: K%256==0 required (K={k}, N={n}, B={b})"
        )));
    }
    let pipeline = backend.pipeline(
        "sgemv_q4_k_f32_lcpp_nr2_batch",
        SGEMV_Q4_K_F32_LCPP_NR2_BATCH_SHADER,
        "sgemv_q4_k_f32_lcpp_nr2_batch",
    )?;
    let dims = [k as u32, n as u32, b as u32];
    backend.with_encoder(|encoder| {
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(x_buf), 0);
        encoder.set_buffer(1, Some(w_q4k_buf), 0);
        encoder.set_buffer(2, Some(out_buf), 0);
        encoder.set_bytes(3, 12, dims.as_ptr() as *const std::ffi::c_void);
        let tg_size = MTLSize::new(32, 1, 1);
        let n_sg_pairs = (n as u64).div_ceil(2);
        let n_tg = n_sg_pairs * b as u64;
        let grid = MTLSize::new(32 * n_tg, 1, 1);
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

// T123 — Batched lcpp_nr2 Q6_K sgemv. Same pattern as T93 lcpp_nr2 Q6_K
// but with batch dimension B. Used by forward_batch for W_down.
const SGEMV_Q6_K_F32_LCPP_NR2_BATCH_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

constant uint Q6K_BYTES = 210u;
constant uint Q6K_WEIGHTS = 256u;
constant short NR0_Q6_BATCH = 2;

kernel void sgemv_q6_k_f32_lcpp_nr2_batch(
    device const float*  x      [[buffer(0)]],   // [B, K]
    device const uchar*  w_q6k  [[buffer(1)]],   // [N, K] Q6_K row-major
    device float*        y      [[buffer(2)]],   // [B, N]
    constant uint3&      dims   [[buffer(3)]],   // (K, N, B)
    uint                 tg_flat [[threadgroup_position_in_grid]],
    ushort               tiisg  [[thread_index_in_simdgroup]]
) {
    constexpr uchar KMASK1 = 0x03;
    constexpr uchar KMASK2 = 0x0C;
    constexpr uchar KMASK3 = 0x30;
    constexpr uchar KMASK4 = 0xC0;

    uint K = dims.x;
    uint N = dims.y;
    uint B = dims.z;
    int nb = (int)(K / Q6K_WEIGHTS);
    uint n_sg_pairs = (N + (uint)NR0_Q6_BATCH - 1u) / (uint)NR0_Q6_BATCH;

    uint b = tg_flat / n_sg_pairs;
    uint sg_pair = tg_flat % n_sg_pairs;
    if (b >= B) return;

    uint first_row = sg_pair * (uint)NR0_Q6_BATCH;
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
    device const float* xb = x + b * K;

    for (int i = ix; i < nb; i += 2) {
        device const float* yptr = xb + i * (int)Q6K_WEIGHTS + (int)y_offset;
        for (short l = 0; l < 4; ++l) {
            yl[4*l + 0] = yptr[l +  0];
            yl[4*l + 1] = yptr[l + 32];
            yl[4*l + 2] = yptr[l + 64];
            yl[4*l + 3] = yptr[l + 96];
        }

        for (short row = 0; row < NR0_Q6_BATCH; ++row) {
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

    for (short row = 0; row < NR0_Q6_BATCH; ++row) {
        uint nrow = first_row + (uint)row;
        if (nrow >= N) break;
        float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) {
            y[b * N + nrow] = sum_all;
        }
    }
}
"#;

/// T123 — Batched lcpp_nr2 Q6_K sgemv. Mirror of T93 with batch dim B.
pub fn sgemv_q6_k_f32_lcpp_nr2_batch_into(
    backend: &MetalBackend,
    x_buf: &Buffer,
    w_q6k_buf: &Buffer,
    out_buf: &Buffer,
    k: usize,
    n: usize,
    b: usize,
) -> Result<(), MetalError> {
    if !backend.supports_metal3() {
        return Err(MetalError::Unsupported(
            "sgemv_q6_k_f32_lcpp_nr2_batch needs Metal3".to_string(),
        ));
    }
    if k == 0 || n == 0 || b == 0 || k % 256 != 0 {
        return Err(MetalError::ShapeMismatch(format!(
            "sgemv_q6_k_f32_lcpp_nr2_batch: K%256==0 required (K={k}, N={n}, B={b})"
        )));
    }
    let pipeline = backend.pipeline(
        "sgemv_q6_k_f32_lcpp_nr2_batch",
        SGEMV_Q6_K_F32_LCPP_NR2_BATCH_SHADER,
        "sgemv_q6_k_f32_lcpp_nr2_batch",
    )?;
    let dims = [k as u32, n as u32, b as u32];
    backend.with_encoder(|encoder| {
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(x_buf), 0);
        encoder.set_buffer(1, Some(w_q6k_buf), 0);
        encoder.set_buffer(2, Some(out_buf), 0);
        encoder.set_bytes(3, 12, dims.as_ptr() as *const std::ffi::c_void);
        let tg_size = MTLSize::new(32, 1, 1);
        let n_sg_pairs = (n as u64).div_ceil(2);
        let n_tg = n_sg_pairs * b as u64;
        let grid = MTLSize::new(32 * n_tg, 1, 1);
        encoder.dispatch_threads(grid, tg_size);
    });
    Ok(())
}

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

// T132 — Q6_K version of the NSG=2 kernel. Same body as T93 lcpp_nr2 with
// llama.cpp's `N_SG_Q6_K = 2` dispatch (2 simdgroups per threadgroup).
const SGEMV_Q6_K_F32_LCPP_NSG2_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

constant uint Q6K_BYTES = 210u;
constant uint Q6K_WEIGHTS = 256u;
constant short NR0_Q6 = 2;
constant short NSG_Q6 = 2;

kernel void sgemv_q6_k_f32_lcpp_nsg2(
    device const float*  x      [[buffer(0)]],
    device const uchar*  w_q6k  [[buffer(1)]],
    device float*        y      [[buffer(2)]],
    constant uint2&      dims   [[buffer(3)]],
    uint                 tg_id  [[threadgroup_position_in_grid]],
    ushort               tiisg  [[thread_index_in_simdgroup]],
    ushort               sgitg  [[simdgroup_index_in_threadgroup]]
) {
    constexpr uchar KMASK1 = 0x03;
    constexpr uchar KMASK2 = 0x0C;
    constexpr uchar KMASK3 = 0x30;
    constexpr uchar KMASK4 = 0xC0;

    uint K = dims.x;
    uint N = dims.y;
    int nb = (int)(K / Q6K_WEIGHTS);

    uint first_row = (tg_id * (uint)NSG_Q6 + (uint)sgitg) * (uint)NR0_Q6;
    if (first_row >= N) return;

    short tid = (short)(tiisg / 2u);
    short ix  = (short)(tiisg % 2u);
    short ip  = tid / 8;
    short il  = tid % 8;
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

// T152 — Q6_K gather sgemv : variante gather du `sgemv_q6_k_f32_lcpp_nsg2`.
// Utilisé pour les `down_exps` Q6_K du Qwen3.6-35B-A3B (layers 34, 38, 39).
const SGEMV_Q6_K_GATHER_F32_LCPP_NSG2_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

constant uint Q6K_BYTES = 210u;
constant uint Q6K_WEIGHTS = 256u;
constant short NR0_Q6 = 2;
constant short NSG_Q6 = 2;

kernel void sgemv_q6_k_gather_f32_lcpp_nsg2(
    device const float*  x         [[buffer(0)]],
    device const uchar*  w_q6k     [[buffer(1)]],
    device const uint*   indices   [[buffer(2)]],
    device float*        y         [[buffer(3)]],
    constant uint4&      dims      [[buffer(4)]],   // (K, N, B, expert_stride_bytes)
    constant uint&       x_stride  [[buffer(5)]],
    uint2                tg_id     [[threadgroup_position_in_grid]],
    ushort               tiisg     [[thread_index_in_simdgroup]],
    ushort               sgitg     [[simdgroup_index_in_threadgroup]]
) {
    constexpr uchar KMASK1 = 0x03;
    constexpr uchar KMASK2 = 0x0C;
    constexpr uchar KMASK3 = 0x30;
    constexpr uchar KMASK4 = 0xC0;

    uint K = dims.x;
    uint N = dims.y;
    uint B = dims.z;
    uint expert_stride = dims.w;
    uint b = tg_id.y;
    if (b >= B) return;

    uint expert = indices[b];
    device const uchar* w_base = w_q6k + (uint64_t)expert * (uint64_t)expert_stride;
    device const float* x_base = x + (uint64_t)b * (uint64_t)x_stride;
    device       float* y_base = y + (uint64_t)b * (uint64_t)N;

    int nb = (int)(K / Q6K_WEIGHTS);
    uint first_row = (tg_id.x * (uint)NSG_Q6 + (uint)sgitg) * (uint)NR0_Q6;
    if (first_row >= N) return;

    short tid = (short)(tiisg / 2u);
    short ix  = (short)(tiisg % 2u);
    short ip  = tid / 8;
    short il  = tid % 8;
    short l0  = 4 * il;
    short is  = 8 * ip + l0 / 16;

    short y_offset   = 128 * ip + l0;
    short q_offset_l = 64 * ip + l0;
    short q_offset_h = 32 * ip + l0;

    float sumf[2] = {0.0, 0.0};
    float yl[16];

    uint row_stride = (uint)nb * Q6K_BYTES;

    for (int i = ix; i < nb; i += 2) {
        device const float* yptr = x_base + i * (int)Q6K_WEIGHTS + (int)y_offset;
        for (short l = 0; l < 4; ++l) {
            yl[4*l + 0] = yptr[l +  0];
            yl[4*l + 1] = yptr[l + 32];
            yl[4*l + 2] = yptr[l + 64];
            yl[4*l + 3] = yptr[l + 96];
        }

        for (short row = 0; row < NR0_Q6; ++row) {
            uint nrow = first_row + (uint)row;
            if (nrow >= N) continue;

            device const uchar* block = w_base + (uint64_t)nrow * row_stride + (uint)i * Q6K_BYTES;
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
            y_base[nrow] = sum_all;
        }
    }
}
"#;

/// T152 — Q6_K gather sgemv. Mêmes paramètres que `sgemv_q4_k_gather_*` /
/// `sgemv_q5_k_gather_*`. Utilisé pour les 3 layers Q6_K du Qwen3.6-35B-A3B.
#[allow(clippy::too_many_arguments)]
pub fn sgemv_q6_k_gather_f32_lcpp_nsg2_into(
    backend: &MetalBackend,
    x_buf: &Buffer,
    w_q6k_stacked_buf: &Buffer,
    indices_buf: &Buffer,
    b: usize,
    out_buf: &Buffer,
    k: usize,
    n: usize,
    expert_stride_bytes: usize,
    x_stride_floats: usize,
) -> Result<(), MetalError> {
    if !backend.supports_metal3() {
        return Err(MetalError::Unsupported(
            "sgemv_q6_k_gather_f32_lcpp_nsg2 needs Metal3".to_string(),
        ));
    }
    if k == 0 || n == 0 || k % 256 != 0 || n % 4 != 0 {
        return Err(MetalError::ShapeMismatch(format!(
            "sgemv_q6_k_gather: K%256==0 && N%4==0 required (K={k}, N={n})"
        )));
    }
    if b == 0 {
        return Ok(());
    }
    if x_stride_floats != 0 && x_stride_floats != k {
        return Err(MetalError::ShapeMismatch(format!(
            "sgemv_q6_k_gather: x_stride_floats must be 0 or K={k}, got {x_stride_floats}"
        )));
    }
    let pipeline = backend.pipeline(
        "sgemv_q6_k_gather_f32_lcpp_nsg2",
        SGEMV_Q6_K_GATHER_F32_LCPP_NSG2_SHADER,
        "sgemv_q6_k_gather_f32_lcpp_nsg2",
    )?;
    let dims = [k as u32, n as u32, b as u32, expert_stride_bytes as u32];
    let x_stride_u32 = x_stride_floats as u32;
    backend.with_encoder(|encoder| {
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(x_buf), 0);
        encoder.set_buffer(1, Some(w_q6k_stacked_buf), 0);
        encoder.set_buffer(2, Some(indices_buf), 0);
        encoder.set_buffer(3, Some(out_buf), 0);
        encoder.set_bytes(4, 16, dims.as_ptr() as *const std::ffi::c_void);
        encoder.set_bytes(5, 4, &x_stride_u32 as *const u32 as *const std::ffi::c_void);
        let tg_size = MTLSize::new(64, 1, 1);
        let n_tg = (n as u64).div_ceil(4);
        let groups = MTLSize::new(n_tg, b as u64, 1);
        encoder.dispatch_thread_groups(groups, tg_size);
    });
    Ok(())
}

/// T132 — Q6_K version of the NSG=2 kernel. Mirror of `sgemv_q4_k_f32_lcpp_nsg2_into`
/// for Q6_K weights (lm_head, some W_V). Each threadgroup processes 4 rows.
/// Auto-falls back to NSG=1 lcpp_nr2 for N not divisible by 4.
pub fn sgemv_q6_k_f32_lcpp_nsg2_into(
    backend: &MetalBackend,
    x_buf: &Buffer,
    w_q6k_buf: &Buffer,
    out_buf: &Buffer,
    k: usize,
    n: usize,
) -> Result<(), MetalError> {
    if !backend.supports_metal3() {
        return Err(MetalError::Unsupported(
            "sgemv_q6_k_f32_lcpp_nsg2 needs Metal3".to_string(),
        ));
    }
    if k == 0 || n == 0 || k % 256 != 0 {
        return Err(MetalError::ShapeMismatch(format!(
            "sgemv_q6_k_f32_lcpp_nsg2: K%256==0 required (K={k}, N={n})"
        )));
    }
    if n % 4 != 0 {
        return sgemv_q6_k_f32_lcpp_nr2_into(backend, x_buf, w_q6k_buf, out_buf, k, n);
    }
    let pipeline = backend.pipeline(
        "sgemv_q6_k_f32_lcpp_nsg2",
        SGEMV_Q6_K_F32_LCPP_NSG2_SHADER,
        "sgemv_q6_k_f32_lcpp_nsg2",
    )?;
    let dims = [k as u32, n as u32];
    backend.with_encoder(|encoder| {
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(x_buf), 0);
        encoder.set_buffer(1, Some(w_q6k_buf), 0);
        encoder.set_buffer(2, Some(out_buf), 0);
        encoder.set_bytes(3, 8, dims.as_ptr() as *const std::ffi::c_void);
        let tg_size = MTLSize::new(64, 1, 1);
        let n_tg = (n as u64).div_ceil(4);
        let groups = MTLSize::new(n_tg, 1, 1);
        encoder.dispatch_thread_groups(groups, tg_size);
    });
    Ok(())
}

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
// (T99 + T115 both tried float4 vectorization, both regressed. Likely
// exp() in float4 form not well-pipelined on Apple GPU at this granularity.
// Kept scalar.)
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

// T116 — Append KV slice with float4 vectorization.
// 1 thread = 4 elements. Layout [n_kv, max_seq, head_dim] row-major.
// For Qwen3-14B: total = n_kv * head_dim = 8 * 128 = 1024 floats per
// dispatch. f4 = 256 threads per dispatch. Called 80×/token (40 layers ×
// K and V) — small individual but cumulative load count matters.
kernel void kv_append_f32(
    device const float* src  [[buffer(0)]],
    device float* dst        [[buffer(1)]],
    constant uint3& dims     [[buffer(2)]],   // (n_kv, head_dim, position)
    constant uint& max_seq   [[buffer(3)]],
    uint gid                 [[thread_position_in_grid]]
) {
    uint n_kv     = dims.x;
    uint head_dim = dims.y;
    uint position = dims.z;
    uint total    = n_kv * head_dim;
    uint t4 = total / 4u;

    if (gid < t4) {
        // Vectorized float4 path
        uint flat_base = gid * 4u;
        // Need to map flat indices to (kvh, dd) — for kv_append they're
        // contiguous in src[flat], so we just copy 4 floats. But dst layout
        // has stride: dst[kvh, position, dd] = kvh * max_seq * head_dim + position * head_dim + dd.
        // If 4 consecutive flat indices stay within same kvh row (head_dim=128
        // is divisible by 4), then dst offsets are also contiguous → 1 float4 store.
        // Check: flat_base / head_dim == (flat_base + 3) / head_dim ?
        // For head_dim multiple of 4: yes, flat_base..flat_base+3 are in same row.
        uint kvh = flat_base / head_dim;
        uint dd  = flat_base % head_dim;
        uint dst_off = kvh * max_seq * head_dim + position * head_dim + dd;
        device const float4* src4 = (device const float4*)src;
        device float4* dst4 = (device float4*)(dst + dst_off);
        dst4[0] = src4[gid];
        return;
    }
    // Scalar tail
    uint i = t4 * 4u + (gid - t4);
    if (i < total) {
        uint kvh = i / head_dim;
        uint dd  = i % head_dim;
        uint dst_off = kvh * max_seq * head_dim + position * head_dim + dd;
        dst[dst_off] = src[i];
    }
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
        // T116 — float4 vectorized: total/4 threads + tail
        let t4 = (total / 4) as u64;
        let tail = (total % 4) as u64;
        let grid = MTLSize::new(t4 + tail, 1, 1);
        encoder.dispatch_threads(grid, tg_size);
    });
    Ok(())
}

// T119 — Batched Q6_K sgemv. B independent (x_b @ W_q6k) computed in
// 1 dispatch. Same Q6_K weight buffer shared across all B batches via
// cache; only x reads scale with B. Mirror of T92 for Q6_K.
const SGEMV_Q6_K_F32_BATCH_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

constant uint Q6K_BYTES = 210u;
constant uint Q6K_WEIGHTS = 256u;

kernel void sgemv_q6_k_f32_batch(
    device const float* x       [[buffer(0)]],   // [B, K]
    device const uchar* w_q6k   [[buffer(1)]],   // [N, K] Q6_K row-major
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

    uint blocks_per_row = K / Q6K_WEIGHTS;
    uint row_off = n_idx * blocks_per_row * Q6K_BYTES;
    device const float* xb = x + batch_idx * K;

    float acc = 0.0;

    for (uint blk = 0; blk < blocks_per_row; ++blk) {
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
                acc += xb[x_h_off + l]      * (s1 * float(q1 - 32));
                acc += xb[x_h_off + l + 32] * (s2 * float(q2 - 32));
                acc += xb[x_h_off + l + 64] * (s3 * float(q3 - 32));
                acc += xb[x_h_off + l + 96] * (s4 * float(q4 - 32));
            }
        }
    }

    y[batch_idx * N + n_idx] = acc;
}
"#;

/// T119 — Batched Q6_K sgemv: B independent (x_b @ W_q6k) in 1 dispatch.
pub fn sgemv_q6_k_f32_batch_into(
    backend: &MetalBackend,
    x_buf: &Buffer,
    w_q6k_buf: &Buffer,
    out_buf: &Buffer,
    k: usize,
    n: usize,
    b: usize,
) -> Result<(), MetalError> {
    if !backend.supports_metal3() {
        return Err(MetalError::Unsupported(
            "sgemv_q6_k_f32_batch needs Metal3".to_string(),
        ));
    }
    if k == 0 || n == 0 || b == 0 || k % 256 != 0 {
        return Err(MetalError::ShapeMismatch(format!(
            "sgemv_q6_k_f32_batch: K%256==0 required (K={k}, N={n}, B={b})"
        )));
    }
    let pipeline = backend.pipeline(
        "sgemv_q6_k_f32_batch",
        SGEMV_Q6_K_F32_BATCH_SHADER,
        "sgemv_q6_k_f32_batch",
    )?;
    let dims = [k as u32, n as u32, b as u32];
    backend.with_encoder(|encoder| {
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(x_buf), 0);
        encoder.set_buffer(1, Some(w_q6k_buf), 0);
        encoder.set_buffer(2, Some(out_buf), 0);
        encoder.set_bytes(3, 12, dims.as_ptr() as *const std::ffi::c_void);
        let tg_size = MTLSize::new(64, 1, 1);
        let grid = MTLSize::new(n as u64, b as u64, 1);
        encoder.dispatch_threads(grid, tg_size);
    });
    Ok(())
}

// T120 — Batched element-wise helpers for forward_batch.

const SWIGLU_BATCHED_F32_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

// Element-wise SwiGLU on B × f arrays.
kernel void swiglu_batched_f32(
    device const float* gate [[buffer(0)]],   // [B, f]
    device const float* up   [[buffer(1)]],   // [B, f]
    device float* y          [[buffer(2)]],   // [B, f]
    constant uint2& dims     [[buffer(3)]],   // (f, B)
    uint gid [[thread_position_in_grid]]
) {
    uint f = dims.x;
    uint B = dims.y;
    uint total = f * B;
    if (gid >= total) return;
    float g = gate[gid];
    float s = g / (1.0 + exp(-g));
    y[gid] = s * up[gid];
}
"#;

/// T120 — Batched SwiGLU: element-wise silu(gate)*up on [B, f] arrays.
pub fn swiglu_batched_f32(
    backend: &MetalBackend,
    gate_buf: &Buffer,
    up_buf: &Buffer,
    y_buf: &Buffer,
    f: usize,
    b: usize,
) -> Result<(), MetalError> {
    let pipeline = backend.pipeline(
        "swiglu_batched_f32",
        SWIGLU_BATCHED_F32_SHADER,
        "swiglu_batched_f32",
    )?;
    let dims = [f as u32, b as u32];
    backend.with_encoder(|encoder| {
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(gate_buf), 0);
        encoder.set_buffer(1, Some(up_buf), 0);
        encoder.set_buffer(2, Some(y_buf), 0);
        encoder.set_bytes(3, 8, dims.as_ptr() as *const std::ffi::c_void);
        let tg_size = MTLSize::new(256, 1, 1);
        let grid = MTLSize::new((f * b) as u64, 1, 1);
        encoder.dispatch_threads(grid, tg_size);
    });
    Ok(())
}

const ADD_INPLACE_BATCHED_F32_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void add_inplace_batched_f32(
    device float* x       [[buffer(0)]],   // [B, d]
    device const float* y [[buffer(1)]],   // [B, d]
    constant uint2& dims  [[buffer(2)]],   // (d, B)
    uint gid [[thread_position_in_grid]]
) {
    uint d = dims.x;
    uint B = dims.y;
    uint total = d * B;
    uint t4 = total / 4u;
    if (gid < t4) {
        device float4* x4       = (device float4*)x;
        device const float4* y4 = (device const float4*)y;
        x4[gid] += y4[gid];
        return;
    }
    uint i = t4 * 4u + (gid - t4);
    if (i < total) {
        x[i] += y[i];
    }
}
"#;

/// T120 — Batched in-place add x[B,d] += y[B,d] (float4 vectorized).
pub fn add_inplace_batched_f32(
    backend: &MetalBackend,
    x_buf: &Buffer,
    y_buf: &Buffer,
    d: usize,
    b: usize,
) -> Result<(), MetalError> {
    let pipeline = backend.pipeline(
        "add_inplace_batched_f32",
        ADD_INPLACE_BATCHED_F32_SHADER,
        "add_inplace_batched_f32",
    )?;
    let dims = [d as u32, b as u32];
    let total = d * b;
    backend.with_encoder(|encoder| {
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(x_buf), 0);
        encoder.set_buffer(1, Some(y_buf), 0);
        encoder.set_bytes(2, 8, dims.as_ptr() as *const std::ffi::c_void);
        let tg_size = MTLSize::new(256, 1, 1);
        let t4 = (total / 4) as u64;
        let tail = (total % 4) as u64;
        let grid = MTLSize::new(t4 + tail, 1, 1);
        encoder.dispatch_threads(grid, tg_size);
    });
    Ok(())
}

const RMS_NORM_PER_HEAD_BATCHED_F32_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

// Per-head RMSNorm batched. Each threadgroup = 1 (batch, head). 32 threads
// cooperate on head_dim. gamma is shared across all heads and batches.
kernel void rms_norm_per_head_batched_f32(
    device float* x           [[buffer(0)]],   // [B, n_heads, head_dim]
    device const float* gamma [[buffer(1)]],   // [head_dim]
    constant uint3& dims      [[buffer(2)]],   // (n_heads, head_dim, B)
    constant float& eps       [[buffer(3)]],
    uint tg_flat              [[threadgroup_position_in_grid]],
    uint tid                  [[thread_position_in_threadgroup]],
    uint sg_size              [[threads_per_simdgroup]]
) {
    uint n_heads = dims.x;
    uint head_dim = dims.y;
    uint B        = dims.z;
    uint total_tg = n_heads * B;
    if (tg_flat >= total_tg) return;

    uint h = tg_flat % n_heads;
    uint b = tg_flat / n_heads;

    device float* head = x + (b * n_heads + h) * head_dim;
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

/// T120 — Batched per-head RMSNorm. 1 threadgroup per (batch, head).
pub fn rms_norm_per_head_batched_f32(
    backend: &MetalBackend,
    x_buf: &Buffer,
    gamma_buf: &Buffer,
    n_heads: usize,
    head_dim: usize,
    b: usize,
    eps: f32,
) -> Result<(), MetalError> {
    let pipeline = backend.pipeline(
        "rms_norm_per_head_batched_f32",
        RMS_NORM_PER_HEAD_BATCHED_F32_SHADER,
        "rms_norm_per_head_batched_f32",
    )?;
    let dims = [n_heads as u32, head_dim as u32, b as u32];
    backend.with_encoder(|encoder| {
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(x_buf), 0);
        encoder.set_buffer(1, Some(gamma_buf), 0);
        encoder.set_bytes(2, 12, dims.as_ptr() as *const std::ffi::c_void);
        encoder.set_bytes(3, 4, &eps as *const f32 as *const std::ffi::c_void);
        let tg_size = MTLSize::new(32, 1, 1);
        let total_tg = (n_heads * b) as u64;
        let grid = MTLSize::new(32 * total_tg, 1, 1);
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

// T118 — Fused rms_norm_per_head + rope_half_split. 1 dispatch instead
// of 2 (saves 80 dispatches/token: 40 layers × 2 for Q and K). Each
// threadgroup = 1 head: 32 threads cooperate on RMSNorm reduction, then
// each thread does its rope rotation in-place.
const RMS_NORM_PER_HEAD_THEN_ROPE_F32_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void rms_norm_per_head_then_rope_f32(
    device float* x              [[buffer(0)]],   // [n_heads * head_dim]
    device const float* gamma    [[buffer(1)]],   // [head_dim]
    device const float* cos_tab  [[buffer(2)]],   // [max_seq, head_dim/2]
    device const float* sin_tab  [[buffer(3)]],
    constant uint3& dims         [[buffer(4)]],   // (n_heads, head_dim, position)
    constant float& eps          [[buffer(5)]],
    uint h                       [[threadgroup_position_in_grid]],
    uint tid                     [[thread_position_in_threadgroup]],
    uint sg_size                 [[threads_per_simdgroup]]
) {
    uint n_heads  = dims.x;
    uint head_dim = dims.y;
    uint position = dims.z;
    if (h >= n_heads) return;

    device float* head = x + h * head_dim;

    // Phase 1: RMSNorm (compute inv_rms, normalize + multiply by gamma)
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

    // Phase 2: RoPE in-place. Each thread handles its (k, k+half_dim) pair
    // for k in [tid, tid + sg_size, tid + 2*sg_size, ...] up to half_dim-1.
    // No barrier needed since all threads have written their normalized
    // values; the rope reads we do here only access positions written by
    // this same threadgroup.
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint half_dim = head_dim / 2u;
    for (uint k = tid; k < half_dim; k += sg_size) {
        uint i0 = k;
        uint i1 = k + half_dim;
        uint tab_off = position * half_dim + k;
        float c = cos_tab[tab_off];
        float s = sin_tab[tab_off];
        float x0 = head[i0];
        float x1 = head[i1];
        head[i0] = x0 * c - x1 * s;
        head[i1] = x1 * c + x0 * s;
    }
}
"#;

/// T118 — Fused per-head RMSNorm + RoPE. Saves 1 dispatch per call.
#[allow(clippy::too_many_arguments)]
pub fn rms_norm_per_head_then_rope_f32(
    backend: &MetalBackend,
    x_buf: &Buffer,
    gamma_buf: &Buffer,
    cos_buf: &Buffer,
    sin_buf: &Buffer,
    n_heads: usize,
    head_dim: usize,
    position: usize,
    eps: f32,
) -> Result<(), MetalError> {
    let pipeline = backend.pipeline(
        "rms_norm_per_head_then_rope_f32",
        RMS_NORM_PER_HEAD_THEN_ROPE_F32_SHADER,
        "rms_norm_per_head_then_rope_f32",
    )?;
    let dims = [n_heads as u32, head_dim as u32, position as u32];
    backend.with_encoder(|encoder| {
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(x_buf), 0);
        encoder.set_buffer(1, Some(gamma_buf), 0);
        encoder.set_buffer(2, Some(cos_buf), 0);
        encoder.set_buffer(3, Some(sin_buf), 0);
        encoder.set_bytes(4, 12, dims.as_ptr() as *const std::ffi::c_void);
        encoder.set_bytes(5, 4, &eps as *const f32 as *const std::ffi::c_void);
        let tg_size = MTLSize::new(32, 1, 1);
        let grid = MTLSize::new(32 * n_heads as u64, 1, 1);
        encoder.dispatch_threads(grid, tg_size);
    });
    Ok(())
}

const ROPE_HALF_SPLIT_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

// Apply RoPE half-split convention in place on one [n_heads * head_dim]
// row. Only the first `rope_dim` dims of each head are rotated; dims
// [rope_dim..head_dim] are left untouched (this is the "partial RoPE"
// convention used by Qwen3.5 / 3.6, where rope_dim < head_dim).
//
// Half-split: pair dim k with dim (k + rope_dim/2). For k in 0..rope_dim/2:
//   x'[k]            = x[k]            * cos(angle) - x[k + R/2] * sin(angle)
//   x'[k+R/2]        = x[k+R/2]        * cos(angle) + x[k]       * sin(angle)
// where R = rope_dim and angle = position * theta_k.
//
// `cos_tab` / `sin_tab` shape: [max_seq, rope_dim/2] row-major.
kernel void rope_half_split_f32(
    device float* x              [[buffer(0)]],   // [n_heads * head_dim]
    device const float* cos_tab  [[buffer(1)]],   // [max_seq, rope_dim/2]
    device const float* sin_tab  [[buffer(2)]],
    constant uint4& dims         [[buffer(3)]],   // (n_heads, head_dim, rope_dim, position)
    uint gid                     [[thread_position_in_grid]]
) {
    uint n_heads  = dims.x;
    uint head_dim = dims.y;
    uint rope_dim = dims.z;
    uint position = dims.w;
    uint half_rope = rope_dim / 2u;
    uint total     = n_heads * half_rope;
    if (gid >= total) return;

    uint h = gid / half_rope;
    uint k = gid % half_rope;
    // Only the first rope_dim of each head_dim are rotated. head_dim
    // is the stride between heads (we never touch dims [rope_dim..head_dim]).
    uint i0 = h * head_dim + k;
    uint i1 = h * head_dim + k + half_rope;

    uint tab_off = position * half_rope + k;
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
///
/// Only the first `rope_dim` dimensions of each head are rotated; the
/// remaining `head_dim - rope_dim` dims (when `rope_dim < head_dim`)
/// are left untouched. For a "full RoPE" model where every head dim
/// participates in the rotation, pass `rope_dim == head_dim`.
///
/// `cos_buf` / `sin_buf` are pre-built tables of shape
/// `[max_seq, rope_dim / 2]` (row-major).
pub fn rope_half_split_f32(
    backend: &MetalBackend,
    x_buf: &Buffer,
    cos_buf: &Buffer,
    sin_buf: &Buffer,
    n_heads: usize,
    head_dim: usize,
    rope_dim: usize,
    position: usize,
) -> Result<(), MetalError> {
    if rope_dim == 0 || rope_dim > head_dim || rope_dim % 2 != 0 {
        return Err(MetalError::ShapeMismatch(format!(
            "rope_half_split_f32: rope_dim={rope_dim} must be even and \
             within (0, head_dim={head_dim}]"
        )));
    }
    let pipeline = backend.pipeline(
        "rope_half_split_f32",
        ROPE_HALF_SPLIT_SHADER,
        "rope_half_split_f32",
    )?;
    let dims = [
        n_heads as u32,
        head_dim as u32,
        rope_dim as u32,
        position as u32,
    ];
    let total = n_heads * (rope_dim / 2);
    backend.with_encoder(|encoder| {
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(x_buf), 0);
        encoder.set_buffer(1, Some(cos_buf), 0);
        encoder.set_buffer(2, Some(sin_buf), 0);
        encoder.set_bytes(3, 16, dims.as_ptr() as *const std::ffi::c_void);
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

// T162 phase 9a — Batched RoPE half-split with partial `rope_dim`. Only the
// first `rope_dim` dims of each head are rotated; the remaining
// `head_dim - rope_dim` dims are untouched. Used by Qwen3.5 / 3.6 which apply
// RoPE to a prefix of head_dim only (`cfg.rope_dim < cfg.head_dim`).
const ROPE_HALF_SPLIT_PARTIAL_BATCHED_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void rope_half_split_partial_batched_f32(
    device float* x              [[buffer(0)]],   // [B, n_heads * head_dim]
    device const float* cos_tab  [[buffer(1)]],   // [max_seq, rope_dim/2]
    device const float* sin_tab  [[buffer(2)]],
    constant uint4& dims_a       [[buffer(3)]],   // (n_heads, head_dim, rope_dim, position_base)
    constant uint&  B_arg        [[buffer(4)]],
    uint2 gid                    [[thread_position_in_grid]]
) {
    uint n_heads  = dims_a.x;
    uint head_dim = dims_a.y;
    uint rope_dim = dims_a.z;
    uint pos_base = dims_a.w;
    uint B        = B_arg;

    uint b = gid.y;
    if (b >= B) return;

    uint half_rope = rope_dim / 2u;
    uint flat = gid.x;
    uint total = n_heads * half_rope;
    if (flat >= total) return;

    uint h = flat / half_rope;
    uint k = flat % half_rope;
    // head_dim is the stride between heads; we never touch dims [rope_dim..head_dim].
    uint row_off = b * (n_heads * head_dim);
    uint i0 = row_off + h * head_dim + k;
    uint i1 = row_off + h * head_dim + k + half_rope;

    uint position = pos_base + b;
    uint tab_off = position * half_rope + k;
    float c = cos_tab[tab_off];
    float s = sin_tab[tab_off];
    float x0 = x[i0];
    float x1 = x[i1];
    x[i0] = x0 * c - x1 * s;
    x[i1] = x1 * c + x0 * s;
}
"#;

/// T162 phase 9a — Batched partial RoPE. Apply half-split RoPE in place to B
/// rows of [n_heads * head_dim], each at sequence position `position_base + b`,
/// rotating only the first `rope_dim` dims of each head (the remaining
/// `head_dim - rope_dim` dims are untouched). Equivalent to B sequential
/// `rope_half_split_f32` calls.
#[allow(clippy::too_many_arguments)]
pub fn rope_half_split_partial_batched_f32(
    backend: &MetalBackend,
    x_buf: &Buffer,
    cos_buf: &Buffer,
    sin_buf: &Buffer,
    n_heads: usize,
    head_dim: usize,
    rope_dim: usize,
    position_base: usize,
    b: usize,
) -> Result<(), MetalError> {
    if rope_dim == 0 || rope_dim > head_dim || rope_dim % 2 != 0 {
        return Err(MetalError::ShapeMismatch(format!(
            "rope_half_split_partial_batched_f32: rope_dim={rope_dim} must be even \
             and within (0, head_dim={head_dim}]"
        )));
    }
    if b == 0 {
        return Err(MetalError::ShapeMismatch(
            "rope_half_split_partial_batched_f32: B must be > 0".to_string(),
        ));
    }
    let pipeline = backend.pipeline(
        "rope_half_split_partial_batched_f32",
        ROPE_HALF_SPLIT_PARTIAL_BATCHED_SHADER,
        "rope_half_split_partial_batched_f32",
    )?;
    let dims_a = [
        n_heads as u32,
        head_dim as u32,
        rope_dim as u32,
        position_base as u32,
    ];
    let b_arg = b as u32;
    let total = n_heads * (rope_dim / 2);
    backend.with_encoder(|encoder| {
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(x_buf), 0);
        encoder.set_buffer(1, Some(cos_buf), 0);
        encoder.set_buffer(2, Some(sin_buf), 0);
        encoder.set_bytes(3, 16, dims_a.as_ptr() as *const std::ffi::c_void);
        encoder.set_bytes(4, 4, &b_arg as *const u32 as *const std::ffi::c_void);
        let tg_size = MTLSize::new(64, 1, 1);
        let grid = MTLSize::new(total as u64, b as u64, 1);
        encoder.dispatch_threads(grid, tg_size);
    });
    Ok(())
}

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

    // T113 — Phase D V-weighted-sum with float4 vectorization.
    // For head_dim=128, hd4=32 = sg_size → each thread handles exactly
    // 1 float4. Reduces 4 scalar reads/multiplies to 1 vector read/multiply.
    // T101 — pre-multiply shared[p] *= inv_sum cooperatively.
    for (uint p = tid; p < kv_len; p += sg_size) {
        shared[p] *= inv_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device const float* v_h_base = v_cache + kv_h * max_seq * head_dim;
    device float* out_h = out + q_h * head_dim;

    uint hd4_d = head_dim / 4u;
    device float4* out_h4 = (device float4*)out_h;
    for (uint d4 = tid; d4 < hd4_d; d4 += sg_size) {
        float4 acc = float4(0.0, 0.0, 0.0, 0.0);
        for (uint p = 0; p < kv_len; ++p) {
            device const float4* v_p4 = (device const float4*)(v_h_base + p * head_dim);
            acc += shared[p] * v_p4[d4];
        }
        out_h4[d4] = acc;
    }
    // Scalar tail if head_dim % 4 != 0
    uint tail_start = hd4_d * 4u;
    for (uint d = tail_start + tid; d < head_dim; d += sg_size) {
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

// T109 — Batched GQA decode. B queries at consecutive positions
// [pos_base, pos_base+1, ..., pos_base+B-1], each attending to KV cache
// up to its own position (causal mask). Threadgroup mapping: (q_h, b).
//
// Shared memory: kv_len_max = pos_base + B floats per threadgroup, used
// for softmax score scratch.
const GQA_DECODE_BATCHED_F32_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void gqa_decode_batched_f32(
    device const float* q       [[buffer(0)]],
    device const float* k_cache [[buffer(1)]],
    device const float* v_cache [[buffer(2)]],
    device float* out           [[buffer(3)]],
    constant uint4& dims_a      [[buffer(4)]],   // (n_heads, n_kv, head_dim, B)
    constant uint2& dims_b      [[buffer(5)]],   // (max_seq, pos_base)
    constant float& inv_sqrt_d  [[buffer(6)]],
    threadgroup float* shared   [[threadgroup(0)]],
    uint tg_flat                [[threadgroup_position_in_grid]],
    uint tid                    [[thread_position_in_threadgroup]],
    uint sg_size                [[threads_per_simdgroup]]
) {
    uint n_heads  = dims_a.x;
    uint n_kv     = dims_a.y;
    uint head_dim = dims_a.z;
    uint B        = dims_a.w;
    uint max_seq  = dims_b.x;
    uint pos_base = dims_b.y;

    // Flat tg id → (q_h, b) via row-major
    uint q_h = tg_flat % n_heads;
    uint b   = tg_flat / n_heads;
    if (q_h >= n_heads || b >= B) return;

    uint group_size = n_heads / n_kv;
    uint kv_h = q_h / group_size;
    uint kv_len = pos_base + b + 1u; // causal: include self at position pos_base+b

    // q location for this (b, q_h)
    device const float* q_h_ptr = q + b * (n_heads * head_dim) + q_h * head_dim;
    device const float* k_h_base = k_cache + kv_h * max_seq * head_dim;

    uint hd4 = head_dim / 4u;
    device const float4* q_h_ptr4 = (device const float4*)q_h_ptr;

    // Phase A: scores
    for (uint p = tid; p < kv_len; p += sg_size) {
        device const float4* k_p4 = (device const float4*)(k_h_base + p * head_dim);
        float4 acc4 = float4(0.0, 0.0, 0.0, 0.0);
        for (uint d4i = 0; d4i < hd4; ++d4i) {
            acc4 += q_h_ptr4[d4i] * k_p4[d4i];
        }
        float dot = acc4.x + acc4.y + acc4.z + acc4.w;
        for (uint d = hd4 * 4u; d < head_dim; ++d) {
            dot += q_h_ptr[d] * k_h_base[p * head_dim + d];
        }
        shared[p] = dot * inv_sqrt_d;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase B: max-shift
    float local_max = -INFINITY;
    for (uint p = tid; p < kv_len; p += sg_size) {
        local_max = max(local_max, shared[p]);
    }
    float max_score = simd_max(local_max);

    // Phase C: exp + sum
    float local_sum = 0.0;
    for (uint p = tid; p < kv_len; p += sg_size) {
        float e = exp(shared[p] - max_score);
        shared[p] = e;
        local_sum += e;
    }
    float sum = simd_sum(local_sum);
    float inv_sum = 1.0 / sum;

    // Pre-multiply
    for (uint p = tid; p < kv_len; p += sg_size) {
        shared[p] *= inv_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase D: weighted sum of V
    device const float* v_h_base = v_cache + kv_h * max_seq * head_dim;
    device float* out_h = out + b * (n_heads * head_dim) + q_h * head_dim;
    for (uint d = tid; d < head_dim; d += sg_size) {
        float acc = 0.0;
        for (uint p = 0; p < kv_len; ++p) {
            acc += shared[p] * v_h_base[p * head_dim + d];
        }
        out_h[d] = acc;
    }
}
"#;

/// T109 — Batched GQA decode. Computes attention for B consecutive query
/// positions [pos_base, pos_base+1, ..., pos_base+B-1] in a single dispatch,
/// each with causal mask.
#[allow(clippy::too_many_arguments)]
pub fn gqa_decode_batched_f32(
    backend: &MetalBackend,
    q_buf: &Buffer,
    k_cache: &Buffer,
    v_cache: &Buffer,
    out_buf: &Buffer,
    n_heads: usize,
    n_kv: usize,
    head_dim: usize,
    pos_base: usize,
    b: usize,
    max_seq: usize,
) -> Result<(), MetalError> {
    let pipeline = backend.pipeline(
        "gqa_decode_batched_f32",
        GQA_DECODE_BATCHED_F32_SHADER,
        "gqa_decode_batched_f32",
    )?;
    let dims_a = [n_heads as u32, n_kv as u32, head_dim as u32, b as u32];
    let dims_b = [max_seq as u32, pos_base as u32];
    let inv_sqrt_d: f32 = 1.0 / (head_dim as f32).sqrt();
    // Max kv_len across batches = pos_base + b
    let kv_len_max = pos_base + b;
    let shared_bytes = (kv_len_max * 4) as u64;
    backend.with_encoder(|encoder| {
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(q_buf), 0);
        encoder.set_buffer(1, Some(k_cache), 0);
        encoder.set_buffer(2, Some(v_cache), 0);
        encoder.set_buffer(3, Some(out_buf), 0);
        encoder.set_bytes(4, 16, dims_a.as_ptr() as *const std::ffi::c_void);
        encoder.set_bytes(5, 8, dims_b.as_ptr() as *const std::ffi::c_void);
        encoder.set_bytes(6, 4, &inv_sqrt_d as *const f32 as *const std::ffi::c_void);
        encoder.set_threadgroup_memory_length(0, shared_bytes);
        let tg_size = MTLSize::new(32, 1, 1);
        // Flat 1D dispatch over (q_h, b) tuples; 32 threads per tg.
        let n_tg = (n_heads * b) as u64;
        let grid = MTLSize::new(32 * n_tg, 1, 1);
        encoder.dispatch_threads(grid, tg_size);
    });
    Ok(())
}

const ADD_INPLACE_F32_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

// T114 — In-place residual add with float4 vectorization.
// 1 thread = 4 elements via float4. d=5120 → 1280 threads instead of 5120.
kernel void add_inplace_f32(
    device float* x       [[buffer(0)]],
    device const float* y [[buffer(1)]],
    constant uint& d      [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    uint d4 = d / 4u;
    if (gid < d4) {
        device float4* x4       = (device float4*)x;
        device const float4* y4 = (device const float4*)y;
        x4[gid] += y4[gid];
        return;
    }
    // Scalar tail
    uint i = d4 * 4u + (gid - d4);
    if (i < d) {
        x[i] += y[i];
    }
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
        // T114 — float4 vectorized: d/4 threads + tail
        let d4 = (d / 4) as u64;
        let tail = (d % 4) as u64;
        let grid = MTLSize::new(d4 + tail, 1, 1);
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

// ============================================================================
// T144 — Qwen3.5/3.6 hybrid SSM (Gated DeltaNet) Metal kernels.
//
// These four kernels make up the new Metal forward path for the SSM
// blocks of Qwen3.6-27B and Qwen3.6-35B-A3B (the attention layers of the
// hybrid models reuse our existing kernels). Decode-time M=1 only —
// prefill batched variants will land later if needed.
//
//   1. ssm_conv1d_step_f32    — depth-wise causal 1-D conv with running
//                                ring buffer. Per-channel kernel of size
//                                conv_kernel (typ. 4) × conv_dim (~10240
//                                for 27B, ~8192 for 35B-A3B). Updates the
//                                ring buffer in place.
//   2. l2_norm_per_head_f32   — per-head L2 normalization (sum-of-squares,
//                                no gamma). Used on Q and K after the conv.
//   3. delta_net_step_f32     — gated delta-net recurrence. For each head,
//                                state := exp(gate_h) * state + beta *
//                                outer(v, k); readout = state @ q.
//   4. rms_norm_per_head_gated_f32 — per-head RMS norm with shared gamma
//                                of size head_dim, multiplied pointwise
//                                by silu(z). Replaces a 3-dispatch chain
//                                (rms_norm + silu + mul) with one fused op.
// ============================================================================

// ----------------------------------------------------------------------------
// 1. Depth-wise causal 1-D conv with ring-buffer state update
//
// Layout:
//   conv_state : f32 [(kernel - 1) * conv_dim], row-major:
//                conv_state[t * conv_dim + c] = channel c at history offset t
//   conv1d_w   : f32 [kernel * conv_dim], row-major:
//                conv1d_w[k * conv_dim + c] = conv kernel for channel c at lag k
//   x_in       : f32 [conv_dim] — the new token's qkv_mixed
//   y_out      : f32 [conv_dim] — output of depth-wise conv (per channel)
//
// For each channel c independently:
//   y[c] = sum_{k=0..K-1} w[k, c] * (k < K-1 ? state[k, c] : x_in[c])
//   shift state left by one row (state[t-1, c] := state[t, c] for t in 1..K-1)
//   state[K-2, c] := x_in[c]   ; place current input at the last history slot
//
// Dispatch: one thread per channel. We use threadgroups of 64 threads (matches
// the rest of our pipeline). conv_dim is divisible by 64 in practice (10240,
// 8192, 6144, etc.), so the dispatch is a clean (conv_dim / 64) threadgroups.
// ----------------------------------------------------------------------------

const SSM_CONV1D_STEP_F32_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void ssm_conv1d_step_f32(
    device const float*  x_in        [[buffer(0)]],
    device const float*  conv1d_w    [[buffer(1)]],
    device float*        conv_state  [[buffer(2)]],
    device float*        y_out       [[buffer(3)]],
    constant uint2&      dims        [[buffer(4)]],   // (conv_kernel, conv_dim)
    uint                 gid         [[thread_position_in_grid]]
) {
    uint kernel_size = dims.x;
    uint conv_dim    = dims.y;
    if (gid >= conv_dim) return;

    // GGUF conv1d weight has on-disk layout [conv_dim, kernel_size]
    // (channel-major). The shape `[kernel_size, conv_dim]` recorded in the
    // GGUF header is just GGUF's reversed-dim convention; the raw byte
    // storage is `weight[ch * kernel_size + kp]`. So the per-channel kernel
    // is contiguous and we stride through it with `kp`.
    uint w_base = gid * kernel_size;
    // Conv state ring buffer uses stride `conv_dim` per timestep (no
    // transpose needed since we own the layout): conv_state[t * conv_dim + gid].
    float acc = 0.0;
    for (uint t = 0; t + 1 < kernel_size; ++t) {
        float v = conv_state[t * conv_dim + gid];
        float w = conv1d_w[w_base + t];
        acc += w * v;
    }
    // Add current input × kernel[K-1].
    float xv = x_in[gid];
    acc += conv1d_w[w_base + (kernel_size - 1)] * xv;
    // T144b — fuse SiLU on conv output (was a separate CPU pass).
    float sig = 1.0 / (1.0 + exp(-acc));
    y_out[gid] = acc * sig;

    // Update ring buffer: shift left by one timestep, append current input
    // at last history slot. Each thread handles its own channel — no race.
    if (kernel_size >= 2) {
        for (uint t = 0; t + 2 < kernel_size; ++t) {
            conv_state[t * conv_dim + gid] = conv_state[(t + 1) * conv_dim + gid];
        }
        conv_state[(kernel_size - 2) * conv_dim + gid] = xv;
    }
}
"#;

/// T144 — depth-wise causal 1-D conv step + ring-buffer update.
/// `conv_state` is read+written in place (length `(kernel - 1) * conv_dim`).
/// `y_out` receives the conv result of length `conv_dim`.
///
/// `kernel_size` must be ≥ 1; for kernel_size == 1 the conv is just a per-
/// channel multiply (no history). Typical kernel_size in Qwen3.6 is 4.
pub fn ssm_conv1d_step_f32(
    backend: &MetalBackend,
    x_in_buf: &Buffer,
    conv1d_w_buf: &Buffer,
    conv_state_buf: &Buffer,
    y_out_buf: &Buffer,
    kernel_size: usize,
    conv_dim: usize,
) -> Result<(), MetalError> {
    if kernel_size == 0 || conv_dim == 0 {
        return Err(MetalError::ShapeMismatch(format!(
            "ssm_conv1d_step_f32: kernel_size={kernel_size}, conv_dim={conv_dim}"
        )));
    }
    let pipeline = backend.pipeline(
        "ssm_conv1d_step_f32",
        SSM_CONV1D_STEP_F32_SHADER,
        "ssm_conv1d_step_f32",
    )?;
    let dims = [kernel_size as u32, conv_dim as u32];
    backend.with_encoder(|encoder| {
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(x_in_buf), 0);
        encoder.set_buffer(1, Some(conv1d_w_buf), 0);
        encoder.set_buffer(2, Some(conv_state_buf), 0);
        encoder.set_buffer(3, Some(y_out_buf), 0);
        encoder.set_bytes(4, 8, dims.as_ptr() as *const std::ffi::c_void);
        let tg = MTLSize::new(64, 1, 1);
        let grid = MTLSize::new(conv_dim as u64, 1, 1);
        encoder.dispatch_threads(grid, tg);
    });
    Ok(())
}

// ----------------------------------------------------------------------------
// 2. Per-head L2 normalization (no gamma). Used on Q and K after conv.
//
//   For each head h in 0..n_heads:
//       norm = sqrt(sum_i x[h, i]^2 + eps)
//       x[h, i] /= norm   for i in 0..head_dim
//
// Dispatch: one threadgroup per head, 32 threads/threadgroup (1 simdgroup).
// Each thread handles head_dim/32 elements (head_dim is 128 in both models).
// ----------------------------------------------------------------------------

const L2_NORM_PER_HEAD_F32_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void l2_norm_per_head_f32(
    device float*       x      [[buffer(0)]],
    constant uint2&     dims   [[buffer(1)]],   // (n_heads, head_dim)
    constant float&     eps    [[buffer(2)]],
    uint                tg_id  [[threadgroup_position_in_grid]],
    ushort              tiisg  [[thread_index_in_simdgroup]]
) {
    uint n_heads  = dims.x;
    uint head_dim = dims.y;
    uint head     = tg_id;
    if (head >= n_heads) return;

    uint base = head * head_dim;

    // Sum of squares across the 32 simdgroup lanes.
    float ss = 0.0;
    for (uint i = tiisg; i < head_dim; i += 32u) {
        float v = x[base + i];
        ss += v * v;
    }
    ss = simd_sum(ss);
    float inv = 1.0 / sqrt(ss + eps);

    for (uint i = tiisg; i < head_dim; i += 32u) {
        x[base + i] = x[base + i] * inv;
    }
}
"#;

/// T144 — per-head L2 normalization in place. `x` has shape `[n_heads,
/// head_dim]`; each head is normalized independently. `eps` is added to
/// the sum-of-squares before the sqrt for numerical stability.
///
/// T150 — fix dispatch dimensionality: previously launched on a 2-D grid
/// `(32, n_heads, 1)` while the kernel captures `tg_id` as a `uint` (the
/// X component only). Metal silently truncated `n_heads` Y-axis groups
/// to `tg_id = 0` for every threadgroup, so only head 0 was normalized
/// and heads 1..n_heads-1 retained their pre-norm magnitudes. Symptom:
/// SSM block contributed near-zero to the residual stream because q/k
/// were not unit-norm, the delta-net step's `(k . q)` ≪ 1, the readout
/// was tiny, the gated norm + ssm_out projection collapsed to ~0, and
/// `xd += o` left the residual virtually unchanged across all 32 SSM
/// layers of Qwen3.6-27B → garbage output. Fixed by collapsing the grid
/// to 1-D `(32 * n_heads, 1, 1)`, matching the working dispatch pattern
/// used by `rms_norm_per_head_f32` (q_norm/k_norm, attention).
pub fn l2_norm_per_head_f32(
    backend: &MetalBackend,
    x_buf: &Buffer,
    n_heads: usize,
    head_dim: usize,
    eps: f32,
) -> Result<(), MetalError> {
    let pipeline = backend.pipeline(
        "l2_norm_per_head_f32",
        L2_NORM_PER_HEAD_F32_SHADER,
        "l2_norm_per_head_f32",
    )?;
    let dims = [n_heads as u32, head_dim as u32];
    backend.with_encoder(|encoder| {
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(x_buf), 0);
        encoder.set_bytes(1, 8, dims.as_ptr() as *const std::ffi::c_void);
        encoder.set_bytes(2, 4, &eps as *const f32 as *const std::ffi::c_void);
        let tg = MTLSize::new(32, 1, 1);
        let grid = MTLSize::new(32 * n_heads as u64, 1, 1);
        encoder.dispatch_threads(grid, tg);
    });
    Ok(())
}

// ----------------------------------------------------------------------------
// 3. Gated Delta-Net step — the heart of the SSM block.
//
// State per-head shape: [head_dim, head_dim] (= [128, 128] in Qwen3.6).
// We follow llama.cpp's `build_delta_net_autoregressive` (delta-net-base.cpp,
// matching HF's Qwen3-Next reference): gated DELTA RULE, not plain linear
// attention. For each head h:
//   gamma   = exp(gate_h[h])              ; per-head decay scalar
//   beta_h  = beta[h]                     ; per-head delta scale (sigmoid'd)
//   q_scale = 1 / sqrt(head_dim)          ; standard attention scale on q
//
//   Step 1 (decay):  state[h, r, c] *= gamma
//   Step 2 (project): proj[h, r] = sum_c state[h, r, c] * k[h, c]
//   Step 3 (delta):  delta_r = beta_h * (v[h, r] - proj[h, r])
//                    state[h, r, c] += delta_r * k[h, c]
//   Step 4 (readout): out[h, r] = sum_c state[h, r, c] * (q[h, c] * q_scale)
//
// q, k passed as [n_k_heads, head_dim] — kernel broadcasts to n_v_heads via
// integer division (head_k = head_v / repeat). v has shape [n_v_heads, head_dim].
//
// Dispatch decomposition: one simdgroup (32 threads) per (head, row) pair.
// Steps 1+2, 3, and 4 each iterate the head_dim columns in stride-32 chunks
// across the simdgroup; `simd_sum` reduces both the projection and the readout.
// ----------------------------------------------------------------------------

const DELTA_NET_STEP_F32_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void delta_net_step_f32(
    device const float*  q          [[buffer(0)]],   // [n_k_heads, head_dim] — broadcast inline
    device const float*  k          [[buffer(1)]],   // [n_k_heads, head_dim] — broadcast inline
    device const float*  v          [[buffer(2)]],   // [n_v_heads, head_dim]
    device const float*  gate_h     [[buffer(3)]],   // [n_v_heads]
    device const float*  beta       [[buffer(4)]],   // [n_v_heads]
    device float*        state      [[buffer(5)]],   // [n_v_heads, head_dim, head_dim]
    device float*        out        [[buffer(6)]],   // [n_v_heads, head_dim]
    constant uint4&      dims       [[buffer(7)]],   // (n_v_heads, head_dim, n_k_heads, repeat)
    uint2                tg_id      [[threadgroup_position_in_grid]],
    ushort               tiisg      [[thread_index_in_simdgroup]]
) {
    uint n_v_heads = dims.x;
    uint head_dim  = dims.y;
    uint n_k_heads = dims.z;
    uint head_v = tg_id.y;
    uint row    = tg_id.x;
    if (head_v >= n_v_heads || row >= head_dim) return;

    // Map V-head to K-head. Qwen3.5 / 3.6 use a TILED V layout (per
    // llama.cpp's `_LinearAttentionVReorderBase` reorder_rows): the GGUF
    // converter permutes V from grouped `[k0_v0, k0_v1, k1_v0, k1_v1, ...]`
    // to tiled `[k0_v0, k1_v0, k0_v1, k1_v1, ...]` so that
    //     k_head = v_head % n_k_heads
    //     v_per_k_idx = v_head / n_k_heads
    // (Qwen3-Next uses the grouped layout — `head_v / repeat` — but we don't
    // load that arch through this kernel; qwen35 / qwen35moe both go through
    // the V-reorder converter.)
    uint head_k = head_v % n_k_heads;

    float gamma    = exp(gate_h[head_v]);
    float beta_val = beta[head_v];
    float v_r      = v[head_v * head_dim + row];
    float q_scale  = 1.0 / sqrt((float)head_dim);

    uint state_off = head_v * head_dim * head_dim + row * head_dim;
    uint qk_off    = head_k * head_dim;

    // Steps 1 + 2 fused: decay state in place and accumulate
    //   proj[r] = sum_c (gamma * state[r, c]) * k[c]
    // into a per-thread partial sum, then simd-reduce.
    float proj_partial = 0.0;
    for (uint c = tiisg; c < head_dim; c += 32u) {
        float decayed = gamma * state[state_off + c];
        state[state_off + c] = decayed;
        proj_partial += decayed * k[qk_off + c];
    }
    float proj_r = simd_sum(proj_partial);

    // Step 3: delta-rule update + Step 4: readout (also fused — we already
    // hold the post-update state value).
    float delta_r = beta_val * (v_r - proj_r);
    float out_partial = 0.0;
    for (uint c = tiisg; c < head_dim; c += 32u) {
        float k_c = k[qk_off + c];
        float q_c = q[qk_off + c];
        float updated = state[state_off + c] + delta_r * k_c;
        state[state_off + c] = updated;
        out_partial += updated * q_c * q_scale;
    }

    float row_sum = simd_sum(out_partial);
    if (tiisg == 0) {
        out[head_v * head_dim + row] = row_sum;
    }
}
"#;

// T154-fast — Variante de delta_net_step_f32 avec L2 norm de q et k absorbée.
//
// Pattern math : inv_q et inv_k se factorisent hors des inner loops comme
// scalaires multiplicatifs. Le kernel commence par 2 simd_sums supplémentaires
// pour calculer ss_q = Σ q[c]² et ss_k = Σ k[c]² sur head_dim, puis applique
// inv_k à proj_r et delta_eff (qui multiplie k dans state update), et inv_q
// à la lecture finale (out).
//
// Ce kernel remplace 2 appels `l2_norm_per_head_f32(q)` + `l2_norm_per_head_f32(k)`
// + 1 appel `delta_net_step_f32` par 1 seul dispatch → -2 dispatches/SSM-layer.
// Sur Qwen3.6-27B (48 SSM layers) : -96 dispatches/token.
//
// Coût ALU additionnel : 2 simd_sums par TG (= 2 × head_dim ALU + 2 reductions)
// × n_v_heads × head_dim TGs/layer. Pour 27B (n_v=48, head_dim=128) : ~1.6M ALU
// additionnel/layer × 48 = 77M ALU/token. À 13.4 TFLOPS M4 Max = 5.7 µs/token,
// très inférieur aux ~2 ms/token de dispatch saving.
//
// Le state SSM reste numériquement identique (math validée, cf doc plan T154-fast).
const DELTA_NET_STEP_WITH_L2_F32_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void delta_net_step_with_l2_f32(
    device const float*  q          [[buffer(0)]],   // [n_k_heads, head_dim] — RAW (not L2-normed)
    device const float*  k          [[buffer(1)]],   // [n_k_heads, head_dim] — RAW (not L2-normed)
    device const float*  v          [[buffer(2)]],   // [n_v_heads, head_dim]
    device const float*  gate_h     [[buffer(3)]],   // [n_v_heads]
    device const float*  beta       [[buffer(4)]],   // [n_v_heads]
    device float*        state      [[buffer(5)]],   // [n_v_heads, head_dim, head_dim]
    device float*        out        [[buffer(6)]],   // [n_v_heads, head_dim]
    constant uint4&      dims       [[buffer(7)]],   // (n_v_heads, head_dim, n_k_heads, repeat)
    constant float&      eps        [[buffer(8)]],
    uint2                tg_id      [[threadgroup_position_in_grid]],
    ushort               tiisg      [[thread_index_in_simdgroup]]
) {
    uint n_v_heads = dims.x;
    uint head_dim  = dims.y;
    uint n_k_heads = dims.z;
    uint head_v = tg_id.y;
    uint row    = tg_id.x;
    if (head_v >= n_v_heads || row >= head_dim) return;

    uint head_k = head_v % n_k_heads;
    uint qk_off = head_k * head_dim;

    // Compute inv_q et inv_k pour ce head_k (chaque TG fait sa copie ; cache
    // L1/L2 amortit les reads multiples sur le même head_k).
    float q_ss_partial = 0.0;
    float k_ss_partial = 0.0;
    for (uint c = tiisg; c < head_dim; c += 32u) {
        float qc = q[qk_off + c];
        float kc = k[qk_off + c];
        q_ss_partial += qc * qc;
        k_ss_partial += kc * kc;
    }
    float q_ss = simd_sum(q_ss_partial);
    float k_ss = simd_sum(k_ss_partial);
    float inv_q = rsqrt(q_ss + eps);
    float inv_k = rsqrt(k_ss + eps);

    float gamma    = exp(gate_h[head_v]);
    float beta_val = beta[head_v];
    float v_r      = v[head_v * head_dim + row];
    float q_scale  = 1.0 / sqrt((float)head_dim);

    uint state_off = head_v * head_dim * head_dim + row * head_dim;

    // Steps 1+2 : decay + proj_raw (lit raw k).
    float proj_partial = 0.0;
    for (uint c = tiisg; c < head_dim; c += 32u) {
        float decayed = gamma * state[state_off + c];
        state[state_off + c] = decayed;
        proj_partial += decayed * k[qk_off + c];
    }
    float proj_r = simd_sum(proj_partial) * inv_k;

    // Step 3+4 : delta-rule + readout. delta_eff = delta_r * inv_k absorbe la
    // L2 norm de k dans le state update.
    float delta_r   = beta_val * (v_r - proj_r);
    float delta_eff = delta_r * inv_k;
    float out_partial = 0.0;
    for (uint c = tiisg; c < head_dim; c += 32u) {
        float k_c = k[qk_off + c];
        float q_c = q[qk_off + c];
        float updated = state[state_off + c] + delta_eff * k_c;
        state[state_off + c] = updated;
        out_partial += updated * q_c * q_scale;
    }

    // Apply inv_q (et q_scale déjà inclus) à la lecture finale.
    float row_sum = simd_sum(out_partial) * inv_q;
    if (tiisg == 0) {
        out[head_v * head_dim + row] = row_sum;
    }
}
"#;

/// T154-fast — gated delta-net step avec L2 norm de q,k absorbée.
///
/// Sémantiquement équivalent à : `l2_norm_per_head_f32(q) ; l2_norm_per_head_f32(k) ;
/// delta_net_step_f32(...)`, mais fait en 1 dispatch au lieu de 3. Le state SSM
/// est strictement identique au comportement legacy.
///
/// `q_buf` et `k_buf` doivent contenir les valeurs RAW (PAS L2-normées). Si
/// vous appelez ce kernel, NE PAS appeler `l2_norm_per_head_f32` avant.
#[allow(clippy::too_many_arguments)]
pub fn delta_net_step_with_l2_f32(
    backend: &MetalBackend,
    q_buf: &Buffer,
    k_buf: &Buffer,
    v_buf: &Buffer,
    gate_h_buf: &Buffer,
    beta_buf: &Buffer,
    state_buf: &Buffer,
    out_buf: &Buffer,
    n_v_heads: usize,
    head_dim: usize,
    n_k_heads: usize,
    eps: f32,
) -> Result<(), MetalError> {
    if n_v_heads == 0 || head_dim == 0 || n_k_heads == 0 || n_v_heads % n_k_heads != 0 {
        return Err(MetalError::ShapeMismatch(format!(
            "delta_net_step_with_l2_f32: n_v_heads={n_v_heads} must be a multiple of n_k_heads={n_k_heads}"
        )));
    }
    let pipeline = backend.pipeline(
        "delta_net_step_with_l2_f32",
        DELTA_NET_STEP_WITH_L2_F32_SHADER,
        "delta_net_step_with_l2_f32",
    )?;
    let repeat = (n_v_heads / n_k_heads) as u32;
    let dims = [n_v_heads as u32, head_dim as u32, n_k_heads as u32, repeat];
    backend.with_encoder(|encoder| {
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(q_buf), 0);
        encoder.set_buffer(1, Some(k_buf), 0);
        encoder.set_buffer(2, Some(v_buf), 0);
        encoder.set_buffer(3, Some(gate_h_buf), 0);
        encoder.set_buffer(4, Some(beta_buf), 0);
        encoder.set_buffer(5, Some(state_buf), 0);
        encoder.set_buffer(6, Some(out_buf), 0);
        encoder.set_bytes(7, 16, dims.as_ptr() as *const std::ffi::c_void);
        encoder.set_bytes(8, 4, &eps as *const f32 as *const std::ffi::c_void);
        let tg_size = MTLSize::new(32, 1, 1);
        let groups = MTLSize::new(head_dim as u64, n_v_heads as u64, 1);
        encoder.dispatch_thread_groups(groups, tg_size);
    });
    Ok(())
}

/// T144 — gated delta-net step (state update + read-out).
///
/// `state` is read+written in place. Shape: `[n_v_heads, head_dim,
/// head_dim]`. After the call, `out` contains the per-head readout
/// vector ready for the gated norm.
///
/// `q`, `k`, `v` are all `[n_v_heads, head_dim]`. The caller is expected
/// to have applied L2 norm to q/k and broadcast from n_k_heads to n_v_heads.
/// `gate_h` is `[n_v_heads]` (already softplus(alpha + dt_bias) * ssm_a).
/// `beta` is `[n_v_heads]` (already sigmoid).
///
/// `q_buf` and `k_buf` are stored as `[n_k_heads, head_dim]` and the kernel
/// broadcasts each Q/K head to `n_v_heads / n_k_heads` consecutive value
/// heads via integer division. This eliminates the prior CPU broadcast pass.
/// `n_v_heads` must be a multiple of `n_k_heads`.
#[allow(clippy::too_many_arguments)]
pub fn delta_net_step_f32(
    backend: &MetalBackend,
    q_buf: &Buffer,
    k_buf: &Buffer,
    v_buf: &Buffer,
    gate_h_buf: &Buffer,
    beta_buf: &Buffer,
    state_buf: &Buffer,
    out_buf: &Buffer,
    n_v_heads: usize,
    head_dim: usize,
    n_k_heads: usize,
) -> Result<(), MetalError> {
    if n_v_heads == 0 || head_dim == 0 || n_k_heads == 0 || n_v_heads % n_k_heads != 0 {
        return Err(MetalError::ShapeMismatch(format!(
            "delta_net_step_f32: n_v_heads={n_v_heads} must be a multiple of \
             n_k_heads={n_k_heads}, head_dim={head_dim}"
        )));
    }
    let pipeline = backend.pipeline(
        "delta_net_step_f32",
        DELTA_NET_STEP_F32_SHADER,
        "delta_net_step_f32",
    )?;
    let repeat = (n_v_heads / n_k_heads) as u32;
    let dims = [n_v_heads as u32, head_dim as u32, n_k_heads as u32, repeat];
    backend.with_encoder(|encoder| {
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(q_buf), 0);
        encoder.set_buffer(1, Some(k_buf), 0);
        encoder.set_buffer(2, Some(v_buf), 0);
        encoder.set_buffer(3, Some(gate_h_buf), 0);
        encoder.set_buffer(4, Some(beta_buf), 0);
        encoder.set_buffer(5, Some(state_buf), 0);
        encoder.set_buffer(6, Some(out_buf), 0);
        encoder.set_bytes(7, 16, dims.as_ptr() as *const std::ffi::c_void);
        let tg = MTLSize::new(32, 1, 1);
        // Grid: x = row index (head_dim values), y = head index, z = 1.
        let grid = MTLSize::new(32 * head_dim as u64, n_v_heads as u64, 1);
        encoder.dispatch_threads(grid, tg);
    });
    Ok(())
}

// ----------------------------------------------------------------------------
// 4. Per-head RMSNorm with shared gamma + silu(z) gate, fused.
//
// out[h, i] = (x[h, i] / sqrt(mean(x[h]^2) + eps)) * gamma[i] * silu(z[h, i])
//
// gamma is shape [head_dim] (shared across heads).
// z is shape [n_heads, head_dim] — same shape as x.
//
// Dispatch: one threadgroup per head, 32 threads/threadgroup.
// ----------------------------------------------------------------------------

const RMS_NORM_PER_HEAD_GATED_F32_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void rms_norm_per_head_gated_f32(
    device float*        x       [[buffer(0)]],   // [n_heads, head_dim] — in place
    device const float*  gamma   [[buffer(1)]],   // [head_dim]
    device const float*  z       [[buffer(2)]],   // [n_heads, head_dim]
    constant uint2&      dims    [[buffer(3)]],   // (n_heads, head_dim)
    constant float&      eps     [[buffer(4)]],
    uint                 tg_id   [[threadgroup_position_in_grid]],
    ushort               tiisg   [[thread_index_in_simdgroup]]
) {
    uint n_heads  = dims.x;
    uint head_dim = dims.y;
    uint head = tg_id;
    if (head >= n_heads) return;

    uint base = head * head_dim;
    float sumsq = 0.0;
    for (uint i = tiisg; i < head_dim; i += 32u) {
        float v = x[base + i];
        sumsq += v * v;
    }
    sumsq = simd_sum(sumsq);
    float inv = 1.0 / sqrt(sumsq / float(head_dim) + eps);

    for (uint i = tiisg; i < head_dim; i += 32u) {
        float zv = z[base + i];
        float silu = zv / (1.0 + exp(-zv));
        x[base + i] = x[base + i] * inv * gamma[i] * silu;
    }
}
"#;

/// T144 — per-head RMS norm × silu(z) fused. `x` is normalized in place
/// per-head with shared gamma `[head_dim]`, then multiplied by `silu(z)`.
///
/// T150 — same dispatch fix as `l2_norm_per_head_f32`: collapse the grid
/// to 1-D `(32 * n_heads, 1, 1)` so `uint tg_id [[threadgroup_position_in_grid]]`
/// captures the head index correctly. Previously the 2-D grid silently
/// gave `tg_id = 0` for every threadgroup → only head 0 was processed.
pub fn rms_norm_per_head_gated_f32(
    backend: &MetalBackend,
    x_buf: &Buffer,
    gamma_buf: &Buffer,
    z_buf: &Buffer,
    n_heads: usize,
    head_dim: usize,
    eps: f32,
) -> Result<(), MetalError> {
    let pipeline = backend.pipeline(
        "rms_norm_per_head_gated_f32",
        RMS_NORM_PER_HEAD_GATED_F32_SHADER,
        "rms_norm_per_head_gated_f32",
    )?;
    let dims = [n_heads as u32, head_dim as u32];
    backend.with_encoder(|encoder| {
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(x_buf), 0);
        encoder.set_buffer(1, Some(gamma_buf), 0);
        encoder.set_buffer(2, Some(z_buf), 0);
        encoder.set_bytes(3, 8, dims.as_ptr() as *const std::ffi::c_void);
        encoder.set_bytes(4, 4, &eps as *const f32 as *const std::ffi::c_void);
        let tg = MTLSize::new(32, 1, 1);
        let grid = MTLSize::new(32 * n_heads as u64, 1, 1);
        encoder.dispatch_threads(grid, tg);
    });
    Ok(())
}

// T152.1b — Top-K softmax + normalize 100% GPU. Élimine le drain qui
// précédait la sélection d'experts dans le path MoE.
//
// Avant : `gate_inp.matmul_into → moe_logits` puis `backend.drain() + CPU
// stable softmax + CPU argsort top-K + CPU renormalize`. Le drain coûte
// ~50-200 µs ; le CPU sort de 256 floats est O(N log N) ≈ 5-15 µs sur
// M4 Max P-core.
//
// Maintenant : 1 dispatch d'un threadgroup unique de 256 threads qui :
//   1. simd-reduce max(logits)
//   2. simd-reduce sum(exp(logit - max))
//   3. probs[i] = exp(logit-max)/sum
//   4. top-K serial scan dans le simd 0 thread 0 (K=8, N=256 → 2k ops trivial)
//   5. renormalize les K poids
//   6. écrit out_idx[K] u32 + out_w[K] f32
//
// Threadgroup memory : 256 floats (= 1 KB), tient largement.
//
// Compatible jusqu'à N=256 / K=16. Au-delà, augmenter MAX_N_EXPERTS.
const TOPK_SOFTMAX_NORM_F32_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

constant uint MAX_N_EXPERTS = 256u;
constant uint TG_SIZE = 256u;
constant uint MAX_K = 16u;

kernel void topk_softmax_norm_f32(
    device const float* logits  [[buffer(0)]],   // [N]
    device       uint*  out_idx [[buffer(1)]],   // [K] u32
    device       float* out_w   [[buffer(2)]],   // [K] f32
    constant uint2&     dims    [[buffer(3)]],   // (N, K)
    uint                lid     [[thread_position_in_threadgroup]],
    uint                lane    [[thread_index_in_simdgroup]],
    uint                sg_idx  [[simdgroup_index_in_threadgroup]]
) {
    uint N = dims.x;
    uint K = dims.y;

    threadgroup float s_buf[MAX_N_EXPERTS];
    threadgroup float s_red[8];  // par simdgroup pour réductions inter-sg

    // 1. Lecture logit (1 par thread, supporte N <= TG_SIZE).
    float v = (lid < N) ? logits[lid] : -INFINITY;

    // 2. Max global via simd_max + reduction inter-simdgroup.
    float m_local = simd_max(v);
    if (lane == 0) s_red[sg_idx] = m_local;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg_idx == 0) {
        float t = (lane < TG_SIZE / 32u) ? s_red[lane] : -INFINITY;
        float m_global = simd_max(t);
        if (lane == 0) s_red[0] = m_global;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float m_global = s_red[0];

    // 3. exp(logit - max), réduction de la somme.
    float e = (lid < N) ? exp(v - m_global) : 0.0;
    float s_local = simd_sum(e);
    if (lane == 0) s_red[sg_idx] = s_local;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg_idx == 0) {
        float t = (lane < TG_SIZE / 32u) ? s_red[lane] : 0.0;
        float s_global = simd_sum(t);
        if (lane == 0) s_red[0] = s_global;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float s_global = s_red[0];

    // 4. Probabilités stockées dans s_buf pour le scan top-K.
    float p = (lid < N) ? e / max(s_global, 1e-30f) : -INFINITY;
    if (lid < N) s_buf[lid] = p;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 5. Top-K serial dans thread 0 (K=8, N=256 → ~2K ops trivial).
    //    Renormalize et écrit le résultat.
    if (lid == 0) {
        float top_w_local[MAX_K];
        uint  top_idx_local[MAX_K];
        for (uint k = 0; k < K; ++k) {
            float best = -INFINITY;
            uint  best_idx = 0;
            for (uint i = 0; i < N; ++i) {
                float pv = s_buf[i];
                if (pv > best) { best = pv; best_idx = i; }
            }
            top_idx_local[k] = best_idx;
            top_w_local[k]   = best;
            s_buf[best_idx]  = -INFINITY;
        }
        float sum_w = 0.0;
        for (uint k = 0; k < K; ++k) sum_w += top_w_local[k];
        float inv = 1.0 / max(sum_w, 6.103515625e-5f);
        for (uint k = 0; k < K; ++k) {
            out_idx[k] = top_idx_local[k];
            out_w[k]   = top_w_local[k] * inv;
        }
    }
}
"#;

// T162 phase 9f — Batched top-K softmax + normalize. Une threadgroup par token.
// Tous les tokens en parallèle dans 1 dispatch (au lieu de B dispatches + drains).
const TOPK_SOFTMAX_NORM_BATCHED_F32_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

constant uint MAX_N_EXPERTS = 256u;
constant uint TG_SIZE = 256u;
constant uint MAX_K = 16u;

kernel void topk_softmax_norm_batched_f32(
    device const float* logits  [[buffer(0)]],   // [B, N]
    device       uint*  out_idx [[buffer(1)]],   // [B, K] u32
    device       float* out_w   [[buffer(2)]],   // [B, K] f32
    constant uint3&     dims    [[buffer(3)]],   // (N, K, B)
    uint                tg_id   [[threadgroup_position_in_grid]],
    uint                lid     [[thread_position_in_threadgroup]],
    uint                lane    [[thread_index_in_simdgroup]],
    uint                sg_idx  [[simdgroup_index_in_threadgroup]]
) {
    uint N = dims.x;
    uint K = dims.y;
    uint B = dims.z;
    uint b = tg_id;
    if (b >= B) return;

    device const float* logits_b = logits + (uint64_t)b * (uint64_t)N;
    device       uint*  out_idx_b = out_idx + (uint64_t)b * (uint64_t)K;
    device       float* out_w_b   = out_w   + (uint64_t)b * (uint64_t)K;

    threadgroup float s_buf[MAX_N_EXPERTS];
    threadgroup float s_red[8];

    float v = (lid < N) ? logits_b[lid] : -INFINITY;

    float m_local = simd_max(v);
    if (lane == 0) s_red[sg_idx] = m_local;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg_idx == 0) {
        float t = (lane < TG_SIZE / 32u) ? s_red[lane] : -INFINITY;
        float m_global = simd_max(t);
        if (lane == 0) s_red[0] = m_global;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float m_global = s_red[0];

    float e = (lid < N) ? exp(v - m_global) : 0.0;
    float s_local = simd_sum(e);
    if (lane == 0) s_red[sg_idx] = s_local;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg_idx == 0) {
        float t = (lane < TG_SIZE / 32u) ? s_red[lane] : 0.0;
        float s_global = simd_sum(t);
        if (lane == 0) s_red[0] = s_global;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float s_global = s_red[0];

    float p = (lid < N) ? e / max(s_global, 1e-30f) : -INFINITY;
    if (lid < N) s_buf[lid] = p;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lid == 0) {
        float top_w_local[MAX_K];
        uint  top_idx_local[MAX_K];
        for (uint k = 0; k < K; ++k) {
            float best = -INFINITY;
            uint  best_idx = 0;
            for (uint i = 0; i < N; ++i) {
                float pv = s_buf[i];
                if (pv > best) { best = pv; best_idx = i; }
            }
            top_idx_local[k] = best_idx;
            top_w_local[k]   = best;
            s_buf[best_idx]  = -INFINITY;
        }
        float sum_w = 0.0;
        for (uint k = 0; k < K; ++k) sum_w += top_w_local[k];
        float inv = 1.0 / max(sum_w, 6.103515625e-5f);
        for (uint k = 0; k < K; ++k) {
            out_idx_b[k] = top_idx_local[k];
            out_w_b[k]   = top_w_local[k] * inv;
        }
    }
}
"#;

/// T162 phase 9f — Batched top-K softmax + normalize. Une threadgroup par token,
/// tous les tokens en parallèle dans 1 dispatch.
pub fn topk_softmax_norm_batched_f32(
    backend: &MetalBackend,
    logits_buf: &Buffer,
    out_idx_buf: &Buffer,
    out_w_buf: &Buffer,
    n_experts: usize,
    k: usize,
    b: usize,
) -> Result<(), MetalError> {
    if n_experts == 0 || k == 0 || b == 0 || n_experts > 256 || k > 16 {
        return Err(MetalError::ShapeMismatch(format!(
            "topk_softmax_norm_batched: 0 < N <= 256, 0 < K <= 16, 0 < B (got N={n_experts}, K={k}, B={b})"
        )));
    }
    let pipeline = backend.pipeline(
        "topk_softmax_norm_batched_f32",
        TOPK_SOFTMAX_NORM_BATCHED_F32_SHADER,
        "topk_softmax_norm_batched_f32",
    )?;
    let dims = [n_experts as u32, k as u32, b as u32];
    backend.with_encoder(|encoder| {
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(logits_buf), 0);
        encoder.set_buffer(1, Some(out_idx_buf), 0);
        encoder.set_buffer(2, Some(out_w_buf), 0);
        encoder.set_bytes(3, 12, dims.as_ptr() as *const std::ffi::c_void);
        let tg_size = MTLSize::new(256, 1, 1);
        let groups = MTLSize::new(b as u64, 1, 1);
        encoder.dispatch_thread_groups(groups, tg_size);
    });
    Ok(())
}

/// T152.1b — top-K softmax + normalize 100% GPU sur 1 threadgroup.
/// Pré-condition : `n_experts <= 256` et `k <= 16`. Pour le 35B-A3B :
/// n_experts=256, k=8 → OK.
pub fn topk_softmax_norm_f32(
    backend: &MetalBackend,
    logits_buf: &Buffer,
    out_idx_buf: &Buffer,
    out_w_buf: &Buffer,
    n_experts: usize,
    k: usize,
) -> Result<(), MetalError> {
    if n_experts == 0 || k == 0 || n_experts > 256 || k > 16 {
        return Err(MetalError::ShapeMismatch(format!(
            "topk_softmax_norm: needs 0 < n_experts <= 256, 0 < k <= 16 (got N={n_experts}, K={k})"
        )));
    }
    let pipeline = backend.pipeline(
        "topk_softmax_norm_f32",
        TOPK_SOFTMAX_NORM_F32_SHADER,
        "topk_softmax_norm_f32",
    )?;
    let dims = [n_experts as u32, k as u32];
    backend.with_encoder(|encoder| {
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(logits_buf), 0);
        encoder.set_buffer(1, Some(out_idx_buf), 0);
        encoder.set_buffer(2, Some(out_w_buf), 0);
        encoder.set_bytes(3, 8, dims.as_ptr() as *const std::ffi::c_void);
        let tg_size = MTLSize::new(256, 1, 1);
        let groups = MTLSize::new(1, 1, 1);
        encoder.dispatch_thread_groups(groups, tg_size);
    });
    Ok(())
}

// T152.1 — fused sigmoid + weighted add pour shared expert MoE.
//
// Le path MoE de Qwen3.5/3.6 a un "shared expert" en plus des routed experts.
// Sa contribution finale au résidu est :
//     scalar = sigmoid(dot(gate_inp_shexp, h))
//     xd[i] += moe_acc[i] + scalar * shared_out[i]
//
// Avant T152.1 ça nécessitait `backend.drain() + CPU sigmoid + CPU add`. Le
// drain coûte ~50-200 µs sur Metal Shared mode. Ce kernel élimine le drain :
// `dot_scalar` est calculé via un sgemv N=1 (`sgemv_f32_lcpp_simd_into(h,
// gate_inp_shexp, dot_scalar, d, 1)`) puis ce kernel le lit directement
// depuis le buffer Metal sans round-trip CPU.
const SIGMOID_ADD_MOE_F32_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void sigmoid_add_moe_f32(
    device const float* moe_acc      [[buffer(0)]],   // [D]
    device const float* shared_out   [[buffer(1)]],   // [D]
    device const float* dot_scalar   [[buffer(2)]],   // [1]
    device       float* xd           [[buffer(3)]],   // [D] in/out
    constant uint&      d            [[buffer(4)]],
    uint                gid          [[thread_position_in_grid]]
) {
    if (gid >= d) return;
    float s = 1.0 / (1.0 + exp(-dot_scalar[0]));
    xd[gid] += moe_acc[gid] + s * shared_out[gid];
}
"#;

// T162 phase 9f — Batched fused dot product + sigmoid_add_moe.
// Computes per token b :
//   scalar[b] = dot(gate_inp_shexp, h_post[b])
//   xd[b, i] += moe_acc[b, i] + sigmoid(scalar[b]) * shared_out[b, i]
// One threadgroup per token. The simdgroup cooperatively reduces the dot
// product, then writes the per-element output across the full D.
// Avoids T per-token drains in the MoE end-of-block.
const SIGMOID_ADD_MOE_BATCHED_F32_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void sigmoid_add_moe_batched_f32(
    device const float* moe_acc      [[buffer(0)]],   // [B, D]
    device const float* shared_out   [[buffer(1)]],   // [B, D]
    device const float* gate_inp_shexp [[buffer(2)]], // [D]
    device const float* h_post       [[buffer(3)]],   // [B, D]
    device       float* xd           [[buffer(4)]],   // [B, D] in/out
    constant uint2&     dims         [[buffer(5)]],   // (D, B)
    uint                tg_id        [[threadgroup_position_in_grid]],
    ushort              tiisg        [[thread_index_in_simdgroup]],
    ushort              sgitg        [[simdgroup_index_in_threadgroup]]
) {
    uint D = dims.x;
    uint B = dims.y;
    uint b = tg_id;
    if (b >= B) return;

    // Threadgroup = 4 simdgroups × 32 threads = 128 threads.
    // Phase 1 : compute dot(gate_inp_shexp, h_post[b]) cooperatively.
    threadgroup float s_red[4];
    device const float* h_b = h_post + (uint64_t)b * (uint64_t)D;
    float partial = 0.0;
    for (uint i = sgitg * 32u + tiisg; i < D; i += 128u) {
        partial += gate_inp_shexp[i] * h_b[i];
    }
    float sg_sum = simd_sum(partial);
    if (tiisg == 0) s_red[sgitg] = sg_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgitg == 0) {
        float t = (tiisg < 4) ? s_red[tiisg] : 0.0;
        float dot_global = simd_sum(t);
        if (tiisg == 0) s_red[0] = dot_global;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float dot_b = s_red[0];
    float sig_b = 1.0 / (1.0 + exp(-dot_b));

    // Phase 2 : write xd[b, i] += moe_acc[b, i] + sig_b * shared_out[b, i].
    device const float* acc_b = moe_acc + (uint64_t)b * (uint64_t)D;
    device const float* out_b = shared_out + (uint64_t)b * (uint64_t)D;
    device       float* xd_b  = xd + (uint64_t)b * (uint64_t)D;
    for (uint i = sgitg * 32u + tiisg; i < D; i += 128u) {
        xd_b[i] += acc_b[i] + sig_b * out_b[i];
    }
}
"#;

/// T162 phase 9f — Batched fused dot + sigmoid_add_moe. Une threadgroup par
/// token, dot product cooperatively (4 simdgroups × 32 threads), puis écriture
/// xd[b, i] += moe_acc[b, i] + sigmoid(dot_b) * shared_out[b, i].
#[allow(clippy::too_many_arguments)]
pub fn sigmoid_add_moe_batched_f32(
    backend: &MetalBackend,
    moe_acc_buf: &Buffer,
    shared_out_buf: &Buffer,
    gate_inp_shexp_buf: &Buffer,
    h_post_buf: &Buffer,
    xd_buf: &Buffer,
    d: usize,
    b: usize,
) -> Result<(), MetalError> {
    if d == 0 || b == 0 {
        return Err(MetalError::ShapeMismatch(format!(
            "sigmoid_add_moe_batched_f32: D={d}, B={b}"
        )));
    }
    let pipeline = backend.pipeline(
        "sigmoid_add_moe_batched_f32",
        SIGMOID_ADD_MOE_BATCHED_F32_SHADER,
        "sigmoid_add_moe_batched_f32",
    )?;
    let dims = [d as u32, b as u32];
    backend.with_encoder(|encoder| {
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(moe_acc_buf), 0);
        encoder.set_buffer(1, Some(shared_out_buf), 0);
        encoder.set_buffer(2, Some(gate_inp_shexp_buf), 0);
        encoder.set_buffer(3, Some(h_post_buf), 0);
        encoder.set_buffer(4, Some(xd_buf), 0);
        encoder.set_bytes(5, 8, dims.as_ptr() as *const std::ffi::c_void);
        let tg_size = MTLSize::new(128, 1, 1);
        let groups = MTLSize::new(b as u64, 1, 1);
        encoder.dispatch_thread_groups(groups, tg_size);
    });
    Ok(())
}

/// T152.1 — `xd[i] += moe_acc[i] + sigmoid(dot_scalar[0]) * shared_out[i]`.
/// Élimine le drain `backend.drain() + CPU sigmoid + CPU add` qui suivait
/// les 4 sgemv de l'expert partagé dans le path MoE. Économie : 1 drain
/// par MoE layer × 16 MoE layers = 16 drains/token sur le 35B-A3B.
pub fn sigmoid_add_moe_f32(
    backend: &MetalBackend,
    moe_acc_buf: &Buffer,
    shared_out_buf: &Buffer,
    dot_scalar_buf: &Buffer,
    xd_buf: &Buffer,
    d: usize,
) -> Result<(), MetalError> {
    if d == 0 {
        return Err(MetalError::ShapeMismatch(
            "sigmoid_add_moe_f32: D must be > 0".to_string(),
        ));
    }
    let pipeline = backend.pipeline(
        "sigmoid_add_moe_f32",
        SIGMOID_ADD_MOE_F32_SHADER,
        "sigmoid_add_moe_f32",
    )?;
    let d_u32 = d as u32;
    backend.with_encoder(|encoder| {
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(moe_acc_buf), 0);
        encoder.set_buffer(1, Some(shared_out_buf), 0);
        encoder.set_buffer(2, Some(dot_scalar_buf), 0);
        encoder.set_buffer(3, Some(xd_buf), 0);
        encoder.set_bytes(4, 4, &d_u32 as *const u32 as *const std::ffi::c_void);
        let tg_size = MTLSize::new(64, 1, 1);
        let grid = MTLSize::new(d as u64, 1, 1);
        encoder.dispatch_threads(grid, tg_size);
    });
    Ok(())
}

// T152 — somme pondérée multi-row : `acc[d] += sum_b weights[b] * src[b, d]`.
// Remplace la boucle de `weighted_add_inplace_f32` après le path MoE gather.
// Sans ça on aurait `n_used` dispatchs séparés ; avec ça, un seul.
const WEIGHTED_REDUCE_ADD_F32_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void weighted_reduce_add_f32(
    device const float* src      [[buffer(0)]],   // [B, D]
    device const float* weights  [[buffer(1)]],   // [B]
    device       float* acc      [[buffer(2)]],   // [D] in/out
    constant uint2&     dims     [[buffer(3)]],   // (B, D)
    uint                gid      [[thread_position_in_grid]]
) {
    uint B = dims.x;
    uint D = dims.y;
    if (gid >= D) return;
    float s = 0.0;
    for (uint b = 0; b < B; b++) {
        s += weights[b] * src[b * D + gid];
    }
    acc[gid] += s;
}
"#;

// T162 phase 9f — Batched weighted reduce per token.
// `acc[t, d] += sum_b weights[t, b] * src[t, b, d]` pour T tokens en parallèle.
// Remplace T per-token weighted_reduce_add (T drains).
const WEIGHTED_REDUCE_ADD_BATCHED_F32_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void weighted_reduce_add_batched_f32(
    device const float* src      [[buffer(0)]],   // [T, B_inner, D]
    device const float* weights  [[buffer(1)]],   // [T, B_inner]
    device       float* acc      [[buffer(2)]],   // [T, D] in/out
    constant uint3&     dims     [[buffer(3)]],   // (B_inner, D, T)
    uint2               gid      [[thread_position_in_grid]]
) {
    uint B = dims.x;
    uint D = dims.y;
    uint T = dims.z;
    uint d = gid.x;
    uint t = gid.y;
    if (d >= D || t >= T) return;
    float s = 0.0;
    uint src_base = t * B * D;
    uint w_base = t * B;
    for (uint b = 0; b < B; b++) {
        s += weights[w_base + b] * src[src_base + b * D + d];
    }
    acc[t * D + d] += s;
}
"#;

/// T162 phase 9f — Batched weighted reduce. Pour T tokens × B_inner experts :
/// `acc[t, d] += sum_b weights[t, b] * src[t, b, d]`.
pub fn weighted_reduce_add_batched_f32(
    backend: &MetalBackend,
    src_buf: &Buffer,
    weights_buf: &Buffer,
    acc_buf: &Buffer,
    b_inner: usize,
    d: usize,
    t: usize,
) -> Result<(), MetalError> {
    if b_inner == 0 || d == 0 || t == 0 {
        return Err(MetalError::ShapeMismatch(format!(
            "weighted_reduce_add_batched_f32: B_inner={b_inner}, D={d}, T={t}"
        )));
    }
    let pipeline = backend.pipeline(
        "weighted_reduce_add_batched_f32",
        WEIGHTED_REDUCE_ADD_BATCHED_F32_SHADER,
        "weighted_reduce_add_batched_f32",
    )?;
    let dims = [b_inner as u32, d as u32, t as u32];
    backend.with_encoder(|encoder| {
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(src_buf), 0);
        encoder.set_buffer(1, Some(weights_buf), 0);
        encoder.set_buffer(2, Some(acc_buf), 0);
        encoder.set_bytes(3, 12, dims.as_ptr() as *const std::ffi::c_void);
        let tg = MTLSize::new(64, 1, 1);
        let grid = MTLSize::new(d as u64, t as u64, 1);
        encoder.dispatch_threads(grid, tg);
    });
    Ok(())
}

/// T152 — `acc[d] += sum_b weights[b] * src[b, d]` en un seul dispatch.
/// Utilisé après les 3 gather sgemv MoE pour combiner les `n_used` outputs
/// experts en un accumulateur de taille `d`. Remplace la boucle
/// `for k in 0..n_used { weighted_add_inplace_f32(...) }`.
pub fn weighted_reduce_add_f32(
    backend: &MetalBackend,
    src_buf: &Buffer,
    weights_buf: &Buffer,
    acc_buf: &Buffer,
    b: usize,
    d: usize,
) -> Result<(), MetalError> {
    if b == 0 || d == 0 {
        return Err(MetalError::ShapeMismatch(format!(
            "weighted_reduce_add_f32: B={b}, D={d} must be > 0"
        )));
    }
    let pipeline = backend.pipeline(
        "weighted_reduce_add_f32",
        WEIGHTED_REDUCE_ADD_F32_SHADER,
        "weighted_reduce_add_f32",
    )?;
    let dims = [b as u32, d as u32];
    backend.with_encoder(|encoder| {
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(src_buf), 0);
        encoder.set_buffer(1, Some(weights_buf), 0);
        encoder.set_buffer(2, Some(acc_buf), 0);
        encoder.set_bytes(3, 8, dims.as_ptr() as *const std::ffi::c_void);
        let tg = MTLSize::new(64, 1, 1);
        let grid = MTLSize::new(d as u64, 1, 1);
        encoder.dispatch_threads(grid, tg);
    });
    Ok(())
}

// T146a — SSM block gate computation in one Metal kernel. Used by every
// SSM layer of the Qwen3.5/3.6 hybrid. Replaces the prior CPU pass:
//   gate_h[i]   = softplus(alpha[i] + dt_bias[i]) * ssm_a[i]
//   beta_sig[i] = sigmoid(beta[i])
// Eliminates one drain per SSM layer × 48 SSM layers = 48 drains/token.
const SSM_APPLY_GATE_F32_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void ssm_apply_gate_f32(
    device const float* alpha    [[buffer(0)]],   // [n_v]
    device const float* beta     [[buffer(1)]],   // [n_v]
    device const float* dt_bias  [[buffer(2)]],   // [n_v]
    device const float* ssm_a    [[buffer(3)]],   // [n_v]
    device float*       gate_h   [[buffer(4)]],   // [n_v] out
    device float*       beta_sig [[buffer(5)]],   // [n_v] out
    constant uint&      n        [[buffer(6)]],
    uint                gid      [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    float a = alpha[gid] + dt_bias[gid];
    // Stable softplus
    float sp;
    if (a > 20.0)       sp = a;
    else if (a < -20.0) sp = exp(a);
    else                sp = log(1.0 + exp(a));
    gate_h[gid] = sp * ssm_a[gid];
    beta_sig[gid] = 1.0 / (1.0 + exp(-beta[gid]));
}
"#;

/// T146a — fused SSM-block gate ops:
///
///   - `gate_h[i] = softplus(alpha[i] + dt_bias[i]) * ssm_a[i]`
///   - `beta_sig[i] = sigmoid(beta[i])`
///
/// All buffers are `n_v`-sized (typ. 32-48). Replaces the CPU helper
/// `ssm_apply_gate_ops` and saves one drain per SSM layer.
#[allow(clippy::too_many_arguments)]
pub fn ssm_apply_gate_f32(
    backend: &MetalBackend,
    alpha_buf: &Buffer,
    beta_buf: &Buffer,
    dt_bias_buf: &Buffer,
    ssm_a_buf: &Buffer,
    gate_h_buf: &Buffer,
    beta_sig_buf: &Buffer,
    n_v: usize,
) -> Result<(), MetalError> {
    if n_v == 0 {
        return Err(MetalError::ShapeMismatch(
            "ssm_apply_gate_f32: n_v must be > 0".to_string(),
        ));
    }
    let pipeline = backend.pipeline(
        "ssm_apply_gate_f32",
        SSM_APPLY_GATE_F32_SHADER,
        "ssm_apply_gate_f32",
    )?;
    let n_u = n_v as u32;
    backend.with_encoder(|encoder| {
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(alpha_buf), 0);
        encoder.set_buffer(1, Some(beta_buf), 0);
        encoder.set_buffer(2, Some(dt_bias_buf), 0);
        encoder.set_buffer(3, Some(ssm_a_buf), 0);
        encoder.set_buffer(4, Some(gate_h_buf), 0);
        encoder.set_buffer(5, Some(beta_sig_buf), 0);
        encoder.set_bytes(6, 4, &n_u as *const u32 as *const std::ffi::c_void);
        let tg = MTLSize::new(32, 1, 1);
        let grid = MTLSize::new(n_v as u64, 1, 1);
        encoder.dispatch_threads(grid, tg);
    });
    Ok(())
}

// T162 phase 9e — batched ssm_apply_gate for B-token prefill.
// Same as ssm_apply_gate_f32 but operates on `[B, n_v]` buffers for alpha/beta
// + outputs gate_h/beta_sig in `[B, n_v]`. dt_bias and ssm_a are shared across
// the batch (per-layer constants).
const SSM_APPLY_GATE_BATCHED_F32_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void ssm_apply_gate_batched_f32(
    device const float* alpha    [[buffer(0)]],   // [B, n_v]
    device const float* beta     [[buffer(1)]],   // [B, n_v]
    device const float* dt_bias  [[buffer(2)]],   // [n_v] (shared)
    device const float* ssm_a    [[buffer(3)]],   // [n_v] (shared)
    device float*       gate_h   [[buffer(4)]],   // [B, n_v] out
    device float*       beta_sig [[buffer(5)]],   // [B, n_v] out
    constant uint2&     dims     [[buffer(6)]],   // (n_v, B)
    uint2               gid      [[thread_position_in_grid]]
) {
    uint n = dims.x;
    uint B = dims.y;
    uint i = gid.x;
    uint b = gid.y;
    if (i >= n || b >= B) return;
    uint off = b * n + i;
    float a = alpha[off] + dt_bias[i];
    float sp;
    if (a > 20.0)       sp = a;
    else if (a < -20.0) sp = exp(a);
    else                sp = log(1.0 + exp(a));
    gate_h[off] = sp * ssm_a[i];
    beta_sig[off] = 1.0 / (1.0 + exp(-beta[off]));
}
"#;

/// T162 phase 9e — Batched fused SSM-block gate ops for B tokens.
/// Equivalent to B sequential `ssm_apply_gate_f32` calls.
/// dt_bias and ssm_a are layer-constants (shared across all B tokens).
#[allow(clippy::too_many_arguments)]
pub fn ssm_apply_gate_batched_f32(
    backend: &MetalBackend,
    alpha_buf: &Buffer,
    beta_buf: &Buffer,
    dt_bias_buf: &Buffer,
    ssm_a_buf: &Buffer,
    gate_h_buf: &Buffer,
    beta_sig_buf: &Buffer,
    n_v: usize,
    b: usize,
) -> Result<(), MetalError> {
    if n_v == 0 || b == 0 {
        return Err(MetalError::ShapeMismatch(format!(
            "ssm_apply_gate_batched_f32: n_v={n_v}, B={b}"
        )));
    }
    let pipeline = backend.pipeline(
        "ssm_apply_gate_batched_f32",
        SSM_APPLY_GATE_BATCHED_F32_SHADER,
        "ssm_apply_gate_batched_f32",
    )?;
    let dims = [n_v as u32, b as u32];
    backend.with_encoder(|encoder| {
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(alpha_buf), 0);
        encoder.set_buffer(1, Some(beta_buf), 0);
        encoder.set_buffer(2, Some(dt_bias_buf), 0);
        encoder.set_buffer(3, Some(ssm_a_buf), 0);
        encoder.set_buffer(4, Some(gate_h_buf), 0);
        encoder.set_buffer(5, Some(beta_sig_buf), 0);
        encoder.set_bytes(6, 8, dims.as_ptr() as *const std::ffi::c_void);
        let tg = MTLSize::new(32, 1, 1);
        let grid = MTLSize::new(n_v as u64, b as u64, 1);
        encoder.dispatch_threads(grid, tg);
    });
    Ok(())
}

// T146b — Per-head split of the combined QG buffer (Q + gate, 2x output
// width) into separate Q and gate buffers. Used by Qwen3Next attention
// where wq outputs `2 * head_dim * n_q` and the first half-per-head is Q,
// second half is the gate. Eliminates one drain per attention layer ×
// 16 attention layers = 16 drains/token.
const SPLIT_QG_PER_HEAD_F32_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void split_qg_per_head_f32(
    device const float* qg       [[buffer(0)]],   // [n_q, 2 * head_dim]
    device float*       q        [[buffer(1)]],   // [n_q, head_dim] out
    device float*       gate     [[buffer(2)]],   // [n_q, head_dim] out
    constant uint2&     dims     [[buffer(3)]],   // (n_q, head_dim)
    uint2               gid      [[thread_position_in_grid]]
) {
    uint n_q = dims.x;
    uint head_dim = dims.y;
    uint h = gid.y;
    uint i = gid.x;
    if (h >= n_q || i >= head_dim) return;
    uint src_off = h * 2u * head_dim;
    uint dst_off = h * head_dim;
    q[dst_off + i] = qg[src_off + i];
    gate[dst_off + i] = qg[src_off + head_dim + i];
}
"#;

/// T146b — per-head split of `qg` into `q` and `gate`. `qg` has shape
/// `[n_q, 2 * head_dim]` (the Qwen3Next combined Q + gate output). After
/// the call, `q[h, i] = qg[h, i]` and `gate[h, i] = qg[h, head_dim + i]`.
pub fn split_qg_per_head_f32(
    backend: &MetalBackend,
    qg_buf: &Buffer,
    q_buf: &Buffer,
    gate_buf: &Buffer,
    n_q: usize,
    head_dim: usize,
) -> Result<(), MetalError> {
    if n_q == 0 || head_dim == 0 {
        return Err(MetalError::ShapeMismatch(format!(
            "split_qg_per_head_f32: n_q={n_q}, head_dim={head_dim}"
        )));
    }
    let pipeline = backend.pipeline(
        "split_qg_per_head_f32",
        SPLIT_QG_PER_HEAD_F32_SHADER,
        "split_qg_per_head_f32",
    )?;
    let dims = [n_q as u32, head_dim as u32];
    backend.with_encoder(|encoder| {
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(qg_buf), 0);
        encoder.set_buffer(1, Some(q_buf), 0);
        encoder.set_buffer(2, Some(gate_buf), 0);
        encoder.set_bytes(3, 8, dims.as_ptr() as *const std::ffi::c_void);
        let tg = MTLSize::new(32, 1, 1);
        let grid = MTLSize::new(head_dim as u64, n_q as u64, 1);
        encoder.dispatch_threads(grid, tg);
    });
    Ok(())
}

// T162 phase 9a — batched variant for B-token prefill of Qwen3Next attn block.
// Each batch entry has `qg [n_q, 2 * head_dim]` and produces `q [n_q, head_dim]`
// and `gate [n_q, head_dim]`. The 3D grid (head_dim, n_q, B) lets each thread
// handle a single (head, dim, batch) triplet — embarrassingly parallel.
const SPLIT_QG_PER_HEAD_BATCHED_F32_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void split_qg_per_head_batched_f32(
    device const float* qg       [[buffer(0)]],   // [B, n_q, 2 * head_dim]
    device float*       q        [[buffer(1)]],   // [B, n_q, head_dim] out
    device float*       gate     [[buffer(2)]],   // [B, n_q, head_dim] out
    constant uint3&     dims     [[buffer(3)]],   // (n_q, head_dim, B)
    uint3               gid      [[thread_position_in_grid]]
) {
    uint n_q = dims.x;
    uint head_dim = dims.y;
    uint B = dims.z;
    uint i = gid.x;
    uint h = gid.y;
    uint b = gid.z;
    if (h >= n_q || i >= head_dim || b >= B) return;
    uint per_batch_qg = n_q * 2u * head_dim;
    uint per_batch_q  = n_q * head_dim;
    uint src_off = b * per_batch_qg + h * 2u * head_dim;
    uint dst_off = b * per_batch_q  + h * head_dim;
    q[dst_off + i]    = qg[src_off + i];
    gate[dst_off + i] = qg[src_off + head_dim + i];
}
"#;

/// T162 phase 9a — batched per-head split for Qwen3Next attn prefill.
/// Equivalent to calling `split_qg_per_head_f32` once per batch entry,
/// but in a single dispatch (no per-token CPU overhead).
///
/// Buffers : `qg_batched` is `[B, n_q, 2 * head_dim]` row-major,
/// `q_batched` and `gate_batched` are `[B, n_q, head_dim]`.
pub fn split_qg_per_head_batched_f32(
    backend: &MetalBackend,
    qg_buf: &Buffer,
    q_buf: &Buffer,
    gate_buf: &Buffer,
    n_q: usize,
    head_dim: usize,
    b: usize,
) -> Result<(), MetalError> {
    if n_q == 0 || head_dim == 0 || b == 0 {
        return Err(MetalError::ShapeMismatch(format!(
            "split_qg_per_head_batched_f32: n_q={n_q}, head_dim={head_dim}, B={b}"
        )));
    }
    let pipeline = backend.pipeline(
        "split_qg_per_head_batched_f32",
        SPLIT_QG_PER_HEAD_BATCHED_F32_SHADER,
        "split_qg_per_head_batched_f32",
    )?;
    let dims = [n_q as u32, head_dim as u32, b as u32];
    backend.with_encoder(|encoder| {
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(qg_buf), 0);
        encoder.set_buffer(1, Some(q_buf), 0);
        encoder.set_buffer(2, Some(gate_buf), 0);
        encoder.set_bytes(3, 12, dims.as_ptr() as *const std::ffi::c_void);
        let tg = MTLSize::new(32, 1, 1);
        let grid = MTLSize::new(head_dim as u64, n_q as u64, b as u64);
        encoder.dispatch_threads(grid, tg);
    });
    Ok(())
}

// T151 — Split a contiguous `[q_len + k_len + v_len]` source buffer into
// three destination buffers on GPU. Replaces the CPU `ptr::copy_nonoverlapping`
// path in `ssm_block_forward` which required a `backend.drain()` to make the
// conv1d output visible to the CPU. With this kernel the SSM block stays
// fully on GPU, eliminating 32 drains/token on Qwen3.6-27B (and ~64 on the
// 35B-A3B's MoE+SSM path).
const SPLIT_QKV_F32_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void split_qkv_f32(
    device const float* src    [[buffer(0)]],   // [q_len + k_len + v_len]
    device float*       q_dst  [[buffer(1)]],   // [q_len]
    device float*       k_dst  [[buffer(2)]],   // [k_len]
    device float*       v_dst  [[buffer(3)]],   // [v_len]
    constant uint3&     dims   [[buffer(4)]],   // (q_len, k_len, v_len)
    uint                gid    [[thread_position_in_grid]]
) {
    uint q_len = dims.x;
    uint k_len = dims.y;
    uint v_len = dims.z;
    uint total = q_len + k_len + v_len;
    if (gid >= total) return;
    float v = src[gid];
    if (gid < q_len) {
        q_dst[gid] = v;
    } else if (gid < q_len + k_len) {
        k_dst[gid - q_len] = v;
    } else {
        v_dst[gid - q_len - k_len] = v;
    }
}
"#;

/// T151 — three-way split of a `[q_len + k_len + v_len]` source buffer
/// into separate `q`, `k`, `v` destination buffers on GPU. Eliminates
/// the CPU `drain + memcpy` pass formerly used in `ssm_block_forward`.
pub fn split_qkv_f32(
    backend: &MetalBackend,
    src_buf: &Buffer,
    q_buf: &Buffer,
    k_buf: &Buffer,
    v_buf: &Buffer,
    q_len: usize,
    k_len: usize,
    v_len: usize,
) -> Result<(), MetalError> {
    if q_len == 0 || k_len == 0 || v_len == 0 {
        return Err(MetalError::ShapeMismatch(format!(
            "split_qkv_f32: q_len={q_len}, k_len={k_len}, v_len={v_len} all must be > 0"
        )));
    }
    let pipeline = backend.pipeline("split_qkv_f32", SPLIT_QKV_F32_SHADER, "split_qkv_f32")?;
    let dims = [q_len as u32, k_len as u32, v_len as u32];
    let total = (q_len + k_len + v_len) as u64;
    backend.with_encoder(|encoder| {
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(src_buf), 0);
        encoder.set_buffer(1, Some(q_buf), 0);
        encoder.set_buffer(2, Some(k_buf), 0);
        encoder.set_buffer(3, Some(v_buf), 0);
        encoder.set_bytes(4, 12, dims.as_ptr() as *const std::ffi::c_void);
        let tg = MTLSize::new(64, 1, 1);
        let grid = MTLSize::new(total, 1, 1);
        encoder.dispatch_threads(grid, tg);
    });
    Ok(())
}

// T147a — `acc += w * src` in place. Used by the MoE FFN forward to
// accumulate weighted per-expert outputs without draining between
// experts. Eliminates one drain per active expert (8 drains × 40 MoE
// layers = 320 drains/token saved on the 35B-A3B).
const WEIGHTED_ADD_INPLACE_F32_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void weighted_add_inplace_f32(
    device float*        acc     [[buffer(0)]],
    device const float*  src     [[buffer(1)]],
    constant float&      w       [[buffer(2)]],
    constant uint&       n       [[buffer(3)]],
    uint                 gid     [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    acc[gid] = acc[gid] + w * src[gid];
}
"#;

/// `acc[i] += weight * src[i]` for `i` in `0..n`. Used by MoE FFN to
/// accumulate per-expert outputs weighted by the top-K softmax probs.
pub fn weighted_add_inplace_f32(
    backend: &MetalBackend,
    acc_buf: &Buffer,
    src_buf: &Buffer,
    weight: f32,
    n: usize,
) -> Result<(), MetalError> {
    if n == 0 {
        return Err(MetalError::ShapeMismatch(
            "weighted_add_inplace_f32: n must be > 0".to_string(),
        ));
    }
    let pipeline = backend.pipeline(
        "weighted_add_inplace_f32",
        WEIGHTED_ADD_INPLACE_F32_SHADER,
        "weighted_add_inplace_f32",
    )?;
    let n_u = n as u32;
    backend.with_encoder(|encoder| {
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(acc_buf), 0);
        encoder.set_buffer(1, Some(src_buf), 0);
        encoder.set_bytes(2, 4, &weight as *const f32 as *const std::ffi::c_void);
        encoder.set_bytes(3, 4, &n_u as *const u32 as *const std::ffi::c_void);
        let tg = MTLSize::new(64, 1, 1);
        let grid = MTLSize::new(n as u64, 1, 1);
        encoder.dispatch_threads(grid, tg);
    });
    Ok(())
}

// T147a — Zero a Metal buffer in-place via GPU. Used to reset the MoE
// accumulator at the start of each layer's MoE FFN. Avoids a drain that
// would otherwise be needed for a CPU memset.
const ZERO_F32_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void zero_f32(
    device float*  buf  [[buffer(0)]],
    constant uint& n    [[buffer(1)]],
    uint           gid  [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    buf[gid] = 0.0;
}
"#;

/// Zero `n` f32 elements at the start of `buf` on the GPU. Used by MoE.
pub fn zero_f32(backend: &MetalBackend, buf: &Buffer, n: usize) -> Result<(), MetalError> {
    if n == 0 {
        return Err(MetalError::ShapeMismatch(
            "zero_f32: n must be > 0".to_string(),
        ));
    }
    let pipeline = backend.pipeline("zero_f32", ZERO_F32_SHADER, "zero_f32")?;
    let n_u = n as u32;
    backend.with_encoder(|encoder| {
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(buf), 0);
        encoder.set_bytes(1, 4, &n_u as *const u32 as *const std::ffi::c_void);
        let tg = MTLSize::new(64, 1, 1);
        let grid = MTLSize::new(n as u64, 1, 1);
        encoder.dispatch_threads(grid, tg);
    });
    Ok(())
}

// T144c — Fused `out *= sigmoid(gate)` in place. Used by Qwen3Next
// attention to apply the per-head gate to the post-GQA output before the
// W_O projection. Eliminates one drain + CPU pass per attention layer
// (16 layers × 1 drain ≈ 1.5 ms per token saved on the 27B).
const SIGMOID_MUL_INPLACE_F32_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void sigmoid_mul_inplace_f32(
    device float*        x       [[buffer(0)]],   // out, in place
    device const float*  gate    [[buffer(1)]],
    constant uint&       n       [[buffer(2)]],
    uint                 gid     [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    float g = gate[gid];
    float sig = 1.0 / (1.0 + exp(-g));
    x[gid] = x[gid] * sig;
}
"#;

/// `x[i] *= sigmoid(gate[i])` for `i` in `0..n`. Both buffers must be
/// `n` f32s long. Used by Qwen3Next attention's per-head gating.
pub fn sigmoid_mul_inplace_f32(
    backend: &MetalBackend,
    x_buf: &Buffer,
    gate_buf: &Buffer,
    n: usize,
) -> Result<(), MetalError> {
    if n == 0 {
        return Err(MetalError::ShapeMismatch(
            "sigmoid_mul_inplace_f32: n must be > 0".to_string(),
        ));
    }
    let pipeline = backend.pipeline(
        "sigmoid_mul_inplace_f32",
        SIGMOID_MUL_INPLACE_F32_SHADER,
        "sigmoid_mul_inplace_f32",
    )?;
    let n_u = n as u32;
    backend.with_encoder(|encoder| {
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(x_buf), 0);
        encoder.set_buffer(1, Some(gate_buf), 0);
        encoder.set_bytes(2, 4, &n_u as *const u32 as *const std::ffi::c_void);
        let tg = MTLSize::new(64, 1, 1);
        let grid = MTLSize::new(n as u64, 1, 1);
        encoder.dispatch_threads(grid, tg);
    });
    Ok(())
}

// T162 phase 9a — batched sigmoid_mul_inplace for B-token prefill.
// Same as sigmoid_mul_inplace but operates on `[B, n]` row-major buffers.
const SIGMOID_MUL_INPLACE_BATCHED_F32_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void sigmoid_mul_inplace_batched_f32(
    device float*        x       [[buffer(0)]],   // [B, n] in/out
    device const float*  gate    [[buffer(1)]],   // [B, n]
    constant uint2&      dims    [[buffer(2)]],   // (n, B)
    uint2                gid     [[thread_position_in_grid]]
) {
    uint n = dims.x;
    uint B = dims.y;
    uint i = gid.x;
    uint b = gid.y;
    if (i >= n || b >= B) return;
    uint off = b * n + i;
    float g = gate[off];
    float sig = 1.0 / (1.0 + exp(-g));
    x[off] = x[off] * sig;
}
"#;

/// T162 phase 9a — batched `x[bi, i] *= sigmoid(gate[bi, i])`.
/// Both buffers are `[B, n]` row-major. Single dispatch.
pub fn sigmoid_mul_inplace_batched_f32(
    backend: &MetalBackend,
    x_buf: &Buffer,
    gate_buf: &Buffer,
    n: usize,
    b: usize,
) -> Result<(), MetalError> {
    if n == 0 || b == 0 {
        return Err(MetalError::ShapeMismatch(format!(
            "sigmoid_mul_inplace_batched_f32: n={n}, B={b}"
        )));
    }
    let pipeline = backend.pipeline(
        "sigmoid_mul_inplace_batched_f32",
        SIGMOID_MUL_INPLACE_BATCHED_F32_SHADER,
        "sigmoid_mul_inplace_batched_f32",
    )?;
    let dims = [n as u32, b as u32];
    backend.with_encoder(|encoder| {
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(x_buf), 0);
        encoder.set_buffer(1, Some(gate_buf), 0);
        encoder.set_bytes(2, 8, dims.as_ptr() as *const std::ffi::c_void);
        let tg = MTLSize::new(64, 1, 1);
        let grid = MTLSize::new(n as u64, b as u64, 1);
        encoder.dispatch_threads(grid, tg);
    });
    Ok(())
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

    /// T158 phase 1b — validation Metal Q3_K kernel vs CPU dequant + naive matmul.
    #[test]
    #[cfg(feature = "gpu-tests")]
    fn sgemv_q3_k_matches_cpu_dequant_reference() {
        use rustorch_gguf::dequant::dequantize_block_chunk;
        use rustorch_gguf::tensor::GgmlType;

        let backend = metal_backend();
        if !backend.supports_metal3() {
            eprintln!(
                "[sgemv_q3_k] skipping: device does not support Metal3 ({})",
                backend.adapter_name()
            );
            return;
        }

        // K = 256 (1 super-block per row), N = 32 (1 threadgroup of 32 threads
        // per row, 32 rows total = 32 threadgroups in dispatch).
        let k = 256_usize;
        let n = 32_usize;
        let blocks_per_row = k / 256;

        // Build deterministic Q3_K bytes : N rows × 110 bytes/super-block.
        // We hand-craft each block with varying d, scales, qs, hmask so the
        // dequant exercises non-trivial paths.
        let mut w_bytes = vec![0u8; n * blocks_per_row * 110];
        for nrow in 0..n {
            let off = nrow * blocks_per_row * 110;
            // hmask : alternate 0xAA / 0x55 patterns so half the weights have
            // h_bit=1 (offset 0) and half h_bit=0 (offset -4).
            for i in 0..32 {
                w_bytes[off + i] = if (nrow + i) % 2 == 0 { 0xAA } else { 0x55 };
            }
            // qs : write incrementing low-2-bit pairs so we get diverse q_lo values.
            for i in 0..64 {
                w_bytes[off + 32 + i] = ((i as u8) & 0xC0)
                    | (((i as u8) << 2) & 0x30)
                    | (((i as u8) << 4) & 0x0C)
                    | (((i as u8) << 6) & 0x03);
            }
            // scales : non-zero packed pattern so unpacking covers all 16 sub-blocks.
            for i in 0..12 {
                w_bytes[off + 96 + i] = (0x40_u8.wrapping_add((nrow as u8) ^ (i as u8))) | 0x10;
            }
            // d = nrow + 1 as half-float (small positive value)
            let d_val = (nrow as f32 + 1.0) * 0.01;
            let d_h = half::f16::from_f32(d_val).to_le_bytes();
            w_bytes[off + 108] = d_h[0];
            w_bytes[off + 109] = d_h[1];
        }

        // CPU reference : dequantize each row to f32 then matmul with x.
        let x = det_vec(k, 1.7);
        let mut y_ref = vec![0.0_f32; n];
        for (nrow, y_slot) in y_ref.iter_mut().enumerate() {
            let row_off = nrow * blocks_per_row * 110;
            let row_bytes = &w_bytes[row_off..row_off + blocks_per_row * 110];
            let mut row_w = vec![0.0_f32; k];
            dequantize_block_chunk(GgmlType::Q3_K, row_bytes, &mut row_w).unwrap();
            let mut s = 0.0_f32;
            for kk in 0..k {
                s += x[kk] * row_w[kk];
            }
            *y_slot = s;
        }

        // Metal kernel.
        let x_buf = backend.alloc_shared(k * 4).unwrap();
        let w_buf = backend.alloc_shared(w_bytes.len()).unwrap();
        let y_buf = backend.alloc_shared(n * 4).unwrap();
        unsafe {
            std::ptr::copy_nonoverlapping(x.as_ptr(), x_buf.contents() as *mut f32, k);
            std::ptr::copy_nonoverlapping(
                w_bytes.as_ptr(),
                w_buf.contents() as *mut u8,
                w_bytes.len(),
            );
        }
        sgemv_q3_k_f32_lcpp_nsg1_into(backend, &x_buf, &w_buf, &y_buf, k, n).unwrap();
        backend.drain();
        let mut y_metal = vec![0.0_f32; n];
        unsafe {
            std::ptr::copy_nonoverlapping(y_buf.contents() as *const f32, y_metal.as_mut_ptr(), n);
        }

        for i in 0..n {
            let abs_err = (y_ref[i] - y_metal[i]).abs();
            let denom = y_ref[i].abs().max(1e-4);
            let rel = abs_err / denom;
            assert!(
                rel < 1e-3,
                "Q3_K mismatch at row {i}: ref={} metal={} (rel err {:.3e})",
                y_ref[i],
                y_metal[i],
                rel
            );
        }
    }

    /// T160 — parité numérique NSG=2 (ix-stripped) vs NSG=1 baseline.
    ///
    /// Exerce ix-stripping (K=1024 → 4 super-blocks, chaque ix∈[0..4) en
    /// visite exactement 1) et la pair sb_first/sb_second pour les 8 tids
    /// possibles. Tolérance 1e-4 (les sommes d'arrondi peuvent micro-diverger
    /// entre les 2 ordres d'accumulation simdgroup, mais doivent rester très
    /// proches).
    #[test]
    #[cfg(feature = "gpu-tests")]
    fn sgemv_q3_k_nsg2_matches_nsg1() {
        let backend = metal_backend();
        if !backend.supports_metal3() {
            eprintln!(
                "[sgemv_q3_k_nsg2] skipping: device does not support Metal3 ({})",
                backend.adapter_name()
            );
            return;
        }

        // K = 1024 (4 super-blocks) pour exercer les 4 ix-stripes,
        // N = 64 (32 TGs NSG=2, donc 32 simdgroups en flight = 64 rows).
        let k = 1024_usize;
        let n = 64_usize;
        let blocks_per_row = k / 256;

        // Build deterministic Q3_K bytes : N rows × blocks_per_row × 110 bytes.
        let mut w_bytes = vec![0u8; n * blocks_per_row * 110];
        for nrow in 0..n {
            for ib in 0..blocks_per_row {
                let off = (nrow * blocks_per_row + ib) * 110;
                // hmask : alternance contrôlée par (nrow, ib, i)
                for i in 0..32 {
                    w_bytes[off + i] = if (nrow + ib + i) % 2 == 0 { 0xAA } else { 0x55 };
                }
                // qs : pattern incrémental pour couvrir tous les q_lo possibles.
                for i in 0..64 {
                    w_bytes[off + 32 + i] = ((i as u8 + ib as u8) & 0xC0)
                        | ((((i as u8) + (ib as u8)) << 2) & 0x30)
                        | ((((i as u8) + (ib as u8)) << 4) & 0x0C)
                        | ((((i as u8) + (ib as u8)) << 6) & 0x03);
                }
                // scales : 12 bytes packés non triviaux.
                for i in 0..12 {
                    w_bytes[off + 96 + i] =
                        (0x40_u8.wrapping_add((nrow as u8) ^ (i as u8) ^ (ib as u8))) | 0x10;
                }
                // d : valeur dépendante de (nrow, ib) en half-float.
                let d_val = ((nrow as f32 + 1.0) * 0.01) + (ib as f32) * 0.001;
                let d_h = half::f16::from_f32(d_val).to_le_bytes();
                w_bytes[off + 108] = d_h[0];
                w_bytes[off + 109] = d_h[1];
            }
        }

        let x = det_vec(k, 2.3);

        // Buffers communs.
        let x_buf = backend.alloc_shared(k * 4).unwrap();
        let w_buf = backend.alloc_shared(w_bytes.len()).unwrap();
        let y1_buf = backend.alloc_shared(n * 4).unwrap();
        let y2_buf = backend.alloc_shared(n * 4).unwrap();
        unsafe {
            std::ptr::copy_nonoverlapping(x.as_ptr(), x_buf.contents() as *mut f32, k);
            std::ptr::copy_nonoverlapping(
                w_bytes.as_ptr(),
                w_buf.contents() as *mut u8,
                w_bytes.len(),
            );
        }

        // NSG=1 baseline.
        sgemv_q3_k_f32_lcpp_nsg1_into(backend, &x_buf, &w_buf, &y1_buf, k, n).unwrap();
        backend.drain();
        let mut y1 = vec![0.0_f32; n];
        unsafe {
            std::ptr::copy_nonoverlapping(y1_buf.contents() as *const f32, y1.as_mut_ptr(), n);
        }

        // NSG=2 (ix-stripped).
        sgemv_q3_k_f32_lcpp_nsg2_into(backend, &x_buf, &w_buf, &y2_buf, k, n).unwrap();
        backend.drain();
        let mut y2 = vec![0.0_f32; n];
        unsafe {
            std::ptr::copy_nonoverlapping(y2_buf.contents() as *const f32, y2.as_mut_ptr(), n);
        }

        for i in 0..n {
            let abs_err = (y1[i] - y2[i]).abs();
            let denom = y1[i].abs().max(1e-4);
            let rel = abs_err / denom;
            assert!(
                rel < 1e-4,
                "Q3_K NSG=2 ≠ NSG=1 at row {i}: nsg1={} nsg2={} (rel err {:.3e})",
                y1[i],
                y2[i],
                rel
            );
        }
    }

    /// T154-fast — parité numérique de `delta_net_step_with_l2_f32` vs
    /// (l2_norm_per_head_f32 × 2 + delta_net_step_f32) legacy.
    ///
    /// Le state SSM ET l'output `out` doivent être numériquement identiques
    /// aux deux branches dans la limite de la tolérance flottante (rsqrt +
    /// simd_sum peuvent micro-diverger).
    #[test]
    #[cfg(feature = "gpu-tests")]
    fn delta_net_step_with_l2_matches_legacy() {
        let backend = metal_backend();
        if !backend.supports_metal3() {
            eprintln!(
                "[delta_net_step_with_l2] skipping: device does not support Metal3 ({})",
                backend.adapter_name()
            );
            return;
        }

        // Configuration cohérente avec Qwen3.6-27B SSM : head_dim=128,
        // n_k_heads=16, n_v_heads=48 (repeat=3).
        let head_dim = 128_usize;
        let n_k_heads = 16_usize;
        let n_v_heads = 48_usize;
        let eps = 1e-5_f32;

        let key_dim = n_k_heads * head_dim;
        let value_dim = n_v_heads * head_dim;
        let state_size = n_v_heads * head_dim * head_dim;

        // Inputs déterministes.
        let q = det_vec(key_dim, 0.7);
        let k = det_vec(key_dim, 1.3);
        let v = det_vec(value_dim, 2.1);
        let gate_h = det_vec(n_v_heads, 0.05);
        let beta = det_vec(n_v_heads, 0.5);
        let state_init = det_vec(state_size, 0.001);

        // ===== Path A : legacy (l2_norm × 2 + delta_net_step) =====
        let q_a_buf = backend.alloc_shared(key_dim * 4).unwrap();
        let k_a_buf = backend.alloc_shared(key_dim * 4).unwrap();
        let v_a_buf = backend.alloc_shared(value_dim * 4).unwrap();
        let gate_a_buf = backend.alloc_shared(n_v_heads * 4).unwrap();
        let beta_a_buf = backend.alloc_shared(n_v_heads * 4).unwrap();
        let state_a_buf = backend.alloc_shared(state_size * 4).unwrap();
        let out_a_buf = backend.alloc_shared(value_dim * 4).unwrap();
        unsafe {
            std::ptr::copy_nonoverlapping(q.as_ptr(), q_a_buf.contents() as *mut f32, key_dim);
            std::ptr::copy_nonoverlapping(k.as_ptr(), k_a_buf.contents() as *mut f32, key_dim);
            std::ptr::copy_nonoverlapping(v.as_ptr(), v_a_buf.contents() as *mut f32, value_dim);
            std::ptr::copy_nonoverlapping(
                gate_h.as_ptr(),
                gate_a_buf.contents() as *mut f32,
                n_v_heads,
            );
            std::ptr::copy_nonoverlapping(
                beta.as_ptr(),
                beta_a_buf.contents() as *mut f32,
                n_v_heads,
            );
            std::ptr::copy_nonoverlapping(
                state_init.as_ptr(),
                state_a_buf.contents() as *mut f32,
                state_size,
            );
        }
        l2_norm_per_head_f32(backend, &q_a_buf, n_k_heads, head_dim, eps).unwrap();
        l2_norm_per_head_f32(backend, &k_a_buf, n_k_heads, head_dim, eps).unwrap();
        delta_net_step_f32(
            backend,
            &q_a_buf,
            &k_a_buf,
            &v_a_buf,
            &gate_a_buf,
            &beta_a_buf,
            &state_a_buf,
            &out_a_buf,
            n_v_heads,
            head_dim,
            n_k_heads,
        )
        .unwrap();
        backend.drain();
        let mut out_a = vec![0.0_f32; value_dim];
        let mut state_a = vec![0.0_f32; state_size];
        unsafe {
            std::ptr::copy_nonoverlapping(
                out_a_buf.contents() as *const f32,
                out_a.as_mut_ptr(),
                value_dim,
            );
            std::ptr::copy_nonoverlapping(
                state_a_buf.contents() as *const f32,
                state_a.as_mut_ptr(),
                state_size,
            );
        }

        // ===== Path B : T154-fast (delta_net_step_with_l2_f32) =====
        let q_b_buf = backend.alloc_shared(key_dim * 4).unwrap();
        let k_b_buf = backend.alloc_shared(key_dim * 4).unwrap();
        let v_b_buf = backend.alloc_shared(value_dim * 4).unwrap();
        let gate_b_buf = backend.alloc_shared(n_v_heads * 4).unwrap();
        let beta_b_buf = backend.alloc_shared(n_v_heads * 4).unwrap();
        let state_b_buf = backend.alloc_shared(state_size * 4).unwrap();
        let out_b_buf = backend.alloc_shared(value_dim * 4).unwrap();
        unsafe {
            std::ptr::copy_nonoverlapping(q.as_ptr(), q_b_buf.contents() as *mut f32, key_dim);
            std::ptr::copy_nonoverlapping(k.as_ptr(), k_b_buf.contents() as *mut f32, key_dim);
            std::ptr::copy_nonoverlapping(v.as_ptr(), v_b_buf.contents() as *mut f32, value_dim);
            std::ptr::copy_nonoverlapping(
                gate_h.as_ptr(),
                gate_b_buf.contents() as *mut f32,
                n_v_heads,
            );
            std::ptr::copy_nonoverlapping(
                beta.as_ptr(),
                beta_b_buf.contents() as *mut f32,
                n_v_heads,
            );
            std::ptr::copy_nonoverlapping(
                state_init.as_ptr(),
                state_b_buf.contents() as *mut f32,
                state_size,
            );
        }
        delta_net_step_with_l2_f32(
            backend,
            &q_b_buf,
            &k_b_buf,
            &v_b_buf,
            &gate_b_buf,
            &beta_b_buf,
            &state_b_buf,
            &out_b_buf,
            n_v_heads,
            head_dim,
            n_k_heads,
            eps,
        )
        .unwrap();
        backend.drain();
        let mut out_b = vec![0.0_f32; value_dim];
        let mut state_b = vec![0.0_f32; state_size];
        unsafe {
            std::ptr::copy_nonoverlapping(
                out_b_buf.contents() as *const f32,
                out_b.as_mut_ptr(),
                value_dim,
            );
            std::ptr::copy_nonoverlapping(
                state_b_buf.contents() as *const f32,
                state_b.as_mut_ptr(),
                state_size,
            );
        }

        // Compare out.
        for i in 0..value_dim {
            let abs_err = (out_a[i] - out_b[i]).abs();
            let denom = out_a[i].abs().max(1e-4);
            let rel = abs_err / denom;
            assert!(
                rel < 1e-3,
                "out mismatch at {i}: legacy={} fused={} rel={:.3e}",
                out_a[i],
                out_b[i],
                rel
            );
        }

        // Compare state (échantillon : strides 257 pour couvrir diverses heads/rows/cols).
        for i in (0..state_size).step_by(257) {
            let abs_err = (state_a[i] - state_b[i]).abs();
            let denom = state_a[i].abs().max(1e-4);
            let rel = abs_err / denom;
            assert!(
                rel < 1e-3,
                "state mismatch at {i}: legacy={} fused={} rel={:.3e}",
                state_a[i],
                state_b[i],
                rel
            );
        }
    }

    /// T162 phase 3 — Q4_K SGEMM tiles 64×64 multi-warp vs CPU dequant + naive matmul.
    ///
    /// Validation correctness du kernel multi-warp avec 4 simdgroups par TG.
    /// PASS gate de la phase 3 (intégration prefill).
    #[test]
    #[cfg(feature = "gpu-tests")]
    fn sgemm_q4_k_f32_simdgroup_matrix_64_matches_cpu() {
        use rustorch_gguf::dequant::dequant_q4_k;

        let backend = metal_backend();
        if !backend.supports_metal3() {
            eprintln!("[sgemm_q4_k_simdmat_64] skipping: no Metal3");
            return;
        }

        // Shape minimale : M=N=64, K=512 (2 super-blocks). Exerce le multi-warp,
        // les 16 C fragments par simdgroup, et le K-loop multi-superblock.
        let m = 64_usize;
        let n = 64_usize;
        let k = 512_usize;
        let blocks_per_row = k / 256;

        // Build deterministic Q4_K bytes (varied scales/qs).
        let mut w_bytes = vec![0u8; n * blocks_per_row * 144];
        for nrow in 0..n {
            for ib in 0..blocks_per_row {
                let off = (nrow * blocks_per_row + ib) * 144;
                let d_val = ((nrow as f32 + 1.0) * 0.005) + (ib as f32) * 0.001;
                let dmin_val = ((nrow as f32) * 0.002) + (ib as f32) * 0.0005;
                let d_h = half::f16::from_f32(d_val).to_le_bytes();
                let dmin_h = half::f16::from_f32(dmin_val).to_le_bytes();
                w_bytes[off] = d_h[0];
                w_bytes[off + 1] = d_h[1];
                w_bytes[off + 2] = dmin_h[0];
                w_bytes[off + 3] = dmin_h[1];
                for i in 0..12 {
                    w_bytes[off + 4 + i] =
                        (0x12_u8.wrapping_add((nrow as u8) ^ (i as u8) ^ (ib as u8))) | 0x05;
                }
                for i in 0..128 {
                    w_bytes[off + 16 + i] =
                        ((nrow as u8) ^ (i as u8) ^ (ib as u8)).wrapping_add(0x29);
                }
            }
        }

        let a = det_vec(m * k, 1.5);
        let mut w_f32 = vec![0.0_f32; n * k];
        for nrow in 0..n {
            let row_off = nrow * blocks_per_row * 144;
            let row_bytes = &w_bytes[row_off..row_off + blocks_per_row * 144];
            let row_dst = &mut w_f32[nrow * k..(nrow + 1) * k];
            dequant_q4_k(row_bytes, row_dst).unwrap();
        }

        let mut c_ref = vec![0.0_f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut s = 0.0_f32;
                for l in 0..k {
                    s += a[i * k + l] * w_f32[j * k + l];
                }
                c_ref[i * n + j] = s;
            }
        }

        let a_buf = backend.alloc_shared(m * k * 4).unwrap();
        let w_buf = backend.alloc_shared(w_bytes.len()).unwrap();
        let c_buf = backend.alloc_shared(m * n * 4).unwrap();
        unsafe {
            std::ptr::copy_nonoverlapping(a.as_ptr(), a_buf.contents() as *mut f32, m * k);
            std::ptr::copy_nonoverlapping(
                w_bytes.as_ptr(),
                w_buf.contents() as *mut u8,
                w_bytes.len(),
            );
        }
        sgemm_q4_k_f32_simdgroup_matrix_64_into(backend, &a_buf, &w_buf, &c_buf, m, n, k).unwrap();
        backend.drain();

        let mut c_metal = vec![0.0_f32; m * n];
        unsafe {
            std::ptr::copy_nonoverlapping(
                c_buf.contents() as *const f32,
                c_metal.as_mut_ptr(),
                m * n,
            );
        }

        for i in 0..(m * n) {
            let abs_err = (c_ref[i] - c_metal[i]).abs();
            let denom = c_ref[i].abs().max(1e-3);
            let rel = abs_err / denom;
            assert!(
                rel < 1e-2,
                "Q4_K SGEMM 64×64 mismatch at {i}: ref={} metal={} (rel {:.3e})",
                c_ref[i],
                c_metal[i],
                rel
            );
        }
    }

    /// T162 phase 9h — Q4_K SGEMM port complet llama.cpp::kernel_mul_mm vs CPU.
    /// Tile 32×64 (M_tok × N_out), 4 SG/TG, swizzled SHM 8x8 blocks,
    /// outer product mc[i] = mb[i/4] @ ma[i%4].
    #[test]
    #[cfg(feature = "gpu-tests")]
    fn sgemm_q4_k_f32_lcpp_ported_matches_cpu() {
        use rustorch_gguf::dequant::dequant_q4_k;

        let backend = metal_backend();
        if !backend.supports_metal3() {
            eprintln!("[sgemm_q4_k_lcpp_ported] skipping: no Metal3");
            return;
        }

        // Shape : M=32 (1 m-tile), N=64 (1 n-tile), K=512 (2 super-blocks).
        let m = 32_usize;
        let n = 64_usize;
        let k = 512_usize;
        let blocks_per_row = k / 256;

        let mut w_bytes = vec![0u8; n * blocks_per_row * 144];
        for nrow in 0..n {
            for ib in 0..blocks_per_row {
                let off = (nrow * blocks_per_row + ib) * 144;
                let d_val = ((nrow as f32 + 1.0) * 0.005) + (ib as f32) * 0.001;
                let dmin_val = ((nrow as f32) * 0.002) + (ib as f32) * 0.0005;
                let d_h = half::f16::from_f32(d_val).to_le_bytes();
                let dmin_h = half::f16::from_f32(dmin_val).to_le_bytes();
                w_bytes[off] = d_h[0];
                w_bytes[off + 1] = d_h[1];
                w_bytes[off + 2] = dmin_h[0];
                w_bytes[off + 3] = dmin_h[1];
                for i in 0..12 {
                    w_bytes[off + 4 + i] =
                        (0x12_u8.wrapping_add((nrow as u8) ^ (i as u8) ^ (ib as u8))) | 0x05;
                }
                for i in 0..128 {
                    w_bytes[off + 16 + i] =
                        ((nrow as u8) ^ (i as u8) ^ (ib as u8)).wrapping_add(0x29);
                }
            }
        }

        let a = det_vec(m * k, 1.5);
        let mut w_f32 = vec![0.0_f32; n * k];
        for nrow in 0..n {
            let row_off = nrow * blocks_per_row * 144;
            let row_bytes = &w_bytes[row_off..row_off + blocks_per_row * 144];
            let row_dst = &mut w_f32[nrow * k..(nrow + 1) * k];
            dequant_q4_k(row_bytes, row_dst).unwrap();
        }

        let mut c_ref = vec![0.0_f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut s = 0.0_f32;
                for l in 0..k {
                    s += a[i * k + l] * w_f32[j * k + l];
                }
                c_ref[i * n + j] = s;
            }
        }

        let a_buf = backend.alloc_shared(m * k * 4).unwrap();
        let w_buf = backend.alloc_shared(w_bytes.len()).unwrap();
        let c_buf = backend.alloc_shared(m * n * 4).unwrap();
        unsafe {
            std::ptr::copy_nonoverlapping(a.as_ptr(), a_buf.contents() as *mut f32, m * k);
            std::ptr::copy_nonoverlapping(
                w_bytes.as_ptr(),
                w_buf.contents() as *mut u8,
                w_bytes.len(),
            );
        }
        sgemm_q4_k_f32_lcpp_ported_into(backend, &a_buf, &w_buf, &c_buf, m, n, k).unwrap();
        backend.drain();

        let mut c_metal = vec![0.0_f32; m * n];
        unsafe {
            std::ptr::copy_nonoverlapping(
                c_buf.contents() as *const f32,
                c_metal.as_mut_ptr(),
                m * n,
            );
        }

        for i in 0..(m * n) {
            let abs_err = (c_ref[i] - c_metal[i]).abs();
            let denom = c_ref[i].abs().max(1e-3);
            let rel = abs_err / denom;
            assert!(
                rel < 1e-2,
                "Q4_K SGEMM lcpp_ported mismatch at {i}: ref={} metal={} (rel {:.3e})",
                c_ref[i],
                c_metal[i],
                rel
            );
        }
    }

    /// T162 phase 2 — Bench Q4_K SGEMM simdgroup_matrix vs sgemv_q4_k loop.
    ///
    /// Le path actuel rustorch pour prefill : `sgemv_q4_k_f32_lcpp_nsg2` appelé
    /// M fois (1 par token). Le nouveau path : 1 seul kernel SGEMM Q4_K avec
    /// simdgroup_matrix.
    #[test]
    #[cfg(feature = "gpu-tests")]
    #[ignore]
    fn sgemm_q4_k_f32_simdgroup_matrix_bench() {
        let backend = metal_backend();
        if !backend.supports_metal3() {
            return;
        }

        // Shape représentative d'un slice prefill 14B-style :
        // K = 5120 hidden_dim, N = 5120 (Q proj) ou 14336 (FFN).
        // BUT : K must be multiple of 256, N multiple of 8.
        for &(m, n_dim, k) in [
            (8, 1024, 1024),
            (16, 1024, 1024),
            (32, 1024, 1024),
            (64, 1024, 1024),
            (128, 1024, 1024),
            (32, 5120, 5120),
            // 14B FFN realistic shapes (prefill M=24 ≈ 19-token prompt rounded up).
            (24, 5120, 5120),  // Q/K/V/O proj : K=hidden, N=hidden
            (24, 14336, 5120), // FFN gate/up : N=intermediate, K=hidden
            (24, 5120, 14336), // FFN down : N=hidden, K=intermediate
            (64, 5120, 5120),
            (64, 14336, 5120),
        ]
        .iter()
        {
            let blocks_per_row = k / 256;
            let mut w_bytes = vec![0u8; n_dim * blocks_per_row * 144];
            // Random-ish bytes (no specific pattern needed for bench).
            for (i, b) in w_bytes.iter_mut().enumerate() {
                *b = ((i as u32).wrapping_mul(0x9E3779B9_u32) >> 24) as u8;
            }
            // Make d / dmin valid f16 (avoid NaN).
            for nrow in 0..n_dim {
                for ib in 0..blocks_per_row {
                    let off = (nrow * blocks_per_row + ib) * 144;
                    let d_h = half::f16::from_f32(0.01_f32).to_le_bytes();
                    w_bytes[off] = d_h[0];
                    w_bytes[off + 1] = d_h[1];
                    w_bytes[off + 2] = d_h[0];
                    w_bytes[off + 3] = d_h[1];
                }
            }

            let a = det_vec(m * k, 1.5);
            let a_buf = backend.alloc_shared(m * k * 4).unwrap();
            let w_buf = backend.alloc_shared(w_bytes.len()).unwrap();
            let c_buf = backend.alloc_shared(m * n_dim * 4).unwrap();
            unsafe {
                std::ptr::copy_nonoverlapping(a.as_ptr(), a_buf.contents() as *mut f32, m * k);
                std::ptr::copy_nonoverlapping(
                    w_bytes.as_ptr(),
                    w_buf.contents() as *mut u8,
                    w_bytes.len(),
                );
            }

            // Path A : SGEMM simdgroup_matrix (1 dispatch).
            for _ in 0..3 {
                sgemm_q4_k_f32_simdgroup_matrix_into(backend, &a_buf, &w_buf, &c_buf, m, n_dim, k)
                    .unwrap();
            }
            backend.drain();
            let iters = 20;
            let t0 = std::time::Instant::now();
            for _ in 0..iters {
                sgemm_q4_k_f32_simdgroup_matrix_into(backend, &a_buf, &w_buf, &c_buf, m, n_dim, k)
                    .unwrap();
            }
            backend.drain();
            let sgemm_ms = t0.elapsed().as_secs_f64() * 1000.0 / iters as f64;

            // Path B : sgemv_q4_k loop (M dispatches).
            let row_buf = backend.alloc_shared(k * 4).unwrap();
            let out_row_buf = backend.alloc_shared(n_dim * 4).unwrap();
            for _ in 0..3 {
                for _row in 0..m {
                    let _ = sgemv_q4_k_f32_lcpp_nsg2_into(
                        backend,
                        &row_buf,
                        &w_buf,
                        &out_row_buf,
                        k,
                        n_dim,
                    );
                }
            }
            backend.drain();
            let t0 = std::time::Instant::now();
            for _ in 0..iters {
                for _row in 0..m {
                    let _ = sgemv_q4_k_f32_lcpp_nsg2_into(
                        backend,
                        &row_buf,
                        &w_buf,
                        &out_row_buf,
                        k,
                        n_dim,
                    );
                }
            }
            backend.drain();
            let sgemv_ms = t0.elapsed().as_secs_f64() * 1000.0 / iters as f64;

            let speedup = sgemv_ms / sgemm_ms;

            // Path C : SGEMM tile 64×64 multi-warp (phase 3) si shape compatible.
            let sgemm64_ms = if m % 64 == 0 && n_dim % 64 == 0 {
                for _ in 0..3 {
                    sgemm_q4_k_f32_simdgroup_matrix_64_into(
                        backend, &a_buf, &w_buf, &c_buf, m, n_dim, k,
                    )
                    .unwrap();
                }
                backend.drain();
                let t0 = std::time::Instant::now();
                for _ in 0..iters {
                    sgemm_q4_k_f32_simdgroup_matrix_64_into(
                        backend, &a_buf, &w_buf, &c_buf, m, n_dim, k,
                    )
                    .unwrap();
                }
                backend.drain();
                t0.elapsed().as_secs_f64() * 1000.0 / iters as f64
            } else {
                f64::NAN
            };
            let speedup64 = if sgemm64_ms.is_finite() {
                sgemv_ms / sgemm64_ms
            } else {
                f64::NAN
            };

            eprintln!(
                "M={m:4}, N={n_dim:5}, K={k:5}: Q4_K phase2(8x8)={sgemm_ms:7.3}ms (×{speedup:5.2}), phase3(64x64)={sgemm64_ms:7.3}ms (×{speedup64:5.2}), sgemv_loop={sgemv_ms:7.3}ms"
            );
        }
    }

    /// T162 phase 5 — qmv_fast Q4_K M=1 vs CPU dequant + naive matmul.
    #[test]
    #[cfg(feature = "gpu-tests")]
    fn sgemv_q4_k_f32_qmv_fast_matches_cpu() {
        use rustorch_gguf::dequant::dequantize_block_chunk;
        use rustorch_gguf::tensor::GgmlType;

        let backend = metal_backend();
        if !backend.supports_metal3() {
            eprintln!("[sgemv_q4_k_qmv_fast] skipping: no Metal3");
            return;
        }

        // K=512 (= 1 K-iter, 2 super-blocks), N=16 rows (= 2 TGs of 8 rows).
        let k = 512_usize;
        let n = 16_usize;
        let blocks_per_row = k / 256;

        // Build deterministic Q4_K bytes (144 bytes / super-block).
        let mut w_bytes = vec![0u8; n * blocks_per_row * 144];
        for nrow in 0..n {
            for ib in 0..blocks_per_row {
                let off = (nrow * blocks_per_row + ib) * 144;
                let d_val = ((nrow as f32 + 1.0) * 0.005) + (ib as f32) * 0.001;
                let dmin_val = ((nrow as f32) * 0.002) + (ib as f32) * 0.0005;
                let d_h = half::f16::from_f32(d_val).to_le_bytes();
                let dmin_h = half::f16::from_f32(dmin_val).to_le_bytes();
                w_bytes[off] = d_h[0];
                w_bytes[off + 1] = d_h[1];
                w_bytes[off + 2] = dmin_h[0];
                w_bytes[off + 3] = dmin_h[1];
                for i in 0..12 {
                    w_bytes[off + 4 + i] =
                        (0x10_u8.wrapping_add((nrow as u8) ^ (i as u8) ^ (ib as u8))) | 0x05;
                }
                for i in 0..128 {
                    w_bytes[off + 16 + i] =
                        ((nrow as u8) ^ (i as u8) ^ (ib as u8)).wrapping_add(0x37);
                }
            }
        }

        let x = det_vec(k, 1.5);
        let mut w_f32 = vec![0.0_f32; n * k];
        for nrow in 0..n {
            let row_off = nrow * blocks_per_row * 144;
            let row_bytes = &w_bytes[row_off..row_off + blocks_per_row * 144];
            let row_dst = &mut w_f32[nrow * k..(nrow + 1) * k];
            dequantize_block_chunk(GgmlType::Q4_K, row_bytes, row_dst).unwrap();
        }

        // CPU naive : y[n] = sum_k w[n, k] * x[k]
        let mut y_ref = vec![0.0_f32; n];
        for j in 0..n {
            let mut s = 0.0_f32;
            for l in 0..k {
                s += x[l] * w_f32[j * k + l];
            }
            y_ref[j] = s;
        }

        let x_buf = backend.alloc_shared(k * 4).unwrap();
        let w_buf = backend.alloc_shared(w_bytes.len()).unwrap();
        let y_buf = backend.alloc_shared(n * 4).unwrap();
        unsafe {
            std::ptr::copy_nonoverlapping(x.as_ptr(), x_buf.contents() as *mut f32, k);
            std::ptr::copy_nonoverlapping(
                w_bytes.as_ptr(),
                w_buf.contents() as *mut u8,
                w_bytes.len(),
            );
        }
        sgemv_q4_k_f32_qmv_fast_into(backend, &x_buf, &w_buf, &y_buf, k, n).unwrap();
        backend.drain();

        let mut y_metal = vec![0.0_f32; n];
        unsafe {
            std::ptr::copy_nonoverlapping(y_buf.contents() as *const f32, y_metal.as_mut_ptr(), n);
        }

        for j in 0..n {
            let abs_err = (y_ref[j] - y_metal[j]).abs();
            let denom = y_ref[j].abs().max(1e-3);
            let rel = abs_err / denom;
            assert!(
                rel < 1e-3,
                "qmv_fast Q4_K mismatch at row {j}: ref={} metal={} (rel {:.3e})",
                y_ref[j],
                y_metal[j],
                rel
            );
        }
    }

    /// T162 phase 7-bis — Q3_K SGEMM tile 64×64 multi-warp vs CPU dequant + naive matmul.
    #[test]
    #[cfg(feature = "gpu-tests")]
    fn sgemm_q3_k_f32_simdgroup_matrix_64_matches_cpu() {
        use rustorch_gguf::dequant::dequantize_block_chunk;
        use rustorch_gguf::tensor::GgmlType;

        let backend = metal_backend();
        if !backend.supports_metal3() {
            eprintln!("[sgemm_q3_k_simdmat_64] skipping: no Metal3");
            return;
        }

        let m = 64_usize;
        let n = 64_usize;
        let k = 512_usize;
        let blocks_per_row = k / 256;

        let mut w_bytes = vec![0u8; n * blocks_per_row * 110];
        for nrow in 0..n {
            for ib in 0..blocks_per_row {
                let off = (nrow * blocks_per_row + ib) * 110;
                for i in 0..32 {
                    w_bytes[off + i] = if (nrow + i + ib) % 2 == 0 { 0xAA } else { 0x55 };
                }
                for i in 0..64 {
                    w_bytes[off + 32 + i] = ((i as u8) & 0xC0)
                        | ((((i as u8).wrapping_add(ib as u8)) << 2) & 0x30)
                        | ((((i as u8).wrapping_add(ib as u8)) << 4) & 0x0C)
                        | ((((i as u8).wrapping_add(ib as u8)) << 6) & 0x03);
                }
                for i in 0..12 {
                    w_bytes[off + 96 + i] =
                        (0x40_u8.wrapping_add((nrow as u8) ^ (i as u8) ^ (ib as u8))) | 0x10;
                }
                let d_val = ((nrow as f32 + 1.0) * 0.005) + (ib as f32) * 0.001;
                let d_h = half::f16::from_f32(d_val).to_le_bytes();
                w_bytes[off + 108] = d_h[0];
                w_bytes[off + 109] = d_h[1];
            }
        }

        let a = det_vec(m * k, 1.5);
        let mut w_f32 = vec![0.0_f32; n * k];
        for nrow in 0..n {
            let row_off = nrow * blocks_per_row * 110;
            let row_bytes = &w_bytes[row_off..row_off + blocks_per_row * 110];
            let row_dst = &mut w_f32[nrow * k..(nrow + 1) * k];
            dequantize_block_chunk(GgmlType::Q3_K, row_bytes, row_dst).unwrap();
        }

        let mut c_ref = vec![0.0_f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut s = 0.0_f32;
                for l in 0..k {
                    s += a[i * k + l] * w_f32[j * k + l];
                }
                c_ref[i * n + j] = s;
            }
        }

        let a_buf = backend.alloc_shared(m * k * 4).unwrap();
        let w_buf = backend.alloc_shared(w_bytes.len()).unwrap();
        let c_buf = backend.alloc_shared(m * n * 4).unwrap();
        unsafe {
            std::ptr::copy_nonoverlapping(a.as_ptr(), a_buf.contents() as *mut f32, m * k);
            std::ptr::copy_nonoverlapping(
                w_bytes.as_ptr(),
                w_buf.contents() as *mut u8,
                w_bytes.len(),
            );
        }
        sgemm_q3_k_f32_simdgroup_matrix_64_into(backend, &a_buf, &w_buf, &c_buf, m, n, k).unwrap();
        backend.drain();

        let mut c_metal = vec![0.0_f32; m * n];
        unsafe {
            std::ptr::copy_nonoverlapping(
                c_buf.contents() as *const f32,
                c_metal.as_mut_ptr(),
                m * n,
            );
        }

        for i in 0..(m * n) {
            let abs_err = (c_ref[i] - c_metal[i]).abs();
            let denom = c_ref[i].abs().max(1e-3);
            let rel = abs_err / denom;
            assert!(
                rel < 1e-2,
                "Q3_K SGEMM 64×64 mismatch at {i}: ref={} metal={} (rel {:.3e})",
                c_ref[i],
                c_metal[i],
                rel
            );
        }
    }

    /// T162 phase 7 — Q3_K SGEMM simdgroup_matrix vs CPU dequant + naive matmul.
    #[test]
    #[cfg(feature = "gpu-tests")]
    fn sgemm_q3_k_f32_simdgroup_matrix_matches_cpu() {
        use rustorch_gguf::dequant::dequantize_block_chunk;
        use rustorch_gguf::tensor::GgmlType;

        let backend = metal_backend();
        if !backend.supports_metal3() {
            eprintln!("[sgemm_q3_k_simdmat] skipping: no Metal3");
            return;
        }

        // M=16, N=16, K=512 (2 super-blocks Q3_K par row).
        let m = 16_usize;
        let n = 16_usize;
        let k = 512_usize;
        let blocks_per_row = k / 256;

        // Build deterministic Q3_K bytes (110 bytes / super-block).
        let mut w_bytes = vec![0u8; n * blocks_per_row * 110];
        for nrow in 0..n {
            for ib in 0..blocks_per_row {
                let off = (nrow * blocks_per_row + ib) * 110;
                for i in 0..32 {
                    w_bytes[off + i] = if (nrow + i + ib) % 2 == 0 { 0xAA } else { 0x55 };
                }
                for i in 0..64 {
                    w_bytes[off + 32 + i] = ((i as u8) & 0xC0)
                        | ((((i as u8).wrapping_add(ib as u8)) << 2) & 0x30)
                        | ((((i as u8).wrapping_add(ib as u8)) << 4) & 0x0C)
                        | ((((i as u8).wrapping_add(ib as u8)) << 6) & 0x03);
                }
                for i in 0..12 {
                    w_bytes[off + 96 + i] =
                        (0x40_u8.wrapping_add((nrow as u8) ^ (i as u8) ^ (ib as u8))) | 0x10;
                }
                let d_val = ((nrow as f32 + 1.0) * 0.005) + (ib as f32) * 0.001;
                let d_h = half::f16::from_f32(d_val).to_le_bytes();
                w_bytes[off + 108] = d_h[0];
                w_bytes[off + 109] = d_h[1];
            }
        }

        let a = det_vec(m * k, 1.5);
        let mut w_f32 = vec![0.0_f32; n * k];
        for nrow in 0..n {
            let row_off = nrow * blocks_per_row * 110;
            let row_bytes = &w_bytes[row_off..row_off + blocks_per_row * 110];
            let row_dst = &mut w_f32[nrow * k..(nrow + 1) * k];
            dequantize_block_chunk(GgmlType::Q3_K, row_bytes, row_dst).unwrap();
        }

        let mut c_ref = vec![0.0_f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut s = 0.0_f32;
                for l in 0..k {
                    s += a[i * k + l] * w_f32[j * k + l];
                }
                c_ref[i * n + j] = s;
            }
        }

        let a_buf = backend.alloc_shared(m * k * 4).unwrap();
        let w_buf = backend.alloc_shared(w_bytes.len()).unwrap();
        let c_buf = backend.alloc_shared(m * n * 4).unwrap();
        unsafe {
            std::ptr::copy_nonoverlapping(a.as_ptr(), a_buf.contents() as *mut f32, m * k);
            std::ptr::copy_nonoverlapping(
                w_bytes.as_ptr(),
                w_buf.contents() as *mut u8,
                w_bytes.len(),
            );
        }
        sgemm_q3_k_f32_simdgroup_matrix_into(backend, &a_buf, &w_buf, &c_buf, m, n, k).unwrap();
        backend.drain();

        let mut c_metal = vec![0.0_f32; m * n];
        unsafe {
            std::ptr::copy_nonoverlapping(
                c_buf.contents() as *const f32,
                c_metal.as_mut_ptr(),
                m * n,
            );
        }

        for i in 0..(m * n) {
            let abs_err = (c_ref[i] - c_metal[i]).abs();
            let denom = c_ref[i].abs().max(1e-3);
            let rel = abs_err / denom;
            assert!(
                rel < 1e-2,
                "Q3_K SGEMM mismatch at {i}: ref={} metal={} (rel {:.3e})",
                c_ref[i],
                c_metal[i],
                rel
            );
        }
    }

    /// T162 phase 5-bis — Q6_K SGEMM tile 64×64 multi-warp vs CPU dequant + naive matmul.
    #[test]
    #[cfg(feature = "gpu-tests")]
    fn sgemm_q6_k_f32_simdgroup_matrix_64_matches_cpu() {
        use rustorch_gguf::dequant::dequantize_block_chunk;
        use rustorch_gguf::tensor::GgmlType;

        let backend = metal_backend();
        if !backend.supports_metal3() {
            eprintln!("[sgemm_q6_k_simdmat_64] skipping: no Metal3");
            return;
        }

        // M=N=64 (shape minimale), K=512 (2 super-blocks).
        let m = 64_usize;
        let n = 64_usize;
        let k = 512_usize;
        let blocks_per_row = k / 256;

        let mut w_bytes = vec![0u8; n * blocks_per_row * 210];
        for nrow in 0..n {
            for ib in 0..blocks_per_row {
                let off = (nrow * blocks_per_row + ib) * 210;
                for i in 0..128 {
                    w_bytes[off + i] = ((nrow as u8) ^ (i as u8) ^ (ib as u8)).wrapping_add(0x37);
                }
                for i in 0..64 {
                    w_bytes[off + 128 + i] =
                        ((nrow as u8) ^ (i as u8) ^ (ib as u8) ^ 0xA5).wrapping_add(0x12);
                }
                for i in 0..16 {
                    let s: i8 = (((nrow as i32 + ib as i32 + i as i32) % 32) - 16) as i8;
                    w_bytes[off + 192 + i] = s as u8;
                }
                let d_val = ((nrow as f32 + 1.0) * 0.005) + (ib as f32) * 0.001;
                let d_h = half::f16::from_f32(d_val).to_le_bytes();
                w_bytes[off + 208] = d_h[0];
                w_bytes[off + 209] = d_h[1];
            }
        }

        let a = det_vec(m * k, 1.5);
        let mut w_f32 = vec![0.0_f32; n * k];
        for nrow in 0..n {
            let row_off = nrow * blocks_per_row * 210;
            let row_bytes = &w_bytes[row_off..row_off + blocks_per_row * 210];
            let row_dst = &mut w_f32[nrow * k..(nrow + 1) * k];
            dequantize_block_chunk(GgmlType::Q6_K, row_bytes, row_dst).unwrap();
        }

        let mut c_ref = vec![0.0_f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut s = 0.0_f32;
                for l in 0..k {
                    s += a[i * k + l] * w_f32[j * k + l];
                }
                c_ref[i * n + j] = s;
            }
        }

        let a_buf = backend.alloc_shared(m * k * 4).unwrap();
        let w_buf = backend.alloc_shared(w_bytes.len()).unwrap();
        let c_buf = backend.alloc_shared(m * n * 4).unwrap();
        unsafe {
            std::ptr::copy_nonoverlapping(a.as_ptr(), a_buf.contents() as *mut f32, m * k);
            std::ptr::copy_nonoverlapping(
                w_bytes.as_ptr(),
                w_buf.contents() as *mut u8,
                w_bytes.len(),
            );
        }
        sgemm_q6_k_f32_simdgroup_matrix_64_into(backend, &a_buf, &w_buf, &c_buf, m, n, k).unwrap();
        backend.drain();

        let mut c_metal = vec![0.0_f32; m * n];
        unsafe {
            std::ptr::copy_nonoverlapping(
                c_buf.contents() as *const f32,
                c_metal.as_mut_ptr(),
                m * n,
            );
        }

        for i in 0..(m * n) {
            let abs_err = (c_ref[i] - c_metal[i]).abs();
            let denom = c_ref[i].abs().max(1e-3);
            let rel = abs_err / denom;
            assert!(
                rel < 1e-2,
                "Q6_K SGEMM 64×64 mismatch at {i}: ref={} metal={} (rel {:.3e})",
                c_ref[i],
                c_metal[i],
                rel
            );
        }
    }

    /// T162 phase 5 — Q6_K SGEMM simdgroup_matrix vs CPU dequant + naive matmul.
    ///
    /// Validation correctness du nouveau kernel Q6_K. Q6_K est utilisé pour
    /// V proj et FFN down sur Qwen3-14B Q4_K_M : ce kernel les couvre côté SGEMM.
    #[test]
    #[cfg(feature = "gpu-tests")]
    fn sgemm_q6_k_f32_simdgroup_matrix_matches_cpu() {
        use rustorch_gguf::dequant::dequantize_block_chunk;
        use rustorch_gguf::tensor::GgmlType;

        let backend = metal_backend();
        if !backend.supports_metal3() {
            eprintln!("[sgemm_q6_k_simdmat] skipping: no Metal3");
            return;
        }

        // M=16, N=16, K=512 (2 super-blocks Q6_K par row).
        let m = 16_usize;
        let n = 16_usize;
        let k = 512_usize;
        let blocks_per_row = k / 256;

        // Build deterministic Q6_K bytes : 210 bytes / super-block.
        // Layout : ql[128] + qh[64] + scales_i8[16] + d (f16) at offset 208.
        let mut w_bytes = vec![0u8; n * blocks_per_row * 210];
        for nrow in 0..n {
            for ib in 0..blocks_per_row {
                let off = (nrow * blocks_per_row + ib) * 210;
                // ql : 128 bytes pseudo-random.
                for i in 0..128 {
                    w_bytes[off + i] = ((nrow as u8) ^ (i as u8) ^ (ib as u8)).wrapping_add(0x37);
                }
                // qh : 64 bytes pseudo-random.
                for i in 0..64 {
                    w_bytes[off + 128 + i] =
                        ((nrow as u8) ^ (i as u8) ^ (ib as u8) ^ 0xA5).wrapping_add(0x12);
                }
                // scales : 16 i8 signed (range -32..31 typique).
                for i in 0..16 {
                    let s: i8 = (((nrow as i32 + ib as i32 + i as i32) % 32) - 16) as i8;
                    w_bytes[off + 192 + i] = s as u8;
                }
                // d (f16) at offset 208.
                let d_val = ((nrow as f32 + 1.0) * 0.005) + (ib as f32) * 0.001;
                let d_h = half::f16::from_f32(d_val).to_le_bytes();
                w_bytes[off + 208] = d_h[0];
                w_bytes[off + 209] = d_h[1];
            }
        }

        let a = det_vec(m * k, 1.5);
        let mut w_f32 = vec![0.0_f32; n * k];
        for nrow in 0..n {
            let row_off = nrow * blocks_per_row * 210;
            let row_bytes = &w_bytes[row_off..row_off + blocks_per_row * 210];
            let row_dst = &mut w_f32[nrow * k..(nrow + 1) * k];
            dequantize_block_chunk(GgmlType::Q6_K, row_bytes, row_dst).unwrap();
        }

        let mut c_ref = vec![0.0_f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut s = 0.0_f32;
                for l in 0..k {
                    s += a[i * k + l] * w_f32[j * k + l];
                }
                c_ref[i * n + j] = s;
            }
        }

        let a_buf = backend.alloc_shared(m * k * 4).unwrap();
        let w_buf = backend.alloc_shared(w_bytes.len()).unwrap();
        let c_buf = backend.alloc_shared(m * n * 4).unwrap();
        unsafe {
            std::ptr::copy_nonoverlapping(a.as_ptr(), a_buf.contents() as *mut f32, m * k);
            std::ptr::copy_nonoverlapping(
                w_bytes.as_ptr(),
                w_buf.contents() as *mut u8,
                w_bytes.len(),
            );
        }
        sgemm_q6_k_f32_simdgroup_matrix_into(backend, &a_buf, &w_buf, &c_buf, m, n, k).unwrap();
        backend.drain();

        let mut c_metal = vec![0.0_f32; m * n];
        unsafe {
            std::ptr::copy_nonoverlapping(
                c_buf.contents() as *const f32,
                c_metal.as_mut_ptr(),
                m * n,
            );
        }

        for i in 0..(m * n) {
            let abs_err = (c_ref[i] - c_metal[i]).abs();
            let denom = c_ref[i].abs().max(1e-3);
            let rel = abs_err / denom;
            assert!(
                rel < 1e-2,
                "Q6_K SGEMM mismatch at {i}: ref={} metal={} (rel {:.3e})",
                c_ref[i],
                c_metal[i],
                rel
            );
        }
    }

    /// T162 phase 2 — Q4_K SGEMM simdgroup_matrix vs CPU dequant + naive matmul.
    ///
    /// PREMIER kernel rustorch combinant Apple AMX et Q4_K. Test gate de toute
    /// la phase 2 (port to prefill + spec decoding). Si ce test passe, on a
    /// validé le path Q4_K simdgroup_matrix de bout en bout.
    #[test]
    #[cfg(feature = "gpu-tests")]
    fn sgemm_q4_k_f32_simdgroup_matrix_matches_cpu() {
        use rustorch_gguf::dequant::dequant_q4_k;

        let backend = metal_backend();
        if !backend.supports_metal3() {
            eprintln!("[sgemm_q4_k_simdmat] skipping: no Metal3");
            return;
        }

        // M=16, N=16, K=512 (= 2 super-blocks par row). Petit mais exerce
        // tous les chemins (multi-tile output, multi-superblock K loop).
        let m = 16_usize;
        let n = 16_usize;
        let k = 512_usize;
        let blocks_per_row = k / 256;

        // Build deterministic Q4_K bytes : N rows × blocks_per_row × 144 bytes.
        let mut w_bytes = vec![0u8; n * blocks_per_row * 144];
        for nrow in 0..n {
            for ib in 0..blocks_per_row {
                let off = (nrow * blocks_per_row + ib) * 144;
                // d, dmin in half-float
                let d_val = ((nrow as f32 + 1.0) * 0.01) + (ib as f32) * 0.001;
                let dmin_val = ((nrow as f32) * 0.005) + (ib as f32) * 0.0005;
                let d_h = half::f16::from_f32(d_val).to_le_bytes();
                let dmin_h = half::f16::from_f32(dmin_val).to_le_bytes();
                w_bytes[off] = d_h[0];
                w_bytes[off + 1] = d_h[1];
                w_bytes[off + 2] = dmin_h[0];
                w_bytes[off + 3] = dmin_h[1];

                // 12 packed scales bytes — non-trivial pattern.
                for i in 0..12 {
                    w_bytes[off + 4 + i] =
                        (0x10_u8.wrapping_add((nrow as u8) ^ (i as u8) ^ (ib as u8))) | 0x05;
                }

                // 128 qs bytes — varied nibbles.
                for i in 0..128 {
                    w_bytes[off + 16 + i] =
                        ((nrow as u8) ^ (i as u8) ^ (ib as u8)).wrapping_add(0x37);
                }
            }
        }

        // CPU reference : dequant W rows then naive matmul C = A @ W^T.
        let a = det_vec(m * k, 1.5);
        let mut w_f32 = vec![0.0_f32; n * k];
        for nrow in 0..n {
            let row_off = nrow * blocks_per_row * 144;
            let row_bytes = &w_bytes[row_off..row_off + blocks_per_row * 144];
            let row_dst = &mut w_f32[nrow * k..(nrow + 1) * k];
            dequant_q4_k(row_bytes, row_dst).unwrap();
        }

        let mut c_ref = vec![0.0_f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut s = 0.0_f32;
                for l in 0..k {
                    s += a[i * k + l] * w_f32[j * k + l]; // W^T : W[j, l]
                }
                c_ref[i * n + j] = s;
            }
        }

        // GPU path.
        let a_buf = backend.alloc_shared(m * k * 4).unwrap();
        let w_buf = backend.alloc_shared(w_bytes.len()).unwrap();
        let c_buf = backend.alloc_shared(m * n * 4).unwrap();
        unsafe {
            std::ptr::copy_nonoverlapping(a.as_ptr(), a_buf.contents() as *mut f32, m * k);
            std::ptr::copy_nonoverlapping(
                w_bytes.as_ptr(),
                w_buf.contents() as *mut u8,
                w_bytes.len(),
            );
        }
        sgemm_q4_k_f32_simdgroup_matrix_into(backend, &a_buf, &w_buf, &c_buf, m, n, k).unwrap();
        backend.drain();

        let mut c_metal = vec![0.0_f32; m * n];
        unsafe {
            std::ptr::copy_nonoverlapping(
                c_buf.contents() as *const f32,
                c_metal.as_mut_ptr(),
                m * n,
            );
        }

        for i in 0..(m * n) {
            let abs_err = (c_ref[i] - c_metal[i]).abs();
            let denom = c_ref[i].abs().max(1e-3);
            let rel = abs_err / denom;
            assert!(
                rel < 1e-2,
                "Q4_K SGEMM simdmat mismatch at {i}: ref={} metal={} (rel {:.3e})",
                c_ref[i],
                c_metal[i],
                rel
            );
        }
    }

    /// T163 phase 9f-ter — Q4_K SGEMM EXPERT-MAJOR : test parité avec stacked
    /// weights, tile_expert_ids buffer, et per-tile expert offset.
    /// Validation : 2 experts différents, 16 rows total (8 par expert), parité
    /// vs CPU dequant-puis-matmul row by row par expert.
    #[test]
    #[cfg(feature = "gpu-tests")]
    fn sgemm_q4_k_f32_expert_major_8x8_matches_cpu() {
        use rustorch_gguf::dequant::dequant_q4_k;

        let backend = metal_backend();
        if !backend.supports_metal3() {
            eprintln!("[sgemm_q4_k_expert_major] skipping: no Metal3");
            return;
        }

        // 4 experts, 16 rows (4 M-tiles × 8 rows), 8 cols, K=512.
        // Tile 0 expert 1, tile 1 expert 2, tile 2 expert 0, tile 3 expert 3
        // (intentionally non-sorted to verify per-tile indirection).
        let n_experts = 4_usize;
        let m = 32_usize; // 4 tiles × 8 rows
        let n = 8_usize;
        let k = 512_usize;
        let blocks_per_row = k / 256;
        let bytes_per_expert = n * blocks_per_row * 144;

        // Build n_experts stacked Q4_K matrices.
        let mut w_stacked = vec![0u8; n_experts * bytes_per_expert];
        for e in 0..n_experts {
            for nrow in 0..n {
                for ib in 0..blocks_per_row {
                    let off = e * bytes_per_expert + (nrow * blocks_per_row + ib) * 144;
                    let d_val = ((nrow as f32 + 1.0 + e as f32 * 2.0) * 0.01) + (ib as f32) * 0.001;
                    let dmin_val = ((nrow as f32 + e as f32) * 0.005) + (ib as f32) * 0.0005;
                    let d_h = half::f16::from_f32(d_val).to_le_bytes();
                    let dmin_h = half::f16::from_f32(dmin_val).to_le_bytes();
                    w_stacked[off] = d_h[0];
                    w_stacked[off + 1] = d_h[1];
                    w_stacked[off + 2] = dmin_h[0];
                    w_stacked[off + 3] = dmin_h[1];
                    for i in 0..12 {
                        w_stacked[off + 4 + i] = (0x10_u8
                            .wrapping_add((nrow as u8) ^ (i as u8) ^ (ib as u8) ^ (e as u8 * 7)))
                            | 0x05;
                    }
                    for i in 0..128 {
                        w_stacked[off + 16 + i] =
                            ((nrow as u8) ^ (i as u8) ^ (ib as u8) ^ (e as u8 * 13))
                                .wrapping_add(0x37);
                    }
                }
            }
        }

        // Tile expert IDs (one per M-tile = 4 entries).
        let tile_expert_ids: Vec<u32> = vec![1, 2, 0, 3];

        // CPU reference : for each tile, dequant the appropriate expert's W,
        // then matmul tile's 8 rows of A.
        let a = det_vec(m * k, 1.7);
        let mut c_ref = vec![0.0_f32; m * n];
        for (tile_idx, &tile_expert_id) in tile_expert_ids.iter().enumerate().take(m / 8) {
            let expert_id = tile_expert_id as usize;
            let mut w_f32 = vec![0.0_f32; n * k];
            for nrow in 0..n {
                let off_w = expert_id * bytes_per_expert + nrow * blocks_per_row * 144;
                let row_bytes = &w_stacked[off_w..off_w + blocks_per_row * 144];
                let row_dst = &mut w_f32[nrow * k..(nrow + 1) * k];
                dequant_q4_k(row_bytes, row_dst).unwrap();
            }
            for i in 0..8 {
                let row_idx = tile_idx * 8 + i;
                for j in 0..n {
                    let mut s = 0.0_f32;
                    for l in 0..k {
                        s += a[row_idx * k + l] * w_f32[j * k + l];
                    }
                    c_ref[row_idx * n + j] = s;
                }
            }
        }

        // GPU path.
        let a_buf = backend.alloc_shared(m * k * 4).unwrap();
        let w_buf = backend.alloc_shared(w_stacked.len()).unwrap();
        let tile_buf = backend.alloc_shared(tile_expert_ids.len() * 4).unwrap();
        let c_buf = backend.alloc_shared(m * n * 4).unwrap();
        unsafe {
            std::ptr::copy_nonoverlapping(a.as_ptr(), a_buf.contents() as *mut f32, m * k);
            std::ptr::copy_nonoverlapping(
                w_stacked.as_ptr(),
                w_buf.contents() as *mut u8,
                w_stacked.len(),
            );
            std::ptr::copy_nonoverlapping(
                tile_expert_ids.as_ptr(),
                tile_buf.contents() as *mut u32,
                tile_expert_ids.len(),
            );
        }
        sgemm_q4_k_f32_expert_major_8x8_into(
            backend,
            &a_buf,
            &w_buf,
            &tile_buf,
            &c_buf,
            m,
            n,
            k,
            bytes_per_expert,
        )
        .unwrap();
        backend.drain();

        let mut c_metal = vec![0.0_f32; m * n];
        unsafe {
            std::ptr::copy_nonoverlapping(
                c_buf.contents() as *const f32,
                c_metal.as_mut_ptr(),
                m * n,
            );
        }

        for i in 0..(m * n) {
            let abs_err = (c_ref[i] - c_metal[i]).abs();
            let denom = c_ref[i].abs().max(1e-3);
            let rel = abs_err / denom;
            assert!(
                rel < 1e-2,
                "Q4_K SGEMM expert_major mismatch at {i}: ref={} metal={} (rel {:.3e})",
                c_ref[i],
                c_metal[i],
                rel
            );
        }
    }

    /// T162 phase 1 — Bench comparatif sgemm_f32_simdgroup_matrix vs sgemv loop.
    ///
    /// Mesure le speedup de notre nouveau path SGEMM Apple AMX vs l'approche
    /// "M iterations de sgemv F32" qui est le path prefill actuel rustorch.
    /// Doit être lancé manuellement (mode bench) car les timings varient avec
    /// la charge système. Le test PASS si simdgroup_matrix wins pour M=64.
    #[test]
    #[cfg(feature = "gpu-tests")]
    #[ignore]
    fn sgemm_f32_simdgroup_matrix_bench() {
        let backend = metal_backend();
        if !backend.supports_metal3() {
            eprintln!("skipping: no Metal3");
            return;
        }

        // Shape représentative d'un slice prefill : M tokens × K hidden × N out.
        // Test à 3 valeurs de M (1=decode, 8=tile minimum, 64=prefill batch).
        for m in [1, 8, 64, 128].iter() {
            let m = *m;
            let k = 1024_usize;
            let n = 1024_usize;

            let a = det_vec(m * k, 0.7);
            let b = det_vec(k * n, 1.3);

            let a_buf = backend.alloc_shared(m * k * 4).unwrap();
            let b_buf = backend.alloc_shared(k * n * 4).unwrap();
            let c_sgemm = backend.alloc_shared(m * n * 4).unwrap();
            let c_sgemv = backend.alloc_shared(m * n * 4).unwrap();
            unsafe {
                std::ptr::copy_nonoverlapping(a.as_ptr(), a_buf.contents() as *mut f32, m * k);
                std::ptr::copy_nonoverlapping(b.as_ptr(), b_buf.contents() as *mut f32, k * n);
            }

            // Path A : simdgroup_matrix sgemm (1 dispatch).
            // Skip si m%8 != 0 (sgemm requires multiples of 8).
            let sgemm_ms = if m % 8 == 0 {
                let warmups = 3;
                for _ in 0..warmups {
                    sgemm_f32_simdgroup_matrix_into(backend, &a_buf, &b_buf, &c_sgemm, m, n, k)
                        .unwrap();
                }
                backend.drain();
                let iters = 20;
                let t0 = std::time::Instant::now();
                for _ in 0..iters {
                    sgemm_f32_simdgroup_matrix_into(backend, &a_buf, &b_buf, &c_sgemm, m, n, k)
                        .unwrap();
                }
                backend.drain();
                t0.elapsed().as_secs_f64() * 1000.0 / iters as f64
            } else {
                f64::NAN
            };

            // Path B : sgemv F32 loop (M dispatches).
            // sgemv attend [N, K] row-major (output is slow axis), nous avons
            // B en [K, N]. Pour comparer fair, on accepte la dérive et ne
            // valide pas la correctness ici (juste timing).
            let warmups = 3;
            for _ in 0..warmups {
                for row in 0..m {
                    let row_offset = row * k * 4;
                    let row_buf = backend.alloc_shared(k * 4).unwrap();
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            (a_buf.contents() as *const u8).add(row_offset) as *const f32,
                            row_buf.contents() as *mut f32,
                            k,
                        );
                    }
                    let _ = sgemv_f32_lcpp_simd_into(backend, &row_buf, &b_buf, &c_sgemv, k, n);
                }
            }
            backend.drain();

            let iters = 20;
            let t0 = std::time::Instant::now();
            for _ in 0..iters {
                for _row in 0..m {
                    let _ = sgemv_f32_lcpp_simd_into(backend, &a_buf, &b_buf, &c_sgemv, k, n);
                }
            }
            backend.drain();
            let sgemv_ms = t0.elapsed().as_secs_f64() * 1000.0 / iters as f64;

            let speedup = sgemv_ms / sgemm_ms;
            eprintln!(
                "M={m:4}, N={n}, K={k}: sgemm_simdmat={sgemm_ms:6.3}ms, sgemv_loop={sgemv_ms:6.3}ms, speedup={speedup:.2}×"
            );
        }
    }

    /// T162 phase 1 — F32 SGEMM via simdgroup_matrix 8×8 vs naive CPU matmul.
    ///
    /// Validation que le kernel compile et exécute correctement sur Apple M4.
    /// Ce test est le GATE de tout le travail simdgroup_matrix futur (T162 Q4_K).
    #[test]
    #[cfg(feature = "gpu-tests")]
    fn sgemm_f32_simdgroup_matrix_matches_naive() {
        let backend = metal_backend();
        if !backend.supports_metal3() {
            eprintln!(
                "[sgemm_f32_simdgroup_matrix] skipping: device does not support Metal3 ({})",
                backend.adapter_name()
            );
            return;
        }

        // Shapes alignées sur multiples de 8. Représentatives d'un slice
        // d'attention head (M=8 query tokens, N=128 head_dim equivalent, K=128).
        let m = 16_usize;
        let n = 32_usize;
        let k = 64_usize;

        let a = det_vec(m * k, 0.7);
        let b = det_vec(k * n, 1.3);

        // CPU naive matmul row-major : C[i,j] = sum_l A[i,l] * B[l,j]
        let mut c_ref = vec![0.0_f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut s = 0.0_f32;
                for l in 0..k {
                    s += a[i * k + l] * b[l * n + j];
                }
                c_ref[i * n + j] = s;
            }
        }

        // GPU path via simdgroup_matrix.
        let a_buf = backend.alloc_shared(m * k * 4).unwrap();
        let b_buf = backend.alloc_shared(k * n * 4).unwrap();
        let c_buf = backend.alloc_shared(m * n * 4).unwrap();
        unsafe {
            std::ptr::copy_nonoverlapping(a.as_ptr(), a_buf.contents() as *mut f32, m * k);
            std::ptr::copy_nonoverlapping(b.as_ptr(), b_buf.contents() as *mut f32, k * n);
        }
        sgemm_f32_simdgroup_matrix_into(backend, &a_buf, &b_buf, &c_buf, m, n, k).unwrap();
        backend.drain();

        let mut c_metal = vec![0.0_f32; m * n];
        unsafe {
            std::ptr::copy_nonoverlapping(
                c_buf.contents() as *const f32,
                c_metal.as_mut_ptr(),
                m * n,
            );
        }

        for i in 0..(m * n) {
            let abs_err = (c_ref[i] - c_metal[i]).abs();
            let denom = c_ref[i].abs().max(1e-4);
            let rel = abs_err / denom;
            assert!(
                rel < 1e-3,
                "sgemm simdgroup_matrix mismatch at {i}: ref={} metal={} (rel {:.3e})",
                c_ref[i],
                c_metal[i],
                rel
            );
        }
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

    /// T109 — Batched GQA decode must match B sequential gqa_decode calls
    /// (each at pos_base+b with kv_len=pos_base+b+1).
    #[test]
    #[cfg(feature = "gpu-tests")]
    fn gqa_decode_batched_f32_matches_singles() {
        let backend = metal_backend();
        let n_heads = 4usize;
        let n_kv = 2usize;
        let head_dim = 64usize;
        let max_seq = 32usize;
        let pos_base = 5usize;
        let b = 4usize;
        let q_size = n_heads * head_dim;
        let kv_size = n_kv * max_seq * head_dim;

        // Build B different Q rows
        let mut q_all = vec![0.0f32; b * q_size];
        for batch in 0..b {
            for i in 0..q_size {
                q_all[batch * q_size + i] = ((i as f32 + 1.0 + batch as f32 * 3.0) * 0.01).sin();
            }
        }
        // Pre-fill K and V cache (positions 0..pos_base+b populated)
        let kv_total = pos_base + b;
        let mut k_cache = vec![0.0f32; kv_size];
        let mut v_cache = vec![0.0f32; kv_size];
        for kvh in 0..n_kv {
            for p in 0..kv_total {
                for d in 0..head_dim {
                    let idx = kvh * max_seq * head_dim + p * head_dim + d;
                    k_cache[idx] = ((kvh + 1) as f32 + p as f32 * 0.7 + d as f32 * 0.01).cos();
                    v_cache[idx] = ((kvh + 1) as f32 + p as f32 * 0.5 + d as f32 * 0.02).sin();
                }
            }
        }

        let q_buf = backend.alloc_shared(b * q_size * 4).unwrap();
        let k_buf = backend.alloc_shared(kv_size * 4).unwrap();
        let v_buf = backend.alloc_shared(kv_size * 4).unwrap();
        unsafe {
            std::ptr::copy_nonoverlapping(q_all.as_ptr(), q_buf.contents() as *mut f32, b * q_size);
            std::ptr::copy_nonoverlapping(k_cache.as_ptr(), k_buf.contents() as *mut f32, kv_size);
            std::ptr::copy_nonoverlapping(v_cache.as_ptr(), v_buf.contents() as *mut f32, kv_size);
        }

        // Reference: B sequential gqa_decode calls
        let mut out_seq = vec![0.0f32; b * q_size];
        for batch in 0..b {
            let q_b_buf = backend.alloc_shared(q_size * 4).unwrap();
            let out_b_buf = backend.alloc_shared(q_size * 4).unwrap();
            unsafe {
                std::ptr::copy_nonoverlapping(
                    q_all[batch * q_size..(batch + 1) * q_size].as_ptr(),
                    q_b_buf.contents() as *mut f32,
                    q_size,
                );
            }
            let kv_len = pos_base + batch + 1;
            gqa_decode_f32(
                backend, &q_b_buf, &k_buf, &v_buf, &out_b_buf, n_heads, n_kv, head_dim, kv_len,
                max_seq,
            )
            .unwrap();
            backend.drain();
            unsafe {
                std::ptr::copy_nonoverlapping(
                    out_b_buf.contents() as *const f32,
                    out_seq[batch * q_size..(batch + 1) * q_size].as_mut_ptr(),
                    q_size,
                );
            }
        }

        // Batched call
        let out_batch_buf = backend.alloc_shared(b * q_size * 4).unwrap();
        gqa_decode_batched_f32(
            backend,
            &q_buf,
            &k_buf,
            &v_buf,
            &out_batch_buf,
            n_heads,
            n_kv,
            head_dim,
            pos_base,
            b,
            max_seq,
        )
        .unwrap();
        backend.drain();
        let out_batch = unsafe {
            std::slice::from_raw_parts(out_batch_buf.contents() as *const f32, b * q_size).to_vec()
        };

        for batch in 0..b {
            for i in 0..q_size {
                let a = out_seq[batch * q_size + i];
                let bv = out_batch[batch * q_size + i];
                let r = (a - bv).abs() / a.abs().max(1e-4);
                assert!(
                    r < 1e-3,
                    "batch={batch} i={i} mismatch: seq={a} batch={bv} (rel {r:.3e})"
                );
            }
        }
    }

    /// T162 phase 9e — batched ssm_apply_gate must match B sequential calls.
    #[test]
    #[cfg(feature = "gpu-tests")]
    fn ssm_apply_gate_batched_f32_matches_singles() {
        let backend = metal_backend();
        let n_v = 32usize;
        let b = 4usize;

        // Build B different (alpha, beta) rows + shared (dt_bias, ssm_a).
        let mut alpha_all = vec![0.0f32; b * n_v];
        let mut beta_all = vec![0.0f32; b * n_v];
        for batch in 0..b {
            for i in 0..n_v {
                alpha_all[batch * n_v + i] =
                    ((i as f32 + 1.0 + batch as f32 * 3.0) * 0.011).sin() * 2.0;
                beta_all[batch * n_v + i] =
                    ((i as f32 + 1.0 + batch as f32 * 5.0) * 0.017).cos() * 0.5;
            }
        }
        let dt_bias: Vec<f32> = (0..n_v).map(|i| 0.05 + (i as f32) * 0.001).collect();
        let ssm_a: Vec<f32> = (0..n_v).map(|i| 1.0 + (i as f32) * 0.003).collect();

        let dt_buf = backend.alloc_shared(n_v * 4).unwrap();
        let a_buf = backend.alloc_shared(n_v * 4).unwrap();
        unsafe {
            std::ptr::copy_nonoverlapping(dt_bias.as_ptr(), dt_buf.contents() as *mut f32, n_v);
            std::ptr::copy_nonoverlapping(ssm_a.as_ptr(), a_buf.contents() as *mut f32, n_v);
        }

        // Reference: B sequential calls.
        let mut gate_h_seq = vec![0.0f32; b * n_v];
        let mut beta_sig_seq = vec![0.0f32; b * n_v];
        for batch in 0..b {
            let alpha_b = backend.alloc_shared(n_v * 4).unwrap();
            let beta_b = backend.alloc_shared(n_v * 4).unwrap();
            let g_b = backend.alloc_shared(n_v * 4).unwrap();
            let bs_b = backend.alloc_shared(n_v * 4).unwrap();
            unsafe {
                std::ptr::copy_nonoverlapping(
                    alpha_all[batch * n_v..(batch + 1) * n_v].as_ptr(),
                    alpha_b.contents() as *mut f32,
                    n_v,
                );
                std::ptr::copy_nonoverlapping(
                    beta_all[batch * n_v..(batch + 1) * n_v].as_ptr(),
                    beta_b.contents() as *mut f32,
                    n_v,
                );
            }
            ssm_apply_gate_f32(
                backend, &alpha_b, &beta_b, &dt_buf, &a_buf, &g_b, &bs_b, n_v,
            )
            .unwrap();
            backend.drain();
            unsafe {
                std::ptr::copy_nonoverlapping(
                    g_b.contents() as *const f32,
                    gate_h_seq[batch * n_v..(batch + 1) * n_v].as_mut_ptr(),
                    n_v,
                );
                std::ptr::copy_nonoverlapping(
                    bs_b.contents() as *const f32,
                    beta_sig_seq[batch * n_v..(batch + 1) * n_v].as_mut_ptr(),
                    n_v,
                );
            }
        }

        // Batched call.
        let alpha_buf = backend.alloc_shared(b * n_v * 4).unwrap();
        let beta_buf = backend.alloc_shared(b * n_v * 4).unwrap();
        let g_buf = backend.alloc_shared(b * n_v * 4).unwrap();
        let bs_buf = backend.alloc_shared(b * n_v * 4).unwrap();
        unsafe {
            std::ptr::copy_nonoverlapping(
                alpha_all.as_ptr(),
                alpha_buf.contents() as *mut f32,
                b * n_v,
            );
            std::ptr::copy_nonoverlapping(
                beta_all.as_ptr(),
                beta_buf.contents() as *mut f32,
                b * n_v,
            );
        }
        ssm_apply_gate_batched_f32(
            backend, &alpha_buf, &beta_buf, &dt_buf, &a_buf, &g_buf, &bs_buf, n_v, b,
        )
        .unwrap();
        backend.drain();

        let g_bat =
            unsafe { std::slice::from_raw_parts(g_buf.contents() as *const f32, b * n_v).to_vec() };
        let bs_bat = unsafe {
            std::slice::from_raw_parts(bs_buf.contents() as *const f32, b * n_v).to_vec()
        };
        for i in 0..b * n_v {
            assert!(
                (gate_h_seq[i] - g_bat[i]).abs() < 1e-5,
                "gate_h[{i}] mismatch: seq={} bat={}",
                gate_h_seq[i],
                g_bat[i]
            );
            assert!(
                (beta_sig_seq[i] - bs_bat[i]).abs() < 1e-5,
                "beta_sig[{i}] mismatch: seq={} bat={}",
                beta_sig_seq[i],
                bs_bat[i]
            );
        }
    }

    /// T162 phase 9a — batched partial-RoPE must match B sequential calls
    /// at consecutive positions, leaving the [rope_dim..head_dim] tail untouched.
    #[test]
    #[cfg(feature = "gpu-tests")]
    fn rope_half_split_partial_batched_f32_matches_singles() {
        let backend = metal_backend();
        let n_heads = 4usize;
        let head_dim = 128usize;
        let rope_dim = 64usize; // partial — tail [64..128] untouched
        let max_seq = 32usize;
        let pos_base = 7usize;
        let b = 4usize;
        let row_size = n_heads * head_dim;
        let half_rope = rope_dim / 2;

        // Build B different rows.
        let mut x_all = vec![0.0f32; b * row_size];
        for batch in 0..b {
            for i in 0..row_size {
                x_all[batch * row_size + i] = ((i as f32 + 1.0 + batch as f32) * 0.013).sin();
            }
        }
        // Build cos/sin tables [max_seq, rope_dim/2].
        let mut cos_tab = vec![0.0f32; max_seq * half_rope];
        let mut sin_tab = vec![0.0f32; max_seq * half_rope];
        for p in 0..max_seq {
            for k in 0..half_rope {
                let theta = (p as f32) * 0.001 * ((k + 1) as f32);
                cos_tab[p * half_rope + k] = theta.cos();
                sin_tab[p * half_rope + k] = theta.sin();
            }
        }
        let cos_buf = backend.alloc_shared(max_seq * half_rope * 4).unwrap();
        let sin_buf = backend.alloc_shared(max_seq * half_rope * 4).unwrap();
        unsafe {
            std::ptr::copy_nonoverlapping(
                cos_tab.as_ptr(),
                cos_buf.contents() as *mut f32,
                max_seq * half_rope,
            );
            std::ptr::copy_nonoverlapping(
                sin_tab.as_ptr(),
                sin_buf.contents() as *mut f32,
                max_seq * half_rope,
            );
        }

        // Reference: B sequential rope_half_split_f32 calls (with rope_dim).
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
                rope_dim,
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

        // Batched call.
        let x_batch_buf = backend.alloc_shared(b * row_size * 4).unwrap();
        unsafe {
            std::ptr::copy_nonoverlapping(
                x_all.as_ptr(),
                x_batch_buf.contents() as *mut f32,
                b * row_size,
            );
        }
        rope_half_split_partial_batched_f32(
            backend,
            &x_batch_buf,
            &cos_buf,
            &sin_buf,
            n_heads,
            head_dim,
            rope_dim,
            pos_base,
            b,
        )
        .unwrap();
        backend.drain();

        let x_bat = unsafe {
            std::slice::from_raw_parts(x_batch_buf.contents() as *const f32, b * row_size).to_vec()
        };
        for batch in 0..b {
            for h in 0..n_heads {
                for i in 0..head_dim {
                    let off = batch * row_size + h * head_dim + i;
                    let a = x_seq[off];
                    let bv = x_bat[off];
                    let r = (a - bv).abs() / a.abs().max(1e-4);
                    assert!(
                        r < 1e-3,
                        "batch={batch} h={h} i={i} mismatch: seq={a} bat={bv} (rel {r:.3e})"
                    );
                    if i >= rope_dim {
                        // Tail must be unchanged from the input.
                        let orig = x_all[off];
                        assert!(
                            (orig - bv).abs() < 1e-6,
                            "tail dim {i} should be unchanged: orig={orig} bat={bv}"
                        );
                    }
                }
            }
        }
    }

    /// T162 phase 9a — batched split_qg_per_head must match B sequential calls.
    #[test]
    #[cfg(feature = "gpu-tests")]
    fn split_qg_per_head_batched_f32_matches_singles() {
        let backend = metal_backend();
        let n_q = 16usize;
        let head_dim = 128usize;
        let b = 4usize;
        let qg_size = n_q * 2 * head_dim;
        let q_size = n_q * head_dim;

        // Build B different qg rows.
        let mut qg_all = vec![0.0f32; b * qg_size];
        for batch in 0..b {
            for i in 0..qg_size {
                qg_all[batch * qg_size + i] = ((i as f32 + 1.0 + batch as f32 * 7.0) * 0.013).sin();
            }
        }
        let qg_buf = backend.alloc_shared(b * qg_size * 4).unwrap();
        let q_seq_buf = backend.alloc_shared(b * q_size * 4).unwrap();
        let gate_seq_buf = backend.alloc_shared(b * q_size * 4).unwrap();
        let q_batch_buf = backend.alloc_shared(b * q_size * 4).unwrap();
        let gate_batch_buf = backend.alloc_shared(b * q_size * 4).unwrap();
        unsafe {
            std::ptr::copy_nonoverlapping(
                qg_all.as_ptr(),
                qg_buf.contents() as *mut f32,
                b * qg_size,
            );
        }

        // Reference: B sequential calls.
        for batch in 0..b {
            let qg_b = backend.alloc_shared(qg_size * 4).unwrap();
            let q_b = backend.alloc_shared(q_size * 4).unwrap();
            let gate_b = backend.alloc_shared(q_size * 4).unwrap();
            unsafe {
                std::ptr::copy_nonoverlapping(
                    qg_all[batch * qg_size..(batch + 1) * qg_size].as_ptr(),
                    qg_b.contents() as *mut f32,
                    qg_size,
                );
            }
            split_qg_per_head_f32(backend, &qg_b, &q_b, &gate_b, n_q, head_dim).unwrap();
            backend.drain();
            unsafe {
                std::ptr::copy_nonoverlapping(
                    q_b.contents() as *const f32,
                    (q_seq_buf.contents() as *mut f32).add(batch * q_size),
                    q_size,
                );
                std::ptr::copy_nonoverlapping(
                    gate_b.contents() as *const f32,
                    (gate_seq_buf.contents() as *mut f32).add(batch * q_size),
                    q_size,
                );
            }
        }

        // Batched call.
        split_qg_per_head_batched_f32(
            backend,
            &qg_buf,
            &q_batch_buf,
            &gate_batch_buf,
            n_q,
            head_dim,
            b,
        )
        .unwrap();
        backend.drain();

        let q_seq = unsafe {
            std::slice::from_raw_parts(q_seq_buf.contents() as *const f32, b * q_size).to_vec()
        };
        let q_bat = unsafe {
            std::slice::from_raw_parts(q_batch_buf.contents() as *const f32, b * q_size).to_vec()
        };
        let gate_seq = unsafe {
            std::slice::from_raw_parts(gate_seq_buf.contents() as *const f32, b * q_size).to_vec()
        };
        let gate_bat = unsafe {
            std::slice::from_raw_parts(gate_batch_buf.contents() as *const f32, b * q_size).to_vec()
        };
        for i in 0..b * q_size {
            assert!(
                (q_seq[i] - q_bat[i]).abs() < 1e-6,
                "Q[{i}] mismatch: seq={} bat={}",
                q_seq[i],
                q_bat[i]
            );
            assert!(
                (gate_seq[i] - gate_bat[i]).abs() < 1e-6,
                "Gate[{i}] mismatch: seq={} bat={}",
                gate_seq[i],
                gate_bat[i]
            );
        }
    }

    /// T162 phase 9a — batched sigmoid_mul_inplace must match B sequential calls.
    #[test]
    #[cfg(feature = "gpu-tests")]
    fn sigmoid_mul_inplace_batched_f32_matches_singles() {
        let backend = metal_backend();
        let n = 2048usize;
        let b = 4usize;

        let mut x_all = vec![0.0f32; b * n];
        let mut g_all = vec![0.0f32; b * n];
        for batch in 0..b {
            for i in 0..n {
                x_all[batch * n + i] = ((i as f32 + 1.0 + batch as f32 * 3.0) * 0.011).sin();
                g_all[batch * n + i] = ((i as f32 + 1.0 + batch as f32 * 5.0) * 0.017).cos() * 0.5;
            }
        }

        // Reference: B sequential calls (each gets its own copy of x).
        let mut y_seq = x_all.clone();
        for batch in 0..b {
            let x_b = backend.alloc_shared(n * 4).unwrap();
            let g_b = backend.alloc_shared(n * 4).unwrap();
            unsafe {
                std::ptr::copy_nonoverlapping(
                    y_seq[batch * n..(batch + 1) * n].as_ptr(),
                    x_b.contents() as *mut f32,
                    n,
                );
                std::ptr::copy_nonoverlapping(
                    g_all[batch * n..(batch + 1) * n].as_ptr(),
                    g_b.contents() as *mut f32,
                    n,
                );
            }
            sigmoid_mul_inplace_f32(backend, &x_b, &g_b, n).unwrap();
            backend.drain();
            unsafe {
                std::ptr::copy_nonoverlapping(
                    x_b.contents() as *const f32,
                    y_seq[batch * n..(batch + 1) * n].as_mut_ptr(),
                    n,
                );
            }
        }

        // Batched call.
        let x_buf = backend.alloc_shared(b * n * 4).unwrap();
        let g_buf = backend.alloc_shared(b * n * 4).unwrap();
        unsafe {
            std::ptr::copy_nonoverlapping(x_all.as_ptr(), x_buf.contents() as *mut f32, b * n);
            std::ptr::copy_nonoverlapping(g_all.as_ptr(), g_buf.contents() as *mut f32, b * n);
        }
        sigmoid_mul_inplace_batched_f32(backend, &x_buf, &g_buf, n, b).unwrap();
        backend.drain();

        let y_bat =
            unsafe { std::slice::from_raw_parts(x_buf.contents() as *const f32, b * n).to_vec() };
        for i in 0..b * n {
            let r = (y_seq[i] - y_bat[i]).abs() / y_seq[i].abs().max(1e-4);
            assert!(
                r < 1e-3,
                "[{i}] mismatch: seq={} bat={} (rel {:.3e})",
                y_seq[i],
                y_bat[i],
                r
            );
        }
    }
}
