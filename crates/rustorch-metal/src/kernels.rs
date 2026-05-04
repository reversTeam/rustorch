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

/// **Wide thread-coarsened matmul (16 sg, 2×8 layout)** — same
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
        // 16 simdgroups × 32 threads = 512 threads.
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
}
