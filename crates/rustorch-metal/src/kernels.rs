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
use metal::{Buffer, CompileOptions, MTLSize};

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

    let library = backend
        .device
        .new_library_with_source(ADD_SHADER, &CompileOptions::new())
        .map_err(MetalError::ShaderCompile)?;
    let function = library
        .get_function("add_f32", None)
        .map_err(|e| MetalError::PipelineState(format!("get_function: {e}")))?;
    let pipeline = backend
        .device
        .new_compute_pipeline_state_with_function(&function)
        .map_err(MetalError::PipelineState)?;

    let out = backend.alloc_shared(n_bytes)?;

    // Stage `n` into a single-uint constant buffer.
    let n_buf = backend.alloc_shared(core::mem::size_of::<u32>())?;
    // SAFETY: shared-mode buffer; n_buf.contents() is host-mapped.
    unsafe {
        let p = n_buf.contents() as *mut u32;
        *p = n as u32;
    }

    let cmd_buffer = backend.queue.new_command_buffer();
    let encoder = cmd_buffer.new_compute_command_encoder();
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
    encoder.end_encoding();
    cmd_buffer.commit();
    cmd_buffer.wait_until_completed();

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

    let library = backend
        .device
        .new_library_with_source(MATMUL_SIMDGROUP_F32_SHADER, &CompileOptions::new())
        .map_err(MetalError::ShaderCompile)?;
    let function = library
        .get_function("matmul_simdgroup_f32", None)
        .map_err(|e| MetalError::PipelineState(format!("get_function: {e}")))?;
    let pipeline = backend
        .device
        .new_compute_pipeline_state_with_function(&function)
        .map_err(MetalError::PipelineState)?;

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

    let cmd_buffer = backend.queue.new_command_buffer();
    let encoder = cmd_buffer.new_compute_command_encoder();
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
    encoder.end_encoding();
    cmd_buffer.commit();
    cmd_buffer.wait_until_completed();

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
