//! Matmul GPU kernel — tiled GEMM with workgroup-shared memory.
//!
//! Computes `C = A @ B` where `A` is `[M, K]`, `B` is `[K, N]`,
//! `C` is `[M, N]`, all F32, all contiguous.
//!
//! Tile size 16×16. Each workgroup computes a single 16×16 output
//! tile by iterating K in chunks of 16, loading the relevant tiles
//! of A and B into workgroup-shared memory, then accumulating.
//!
//! v1: square-tile kernel only (no shape-specialised variants yet).

use crate::backend::WgpuBackend;
use crate::cache::PipelineKey;
use crate::error::WgpuError;
use crate::storage::WgpuStorage;
use rustorch_core::tensor::dtype::Dtype;

const TILE: u32 = 16;

/// WGSL source for the tiled matmul. `meta = [M, K, N, _]`.
fn matmul_tiled_wgsl() -> String {
    format!(
        r#"
@group(0) @binding(0) var<storage, read>  lhs: array<f32>;
@group(0) @binding(1) var<storage, read>  rhs: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> params: array<vec4<u32>, 1>;

var<workgroup> a_tile: array<array<f32, {TILE}u>, {TILE}u>;
var<workgroup> b_tile: array<array<f32, {TILE}u>, {TILE}u>;

@compute @workgroup_size({TILE}, {TILE}, 1)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id)  lid: vec3<u32>,
    @builtin(workgroup_id)         wgid: vec3<u32>,
) {{
    let m = params[0].x;
    let k = params[0].y;
    let n = params[0].z;

    let row = wgid.y * {TILE}u + lid.y;   // 0..M
    let col = wgid.x * {TILE}u + lid.x;   // 0..N

    var acc: f32 = 0.0;
    let n_tiles = (k + {TILE}u - 1u) / {TILE}u;

    for (var t: u32 = 0u; t < n_tiles; t = t + 1u) {{
        let a_col = t * {TILE}u + lid.x;
        let b_row = t * {TILE}u + lid.y;
        if (row < m && a_col < k) {{
            a_tile[lid.y][lid.x] = lhs[row * k + a_col];
        }} else {{
            a_tile[lid.y][lid.x] = 0.0;
        }}
        if (b_row < k && col < n) {{
            b_tile[lid.y][lid.x] = rhs[b_row * n + col];
        }} else {{
            b_tile[lid.y][lid.x] = 0.0;
        }}
        workgroupBarrier();
        for (var i: u32 = 0u; i < {TILE}u; i = i + 1u) {{
            acc = acc + a_tile[lid.y][i] * b_tile[i][lid.x];
        }}
        workgroupBarrier();
    }}

    if (row < m && col < n) {{
        out[row * n + col] = acc;
    }}
}}
"#,
        TILE = TILE
    )
}

fn build_pipeline(backend: &WgpuBackend) -> wgpu::ComputePipeline {
    let module = backend
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("matmul"),
            source: wgpu::ShaderSource::Wgsl(matmul_tiled_wgsl().into()),
        });
    backend
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("matmul"),
            layout: None,
            module: &module,
            entry_point: "main",
            compilation_options: Default::default(),
            cache: None,
        })
}

/// Variant of [`matmul`] with explicit transpose flags. Computes
///
/// ```text
///   C = (transpose_a ? Aᵀ : A) @ (transpose_b ? Bᵀ : B)
/// ```
///
/// Inputs:
/// - `lhs` : `[m, k]` if `!transpose_a`, else `[k, m]`.
/// - `rhs` : `[k, n]` if `!transpose_b`, else `[n, k]`.
/// - `m`, `k`, `n` describe the **logical** GEMM dims (post-transpose).
///
/// Implementation: applies [`crate::transpose2d`] on the host as a
/// pre-pass for any flagged operand, then dispatches the standard
/// tiled matmul. Two extra full-tensor reads vs a fused
/// transpose-aware kernel — trade-off favors code clarity at this
/// scope; a fused variant is a follow-up.
pub fn matmul_with_transposes(
    backend: &WgpuBackend,
    lhs: &WgpuStorage,
    rhs: &WgpuStorage,
    m: usize,
    k: usize,
    n: usize,
    transpose_a: bool,
    transpose_b: bool,
) -> Result<WgpuStorage, WgpuError> {
    use crate::transpose::transpose2d;
    let lhs_actual = if transpose_a {
        // lhs is stored as [k, m] → produce [m, k].
        transpose2d(backend, lhs, k, m)?
    } else {
        lhs.clone()
    };
    let rhs_actual = if transpose_b {
        // rhs is stored as [n, k] → produce [k, n].
        transpose2d(backend, rhs, n, k)?
    } else {
        rhs.clone()
    };
    matmul(backend, &lhs_actual, &rhs_actual, m, k, n)
}

/// `C = A @ B` on the GPU. `lhs` shape `[M, K]`, `rhs` shape `[K, N]`,
/// returns shape `[M, N]`.
pub fn matmul(
    backend: &WgpuBackend,
    lhs: &WgpuStorage,
    rhs: &WgpuStorage,
    m: usize,
    k: usize,
    n: usize,
) -> Result<WgpuStorage, WgpuError> {
    if lhs.numel != m * k || rhs.numel != k * n {
        return Err(WgpuError::ShapeMismatch(format!(
            "matmul: lhs numel {} != m*k {}, rhs numel {} != k*n {}",
            lhs.numel,
            m * k,
            rhs.numel,
            k * n
        )));
    }
    if lhs.dtype != Dtype::F32 || rhs.dtype != Dtype::F32 {
        return Err(WgpuError::UnsupportedDtype(lhs.dtype));
    }

    let key = PipelineKey {
        op: "matmul",
        dtype: "f32",
        variant: "tiled16",
    };
    let pipeline = backend
        .cache
        .get_or_insert_with(key, || build_pipeline(backend));

    let out = WgpuStorage::allocate(&backend.device, m * n, Dtype::F32)?;

    // meta = [M, K, N, _]
    let params_data = [m as u32, k as u32, n as u32, 0_u32];
    let meta = backend.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("matmul-meta"),
        size: 16,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    backend
        .queue
        .write_buffer(&meta, 0, bytemuck::cast_slice(&params_data));

    let bind_group_layout = pipeline.get_bind_group_layout(0);
    let bind_group = backend
        .device
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("matmul"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: lhs.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: rhs.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: out.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: meta.as_entire_binding(),
                },
            ],
        });

    let groups_x = (n as u32).div_ceil(TILE).max(1);
    let groups_y = (m as u32).div_ceil(TILE).max(1);

    let mut encoder = backend
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("matmul"),
        });
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("matmul"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.dispatch_workgroups(groups_x, groups_y, 1);
    }
    backend.queue.submit(Some(encoder.finish()));
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wgsl_source_contains_tile_loop() {
        let src = matmul_tiled_wgsl();
        assert!(src.contains("workgroupBarrier"));
        assert!(src.contains("a_tile"));
        assert!(src.contains("b_tile"));
    }
}

#[cfg(all(test, feature = "gpu-tests"))]
mod gpu_tests {
    use super::*;
    use crate::transfer::{to_cpu, to_gpu};
    use rustorch_core::tensor::tensor_impl::Tensor;

    fn cpu_matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
        let mut out = vec![0.0_f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut s = 0.0_f32;
                for kk in 0..k {
                    s += a[i * k + kk] * b[kk * n + j];
                }
                out[i * n + j] = s;
            }
        }
        out
    }

    #[test]
    fn matmul_2x3_3x2() {
        let backend = WgpuBackend::new_blocking().expect("init");
        let a = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]; // [2, 3]
        let b = vec![7.0_f32, 8.0, 9.0, 10.0, 11.0, 12.0]; // [3, 2]
        let ta = Tensor::from_vec([2_usize, 3], a.clone()).unwrap();
        let tb = Tensor::from_vec([3_usize, 2], b.clone()).unwrap();
        let ga = to_gpu(&backend, &ta).unwrap();
        let gb = to_gpu(&backend, &tb).unwrap();
        let gc = matmul(&backend, &ga, &gb, 2, 3, 2).unwrap();
        let c = to_cpu(&backend, &gc, vec![2, 2]).unwrap();
        let expected = cpu_matmul(&a, &b, 2, 3, 2);
        for (g, e) in c.as_slice::<f32>().unwrap().iter().zip(&expected) {
            assert!((g - e).abs() < 1e-4, "{g} vs {e}");
        }
    }

    #[test]
    fn matmul_64x64_parity() {
        let backend = WgpuBackend::new_blocking().expect("init");
        let m = 64;
        let k = 64;
        let n = 64;
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.01).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.005).collect();
        let ta = Tensor::from_vec([m, k], a.clone()).unwrap();
        let tb = Tensor::from_vec([k, n], b.clone()).unwrap();
        let ga = to_gpu(&backend, &ta).unwrap();
        let gb = to_gpu(&backend, &tb).unwrap();
        let gc = matmul(&backend, &ga, &gb, m, k, n).unwrap();
        let c = to_cpu(&backend, &gc, vec![m, n]).unwrap();
        let expected = cpu_matmul(&a, &b, m, k, n);
        for (g, e) in c.as_slice::<f32>().unwrap().iter().zip(&expected) {
            assert!((g - e).abs() / e.abs().max(1e-3) < 1e-3, "{g} vs {e}");
        }
    }

    /// Helper: CPU reference for `(transpose_a ? Aᵀ : A) @ (transpose_b ? Bᵀ : B)`.
    /// Stored layouts: lhs is `[lhs_dim0, lhs_dim1]`, rhs `[rhs_dim0, rhs_dim1]`.
    #[allow(clippy::too_many_arguments)]
    fn cpu_matmul_t(
        a: &[f32],
        a_dim0: usize,
        a_dim1: usize,
        b: &[f32],
        b_dim0: usize,
        b_dim1: usize,
        ta: bool,
        tb: bool,
    ) -> Vec<f32> {
        let m = if ta { a_dim1 } else { a_dim0 };
        let k = if ta { a_dim0 } else { a_dim1 };
        let n = if tb { b_dim0 } else { b_dim1 };
        // Sanity: K must match.
        let kb = if tb { b_dim1 } else { b_dim0 };
        assert_eq!(k, kb, "K mismatch in cpu_matmul_t");
        let mut out = vec![0.0_f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut s = 0.0_f32;
                for kk in 0..k {
                    let av = if ta {
                        a[kk * a_dim1 + i]
                    } else {
                        a[i * a_dim1 + kk]
                    };
                    let bv = if tb {
                        b[j * b_dim1 + kk]
                    } else {
                        b[kk * b_dim1 + j]
                    };
                    s += av * bv;
                }
                out[i * n + j] = s;
            }
        }
        out
    }

    #[test]
    fn matmul_with_transposes_nt_8x4_4x6() {
        // (no-transpose-A, transpose-B) — A=[8,4], B-stored=[6,4] → B^T=[4,6] → C=[8,6].
        let backend = WgpuBackend::new_blocking().expect("init");
        let (m, k, n) = (8, 4, 6);
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.1).collect();
        let b: Vec<f32> = (0..n * k).map(|i| (i as f32) * 0.2 - 0.5).collect();
        let ta = Tensor::from_vec([m, k], a.clone()).unwrap();
        let tb = Tensor::from_vec([n, k], b.clone()).unwrap();
        let ga = to_gpu(&backend, &ta).unwrap();
        let gb = to_gpu(&backend, &tb).unwrap();
        let gc = matmul_with_transposes(&backend, &ga, &gb, m, k, n, false, true).unwrap();
        let c = to_cpu(&backend, &gc, vec![m, n]).unwrap();
        let expected = cpu_matmul_t(&a, m, k, &b, n, k, false, true);
        for (g, e) in c.as_slice::<f32>().unwrap().iter().zip(&expected) {
            assert!((g - e).abs() / e.abs().max(1e-3) < 1e-3, "{g} vs {e}");
        }
    }

    #[test]
    fn matmul_with_transposes_tn_5x4_5x3() {
        // (transpose-A, no-transpose-B) — A-stored=[5,4]→A^T=[4,5]; B=[5,3] → C=[4,3].
        let backend = WgpuBackend::new_blocking().expect("init");
        let (m, k, n) = (4, 5, 3);
        let a: Vec<f32> = (0..k * m).map(|i| (i as f32) * 0.05).collect(); // stored as [5,4]
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.07 - 0.3).collect();
        let ta = Tensor::from_vec([k, m], a.clone()).unwrap();
        let tb = Tensor::from_vec([k, n], b.clone()).unwrap();
        let ga = to_gpu(&backend, &ta).unwrap();
        let gb = to_gpu(&backend, &tb).unwrap();
        let gc = matmul_with_transposes(&backend, &ga, &gb, m, k, n, true, false).unwrap();
        let c = to_cpu(&backend, &gc, vec![m, n]).unwrap();
        let expected = cpu_matmul_t(&a, k, m, &b, k, n, true, false);
        for (g, e) in c.as_slice::<f32>().unwrap().iter().zip(&expected) {
            assert!((g - e).abs() / e.abs().max(1e-3) < 1e-3, "{g} vs {e}");
        }
    }

    #[test]
    fn matmul_with_transposes_tt_3x4_5x3() {
        // (transpose-A, transpose-B) — A-stored=[4,3]→[3,4]; B-stored=[5,4]→[4,5]; C=[3,5].
        let backend = WgpuBackend::new_blocking().expect("init");
        let (m, k, n) = (3, 4, 5);
        let a: Vec<f32> = (0..k * m).map(|i| (i as f32) * 0.1).collect();
        let b: Vec<f32> = (0..n * k).map(|i| (i as f32) * 0.05).collect();
        let ta = Tensor::from_vec([k, m], a.clone()).unwrap();
        let tb = Tensor::from_vec([n, k], b.clone()).unwrap();
        let ga = to_gpu(&backend, &ta).unwrap();
        let gb = to_gpu(&backend, &tb).unwrap();
        let gc = matmul_with_transposes(&backend, &ga, &gb, m, k, n, true, true).unwrap();
        let c = to_cpu(&backend, &gc, vec![m, n]).unwrap();
        let expected = cpu_matmul_t(&a, k, m, &b, n, k, true, true);
        for (g, e) in c.as_slice::<f32>().unwrap().iter().zip(&expected) {
            assert!((g - e).abs() / e.abs().max(1e-3) < 1e-3, "{g} vs {e}");
        }
    }

    #[test]
    fn matmul_non_tile_aligned() {
        // 7×11 @ 11×5 — none of M, K, N is a multiple of 16.
        let backend = WgpuBackend::new_blocking().expect("init");
        let (m, k, n) = (7, 11, 5);
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32 + 1.0) * 0.1).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32 + 2.0) * 0.05).collect();
        let ta = Tensor::from_vec([m, k], a.clone()).unwrap();
        let tb = Tensor::from_vec([k, n], b.clone()).unwrap();
        let ga = to_gpu(&backend, &ta).unwrap();
        let gb = to_gpu(&backend, &tb).unwrap();
        let gc = matmul(&backend, &ga, &gb, m, k, n).unwrap();
        let c = to_cpu(&backend, &gc, vec![m, n]).unwrap();
        let expected = cpu_matmul(&a, &b, m, k, n);
        for (g, e) in c.as_slice::<f32>().unwrap().iter().zip(&expected) {
            assert!((g - e).abs() / e.abs().max(1e-3) < 1e-3);
        }
    }
}
