//! 2D transpose on the GPU.
//!
//! `out[j, i] = inp[i, j]` where input is `[M, N]` and output `[N, M]`,
//! both row-major contiguous F32. One thread per output element, 2D
//! dispatch to lift the 65535-per-dim cap.

use crate::backend::WgpuBackend;
use crate::cache::PipelineKey;
use crate::elementwise::split_dispatch;
use crate::error::WgpuError;
use crate::storage::WgpuStorage;
use rustorch_core::tensor::dtype::Dtype;

const BLOCK: u32 = 64;

fn transpose2d_wgsl() -> String {
    format!(
        r#"
@group(0) @binding(0) var<storage, read>  inp: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> params: array<vec4<u32>, 1>;

@compute @workgroup_size({BLOCK})
fn main(
    @builtin(workgroup_id)         wgid: vec3<u32>,
    @builtin(local_invocation_id)  lid:  vec3<u32>,
    @builtin(num_workgroups)       nwg:  vec3<u32>,
) {{
    let m = params[0].x;
    let n = params[0].y;
    let total = m * n;

    let idx = (wgid.y * nwg.x + wgid.x) * {BLOCK}u + lid.x;
    if (idx >= total) {{ return; }}

    // input layout : [M, N] row-major → idx = i * N + j
    let i = idx / n;
    let j = idx % n;
    // output layout: [N, M] row-major → dst = j * M + i
    let dst = j * m + i;
    out[dst] = inp[idx];
}}
"#,
        BLOCK = BLOCK
    )
}

fn build_pipeline(backend: &WgpuBackend) -> wgpu::ComputePipeline {
    let module = backend
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("transpose2d"),
            source: wgpu::ShaderSource::Wgsl(transpose2d_wgsl().into()),
        });
    backend
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("transpose2d"),
            layout: None,
            module: &module,
            entry_point: "main",
            compilation_options: Default::default(),
            cache: None,
        })
}

/// Compute `out = inp.transpose()` on the GPU. `inp` is `[m, n]`,
/// `out` is `[n, m]`, both row-major contiguous f32.
pub fn transpose2d(
    backend: &WgpuBackend,
    inp: &WgpuStorage,
    m: usize,
    n: usize,
) -> Result<WgpuStorage, WgpuError> {
    if inp.numel != m * n {
        return Err(WgpuError::ShapeMismatch(format!(
            "transpose2d: numel {} != m*n {}",
            inp.numel,
            m * n
        )));
    }
    if inp.dtype != Dtype::F32 {
        return Err(WgpuError::UnsupportedDtype(inp.dtype));
    }

    let key = PipelineKey {
        op: "transpose2d",
        dtype: "f32",
        variant: "default",
    };
    let pipeline = backend
        .cache
        .get_or_insert_with(key, || build_pipeline(backend));

    let out = WgpuStorage::allocate(&backend.device, m * n, Dtype::F32)?;
    let params_data = [m as u32, n as u32, 0_u32, 0_u32];
    let meta = backend.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("transpose2d-meta"),
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
            label: Some("transpose2d"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: inp.buffer.as_entire_binding(),
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

    let mut encoder = backend
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("transpose2d"),
        });
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("transpose2d"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        let total = (m * n) as u32;
        let groups = total.div_ceil(BLOCK).max(1);
        let (gx, gy) = split_dispatch(groups);
        cpass.dispatch_workgroups(gx, gy, 1);
    }
    backend.queue.submit(Some(encoder.finish()));
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wgsl_uses_2d_dispatch_pattern() {
        let s = transpose2d_wgsl();
        assert!(s.contains("num_workgroups"));
        assert!(s.contains("wgid.y * nwg.x + wgid.x"));
    }
}

#[cfg(all(test, feature = "gpu-tests"))]
mod gpu_tests {
    use super::*;
    use crate::transfer::{to_cpu, to_gpu};
    use rustorch_core::tensor::tensor_impl::Tensor;

    fn cpu_transpose(x: &[f32], m: usize, n: usize) -> Vec<f32> {
        let mut out = vec![0.0_f32; m * n];
        for i in 0..m {
            for j in 0..n {
                out[j * m + i] = x[i * n + j];
            }
        }
        out
    }

    #[test]
    fn transpose_3x4() {
        let backend = WgpuBackend::new_blocking().expect("init");
        let x: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let t = Tensor::from_vec([3_usize, 4], x.clone()).unwrap();
        let g = to_gpu(&backend, &t).unwrap();
        let r = transpose2d(&backend, &g, 3, 4).unwrap();
        let c = to_cpu(&backend, &r, vec![4, 3]).unwrap();
        let expected = cpu_transpose(&x, 3, 4);
        assert_eq!(c.as_slice::<f32>().unwrap(), expected.as_slice());
    }

    #[test]
    fn transpose_round_trip_is_identity() {
        let backend = WgpuBackend::new_blocking().expect("init");
        let (m, n) = (5, 7);
        let x: Vec<f32> = (0..m * n).map(|i| (i as f32) * 0.3 - 1.5).collect();
        let t = Tensor::from_vec([m, n], x.clone()).unwrap();
        let g = to_gpu(&backend, &t).unwrap();
        let g_t = transpose2d(&backend, &g, m, n).unwrap();
        let g_tt = transpose2d(&backend, &g_t, n, m).unwrap();
        let c = to_cpu(&backend, &g_tt, vec![m, n]).unwrap();
        assert_eq!(c.as_slice::<f32>().unwrap(), x.as_slice());
    }
}
