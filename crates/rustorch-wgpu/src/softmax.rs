//! Row-wise softmax on the GPU.
//!
//! Computes `softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))` along
//! the last axis of an `[B, K]` matrix. One workgroup per row.
//!
//! v1: 1-pass numerically stable softmax. Each workgroup:
//!   1. Cooperatively finds row max.
//!   2. Computes exp(x_i - max), accumulates into row sum.
//!   3. Writes exp / sum to output.

use crate::backend::WgpuBackend;
use crate::cache::PipelineKey;
use crate::error::WgpuError;
use crate::storage::WgpuStorage;
use rustorch_core::tensor::dtype::Dtype;

const BLOCK: u32 = 256;

fn softmax_wgsl() -> String {
    format!(
        r#"
@group(0) @binding(0) var<storage, read>  inp: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> params: array<vec4<u32>, 1>;

var<workgroup> shared_max: array<f32, {BLOCK}u>;
var<workgroup> shared_sum: array<f32, {BLOCK}u>;

@compute @workgroup_size({BLOCK})
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id)        wgid: vec3<u32>,
) {{
    let b = params[0].x;
    let k = params[0].y;
    let row = wgid.x;
    if (row >= b) {{ return; }}

    // Phase 1: row max
    var local_max: f32 = -3.4e38;
    var i: u32 = lid.x;
    loop {{
        if (i >= k) {{ break; }}
        local_max = max(local_max, inp[row * k + i]);
        i = i + {BLOCK}u;
    }}
    shared_max[lid.x] = local_max;
    workgroupBarrier();
    var stride: u32 = {BLOCK}u / 2u;
    loop {{
        if (stride == 0u) {{ break; }}
        if (lid.x < stride) {{
            shared_max[lid.x] = max(shared_max[lid.x], shared_max[lid.x + stride]);
        }}
        workgroupBarrier();
        stride = stride / 2u;
    }}
    let row_max = shared_max[0];

    // Phase 2: sum of exp(x - max)
    var local_sum: f32 = 0.0;
    i = lid.x;
    loop {{
        if (i >= k) {{ break; }}
        local_sum = local_sum + exp(inp[row * k + i] - row_max);
        i = i + {BLOCK}u;
    }}
    shared_sum[lid.x] = local_sum;
    workgroupBarrier();
    stride = {BLOCK}u / 2u;
    loop {{
        if (stride == 0u) {{ break; }}
        if (lid.x < stride) {{
            shared_sum[lid.x] = shared_sum[lid.x] + shared_sum[lid.x + stride];
        }}
        workgroupBarrier();
        stride = stride / 2u;
    }}
    let row_sum = shared_sum[0];

    // Phase 3: write out
    i = lid.x;
    loop {{
        if (i >= k) {{ break; }}
        out[row * k + i] = exp(inp[row * k + i] - row_max) / row_sum;
        i = i + {BLOCK}u;
    }}
}}
"#,
        BLOCK = BLOCK
    )
}

fn build_pipeline(backend: &WgpuBackend) -> wgpu::ComputePipeline {
    let module = backend
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("softmax"),
            source: wgpu::ShaderSource::Wgsl(softmax_wgsl().into()),
        });
    backend
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("softmax"),
            layout: None,
            module: &module,
            entry_point: "main",
            compilation_options: Default::default(),
            cache: None,
        })
}

/// WGSL for log_softmax — same 3-phase structure as softmax but the
/// final write is `(x - row_max) - log(row_sum)` instead of
/// `exp(x - row_max) / row_sum`. Mathematically equivalent and more
/// numerically stable for downstream cross-entropy.
fn log_softmax_wgsl() -> String {
    format!(
        r#"
@group(0) @binding(0) var<storage, read>  inp: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> params: array<vec4<u32>, 1>;

var<workgroup> shared_max: array<f32, {BLOCK}u>;
var<workgroup> shared_sum: array<f32, {BLOCK}u>;

@compute @workgroup_size({BLOCK})
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id)        wgid: vec3<u32>,
) {{
    let b = params[0].x;
    let k = params[0].y;
    let row = wgid.x;
    if (row >= b) {{ return; }}

    // Phase 1: row max
    var local_max: f32 = -3.4e38;
    var i: u32 = lid.x;
    loop {{
        if (i >= k) {{ break; }}
        local_max = max(local_max, inp[row * k + i]);
        i = i + {BLOCK}u;
    }}
    shared_max[lid.x] = local_max;
    workgroupBarrier();
    var stride: u32 = {BLOCK}u / 2u;
    loop {{
        if (stride == 0u) {{ break; }}
        if (lid.x < stride) {{
            shared_max[lid.x] = max(shared_max[lid.x], shared_max[lid.x + stride]);
        }}
        workgroupBarrier();
        stride = stride / 2u;
    }}
    let row_max = shared_max[0];

    // Phase 2: sum of exp(x - max)
    var local_sum: f32 = 0.0;
    i = lid.x;
    loop {{
        if (i >= k) {{ break; }}
        local_sum = local_sum + exp(inp[row * k + i] - row_max);
        i = i + {BLOCK}u;
    }}
    shared_sum[lid.x] = local_sum;
    workgroupBarrier();
    stride = {BLOCK}u / 2u;
    loop {{
        if (stride == 0u) {{ break; }}
        if (lid.x < stride) {{
            shared_sum[lid.x] = shared_sum[lid.x] + shared_sum[lid.x + stride];
        }}
        workgroupBarrier();
        stride = stride / 2u;
    }}
    let log_z = log(shared_sum[0]);

    // Phase 3: write out
    i = lid.x;
    loop {{
        if (i >= k) {{ break; }}
        out[row * k + i] = (inp[row * k + i] - row_max) - log_z;
        i = i + {BLOCK}u;
    }}
}}
"#,
        BLOCK = BLOCK
    )
}

fn build_log_softmax_pipeline(backend: &WgpuBackend) -> wgpu::ComputePipeline {
    let module = backend
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("log_softmax"),
            source: wgpu::ShaderSource::Wgsl(log_softmax_wgsl().into()),
        });
    backend
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("log_softmax"),
            layout: None,
            module: &module,
            entry_point: "main",
            compilation_options: Default::default(),
            cache: None,
        })
}

/// Row-wise log-softmax: `out[r, j] = (x[r, j] - max_r) - log(sum_r exp(x - max_r))`.
/// Numerically more stable than `log(softmax(x))` for cross-entropy.
pub fn log_softmax_rows(
    backend: &WgpuBackend,
    inp: &WgpuStorage,
    b: usize,
    k: usize,
) -> Result<WgpuStorage, WgpuError> {
    if inp.numel != b * k {
        return Err(WgpuError::ShapeMismatch(format!(
            "log_softmax_rows: numel {} != b*k {}",
            inp.numel,
            b * k
        )));
    }
    if inp.dtype != Dtype::F32 {
        return Err(WgpuError::UnsupportedDtype(inp.dtype));
    }

    let key = PipelineKey {
        op: "log_softmax",
        dtype: "f32",
        variant: "row",
    };
    let pipeline = backend
        .cache
        .get_or_insert_with(key, || build_log_softmax_pipeline(backend));

    let out = WgpuStorage::allocate(&backend.device, b * k, Dtype::F32)?;
    let params_data = [b as u32, k as u32, 0_u32, 0_u32];
    let meta = backend.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("log_softmax-meta"),
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
            label: Some("log_softmax"),
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
            label: Some("log_softmax"),
        });
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("log_softmax"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.dispatch_workgroups(b as u32, 1, 1);
    }
    backend.queue.submit(Some(encoder.finish()));
    Ok(out)
}

/// Row-wise softmax: input `[b, k]` → output `[b, k]`. Numerically stable.
pub fn softmax_rows(
    backend: &WgpuBackend,
    inp: &WgpuStorage,
    b: usize,
    k: usize,
) -> Result<WgpuStorage, WgpuError> {
    if inp.numel != b * k {
        return Err(WgpuError::ShapeMismatch(format!(
            "softmax_rows: numel {} != b*k {}",
            inp.numel,
            b * k
        )));
    }
    if inp.dtype != Dtype::F32 {
        return Err(WgpuError::UnsupportedDtype(inp.dtype));
    }

    let key = PipelineKey {
        op: "softmax",
        dtype: "f32",
        variant: "row",
    };
    let pipeline = backend
        .cache
        .get_or_insert_with(key, || build_pipeline(backend));

    let out = WgpuStorage::allocate(&backend.device, b * k, Dtype::F32)?;
    let params_data = [b as u32, k as u32, 0_u32, 0_u32];
    let meta = backend.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("softmax-meta"),
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
            label: Some("softmax"),
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
            label: Some("softmax"),
        });
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("softmax"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.dispatch_workgroups(b as u32, 1, 1);
    }
    backend.queue.submit(Some(encoder.finish()));
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wgsl_contains_three_phases() {
        let s = softmax_wgsl();
        assert!(s.contains("Phase 1"));
        assert!(s.contains("Phase 2"));
        assert!(s.contains("Phase 3"));
        assert!(s.contains("workgroupBarrier"));
    }
}

#[cfg(all(test, feature = "gpu-tests"))]
mod gpu_tests {
    use super::*;
    use crate::transfer::{to_cpu, to_gpu};
    use rustorch_core::tensor::tensor_impl::Tensor;

    fn run(data: &[f32], b: usize, k: usize) -> Vec<f32> {
        let backend = WgpuBackend::new_blocking().expect("init");
        let t = Tensor::from_vec([b, k], data.to_vec()).unwrap();
        let g = to_gpu(&backend, &t).unwrap();
        let r = softmax_rows(&backend, &g, b, k).unwrap();
        let c = to_cpu(&backend, &r, vec![b, k]).unwrap();
        c.as_slice::<f32>().unwrap().to_vec()
    }

    fn cpu_softmax(data: &[f32], b: usize, k: usize) -> Vec<f32> {
        let mut out = vec![0.0_f32; b * k];
        for r in 0..b {
            let row = &data[r * k..(r + 1) * k];
            let m = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = row.iter().map(|x| (x - m).exp()).collect();
            let s: f32 = exps.iter().sum();
            for (j, e) in exps.iter().enumerate() {
                out[r * k + j] = e / s;
            }
        }
        out
    }

    #[test]
    fn softmax_row_sums_to_one() {
        let r = run(&[1.0, 2.0, 3.0, 4.0], 1, 4);
        let s: f32 = r.iter().sum();
        assert!((s - 1.0).abs() < 1e-5);
    }

    #[test]
    fn softmax_parity_2x4() {
        let data = vec![1.0_f32, 2.0, 3.0, 4.0, -1.0, 0.0, 1.0, 2.0];
        let g = run(&data, 2, 4);
        let c = cpu_softmax(&data, 2, 4);
        for (a, b) in g.iter().zip(&c) {
            assert!((a - b).abs() < 1e-5, "{a} vs {b}");
        }
    }

    #[test]
    fn log_softmax_parity_2x4() {
        let backend = WgpuBackend::new_blocking().expect("init");
        let data = vec![1.0_f32, 2.0, 3.0, 4.0, -1.0, 0.0, 1.0, 2.0];
        let t = Tensor::from_vec([2_usize, 4], data.clone()).unwrap();
        let g = to_gpu(&backend, &t).unwrap();
        let r = log_softmax_rows(&backend, &g, 2, 4).unwrap();
        let c = to_cpu(&backend, &r, vec![2, 4]).unwrap();
        // CPU reference: log_softmax(row) = (row - max) - log(sum exp(row - max))
        let mut expected = vec![0.0_f32; 8];
        for row_idx in 0..2 {
            let row = &data[row_idx * 4..(row_idx + 1) * 4];
            let m = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let z: f32 = row.iter().map(|x| (x - m).exp()).sum();
            for j in 0..4 {
                expected[row_idx * 4 + j] = (row[j] - m) - z.ln();
            }
        }
        for (g, e) in c.as_slice::<f32>().unwrap().iter().zip(&expected) {
            assert!((g - e).abs() < 1e-5, "{g} vs {e}");
        }
    }

    #[test]
    fn softmax_large_row() {
        let k = 1024;
        let data: Vec<f32> = (0..k).map(|i| (i as f32 * 0.01).sin()).collect();
        let g = run(&data, 1, k);
        let c = cpu_softmax(&data, 1, k);
        for (a, b) in g.iter().zip(&c) {
            assert!((a - b).abs() < 1e-5);
        }
    }
}
