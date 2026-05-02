//! Row-wise LayerNorm on the GPU.
//!
//! `out[r,j] = ((x[r,j] - mean[r]) / sqrt(var[r] + eps)) * gamma[j] + beta[j]`
//! along the last axis of `[B, K]`. One workgroup per row.
//!
//! Two implementations:
//! - [`layernorm_rows`] — 2-pass mean + variance (the textbook
//!   algorithm; numerically fine for typical inputs).
//! - [`layernorm_welford_rows`] — single-pass parallel Welford
//!   recurrence (Chan / Welford merge formula). Better numerical
//!   stability when `K` is large or values span many orders of
//!   magnitude. Each thread maintains a running `(count, mean, M2)`
//!   triple; the workgroup-tree merges them.
//!
//! [`rmsnorm_rows`] is the variant without mean-subtraction or beta
//! (Llama-class LLMs).

use crate::backend::WgpuBackend;
use crate::cache::PipelineKey;
use crate::error::WgpuError;
use crate::storage::WgpuStorage;
use rustorch_core::tensor::dtype::Dtype;

const BLOCK: u32 = 256;

fn layernorm_wgsl() -> String {
    format!(
        r#"
@group(0) @binding(0) var<storage, read>  inp: array<f32>;
@group(0) @binding(1) var<storage, read>  gamma: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> params: array<vec4<u32>, 1>;
@group(0) @binding(4) var<storage, read>  beta: array<f32>;

var<workgroup> shared_acc: array<f32, {BLOCK}u>;

@compute @workgroup_size({BLOCK})
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id)        wgid: vec3<u32>,
) {{
    let b = params[0].x;
    let k = params[0].y;
    let eps = bitcast<f32>(params[0].z);
    let row = wgid.x;
    if (row >= b) {{ return; }}

    // Phase 1: row mean
    var local_sum: f32 = 0.0;
    var i: u32 = lid.x;
    loop {{
        if (i >= k) {{ break; }}
        local_sum = local_sum + inp[row * k + i];
        i = i + {BLOCK}u;
    }}
    shared_acc[lid.x] = local_sum;
    workgroupBarrier();
    var stride: u32 = {BLOCK}u / 2u;
    loop {{
        if (stride == 0u) {{ break; }}
        if (lid.x < stride) {{
            shared_acc[lid.x] = shared_acc[lid.x] + shared_acc[lid.x + stride];
        }}
        workgroupBarrier();
        stride = stride / 2u;
    }}
    let mean = shared_acc[0] / f32(k);
    workgroupBarrier();

    // Phase 2: row variance
    var local_var: f32 = 0.0;
    i = lid.x;
    loop {{
        if (i >= k) {{ break; }}
        let d = inp[row * k + i] - mean;
        local_var = local_var + d * d;
        i = i + {BLOCK}u;
    }}
    shared_acc[lid.x] = local_var;
    workgroupBarrier();
    stride = {BLOCK}u / 2u;
    loop {{
        if (stride == 0u) {{ break; }}
        if (lid.x < stride) {{
            shared_acc[lid.x] = shared_acc[lid.x] + shared_acc[lid.x + stride];
        }}
        workgroupBarrier();
        stride = stride / 2u;
    }}
    let var_ = shared_acc[0] / f32(k);
    let inv_std = 1.0 / sqrt(var_ + eps);

    // Phase 3: normalize + affine
    i = lid.x;
    loop {{
        if (i >= k) {{ break; }}
        let v = (inp[row * k + i] - mean) * inv_std;
        out[row * k + i] = v * gamma[i] + beta[i];
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
            label: Some("layernorm"),
            source: wgpu::ShaderSource::Wgsl(layernorm_wgsl().into()),
        });
    backend
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("layernorm"),
            layout: None,
            module: &module,
            entry_point: "main",
            compilation_options: Default::default(),
            cache: None,
        })
}

/// WGSL for RMSNorm: `out[r, j] = x[r, j] / sqrt(mean(x²) + eps) * gamma[j]`.
/// No mean-subtraction, no beta. One pass for the squared mean.
fn rmsnorm_wgsl() -> String {
    format!(
        r#"
@group(0) @binding(0) var<storage, read>  inp: array<f32>;
@group(0) @binding(1) var<storage, read>  gamma: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> params: array<vec4<u32>, 1>;

var<workgroup> shared_acc: array<f32, {BLOCK}u>;

@compute @workgroup_size({BLOCK})
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id)        wgid: vec3<u32>,
) {{
    let b = params[0].x;
    let k = params[0].y;
    let eps = bitcast<f32>(params[0].z);
    let row = wgid.x;
    if (row >= b) {{ return; }}

    // Phase 1: row sum-of-squares
    var local_sq: f32 = 0.0;
    var i: u32 = lid.x;
    loop {{
        if (i >= k) {{ break; }}
        let v = inp[row * k + i];
        local_sq = local_sq + v * v;
        i = i + {BLOCK}u;
    }}
    shared_acc[lid.x] = local_sq;
    workgroupBarrier();
    var stride: u32 = {BLOCK}u / 2u;
    loop {{
        if (stride == 0u) {{ break; }}
        if (lid.x < stride) {{
            shared_acc[lid.x] = shared_acc[lid.x] + shared_acc[lid.x + stride];
        }}
        workgroupBarrier();
        stride = stride / 2u;
    }}
    let rms_inv = 1.0 / sqrt(shared_acc[0] / f32(k) + eps);

    // Phase 2: normalize + scale
    i = lid.x;
    loop {{
        if (i >= k) {{ break; }}
        out[row * k + i] = inp[row * k + i] * rms_inv * gamma[i];
        i = i + {BLOCK}u;
    }}
}}
"#,
        BLOCK = BLOCK
    )
}

fn build_rmsnorm_pipeline(backend: &WgpuBackend) -> wgpu::ComputePipeline {
    let module = backend
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("rmsnorm"),
            source: wgpu::ShaderSource::Wgsl(rmsnorm_wgsl().into()),
        });
    backend
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("rmsnorm"),
            layout: None,
            module: &module,
            entry_point: "main",
            compilation_options: Default::default(),
            cache: None,
        })
}

/// Row-wise RMSNorm: `out[r, j] = x[r, j] / sqrt(mean(x²) + eps) * gamma[j]`.
/// No mean-subtraction, no bias. Standard in Llama-class LLMs.
pub fn rmsnorm_rows(
    backend: &WgpuBackend,
    inp: &WgpuStorage,
    gamma: &WgpuStorage,
    b: usize,
    k: usize,
    eps: f32,
) -> Result<WgpuStorage, WgpuError> {
    if inp.numel != b * k {
        return Err(WgpuError::ShapeMismatch(format!(
            "rmsnorm: inp numel {} != b*k {}",
            inp.numel,
            b * k
        )));
    }
    if gamma.numel != k {
        return Err(WgpuError::ShapeMismatch(format!(
            "rmsnorm: gamma numel {} != k {}",
            gamma.numel, k
        )));
    }
    if inp.dtype != Dtype::F32 || gamma.dtype != Dtype::F32 {
        return Err(WgpuError::UnsupportedDtype(inp.dtype));
    }

    let key = PipelineKey {
        op: "rmsnorm",
        dtype: "f32",
        variant: "row",
    };
    let pipeline = backend
        .cache
        .get_or_insert_with(key, || build_rmsnorm_pipeline(backend));

    let out = WgpuStorage::allocate(&backend.device, b * k, Dtype::F32)?;
    let params_data = [b as u32, k as u32, eps.to_bits(), 0_u32];
    let meta = backend.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("rmsnorm-meta"),
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
            label: Some("rmsnorm"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: inp.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: gamma.buffer.as_entire_binding(),
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
            label: Some("rmsnorm"),
        });
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("rmsnorm"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.dispatch_workgroups(b as u32, 1, 1);
    }
    backend.queue.submit(Some(encoder.finish()));
    Ok(out)
}

/// WGSL for the parallel Welford LayerNorm.
///
/// Each thread accumulates a per-thread `(count, mean, M2)` triple
/// over its strided slice of the row. The workgroup then merges
/// pairs of triples via the Chan / Welford parallel recurrence:
///
/// ```text
/// delta   = mean_b - mean_a
/// new_n   = n_a + n_b
/// new_m   = mean_a + delta * n_b / new_n
/// new_M2  = M2_a + M2_b + delta² * n_a * n_b / new_n
/// ```
///
/// At the end, `mean_0 / M2_0` are valid for the whole row and a
/// final pass writes the affine-normalised output.
fn layernorm_welford_wgsl() -> String {
    format!(
        r#"
@group(0) @binding(0) var<storage, read>  inp: array<f32>;
@group(0) @binding(1) var<storage, read>  gamma: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> params: array<vec4<u32>, 1>;
@group(0) @binding(4) var<storage, read>  beta: array<f32>;

var<workgroup> shared_n:  array<f32, {BLOCK}u>;
var<workgroup> shared_m:  array<f32, {BLOCK}u>;
var<workgroup> shared_m2: array<f32, {BLOCK}u>;

@compute @workgroup_size({BLOCK})
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id)        wgid: vec3<u32>,
) {{
    let b = params[0].x;
    let k = params[0].y;
    let eps = bitcast<f32>(params[0].z);
    let row = wgid.x;
    if (row >= b) {{ return; }}

    // Per-thread Welford triple.
    var n: f32 = 0.0;
    var mean: f32 = 0.0;
    var m2: f32 = 0.0;
    var i: u32 = lid.x;
    loop {{
        if (i >= k) {{ break; }}
        let v = inp[row * k + i];
        n = n + 1.0;
        let delta = v - mean;
        mean = mean + delta / n;
        let delta2 = v - mean;
        m2 = m2 + delta * delta2;
        i = i + {BLOCK}u;
    }}
    shared_n[lid.x] = n;
    shared_m[lid.x] = mean;
    shared_m2[lid.x] = m2;
    workgroupBarrier();

    // Tree-merge pairs.
    var stride: u32 = {BLOCK}u / 2u;
    loop {{
        if (stride == 0u) {{ break; }}
        if (lid.x < stride) {{
            let n_a = shared_n[lid.x];
            let n_b = shared_n[lid.x + stride];
            let m_a = shared_m[lid.x];
            let m_b = shared_m[lid.x + stride];
            let q_a = shared_m2[lid.x];
            let q_b = shared_m2[lid.x + stride];
            let new_n = n_a + n_b;
            if (new_n > 0.0) {{
                let delta = m_b - m_a;
                shared_m[lid.x] = m_a + delta * n_b / new_n;
                shared_m2[lid.x] = q_a + q_b + delta * delta * n_a * n_b / new_n;
                shared_n[lid.x] = new_n;
            }}
        }}
        workgroupBarrier();
        stride = stride / 2u;
    }}
    let total_n = shared_n[0];
    let mean_row = shared_m[0];
    let var_row = shared_m2[0] / total_n;
    let inv_std = 1.0 / sqrt(var_row + eps);

    // Phase 2: normalize + affine.
    i = lid.x;
    loop {{
        if (i >= k) {{ break; }}
        let v = (inp[row * k + i] - mean_row) * inv_std;
        out[row * k + i] = v * gamma[i] + beta[i];
        i = i + {BLOCK}u;
    }}
}}
"#,
        BLOCK = BLOCK
    )
}

fn build_welford_pipeline(backend: &WgpuBackend) -> wgpu::ComputePipeline {
    let module = backend
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("layernorm_welford"),
            source: wgpu::ShaderSource::Wgsl(layernorm_welford_wgsl().into()),
        });
    backend
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("layernorm_welford"),
            layout: None,
            module: &module,
            entry_point: "main",
            compilation_options: Default::default(),
            cache: None,
        })
}

/// LayerNorm via the single-pass parallel Welford recurrence.
/// Returns the same result as [`layernorm_rows`] within fp32
/// rounding; numerically more stable for very long rows or rows
/// whose values span many orders of magnitude.
pub fn layernorm_welford_rows(
    backend: &WgpuBackend,
    inp: &WgpuStorage,
    gamma: &WgpuStorage,
    beta: &WgpuStorage,
    b: usize,
    k: usize,
    eps: f32,
) -> Result<WgpuStorage, WgpuError> {
    if inp.numel != b * k {
        return Err(WgpuError::ShapeMismatch(format!(
            "layernorm_welford: inp numel {} != b*k {}",
            inp.numel,
            b * k
        )));
    }
    if gamma.numel != k || beta.numel != k {
        return Err(WgpuError::ShapeMismatch(format!(
            "layernorm_welford: gamma/beta numel {} {} != k {}",
            gamma.numel, beta.numel, k
        )));
    }
    if inp.dtype != Dtype::F32 || gamma.dtype != Dtype::F32 || beta.dtype != Dtype::F32 {
        return Err(WgpuError::UnsupportedDtype(inp.dtype));
    }

    let key = PipelineKey {
        op: "layernorm_welford",
        dtype: "f32",
        variant: "row",
    };
    let pipeline = backend
        .cache
        .get_or_insert_with(key, || build_welford_pipeline(backend));

    let out = WgpuStorage::allocate(&backend.device, b * k, Dtype::F32)?;
    let params_data = [b as u32, k as u32, eps.to_bits(), 0_u32];
    let meta = backend.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("layernorm_welford-meta"),
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
            label: Some("layernorm_welford"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: inp.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: gamma.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: out.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: meta.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: beta.buffer.as_entire_binding(),
                },
            ],
        });

    let mut encoder = backend
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("layernorm_welford"),
        });
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("layernorm_welford"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.dispatch_workgroups(b as u32, 1, 1);
    }
    backend.queue.submit(Some(encoder.finish()));
    Ok(out)
}

/// Row-wise LayerNorm: input `[b, k]`, gamma & beta `[k]`, output `[b, k]`.
pub fn layernorm_rows(
    backend: &WgpuBackend,
    inp: &WgpuStorage,
    gamma: &WgpuStorage,
    beta: &WgpuStorage,
    b: usize,
    k: usize,
    eps: f32,
) -> Result<WgpuStorage, WgpuError> {
    if inp.numel != b * k {
        return Err(WgpuError::ShapeMismatch(format!(
            "layernorm: inp numel {} != b*k {}",
            inp.numel,
            b * k
        )));
    }
    if gamma.numel != k || beta.numel != k {
        return Err(WgpuError::ShapeMismatch(format!(
            "layernorm: gamma/beta numel {} {} != k {}",
            gamma.numel, beta.numel, k
        )));
    }
    if inp.dtype != Dtype::F32 || gamma.dtype != Dtype::F32 || beta.dtype != Dtype::F32 {
        return Err(WgpuError::UnsupportedDtype(inp.dtype));
    }

    let key = PipelineKey {
        op: "layernorm",
        dtype: "f32",
        variant: "row",
    };
    let pipeline = backend
        .cache
        .get_or_insert_with(key, || build_pipeline(backend));

    let out = WgpuStorage::allocate(&backend.device, b * k, Dtype::F32)?;
    let params_data = [b as u32, k as u32, eps.to_bits(), 0_u32];
    let meta = backend.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("layernorm-meta"),
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
            label: Some("layernorm"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: inp.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: gamma.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: out.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: meta.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: beta.buffer.as_entire_binding(),
                },
            ],
        });

    let mut encoder = backend
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("layernorm"),
        });
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("layernorm"),
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
    fn wgsl_uses_bitcast_for_eps() {
        let s = layernorm_wgsl();
        assert!(s.contains("bitcast<f32>"));
        assert!(s.contains("inv_std"));
    }
}

#[cfg(all(test, feature = "gpu-tests"))]
mod gpu_tests {
    use super::*;
    use crate::transfer::{to_cpu, to_gpu};
    use rustorch_core::tensor::tensor_impl::Tensor;

    fn cpu_layernorm(
        x: &[f32],
        gamma: &[f32],
        beta: &[f32],
        b: usize,
        k: usize,
        eps: f32,
    ) -> Vec<f32> {
        let mut out = vec![0.0_f32; b * k];
        for r in 0..b {
            let row = &x[r * k..(r + 1) * k];
            let mean: f32 = row.iter().sum::<f32>() / k as f32;
            let var: f32 = row.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / k as f32;
            let inv_std = 1.0 / (var + eps).sqrt();
            for j in 0..k {
                out[r * k + j] = (row[j] - mean) * inv_std * gamma[j] + beta[j];
            }
        }
        out
    }

    #[test]
    fn layernorm_parity_2x4() {
        let backend = WgpuBackend::new_blocking().expect("init");
        let x = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let gamma = vec![1.0_f32, 1.0, 1.0, 1.0];
        let beta = vec![0.0_f32, 0.0, 0.0, 0.0];
        let tx = Tensor::from_vec([2_usize, 4], x.clone()).unwrap();
        let tg = Tensor::from_vec([4_usize], gamma.clone()).unwrap();
        let tb = Tensor::from_vec([4_usize], beta.clone()).unwrap();
        let gx = to_gpu(&backend, &tx).unwrap();
        let gg = to_gpu(&backend, &tg).unwrap();
        let gb = to_gpu(&backend, &tb).unwrap();
        let go = layernorm_rows(&backend, &gx, &gg, &gb, 2, 4, 1e-5).unwrap();
        let c = to_cpu(&backend, &go, vec![2, 4]).unwrap();
        let expected = cpu_layernorm(&x, &gamma, &beta, 2, 4, 1e-5);
        for (g, e) in c.as_slice::<f32>().unwrap().iter().zip(&expected) {
            assert!((g - e).abs() < 1e-4, "{g} vs {e}");
        }
    }

    #[test]
    fn rmsnorm_parity_2x4() {
        let backend = WgpuBackend::new_blocking().expect("init");
        let x = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let gamma = vec![1.0_f32, 1.0, 1.0, 1.0];
        let tx = Tensor::from_vec([2_usize, 4], x.clone()).unwrap();
        let tg = Tensor::from_vec([4_usize], gamma.clone()).unwrap();
        let gx = to_gpu(&backend, &tx).unwrap();
        let gg = to_gpu(&backend, &tg).unwrap();
        let go = rmsnorm_rows(&backend, &gx, &gg, 2, 4, 1e-5).unwrap();
        let c = to_cpu(&backend, &go, vec![2, 4]).unwrap();
        // CPU reference
        let mut expected = vec![0.0_f32; 8];
        for r in 0..2 {
            let row = &x[r * 4..(r + 1) * 4];
            let ms = row.iter().map(|v| v * v).sum::<f32>() / 4.0;
            let inv = 1.0 / (ms + 1e-5).sqrt();
            for j in 0..4 {
                expected[r * 4 + j] = row[j] * inv * gamma[j];
            }
        }
        for (g, e) in c.as_slice::<f32>().unwrap().iter().zip(&expected) {
            assert!((g - e).abs() < 1e-4, "{g} vs {e}");
        }
    }

    #[test]
    fn layernorm_with_affine() {
        let backend = WgpuBackend::new_blocking().expect("init");
        let x: Vec<f32> = (0..32).map(|i| (i as f32) * 0.1).collect();
        let gamma: Vec<f32> = (0..16).map(|i| 1.0 + (i as f32) * 0.05).collect();
        let beta: Vec<f32> = (0..16).map(|i| (i as f32) * 0.01).collect();
        let tx = Tensor::from_vec([2_usize, 16], x.clone()).unwrap();
        let tg = Tensor::from_vec([16_usize], gamma.clone()).unwrap();
        let tb = Tensor::from_vec([16_usize], beta.clone()).unwrap();
        let gx = to_gpu(&backend, &tx).unwrap();
        let gg = to_gpu(&backend, &tg).unwrap();
        let gbb = to_gpu(&backend, &tb).unwrap();
        let go = layernorm_rows(&backend, &gx, &gg, &gbb, 2, 16, 1e-5).unwrap();
        let c = to_cpu(&backend, &go, vec![2, 16]).unwrap();
        let expected = cpu_layernorm(&x, &gamma, &beta, 2, 16, 1e-5);
        for (g, e) in c.as_slice::<f32>().unwrap().iter().zip(&expected) {
            assert!((g - e).abs() < 1e-4, "{g} vs {e}");
        }
    }

    #[test]
    fn layernorm_welford_matches_layernorm_rows() {
        // Same inputs → both kernels must produce the same output
        // within fp32 noise. Welford is more numerically stable but
        // for normal-magnitude inputs the two converge.
        let backend = WgpuBackend::new_blocking().expect("init");
        let (b, k) = (4, 32);
        let x: Vec<f32> = (0..b * k)
            .map(|i| ((i as f32) * 0.013).sin() * 2.0)
            .collect();
        let gamma: Vec<f32> = (0..k).map(|i| 1.0 + (i as f32) * 0.05).collect();
        let beta: Vec<f32> = (0..k).map(|i| (i as f32) * 0.02 - 0.3).collect();
        let tx = Tensor::from_vec([b, k], x).unwrap();
        let tg = Tensor::from_vec([k], gamma).unwrap();
        let tb = Tensor::from_vec([k], beta).unwrap();
        let gx = to_gpu(&backend, &tx).unwrap();
        let gg = to_gpu(&backend, &tg).unwrap();
        let gbb = to_gpu(&backend, &tb).unwrap();
        let g_classic = layernorm_rows(&backend, &gx, &gg, &gbb, b, k, 1e-5).unwrap();
        let g_welford = layernorm_welford_rows(&backend, &gx, &gg, &gbb, b, k, 1e-5).unwrap();
        let c1 = to_cpu(&backend, &g_classic, vec![b, k]).unwrap();
        let c2 = to_cpu(&backend, &g_welford, vec![b, k]).unwrap();
        for (a, b) in c1
            .as_slice::<f32>()
            .unwrap()
            .iter()
            .zip(c2.as_slice::<f32>().unwrap())
        {
            assert!((a - b).abs() < 1e-3, "{a} vs {b}");
        }
    }

    #[test]
    fn layernorm_welford_long_row_stable() {
        // K = 1024 — exercise the cooperative reduction across many
        // BLOCK-strided iterations.
        let backend = WgpuBackend::new_blocking().expect("init");
        let (b, k) = (1, 1024);
        let x: Vec<f32> = (0..k).map(|i| ((i as f32) * 0.001).sin() * 100.0).collect();
        let gamma = vec![1.0_f32; k];
        let beta = vec![0.0_f32; k];
        let tx = Tensor::from_vec([b, k], x.clone()).unwrap();
        let tg = Tensor::from_vec([k], gamma).unwrap();
        let tb = Tensor::from_vec([k], beta).unwrap();
        let gx = to_gpu(&backend, &tx).unwrap();
        let gg = to_gpu(&backend, &tg).unwrap();
        let gbb = to_gpu(&backend, &tb).unwrap();
        let g = layernorm_welford_rows(&backend, &gx, &gg, &gbb, b, k, 1e-5).unwrap();
        let c = to_cpu(&backend, &g, vec![b, k]).unwrap();
        let raw = c.as_slice::<f32>().unwrap();
        // After LayerNorm, mean(out) ≈ 0 and var(out) ≈ 1.
        let mean: f32 = raw.iter().sum::<f32>() / k as f32;
        let var: f32 = raw.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / k as f32;
        assert!(mean.abs() < 1e-3, "mean = {mean}");
        assert!((var - 1.0).abs() < 1e-2, "var = {var}");
    }
}
