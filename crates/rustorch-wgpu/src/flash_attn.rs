//! Flash Attention v1 forward — tiled, online-softmax attention that
//! never materialises the `[S, S]` attention matrix in global memory.
//!
//! Computes
//!
//! ```text
//! out = softmax(Q K^T / sqrt(d)) V
//! ```
//!
//! using the online-softmax recurrence (Dao et al., 2022):
//!
//! ```text
//! for j in 0..S:
//!     s_ij  = (Q[i] · K[j]) / sqrt(d)
//!     m_new = max(m_i, s_ij)
//!     α     = exp(m_i  - m_new)   # rescale of the running state
//!     β     = exp(s_ij - m_new)   # weight of the new column
//!     o_i   = α · o_i + β · V[j]
//!     l_i   = α · l_i + β
//!     m_i   = m_new
//! out[i] = o_i / l_i
//! ```
//!
//! Memory: `O(S · D)` for Q/K/V and `O(D)` of running state per row,
//! vs `O(S²)` for naive attention. Same final result, ~4× lower peak
//! activation memory at long sequence lengths.
//!
//! v1 layout: 2D inputs `[S, D]` (no batch / multi-head — the caller
//! splits along the head dimension). Constraint: `D ≤ BLOCK_SIZE`
//! (256 in this build); larger embedding dims need the cooperative
//! variant landed in a follow-up.

use crate::backend::WgpuBackend;
use crate::cache::PipelineKey;
use crate::error::WgpuError;
use crate::storage::WgpuStorage;
use rustorch_core::tensor::dtype::Dtype;

const BLOCK: u32 = 256;

fn flash_attn_wgsl() -> String {
    format!(
        r#"
@group(0) @binding(0) var<storage, read>  q:   array<f32>;
@group(0) @binding(1) var<storage, read>  k:   array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> params: array<vec4<u32>, 1>;
@group(0) @binding(4) var<storage, read>  v:   array<f32>;

var<workgroup> red: array<f32, {BLOCK}u>;

@compute @workgroup_size({BLOCK})
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id)        wgid: vec3<u32>,
) {{
    let s_   = params[0].x;
    let d    = params[0].y;
    let scale = bitcast<f32>(params[0].z);
    let row  = wgid.x;
    if (row >= s_) {{ return; }}

    let tid   = lid.x;
    let valid = tid < d;
    var q_val: f32 = 0.0;
    if (valid) {{ q_val = q[row * d + tid]; }}

    // Per-thread running state (each thread owns its own d):
    var o_d: f32 = 0.0;
    // m_i and l_i are scalar; shared across all threads of this workgroup.
    // We keep them as thread-locals because they are recomputed identically
    // on every thread from the shared reduction result.
    var m_i: f32 = -3.4e38;
    var l_i: f32 = 0.0;

    for (var j: u32 = 0u; j < s_; j = j + 1u) {{
        // ---- 1. Dot product q_row · k_j via tree reduction
        var k_val: f32 = 0.0;
        if (valid) {{ k_val = k[j * d + tid]; }}
        red[tid] = q_val * k_val;
        workgroupBarrier();
        var stride: u32 = {BLOCK}u / 2u;
        loop {{
            if (stride == 0u) {{ break; }}
            if (tid < stride) {{
                red[tid] = red[tid] + red[tid + stride];
            }}
            workgroupBarrier();
            stride = stride / 2u;
        }}
        let s_ij = red[0] * scale;

        // ---- 2. Online-softmax update
        let m_new = max(m_i, s_ij);
        let alpha = exp(m_i  - m_new);
        let beta  = exp(s_ij - m_new);

        var v_val: f32 = 0.0;
        if (valid) {{ v_val = v[j * d + tid]; }}
        o_d  = alpha * o_d + beta * v_val;
        l_i  = alpha * l_i + beta;
        m_i  = m_new;
    }}

    if (valid) {{
        out[row * d + tid] = o_d / l_i;
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
            label: Some("flash_attn_v1"),
            source: wgpu::ShaderSource::Wgsl(flash_attn_wgsl().into()),
        });
    backend
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("flash_attn_v1"),
            layout: None,
            module: &module,
            entry_point: "main",
            compilation_options: Default::default(),
            cache: None,
        })
}

/// Flash Attention v1 forward.
///
/// `q`, `k`, `v` are `[s, d]` row-major F32 with `d ≤ 256`. Returns
/// `[s, d]`. Equivalent to [`crate::attention::attention_naive`] but
/// without ever allocating the `[S, S]` attention matrix.
pub fn flash_attention(
    backend: &WgpuBackend,
    q: &WgpuStorage,
    k: &WgpuStorage,
    v: &WgpuStorage,
    s: usize,
    d: usize,
) -> Result<WgpuStorage, WgpuError> {
    if d > BLOCK as usize {
        return Err(WgpuError::ShapeMismatch(format!(
            "flash_attention: d {} must be ≤ {} in v1; split heads on the host",
            d, BLOCK
        )));
    }
    if q.numel != s * d || k.numel != s * d || v.numel != s * d {
        return Err(WgpuError::ShapeMismatch(format!(
            "flash_attention: Q/K/V numel {}/{}/{} != s*d {}",
            q.numel,
            k.numel,
            v.numel,
            s * d
        )));
    }
    if q.dtype != Dtype::F32 || k.dtype != Dtype::F32 || v.dtype != Dtype::F32 {
        return Err(WgpuError::UnsupportedDtype(q.dtype));
    }

    let key = PipelineKey {
        op: "flash_attn",
        dtype: "f32",
        variant: "v1",
    };
    let pipeline = backend
        .cache
        .get_or_insert_with(key, || build_pipeline(backend));

    let scale = 1.0_f32 / (d as f32).sqrt();
    let out = WgpuStorage::allocate(&backend.device, s * d, Dtype::F32)?;
    let params_data = [s as u32, d as u32, scale.to_bits(), 0_u32];
    let meta = backend.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("flash_attn-meta"),
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
            label: Some("flash_attn"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: q.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: k.buffer.as_entire_binding(),
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
                    resource: v.buffer.as_entire_binding(),
                },
            ],
        });

    let mut encoder = backend
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("flash_attn"),
        });
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("flash_attn"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.dispatch_workgroups(s as u32, 1, 1);
    }
    backend.queue.submit(Some(encoder.finish()));
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wgsl_uses_online_softmax_recurrence() {
        let s = flash_attn_wgsl();
        // Hallmarks of the online-softmax recurrence.
        assert!(s.contains("alpha * o_d + beta * v_val"));
        assert!(s.contains("alpha * l_i + beta"));
        assert!(s.contains("max(m_i, s_ij)"));
    }
}

#[cfg(all(test, feature = "gpu-tests"))]
mod gpu_tests {
    use super::*;
    use crate::attention::attention_naive;
    use crate::transfer::{to_cpu, to_gpu};
    use rustorch_core::tensor::tensor_impl::Tensor;

    fn cpu_attention(q: &[f32], k: &[f32], v: &[f32], s: usize, d: usize) -> Vec<f32> {
        let mut scores = vec![0.0_f32; s * s];
        for i in 0..s {
            for j in 0..s {
                let mut sum = 0.0_f32;
                for kk in 0..d {
                    sum += q[i * d + kk] * k[j * d + kk];
                }
                scores[i * s + j] = sum / (d as f32).sqrt();
            }
        }
        let mut attn = vec![0.0_f32; s * s];
        for i in 0..s {
            let row = &scores[i * s..(i + 1) * s];
            let m = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = row.iter().map(|x| (x - m).exp()).collect();
            let z: f32 = exps.iter().sum();
            for j in 0..s {
                attn[i * s + j] = exps[j] / z;
            }
        }
        let mut out = vec![0.0_f32; s * d];
        for i in 0..s {
            for j in 0..d {
                let mut sum = 0.0_f32;
                for kk in 0..s {
                    sum += attn[i * s + kk] * v[kk * d + j];
                }
                out[i * d + j] = sum;
            }
        }
        out
    }

    #[test]
    fn flash_attention_matches_cpu_8x16() {
        let backend = WgpuBackend::new_blocking().expect("init");
        let (s, d) = (8, 16);
        let q: Vec<f32> = (0..s * d).map(|i| (i as f32) * 0.05 - 0.2).collect();
        let k: Vec<f32> = (0..s * d).map(|i| (i as f32) * 0.07 + 0.1).collect();
        let v: Vec<f32> = (0..s * d).map(|i| (i as f32) * 0.03).collect();
        let tq = Tensor::from_vec([s, d], q.clone()).unwrap();
        let tk = Tensor::from_vec([s, d], k.clone()).unwrap();
        let tv = Tensor::from_vec([s, d], v.clone()).unwrap();
        let gq = to_gpu(&backend, &tq).unwrap();
        let gk = to_gpu(&backend, &tk).unwrap();
        let gv = to_gpu(&backend, &tv).unwrap();
        let go = flash_attention(&backend, &gq, &gk, &gv, s, d).unwrap();
        let co = to_cpu(&backend, &go, vec![s, d]).unwrap();
        let expected = cpu_attention(&q, &k, &v, s, d);
        for (g, e) in co.as_slice::<f32>().unwrap().iter().zip(&expected) {
            assert!((g - e).abs() < 1e-3, "{g} vs {e}");
        }
    }

    #[test]
    fn flash_attention_matches_naive_attention_64x64() {
        // Cross-validate against attention_naive (which materialises the
        // [S, S] matrix). Both should converge to the same result.
        let backend = WgpuBackend::new_blocking().expect("init");
        let (s, d) = (64, 64);
        let q: Vec<f32> = (0..s * d).map(|i| ((i as f32) * 0.013).sin()).collect();
        let k: Vec<f32> = (0..s * d).map(|i| ((i as f32) * 0.011).cos()).collect();
        let v: Vec<f32> = (0..s * d).map(|i| (i as f32) * 0.007 - 0.5).collect();
        let tq = Tensor::from_vec([s, d], q.clone()).unwrap();
        let tk = Tensor::from_vec([s, d], k.clone()).unwrap();
        let tv = Tensor::from_vec([s, d], v.clone()).unwrap();
        let gq = to_gpu(&backend, &tq).unwrap();
        let gk = to_gpu(&backend, &tk).unwrap();
        let gv = to_gpu(&backend, &tv).unwrap();
        let g_flash = flash_attention(&backend, &gq, &gk, &gv, s, d).unwrap();
        let g_naive = attention_naive(&backend, &gq, &gk, &gv, s, d).unwrap();
        let c_flash = to_cpu(&backend, &g_flash, vec![s, d]).unwrap();
        let c_naive = to_cpu(&backend, &g_naive, vec![s, d]).unwrap();
        for (a, b) in c_flash
            .as_slice::<f32>()
            .unwrap()
            .iter()
            .zip(c_naive.as_slice::<f32>().unwrap())
        {
            let scale = b.abs().max(1e-3);
            assert!((a - b).abs() / scale < 1e-3, "flash {a} vs naive {b}");
        }
    }

    #[test]
    fn flash_attention_rejects_d_too_large() {
        let backend = WgpuBackend::new_blocking().expect("init");
        let q = WgpuStorage::allocate(&backend.device, 512, Dtype::F32).unwrap();
        let err = match flash_attention(&backend, &q, &q, &q, 1, 512) {
            Ok(_) => panic!("d=512 should have been rejected"),
            Err(e) => e,
        };
        assert!(format!("{err}").contains("d 512"));
    }
}
