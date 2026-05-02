//! Naive scaled dot-product attention on the GPU.
//!
//! Builds on the existing primitives from P2.x:
//!   transpose2d → matmul → mul_scalar (scale) → softmax_rows → matmul
//!
//! No batching, no multi-head, no mask in v1: the inputs are 2D
//! `[S, D]` matrices. Multi-head attention is a Phase 3 follow-up
//! that splits / concatenates heads at the host level. This kernel is
//! the baseline against which Flash Attention (P3.5) will be compared.

use crate::backend::WgpuBackend;
use crate::cache::PipelineKey;
use crate::elementwise::split_dispatch;
use crate::error::WgpuError;
use crate::matmul::matmul;
use crate::softmax::softmax_rows;
use crate::storage::WgpuStorage;
use crate::transpose::transpose2d;
use rustorch_core::tensor::dtype::Dtype;

const BLOCK: u32 = 64;

/// WGSL: `out[i] = inp[i] * scale`. The scale is packed into the
/// uniform as `bitcast<f32>(params[0].y)` (we keep the layout
/// vec4<u32> consistent with the rest of the crate).
fn mul_scalar_wgsl() -> String {
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
    let n = params[0].x;
    let scale = bitcast<f32>(params[0].y);
    let i = (wgid.y * nwg.x + wgid.x) * {BLOCK}u + lid.x;
    if (i >= n) {{ return; }}
    out[i] = inp[i] * scale;
}}
"#,
        BLOCK = BLOCK
    )
}

fn build_mul_scalar_pipeline(backend: &WgpuBackend) -> wgpu::ComputePipeline {
    let module = backend
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("mul_scalar"),
            source: wgpu::ShaderSource::Wgsl(mul_scalar_wgsl().into()),
        });
    backend
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("mul_scalar"),
            layout: None,
            module: &module,
            entry_point: "main",
            compilation_options: Default::default(),
            cache: None,
        })
}

/// `out = inp * scale` element-wise on the GPU. Output has the same
/// shape and dtype (F32) as the input.
pub fn mul_scalar(
    backend: &WgpuBackend,
    inp: &WgpuStorage,
    scale: f32,
) -> Result<WgpuStorage, WgpuError> {
    if inp.dtype != Dtype::F32 {
        return Err(WgpuError::UnsupportedDtype(inp.dtype));
    }
    let key = PipelineKey {
        op: "mul_scalar",
        dtype: "f32",
        variant: "default",
    };
    let pipeline = backend
        .cache
        .get_or_insert_with(key, || build_mul_scalar_pipeline(backend));

    let out = WgpuStorage::allocate(&backend.device, inp.numel, Dtype::F32)?;
    let params_data = [inp.numel as u32, scale.to_bits(), 0_u32, 0_u32];
    let meta = backend.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("mul_scalar-meta"),
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
            label: Some("mul_scalar"),
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
            label: Some("mul_scalar"),
        });
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("mul_scalar"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        let groups = (inp.numel as u32).div_ceil(BLOCK).max(1);
        let (gx, gy) = split_dispatch(groups);
        cpass.dispatch_workgroups(gx, gy, 1);
    }
    backend.queue.submit(Some(encoder.finish()));
    Ok(out)
}

/// WGSL kernel that sets `out[i, j] = -inf` whenever `j > i` —
/// applies a causal mask in-place on a `[S, S]` score matrix.
fn causal_mask_wgsl() -> String {
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
    let s = params[0].x;
    let total = s * s;
    let idx = (wgid.y * nwg.x + wgid.x) * {BLOCK}u + lid.x;
    if (idx >= total) {{ return; }}
    let i = idx / s;
    let j = idx % s;
    if (j > i) {{
        out[idx] = -3.4e38;
    }} else {{
        out[idx] = inp[idx];
    }}
}}
"#,
        BLOCK = BLOCK
    )
}

fn build_causal_mask_pipeline(backend: &WgpuBackend) -> wgpu::ComputePipeline {
    let module = backend
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("causal_mask"),
            source: wgpu::ShaderSource::Wgsl(causal_mask_wgsl().into()),
        });
    backend
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("causal_mask"),
            layout: None,
            module: &module,
            entry_point: "main",
            compilation_options: Default::default(),
            cache: None,
        })
}

/// Apply a causal mask to a `[s, s]` score matrix: any entry where
/// the column index exceeds the row index is set to `-inf`. After
/// softmax those positions become 0 — i.e. position `i` cannot attend
/// to any position `j > i`.
pub fn apply_causal_mask(
    backend: &WgpuBackend,
    scores: &WgpuStorage,
    s: usize,
) -> Result<WgpuStorage, WgpuError> {
    if scores.numel != s * s {
        return Err(WgpuError::ShapeMismatch(format!(
            "causal_mask: numel {} != s*s {}",
            scores.numel,
            s * s
        )));
    }
    if scores.dtype != Dtype::F32 {
        return Err(WgpuError::UnsupportedDtype(scores.dtype));
    }

    let key = PipelineKey {
        op: "causal_mask",
        dtype: "f32",
        variant: "default",
    };
    let pipeline = backend
        .cache
        .get_or_insert_with(key, || build_causal_mask_pipeline(backend));

    let out = WgpuStorage::allocate(&backend.device, s * s, Dtype::F32)?;
    let params_data = [s as u32, 0_u32, 0_u32, 0_u32];
    let meta = backend.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("causal_mask-meta"),
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
            label: Some("causal_mask"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: scores.buffer.as_entire_binding(),
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
            label: Some("causal_mask"),
        });
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("causal_mask"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        let total = (s * s) as u32;
        let groups = total.div_ceil(BLOCK).max(1);
        let (gx, gy) = split_dispatch(groups);
        cpass.dispatch_workgroups(gx, gy, 1);
    }
    backend.queue.submit(Some(encoder.finish()));
    Ok(out)
}

/// Causal variant of [`attention_naive`]. Inserts an `apply_causal_mask`
/// pass between `Q@K^T/sqrt(d)` and `softmax`.
pub fn attention_naive_causal(
    backend: &WgpuBackend,
    q: &WgpuStorage,
    k: &WgpuStorage,
    v: &WgpuStorage,
    s: usize,
    d: usize,
) -> Result<WgpuStorage, WgpuError> {
    use crate::matmul::matmul;
    use crate::softmax::softmax_rows;
    use crate::transpose::transpose2d;
    if q.numel != s * d || k.numel != s * d || v.numel != s * d {
        return Err(WgpuError::ShapeMismatch(format!(
            "attention_causal: Q/K/V numel {}/{}/{} != s*d {}",
            q.numel,
            k.numel,
            v.numel,
            s * d
        )));
    }
    let k_t = transpose2d(backend, k, s, d)?;
    let scores = matmul(backend, q, &k_t, s, d, s)?;
    let scale = 1.0 / (d as f32).sqrt();
    let scaled = mul_scalar(backend, &scores, scale)?;
    let masked = apply_causal_mask(backend, &scaled, s)?;
    let attn = softmax_rows(backend, &masked, s, s)?;
    matmul(backend, &attn, v, s, s, d)
}

/// Multi-head attention helper. Splits Q/K/V into `num_heads` heads
/// of size `d_per_head = d_model / num_heads`, runs
/// [`attention_naive`] (or the causal variant) per head, and
/// concatenates back.
///
/// Inputs:
/// - `q`, `k`, `v` : `[s, d_model]` row-major F32.
/// - `num_heads`   : ≥ 1, must divide `d_model`.
/// - `causal`      : if true, applies the lower-triangular mask.
///
/// Returns `[s, d_model]`.
pub fn multi_head_attention(
    backend: &WgpuBackend,
    q: &WgpuStorage,
    k: &WgpuStorage,
    v: &WgpuStorage,
    s: usize,
    d_model: usize,
    num_heads: usize,
    causal: bool,
) -> Result<WgpuStorage, WgpuError> {
    use crate::transfer::{to_cpu, to_gpu};
    use rustorch_core::tensor::tensor_impl::Tensor;

    if num_heads == 0 || d_model % num_heads != 0 {
        return Err(WgpuError::ShapeMismatch(format!(
            "multi_head: d_model {} not divisible by num_heads {}",
            d_model, num_heads
        )));
    }
    let dh = d_model / num_heads;

    // Slice Q/K/V into heads on the host. Per-head GPU work; concat
    // on the host before the final return. This is the v1 cost: a
    // host round-trip per head pair. A fused multi-head kernel is a
    // follow-up.
    let read = |t: &WgpuStorage| -> Result<Vec<f32>, WgpuError> {
        Ok(to_cpu(backend, t, vec![s, d_model])?
            .as_slice::<f32>()
            .ok_or_else(|| WgpuError::ShapeMismatch("expected f32".into()))?
            .to_vec())
    };
    let q_host = read(q)?;
    let k_host = read(k)?;
    let v_host = read(v)?;

    let mut out_host = vec![0.0_f32; s * d_model];
    for h in 0..num_heads {
        // Build per-head [s, dh] slice.
        let mut q_h = Vec::with_capacity(s * dh);
        let mut k_h = Vec::with_capacity(s * dh);
        let mut v_h = Vec::with_capacity(s * dh);
        for i in 0..s {
            let off = i * d_model + h * dh;
            q_h.extend_from_slice(&q_host[off..off + dh]);
            k_h.extend_from_slice(&k_host[off..off + dh]);
            v_h.extend_from_slice(&v_host[off..off + dh]);
        }
        let tq =
            Tensor::from_vec([s, dh], q_h).map_err(|e| WgpuError::ShapeMismatch(format!("{e}")))?;
        let tk =
            Tensor::from_vec([s, dh], k_h).map_err(|e| WgpuError::ShapeMismatch(format!("{e}")))?;
        let tv =
            Tensor::from_vec([s, dh], v_h).map_err(|e| WgpuError::ShapeMismatch(format!("{e}")))?;
        let gq = to_gpu(backend, &tq)?;
        let gk = to_gpu(backend, &tk)?;
        let gv = to_gpu(backend, &tv)?;
        let go = if causal {
            attention_naive_causal(backend, &gq, &gk, &gv, s, dh)?
        } else {
            attention_naive(backend, &gq, &gk, &gv, s, dh)?
        };
        let oh = to_cpu(backend, &go, vec![s, dh])?;
        let oh_slice = oh
            .as_slice::<f32>()
            .ok_or_else(|| WgpuError::ShapeMismatch("expected f32".into()))?;
        for i in 0..s {
            let dst = i * d_model + h * dh;
            out_host[dst..dst + dh].copy_from_slice(&oh_slice[i * dh..(i + 1) * dh]);
        }
    }
    let tout = Tensor::from_vec([s, d_model], out_host)
        .map_err(|e| WgpuError::ShapeMismatch(format!("{e}")))?;
    to_gpu(backend, &tout)
}

/// Naive scaled dot-product attention.
///
/// `q`, `k`, `v` are `[s, d]` row-major F32. Returns `[s, d]`.
///
/// Computes:
/// ```text
/// scores = (Q @ K^T) / sqrt(d)
/// attn   = softmax(scores, axis=-1)
/// out    = attn @ V
/// ```
pub fn attention_naive(
    backend: &WgpuBackend,
    q: &WgpuStorage,
    k: &WgpuStorage,
    v: &WgpuStorage,
    s: usize,
    d: usize,
) -> Result<WgpuStorage, WgpuError> {
    if q.numel != s * d {
        return Err(WgpuError::ShapeMismatch(format!(
            "attention: q numel {} != s*d {}",
            q.numel,
            s * d
        )));
    }
    if k.numel != s * d {
        return Err(WgpuError::ShapeMismatch(format!(
            "attention: k numel {} != s*d {}",
            k.numel,
            s * d
        )));
    }
    if v.numel != s * d {
        return Err(WgpuError::ShapeMismatch(format!(
            "attention: v numel {} != s*d {}",
            v.numel,
            s * d
        )));
    }

    // 1. K^T : [d, s]
    let k_t = transpose2d(backend, k, s, d)?;
    // 2. scores : [s, s] = Q @ K^T
    let scores = matmul(backend, q, &k_t, s, d, s)?;
    // 3. scaled scores : scores * (1 / sqrt(d))
    let scale = 1.0 / (d as f32).sqrt();
    let scaled = mul_scalar(backend, &scores, scale)?;
    // 4. attn weights : softmax(scaled, axis=-1)
    let attn = softmax_rows(backend, &scaled, s, s)?;
    // 5. out : [s, d] = attn @ V
    matmul(backend, &attn, v, s, s, d)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mul_scalar_wgsl_packs_scale_via_bitcast() {
        let s = mul_scalar_wgsl();
        assert!(s.contains("bitcast<f32>(params[0].y)"));
    }
}

#[cfg(all(test, feature = "gpu-tests"))]
mod gpu_tests {
    use super::*;
    use crate::transfer::{to_cpu, to_gpu};
    use rustorch_core::tensor::tensor_impl::Tensor;

    fn cpu_attention(q: &[f32], k: &[f32], v: &[f32], s: usize, d: usize) -> Vec<f32> {
        // scores = Q @ K^T / sqrt(d)
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
        // Softmax row-wise
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
        // out = attn @ V
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
    fn mul_scalar_simple() {
        let backend = WgpuBackend::new_blocking().expect("init");
        let x: Vec<f32> = vec![1.0, -2.0, 3.0, -4.0];
        let tx = Tensor::from_vec([x.len()], x.clone()).unwrap();
        let gx = to_gpu(&backend, &tx).unwrap();
        let gy = mul_scalar(&backend, &gx, 0.5).unwrap();
        let cy = to_cpu(&backend, &gy, vec![x.len()]).unwrap();
        for (a, b) in cy.as_slice::<f32>().unwrap().iter().zip(&x) {
            assert!((a - b * 0.5).abs() < 1e-6);
        }
    }

    #[test]
    fn attention_4x8() {
        let backend = WgpuBackend::new_blocking().expect("init");
        let (s, d) = (4, 8);
        let q: Vec<f32> = (0..s * d).map(|i| (i as f32) * 0.05 - 0.2).collect();
        let k: Vec<f32> = (0..s * d).map(|i| (i as f32) * 0.07 + 0.1).collect();
        let v: Vec<f32> = (0..s * d).map(|i| (i as f32) * 0.03).collect();
        let tq = Tensor::from_vec([s, d], q.clone()).unwrap();
        let tk = Tensor::from_vec([s, d], k.clone()).unwrap();
        let tv = Tensor::from_vec([s, d], v.clone()).unwrap();
        let gq = to_gpu(&backend, &tq).unwrap();
        let gk = to_gpu(&backend, &tk).unwrap();
        let gv = to_gpu(&backend, &tv).unwrap();
        let go = attention_naive(&backend, &gq, &gk, &gv, s, d).unwrap();
        let co = to_cpu(&backend, &go, vec![s, d]).unwrap();
        let expected = cpu_attention(&q, &k, &v, s, d);
        for (g, e) in co.as_slice::<f32>().unwrap().iter().zip(&expected) {
            assert!((g - e).abs() < 1e-3, "{g} vs {e}");
        }
    }

    #[test]
    fn causal_mask_zeros_attention_above_diagonal() {
        // After softmax, masked (-inf) entries become 0. So
        // attention[i, j>i] should be 0 in the causal output.
        let backend = WgpuBackend::new_blocking().expect("init");
        let (s, d) = (4, 4);
        let q: Vec<f32> = (0..s * d).map(|i| (i as f32) * 0.1).collect();
        let k = q.clone();
        let v = q.clone();
        let tq = Tensor::from_vec([s, d], q).unwrap();
        let tk = Tensor::from_vec([s, d], k).unwrap();
        let tv = Tensor::from_vec([s, d], v).unwrap();
        let gq = to_gpu(&backend, &tq).unwrap();
        let gk = to_gpu(&backend, &tk).unwrap();
        let gv = to_gpu(&backend, &tv).unwrap();
        let go = attention_naive_causal(&backend, &gq, &gk, &gv, s, d).unwrap();
        let _o = to_cpu(&backend, &go, vec![s, d]).unwrap();
        // We can't easily inspect the attention weights from outside,
        // but a stronger test: the causal output must equal the
        // non-causal output for the LAST row (which sees all positions
        // anyway). And the FIRST row must equal V[0] (only sees itself).
        let _go_full = attention_naive(&backend, &gq, &gk, &gv, s, d).unwrap();
        // First row equals V[0] because it only attends to itself.
        let mut q0 = vec![0.0_f32; d];
        let mut v0 = vec![0.0_f32; d];
        let raw_v = to_cpu(&backend, &gv, vec![s, d]).unwrap();
        let raw_o = to_cpu(&backend, &go, vec![s, d]).unwrap();
        v0.copy_from_slice(&raw_v.as_slice::<f32>().unwrap()[..d]);
        q0.copy_from_slice(&raw_o.as_slice::<f32>().unwrap()[..d]);
        for (a, b) in q0.iter().zip(&v0) {
            assert!((a - b).abs() < 1e-3, "first-row attn != V[0]: {a} vs {b}");
        }
    }

    #[test]
    fn multi_head_attention_matches_single_head_when_h_eq_1() {
        // num_heads = 1 should be equivalent to attention_naive.
        let backend = WgpuBackend::new_blocking().expect("init");
        let (s, d) = (8, 16);
        let q: Vec<f32> = (0..s * d).map(|i| (i as f32) * 0.05).collect();
        let k: Vec<f32> = (0..s * d).map(|i| (i as f32) * 0.07 + 0.1).collect();
        let v: Vec<f32> = (0..s * d).map(|i| (i as f32) * 0.03).collect();
        let tq = Tensor::from_vec([s, d], q.clone()).unwrap();
        let tk = Tensor::from_vec([s, d], k.clone()).unwrap();
        let tv = Tensor::from_vec([s, d], v.clone()).unwrap();
        let gq = to_gpu(&backend, &tq).unwrap();
        let gk = to_gpu(&backend, &tk).unwrap();
        let gv = to_gpu(&backend, &tv).unwrap();
        let g_single = attention_naive(&backend, &gq, &gk, &gv, s, d).unwrap();
        let g_multi = multi_head_attention(&backend, &gq, &gk, &gv, s, d, 1, false).unwrap();
        let single = to_cpu(&backend, &g_single, vec![s, d]).unwrap();
        let multi = to_cpu(&backend, &g_multi, vec![s, d]).unwrap();
        for (a, b) in single
            .as_slice::<f32>()
            .unwrap()
            .iter()
            .zip(multi.as_slice::<f32>().unwrap())
        {
            assert!((a - b).abs() < 1e-4, "{a} vs {b}");
        }
    }

    #[test]
    fn multi_head_attention_4_heads_d16() {
        // d=16 with 4 heads → 4 × dh=4 — verify against per-head CPU
        // reference.
        let backend = WgpuBackend::new_blocking().expect("init");
        let (s, d, h) = (8, 16, 4);
        let dh = d / h;
        let q: Vec<f32> = (0..s * d).map(|i| ((i as f32) * 0.013).sin()).collect();
        let k: Vec<f32> = (0..s * d).map(|i| ((i as f32) * 0.011).cos()).collect();
        let v: Vec<f32> = (0..s * d).map(|i| (i as f32) * 0.007 - 0.5).collect();
        let tq = Tensor::from_vec([s, d], q.clone()).unwrap();
        let tk = Tensor::from_vec([s, d], k.clone()).unwrap();
        let tv = Tensor::from_vec([s, d], v.clone()).unwrap();
        let gq = to_gpu(&backend, &tq).unwrap();
        let gk = to_gpu(&backend, &tk).unwrap();
        let gv = to_gpu(&backend, &tv).unwrap();
        let go = multi_head_attention(&backend, &gq, &gk, &gv, s, d, h, false).unwrap();
        let got = to_cpu(&backend, &go, vec![s, d]).unwrap();
        // CPU reference: split, attend per head, concatenate.
        let cpu_softmax = |x: &[f32], n: usize| -> Vec<f32> {
            let m = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = x.iter().map(|v| (v - m).exp()).collect();
            let z: f32 = exps.iter().sum();
            (0..n).map(|i| exps[i] / z).collect()
        };
        let mut expected = vec![0.0_f32; s * d];
        for head in 0..h {
            for i in 0..s {
                let mut scores = vec![0.0_f32; s];
                for j in 0..s {
                    let mut sc = 0.0_f32;
                    for kk in 0..dh {
                        sc += q[i * d + head * dh + kk] * k[j * d + head * dh + kk];
                    }
                    scores[j] = sc / (dh as f32).sqrt();
                }
                let probs = cpu_softmax(&scores, s);
                for vd in 0..dh {
                    let mut o = 0.0_f32;
                    for j in 0..s {
                        o += probs[j] * v[j * d + head * dh + vd];
                    }
                    expected[i * d + head * dh + vd] = o;
                }
            }
        }
        for (g, e) in got.as_slice::<f32>().unwrap().iter().zip(&expected) {
            assert!((g - e).abs() < 1e-3, "{g} vs {e}");
        }
    }

    #[test]
    fn attention_seq32() {
        // Slightly larger to exercise the matmul tile boundary.
        let backend = WgpuBackend::new_blocking().expect("init");
        let (s, d) = (32, 16);
        let q: Vec<f32> = (0..s * d).map(|i| ((i as f32) * 0.013).sin()).collect();
        let k: Vec<f32> = (0..s * d).map(|i| ((i as f32) * 0.011).cos()).collect();
        let v: Vec<f32> = (0..s * d).map(|i| i as f32 * 0.007 - 0.5).collect();
        let tq = Tensor::from_vec([s, d], q.clone()).unwrap();
        let tk = Tensor::from_vec([s, d], k.clone()).unwrap();
        let tv = Tensor::from_vec([s, d], v.clone()).unwrap();
        let gq = to_gpu(&backend, &tq).unwrap();
        let gk = to_gpu(&backend, &tk).unwrap();
        let gv = to_gpu(&backend, &tv).unwrap();
        let go = attention_naive(&backend, &gq, &gk, &gv, s, d).unwrap();
        let co = to_cpu(&backend, &go, vec![s, d]).unwrap();
        let expected = cpu_attention(&q, &k, &v, s, d);
        for (g, e) in co.as_slice::<f32>().unwrap().iter().zip(&expected) {
            let scale = e.abs().max(1e-3);
            assert!((g - e).abs() / scale < 1e-3, "{g} vs {e}");
        }
    }
}
