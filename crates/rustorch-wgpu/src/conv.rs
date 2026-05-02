//! Conv2d on the GPU via im2col + GEMM.
//!
//! Strategy: reshape the input `[N, C, H, W]` into a column matrix
//! `cols = [N*Hout*Wout, C*kH*kW]`, then compute
//! `out_cols = cols @ weight^T` where the weight is `[Cout, C, kH, kW]`
//! flattened to `[Cout, C*kH*kW]`. Finally reshape back to
//! `[N, Cout, Hout, Wout]` (no transpose needed because the matmul
//! already lays things out contiguously).
//!
//! v1: stride/pad/dilation = (s, p, 1). No bias. F32 only.

use crate::backend::WgpuBackend;
use crate::cache::PipelineKey;
use crate::error::WgpuError;
use crate::matmul::matmul;
use crate::storage::WgpuStorage;
use rustorch_core::tensor::dtype::Dtype;

const BLOCK: u32 = 64;

/// WGSL source for im2col.
///
/// `params` layout (4 vec4<u32>):
/// - `[0]` = (N, C, H, W)
/// - `[1]` = (kH, kW, sH, sW)
/// - `[2]` = (pH, pW, Hout, Wout)
/// - `[3]` = (total, dH, dW, _)
///
/// where `total = N · Hout · Wout · (C · kH · kW)` and `dH/dW` are the
/// dilation factors (1 = no dilation).
fn im2col_wgsl() -> String {
    format!(
        r#"
@group(0) @binding(0) var<storage, read>  inp: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> params: array<vec4<u32>, 4>;

@compute @workgroup_size({BLOCK})
fn main(
    @builtin(workgroup_id)         wgid: vec3<u32>,
    @builtin(local_invocation_id)  lid:  vec3<u32>,
    @builtin(num_workgroups)       nwg:  vec3<u32>,
) {{
    let n_   = params[0].x;
    let c    = params[0].y;
    let h    = params[0].z;
    let w    = params[0].w;
    let kh   = params[1].x;
    let kw   = params[1].y;
    let sh_  = params[1].z;
    let sw   = params[1].w;
    let ph   = params[2].x;
    let pw   = params[2].y;
    let hout = params[2].z;
    let wout = params[2].w;
    let total = params[3].x;
    let dh   = params[3].y;
    let dw   = params[3].z;

    let idx = (wgid.y * nwg.x + wgid.x) * {BLOCK}u + lid.x;
    if (idx >= total) {{ return; }}

    let cols   = c * kh * kw;
    let row    = idx / cols;
    let col    = idx % cols;

    let ow = row % wout;
    let oh = (row / wout) % hout;
    let n_idx = row / (hout * wout);

    let kj = col % kw;
    let ki = (col / kw) % kh;
    let cc = col / (kh * kw);

    // Dilated kernel position in the input: ki·dh, kj·dw.
    let ih_signed: i32 = i32(oh) * i32(sh_) + i32(ki) * i32(dh) - i32(ph);
    let iw_signed: i32 = i32(ow) * i32(sw) + i32(kj) * i32(dw) - i32(pw);

    var v: f32 = 0.0;
    if (ih_signed >= 0 && ih_signed < i32(h) && iw_signed >= 0 && iw_signed < i32(w)) {{
        let ih = u32(ih_signed);
        let iw = u32(iw_signed);
        let src = ((n_idx * c + cc) * h + ih) * w + iw;
        v = inp[src];
    }}
    out[idx] = v;
}}
"#,
        BLOCK = BLOCK
    )
}

fn build_im2col_pipeline(backend: &WgpuBackend) -> wgpu::ComputePipeline {
    let module = backend
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("im2col"),
            source: wgpu::ShaderSource::Wgsl(im2col_wgsl().into()),
        });
    backend
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("im2col"),
            layout: None,
            module: &module,
            entry_point: "main",
            compilation_options: Default::default(),
            cache: None,
        })
}

/// Conv2d configuration.
#[derive(Debug, Clone, Copy)]
pub struct Conv2dCfg {
    /// Kernel height.
    pub kh: usize,
    /// Kernel width.
    pub kw: usize,
    /// Stride height.
    pub sh: usize,
    /// Stride width.
    pub sw: usize,
    /// Padding height (zero-pad).
    pub ph: usize,
    /// Padding width (zero-pad).
    pub pw: usize,
    /// Dilation height (default 1; >1 spaces the kernel taps apart).
    pub dh: usize,
    /// Dilation width (default 1).
    pub dw: usize,
    /// Number of groups for grouped convolution. `groups = 1` is the
    /// standard dense conv; `groups = c_in` is depthwise; in between
    /// is "grouped" (e.g. ResNeXt). `c_in` and `c_out` must both be
    /// divisible by `groups`.
    pub groups: usize,
}

impl Default for Conv2dCfg {
    fn default() -> Self {
        Conv2dCfg {
            kh: 3,
            kw: 3,
            sh: 1,
            sw: 1,
            ph: 0,
            pw: 0,
            dh: 1,
            dw: 1,
            groups: 1,
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn dispatch_im2col(
    backend: &WgpuBackend,
    inp: &WgpuStorage,
    n: usize,
    c: usize,
    h: usize,
    w: usize,
    cfg: Conv2dCfg,
    hout: usize,
    wout: usize,
) -> Result<WgpuStorage, WgpuError> {
    let cols = c * cfg.kh * cfg.kw;
    let rows = n * hout * wout;
    let total = rows * cols;
    let out = WgpuStorage::allocate(&backend.device, total, Dtype::F32)?;

    let key = PipelineKey {
        op: "im2col",
        dtype: "f32",
        variant: "default",
    };
    let pipeline = backend
        .cache
        .get_or_insert_with(key, || build_im2col_pipeline(backend));

    let params_data: [u32; 16] = [
        n as u32,
        c as u32,
        h as u32,
        w as u32,
        cfg.kh as u32,
        cfg.kw as u32,
        cfg.sh as u32,
        cfg.sw as u32,
        cfg.ph as u32,
        cfg.pw as u32,
        hout as u32,
        wout as u32,
        total as u32,
        cfg.dh as u32,
        cfg.dw as u32,
        0,
    ];
    let meta = backend.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("im2col-meta"),
        size: 64,
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
            label: Some("im2col"),
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
            label: Some("im2col"),
        });
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("im2col"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        let groups = (total as u32).div_ceil(BLOCK).max(1);
        let (gx, gy) = crate::elementwise::split_dispatch(groups);
        cpass.dispatch_workgroups(gx, gy, 1);
    }
    backend.queue.submit(Some(encoder.finish()));
    Ok(out)
}

/// Permute `[a, b, c]` flat buffer to `[b, a, c]` on the GPU. Used to
/// reshape the matmul output from `[N*Hout*Wout, Cout]` to
/// `[N, Cout, Hout*Wout]` (then a final view as `[N, Cout, Hout, Wout]`).
fn permute_nhwc_to_nchw_wgsl() -> String {
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
    // params = [N, Spatial(=Hout*Wout), Cout, Total]
    let n_     = params[0].x;
    let spatial = params[0].y;
    let cout   = params[0].z;
    let total  = params[0].w;

    let idx = (wgid.y * nwg.x + wgid.x) * {BLOCK}u + lid.x;
    if (idx >= total) {{ return; }}

    // input layout : [N*Spatial, Cout]   (row-major)
    // output layout: [N, Cout, Spatial]
    let row = idx / cout;        // 0..N*Spatial
    let co  = idx % cout;        // 0..Cout
    let s   = row % spatial;
    let nn  = row / spatial;

    let dst = (nn * cout + co) * spatial + s;
    out[dst] = inp[idx];
}}
"#,
        BLOCK = BLOCK
    )
}

fn build_permute_pipeline(backend: &WgpuBackend) -> wgpu::ComputePipeline {
    let module = backend
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("permute_nhwc_to_nchw"),
            source: wgpu::ShaderSource::Wgsl(permute_nhwc_to_nchw_wgsl().into()),
        });
    backend
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("permute_nhwc_to_nchw"),
            layout: None,
            module: &module,
            entry_point: "main",
            compilation_options: Default::default(),
            cache: None,
        })
}

fn dispatch_permute(
    backend: &WgpuBackend,
    inp: &WgpuStorage,
    n: usize,
    cout: usize,
    spatial: usize,
) -> Result<WgpuStorage, WgpuError> {
    let total = n * cout * spatial;
    let out = WgpuStorage::allocate(&backend.device, total, Dtype::F32)?;
    let key = PipelineKey {
        op: "permute_nhwc_to_nchw",
        dtype: "f32",
        variant: "default",
    };
    let pipeline = backend
        .cache
        .get_or_insert_with(key, || build_permute_pipeline(backend));
    let params_data: [u32; 4] = [n as u32, spatial as u32, cout as u32, total as u32];
    let meta = backend.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("permute-meta"),
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
            label: Some("permute_nhwc_to_nchw"),
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
            label: Some("permute_nhwc_to_nchw"),
        });
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("permute_nhwc_to_nchw"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        let groups = (total as u32).div_ceil(BLOCK).max(1);
        let (gx, gy) = crate::elementwise::split_dispatch(groups);
        cpass.dispatch_workgroups(gx, gy, 1);
    }
    backend.queue.submit(Some(encoder.finish()));
    Ok(out)
}

/// Helper that transposes a `[Cout, K]` weight into a `[K, Cout]` matrix
/// using the same permute kernel (Cout = batch dim, spatial = K=1, …).
/// Done here as a CPU-side trick: instead of a transpose kernel, we
/// just call matmul with the natural `[M, K] @ [K, N]` shape directly,
/// where K = `C*kH*kW` and N = `Cout`. We build the second operand as
/// `weight^T` of shape `[C*kH*kW, Cout]`. The CPU helper does that.
///
/// For now we keep this as a CPU step: callers pass weight already in
/// `[K, Cout]` layout. The high-level helper [`conv2d_forward`] does
/// the transpose on the host before uploading. v2 will add a GPU
/// transpose kernel.
#[allow(dead_code)]
fn _unused_transpose_marker() {}

/// `conv2d_forward(input, weight, cfg) → output` on the GPU.
///
/// Compute `(Hout, Wout)` from the input shape and a [`Conv2dCfg`].
/// Honours `dh / dw` (dilation) and the standard
/// `H_out = (H + 2·pH − dH·(kH − 1) − 1) / sH + 1` formula.
pub fn output_shape(h: usize, w: usize, cfg: Conv2dCfg) -> (usize, usize) {
    let kh_eff = cfg.dh * (cfg.kh.saturating_sub(1)) + 1;
    let kw_eff = cfg.dw * (cfg.kw.saturating_sub(1)) + 1;
    let hout = (h + 2 * cfg.ph).saturating_sub(kh_eff) / cfg.sh + 1;
    let wout = (w + 2 * cfg.pw).saturating_sub(kw_eff) / cfg.sw + 1;
    (hout, wout)
}

/// `input`  : `[N, C, H, W]`.
/// `weight` : `[K, Cout]` where `K = C * kH * kW` (i.e. weight already
///   transposed). Use [`transpose_weight`] on the host to convert from
///   the conventional `[Cout, C, kH, kW]` layout.
/// Returns: `(WgpuStorage of shape [N, Cout, Hout, Wout], Hout, Wout)`.
///
/// Honours `cfg.dh`, `cfg.dw` (dilation) and `cfg.groups` (grouped
/// conv via host-side splitting; v1 cost: one round-trip per group).
#[allow(clippy::too_many_arguments)]
pub fn conv2d_forward(
    backend: &WgpuBackend,
    input: &WgpuStorage,
    weight_kt: &WgpuStorage,
    n: usize,
    c: usize,
    h: usize,
    w: usize,
    cout: usize,
    cfg: Conv2dCfg,
) -> Result<(WgpuStorage, usize, usize), WgpuError> {
    let (hout, wout) = output_shape(h, w, cfg);
    if input.numel != n * c * h * w {
        return Err(WgpuError::ShapeMismatch(format!(
            "conv2d: input numel {} != N*C*H*W {}",
            input.numel,
            n * c * h * w
        )));
    }
    if cfg.groups == 0 {
        return Err(WgpuError::ShapeMismatch(
            "conv2d: groups must be ≥ 1".to_string(),
        ));
    }
    if c % cfg.groups != 0 || cout % cfg.groups != 0 {
        return Err(WgpuError::ShapeMismatch(format!(
            "conv2d: c {} and cout {} must both be divisible by groups {}",
            c, cout, cfg.groups
        )));
    }

    // ===== Fast path: groups = 1 =====
    if cfg.groups == 1 {
        let k = c * cfg.kh * cfg.kw;
        if weight_kt.numel != k * cout {
            return Err(WgpuError::ShapeMismatch(format!(
                "conv2d: weight^T numel {} != K*Cout {}",
                weight_kt.numel,
                k * cout
            )));
        }
        let cols = dispatch_im2col(backend, input, n, c, h, w, cfg, hout, wout)?;
        let rows = n * hout * wout;
        let mm_out = matmul(backend, &cols, weight_kt, rows, k, cout)?;
        let permuted = dispatch_permute(backend, &mm_out, n, cout, hout * wout)?;
        return Ok((permuted, hout, wout));
    }

    // ===== Grouped conv (groups > 1) =====
    // Strategy: split input on its C dim into `groups` slices of
    // c/g channels each, split weight similarly, run conv2d_forward
    // on each pair, concatenate the outputs on the Cout dim. v1
    // does this on the host — fused-kernel grouped conv is a
    // follow-up optimisation.
    use crate::transfer::{to_cpu, to_gpu};
    use rustorch_core::tensor::tensor_impl::Tensor;

    let g = cfg.groups;
    let c_per = c / g;
    let cout_per = cout / g;
    let k_per = c_per * cfg.kh * cfg.kw;
    if weight_kt.numel != k_per * cout {
        return Err(WgpuError::ShapeMismatch(format!(
            "conv2d (grouped): weight^T numel {} != (k/g)*Cout {}",
            weight_kt.numel,
            k_per * cout
        )));
    }

    let input_host = to_cpu(backend, input, vec![n, c, h, w])?;
    let input_raw = input_host
        .as_slice::<f32>()
        .ok_or_else(|| WgpuError::ShapeMismatch("input not f32".into()))?;
    let weight_host = to_cpu(backend, weight_kt, vec![k_per, cout])?;
    let weight_raw = weight_host
        .as_slice::<f32>()
        .ok_or_else(|| WgpuError::ShapeMismatch("weight not f32".into()))?;

    // Build per-group GPU storages by slicing the host buffers.
    let mut out_per_group: Vec<Tensor> = Vec::with_capacity(g);
    let mut cfg_inner = cfg;
    cfg_inner.groups = 1;
    for grp in 0..g {
        // Input slice [n, c_per, h, w].
        let mut x_g = Vec::with_capacity(n * c_per * h * w);
        for nn in 0..n {
            for cc in 0..c_per {
                let src_c = grp * c_per + cc;
                let off = (nn * c + src_c) * h * w;
                x_g.extend_from_slice(&input_raw[off..off + h * w]);
            }
        }
        // Weight slice — weight is laid out as [K, Cout]. We want the
        // sub-block [k_per, cout_per] at K rows in [0..k_per] and Cout
        // columns in [grp*cout_per, (grp+1)*cout_per].
        let mut w_g = Vec::with_capacity(k_per * cout_per);
        for kk in 0..k_per {
            for co in 0..cout_per {
                w_g.push(weight_raw[kk * cout + grp * cout_per + co]);
            }
        }
        let tx = Tensor::from_vec([n, c_per, h, w], x_g)
            .map_err(|e| WgpuError::ShapeMismatch(format!("{e}")))?;
        let tw = Tensor::from_vec([k_per, cout_per], w_g)
            .map_err(|e| WgpuError::ShapeMismatch(format!("{e}")))?;
        let gx = to_gpu(backend, &tx)?;
        let gw = to_gpu(backend, &tw)?;
        let (gy, _, _) = conv2d_forward(backend, &gx, &gw, n, c_per, h, w, cout_per, cfg_inner)?;
        let host = to_cpu(backend, &gy, vec![n, cout_per, hout, wout])?;
        out_per_group.push(host);
    }

    // Concatenate per-group outputs on the Cout axis.
    let mut out_full = vec![0.0_f32; n * cout * hout * wout];
    for (grp, host) in out_per_group.iter().enumerate() {
        let raw = host
            .as_slice::<f32>()
            .ok_or_else(|| WgpuError::ShapeMismatch("group out not f32".into()))?;
        for nn in 0..n {
            for co in 0..cout_per {
                let dst_co = grp * cout_per + co;
                let src_off = (nn * cout_per + co) * hout * wout;
                let dst_off = (nn * cout + dst_co) * hout * wout;
                out_full[dst_off..dst_off + hout * wout]
                    .copy_from_slice(&raw[src_off..src_off + hout * wout]);
            }
        }
    }
    let tout = Tensor::from_vec([n, cout, hout, wout], out_full)
        .map_err(|e| WgpuError::ShapeMismatch(format!("{e}")))?;
    Ok((to_gpu(backend, &tout)?, hout, wout))
}

/// Host-side helper: convert weight `[Cout, C, kH, kW]` (row-major) to
/// `[K, Cout]` where `K = C*kH*kW`, suitable for [`conv2d_forward`].
pub fn transpose_weight(w: &[f32], cout: usize, c: usize, kh: usize, kw: usize) -> Vec<f32> {
    let k = c * kh * kw;
    let mut out = vec![0.0_f32; k * cout];
    for co in 0..cout {
        for kk in 0..k {
            // src layout : [Cout, K] flattened as out[co * K + kk]
            // dst layout : [K, Cout] flattened as out[kk * Cout + co]
            out[kk * cout + co] = w[co * k + kk];
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn im2col_wgsl_contains_branches() {
        let s = im2col_wgsl();
        assert!(s.contains("ih_signed"));
        assert!(s.contains("iw_signed"));
        assert!(s.contains("inp[src]"));
    }

    #[test]
    fn transpose_weight_round_trip() {
        // [Cout=2, C=1, kH=2, kW=2] -> K=4
        let w = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let t = transpose_weight(&w, 2, 1, 2, 2);
        // out[kk * 2 + co] : at kk=0 co=0 -> w[0*4+0]=1, co=1 -> w[1*4+0]=5
        assert_eq!(t[0], 1.0);
        assert_eq!(t[1], 5.0);
        assert_eq!(t[2], 2.0);
        assert_eq!(t[3], 6.0);
    }
}

#[cfg(all(test, feature = "gpu-tests"))]
mod gpu_tests {
    use super::*;
    use crate::transfer::{to_cpu, to_gpu};
    use rustorch_core::tensor::tensor_impl::Tensor;

    #[allow(clippy::too_many_arguments)]
    fn cpu_conv2d(
        x: &[f32],
        w: &[f32],
        n: usize,
        c: usize,
        h: usize,
        wd: usize,
        cout: usize,
        kh: usize,
        kw: usize,
        sh: usize,
        sw: usize,
        ph: usize,
        pw: usize,
    ) -> (Vec<f32>, usize, usize) {
        let hout = (h + 2 * ph).saturating_sub(kh) / sh + 1;
        let wout = (wd + 2 * pw).saturating_sub(kw) / sw + 1;
        let mut out = vec![0.0_f32; n * cout * hout * wout];
        for nn in 0..n {
            for co in 0..cout {
                for oh in 0..hout {
                    for ow in 0..wout {
                        let mut s = 0.0_f32;
                        for cc in 0..c {
                            for ki in 0..kh {
                                for kj in 0..kw {
                                    let ih = oh as isize * sh as isize + ki as isize - ph as isize;
                                    let iw = ow as isize * sw as isize + kj as isize - pw as isize;
                                    if ih >= 0 && ih < h as isize && iw >= 0 && iw < wd as isize {
                                        let xv =
                                            x[((nn * c + cc) * h + ih as usize) * wd + iw as usize];
                                        let wv = w[((co * c + cc) * kh + ki) * kw + kj];
                                        s += xv * wv;
                                    }
                                }
                            }
                        }
                        out[((nn * cout + co) * hout + oh) * wout + ow] = s;
                    }
                }
            }
        }
        (out, hout, wout)
    }

    #[test]
    fn conv2d_3x3_no_pad() {
        let backend = WgpuBackend::new_blocking().expect("init");
        let n = 1;
        let c = 1;
        let h = 5;
        let wd = 5;
        let cout = 1;
        let kh = 3;
        let kw = 3;
        let x: Vec<f32> = (0..n * c * h * wd).map(|i| i as f32 + 1.0).collect();
        let w: Vec<f32> = vec![1.0, 0.0, -1.0, 1.0, 0.0, -1.0, 1.0, 0.0, -1.0];
        let (expected, ho, wo) = cpu_conv2d(&x, &w, n, c, h, wd, cout, kh, kw, 1, 1, 0, 0);
        let weight_t = transpose_weight(&w, cout, c, kh, kw);
        let tx = Tensor::from_vec([n, c, h, wd], x).unwrap();
        let tw = Tensor::from_vec([c * kh * kw, cout], weight_t).unwrap();
        let gx = to_gpu(&backend, &tx).unwrap();
        let gw = to_gpu(&backend, &tw).unwrap();
        let cfg = Conv2dCfg {
            kh,
            kw,
            sh: 1,
            sw: 1,
            ph: 0,
            pw: 0,
            ..Default::default()
        };
        let (gy, ho2, wo2) = conv2d_forward(&backend, &gx, &gw, n, c, h, wd, cout, cfg).unwrap();
        assert_eq!((ho, wo), (ho2, wo2));
        let y = to_cpu(&backend, &gy, vec![n, cout, ho, wo]).unwrap();
        for (g, e) in y.as_slice::<f32>().unwrap().iter().zip(&expected) {
            assert!((g - e).abs() < 1e-4, "{g} vs {e}");
        }
    }

    #[test]
    fn conv2d_3x3_pad1_stride1() {
        let backend = WgpuBackend::new_blocking().expect("init");
        let (n, c, h, wd, cout, kh, kw) = (2, 3, 7, 7, 4, 3, 3);
        let x: Vec<f32> = (0..n * c * h * wd).map(|i| (i as f32) * 0.05).collect();
        let w: Vec<f32> = (0..cout * c * kh * kw)
            .map(|i| (i as f32) * 0.01 - 0.1)
            .collect();
        let (expected, ho, wo) = cpu_conv2d(&x, &w, n, c, h, wd, cout, kh, kw, 1, 1, 1, 1);
        let weight_t = transpose_weight(&w, cout, c, kh, kw);
        let tx = Tensor::from_vec([n, c, h, wd], x).unwrap();
        let tw = Tensor::from_vec([c * kh * kw, cout], weight_t).unwrap();
        let gx = to_gpu(&backend, &tx).unwrap();
        let gw = to_gpu(&backend, &tw).unwrap();
        let cfg = Conv2dCfg {
            kh,
            kw,
            sh: 1,
            sw: 1,
            ph: 1,
            pw: 1,
            ..Default::default()
        };
        let (gy, _, _) = conv2d_forward(&backend, &gx, &gw, n, c, h, wd, cout, cfg).unwrap();
        let y = to_cpu(&backend, &gy, vec![n, cout, ho, wo]).unwrap();
        for (g, e) in y.as_slice::<f32>().unwrap().iter().zip(&expected) {
            assert!((g - e).abs() < 1e-3, "{g} vs {e}");
        }
    }

    #[test]
    fn conv2d_stride2() {
        let backend = WgpuBackend::new_blocking().expect("init");
        let (n, c, h, wd, cout, kh, kw, sh, sw) = (1, 2, 8, 8, 2, 3, 3, 2, 2);
        let x: Vec<f32> = (0..n * c * h * wd).map(|i| (i as f32) * 0.03).collect();
        let w: Vec<f32> = (0..cout * c * kh * kw).map(|i| (i as f32) * 0.02).collect();
        let (expected, ho, wo) = cpu_conv2d(&x, &w, n, c, h, wd, cout, kh, kw, sh, sw, 0, 0);
        let weight_t = transpose_weight(&w, cout, c, kh, kw);
        let tx = Tensor::from_vec([n, c, h, wd], x).unwrap();
        let tw = Tensor::from_vec([c * kh * kw, cout], weight_t).unwrap();
        let gx = to_gpu(&backend, &tx).unwrap();
        let gw = to_gpu(&backend, &tw).unwrap();
        let cfg = Conv2dCfg {
            kh,
            kw,
            sh,
            sw,
            ph: 0,
            pw: 0,
            ..Default::default()
        };
        let (gy, _, _) = conv2d_forward(&backend, &gx, &gw, n, c, h, wd, cout, cfg).unwrap();
        let y = to_cpu(&backend, &gy, vec![n, cout, ho, wo]).unwrap();
        for (g, e) in y.as_slice::<f32>().unwrap().iter().zip(&expected) {
            assert!((g - e).abs() < 1e-3, "{g} vs {e}");
        }
    }

    #[test]
    fn conv2d_dilation_3x3_d2() {
        // Dilation = 2 on a 3x3 kernel: receptive field becomes 5x5.
        // CPU reference also dilates the kernel taps.
        #[allow(clippy::too_many_arguments)]
        fn cpu_conv2d_dil(
            x: &[f32],
            w: &[f32],
            n: usize,
            c: usize,
            h: usize,
            wd: usize,
            cout: usize,
            kh: usize,
            kw: usize,
            ph: usize,
            pw: usize,
            dh: usize,
            dw: usize,
        ) -> (Vec<f32>, usize, usize) {
            let kh_e = dh * (kh - 1) + 1;
            let kw_e = dw * (kw - 1) + 1;
            let hout = (h + 2 * ph).saturating_sub(kh_e) + 1;
            let wout = (wd + 2 * pw).saturating_sub(kw_e) + 1;
            let mut out = vec![0.0_f32; n * cout * hout * wout];
            for nn in 0..n {
                for co in 0..cout {
                    for oh in 0..hout {
                        for ow in 0..wout {
                            let mut s = 0.0;
                            for cc in 0..c {
                                for ki in 0..kh {
                                    for kj in 0..kw {
                                        let ih = oh as isize + (ki * dh) as isize - ph as isize;
                                        let iw = ow as isize + (kj * dw) as isize - pw as isize;
                                        if ih >= 0 && ih < h as isize && iw >= 0 && iw < wd as isize
                                        {
                                            let xv = x[((nn * c + cc) * h + ih as usize) * wd
                                                + iw as usize];
                                            let wv = w[((co * c + cc) * kh + ki) * kw + kj];
                                            s += xv * wv;
                                        }
                                    }
                                }
                            }
                            out[((nn * cout + co) * hout + oh) * wout + ow] = s;
                        }
                    }
                }
            }
            (out, hout, wout)
        }

        let backend = WgpuBackend::new_blocking().expect("init");
        let (n, c, h, wd, cout, kh, kw) = (1, 2, 9, 9, 2, 3, 3);
        let x: Vec<f32> = (0..n * c * h * wd).map(|i| (i as f32) * 0.05).collect();
        let w: Vec<f32> = (0..cout * c * kh * kw)
            .map(|i| (i as f32) * 0.03 - 0.4)
            .collect();
        let (expected, ho, wo) = cpu_conv2d_dil(&x, &w, n, c, h, wd, cout, kh, kw, 0, 0, 2, 2);
        let weight_t = transpose_weight(&w, cout, c, kh, kw);
        let tx = Tensor::from_vec([n, c, h, wd], x).unwrap();
        let tw = Tensor::from_vec([c * kh * kw, cout], weight_t).unwrap();
        let gx = to_gpu(&backend, &tx).unwrap();
        let gw = to_gpu(&backend, &tw).unwrap();
        let cfg = Conv2dCfg {
            kh,
            kw,
            dh: 2,
            dw: 2,
            ..Default::default()
        };
        let (gy, ho2, wo2) = conv2d_forward(&backend, &gx, &gw, n, c, h, wd, cout, cfg).unwrap();
        assert_eq!((ho, wo), (ho2, wo2));
        let y = to_cpu(&backend, &gy, vec![n, cout, ho, wo]).unwrap();
        for (g, e) in y.as_slice::<f32>().unwrap().iter().zip(&expected) {
            assert!((g - e).abs() < 1e-3, "{g} vs {e}");
        }
    }

    #[test]
    fn conv2d_groups_2_depthwise_like() {
        // c=4, cout=4, groups=2 → 2 sub-convs each [c_in_per=2, cout_per=2].
        // Each group's weight is independent of the other group's input.
        let backend = WgpuBackend::new_blocking().expect("init");
        let (n, c, h, wd, cout, kh, kw, g) = (1, 4, 5, 5, 4, 3, 3, 2);
        let c_per = c / g;
        let cout_per = cout / g;
        let x: Vec<f32> = (0..n * c * h * wd)
            .map(|i| (i as f32) * 0.04 - 0.5)
            .collect();
        // Weight in [Cout, C_in_per_group, kH, kW] = grouped weight layout.
        // Total weight = cout * c_per * kh * kw.
        let w_grouped: Vec<f32> = (0..cout * c_per * kh * kw)
            .map(|i| (i as f32) * 0.03 - 0.2)
            .collect();

        // CPU reference: conv per group, concat on Cout.
        let mut expected = vec![0.0_f32; n * cout * (h - kh + 1) * (wd - kw + 1)];
        let hout = h - kh + 1;
        let wout = wd - kw + 1;
        for grp in 0..g {
            for nn in 0..n {
                for co in 0..cout_per {
                    let dst_co = grp * cout_per + co;
                    for oh in 0..hout {
                        for ow in 0..wout {
                            let mut s = 0.0_f32;
                            for cc in 0..c_per {
                                let src_c = grp * c_per + cc;
                                for ki in 0..kh {
                                    for kj in 0..kw {
                                        let ih = oh + ki;
                                        let iw = ow + kj;
                                        let xv = x[((nn * c + src_c) * h + ih) * wd + iw];
                                        let wv = w_grouped
                                            [(((dst_co * c_per) + cc) * kh + ki) * kw + kj];
                                        s += xv * wv;
                                    }
                                }
                            }
                            expected[((nn * cout + dst_co) * hout + oh) * wout + ow] = s;
                        }
                    }
                }
            }
        }
        // Convert weight to the [K, Cout] layout expected by conv2d_forward.
        // K = c_per * kh * kw (per-group K).
        let weight_t = transpose_weight(&w_grouped, cout, c_per, kh, kw);
        let tx = Tensor::from_vec([n, c, h, wd], x).unwrap();
        let tw = Tensor::from_vec([c_per * kh * kw, cout], weight_t).unwrap();
        let gx = to_gpu(&backend, &tx).unwrap();
        let gw = to_gpu(&backend, &tw).unwrap();
        let cfg = Conv2dCfg {
            kh,
            kw,
            groups: g,
            ..Default::default()
        };
        let (gy, _, _) = conv2d_forward(&backend, &gx, &gw, n, c, h, wd, cout, cfg).unwrap();
        let y = to_cpu(&backend, &gy, vec![n, cout, hout, wout]).unwrap();
        for (g_, e) in y.as_slice::<f32>().unwrap().iter().zip(&expected) {
            assert!((g_ - e).abs() < 1e-3, "{g_} vs {e}");
        }
    }
}
