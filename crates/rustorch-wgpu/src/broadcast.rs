//! Broadcast-aware element-wise ops on the GPU.
//!
//! Implements PyTorch-compatible broadcasting (right-align dims;
//! dimensions of size 1 broadcast over the matching axis) up to 8
//! dimensions. The kernel takes the **output shape** plus per-input
//! **strides** (with stride 0 wherever an input dimension was
//! broadcast) and computes its read indices on the fly.
//!
//! For inputs that already share the same shape, prefer the
//! non-broadcasting [`crate::elementwise::dispatch_binary`] — it has
//! a much smaller WGSL surface and skips the index arithmetic.

use crate::backend::WgpuBackend;
use crate::cache::PipelineKey;
use crate::elementwise::split_dispatch;
use crate::error::WgpuError;
use crate::storage::WgpuStorage;
use rustorch_core::tensor::dtype::Dtype;

/// Maximum number of broadcast dimensions supported.
pub const MAX_BROADCAST_DIMS: usize = 8;
const WORKGROUP_SIZE: u32 = 64;

/// Output of [`broadcast_shape`]: result shape + per-input strides
/// that broadcast (0 strides on size-1 axes).
pub type BroadcastPlan = (Vec<usize>, Vec<usize>, Vec<usize>);

/// PyTorch-style broadcast: right-align `lhs_shape` and `rhs_shape`,
/// pad the shorter with leading 1s, then for each dim require that
/// the two sizes match or one of them is 1.
///
/// Returns `(out_shape, lhs_strides, rhs_strides)` all of length
/// `out_shape.len()`, where strides correspond to a row-major
/// layout of the **original** inputs (not the broadcast view) and
/// are set to 0 for any axis that this input broadcasts.
pub fn broadcast_shape(
    lhs_shape: &[usize],
    rhs_shape: &[usize],
) -> Result<BroadcastPlan, WgpuError> {
    let n = lhs_shape.len().max(rhs_shape.len());
    if n > MAX_BROADCAST_DIMS {
        return Err(WgpuError::ShapeMismatch(format!(
            "broadcast: result rank {} exceeds MAX_BROADCAST_DIMS {}",
            n, MAX_BROADCAST_DIMS
        )));
    }
    let l_pad: Vec<usize> = std::iter::repeat(1)
        .take(n - lhs_shape.len())
        .chain(lhs_shape.iter().copied())
        .collect();
    let r_pad: Vec<usize> = std::iter::repeat(1)
        .take(n - rhs_shape.len())
        .chain(rhs_shape.iter().copied())
        .collect();

    let mut out_shape = vec![0_usize; n];
    for i in 0..n {
        let (a, b) = (l_pad[i], r_pad[i]);
        out_shape[i] = match (a, b) {
            (1, x) | (x, 1) => x,
            (a, b) if a == b => a,
            _ => {
                return Err(WgpuError::ShapeMismatch(format!(
                    "broadcast: incompatible dims at axis {i}: {a} vs {b}"
                )))
            },
        };
    }

    // Compute row-major contiguous strides for each input's *original*
    // (un-padded) shape, then prepend 0s for the leading broadcast
    // axes and replace any size-1 stride with 0.
    fn make_strides(orig: &[usize], padded: &[usize]) -> Vec<usize> {
        // Strides of the original, contiguous shape:
        let mut orig_strides = vec![1_usize; orig.len()];
        for i in (0..orig.len().saturating_sub(1)).rev() {
            orig_strides[i] = orig_strides[i + 1] * orig[i + 1];
        }
        let pad_amount = padded.len() - orig.len();
        let mut out = vec![0_usize; padded.len()];
        for i in 0..orig.len() {
            // If this dim is 1, stride is 0 (broadcast).
            out[pad_amount + i] = if orig[i] == 1 { 0 } else { orig_strides[i] };
        }
        out
    }
    let lhs_strides = make_strides(lhs_shape, &l_pad);
    let rhs_strides = make_strides(rhs_shape, &r_pad);
    Ok((out_shape, lhs_strides, rhs_strides))
}

fn broadcast_wgsl(op: &str) -> String {
    let op_expr = match op {
        "add" => "a + b_v",
        "sub" => "a - b_v",
        "mul" => "a * b_v",
        "div" => "a / b_v",
        _ => panic!("unsupported broadcast op: {op}"),
    };
    // 8 dims max; params layout below uses 4 vec4<u32> (16 u32s).
    format!(
        r#"
@group(0) @binding(0) var<storage, read>  lhs: array<f32>;
@group(0) @binding(1) var<storage, read>  rhs: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> params: array<vec4<u32>, 7>;

// Layout of `params` (32 u32 slots = 8 dims × 4 fields):
//   params[0..1] (8 u32) = out_shape
//   params[2..3] (8 u32) = lhs_strides
//   params[4..5] (8 u32) = rhs_strides
//   params[6].x          = ndim (active dim count, 1..=8)
//   params[6].y          = total numel of out

fn dim(arr_base: u32, i: u32) -> u32 {{
    // arr_base = offset in u32 slots; shape lives at slots arr_base..arr_base+8
    // params is array<vec4<u32>, 7> → slot k of vec4 j is params[j][k%4]
    let v = arr_base + i;
    let j = v / 4u;
    let k = v % 4u;
    return params[j][k];
}}

@compute @workgroup_size({WG})
fn main(
    @builtin(workgroup_id)         wgid: vec3<u32>,
    @builtin(local_invocation_id)  lid:  vec3<u32>,
    @builtin(num_workgroups)       nwg:  vec3<u32>,
) {{
    let total = params[6].y;
    let ndim  = params[6].x;
    let idx = (wgid.y * nwg.x + wgid.x) * {WG}u + lid.x;
    if (idx >= total) {{ return; }}

    // Decode `idx` (linear in row-major out_shape) into a per-axis
    // coordinate, then accumulate the read offsets in lhs/rhs via
    // the per-axis strides.
    var rem: u32 = idx;
    var lhs_off: u32 = 0u;
    var rhs_off: u32 = 0u;
    var coord_acc: array<u32, 8>;
    // Compute coords from last to first (row-major).
    var i: i32 = i32(ndim) - 1;
    loop {{
        if (i < 0) {{ break; }}
        let s = dim(0u, u32(i));
        coord_acc[i] = rem % s;
        rem = rem / s;
        i = i - 1;
    }}
    // Then compute offsets:
    for (var j: u32 = 0u; j < ndim; j = j + 1u) {{
        let lstride = dim(8u, j);
        let rstride = dim(16u, j);
        lhs_off = lhs_off + coord_acc[j] * lstride;
        rhs_off = rhs_off + coord_acc[j] * rstride;
    }}

    let a = lhs[lhs_off];
    let b_v = rhs[rhs_off];
    out[idx] = {op_expr};
}}
"#,
        WG = WORKGROUP_SIZE,
        op_expr = op_expr,
    )
}

fn build_pipeline(backend: &WgpuBackend, op: &'static str) -> wgpu::ComputePipeline {
    let module = backend
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(op),
            source: wgpu::ShaderSource::Wgsl(broadcast_wgsl(op).into()),
        });
    backend
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(op),
            layout: None,
            module: &module,
            entry_point: "main",
            compilation_options: Default::default(),
            cache: None,
        })
}

/// Element-wise binary op with PyTorch-style broadcasting.
///
/// `op` is one of `"add"`, `"sub"`, `"mul"`, `"div"`. Other ops
/// raise `UnsupportedDtype` (used as a generic "not-implemented"
/// channel). Both inputs must be F32 contiguous; the output is
/// also F32.
pub fn dispatch_binary_broadcast(
    backend: &WgpuBackend,
    op: &'static str,
    lhs: &WgpuStorage,
    lhs_shape: &[usize],
    rhs: &WgpuStorage,
    rhs_shape: &[usize],
) -> Result<(WgpuStorage, Vec<usize>), WgpuError> {
    if !matches!(op, "add" | "sub" | "mul" | "div") {
        return Err(WgpuError::UnsupportedDtype(lhs.dtype));
    }
    if lhs.dtype != Dtype::F32 || rhs.dtype != Dtype::F32 {
        return Err(WgpuError::UnsupportedDtype(lhs.dtype));
    }
    let (out_shape, lhs_strides, rhs_strides) = broadcast_shape(lhs_shape, rhs_shape)?;
    let total: usize = out_shape.iter().product();
    let ndim = out_shape.len();

    // Validate input numel matches their shape (callers should guarantee).
    let lhs_numel: usize = lhs_shape.iter().product();
    let rhs_numel: usize = rhs_shape.iter().product();
    if lhs.numel != lhs_numel {
        return Err(WgpuError::ShapeMismatch(format!(
            "broadcast: lhs numel {} != prod(shape) {}",
            lhs.numel, lhs_numel
        )));
    }
    if rhs.numel != rhs_numel {
        return Err(WgpuError::ShapeMismatch(format!(
            "broadcast: rhs numel {} != prod(shape) {}",
            rhs.numel, rhs_numel
        )));
    }

    let key = PipelineKey {
        op,
        dtype: "f32",
        variant: "broadcast",
    };
    let pipeline = backend
        .cache
        .get_or_insert_with(key, || build_pipeline(backend, op));

    let out = WgpuStorage::allocate(&backend.device, total, Dtype::F32)?;

    // Pack params: 28 u32 slots (8 + 8 + 8 + 4) → array<vec4<u32>, 7>.
    let mut params: [u32; 28] = [0; 28];
    for (i, &s) in out_shape.iter().enumerate().take(MAX_BROADCAST_DIMS) {
        params[i] = s as u32;
    }
    for (i, &s) in lhs_strides.iter().enumerate().take(MAX_BROADCAST_DIMS) {
        params[8 + i] = s as u32;
    }
    for (i, &s) in rhs_strides.iter().enumerate().take(MAX_BROADCAST_DIMS) {
        params[16 + i] = s as u32;
    }
    params[24] = ndim as u32;
    params[25] = total as u32;
    let meta = backend.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("broadcast-meta"),
        size: 112, // 28 * 4
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    backend
        .queue
        .write_buffer(&meta, 0, bytemuck::cast_slice(&params));

    let bind_group_layout = pipeline.get_bind_group_layout(0);
    let bind_group = backend
        .device
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(op),
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

    let mut encoder = backend
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(op) });
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(op),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        let groups = (total as u32).div_ceil(WORKGROUP_SIZE).max(1);
        let (gx, gy) = split_dispatch(groups);
        cpass.dispatch_workgroups(gx, gy, 1);
    }
    backend.queue.submit(Some(encoder.finish()));
    Ok((out, out_shape))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn broadcast_shape_pad_left() {
        let (out, _, _) = broadcast_shape(&[3, 4], &[4]).unwrap();
        assert_eq!(out, vec![3, 4]);
    }

    #[test]
    fn broadcast_shape_size_one_axis() {
        let (out, ls, rs) = broadcast_shape(&[2, 1, 4], &[3, 4]).unwrap();
        assert_eq!(out, vec![2, 3, 4]);
        // lhs original strides for [2,1,4]: [4, 4, 1] → after stride-0
        // rule for size-1 dim: [4, 0, 1].
        assert_eq!(ls, vec![4, 0, 1]);
        // rhs padded to 3 dims: [1,3,4] → leading 0, strides [4, 1].
        assert_eq!(rs, vec![0, 4, 1]);
    }

    #[test]
    fn broadcast_shape_incompatible_errors() {
        let err = broadcast_shape(&[2, 3], &[4]).unwrap_err();
        assert!(format!("{err}").contains("incompatible"));
    }

    #[test]
    fn broadcast_shape_too_many_dims_errors() {
        let big = [1_usize; 9];
        let err = broadcast_shape(&big, &[1]).unwrap_err();
        assert!(format!("{err}").contains("MAX_BROADCAST_DIMS"));
    }
}

#[cfg(all(test, feature = "gpu-tests"))]
mod gpu_tests {
    use super::*;
    use crate::transfer::{to_cpu, to_gpu};
    use rustorch_core::tensor::tensor_impl::Tensor;

    fn run_add(
        a: &[f32],
        a_shape: &[usize],
        b: &[f32],
        b_shape: &[usize],
    ) -> (Vec<f32>, Vec<usize>) {
        let backend = WgpuBackend::new_blocking().expect("init");
        let ta = Tensor::from_vec(a_shape.to_vec(), a.to_vec()).unwrap();
        let tb = Tensor::from_vec(b_shape.to_vec(), b.to_vec()).unwrap();
        let ga = to_gpu(&backend, &ta).unwrap();
        let gb = to_gpu(&backend, &tb).unwrap();
        let (gout, out_shape) =
            dispatch_binary_broadcast(&backend, "add", &ga, a_shape, &gb, b_shape).unwrap();
        let out = to_cpu(&backend, &gout, out_shape.clone()).unwrap();
        (out.as_slice::<f32>().unwrap().to_vec(), out_shape)
    }

    #[test]
    fn broadcast_row_vector_to_matrix() {
        // [2, 3] + [3] → [2, 3]
        let a: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b: Vec<f32> = vec![10.0, 20.0, 30.0];
        let (out, shape) = run_add(&a, &[2, 3], &b, &[3]);
        assert_eq!(shape, vec![2, 3]);
        assert_eq!(out, vec![11.0, 22.0, 33.0, 14.0, 25.0, 36.0]);
    }

    #[test]
    fn broadcast_column_vector() {
        // [2, 1] + [1, 3] → [2, 3] (outer add).
        let a = vec![1.0, 2.0];
        let b = vec![10.0, 20.0, 30.0];
        let (out, shape) = run_add(&a, &[2, 1], &b, &[1, 3]);
        assert_eq!(shape, vec![2, 3]);
        assert_eq!(out, vec![11.0, 21.0, 31.0, 12.0, 22.0, 32.0]);
    }

    #[test]
    fn broadcast_4d_with_singletons() {
        // [1, 2, 1, 4] + [3, 1, 5, 1] → [3, 2, 5, 4] (60 elements).
        let backend = WgpuBackend::new_blocking().expect("init");
        let a: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..15).map(|i| (i as f32) * 100.0).collect();
        let ta = Tensor::from_vec([1_usize, 2, 1, 4], a.clone()).unwrap();
        let tb = Tensor::from_vec([3_usize, 1, 5, 1], b.clone()).unwrap();
        let ga = to_gpu(&backend, &ta).unwrap();
        let gb = to_gpu(&backend, &tb).unwrap();
        let (gout, out_shape) =
            dispatch_binary_broadcast(&backend, "add", &ga, &[1, 2, 1, 4], &gb, &[3, 1, 5, 1])
                .unwrap();
        assert_eq!(out_shape, vec![3, 2, 5, 4]);
        let out = to_cpu(&backend, &gout, out_shape).unwrap();
        let got = out.as_slice::<f32>().unwrap();
        // Spot-check a few coordinates against the broadcast definition.
        // out[i, j, k, l] = a[0, j, 0, l] + b[i, 0, k, 0]
        let a_at = |j: usize, l: usize| -> f32 { a[j * 4 + l] };
        let b_at = |i: usize, k: usize| -> f32 { b[i * 5 + k] };
        for i in [0_usize, 2] {
            for j in [0_usize, 1] {
                for k in [0_usize, 4] {
                    for l in [0_usize, 3] {
                        let idx = ((i * 2 + j) * 5 + k) * 4 + l;
                        let expected = a_at(j, l) + b_at(i, k);
                        assert!((got[idx] - expected).abs() < 1e-6);
                    }
                }
            }
        }
    }

    #[test]
    fn broadcast_div_works() {
        // Sanity check that ops other than add are routed correctly.
        let backend = WgpuBackend::new_blocking().expect("init");
        let a = vec![10.0_f32, 20.0, 30.0];
        let b = vec![2.0_f32];
        let ta = Tensor::from_vec([3_usize], a).unwrap();
        let tb = Tensor::from_vec([1_usize], b).unwrap();
        let ga = to_gpu(&backend, &ta).unwrap();
        let gb = to_gpu(&backend, &tb).unwrap();
        let (gout, _) = dispatch_binary_broadcast(&backend, "div", &ga, &[3], &gb, &[1]).unwrap();
        let out = to_cpu(&backend, &gout, vec![3]).unwrap();
        assert_eq!(out.as_slice::<f32>().unwrap(), &[5.0_f32, 10.0, 15.0]);
    }
}
