//! Row-wise argmax / argmin on the GPU.
//!
//! Treats the input as a `[B, K]` matrix and produces a `[B]` u32
//! output: the index in `0..K` of the row's maximum (or minimum).
//! One workgroup per row, threads cooperate via two workgroup-shared
//! arrays (value + index) for the reduction. Stable: ties resolve to
//! the first (lowest index) occurrence, mirroring `torch.argmax`.

use crate::backend::WgpuBackend;
use crate::cache::PipelineKey;
use crate::error::WgpuError;
use crate::storage::WgpuStorage;
use rustorch_core::tensor::dtype::Dtype;

const BLOCK: u32 = 256;

/// Direction of the argmax-style reduction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ArgKind {
    /// Index of the maximum value (ties → lowest index).
    Max,
    /// Index of the minimum value (ties → lowest index).
    Min,
}

impl ArgKind {
    fn op_name(self) -> &'static str {
        match self {
            ArgKind::Max => "argmax",
            ArgKind::Min => "argmin",
        }
    }
    fn init(self) -> &'static str {
        match self {
            ArgKind::Max => "-3.4e38",
            ArgKind::Min => "3.4e38",
        }
    }
    /// WGSL expression: replaces (acc, idx) with (v, i) iff `cond`.
    fn cond(self) -> &'static str {
        // "first occurrence wins" => strict comparison.
        match self {
            ArgKind::Max => "v > acc",
            ArgKind::Min => "v < acc",
        }
    }
}

fn argmax_wgsl(kind: ArgKind) -> String {
    let init = kind.init();
    let cond = kind.cond();
    format!(
        r#"
@group(0) @binding(0) var<storage, read>  inp: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<u32>;
@group(0) @binding(3) var<uniform> params: array<vec4<u32>, 1>;

var<workgroup> shared_val: array<f32, {BLOCK}u>;
var<workgroup> shared_idx: array<u32, {BLOCK}u>;

@compute @workgroup_size({BLOCK})
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id)        wgid: vec3<u32>,
) {{
    let b = params[0].x;
    let k = params[0].y;
    let row = wgid.x;
    if (row >= b) {{ return; }}

    var acc: f32 = {init};
    var idx: u32 = 0u;
    var i: u32 = lid.x;
    loop {{
        if (i >= k) {{ break; }}
        let v = inp[row * k + i];
        if ({cond}) {{
            acc = v;
            idx = i;
        }}
        i = i + {BLOCK}u;
    }}
    shared_val[lid.x] = acc;
    shared_idx[lid.x] = idx;
    workgroupBarrier();

    var stride: u32 = {BLOCK}u / 2u;
    loop {{
        if (stride == 0u) {{ break; }}
        if (lid.x < stride) {{
            let other_v = shared_val[lid.x + stride];
            let other_i = shared_idx[lid.x + stride];
            let v = other_v;
            let acc2 = shared_val[lid.x];
            // Stable tie-break: prefer the LOWER index when values are
            // tied, so the result mirrors torch.argmax.
            let take_other = ({cond_acc2}) || (v == acc2 && other_i < shared_idx[lid.x]);
            if (take_other) {{
                shared_val[lid.x] = v;
                shared_idx[lid.x] = other_i;
            }}
        }}
        workgroupBarrier();
        stride = stride / 2u;
    }}

    if (lid.x == 0u) {{
        out[row] = shared_idx[0];
    }}
}}
"#,
        BLOCK = BLOCK,
        init = init,
        cond = cond,
        // Reuse the same comparison but on shared_val[lid.x] = acc2.
        cond_acc2 = match kind {
            ArgKind::Max => "v > acc2",
            ArgKind::Min => "v < acc2",
        },
    )
}

fn build_pipeline(backend: &WgpuBackend, kind: ArgKind) -> wgpu::ComputePipeline {
    let module = backend
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(kind.op_name()),
            source: wgpu::ShaderSource::Wgsl(argmax_wgsl(kind).into()),
        });
    backend
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(kind.op_name()),
            layout: None,
            module: &module,
            entry_point: "main",
            compilation_options: Default::default(),
            cache: None,
        })
}

/// Row-wise argmax / argmin on `[b, k]` → `[b]` u32 index buffer.
pub fn argmax_rows(
    backend: &WgpuBackend,
    inp: &WgpuStorage,
    b: usize,
    k: usize,
    kind: ArgKind,
) -> Result<WgpuStorage, WgpuError> {
    if inp.numel != b * k {
        return Err(WgpuError::ShapeMismatch(format!(
            "argmax_rows: numel {} != b*k {}",
            inp.numel,
            b * k
        )));
    }
    if inp.dtype != Dtype::F32 {
        return Err(WgpuError::UnsupportedDtype(inp.dtype));
    }

    let key = PipelineKey {
        op: kind.op_name(),
        dtype: "u32",
        variant: "row",
    };
    let pipeline = backend
        .cache
        .get_or_insert_with(key, || build_pipeline(backend, kind));

    // Output is u32 indices, one per row.
    let out = WgpuStorage::allocate(&backend.device, b, Dtype::I32)?;
    let params_data = [b as u32, k as u32, 0_u32, 0_u32];
    let meta = backend.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("argmax-meta"),
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
            label: Some(kind.op_name()),
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
            label: Some(kind.op_name()),
        });
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(kind.op_name()),
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
    fn wgsl_uses_two_shared_arrays() {
        for k in [ArgKind::Max, ArgKind::Min] {
            let s = argmax_wgsl(k);
            assert!(s.contains("shared_val"));
            assert!(s.contains("shared_idx"));
            assert!(s.contains("workgroupBarrier"));
        }
    }
}

#[cfg(all(test, feature = "gpu-tests"))]
mod gpu_tests {
    use super::*;
    use crate::transfer::to_gpu;
    use rustorch_core::tensor::tensor_impl::Tensor;

    fn run(kind: ArgKind, data: &[f32], b: usize, k: usize) -> Vec<i32> {
        let backend = WgpuBackend::new_blocking().expect("init");
        let t = Tensor::from_vec([b, k], data.to_vec()).unwrap();
        let g = to_gpu(&backend, &t).unwrap();
        let r = argmax_rows(&backend, &g, b, k, kind).unwrap();
        // Output is stored as I32 dtype (4-byte words). Read back via
        // dtype reinterpret on host.
        let bytes = pollster::block_on(crate::transfer::to_cpu_async(
            &backend,
            // Cheating: reinterpret as f32 then convert. Cleaner path
            // is a dtype-aware to_cpu, but i32 round-trip via f32
            // bitcast is fine for reads.
            &WgpuStorage {
                buffer: r.buffer.clone(),
                dtype: Dtype::F32,
                numel: r.numel,
            },
            vec![b],
        ))
        .unwrap();
        bytes
            .as_slice::<f32>()
            .unwrap()
            .iter()
            .map(|f| f.to_bits() as i32)
            .collect()
    }

    #[test]
    fn argmax_simple() {
        let r = run(
            ArgKind::Max,
            &[1.0, 5.0, 3.0, 2.0, /**/ 7.0, -1.0, 4.0, 0.0],
            2,
            4,
        );
        assert_eq!(r, vec![1, 0]);
    }

    #[test]
    fn argmin_simple() {
        let r = run(
            ArgKind::Min,
            &[1.0, 5.0, 3.0, 2.0, /**/ 7.0, -1.0, 4.0, 0.0],
            2,
            4,
        );
        assert_eq!(r, vec![0, 1]);
    }

    #[test]
    fn argmax_picks_first_occurrence_on_ties() {
        // All-equal row should return 0 (lowest index wins).
        let r = run(ArgKind::Max, &[1.0_f32, 1.0, 1.0, 1.0], 1, 4);
        assert_eq!(r, vec![0]);
    }
}
