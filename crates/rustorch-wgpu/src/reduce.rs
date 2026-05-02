//! Row-wise reductions on the GPU.
//!
//! Treats the input as a `[B, K]` matrix and produces a `[B]` output:
//! one workgroup per row, threads cooperate via workgroup-shared
//! memory to reduce K elements down to a single scalar.
//!
//! v1 supports `sum`, `mean`, `max`. Backwards (e.g. `softmax`) is
//! built on top in [`softmax`](crate::softmax).

use crate::backend::WgpuBackend;
use crate::cache::PipelineKey;
use crate::error::WgpuError;
use crate::storage::WgpuStorage;
use rustorch_core::tensor::dtype::Dtype;

const BLOCK: u32 = 256;

/// Reduction kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReduceKind {
    /// Sum along the row.
    Sum,
    /// Arithmetic mean along the row.
    Mean,
    /// Maximum along the row.
    Max,
    /// Minimum along the row.
    Min,
    /// Product along the row.
    Prod,
    /// L2 norm: sqrt(sum(x²)) along the row.
    L2Norm,
}

impl ReduceKind {
    fn op_name(self) -> &'static str {
        match self {
            ReduceKind::Sum => "reduce_sum",
            ReduceKind::Mean => "reduce_mean",
            ReduceKind::Max => "reduce_max",
            ReduceKind::Min => "reduce_min",
            ReduceKind::Prod => "reduce_prod",
            ReduceKind::L2Norm => "reduce_l2norm",
        }
    }

    fn init_value(self) -> &'static str {
        match self {
            ReduceKind::Sum | ReduceKind::Mean | ReduceKind::L2Norm => "0.0",
            ReduceKind::Max => "-3.4e38",
            ReduceKind::Min => "3.4e38",
            ReduceKind::Prod => "1.0",
        }
    }

    /// Combine expression for the per-element loop where `b` is the
    /// freshly read input value `v`. For L2Norm this squares `b`.
    fn combine_loop(self) -> &'static str {
        match self {
            ReduceKind::Sum | ReduceKind::Mean => "a + b",
            ReduceKind::Max => "max(a, b)",
            ReduceKind::Min => "min(a, b)",
            ReduceKind::Prod => "a * b",
            ReduceKind::L2Norm => "a + b * b",
        }
    }

    /// Combine expression for the tree-reduction step. Both `a` and
    /// `b` are partial accumulators of the same kind, so L2Norm
    /// just adds (the squaring already happened in the loop).
    fn combine_tree(self) -> &'static str {
        match self {
            ReduceKind::Sum | ReduceKind::Mean | ReduceKind::L2Norm => "a + b",
            ReduceKind::Max => "max(a, b)",
            ReduceKind::Min => "min(a, b)",
            ReduceKind::Prod => "a * b",
        }
    }

    fn finalize(self) -> &'static str {
        match self {
            ReduceKind::Mean => "acc / f32(k)",
            ReduceKind::L2Norm => "sqrt(acc)",
            _ => "acc",
        }
    }
}

fn reduce_wgsl(kind: ReduceKind) -> String {
    let init = kind.init_value();
    let combine_loop = kind.combine_loop().replace("b", "b_");
    let combine_tree = kind.combine_tree().replace("b", "b_");
    let finalize = kind.finalize();
    format!(
        r#"
@group(0) @binding(0) var<storage, read>  inp: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> params: array<vec4<u32>, 1>;

var<workgroup> shared_buf: array<f32, {BLOCK}u>;

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
    var i: u32 = lid.x;
    loop {{
        if (i >= k) {{ break; }}
        let v = inp[row * k + i];
        let a = acc; let b_ = v;
        acc = {combine_loop};
        i = i + {BLOCK}u;
    }}
    shared_buf[lid.x] = acc;
    workgroupBarrier();

    var stride: u32 = {BLOCK}u / 2u;
    loop {{
        if (stride == 0u) {{ break; }}
        if (lid.x < stride) {{
            let a = shared_buf[lid.x]; let b_ = shared_buf[lid.x + stride];
            shared_buf[lid.x] = {combine_tree};
        }}
        workgroupBarrier();
        stride = stride / 2u;
    }}

    if (lid.x == 0u) {{
        let acc = shared_buf[0];
        out[row] = {finalize};
    }}
}}
"#,
        BLOCK = BLOCK,
        init = init,
        combine_loop = combine_loop,
        combine_tree = combine_tree,
        finalize = finalize,
    )
}

fn build_pipeline(backend: &WgpuBackend, kind: ReduceKind) -> wgpu::ComputePipeline {
    let module = backend
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(kind.op_name()),
            source: wgpu::ShaderSource::Wgsl(reduce_wgsl(kind).into()),
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

/// Row-wise reduce: input `[b, k]` → output `[b]`.
pub fn reduce_rows(
    backend: &WgpuBackend,
    inp: &WgpuStorage,
    b: usize,
    k: usize,
    kind: ReduceKind,
) -> Result<WgpuStorage, WgpuError> {
    if inp.numel != b * k {
        return Err(WgpuError::ShapeMismatch(format!(
            "reduce_rows: numel {} != b*k {}",
            inp.numel,
            b * k
        )));
    }
    if inp.dtype != Dtype::F32 {
        return Err(WgpuError::UnsupportedDtype(inp.dtype));
    }

    let key = PipelineKey {
        op: kind.op_name(),
        dtype: "f32",
        variant: "row",
    };
    let pipeline = backend
        .cache
        .get_or_insert_with(key, || build_pipeline(backend, kind));

    let out = WgpuStorage::allocate(&backend.device, b, Dtype::F32)?;
    let params_data = [b as u32, k as u32, 0_u32, 0_u32];
    let meta = backend.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("reduce-meta"),
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
    fn wgsl_contains_workgroup_barrier() {
        for k in [ReduceKind::Sum, ReduceKind::Mean, ReduceKind::Max] {
            let s = reduce_wgsl(k);
            assert!(s.contains("workgroupBarrier"), "{}", k.op_name());
            assert!(s.contains("shared_buf"), "{}", k.op_name());
        }
    }

    #[test]
    fn op_names_are_distinct() {
        assert_ne!(ReduceKind::Sum.op_name(), ReduceKind::Mean.op_name());
        assert_ne!(ReduceKind::Sum.op_name(), ReduceKind::Max.op_name());
    }
}

#[cfg(all(test, feature = "gpu-tests"))]
mod gpu_tests {
    use super::*;
    use crate::transfer::{to_cpu, to_gpu};
    use rustorch_core::tensor::tensor_impl::Tensor;

    fn run(kind: ReduceKind, data: &[f32], b: usize, k: usize) -> Vec<f32> {
        let backend = WgpuBackend::new_blocking().expect("init");
        let t = Tensor::from_vec([b, k], data.to_vec()).unwrap();
        let g = to_gpu(&backend, &t).unwrap();
        let r = reduce_rows(&backend, &g, b, k, kind).unwrap();
        let c = to_cpu(&backend, &r, vec![b]).unwrap();
        c.as_slice::<f32>().unwrap().to_vec()
    }

    #[test]
    fn sum_rows_simple() {
        let r = run(ReduceKind::Sum, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        assert!((r[0] - 6.0).abs() < 1e-5);
        assert!((r[1] - 15.0).abs() < 1e-5);
    }

    #[test]
    fn mean_rows_simple() {
        let r = run(ReduceKind::Mean, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        assert!((r[0] - 2.0).abs() < 1e-5);
        assert!((r[1] - 5.0).abs() < 1e-5);
    }

    #[test]
    fn max_rows_simple() {
        let r = run(
            ReduceKind::Max,
            &[1.0, 5.0, 3.0, -2.0, -4.0, -1.0, 7.0, 2.0],
            2,
            4,
        );
        assert!((r[0] - 5.0).abs() < 1e-5);
        assert!((r[1] - 7.0).abs() < 1e-5);
    }

    #[test]
    fn min_rows_simple() {
        let r = run(
            ReduceKind::Min,
            &[1.0, 5.0, 3.0, -2.0, -4.0, -1.0, 7.0, 2.0],
            2,
            4,
        );
        assert!((r[0] - (-2.0)).abs() < 1e-5);
        assert!((r[1] - (-4.0)).abs() < 1e-5);
    }

    #[test]
    fn prod_rows_simple() {
        // Row 1: 1*2*3*4 = 24; Row 2: 0.5*0.5*0.5*0.5 = 0.0625
        let r = run(
            ReduceKind::Prod,
            &[1.0, 2.0, 3.0, 4.0, 0.5, 0.5, 0.5, 0.5],
            2,
            4,
        );
        assert!((r[0] - 24.0).abs() < 1e-5);
        assert!((r[1] - 0.0625).abs() < 1e-6);
    }

    #[test]
    fn l2norm_rows_simple() {
        // Row 1: sqrt(3² + 4²) = 5
        // Row 2: sqrt(1 + 1 + 1 + 1) = 2
        let r = run(
            ReduceKind::L2Norm,
            &[3.0, 4.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
            2,
            4,
        );
        assert!((r[0] - 5.0).abs() < 1e-5);
        assert!((r[1] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn sum_long_row() {
        // K > BLOCK to exercise the strided loop.
        let k = 1024;
        let data: Vec<f32> = (0..k).map(|i| i as f32).collect();
        let r = run(ReduceKind::Sum, &data, 1, k);
        let expected = (0..k).map(|i| i as f32).sum::<f32>();
        assert!((r[0] - expected).abs() / expected.abs() < 1e-4);
    }
}
