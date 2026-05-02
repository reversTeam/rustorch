//! Element-wise op dispatch for the wgpu backend.
//!
//! `dispatch_binary(backend, op, lhs, rhs) → out` and
//! `dispatch_unary(backend, op, inp) → out` build/grab the cached
//! pipeline, encode the bind groups, submit, and return a fresh
//! `WgpuStorage`. Output buffer numel matches the inputs (no
//! broadcasting in v1 — broadcasted shapes go through dedicated
//! kernels in a follow-up).

use crate::backend::WgpuBackend;
use crate::cache::PipelineKey;
use crate::error::WgpuError;
use crate::shaders;
use crate::storage::WgpuStorage;
use rustorch_core::tensor::dtype::Dtype;
use std::sync::Arc;

const WORKGROUP_SIZE: u32 = 64;
const MAX_DIM: u32 = 65535;

/// Compute a 2D dispatch grid that covers `n_groups` workgroups while
/// respecting the per-dimension cap of [`MAX_DIM`]. Returns `(x, y)`.
///
/// Used by elementwise / im2col / permute kernels — the WGSL side
/// recomputes the flat index as `(wgid.y * nwg.x + wgid.x) * WG + lid.x`.
pub(crate) fn split_dispatch(n_groups: u32) -> (u32, u32) {
    if n_groups <= MAX_DIM {
        (n_groups.max(1), 1)
    } else {
        // Roughly square split: x = MAX_DIM, y = ceil(n_groups / MAX_DIM).
        let y = n_groups.div_ceil(MAX_DIM);
        (MAX_DIM, y)
    }
}

fn build_pipeline(
    backend: &WgpuBackend,
    op: &'static str,
    source: String,
) -> wgpu::ComputePipeline {
    let module = backend
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(op),
            source: wgpu::ShaderSource::Wgsl(source.into()),
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

fn pipeline_for(backend: &WgpuBackend, op: &'static str) -> Arc<wgpu::ComputePipeline> {
    let key = PipelineKey {
        op,
        dtype: "f32",
        variant: "default",
    };
    backend
        .cache
        .get_or_insert_with(key, || build_pipeline(backend, op, shaders::source_for(op)))
}

fn meta_buffer(backend: &WgpuBackend, numel: u32) -> wgpu::Buffer {
    // 16-byte aligned (vec4<u32>) — matches the WGSL `meta: array<vec4<u32>, 1>`.
    let data = [numel, 0_u32, 0_u32, 0_u32];
    let buf = backend.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("meta"),
        size: 16,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    backend
        .queue
        .write_buffer(&buf, 0, bytemuck::cast_slice(&data));
    buf
}

/// Run a binary elementwise op. `lhs` and `rhs` must have the same
/// `numel`; the output has the same numel.
pub fn dispatch_binary(
    backend: &WgpuBackend,
    op: &'static str,
    lhs: &WgpuStorage,
    rhs: &WgpuStorage,
) -> Result<WgpuStorage, WgpuError> {
    if lhs.numel != rhs.numel {
        return Err(WgpuError::ShapeMismatch(format!(
            "{op}: lhs numel {} != rhs numel {}",
            lhs.numel, rhs.numel
        )));
    }
    if lhs.dtype != Dtype::F32 || rhs.dtype != Dtype::F32 {
        return Err(WgpuError::UnsupportedDtype(lhs.dtype));
    }
    let pipeline = pipeline_for(backend, op);
    let out = WgpuStorage::allocate(&backend.device, lhs.numel, Dtype::F32)?;
    let meta = meta_buffer(backend, lhs.numel as u32);

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
        let groups = (lhs.numel as u32).div_ceil(WORKGROUP_SIZE).max(1);
        let (gx, gy) = split_dispatch(groups);
        cpass.dispatch_workgroups(gx, gy, 1);
    }
    backend.queue.submit(Some(encoder.finish()));
    Ok(out)
}

/// Run a unary elementwise op (binding 0 = input, binding 2 = output).
pub fn dispatch_unary(
    backend: &WgpuBackend,
    op: &'static str,
    inp: &WgpuStorage,
) -> Result<WgpuStorage, WgpuError> {
    if inp.dtype != Dtype::F32 {
        return Err(WgpuError::UnsupportedDtype(inp.dtype));
    }
    let pipeline = pipeline_for(backend, op);
    let out = WgpuStorage::allocate(&backend.device, inp.numel, Dtype::F32)?;
    let meta = meta_buffer(backend, inp.numel as u32);

    let bind_group_layout = pipeline.get_bind_group_layout(0);
    let bind_group = backend
        .device
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(op),
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
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(op) });
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(op),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        let groups = (inp.numel as u32).div_ceil(WORKGROUP_SIZE).max(1);
        let (gx, gy) = split_dispatch(groups);
        cpass.dispatch_workgroups(gx, gy, 1);
    }
    backend.queue.submit(Some(encoder.finish()));
    Ok(out)
}

#[cfg(test)]
mod unit_tests {
    use super::split_dispatch;

    #[test]
    fn split_dispatch_below_cap_stays_1d() {
        assert_eq!(split_dispatch(1), (1, 1));
        assert_eq!(split_dispatch(64), (64, 1));
        assert_eq!(split_dispatch(65_535), (65_535, 1));
    }

    #[test]
    fn split_dispatch_above_cap_goes_2d() {
        // n_groups just above the cap → y=2.
        assert_eq!(split_dispatch(65_536), (65_535, 2));
        // 4× the cap → y=4 (262_140 covered, then we trim n inside the kernel).
        assert_eq!(split_dispatch(262_140), (65_535, 4));
        // Worst case the bench triggered.
        let (gx, gy) = split_dispatch((4_194_304_u32).div_ceil(64));
        assert!(gx <= 65_535 && gy <= 65_535);
        assert!(gx as u64 * gy as u64 * 64 >= 4_194_304);
    }
}

#[cfg(all(test, feature = "gpu-tests"))]
mod tests {
    use super::*;
    use crate::transfer::{to_cpu, to_gpu};
    use rustorch_core::tensor::tensor_impl::Tensor;

    fn run_binary(op: &'static str, a: &[f32], b: &[f32]) -> Vec<f32> {
        let backend = WgpuBackend::new_blocking().expect("init wgpu");
        let ta = Tensor::from_vec([a.len()], a.to_vec()).unwrap();
        let tb = Tensor::from_vec([b.len()], b.to_vec()).unwrap();
        let ga = to_gpu(&backend, &ta).unwrap();
        let gb = to_gpu(&backend, &tb).unwrap();
        let go = dispatch_binary(&backend, op, &ga, &gb).unwrap();
        let out = to_cpu(&backend, &go, vec![a.len()]).unwrap();
        out.as_slice::<f32>().unwrap().to_vec()
    }

    fn run_unary(op: &'static str, a: &[f32]) -> Vec<f32> {
        let backend = WgpuBackend::new_blocking().expect("init wgpu");
        let ta = Tensor::from_vec([a.len()], a.to_vec()).unwrap();
        let ga = to_gpu(&backend, &ta).unwrap();
        let go = dispatch_unary(&backend, op, &ga).unwrap();
        let out = to_cpu(&backend, &go, vec![a.len()]).unwrap();
        out.as_slice::<f32>().unwrap().to_vec()
    }

    #[test]
    fn add_f32() {
        let r = run_binary("add", &[1.0, 2.0, 3.0], &[10.0, 20.0, 30.0]);
        assert_eq!(r, vec![11.0_f32, 22.0, 33.0]);
    }

    #[test]
    fn sub_f32() {
        let r = run_binary("sub", &[5.0, 7.0, 9.0], &[1.0, 2.0, 3.0]);
        assert_eq!(r, vec![4.0_f32, 5.0, 6.0]);
    }

    #[test]
    fn mul_f32() {
        let r = run_binary("mul", &[2.0, 3.0, 4.0], &[5.0, 6.0, 7.0]);
        assert_eq!(r, vec![10.0_f32, 18.0, 28.0]);
    }

    #[test]
    fn div_f32() {
        let r = run_binary("div", &[10.0, 20.0, 30.0], &[2.0, 4.0, 5.0]);
        assert_eq!(r, vec![5.0_f32, 5.0, 6.0]);
    }

    #[test]
    fn relu_f32() {
        let r = run_unary("relu", &[-1.0, 0.0, 1.0, 2.0]);
        assert_eq!(r, vec![0.0_f32, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn neg_f32() {
        let r = run_unary("neg", &[1.0, -2.0, 3.0]);
        assert_eq!(r, vec![-1.0_f32, 2.0, -3.0]);
    }

    #[test]
    fn sigmoid_f32_at_zero_is_half() {
        let r = run_unary("sigmoid", &[0.0]);
        assert!((r[0] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn tanh_f32_at_zero_is_zero() {
        let r = run_unary("tanh", &[0.0]);
        assert!(r[0].abs() < 1e-6);
    }

    #[test]
    fn silu_f32_at_zero_is_zero() {
        let r = run_unary("silu", &[0.0]);
        assert!(r[0].abs() < 1e-6);
    }

    #[test]
    fn add_large_vector_parity() {
        // 1k random-ish elements, parity vs CPU.
        let a: Vec<f32> = (0..1024).map(|i| i as f32 * 0.1).collect();
        let b: Vec<f32> = (0..1024).map(|i| i as f32 * 0.05).collect();
        let gpu = run_binary("add", &a, &b);
        let cpu: Vec<f32> = a.iter().zip(&b).map(|(x, y)| x + y).collect();
        for (g, c) in gpu.iter().zip(&cpu) {
            assert!((g - c).abs() < 1e-5, "mismatch: gpu {g} vs cpu {c}");
        }
    }

    #[test]
    fn add_above_65535_workgroup_cap() {
        // 5M elements → 5_000_000 / 64 ≈ 78_125 workgroups: above the
        // 65535 single-dim cap on dispatch_workgroups. Regression test
        // for the 2D dispatch fix.
        let n = 5_000_000;
        let a: Vec<f32> = (0..n).map(|i| (i as f32) * 1e-4).collect();
        let b: Vec<f32> = vec![1.0_f32; n];
        let gpu = run_binary("add", &a, &b);
        // Spot-check a few positions instead of the whole vector.
        for &idx in &[0_usize, 1, 65_535 * 64, 65_536 * 64, n - 1] {
            assert!(
                (gpu[idx] - (a[idx] + 1.0)).abs() < 1e-5,
                "mismatch at idx={idx}: gpu={} expected={}",
                gpu[idx],
                a[idx] + 1.0
            );
        }
    }
}
