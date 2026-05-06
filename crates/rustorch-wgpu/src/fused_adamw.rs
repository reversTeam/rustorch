//! Fused AdamW optimizer step — single WGSL dispatch.
//!
//! P3.Z Task A perf — eliminates the host round-trip that the
//! generic `rustorch-optim` AdamW path pays for GPU-resident params:
//! download `param` (auto-materialised by `Variable::data_snapshot`),
//! download `grad` (auto-materialised by `Variable::grad`), CPU
//! arithmetic, upload via `to_gpu` on the next step's first kernel
//! use. For a 1024² weight matrix that's 4 MB downloaded + 4 MB
//! uploaded EVERY step, dominating the wall-clock cost on M4 Max.
//!
//! Replacement: one WGSL dispatch reads `param` + `grad` + `m` + `v`
//! buffers, writes new `param` (fresh buffer) + updated `m` and `v`
//! (in-place on their existing buffers). Zero host trips.
//!
//! Math (decoupled-WD AdamW, matching `rustorch_optim::adam::AdamW`):
//!
//! ```text
//!   m  ← β₁ · m + (1 − β₁) · g
//!   v  ← β₂ · v + (1 − β₂) · g²
//!   m̂ = m / (1 − β₁ᵗ)            // bias-corrected
//!   v̂ = v / (1 − β₂ᵗ)
//!   p ← p − lr · m̂ / (√v̂ + ε)    // Adam step
//!   p ← p − lr · wd · p_old        // decoupled weight decay (AdamW)
//! ```
//!
//! `p_old` is the parameter value BEFORE the Adam step — the WGSL
//! kernel snapshots it locally before mutating.

use crate::backend::WgpuBackend;
use crate::cache::PipelineKey;
use crate::error::WgpuError;
use crate::storage::WgpuStorage;
use rustorch_core::tensor::dtype::Dtype;

/// Workgroup size — 256 threads × N workgroups covers any param size
/// up to wgpu's per-dim limit (`max_compute_workgroups_per_dimension`,
/// typically 65535) — i.e. up to ≥ 16 M elements, way past our needs.
const WORKGROUP_SIZE: u32 = 256;

/// Hyperparameters laid out for the WGSL `Params` uniform. Kept as
/// `repr(C)` so `bytemuck::cast_slice` produces the exact 32-byte
/// payload the shader expects.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct AdamWUniform {
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    bc1: f32, // 1 − β₁ᵗ  (precomputed CPU-side because t is small)
    bc2: f32, // 1 − β₂ᵗ
    n: u32,
}

// SAFETY: AdamWUniform is a plain repr(C) POD with no padding holes
// (all 32 bytes are meaningful), so bytemuck reflection is correct.
unsafe impl bytemuck::Zeroable for AdamWUniform {}
unsafe impl bytemuck::Pod for AdamWUniform {}

/// Hyperparameters as the public AdamW caller sees them. The WGSL
/// kernel needs the bias-corrected denominators precomputed
/// (`1 − β^t`), which we do CPU-side from `t`.
#[derive(Debug, Clone, Copy)]
pub struct AdamWStepParams {
    /// Learning rate.
    pub lr: f32,
    /// β₁ (first-moment decay rate).
    pub beta1: f32,
    /// β₂ (second-moment decay rate).
    pub beta2: f32,
    /// ε (denominator stabiliser).
    pub eps: f32,
    /// Decoupled weight-decay coefficient (AdamW).
    pub weight_decay: f32,
    /// Step counter (one-based, used for the bias-correction
    /// denominators `1 − β^t`).
    pub t: u32,
}

impl AdamWStepParams {
    fn into_uniform(self, n: u32) -> AdamWUniform {
        let bc1 = 1.0 - self.beta1.powi(self.t as i32);
        let bc2 = 1.0 - self.beta2.powi(self.t as i32);
        AdamWUniform {
            lr: self.lr,
            beta1: self.beta1,
            beta2: self.beta2,
            eps: self.eps,
            weight_decay: self.weight_decay,
            bc1,
            bc2,
            n,
        }
    }
}

const SHADER: &str = r#"
struct Params {
    lr:           f32,
    beta1:        f32,
    beta2:        f32,
    eps:          f32,
    weight_decay: f32,
    bc1:          f32,
    bc2:          f32,
    n:            u32,
};

@group(0) @binding(0) var<uniform>             params: Params;
@group(0) @binding(1) var<storage, read>       param_in: array<f32>;
@group(0) @binding(2) var<storage, read>       grad:     array<f32>;
@group(0) @binding(3) var<storage, read_write> m:        array<f32>;
@group(0) @binding(4) var<storage, read_write> v:        array<f32>;
@group(0) @binding(5) var<storage, read_write> param_out: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.n) { return; }

    let p_old = param_in[i];
    let g = grad[i];

    // m ← β₁·m + (1−β₁)·g
    let new_m = params.beta1 * m[i] + (1.0 - params.beta1) * g;
    m[i] = new_m;

    // v ← β₂·v + (1−β₂)·g²
    let new_v = params.beta2 * v[i] + (1.0 - params.beta2) * g * g;
    v[i] = new_v;

    // m̂ = m / bc1, v̂ = v / bc2
    let m_hat = new_m / params.bc1;
    let v_hat = new_v / params.bc2;

    // p ← p − lr · m̂ / (√v̂ + ε)
    var p_new = p_old - params.lr * m_hat / (sqrt(v_hat) + params.eps);

    // Decoupled weight decay (AdamW)
    if (params.weight_decay > 0.0) {
        p_new = p_new - params.lr * params.weight_decay * p_old;
    }

    param_out[i] = p_new;
}
"#;

fn build_pipeline(backend: &WgpuBackend) -> wgpu::ComputePipeline {
    let module = backend
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("fused_adamw"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(SHADER)),
        });
    backend
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("fused_adamw"),
            layout: None,
            module: &module,
            entry_point: "main",
            compilation_options: Default::default(),
            cache: None,
        })
}

/// Run one AdamW step on GPU memory.
///
/// `param_in`, `grad`, `m`, `v` must all be F32 Wgpu storages of the
/// same numel. `m` and `v` are mutated in-place (their underlying
/// `wgpu::Buffer`s are updated by the kernel and remain valid for
/// reuse across steps — caller keeps the same `WgpuStorage` clones).
/// A FRESH `param_out` storage is allocated and returned so the
/// previous param's buffer can stay alive (autograd `SavedTensor`s
/// may still hold an Arc clone).
///
/// Single dispatch — zero host trips, zero CPU arithmetic.
pub fn fused_adamw_step(
    backend: &WgpuBackend,
    param_in: &WgpuStorage,
    grad: &WgpuStorage,
    m: &WgpuStorage,
    v: &WgpuStorage,
    params: AdamWStepParams,
) -> Result<WgpuStorage, WgpuError> {
    let n = param_in.numel;
    if grad.numel != n || m.numel != n || v.numel != n {
        return Err(WgpuError::ShapeMismatch(format!(
            "fused_adamw: numel mismatch (param={}, grad={}, m={}, v={})",
            n, grad.numel, m.numel, v.numel
        )));
    }
    if param_in.dtype != Dtype::F32
        || grad.dtype != Dtype::F32
        || m.dtype != Dtype::F32
        || v.dtype != Dtype::F32
    {
        return Err(WgpuError::UnsupportedDtype(param_in.dtype));
    }

    let key = PipelineKey {
        op: "fused_adamw",
        dtype: "f32",
        variant: "decoupled-wd",
    };
    let pipeline = backend
        .cache
        .get_or_insert_with(key, || build_pipeline(backend));

    let param_out = WgpuStorage::allocate(&backend.device, n, Dtype::F32)?;

    let uniform = params.into_uniform(n as u32);
    let uniform_buf = backend.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("fused_adamw-uniform"),
        size: core::mem::size_of::<AdamWUniform>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    backend
        .queue
        .write_buffer(&uniform_buf, 0, bytemuck::bytes_of(&uniform));

    let bind_group_layout = pipeline.get_bind_group_layout(0);
    let bind_group = backend
        .device
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fused_adamw-bind"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: param_in.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: grad.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: m.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: v.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: param_out.buffer.as_entire_binding(),
                },
            ],
        });

    let mut encoder = backend
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("fused_adamw-encoder"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("fused_adamw-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let n_groups = (n as u32).div_ceil(WORKGROUP_SIZE);
        pass.dispatch_workgroups(n_groups, 1, 1);
    }
    backend.queue.submit(Some(encoder.finish()));

    Ok(param_out)
}

/// Allocate a zero-initialised GPU buffer of `numel` F32 elements,
/// suitable as initial `m` / `v` storage. Used by `rustorch-optim`'s
/// AdamW lazy-init path the first time a parameter is `step()`ed on
/// the GPU.
pub fn allocate_zeros(backend: &WgpuBackend, numel: usize) -> Result<WgpuStorage, WgpuError> {
    let storage = WgpuStorage::allocate(&backend.device, numel, Dtype::F32)?;
    // wgpu::Buffer::create_buffer with COPY_DST | STORAGE doesn't
    // zero by default. Easiest cross-platform zero-init: write a
    // zero-filled CPU buffer of the same size via `queue.write_buffer`.
    let zeros = vec![0_u8; numel * 4];
    backend.queue.write_buffer(&storage.buffer, 0, &zeros);
    backend
        .queue
        .submit(std::iter::empty::<wgpu::CommandBuffer>());
    Ok(storage)
}
