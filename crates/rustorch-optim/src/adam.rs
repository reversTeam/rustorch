//! Adam / AdamW optimisers (P1.7).
//!
//! Adam (Kingma & Ba 2015):
//! ```text
//!   m_t = β₁ * m_{t-1} + (1 - β₁) * g
//!   v_t = β₂ * v_{t-1} + (1 - β₂) * g²
//!   m̂_t = m_t / (1 - β₁^t)
//!   v̂_t = v_t / (1 - β₂^t)
//!   θ_t = θ_{t-1} - lr * m̂_t / (sqrt(v̂_t) + ε)
//! ```
//!
//! AdamW: same update with **decoupled** weight decay applied directly
//! on the parameter (not via the gradient).

use crate::state_dict::{flatten_buffers, unflatten_buffers, OptimMeta};
use crate::{write_param_data, Optimizer};
use rustorch_autograd::Variable;
use rustorch_core::tensor::tensor_impl::Tensor;
use std::collections::BTreeMap;

/// Adam optimiser with bias correction.
pub struct Adam {
    params: Vec<Variable>,
    lr: f32,
    betas: (f32, f32),
    eps: f32,
    weight_decay: f32,
    decoupled_wd: bool,
    step_t: usize,
    m: Vec<Option<Vec<f32>>>,
    v: Vec<Option<Vec<f32>>>,
    /// Per-parameter GPU `m` state (allocated lazily on the first GPU
    /// step). When present, the WGSL FusedAdamW kernel is used in
    /// place of the CPU loop for that parameter — single dispatch,
    /// zero host trip. P3.Z Task A perf path.
    #[cfg(feature = "wgpu")]
    m_gpu: Vec<Option<rustorch_wgpu::storage::WgpuStorage>>,
    /// Per-parameter GPU `v` state (lazily allocated, see [`Self::m_gpu`]).
    #[cfg(feature = "wgpu")]
    v_gpu: Vec<Option<rustorch_wgpu::storage::WgpuStorage>>,
    /// Per-parameter Metal `m` state (lazily allocated). Populates the
    /// FusedAdamW Metal kernel for params whose tensor lives on
    /// `Storage::Metal` — same in-place mutate-across-steps pattern
    /// as the WGSL version. P3.Z Task J Phase 3.
    #[cfg(all(feature = "metal", target_os = "macos"))]
    m_metal: Vec<Option<rustorch_metal::Buffer>>,
    /// Per-parameter Metal `v` state.
    #[cfg(all(feature = "metal", target_os = "macos"))]
    v_metal: Vec<Option<rustorch_metal::Buffer>>,
}

impl Adam {
    /// Build an Adam optimiser with the given learning rate.
    pub fn new(params: Vec<Variable>, lr: f32) -> Self {
        let n = params.len();
        Adam {
            params,
            lr,
            betas: (0.9, 0.999),
            eps: 1e-8,
            weight_decay: 0.0,
            decoupled_wd: false,
            step_t: 0,
            m: vec![None; n],
            v: vec![None; n],
            #[cfg(feature = "wgpu")]
            m_gpu: (0..n).map(|_| None).collect(),
            #[cfg(feature = "wgpu")]
            v_gpu: (0..n).map(|_| None).collect(),
            #[cfg(all(feature = "metal", target_os = "macos"))]
            m_metal: (0..n).map(|_| None).collect(),
            #[cfg(all(feature = "metal", target_os = "macos"))]
            v_metal: (0..n).map(|_| None).collect(),
        }
    }

    /// Override `(β₁, β₂)`. Defaults are (0.9, 0.999).
    #[must_use]
    pub fn betas(mut self, b1: f32, b2: f32) -> Self {
        self.betas = (b1, b2);
        self
    }

    /// Override `ε`. Default is 1e-8.
    #[must_use]
    pub fn eps(mut self, e: f32) -> Self {
        self.eps = e;
        self
    }

    /// Set L2 weight decay coefficient (added to the gradient).
    #[must_use]
    pub fn weight_decay(mut self, wd: f32) -> Self {
        self.weight_decay = wd;
        self
    }

    /// Borrow the parameter list (read-only).
    pub fn parameters(&self) -> &[Variable] {
        &self.params
    }

    /// Serialise per-parameter `m` / `v` buffers as a tensor map keyed
    /// by `"m.{i}"` / `"v.{i}"`. Hyperparameters (lr, betas, eps,
    /// weight_decay, step_t) are returned separately via [`Self::meta`].
    pub fn state_dict(&self) -> BTreeMap<String, Tensor> {
        let mut out = BTreeMap::new();
        flatten_buffers(&mut out, "m", &self.m);
        flatten_buffers(&mut out, "v", &self.v);
        out
    }

    /// Restore `m` / `v` buffers from a tensor map. The number of
    /// parameters must match the optimizer's `params.len()`.
    pub fn load_state_dict(&mut self, sd: &BTreeMap<String, Tensor>) {
        self.m = unflatten_buffers(sd, "m", self.params.len());
        self.v = unflatten_buffers(sd, "v", self.params.len());
    }

    /// Hyperparameters / step counter snapshot.
    pub fn meta(&self) -> OptimMeta {
        OptimMeta {
            lr: self.lr,
            step_t: self.step_t,
            betas: Some(self.betas),
            eps: Some(self.eps),
            weight_decay: Some(self.weight_decay),
            ..OptimMeta::default()
        }
    }

    /// Restore from an [`OptimMeta`] snapshot.
    pub fn load_meta(&mut self, meta: &OptimMeta) {
        self.lr = meta.lr;
        self.step_t = meta.step_t;
        if let Some(b) = meta.betas {
            self.betas = b;
        }
        if let Some(e) = meta.eps {
            self.eps = e;
        }
        if let Some(wd) = meta.weight_decay {
            self.weight_decay = wd;
        }
    }
}

/// AdamW = Adam with decoupled weight decay (Loshchilov & Hutter 2019).
pub struct AdamW(Adam);

impl AdamW {
    /// Build an AdamW optimiser. Default weight_decay = 0.01.
    pub fn new(params: Vec<Variable>, lr: f32) -> Self {
        let mut inner = Adam::new(params, lr);
        inner.weight_decay = 0.01;
        inner.decoupled_wd = true;
        AdamW(inner)
    }

    /// Override betas (passes through to inner Adam).
    #[must_use]
    pub fn betas(mut self, b1: f32, b2: f32) -> Self {
        self.0.betas = (b1, b2);
        self
    }

    /// Override weight decay (decoupled).
    #[must_use]
    pub fn weight_decay(mut self, wd: f32) -> Self {
        self.0.weight_decay = wd;
        self
    }

    /// Override eps.
    #[must_use]
    pub fn eps(mut self, e: f32) -> Self {
        self.0.eps = e;
        self
    }

    /// Borrow the parameter list.
    pub fn parameters(&self) -> &[Variable] {
        self.0.parameters()
    }
}

#[cfg(feature = "wgpu")]
impl Adam {
    /// Try to run a fused AdamW step entirely on GPU memory.
    ///
    /// Returns `true` when the step was dispatched on GPU (caller
    /// should `continue` to the next param), `false` when the param
    /// or grad isn't on Wgpu storage and the caller must fall back
    /// to the CPU path. Only handles the **decoupled-WD** variant
    /// (i.e. `AdamW`) — classical Adam's coupled WD takes the CPU
    /// path because the kernel doesn't yet branch for it.
    ///
    /// Lazily allocates the per-parameter `m_gpu` / `v_gpu` zero
    /// buffers on the first GPU step. State persists across steps
    /// inside the kernel-mutated buffers — no upload, no download.
    fn try_step_wgpu(&mut self, i: usize, _bc1: f32, _bc2: f32) -> bool {
        use rustorch_core::tensor::device::Device;
        use rustorch_wgpu::backend_singleton::wgpu_backend;
        use rustorch_wgpu::fused_adamw::{allocate_zeros, fused_adamw_step, AdamWStepParams};
        use rustorch_wgpu::transfer::to_gpu;

        let param = self.params[i].clone();
        let param_tensor = param.tensor();

        // Only run on Wgpu-targeted params. Tensors flagged
        // Device::Cpu skip the GPU path entirely.
        if param_tensor.device() != Device::Wgpu {
            return false;
        }
        let backend = wgpu_backend();

        // First-step bridge: a parameter tagged Wgpu may still hold
        // its initial CPU storage (typical pattern is
        // `Tensor::from_vec(...).with_device(Wgpu)` for weight init).
        // Upload it once here so subsequent steps reuse the GPU
        // buffer for free.
        let param_storage = match param_tensor.as_wgpu_storage() {
            Some(s) => s.clone(),
            None => match to_gpu(backend, &param_tensor) {
                Ok(uploaded) => {
                    let core_handle = uploaded.buffer.clone();
                    let promoted = rustorch_core::tensor::tensor_impl::Tensor::from_wgpu_storage(
                        core_handle.clone(),
                        param_tensor.shape().to_vec(),
                        param_tensor.dtype(),
                    );
                    param.set_data(promoted);
                    core_handle
                },
                Err(_) => return false,
            },
        };

        // Fetch the raw gradient (no auto-materialise); skip if absent
        // or if it lives on the host (the kernel needs both buffers
        // on the same device).
        let grad_tensor = match param.raw_grad() {
            Some(g) => g,
            None => return true, // No gradient ⇒ nothing to do, but counts as "handled".
        };
        let grad_storage = match grad_tensor.as_wgpu_storage() {
            Some(s) => s.clone(),
            None => match to_gpu(backend, &grad_tensor) {
                Ok(uploaded) => uploaded.buffer.clone(),
                Err(_) => return false,
            },
        };

        let n = param_tensor.numel();
        let backend = wgpu_backend();

        // Lazy-init m / v on GPU. Once allocated, the same buffers
        // are reused every step — the kernel mutates them in-place,
        // so state survives across calls without any copy.
        if self.m_gpu[i].is_none() {
            let m = match allocate_zeros(backend, n) {
                Ok(m) => m,
                Err(_) => return false,
            };
            self.m_gpu[i] = Some(m);
        }
        if self.v_gpu[i].is_none() {
            let v = match allocate_zeros(backend, n) {
                Ok(v) => v,
                Err(_) => return false,
            };
            self.v_gpu[i] = Some(v);
        }

        let m = self.m_gpu[i].as_ref().expect("m_gpu just allocated above");
        let v = self.v_gpu[i].as_ref().expect("v_gpu just allocated above");

        let core = rustorch_wgpu::storage::WgpuStorage {
            buffer: param_storage,
            dtype: param_tensor.dtype(),
            numel: n,
        };
        let core_grad = rustorch_wgpu::storage::WgpuStorage {
            buffer: grad_storage,
            dtype: grad_tensor.dtype(),
            numel: grad_tensor.numel(),
        };

        let step_params = AdamWStepParams {
            lr: self.lr,
            beta1: self.betas.0,
            beta2: self.betas.1,
            eps: self.eps,
            weight_decay: self.weight_decay,
            t: self.step_t as u32,
        };
        let new_param = match fused_adamw_step(backend, &core, &core_grad, m, v, step_params) {
            Ok(p) => p,
            Err(_) => return false,
        };

        // Wrap the freshly-allocated GPU buffer back into a Tensor
        // with `Storage::Wgpu(...)` so the next forward pass's
        // `to_gpu` takes the fast path (clone Arc, no upload).
        let shape = param_tensor.shape().to_vec();
        let dtype = param_tensor.dtype();
        let new_tensor = rustorch_core::tensor::tensor_impl::Tensor::from_wgpu_storage(
            new_param.buffer,
            shape,
            dtype,
        );
        param.set_data(new_tensor);
        true
    }
}

#[cfg(all(feature = "metal", target_os = "macos"))]
impl Adam {
    /// Try to run a fused AdamW step on Apple Metal direct.
    /// Mirror of [`Self::try_step_wgpu`] for the rustorch-metal path.
    /// Returns `true` on dispatch (caller should `continue`), `false`
    /// when the param/grad isn't on Metal storage.
    fn try_step_metal(&mut self, i: usize) -> bool {
        use rustorch_core::tensor::device::Device;
        use rustorch_metal::backend_singleton::metal_backend;
        use rustorch_metal::fused_adamw::{allocate_zeros, AdamWStepParams};

        let param = self.params[i].clone();
        let param_tensor = param.tensor();

        if param_tensor.device() != Device::Metal {
            return false;
        }
        let backend = metal_backend();

        // First-step bridge: param tagged Metal but still in CPU
        // storage. Upload via shared-mode buffer (essentially a memcpy
        // on Apple Silicon thanks to unified memory).
        let param_buf: rustorch_metal::Buffer = match param_tensor.as_metal_storage() {
            Some(s) => (**s).clone(),
            None => {
                let n_bytes = param_tensor.numel() * 4;
                let buf = match backend.alloc_shared(n_bytes) {
                    Ok(b) => b,
                    Err(_) => return false,
                };
                let data = match param_tensor.as_slice::<f32>() {
                    Some(d) => d,
                    None => return false,
                };
                // SAFETY: shared-storage buffer; pointer valid for n_bytes.
                unsafe {
                    let dst = buf.contents() as *mut f32;
                    for (j, &v) in data.iter().enumerate() {
                        *dst.add(j) = v;
                    }
                }
                let core =
                    rustorch_core::tensor::storage::MetalStorage::standalone(buf.clone(), n_bytes);
                let promoted = rustorch_core::tensor::tensor_impl::Tensor::from_metal_storage(
                    core,
                    param_tensor.shape().to_vec(),
                    param_tensor.dtype(),
                );
                param.set_data(promoted);
                buf
            },
        };

        // Get the gradient buffer (raw — no auto-materialise).
        let grad_tensor = match param.raw_grad() {
            Some(g) => g,
            None => return true, // No gradient ⇒ nothing to do.
        };
        let grad_buf: rustorch_metal::Buffer = match grad_tensor.as_metal_storage() {
            Some(s) => (**s).clone(),
            None => {
                // Gradient is on CPU storage (e.g. via auto-materialise
                // from an upstream backward op). Upload before dispatch.
                let n_bytes = grad_tensor.numel() * 4;
                let buf = match backend.alloc_shared(n_bytes) {
                    Ok(b) => b,
                    Err(_) => return false,
                };
                let data = match grad_tensor.as_slice::<f32>() {
                    Some(d) => d,
                    None => return false,
                };
                // SAFETY: shared-storage buffer; pointer valid for n_bytes.
                unsafe {
                    let dst = buf.contents() as *mut f32;
                    for (j, &v) in data.iter().enumerate() {
                        *dst.add(j) = v;
                    }
                }
                buf
            },
        };

        let n = param_tensor.numel();

        if self.m_metal[i].is_none() {
            let m = match allocate_zeros(backend, n) {
                Ok(m) => m,
                Err(_) => return false,
            };
            self.m_metal[i] = Some(m);
        }
        if self.v_metal[i].is_none() {
            let v = match allocate_zeros(backend, n) {
                Ok(v) => v,
                Err(_) => return false,
            };
            self.v_metal[i] = Some(v);
        }
        let m = self.m_metal[i].as_ref().expect("m_metal just allocated");
        let v = self.v_metal[i].as_ref().expect("v_metal just allocated");

        let step_params = AdamWStepParams {
            lr: self.lr,
            beta1: self.betas.0,
            beta2: self.betas.1,
            eps: self.eps,
            weight_decay: self.weight_decay,
            t: self.step_t as u32,
        };
        // In-place AdamW: write back into `param_buf` instead of
        // allocating a fresh ~4 MB output. The Tensor wrapping
        // `param_buf` is shared via `Arc<MetalStorageInner>`, so
        // mutating the underlying MTLBuffer is observable to all
        // current readers. No `param.set_data()` needed because the
        // Tensor's storage Arc is unchanged.
        use rustorch_metal::fused_adamw::fused_adamw_step_inplace;
        if fused_adamw_step_inplace(backend, &param_buf, &grad_buf, m, v, n, step_params).is_err() {
            return false;
        }
        // bf16 cache for `param_buf` is now stale (param values just
        // changed). Evict so the next forward/backward re-casts.
        if let Some(s) = param_tensor.as_metal_storage() {
            backend.evict_bf16(s.cache_key());
        }
        // No `param.set_data()` needed — the Tensor still wraps the
        // (now in-place updated) `param_buf` Arc. This also avoids the
        // first-step "promote CPU storage to Metal" branch on step 2+
        // since param.tensor() already reports `Storage::Metal`.
        true
    }
}

impl Optimizer for Adam {
    fn step(&mut self) {
        self.step_t += 1;
        let t = self.step_t as f32;
        let bc1 = 1.0 - self.betas.0.powf(t);
        let bc2 = 1.0 - self.betas.1.powf(t);

        // Snapshot the indices of params we're iterating before the
        // borrow checker complains about `self.params.iter()` aliasing
        // `&mut self.m_gpu` etc. The expensive work is per-param so
        // the index-based loop has the same shape as the original.
        let n_params = self.params.len();
        for i in 0..n_params {
            // P3.Z Task A GPU fast path: when wgpu feature is on AND
            // both `param` and `grad` live on `Storage::Wgpu`, dispatch
            // a single WGSL kernel that computes the entire AdamW step
            // in-place on GPU memory — no host trip, no CPU compute.
            #[cfg(feature = "wgpu")]
            {
                if self.decoupled_wd && self.try_step_wgpu(i, bc1, bc2) {
                    continue;
                }
            }
            // P3.Z Task J Phase 3 — same pattern for Apple Metal direct.
            #[cfg(all(feature = "metal", target_os = "macos"))]
            {
                if self.decoupled_wd && self.try_step_metal(i) {
                    continue;
                }
            }

            let param = &self.params[i];
            let grad = match param.grad() {
                Some(g) => g,
                None => continue,
            };
            let snapshot = param.data_snapshot();
            let p_data: &[f32] = snapshot.as_slice::<f32>().expect("Adam: F32 only");
            let g_data: &[f32] = grad.as_slice::<f32>().expect("Adam: F32 only");

            // Effective gradient: classical Adam couples weight decay
            // into the gradient; AdamW applies it directly to the param.
            let g_eff: Vec<f32> = if !self.decoupled_wd && self.weight_decay > 0.0 {
                p_data
                    .iter()
                    .zip(g_data.iter())
                    .map(|(&p, &g)| g + self.weight_decay * p)
                    .collect()
            } else {
                g_data.to_vec()
            };

            // m and v buffers
            let mut m_buf = self.m[i]
                .take()
                .unwrap_or_else(|| vec![0.0_f32; p_data.len()]);
            let mut v_buf = self.v[i]
                .take()
                .unwrap_or_else(|| vec![0.0_f32; p_data.len()]);
            for k in 0..p_data.len() {
                m_buf[k] = self.betas.0 * m_buf[k] + (1.0 - self.betas.0) * g_eff[k];
                v_buf[k] = self.betas.1 * v_buf[k] + (1.0 - self.betas.1) * g_eff[k] * g_eff[k];
            }

            let mut new = Vec::with_capacity(p_data.len());
            for k in 0..p_data.len() {
                let m_hat = m_buf[k] / bc1;
                let v_hat = v_buf[k] / bc2;
                let mut p_new = p_data[k] - self.lr * m_hat / (v_hat.sqrt() + self.eps);
                if self.decoupled_wd && self.weight_decay > 0.0 {
                    p_new -= self.lr * self.weight_decay * p_data[k];
                }
                new.push(p_new);
            }
            write_param_data(param, new);
            self.m[i] = Some(m_buf);
            self.v[i] = Some(v_buf);
        }
        // No reload needed — Variable::tensor() reads fresh from `data`.
    }

    fn zero_grad(&mut self) {
        for param in &self.params {
            param.zero_grad();
        }
    }

    fn lr(&self) -> f32 {
        self.lr
    }

    fn set_lr(&mut self, lr: f32) {
        self.lr = lr;
    }
}

impl Optimizer for AdamW {
    fn step(&mut self) {
        self.0.step();
    }
    fn zero_grad(&mut self) {
        self.0.zero_grad();
    }
    fn lr(&self) -> f32 {
        self.0.lr()
    }
    fn set_lr(&mut self, lr: f32) {
        self.0.set_lr(lr);
    }
}
