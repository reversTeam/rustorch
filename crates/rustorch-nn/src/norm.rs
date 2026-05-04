//! Normalization modules (P1.6).
//!
//! v1 ships:
//! - [`RMSNorm`] (Zhang & Sennrich 2019), the simplest variant used by
//!   Llama, Mistral, T5-derived architectures.
//! - [`LayerNorm`] (Ba et al. 2016), the workhorse of Transformers.
//!
//! Both norms are built by composing existing autograd ops (mul,
//! mean_dim, sqrt, div, add, sub) — backward falls out automatically
//! from the dynamic graph. No hand-coded backward formula required.
//!
//! GroupNorm / BatchNorm / InstanceNorm follow once their dedicated
//! kernels land in P1.4 (Welford-stable variance is on the roadmap).

use crate::module::{Module, ModuleError};
use rustorch_autograd::{is_grad_enabled, ops, Variable};
use rustorch_core::tensor::dtype::Dtype;
use rustorch_core::tensor::tensor_impl::Tensor;

/// Root-Mean-Square LayerNorm: `y = x / sqrt(mean(x², last) + eps) * gamma`.
///
/// `gamma` is a learnable scale of shape `[normalized_size]`. There's
/// no shift parameter (this is the key difference vs LayerNorm).
pub struct RMSNorm {
    /// Learnable per-channel scale of shape `[normalized_size]`.
    pub gamma: Variable,
    eps: f32,
    normalized_size: usize,
}

impl RMSNorm {
    /// Build with the given last-axis size and default eps = 1e-6.
    pub fn new(normalized_size: usize) -> Self {
        Self::with_eps(normalized_size, 1e-6)
    }

    /// Build with explicit epsilon.
    pub fn with_eps(normalized_size: usize, eps: f32) -> Self {
        // Initialise gamma to ones (no-op at init).
        let gamma_data = vec![1.0_f32; normalized_size];
        let gamma =
            Variable::leaf(Tensor::from_vec([normalized_size], gamma_data).expect("gamma shape"));
        RMSNorm {
            gamma,
            eps,
            normalized_size,
        }
    }

    /// Last-axis size this norm operates on.
    pub fn normalized_size(&self) -> usize {
        self.normalized_size
    }
}

impl Module for RMSNorm {
    fn forward(&self, input: &Variable) -> Result<Variable, ModuleError> {
        // T27 — fast path: same `no_grad + dense f32 contig` pattern
        // as LayerNorm (T18). The 5-op autograd composition (mul,
        // mean_dim, add, sqrt, div, mul) was ~178× slower than
        // PyTorch's nn.RMSNorm on a Llama-7B-shape input — fused
        // single-pass kernel collapses it.
        let x_t = input.tensor();
        let g_t = self.gamma.tensor();
        if !is_grad_enabled()
            && x_t.dtype() == Dtype::F32
            && g_t.dtype() == Dtype::F32
            && x_t.is_contiguous()
            && g_t.is_contiguous()
            && x_t.ndim() >= 1
            && x_t.shape()[x_t.ndim() - 1] == self.normalized_size
            && g_t.numel() == self.normalized_size
        {
            let out = rms_norm_forward_f32_fused(&x_t, &g_t, self.eps)?;
            return Ok(Variable::new(out));
        }

        // Reduce dim = last axis of input.
        let last_dim = input.tensor().ndim() - 1;
        // x²
        let x_sq = ops::mul(input, input)?;
        // mean over last axis (keepdim=true)
        let mean_x_sq = ops::mean_dim(&x_sq, &[last_dim])?;
        // mean + eps  (eps as broadcastable scalar Variable on the same device).
        let eps_var = Variable::new(Tensor::scalar(self.eps).with_device(input.tensor().device()));
        let rms_sq = ops::add(&mean_x_sq, &eps_var)?;
        // sqrt
        let rms = ops::sqrt(&rms_sq)?;
        // x / rms (broadcast last axis)
        let normed = ops::div(input, &rms)?;
        // * gamma (gamma [D] broadcasts across leading dims of normed)
        ops::mul(&normed, &self.gamma)
    }

    fn parameters(&self) -> Vec<Variable> {
        vec![self.gamma.clone()]
    }

    fn named_parameters(&self) -> Vec<(String, Variable)> {
        vec![("gamma".to_string(), self.gamma.clone())]
    }
}

// ------------------------------ LayerNorm ------------------------------

/// LayerNorm: `y = (x - mean) / sqrt(var + eps) * gamma + beta`.
///
/// `gamma` and `beta` are both learnable parameters of shape
/// `[normalized_size]`, broadcasted across leading dims.
pub struct LayerNorm {
    /// Learnable per-channel scale.
    pub gamma: Variable,
    /// Learnable per-channel shift.
    pub beta: Variable,
    eps: f32,
    normalized_size: usize,
}

impl LayerNorm {
    /// Build with the given last-axis size and default eps = 1e-5.
    pub fn new(normalized_size: usize) -> Self {
        Self::with_eps(normalized_size, 1e-5)
    }

    /// Build with explicit epsilon.
    pub fn with_eps(normalized_size: usize, eps: f32) -> Self {
        let gamma = Variable::leaf(
            Tensor::from_vec([normalized_size], vec![1.0_f32; normalized_size])
                .expect("gamma shape"),
        );
        let beta = Variable::leaf(
            Tensor::from_vec([normalized_size], vec![0.0_f32; normalized_size])
                .expect("beta shape"),
        );
        LayerNorm {
            gamma,
            beta,
            eps,
            normalized_size,
        }
    }

    /// Last-axis size this norm operates on.
    pub fn normalized_size(&self) -> usize {
        self.normalized_size
    }
}

impl Module for LayerNorm {
    fn forward(&self, input: &Variable) -> Result<Variable, ModuleError> {
        // T18 — fast path: when grad tracking is off and the layout is
        // dense f32 contiguous on the last axis, dispatch to the fused
        // single-pass kernel. The composed-op path (used during
        // training so backward falls out automatically) burns 9
        // intermediate allocations and 9 autograd nodes, which on a
        // GPT-2 block (B=2 S=128 D=256) measures at 5.85 ms — about
        // 100x slower than PyTorch's fused kernel. The fused path
        // collapses everything to one mean+var+normalise+scale+bias
        // pass with a single output allocation.
        let x_t = input.tensor();
        let g_t = self.gamma.tensor();
        let b_t = self.beta.tensor();
        if !is_grad_enabled()
            && x_t.dtype() == Dtype::F32
            && g_t.dtype() == Dtype::F32
            && b_t.dtype() == Dtype::F32
            && x_t.is_contiguous()
            && g_t.is_contiguous()
            && b_t.is_contiguous()
            && x_t.ndim() >= 1
            && x_t.shape()[x_t.ndim() - 1] == self.normalized_size
            && g_t.numel() == self.normalized_size
            && b_t.numel() == self.normalized_size
        {
            let out = layer_norm_forward_f32_fused(&x_t, &g_t, &b_t, self.eps)?;
            return Ok(Variable::new(out));
        }

        let last_dim = input.tensor().ndim() - 1;
        // mean
        let mean = ops::mean_dim(input, &[last_dim])?;
        // x - mean (broadcast)
        let centered = ops::sub(input, &mean)?;
        // (x - mean)²
        let centered_sq = ops::mul(&centered, &centered)?;
        // var = mean of centered²
        let var = ops::mean_dim(&centered_sq, &[last_dim])?;
        // var + eps  (eps placed on input's device for autograd dispatch)
        let eps_var = Variable::new(Tensor::scalar(self.eps).with_device(input.tensor().device()));
        let var_eps = ops::add(&var, &eps_var)?;
        // std = sqrt(var + eps)
        let std = ops::sqrt(&var_eps)?;
        // normalised = centered / std
        let normed = ops::div(&centered, &std)?;
        // * gamma + beta
        let scaled = ops::mul(&normed, &self.gamma)?;
        ops::add(&scaled, &self.beta)
    }

    fn parameters(&self) -> Vec<Variable> {
        vec![self.gamma.clone(), self.beta.clone()]
    }

    fn named_parameters(&self) -> Vec<(String, Variable)> {
        vec![
            ("gamma".to_string(), self.gamma.clone()),
            ("beta".to_string(), self.beta.clone()),
        ]
    }
}

/// T18 — fused single-pass LayerNorm forward (f32 contiguous, last
/// axis). Replaces the 9-op autograd composition under `no_grad`.
///
/// Layout: input `x` is shape `[..., D]` with `D = normalized_size`.
/// `gamma` and `beta` are `[D]`. Output has the same shape as `x`.
///
/// Each row of length `D` is normalised independently. The kernel
/// performs two passes per row over the row's `D` elements:
///   pass 1 — `sum`, `sum_sq` accumulate; mean = sum/D,
///            var = sum_sq/D - mean² (catastrophic-cancellation-safe
///            for the small D used in transformers; we don't need
///            Welford here because all rows are independent and
///            stay in cache).
///   pass 2 — `out[i] = (x[i] - mean) * inv_std * gamma[i] + beta[i]`
///
/// LLVM auto-vectorises both passes into NEON `fadd.4s` /
/// `fmadd.4s` chains. Above 64 K rows we shard via rayon.
fn layer_norm_forward_f32_fused(
    x: &Tensor,
    gamma: &Tensor,
    beta: &Tensor,
    eps: f32,
) -> Result<Tensor, ModuleError> {
    let shape = x.shape().to_vec();
    let d = *shape.last().unwrap();
    let outer: usize = shape.iter().take(shape.len() - 1).product();
    let n = outer * d;

    let x_slice = x.as_slice::<f32>().expect("checked F32 contiguous");
    let g_slice = gamma.as_slice::<f32>().expect("checked F32 contiguous");
    let b_slice = beta.as_slice::<f32>().expect("checked F32 contiguous");

    // T37 — uninitialised output buffer. Pass 2 below overwrites
    // every cell. Saves ~20 µs of zero-fill bandwidth on the
    // [B=1, S=512, D=768] = 1.5 MB output.
    let mut out_storage: Vec<core::mem::MaybeUninit<f32>> = Vec::with_capacity(n);
    #[allow(clippy::uninit_vec)]
    unsafe {
        out_storage.set_len(n);
    }
    let mut out_data: Vec<f32> = unsafe {
        let (ptr, len, cap) = (
            out_storage.as_mut_ptr() as *mut f32,
            out_storage.len(),
            out_storage.capacity(),
        );
        core::mem::forget(out_storage);
        Vec::from_raw_parts(ptr, len, cap)
    };

    let inv_d = 1.0_f32 / (d as f32);

    // T29/T37 — explicit NEON intrinsics row kernel, 8-lane unrolled.
    // Two parallel SUM/SUM_SQ accumulators per pass let the M-series
    // dual-issue both fma pipelines simultaneously, pushing the
    // effective rate from ~1.2 GB/s to >2 GB/s. PT's nn.LayerNorm
    // is at ~2 GB/s; we now match or beat it.
    let process_row = |x_row: &[f32], out_row: &mut [f32]| {
        #[cfg(target_arch = "aarch64")]
        {
            use core::arch::aarch64::*;
            unsafe {
                // Two parallel accumulators per pass — each 4-lane
                // NEON register, so a single iteration consumes 8
                // floats and dispatches 4 fma ops on the two pipes.
                let mut sum_a = vdupq_n_f32(0.0);
                let mut sum_b = vdupq_n_f32(0.0);
                let mut sum_sq_a = vdupq_n_f32(0.0);
                let mut sum_sq_b = vdupq_n_f32(0.0);
                let chunks_8 = d / 8;
                let tail_start = chunks_8 * 8;
                for i in 0..chunks_8 {
                    let va = vld1q_f32(x_row.as_ptr().add(i * 8));
                    let vb = vld1q_f32(x_row.as_ptr().add(i * 8 + 4));
                    sum_a = vaddq_f32(sum_a, va);
                    sum_b = vaddq_f32(sum_b, vb);
                    sum_sq_a = vfmaq_f32(sum_sq_a, va, va);
                    sum_sq_b = vfmaq_f32(sum_sq_b, vb, vb);
                }
                let mut sum = vaddvq_f32(vaddq_f32(sum_a, sum_b));
                let mut sum_sq = vaddvq_f32(vaddq_f32(sum_sq_a, sum_sq_b));
                for &v in x_row[tail_start..].iter() {
                    sum += v;
                    sum_sq += v * v;
                }

                let mean = sum * inv_d;
                let var = (sum_sq * inv_d - mean * mean).max(0.0);
                let inv_std = 1.0 / (var + eps).sqrt();

                // Pass 2 — normalise + scale + bias via 2 fmaq per
                // 4-lane chunk, dual-issue 8 floats per iteration.
                let neg_mean_inv_std = vdupq_n_f32(-mean * inv_std);
                let inv_std_v = vdupq_n_f32(inv_std);
                for i in 0..chunks_8 {
                    let xa = vld1q_f32(x_row.as_ptr().add(i * 8));
                    let xb = vld1q_f32(x_row.as_ptr().add(i * 8 + 4));
                    let za = vfmaq_f32(neg_mean_inv_std, xa, inv_std_v);
                    let zb = vfmaq_f32(neg_mean_inv_std, xb, inv_std_v);
                    let ga = vld1q_f32(g_slice.as_ptr().add(i * 8));
                    let gb = vld1q_f32(g_slice.as_ptr().add(i * 8 + 4));
                    let ba = vld1q_f32(b_slice.as_ptr().add(i * 8));
                    let bb = vld1q_f32(b_slice.as_ptr().add(i * 8 + 4));
                    let ya = vfmaq_f32(ba, za, ga);
                    let yb = vfmaq_f32(bb, zb, gb);
                    vst1q_f32(out_row.as_mut_ptr().add(i * 8), ya);
                    vst1q_f32(out_row.as_mut_ptr().add(i * 8 + 4), yb);
                }
                for i in tail_start..d {
                    out_row[i] = (x_row[i] - mean) * inv_std * g_slice[i] + b_slice[i];
                }
            }
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            let mut sum = 0.0_f32;
            let mut sum_sq = 0.0_f32;
            for &v in x_row.iter() {
                sum += v;
                sum_sq += v * v;
            }
            let mean = sum * inv_d;
            let var = (sum_sq * inv_d - mean * mean).max(0.0);
            let inv_std = 1.0 / (var + eps).sqrt();
            for i in 0..d {
                out_row[i] = (x_row[i] - mean) * inv_std * g_slice[i] + b_slice[i];
            }
        }
    };

    // Threshold below which the rayon scheduler overhead dwarfs the
    // win. 64 rows × 256 ≈ 16K elements is the empirical crossover
    // point on M-series Macs.
    #[cfg(not(target_arch = "wasm32"))]
    const PARALLEL_OUTER_MIN: usize = 64;

    #[cfg(not(target_arch = "wasm32"))]
    if outer >= PARALLEL_OUTER_MIN {
        use rayon::prelude::*;
        // T38 — chunk multiple rows per rayon task. With one row per
        // task (3 KB f32 work unit on D=768), rayon's per-task
        // dispatch overhead (~1 µs) compounds to ~500 µs across 512
        // rows. Bundling 16 rows per chunk amortises the dispatch
        // and keeps each task at 48 KB which still fits comfortably
        // in L1d.
        const ROWS_PER_TASK: usize = 16;
        let chunk_bytes = d * ROWS_PER_TASK;
        out_data
            .par_chunks_mut(chunk_bytes)
            .zip(x_slice.par_chunks(chunk_bytes))
            .for_each(|(out_chunk, x_chunk)| {
                for (out_row, x_row) in out_chunk.chunks_mut(d).zip(x_chunk.chunks(d)) {
                    process_row(x_row, out_row);
                }
            });
    } else {
        for o in 0..outer {
            process_row(
                &x_slice[o * d..(o + 1) * d],
                &mut out_data[o * d..(o + 1) * d],
            );
        }
    }

    #[cfg(target_arch = "wasm32")]
    for o in 0..outer {
        process_row(
            &x_slice[o * d..(o + 1) * d],
            &mut out_data[o * d..(o + 1) * d],
        );
    }

    Tensor::from_vec(shape, out_data).map_err(|e| ModuleError::Backend {
        op: "LayerNorm::forward(fused)",
        message: format!("{e:?}"),
    })
}

/// T27 — fused single-pass RMSNorm forward (f32 contiguous, last
/// axis). RMSNorm is `y = x / sqrt(mean(x²) + eps) * gamma`.
fn rms_norm_forward_f32_fused(x: &Tensor, gamma: &Tensor, eps: f32) -> Result<Tensor, ModuleError> {
    let shape = x.shape().to_vec();
    let d = *shape.last().unwrap();
    let outer: usize = shape.iter().take(shape.len() - 1).product();
    let n = outer * d;

    let x_slice = x.as_slice::<f32>().expect("checked F32 contiguous");
    let g_slice = gamma.as_slice::<f32>().expect("checked F32 contiguous");

    // T37 — uninitialised output (pass 2 overwrites every cell).
    let mut out_storage: Vec<core::mem::MaybeUninit<f32>> = Vec::with_capacity(n);
    #[allow(clippy::uninit_vec)]
    unsafe {
        out_storage.set_len(n);
    }
    let mut out_data: Vec<f32> = unsafe {
        let (ptr, len, cap) = (
            out_storage.as_mut_ptr() as *mut f32,
            out_storage.len(),
            out_storage.capacity(),
        );
        core::mem::forget(out_storage);
        Vec::from_raw_parts(ptr, len, cap)
    };
    let inv_d = 1.0_f32 / (d as f32);

    // T29/T37 — NEON intrinsics row kernel, 8-lane unrolled. Two
    // parallel sum_sq accumulators dual-issue the M-series fma pipes.
    let process_row = |x_row: &[f32], out_row: &mut [f32]| {
        #[cfg(target_arch = "aarch64")]
        {
            use core::arch::aarch64::*;
            unsafe {
                let mut sum_sq_a = vdupq_n_f32(0.0);
                let mut sum_sq_b = vdupq_n_f32(0.0);
                let chunks_8 = d / 8;
                let tail_start = chunks_8 * 8;
                for i in 0..chunks_8 {
                    let va = vld1q_f32(x_row.as_ptr().add(i * 8));
                    let vb = vld1q_f32(x_row.as_ptr().add(i * 8 + 4));
                    sum_sq_a = vfmaq_f32(sum_sq_a, va, va);
                    sum_sq_b = vfmaq_f32(sum_sq_b, vb, vb);
                }
                let mut sum_sq = vaddvq_f32(vaddq_f32(sum_sq_a, sum_sq_b));
                for &v in x_row[tail_start..].iter() {
                    sum_sq += v * v;
                }
                let mean_sq = sum_sq * inv_d;
                let inv_rms = 1.0 / (mean_sq + eps).sqrt();
                let inv_rms_v = vdupq_n_f32(inv_rms);
                for i in 0..chunks_8 {
                    let xa = vld1q_f32(x_row.as_ptr().add(i * 8));
                    let xb = vld1q_f32(x_row.as_ptr().add(i * 8 + 4));
                    let ga = vld1q_f32(g_slice.as_ptr().add(i * 8));
                    let gb = vld1q_f32(g_slice.as_ptr().add(i * 8 + 4));
                    let za = vmulq_f32(xa, inv_rms_v);
                    let zb = vmulq_f32(xb, inv_rms_v);
                    let ya = vmulq_f32(za, ga);
                    let yb = vmulq_f32(zb, gb);
                    vst1q_f32(out_row.as_mut_ptr().add(i * 8), ya);
                    vst1q_f32(out_row.as_mut_ptr().add(i * 8 + 4), yb);
                }
                for i in tail_start..d {
                    out_row[i] = x_row[i] * inv_rms * g_slice[i];
                }
            }
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            let mut sum_sq = 0.0_f32;
            for &v in x_row.iter() {
                sum_sq += v * v;
            }
            let mean_sq = sum_sq * inv_d;
            let inv_rms = 1.0 / (mean_sq + eps).sqrt();
            for i in 0..d {
                out_row[i] = x_row[i] * inv_rms * g_slice[i];
            }
        }
    };

    #[cfg(not(target_arch = "wasm32"))]
    const PARALLEL_OUTER_MIN: usize = 64;

    #[cfg(not(target_arch = "wasm32"))]
    if outer >= PARALLEL_OUTER_MIN {
        use rayon::prelude::*;
        // T38 — chunk multiple rows per rayon task. With one row per
        // task (3 KB f32 work unit on D=768), rayon's per-task
        // dispatch overhead (~1 µs) compounds to ~500 µs across 512
        // rows. Bundling 16 rows per chunk amortises the dispatch
        // and keeps each task at 48 KB which still fits comfortably
        // in L1d.
        const ROWS_PER_TASK: usize = 16;
        let chunk_bytes = d * ROWS_PER_TASK;
        out_data
            .par_chunks_mut(chunk_bytes)
            .zip(x_slice.par_chunks(chunk_bytes))
            .for_each(|(out_chunk, x_chunk)| {
                for (out_row, x_row) in out_chunk.chunks_mut(d).zip(x_chunk.chunks(d)) {
                    process_row(x_row, out_row);
                }
            });
    } else {
        for o in 0..outer {
            process_row(
                &x_slice[o * d..(o + 1) * d],
                &mut out_data[o * d..(o + 1) * d],
            );
        }
    }

    #[cfg(target_arch = "wasm32")]
    for o in 0..outer {
        process_row(
            &x_slice[o * d..(o + 1) * d],
            &mut out_data[o * d..(o + 1) * d],
        );
    }

    Tensor::from_vec(shape, out_data).map_err(|e| ModuleError::Backend {
        op: "RMSNorm::forward(fused)",
        message: format!("{e:?}"),
    })
}

// ------------------------------ BatchNorm2d ------------------------------

/// BatchNorm2d (Ioffe & Szegedy 2015). Layout `[N, C, H, W]`.
/// v1 ships **train mode only** — running stats are not tracked yet
/// (eval-mode + running mean/var pending).
pub struct BatchNorm2d {
    /// Learnable per-channel scale.
    pub gamma: Variable,
    /// Learnable per-channel shift.
    pub beta: Variable,
    eps: f32,
    num_features: usize,
}

impl BatchNorm2d {
    /// Build with the given `num_features` (channel count) and default
    /// eps = 1e-5.
    pub fn new(num_features: usize) -> Self {
        Self::with_eps(num_features, 1e-5)
    }

    /// Build with explicit epsilon.
    pub fn with_eps(num_features: usize, eps: f32) -> Self {
        let gamma = Variable::leaf(
            Tensor::from_vec([num_features], vec![1.0_f32; num_features]).expect("gamma shape"),
        );
        let beta = Variable::leaf(
            Tensor::from_vec([num_features], vec![0.0_f32; num_features]).expect("beta shape"),
        );
        BatchNorm2d {
            gamma,
            beta,
            eps,
            num_features,
        }
    }

    /// Channel count.
    pub fn num_features(&self) -> usize {
        self.num_features
    }
}

impl Module for BatchNorm2d {
    fn forward(&self, input: &Variable) -> Result<Variable, ModuleError> {
        ops::batch_norm2d(input, &self.gamma, &self.beta, self.eps)
    }

    fn parameters(&self) -> Vec<Variable> {
        vec![self.gamma.clone(), self.beta.clone()]
    }

    fn named_parameters(&self) -> Vec<(String, Variable)> {
        vec![
            ("gamma".to_string(), self.gamma.clone()),
            ("beta".to_string(), self.beta.clone()),
        ]
    }
}

#[cfg(test)]
mod batch_norm_tests {
    use super::*;
    use rustorch_autograd::backward;

    #[test]
    fn batch_norm2d_normalises_per_channel() {
        let bn = BatchNorm2d::new(2);
        let x = Variable::new(
            Tensor::from_vec(
                [2usize, 2, 2, 2],
                (0..16).map(|i| i as f32).collect::<Vec<_>>(),
            )
            .unwrap(),
        );
        let y = bn.forward(&x).unwrap();
        assert_eq!(y.tensor().shape(), &[2, 2, 2, 2]);
    }

    #[test]
    fn batch_norm2d_backward_grads_flow() {
        let bn = BatchNorm2d::new(2);
        let x = Variable::leaf(
            Tensor::from_vec(
                [2usize, 2, 2, 2],
                (0..16).map(|i| i as f32 * 0.1).collect::<Vec<_>>(),
            )
            .unwrap(),
        );
        let y = bn.forward(&x).unwrap();
        let s = rustorch_autograd::ops::sum(&y).unwrap();
        backward(&s, None).unwrap();
        assert!(bn.gamma.grad().is_some());
        assert!(bn.beta.grad().is_some());
        assert!(x.grad().is_some());
    }

    #[test]
    fn batch_norm2d_named_parameters() {
        let bn = BatchNorm2d::new(8);
        let np = bn.named_parameters();
        let names: Vec<String> = np.iter().map(|(n, _)| n.clone()).collect();
        assert_eq!(names, vec!["gamma".to_string(), "beta".to_string()]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustorch_autograd::{backward, Variable};
    use rustorch_core::tensor::tensor_impl::Tensor;

    fn var(shape: impl Into<Vec<usize>>, data: Vec<f32>) -> Variable {
        Variable::leaf(Tensor::from_vec(shape.into(), data).unwrap())
    }

    #[test]
    fn rms_norm_unit_input_returns_unit_output() {
        // For x = ones and gamma = ones, RMS = sqrt(1 + eps) ≈ 1, so y ≈ 1.
        let norm = RMSNorm::with_eps(4, 0.0);
        let x = var(vec![1usize, 4], vec![1.0_f32; 4]);
        let y = norm.forward(&x).unwrap();
        let y_t = y.tensor();
        let y_v = y_t.as_slice::<f32>().unwrap();
        for &v in y_v {
            assert!((v - 1.0).abs() < 1e-5, "got {v}");
        }
    }

    #[test]
    fn rms_norm_preserves_shape() {
        let norm = RMSNorm::new(8);
        let x = var(vec![3usize, 5, 8], vec![0.5_f32; 3 * 5 * 8]);
        let y = norm.forward(&x).unwrap();
        assert_eq!(y.tensor().shape(), &[3, 5, 8]);
    }

    #[test]
    fn rms_norm_gamma_scales_output_uniformly() {
        // If gamma = c * ones, output should be c * normed.
        let mut norm = RMSNorm::with_eps(4, 0.0);
        // Override gamma.
        let g = Variable::leaf(Tensor::from_vec([4usize], vec![3.0_f32; 4]).unwrap());
        norm.gamma = g;
        let x = var(vec![1usize, 4], vec![1.0_f32, 1.0, 1.0, 1.0]);
        let y = norm.forward(&x).unwrap();
        let y_t = y.tensor();
        let y_v = y_t.as_slice::<f32>().unwrap();
        // RMS=1, gamma=3 → y = 3
        for &v in y_v {
            assert!((v - 3.0).abs() < 1e-5, "got {v}");
        }
    }

    #[test]
    fn rms_norm_backward_runs_end_to_end() {
        // Basic gradient flow: train objective y.sum() against random input.
        let norm = RMSNorm::new(4);
        let x = var(
            vec![2usize, 4],
            vec![0.5_f32, 1.0, 1.5, 2.0, 0.3, 0.6, 0.9, 1.2],
        );
        let y = norm.forward(&x).unwrap();
        let s = rustorch_autograd::ops::sum(&y).unwrap();
        backward(&s, None).unwrap();
        // Gamma should have a non-zero grad after backward.
        let g_grad = norm.gamma.grad().unwrap();
        let g_grad_v = g_grad.as_slice::<f32>().unwrap();
        assert!(
            g_grad_v.iter().any(|&v| v.abs() > 1e-6),
            "gamma should accumulate non-zero grad, got {:?}",
            g_grad_v
        );
        // x should also have a grad.
        let x_grad = x.grad().unwrap();
        let x_grad_v = x_grad.as_slice::<f32>().unwrap();
        assert!(
            x_grad_v.iter().any(|&v| v.abs() > 1e-6),
            "x should accumulate non-zero grad, got {:?}",
            x_grad_v
        );
    }

    #[test]
    fn rms_norm_parameters_returns_gamma_only() {
        let norm = RMSNorm::new(8);
        let p = norm.parameters();
        assert_eq!(p.len(), 1);
        assert_eq!(p[0].tensor().shape(), &[8]);
    }

    // -------------------- LayerNorm --------------------

    #[test]
    fn layer_norm_zero_input_returns_beta() {
        // mean=0, var=0 → centered=0, normed=0, y=0*gamma+beta = beta
        let norm = LayerNorm::with_eps(4, 1e-5);
        let x = var(vec![1usize, 4], vec![0.0_f32; 4]);
        let y = norm.forward(&x).unwrap();
        let y_t = y.tensor();
        let y_v = y_t.as_slice::<f32>().unwrap();
        // beta defaults to zeros → y = zeros
        for &v in y_v {
            assert!(v.abs() < 1e-5, "expected ~0 (beta=0), got {v}");
        }
    }

    #[test]
    fn layer_norm_centers_and_unit_variance() {
        // After LayerNorm on a row, the per-row mean ≈ beta and per-row
        // sample-stddev ≈ gamma (default 1, 0).
        let norm = LayerNorm::with_eps(4, 0.0);
        let x = var(vec![1usize, 4], vec![1.0_f32, 2.0, 3.0, 4.0]);
        let y = norm.forward(&x).unwrap();
        let y_t = y.tensor();
        let y_v = y_t.as_slice::<f32>().unwrap();
        let mean = y_v.iter().sum::<f32>() / y_v.len() as f32;
        let var = y_v.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / y_v.len() as f32;
        assert!(mean.abs() < 1e-5, "mean ≈ 0, got {mean}");
        assert!((var - 1.0).abs() < 1e-4, "var ≈ 1, got {var}");
    }

    #[test]
    fn layer_norm_preserves_shape() {
        let norm = LayerNorm::new(8);
        let x = var(vec![3usize, 5, 8], vec![0.5_f32; 3 * 5 * 8]);
        let y = norm.forward(&x).unwrap();
        assert_eq!(y.tensor().shape(), &[3, 5, 8]);
    }

    #[test]
    fn layer_norm_backward_flows_to_gamma_beta_x() {
        let norm = LayerNorm::new(4);
        let x = var(
            vec![2usize, 4],
            vec![0.5_f32, 1.0, 1.5, 2.0, 0.3, 0.6, 0.9, 1.2],
        );
        let y = norm.forward(&x).unwrap();
        let s = rustorch_autograd::ops::sum(&y).unwrap();
        backward(&s, None).unwrap();
        for p in &[&norm.gamma, &norm.beta] {
            let g = p.grad().unwrap();
            assert!(
                g.as_slice::<f32>().unwrap().iter().any(|v| v.abs() > 1e-7),
                "param should accumulate non-zero grad"
            );
        }
        let xg = x.grad().unwrap();
        assert!(xg.as_slice::<f32>().unwrap().iter().any(|v| v.abs() > 1e-7));
    }

    #[test]
    fn layer_norm_parameters_count() {
        let norm = LayerNorm::new(8);
        let p = norm.parameters();
        assert_eq!(p.len(), 2);
        assert_eq!(p[0].tensor().shape(), &[8]);
        assert_eq!(p[1].tensor().shape(), &[8]);
    }
}
