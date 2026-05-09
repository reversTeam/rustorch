//! Custom CUDA kernels pour LLM inference (T241.1).
//!
//! Compile à la volée via nvrtc, cache les modules, dispatche depuis Rust.
//! Référence : rustorch-metal/src/kernels.rs (port pattern-pour-pattern).
//!
//! Kernels BF16 disponibles :
//! - `rms_norm_bf16` — RMSNorm inplace (Qwen, Llama)
//! - `silu_bf16` — SiLU/Swish activation inplace
//! - `swiglu_bf16` — fused silu(gate) * up (sortie BF16)
//! - `rope_half_split_bf16` — Rotary position embeddings on Q,K
//! - `embedding_lookup_bf16` — gather rows from token embedding table
//! - `argmax_bf16` — greedy sampling (returns u32 token id)
//!
//! ## Cache strategy
//!
//! Chaque kernel est compilé une fois sur la première utilisation
//! (paresseux). Le PTX et le `CudaFunction` sont stockés dans un
//! `LlmKernels` que le caller passe à chaque call.

#[cfg(feature = "cuda")]
use crate::error::CudaError;

#[cfg(feature = "cuda")]
use cudarc::driver::{CudaContext, CudaFunction, CudaModule, CudaStream, PushKernelArg};
#[cfg(feature = "cuda")]
use std::sync::Arc;

#[cfg(feature = "cuda")]
const RMS_NORM_BF16_SRC: &str = r#"
#include <cuda_bf16.h>

extern "C" __global__ void rms_norm_bf16(
    __nv_bfloat16* __restrict__ x,            // [batch, n] in/out
    const __nv_bfloat16* __restrict__ gamma,  // [n]
    float eps,
    int n,
    int batch
) {
    int b = blockIdx.x;
    if (b >= batch) return;

    extern __shared__ float sdata[];

    // 1. Sum of squares
    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float v = (float)x[b * n + i];
        sum_sq += v * v;
    }
    sdata[threadIdx.x] = sum_sq;
    __syncthreads();

    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }

    float inv_rms = rsqrtf(sdata[0] / (float)n + eps);

    // 2. Normalize + gamma
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float v = (float)x[b * n + i] * inv_rms * (float)gamma[i];
        x[b * n + i] = (__nv_bfloat16)v;
    }
}
"#;

#[cfg(feature = "cuda")]
const SILU_BF16_SRC: &str = r#"
#include <cuda_bf16.h>

extern "C" __global__ void silu_bf16(__nv_bfloat16* __restrict__ x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = (float)x[i];
    float s = v / (1.0f + expf(-v));    // silu(v) = v * sigmoid(v)
    x[i] = (__nv_bfloat16)s;
}
"#;

#[cfg(feature = "cuda")]
const SWIGLU_BF16_SRC: &str = r#"
#include <cuda_bf16.h>

// Fused : out[i] = silu(gate[i]) * up[i]
// Note : gate and up may be the two halves of a fused matmul output
// (cublasLt fused gate+up matmul), so the caller passes pointers to
// the halves explicitly.
extern "C" __global__ void swiglu_bf16(
    const __nv_bfloat16* __restrict__ gate,
    const __nv_bfloat16* __restrict__ up,
    __nv_bfloat16* __restrict__ out,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = (float)gate[i];
    float u = (float)up[i];
    float s = g / (1.0f + expf(-g));    // silu(g)
    out[i] = (__nv_bfloat16)(s * u);
}
"#;

#[cfg(feature = "cuda")]
const ROPE_HALF_SPLIT_BF16_SRC: &str = r#"
#include <cuda_bf16.h>

// "Half-split" RoPE (the PyTorch / Llama / Qwen convention) :
//   For dim_pair p in [0, head_dim/2) :
//     a = x[b, h, 2p]
//     b = x[b, h, 2p+1]
//     theta = inv_freq[p] * pos
//     cos_p = cos(theta), sin_p = sin(theta)
//     x[b, h, 2p]   = a * cos_p - b * sin_p
//     x[b, h, 2p+1] = a * sin_p + b * cos_p
//
// `inv_freq` is precomputed [head_dim / 2] : 1 / (theta_base ^ (2p / head_dim))
//
// Layout : x is (n_heads, head_dim) for ONE token at position `pos`.
// For multi-token prefill, launch with grid_dim.y = seq and seq_offset.
extern "C" __global__ void rope_half_split_bf16(
    __nv_bfloat16* __restrict__ x,
    const float* __restrict__ inv_freq,   // [head_dim/2]
    int pos,
    int n_heads,
    int head_dim
) {
    int h = blockIdx.x;
    if (h >= n_heads) return;
    int half = head_dim / 2;
    int p = blockIdx.y * blockDim.x + threadIdx.x;
    if (p >= half) return;

    float theta = inv_freq[p] * (float)pos;
    float cos_p, sin_p;
    sincosf(theta, &sin_p, &cos_p);

    int base = h * head_dim + 2 * p;
    float a = (float)x[base];
    float b = (float)x[base + 1];
    x[base]     = (__nv_bfloat16)(a * cos_p - b * sin_p);
    x[base + 1] = (__nv_bfloat16)(a * sin_p + b * cos_p);
}
"#;

#[cfg(feature = "cuda")]
const EMBEDDING_LOOKUP_BF16_SRC: &str = r#"
#include <cuda_bf16.h>

// Gather rows from embedding table.
//   table : [vocab, hidden] BF16
//   ids   : [seq] u32
//   out   : [seq, hidden] BF16
extern "C" __global__ void embedding_lookup_bf16(
    const __nv_bfloat16* __restrict__ table,
    const unsigned int* __restrict__ ids,
    __nv_bfloat16* __restrict__ out,
    int seq,
    int hidden
) {
    int s = blockIdx.x;
    if (s >= seq) return;
    int id = (int)ids[s];
    int i = blockIdx.y * blockDim.x + threadIdx.x;
    if (i >= hidden) return;
    out[s * hidden + i] = table[id * hidden + i];
}
"#;

#[cfg(feature = "cuda")]
const ARGMAX_BF16_SRC: &str = r#"
#include <cuda_bf16.h>

// Greedy sampling : argmax over BF16 logits, single block.
extern "C" __global__ void argmax_bf16(
    const __nv_bfloat16* __restrict__ logits,
    unsigned int* __restrict__ out,
    int n
) {
    extern __shared__ float sdata[];
    int* sidx = (int*)(sdata + blockDim.x);

    float local_max = -1e30f;
    int local_idx = 0;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float v = (float)logits[i];
        if (v > local_max) { local_max = v; local_idx = i; }
    }
    sdata[threadIdx.x] = local_max;
    sidx[threadIdx.x]  = local_idx;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            if (sdata[threadIdx.x + s] > sdata[threadIdx.x]) {
                sdata[threadIdx.x] = sdata[threadIdx.x + s];
                sidx[threadIdx.x]  = sidx[threadIdx.x + s];
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) *out = (unsigned int)sidx[0];
}
"#;

#[cfg(feature = "cuda")]
const ADD_INPLACE_BF16_SRC: &str = r#"
#include <cuda_bf16.h>

// y[i] += x[i]
extern "C" __global__ void add_inplace_bf16(
    __nv_bfloat16* __restrict__ y,
    const __nv_bfloat16* __restrict__ x,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float a = (float)y[i];
    float b = (float)x[i];
    y[i] = (__nv_bfloat16)(a + b);
}
"#;

#[cfg(feature = "cuda")]
const COPY_BF16_SRC: &str = r#"
#include <cuda_bf16.h>

// dst[i] = src[i]
extern "C" __global__ void copy_bf16(
    __nv_bfloat16* __restrict__ dst,
    const __nv_bfloat16* __restrict__ src,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[i] = src[i];
}
"#;

#[cfg(feature = "cuda")]
const KV_APPEND_BF16_SRC: &str = r#"
#include <cuda_bf16.h>

// Append k_in [kv_dim] / v_in [kv_dim] to KV cache at position `pos`.
// KV cache layout : [n_kv, max_seq, head_dim] = [kv_dim_groups, max_seq * head_dim]
// (compact storage : groupe kv_h occupe max_seq * head_dim contigus).
//
// Indexing : k_cache[kv_h * max_seq * head_dim + pos * head_dim + i] = k_in[kv_h * head_dim + i]
extern "C" __global__ void kv_append_bf16(
    __nv_bfloat16* __restrict__ k_cache,
    __nv_bfloat16* __restrict__ v_cache,
    const __nv_bfloat16* __restrict__ k_in,
    const __nv_bfloat16* __restrict__ v_in,
    int pos,
    int n_kv,
    int head_dim,
    int max_seq
) {
    int kv_h = blockIdx.x;
    if (kv_h >= n_kv) return;
    int i = blockIdx.y * blockDim.x + threadIdx.x;
    if (i >= head_dim) return;

    int cache_off = kv_h * max_seq * head_dim + pos * head_dim + i;
    int in_off = kv_h * head_dim + i;
    k_cache[cache_off] = k_in[in_off];
    v_cache[cache_off] = v_in[in_off];
}
"#;

#[cfg(feature = "cuda")]
const GQA_DECODE_NAIVE_BF16_SRC: &str = r#"
#include <cuda_bf16.h>

// Naive GQA decode (single-token Q against KV cache prefix).
//
// Inputs :
//   q     : [n_heads, head_dim]            BF16  current token Q
//   k_cache : [n_kv, max_seq, head_dim]   BF16  cache (only first kv_len valid)
//   v_cache : same shape                    BF16
//   out   : [n_heads, head_dim]            BF16  attention output (one token)
//
// One block per head. Threads cooperate to compute scores + softmax + V·P.
// Shared memory : kv_len floats for scores.
extern "C" __global__ void gqa_decode_naive_bf16(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k_cache,
    const __nv_bfloat16* __restrict__ v_cache,
    __nv_bfloat16* __restrict__ out,
    int n_heads,
    int n_kv,
    int kv_len,
    int head_dim,
    int max_seq,
    float scale       // 1.0 / sqrt(head_dim)
) {
    int h = blockIdx.x;
    if (h >= n_heads) return;
    int kv_h = h * n_kv / n_heads;          // GQA group mapping

    extern __shared__ float scores[];        // [kv_len]

    // 1. Pass : compute scores[t] = Q[h] · K[kv_h, t] * scale, find max
    float local_max = -1e30f;
    for (int t = threadIdx.x; t < kv_len; t += blockDim.x) {
        float dot = 0.0f;
        for (int i = 0; i < head_dim; ++i) {
            float qi = (float)q[h * head_dim + i];
            float ki = (float)k_cache[(kv_h * max_seq + t) * head_dim + i];
            dot += qi * ki;
        }
        scores[t] = dot * scale;
        if (scores[t] > local_max) local_max = scores[t];
    }

    // Block-reduce max
    __shared__ float block_max;
    if (threadIdx.x == 0) block_max = -1e30f;
    __syncthreads();
    atomicMax((int*)&block_max, __float_as_int(local_max));
    __syncthreads();
    float max_score = block_max;

    // 2. Pass : exp(scores - max) and sum
    float local_sum = 0.0f;
    for (int t = threadIdx.x; t < kv_len; t += blockDim.x) {
        scores[t] = expf(scores[t] - max_score);
        local_sum += scores[t];
    }
    __shared__ float block_sum;
    if (threadIdx.x == 0) block_sum = 0.0f;
    __syncthreads();
    atomicAdd(&block_sum, local_sum);
    __syncthreads();
    float sum = block_sum;

    // 3. Pass : out[h, i] = sum_t (scores[t]/sum) * V[kv_h, t, i]
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        float acc = 0.0f;
        for (int t = 0; t < kv_len; ++t) {
            float vi = (float)v_cache[(kv_h * max_seq + t) * head_dim + i];
            acc += (scores[t] / sum) * vi;
        }
        out[h * head_dim + i] = (__nv_bfloat16)acc;
    }
}
"#;

/// Container des kernels LLM CUDA, compilés paresseusement et cachés.
#[cfg(feature = "cuda")]
pub struct LlmKernels {
    ctx: Arc<CudaContext>,
    rms_norm: std::sync::OnceLock<(Arc<CudaModule>, CudaFunction)>,
    silu: std::sync::OnceLock<(Arc<CudaModule>, CudaFunction)>,
    swiglu: std::sync::OnceLock<(Arc<CudaModule>, CudaFunction)>,
    rope: std::sync::OnceLock<(Arc<CudaModule>, CudaFunction)>,
    embed: std::sync::OnceLock<(Arc<CudaModule>, CudaFunction)>,
    argmax: std::sync::OnceLock<(Arc<CudaModule>, CudaFunction)>,
    add_inplace: std::sync::OnceLock<(Arc<CudaModule>, CudaFunction)>,
    copy: std::sync::OnceLock<(Arc<CudaModule>, CudaFunction)>,
    kv_append: std::sync::OnceLock<(Arc<CudaModule>, CudaFunction)>,
    gqa_decode: std::sync::OnceLock<(Arc<CudaModule>, CudaFunction)>,
}

#[cfg(feature = "cuda")]
impl LlmKernels {
    /// Crée un container de kernels lié au contexte CUDA. Les kernels
    /// sont compilés au premier appel (lazy via OnceLock).
    pub fn new(ctx: Arc<CudaContext>) -> Self {
        Self {
            ctx,
            rms_norm: std::sync::OnceLock::new(),
            silu: std::sync::OnceLock::new(),
            swiglu: std::sync::OnceLock::new(),
            rope: std::sync::OnceLock::new(),
            embed: std::sync::OnceLock::new(),
            argmax: std::sync::OnceLock::new(),
            add_inplace: std::sync::OnceLock::new(),
            copy: std::sync::OnceLock::new(),
            kv_append: std::sync::OnceLock::new(),
            gqa_decode: std::sync::OnceLock::new(),
        }
    }

    fn compile_or_get(
        &self,
        slot: &std::sync::OnceLock<(Arc<CudaModule>, CudaFunction)>,
        src: &str,
        name: &str,
    ) -> Result<(Arc<CudaModule>, CudaFunction), CudaError> {
        if let Some(pair) = slot.get() {
            return Ok((pair.0.clone(), pair.1.clone()));
        }
        // PTX forward-compatibility chain : on essaie plusieurs arch dans
        // l'ordre. Avec cudarc compilé contre CUDA 13.0 ABI, nvrtc peut ne
        // pas connaître sm_121 (GB10) — fallback vers archs plus anciens
        // que le driver runtime peut JIT-recompiler vers sm_121.
        // Override via RUSTORCH_NVRTC_ARCH=sm_xxx pour forcer.
        let candidates: Vec<&'static str> = if let Ok(forced) = std::env::var("RUSTORCH_NVRTC_ARCH")
        {
            let s: &'static str = Box::leak(forced.into_boxed_str());
            vec![s]
        } else {
            vec![
                "sm_121", "sm_120", "sm_100", "sm_90", "sm_89", "sm_86", "sm_80",
            ]
        };
        let mut last_err: Option<String> = None;
        let mut ptx_opt: Option<cudarc::nvrtc::Ptx> = None;
        for arch in &candidates {
            let opts = cudarc::nvrtc::CompileOptions {
                arch: Some(*arch),
                include_paths: vec![
                    "/usr/local/cuda/include".to_string(),
                    "/usr/local/cuda-13.2/include".to_string(),
                    "/usr/local/cuda-13.0/include".to_string(),
                ],
                use_fast_math: Some(true),
                ..Default::default()
            };
            match cudarc::nvrtc::compile_ptx_with_opts(src, opts) {
                Ok(p) => {
                    ptx_opt = Some(p);
                    eprintln!("[llm_kernels] {name} compiled with arch={arch}");
                    break;
                },
                Err(e) => {
                    last_err = Some(format!("{arch}: {e:?}"));
                },
            }
        }
        let ptx = ptx_opt.ok_or_else(|| CudaError::Unsupported {
            msg: format!(
                "nvrtc compile {name} failed for all arch candidates: {:?}",
                last_err
            ),
        })?;
        let module = self.ctx.load_module(ptx).map_err(|e| CudaError::Driver {
            code: format!("{e:?}").len() as i32,
            location: "LlmKernels::load_module",
        })?;
        let func = module.load_function(name).map_err(|e| CudaError::Driver {
            code: format!("{e:?}").len() as i32,
            location: "LlmKernels::load_function",
        })?;
        let _ = slot.set((module.clone(), func.clone()));
        Ok((module, func))
    }

    /// RMSNorm inplace : x[b, i] = x[b, i] / sqrt(mean(x[b]²) + eps) · gamma[i].
    ///
    /// # Safety
    /// `x` and `gamma` must be valid CudaSlice<bf16>. `n` = hidden size,
    /// `batch` = nombre de rows.
    pub unsafe fn rms_norm_bf16(
        &self,
        stream: &Arc<CudaStream>,
        x: u64,     // device ptr to bf16 buffer [batch * n]
        gamma: u64, // device ptr to bf16 [n]
        eps: f32,
        n: i32,
        batch: i32,
    ) -> Result<(), CudaError> {
        let (_module, func) =
            self.compile_or_get(&self.rms_norm, RMS_NORM_BF16_SRC, "rms_norm_bf16")?;
        // 256 threads per block, shared mem = 256 * 4 bytes
        let block_dim = 256u32;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (batch as u32, 1, 1),
            block_dim: (block_dim, 1, 1),
            shared_mem_bytes: block_dim * 4,
        };
        let mut launcher = stream.launch_builder(&func);
        launcher.arg(&x).arg(&gamma).arg(&eps).arg(&n).arg(&batch);
        launcher.launch(cfg).map_err(|e| CudaError::Driver {
            code: format!("{e:?}").len() as i32,
            location: "rms_norm_bf16::launch",
        })?;
        Ok(())
    }

    /// SiLU (Swish) inplace : x[i] = x[i] / (1 + exp(-x[i])).
    pub unsafe fn silu_bf16(
        &self,
        stream: &Arc<CudaStream>,
        x: u64,
        n: i32,
    ) -> Result<(), CudaError> {
        let (_module, func) = self.compile_or_get(&self.silu, SILU_BF16_SRC, "silu_bf16")?;
        let block_dim = 256u32;
        let grid_dim = (n as u32 + block_dim - 1) / block_dim;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (grid_dim, 1, 1),
            block_dim: (block_dim, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut launcher = stream.launch_builder(&func);
        launcher.arg(&x).arg(&n);
        launcher.launch(cfg).map_err(|e| CudaError::Driver {
            code: format!("{e:?}").len() as i32,
            location: "silu_bf16::launch",
        })?;
        Ok(())
    }

    /// SwiGLU fused : out[i] = silu(gate[i]) * up[i].
    pub unsafe fn swiglu_bf16(
        &self,
        stream: &Arc<CudaStream>,
        gate: u64,
        up: u64,
        out: u64,
        n: i32,
    ) -> Result<(), CudaError> {
        let (_module, func) = self.compile_or_get(&self.swiglu, SWIGLU_BF16_SRC, "swiglu_bf16")?;
        let block_dim = 256u32;
        let grid_dim = (n as u32 + block_dim - 1) / block_dim;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (grid_dim, 1, 1),
            block_dim: (block_dim, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut launcher = stream.launch_builder(&func);
        launcher.arg(&gate).arg(&up).arg(&out).arg(&n);
        launcher.launch(cfg).map_err(|e| CudaError::Driver {
            code: format!("{e:?}").len() as i32,
            location: "swiglu_bf16::launch",
        })?;
        Ok(())
    }

    /// RoPE half-split inplace sur (n_heads, head_dim) pour position `pos`.
    pub unsafe fn rope_half_split_bf16(
        &self,
        stream: &Arc<CudaStream>,
        x: u64,
        inv_freq: u64,
        pos: i32,
        n_heads: i32,
        head_dim: i32,
    ) -> Result<(), CudaError> {
        let (_module, func) =
            self.compile_or_get(&self.rope, ROPE_HALF_SPLIT_BF16_SRC, "rope_half_split_bf16")?;
        let half = head_dim / 2;
        let block_dim = 64u32.min(half as u32);
        let grid_y = (half as u32 + block_dim - 1) / block_dim;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (n_heads as u32, grid_y, 1),
            block_dim: (block_dim, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut launcher = stream.launch_builder(&func);
        launcher
            .arg(&x)
            .arg(&inv_freq)
            .arg(&pos)
            .arg(&n_heads)
            .arg(&head_dim);
        launcher.launch(cfg).map_err(|e| CudaError::Driver {
            code: format!("{e:?}").len() as i32,
            location: "rope_half_split_bf16::launch",
        })?;
        Ok(())
    }

    /// Embedding lookup : pour chaque id dans `ids[0..seq]`, copie la
    /// row correspondante de `table` dans `out`.
    pub unsafe fn embedding_lookup_bf16(
        &self,
        stream: &Arc<CudaStream>,
        table: u64,
        ids: u64,
        out: u64,
        seq: i32,
        hidden: i32,
    ) -> Result<(), CudaError> {
        let (_module, func) = self.compile_or_get(
            &self.embed,
            EMBEDDING_LOOKUP_BF16_SRC,
            "embedding_lookup_bf16",
        )?;
        let block_dim = 256u32;
        let grid_y = (hidden as u32 + block_dim - 1) / block_dim;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (seq as u32, grid_y, 1),
            block_dim: (block_dim, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut launcher = stream.launch_builder(&func);
        launcher
            .arg(&table)
            .arg(&ids)
            .arg(&out)
            .arg(&seq)
            .arg(&hidden);
        launcher.launch(cfg).map_err(|e| CudaError::Driver {
            code: format!("{e:?}").len() as i32,
            location: "embedding_lookup_bf16::launch",
        })?;
        Ok(())
    }

    /// Greedy sampling : argmax sur les BF16 logits dans un seul kernel.
    /// Écrit le token id (u32) dans `out`.
    pub unsafe fn argmax_bf16(
        &self,
        stream: &Arc<CudaStream>,
        logits: u64,
        out: u64,
        n: i32,
    ) -> Result<(), CudaError> {
        let (_module, func) = self.compile_or_get(&self.argmax, ARGMAX_BF16_SRC, "argmax_bf16")?;
        let block_dim = 256u32;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (block_dim, 1, 1),
            shared_mem_bytes: block_dim * 8, // float + int per thread
        };
        let mut launcher = stream.launch_builder(&func);
        launcher.arg(&logits).arg(&out).arg(&n);
        launcher.launch(cfg).map_err(|e| CudaError::Driver {
            code: format!("{e:?}").len() as i32,
            location: "argmax_bf16::launch",
        })?;
        Ok(())
    }

    /// Add inplace : y += x.
    pub unsafe fn add_inplace_bf16(
        &self,
        stream: &Arc<CudaStream>,
        y: u64,
        x: u64,
        n: i32,
    ) -> Result<(), CudaError> {
        let (_module, func) =
            self.compile_or_get(&self.add_inplace, ADD_INPLACE_BF16_SRC, "add_inplace_bf16")?;
        let block_dim = 256u32;
        let grid_dim = (n as u32 + block_dim - 1) / block_dim;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (grid_dim, 1, 1),
            block_dim: (block_dim, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut launcher = stream.launch_builder(&func);
        launcher.arg(&y).arg(&x).arg(&n);
        launcher.launch(cfg).map_err(|e| CudaError::Driver {
            code: format!("{e:?}").len() as i32,
            location: "add_inplace_bf16::launch",
        })?;
        Ok(())
    }

    /// Copy device-to-device : dst[i] = src[i] (BF16).
    pub unsafe fn copy_bf16(
        &self,
        stream: &Arc<CudaStream>,
        dst: u64,
        src: u64,
        n: i32,
    ) -> Result<(), CudaError> {
        let (_module, func) = self.compile_or_get(&self.copy, COPY_BF16_SRC, "copy_bf16")?;
        let block_dim = 256u32;
        let grid_dim = (n as u32 + block_dim - 1) / block_dim;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (grid_dim, 1, 1),
            block_dim: (block_dim, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut launcher = stream.launch_builder(&func);
        launcher.arg(&dst).arg(&src).arg(&n);
        launcher.launch(cfg).map_err(|e| CudaError::Driver {
            code: format!("{e:?}").len() as i32,
            location: "copy_bf16::launch",
        })?;
        Ok(())
    }

    /// Append (k_in, v_in) to KV cache at position `pos`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn kv_append_bf16(
        &self,
        stream: &Arc<CudaStream>,
        k_cache: u64,
        v_cache: u64,
        k_in: u64,
        v_in: u64,
        pos: i32,
        n_kv: i32,
        head_dim: i32,
        max_seq: i32,
    ) -> Result<(), CudaError> {
        let (_module, func) =
            self.compile_or_get(&self.kv_append, KV_APPEND_BF16_SRC, "kv_append_bf16")?;
        let block_dim = 64u32.min(head_dim as u32);
        let grid_y = (head_dim as u32 + block_dim - 1) / block_dim;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (n_kv as u32, grid_y, 1),
            block_dim: (block_dim, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut launcher = stream.launch_builder(&func);
        launcher
            .arg(&k_cache)
            .arg(&v_cache)
            .arg(&k_in)
            .arg(&v_in)
            .arg(&pos)
            .arg(&n_kv)
            .arg(&head_dim)
            .arg(&max_seq);
        launcher.launch(cfg).map_err(|e| CudaError::Driver {
            code: format!("{e:?}").len() as i32,
            location: "kv_append_bf16::launch",
        })?;
        Ok(())
    }

    /// Naive GQA decode : Q against KV cache prefix.
    ///
    /// Q : [n_heads, head_dim] BF16  (single-token query)
    /// K/V cache : [n_kv, max_seq, head_dim] BF16  (only first kv_len valid)
    /// out : [n_heads, head_dim] BF16
    ///
    /// Each block handles one head, threads cooperate on softmax + V matmul.
    /// Shared mem = kv_len * 4 bytes (scores).
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn gqa_decode_naive_bf16(
        &self,
        stream: &Arc<CudaStream>,
        q: u64,
        k_cache: u64,
        v_cache: u64,
        out: u64,
        n_heads: i32,
        n_kv: i32,
        kv_len: i32,
        head_dim: i32,
        max_seq: i32,
    ) -> Result<(), CudaError> {
        let (_module, func) = self.compile_or_get(
            &self.gqa_decode,
            GQA_DECODE_NAIVE_BF16_SRC,
            "gqa_decode_naive_bf16",
        )?;
        let scale: f32 = 1.0 / (head_dim as f32).sqrt();
        let block_dim = 128u32.min(head_dim as u32);
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (n_heads as u32, 1, 1),
            block_dim: (block_dim, 1, 1),
            shared_mem_bytes: (kv_len as u32) * 4 + 8, // scores + block_max + block_sum
        };
        let mut launcher = stream.launch_builder(&func);
        launcher
            .arg(&q)
            .arg(&k_cache)
            .arg(&v_cache)
            .arg(&out)
            .arg(&n_heads)
            .arg(&n_kv)
            .arg(&kv_len)
            .arg(&head_dim)
            .arg(&max_seq)
            .arg(&scale);
        launcher.launch(cfg).map_err(|e| CudaError::Driver {
            code: format!("{e:?}").len() as i32,
            location: "gqa_decode_naive_bf16::launch",
        })?;
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────
// Stub sans cuda
// ─────────────────────────────────────────────────────────────────────────

#[cfg(not(feature = "cuda"))]
/// Stub when cuda feature is OFF.
pub struct LlmKernels;

#[cfg(not(feature = "cuda"))]
impl LlmKernels {
    /// Stub — returns Err when called without --features cuda.
    pub fn new<T>(_ctx: T) -> Self {
        Self
    }
}
