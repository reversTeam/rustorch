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

// "Half-split" RoPE — the HuggingFace `apply_rotary_pos_emb` convention
// used by Llama / Qwen / Mistral / Phi when loaded from HF or GGUF :
//
//   For dim k in [0, head_dim/2) :
//     a   = x[b, h, k]            (first half)
//     b   = x[b, h, k + half]     (second half)
//     theta = inv_freq[k] * pos
//     cos_k = cos(theta), sin_k = sin(theta)
//     x[b, h, k]        = a * cos_k - b * sin_k
//     x[b, h, k + half] = a * sin_k + b * cos_k
//
// NOT to be confused with the GPT-NeoX / RoFormer "interleaved" convention
// which pairs (2k, 2k+1). HF weights are baked for the half-split layout
// so using interleaved here scrambles Q/K and breaks attention.
//
// `inv_freq` is precomputed [head_dim / 2] : 1 / (theta_base ^ (2k / head_dim))
//
// Layout : x is (n_heads, head_dim) for ONE token at position `pos`.
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
    int k = blockIdx.y * blockDim.x + threadIdx.x;
    if (k >= half) return;

    float theta = inv_freq[k] * (float)pos;
    float cos_k, sin_k;
    sincosf(theta, &sin_k, &cos_k);

    int row = h * head_dim;
    float a = (float)x[row + k];
    float b = (float)x[row + k + half];
    x[row + k]        = (__nv_bfloat16)(a * cos_k - b * sin_k);
    x[row + k + half] = (__nv_bfloat16)(a * sin_k + b * cos_k);
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
const QUANTIZE_BF16_TO_NVFP4_SRC: &str = r#"
#include <cuda_bf16.h>

// BF16 → NVFP4 (E2M1) avec scales VEC16_UE4M3.
//
// Inputs:
//   x_bf16  : [n] BF16 input
// Outputs:
//   out_fp4 : [n/2] u8 packed (1 byte = 2 FP4 elements, low nibble = even index)
//   out_scale : [n/16] u8 UE4M3 (1 byte par bloc de 16 elements)
//
// FP4 E2M1 format : signe 1 bit, exp 2 bits (bias 1), mantissa 1 bit
// Valeurs représentables : ±0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6
//   (les magnitudes sont 0, 0.5, 1, 1.5, 2, 3, 4, 6 — max = 6.0)
//
// Algorithme :
//   1. Pour chaque bloc de 16 BF16 :
//      a. max_abs = max(|x_i|) sur le bloc
//      b. scale = max_abs / 6.0  (si 0 → scale = 1)
//      c. encode scale en UE4M3 byte
//      d. pour chaque x_i : x_q = round_to_fp4(x_i / scale)
//   2. Pack 2 FP4 par byte
//
// UE4M3 encoding (8-bit unsigned, exp 4 bits bias 7, mantissa 3 bits) :
//   value = 2^(E - 7) * (1 + M/8) si E != 0
//   value = 2^(-6) * M/8         si E == 0 (subnormal)
//   max value ≈ 240, ~1.0 == 0x70 (E=7, M=0)
extern "C" __global__ void quantize_bf16_to_nvfp4(
    const __nv_bfloat16* __restrict__ x,
    unsigned char* __restrict__ out_fp4,
    unsigned char* __restrict__ out_scale,
    int n
) {
    int blk = blockIdx.x;
    int block_off = blk * 16;
    if (block_off >= n) return;

    // 1. find max abs in 16 elements
    float max_abs = 0.0f;
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        if (block_off + i >= n) break;
        float v = fabsf((float)x[block_off + i]);
        if (v > max_abs) max_abs = v;
    }

    // 2. scale = max_abs / 6.0  (FP4 max = 6)
    float scale = max_abs / 6.0f;
    if (scale < 1e-12f) scale = 1.0f;

    // 3. Encode scale in UE4M3 — simplified : compute log2 + mantissa
    // float bits : sign(1) exp(8 bias 127) mantissa(23)
    // We want UE4M3 : exp(4 bias 7) mantissa(3)
    unsigned int sb = __float_as_uint(scale);
    int fexp = (int)((sb >> 23) & 0xff) - 127;     // unbiased exponent
    int fmant = (int)(sb >> 20) & 0x7;             // top 3 bits of mantissa
    int ue_exp = fexp + 7;                          // re-bias to UE4M3
    unsigned char scale_byte;
    if (ue_exp <= 0) {
        // subnormal or underflow → encode 0 (effectively scale=0, but we floored above)
        scale_byte = (unsigned char)(fmant);
    } else if (ue_exp >= 15) {
        scale_byte = 0xf0 | 0x7;  // saturate
    } else {
        scale_byte = (unsigned char)((ue_exp << 3) | fmant);
    }
    out_scale[blk] = scale_byte;

    // 4. quantize 16 BF16 → 8 packed bytes (FP4 each = 4 bits)
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        if (block_off + 2*i >= n) break;
        float a = (float)x[block_off + 2*i] / scale;
        float b = (block_off + 2*i + 1 < n) ? (float)x[block_off + 2*i + 1] / scale : 0.0f;

        // round-to-nearest FP4 E2M1 (signe 1, exp 2, mant 1)
        // Code des valeurs FP4 (4 bits) :
        //   0=+0, 1=+0.5, 2=+1, 3=+1.5, 4=+2, 5=+3, 6=+4, 7=+6
        //   8=-0, 9=-0.5, a=-1, b=-1.5, c=-2, d=-3, e=-4, f=-6
        auto encode = [](float x_norm) -> unsigned int {
            unsigned int sign = (x_norm < 0.0f) ? 8u : 0u;
            float ax = fabsf(x_norm);
            // Map ax ∈ [0, 6+ε] to one of [0, 0.5, 1, 1.5, 2, 3, 4, 6]
            unsigned int code;
            if (ax < 0.25f) code = 0;
            else if (ax < 0.75f) code = 1;
            else if (ax < 1.25f) code = 2;
            else if (ax < 1.75f) code = 3;
            else if (ax < 2.5f)  code = 4;
            else if (ax < 3.5f)  code = 5;
            else if (ax < 5.0f)  code = 6;
            else code = 7;
            return sign | code;
        };

        unsigned int qa = encode(a);
        unsigned int qb = encode(b);
        out_fp4[blk * 8 + i] = (unsigned char)((qa & 0xf) | ((qb & 0xf) << 4));
    }
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
const GQA_DECODE_ONLINE_BF16_SRC: &str = r#"
#include <cuda_bf16.h>

// Online-softmax GQA decode (FlashAttention-style, single-pass).
//
// Au lieu du naive 3-pass (compute scores, max+exp+sum, then weighted sum),
// on fait un seul pass à travers le KV cache :
//   m = -inf, l = 0, o = 0
//   for t in 0..kv_len:
//     s_t = q · k[t] * scale
//     new_m = max(m, s_t)
//     correction = exp(m - new_m)
//     o = o * correction + exp(s_t - new_m) * v[t]
//     l = l * correction + exp(s_t - new_m)
//     m = new_m
//   return o / l
//
// Avantages : 1 lecture KV cache, pas de buffer scores, plus cache-friendly.
// Pour single-token decode c'est très efficace si threadDim_x = head_dim
// (chaque thread accumule un élément du output vector).
extern "C" __global__ void gqa_decode_online_bf16(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k_cache,
    const __nv_bfloat16* __restrict__ v_cache,
    __nv_bfloat16* __restrict__ out,
    int n_heads,
    int n_kv,
    int kv_len,
    int head_dim,
    int max_seq,
    float scale
) {
    int h = blockIdx.x;
    if (h >= n_heads) return;
    int kv_h = h * n_kv / n_heads;

    int tid = threadIdx.x;

    // Each thread handles one element of head_dim
    if (tid >= head_dim) return;

    float q_i = (float)q[h * head_dim + tid];

    // Online softmax state per thread
    float m = -1e30f;
    float l = 0.0f;
    float o = 0.0f;

    // Shared mem for cross-thread Q·K dot product reduction
    extern __shared__ float sdata[];

    for (int t = 0; t < kv_len; ++t) {
        // Compute s_t = Q[h] · K[kv_h, t] * scale
        // Each thread contributes q_i * k_i, then we reduce across threads
        float k_i = (float)k_cache[(kv_h * max_seq + t) * head_dim + tid];
        float partial = q_i * k_i;
        sdata[tid] = partial;
        __syncthreads();
        // Tree reduction
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s && tid + s < head_dim) {
                sdata[tid] += sdata[tid + s];
            }
            __syncthreads();
        }
        float s_t = sdata[0] * scale;
        __syncthreads();

        // Online softmax update
        float new_m = fmaxf(m, s_t);
        float correction = expf(m - new_m);
        float p = expf(s_t - new_m);
        // Each thread updates its element of o
        float v_i = (float)v_cache[(kv_h * max_seq + t) * head_dim + tid];
        o = o * correction + p * v_i;
        l = l * correction + p;
        m = new_m;
    }

    // Final : out[h, tid] = o / l
    out[h * head_dim + tid] = (__nv_bfloat16)(o / fmaxf(l, 1e-12f));
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
    quantize_nvfp4: std::sync::OnceLock<(Arc<CudaModule>, CudaFunction)>,
    gqa_decode_online: std::sync::OnceLock<(Arc<CudaModule>, CudaFunction)>,
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
            quantize_nvfp4: std::sync::OnceLock::new(),
            gqa_decode_online: std::sync::OnceLock::new(),
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

    /// GQA decode avec online softmax (FlashAttention-style, 1-pass au
    /// lieu de 3-pass naive). Plus efficace pour grands kv_len.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn gqa_decode_online_bf16(
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
            &self.gqa_decode_online,
            GQA_DECODE_ONLINE_BF16_SRC,
            "gqa_decode_online_bf16",
        )?;
        let scale: f32 = 1.0 / (head_dim as f32).sqrt();
        // Threads = head_dim (each thread handles 1 dim)
        let block_dim = head_dim as u32;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (n_heads as u32, 1, 1),
            block_dim: (block_dim, 1, 1),
            shared_mem_bytes: block_dim * 4, // for tree reduction
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
            location: "gqa_decode_online_bf16::launch",
        })?;
        Ok(())
    }

    /// Quantize BF16 → NVFP4 (E2M1) avec scales VEC16_UE4M3.
    /// `out_fp4` doit être de taille n/2 bytes, `out_scale` n/16 bytes.
    pub unsafe fn quantize_bf16_to_nvfp4(
        &self,
        stream: &Arc<CudaStream>,
        x: u64,
        out_fp4: u64,
        out_scale: u64,
        n: i32,
    ) -> Result<(), CudaError> {
        let (_module, func) = self.compile_or_get(
            &self.quantize_nvfp4,
            QUANTIZE_BF16_TO_NVFP4_SRC,
            "quantize_bf16_to_nvfp4",
        )?;
        // 1 block par bloc de 16 elements
        let n_blocks = (n + 15) / 16;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (n_blocks as u32, 1, 1),
            block_dim: (1, 1, 1), // sequential per block
            shared_mem_bytes: 0,
        };
        let mut launcher = stream.launch_builder(&func);
        launcher.arg(&x).arg(&out_fp4).arg(&out_scale).arg(&n);
        launcher.launch(cfg).map_err(|e| CudaError::Driver {
            code: format!("{e:?}").len() as i32,
            location: "quantize_bf16_to_nvfp4::launch",
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
// Parity tests — kernel CUDA vs reference CPU
// ─────────────────────────────────────────────────────────────────────────

#[cfg(all(test, feature = "cuda"))]
mod parity_tests {
    use super::*;
    use cudarc::driver::{CudaContext, DevicePtr, DevicePtrMut};

    /// CPU RMSNorm reference (matches `rms_norm_inplace` in rustorch-llm).
    fn cpu_rms_norm(x: &mut [f32], gamma: &[f32], eps: f32) {
        let n = x.len();
        let sum_sq: f32 = x.iter().map(|v| v * v).sum();
        let inv_rms = 1.0 / (sum_sq / n as f32 + eps).sqrt();
        for (xi, gi) in x.iter_mut().zip(gamma.iter()) {
            *xi = *xi * inv_rms * *gi;
        }
    }

    /// CPU RoPE half-split reference (matches `apply_inplace_half_split`).
    /// inv_freq[k] = base^(-2k/D), pos = position offset.
    #[allow(clippy::too_many_arguments)]
    fn cpu_rope_half_split(
        x: &mut [f32],
        inv_freq: &[f32],
        pos: usize,
        n_heads: usize,
        head_dim: usize,
    ) {
        let half = head_dim / 2;
        for h in 0..n_heads {
            let row = h * head_dim;
            for k in 0..half {
                let theta = inv_freq[k] * pos as f32;
                let (sin_k, cos_k) = theta.sin_cos();
                let a = x[row + k];
                let b = x[row + k + half];
                x[row + k] = a * cos_k - b * sin_k;
                x[row + k + half] = a * sin_k + b * cos_k;
            }
        }
    }

    /// T241.6b regression guard — CUDA `rms_norm_bf16` matches CPU rms_norm.
    /// Catches the kind of bug we'd suspect : wrong axis sum, missing eps,
    /// gamma misapplied.
    #[test]
    fn rms_norm_bf16_matches_cpu() {
        let n = 128usize;
        let eps = 1e-6f32;
        // Hand-picked input + gamma — odd values to detect bugs in iteration.
        let x_f32: Vec<f32> = (0..n).map(|i| ((i as f32 * 0.13).sin()) + 0.1).collect();
        let gamma_f32: Vec<f32> = (0..n).map(|i| 1.0 + 0.01 * (i as f32)).collect();

        let mut x_cpu = x_f32.clone();
        cpu_rms_norm(&mut x_cpu, &gamma_f32, eps);

        let ctx = CudaContext::new(0).expect("CudaContext");
        let stream = ctx.default_stream();
        let kernels = LlmKernels::new(ctx);
        let x_bf: Vec<half::bf16> = x_f32.iter().copied().map(half::bf16::from_f32).collect();
        let gamma_bf: Vec<half::bf16> = gamma_f32
            .iter()
            .copied()
            .map(half::bf16::from_f32)
            .collect();
        let mut x_dev = stream.memcpy_stod(&x_bf).expect("upload");
        let gamma_dev = stream.memcpy_stod(&gamma_bf).expect("upload");
        unsafe {
            let (x_p, _r1) = x_dev.device_ptr_mut(&stream);
            let (g_p, _r2) = gamma_dev.device_ptr(&stream);
            kernels
                .rms_norm_bf16(&stream, x_p, g_p, eps, n as i32, 1)
                .expect("rms_norm");
        }
        let x_host: Vec<half::bf16> = stream.memcpy_dtov(&x_dev).expect("dtov");
        let x_gpu: Vec<f32> = x_host.into_iter().map(|x| x.to_f32()).collect();

        for i in 0..n {
            let diff = (x_cpu[i] - x_gpu[i]).abs();
            let tol = x_cpu[i].abs() * 5e-2 + 1e-2; // BF16 ~= 1% relative
            assert!(
                diff <= tol,
                "rms_norm[{i}] cpu={} gpu={} diff={}",
                x_cpu[i],
                x_gpu[i],
                diff
            );
        }
    }

    /// T241.6b regression guard — CUDA `rope_half_split_bf16` matches CPU.
    /// Catches the bug we just fixed : kernel was using interleaved indexing
    /// `(2k, 2k+1)` while the CPU uses half-split `(k, k+half)`.
    #[test]
    fn rope_half_split_bf16_matches_cpu() {
        let head_dim = 32usize;
        let n_heads = 4usize;
        let pos = 5usize;
        let base = 10_000.0f32;
        let half = head_dim / 2;
        let inv_freq: Vec<f32> = (0..half)
            .map(|k| 1.0 / base.powf((2 * k) as f32 / head_dim as f32))
            .collect();

        // Asymmetric input — so any axis-pair bug shows up.
        let x_f32: Vec<f32> = (0..(n_heads * head_dim))
            .map(|i| ((i as f32 * 0.07).cos()) * 0.5 + 0.1)
            .collect();

        let mut x_cpu = x_f32.clone();
        cpu_rope_half_split(&mut x_cpu, &inv_freq, pos, n_heads, head_dim);

        let ctx = CudaContext::new(0).expect("CudaContext");
        let stream = ctx.default_stream();
        let kernels = LlmKernels::new(ctx);
        let x_bf: Vec<half::bf16> = x_f32.iter().copied().map(half::bf16::from_f32).collect();
        let mut x_dev = stream.memcpy_stod(&x_bf).expect("upload");
        let inv_freq_dev = stream.memcpy_stod(&inv_freq).expect("upload");
        unsafe {
            let (x_p, _r1) = x_dev.device_ptr_mut(&stream);
            let (inv_p, _r2) = inv_freq_dev.device_ptr(&stream);
            kernels
                .rope_half_split_bf16(
                    &stream,
                    x_p,
                    inv_p,
                    pos as i32,
                    n_heads as i32,
                    head_dim as i32,
                )
                .expect("rope");
        }
        let x_host: Vec<half::bf16> = stream.memcpy_dtov(&x_dev).expect("dtov");
        let x_gpu: Vec<f32> = x_host.into_iter().map(|x| x.to_f32()).collect();

        for i in 0..(n_heads * head_dim) {
            let diff = (x_cpu[i] - x_gpu[i]).abs();
            let tol = x_cpu[i].abs() * 1e-1 + 1e-2;
            assert!(
                diff <= tol,
                "rope[{i}] (h={}, j={}) cpu={} gpu={} diff={}",
                i / head_dim,
                i % head_dim,
                x_cpu[i],
                x_gpu[i],
                diff
            );
        }
    }

    /// T241.6b — embedding lookup CUDA vs CPU. Simple gather, but the test
    /// catches off-by-one strides and wrong dtype.
    #[test]
    fn embedding_lookup_bf16_matches_cpu() {
        let vocab = 16usize;
        let hidden = 8usize;
        let ids: Vec<u32> = vec![3, 7, 0, 15];
        let table_f32: Vec<f32> = (0..(vocab * hidden)).map(|i| i as f32 * 0.01).collect();

        let mut out_cpu = vec![0.0f32; ids.len() * hidden];
        for (s, &id) in ids.iter().enumerate() {
            let off = (id as usize) * hidden;
            out_cpu[s * hidden..(s + 1) * hidden].copy_from_slice(&table_f32[off..off + hidden]);
        }

        let ctx = CudaContext::new(0).expect("CudaContext");
        let stream = ctx.default_stream();
        let kernels = LlmKernels::new(ctx);
        let table_bf: Vec<half::bf16> = table_f32
            .iter()
            .copied()
            .map(half::bf16::from_f32)
            .collect();
        let table_dev = stream.memcpy_stod(&table_bf).expect("upload");
        let ids_dev = stream.memcpy_stod(&ids).expect("upload");
        let mut out_dev = stream
            .alloc_zeros::<half::bf16>(ids.len() * hidden)
            .expect("alloc out");
        unsafe {
            let (t_p, _r1) = table_dev.device_ptr(&stream);
            let (i_p, _r2) = ids_dev.device_ptr(&stream);
            let (o_p, _r3) = out_dev.device_ptr_mut(&stream);
            kernels
                .embedding_lookup_bf16(&stream, t_p, i_p, o_p, ids.len() as i32, hidden as i32)
                .expect("embedding");
        }
        let out_host: Vec<half::bf16> = stream.memcpy_dtov(&out_dev).expect("dtov");
        let out_gpu: Vec<f32> = out_host.into_iter().map(|x| x.to_f32()).collect();

        for i in 0..(ids.len() * hidden) {
            let diff = (out_cpu[i] - out_gpu[i]).abs();
            assert!(
                diff < 1e-2,
                "embedding[{i}] cpu={} gpu={}",
                out_cpu[i],
                out_gpu[i]
            );
        }
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
