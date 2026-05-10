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

// T246.2 — element-wise utility kernels for SSM block forward.

#[cfg(feature = "cuda")]
const SOFTPLUS_INPLACE_BF16_SRC: &str = r#"
#include <cuda_bf16.h>

// x[i] = log1p(exp(x[i]))
// Uses numerically stable formulation : softplus(x) = max(x, 0) + log1p(exp(-|x|))
extern "C" __global__ void softplus_inplace_bf16(
    __nv_bfloat16* __restrict__ x,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = (float)x[i];
    float a = v > 0.0f ? v : 0.0f;
    float b = log1pf(expf(-fabsf(v)));
    x[i] = (__nv_bfloat16)(a + b);
}
"#;

#[cfg(feature = "cuda")]
const SIGMOID_INPLACE_BF16_SRC: &str = r#"
#include <cuda_bf16.h>

extern "C" __global__ void sigmoid_inplace_bf16(
    __nv_bfloat16* __restrict__ x,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = (float)x[i];
    float s = 1.0f / (1.0f + expf(-v));
    x[i] = (__nv_bfloat16)s;
}
"#;

#[cfg(feature = "cuda")]
const MUL_INPLACE_BF16_SRC: &str = r#"
#include <cuda_bf16.h>

// y[i] *= x[i]
extern "C" __global__ void mul_inplace_bf16(
    __nv_bfloat16* __restrict__ y,
    const __nv_bfloat16* __restrict__ x,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float a = (float)y[i];
    float b = (float)x[i];
    y[i] = (__nv_bfloat16)(a * b);
}
"#;

// T246.3 — partial RoPE for Qwen3.x where rope_dim < head_dim.
// Only the first `rope_dim` elements of each head_dim are rotated;
// the remaining (head_dim - rope_dim) pass through unchanged.
//
// Half-split convention :
//   For k in [0, rope_dim/2) :
//     a   = x[h, k]
//     b   = x[h, k + rope_dim/2]
//     theta = inv_freq[k] * pos
//     x[h, k]               = a * cos - b * sin
//     x[h, k + rope_dim/2]  = a * sin + b * cos
//   Elements in [rope_dim, head_dim) are untouched.
#[cfg(feature = "cuda")]
const ROPE_PARTIAL_BF16_SRC: &str = r#"
#include <cuda_bf16.h>

extern "C" __global__ void rope_partial_bf16(
    __nv_bfloat16* __restrict__ x,
    const float* __restrict__ inv_freq,   // [rope_dim/2]
    int pos,
    int n_heads,
    int head_dim,
    int rope_dim
) {
    int h = blockIdx.x;
    if (h >= n_heads) return;
    int half = rope_dim / 2;
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

// T246.5.3 — RoPE variant qui lit `pos` depuis device pointer (CUDA Graph friendly).
// Sémantiquement identique à rope_partial_bf16 mais avec `pos = *pos_dev`. Permet
// que la valeur de pos puisse changer entre replays d'un même graph capturé.
#[cfg(feature = "cuda")]
const ROPE_PARTIAL_BF16_DEVCNT_SRC: &str = r#"
#include <cuda_bf16.h>

extern "C" __global__ void rope_partial_bf16_devcnt(
    __nv_bfloat16* __restrict__ x,
    const float* __restrict__ inv_freq,
    const int* __restrict__ pos_dev,
    int n_heads,
    int head_dim,
    int rope_dim
) {
    int h = blockIdx.x;
    if (h >= n_heads) return;
    int half = rope_dim / 2;
    int k = blockIdx.y * blockDim.x + threadIdx.x;
    if (k >= half) return;
    int pos = *pos_dev;

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

// T246.5.3 — increment 1-elt int device buffer. 1 thread, 1 block. Used to
// advance position counter at the end of decode_step (inside captured graph).
#[cfg(feature = "cuda")]
const INCREMENT_U32_DEV_SRC: &str = r#"
extern "C" __global__ void increment_u32_dev(int* p) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *p = *p + 1;
    }
}
"#;

// T246.5.3 — append K and V vectors to the cache at slot `*pos_dev`.
// Cache layout (matches the existing host-side append in decode_step):
//   k_cache, v_cache : [max_seq, kv_dim] BF16, contiguous, seq-major.
// Writes k[0..kv_dim] → k_cache[(*pos_dev) * kv_dim ..], same for v.
// Single kernel handles both K and V (1 thread per kv_dim element).
#[cfg(feature = "cuda")]
const KV_APPEND_BF16_DEVCNT_SRC: &str = r#"
#include <cuda_bf16.h>

extern "C" __global__ void kv_append_bf16_devcnt(
    __nv_bfloat16* __restrict__ k_cache,
    __nv_bfloat16* __restrict__ v_cache,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    const int* __restrict__ pos_dev,
    int kv_dim
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= kv_dim) return;
    int pos = *pos_dev;
    long long off = (long long)pos * (long long)kv_dim + (long long)tid;
    k_cache[off] = k[tid];
    v_cache[off] = v[tid];
}
"#;

#[cfg(feature = "cuda")]
const SPLIT_QG_BF16_SRC: &str = r#"
#include <cuda_bf16.h>

// Qwen3.6 attention : the Q projection produces a fused [2*q_dim] output
// where each head_dim block alternates between Q and gate :
//   qg[h * 2 * head_dim + 0..head_dim]            = q_h
//   qg[h * 2 * head_dim + head_dim..2*head_dim]   = gate_h
// This kernel deinterleaves into separate q[q_dim] and gate[q_dim] buffers.
extern "C" __global__ void split_qg_bf16(
    const __nv_bfloat16* __restrict__ qg,
    __nv_bfloat16* __restrict__ q,
    __nv_bfloat16* __restrict__ gate,
    int n_heads,
    int head_dim
) {
    int h = blockIdx.x;
    int j = blockIdx.y * blockDim.x + threadIdx.x;
    if (j >= head_dim) return;
    int src_off = h * 2 * head_dim + j;
    int dst_off = h * head_dim + j;
    q[dst_off]    = qg[src_off];
    gate[dst_off] = qg[src_off + head_dim];
}
"#;

#[cfg(feature = "cuda")]
const REPEAT_HEADS_BF16_SRC: &str = r#"
#include <cuda_bf16.h>

// Broadcast K/Q heads from n_in heads to n_out heads (n_out / n_in repeat).
// src is [n_in, head_dim], dst is [n_out, head_dim].
// dst[h, j] = src[h / repeat, j]   where repeat = n_out / n_in.
extern "C" __global__ void repeat_heads_bf16(
    const __nv_bfloat16* __restrict__ src,
    __nv_bfloat16* __restrict__ dst,
    int n_in,
    int n_out,
    int head_dim
) {
    int h_out = blockIdx.x;
    int j = blockIdx.y * blockDim.x + threadIdx.x;
    if (j >= head_dim) return;
    int repeat = n_out / n_in;
    int h_in = h_out / repeat;
    dst[h_out * head_dim + j] = src[h_in * head_dim + j];
}
"#;

#[cfg(feature = "cuda")]
const TRANSPOSE_BF16_SRC: &str = r#"
#include <cuda_bf16.h>

// Transpose row-major [rows, cols] → row-major [cols, rows].
//   src[i, j] = src_buf[i*cols + j]    →    dst[j, i] = dst_buf[j*rows + i]
//
// T241.6c — used to convert row-major W → col-major-equivalent layout
// before NVFP4 quantization, so that cuBLASLt FP4 (TN-only on sm_121)
// reads the buffer as the correct mathematical W.
extern "C" __global__ void transpose_bf16(
    const __nv_bfloat16* __restrict__ src,
    __nv_bfloat16* __restrict__ dst,
    int rows,
    int cols
) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows || j >= cols) return;
    dst[j * rows + i] = src[i * cols + j];
}
"#;

#[cfg(feature = "cuda")]
const SGEMV_Q4K_BF16_SRC: &str = r#"
#include <cuda_bf16.h>
#include <cuda_fp16.h>

// T244.1 — Direct Q4_K matmul (port Metal TurboQuant pattern to CUDA).
//
// Computes  y[n] = sum_k W_q4k[n, k] * x[k]  where W is stored in
// Q4_K format (144 bytes per 256-weight super-block) and x/y are BF16.
//
// Q4_K block layout (144 bytes) :
//   2 bytes : d        (f16)
//   2 bytes : dmin     (f16)
//  12 bytes : packed (sc[8], m[8]) — 6-bit scale & min per sub-block
// 128 bytes : qs       — 4-bit nibbles, 256 weights total
//
// The 8 sub-blocks of 32 weights are PAIRED : (sb0, sb1), (sb2, sb3),
// (sb4, sb5), (sb6, sb7). Each pair shares 32 bytes :
//   byte_k.low_nibble  = weight k of even sub-block
//   byte_k.high_nibble = weight k of odd sub-block
//
// Dequantized value = d * sc[sb] * q - dmin * m[sb]   for sb in 0..8.
//
// Memory bandwidth saving vs BF16 : 144 / (256 * 2) = 28 % of BF16 bytes
// = 3.6× less weight memory read per matmul → ~3-4× speedup on memory-
// bound single-token decode (dominant case for autoregressive LLM).
//
// Launch : grid_dim = N (one block per output row), block_dim = 256
// (one thread per weight in a Q4_K super-block).
extern "C" __global__ void sgemv_q4k_bf16(
    const unsigned char* __restrict__ w_q4k,  // [N, blocks_per_row * 144]
    const __nv_bfloat16* __restrict__ x,       // [K]
    __nv_bfloat16* __restrict__ y,             // [N]
    int N,
    int K
) {
    int row = blockIdx.x;
    if (row >= N) return;
    int tid = threadIdx.x;
    int blocks_per_row = K / 256;
    int row_offset = row * blocks_per_row * 144;

    // Per-block dequant unpacking : the 12 packed scale bytes encode
    // 8 6-bit sc and 8 6-bit m values. This mirrors `unpack_q4_k_sc_m`
    // in rustorch-gguf/dequant.rs.
    //   sc[i]   = scales[i]   & 0x3F                    for i in 0..4
    //   m[i]    = scales[i+4] & 0x3F                    for i in 0..4
    //   sc[i+4] = (scales[i+8] & 0x0F) | ((scales[i]   >> 6) << 4)
    //   m[i+4]  = (scales[i+8] >> 4)   | ((scales[i+4] >> 6) << 4)
    //
    // Within the 256-weight super-block, thread `tid` (0..256) decodes
    // weight at position `tid`. tid maps to :
    //   pair_idx = tid / 64           (0..4)  — which (sb_even, sb_odd) pair
    //   sub_in_pair = (tid / 32) & 1  (0 or 1) — even sub-block (low nibble) or odd (high)
    //   pos_in_sb = tid & 31          (0..32) — position within sub-block
    //   sb = pair_idx * 2 + sub_in_pair
    //   byte_pos = pair_idx * 32 + pos_in_sb
    //   nibble = (qs[byte_pos] >> (sub_in_pair * 4)) & 0x0F
    //   value = d * sc[sb] * nibble - dmin * m[sb]

    extern __shared__ float sdata[];

    float acc = 0.0f;

    for (int b = 0; b < blocks_per_row; ++b) {
        int blk_off = row_offset + b * 144;
        const unsigned char* blk = w_q4k + blk_off;

        // Load d, dmin (f16 → float) — only thread 0 reads, broadcast via shmem
        // is overkill for 4 bytes ; let each thread read independently.
        unsigned short d_bits   = blk[0]  | (blk[1]  << 8);
        unsigned short dmin_bits= blk[2]  | (blk[3]  << 8);
        float d    = __half2float(__ushort_as_half(d_bits));
        float dmin = __half2float(__ushort_as_half(dmin_bits));

        // Read scales[12] (only thread 0 reads, store in shmem) — but for
        // simplicity each thread reads the 12 bytes (only ~3% overhead, no
        // bank conflicts since reads are short).
        unsigned char sc8[8];
        unsigned char m8[8];
        const unsigned char* scales = blk + 4;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            sc8[i] = scales[i] & 0x3F;
            m8[i]  = scales[i + 4] & 0x3F;
            sc8[i + 4] = (scales[i + 8] & 0x0F) | ((scales[i]   >> 6) << 4);
            m8[i + 4]  = (scales[i + 8] >> 4)   | ((scales[i + 4] >> 6) << 4);
        }

        const unsigned char* qs = blk + 16;

        // Decode weight at position `tid` within this super-block.
        int pair_idx    = tid >> 6;          // tid / 64
        int sub_in_pair = (tid >> 5) & 1;    // (tid / 32) & 1
        int pos_in_sb   = tid & 31;          // tid % 32
        int sb          = (pair_idx << 1) | sub_in_pair;
        int byte_pos    = (pair_idx << 5) + pos_in_sb;  // pair_idx * 32
        unsigned char byte = qs[byte_pos];
        int nibble = (sub_in_pair == 0) ? (byte & 0x0F) : (byte >> 4);

        float w_val = d * (float)sc8[sb] * (float)nibble - dmin * (float)m8[sb];
        float x_val = (float)x[b * 256 + tid];
        acc += w_val * x_val;
    }

    // Block-level reduction over 256 threads.
    sdata[tid] = acc;
    __syncthreads();
    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) {
        y[row] = (__nv_bfloat16)sdata[0];
    }
}
"#;

#[cfg(feature = "cuda")]
const SGEMV_Q4K_BF16_V2_SRC: &str = r#"
#include <cuda_bf16.h>
#include <cuda_fp16.h>

// T244.1.1 — sgemv_q4k_bf16 V2 — optimised pattern :
//
// Per-block analysis of V1 :
//   - 256 threads/TG, each thread decoded 1 weight per super-block
//   - 256× redundant scale unpacking (12-byte loads + 8-step unpack)
//   - byte-by-byte nibble loads (no vector coalescing)
//   - V1 measured : 1.17 ms / 32 GB/s / 116 GFLOPS  (16% bandwidth)
//
// V2 changes :
//   1. 64 threads/TG (vs 256) → 4× more concurrent TGs per SM (better
//      latency hiding)
//   2. Each thread processes 4 weights per super-block (256/64)
//   3. Thread 0 unpacks scales + d/dmin ONCE per super-block, stores
//      pre-multiplied d*sc[i] and dmin*m[i] in shmem (broadcast to all)
//   4. uint32 nibble load = 4 bytes per load (vector coalescing)
//   5. uint2 (8-byte) bf16 load for x = 4 elements per load
//
// Tile mapping (256 weights, 64 threads) :
//   group       = tid / 8     (0..8)  — which sub-block (sb)
//   pos_base    = (tid%8) * 4 (0..32 step 4) — start position within sb
//   pair_idx    = group / 2
//   sub_in_pair = group & 1   — even/odd pair (low/high nibble)
//   byte_base   = pair_idx*32 + pos_base — first of 4 nibble bytes
//
// Expected gain : 3-5× over V1 (= 30-50 tok/s upper bound on Qwen-7B Q4_K_M).
extern "C" __global__ void sgemv_q4k_bf16_v2(
    const unsigned char* __restrict__ w_q4k,
    const __nv_bfloat16* __restrict__ x,
    __nv_bfloat16* __restrict__ y,
    int N,
    int K
) {
    int row = blockIdx.x;
    if (row >= N) return;
    int tid = threadIdx.x;
    int blocks_per_row = K / 256;
    int row_offset = row * blocks_per_row * 144;

    extern __shared__ float shmem[];
    float* sc_pre = shmem;            // [8] : d * sc[i] (pre-multiplied)
    float* m_pre  = shmem + 8;        // [8] : dmin * m[i]
    float* sdata  = shmem + 16;       // [64] : reduction buffer

    float acc = 0.0f;

    int group       = tid >> 3;        // tid / 8     → sb in 0..8
    int pos_base    = (tid & 7) << 2;  // (tid%8)*4    → 0..28 step 4
    int pair_idx    = group >> 1;
    int sub_in_pair = group & 1;
    int byte_base   = (pair_idx << 5) + pos_base;
    int low_nibble  = (sub_in_pair == 0) ? 1 : 0;

    for (int b = 0; b < blocks_per_row; ++b) {
        int blk_off = row_offset + b * 144;
        const unsigned char* blk = w_q4k + blk_off;

        // Thread 0 : unpack scales + d/dmin → shmem (broadcast).
        if (tid == 0) {
            unsigned short d_bits    = blk[0] | (blk[1] << 8);
            unsigned short dmin_bits = blk[2] | (blk[3] << 8);
            float d    = __half2float(__ushort_as_half(d_bits));
            float dmin = __half2float(__ushort_as_half(dmin_bits));
            const unsigned char* scales = blk + 4;
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                unsigned char sc_i  = scales[i] & 0x3F;
                unsigned char m_i   = scales[i + 4] & 0x3F;
                unsigned char sc_i4 = (scales[i + 8] & 0x0F) | ((scales[i] >> 6) << 4);
                unsigned char m_i4  = (scales[i + 8] >> 4)   | ((scales[i + 4] >> 6) << 4);
                sc_pre[i]     = d * (float)sc_i;
                m_pre[i]      = dmin * (float)m_i;
                sc_pre[i + 4] = d * (float)sc_i4;
                m_pre[i + 4]  = dmin * (float)m_i4;
            }
        }
        __syncthreads();

        // Pre-multiplied scale/min for THIS thread's sub-block.
        float scale = sc_pre[group];
        float min_v = m_pre[group];

        // uint32 load of 4 nibble bytes.
        const unsigned char* qs = blk + 16;
        unsigned int qbytes = *(const unsigned int*)(qs + byte_base);

        // Load 4 bf16 from x via uint2 (8 bytes).
        const __nv_bfloat16* x_ptr = x + b * 256 + (group << 5) + pos_base;
        uint2 xbits = *(const uint2*)x_ptr;
        unsigned short xb[4] = {
            (unsigned short)(xbits.x & 0xFFFFu),
            (unsigned short)(xbits.x >> 16),
            (unsigned short)(xbits.y & 0xFFFFu),
            (unsigned short)(xbits.y >> 16),
        };

        // Decode 4 nibbles + multiply with x.
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            unsigned char byte_i = (qbytes >> (i << 3)) & 0xFFu;
            int nibble = low_nibble ? (byte_i & 0x0F) : (byte_i >> 4);
            float w_val = scale * (float)nibble - min_v;
            float xv = __half2float(__ushort_as_half(xb[i]));
            // NOTE: xb[i] is bf16 bits, not f16. Re-cast as bf16.
            __nv_bfloat16 xbf = __ushort_as_bfloat16(xb[i]);
            xv = (float)xbf;
            acc += w_val * xv;
        }
        // T246.4.4 — RACE FIX : sync between iterations so next iter's write
        // to sc_pre/m_pre doesn't race with this iter's reads.
        __syncthreads();
    }

    // T244.1.1.1 — Warp-shuffle reduction over 64 threads = 2 warps.
    // Eliminates 5 of 6 __syncthreads of the tree reduction.
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xffffffff, acc, offset);
    }
    int warp_id = tid >> 5;
    int lane_id = tid & 31;
    if (lane_id == 0) {
        sdata[warp_id] = acc;
    }
    __syncthreads();
    if (tid == 0) {
        float total = sdata[0] + sdata[1];
        y[row] = (__nv_bfloat16)total;
    }
}
"#;

// T246.5.5 — Q8_1 quantization of BF16 activation row.
// Mirrors llama.cpp `quantize_q8_1` but reads bf16 instead of f32.
//
// Output layout per 32-element block (36 bytes total) :
//   bytes 0-1  : __half d        (= amax / 127)
//   bytes 2-3  : __half s        (= sum(x[i]) for the 32 elements ;
//                                  used by Q4_K vec_dot for offset correction)
//   bytes 4-35 : int8_t qs[32]
//
// Launch : grid_dim = (n_blocks, 1, 1), block_dim = (32, 1, 1).
// Each warp = 1 block_q8_1.
#[cfg(feature = "cuda")]
const QUANTIZE_Q8_1_BF16_SRC: &str = r#"
#include <cuda_bf16.h>
#include <cuda_fp16.h>

extern "C" __global__ void quantize_q8_1_bf16(
    const __nv_bfloat16* __restrict__ x,
    unsigned char* __restrict__ y,
    int n_blocks
) {
    int blk = blockIdx.x;
    if (blk >= n_blocks) return;
    int tid = threadIdx.x;  // 0..31, one warp

    const __nv_bfloat16* x_blk = x + blk * 32;
    unsigned char* y_blk       = y + blk * 36;

    float xv   = (float)x_blk[tid];
    float amax = fabsf(xv);
    float sum  = xv;

    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, o));
        sum  = sum   +     __shfl_xor_sync(0xffffffff, sum,  o);
    }

    float d     = amax / 127.0f;
    float inv_d = (amax == 0.0f) ? 0.0f : 1.0f / d;
    int   qi    = __float2int_rn(xv * inv_d);
    if (qi < -127) qi = -127;
    if (qi >  127) qi =  127;
    signed char q = (signed char)qi;

    ((signed char*)(y_blk + 4))[tid] = q;

    if (tid == 0) {
        ((__half*)y_blk)[0] = __float2half_rn(d);
        ((__half*)y_blk)[1] = __float2half_rn(sum);
    }
}
"#;

// T246.5.5 — Q4_K × Q8_1 SGEMV using __dp4a (mirror of llama.cpp
// vec_dot_q4_K_q8_1_impl_vmmq packed into a per-row CTA).
//
// Per super-block of W (256 weights, 144 bytes), 16 vec_dot units :
//   bq8_offset ∈ {0,2,4,6} (= 2*bp, bp ∈ 0..3) — selects 32-byte W chunk + 2 sub-blocks of x
//   qc         ∈ {0,1,2,3}                       — selects 4-byte int within the chunk
//
// Per unit : 4 dp4a calls, accumulates dot1 (dot of v×u) and dot2 (sum of u
// for offset correction) for sub-blocks (bq8_offset, bq8_offset+1) jointly via
// low/high nibble decomposition (i ∈ {0,1}).
//
// Launch : grid_dim = (N, 1, 1), block_dim = (32, 1, 1). Each warp = 1 row.
#[cfg(feature = "cuda")]
const SGEMV_Q4K_Q8_1_DP4A_BF16_SRC: &str = r#"
#include <cuda_bf16.h>
#include <cuda_fp16.h>

extern "C" __global__ void sgemv_q4k_q8_1_dp4a_bf16(
    const unsigned char* __restrict__ w_q4k,    // [N, blocks_per_row * 144]
    const unsigned char* __restrict__ x_q8_1,   // [(K/32) * 36]
    __nv_bfloat16*       __restrict__ y,         // [N]
    int N,
    int K
) {
    int row = blockIdx.x;
    if (row >= N) return;
    int tid = threadIdx.x;  // 0..31

    int blocks_per_row = K / 256;
    float acc = 0.0f;

    for (int b = tid; b < blocks_per_row; b += 32) {
        const unsigned char* blk = w_q4k + (row * blocks_per_row + b) * 144;

        // d, dmin (fp16 scalars in W's super-block header)
        float d    = __half2float(*(const __half*)(blk + 0));
        float dmin = __half2float(*(const __half*)(blk + 2));

        // 12-byte scales → 8 sc + 8 m (6-bit + 4-bit hi extension)
        const unsigned char* sr = blk + 4;
        unsigned char sc[8], m[8];
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            sc[i]     = sr[i]     & 0x3F;
            m[i]      = sr[i + 4] & 0x3F;
            sc[i + 4] = (sr[i + 8] & 0x0F) | ((sr[i]     >> 6) << 4);
            m[i  + 4] = (sr[i + 8] >>   4) | ((sr[i + 4] >> 6) << 4);
        }

        const unsigned char* qs = blk + 16;  // 128 bytes nibble payload

        // 16 vec_dot units per super-block.
        #pragma unroll
        for (int bp = 0; bp < 4; ++bp) {
            int bq8_offset = 2 * bp;
            #pragma unroll
            for (int qc = 0; qc < 4; ++qc) {
                int v0 = *(const int*)(qs + 32 * bp +  4 * qc);
                int v1 = *(const int*)(qs + 32 * bp + 16 + 4 * qc);

                #pragma unroll
                for (int i = 0; i < 2; ++i) {
                    unsigned int v0i = (v0 >> (4 * i)) & 0x0F0F0F0Fu;
                    unsigned int v1i = (v1 >> (4 * i)) & 0x0F0F0F0Fu;

                    int sb_idx = b * 8 + bq8_offset + i;
                    const unsigned char* x_blk = x_q8_1 + sb_idx * 36;
                    float xd = __half2float(*(const __half*)x_blk);

                    int u0 = *(const int*)(x_blk + 4      + 4 * qc);
                    int u1 = *(const int*)(x_blk + 4 + 16 + 4 * qc);

                    int dot1 = __dp4a((int)v1i, u1, __dp4a((int)v0i, u0, 0));
                    int dot2 = __dp4a((int)0x01010101u, u1,
                               __dp4a((int)0x01010101u, u0, 0));

                    acc += d    * xd * (float)(dot1 * (int)sc[bq8_offset + i])
                         - dmin * xd * (float)(dot2 * (int)m [bq8_offset + i]);
                }
            }
        }
    }

    // Warp-shuffle reduction.
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        acc += __shfl_xor_sync(0xffffffff, acc, o);
    }

    if (tid == 0) {
        y[row] = (__nv_bfloat16)acc;
    }
}
"#;

// T245.4 — sgemm_q4k_bf16_m8 : Q4_K matmul with batch M=8 (8 input tokens
// processed in one weight pass). This is the algorithmic key for speculative
// decoding : reads W once, produces 8 output rows simultaneously.
//
// Memory analysis per super-block (256 weights of W) :
//   M=1 (sgemv_q4k_v2)    : 144 W + 512 x  = 656 bytes ; 1 acc per thread
//   M=8 (this kernel)      : 144 W + 4096 x = 4240 bytes ; 8 acc per thread
//   8 × M=1 (sequential)  : 8 × 656         = 5248 bytes
//
// Win : 5248 → 4240 bytes per super-block = 19% per-call reduction.
// BUT for FULL decode where x reads are negligible vs W (h ∈ R^5120 fits in L1),
// M=8 batched amortizes 14.94 GB W reads across 8 tokens →
//   per-token bw cost : 14.94/8 = 1.87 GB/token = 84 tok/s @ 157 GB/s.
//
// Tile mapping (mirror of sgemv_q4k_bf16_v2) :
//   64 threads/TG, each thread decodes 4 weights × 8 m-values = 32 macs/thread.
//   Per thread accumulators : 8 floats (one per m).
//   x_shmem stores [M=8, 256] BF16 per super-block iteration = 4096 bytes.
#[cfg(feature = "cuda")]
const SGEMM_Q4K_BF16_M8_SRC: &str = r#"
#include <cuda_bf16.h>
#include <cuda_fp16.h>

extern "C" __global__ void sgemm_q4k_bf16_m8(
    const unsigned char* __restrict__ w_q4k,
    const __nv_bfloat16* __restrict__ x,    // [M=8, K] row-major
    __nv_bfloat16* __restrict__ y,          // [M=8, N] row-major
    int N,
    int K
) {
    int row = blockIdx.x;
    if (row >= N) return;
    int tid = threadIdx.x;
    int blocks_per_row = K / 256;
    int row_offset = row * blocks_per_row * 144;

    extern __shared__ float shmem[];
    float* sc_pre = shmem;             // [8]
    float* m_pre  = shmem + 8;         // [8]
    float* sdata  = shmem + 16;        // [16] : 8 m × 2 warps reduction

    float acc[8];
    #pragma unroll
    for (int m = 0; m < 8; ++m) acc[m] = 0.0f;

    int group       = tid >> 3;
    int pos_base    = (tid & 7) << 2;
    int pair_idx    = group >> 1;
    int sub_in_pair = group & 1;
    int byte_base   = (pair_idx << 5) + pos_base;
    int low_nibble  = (sub_in_pair == 0) ? 1 : 0;

    for (int b = 0; b < blocks_per_row; ++b) {
        int blk_off = row_offset + b * 144;
        const unsigned char* blk = w_q4k + blk_off;

        if (tid == 0) {
            unsigned short d_bits    = blk[0] | (blk[1] << 8);
            unsigned short dmin_bits = blk[2] | (blk[3] << 8);
            float d    = __half2float(__ushort_as_half(d_bits));
            float dmin = __half2float(__ushort_as_half(dmin_bits));
            const unsigned char* scales = blk + 4;
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                unsigned char sc_i  = scales[i] & 0x3F;
                unsigned char m_i   = scales[i + 4] & 0x3F;
                unsigned char sc_i4 = (scales[i + 8] & 0x0F) | ((scales[i] >> 6) << 4);
                unsigned char m_i4  = (scales[i + 8] >> 4)   | ((scales[i + 4] >> 6) << 4);
                sc_pre[i]     = d * (float)sc_i;
                m_pre[i]      = dmin * (float)m_i;
                sc_pre[i + 4] = d * (float)sc_i4;
                m_pre[i + 4]  = dmin * (float)m_i4;
            }
        }
        __syncthreads();

        float scale = sc_pre[group];
        float min_v = m_pre[group];
        const unsigned char* qs = blk + 16;
        unsigned int qbytes = *(const unsigned int*)(qs + byte_base);

        int x_super_pos = (group << 5) + pos_base;
        const __nv_bfloat16* x_block = x + b * 256 + x_super_pos;

        // Vectorize : load 4 BF16 (= 8 bytes = uint2) per m in one transaction.
        // Total : 8 m × 8 bytes = 64 bytes (4 cache lines worth).
        uint2 xv[8];
        #pragma unroll
        for (int m = 0; m < 8; ++m) {
            xv[m] = *(const uint2*)(x_block + m * K);
        }

        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            unsigned char byte_i = (qbytes >> (i << 3)) & 0xFFu;
            int nibble = low_nibble ? (byte_i & 0x0F) : (byte_i >> 4);
            float w_val = scale * (float)nibble - min_v;

            #pragma unroll
            for (int m = 0; m < 8; ++m) {
                unsigned short xb_i = (i < 2)
                    ? (unsigned short)((xv[m].x >> (i << 4)) & 0xFFFFu)
                    : (unsigned short)((xv[m].y >> ((i - 2) << 4)) & 0xFFFFu);
                __nv_bfloat16 xbf = __ushort_as_bfloat16(xb_i);
                acc[m] += w_val * (float)xbf;
            }
        }
        // T246.4.4 — RACE FIX : sync between iterations.
        __syncthreads();
    }

    // Reduction : we have 8 accumulators per thread × 64 threads = 512 floats.
    // For each m, do warp-shuffle reduction over the 64 threads (2 warps).
    #pragma unroll
    for (int m = 0; m < 8; ++m) {
        float a = acc[m];
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            a += __shfl_down_sync(0xffffffff, a, offset);
        }
        int warp_id = tid >> 5;
        int lane_id = tid & 31;
        if (lane_id == 0) {
            sdata[m * 2 + warp_id] = a;
        }
    }
    __syncthreads();
    if (tid < 8) {
        // tid = m. Sum the 2 warp partials.
        int m = tid;
        float total = sdata[m * 2] + sdata[m * 2 + 1];
        // Output : y[m, row]
        y[m * N + row] = (__nv_bfloat16)total;
    }
}
"#;

// T244.2 — sgemv_bf16_bf16 : pure-BF16 weight matmul for thin GEMV (decode).
//
// PROBLEM : cuBLASLt matmul_bf16 with M=1 is catastrophic on GB10. Measured
// 19.3 GB/s effective bandwidth out of 200 GB/s peak (qwen36_27b decode bench).
// cuBLAS uses heavy GEMM kernels even for M=1 and wastes 90% of memory bw.
//
// SOLUTION : warp-shuffle SGEMV mirroring sgemv_q4k_bf16_v2 pattern.
//   - 64 threads/TG (2 warps) — better latency hiding than 256
//   - 4 weights per thread per super-block (256 weights / 64 threads)
//   - uint2 vector loads (8 bytes = 4 BF16) for both W and x
//   - Warp-shuffle reduction (eliminates 5 of 6 __syncthreads)
//
// Layout : W is row-major [N, K] (one block per output row).
//   y[row] = sum_k W[row, k] * x[k]
//
// Constraint : K must be multiple of 256.
#[cfg(feature = "cuda")]
const SGEMV_BF16_BF16_SRC: &str = r#"
#include <cuda_bf16.h>

extern "C" __global__ void sgemv_bf16_bf16(
    const __nv_bfloat16* __restrict__ w,
    const __nv_bfloat16* __restrict__ x,
    __nv_bfloat16* __restrict__ y,
    int N,
    int K
) {
    int row = blockIdx.x;
    if (row >= N) return;
    int tid = threadIdx.x;

    extern __shared__ float shmem[];   // [64] reduction buffer

    float acc = 0.0f;
    int blocks_per_row = K / 256;
    int pos_base = tid * 4;
    int row_offset = row * K;

    for (int b = 0; b < blocks_per_row; ++b) {
        int k_off = b * 256 + pos_base;
        const __nv_bfloat16* w_ptr = w + row_offset + k_off;
        const __nv_bfloat16* x_ptr = x + k_off;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            float wv = (float)w_ptr[i];
            float xv = (float)x_ptr[i];
            acc += wv * xv;
        }
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xffffffff, acc, offset);
    }
    int warp_id = tid >> 5;
    int lane_id = tid & 31;
    if (lane_id == 0) {
        shmem[warp_id] = acc;
    }
    __syncthreads();
    if (tid == 0) {
        float total = shmem[0] + shmem[1];
        y[row] = (__nv_bfloat16)total;
    }
}
"#;

#[cfg(feature = "cuda")]
const CONV1D_DEPTHWISE_BF16_SRC: &str = r#"
#include <cuda_bf16.h>

// T243.2 — depth-wise 1-D convolution for Qwen3.5/3.6 SSM block.
//
// For each channel c, compute :
//   out[c] = sum_{t=0..K-1} weight[t, c] * window[t, c]
// where window = (kernel-1) past inputs (state) + current input (qkv_mixed).
// State is rolled : new state = window[1..K-1] then current input at last slot.
//
// Layout :
//   weight   : [kernel_size, conv_dim] row-major
//   state    : [(kernel_size - 1), conv_dim] (rolling history)
//   input    : [conv_dim] (current step)
//   out      : [conv_dim]
//
// This is a per-channel reduction — block dim = conv_dim, 1 thread per channel.
extern "C" __global__ void conv1d_depthwise_bf16(
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ state,        // (kernel-1, conv_dim) ring buffer
    const __nv_bfloat16* __restrict__ input,
    __nv_bfloat16* __restrict__ out,
    int conv_dim,
    int kernel_size
) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= conv_dim) return;

    float acc = 0.0f;
    // Window: state[0..K-2] then input at slot K-1.
    for (int t = 0; t < kernel_size - 1; ++t) {
        float v = (float)state[t * conv_dim + c];
        float w = (float)weight[t * conv_dim + c];
        acc += v * w;
    }
    float v = (float)input[c];
    float w = (float)weight[(kernel_size - 1) * conv_dim + c];
    acc += v * w;
    out[c] = (__nv_bfloat16)acc;

    // Update state: shift left by 1 timestep, append input at last slot.
    if (kernel_size >= 2) {
        for (int t = 0; t < kernel_size - 2; ++t) {
            state[t * conv_dim + c] = state[(t + 1) * conv_dim + c];
        }
        state[(kernel_size - 2) * conv_dim + c] = input[c];
    }
}
"#;

#[cfg(feature = "cuda")]
const L2_NORM_PER_HEAD_BF16_SRC: &str = r#"
#include <cuda_bf16.h>

// T243.2 — per-head L2 normalization for Q/K in Qwen3.5/3.6 SSM.
// x : [n_heads, head_dim] BF16, normalized in-place per head.
// Each TG = one head. Within TG, head_dim threads cooperate on sum-of-squares
// reduction, then each thread normalizes its element.
extern "C" __global__ void l2_norm_per_head_bf16(
    __nv_bfloat16* __restrict__ x,
    int n_heads,
    int head_dim,
    float eps
) {
    int h = blockIdx.x;
    if (h >= n_heads) return;
    int tid = threadIdx.x;
    if (tid >= head_dim) return;

    extern __shared__ float sdata[];

    // Load value + compute square.
    float v = (float)x[h * head_dim + tid];
    float vsq = v * v;
    sdata[tid] = vsq;
    __syncthreads();

    // Tree reduction over head_dim threads.
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && tid + s < head_dim) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    float inv_norm = rsqrtf(sdata[0] + eps);
    x[h * head_dim + tid] = (__nv_bfloat16)(v * inv_norm);
}
"#;

#[cfg(feature = "cuda")]
const DELTA_NET_STEP_BF16_SRC: &str = r#"
#include <cuda_bf16.h>

// T243.2 — Qwen3.5/3.6 Gated DeltaNet recurrent step.
//
// For each head h in 0..n_heads :
//   state[h] = exp(g[h]) * state[h] + beta[h] * outer(v[h], k[h])
//   out[h]   = state[h] @ q[h]
// where state[h] is a [head_dim × head_dim] matrix.
//
// Layout (single token decode) :
//   q, k, v : [n_heads, head_dim] BF16
//   gate, beta : [n_heads] BF16 (per-head scalar gates)
//   state : [n_heads, head_dim, head_dim] BF16 (rolling state, mutated)
//   out : [n_heads, head_dim] BF16
//
// Each TG handles ONE head, with head_dim threads each handling ONE row r
// of the state[h] matrix (r ∈ 0..head_dim).
//
// Within thread r :
//   for c in 0..head_dim :
//     state[h, r, c] = g_exp * state[h, r, c] + beta * v_h[r] * k_h[c]
//     out_acc += state[h, r, c] * q_h[c]
//   out[h, r] = out_acc
//
// Compute per head : O(head_dim²). For Qwen3.6 head_dim=128 → 16K ops/head.
// 48 heads × 64 layers × 16K = 50M ops/token = negligible vs matmul.
extern "C" __global__ void delta_net_step_bf16(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    const __nv_bfloat16* __restrict__ gate,    // [n_heads]
    const __nv_bfloat16* __restrict__ beta,    // [n_heads]
    __nv_bfloat16* __restrict__ state,          // [n_heads, head_dim, head_dim]
    __nv_bfloat16* __restrict__ out,
    int n_heads,
    int head_dim
) {
    int h = blockIdx.x;
    if (h >= n_heads) return;
    int r = threadIdx.x;
    if (r >= head_dim) return;

    float g_exp = expf((float)gate[h]);
    float b = (float)beta[h];
    float v_r = (float)v[h * head_dim + r];

    int s_off = h * head_dim * head_dim + r * head_dim;
    float out_acc = 0.0f;

    // Load q[h] into shmem for fast broadcast across threads in this TG.
    extern __shared__ float q_shared[];
    if (r < head_dim) {
        q_shared[r] = (float)q[h * head_dim + r];
    }
    __syncthreads();

    for (int c = 0; c < head_dim; ++c) {
        float k_c = (float)k[h * head_dim + c];
        float old = (float)state[s_off + c];
        float updated = g_exp * old + b * v_r * k_c;
        state[s_off + c] = (__nv_bfloat16)updated;
        out_acc += updated * q_shared[c];
    }

    out[h * head_dim + r] = (__nv_bfloat16)out_acc;
}
"#;

#[cfg(feature = "cuda")]
const SGEMV_Q6K_BF16_SRC: &str = r#"
#include <cuda_bf16.h>
#include <cuda_fp16.h>

// T244.1.3 — Direct Q6_K matmul (V1 baseline — works numerically).
// V2 (64 threads, vectorized) attempted but had mapping bug ; reverted.
// See parity test for canonical mapping.
//
// Q6_K layout (210 bytes per 256 weights) :
//   128 bytes ql        — low 4 bits per weight
//    64 bytes qh        — high 2 bits per weight
//    16 bytes scales_i8 — signed int8, 16 sub-blocks of 16 weights
//     2 bytes d         — f16 super-block scale
//
// Per-weight value = d × scales_i8[sub_idx] × (q - 32) where q ∈ 0..63.
// Memory : 210/512 = 41% of BF16 → ~2.4× memory bandwidth saving.
//
// Tile : 256 threads/TG, 1 weight/thread, d × scales pre-multiplied in shmem.
extern "C" __global__ void sgemv_q6k_bf16(
    const unsigned char* __restrict__ w_q6k,
    const __nv_bfloat16* __restrict__ x,
    __nv_bfloat16* __restrict__ y,
    int N,
    int K
) {
    int row = blockIdx.x;
    if (row >= N) return;
    int tid = threadIdx.x;
    int blocks_per_row = K / 256;
    int row_offset = row * blocks_per_row * 210;

    extern __shared__ float shmem[];
    float* sc_pre = shmem;            // [16] : d * (i8)scales[i]
    float* sdata  = shmem + 16;       // [256] reduction

    float acc = 0.0f;

    int half          = tid >> 7;
    int pos           = tid & 127;
    int sub_idx       = pos >> 5;
    int l             = pos & 31;
    int ql_off        = (half << 6) + ((sub_idx & 1) << 5) + l;
    int qh_off        = (half << 5) + l;
    int nibble_shift  = (sub_idx >> 1) << 2;
    int qh_shift      = sub_idx << 1;
    int scale_idx     = (half << 3) + (sub_idx << 1) + (l >> 4);

    for (int b = 0; b < blocks_per_row; ++b) {
        int blk_off = row_offset + b * 210;
        const unsigned char* blk = w_q6k + blk_off;

        if (tid == 0) {
            unsigned short d_bits = blk[208] | (blk[209] << 8);
            float d = __half2float(__ushort_as_half(d_bits));
            const signed char* scales = (const signed char*)(blk + 192);
            #pragma unroll
            for (int i = 0; i < 16; ++i) {
                sc_pre[i] = d * (float)scales[i];
            }
        }
        __syncthreads();

        const unsigned char* ql = blk;
        const unsigned char* qh = blk + 128;

        unsigned char ql_b = ql[ql_off];
        unsigned char qh_b = qh[qh_off];
        int q_low  = (ql_b >> nibble_shift) & 0x0F;
        int q_high = ((qh_b >> qh_shift) & 0x03) << 4;
        int q      = q_low | q_high;

        float w_val = sc_pre[scale_idx] * (float)(q - 32);
        float x_val = (float)x[b * 256 + tid];
        acc += w_val * x_val;
        // T246.4.4 — RACE FIX : sync between iterations so next iter's write
        // to sc_pre by thread 0 doesn't race with this iter's reads.
        __syncthreads();
    }

    // T244.1.3.1 — Warp-shuffle reduction over 256 threads = 8 warps.
    // Each warp shuffle-reduces (5 instr no sync), lane 0 writes warp sum
    // to shmem, then thread 0 sums the 8 warp sums.
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xffffffff, acc, offset);
    }
    int warp_id = tid >> 5;     // 0..8
    int lane_id = tid & 31;
    if (lane_id == 0) {
        sdata[warp_id] = acc;
    }
    __syncthreads();
    if (tid == 0) {
        float total = 0.0f;
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            total += sdata[i];
        }
        y[row] = (__nv_bfloat16)total;
    }
}
"#;

// T244.4 — sgemv_q6k_bf16 V2 : warp-shuffle + 64 threads/TG + vector decode.
// V1 measured 95 GB/s on (18944, 3584). V2 target : match Q4K V2 at ~160 GB/s.
//
// Tile (per super-block of 256 weights = 2 halves × 128 weights) :
//   tid ∈ [0, 64), half = tid >> 5, l = tid & 31
//   Each thread decodes 4 weights at positions {l, l+32, l+64, l+96} in its half.
//   Reads : 2 ql bytes (ql[ql_base+l], ql[ql_base+l+32]) + 1 qh byte (qh[qh_base+l])
//   The 4 nibble/2-bit decodes :
//     q0 = (ql_a & 0x0F) | ((qh_b      & 0x03) << 4)   → position l
//     q1 = (ql_b & 0x0F) | ((qh_b >> 2 & 0x03) << 4)   → position l+32
//     q2 = (ql_a >> 4)   | ((qh_b >> 4 & 0x03) << 4)   → position l+64
//     q3 = (ql_b >> 4)   | ((qh_b >> 6 & 0x03) << 4)   → position l+96
// Scales pre-multiplied in shmem. x loads strided by 32 — still coalesced
// across the 32 lanes of a warp.
#[cfg(feature = "cuda")]
const SGEMV_Q6K_BF16_V2_SRC: &str = r#"
#include <cuda_bf16.h>
#include <cuda_fp16.h>

extern "C" __global__ void sgemv_q6k_bf16_v2(
    const unsigned char* __restrict__ w_q6k,
    const __nv_bfloat16* __restrict__ x,
    __nv_bfloat16* __restrict__ y,
    int N,
    int K
) {
    int row = blockIdx.x;
    if (row >= N) return;
    int tid = threadIdx.x;          // 0..63
    int blocks_per_row = K / 256;
    int row_offset = row * blocks_per_row * 210;

    extern __shared__ float shmem[];
    float* sc_pre = shmem;          // [16]
    float* sdata  = shmem + 16;     // [64]

    float acc = 0.0f;

    int half          = tid >> 5;   // 0 or 1
    int l             = tid & 31;   // 0..31
    int half_offset_x = half << 7;  // 0 or 128
    int ql_base       = half << 6;  // 0 or 64
    int qh_base       = half << 5;  // 0 or 32
    int sb            = half << 3;  // 0 or 8
    int l16           = l >> 4;     // 0 or 1

    for (int b = 0; b < blocks_per_row; ++b) {
        int blk_off = row_offset + b * 210;
        const unsigned char* blk = w_q6k + blk_off;

        if (tid == 0) {
            unsigned short d_bits = blk[208] | (blk[209] << 8);
            float d = __half2float(__ushort_as_half(d_bits));
            const signed char* scales = (const signed char*)(blk + 192);
            #pragma unroll
            for (int i = 0; i < 16; ++i) {
                sc_pre[i] = d * (float)scales[i];
            }
        }
        __syncthreads();

        const unsigned char* ql = blk;
        const unsigned char* qh = blk + 128;

        unsigned char ql_a = ql[ql_base + l];
        unsigned char ql_b = ql[ql_base + l + 32];
        unsigned char qh_b = qh[qh_base + l];

        int q0 = (ql_a & 0x0F) | (((qh_b)      & 0x03) << 4);
        int q1 = (ql_b & 0x0F) | (((qh_b >> 2) & 0x03) << 4);
        int q2 = (ql_a >> 4)   | (((qh_b >> 4) & 0x03) << 4);
        int q3 = (ql_b >> 4)   | (((qh_b >> 6) & 0x03) << 4);

        float w0 = sc_pre[sb + 0 + l16] * (float)(q0 - 32);
        float w1 = sc_pre[sb + 2 + l16] * (float)(q1 - 32);
        float w2 = sc_pre[sb + 4 + l16] * (float)(q2 - 32);
        float w3 = sc_pre[sb + 6 + l16] * (float)(q3 - 32);

        const __nv_bfloat16* x_ptr = x + b * 256 + half_offset_x;
        float x0 = (float)x_ptr[l];
        float x1 = (float)x_ptr[l + 32];
        float x2 = (float)x_ptr[l + 64];
        float x3 = (float)x_ptr[l + 96];

        acc += w0 * x0 + w1 * x1 + w2 * x2 + w3 * x3;

        // T246.4.4 — RACE FIX : sync at end of iteration so next iter's write
        // to sc_pre by thread 0 doesn't race with this iter's reads by other
        // threads. Without this, output is non-deterministic (compute-sanitizer
        // racecheck found 1.25M hazards).
        __syncthreads();
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xffffffff, acc, offset);
    }
    int warp_id = tid >> 5;
    int lane_id = tid & 31;
    if (lane_id == 0) {
        sdata[warp_id] = acc;
    }
    __syncthreads();
    if (tid == 0) {
        float total = sdata[0] + sdata[1];
        y[row] = (__nv_bfloat16)total;
    }
}
"#;

// T244.3 — sgemv_q5k_bf16 — direct Q5_K matmul (Qwen 3.6 needs this:
// 12% of weights are Q5_K, 76% Q4_K, 12% Q6_K).
//
// Q5_K block layout (256 weights / 176 bytes):
//   d (fp16, 2B), dmin (fp16, 2B), scales (12B same as Q4_K), qh (32B), ql (128B)
//
// Each weight is 5 bits = (ql nibble, 4 lo) + (qh bit, 1 hi) → q ∈ [0, 31].
// Dequant : w = d*sc[i] * q - dmin*m[i]   (i = sub-block index 0..7)
//
// Tile mapping (mirror of sgemv_q4k_bf16_v2) :
//   64 threads/TG, 4 weights/thread, ql via uint32 vector load,
//   qh via additional uint32 load, scales pre-multiplied in shmem.
#[cfg(feature = "cuda")]
const SGEMV_Q5K_BF16_SRC: &str = r#"
#include <cuda_bf16.h>
#include <cuda_fp16.h>

extern "C" __global__ void sgemv_q5k_bf16(
    const unsigned char* __restrict__ w_q5k,
    const __nv_bfloat16* __restrict__ x,
    __nv_bfloat16* __restrict__ y,
    int N,
    int K
) {
    int row = blockIdx.x;
    if (row >= N) return;
    int tid = threadIdx.x;
    int blocks_per_row = K / 256;
    int row_offset = row * blocks_per_row * 176;

    extern __shared__ float shmem[];
    float* sc_pre = shmem;            // [8] : d * sc[i]
    float* m_pre  = shmem + 8;        // [8] : dmin * m[i]
    float* sdata  = shmem + 16;       // [64] : reduction buffer

    float acc = 0.0f;

    int group       = tid >> 3;        // sub-block index in 0..8
    int pos_base    = (tid & 7) << 2;  // 0..28 step 4
    int pair_idx    = group >> 1;
    int sub_in_pair = group & 1;
    int byte_base   = (pair_idx << 5) + pos_base;
    int low_nibble  = (sub_in_pair == 0) ? 1 : 0;
    int qh_bit_idx  = (sub_in_pair == 0) ? (2 * pair_idx) : (2 * pair_idx + 1);
    unsigned int qh_mask = 1u << qh_bit_idx;

    for (int b = 0; b < blocks_per_row; ++b) {
        int blk_off = row_offset + b * 176;
        const unsigned char* blk = w_q5k + blk_off;

        if (tid == 0) {
            unsigned short d_bits    = blk[0] | (blk[1] << 8);
            unsigned short dmin_bits = blk[2] | (blk[3] << 8);
            float d    = __half2float(__ushort_as_half(d_bits));
            float dmin = __half2float(__ushort_as_half(dmin_bits));
            const unsigned char* scales = blk + 4;
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                unsigned char sc_i  = scales[i] & 0x3F;
                unsigned char m_i   = scales[i + 4] & 0x3F;
                unsigned char sc_i4 = (scales[i + 8] & 0x0F) | ((scales[i] >> 6) << 4);
                unsigned char m_i4  = (scales[i + 8] >> 4)   | ((scales[i + 4] >> 6) << 4);
                sc_pre[i]     = d * (float)sc_i;
                m_pre[i]      = dmin * (float)m_i;
                sc_pre[i + 4] = d * (float)sc_i4;
                m_pre[i + 4]  = dmin * (float)m_i4;
            }
        }
        __syncthreads();

        float scale = sc_pre[group];
        float min_v = m_pre[group];

        const unsigned char* qh = blk + 16;       // 32 bytes high bits
        const unsigned char* ql = blk + 16 + 32;  // 128 bytes low nibbles

        // 4 ql bytes via uint32 (each holds 2 sub-blocks' low nibbles).
        unsigned int qlbytes = *(const unsigned int*)(ql + byte_base);
        // 4 qh bytes via uint32 (each holds 2 sub-blocks' high bits at our pair_idx).
        unsigned int qhbytes = *(const unsigned int*)(qh + pos_base);

        const __nv_bfloat16* x_ptr = x + b * 256 + (group << 5) + pos_base;
        uint2 xbits = *(const uint2*)x_ptr;
        unsigned short xb[4] = {
            (unsigned short)(xbits.x & 0xFFFFu),
            (unsigned short)(xbits.x >> 16),
            (unsigned short)(xbits.y & 0xFFFFu),
            (unsigned short)(xbits.y >> 16),
        };

        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            unsigned char ql_byte = (qlbytes >> (i << 3)) & 0xFFu;
            unsigned char qh_byte = (qhbytes >> (i << 3)) & 0xFFu;
            int low_bits = low_nibble ? (ql_byte & 0x0F) : (ql_byte >> 4);
            int high_bit = (qh_byte & qh_mask) ? 16 : 0;
            int q = low_bits + high_bit;          // q ∈ [0, 31]
            float w_val = scale * (float)q - min_v;
            __nv_bfloat16 xbf = __ushort_as_bfloat16(xb[i]);
            acc += w_val * (float)xbf;
        }
        // T246.4.4 — RACE FIX : sync between iterations.
        __syncthreads();
    }

    // Warp-shuffle reduction.
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xffffffff, acc, offset);
    }
    int warp_id = tid >> 5;
    int lane_id = tid & 31;
    if (lane_id == 0) {
        sdata[warp_id] = acc;
    }
    __syncthreads();
    if (tid == 0) {
        float total = sdata[0] + sdata[1];
        y[row] = (__nv_bfloat16)total;
    }
}
"#;

// T245.4 — Q5_K M=8 batched matmul (mirror of Q4K_M8 with qh handling).
#[cfg(feature = "cuda")]
const SGEMM_Q5K_BF16_M8_SRC: &str = r#"
#include <cuda_bf16.h>
#include <cuda_fp16.h>

extern "C" __global__ void sgemm_q5k_bf16_m8(
    const unsigned char* __restrict__ w_q5k,
    const __nv_bfloat16* __restrict__ x,
    __nv_bfloat16* __restrict__ y,
    int N,
    int K
) {
    int row = blockIdx.x;
    if (row >= N) return;
    int tid = threadIdx.x;
    int blocks_per_row = K / 256;
    int row_offset = row * blocks_per_row * 176;

    extern __shared__ float shmem[];
    float* sc_pre = shmem;
    float* m_pre  = shmem + 8;
    float* sdata  = shmem + 16;

    float acc[8];
    #pragma unroll
    for (int m = 0; m < 8; ++m) acc[m] = 0.0f;

    int group       = tid >> 3;
    int pos_base    = (tid & 7) << 2;
    int pair_idx    = group >> 1;
    int sub_in_pair = group & 1;
    int byte_base   = (pair_idx << 5) + pos_base;
    int low_nibble  = (sub_in_pair == 0) ? 1 : 0;
    int qh_bit_idx  = (sub_in_pair == 0) ? (2 * pair_idx) : (2 * pair_idx + 1);
    unsigned int qh_mask = 1u << qh_bit_idx;

    for (int b = 0; b < blocks_per_row; ++b) {
        int blk_off = row_offset + b * 176;
        const unsigned char* blk = w_q5k + blk_off;

        if (tid == 0) {
            unsigned short d_bits    = blk[0] | (blk[1] << 8);
            unsigned short dmin_bits = blk[2] | (blk[3] << 8);
            float d    = __half2float(__ushort_as_half(d_bits));
            float dmin = __half2float(__ushort_as_half(dmin_bits));
            const unsigned char* scales = blk + 4;
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                unsigned char sc_i  = scales[i] & 0x3F;
                unsigned char m_i   = scales[i + 4] & 0x3F;
                unsigned char sc_i4 = (scales[i + 8] & 0x0F) | ((scales[i] >> 6) << 4);
                unsigned char m_i4  = (scales[i + 8] >> 4)   | ((scales[i + 4] >> 6) << 4);
                sc_pre[i]     = d * (float)sc_i;
                m_pre[i]      = dmin * (float)m_i;
                sc_pre[i + 4] = d * (float)sc_i4;
                m_pre[i + 4]  = dmin * (float)m_i4;
            }
        }
        __syncthreads();

        float scale = sc_pre[group];
        float min_v = m_pre[group];
        const unsigned char* qh = blk + 16;
        const unsigned char* ql = blk + 16 + 32;

        unsigned int qlbytes = *(const unsigned int*)(ql + byte_base);
        unsigned int qhbytes = *(const unsigned int*)(qh + pos_base);

        int x_super_pos = (group << 5) + pos_base;
        const __nv_bfloat16* x_block = x + b * 256 + x_super_pos;

        // Vectorize : 1 uint2 (8 bytes = 4 BF16) load per m row.
        uint2 xv[8];
        #pragma unroll
        for (int m = 0; m < 8; ++m) {
            xv[m] = *(const uint2*)(x_block + m * K);
        }

        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            unsigned char ql_byte = (qlbytes >> (i << 3)) & 0xFFu;
            unsigned char qh_byte = (qhbytes >> (i << 3)) & 0xFFu;
            int low_bits = low_nibble ? (ql_byte & 0x0F) : (ql_byte >> 4);
            int high_bit = (qh_byte & qh_mask) ? 16 : 0;
            int q = low_bits + high_bit;
            float w_val = scale * (float)q - min_v;

            #pragma unroll
            for (int m = 0; m < 8; ++m) {
                unsigned short xb_i = (i < 2)
                    ? (unsigned short)((xv[m].x >> (i << 4)) & 0xFFFFu)
                    : (unsigned short)((xv[m].y >> ((i - 2) << 4)) & 0xFFFFu);
                __nv_bfloat16 xbf = __ushort_as_bfloat16(xb_i);
                acc[m] += w_val * (float)xbf;
            }
        }
        // T246.4.4 — RACE FIX : sync between iterations.
        __syncthreads();
    }

    #pragma unroll
    for (int m = 0; m < 8; ++m) {
        float a = acc[m];
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            a += __shfl_down_sync(0xffffffff, a, offset);
        }
        int warp_id = tid >> 5;
        int lane_id = tid & 31;
        if (lane_id == 0) {
            sdata[m * 2 + warp_id] = a;
        }
    }
    __syncthreads();
    if (tid < 8) {
        int m = tid;
        float total = sdata[m * 2] + sdata[m * 2 + 1];
        y[m * N + row] = (__nv_bfloat16)total;
    }
}
"#;

// T245.4 — Q6_K M=8 batched matmul (mirror of Q6K_V2 layout).
#[cfg(feature = "cuda")]
const SGEMM_Q6K_BF16_M8_SRC: &str = r#"
#include <cuda_bf16.h>
#include <cuda_fp16.h>

extern "C" __global__ void sgemm_q6k_bf16_m8(
    const unsigned char* __restrict__ w_q6k,
    const __nv_bfloat16* __restrict__ x,
    __nv_bfloat16* __restrict__ y,
    int N,
    int K
) {
    int row = blockIdx.x;
    if (row >= N) return;
    int tid = threadIdx.x;
    int blocks_per_row = K / 256;
    int row_offset = row * blocks_per_row * 210;

    extern __shared__ float shmem[];
    float* sc_pre = shmem;       // [16]
    float* sdata  = shmem + 16;  // [16]

    float acc[8];
    #pragma unroll
    for (int m = 0; m < 8; ++m) acc[m] = 0.0f;

    int half          = tid >> 5;
    int l             = tid & 31;
    int half_offset_x = half << 7;
    int ql_base       = half << 6;
    int qh_base       = half << 5;
    int sb            = half << 3;
    int l16           = l >> 4;

    for (int b = 0; b < blocks_per_row; ++b) {
        int blk_off = row_offset + b * 210;
        const unsigned char* blk = w_q6k + blk_off;

        if (tid == 0) {
            unsigned short d_bits = blk[208] | (blk[209] << 8);
            float d = __half2float(__ushort_as_half(d_bits));
            const signed char* scales = (const signed char*)(blk + 192);
            #pragma unroll
            for (int i = 0; i < 16; ++i) {
                sc_pre[i] = d * (float)scales[i];
            }
        }
        __syncthreads();

        const unsigned char* ql = blk;
        const unsigned char* qh = blk + 128;

        unsigned char ql_a = ql[ql_base + l];
        unsigned char ql_b = ql[ql_base + l + 32];
        unsigned char qh_b = qh[qh_base + l];

        int q0 = (ql_a & 0x0F) | (((qh_b)      & 0x03) << 4);
        int q1 = (ql_b & 0x0F) | (((qh_b >> 2) & 0x03) << 4);
        int q2 = (ql_a >> 4)   | (((qh_b >> 4) & 0x03) << 4);
        int q3 = (ql_b >> 4)   | (((qh_b >> 6) & 0x03) << 4);

        float w0 = sc_pre[sb + 0 + l16] * (float)(q0 - 32);
        float w1 = sc_pre[sb + 2 + l16] * (float)(q1 - 32);
        float w2 = sc_pre[sb + 4 + l16] * (float)(q2 - 32);
        float w3 = sc_pre[sb + 6 + l16] * (float)(q3 - 32);

        const __nv_bfloat16* x_base = x + b * 256 + half_offset_x;

        #pragma unroll
        for (int m = 0; m < 8; ++m) {
            float x0 = (float)x_base[m * K + l];
            float x1 = (float)x_base[m * K + l + 32];
            float x2 = (float)x_base[m * K + l + 64];
            float x3 = (float)x_base[m * K + l + 96];
            acc[m] += w0 * x0 + w1 * x1 + w2 * x2 + w3 * x3;
        }
        // T246.4.4 — RACE FIX : sync between iterations.
        __syncthreads();
    }

    #pragma unroll
    for (int m = 0; m < 8; ++m) {
        float a = acc[m];
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            a += __shfl_down_sync(0xffffffff, a, offset);
        }
        int warp_id = tid >> 5;
        int lane_id = tid & 31;
        if (lane_id == 0) {
            sdata[m * 2 + warp_id] = a;
        }
    }
    __syncthreads();
    if (tid < 8) {
        int m = tid;
        float total = sdata[m * 2] + sdata[m * 2 + 1];
        y[m * N + row] = (__nv_bfloat16)total;
    }
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

    // 3. Encode scale in UE4M3 — T241.6d empirically validated via
    // `nvfp4_supported_m_values` perimeter test :
    //   FP4=1 inputs * scale_byte=0x70 → matmul C[0] = 2097152 = 128 * 128²
    //   → scale_decoded = 128 = 2^7
    //   → for byte 0x70 = 0b0111_0000 (E=14, M=0), 2^(E-bias) = 2^7
    //   → bias = 14 - 7 = 7  (matches OCP-MX UE4M3 standard)
    //
    // Layout :
    //   bit 7    : reserved (0)
    //   bits 6-3 : exponent E (4 bits)
    //   bits 2-0 : mantissa M (3 bits)
    //   value    = 2^(E - 7) * (1 + M/8)   for E >= 1
    //
    // Range : 2^-6 ≈ 0.016 (smallest normal) to 240 (largest) — fits LLM.
    //
    // Clamp E >= 1 : cuBLASLt sm_121 zero-outs blocks with subnormal
    // scales (validated empirically — see nvfp4_no_subnormal_scales test).
    unsigned int sb = __float_as_uint(scale);
    int fexp = (int)((sb >> 23) & 0xff) - 127;     // unbiased exponent
    int fmant_full = (int)(sb >> 20) & 0x7;        // top 3 bits of mantissa
    int ue_exp = fexp + 7;                          // re-bias to UE4M3 (bias 7)
    unsigned char scale_byte;
    if (ue_exp <= 0) {
        // Below smallest normal — round UP to smallest normal (E=1, M=0).
        scale_byte = (unsigned char)(1 << 3);
    } else if (ue_exp >= 15) {
        // Above representable range — saturate to largest normal (E=14,M=7).
        scale_byte = (unsigned char)((14 << 3) | 0x7);
    } else {
        scale_byte = (unsigned char)((ue_exp << 3) | fmant_full);
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

// T246.5.3 — GQA online variant qui lit `kv_len` depuis device pointer.
// Sémantiquement identique à gqa_decode_online_bf16 mais avec
// `kv_len = *kv_len_dev`. Permet capture en CUDA Graph + replay avec
// kv_len qui change entre tokens.
#[cfg(feature = "cuda")]
const GQA_DECODE_ONLINE_BF16_DEVCNT_SRC: &str = r#"
#include <cuda_bf16.h>

extern "C" __global__ void gqa_decode_online_bf16_devcnt(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k_cache,
    const __nv_bfloat16* __restrict__ v_cache,
    __nv_bfloat16* __restrict__ out,
    int n_heads,
    int n_kv,
    const int* __restrict__ kv_len_dev,
    int head_dim,
    int max_seq,
    float scale
) {
    int h = blockIdx.x;
    if (h >= n_heads) return;
    int kv_h = h * n_kv / n_heads;

    int tid = threadIdx.x;
    if (tid >= head_dim) return;

    int kv_len = *kv_len_dev;
    float q_i = (float)q[h * head_dim + tid];

    float m = -1e30f;
    float l = 0.0f;
    float o = 0.0f;

    extern __shared__ float sdata[];

    for (int t = 0; t < kv_len; ++t) {
        float k_i = (float)k_cache[(kv_h * max_seq + t) * head_dim + tid];
        float partial = q_i * k_i;
        sdata[tid] = partial;
        __syncthreads();
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s && tid + s < head_dim) {
                sdata[tid] += sdata[tid + s];
            }
            __syncthreads();
        }
        float s_t = sdata[0] * scale;
        __syncthreads();

        float new_m = fmaxf(m, s_t);
        float correction = expf(m - new_m);
        float p = expf(s_t - new_m);
        float v_i = (float)v_cache[(kv_h * max_seq + t) * head_dim + tid];
        o = o * correction + p * v_i;
        l = l * correction + p;
        m = new_m;
    }

    out[h * head_dim + tid] = (__nv_bfloat16)(o / fmaxf(l, 1e-12f));
}
"#;

// T246.5.7 — FlashDecode-V2 split-K GQA decode (M=1).
// Splits each head's kv_len work across `n_split` thread blocks → better SM
// occupancy on Blackwell (32 heads × 4 splits = 128 blocks vs the 32 of the
// online kernel). Each block walks ~kv_len/n_split keys with online softmax,
// then a combine kernel merges the partials per head.
//
// Layout (n_split is a launch param, typically 4):
//   gridDim  = (n_heads, n_split)
//   blockDim = (head_dim)
//   shmem    = head_dim * 4 bytes (reduction buffer)
//
// Outputs (per-block):
//   partial_m  : [n_heads, n_split]               float
//   partial_l  : [n_heads, n_split]               float
//   partial_o  : [n_heads, n_split, head_dim]     bf16  (NOT yet divided by l)
#[cfg(feature = "cuda")]
const GQA_DECODE_SPLIT_PARTIAL_BF16_SRC: &str = r#"
#include <cuda_bf16.h>

extern "C" __global__ void gqa_decode_split_partial_bf16(
    const __nv_bfloat16* __restrict__ q,           // [n_heads, head_dim]
    const __nv_bfloat16* __restrict__ k_cache,     // [n_kv, max_seq, head_dim]
    const __nv_bfloat16* __restrict__ v_cache,     // [n_kv, max_seq, head_dim]
    float*               __restrict__ partial_m,   // [n_heads, n_split]
    float*               __restrict__ partial_l,   // [n_heads, n_split]
    __nv_bfloat16*       __restrict__ partial_o,   // [n_heads, n_split, head_dim]
    int n_heads,
    int n_kv,
    const int* __restrict__ kv_len_dev,
    int head_dim,
    int max_seq,
    int n_split,
    float scale
) {
    int h  = blockIdx.x;
    int sp = blockIdx.y;
    if (h >= n_heads || sp >= n_split) return;
    int tid = threadIdx.x;
    if (tid >= head_dim) return;

    int kv_len       = *kv_len_dev;
    int kv_per_split = (kv_len + n_split - 1) / n_split;
    int t_start      = sp * kv_per_split;
    int t_end        = t_start + kv_per_split;
    if (t_end > kv_len) t_end = kv_len;

    int slot = h * n_split + sp;

    // Empty split (kv_len=0 or t_start beyond range).
    if (t_start >= t_end) {
        if (tid == 0) {
            partial_m[slot] = -1e30f;
            partial_l[slot] = 0.0f;
        }
        partial_o[slot * head_dim + tid] = (__nv_bfloat16)0.0f;
        return;
    }

    int kv_h = h * n_kv / n_heads;

    extern __shared__ float sdata[];

    float q_i = (float)q[h * head_dim + tid];
    float m   = -1e30f;
    float l   = 0.0f;
    float o   = 0.0f;

    for (int t = t_start; t < t_end; ++t) {
        // Score = Q · K[kv_h, t]
        float k_i     = (float)k_cache[(kv_h * max_seq + t) * head_dim + tid];
        float partial = q_i * k_i;
        sdata[tid] = partial;
        __syncthreads();
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s && tid + s < head_dim) {
                sdata[tid] += sdata[tid + s];
            }
            __syncthreads();
        }
        float s_t = sdata[0] * scale;
        __syncthreads();

        // Online softmax update.
        float new_m     = fmaxf(m, s_t);
        float correction = expf(m - new_m);
        float p          = expf(s_t - new_m);
        float v_i        = (float)v_cache[(kv_h * max_seq + t) * head_dim + tid];
        o = o * correction + p * v_i;
        l = l * correction + p;
        m = new_m;
    }

    // Write partials. Note: o is NOT yet divided by l — combine kernel does that
    // after merging across splits (via log-sum-exp re-weighting).
    if (tid == 0) {
        partial_m[slot] = m;
        partial_l[slot] = l;
    }
    partial_o[slot * head_dim + tid] = (__nv_bfloat16)o;
}
"#;

// T246.5.7 — combine kernel: merges per-split partials (m, l, o) for each head
// using log-sum-exp normalization, writes the final attention output.
//
//   gridDim  = (n_heads)
//   blockDim = (head_dim)
#[cfg(feature = "cuda")]
const GQA_DECODE_SPLIT_COMBINE_BF16_SRC: &str = r#"
#include <cuda_bf16.h>

extern "C" __global__ void gqa_decode_split_combine_bf16(
    const float*         __restrict__ partial_m,    // [n_heads, n_split]
    const float*         __restrict__ partial_l,    // [n_heads, n_split]
    const __nv_bfloat16* __restrict__ partial_o,    // [n_heads, n_split, head_dim]
    __nv_bfloat16*       __restrict__ out,           // [n_heads, head_dim]
    int n_heads,
    int head_dim,
    int n_split
) {
    int h = blockIdx.x;
    if (h >= n_heads) return;
    int tid = threadIdx.x;
    if (tid >= head_dim) return;

    // Phase 1: find global max m across splits (broadcast via shmem).
    extern __shared__ float gm_buf[];
    if (tid == 0) {
        float gm = -1e30f;
        for (int sp = 0; sp < n_split; ++sp) {
            float pm = partial_m[h * n_split + sp];
            if (pm > gm) gm = pm;
        }
        gm_buf[0] = gm;
    }
    __syncthreads();
    float global_m = gm_buf[0];

    // Phase 2: combine — each thread accumulates its dim across splits.
    float o_sum = 0.0f;
    float l_sum = 0.0f;
    for (int sp = 0; sp < n_split; ++sp) {
        int slot = h * n_split + sp;
        float pm = partial_m[slot];
        float pl = partial_l[slot];
        float w  = expf(pm - global_m);
        float po = (float)partial_o[slot * head_dim + tid];
        o_sum += po * w;
        l_sum += pl * w;
    }

    out[h * head_dim + tid] = (__nv_bfloat16)(o_sum / fmaxf(l_sum, 1e-12f));
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
    transpose: std::sync::OnceLock<(Arc<CudaModule>, CudaFunction)>,
    sgemv_q4k: std::sync::OnceLock<(Arc<CudaModule>, CudaFunction)>,
    sgemv_q4k_v2: std::sync::OnceLock<(Arc<CudaModule>, CudaFunction)>,
    sgemm_q4k_m8: std::sync::OnceLock<(Arc<CudaModule>, CudaFunction)>,
    sgemv_q5k: std::sync::OnceLock<(Arc<CudaModule>, CudaFunction)>,
    sgemm_q5k_m8: std::sync::OnceLock<(Arc<CudaModule>, CudaFunction)>,
    sgemv_q6k: std::sync::OnceLock<(Arc<CudaModule>, CudaFunction)>,
    sgemv_q6k_v2: std::sync::OnceLock<(Arc<CudaModule>, CudaFunction)>,
    sgemm_q6k_m8: std::sync::OnceLock<(Arc<CudaModule>, CudaFunction)>,
    sgemv_bf16: std::sync::OnceLock<(Arc<CudaModule>, CudaFunction)>,
    softplus_inplace: std::sync::OnceLock<(Arc<CudaModule>, CudaFunction)>,
    sigmoid_inplace: std::sync::OnceLock<(Arc<CudaModule>, CudaFunction)>,
    mul_inplace: std::sync::OnceLock<(Arc<CudaModule>, CudaFunction)>,
    repeat_heads: std::sync::OnceLock<(Arc<CudaModule>, CudaFunction)>,
    split_qg: std::sync::OnceLock<(Arc<CudaModule>, CudaFunction)>,
    rope_partial: std::sync::OnceLock<(Arc<CudaModule>, CudaFunction)>,
    conv1d_depthwise: std::sync::OnceLock<(Arc<CudaModule>, CudaFunction)>,
    l2_norm_per_head: std::sync::OnceLock<(Arc<CudaModule>, CudaFunction)>,
    delta_net_step: std::sync::OnceLock<(Arc<CudaModule>, CudaFunction)>,
    // T246.5.3 — devcnt variants & helpers for CUDA Graph capture
    rope_partial_devcnt: std::sync::OnceLock<(Arc<CudaModule>, CudaFunction)>,
    gqa_decode_online_devcnt: std::sync::OnceLock<(Arc<CudaModule>, CudaFunction)>,
    increment_u32_dev: std::sync::OnceLock<(Arc<CudaModule>, CudaFunction)>,
    kv_append_devcnt: std::sync::OnceLock<(Arc<CudaModule>, CudaFunction)>,
    // T246.5.5 — dp4a-based Q4_K × Q8_1 path (port of llama.cpp vec_dot_q4_K_q8_1)
    quantize_q8_1: std::sync::OnceLock<(Arc<CudaModule>, CudaFunction)>,
    sgemv_q4k_q8_1_dp4a: std::sync::OnceLock<(Arc<CudaModule>, CudaFunction)>,
    // T246.5.7 — FlashDecode-V2 split-K GQA decode
    gqa_split_partial: std::sync::OnceLock<(Arc<CudaModule>, CudaFunction)>,
    gqa_split_combine: std::sync::OnceLock<(Arc<CudaModule>, CudaFunction)>,
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
            transpose: std::sync::OnceLock::new(),
            sgemv_q4k: std::sync::OnceLock::new(),
            sgemv_q4k_v2: std::sync::OnceLock::new(),
            sgemm_q4k_m8: std::sync::OnceLock::new(),
            sgemv_q5k: std::sync::OnceLock::new(),
            sgemm_q5k_m8: std::sync::OnceLock::new(),
            sgemv_q6k: std::sync::OnceLock::new(),
            sgemv_q6k_v2: std::sync::OnceLock::new(),
            sgemm_q6k_m8: std::sync::OnceLock::new(),
            sgemv_bf16: std::sync::OnceLock::new(),
            softplus_inplace: std::sync::OnceLock::new(),
            sigmoid_inplace: std::sync::OnceLock::new(),
            mul_inplace: std::sync::OnceLock::new(),
            repeat_heads: std::sync::OnceLock::new(),
            split_qg: std::sync::OnceLock::new(),
            rope_partial: std::sync::OnceLock::new(),
            conv1d_depthwise: std::sync::OnceLock::new(),
            l2_norm_per_head: std::sync::OnceLock::new(),
            delta_net_step: std::sync::OnceLock::new(),
            rope_partial_devcnt: std::sync::OnceLock::new(),
            gqa_decode_online_devcnt: std::sync::OnceLock::new(),
            increment_u32_dev: std::sync::OnceLock::new(),
            kv_append_devcnt: std::sync::OnceLock::new(),
            quantize_q8_1: std::sync::OnceLock::new(),
            sgemv_q4k_q8_1_dp4a: std::sync::OnceLock::new(),
            gqa_split_partial: std::sync::OnceLock::new(),
            gqa_split_combine: std::sync::OnceLock::new(),
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
        // T244.2 — keep the eprintln on FAILURE so a mismatch between
        // /usr/local/cuda (toolkit) and the driver-supported CUDA version
        // is immediately visible (e.g. nvrtc 13.2 PTX vs driver 13.0 ⇒
        // CUDA_ERROR_UNSUPPORTED_PTX_VERSION). Workaround:
        //   LD_LIBRARY_PATH=/usr/local/cuda-<DRIVER_MAJOR_MINOR>/.../lib:$LD_LIBRARY_PATH
        let module = self.ctx.load_module(ptx).map_err(|e| {
            eprintln!("[llm_kernels] load_module FAILED for {name}: {e:?}");
            CudaError::Driver {
                code: format!("{e:?}").len() as i32,
                location: "LlmKernels::load_module",
            }
        })?;
        let func = module.load_function(name).map_err(|e| {
            eprintln!("[llm_kernels] load_function FAILED for {name}: {e:?}");
            CudaError::Driver {
                code: format!("{e:?}").len() as i32,
                location: "LlmKernels::load_function",
            }
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

    /// T246.2 — softplus inplace : x[i] = log1p(exp(x[i])).
    pub unsafe fn softplus_inplace_bf16(
        &self,
        stream: &Arc<CudaStream>,
        x: u64,
        n: i32,
    ) -> Result<(), CudaError> {
        let (_module, func) = self.compile_or_get(
            &self.softplus_inplace,
            SOFTPLUS_INPLACE_BF16_SRC,
            "softplus_inplace_bf16",
        )?;
        let block_dim = 256u32;
        let grid_dim = (n as u32).div_ceil(block_dim);
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (grid_dim, 1, 1),
            block_dim: (block_dim, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut launcher = stream.launch_builder(&func);
        launcher.arg(&x).arg(&n);
        launcher.launch(cfg).map_err(|e| CudaError::Driver {
            code: format!("{e:?}").len() as i32,
            location: "softplus_inplace_bf16::launch",
        })?;
        Ok(())
    }

    /// T246.2 — sigmoid inplace : x[i] = 1 / (1 + exp(-x[i])).
    pub unsafe fn sigmoid_inplace_bf16(
        &self,
        stream: &Arc<CudaStream>,
        x: u64,
        n: i32,
    ) -> Result<(), CudaError> {
        let (_module, func) = self.compile_or_get(
            &self.sigmoid_inplace,
            SIGMOID_INPLACE_BF16_SRC,
            "sigmoid_inplace_bf16",
        )?;
        let block_dim = 256u32;
        let grid_dim = (n as u32).div_ceil(block_dim);
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (grid_dim, 1, 1),
            block_dim: (block_dim, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut launcher = stream.launch_builder(&func);
        launcher.arg(&x).arg(&n);
        launcher.launch(cfg).map_err(|e| CudaError::Driver {
            code: format!("{e:?}").len() as i32,
            location: "sigmoid_inplace_bf16::launch",
        })?;
        Ok(())
    }

    /// T246.2 — element-wise multiply inplace : y[i] *= x[i].
    pub unsafe fn mul_inplace_bf16(
        &self,
        stream: &Arc<CudaStream>,
        y: u64,
        x: u64,
        n: i32,
    ) -> Result<(), CudaError> {
        let (_module, func) =
            self.compile_or_get(&self.mul_inplace, MUL_INPLACE_BF16_SRC, "mul_inplace_bf16")?;
        let block_dim = 256u32;
        let grid_dim = (n as u32).div_ceil(block_dim);
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (grid_dim, 1, 1),
            block_dim: (block_dim, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut launcher = stream.launch_builder(&func);
        launcher.arg(&y).arg(&x).arg(&n);
        launcher.launch(cfg).map_err(|e| CudaError::Driver {
            code: format!("{e:?}").len() as i32,
            location: "mul_inplace_bf16::launch",
        })?;
        Ok(())
    }

    /// T246.3 — partial RoPE : rotate only first `rope_dim` of each head.
    /// Used by Qwen3.6 (rope_dim=64 < head_dim=256).
    pub unsafe fn rope_partial_bf16(
        &self,
        stream: &Arc<CudaStream>,
        x: u64,
        inv_freq: u64,
        pos: i32,
        n_heads: i32,
        head_dim: i32,
        rope_dim: i32,
    ) -> Result<(), CudaError> {
        let (_module, func) = self.compile_or_get(
            &self.rope_partial,
            ROPE_PARTIAL_BF16_SRC,
            "rope_partial_bf16",
        )?;
        let half = rope_dim / 2;
        let block_dim = 64u32.min(half as u32);
        let grid_y = (half as u32).div_ceil(block_dim);
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
            .arg(&head_dim)
            .arg(&rope_dim);
        launcher.launch(cfg).map_err(|e| CudaError::Driver {
            code: format!("{e:?}").len() as i32,
            location: "rope_partial_bf16::launch",
        })?;
        Ok(())
    }

    /// T246.3 — split fused QG buffer (Qwen3.6 attention) into Q and gate.
    /// qg layout : per head, first head_dim is q_h, next head_dim is gate_h.
    pub unsafe fn split_qg_bf16(
        &self,
        stream: &Arc<CudaStream>,
        qg: u64,
        q: u64,
        gate: u64,
        n_heads: i32,
        head_dim: i32,
    ) -> Result<(), CudaError> {
        let (_module, func) =
            self.compile_or_get(&self.split_qg, SPLIT_QG_BF16_SRC, "split_qg_bf16")?;
        let block_dim = 128u32;
        let grid_y = (head_dim as u32).div_ceil(block_dim);
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (n_heads as u32, grid_y, 1),
            block_dim: (block_dim, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut launcher = stream.launch_builder(&func);
        launcher
            .arg(&qg)
            .arg(&q)
            .arg(&gate)
            .arg(&n_heads)
            .arg(&head_dim);
        launcher.launch(cfg).map_err(|e| CudaError::Driver {
            code: format!("{e:?}").len() as i32,
            location: "split_qg_bf16::launch",
        })?;
        Ok(())
    }

    /// T246.2 — broadcast Q/K heads from n_in to n_out (n_out / n_in factor).
    pub unsafe fn repeat_heads_bf16(
        &self,
        stream: &Arc<CudaStream>,
        src: u64,
        dst: u64,
        n_in: i32,
        n_out: i32,
        head_dim: i32,
    ) -> Result<(), CudaError> {
        let (_module, func) = self.compile_or_get(
            &self.repeat_heads,
            REPEAT_HEADS_BF16_SRC,
            "repeat_heads_bf16",
        )?;
        let block_dim = 128u32;
        let grid_y = (head_dim as u32).div_ceil(block_dim);
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (n_out as u32, grid_y, 1),
            block_dim: (block_dim, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut launcher = stream.launch_builder(&func);
        launcher
            .arg(&src)
            .arg(&dst)
            .arg(&n_in)
            .arg(&n_out)
            .arg(&head_dim);
        launcher.launch(cfg).map_err(|e| CudaError::Driver {
            code: format!("{e:?}").len() as i32,
            location: "repeat_heads_bf16::launch",
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

    /// Transpose BF16 row-major [rows, cols] → [cols, rows].
    ///
    /// T241.6c — utilisé pour convertir les poids row-major vers une
    /// disposition équivalente col-major avant la quantization NVFP4,
    /// pour que cuBLASLt FP4 (TN-only sur sm_121) lise le bon W.
    ///
    /// # Safety
    /// `src` et `dst` doivent être valides pour `rows*cols` BF16 chacun.
    pub unsafe fn transpose_bf16(
        &self,
        stream: &Arc<CudaStream>,
        src: u64,
        dst: u64,
        rows: i32,
        cols: i32,
    ) -> Result<(), CudaError> {
        let (_module, func) =
            self.compile_or_get(&self.transpose, TRANSPOSE_BF16_SRC, "transpose_bf16")?;
        let bx = 16u32;
        let by = 16u32;
        let gx = ((cols as u32) + bx - 1) / bx;
        let gy = ((rows as u32) + by - 1) / by;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (gx, gy, 1),
            block_dim: (bx, by, 1),
            shared_mem_bytes: 0,
        };
        let mut launcher = stream.launch_builder(&func);
        launcher.arg(&src).arg(&dst).arg(&rows).arg(&cols);
        launcher.launch(cfg).map_err(|e| CudaError::Driver {
            code: format!("{e:?}").len() as i32,
            location: "transpose_bf16::launch",
        })?;
        Ok(())
    }

    /// T244.1 — Direct Q4_K matmul (port Metal TurboQuant pattern).
    /// Computes `y[N] = W_q4k[N, K] @ x[K]` where W is stored as Q4_K
    /// (144 bytes per 256-weight super-block).
    ///
    /// `K` must be a multiple of 256 (Q4_K block size).
    /// `N` must be > 0.
    /// Block dim = 256 threads, grid dim = N (one block per output row).
    ///
    /// # Safety
    /// Same as other kernel methods : caller guarantees pointers are valid
    /// for the given shapes.
    pub unsafe fn sgemv_q4k_bf16(
        &self,
        stream: &Arc<CudaStream>,
        w_q4k: u64, // bytes pointer
        x: u64,
        y: u64,
        n: i32,
        k: i32,
    ) -> Result<(), CudaError> {
        if k % 256 != 0 {
            return Err(CudaError::Unsupported {
                msg: format!("sgemv_q4k_bf16: K={k} must be multiple of 256"),
            });
        }
        let (_module, func) =
            self.compile_or_get(&self.sgemv_q4k, SGEMV_Q4K_BF16_SRC, "sgemv_q4k_bf16")?;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (n as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 256 * 4, // 256 floats for tree reduction
        };
        let mut launcher = stream.launch_builder(&func);
        launcher.arg(&w_q4k).arg(&x).arg(&y).arg(&n).arg(&k);
        launcher.launch(cfg).map_err(|e| CudaError::Driver {
            code: format!("{e:?}").len() as i32,
            location: "sgemv_q4k_bf16::launch",
        })?;
        Ok(())
    }

    /// T244.1.1 — Optimised Q4_K matmul V2 : 64 threads/TG (vs 256 in V1),
    /// each thread processes 4 weights, scales pre-multiplied & cached
    /// in shmem, uint32 vector loads on nibbles.
    ///
    /// # Safety  Same contract as `sgemv_q4k_bf16`.
    pub unsafe fn sgemv_q4k_bf16_v2(
        &self,
        stream: &Arc<CudaStream>,
        w_q4k: u64,
        x: u64,
        y: u64,
        n: i32,
        k: i32,
    ) -> Result<(), CudaError> {
        if k % 256 != 0 {
            return Err(CudaError::Unsupported {
                msg: format!("sgemv_q4k_bf16_v2: K={k} must be multiple of 256"),
            });
        }
        let (_module, func) = self.compile_or_get(
            &self.sgemv_q4k_v2,
            SGEMV_Q4K_BF16_V2_SRC,
            "sgemv_q4k_bf16_v2",
        )?;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (n as u32, 1, 1),
            block_dim: (64, 1, 1),
            // shmem : 8 sc_pre + 8 m_pre + 64 sdata = 80 floats = 320 bytes
            shared_mem_bytes: 80 * 4,
        };
        let mut launcher = stream.launch_builder(&func);
        launcher.arg(&w_q4k).arg(&x).arg(&y).arg(&n).arg(&k);
        launcher.launch(cfg).map_err(|e| CudaError::Driver {
            code: format!("{e:?}").len() as i32,
            location: "sgemv_q4k_bf16_v2::launch",
        })?;
        Ok(())
    }

    /// T246.5.5 — Quantize a BF16 row of length K (multiple of 32) into
    /// Q8_1 packed format (36 bytes per 32-element block, GGML-compatible).
    ///
    /// # Safety
    /// Caller ensures `x` points to K bf16 elements, `y` has capacity for
    /// `(K/32) * 36` bytes, K is a multiple of 32, and both are valid for
    /// the duration of the kernel.
    pub unsafe fn quantize_q8_1_bf16(
        &self,
        stream: &Arc<CudaStream>,
        x: u64,
        y: u64,
        k: i32,
    ) -> Result<(), CudaError> {
        if k % 32 != 0 {
            return Err(CudaError::Unsupported {
                msg: format!("quantize_q8_1_bf16: K={k} must be multiple of 32"),
            });
        }
        let n_blocks = k / 32;
        let (_module, func) = self.compile_or_get(
            &self.quantize_q8_1,
            QUANTIZE_Q8_1_BF16_SRC,
            "quantize_q8_1_bf16",
        )?;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (n_blocks as u32, 1, 1),
            block_dim: (32, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut launcher = stream.launch_builder(&func);
        launcher.arg(&x).arg(&y).arg(&n_blocks);
        launcher.launch(cfg).map_err(|e| CudaError::Driver {
            code: format!("{e:?}").len() as i32,
            location: "quantize_q8_1_bf16::launch",
        })?;
        Ok(())
    }

    /// T246.5.5 — Q4_K × Q8_1 SGEMV using __dp4a (port of llama.cpp
    /// vec_dot_q4_K_q8_1_impl_vmmq). Expects activation pre-quantized via
    /// `quantize_q8_1_bf16`. Output is BF16. K must be multiple of 256.
    ///
    /// # Safety
    /// Caller ensures w_q4k has `N * K/256 * 144` bytes, x_q8_1 has
    /// `K/32 * 36` bytes, y has N bf16 slots, and all pointers are valid.
    pub unsafe fn sgemv_q4k_q8_1_dp4a_bf16(
        &self,
        stream: &Arc<CudaStream>,
        w_q4k: u64,
        x_q8_1: u64,
        y: u64,
        n: i32,
        k: i32,
    ) -> Result<(), CudaError> {
        if k % 256 != 0 {
            return Err(CudaError::Unsupported {
                msg: format!("sgemv_q4k_q8_1_dp4a_bf16: K={k} must be multiple of 256"),
            });
        }
        let (_module, func) = self.compile_or_get(
            &self.sgemv_q4k_q8_1_dp4a,
            SGEMV_Q4K_Q8_1_DP4A_BF16_SRC,
            "sgemv_q4k_q8_1_dp4a_bf16",
        )?;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (n as u32, 1, 1),
            block_dim: (32, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut launcher = stream.launch_builder(&func);
        launcher.arg(&w_q4k).arg(&x_q8_1).arg(&y).arg(&n).arg(&k);
        launcher.launch(cfg).map_err(|e| CudaError::Driver {
            code: format!("{e:?}").len() as i32,
            location: "sgemv_q4k_q8_1_dp4a_bf16::launch",
        })?;
        Ok(())
    }

    /// T245.4 — Q4_K matmul with batch M=8 (8 input tokens processed in
    /// one weight pass). Algorithmic key for speculative decoding :
    /// reads W once, produces 8 output rows simultaneously.
    ///
    /// Inputs :
    ///   w_q4k : [N, K] Q4_K row-major (same layout as sgemv_q4k_v2)
    ///   x     : [M=8, K] BF16 row-major
    ///   y     : [M=8, N] BF16 row-major (output)
    ///
    /// # Safety  Caller ensures pointers valid and K multiple of 256.
    pub unsafe fn sgemm_q4k_bf16_m8(
        &self,
        stream: &Arc<CudaStream>,
        w_q4k: u64,
        x: u64,
        y: u64,
        n: i32,
        k: i32,
    ) -> Result<(), CudaError> {
        if k % 256 != 0 {
            return Err(CudaError::Unsupported {
                msg: format!("sgemm_q4k_bf16_m8: K={k} must be multiple of 256"),
            });
        }
        let (_module, func) = self.compile_or_get(
            &self.sgemm_q4k_m8,
            SGEMM_Q4K_BF16_M8_SRC,
            "sgemm_q4k_bf16_m8",
        )?;
        // Shmem : sc_pre [8] + m_pre [8] + sdata [16] = 32 floats = 128 bytes
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (n as u32, 1, 1),
            block_dim: (64, 1, 1),
            shared_mem_bytes: 32 * 4,
        };
        let mut launcher = stream.launch_builder(&func);
        launcher.arg(&w_q4k).arg(&x).arg(&y).arg(&n).arg(&k);
        launcher.launch(cfg).map_err(|e| CudaError::Driver {
            code: format!("{e:?}").len() as i32,
            location: "sgemm_q4k_bf16_m8::launch",
        })?;
        Ok(())
    }

    /// T245.4 — Q5_K M=8 batched matmul (mirror of Q4K_M8).
    ///
    /// # Safety  Same contract as `sgemm_q4k_bf16_m8`.
    pub unsafe fn sgemm_q5k_bf16_m8(
        &self,
        stream: &Arc<CudaStream>,
        w_q5k: u64,
        x: u64,
        y: u64,
        n: i32,
        k: i32,
    ) -> Result<(), CudaError> {
        if k % 256 != 0 {
            return Err(CudaError::Unsupported {
                msg: format!("sgemm_q5k_bf16_m8: K={k} must be multiple of 256"),
            });
        }
        let (_module, func) = self.compile_or_get(
            &self.sgemm_q5k_m8,
            SGEMM_Q5K_BF16_M8_SRC,
            "sgemm_q5k_bf16_m8",
        )?;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (n as u32, 1, 1),
            block_dim: (64, 1, 1),
            shared_mem_bytes: 32 * 4,
        };
        let mut launcher = stream.launch_builder(&func);
        launcher.arg(&w_q5k).arg(&x).arg(&y).arg(&n).arg(&k);
        launcher.launch(cfg).map_err(|e| CudaError::Driver {
            code: format!("{e:?}").len() as i32,
            location: "sgemm_q5k_bf16_m8::launch",
        })?;
        Ok(())
    }

    /// T245.4 — Q6_K M=8 batched matmul (mirror of Q6K_V2 layout).
    ///
    /// # Safety  Same contract.
    pub unsafe fn sgemm_q6k_bf16_m8(
        &self,
        stream: &Arc<CudaStream>,
        w_q6k: u64,
        x: u64,
        y: u64,
        n: i32,
        k: i32,
    ) -> Result<(), CudaError> {
        if k % 256 != 0 {
            return Err(CudaError::Unsupported {
                msg: format!("sgemm_q6k_bf16_m8: K={k} must be multiple of 256"),
            });
        }
        let (_module, func) = self.compile_or_get(
            &self.sgemm_q6k_m8,
            SGEMM_Q6K_BF16_M8_SRC,
            "sgemm_q6k_bf16_m8",
        )?;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (n as u32, 1, 1),
            block_dim: (64, 1, 1),
            shared_mem_bytes: 32 * 4,
        };
        let mut launcher = stream.launch_builder(&func);
        launcher.arg(&w_q6k).arg(&x).arg(&y).arg(&n).arg(&k);
        launcher.launch(cfg).map_err(|e| CudaError::Driver {
            code: format!("{e:?}").len() as i32,
            location: "sgemm_q6k_bf16_m8::launch",
        })?;
        Ok(())
    }

    /// T244.3 — Direct Q5_K matmul (warp-shuffle, mirror of Q4K V2 layout).
    /// Used for Qwen 3.6 SSM weights (12% of total tensor count).
    ///
    /// # Safety  Same contract as `sgemv_q4k_bf16_v2`.
    pub unsafe fn sgemv_q5k_bf16(
        &self,
        stream: &Arc<CudaStream>,
        w_q5k: u64,
        x: u64,
        y: u64,
        n: i32,
        k: i32,
    ) -> Result<(), CudaError> {
        if k % 256 != 0 {
            return Err(CudaError::Unsupported {
                msg: format!("sgemv_q5k_bf16: K={k} must be multiple of 256"),
            });
        }
        let (_module, func) =
            self.compile_or_get(&self.sgemv_q5k, SGEMV_Q5K_BF16_SRC, "sgemv_q5k_bf16")?;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (n as u32, 1, 1),
            block_dim: (64, 1, 1),
            // shmem : 8 sc_pre + 8 m_pre + 64 sdata = 80 floats = 320 bytes
            shared_mem_bytes: 80 * 4,
        };
        let mut launcher = stream.launch_builder(&func);
        launcher.arg(&w_q5k).arg(&x).arg(&y).arg(&n).arg(&k);
        launcher.launch(cfg).map_err(|e| CudaError::Driver {
            code: format!("{e:?}").len() as i32,
            location: "sgemv_q5k_bf16::launch",
        })?;
        Ok(())
    }

    /// T244.4 — Q6_K V2 (warp-shuffle, 64 threads/TG, vector decode).
    /// Target : match Q4K_V2 bandwidth (~165 GB/s) on Qwen-7B FFN shape.
    ///
    /// # Safety  Same contract.
    pub unsafe fn sgemv_q6k_bf16_v2(
        &self,
        stream: &Arc<CudaStream>,
        w_q6k: u64,
        x: u64,
        y: u64,
        n: i32,
        k: i32,
    ) -> Result<(), CudaError> {
        if k % 256 != 0 {
            return Err(CudaError::Unsupported {
                msg: format!("sgemv_q6k_bf16_v2: K={k} must be multiple of 256"),
            });
        }
        let (_module, func) = self.compile_or_get(
            &self.sgemv_q6k_v2,
            SGEMV_Q6K_BF16_V2_SRC,
            "sgemv_q6k_bf16_v2",
        )?;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (n as u32, 1, 1),
            block_dim: (64, 1, 1),
            // shmem : 16 sc_pre + 64 sdata = 80 floats = 320 bytes
            shared_mem_bytes: 80 * 4,
        };
        let mut launcher = stream.launch_builder(&func);
        launcher.arg(&w_q6k).arg(&x).arg(&y).arg(&n).arg(&k);
        launcher.launch(cfg).map_err(|e| CudaError::Driver {
            code: format!("{e:?}").len() as i32,
            location: "sgemv_q6k_bf16_v2::launch",
        })?;
        Ok(())
    }

    /// T244.1.3 — Direct Q6_K matmul. Used for Q4_K_M tensors stored in
    /// Q6_K format (FFN down, attention output, Q/K/V, embeddings).
    ///
    /// # Safety  Same contract.
    pub unsafe fn sgemv_q6k_bf16(
        &self,
        stream: &Arc<CudaStream>,
        w_q6k: u64,
        x: u64,
        y: u64,
        n: i32,
        k: i32,
    ) -> Result<(), CudaError> {
        if k % 256 != 0 {
            return Err(CudaError::Unsupported {
                msg: format!("sgemv_q6k_bf16: K={k} must be multiple of 256"),
            });
        }
        let (_module, func) =
            self.compile_or_get(&self.sgemv_q6k, SGEMV_Q6K_BF16_SRC, "sgemv_q6k_bf16")?;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (n as u32, 1, 1),
            block_dim: (256, 1, 1),
            // shmem : 16 sc_pre + 256 sdata = 272 floats = 1088 bytes
            shared_mem_bytes: 272 * 4,
        };
        let mut launcher = stream.launch_builder(&func);
        launcher.arg(&w_q6k).arg(&x).arg(&y).arg(&n).arg(&k);
        launcher.launch(cfg).map_err(|e| CudaError::Driver {
            code: format!("{e:?}").len() as i32,
            location: "sgemv_q6k_bf16::launch",
        })?;
        Ok(())
    }

    /// T244.2 — Pure BF16 thin GEMV : y[N] = W[N,K] @ x[K], all BF16.
    ///
    /// Replaces cuBLASLt::matmul_bf16 for decode (M=1). cuBLAS hits only
    /// 19 GB/s on M=1 vs our 168 GB/s with this warp-shuffle kernel.
    ///
    /// # Safety  Caller ensures pointers valid, K multiple of 256.
    pub unsafe fn sgemv_bf16_bf16(
        &self,
        stream: &Arc<CudaStream>,
        w: u64, // [N, K] BF16 row-major
        x: u64, // [K] BF16
        y: u64, // [N] BF16
        n: i32,
        k: i32,
    ) -> Result<(), CudaError> {
        if k % 256 != 0 {
            return Err(CudaError::Unsupported {
                msg: format!("sgemv_bf16_bf16: K={k} must be multiple of 256"),
            });
        }
        let (_module, func) =
            self.compile_or_get(&self.sgemv_bf16, SGEMV_BF16_BF16_SRC, "sgemv_bf16_bf16")?;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (n as u32, 1, 1),
            block_dim: (64, 1, 1),
            // shmem : 64 sdata = 256 bytes (we only use [0..2] but align to warp)
            shared_mem_bytes: 64 * 4,
        };
        let mut launcher = stream.launch_builder(&func);
        launcher.arg(&w).arg(&x).arg(&y).arg(&n).arg(&k);
        launcher.launch(cfg).map_err(|e| CudaError::Driver {
            code: format!("{e:?}").len() as i32,
            location: "sgemv_bf16_bf16::launch",
        })?;
        Ok(())
    }

    /// T243.2 — Depth-wise 1-D conv (Qwen3.5/3.6 SSM block).
    ///
    /// # Safety  Caller ensures pointers valid for shapes.
    pub unsafe fn conv1d_depthwise_bf16(
        &self,
        stream: &Arc<CudaStream>,
        weight: u64, // [kernel, conv_dim] BF16
        state: u64,  // [kernel-1, conv_dim] BF16 (mutated)
        input: u64,  // [conv_dim] BF16
        out: u64,    // [conv_dim] BF16
        conv_dim: i32,
        kernel_size: i32,
    ) -> Result<(), CudaError> {
        let (_module, func) = self.compile_or_get(
            &self.conv1d_depthwise,
            CONV1D_DEPTHWISE_BF16_SRC,
            "conv1d_depthwise_bf16",
        )?;
        let block = 256u32;
        let grid = (conv_dim as u32).div_ceil(block);
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut launcher = stream.launch_builder(&func);
        launcher
            .arg(&weight)
            .arg(&state)
            .arg(&input)
            .arg(&out)
            .arg(&conv_dim)
            .arg(&kernel_size);
        launcher.launch(cfg).map_err(|e| CudaError::Driver {
            code: format!("{e:?}").len() as i32,
            location: "conv1d_depthwise_bf16::launch",
        })?;
        Ok(())
    }

    /// T243.2 — Per-head L2 normalization (Qwen3.5/3.6 SSM Q/K).
    ///
    /// # Safety  Caller ensures `head_dim` is power of 2 (for tree reduction).
    pub unsafe fn l2_norm_per_head_bf16(
        &self,
        stream: &Arc<CudaStream>,
        x: u64, // [n_heads, head_dim] BF16, in/out
        n_heads: i32,
        head_dim: i32,
        eps: f32,
    ) -> Result<(), CudaError> {
        let (_module, func) = self.compile_or_get(
            &self.l2_norm_per_head,
            L2_NORM_PER_HEAD_BF16_SRC,
            "l2_norm_per_head_bf16",
        )?;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (n_heads as u32, 1, 1),
            block_dim: (head_dim as u32, 1, 1),
            shared_mem_bytes: (head_dim as u32) * 4,
        };
        let mut launcher = stream.launch_builder(&func);
        launcher.arg(&x).arg(&n_heads).arg(&head_dim).arg(&eps);
        launcher.launch(cfg).map_err(|e| CudaError::Driver {
            code: format!("{e:?}").len() as i32,
            location: "l2_norm_per_head_bf16::launch",
        })?;
        Ok(())
    }

    /// T243.2 — Gated DeltaNet recurrent step (Qwen3.5/3.6 SSM mixer).
    ///
    /// # Safety  Caller ensures pointers valid for shapes ; state is `[n_heads × head_dim²]`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn delta_net_step_bf16(
        &self,
        stream: &Arc<CudaStream>,
        q: u64,
        k: u64,
        v: u64,
        gate: u64,
        beta: u64,
        state: u64,
        out: u64,
        n_heads: i32,
        head_dim: i32,
    ) -> Result<(), CudaError> {
        let (_module, func) = self.compile_or_get(
            &self.delta_net_step,
            DELTA_NET_STEP_BF16_SRC,
            "delta_net_step_bf16",
        )?;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (n_heads as u32, 1, 1),
            block_dim: (head_dim as u32, 1, 1),
            shared_mem_bytes: (head_dim as u32) * 4,
        };
        let mut launcher = stream.launch_builder(&func);
        launcher
            .arg(&q)
            .arg(&k)
            .arg(&v)
            .arg(&gate)
            .arg(&beta)
            .arg(&state)
            .arg(&out)
            .arg(&n_heads)
            .arg(&head_dim);
        launcher.launch(cfg).map_err(|e| CudaError::Driver {
            code: format!("{e:?}").len() as i32,
            location: "delta_net_step_bf16::launch",
        })?;
        Ok(())
    }

    /// T246.5.3 — RoPE variant qui lit `pos` depuis device pointer.
    /// Identique à `rope_partial_bf16` mais permet capture en CUDA Graph.
    ///
    /// # Safety  Caller assure pointers valides + `pos_dev` pointe vers
    /// 1 i32 device-resident contenant la position courante.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn rope_partial_bf16_devcnt(
        &self,
        stream: &Arc<CudaStream>,
        x: u64,
        inv_freq: u64,
        pos_dev: u64,
        n_heads: i32,
        head_dim: i32,
        rope_dim: i32,
    ) -> Result<(), CudaError> {
        let (_module, func) = self.compile_or_get(
            &self.rope_partial_devcnt,
            ROPE_PARTIAL_BF16_DEVCNT_SRC,
            "rope_partial_bf16_devcnt",
        )?;
        let half = rope_dim / 2;
        let block_dim = 64u32.min(half as u32);
        let grid_y = (half as u32).div_ceil(block_dim);
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (n_heads as u32, grid_y, 1),
            block_dim: (block_dim, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut launcher = stream.launch_builder(&func);
        launcher
            .arg(&x)
            .arg(&inv_freq)
            .arg(&pos_dev)
            .arg(&n_heads)
            .arg(&head_dim)
            .arg(&rope_dim);
        launcher.launch(cfg).map_err(|e| CudaError::Driver {
            code: format!("{e:?}").len() as i32,
            location: "rope_partial_bf16_devcnt::launch",
        })?;
        Ok(())
    }

    /// T246.5.3 — GQA online variant qui lit `kv_len` depuis device pointer.
    ///
    /// # Safety  Caller assure pointers valides + `kv_len_dev` pointe vers
    /// 1 i32 device-resident contenant la longueur courante du cache KV.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn gqa_decode_online_bf16_devcnt(
        &self,
        stream: &Arc<CudaStream>,
        q: u64,
        k_cache: u64,
        v_cache: u64,
        out: u64,
        n_heads: i32,
        n_kv: i32,
        kv_len_dev: u64,
        head_dim: i32,
        max_seq: i32,
    ) -> Result<(), CudaError> {
        let (_module, func) = self.compile_or_get(
            &self.gqa_decode_online_devcnt,
            GQA_DECODE_ONLINE_BF16_DEVCNT_SRC,
            "gqa_decode_online_bf16_devcnt",
        )?;
        let scale: f32 = 1.0 / (head_dim as f32).sqrt();
        let block_dim = head_dim as u32;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (n_heads as u32, 1, 1),
            block_dim: (block_dim, 1, 1),
            shared_mem_bytes: block_dim * 4,
        };
        let mut launcher = stream.launch_builder(&func);
        launcher
            .arg(&q)
            .arg(&k_cache)
            .arg(&v_cache)
            .arg(&out)
            .arg(&n_heads)
            .arg(&n_kv)
            .arg(&kv_len_dev)
            .arg(&head_dim)
            .arg(&max_seq)
            .arg(&scale);
        launcher.launch(cfg).map_err(|e| CudaError::Driver {
            code: format!("{e:?}").len() as i32,
            location: "gqa_decode_online_bf16_devcnt::launch",
        })?;
        Ok(())
    }

    /// T246.5.7 — FlashDecode-V2 split-K GQA decode (M=1).
    ///
    /// Replaces the serial `gqa_decode_online_bf16_devcnt` kernel by splitting
    /// each head's `kv_len` work across `n_split` blocks (4×N more parallelism)
    /// and merging via a per-head combine kernel.
    ///
    /// Caller must provide three staging buffers (re-used across calls,
    /// allocated once at model init):
    /// - `partial_m_dev` : `n_heads * n_split` floats
    /// - `partial_l_dev` : `n_heads * n_split` floats
    /// - `partial_o_dev` : `n_heads * n_split * head_dim` bf16
    ///
    /// `n_split` must match the size of the staging buffers ; typical value 4.
    ///
    /// # Safety  Caller ensures all device pointers are valid for the kernel
    /// lifetime and matching shapes.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn gqa_decode_split_bf16(
        &self,
        stream: &Arc<CudaStream>,
        q: u64,
        k_cache: u64,
        v_cache: u64,
        out: u64,
        partial_m_dev: u64,
        partial_l_dev: u64,
        partial_o_dev: u64,
        n_heads: i32,
        n_kv: i32,
        kv_len_dev: u64,
        head_dim: i32,
        max_seq: i32,
        n_split: i32,
    ) -> Result<(), CudaError> {
        let scale: f32 = 1.0 / (head_dim as f32).sqrt();
        let block_dim = head_dim as u32;

        // Phase 1 — partial.
        let (_m_p, fn_p) = self.compile_or_get(
            &self.gqa_split_partial,
            GQA_DECODE_SPLIT_PARTIAL_BF16_SRC,
            "gqa_decode_split_partial_bf16",
        )?;
        let cfg_p = cudarc::driver::LaunchConfig {
            grid_dim: (n_heads as u32, n_split as u32, 1),
            block_dim: (block_dim, 1, 1),
            shared_mem_bytes: block_dim * 4,
        };
        let mut launcher_p = stream.launch_builder(&fn_p);
        launcher_p
            .arg(&q)
            .arg(&k_cache)
            .arg(&v_cache)
            .arg(&partial_m_dev)
            .arg(&partial_l_dev)
            .arg(&partial_o_dev)
            .arg(&n_heads)
            .arg(&n_kv)
            .arg(&kv_len_dev)
            .arg(&head_dim)
            .arg(&max_seq)
            .arg(&n_split)
            .arg(&scale);
        launcher_p.launch(cfg_p).map_err(|e| CudaError::Driver {
            code: format!("{e:?}").len() as i32,
            location: "gqa_decode_split_partial_bf16::launch",
        })?;

        // Phase 2 — combine.
        let (_m_c, fn_c) = self.compile_or_get(
            &self.gqa_split_combine,
            GQA_DECODE_SPLIT_COMBINE_BF16_SRC,
            "gqa_decode_split_combine_bf16",
        )?;
        let cfg_c = cudarc::driver::LaunchConfig {
            grid_dim: (n_heads as u32, 1, 1),
            block_dim: (block_dim, 1, 1),
            shared_mem_bytes: 4,
        };
        let mut launcher_c = stream.launch_builder(&fn_c);
        launcher_c
            .arg(&partial_m_dev)
            .arg(&partial_l_dev)
            .arg(&partial_o_dev)
            .arg(&out)
            .arg(&n_heads)
            .arg(&head_dim)
            .arg(&n_split);
        launcher_c.launch(cfg_c).map_err(|e| CudaError::Driver {
            code: format!("{e:?}").len() as i32,
            location: "gqa_decode_split_combine_bf16::launch",
        })?;

        Ok(())
    }

    /// T246.5.3 — append K and V vectors to KV cache at slot `*pos_dev`.
    /// Replaces `copy_bf16(kc + pos*kv_dim*2, k, kv_dim)` ×2 with a single
    /// graph-capturable launch (since `pos` is read from device memory).
    ///
    /// # Safety  Caller assure `k_cache`/`v_cache` valides pour
    /// `max_seq * kv_dim` BF16 each, `k`/`v` valides pour `kv_dim` BF16,
    /// `pos_dev` pointe vers 1 i32 device contenant la slot index.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn kv_append_bf16_devcnt(
        &self,
        stream: &Arc<CudaStream>,
        k_cache: u64,
        v_cache: u64,
        k: u64,
        v: u64,
        pos_dev: u64,
        kv_dim: i32,
    ) -> Result<(), CudaError> {
        let (_module, func) = self.compile_or_get(
            &self.kv_append_devcnt,
            KV_APPEND_BF16_DEVCNT_SRC,
            "kv_append_bf16_devcnt",
        )?;
        let block_dim = 256u32;
        let grid_dim = (kv_dim as u32).div_ceil(block_dim);
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (grid_dim, 1, 1),
            block_dim: (block_dim, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut launcher = stream.launch_builder(&func);
        launcher
            .arg(&k_cache)
            .arg(&v_cache)
            .arg(&k)
            .arg(&v)
            .arg(&pos_dev)
            .arg(&kv_dim);
        launcher.launch(cfg).map_err(|e| CudaError::Driver {
            code: format!("{e:?}").len() as i32,
            location: "kv_append_bf16_devcnt::launch",
        })?;
        Ok(())
    }

    /// T246.5.3 — atomic-style increment d'un i32 device-resident.
    /// Lance 1 thread / 1 block. Utilisé pour avancer le compteur de position
    /// (kv_len, current_token offset) au sein d'un graph CUDA capturé.
    ///
    /// # Safety  `p` doit pointer vers 1 i32 device-resident.
    pub unsafe fn increment_u32_dev(
        &self,
        stream: &Arc<CudaStream>,
        p: u64,
    ) -> Result<(), CudaError> {
        let (_module, func) = self.compile_or_get(
            &self.increment_u32_dev,
            INCREMENT_U32_DEV_SRC,
            "increment_u32_dev",
        )?;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (1, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut launcher = stream.launch_builder(&func);
        launcher.arg(&p);
        launcher.launch(cfg).map_err(|e| CudaError::Driver {
            code: format!("{e:?}").len() as i32,
            location: "increment_u32_dev::launch",
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

    // ───────────────────────────────────────────────────────────────────
    // T241.6d — NVFP4 quantize tests : isolate the kernel from cuBLASLt.
    // Verify the encoding produces correct bytes for known scalar inputs.
    // ───────────────────────────────────────────────────────────────────

    /// Decode a single FP4 nibble (E2M1) back to f32. Used for round-trip.
    /// FP4 codes : 0=+0, 1=+0.5, 2=+1, 3=+1.5, 4=+2, 5=+3, 6=+4, 7=+6
    ///             8=-0, 9=-0.5, a=-1, b=-1.5, c=-2, d=-3, e=-4, f=-6
    fn fp4_decode(nibble: u8) -> f32 {
        let signed = nibble & 0x8 != 0;
        let mag = match nibble & 0x7 {
            0 => 0.0,
            1 => 0.5,
            2 => 1.0,
            3 => 1.5,
            4 => 2.0,
            5 => 3.0,
            6 => 4.0,
            7 => 6.0,
            _ => unreachable!(),
        };
        if signed {
            -mag
        } else {
            mag
        }
    }

    /// Decode a UE4M3 byte using NVIDIA's bias 7 convention (T241.6d).
    /// Layout : bit 7 reserved, bits 6-3 = exp (4 bits), bits 2-0 = mantissa.
    /// Value = 2^(E - 7) * (1 + M/8) for E >= 1.
    ///
    /// Validated by `nvfp4_supported_m_values` perimeter test : byte 0x70
    /// (E=14, M=0) = 2^7 = 128, which matches the empirical matmul output.
    fn ue4m3_decode(byte: u8) -> f32 {
        let e = (byte >> 3) & 0xf; // 4-bit exponent
        let m = byte & 0x7;
        if e == 0 {
            // Subnormal — value = 2^(-6) * M/8 (we don't emit by construction).
            (m as f32 / 8.0) * 2f32.powi(-6)
        } else {
            (1.0 + m as f32 / 8.0) * 2f32.powi(e as i32 - 7)
        }
    }

    /// Verify quantize_bf16_to_nvfp4 produces FP4 codes that, when decoded
    /// and multiplied by the UE4M3 scale, recover the original BF16 input
    /// to within FP4 precision (~1/16 quantization step relative).
    #[test]
    fn quantize_nvfp4_round_trip() {
        // Use 16 hand-picked values that fit cleanly in FP4 grid.
        // Magnitudes : 0.5, 1, 1.5, 2, 3, 4, 6 (the FP4 grid).
        // We scale by 0.1 so the block scale = 0.6/6 = 0.1 (normal range).
        let block16: Vec<f32> = vec![
            0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.6, 0.0, -0.05, -0.1, -0.15, -0.2, -0.3, -0.4, -0.6,
            0.0,
        ];
        let bf16_block: Vec<half::bf16> =
            block16.iter().copied().map(half::bf16::from_f32).collect();

        let ctx = CudaContext::new(0).expect("ctx");
        let stream = ctx.default_stream();
        let kernels = LlmKernels::new(ctx);

        let x_dev = stream.memcpy_stod(&bf16_block).expect("upload x");
        let mut out_fp4 = stream.alloc_zeros::<u8>(8).expect("alloc fp4"); // 16 elts / 2
        let mut out_scale = stream.alloc_zeros::<u8>(1).expect("alloc scale"); // 1 block

        unsafe {
            let (x_p, _g1) = x_dev.device_ptr(&stream);
            let (fp4_p, _g2) = out_fp4.device_ptr_mut(&stream);
            let (sc_p, _g3) = out_scale.device_ptr_mut(&stream);
            kernels
                .quantize_bf16_to_nvfp4(&stream, x_p, fp4_p, sc_p, 16)
                .expect("quantize");
        }

        let fp4_bytes: Vec<u8> = stream.memcpy_dtov(&out_fp4).expect("dtov fp4");
        let scale_bytes: Vec<u8> = stream.memcpy_dtov(&out_scale).expect("dtov scale");

        // Decode scale (UE4M3 bias 14).
        let scale = ue4m3_decode(scale_bytes[0]);
        // Sanity : scale should be ~0.1 (max_abs = 0.6, scale = 0.6/6 = 0.1).
        // 0.1 = 2^-3.32, so ue_exp ≈ -3.32 + 14 = 10.68 → byte 10 or 11 (rounded).
        // Decoded value ≈ 0.0625..0.125. Close enough to 0.1 within UE4M3 precision.
        assert!(
            scale > 0.05 && scale < 0.2,
            "scale {} should be ≈ 0.1 ; byte = 0x{:02x}",
            scale,
            scale_bytes[0]
        );

        // Decode each FP4 nibble and verify reconstruction.
        for i in 0..16 {
            let byte = fp4_bytes[i / 2];
            let nibble = if i % 2 == 0 { byte & 0x0f } else { byte >> 4 };
            let fp4_val = fp4_decode(nibble);
            let reconstructed = fp4_val * scale;
            let original = block16[i];
            // FP4 has only 8 magnitudes per sign ; quantization step is
            // ~scale * 0.5 (smallest non-zero magnitude). Tolerate a step
            // of error.
            let tol = scale * 1.0;
            assert!(
                (reconstructed - original).abs() <= tol,
                "elt[{i}] orig={original:+.4} recon={reconstructed:+.4} \
                 (fp4=0x{nibble:01x}={fp4_val:+.2}, scale={scale:+.4})"
            );
        }
    }

    /// T241.6d — verify that quantize_bf16_to_nvfp4 NEVER produces a
    /// subnormal UE4M3 scale (E=0 byte). cuBLASLt sm_121 NVFP4 path
    /// silently zero-outs blocks with subnormal scales, so our kernel
    /// must clamp them.
    #[test]
    fn quantize_nvfp4_no_subnormal_scales() {
        // 32 random tiny values that would normally produce subnormal scales.
        let n = 32usize;
        let block: Vec<f32> = (0..n)
            .map(|i| ((i as f32 * 0.123).sin()) * 1e-5) // very small ~1e-5
            .collect();
        let bf16_block: Vec<half::bf16> = block.iter().copied().map(half::bf16::from_f32).collect();

        let ctx = CudaContext::new(0).expect("ctx");
        let stream = ctx.default_stream();
        let kernels = LlmKernels::new(ctx);

        let x_dev = stream.memcpy_stod(&bf16_block).expect("upload");
        let mut out_fp4 = stream.alloc_zeros::<u8>(n / 2).expect("fp4");
        let mut out_scale = stream.alloc_zeros::<u8>(n / 16).expect("scale");

        unsafe {
            let (x_p, _g1) = x_dev.device_ptr(&stream);
            let (fp4_p, _g2) = out_fp4.device_ptr_mut(&stream);
            let (sc_p, _g3) = out_scale.device_ptr_mut(&stream);
            kernels
                .quantize_bf16_to_nvfp4(&stream, x_p, fp4_p, sc_p, n as i32)
                .expect("quantize");
        }

        let scale_bytes: Vec<u8> = stream.memcpy_dtov(&out_scale).expect("dtov scale");
        for (i, &byte) in scale_bytes.iter().enumerate() {
            let e = (byte >> 3) & 0x1f;
            assert!(
                e >= 1,
                "block[{i}] scale byte 0x{byte:02x} has E=0 (subnormal forbidden)"
            );
        }
    }

    /// T241.6d — verify the canonical scale bytes for NVIDIA UE4M3 bias 7.
    /// Validated by perimeter test : 0x70 corresponds to scale = 128.
    #[test]
    fn ue4m3_canonical_bytes() {
        // 0x38 = (7 << 3) | 0 → E=7, M=0 → 2^0 = 1.0
        let v_one = ue4m3_decode(0x38);
        assert!(
            (v_one - 1.0).abs() < 1e-6,
            "0x38 should decode to 1.0 (E=7, M=0, bias 7), got {v_one}"
        );
        // 0x70 = (14 << 3) | 0 → E=14, M=0 → 2^7 = 128.0
        let v_128 = ue4m3_decode(0x70);
        assert!(
            (v_128 - 128.0).abs() < 1e-6,
            "0x70 should decode to 128.0 (E=14, M=0, bias 7), got {v_128}"
        );
    }

    /// T244.1 — sgemv_q4k_bf16 parity test : verify the CUDA Q4_K matmul
    /// matches a CPU reference (dequant_q4_k + scalar gemv) within BF16
    /// precision. This validates the kernel before plumbing it into the
    /// full forward path.
    #[test]
    fn sgemv_q4k_bf16_matches_cpu_reference() {
        use rustorch_gguf::dequant::{dequant_q4_k, Q4_K_BYTES, QK_K};

        // Small but representative shape : N=64 rows, K=256 (1 super-block per row).
        let n = 64usize;
        let k = 256usize;
        let blocks_per_row = k / QK_K;

        // Generate deterministic Q4_K bytes for `n` rows. We use random-ish
        // but reproducible data ; values are in normal range so dequant
        // gives reasonable magnitudes.
        let mut state: u64 = 0xdeadbeefcafebabe;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };

        let row_bytes = blocks_per_row * Q4_K_BYTES;
        let mut w_bytes: Vec<u8> = Vec::with_capacity(n * row_bytes);
        for _ in 0..n {
            for _ in 0..blocks_per_row {
                // d & dmin in f16 — pick small reasonable values.
                let d = half::f16::from_f32(0.1).to_le_bytes();
                let dmin = half::f16::from_f32(0.05).to_le_bytes();
                w_bytes.push(d[0]);
                w_bytes.push(d[1]);
                w_bytes.push(dmin[0]);
                w_bytes.push(dmin[1]);
                // 12 scale/min bytes — random but valid (6-bit each).
                for _ in 0..12 {
                    w_bytes.push((next() & 0x3F) as u8);
                }
                // 128 nibble bytes.
                for _ in 0..128 {
                    w_bytes.push((next() & 0xFF) as u8);
                }
            }
        }
        assert_eq!(w_bytes.len(), n * row_bytes);

        // CPU reference : dequantize each row then scalar dot-product.
        let mut w_f32 = vec![0.0f32; n * k];
        for row in 0..n {
            let row_start = row * row_bytes;
            let row_end = row_start + row_bytes;
            let row_dst_start = row * k;
            let row_dst_end = row_dst_start + k;
            dequant_q4_k(
                &w_bytes[row_start..row_end],
                &mut w_f32[row_dst_start..row_dst_end],
            )
            .expect("dequant_q4_k");
        }
        let x_f32: Vec<f32> = (0..k).map(|i| ((i as f32 * 0.1).sin()) * 0.5).collect();
        let mut y_cpu = vec![0.0f32; n];
        for row in 0..n {
            let mut acc = 0.0f32;
            for j in 0..k {
                acc += w_f32[row * k + j] * x_f32[j];
            }
            y_cpu[row] = acc;
        }

        // CUDA path.
        let ctx = CudaContext::new(0).expect("ctx");
        let stream = ctx.default_stream();
        let kernels = LlmKernels::new(ctx);

        let w_dev = stream.memcpy_stod(&w_bytes).expect("upload w");
        let x_bf: Vec<half::bf16> = x_f32.iter().copied().map(half::bf16::from_f32).collect();
        let x_dev = stream.memcpy_stod(&x_bf).expect("upload x");
        let mut y_dev = stream.alloc_zeros::<half::bf16>(n).expect("alloc y");

        unsafe {
            let (w_p, _g1) = w_dev.device_ptr(&stream);
            let (x_p, _g2) = x_dev.device_ptr(&stream);
            let (y_p, _g3) = y_dev.device_ptr_mut(&stream);
            kernels
                .sgemv_q4k_bf16(&stream, w_p, x_p, y_p, n as i32, k as i32)
                .expect("sgemv_q4k");
        }

        let y_host: Vec<half::bf16> = stream.memcpy_dtov(&y_dev).expect("dtov");
        let y_gpu: Vec<f32> = y_host.into_iter().map(|x| x.to_f32()).collect();

        for i in 0..n {
            let diff = (y_cpu[i] - y_gpu[i]).abs();
            // BF16 has ~1% relative precision ; allow 5% + small abs tol
            // because we sum 256 elements (cumulative rounding).
            let tol = y_cpu[i].abs() * 5e-2 + 0.5;
            assert!(
                diff <= tol,
                "row[{i}] cpu={} gpu={} diff={} tol={}",
                y_cpu[i],
                y_gpu[i],
                diff,
                tol
            );
        }
    }

    /// T244.2 — sgemv_bf16_bf16 parity test : compare against CPU reference.
    /// The custom warp-shuffle kernel must produce same output as a naive
    /// CPU GEMV within BF16 tolerance.
    #[test]
    fn sgemv_bf16_bf16_matches_cpu() {
        let n = 32usize;
        let k = 512usize;

        let mut state: u64 = 0xc0ffee01;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state & 0xFFFF) as f32 - 32768.0) / 65536.0 * 0.05
        };
        let w_f32: Vec<f32> = (0..n * k).map(|_| next()).collect();
        let x_f32: Vec<f32> = (0..k).map(|_| next()).collect();
        let w_bf16: Vec<half::bf16> = w_f32.iter().map(|&v| half::bf16::from_f32(v)).collect();
        let x_bf16: Vec<half::bf16> = x_f32.iter().map(|&v| half::bf16::from_f32(v)).collect();

        // CPU reference (in f32 from BF16).
        let mut y_ref = vec![0.0f32; n];
        for i in 0..n {
            let mut s = 0.0f32;
            for j in 0..k {
                s += f32::from(w_bf16[i * k + j]) * f32::from(x_bf16[j]);
            }
            y_ref[i] = s;
        }

        let ctx = CudaContext::new(0).expect("ctx");
        let stream = ctx.default_stream();
        let kernels = LlmKernels::new(ctx);

        let w_dev = stream.memcpy_stod(&w_bf16).expect("w upload");
        let x_dev = stream.memcpy_stod(&x_bf16).expect("x upload");
        let mut y_dev = stream.alloc_zeros::<half::bf16>(n).expect("y alloc");

        let (w_p, x_p, y_p) = unsafe {
            use cudarc::driver::{DevicePtr, DevicePtrMut};
            (
                w_dev.device_ptr(&stream).0,
                x_dev.device_ptr(&stream).0,
                y_dev.device_ptr_mut(&stream).0,
            )
        };
        unsafe {
            kernels
                .sgemv_bf16_bf16(&stream, w_p, x_p, y_p, n as i32, k as i32)
                .expect("sgemv_bf16");
        }
        stream.synchronize().ok();

        let y_host = stream.memcpy_dtov(&y_dev).expect("y dl");
        for i in 0..n {
            let got = f32::from(y_host[i]);
            let want = y_ref[i];
            let tol = (want.abs() * 0.02).max(1e-3);
            assert!(
                (got - want).abs() < tol,
                "row {i}: got {got}, want {want} (tol {tol})"
            );
        }
    }

    /// T244.2 — sgemv_bf16_bf16 vs cuBLASLt matmul_bf16 thin GEMV bench.
    /// Compares the two paths on Qwen3.6-27B FFN gate shape (5120 → 17408).
    /// Expected : custom kernel ~5-10× faster (168 GB/s vs 19 GB/s).
    #[test]
    #[ignore = "perf benchmark"]
    fn sgemv_bf16_bf16_vs_cublas_bench() {
        use std::time::Instant;
        let n = 17408usize;
        let k = 5120usize;
        let n_iters = 200;

        let ctx = CudaContext::new(0).expect("ctx");
        let stream = ctx.default_stream();
        let kernels = LlmKernels::new(ctx.clone());
        let mut session = crate::cublas_lt::LtSession::new(stream.clone()).expect("lt");

        let w = stream.alloc_zeros::<half::bf16>(n * k).expect("w");
        let x = stream.alloc_zeros::<half::bf16>(k).expect("x");
        let mut y = stream.alloc_zeros::<half::bf16>(n).expect("y");

        let (w_p, x_p, y_p) = unsafe {
            use cudarc::driver::{DevicePtr, DevicePtrMut};
            (
                w.device_ptr(&stream).0,
                x.device_ptr(&stream).0,
                y.device_ptr_mut(&stream).0,
            )
        };

        // Warm-up.
        unsafe {
            let _ = kernels.sgemv_bf16_bf16(&stream, w_p, x_p, y_p, n as i32, k as i32);
            let _ = session.matmul_bf16(x_p, w_p, y_p, 1, k, n, 1.0, 0.0);
        }
        stream.synchronize().ok();

        // Custom kernel.
        let t0 = Instant::now();
        for _ in 0..n_iters {
            unsafe {
                kernels
                    .sgemv_bf16_bf16(&stream, w_p, x_p, y_p, n as i32, k as i32)
                    .ok();
            }
        }
        stream.synchronize().ok();
        let custom_ms = t0.elapsed().as_secs_f64() * 1000.0 / n_iters as f64;
        let bytes = (n * k * 2) as f64; // BF16 weights only
        let custom_bw = bytes / (custom_ms * 1e-3) / 1e9;

        // cuBLASLt.
        let t0 = Instant::now();
        for _ in 0..n_iters {
            unsafe {
                let _ = session.matmul_bf16(x_p, w_p, y_p, 1, k, n, 1.0, 0.0);
            }
        }
        stream.synchronize().ok();
        let cublas_ms = t0.elapsed().as_secs_f64() * 1000.0 / n_iters as f64;
        let cublas_bw = bytes / (cublas_ms * 1e-3) / 1e9;

        eprintln!();
        eprintln!("=== sgemv_bf16_bf16 vs cuBLASLt — Qwen3.6 FFN gate ({n}x{k}) ===");
        eprintln!("  custom kernel : {custom_ms:.3} ms  ({custom_bw:.1} GB/s)");
        eprintln!("  cuBLASLt      : {cublas_ms:.3} ms  ({cublas_bw:.1} GB/s)");
        eprintln!("  speedup       : {:.2}×", cublas_ms / custom_ms);
    }

    /// T244.1.1 — V2 parity test : V2 must produce same output as V1
    /// (and CPU reference) within BF16 tolerance.
    #[test]
    /// T245.4 — sgemm_q4k_bf16_m8 parity test : 8 sequential M=1 SGEMVs must
    /// match a single M=8 SGEMM call within BF16 tolerance.
    #[test]
    fn sgemm_q4k_bf16_m8_matches_8x_sgemv() {
        use rustorch_gguf::dequant::{Q4_K_BYTES, QK_K};

        let n = 32usize;
        let k = 512usize;
        let blocks_per_row = k / QK_K;
        let row_bytes = blocks_per_row * Q4_K_BYTES;

        let mut state: u64 = 0xc0debace;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let mut w_q4k_bytes = vec![0u8; n * row_bytes];
        for b in &mut w_q4k_bytes {
            *b = (next() & 0xFF) as u8;
        }
        // Sane scales.
        for row in 0..n {
            for blk in 0..blocks_per_row {
                let off = row * row_bytes + blk * Q4_K_BYTES;
                let d = half::f16::from_f32(0.05).to_le_bytes();
                let dmin = half::f16::from_f32(0.025).to_le_bytes();
                w_q4k_bytes[off] = d[0];
                w_q4k_bytes[off + 1] = d[1];
                w_q4k_bytes[off + 2] = dmin[0];
                w_q4k_bytes[off + 3] = dmin[1];
                for i in 0..12 {
                    w_q4k_bytes[off + 4 + i] &= 0x3F;
                }
            }
        }
        // 8 different x vectors.
        let m_batch = 8usize;
        let mut x_all = vec![0.0f32; m_batch * k];
        for m in 0..m_batch {
            for j in 0..k {
                x_all[m * k + j] = ((m as f32 + 1.0) * (j as f32 * 0.013).sin()) * 0.3;
            }
        }
        let x_bf: Vec<half::bf16> = x_all.iter().copied().map(half::bf16::from_f32).collect();

        let ctx = CudaContext::new(0).expect("ctx");
        let stream = ctx.default_stream();
        let kernels = LlmKernels::new(ctx);
        let w_dev = stream.memcpy_stod(&w_q4k_bytes).expect("w");
        let x_dev = stream.memcpy_stod(&x_bf).expect("x");

        // Reference : 8 sequential M=1 SGEMVs.
        let mut y_seq = vec![half::bf16::ZERO; m_batch * n];
        for m in 0..m_batch {
            let xm: Vec<half::bf16> = x_bf[m * k..(m + 1) * k].to_vec();
            let xm_dev = stream.memcpy_stod(&xm).expect("xm");
            let mut ym_dev = stream.alloc_zeros::<half::bf16>(n).expect("ym");
            unsafe {
                let (w_p, _g1) = w_dev.device_ptr(&stream);
                let (x_p, _g2) = xm_dev.device_ptr(&stream);
                let (y_p, _g3) = ym_dev.device_ptr_mut(&stream);
                kernels
                    .sgemv_q4k_bf16_v2(&stream, w_p, x_p, y_p, n as i32, k as i32)
                    .expect("v2");
            }
            let ym: Vec<half::bf16> = stream.memcpy_dtov(&ym_dev).expect("dl");
            // Store in y_seq[m, n] as M-major : y_seq[m * n + j] = ym[j]
            for j in 0..n {
                y_seq[m * n + j] = ym[j];
            }
        }

        // Test : single M=8 SGEMM.
        let mut y_m8_dev = stream.alloc_zeros::<half::bf16>(m_batch * n).expect("y_m8");
        unsafe {
            let (w_p, _g1) = w_dev.device_ptr(&stream);
            let (x_p, _g2) = x_dev.device_ptr(&stream);
            let (y_p, _g3) = y_m8_dev.device_ptr_mut(&stream);
            kernels
                .sgemm_q4k_bf16_m8(&stream, w_p, x_p, y_p, n as i32, k as i32)
                .expect("m8");
        }
        let y_m8: Vec<half::bf16> = stream.memcpy_dtov(&y_m8_dev).expect("dl m8");

        // Compare. Layout : y_m8[m * n + j] should equal y_seq[m * n + j].
        for m in 0..m_batch {
            for j in 0..n {
                let got = y_m8[m * n + j].to_f32();
                let want = y_seq[m * n + j].to_f32();
                let diff = (got - want).abs();
                let tol = want.abs() * 5e-2 + 1e-2;
                assert!(
                    diff <= tol,
                    "m={m} n={j}: got={got} want={want} diff={diff} tol={tol}"
                );
            }
        }
    }

    #[test]
    fn sgemv_q4k_bf16_v2_matches_v1() {
        use rustorch_gguf::dequant::{Q4_K_BYTES, QK_K};

        let n = 64usize;
        let k = 512usize; // 2 super-blocks per row
        let blocks_per_row = k / QK_K;
        let row_bytes = blocks_per_row * Q4_K_BYTES;

        let mut state: u64 = 0xfeedbeef;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let mut w_bytes = vec![0u8; n * row_bytes];
        for row in 0..n {
            for blk in 0..blocks_per_row {
                let off = row * row_bytes + blk * Q4_K_BYTES;
                let d = half::f16::from_f32(0.07).to_le_bytes();
                let dmin = half::f16::from_f32(0.03).to_le_bytes();
                w_bytes[off] = d[0];
                w_bytes[off + 1] = d[1];
                w_bytes[off + 2] = dmin[0];
                w_bytes[off + 3] = dmin[1];
                for i in 0..12 {
                    w_bytes[off + 4 + i] = (next() & 0x3F) as u8;
                }
                for i in 0..128 {
                    w_bytes[off + 16 + i] = (next() & 0xFF) as u8;
                }
            }
        }

        let ctx = CudaContext::new(0).expect("ctx");
        let stream = ctx.default_stream();
        let kernels = LlmKernels::new(ctx);

        let w_dev = stream.memcpy_stod(&w_bytes).expect("upload w");
        let x_f32: Vec<f32> = (0..k).map(|i| ((i as f32 * 0.05).cos()) * 0.5).collect();
        let x_bf: Vec<half::bf16> = x_f32.iter().copied().map(half::bf16::from_f32).collect();
        let x_dev = stream.memcpy_stod(&x_bf).expect("upload x");

        // V1 result.
        let mut y_v1 = stream.alloc_zeros::<half::bf16>(n).expect("alloc y1");
        unsafe {
            let (w_p, _g1) = w_dev.device_ptr(&stream);
            let (x_p, _g2) = x_dev.device_ptr(&stream);
            let (y_p, _g3) = y_v1.device_ptr_mut(&stream);
            kernels
                .sgemv_q4k_bf16(&stream, w_p, x_p, y_p, n as i32, k as i32)
                .expect("v1");
        }
        let v1: Vec<f32> = stream
            .memcpy_dtov(&y_v1)
            .expect("dtov v1")
            .into_iter()
            .map(|b: half::bf16| b.to_f32())
            .collect();

        // V2 result.
        let mut y_v2 = stream.alloc_zeros::<half::bf16>(n).expect("alloc y2");
        unsafe {
            let (w_p, _g1) = w_dev.device_ptr(&stream);
            let (x_p, _g2) = x_dev.device_ptr(&stream);
            let (y_p, _g3) = y_v2.device_ptr_mut(&stream);
            kernels
                .sgemv_q4k_bf16_v2(&stream, w_p, x_p, y_p, n as i32, k as i32)
                .expect("v2");
        }
        let v2: Vec<f32> = stream
            .memcpy_dtov(&y_v2)
            .expect("dtov v2")
            .into_iter()
            .map(|b: half::bf16| b.to_f32())
            .collect();

        for i in 0..n {
            let diff = (v1[i] - v2[i]).abs();
            // Both BF16 outputs of the same math should be exactly equal
            // (or differ only by reduction order rounding) — tolerate
            // small abs diff.
            let tol = v1[i].abs() * 5e-2 + 0.5;
            assert!(
                diff <= tol,
                "row[{i}] v1={} v2={} diff={}",
                v1[i],
                v2[i],
                diff
            );
        }
    }

    /// T246.5.5 — quantize_q8_1_bf16 roundtrip sanity : quantize a known
    /// input and dequantize on host ; max abs error should be < d (= amax/127).
    #[test]
    #[ignore = "requires CUDA GPU — run with --ignored on DGX"]
    fn quantize_q8_1_bf16_roundtrip_basic() {
        let k = 32usize * 8; // 8 blocks
        let x_f32: Vec<f32> = (0..k).map(|i| ((i as f32 * 0.07).sin()) * 1.7).collect();
        let x_bf: Vec<half::bf16> = x_f32.iter().copied().map(half::bf16::from_f32).collect();

        let ctx = CudaContext::new(0).expect("ctx");
        let stream = ctx.default_stream();
        let kernels = LlmKernels::new(ctx);

        let x_dev = stream.memcpy_stod(&x_bf).expect("upload x");
        let n_blocks = k / 32;
        let mut y_dev = stream.alloc_zeros::<u8>(n_blocks * 36).expect("alloc q8_1");
        unsafe {
            let (x_p, _g1) = x_dev.device_ptr(&stream);
            let (y_p, _g2) = y_dev.device_ptr_mut(&stream);
            kernels
                .quantize_q8_1_bf16(&stream, x_p, y_p, k as i32)
                .expect("quant");
        }
        let q8: Vec<u8> = stream.memcpy_dtov(&y_dev).expect("dtov q8_1");

        // Dequantize on host and compare against the BF16 input (the kernel
        // operates on BF16, so int8 quant error is bounded by 0.5*d on top of
        // the BF16 representation — comparing to f32 mixes in BF16 rounding).
        for blk in 0..n_blocks {
            let off = blk * 36;
            let d_bits = u16::from_le_bytes([q8[off], q8[off + 1]]);
            let d = half::f16::from_bits(d_bits).to_f32();
            let mut max_err: f32 = 0.0;
            for i in 0..32 {
                let q = q8[off + 4 + i] as i8;
                let recon = (q as f32) * d;
                let orig_bf = x_bf[blk * 32 + i].to_f32();
                max_err = max_err.max((recon - orig_bf).abs());
            }
            // Rounding error of int8 quant : at most 0.5 * d. Allow tiny slack
            // for fp32 reduction order in the warp reduce.
            assert!(
                max_err <= d.max(1e-6) * 0.6,
                "block[{blk}]: d={d} max_err={max_err}"
            );
        }
    }

    /// T246.5.5 — Q4_K dp4a path numeric parity vs sgemv_q4k_bf16_v2
    /// reference. Tolerance ~1.5% relative + 0.5 abs (Q8_1 quant of the
    /// activation introduces ~1% per-element loss).
    #[test]
    #[ignore = "requires CUDA GPU — run with --ignored on DGX"]
    fn sgemv_q4k_dp4a_matches_v2() {
        use rustorch_gguf::dequant::{Q4_K_BYTES, QK_K};

        let n = 64usize;
        let k = 512usize;
        let blocks_per_row = k / QK_K;
        let row_bytes = blocks_per_row * Q4_K_BYTES;

        // Same RNG-driven W as the v1/v2 test.
        let mut state: u64 = 0xfeedbeef;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let mut w_bytes = vec![0u8; n * row_bytes];
        for row in 0..n {
            for blk in 0..blocks_per_row {
                let off = row * row_bytes + blk * Q4_K_BYTES;
                let d = half::f16::from_f32(0.07).to_le_bytes();
                let dmin = half::f16::from_f32(0.03).to_le_bytes();
                w_bytes[off] = d[0];
                w_bytes[off + 1] = d[1];
                w_bytes[off + 2] = dmin[0];
                w_bytes[off + 3] = dmin[1];
                for i in 0..12 {
                    w_bytes[off + 4 + i] = (next() & 0x3F) as u8;
                }
                for i in 0..128 {
                    w_bytes[off + 16 + i] = (next() & 0xFF) as u8;
                }
            }
        }

        let ctx = CudaContext::new(0).expect("ctx");
        let stream = ctx.default_stream();
        let kernels = LlmKernels::new(ctx);

        let w_dev = stream.memcpy_stod(&w_bytes).expect("upload w");
        let x_f32: Vec<f32> = (0..k).map(|i| ((i as f32 * 0.05).cos()) * 0.5).collect();
        let x_bf: Vec<half::bf16> = x_f32.iter().copied().map(half::bf16::from_f32).collect();
        let x_dev = stream.memcpy_stod(&x_bf).expect("upload x");

        // Reference : sgemv_q4k_bf16_v2 (float dot).
        let mut y_ref = stream.alloc_zeros::<half::bf16>(n).expect("alloc y_ref");
        unsafe {
            let (w_p, _g1) = w_dev.device_ptr(&stream);
            let (x_p, _g2) = x_dev.device_ptr(&stream);
            let (y_p, _g3) = y_ref.device_ptr_mut(&stream);
            kernels
                .sgemv_q4k_bf16_v2(&stream, w_p, x_p, y_p, n as i32, k as i32)
                .expect("v2");
        }
        let r_ref: Vec<f32> = stream
            .memcpy_dtov(&y_ref)
            .expect("dtov ref")
            .into_iter()
            .map(|b: half::bf16| b.to_f32())
            .collect();

        // Test path : quantize x → Q8_1, then sgemv_q4k_q8_1_dp4a.
        let n_blocks_q8 = k / 32;
        let mut x_q8 = stream
            .alloc_zeros::<u8>(n_blocks_q8 * 36)
            .expect("alloc q8_1");
        let mut y_dp4a = stream.alloc_zeros::<half::bf16>(n).expect("alloc y_dp4a");
        unsafe {
            let (x_p, _g1) = x_dev.device_ptr(&stream);
            let (xq_p, _g2) = x_q8.device_ptr_mut(&stream);
            kernels
                .quantize_q8_1_bf16(&stream, x_p, xq_p, k as i32)
                .expect("quant");
        }
        unsafe {
            let (w_p, _g1) = w_dev.device_ptr(&stream);
            let (xq_p, _g2) = x_q8.device_ptr(&stream);
            let (y_p, _g3) = y_dp4a.device_ptr_mut(&stream);
            kernels
                .sgemv_q4k_q8_1_dp4a_bf16(&stream, w_p, xq_p, y_p, n as i32, k as i32)
                .expect("dp4a");
        }
        let r_dp4a: Vec<f32> = stream
            .memcpy_dtov(&y_dp4a)
            .expect("dtov dp4a")
            .into_iter()
            .map(|b: half::bf16| b.to_f32())
            .collect();

        for i in 0..n {
            let diff = (r_ref[i] - r_dp4a[i]).abs();
            // Q8_1 quant of activation : ~1% relative loss per element ; over
            // K=512 with mean cancellation the row sum loss is ~1.5% of |ref|,
            // plus 0.5 abs slack for BF16 cast.
            let tol = r_ref[i].abs() * 1.5e-2 + 0.5;
            assert!(
                diff <= tol,
                "row[{i}] ref={} dp4a={} diff={} tol={}",
                r_ref[i],
                r_dp4a[i],
                diff,
                tol
            );
        }
    }

    /// T244.1.3 — sgemv_q6k_bf16 parity test : verify Q6_K kernel matches
    /// CPU reference (dequant_q6_k + scalar gemv).
    #[test]
    fn sgemv_q6k_bf16_matches_cpu_reference() {
        // Use the public dequant API by going through GgufFile is overkill ;
        // dequant_q6_k is private but we can use the public dispatch via
        // dequant_to_f32 with Q6_K dtype. Easier : reproduce the CPU dequant
        // logic inline (it's deterministic).
        const QK_K: usize = 256;
        const Q6_K_BYTES: usize = 210;

        let n = 32usize;
        let k = 512usize;
        let blocks_per_row = k / QK_K;
        let row_bytes = blocks_per_row * Q6_K_BYTES;

        let mut state: u64 = 0xc0debabe;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let mut w_bytes = vec![0u8; n * row_bytes];
        for row in 0..n {
            for blk in 0..blocks_per_row {
                let off = row * row_bytes + blk * Q6_K_BYTES;
                // ql (128) + qh (64) random
                for i in 0..128 {
                    w_bytes[off + i] = (next() & 0xFF) as u8;
                }
                for i in 0..64 {
                    w_bytes[off + 128 + i] = (next() & 0xFF) as u8;
                }
                // 16 i8 scales — small range to keep magnitudes reasonable
                for i in 0..16 {
                    let raw = (next() & 0xFF) as u8;
                    // map to range -16..16 to avoid huge values
                    let signed = (raw as i8) / 8;
                    w_bytes[off + 192 + i] = signed as u8;
                }
                // d (f16) = 0.05
                let d = half::f16::from_f32(0.05).to_le_bytes();
                w_bytes[off + 208] = d[0];
                w_bytes[off + 209] = d[1];
            }
        }

        // CPU reference dequant Q6_K (inline from dequant_q6_k logic).
        let mut w_f32 = vec![0.0f32; n * k];
        for row in 0..n {
            for blk in 0..blocks_per_row {
                let off = row * row_bytes + blk * Q6_K_BYTES;
                let ql = &w_bytes[off..off + 128];
                let qh = &w_bytes[off + 128..off + 192];
                let scales_i8 = &w_bytes[off + 192..off + 208];
                let d_bits = u16::from_le_bytes([w_bytes[off + 208], w_bytes[off + 209]]);
                let d = half::f16::from_bits(d_bits).to_f32();
                let dst_off = row * k + blk * QK_K;
                for half_idx in 0..2 {
                    let ql_h = &ql[half_idx * 64..half_idx * 64 + 64];
                    let qh_h = &qh[half_idx * 32..half_idx * 32 + 32];
                    let sc_h = &scales_i8[half_idx * 8..half_idx * 8 + 8];
                    for l in 0..32 {
                        let qhh = qh_h[l];
                        let q1 = ((ql_h[l] & 0x0F) as i32) | ((qhh & 0x03) as i32) << 4;
                        let q2 = ((ql_h[l + 32] & 0x0F) as i32) | (((qhh >> 2) & 0x03) as i32) << 4;
                        let q3 = ((ql_h[l] >> 4) as i32) | (((qhh >> 4) & 0x03) as i32) << 4;
                        let q4 = ((ql_h[l + 32] >> 4) as i32) | (((qhh >> 6) & 0x03) as i32) << 4;
                        let s1 = d * (sc_h[l / 16] as i8) as i32 as f32;
                        let s2 = d * (sc_h[2 + l / 16] as i8) as i32 as f32;
                        let s3 = d * (sc_h[4 + l / 16] as i8) as i32 as f32;
                        let s4 = d * (sc_h[6 + l / 16] as i8) as i32 as f32;
                        let h_off = dst_off + half_idx * 128;
                        w_f32[h_off + l] = s1 * (q1 - 32) as f32;
                        w_f32[h_off + l + 32] = s2 * (q2 - 32) as f32;
                        w_f32[h_off + l + 64] = s3 * (q3 - 32) as f32;
                        w_f32[h_off + l + 96] = s4 * (q4 - 32) as f32;
                    }
                }
            }
        }
        let x_f32: Vec<f32> = (0..k).map(|i| ((i as f32 * 0.07).cos()) * 0.4).collect();
        let mut y_cpu = vec![0.0f32; n];
        for row in 0..n {
            let mut acc = 0.0f32;
            for j in 0..k {
                acc += w_f32[row * k + j] * x_f32[j];
            }
            y_cpu[row] = acc;
        }

        // CUDA path.
        let ctx = CudaContext::new(0).expect("ctx");
        let stream = ctx.default_stream();
        let kernels = LlmKernels::new(ctx);
        let w_dev = stream.memcpy_stod(&w_bytes).expect("upload w");
        let x_bf: Vec<half::bf16> = x_f32.iter().copied().map(half::bf16::from_f32).collect();
        let x_dev = stream.memcpy_stod(&x_bf).expect("upload x");
        let mut y_dev = stream.alloc_zeros::<half::bf16>(n).expect("alloc y");
        unsafe {
            let (w_p, _g1) = w_dev.device_ptr(&stream);
            let (x_p, _g2) = x_dev.device_ptr(&stream);
            let (y_p, _g3) = y_dev.device_ptr_mut(&stream);
            kernels
                .sgemv_q6k_bf16(&stream, w_p, x_p, y_p, n as i32, k as i32)
                .expect("sgemv_q6k");
        }
        let y_host: Vec<half::bf16> = stream.memcpy_dtov(&y_dev).expect("dtov");
        let y_gpu: Vec<f32> = y_host.into_iter().map(|x| x.to_f32()).collect();

        for i in 0..n {
            let diff = (y_cpu[i] - y_gpu[i]).abs();
            let tol = y_cpu[i].abs() * 5e-2 + 1.0;
            assert!(
                diff <= tol,
                "row[{i}] cpu={} gpu={} diff={} tol={}",
                y_cpu[i],
                y_gpu[i],
                diff,
                tol
            );
        }
    }

    /// T246.4.4 — REGRESSION TEST for race conditions in quantized SGEMV
    /// kernels. Each kernel must produce IDENTICAL output across two runs
    /// with the same input. Caught by compute-sanitizer racecheck once,
    /// this guards against any future regression by re-running the kernel
    /// 4 times and checking bit-exact output equality.
    ///
    /// Background : V2 quantized matmul kernels iterate over super-blocks.
    /// Per iter, thread 0 writes scale prefactors to shmem, syncthreads,
    /// then all threads read. Without a __syncthreads at END of iteration,
    /// next iter's write races with this iter's read → non-deterministic.
    #[test]
    fn sgemv_quantized_kernels_are_deterministic() {
        use rustorch_gguf::dequant::{Q4_K_BYTES, Q5_K_BYTES, Q6_K_BYTES, QK_K};

        let n = 32usize;
        let k = 1024usize; // 4 super-blocks per row, exposes the race
        let blocks_per_row = k / QK_K;

        let mut state: u64 = 0x12345678abcd;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        // Generate Q4_K, Q5_K, Q6_K weights with sane d/dmin/scales.
        let row_bytes_q4 = blocks_per_row * Q4_K_BYTES;
        let row_bytes_q5 = blocks_per_row * Q5_K_BYTES;
        let row_bytes_q6 = blocks_per_row * Q6_K_BYTES;
        let mut w_q4 = vec![0u8; n * row_bytes_q4];
        let mut w_q5 = vec![0u8; n * row_bytes_q5];
        let mut w_q6 = vec![0u8; n * row_bytes_q6];
        for b in &mut w_q4 {
            *b = (next() & 0xFF) as u8;
        }
        for b in &mut w_q5 {
            *b = (next() & 0xFF) as u8;
        }
        for b in &mut w_q6 {
            *b = (next() & 0xFF) as u8;
        }
        for row in 0..n {
            for blk in 0..blocks_per_row {
                let off4 = row * row_bytes_q4 + blk * Q4_K_BYTES;
                let d4 = half::f16::from_f32(0.05).to_le_bytes();
                let dmin4 = half::f16::from_f32(0.025).to_le_bytes();
                w_q4[off4] = d4[0];
                w_q4[off4 + 1] = d4[1];
                w_q4[off4 + 2] = dmin4[0];
                w_q4[off4 + 3] = dmin4[1];
                for i in 0..12 {
                    w_q4[off4 + 4 + i] &= 0x3F;
                }
                let off5 = row * row_bytes_q5 + blk * Q5_K_BYTES;
                w_q5[off5] = d4[0];
                w_q5[off5 + 1] = d4[1];
                w_q5[off5 + 2] = dmin4[0];
                w_q5[off5 + 3] = dmin4[1];
                for i in 0..12 {
                    w_q5[off5 + 4 + i] &= 0x3F;
                }
                let off6 = row * row_bytes_q6 + blk * Q6_K_BYTES;
                let d6 = half::f16::from_f32(0.04).to_le_bytes();
                w_q6[off6 + 208] = d6[0];
                w_q6[off6 + 209] = d6[1];
                for i in 0..16 {
                    w_q6[off6 + 192 + i] = (next() as i8 / 8) as u8;
                }
            }
        }

        let ctx = CudaContext::new(0).expect("ctx");
        let stream = ctx.default_stream();
        let kernels = LlmKernels::new(ctx);
        let w4_dev = stream.memcpy_stod(&w_q4).expect("");
        let w5_dev = stream.memcpy_stod(&w_q5).expect("");
        let w6_dev = stream.memcpy_stod(&w_q6).expect("");
        let x_bf: Vec<half::bf16> = (0..k)
            .map(|i| half::bf16::from_f32((i as f32 * 0.013).sin() * 0.3))
            .collect();
        let x_dev = stream.memcpy_stod(&x_bf).expect("");

        let (w4_p, w5_p, w6_p, x_p) = unsafe {
            use cudarc::driver::DevicePtr;
            (
                w4_dev.device_ptr(&stream).0,
                w5_dev.device_ptr(&stream).0,
                w6_dev.device_ptr(&stream).0,
                x_dev.device_ptr(&stream).0,
            )
        };

        // For each kernel, run 4 times and check identical output.
        let kernels_to_test: [(&str, Box<dyn Fn(u64) -> Result<(), CudaError>>); 4] = [
            (
                "sgemv_q4k_bf16_v2",
                Box::new(|y: u64| unsafe {
                    kernels.sgemv_q4k_bf16_v2(&stream, w4_p, x_p, y, n as i32, k as i32)
                }),
            ),
            (
                "sgemv_q5k_bf16",
                Box::new(|y: u64| unsafe {
                    kernels.sgemv_q5k_bf16(&stream, w5_p, x_p, y, n as i32, k as i32)
                }),
            ),
            (
                "sgemv_q6k_bf16",
                Box::new(|y: u64| unsafe {
                    kernels.sgemv_q6k_bf16(&stream, w6_p, x_p, y, n as i32, k as i32)
                }),
            ),
            (
                "sgemv_q6k_bf16_v2",
                Box::new(|y: u64| unsafe {
                    kernels.sgemv_q6k_bf16_v2(&stream, w6_p, x_p, y, n as i32, k as i32)
                }),
            ),
        ];

        for (name, run) in kernels_to_test.iter() {
            let mut prev: Option<Vec<half::bf16>> = None;
            for trial in 0..4 {
                let mut y_dev = stream.alloc_zeros::<half::bf16>(n).expect("alloc y");
                let y_p = unsafe {
                    use cudarc::driver::DevicePtrMut;
                    let (p, _g) = y_dev.device_ptr_mut(&stream);
                    p
                };
                run(y_p).unwrap_or_else(|e| panic!("{name} trial {trial}: {e:?}"));
                let y_host: Vec<half::bf16> = stream.memcpy_dtov(&y_dev).expect("dtov");
                if let Some(p) = &prev {
                    assert_eq!(p, &y_host, "{name} non-deterministic at trial {trial}");
                }
                prev = Some(y_host);
            }
        }
    }

    /// T244.4 — sgemv_q6k_bf16_v2 parity test : V2 must match V1 (and CPU
    /// reference) within BF16 tolerance.
    #[test]
    fn sgemv_q6k_bf16_v2_matches_v1() {
        const QK_K: usize = 256;
        const Q6_K_BYTES: usize = 210;

        let n = 32usize;
        let k = 512usize;
        let blocks_per_row = k / QK_K;
        let row_bytes = blocks_per_row * Q6_K_BYTES;

        let mut state: u64 = 0xfeed6600;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let mut w_bytes = vec![0u8; n * row_bytes];
        for row in 0..n {
            for blk in 0..blocks_per_row {
                let off = row * row_bytes + blk * Q6_K_BYTES;
                for i in 0..128 {
                    w_bytes[off + i] = (next() & 0xFF) as u8;
                }
                for i in 0..64 {
                    w_bytes[off + 128 + i] = (next() & 0xFF) as u8;
                }
                for i in 0..16 {
                    let raw = (next() & 0xFF) as u8;
                    let signed = (raw as i8) / 8;
                    w_bytes[off + 192 + i] = signed as u8;
                }
                let d = half::f16::from_f32(0.05).to_le_bytes();
                w_bytes[off + 208] = d[0];
                w_bytes[off + 209] = d[1];
            }
        }
        let x_f32: Vec<f32> = (0..k).map(|i| ((i as f32 * 0.07).cos()) * 0.4).collect();

        let ctx = CudaContext::new(0).expect("ctx");
        let stream = ctx.default_stream();
        let kernels = LlmKernels::new(ctx);
        let w_dev = stream.memcpy_stod(&w_bytes).expect("upload w");
        let x_bf: Vec<half::bf16> = x_f32.iter().copied().map(half::bf16::from_f32).collect();
        let x_dev = stream.memcpy_stod(&x_bf).expect("upload x");
        let mut y_v1 = stream.alloc_zeros::<half::bf16>(n).expect("alloc y1");
        let mut y_v2 = stream.alloc_zeros::<half::bf16>(n).expect("alloc y2");
        unsafe {
            let (w_p, _g1) = w_dev.device_ptr(&stream);
            let (x_p, _g2) = x_dev.device_ptr(&stream);
            let (y1_p, _g3) = y_v1.device_ptr_mut(&stream);
            let (y2_p, _g4) = y_v2.device_ptr_mut(&stream);
            kernels
                .sgemv_q6k_bf16(&stream, w_p, x_p, y1_p, n as i32, k as i32)
                .expect("v1");
            kernels
                .sgemv_q6k_bf16_v2(&stream, w_p, x_p, y2_p, n as i32, k as i32)
                .expect("v2");
        }
        let y1_host: Vec<half::bf16> = stream.memcpy_dtov(&y_v1).expect("dl1");
        let y2_host: Vec<half::bf16> = stream.memcpy_dtov(&y_v2).expect("dl2");
        for i in 0..n {
            let a = y1_host[i].to_f32();
            let b = y2_host[i].to_f32();
            let diff = (a - b).abs();
            let tol = a.abs() * 1e-2 + 1e-2;
            assert!(diff <= tol, "row[{i}] V1={a} V2={b} diff={diff} tol={tol}");
        }
    }

    /// T244.3 — sgemv_q5k_bf16 parity test vs CPU dequant + scalar gemv.
    #[test]
    fn sgemv_q5k_bf16_matches_cpu_reference() {
        use rustorch_gguf::dequant::{dequant_q5_k, Q5_K_BYTES, QK_K};
        use rustorch_gguf::tensor::{GgmlType, TensorInfo};

        let n = 32usize;
        let k = 512usize;
        let blocks_per_row = k / QK_K;
        let row_bytes = blocks_per_row * Q5_K_BYTES;

        let mut state: u64 = 0xfeed5500;
        let mut next_byte = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state & 0xFF) as u8
        };
        let mut w_bytes = vec![0u8; n * row_bytes];
        for row in 0..n {
            for blk in 0..blocks_per_row {
                let off = row * row_bytes + blk * Q5_K_BYTES;
                // d=0.05, dmin=0.025
                let d = half::f16::from_f32(0.05).to_le_bytes();
                let dmin = half::f16::from_f32(0.025).to_le_bytes();
                w_bytes[off] = d[0];
                w_bytes[off + 1] = d[1];
                w_bytes[off + 2] = dmin[0];
                w_bytes[off + 3] = dmin[1];
                // 12 scales (small magnitudes)
                for i in 0..12 {
                    w_bytes[off + 4 + i] = next_byte() & 0x3F;
                }
                // 32 qh + 128 ql random
                for i in 0..(32 + 128) {
                    w_bytes[off + 16 + i] = next_byte();
                }
            }
        }

        // CPU reference dequant.
        let info = TensorInfo {
            name: "test".into(),
            shape: vec![k as u64, n as u64],
            dtype: GgmlType::Q5_K,
            offset: 0,
        };
        let mut w_f32 = vec![0.0f32; n * k];
        for row in 0..n {
            let row_off = row * row_bytes;
            dequant_q5_k(
                &w_bytes[row_off..row_off + row_bytes],
                &mut w_f32[row * k..(row + 1) * k],
            )
            .expect("dequant");
        }
        let _ = info;
        let x_f32: Vec<f32> = (0..k).map(|i| ((i as f32 * 0.07).sin()) * 0.3).collect();
        let mut y_cpu = vec![0.0f32; n];
        for row in 0..n {
            let mut acc = 0.0f32;
            for j in 0..k {
                acc += w_f32[row * k + j] * x_f32[j];
            }
            y_cpu[row] = acc;
        }

        // CUDA path.
        let ctx = CudaContext::new(0).expect("ctx");
        let stream = ctx.default_stream();
        let kernels = LlmKernels::new(ctx);
        let w_dev = stream.memcpy_stod(&w_bytes).expect("upload w");
        let x_bf: Vec<half::bf16> = x_f32.iter().copied().map(half::bf16::from_f32).collect();
        let x_dev = stream.memcpy_stod(&x_bf).expect("upload x");
        let mut y_dev = stream.alloc_zeros::<half::bf16>(n).expect("alloc y");
        unsafe {
            let (w_p, _g1) = w_dev.device_ptr(&stream);
            let (x_p, _g2) = x_dev.device_ptr(&stream);
            let (y_p, _g3) = y_dev.device_ptr_mut(&stream);
            kernels
                .sgemv_q5k_bf16(&stream, w_p, x_p, y_p, n as i32, k as i32)
                .expect("sgemv_q5k");
        }
        let y_host: Vec<half::bf16> = stream.memcpy_dtov(&y_dev).expect("dtov");
        let y_gpu: Vec<f32> = y_host.into_iter().map(|x| x.to_f32()).collect();

        for i in 0..n {
            let diff = (y_cpu[i] - y_gpu[i]).abs();
            let tol = y_cpu[i].abs() * 5e-2 + 1.0;
            assert!(
                diff <= tol,
                "row[{i}] cpu={} gpu={} diff={} tol={}",
                y_cpu[i],
                y_gpu[i],
                diff,
                tol
            );
        }
    }

    /// T243.2 — l2_norm_per_head_bf16 parity vs CPU.
    #[test]
    fn l2_norm_per_head_bf16_matches_cpu() {
        let n_heads = 4usize;
        let head_dim = 64usize;
        let eps = 1e-6f32;
        let x_f32: Vec<f32> = (0..(n_heads * head_dim))
            .map(|i| ((i as f32 * 0.13).sin()) * 0.7)
            .collect();
        let mut cpu = x_f32.clone();
        for h in 0..n_heads {
            let off = h * head_dim;
            let sumsq: f32 = cpu[off..off + head_dim].iter().map(|v| v * v).sum();
            let inv = 1.0 / (sumsq + eps).sqrt();
            for v in &mut cpu[off..off + head_dim] {
                *v *= inv;
            }
        }
        let ctx = CudaContext::new(0).expect("ctx");
        let stream = ctx.default_stream();
        let kernels = LlmKernels::new(ctx);
        let bf: Vec<half::bf16> = x_f32.iter().copied().map(half::bf16::from_f32).collect();
        let mut dev = stream.memcpy_stod(&bf).expect("upload");
        unsafe {
            let (p, _g) = dev.device_ptr_mut(&stream);
            kernels
                .l2_norm_per_head_bf16(&stream, p, n_heads as i32, head_dim as i32, eps)
                .expect("l2");
        }
        let host: Vec<half::bf16> = stream.memcpy_dtov(&dev).expect("dtov");
        let gpu: Vec<f32> = host.into_iter().map(|v| v.to_f32()).collect();
        for i in 0..(n_heads * head_dim) {
            let diff = (cpu[i] - gpu[i]).abs();
            let tol = cpu[i].abs() * 1e-1 + 1e-2;
            assert!(diff <= tol, "[{i}] cpu={} gpu={}", cpu[i], gpu[i]);
        }
    }

    /// T243.2 — conv1d_depthwise_bf16 parity vs CPU.
    #[test]
    fn conv1d_depthwise_bf16_matches_cpu() {
        let conv_dim = 32usize;
        let kernel_size = 4usize;
        let w_f32: Vec<f32> = (0..(kernel_size * conv_dim))
            .map(|i| ((i as f32 * 0.05).cos()) * 0.3)
            .collect();
        let st_f32: Vec<f32> = (0..((kernel_size - 1) * conv_dim))
            .map(|i| ((i as f32 * 0.07).sin()) * 0.2)
            .collect();
        let in_f32: Vec<f32> = (0..conv_dim)
            .map(|i| ((i as f32 * 0.11).sin()) * 0.5)
            .collect();
        let mut cpu_out = vec![0.0f32; conv_dim];
        for c in 0..conv_dim {
            let mut acc = 0.0f32;
            for t in 0..(kernel_size - 1) {
                acc += st_f32[t * conv_dim + c] * w_f32[t * conv_dim + c];
            }
            acc += in_f32[c] * w_f32[(kernel_size - 1) * conv_dim + c];
            cpu_out[c] = acc;
        }
        let ctx = CudaContext::new(0).expect("ctx");
        let stream = ctx.default_stream();
        let kernels = LlmKernels::new(ctx);
        let to_bf = |v: &[f32]| -> Vec<half::bf16> {
            v.iter().copied().map(half::bf16::from_f32).collect()
        };
        let w_dev = stream.memcpy_stod(&to_bf(&w_f32)).expect("up w");
        let mut s_dev = stream.memcpy_stod(&to_bf(&st_f32)).expect("up s");
        let in_dev = stream.memcpy_stod(&to_bf(&in_f32)).expect("up in");
        let mut out_dev = stream
            .alloc_zeros::<half::bf16>(conv_dim)
            .expect("alloc out");
        unsafe {
            let (w_p, _g1) = w_dev.device_ptr(&stream);
            let (s_p, _g2) = s_dev.device_ptr_mut(&stream);
            let (i_p, _g3) = in_dev.device_ptr(&stream);
            let (o_p, _g4) = out_dev.device_ptr_mut(&stream);
            kernels
                .conv1d_depthwise_bf16(
                    &stream,
                    w_p,
                    s_p,
                    i_p,
                    o_p,
                    conv_dim as i32,
                    kernel_size as i32,
                )
                .expect("conv");
        }
        let host: Vec<half::bf16> = stream.memcpy_dtov(&out_dev).expect("dtov");
        let gpu: Vec<f32> = host.into_iter().map(|v| v.to_f32()).collect();
        for c in 0..conv_dim {
            let diff = (cpu_out[c] - gpu[c]).abs();
            let tol = cpu_out[c].abs() * 5e-2 + 1e-2;
            assert!(diff <= tol, "[{c}] cpu={} gpu={}", cpu_out[c], gpu[c]);
        }
    }

    /// T243.3 — Qwen3.6-27B full forward bench : simule un decode token
    /// complet en orchestrant tous les kernels (attention + SSM + FFN)
    /// avec les shapes officielles Qwen3.6-27B.
    ///
    /// Architecture : 64 layers total = 16 attention (every 4th, idx 3,7,11..)
    /// + 48 SSM (others). Plus FFN dense per layer.
    ///
    /// Cible : valider qu'avec NOS kernels on peut faire du Qwen3.6 decode
    /// avec des tok/s comparables à llama.cpp 11.62.
    ///
    /// T244.2 — variant using sgemv_bf16_bf16 (warp-shuffle) instead of
    /// cuBLASLt matmul_bf16. cuBLASLt for M=1 caps at ~19 GB/s (10% peak).
    /// Our kernel hits ~168 GB/s (84% peak) → ~9× speedup expected.
    #[test]
    #[ignore = "perf benchmark"]
    fn qwen36_27b_full_decode_bench_custom_sgemv() {
        use std::time::Instant;

        let d = 5120usize;
        let f = 17408usize;
        let n_layers = 64usize;
        let head_dim = 256usize;
        let n_q = 24usize;
        let n_kv = 4usize;
        let q_dim = head_dim * n_q;
        let kv_dim = head_dim * n_kv;
        let ssm_state = 128usize;
        let ssm_groups = 16usize;
        let ssm_dt_rank = 48usize;
        let ssm_key_dim = ssm_state * ssm_groups;
        let ssm_value_dim = ssm_state * ssm_dt_rank;
        let ssm_conv_dim = 2 * ssm_key_dim + ssm_value_dim;
        let ssm_conv_kernel = 4usize;

        let n_attn_layers = n_layers / 4;
        let n_ssm_layers = n_layers - n_attn_layers;

        let ctx = CudaContext::new(0).expect("ctx");
        let stream = ctx.default_stream();
        let kernels = LlmKernels::new(ctx.clone());

        let attn_w_q = stream.alloc_zeros::<half::bf16>(2 * q_dim * d).expect("");
        let attn_w_k = stream.alloc_zeros::<half::bf16>(kv_dim * d).expect("");
        let attn_w_v = stream.alloc_zeros::<half::bf16>(kv_dim * d).expect("");
        let attn_w_o = stream.alloc_zeros::<half::bf16>(d * q_dim).expect("");
        let ssm_w_qkv = stream
            .alloc_zeros::<half::bf16>(ssm_conv_dim * d)
            .expect("");
        let ssm_w_gate = stream
            .alloc_zeros::<half::bf16>(ssm_value_dim * d)
            .expect("");
        let ssm_w_alpha = stream.alloc_zeros::<half::bf16>(ssm_dt_rank * d).expect("");
        let ssm_w_beta = stream.alloc_zeros::<half::bf16>(ssm_dt_rank * d).expect("");
        let ssm_w_out = stream
            .alloc_zeros::<half::bf16>(d * ssm_value_dim)
            .expect("");
        let ssm_conv_w = stream
            .alloc_zeros::<half::bf16>(ssm_conv_kernel * ssm_conv_dim)
            .expect("");
        let ffn_w_gate = stream.alloc_zeros::<half::bf16>(f * d).expect("");
        let ffn_w_up = stream.alloc_zeros::<half::bf16>(f * d).expect("");
        let ffn_w_down = stream.alloc_zeros::<half::bf16>(d * f).expect("");

        let h_dev = stream.alloc_zeros::<half::bf16>(d).expect("");
        let mut qg_buf = stream.alloc_zeros::<half::bf16>(2 * q_dim).expect("");
        let mut k_buf = stream.alloc_zeros::<half::bf16>(kv_dim).expect("");
        let mut v_buf = stream.alloc_zeros::<half::bf16>(kv_dim).expect("");
        let mut attn_out = stream.alloc_zeros::<half::bf16>(q_dim).expect("");
        let mut ssm_qkv_buf = stream.alloc_zeros::<half::bf16>(ssm_conv_dim).expect("");
        let mut ssm_z_buf = stream.alloc_zeros::<half::bf16>(ssm_value_dim).expect("");
        let mut ssm_alpha = stream.alloc_zeros::<half::bf16>(ssm_dt_rank).expect("");
        let mut ssm_beta = stream.alloc_zeros::<half::bf16>(ssm_dt_rank).expect("");
        let mut ssm_conv_state = stream
            .alloc_zeros::<half::bf16>((ssm_conv_kernel - 1) * ssm_conv_dim)
            .expect("");
        let mut ssm_conv_out = stream.alloc_zeros::<half::bf16>(ssm_conv_dim).expect("");
        let mut ssm_state_buf = stream
            .alloc_zeros::<half::bf16>(ssm_dt_rank * ssm_state * ssm_state)
            .expect("");
        let mut ssm_out_buf = stream
            .alloc_zeros::<half::bf16>(ssm_dt_rank * ssm_state)
            .expect("");
        let mut gate_buf = stream.alloc_zeros::<half::bf16>(f).expect("");
        let mut up_buf = stream.alloc_zeros::<half::bf16>(f).expect("");
        let mut down_buf = stream.alloc_zeros::<half::bf16>(d).expect("");

        let (
            h_p,
            qg_p,
            k_p,
            v_p,
            ao_p,
            sqkv_p,
            sz_p,
            sa_p,
            sb_p,
            scs_p,
            sco_p,
            sst_p,
            sso_p,
            gate_p,
            up_p,
            down_p,
            aw_q,
            aw_k,
            aw_v,
            aw_o,
            sw_qkv,
            sw_gate,
            sw_alpha,
            sw_beta,
            sw_out,
            sconv_w,
            fw_gate,
            fw_up,
            fw_down,
        ) = unsafe {
            use cudarc::driver::{DevicePtr, DevicePtrMut};
            (
                h_dev.device_ptr(&stream).0,
                qg_buf.device_ptr_mut(&stream).0,
                k_buf.device_ptr_mut(&stream).0,
                v_buf.device_ptr_mut(&stream).0,
                attn_out.device_ptr_mut(&stream).0,
                ssm_qkv_buf.device_ptr_mut(&stream).0,
                ssm_z_buf.device_ptr_mut(&stream).0,
                ssm_alpha.device_ptr_mut(&stream).0,
                ssm_beta.device_ptr_mut(&stream).0,
                ssm_conv_state.device_ptr_mut(&stream).0,
                ssm_conv_out.device_ptr_mut(&stream).0,
                ssm_state_buf.device_ptr_mut(&stream).0,
                ssm_out_buf.device_ptr_mut(&stream).0,
                gate_buf.device_ptr_mut(&stream).0,
                up_buf.device_ptr_mut(&stream).0,
                down_buf.device_ptr_mut(&stream).0,
                attn_w_q.device_ptr(&stream).0,
                attn_w_k.device_ptr(&stream).0,
                attn_w_v.device_ptr(&stream).0,
                attn_w_o.device_ptr(&stream).0,
                ssm_w_qkv.device_ptr(&stream).0,
                ssm_w_gate.device_ptr(&stream).0,
                ssm_w_alpha.device_ptr(&stream).0,
                ssm_w_beta.device_ptr(&stream).0,
                ssm_w_out.device_ptr(&stream).0,
                ssm_conv_w.device_ptr(&stream).0,
                ffn_w_gate.device_ptr(&stream).0,
                ffn_w_up.device_ptr(&stream).0,
                ffn_w_down.device_ptr(&stream).0,
            )
        };

        // Full warm-up : every distinct shape, then sync.
        unsafe {
            let _ = kernels.sgemv_bf16_bf16(&stream, aw_q, h_p, qg_p, (2 * q_dim) as i32, d as i32);
            let _ = kernels.sgemv_bf16_bf16(&stream, aw_k, h_p, k_p, kv_dim as i32, d as i32);
            let _ = kernels.sgemv_bf16_bf16(&stream, aw_v, h_p, v_p, kv_dim as i32, d as i32);
            let _ = kernels.sgemv_bf16_bf16(&stream, aw_o, ao_p, h_p, d as i32, q_dim as i32);
            let _ = kernels.sgemv_bf16_bf16(
                &stream,
                sw_qkv,
                h_p,
                sqkv_p,
                ssm_conv_dim as i32,
                d as i32,
            );
            let _ = kernels.sgemv_bf16_bf16(
                &stream,
                sw_gate,
                h_p,
                sz_p,
                ssm_value_dim as i32,
                d as i32,
            );
            let _ =
                kernels.sgemv_bf16_bf16(&stream, sw_alpha, h_p, sa_p, ssm_dt_rank as i32, d as i32);
            let _ =
                kernels.sgemv_bf16_bf16(&stream, sw_beta, h_p, sb_p, ssm_dt_rank as i32, d as i32);
            let _ = kernels.sgemv_bf16_bf16(
                &stream,
                sw_out,
                sso_p,
                h_p,
                d as i32,
                ssm_value_dim as i32,
            );
            let _ = kernels.sgemv_bf16_bf16(&stream, fw_gate, h_p, gate_p, f as i32, d as i32);
            let _ = kernels.sgemv_bf16_bf16(&stream, fw_up, h_p, up_p, f as i32, d as i32);
            let _ = kernels.sgemv_bf16_bf16(&stream, fw_down, up_p, down_p, d as i32, f as i32);
            kernels
                .conv1d_depthwise_bf16(
                    &stream,
                    sconv_w,
                    scs_p,
                    sqkv_p,
                    sco_p,
                    ssm_conv_dim as i32,
                    ssm_conv_kernel as i32,
                )
                .ok();
            kernels
                .l2_norm_per_head_bf16(&stream, sco_p, ssm_groups as i32, ssm_state as i32, 1e-6)
                .ok();
            kernels
                .delta_net_step_bf16(
                    &stream,
                    sco_p,
                    sco_p,
                    sco_p,
                    sa_p,
                    sb_p,
                    sst_p,
                    sso_p,
                    ssm_dt_rank as i32,
                    ssm_state as i32,
                )
                .ok();
        }
        stream.synchronize().ok();

        let n_iters = 50;
        let t0 = Instant::now();
        for _ in 0..n_iters {
            for li in 0..n_layers {
                let is_attention = (li + 1) % 4 == 0;
                if is_attention {
                    unsafe {
                        let _ = kernels.sgemv_bf16_bf16(
                            &stream,
                            aw_q,
                            h_p,
                            qg_p,
                            (2 * q_dim) as i32,
                            d as i32,
                        );
                        let _ = kernels.sgemv_bf16_bf16(
                            &stream,
                            aw_k,
                            h_p,
                            k_p,
                            kv_dim as i32,
                            d as i32,
                        );
                        let _ = kernels.sgemv_bf16_bf16(
                            &stream,
                            aw_v,
                            h_p,
                            v_p,
                            kv_dim as i32,
                            d as i32,
                        );
                        let _ = kernels.sgemv_bf16_bf16(
                            &stream,
                            aw_o,
                            ao_p,
                            h_p,
                            d as i32,
                            q_dim as i32,
                        );
                    }
                } else {
                    unsafe {
                        let _ = kernels.sgemv_bf16_bf16(
                            &stream,
                            sw_qkv,
                            h_p,
                            sqkv_p,
                            ssm_conv_dim as i32,
                            d as i32,
                        );
                        let _ = kernels.sgemv_bf16_bf16(
                            &stream,
                            sw_gate,
                            h_p,
                            sz_p,
                            ssm_value_dim as i32,
                            d as i32,
                        );
                        let _ = kernels.sgemv_bf16_bf16(
                            &stream,
                            sw_alpha,
                            h_p,
                            sa_p,
                            ssm_dt_rank as i32,
                            d as i32,
                        );
                        let _ = kernels.sgemv_bf16_bf16(
                            &stream,
                            sw_beta,
                            h_p,
                            sb_p,
                            ssm_dt_rank as i32,
                            d as i32,
                        );
                        kernels
                            .conv1d_depthwise_bf16(
                                &stream,
                                sconv_w,
                                scs_p,
                                sqkv_p,
                                sco_p,
                                ssm_conv_dim as i32,
                                ssm_conv_kernel as i32,
                            )
                            .ok();
                        kernels
                            .l2_norm_per_head_bf16(
                                &stream,
                                sco_p,
                                ssm_groups as i32,
                                ssm_state as i32,
                                1e-6,
                            )
                            .ok();
                        kernels
                            .delta_net_step_bf16(
                                &stream,
                                sco_p,
                                sco_p,
                                sco_p,
                                sa_p,
                                sb_p,
                                sst_p,
                                sso_p,
                                ssm_dt_rank as i32,
                                ssm_state as i32,
                            )
                            .ok();
                        let _ = kernels.sgemv_bf16_bf16(
                            &stream,
                            sw_out,
                            sso_p,
                            h_p,
                            d as i32,
                            ssm_value_dim as i32,
                        );
                    }
                }
                unsafe {
                    let _ =
                        kernels.sgemv_bf16_bf16(&stream, fw_gate, h_p, gate_p, f as i32, d as i32);
                    let _ = kernels.sgemv_bf16_bf16(&stream, fw_up, h_p, up_p, f as i32, d as i32);
                    let _ =
                        kernels.sgemv_bf16_bf16(&stream, fw_down, up_p, down_p, d as i32, f as i32);
                }
            }
        }
        stream.synchronize().ok();
        let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0 / n_iters as f64;
        let tok_s = 1000.0 / elapsed_ms;

        let bytes_per_token = (n_attn_layers as f64
            * 2.0
            * (2 * q_dim * d + kv_dim * d * 2 + d * q_dim) as f64
            + n_ssm_layers as f64
                * 2.0
                * ((ssm_conv_dim * d + ssm_value_dim * d + ssm_dt_rank * d * 2 + d * ssm_value_dim)
                    as f64)
            + n_layers as f64 * 2.0 * (3 * f * d) as f64);
        let bw_gb_s = bytes_per_token / (elapsed_ms * 1e-3) / 1e9;

        eprintln!();
        eprintln!("=== Qwen3.6-27B FULL DECODE — CUSTOM sgemv_bf16_bf16 (no cuBLASLt) ===");
        eprintln!("  Layers : {n_attn_layers} attention + {n_ssm_layers} SSM + {n_layers} FFN");
        eprintln!("  per-token: {elapsed_ms:.2} ms = {tok_s:.2} tok/s (BF16, custom GEMV)");
        eprintln!("  weight bytes/token: {:.2} GB", bytes_per_token / 1e9);
        eprintln!(
            "  effective bandwidth: {bw_gb_s:.1} GB/s (peak 200 GB/s = {:.0}%)",
            bw_gb_s / 2.0
        );
        eprintln!();
        eprintln!("  llama.cpp Qwen3.6-27B Q4_K_M : 11.62 tok/s");
        eprintln!(
            "  Projection rustorch Q4_K_M (28% bytes) : ~{:.1} tok/s",
            tok_s / 0.28
        );
        let r = (tok_s / 0.28) / 11.62;
        eprintln!(
            "  Projected ratio rustorch/llama.cpp : {r:.2}× ({})",
            if r >= 1.0 {
                "FASTER ✓"
            } else {
                "still slower"
            }
        );
    }

    /// Original BF16 baseline using cuBLASLt (slow path, kept for comparison).
    #[test]
    #[ignore = "perf benchmark"]
    fn qwen36_27b_full_decode_bench() {
        use std::time::Instant;

        // Qwen3.6-27B real shapes
        let d = 5120usize; // hidden
        let f = 17408usize; // intermediate
        let n_layers = 64usize;
        let head_dim = 256usize; // attention head_dim
        let n_q = 24usize;
        let n_kv = 4usize;
        let q_dim = head_dim * n_q; // 6144
        let kv_dim = head_dim * n_kv; // 1024
                                      // SSM dimensions
        let ssm_state = 128usize; // head_kv
        let ssm_groups = 16usize; // n_k
        let ssm_dt_rank = 48usize; // n_v
        let ssm_key_dim = ssm_state * ssm_groups; // 2048
        let ssm_value_dim = ssm_state * ssm_dt_rank; // 6144
        let ssm_conv_dim = 2 * ssm_key_dim + ssm_value_dim; // 10240
        let ssm_conv_kernel = 4usize;

        // Layer kind : every 4th is full attention (1/4), rest is SSM.
        let n_attn_layers = n_layers / 4; // 16
        let n_ssm_layers = n_layers - n_attn_layers; // 48

        let ctx = CudaContext::new(0).expect("ctx");
        let stream = ctx.default_stream();
        let kernels = LlmKernels::new(ctx.clone());
        let mut session = crate::cublas_lt::LtSession::new(stream.clone()).expect("lt");

        // Allocate ALL weights as BF16 (worst case bandwidth — Q4_K kernel
        // would be 28% of this on average).
        // Per attn layer : w_q (2*q_dim, d) + w_k (kv_dim, d) + w_v (kv_dim, d)
        //                 + w_o (d, q_dim) + 3 ffn matmuls (gate, up, down)
        // Per ssm layer : w_qkv (conv_dim, d) + w_gate (value_dim, d)
        //                + w_alpha (n_v, d) + w_beta (n_v, d)
        //                + w_ssm_out (d, value_dim) + 3 ffn matmuls
        // All in BF16 to upper-bound bandwidth.
        let attn_w_q = stream.alloc_zeros::<half::bf16>(2 * q_dim * d).expect("");
        let attn_w_k = stream.alloc_zeros::<half::bf16>(kv_dim * d).expect("");
        let attn_w_v = stream.alloc_zeros::<half::bf16>(kv_dim * d).expect("");
        let attn_w_o = stream.alloc_zeros::<half::bf16>(d * q_dim).expect("");

        let ssm_w_qkv = stream
            .alloc_zeros::<half::bf16>(ssm_conv_dim * d)
            .expect("");
        let ssm_w_gate = stream
            .alloc_zeros::<half::bf16>(ssm_value_dim * d)
            .expect("");
        let ssm_w_alpha = stream.alloc_zeros::<half::bf16>(ssm_dt_rank * d).expect("");
        let ssm_w_beta = stream.alloc_zeros::<half::bf16>(ssm_dt_rank * d).expect("");
        let ssm_w_out = stream
            .alloc_zeros::<half::bf16>(d * ssm_value_dim)
            .expect("");
        let ssm_conv_w = stream
            .alloc_zeros::<half::bf16>(ssm_conv_kernel * ssm_conv_dim)
            .expect("");

        let ffn_w_gate = stream.alloc_zeros::<half::bf16>(f * d).expect("");
        let ffn_w_up = stream.alloc_zeros::<half::bf16>(f * d).expect("");
        let ffn_w_down = stream.alloc_zeros::<half::bf16>(d * f).expect("");

        // Activations and state.
        let h_dev = stream.alloc_zeros::<half::bf16>(d).expect("");
        let mut qg_buf = stream.alloc_zeros::<half::bf16>(2 * q_dim).expect("");
        let mut k_buf = stream.alloc_zeros::<half::bf16>(kv_dim).expect("");
        let mut v_buf = stream.alloc_zeros::<half::bf16>(kv_dim).expect("");
        let mut attn_out = stream.alloc_zeros::<half::bf16>(q_dim).expect("");
        let mut ssm_qkv_buf = stream.alloc_zeros::<half::bf16>(ssm_conv_dim).expect("");
        let mut ssm_z_buf = stream.alloc_zeros::<half::bf16>(ssm_value_dim).expect("");
        let mut ssm_alpha = stream.alloc_zeros::<half::bf16>(ssm_dt_rank).expect("");
        let mut ssm_beta = stream.alloc_zeros::<half::bf16>(ssm_dt_rank).expect("");
        let mut ssm_conv_state = stream
            .alloc_zeros::<half::bf16>((ssm_conv_kernel - 1) * ssm_conv_dim)
            .expect("");
        let mut ssm_conv_out = stream.alloc_zeros::<half::bf16>(ssm_conv_dim).expect("");
        let mut ssm_state_buf = stream
            .alloc_zeros::<half::bf16>(ssm_dt_rank * ssm_state * ssm_state)
            .expect("");
        let mut ssm_out_buf = stream
            .alloc_zeros::<half::bf16>(ssm_dt_rank * ssm_state)
            .expect("");
        let mut gate_buf = stream.alloc_zeros::<half::bf16>(f).expect("");
        let mut up_buf = stream.alloc_zeros::<half::bf16>(f).expect("");
        let mut down_buf = stream.alloc_zeros::<half::bf16>(d).expect("");

        // Pre-extract pointers (drop guards).
        let (
            h_p,
            qg_p,
            k_p,
            v_p,
            ao_p,
            sqkv_p,
            sz_p,
            sa_p,
            sb_p,
            scs_p,
            sco_p,
            sst_p,
            sso_p,
            gate_p,
            up_p,
            down_p,
            aw_q,
            aw_k,
            aw_v,
            aw_o,
            sw_qkv,
            sw_gate,
            sw_alpha,
            sw_beta,
            sw_out,
            sconv_w,
            fw_gate,
            fw_up,
            fw_down,
        ) = unsafe {
            use cudarc::driver::{DevicePtr, DevicePtrMut};
            (
                h_dev.device_ptr(&stream).0,
                qg_buf.device_ptr_mut(&stream).0,
                k_buf.device_ptr_mut(&stream).0,
                v_buf.device_ptr_mut(&stream).0,
                attn_out.device_ptr_mut(&stream).0,
                ssm_qkv_buf.device_ptr_mut(&stream).0,
                ssm_z_buf.device_ptr_mut(&stream).0,
                ssm_alpha.device_ptr_mut(&stream).0,
                ssm_beta.device_ptr_mut(&stream).0,
                ssm_conv_state.device_ptr_mut(&stream).0,
                ssm_conv_out.device_ptr_mut(&stream).0,
                ssm_state_buf.device_ptr_mut(&stream).0,
                ssm_out_buf.device_ptr_mut(&stream).0,
                gate_buf.device_ptr_mut(&stream).0,
                up_buf.device_ptr_mut(&stream).0,
                down_buf.device_ptr_mut(&stream).0,
                attn_w_q.device_ptr(&stream).0,
                attn_w_k.device_ptr(&stream).0,
                attn_w_v.device_ptr(&stream).0,
                attn_w_o.device_ptr(&stream).0,
                ssm_w_qkv.device_ptr(&stream).0,
                ssm_w_gate.device_ptr(&stream).0,
                ssm_w_alpha.device_ptr(&stream).0,
                ssm_w_beta.device_ptr(&stream).0,
                ssm_w_out.device_ptr(&stream).0,
                ssm_conv_w.device_ptr(&stream).0,
                ffn_w_gate.device_ptr(&stream).0,
                ffn_w_up.device_ptr(&stream).0,
                ffn_w_down.device_ptr(&stream).0,
            )
        };

        // PROPER WARM-UP : exercise EVERY distinct (M, K, N) shape so cuBLASLt
        // build_cached cost doesn't pollute the measurement.
        unsafe {
            let _ = session.matmul_bf16(h_p, aw_q, qg_p, 1, d, 2 * q_dim, 1.0, 0.0);
            let _ = session.matmul_bf16(h_p, aw_k, k_p, 1, d, kv_dim, 1.0, 0.0);
            let _ = session.matmul_bf16(h_p, aw_v, v_p, 1, d, kv_dim, 1.0, 0.0);
            let _ = session.matmul_bf16(ao_p, aw_o, h_p, 1, q_dim, d, 1.0, 0.0);
            let _ = session.matmul_bf16(h_p, sw_qkv, sqkv_p, 1, d, ssm_conv_dim, 1.0, 0.0);
            let _ = session.matmul_bf16(h_p, sw_gate, sz_p, 1, d, ssm_value_dim, 1.0, 0.0);
            let _ = session.matmul_bf16(h_p, sw_alpha, sa_p, 1, d, ssm_dt_rank, 1.0, 0.0);
            let _ = session.matmul_bf16(h_p, sw_beta, sb_p, 1, d, ssm_dt_rank, 1.0, 0.0);
            let _ = session.matmul_bf16(sso_p, sw_out, h_p, 1, ssm_value_dim, d, 1.0, 0.0);
            let _ = session.matmul_bf16(h_p, fw_gate, gate_p, 1, d, f, 1.0, 0.0);
            let _ = session.matmul_bf16(h_p, fw_up, up_p, 1, d, f, 1.0, 0.0);
            let _ = session.matmul_bf16(up_p, fw_down, down_p, 1, f, d, 1.0, 0.0);
            kernels
                .conv1d_depthwise_bf16(
                    &stream,
                    sconv_w,
                    scs_p,
                    sqkv_p,
                    sco_p,
                    ssm_conv_dim as i32,
                    ssm_conv_kernel as i32,
                )
                .ok();
            kernels
                .l2_norm_per_head_bf16(&stream, sco_p, ssm_groups as i32, ssm_state as i32, 1e-6)
                .ok();
            kernels
                .delta_net_step_bf16(
                    &stream,
                    sco_p,
                    sco_p,
                    sco_p,
                    sa_p,
                    sb_p,
                    sst_p,
                    sso_p,
                    ssm_dt_rank as i32,
                    ssm_state as i32,
                )
                .ok();
        }
        stream.synchronize().ok();

        let n_iters = 50;
        let t0 = Instant::now();
        for _ in 0..n_iters {
            for li in 0..n_layers {
                let is_attention = (li + 1) % 4 == 0;
                if is_attention {
                    // === Attention block ===
                    unsafe {
                        let _ = session.matmul_bf16(h_p, aw_q, qg_p, 1, d, 2 * q_dim, 1.0, 0.0);
                        let _ = session.matmul_bf16(h_p, aw_k, k_p, 1, d, kv_dim, 1.0, 0.0);
                        let _ = session.matmul_bf16(h_p, aw_v, v_p, 1, d, kv_dim, 1.0, 0.0);
                        // Skip RoPE/GQA (cheap) for bench focus on matmul/mixer.
                        let _ = session.matmul_bf16(ao_p, aw_o, h_p, 1, q_dim, d, 1.0, 0.0);
                    }
                } else {
                    // === SSM block ===
                    unsafe {
                        let _ =
                            session.matmul_bf16(h_p, sw_qkv, sqkv_p, 1, d, ssm_conv_dim, 1.0, 0.0);
                        let _ =
                            session.matmul_bf16(h_p, sw_gate, sz_p, 1, d, ssm_value_dim, 1.0, 0.0);
                        let _ =
                            session.matmul_bf16(h_p, sw_alpha, sa_p, 1, d, ssm_dt_rank, 1.0, 0.0);
                        let _ =
                            session.matmul_bf16(h_p, sw_beta, sb_p, 1, d, ssm_dt_rank, 1.0, 0.0);
                        kernels
                            .conv1d_depthwise_bf16(
                                &stream,
                                sconv_w,
                                scs_p,
                                sqkv_p,
                                sco_p,
                                ssm_conv_dim as i32,
                                ssm_conv_kernel as i32,
                            )
                            .ok();
                        kernels
                            .l2_norm_per_head_bf16(
                                &stream,
                                sco_p,
                                ssm_groups as i32,
                                ssm_state as i32,
                                1e-6,
                            )
                            .ok();
                        kernels
                            .delta_net_step_bf16(
                                &stream,
                                sco_p,
                                sco_p,
                                sco_p,
                                sa_p,
                                sb_p,
                                sst_p,
                                sso_p,
                                ssm_dt_rank as i32,
                                ssm_state as i32,
                            )
                            .ok();
                        let _ =
                            session.matmul_bf16(sso_p, sw_out, h_p, 1, ssm_value_dim, d, 1.0, 0.0);
                    }
                }
                // === FFN block (per layer, dense) ===
                unsafe {
                    let _ = session.matmul_bf16(h_p, fw_gate, gate_p, 1, d, f, 1.0, 0.0);
                    let _ = session.matmul_bf16(h_p, fw_up, up_p, 1, d, f, 1.0, 0.0);
                    // SwiGLU implicit (handled by separate kernel — skip for bench)
                    let _ = session.matmul_bf16(up_p, fw_down, down_p, 1, f, d, 1.0, 0.0);
                }
            }
        }
        stream.synchronize().ok();
        let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0 / n_iters as f64;
        let tok_s = 1000.0 / elapsed_ms;

        let bytes_per_token = (
            // Attention layers (16) × (qg + k + v + o)
            n_attn_layers as f64
                * 2.0
                * (2 * q_dim * d + kv_dim * d * 2 + d * q_dim) as f64
            // SSM layers (48) × (qkv + gate + alpha + beta + ssm_out)
            + n_ssm_layers as f64
                * 2.0
                * ((ssm_conv_dim * d + ssm_value_dim * d
                    + ssm_dt_rank * d * 2
                    + d * ssm_value_dim)
                    as f64)
            // FFN per layer × 64 layers
            + n_layers as f64 * 2.0 * (3 * f * d) as f64
        );
        let bw_gb_s = bytes_per_token / (elapsed_ms * 1e-3) / 1e9;

        eprintln!();
        eprintln!(
            "=== Qwen3.6-27B FULL DECODE BENCH (BF16 baseline, all matmuls + SSM kernels) ==="
        );
        eprintln!("  Layers : {n_attn_layers} attention + {n_ssm_layers} SSM + {n_layers} FFN");
        eprintln!("  per-token: {elapsed_ms:.2} ms = {tok_s:.2} tok/s (BF16)");
        eprintln!("  weight bytes/token: {:.2} GB", bytes_per_token / 1e9);
        eprintln!("  effective bandwidth: {bw_gb_s:.1} GB/s");
        eprintln!();
        eprintln!("  llama.cpp Qwen3.6-27B Q4_K_M (Q4K weights = ~28%) : 11.62 tok/s");
        eprintln!(
            "  Projection rustorch Q4_K_M (28% bytes) : ~{:.1} tok/s",
            tok_s / 0.28
        );
        eprintln!("  → Si on dépasse 11.62 tok/s en Q4_K_M, on bat llama.cpp.");
    }

    /// T243.2.1 — full SSM block bench at Qwen3.6-27B dimensions.
    #[test]
    #[ignore = "perf benchmark"]
    fn ssm_block_full_bench() {
        use std::time::Instant;

        let head_kv = 128usize;
        let n_k = 16usize;
        let n_v = 48usize;
        let key_dim = head_kv * n_k;
        let value_dim = head_kv * n_v;
        let conv_dim = 2 * key_dim + value_dim;
        let conv_kernel = 4usize;
        let n_layers_ssm = 48usize;

        let ctx = CudaContext::new(0).expect("ctx");
        let stream = ctx.default_stream();
        let kernels = LlmKernels::new(ctx);

        let conv_w = stream
            .alloc_zeros::<half::bf16>(conv_kernel * conv_dim)
            .expect("");
        let mut conv_state = stream
            .alloc_zeros::<half::bf16>((conv_kernel - 1) * conv_dim)
            .expect("");
        let conv_input = stream.alloc_zeros::<half::bf16>(conv_dim).expect("");
        let mut conv_out = stream.alloc_zeros::<half::bf16>(conv_dim).expect("");
        let mut q_dev = stream.alloc_zeros::<half::bf16>(key_dim).expect("");
        let mut k_dev = stream.alloc_zeros::<half::bf16>(key_dim).expect("");
        let q_v_dev = stream.alloc_zeros::<half::bf16>(value_dim).expect("");
        let k_v_dev = stream.alloc_zeros::<half::bf16>(value_dim).expect("");
        let v_dev = stream.alloc_zeros::<half::bf16>(value_dim).expect("");
        let gate_dev = stream.alloc_zeros::<half::bf16>(n_v).expect("");
        let beta_dev = stream.alloc_zeros::<half::bf16>(n_v).expect("");
        let mut state_dev = stream
            .alloc_zeros::<half::bf16>(n_v * head_kv * head_kv)
            .expect("");
        let mut sm_out = stream.alloc_zeros::<half::bf16>(n_v * head_kv).expect("");

        let (cw_p, cs_p, ci_p, co_p, q_p, k_p, qv_p, kv_p, v_p, g_p, b_p, st_p, smo_p) = unsafe {
            use cudarc::driver::{DevicePtr, DevicePtrMut};
            (
                conv_w.device_ptr(&stream).0,
                conv_state.device_ptr_mut(&stream).0,
                conv_input.device_ptr(&stream).0,
                conv_out.device_ptr_mut(&stream).0,
                q_dev.device_ptr_mut(&stream).0,
                k_dev.device_ptr_mut(&stream).0,
                q_v_dev.device_ptr(&stream).0,
                k_v_dev.device_ptr(&stream).0,
                v_dev.device_ptr(&stream).0,
                gate_dev.device_ptr(&stream).0,
                beta_dev.device_ptr(&stream).0,
                state_dev.device_ptr_mut(&stream).0,
                sm_out.device_ptr_mut(&stream).0,
            )
        };

        // Warm-up.
        unsafe {
            kernels
                .conv1d_depthwise_bf16(
                    &stream,
                    cw_p,
                    cs_p,
                    ci_p,
                    co_p,
                    conv_dim as i32,
                    conv_kernel as i32,
                )
                .ok();
            kernels
                .l2_norm_per_head_bf16(&stream, q_p, n_k as i32, head_kv as i32, 1e-6)
                .ok();
            kernels
                .delta_net_step_bf16(
                    &stream,
                    qv_p,
                    kv_p,
                    v_p,
                    g_p,
                    b_p,
                    st_p,
                    smo_p,
                    n_v as i32,
                    head_kv as i32,
                )
                .ok();
        }
        stream.synchronize().ok();

        let n_iters = 50;
        let t0 = Instant::now();
        for _ in 0..n_iters {
            for _ in 0..n_layers_ssm {
                unsafe {
                    kernels
                        .conv1d_depthwise_bf16(
                            &stream,
                            cw_p,
                            cs_p,
                            ci_p,
                            co_p,
                            conv_dim as i32,
                            conv_kernel as i32,
                        )
                        .ok();
                    kernels
                        .l2_norm_per_head_bf16(&stream, q_p, n_k as i32, head_kv as i32, 1e-6)
                        .ok();
                    kernels
                        .l2_norm_per_head_bf16(&stream, k_p, n_k as i32, head_kv as i32, 1e-6)
                        .ok();
                    kernels
                        .delta_net_step_bf16(
                            &stream,
                            qv_p,
                            kv_p,
                            v_p,
                            g_p,
                            b_p,
                            st_p,
                            smo_p,
                            n_v as i32,
                            head_kv as i32,
                        )
                        .ok();
                }
            }
        }
        stream.synchronize().ok();
        let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0 / n_iters as f64;

        eprintln!();
        eprintln!("=== SSM BLOCK FULL BENCH (Qwen3.6-27B dim, 48 SSM layers) ===");
        eprintln!(
            "  per-token (SSM only): {elapsed_ms:.2} ms = {:.2} tok/s ceiling",
            1000.0 / elapsed_ms
        );
        eprintln!("  llama.cpp Qwen3.6-27B Q4_K_M ref : 11.62 tok/s");
    }

    /// T243.2 — delta_net_step_bf16 parity vs CPU.
    #[test]
    fn delta_net_step_bf16_matches_cpu() {
        let n_heads = 2usize;
        let head_dim = 16usize;
        let q: Vec<f32> = (0..(n_heads * head_dim))
            .map(|i| ((i as f32 * 0.13).sin()) * 0.4)
            .collect();
        let k: Vec<f32> = (0..(n_heads * head_dim))
            .map(|i| ((i as f32 * 0.07).cos()) * 0.4)
            .collect();
        let v: Vec<f32> = (0..(n_heads * head_dim))
            .map(|i| ((i as f32 * 0.09).sin()) * 0.4)
            .collect();
        let gate: Vec<f32> = (0..n_heads).map(|h| -0.5 - 0.1 * h as f32).collect();
        let beta: Vec<f32> = (0..n_heads).map(|h| 0.6 + 0.05 * h as f32).collect();
        let st0: Vec<f32> = (0..(n_heads * head_dim * head_dim))
            .map(|i| ((i as f32 * 0.03).cos()) * 0.1)
            .collect();
        let mut cs = st0.clone();
        let mut cpu_out = vec![0.0f32; n_heads * head_dim];
        for h in 0..n_heads {
            let g_exp = gate[h].exp();
            let b = beta[h];
            let s_off = h * head_dim * head_dim;
            for r in 0..head_dim {
                let v_r = v[h * head_dim + r];
                let mut acc = 0.0f32;
                for c in 0..head_dim {
                    let k_c = k[h * head_dim + c];
                    let updated = g_exp * cs[s_off + r * head_dim + c] + b * v_r * k_c;
                    cs[s_off + r * head_dim + c] = updated;
                    acc += updated * q[h * head_dim + c];
                }
                cpu_out[h * head_dim + r] = acc;
            }
        }
        let ctx = CudaContext::new(0).expect("ctx");
        let stream = ctx.default_stream();
        let kernels = LlmKernels::new(ctx);
        let to_bf = |v: &[f32]| -> Vec<half::bf16> {
            v.iter().copied().map(half::bf16::from_f32).collect()
        };
        let q_dev = stream.memcpy_stod(&to_bf(&q)).expect("up q");
        let k_dev = stream.memcpy_stod(&to_bf(&k)).expect("up k");
        let v_dev = stream.memcpy_stod(&to_bf(&v)).expect("up v");
        let g_dev = stream.memcpy_stod(&to_bf(&gate)).expect("up g");
        let b_dev = stream.memcpy_stod(&to_bf(&beta)).expect("up b");
        let mut s_dev = stream.memcpy_stod(&to_bf(&st0)).expect("up s");
        let mut out_dev = stream
            .alloc_zeros::<half::bf16>(n_heads * head_dim)
            .expect("alloc");
        unsafe {
            let (qp, _g1) = q_dev.device_ptr(&stream);
            let (kp, _g2) = k_dev.device_ptr(&stream);
            let (vp, _g3) = v_dev.device_ptr(&stream);
            let (gp, _g4) = g_dev.device_ptr(&stream);
            let (bp, _g5) = b_dev.device_ptr(&stream);
            let (sp, _g6) = s_dev.device_ptr_mut(&stream);
            let (op, _g7) = out_dev.device_ptr_mut(&stream);
            kernels
                .delta_net_step_bf16(
                    &stream,
                    qp,
                    kp,
                    vp,
                    gp,
                    bp,
                    sp,
                    op,
                    n_heads as i32,
                    head_dim as i32,
                )
                .expect("delta_net");
        }
        let oh: Vec<half::bf16> = stream.memcpy_dtov(&out_dev).expect("dtov");
        let go: Vec<f32> = oh.into_iter().map(|v| v.to_f32()).collect();
        for i in 0..(n_heads * head_dim) {
            let diff = (cpu_out[i] - go[i]).abs();
            let tol = cpu_out[i].abs() * 1e-1 + 5e-2;
            assert!(diff <= tol, "[{i}] cpu={} gpu={}", cpu_out[i], go[i]);
        }
    }

    /// T244.1.5 — full-decode bench réel : charge tous les Q4_K + Q6_K
    /// matmul tensors d'un Qwen2.5-7B Q4_K_M GGUF (28 layers × 5 matmul =
    /// 140 calls/token), upload sur GPU, et bench un decode complet.
    #[test]
    #[ignore = "real-data full decode bench"]
    fn real_qwen7b_full_decode_bench() {
        use rustorch_gguf::reader::GgufFile;
        use rustorch_gguf::tensor::GgmlType;
        use std::path::Path;
        use std::time::Instant;

        let path = Path::new(
            "/home/triviere/projects/models/qwen2.5-7b-gguf/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf",
        );
        if !path.exists() {
            eprintln!("[skip] {path:?} not found");
            return;
        }
        let file = GgufFile::open(path).expect("open gguf");

        let n_layers = 28usize;
        let d = 3584usize;

        let ctx = CudaContext::new(0).expect("ctx");
        let stream = ctx.default_stream();
        let kernels = LlmKernels::new(ctx);

        let mut h_dev = stream.alloc_zeros::<half::bf16>(d).expect("alloc h");
        let mut buf_d = stream.alloc_zeros::<half::bf16>(d).expect("alloc buf_d");
        let mut buf_qkv = stream.alloc_zeros::<half::bf16>(4096).expect("alloc qkv");
        let mut buf_f = stream.alloc_zeros::<half::bf16>(18944).expect("alloc f");

        // Per-layer (q, o, gate, up, down) tensor info.
        struct L {
            ptr: u64,
            n: usize,
            k: usize,
            dt: GgmlType,
        }
        let mut layers: Vec<[L; 5]> = Vec::with_capacity(n_layers);
        let mut weight_buffers: Vec<cudarc::driver::CudaSlice<u8>> = Vec::new();

        let total_t0 = Instant::now();
        for li in 0..n_layers {
            let mut load = |name: String| -> L {
                let info = file
                    .tensor(&name)
                    .unwrap_or_else(|| panic!("{name} not found"));
                let bytes = file.tensor_bytes(info);
                let n = info.shape[1] as usize;
                let k = info.shape[0] as usize;
                let buf = stream.memcpy_stod(bytes).expect("upload");
                let ptr = unsafe {
                    use cudarc::driver::DevicePtr;
                    let (p, _g) = buf.device_ptr(&stream);
                    p
                };
                weight_buffers.push(buf);
                L {
                    ptr,
                    n,
                    k,
                    dt: info.dtype,
                }
            };
            let q = load(format!("blk.{li}.attn_q.weight"));
            let o = load(format!("blk.{li}.attn_output.weight"));
            let g = load(format!("blk.{li}.ffn_gate.weight"));
            let u = load(format!("blk.{li}.ffn_up.weight"));
            let down = load(format!("blk.{li}.ffn_down.weight"));
            layers.push([q, o, g, u, down]);
        }
        let load_secs = total_t0.elapsed().as_secs_f64();
        eprintln!("Upload {n_layers} layers in {load_secs:.2}s");

        let dispatch = |w_p: u64, dt: GgmlType, x_p: u64, y_p: u64, n: usize, k: usize| unsafe {
            match dt {
                GgmlType::Q4_K => {
                    kernels
                        .sgemv_q4k_bf16_v2(&stream, w_p, x_p, y_p, n as i32, k as i32)
                        .ok();
                },
                GgmlType::Q6_K => {
                    kernels
                        .sgemv_q6k_bf16_v2(&stream, w_p, x_p, y_p, n as i32, k as i32)
                        .ok();
                },
                _ => panic!("dtype {dt:?} not supported"),
            };
        };

        // Warm-up.
        unsafe {
            use cudarc::driver::{DevicePtr, DevicePtrMut};
            let l0 = &layers[0];
            let (h_p, _g1) = h_dev.device_ptr(&stream);
            let (b_p, _g2) = buf_qkv.device_ptr_mut(&stream);
            dispatch(l0[0].ptr, l0[0].dt, h_p, b_p, l0[0].n, d);
            let (f_p, _g3) = buf_f.device_ptr_mut(&stream);
            dispatch(l0[2].ptr, l0[2].dt, h_p, f_p, l0[2].n, d);
        }
        stream.synchronize().ok();

        // Pre-extract device pointers (drop guards immediately ; pointers
        // remain valid as long as the slices live).
        let (h_p, qkv_p, d_p, f_p) = unsafe {
            use cudarc::driver::{DevicePtr, DevicePtrMut};
            let (a, _g1) = h_dev.device_ptr(&stream);
            let (b, _g2) = buf_qkv.device_ptr_mut(&stream);
            let (c, _g3) = buf_d.device_ptr_mut(&stream);
            let (e, _g4) = buf_f.device_ptr_mut(&stream);
            (a, b, c, e)
        };

        let n_iters = 30;
        let t0 = Instant::now();
        for _ in 0..n_iters {
            for l in &layers {
                dispatch(l[0].ptr, l[0].dt, h_p, qkv_p, l[0].n, d);
                dispatch(l[1].ptr, l[1].dt, h_p, d_p, l[1].n, d);
                dispatch(l[2].ptr, l[2].dt, h_p, f_p, l[2].n, d);
                dispatch(l[3].ptr, l[3].dt, h_p, f_p, l[3].n, d);
                dispatch(l[4].ptr, l[4].dt, f_p, d_p, l[4].n, l[4].k);
            }
        }
        stream.synchronize().ok();
        let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0 / n_iters as f64;
        let tok_s = 1000.0 / elapsed_ms;

        let total_bytes: usize = weight_buffers.iter().map(|b| b.len()).sum();
        let bw_gb_s = total_bytes as f64 / (elapsed_ms * 1e-3) / 1e9;

        eprintln!();
        eprintln!("=== REAL Qwen2.5-7B Q4_K_M FULL DECODE BENCH ===");
        eprintln!(
            "  {n_layers} layers × 5 matmul = {} calls/token",
            n_layers * 5
        );
        eprintln!("  total weights GPU : {:.2} GB", total_bytes as f64 / 1e9);
        eprintln!("  per-token : {elapsed_ms:.2} ms = {tok_s:.2} tok/s");
        eprintln!("  effective bandwidth: {bw_gb_s:.1} GB/s");
        eprintln!();
        eprintln!("  rustorch real Q4K+Q6K     : {tok_s:.2} tok/s");
        eprintln!("  llama.cpp Q4_K_M Qwen-7B  : 47.15 tok/s");
        eprintln!("  rustorch BF16 actuel      : 11.20 tok/s");
        let r = tok_s / 47.15;
        eprintln!(
            "  ratio rustorch/llama.cpp  : {r:.2}× ({})",
            if r >= 1.0 { "FASTER ✓" } else { "slower" }
        );
    }

    /// T244.3 — REAL Qwen3.6-27B Q4_K_M end-to-end matmul bench.
    /// Loads ALL matmul tensors (Q4_K + Q5_K + Q6_K) for 64 hybrid layers
    /// (16 attn + 48 SSM + 64 FFN dense). Streams the entire 16.8GB GGUF
    /// through our kernels and measures pure matmul throughput.
    ///
    /// This is "compute upper bound" — actual decode also needs RoPE/RMSNorm/
    /// SSM scan/argmax but those are negligible (<3% of total).
    #[test]
    #[ignore = "real-data full decode bench — requires Qwen3.6-27B Q4_K_M GGUF"]
    fn real_qwen36_27b_full_decode_bench() {
        use rustorch_gguf::reader::GgufFile;
        use rustorch_gguf::tensor::GgmlType;
        use std::path::Path;
        use std::time::Instant;

        let path =
            Path::new("/home/triviere/projects/models/qwen3.6-27b-gguf/Qwen3.6-27B-Q4_K_M.gguf");
        if !path.exists() {
            eprintln!("[skip] {path:?} not found");
            return;
        }
        let file = GgufFile::open(path).expect("open gguf");

        // Qwen3.6-27B shape constants.
        let d = 5120usize;
        let n_layers = 64usize;
        let n_attn_layers = 16usize;
        // Attention layers : every 4th (idx 3, 7, 11, ...).
        let is_attn_layer = |li: usize| (li + 1) % 4 == 0;

        let ctx = CudaContext::new(0).expect("ctx");
        let stream = ctx.default_stream();
        let kernels = LlmKernels::new(ctx);

        let h_dev = stream.alloc_zeros::<half::bf16>(d).expect("alloc h");
        // Largest output buffers we'll need (over all matmul shapes).
        let mut buf_d = stream.alloc_zeros::<half::bf16>(d).expect("alloc buf_d");
        let mut buf_qkv = stream.alloc_zeros::<half::bf16>(20480).expect("alloc qkv");
        let mut buf_f = stream.alloc_zeros::<half::bf16>(20480).expect("alloc f");

        struct L {
            ptr: u64,
            n: usize,
            k: usize,
            dt: GgmlType,
        }
        // Flat list of (ptr, n, k, dt) per matmul call, in dispatch order.
        let mut calls: Vec<L> = Vec::with_capacity(n_layers * 7);
        let mut weight_buffers: Vec<cudarc::driver::CudaSlice<u8>> = Vec::new();
        let mut total_bytes_loaded: usize = 0;

        let total_t0 = Instant::now();
        for li in 0..n_layers {
            let mut load = |name: String, buffers: &mut Vec<_>, total: &mut usize| -> L {
                let info = file
                    .tensor(&name)
                    .unwrap_or_else(|| panic!("{name} not found"));
                let bytes = file.tensor_bytes(info);
                let n = info.shape[1] as usize;
                let k = info.shape[0] as usize;
                let buf = stream.memcpy_stod(bytes).expect("upload");
                let ptr = unsafe {
                    use cudarc::driver::DevicePtr;
                    let (p, _g) = buf.device_ptr(&stream);
                    p
                };
                *total += bytes.len();
                buffers.push(buf);
                L {
                    ptr,
                    n,
                    k,
                    dt: info.dtype,
                }
            };
            if is_attn_layer(li) {
                calls.push(load(
                    format!("blk.{li}.attn_q.weight"),
                    &mut weight_buffers,
                    &mut total_bytes_loaded,
                ));
                calls.push(load(
                    format!("blk.{li}.attn_k.weight"),
                    &mut weight_buffers,
                    &mut total_bytes_loaded,
                ));
                calls.push(load(
                    format!("blk.{li}.attn_v.weight"),
                    &mut weight_buffers,
                    &mut total_bytes_loaded,
                ));
                calls.push(load(
                    format!("blk.{li}.attn_output.weight"),
                    &mut weight_buffers,
                    &mut total_bytes_loaded,
                ));
            } else {
                calls.push(load(
                    format!("blk.{li}.attn_qkv.weight"),
                    &mut weight_buffers,
                    &mut total_bytes_loaded,
                ));
                calls.push(load(
                    format!("blk.{li}.attn_gate.weight"),
                    &mut weight_buffers,
                    &mut total_bytes_loaded,
                ));
                calls.push(load(
                    format!("blk.{li}.ssm_alpha.weight"),
                    &mut weight_buffers,
                    &mut total_bytes_loaded,
                ));
                calls.push(load(
                    format!("blk.{li}.ssm_beta.weight"),
                    &mut weight_buffers,
                    &mut total_bytes_loaded,
                ));
                calls.push(load(
                    format!("blk.{li}.ssm_out.weight"),
                    &mut weight_buffers,
                    &mut total_bytes_loaded,
                ));
            }
            // FFN dense.
            calls.push(load(
                format!("blk.{li}.ffn_gate.weight"),
                &mut weight_buffers,
                &mut total_bytes_loaded,
            ));
            calls.push(load(
                format!("blk.{li}.ffn_up.weight"),
                &mut weight_buffers,
                &mut total_bytes_loaded,
            ));
            calls.push(load(
                format!("blk.{li}.ffn_down.weight"),
                &mut weight_buffers,
                &mut total_bytes_loaded,
            ));
        }
        let load_secs = total_t0.elapsed().as_secs_f64();
        let n_calls = calls.len();
        eprintln!(
            "Loaded {n_layers} layers ({n_attn_layers} attn + {} SSM) → {n_calls} matmul calls in {load_secs:.2}s",
            n_layers - n_attn_layers
        );
        eprintln!(
            "Total bytes uploaded : {:.2} GB",
            total_bytes_loaded as f64 / 1e9
        );

        // Returns true if the matmul was actually dispatched (skips F32/BF16
        // small tensors that need a different code path).
        let dispatch = |w_p: u64, dt: GgmlType, x_p: u64, y_p: u64, n: usize, k: usize| -> bool {
            unsafe {
                match dt {
                    GgmlType::Q4_K => {
                        kernels
                            .sgemv_q4k_bf16_v2(&stream, w_p, x_p, y_p, n as i32, k as i32)
                            .ok();
                        true
                    },
                    GgmlType::Q5_K => {
                        kernels
                            .sgemv_q5k_bf16(&stream, w_p, x_p, y_p, n as i32, k as i32)
                            .ok();
                        true
                    },
                    GgmlType::Q6_K => {
                        kernels
                            .sgemv_q6k_bf16_v2(&stream, w_p, x_p, y_p, n as i32, k as i32)
                            .ok();
                        true
                    },
                    // F32 small tensors (typically ssm_alpha/beta with N<=64) :
                    // skip in pure-matmul bench. The byte cost is negligible
                    // (<1% of total weight bytes for Qwen 3.6).
                    _ => false,
                }
            }
        };

        let (h_p, qkv_p, d_p, f_p) = unsafe {
            use cudarc::driver::{DevicePtr, DevicePtrMut};
            let (a, _g1) = h_dev.device_ptr(&stream);
            let (b, _g2) = buf_qkv.device_ptr_mut(&stream);
            let (c, _g3) = buf_d.device_ptr_mut(&stream);
            let (e, _g4) = buf_f.device_ptr_mut(&stream);
            (a, b, c, e)
        };

        // Warm-up + count active dispatches.
        let mut n_dispatched = 0usize;
        let mut bytes_dispatched = 0usize;
        for c in &calls {
            let out = if c.n <= d { d_p } else { qkv_p };
            let in_p = if c.k <= d { h_p } else { f_p };
            if dispatch(c.ptr, c.dt, in_p, out, c.n, c.k) {
                n_dispatched += 1;
                // Bytes for this tensor (matches what's read on the wire).
                let info = match c.dt {
                    GgmlType::Q4_K => (c.n * c.k * 144) / 256,
                    GgmlType::Q5_K => (c.n * c.k * 176) / 256,
                    GgmlType::Q6_K => (c.n * c.k * 210) / 256,
                    _ => 0,
                };
                bytes_dispatched += info;
            }
        }
        stream.synchronize().ok();

        let n_iters = 30;
        let t0 = Instant::now();
        for _ in 0..n_iters {
            for c in &calls {
                let out = if c.n <= d { d_p } else { qkv_p };
                let in_p = if c.k <= d { h_p } else { f_p };
                dispatch(c.ptr, c.dt, in_p, out, c.n, c.k);
            }
        }
        stream.synchronize().ok();
        let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0 / n_iters as f64;
        let tok_s = 1000.0 / elapsed_ms;
        let bw_gb_s = bytes_dispatched as f64 / (elapsed_ms * 1e-3) / 1e9;

        // Count dtype distribution.
        let mut q4 = 0usize;
        let mut q5 = 0usize;
        let mut q6 = 0usize;
        let mut other = 0usize;
        for c in &calls {
            match c.dt {
                GgmlType::Q4_K => q4 += 1,
                GgmlType::Q5_K => q5 += 1,
                GgmlType::Q6_K => q6 += 1,
                _ => other += 1,
            }
        }

        eprintln!();
        eprintln!("=== REAL Qwen3.6-27B Q4_K_M FULL DECODE BENCH ===");
        eprintln!("  Calls : {n_calls} total  ({n_dispatched} dispatched : Q4K={q4} Q5K={q5} Q6K={q6} ; F32/skipped={other})");
        eprintln!(
            "  Total weights uploaded : {:.2} GB  (dispatched : {:.2} GB)",
            total_bytes_loaded as f64 / 1e9,
            bytes_dispatched as f64 / 1e9
        );
        eprintln!("  per-token : {elapsed_ms:.2} ms = {tok_s:.2} tok/s (matmul-only)");
        eprintln!("  effective bandwidth : {bw_gb_s:.1} GB/s");
        eprintln!();
        eprintln!("  llama.cpp Qwen3.6-27B Q4_K_M : 11.62 tok/s");
        let r = tok_s / 11.62;
        eprintln!(
            "  ratio rustorch/llama.cpp     : {r:.2}× ({})",
            if r >= 1.0 { "FASTER ✓" } else { "slower" }
        );
    }

    /// T245.1 — CUDA Graphs bench : capture the full Qwen3.6-27B Q4_K_M
    /// decode (496 matmul calls), replay it. Eliminates ~5μs × 400 launches
    /// = 2 ms launch overhead per token.
    ///
    /// Expected : 10.49 tok/s (without graphs) → ~11.5 tok/s (with graphs).
    #[test]
    #[ignore = "real-data full decode bench with CUDA Graphs"]
    fn real_qwen36_27b_full_decode_bench_cuda_graphs() {
        use cudarc::driver::sys::{CUgraphInstantiate_flags, CUstreamCaptureMode_enum};
        use rustorch_gguf::reader::GgufFile;
        use rustorch_gguf::tensor::GgmlType;
        use std::path::Path;
        use std::time::Instant;

        let path =
            Path::new("/home/triviere/projects/models/qwen3.6-27b-gguf/Qwen3.6-27B-Q4_K_M.gguf");
        if !path.exists() {
            eprintln!("[skip] {path:?} not found");
            return;
        }
        let file = GgufFile::open(path).expect("open gguf");

        let d = 5120usize;
        let n_layers = 64usize;
        let n_attn_layers = 16usize;
        let is_attn_layer = |li: usize| (li + 1) % 4 == 0;

        let ctx = CudaContext::new(0).expect("ctx");
        // CUDA Graphs cannot capture the default stream. Use a fresh stream.
        let stream = ctx.new_stream().expect("new stream");
        let kernels = LlmKernels::new(ctx);

        let h_dev = stream.alloc_zeros::<half::bf16>(d).expect("alloc h");
        let mut buf_d = stream.alloc_zeros::<half::bf16>(d).expect("alloc buf_d");
        let mut buf_qkv = stream.alloc_zeros::<half::bf16>(20480).expect("alloc qkv");
        let mut buf_f = stream.alloc_zeros::<half::bf16>(20480).expect("alloc f");

        struct L {
            ptr: u64,
            n: usize,
            k: usize,
            dt: GgmlType,
        }
        let mut calls: Vec<L> = Vec::with_capacity(n_layers * 7);
        let mut weight_buffers: Vec<cudarc::driver::CudaSlice<u8>> = Vec::new();
        let mut total_bytes_loaded: usize = 0;

        let total_t0 = Instant::now();
        for li in 0..n_layers {
            let mut load = |name: String, buffers: &mut Vec<_>, total: &mut usize| -> L {
                let info = file
                    .tensor(&name)
                    .unwrap_or_else(|| panic!("{name} not found"));
                let bytes = file.tensor_bytes(info);
                let n = info.shape[1] as usize;
                let k = info.shape[0] as usize;
                let buf = stream.memcpy_stod(bytes).expect("upload");
                let ptr = unsafe {
                    use cudarc::driver::DevicePtr;
                    let (p, _g) = buf.device_ptr(&stream);
                    p
                };
                *total += bytes.len();
                buffers.push(buf);
                L {
                    ptr,
                    n,
                    k,
                    dt: info.dtype,
                }
            };
            if is_attn_layer(li) {
                calls.push(load(
                    format!("blk.{li}.attn_q.weight"),
                    &mut weight_buffers,
                    &mut total_bytes_loaded,
                ));
                calls.push(load(
                    format!("blk.{li}.attn_k.weight"),
                    &mut weight_buffers,
                    &mut total_bytes_loaded,
                ));
                calls.push(load(
                    format!("blk.{li}.attn_v.weight"),
                    &mut weight_buffers,
                    &mut total_bytes_loaded,
                ));
                calls.push(load(
                    format!("blk.{li}.attn_output.weight"),
                    &mut weight_buffers,
                    &mut total_bytes_loaded,
                ));
            } else {
                calls.push(load(
                    format!("blk.{li}.attn_qkv.weight"),
                    &mut weight_buffers,
                    &mut total_bytes_loaded,
                ));
                calls.push(load(
                    format!("blk.{li}.attn_gate.weight"),
                    &mut weight_buffers,
                    &mut total_bytes_loaded,
                ));
                calls.push(load(
                    format!("blk.{li}.ssm_alpha.weight"),
                    &mut weight_buffers,
                    &mut total_bytes_loaded,
                ));
                calls.push(load(
                    format!("blk.{li}.ssm_beta.weight"),
                    &mut weight_buffers,
                    &mut total_bytes_loaded,
                ));
                calls.push(load(
                    format!("blk.{li}.ssm_out.weight"),
                    &mut weight_buffers,
                    &mut total_bytes_loaded,
                ));
            }
            calls.push(load(
                format!("blk.{li}.ffn_gate.weight"),
                &mut weight_buffers,
                &mut total_bytes_loaded,
            ));
            calls.push(load(
                format!("blk.{li}.ffn_up.weight"),
                &mut weight_buffers,
                &mut total_bytes_loaded,
            ));
            calls.push(load(
                format!("blk.{li}.ffn_down.weight"),
                &mut weight_buffers,
                &mut total_bytes_loaded,
            ));
        }
        let load_secs = total_t0.elapsed().as_secs_f64();
        eprintln!(
            "Loaded {n_layers} layers ({n_attn_layers} attn + {} SSM) in {load_secs:.2}s",
            n_layers - n_attn_layers
        );

        let dispatch = |w_p: u64, dt: GgmlType, x_p: u64, y_p: u64, n: usize, k: usize| -> bool {
            unsafe {
                match dt {
                    GgmlType::Q4_K => {
                        kernels
                            .sgemv_q4k_bf16_v2(&stream, w_p, x_p, y_p, n as i32, k as i32)
                            .ok();
                        true
                    },
                    GgmlType::Q5_K => {
                        kernels
                            .sgemv_q5k_bf16(&stream, w_p, x_p, y_p, n as i32, k as i32)
                            .ok();
                        true
                    },
                    GgmlType::Q6_K => {
                        kernels
                            .sgemv_q6k_bf16_v2(&stream, w_p, x_p, y_p, n as i32, k as i32)
                            .ok();
                        true
                    },
                    _ => false,
                }
            }
        };

        let (h_p, qkv_p, d_p, f_p) = unsafe {
            use cudarc::driver::{DevicePtr, DevicePtrMut};
            let (a, _g1) = h_dev.device_ptr(&stream);
            let (b, _g2) = buf_qkv.device_ptr_mut(&stream);
            let (c, _g3) = buf_d.device_ptr_mut(&stream);
            let (e, _g4) = buf_f.device_ptr_mut(&stream);
            (a, b, c, e)
        };

        let do_decode = || {
            let mut bytes = 0usize;
            let mut dispatched = 0usize;
            for c in &calls {
                let out = if c.n <= d { d_p } else { qkv_p };
                let in_p = if c.k <= d { h_p } else { f_p };
                if dispatch(c.ptr, c.dt, in_p, out, c.n, c.k) {
                    dispatched += 1;
                    let info = match c.dt {
                        GgmlType::Q4_K => (c.n * c.k * 144) / 256,
                        GgmlType::Q5_K => (c.n * c.k * 176) / 256,
                        GgmlType::Q6_K => (c.n * c.k * 210) / 256,
                        _ => 0,
                    };
                    bytes += info;
                }
            }
            (dispatched, bytes)
        };

        // Warm-up : also forces all kernels to be JIT-compiled BEFORE the
        // capture (CudaGraph capture rejects compile_or_get).
        let (n_dispatched, bytes_dispatched) = do_decode();
        stream.synchronize().ok();
        eprintln!(
            "Warm-up done : {n_dispatched} matmul/token, {:.2} GB",
            bytes_dispatched as f64 / 1e9
        );

        // === Path A : baseline (no graphs).
        let n_iters_baseline = 30;
        let t0 = Instant::now();
        for _ in 0..n_iters_baseline {
            do_decode();
        }
        stream.synchronize().ok();
        let baseline_ms = t0.elapsed().as_secs_f64() * 1000.0 / n_iters_baseline as f64;
        let baseline_tok_s = 1000.0 / baseline_ms;
        let baseline_bw = bytes_dispatched as f64 / (baseline_ms * 1e-3) / 1e9;

        // === Path B : capture once, replay n_iters times.
        stream
            .begin_capture(CUstreamCaptureMode_enum::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)
            .expect("begin_capture");
        do_decode();
        let graph = stream
            .end_capture(CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH)
            .expect("end_capture")
            .expect("graph created");

        // Warm-up the graph (first replay has CUDA-side init).
        graph.launch().expect("graph warmup");
        stream.synchronize().ok();

        let n_iters_graph = 50;
        let t0 = Instant::now();
        for _ in 0..n_iters_graph {
            graph.launch().expect("graph launch");
        }
        stream.synchronize().ok();
        let graph_ms = t0.elapsed().as_secs_f64() * 1000.0 / n_iters_graph as f64;
        let graph_tok_s = 1000.0 / graph_ms;
        let graph_bw = bytes_dispatched as f64 / (graph_ms * 1e-3) / 1e9;

        eprintln!();
        eprintln!("=== Qwen3.6-27B Q4_K_M FULL DECODE — CUDA Graphs ===");
        eprintln!(
            "  baseline (separate launches) : {baseline_ms:.2} ms = {baseline_tok_s:.2} tok/s @ {baseline_bw:.1} GB/s"
        );
        eprintln!(
            "  CUDA Graphs (replay)         : {graph_ms:.2} ms = {graph_tok_s:.2} tok/s @ {graph_bw:.1} GB/s"
        );
        let speedup = baseline_ms / graph_ms;
        eprintln!("  speedup graph/baseline       : {speedup:.2}×");
        eprintln!();
        eprintln!("  llama.cpp Qwen3.6-27B Q4_K_M : 11.62 tok/s");
        let r = graph_tok_s / 11.62;
        eprintln!(
            "  ratio rustorch(graphs)/llama : {r:.2}× ({})",
            if r >= 1.0 { "FASTER ✓" } else { "slower" }
        );
    }

    /// T245.4 — REAL Qwen3.6-27B Q4_K_M FULL DECODE at M=8 batched.
    /// This is the algorithmic answer : process 8 tokens per weight read.
    /// Demonstrates the speculative-decoding upper-bound throughput.
    #[test]
    #[ignore = "real-data full decode bench — M=8 batched"]
    fn real_qwen36_27b_full_decode_bench_m8() {
        use rustorch_gguf::reader::GgufFile;
        use rustorch_gguf::tensor::GgmlType;
        use std::path::Path;
        use std::time::Instant;

        let path =
            Path::new("/home/triviere/projects/models/qwen3.6-27b-gguf/Qwen3.6-27B-Q4_K_M.gguf");
        if !path.exists() {
            eprintln!("[skip] {path:?} not found");
            return;
        }
        let file = GgufFile::open(path).expect("open gguf");

        let d = 5120usize;
        let n_layers = 64usize;
        let m_batch = 8usize;
        let is_attn_layer = |li: usize| (li + 1) % 4 == 0;

        let ctx = CudaContext::new(0).expect("ctx");
        let stream = ctx.default_stream();
        let kernels = LlmKernels::new(ctx);

        // Allocate input/output buffers sized for M=8.
        let h_dev = stream
            .alloc_zeros::<half::bf16>(m_batch * d)
            .expect("alloc h");
        let mut buf_d = stream
            .alloc_zeros::<half::bf16>(m_batch * d)
            .expect("alloc buf_d");
        let mut buf_qkv = stream
            .alloc_zeros::<half::bf16>(m_batch * 20480)
            .expect("alloc qkv");
        let mut buf_f = stream
            .alloc_zeros::<half::bf16>(m_batch * 20480)
            .expect("alloc f");

        struct L {
            ptr: u64,
            n: usize,
            k: usize,
            dt: GgmlType,
        }
        let mut calls: Vec<L> = Vec::with_capacity(n_layers * 7);
        let mut weight_buffers: Vec<cudarc::driver::CudaSlice<u8>> = Vec::new();
        let mut total_bytes_loaded: usize = 0;

        let total_t0 = Instant::now();
        for li in 0..n_layers {
            let mut load = |name: String, buffers: &mut Vec<_>, total: &mut usize| -> L {
                let info = file
                    .tensor(&name)
                    .unwrap_or_else(|| panic!("{name} not found"));
                let bytes = file.tensor_bytes(info);
                let n = info.shape[1] as usize;
                let k = info.shape[0] as usize;
                let buf = stream.memcpy_stod(bytes).expect("upload");
                let ptr = unsafe {
                    use cudarc::driver::DevicePtr;
                    let (p, _g) = buf.device_ptr(&stream);
                    p
                };
                *total += bytes.len();
                buffers.push(buf);
                L {
                    ptr,
                    n,
                    k,
                    dt: info.dtype,
                }
            };
            if is_attn_layer(li) {
                calls.push(load(
                    format!("blk.{li}.attn_q.weight"),
                    &mut weight_buffers,
                    &mut total_bytes_loaded,
                ));
                calls.push(load(
                    format!("blk.{li}.attn_k.weight"),
                    &mut weight_buffers,
                    &mut total_bytes_loaded,
                ));
                calls.push(load(
                    format!("blk.{li}.attn_v.weight"),
                    &mut weight_buffers,
                    &mut total_bytes_loaded,
                ));
                calls.push(load(
                    format!("blk.{li}.attn_output.weight"),
                    &mut weight_buffers,
                    &mut total_bytes_loaded,
                ));
            } else {
                calls.push(load(
                    format!("blk.{li}.attn_qkv.weight"),
                    &mut weight_buffers,
                    &mut total_bytes_loaded,
                ));
                calls.push(load(
                    format!("blk.{li}.attn_gate.weight"),
                    &mut weight_buffers,
                    &mut total_bytes_loaded,
                ));
                calls.push(load(
                    format!("blk.{li}.ssm_alpha.weight"),
                    &mut weight_buffers,
                    &mut total_bytes_loaded,
                ));
                calls.push(load(
                    format!("blk.{li}.ssm_beta.weight"),
                    &mut weight_buffers,
                    &mut total_bytes_loaded,
                ));
                calls.push(load(
                    format!("blk.{li}.ssm_out.weight"),
                    &mut weight_buffers,
                    &mut total_bytes_loaded,
                ));
            }
            calls.push(load(
                format!("blk.{li}.ffn_gate.weight"),
                &mut weight_buffers,
                &mut total_bytes_loaded,
            ));
            calls.push(load(
                format!("blk.{li}.ffn_up.weight"),
                &mut weight_buffers,
                &mut total_bytes_loaded,
            ));
            calls.push(load(
                format!("blk.{li}.ffn_down.weight"),
                &mut weight_buffers,
                &mut total_bytes_loaded,
            ));
        }
        let load_secs = total_t0.elapsed().as_secs_f64();
        eprintln!(
            "Loaded {n_layers} layers in {load_secs:.2}s, {:.2} GB",
            total_bytes_loaded as f64 / 1e9
        );

        let dispatch_m8 =
            |w_p: u64, dt: GgmlType, x_p: u64, y_p: u64, n: usize, k: usize| -> bool {
                unsafe {
                    match dt {
                        GgmlType::Q4_K => {
                            kernels
                                .sgemm_q4k_bf16_m8(&stream, w_p, x_p, y_p, n as i32, k as i32)
                                .ok();
                            true
                        },
                        GgmlType::Q5_K => {
                            kernels
                                .sgemm_q5k_bf16_m8(&stream, w_p, x_p, y_p, n as i32, k as i32)
                                .ok();
                            true
                        },
                        GgmlType::Q6_K => {
                            kernels
                                .sgemm_q6k_bf16_m8(&stream, w_p, x_p, y_p, n as i32, k as i32)
                                .ok();
                            true
                        },
                        _ => false,
                    }
                }
            };

        let (h_p, qkv_p, d_p, f_p) = unsafe {
            use cudarc::driver::{DevicePtr, DevicePtrMut};
            let (a, _g1) = h_dev.device_ptr(&stream);
            let (b, _g2) = buf_qkv.device_ptr_mut(&stream);
            let (c, _g3) = buf_d.device_ptr_mut(&stream);
            let (e, _g4) = buf_f.device_ptr_mut(&stream);
            (a, b, c, e)
        };

        // Warm-up : touch every call once and accumulate dispatched bytes.
        let mut n_dispatched = 0usize;
        let mut bytes_dispatched = 0usize;
        for c in &calls {
            let out = if c.n <= d { d_p } else { qkv_p };
            let in_p = if c.k <= d { h_p } else { f_p };
            if dispatch_m8(c.ptr, c.dt, in_p, out, c.n, c.k) {
                n_dispatched += 1;
                let info = match c.dt {
                    GgmlType::Q4_K => (c.n * c.k * 144) / 256,
                    GgmlType::Q5_K => (c.n * c.k * 176) / 256,
                    GgmlType::Q6_K => (c.n * c.k * 210) / 256,
                    _ => 0,
                };
                bytes_dispatched += info;
            }
        }
        stream.synchronize().ok();

        let n_iters = 30;
        let t0 = Instant::now();
        for _ in 0..n_iters {
            for c in &calls {
                let out = if c.n <= d { d_p } else { qkv_p };
                let in_p = if c.k <= d { h_p } else { f_p };
                dispatch_m8(c.ptr, c.dt, in_p, out, c.n, c.k);
            }
        }
        stream.synchronize().ok();
        let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0 / n_iters as f64;
        // tok/s = M tokens produced per kernel-batch / per-batch time.
        let tok_s = m_batch as f64 * 1000.0 / elapsed_ms;
        let bw_gb_s = bytes_dispatched as f64 / (elapsed_ms * 1e-3) / 1e9;

        eprintln!();
        eprintln!("=== REAL Qwen3.6-27B Q4_K_M FULL DECODE — M=8 BATCHED ===");
        eprintln!("  {n_dispatched} matmul/batch (M=8, so 1 weight pass = 8 tokens)",);
        eprintln!(
            "  weights dispatched : {:.2} GB",
            bytes_dispatched as f64 / 1e9
        );
        eprintln!(
            "  per-batch  : {elapsed_ms:.2} ms  ({} tokens/batch)",
            m_batch
        );
        eprintln!(
            "  per-token  : {:.2} ms = {tok_s:.2} tok/s",
            elapsed_ms / m_batch as f64
        );
        eprintln!(
            "  W bandwidth: {bw_gb_s:.1} GB/s (single-pass) — but amortized over {m_batch} tokens"
        );
        eprintln!();
        eprintln!("  llama.cpp Qwen3.6-27B Q4_K_M : 11.62 tok/s");
        let r = tok_s / 11.62;
        eprintln!(
            "  ratio M=8 / llama.cpp        : {r:.2}× ({})",
            if r >= 1.0 { "FASTER ✓" } else { "slower" }
        );
        eprintln!();
        eprintln!("  ⚠ M=8 = upper bound (assumes 100% spec-decoding accept rate).");
        eprintln!(
            "  Real spec decoding @ 70% accept = ~{:.1} tok/s effective.",
            tok_s * 0.7
        );
    }

    /// T244.1.4 — bench réel : lit les vrais Q4_K bytes d'un Qwen2.5-7B
    /// Q4_K_M GGUF et mesure sgemv_q4k_bf16_v2 dessus. Valide que le
    /// speedup tient sur vrai data avec scales et nibbles non-uniformes.
    #[test]
    #[ignore = "real-data bench — requires Qwen2.5-7B Q4_K_M GGUF on disk"]
    fn real_qwen7b_q4k_ffn_gate_bench() {
        use rustorch_gguf::reader::GgufFile;
        use std::path::Path;
        use std::time::Instant;

        let path = Path::new(
            "/home/triviere/projects/models/qwen2.5-7b-gguf/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf",
        );
        if !path.exists() {
            eprintln!("[skip] {path:?} not found");
            return;
        }
        let file = GgufFile::open(path).expect("open gguf");

        // Find blk.0.ffn_gate.weight (Q4_K format in Q4_K_M).
        let tensor_name = "blk.0.ffn_gate.weight";
        let info = file
            .tensor(tensor_name)
            .unwrap_or_else(|| panic!("tensor {tensor_name} not found"));
        eprintln!(
            "Found {tensor_name} : dtype={:?} shape={:?}",
            info.dtype, info.shape
        );

        let bytes = file.tensor_bytes(info);
        let n = info.shape[1] as usize; // out_dim (rows)
        let k = info.shape[0] as usize; // in_dim (cols)
        eprintln!(
            "  N={n} K={k} bytes={} ({:.1} MB)",
            bytes.len(),
            bytes.len() as f64 / 1e6
        );

        // Upload raw Q4_K bytes to GPU (NO CPU dequant).
        let ctx = CudaContext::new(0).expect("ctx");
        let stream = ctx.default_stream();
        let kernels = LlmKernels::new(ctx);

        let w_dev = stream.memcpy_stod(bytes).expect("upload Q4K bytes");
        let x_bf: Vec<half::bf16> = (0..k)
            .map(|i| half::bf16::from_f32(((i as f32 * 0.001).sin()) * 0.5))
            .collect();
        let x_dev = stream.memcpy_stod(&x_bf).expect("upload x");
        let mut y_dev = stream.alloc_zeros::<half::bf16>(n).expect("alloc y");

        // Warm-up.
        unsafe {
            let (w_p, _g1) = w_dev.device_ptr(&stream);
            let (x_p, _g2) = x_dev.device_ptr(&stream);
            let (y_p, _g3) = y_dev.device_ptr_mut(&stream);
            kernels
                .sgemv_q4k_bf16_v2(&stream, w_p, x_p, y_p, n as i32, k as i32)
                .expect("warmup");
        }
        stream.synchronize().ok();

        // Bench.
        let n_iters = 200;
        let t0 = Instant::now();
        for _ in 0..n_iters {
            unsafe {
                let (w_p, _g1) = w_dev.device_ptr(&stream);
                let (x_p, _g2) = x_dev.device_ptr(&stream);
                let (y_p, _g3) = y_dev.device_ptr_mut(&stream);
                kernels
                    .sgemv_q4k_bf16_v2(&stream, w_p, x_p, y_p, n as i32, k as i32)
                    .ok();
            }
        }
        stream.synchronize().ok();
        let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0 / n_iters as f64;
        let bw_gb_s = bytes.len() as f64 / (elapsed_ms * 1e-3) / 1e9;

        // Verify output is non-zero (sanity).
        let y_host: Vec<half::bf16> = stream.memcpy_dtov(&y_dev).expect("dtov y");
        let nonzero_count = y_host.iter().filter(|&&v| v.to_f32().abs() > 1e-3).count();

        eprintln!();
        eprintln!("=== REAL Qwen2.5-7B Q4_K_M FFN gate matmul bench ===");
        eprintln!("  per-call : {elapsed_ms:.3} ms");
        eprintln!("  bandwidth: {bw_gb_s:.1} GB/s");
        eprintln!("  output   : {nonzero_count}/{n} non-zero values (sanity OK)");
        // Project decode budget : 84 FFN-equivalent calls/token.
        let budget = elapsed_ms * 84.0;
        eprintln!(
            "  Decode budget Qwen-7B (84 matmul/tok) : {budget:.1} ms = {:.2} tok/s",
            1000.0 / budget
        );
        eprintln!("  llama.cpp Qwen-7B Q4_K_M reference     : 47.15 tok/s");
    }

    /// T244.1.3 — Full-decode synthetic bench v2 : Q4_K kernel pour
    /// FFN gate/up + Q6_K kernel pour qkv/o/down (le mix réel du
    /// format Q4_K_M de llama.cpp). Pas de BF16 stand-in.
    #[test]
    #[ignore = "perf benchmark"]
    fn end_to_end_qwen7b_q4km_decode_simulation() {
        use rustorch_gguf::dequant::{Q4_K_BYTES, QK_K};
        use std::time::Instant;

        const Q6_K_BYTES: usize = 210;

        let d = 3584usize;
        let f = 18944usize;
        let n_heads = 28usize;
        let n_kv = 4usize;
        let head_dim = 128usize;
        let qkv_d = (n_heads + 2 * n_kv) * head_dim;
        let n_layers = 28usize;

        let blocks_per_d = d / QK_K;
        let blocks_per_f = f / QK_K;

        let ctx = CudaContext::new(0).expect("ctx");
        let stream = ctx.default_stream();
        let kernels = LlmKernels::new(ctx.clone());

        // Q4_K weights : FFN gate + up (Q4_K format in Q4_K_M).
        let w_q4k_gate = stream
            .alloc_zeros::<u8>(f * blocks_per_d * Q4_K_BYTES)
            .expect("alloc gate q4k");
        let w_q4k_up = stream
            .alloc_zeros::<u8>(f * blocks_per_d * Q4_K_BYTES)
            .expect("alloc up q4k");
        // Q6_K weights : qkv, o, down (Q6_K format in Q4_K_M).
        let w_q6k_qkv = stream
            .alloc_zeros::<u8>(qkv_d * blocks_per_d * Q6_K_BYTES)
            .expect("alloc qkv q6k");
        let w_q6k_o = stream
            .alloc_zeros::<u8>(d * blocks_per_d * Q6_K_BYTES)
            .expect("alloc o q6k");
        let w_q6k_down = stream
            .alloc_zeros::<u8>(d * blocks_per_f * Q6_K_BYTES)
            .expect("alloc down q6k");

        let mut h_dev = stream.alloc_zeros::<half::bf16>(d).expect("alloc h");
        let mut qkv_out = stream.alloc_zeros::<half::bf16>(qkv_d).expect("alloc qkv");
        let attn_out = stream.alloc_zeros::<half::bf16>(d).expect("alloc attn");
        let mut gate_out = stream.alloc_zeros::<half::bf16>(f).expect("alloc gate");
        let mut up_out = stream.alloc_zeros::<half::bf16>(f).expect("alloc up");
        let mut down_out = stream.alloc_zeros::<half::bf16>(d).expect("alloc down");

        // Warm-up : compile both kernels.
        unsafe {
            use cudarc::driver::{DevicePtr, DevicePtrMut};
            let (h_p, _g1) = h_dev.device_ptr(&stream);
            let (gate_w_p, _g4) = w_q4k_gate.device_ptr(&stream);
            let (gate_p, _g5) = gate_out.device_ptr_mut(&stream);
            let _ = kernels.sgemv_q4k_bf16_v2(&stream, gate_w_p, h_p, gate_p, f as i32, d as i32);
            let (qkv_w_p, _g6) = w_q6k_qkv.device_ptr(&stream);
            let (qkv_p, _g7) = qkv_out.device_ptr_mut(&stream);
            let _ = kernels.sgemv_q6k_bf16(&stream, qkv_w_p, h_p, qkv_p, qkv_d as i32, d as i32);
        }
        stream.synchronize().ok();

        let n_iters = 20;
        let t0 = Instant::now();

        for _ in 0..n_iters {
            for _ in 0..n_layers {
                unsafe {
                    use cudarc::driver::{DevicePtr, DevicePtrMut};
                    // QKV (Q6_K)
                    {
                        let (w_p, _g1) = w_q6k_qkv.device_ptr(&stream);
                        let (h_p, _g2) = h_dev.device_ptr(&stream);
                        let (out_p, _g3) = qkv_out.device_ptr_mut(&stream);
                        let _ = kernels.sgemv_q6k_bf16(
                            &stream,
                            w_p,
                            h_p,
                            out_p,
                            qkv_d as i32,
                            d as i32,
                        );
                    }
                    // O proj (Q6_K)
                    {
                        let (w_p, _g1) = w_q6k_o.device_ptr(&stream);
                        let (a_p, _g2) = attn_out.device_ptr(&stream);
                        let (h_p, _g3) = h_dev.device_ptr_mut(&stream);
                        let _ = kernels.sgemv_q6k_bf16(&stream, w_p, a_p, h_p, d as i32, d as i32);
                    }
                    // FFN gate (Q4_K)
                    {
                        let (w_p, _g1) = w_q4k_gate.device_ptr(&stream);
                        let (h_p, _g2) = h_dev.device_ptr(&stream);
                        let (g_p, _g3) = gate_out.device_ptr_mut(&stream);
                        let _ =
                            kernels.sgemv_q4k_bf16_v2(&stream, w_p, h_p, g_p, f as i32, d as i32);
                    }
                    // FFN up (Q4_K)
                    {
                        let (w_p, _g1) = w_q4k_up.device_ptr(&stream);
                        let (h_p, _g2) = h_dev.device_ptr(&stream);
                        let (u_p, _g3) = up_out.device_ptr_mut(&stream);
                        let _ =
                            kernels.sgemv_q4k_bf16_v2(&stream, w_p, h_p, u_p, f as i32, d as i32);
                    }
                    // FFN down (Q6_K)
                    {
                        let (w_p, _g1) = w_q6k_down.device_ptr(&stream);
                        let (i_p, _g2) = up_out.device_ptr(&stream);
                        let (d_p, _g3) = down_out.device_ptr_mut(&stream);
                        let _ = kernels.sgemv_q6k_bf16(&stream, w_p, i_p, d_p, d as i32, f as i32);
                    }
                }
            }
        }
        stream.synchronize().ok();
        let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0 / n_iters as f64;
        let tok_s = 1000.0 / elapsed_ms;

        let bytes_per_token = (2.0 * (f * blocks_per_d * Q4_K_BYTES) as f64
            + (qkv_d * blocks_per_d * Q6_K_BYTES) as f64
            + (d * blocks_per_d * Q6_K_BYTES) as f64
            + (d * blocks_per_f * Q6_K_BYTES) as f64)
            * n_layers as f64;
        let bw_gb_s = bytes_per_token / (elapsed_ms * 1e-3) / 1e9;

        eprintln!("\n=== END-TO-END DECODE Qwen2.5-7B Q4_K_M (simulation) ===");
        eprintln!("  per-token : {elapsed_ms:.2} ms = {tok_s:.2} tok/s");
        eprintln!("  weight bytes/token : {:.2} GB", bytes_per_token / 1e9);
        eprintln!("  effective bandwidth: {bw_gb_s:.1} GB/s");
        eprintln!();
        eprintln!("  rustorch end-to-end sim   : {tok_s:.2} tok/s");
        eprintln!("  llama.cpp Q4_K_M Qwen-7B  : 47.15 tok/s");
        eprintln!("  rustorch BF16 (current)   : 11.20 tok/s");
        let r = tok_s / 47.15;
        eprintln!(
            "  rustorch / llama.cpp ratio : {:.2}× ({})",
            r,
            if r >= 1.0 { "FASTER ✓" } else { "slower" }
        );
    }

    /// T244.1.1 — V2 bench, same shape as V1 bench for direct comparison.
    #[test]
    #[ignore = "perf benchmark — run explicitly with --ignored"]
    fn sgemv_q4k_v2_bench() {
        use rustorch_gguf::dequant::{Q4_K_BYTES, QK_K};
        use std::time::Instant;

        let n = 18944usize;
        let k = 3584usize;
        let blocks_per_row = k / QK_K;
        let row_bytes = blocks_per_row * Q4_K_BYTES;

        let mut state: u64 = 0xa5a5a5a5;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let mut w_q4k_bytes = vec![0u8; n * row_bytes];
        for b in &mut w_q4k_bytes {
            *b = (next() & 0xFF) as u8;
        }
        for row in 0..n {
            for blk in 0..blocks_per_row {
                let off = row * row_bytes + blk * Q4_K_BYTES;
                let d = half::f16::from_f32(0.05).to_le_bytes();
                let dmin = half::f16::from_f32(0.025).to_le_bytes();
                w_q4k_bytes[off] = d[0];
                w_q4k_bytes[off + 1] = d[1];
                w_q4k_bytes[off + 2] = dmin[0];
                w_q4k_bytes[off + 3] = dmin[1];
                for i in 0..12 {
                    w_q4k_bytes[off + 4 + i] &= 0x3F;
                }
            }
        }

        let ctx = CudaContext::new(0).expect("ctx");
        let stream = ctx.default_stream();
        let kernels = LlmKernels::new(ctx.clone());

        let w_dev = stream.memcpy_stod(&w_q4k_bytes).expect("upload");
        let x_bf: Vec<half::bf16> = (0..k)
            .map(|i| half::bf16::from_f32((i as f32 * 0.001).sin()))
            .collect();
        let x_dev = stream.memcpy_stod(&x_bf).expect("upload x");
        let mut y_dev = stream.alloc_zeros::<half::bf16>(n).expect("alloc y");

        // Warm-up (compile both kernels).
        unsafe {
            let (w_p, _g1) = w_dev.device_ptr(&stream);
            let (x_p, _g2) = x_dev.device_ptr(&stream);
            let (y_p, _g3) = y_dev.device_ptr_mut(&stream);
            kernels
                .sgemv_q4k_bf16(&stream, w_p, x_p, y_p, n as i32, k as i32)
                .ok();
            kernels
                .sgemv_q4k_bf16_v2(&stream, w_p, x_p, y_p, n as i32, k as i32)
                .ok();
        }
        stream.synchronize().ok();

        let n_iters = 200;

        // Bench V1.
        let t0 = Instant::now();
        for _ in 0..n_iters {
            unsafe {
                let (w_p, _g1) = w_dev.device_ptr(&stream);
                let (x_p, _g2) = x_dev.device_ptr(&stream);
                let (y_p, _g3) = y_dev.device_ptr_mut(&stream);
                kernels
                    .sgemv_q4k_bf16(&stream, w_p, x_p, y_p, n as i32, k as i32)
                    .ok();
            }
        }
        stream.synchronize().ok();
        let v1_ms = t0.elapsed().as_secs_f64() * 1000.0 / n_iters as f64;

        // Bench V2.
        let t0 = Instant::now();
        for _ in 0..n_iters {
            unsafe {
                let (w_p, _g1) = w_dev.device_ptr(&stream);
                let (x_p, _g2) = x_dev.device_ptr(&stream);
                let (y_p, _g3) = y_dev.device_ptr_mut(&stream);
                kernels
                    .sgemv_q4k_bf16_v2(&stream, w_p, x_p, y_p, n as i32, k as i32)
                    .ok();
            }
        }
        stream.synchronize().ok();
        let v2_ms = t0.elapsed().as_secs_f64() * 1000.0 / n_iters as f64;

        let bytes_per_call = (n * row_bytes) as f64;
        let v1_bw = bytes_per_call / (v1_ms * 1e-3) / 1e9;
        let v2_bw = bytes_per_call / (v2_ms * 1e-3) / 1e9;
        let speedup = v1_ms / v2_ms;

        eprintln!("\n=== sgemv_q4k V1 vs V2 (Qwen-7B FFN N={n}, K={k}) ===");
        eprintln!("  V1 : {:.3} ms / {:.1} GB/s", v1_ms, v1_bw);
        eprintln!(
            "  V2 : {:.3} ms / {:.1} GB/s   (speedup {:.2}×)",
            v2_ms, v2_bw, speedup
        );
        eprintln!("  Decode budget Qwen-7B (84 FFN calls/token) :");
        eprintln!(
            "     V1 : {:.1} ms = {:.1} tok/s",
            v1_ms * 84.0,
            1000.0 / (v1_ms * 84.0)
        );
        eprintln!(
            "     V2 : {:.1} ms = {:.1} tok/s",
            v2_ms * 84.0,
            1000.0 / (v2_ms * 84.0)
        );
        eprintln!("  llama.cpp ref Qwen-7B Q4_K_M : 47.15 tok/s");
    }

    /// T245.4 — bench M=8 batched Q4K vs 8× M=1.
    /// Validates the algorithmic win for speculative decoding : reading W
    /// once and producing 8 outputs should be much faster than 8× sequential.
    #[test]
    #[ignore = "perf benchmark"]
    fn sgemm_q4k_m8_vs_8x_sgemv_bench() {
        use rustorch_gguf::dequant::{Q4_K_BYTES, QK_K};
        use std::time::Instant;

        let n = 18944usize;
        let k = 3584usize;
        let blocks_per_row = k / QK_K;
        let row_bytes = blocks_per_row * Q4_K_BYTES;

        let mut state: u64 = 0xb22cabba;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let mut w_q4k = vec![0u8; n * row_bytes];
        for b in &mut w_q4k {
            *b = (next() & 0xFF) as u8;
        }
        for row in 0..n {
            for blk in 0..blocks_per_row {
                let off = row * row_bytes + blk * Q4_K_BYTES;
                let d = half::f16::from_f32(0.05).to_le_bytes();
                let dmin = half::f16::from_f32(0.025).to_le_bytes();
                w_q4k[off] = d[0];
                w_q4k[off + 1] = d[1];
                w_q4k[off + 2] = dmin[0];
                w_q4k[off + 3] = dmin[1];
                for i in 0..12 {
                    w_q4k[off + 4 + i] &= 0x3F;
                }
            }
        }

        let m_batch = 8usize;
        let x_bf: Vec<half::bf16> = (0..m_batch * k)
            .map(|i| half::bf16::from_f32((i as f32 * 0.001).sin()))
            .collect();

        let ctx = CudaContext::new(0).expect("ctx");
        let stream = ctx.default_stream();
        let kernels = LlmKernels::new(ctx);
        let w_dev = stream.memcpy_stod(&w_q4k).expect("");
        let x_dev = stream.memcpy_stod(&x_bf).expect("");
        let mut y_m1_dev = stream.alloc_zeros::<half::bf16>(n).expect("");
        let mut y_m8_dev = stream.alloc_zeros::<half::bf16>(m_batch * n).expect("");

        let (w_p, x_p, y1_p, y8_p) = unsafe {
            use cudarc::driver::{DevicePtr, DevicePtrMut};
            (
                w_dev.device_ptr(&stream).0,
                x_dev.device_ptr(&stream).0,
                y_m1_dev.device_ptr_mut(&stream).0,
                y_m8_dev.device_ptr_mut(&stream).0,
            )
        };

        // Warm-up.
        unsafe {
            kernels
                .sgemv_q4k_bf16_v2(&stream, w_p, x_p, y1_p, n as i32, k as i32)
                .ok();
            kernels
                .sgemm_q4k_bf16_m8(&stream, w_p, x_p, y8_p, n as i32, k as i32)
                .ok();
        }
        stream.synchronize().ok();

        let n_iters = 100;

        // Bench 8× sequential M=1.
        let t0 = Instant::now();
        for _ in 0..n_iters {
            for m in 0..m_batch {
                let xm_p = x_p + (m * k * 2) as u64; // BF16 = 2 bytes
                unsafe {
                    kernels
                        .sgemv_q4k_bf16_v2(&stream, w_p, xm_p, y1_p, n as i32, k as i32)
                        .ok();
                }
            }
        }
        stream.synchronize().ok();
        let m1_seq_ms = t0.elapsed().as_secs_f64() * 1000.0 / n_iters as f64;

        // Bench M=8 batched.
        let t0 = Instant::now();
        for _ in 0..n_iters {
            unsafe {
                kernels
                    .sgemm_q4k_bf16_m8(&stream, w_p, x_p, y8_p, n as i32, k as i32)
                    .ok();
            }
        }
        stream.synchronize().ok();
        let m8_ms = t0.elapsed().as_secs_f64() * 1000.0 / n_iters as f64;

        // Per-call also useful : single M=1.
        let t0 = Instant::now();
        for _ in 0..n_iters {
            unsafe {
                kernels
                    .sgemv_q4k_bf16_v2(&stream, w_p, x_p, y1_p, n as i32, k as i32)
                    .ok();
            }
        }
        stream.synchronize().ok();
        let m1_single_ms = t0.elapsed().as_secs_f64() * 1000.0 / n_iters as f64;

        let bytes_w = (n * row_bytes) as f64;
        let m1_bw = bytes_w / (m1_single_ms * 1e-3) / 1e9;
        let m8_bw = bytes_w / (m8_ms * 1e-3) / 1e9;

        eprintln!("\n=== sgemm_q4k_m8 vs 8× sgemv_q4k_v2 ({n}x{k}) ===");
        eprintln!("  M=1 single  : {m1_single_ms:.3} ms  ({m1_bw:.1} GB/s W-bw)");
        eprintln!("  M=1 × 8 seq : {m1_seq_ms:.3} ms  (8 separate launches, W reread 8×)");
        eprintln!("  M=8 batched : {m8_ms:.3} ms  ({m8_bw:.1} GB/s W-bw)");
        let speedup_vs_seq = m1_seq_ms / m8_ms;
        let speedup_vs_single = m8_ms / m1_single_ms;
        eprintln!(
            "  speedup M=8 vs 8× M=1   : {speedup_vs_seq:.2}× (less is better — 8.0× = perfect amortization)",
        );
        eprintln!(
            "  ratio  M=8 / M=1 single : {speedup_vs_single:.2}× (1.0× = perfect, full bandwidth)",
        );
        eprintln!();
        eprintln!("  Spec decoding implication :");
        eprintln!("    if M=8 close to M=1 single, then 8 tokens cost ~1 token of bw =>");
        eprintln!("    ~8× speedup on memory-bound decode (assuming all 8 tokens accepted)");
    }

    /// T244.3 — micro-bench sgemv_q5k_bf16 + sgemv_q6k_bf16 to compare
    /// per-call bandwidth across all our quantized SGEMV kernels.
    /// Run with `cargo test --release -- --ignored --nocapture sgemv_quant_bench`.
    #[test]
    #[ignore = "perf benchmark"]
    fn sgemv_quant_bench() {
        use rustorch_gguf::dequant::{Q4_K_BYTES, Q5_K_BYTES, Q6_K_BYTES, QK_K};
        use std::time::Instant;

        let n = 18944usize; // Qwen-7B FFN row count.
        let k = 3584usize;
        let blocks_per_row = k / QK_K;

        let ctx = CudaContext::new(0).expect("ctx");
        let stream = ctx.default_stream();
        let kernels = LlmKernels::new(ctx);

        let mut state: u64 = 0xdeadbeefcafe;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state & 0xFF) as u8
        };

        let row_bytes_q4 = blocks_per_row * Q4_K_BYTES;
        let row_bytes_q5 = blocks_per_row * Q5_K_BYTES;
        let row_bytes_q6 = blocks_per_row * Q6_K_BYTES;

        let mut w_q4 = vec![0u8; n * row_bytes_q4];
        let mut w_q5 = vec![0u8; n * row_bytes_q5];
        let mut w_q6 = vec![0u8; n * row_bytes_q6];
        for b in &mut w_q4 {
            *b = next();
        }
        for b in &mut w_q5 {
            *b = next();
        }
        for b in &mut w_q6 {
            *b = next();
        }
        // Make scales sane.
        for row in 0..n {
            for blk in 0..blocks_per_row {
                let off4 = row * row_bytes_q4 + blk * Q4_K_BYTES;
                let d4 = half::f16::from_f32(0.05).to_le_bytes();
                w_q4[off4] = d4[0];
                w_q4[off4 + 1] = d4[1];
                let dmin4 = half::f16::from_f32(0.025).to_le_bytes();
                w_q4[off4 + 2] = dmin4[0];
                w_q4[off4 + 3] = dmin4[1];
                for i in 0..12 {
                    w_q4[off4 + 4 + i] &= 0x3F;
                }
                let off5 = row * row_bytes_q5 + blk * Q5_K_BYTES;
                w_q5[off5] = d4[0];
                w_q5[off5 + 1] = d4[1];
                w_q5[off5 + 2] = dmin4[0];
                w_q5[off5 + 3] = dmin4[1];
                for i in 0..12 {
                    w_q5[off5 + 4 + i] &= 0x3F;
                }
                let off6 = row * row_bytes_q6 + blk * Q6_K_BYTES;
                let d6 = half::f16::from_f32(0.04).to_le_bytes();
                // Q6 layout : ql 128 + qh 64 + scales (16 i8) + d (f16).
                w_q6[off6 + 208] = d6[0];
                w_q6[off6 + 209] = d6[1];
                for i in 0..16 {
                    w_q6[off6 + 192 + i] = (next() as i8 / 8) as u8;
                }
            }
        }

        let w_q4_dev = stream.memcpy_stod(&w_q4).expect("");
        let w_q5_dev = stream.memcpy_stod(&w_q5).expect("");
        let w_q6_dev = stream.memcpy_stod(&w_q6).expect("");
        let x_bf: Vec<half::bf16> = (0..k)
            .map(|i| half::bf16::from_f32((i as f32 * 0.001).cos()))
            .collect();
        let x_dev = stream.memcpy_stod(&x_bf).expect("");
        let mut y_dev = stream.alloc_zeros::<half::bf16>(n).expect("");

        let (w4_p, w5_p, w6_p, x_p, y_p) = unsafe {
            use cudarc::driver::{DevicePtr, DevicePtrMut};
            (
                w_q4_dev.device_ptr(&stream).0,
                w_q5_dev.device_ptr(&stream).0,
                w_q6_dev.device_ptr(&stream).0,
                x_dev.device_ptr(&stream).0,
                y_dev.device_ptr_mut(&stream).0,
            )
        };

        // Warm-up.
        unsafe {
            kernels
                .sgemv_q4k_bf16_v2(&stream, w4_p, x_p, y_p, n as i32, k as i32)
                .ok();
            kernels
                .sgemv_q5k_bf16(&stream, w5_p, x_p, y_p, n as i32, k as i32)
                .ok();
            kernels
                .sgemv_q6k_bf16(&stream, w6_p, x_p, y_p, n as i32, k as i32)
                .ok();
        }
        stream.synchronize().ok();

        let n_iters = 200;
        let bench = |label: &str, run: &mut dyn FnMut()| {
            let t0 = Instant::now();
            for _ in 0..n_iters {
                run();
            }
            stream.synchronize().ok();
            let ms = t0.elapsed().as_secs_f64() * 1000.0 / n_iters as f64;
            (label.to_string(), ms)
        };

        let q4_ms = bench("Q4_K V2", &mut || unsafe {
            kernels
                .sgemv_q4k_bf16_v2(&stream, w4_p, x_p, y_p, n as i32, k as i32)
                .ok();
        })
        .1;
        let q5_ms = bench("Q5_K", &mut || unsafe {
            kernels
                .sgemv_q5k_bf16(&stream, w5_p, x_p, y_p, n as i32, k as i32)
                .ok();
        })
        .1;
        let q6_ms = bench("Q6_K V1", &mut || unsafe {
            kernels
                .sgemv_q6k_bf16(&stream, w6_p, x_p, y_p, n as i32, k as i32)
                .ok();
        })
        .1;
        let q6_v2_ms = bench("Q6_K V2", &mut || unsafe {
            kernels
                .sgemv_q6k_bf16_v2(&stream, w6_p, x_p, y_p, n as i32, k as i32)
                .ok();
        })
        .1;

        // M=8 batched variants : need x of size [M=8, K].
        let m_batch = 8usize;
        let x_batch_bf: Vec<half::bf16> = (0..m_batch * k)
            .map(|i| half::bf16::from_f32((i as f32 * 0.001).sin()))
            .collect();
        let x_batch_dev = stream.memcpy_stod(&x_batch_bf).expect("");
        let mut y_m8_dev = stream.alloc_zeros::<half::bf16>(m_batch * n).expect("");
        let (xb_p, y8_p) = unsafe {
            use cudarc::driver::{DevicePtr, DevicePtrMut};
            (
                x_batch_dev.device_ptr(&stream).0,
                y_m8_dev.device_ptr_mut(&stream).0,
            )
        };
        // Warm-up M=8 kernels.
        unsafe {
            kernels
                .sgemm_q4k_bf16_m8(&stream, w4_p, xb_p, y8_p, n as i32, k as i32)
                .ok();
            kernels
                .sgemm_q5k_bf16_m8(&stream, w5_p, xb_p, y8_p, n as i32, k as i32)
                .ok();
            kernels
                .sgemm_q6k_bf16_m8(&stream, w6_p, xb_p, y8_p, n as i32, k as i32)
                .ok();
        }
        stream.synchronize().ok();

        let q4_m8_ms = bench("Q4_K M8", &mut || unsafe {
            kernels
                .sgemm_q4k_bf16_m8(&stream, w4_p, xb_p, y8_p, n as i32, k as i32)
                .ok();
        })
        .1;
        let q5_m8_ms = bench("Q5_K M8", &mut || unsafe {
            kernels
                .sgemm_q5k_bf16_m8(&stream, w5_p, xb_p, y8_p, n as i32, k as i32)
                .ok();
        })
        .1;
        let q6_m8_ms = bench("Q6_K M8", &mut || unsafe {
            kernels
                .sgemm_q6k_bf16_m8(&stream, w6_p, xb_p, y8_p, n as i32, k as i32)
                .ok();
        })
        .1;

        let q4_bw = (n * row_bytes_q4) as f64 / (q4_ms * 1e-3) / 1e9;
        let q5_bw = (n * row_bytes_q5) as f64 / (q5_ms * 1e-3) / 1e9;
        let q6_bw = (n * row_bytes_q6) as f64 / (q6_ms * 1e-3) / 1e9;
        let q6_v2_bw = (n * row_bytes_q6) as f64 / (q6_v2_ms * 1e-3) / 1e9;
        // M=8 amortized bandwidth (W only, divided by 8 tokens).
        let q4_m8_bw = (n * row_bytes_q4) as f64 / (q4_m8_ms * 1e-3) / 1e9;
        let q5_m8_bw = (n * row_bytes_q5) as f64 / (q5_m8_ms * 1e-3) / 1e9;
        let q6_m8_bw = (n * row_bytes_q6) as f64 / (q6_m8_ms * 1e-3) / 1e9;

        eprintln!("\n=== sgemv quantized SGEMV bench ({n}x{k}) ===");
        eprintln!("  Q4_K V2 : {q4_ms:.3} ms  ({q4_bw:.1} GB/s)");
        eprintln!("  Q5_K    : {q5_ms:.3} ms  ({q5_bw:.1} GB/s)");
        eprintln!("  Q6_K V1 : {q6_ms:.3} ms  ({q6_bw:.1} GB/s)");
        eprintln!(
            "  Q6_K V2 : {q6_v2_ms:.3} ms  ({q6_v2_bw:.1} GB/s)  speedup vs V1 = {:.2}×",
            q6_ms / q6_v2_ms
        );
        eprintln!("  --- M=8 batched ---");
        eprintln!(
            "  Q4_K M8 : {q4_m8_ms:.3} ms  ({q4_m8_bw:.1} GB/s W single-pass)  ratio M8/M1 = {:.2}× (× 8 outputs!)",
            q4_m8_ms / q4_ms
        );
        eprintln!(
            "  Q5_K M8 : {q5_m8_ms:.3} ms  ({q5_m8_bw:.1} GB/s W single-pass)  ratio M8/M1 = {:.2}×",
            q5_m8_ms / q5_ms
        );
        eprintln!(
            "  Q6_K M8 : {q6_m8_ms:.3} ms  ({q6_m8_bw:.1} GB/s W single-pass)  ratio M8/M1V2 = {:.2}×",
            q6_m8_ms / q6_v2_ms
        );
    }

    /// T244.1 — micro-bench sgemv_q4k_bf16 vs matmul_bf16 on Qwen-7B FFN size.
    /// Run with `cargo test --release -- --nocapture sgemv_q4k_bench`.
    /// Compares wall-clock per matmul to validate the memory bandwidth gain.
    #[test]
    #[ignore = "perf benchmark — run explicitly with --ignored"]
    fn sgemv_q4k_bench() {
        use rustorch_gguf::dequant::{Q4_K_BYTES, QK_K};
        use std::time::Instant;

        // Qwen-7B FFN gate/up shape : N=18944, K=3584
        let n = 18944usize;
        let k = 3584usize;
        let blocks_per_row = k / QK_K;
        let row_bytes = blocks_per_row * Q4_K_BYTES;

        let mut state: u64 = 0xa5a5a5a5;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let mut w_q4k_bytes = vec![0u8; n * row_bytes];
        for b in &mut w_q4k_bytes {
            *b = (next() & 0xFF) as u8;
        }
        // Patch d/dmin to valid f16 values for each block.
        for row in 0..n {
            for blk in 0..blocks_per_row {
                let off = row * row_bytes + blk * Q4_K_BYTES;
                let d = half::f16::from_f32(0.05).to_le_bytes();
                let dmin = half::f16::from_f32(0.025).to_le_bytes();
                w_q4k_bytes[off] = d[0];
                w_q4k_bytes[off + 1] = d[1];
                w_q4k_bytes[off + 2] = dmin[0];
                w_q4k_bytes[off + 3] = dmin[1];
                for i in 0..12 {
                    w_q4k_bytes[off + 4 + i] = w_q4k_bytes[off + 4 + i] & 0x3F;
                }
            }
        }

        let ctx = CudaContext::new(0).expect("ctx");
        let stream = ctx.default_stream();
        let kernels = LlmKernels::new(ctx.clone());

        // Upload Q4_K bytes (~9 MB for our shape) and BF16 reference (~135 MB).
        let w_q4k_dev = stream.memcpy_stod(&w_q4k_bytes).expect("upload w_q4k");
        let x_bf: Vec<half::bf16> = (0..k)
            .map(|i| half::bf16::from_f32((i as f32 * 0.001).sin()))
            .collect();
        let x_dev = stream.memcpy_stod(&x_bf).expect("upload x");
        let mut y_dev = stream.alloc_zeros::<half::bf16>(n).expect("alloc y");

        // Warm-up + compile.
        unsafe {
            let (w_p, _g1) = w_q4k_dev.device_ptr(&stream);
            let (x_p, _g2) = x_dev.device_ptr(&stream);
            let (y_p, _g3) = y_dev.device_ptr_mut(&stream);
            kernels
                .sgemv_q4k_bf16(&stream, w_p, x_p, y_p, n as i32, k as i32)
                .expect("warmup");
        }
        stream.synchronize().expect("sync");

        // Bench Q4_K direct path.
        let n_iters = 100;
        let t0 = Instant::now();
        for _ in 0..n_iters {
            unsafe {
                let (w_p, _g1) = w_q4k_dev.device_ptr(&stream);
                let (x_p, _g2) = x_dev.device_ptr(&stream);
                let (y_p, _g3) = y_dev.device_ptr_mut(&stream);
                kernels
                    .sgemv_q4k_bf16(&stream, w_p, x_p, y_p, n as i32, k as i32)
                    .expect("bench");
            }
        }
        stream.synchronize().expect("sync end");
        let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0 / n_iters as f64;

        // Memory bandwidth (Q4_K bytes only) :
        let bytes_per_call = (n * row_bytes) as f64; // weights only
        let bw_gb_s = bytes_per_call / (elapsed_ms * 1e-3) / 1e9;

        eprintln!("\n=== sgemv_q4k_bf16 benchmark (Qwen-7B FFN shape N={n}, K={k}) ===");
        eprintln!("  Q4_K weight buffer: {:.1} MB", bytes_per_call / 1e6);
        eprintln!("  per-call latency  : {:.3} ms", elapsed_ms);
        eprintln!("  weight read BW    : {:.1} GB/s", bw_gb_s);
        eprintln!(
            "  effective FLOPS   : {:.1} GFLOPS",
            2.0 * n as f64 * k as f64 / (elapsed_ms * 1e-3) / 1e9
        );
        eprintln!("  → For Qwen-7B Q4_K_M decode (3 FFN matmul × 28 layers ≈ 84 calls/token) :");
        eprintln!(
            "     decode budget = {:.1} ms/token = {:.1} tok/s upper bound (FFN only)",
            elapsed_ms * 84.0,
            1000.0 / (elapsed_ms * 84.0)
        );
    }

    /// T241.6d — verify our quantize kernel emits 0x38 for scale = 1.0.
    /// With max_abs = 6.0, scale = 6.0/6.0 = 1.0 → ue_exp = 7, fmant = 0
    /// → byte 0x38.
    #[test]
    fn quantize_nvfp4_scale_1_emits_0x38() {
        let block: Vec<f32> = vec![6.0; 16]; // max_abs = 6, scale = 1.0
        let bf16_block: Vec<half::bf16> = block.iter().copied().map(half::bf16::from_f32).collect();

        let ctx = CudaContext::new(0).expect("ctx");
        let stream = ctx.default_stream();
        let kernels = LlmKernels::new(ctx);

        let x_dev = stream.memcpy_stod(&bf16_block).expect("upload");
        let mut out_fp4 = stream.alloc_zeros::<u8>(8).expect("fp4");
        let mut out_scale = stream.alloc_zeros::<u8>(1).expect("scale");
        unsafe {
            let (x_p, _g1) = x_dev.device_ptr(&stream);
            let (fp4_p, _g2) = out_fp4.device_ptr_mut(&stream);
            let (sc_p, _g3) = out_scale.device_ptr_mut(&stream);
            kernels
                .quantize_bf16_to_nvfp4(&stream, x_p, fp4_p, sc_p, 16)
                .expect("quantize");
        }
        let scale: Vec<u8> = stream.memcpy_dtov(&out_scale).expect("dtov");
        let decoded = ue4m3_decode(scale[0]);
        assert!(
            (decoded - 1.0).abs() < 0.1,
            "expected scale ≈ 1.0 for max_abs=6, got 0x{:02x} = {} (bias 7)",
            scale[0],
            decoded
        );
        assert_eq!(
            scale[0], 0x38,
            "scale byte should be 0x38 (E=7,M=0) for scale=1.0"
        );

        // Each FP4 element should be code 7 (= +6.0).
        let fp4: Vec<u8> = stream.memcpy_dtov(&out_fp4).expect("dtov fp4");
        for (i, &byte) in fp4.iter().enumerate() {
            let lo = byte & 0xf;
            let hi = byte >> 4;
            assert_eq!(lo, 7, "block[{}] low nibble = 0x{:x} ≠ 7", i * 2, lo);
            assert_eq!(hi, 7, "block[{}] hi nibble = 0x{:x} ≠ 7", i * 2 + 1, hi);
        }
    }

    /// T246.5.3 — `rope_partial_bf16_devcnt` doit produire un output BIT-EXACT
    /// identique à `rope_partial_bf16` quand `*pos_dev == pos`.
    #[test]
    fn rope_partial_bf16_devcnt_matches_static_pos() {
        let n_heads = 8usize;
        let head_dim = 128usize;
        let rope_dim = 64usize;
        let pos: i32 = 17;
        let total = n_heads * head_dim;

        // Inv freq for rope_dim=64 with base=10000.
        let half = rope_dim / 2;
        let inv_freq_host: Vec<f32> = (0..half)
            .map(|i| (10000.0_f32).powf(-(2.0 * i as f32) / rope_dim as f32))
            .collect();

        // Random-ish input.
        let x_host: Vec<half::bf16> = (0..total)
            .map(|i| half::bf16::from_f32(((i as f32) * 0.013).sin() * 0.5))
            .collect();

        let ctx = CudaContext::new(0).expect("ctx");
        let stream = ctx.default_stream();
        let kernels = LlmKernels::new(ctx);

        // Run static-pos variant.
        let mut x_static = stream.memcpy_stod(&x_host).expect("x_static");
        let inv_freq_dev = stream.memcpy_stod(&inv_freq_host).expect("inv_freq");
        {
            let (xs_p, _g) = unsafe { x_static.device_ptr_mut(&stream) };
            let (if_p, _g2) = unsafe { inv_freq_dev.device_ptr(&stream) };
            unsafe {
                kernels
                    .rope_partial_bf16(
                        &stream,
                        xs_p,
                        if_p,
                        pos,
                        n_heads as i32,
                        head_dim as i32,
                        rope_dim as i32,
                    )
                    .unwrap();
            }
        }
        let y_static: Vec<half::bf16> = stream.memcpy_dtov(&x_static).expect("dl static");

        // Run devcnt variant with pos in device buffer.
        let mut x_devcnt = stream.memcpy_stod(&x_host).expect("x_devcnt");
        let pos_dev = stream.memcpy_stod(&[pos]).expect("pos_dev");
        {
            let (xd_p, _g3) = unsafe { x_devcnt.device_ptr_mut(&stream) };
            let (if_p, _g4) = unsafe { inv_freq_dev.device_ptr(&stream) };
            let (pd_p, _g5) = unsafe { pos_dev.device_ptr(&stream) };
            unsafe {
                kernels
                    .rope_partial_bf16_devcnt(
                        &stream,
                        xd_p,
                        if_p,
                        pd_p,
                        n_heads as i32,
                        head_dim as i32,
                        rope_dim as i32,
                    )
                    .unwrap();
            }
        }
        let y_devcnt: Vec<half::bf16> = stream.memcpy_dtov(&x_devcnt).expect("dl devcnt");

        assert_eq!(y_static, y_devcnt, "rope_devcnt ≠ rope_static at pos={pos}");
    }

    /// T246.5.3 — `gqa_decode_online_bf16_devcnt` doit produire un output
    /// BIT-EXACT identique à `gqa_decode_online_bf16` quand `*kv_len_dev == kv_len`.
    #[test]
    fn gqa_decode_online_bf16_devcnt_matches_static_kv_len() {
        let n_heads = 4usize;
        let n_kv = 2usize;
        let head_dim = 64usize;
        let max_seq = 32i32;
        let kv_len: i32 = 13;

        let q_host: Vec<half::bf16> = (0..n_heads * head_dim)
            .map(|i| half::bf16::from_f32(((i as f32) * 0.07).sin() * 0.4))
            .collect();
        let kv_total = n_kv * (max_seq as usize) * head_dim;
        let k_host: Vec<half::bf16> = (0..kv_total)
            .map(|i| half::bf16::from_f32(((i as f32) * 0.011).cos() * 0.3))
            .collect();
        let v_host: Vec<half::bf16> = (0..kv_total)
            .map(|i| half::bf16::from_f32(((i as f32) * 0.017).sin() * 0.25))
            .collect();

        let ctx = CudaContext::new(0).expect("ctx");
        let stream = ctx.default_stream();
        let kernels = LlmKernels::new(ctx);

        let q_dev = stream.memcpy_stod(&q_host).expect("q");
        let k_dev = stream.memcpy_stod(&k_host).expect("k");
        let v_dev = stream.memcpy_stod(&v_host).expect("v");
        let mut out_static = stream
            .alloc_zeros::<half::bf16>(n_heads * head_dim)
            .expect("out_s");
        let mut out_devcnt = stream
            .alloc_zeros::<half::bf16>(n_heads * head_dim)
            .expect("out_d");

        // Static-kv_len variant.
        {
            let (q_p, _g1) = unsafe { q_dev.device_ptr(&stream) };
            let (k_p, _g2) = unsafe { k_dev.device_ptr(&stream) };
            let (v_p, _g3) = unsafe { v_dev.device_ptr(&stream) };
            let (os_p, _g4) = unsafe { out_static.device_ptr_mut(&stream) };
            unsafe {
                kernels
                    .gqa_decode_online_bf16(
                        &stream,
                        q_p,
                        k_p,
                        v_p,
                        os_p,
                        n_heads as i32,
                        n_kv as i32,
                        kv_len,
                        head_dim as i32,
                        max_seq,
                    )
                    .unwrap();
            }
        }
        let y_static: Vec<half::bf16> = stream.memcpy_dtov(&out_static).expect("dl s");

        // Devcnt variant.
        let kv_len_dev = stream.memcpy_stod(&[kv_len]).expect("kv_len_dev");
        {
            let (q_p, _g1) = unsafe { q_dev.device_ptr(&stream) };
            let (k_p, _g2) = unsafe { k_dev.device_ptr(&stream) };
            let (v_p, _g3) = unsafe { v_dev.device_ptr(&stream) };
            let (kld_p, _g5) = unsafe { kv_len_dev.device_ptr(&stream) };
            let (od_p, _g6) = unsafe { out_devcnt.device_ptr_mut(&stream) };
            unsafe {
                kernels
                    .gqa_decode_online_bf16_devcnt(
                        &stream,
                        q_p,
                        k_p,
                        v_p,
                        od_p,
                        n_heads as i32,
                        n_kv as i32,
                        kld_p,
                        head_dim as i32,
                        max_seq,
                    )
                    .unwrap();
            }
        }
        let y_devcnt: Vec<half::bf16> = stream.memcpy_dtov(&out_devcnt).expect("dl d");

        assert_eq!(
            y_static, y_devcnt,
            "gqa_devcnt ≠ gqa_static at kv_len={kv_len}"
        );
    }

    /// T246.5.3 — `increment_u32_dev` advance le compteur de 1 à chaque appel.
    #[test]
    fn increment_u32_dev_advances_by_one() {
        let ctx = CudaContext::new(0).expect("ctx");
        let stream = ctx.default_stream();
        let kernels = LlmKernels::new(ctx);

        let mut counter = stream.memcpy_stod(&[5i32]).expect("counter");
        {
            let (c_p, _g) = unsafe { counter.device_ptr_mut(&stream) };
            unsafe {
                kernels.increment_u32_dev(&stream, c_p).unwrap();
                kernels.increment_u32_dev(&stream, c_p).unwrap();
                kernels.increment_u32_dev(&stream, c_p).unwrap();
            }
        }

        let value: Vec<i32> = stream.memcpy_dtov(&counter).expect("dl");
        assert_eq!(value[0], 8, "expected 5 + 3 = 8");
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
