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
    }

    // Reduction over 64 threads.
    sdata[tid] = acc;
    __syncthreads();
    #pragma unroll
    for (int s = 32; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) {
        y[row] = (__nv_bfloat16)sdata[0];
    }
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
    }

    sdata[tid] = acc;
    __syncthreads();
    #pragma unroll
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
    sgemv_q6k: std::sync::OnceLock<(Arc<CudaModule>, CudaFunction)>,
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
            sgemv_q6k: std::sync::OnceLock::new(),
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

    /// T244.1.1 — V2 parity test : V2 must produce same output as V1
    /// (and CPU reference) within BF16 tolerance.
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
