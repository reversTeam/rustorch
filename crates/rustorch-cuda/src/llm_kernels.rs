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

// T246.6.3 — `y[i] += alpha * x[i]` for two BF16 vectors, alpha float.
// Used to accumulate weighted expert outputs in MoE FFN forward.
#[cfg(feature = "cuda")]
const SCALED_ADD_INPLACE_BF16_SRC: &str = r#"
#include <cuda_bf16.h>

extern "C" __global__ void scaled_add_inplace_bf16(
    __nv_bfloat16*       __restrict__ y,
    const __nv_bfloat16* __restrict__ x,
    float alpha,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float yi = (float)y[i];
    float xi = (float)x[i];
    y[i] = (__nv_bfloat16)(yi + alpha * xi);
}
"#;

// T246.6.2 — Top-K softmax routing kernel for MoE FFN (Qwen3.6-35B-A3B).
//
// Input  : raw scores from `gate_inp @ h` of shape [n_experts] (BF16).
// Output : top-K expert indices + renormalized softmax weights such that
//          `sum(weights[0..K]) == 1`.
//
// Algorithm (single block, 32 threads — sufficient since n_experts ≤ 256):
//   1. Each thread loads a stride of scores → finds local max
//   2. Warp-reduce to global max
//   3. Each thread computes partial sum-exp of its strided values
//   4. Warp-reduce to total Z
//   5. Probabilities = exp(s - max) / Z
//   6. K-rank selection : K rounds, each finds the global argmax over
//      not-yet-selected slots, marks it selected (sets prob to -inf),
//      writes (idx, prob) to output, and accumulates renorm Z2
//   7. Final pass : weights[i] /= Z2
//
// This is sufficient up to n_experts = 1024. For larger n_experts a
// multi-block version with shared-memory partial sorts would be needed.
#[cfg(feature = "cuda")]
const TOPK_SOFTMAX_BF16_SRC: &str = r#"
#include <cuda_bf16.h>

extern "C" __global__ void topk_softmax_bf16(
    const __nv_bfloat16* __restrict__ scores,    // [n_experts]
    int*                 __restrict__ indices,    // [k]   (output)
    __nv_bfloat16*       __restrict__ weights,    // [k]   (output, renormalized)
    int n_experts,
    int k
) {
    extern __shared__ float sh[];
    // sh[0..n_experts] : scratch for probabilities (mutable for K-selection)
    // sh[n_experts]    : global max
    // sh[n_experts+1]  : Z (sum-exp)
    // sh[n_experts+2]  : Z2 (renorm)

    int tid = threadIdx.x;

    // ---- 1. Find max ----
    float local_max = -1e30f;
    for (int i = tid; i < n_experts; i += blockDim.x) {
        float v = (float)scores[i];
        if (v > local_max) local_max = v;
    }
    sh[tid] = local_max;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sh[tid] = fmaxf(sh[tid], sh[tid + s]);
        __syncthreads();
    }
    float gmax = sh[0];
    __syncthreads();

    // ---- 2. Sum-exp for full softmax ----
    float local_sum = 0.0f;
    for (int i = tid; i < n_experts; i += blockDim.x) {
        float v = (float)scores[i];
        local_sum += expf(v - gmax);
    }
    sh[tid] = local_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sh[tid] += sh[tid + s];
        __syncthreads();
    }
    float Z = sh[0];
    __syncthreads();

    // ---- 3. Compute probs[i] in shared scratch ----
    // Reuse sh[0..n_experts] for probs (different scope from reduction).
    for (int i = tid; i < n_experts; i += blockDim.x) {
        float v = (float)scores[i];
        sh[i] = expf(v - gmax) / Z;
    }
    __syncthreads();

    // ---- 4. K rounds : find argmax of remaining probs, mark selected ----
    // Single-thread (tid==0) does the K rounds : K is typically ≤ 8 so this
    // O(K * n_experts) loop is fine. Could parallelize via warp reductions
    // if K > ~16.
    if (tid == 0) {
        float Z2 = 0.0f;
        for (int kk = 0; kk < k; ++kk) {
            float best_p = -1.0f;
            int   best_i = 0;
            for (int i = 0; i < n_experts; ++i) {
                float p = sh[i];
                if (p > best_p) { best_p = p; best_i = i; }
            }
            indices[kk] = best_i;
            weights[kk] = (__nv_bfloat16)best_p;  // un-normalized for now
            sh[best_i] = -1.0f;                    // mark consumed
            Z2 += best_p;
        }
        // Renormalize.
        float inv_Z2 = (Z2 > 0.0f) ? (1.0f / Z2) : 0.0f;
        for (int kk = 0; kk < k; ++kk) {
            float w = (float)weights[kk];
            weights[kk] = (__nv_bfloat16)(w * inv_Z2);
        }
    }
}
"#;

// T247.7 — GQA decode backward (Flash-Attention style, M=1).
//
// Inputs (from training-aware forward) :
//   q          : [n_heads, head_dim]            BF16
//   k_cache    : [n_kv, max_seq, head_dim]      BF16
//   v_cache    : [n_kv, max_seq, head_dim]      BF16
//   m_saved    : [n_heads]                      float (max score from softmax)
//   l_saved    : [n_heads]                      float (sum-exp from softmax)
//   do         : [n_heads, head_dim]            BF16  (output gradient)
// Outputs :
//   dq         : [n_heads, head_dim]            BF16  (overwritten)
//   dk_accum   : [n_kv, max_seq, head_dim]      float (atomic-summed)
//   dv_accum   : [n_kv, max_seq, head_dim]      float (atomic-summed)
//
// Probabilities `p_t = exp(s_t - m) / l` are recomputed inside the kernel
// from `m_saved`, `l_saved` and the dot product Q·K_t (avoids storing the
// full P matrix per layer).
//
// Caller zeroes `dk_accum` and `dv_accum` before the first call of an
// iteration (multi-layer training accumulates via atomic adds across calls
// — but for the per-layer dQ output, each call OWNS its slot and overwrites).
//
// Math :
//   dV_t = p_t · do                 (atomic add into dv_accum[kv_h, t])
//   dp_t = do · V_t                 (sum over head_dim)
//   D    = Σ_t p_t · dp_t           (scalar per head)
//   ds_t = p_t · (dp_t - D)
//   dQ   = scale · Σ_t ds_t · K_t   (per head_dim)
//   dK_t = scale · ds_t · Q         (atomic add into dk_accum[kv_h, t])
//
// Single block per head, `head_dim` threads. 3 passes over kv_len (compute
// D, then per-t for dV/dK/dQ-accum).
#[cfg(feature = "cuda")]
const GQA_DECODE_GRAD_BF16_SRC: &str = r#"
#include <cuda_bf16.h>

extern "C" __global__ void gqa_decode_grad_bf16(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k_cache,
    const __nv_bfloat16* __restrict__ v_cache,
    const float*         __restrict__ m_saved,
    const float*         __restrict__ l_saved,
    const __nv_bfloat16* __restrict__ do_,
    __nv_bfloat16*       __restrict__ dq,
    float*               __restrict__ dk_accum,
    float*               __restrict__ dv_accum,
    int n_heads,
    int n_kv,
    int kv_len,
    int head_dim,
    int max_seq,
    float scale
) {
    int h   = blockIdx.x;
    if (h >= n_heads) return;
    int tid = threadIdx.x;
    if (tid >= head_dim) return;
    int kv_h = h * n_kv / n_heads;

    extern __shared__ float sdata[];

    float qd  = (float)q[h * head_dim + tid];
    float dod = (float)do_[h * head_dim + tid];
    float m_h = m_saved[h];
    float l_h = l_saved[h];
    float inv_l = 1.0f / l_h;

    // ---- Pass 1 : D = Σ_t p_t · (do · V_t) ----
    float D_local = 0.0f;
    for (int t = 0; t < kv_len; ++t) {
        // Recompute s_t = scale · Q · K_t (per-thread partial then reduce).
        float kd = (float)k_cache[(kv_h * max_seq + t) * head_dim + tid];
        sdata[tid] = qd * kd;
        __syncthreads();
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s && tid + s < head_dim) sdata[tid] += sdata[tid + s];
            __syncthreads();
        }
        float s_t = sdata[0] * scale;
        __syncthreads();

        float p_t = expf(s_t - m_h) * inv_l;

        // dp_t · do_d component for this dim (scaled by p_t)
        float vd = (float)v_cache[(kv_h * max_seq + t) * head_dim + tid];
        D_local += p_t * vd * dod;
    }
    sdata[tid] = D_local;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && tid + s < head_dim) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float D = sdata[0];
    __syncthreads();

    // ---- Pass 2 : per-t writes (dV, dK, accum dQ) ----
    float dq_acc = 0.0f;
    for (int t = 0; t < kv_len; ++t) {
        float kd = (float)k_cache[(kv_h * max_seq + t) * head_dim + tid];
        float vd = (float)v_cache[(kv_h * max_seq + t) * head_dim + tid];

        // Recompute s_t and p_t (needed again because we don't store them).
        sdata[tid] = qd * kd;
        __syncthreads();
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s && tid + s < head_dim) sdata[tid] += sdata[tid + s];
            __syncthreads();
        }
        float s_t = sdata[0] * scale;
        __syncthreads();
        float p_t = expf(s_t - m_h) * inv_l;

        // dp_t = do · V_t (reduce over head_dim).
        sdata[tid] = vd * dod;
        __syncthreads();
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s && tid + s < head_dim) sdata[tid] += sdata[tid + s];
            __syncthreads();
        }
        float dp_t = sdata[0];
        __syncthreads();

        float ds_t = p_t * (dp_t - D);

        // dQ accumulation (own this dim, no atomic needed).
        dq_acc += scale * ds_t * kd;

        // dK[kv_h, t, dim] += scale · ds_t · Q[h, dim]    (atomic — GQA share)
        atomicAdd(&dk_accum[(kv_h * max_seq + t) * head_dim + tid], scale * ds_t * qd);

        // dV[kv_h, t, dim] += p_t · do[h, dim]            (atomic — GQA share)
        atomicAdd(&dv_accum[(kv_h * max_seq + t) * head_dim + tid], p_t * dod);
    }

    dq[h * head_dim + tid] = (__nv_bfloat16)dq_acc;
}
"#;

// T247.6 — Q4_K SGEMV backward, dx only (W is frozen for LoRA fine-tuning).
//
// Forward:  y = W·x         where W is [N,K] in Q4_K, x BF16 [K], y BF16 [N]
// Backward: dx = Wᵀ·dy      so each output dx[k] = Σ_n W[n,k] · dy[n]
//
// Each thread handles 1 output column k of W (= 1 element of dx). It
// iterates over rows n=0..N, dequantizes W[n,k] from the Q4_K block format,
// multiplies by dy[n], accumulates. This is intentionally a column-major
// access pattern over W (uncoalesced) — pilot focused on correctness, not
// perf. Production uses tile-based or pre-transpose.
//
// Q4_K column-→nibble decode (mirroring the V2 forward tile mapping):
//   sb       = k / 256              — super-block in row
//   within   = k % 256
//   sub      = within / 32          — sub-block index 0..7
//   pos      = within % 32          — position within sub-block
//   pair_idx = sub / 2
//   nib_lo   = (sub & 1) == 0       — low nibble for even sub-blocks
//   byte_off = 16 + pair_idx*32 + pos
//   nibble   = nib_lo ? (byte & 0xF) : (byte >> 4)
//
// Scale unpacking : if sub < 4, sc/m direct from `scales[sub]`/`scales[sub+4]`
// masked with 0x3F. If sub >= 4, reconstructed from the upper 2 bits of
// `scales[sub-4]`/`scales[sub]` and `scales[sub+4]` (matches V2 forward).
#[cfg(feature = "cuda")]
const SGEMV_Q4K_GRAD_DX_BF16_SRC: &str = r#"
#include <cuda_bf16.h>
#include <cuda_fp16.h>

extern "C" __global__ void sgemv_q4k_grad_dx_bf16(
    const unsigned char* __restrict__ w_q4k,
    const __nv_bfloat16* __restrict__ dy,
    __nv_bfloat16*       __restrict__ dx,
    int N,
    int K
) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= K) return;

    int blocks_per_row = K / 256;
    int sb        = k / 256;
    int within    = k - sb * 256;
    int sub       = within >> 5;
    int pos       = within & 31;
    int pair_idx  = sub >> 1;
    int sub_in_pair = sub & 1;       // 0 = low nibble, 1 = high
    int byte_off  = 16 + pair_idx * 32 + pos;

    float acc = 0.0f;

    for (int n = 0; n < N; ++n) {
        const unsigned char* blk =
            w_q4k + ((size_t)n * blocks_per_row + sb) * 144;

        unsigned short d_bits    = blk[0] | (blk[1] << 8);
        unsigned short dmin_bits = blk[2] | (blk[3] << 8);
        float d    = __half2float(__ushort_as_half(d_bits));
        float dmin = __half2float(__ushort_as_half(dmin_bits));

        const unsigned char* scales = blk + 4;
        unsigned char sc_i, m_i;
        if (sub < 4) {
            sc_i = scales[sub]     & 0x3F;
            m_i  = scales[sub + 4] & 0x3F;
        } else {
            int s = sub - 4;
            sc_i = (scales[s + 8] & 0x0F) | ((scales[s]     >> 6) << 4);
            m_i  = (scales[s + 8] >> 4)   | ((scales[s + 4] >> 6) << 4);
        }
        float scale = d    * (float)sc_i;
        float min_v = dmin * (float)m_i;

        unsigned char byte = blk[byte_off];
        int nibble = sub_in_pair ? (byte >> 4) : (byte & 0x0F);

        float w_val = scale * (float)nibble - min_v;
        acc += w_val * (float)dy[n];
    }

    dx[k] = (__nv_bfloat16)acc;
}
"#;

// T247.5 — Embedding lookup backward (sparse scatter via atomicAdd).
//
// Forward : `out[i] = embed_table[token_id, i]`
// Backward: `d_embed_accum[token_id, i] += dy[i]`  (atomic, float accumulator)
//
// The accumulator is float32 (atomicAdd on float is universally supported and
// avoids the precision pitfalls of bf16/half atomic adds). Caller zeroes the
// accumulator before the first call of an iteration ; multi-token batches
// accumulate via repeated calls.
#[cfg(feature = "cuda")]
const EMBEDDING_LOOKUP_GRAD_BF16_SRC: &str = r#"
#include <cuda_bf16.h>

extern "C" __global__ void embedding_lookup_grad_bf16(
    const __nv_bfloat16* __restrict__ dy,             // [d]
    int                              token_id,
    float*               __restrict__ d_embed_accum,  // [vocab, d]
    int d
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= d) return;
    float dy_i = (float)dy[i];
    atomicAdd(&d_embed_accum[(long long)token_id * d + i], dy_i);
}
"#;

// T247.4 — Cross-entropy from logits, fused forward + backward.
// Single-row : given a logits vector of length `vocab` and a `target` token id,
// computes (a) the scalar loss = -log(softmax(logits)[target]) and (b) the
// gradient dlogits[i] = softmax(logits)[i] - δ_{i,target}.
//
// Numerically stable : max-shift before exp.
// Three passes over `vocab` : max-reduce, sum-exp, write.
//
// Block dim should be a power of 2 ≤ 1024. Single block per call.
#[cfg(feature = "cuda")]
const CROSS_ENTROPY_LOSS_GRAD_BF16_SRC: &str = r#"
#include <cuda_bf16.h>

extern "C" __global__ void cross_entropy_loss_grad_bf16(
    const __nv_bfloat16* __restrict__ logits,   // [vocab]
    int target,
    float*               __restrict__ loss_out, // [1]
    __nv_bfloat16*       __restrict__ dlogits,  // [vocab]
    int vocab
) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;

    // ---- Pass 1 : find max(logits) for numerical stability ----
    float local_max = -1e30f;
    for (int i = tid; i < vocab; i += blockDim.x) {
        float v = (float)logits[i];
        if (v > local_max) local_max = v;
    }
    sdata[tid] = local_max;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }
    float max_l = sdata[0];
    __syncthreads();

    // ---- Pass 2 : sum_i exp(l_i - max) ----
    float local_sum = 0.0f;
    for (int i = tid; i < vocab; i += blockDim.x) {
        float v = (float)logits[i];
        local_sum += expf(v - max_l);
    }
    sdata[tid] = local_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float Z     = sdata[0];
    float log_Z = logf(Z);
    __syncthreads();

    // ---- Pass 3 : loss + dlogits ----
    if (tid == 0) {
        float l_target = (float)logits[target];
        *loss_out = -(l_target - max_l - log_Z);
    }
    for (int i = tid; i < vocab; i += blockDim.x) {
        float v = (float)logits[i];
        float p = expf(v - max_l) / Z;
        float g = p - ((i == target) ? 1.0f : 0.0f);
        dlogits[i] = (__nv_bfloat16)g;
    }
}
"#;

// T247.3 — RoPE partial backward kernel (BF16). Same layout as forward
// (pairs (k, k+half)) but applies the inverse rotation matrix.
//   Forward (per pair):
//     y_a = a·cos - b·sin
//     y_b = a·sin + b·cos
//   Backward (given dy):
//     da =  dy_a · cos + dy_b · sin
//     db = -dy_a · sin + dy_b · cos
//
// In-place: input dy is read, output dx is written to the same buffer
// (or pass separate buffers — see wrapper).
#[cfg(feature = "cuda")]
const ROPE_PARTIAL_GRAD_BF16_SRC: &str = r#"
#include <cuda_bf16.h>

extern "C" __global__ void rope_partial_grad_bf16(
    const __nv_bfloat16* __restrict__ dy,
    const float*         __restrict__ inv_freq,
    int pos,
    __nv_bfloat16*       __restrict__ dx,
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
    float dya = (float)dy[row + k];
    float dyb = (float)dy[row + k + half];

    dx[row + k]        = (__nv_bfloat16)( dya * cos_k + dyb * sin_k);
    dx[row + k + half] = (__nv_bfloat16)(-dya * sin_k + dyb * cos_k);
}
"#;

// T247.2 — SwiGLU backward kernel (BF16). Elementwise, no reduction.
//   silu(g)       = g * σ(g)            where σ(g) = 1/(1+e^-g)
//   silu'(g)      = σ(g) + g·σ(g)·(1-σ(g))
//   dgate_i       = dy_i · silu'(gate_i) · up_i
//   dup_i         = dy_i · silu(gate_i)
#[cfg(feature = "cuda")]
const SWIGLU_GRAD_BF16_SRC: &str = r#"
#include <cuda_bf16.h>

extern "C" __global__ void swiglu_grad_bf16(
    const __nv_bfloat16* __restrict__ gate,
    const __nv_bfloat16* __restrict__ up,
    const __nv_bfloat16* __restrict__ dy,
    __nv_bfloat16*       __restrict__ dgate,
    __nv_bfloat16*       __restrict__ dup,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g   = (float)gate[i];
    float u   = (float)up[i];
    float dyi = (float)dy[i];

    float sig         = 1.0f / (1.0f + expf(-g));
    float silu        = g * sig;
    float silu_prime  = sig + g * sig * (1.0f - sig);

    dgate[i] = (__nv_bfloat16)(dyi * silu_prime * u);
    dup[i]   = (__nv_bfloat16)(dyi * silu);
}
"#;

// T247.1 — RMSNorm backward kernel (BF16). Pilot for the training-ready
// CUDA backward path. Single-row (outer=1) implementation suitable for the
// LLM decode regime. Multi-row training (outer > 1) requires either an
// atomic dgamma accumulator or a per-row partial buffer + reduce kernel.
//
// Math :
//   r       = sqrt(mean(x²) + eps)
//   inv_r   = 1/r
//   s_acc   = sum(dy_i * gamma_i * x_i)
//   dx_i    = inv_r * dy_i * gamma_i  -  x_i * s_acc / (d * r³)
//   dgamma_i = dy_i * x_i * inv_r
//
// Kernel uses 2 sequential warp reductions (sum_sq, then s_acc) within a
// single block per row.
#[cfg(feature = "cuda")]
const RMS_NORM_GRAD_BF16_SRC: &str = r#"
#include <cuda_bf16.h>

extern "C" __global__ void rms_norm_grad_bf16(
    const __nv_bfloat16* __restrict__ x,         // [d]
    const __nv_bfloat16* __restrict__ gamma,     // [d]
    const __nv_bfloat16* __restrict__ dy,        // [d]
    __nv_bfloat16*       __restrict__ dx,        // [d]
    __nv_bfloat16*       __restrict__ dgamma,    // [d]
    int d,
    float eps
) {
    extern __shared__ float sdata[];

    // ---- Pass 1 : sum of squares (for r). Grid-stride loop. ----
    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        float xi = (float)x[i];
        sum_sq += xi * xi;
    }
    sdata[threadIdx.x] = sum_sq;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    float r     = sqrtf(sdata[0] / (float)d + eps);
    float inv_r = 1.0f / r;
    __syncthreads();

    // ---- Pass 2 : s_acc = sum(dy * gamma * x). ----
    float s_local = 0.0f;
    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        float xi  = (float)x[i];
        float gi  = (float)gamma[i];
        float dyi = (float)dy[i];
        s_local += dyi * gi * xi;
    }
    sdata[threadIdx.x] = s_local;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    float s_acc = sdata[0];
    __syncthreads();

    // ---- Pass 3 : write dx and dgamma. ----
    float r3_inv = inv_r * inv_r * inv_r;  // 1/r³
    float coeff  = s_acc / (float)d * r3_inv;
    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        float xi  = (float)x[i];
        float gi  = (float)gamma[i];
        float dyi = (float)dy[i];
        float dx_i  = inv_r * dyi * gi - xi * coeff;
        float dg_i  = dyi * xi * inv_r;
        dx[i]     = (__nv_bfloat16)dx_i;
        dgamma[i] = (__nv_bfloat16)dg_i;
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

// T246.7 P1.3c — atomic-style add of a host-supplied int value to a 1-elt
// device buffer. 1 thread, 1 block, no synchronization needed (called
// between distinct stream ops in decode_step_tree).
#[cfg(feature = "cuda")]
const ADD_U32_DEV_SRC: &str = r#"
extern "C" __global__ void add_u32_dev(int* p, int value) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *p = *p + value;
    }
}
"#;

// T246.7 P1.3c — write a host-supplied int value to a 1-elt device buffer.
// 1 thread, 1 block.
#[cfg(feature = "cuda")]
const SET_U32_DEV_SRC: &str = r#"
extern "C" __global__ void set_u32_dev(int* p, int value) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *p = value;
    }
}
"#;

// T246.7 P1.3c — argmax over `tree_size` independent rows of `vocab` BF16
// logits each. One block per row, block_dim = 256. Bit-equivalent to
// calling `argmax_bf16` `tree_size` times (uses the same reduction).
#[cfg(feature = "cuda")]
const ARGMAX_LOGITS_TREE_BF16_SRC: &str = r#"
#include <cuda_bf16.h>

extern "C" __global__ void argmax_logits_tree_bf16(
    const __nv_bfloat16* __restrict__ logits,   // [tree_size, vocab]
    unsigned int*        __restrict__ tokens,   // [tree_size]
    int tree_size,
    int vocab
) {
    extern __shared__ float sdata[];
    int* sidx = (int*)(sdata + blockDim.x);

    int row = blockIdx.x;
    if (row >= tree_size) return;
    const __nv_bfloat16* row_ptr = logits + (long long)row * (long long)vocab;

    float local_max = -1e30f;
    int   local_idx = 0;
    for (int i = threadIdx.x; i < vocab; i += blockDim.x) {
        float v = (float)row_ptr[i];
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
    if (threadIdx.x == 0) tokens[row] = (unsigned int)sidx[0];
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

// T246.7 P1.3a — tree-aware KV append. Writes `tree_size` consecutive K/V
// rows starting at slot `*pos_dev`. Layout matches `kv_append_bf16_devcnt`
// exactly so the (tree_size=1) case is bit-equivalent.
//   k_in, v_in : [tree_size, kv_dim]            BF16
//   k_cache, v_cache : [max_seq, kv_dim]        BF16
// For row r in [0..tree_size), writes to slot `*pos_dev + r`.
// Grid : (ceil(kv_dim/256), tree_size, 1)   Block : (256, 1, 1).
#[cfg(feature = "cuda")]
const KV_APPEND_TREE_BF16_SRC: &str = r#"
#include <cuda_bf16.h>

extern "C" __global__ void kv_append_tree_bf16(
    __nv_bfloat16* __restrict__ k_cache,
    __nv_bfloat16* __restrict__ v_cache,
    const __nv_bfloat16* __restrict__ k_in,
    const __nv_bfloat16* __restrict__ v_in,
    const int* __restrict__ pos_dev,
    int tree_size,
    int kv_dim,
    int max_seq
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y;
    if (tid >= kv_dim) return;
    if (row >= tree_size) return;
    int pos = *pos_dev + row;
    if (pos >= max_seq) return;
    long long off_cache = (long long)pos * (long long)kv_dim + (long long)tid;
    long long off_in    = (long long)row * (long long)kv_dim + (long long)tid;
    k_cache[off_cache] = k_in[off_in];
    v_cache[off_cache] = v_in[off_in];
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

// T246.6.7 — sgemv_q4k_bf16 V3 — high-bandwidth Q4_K SGEMV (multi-row).
//
// Profile (nsys, 27B Q4_K_M, sm_121, LPDDR5X 273 GB/s peak) revealed V2
// achieves only ~38% of peak BW on FFN matmuls. V3 targets ~60-70% via:
//   1. **4 rows per block** — shares the activation read across 4 output
//      rows (x cache reuse + lower per-row overhead). N=25600 (FFN) →
//      6400 blocks (still > 144 SMs).
//   2. **Per-thread scale compute** — eliminates the V2 `if(tid==0) +
//      __syncthreads()` scale broadcast. Saves blocks_per_row syncthreads.
//   3. `__launch_bounds__(128, 8)` — limits register spilling, allows
//      higher SM occupancy and better instruction-level parallelism.
//
// Tile mapping (128 threads, 4 rows × 32 lanes per row):
//   row_in_block = tid >> 5    (0..3) — which output row
//   lane         = tid & 31    (0..31) — lane within row
//   group        = lane >> 3   (0..3) — pair_idx (4 sub-block pairs total)
//   pos_base     = (lane & 7) << 2  — byte offset within pair (0..28 step 4)
//   Each lane reads 4 bytes (uint32) → 8 nibbles → contributes to BOTH
//   sub-blocks of the pair (sub 2*group via low nibble + sub 2*group+1
//   via high nibble). 32 lanes × 4 pairs × 4 bytes = 512 bytes per pass...
//   but only 4 distinct pairs cover the 128-byte payload, so each lane
//   covers 1/8 of the bytes (4 bytes), and 32 lanes cover the full pair
//   payload = 4×32 = 128 bytes ✓.
//
// Per super-block contribution per lane :
//   - 4 weights of sub-block (2*group) via low nibbles
//   - 4 weights of sub-block (2*group+1) via high nibbles
//   - Total 8 weights/lane × 32 lanes = 256 weights ✓
#[cfg(feature = "cuda")]
const SGEMV_Q4K_BF16_V3_SRC: &str = r#"
#include <cuda_bf16.h>
#include <cuda_fp16.h>

extern "C" __launch_bounds__(128, 8)
__global__ void sgemv_q4k_bf16_v3(
    const unsigned char* __restrict__ w_q4k,
    const __nv_bfloat16* __restrict__ x,
    __nv_bfloat16* __restrict__ y,
    int N,
    int K
) {
    int row0 = blockIdx.x * 4;
    int tid  = threadIdx.x;
    int row_in_block = tid >> 5;        // 0..3
    int lane         = tid & 31;
    int row          = row0 + row_in_block;
    if (row >= N) return;

    int blocks_per_row = K / 256;
    int row_offset     = row * blocks_per_row * 144;

    int group     = lane >> 3;            // 0..3 — pair_idx
    int pos_base  = (lane & 7) << 2;      // 0..28 step 4
    int byte_base = (group << 5) + pos_base;  // pair_idx*32 + pos_base

    float acc = 0.0f;

    for (int b = 0; b < blocks_per_row; ++b) {
        int blk_off = row_offset + b * 144;
        const unsigned char* blk = w_q4k + blk_off;

        // Header : d, dmin (broadcast load via L1, all threads).
        unsigned short d_bits    = blk[0] | (blk[1] << 8);
        unsigned short dmin_bits = blk[2] | (blk[3] << 8);
        float d    = __half2float(__ushort_as_half(d_bits));
        float dmin = __half2float(__ushort_as_half(dmin_bits));

        // Per-thread scale unpack for BOTH sub-blocks of this pair.
        // sub_a = 2*group (low nibble), sub_b = 2*group + 1 (high nibble).
        const unsigned char* scales = blk + 4;
        int sub_a = group * 2;
        int sub_b = sub_a + 1;
        unsigned char sc_a, m_a, sc_b, m_b;
        // sub_a in {0,2,4,6} ; sub_b in {1,3,5,7}.
        // First 4 sc/m use scales[0..7], last 4 use derived bits from
        // scales[8..11] + upper 2 bits of scales[0..7].
        if (sub_a < 4) {
            sc_a = scales[sub_a]     & 0x3F;
            m_a  = scales[sub_a + 4] & 0x3F;
        } else {
            int ga = sub_a - 4;
            sc_a = (scales[ga + 8] & 0x0F) | ((scales[ga]     >> 6) << 4);
            m_a  = (scales[ga + 8] >> 4)   | ((scales[ga + 4] >> 6) << 4);
        }
        if (sub_b < 4) {
            sc_b = scales[sub_b]     & 0x3F;
            m_b  = scales[sub_b + 4] & 0x3F;
        } else {
            int gb = sub_b - 4;
            sc_b = (scales[gb + 8] & 0x0F) | ((scales[gb]     >> 6) << 4);
            m_b  = (scales[gb + 8] >> 4)   | ((scales[gb + 4] >> 6) << 4);
        }
        float scale_a = d    * (float)sc_a;
        float min_a   = dmin * (float)m_a;
        float scale_b = d    * (float)sc_b;
        float min_b   = dmin * (float)m_b;

        // uint32 load of 4 nibble bytes for THIS thread.
        const unsigned char* qs = blk + 16;
        unsigned int qbytes = *(const unsigned int*)(qs + byte_base);

        // Load 4 BF16 of x for sub-block A (sub_a) and 4 for sub-block B (sub_b).
        const __nv_bfloat16* xa_ptr = x + b * 256 + sub_a * 32 + pos_base;
        const __nv_bfloat16* xb_ptr = x + b * 256 + sub_b * 32 + pos_base;
        uint2 xa = *(const uint2*)xa_ptr;
        uint2 xb = *(const uint2*)xb_ptr;
        float xa0 = (float)__ushort_as_bfloat16((unsigned short)(xa.x & 0xFFFFu));
        float xa1 = (float)__ushort_as_bfloat16((unsigned short)(xa.x >> 16));
        float xa2 = (float)__ushort_as_bfloat16((unsigned short)(xa.y & 0xFFFFu));
        float xa3 = (float)__ushort_as_bfloat16((unsigned short)(xa.y >> 16));
        float xb0 = (float)__ushort_as_bfloat16((unsigned short)(xb.x & 0xFFFFu));
        float xb1 = (float)__ushort_as_bfloat16((unsigned short)(xb.x >> 16));
        float xb2 = (float)__ushort_as_bfloat16((unsigned short)(xb.y & 0xFFFFu));
        float xb3 = (float)__ushort_as_bfloat16((unsigned short)(xb.y >> 16));

        // Decode 4 LOW nibbles → sub-block A weights, 4 HIGH nibbles → sub-block B.
        unsigned char by0 = (qbytes      ) & 0xFFu;
        unsigned char by1 = (qbytes >>  8) & 0xFFu;
        unsigned char by2 = (qbytes >> 16) & 0xFFu;
        unsigned char by3 = (qbytes >> 24) & 0xFFu;
        int na0 = by0 & 0x0F, na1 = by1 & 0x0F, na2 = by2 & 0x0F, na3 = by3 & 0x0F;
        int nb0 = by0 >>   4, nb1 = by1 >>   4, nb2 = by2 >>   4, nb3 = by3 >>   4;

        acc += (scale_a * (float)na0 - min_a) * xa0;
        acc += (scale_a * (float)na1 - min_a) * xa1;
        acc += (scale_a * (float)na2 - min_a) * xa2;
        acc += (scale_a * (float)na3 - min_a) * xa3;
        acc += (scale_b * (float)nb0 - min_b) * xb0;
        acc += (scale_b * (float)nb1 - min_b) * xb1;
        acc += (scale_b * (float)nb2 - min_b) * xb2;
        acc += (scale_b * (float)nb3 - min_b) * xb3;
    }

    // Warp-shuffle reduction WITHIN this row's 32 lanes (1 warp).
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_xor_sync(0xffffffffu, acc, offset);
    }
    if (lane == 0) {
        y[row] = (__nv_bfloat16)acc;
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

// T246.10 A6b.1 — sgemm_q4k_bf16_mvar : Q4_K matmul with arbitrary batch M (1..MAX).
// Extension of sgemm_q4k_bf16_m8 to variable M. Reads each W super-block ONCE
// per block and accumulates against all M m-rows (tiled in chunks of 8).
//
// Bandwidth analysis (Qwen3.6 35B-A3B Q4_K_M, N=4096, K=2048, blocks_per_row=8) :
//   M=1   : per-token W reads  = 144*8 * 4096 = 4.7 MB    (M=1 baseline)
//   M=8   : per-token W reads  = 4.7 / 8     ≈ 590 KB     (M=8 win)
//   M=512 : per-token W reads  = 4.7 / 512   ≈ 9 KB       (M=512 dream)
//
// Mirror of M=8 kernel : 64 threads/TG, per-thread accumulators [8] floats.
// For M > 8 we tile : outer loop processes ceil(M/8) m-tiles, each tile
// re-reads x and accumulates the same 4 weights into 8 accumulators.
//
// The kernel re-reads the W super-block 1 time per M-tile (the dequant scales
// are cached in shmem and re-used across m-tiles, the W bytes themselves are
// in L1 cache for the second-and-later tile reads).
//
// Layout matches sgemm_q4k_bf16_m8 EXACTLY for M=1..8 (tile_idx=0 only).
// For M > 8 each block processes ALL m-tiles in sequence.
//
// Inputs :
//   w_q4k : [N, K] Q4_K row-major (same layout as sgemv_q4k_v2 / sgemm_m8)
//   x     : [M, K] BF16 row-major
//   y     : [M, N] BF16 row-major (output)
//
// Constraint : K must be multiple of 256.
#[cfg(feature = "cuda")]
const SGEMM_Q4K_BF16_MVAR_SRC: &str = r#"
#include <cuda_bf16.h>
#include <cuda_fp16.h>

extern "C" __global__ void sgemm_q4k_bf16_mvar(
    const unsigned char* __restrict__ w_q4k,
    const __nv_bfloat16* __restrict__ x,    // [M, K] row-major
    __nv_bfloat16* __restrict__ y,          // [M, N] row-major
    int M,
    int N,
    int K
) {
    int row = blockIdx.x;
    if (row >= N) return;
    int tid = threadIdx.x;
    int blocks_per_row = K / 256;
    int row_offset = row * blocks_per_row * 144;
    int n_mtiles = (M + 7) >> 3;
    int mtile_idx = blockIdx.y;
    if (mtile_idx >= n_mtiles) return;
    int m_base = mtile_idx << 3;
    int m_cnt = M - m_base;
    if (m_cnt > 8) m_cnt = 8;

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
        // x base for THIS m-tile : x + (m_base + m) * K + b*256 + x_super_pos
        const __nv_bfloat16* x_block = x + (long long)m_base * K + b * 256 + x_super_pos;

        // Vectorize : load 4 BF16 per m row.
        uint2 xv[8];
        #pragma unroll
        for (int m = 0; m < 8; ++m) {
            // Out-of-range m-rows get a zero stub : safe because we only
            // write acc to y for m < m_cnt. Zero load avoids OOB read.
            if (m < m_cnt) {
                xv[m] = *(const uint2*)(x_block + m * K);
            } else {
                xv[m].x = 0; xv[m].y = 0;
            }
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
        __syncthreads();
    }

    // Reduction : 8 accumulators per thread × 64 threads (2 warps).
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
        if (m < m_cnt) {
            float total = sdata[m * 2] + sdata[m * 2 + 1];
            // Output : y[m_base + m, row]
            y[(long long)(m_base + m) * N + row] = (__nv_bfloat16)total;
        }
    }
}
"#;

// T246.10 A6b.1 — sgemm_q5k_bf16_mvar : Q5_K matmul with arbitrary batch M.
// Mirror of sgemm_q5k_bf16_m8 extended to M-variable. See SGEMM_Q4K_BF16_MVAR_SRC
// for the design rationale (W-amortization).
#[cfg(feature = "cuda")]
const SGEMM_Q5K_BF16_MVAR_SRC: &str = r#"
#include <cuda_bf16.h>
#include <cuda_fp16.h>

extern "C" __global__ void sgemm_q5k_bf16_mvar(
    const unsigned char* __restrict__ w_q5k,
    const __nv_bfloat16* __restrict__ x,
    __nv_bfloat16* __restrict__ y,
    int M,
    int N,
    int K
) {
    int row = blockIdx.x;
    if (row >= N) return;
    int tid = threadIdx.x;
    int blocks_per_row = K / 256;
    int row_offset = row * blocks_per_row * 176;
    int n_mtiles = (M + 7) >> 3;
    int mtile_idx = blockIdx.y;
    if (mtile_idx >= n_mtiles) return;
    int m_base = mtile_idx << 3;
    int m_cnt = M - m_base;
    if (m_cnt > 8) m_cnt = 8;

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
        const __nv_bfloat16* x_block = x + (long long)m_base * K + b * 256 + x_super_pos;

        uint2 xv[8];
        #pragma unroll
        for (int m = 0; m < 8; ++m) {
            if (m < m_cnt) {
                xv[m] = *(const uint2*)(x_block + m * K);
            } else {
                xv[m].x = 0; xv[m].y = 0;
            }
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
        if (m < m_cnt) {
            float total = sdata[m * 2] + sdata[m * 2 + 1];
            y[(long long)(m_base + m) * N + row] = (__nv_bfloat16)total;
        }
    }
}
"#;

// T246.10 A6b.1 — sgemm_q6k_bf16_mvar : Q6_K matmul with arbitrary batch M.
// Mirror of sgemm_q6k_bf16_m8 extended to M-variable.
#[cfg(feature = "cuda")]
const SGEMM_Q6K_BF16_MVAR_SRC: &str = r#"
#include <cuda_bf16.h>
#include <cuda_fp16.h>

extern "C" __global__ void sgemm_q6k_bf16_mvar(
    const unsigned char* __restrict__ w_q6k,
    const __nv_bfloat16* __restrict__ x,
    __nv_bfloat16* __restrict__ y,
    int M,
    int N,
    int K
) {
    int row = blockIdx.x;
    if (row >= N) return;
    int tid = threadIdx.x;
    int blocks_per_row = K / 256;
    int row_offset = row * blocks_per_row * 210;
    int n_mtiles = (M + 7) >> 3;
    int mtile_idx = blockIdx.y;
    if (mtile_idx >= n_mtiles) return;
    int m_base = mtile_idx << 3;
    int m_cnt = M - m_base;
    if (m_cnt > 8) m_cnt = 8;

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

        const __nv_bfloat16* x_base = x + (long long)m_base * K + b * 256 + half_offset_x;

        #pragma unroll
        for (int m = 0; m < 8; ++m) {
            if (m < m_cnt) {
                float x0 = (float)x_base[m * K + l];
                float x1 = (float)x_base[m * K + l + 32];
                float x2 = (float)x_base[m * K + l + 64];
                float x3 = (float)x_base[m * K + l + 96];
                acc[m] += w0 * x0 + w1 * x1 + w2 * x2 + w3 * x3;
            }
        }
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
        if (m < m_cnt) {
            float total = sdata[m * 2] + sdata[m * 2 + 1];
            y[(long long)(m_base + m) * N + row] = (__nv_bfloat16)total;
        }
    }
}
"#;

// T246.10 A6b.1 — sgemm_bf16_bf16_mvar : pure-BF16 weight matmul with arbitrary M.
// Mirrors sgemv_bf16_bf16 but produces M output rows per launch.
// Used for : SSM in_proj / out_proj / attention QKV/O if not quantized.
#[cfg(feature = "cuda")]
const SGEMM_BF16_BF16_MVAR_SRC: &str = r#"
#include <cuda_bf16.h>

extern "C" __global__ void sgemm_bf16_bf16_mvar(
    const __nv_bfloat16* __restrict__ w,
    const __nv_bfloat16* __restrict__ x,    // [M, K] row-major
    __nv_bfloat16* __restrict__ y,          // [M, N] row-major
    int M,
    int N,
    int K
) {
    int row = blockIdx.x;
    if (row >= N) return;
    int tid = threadIdx.x;
    int n_mtiles = (M + 7) >> 3;
    int mtile_idx = blockIdx.y;
    if (mtile_idx >= n_mtiles) return;
    int m_base = mtile_idx << 3;
    int m_cnt = M - m_base;
    if (m_cnt > 8) m_cnt = 8;

    extern __shared__ float shmem[];     // [8 m × 2 warps] = 16 floats

    float acc[8];
    #pragma unroll
    for (int m = 0; m < 8; ++m) acc[m] = 0.0f;

    int blocks_per_row = K / 256;
    int pos_base = tid * 4;
    int row_offset = row * K;

    for (int b = 0; b < blocks_per_row; ++b) {
        int k_off = b * 256 + pos_base;
        const __nv_bfloat16* w_ptr = w + row_offset + k_off;
        const __nv_bfloat16* x_base = x + (long long)m_base * K + k_off;

        // Load 4 BF16 of W (shared across all m).
        float wv0 = (float)w_ptr[0];
        float wv1 = (float)w_ptr[1];
        float wv2 = (float)w_ptr[2];
        float wv3 = (float)w_ptr[3];

        #pragma unroll
        for (int m = 0; m < 8; ++m) {
            if (m < m_cnt) {
                const __nv_bfloat16* x_ptr = x_base + m * K;
                acc[m] += wv0 * (float)x_ptr[0]
                        + wv1 * (float)x_ptr[1]
                        + wv2 * (float)x_ptr[2]
                        + wv3 * (float)x_ptr[3];
            }
        }
    }

    // Warp-shuffle reduction (each m independently).
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
            shmem[m * 2 + warp_id] = a;
        }
    }
    __syncthreads();
    if (tid < 8) {
        int m = tid;
        if (m < m_cnt) {
            float total = shmem[m * 2] + shmem[m * 2 + 1];
            y[(long long)(m_base + m) * N + row] = (__nv_bfloat16)total;
        }
    }
}
"#;

// T246.10 TrackG-lite — gemm_bf16_bf16_mma_m16n8k16 : BF16×BF16 GEMM via
// mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 tensor cores.
//
// Replaces sgemm_bf16_bf16_mvar (warp-shuffle scalar) on the M >= 16 batch
// regime. At M = 1..15 the warp-shuffle baseline still wins (per A3 negative
// result — broadcast wastes mma compute on SGEMV).
//
// Layout (matches sgemm_bf16_bf16_mvar) :
//   W : [N, K] BF16 row-major  (one row per output column n)
//   X : [M, K] BF16 row-major
//   Y : [M, N] BF16 row-major
//
// Block topology (canonical Ampere mma m16n8k16) :
//   - Each warp computes ONE 16(M) × 8(N) output tile in FP32 accumulator.
//   - 4 warps per block, each warp owns a different N-tile :
//       warp w (w in 0..3) → owns columns [n_base + 8*w .. n_base + 8*w + 7]
//   - block_dim = (32, 4, 1) = 128 threads
//   - grid_dim  = (ceil(N/32), ceil(M/16), 1)
//
// Smem tile :
//   A_smem[16][16] BF16 (X tile)           = 512 B
//   B_smem[32][16] BF16 (W tile, 4 warps)  = 1024 B
//   Total                                  = 1536 B per block (cheap)
//
// K loop : iterates over K in chunks of 16 BF16 (= 1 mma k-step).
//
// References :
//   - PTX ISA 9.2 §9.7.14.5 mma.sync.aligned.m16n8k16
//   - llama.cpp ggml/src/ggml-cuda/mma.cuh:1064-1077 (BF16 mma overload)
//   - C-fragment (16x8 FP32) layout : thread t holds D[(l/2)*8 + t/4]
//                                                    [(t%4)*2 + l%2] for l=0..3
//
// Constraints :
//   - K must be multiple of 16 (we enforce 256 in wrapper to keep parity with
//     the existing sgemm_bf16_bf16_mvar constraint).
//   - M and N can be arbitrary positive integers (boundary checks at write).
#[cfg(feature = "cuda")]
const GEMM_BF16_BF16_MMA_M16N8K16_SRC: &str = r#"
#include <cuda_bf16.h>

#define WARP_SIZE 32
#define WARPS_PER_BLOCK 4
#define BM 16
#define BN 32
#define BK 16

extern "C" __global__ void gemm_bf16_bf16_mma_m16n8k16(
    const __nv_bfloat16* __restrict__ w,    // [N, K] row-major
    const __nv_bfloat16* __restrict__ x,    // [M, K] row-major
    __nv_bfloat16* __restrict__ y,          // [M, N] row-major
    int M,
    int N,
    int K
) {
    const int m_base = blockIdx.y * BM;     // 16
    const int n_base = blockIdx.x * BN;     // 32
    const int warp_id = threadIdx.y;        // 0..3
    const int lane    = threadIdx.x;        // 0..31
    const int tid     = warp_id * WARP_SIZE + lane;
    // Each warp owns 8 N-cols starting at n_base + 8*warp_id.
    const int n_warp_base = n_base + warp_id * 8;

    // Shared memory : two tiles staged per K-chunk.
    //   A_smem layout : A_smem[m][k] (row-major) — 16 rows × 16 BF16
    //   B_smem layout : B_smem[n][k] (row-major) — 32 rows × 16 BF16
    __shared__ __nv_bfloat16 A_smem[BM][BK];
    __shared__ __nv_bfloat16 B_smem[BN][BK];

    // FP32 accumulator : per warp, 4 floats covering its 16x8 output tile.
    float Dx[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    // ------------------------------------------------------------------
    // Cooperative tile load helpers.
    //
    // A_smem load : 16 * 16 = 256 BF16 to load. 128 threads → 2 BF16/thread.
    //   We pack as one .u32 per pair of BF16, so 128 ints (16 rows × 8
    //   bf162 cols) loaded by 128 threads = 1 int / thread.
    //
    // B_smem load : 32 * 16 = 512 BF16 = 256 bf162. 128 threads → 2 bf162/thr.
    //   We do 2 ints per thread.
    // ------------------------------------------------------------------

    const int n_kchunks = K / BK;

    for (int kc = 0; kc < n_kchunks; ++kc) {
        const int k_base = kc * BK;

        // --- Load A tile : X[m_base .. m_base+15][k_base .. k_base+15] ---
        // 16 rows × 8 bf162 cols ; thread tid (0..127) loads element
        //   row = tid / 8        (0..15)
        //   col = (tid % 8) * 2  (0..14)
        {
            const int row = tid >> 3;
            const int col = (tid & 7) << 1;
            const int gm = m_base + row;
            __nv_bfloat16 a0 = __float2bfloat16(0.0f);
            __nv_bfloat16 a1 = __float2bfloat16(0.0f);
            if (gm < M) {
                const __nv_bfloat16* xp = x + (long long)gm * K + k_base + col;
                a0 = xp[0];
                a1 = xp[1];
            }
            A_smem[row][col]     = a0;
            A_smem[row][col + 1] = a1;
        }

        // --- Load B tile : W[n_base .. n_base+31][k_base .. k_base+15] ---
        // 32 rows × 8 bf162 cols = 256 bf162. 128 threads → 2 ints/thread.
        // Strategy : each thread loads 2 BF16 pairs.
        //   element index e = tid + p * 128 for p in {0,1}
        //   row = e / 8 ; col = (e % 8) * 2
        #pragma unroll
        for (int p = 0; p < 2; ++p) {
            const int e   = tid + p * 128;
            const int row = e >> 3;          // 0..31
            const int col = (e & 7) << 1;    // 0..14 even
            const int gn  = n_base + row;
            __nv_bfloat16 b0 = __float2bfloat16(0.0f);
            __nv_bfloat16 b1 = __float2bfloat16(0.0f);
            if (gn < N) {
                const __nv_bfloat16* wp = w + (long long)gn * K + k_base + col;
                b0 = wp[0];
                b1 = wp[1];
            }
            B_smem[row][col]     = b0;
            B_smem[row][col + 1] = b1;
        }

        __syncthreads();

        // --- Build A register fragment (tile<16,8,bf162>, ne=4 ints/thread).
        //
        // Canonical Ampere A-fragment thread layout for m16n8k16 (from
        // llama.cpp mma.cuh tile<16,8,bf162>::get_i / get_j) :
        //   Per thread holds 4 .b32 = 4 bf162 = 8 BF16. Element l (0..3) :
        //     get_i(l) = ((l & 1) << 3) + (lane >> 2)   row in A     (0..15)
        //     get_j(l) = ((l >> 1) << 2) + (lane & 3)   bf162 col    (0..7)
        //   Equivalently, BF16 K-col = 2 * get_j(l).
        //
        //   l = 0 : row = lane/4 ,     k_bf16_col = (lane%4)*2
        //   l = 1 : row = lane/4 + 8 , k_bf16_col = (lane%4)*2
        //   l = 2 : row = lane/4 ,     k_bf16_col = (lane%4)*2 + 8
        //   l = 3 : row = lane/4 + 8 , k_bf16_col = (lane%4)*2 + 8
        //
        // Each .b32 holds (BF16[col], BF16[col+1]) — natural bf162 layout
        // from row-major smem A_smem[row][col].
        unsigned int A0, A1, A2, A3;
        {
            const int lane_div_4 = lane >> 2;        // 0..7
            const int lane_mod_4 = lane & 3;         // 0..3
            const int row_lo = lane_div_4;           // for l = 0, 2
            const int row_hi = lane_div_4 + 8;       // for l = 1, 3
            const int k_col_lo = (lane_mod_4 << 1);          // 0..6 even, l = 0, 1
            const int k_col_hi = (lane_mod_4 << 1) + 8;      // 8..14 even, l = 2, 3
            A0 = *reinterpret_cast<const unsigned int*>(&A_smem[row_lo][k_col_lo]);
            A1 = *reinterpret_cast<const unsigned int*>(&A_smem[row_hi][k_col_lo]);
            A2 = *reinterpret_cast<const unsigned int*>(&A_smem[row_lo][k_col_hi]);
            A3 = *reinterpret_cast<const unsigned int*>(&A_smem[row_hi][k_col_hi]);
        }

        // --- Build B register fragment (2 .b32 ints/thread, .col major).
        //
        // Canonical PTX B-fragment thread layout for m16n8k16 (PTX ISA
        // 9.7.14.6, B operand for m16n8k16) :
        //   groupID = laneid / 4   ; threadID_in_group = laneid % 4
        //   b[0..1] : (row K = tg*2, col N = groupID), (row K = tg*2+1, col N = groupID)
        //             → packed as R0 (bf162 over 2 K rows for one N col)
        //   b[2..3] : (row K = tg*2+8, col N = groupID), (row K = tg*2+9, col N = groupID)
        //             → packed as R1
        //
        // Per-thread :
        //   R0 = bf162 at  N col = lane/4 , BF16 K cols = [(lane%4)*2, (lane%4)*2+1]
        //   R1 = bf162 at  N col = lane/4 , BF16 K cols = [(lane%4)*2+8, (lane%4)*2+9]
        //
        // B_smem is stored [N row][K BF16 col] row-major over N. Reading
        // `*(uint32_t*)&B_smem[n][k]` packs (B_smem[n][k], B_smem[n][k+1])
        // as a bf162 — i.e. 2 BF16 over consecutive K cols for one N row,
        // which matches the required B fragment packing (2 K rows for one
        // N col). Each warp owns N ∈ [warp_id*8 .. warp_id*8+7].
        unsigned int B0, B1;
        {
            const int lane_div_4 = lane >> 2;            // 0..7  → N within warp
            const int lane_mod_4 = lane & 3;             // 0..3  → K row pair index
            const int b_n_row = warp_id * 8 + lane_div_4;        // 0..31 in B_smem
            const int k_bf16_col_lo = lane_mod_4 << 1;           // 0,2,4,6
            const int k_bf16_col_hi = (lane_mod_4 << 1) + 8;     // 8,10,12,14
            B0 = *reinterpret_cast<const unsigned int*>(&B_smem[b_n_row][k_bf16_col_lo]);
            B1 = *reinterpret_cast<const unsigned int*>(&B_smem[b_n_row][k_bf16_col_hi]);
        }

        // --- mma.sync m16n8k16 BF16×BF16 → FP32 accumulate in-place ---
        unsigned int *Dxi = reinterpret_cast<unsigned int*>(Dx);
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
            : "+r"(Dxi[0]), "+r"(Dxi[1]), "+r"(Dxi[2]), "+r"(Dxi[3])
            : "r"(A0), "r"(A1), "r"(A2), "r"(A3),
              "r"(B0), "r"(B1)
        );

        __syncthreads();
    }

    // --- Write Y from D fragment ---
    // C-fragment (16x8 FP32) layout : ne=4, per llama.cpp `get_i`/`get_j` :
    //   get_i(l) = (l/2)*8 + lane/4    (0..15) → m offset in 16-row tile
    //   get_j(l) = (lane%4)*2 + l%2    (0..7)  → n offset in 8-col tile (this warp)
    #pragma unroll
    for (int l = 0; l < 4; ++l) {
        const int mi = ((l >> 1) << 3) + (lane >> 2);     // 0..15
        const int nj = ((lane & 3) << 1) + (l & 1);       // 0..7
        const int gm = m_base + mi;
        const int gn = n_warp_base + nj;
        if (gm < M && gn < N) {
            y[(long long)gm * N + gn] = __float2bfloat16(Dx[l]);
        }
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

// T246.8 A3 — sgemv_bf16_bf16_v2 : multi-row block SGEMV (BIT-EXACT with V1).
//
// Mirrors V1's per-thread arithmetic exactly to guarantee bit-exact match
// for unaltered argmax decode parity, while reducing block-launch count 4×
// by handling 4 output rows per block (one warp per row).
//
// Design (revised after mma.sync m16n8k16 attempt showed 30% slowdown in
// end-to-end model — see note 0418e02c) :
//
// 1 block = 4 warps × 64 threads = 256 threads. EACH WARP-PAIR (64 threads,
// = 1 "row warp") handles ONE output row, 4 rows per block. Within each
// row-warp the per-thread MAC pattern is BIT-IDENTICAL to V1 :
//
//   per-row warp (64 threads, tid in 0..63) :
//     pos_base = tid * 4
//     for s in 0..K/256 :
//       k_off = s * 256 + pos_base
//       acc += W[row, k_off+0..3] * x[k_off+0..3]
//     warp-shuffle reduce, lane 0 stores y[row]
//
// This is V1's exact arithmetic — same per-thread MAC sequence, same
// warp reduction — just packaged 4-rows-per-block instead of 1-per-block.
// The benefit comes from :
//   - 4× fewer block launches (less SM scheduling overhead)
//   - L1 cache reuse for x : 4 rows per block all read the same x[k_off..]
//     pattern → L1 hits on x for warps 1..3 of each block (warp 0 misses).
//
// Constraint : K must be multiple of 256 (V1's constraint, kept identical
// for bit-exact MAC sequence).
//
// Per-block layout : block_dim = (64, 4, 1) — 64 lanes × 4 row-warps.
//   threadIdx.x = lane (0..63 within row's warp-pair)
//   threadIdx.y = row_in_block (0..3)
//   row = blockIdx.x * 4 + threadIdx.y
//
// Note : (64, 4, 1) launches 256 threads = 8 hardware warps per block. Each
// "row warp-pair" is actually 2 hardware warps (lanes 0..31 and 32..63). The
// reduction must work across this 64-thread group, not just 32. We use the
// V1 pattern : warp-shuffle within 32 lanes, then shared-mem combine of the
// 2 halves.
#[cfg(feature = "cuda")]
const SGEMV_BF16_BF16_V2_SRC: &str = r#"
#include <cuda_bf16.h>

extern "C" __global__ void sgemv_bf16_bf16_v2(
    const __nv_bfloat16* __restrict__ w,    // [N, K] row-major
    const __nv_bfloat16* __restrict__ x,    // [K]
    __nv_bfloat16* __restrict__ y,          // [N]
    int N,
    int K
) {
    const int row_in_block = threadIdx.y;          // 0..3
    const int row = blockIdx.x * 4 + row_in_block;
    if (row >= N) return;
    const int tid = threadIdx.x;                   // 0..63 (V1 pattern)

    // Shared memory : 2-halfwarp reduction buffer per row.
    // [4 rows][2 half-warps] floats = 32 bytes total
    extern __shared__ float sdata[];               // size : 4*2 = 8 floats
    float* row_sdata = sdata + row_in_block * 2;

    float acc = 0.0f;
    const int blocks_per_row = K / 256;
    const int pos_base = tid * 4;
    const int row_offset = row * K;

    // V1's exact MAC loop — preserved bit-exact.
    for (int b = 0; b < blocks_per_row; ++b) {
        const int k_off = b * 256 + pos_base;
        const __nv_bfloat16* w_ptr = w + row_offset + k_off;
        const __nv_bfloat16* x_ptr = x + k_off;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            float wv = (float)w_ptr[i];
            float xv = (float)x_ptr[i];
            acc += wv * xv;
        }
    }

    // Warp-shuffle reduction (same as V1 : __shfl_down_sync to stay bit-exact).
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xffffffff, acc, offset);
    }
    const int warp_id = tid >> 5;     // 0 or 1 (within this row's 64-thread group)
    const int lane_id = tid & 31;
    if (lane_id == 0) {
        row_sdata[warp_id] = acc;
    }
    __syncthreads();
    if (tid == 0) {
        const float total = row_sdata[0] + row_sdata[1];
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

// T246.8 A1.1 — Fused SSM pre-step elementwise chain.
//
// Replaces 4 separate kernel launches per SSM layer per token by 1 fused
// kernel :
//   1. beta[i]  = sigmoid(beta[i])                   for i in 0..n
//   2. alpha[i] = alpha[i] + dt_bias[i]              for i in 0..n
//   3. alpha[i] = softplus(alpha[i])                  for i in 0..n
//   4. alpha[i] = alpha[i] * ssm_a[i]                 for i in 0..n
//
// All four operate on the same length-n bf16 vectors, so each thread
// handles exactly one element with the same FP order as the unfused
// chain (each unfused kernel writes its result back to bf16 before the
// next reads it — we mirror that bit-exact by round-tripping through
// bf16 between phases).
//
// Bit-exact parity with the four-kernel sequence is preserved because
// each thread independently performs the same scalar steps on its own
// element, and the bf16 round-trip after every step matches the
// unfused version's writeback.
#[cfg(feature = "cuda")]
const SSM_PRE_STEP_BF16_SRC: &str = r#"
#include <cuda_bf16.h>

extern "C" __global__ void ssm_pre_step_bf16(
    __nv_bfloat16*       __restrict__ alpha,    // [n]   in/out
    __nv_bfloat16*       __restrict__ beta,     // [n]   in/out
    const __nv_bfloat16* __restrict__ dt_bias,  // [n]
    const __nv_bfloat16* __restrict__ ssm_a,    // [n]
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    // Step 1 — sigmoid(beta) inplace, mirroring sigmoid_inplace_bf16.
    {
        float v = (float)beta[i];
        float s = 1.0f / (1.0f + expf(-v));
        beta[i] = (__nv_bfloat16)s;
    }

    // Step 2 — alpha += dt_bias, mirroring add_inplace_bf16.
    float a;
    {
        float ya = (float)alpha[i];
        float xa = (float)dt_bias[i];
        a = ya + xa;
        alpha[i] = (__nv_bfloat16)a;
    }

    // Step 3 — softplus(alpha) inplace, mirroring softplus_inplace_bf16.
    {
        // Re-read from bf16 to match unfused FP order exactly.
        float v = (float)alpha[i];
        float am = v > 0.0f ? v : 0.0f;
        float bm = log1pf(expf(-fabsf(v)));
        alpha[i] = (__nv_bfloat16)(am + bm);
    }

    // Step 4 — alpha *= ssm_a, mirroring mul_inplace_bf16.
    {
        float ya = (float)alpha[i];
        float xa = (float)ssm_a[i];
        alpha[i] = (__nv_bfloat16)(ya * xa);
    }
}
"#;

// T246.8 A1.2 — Fused SSM post-step output processing.
//
// Replaces 3 separate kernel launches per SSM layer per token by 1 fused
// kernel :
//   1. ssm_norm : RMSNorm on `out` per head (n_v batches of head_kv each),
//                 with shared `gamma`. Mirrors rms_norm_bf16(out, sn,
//                 head_kv, n_v).
//   2. silu(z) inplace on length-(n_v * head_kv) bf16 vector, mirroring
//                 silu_bf16.
//   3. out *= silu(z) elementwise, mirroring mul_inplace_bf16.
//
// One block per head : block.x = head_kv threads (one per channel within
// a head), grid.x = n_v heads. The reduction over head_kv mirrors the
// existing rms_norm_bf16 reduction (tree reduction in shared memory),
// so the resulting inv_rms is bit-identical to the unfused call.
//
// Each thread then :
//   * reads original out[h, c] (loaded into shmem before reduction),
//   * applies inv_rms * gamma[c] → ssm_norm result,
//   * computes silu(z[h, c]) = z / (1 + exp(-z)),
//   * writes out[h, c] = ssm_norm * silu(z),
//   * writes z[h, c] = silu(z) (matches the inplace silu writeback).
#[cfg(feature = "cuda")]
const SSM_POST_STEP_BF16_SRC: &str = r#"
#include <cuda_bf16.h>

extern "C" __global__ void ssm_post_step_bf16(
    __nv_bfloat16*       __restrict__ out,    // [n_v, head_kv]   in/out (norm * silu(z))
    __nv_bfloat16*       __restrict__ z,      // [n_v, head_kv]   in/out (becomes silu(z))
    const __nv_bfloat16* __restrict__ gamma,  // [head_kv]
    float eps,
    int head_kv
) {
    int h   = blockIdx.x;
    int tid = threadIdx.x;
    if (tid >= head_kv) return;

    extern __shared__ float sdata[];

    // Load out[h, tid] and compute square (rms_norm_bf16 reduction).
    float v   = (float)out[h * head_kv + tid];
    float vsq = v * v;
    sdata[tid] = vsq;
    __syncthreads();

    // Tree reduction over head_kv threads (same shape as rms_norm_bf16).
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    // sdata[0] = sum of squares ; rms_norm uses (sum/n + eps).
    float inv_rms = rsqrtf(sdata[0] / (float)head_kv + eps);

    // Step 1 — out = ssm_norm(out, gamma) (mirrors rms_norm_bf16
    // writeback : v * inv_rms * gamma[tid]).
    float normed = v * inv_rms * (float)gamma[tid];
    // Round-trip through bf16 to match the unfused intermediate,
    // then re-read as float for the multiply (bit-identical to
    // rms_norm_bf16 then mul_inplace_bf16).
    __nv_bfloat16 normed_bf = (__nv_bfloat16)normed;
    float normed_f = (float)normed_bf;

    // Step 2 — z = silu(z) inplace (mirrors silu_bf16).
    float zv = (float)z[h * head_kv + tid];
    float sz = zv / (1.0f + expf(-zv));
    __nv_bfloat16 sz_bf = (__nv_bfloat16)sz;
    z[h * head_kv + tid] = sz_bf;
    float sz_f = (float)sz_bf;

    // Step 3 — out *= silu(z) (mirrors mul_inplace_bf16).
    out[h * head_kv + tid] = (__nv_bfloat16)(normed_f * sz_f);
}
"#;

#[cfg(feature = "cuda")]
const DELTA_NET_STEP_TREE_BF16_SRC: &str = r#"
#include <cuda_bf16.h>

// T246.7 TrackC.1 — Tree-aware Gated DeltaNet recurrent step.
//
// Mirrors `delta_net_step_bf16` exactly but operates on a draft tree :
// each tree node carries its own SSM state slot. The kernel reads the
// *parent's* state (per the `parents[]` array, BFS order, parents[i] < i
// for i > 0) into a thread-local copy of the [head_dim] row, applies the
// same delta-net update, and writes the resulting state into the *child's*
// slot. This way per-branch state forking is automatic : two siblings
// sharing parent p both fork from `tree_states[p]` independently.
//
// Layout (tree-aware) :
//   q, k, v       : [tree_size, n_heads, head_dim] BF16
//   gate, beta    : [tree_size, n_heads]            BF16 (per-row scalars)
//   parents       : [tree_size]                     i32 (-1 for root, 0..i for i>0)
//   tree_states   : [tree_size, n_heads, head_dim, head_dim] BF16
//   out           : [tree_size, n_heads, head_dim] BF16
//
// IMPORTANT — root state (`parents[i] == -1`) reads from `tree_states[i]`
// itself. The Rust caller MUST pre-load the model's current per-layer SSM
// state into `tree_states[root_index]` (typically index 0) BEFORE calling
// this kernel. After acceptance the caller copies the deepest accepted
// node's slot back into the model's per-layer SSM state.
//
// Each TG handles ONE (wave-position, head) pair, with head_dim threads each
// handling ONE row r of the state[h] matrix. With wave_size=1 +
// wave_indices=[0] + parents[0]=-1 + state pre-loaded into slot 0, this
// MUST produce a result bit-exact equivalent to the scalar
// `delta_net_step_bf16` kernel (parity gate).
//
// CRITICAL — caller MUST launch one kernel per BFS depth wave. All nodes
// at depth d MUST execute strictly after all nodes at depth d-1 (so child
// reads see committed parent writes). Because BFS guarantees parents[i] < i
// and depths[parent] < depths[child], grouping by depth and serializing
// across depths via separate kernel launches gives the necessary read-
// after-write ordering. Within a wave, all nodes have parents in earlier
// waves so there is no intra-wave ordering requirement.
extern "C" __global__ void delta_net_step_tree_bf16(
    const __nv_bfloat16* __restrict__ q,           // [tree_size, n_heads, head_dim]
    const __nv_bfloat16* __restrict__ k,           // [tree_size, n_heads, head_dim]
    const __nv_bfloat16* __restrict__ v,           // [tree_size, n_heads, head_dim]
    const __nv_bfloat16* __restrict__ gate,        // [tree_size, n_heads]
    const __nv_bfloat16* __restrict__ beta,        // [tree_size, n_heads]
    const int*           __restrict__ parents,     // [tree_size]
    const int*           __restrict__ wave_indices,// [wave_size] — tree-row indices in this depth wave
    __nv_bfloat16*       __restrict__ tree_states, // [tree_size, n_heads, head_dim, head_dim]
    __nv_bfloat16*       __restrict__ out,         // [tree_size, n_heads, head_dim]
    int wave_size,
    int n_heads,
    int head_dim
) {
    int wp = blockIdx.y;
    int h  = blockIdx.x;
    if (wp >= wave_size || h >= n_heads) return;
    int tr = wave_indices[wp];      // tree row
    int r  = threadIdx.x;           // state-matrix row
    if (r >= head_dim) return;

    long long state_per_node = (long long)n_heads * head_dim * head_dim;
    long long io_per_node    = (long long)n_heads * head_dim;
    long long g_per_node     = (long long)n_heads;

    int parent = parents[tr];
    long long src_node = (parent < 0) ? (long long)tr : (long long)parent;

    long long base_io  = (long long)tr * io_per_node + (long long)h * head_dim;
    long long base_g   = (long long)tr * g_per_node  + h;
    long long base_dst = (long long)tr * state_per_node + (long long)h * head_dim * head_dim
                         + (long long)r * head_dim;
    long long base_src = src_node * state_per_node + (long long)h * head_dim * head_dim
                         + (long long)r * head_dim;

    float g_exp = expf((float)gate[base_g]);
    float b     = (float)beta[base_g];
    float v_r   = (float)v[base_io + r];

    // Broadcast q across threads in this TG via shmem.
    extern __shared__ float q_shared[];
    if (r < head_dim) {
        q_shared[r] = (float)q[base_io + r];
    }
    __syncthreads();

    // Stream over `c`. For src_node == tr (root) the read and write hit the
    // SAME address — pure same-thread RMW, no aliasing (thread r owns row r
    // exclusively across the whole TG and across the kernel grid since each
    // (tr, h) pair is a unique TG). For src_node != tr the addresses are in
    // disjoint slots → also safe. Layout matches the scalar kernel.
    float out_acc = 0.0f;
    for (int c = 0; c < head_dim; ++c) {
        float k_c     = (float)k[base_io + c];
        float old     = (float)tree_states[base_src + c];
        float updated = g_exp * old + b * v_r * k_c;
        tree_states[base_dst + c] = (__nv_bfloat16)updated;
        out_acc += updated * q_shared[c];
    }

    out[base_io + r] = (__nv_bfloat16)out_acc;
}
"#;

// T246.10 TrackF — Column-parallel variant of `delta_net_step_tree_bf16`.
//
// Mathematically identical to the baseline. The baseline uses
// `threadIdx.x = r` (state row), which produces strided global accesses
// to `tree_states[h, r, c]` because adjacent threads differ in `r`
// (256-byte stride) — for each `c` iteration the 128 threads in a warp
// touch 128 separate cache lines, wasting most of the BW.
//
// This kernel swaps the role : `threadIdx.x = c` (state column). For
// each `r` in the inner loop, the 128 threads read state[h, r, 0..127]
// at adjacent addresses (stride = 1 BF16) → fully coalesced (8 sectors
// for one row). The same applies to k[c], q[c] (now per-thread instead
// of broadcast through shared memory) and the writeback.
//
// To produce `out[h, r] = Σ_c state[h, r, c] * q[c]` we still need a
// block-wide reduction across the 128 column-threads. The reduction is
// done per row via warp-shuffle + small shared-mem inter-warp combine,
// then thread 0 writes the final out value for row `r`. The cost of
// 128 small reductions (one per row) is amortized by the 8× BW saving
// on the state RMW, which is the dominant term.
//
// Output behavior, parents traversal, root-state read-from-self and
// out tensor aliasing semantics MATCH the baseline kernel exactly.
//
// Bit-exact parity vs baseline is NOT guaranteed because the reduction
// is performed in a different order (warp-shuffle tree vs per-row
// sequential FMAs) ; FP32 add is non-associative, so the last few
// mantissa bits of `out` may differ. We assert a `≥ 99.5 %` BF16 bit-
// match + max abs delta ≤ 0.005 in the parity test.
#[cfg(feature = "cuda")]
const DELTA_NET_STEP_TREE_BF16_OPT_SRC: &str = r#"
#include <cuda_bf16.h>

extern "C" __global__ void delta_net_step_tree_bf16_opt(
    const __nv_bfloat16* __restrict__ q,           // [tree_size, n_heads, head_dim]
    const __nv_bfloat16* __restrict__ k,           // [tree_size, n_heads, head_dim]
    const __nv_bfloat16* __restrict__ v,           // [tree_size, n_heads, head_dim]
    const __nv_bfloat16* __restrict__ gate,        // [tree_size, n_heads]
    const __nv_bfloat16* __restrict__ beta,        // [tree_size, n_heads]
    const int*           __restrict__ parents,     // [tree_size]
    const int*           __restrict__ wave_indices,// [wave_size]
    __nv_bfloat16*       __restrict__ tree_states, // [tree_size, n_heads, head_dim, head_dim]
    __nv_bfloat16*       __restrict__ out,         // [tree_size, n_heads, head_dim]
    int wave_size,
    int n_heads,
    int head_dim
) {
    int wp = blockIdx.y;
    int h  = blockIdx.x;
    if (wp >= wave_size || h >= n_heads) return;
    int tr = wave_indices[wp];
    int c  = threadIdx.x;
    if (c >= head_dim) return;

    long long state_per_node = (long long)n_heads * head_dim * head_dim;
    long long io_per_node    = (long long)n_heads * head_dim;
    long long g_per_node     = (long long)n_heads;

    int parent = parents[tr];
    long long src_node = (parent < 0) ? (long long)tr : (long long)parent;

    long long base_io  = (long long)tr * io_per_node + (long long)h * head_dim;
    long long base_g   = (long long)tr * g_per_node  + h;
    long long base_h_dst = (long long)tr       * state_per_node + (long long)h * head_dim * head_dim;
    long long base_h_src = src_node            * state_per_node + (long long)h * head_dim * head_dim;

    float g_exp = expf((float)gate[base_g]);
    float b     = (float)beta[base_g];

    // Each thread caches its own k[c] and q[c] in registers.
    float k_c = (float)k[base_io + c];
    float q_c = (float)q[base_io + c];

    // Inter-warp reduction scratchpad (one float per warp ≤ 4 warps
    // for head_dim=128). 4 × 4 = 16 bytes ; we allocate `head_dim/32`
    // dynamically via the shared_mem_bytes launch param.
    extern __shared__ float warp_sums[];

    const int lane    = c & 31;
    const int warp_id = c >> 5;
    const int n_warps = (head_dim + 31) >> 5;
    const int warp_size = 32;

    // Active mask + effective lane count for the intra-warp reduction.
    // For head_dim ≥ 32 the warp is fully populated → full mask.
    // For head_dim < 32 only the first head_dim lanes participate.
    // NOTE : `1u << 32` is undefined behavior in C — guard explicitly.
    unsigned int warp_active;
    int warp_eff;
    if (head_dim >= warp_size) {
        warp_active = 0xffffffffu;
        warp_eff = warp_size;
    } else if (head_dim == 0) {
        warp_active = 0u;
        warp_eff = 0;
    } else {
        warp_active = (1u << head_dim) - 1u;
        warp_eff = head_dim;
    }

    // Walk all rows ; for each row, this thread reads its (r, c) state cell
    // — addresses are consecutive across threads → coalesced.
    for (int r = 0; r < head_dim; ++r) {
        long long addr_src = base_h_src + (long long)r * head_dim + c;
        long long addr_dst = base_h_dst + (long long)r * head_dim + c;

        float v_r = (float)v[base_io + r];
        float old = (float)tree_states[addr_src];

        float updated = g_exp * old + b * v_r * k_c;
        tree_states[addr_dst] = (__nv_bfloat16)updated;

        // partial contribution to out[h, r] from this column.
        float partial = updated * q_c;

        // Warp-reduce within the warp. Use the active mask so that
        // shuffles are well-defined when head_dim < 32.
        for (int off = warp_eff >> 1; off > 0; off >>= 1) {
            partial += __shfl_xor_sync(warp_active, partial, off);
        }
        // lane 0 of each warp holds the warp sum.
        if (lane == 0) {
            warp_sums[warp_id] = partial;
        }
        __syncthreads();

        // Warp 0 finalizes : lane i loads warp_sums[i] for i in 0..n_warps,
        // (lanes beyond n_warps load 0.0f) and lane 0 writes out[h, r].
        //
        // Only warp 0 executes this path. If head_dim < 32 then we never
        // launched a second warp at all — warp 0 still has `warp_size`
        // active lanes thanks to block_dim ≤ head_dim and head_dim ≤
        // warp_size in that case (we'd be in the "single warp" regime
        // already accounted for above).
        if (warp_id == 0 && warp_eff == warp_size) {
            float ws = (lane < n_warps) ? warp_sums[lane] : 0.0f;
            // All 32 lanes of warp 0 participate ; lanes ≥ n_warps just
            // contribute 0 to the reduction. We can therefore use a full
            // 32-lane shuffle mask safely.
            for (int off = 16; off > 0; off >>= 1) {
                ws += __shfl_xor_sync(0xffffffffu, ws, off);
            }
            if (lane == 0) {
                out[base_io + r] = (__nv_bfloat16)ws;
            }
        } else if (warp_id == 0) {
            // Single-warp case (head_dim ≤ 32) : the warp reduction above
            // already produced the final sum in lane 0. Just write it.
            if (lane == 0) {
                out[base_io + r] = (__nv_bfloat16)partial;
            }
        }
        __syncthreads();
    }
}
"#;

// NEW-SSM — bandwidth-saturated rewrite of `delta_net_step_tree_bf16`.
//
// Per Phase 1 nsys baseline (note `7ef960d7`) :
//   * baseline kernel reads + writes 1.5 MB state per (slot, head) per call
//     and averages 40.5 µs, achieving ~74 GB/s effective HBM = 37 % of the
//     200 GB/s GB10 peak.
//   * the bottleneck is uncoalesced state RMW : baseline uses
//     `threadIdx.x = r` so adjacent lanes hit cache lines 256 B apart
//     (stride = head_dim * 2 B), wasting 15/16 of every loaded sector.
//   * the failed TrackF `_opt` variant switched to `threadIdx.x = c` (which
//     coalesces) but paid for it with 128 block-wide reductions per launch
//     including `__syncthreads()` — net regression −7 %.
//
// This kernel keeps coalesced state access AND eliminates the block-wide
// sync :
//
//   * Block layout : `(WARP_SIZE=32 lanes) × (n_warps=4 row-warps)` =
//     128 threads (when head_dim=128). Each warp OWNS a disjoint subset
//     of state rows (rows_per_warp = head_dim / n_warps = 32). No inter-
//     warp dependence ⇒ no `__syncthreads()` anywhere in the hot loop.
//
//   * Within a warp the 32 lanes parallelize over state columns. For
//     head_dim=128 each lane handles cols_per_lane = 4 consecutive
//     columns (cols 4*lane .. 4*lane+3), accessed via two packed
//     `__nv_bfloat162` loads per row.
//
//   * State R/W per row per warp : 32 lanes × 4 BF16 = 256 B = exactly
//     one 128-byte aligned cache sector pair, fully coalesced.
//
//   * Per-row output reduction = one 32-lane `__shfl_xor_sync` tree
//     (5 instructions, register-only). Lane 0 of the warp writes
//     `out[h, r]` for its owned row. No smem, no sync.
//
//   * Per-thread caches : g_exp, b are scalar (per head). k[4 cols]
//     and q[4 cols] live in registers for the entire kernel (each
//     lane owns its own slice). v[r] is reloaded per row inside the
//     warp's row loop (n_v / n_warps = 32 BF16 each = 64 B, trivial).
//
// Numerical equivalence : the FP32 sum order for `out[h, r]` differs
// from the baseline (warp tree vs sequential FMAs along c), so
// bit-exact parity is NOT guaranteed. We assert the same loose
// tolerance as the `_opt` test : ≥ 95 % BF16 bit-match on out + state
// + max abs delta ≤ 0.05.
//
// Launch contract :
//   * grid = (n_heads, wave_size, 1)
//   * block = (WARP_SIZE, n_warps, 1) where n_warps = head_dim / WARP_SIZE
//     when head_dim ≥ WARP_SIZE (which is always true here, head_dim ∈
//     {32, 64, 128, 256}). If head_dim < WARP_SIZE we fall back to a
//     single warp with cols_per_lane = head_dim/WARP_SIZE (or 1 if
//     head_dim < WARP_SIZE — gracefully handled).
//   * shared_mem_bytes = 0 (none used).
//
// Assumptions :
//   * head_dim is a multiple of WARP_SIZE (32). Qwen3.6 head_kv = 128
//     ⇒ n_warps = 4, cols_per_lane = 4. If a future model violates this
//     we'd need to extend the kernel.
//   * (head_dim / WARP_SIZE) divides head_dim ⇒ rows_per_warp = head_dim
//     / n_warps is integer.
#[cfg(feature = "cuda")]
const DELTA_NET_STEP_TREE_BF16_V2_SRC: &str = r#"
#include <cuda_bf16.h>

extern "C" __global__ void delta_net_step_tree_bf16_v2(
    const __nv_bfloat16* __restrict__ q,           // [tree_size, n_heads, head_dim]
    const __nv_bfloat16* __restrict__ k,           // [tree_size, n_heads, head_dim]
    const __nv_bfloat16* __restrict__ v,           // [tree_size, n_heads, head_dim]
    const __nv_bfloat16* __restrict__ gate,        // [tree_size, n_heads]
    const __nv_bfloat16* __restrict__ beta,        // [tree_size, n_heads]
    const int*           __restrict__ parents,     // [tree_size]
    const int*           __restrict__ wave_indices,// [wave_size]
    __nv_bfloat16*       __restrict__ tree_states, // [tree_size, n_heads, head_dim, head_dim]
    __nv_bfloat16*       __restrict__ out,         // [tree_size, n_heads, head_dim]
    int wave_size,
    int n_heads,
    int head_dim
) {
    const int WARP_SIZE = 32;
    int wp = blockIdx.y;
    int h  = blockIdx.x;
    if (wp >= wave_size || h >= n_heads) return;

    int lane    = threadIdx.x;        // 0..WARP_SIZE-1, column-thread
    int warp_id = threadIdx.y;        // 0..n_warps-1, row-warp id

    // Layout assumes head_dim is a multiple of WARP_SIZE.
    // cols_per_lane = head_dim / WARP_SIZE, rows_per_warp = head_dim / n_warps.
    int n_warps         = blockDim.y;
    int cols_per_lane   = head_dim / WARP_SIZE;       // e.g. 128/32 = 4
    int rows_per_warp   = head_dim / n_warps;          // e.g. 128/4 = 32
    int lane_col_base   = lane * cols_per_lane;        // first c owned by this lane
    int warp_row_base   = warp_id * rows_per_warp;     // first r owned by this warp

    int tr = wave_indices[wp];

    long long state_per_node = (long long)n_heads * head_dim * head_dim;
    long long io_per_node    = (long long)n_heads * head_dim;
    long long g_per_node     = (long long)n_heads;

    int parent = parents[tr];
    long long src_node = (parent < 0) ? (long long)tr : (long long)parent;

    long long base_io   = (long long)tr      * io_per_node    + (long long)h * head_dim;
    long long base_g    = (long long)tr      * g_per_node     + h;
    long long base_h_dst = (long long)tr      * state_per_node + (long long)h * head_dim * head_dim;
    long long base_h_src = src_node                  * state_per_node + (long long)h * head_dim * head_dim;

    float g_exp = expf((float)gate[base_g]);
    float b     = (float)beta[base_g];

    // Per-lane caches : `cols_per_lane` (≤8) consecutive cols of k and q.
    // Held in scalar registers. Loaded once per kernel using BF16x2 packed
    // loads (one __nv_bfloat162 = 2 BF16) when cols_per_lane is even.
    float k_cols[8];
    float q_cols[8];

    #pragma unroll
    for (int j = 0; j < 8; j += 2) {
        if (j + 1 < cols_per_lane) {
            int c = lane_col_base + j;
            __nv_bfloat162 kk = *reinterpret_cast<const __nv_bfloat162*>(&k[base_io + c]);
            __nv_bfloat162 qq = *reinterpret_cast<const __nv_bfloat162*>(&q[base_io + c]);
            k_cols[j    ] = __low2float(kk);
            k_cols[j + 1] = __high2float(kk);
            q_cols[j    ] = __low2float(qq);
            q_cols[j + 1] = __high2float(qq);
        } else if (j < cols_per_lane) {
            int c = lane_col_base + j;
            k_cols[j] = (float)k[base_io + c];
            q_cols[j] = (float)q[base_io + c];
        } else {
            k_cols[j    ] = 0.0f; k_cols[j + 1] = 0.0f;
            q_cols[j    ] = 0.0f; q_cols[j + 1] = 0.0f;
        }
    }

    // Pre-multiply b*k_cols once — constant across rows. This lets the
    // inner FMA chain become `updated = g_exp*old + v_r * bk_cols[j]`,
    // saving 1 FMA per (row, col) without changing the FP order.
    float bk_cols[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        bk_cols[j] = b * k_cols[j];
    }

    // Process owned rows two-at-a-time to extract memory ILP. Each
    // inner iteration issues 2 independent state loads/stores per
    // packed BF162 ⇒ the LSU can have ~4 transactions in flight,
    // hiding more of the HBM round-trip latency.
    //
    // The 2 rows are independent : separate state addresses, separate
    // v_r values, separate partial accumulators. The warp reduction is
    // still per-row so two `__shfl_xor_sync` chains execute at the end.
    int r_off = 0;
    for (; r_off + 1 < rows_per_warp; r_off += 2) {
        int r0 = warp_row_base + r_off;
        int r1 = warp_row_base + r_off + 1;
        float v_r0 = (float)v[base_io + r0];
        float v_r1 = (float)v[base_io + r1];

        long long row_off0 = (long long)r0 * head_dim;
        long long row_off1 = (long long)r1 * head_dim;
        long long src_row0 = base_h_src + row_off0;
        long long dst_row0 = base_h_dst + row_off0;
        long long src_row1 = base_h_src + row_off1;
        long long dst_row1 = base_h_dst + row_off1;

        float partial0 = 0.0f;
        float partial1 = 0.0f;

        #pragma unroll
        for (int j = 0; j < 8; j += 2) {
            if (j + 1 < cols_per_lane) {
                int c = lane_col_base + j;
                // Issue both loads back-to-back ; the compiler / LSU
                // schedules them as independent transactions.
                __nv_bfloat162 oldp0 = *reinterpret_cast<const __nv_bfloat162*>(
                    &tree_states[src_row0 + c]);
                __nv_bfloat162 oldp1 = *reinterpret_cast<const __nv_bfloat162*>(
                    &tree_states[src_row1 + c]);

                float o00 = __low2float(oldp0);
                float o01 = __high2float(oldp0);
                float o10 = __low2float(oldp1);
                float o11 = __high2float(oldp1);

                float u00 = g_exp * o00 + v_r0 * bk_cols[j    ];
                float u01 = g_exp * o01 + v_r0 * bk_cols[j + 1];
                float u10 = g_exp * o10 + v_r1 * bk_cols[j    ];
                float u11 = g_exp * o11 + v_r1 * bk_cols[j + 1];

                *reinterpret_cast<__nv_bfloat162*>(&tree_states[dst_row0 + c]) =
                    __floats2bfloat162_rn(u00, u01);
                *reinterpret_cast<__nv_bfloat162*>(&tree_states[dst_row1 + c]) =
                    __floats2bfloat162_rn(u10, u11);

                partial0 += u00 * q_cols[j    ];
                partial0 += u01 * q_cols[j + 1];
                partial1 += u10 * q_cols[j    ];
                partial1 += u11 * q_cols[j + 1];
            } else if (j < cols_per_lane) {
                int c = lane_col_base + j;
                float old0 = (float)tree_states[src_row0 + c];
                float old1 = (float)tree_states[src_row1 + c];
                float u0 = g_exp * old0 + v_r0 * bk_cols[j];
                float u1 = g_exp * old1 + v_r1 * bk_cols[j];
                tree_states[dst_row0 + c] = (__nv_bfloat16)u0;
                tree_states[dst_row1 + c] = (__nv_bfloat16)u1;
                partial0 += u0 * q_cols[j];
                partial1 += u1 * q_cols[j];
            }
        }

        // Two independent warp reductions ; compiler can interleave their
        // shuffle instructions.
        partial0 += __shfl_xor_sync(0xffffffffu, partial0, 16);
        partial1 += __shfl_xor_sync(0xffffffffu, partial1, 16);
        partial0 += __shfl_xor_sync(0xffffffffu, partial0,  8);
        partial1 += __shfl_xor_sync(0xffffffffu, partial1,  8);
        partial0 += __shfl_xor_sync(0xffffffffu, partial0,  4);
        partial1 += __shfl_xor_sync(0xffffffffu, partial1,  4);
        partial0 += __shfl_xor_sync(0xffffffffu, partial0,  2);
        partial1 += __shfl_xor_sync(0xffffffffu, partial1,  2);
        partial0 += __shfl_xor_sync(0xffffffffu, partial0,  1);
        partial1 += __shfl_xor_sync(0xffffffffu, partial1,  1);

        if (lane == 0) {
            out[base_io + r0] = (__nv_bfloat16)partial0;
            out[base_io + r1] = (__nv_bfloat16)partial1;
        }
    }

    // Tail : odd row(s) — handle 1 at a time.
    for (; r_off < rows_per_warp; ++r_off) {
        int r = warp_row_base + r_off;
        float v_r = (float)v[base_io + r];

        long long row_off = (long long)r * head_dim;
        long long src_row = base_h_src + row_off;
        long long dst_row = base_h_dst + row_off;

        float partial = 0.0f;

        #pragma unroll
        for (int j = 0; j < 8; j += 2) {
            if (j + 1 < cols_per_lane) {
                int c = lane_col_base + j;
                __nv_bfloat162 oldp = *reinterpret_cast<const __nv_bfloat162*>(
                    &tree_states[src_row + c]);
                float o0 = __low2float(oldp);
                float o1 = __high2float(oldp);
                float u0 = g_exp * o0 + v_r * bk_cols[j    ];
                float u1 = g_exp * o1 + v_r * bk_cols[j + 1];
                __nv_bfloat162 upd = __floats2bfloat162_rn(u0, u1);
                *reinterpret_cast<__nv_bfloat162*>(&tree_states[dst_row + c]) = upd;
                partial += u0 * q_cols[j    ];
                partial += u1 * q_cols[j + 1];
            } else if (j < cols_per_lane) {
                int c = lane_col_base + j;
                float old = (float)tree_states[src_row + c];
                float updated = g_exp * old + v_r * bk_cols[j];
                tree_states[dst_row + c] = (__nv_bfloat16)updated;
                partial += updated * q_cols[j];
            }
        }

        partial += __shfl_xor_sync(0xffffffffu, partial, 16);
        partial += __shfl_xor_sync(0xffffffffu, partial,  8);
        partial += __shfl_xor_sync(0xffffffffu, partial,  4);
        partial += __shfl_xor_sync(0xffffffffu, partial,  2);
        partial += __shfl_xor_sync(0xffffffffu, partial,  1);

        if (lane == 0) {
            out[base_io + r] = (__nv_bfloat16)partial;
        }
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

// T246.6.8 — sgemv_q6k_bf16 V4 — same speedup pattern as Q4_K V3, but
// keeping V2's 64-thread/row layout (less register pressure than V3)
// + 4-row multi-row blocks → 256 threads/block.
//
// V3 regressed on the LM head (9.32 vs V2's 9.50 tok/s on 27B) because
// 32-lane-per-row layout forced each lane to handle 16 weights/super-
// block — too much register pressure for Q6_K's 6-bit decode + 16
// distinct scales. V4 reuses V2's 64 threads/row (each handles 4
// weights, 4 scales) but stacks 4 rows in 1 block.
#[cfg(feature = "cuda")]
const SGEMV_Q6K_BF16_V4_SRC: &str = r#"
#include <cuda_bf16.h>
#include <cuda_fp16.h>

extern "C" __launch_bounds__(256, 4)
__global__ void sgemv_q6k_bf16_v4(
    const unsigned char* __restrict__ w_q6k,
    const __nv_bfloat16* __restrict__ x,
    __nv_bfloat16* __restrict__ y,
    int N,
    int K
) {
    int row0 = blockIdx.x * 4;
    int tid  = threadIdx.x;
    int row_in_block = tid >> 6;        // 0..3 (4 rows × 64 threads = 256)
    int row_tid      = tid & 63;          // 0..63 (V2's per-row tid)
    int row          = row0 + row_in_block;
    if (row >= N) return;

    int blocks_per_row = K / 256;
    int row_offset     = row * blocks_per_row * 210;

    // V2-like tile mapping (per row).
    int half          = row_tid >> 5;   // 0 or 1
    int l             = row_tid & 31;   // 0..31
    int half_offset_x = half << 7;      // 0 or 128
    int ql_base       = half << 6;      // 0 or 64
    int qh_base       = half << 5;      // 0 or 32
    int sb            = half << 3;      // 0 or 8
    int l16           = l >> 4;         // 0 or 1

    float acc = 0.0f;

    for (int b = 0; b < blocks_per_row; ++b) {
        int blk_off = row_offset + b * 210;
        const unsigned char* blk = w_q6k + blk_off;

        // Header : d (super-block float scale) — broadcast load via L1.
        unsigned short d_bits = blk[208] | (blk[209] << 8);
        float d = __half2float(__ushort_as_half(d_bits));

        // Per-thread scale unpack — 4 distinct scales used by this thread.
        const signed char* scales = (const signed char*)(blk + 192);
        float sc0 = d * (float)scales[sb + 0 + l16];
        float sc2 = d * (float)scales[sb + 2 + l16];
        float sc4 = d * (float)scales[sb + 4 + l16];
        float sc6 = d * (float)scales[sb + 6 + l16];

        const unsigned char* ql = blk;
        const unsigned char* qh = blk + 128;

        unsigned char ql_a = ql[ql_base + l];
        unsigned char ql_b = ql[ql_base + l + 32];
        unsigned char qh_b = qh[qh_base + l];

        int q0 = (ql_a & 0x0F) | (((qh_b)      & 0x03) << 4);
        int q1 = (ql_b & 0x0F) | (((qh_b >> 2) & 0x03) << 4);
        int q2 = (ql_a >> 4)   | (((qh_b >> 4) & 0x03) << 4);
        int q3 = (ql_b >> 4)   | (((qh_b >> 6) & 0x03) << 4);

        const __nv_bfloat16* x_ptr = x + b * 256 + half_offset_x;
        float x0 = (float)x_ptr[l];
        float x1 = (float)x_ptr[l + 32];
        float x2 = (float)x_ptr[l + 64];
        float x3 = (float)x_ptr[l + 96];

        acc += sc0 * (float)(q0 - 32) * x0;
        acc += sc2 * (float)(q1 - 32) * x1;
        acc += sc4 * (float)(q2 - 32) * x2;
        acc += sc6 * (float)(q3 - 32) * x3;
        // No __syncthreads — per-thread scale eliminates V2's race.
    }

    // Per-row reduction over 64 threads (2 warps).
    // Step 1: warp-shuffle within each warp.
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_xor_sync(0xffffffffu, acc, offset);
    }
    // Step 2: cross-warp via 4*2 = 8 shmem slots (4 rows × 2 warps).
    extern __shared__ float sdata[];  // [4 * 2] = 8 floats
    int warp_in_row = (row_tid >> 5) & 1;  // 0 or 1
    if ((row_tid & 31) == 0) {
        sdata[row_in_block * 2 + warp_in_row] = acc;
    }
    __syncthreads();
    if (row_tid == 0) {
        float total = sdata[row_in_block * 2 + 0] + sdata[row_in_block * 2 + 1];
        y[row] = (__nv_bfloat16)total;
    }
}
"#;

// T246.6.7 — sgemv_q6k_bf16 V3 — same multi-row + per-thread-scale pattern
// as Q4_K V3, adapted to Q6_K's 6-bit packed layout. Targets the LM head
// (N=152064) which is 27% of GPU time on 27B Q4_K_M (1 huge call/token).
#[cfg(feature = "cuda")]
const SGEMV_Q6K_BF16_V3_SRC: &str = r#"
#include <cuda_bf16.h>
#include <cuda_fp16.h>

extern "C" __launch_bounds__(128, 8)
__global__ void sgemv_q6k_bf16_v3(
    const unsigned char* __restrict__ w_q6k,
    const __nv_bfloat16* __restrict__ x,
    __nv_bfloat16* __restrict__ y,
    int N,
    int K
) {
    int row0 = blockIdx.x * 4;
    int tid  = threadIdx.x;
    int row_in_block = tid >> 5;        // 0..3
    int lane         = tid & 31;
    int row          = row0 + row_in_block;
    if (row >= N) return;

    int blocks_per_row = K / 256;
    int row_offset     = row * blocks_per_row * 210;
    int l16            = lane >> 4;     // 0 or 1 (used for sc selection)

    float acc = 0.0f;

    for (int b = 0; b < blocks_per_row; ++b) {
        int blk_off = row_offset + b * 210;
        const unsigned char* blk = w_q6k + blk_off;

        // Header : d (super-block float scale) — broadcast load via L1.
        unsigned short d_bits = blk[208] | (blk[209] << 8);
        float d = __half2float(__ushort_as_half(d_bits));

        // Per-thread scale unpack (16 signed-char scales at offset 192).
        // Each lane uses 8 distinct scales : sb + 0/2/4/6 + l16 for both halves.
        const signed char* scales = (const signed char*)(blk + 192);
        // For half=0 : sb=0, scales used = {0,2,4,6} + l16
        // For half=1 : sb=8, scales used = {8,10,12,14} + l16
        float sc0_a = d * (float)scales[0 + l16];
        float sc2_a = d * (float)scales[2 + l16];
        float sc4_a = d * (float)scales[4 + l16];
        float sc6_a = d * (float)scales[6 + l16];
        float sc0_b = d * (float)scales[8 + l16];
        float sc2_b = d * (float)scales[10 + l16];
        float sc4_b = d * (float)scales[12 + l16];
        float sc6_b = d * (float)scales[14 + l16];

        const unsigned char* ql = blk;
        const unsigned char* qh = blk + 128;

        // -------- Half 0 (weights 0..127) --------
        unsigned char ql_a0 = ql[0 + lane];
        unsigned char ql_b0 = ql[0 + lane + 32];
        unsigned char qh_0  = qh[0 + lane];
        int q0a = (ql_a0 & 0x0F) | (((qh_0)      & 0x03) << 4);
        int q1a = (ql_b0 & 0x0F) | (((qh_0 >> 2) & 0x03) << 4);
        int q2a = (ql_a0 >> 4)   | (((qh_0 >> 4) & 0x03) << 4);
        int q3a = (ql_b0 >> 4)   | (((qh_0 >> 6) & 0x03) << 4);

        const __nv_bfloat16* x_ptr_a = x + b * 256 + 0;
        float x0a = (float)x_ptr_a[lane];
        float x1a = (float)x_ptr_a[lane + 32];
        float x2a = (float)x_ptr_a[lane + 64];
        float x3a = (float)x_ptr_a[lane + 96];

        acc += sc0_a * (float)(q0a - 32) * x0a;
        acc += sc2_a * (float)(q1a - 32) * x1a;
        acc += sc4_a * (float)(q2a - 32) * x2a;
        acc += sc6_a * (float)(q3a - 32) * x3a;

        // -------- Half 1 (weights 128..255) --------
        unsigned char ql_a1 = ql[64 + lane];
        unsigned char ql_b1 = ql[64 + lane + 32];
        unsigned char qh_1  = qh[32 + lane];
        int q0b = (ql_a1 & 0x0F) | (((qh_1)      & 0x03) << 4);
        int q1b = (ql_b1 & 0x0F) | (((qh_1 >> 2) & 0x03) << 4);
        int q2b = (ql_a1 >> 4)   | (((qh_1 >> 4) & 0x03) << 4);
        int q3b = (ql_b1 >> 4)   | (((qh_1 >> 6) & 0x03) << 4);

        const __nv_bfloat16* x_ptr_b = x + b * 256 + 128;
        float x0b = (float)x_ptr_b[lane];
        float x1b = (float)x_ptr_b[lane + 32];
        float x2b = (float)x_ptr_b[lane + 64];
        float x3b = (float)x_ptr_b[lane + 96];

        acc += sc0_b * (float)(q0b - 32) * x0b;
        acc += sc2_b * (float)(q1b - 32) * x1b;
        acc += sc4_b * (float)(q2b - 32) * x2b;
        acc += sc6_b * (float)(q3b - 32) * x3b;
    }

    // Warp-shuffle reduction within row's 32 lanes.
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_xor_sync(0xffffffffu, acc, offset);
    }
    if (lane == 0) {
        y[row] = (__nv_bfloat16)acc;
    }
}
"#;

// T246.6.9 — sgemv_q5k_bf16 V3 — same multi-row + per-thread scale pattern
// as Q4_K V3, with qh-bit extraction for the 5th bit per weight.
// Each lane covers BOTH sub-blocks of a pair (sub_a = 2*pair via low
// nibble + sub_b = 2*pair+1 via high nibble), using their distinct qh
// bit positions (2*pair vs 2*pair+1).
#[cfg(feature = "cuda")]
const SGEMV_Q5K_BF16_V3_SRC: &str = r#"
#include <cuda_bf16.h>
#include <cuda_fp16.h>

extern "C" __launch_bounds__(128, 8)
__global__ void sgemv_q5k_bf16_v3(
    const unsigned char* __restrict__ w_q5k,
    const __nv_bfloat16* __restrict__ x,
    __nv_bfloat16* __restrict__ y,
    int N,
    int K
) {
    int row0 = blockIdx.x * 4;
    int tid  = threadIdx.x;
    int row_in_block = tid >> 5;
    int lane         = tid & 31;
    int row          = row0 + row_in_block;
    if (row >= N) return;

    int blocks_per_row = K / 256;
    int row_offset     = row * blocks_per_row * 176;

    int group     = lane >> 3;            // 0..3 — pair_idx
    int pos_base  = (lane & 7) << 2;      // 0..28 step 4
    int byte_base = (group << 5) + pos_base;
    unsigned int qh_mask_a = 1u << (2 * group);       // sub_a (low nibble half)
    unsigned int qh_mask_b = 1u << (2 * group + 1);   // sub_b (high nibble half)

    float acc = 0.0f;

    for (int b = 0; b < blocks_per_row; ++b) {
        int blk_off = row_offset + b * 176;
        const unsigned char* blk = w_q5k + blk_off;

        // Header.
        unsigned short d_bits    = blk[0] | (blk[1] << 8);
        unsigned short dmin_bits = blk[2] | (blk[3] << 8);
        float d    = __half2float(__ushort_as_half(d_bits));
        float dmin = __half2float(__ushort_as_half(dmin_bits));

        // Per-thread scale unpack for both sub-blocks of this pair.
        const unsigned char* scales = blk + 4;
        int sub_a = group * 2;
        int sub_b = sub_a + 1;
        unsigned char sc_a, m_a, sc_b, m_b;
        if (sub_a < 4) {
            sc_a = scales[sub_a]     & 0x3F;
            m_a  = scales[sub_a + 4] & 0x3F;
        } else {
            int ga = sub_a - 4;
            sc_a = (scales[ga + 8] & 0x0F) | ((scales[ga]     >> 6) << 4);
            m_a  = (scales[ga + 8] >> 4)   | ((scales[ga + 4] >> 6) << 4);
        }
        if (sub_b < 4) {
            sc_b = scales[sub_b]     & 0x3F;
            m_b  = scales[sub_b + 4] & 0x3F;
        } else {
            int gb = sub_b - 4;
            sc_b = (scales[gb + 8] & 0x0F) | ((scales[gb]     >> 6) << 4);
            m_b  = (scales[gb + 8] >> 4)   | ((scales[gb + 4] >> 6) << 4);
        }
        float scale_a = d    * (float)sc_a;
        float min_a   = dmin * (float)m_a;
        float scale_b = d    * (float)sc_b;
        float min_b   = dmin * (float)m_b;

        const unsigned char* qh = blk + 16;       // 32 bytes high bits
        const unsigned char* ql = blk + 16 + 32;  // 128 bytes low nibbles

        // 4 ql bytes (uint32) covering byte_base..byte_base+3.
        unsigned int qlbytes = *(const unsigned int*)(ql + byte_base);
        // 4 qh bytes (uint32) at the same pos_base — qh has 32 bytes per
        // super-block (1 byte per pos in 0..31), shared by all pair_idxs.
        unsigned int qhbytes = *(const unsigned int*)(qh + pos_base);

        // Load BF16 x for sub_a and sub_b.
        const __nv_bfloat16* xa_ptr = x + b * 256 + sub_a * 32 + pos_base;
        const __nv_bfloat16* xb_ptr = x + b * 256 + sub_b * 32 + pos_base;
        uint2 xa = *(const uint2*)xa_ptr;
        uint2 xb = *(const uint2*)xb_ptr;
        float xa0 = (float)__ushort_as_bfloat16((unsigned short)(xa.x & 0xFFFFu));
        float xa1 = (float)__ushort_as_bfloat16((unsigned short)(xa.x >> 16));
        float xa2 = (float)__ushort_as_bfloat16((unsigned short)(xa.y & 0xFFFFu));
        float xa3 = (float)__ushort_as_bfloat16((unsigned short)(xa.y >> 16));
        float xb0 = (float)__ushort_as_bfloat16((unsigned short)(xb.x & 0xFFFFu));
        float xb1 = (float)__ushort_as_bfloat16((unsigned short)(xb.x >> 16));
        float xb2 = (float)__ushort_as_bfloat16((unsigned short)(xb.y & 0xFFFFu));
        float xb3 = (float)__ushort_as_bfloat16((unsigned short)(xb.y >> 16));

        // Decode 4 weights for sub_a (LOW nibbles + qh_mask_a bit) and 4
        // for sub_b (HIGH nibbles + qh_mask_b bit).
        unsigned char qb0 = (qlbytes      ) & 0xFFu;
        unsigned char qb1 = (qlbytes >>  8) & 0xFFu;
        unsigned char qb2 = (qlbytes >> 16) & 0xFFu;
        unsigned char qb3 = (qlbytes >> 24) & 0xFFu;
        unsigned char hb0 = (qhbytes      ) & 0xFFu;
        unsigned char hb1 = (qhbytes >>  8) & 0xFFu;
        unsigned char hb2 = (qhbytes >> 16) & 0xFFu;
        unsigned char hb3 = (qhbytes >> 24) & 0xFFu;

        int qa0 = (qb0 & 0x0F) + ((hb0 & qh_mask_a) ? 16 : 0);
        int qa1 = (qb1 & 0x0F) + ((hb1 & qh_mask_a) ? 16 : 0);
        int qa2 = (qb2 & 0x0F) + ((hb2 & qh_mask_a) ? 16 : 0);
        int qa3 = (qb3 & 0x0F) + ((hb3 & qh_mask_a) ? 16 : 0);
        int qbq0 = (qb0 >>   4) + ((hb0 & qh_mask_b) ? 16 : 0);
        int qbq1 = (qb1 >>   4) + ((hb1 & qh_mask_b) ? 16 : 0);
        int qbq2 = (qb2 >>   4) + ((hb2 & qh_mask_b) ? 16 : 0);
        int qbq3 = (qb3 >>   4) + ((hb3 & qh_mask_b) ? 16 : 0);

        acc += (scale_a * (float)qa0  - min_a) * xa0;
        acc += (scale_a * (float)qa1  - min_a) * xa1;
        acc += (scale_a * (float)qa2  - min_a) * xa2;
        acc += (scale_a * (float)qa3  - min_a) * xa3;
        acc += (scale_b * (float)qbq0 - min_b) * xb0;
        acc += (scale_b * (float)qbq1 - min_b) * xb1;
        acc += (scale_b * (float)qbq2 - min_b) * xb2;
        acc += (scale_b * (float)qbq3 - min_b) * xb3;
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_xor_sync(0xffffffffu, acc, offset);
    }
    if (lane == 0) {
        y[row] = (__nv_bfloat16)acc;
    }
}
"#;

// ─────────────────────────────────────────────────────────────────────────
// T246.8 A5 — SPLIT-K Q6_K SGEMV for the lm_head ceiling.
//
// V2 launches N blocks (one per output row) × 64 threads/block. For
// lm_head N=152064, K=2048 on Qwen3.6-35B-A3B this gives 152064 blocks
// — plenty for SM count (~128 SMs on GB10) but each block reads its 4
// rows of Q6_K weights serially over K. The kernel is bandwidth-bound
// at ~42% HBM peak (A3 measurement, note 8e23a850).
//
// Split-K strategy : split K into K_CHUNKS super-block-aligned chunks.
// For K=2048 → 8 super-blocks/row, K_CHUNKS=8 → blocks_per_chunk=1.
// Per-block work is the same V2 body but restricted to one chunk's
// super-blocks. Grid = (N, K_CHUNKS) blocks → 1.2M blocks for lm_head.
// Per-(row, chunk) pair we write a FP32 partial sum to a global
// staging buffer of layout [K_CHUNKS, N]. A reduction kernel
// (`reduce_split_k_bf16`) then sums dim 0 → BF16 [N] output.
//
// FP non-associativity caveat : V2 = warp_reduce(sum_{b} per-thread)
// while split-K = sum_{c} warp_reduce(sum_{b in c} per-thread). For 1
// super-block per chunk both reduce to the same identity (chunks of 1
// have no inner accumulation) so the only difference is the final
// reducer's chunk-summation order. Target : within 1 BF16 ULP of V2.
//
// Kernel body : VERBATIM copy of V2's per-thread Q6_K decode + warp
// reduce, with K range parameterized by (b_start, b_end).
#[cfg(feature = "cuda")]
const SGEMV_Q6K_BF16_SPLIT_K_PARTIAL_SRC: &str = r#"
#include <cuda_bf16.h>
#include <cuda_fp16.h>

extern "C" __global__ void sgemv_q6k_bf16_split_k_partial(
    const unsigned char* __restrict__ w_q6k,
    const __nv_bfloat16* __restrict__ x,
    float*               __restrict__ partial,   // [K_CHUNKS, N] FP32
    int N,
    int K,
    int K_CHUNKS                                   // = blocks_per_row / blocks_per_chunk
) {
    int row       = blockIdx.x;
    int chunk_idx = blockIdx.y;
    if (row >= N) return;
    int tid = threadIdx.x;          // 0..63

    int blocks_per_row   = K / 256;
    int blocks_per_chunk = blocks_per_row / K_CHUNKS;
    int b_start          = chunk_idx * blocks_per_chunk;
    int b_end            = b_start + blocks_per_chunk;
    int row_offset       = row * blocks_per_row * 210;

    extern __shared__ float shmem[];
    float* sc_pre = shmem;          // [16]
    float* sdata  = shmem + 16;     // [2]

    float acc = 0.0f;

    int half          = tid >> 5;   // 0 or 1
    int l             = tid & 31;   // 0..31
    int half_offset_x = half << 7;  // 0 or 128
    int ql_base       = half << 6;  // 0 or 64
    int qh_base       = half << 5;  // 0 or 32
    int sb            = half << 3;  // 0 or 8
    int l16           = l >> 4;     // 0 or 1

    for (int b = b_start; b < b_end; ++b) {
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

        __syncthreads();  // RACE FIX (matches V2)
    }

    // Warp-shuffle within each warp (32 lanes).
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
        partial[chunk_idx * N + row] = total;
    }
}
"#;

// Reduction kernel : sums [K_CHUNKS, N] FP32 partials → [N] BF16 output.
// One thread per output row ; loops over K_CHUNKS (max ~16) and writes
// the BF16 truncation of the FP32 sum. K_CHUNKS sequential adds done in
// chunk-major order (0, 1, ..., K_CHUNKS-1).
#[cfg(feature = "cuda")]
const REDUCE_SPLIT_K_BF16_SRC: &str = r#"
#include <cuda_bf16.h>

extern "C" __global__ void reduce_split_k_bf16(
    const float*       __restrict__ partial,   // [K_CHUNKS, N] FP32
    __nv_bfloat16*     __restrict__ y,         // [N] BF16
    int N,
    int K_CHUNKS
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N) return;

    float sum = 0.0f;
    // Sequential chunk-major sum to match the natural V2 reduction order.
    for (int c = 0; c < K_CHUNKS; ++c) {
        sum += partial[c * N + row];
    }
    y[row] = (__nv_bfloat16)sum;
}
"#;

// ─────────────────────────────────────────────────────────────────────────
// T246.8 A2.1 — INDEXED SGEMV variants for MoE FFN (Qwen3.6-35B-A3B).
//
// Each kernel below mirrors the body of its non-indexed v3 counterpart but
// adds a tiny prologue : it reads `topk_indices[slot]` from device memory
// and uses it to fetch the per-expert weight base pointer from a
// device-resident `expert_ptrs[]` array. This eliminates the host-side
// `memcpy_dtov(topk_idx_slice)` that previously preceded the per-expert
// matmul dispatch loop, removing ~150 host syncs per MoE token and
// allowing the entire decode body to be captured by a CUDA Graph.
//
// Layout of `expert_ptrs` (host-build, device-resident `CudaSlice<u64>`):
//   expert_ptrs[e] = device pointer (as u64) to expert e's weight base.
//   This is built once at load_ffn time, never modified per-step.
//
// Note : Q4_K dp4a path needs both the indexed-w prologue AND the host-
// supplied x_q8_1 (no per-expert variation in x_q8_1 since x is shared).
// ─────────────────────────────────────────────────────────────────────────

// Indexed Q4_K v3 — same body as SGEMV_Q4K_BF16_V3_SRC, but the weight
// pointer is fetched from `expert_ptrs[topk_indices[slot]]` on device.
#[cfg(feature = "cuda")]
const SGEMV_Q4K_BF16_V3_INDEXED_SRC: &str = r#"
#include <cuda_bf16.h>
#include <cuda_fp16.h>

extern "C" __launch_bounds__(128, 8)
__global__ void sgemv_q4k_bf16_v3_indexed(
    const unsigned long long* __restrict__ expert_ptrs,  // [n_experts] u64
    const int*                __restrict__ topk_indices, // [k] device i32
    int slot,                                            // 0..k-1 (host const)
    const __nv_bfloat16*      __restrict__ x,
    __nv_bfloat16*            __restrict__ y,
    int N,
    int K
) {
    int e_idx = topk_indices[slot];
    const unsigned char* __restrict__ w_q4k =
        (const unsigned char* __restrict__)expert_ptrs[e_idx];

    int row0 = blockIdx.x * 4;
    int tid  = threadIdx.x;
    int row_in_block = tid >> 5;
    int lane         = tid & 31;
    int row          = row0 + row_in_block;
    if (row >= N) return;

    int blocks_per_row = K / 256;
    int row_offset     = row * blocks_per_row * 144;

    int group     = lane >> 3;
    int pos_base  = (lane & 7) << 2;
    int byte_base = (group << 5) + pos_base;

    float acc = 0.0f;

    for (int b = 0; b < blocks_per_row; ++b) {
        int blk_off = row_offset + b * 144;
        const unsigned char* blk = w_q4k + blk_off;

        unsigned short d_bits    = blk[0] | (blk[1] << 8);
        unsigned short dmin_bits = blk[2] | (blk[3] << 8);
        float d    = __half2float(__ushort_as_half(d_bits));
        float dmin = __half2float(__ushort_as_half(dmin_bits));

        const unsigned char* scales = blk + 4;
        int sub_a = group * 2;
        int sub_b = sub_a + 1;
        unsigned char sc_a, m_a, sc_b, m_b;
        if (sub_a < 4) {
            sc_a = scales[sub_a]     & 0x3F;
            m_a  = scales[sub_a + 4] & 0x3F;
        } else {
            int ga = sub_a - 4;
            sc_a = (scales[ga + 8] & 0x0F) | ((scales[ga]     >> 6) << 4);
            m_a  = (scales[ga + 8] >> 4)   | ((scales[ga + 4] >> 6) << 4);
        }
        if (sub_b < 4) {
            sc_b = scales[sub_b]     & 0x3F;
            m_b  = scales[sub_b + 4] & 0x3F;
        } else {
            int gb = sub_b - 4;
            sc_b = (scales[gb + 8] & 0x0F) | ((scales[gb]     >> 6) << 4);
            m_b  = (scales[gb + 8] >> 4)   | ((scales[gb + 4] >> 6) << 4);
        }
        float scale_a = d    * (float)sc_a;
        float min_a   = dmin * (float)m_a;
        float scale_b = d    * (float)sc_b;
        float min_b   = dmin * (float)m_b;

        const unsigned char* qs = blk + 16;
        unsigned int qbytes = *(const unsigned int*)(qs + byte_base);

        const __nv_bfloat16* xa_ptr = x + b * 256 + sub_a * 32 + pos_base;
        const __nv_bfloat16* xb_ptr = x + b * 256 + sub_b * 32 + pos_base;
        uint2 xa = *(const uint2*)xa_ptr;
        uint2 xb = *(const uint2*)xb_ptr;
        float xa0 = (float)__ushort_as_bfloat16((unsigned short)(xa.x & 0xFFFFu));
        float xa1 = (float)__ushort_as_bfloat16((unsigned short)(xa.x >> 16));
        float xa2 = (float)__ushort_as_bfloat16((unsigned short)(xa.y & 0xFFFFu));
        float xa3 = (float)__ushort_as_bfloat16((unsigned short)(xa.y >> 16));
        float xb0 = (float)__ushort_as_bfloat16((unsigned short)(xb.x & 0xFFFFu));
        float xb1 = (float)__ushort_as_bfloat16((unsigned short)(xb.x >> 16));
        float xb2 = (float)__ushort_as_bfloat16((unsigned short)(xb.y & 0xFFFFu));
        float xb3 = (float)__ushort_as_bfloat16((unsigned short)(xb.y >> 16));

        unsigned char by0 = (qbytes      ) & 0xFFu;
        unsigned char by1 = (qbytes >>  8) & 0xFFu;
        unsigned char by2 = (qbytes >> 16) & 0xFFu;
        unsigned char by3 = (qbytes >> 24) & 0xFFu;
        int na0 = by0 & 0x0F, na1 = by1 & 0x0F, na2 = by2 & 0x0F, na3 = by3 & 0x0F;
        int nb0 = by0 >>   4, nb1 = by1 >>   4, nb2 = by2 >>   4, nb3 = by3 >>   4;

        acc += (scale_a * (float)na0 - min_a) * xa0;
        acc += (scale_a * (float)na1 - min_a) * xa1;
        acc += (scale_a * (float)na2 - min_a) * xa2;
        acc += (scale_a * (float)na3 - min_a) * xa3;
        acc += (scale_b * (float)nb0 - min_b) * xb0;
        acc += (scale_b * (float)nb1 - min_b) * xb1;
        acc += (scale_b * (float)nb2 - min_b) * xb2;
        acc += (scale_b * (float)nb3 - min_b) * xb3;
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_xor_sync(0xffffffffu, acc, offset);
    }
    if (lane == 0) {
        y[row] = (__nv_bfloat16)acc;
    }
}
"#;

// Indexed Q4_K dp4a — same body as SGEMV_Q4K_Q8_1_DP4A_BF16_SRC, but the
// weight pointer is fetched from `expert_ptrs[topk_indices[slot]]`. The
// activation x_q8_1 is shared across all experts (no per-expert variation).
#[cfg(feature = "cuda")]
const SGEMV_Q4K_Q8_1_DP4A_BF16_INDEXED_SRC: &str = r#"
#include <cuda_bf16.h>
#include <cuda_fp16.h>

extern "C" __global__ void sgemv_q4k_q8_1_dp4a_bf16_indexed(
    const unsigned long long* __restrict__ expert_ptrs,  // [n_experts] u64
    const int*                __restrict__ topk_indices, // [k] device i32
    int slot,                                            // host const
    const unsigned char*      __restrict__ x_q8_1,
    __nv_bfloat16*            __restrict__ y,
    int N,
    int K
) {
    int e_idx = topk_indices[slot];
    const unsigned char* __restrict__ w_q4k =
        (const unsigned char* __restrict__)expert_ptrs[e_idx];

    int row = blockIdx.x;
    if (row >= N) return;
    int tid = threadIdx.x;

    int blocks_per_row = K / 256;
    float acc = 0.0f;

    for (int b = tid; b < blocks_per_row; b += 32) {
        const unsigned char* blk = w_q4k + (row * blocks_per_row + b) * 144;

        float d    = __half2float(*(const __half*)(blk + 0));
        float dmin = __half2float(*(const __half*)(blk + 2));

        const unsigned char* sr = blk + 4;
        unsigned char sc[8], m[8];
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            sc[i]     = sr[i]     & 0x3F;
            m[i]      = sr[i + 4] & 0x3F;
            sc[i + 4] = (sr[i + 8] & 0x0F) | ((sr[i]     >> 6) << 4);
            m[i  + 4] = (sr[i + 8] >>   4) | ((sr[i + 4] >> 6) << 4);
        }

        const unsigned char* qs = blk + 16;

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

    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        acc += __shfl_xor_sync(0xffffffff, acc, o);
    }

    if (tid == 0) {
        y[row] = (__nv_bfloat16)acc;
    }
}
"#;

// Indexed Q5_K v3 — same body as SGEMV_Q5K_BF16_V3_SRC, with indexed-w prologue.
#[cfg(feature = "cuda")]
const SGEMV_Q5K_BF16_V3_INDEXED_SRC: &str = r#"
#include <cuda_bf16.h>
#include <cuda_fp16.h>

extern "C" __launch_bounds__(128, 8)
__global__ void sgemv_q5k_bf16_v3_indexed(
    const unsigned long long* __restrict__ expert_ptrs,
    const int*                __restrict__ topk_indices,
    int slot,
    const __nv_bfloat16*      __restrict__ x,
    __nv_bfloat16*            __restrict__ y,
    int N,
    int K
) {
    int e_idx = topk_indices[slot];
    const unsigned char* __restrict__ w_q5k =
        (const unsigned char* __restrict__)expert_ptrs[e_idx];

    int row0 = blockIdx.x * 4;
    int tid  = threadIdx.x;
    int row_in_block = tid >> 5;
    int lane         = tid & 31;
    int row          = row0 + row_in_block;
    if (row >= N) return;

    int blocks_per_row = K / 256;
    int row_offset     = row * blocks_per_row * 176;

    int group     = lane >> 3;
    int pos_base  = (lane & 7) << 2;
    int byte_base = (group << 5) + pos_base;
    unsigned int qh_mask_a = 1u << (2 * group);
    unsigned int qh_mask_b = 1u << (2 * group + 1);

    float acc = 0.0f;

    for (int b = 0; b < blocks_per_row; ++b) {
        int blk_off = row_offset + b * 176;
        const unsigned char* blk = w_q5k + blk_off;

        unsigned short d_bits    = blk[0] | (blk[1] << 8);
        unsigned short dmin_bits = blk[2] | (blk[3] << 8);
        float d    = __half2float(__ushort_as_half(d_bits));
        float dmin = __half2float(__ushort_as_half(dmin_bits));

        const unsigned char* scales = blk + 4;
        int sub_a = group * 2;
        int sub_b = sub_a + 1;
        unsigned char sc_a, m_a, sc_b, m_b;
        if (sub_a < 4) {
            sc_a = scales[sub_a]     & 0x3F;
            m_a  = scales[sub_a + 4] & 0x3F;
        } else {
            int ga = sub_a - 4;
            sc_a = (scales[ga + 8] & 0x0F) | ((scales[ga]     >> 6) << 4);
            m_a  = (scales[ga + 8] >> 4)   | ((scales[ga + 4] >> 6) << 4);
        }
        if (sub_b < 4) {
            sc_b = scales[sub_b]     & 0x3F;
            m_b  = scales[sub_b + 4] & 0x3F;
        } else {
            int gb = sub_b - 4;
            sc_b = (scales[gb + 8] & 0x0F) | ((scales[gb]     >> 6) << 4);
            m_b  = (scales[gb + 8] >> 4)   | ((scales[gb + 4] >> 6) << 4);
        }
        float scale_a = d    * (float)sc_a;
        float min_a   = dmin * (float)m_a;
        float scale_b = d    * (float)sc_b;
        float min_b   = dmin * (float)m_b;

        const unsigned char* qh = blk + 16;
        const unsigned char* ql = blk + 16 + 32;

        unsigned int qlbytes = *(const unsigned int*)(ql + byte_base);
        unsigned int qhbytes = *(const unsigned int*)(qh + pos_base);

        const __nv_bfloat16* xa_ptr = x + b * 256 + sub_a * 32 + pos_base;
        const __nv_bfloat16* xb_ptr = x + b * 256 + sub_b * 32 + pos_base;
        uint2 xa = *(const uint2*)xa_ptr;
        uint2 xb = *(const uint2*)xb_ptr;
        float xa0 = (float)__ushort_as_bfloat16((unsigned short)(xa.x & 0xFFFFu));
        float xa1 = (float)__ushort_as_bfloat16((unsigned short)(xa.x >> 16));
        float xa2 = (float)__ushort_as_bfloat16((unsigned short)(xa.y & 0xFFFFu));
        float xa3 = (float)__ushort_as_bfloat16((unsigned short)(xa.y >> 16));
        float xb0 = (float)__ushort_as_bfloat16((unsigned short)(xb.x & 0xFFFFu));
        float xb1 = (float)__ushort_as_bfloat16((unsigned short)(xb.x >> 16));
        float xb2 = (float)__ushort_as_bfloat16((unsigned short)(xb.y & 0xFFFFu));
        float xb3 = (float)__ushort_as_bfloat16((unsigned short)(xb.y >> 16));

        unsigned char qb0 = (qlbytes      ) & 0xFFu;
        unsigned char qb1 = (qlbytes >>  8) & 0xFFu;
        unsigned char qb2 = (qlbytes >> 16) & 0xFFu;
        unsigned char qb3 = (qlbytes >> 24) & 0xFFu;
        unsigned char hb0 = (qhbytes      ) & 0xFFu;
        unsigned char hb1 = (qhbytes >>  8) & 0xFFu;
        unsigned char hb2 = (qhbytes >> 16) & 0xFFu;
        unsigned char hb3 = (qhbytes >> 24) & 0xFFu;

        int qa0 = (qb0 & 0x0F) + ((hb0 & qh_mask_a) ? 16 : 0);
        int qa1 = (qb1 & 0x0F) + ((hb1 & qh_mask_a) ? 16 : 0);
        int qa2 = (qb2 & 0x0F) + ((hb2 & qh_mask_a) ? 16 : 0);
        int qa3 = (qb3 & 0x0F) + ((hb3 & qh_mask_a) ? 16 : 0);
        int qbq0 = (qb0 >>   4) + ((hb0 & qh_mask_b) ? 16 : 0);
        int qbq1 = (qb1 >>   4) + ((hb1 & qh_mask_b) ? 16 : 0);
        int qbq2 = (qb2 >>   4) + ((hb2 & qh_mask_b) ? 16 : 0);
        int qbq3 = (qb3 >>   4) + ((hb3 & qh_mask_b) ? 16 : 0);

        acc += (scale_a * (float)qa0  - min_a) * xa0;
        acc += (scale_a * (float)qa1  - min_a) * xa1;
        acc += (scale_a * (float)qa2  - min_a) * xa2;
        acc += (scale_a * (float)qa3  - min_a) * xa3;
        acc += (scale_b * (float)qbq0 - min_b) * xb0;
        acc += (scale_b * (float)qbq1 - min_b) * xb1;
        acc += (scale_b * (float)qbq2 - min_b) * xb2;
        acc += (scale_b * (float)qbq3 - min_b) * xb3;
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_xor_sync(0xffffffffu, acc, offset);
    }
    if (lane == 0) {
        y[row] = (__nv_bfloat16)acc;
    }
}
"#;

// Indexed Q6_K v3 — same body as SGEMV_Q6K_BF16_V3_SRC, with indexed-w prologue.
#[cfg(feature = "cuda")]
const SGEMV_Q6K_BF16_V3_INDEXED_SRC: &str = r#"
#include <cuda_bf16.h>
#include <cuda_fp16.h>

extern "C" __launch_bounds__(128, 8)
__global__ void sgemv_q6k_bf16_v3_indexed(
    const unsigned long long* __restrict__ expert_ptrs,
    const int*                __restrict__ topk_indices,
    int slot,
    const __nv_bfloat16*      __restrict__ x,
    __nv_bfloat16*            __restrict__ y,
    int N,
    int K
) {
    int e_idx = topk_indices[slot];
    const unsigned char* __restrict__ w_q6k =
        (const unsigned char* __restrict__)expert_ptrs[e_idx];

    int row0 = blockIdx.x * 4;
    int tid  = threadIdx.x;
    int row_in_block = tid >> 5;
    int lane         = tid & 31;
    int row          = row0 + row_in_block;
    if (row >= N) return;

    int blocks_per_row = K / 256;
    int row_offset     = row * blocks_per_row * 210;
    int l16            = lane >> 4;

    float acc = 0.0f;

    for (int b = 0; b < blocks_per_row; ++b) {
        int blk_off = row_offset + b * 210;
        const unsigned char* blk = w_q6k + blk_off;

        unsigned short d_bits = blk[208] | (blk[209] << 8);
        float d = __half2float(__ushort_as_half(d_bits));

        const signed char* scales = (const signed char*)(blk + 192);
        float sc0_a = d * (float)scales[0 + l16];
        float sc2_a = d * (float)scales[2 + l16];
        float sc4_a = d * (float)scales[4 + l16];
        float sc6_a = d * (float)scales[6 + l16];
        float sc0_b = d * (float)scales[8 + l16];
        float sc2_b = d * (float)scales[10 + l16];
        float sc4_b = d * (float)scales[12 + l16];
        float sc6_b = d * (float)scales[14 + l16];

        const unsigned char* ql = blk;
        const unsigned char* qh = blk + 128;

        unsigned char ql_a0 = ql[0 + lane];
        unsigned char ql_b0 = ql[0 + lane + 32];
        unsigned char qh_0  = qh[0 + lane];
        int q0a = (ql_a0 & 0x0F) | (((qh_0)      & 0x03) << 4);
        int q1a = (ql_b0 & 0x0F) | (((qh_0 >> 2) & 0x03) << 4);
        int q2a = (ql_a0 >> 4)   | (((qh_0 >> 4) & 0x03) << 4);
        int q3a = (ql_b0 >> 4)   | (((qh_0 >> 6) & 0x03) << 4);

        const __nv_bfloat16* x_ptr_a = x + b * 256 + 0;
        float x0a = (float)x_ptr_a[lane];
        float x1a = (float)x_ptr_a[lane + 32];
        float x2a = (float)x_ptr_a[lane + 64];
        float x3a = (float)x_ptr_a[lane + 96];

        acc += sc0_a * (float)(q0a - 32) * x0a;
        acc += sc2_a * (float)(q1a - 32) * x1a;
        acc += sc4_a * (float)(q2a - 32) * x2a;
        acc += sc6_a * (float)(q3a - 32) * x3a;

        unsigned char ql_a1 = ql[64 + lane];
        unsigned char ql_b1 = ql[64 + lane + 32];
        unsigned char qh_1  = qh[32 + lane];
        int q0b = (ql_a1 & 0x0F) | (((qh_1)      & 0x03) << 4);
        int q1b = (ql_b1 & 0x0F) | (((qh_1 >> 2) & 0x03) << 4);
        int q2b = (ql_a1 >> 4)   | (((qh_1 >> 4) & 0x03) << 4);
        int q3b = (ql_b1 >> 4)   | (((qh_1 >> 6) & 0x03) << 4);

        const __nv_bfloat16* x_ptr_b = x + b * 256 + 128;
        float x0b = (float)x_ptr_b[lane];
        float x1b = (float)x_ptr_b[lane + 32];
        float x2b = (float)x_ptr_b[lane + 64];
        float x3b = (float)x_ptr_b[lane + 96];

        acc += sc0_b * (float)(q0b - 32) * x0b;
        acc += sc2_b * (float)(q1b - 32) * x1b;
        acc += sc4_b * (float)(q2b - 32) * x2b;
        acc += sc6_b * (float)(q3b - 32) * x3b;
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_xor_sync(0xffffffffu, acc, offset);
    }
    if (lane == 0) {
        y[row] = (__nv_bfloat16)acc;
    }
}
"#;

// Indexed BF16 sgemv — same body as SGEMV_BF16_BF16_SRC, with indexed-w prologue.
#[cfg(feature = "cuda")]
const SGEMV_BF16_BF16_INDEXED_SRC: &str = r#"
#include <cuda_bf16.h>

extern "C" __global__ void sgemv_bf16_bf16_indexed(
    const unsigned long long* __restrict__ expert_ptrs,
    const int*                __restrict__ topk_indices,
    int slot,
    const __nv_bfloat16*      __restrict__ x,
    __nv_bfloat16*            __restrict__ y,
    int N,
    int K
) {
    int e_idx = topk_indices[slot];
    const __nv_bfloat16* __restrict__ w =
        (const __nv_bfloat16* __restrict__)expert_ptrs[e_idx];

    int row = blockIdx.x;
    if (row >= N) return;
    int tid = threadIdx.x;

    extern __shared__ float shmem[];

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

// T246.8 A2.1 — `y[i] += alpha_dev[slot] * x[i]` where alpha_dev is a
// device-resident bf16 vector and `slot` is a host-side index. Used to
// accumulate routed-expert outputs scaled by topk_w[slot] without ever
// reading the topk weights back to host. For the shared-expert sigmoid
// path : caller passes `slot=0` and a 1-element alpha_dev that holds the
// post-sigmoid weight (computed on device by sigmoid_inplace_bf16).
#[cfg(feature = "cuda")]
const SCALED_ADD_INPLACE_BF16_DEVSCALAR_SRC: &str = r#"
#include <cuda_bf16.h>

extern "C" __global__ void scaled_add_inplace_bf16_devscalar(
    __nv_bfloat16*       __restrict__ y,
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ alpha_dev,
    int slot,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float alpha = (float)alpha_dev[slot];
    float yi = (float)y[i];
    float xi = (float)x[i];
    y[i] = (__nv_bfloat16)(yi + alpha * xi);
}
"#;

// T246.8 A2.1 — Fused shared-expert sigmoid + scaled_add for MoE FFN.
//
// `y[i] += sigmoid((float)dot_bf16[0]) * x[i]`
//
// Replaces the host-sync sequence of (DtoH dot → host sigmoid in f32 →
// scaled_add with f32 alpha) in the routed-MoE shared-expert path.
// CRUCIAL : the sigmoid value is computed and held in float precision
// per thread, NOT stored back to bf16, so this matches the sync path's
// `1.0_f32 / (1.0 + (-v).exp())` precision exactly. Using
// `sigmoid_inplace_bf16 + scaled_add_inplace_bf16_devscalar` instead
// would round the sigmoid to bf16 between ops and cause ~7-bit drift
// per layer, which compounds across the 64 MoE layers and produces
// different greedy tokens from token 1.
#[cfg(feature = "cuda")]
const SCALED_ADD_SIGMOID_DEVSCALAR_BF16_SRC: &str = r#"
#include <cuda_bf16.h>

extern "C" __global__ void scaled_add_sigmoid_devscalar_bf16(
    __nv_bfloat16*       __restrict__ y,
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ dot_bf16,  // 1 element
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float dot = (float)dot_bf16[0];
    float alpha = 1.0f / (1.0f + expf(-dot));
    float yi = (float)y[i];
    float xi = (float)x[i];
    y[i] = (__nv_bfloat16)(yi + alpha * xi);
}
"#;

// T246.8 A2.1 — zero a BF16 vector (n elements). Replaces the
// `scaled_add_inplace_bf16(y, y, -1, n)` trick used to zero h_p in MoE
// FFN start ; the trick has a benign data race that's fine but we want
// a clean primitive when capturing CUDA Graphs (the alias-self pattern
// is OK for graphs but reads cleaner).
#[cfg(feature = "cuda")]
const ZERO_BF16_SRC: &str = r#"
#include <cuda_bf16.h>

extern "C" __global__ void zero_bf16(
    __nv_bfloat16* __restrict__ y,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    y[i] = (__nv_bfloat16)0.0f;
}
"#;

// ────────────────────────────────────────────────────────────────────────
// T246.9 NVFP4.2 — NVFP4 SGEMV (vLLM `nvfp4-pack-quantized` layout)
//
// Per-row dot product against a NVFP4-quantized weight matrix.
//
// Weight tensor 4-tuple per Linear (vLLM convention) :
//   weight_packed       [N, K/2]  U8       — 2 FP4 (E2M1) per byte, low nibble = even index
//   weight_scale        [N, K/16] F8_E4M3  — UE4M3 per-16-element micro-block scale
//   weight_global_scale [1]       F32      — per-tensor calibration scale
//   input_global_scale  [1]       F32      — per-tensor activation scale
//
// At call time the kernel takes `alpha = 1 / (weight_global * input_global)`
// and applies it once at the end of the K reduction (folded into the
// final BF16 down-cast).
//
// FP4 E2M1 decode table (4 bits) :
//   0=+0,    1=+0.5,  2=+1,   3=+1.5,  4=+2,   5=+3,   6=+4,   7=+6
//   8=-0,    9=-0.5,  a=-1,   b=-1.5,  c=-2,   d=-3,   e=-4,   f=-6
// Magnitudes : {0, 0.5, 1, 1.5, 2, 3, 4, 6} — max = 6.0.
//
// UE4M3 scale decode (8-bit unsigned, exp 4 bits bias 7, mantissa 3 bits) :
//   value = 2^(E - 7) * (1 + M/8)   for E >= 1
//   value = 0                        for E == 0   (subnormal — RFC R3 says
//                                                  the calibrated checkpoint
//                                                  shouldn't have these)
// Layout : bit 7 reserved, bits 6-3 = E, bits 2-0 = M.
//
// Block layout : `(N, 1, 1)` × 32 threads (1 warp / row). Each lane handles
// `K/16 / 32 = K/512` micro-blocks. Final acc is reduced via warp-shuffle.
// For Qwen3.6-A3B (K=2048 → 128 micro-blocks) each lane processes 4 blocks
// (i.e. 64 FP4 weights = 32 packed bytes per lane per row). Sufficient ILP
// for the M=1 decode case ; for prompt-processing batches we'd switch to
// the cuBLASLt `matmul_mxfp4` path.
// Reserved : a shared device-header that future NVFP4 kernels (mul_mm_id
// FP4 variants, prompt-batch GEMMs) can include. Kept inlined into the
// per-kernel SRC strings for now to avoid plumbing nvrtc include paths.
#[allow(dead_code)]
#[cfg(feature = "cuda")]
const NVFP4_DECODE_DEVICE_HEADER: &str = r#"
#include <cuda_bf16.h>
#include <cuda_fp16.h>

// E2M1 → fp32 LUT, 16 entries indexed by the raw 4-bit code.
__device__ __forceinline__ float fp4_to_fp32(unsigned int code) {
    // Table sized to 16 because the full 4-bit code (incl. sign) is the index.
    // mag[c & 7] = magnitude; sign = (c & 8) ? -1 : +1.
    static const float MAG[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    float m = MAG[code & 0x7];
    return (code & 0x8) ? -m : m;
}

// UE4M3 byte → fp32 (NVFP4 per-block scale convention).
__device__ __forceinline__ float ue4m3_to_fp32(unsigned char b) {
    unsigned int e = (b >> 3) & 0xF;
    unsigned int m = b & 0x7;
    if (e == 0) {
        // Subnormal — calibrated checkpoint shouldn't have these.
        // Fold mantissa as 2^(-7) * M/8 to match OCP-MX UE4M3 strict spec.
        return (float)m * 0.0009765625f;  // 2^-7 / 8 = 1/1024 ≈ 0.000977
    }
    // value = 2^(E-7) * (1 + M/8)
    int exp_unbiased = (int)e - 7;
    float scale = ldexpf(1.0f + (float)m * 0.125f, exp_unbiased);
    return scale;
}

// Decode 1 micro-block of 16 FP4 values into 16 fp32 elements.
// `packed` is 8 bytes (16 nibbles), `scale_byte` is 1 UE4M3.
// `out[16]` is the dequantized values (already × scale).
__device__ __forceinline__ void nvfp4_decode_block16(
    const unsigned char* __restrict__ packed_8b,
    unsigned char scale_byte,
    float* __restrict__ out16
) {
    float s = ue4m3_to_fp32(scale_byte);
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        unsigned char b = packed_8b[i];
        unsigned int lo = b & 0xF;
        unsigned int hi = (b >> 4) & 0xF;
        out16[2*i + 0] = fp4_to_fp32(lo) * s;
        out16[2*i + 1] = fp4_to_fp32(hi) * s;
    }
}
"#;

#[cfg(feature = "cuda")]
const SGEMV_NVFP4_BF16_INDEXED_SRC: &str = r#"
#include <cuda_bf16.h>
#include <cuda_fp16.h>

__device__ __forceinline__ float fp4_to_fp32_idx(unsigned int code) {
    static const float MAG[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    float m = MAG[code & 0x7];
    return (code & 0x8) ? -m : m;
}

__device__ __forceinline__ float ue4m3_to_fp32_idx(unsigned char b) {
    unsigned int e = (b >> 3) & 0xF;
    unsigned int m = b & 0x7;
    if (e == 0) {
        return (float)m * 0.0009765625f;
    }
    int exp_unbiased = (int)e - 7;
    return ldexpf(1.0f + (float)m * 0.125f, exp_unbiased);
}

extern "C" __global__ void sgemv_nvfp4_bf16_indexed(
    const unsigned long long* __restrict__ expert_packed_ptrs,  // [n_experts] u64
    const unsigned long long* __restrict__ expert_scale_ptrs,   // [n_experts] u64
    const float*              __restrict__ expert_alphas,       // [n_experts] f32 = 1/(w_g*in_g)
    const int*                __restrict__ topk_indices,        // [k_used] device i32
    int slot,                                                   // host-side const
    const __nv_bfloat16*      __restrict__ x,                   // [K] activation
    __nv_bfloat16*            __restrict__ y,                   // [N] output
    int N,
    int K
) {
    int e_idx = topk_indices[slot];
    const unsigned char* __restrict__ packed = (const unsigned char*)expert_packed_ptrs[e_idx];
    const unsigned char* __restrict__ scale  = (const unsigned char*)expert_scale_ptrs[e_idx];
    float alpha = expert_alphas[e_idx];

    int row = blockIdx.x;
    if (row >= N) return;
    int tid = threadIdx.x;  // [0, 32)

    int n_blocks = K / 16;
    int packed_row_off = row * (K / 2);  // bytes, packed
    int scale_row_off  = row * n_blocks; // bytes, scale

    float acc = 0.0f;

    // Stride-32 over micro-blocks. Each iteration processes 1 micro-block
    // (16 FP4 weights, 8 packed bytes, 1 UE4M3 scale, 16 BF16 inputs).
    for (int b = tid; b < n_blocks; b += 32) {
        const unsigned char* p8 = packed + packed_row_off + b * 8;
        unsigned char sb = scale[scale_row_off + b];
        float s = ue4m3_to_fp32_idx(sb);

        const __nv_bfloat16* xptr = x + b * 16;
        // Load 8 packed bytes + 16 BF16 + accumulate.
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            unsigned char by = p8[i];
            unsigned int lo = by & 0xF;
            unsigned int hi = (by >> 4) & 0xF;
            float w0 = fp4_to_fp32_idx(lo) * s;
            float w1 = fp4_to_fp32_idx(hi) * s;
            float x0 = (float)xptr[2*i + 0];
            float x1 = (float)xptr[2*i + 1];
            acc += w0 * x0 + w1 * x1;
        }
    }

    // Warp-shuffle reduction.
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        acc += __shfl_xor_sync(0xffffffffu, acc, o);
    }

    if (tid == 0) {
        y[row] = (__nv_bfloat16)(acc * alpha);
    }
}
"#;

#[cfg(feature = "cuda")]
const SGEMV_NVFP4_BF16_SRC: &str = r#"
#include <cuda_bf16.h>
#include <cuda_fp16.h>

__device__ __forceinline__ float fp4_to_fp32_solo(unsigned int code) {
    static const float MAG[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    float m = MAG[code & 0x7];
    return (code & 0x8) ? -m : m;
}

__device__ __forceinline__ float ue4m3_to_fp32_solo(unsigned char b) {
    unsigned int e = (b >> 3) & 0xF;
    unsigned int m = b & 0x7;
    if (e == 0) {
        return (float)m * 0.0009765625f;
    }
    int exp_unbiased = (int)e - 7;
    return ldexpf(1.0f + (float)m * 0.125f, exp_unbiased);
}

extern "C" __global__ void sgemv_nvfp4_bf16(
    const unsigned char*  __restrict__ packed,      // [N, K/2] U8
    const unsigned char*  __restrict__ scale,       // [N, K/16] U8 (UE4M3)
    float                              alpha,       // 1 / (w_g * in_g)
    const __nv_bfloat16*  __restrict__ x,
    __nv_bfloat16*        __restrict__ y,
    int N,
    int K
) {
    int row = blockIdx.x;
    if (row >= N) return;
    int tid = threadIdx.x;

    int n_blocks = K / 16;
    int packed_row_off = row * (K / 2);
    int scale_row_off  = row * n_blocks;

    float acc = 0.0f;

    for (int b = tid; b < n_blocks; b += 32) {
        const unsigned char* p8 = packed + packed_row_off + b * 8;
        unsigned char sb = scale[scale_row_off + b];
        float s = ue4m3_to_fp32_solo(sb);

        const __nv_bfloat16* xptr = x + b * 16;
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            unsigned char by = p8[i];
            unsigned int lo = by & 0xF;
            unsigned int hi = (by >> 4) & 0xF;
            float w0 = fp4_to_fp32_solo(lo) * s;
            float w1 = fp4_to_fp32_solo(hi) * s;
            float x0 = (float)xptr[2*i + 0];
            float x1 = (float)xptr[2*i + 1];
            acc += w0 * x0 + w1 * x1;
        }
    }

    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        acc += __shfl_xor_sync(0xffffffffu, acc, o);
    }

    if (tid == 0) {
        y[row] = (__nv_bfloat16)(acc * alpha);
    }
}
"#;

// ────────────────────────────────────────────────────────────────────────
// T246.8 A4 — mul_mm_id mega-kernels (one launch per gate/up/down matmul)
//
// Replaces the K-iteration `sgemv_q?k_bf16_v?_indexed` dispatch loop in
// `moe_ffn_forward_step_async` with a single launch per matmul. The
// canonical llama.cpp pattern (mmvq.cu:597-654, mul_mat_vec_q_moe) maps
// `(blockIdx.y, blockIdx.x)` → `(slot, row_tile)` so each block reads
// `e_idx = topk_indices[blockIdx.y]` instead of the host-bound `slot`.
//
// Output layout : `[k_used, N]` row-major. Per-slot-row scalar lands at
// `y[slot * N + row]`. The caller either dispatches the existing
// per-slot `scaled_add_inplace_bf16_devscalar` epilogue (initial wiring,
// trivially bit-exact) or a fused routed-reduce primitive
// (`scaled_add_routed_bf16`, see below) that sums all K slots in one
// launch.
//
// The PER-SLOT BODY is byte-for-byte identical to the v3 indexed kernel
// it replaces — A3 demonstrated that any reduction-order change (e.g.
// 32→64 lane warp split) produces token-level drift even when synthetic
// parity holds. Reuse > rewrite for FP non-associativity safety.
//
// Block topology mirrors the v3 indexed kernels :
//   block_dim = (128, 1, 1)        — 4 rows × 32 lanes (warp-shuffle SGEMV)
//   grid_dim  = (ceil(N/4), k_used, 1)
//
// Q4_K dp4a path uses the dp4a body : block_dim = (32, 1, 1), one warp
// per row → grid = (N, k_used, 1).
// BF16 path uses the bf16 body : block_dim = (64, 1, 1), 2 warps per
// row → grid = (N, k_used, 1).

#[cfg(feature = "cuda")]
const MUL_MM_ID_Q4_K_BF16_SRC: &str = r#"
#include <cuda_bf16.h>
#include <cuda_fp16.h>

extern "C" __launch_bounds__(128, 8)
__global__ void mul_mm_id_q4_k_bf16(
    const unsigned long long* __restrict__ expert_ptrs,  // [n_experts] u64
    const int*                __restrict__ topk_indices, // [k_used] device i32
    const __nv_bfloat16*      __restrict__ x,            // [K]
    __nv_bfloat16*            __restrict__ y,            // [k_used, N]
    int N,
    int K
) {
    int slot = blockIdx.y;
    int e_idx = topk_indices[slot];
    const unsigned char* __restrict__ w_q4k =
        (const unsigned char* __restrict__)expert_ptrs[e_idx];

    int row0 = blockIdx.x * 4;
    int tid  = threadIdx.x;
    int row_in_block = tid >> 5;
    int lane         = tid & 31;
    int row          = row0 + row_in_block;
    if (row >= N) return;

    int blocks_per_row = K / 256;
    int row_offset     = row * blocks_per_row * 144;

    int group     = lane >> 3;
    int pos_base  = (lane & 7) << 2;
    int byte_base = (group << 5) + pos_base;

    float acc = 0.0f;

    for (int b = 0; b < blocks_per_row; ++b) {
        int blk_off = row_offset + b * 144;
        const unsigned char* blk = w_q4k + blk_off;

        unsigned short d_bits    = blk[0] | (blk[1] << 8);
        unsigned short dmin_bits = blk[2] | (blk[3] << 8);
        float d    = __half2float(__ushort_as_half(d_bits));
        float dmin = __half2float(__ushort_as_half(dmin_bits));

        const unsigned char* scales = blk + 4;
        int sub_a = group * 2;
        int sub_b = sub_a + 1;
        unsigned char sc_a, m_a, sc_b, m_b;
        if (sub_a < 4) {
            sc_a = scales[sub_a]     & 0x3F;
            m_a  = scales[sub_a + 4] & 0x3F;
        } else {
            int ga = sub_a - 4;
            sc_a = (scales[ga + 8] & 0x0F) | ((scales[ga]     >> 6) << 4);
            m_a  = (scales[ga + 8] >> 4)   | ((scales[ga + 4] >> 6) << 4);
        }
        if (sub_b < 4) {
            sc_b = scales[sub_b]     & 0x3F;
            m_b  = scales[sub_b + 4] & 0x3F;
        } else {
            int gb = sub_b - 4;
            sc_b = (scales[gb + 8] & 0x0F) | ((scales[gb]     >> 6) << 4);
            m_b  = (scales[gb + 8] >> 4)   | ((scales[gb + 4] >> 6) << 4);
        }
        float scale_a = d    * (float)sc_a;
        float min_a   = dmin * (float)m_a;
        float scale_b = d    * (float)sc_b;
        float min_b   = dmin * (float)m_b;

        const unsigned char* qs = blk + 16;
        unsigned int qbytes = *(const unsigned int*)(qs + byte_base);

        const __nv_bfloat16* xa_ptr = x + b * 256 + sub_a * 32 + pos_base;
        const __nv_bfloat16* xb_ptr = x + b * 256 + sub_b * 32 + pos_base;
        uint2 xa = *(const uint2*)xa_ptr;
        uint2 xb = *(const uint2*)xb_ptr;
        float xa0 = (float)__ushort_as_bfloat16((unsigned short)(xa.x & 0xFFFFu));
        float xa1 = (float)__ushort_as_bfloat16((unsigned short)(xa.x >> 16));
        float xa2 = (float)__ushort_as_bfloat16((unsigned short)(xa.y & 0xFFFFu));
        float xa3 = (float)__ushort_as_bfloat16((unsigned short)(xa.y >> 16));
        float xb0 = (float)__ushort_as_bfloat16((unsigned short)(xb.x & 0xFFFFu));
        float xb1 = (float)__ushort_as_bfloat16((unsigned short)(xb.x >> 16));
        float xb2 = (float)__ushort_as_bfloat16((unsigned short)(xb.y & 0xFFFFu));
        float xb3 = (float)__ushort_as_bfloat16((unsigned short)(xb.y >> 16));

        unsigned char by0 = (qbytes      ) & 0xFFu;
        unsigned char by1 = (qbytes >>  8) & 0xFFu;
        unsigned char by2 = (qbytes >> 16) & 0xFFu;
        unsigned char by3 = (qbytes >> 24) & 0xFFu;
        int na0 = by0 & 0x0F, na1 = by1 & 0x0F, na2 = by2 & 0x0F, na3 = by3 & 0x0F;
        int nb0 = by0 >>   4, nb1 = by1 >>   4, nb2 = by2 >>   4, nb3 = by3 >>   4;

        acc += (scale_a * (float)na0 - min_a) * xa0;
        acc += (scale_a * (float)na1 - min_a) * xa1;
        acc += (scale_a * (float)na2 - min_a) * xa2;
        acc += (scale_a * (float)na3 - min_a) * xa3;
        acc += (scale_b * (float)nb0 - min_b) * xb0;
        acc += (scale_b * (float)nb1 - min_b) * xb1;
        acc += (scale_b * (float)nb2 - min_b) * xb2;
        acc += (scale_b * (float)nb3 - min_b) * xb3;
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_xor_sync(0xffffffffu, acc, offset);
    }
    if (lane == 0) {
        y[slot * N + row] = (__nv_bfloat16)acc;
    }
}
"#;

#[cfg(feature = "cuda")]
const MUL_MM_ID_Q4_K_Q8_1_DP4A_BF16_SRC: &str = r#"
#include <cuda_bf16.h>
#include <cuda_fp16.h>

extern "C" __global__ void mul_mm_id_q4_k_q8_1_dp4a_bf16(
    const unsigned long long* __restrict__ expert_ptrs,
    const int*                __restrict__ topk_indices,
    const unsigned char*      __restrict__ x_q8_1,
    __nv_bfloat16*            __restrict__ y,
    int N,
    int K
) {
    int slot = blockIdx.y;
    int e_idx = topk_indices[slot];
    const unsigned char* __restrict__ w_q4k =
        (const unsigned char* __restrict__)expert_ptrs[e_idx];

    int row = blockIdx.x;
    if (row >= N) return;
    int tid = threadIdx.x;

    int blocks_per_row = K / 256;
    float acc = 0.0f;

    for (int b = tid; b < blocks_per_row; b += 32) {
        const unsigned char* blk = w_q4k + (row * blocks_per_row + b) * 144;

        float d    = __half2float(*(const __half*)(blk + 0));
        float dmin = __half2float(*(const __half*)(blk + 2));

        const unsigned char* sr = blk + 4;
        unsigned char sc[8], m[8];
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            sc[i]     = sr[i]     & 0x3F;
            m[i]      = sr[i + 4] & 0x3F;
            sc[i + 4] = (sr[i + 8] & 0x0F) | ((sr[i]     >> 6) << 4);
            m[i  + 4] = (sr[i + 8] >>   4) | ((sr[i + 4] >> 6) << 4);
        }

        const unsigned char* qs = blk + 16;

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

    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        acc += __shfl_xor_sync(0xffffffff, acc, o);
    }

    if (tid == 0) {
        y[slot * N + row] = (__nv_bfloat16)acc;
    }
}
"#;

#[cfg(feature = "cuda")]
const MUL_MM_ID_Q5_K_BF16_SRC: &str = r#"
#include <cuda_bf16.h>
#include <cuda_fp16.h>

extern "C" __launch_bounds__(128, 8)
__global__ void mul_mm_id_q5_k_bf16(
    const unsigned long long* __restrict__ expert_ptrs,
    const int*                __restrict__ topk_indices,
    const __nv_bfloat16*      __restrict__ x,
    __nv_bfloat16*            __restrict__ y,
    int N,
    int K
) {
    int slot = blockIdx.y;
    int e_idx = topk_indices[slot];
    const unsigned char* __restrict__ w_q5k =
        (const unsigned char* __restrict__)expert_ptrs[e_idx];

    int row0 = blockIdx.x * 4;
    int tid  = threadIdx.x;
    int row_in_block = tid >> 5;
    int lane         = tid & 31;
    int row          = row0 + row_in_block;
    if (row >= N) return;

    int blocks_per_row = K / 256;
    int row_offset     = row * blocks_per_row * 176;

    int group     = lane >> 3;
    int pos_base  = (lane & 7) << 2;
    int byte_base = (group << 5) + pos_base;
    unsigned int qh_mask_a = 1u << (2 * group);
    unsigned int qh_mask_b = 1u << (2 * group + 1);

    float acc = 0.0f;

    for (int b = 0; b < blocks_per_row; ++b) {
        int blk_off = row_offset + b * 176;
        const unsigned char* blk = w_q5k + blk_off;

        unsigned short d_bits    = blk[0] | (blk[1] << 8);
        unsigned short dmin_bits = blk[2] | (blk[3] << 8);
        float d    = __half2float(__ushort_as_half(d_bits));
        float dmin = __half2float(__ushort_as_half(dmin_bits));

        const unsigned char* scales = blk + 4;
        int sub_a = group * 2;
        int sub_b = sub_a + 1;
        unsigned char sc_a, m_a, sc_b, m_b;
        if (sub_a < 4) {
            sc_a = scales[sub_a]     & 0x3F;
            m_a  = scales[sub_a + 4] & 0x3F;
        } else {
            int ga = sub_a - 4;
            sc_a = (scales[ga + 8] & 0x0F) | ((scales[ga]     >> 6) << 4);
            m_a  = (scales[ga + 8] >> 4)   | ((scales[ga + 4] >> 6) << 4);
        }
        if (sub_b < 4) {
            sc_b = scales[sub_b]     & 0x3F;
            m_b  = scales[sub_b + 4] & 0x3F;
        } else {
            int gb = sub_b - 4;
            sc_b = (scales[gb + 8] & 0x0F) | ((scales[gb]     >> 6) << 4);
            m_b  = (scales[gb + 8] >> 4)   | ((scales[gb + 4] >> 6) << 4);
        }
        float scale_a = d    * (float)sc_a;
        float min_a   = dmin * (float)m_a;
        float scale_b = d    * (float)sc_b;
        float min_b   = dmin * (float)m_b;

        const unsigned char* qh = blk + 16;
        const unsigned char* ql = blk + 16 + 32;

        unsigned int qlbytes = *(const unsigned int*)(ql + byte_base);
        unsigned int qhbytes = *(const unsigned int*)(qh + pos_base);

        const __nv_bfloat16* xa_ptr = x + b * 256 + sub_a * 32 + pos_base;
        const __nv_bfloat16* xb_ptr = x + b * 256 + sub_b * 32 + pos_base;
        uint2 xa = *(const uint2*)xa_ptr;
        uint2 xb = *(const uint2*)xb_ptr;
        float xa0 = (float)__ushort_as_bfloat16((unsigned short)(xa.x & 0xFFFFu));
        float xa1 = (float)__ushort_as_bfloat16((unsigned short)(xa.x >> 16));
        float xa2 = (float)__ushort_as_bfloat16((unsigned short)(xa.y & 0xFFFFu));
        float xa3 = (float)__ushort_as_bfloat16((unsigned short)(xa.y >> 16));
        float xb0 = (float)__ushort_as_bfloat16((unsigned short)(xb.x & 0xFFFFu));
        float xb1 = (float)__ushort_as_bfloat16((unsigned short)(xb.x >> 16));
        float xb2 = (float)__ushort_as_bfloat16((unsigned short)(xb.y & 0xFFFFu));
        float xb3 = (float)__ushort_as_bfloat16((unsigned short)(xb.y >> 16));

        unsigned char qb0 = (qlbytes      ) & 0xFFu;
        unsigned char qb1 = (qlbytes >>  8) & 0xFFu;
        unsigned char qb2 = (qlbytes >> 16) & 0xFFu;
        unsigned char qb3 = (qlbytes >> 24) & 0xFFu;
        unsigned char hb0 = (qhbytes      ) & 0xFFu;
        unsigned char hb1 = (qhbytes >>  8) & 0xFFu;
        unsigned char hb2 = (qhbytes >> 16) & 0xFFu;
        unsigned char hb3 = (qhbytes >> 24) & 0xFFu;

        int qa0 = (qb0 & 0x0F) + ((hb0 & qh_mask_a) ? 16 : 0);
        int qa1 = (qb1 & 0x0F) + ((hb1 & qh_mask_a) ? 16 : 0);
        int qa2 = (qb2 & 0x0F) + ((hb2 & qh_mask_a) ? 16 : 0);
        int qa3 = (qb3 & 0x0F) + ((hb3 & qh_mask_a) ? 16 : 0);
        int qbq0 = (qb0 >>   4) + ((hb0 & qh_mask_b) ? 16 : 0);
        int qbq1 = (qb1 >>   4) + ((hb1 & qh_mask_b) ? 16 : 0);
        int qbq2 = (qb2 >>   4) + ((hb2 & qh_mask_b) ? 16 : 0);
        int qbq3 = (qb3 >>   4) + ((hb3 & qh_mask_b) ? 16 : 0);

        acc += (scale_a * (float)qa0  - min_a) * xa0;
        acc += (scale_a * (float)qa1  - min_a) * xa1;
        acc += (scale_a * (float)qa2  - min_a) * xa2;
        acc += (scale_a * (float)qa3  - min_a) * xa3;
        acc += (scale_b * (float)qbq0 - min_b) * xb0;
        acc += (scale_b * (float)qbq1 - min_b) * xb1;
        acc += (scale_b * (float)qbq2 - min_b) * xb2;
        acc += (scale_b * (float)qbq3 - min_b) * xb3;
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_xor_sync(0xffffffffu, acc, offset);
    }
    if (lane == 0) {
        y[slot * N + row] = (__nv_bfloat16)acc;
    }
}
"#;

#[cfg(feature = "cuda")]
const MUL_MM_ID_Q6_K_BF16_SRC: &str = r#"
#include <cuda_bf16.h>
#include <cuda_fp16.h>

extern "C" __launch_bounds__(128, 8)
__global__ void mul_mm_id_q6_k_bf16(
    const unsigned long long* __restrict__ expert_ptrs,
    const int*                __restrict__ topk_indices,
    const __nv_bfloat16*      __restrict__ x,
    __nv_bfloat16*            __restrict__ y,
    int N,
    int K
) {
    int slot = blockIdx.y;
    int e_idx = topk_indices[slot];
    const unsigned char* __restrict__ w_q6k =
        (const unsigned char* __restrict__)expert_ptrs[e_idx];

    int row0 = blockIdx.x * 4;
    int tid  = threadIdx.x;
    int row_in_block = tid >> 5;
    int lane         = tid & 31;
    int row          = row0 + row_in_block;
    if (row >= N) return;

    int blocks_per_row = K / 256;
    int row_offset     = row * blocks_per_row * 210;
    int l16            = lane >> 4;

    float acc = 0.0f;

    for (int b = 0; b < blocks_per_row; ++b) {
        int blk_off = row_offset + b * 210;
        const unsigned char* blk = w_q6k + blk_off;

        unsigned short d_bits = blk[208] | (blk[209] << 8);
        float d = __half2float(__ushort_as_half(d_bits));

        const signed char* scales = (const signed char*)(blk + 192);
        float sc0_a = d * (float)scales[0 + l16];
        float sc2_a = d * (float)scales[2 + l16];
        float sc4_a = d * (float)scales[4 + l16];
        float sc6_a = d * (float)scales[6 + l16];
        float sc0_b = d * (float)scales[8 + l16];
        float sc2_b = d * (float)scales[10 + l16];
        float sc4_b = d * (float)scales[12 + l16];
        float sc6_b = d * (float)scales[14 + l16];

        const unsigned char* ql = blk;
        const unsigned char* qh = blk + 128;

        unsigned char ql_a0 = ql[0 + lane];
        unsigned char ql_b0 = ql[0 + lane + 32];
        unsigned char qh_0  = qh[0 + lane];
        int q0a = (ql_a0 & 0x0F) | (((qh_0)      & 0x03) << 4);
        int q1a = (ql_b0 & 0x0F) | (((qh_0 >> 2) & 0x03) << 4);
        int q2a = (ql_a0 >> 4)   | (((qh_0 >> 4) & 0x03) << 4);
        int q3a = (ql_b0 >> 4)   | (((qh_0 >> 6) & 0x03) << 4);

        const __nv_bfloat16* x_ptr_a = x + b * 256 + 0;
        float x0a = (float)x_ptr_a[lane];
        float x1a = (float)x_ptr_a[lane + 32];
        float x2a = (float)x_ptr_a[lane + 64];
        float x3a = (float)x_ptr_a[lane + 96];

        acc += sc0_a * (float)(q0a - 32) * x0a;
        acc += sc2_a * (float)(q1a - 32) * x1a;
        acc += sc4_a * (float)(q2a - 32) * x2a;
        acc += sc6_a * (float)(q3a - 32) * x3a;

        unsigned char ql_a1 = ql[64 + lane];
        unsigned char ql_b1 = ql[64 + lane + 32];
        unsigned char qh_1  = qh[32 + lane];
        int q0b = (ql_a1 & 0x0F) | (((qh_1)      & 0x03) << 4);
        int q1b = (ql_b1 & 0x0F) | (((qh_1 >> 2) & 0x03) << 4);
        int q2b = (ql_a1 >> 4)   | (((qh_1 >> 4) & 0x03) << 4);
        int q3b = (ql_b1 >> 4)   | (((qh_1 >> 6) & 0x03) << 4);

        const __nv_bfloat16* x_ptr_b = x + b * 256 + 128;
        float x0b = (float)x_ptr_b[lane];
        float x1b = (float)x_ptr_b[lane + 32];
        float x2b = (float)x_ptr_b[lane + 64];
        float x3b = (float)x_ptr_b[lane + 96];

        acc += sc0_b * (float)(q0b - 32) * x0b;
        acc += sc2_b * (float)(q1b - 32) * x1b;
        acc += sc4_b * (float)(q2b - 32) * x2b;
        acc += sc6_b * (float)(q3b - 32) * x3b;
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_xor_sync(0xffffffffu, acc, offset);
    }
    if (lane == 0) {
        y[slot * N + row] = (__nv_bfloat16)acc;
    }
}
"#;

#[cfg(feature = "cuda")]
const MUL_MM_ID_BF16_BF16_SRC: &str = r#"
#include <cuda_bf16.h>

extern "C" __global__ void mul_mm_id_bf16_bf16(
    const unsigned long long* __restrict__ expert_ptrs,
    const int*                __restrict__ topk_indices,
    const __nv_bfloat16*      __restrict__ x,
    __nv_bfloat16*            __restrict__ y,
    int N,
    int K
) {
    int slot = blockIdx.y;
    int e_idx = topk_indices[slot];
    const __nv_bfloat16* __restrict__ w =
        (const __nv_bfloat16* __restrict__)expert_ptrs[e_idx];

    int row = blockIdx.x;
    if (row >= N) return;
    int tid = threadIdx.x;

    extern __shared__ float shmem[];

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
        y[slot * N + row] = (__nv_bfloat16)total;
    }
}
"#;

// =====================================================================
// T246.10 TrackE.2 — batched MoE Group-GEMM kernels.
//
// Extension of the A4 `mul_mm_id_*` family (M=1, indexed) to variable M.
// Each block processes exactly one (row, slot, token) triple — identical
// per-block body to the M=1 variant — but the grid is 3D
// `(rows/ROWS_PER_BLOCK, k_used, M)` so a single launch covers ALL
// `(token, slot)` pairs across the prefill tree.
//
// Layout :
//   - expert_ptrs  : [n_experts]  u64 device pointers (one per stacked expert)
//   - topk_indices : [M, k_used]  i32 row-major (per-token routing)
//   - x            : [M, K]       BF16 row-major (per-token activations)
//   - y            : [M, k_used, N] BF16, token-major slot-major (output)
//
// Inside the kernel : block (bx, by, bz) reads
//   token  = bz
//   slot   = by
//   row0   = bx * ROWS_PER_BLOCK
//   e_idx  = topk_indices[token * k_used + slot]
//   x_ptr  = x + token * K
//   y_ptr  = y + (token * k_used + slot) * N + row
//
// This is byte-for-byte identical to running the M=1 kernel M times with
// (token, slot) input. Bit-exact parity is guaranteed.
//
// Launch-count reduction (Qwen3.6-A3B prefill, N=512, k=8, 64 MoE layers) :
//   before : 512 × (1 gate + 1 up + 8 down) per layer = 5120 launches/layer
//          × 64 MoE layers = 327k launches/prefill (gate/up via mega + down
//          per-slot via dispatch_indexed_matmul_m1 inner loop)
//   after  : 1 gate + 1 up + 8 down = 10 launches/layer × 64 = 640/prefill
//   net    : ~500× launch reduction on the MoE FFN hot path.
//
// W-amortization across M : when multiple tokens route to the same expert
// for some slot (~M*k_used / n_experts ≈ 32 collisions/expert at M=512,
// k=8, n_e=128), the same expert weight rows are read by multiple blocks
// in the same launch. L2 cache absorbs the redundancy. A canonical
// sort-permutation kernel would group all colliding (token, slot) into
// adjacent blocks for guaranteed L1 reuse, but per the A6.b finding the
// warp-shuffle path is memory-bound at the bandwidth limit, so the simpler
// non-sorted version is expected to capture most of the win.

#[cfg(feature = "cuda")]
const MUL_MM_ID_GEMM_Q4_K_BF16_SRC: &str = r#"
#include <cuda_bf16.h>
#include <cuda_fp16.h>

extern "C" __global__ void mul_mm_id_gemm_q4_k_bf16(
    const unsigned long long* __restrict__ expert_ptrs,  // [n_experts]
    const int*                __restrict__ topk_indices, // [M, k_used]
    const __nv_bfloat16*      __restrict__ x,            // [M, K]
    __nv_bfloat16*            __restrict__ y,            // [M, k_used, N]
    int N,
    int K,
    int k_used
) {
    int token = blockIdx.z;
    int slot  = blockIdx.y;
    int e_idx = topk_indices[token * k_used + slot];
    const unsigned char* __restrict__ w_q4k =
        (const unsigned char* __restrict__)expert_ptrs[e_idx];

    int row0 = blockIdx.x * 4;
    int tid  = threadIdx.x;
    int row_in_block = tid >> 5;
    int lane         = tid & 31;
    int row          = row0 + row_in_block;
    if (row >= N) return;

    int blocks_per_row = K / 256;
    int row_offset     = row * blocks_per_row * 144;

    int group     = lane >> 3;
    int pos_base  = (lane & 7) << 2;
    int byte_base = (group << 5) + pos_base;

    // Per-token x base pointer (each token has its own [K] activation row).
    const __nv_bfloat16* x_tok = x + (long long)token * K;

    float acc = 0.0f;

    for (int b = 0; b < blocks_per_row; ++b) {
        int blk_off = row_offset + b * 144;
        const unsigned char* blk = w_q4k + blk_off;

        unsigned short d_bits    = blk[0] | (blk[1] << 8);
        unsigned short dmin_bits = blk[2] | (blk[3] << 8);
        float d    = __half2float(__ushort_as_half(d_bits));
        float dmin = __half2float(__ushort_as_half(dmin_bits));

        const unsigned char* scales = blk + 4;
        int sub_a = group * 2;
        int sub_b = sub_a + 1;
        unsigned char sc_a, m_a, sc_b, m_b;
        if (sub_a < 4) {
            sc_a = scales[sub_a]     & 0x3F;
            m_a  = scales[sub_a + 4] & 0x3F;
        } else {
            int ga = sub_a - 4;
            sc_a = (scales[ga + 8] & 0x0F) | ((scales[ga]     >> 6) << 4);
            m_a  = (scales[ga + 8] >> 4)   | ((scales[ga + 4] >> 6) << 4);
        }
        if (sub_b < 4) {
            sc_b = scales[sub_b]     & 0x3F;
            m_b  = scales[sub_b + 4] & 0x3F;
        } else {
            int gb = sub_b - 4;
            sc_b = (scales[gb + 8] & 0x0F) | ((scales[gb]     >> 6) << 4);
            m_b  = (scales[gb + 8] >> 4)   | ((scales[gb + 4] >> 6) << 4);
        }
        float scale_a = d    * (float)sc_a;
        float min_a   = dmin * (float)m_a;
        float scale_b = d    * (float)sc_b;
        float min_b   = dmin * (float)m_b;

        const unsigned char* qs = blk + 16;
        unsigned int qbytes = *(const unsigned int*)(qs + byte_base);

        const __nv_bfloat16* xa_ptr = x_tok + b * 256 + sub_a * 32 + pos_base;
        const __nv_bfloat16* xb_ptr = x_tok + b * 256 + sub_b * 32 + pos_base;
        uint2 xa = *(const uint2*)xa_ptr;
        uint2 xb = *(const uint2*)xb_ptr;
        float xa0 = (float)__ushort_as_bfloat16((unsigned short)(xa.x & 0xFFFFu));
        float xa1 = (float)__ushort_as_bfloat16((unsigned short)(xa.x >> 16));
        float xa2 = (float)__ushort_as_bfloat16((unsigned short)(xa.y & 0xFFFFu));
        float xa3 = (float)__ushort_as_bfloat16((unsigned short)(xa.y >> 16));
        float xb0 = (float)__ushort_as_bfloat16((unsigned short)(xb.x & 0xFFFFu));
        float xb1 = (float)__ushort_as_bfloat16((unsigned short)(xb.x >> 16));
        float xb2 = (float)__ushort_as_bfloat16((unsigned short)(xb.y & 0xFFFFu));
        float xb3 = (float)__ushort_as_bfloat16((unsigned short)(xb.y >> 16));

        unsigned char by0 = (qbytes      ) & 0xFFu;
        unsigned char by1 = (qbytes >>  8) & 0xFFu;
        unsigned char by2 = (qbytes >> 16) & 0xFFu;
        unsigned char by3 = (qbytes >> 24) & 0xFFu;
        int na0 = by0 & 0x0F, na1 = by1 & 0x0F, na2 = by2 & 0x0F, na3 = by3 & 0x0F;
        int nb0 = by0 >>   4, nb1 = by1 >>   4, nb2 = by2 >>   4, nb3 = by3 >>   4;

        acc += (scale_a * (float)na0 - min_a) * xa0;
        acc += (scale_a * (float)na1 - min_a) * xa1;
        acc += (scale_a * (float)na2 - min_a) * xa2;
        acc += (scale_a * (float)na3 - min_a) * xa3;
        acc += (scale_b * (float)nb0 - min_b) * xb0;
        acc += (scale_b * (float)nb1 - min_b) * xb1;
        acc += (scale_b * (float)nb2 - min_b) * xb2;
        acc += (scale_b * (float)nb3 - min_b) * xb3;
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_xor_sync(0xffffffffu, acc, offset);
    }
    if (lane == 0) {
        // Output layout : y[token, slot, row] in [M, k_used, N] row-major.
        y[((long long)token * k_used + slot) * N + row] = (__nv_bfloat16)acc;
    }
}
"#;

#[cfg(feature = "cuda")]
const MUL_MM_ID_GEMM_Q4_K_Q8_1_DP4A_BF16_SRC: &str = r#"
#include <cuda_bf16.h>
#include <cuda_fp16.h>

extern "C" __global__ void mul_mm_id_gemm_q4_k_q8_1_dp4a_bf16(
    const unsigned long long* __restrict__ expert_ptrs,
    const int*                __restrict__ topk_indices, // [M, k_used]
    const unsigned char*      __restrict__ x_q8_1,       // [M, K/32 * 36]
    __nv_bfloat16*            __restrict__ y,            // [M, k_used, N]
    int N,
    int K,
    int k_used
) {
    int token = blockIdx.z;
    int slot  = blockIdx.y;
    int e_idx = topk_indices[token * k_used + slot];
    const unsigned char* __restrict__ w_q4k =
        (const unsigned char* __restrict__)expert_ptrs[e_idx];

    int row = blockIdx.x;
    if (row >= N) return;
    int tid = threadIdx.x;

    int blocks_per_row = K / 256;
    // Per-token q8_1 buffer base (each token's K activation is K/32 super-blocks
    // of 36 bytes each).
    int q8_per_tok = (K / 32) * 36;
    const unsigned char* x_q8_tok = x_q8_1 + (long long)token * q8_per_tok;

    float acc = 0.0f;

    for (int b = tid; b < blocks_per_row; b += 32) {
        const unsigned char* blk = w_q4k + (row * blocks_per_row + b) * 144;

        float d    = __half2float(*(const __half*)(blk + 0));
        float dmin = __half2float(*(const __half*)(blk + 2));

        const unsigned char* sr = blk + 4;
        unsigned char sc[8], m[8];
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            sc[i]     = sr[i]     & 0x3F;
            m[i]      = sr[i + 4] & 0x3F;
            sc[i + 4] = (sr[i + 8] & 0x0F) | ((sr[i]     >> 6) << 4);
            m[i  + 4] = (sr[i + 8] >>   4) | ((sr[i + 4] >> 6) << 4);
        }

        const unsigned char* qs = blk + 16;

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
                    const unsigned char* x_blk = x_q8_tok + sb_idx * 36;
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

    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        acc += __shfl_xor_sync(0xffffffff, acc, o);
    }

    if (tid == 0) {
        y[((long long)token * k_used + slot) * N + row] = (__nv_bfloat16)acc;
    }
}
"#;

// ─── TrackE.3 — sort-permutation Group-GEMM (cache-reuse win) ─────────────
//
// Two kernels port the llama.cpp mmid.cu + mmvq.cu mul_mat_vec_q_moe pattern :
//
//   1. mm_ids_helper_bf16<8>   : one block per expert, scans the [M, k_used]
//      topk_indices tensor and emits compact-by-expert permutation arrays
//      `ids_src1[M*k_used]`, `ids_dst[M*k_used]`, `expert_bounds[n_experts+1]`.
//      The compact layout has all slots routed to expert 0 first, then expert 1,
//      etc. — so the consumer kernel below can iterate slot 0..M*k_used and
//      adjacent blocks share the SAME expert → L1/L2 weight reuse.
//
//   2. mul_mm_id_gemm_q4_k_sorted_bf16 : same per-(slot, row) body as
//      mul_mm_id_gemm_q4_k_bf16 (warp-shuffle Q4_K BF16 dot product), but
//      gridZ iterates the COMPACT slot index. Each block reads
//      `ids_src1[c] → token_src` (= which token's [K] activation row to read)
//      and `ids_dst[c] → dst_index` (= which (token, slot_orig) position in
//      the output [M, k_used, N] to write). Expert pointer is selected via
//      `expert_id` which is implicit in the sorted order — we look it up
//      from the original topk_indices using `ids_src1[c]` and `ids_dst[c]`.
//
// CRITICAL : the FP arithmetic in the inner loop is byte-for-byte identical
// to TrackE.2's `mul_mm_id_gemm_q4_k_bf16` — only the schedule differs.
// → bit-exact parity is REQUIRED and tested.

#[cfg(feature = "cuda")]
const MM_IDS_HELPER_BF16_SRC: &str = r#"
#ifndef INT_MAX
#define INT_MAX 0x7FFFFFFF
#endif
// One block per expert. Block has 1 warp (32 threads). Scans the [M, k_used]
// topk_indices tensor and emits :
//   ids_src1[M*k_used] : compact_idx → source token (which row of x to read).
//   ids_dst [M*k_used] : compact_idx → destination row in [M, k_used] flat layout
//                        (= token_src * k_used + slot_orig).
//   expert_bounds[n_experts+1] : prefix-sum offsets per expert (last is total slots).
//
// Templated on k_used (Qwen3.6 uses k_used=8 ; pad to next power-of-2 for warp scan).
// For k_used=8, neu_padded = 8 ; 32/8 = 4 tokens processed per warp-iteration.

extern "C" __global__ void mm_ids_helper_bf16(
    const int* __restrict__ topk_indices,  // [n_tokens, k_used]
    int*       __restrict__ ids_src1,      // [n_tokens * k_used]
    int*       __restrict__ ids_dst,       // [n_tokens * k_used]
    int*       __restrict__ expert_bounds, // [n_experts + 1]
    int n_tokens,
    int k_used
) {
    const int expert = blockIdx.x;
    const int n_experts = gridDim.x;
    const int warp_size = 32;
    const int neu_padded = (k_used == 6) ? 8 : k_used;
    const int tid = threadIdx.x;

    extern __shared__ unsigned int smem_store[]; // [n_tokens] packed (token:22, slot:10)

    int nex_prev   = 0;
    int it_compact = 0;

    // Iterate tokens, warp_size / neu_padded tokens per warp-iteration.
    for (int it0 = 0; it0 < n_tokens; it0 += warp_size / neu_padded) {
        int it  = it0 + tid / neu_padded;
        int iex = tid % neu_padded;

        int expert_used;
        if (iex < k_used && it < n_tokens) {
            expert_used = topk_indices[it * k_used + iex];
        } else {
            expert_used = INT_MAX;
        }

        int iex_used = (expert_used == expert) ? iex : -1;
        nex_prev += (expert_used < expert);

        // Per-token presence flag : whether ANY thread in this token's group
        // selected the expert.
        unsigned int mask_padded = (1u << neu_padded) - 1u;
        unsigned int group_mask = mask_padded << ((tid / neu_padded) * neu_padded);
        int it_compact_add_self = (__ballot_sync(0xFFFFFFFFu, iex_used != -1) & group_mask) ? 1 : 0;

        // Scan over LOWER token positions in the warp (warp prefix sum).
        int it_compact_add_lower = 0;
        #pragma unroll
        for (int offset = neu_padded; offset < warp_size; offset += neu_padded) {
            int tmp = __shfl_up_sync(0xFFFFFFFFu, it_compact_add_self, offset, warp_size);
            if (tid >= (unsigned)offset) {
                it_compact_add_lower += tmp;
            }
        }

        if (iex_used != -1) {
            // Pack (token:22, slot:10) into a single uint32 for compact smem.
            unsigned int packed =
                ((unsigned int)it & 0x003FFFFFu) |
                ((unsigned int)iex_used << 22);
            smem_store[it_compact + it_compact_add_lower] = packed;
        }

        // Total tokens in this iteration : the highest-laned thread has the
        // full warp sum.
        int batch_total = __shfl_sync(
            0xFFFFFFFFu,
            it_compact_add_lower + it_compact_add_self,
            warp_size - 1, warp_size);
        it_compact += batch_total;
    }

    // Warp-reduce nex_prev (total slots routed to experts with lower id).
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        nex_prev += __shfl_xor_sync(0xFFFFFFFFu, nex_prev, offset);
    }

    // Write our slice of (ids_src1, ids_dst) — one entry per compact slot for this expert.
    for (int itc = tid; itc < it_compact; itc += warp_size) {
        unsigned int packed = smem_store[itc];
        int token    = (int)(packed & 0x003FFFFFu);
        int slot_orig = (int)(packed >> 22);
        ids_src1[nex_prev + itc] = token;
        // ids_dst is the linear index in the [M, k_used] flat output layout :
        ids_dst[nex_prev + itc] = token * k_used + slot_orig;
    }

    if (tid != 0) return;
    expert_bounds[expert] = nex_prev;
    if (expert == n_experts - 1) {
        expert_bounds[n_experts] = nex_prev + it_compact;
    }
}
"#;

#[cfg(feature = "cuda")]
const MUL_MM_ID_GEMM_Q4_K_SORTED_BF16_SRC: &str = r#"
#include <cuda_bf16.h>
#include <cuda_fp16.h>

// Sort-permutation Q4_K group-GEMM. Same per-(slot, row) inner body as
// mul_mm_id_gemm_q4_k_bf16, but iterates the COMPACT slot index in gridZ.
// `ids_src1[compact_idx]` gives the source token. `ids_dst[compact_idx]`
// gives the destination flat row index in [M, k_used]. Expert pointer is
// recovered via topk_indices[ids_src1[c] * k_used + (ids_dst[c] mod k_used)].
//
// FP arithmetic IDENTICAL to mul_mm_id_gemm_q4_k_bf16 → bit-exact parity.

extern "C" __global__ void mul_mm_id_gemm_q4_k_sorted_bf16(
    const unsigned long long* __restrict__ expert_ptrs,
    const int*                __restrict__ topk_indices,  // [M, k_used]
    const int*                __restrict__ ids_src1,      // [n_slots] (compact_idx → token)
    const int*                __restrict__ ids_dst,       // [n_slots] (compact_idx → token*k_used+slot)
    const __nv_bfloat16*      __restrict__ x,             // [M, K]
    __nv_bfloat16*            __restrict__ y,             // [M, k_used, N]
    int N,
    int K,
    int k_used
) {
    int compact_idx = blockIdx.z;
    int token   = ids_src1[compact_idx];
    int dst_lin = ids_dst[compact_idx];
    int slot    = dst_lin - token * k_used; // == dst_lin % k_used, but cheap
    int e_idx = topk_indices[token * k_used + slot];
    const unsigned char* __restrict__ w_q4k =
        (const unsigned char* __restrict__)expert_ptrs[e_idx];

    int row0 = blockIdx.x * 4;
    int tid  = threadIdx.x;
    int row_in_block = tid >> 5;
    int lane         = tid & 31;
    int row          = row0 + row_in_block;
    if (row >= N) return;

    int blocks_per_row = K / 256;
    int row_offset     = row * blocks_per_row * 144;

    int group     = lane >> 3;
    int pos_base  = (lane & 7) << 2;
    int byte_base = (group << 5) + pos_base;

    const __nv_bfloat16* x_tok = x + (long long)token * K;

    float acc = 0.0f;

    for (int b = 0; b < blocks_per_row; ++b) {
        int blk_off = row_offset + b * 144;
        const unsigned char* blk = w_q4k + blk_off;

        unsigned short d_bits    = blk[0] | (blk[1] << 8);
        unsigned short dmin_bits = blk[2] | (blk[3] << 8);
        float d    = __half2float(__ushort_as_half(d_bits));
        float dmin = __half2float(__ushort_as_half(dmin_bits));

        const unsigned char* scales = blk + 4;
        int sub_a = group * 2;
        int sub_b = sub_a + 1;
        unsigned char sc_a, m_a, sc_b, m_b;
        if (sub_a < 4) {
            sc_a = scales[sub_a]     & 0x3F;
            m_a  = scales[sub_a + 4] & 0x3F;
        } else {
            int ga = sub_a - 4;
            sc_a = (scales[ga + 8] & 0x0F) | ((scales[ga]     >> 6) << 4);
            m_a  = (scales[ga + 8] >> 4)   | ((scales[ga + 4] >> 6) << 4);
        }
        if (sub_b < 4) {
            sc_b = scales[sub_b]     & 0x3F;
            m_b  = scales[sub_b + 4] & 0x3F;
        } else {
            int gb = sub_b - 4;
            sc_b = (scales[gb + 8] & 0x0F) | ((scales[gb]     >> 6) << 4);
            m_b  = (scales[gb + 8] >> 4)   | ((scales[gb + 4] >> 6) << 4);
        }
        float scale_a = d    * (float)sc_a;
        float min_a   = dmin * (float)m_a;
        float scale_b = d    * (float)sc_b;
        float min_b   = dmin * (float)m_b;

        const unsigned char* qs = blk + 16;
        unsigned int qbytes = *(const unsigned int*)(qs + byte_base);

        const __nv_bfloat16* xa_ptr = x_tok + b * 256 + sub_a * 32 + pos_base;
        const __nv_bfloat16* xb_ptr = x_tok + b * 256 + sub_b * 32 + pos_base;
        uint2 xa = *(const uint2*)xa_ptr;
        uint2 xb = *(const uint2*)xb_ptr;
        float xa0 = (float)__ushort_as_bfloat16((unsigned short)(xa.x & 0xFFFFu));
        float xa1 = (float)__ushort_as_bfloat16((unsigned short)(xa.x >> 16));
        float xa2 = (float)__ushort_as_bfloat16((unsigned short)(xa.y & 0xFFFFu));
        float xa3 = (float)__ushort_as_bfloat16((unsigned short)(xa.y >> 16));
        float xb0 = (float)__ushort_as_bfloat16((unsigned short)(xb.x & 0xFFFFu));
        float xb1 = (float)__ushort_as_bfloat16((unsigned short)(xb.x >> 16));
        float xb2 = (float)__ushort_as_bfloat16((unsigned short)(xb.y & 0xFFFFu));
        float xb3 = (float)__ushort_as_bfloat16((unsigned short)(xb.y >> 16));

        unsigned char by0 = (qbytes      ) & 0xFFu;
        unsigned char by1 = (qbytes >>  8) & 0xFFu;
        unsigned char by2 = (qbytes >> 16) & 0xFFu;
        unsigned char by3 = (qbytes >> 24) & 0xFFu;
        int na0 = by0 & 0x0F, na1 = by1 & 0x0F, na2 = by2 & 0x0F, na3 = by3 & 0x0F;
        int nb0 = by0 >>   4, nb1 = by1 >>   4, nb2 = by2 >>   4, nb3 = by3 >>   4;

        acc += (scale_a * (float)na0 - min_a) * xa0;
        acc += (scale_a * (float)na1 - min_a) * xa1;
        acc += (scale_a * (float)na2 - min_a) * xa2;
        acc += (scale_a * (float)na3 - min_a) * xa3;
        acc += (scale_b * (float)nb0 - min_b) * xb0;
        acc += (scale_b * (float)nb1 - min_b) * xb1;
        acc += (scale_b * (float)nb2 - min_b) * xb2;
        acc += (scale_b * (float)nb3 - min_b) * xb3;
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_xor_sync(0xffffffffu, acc, offset);
    }
    if (lane == 0) {
        // Output : y[token, slot, row] in flat [M, k_used, N] layout.
        // dst_lin = token * k_used + slot, so y[dst_lin * N + row].
        y[(long long)dst_lin * N + row] = (__nv_bfloat16)acc;
    }
}
"#;

// Q4K-MOE-CUBLAS.1 — Q4_K -> BF16 dequant device kernel.
//
// Used by the Phase 1 validation bench (dequant + cuBLASLt) and by the Phase 2
// MoE dispatch when RUSTORCH_Q4K_MOE_CUBLAS=1.
//
// Layout : Q4_K super-block = 144 bytes for 256 elements. Each super-block has
//   - 2 bytes d (f16)
//   - 2 bytes dmin (f16)
//   - 12 bytes packed (sc, m) for 8 sub-blocks of 32 elements each
//   - 128 bytes quants (4 bits per element)
//
// Each thread block dequantizes ONE super-block (256 BF16 outputs). Grid is
// 1-D : total_blocks = (total_elements / 256). Block size = 64 threads, each
// thread emits 4 BF16 outputs (the 4 nibbles of 2 packed bytes).
//
// The FP arithmetic mirrors `sgemv_q4k_bf16` / `mul_mm_id_gemm_q4_k_*` :
//     value = d * sc[sub] * nibble - dmin * m[sub]
#[cfg(feature = "cuda")]
const DEQUANT_Q4_K_TO_BF16_SRC: &str = r#"
#include <cuda_bf16.h>
#include <cuda_fp16.h>

extern "C" __global__ void dequant_q4_k_to_bf16(
    const unsigned char* __restrict__ w_q4k, // [n_blocks * 144]
    __nv_bfloat16*       __restrict__ out,   // [n_blocks * 256]
    long long                          n_blocks
) {
    long long blk_idx = (long long)blockIdx.x;
    if (blk_idx >= n_blocks) return;
    int tid = threadIdx.x;

    const unsigned char* blk = w_q4k + blk_idx * 144;

    unsigned short d_bits    = blk[0] | (blk[1] << 8);
    unsigned short dmin_bits = blk[2] | (blk[3] << 8);
    float d    = __half2float(__ushort_as_half(d_bits));
    float dmin = __half2float(__ushort_as_half(dmin_bits));

    // Decode 8 (sc, m) pairs from the 12 packed scale bytes.
    // (Stored as __shared__ float so threads in the same block share the
    // decode work — 16 values × 4 bytes = 64 B, well within shmem budget.)
    __shared__ float scale[8];
    __shared__ float min_v[8];
    if (tid < 8) {
        const unsigned char* scales = blk + 4;
        unsigned char sc, m;
        if (tid < 4) {
            sc = scales[tid]     & 0x3F;
            m  = scales[tid + 4] & 0x3F;
        } else {
            int g = tid - 4;
            sc = (scales[g + 8] & 0x0F) | ((scales[g]     >> 6) << 4);
            m  = (scales[g + 8] >> 4)   | ((scales[g + 4] >> 6) << 4);
        }
        scale[tid] = d    * (float)sc;
        min_v[tid] = dmin * (float)m;
    }
    __syncthreads();

    // Each thread emits 4 outputs : 2 from the low nibble (sub_a) and
    // 2 from the high nibble (sub_b) of 2 consecutive packed bytes.
    // tid in [0, 64) maps to :
    //   group   = tid >> 3   ∈ [0, 8)  — which (sub_a, sub_b) pair
    //   pos_lo  = (tid & 7) << 2       — byte offset within the group's 32-byte qs slice
    //   byte_base = (group << 5) + pos_lo
    const unsigned char* qs = blk + 16;
    int group     = tid >> 3;
    int pos_base  = (tid & 7) << 2;
    int byte_base = (group << 5) + pos_base;
    int sub_a = group * 2;
    int sub_b = sub_a + 1;
    float sa = scale[sub_a], ma = min_v[sub_a];
    float sb = scale[sub_b], mb = min_v[sub_b];

    unsigned int qbytes = *(const unsigned int*)(qs + byte_base);
    unsigned char by0 = (qbytes      ) & 0xFFu;
    unsigned char by1 = (qbytes >>  8) & 0xFFu;
    unsigned char by2 = (qbytes >> 16) & 0xFFu;
    unsigned char by3 = (qbytes >> 24) & 0xFFu;
    int na0 = by0 & 0x0F, na1 = by1 & 0x0F, na2 = by2 & 0x0F, na3 = by3 & 0x0F;
    int nb0 = by0 >>   4, nb1 = by1 >>   4, nb2 = by2 >>   4, nb3 = by3 >>   4;

    long long out_base = blk_idx * 256;

    // Layout of the dequantized super-block matches the natural quantizer
    // layout — sub-block i occupies elements [i*32, (i+1)*32).
    long long off_a = out_base + (long long)sub_a * 32 + pos_base;
    long long off_b = out_base + (long long)sub_b * 32 + pos_base;
    out[off_a + 0] = (__nv_bfloat16)(sa * (float)na0 - ma);
    out[off_a + 1] = (__nv_bfloat16)(sa * (float)na1 - ma);
    out[off_a + 2] = (__nv_bfloat16)(sa * (float)na2 - ma);
    out[off_a + 3] = (__nv_bfloat16)(sa * (float)na3 - ma);
    out[off_b + 0] = (__nv_bfloat16)(sb * (float)nb0 - mb);
    out[off_b + 1] = (__nv_bfloat16)(sb * (float)nb1 - mb);
    out[off_b + 2] = (__nv_bfloat16)(sb * (float)nb2 - mb);
    out[off_b + 3] = (__nv_bfloat16)(sb * (float)nb3 - mb);
}
"#;

// T246.10 TrackE.4 — Q5_K sort-permutation Group-GEMM (cache reuse).
//
// Same per-(slot, row) inner body as `mul_mm_id_gemm_q5_k_bf16`, but iterates
// the COMPACT slot index in gridZ. `ids_src1[compact_idx]` → source token.
// `ids_dst[compact_idx]` → destination flat row index in `[M, k_used]`.
// Expert pointer is recovered via `topk_indices[token * k_used + slot]`.
//
// FP arithmetic IDENTICAL to `mul_mm_id_gemm_q5_k_bf16` → bit-exact parity.
#[cfg(feature = "cuda")]
const MUL_MM_ID_GEMM_Q5_K_SORTED_BF16_SRC: &str = r#"
#include <cuda_bf16.h>
#include <cuda_fp16.h>

extern "C" __global__ void mul_mm_id_gemm_q5_k_sorted_bf16(
    const unsigned long long* __restrict__ expert_ptrs,
    const int*                __restrict__ topk_indices,  // [M, k_used]
    const int*                __restrict__ ids_src1,      // [n_slots] (compact_idx → token)
    const int*                __restrict__ ids_dst,       // [n_slots] (compact_idx → token*k_used+slot)
    const __nv_bfloat16*      __restrict__ x,             // [M, K]
    __nv_bfloat16*            __restrict__ y,             // [M, k_used, N]
    int N,
    int K,
    int k_used
) {
    int compact_idx = blockIdx.z;
    int token   = ids_src1[compact_idx];
    int dst_lin = ids_dst[compact_idx];
    int slot    = dst_lin - token * k_used; // == dst_lin % k_used, but cheap
    int e_idx = topk_indices[token * k_used + slot];
    const unsigned char* __restrict__ w_q5k =
        (const unsigned char* __restrict__)expert_ptrs[e_idx];

    int row0 = blockIdx.x * 4;
    int tid  = threadIdx.x;
    int row_in_block = tid >> 5;
    int lane         = tid & 31;
    int row          = row0 + row_in_block;
    if (row >= N) return;

    int blocks_per_row = K / 256;
    int row_offset     = row * blocks_per_row * 176;

    int group     = lane >> 3;
    int pos_base  = (lane & 7) << 2;
    int byte_base = (group << 5) + pos_base;
    unsigned int qh_mask_a = 1u << (2 * group);
    unsigned int qh_mask_b = 1u << (2 * group + 1);

    const __nv_bfloat16* x_tok = x + (long long)token * K;

    float acc = 0.0f;

    for (int b = 0; b < blocks_per_row; ++b) {
        int blk_off = row_offset + b * 176;
        const unsigned char* blk = w_q5k + blk_off;

        unsigned short d_bits    = blk[0] | (blk[1] << 8);
        unsigned short dmin_bits = blk[2] | (blk[3] << 8);
        float d    = __half2float(__ushort_as_half(d_bits));
        float dmin = __half2float(__ushort_as_half(dmin_bits));

        const unsigned char* scales = blk + 4;
        int sub_a = group * 2;
        int sub_b = sub_a + 1;
        unsigned char sc_a, m_a, sc_b, m_b;
        if (sub_a < 4) {
            sc_a = scales[sub_a]     & 0x3F;
            m_a  = scales[sub_a + 4] & 0x3F;
        } else {
            int ga = sub_a - 4;
            sc_a = (scales[ga + 8] & 0x0F) | ((scales[ga]     >> 6) << 4);
            m_a  = (scales[ga + 8] >> 4)   | ((scales[ga + 4] >> 6) << 4);
        }
        if (sub_b < 4) {
            sc_b = scales[sub_b]     & 0x3F;
            m_b  = scales[sub_b + 4] & 0x3F;
        } else {
            int gb = sub_b - 4;
            sc_b = (scales[gb + 8] & 0x0F) | ((scales[gb]     >> 6) << 4);
            m_b  = (scales[gb + 8] >> 4)   | ((scales[gb + 4] >> 6) << 4);
        }
        float scale_a = d    * (float)sc_a;
        float min_a   = dmin * (float)m_a;
        float scale_b = d    * (float)sc_b;
        float min_b   = dmin * (float)m_b;

        const unsigned char* qh = blk + 16;
        const unsigned char* ql = blk + 16 + 32;

        unsigned int qlbytes = *(const unsigned int*)(ql + byte_base);
        unsigned int qhbytes = *(const unsigned int*)(qh + pos_base);

        const __nv_bfloat16* xa_ptr = x_tok + b * 256 + sub_a * 32 + pos_base;
        const __nv_bfloat16* xb_ptr = x_tok + b * 256 + sub_b * 32 + pos_base;
        uint2 xa = *(const uint2*)xa_ptr;
        uint2 xb = *(const uint2*)xb_ptr;
        float xa0 = (float)__ushort_as_bfloat16((unsigned short)(xa.x & 0xFFFFu));
        float xa1 = (float)__ushort_as_bfloat16((unsigned short)(xa.x >> 16));
        float xa2 = (float)__ushort_as_bfloat16((unsigned short)(xa.y & 0xFFFFu));
        float xa3 = (float)__ushort_as_bfloat16((unsigned short)(xa.y >> 16));
        float xb0 = (float)__ushort_as_bfloat16((unsigned short)(xb.x & 0xFFFFu));
        float xb1 = (float)__ushort_as_bfloat16((unsigned short)(xb.x >> 16));
        float xb2 = (float)__ushort_as_bfloat16((unsigned short)(xb.y & 0xFFFFu));
        float xb3 = (float)__ushort_as_bfloat16((unsigned short)(xb.y >> 16));

        unsigned char qb0 = (qlbytes      ) & 0xFFu;
        unsigned char qb1 = (qlbytes >>  8) & 0xFFu;
        unsigned char qb2 = (qlbytes >> 16) & 0xFFu;
        unsigned char qb3 = (qlbytes >> 24) & 0xFFu;
        unsigned char hb0 = (qhbytes      ) & 0xFFu;
        unsigned char hb1 = (qhbytes >>  8) & 0xFFu;
        unsigned char hb2 = (qhbytes >> 16) & 0xFFu;
        unsigned char hb3 = (qhbytes >> 24) & 0xFFu;

        int qa0 = (qb0 & 0x0F) + ((hb0 & qh_mask_a) ? 16 : 0);
        int qa1 = (qb1 & 0x0F) + ((hb1 & qh_mask_a) ? 16 : 0);
        int qa2 = (qb2 & 0x0F) + ((hb2 & qh_mask_a) ? 16 : 0);
        int qa3 = (qb3 & 0x0F) + ((hb3 & qh_mask_a) ? 16 : 0);
        int qbq0 = (qb0 >>   4) + ((hb0 & qh_mask_b) ? 16 : 0);
        int qbq1 = (qb1 >>   4) + ((hb1 & qh_mask_b) ? 16 : 0);
        int qbq2 = (qb2 >>   4) + ((hb2 & qh_mask_b) ? 16 : 0);
        int qbq3 = (qb3 >>   4) + ((hb3 & qh_mask_b) ? 16 : 0);

        acc += (scale_a * (float)qa0  - min_a) * xa0;
        acc += (scale_a * (float)qa1  - min_a) * xa1;
        acc += (scale_a * (float)qa2  - min_a) * xa2;
        acc += (scale_a * (float)qa3  - min_a) * xa3;
        acc += (scale_b * (float)qbq0 - min_b) * xb0;
        acc += (scale_b * (float)qbq1 - min_b) * xb1;
        acc += (scale_b * (float)qbq2 - min_b) * xb2;
        acc += (scale_b * (float)qbq3 - min_b) * xb3;
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_xor_sync(0xffffffffu, acc, offset);
    }
    if (lane == 0) {
        // Output : y[token, slot, row] in flat [M, k_used, N] layout.
        // dst_lin = token * k_used + slot, so y[dst_lin * N + row].
        y[(long long)dst_lin * N + row] = (__nv_bfloat16)acc;
    }
}
"#;

#[cfg(feature = "cuda")]
const MUL_MM_ID_GEMM_Q5_K_BF16_SRC: &str = r#"
#include <cuda_bf16.h>
#include <cuda_fp16.h>

extern "C" __launch_bounds__(128, 8)
__global__ void mul_mm_id_gemm_q5_k_bf16(
    const unsigned long long* __restrict__ expert_ptrs,
    const int*                __restrict__ topk_indices, // [M, k_used]
    const __nv_bfloat16*      __restrict__ x,            // [M, K]
    __nv_bfloat16*            __restrict__ y,            // [M, k_used, N]
    int N,
    int K,
    int k_used
) {
    int token = blockIdx.z;
    int slot  = blockIdx.y;
    int e_idx = topk_indices[token * k_used + slot];
    const unsigned char* __restrict__ w_q5k =
        (const unsigned char* __restrict__)expert_ptrs[e_idx];

    int row0 = blockIdx.x * 4;
    int tid  = threadIdx.x;
    int row_in_block = tid >> 5;
    int lane         = tid & 31;
    int row          = row0 + row_in_block;
    if (row >= N) return;

    int blocks_per_row = K / 256;
    int row_offset     = row * blocks_per_row * 176;

    int group     = lane >> 3;
    int pos_base  = (lane & 7) << 2;
    int byte_base = (group << 5) + pos_base;
    unsigned int qh_mask_a = 1u << (2 * group);
    unsigned int qh_mask_b = 1u << (2 * group + 1);

    const __nv_bfloat16* x_tok = x + (long long)token * K;

    float acc = 0.0f;

    for (int b = 0; b < blocks_per_row; ++b) {
        int blk_off = row_offset + b * 176;
        const unsigned char* blk = w_q5k + blk_off;

        unsigned short d_bits    = blk[0] | (blk[1] << 8);
        unsigned short dmin_bits = blk[2] | (blk[3] << 8);
        float d    = __half2float(__ushort_as_half(d_bits));
        float dmin = __half2float(__ushort_as_half(dmin_bits));

        const unsigned char* scales = blk + 4;
        int sub_a = group * 2;
        int sub_b = sub_a + 1;
        unsigned char sc_a, m_a, sc_b, m_b;
        if (sub_a < 4) {
            sc_a = scales[sub_a]     & 0x3F;
            m_a  = scales[sub_a + 4] & 0x3F;
        } else {
            int ga = sub_a - 4;
            sc_a = (scales[ga + 8] & 0x0F) | ((scales[ga]     >> 6) << 4);
            m_a  = (scales[ga + 8] >> 4)   | ((scales[ga + 4] >> 6) << 4);
        }
        if (sub_b < 4) {
            sc_b = scales[sub_b]     & 0x3F;
            m_b  = scales[sub_b + 4] & 0x3F;
        } else {
            int gb = sub_b - 4;
            sc_b = (scales[gb + 8] & 0x0F) | ((scales[gb]     >> 6) << 4);
            m_b  = (scales[gb + 8] >> 4)   | ((scales[gb + 4] >> 6) << 4);
        }
        float scale_a = d    * (float)sc_a;
        float min_a   = dmin * (float)m_a;
        float scale_b = d    * (float)sc_b;
        float min_b   = dmin * (float)m_b;

        const unsigned char* qh = blk + 16;
        const unsigned char* ql = blk + 16 + 32;

        unsigned int qlbytes = *(const unsigned int*)(ql + byte_base);
        unsigned int qhbytes = *(const unsigned int*)(qh + pos_base);

        const __nv_bfloat16* xa_ptr = x_tok + b * 256 + sub_a * 32 + pos_base;
        const __nv_bfloat16* xb_ptr = x_tok + b * 256 + sub_b * 32 + pos_base;
        uint2 xa = *(const uint2*)xa_ptr;
        uint2 xb = *(const uint2*)xb_ptr;
        float xa0 = (float)__ushort_as_bfloat16((unsigned short)(xa.x & 0xFFFFu));
        float xa1 = (float)__ushort_as_bfloat16((unsigned short)(xa.x >> 16));
        float xa2 = (float)__ushort_as_bfloat16((unsigned short)(xa.y & 0xFFFFu));
        float xa3 = (float)__ushort_as_bfloat16((unsigned short)(xa.y >> 16));
        float xb0 = (float)__ushort_as_bfloat16((unsigned short)(xb.x & 0xFFFFu));
        float xb1 = (float)__ushort_as_bfloat16((unsigned short)(xb.x >> 16));
        float xb2 = (float)__ushort_as_bfloat16((unsigned short)(xb.y & 0xFFFFu));
        float xb3 = (float)__ushort_as_bfloat16((unsigned short)(xb.y >> 16));

        unsigned char qb0 = (qlbytes      ) & 0xFFu;
        unsigned char qb1 = (qlbytes >>  8) & 0xFFu;
        unsigned char qb2 = (qlbytes >> 16) & 0xFFu;
        unsigned char qb3 = (qlbytes >> 24) & 0xFFu;
        unsigned char hb0 = (qhbytes      ) & 0xFFu;
        unsigned char hb1 = (qhbytes >>  8) & 0xFFu;
        unsigned char hb2 = (qhbytes >> 16) & 0xFFu;
        unsigned char hb3 = (qhbytes >> 24) & 0xFFu;

        int qa0 = (qb0 & 0x0F) + ((hb0 & qh_mask_a) ? 16 : 0);
        int qa1 = (qb1 & 0x0F) + ((hb1 & qh_mask_a) ? 16 : 0);
        int qa2 = (qb2 & 0x0F) + ((hb2 & qh_mask_a) ? 16 : 0);
        int qa3 = (qb3 & 0x0F) + ((hb3 & qh_mask_a) ? 16 : 0);
        int qbq0 = (qb0 >>   4) + ((hb0 & qh_mask_b) ? 16 : 0);
        int qbq1 = (qb1 >>   4) + ((hb1 & qh_mask_b) ? 16 : 0);
        int qbq2 = (qb2 >>   4) + ((hb2 & qh_mask_b) ? 16 : 0);
        int qbq3 = (qb3 >>   4) + ((hb3 & qh_mask_b) ? 16 : 0);

        acc += (scale_a * (float)qa0  - min_a) * xa0;
        acc += (scale_a * (float)qa1  - min_a) * xa1;
        acc += (scale_a * (float)qa2  - min_a) * xa2;
        acc += (scale_a * (float)qa3  - min_a) * xa3;
        acc += (scale_b * (float)qbq0 - min_b) * xb0;
        acc += (scale_b * (float)qbq1 - min_b) * xb1;
        acc += (scale_b * (float)qbq2 - min_b) * xb2;
        acc += (scale_b * (float)qbq3 - min_b) * xb3;
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_xor_sync(0xffffffffu, acc, offset);
    }
    if (lane == 0) {
        y[((long long)token * k_used + slot) * N + row] = (__nv_bfloat16)acc;
    }
}
"#;

#[cfg(feature = "cuda")]
const MUL_MM_ID_GEMM_Q6_K_BF16_SRC: &str = r#"
#include <cuda_bf16.h>
#include <cuda_fp16.h>

extern "C" __launch_bounds__(128, 8)
__global__ void mul_mm_id_gemm_q6_k_bf16(
    const unsigned long long* __restrict__ expert_ptrs,
    const int*                __restrict__ topk_indices, // [M, k_used]
    const __nv_bfloat16*      __restrict__ x,            // [M, K]
    __nv_bfloat16*            __restrict__ y,            // [M, k_used, N]
    int N,
    int K,
    int k_used
) {
    int token = blockIdx.z;
    int slot  = blockIdx.y;
    int e_idx = topk_indices[token * k_used + slot];
    const unsigned char* __restrict__ w_q6k =
        (const unsigned char* __restrict__)expert_ptrs[e_idx];

    int row0 = blockIdx.x * 4;
    int tid  = threadIdx.x;
    int row_in_block = tid >> 5;
    int lane         = tid & 31;
    int row          = row0 + row_in_block;
    if (row >= N) return;

    int blocks_per_row = K / 256;
    int row_offset     = row * blocks_per_row * 210;
    int l16            = lane >> 4;

    const __nv_bfloat16* x_tok = x + (long long)token * K;

    float acc = 0.0f;

    for (int b = 0; b < blocks_per_row; ++b) {
        int blk_off = row_offset + b * 210;
        const unsigned char* blk = w_q6k + blk_off;

        unsigned short d_bits = blk[208] | (blk[209] << 8);
        float d = __half2float(__ushort_as_half(d_bits));

        const signed char* scales = (const signed char*)(blk + 192);
        float sc0_a = d * (float)scales[0 + l16];
        float sc2_a = d * (float)scales[2 + l16];
        float sc4_a = d * (float)scales[4 + l16];
        float sc6_a = d * (float)scales[6 + l16];
        float sc0_b = d * (float)scales[8 + l16];
        float sc2_b = d * (float)scales[10 + l16];
        float sc4_b = d * (float)scales[12 + l16];
        float sc6_b = d * (float)scales[14 + l16];

        const unsigned char* ql = blk;
        const unsigned char* qh = blk + 128;

        unsigned char ql_a0 = ql[0 + lane];
        unsigned char ql_b0 = ql[0 + lane + 32];
        unsigned char qh_0  = qh[0 + lane];
        int q0a = (ql_a0 & 0x0F) | (((qh_0)      & 0x03) << 4);
        int q1a = (ql_b0 & 0x0F) | (((qh_0 >> 2) & 0x03) << 4);
        int q2a = (ql_a0 >> 4)   | (((qh_0 >> 4) & 0x03) << 4);
        int q3a = (ql_b0 >> 4)   | (((qh_0 >> 6) & 0x03) << 4);

        const __nv_bfloat16* x_ptr_a = x_tok + b * 256 + 0;
        float x0a = (float)x_ptr_a[lane];
        float x1a = (float)x_ptr_a[lane + 32];
        float x2a = (float)x_ptr_a[lane + 64];
        float x3a = (float)x_ptr_a[lane + 96];

        acc += sc0_a * (float)(q0a - 32) * x0a;
        acc += sc2_a * (float)(q1a - 32) * x1a;
        acc += sc4_a * (float)(q2a - 32) * x2a;
        acc += sc6_a * (float)(q3a - 32) * x3a;

        unsigned char ql_a1 = ql[64 + lane];
        unsigned char ql_b1 = ql[64 + lane + 32];
        unsigned char qh_1  = qh[32 + lane];
        int q0b = (ql_a1 & 0x0F) | (((qh_1)      & 0x03) << 4);
        int q1b = (ql_b1 & 0x0F) | (((qh_1 >> 2) & 0x03) << 4);
        int q2b = (ql_a1 >> 4)   | (((qh_1 >> 4) & 0x03) << 4);
        int q3b = (ql_b1 >> 4)   | (((qh_1 >> 6) & 0x03) << 4);

        const __nv_bfloat16* x_ptr_b = x_tok + b * 256 + 128;
        float x0b = (float)x_ptr_b[lane];
        float x1b = (float)x_ptr_b[lane + 32];
        float x2b = (float)x_ptr_b[lane + 64];
        float x3b = (float)x_ptr_b[lane + 96];

        acc += sc0_b * (float)(q0b - 32) * x0b;
        acc += sc2_b * (float)(q1b - 32) * x1b;
        acc += sc4_b * (float)(q2b - 32) * x2b;
        acc += sc6_b * (float)(q3b - 32) * x3b;
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_xor_sync(0xffffffffu, acc, offset);
    }
    if (lane == 0) {
        y[((long long)token * k_used + slot) * N + row] = (__nv_bfloat16)acc;
    }
}
"#;

#[cfg(feature = "cuda")]
const MUL_MM_ID_GEMM_BF16_BF16_SRC: &str = r#"
#include <cuda_bf16.h>

extern "C" __global__ void mul_mm_id_gemm_bf16_bf16(
    const unsigned long long* __restrict__ expert_ptrs,
    const int*                __restrict__ topk_indices, // [M, k_used]
    const __nv_bfloat16*      __restrict__ x,            // [M, K]
    __nv_bfloat16*            __restrict__ y,            // [M, k_used, N]
    int N,
    int K,
    int k_used
) {
    int token = blockIdx.z;
    int slot  = blockIdx.y;
    int e_idx = topk_indices[token * k_used + slot];
    const __nv_bfloat16* __restrict__ w =
        (const __nv_bfloat16* __restrict__)expert_ptrs[e_idx];

    int row = blockIdx.x;
    if (row >= N) return;
    int tid = threadIdx.x;

    extern __shared__ float shmem[];

    const __nv_bfloat16* x_tok = x + (long long)token * K;

    float acc = 0.0f;
    int blocks_per_row = K / 256;
    int pos_base = tid * 4;
    int row_offset = row * K;

    for (int b = 0; b < blocks_per_row; ++b) {
        int k_off = b * 256 + pos_base;
        const __nv_bfloat16* w_ptr = w + row_offset + k_off;
        const __nv_bfloat16* x_ptr = x_tok + k_off;
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
        y[((long long)token * k_used + slot) * N + row] = (__nv_bfloat16)total;
    }
}
"#;

// T246.8 A4 — fused routed reduce :
//
//   y[i] += sum_{slot=0..K-1} alpha_dev[slot] * x[slot, i]
//
// Replaces the K-iteration `scaled_add_inplace_bf16_devscalar` epilogue
// loop in the routed-MoE down path. Equivalent (in float-precision
// accumulator) to the per-slot loop, modulo intra-row reduction order
// (sum across slots is performed in a single thread, low-to-high slot
// index — same as the per-slot host loop).
//
// Each thread accumulates over the slot axis in float, then writes back
// once per output element. K_MAX = 16 covers Qwen3-MoE (k=8) and any
// reasonable extension.
#[cfg(feature = "cuda")]
const SCALED_ADD_ROUTED_BF16_SRC: &str = r#"
#include <cuda_bf16.h>

extern "C" __global__ void scaled_add_routed_bf16(
    __nv_bfloat16*       __restrict__ y,            // [N]
    const __nv_bfloat16* __restrict__ x,            // [K, N] slot-major
    const __nv_bfloat16* __restrict__ alpha_dev,    // [K]
    int n,
    int k_used
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float yi = (float)y[i];
    for (int s = 0; s < k_used; ++s) {
        float a = (float)alpha_dev[s];
        float xi = (float)x[s * n + i];
        yi += a * xi;
    }
    y[i] = (__nv_bfloat16)yi;
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
    const __nv_bfloat16* __restrict__ k_cache,     // [max_seq, kv_dim] (P1.5 — Layout A)
    const __nv_bfloat16* __restrict__ v_cache,     // [max_seq, kv_dim] (P1.5 — Layout A)
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
    int kv_dim = n_kv * head_dim;

    extern __shared__ float sdata[];

    float q_i = (float)q[h * head_dim + tid];
    float m   = -1e30f;
    float l   = 0.0f;
    float o   = 0.0f;

    for (int t = t_start; t < t_end; ++t) {
        // Score = Q · K[t, kv_h]   (Layout A : k_cache[t, kv_h, i] = k_cache[t*kv_dim + kv_h*head_dim + i])
        long long kv_off = (long long)t * (long long)kv_dim
                         + (long long)kv_h * (long long)head_dim
                         + (long long)tid;
        float k_i     = (float)k_cache[kv_off];
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
        float v_i        = (float)v_cache[kv_off];
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

// T246.7 P1.3b — tree-attention partial kernel (FlashDecode-V2 split-K with
// per-token tree-mask). For each draft token in [0..tree_size), runs the
// same online-softmax FlashDecode pass but visible KV positions are :
//   (a) all base context : [0 .. *kv_len_dev)
//   (b) the token's ancestor chain in the draft tree (excluding root, which
//       is already covered by Phase A), mapped to slots [kv_len + anc - 1].
//
// Cache slot mapping : tree node `x` (BFS index) writes its K/V at slot
// `*pos_dev + x` (via `kv_append_tree_bf16`). With the existing convention
// `*kv_len_dev = *pos_dev + 1`, slot for node x is `*kv_len_dev + x - 1`.
// In particular slot for x=0 is `kv_len - 1` ∈ [0, kv_len), so Phase A
// already attends to the root's KV — root needs NO Phase B, making
// `tree_size=1, parent=[-1]` bit-equivalent to `gqa_decode_split_bf16`.
//
// For r > 0 : Phase B walks the parent chain from r up to root and attends
// to slot kv_len + anc - 1 for every ancestor anc with anc >= 1 (skipping
// the root anc=0 since it's already in Phase A). The chain includes r itself.
//
// Tree encoding :
//   parents : [tree_size] i32  — parent index in BFS order, root = -1.
//   depths  : [tree_size] u16  — depth of each node (root depth = 0). T246.10 A6 widened u8→u16 to support depth ≥ 256 for batched prefill.
//
// Grid : (n_q_heads, n_split, tree_size).  Block : (head_dim, 1, 1).
// Outputs (per-block) :
//   partial_m : [tree_size, n_q, n_split]            float
//   partial_l : [tree_size, n_q, n_split]            float
//   partial_o : [tree_size, n_q, n_split, head_dim]  bf16
#[cfg(feature = "cuda")]
const GQA_DECODE_TREE_PARTIAL_BF16_SRC: &str = r#"
#include <cuda_bf16.h>

extern "C" __global__ void gqa_decode_tree_partial_bf16(
    const __nv_bfloat16* __restrict__ q,           // [tree_size, n_heads, head_dim]
    const __nv_bfloat16* __restrict__ k_cache,     // [max_seq, kv_dim] = [max_seq, n_kv, head_dim] (P1.5 — Layout A, matches kv_append_*_devcnt)
    const __nv_bfloat16* __restrict__ v_cache,
    const int*           __restrict__ parents,     // [tree_size]
    const unsigned short* __restrict__ depths,     // [tree_size] (T246.10 A6 widened u8→u16 to support depth ≥ 256 for batched prefill)
    float*               __restrict__ partial_m,   // [tree_size, n_heads, n_split]
    float*               __restrict__ partial_l,   // [tree_size, n_heads, n_split]
    __nv_bfloat16*       __restrict__ partial_o,   // [tree_size, n_heads, n_split, head_dim]
    int n_heads,
    int n_kv,
    const int* __restrict__ kv_len_dev,
    int head_dim,
    int max_seq,
    int n_split,
    int tree_size,
    float scale
) {
    int h    = blockIdx.x;
    int sp   = blockIdx.y;
    int rnod = blockIdx.z;
    if (h >= n_heads || sp >= n_split || rnod >= tree_size) return;
    int tid = threadIdx.x;
    if (tid >= head_dim) return;

    int kv_len = *kv_len_dev;

    // Total visible positions = kv_len base + (depth(rnod) + 1) tree positions.
    // Tree positions are encoded virtually : we map them to cache slots
    // kv_len + ancestor_index. We split the `kv_len` base context across
    // n_split, and the (depth+1) tree positions are appended as a tail
    // walked entirely by split 0 (the small tail is cheap → no benefit
    // from splitting it further).
    int kv_per_split = (kv_len + n_split - 1) / n_split;
    int t_start      = sp * kv_per_split;
    int t_end        = t_start + kv_per_split;
    if (t_end > kv_len) t_end = kv_len;

    // partial slot indexing : (rnod, h, sp)
    long long slot = ((long long)rnod * (long long)n_heads + (long long)h) * (long long)n_split + (long long)sp;

    int kv_h = h * n_kv / n_heads;

    extern __shared__ float sdata[];

    // P1.5 — empty-split handling : when n_split > kv_len, splits beyond
    // the data range have nothing to do. Combine kernel reads partial_m /
    // partial_l / partial_o from the same buffer slot regardless ; we MUST
    // initialize them to the neutral element (-inf, 0, 0) or stale values
    // from the previous decode_step_tree call corrupt the combine result.
    // Phase B (tree positions) only runs in split 0, so non-zero splits
    // with t_start >= t_end have nothing to contribute.
    if (sp != 0 && t_start >= t_end) {
        if (tid == 0) {
            partial_m[slot] = -1e30f;
            partial_l[slot] = 0.0f;
        }
        partial_o[slot * (long long)head_dim + tid] = (__nv_bfloat16)0.0f;
        return;
    }

    float q_i = (float)q[((long long)rnod * (long long)n_heads + (long long)h) * (long long)head_dim + tid];
    float m   = -1e30f;
    float l   = 0.0f;
    float o   = 0.0f;

    // P1.5 — KV cache layout is [max_seq, kv_dim] (Layout A) where
    // kv_dim = n_kv * head_dim and a slot at position `t` for head `kv_h`
    // lives at offset `t * kv_dim + kv_h * head_dim + i`. This matches
    // `kv_append_bf16_devcnt` and `kv_append_tree_bf16` (both Layout A).
    int kv_dim = n_kv * head_dim;

    // ── Phase A : base context [t_start..t_end) ────────────────────────
    for (int t = t_start; t < t_end; ++t) {
        long long kv_off = (long long)t * (long long)kv_dim
                         + (long long)kv_h * (long long)head_dim
                         + (long long)tid;
        float k_i     = (float)k_cache[kv_off];
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

        float new_m      = fmaxf(m, s_t);
        float correction = expf(m - new_m);
        float p          = expf(s_t - new_m);
        float v_i        = (float)v_cache[kv_off];
        o = o * correction + p * v_i;
        l = l * correction + p;
        m = new_m;
    }

    // ── Phase B : ancestor tree positions, ONLY in split 0 ─────────────
    // Walk parent chain from root → rnod (causal order). Tree node x lives
    // at cache slot kv_len + x - 1. The root (x=0) maps to slot kv_len - 1,
    // already covered by Phase A → skip it. Max chain length = depths[rnod]+1.
    // Recompute the j-th ancestor on the fly via a reverse walk from rnod
    // (O(depth) per j → O(depth^2) total ; depth ≤ 32 → ≤ 1024 cheap ops,
    // dwarfed by the kv_len dot-product loop above).
    if (sp == 0) {
        int chain_len = (int)depths[rnod] + 1;
        // Iterate j from 1 (skip root at j=0) to chain_len-1 (rnod itself).
        for (int j = 1; j < chain_len; ++j) {
            // anc = the j-th ancestor of rnod in root-first order.
            int anc = rnod;
            for (int k = chain_len - 1; k > j; --k) {
                anc = parents[anc];
            }
            int t = kv_len + anc - 1;
            long long kv_off = (long long)t * (long long)kv_dim
                             + (long long)kv_h * (long long)head_dim
                             + (long long)tid;
            float k_i     = (float)k_cache[kv_off];
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

            float new_m      = fmaxf(m, s_t);
            float correction = expf(m - new_m);
            float p          = expf(s_t - new_m);
            float v_i        = (float)v_cache[kv_off];
            o = o * correction + p * v_i;
            l = l * correction + p;
            m = new_m;
        }
    }

    if (tid == 0) {
        partial_m[slot] = m;
        partial_l[slot] = l;
    }
    partial_o[slot * (long long)head_dim + tid] = (__nv_bfloat16)o;
}
"#;

// T246.7 P1.3b — tree-attention combine kernel : merges per-split partials
// for each (tree node, head) pair. Mirrors `gqa_decode_split_combine_bf16`
// with an extra outer loop over tree_size.
//   gridDim  = (n_heads, tree_size)
//   blockDim = (head_dim)
#[cfg(feature = "cuda")]
const GQA_DECODE_TREE_COMBINE_BF16_SRC: &str = r#"
#include <cuda_bf16.h>

extern "C" __global__ void gqa_decode_tree_combine_bf16(
    const float*         __restrict__ partial_m,    // [tree_size, n_heads, n_split]
    const float*         __restrict__ partial_l,    // [tree_size, n_heads, n_split]
    const __nv_bfloat16* __restrict__ partial_o,    // [tree_size, n_heads, n_split, head_dim]
    __nv_bfloat16*       __restrict__ out,           // [tree_size, n_heads, head_dim]
    int n_heads,
    int head_dim,
    int n_split,
    int tree_size
) {
    int h    = blockIdx.x;
    int rnod = blockIdx.y;
    if (h >= n_heads || rnod >= tree_size) return;
    int tid = threadIdx.x;
    if (tid >= head_dim) return;

    long long row_off = ((long long)rnod * (long long)n_heads + (long long)h) * (long long)n_split;

    extern __shared__ float gm_buf[];
    if (tid == 0) {
        float gm = -1e30f;
        for (int sp = 0; sp < n_split; ++sp) {
            float pm = partial_m[row_off + sp];
            if (pm > gm) gm = pm;
        }
        gm_buf[0] = gm;
    }
    __syncthreads();
    float global_m = gm_buf[0];

    float o_sum = 0.0f;
    float l_sum = 0.0f;
    for (int sp = 0; sp < n_split; ++sp) {
        long long slot = row_off + sp;
        float pm = partial_m[slot];
        float pl = partial_l[slot];
        float w  = expf(pm - global_m);
        float po = (float)partial_o[slot * (long long)head_dim + tid];
        o_sum += po * w;
        l_sum += pl * w;
    }

    long long out_off = ((long long)rnod * (long long)n_heads + (long long)h) * (long long)head_dim + tid;
    out[out_off] = (__nv_bfloat16)(o_sum / fmaxf(l_sum, 1e-12f));
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

// ─────────────────────────────────────────────────────────────────────────
// MMQ_WHOLESALE — 1:1 translation of llama.cpp Q4_K MMQ INT8 mma path
// (T246.10 / M-LLAMA-PARITY).
//
// Translates 5 llama.cpp source regions into a self-contained NVRTC module :
//   1. mma.cuh                         — mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 wrapper
//   2. quantize.cu:176-271             — quantize_mmq_q8_1<DS4> kernel (BF16 input)
//   3. vecdotq.cuh:530-555             — vec_dot_q4_K_q8_1_impl_mmq (dp4a fallback path)
//   4. mmq.cuh:2034-2141 (load_tiles_q4_K) — smem tile loader, MMQ_MMA_TILE_X_K_Q8_1 = 76
//   5. mmq.cuh:1271-1397 (vec_dot_q8_1_q8_1_mma) — INT8 m16n8k16 mma matmul
//
// Design decisions :
//   - Fixed instantiation : mmq_x=64, mmq_y=64, nwarps=4 (no template
//     explosion ; the prefill regime M ∈ [64..512] is the only one where we
//     gain over TrackG-lite BF16 mma). Smaller M falls back to TrackG-lite.
//   - x_q8_1 produced by a SEPARATE prepass kernel (`quantize_mmq_q8_1_bf16_ds4`)
//     using the packed `block_q8_1_mmq` layout (128 quants + 4 half2 = 144 B).
//   - mma C-fragment write-back applies the per-Q4_K-superblock (d, dmin) +
//     per-32-subblock (sc, m) scales in FP32 to produce final BF16 output.
//
// Output : Y[M, N] BF16 row-major   (M tokens × N output cols).
//
// Constraints :
//   - K % 256 == 0  (Q4_K super-block size)
//   - M % 1 OK, N % 1 OK (boundary checks at write)
//   - Compute capability >= 6.1 for dp4a path ; >= 8.0 for the INT8 mma
//     extension (asm wrapper kept in source for follow-up activation).
//
// References :
//   - llama.cpp mma.cuh:827-847 (s8 mma m16n8k16 PTX)
//   - llama.cpp mmq.cuh:2034-2141 (load_tiles_q4_K)
//   - llama.cpp mmq.cuh:1325-1396 (vec_dot_q8_1_q8_1_mma NVIDIA path)
//   - RFC note 3be23924-bf2f-449b-8de8-11fc4ff23011 (RESEARCH-mma)
//   - RFC note a0cd33f2-e738-41b9-9b37-42663a10ee92 (RESEARCH-Q8_1)

// Quantize BF16 activation row to packed Q8_1 MMQ layout (block_q8_1_mmq).
//
// Output layout per 128-element row segment (= 4 sub-blocks of 32 = 144 B total) :
//   bytes  0-15  : half2 ds4[4]  → ds4[s] = (d_s, sum_s) for sub-block s ∈ {0..3}
//   bytes 16-143 : int8 qs[128]  → 128 quantized values (4 × 32-elem sub-blocks)
//
// Launch params (matches llama.cpp quantize_mmq_q8_1<DS4>) :
//   grid = (M, ceil(K / 512), 1) ; block = (128, 1, 1)
//   each thread owns 4 consecutive BF16 elements ; per-32-elem sub-block
//   amax/sum reduction via warp shuffles over 8 lanes.
//
// Reference : llama.cpp quantize.cu:176-271 (with BF16 input cast vs F32).
#[cfg(feature = "cuda")]
const QUANTIZE_MMQ_Q8_1_BF16_DS4_SRC: &str = r#"
#include <cuda_bf16.h>
#include <cuda_fp16.h>

extern "C" __global__ void quantize_mmq_q8_1_bf16_ds4(
    const __nv_bfloat16* __restrict__ x,    // [M, K] BF16 row-major
    unsigned char*       __restrict__ y,    // [M, K/128] block_q8_1_mmq (144 B each)
    int M,
    int K)
{
    const int i0 = (blockIdx.y * blockDim.x + threadIdx.x) * 4;
    if (i0 >= K) return;

    const int m = blockIdx.x;

    const int ib  = i0 / 128;     // outer block_q8_1_mmq index in this row
    const int iqs = i0 % 128;     // element offset within the block (0..124, step 4)
    const int sub = iqs / 32;     // sub-block index 0..3

    // Load 4 BF16 → 4 FP32.
    const __nv_bfloat16* xp = x + (long long)m * K + i0;
    float x0 = (float)xp[0];
    float x1 = (float)xp[1];
    float x2 = (float)xp[2];
    float x3 = (float)xp[3];

    float amax = fmaxf(fmaxf(fabsf(x0), fabsf(x1)), fmaxf(fabsf(x2), fabsf(x3)));
    float sum  = x0 + x1 + x2 + x3;

    // Warp reduction over 8 lanes (= 32-elem sub-block, since 4 elems/thread).
    // DS4 layout : vals_per_scale = vals_per_sum = 32 → offset start = 4.
    #pragma unroll
    for (int off = 4; off > 0; off >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, off, 32));
        sum  = sum +      __shfl_xor_sync(0xFFFFFFFF, sum,  off, 32);
    }

    float d_inv = (amax == 0.0f) ? 0.0f : 127.0f / amax;
    int q0 = __float2int_rn(x0 * d_inv);
    int q1 = __float2int_rn(x1 * d_inv);
    int q2 = __float2int_rn(x2 * d_inv);
    int q3 = __float2int_rn(x3 * d_inv);
    if (q0 < -127) q0 = -127; if (q0 > 127) q0 = 127;
    if (q1 < -127) q1 = -127; if (q1 > 127) q1 = 127;
    if (q2 < -127) q2 = -127; if (q2 > 127) q2 = 127;
    if (q3 < -127) q3 = -127; if (q3 > 127) q3 = 127;

    unsigned char* y_blk = y + ((long long)m * (K / 128) + ib) * 144;

    // qs[128] starts at byte offset 16 (after 4 half2 ds4[4] scales).
    char4 q = make_char4((char)q0, (char)q1, (char)q2, (char)q3);
    char4* yqs4 = (char4*)(y_blk + 16);
    yqs4[iqs / 4] = q;

    // First thread of each 32-elem sub-block writes the (d, sum) half2.
    if ((iqs % 32) == 0) {
        float d = (amax == 0.0f) ? 0.0f : (amax / 127.0f);
        __half2* ds4 = (__half2*)y_blk;
        ds4[sub] = __floats2half2_rn(d, sum);
    }
}
"#;

// Q4_K × Q8_1 INT8-staged mma matmul — 1:1 port of llama.cpp's
// `mul_mat_q4_k_q8_1_mma` body for mmq_x=64, mmq_y=64, nwarps=4 instantiation.
//
// Block topology :
//   - grid = (ceil(N/64), ceil(M/64), 1)            ← (cols-of-W tile, cols-of-X tile)
//   - block = (32, 4, 1) = 128 threads (4 warps)
//
// Per-CTA work :
//   - Computes a 64(rows of W = "i" axis = N) × 64(cols of X = "j" axis = M tokens) tile.
//   - K iterates in chunks of MMQ_ITER_K = 256 (one Q4_K super-block).
//
// granularity (mmq_get_granularity_device for mmq_x=64) = 16
//   → rows_per_warp = 2 * 16 = 32
//   → ntx = 32 / 16 = 2 minitiles (16-row each) per warp
//
// Inner k01 loop : runs 8 sub-stripes of 32 K within each super-block,
// re-staging tile_y per stripe to bound smem at MMQ_X * MMQ_TILE_Y_K ints.
//
// Smem layout (static) :
//   tile_x : MMQ_Y * MMQ_MMA_TILE_X_K_Q8_1 ints = 64 * 76 = 19 456 B
//   tile_y : MMQ_X * MMQ_TILE_Y_K ints          = 64 * 36 =  9 216 B
//   Total                                                = 28 672 B (28 KB)
//
// FP arithmetic : the inner reduction uses __dp4a (1:1 port of
// vec_dot_q4_K_q8_1_impl_mmq from vecdotq.cuh:530-555). The mma.sync s8
// PTX wrapper is included in the source for follow-up Phase 5 activation
// once the per-lane fragment-layout has been validated against the s8
// ldmatrix.x4 stride semantics on sm_121.
//
// Reference : llama.cpp mmq.cuh:1325-1396 (vec_dot_q8_1_q8_1_mma NVIDIA)
//             llama.cpp mmq.cuh:2034-2141 (load_tiles_q4_K Turing+)
#[cfg(feature = "cuda")]
const MUL_MAT_Q4_K_Q8_1_MMA_SRC: &str = r#"
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#define MMQ_X       64
#define MMQ_Y       64
#define NWARPS      4
#define WARP_SIZE   32
#define MMQ_TILE_NE_K          32
#define MMQ_ITER_K             256
#define QI8_1                  8       // = QK8_1/4 = 32/4
#define QR4_K                  2
#define MMQ_MMA_TILE_X_K_Q8_1  76      // 2*32 + 2*32/8 + 4 = 76
#define MMQ_TILE_Y_K           36      // = 32 + 32/8

// Unpack the 12-byte Q4_K scale/min header into 8 (sc, m) uint8 pairs.
// Mirrors unpack_scales_q45_K (mmq.cuh:2024-2032) byte-wise.
__device__ __forceinline__ void unpack_q4k_scales(
    const unsigned char* sr, unsigned char sc[8], unsigned char m[8])
{
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        sc[i]     = sr[i]     & 0x3F;
        m[i]      = sr[i + 4] & 0x3F;
        sc[i + 4] = (sr[i + 8] & 0x0F) | ((sr[i]     >> 6) << 4);
        m[i  + 4] = (sr[i + 8] >>   4) | ((sr[i + 4] >> 6) << 4);
    }
}

// mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 wrapper.
// Direct port of llama.cpp mma.cuh:827-847 (Ampere+ path).
// D[4] is the INT32 accumulator (in-place), A is 2 ints (8 packed s8),
// B is 1 int (4 packed s8).
__device__ __forceinline__ void mma_s8_m16n8k16(
    int* D, unsigned int A0, unsigned int A1, unsigned int B0)
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 "
        "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};"
        : "+r"(D[0]), "+r"(D[1]), "+r"(D[2]), "+r"(D[3])
        : "r"(A0), "r"(A1), "r"(B0)
    );
}

extern "C" __global__ __launch_bounds__(128, 1)
void mul_mat_q4_k_q8_1_mma_kernel(
    const unsigned char* __restrict__ W,        // [N, K/256 * 144] Q4_K row-major
    const unsigned char* __restrict__ X_q8_1,   // [M, K/128 * 144] packed q8_1_mmq
    __nv_bfloat16*       __restrict__ Y,        // [M, N] BF16 row-major
    int M,
    int N,
    int K)
{
    const int n_base = blockIdx.x * MMQ_Y;     // first row of W in this CTA
    const int m_base = blockIdx.y * MMQ_X;     // first col of X in this CTA
    const int warp_id = threadIdx.y;           // 0..3
    const int lane    = threadIdx.x;           // 0..31
    const int tid     = warp_id * WARP_SIZE + lane;   // 0..127

    __shared__ int tile_x[MMQ_Y * MMQ_MMA_TILE_X_K_Q8_1];     // 19 456 B
    __shared__ int tile_y[MMQ_X * MMQ_TILE_Y_K];               // 9 216 B

    // 32 FP32 accumulators per thread (= MMQ_X * MMQ_Y / (NWARPS * WARP_SIZE)).
    float sum[32];
    #pragma unroll
    for (int s = 0; s < 32; ++s) sum[s] = 0.0f;

    // Each warp owns rows i ∈ [i0_warp, i0_warp + rows_per_warp).
    // ntx=2 i-minitiles per warp, each 16 rows.
    const int i0_warp = (warp_id / 2) * 32;

    const int blocks_per_row_W = K / 256;

    // K-loop : iterates over Q4_K super-blocks (one per MMQ_ITER_K=256 K).
    for (int kb = 0; kb < blocks_per_row_W; ++kb) {
        // ========= STAGE 1 — load_tiles_q4_K Q4_K nibble payload =========
        // Direct port of mmq.cuh:2049-2070 Turing+ path.
        {
            const int txi = lane;   // 0..31 (threads_per_row = MMQ_ITER_K/(4*QR4_K) = 32)
            #pragma unroll
            for (int i0 = 0; i0 < MMQ_Y; i0 += NWARPS) {
                int i = i0 + warp_id;
                int gn = n_base + i;
                gn = (gn < N) ? gn : (N - 1);
                const unsigned char* bxi =
                    W + ((long long)gn * blocks_per_row_W + kb) * 144;
                const int qs0 = ((const int*)(bxi + 16))[txi];
                tile_x[i * MMQ_MMA_TILE_X_K_Q8_1 + 16*(txi/8) + (txi%8) + 0] = (qs0 >> 0) & 0x0F0F0F0F;
                tile_x[i * MMQ_MMA_TILE_X_K_Q8_1 + 16*(txi/8) + (txi%8) + 8] = (qs0 >> 4) & 0x0F0F0F0F;
            }
        }

        // ========= STAGE 2 — load (d, -dmin) * (sc, m) into x_dm =========
        // Direct port of mmq.cuh:2072-2107 Turing+ path.
        // Layout : x_dm[i * 76 + sizeof(int)*ksc + l] for ksc ∈ {0,1}, l ∈ {0..3}.
        // We map (lane / 2) → row offset (rows_per_warp_local = 16),
        // (lane & 1) → ksc.
        {
            const int rows_per_warp_local = WARP_SIZE / 2;
            #pragma unroll
            for (int i0 = 0; i0 < MMQ_Y; i0 += NWARPS * rows_per_warp_local) {
                int i = i0 + warp_id * rows_per_warp_local + lane / 2;
                if (i < MMQ_Y) {
                    int gn = n_base + i;
                    gn = (gn < N) ? gn : (N - 1);
                    const unsigned char* bxi =
                        W + ((long long)gn * blocks_per_row_W + kb) * 144;

                    float d    = __half2float(*(const __half*)(bxi + 0));
                    float dmin = __half2float(*(const __half*)(bxi + 2));

                    const unsigned char* sr = bxi + 4;
                    unsigned char sc8[8], m8[8];
                    unpack_q4k_scales(sr, sc8, m8);

                    int ksc = lane & 1;
                    __half2* x_dm = (__half2*)(tile_x + i * MMQ_MMA_TILE_X_K_Q8_1 + 2*MMQ_TILE_NE_K);
                    #pragma unroll
                    for (int l = 0; l < 4; ++l) {
                        float ds  = d    * (float)sc8[ksc*4 + l];
                        float dmm = -dmin * (float)m8 [ksc*4 + l];
                        x_dm[ksc*4 + l] = __floats2half2_rn(ds, dmm);
                    }
                }
            }
        }

        // ========= STAGE 3+4 — stream 8 K-stripes of 32 (= 1 super-block) =========
        // For each kc ∈ {0..7} : re-stage 32 K of X_q8_1 into tile_y, then
        // run the inner dp4a accumulation for this stripe across all (i, j)
        // fragment positions assigned to this warp.

        #pragma unroll
        for (int kc = 0; kc < 8; ++kc) {
            const int packed_idx_in_token = kb * 2 + (kc / 4);
            const int sub_in_packed       = kc % 4;

            // Stage Q8_1 stripe into tile_y. 128 threads × 4 outer iters cover
            // 64 tokens × 32 K = 2048 int8 = 512 ints + 64 half2 ds entries.
            #pragma unroll
            for (int p = 0; p < 4; ++p) {
                const int e = tid + p * 128;
                const int j_local = e / 8;
                const int k_int   = e % 8;
                const int gm = m_base + j_local;
                const unsigned char* x_ptr = X_q8_1
                    + ((long long)((gm < M) ? gm : (M - 1)) * (K / 128) + packed_idx_in_token) * 144;
                const int qs_int = ((const int*)(x_ptr + 16 + sub_in_packed * 32))[k_int];

                tile_y[j_local * MMQ_TILE_Y_K + 4 + k_int] = qs_int;

                if (k_int == 0) {
                    const __half2* ds_src = (const __half2*)x_ptr;
                    __half2* tile_y_ds = (__half2*)(tile_y + j_local * MMQ_TILE_Y_K);
                    tile_y_ds[0] = ds_src[sub_in_packed];
                }
            }

            __syncthreads();

            // ===== Inner accumulation : dp4a Q4_K × Q8_1 (1:1 port of
            //       vecdotq.cuh:530-555 vec_dot_q4_K_q8_1_impl_mmq, but
            //       consuming already-unpacked s8 tile_x ints).
            //
            // Per i-minitile n ∈ {0, 1} and j0 step (j0 ∈ {0,16,32,48}) :
            //   - 4 fragment elements l ∈ {0..3} per (n, j0).
            //   - Each element maps to (i_off, j_off) within the 16×8 tile_C.
            //   - Run __dp4a over the 8 ints of the 32-K stripe.
            //
            // x_dm sub-block index = kc (matches the 1:1 mapping from STAGE-2's
            // layout: x_dm[i*76 + 64 + kc] holds (d*sc[kc], -dmin*m[kc])).
            #pragma unroll
            for (int n = 0; n < 2; ++n) {
                const int i_minitile_base = i0_warp + n * 16;

                #pragma unroll
                for (int l = 0; l < 4; ++l) {
                    const int i_off = ((l >> 1) << 3) + (lane >> 2);   // 0..15
                    const int j_off = ((lane & 3) << 1) + (l & 1);      // 0..7
                    const int gi    = i_minitile_base + i_off;
                    if (gi >= MMQ_Y) continue;

                    const __half2* x_dm_row = (const __half2*)(tile_x
                        + gi * MMQ_MMA_TILE_X_K_Q8_1 + 2*MMQ_TILE_NE_K);
                    float2 dmA = __half22float2(x_dm_row[kc]);

                    // W stripe int offset for this (kc, gi) :
                    //   tile_x[gi, w_int_base..w_int_base+7] = 8 ints of s8 weights
                    //   (low or high nibble bytes per kc%2 selector).
                    const int w_int_base = (kc / 2) * 16 + (kc % 2) * 8;

                    #pragma unroll
                    for (int j0 = 0; j0 < MMQ_X; j0 += 16) {
                        const int j_warp_base = j0 + (warp_id & 1) * 8;
                        const int jg = j_warp_base + j_off;
                        if (jg >= MMQ_X) continue;

                        const __half2* y_ds_row = (const __half2*)(tile_y + jg * MMQ_TILE_Y_K);
                        float2 dsB = __half22float2(y_ds_row[0]);

                        int sumi_d = 0;
                        #pragma unroll
                        for (int k_int = 0; k_int < 8; ++k_int) {
                            int v = tile_x[gi * MMQ_MMA_TILE_X_K_Q8_1 + w_int_base + k_int];
                            int u = tile_y[jg * MMQ_TILE_Y_K + 4 + k_int];
                            sumi_d = __dp4a(v, u, sumi_d);
                        }

                        // Canonical Q4_K × Q8_1 final reduction (mmq.cuh:1390-1391) :
                        //   sum += dmA.x * dsB.x * dot     ← (d·sc) · d_y · Σ q4·q8
                        //   sum += dmA.y * dsB.y           ← (-dmin·m) · sum_y
                        sum[(j0 / 8 + n) * 4 + l] += dmA.x * dsB.x * (float)sumi_d;
                        sum[(j0 / 8 + n) * 4 + l] += dmA.y * dsB.y;
                    }
                }
            }

            __syncthreads();
        } // kc 32-K stripe loop
    } // kb super-block loop

    // ========= STAGE 5 — write back Y[M, N] =========
    #pragma unroll
    for (int n = 0; n < 2; ++n) {
        const int i_minitile_base = i0_warp + n * 16;
        #pragma unroll
        for (int j0 = 0; j0 < MMQ_X; j0 += 16) {
            const int j_warp_base = j0 + (warp_id & 1) * 8;
            #pragma unroll
            for (int l = 0; l < 4; ++l) {
                const int i_off = ((l >> 1) << 3) + (lane >> 2);
                const int j_off = ((lane & 3) << 1) + (l & 1);
                const int gi = n_base + i_minitile_base + i_off;
                const int gj = m_base + j_warp_base + j_off;
                if (gi < N && gj < M) {
                    float v = sum[(j0 / 8 + n) * 4 + l];
                    Y[(long long)gj * N + gi] = __float2bfloat16(v);
                }
            }
        }
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
    sgemv_q4k_v3: std::sync::OnceLock<(Arc<CudaModule>, CudaFunction)>,
    sgemm_q4k_m8: std::sync::OnceLock<(Arc<CudaModule>, CudaFunction)>,
    /// T246.10 A6b.1 — Q4_K matmul with arbitrary batch M (1..MAX_TREE_SIZE).
    sgemm_q4k_mvar: std::sync::OnceLock<(Arc<CudaModule>, CudaFunction)>,
    sgemv_q5k: std::sync::OnceLock<(Arc<CudaModule>, CudaFunction)>,
    sgemv_q5k_v3: std::sync::OnceLock<(Arc<CudaModule>, CudaFunction)>,
    sgemm_q5k_m8: std::sync::OnceLock<(Arc<CudaModule>, CudaFunction)>,
    /// T246.10 A6b.1 — Q5_K matmul with arbitrary batch M.
    sgemm_q5k_mvar: std::sync::OnceLock<(Arc<CudaModule>, CudaFunction)>,
    sgemv_q6k: std::sync::OnceLock<(Arc<CudaModule>, CudaFunction)>,
    sgemv_q6k_v2: std::sync::OnceLock<(Arc<CudaModule>, CudaFunction)>,
    sgemv_q6k_v3: std::sync::OnceLock<(Arc<CudaModule>, CudaFunction)>,
    sgemv_q6k_v4: std::sync::OnceLock<(Arc<CudaModule>, CudaFunction)>,
    sgemm_q6k_m8: std::sync::OnceLock<(Arc<CudaModule>, CudaFunction)>,
    /// T246.10 A6b.1 — Q6_K matmul with arbitrary batch M.
    sgemm_q6k_mvar: std::sync::OnceLock<(Arc<CudaModule>, CudaFunction)>,
    /// T246.10 A6b.1 — pure-BF16 matmul with arbitrary batch M.
    sgemm_bf16_mvar: std::sync::OnceLock<(Arc<CudaModule>, CudaFunction)>,
    /// T246.10 TrackG-lite — BF16×BF16 GEMM via mma.sync.aligned.m16n8k16.
    gemm_bf16_mma: std::sync::OnceLock<(Arc<CudaModule>, CudaFunction)>,
    sgemv_bf16: std::sync::OnceLock<(Arc<CudaModule>, CudaFunction)>,
    /// T246.8 A3 — sgemv_bf16_bf16_v2 mma.sync m16n8k16 tensor-core SGEMV.
    sgemv_bf16_v2: std::sync::OnceLock<(Arc<CudaModule>, CudaFunction)>,
    softplus_inplace: std::sync::OnceLock<(Arc<CudaModule>, CudaFunction)>,
    sigmoid_inplace: std::sync::OnceLock<(Arc<CudaModule>, CudaFunction)>,
    mul_inplace: std::sync::OnceLock<(Arc<CudaModule>, CudaFunction)>,
    repeat_heads: std::sync::OnceLock<(Arc<CudaModule>, CudaFunction)>,
    split_qg: std::sync::OnceLock<(Arc<CudaModule>, CudaFunction)>,
    rope_partial: std::sync::OnceLock<(Arc<CudaModule>, CudaFunction)>,
    conv1d_depthwise: std::sync::OnceLock<(Arc<CudaModule>, CudaFunction)>,
    l2_norm_per_head: std::sync::OnceLock<(Arc<CudaModule>, CudaFunction)>,
    delta_net_step: std::sync::OnceLock<(Arc<CudaModule>, CudaFunction)>,
    // T246.7 TrackC.1 — tree-aware DeltaNet step for SSM-hybrid Lookahead
    delta_net_step_tree: std::sync::OnceLock<(Arc<CudaModule>, CudaFunction)>,
    // T246.10 TrackF — column-parallel optimized variant of delta_net_step_tree
    delta_net_step_tree_opt: std::sync::OnceLock<(Arc<CudaModule>, CudaFunction)>,
    // NEW-SSM — bandwidth-saturated variant of delta_net_step_tree (per-warp
    // row ownership + coalesced state RMW, no block-wide sync). Gate
    // `RUSTORCH_DELTA_NET_NEW=1`.
    delta_net_step_tree_v2: std::sync::OnceLock<(Arc<CudaModule>, CudaFunction)>,
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
    // T247.1 — RMSNorm backward kernel (training-ready pilot)
    rms_norm_grad: std::sync::OnceLock<(Arc<CudaModule>, CudaFunction)>,
    // T247.2 — SwiGLU backward kernel
    swiglu_grad: std::sync::OnceLock<(Arc<CudaModule>, CudaFunction)>,
    // T247.3 — RoPE partial backward kernel
    rope_partial_grad: std::sync::OnceLock<(Arc<CudaModule>, CudaFunction)>,
    // T247.4 — fused cross-entropy loss + gradient
    cross_entropy_loss_grad: std::sync::OnceLock<(Arc<CudaModule>, CudaFunction)>,
    // T247.5 — embedding lookup backward (sparse scatter)
    embedding_lookup_grad: std::sync::OnceLock<(Arc<CudaModule>, CudaFunction)>,
    // T247.6 — Q4_K SGEMV backward dx (frozen W)
    sgemv_q4k_grad_dx: std::sync::OnceLock<(Arc<CudaModule>, CudaFunction)>,
    // T247.7 — GQA decode backward (Flash-Attention style)
    gqa_decode_grad: std::sync::OnceLock<(Arc<CudaModule>, CudaFunction)>,
    // T246.6.2 — Top-K softmax router for MoE FFN
    topk_softmax: std::sync::OnceLock<(Arc<CudaModule>, CudaFunction)>,
    // T246.6.3 — scaled add-in-place (used by MoE expert weighted accumulation)
    scaled_add_inplace: std::sync::OnceLock<(Arc<CudaModule>, CudaFunction)>,
    // T246.7 P1.3 — Lookahead Decoding tree-attention kernels
    kv_append_tree: std::sync::OnceLock<(Arc<CudaModule>, CudaFunction)>,
    gqa_decode_tree_partial: std::sync::OnceLock<(Arc<CudaModule>, CudaFunction)>,
    gqa_decode_tree_combine: std::sync::OnceLock<(Arc<CudaModule>, CudaFunction)>,
    argmax_logits_tree: std::sync::OnceLock<(Arc<CudaModule>, CudaFunction)>,
    add_u32_dev: std::sync::OnceLock<(Arc<CudaModule>, CudaFunction)>,
    set_u32_dev: std::sync::OnceLock<(Arc<CudaModule>, CudaFunction)>,
    // T246.8 A1 — fused SSM block element-wise mega-kernels
    ssm_pre_step: std::sync::OnceLock<(Arc<CudaModule>, CudaFunction)>,
    ssm_post_step: std::sync::OnceLock<(Arc<CudaModule>, CudaFunction)>,
    // T246.8 A2 — indexed MoE FFN dispatch kernels (device-side expert lookup)
    sgemv_q4k_v3_indexed: std::sync::OnceLock<(Arc<CudaModule>, CudaFunction)>,
    sgemv_q4k_q8_1_dp4a_indexed: std::sync::OnceLock<(Arc<CudaModule>, CudaFunction)>,
    sgemv_q5k_v3_indexed: std::sync::OnceLock<(Arc<CudaModule>, CudaFunction)>,
    sgemv_q6k_v3_indexed: std::sync::OnceLock<(Arc<CudaModule>, CudaFunction)>,
    sgemv_bf16_indexed: std::sync::OnceLock<(Arc<CudaModule>, CudaFunction)>,
    scaled_add_inplace_devscalar: std::sync::OnceLock<(Arc<CudaModule>, CudaFunction)>,
    scaled_add_sigmoid_devscalar: std::sync::OnceLock<(Arc<CudaModule>, CudaFunction)>,
    zero_bf16: std::sync::OnceLock<(Arc<CudaModule>, CudaFunction)>,
    // T246.8 A4 — mul_mm_id mega-kernels (one launch per gate/up/down matmul)
    mul_mm_id_q4_k_bf16: std::sync::OnceLock<(Arc<CudaModule>, CudaFunction)>,
    mul_mm_id_q4_k_q8_1_dp4a_bf16: std::sync::OnceLock<(Arc<CudaModule>, CudaFunction)>,
    mul_mm_id_q5_k_bf16: std::sync::OnceLock<(Arc<CudaModule>, CudaFunction)>,
    mul_mm_id_q6_k_bf16: std::sync::OnceLock<(Arc<CudaModule>, CudaFunction)>,
    mul_mm_id_bf16_bf16: std::sync::OnceLock<(Arc<CudaModule>, CudaFunction)>,
    // T246.10 TrackE.2 — Group-GEMM (M-variable) extensions.
    mul_mm_id_gemm_q4_k_bf16: std::sync::OnceLock<(Arc<CudaModule>, CudaFunction)>,
    mul_mm_id_gemm_q4_k_q8_1_dp4a_bf16: std::sync::OnceLock<(Arc<CudaModule>, CudaFunction)>,
    mul_mm_id_gemm_q5_k_bf16: std::sync::OnceLock<(Arc<CudaModule>, CudaFunction)>,
    mul_mm_id_gemm_q6_k_bf16: std::sync::OnceLock<(Arc<CudaModule>, CudaFunction)>,
    mul_mm_id_gemm_bf16_bf16: std::sync::OnceLock<(Arc<CudaModule>, CudaFunction)>,
    // T246.10 TrackE.3 — sort-permutation Group-GEMM (cache reuse).
    mm_ids_helper_bf16: std::sync::OnceLock<(Arc<CudaModule>, CudaFunction)>,
    mul_mm_id_gemm_q4_k_sorted_bf16: std::sync::OnceLock<(Arc<CudaModule>, CudaFunction)>,
    // T246.10 TrackE.4 — Q5_K sort-permutation Group-GEMM (cache reuse).
    mul_mm_id_gemm_q5_k_sorted_bf16: std::sync::OnceLock<(Arc<CudaModule>, CudaFunction)>,
    scaled_add_routed_bf16: std::sync::OnceLock<(Arc<CudaModule>, CudaFunction)>,
    // T246.9 NVFP4.2 — indexed NVFP4 SGEMV for MoE FFN forward (single-token decode)
    sgemv_nvfp4_bf16_indexed: std::sync::OnceLock<(Arc<CudaModule>, CudaFunction)>,
    // T246.9 NVFP4.2 — non-indexed NVFP4 SGEMV (single-Linear decode, no expert dispatch)
    sgemv_nvfp4_bf16: std::sync::OnceLock<(Arc<CudaModule>, CudaFunction)>,
    // T246.8 A5 — split-K Q6_K SGEMV for lm_head ceiling.
    sgemv_q6k_split_k_partial: std::sync::OnceLock<(Arc<CudaModule>, CudaFunction)>,
    reduce_split_k_bf16: std::sync::OnceLock<(Arc<CudaModule>, CudaFunction)>,
    // T246.10 MMQ-WHOLESALE — Q4_K × Q8_1 packed mma-staged kernel.
    quantize_mmq_q8_1_ds4: std::sync::OnceLock<(Arc<CudaModule>, CudaFunction)>,
    mul_mat_q4_k_q8_1_mma: std::sync::OnceLock<(Arc<CudaModule>, CudaFunction)>,
    // T246.10 Q4K-MOE-CUBLAS — Q4_K -> BF16 device-side dequant.
    dequant_q4_k_to_bf16: std::sync::OnceLock<(Arc<CudaModule>, CudaFunction)>,
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
            sgemv_q4k_v3: std::sync::OnceLock::new(),
            sgemm_q4k_m8: std::sync::OnceLock::new(),
            sgemm_q4k_mvar: std::sync::OnceLock::new(),
            sgemv_q5k: std::sync::OnceLock::new(),
            sgemv_q5k_v3: std::sync::OnceLock::new(),
            sgemm_q5k_m8: std::sync::OnceLock::new(),
            sgemm_q5k_mvar: std::sync::OnceLock::new(),
            sgemv_q6k: std::sync::OnceLock::new(),
            sgemv_q6k_v2: std::sync::OnceLock::new(),
            sgemv_q6k_v3: std::sync::OnceLock::new(),
            sgemv_q6k_v4: std::sync::OnceLock::new(),
            sgemm_q6k_m8: std::sync::OnceLock::new(),
            sgemm_q6k_mvar: std::sync::OnceLock::new(),
            sgemm_bf16_mvar: std::sync::OnceLock::new(),
            gemm_bf16_mma: std::sync::OnceLock::new(),
            sgemv_bf16: std::sync::OnceLock::new(),
            sgemv_bf16_v2: std::sync::OnceLock::new(),
            softplus_inplace: std::sync::OnceLock::new(),
            sigmoid_inplace: std::sync::OnceLock::new(),
            mul_inplace: std::sync::OnceLock::new(),
            repeat_heads: std::sync::OnceLock::new(),
            split_qg: std::sync::OnceLock::new(),
            rope_partial: std::sync::OnceLock::new(),
            conv1d_depthwise: std::sync::OnceLock::new(),
            l2_norm_per_head: std::sync::OnceLock::new(),
            delta_net_step: std::sync::OnceLock::new(),
            delta_net_step_tree: std::sync::OnceLock::new(),
            delta_net_step_tree_opt: std::sync::OnceLock::new(),
            delta_net_step_tree_v2: std::sync::OnceLock::new(),
            rope_partial_devcnt: std::sync::OnceLock::new(),
            gqa_decode_online_devcnt: std::sync::OnceLock::new(),
            increment_u32_dev: std::sync::OnceLock::new(),
            kv_append_devcnt: std::sync::OnceLock::new(),
            quantize_q8_1: std::sync::OnceLock::new(),
            sgemv_q4k_q8_1_dp4a: std::sync::OnceLock::new(),
            gqa_split_partial: std::sync::OnceLock::new(),
            gqa_split_combine: std::sync::OnceLock::new(),
            rms_norm_grad: std::sync::OnceLock::new(),
            swiglu_grad: std::sync::OnceLock::new(),
            rope_partial_grad: std::sync::OnceLock::new(),
            cross_entropy_loss_grad: std::sync::OnceLock::new(),
            embedding_lookup_grad: std::sync::OnceLock::new(),
            sgemv_q4k_grad_dx: std::sync::OnceLock::new(),
            gqa_decode_grad: std::sync::OnceLock::new(),
            topk_softmax: std::sync::OnceLock::new(),
            scaled_add_inplace: std::sync::OnceLock::new(),
            // T246.7 P1.3 — Lookahead tree kernels
            kv_append_tree: std::sync::OnceLock::new(),
            gqa_decode_tree_partial: std::sync::OnceLock::new(),
            gqa_decode_tree_combine: std::sync::OnceLock::new(),
            argmax_logits_tree: std::sync::OnceLock::new(),
            add_u32_dev: std::sync::OnceLock::new(),
            set_u32_dev: std::sync::OnceLock::new(),
            // T246.8 A1 — fused SSM mega-kernels
            ssm_pre_step: std::sync::OnceLock::new(),
            ssm_post_step: std::sync::OnceLock::new(),
            // T246.8 A2 — indexed MoE FFN dispatch kernels
            sgemv_q4k_v3_indexed: std::sync::OnceLock::new(),
            sgemv_q4k_q8_1_dp4a_indexed: std::sync::OnceLock::new(),
            sgemv_q5k_v3_indexed: std::sync::OnceLock::new(),
            sgemv_q6k_v3_indexed: std::sync::OnceLock::new(),
            sgemv_bf16_indexed: std::sync::OnceLock::new(),
            scaled_add_inplace_devscalar: std::sync::OnceLock::new(),
            scaled_add_sigmoid_devscalar: std::sync::OnceLock::new(),
            zero_bf16: std::sync::OnceLock::new(),
            // T246.8 A4 — mul_mm_id mega-kernels
            mul_mm_id_q4_k_bf16: std::sync::OnceLock::new(),
            mul_mm_id_q4_k_q8_1_dp4a_bf16: std::sync::OnceLock::new(),
            mul_mm_id_q5_k_bf16: std::sync::OnceLock::new(),
            mul_mm_id_q6_k_bf16: std::sync::OnceLock::new(),
            mul_mm_id_bf16_bf16: std::sync::OnceLock::new(),
            mul_mm_id_gemm_q4_k_bf16: std::sync::OnceLock::new(),
            mul_mm_id_gemm_q4_k_q8_1_dp4a_bf16: std::sync::OnceLock::new(),
            mul_mm_id_gemm_q5_k_bf16: std::sync::OnceLock::new(),
            mul_mm_id_gemm_q6_k_bf16: std::sync::OnceLock::new(),
            mul_mm_id_gemm_bf16_bf16: std::sync::OnceLock::new(),
            // T246.10 TrackE.3 — sort-permutation Group-GEMM (cache reuse).
            mm_ids_helper_bf16: std::sync::OnceLock::new(),
            mul_mm_id_gemm_q4_k_sorted_bf16: std::sync::OnceLock::new(),
            // T246.10 TrackE.4 — Q5_K sort-permutation Group-GEMM (cache reuse).
            mul_mm_id_gemm_q5_k_sorted_bf16: std::sync::OnceLock::new(),
            scaled_add_routed_bf16: std::sync::OnceLock::new(),
            // T246.9 NVFP4.2 — indexed/non-indexed NVFP4 SGEMV
            sgemv_nvfp4_bf16_indexed: std::sync::OnceLock::new(),
            sgemv_nvfp4_bf16: std::sync::OnceLock::new(),
            // T246.8 A5 — split-K Q6_K SGEMV for lm_head ceiling.
            sgemv_q6k_split_k_partial: std::sync::OnceLock::new(),
            reduce_split_k_bf16: std::sync::OnceLock::new(),
            // T246.10 MMQ-WHOLESALE.
            quantize_mmq_q8_1_ds4: std::sync::OnceLock::new(),
            mul_mat_q4_k_q8_1_mma: std::sync::OnceLock::new(),
            // T246.10 Q4K-MOE-CUBLAS.
            dequant_q4_k_to_bf16: std::sync::OnceLock::new(),
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

    /// T246.6.7 — Q4_K SGEMV V3 : 4 rows per block + per-thread scale +
    /// `__launch_bounds__`. Targets ~60-70% of LPDDR5X peak BW (vs V2's
    /// ~38%) on FFN-shape matmuls.
    ///
    /// # Safety  Same contract as `sgemv_q4k_bf16_v2`. `N` must be ≥ 1 ;
    /// rows beyond N within the last 4-row block are masked off.
    pub unsafe fn sgemv_q4k_bf16_v3(
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
                msg: format!("sgemv_q4k_bf16_v3: K={k} must be multiple of 256"),
            });
        }
        let (_module, func) = self.compile_or_get(
            &self.sgemv_q4k_v3,
            SGEMV_Q4K_BF16_V3_SRC,
            "sgemv_q4k_bf16_v3",
        )?;
        const ROWS_PER_BLOCK: i32 = 4;
        let n_blocks = ((n + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK) as u32;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (n_blocks, 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0, // x_shared declared but unused
        };
        let mut launcher = stream.launch_builder(&func);
        launcher.arg(&w_q4k).arg(&x).arg(&y).arg(&n).arg(&k);
        launcher.launch(cfg).map_err(|e| CudaError::Driver {
            code: format!("{e:?}").len() as i32,
            location: "sgemv_q4k_bf16_v3::launch",
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

    /// T246.10 A6b.1 — Q4_K matmul with arbitrary batch M (1..MAX_TREE_SIZE).
    /// Generalises `sgemm_q4k_bf16_m8` to M-variable. The kernel reads each
    /// W super-block once per block and accumulates against `ceil(M/8)`
    /// m-tiles of up to 8 m-rows.
    ///
    /// Inputs :
    ///   w_q4k : [N, K] Q4_K row-major
    ///   x     : [M, K] BF16 row-major
    ///   y     : [M, N] BF16 row-major (output)
    ///
    /// # Safety
    /// Caller ensures pointers valid and K multiple of 256.
    pub unsafe fn sgemm_q4k_bf16_mvar(
        &self,
        stream: &Arc<CudaStream>,
        w_q4k: u64,
        x: u64,
        y: u64,
        m: i32,
        n: i32,
        k: i32,
    ) -> Result<(), CudaError> {
        if k % 256 != 0 {
            return Err(CudaError::Unsupported {
                msg: format!("sgemm_q4k_bf16_mvar: K={k} must be multiple of 256"),
            });
        }
        if m <= 0 || n <= 0 {
            return Err(CudaError::Unsupported {
                msg: format!("sgemm_q4k_bf16_mvar: M={m} N={n} must be positive"),
            });
        }
        let (_module, func) = self.compile_or_get(
            &self.sgemm_q4k_mvar,
            SGEMM_Q4K_BF16_MVAR_SRC,
            "sgemm_q4k_bf16_mvar",
        )?;
        let n_mtiles = ((m as u32) + 7) >> 3;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (n as u32, n_mtiles, 1),
            block_dim: (64, 1, 1),
            shared_mem_bytes: 32 * 4,
        };
        let mut launcher = stream.launch_builder(&func);
        launcher.arg(&w_q4k).arg(&x).arg(&y).arg(&m).arg(&n).arg(&k);
        launcher.launch(cfg).map_err(|e| CudaError::Driver {
            code: format!("{e:?}").len() as i32,
            location: "sgemm_q4k_bf16_mvar::launch",
        })?;
        Ok(())
    }

    /// T246.10 A6b.1 — Q5_K matmul with arbitrary batch M.
    ///
    /// # Safety
    /// Same contract as `sgemm_q4k_bf16_mvar`.
    pub unsafe fn sgemm_q5k_bf16_mvar(
        &self,
        stream: &Arc<CudaStream>,
        w_q5k: u64,
        x: u64,
        y: u64,
        m: i32,
        n: i32,
        k: i32,
    ) -> Result<(), CudaError> {
        if k % 256 != 0 {
            return Err(CudaError::Unsupported {
                msg: format!("sgemm_q5k_bf16_mvar: K={k} must be multiple of 256"),
            });
        }
        if m <= 0 || n <= 0 {
            return Err(CudaError::Unsupported {
                msg: format!("sgemm_q5k_bf16_mvar: M={m} N={n} must be positive"),
            });
        }
        let (_module, func) = self.compile_or_get(
            &self.sgemm_q5k_mvar,
            SGEMM_Q5K_BF16_MVAR_SRC,
            "sgemm_q5k_bf16_mvar",
        )?;
        let n_mtiles = ((m as u32) + 7) >> 3;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (n as u32, n_mtiles, 1),
            block_dim: (64, 1, 1),
            shared_mem_bytes: 32 * 4,
        };
        let mut launcher = stream.launch_builder(&func);
        launcher.arg(&w_q5k).arg(&x).arg(&y).arg(&m).arg(&n).arg(&k);
        launcher.launch(cfg).map_err(|e| CudaError::Driver {
            code: format!("{e:?}").len() as i32,
            location: "sgemm_q5k_bf16_mvar::launch",
        })?;
        Ok(())
    }

    /// T246.10 A6b.1 — Q6_K matmul with arbitrary batch M.
    ///
    /// # Safety
    /// Same contract as `sgemm_q4k_bf16_mvar`.
    pub unsafe fn sgemm_q6k_bf16_mvar(
        &self,
        stream: &Arc<CudaStream>,
        w_q6k: u64,
        x: u64,
        y: u64,
        m: i32,
        n: i32,
        k: i32,
    ) -> Result<(), CudaError> {
        if k % 256 != 0 {
            return Err(CudaError::Unsupported {
                msg: format!("sgemm_q6k_bf16_mvar: K={k} must be multiple of 256"),
            });
        }
        if m <= 0 || n <= 0 {
            return Err(CudaError::Unsupported {
                msg: format!("sgemm_q6k_bf16_mvar: M={m} N={n} must be positive"),
            });
        }
        let (_module, func) = self.compile_or_get(
            &self.sgemm_q6k_mvar,
            SGEMM_Q6K_BF16_MVAR_SRC,
            "sgemm_q6k_bf16_mvar",
        )?;
        let n_mtiles = ((m as u32) + 7) >> 3;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (n as u32, n_mtiles, 1),
            block_dim: (64, 1, 1),
            shared_mem_bytes: 32 * 4,
        };
        let mut launcher = stream.launch_builder(&func);
        launcher.arg(&w_q6k).arg(&x).arg(&y).arg(&m).arg(&n).arg(&k);
        launcher.launch(cfg).map_err(|e| CudaError::Driver {
            code: format!("{e:?}").len() as i32,
            location: "sgemm_q6k_bf16_mvar::launch",
        })?;
        Ok(())
    }

    /// T246.10 A6b.1 — pure-BF16 matmul with arbitrary batch M.
    ///
    /// Inputs :
    ///   w : [N, K] BF16 row-major
    ///   x : [M, K] BF16 row-major
    ///   y : [M, N] BF16 row-major (output)
    ///
    /// # Safety
    /// Caller ensures pointers valid and K multiple of 256.
    pub unsafe fn sgemm_bf16_bf16_mvar(
        &self,
        stream: &Arc<CudaStream>,
        w: u64,
        x: u64,
        y: u64,
        m: i32,
        n: i32,
        k: i32,
    ) -> Result<(), CudaError> {
        if k % 256 != 0 {
            return Err(CudaError::Unsupported {
                msg: format!("sgemm_bf16_bf16_mvar: K={k} must be multiple of 256"),
            });
        }
        if m <= 0 || n <= 0 {
            return Err(CudaError::Unsupported {
                msg: format!("sgemm_bf16_bf16_mvar: M={m} N={n} must be positive"),
            });
        }
        let (_module, func) = self.compile_or_get(
            &self.sgemm_bf16_mvar,
            SGEMM_BF16_BF16_MVAR_SRC,
            "sgemm_bf16_bf16_mvar",
        )?;
        let n_mtiles = ((m as u32) + 7) >> 3;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (n as u32, n_mtiles, 1),
            block_dim: (64, 1, 1),
            shared_mem_bytes: 16 * 4,
        };
        let mut launcher = stream.launch_builder(&func);
        launcher.arg(&w).arg(&x).arg(&y).arg(&m).arg(&n).arg(&k);
        launcher.launch(cfg).map_err(|e| CudaError::Driver {
            code: format!("{e:?}").len() as i32,
            location: "sgemm_bf16_bf16_mvar::launch",
        })?;
        Ok(())
    }

    /// T246.10 TrackG-lite — BF16×BF16 GEMM via mma.sync m16n8k16 tensor
    /// cores (Ampere+ / Blackwell sm_121).
    ///
    /// API mirrors `sgemm_bf16_bf16_mvar` exactly :
    ///   w : [N, K] BF16 row-major
    ///   x : [M, K] BF16 row-major
    ///   y : [M, N] BF16 row-major
    ///
    /// Constraints :
    ///   - K must be multiple of 16 (we enforce 256 to keep parity with the
    ///     baseline kernel's constraint).
    ///   - M, N > 0. For M < 16 the warp-shuffle baseline is preferred —
    ///     A3 proved mma.sync at M=1 loses to warp-shuffle.
    ///
    /// # Safety
    /// Caller ensures device pointers valid, layouts as documented, and the
    /// stream is the one bound to this kernel's CUDA context.
    pub unsafe fn gemm_bf16_bf16_mma(
        &self,
        stream: &Arc<CudaStream>,
        w: u64,
        x: u64,
        y: u64,
        m: i32,
        n: i32,
        k: i32,
    ) -> Result<(), CudaError> {
        if k % 16 != 0 {
            return Err(CudaError::Unsupported {
                msg: format!("gemm_bf16_bf16_mma: K={k} must be multiple of 16"),
            });
        }
        if m <= 0 || n <= 0 {
            return Err(CudaError::Unsupported {
                msg: format!("gemm_bf16_bf16_mma: M={m} N={n} must be positive"),
            });
        }
        let (_module, func) = self.compile_or_get(
            &self.gemm_bf16_mma,
            GEMM_BF16_BF16_MMA_M16N8K16_SRC,
            "gemm_bf16_bf16_mma_m16n8k16",
        )?;
        // Block : (warp_size=32, warps=4, 1) = 128 threads ; one warp per
        // 8-col N-tile (4 warps × 8 cols = 32 cols/block) × one shared 16-row
        // M-tile.
        let grid_x = ((n as u32) + 31) / 32;
        let grid_y = ((m as u32) + 15) / 16;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (grid_x, grid_y, 1),
            block_dim: (32, 4, 1),
            // smem usage is static (A_smem + B_smem declared __shared__ in
            // the kernel) ; no dynamic smem needed.
            shared_mem_bytes: 0,
        };
        let mut launcher = stream.launch_builder(&func);
        launcher.arg(&w).arg(&x).arg(&y).arg(&m).arg(&n).arg(&k);
        launcher.launch(cfg).map_err(|e| CudaError::Driver {
            code: format!("{e:?}").len() as i32,
            location: "gemm_bf16_bf16_mma::launch",
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

    /// T246.6.9 — Q5_K SGEMV V3 (multi-row + per-thread scale +
    /// __launch_bounds__). Same speedup pattern as Q4_K V3, with qh-bit
    /// extraction for the 5th bit per weight.
    ///
    /// # Safety  Same contract as `sgemv_q5k_bf16`.
    pub unsafe fn sgemv_q5k_bf16_v3(
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
                msg: format!("sgemv_q5k_bf16_v3: K={k} must be multiple of 256"),
            });
        }
        let (_module, func) = self.compile_or_get(
            &self.sgemv_q5k_v3,
            SGEMV_Q5K_BF16_V3_SRC,
            "sgemv_q5k_bf16_v3",
        )?;
        const ROWS_PER_BLOCK: i32 = 4;
        let n_blocks = ((n + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK) as u32;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (n_blocks, 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut launcher = stream.launch_builder(&func);
        launcher.arg(&w_q5k).arg(&x).arg(&y).arg(&n).arg(&k);
        launcher.launch(cfg).map_err(|e| CudaError::Driver {
            code: format!("{e:?}").len() as i32,
            location: "sgemv_q5k_bf16_v3::launch",
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

    /// T246.6.8 — Q6_K SGEMV V4 (V2's 64-thread/row layout + 4-row blocks
    /// + per-thread scale + __launch_bounds__). Avoids V3's register
    /// pressure regression by keeping V2's per-thread weight count.
    ///
    /// # Safety  Same contract as `sgemv_q6k_bf16_v2`.
    pub unsafe fn sgemv_q6k_bf16_v4(
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
                msg: format!("sgemv_q6k_bf16_v4: K={k} must be multiple of 256"),
            });
        }
        let (_module, func) = self.compile_or_get(
            &self.sgemv_q6k_v4,
            SGEMV_Q6K_BF16_V4_SRC,
            "sgemv_q6k_bf16_v4",
        )?;
        const ROWS_PER_BLOCK: i32 = 4;
        let n_blocks = ((n + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK) as u32;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (n_blocks, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 4 * 2 * 4, // 4 rows × 2 warps = 8 float slots
        };
        let mut launcher = stream.launch_builder(&func);
        launcher.arg(&w_q6k).arg(&x).arg(&y).arg(&n).arg(&k);
        launcher.launch(cfg).map_err(|e| CudaError::Driver {
            code: format!("{e:?}").len() as i32,
            location: "sgemv_q6k_bf16_v4::launch",
        })?;
        Ok(())
    }

    /// T246.6.7 — Q6_K SGEMV V3 (multi-row, per-thread scale). Same speedup
    /// pattern as Q4_K V3, applied to Q6_K's 6-bit packed layout. Critical
    /// for the LM head matmul (N=152064 vocab × K=5120 hidden).
    ///
    /// # Safety  Same contract as `sgemv_q6k_bf16_v2`.
    pub unsafe fn sgemv_q6k_bf16_v3(
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
                msg: format!("sgemv_q6k_bf16_v3: K={k} must be multiple of 256"),
            });
        }
        let (_module, func) = self.compile_or_get(
            &self.sgemv_q6k_v3,
            SGEMV_Q6K_BF16_V3_SRC,
            "sgemv_q6k_bf16_v3",
        )?;
        const ROWS_PER_BLOCK: i32 = 4;
        let n_blocks = ((n + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK) as u32;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (n_blocks, 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut launcher = stream.launch_builder(&func);
        launcher.arg(&w_q6k).arg(&x).arg(&y).arg(&n).arg(&k);
        launcher.launch(cfg).map_err(|e| CudaError::Driver {
            code: format!("{e:?}").len() as i32,
            location: "sgemv_q6k_bf16_v3::launch",
        })?;
        Ok(())
    }

    /// T246.8 A5 — Split-K Q6_K SGEMV for the lm_head matmul.
    ///
    /// Splits K into `k_chunks` super-block-aligned chunks. Each chunk is
    /// processed by one (n_row, chunk) CUDA block ; the kernel writes
    /// FP32 partials of shape `[k_chunks, N]` to `partial`. A reduction
    /// kernel then sums dim 0 → BF16 `[N]` output. Caller provides a
    /// pre-allocated `partial` buffer of capacity ≥ `k_chunks * N` FP32.
    ///
    /// Constraints :
    /// - `k` must be a multiple of 256 (Q6_K super-block size).
    /// - `k_chunks` must divide `k / 256` (blocks_per_row).
    ///
    /// # Safety  Caller ensures all device pointers are valid for the
    /// duration of the launch, and `partial` is at least `k_chunks * n`
    /// FP32 elements.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn sgemv_q6k_bf16_split_k(
        &self,
        stream: &Arc<CudaStream>,
        w_q6k: u64,
        x: u64,
        partial: u64, // [k_chunks, N] FP32 staging
        y: u64,       // [N] BF16 output
        n: i32,
        k: i32,
        k_chunks: i32,
    ) -> Result<(), CudaError> {
        if k % 256 != 0 {
            return Err(CudaError::Unsupported {
                msg: format!("sgemv_q6k_bf16_split_k: K={k} must be multiple of 256"),
            });
        }
        let blocks_per_row = k / 256;
        if blocks_per_row % k_chunks != 0 {
            return Err(CudaError::Unsupported {
                msg: format!(
                    "sgemv_q6k_bf16_split_k: blocks_per_row={blocks_per_row} not divisible by k_chunks={k_chunks}"
                ),
            });
        }
        // ---- Partial kernel : Grid = (N, k_chunks) × 64 threads/block ----
        let (_pm, p_func) = self.compile_or_get(
            &self.sgemv_q6k_split_k_partial,
            SGEMV_Q6K_BF16_SPLIT_K_PARTIAL_SRC,
            "sgemv_q6k_bf16_split_k_partial",
        )?;
        let p_cfg = cudarc::driver::LaunchConfig {
            grid_dim: (n as u32, k_chunks as u32, 1),
            block_dim: (64, 1, 1),
            // shmem : 16 sc_pre + 2 sdata = 18 floats = 72 bytes
            shared_mem_bytes: 18 * 4,
        };
        let mut p_launcher = stream.launch_builder(&p_func);
        p_launcher
            .arg(&w_q6k)
            .arg(&x)
            .arg(&partial)
            .arg(&n)
            .arg(&k)
            .arg(&k_chunks);
        p_launcher.launch(p_cfg).map_err(|e| CudaError::Driver {
            code: format!("{e:?}").len() as i32,
            location: "sgemv_q6k_bf16_split_k_partial::launch",
        })?;
        // ---- Reduction kernel : 256-thread blocks over N rows ----
        let (_rm, r_func) = self.compile_or_get(
            &self.reduce_split_k_bf16,
            REDUCE_SPLIT_K_BF16_SRC,
            "reduce_split_k_bf16",
        )?;
        const R_BLOCK: i32 = 256;
        let r_grid = ((n + R_BLOCK - 1) / R_BLOCK) as u32;
        let r_cfg = cudarc::driver::LaunchConfig {
            grid_dim: (r_grid, 1, 1),
            block_dim: (R_BLOCK as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut r_launcher = stream.launch_builder(&r_func);
        r_launcher.arg(&partial).arg(&y).arg(&n).arg(&k_chunks);
        r_launcher.launch(r_cfg).map_err(|e| CudaError::Driver {
            code: format!("{e:?}").len() as i32,
            location: "reduce_split_k_bf16::launch",
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

    /// T246.8 A3 — Multi-row block BF16 thin GEMV (V2, bit-exact with V1).
    ///
    /// V1 = 1 row per block, 64 threads. V2 = 4 rows per block, 64 threads
    /// per row × 4 rows = 256 threads/block, with V1's exact per-thread
    /// arithmetic preserved → guaranteed bit-exact with V1 for unaltered
    /// argmax decode parity. Wins from :
    ///   - 4× fewer block launches (less SM scheduling overhead)
    ///   - L1 reuse on x : 4 row-warps in a block share x access pattern,
    ///     warps 1..3 hit L1 for x reads
    ///
    /// First-attempt mma.sync m16n8k16 variant was abandoned : it wasted
    /// 87.5% of compute on broadcast-x cols 1..7 of D and ran 30% slower
    /// than V1 end-to-end (note 0418e02c). Second attempt (32-thread/row
    /// warp-shuffle V2) was bit-exact in synthetic tests but produced
    /// different argmax tokens in the model — different MAC count per
    /// thread (K/32 vs V1's K/64) → different FP non-associativity.
    /// This 3rd version keeps V1's exact 64-thread/row pattern.
    ///
    /// # Safety
    ///
    /// Caller ensures pointers `w`, `x`, `y` are valid for the lifetime of
    /// the kernel and reference at least `N*K`, `K`, and `N` BF16 elements
    /// respectively. K must be a multiple of 256 (V1's constraint, kept
    /// identical for bit-exact MAC sequence).
    pub unsafe fn sgemv_bf16_bf16_v2(
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
                msg: format!("sgemv_bf16_bf16_v2: K={k} must be multiple of 256"),
            });
        }
        let (_module, func) = self.compile_or_get(
            &self.sgemv_bf16_v2,
            SGEMV_BF16_BF16_V2_SRC,
            "sgemv_bf16_bf16_v2",
        )?;
        const ROWS_PER_BLOCK: i32 = 4;
        let n_blocks = (n as u32).div_ceil(ROWS_PER_BLOCK as u32);
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (n_blocks, 1, 1),
            // 64 threads/row × 4 rows = 256 threads/block. (64,4) layout
            // keeps lane-id = threadIdx.x (0..63) bit-identical to V1's tid.
            block_dim: (64, ROWS_PER_BLOCK as u32, 1),
            // 4 rows × 2 half-warps = 8 floats reduction buffer.
            shared_mem_bytes: (ROWS_PER_BLOCK as u32) * 2 * 4,
        };
        let mut launcher = stream.launch_builder(&func);
        launcher.arg(&w).arg(&x).arg(&y).arg(&n).arg(&k);
        launcher.launch(cfg).map_err(|e| CudaError::Driver {
            code: format!("{e:?}").len() as i32,
            location: "sgemv_bf16_bf16_v2::launch",
        })?;
        Ok(())
    }

    /// T246.8 A3 — Dispatch helper : V2 (tensor-core) for N >= 128 and
    /// K multiple of 16, else V1 (warp-shuffle, requires K multiple of 256).
    ///
    /// If neither is applicable (K < 256 not multiple of 16) we still try V2
    /// since its only constraint is K % 16 == 0 ; otherwise we propagate the
    /// V1 error to surface the unsupported shape.
    ///
    /// # Safety
    ///
    /// Same contract as `sgemv_bf16_bf16` / `sgemv_bf16_bf16_v2` : pointers
    /// must be valid for the duration of the kernel.
    pub unsafe fn sgemv_bf16_bf16_dispatch(
        &self,
        stream: &Arc<CudaStream>,
        w: u64,
        x: u64,
        y: u64,
        n: i32,
        k: i32,
    ) -> Result<(), CudaError> {
        // V2 (4-row-per-block, bit-exact with V1) shows neutral perf in
        // Qwen3.6-35B-A3B end-to-end bench (32.55 V1 vs 32.36 V2 tok/s on
        // GB10 sm_121, T246.8 A3 — see note 0418e02c & superseder). The
        // BF16 SGEMV is bandwidth-bound and topology changes alone don't
        // recover a meaningful win ; the real bottleneck is the per-tensor
        // launch sequence which only A4 (mul_mm_id mega-kernel) can fix.
        //
        // V2 is therefore DEFAULT OFF (env-opt-in via RUSTORCH_ENABLE_BF16_V2=1)
        // to keep the simpler V1 path the default. Code is retained because :
        //   1. Parity tests pass bit-exact
        //   2. V2 may win on different hardware (Hopper / Ada) where launch
        //      overhead dominates more than bandwidth
        //   3. Future work can restructure V2 (e.g. larger blocks, persistent
        //      kernel) without rewriting the dispatch surface
        static ENABLE_V2: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let enable_v2 = *ENABLE_V2.get_or_init(|| {
            std::env::var("RUSTORCH_ENABLE_BF16_V2")
                .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                .unwrap_or(false)
        });
        if enable_v2 && n >= 128 && k % 256 == 0 {
            self.sgemv_bf16_bf16_v2(stream, w, x, y, n, k)
        } else {
            self.sgemv_bf16_bf16(stream, w, x, y, n, k)
        }
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

    /// T246.7 TrackC.1 — tree-aware Gated DeltaNet recurrent step for one
    /// BFS depth wave. The caller invokes this once per depth (root wave
    /// first, then depth-1, depth-2, …) ; `wave_indices_dev` lists the
    /// tree-row indices to process in this wave.
    ///
    /// Per-branch state forking is automatic : when `parents[tr] >= 0` the
    /// kernel reads the source state from `tree_states[parents[tr]]` and
    /// writes to `tree_states[tr]`. Sibling branches sharing a parent
    /// thus fork independently. For the root (parents[tr] < 0) the source
    /// equals the destination ; the caller MUST pre-load the model's
    /// current per-layer SSM state into the root slot before launching.
    ///
    /// # Safety
    /// Caller ensures :
    ///   - `q, k, v` are valid for `tree_size × n_heads × head_dim` BF16 each.
    ///   - `gate, beta` are valid for `tree_size × n_heads` BF16 each.
    ///   - `parents` is a `tree_size`-long `i32` device array, BFS order
    ///     (`parents[i] < i` for `i > 0`, `parents[0] == -1`).
    ///   - `wave_indices` is a `wave_size`-long `i32` device array of
    ///     tree-row indices in this depth wave.
    ///   - `tree_states` is `tree_size × n_heads × head_dim²` BF16, with
    ///     the root slot pre-loaded with the model's current state.
    ///   - `out` is `tree_size × n_heads × head_dim` BF16.
    ///   - `head_dim ≤ 256`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn delta_net_step_tree_bf16(
        &self,
        stream: &Arc<CudaStream>,
        q: u64,
        k: u64,
        v: u64,
        gate: u64,
        beta: u64,
        parents: u64,
        wave_indices: u64,
        tree_states: u64,
        out: u64,
        wave_size: i32,
        n_heads: i32,
        head_dim: i32,
    ) -> Result<(), CudaError> {
        let (_module, func) = self.compile_or_get(
            &self.delta_net_step_tree,
            DELTA_NET_STEP_TREE_BF16_SRC,
            "delta_net_step_tree_bf16",
        )?;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (n_heads as u32, wave_size as u32, 1),
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
            .arg(&parents)
            .arg(&wave_indices)
            .arg(&tree_states)
            .arg(&out)
            .arg(&wave_size)
            .arg(&n_heads)
            .arg(&head_dim);
        launcher.launch(cfg).map_err(|e| CudaError::Driver {
            code: format!("{e:?}").len() as i32,
            location: "delta_net_step_tree_bf16::launch",
        })?;
        Ok(())
    }

    /// T246.10 TrackF — Column-parallel optimized variant of
    /// `delta_net_step_tree_bf16`.
    ///
    /// Same algorithm, same I/O contract, same launch dimensions
    /// (block = head_dim threads, grid = (n_heads, wave_size, 1)).
    /// The only differences are :
    ///   * `threadIdx.x` indexes the state column `c` rather than the
    ///     state row `r` → coalesced global memory access on the
    ///     `tree_states` RMW (which is the dominant cost).
    ///   * Each row's output is computed via a block-wide warp-shuffle
    ///     reduction over the column-threads.
    ///
    /// Numerically equivalent up to FP32 reduction-order : results may
    /// differ in the last few BF16 ULPs from the baseline. The dispatch
    /// gate `RUSTORCH_DELTA_NET_OPT=1` in `qwen35_cuda_q4k.rs` controls
    /// adoption ; default OFF preserves bit-exact parity with the
    /// baseline kernel.
    ///
    /// # Safety
    /// All pointers must reference valid CUDA device memory with the
    /// shapes documented for `delta_net_step_tree_bf16`. `head_dim` must
    /// be a multiple of 32 and ≤ 1024.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn delta_net_step_tree_bf16_opt(
        &self,
        stream: &Arc<CudaStream>,
        q: u64,
        k: u64,
        v: u64,
        gate: u64,
        beta: u64,
        parents: u64,
        wave_indices: u64,
        tree_states: u64,
        out: u64,
        wave_size: i32,
        n_heads: i32,
        head_dim: i32,
    ) -> Result<(), CudaError> {
        let (_module, func) = self.compile_or_get(
            &self.delta_net_step_tree_opt,
            DELTA_NET_STEP_TREE_BF16_OPT_SRC,
            "delta_net_step_tree_bf16_opt",
        )?;
        // Dynamic smem : one float per warp for inter-warp reduction.
        let n_warps = ((head_dim as u32) + 31) / 32;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (n_heads as u32, wave_size as u32, 1),
            block_dim: (head_dim as u32, 1, 1),
            shared_mem_bytes: n_warps * 4,
        };
        let mut launcher = stream.launch_builder(&func);
        launcher
            .arg(&q)
            .arg(&k)
            .arg(&v)
            .arg(&gate)
            .arg(&beta)
            .arg(&parents)
            .arg(&wave_indices)
            .arg(&tree_states)
            .arg(&out)
            .arg(&wave_size)
            .arg(&n_heads)
            .arg(&head_dim);
        launcher.launch(cfg).map_err(|e| CudaError::Driver {
            code: format!("{e:?}").len() as i32,
            location: "delta_net_step_tree_bf16_opt::launch",
        })?;
        Ok(())
    }

    /// NEW-SSM — bandwidth-saturated variant of `delta_net_step_tree_bf16`.
    ///
    /// Same I/O contract as the baseline, different launch geometry :
    /// `block = (WARP_SIZE=32, n_warps = head_dim/32, 1)`. Each warp owns
    /// `head_dim / n_warps` state rows and accesses its columns coalesced.
    /// No block-wide sync ; per-row output via warp-shuffle reduction.
    ///
    /// Gated by `RUSTORCH_DELTA_NET_NEW=1` in `qwen35_cuda_q4k.rs`.
    /// Default OFF preserves bit-exact parity with the baseline kernel.
    ///
    /// # Safety
    /// All pointers must reference valid CUDA device memory with the
    /// shapes documented for `delta_net_step_tree_bf16`. Requires
    /// `head_dim` to be a multiple of 32 and ≤ 256.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn delta_net_step_tree_bf16_v2(
        &self,
        stream: &Arc<CudaStream>,
        q: u64,
        k: u64,
        v: u64,
        gate: u64,
        beta: u64,
        parents: u64,
        wave_indices: u64,
        tree_states: u64,
        out: u64,
        wave_size: i32,
        n_heads: i32,
        head_dim: i32,
    ) -> Result<(), CudaError> {
        let (_module, func) = self.compile_or_get(
            &self.delta_net_step_tree_v2,
            DELTA_NET_STEP_TREE_BF16_V2_SRC,
            "delta_net_step_tree_bf16_v2",
        )?;
        // n_warps = head_dim / 32 ; for head_dim=128 ⇒ 4 warps × 32 lanes = 128 threads/CTA.
        // Clamp head_dim to multiple of 32 (caller enforces).
        let warp_size: u32 = 32;
        let n_warps: u32 = (head_dim as u32) / warp_size;
        let n_warps = n_warps.max(1);
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (n_heads as u32, wave_size as u32, 1),
            block_dim: (warp_size, n_warps, 1),
            shared_mem_bytes: 0,
        };
        let mut launcher = stream.launch_builder(&func);
        launcher
            .arg(&q)
            .arg(&k)
            .arg(&v)
            .arg(&gate)
            .arg(&beta)
            .arg(&parents)
            .arg(&wave_indices)
            .arg(&tree_states)
            .arg(&out)
            .arg(&wave_size)
            .arg(&n_heads)
            .arg(&head_dim);
        launcher.launch(cfg).map_err(|e| CudaError::Driver {
            code: format!("{e:?}").len() as i32,
            location: "delta_net_step_tree_bf16_v2::launch",
        })?;
        Ok(())
    }

    /// T246.8 A1.1 — Fused SSM pre-step elementwise chain.
    ///
    /// Replaces 4 separate kernel launches per SSM layer per token :
    ///   1. `sigmoid_inplace_bf16(beta, n)`
    ///   2. `add_inplace_bf16(alpha, dt_bias, n)`
    ///   3. `softplus_inplace_bf16(alpha, n)`
    ///   4. `mul_inplace_bf16(alpha, ssm_a, n)`
    ///
    /// Each thread handles ONE element ; intermediate values round-trip
    /// through bf16 between phases to preserve bit-exact parity with
    /// the four-kernel sequence.
    ///
    /// # Safety  All four pointers must reference length-`n` BF16 device
    /// buffers ; `alpha` and `beta` are mutated in place.
    pub unsafe fn ssm_pre_step_bf16(
        &self,
        stream: &Arc<CudaStream>,
        alpha: u64,
        beta: u64,
        dt_bias: u64,
        ssm_a: u64,
        n: i32,
    ) -> Result<(), CudaError> {
        let (_module, func) = self.compile_or_get(
            &self.ssm_pre_step,
            SSM_PRE_STEP_BF16_SRC,
            "ssm_pre_step_bf16",
        )?;
        let block_dim: u32 = 256;
        let grid_dim = (n as u32).div_ceil(block_dim);
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (grid_dim, 1, 1),
            block_dim: (block_dim, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut launcher = stream.launch_builder(&func);
        launcher
            .arg(&alpha)
            .arg(&beta)
            .arg(&dt_bias)
            .arg(&ssm_a)
            .arg(&n);
        launcher.launch(cfg).map_err(|e| CudaError::Driver {
            code: format!("{e:?}").len() as i32,
            location: "ssm_pre_step_bf16::launch",
        })?;
        Ok(())
    }

    /// T246.8 A1.2 — Fused SSM post-step output processing.
    ///
    /// Replaces 3 separate kernel launches per SSM layer per token :
    ///   1. `rms_norm_bf16(out, gamma, eps, head_kv, n_v)` — per head
    ///   2. `silu_bf16(z, n_v * head_kv)` (inplace)
    ///   3. `mul_inplace_bf16(out, z, n_v * head_kv)`
    ///
    /// One block per head, `head_kv` threads ; tree reduction over
    /// head_kv mirrors `rms_norm_bf16` so `inv_rms` is bit-identical.
    /// All intermediates round-trip through bf16 to preserve bit-exact
    /// parity with the unfused chain.
    ///
    /// # Safety  `out` and `z` are mutated in place. `out` and `z` must
    /// be `n_v * head_kv` BF16 each ; `gamma` is `head_kv` BF16.
    /// `head_kv` must be a power of two ≤ 1024 (constraint inherited
    /// from the tree-reduction shape).
    pub unsafe fn ssm_post_step_bf16(
        &self,
        stream: &Arc<CudaStream>,
        out: u64,
        z: u64,
        gamma: u64,
        eps: f32,
        n_v: i32,
        head_kv: i32,
    ) -> Result<(), CudaError> {
        let (_module, func) = self.compile_or_get(
            &self.ssm_post_step,
            SSM_POST_STEP_BF16_SRC,
            "ssm_post_step_bf16",
        )?;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (n_v as u32, 1, 1),
            block_dim: (head_kv as u32, 1, 1),
            shared_mem_bytes: (head_kv as u32) * 4,
        };
        let mut launcher = stream.launch_builder(&func);
        launcher
            .arg(&out)
            .arg(&z)
            .arg(&gamma)
            .arg(&eps)
            .arg(&head_kv);
        launcher.launch(cfg).map_err(|e| CudaError::Driver {
            code: format!("{e:?}").len() as i32,
            location: "ssm_post_step_bf16::launch",
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

    /// T247.1 — RMSNorm backward kernel (single-row, training-ready pilot).
    ///
    /// Computes `dx[i]` and `dgamma[i]` given `x`, `gamma`, `dy`. Math :
    ///   dx_i     = (dy_i * gamma_i) / r  -  x_i * sum(dy*gamma*x) / (d * r³)
    ///   dgamma_i = dy_i * x_i / r
    /// where r = sqrt(mean(x²) + eps).
    ///
    /// Single-row (outer=1). For multi-row training, wrap with a per-row
    /// loop or extend to grid_dim.x = outer with atomic dgamma reduction.
    ///
    /// # Safety  Pointers must be valid bf16 buffers of length d (or d²)
    /// for the kernel lifetime.
    pub unsafe fn rms_norm_grad_bf16(
        &self,
        stream: &Arc<CudaStream>,
        x: u64,
        gamma: u64,
        dy: u64,
        dx: u64,
        dgamma: u64,
        d: i32,
        eps: f32,
    ) -> Result<(), CudaError> {
        let (_module, func) = self.compile_or_get(
            &self.rms_norm_grad,
            RMS_NORM_GRAD_BF16_SRC,
            "rms_norm_grad_bf16",
        )?;
        // Pick a power-of-2 block dim ≥ 32, ≤ 1024. For d ≤ 1024, block = d
        // (rounded up to next power of 2 for the tree reduction). For d > 1024,
        // use 1024 and let the grid-stride loop handle the rest.
        let bd = (d as u32).next_power_of_two().clamp(32, 1024);
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (bd, 1, 1),
            shared_mem_bytes: bd * 4,
        };
        let mut launcher = stream.launch_builder(&func);
        launcher
            .arg(&x)
            .arg(&gamma)
            .arg(&dy)
            .arg(&dx)
            .arg(&dgamma)
            .arg(&d)
            .arg(&eps);
        launcher.launch(cfg).map_err(|e| CudaError::Driver {
            code: format!("{e:?}").len() as i32,
            location: "rms_norm_grad_bf16::launch",
        })?;
        Ok(())
    }

    /// T246.6.3 — `y[i] += alpha · x[i]` for length-n bf16 vectors.
    /// Used by MoE FFN forward to accumulate weighted expert outputs.
    ///
    /// # Safety  Both pointers must reference length-n bf16 device buffers.
    pub unsafe fn scaled_add_inplace_bf16(
        &self,
        stream: &Arc<CudaStream>,
        y: u64,
        x: u64,
        alpha: f32,
        n: i32,
    ) -> Result<(), CudaError> {
        let (_module, func) = self.compile_or_get(
            &self.scaled_add_inplace,
            SCALED_ADD_INPLACE_BF16_SRC,
            "scaled_add_inplace_bf16",
        )?;
        let block_dim: u32 = 256;
        let grid_dim = ((n as u32) + block_dim - 1) / block_dim;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (grid_dim, 1, 1),
            block_dim: (block_dim, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut launcher = stream.launch_builder(&func);
        launcher.arg(&y).arg(&x).arg(&alpha).arg(&n);
        launcher.launch(cfg).map_err(|e| CudaError::Driver {
            code: format!("{e:?}").len() as i32,
            location: "scaled_add_inplace_bf16::launch",
        })?;
        Ok(())
    }

    /// T246.6.2 — Top-K softmax routing for MoE FFN. Given raw scores
    /// `[n_experts]`, returns top-K indices and renormalized softmax
    /// weights such that `sum(weights[0..K]) == 1`.
    ///
    /// # Safety  Caller ensures `scores` is bf16 length `n_experts`,
    /// `indices` is i32 length k, `weights` is bf16 length k. All device-
    /// resident. n_experts ≤ 1024.
    pub unsafe fn topk_softmax_bf16(
        &self,
        stream: &Arc<CudaStream>,
        scores: u64,
        indices: u64,
        weights: u64,
        n_experts: i32,
        k: i32,
    ) -> Result<(), CudaError> {
        let (_module, func) = self.compile_or_get(
            &self.topk_softmax,
            TOPK_SOFTMAX_BF16_SRC,
            "topk_softmax_bf16",
        )?;
        let block_dim: u32 = 32;
        // shmem: n_experts probs + 3 scratch (max, Z, Z2)
        let shmem = ((n_experts as u32) + 3) * 4;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (block_dim, 1, 1),
            shared_mem_bytes: shmem,
        };
        let mut launcher = stream.launch_builder(&func);
        launcher
            .arg(&scores)
            .arg(&indices)
            .arg(&weights)
            .arg(&n_experts)
            .arg(&k);
        launcher.launch(cfg).map_err(|e| CudaError::Driver {
            code: format!("{e:?}").len() as i32,
            location: "topk_softmax_bf16::launch",
        })?;
        Ok(())
    }

    /// T247.7 — GQA decode backward (Flash-Attention style, M=1).
    ///
    /// Computes dQ/dK/dV for one decode step given the saved softmax
    /// statistics (m, l) from a training-aware forward pass. The backward
    /// recomputes the attention probabilities `p` on the fly from
    /// `m_saved`, `l_saved`, and `Q·K_t` — avoids storing the full P
    /// matrix per layer.
    ///
    /// # Safety  All bf16 buffers are sized for n_heads × head_dim
    /// (q, do, dq) or n_kv × max_seq × head_dim (k_cache, v_cache).
    /// `m_saved`, `l_saved` are length-`n_heads` floats. `dk_accum` and
    /// `dv_accum` are float accumulators of size `n_kv × max_seq × head_dim`,
    /// caller must zero them before the first call of an iteration.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn gqa_decode_grad_bf16(
        &self,
        stream: &Arc<CudaStream>,
        q: u64,
        k_cache: u64,
        v_cache: u64,
        m_saved: u64,
        l_saved: u64,
        do_: u64,
        dq: u64,
        dk_accum: u64,
        dv_accum: u64,
        n_heads: i32,
        n_kv: i32,
        kv_len: i32,
        head_dim: i32,
        max_seq: i32,
    ) -> Result<(), CudaError> {
        let (_module, func) = self.compile_or_get(
            &self.gqa_decode_grad,
            GQA_DECODE_GRAD_BF16_SRC,
            "gqa_decode_grad_bf16",
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
            .arg(&m_saved)
            .arg(&l_saved)
            .arg(&do_)
            .arg(&dq)
            .arg(&dk_accum)
            .arg(&dv_accum)
            .arg(&n_heads)
            .arg(&n_kv)
            .arg(&kv_len)
            .arg(&head_dim)
            .arg(&max_seq)
            .arg(&scale);
        launcher.launch(cfg).map_err(|e| CudaError::Driver {
            code: format!("{e:?}").len() as i32,
            location: "gqa_decode_grad_bf16::launch",
        })?;
        Ok(())
    }

    /// T247.6 — Q4_K SGEMV backward, dx only (W frozen for LoRA fine-tune).
    /// Computes `dx[k] = Σ_n W[n,k] · dy[n]`. Pilot impl is correctness-
    /// focused (column-major access over W — uncoalesced) ; production
    /// optimization (tile / pre-transpose) tracked in T247.6 follow-up.
    ///
    /// # Safety  K must be multiple of 256 ; `w_q4k` is row-major Q4_K
    /// `N * K/256 * 144` bytes ; `dy` length N bf16 ; `dx` length K bf16.
    pub unsafe fn sgemv_q4k_grad_dx_bf16(
        &self,
        stream: &Arc<CudaStream>,
        w_q4k: u64,
        dy: u64,
        dx: u64,
        n: i32,
        k: i32,
    ) -> Result<(), CudaError> {
        if k % 256 != 0 {
            return Err(CudaError::Unsupported {
                msg: format!("sgemv_q4k_grad_dx_bf16: K={k} must be multiple of 256"),
            });
        }
        let (_module, func) = self.compile_or_get(
            &self.sgemv_q4k_grad_dx,
            SGEMV_Q4K_GRAD_DX_BF16_SRC,
            "sgemv_q4k_grad_dx_bf16",
        )?;
        let block_dim: u32 = 256;
        let grid_dim = ((k as u32) + block_dim - 1) / block_dim;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (grid_dim, 1, 1),
            block_dim: (block_dim, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut launcher = stream.launch_builder(&func);
        launcher.arg(&w_q4k).arg(&dy).arg(&dx).arg(&n).arg(&k);
        launcher.launch(cfg).map_err(|e| CudaError::Driver {
            code: format!("{e:?}").len() as i32,
            location: "sgemv_q4k_grad_dx_bf16::launch",
        })?;
        Ok(())
    }

    /// T247.5 — Embedding lookup backward — atomic scatter into a float
    /// accumulator. Caller zeroes `d_embed_accum` (length `vocab * d`)
    /// before the first call of a training iteration.
    ///
    /// # Safety  `dy` is length-`d` bf16, `d_embed_accum` is length-`vocab*d`
    /// float, both device-resident.
    pub unsafe fn embedding_lookup_grad_bf16(
        &self,
        stream: &Arc<CudaStream>,
        dy: u64,
        token_id: i32,
        d_embed_accum: u64,
        d: i32,
    ) -> Result<(), CudaError> {
        let (_module, func) = self.compile_or_get(
            &self.embedding_lookup_grad,
            EMBEDDING_LOOKUP_GRAD_BF16_SRC,
            "embedding_lookup_grad_bf16",
        )?;
        let block_dim: u32 = 256;
        let grid_dim = ((d as u32) + block_dim - 1) / block_dim;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (grid_dim, 1, 1),
            block_dim: (block_dim, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut launcher = stream.launch_builder(&func);
        launcher.arg(&dy).arg(&token_id).arg(&d_embed_accum).arg(&d);
        launcher.launch(cfg).map_err(|e| CudaError::Driver {
            code: format!("{e:?}").len() as i32,
            location: "embedding_lookup_grad_bf16::launch",
        })?;
        Ok(())
    }

    /// T247.4 — Cross-entropy from logits, fused forward + backward.
    ///
    /// Returns scalar loss in `loss_out` and the gradient w.r.t. logits in
    /// `dlogits`. Numerically stable via max-shift.
    ///
    /// # Safety  `logits` and `dlogits` are length-`vocab` bf16 buffers,
    /// `loss_out` is a length-1 float buffer, all device-resident.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn cross_entropy_loss_grad_bf16(
        &self,
        stream: &Arc<CudaStream>,
        logits: u64,
        target: i32,
        loss_out: u64,
        dlogits: u64,
        vocab: i32,
    ) -> Result<(), CudaError> {
        let (_module, func) = self.compile_or_get(
            &self.cross_entropy_loss_grad,
            CROSS_ENTROPY_LOSS_GRAD_BF16_SRC,
            "cross_entropy_loss_grad_bf16",
        )?;
        // Pick max-power-of-2 block dim ≤ 1024 such that block_dim ≤ vocab.
        let bd = (vocab as u32).next_power_of_two().min(1024).max(32);
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (bd, 1, 1),
            shared_mem_bytes: bd * 4,
        };
        let mut launcher = stream.launch_builder(&func);
        launcher
            .arg(&logits)
            .arg(&target)
            .arg(&loss_out)
            .arg(&dlogits)
            .arg(&vocab);
        launcher.launch(cfg).map_err(|e| CudaError::Driver {
            code: format!("{e:?}").len() as i32,
            location: "cross_entropy_loss_grad_bf16::launch",
        })?;
        Ok(())
    }

    /// T247.3 — RoPE partial backward kernel. Inverse rotation per pair.
    ///
    /// # Safety  All bf16 buffers length `n_heads * head_dim`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn rope_partial_grad_bf16(
        &self,
        stream: &Arc<CudaStream>,
        dy: u64,
        inv_freq: u64,
        pos: i32,
        dx: u64,
        n_heads: i32,
        head_dim: i32,
        rope_dim: i32,
    ) -> Result<(), CudaError> {
        let (_module, func) = self.compile_or_get(
            &self.rope_partial_grad,
            ROPE_PARTIAL_GRAD_BF16_SRC,
            "rope_partial_grad_bf16",
        )?;
        let half = (rope_dim / 2) as u32;
        let block_dim: u32 = 32.min(half.max(1));
        let grid_y = (half + block_dim - 1) / block_dim;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (n_heads as u32, grid_y, 1),
            block_dim: (block_dim, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut launcher = stream.launch_builder(&func);
        launcher
            .arg(&dy)
            .arg(&inv_freq)
            .arg(&pos)
            .arg(&dx)
            .arg(&n_heads)
            .arg(&head_dim)
            .arg(&rope_dim);
        launcher.launch(cfg).map_err(|e| CudaError::Driver {
            code: format!("{e:?}").len() as i32,
            location: "rope_partial_grad_bf16::launch",
        })?;
        Ok(())
    }

    /// T247.2 — SwiGLU backward kernel. Elementwise, n threads total.
    ///
    /// Inputs (forward saved): gate, up. Plus output gradient dy.
    /// Outputs: dgate (gradient w.r.t. gate input), dup (gradient w.r.t. up).
    ///
    /// # Safety  All five buffers are length-`n` bf16 device pointers.
    pub unsafe fn swiglu_grad_bf16(
        &self,
        stream: &Arc<CudaStream>,
        gate: u64,
        up: u64,
        dy: u64,
        dgate: u64,
        dup: u64,
        n: i32,
    ) -> Result<(), CudaError> {
        let (_module, func) =
            self.compile_or_get(&self.swiglu_grad, SWIGLU_GRAD_BF16_SRC, "swiglu_grad_bf16")?;
        let block_dim: u32 = 256;
        let grid_dim = ((n as u32) + block_dim - 1) / block_dim;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (grid_dim, 1, 1),
            block_dim: (block_dim, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut launcher = stream.launch_builder(&func);
        launcher
            .arg(&gate)
            .arg(&up)
            .arg(&dy)
            .arg(&dgate)
            .arg(&dup)
            .arg(&n);
        launcher.launch(cfg).map_err(|e| CudaError::Driver {
            code: format!("{e:?}").len() as i32,
            location: "swiglu_grad_bf16::launch",
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

    // ════════════════════════════════════════════════════════════════════
    //  T246.7 P1.3 — Lookahead Decoding tree kernels
    // ════════════════════════════════════════════════════════════════════

    /// T246.7 P1.3a — append `tree_size` consecutive K/V rows to the cache
    /// starting at slot `*pos_dev`. Bit-equivalent to `tree_size` calls of
    /// `kv_append_bf16_devcnt` when tree_size=1.
    ///
    /// # Safety  Caller assures `k_cache`/`v_cache` valides pour
    /// `max_seq * kv_dim` BF16, `k_in`/`v_in` for `tree_size * kv_dim`,
    /// `pos_dev` pointe vers 1 i32 device. `*pos_dev + tree_size <= max_seq`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn kv_append_tree_bf16(
        &self,
        stream: &Arc<CudaStream>,
        k_cache: u64,
        v_cache: u64,
        k_in: u64,
        v_in: u64,
        pos_dev: u64,
        tree_size: i32,
        kv_dim: i32,
        max_seq: i32,
    ) -> Result<(), CudaError> {
        let (_module, func) = self.compile_or_get(
            &self.kv_append_tree,
            KV_APPEND_TREE_BF16_SRC,
            "kv_append_tree_bf16",
        )?;
        let block_dim = 256u32;
        let grid_x = (kv_dim as u32).div_ceil(block_dim);
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (grid_x, tree_size as u32, 1),
            block_dim: (block_dim, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut launcher = stream.launch_builder(&func);
        launcher
            .arg(&k_cache)
            .arg(&v_cache)
            .arg(&k_in)
            .arg(&v_in)
            .arg(&pos_dev)
            .arg(&tree_size)
            .arg(&kv_dim)
            .arg(&max_seq);
        launcher.launch(cfg).map_err(|e| CudaError::Driver {
            code: format!("{e:?}").len() as i32,
            location: "kv_append_tree_bf16::launch",
        })?;
        Ok(())
    }

    /// T246.7 P1.3b — tree-aware GQA decode (FlashDecode-V2 split-K with
    /// per-token tree-attention mask). Two-phase like `gqa_decode_split_bf16`.
    ///
    /// # Safety  Same as `gqa_decode_split_bf16`, plus `parents_dev`
    /// (i32 [tree_size]) and `depths_dev` (u16 [tree_size]) device pointers.
    /// Partial buffer sizes : m/l = tree_size * n_q * n_split floats ;
    /// o = tree_size * n_q * n_split * head_dim BF16.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn gqa_decode_tree_bf16(
        &self,
        stream: &Arc<CudaStream>,
        q: u64,
        k_cache: u64,
        v_cache: u64,
        out: u64,
        parents_dev: u64,
        depths_dev: u64,
        partial_m_dev: u64,
        partial_l_dev: u64,
        partial_o_dev: u64,
        n_heads: i32,
        n_kv: i32,
        kv_len_dev: u64,
        head_dim: i32,
        max_seq: i32,
        n_split: i32,
        tree_size: i32,
    ) -> Result<(), CudaError> {
        let scale: f32 = 1.0 / (head_dim as f32).sqrt();
        let block_dim = head_dim as u32;

        // Phase 1 — partial.
        let (_m_p, fn_p) = self.compile_or_get(
            &self.gqa_decode_tree_partial,
            GQA_DECODE_TREE_PARTIAL_BF16_SRC,
            "gqa_decode_tree_partial_bf16",
        )?;
        let cfg_p = cudarc::driver::LaunchConfig {
            grid_dim: (n_heads as u32, n_split as u32, tree_size as u32),
            block_dim: (block_dim, 1, 1),
            shared_mem_bytes: block_dim * 4,
        };
        let mut launcher_p = stream.launch_builder(&fn_p);
        launcher_p
            .arg(&q)
            .arg(&k_cache)
            .arg(&v_cache)
            .arg(&parents_dev)
            .arg(&depths_dev)
            .arg(&partial_m_dev)
            .arg(&partial_l_dev)
            .arg(&partial_o_dev)
            .arg(&n_heads)
            .arg(&n_kv)
            .arg(&kv_len_dev)
            .arg(&head_dim)
            .arg(&max_seq)
            .arg(&n_split)
            .arg(&tree_size)
            .arg(&scale);
        launcher_p.launch(cfg_p).map_err(|e| CudaError::Driver {
            code: format!("{e:?}").len() as i32,
            location: "gqa_decode_tree_partial_bf16::launch",
        })?;

        // Phase 2 — combine.
        let (_m_c, fn_c) = self.compile_or_get(
            &self.gqa_decode_tree_combine,
            GQA_DECODE_TREE_COMBINE_BF16_SRC,
            "gqa_decode_tree_combine_bf16",
        )?;
        let cfg_c = cudarc::driver::LaunchConfig {
            grid_dim: (n_heads as u32, tree_size as u32, 1),
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
            .arg(&n_split)
            .arg(&tree_size);
        launcher_c.launch(cfg_c).map_err(|e| CudaError::Driver {
            code: format!("{e:?}").len() as i32,
            location: "gqa_decode_tree_combine_bf16::launch",
        })?;

        Ok(())
    }

    /// T246.7 P1.3c — argmax over `tree_size` independent rows of `vocab`
    /// BF16 logits. Bit-equivalent to looping `argmax_bf16` `tree_size` times.
    ///
    /// # Safety  `logits` valid for tree_size × vocab BF16 ; `tokens_out`
    /// valid for tree_size u32.
    pub unsafe fn argmax_logits_tree_bf16(
        &self,
        stream: &Arc<CudaStream>,
        logits: u64,
        tokens_out: u64,
        tree_size: i32,
        vocab: i32,
    ) -> Result<(), CudaError> {
        let (_module, func) = self.compile_or_get(
            &self.argmax_logits_tree,
            ARGMAX_LOGITS_TREE_BF16_SRC,
            "argmax_logits_tree_bf16",
        )?;
        let block_dim = 256u32;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (tree_size as u32, 1, 1),
            block_dim: (block_dim, 1, 1),
            shared_mem_bytes: block_dim * 8, // float + int per thread
        };
        let mut launcher = stream.launch_builder(&func);
        launcher
            .arg(&logits)
            .arg(&tokens_out)
            .arg(&tree_size)
            .arg(&vocab);
        launcher.launch(cfg).map_err(|e| CudaError::Driver {
            code: format!("{e:?}").len() as i32,
            location: "argmax_logits_tree_bf16::launch",
        })?;
        Ok(())
    }

    /// T246.7 P1.3c — `*p += value` (atomic-style, single thread).
    ///
    /// # Safety  `p` must point to 1 i32 device-resident.
    pub unsafe fn add_u32_dev(
        &self,
        stream: &Arc<CudaStream>,
        p: u64,
        value: i32,
    ) -> Result<(), CudaError> {
        let (_module, func) =
            self.compile_or_get(&self.add_u32_dev, ADD_U32_DEV_SRC, "add_u32_dev")?;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (1, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut launcher = stream.launch_builder(&func);
        launcher.arg(&p).arg(&value);
        launcher.launch(cfg).map_err(|e| CudaError::Driver {
            code: format!("{e:?}").len() as i32,
            location: "add_u32_dev::launch",
        })?;
        Ok(())
    }

    /// T246.7 P1.3c — `*p = value`.
    ///
    /// # Safety  `p` must point to 1 i32 device-resident.
    pub unsafe fn set_u32_dev(
        &self,
        stream: &Arc<CudaStream>,
        p: u64,
        value: i32,
    ) -> Result<(), CudaError> {
        let (_module, func) =
            self.compile_or_get(&self.set_u32_dev, SET_U32_DEV_SRC, "set_u32_dev")?;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (1, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut launcher = stream.launch_builder(&func);
        launcher.arg(&p).arg(&value);
        launcher.launch(cfg).map_err(|e| CudaError::Driver {
            code: format!("{e:?}").len() as i32,
            location: "set_u32_dev::launch",
        })?;
        Ok(())
    }

    // ─── T246.8 A2 — INDEXED MoE FFN dispatch wrappers ───────────────────

    /// Indexed Q4_K v3 SGEMV : `y = expert_ptrs[topk_indices[slot]] @ x`.
    /// `expert_ptrs` is a device-resident `[n_experts]` array of `u64`
    /// pointers to per-expert Q4_K weight buffers ; `topk_indices` is a
    /// device-resident `[k]` array (output of `topk_softmax_bf16`).
    ///
    /// # Safety  `expert_ptrs[i]` must point to a Q4_K block of `(K/256)*144`
    /// bytes. `topk_indices[slot] ∈ [0, n_experts)`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn sgemv_q4k_bf16_v3_indexed(
        &self,
        stream: &Arc<CudaStream>,
        expert_ptrs: u64,
        topk_indices: u64,
        slot: i32,
        x: u64,
        y: u64,
        n: i32,
        k: i32,
    ) -> Result<(), CudaError> {
        if k % 256 != 0 {
            return Err(CudaError::Unsupported {
                msg: format!("sgemv_q4k_bf16_v3_indexed: K={k} must be multiple of 256"),
            });
        }
        let (_module, func) = self.compile_or_get(
            &self.sgemv_q4k_v3_indexed,
            SGEMV_Q4K_BF16_V3_INDEXED_SRC,
            "sgemv_q4k_bf16_v3_indexed",
        )?;
        const ROWS_PER_BLOCK: i32 = 4;
        let n_blocks = ((n + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK) as u32;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (n_blocks, 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut launcher = stream.launch_builder(&func);
        launcher
            .arg(&expert_ptrs)
            .arg(&topk_indices)
            .arg(&slot)
            .arg(&x)
            .arg(&y)
            .arg(&n)
            .arg(&k);
        launcher.launch(cfg).map_err(|e| CudaError::Driver {
            code: format!("{e:?}").len() as i32,
            location: "sgemv_q4k_bf16_v3_indexed::launch",
        })?;
        Ok(())
    }

    /// Indexed Q4_K dp4a SGEMV : `y = expert_ptrs[topk_indices[slot]] @ x_q8_1`.
    ///
    /// # Safety  Same as `sgemv_q4k_bf16_v3_indexed` for `expert_ptrs` and
    /// `topk_indices`. `x_q8_1` is the Q8_1-quantized activation row
    /// (`(K/32)*36` bytes), shared across all experts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn sgemv_q4k_q8_1_dp4a_bf16_indexed(
        &self,
        stream: &Arc<CudaStream>,
        expert_ptrs: u64,
        topk_indices: u64,
        slot: i32,
        x_q8_1: u64,
        y: u64,
        n: i32,
        k: i32,
    ) -> Result<(), CudaError> {
        if k % 256 != 0 {
            return Err(CudaError::Unsupported {
                msg: format!("sgemv_q4k_q8_1_dp4a_bf16_indexed: K={k} must be multiple of 256"),
            });
        }
        let (_module, func) = self.compile_or_get(
            &self.sgemv_q4k_q8_1_dp4a_indexed,
            SGEMV_Q4K_Q8_1_DP4A_BF16_INDEXED_SRC,
            "sgemv_q4k_q8_1_dp4a_bf16_indexed",
        )?;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (n as u32, 1, 1),
            block_dim: (32, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut launcher = stream.launch_builder(&func);
        launcher
            .arg(&expert_ptrs)
            .arg(&topk_indices)
            .arg(&slot)
            .arg(&x_q8_1)
            .arg(&y)
            .arg(&n)
            .arg(&k);
        launcher.launch(cfg).map_err(|e| CudaError::Driver {
            code: format!("{e:?}").len() as i32,
            location: "sgemv_q4k_q8_1_dp4a_bf16_indexed::launch",
        })?;
        Ok(())
    }

    /// Indexed Q5_K v3 SGEMV.
    ///
    /// # Safety  Same as `sgemv_q4k_bf16_v3_indexed` (Q5_K layout
    /// = `(K/256)*176` bytes per expert).
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn sgemv_q5k_bf16_v3_indexed(
        &self,
        stream: &Arc<CudaStream>,
        expert_ptrs: u64,
        topk_indices: u64,
        slot: i32,
        x: u64,
        y: u64,
        n: i32,
        k: i32,
    ) -> Result<(), CudaError> {
        if k % 256 != 0 {
            return Err(CudaError::Unsupported {
                msg: format!("sgemv_q5k_bf16_v3_indexed: K={k} must be multiple of 256"),
            });
        }
        let (_module, func) = self.compile_or_get(
            &self.sgemv_q5k_v3_indexed,
            SGEMV_Q5K_BF16_V3_INDEXED_SRC,
            "sgemv_q5k_bf16_v3_indexed",
        )?;
        const ROWS_PER_BLOCK: i32 = 4;
        let n_blocks = ((n + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK) as u32;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (n_blocks, 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut launcher = stream.launch_builder(&func);
        launcher
            .arg(&expert_ptrs)
            .arg(&topk_indices)
            .arg(&slot)
            .arg(&x)
            .arg(&y)
            .arg(&n)
            .arg(&k);
        launcher.launch(cfg).map_err(|e| CudaError::Driver {
            code: format!("{e:?}").len() as i32,
            location: "sgemv_q5k_bf16_v3_indexed::launch",
        })?;
        Ok(())
    }

    /// Indexed Q6_K v3 SGEMV.
    ///
    /// # Safety  Same as `sgemv_q4k_bf16_v3_indexed` (Q6_K layout
    /// = `(K/256)*210` bytes per expert).
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn sgemv_q6k_bf16_v3_indexed(
        &self,
        stream: &Arc<CudaStream>,
        expert_ptrs: u64,
        topk_indices: u64,
        slot: i32,
        x: u64,
        y: u64,
        n: i32,
        k: i32,
    ) -> Result<(), CudaError> {
        if k % 256 != 0 {
            return Err(CudaError::Unsupported {
                msg: format!("sgemv_q6k_bf16_v3_indexed: K={k} must be multiple of 256"),
            });
        }
        let (_module, func) = self.compile_or_get(
            &self.sgemv_q6k_v3_indexed,
            SGEMV_Q6K_BF16_V3_INDEXED_SRC,
            "sgemv_q6k_bf16_v3_indexed",
        )?;
        const ROWS_PER_BLOCK: i32 = 4;
        let n_blocks = ((n + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK) as u32;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (n_blocks, 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut launcher = stream.launch_builder(&func);
        launcher
            .arg(&expert_ptrs)
            .arg(&topk_indices)
            .arg(&slot)
            .arg(&x)
            .arg(&y)
            .arg(&n)
            .arg(&k);
        launcher.launch(cfg).map_err(|e| CudaError::Driver {
            code: format!("{e:?}").len() as i32,
            location: "sgemv_q6k_bf16_v3_indexed::launch",
        })?;
        Ok(())
    }

    /// Indexed BF16 SGEMV.
    ///
    /// # Safety  Expert weights are `[N, K]` row-major BF16. See
    /// `sgemv_q4k_bf16_v3_indexed` for the `expert_ptrs` / `topk_indices`
    /// contract.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn sgemv_bf16_bf16_indexed(
        &self,
        stream: &Arc<CudaStream>,
        expert_ptrs: u64,
        topk_indices: u64,
        slot: i32,
        x: u64,
        y: u64,
        n: i32,
        k: i32,
    ) -> Result<(), CudaError> {
        if k % 256 != 0 {
            return Err(CudaError::Unsupported {
                msg: format!("sgemv_bf16_bf16_indexed: K={k} must be multiple of 256"),
            });
        }
        let (_module, func) = self.compile_or_get(
            &self.sgemv_bf16_indexed,
            SGEMV_BF16_BF16_INDEXED_SRC,
            "sgemv_bf16_bf16_indexed",
        )?;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (n as u32, 1, 1),
            block_dim: (64, 1, 1),
            shared_mem_bytes: 64 * 4,
        };
        let mut launcher = stream.launch_builder(&func);
        launcher
            .arg(&expert_ptrs)
            .arg(&topk_indices)
            .arg(&slot)
            .arg(&x)
            .arg(&y)
            .arg(&n)
            .arg(&k);
        launcher.launch(cfg).map_err(|e| CudaError::Driver {
            code: format!("{e:?}").len() as i32,
            location: "sgemv_bf16_bf16_indexed::launch",
        })?;
        Ok(())
    }

    // ────────────────────────────────────────────────────────────────────
    // T246.8 A4 — mul_mm_id mega-kernel wrappers
    //
    // Each wrapper replaces a `for slot in 0..k_used` loop of the
    // corresponding `sgemv_*_indexed` call with a single launch whose
    // grid covers all (row_tile, slot) pairs. Output `y` has shape
    // `[k_used, N]` BF16 (slot-major). Per-slot row body is byte-for-byte
    // the same as the indexed kernel — A3 demonstrated reduction-order
    // changes break model decode bit-parity even when synthetic parity holds.

    /// Mega-kernel : `y[slot, :] = expert[topk_indices[slot]] @ x` for all
    /// `slot in 0..k_used` in one launch.
    ///
    /// # Safety  See `sgemv_q4k_bf16_v3_indexed` — `expert_ptrs` is a
    /// device array of `n_experts` u64 base pointers, `topk_indices` is a
    /// device array of `k_used` i32 expert IDs, `y` must be at least
    /// `k_used * N` BF16 elements.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn mul_mm_id_q4_k_bf16(
        &self,
        stream: &Arc<CudaStream>,
        expert_ptrs: u64,
        topk_indices: u64,
        x: u64,
        y: u64,
        n: i32,
        k: i32,
        k_used: i32,
    ) -> Result<(), CudaError> {
        if k % 256 != 0 {
            return Err(CudaError::Unsupported {
                msg: format!("mul_mm_id_q4_k_bf16: K={k} must be multiple of 256"),
            });
        }
        let (_module, func) = self.compile_or_get(
            &self.mul_mm_id_q4_k_bf16,
            MUL_MM_ID_Q4_K_BF16_SRC,
            "mul_mm_id_q4_k_bf16",
        )?;
        const ROWS_PER_BLOCK: i32 = 4;
        let n_blocks = ((n + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK) as u32;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (n_blocks, k_used as u32, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut launcher = stream.launch_builder(&func);
        launcher
            .arg(&expert_ptrs)
            .arg(&topk_indices)
            .arg(&x)
            .arg(&y)
            .arg(&n)
            .arg(&k);
        launcher.launch(cfg).map_err(|e| CudaError::Driver {
            code: format!("{e:?}").len() as i32,
            location: "mul_mm_id_q4_k_bf16::launch",
        })?;
        Ok(())
    }

    /// Q4_K dp4a mega-kernel : `y[slot, :] = expert[topk[slot]] @ x_q8_1`.
    ///
    /// # Safety  See `sgemv_q4k_q8_1_dp4a_bf16_indexed`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn mul_mm_id_q4_k_q8_1_dp4a_bf16(
        &self,
        stream: &Arc<CudaStream>,
        expert_ptrs: u64,
        topk_indices: u64,
        x_q8_1: u64,
        y: u64,
        n: i32,
        k: i32,
        k_used: i32,
    ) -> Result<(), CudaError> {
        if k % 256 != 0 {
            return Err(CudaError::Unsupported {
                msg: format!("mul_mm_id_q4_k_q8_1_dp4a_bf16: K={k} must be multiple of 256"),
            });
        }
        let (_module, func) = self.compile_or_get(
            &self.mul_mm_id_q4_k_q8_1_dp4a_bf16,
            MUL_MM_ID_Q4_K_Q8_1_DP4A_BF16_SRC,
            "mul_mm_id_q4_k_q8_1_dp4a_bf16",
        )?;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (n as u32, k_used as u32, 1),
            block_dim: (32, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut launcher = stream.launch_builder(&func);
        launcher
            .arg(&expert_ptrs)
            .arg(&topk_indices)
            .arg(&x_q8_1)
            .arg(&y)
            .arg(&n)
            .arg(&k);
        launcher.launch(cfg).map_err(|e| CudaError::Driver {
            code: format!("{e:?}").len() as i32,
            location: "mul_mm_id_q4_k_q8_1_dp4a_bf16::launch",
        })?;
        Ok(())
    }

    /// Q5_K mega-kernel.
    ///
    /// # Safety  See `sgemv_q5k_bf16_v3_indexed`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn mul_mm_id_q5_k_bf16(
        &self,
        stream: &Arc<CudaStream>,
        expert_ptrs: u64,
        topk_indices: u64,
        x: u64,
        y: u64,
        n: i32,
        k: i32,
        k_used: i32,
    ) -> Result<(), CudaError> {
        if k % 256 != 0 {
            return Err(CudaError::Unsupported {
                msg: format!("mul_mm_id_q5_k_bf16: K={k} must be multiple of 256"),
            });
        }
        let (_module, func) = self.compile_or_get(
            &self.mul_mm_id_q5_k_bf16,
            MUL_MM_ID_Q5_K_BF16_SRC,
            "mul_mm_id_q5_k_bf16",
        )?;
        const ROWS_PER_BLOCK: i32 = 4;
        let n_blocks = ((n + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK) as u32;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (n_blocks, k_used as u32, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut launcher = stream.launch_builder(&func);
        launcher
            .arg(&expert_ptrs)
            .arg(&topk_indices)
            .arg(&x)
            .arg(&y)
            .arg(&n)
            .arg(&k);
        launcher.launch(cfg).map_err(|e| CudaError::Driver {
            code: format!("{e:?}").len() as i32,
            location: "mul_mm_id_q5_k_bf16::launch",
        })?;
        Ok(())
    }

    /// Q6_K mega-kernel.
    ///
    /// # Safety  See `sgemv_q6k_bf16_v3_indexed`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn mul_mm_id_q6_k_bf16(
        &self,
        stream: &Arc<CudaStream>,
        expert_ptrs: u64,
        topk_indices: u64,
        x: u64,
        y: u64,
        n: i32,
        k: i32,
        k_used: i32,
    ) -> Result<(), CudaError> {
        if k % 256 != 0 {
            return Err(CudaError::Unsupported {
                msg: format!("mul_mm_id_q6_k_bf16: K={k} must be multiple of 256"),
            });
        }
        let (_module, func) = self.compile_or_get(
            &self.mul_mm_id_q6_k_bf16,
            MUL_MM_ID_Q6_K_BF16_SRC,
            "mul_mm_id_q6_k_bf16",
        )?;
        const ROWS_PER_BLOCK: i32 = 4;
        let n_blocks = ((n + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK) as u32;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (n_blocks, k_used as u32, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut launcher = stream.launch_builder(&func);
        launcher
            .arg(&expert_ptrs)
            .arg(&topk_indices)
            .arg(&x)
            .arg(&y)
            .arg(&n)
            .arg(&k);
        launcher.launch(cfg).map_err(|e| CudaError::Driver {
            code: format!("{e:?}").len() as i32,
            location: "mul_mm_id_q6_k_bf16::launch",
        })?;
        Ok(())
    }

    /// BF16 mega-kernel.
    ///
    /// # Safety  See `sgemv_bf16_bf16_indexed` — expert weights are `[N, K]`
    /// row-major BF16.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn mul_mm_id_bf16_bf16(
        &self,
        stream: &Arc<CudaStream>,
        expert_ptrs: u64,
        topk_indices: u64,
        x: u64,
        y: u64,
        n: i32,
        k: i32,
        k_used: i32,
    ) -> Result<(), CudaError> {
        if k % 256 != 0 {
            return Err(CudaError::Unsupported {
                msg: format!("mul_mm_id_bf16_bf16: K={k} must be multiple of 256"),
            });
        }
        let (_module, func) = self.compile_or_get(
            &self.mul_mm_id_bf16_bf16,
            MUL_MM_ID_BF16_BF16_SRC,
            "mul_mm_id_bf16_bf16",
        )?;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (n as u32, k_used as u32, 1),
            block_dim: (64, 1, 1),
            shared_mem_bytes: 64 * 4,
        };
        let mut launcher = stream.launch_builder(&func);
        launcher
            .arg(&expert_ptrs)
            .arg(&topk_indices)
            .arg(&x)
            .arg(&y)
            .arg(&n)
            .arg(&k);
        launcher.launch(cfg).map_err(|e| CudaError::Driver {
            code: format!("{e:?}").len() as i32,
            location: "mul_mm_id_bf16_bf16::launch",
        })?;
        Ok(())
    }

    // ──────────────────────────────────────────────────────────────────
    // T246.10 TrackE.2 — Group-GEMM (M-variable) launch wrappers.
    //
    // Each wrapper extends the corresponding M=1 `mul_mm_id_*` to a
    // 3D-grid launch covering ALL `(token, slot)` pairs in one call.
    // Per-token rows of `x` and `topk_indices` are addressed by `blockIdx.z`.
    // Output `y` is `[M, k_used, N]` BF16, contiguous in the last dim.
    // ──────────────────────────────────────────────────────────────────

    /// Q4_K group-GEMM : `y[m, slot, :] = expert[topk[m, slot]] @ x[m, :]`.
    ///
    /// # Safety  See `mul_mm_id_q4_k_bf16` — `expert_ptrs[i]` must point
    /// to a `[N, K]` row-major Q4_K block array. `topk_indices` is `[M, k_used]`
    /// i32 row-major. `x` is `[M, K]` BF16 row-major.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn mul_mm_id_gemm_q4_k_bf16(
        &self,
        stream: &Arc<CudaStream>,
        expert_ptrs: u64,
        topk_indices: u64,
        x: u64,
        y: u64,
        m: i32,
        n: i32,
        k: i32,
        k_used: i32,
    ) -> Result<(), CudaError> {
        if k % 256 != 0 {
            return Err(CudaError::Unsupported {
                msg: format!("mul_mm_id_gemm_q4_k_bf16: K={k} must be multiple of 256"),
            });
        }
        if m <= 0 || n <= 0 || k_used <= 0 {
            return Err(CudaError::Unsupported {
                msg: format!("mul_mm_id_gemm_q4_k_bf16: M={m} N={n} k_used={k_used} must be > 0"),
            });
        }
        let (_module, func) = self.compile_or_get(
            &self.mul_mm_id_gemm_q4_k_bf16,
            MUL_MM_ID_GEMM_Q4_K_BF16_SRC,
            "mul_mm_id_gemm_q4_k_bf16",
        )?;
        const ROWS_PER_BLOCK: i32 = 4;
        let n_blocks = ((n + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK) as u32;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (n_blocks, k_used as u32, m as u32),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut launcher = stream.launch_builder(&func);
        launcher
            .arg(&expert_ptrs)
            .arg(&topk_indices)
            .arg(&x)
            .arg(&y)
            .arg(&n)
            .arg(&k)
            .arg(&k_used);
        launcher.launch(cfg).map_err(|e| CudaError::Driver {
            code: format!("{e:?}").len() as i32,
            location: "mul_mm_id_gemm_q4_k_bf16::launch",
        })?;
        Ok(())
    }

    /// Q4_K dp4a group-GEMM (input pre-quantized to Q8_1).
    ///
    /// # Safety  `x_q8_1` is `[M, K/32 * 36]` u8 row-major (per-token Q8_1
    /// super-blocks). Other contract identical to `mul_mm_id_gemm_q4_k_bf16`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn mul_mm_id_gemm_q4_k_q8_1_dp4a_bf16(
        &self,
        stream: &Arc<CudaStream>,
        expert_ptrs: u64,
        topk_indices: u64,
        x_q8_1: u64,
        y: u64,
        m: i32,
        n: i32,
        k: i32,
        k_used: i32,
    ) -> Result<(), CudaError> {
        if k % 256 != 0 {
            return Err(CudaError::Unsupported {
                msg: format!("mul_mm_id_gemm_q4_k_q8_1_dp4a_bf16: K={k} must be multiple of 256"),
            });
        }
        if m <= 0 || n <= 0 || k_used <= 0 {
            return Err(CudaError::Unsupported {
                msg: format!(
                    "mul_mm_id_gemm_q4_k_q8_1_dp4a_bf16: M={m} N={n} k_used={k_used} must be > 0"
                ),
            });
        }
        let (_module, func) = self.compile_or_get(
            &self.mul_mm_id_gemm_q4_k_q8_1_dp4a_bf16,
            MUL_MM_ID_GEMM_Q4_K_Q8_1_DP4A_BF16_SRC,
            "mul_mm_id_gemm_q4_k_q8_1_dp4a_bf16",
        )?;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (n as u32, k_used as u32, m as u32),
            block_dim: (32, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut launcher = stream.launch_builder(&func);
        launcher
            .arg(&expert_ptrs)
            .arg(&topk_indices)
            .arg(&x_q8_1)
            .arg(&y)
            .arg(&n)
            .arg(&k)
            .arg(&k_used);
        launcher.launch(cfg).map_err(|e| CudaError::Driver {
            code: format!("{e:?}").len() as i32,
            location: "mul_mm_id_gemm_q4_k_q8_1_dp4a_bf16::launch",
        })?;
        Ok(())
    }

    /// Q5_K group-GEMM.
    ///
    /// # Safety  See `mul_mm_id_q5_k_bf16` — expert weights `[N, K]` Q5_K.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn mul_mm_id_gemm_q5_k_bf16(
        &self,
        stream: &Arc<CudaStream>,
        expert_ptrs: u64,
        topk_indices: u64,
        x: u64,
        y: u64,
        m: i32,
        n: i32,
        k: i32,
        k_used: i32,
    ) -> Result<(), CudaError> {
        if k % 256 != 0 {
            return Err(CudaError::Unsupported {
                msg: format!("mul_mm_id_gemm_q5_k_bf16: K={k} must be multiple of 256"),
            });
        }
        if m <= 0 || n <= 0 || k_used <= 0 {
            return Err(CudaError::Unsupported {
                msg: format!("mul_mm_id_gemm_q5_k_bf16: M={m} N={n} k_used={k_used} must be > 0"),
            });
        }
        let (_module, func) = self.compile_or_get(
            &self.mul_mm_id_gemm_q5_k_bf16,
            MUL_MM_ID_GEMM_Q5_K_BF16_SRC,
            "mul_mm_id_gemm_q5_k_bf16",
        )?;
        const ROWS_PER_BLOCK: i32 = 4;
        let n_blocks = ((n + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK) as u32;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (n_blocks, k_used as u32, m as u32),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut launcher = stream.launch_builder(&func);
        launcher
            .arg(&expert_ptrs)
            .arg(&topk_indices)
            .arg(&x)
            .arg(&y)
            .arg(&n)
            .arg(&k)
            .arg(&k_used);
        launcher.launch(cfg).map_err(|e| CudaError::Driver {
            code: format!("{e:?}").len() as i32,
            location: "mul_mm_id_gemm_q5_k_bf16::launch",
        })?;
        Ok(())
    }

    /// Q6_K group-GEMM.
    ///
    /// # Safety  See `mul_mm_id_q6_k_bf16` — expert weights `[N, K]` Q6_K.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn mul_mm_id_gemm_q6_k_bf16(
        &self,
        stream: &Arc<CudaStream>,
        expert_ptrs: u64,
        topk_indices: u64,
        x: u64,
        y: u64,
        m: i32,
        n: i32,
        k: i32,
        k_used: i32,
    ) -> Result<(), CudaError> {
        if k % 256 != 0 {
            return Err(CudaError::Unsupported {
                msg: format!("mul_mm_id_gemm_q6_k_bf16: K={k} must be multiple of 256"),
            });
        }
        if m <= 0 || n <= 0 || k_used <= 0 {
            return Err(CudaError::Unsupported {
                msg: format!("mul_mm_id_gemm_q6_k_bf16: M={m} N={n} k_used={k_used} must be > 0"),
            });
        }
        let (_module, func) = self.compile_or_get(
            &self.mul_mm_id_gemm_q6_k_bf16,
            MUL_MM_ID_GEMM_Q6_K_BF16_SRC,
            "mul_mm_id_gemm_q6_k_bf16",
        )?;
        const ROWS_PER_BLOCK: i32 = 4;
        let n_blocks = ((n + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK) as u32;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (n_blocks, k_used as u32, m as u32),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut launcher = stream.launch_builder(&func);
        launcher
            .arg(&expert_ptrs)
            .arg(&topk_indices)
            .arg(&x)
            .arg(&y)
            .arg(&n)
            .arg(&k)
            .arg(&k_used);
        launcher.launch(cfg).map_err(|e| CudaError::Driver {
            code: format!("{e:?}").len() as i32,
            location: "mul_mm_id_gemm_q6_k_bf16::launch",
        })?;
        Ok(())
    }

    /// BF16 group-GEMM.
    ///
    /// # Safety  See `mul_mm_id_bf16_bf16` — expert weights `[N, K]` BF16.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn mul_mm_id_gemm_bf16_bf16(
        &self,
        stream: &Arc<CudaStream>,
        expert_ptrs: u64,
        topk_indices: u64,
        x: u64,
        y: u64,
        m: i32,
        n: i32,
        k: i32,
        k_used: i32,
    ) -> Result<(), CudaError> {
        if k % 256 != 0 {
            return Err(CudaError::Unsupported {
                msg: format!("mul_mm_id_gemm_bf16_bf16: K={k} must be multiple of 256"),
            });
        }
        if m <= 0 || n <= 0 || k_used <= 0 {
            return Err(CudaError::Unsupported {
                msg: format!("mul_mm_id_gemm_bf16_bf16: M={m} N={n} k_used={k_used} must be > 0"),
            });
        }
        let (_module, func) = self.compile_or_get(
            &self.mul_mm_id_gemm_bf16_bf16,
            MUL_MM_ID_GEMM_BF16_BF16_SRC,
            "mul_mm_id_gemm_bf16_bf16",
        )?;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (n as u32, k_used as u32, m as u32),
            block_dim: (64, 1, 1),
            shared_mem_bytes: 64 * 4,
        };
        let mut launcher = stream.launch_builder(&func);
        launcher
            .arg(&expert_ptrs)
            .arg(&topk_indices)
            .arg(&x)
            .arg(&y)
            .arg(&n)
            .arg(&k)
            .arg(&k_used);
        launcher.launch(cfg).map_err(|e| CudaError::Driver {
            code: format!("{e:?}").len() as i32,
            location: "mul_mm_id_gemm_bf16_bf16::launch",
        })?;
        Ok(())
    }

    // ──────────────────────────────────────────────────────────────────
    // TrackE.3 — sort-permutation Group-GEMM helpers + kernels
    //
    // 1. `mm_ids_helper_bf16` builds the compact-by-expert permutation arrays
    //    (`ids_src1`, `ids_dst`, `expert_bounds`) on device. One launch per
    //    layer, grid = `(n_experts, 1, 1)`, 1 warp/block, smem = `M * 4 B`.
    //
    // 2. `mul_mm_id_gemm_q4_k_sorted_bf16` iterates the compact slot index
    //    in gridZ. Adjacent compact_idx values share the same expert, so
    //    consecutive blocks reuse weight tiles in L1/L2 → cache-reuse win.
    // ──────────────────────────────────────────────────────────────────

    /// Build compact-by-expert permutation arrays from `topk_indices`.
    ///
    /// Outputs three device arrays of total length `n_tokens * k_used` (slots)
    /// for the permutation tables, and `n_experts + 1` for the bounds :
    ///
    /// - `ids_src1[c]`     : compact_idx → source token (which row of `x`).
    /// - `ids_dst[c]`      : compact_idx → flat dst row (`token * k_used + slot_orig`).
    /// - `expert_bounds[e]`: start of expert `e`'s slot range in the compact arrays.
    ///                       `expert_bounds[n_experts]` == total active slots.
    ///
    /// # Safety  All output pointers must be valid device allocations of the
    /// expected size. `topk_indices` must be `[n_tokens, k_used]` i32 row-major.
    /// `n_tokens` must fit in 22 bits, `k_used` in 10 bits.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn mm_ids_helper_bf16(
        &self,
        stream: &Arc<CudaStream>,
        topk_indices: u64,
        ids_src1: u64,
        ids_dst: u64,
        expert_bounds: u64,
        n_tokens: i32,
        k_used: i32,
        n_experts: i32,
    ) -> Result<(), CudaError> {
        if n_tokens <= 0 || k_used <= 0 || n_experts <= 0 {
            return Err(CudaError::Unsupported {
                msg: format!(
                    "mm_ids_helper_bf16: n_tokens={n_tokens} k_used={k_used} n_experts={n_experts} must be > 0"
                ),
            });
        }
        if (n_tokens as u32) >= (1u32 << 22) {
            return Err(CudaError::Unsupported {
                msg: format!(
                    "mm_ids_helper_bf16: n_tokens={n_tokens} exceeds 22-bit packing limit"
                ),
            });
        }
        if (k_used as u32) >= (1u32 << 10) {
            return Err(CudaError::Unsupported {
                msg: format!("mm_ids_helper_bf16: k_used={k_used} exceeds 10-bit packing limit"),
            });
        }
        let (_module, func) = self.compile_or_get(
            &self.mm_ids_helper_bf16,
            MM_IDS_HELPER_BF16_SRC,
            "mm_ids_helper_bf16",
        )?;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (n_experts as u32, 1, 1),
            block_dim: (32, 1, 1),
            shared_mem_bytes: (n_tokens as u32) * 4,
        };
        let mut launcher = stream.launch_builder(&func);
        launcher
            .arg(&topk_indices)
            .arg(&ids_src1)
            .arg(&ids_dst)
            .arg(&expert_bounds)
            .arg(&n_tokens)
            .arg(&k_used);
        launcher.launch(cfg).map_err(|e| CudaError::Driver {
            code: format!("{e:?}").len() as i32,
            location: "mm_ids_helper_bf16::launch",
        })?;
        Ok(())
    }

    /// Q4_K sort-permutation Group-GEMM.
    /// `y[token, slot, :] = expert[topk[token, slot]] @ x[token, :]`
    /// — same output layout as `mul_mm_id_gemm_q4_k_bf16`, but the kernel
    /// iterates the COMPACT slot index from `ids_src1` / `ids_dst` produced
    /// by `mm_ids_helper_bf16`. Adjacent compact slots share the SAME expert
    /// → L1/L2 weight tile reuse.
    ///
    /// # Safety  See `mul_mm_id_gemm_q4_k_bf16`. `ids_src1` / `ids_dst` must
    /// have been produced by a prior `mm_ids_helper_bf16` call against the
    /// same `topk_indices` ; `n_slots == m * k_used` is the array length.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn mul_mm_id_gemm_q4_k_sorted_bf16(
        &self,
        stream: &Arc<CudaStream>,
        expert_ptrs: u64,
        topk_indices: u64,
        ids_src1: u64,
        ids_dst: u64,
        x: u64,
        y: u64,
        m: i32,
        n: i32,
        k: i32,
        k_used: i32,
    ) -> Result<(), CudaError> {
        if k % 256 != 0 {
            return Err(CudaError::Unsupported {
                msg: format!("mul_mm_id_gemm_q4_k_sorted_bf16: K={k} must be multiple of 256"),
            });
        }
        if m <= 0 || n <= 0 || k_used <= 0 {
            return Err(CudaError::Unsupported {
                msg: format!(
                    "mul_mm_id_gemm_q4_k_sorted_bf16: M={m} N={n} k_used={k_used} must be > 0"
                ),
            });
        }
        let (_module, func) = self.compile_or_get(
            &self.mul_mm_id_gemm_q4_k_sorted_bf16,
            MUL_MM_ID_GEMM_Q4_K_SORTED_BF16_SRC,
            "mul_mm_id_gemm_q4_k_sorted_bf16",
        )?;
        const ROWS_PER_BLOCK: i32 = 4;
        let n_blocks = ((n + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK) as u32;
        let n_slots = (m as u32) * (k_used as u32);
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (n_blocks, 1, n_slots),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut launcher = stream.launch_builder(&func);
        launcher
            .arg(&expert_ptrs)
            .arg(&topk_indices)
            .arg(&ids_src1)
            .arg(&ids_dst)
            .arg(&x)
            .arg(&y)
            .arg(&n)
            .arg(&k)
            .arg(&k_used);
        launcher.launch(cfg).map_err(|e| CudaError::Driver {
            code: format!("{e:?}").len() as i32,
            location: "mul_mm_id_gemm_q4_k_sorted_bf16::launch",
        })?;
        Ok(())
    }

    /// T246.10 TrackE.4 — Q5_K sort-permutation Group-GEMM.
    ///
    /// Same output layout as `mul_mm_id_gemm_q5_k_bf16` but iterates the
    /// COMPACT slot index from `ids_src1` / `ids_dst` produced by
    /// `mm_ids_helper_bf16`. Adjacent compact slots share the SAME expert →
    /// L1/L2 weight tile reuse.
    ///
    /// # Safety
    ///
    /// See `mul_mm_id_gemm_q5_k_bf16`. `ids_src1` / `ids_dst` must
    /// have been produced by a prior `mm_ids_helper_bf16` call against the
    /// same `topk_indices` ; `n_slots == m * k_used` is the array length.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn mul_mm_id_gemm_q5_k_sorted_bf16(
        &self,
        stream: &Arc<CudaStream>,
        expert_ptrs: u64,
        topk_indices: u64,
        ids_src1: u64,
        ids_dst: u64,
        x: u64,
        y: u64,
        m: i32,
        n: i32,
        k: i32,
        k_used: i32,
    ) -> Result<(), CudaError> {
        if k % 256 != 0 {
            return Err(CudaError::Unsupported {
                msg: format!("mul_mm_id_gemm_q5_k_sorted_bf16: K={k} must be multiple of 256"),
            });
        }
        if m <= 0 || n <= 0 || k_used <= 0 {
            return Err(CudaError::Unsupported {
                msg: format!(
                    "mul_mm_id_gemm_q5_k_sorted_bf16: M={m} N={n} k_used={k_used} must be > 0"
                ),
            });
        }
        let (_module, func) = self.compile_or_get(
            &self.mul_mm_id_gemm_q5_k_sorted_bf16,
            MUL_MM_ID_GEMM_Q5_K_SORTED_BF16_SRC,
            "mul_mm_id_gemm_q5_k_sorted_bf16",
        )?;
        const ROWS_PER_BLOCK: i32 = 4;
        let n_blocks = ((n + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK) as u32;
        let n_slots = (m as u32) * (k_used as u32);
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (n_blocks, 1, n_slots),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut launcher = stream.launch_builder(&func);
        launcher
            .arg(&expert_ptrs)
            .arg(&topk_indices)
            .arg(&ids_src1)
            .arg(&ids_dst)
            .arg(&x)
            .arg(&y)
            .arg(&n)
            .arg(&k)
            .arg(&k_used);
        launcher.launch(cfg).map_err(|e| CudaError::Driver {
            code: format!("{e:?}").len() as i32,
            location: "mul_mm_id_gemm_q5_k_sorted_bf16::launch",
        })?;
        Ok(())
    }

    /// Routed reduce : `y[i] += sum_{slot} alpha_dev[slot] * x[slot, i]`.
    /// Replaces the K-iteration `scaled_add_inplace_bf16_devscalar` epilogue.
    ///
    /// # Safety  `y` is `[N]` BF16 device, `x` is `[k_used, N]` BF16 device,
    /// `alpha_dev` is `[k_used]` BF16 device.
    pub unsafe fn scaled_add_routed_bf16(
        &self,
        stream: &Arc<CudaStream>,
        y: u64,
        x: u64,
        alpha_dev: u64,
        n: i32,
        k_used: i32,
    ) -> Result<(), CudaError> {
        let (_module, func) = self.compile_or_get(
            &self.scaled_add_routed_bf16,
            SCALED_ADD_ROUTED_BF16_SRC,
            "scaled_add_routed_bf16",
        )?;
        let block_dim: u32 = 256;
        let grid_dim: u32 = (n as u32).div_ceil(block_dim);
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (grid_dim, 1, 1),
            block_dim: (block_dim, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut launcher = stream.launch_builder(&func);
        launcher
            .arg(&y)
            .arg(&x)
            .arg(&alpha_dev)
            .arg(&n)
            .arg(&k_used);
        launcher.launch(cfg).map_err(|e| CudaError::Driver {
            code: format!("{e:?}").len() as i32,
            location: "scaled_add_routed_bf16::launch",
        })?;
        Ok(())
    }

    /// `y[i] += alpha_dev[slot] * x[i]`. `alpha_dev` is a device-resident
    /// BF16 vector (top-K weights for routed experts, or 1-elem post-sigmoid
    /// shared-expert weight). Eliminates the host readback that the
    /// existing `scaled_add_inplace_bf16(alpha: f32)` requires.
    ///
    /// # Safety  Same as `scaled_add_inplace_bf16` for y/x/n. `alpha_dev`
    /// is a device pointer to ≥ `slot+1` BF16 elements.
    pub unsafe fn scaled_add_inplace_bf16_devscalar(
        &self,
        stream: &Arc<CudaStream>,
        y: u64,
        x: u64,
        alpha_dev: u64,
        slot: i32,
        n: i32,
    ) -> Result<(), CudaError> {
        let (_module, func) = self.compile_or_get(
            &self.scaled_add_inplace_devscalar,
            SCALED_ADD_INPLACE_BF16_DEVSCALAR_SRC,
            "scaled_add_inplace_bf16_devscalar",
        )?;
        let block_dim: u32 = 256;
        let grid_dim: u32 = (n as u32).div_ceil(block_dim);
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (grid_dim, 1, 1),
            block_dim: (block_dim, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut launcher = stream.launch_builder(&func);
        launcher.arg(&y).arg(&x).arg(&alpha_dev).arg(&slot).arg(&n);
        launcher.launch(cfg).map_err(|e| CudaError::Driver {
            code: format!("{e:?}").len() as i32,
            location: "scaled_add_inplace_bf16_devscalar::launch",
        })?;
        Ok(())
    }

    /// Fused shared-expert sigmoid + scaled_add :
    /// `y[i] += sigmoid((float)dot_bf16[0]) * x[i]`. The sigmoid is
    /// computed in float precision per-thread (NOT rounded to bf16
    /// between sigmoid and multiply). This bit-matches the original
    /// MoE sync path's `host(sigmoid_f32)(... f32 alpha)` precision
    /// for graph-capturable execution.
    ///
    /// # Safety  `y`/`x` length n BF16, `dot_bf16` ≥ 1 BF16.
    pub unsafe fn scaled_add_sigmoid_devscalar_bf16(
        &self,
        stream: &Arc<CudaStream>,
        y: u64,
        x: u64,
        dot_bf16: u64,
        n: i32,
    ) -> Result<(), CudaError> {
        let (_module, func) = self.compile_or_get(
            &self.scaled_add_sigmoid_devscalar,
            SCALED_ADD_SIGMOID_DEVSCALAR_BF16_SRC,
            "scaled_add_sigmoid_devscalar_bf16",
        )?;
        let block_dim: u32 = 256;
        let grid_dim: u32 = (n as u32).div_ceil(block_dim);
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (grid_dim, 1, 1),
            block_dim: (block_dim, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut launcher = stream.launch_builder(&func);
        launcher.arg(&y).arg(&x).arg(&dot_bf16).arg(&n);
        launcher.launch(cfg).map_err(|e| CudaError::Driver {
            code: format!("{e:?}").len() as i32,
            location: "scaled_add_sigmoid_devscalar_bf16::launch",
        })?;
        Ok(())
    }

    /// Zero a BF16 vector of length `n`. # Safety : `y` is `n` BF16 elements.
    pub unsafe fn zero_bf16(
        &self,
        stream: &Arc<CudaStream>,
        y: u64,
        n: i32,
    ) -> Result<(), CudaError> {
        let (_module, func) = self.compile_or_get(&self.zero_bf16, ZERO_BF16_SRC, "zero_bf16")?;
        let block_dim: u32 = 256;
        let grid_dim: u32 = (n as u32).div_ceil(block_dim);
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (grid_dim, 1, 1),
            block_dim: (block_dim, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut launcher = stream.launch_builder(&func);
        launcher.arg(&y).arg(&n);
        launcher.launch(cfg).map_err(|e| CudaError::Driver {
            code: format!("{e:?}").len() as i32,
            location: "zero_bf16::launch",
        })?;
        Ok(())
    }

    // ─────────────────────────────────────────────────────────────────
    // T246.9 NVFP4.2 — NVFP4 SGEMV (vLLM nvfp4-pack-quantized layout)
    // ─────────────────────────────────────────────────────────────────

    /// Indexed NVFP4 SGEMV : `y = expert_ptrs[topk_indices[slot]] @ x`.
    ///
    /// Weight layout per expert :
    /// - `weight_packed [N, K/2]` U8 — 2 FP4 (E2M1) per byte, low nibble = even index
    /// - `weight_scale [N, K/16]` U8 (UE4M3) — 1 byte per micro-block of 16 elements
    /// - `alpha = 1 / (weight_global_scale * input_global_scale)` — applied at the end
    ///
    /// `expert_ptrs` is `[n_experts]` u64 of weight_packed base pointers.
    /// `expert_scale_ptrs` is `[n_experts]` u64 of weight_scale base pointers.
    /// `expert_alphas` is `[n_experts]` f32 of per-expert alpha scalars.
    ///
    /// `K` must be a multiple of 16. `N` is arbitrary.
    ///
    /// # Safety
    /// All `expert_ptrs[*]` and `expert_scale_ptrs[*]` must be valid for
    /// `(N*K/2)` and `(N*K/16)` bytes respectively. `topk_indices[slot]
    /// ∈ [0, n_experts)`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn sgemv_nvfp4_bf16_indexed(
        &self,
        stream: &Arc<CudaStream>,
        expert_ptrs: u64,
        expert_scale_ptrs: u64,
        expert_alphas: u64,
        topk_indices: u64,
        slot: i32,
        x: u64,
        y: u64,
        n: i32,
        k: i32,
    ) -> Result<(), CudaError> {
        if k % 16 != 0 {
            return Err(CudaError::Unsupported {
                msg: format!("sgemv_nvfp4_bf16_indexed: K={k} must be multiple of 16"),
            });
        }
        let (_module, func) = self.compile_or_get(
            &self.sgemv_nvfp4_bf16_indexed,
            SGEMV_NVFP4_BF16_INDEXED_SRC,
            "sgemv_nvfp4_bf16_indexed",
        )?;
        // 1 row per block, 32 threads (1 warp) — like q4k_dp4a_indexed,
        // K-loop split across 32 lanes via stride-32.
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (n as u32, 1, 1),
            block_dim: (32, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut launcher = stream.launch_builder(&func);
        launcher
            .arg(&expert_ptrs)
            .arg(&expert_scale_ptrs)
            .arg(&expert_alphas)
            .arg(&topk_indices)
            .arg(&slot)
            .arg(&x)
            .arg(&y)
            .arg(&n)
            .arg(&k);
        launcher.launch(cfg).map_err(|e| CudaError::Driver {
            code: format!("{e:?}").len() as i32,
            location: "sgemv_nvfp4_bf16_indexed::launch",
        })?;
        Ok(())
    }

    /// Non-indexed NVFP4 SGEMV : `y = W @ x` (single Linear, no MoE
    /// dispatch). Used by attention QKV/O projections and shared-expert
    /// matmuls in `Qwen35ModelCudaNVFP4`.
    ///
    /// # Safety
    /// `weight_packed` is `(N*K/2)` bytes ; `weight_scale` is `(N*K/16)`
    /// bytes ; `alpha = 1 / (weight_global * input_global)` applied at end.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn sgemv_nvfp4_bf16(
        &self,
        stream: &Arc<CudaStream>,
        weight_packed: u64,
        weight_scale: u64,
        alpha: f32,
        x: u64,
        y: u64,
        n: i32,
        k: i32,
    ) -> Result<(), CudaError> {
        if k % 16 != 0 {
            return Err(CudaError::Unsupported {
                msg: format!("sgemv_nvfp4_bf16: K={k} must be multiple of 16"),
            });
        }
        let (_module, func) = self.compile_or_get(
            &self.sgemv_nvfp4_bf16,
            SGEMV_NVFP4_BF16_SRC,
            "sgemv_nvfp4_bf16",
        )?;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (n as u32, 1, 1),
            block_dim: (32, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut launcher = stream.launch_builder(&func);
        launcher
            .arg(&weight_packed)
            .arg(&weight_scale)
            .arg(&alpha)
            .arg(&x)
            .arg(&y)
            .arg(&n)
            .arg(&k);
        launcher.launch(cfg).map_err(|e| CudaError::Driver {
            code: format!("{e:?}").len() as i32,
            location: "sgemv_nvfp4_bf16::launch",
        })?;
        Ok(())
    }

    // ─────────────────────────────────────────────────────────────────────
    // T246.10 MMQ-WHOLESALE — Q4_K × Q8_1 staged kernel + Q8_1 prepass.
    //
    // The pair :
    //   - quantize_mmq_q8_1_bf16_ds4  : BF16 → packed block_q8_1_mmq (144 B)
    //   - mul_mat_q4_k_q8_1_mma       : Q4_K × Q8_1_mmq → BF16 matmul
    //
    // Together they replace the BF16 mma path (TrackG-lite) for prefill MoE
    // FFN matmuls when RUSTORCH_MMQ_WHOLESALE=1 is set. The Q8_1 input cuts
    // memory bandwidth in half vs BF16 (1.125 B/elem vs 2 B/elem). The
    // staged tile_x/tile_y layout matches llama.cpp's MMQ_MMA_TILE_X_K_Q8_1
    // exactly so we can flip the inner dp4a loop to mma.sync s8 later.
    // ─────────────────────────────────────────────────────────────────────

    /// Quantize a BF16 activation tensor to the packed Q8_1 MMQ layout
    /// (block_q8_1_mmq, 144 B per 128-element row segment).
    ///
    /// Layout of `y_q8_1` :  M rows × (K/128) packed blocks × 144 B each.
    /// Each block holds 4 half2 (d, sum) scales + 128 int8 quants.
    ///
    /// Constraint : K % 128 == 0 (= 4 × QK8_1=32 sub-blocks per packed block).
    ///
    /// # Safety
    /// Caller ensures `x` points to `M * K` BF16 elements and `y_q8_1`
    /// points to `M * (K/128) * 144` bytes ; stream is the kernel's CUDA
    /// context.
    pub unsafe fn quantize_mmq_q8_1_bf16_ds4(
        &self,
        stream: &Arc<CudaStream>,
        x: u64,
        y_q8_1: u64,
        m: i32,
        k: i32,
    ) -> Result<(), CudaError> {
        if k % 128 != 0 {
            return Err(CudaError::Unsupported {
                msg: format!("quantize_mmq_q8_1_bf16_ds4: K={k} must be multiple of 128"),
            });
        }
        if m <= 0 || k <= 0 {
            return Err(CudaError::Unsupported {
                msg: format!("quantize_mmq_q8_1_bf16_ds4: M={m} K={k} must be positive"),
            });
        }
        let (_module, func) = self.compile_or_get(
            &self.quantize_mmq_q8_1_ds4,
            QUANTIZE_MMQ_Q8_1_BF16_DS4_SRC,
            "quantize_mmq_q8_1_bf16_ds4",
        )?;
        let block_num_y = ((k as u32) + 4 * 128 - 1) / (4 * 128);
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (m as u32, block_num_y, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut launcher = stream.launch_builder(&func);
        launcher.arg(&x).arg(&y_q8_1).arg(&m).arg(&k);
        launcher.launch(cfg).map_err(|e| CudaError::Driver {
            code: format!("{e:?}").len() as i32,
            location: "quantize_mmq_q8_1_bf16_ds4::launch",
        })?;
        Ok(())
    }

    /// Q4_K × Q8_1_mmq matmul via INT8-staged tile GEMM (port of llama.cpp
    /// `mul_mat_q<Q4_K>` body, mmq_x=64, mmq_y=64, nwarps=4 instantiation).
    ///
    /// Inputs :
    ///   - `w_q4k` : [N, K/256 * 144] Q4_K row-major (W weights)
    ///   - `x_q8_1`: [M, K/128 * 144] packed block_q8_1_mmq (activation, pre-quantized)
    ///   - `y`     : [M, N] BF16 row-major (output)
    ///
    /// Constraint : K % 256 == 0 (= Q4_K super-block size).
    ///
    /// # Safety
    /// Caller ensures pointers are valid device pointers with the indicated
    /// shapes ; stream is the kernel's CUDA context ; M, N, K are positive.
    pub unsafe fn mul_mat_q4_k_q8_1_mma(
        &self,
        stream: &Arc<CudaStream>,
        w_q4k: u64,
        x_q8_1: u64,
        y: u64,
        m: i32,
        n: i32,
        k: i32,
    ) -> Result<(), CudaError> {
        if k % 256 != 0 {
            return Err(CudaError::Unsupported {
                msg: format!("mul_mat_q4_k_q8_1_mma: K={k} must be multiple of 256"),
            });
        }
        if m <= 0 || n <= 0 || k <= 0 {
            return Err(CudaError::Unsupported {
                msg: format!("mul_mat_q4_k_q8_1_mma: M={m} N={n} K={k} must be positive"),
            });
        }
        let (_module, func) = self.compile_or_get(
            &self.mul_mat_q4_k_q8_1_mma,
            MUL_MAT_Q4_K_Q8_1_MMA_SRC,
            "mul_mat_q4_k_q8_1_mma_kernel",
        )?;
        let grid_x = ((n as u32) + 63) / 64;
        let grid_y = ((m as u32) + 63) / 64;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (grid_x, grid_y, 1),
            block_dim: (32, 4, 1),
            shared_mem_bytes: 0,
        };
        let mut launcher = stream.launch_builder(&func);
        launcher
            .arg(&w_q4k)
            .arg(&x_q8_1)
            .arg(&y)
            .arg(&m)
            .arg(&n)
            .arg(&k);
        launcher.launch(cfg).map_err(|e| CudaError::Driver {
            code: format!("{e:?}").len() as i32,
            location: "mul_mat_q4_k_q8_1_mma::launch",
        })?;
        Ok(())
    }

    /// Q4K-MOE-CUBLAS.1 — Q4_K → BF16 device-side dequant.
    ///
    /// Dequantizes a contiguous Q4_K buffer of `n_blocks` 144-byte super-blocks
    /// into a flat BF16 buffer of `n_blocks * 256` elements. Layout is the
    /// natural per-super-block order (sub-block `i` occupies elements
    /// `[i*32, (i+1)*32)`), matching what `dequant_q4_k` in `rustorch-gguf`
    /// produces row-major when called per row.
    ///
    /// For a Q4_K weight tensor `W[N, K]` stored as `N * (K/256)` super-blocks
    /// in row-major order, the output is `W_bf16[N, K]` in the SAME row-major
    /// layout — so cuBLASLt can consume it as `[N, K] ld=K` directly.
    ///
    /// # Safety
    /// `w_q4k_dev` must point to a Q4_K buffer of at least `n_blocks * 144`
    /// bytes ; `out_bf16_dev` must point to at least `n_blocks * 256` BF16
    /// elements.
    pub unsafe fn dequant_q4_k_to_bf16(
        &self,
        stream: &Arc<CudaStream>,
        w_q4k_dev: u64,
        out_bf16_dev: u64,
        n_blocks: i64,
    ) -> Result<(), CudaError> {
        if n_blocks <= 0 {
            return Ok(());
        }
        let (_module, func) = self.compile_or_get(
            &self.dequant_q4_k_to_bf16,
            DEQUANT_Q4_K_TO_BF16_SRC,
            "dequant_q4_k_to_bf16",
        )?;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (n_blocks as u32, 1, 1),
            block_dim: (64, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut launcher = stream.launch_builder(&func);
        launcher.arg(&w_q4k_dev).arg(&out_bf16_dev).arg(&n_blocks);
        launcher.launch(cfg).map_err(|e| CudaError::Driver {
            code: format!("{e:?}").len() as i32,
            location: "dequant_q4_k_to_bf16::launch",
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

    /// T246.10 A6b.1 — sgemm_q4k_bf16_mvar parity test : variable-M SGEMM
    /// must match M sequential M=1 SGEMVs within BF16 tolerance, for several
    /// M values including 1, 8, 15 (partial last tile), 16, 17, 32, 64.
    /// Also verifies the M=8 path is bit-equivalent to sgemm_q4k_bf16_m8.
    #[test]
    fn sgemm_q4k_bf16_mvar_matches_seq_sgemv() {
        use rustorch_gguf::dequant::{Q4_K_BYTES, QK_K};

        let n = 64usize;
        let k = 512usize;
        let blocks_per_row = k / QK_K;
        let row_bytes = blocks_per_row * Q4_K_BYTES;

        let mut state: u64 = 0xa6b15eed;
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
        let kernels = LlmKernels::new(ctx);
        let w_dev = stream.memcpy_stod(&w_q4k_bytes).expect("w");

        for &m_batch in &[1usize, 8, 15, 16, 17, 32, 64] {
            let mut x_all = vec![0.0f32; m_batch * k];
            for m in 0..m_batch {
                for j in 0..k {
                    x_all[m * k + j] = ((m as f32 + 1.0) * (j as f32 * 0.013).sin()) * 0.3;
                }
            }
            let x_bf: Vec<half::bf16> = x_all.iter().copied().map(half::bf16::from_f32).collect();
            let x_dev = stream.memcpy_stod(&x_bf).expect("x");

            // Reference : m_batch sequential M=1 SGEMVs.
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
                for j in 0..n {
                    y_seq[m * n + j] = ym[j];
                }
            }

            // M-variable test.
            let mut y_dev = stream
                .alloc_zeros::<half::bf16>(m_batch * n)
                .expect("y_dev");
            unsafe {
                let (w_p, _g1) = w_dev.device_ptr(&stream);
                let (x_p, _g2) = x_dev.device_ptr(&stream);
                let (y_p, _g3) = y_dev.device_ptr_mut(&stream);
                kernels
                    .sgemm_q4k_bf16_mvar(&stream, w_p, x_p, y_p, m_batch as i32, n as i32, k as i32)
                    .expect("mvar");
            }
            let y_mvar: Vec<half::bf16> = stream.memcpy_dtov(&y_dev).expect("dl mvar");

            for m in 0..m_batch {
                for j in 0..n {
                    let got = y_mvar[m * n + j].to_f32();
                    let want = y_seq[m * n + j].to_f32();
                    let diff = (got - want).abs();
                    let tol = want.abs() * 5e-2 + 1e-2;
                    assert!(
                        diff <= tol,
                        "M={m_batch} m={m} n={j}: got={got} want={want} diff={diff} tol={tol}"
                    );
                }
            }
        }
    }

    /// T246.10 A6b.1 — sgemm_bf16_bf16_mvar parity test.
    #[test]
    fn sgemm_bf16_bf16_mvar_matches_seq_sgemv() {
        let n = 64usize;
        let k = 512usize;

        let mut state: u64 = 0xbf16a6b;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state as u32
        };
        let mut w_f32 = vec![0.0f32; n * k];
        for v in &mut w_f32 {
            *v = ((next() % 1024) as f32 - 512.0) * 0.001;
        }
        let w_bf: Vec<half::bf16> = w_f32.iter().copied().map(half::bf16::from_f32).collect();

        let ctx = CudaContext::new(0).expect("ctx");
        let stream = ctx.default_stream();
        let kernels = LlmKernels::new(ctx);
        let w_dev = stream.memcpy_stod(&w_bf).expect("w");

        for &m_batch in &[1usize, 8, 16, 17, 32] {
            let mut x_all = vec![0.0f32; m_batch * k];
            for m in 0..m_batch {
                for j in 0..k {
                    x_all[m * k + j] = ((m as f32 + 1.0) * (j as f32 * 0.013).sin()) * 0.3;
                }
            }
            let x_bf: Vec<half::bf16> = x_all.iter().copied().map(half::bf16::from_f32).collect();
            let x_dev = stream.memcpy_stod(&x_bf).expect("x");

            // Reference : m_batch sequential M=1 SGEMVs (via sgemv_bf16_bf16).
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
                        .sgemv_bf16_bf16(&stream, w_p, x_p, y_p, n as i32, k as i32)
                        .expect("v1");
                }
                let ym: Vec<half::bf16> = stream.memcpy_dtov(&ym_dev).expect("dl");
                for j in 0..n {
                    y_seq[m * n + j] = ym[j];
                }
            }

            let mut y_dev = stream
                .alloc_zeros::<half::bf16>(m_batch * n)
                .expect("y_dev");
            unsafe {
                let (w_p, _g1) = w_dev.device_ptr(&stream);
                let (x_p, _g2) = x_dev.device_ptr(&stream);
                let (y_p, _g3) = y_dev.device_ptr_mut(&stream);
                kernels
                    .sgemm_bf16_bf16_mvar(
                        &stream,
                        w_p,
                        x_p,
                        y_p,
                        m_batch as i32,
                        n as i32,
                        k as i32,
                    )
                    .expect("mvar");
            }
            let y_mvar: Vec<half::bf16> = stream.memcpy_dtov(&y_dev).expect("dl mvar");

            for m in 0..m_batch {
                for j in 0..n {
                    let got = y_mvar[m * n + j].to_f32();
                    let want = y_seq[m * n + j].to_f32();
                    let diff = (got - want).abs();
                    let tol = want.abs() * 5e-2 + 1e-2;
                    assert!(
                        diff <= tol,
                        "M={m_batch} m={m} n={j}: got={got} want={want} diff={diff} tol={tol}"
                    );
                }
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

    /// T246.6.7 — V3 parity vs V2 (bit-exact-ish since same math, same path).
    #[test]
    #[ignore = "requires CUDA GPU — run with --ignored on DGX"]
    fn sgemv_q4k_v3_matches_v2() {
        use rustorch_gguf::dequant::{Q4_K_BYTES, QK_K};

        let n = 64usize;
        let k = 512usize;
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

        let mut y_v2 = stream.alloc_zeros::<half::bf16>(n).expect("alloc y2");
        let mut y_v3 = stream.alloc_zeros::<half::bf16>(n).expect("alloc y3");
        unsafe {
            let (w_p, _g1) = w_dev.device_ptr(&stream);
            let (x_p, _g2) = x_dev.device_ptr(&stream);
            let (y_p, _g3) = y_v2.device_ptr_mut(&stream);
            kernels
                .sgemv_q4k_bf16_v2(&stream, w_p, x_p, y_p, n as i32, k as i32)
                .expect("v2");
        }
        unsafe {
            let (w_p, _g1) = w_dev.device_ptr(&stream);
            let (x_p, _g2) = x_dev.device_ptr(&stream);
            let (y_p, _g3) = y_v3.device_ptr_mut(&stream);
            kernels
                .sgemv_q4k_bf16_v3(&stream, w_p, x_p, y_p, n as i32, k as i32)
                .expect("v3");
        }
        let v2: Vec<f32> = stream
            .memcpy_dtov(&y_v2)
            .expect("dtov v2")
            .into_iter()
            .map(|b: half::bf16| b.to_f32())
            .collect();
        let v3: Vec<f32> = stream
            .memcpy_dtov(&y_v3)
            .expect("dtov v3")
            .into_iter()
            .map(|b: half::bf16| b.to_f32())
            .collect();

        for i in 0..n {
            let diff = (v2[i] - v3[i]).abs();
            // Same math (per-thread scale instead of shmem broadcast,
            // identical accumulation order within row). Tight tolerance.
            let tol = v2[i].abs() * 1e-2 + 0.1;
            assert!(
                diff <= tol,
                "row[{i}] v2={} v3={} diff={} tol={}",
                v2[i],
                v3[i],
                diff,
                tol
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

    /// T247.1 — RMSNorm backward kernel parity test.
    ///
    /// CPU reference computes (dx, dgamma) from the closed-form gradient ;
    /// CUDA kernel result must match within BF16 quantization tolerance.
    /// This is the pilot proving the training-ready CUDA path : a kernel can
    /// produce numerically correct gradients for an LLM operator on GPU.
    #[test]
    #[ignore = "requires CUDA GPU — run with --ignored on DGX"]
    fn rms_norm_grad_bf16_matches_cpu_reference() {
        let d = 256usize;
        let eps = 1e-6_f32;

        let mut state: u64 = 0xc0ffee;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (((state >> 16) & 0xFFFF) as f32 / 65535.0) - 0.5
        };

        let x_f32: Vec<f32> = (0..d).map(|_| next() * 2.0).collect();
        let g_f32: Vec<f32> = (0..d).map(|_| 0.5 + next() * 0.4).collect();
        let dy_f32: Vec<f32> = (0..d).map(|_| next() * 0.3).collect();

        // CPU reference.
        let mean_sq: f32 = x_f32.iter().map(|v| v * v).sum::<f32>() / d as f32;
        let r = (mean_sq + eps).sqrt();
        let inv_r = 1.0 / r;
        let s_acc: f32 = x_f32
            .iter()
            .zip(&g_f32)
            .zip(&dy_f32)
            .map(|((&xi, &gi), &dyi)| dyi * gi * xi)
            .sum();
        let coeff = s_acc / (d as f32) * inv_r * inv_r * inv_r;
        let dx_ref: Vec<f32> = x_f32
            .iter()
            .zip(&g_f32)
            .zip(&dy_f32)
            .map(|((&xi, &gi), &dyi)| inv_r * dyi * gi - xi * coeff)
            .collect();
        let dg_ref: Vec<f32> = x_f32
            .iter()
            .zip(&dy_f32)
            .map(|(&xi, &dyi)| dyi * xi * inv_r)
            .collect();

        // CUDA path.
        let ctx = CudaContext::new(0).expect("ctx");
        let stream = ctx.default_stream();
        let kernels = LlmKernels::new(ctx);

        let to_bf = |v: &[f32]| -> Vec<half::bf16> {
            v.iter().copied().map(half::bf16::from_f32).collect()
        };
        let x_bf = to_bf(&x_f32);
        let g_bf = to_bf(&g_f32);
        let dy_bf = to_bf(&dy_f32);

        let x_dev = stream.memcpy_stod(&x_bf).expect("upload x");
        let g_dev = stream.memcpy_stod(&g_bf).expect("upload gamma");
        let dy_dev = stream.memcpy_stod(&dy_bf).expect("upload dy");
        let mut dx_dev = stream.alloc_zeros::<half::bf16>(d).expect("alloc dx");
        let mut dg_dev = stream.alloc_zeros::<half::bf16>(d).expect("alloc dgamma");

        unsafe {
            use cudarc::driver::{DevicePtr, DevicePtrMut};
            let (xp, _g0) = x_dev.device_ptr(&stream);
            let (gp, _g1) = g_dev.device_ptr(&stream);
            let (dyp, _g2) = dy_dev.device_ptr(&stream);
            let (dxp, _g3) = dx_dev.device_ptr_mut(&stream);
            let (dgp, _g4) = dg_dev.device_ptr_mut(&stream);
            kernels
                .rms_norm_grad_bf16(&stream, xp, gp, dyp, dxp, dgp, d as i32, eps)
                .expect("rms_norm_grad");
        }

        let dx_cuda: Vec<f32> = stream
            .memcpy_dtov(&dx_dev)
            .expect("dtov dx")
            .into_iter()
            .map(|b: half::bf16| b.to_f32())
            .collect();
        let dg_cuda: Vec<f32> = stream
            .memcpy_dtov(&dg_dev)
            .expect("dtov dgamma")
            .into_iter()
            .map(|b: half::bf16| b.to_f32())
            .collect();

        // Compare. BF16 has ~7 mantissa bits → ~1% rel error. Add small abs
        // slack for very-small reference values (subnormals).
        for i in 0..d {
            let diff = (dx_ref[i] - dx_cuda[i]).abs();
            let tol = dx_ref[i].abs() * 1.5e-2 + 5e-3;
            assert!(
                diff <= tol,
                "dx[{i}] ref={} cuda={} diff={} tol={}",
                dx_ref[i],
                dx_cuda[i],
                diff,
                tol
            );
        }
        for i in 0..d {
            let diff = (dg_ref[i] - dg_cuda[i]).abs();
            let tol = dg_ref[i].abs() * 1.5e-2 + 5e-3;
            assert!(
                diff <= tol,
                "dgamma[{i}] ref={} cuda={} diff={} tol={}",
                dg_ref[i],
                dg_cuda[i],
                diff,
                tol
            );
        }
    }

    /// T247.2 — SwiGLU backward kernel parity test.
    #[test]
    #[ignore = "requires CUDA GPU — run with --ignored on DGX"]
    fn swiglu_grad_bf16_matches_cpu_reference() {
        let n = 1024usize;
        let mut state: u64 = 0xbadc0ffe;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (((state >> 16) & 0xFFFF) as f32 / 65535.0) - 0.5
        };

        let gate_f32: Vec<f32> = (0..n).map(|_| next() * 4.0).collect();
        let up_f32: Vec<f32> = (0..n).map(|_| next() * 4.0).collect();
        let dy_f32: Vec<f32> = (0..n).map(|_| next() * 0.5).collect();

        // CPU reference.
        let mut dgate_ref = vec![0.0_f32; n];
        let mut dup_ref = vec![0.0_f32; n];
        for i in 0..n {
            let g = gate_f32[i];
            let u = up_f32[i];
            let dyi = dy_f32[i];
            let sig = 1.0 / (1.0 + (-g).exp());
            let silu = g * sig;
            let silu_prime = sig + g * sig * (1.0 - sig);
            dgate_ref[i] = dyi * silu_prime * u;
            dup_ref[i] = dyi * silu;
        }

        let ctx = CudaContext::new(0).expect("ctx");
        let stream = ctx.default_stream();
        let kernels = LlmKernels::new(ctx);

        let to_bf = |v: &[f32]| -> Vec<half::bf16> {
            v.iter().copied().map(half::bf16::from_f32).collect()
        };
        let gate_dev = stream.memcpy_stod(&to_bf(&gate_f32)).expect("upload gate");
        let up_dev = stream.memcpy_stod(&to_bf(&up_f32)).expect("upload up");
        let dy_dev = stream.memcpy_stod(&to_bf(&dy_f32)).expect("upload dy");
        let mut dgate_dev = stream.alloc_zeros::<half::bf16>(n).expect("alloc dgate");
        let mut dup_dev = stream.alloc_zeros::<half::bf16>(n).expect("alloc dup");

        unsafe {
            use cudarc::driver::{DevicePtr, DevicePtrMut};
            let (gp, _g0) = gate_dev.device_ptr(&stream);
            let (up_p, _g1) = up_dev.device_ptr(&stream);
            let (dyp, _g2) = dy_dev.device_ptr(&stream);
            let (dgp, _g3) = dgate_dev.device_ptr_mut(&stream);
            let (dup_p, _g4) = dup_dev.device_ptr_mut(&stream);
            kernels
                .swiglu_grad_bf16(&stream, gp, up_p, dyp, dgp, dup_p, n as i32)
                .expect("swiglu_grad");
        }

        let dgate_cuda: Vec<f32> = stream
            .memcpy_dtov(&dgate_dev)
            .expect("dtov dgate")
            .into_iter()
            .map(|b: half::bf16| b.to_f32())
            .collect();
        let dup_cuda: Vec<f32> = stream
            .memcpy_dtov(&dup_dev)
            .expect("dtov dup")
            .into_iter()
            .map(|b: half::bf16| b.to_f32())
            .collect();

        for i in 0..n {
            let diff = (dgate_ref[i] - dgate_cuda[i]).abs();
            let tol = dgate_ref[i].abs() * 1.5e-2 + 5e-3;
            assert!(
                diff <= tol,
                "dgate[{i}] ref={} cuda={} diff={} tol={}",
                dgate_ref[i],
                dgate_cuda[i],
                diff,
                tol
            );
        }
        for i in 0..n {
            let diff = (dup_ref[i] - dup_cuda[i]).abs();
            let tol = dup_ref[i].abs() * 1.5e-2 + 5e-3;
            assert!(
                diff <= tol,
                "dup[{i}] ref={} cuda={} diff={} tol={}",
                dup_ref[i],
                dup_cuda[i],
                diff,
                tol
            );
        }
    }

    /// T246.6.2 — Top-K softmax router parity test.
    #[test]
    #[ignore = "requires CUDA GPU — run with --ignored on DGX"]
    fn topk_softmax_bf16_matches_cpu_reference() {
        let n_experts = 64usize;
        let k = 8usize;

        let mut state: u64 = 0xa3b1234;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (((state >> 16) & 0xFFFF) as f32 / 65535.0) - 0.5
        };
        let scores_f32: Vec<f32> = (0..n_experts).map(|_| next() * 4.0).collect();

        // CPU reference : softmax → top-K → renormalize.
        let max_s = scores_f32.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = scores_f32.iter().map(|&v| (v - max_s).exp()).collect();
        let z: f32 = exps.iter().sum();
        let probs: Vec<f32> = exps.iter().map(|&e| e / z).collect();
        let mut idx_p: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
        idx_p.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        idx_p.truncate(k);
        let sum_topk: f32 = idx_p.iter().map(|(_, p)| *p).sum();
        for (_, p) in idx_p.iter_mut() {
            *p /= sum_topk;
        }
        let indices_ref: Vec<i32> = idx_p.iter().map(|(i, _)| *i as i32).collect();
        let weights_ref: Vec<f32> = idx_p.iter().map(|(_, p)| *p).collect();

        // CUDA path.
        let ctx = CudaContext::new(0).expect("ctx");
        let stream = ctx.default_stream();
        let kernels = LlmKernels::new(ctx);

        let scores_bf: Vec<half::bf16> = scores_f32
            .iter()
            .copied()
            .map(half::bf16::from_f32)
            .collect();
        let scores_dev = stream.memcpy_stod(&scores_bf).expect("upload scores");
        let mut indices_dev = stream.alloc_zeros::<i32>(k).expect("alloc idx");
        let mut weights_dev = stream.alloc_zeros::<half::bf16>(k).expect("alloc w");

        unsafe {
            use cudarc::driver::{DevicePtr, DevicePtrMut};
            let (sp, _g0) = scores_dev.device_ptr(&stream);
            let (ip, _g1) = indices_dev.device_ptr_mut(&stream);
            let (wp, _g2) = weights_dev.device_ptr_mut(&stream);
            kernels
                .topk_softmax_bf16(&stream, sp, ip, wp, n_experts as i32, k as i32)
                .expect("topk");
        }

        let indices_cuda: Vec<i32> = stream.memcpy_dtov(&indices_dev).expect("dtov idx");
        let weights_cuda: Vec<f32> = stream
            .memcpy_dtov(&weights_dev)
            .expect("dtov w")
            .into_iter()
            .map(|b: half::bf16| b.to_f32())
            .collect();

        // Indices should match exactly (same sort).
        assert_eq!(
            indices_cuda, indices_ref,
            "top-K indices mismatch: cuda={:?}, ref={:?}",
            indices_cuda, indices_ref
        );

        // Weights should sum to 1 and match per-element.
        let sum_cuda: f32 = weights_cuda.iter().sum();
        assert!(
            (sum_cuda - 1.0).abs() < 0.02,
            "weights should sum to 1, got {sum_cuda}"
        );
        for i in 0..k {
            let diff = (weights_ref[i] - weights_cuda[i]).abs();
            let tol = weights_ref[i].abs() * 1.5e-2 + 5e-3;
            assert!(
                diff <= tol,
                "weights[{i}] ref={} cuda={} diff={} tol={}",
                weights_ref[i],
                weights_cuda[i],
                diff,
                tol
            );
        }
    }

    /// T246.8 A2.1 — Indexed Q4_K v3 SGEMV parity vs non-indexed v3.
    ///
    /// Build a small "MoE FFN" : 3 experts (Q4_K), each with N=8, K=256.
    /// Pre-cache device-pointer array, fake topk_indices = [2, 0], slot=0
    /// then slot=1. Compare against direct call to `sgemv_q4k_bf16_v3` on
    /// the corresponding expert.
    #[test]
    #[ignore = "requires CUDA GPU — run with --ignored on DGX"]
    fn sgemv_q4k_bf16_v3_indexed_matches_v3() {
        use cudarc::driver::DevicePtrMut;
        let n_experts = 3usize;
        let n = 8usize;
        let k = 256usize;
        let row_bytes = (k / 256) * 144;
        let expert_bytes = n * row_bytes;

        let mut state: u64 = 0xa2_b3_4d_e5;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };

        // Build n_experts independent Q4_K weight blobs.
        let mut all_w: Vec<Vec<u8>> = Vec::with_capacity(n_experts);
        for _ in 0..n_experts {
            let mut w = vec![0u8; expert_bytes];
            for b in &mut w {
                *b = (next() & 0xFF) as u8;
            }
            for row in 0..n {
                for blk in 0..(k / 256) {
                    let off = row * row_bytes + blk * 144;
                    let d = half::f16::from_f32(0.05).to_le_bytes();
                    let dmin = half::f16::from_f32(0.025).to_le_bytes();
                    w[off] = d[0];
                    w[off + 1] = d[1];
                    w[off + 2] = dmin[0];
                    w[off + 3] = dmin[1];
                    for i in 0..12 {
                        w[off + 4 + i] &= 0x3F;
                    }
                }
            }
            all_w.push(w);
        }

        let ctx = CudaContext::new(0).expect("ctx");
        let stream = ctx.default_stream();
        let kernels = LlmKernels::new(ctx);

        // Upload each expert independently → collect device base pointers.
        let mut exp_dev: Vec<cudarc::driver::CudaSlice<u8>> = Vec::new();
        for w in &all_w {
            exp_dev.push(stream.memcpy_stod(w).expect("upload expert"));
        }
        // Build device-pointer array.
        let mut ptrs: Vec<u64> = Vec::with_capacity(n_experts);
        for d in &exp_dev {
            let (p, _g) = d.device_ptr(&stream);
            ptrs.push(p);
        }
        let ptrs_dev = stream.memcpy_stod(&ptrs).expect("upload ptrs");

        // Activation x.
        let x_bf: Vec<half::bf16> = (0..k)
            .map(|i| half::bf16::from_f32((i as f32 * 0.01).sin()))
            .collect();
        let x_dev = stream.memcpy_stod(&x_bf).expect("upload x");

        // Top-K indices (slot 0 → expert 2, slot 1 → expert 0).
        let topk: Vec<i32> = vec![2, 0];
        let topk_dev = stream.memcpy_stod(&topk).expect("upload topk");

        let mut y_ref = stream.alloc_zeros::<half::bf16>(n).expect("alloc y_ref");
        let mut y_idx = stream.alloc_zeros::<half::bf16>(n).expect("alloc y_idx");

        for (slot, &e_idx) in topk.iter().enumerate() {
            // Reference : direct v3 on expert e_idx.
            unsafe {
                let (w_p, _g) = exp_dev[e_idx as usize].device_ptr(&stream);
                let (x_p, _g2) = x_dev.device_ptr(&stream);
                let (y_p, _g3) = y_ref.device_ptr_mut(&stream);
                kernels
                    .sgemv_q4k_bf16_v3(&stream, w_p, x_p, y_p, n as i32, k as i32)
                    .expect("v3");
            }
            // Indexed.
            unsafe {
                let (pp, _g) = ptrs_dev.device_ptr(&stream);
                let (tp, _g2) = topk_dev.device_ptr(&stream);
                let (x_p, _g3) = x_dev.device_ptr(&stream);
                let (y_p, _g4) = y_idx.device_ptr_mut(&stream);
                kernels
                    .sgemv_q4k_bf16_v3_indexed(
                        &stream,
                        pp,
                        tp,
                        slot as i32,
                        x_p,
                        y_p,
                        n as i32,
                        k as i32,
                    )
                    .expect("v3_indexed");
            }
            let r_ref: Vec<half::bf16> = stream.memcpy_dtov(&y_ref).expect("dtov ref");
            let r_idx: Vec<half::bf16> = stream.memcpy_dtov(&y_idx).expect("dtov idx");
            for i in 0..n {
                assert_eq!(
                    r_ref[i].to_bits(),
                    r_idx[i].to_bits(),
                    "slot={slot} (expert {e_idx}) row[{i}] ref={} idx={}",
                    r_ref[i].to_f32(),
                    r_idx[i].to_f32(),
                );
            }
        }
    }

    /// T246.8 A2.1 — Indexed Q4_K dp4a vs non-indexed dp4a parity.
    #[test]
    #[ignore = "requires CUDA GPU — run with --ignored on DGX"]
    fn sgemv_q4k_q8_1_dp4a_bf16_indexed_matches_dp4a() {
        use cudarc::driver::DevicePtrMut;
        let n_experts = 3usize;
        let n = 8usize;
        let k = 256usize;
        let row_bytes = (k / 256) * 144;
        let expert_bytes = n * row_bytes;

        let mut state: u64 = 0x12_34_56_78;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };

        let mut all_w: Vec<Vec<u8>> = Vec::with_capacity(n_experts);
        for _ in 0..n_experts {
            let mut w = vec![0u8; expert_bytes];
            for b in &mut w {
                *b = (next() & 0xFF) as u8;
            }
            for row in 0..n {
                for blk in 0..(k / 256) {
                    let off = row * row_bytes + blk * 144;
                    let d = half::f16::from_f32(0.05).to_le_bytes();
                    let dmin = half::f16::from_f32(0.025).to_le_bytes();
                    w[off] = d[0];
                    w[off + 1] = d[1];
                    w[off + 2] = dmin[0];
                    w[off + 3] = dmin[1];
                    for i in 0..12 {
                        w[off + 4 + i] &= 0x3F;
                    }
                }
            }
            all_w.push(w);
        }

        let ctx = CudaContext::new(0).expect("ctx");
        let stream = ctx.default_stream();
        let kernels = LlmKernels::new(ctx);

        let mut exp_dev: Vec<cudarc::driver::CudaSlice<u8>> = Vec::new();
        for w in &all_w {
            exp_dev.push(stream.memcpy_stod(w).expect("upload expert"));
        }
        let mut ptrs: Vec<u64> = Vec::with_capacity(n_experts);
        for d in &exp_dev {
            let (p, _g) = d.device_ptr(&stream);
            ptrs.push(p);
        }
        let ptrs_dev = stream.memcpy_stod(&ptrs).expect("upload ptrs");

        let x_bf: Vec<half::bf16> = (0..k)
            .map(|i| half::bf16::from_f32((i as f32 * 0.01).sin()))
            .collect();
        let x_dev = stream.memcpy_stod(&x_bf).expect("upload x");

        let mut x_q8 = stream.alloc_zeros::<u8>((k / 32) * 36).expect("q8 alloc");
        unsafe {
            let (xp, _g) = x_dev.device_ptr(&stream);
            let (xq, _g2) = x_q8.device_ptr_mut(&stream);
            kernels
                .quantize_q8_1_bf16(&stream, xp, xq, k as i32)
                .expect("quant q8_1");
        }

        let topk: Vec<i32> = vec![2, 0];
        let topk_dev = stream.memcpy_stod(&topk).expect("upload topk");

        let mut y_ref = stream.alloc_zeros::<half::bf16>(n).expect("alloc y_ref");
        let mut y_idx = stream.alloc_zeros::<half::bf16>(n).expect("alloc y_idx");

        for (slot, &e_idx) in topk.iter().enumerate() {
            unsafe {
                let (w_p, _g) = exp_dev[e_idx as usize].device_ptr(&stream);
                let (xq, _g2) = x_q8.device_ptr(&stream);
                let (y_p, _g3) = y_ref.device_ptr_mut(&stream);
                kernels
                    .sgemv_q4k_q8_1_dp4a_bf16(&stream, w_p, xq, y_p, n as i32, k as i32)
                    .expect("dp4a");
            }
            unsafe {
                let (pp, _g) = ptrs_dev.device_ptr(&stream);
                let (tp, _g2) = topk_dev.device_ptr(&stream);
                let (xq, _g3) = x_q8.device_ptr(&stream);
                let (y_p, _g4) = y_idx.device_ptr_mut(&stream);
                kernels
                    .sgemv_q4k_q8_1_dp4a_bf16_indexed(
                        &stream,
                        pp,
                        tp,
                        slot as i32,
                        xq,
                        y_p,
                        n as i32,
                        k as i32,
                    )
                    .expect("dp4a_indexed");
            }
            let r_ref: Vec<half::bf16> = stream.memcpy_dtov(&y_ref).expect("dtov ref");
            let r_idx: Vec<half::bf16> = stream.memcpy_dtov(&y_idx).expect("dtov idx");
            for i in 0..n {
                assert_eq!(
                    r_ref[i].to_bits(),
                    r_idx[i].to_bits(),
                    "slot={slot} expert={e_idx} row[{i}] ref={} idx={}",
                    r_ref[i].to_f32(),
                    r_idx[i].to_f32(),
                );
            }
        }
    }

    /// T246.8 A2.1 — Indexed Q5_K v3 vs non-indexed Q5_K v3 parity.
    #[test]
    #[ignore = "requires CUDA GPU — run with --ignored on DGX"]
    fn sgemv_q5k_bf16_v3_indexed_matches_v3() {
        use cudarc::driver::DevicePtrMut;
        let n_experts = 3usize;
        let n = 8usize; // we go through v3 path (n_blocks = 2 since 8/4 = 2)
        let k = 256usize;
        let row_bytes = (k / 256) * 176;
        let expert_bytes = n * row_bytes;

        let mut state: u64 = 0xab_cd_ef_10;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };

        let mut all_w: Vec<Vec<u8>> = Vec::with_capacity(n_experts);
        for _ in 0..n_experts {
            let mut w = vec![0u8; expert_bytes];
            for b in &mut w {
                *b = (next() & 0xFF) as u8;
            }
            for row in 0..n {
                for blk in 0..(k / 256) {
                    let off = row * row_bytes + blk * 176;
                    let d = half::f16::from_f32(0.05).to_le_bytes();
                    let dmin = half::f16::from_f32(0.025).to_le_bytes();
                    w[off] = d[0];
                    w[off + 1] = d[1];
                    w[off + 2] = dmin[0];
                    w[off + 3] = dmin[1];
                    for i in 0..12 {
                        w[off + 4 + i] &= 0x3F;
                    }
                }
            }
            all_w.push(w);
        }

        let ctx = CudaContext::new(0).expect("ctx");
        let stream = ctx.default_stream();
        let kernels = LlmKernels::new(ctx);

        let mut exp_dev: Vec<cudarc::driver::CudaSlice<u8>> = Vec::new();
        for w in &all_w {
            exp_dev.push(stream.memcpy_stod(w).expect("upload expert"));
        }
        let mut ptrs: Vec<u64> = Vec::with_capacity(n_experts);
        for d in &exp_dev {
            let (p, _g) = d.device_ptr(&stream);
            ptrs.push(p);
        }
        let ptrs_dev = stream.memcpy_stod(&ptrs).expect("upload ptrs");

        let x_bf: Vec<half::bf16> = (0..k)
            .map(|i| half::bf16::from_f32((i as f32 * 0.01).sin()))
            .collect();
        let x_dev = stream.memcpy_stod(&x_bf).expect("upload x");

        let topk: Vec<i32> = vec![2, 0];
        let topk_dev = stream.memcpy_stod(&topk).expect("upload topk");

        let mut y_ref = stream.alloc_zeros::<half::bf16>(n).expect("alloc y_ref");
        let mut y_idx = stream.alloc_zeros::<half::bf16>(n).expect("alloc y_idx");

        for (slot, &e_idx) in topk.iter().enumerate() {
            unsafe {
                let (w_p, _g) = exp_dev[e_idx as usize].device_ptr(&stream);
                let (x_p, _g2) = x_dev.device_ptr(&stream);
                let (y_p, _g3) = y_ref.device_ptr_mut(&stream);
                kernels
                    .sgemv_q5k_bf16_v3(&stream, w_p, x_p, y_p, n as i32, k as i32)
                    .expect("v3");
            }
            unsafe {
                let (pp, _g) = ptrs_dev.device_ptr(&stream);
                let (tp, _g2) = topk_dev.device_ptr(&stream);
                let (x_p, _g3) = x_dev.device_ptr(&stream);
                let (y_p, _g4) = y_idx.device_ptr_mut(&stream);
                kernels
                    .sgemv_q5k_bf16_v3_indexed(
                        &stream,
                        pp,
                        tp,
                        slot as i32,
                        x_p,
                        y_p,
                        n as i32,
                        k as i32,
                    )
                    .expect("v3_indexed");
            }
            let r_ref: Vec<half::bf16> = stream.memcpy_dtov(&y_ref).expect("dtov ref");
            let r_idx: Vec<half::bf16> = stream.memcpy_dtov(&y_idx).expect("dtov idx");
            for i in 0..n {
                assert_eq!(
                    r_ref[i].to_bits(),
                    r_idx[i].to_bits(),
                    "Q5_K slot={slot} expert={e_idx} row[{i}] ref={} idx={}",
                    r_ref[i].to_f32(),
                    r_idx[i].to_f32(),
                );
            }
        }
    }

    /// T246.8 A2.1 — `scaled_add_inplace_bf16_devscalar` parity vs the
    /// existing host-alpha variant.
    #[test]
    #[ignore = "requires CUDA GPU — run with --ignored on DGX"]
    fn scaled_add_inplace_bf16_devscalar_matches_host_alpha() {
        use cudarc::driver::DevicePtrMut;
        let n = 1024usize;
        let alpha = 0.314_f32;
        let alphas: Vec<half::bf16> = vec![half::bf16::from_f32(0.0), half::bf16::from_f32(alpha)];

        let mut state: u64 = 0xfeed_beef;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (((state >> 16) & 0xFFFF) as f32 / 65535.0) - 0.5
        };
        let y_init: Vec<half::bf16> = (0..n).map(|_| half::bf16::from_f32(next())).collect();
        let x_data: Vec<half::bf16> = (0..n).map(|_| half::bf16::from_f32(next())).collect();

        let ctx = CudaContext::new(0).expect("ctx");
        let stream = ctx.default_stream();
        let kernels = LlmKernels::new(ctx);

        let alphas_dev = stream.memcpy_stod(&alphas).expect("upload alphas");
        let x_dev = stream.memcpy_stod(&x_data).expect("upload x");

        let mut y_host_a = stream.memcpy_stod(&y_init).expect("upload y host");
        let mut y_dev_a = stream.memcpy_stod(&y_init).expect("upload y dev");
        unsafe {
            let (xp, _g0) = x_dev.device_ptr(&stream);
            let (yp_h, _g1) = y_host_a.device_ptr_mut(&stream);
            kernels
                .scaled_add_inplace_bf16(&stream, yp_h, xp, alpha, n as i32)
                .expect("host-alpha");

            let (ap, _g2) = alphas_dev.device_ptr(&stream);
            let (yp_d, _g3) = y_dev_a.device_ptr_mut(&stream);
            kernels
                .scaled_add_inplace_bf16_devscalar(&stream, yp_d, xp, ap, 1, n as i32)
                .expect("devscalar");
        }

        let r_h: Vec<half::bf16> = stream.memcpy_dtov(&y_host_a).expect("dtov h");
        let r_d: Vec<half::bf16> = stream.memcpy_dtov(&y_dev_a).expect("dtov d");
        for i in 0..n {
            // BF16 round-tripping the alpha through device storage may flip
            // ULPs ; allow 1 ULP diff on the result.
            let h = r_h[i].to_f32();
            let d = r_d[i].to_f32();
            let diff = (h - d).abs();
            assert!(
                diff <= h.abs() * 1e-2 + 1e-3,
                "i={i} host={h} dev={d} diff={diff}",
            );
        }
    }

    /// T246.8 A2.1 — `zero_bf16` writes zeros.
    #[test]
    #[ignore = "requires CUDA GPU — run with --ignored on DGX"]
    fn zero_bf16_writes_zeros() {
        use cudarc::driver::DevicePtrMut;
        let n = 1024usize;
        let init: Vec<half::bf16> = (0..n).map(|i| half::bf16::from_f32(i as f32)).collect();

        let ctx = CudaContext::new(0).expect("ctx");
        let stream = ctx.default_stream();
        let kernels = LlmKernels::new(ctx);

        let mut buf = stream.memcpy_stod(&init).expect("upload");
        unsafe {
            let (p, _g) = buf.device_ptr_mut(&stream);
            kernels.zero_bf16(&stream, p, n as i32).expect("zero");
        }
        let r: Vec<half::bf16> = stream.memcpy_dtov(&buf).expect("dtov");
        for (i, v) in r.iter().enumerate() {
            assert_eq!(v.to_f32(), 0.0, "i={i} got {}", v.to_f32());
        }
    }

    /// T247.7 — GQA attention backward parity test.
    ///
    /// Runs a CPU forward (saving m, l, p), a CPU backward (computing
    /// dQ/dK/dV from p, do, Q, K, V), and the CUDA kernel ; compares.
    /// Uses small dims (n_heads=2, n_kv=1, kv_len=4, head_dim=8) so the
    /// CPU reference is hand-traceable.
    #[test]
    #[ignore = "requires CUDA GPU — run with --ignored on DGX"]
    fn gqa_decode_grad_bf16_matches_cpu_reference() {
        let n_heads = 2usize;
        let n_kv = 1usize;
        let kv_len = 4usize;
        let max_seq = 8usize;
        let head_dim = 8usize;
        let scale = 1.0 / (head_dim as f32).sqrt();

        let mut state: u64 = 0xc0ffeecafe;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (((state >> 16) & 0xFFFF) as f32 / 65535.0) - 0.5
        };

        // Random Q [n_heads, head_dim], K/V [n_kv, max_seq, head_dim], do
        let q_f32: Vec<f32> = (0..n_heads * head_dim).map(|_| next()).collect();
        let k_full: Vec<f32> = (0..n_kv * max_seq * head_dim).map(|_| next()).collect();
        let v_full: Vec<f32> = (0..n_kv * max_seq * head_dim).map(|_| next()).collect();
        let do_f32: Vec<f32> = (0..n_heads * head_dim).map(|_| next() * 0.3).collect();

        // Round-trip Q/K/V/do through bf16 for apples-to-apples
        let to_bf = |v: &[f32]| -> Vec<half::bf16> {
            v.iter().copied().map(half::bf16::from_f32).collect()
        };
        let q_bf = to_bf(&q_f32);
        let k_bf = to_bf(&k_full);
        let v_bf = to_bf(&v_full);
        let do_bf = to_bf(&do_f32);
        let q_q: Vec<f32> = q_bf.iter().map(|b| b.to_f32()).collect();
        let k_q: Vec<f32> = k_bf.iter().map(|b| b.to_f32()).collect();
        let v_q: Vec<f32> = v_bf.iter().map(|b| b.to_f32()).collect();
        let do_q: Vec<f32> = do_bf.iter().map(|b| b.to_f32()).collect();

        // ---- CPU forward (compute m, l, p) ----
        let mut m_saved = vec![0.0_f32; n_heads];
        let mut l_saved = vec![0.0_f32; n_heads];
        let mut p_saved = vec![0.0_f32; n_heads * kv_len];
        for h in 0..n_heads {
            let kv_h = h * n_kv / n_heads;
            // Compute scores
            let mut s = vec![0.0_f32; kv_len];
            for t in 0..kv_len {
                let mut dot = 0.0;
                for d in 0..head_dim {
                    dot += q_q[h * head_dim + d] * k_q[(kv_h * max_seq + t) * head_dim + d];
                }
                s[t] = dot * scale;
            }
            let m = s.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum_exp = 0.0;
            let mut p = vec![0.0_f32; kv_len];
            for t in 0..kv_len {
                p[t] = (s[t] - m).exp();
                sum_exp += p[t];
            }
            for t in 0..kv_len {
                p[t] /= sum_exp;
                p_saved[h * kv_len + t] = p[t];
            }
            m_saved[h] = m;
            l_saved[h] = sum_exp;
        }

        // ---- CPU backward reference ----
        let mut dq_ref = vec![0.0_f32; n_heads * head_dim];
        let mut dk_ref = vec![0.0_f32; n_kv * max_seq * head_dim];
        let mut dv_ref = vec![0.0_f32; n_kv * max_seq * head_dim];
        for h in 0..n_heads {
            let kv_h = h * n_kv / n_heads;
            // dp_t = do_h · V_t
            let mut dp = vec![0.0_f32; kv_len];
            for t in 0..kv_len {
                let mut acc = 0.0;
                for d in 0..head_dim {
                    acc += do_q[h * head_dim + d] * v_q[(kv_h * max_seq + t) * head_dim + d];
                }
                dp[t] = acc;
            }
            // D = Σ_t p_t · dp_t
            let big_d: f32 = (0..kv_len).map(|t| p_saved[h * kv_len + t] * dp[t]).sum();
            // ds_t = p_t · (dp_t - D)
            let ds: Vec<f32> = (0..kv_len)
                .map(|t| p_saved[h * kv_len + t] * (dp[t] - big_d))
                .collect();
            // dQ = scale · Σ_t ds_t · K_t
            for d in 0..head_dim {
                let mut acc = 0.0;
                for t in 0..kv_len {
                    acc += ds[t] * k_q[(kv_h * max_seq + t) * head_dim + d];
                }
                dq_ref[h * head_dim + d] = scale * acc;
            }
            // dK_t = scale · ds_t · Q_h    (atomic accumulate across heads sharing kv_h)
            // dV_t = p_t · do_h
            for t in 0..kv_len {
                for d in 0..head_dim {
                    dk_ref[(kv_h * max_seq + t) * head_dim + d] +=
                        scale * ds[t] * q_q[h * head_dim + d];
                    dv_ref[(kv_h * max_seq + t) * head_dim + d] +=
                        p_saved[h * kv_len + t] * do_q[h * head_dim + d];
                }
            }
        }

        // ---- CUDA kernel ----
        let ctx = CudaContext::new(0).expect("ctx");
        let stream = ctx.default_stream();
        let kernels = LlmKernels::new(ctx);

        let q_dev = stream.memcpy_stod(&q_bf).expect("upload q");
        let k_dev = stream.memcpy_stod(&k_bf).expect("upload k");
        let v_dev = stream.memcpy_stod(&v_bf).expect("upload v");
        let m_dev = stream.memcpy_stod(&m_saved).expect("upload m");
        let l_dev = stream.memcpy_stod(&l_saved).expect("upload l");
        let do_dev = stream.memcpy_stod(&do_bf).expect("upload do");
        let mut dq_dev = stream
            .alloc_zeros::<half::bf16>(n_heads * head_dim)
            .expect("alloc dq");
        let mut dk_dev = stream
            .alloc_zeros::<f32>(n_kv * max_seq * head_dim)
            .expect("alloc dk");
        let mut dv_dev = stream
            .alloc_zeros::<f32>(n_kv * max_seq * head_dim)
            .expect("alloc dv");

        unsafe {
            use cudarc::driver::{DevicePtr, DevicePtrMut};
            let (q_p, _g0) = q_dev.device_ptr(&stream);
            let (k_p, _g1) = k_dev.device_ptr(&stream);
            let (v_p, _g2) = v_dev.device_ptr(&stream);
            let (m_p, _g3) = m_dev.device_ptr(&stream);
            let (l_p, _g4) = l_dev.device_ptr(&stream);
            let (do_p, _g5) = do_dev.device_ptr(&stream);
            let (dq_p, _g6) = dq_dev.device_ptr_mut(&stream);
            let (dk_p, _g7) = dk_dev.device_ptr_mut(&stream);
            let (dv_p, _g8) = dv_dev.device_ptr_mut(&stream);
            kernels
                .gqa_decode_grad_bf16(
                    &stream,
                    q_p,
                    k_p,
                    v_p,
                    m_p,
                    l_p,
                    do_p,
                    dq_p,
                    dk_p,
                    dv_p,
                    n_heads as i32,
                    n_kv as i32,
                    kv_len as i32,
                    head_dim as i32,
                    max_seq as i32,
                )
                .expect("gqa_decode_grad");
        }

        let dq_cuda: Vec<f32> = stream
            .memcpy_dtov(&dq_dev)
            .expect("dtov dq")
            .into_iter()
            .map(|b: half::bf16| b.to_f32())
            .collect();
        let dk_cuda: Vec<f32> = stream.memcpy_dtov(&dk_dev).expect("dtov dk");
        let dv_cuda: Vec<f32> = stream.memcpy_dtov(&dv_dev).expect("dtov dv");

        // dQ — bf16 quantized output, accumulate over kv_len terms.
        for i in 0..n_heads * head_dim {
            let diff = (dq_ref[i] - dq_cuda[i]).abs();
            let tol = dq_ref[i].abs() * 5e-2 + 1e-2;
            assert!(
                diff <= tol,
                "dQ[{i}] ref={} cuda={} diff={} tol={}",
                dq_ref[i],
                dq_cuda[i],
                diff,
                tol
            );
        }
        // dK / dV are float accumulators — tighter tolerance.
        for i in 0..n_kv * max_seq * head_dim {
            // Skip slots where t >= kv_len (kernel doesn't touch them).
            let slot_in_kv = i / head_dim;
            let t = slot_in_kv % max_seq;
            if t >= kv_len {
                assert!(dk_cuda[i].abs() < 1e-6, "dK at unused t should be 0");
                assert!(dv_cuda[i].abs() < 1e-6, "dV at unused t should be 0");
                continue;
            }
            let diff_k = (dk_ref[i] - dk_cuda[i]).abs();
            let tol_k = dk_ref[i].abs() * 1e-2 + 1e-3;
            assert!(
                diff_k <= tol_k,
                "dK[{i}] ref={} cuda={} diff={} tol={}",
                dk_ref[i],
                dk_cuda[i],
                diff_k,
                tol_k
            );
            let diff_v = (dv_ref[i] - dv_cuda[i]).abs();
            let tol_v = dv_ref[i].abs() * 1e-2 + 1e-3;
            assert!(
                diff_v <= tol_v,
                "dV[{i}] ref={} cuda={} diff={} tol={}",
                dv_ref[i],
                dv_cuda[i],
                diff_v,
                tol_v
            );
        }
    }

    /// T247.6 — Q4_K backward dx parity test : compares the CUDA kernel to
    /// a CPU reference computed via the official `rustorch_gguf::dequant`
    /// dequantizer + scalar Wᵀ·dy. Tolerance accounts for BF16 + minor
    /// reduction order differences.
    #[test]
    #[ignore = "requires CUDA GPU — run with --ignored on DGX"]
    fn sgemv_q4k_grad_dx_bf16_matches_cpu_reference() {
        use rustorch_gguf::dequant::{dequant_q4_k, Q4_K_BYTES, QK_K};

        let n = 64usize;
        let k = 512usize;
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

        // Random dy. Round-trip through bf16 so the CPU reference sees the
        // same quantized values the kernel reads (the kernel casts bf16→f32
        // internally ; without this round-trip the cumulative bf16-rounding
        // error inflates parity diff at small dx values).
        let dy_f32: Vec<f32> = (0..n)
            .map(|i| half::bf16::from_f32(((i as f32 * 0.13).sin()) * 0.5).to_f32())
            .collect();

        // CPU reference : dequant each row, then dx[k] = Σ_n W[n,k]·dy[n].
        let mut block_buf = [0.0_f32; QK_K];
        let mut w_dequant = vec![0.0_f32; n * k];
        for row in 0..n {
            for blk in 0..blocks_per_row {
                let blk_off = row * row_bytes + blk * Q4_K_BYTES;
                dequant_q4_k(&w_bytes[blk_off..blk_off + Q4_K_BYTES], &mut block_buf)
                    .expect("dequant_q4_k");
                let dst = &mut w_dequant[row * k + blk * QK_K..row * k + (blk + 1) * QK_K];
                dst.copy_from_slice(&block_buf);
            }
        }
        let mut dx_ref = vec![0.0_f32; k];
        for kk in 0..k {
            let mut acc = 0.0_f32;
            for nn in 0..n {
                acc += w_dequant[nn * k + kk] * dy_f32[nn];
            }
            dx_ref[kk] = acc;
        }

        // CUDA path.
        let ctx = CudaContext::new(0).expect("ctx");
        let stream = ctx.default_stream();
        let kernels = LlmKernels::new(ctx);

        let dy_bf: Vec<half::bf16> = dy_f32.iter().copied().map(half::bf16::from_f32).collect();

        let w_dev = stream.memcpy_stod(&w_bytes).expect("upload w");
        let dy_dev = stream.memcpy_stod(&dy_bf).expect("upload dy");
        let mut dx_dev = stream.alloc_zeros::<half::bf16>(k).expect("alloc dx");

        unsafe {
            use cudarc::driver::{DevicePtr, DevicePtrMut};
            let (wp, _g0) = w_dev.device_ptr(&stream);
            let (dyp, _g1) = dy_dev.device_ptr(&stream);
            let (dxp, _g2) = dx_dev.device_ptr_mut(&stream);
            kernels
                .sgemv_q4k_grad_dx_bf16(&stream, wp, dyp, dxp, n as i32, k as i32)
                .expect("sgemv_q4k_grad_dx");
        }

        let dx_cuda: Vec<f32> = stream
            .memcpy_dtov(&dx_dev)
            .expect("dtov dx")
            .into_iter()
            .map(|b: half::bf16| b.to_f32())
            .collect();

        for i in 0..k {
            let diff = (dx_ref[i] - dx_cuda[i]).abs();
            // BF16 quant on dy + W dequant scale path accumulates ~2-4% over
            // a 64-row reduction. Match the established tolerance from the
            // forward parity test `sgemv_q4k_bf16_v2_matches_v1` (5% + 0.5).
            let tol = dx_ref[i].abs() * 5e-2 + 1e-2;
            assert!(
                diff <= tol,
                "dx[{i}] ref={} cuda={} diff={} tol={}",
                dx_ref[i],
                dx_cuda[i],
                diff,
                tol
            );
        }
    }

    /// T247.5 — Embedding lookup backward (atomic scatter) parity test.
    /// Verifies multi-token accumulation : 3 token_ids (one repeated) →
    /// the accumulator should hold dy_a + dy_c at row token_a (since we
    /// reuse it twice) and dy_b at row token_b.
    #[test]
    #[ignore = "requires CUDA GPU — run with --ignored on DGX"]
    fn embedding_lookup_grad_bf16_accumulates() {
        let vocab = 64usize;
        let d = 128usize;

        let mut state: u64 = 0xdeadbeef;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (((state >> 16) & 0xFFFF) as f32 / 65535.0) - 0.5
        };

        let dy_a: Vec<f32> = (0..d).map(|_| next() * 0.5).collect();
        let dy_b: Vec<f32> = (0..d).map(|_| next() * 0.5).collect();
        let dy_c: Vec<f32> = (0..d).map(|_| next() * 0.5).collect();
        let tok_a: i32 = 7;
        let tok_b: i32 = 23;
        let tok_c: i32 = 7; // repeated → accumulates with dy_a

        // CPU reference (full vocab × d).
        let mut accum_ref = vec![0.0_f32; vocab * d];
        for i in 0..d {
            accum_ref[(tok_a as usize) * d + i] += dy_a[i];
            accum_ref[(tok_b as usize) * d + i] += dy_b[i];
            accum_ref[(tok_c as usize) * d + i] += dy_c[i];
        }

        let ctx = CudaContext::new(0).expect("ctx");
        let stream = ctx.default_stream();
        let kernels = LlmKernels::new(ctx);

        let to_bf = |v: &[f32]| -> Vec<half::bf16> {
            v.iter().copied().map(half::bf16::from_f32).collect()
        };
        let dy_a_dev = stream.memcpy_stod(&to_bf(&dy_a)).expect("upload dy_a");
        let dy_b_dev = stream.memcpy_stod(&to_bf(&dy_b)).expect("upload dy_b");
        let dy_c_dev = stream.memcpy_stod(&to_bf(&dy_c)).expect("upload dy_c");
        let mut accum_dev = stream.alloc_zeros::<f32>(vocab * d).expect("alloc accum");

        unsafe {
            use cudarc::driver::{DevicePtr, DevicePtrMut};
            // Each scatter scoped so the `accum_dev` mutable borrow guard
            // drops before the next call (Rust forbids 2 mut borrows
            // simultaneously).
            {
                let (a_p, _g0) = dy_a_dev.device_ptr(&stream);
                let (acc_p, _g1) = accum_dev.device_ptr_mut(&stream);
                kernels
                    .embedding_lookup_grad_bf16(&stream, a_p, tok_a, acc_p, d as i32)
                    .expect("scatter a");
            }
            {
                let (b_p, _g2) = dy_b_dev.device_ptr(&stream);
                let (acc_p, _g3) = accum_dev.device_ptr_mut(&stream);
                kernels
                    .embedding_lookup_grad_bf16(&stream, b_p, tok_b, acc_p, d as i32)
                    .expect("scatter b");
            }
            {
                let (c_p, _g4) = dy_c_dev.device_ptr(&stream);
                let (acc_p, _g5) = accum_dev.device_ptr_mut(&stream);
                kernels
                    .embedding_lookup_grad_bf16(&stream, c_p, tok_c, acc_p, d as i32)
                    .expect("scatter c");
            }
        }

        let accum_cuda: Vec<f32> = stream.memcpy_dtov(&accum_dev).expect("dtov accum");

        // Tolerance accounts for bf16-quant of dy + float accum.
        for i in 0..vocab * d {
            let diff = (accum_ref[i] - accum_cuda[i]).abs();
            let tol = accum_ref[i].abs() * 1.5e-2 + 5e-3;
            assert!(
                diff <= tol,
                "accum[{i}] (vocab_row={}) ref={} cuda={} diff={} tol={}",
                i / d,
                accum_ref[i],
                accum_cuda[i],
                diff,
                tol
            );
        }
    }

    /// T247.4 — Cross-entropy fused forward+backward parity test.
    #[test]
    #[ignore = "requires CUDA GPU — run with --ignored on DGX"]
    fn cross_entropy_loss_grad_bf16_matches_cpu_reference() {
        let vocab = 4096usize;
        let target: i32 = 1234;

        let mut state: u64 = 0xfeed1234;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (((state >> 16) & 0xFFFF) as f32 / 65535.0) - 0.5
        };
        let logits_f32: Vec<f32> = (0..vocab).map(|_| next() * 4.0).collect();

        // CPU reference (numerically stable).
        let max_l = logits_f32.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = logits_f32.iter().map(|&v| (v - max_l).exp()).collect();
        let z: f32 = exps.iter().sum();
        let log_z = z.ln();
        let loss_ref = -(logits_f32[target as usize] - max_l - log_z);
        let dlogits_ref: Vec<f32> = logits_f32
            .iter()
            .enumerate()
            .map(|(i, &v)| {
                let p = (v - max_l).exp() / z;
                p - if i == target as usize { 1.0 } else { 0.0 }
            })
            .collect();

        let ctx = CudaContext::new(0).expect("ctx");
        let stream = ctx.default_stream();
        let kernels = LlmKernels::new(ctx);

        let logits_bf: Vec<half::bf16> = logits_f32
            .iter()
            .copied()
            .map(half::bf16::from_f32)
            .collect();
        let logits_dev = stream.memcpy_stod(&logits_bf).expect("upload logits");
        let mut loss_dev = stream.alloc_zeros::<f32>(1).expect("alloc loss");
        let mut dlogits_dev = stream
            .alloc_zeros::<half::bf16>(vocab)
            .expect("alloc dlogits");

        unsafe {
            use cudarc::driver::{DevicePtr, DevicePtrMut};
            let (lp, _g0) = logits_dev.device_ptr(&stream);
            let (loss_p, _g1) = loss_dev.device_ptr_mut(&stream);
            let (dlp, _g2) = dlogits_dev.device_ptr_mut(&stream);
            kernels
                .cross_entropy_loss_grad_bf16(&stream, lp, target, loss_p, dlp, vocab as i32)
                .expect("xent");
        }

        let loss_cuda: f32 = stream.memcpy_dtov(&loss_dev).expect("dtov loss")[0];
        let dlogits_cuda: Vec<f32> = stream
            .memcpy_dtov(&dlogits_dev)
            .expect("dtov dlogits")
            .into_iter()
            .map(|b: half::bf16| b.to_f32())
            .collect();

        // Loss is a scalar — float32 precision, very tight.
        assert!(
            (loss_ref - loss_cuda).abs() <= loss_ref.abs() * 1e-4 + 1e-4,
            "loss ref={} cuda={} diff={}",
            loss_ref,
            loss_cuda,
            (loss_ref - loss_cuda).abs(),
        );

        // dlogits — BF16 quantized.
        for i in 0..vocab {
            let diff = (dlogits_ref[i] - dlogits_cuda[i]).abs();
            let tol = dlogits_ref[i].abs() * 2e-2 + 5e-3;
            assert!(
                diff <= tol,
                "dlogits[{i}] ref={} cuda={} diff={} tol={}",
                dlogits_ref[i],
                dlogits_cuda[i],
                diff,
                tol
            );
        }
    }

    /// T247.3 — RoPE partial backward kernel parity test.
    #[test]
    #[ignore = "requires CUDA GPU — run with --ignored on DGX"]
    fn rope_partial_grad_bf16_matches_cpu_reference() {
        let n_heads = 4usize;
        let head_dim = 32usize;
        let rope_dim = 32usize;
        let half = rope_dim / 2;
        let pos = 7i32;

        let mut state: u64 = 0xfacefeed;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (((state >> 16) & 0xFFFF) as f32 / 65535.0) - 0.5
        };

        let dy_f32: Vec<f32> = (0..n_heads * head_dim).map(|_| next() * 0.5).collect();

        // inv_freq[k] = 1 / 10000^(2k / rope_dim) — standard RoPE.
        let inv_freq: Vec<f32> = (0..half)
            .map(|k| 1.0 / (10000.0_f32).powf(2.0 * k as f32 / rope_dim as f32))
            .collect();

        // CPU reference: per pair (k, k+half), inverse rotation.
        let mut dx_ref = vec![0.0_f32; n_heads * head_dim];
        for h in 0..n_heads {
            for k in 0..half {
                let theta = inv_freq[k] * pos as f32;
                let (sin_k, cos_k) = theta.sin_cos();
                let row = h * head_dim;
                let dya = dy_f32[row + k];
                let dyb = dy_f32[row + k + half];
                dx_ref[row + k] = dya * cos_k + dyb * sin_k;
                dx_ref[row + k + half] = -dya * sin_k + dyb * cos_k;
            }
        }

        let ctx = CudaContext::new(0).expect("ctx");
        let stream = ctx.default_stream();
        let kernels = LlmKernels::new(ctx);

        let dy_bf: Vec<half::bf16> = dy_f32.iter().copied().map(half::bf16::from_f32).collect();
        let dy_dev = stream.memcpy_stod(&dy_bf).expect("upload dy");
        let if_dev = stream.memcpy_stod(&inv_freq).expect("upload inv_freq");
        let mut dx_dev = stream
            .alloc_zeros::<half::bf16>(n_heads * head_dim)
            .expect("alloc dx");

        unsafe {
            use cudarc::driver::{DevicePtr, DevicePtrMut};
            let (dyp, _g0) = dy_dev.device_ptr(&stream);
            let (ifp, _g1) = if_dev.device_ptr(&stream);
            let (dxp, _g2) = dx_dev.device_ptr_mut(&stream);
            kernels
                .rope_partial_grad_bf16(
                    &stream,
                    dyp,
                    ifp,
                    pos,
                    dxp,
                    n_heads as i32,
                    head_dim as i32,
                    rope_dim as i32,
                )
                .expect("rope_partial_grad");
        }

        let dx_cuda: Vec<f32> = stream
            .memcpy_dtov(&dx_dev)
            .expect("dtov dx")
            .into_iter()
            .map(|b: half::bf16| b.to_f32())
            .collect();

        for i in 0..n_heads * head_dim {
            let diff = (dx_ref[i] - dx_cuda[i]).abs();
            let tol = dx_ref[i].abs() * 1.5e-2 + 5e-3;
            assert!(
                diff <= tol,
                "dx[{i}] ref={} cuda={} diff={} tol={}",
                dx_ref[i],
                dx_cuda[i],
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
