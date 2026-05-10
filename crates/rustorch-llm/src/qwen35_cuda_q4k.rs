//! T246 — `Qwen35ModelCudaQ4K` : full e2e CUDA forward path for Qwen3.5/3.6
//! hybrid (SSM + Attention) Q4_K_M GGUF models.
//!
//! Architecture vs `LlamaModelCudaQ4K` (Qwen2.5) :
//! - Same Q4_K / Q5_K / Q6_K matmul weights stored as raw bytes on GPU.
//! - Hybrid : 16/64 layers are full attention, 48/64 are SSM (gated delta net).
//! - Each block holds a different set of weights depending on its kind.
//!
//! Performance target on Qwen3.6-27B Q4_K_M : >= 11.73 tok/s (llama.cpp tg128
//! parity) with coherent text output, then push past via batched M=N forward.
//!
//! Status (T246.1) : LOADER ONLY. decode_step still returns Err — implemented
//! incrementally in T246.2 (SSM block), T246.3 (attn block), T246.4 (FFN + sample),
//! T246.5 (prefill).

#![cfg(feature = "cuda")]

use crate::qwen35::{LayerKind, Qwen35Config, Qwen35Variant};
use crate::LlmError;
use cudarc::driver::sys::{CUgraphInstantiate_flags, CUstreamCaptureMode};

/// T246.5.7 — number of kv-len splits in the FlashDecode-V2 GQA kernel.
/// 4 gives a 4× boost in attention-block parallelism (32 heads × 4 splits =
/// 128 blocks/layer vs 32 with the serial online kernel) and is fixed at
/// capture time so the staging buffers can be sized once.
const GQA_N_SPLIT: usize = 4;

/// T246.7 P1.3 — maximum draft tree size accepted by `decode_step_tree`.
/// Sized to give headroom over the (W=5, L=5) Jacobi window which yields
/// at most 21 tree nodes (1 + W*(L-1)). 32 leaves room for (W=5, L=7) and
/// future tweaks without re-allocating the tree scratch buffers.
pub const MAX_TREE_SIZE: usize = 32;
use cudarc::driver::{CudaContext, CudaGraph, CudaSlice, CudaStream, PinnedHostSlice};
use rustorch_cuda::cublas_lt::LtSession;
use rustorch_cuda::llm_kernels::LlmKernels;
use rustorch_gguf::reader::GgufFile;
use rustorch_gguf::tensor::GgmlType;
use std::path::Path;
use std::sync::Arc;

/// One matmul weight on GPU. Qwen3.6 Q4_K_M uses a mix of Q4_K / Q5_K /
/// Q6_K for big matmuls and F32 (→ BF16) for small ones (ssm_alpha, ssm_beta
/// when n_v_heads is small : n=48 typical, K=hidden_size, ≈1 MB each).
pub(crate) enum QuantTensor {
    Bf16 {
        weights: CudaSlice<half::bf16>,
        n: usize,
        k: usize,
    },
    Q4K {
        bytes: CudaSlice<u8>,
        n: usize, // out dim (rows)
        k: usize, // in dim (cols)
    },
    Q5K {
        bytes: CudaSlice<u8>,
        n: usize,
        k: usize,
    },
    Q6K {
        bytes: CudaSlice<u8>,
        n: usize,
        k: usize,
    },
}

impl QuantTensor {
    /// Dispatch a M=1 GEMV through the appropriate kernel.
    ///
    /// `x_q8_staging` is a device pointer to a scratch buffer of at least
    /// `(K/32) * 36` bytes used by the Q4_K dp4a path (T246.5.5) for the
    /// on-the-fly Q8_1 quantization of the activation. Pass 0 to disable
    /// dp4a and force the float v2 fallback. The buffer is overwritten
    /// each call so the same one can be reused across all matmuls.
    pub(crate) fn dispatch_matmul_m1(
        &self,
        kernels: &LlmKernels,
        stream: &Arc<CudaStream>,
        x: u64,
        y: u64,
        x_q8_staging: u64,
    ) -> Result<(), LlmError> {
        unsafe {
            use cudarc::driver::DevicePtr;
            match self {
                QuantTensor::Bf16 { weights, n, k } => {
                    let (w, _g) = weights.device_ptr(stream);
                    kernels
                        .sgemv_bf16_bf16(stream, w, x, y, *n as i32, *k as i32)
                        .map_err(|e| LlmError::Backend(format!("sgemv_bf16: {e:?}")))
                },
                QuantTensor::Q4K { bytes, n, k } => {
                    let (w, _g) = bytes.device_ptr(stream);
                    if x_q8_staging != 0 {
                        // T246.5.5 — dp4a path: quantize x → Q8_1, then
                        // Q4_K × Q8_1 SGEMV using __dp4a (mirror of
                        // llama.cpp vec_dot_q4_K_q8_1_impl_vmmq).
                        kernels
                            .quantize_q8_1_bf16(stream, x, x_q8_staging, *k as i32)
                            .map_err(|e| LlmError::Backend(format!("quant q8_1: {e:?}")))?;
                        kernels
                            .sgemv_q4k_q8_1_dp4a_bf16(
                                stream,
                                w,
                                x_q8_staging,
                                y,
                                *n as i32,
                                *k as i32,
                            )
                            .map_err(|e| LlmError::Backend(format!("sgemv_q4k_dp4a: {e:?}")))
                    } else {
                        // T246.6.7 — V3 (4-row-per-block, per-thread scale,
                        // __launch_bounds__) targets 60-70% peak BW vs V2's
                        // ~38%. Parity-tested vs V2.
                        kernels
                            .sgemv_q4k_bf16_v3(stream, w, x, y, *n as i32, *k as i32)
                            .map_err(|e| LlmError::Backend(format!("sgemv_q4k_v3: {e:?}")))
                    }
                },
                QuantTensor::Q5K { bytes, n, k } => {
                    let (w, _g) = bytes.device_ptr(stream);
                    // T246.6.9 — Heuristic: V3 (multi-row) when N is large
                    // enough to saturate SMs ; V2 for small SSM α/β matmuls
                    // where V3's 4-row block underutilizes occupancy.
                    if *n >= 256 {
                        kernels
                            .sgemv_q5k_bf16_v3(stream, w, x, y, *n as i32, *k as i32)
                            .map_err(|e| LlmError::Backend(format!("sgemv_q5k_v3: {e:?}")))
                    } else {
                        kernels
                            .sgemv_q5k_bf16(stream, w, x, y, *n as i32, *k as i32)
                            .map_err(|e| LlmError::Backend(format!("sgemv_q5k: {e:?}")))
                    }
                },
                QuantTensor::Q6K { bytes, n, k } => {
                    let (w, _g) = bytes.device_ptr(stream);
                    kernels
                        // T246.6.8 — V4 measured equal to V2 (no win) on the
                        // LM head shape (N=152064 × K=5120). V4 kernel kept
                        // for future investigation. Sticking with V2.
                        .sgemv_q6k_bf16_v2(stream, w, x, y, *n as i32, *k as i32)
                        .map_err(|e| LlmError::Backend(format!("sgemv_q6k_v2: {e:?}")))
                },
            }
        }
    }

    /// Dispatch a M=8 batched GEMM through the appropriate kernel.
    /// Used by prefill (T246.5) and speculative decoding.
    #[allow(dead_code)]
    pub(crate) fn dispatch_matmul_m8(
        &self,
        kernels: &LlmKernels,
        stream: &Arc<CudaStream>,
        x: u64,
        y: u64,
    ) -> Result<(), LlmError> {
        unsafe {
            use cudarc::driver::DevicePtr;
            match self {
                QuantTensor::Bf16 { .. } => Err(LlmError::Backend(
                    "BF16 M=8 batched not yet implemented (T246.5)".into(),
                )),
                QuantTensor::Q4K { bytes, n, k } => {
                    let (w, _g) = bytes.device_ptr(stream);
                    kernels
                        .sgemm_q4k_bf16_m8(stream, w, x, y, *n as i32, *k as i32)
                        .map_err(|e| LlmError::Backend(format!("sgemm_q4k_m8: {e:?}")))
                },
                QuantTensor::Q5K { bytes, n, k } => {
                    let (w, _g) = bytes.device_ptr(stream);
                    kernels
                        .sgemm_q5k_bf16_m8(stream, w, x, y, *n as i32, *k as i32)
                        .map_err(|e| LlmError::Backend(format!("sgemm_q5k_m8: {e:?}")))
                },
                QuantTensor::Q6K { bytes, n, k } => {
                    let (w, _g) = bytes.device_ptr(stream);
                    kernels
                        .sgemm_q6k_bf16_m8(stream, w, x, y, *n as i32, *k as i32)
                        .map_err(|e| LlmError::Backend(format!("sgemm_q6k_m8: {e:?}")))
                },
            }
        }
    }

    pub(crate) fn shape(&self) -> (usize, usize) {
        match self {
            QuantTensor::Bf16 { n, k, .. } => (*n, *k),
            QuantTensor::Q4K { n, k, .. } => (*n, *k),
            QuantTensor::Q5K { n, k, .. } => (*n, *k),
            QuantTensor::Q6K { n, k, .. } => (*n, *k),
        }
    }
}

/// MoE FFN weights for one layer (Qwen3.6-35B-A3B). Top-K routed experts +
/// parallel shared expert. T246.6.
pub(crate) struct MoeFfnQ4K {
    /// `[n_experts, D]` quantized — router logits.
    pub(crate) gate_inp: QuantTensor,
    /// `n_experts` × `[expert_f, D]` quantized — per-expert gate.
    pub(crate) gate_exps: Vec<QuantTensor>,
    /// `n_experts` × `[expert_f, D]` quantized — per-expert up.
    pub(crate) up_exps: Vec<QuantTensor>,
    /// `n_experts` × `[D, expert_f]` quantized — per-expert down.
    pub(crate) down_exps: Vec<QuantTensor>,
    /// `[D]` BF16 — shared-expert routing gain (sigmoid-gated scalar).
    pub(crate) gate_inp_shexp: CudaSlice<half::bf16>,
    /// `[expert_f, D]` quantized — shared-expert gate.
    pub(crate) gate_shexp: QuantTensor,
    /// `[expert_f, D]` quantized — shared-expert up.
    pub(crate) up_shexp: QuantTensor,
    /// `[D, expert_f]` quantized — shared-expert down.
    pub(crate) down_shexp: QuantTensor,
}

/// FFN variant — dense SwiGLU or top-K MoE.
pub(crate) enum FfnQ4K {
    Dense {
        gate: QuantTensor,
        up: QuantTensor,
        down: QuantTensor,
    },
    Moe(Box<MoeFfnQ4K>),
}

/// Attention block weights (one of every 4 layers in Qwen3.6).
pub(crate) struct AttnBlockQ4K {
    /// `[D]` BF16 — pre-attention RMSNorm gain.
    pub(crate) attn_norm: CudaSlice<half::bf16>,
    /// `[D]` BF16 — post-attention RMSNorm gain (HF naming `post_attention_norm`).
    pub(crate) post_norm: CudaSlice<half::bf16>,
    /// `[head_dim]` BF16 — per-head Q normalization gain (RMSNorm applied per head).
    pub(crate) q_norm: CudaSlice<half::bf16>,
    /// `[head_dim]` BF16 — per-head K normalization gain.
    pub(crate) k_norm: CudaSlice<half::bf16>,
    /// `[n_q_heads * head_dim, D]` quantized — Q projection.
    pub(crate) w_q: QuantTensor,
    /// `[n_kv_heads * head_dim, D]` quantized — K projection.
    pub(crate) w_k: QuantTensor,
    /// `[n_kv_heads * head_dim, D]` quantized — V projection.
    pub(crate) w_v: QuantTensor,
    /// `[D, n_q_heads * head_dim]` quantized — output projection.
    pub(crate) w_o: QuantTensor,
    /// FFN — dense (Qwen3.6-27B) or MoE (Qwen3.6-35B-A3B).
    pub(crate) ffn: FfnQ4K,
}

/// SSM (gated delta net) block weights — 48/64 layers in Qwen3.6.
pub(crate) struct SsmBlockQ4K {
    /// `[D]` BF16 — pre-SSM RMSNorm gain.
    pub(crate) attn_norm: CudaSlice<half::bf16>,
    /// `[D]` BF16 — post-SSM RMSNorm gain.
    pub(crate) post_norm: CudaSlice<half::bf16>,
    /// `[conv_dim, D]` quantized — fused QKV projection (split after conv).
    pub(crate) w_qkv: QuantTensor,
    /// `[value_dim, D]` quantized — gate (z) projection.
    pub(crate) w_gate: QuantTensor,
    /// `[conv_kernel, conv_dim]` BF16 — depth-wise 1-D conv kernel.
    pub(crate) conv1d: CudaSlice<half::bf16>,
    /// `[num_v_heads, D]` quantized — alpha (dt) projection.
    pub(crate) w_alpha: QuantTensor,
    /// `[num_v_heads, D]` quantized — beta projection.
    pub(crate) w_beta: QuantTensor,
    /// `[num_v_heads]` BF16 — dt bias.
    pub(crate) dt_bias: CudaSlice<half::bf16>,
    /// `[num_v_heads]` BF16 — A scalar per head (from log A).
    pub(crate) ssm_a: CudaSlice<half::bf16>,
    /// `[head_v_dim]` BF16 — gated RMSNorm gain.
    pub(crate) ssm_norm: CudaSlice<half::bf16>,
    /// `[D, value_dim]` quantized — output projection.
    pub(crate) ssm_out: QuantTensor,
    /// FFN — dense or MoE (T246.6, Qwen3.6-35B-A3B).
    pub(crate) ffn: FfnQ4K,
}

/// Per-block weights : either attention or SSM.
pub(crate) enum BlockQ4K {
    Attn(AttnBlockQ4K),
    Ssm(SsmBlockQ4K),
}

/// Per-attention-layer KV cache (allocated up to `max_seq` tokens).
pub(crate) struct KvCache {
    /// `[max_seq, n_kv_heads * head_dim]` BF16.
    pub(crate) k: CudaSlice<half::bf16>,
    /// `[max_seq, n_kv_heads * head_dim]` BF16.
    pub(crate) v: CudaSlice<half::bf16>,
}

/// Precomputed RoPE inverse frequencies `[rope_dim / 2]` F32, on GPU.
pub(crate) struct RopeFreqs {
    pub(crate) inv_freq: CudaSlice<f32>,
}

/// Per-SSM-layer recurrent state.
pub(crate) struct SsmState {
    /// `[num_v_heads, head_v_dim, head_v_dim]` BF16 — outer-product state.
    pub(crate) state: CudaSlice<half::bf16>,
    /// `[(conv_kernel - 1), conv_dim]` BF16 — ring buffer for depth-wise conv.
    pub(crate) conv_state: CudaSlice<half::bf16>,
}

/// Convert a global layer index into its position in `ssm_states` Vec.
fn get_ssm_layer_idx(cfg: &Qwen35Config, global_li: usize) -> usize {
    cfg.ssm_indices
        .iter()
        .position(|&i| i == global_li)
        .unwrap_or_else(|| panic!("layer {global_li} not in ssm_indices"))
}

/// Convert a global layer index into its position in `kv_caches` Vec.
#[allow(dead_code)]
fn get_attn_layer_idx(cfg: &Qwen35Config, global_li: usize) -> usize {
    cfg.attention_indices
        .iter()
        .position(|&i| i == global_li)
        .unwrap_or_else(|| panic!("layer {global_li} not in attention_indices"))
}

/// Pre-allocated scratch buffers for decode_step (hoisted out of the hot path).
pub(crate) struct DecodeScratch {
    pub(crate) h: CudaSlice<half::bf16>,
    pub(crate) h_norm: CudaSlice<half::bf16>,
    pub(crate) residual: CudaSlice<half::bf16>,
    pub(crate) qkv_mixed: CudaSlice<half::bf16>,
    pub(crate) conv_out: CudaSlice<half::bf16>,
    pub(crate) z: CudaSlice<half::bf16>,
    pub(crate) alpha: CudaSlice<half::bf16>,
    pub(crate) beta: CudaSlice<half::bf16>,
    pub(crate) q_v: CudaSlice<half::bf16>,
    pub(crate) k_v: CudaSlice<half::bf16>,
    pub(crate) ssm_out_buf: CudaSlice<half::bf16>,
    pub(crate) q_buf: CudaSlice<half::bf16>,
    pub(crate) k_buf: CudaSlice<half::bf16>,
    pub(crate) v_buf: CudaSlice<half::bf16>,
    pub(crate) attn_out: CudaSlice<half::bf16>,
    pub(crate) gate_buf: CudaSlice<half::bf16>,
    pub(crate) up_buf: CudaSlice<half::bf16>,
    pub(crate) down_buf: CudaSlice<half::bf16>,
    pub(crate) logits: CudaSlice<half::bf16>,
    pub(crate) next_token: CudaSlice<u32>,
    /// T246.5.5 — staging buffer for Q8_1 quantized activation (used by the
    /// dp4a Q4_K matmul path). Sized to accommodate the largest K seen in
    /// any matmul of the model: `(max_k / 32) * 36` bytes.
    pub(crate) x_q8_scratch: CudaSlice<u8>,
    /// T246.5.7 — FlashDecode split-K staging buffers.
    /// `[n_q_heads, N_SPLIT]` floats — partial max scores per (head, split).
    pub(crate) gqa_partial_m: CudaSlice<f32>,
    /// `[n_q_heads, N_SPLIT]` floats — partial l (sum of expf weights).
    pub(crate) gqa_partial_l: CudaSlice<f32>,
    /// `[n_q_heads, N_SPLIT, head_dim]` bf16 — partial output.
    pub(crate) gqa_partial_o: CudaSlice<half::bf16>,
    /// T246.6 — MoE staging.
    /// `[n_experts]` BF16 — router logits scratch.
    pub(crate) moe_router_logits: CudaSlice<half::bf16>,
    /// `[k]` i32 — top-K expert indices.
    pub(crate) moe_topk_idx: CudaSlice<i32>,
    /// `[k]` BF16 — renormalized top-K weights.
    pub(crate) moe_topk_w: CudaSlice<half::bf16>,
    /// `[expert_f]` BF16 — per-expert gate buffer.
    pub(crate) moe_expert_gate: CudaSlice<half::bf16>,
    /// `[expert_f]` BF16 — per-expert up buffer.
    pub(crate) moe_expert_up: CudaSlice<half::bf16>,
    /// `[d]` BF16 — per-expert output (down result, accumulated into h).
    pub(crate) moe_expert_out: CudaSlice<half::bf16>,
    /// `[1]` BF16 — shared-expert sigmoid dot product scratch.
    pub(crate) moe_shexp_dot: CudaSlice<half::bf16>,
    // ── T246.7 P1.3 — Lookahead Decoding tree-attention scratch ──
    /// `[MAX_TREE_SIZE]` u32 — input draft tokens for `decode_step_tree`.
    pub(crate) tree_drafts: CudaSlice<u32>,
    /// `[MAX_TREE_SIZE]` i32 — parent pointer per tree node, root = -1.
    pub(crate) tree_parents: CudaSlice<i32>,
    /// `[MAX_TREE_SIZE]` u8 — depth per tree node, root = 0.
    pub(crate) tree_depths: CudaSlice<u8>,
    /// `[MAX_TREE_SIZE]` u32 — argmax token per tree node row of logits.
    pub(crate) tree_argmax: CudaSlice<u32>,
    /// `[MAX_TREE_SIZE, n_q, n_split]` f32 — partial m for tree GQA.
    pub(crate) tree_gqa_partial_m: CudaSlice<f32>,
    /// `[MAX_TREE_SIZE, n_q, n_split]` f32 — partial l for tree GQA.
    pub(crate) tree_gqa_partial_l: CudaSlice<f32>,
    /// `[MAX_TREE_SIZE, n_q, n_split, head_dim]` BF16 — partial o for tree GQA.
    pub(crate) tree_gqa_partial_o: CudaSlice<half::bf16>,
}

/// CUDA-resident Qwen3.5/3.6 hybrid model with Q4_K_M weights.
pub struct Qwen35ModelCudaQ4K {
    pub config: Qwen35Config,
    pub(crate) ctx: Arc<CudaContext>,
    pub(crate) stream: Arc<CudaStream>,
    #[allow(dead_code)]
    pub(crate) session: LtSession,
    pub(crate) kernels: LlmKernels,
    /// Token embedding `[V, D]` BF16 (dequantized at load — single read at start).
    pub(crate) token_emb: CudaSlice<half::bf16>,
    /// Final RMSNorm gain `[D]` BF16.
    pub(crate) final_norm: CudaSlice<half::bf16>,
    /// LM head `[V, D]` quantized (typically Q6_K).
    pub(crate) lm_head: QuantTensor,
    /// Precomputed RoPE inv_freq buffer.
    pub(crate) rope_freqs: RopeFreqs,
    /// Per-block weights, one per layer.
    pub(crate) blocks: Vec<BlockQ4K>,
    /// KV cache, one entry per attention layer (in original layer-index order).
    pub(crate) kv_caches: Vec<KvCache>,
    /// SSM state, one entry per SSM layer.
    pub(crate) ssm_states: Vec<SsmState>,
    /// Maximum sequence length supported by the pre-allocated KV cache.
    pub(crate) max_seq: usize,
    /// Current decode position (incremented after each forward_token).
    pub(crate) position: usize,
    /// Pre-allocated scratch buffers (one-time alloc).
    pub(crate) scratch: DecodeScratch,
    /// T246.5.3 — device-resident position counter, mirror of `position`.
    /// Used by `rope_partial_bf16_devcnt` so RoPE remains correct across
    /// CUDA Graph replays. Updated each step via tiny H2D (Phase B).
    pub(crate) position_dev: CudaSlice<i32>,
    /// T246.5.3 — device-resident kv_len counter (= position + 1).
    /// Used by `gqa_decode_online_bf16_devcnt`. Updated each step.
    pub(crate) kv_len_dev: CudaSlice<i32>,
    /// T246.5.3 — device-resident current-token-id buffer.
    /// Used by `embedding_lookup_bf16` so the per-step token id is read
    /// from device memory inside the captured graph.
    pub(crate) current_token_dev: CudaSlice<u32>,
    /// T246.5.3 — captured CUDA Graph of one full decode_step body.
    /// Initialized lazily on the 2nd decode call (after the 1st-call warmup
    /// triggers all kernel JIT compilations). Once set, every subsequent
    /// decode_step replays this graph in a single launch instead of issuing
    /// ~1908 individual cuLaunchKernel calls.
    pub(crate) decode_graph: Option<CudaGraph>,
    /// T246.5.5 — toggle for the dp4a Q4_K matmul path. Off by default:
    /// current dp4a kernel is W-bandwidth-bound on M=1 (same as the float
    /// v2 path) but adds a quantize-q8_1 kernel per matmul → ~1% slower
    /// than v2 today. Set `RUSTORCH_USE_DP4A_Q4K=1` at process start to
    /// enable for benchmarking / iterative kernel optimization.
    pub(crate) use_dp4a_q4k: bool,
    /// T246.5.6 — pinned host buffer (1× u32) for the per-step next-token
    /// DtoH. Replacing `memcpy_dtov` (Vec<u32> on pageable mem, which
    /// the driver implicitly synchronizes) with `memcpy_dtoh` to this
    /// pinned slice keeps the DtoH truly async and lets the driver DMA
    /// directly into host memory ; the only sync point becomes the
    /// `as_slice()` call which waits on a dedicated event instead of the
    /// full stream.
    pub(crate) next_token_host_pinned: PinnedHostSlice<u32>,
}

impl Qwen35ModelCudaQ4K {
    /// Load Qwen3.5/3.6 hybrid GGUF (Q4_K_M typically) directly to GPU using
    /// raw Q4_K / Q5_K / Q6_K bytes for matmuls. Norms, biases, ssm_a,
    /// dt_bias, conv1d : F32 → BF16 dequant (small tensors).
    ///
    /// Multi-shard GGUF : caller must pre-merge shards (TODO T246.5+).
    pub fn from_gguf(path: &Path, max_seq: usize) -> Result<Self, LlmError> {
        let ctx =
            CudaContext::new(0).map_err(|e| LlmError::Backend(format!("CudaContext: {e:?}")))?;
        // T246.5.3 — Disable cudarc's automatic event tracking BEFORE allocating
        // anything. With multi-stream mode + event tracking on, every
        // device_ptr_mut() injects `stream.wait(event)` calls referencing events
        // recorded on the warmup pass. During CUDA Graph capture those waits
        // reference an event NOT belonging to the capture → the graph is
        // invalidated immediately (CUDA_ERROR_STREAM_CAPTURE_INVALIDATED on
        // the first op after begin_capture). All decode work runs on a single
        // stream so we don't need cross-stream safety here.
        // SAFETY: disable_event_tracking only affects slices created AFTER this
        // call. We immediately allocate everything below, so no pre-existing
        // tracked slices exist.
        unsafe {
            ctx.disable_event_tracking();
        }
        // T246.5.3 — dedicated non-default stream for CUDA Graph capture.
        // CUDA Graphs cannot capture the default/legacy stream — begin_capture
        // returns CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED on it.
        let stream = ctx
            .new_stream()
            .map_err(|e| LlmError::Backend(format!("new_stream: {e:?}")))?;
        let session = LtSession::new(stream.clone())
            .map_err(|e| LlmError::Backend(format!("LtSession: {e:?}")))?;
        let kernels = LlmKernels::new(ctx.clone());

        let cfg = crate::qwen35::parse_config(path)
            .map_err(|e| LlmError::Backend(format!("parse_config: {e:?}")))?;
        let file = GgufFile::open(path).map_err(|e| LlmError::Backend(format!("gguf: {e:?}")))?;

        // T246.6 — MoE (Qwen3.6-35B-A3B) is now supported via FfnQ4K::Moe.
        // Dense (Qwen3.6-27B) continues through the original path.

        // ---- Helpers ----
        let load_quant = |name: &str| -> Result<QuantTensor, LlmError> {
            let info = file
                .tensor(name)
                .ok_or_else(|| LlmError::MissingWeight(name.to_string()))?;
            let bytes = file.tensor_bytes(info);
            let n = info.shape[1] as usize;
            let k = info.shape[0] as usize;
            match info.dtype {
                GgmlType::Q4_K => {
                    let dev = stream
                        .memcpy_stod(bytes)
                        .map_err(|e| LlmError::Backend(format!("upload {name}: {e:?}")))?;
                    Ok(QuantTensor::Q4K { bytes: dev, n, k })
                },
                GgmlType::Q5_K => {
                    let dev = stream
                        .memcpy_stod(bytes)
                        .map_err(|e| LlmError::Backend(format!("upload {name}: {e:?}")))?;
                    Ok(QuantTensor::Q5K { bytes: dev, n, k })
                },
                GgmlType::Q6_K => {
                    let dev = stream
                        .memcpy_stod(bytes)
                        .map_err(|e| LlmError::Backend(format!("upload {name}: {e:?}")))?;
                    Ok(QuantTensor::Q6K { bytes: dev, n, k })
                },
                // F32 / F16 / Q8_0 fallback : dequant to BF16 at load time,
                // dispatch via sgemv_bf16_bf16. Q8_0 unblocks 35B-A3B
                // (T246.6) where attn_qkv / attn_output are Q8_0 not Q4_K
                // in the UD-Q4_K_M repack.
                GgmlType::F32 | GgmlType::F16 | GgmlType::Q8_0 => {
                    let f32_buf = rustorch_gguf::dequant::dequant_to_f32(info, bytes)
                        .map_err(|e| LlmError::Backend(format!("dequant {name}: {e:?}")))?;
                    let bf: Vec<half::bf16> =
                        f32_buf.iter().copied().map(half::bf16::from_f32).collect();
                    let dev = stream
                        .memcpy_stod(&bf)
                        .map_err(|e| LlmError::Backend(format!("upload {name}: {e:?}")))?;
                    Ok(QuantTensor::Bf16 { weights: dev, n, k })
                },
                other => Err(LlmError::Backend(format!(
                    "unsupported dtype {other:?} for {name} — only Q4_K/Q5_K/Q6_K/Q8_0/F32/F16 supported"
                ))),
            }
        };

        let load_bf16 = |name: &str| -> Result<CudaSlice<half::bf16>, LlmError> {
            let info = file
                .tensor(name)
                .ok_or_else(|| LlmError::MissingWeight(name.to_string()))?;
            let bytes = file.tensor_bytes(info);
            let f32_buf = rustorch_gguf::dequant::dequant_to_f32(info, bytes)
                .map_err(|e| LlmError::Backend(format!("dequant {name}: {e:?}")))?;
            let bf: Vec<half::bf16> = f32_buf.iter().copied().map(half::bf16::from_f32).collect();
            stream
                .memcpy_stod(&bf)
                .map_err(|e| LlmError::Backend(format!("upload {name}: {e:?}")))
        };

        // T246.6 — Load a stacked-expert quantized tensor. GGUF stores
        // `[d_inner, rows, n_experts]` as one quantized blob. We slice and
        // upload each expert independently. Supports K-quants (Q4/5/6_K
        // preserved on-device) and Q8_0 / F32 / F16 (dequantized to BF16
        // on the host side). 35B-A3B is mixed Q4_K + Q5_K + Q8_0 in the
        // UD-Q4_K_M repack.
        let load_stacked_quant_experts = |name: &str,
                                          n_experts: usize,
                                          rows: usize,
                                          k: usize|
         -> Result<Vec<QuantTensor>, LlmError> {
            let info = file
                .tensor(name)
                .ok_or_else(|| LlmError::MissingWeight(name.to_string()))?;
            let bytes = file.tensor_bytes(info);
            let block_size = info.dtype.block_size();
            let type_size = info.dtype.type_size();
            let weights_per_expert = rows * k;
            if weights_per_expert % block_size != 0 {
                return Err(LlmError::Backend(format!(
                    "{name}: expert size {} not aligned on {} ({:?})",
                    weights_per_expert, block_size, info.dtype
                )));
            }
            let stride = weights_per_expert / block_size * type_size;
            if bytes.len() != n_experts * stride {
                return Err(LlmError::Backend(format!(
                    "{name}: expected {} bytes ({} × {}), got {}",
                    n_experts * stride,
                    n_experts,
                    stride,
                    bytes.len()
                )));
            }
            let mut out = Vec::with_capacity(n_experts);
            for e in 0..n_experts {
                let slice = &bytes[e * stride..(e + 1) * stride];
                let qt = match info.dtype {
                    GgmlType::Q4_K => {
                        let dev = stream.memcpy_stod(slice).map_err(|err| {
                            LlmError::Backend(format!("upload {name}#{e}: {err:?}"))
                        })?;
                        QuantTensor::Q4K {
                            bytes: dev,
                            n: rows,
                            k,
                        }
                    },
                    GgmlType::Q5_K => {
                        let dev = stream.memcpy_stod(slice).map_err(|err| {
                            LlmError::Backend(format!("upload {name}#{e}: {err:?}"))
                        })?;
                        QuantTensor::Q5K {
                            bytes: dev,
                            n: rows,
                            k,
                        }
                    },
                    GgmlType::Q6_K => {
                        let dev = stream.memcpy_stod(slice).map_err(|err| {
                            LlmError::Backend(format!("upload {name}#{e}: {err:?}"))
                        })?;
                        QuantTensor::Q6K {
                            bytes: dev,
                            n: rows,
                            k,
                        }
                    },
                    // Dequant path : produce per-expert BF16 device buffer.
                    GgmlType::Q8_0 | GgmlType::F32 | GgmlType::F16 => {
                        let synth = rustorch_gguf::tensor::TensorInfo {
                            name: format!("{name}#{e}"),
                            shape: vec![k as u64, rows as u64],
                            dtype: info.dtype,
                            offset: 0,
                        };
                        let f32_buf = rustorch_gguf::dequant::dequant_to_f32(&synth, slice)
                            .map_err(|err| {
                                LlmError::Backend(format!("dequant {name}#{e}: {err:?}"))
                            })?;
                        let bf: Vec<half::bf16> =
                            f32_buf.iter().copied().map(half::bf16::from_f32).collect();
                        let dev = stream.memcpy_stod(&bf).map_err(|err| {
                            LlmError::Backend(format!("upload {name}#{e}: {err:?}"))
                        })?;
                        QuantTensor::Bf16 {
                            weights: dev,
                            n: rows,
                            k,
                        }
                    },
                    other => {
                        return Err(LlmError::Backend(format!(
                            "{name}: unsupported expert dtype {other:?}"
                        )));
                    },
                };
                out.push(qt);
            }
            Ok(out)
        };

        // T246.6 — Load the FFN portion for layer `li`. Returns Dense for
        // Qwen3.6-27B (and Qwen3PureTransformer if it ever lands), Moe for
        // Qwen3.6-35B-A3B.
        let load_ffn = |li: usize| -> Result<FfnQ4K, LlmError> {
            let key = |s: &str| format!("blk.{li}.{s}");
            match cfg.variant {
                Qwen35Variant::Dense | Qwen35Variant::Qwen3PureTransformer => Ok(FfnQ4K::Dense {
                    gate: load_quant(&key("ffn_gate.weight"))?,
                    up: load_quant(&key("ffn_up.weight"))?,
                    down: load_quant(&key("ffn_down.weight"))?,
                }),
                Qwen35Variant::Moe => {
                    let n_e = cfg.n_experts;
                    let ef = cfg.expert_f;
                    let d = cfg.d;
                    let moe = MoeFfnQ4K {
                        gate_inp: load_quant(&key("ffn_gate_inp.weight"))?,
                        gate_exps: load_stacked_quant_experts(
                            &key("ffn_gate_exps.weight"),
                            n_e,
                            ef,
                            d,
                        )?,
                        up_exps: load_stacked_quant_experts(
                            &key("ffn_up_exps.weight"),
                            n_e,
                            ef,
                            d,
                        )?,
                        down_exps: load_stacked_quant_experts(
                            &key("ffn_down_exps.weight"),
                            n_e,
                            d,
                            ef,
                        )?,
                        gate_inp_shexp: load_bf16(&key("ffn_gate_inp_shexp.weight"))?,
                        gate_shexp: load_quant(&key("ffn_gate_shexp.weight"))?,
                        up_shexp: load_quant(&key("ffn_up_shexp.weight"))?,
                        down_shexp: load_quant(&key("ffn_down_shexp.weight"))?,
                    };
                    Ok(FfnQ4K::Moe(Box::new(moe)))
                },
            }
        };

        // ---- Globals ----
        let token_emb = load_bf16("token_embd.weight")?;
        let final_norm = load_bf16("output_norm.weight")?;
        // Precompute RoPE inv_freq for the attention layers.
        let rope_dim = cfg.rope_dim;
        let inv_freq_host: Vec<f32> = (0..rope_dim / 2)
            .map(|i| (cfg.rope_base).powf(-(2.0 * i as f32) / rope_dim as f32))
            .collect();
        let inv_freq_dev = stream
            .memcpy_stod(&inv_freq_host)
            .map_err(|e| LlmError::Backend(format!("rope_inv_freq: {e:?}")))?;
        let rope_freqs = RopeFreqs {
            inv_freq: inv_freq_dev,
        };
        let lm_head = if file.tensor("output.weight").is_some() {
            load_quant("output.weight")?
        } else {
            // Tied embeddings — for simplicity we don't yet share storage.
            // Allocate a separate Q4_K copy or keep BF16 fallback.
            return Err(LlmError::Backend(
                "tied embeddings not yet supported in Q4K path (Qwen3.6 has separate output.weight)".into(),
            ));
        };

        // ---- Per-layer weights ----
        let mut blocks: Vec<BlockQ4K> = Vec::with_capacity(cfg.n_layers);
        let mut kv_caches: Vec<KvCache> = Vec::new();
        let mut ssm_states: Vec<SsmState> = Vec::new();

        let kv_dim = cfg.n_kv_heads * cfg.head_dim();
        let value_dim = cfg.ssm_dt_rank * cfg.ssm_state;
        let key_dim = cfg.ssm_groups * cfg.ssm_state;
        let conv_dim = 2 * key_dim + value_dim;
        let conv_kernel = cfg.ssm_conv_kernel;

        for li in 0..cfg.n_layers {
            let kind = cfg.layer_kind(li);
            let key = |s: &str| format!("blk.{li}.{s}");
            let block = match kind {
                LayerKind::Attention => {
                    let attn = AttnBlockQ4K {
                        attn_norm: load_bf16(&key("attn_norm.weight"))?,
                        post_norm: load_bf16(&key("post_attention_norm.weight"))?,
                        q_norm: load_bf16(&key("attn_q_norm.weight"))?,
                        k_norm: load_bf16(&key("attn_k_norm.weight"))?,
                        w_q: load_quant(&key("attn_q.weight"))?,
                        w_k: load_quant(&key("attn_k.weight"))?,
                        w_v: load_quant(&key("attn_v.weight"))?,
                        w_o: load_quant(&key("attn_output.weight"))?,
                        ffn: load_ffn(li)?,
                    };
                    // Allocate KV cache for this layer.
                    let k_cache =
                        stream
                            .alloc_zeros::<half::bf16>(max_seq * kv_dim)
                            .map_err(|e| {
                                LlmError::Backend(format!("alloc K cache layer {li}: {e:?}"))
                            })?;
                    let v_cache =
                        stream
                            .alloc_zeros::<half::bf16>(max_seq * kv_dim)
                            .map_err(|e| {
                                LlmError::Backend(format!("alloc V cache layer {li}: {e:?}"))
                            })?;
                    kv_caches.push(KvCache {
                        k: k_cache,
                        v: v_cache,
                    });
                    BlockQ4K::Attn(attn)
                },
                LayerKind::Ssm => {
                    let ssm = SsmBlockQ4K {
                        attn_norm: load_bf16(&key("attn_norm.weight"))?,
                        post_norm: load_bf16(&key("post_attention_norm.weight"))?,
                        w_qkv: load_quant(&key("attn_qkv.weight"))?,
                        w_gate: load_quant(&key("attn_gate.weight"))?,
                        conv1d: load_bf16(&key("ssm_conv1d.weight"))?,
                        w_alpha: load_quant(&key("ssm_alpha.weight"))?,
                        w_beta: load_quant(&key("ssm_beta.weight"))?,
                        dt_bias: load_bf16(&key("ssm_dt.bias"))?,
                        ssm_a: load_bf16(&key("ssm_a"))?,
                        ssm_norm: load_bf16(&key("ssm_norm.weight"))?,
                        ssm_out: load_quant(&key("ssm_out.weight"))?,
                        ffn: load_ffn(li)?,
                    };
                    let head_v_dim = cfg.ssm_state;
                    let n_v_heads = cfg.ssm_dt_rank;
                    let state = stream
                        .alloc_zeros::<half::bf16>(n_v_heads * head_v_dim * head_v_dim)
                        .map_err(|e| {
                            LlmError::Backend(format!("alloc SSM state layer {li}: {e:?}"))
                        })?;
                    let conv_state = stream
                        .alloc_zeros::<half::bf16>((conv_kernel - 1) * conv_dim)
                        .map_err(|e| {
                            LlmError::Backend(format!("alloc conv_state layer {li}: {e:?}"))
                        })?;
                    ssm_states.push(SsmState { state, conv_state });
                    BlockQ4K::Ssm(ssm)
                },
            };
            blocks.push(block);
        }

        // ---- Pre-allocate decode scratch buffers (one-time) ----
        let kv_dim_attn = cfg.n_kv_heads * cfg.head_dim();
        let q_dim = cfg.n_q_heads * cfg.head_dim();
        // up_buf is (re)used as QG buffer (size 2*q_dim) ; gate_buf used as
        // attn raw output (size q_dim). Both must fit f for FFN AND 2*q_dim for QG.
        let scratch_up = cfg.f.max(2 * q_dim);
        let scratch_gate = cfg.f.max(q_dim);
        let scratch = DecodeScratch {
            h: stream
                .alloc_zeros::<half::bf16>(cfg.d)
                .map_err(|e| LlmError::Backend(format!("scratch h: {e:?}")))?,
            h_norm: stream
                .alloc_zeros::<half::bf16>(cfg.d)
                .map_err(|e| LlmError::Backend(format!("scratch h_norm: {e:?}")))?,
            residual: stream
                .alloc_zeros::<half::bf16>(cfg.d)
                .map_err(|e| LlmError::Backend(format!("scratch residual: {e:?}")))?,
            qkv_mixed: stream
                .alloc_zeros::<half::bf16>(2 * key_dim + value_dim)
                .map_err(|e| LlmError::Backend(format!("scratch qkv: {e:?}")))?,
            conv_out: stream
                .alloc_zeros::<half::bf16>(2 * key_dim + value_dim)
                .map_err(|e| LlmError::Backend(format!("scratch conv_out: {e:?}")))?,
            z: stream
                .alloc_zeros::<half::bf16>(value_dim)
                .map_err(|e| LlmError::Backend(format!("scratch z: {e:?}")))?,
            alpha: stream
                .alloc_zeros::<half::bf16>(cfg.ssm_dt_rank)
                .map_err(|e| LlmError::Backend(format!("scratch alpha: {e:?}")))?,
            beta: stream
                .alloc_zeros::<half::bf16>(cfg.ssm_dt_rank)
                .map_err(|e| LlmError::Backend(format!("scratch beta: {e:?}")))?,
            q_v: stream
                .alloc_zeros::<half::bf16>(cfg.ssm_dt_rank * cfg.ssm_state)
                .map_err(|e| LlmError::Backend(format!("scratch q_v: {e:?}")))?,
            k_v: stream
                .alloc_zeros::<half::bf16>(cfg.ssm_dt_rank * cfg.ssm_state)
                .map_err(|e| LlmError::Backend(format!("scratch k_v: {e:?}")))?,
            ssm_out_buf: stream
                .alloc_zeros::<half::bf16>(cfg.ssm_dt_rank * cfg.ssm_state)
                .map_err(|e| LlmError::Backend(format!("scratch ssm_out: {e:?}")))?,
            q_buf: stream
                .alloc_zeros::<half::bf16>(q_dim)
                .map_err(|e| LlmError::Backend(format!("scratch q_buf: {e:?}")))?,
            k_buf: stream
                .alloc_zeros::<half::bf16>(kv_dim_attn)
                .map_err(|e| LlmError::Backend(format!("scratch k_buf: {e:?}")))?,
            v_buf: stream
                .alloc_zeros::<half::bf16>(kv_dim_attn)
                .map_err(|e| LlmError::Backend(format!("scratch v_buf: {e:?}")))?,
            attn_out: stream
                .alloc_zeros::<half::bf16>(q_dim)
                .map_err(|e| LlmError::Backend(format!("scratch attn_out: {e:?}")))?,
            gate_buf: stream
                .alloc_zeros::<half::bf16>(scratch_gate)
                .map_err(|e| LlmError::Backend(format!("scratch gate: {e:?}")))?,
            up_buf: stream
                .alloc_zeros::<half::bf16>(scratch_up)
                .map_err(|e| LlmError::Backend(format!("scratch up: {e:?}")))?,
            down_buf: stream
                .alloc_zeros::<half::bf16>(cfg.d)
                .map_err(|e| LlmError::Backend(format!("scratch down: {e:?}")))?,
            logits: stream
                .alloc_zeros::<half::bf16>(cfg.vocab)
                .map_err(|e| LlmError::Backend(format!("scratch logits: {e:?}")))?,
            next_token: stream
                .alloc_zeros::<u32>(1)
                .map_err(|e| LlmError::Backend(format!("scratch token: {e:?}")))?,
            x_q8_scratch: {
                // Max K across all matmuls = cfg.f (down_proj K=F). Round up
                // to next multiple of 32 for safety.
                let max_k = cfg.f.max(cfg.d).max(q_dim).max(value_dim);
                let max_k_blocks = max_k.div_ceil(32);
                stream
                    .alloc_zeros::<u8>(max_k_blocks * 36)
                    .map_err(|e| LlmError::Backend(format!("scratch x_q8: {e:?}")))?
            },
            // T246.5.7 — FlashDecode split-K staging.
            gqa_partial_m: stream
                .alloc_zeros::<f32>(cfg.n_q_heads * GQA_N_SPLIT)
                .map_err(|e| LlmError::Backend(format!("scratch gqa_m: {e:?}")))?,
            gqa_partial_l: stream
                .alloc_zeros::<f32>(cfg.n_q_heads * GQA_N_SPLIT)
                .map_err(|e| LlmError::Backend(format!("scratch gqa_l: {e:?}")))?,
            gqa_partial_o: stream
                .alloc_zeros::<half::bf16>(cfg.n_q_heads * GQA_N_SPLIT * cfg.head_dim())
                .map_err(|e| LlmError::Backend(format!("scratch gqa_o: {e:?}")))?,
            // T246.6 — MoE staging (allocated even for Dense to keep struct
            // shape stable ; for Dense the buffers are tiny and unused).
            moe_router_logits: stream
                .alloc_zeros::<half::bf16>(cfg.n_experts.max(1))
                .map_err(|e| LlmError::Backend(format!("scratch moe_router: {e:?}")))?,
            moe_topk_idx: stream
                .alloc_zeros::<i32>(cfg.n_experts_used.max(1))
                .map_err(|e| LlmError::Backend(format!("scratch moe_idx: {e:?}")))?,
            moe_topk_w: stream
                .alloc_zeros::<half::bf16>(cfg.n_experts_used.max(1))
                .map_err(|e| LlmError::Backend(format!("scratch moe_w: {e:?}")))?,
            moe_expert_gate: stream
                .alloc_zeros::<half::bf16>(cfg.expert_f.max(1))
                .map_err(|e| LlmError::Backend(format!("scratch moe_gate: {e:?}")))?,
            moe_expert_up: stream
                .alloc_zeros::<half::bf16>(cfg.expert_f.max(1))
                .map_err(|e| LlmError::Backend(format!("scratch moe_up: {e:?}")))?,
            moe_expert_out: stream
                .alloc_zeros::<half::bf16>(cfg.d)
                .map_err(|e| LlmError::Backend(format!("scratch moe_out: {e:?}")))?,
            moe_shexp_dot: stream
                .alloc_zeros::<half::bf16>(1)
                .map_err(|e| LlmError::Backend(format!("scratch moe_shexp_dot: {e:?}")))?,
            // T246.7 P1.3 — Lookahead Decoding tree-attention scratch.
            tree_drafts: stream
                .alloc_zeros::<u32>(MAX_TREE_SIZE)
                .map_err(|e| LlmError::Backend(format!("scratch tree_drafts: {e:?}")))?,
            tree_parents: stream
                .alloc_zeros::<i32>(MAX_TREE_SIZE)
                .map_err(|e| LlmError::Backend(format!("scratch tree_parents: {e:?}")))?,
            tree_depths: stream
                .alloc_zeros::<u8>(MAX_TREE_SIZE)
                .map_err(|e| LlmError::Backend(format!("scratch tree_depths: {e:?}")))?,
            tree_argmax: stream
                .alloc_zeros::<u32>(MAX_TREE_SIZE)
                .map_err(|e| LlmError::Backend(format!("scratch tree_argmax: {e:?}")))?,
            tree_gqa_partial_m: stream
                .alloc_zeros::<f32>(MAX_TREE_SIZE * cfg.n_q_heads * GQA_N_SPLIT)
                .map_err(|e| LlmError::Backend(format!("scratch tree_gqa_m: {e:?}")))?,
            tree_gqa_partial_l: stream
                .alloc_zeros::<f32>(MAX_TREE_SIZE * cfg.n_q_heads * GQA_N_SPLIT)
                .map_err(|e| LlmError::Backend(format!("scratch tree_gqa_l: {e:?}")))?,
            tree_gqa_partial_o: stream
                .alloc_zeros::<half::bf16>(
                    MAX_TREE_SIZE * cfg.n_q_heads * GQA_N_SPLIT * cfg.head_dim(),
                )
                .map_err(|e| LlmError::Backend(format!("scratch tree_gqa_o: {e:?}")))?,
        };

        // T246.5.3 — device-resident counters for CUDA Graph capture.
        let position_dev = stream
            .memcpy_stod(&[0i32])
            .map_err(|e| LlmError::Backend(format!("position_dev: {e:?}")))?;
        let kv_len_dev = stream
            .memcpy_stod(&[1i32])
            .map_err(|e| LlmError::Backend(format!("kv_len_dev: {e:?}")))?;
        let current_token_dev = stream
            .memcpy_stod(&[0u32])
            .map_err(|e| LlmError::Backend(format!("current_token_dev: {e:?}")))?;

        // T246.5.6 — 1× u32 pinned host buffer for async next-token DtoH
        // (allocated before the struct ctor to avoid moving `ctx` early).
        let next_token_host_pinned = unsafe { ctx.alloc_pinned::<u32>(1) }
            .map_err(|e| LlmError::Backend(format!("alloc_pinned next_token: {e:?}")))?;

        Ok(Self {
            config: cfg,
            ctx,
            stream,
            session,
            kernels,
            token_emb,
            final_norm,
            lm_head,
            rope_freqs,
            blocks,
            kv_caches,
            ssm_states,
            max_seq,
            position: 0,
            scratch,
            position_dev,
            kv_len_dev,
            current_token_dev,
            decode_graph: None,
            use_dp4a_q4k: std::env::var("RUSTORCH_USE_DP4A_Q4K")
                .map(|v| v != "0" && !v.is_empty())
                .unwrap_or(false),
            next_token_host_pinned,
        })
    }

    /// Reset per-conversation state (KV cache + SSM state) to start fresh.
    pub fn reset_state(&mut self) -> Result<(), LlmError> {
        for kv in self.kv_caches.iter_mut() {
            self.stream
                .memset_zeros(&mut kv.k)
                .map_err(|e| LlmError::Backend(format!("zero K cache: {e:?}")))?;
            self.stream
                .memset_zeros(&mut kv.v)
                .map_err(|e| LlmError::Backend(format!("zero V cache: {e:?}")))?;
        }
        for ssm in self.ssm_states.iter_mut() {
            self.stream
                .memset_zeros(&mut ssm.state)
                .map_err(|e| LlmError::Backend(format!("zero SSM state: {e:?}")))?;
            self.stream
                .memset_zeros(&mut ssm.conv_state)
                .map_err(|e| LlmError::Backend(format!("zero conv state: {e:?}")))?;
        }
        self.position = 0;
        // T246.5.3 — also reset device-side counters.
        self.stream
            .memcpy_htod(&[0i32], &mut self.position_dev)
            .map_err(|e| LlmError::Backend(format!("reset position_dev: {e:?}")))?;
        self.stream
            .memcpy_htod(&[1i32], &mut self.kv_len_dev)
            .map_err(|e| LlmError::Backend(format!("reset kv_len_dev: {e:?}")))?;
        // T246.5.3 — captured graph is no longer valid for the new state.
        // It will be re-captured on the 2nd decode_step after this reset.
        self.decode_graph = None;
        Ok(())
    }

    /// Decode one token. T246.2-4 implementation : full Qwen3.6 forward
    /// (hybrid SSM + Attention) at M=1, returning next token id.
    pub fn decode_step(&mut self, token_id: u32) -> Result<u32, LlmError> {
        use cudarc::driver::{DevicePtr, DevicePtrMut};

        let cfg = self.config.clone();
        let d = cfg.d;
        let f = cfg.f;
        let n_q = cfg.n_q_heads;
        let n_kv = cfg.n_kv_heads;
        let head_dim = cfg.head_dim();
        let q_dim = n_q * head_dim;
        let kv_dim = n_kv * head_dim;
        let head_kv = cfg.ssm_state;
        let n_k = cfg.ssm_groups;
        let n_v = cfg.ssm_dt_rank;
        let key_dim = head_kv * n_k;
        let value_dim = head_kv * n_v;
        let conv_dim = 2 * key_dim + value_dim;
        let conv_kernel = cfg.ssm_conv_kernel;
        let eps = cfg.rms_eps;

        // ---- Use pre-allocated scratch (no per-step alloc, T246.4.1) ----
        let _ = (conv_dim, value_dim, n_v, head_kv, q_dim, kv_dim, f);

        // T246.5.3 — Always update current_token_dev (1 × 4-byte H2D, OUTSIDE
        // any capture region so it executes immediately and remains mutable
        // between graph replays). position_dev and kv_len_dev are self-managing
        // via increment_u32_dev kernels at the end of the body — no need to
        // write them from host after init.
        self.stream
            .memcpy_htod(&[token_id], &mut self.current_token_dev)
            .map_err(|e| LlmError::Backend(format!("upload token id: {e:?}")))?;

        // T246.5.3 — Replay path: if a graph was captured on a previous step,
        // just launch it. Skips ~1908 cuLaunchKernel calls for one replay.
        if let Some(graph) = &self.decode_graph {
            graph
                .launch()
                .map_err(|e| LlmError::Backend(format!("graph launch: {e:?}")))?;
            // T246.5.6 — async DtoH into pinned host buffer ; sync only on
            // the dedicated event when reading the value back.
            self.stream
                .memcpy_dtoh(&self.scratch.next_token, &mut self.next_token_host_pinned)
                .map_err(|e| LlmError::Backend(format!("dtoh next_token: {e:?}")))?;
            let token_id = self
                .next_token_host_pinned
                .as_slice()
                .map_err(|e| LlmError::Backend(format!("read pinned: {e:?}")))?[0];
            self.position += 1;
            return Ok(token_id);
        }

        // T246.5.3 — Capture path: on the 2nd decode call (position == 1),
        // wrap the body in begin_capture / end_capture so all kernels become
        // a single replayable CUDA Graph. The 1st call (position == 0) is a
        // warmup that triggers nvrtc compile + cuModuleLoadData for every
        // kernel — these are NOT capturable and must happen before begin_capture.
        // T246.6 — MoE variant uses memcpy_dtov inside the decode body to
        // read top-K indices to host. That host sync is INCOMPATIBLE with
        // CUDA Graph capture (the stream is captured → DtoH stalls and
        // returns garbage). Skip capture for MoE.
        let moe_in_use = matches!(cfg.variant, Qwen35Variant::Moe);
        let should_capture = !moe_in_use && self.position == 1 && self.decode_graph.is_none();
        if should_capture {
            // T246.5.3 — Drain pending stream work before begin_capture. The
            // H2D upload of token_id above is cuMemcpyHtoDAsync on pageable
            // memory, which is *not* a captureable op. If it bleeds into the
            // capture region the capture is invalidated immediately on the
            // next op (CUDA_ERROR_STREAM_CAPTURE_INVALIDATED).
            self.stream
                .synchronize()
                .map_err(|e| LlmError::Backend(format!("pre-capture sync: {e:?}")))?;
            self.stream
                .begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)
                .map_err(|e| LlmError::Backend(format!("begin_capture: {e:?}")))?;
        }

        // T246.4.2 — zero scratch buffers at start of each step. Without this,
        // residual reads of stale buffers cause non-deterministic output across
        // runs even with the same input.
        self.stream
            .memset_zeros(&mut self.scratch.h)
            .map_err(|e| LlmError::Backend(format!("zero h: {e:?}")))?;
        self.stream
            .memset_zeros(&mut self.scratch.h_norm)
            .map_err(|e| LlmError::Backend(format!("zero h_norm: {e:?}")))?;
        self.stream
            .memset_zeros(&mut self.scratch.residual)
            .map_err(|e| LlmError::Backend(format!("zero residual: {e:?}")))?;
        self.stream
            .memset_zeros(&mut self.scratch.qkv_mixed)
            .map_err(|e| LlmError::Backend(format!("zero qkv_mixed: {e:?}")))?;
        self.stream
            .memset_zeros(&mut self.scratch.conv_out)
            .map_err(|e| LlmError::Backend(format!("zero conv_out: {e:?}")))?;
        self.stream
            .memset_zeros(&mut self.scratch.z)
            .map_err(|e| LlmError::Backend(format!("zero z: {e:?}")))?;
        self.stream
            .memset_zeros(&mut self.scratch.alpha)
            .map_err(|e| LlmError::Backend(format!("zero alpha: {e:?}")))?;
        self.stream
            .memset_zeros(&mut self.scratch.beta)
            .map_err(|e| LlmError::Backend(format!("zero beta: {e:?}")))?;
        self.stream
            .memset_zeros(&mut self.scratch.q_v)
            .map_err(|e| LlmError::Backend(format!("zero q_v: {e:?}")))?;
        self.stream
            .memset_zeros(&mut self.scratch.k_v)
            .map_err(|e| LlmError::Backend(format!("zero k_v: {e:?}")))?;
        self.stream
            .memset_zeros(&mut self.scratch.ssm_out_buf)
            .map_err(|e| LlmError::Backend(format!("zero ssm_out: {e:?}")))?;
        self.stream
            .memset_zeros(&mut self.scratch.q_buf)
            .map_err(|e| LlmError::Backend(format!("zero q_buf: {e:?}")))?;
        self.stream
            .memset_zeros(&mut self.scratch.k_buf)
            .map_err(|e| LlmError::Backend(format!("zero k_buf: {e:?}")))?;
        self.stream
            .memset_zeros(&mut self.scratch.v_buf)
            .map_err(|e| LlmError::Backend(format!("zero v_buf: {e:?}")))?;
        self.stream
            .memset_zeros(&mut self.scratch.attn_out)
            .map_err(|e| LlmError::Backend(format!("zero attn_out: {e:?}")))?;
        self.stream
            .memset_zeros(&mut self.scratch.gate_buf)
            .map_err(|e| LlmError::Backend(format!("zero gate: {e:?}")))?;
        self.stream
            .memset_zeros(&mut self.scratch.up_buf)
            .map_err(|e| LlmError::Backend(format!("zero up: {e:?}")))?;
        self.stream
            .memset_zeros(&mut self.scratch.down_buf)
            .map_err(|e| LlmError::Backend(format!("zero down: {e:?}")))?;
        self.stream
            .memset_zeros(&mut self.scratch.logits)
            .map_err(|e| LlmError::Backend(format!("zero logits: {e:?}")))?;

        // Pre-extract device pointers from scratch (drop guards).
        let (
            h_p,
            h_norm_p,
            res_p,
            qkv_mixed_p,
            conv_out_p,
            z_p,
            alpha_p,
            beta_p,
            qv_p,
            kv_v_p,
            sso_p,
            q_p,
            k_p,
            v_p,
            ao_p,
            gate_p,
            up_p,
            _down_p,
            logits_p,
            tok_p,
            final_norm_p,
            x_q8_p,
            gqa_m_p,
            gqa_l_p,
            gqa_o_p,
            moe_router_p,
            moe_idx_p,
            moe_w_p,
            moe_egate_p,
            moe_eup_p,
            moe_eout_p,
            moe_sd_p,
        ) = unsafe {
            let (a, _g0) = self.scratch.h.device_ptr_mut(&self.stream);
            let (b, _g1) = self.scratch.h_norm.device_ptr_mut(&self.stream);
            let (c, _g2) = self.scratch.residual.device_ptr_mut(&self.stream);
            let (d_, _g3) = self.scratch.qkv_mixed.device_ptr_mut(&self.stream);
            let (e, _g4) = self.scratch.conv_out.device_ptr_mut(&self.stream);
            let (f_, _g5) = self.scratch.z.device_ptr_mut(&self.stream);
            let (g, _g6) = self.scratch.alpha.device_ptr_mut(&self.stream);
            let (h_, _g7) = self.scratch.beta.device_ptr_mut(&self.stream);
            let (i, _g8) = self.scratch.q_v.device_ptr_mut(&self.stream);
            let (j, _g9) = self.scratch.k_v.device_ptr_mut(&self.stream);
            let (k_, _g10) = self.scratch.ssm_out_buf.device_ptr_mut(&self.stream);
            let (l, _g11) = self.scratch.q_buf.device_ptr_mut(&self.stream);
            let (m, _g12) = self.scratch.k_buf.device_ptr_mut(&self.stream);
            let (n_, _g13) = self.scratch.v_buf.device_ptr_mut(&self.stream);
            let (o, _g14) = self.scratch.attn_out.device_ptr_mut(&self.stream);
            let (p, _g15) = self.scratch.gate_buf.device_ptr_mut(&self.stream);
            let (q_, _g16) = self.scratch.up_buf.device_ptr_mut(&self.stream);
            let (r, _g17) = self.scratch.down_buf.device_ptr_mut(&self.stream);
            let (s, _g18) = self.scratch.logits.device_ptr_mut(&self.stream);
            let (t, _g19) = self.scratch.next_token.device_ptr_mut(&self.stream);
            let (u, _g20) = self.final_norm.device_ptr(&self.stream);
            // T246.5.5 — only expose the Q8_1 staging ptr when the dp4a
            // path is enabled (env-gated). Passing 0 forces the float v2
            // fallback in dispatch_matmul_m1.
            let v = if self.use_dp4a_q4k {
                let (p_, _g21) = self.scratch.x_q8_scratch.device_ptr_mut(&self.stream);
                p_
            } else {
                0u64
            };
            // T246.5.7 — FlashDecode split-K staging ptrs.
            let (gm_, _g22) = self.scratch.gqa_partial_m.device_ptr_mut(&self.stream);
            let (gl_, _g23) = self.scratch.gqa_partial_l.device_ptr_mut(&self.stream);
            let (go_, _g24) = self.scratch.gqa_partial_o.device_ptr_mut(&self.stream);
            // T246.6 — MoE staging ptrs.
            let (mr_, _g25) = self.scratch.moe_router_logits.device_ptr_mut(&self.stream);
            let (mi_, _g26) = self.scratch.moe_topk_idx.device_ptr_mut(&self.stream);
            let (mw_, _g27) = self.scratch.moe_topk_w.device_ptr_mut(&self.stream);
            let (mg_, _g28) = self.scratch.moe_expert_gate.device_ptr_mut(&self.stream);
            let (mu_, _g29) = self.scratch.moe_expert_up.device_ptr_mut(&self.stream);
            let (mo_, _g30) = self.scratch.moe_expert_out.device_ptr_mut(&self.stream);
            let (msd_, _g31) = self.scratch.moe_shexp_dot.device_ptr_mut(&self.stream);
            (
                a, b, c, d_, e, f_, g, h_, i, j, k_, l, m, n_, o, p, q_, r, s, t, u, v, gm_, gl_,
                go_, mr_, mi_, mw_, mg_, mu_, mo_, msd_,
            )
        };

        // T246.5.3 — removed `self.stream.synchronize()` here. host syncs are
        // INVALID during CUDA Graph capture. The memset_zeros above are
        // ordered on the same stream as the kernels below, so they execute
        // in order without needing a host barrier.

        // T246.4.3 — DEBUG : if env var set, dump KV cache + SSM state values
        // BEFORE any computation, to verify alloc_zeros gave us actual zeros.
        if self.position == 0 && std::env::var("RUSTORCH_DEBUG_INIT").is_ok() {
            if let Some(kv) = self.kv_caches.first() {
                let k_bytes: Vec<half::bf16> = self
                    .stream
                    .memcpy_dtov(&kv.k)
                    .map_err(|e| LlmError::Backend(format!("dl k: {e:?}")))?;
                let nz = k_bytes
                    .iter()
                    .take(1024)
                    .filter(|&&v| v.to_f32() != 0.0)
                    .count();
                let sample: Vec<f32> = k_bytes.iter().take(8).map(|v| v.to_f32()).collect();
                eprintln!("[debug-init] kv0.k first 8 = {sample:?} ({nz}/1024 nonzero)");
            }
            if let Some(ss) = self.ssm_states.first() {
                let s_bytes: Vec<half::bf16> = self
                    .stream
                    .memcpy_dtov(&ss.state)
                    .map_err(|e| LlmError::Backend(format!("dl state: {e:?}")))?;
                let nz = s_bytes
                    .iter()
                    .take(1024)
                    .filter(|&&v| v.to_f32() != 0.0)
                    .count();
                let sample: Vec<f32> = s_bytes.iter().take(8).map(|v| v.to_f32()).collect();
                eprintln!("[debug-init] ssm0.state first 8 = {sample:?} ({nz}/1024 nonzero)");
            }
            // Also dump token_emb row for token_id=1 to verify deterministic load.
            let te_bytes: Vec<half::bf16> = self
                .stream
                .memcpy_dtov(&self.token_emb)
                .map_err(|e| LlmError::Backend(format!("dl te: {e:?}")))?;
            let row_off = (token_id as usize) * d;
            let sample: Vec<f32> = te_bytes
                .iter()
                .skip(row_off)
                .take(8)
                .map(|v| v.to_f32())
                .collect();
            eprintln!("[debug-init] token_emb[token_id={token_id}] first 8 = {sample:?}");
        }

        // ---- Step 0 : Load h from token_emb[current_token_dev, :] ----
        // T246.5.3 — use embedding_lookup_bf16 with token_id read from device.
        // This makes the lookup CUDA-Graph-capturable (was: host pointer
        // arithmetic on token_id).
        unsafe {
            let (te_p, _g) = self.token_emb.device_ptr(&self.stream);
            let (ct_p, _g2) = self.current_token_dev.device_ptr(&self.stream);
            self.kernels
                .embedding_lookup_bf16(&self.stream, te_p, ct_p, h_p, 1, d as i32)
                .map_err(|e| LlmError::Backend(format!("embed lookup: {e:?}")))?;
        }
        // T246.5.3 — removed sync here, same reason as above.

        // ---- Iterate over all 64 layers ----
        for (li, block) in self.blocks.iter_mut().enumerate() {
            // residual = h
            unsafe {
                self.kernels
                    .copy_bf16(&self.stream, res_p, h_p, d as i32)
                    .map_err(|e| LlmError::Backend(format!("copy res: {e:?}")))?;
            }

            match block {
                BlockQ4K::Ssm(ssm) => {
                    // 1. h_norm = rms_norm(h, attn_norm)
                    unsafe {
                        let (an, _g) = ssm.attn_norm.device_ptr(&self.stream);
                        // rms_norm_bf16 is in-place on x; copy h → h_norm first.
                        self.kernels
                            .copy_bf16(&self.stream, h_norm_p, h_p, d as i32)
                            .map_err(|e| LlmError::Backend(format!("copy h_norm: {e:?}")))?;
                        self.kernels
                            .rms_norm_bf16(&self.stream, h_norm_p, an, eps, d as i32, 1)
                            .map_err(|e| LlmError::Backend(format!("rms_norm: {e:?}")))?;
                    }

                    // 2. qkv_mixed = w_qkv @ h_norm
                    ssm.w_qkv.dispatch_matmul_m1(
                        &self.kernels,
                        &self.stream,
                        h_norm_p,
                        qkv_mixed_p,
                        x_q8_p,
                    )?;

                    // 3. z = w_gate @ h_norm
                    ssm.w_gate.dispatch_matmul_m1(
                        &self.kernels,
                        &self.stream,
                        h_norm_p,
                        z_p,
                        x_q8_p,
                    )?;

                    // 4. alpha = w_alpha @ h_norm
                    ssm.w_alpha.dispatch_matmul_m1(
                        &self.kernels,
                        &self.stream,
                        h_norm_p,
                        alpha_p,
                        x_q8_p,
                    )?;

                    // 5. beta_logit = w_beta @ h_norm ; beta = sigmoid
                    ssm.w_beta.dispatch_matmul_m1(
                        &self.kernels,
                        &self.stream,
                        h_norm_p,
                        beta_p,
                        x_q8_p,
                    )?;
                    unsafe {
                        self.kernels
                            .sigmoid_inplace_bf16(&self.stream, beta_p, n_v as i32)
                            .map_err(|e| LlmError::Backend(format!("sigmoid: {e:?}")))?;
                    }

                    // 6. alpha += dt_bias ; softplus(alpha) ; alpha *= ssm_a → gate_h
                    unsafe {
                        let (db, _g) = ssm.dt_bias.device_ptr(&self.stream);
                        self.kernels
                            .add_inplace_bf16(&self.stream, alpha_p, db, n_v as i32)
                            .map_err(|e| LlmError::Backend(format!("alpha+dt_bias: {e:?}")))?;
                        self.kernels
                            .softplus_inplace_bf16(&self.stream, alpha_p, n_v as i32)
                            .map_err(|e| LlmError::Backend(format!("softplus: {e:?}")))?;
                        let (sa, _g2) = ssm.ssm_a.device_ptr(&self.stream);
                        self.kernels
                            .mul_inplace_bf16(&self.stream, alpha_p, sa, n_v as i32)
                            .map_err(|e| LlmError::Backend(format!("mul ssm_a: {e:?}")))?;
                    }
                    // alpha is now gate_h (alpha_softplus * ssm_a).

                    // 7. conv_out = conv1d_depthwise(conv1d, conv_state, qkv_mixed)
                    let ssm_state_layer = &mut self.ssm_states[get_ssm_layer_idx(&cfg, li)];
                    unsafe {
                        let (cw, _g) = ssm.conv1d.device_ptr(&self.stream);
                        let (cs, _g2) = ssm_state_layer.conv_state.device_ptr_mut(&self.stream);
                        self.kernels
                            .conv1d_depthwise_bf16(
                                &self.stream,
                                cw,
                                cs,
                                qkv_mixed_p,
                                conv_out_p,
                                conv_dim as i32,
                                conv_kernel as i32,
                            )
                            .map_err(|e| LlmError::Backend(format!("conv1d: {e:?}")))?;
                    }

                    // 8. silu(conv_out)
                    unsafe {
                        self.kernels
                            .silu_bf16(&self.stream, conv_out_p, conv_dim as i32)
                            .map_err(|e| LlmError::Backend(format!("silu conv: {e:?}")))?;
                    }

                    // 9. q,k,v = split(conv_out)
                    let q_ptr = conv_out_p;
                    let k_ptr = conv_out_p + (key_dim * 2) as u64;
                    let v_ptr = conv_out_p + (2 * key_dim * 2) as u64;

                    // 10. l2_norm_per_head on q (n_k heads of head_kv) and k
                    unsafe {
                        self.kernels
                            .l2_norm_per_head_bf16(
                                &self.stream,
                                q_ptr,
                                n_k as i32,
                                head_kv as i32,
                                eps,
                            )
                            .map_err(|e| LlmError::Backend(format!("l2 q: {e:?}")))?;
                        self.kernels
                            .l2_norm_per_head_bf16(
                                &self.stream,
                                k_ptr,
                                n_k as i32,
                                head_kv as i32,
                                eps,
                            )
                            .map_err(|e| LlmError::Backend(format!("l2 k: {e:?}")))?;
                    }

                    // 11. Broadcast q,k from n_k to n_v heads (factor n_v / n_k)
                    let q_for_delta = if n_k == n_v {
                        q_ptr
                    } else {
                        unsafe {
                            self.kernels
                                .repeat_heads_bf16(
                                    &self.stream,
                                    q_ptr,
                                    qv_p,
                                    n_k as i32,
                                    n_v as i32,
                                    head_kv as i32,
                                )
                                .map_err(|e| LlmError::Backend(format!("repeat q: {e:?}")))?;
                        }
                        qv_p
                    };
                    let k_for_delta = if n_k == n_v {
                        k_ptr
                    } else {
                        unsafe {
                            self.kernels
                                .repeat_heads_bf16(
                                    &self.stream,
                                    k_ptr,
                                    kv_v_p,
                                    n_k as i32,
                                    n_v as i32,
                                    head_kv as i32,
                                )
                                .map_err(|e| LlmError::Backend(format!("repeat k: {e:?}")))?;
                        }
                        kv_v_p
                    };

                    // 12. delta_net_step : state update + out
                    unsafe {
                        let (st, _g) = ssm_state_layer.state.device_ptr_mut(&self.stream);
                        self.kernels
                            .delta_net_step_bf16(
                                &self.stream,
                                q_for_delta,
                                k_for_delta,
                                v_ptr,
                                alpha_p, // gate_h
                                beta_p,
                                st,
                                sso_p,
                                n_v as i32,
                                head_kv as i32,
                            )
                            .map_err(|e| LlmError::Backend(format!("delta_net: {e:?}")))?;
                    }

                    // 13. ssm_norm per head + multiply by silu(z)
                    unsafe {
                        let (sn, _g) = ssm.ssm_norm.device_ptr(&self.stream);
                        // RMSNorm per head : we have rms_norm_bf16 with batch parameter.
                        // Use n_v batches, each of size head_kv, with same gamma.
                        self.kernels
                            .rms_norm_bf16(&self.stream, sso_p, sn, eps, head_kv as i32, n_v as i32)
                            .map_err(|e| LlmError::Backend(format!("ssm_norm: {e:?}")))?;
                        // silu(z) inplace
                        self.kernels
                            .silu_bf16(&self.stream, z_p, value_dim as i32)
                            .map_err(|e| LlmError::Backend(format!("silu z: {e:?}")))?;
                        // out *= silu(z)
                        self.kernels
                            .mul_inplace_bf16(&self.stream, sso_p, z_p, value_dim as i32)
                            .map_err(|e| LlmError::Backend(format!("mul gated: {e:?}")))?;
                    }

                    // 14. h = ssm_out @ gated  (overwrites h)
                    ssm.ssm_out.dispatch_matmul_m1(
                        &self.kernels,
                        &self.stream,
                        sso_p,
                        h_p,
                        x_q8_p,
                    )?;

                    // 15. residual : h += residual
                    unsafe {
                        self.kernels
                            .add_inplace_bf16(&self.stream, h_p, res_p, d as i32)
                            .map_err(|e| LlmError::Backend(format!("ssm residual: {e:?}")))?;
                    }
                },
                BlockQ4K::Attn(attn) => {
                    // T246.3 — full attention block forward at M=1.

                    // 1. h_norm = rms_norm(h, attn_norm)
                    unsafe {
                        let (an, _g) = attn.attn_norm.device_ptr(&self.stream);
                        self.kernels
                            .copy_bf16(&self.stream, h_norm_p, h_p, d as i32)
                            .map_err(|e| LlmError::Backend(format!("copy h_norm attn: {e:?}")))?;
                        self.kernels
                            .rms_norm_bf16(&self.stream, h_norm_p, an, eps, d as i32, 1)
                            .map_err(|e| LlmError::Backend(format!("rms_norm attn: {e:?}")))?;
                    }

                    // 2. qg = w_q @ h_norm    (output dim 2*q_dim, fused Q + per-head gate)
                    let (qg_n, _qg_k) = attn.w_q.shape();
                    debug_assert_eq!(qg_n, 2 * q_dim);
                    // Use the conv_out_p buffer as a scratch (size conv_dim ≥ 2*q_dim
                    // typically) — except for Qwen3.6-27B where conv_dim=10240 and
                    // 2*q_dim=12288. We need a dedicated buffer. Use qkv_mixed (conv_dim)
                    // and check it fits, else use h_norm... Hmm.
                    // Safer : allocate a fresh scratch sized for 2*q_dim.
                    // For decode (single token) this is small.

                    // Reuse `up_buf` (size f=17408 ≥ 2*q_dim) as QG scratch.
                    let qg_p = up_p;
                    attn.w_q.dispatch_matmul_m1(
                        &self.kernels,
                        &self.stream,
                        h_norm_p,
                        qg_p,
                        x_q8_p,
                    )?;

                    // 3. Split qg into q (q_dim) and gate (q_dim, used as sigmoid gate).
                    //    Use q_buf and (reuse) attn_out as gate buffer.
                    unsafe {
                        self.kernels
                            .split_qg_bf16(
                                &self.stream,
                                qg_p,
                                q_p,
                                ao_p, // store gate here temporarily
                                n_q as i32,
                                head_dim as i32,
                            )
                            .map_err(|e| LlmError::Backend(format!("split_qg: {e:?}")))?;
                    }

                    // 4. K, V projections.
                    attn.w_k.dispatch_matmul_m1(
                        &self.kernels,
                        &self.stream,
                        h_norm_p,
                        k_p,
                        x_q8_p,
                    )?;
                    attn.w_v.dispatch_matmul_m1(
                        &self.kernels,
                        &self.stream,
                        h_norm_p,
                        v_p,
                        x_q8_p,
                    )?;

                    // 5. Per-head Q-norm and K-norm (RMSNorm with shared gamma).
                    unsafe {
                        let (qn, _g1) = attn.q_norm.device_ptr(&self.stream);
                        let (kn, _g2) = attn.k_norm.device_ptr(&self.stream);
                        // rms_norm_bf16(x, gamma, eps, n, batch) where each batch is size n.
                        self.kernels
                            .rms_norm_bf16(&self.stream, q_p, qn, eps, head_dim as i32, n_q as i32)
                            .map_err(|e| LlmError::Backend(format!("q_norm: {e:?}")))?;
                        self.kernels
                            .rms_norm_bf16(&self.stream, k_p, kn, eps, head_dim as i32, n_kv as i32)
                            .map_err(|e| LlmError::Backend(format!("k_norm: {e:?}")))?;
                    }

                    // 6. RoPE on q (n_q heads) and k (n_kv heads).
                    // Qwen3.6 : rope_dim=64 < head_dim=256. Only first 64 dims rotate.
                    // T246.5.3 — devcnt variant reads `pos` from position_dev.
                    let position = self.position;
                    let rope_dim = cfg.rope_dim;
                    unsafe {
                        let (inv_p, _g_inv) = self.rope_freqs.inv_freq.device_ptr(&self.stream);
                        let (pos_p, _g_pos) = self.position_dev.device_ptr(&self.stream);
                        self.kernels
                            .rope_partial_bf16_devcnt(
                                &self.stream,
                                q_p,
                                inv_p,
                                pos_p,
                                n_q as i32,
                                head_dim as i32,
                                rope_dim as i32,
                            )
                            .map_err(|e| LlmError::Backend(format!("rope q: {e:?}")))?;
                        self.kernels
                            .rope_partial_bf16_devcnt(
                                &self.stream,
                                k_p,
                                inv_p,
                                pos_p,
                                n_kv as i32,
                                head_dim as i32,
                                rope_dim as i32,
                            )
                            .map_err(|e| LlmError::Backend(format!("rope k: {e:?}")))?;
                    }

                    // 7. Append k, v to KV cache at slot `*position_dev`.
                    // T246.5.3 — single graph-capturable kernel handles both
                    // K and V, with slot index read from device pointer.
                    let attn_idx = get_attn_layer_idx(&cfg, li);
                    let kv_cache = &mut self.kv_caches[attn_idx];
                    unsafe {
                        let (kc_p, _g1) = kv_cache.k.device_ptr_mut(&self.stream);
                        let (vc_p, _g2) = kv_cache.v.device_ptr_mut(&self.stream);
                        let (pos_p, _g3) = self.position_dev.device_ptr(&self.stream);
                        self.kernels
                            .kv_append_bf16_devcnt(
                                &self.stream,
                                kc_p,
                                vc_p,
                                k_p,
                                v_p,
                                pos_p,
                                kv_dim as i32,
                            )
                            .map_err(|e| LlmError::Backend(format!("kv append: {e:?}")))?;
                    }

                    // 8. GQA decode — T246.5.7 FlashDecode-V2 split-K kernel
                    // (4× more SM occupancy than the serial online kernel by
                    // splitting kv_len work across GQA_N_SPLIT blocks/head).
                    let _ = position; // kept for the host-side KV append above
                    unsafe {
                        let (kc_p, _g1) = kv_cache.k.device_ptr(&self.stream);
                        let (vc_p, _g2) = kv_cache.v.device_ptr(&self.stream);
                        let (kv_len_p, _g3) = self.kv_len_dev.device_ptr(&self.stream);
                        self.kernels
                            .gqa_decode_split_bf16(
                                &self.stream,
                                q_p,
                                kc_p,
                                vc_p,
                                gate_p, // attn raw output (size f ≥ q_dim)
                                gqa_m_p,
                                gqa_l_p,
                                gqa_o_p,
                                n_q as i32,
                                n_kv as i32,
                                kv_len_p,
                                head_dim as i32,
                                self.max_seq as i32,
                                GQA_N_SPLIT as i32,
                            )
                            .map_err(|e| LlmError::Backend(format!("gqa_decode_split: {e:?}")))?;
                    }

                    // 9. sigmoid(gate) ; attn_out *= gate
                    unsafe {
                        self.kernels
                            .sigmoid_inplace_bf16(&self.stream, ao_p, q_dim as i32)
                            .map_err(|e| LlmError::Backend(format!("sigmoid gate: {e:?}")))?;
                        self.kernels
                            .mul_inplace_bf16(&self.stream, gate_p, ao_p, q_dim as i32)
                            .map_err(|e| LlmError::Backend(format!("attn*gate: {e:?}")))?;
                    }

                    // 10. h = w_o @ gated_attn  (output dim d)
                    attn.w_o.dispatch_matmul_m1(
                        &self.kernels,
                        &self.stream,
                        gate_p,
                        h_p,
                        x_q8_p,
                    )?;

                    // 11. residual : h += residual
                    unsafe {
                        self.kernels
                            .add_inplace_bf16(&self.stream, h_p, res_p, d as i32)
                            .map_err(|e| LlmError::Backend(format!("attn residual: {e:?}")))?;
                    }
                },
            }

            // ---- FFN ----
            let (ffn_ref, post_norm) = match block {
                BlockQ4K::Ssm(s) => (&s.ffn, &s.post_norm),
                BlockQ4K::Attn(a) => (&a.ffn, &a.post_norm),
            };
            // residual = h (pre-FFN value, after attn/ssm + first residual)
            unsafe {
                self.kernels
                    .copy_bf16(&self.stream, res_p, h_p, d as i32)
                    .map_err(|e| LlmError::Backend(format!("copy res ffn: {e:?}")))?;
            }
            // h_norm = rms_norm(h, post_norm)
            unsafe {
                let (pn, _g) = post_norm.device_ptr(&self.stream);
                self.kernels
                    .copy_bf16(&self.stream, h_norm_p, h_p, d as i32)
                    .map_err(|e| LlmError::Backend(format!("copy h_norm ffn: {e:?}")))?;
                self.kernels
                    .rms_norm_bf16(&self.stream, h_norm_p, pn, eps, d as i32, 1)
                    .map_err(|e| LlmError::Backend(format!("rms_norm post: {e:?}")))?;
            }
            match ffn_ref {
                FfnQ4K::Dense { gate, up, down } => {
                    // gate = w_gate @ h_norm ; up = w_up @ h_norm ; gate = silu(gate) * up
                    gate.dispatch_matmul_m1(&self.kernels, &self.stream, h_norm_p, gate_p, x_q8_p)?;
                    up.dispatch_matmul_m1(&self.kernels, &self.stream, h_norm_p, up_p, x_q8_p)?;
                    unsafe {
                        self.kernels
                            .swiglu_bf16(&self.stream, gate_p, up_p, gate_p, f as i32)
                            .map_err(|e| LlmError::Backend(format!("swiglu: {e:?}")))?;
                    }
                    // h = w_down @ gate
                    down.dispatch_matmul_m1(&self.kernels, &self.stream, gate_p, h_p, x_q8_p)?;
                },
                FfnQ4K::Moe(moe) => {
                    moe_ffn_forward_step(
                        moe,
                        &self.kernels,
                        &self.stream,
                        &cfg,
                        h_norm_p,
                        h_p,
                        x_q8_p,
                        moe_router_p,
                        moe_idx_p,
                        &self.scratch.moe_topk_idx,
                        moe_w_p,
                        &self.scratch.moe_topk_w,
                        moe_egate_p,
                        moe_eup_p,
                        moe_eout_p,
                        moe_sd_p,
                        &self.scratch.moe_shexp_dot,
                    )?;
                },
            }
            // h += residual
            unsafe {
                self.kernels
                    .add_inplace_bf16(&self.stream, h_p, res_p, d as i32)
                    .map_err(|e| LlmError::Backend(format!("ffn residual: {e:?}")))?;
            }
        }

        // ---- Final RMSNorm + LM head ----
        unsafe {
            self.kernels
                .rms_norm_bf16(&self.stream, h_p, final_norm_p, eps, d as i32, 1)
                .map_err(|e| LlmError::Backend(format!("final_norm: {e:?}")))?;
        }
        self.lm_head
            .dispatch_matmul_m1(&self.kernels, &self.stream, h_p, logits_p, x_q8_p)?;

        // ---- Sample (argmax for now) ----
        unsafe {
            self.kernels
                .argmax_bf16(&self.stream, logits_p, tok_p, cfg.vocab as i32)
                .map_err(|e| LlmError::Backend(format!("argmax: {e:?}")))?;
        }
        let _ = (tok_p, logits_p, final_norm_p); // silence unused warnings

        // T246.5.3 — Auto-advance device counters at the end of body. These
        // become part of the captured graph so each replay also advances the
        // counters, eliminating the need for host-side writes between replays.
        unsafe {
            let (pos_p, _g_pos) = self.position_dev.device_ptr_mut(&self.stream);
            self.kernels
                .increment_u32_dev(&self.stream, pos_p)
                .map_err(|e| LlmError::Backend(format!("inc position_dev: {e:?}")))?;
        }
        unsafe {
            let (kvl_p, _g_kvl) = self.kv_len_dev.device_ptr_mut(&self.stream);
            self.kernels
                .increment_u32_dev(&self.stream, kvl_p)
                .map_err(|e| LlmError::Backend(format!("inc kv_len_dev: {e:?}")))?;
        }

        // T246.5.3 — Capture path closure: end_capture + first launch.
        if should_capture {
            let graph = self
                .stream
                .end_capture(
                    CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
                )
                .map_err(|e| LlmError::Backend(format!("end_capture: {e:?}")))?
                .ok_or_else(|| LlmError::Backend("end_capture returned no graph".into()))?;
            // Launch once now to actually execute the recorded body for this step.
            graph
                .launch()
                .map_err(|e| LlmError::Backend(format!("first graph launch: {e:?}")))?;
            self.decode_graph = Some(graph);
        }

        // Read back next token (host op, OUTSIDE any capture). T246.5.6 —
        // async DtoH to pinned host memory + event-based sync.
        self.stream
            .memcpy_dtoh(&self.scratch.next_token, &mut self.next_token_host_pinned)
            .map_err(|e| LlmError::Backend(format!("dtoh token: {e:?}")))?;
        let token_id = self
            .next_token_host_pinned
            .as_slice()
            .map_err(|e| LlmError::Backend(format!("read pinned: {e:?}")))?[0];

        self.position += 1;
        Ok(token_id)
    }

    /// Process a prompt at once. **STATUS T246.1** : returns Err. T246.5.
    pub fn prefill_tokens(
        &mut self,
        _token_ids: &[u32],
        _start_pos: usize,
    ) -> Result<u32, LlmError> {
        Err(LlmError::Backend(
            "Qwen35ModelCudaQ4K::prefill_tokens not yet implemented (T246.5)".into(),
        ))
    }

    /// T246.7 P1.3d — tree-aware decode step for Lookahead Decoding (RFC
    /// f0e68045, design D1).
    ///
    /// Takes a draft tree encoded as a flat BFS-ordered array :
    /// - `drafts[r]`  = the token id at tree node r ; `drafts[0]` is the
    ///   "seed" (the most-recently-accepted token from the previous step).
    /// - `parents[r]` = parent node index in the tree, root (r=0) has -1.
    /// - `depths[r]`  = depth of node r (root depth = 0).
    ///
    /// All inputs must have the same length, ≤ `MAX_TREE_SIZE`. Returns the
    /// accepted token suffix `[1..=accept_len]` ; the seed (`drafts[0]`)
    /// itself is NOT returned (it was already in the model's KV state).
    ///
    /// **Phase 1 scope** : tree_size = 1 is a strict drop-in replacement for
    /// `decode_step(drafts[0])` — full kernel parity (including CUDA Graph
    /// replay) and bit-exact output. tree_size > 1 returns `LlmError::Backend`
    /// for now ; the multi-branch forward + acceptance walk lands in P1.4
    /// alongside the full Lookahead manager loop. The CUDA primitives needed
    /// for tree_size > 1 (`kv_append_tree_bf16`, `gqa_decode_tree_bf16`,
    /// `argmax_logits_tree_bf16`, `add_u32_dev`, `set_u32_dev`) are already
    /// in place ; only the per-tree-token forward orchestration is deferred.
    ///
    /// Per RFC design decisions :
    ///   - D1 : NEW method, never modifies `decode_step`.
    ///   - D2 : write-then-truncate KV (no scratch cache copy).
    ///   - D4 : no graph capture for verify (the inner `decode_step` call
    ///          may use its own captured graph for tree_size=1 ; tree_size>1
    ///          would explicitly disable capture).
    pub fn decode_step_tree(
        &mut self,
        drafts: &[u32],
        parents: &[i32],
        depths: &[u8],
    ) -> Result<Vec<u32>, LlmError> {
        // ── Validation ───────────────────────────────────────────────────
        if drafts.is_empty() {
            return Err(LlmError::Backend(
                "decode_step_tree: drafts must be non-empty".into(),
            ));
        }
        if drafts.len() != parents.len() || drafts.len() != depths.len() {
            return Err(LlmError::Backend(format!(
                "decode_step_tree: length mismatch — drafts={}, parents={}, depths={}",
                drafts.len(),
                parents.len(),
                depths.len()
            )));
        }
        if drafts.len() > MAX_TREE_SIZE {
            return Err(LlmError::Backend(format!(
                "decode_step_tree: tree_size {} exceeds MAX_TREE_SIZE {}",
                drafts.len(),
                MAX_TREE_SIZE
            )));
        }
        if parents[0] != -1 {
            return Err(LlmError::Backend(format!(
                "decode_step_tree: root must have parent=-1, got {}",
                parents[0]
            )));
        }
        if depths[0] != 0 {
            return Err(LlmError::Backend(format!(
                "decode_step_tree: root must have depth=0, got {}",
                depths[0]
            )));
        }
        for (r, &p) in parents.iter().enumerate().skip(1) {
            if p < 0 || (p as usize) >= r {
                return Err(LlmError::Backend(format!(
                    "decode_step_tree: parents[{r}] = {p} must be in [0, {r})"
                )));
            }
            let expected_depth = depths[p as usize] + 1;
            if depths[r] != expected_depth {
                return Err(LlmError::Backend(format!(
                    "decode_step_tree: depths[{r}] = {} must equal depths[parent={}] + 1 = {}",
                    depths[r], p, expected_depth
                )));
            }
        }

        let tree_size = drafts.len();

        // ── tree_size = 1 : delegate to decode_step (bit-exact equivalence) ──
        // The root's own KV is at slot kv_len-1, fully covered by Phase A
        // of the existing M=1 attention path. The "tree" semantics
        // degenerate to a single forward pass on drafts[0], whose output
        // we return as the accepted token suffix of length 1.
        if tree_size == 1 {
            let tok = self.decode_step(drafts[0])?;
            return Ok(vec![tok]);
        }

        // ── tree_size > 1 : full multi-token tree forward (P1.4) ─────────
        // The CUDA primitives are wired and parity-tested — see the
        // `kv_append_tree_bf16`, `gqa_decode_tree_bf16`, `argmax_logits_tree_bf16`,
        // `add_u32_dev`, `set_u32_dev` kernels. The orchestration of the
        // per-tree-token forward (loop-batched embedding/RMSNorm/QKV/RoPE
        // + tree-aware attention + per-token gate/W_o/FFN, then host-side
        // acceptance walk + position counter advance) ships in P1.4 along
        // with the host-side Lookahead manager that produces non-trivial
        // tree topologies.
        Err(LlmError::Backend(format!(
            "decode_step_tree: tree_size > 1 (got {tree_size}) is not yet implemented; \
             P1.3 ships the CUDA kernels and the tree_size=1 path. Full multi-token \
             forward lands in P1.4 alongside the Lookahead manager. Use \
             decode_step(drafts[0]) for now."
        )))
    }
}

/// T246.6.5 — MoE FFN forward step (Qwen3.6-35B-A3B).
///
/// Computes one token's MoE output : top-K routed experts (each a Q4_K
/// SwiGLU FFN) + parallel shared expert. Writes the routed sum into `h_p`
/// (which is then accumulated with the residual by the caller).
///
/// Caller passes the typed slice references for the three buffers we
/// memcpy_dtov from (top-K indices, top-K weights, shared-expert dot
/// product) — Rust's borrow checker accepts these alongside the loop's
/// `&mut self.blocks` mutable borrow because they're disjoint fields of
/// `self.scratch`. The same buffers' raw u64 ptrs are passed for kernel
/// arguments (since `LlmKernels::*_bf16` APIs are u64-typed).
///
/// Semantics mirror `qwen35_cpu::ffn_moe_forward`.
#[allow(clippy::too_many_arguments)]
fn moe_ffn_forward_step(
    moe: &MoeFfnQ4K,
    kernels: &LlmKernels,
    stream: &Arc<CudaStream>,
    cfg: &Qwen35Config,
    h_norm_p: u64,
    h_p: u64,
    x_q8_p: u64,
    router_logits_p: u64,
    topk_idx_p: u64,
    topk_idx_slice: &CudaSlice<i32>,
    topk_w_p: u64,
    topk_w_slice: &CudaSlice<half::bf16>,
    expert_gate_p: u64,
    expert_up_p: u64,
    expert_out_p: u64,
    shexp_dot_p: u64,
    shexp_dot_slice: &CudaSlice<half::bf16>,
) -> Result<(), LlmError> {
    let d = cfg.d as i32;
    let ef = cfg.expert_f as i32;
    let n_e = cfg.n_experts as i32;
    let k = cfg.n_experts_used as i32;

    // ---- 1. Router logits ----
    moe.gate_inp
        .dispatch_matmul_m1(kernels, stream, h_norm_p, router_logits_p, x_q8_p)?;

    // ---- 2. Top-K softmax → indices + renormalized weights on device ----
    unsafe {
        kernels
            .topk_softmax_bf16(stream, router_logits_p, topk_idx_p, topk_w_p, n_e, k)
            .map_err(|e| LlmError::Backend(format!("topk_softmax: {e:?}")))?;
    }

    // ---- 3. Download top-K to host (one DtoH sync per layer) ----
    // Note : this sync is incompatible with CUDA Graph capture ; the model
    // ctor disables graph capture when variant == Moe (handled elsewhere
    // by the `position == 1 && decode_graph.is_none()` guard, since the
    // capture path branch is taken only after step 0 warmup).
    let topk_idx: Vec<i32> = stream
        .memcpy_dtov(topk_idx_slice)
        .map_err(|e| LlmError::Backend(format!("dtov topk_idx: {e:?}")))?;
    let topk_w_bf: Vec<half::bf16> = stream
        .memcpy_dtov(topk_w_slice)
        .map_err(|e| LlmError::Backend(format!("dtov topk_w: {e:?}")))?;

    // ---- 4. Zero h_p (start fresh accumulation for routed sum) ----
    // Use scaled_add with y=x=h_p, alpha=-1 → h_p -= h_p = 0 (aliasing OK
    // since each thread reads then writes its own index in one go).
    unsafe {
        kernels
            .scaled_add_inplace_bf16(stream, h_p, h_p, -1.0, d)
            .map_err(|e| LlmError::Backend(format!("zero h_p: {e:?}")))?;
    }

    // ---- 5. Routed experts loop ----
    for i in 0..k as usize {
        let e_idx = topk_idx[i] as usize;
        let w_e = topk_w_bf[i].to_f32();
        if e_idx >= moe.gate_exps.len() {
            return Err(LlmError::Backend(format!(
                "MoE expert idx {e_idx} out of range (n_experts={})",
                moe.gate_exps.len()
            )));
        }

        moe.gate_exps[e_idx].dispatch_matmul_m1(
            kernels,
            stream,
            h_norm_p,
            expert_gate_p,
            x_q8_p,
        )?;
        moe.up_exps[e_idx].dispatch_matmul_m1(kernels, stream, h_norm_p, expert_up_p, x_q8_p)?;
        unsafe {
            kernels
                .swiglu_bf16(stream, expert_gate_p, expert_up_p, expert_gate_p, ef)
                .map_err(|e| LlmError::Backend(format!("swiglu expert {e_idx}: {e:?}")))?;
        }
        moe.down_exps[e_idx].dispatch_matmul_m1(
            kernels,
            stream,
            expert_gate_p,
            expert_out_p,
            x_q8_p,
        )?;
        unsafe {
            kernels
                .scaled_add_inplace_bf16(stream, h_p, expert_out_p, w_e, d)
                .map_err(|e| LlmError::Backend(format!("scaled_add expert {e_idx}: {e:?}")))?;
        }
    }

    // ---- 6. Shared expert (parallel path) ----
    // shexp_dot = gate_inp_shexp · h_norm  (1×D BF16 matmul)
    unsafe {
        use cudarc::driver::DevicePtr;
        let (gip_p, _g) = moe.gate_inp_shexp.device_ptr(stream);
        kernels
            .sgemv_bf16_bf16(stream, gip_p, h_norm_p, shexp_dot_p, 1, d)
            .map_err(|e| LlmError::Backend(format!("shexp dot: {e:?}")))?;
    }
    let dot_bf: Vec<half::bf16> = stream
        .memcpy_dtov(shexp_dot_slice)
        .map_err(|e| LlmError::Backend(format!("dtov shexp_dot: {e:?}")))?;
    let shexp_w = {
        let v = dot_bf[0].to_f32();
        1.0_f32 / (1.0 + (-v).exp())
    };

    moe.gate_shexp
        .dispatch_matmul_m1(kernels, stream, h_norm_p, expert_gate_p, x_q8_p)?;
    moe.up_shexp
        .dispatch_matmul_m1(kernels, stream, h_norm_p, expert_up_p, x_q8_p)?;
    unsafe {
        kernels
            .swiglu_bf16(stream, expert_gate_p, expert_up_p, expert_gate_p, ef)
            .map_err(|e| LlmError::Backend(format!("swiglu shexp: {e:?}")))?;
    }
    moe.down_shexp
        .dispatch_matmul_m1(kernels, stream, expert_gate_p, expert_out_p, x_q8_p)?;
    unsafe {
        kernels
            .scaled_add_inplace_bf16(stream, h_p, expert_out_p, shexp_w, d)
            .map_err(|e| LlmError::Backend(format!("scaled_add shexp: {e:?}")))?;
    }

    Ok(())
}
