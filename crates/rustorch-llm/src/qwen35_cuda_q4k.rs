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
use cudarc::driver::{CudaContext, CudaSlice, CudaStream};
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
    pub(crate) fn dispatch_matmul_m1(
        &self,
        kernels: &LlmKernels,
        stream: &Arc<CudaStream>,
        x: u64,
        y: u64,
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
                    kernels
                        .sgemv_q4k_bf16_v2(stream, w, x, y, *n as i32, *k as i32)
                        .map_err(|e| LlmError::Backend(format!("sgemv_q4k: {e:?}")))
                },
                QuantTensor::Q5K { bytes, n, k } => {
                    let (w, _g) = bytes.device_ptr(stream);
                    kernels
                        .sgemv_q5k_bf16(stream, w, x, y, *n as i32, *k as i32)
                        .map_err(|e| LlmError::Backend(format!("sgemv_q5k: {e:?}")))
                },
                QuantTensor::Q6K { bytes, n, k } => {
                    let (w, _g) = bytes.device_ptr(stream);
                    kernels
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
    /// FFN dense (gate/up/down).
    pub(crate) w_gate_ffn: QuantTensor,
    pub(crate) w_up_ffn: QuantTensor,
    pub(crate) w_down_ffn: QuantTensor,
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
    /// FFN dense (gate/up/down). Same as attention block.
    pub(crate) w_gate_ffn: QuantTensor,
    pub(crate) w_up_ffn: QuantTensor,
    pub(crate) w_down_ffn: QuantTensor,
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
        let stream = ctx.default_stream();
        let session = LtSession::new(stream.clone())
            .map_err(|e| LlmError::Backend(format!("LtSession: {e:?}")))?;
        let kernels = LlmKernels::new(ctx.clone());

        let cfg = crate::qwen35::parse_config(path)
            .map_err(|e| LlmError::Backend(format!("parse_config: {e:?}")))?;
        let file = GgufFile::open(path).map_err(|e| LlmError::Backend(format!("gguf: {e:?}")))?;

        if cfg.variant == Qwen35Variant::Moe {
            return Err(LlmError::Backend(
                "Qwen35ModelCudaQ4K: MoE variant (35B-A3B) not yet supported (T246.6)".into(),
            ));
        }

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
                // F32 fallback : small tensors (typically ssm_alpha, ssm_beta with
                // n=48 — only ~1 MB each). Dequant to BF16, dispatch via sgemv_bf16.
                GgmlType::F32 => {
                    let f32_buf = rustorch_gguf::dequant::dequant_to_f32(info, bytes)
                        .map_err(|e| LlmError::Backend(format!("dequant F32 {name}: {e:?}")))?;
                    let bf: Vec<half::bf16> =
                        f32_buf.iter().copied().map(half::bf16::from_f32).collect();
                    let dev = stream
                        .memcpy_stod(&bf)
                        .map_err(|e| LlmError::Backend(format!("upload {name}: {e:?}")))?;
                    Ok(QuantTensor::Bf16 { weights: dev, n, k })
                },
                other => Err(LlmError::Backend(format!(
                    "unsupported dtype {other:?} for {name} — only Q4_K/Q5_K/Q6_K/F32 supported"
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
                        w_gate_ffn: load_quant(&key("ffn_gate.weight"))?,
                        w_up_ffn: load_quant(&key("ffn_up.weight"))?,
                        w_down_ffn: load_quant(&key("ffn_down.weight"))?,
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
                        w_gate_ffn: load_quant(&key("ffn_gate.weight"))?,
                        w_up_ffn: load_quant(&key("ffn_up.weight"))?,
                        w_down_ffn: load_quant(&key("ffn_down.weight"))?,
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

        // T246.5.3 — sync host counters → device counters. 3 × 4-byte H2D.
        // These mirror `self.position` so the *_devcnt kernels read the right
        // values inside a captured CUDA Graph (Phase C).
        let pos_i32 = self.position as i32;
        self.stream
            .memcpy_htod(&[token_id], &mut self.current_token_dev)
            .map_err(|e| LlmError::Backend(format!("upload token id: {e:?}")))?;
        self.stream
            .memcpy_htod(&[pos_i32], &mut self.position_dev)
            .map_err(|e| LlmError::Backend(format!("upload position: {e:?}")))?;
        self.stream
            .memcpy_htod(&[pos_i32 + 1], &mut self.kv_len_dev)
            .map_err(|e| LlmError::Backend(format!("upload kv_len: {e:?}")))?;

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
            (
                a, b, c, d_, e, f_, g, h_, i, j, k_, l, m, n_, o, p, q_, r, s, t, u,
            )
        };

        // Force sync after zeroing scratch.
        self.stream.synchronize().ok();

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
        self.stream.synchronize().ok();

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
                    )?;

                    // 3. z = w_gate @ h_norm
                    ssm.w_gate
                        .dispatch_matmul_m1(&self.kernels, &self.stream, h_norm_p, z_p)?;

                    // 4. alpha = w_alpha @ h_norm
                    ssm.w_alpha.dispatch_matmul_m1(
                        &self.kernels,
                        &self.stream,
                        h_norm_p,
                        alpha_p,
                    )?;

                    // 5. beta_logit = w_beta @ h_norm ; beta = sigmoid
                    ssm.w_beta
                        .dispatch_matmul_m1(&self.kernels, &self.stream, h_norm_p, beta_p)?;
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
                    ssm.ssm_out
                        .dispatch_matmul_m1(&self.kernels, &self.stream, sso_p, h_p)?;

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
                    attn.w_q
                        .dispatch_matmul_m1(&self.kernels, &self.stream, h_norm_p, qg_p)?;

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
                    attn.w_k
                        .dispatch_matmul_m1(&self.kernels, &self.stream, h_norm_p, k_p)?;
                    attn.w_v
                        .dispatch_matmul_m1(&self.kernels, &self.stream, h_norm_p, v_p)?;

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

                    // 8. GQA decode online softmax.
                    // T246.5.3 — devcnt variant reads `kv_len` from kv_len_dev.
                    let _ = position; // kept for the host-side KV append above
                    unsafe {
                        let (kc_p, _g1) = kv_cache.k.device_ptr(&self.stream);
                        let (vc_p, _g2) = kv_cache.v.device_ptr(&self.stream);
                        let (kv_len_p, _g3) = self.kv_len_dev.device_ptr(&self.stream);
                        self.kernels
                            .gqa_decode_online_bf16_devcnt(
                                &self.stream,
                                q_p,
                                kc_p,
                                vc_p,
                                gate_p, // attn raw output (size f ≥ q_dim)
                                n_q as i32,
                                n_kv as i32,
                                kv_len_p,
                                head_dim as i32,
                                self.max_seq as i32,
                            )
                            .map_err(|e| LlmError::Backend(format!("gqa_decode: {e:?}")))?;
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
                    attn.w_o
                        .dispatch_matmul_m1(&self.kernels, &self.stream, gate_p, h_p)?;

                    // 11. residual : h += residual
                    unsafe {
                        self.kernels
                            .add_inplace_bf16(&self.stream, h_p, res_p, d as i32)
                            .map_err(|e| LlmError::Backend(format!("attn residual: {e:?}")))?;
                    }
                },
            }

            // ---- FFN dense (T246.4) ----
            let (gate_w, up_w, down_w, post_norm) = match block {
                BlockQ4K::Ssm(s) => (&s.w_gate_ffn, &s.w_up_ffn, &s.w_down_ffn, &s.post_norm),
                BlockQ4K::Attn(a) => (&a.w_gate_ffn, &a.w_up_ffn, &a.w_down_ffn, &a.post_norm),
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
            // gate = w_gate @ h_norm ; up = w_up @ h_norm ; gate = silu(gate) * up
            gate_w.dispatch_matmul_m1(&self.kernels, &self.stream, h_norm_p, gate_p)?;
            up_w.dispatch_matmul_m1(&self.kernels, &self.stream, h_norm_p, up_p)?;
            unsafe {
                self.kernels
                    .swiglu_bf16(&self.stream, gate_p, up_p, gate_p, f as i32)
                    .map_err(|e| LlmError::Backend(format!("swiglu: {e:?}")))?;
            }
            // h = w_down @ gate
            down_w.dispatch_matmul_m1(&self.kernels, &self.stream, gate_p, h_p)?;
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
            .dispatch_matmul_m1(&self.kernels, &self.stream, h_p, logits_p)?;

        // ---- Sample (argmax for now) ----
        unsafe {
            self.kernels
                .argmax_bf16(&self.stream, logits_p, tok_p, cfg.vocab as i32)
                .map_err(|e| LlmError::Backend(format!("argmax: {e:?}")))?;
        }
        self.stream.synchronize().ok();

        let next_id_host: Vec<u32> = self
            .stream
            .memcpy_dtov(&self.scratch.next_token)
            .map_err(|e| LlmError::Backend(format!("dl token: {e:?}")))?;
        let _ = (tok_p, logits_p, final_norm_p); // silence unused warnings

        // T246.4.2 — DEBUG : print first few logits to diagnose non-determinism.
        if self.position == 0 && std::env::var("RUSTORCH_DEBUG_LOGITS").is_ok() {
            let logits_host: Vec<half::bf16> = self
                .stream
                .memcpy_dtov(&self.scratch.logits)
                .map_err(|e| LlmError::Backend(format!("dl logits: {e:?}")))?;
            eprintln!(
                "[debug] first 5 logits : {:?}",
                logits_host
                    .iter()
                    .take(5)
                    .map(|v| v.to_f32())
                    .collect::<Vec<_>>()
            );
            eprintln!(
                "[debug] argmax token   : {} (logit {:.4})",
                next_id_host[0],
                logits_host[next_id_host[0] as usize].to_f32()
            );
            // Sample some h state in scratch.
            let h_host: Vec<half::bf16> = self
                .stream
                .memcpy_dtov(&self.scratch.h)
                .map_err(|e| LlmError::Backend(format!("dl h: {e:?}")))?;
            eprintln!(
                "[debug] first 5 h     : {:?}",
                h_host
                    .iter()
                    .take(5)
                    .map(|v| v.to_f32())
                    .collect::<Vec<_>>()
            );
        }

        self.position += 1;
        Ok(next_id_host[0])
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
}
