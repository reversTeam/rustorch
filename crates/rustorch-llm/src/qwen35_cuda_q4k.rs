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

/// One quantized matmul weight on GPU. Qwen3.6 Q4_K_M uses a mix of Q4_K /
/// Q5_K / Q6_K (per llama.cpp's quantization rules).
pub(crate) enum QuantTensor {
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

/// Per-SSM-layer recurrent state.
pub(crate) struct SsmState {
    /// `[num_v_heads, head_v_dim, head_v_dim]` BF16 — outer-product state.
    pub(crate) state: CudaSlice<half::bf16>,
    /// `[(conv_kernel - 1), conv_dim]` BF16 — ring buffer for depth-wise conv.
    pub(crate) conv_state: CudaSlice<half::bf16>,
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
            let dev = stream
                .memcpy_stod(bytes)
                .map_err(|e| LlmError::Backend(format!("upload {name}: {e:?}")))?;
            match info.dtype {
                GgmlType::Q4_K => Ok(QuantTensor::Q4K { bytes: dev, n, k }),
                GgmlType::Q5_K => Ok(QuantTensor::Q5K { bytes: dev, n, k }),
                GgmlType::Q6_K => Ok(QuantTensor::Q6K { bytes: dev, n, k }),
                other => Err(LlmError::Backend(format!(
                    "unsupported dtype {other:?} for {name} — only Q4_K/Q5_K/Q6_K supported in matmul path"
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

        Ok(Self {
            config: cfg,
            ctx,
            stream,
            session,
            kernels,
            token_emb,
            final_norm,
            lm_head,
            blocks,
            kv_caches,
            ssm_states,
            max_seq,
            position: 0,
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
        Ok(())
    }

    /// Decode one token. **STATUS T246.1** : returns Err. Implementation
    /// in T246.2 (SSM block), T246.3 (attn block), T246.4 (FFN + sample).
    pub fn decode_step(&mut self, _token_id: u32) -> Result<u32, LlmError> {
        Err(LlmError::Backend(
            "Qwen35ModelCudaQ4K::decode_step not yet implemented (T246.2-4)".into(),
        ))
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
