//! CUDA backend pour LLM inference (T241.3).
//!
//! Mirror de [`LlamaModel`] mais avec poids résident-device (BF16 ou
//! NVFP4 packed). Construit sur :
//! - `rustorch_cuda::cublas_lt::LtSession` pour les matmuls
//! - `rustorch_cuda::llm_kernels::LlmKernels` pour RMSNorm, RoPE,
//!   SwiGLU, embedding, sampling
//!
//! L'API publique reste compatible avec [`LlamaModel`] :
//!
//! ```ignore
//! let model = LlamaModelCuda::from_gguf("Qwen3.6-27B.gguf", 8192)?;
//! let tokens = model.generate(prompt_ids, &SamplingConfig::greedy(), 128)?;
//! ```
//!
//! ## Statut MVP (T241.3)
//!
//! - GGUF loader → dequant CPU → upload BF16 device (gros peak RAM
//!   pendant le load, optimisé en T241.5 avec dequant kernels CUDA)
//! - Forward via LtSession matmul_bf16 + custom kernels
//! - Decode-loop autoregressive
//! - Greedy sampling (argmax_bf16 device-side)
//!
//! ## Limitations actuelles
//! - F32 norm weights (gamma) — small enough that BF16 conversion is moot
//! - Pas encore de causal mask attention (decode_step seulement; prefill
//!   = boucle de decode_step pour MVP)
//! - Sampling stochastic (top-k, top-p) en T241.6

#![cfg(feature = "cuda")]

use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaSlice, CudaStream};
use rustorch_cuda::cublas_lt::LtSession;
use rustorch_cuda::llm_kernels::LlmKernels;

use crate::{LlamaConfig, LlamaModel, LlmError};

/// Per-layer weights resident sur device en BF16.
struct BlockWeightsCuda {
    /// `[D]` BF16 — RMSNorm pre-attention gamma.
    rms_attn: CudaSlice<half::bf16>,
    /// `[D, D + 2·KV_DIM]` BF16 fused QKV projection.
    w_qkv: CudaSlice<half::bf16>,
    /// `[D, D]` attention output projection.
    w_o: CudaSlice<half::bf16>,
    /// `[D]` BF16 — RMSNorm pre-FFN gamma.
    rms_ffn: CudaSlice<half::bf16>,
    /// `[D, 2·F]` fused gate || up.
    w_gate_up: CudaSlice<half::bf16>,
    /// `[F, D]` FFN down projection.
    w_down: CudaSlice<half::bf16>,
    /// Optional Qwen3 per-head Q/K norms — `[head_dim]` BF16.
    q_norm: Option<CudaSlice<half::bf16>>,
    k_norm: Option<CudaSlice<half::bf16>>,
}

/// Buffers de scratch device-resident, alloués une fois et réutilisés
/// d'une layer à l'autre (et d'un step à l'autre).
struct ScratchCuda {
    /// `[D]` BF16 — token activation (after embedding lookup, between layers).
    x: CudaSlice<half::bf16>,
    /// `[D]` BF16 — pre-norm scratch.
    h: CudaSlice<half::bf16>,
    /// `[D + 2·KV_DIM]` BF16 — fused QKV output.
    qkv: CudaSlice<half::bf16>,
    /// `[2·F]` BF16 — fused gate||up output.
    gate_up: CudaSlice<half::bf16>,
    /// `[F]` BF16 — FFN intermediate (silu(gate) * up).
    ffn_inter: CudaSlice<half::bf16>,
    /// `[D]` BF16 — final output of one block (added to residual).
    block_out: CudaSlice<half::bf16>,
    /// `[V]` BF16 — final logits.
    logits: CudaSlice<half::bf16>,
    /// `[1]` u32 — sampling output token id.
    sample_out: CudaSlice<u32>,
}

/// LLM resident sur GPU CUDA, prêt pour autoregressive decode.
///
/// Ports exactly the [`LlamaModel`] structure but stores all the heavy
/// weights as `CudaSlice<bf16>` so the matmul / kernel hot path never
/// has to copy them across the PCIe bus.
pub struct LlamaModelCuda {
    /// Configuration parsée du modèle (Llama / Qwen2.5 / Qwen3).
    pub config: LlamaConfig,
    /// CUDA stream + context lié à toutes les opérations du modèle.
    stream: Arc<CudaStream>,
    ctx: Arc<CudaContext>,
    /// cuBLASLt session pour les matmuls BF16 (T240.8 path).
    session: LtSession,
    /// Custom kernels (RMSNorm, RoPE, SwiGLU, ...) compilés via nvrtc.
    kernels: LlmKernels,
    /// Per-layer weights device-resident.
    blocks: Vec<BlockWeightsCuda>,
    /// `[V, D]` BF16 token embedding table.
    token_emb: CudaSlice<half::bf16>,
    /// `[D]` BF16 final RMSNorm gamma.
    final_norm: CudaSlice<half::bf16>,
    /// `[D, V]` BF16 LM head (transposed of embedding if tie_word_embeddings).
    lm_head: CudaSlice<half::bf16>,
    /// Pré-calculé `inv_freq` pour RoPE = `[head_dim/2]` F32.
    rope_inv_freq: CudaSlice<f32>,
    /// Scratch buffers, alloués au .new() pour éviter les realloc/step.
    scratch: ScratchCuda,
    /// Per-layer KV cache resident-device : K = `[max_seq, kv_dim]` BF16,
    /// V = pareil. Une paire par layer.
    kv_cache_k: Vec<CudaSlice<half::bf16>>,
    kv_cache_v: Vec<CudaSlice<half::bf16>>,
    /// Position courante dans le KV cache (token next à écrire).
    kv_pos: usize,
    /// Capacity de la KV cache (`max_seq`).
    max_seq: usize,
}

impl LlamaModelCuda {
    /// Construit la version CUDA à partir d'un [`LlamaModel`] déjà chargé
    /// (en F32). Convertit chaque tensor en BF16 et upload.
    ///
    /// Pour MVP, le caller charge le GGUF via la voie existante :
    /// ```ignore
    /// let cfg = LlamaConfig::from_hf_dir("Qwen3.6-27B/")?;
    /// let weights = GgufWeights::from_path("model.gguf")?;
    /// let cpu = LlamaModel::from_gguf(cfg, weights, max_seq)?;
    /// let cuda = LlamaModelCuda::from_cpu(cpu, max_seq)?;
    /// ```
    /// T241.5 ajoutera un `from_gguf(path, max_seq)` direct avec dequant
    /// kernels CUDA (sans le détour CPU).
    pub fn from_cpu(cpu: LlamaModel, max_seq: usize) -> Result<Self, LlmError> {
        let ctx = CudaContext::new(0).map_err(|e| LlmError::Backend(format!("ctx: {e:?}")))?;
        let stream = ctx.default_stream();
        let session = LtSession::new(stream.clone())
            .map_err(|e| LlmError::Backend(format!("LtSession: {e:?}")))?;
        let kernels = LlmKernels::new(ctx.clone());

        let cfg = &cpu.config;
        let d = cfg.hidden_size;
        let kv_dim = cfg.n_kv_heads() * cfg.head_dim();
        let f = cfg.intermediate_size;
        let head_dim = cfg.head_dim();

        // Helper : F32 vec → BF16 vec → device
        let upload_bf16 =
            |stream: &Arc<CudaStream>, data: &[f32]| -> Result<CudaSlice<half::bf16>, LlmError> {
                let bf: Vec<half::bf16> = data.iter().copied().map(half::bf16::from_f32).collect();
                stream
                    .memcpy_stod(&bf)
                    .map_err(|e| LlmError::Backend(format!("upload_bf16: {e:?}")))
            };

        let token_emb_dev = upload_bf16(&stream, &cpu.token_emb)?;
        let final_norm_dev = upload_bf16(&stream, &cpu.final_norm)?;
        let lm_head_dev = upload_bf16(&stream, &cpu.lm_head)?;

        let mut blocks: Vec<BlockWeightsCuda> = Vec::with_capacity(cfg.num_hidden_layers);
        for blk in cpu.blocks.iter() {
            blocks.push(BlockWeightsCuda {
                rms_attn: upload_bf16(&stream, &blk.rms_attn)?,
                w_qkv: upload_bf16(&stream, &blk.w_qkv)?,
                w_o: upload_bf16(&stream, &blk.w_o)?,
                rms_ffn: upload_bf16(&stream, &blk.rms_ffn)?,
                w_gate_up: upload_bf16(&stream, &blk.w_gate_up)?,
                w_down: upload_bf16(&stream, &blk.w_down)?,
                q_norm: blk
                    .q_norm
                    .as_ref()
                    .map(|v| upload_bf16(&stream, v))
                    .transpose()?,
                k_norm: blk
                    .k_norm
                    .as_ref()
                    .map(|v| upload_bf16(&stream, v))
                    .transpose()?,
            });
        }

        // Pré-calcul inv_freq pour RoPE = 1 / (theta_base^(2i/head_dim))
        let inv_freq_host: Vec<f32> = (0..head_dim / 2)
            .map(|i| (cfg.rope_theta as f32).powf(-(2.0 * i as f32) / head_dim as f32))
            .collect();
        let rope_inv_freq = stream
            .memcpy_stod(&inv_freq_host)
            .map_err(|e| LlmError::Backend(format!("rope_inv_freq: {e:?}")))?;

        // Scratch buffers
        let zeros_bf16 = |n: usize| -> Result<CudaSlice<half::bf16>, LlmError> {
            stream
                .alloc_zeros::<half::bf16>(n)
                .map_err(|e| LlmError::Backend(format!("alloc {n}: {e:?}")))
        };
        let scratch = ScratchCuda {
            x: zeros_bf16(d)?,
            h: zeros_bf16(d)?,
            qkv: zeros_bf16(d + 2 * kv_dim)?,
            gate_up: zeros_bf16(2 * f)?,
            ffn_inter: zeros_bf16(f)?,
            block_out: zeros_bf16(d)?,
            logits: zeros_bf16(cfg.vocab_size)?,
            sample_out: stream
                .alloc_zeros::<u32>(1)
                .map_err(|e| LlmError::Backend(format!("alloc sample: {e:?}")))?,
        };

        // KV cache : 2 tensors par layer, [max_seq, kv_dim] BF16
        let mut kv_cache_k = Vec::with_capacity(cfg.num_hidden_layers);
        let mut kv_cache_v = Vec::with_capacity(cfg.num_hidden_layers);
        for _ in 0..cfg.num_hidden_layers {
            kv_cache_k.push(zeros_bf16(max_seq * kv_dim)?);
            kv_cache_v.push(zeros_bf16(max_seq * kv_dim)?);
        }

        Ok(Self {
            config: cpu.config,
            stream,
            ctx,
            session,
            kernels,
            blocks,
            token_emb: token_emb_dev,
            final_norm: final_norm_dev,
            lm_head: lm_head_dev,
            rope_inv_freq,
            scratch,
            kv_cache_k,
            kv_cache_v,
            kv_pos: 0,
            max_seq,
        })
    }

    /// Reset la position du KV cache (pour redémarrer une génération).
    pub fn reset_kv(&mut self) {
        self.kv_pos = 0;
    }

    /// Dimension hidden du modèle.
    pub fn hidden(&self) -> usize {
        self.config.hidden_size
    }

    /// Vocab size.
    pub fn vocab_size(&self) -> usize {
        self.config.vocab_size
    }
}

// Suppress dead_code warnings on fields used only at runtime by future kernels.
#[allow(dead_code)]
const _: () = ();
