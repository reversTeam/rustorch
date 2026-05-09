//! `qwen35_cuda` — CUDA backend pour Qwen3.5 / Qwen3.6 hybrid (T243).
//!
//! **STATUS: SKELETON (Session 1/4-5)**.
//!
//! Architecture cible :
//! - All matmul weights (W_q, W_k, W_v, W_o, W_qkv_ssm, W_gate_ssm,
//!   W_alpha, W_beta, W_out_ssm, W_gate_ffn, W_up, W_down) stored
//!   device-resident in BF16 (no CPU duplicate).
//! - All scratch buffers + activations device-resident.
//! - `forward_token` runs ENTIRELY on GPU :
//!   - Embedding lookup, RMSNorm, RoPE (existing kernels in llm_kernels)
//!   - matmul_bf16 via cuBLASLt for Q/K/V/O, gate/up/down, SSM projs
//!   - Attention layers : reuse existing decode_step path from cuda_backend
//!   - SSM layers : NEW conv1d + scan + gated_norm kernels (T243.2)
//!   - MoE for 35B-A3B : router + 256-expert dispatch (T243.3)
//! - Final RMSNorm + LM head + argmax → token id.
//!
//! Performance target (on DGX Spark GB10) :
//! - Qwen3.6-27B BF16  : ~10-15 tok/s  (memory-bound by ~54GB weights)
//! - Qwen3.6-27B NVFP4 : ~30-50 tok/s  (~13.5GB weights)
//! - Qwen3.6-35B-A3B BF16 (3B active) : ~50-70 tok/s
//! - Qwen3.6-35B-A3B NVFP4 (3B active) : ~100-150 tok/s
//!
//! ## Sessions plan
//!
//! - **T243.1 (this commit)** : skeleton + struct + Stub `from_cpu` /
//!   `forward_token` returning 0. Compiles & runs but produces no output.
//! - **T243.2** : SSM kernels CUDA (conv1d + scan + gated norm).
//! - **T243.3** : Wire transformer + SSM + FFN dense in `forward_token`.
//! - **T243.4** : MoE for 35B-A3B (router + topk + expert dispatch).
//! - **T243.5** : NVFP4 storage + perf tuning to hit 70 tok/s on real GGUF.

#[cfg(feature = "cuda")]
use crate::qwen35_cpu::{FfnWeights, LayerWeights, Qwen35Weights};
#[cfg(feature = "cuda")]
use crate::LlmError;

#[cfg(feature = "cuda")]
use cudarc::driver::{CudaContext, CudaSlice, CudaStream};
#[cfg(feature = "cuda")]
use std::sync::Arc;

#[cfg(feature = "cuda")]
fn upload_bf16(stream: &Arc<CudaStream>, data: &[f32]) -> Result<CudaSlice<half::bf16>, LlmError> {
    let bf: Vec<half::bf16> = data.iter().copied().map(half::bf16::from_f32).collect();
    stream
        .memcpy_stod(&bf)
        .map_err(|e| LlmError::Backend(format!("upload_bf16: {e:?}")))
}

/// CUDA-resident state for a Qwen3.5/3.6 model.
///
/// **Skeleton**. Will be filled across T243.1-5.
#[cfg(feature = "cuda")]
pub struct Qwen35ModelCuda {
    pub cfg: crate::qwen35::Qwen35Config,
    /// CUDA context owning device memory.
    #[allow(dead_code)]
    ctx: Arc<CudaContext>,
    /// Stream for kernel launches.
    #[allow(dead_code)]
    stream: Arc<CudaStream>,
    /// Token embedding table : `[V, D]` BF16.
    #[allow(dead_code)]
    token_emb: CudaSlice<half::bf16>,
    /// Final RMSNorm gamma : `[D]` BF16.
    #[allow(dead_code)]
    final_norm: CudaSlice<half::bf16>,
    /// LM head : `[D, V]` BF16.
    #[allow(dead_code)]
    lm_head: CudaSlice<half::bf16>,
    /// Per-block weights (heterogeneous : attention OR ssm).
    #[allow(dead_code)]
    blocks: Vec<BlockCuda>,
    /// Maximum sequence length for KV cache.
    #[allow(dead_code)]
    max_seq: usize,
}

#[cfg(feature = "cuda")]
#[allow(dead_code)]
enum BlockCuda {
    Attn(AttnBlockCuda),
    Ssm(SsmBlockCuda),
}

#[cfg(feature = "cuda")]
#[allow(dead_code)]
struct AttnBlockCuda {
    /// Pre-attn RMSNorm γ : `[D]` BF16.
    attn_norm: CudaSlice<half::bf16>,
    /// Post-attn RMSNorm γ : `[D]` BF16.
    attn_post_norm: CudaSlice<half::bf16>,
    /// Q + per-head gate fused : `[2*head_dim*n_q_heads, D]` BF16.
    w_q: CudaSlice<half::bf16>,
    /// K projection : `[head_dim*n_kv_heads, D]` BF16.
    w_k: CudaSlice<half::bf16>,
    /// V projection : `[head_dim*n_kv_heads, D]` BF16.
    w_v: CudaSlice<half::bf16>,
    /// O projection : `[D, head_dim*n_q_heads]` BF16.
    w_o: CudaSlice<half::bf16>,
    /// Per-head Q/K norm γ : `[head_dim]` BF16.
    q_norm: CudaSlice<half::bf16>,
    k_norm: CudaSlice<half::bf16>,
    /// Dense FFN: gate, up, down (Qwen3.5/3.6 27B uses dense per-block).
    w_gate_ffn: CudaSlice<half::bf16>,
    w_up_ffn: CudaSlice<half::bf16>,
    w_down_ffn: CudaSlice<half::bf16>,
}

#[cfg(feature = "cuda")]
#[allow(dead_code)]
struct SsmBlockCuda {
    /// Pre-SSM RMSNorm γ : `[D]` BF16.
    attn_norm: CudaSlice<half::bf16>,
    /// Post-SSM RMSNorm γ : `[D]` BF16.
    attn_post_norm: CudaSlice<half::bf16>,
    /// QKV combined projection : `[conv_dim, D]` BF16.
    w_qkv: CudaSlice<half::bf16>,
    /// z gate input : `[value_dim, D]` BF16.
    w_gate: CudaSlice<half::bf16>,
    /// 1-D depth-wise conv kernel : `[conv_kernel, conv_dim]` BF16.
    conv1d: CudaSlice<half::bf16>,
    /// α projection : `[num_v_heads, D]` BF16.
    w_alpha: CudaSlice<half::bf16>,
    /// β projection : `[num_v_heads, D]` BF16.
    w_beta: CudaSlice<half::bf16>,
    /// dt bias : `[num_v_heads]` BF16.
    dt_bias: CudaSlice<half::bf16>,
    /// A_log scalars : `[num_v_heads]` BF16.
    ssm_a: CudaSlice<half::bf16>,
    /// Gated RMSNorm γ : `[head_v_dim]` BF16.
    ssm_norm: CudaSlice<half::bf16>,
    /// Output projection : `[D, value_dim]` BF16.
    ssm_out: CudaSlice<half::bf16>,
    /// Same dense FFN as the attention block.
    w_gate_ffn: CudaSlice<half::bf16>,
    w_up_ffn: CudaSlice<half::bf16>,
    w_down_ffn: CudaSlice<half::bf16>,
}

#[cfg(feature = "cuda")]
impl Qwen35ModelCuda {
    /// Upload all weights from CPU `Qwen35Weights` to GPU as BF16.
    /// Implementation T243.1 — uploads global tensors + per-layer tensors
    /// to device. forward_token still returns Err (T243.2/3).
    ///
    /// MoE FFN (35B-A3B) currently rejected — needs T243.4 (per-expert
    /// upload + router). Dense FFN (27B) supported.
    pub fn from_cpu(cpu: Qwen35Weights, max_seq: usize) -> Result<Self, LlmError> {
        let cfg = cpu.cfg.clone();
        let ctx =
            CudaContext::new(0).map_err(|e| LlmError::Backend(format!("CudaContext: {e:?}")))?;
        let stream = ctx.default_stream();

        let token_emb = upload_bf16(&stream, &cpu.tok_embd)?;
        let final_norm = upload_bf16(&stream, &cpu.output_norm)?;
        let lm_head = upload_bf16(&stream, &cpu.output)?;

        let mut blocks: Vec<BlockCuda> = Vec::with_capacity(cfg.n_layers);
        for layer in cpu.layers.into_iter() {
            // Extract dense FFN — MoE rejected for now.
            let block = match layer {
                LayerWeights::Attn { attn, ffn } => {
                    let (w_gate_ffn, w_up_ffn, w_down_ffn) = match ffn {
                        FfnWeights::Dense(d) => (
                            upload_bf16(&stream, &d.w_gate.data)?,
                            upload_bf16(&stream, &d.w_up.data)?,
                            upload_bf16(&stream, &d.w_down.data)?,
                        ),
                        FfnWeights::Moe(_) => {
                            return Err(LlmError::Backend(
                                "Qwen35ModelCuda: MoE FFN not yet supported (T243.4)".into(),
                            ));
                        },
                    };
                    BlockCuda::Attn(AttnBlockCuda {
                        attn_norm: upload_bf16(&stream, &attn.attn_norm)?,
                        attn_post_norm: upload_bf16(&stream, &attn.attn_post_norm)?,
                        w_q: upload_bf16(&stream, &attn.w_q.data)?,
                        w_k: upload_bf16(&stream, &attn.w_k.data)?,
                        w_v: upload_bf16(&stream, &attn.w_v.data)?,
                        w_o: upload_bf16(&stream, &attn.w_o.data)?,
                        q_norm: upload_bf16(&stream, &attn.q_norm)?,
                        k_norm: upload_bf16(&stream, &attn.k_norm)?,
                        w_gate_ffn,
                        w_up_ffn,
                        w_down_ffn,
                    })
                },
                LayerWeights::Ssm { ssm, ffn } => {
                    let (w_gate_ffn, w_up_ffn, w_down_ffn) = match ffn {
                        FfnWeights::Dense(d) => (
                            upload_bf16(&stream, &d.w_gate.data)?,
                            upload_bf16(&stream, &d.w_up.data)?,
                            upload_bf16(&stream, &d.w_down.data)?,
                        ),
                        FfnWeights::Moe(_) => {
                            return Err(LlmError::Backend(
                                "Qwen35ModelCuda: MoE FFN not yet supported (T243.4)".into(),
                            ));
                        },
                    };
                    BlockCuda::Ssm(SsmBlockCuda {
                        attn_norm: upload_bf16(&stream, &ssm.attn_norm)?,
                        attn_post_norm: upload_bf16(&stream, &ssm.attn_post_norm)?,
                        w_qkv: upload_bf16(&stream, &ssm.w_qkv.data)?,
                        w_gate: upload_bf16(&stream, &ssm.w_gate.data)?,
                        conv1d: upload_bf16(&stream, &ssm.conv1d)?,
                        w_alpha: upload_bf16(&stream, &ssm.ssm_alpha.data)?,
                        w_beta: upload_bf16(&stream, &ssm.ssm_beta.data)?,
                        dt_bias: upload_bf16(&stream, &ssm.dt_bias)?,
                        ssm_a: upload_bf16(&stream, &ssm.ssm_a)?,
                        ssm_norm: upload_bf16(&stream, &ssm.ssm_norm)?,
                        ssm_out: upload_bf16(&stream, &ssm.ssm_out.data)?,
                        w_gate_ffn,
                        w_up_ffn,
                        w_down_ffn,
                    })
                },
            };
            blocks.push(block);
        }

        Ok(Self {
            cfg,
            ctx,
            stream,
            token_emb,
            final_norm,
            lm_head,
            blocks,
            max_seq,
        })
    }

    /// Decode 1 token autoregressive — returns next-token id.
    /// **STATUS T243.1** : stub returning 0.
    pub fn decode_step(&mut self, _token_id: u32) -> Result<u32, LlmError> {
        Err(LlmError::Backend(
            "Qwen35ModelCuda::decode_step not yet implemented".into(),
        ))
    }

    pub fn reset_state(&mut self) {
        // Stub.
    }

    pub fn last_logits(&self) -> Result<Vec<f32>, LlmError> {
        Err(LlmError::Backend(
            "Qwen35ModelCuda::last_logits not yet implemented".into(),
        ))
    }
}

// Stub when cuda feature is OFF.
#[cfg(not(feature = "cuda"))]
pub struct Qwen35ModelCuda;
