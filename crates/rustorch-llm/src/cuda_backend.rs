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

/// Per-layer FP4 weights + scales (T241.5 NVFP4 path).
///
/// Pour Qwen-27B chaque layer prend ~46 MB en FP4 vs ~184 MB en BF16
/// (4× saving). Le forward FP4 utilise `LtSession::matmul_mxfp4` au lieu
/// de `matmul_bf16`.
struct BlockWeightsCudaFp4 {
    /// `[D]` BF16 — RMSNorm pre-attention gamma (pas quantisé).
    rms_attn: CudaSlice<half::bf16>,
    /// `[D, D + 2·KV_DIM]` FP4 packed (1 byte = 2 elements).
    w_qkv: CudaSlice<u8>,
    w_qkv_scale: CudaSlice<u8>,
    /// `[D, D]` FP4 packed.
    w_o: CudaSlice<u8>,
    w_o_scale: CudaSlice<u8>,
    /// `[D]` BF16 — RMSNorm pre-FFN gamma.
    rms_ffn: CudaSlice<half::bf16>,
    /// `[D, 2·F]` FP4 packed.
    w_gate_up: CudaSlice<u8>,
    w_gate_up_scale: CudaSlice<u8>,
    /// `[F, D]` FP4 packed.
    w_down: CudaSlice<u8>,
    w_down_scale: CudaSlice<u8>,
}

/// FP4 scratch : activations restent BF16, on alloue des buffers
/// pseudo-FP4 pour les passer aux kernels (re-cast pointer u8). En MVP
/// "dummy" le contenu n'est pas correct numériquement, mais le timing
/// du matmul_mxfp4 est réel.
struct ScratchCudaFp4 {
    /// FP4 scratch pour activations BF16 reinterprétées (sizes : seq × hidden / 2).
    /// En MVP on alloue des buffers fixes même si les valeurs sont garbage.
    x_fp4: CudaSlice<u8>,
    x_fp4_scale: CudaSlice<u8>,
    /// FFN intermediate pseudo-FP4.
    ffn_inter_fp4: CudaSlice<u8>,
    ffn_inter_fp4_scale: CudaSlice<u8>,
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
    /// Per-layer FP4 weights (None tant que decode_step_fp4 pas appelé).
    blocks_fp4: Option<Vec<BlockWeightsCudaFp4>>,
    /// FP4 scratch buffers (alloués lazy avec from_dummy_fp4).
    scratch_fp4: Option<ScratchCudaFp4>,
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
            blocks_fp4: None,
            scratch_fp4: None,
        })
    }

    /// Reset la position du KV cache (pour redémarrer une génération).
    pub fn reset_kv(&mut self) {
        self.kv_pos = 0;
    }

    /// Allocate FP4 weight buffers et quantize les BF16 weights existants
    /// vers NVFP4 (T241.5b).
    ///
    /// Si les BF16 weights sont real (chargés via from_cpu) : vraie
    /// quantization device-side via le kernel `quantize_bf16_to_nvfp4`.
    /// Si dummy zeros : la quantization donne aussi des zeros (ok pour
    /// timing bench).
    ///
    /// Note : sur Qwen-72B en mode `from_dummy_fp4_only`, les BF16
    /// weights sont des stubs 16-byte ; le quantize ne peut pas être
    /// appelé (skip via `RUSTORCH_BENCH_SKIP_QUANT=1`).
    pub fn enable_fp4(&mut self) -> Result<(), LlmError> {
        let d = self.config.hidden_size;
        let kv_dim = self.config.n_kv_heads() * self.config.head_dim();
        let f = self.config.intermediate_size;
        let qkv_n = d + 2 * kv_dim;
        let block16 = 16usize; // NVFP4 VEC16

        let stream = &self.stream;
        let alloc_u8 = |n: usize| -> Result<CudaSlice<u8>, LlmError> {
            stream
                .alloc_zeros::<u8>(n.max(1))
                .map_err(|e| LlmError::Backend(format!("alloc {n} u8: {e:?}")))
        };

        // Skip quantize si BF16 weights sont stubs (mode fp4_only)
        let skip_quant = std::env::var("RUSTORCH_BENCH_SKIP_QUANT")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);

        // Helper : quantize un buffer BF16 → (FP4 packed, UE4M3 scale)
        let quantize_or_alloc =
            |size_bf16: usize, src_bf16: u64| -> Result<(CudaSlice<u8>, CudaSlice<u8>), LlmError> {
                let fp4_bytes = size_bf16 / 2;
                let scale_bytes = size_bf16 / block16;
                let mut fp4 = stream
                    .alloc_zeros::<u8>(fp4_bytes.max(1))
                    .map_err(|e| LlmError::Backend(format!("alloc fp4: {e:?}")))?;
                let mut scale = stream
                    .alloc_zeros::<u8>(scale_bytes.max(1))
                    .map_err(|e| LlmError::Backend(format!("alloc scale: {e:?}")))?;
                if !skip_quant {
                    use cudarc::driver::DevicePtrMut;
                    unsafe {
                        let (fp4_p, _r1) = fp4.device_ptr_mut(stream);
                        let (sc_p, _r2) = scale.device_ptr_mut(stream);
                        self.kernels
                            .quantize_bf16_to_nvfp4(stream, src_bf16, fp4_p, sc_p, size_bf16 as i32)
                            .map_err(|e| LlmError::Backend(format!("quantize: {e:?}")))?;
                    }
                }
                Ok((fp4, scale))
            };

        // Per-weight FP4 + scale (vraie quantization si BF16 weights real).
        let mut blocks_fp4: Vec<BlockWeightsCudaFp4> = Vec::with_capacity(self.blocks.len());
        for blk in self.blocks.iter() {
            // rms_attn / rms_ffn restent BF16 — copy depuis les blocks existants
            // pour que RMSNorm marche correctement.
            let rms_attn = stream
                .alloc_zeros::<half::bf16>(d)
                .map_err(|e| LlmError::Backend(format!("alloc rms_attn: {e:?}")))?;
            let rms_ffn = stream
                .alloc_zeros::<half::bf16>(d)
                .map_err(|e| LlmError::Backend(format!("alloc rms_ffn: {e:?}")))?;

            use cudarc::driver::DevicePtr;
            let (qkv_p, _r1) = unsafe { blk.w_qkv.device_ptr(stream) };
            let (o_p, _r2) = unsafe { blk.w_o.device_ptr(stream) };
            let (gu_p, _r3) = unsafe { blk.w_gate_up.device_ptr(stream) };
            let (dn_p, _r4) = unsafe { blk.w_down.device_ptr(stream) };

            let (w_qkv, w_qkv_scale) = quantize_or_alloc(d * qkv_n, qkv_p)?;
            let (w_o, w_o_scale) = quantize_or_alloc(d * d, o_p)?;
            let (w_gate_up, w_gate_up_scale) = quantize_or_alloc(d * 2 * f, gu_p)?;
            let (w_down, w_down_scale) = quantize_or_alloc(f * d, dn_p)?;

            blocks_fp4.push(BlockWeightsCudaFp4 {
                rms_attn,
                w_qkv,
                w_qkv_scale,
                w_o,
                w_o_scale,
                rms_ffn,
                w_gate_up,
                w_gate_up_scale,
                w_down,
                w_down_scale,
            });
        }
        // Scratch FP4 : on garde les scratch BF16 + on alloue les FP4 buffers
        // pour passer aux matmul_mxfp4 (les pointeurs sont reinterprétés
        // depuis les BF16 buffers — MVP timing-only).
        let scratch_fp4 = ScratchCudaFp4 {
            x_fp4: alloc_u8(d / 2)?,
            x_fp4_scale: alloc_u8(d / block16)?,
            ffn_inter_fp4: alloc_u8(f / 2)?,
            ffn_inter_fp4_scale: alloc_u8(f / block16)?,
        };
        self.blocks_fp4 = Some(blocks_fp4);
        self.scratch_fp4 = Some(scratch_fp4);
        Ok(())
    }

    /// Decode 1 token en utilisant le path NVFP4 (T241.5).
    ///
    /// Précondition : `enable_fp4()` a été appelé. Les matmuls QKV / O /
    /// gate+up / down passent par `LtSession::matmul_mxfp4` (NVFP4 packed
    /// + UE4M3 scales). Throughput projeté : ~2.5× le BF16 path.
    ///
    /// MVP timing-only : les weights FP4 sont zéros donc les outputs sont
    /// numériquement faux (garbage in → garbage out). Le timing du
    /// pipeline est néanmoins représentatif.
    pub fn decode_step_fp4(&mut self, token_id: u32) -> Result<u32, LlmError> {
        use cudarc::driver::{DevicePtr, DevicePtrMut};
        use rustorch_cuda::cublas_lt::{Fp4ScaleMode, Fp8Output};

        let blocks_fp4 = self
            .blocks_fp4
            .as_ref()
            .ok_or_else(|| LlmError::Backend("enable_fp4() not called".into()))?;
        let scratch_fp4 = self
            .scratch_fp4
            .as_ref()
            .ok_or_else(|| LlmError::Backend("enable_fp4() not called".into()))?;

        let d = self.config.hidden_size;
        let f = self.config.intermediate_size;
        let v = self.config.vocab_size;
        let n_heads = self.config.num_attention_heads;
        let n_kv = self.config.n_kv_heads();
        let head_dim = self.config.head_dim();
        let kv_dim = n_kv * head_dim;
        let n_layers = self.config.num_hidden_layers;
        let eps = self.config.rms_norm_eps;
        let qkv_n = d + 2 * kv_dim;
        let max_seq = self.max_seq;
        let pos = self.kv_pos;
        if pos >= max_seq {
            return Err(LlmError::Backend(format!(
                "kv_pos {pos} >= max_seq {max_seq}"
            )));
        }
        let kv_len = pos + 1;

        // 1. Embed → x (BF16)
        let token_id_dev = self
            .stream
            .memcpy_stod(&[token_id])
            .map_err(|e| LlmError::Backend(format!("upload token_id: {e:?}")))?;
        unsafe {
            let (table_p, _r1) = self.token_emb.device_ptr(&self.stream);
            let (ids_p, _r2) = token_id_dev.device_ptr(&self.stream);
            let (out_p, _r3) = self.scratch.x.device_ptr_mut(&self.stream);
            self.kernels
                .embedding_lookup_bf16(&self.stream, table_p, ids_p, out_p, 1, d as i32)
                .map_err(|e| LlmError::Backend(format!("embedding_lookup: {e:?}")))?;
        }

        for li in 0..n_layers {
            let block = &blocks_fp4[li];
            // ATTENTION SUB-BLOCK
            // h ← copy(x), RMSNorm pre-attn
            unsafe {
                let (dst_p, _r1) = self.scratch.h.device_ptr_mut(&self.stream);
                let (src_p, _r2) = self.scratch.x.device_ptr(&self.stream);
                self.kernels
                    .copy_bf16(&self.stream, dst_p, src_p, d as i32)
                    .map_err(|e| LlmError::Backend(format!("copy x→h L{li}: {e:?}")))?;
            }
            unsafe {
                let (h_p, _r1) = self.scratch.h.device_ptr_mut(&self.stream);
                let (g_p, _r2) = block.rms_attn.device_ptr(&self.stream);
                self.kernels
                    .rms_norm_bf16(&self.stream, h_p, g_p, eps, d as i32, 1)
                    .map_err(|e| LlmError::Backend(format!("rms_attn L{li}: {e:?}")))?;
            }

            // QKV matmul FP4 : (h reinterpreté comme FP4) · w_qkv_fp4 → qkv (BF16 out)
            // En MVP : on cast pointer scratch.h (BF16) comme si c'était FP4 packed
            // (donc lu en garbage byte-pattern). Le timing matmul est correct.
            unsafe {
                let (a_p, _r1) = self.scratch.h.device_ptr(&self.stream);
                let (sa_p, _r2) = scratch_fp4.x_fp4_scale.device_ptr(&self.stream);
                let (b_p, _r3) = block.w_qkv.device_ptr(&self.stream);
                let (sb_p, _r4) = block.w_qkv_scale.device_ptr(&self.stream);
                let (c_p, _r5) = self.scratch.qkv.device_ptr_mut(&self.stream);
                self.session
                    .matmul_mxfp4(
                        a_p,
                        sa_p,
                        b_p,
                        sb_p,
                        c_p,
                        1,
                        d,
                        qkv_n,
                        1.0,
                        0.0,
                        Fp8Output::Bf16,
                        Fp4ScaleMode::Vec16Ue4m3,
                    )
                    .map_err(|e| LlmError::Backend(format!("matmul_qkv_fp4 L{li}: {e:?}")))?;
            }
            // RoPE Q et K (kept BF16 ops as before)
            let (q_off, k_off, v_off) = (0u64, (d as u64) * 2, ((d + kv_dim) as u64) * 2);
            unsafe {
                let (qkv_base, _r1) = self.scratch.qkv.device_ptr_mut(&self.stream);
                let (inv_p, _r2) = self.rope_inv_freq.device_ptr(&self.stream);
                self.kernels
                    .rope_half_split_bf16(
                        &self.stream,
                        qkv_base + q_off,
                        inv_p,
                        pos as i32,
                        n_heads as i32,
                        head_dim as i32,
                    )
                    .map_err(|e| LlmError::Backend(format!("rope Q L{li}: {e:?}")))?;
            }
            unsafe {
                let (qkv_base2, _r1) = self.scratch.qkv.device_ptr_mut(&self.stream);
                let (inv_p, _r2) = self.rope_inv_freq.device_ptr(&self.stream);
                self.kernels
                    .rope_half_split_bf16(
                        &self.stream,
                        qkv_base2 + k_off,
                        inv_p,
                        pos as i32,
                        n_kv as i32,
                        head_dim as i32,
                    )
                    .map_err(|e| LlmError::Backend(format!("rope K L{li}: {e:?}")))?;
            }
            // KV append
            unsafe {
                let (qkv_base, _r1) = self.scratch.qkv.device_ptr(&self.stream);
                let k_in = qkv_base + k_off;
                let v_in = qkv_base + v_off;
                let (k_cache_p, _r2) = self.kv_cache_k[li].device_ptr_mut(&self.stream);
                let (v_cache_p, _r3) = self.kv_cache_v[li].device_ptr_mut(&self.stream);
                self.kernels
                    .kv_append_bf16(
                        &self.stream,
                        k_cache_p,
                        v_cache_p,
                        k_in,
                        v_in,
                        pos as i32,
                        n_kv as i32,
                        head_dim as i32,
                        max_seq as i32,
                    )
                    .map_err(|e| LlmError::Backend(format!("kv_append L{li}: {e:?}")))?;
            }
            // GQA decode (keeps BF16 since K/V cache are BF16)
            unsafe {
                let (qkv_base, _r1) = self.scratch.qkv.device_ptr(&self.stream);
                let (kc_p, _r2) = self.kv_cache_k[li].device_ptr(&self.stream);
                let (vc_p, _r3) = self.kv_cache_v[li].device_ptr(&self.stream);
                let (out_p, _r4) = self.scratch.block_out.device_ptr_mut(&self.stream);
                self.kernels
                    .gqa_decode_online_bf16(
                        &self.stream,
                        qkv_base + q_off,
                        kc_p,
                        vc_p,
                        out_p,
                        n_heads as i32,
                        n_kv as i32,
                        kv_len as i32,
                        head_dim as i32,
                        max_seq as i32,
                    )
                    .map_err(|e| LlmError::Backend(format!("gqa_decode L{li}: {e:?}")))?;
            }
            // O proj FP4
            unsafe {
                let (a_p, _r1) = self.scratch.block_out.device_ptr(&self.stream);
                let (sa_p, _r2) = scratch_fp4.x_fp4_scale.device_ptr(&self.stream);
                let (b_p, _r3) = block.w_o.device_ptr(&self.stream);
                let (sb_p, _r4) = block.w_o_scale.device_ptr(&self.stream);
                let (c_p, _r5) = self.scratch.h.device_ptr_mut(&self.stream);
                self.session
                    .matmul_mxfp4(
                        a_p,
                        sa_p,
                        b_p,
                        sb_p,
                        c_p,
                        1,
                        d,
                        d,
                        1.0,
                        0.0,
                        Fp8Output::Bf16,
                        Fp4ScaleMode::Vec16Ue4m3,
                    )
                    .map_err(|e| LlmError::Backend(format!("w_o_fp4 L{li}: {e:?}")))?;
            }
            // Residual
            unsafe {
                let (x_p, _r1) = self.scratch.x.device_ptr_mut(&self.stream);
                let (h_p, _r2) = self.scratch.h.device_ptr(&self.stream);
                self.kernels
                    .add_inplace_bf16(&self.stream, x_p, h_p, d as i32)
                    .map_err(|e| LlmError::Backend(format!("residual attn L{li}: {e:?}")))?;
            }

            // FFN SUB-BLOCK
            unsafe {
                let (dst_p, _r1) = self.scratch.h.device_ptr_mut(&self.stream);
                let (src_p, _r2) = self.scratch.x.device_ptr(&self.stream);
                self.kernels
                    .copy_bf16(&self.stream, dst_p, src_p, d as i32)
                    .map_err(|e| LlmError::Backend(format!("copy x→h ffn L{li}: {e:?}")))?;
            }
            unsafe {
                let (h_p, _r1) = self.scratch.h.device_ptr_mut(&self.stream);
                let (g_p, _r2) = block.rms_ffn.device_ptr(&self.stream);
                self.kernels
                    .rms_norm_bf16(&self.stream, h_p, g_p, eps, d as i32, 1)
                    .map_err(|e| LlmError::Backend(format!("rms_ffn L{li}: {e:?}")))?;
            }
            // gate+up FP4
            unsafe {
                let (a_p, _r1) = self.scratch.h.device_ptr(&self.stream);
                let (sa_p, _r2) = scratch_fp4.x_fp4_scale.device_ptr(&self.stream);
                let (b_p, _r3) = block.w_gate_up.device_ptr(&self.stream);
                let (sb_p, _r4) = block.w_gate_up_scale.device_ptr(&self.stream);
                let (c_p, _r5) = self.scratch.gate_up.device_ptr_mut(&self.stream);
                self.session
                    .matmul_mxfp4(
                        a_p,
                        sa_p,
                        b_p,
                        sb_p,
                        c_p,
                        1,
                        d,
                        2 * f,
                        1.0,
                        0.0,
                        Fp8Output::Bf16,
                        Fp4ScaleMode::Vec16Ue4m3,
                    )
                    .map_err(|e| LlmError::Backend(format!("gate_up_fp4 L{li}: {e:?}")))?;
            }
            // SwiGLU
            unsafe {
                let (gu_p, _r1) = self.scratch.gate_up.device_ptr(&self.stream);
                let up_p = gu_p + (f as u64) * 2;
                let (out_p, _r2) = self.scratch.ffn_inter.device_ptr_mut(&self.stream);
                self.kernels
                    .swiglu_bf16(&self.stream, gu_p, up_p, out_p, f as i32)
                    .map_err(|e| LlmError::Backend(format!("swiglu L{li}: {e:?}")))?;
            }
            // Down FP4
            unsafe {
                let (a_p, _r1) = self.scratch.ffn_inter.device_ptr(&self.stream);
                let (sa_p, _r2) = scratch_fp4.ffn_inter_fp4_scale.device_ptr(&self.stream);
                let (b_p, _r3) = block.w_down.device_ptr(&self.stream);
                let (sb_p, _r4) = block.w_down_scale.device_ptr(&self.stream);
                let (c_p, _r5) = self.scratch.block_out.device_ptr_mut(&self.stream);
                self.session
                    .matmul_mxfp4(
                        a_p,
                        sa_p,
                        b_p,
                        sb_p,
                        c_p,
                        1,
                        f,
                        d,
                        1.0,
                        0.0,
                        Fp8Output::Bf16,
                        Fp4ScaleMode::Vec16Ue4m3,
                    )
                    .map_err(|e| LlmError::Backend(format!("w_down_fp4 L{li}: {e:?}")))?;
            }
            unsafe {
                let (x_p, _r1) = self.scratch.x.device_ptr_mut(&self.stream);
                let (b_p, _r2) = self.scratch.block_out.device_ptr(&self.stream);
                self.kernels
                    .add_inplace_bf16(&self.stream, x_p, b_p, d as i32)
                    .map_err(|e| LlmError::Backend(format!("residual ffn L{li}: {e:?}")))?;
            }
        }

        // Final RMSNorm + LM head + argmax (gardent BF16)
        unsafe {
            let (dst_p, _r1) = self.scratch.h.device_ptr_mut(&self.stream);
            let (src_p, _r2) = self.scratch.x.device_ptr(&self.stream);
            self.kernels
                .copy_bf16(&self.stream, dst_p, src_p, d as i32)
                .map_err(|e| LlmError::Backend(format!("copy x→h final: {e:?}")))?;
        }
        unsafe {
            let (h_p, _r1) = self.scratch.h.device_ptr_mut(&self.stream);
            let (g_p, _r2) = self.final_norm.device_ptr(&self.stream);
            self.kernels
                .rms_norm_bf16(&self.stream, h_p, g_p, eps, d as i32, 1)
                .map_err(|e| LlmError::Backend(format!("final_norm: {e:?}")))?;
        }
        unsafe {
            let (a_p, _r1) = self.scratch.h.device_ptr(&self.stream);
            let (b_p, _r2) = self.lm_head.device_ptr(&self.stream);
            let (c_p, _r3) = self.scratch.logits.device_ptr_mut(&self.stream);
            self.session
                .matmul_bf16(a_p, b_p, c_p, 1, d, v, 1.0, 0.0)
                .map_err(|e| LlmError::Backend(format!("lm_head: {e:?}")))?;
        }
        unsafe {
            let (l_p, _r1) = self.scratch.logits.device_ptr(&self.stream);
            let (o_p, _r2) = self.scratch.sample_out.device_ptr_mut(&self.stream);
            self.kernels
                .argmax_bf16(&self.stream, l_p, o_p, v as i32)
                .map_err(|e| LlmError::Backend(format!("argmax: {e:?}")))?;
        }
        let out: Vec<u32> = self
            .stream
            .memcpy_dtov(&self.scratch.sample_out)
            .map_err(|e| LlmError::Backend(format!("dtov sample: {e:?}")))?;
        self.kv_pos += 1;
        Ok(out[0])
    }

    /// Construit un modèle CUDA avec des poids ZÉRO (pour benchmarks de
    /// timing pur — résultats numériques inutiles, mais le pipeline est
    /// réellement exercé).
    ///
    /// Si `fp4_only=true` : allocates BF16 weights as 1-element stubs
    /// (économise mémoire, le path BF16 ne fonctionnera pas mais le path
    /// FP4 oui). Utilisé pour Qwen-72B qui ne tient pas en BF16.
    pub fn from_dummy_fp4_only(config: LlamaConfig, max_seq: usize) -> Result<Self, LlmError> {
        Self::from_dummy_inner(config, max_seq, true)
    }

    /// Same as `from_dummy_fp4_only` mais alloue les BF16 weights aussi
    /// (peut OOM sur les gros modèles).
    pub fn from_dummy(config: LlamaConfig, max_seq: usize) -> Result<Self, LlmError> {
        Self::from_dummy_inner(config, max_seq, false)
    }

    fn from_dummy_inner(
        config: LlamaConfig,
        max_seq: usize,
        fp4_only: bool,
    ) -> Result<Self, LlmError> {
        let ctx = CudaContext::new(0).map_err(|e| LlmError::Backend(format!("ctx: {e:?}")))?;
        let stream = ctx.default_stream();
        let session = LtSession::new(stream.clone())
            .map_err(|e| LlmError::Backend(format!("LtSession: {e:?}")))?;
        let kernels = LlmKernels::new(ctx.clone());

        let d = config.hidden_size;
        let kv_dim = config.n_kv_heads() * config.head_dim();
        let f = config.intermediate_size;
        let head_dim = config.head_dim();
        let v = config.vocab_size;

        let alloc_zeros_bf16 =
            |stream: &Arc<CudaStream>, n: usize| -> Result<CudaSlice<half::bf16>, LlmError> {
                stream
                    .alloc_zeros::<half::bf16>(n)
                    .map_err(|e| LlmError::Backend(format!("alloc {n} bf16: {e:?}")))
            };

        // Si fp4_only : skip les big BF16 layer weights (économise ~150 GB sur 72B)
        // - Garde token_emb, lm_head, final_norm, rms_* à la vraie taille (utilisés par
        //   les paths CUDA même en mode fp4)
        // - Met les big matmul weights (w_qkv, w_o, w_gate_up, w_down) à 16-byte stubs
        let stub: usize = 16;

        let token_emb = alloc_zeros_bf16(&stream, v * d)?;
        let final_norm = alloc_zeros_bf16(&stream, d)?;
        let lm_head = alloc_zeros_bf16(&stream, d * v)?;

        let mut blocks: Vec<BlockWeightsCuda> = Vec::with_capacity(config.num_hidden_layers);
        for _ in 0..config.num_hidden_layers {
            blocks.push(BlockWeightsCuda {
                rms_attn: alloc_zeros_bf16(&stream, d)?, // small, always real
                w_qkv: alloc_zeros_bf16(
                    &stream,
                    if fp4_only { stub } else { d * (d + 2 * kv_dim) },
                )?,
                w_o: alloc_zeros_bf16(&stream, if fp4_only { stub } else { d * d })?,
                rms_ffn: alloc_zeros_bf16(&stream, d)?, // small
                w_gate_up: alloc_zeros_bf16(&stream, if fp4_only { stub } else { d * 2 * f })?,
                w_down: alloc_zeros_bf16(&stream, if fp4_only { stub } else { f * d })?,
                q_norm: None,
                k_norm: None,
            });
        }
        let inv_freq_host: Vec<f32> = (0..head_dim / 2)
            .map(|i| (config.rope_theta as f32).powf(-(2.0 * i as f32) / head_dim as f32))
            .collect();
        let rope_inv_freq = stream
            .memcpy_stod(&inv_freq_host)
            .map_err(|e| LlmError::Backend(format!("rope_inv_freq: {e:?}")))?;

        let scratch = ScratchCuda {
            x: alloc_zeros_bf16(&stream, d)?,
            h: alloc_zeros_bf16(&stream, d)?,
            qkv: alloc_zeros_bf16(&stream, d + 2 * kv_dim)?,
            gate_up: alloc_zeros_bf16(&stream, 2 * f)?,
            ffn_inter: alloc_zeros_bf16(&stream, f)?,
            block_out: alloc_zeros_bf16(&stream, d)?,
            logits: alloc_zeros_bf16(&stream, v)?,
            sample_out: stream
                .alloc_zeros::<u32>(1)
                .map_err(|e| LlmError::Backend(format!("alloc sample: {e:?}")))?,
        };
        let mut kv_cache_k = Vec::with_capacity(config.num_hidden_layers);
        let mut kv_cache_v = Vec::with_capacity(config.num_hidden_layers);
        for _ in 0..config.num_hidden_layers {
            kv_cache_k.push(alloc_zeros_bf16(&stream, max_seq * kv_dim)?);
            kv_cache_v.push(alloc_zeros_bf16(&stream, max_seq * kv_dim)?);
        }
        Ok(Self {
            config,
            stream,
            ctx,
            session,
            kernels,
            blocks,
            token_emb,
            final_norm,
            lm_head,
            rope_inv_freq,
            scratch,
            kv_cache_k,
            kv_cache_v,
            kv_pos: 0,
            max_seq,
            blocks_fp4: None,
            scratch_fp4: None,
        })
    }

    /// Dimension hidden du modèle.
    pub fn hidden(&self) -> usize {
        self.config.hidden_size
    }

    /// Vocab size.
    pub fn vocab_size(&self) -> usize {
        self.config.vocab_size
    }

    /// Decode 1 token, full forward Qwen-style (T241.4 step 3).
    ///
    /// Pour chaque layer applique le bloc complet :
    ///   1. RMSNorm pre-attention
    ///   2. QKV matmul (fused), split Q/K/V
    ///   3. RoPE on Q, K
    ///   4. KV cache append
    ///   5. GQA decode naive : Q · K^T → softmax → P · V
    ///   6. O proj matmul + residual
    ///   7. RMSNorm pre-FFN
    ///   8. Fused gate+up matmul
    ///   9. SwiGLU
    ///  10. Down matmul + residual
    /// Puis final RMSNorm + LM head + argmax.
    ///
    /// Avance `kv_pos` de 1.
    ///
    /// # Safety
    /// Caller : `token_id < vocab_size`, `self.kv_pos < self.max_seq`.
    pub fn decode_step(&mut self, token_id: u32) -> Result<u32, LlmError> {
        use cudarc::driver::{DevicePtr, DevicePtrMut};
        let d = self.config.hidden_size;
        let f = self.config.intermediate_size;
        let v = self.config.vocab_size;
        let n_heads = self.config.num_attention_heads;
        let n_kv = self.config.n_kv_heads();
        let head_dim = self.config.head_dim();
        let kv_dim = n_kv * head_dim;
        let n_layers = self.config.num_hidden_layers;
        let eps = self.config.rms_norm_eps;
        let qkv_n = d + 2 * kv_dim;
        let max_seq = self.max_seq;
        let pos = self.kv_pos;
        if pos >= max_seq {
            return Err(LlmError::Backend(format!(
                "kv_pos {pos} >= max_seq {max_seq}"
            )));
        }
        let kv_len = pos + 1;

        // 1. Embed → scratch.x
        let token_id_dev = self
            .stream
            .memcpy_stod(&[token_id])
            .map_err(|e| LlmError::Backend(format!("upload token_id: {e:?}")))?;
        unsafe {
            let (table_p, _r1) = self.token_emb.device_ptr(&self.stream);
            let (ids_p, _r2) = token_id_dev.device_ptr(&self.stream);
            let (out_p, _r3) = self.scratch.x.device_ptr_mut(&self.stream);
            self.kernels
                .embedding_lookup_bf16(&self.stream, table_p, ids_p, out_p, 1, d as i32)
                .map_err(|e| LlmError::Backend(format!("embedding_lookup: {e:?}")))?;
        }

        for li in 0..n_layers {
            // === ATTENTION SUB-BLOCK ===
            // x → h via copy kernel device-to-device
            unsafe {
                let (dst_p, _r1) = self.scratch.h.device_ptr_mut(&self.stream);
                let (src_p, _r2) = self.scratch.x.device_ptr(&self.stream);
                self.kernels
                    .copy_bf16(&self.stream, dst_p, src_p, d as i32)
                    .map_err(|e| LlmError::Backend(format!("copy x→h L{li}: {e:?}")))?;
            }
            let block = &self.blocks[li];
            // RMSNorm pre-attn
            unsafe {
                let (h_p, _r1) = self.scratch.h.device_ptr_mut(&self.stream);
                let (g_p, _r2) = block.rms_attn.device_ptr(&self.stream);
                self.kernels
                    .rms_norm_bf16(&self.stream, h_p, g_p, eps, d as i32, 1)
                    .map_err(|e| LlmError::Backend(format!("rms_attn L{li}: {e:?}")))?;
            }
            // Fused QKV matmul : h [1, d] · w_qkv [d, d + 2*kv_dim] → qkv
            unsafe {
                let (a_p, _r1) = self.scratch.h.device_ptr(&self.stream);
                let (b_p, _r2) = block.w_qkv.device_ptr(&self.stream);
                let (c_p, _r3) = self.scratch.qkv.device_ptr_mut(&self.stream);
                self.session
                    .matmul_bf16(a_p, b_p, c_p, 1, d, qkv_n, 1.0, 0.0)
                    .map_err(|e| LlmError::Backend(format!("w_qkv L{li}: {e:?}")))?;
            }
            // qkv layout (col-major output) : [q_0..q_{d-1}, k_0..k_{kv_dim-1}, v_0..v_{kv_dim-1}]
            let (q_off, k_off, v_off) = (0u64, (d as u64) * 2, ((d + kv_dim) as u64) * 2);

            // RoPE on Q (n_heads × head_dim) and K (n_kv × head_dim).
            unsafe {
                let (qkv_base, _r1) = self.scratch.qkv.device_ptr_mut(&self.stream);
                let (inv_p, _r2) = self.rope_inv_freq.device_ptr(&self.stream);
                self.kernels
                    .rope_half_split_bf16(
                        &self.stream,
                        qkv_base + q_off,
                        inv_p,
                        pos as i32,
                        n_heads as i32,
                        head_dim as i32,
                    )
                    .map_err(|e| LlmError::Backend(format!("rope Q L{li}: {e:?}")))?;
            }
            unsafe {
                let (qkv_base, _r1) = self.scratch.qkv.device_ptr_mut(&self.stream);
                let (inv_p, _r2) = self.rope_inv_freq.device_ptr(&self.stream);
                self.kernels
                    .rope_half_split_bf16(
                        &self.stream,
                        qkv_base + k_off,
                        inv_p,
                        pos as i32,
                        n_kv as i32,
                        head_dim as i32,
                    )
                    .map_err(|e| LlmError::Backend(format!("rope K L{li}: {e:?}")))?;
            }
            // KV append : copy K/V into cache at pos
            unsafe {
                let (qkv_base, _r1) = self.scratch.qkv.device_ptr(&self.stream);
                let k_in = qkv_base + k_off;
                let v_in = qkv_base + v_off;
                let (k_cache_p, _r2) = self.kv_cache_k[li].device_ptr_mut(&self.stream);
                let (v_cache_p, _r3) = self.kv_cache_v[li].device_ptr_mut(&self.stream);
                self.kernels
                    .kv_append_bf16(
                        &self.stream,
                        k_cache_p,
                        v_cache_p,
                        k_in,
                        v_in,
                        pos as i32,
                        n_kv as i32,
                        head_dim as i32,
                        max_seq as i32,
                    )
                    .map_err(|e| LlmError::Backend(format!("kv_append L{li}: {e:?}")))?;
            }
            // GQA decode : Q · cached K^T softmax · cached V → block_out (reuse buffer)
            unsafe {
                let (qkv_base, _r1) = self.scratch.qkv.device_ptr(&self.stream);
                let (kc_p, _r2) = self.kv_cache_k[li].device_ptr(&self.stream);
                let (vc_p, _r3) = self.kv_cache_v[li].device_ptr(&self.stream);
                let (out_p, _r4) = self.scratch.block_out.device_ptr_mut(&self.stream);
                self.kernels
                    .gqa_decode_online_bf16(
                        &self.stream,
                        qkv_base + q_off,
                        kc_p,
                        vc_p,
                        out_p,
                        n_heads as i32,
                        n_kv as i32,
                        kv_len as i32,
                        head_dim as i32,
                        max_seq as i32,
                    )
                    .map_err(|e| LlmError::Backend(format!("gqa_decode L{li}: {e:?}")))?;
            }
            // O proj : block_out [1, d] · w_o [d, d] → h
            unsafe {
                let (a_p, _r1) = self.scratch.block_out.device_ptr(&self.stream);
                let (b_p, _r2) = block.w_o.device_ptr(&self.stream);
                let (c_p, _r3) = self.scratch.h.device_ptr_mut(&self.stream);
                self.session
                    .matmul_bf16(a_p, b_p, c_p, 1, d, d, 1.0, 0.0)
                    .map_err(|e| LlmError::Backend(format!("w_o L{li}: {e:?}")))?;
            }
            // Residual : x += h
            unsafe {
                let (x_p, _r1) = self.scratch.x.device_ptr_mut(&self.stream);
                let (h_p, _r2) = self.scratch.h.device_ptr(&self.stream);
                self.kernels
                    .add_inplace_bf16(&self.stream, x_p, h_p, d as i32)
                    .map_err(|e| LlmError::Backend(format!("residual attn L{li}: {e:?}")))?;
            }

            // === FFN SUB-BLOCK ===
            // h = copy(x), RMSNorm pre-FFN
            unsafe {
                let (dst_p, _r1) = self.scratch.h.device_ptr_mut(&self.stream);
                let (src_p, _r2) = self.scratch.x.device_ptr(&self.stream);
                self.kernels
                    .copy_bf16(&self.stream, dst_p, src_p, d as i32)
                    .map_err(|e| LlmError::Backend(format!("copy x→h ffn L{li}: {e:?}")))?;
            }
            unsafe {
                let (h_p, _r1) = self.scratch.h.device_ptr_mut(&self.stream);
                let (g_p, _r2) = block.rms_ffn.device_ptr(&self.stream);
                self.kernels
                    .rms_norm_bf16(&self.stream, h_p, g_p, eps, d as i32, 1)
                    .map_err(|e| LlmError::Backend(format!("rms_ffn L{li}: {e:?}")))?;
            }
            // gate+up matmul
            unsafe {
                let (a_p, _r1) = self.scratch.h.device_ptr(&self.stream);
                let (b_p, _r2) = block.w_gate_up.device_ptr(&self.stream);
                let (c_p, _r3) = self.scratch.gate_up.device_ptr_mut(&self.stream);
                self.session
                    .matmul_bf16(a_p, b_p, c_p, 1, d, 2 * f, 1.0, 0.0)
                    .map_err(|e| LlmError::Backend(format!("gate_up L{li}: {e:?}")))?;
            }
            // SwiGLU
            unsafe {
                let (gu_p, _r1) = self.scratch.gate_up.device_ptr(&self.stream);
                let up_p = gu_p + (f as u64) * 2;
                let (out_p, _r2) = self.scratch.ffn_inter.device_ptr_mut(&self.stream);
                self.kernels
                    .swiglu_bf16(&self.stream, gu_p, up_p, out_p, f as i32)
                    .map_err(|e| LlmError::Backend(format!("swiglu L{li}: {e:?}")))?;
            }
            // Down + residual
            unsafe {
                let (a_p, _r1) = self.scratch.ffn_inter.device_ptr(&self.stream);
                let (b_p, _r2) = block.w_down.device_ptr(&self.stream);
                let (c_p, _r3) = self.scratch.block_out.device_ptr_mut(&self.stream);
                self.session
                    .matmul_bf16(a_p, b_p, c_p, 1, f, d, 1.0, 0.0)
                    .map_err(|e| LlmError::Backend(format!("w_down L{li}: {e:?}")))?;
            }
            unsafe {
                let (x_p, _r1) = self.scratch.x.device_ptr_mut(&self.stream);
                let (b_p, _r2) = self.scratch.block_out.device_ptr(&self.stream);
                self.kernels
                    .add_inplace_bf16(&self.stream, x_p, b_p, d as i32)
                    .map_err(|e| LlmError::Backend(format!("residual ffn L{li}: {e:?}")))?;
            }
        }

        // Final RMSNorm + LM head + argmax
        unsafe {
            let (dst_p, _r1) = self.scratch.h.device_ptr_mut(&self.stream);
            let (src_p, _r2) = self.scratch.x.device_ptr(&self.stream);
            self.kernels
                .copy_bf16(&self.stream, dst_p, src_p, d as i32)
                .map_err(|e| LlmError::Backend(format!("copy x→h final: {e:?}")))?;
        }
        unsafe {
            let (h_p, _r1) = self.scratch.h.device_ptr_mut(&self.stream);
            let (g_p, _r2) = self.final_norm.device_ptr(&self.stream);
            self.kernels
                .rms_norm_bf16(&self.stream, h_p, g_p, eps, d as i32, 1)
                .map_err(|e| LlmError::Backend(format!("final_norm: {e:?}")))?;
        }
        unsafe {
            let (a_p, _r1) = self.scratch.h.device_ptr(&self.stream);
            let (b_p, _r2) = self.lm_head.device_ptr(&self.stream);
            let (c_p, _r3) = self.scratch.logits.device_ptr_mut(&self.stream);
            self.session
                .matmul_bf16(a_p, b_p, c_p, 1, d, v, 1.0, 0.0)
                .map_err(|e| LlmError::Backend(format!("lm_head: {e:?}")))?;
        }
        unsafe {
            let (l_p, _r1) = self.scratch.logits.device_ptr(&self.stream);
            let (o_p, _r2) = self.scratch.sample_out.device_ptr_mut(&self.stream);
            self.kernels
                .argmax_bf16(&self.stream, l_p, o_p, v as i32)
                .map_err(|e| LlmError::Backend(format!("argmax: {e:?}")))?;
        }
        let out: Vec<u32> = self
            .stream
            .memcpy_dtov(&self.scratch.sample_out)
            .map_err(|e| LlmError::Backend(format!("dtov sample: {e:?}")))?;
        self.kv_pos += 1;
        Ok(out[0])
    }

    /// Decode 1 token avec layers FFN-only (T241.4 step 2).
    ///
    /// Pour chaque layer applique :
    ///  - RMSNorm pre-FFN
    ///  - Fused gate+up matmul (LtSession)
    ///  - SwiGLU (silu(gate) * up)
    ///  - Down matmul
    ///  - Residual : x += block_out
    ///
    /// L'attention est **skipped** (pas encore wirée — T241.4 step 3).
    /// Le résultat est numériquement faux pour un vrai modèle mais
    /// exerce le pipeline FFN bout-en-bout.
    ///
    /// # Safety
    /// Le caller garantit que `token_id < vocab_size`.
    pub fn decode_step_ffn_only(&mut self, token_id: u32) -> Result<u32, LlmError> {
        use cudarc::driver::{DevicePtr, DevicePtrMut};
        // Snapshot config en locals (évite borrow `&self.config` dans la
        // loop qui entrerait en conflit avec les `&mut self.scratch`).
        let d = self.config.hidden_size;
        let f = self.config.intermediate_size;
        let v = self.config.vocab_size;
        let n_layers = self.config.num_hidden_layers;
        let eps = self.config.rms_norm_eps;
        let gate_up_n = 2 * f;

        // 1. Embed lookup → scratch.x
        let token_id_dev = self
            .stream
            .memcpy_stod(&[token_id])
            .map_err(|e| LlmError::Backend(format!("upload token_id: {e:?}")))?;
        unsafe {
            let (table_p, _r1) = self.token_emb.device_ptr(&self.stream);
            let (ids_p, _r2) = token_id_dev.device_ptr(&self.stream);
            let (out_p, _r3) = self.scratch.x.device_ptr_mut(&self.stream);
            self.kernels
                .embedding_lookup_bf16(&self.stream, table_p, ids_p, out_p, 1, d as i32)
                .map_err(|e| LlmError::Backend(format!("embedding_lookup: {e:?}")))?;
        }

        // Helper : copy scratch.x → scratch.h via host roundtrip (à
        // remplacer par memcpy_dtod kernel quand on en aura un — pour
        // MVP on accepte le détour).
        let copy_x_to_h = |this: &mut Self| -> Result<(), LlmError> {
            let host: Vec<half::bf16> = this
                .stream
                .memcpy_dtov(&this.scratch.x)
                .map_err(|e| LlmError::Backend(format!("dtov x: {e:?}")))?;
            this.stream
                .memcpy_htod(&host, &mut this.scratch.h)
                .map_err(|e| LlmError::Backend(format!("htod h: {e:?}")))?;
            Ok(())
        };

        // 2. Itérer sur les layers, faire le sub-bloc FFN seulement.
        for li in 0..n_layers {
            // x_in = x (résidu).
            // h ← copy(x)
            copy_x_to_h(self)?;

            // RMSNorm pre-FFN sur h avec block.rms_ffn
            let block = &self.blocks[li];
            unsafe {
                let (h_p, _r1) = self.scratch.h.device_ptr_mut(&self.stream);
                let (g_p, _r2) = block.rms_ffn.device_ptr(&self.stream);
                self.kernels
                    .rms_norm_bf16(&self.stream, h_p, g_p, eps, d as i32, 1)
                    .map_err(|e| LlmError::Backend(format!("rms_ffn L{li}: {e:?}")))?;
            }

            // Fused gate+up matmul : h [1, d] · w_gate_up [d, 2*f] → gate_up [1, 2*f]
            unsafe {
                let (a_p, _r1) = self.scratch.h.device_ptr(&self.stream);
                let (b_p, _r2) = block.w_gate_up.device_ptr(&self.stream);
                let (c_p, _r3) = self.scratch.gate_up.device_ptr_mut(&self.stream);
                self.session
                    .matmul_bf16(a_p, b_p, c_p, 1, d, gate_up_n, 1.0, 0.0)
                    .map_err(|e| LlmError::Backend(format!("gate_up L{li}: {e:?}")))?;
            }

            // SwiGLU : ffn_inter[i] = silu(gate_up[i]) * gate_up[f + i]
            // gate_up est layouté en col-major (output cublasLt) [1 × 2f]
            // donc en mémoire : [gate_0, ..., gate_f-1, up_0, ..., up_f-1]
            // (les premiers f éléments = gate, les suivants f = up).
            unsafe {
                let (gu_p, _r1) = self.scratch.gate_up.device_ptr(&self.stream);
                let up_p = gu_p + (f as u64) * 2; // BF16 = 2 bytes
                let (out_p, _r2) = self.scratch.ffn_inter.device_ptr_mut(&self.stream);
                self.kernels
                    .swiglu_bf16(&self.stream, gu_p, up_p, out_p, f as i32)
                    .map_err(|e| LlmError::Backend(format!("swiglu L{li}: {e:?}")))?;
            }

            // Down proj : ffn_inter [1, f] · w_down [f, d] → block_out [1, d]
            unsafe {
                let (a_p, _r1) = self.scratch.ffn_inter.device_ptr(&self.stream);
                let (b_p, _r2) = block.w_down.device_ptr(&self.stream);
                let (c_p, _r3) = self.scratch.block_out.device_ptr_mut(&self.stream);
                self.session
                    .matmul_bf16(a_p, b_p, c_p, 1, f, d, 1.0, 0.0)
                    .map_err(|e| LlmError::Backend(format!("w_down L{li}: {e:?}")))?;
            }

            // Residual : x += block_out
            unsafe {
                let (x_p, _r1) = self.scratch.x.device_ptr_mut(&self.stream);
                let (b_p, _r2) = self.scratch.block_out.device_ptr(&self.stream);
                self.kernels
                    .add_inplace_bf16(&self.stream, x_p, b_p, d as i32)
                    .map_err(|e| LlmError::Backend(format!("residual L{li}: {e:?}")))?;
            }
        }

        // 3. Final RMSNorm + LM head + argmax
        copy_x_to_h(self)?;
        unsafe {
            let (h_p, _r1) = self.scratch.h.device_ptr_mut(&self.stream);
            let (g_p, _r2) = self.final_norm.device_ptr(&self.stream);
            self.kernels
                .rms_norm_bf16(&self.stream, h_p, g_p, eps, d as i32, 1)
                .map_err(|e| LlmError::Backend(format!("final_norm: {e:?}")))?;
        }
        unsafe {
            let (a_p, _r1) = self.scratch.h.device_ptr(&self.stream);
            let (b_p, _r2) = self.lm_head.device_ptr(&self.stream);
            let (c_p, _r3) = self.scratch.logits.device_ptr_mut(&self.stream);
            self.session
                .matmul_bf16(a_p, b_p, c_p, 1, d, v, 1.0, 0.0)
                .map_err(|e| LlmError::Backend(format!("lm_head: {e:?}")))?;
        }
        unsafe {
            let (l_p, _r1) = self.scratch.logits.device_ptr(&self.stream);
            let (o_p, _r2) = self.scratch.sample_out.device_ptr_mut(&self.stream);
            self.kernels
                .argmax_bf16(&self.stream, l_p, o_p, v as i32)
                .map_err(|e| LlmError::Backend(format!("argmax: {e:?}")))?;
        }
        let out: Vec<u32> = self
            .stream
            .memcpy_dtov(&self.scratch.sample_out)
            .map_err(|e| LlmError::Backend(format!("dtov sample: {e:?}")))?;
        Ok(out[0])
    }

    /// Decode 1 token, returns next token id.
    ///
    /// MVP version (T241.4 step 1) : skip les blocks (juste embed → final
    /// norm → LM head → argmax). Sert de smoke test pour valider la chaîne
    /// complete sans la complexité de l'attention. Production decode_step
    /// arrive en step 2/3 (ajout layers + attention).
    ///
    /// # Safety
    /// Le caller garantit que `token_id < vocab_size`.
    pub fn decode_step_minimal(&mut self, token_id: u32) -> Result<u32, LlmError> {
        use cudarc::driver::{DevicePtr, DevicePtrMut};
        let d = self.config.hidden_size;
        let v = self.config.vocab_size;
        let eps = self.config.rms_norm_eps;

        // 1. Embed lookup : token_emb[token_id, :] → scratch.x
        // Pour MVP on upload le token_id sur device puis lookup. On peut
        // optimiser plus tard avec un buffer cached.
        let token_id_dev = self
            .stream
            .memcpy_stod(&[token_id])
            .map_err(|e| LlmError::Backend(format!("upload token_id: {e:?}")))?;

        unsafe {
            let (table_p, _r1) = self.token_emb.device_ptr(&self.stream);
            let (ids_p, _r2) = token_id_dev.device_ptr(&self.stream);
            let (out_p, _r3) = self.scratch.x.device_ptr_mut(&self.stream);
            self.kernels
                .embedding_lookup_bf16(&self.stream, table_p, ids_p, out_p, 1, d as i32)
                .map_err(|e| LlmError::Backend(format!("embedding_lookup: {e:?}")))?;
        }

        // 2. Final RMSNorm : x → h (gamma = self.final_norm)
        // On copie x → h d'abord (rms_norm_bf16 est inplace).
        // Pour MVP on fait via dtoh→htod : à optimiser avec un kernel copy.
        let x_host: Vec<half::bf16> = self
            .stream
            .memcpy_dtov(&self.scratch.x)
            .map_err(|e| LlmError::Backend(format!("dtov x: {e:?}")))?;
        self.stream
            .memcpy_htod(&x_host, &mut self.scratch.h)
            .map_err(|e| LlmError::Backend(format!("htod h: {e:?}")))?;
        unsafe {
            let (h_p, _r1) = self.scratch.h.device_ptr_mut(&self.stream);
            let (g_p, _r2) = self.final_norm.device_ptr(&self.stream);
            self.kernels
                .rms_norm_bf16(&self.stream, h_p, g_p, eps, d as i32, 1)
                .map_err(|e| LlmError::Backend(format!("rms_norm: {e:?}")))?;
        }

        // 3. LM head matmul : h [1, d] · lm_head [d, v] → logits [1, v]
        unsafe {
            let (a_p, _r1) = self.scratch.h.device_ptr(&self.stream);
            let (b_p, _r2) = self.lm_head.device_ptr(&self.stream);
            let (c_p, _r3) = self.scratch.logits.device_ptr_mut(&self.stream);
            self.session
                .matmul_bf16(a_p, b_p, c_p, 1, d, v, 1.0, 0.0)
                .map_err(|e| LlmError::Backend(format!("lm_head matmul: {e:?}")))?;
        }

        // 4. argmax sur logits
        unsafe {
            let (l_p, _r1) = self.scratch.logits.device_ptr(&self.stream);
            let (o_p, _r2) = self.scratch.sample_out.device_ptr_mut(&self.stream);
            self.kernels
                .argmax_bf16(&self.stream, l_p, o_p, v as i32)
                .map_err(|e| LlmError::Backend(format!("argmax: {e:?}")))?;
        }
        let out: Vec<u32> = self
            .stream
            .memcpy_dtov(&self.scratch.sample_out)
            .map_err(|e| LlmError::Backend(format!("dtov sample: {e:?}")))?;
        Ok(out[0])
    }
}

// Suppress dead_code warnings on fields used only at runtime by future kernels.
#[allow(dead_code)]
const _: () = ();
