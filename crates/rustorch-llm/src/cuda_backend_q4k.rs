//! T244.1.6 — `LlamaModelCudaQ4K` : CUDA forward path for Qwen2.5-style
//! GGUF Q4_K_M models, using sgemv_q4k_bf16_v2 + sgemv_q6k_bf16 directly
//! (no CPU dequant detour).
//!
//! Architecture vs `LlamaModelCuda` :
//! - Same Llama transformer (no SSM, no MoE).
//! - Weights stored as RAW Q4_K / Q6_K bytes on GPU (4× less memory than BF16).
//! - matmul dispatched via dtype : Q4_K → sgemv_q4k_bf16_v2, Q6_K → sgemv_q6k_bf16.
//! - Norms, biases, embeddings : F32 dequant to BF16 (small).
//!
//! Performance target on Qwen2.5-7B Q4_K_M : 35-45 tok/s (vs llama.cpp 47.15).

#![cfg(feature = "cuda")]

use crate::gguf_loader::GgufWeights;
use crate::{LlamaConfig, LlmError};
use cudarc::driver::{CudaContext, CudaSlice, CudaStream};
use rustorch_cuda::cublas_lt::LtSession;
use rustorch_cuda::llm_kernels::LlmKernels;
use rustorch_gguf::reader::GgufFile;
use rustorch_gguf::tensor::GgmlType;
use std::path::Path;
use std::sync::Arc;

/// One quantized matmul weight on GPU : Q4_K bytes, Q6_K bytes, or BF16 dequant.
enum QuantTensor {
    Bf16(CudaSlice<half::bf16>),
    Q4K {
        bytes: CudaSlice<u8>,
        n: usize, // out dim (rows)
        k: usize, // in dim (cols)
    },
    Q6K {
        bytes: CudaSlice<u8>,
        n: usize,
        k: usize,
    },
}

impl QuantTensor {
    fn dispatch_matmul(
        &self,
        kernels: &LlmKernels,
        session: &mut LtSession,
        stream: &Arc<CudaStream>,
        x: u64,
        y: u64,
        m: usize,
    ) -> Result<(), LlmError> {
        unsafe {
            use cudarc::driver::DevicePtr;
            match self {
                QuantTensor::Bf16(buf) => {
                    let (w, _g) = buf.device_ptr(stream);
                    // Need n, k from buffer size — caller passes m (=1) ; for BF16
                    // we can't recover n/k here. Caller must use specialized call.
                    // → BF16 path uses dispatch_matmul_bf16 below.
                    return Err(LlmError::Backend(format!(
                        "Bf16 dispatch needs explicit n/k ; use dispatch_matmul_bf16. ptr={w:#x} m={m}"
                    )));
                },
                QuantTensor::Q4K { bytes, n, k } => {
                    let (w, _g) = bytes.device_ptr(stream);
                    kernels
                        .sgemv_q4k_bf16_v2(stream, w, x, y, *n as i32, *k as i32)
                        .map_err(|e| LlmError::Backend(format!("sgemv_q4k: {e:?}")))?;
                },
                QuantTensor::Q6K { bytes, n, k } => {
                    let (w, _g) = bytes.device_ptr(stream);
                    // T244.4 — use V2 kernel (1.39× V1, 95→135 GB/s).
                    kernels
                        .sgemv_q6k_bf16_v2(stream, w, x, y, *n as i32, *k as i32)
                        .map_err(|e| LlmError::Backend(format!("sgemv_q6k_v2: {e:?}")))?;
                },
            }
        }
        Ok(())
    }

    /// BF16 fallback : uses cuBLASLt matmul_bf16 (caller provides shape since
    /// we can't recover from byte size of an opaque CudaSlice).
    fn dispatch_matmul_bf16_explicit(
        &self,
        session: &mut LtSession,
        stream: &Arc<CudaStream>,
        x: u64,
        y: u64,
        m: usize,
        k_in: usize,
        n_out: usize,
    ) -> Result<(), LlmError> {
        match self {
            QuantTensor::Bf16(buf) => {
                use cudarc::driver::DevicePtr;
                unsafe {
                    let (w, _g) = buf.device_ptr(stream);
                    session
                        .matmul_bf16(x, w, y, m, k_in, n_out, 1.0, 0.0)
                        .map_err(|e| LlmError::Backend(format!("matmul_bf16: {e:?}")))?;
                }
            },
            _ => {
                return Err(LlmError::Backend(
                    "dispatch_matmul_bf16_explicit called on non-BF16 tensor".into(),
                ))
            },
        }
        Ok(())
    }
}

/// Per-layer weights (mixed Q4K/Q6K for matmuls, BF16 for norms/biases).
struct BlockQ4K {
    /// `[D]` BF16 — RMSNorm pre-attention.
    attn_norm: CudaSlice<half::bf16>,
    /// `[D]` BF16 — RMSNorm pre-FFN (post-attention norm in HF naming).
    ffn_norm: CudaSlice<half::bf16>,
    /// Attention projections (typically Q6_K in Q4_K_M).
    w_q: QuantTensor,
    w_k: QuantTensor,
    w_v: QuantTensor,
    w_o: QuantTensor,
    /// Optional QKV biases (Qwen2/2.5/3) — BF16, applied post-matmul.
    b_q: Option<CudaSlice<half::bf16>>,
    b_k: Option<CudaSlice<half::bf16>>,
    b_v: Option<CudaSlice<half::bf16>>,
    /// FFN (gate+up Q4_K, down Q6_K in K_M).
    w_gate: QuantTensor,
    w_up: QuantTensor,
    w_down: QuantTensor,
}

/// CUDA model with Q4_K/Q6_K matmul weights for Qwen2.5-style GGUF.
pub struct LlamaModelCudaQ4K {
    pub config: LlamaConfig,
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    session: LtSession,
    kernels: LlmKernels,
    blocks: Vec<BlockQ4K>,
    /// Token embedding `[V, D]` BF16 (typically Q6_K in K_M, dequantized at load).
    token_emb: CudaSlice<half::bf16>,
    /// Final RMSNorm gamma `[D]` BF16.
    final_norm: CudaSlice<half::bf16>,
    /// LM head `[D, V]` BF16 (typically tied or Q6_K).
    lm_head: CudaSlice<half::bf16>,
    max_seq: usize,
}

impl LlamaModelCudaQ4K {
    /// Load Qwen-style GGUF (Q4_K_M typically) directly to GPU using
    /// raw Q4_K / Q6_K bytes for matmuls. Avoids the F32→BF16 detour.
    pub fn from_gguf(path: &Path, max_seq: usize) -> Result<Self, LlmError> {
        let ctx =
            CudaContext::new(0).map_err(|e| LlmError::Backend(format!("CudaContext: {e:?}")))?;
        let stream = ctx.default_stream();
        let session = LtSession::new(stream.clone())
            .map_err(|e| LlmError::Backend(format!("LtSession: {e:?}")))?;
        let kernels = LlmKernels::new(ctx.clone());

        let file = GgufFile::open(path).map_err(|e| LlmError::Backend(format!("gguf: {e:?}")))?;
        let cfg = LlamaConfig::from_gguf(&file)?;

        // Helper : load raw bytes for matmul weight (Q4_K or Q6_K).
        let load_quant = |name: &str| -> Result<QuantTensor, LlmError> {
            let info = file
                .tensor(name)
                .ok_or_else(|| LlmError::MissingWeight(name.to_string()))?;
            let bytes = file.tensor_bytes(info);
            let n = info.shape[1] as usize; // out
            let k = info.shape[0] as usize; // in
            let dev = stream
                .memcpy_stod(bytes)
                .map_err(|e| LlmError::Backend(format!("upload {name}: {e:?}")))?;
            match info.dtype {
                GgmlType::Q4_K => Ok(QuantTensor::Q4K { bytes: dev, n, k }),
                GgmlType::Q6_K => Ok(QuantTensor::Q6K { bytes: dev, n, k }),
                other => Err(LlmError::Backend(format!(
                    "unsupported dtype {other:?} for {name} — only Q4_K/Q6_K supported"
                ))),
            }
        };

        // Helper : load + dequantize tensor to BF16 (norms, biases, small tensors).
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

        let load_bf16_opt = |name: &str| -> Result<Option<CudaSlice<half::bf16>>, LlmError> {
            if file.tensor(name).is_none() {
                Ok(None)
            } else {
                load_bf16(name).map(Some)
            }
        };

        let token_emb = load_bf16("token_embd.weight")?;
        let final_norm = load_bf16("output_norm.weight")?;
        let lm_head = if file.tensor("output.weight").is_some() {
            load_bf16("output.weight")?
        } else {
            // tied embeddings — alias would be cleaner but for simplicity copy.
            load_bf16("token_embd.weight")?
        };

        let mut blocks: Vec<BlockQ4K> = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            let key = |s: &str| format!("blk.{i}.{s}");
            blocks.push(BlockQ4K {
                attn_norm: load_bf16(&key("attn_norm.weight"))?,
                ffn_norm: load_bf16(&key("ffn_norm.weight"))?,
                w_q: load_quant(&key("attn_q.weight"))?,
                w_k: load_quant(&key("attn_k.weight"))?,
                w_v: load_quant(&key("attn_v.weight"))?,
                w_o: load_quant(&key("attn_output.weight"))?,
                b_q: load_bf16_opt(&key("attn_q.bias"))?,
                b_k: load_bf16_opt(&key("attn_k.bias"))?,
                b_v: load_bf16_opt(&key("attn_v.bias"))?,
                w_gate: load_quant(&key("ffn_gate.weight"))?,
                w_up: load_quant(&key("ffn_up.weight"))?,
                w_down: load_quant(&key("ffn_down.weight"))?,
            });
        }

        Ok(Self {
            config: cfg,
            ctx,
            stream,
            session,
            kernels,
            blocks,
            token_emb,
            final_norm,
            lm_head,
            max_seq,
        })
    }

    /// Bench wall-clock for `n_iters` token decodes (matmul-only, skip
    /// attention/RMSNorm/RoPE/embedding which are <5% of compute).
    pub fn bench_decode_matmul_only(&mut self, n_iters: usize) -> Result<f64, LlmError> {
        use cudarc::driver::{DevicePtr, DevicePtrMut};
        let cfg = &self.config;
        let d = cfg.hidden_size;
        let f = cfg.intermediate_size;
        let kv_dim = cfg.n_kv_heads() * cfg.head_dim();

        // Allocate scratch buffers.
        let mut h_dev = self
            .stream
            .alloc_zeros::<half::bf16>(d)
            .map_err(|e| LlmError::Backend(format!("alloc h: {e:?}")))?;
        let mut q_buf = self
            .stream
            .alloc_zeros::<half::bf16>(d)
            .map_err(|e| LlmError::Backend(format!("alloc q: {e:?}")))?;
        let mut kv_buf = self
            .stream
            .alloc_zeros::<half::bf16>(kv_dim)
            .map_err(|e| LlmError::Backend(format!("alloc kv: {e:?}")))?;
        let mut o_buf = self
            .stream
            .alloc_zeros::<half::bf16>(d)
            .map_err(|e| LlmError::Backend(format!("alloc o: {e:?}")))?;
        let mut gate_buf = self
            .stream
            .alloc_zeros::<half::bf16>(f)
            .map_err(|e| LlmError::Backend(format!("alloc gate: {e:?}")))?;
        let mut up_buf = self
            .stream
            .alloc_zeros::<half::bf16>(f)
            .map_err(|e| LlmError::Backend(format!("alloc up: {e:?}")))?;
        let mut down_buf = self
            .stream
            .alloc_zeros::<half::bf16>(d)
            .map_err(|e| LlmError::Backend(format!("alloc down: {e:?}")))?;

        let (h_p, q_p, kv_p, o_p, gate_p, up_p, down_p) = unsafe {
            (
                h_dev.device_ptr(&self.stream).0,
                q_buf.device_ptr_mut(&self.stream).0,
                kv_buf.device_ptr_mut(&self.stream).0,
                o_buf.device_ptr_mut(&self.stream).0,
                gate_buf.device_ptr_mut(&self.stream).0,
                up_buf.device_ptr_mut(&self.stream).0,
                down_buf.device_ptr_mut(&self.stream).0,
            )
        };

        // Warm-up.
        if let Some(b) = self.blocks.first() {
            b.w_q
                .dispatch_matmul(&self.kernels, &mut self.session, &self.stream, h_p, q_p, 1)?;
            b.w_gate.dispatch_matmul(
                &self.kernels,
                &mut self.session,
                &self.stream,
                h_p,
                gate_p,
                1,
            )?;
        }
        self.stream.synchronize().ok();

        let t0 = std::time::Instant::now();
        for _ in 0..n_iters {
            for b in &self.blocks {
                // Q, K, V projections (Q6_K typically in K_M).
                b.w_q.dispatch_matmul(
                    &self.kernels,
                    &mut self.session,
                    &self.stream,
                    h_p,
                    q_p,
                    1,
                )?;
                b.w_k.dispatch_matmul(
                    &self.kernels,
                    &mut self.session,
                    &self.stream,
                    h_p,
                    kv_p,
                    1,
                )?;
                b.w_v.dispatch_matmul(
                    &self.kernels,
                    &mut self.session,
                    &self.stream,
                    h_p,
                    kv_p,
                    1,
                )?;
                // O projection.
                b.w_o.dispatch_matmul(
                    &self.kernels,
                    &mut self.session,
                    &self.stream,
                    o_p,
                    h_p,
                    1,
                )?;
                // FFN gate + up (Q4_K typically).
                b.w_gate.dispatch_matmul(
                    &self.kernels,
                    &mut self.session,
                    &self.stream,
                    h_p,
                    gate_p,
                    1,
                )?;
                b.w_up.dispatch_matmul(
                    &self.kernels,
                    &mut self.session,
                    &self.stream,
                    h_p,
                    up_p,
                    1,
                )?;
                // FFN down (Q6_K typically).
                b.w_down.dispatch_matmul(
                    &self.kernels,
                    &mut self.session,
                    &self.stream,
                    up_p,
                    down_p,
                    1,
                )?;
            }
        }
        self.stream.synchronize().ok();
        let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0 / n_iters as f64;
        Ok(elapsed_ms)
    }
}
