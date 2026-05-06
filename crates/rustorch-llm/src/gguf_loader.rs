//! GGUF backend for [`LlamaConfig`] / [`LlamaModel`] (T61).
//!
//! Reads a llama.cpp GGUF file (Q4_K_M / Q6_K), dequantizes every
//! weight tensor to f32, and feeds the result into the same internal
//! [`BlockWeights`] layout used by [`crate::LlamaModel::from_hf`].
//!
//! ## GGUF tensor naming
//!
//! GGUF uses Llama.cpp's flat naming scheme:
//!
//! | role                       | name                                  |
//! |----------------------------|---------------------------------------|
//! | token embedding lookup     | `token_embd.weight`                   |
//! | LM head (output projection)| `output.weight` (absent if tied)      |
//! | final RMSNorm              | `output_norm.weight`                  |
//! | per-block input RMSNorm    | `blk.{i}.attn_norm.weight`            |
//! | per-block Q proj           | `blk.{i}.attn_q.weight`               |
//! | per-block K proj           | `blk.{i}.attn_k.weight`               |
//! | per-block V proj           | `blk.{i}.attn_v.weight`               |
//! | per-block O proj           | `blk.{i}.attn_output.weight`          |
//! | per-block Q-norm (Qwen3)   | `blk.{i}.attn_q_norm.weight`          |
//! | per-block K-norm (Qwen3)   | `blk.{i}.attn_k_norm.weight`          |
//! | per-block FFN RMSNorm      | `blk.{i}.ffn_norm.weight`             |
//! | per-block SwiGLU gate      | `blk.{i}.ffn_gate.weight`             |
//! | per-block SwiGLU up        | `blk.{i}.ffn_up.weight`               |
//! | per-block SwiGLU down      | `blk.{i}.ffn_down.weight`             |
//!
//! ## GGUF shape convention
//!
//! GGML stores `tensor[ne0, ne1]` with `ne0` as the inner / most
//! contiguous dim. So a GGUF tensor reported as `[5120, 1024]` is
//! laid out in memory the same as a numpy `[1024, 5120]` row-major
//! array — i.e. exactly the HF convention `[out, in]`. Our sgemv
//! kernel wants `[in, out]`, so we apply the same transpose as
//! [`crate::LlamaModel::from_hf`].
//!
//! ## Memory budget
//!
//! Q4_K_M weights expand ~8× when dequantized to f32 (4-bit → 32-bit
//! plus a tiny amount of dtype-tag overhead). A 14 B Q4_K_M model at
//! 8.4 GiB on disk becomes ~30 GiB of f32 RAM. For now we materialize
//! every tensor up front; an on-the-fly dequant-then-matmul path is
//! a follow-up task.

use std::path::Path;

use rustorch_gguf::{dequant_to_f32, GgmlType, GgufError, GgufFile, TensorInfo};

use crate::{LlamaConfig, LlmError};

impl From<GgufError> for LlmError {
    fn from(e: GgufError) -> Self {
        LlmError::Io(format!("gguf: {e}"))
    }
}

/// All weights extracted from a GGUF file, dequantized to f32 and
/// organized as ready-to-feed buffers for the existing `BlockWeights`
/// builder in `model.rs`.
///
/// Linear weight buffers are stored in the **HF / numpy convention**
/// (`[out, in]` row-major) — i.e. with shape dims swapped relative to
/// the GGUF header. The `LlamaModel::from_gguf` constructor applies
/// the same `transpose_2d` as the HF path to convert into the
/// sgemv-friendly `[in, out]` layout.
pub struct GgufWeights {
    /// `[V, D]` row-major (numpy convention) — direct embedding lookup.
    pub token_emb: Vec<f32>,
    /// `[D]`
    pub final_norm: Vec<f32>,
    /// `[V, D]` row-major — direct from `output.weight` if present, or
    /// aliased from `token_emb` when `tie_word_embeddings == true`.
    pub lm_head_t: Vec<f32>,
    /// True if the file omitted `output.weight` and we reused the
    /// embedding table.
    pub tied_lm_head: bool,
    /// Per-layer weights.
    pub blocks: Vec<GgufBlockWeights>,
}

/// All per-block weights for one decoder layer, dequantized to f32 and
/// in HF (`[out, in]` row-major) layout.
pub struct GgufBlockWeights {
    pub attn_norm: Vec<f32>, // [D]
    pub w_q: Vec<f32>,       // [D, D]   (out=D, in=D)
    pub w_k: Vec<f32>,       // [KV, D]
    pub w_v: Vec<f32>,       // [KV, D]
    pub w_o: Vec<f32>,       // [D, D]
    pub ffn_norm: Vec<f32>,  // [D]
    pub w_gate: Vec<f32>,    // [F, D]
    pub w_up: Vec<f32>,      // [F, D]
    pub w_down: Vec<f32>,    // [D, F]
    /// Optional Qwen3 per-head Q-norm — shape `[head_dim]`.
    pub attn_q_norm: Option<Vec<f32>>,
    /// Optional Qwen3 per-head K-norm — shape `[head_dim]`.
    pub attn_k_norm: Option<Vec<f32>>,
}

impl LlamaConfig {
    /// Build a [`LlamaConfig`] from a GGUF file's metadata. Supports
    /// the `qwen2`, `qwen3`, `llama` and similar dense architectures
    /// that use the canonical key set
    /// (`<arch>.embedding_length`, `<arch>.attention.head_count`,
    /// `<arch>.attention.head_count_kv`, `<arch>.feed_forward_length`,
    /// `<arch>.block_count`, `<arch>.context_length`,
    /// `<arch>.attention.layer_norm_rms_epsilon`,
    /// `<arch>.rope.freq_base`).
    ///
    /// `vocab_size` is inferred from the `token_embd.weight` tensor
    /// shape — GGUF stores it as `[D, V]` so we read the second dim.
    pub fn from_gguf(file: &GgufFile) -> Result<Self, LlmError> {
        let meta = file.metadata();
        let arch = meta.architecture().ok_or_else(|| {
            LlmError::Config("missing general.architecture metadata key".to_string())
        })?;

        let need_u32 = |suffix: &str| -> Result<usize, LlmError> {
            meta.get_arch(suffix)
                .and_then(|v| v.as_u32())
                .map(|v| v as usize)
                .ok_or_else(|| LlmError::Config(format!("missing {arch}.{suffix}")))
        };

        let hidden_size = need_u32("embedding_length")?;
        let num_attention_heads = need_u32("attention.head_count")?;
        let num_key_value_heads = meta
            .get_arch("attention.head_count_kv")
            .and_then(|v| v.as_u32())
            .map(|v| v as usize);
        let intermediate_size = need_u32("feed_forward_length")?;
        let num_hidden_layers = need_u32("block_count")?;

        let max_position_embeddings = meta
            .get_arch("context_length")
            .and_then(|v| v.as_u32())
            .map(|v| v as usize)
            .unwrap_or(2048);

        let rms_norm_eps = meta
            .get_arch("attention.layer_norm_rms_epsilon")
            .and_then(|v| v.as_f32())
            .unwrap_or(1e-5);

        let rope_theta = meta
            .get_arch("rope.freq_base")
            .and_then(|v| v.as_f32())
            .unwrap_or(10_000.0);

        // Vocab size from token_embd.weight — GGUF shape `[D, V]`.
        let token_embd = file.tensor("token_embd.weight").ok_or_else(|| {
            LlmError::MissingWeight("token_embd.weight (required to infer vocab_size)".to_string())
        })?;
        if token_embd.shape.len() != 2 {
            return Err(LlmError::Config(format!(
                "token_embd.weight: expected 2 dims, got {:?}",
                token_embd.shape
            )));
        }
        let (gguf_d, gguf_v) = (token_embd.shape[0] as usize, token_embd.shape[1] as usize);
        if gguf_d != hidden_size {
            return Err(LlmError::Config(format!(
                "token_embd.weight inner dim {} != {arch}.embedding_length {hidden_size}",
                gguf_d
            )));
        }
        let vocab_size = gguf_v;

        // GGUF: tied LM head ⇔ no `output.weight` tensor.
        let tie_word_embeddings = file.tensor("output.weight").is_none();

        Ok(LlamaConfig {
            hidden_size,
            num_attention_heads,
            num_key_value_heads,
            intermediate_size,
            num_hidden_layers,
            vocab_size,
            max_position_embeddings,
            rms_norm_eps,
            rope_theta,
            tie_word_embeddings,
        })
    }
}

impl GgufWeights {
    /// Open a GGUF file, parse it, and dequantize every required
    /// tensor to f32.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<(LlamaConfig, Self), LlmError> {
        let file = GgufFile::open(path).map_err(LlmError::from)?;
        let cfg = LlamaConfig::from_gguf(&file)?;
        let weights = Self::from_file(&file, &cfg)?;
        Ok((cfg, weights))
    }

    /// Build from a parsed [`GgufFile`] and an already-extracted
    /// [`LlamaConfig`].
    pub fn from_file(file: &GgufFile, cfg: &LlamaConfig) -> Result<Self, LlmError> {
        // Top-level tensors.
        let token_emb_t = file
            .tensor("token_embd.weight")
            .ok_or_else(|| LlmError::MissingWeight("token_embd.weight".to_string()))?;
        let token_emb = dequant_named(file, token_emb_t, "token_embd.weight")?;
        if token_emb.len() != cfg.vocab_size * cfg.hidden_size {
            return Err(LlmError::Config(format!(
                "token_embd: got {} f32, expected V*D = {}",
                token_emb.len(),
                cfg.vocab_size * cfg.hidden_size
            )));
        }

        let final_norm_t = file
            .tensor("output_norm.weight")
            .ok_or_else(|| LlmError::MissingWeight("output_norm.weight".to_string()))?;
        let final_norm = dequant_named(file, final_norm_t, "output_norm.weight")?;

        let (lm_head_t, tied_lm_head) = if let Some(t) = file.tensor("output.weight") {
            (dequant_named(file, t, "output.weight")?, false)
        } else {
            // Tied embedding — reuse token_emb. Same `[V, D]` layout.
            (token_emb.clone(), true)
        };

        // Per-block weights.
        let mut blocks: Vec<GgufBlockWeights> = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            blocks.push(load_block(file, cfg, i)?);
        }

        Ok(GgufWeights {
            token_emb,
            final_norm,
            lm_head_t,
            tied_lm_head,
            blocks,
        })
    }
}

fn load_block(file: &GgufFile, _cfg: &LlamaConfig, i: usize) -> Result<GgufBlockWeights, LlmError> {
    let key = |suffix: &str| -> String { format!("blk.{i}.{suffix}") };

    let need = |suffix: &str| -> Result<Vec<f32>, LlmError> {
        let k = key(suffix);
        let t = file
            .tensor(&k)
            .ok_or_else(|| LlmError::MissingWeight(k.clone()))?;
        dequant_named(file, t, &k)
    };
    let opt = |suffix: &str| -> Result<Option<Vec<f32>>, LlmError> {
        let k = key(suffix);
        match file.tensor(&k) {
            Some(t) => Ok(Some(dequant_named(file, t, &k)?)),
            None => Ok(None),
        }
    };

    let attn_norm = need("attn_norm.weight")?;
    let w_q = need("attn_q.weight")?;
    let w_k = need("attn_k.weight")?;
    let w_v = need("attn_v.weight")?;
    let w_o = need("attn_output.weight")?;
    let ffn_norm = need("ffn_norm.weight")?;
    let w_gate = need("ffn_gate.weight")?;
    let w_up = need("ffn_up.weight")?;
    let w_down = need("ffn_down.weight")?;
    let attn_q_norm = opt("attn_q_norm.weight")?;
    let attn_k_norm = opt("attn_k_norm.weight")?;

    Ok(GgufBlockWeights {
        attn_norm,
        w_q,
        w_k,
        w_v,
        w_o,
        ffn_norm,
        w_gate,
        w_up,
        w_down,
        attn_q_norm,
        attn_k_norm,
    })
}

/// Dequantize a tensor by name and produce a `Vec<f32>` of length
/// `n_elements`, with friendly error messages.
fn dequant_named(file: &GgufFile, t: &TensorInfo, name: &str) -> Result<Vec<f32>, LlmError> {
    if !t.dtype.is_dequant_supported() {
        return Err(LlmError::Io(format!(
            "tensor {name}: dtype {:?} not supported by rustorch-gguf yet",
            t.dtype
        )));
    }
    let bytes = file.tensor_bytes(t);
    dequant_to_f32(t, bytes).map_err(|e| LlmError::Io(format!("dequant {name}: {e}")))
}

/// True if a tensor's dtype is one we can load.
pub fn is_loadable(dtype: GgmlType) -> bool {
    dtype.is_dequant_supported()
}

#[cfg(test)]
mod tests {
    use super::*;
    // Real-file integration tests live in `examples/qwen_inference.rs` and
    // are gated by the model files being present locally.

    #[test]
    fn missing_required_metadata_errors() {
        // Build an in-memory GGUF with no `qwen3.*` keys and verify
        // we get a clean Config error.
        let mut buf: Vec<u8> = Vec::new();
        buf.extend_from_slice(&rustorch_gguf::GGUF_MAGIC.to_le_bytes());
        buf.extend_from_slice(&3u32.to_le_bytes()); // version
        buf.extend_from_slice(&0u64.to_le_bytes()); // tensor_count
        buf.extend_from_slice(&1u64.to_le_bytes()); // metadata_count

        let key = b"general.architecture";
        buf.extend_from_slice(&(key.len() as u64).to_le_bytes());
        buf.extend_from_slice(key);
        buf.extend_from_slice(&8u32.to_le_bytes()); // String type
        let val = b"qwen3";
        buf.extend_from_slice(&(val.len() as u64).to_le_bytes());
        buf.extend_from_slice(val);

        // Pad to default alignment.
        while buf.len() % 32 != 0 {
            buf.push(0);
        }

        let file = GgufFile::from_bytes(buf).expect("parse minimal");
        let err = LlamaConfig::from_gguf(&file).unwrap_err();
        match err {
            LlmError::Config(msg) => {
                assert!(
                    msg.contains("embedding_length"),
                    "expected embedding_length error, got: {msg}"
                );
            },
            other => panic!("expected Config error, got: {other:?}"),
        }
    }
}
