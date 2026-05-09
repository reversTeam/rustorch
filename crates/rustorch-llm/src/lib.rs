//! LLM-runner core for RusTorch — Llama / Qwen architecture +
//! HuggingFace safetensors checkpoint loader (T57).
//!
//! ## What this crate provides
//!
//! - [`LlamaConfig`] — parses HF `config.json` (Llama, Qwen2.5,
//!   Qwen3 dense). Captures the dimensions, RoPE base, RMSNorm
//!   epsilon, vocab size, and whether QKV uses bias.
//! - [`HfWeights`] — dictionary of `String -> Tensor` parsed from
//!   one or more safetensors shards (loads the
//!   `model.safetensors.index.json` HF sharded layout used by
//!   models > 8 GB).
//! - Naming-mapping helpers ([`HfWeights::expect`],
//!   [`hf_block_key`]) that translate the canonical HuggingFace
//!   weight names (e.g. `model.layers.0.self_attn.q_proj.weight`)
//!   to the slices the decode loop expects.
//!
//! ## What this crate does NOT do (yet)
//!
//! - INT4 / GGUF dequantization (T58 — needed for Qwen 32B since
//!   fp16 weights are 64 GB).
//! - The autoregressive `generate()` loop integration with
//!   `kv_cache + rope + sampling` modules (T59 — the
//!   examples/llama_decode_demo.rs sketches it; making it
//!   reusable across modules is T59).
//!
//! ## Loading a Qwen checkpoint (when those land)
//!
//! ```ignore
//! use rustorch_llm::{LlamaConfig, HfWeights};
//!
//! let cfg = LlamaConfig::from_hf_dir("models/qwen2.5-32b/")?;
//! let weights = HfWeights::from_hf_dir("models/qwen2.5-32b/")?;
//!
//! // Look up specific layer params:
//! let q_proj_layer3 = weights.expect("model.layers.3.self_attn.q_proj.weight")?;
//! // q_proj_layer3 is a Tensor of shape [D, D].
//! ```

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use rustorch_core::tensor::tensor_impl::Tensor;
use serde::Deserialize;

#[cfg(feature = "cuda")]
pub mod cuda_backend;
pub mod gguf_loader;
pub mod model;
pub mod qwen35;
pub mod qwen35_cpu;
pub mod qwen35_cuda;
#[cfg(feature = "cuda")]
pub use cuda_backend::LlamaModelCuda;
pub use gguf_loader::{GgufBlockWeights, GgufWeights};
pub use model::LlamaModel;
pub use qwen35::{
    describe_model, full_inventory, layer_kind_for_index, missing_tensors, parse_config, LayerKind,
    Qwen35Config, Qwen35LoadError, Qwen35Variant, TensorRef,
};

/// Errors raised by this crate.
#[derive(Debug)]
pub enum LlmError {
    /// `config.json` parse / I/O error.
    Config(String),
    /// `safetensors` parse / I/O error.
    Safetensors(String),
    /// `model.safetensors.index.json` parse error.
    Index(String),
    /// A required weight name is missing from the loaded set.
    MissingWeight(String),
    /// Filesystem error.
    Io(String),
    /// CUDA backend error (T241.3 — gated by `cuda` feature).
    Backend(String),
}

impl std::fmt::Display for LlmError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LlmError::Config(s) => write!(f, "config: {s}"),
            LlmError::Safetensors(s) => write!(f, "safetensors: {s}"),
            LlmError::Index(s) => write!(f, "index: {s}"),
            LlmError::MissingWeight(s) => write!(f, "missing weight: {s}"),
            LlmError::Io(s) => write!(f, "io: {s}"),
            LlmError::Backend(s) => write!(f, "backend: {s}"),
        }
    }
}

impl std::error::Error for LlmError {}

// ---------------------------------------------------------------------------
// LlamaConfig — JSON-friendly representation of HF `config.json`.
// ---------------------------------------------------------------------------

/// Llama / Qwen dense architecture configuration.
///
/// Field names match the canonical HuggingFace `config.json`
/// schema for `LlamaForCausalLM` and `Qwen2ForCausalLM` /
/// `Qwen3ForCausalLM` (the dense variants).
///
/// Optional fields default to the Llama-2 conventions when missing
/// from the JSON; concrete checkpoints typically set every value
/// explicitly.
#[derive(Debug, Clone, Deserialize)]
pub struct LlamaConfig {
    /// Hidden state dimension (e.g. 4096 for Llama-7B, 5120 for
    /// Qwen2.5-32B).
    pub hidden_size: usize,
    /// Total number of attention heads.
    pub num_attention_heads: usize,
    /// Number of key/value heads (GQA). Equals `num_attention_heads`
    /// for plain MHA. For Qwen2.5-32B = 8.
    #[serde(default)]
    pub num_key_value_heads: Option<usize>,
    /// FFN intermediate dimension. Llama / Qwen use SwiGLU so this
    /// is the up_proj / gate_proj output dim (the actual FFN width).
    pub intermediate_size: usize,
    /// Number of decoder layers.
    pub num_hidden_layers: usize,
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Maximum sequence length the model was trained on (RoPE
    /// extrapolation beyond this works but degrades quality without
    /// NTK-aware scaling).
    #[serde(default = "default_max_position")]
    pub max_position_embeddings: usize,
    /// RMSNorm epsilon. Llama: 1e-5, Qwen: 1e-6 typically.
    #[serde(default = "default_rms_eps")]
    pub rms_norm_eps: f32,
    /// RoPE base frequency (`theta`). Llama-2: 10_000, Llama-3:
    /// 500_000, Qwen2.5: 1_000_000, Qwen3: 1_000_000.
    #[serde(default = "default_rope_base")]
    pub rope_theta: f32,
    /// Tied input embedding + output (LM head) weights. False for
    /// Llama, true for some smaller models.
    #[serde(default)]
    pub tie_word_embeddings: bool,
}

fn default_max_position() -> usize {
    2048
}
fn default_rms_eps() -> f32 {
    1e-5
}
fn default_rope_base() -> f32 {
    10_000.0
}

impl LlamaConfig {
    /// Parse from a JSON string (typically the contents of
    /// `config.json`).
    pub fn from_json(s: &str) -> Result<Self, LlmError> {
        serde_json::from_str(s).map_err(|e| LlmError::Config(format!("{e}")))
    }

    /// Read `config.json` from a HuggingFace model directory.
    pub fn from_hf_dir<P: AsRef<Path>>(dir: P) -> Result<Self, LlmError> {
        let path = dir.as_ref().join("config.json");
        let s =
            std::fs::read_to_string(&path).map_err(|e| LlmError::Io(format!("{path:?}: {e}")))?;
        Self::from_json(&s)
    }

    /// Number of K/V heads — defaults to `num_attention_heads` if
    /// the field is missing (plain MHA).
    pub fn n_kv_heads(&self) -> usize {
        self.num_key_value_heads.unwrap_or(self.num_attention_heads)
    }

    /// Per-head feature dim.
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    /// GQA group size (`n_heads / n_kv_heads`).
    pub fn group_size(&self) -> usize {
        self.num_attention_heads / self.n_kv_heads()
    }
}

// ---------------------------------------------------------------------------
// HfWeights — sharded safetensors loader.
// ---------------------------------------------------------------------------

/// A flat dictionary of `name → Tensor` loaded from one or more
/// safetensors shards. The ordering follows the HuggingFace
/// `tensor_to_filename` mapping when an index file is present, or
/// the natural directory order otherwise.
pub struct HfWeights {
    tensors: BTreeMap<String, Tensor>,
}

impl HfWeights {
    /// Load all weights from a HuggingFace model directory.
    ///
    /// If `model.safetensors.index.json` is present, every shard
    /// listed in it is parsed and merged. Otherwise, every
    /// `*.safetensors` file in the directory is loaded in
    /// alphabetical order.
    pub fn from_hf_dir<P: AsRef<Path>>(dir: P) -> Result<Self, LlmError> {
        let dir = dir.as_ref();
        let index_path = dir.join("model.safetensors.index.json");
        let shard_paths: Vec<PathBuf> = if index_path.exists() {
            // Sharded layout: parse the index to find every shard
            // referenced.
            let s = std::fs::read_to_string(&index_path)
                .map_err(|e| LlmError::Io(format!("{index_path:?}: {e}")))?;
            let parsed: HfShardIndex =
                serde_json::from_str(&s).map_err(|e| LlmError::Index(format!("{e}")))?;
            let mut uniq: Vec<String> = parsed.weight_map.values().cloned().collect::<Vec<_>>();
            uniq.sort();
            uniq.dedup();
            uniq.into_iter().map(|f| dir.join(f)).collect()
        } else {
            // Flat layout: load every *.safetensors file.
            let mut paths: Vec<PathBuf> = std::fs::read_dir(dir)
                .map_err(|e| LlmError::Io(format!("{dir:?}: {e}")))?
                .filter_map(|entry| entry.ok())
                .map(|entry| entry.path())
                .filter(|p| p.extension().map(|e| e == "safetensors").unwrap_or(false))
                .collect();
            paths.sort();
            if paths.is_empty() {
                return Err(LlmError::Io(format!("no safetensors shards in {dir:?}")));
            }
            paths
        };

        let mut tensors: BTreeMap<String, Tensor> = BTreeMap::new();
        for shard in shard_paths {
            let parsed = rustorch_serde::read_path(&shard)
                .map_err(|e| LlmError::Safetensors(format!("{shard:?}: {e:?}")))?;
            for (k, v) in parsed {
                tensors.insert(k, v);
            }
        }
        Ok(HfWeights { tensors })
    }

    /// Build directly from an in-memory tensor map (used by tests
    /// and by callers that want to inject pre-loaded data).
    pub fn from_map(tensors: BTreeMap<String, Tensor>) -> Self {
        HfWeights { tensors }
    }

    /// Look up a weight tensor by its full HuggingFace name. Returns
    /// `Err(MissingWeight)` if absent.
    pub fn expect(&self, name: &str) -> Result<&Tensor, LlmError> {
        self.tensors
            .get(name)
            .ok_or_else(|| LlmError::MissingWeight(name.to_string()))
    }

    /// Optional access — `None` if the weight is absent.
    pub fn get(&self, name: &str) -> Option<&Tensor> {
        self.tensors.get(name)
    }

    /// Iterate over `(name, tensor)` pairs.
    pub fn iter(&self) -> impl Iterator<Item = (&String, &Tensor)> {
        self.tensors.iter()
    }

    /// Number of tensors loaded.
    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    /// Whether the weight set is empty.
    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }
}

#[derive(Debug, Deserialize)]
struct HfShardIndex {
    weight_map: BTreeMap<String, String>,
}

// ---------------------------------------------------------------------------
// HF naming helpers.
// ---------------------------------------------------------------------------

/// Build the canonical HF weight name for a per-block parameter.
///
/// Examples:
///
/// ```
/// use rustorch_llm::hf_block_key;
/// assert_eq!(hf_block_key(3, "self_attn.q_proj.weight"),
///            "model.layers.3.self_attn.q_proj.weight");
/// ```
pub fn hf_block_key(layer_idx: usize, suffix: &str) -> String {
    format!("model.layers.{layer_idx}.{suffix}")
}

/// Canonical HF top-level weight names.
pub mod hf_keys {
    /// Token embedding lookup table.
    pub const EMBED_TOKENS: &str = "model.embed_tokens.weight";
    /// Final RMSNorm gamma applied just before the LM head.
    pub const FINAL_NORM: &str = "model.norm.weight";
    /// LM head projection weight.
    pub const LM_HEAD: &str = "lm_head.weight";
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_qwen_config_json() -> &'static str {
        // Qwen2.5-32B canonical config (subset of the real fields,
        // sufficient to exercise the parser).
        r#"{
            "hidden_size": 5120,
            "num_attention_heads": 40,
            "num_key_value_heads": 8,
            "intermediate_size": 27648,
            "num_hidden_layers": 64,
            "vocab_size": 152064,
            "max_position_embeddings": 32768,
            "rms_norm_eps": 1e-06,
            "rope_theta": 1000000.0,
            "tie_word_embeddings": false
        }"#
    }

    #[test]
    fn parses_qwen_config_json() {
        let cfg = LlamaConfig::from_json(sample_qwen_config_json()).unwrap();
        assert_eq!(cfg.hidden_size, 5120);
        assert_eq!(cfg.num_attention_heads, 40);
        assert_eq!(cfg.n_kv_heads(), 8);
        assert_eq!(cfg.head_dim(), 128);
        assert_eq!(cfg.group_size(), 5);
        assert_eq!(cfg.intermediate_size, 27648);
        assert_eq!(cfg.vocab_size, 152064);
        assert!((cfg.rope_theta - 1_000_000.0).abs() < 1.0);
        assert!((cfg.rms_norm_eps - 1e-6).abs() < 1e-9);
    }

    #[test]
    fn config_defaults_when_optional_fields_missing() {
        let json = r#"{
            "hidden_size": 1024,
            "num_attention_heads": 16,
            "intermediate_size": 4096,
            "num_hidden_layers": 12,
            "vocab_size": 32000
        }"#;
        let cfg = LlamaConfig::from_json(json).unwrap();
        assert_eq!(cfg.n_kv_heads(), 16); // default = num_attention_heads
        assert_eq!(cfg.max_position_embeddings, 2048);
        assert!((cfg.rope_theta - 10_000.0).abs() < 1.0);
        assert!(!cfg.tie_word_embeddings);
    }

    #[test]
    fn hf_block_key_format() {
        assert_eq!(
            hf_block_key(3, "self_attn.q_proj.weight"),
            "model.layers.3.self_attn.q_proj.weight"
        );
        assert_eq!(
            hf_block_key(0, "input_layernorm.weight"),
            "model.layers.0.input_layernorm.weight"
        );
    }

    #[test]
    fn hf_weights_expect_missing_returns_err() {
        let weights = HfWeights::from_map(BTreeMap::new());
        let err = weights.expect("model.embed_tokens.weight").unwrap_err();
        assert!(matches!(err, LlmError::MissingWeight(_)));
    }

    #[test]
    fn hf_weights_round_trip_via_in_memory_map() {
        let mut map: BTreeMap<String, Tensor> = BTreeMap::new();
        let t = Tensor::from_vec(vec![2usize, 3], vec![1.0_f32; 6]).unwrap();
        map.insert("model.embed_tokens.weight".to_string(), t);
        let weights = HfWeights::from_map(map);
        assert_eq!(weights.len(), 1);
        let got = weights.expect("model.embed_tokens.weight").unwrap();
        assert_eq!(got.shape(), &[2, 3]);
    }
}
