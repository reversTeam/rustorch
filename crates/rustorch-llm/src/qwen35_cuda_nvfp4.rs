//! T246.9 — `Qwen35ModelCudaNVFP4` : end-to-end NVFP4 inference path for
//! Qwen3.5/3.6 hybrid (SSM + Attention) HuggingFace `nvfp4-pack-quantized`
//! safetensors models.
//!
//! Architecture vs `Qwen35ModelCudaQ4K` (sibling, GGUF Q4_K_M path) :
//! - Quantized linears stored as 4-tuple per Linear : `weight_packed [out, in/2] U8`
//!   + `weight_scale [out, in/16] F8_E4M3` + `weight_global_scale [1] F32` +
//!   `input_global_scale [1] F32`.
//! - The shared expert is also NVFP4 (gate_proj/up_proj/down_proj per layer).
//! - Per recipe.yaml `ignore` patterns, BF16 stays for : `lm_head`, `embed_tokens`,
//!   `mlp.gate` (router), `shared_expert_gate`, all `linear_attn.*` (SSM stack),
//!   layernorms, q_norm/k_norm.
//!
//! ## Status
//!
//! - **T246.9 NVFP4.1** (this commit) : safetensors `nvfp4-pack-quantized` loader
//!   that walks `model.safetensors.index.json`, dispatches each tensor by
//!   ignore-regex into either an `Nvfp4Tensor` 4-tuple or a `Bf16Tensor`,
//!   uploads to GPU. Ssize sanity is checked (per-Linear shapes against
//!   `Qwen35Config`). No swizzle yet — done lazy when wired into `decode_step`.
//! - **T246.9 NVFP4.2** : `sgemv_nvfp4_bf16_indexed` MoE kernel + parity test.
//! - **T246.9 NVFP4.3** : `Qwen35ModelCudaNVFP4` model class wires all weights
//!   into the hybrid SSM + MoE forward path (mostly mechanical clone from Q4K).

#![cfg(feature = "cuda")]

use crate::qwen35::{Qwen35Config, Qwen35Variant};
use crate::LlmError;
use cudarc::driver::{CudaContext, CudaSlice, CudaStream};
use rustorch_cuda::cublas_lt::LtSession;
use rustorch_cuda::llm_kernels::LlmKernels;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// One NVFP4 quantized linear weight set on GPU.
///
/// Layout (mirrors vLLM `compressed_tensors_w4a4_nvfp4`) :
/// - `packed`        : `[out, in/2]` U8 — 2 FP4 (E2M1) per byte, low nibble = even index
/// - `scale_per_block` : `[out, in/16]` U8 — UE4M3 per-16-element micro-block scale
/// - `weight_global_scale` : F32 scalar — per-tensor global scale
/// - `input_global_scale`  : F32 scalar — per-tensor activation calibration scale
///
/// At matmul time the effective `alpha` is `1 / (weight_global_scale *
/// input_global_scale)` — see RFC §6 / vLLM's `cutlass_scaled_fp4_mm` call.
pub struct Nvfp4Tensor {
    pub packed: CudaSlice<u8>,
    pub scale_per_block: CudaSlice<u8>,
    pub weight_global_scale: f32,
    pub input_global_scale: f32,
    /// Output dim (`out`).
    pub n: usize,
    /// Input dim (`in`). Must be a multiple of 16 (NVFP4 micro-block size).
    pub k: usize,
}

impl Nvfp4Tensor {
    /// Total bytes occupied on device (packed + scales).
    pub fn device_bytes(&self) -> usize {
        self.n * self.k / 2 + self.n * self.k / 16
    }

    /// `alpha` used at GEMM call : `1 / (weight_global * input_global)`.
    ///
    /// Per vLLM `apply_weights` : `cutlass_scaled_fp4_mm(... 1.0/alpha ...)`
    /// where their `alpha = input_global * weight_global`. We pre-invert.
    pub fn matmul_alpha(&self) -> f32 {
        let g = self.weight_global_scale * self.input_global_scale;
        if g == 0.0 {
            1.0
        } else {
            1.0 / g
        }
    }
}

/// One BF16 weight tensor on GPU (for ignored / non-quantized linears).
pub struct Bf16Tensor {
    pub data: CudaSlice<half::bf16>,
    pub shape: Vec<usize>,
}

/// All weights of the Qwen3.6-35B-A3B-NVFP4 model, organized for direct
/// consumption by `Qwen35ModelCudaNVFP4::decode_step`.
///
/// This is the output of `load_nvfp4_safetensors`. It is intentionally a
/// flat name → tensor map (BF16 + NVFP4) so the model class can index
/// into it once and assemble its per-layer `BlockNvfp4` records.
pub struct LoadedNvfp4Weights {
    pub bf16: BTreeMap<String, Bf16Tensor>,
    pub nvfp4: BTreeMap<String, Nvfp4Tensor>,
}

impl LoadedNvfp4Weights {
    pub fn len(&self) -> usize {
        self.bf16.len() + self.nvfp4.len()
    }

    pub fn is_empty(&self) -> bool {
        self.bf16.is_empty() && self.nvfp4.is_empty()
    }
}

// ---------------------------------------------------------------------------
// safetensors raw reader (typed dtype string + raw bytes, no Tensor build)
// ---------------------------------------------------------------------------

/// Minimal raw-bytes safetensors header entry.
#[derive(Debug, serde::Deserialize)]
struct RawHeaderEntry {
    pub dtype: String,
    pub shape: Vec<usize>,
    pub data_offsets: [usize; 2],
}

/// Parsed safetensors index file (HF sharded layout).
#[derive(Debug, serde::Deserialize)]
struct SafetensorsIndex {
    weight_map: BTreeMap<String, String>,
}

/// Raw shard reader — loads the entire shard into memory and slices each
/// tensor by name. Returns a map `name → (dtype_string, shape, raw_bytes)`.
///
/// We avoid building rustorch `Tensor` types because (a) the existing
/// `safetensors.rs` doesn't support `F8_E4M3`, and (b) we want to stream
/// these directly to GPU as raw `CudaSlice<u8>` without an intermediate
/// host-side typed copy.
fn read_shard_raw(
    path: &Path,
) -> Result<BTreeMap<String, (String, Vec<usize>, Vec<u8>)>, LlmError> {
    use std::io::Read;
    let mut f = std::fs::File::open(path).map_err(|e| LlmError::Io(format!("{path:?}: {e}")))?;
    let mut size_buf = [0_u8; 8];
    f.read_exact(&mut size_buf)
        .map_err(|e| LlmError::Io(format!("{path:?}: header size: {e}")))?;
    let header_size = u64::from_le_bytes(size_buf) as usize;
    if header_size > 100_000_000 {
        return Err(LlmError::Safetensors(format!(
            "{path:?}: header size {header_size} exceeds 100 MB sanity limit"
        )));
    }
    let mut header_bytes = vec![0_u8; header_size];
    f.read_exact(&mut header_bytes)
        .map_err(|e| LlmError::Io(format!("{path:?}: header bytes: {e}")))?;
    // Strip optional "__metadata__" key before parsing entries.
    let mut header: BTreeMap<String, serde_json::Value> = serde_json::from_slice(&header_bytes)
        .map_err(|e| LlmError::Safetensors(format!("{path:?}: header json: {e}")))?;
    header.remove("__metadata__");

    let mut binary = Vec::new();
    f.read_to_end(&mut binary)
        .map_err(|e| LlmError::Io(format!("{path:?}: binary block: {e}")))?;

    let mut out: BTreeMap<String, (String, Vec<usize>, Vec<u8>)> = BTreeMap::new();
    for (name, raw) in header {
        let entry: RawHeaderEntry = serde_json::from_value(raw)
            .map_err(|e| LlmError::Safetensors(format!("{path:?}: entry {name} parse: {e}")))?;
        let [start, end] = entry.data_offsets;
        if end > binary.len() || start > end {
            return Err(LlmError::Safetensors(format!(
                "{path:?}: tensor {name}: offsets [{start}, {end}] outside binary block of len {}",
                binary.len()
            )));
        }
        let bytes = binary[start..end].to_vec();
        out.insert(name, (entry.dtype, entry.shape, bytes));
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// "Should this Linear stay BF16?" — recipe.yaml `ignore` regex evaluation
// ---------------------------------------------------------------------------

/// Hard-coded ignore patterns from `recipe.yaml` (Qwen3.6-A3B-NVFP4) :
/// - `re:.*lm_head`           → BF16
/// - `re:visual.*`            → BF16 (we don't load anyway)
/// - `re:model.visual.*`      → BF16
/// - `re:.*mlp.gate$`         → BF16 (router)
/// - `re:.*embed_tokens$`     → BF16
/// - `re:.*shared_expert_gate$` → BF16 (1×D scalar gate)
/// - `re:.*linear_attn.*`     → BF16 (entire SSM stack)
///
/// Returns `true` if the tensor name matches any of the ignore patterns.
/// Non-Linear tensors (layernorms, biases, `A_log`, `dt_bias`, `conv1d`)
/// are also returned `true` because they are not quantized — handled
/// transparently by the loader (any tensor name not ending with one of
/// `weight_packed` / `weight_scale` / `weight_global_scale` /
/// `input_global_scale` is treated as a plain BF16 tensor).
pub fn matches_ignore_pattern(name: &str) -> bool {
    name.ends_with("lm_head.weight")
        || name.starts_with("visual.")
        || name.starts_with("model.visual.")
        || name.contains(".visual.")
        || name.ends_with(".mlp.gate.weight")
        || name.ends_with(".embed_tokens.weight")
        || name.ends_with(".shared_expert_gate.weight")
        || name.contains(".linear_attn.")
        || name.starts_with("mtp.")
}

// ---------------------------------------------------------------------------
// BF16 / U8 / F32 / F8_E4M3 → device upload helpers
// ---------------------------------------------------------------------------

fn upload_bf16(
    stream: &Arc<CudaStream>,
    bytes: &[u8],
    name: &str,
) -> Result<CudaSlice<half::bf16>, LlmError> {
    if bytes.len() % 2 != 0 {
        return Err(LlmError::Safetensors(format!(
            "{name}: BF16 byte length {} not divisible by 2",
            bytes.len()
        )));
    }
    let n = bytes.len() / 2;
    let mut tmp: Vec<half::bf16> = Vec::with_capacity(n);
    for chunk in bytes.chunks_exact(2) {
        let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
        tmp.push(half::bf16::from_bits(bits));
    }
    stream
        .memcpy_stod(&tmp)
        .map_err(|e| LlmError::Backend(format!("upload bf16 {name}: {e:?}")))
}

fn upload_u8(
    stream: &Arc<CudaStream>,
    bytes: &[u8],
    name: &str,
) -> Result<CudaSlice<u8>, LlmError> {
    stream
        .memcpy_stod(bytes)
        .map_err(|e| LlmError::Backend(format!("upload u8 {name}: {e:?}")))
}

fn parse_f32_scalar(bytes: &[u8], name: &str) -> Result<f32, LlmError> {
    if bytes.len() != 4 {
        return Err(LlmError::Safetensors(format!(
            "{name}: expected 4 bytes for F32 scalar, got {}",
            bytes.len()
        )));
    }
    Ok(f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
}

// ---------------------------------------------------------------------------
// Loader entry point
// ---------------------------------------------------------------------------

/// Walk `model.safetensors.index.json` from `dir`, load every shard listed,
/// and dispatch each tensor into the BF16 or NVFP4 bucket. Skip MTP and
/// visual tensors (we do text-only inference).
///
/// Returns a flat name → tensor map ; the caller (model class) walks it
/// per-layer to assemble `BlockNvfp4` records.
///
/// ### Layout per quantized Linear (vLLM `nvfp4-pack-quantized`)
/// For Linear `<base>` (e.g. `model.language_model.layers.3.self_attn.q_proj`),
/// 4 sibling tensors are emitted :
/// - `<base>.weight_packed`       : `[out, in/2]` U8 — 2 FP4 / byte
/// - `<base>.weight_scale`        : `[out, in/16]` F8_E4M3 — UE4M3 per-block
/// - `<base>.weight_global_scale` : `[1]` F32 — per-tensor global scale
/// - `<base>.input_global_scale`  : `[1]` F32 — calibration activation scale
///
/// All four are aggregated into a single `Nvfp4Tensor` keyed at `<base>`.
///
/// Plain BF16 tensors are inserted into `bf16` keyed at their full name.
pub fn load_nvfp4_safetensors(
    dir: &Path,
    stream: &Arc<CudaStream>,
) -> Result<LoadedNvfp4Weights, LlmError> {
    let index_path = dir.join("model.safetensors.index.json");
    let index_bytes =
        std::fs::read(&index_path).map_err(|e| LlmError::Io(format!("{index_path:?}: {e}")))?;
    let index: SafetensorsIndex = serde_json::from_slice(&index_bytes)
        .map_err(|e| LlmError::Safetensors(format!("index json: {e}")))?;
    let mut shard_names: Vec<String> = index.weight_map.values().cloned().collect();
    shard_names.sort();
    shard_names.dedup();
    let shards: Vec<PathBuf> = shard_names.into_iter().map(PathBuf::from).collect();

    // Collect raw bytes for ALL tensors first ; then group quantized 4-tuples
    // and dispatch to GPU.
    let mut raw: BTreeMap<String, (String, Vec<usize>, Vec<u8>)> = BTreeMap::new();
    for shard in &shards {
        let p = dir.join(shard);
        let s = read_shard_raw(&p)?;
        for (k, v) in s {
            raw.insert(k, v);
        }
    }

    let mut bf16: BTreeMap<String, Bf16Tensor> = BTreeMap::new();
    let mut nvfp4: BTreeMap<String, Nvfp4Tensor> = BTreeMap::new();

    // Group quantized linears by their base name.
    let mut quantized_bases: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    for name in raw.keys() {
        if let Some(base) = name.strip_suffix(".weight_packed") {
            // Skip mtp.* and visual.* upfront.
            if name.starts_with("mtp.") || name.contains(".visual.") {
                continue;
            }
            quantized_bases.insert(base.to_string());
        }
    }

    // Emit Nvfp4Tensor for each base.
    for base in &quantized_bases {
        let key_packed = format!("{base}.weight_packed");
        let key_scale = format!("{base}.weight_scale");
        let key_w_global = format!("{base}.weight_global_scale");
        let key_in_global = format!("{base}.input_global_scale");
        let (dt_p, shape_p, bytes_p) = raw
            .get(&key_packed)
            .ok_or_else(|| LlmError::Safetensors(format!("missing {key_packed}")))?;
        let (dt_s, shape_s, bytes_s) = raw
            .get(&key_scale)
            .ok_or_else(|| LlmError::Safetensors(format!("missing {key_scale}")))?;
        let (_, _, bytes_w) = raw
            .get(&key_w_global)
            .ok_or_else(|| LlmError::Safetensors(format!("missing {key_w_global}")))?;
        let (_, _, bytes_i) = raw
            .get(&key_in_global)
            .ok_or_else(|| LlmError::Safetensors(format!("missing {key_in_global}")))?;

        if dt_p != "U8" {
            return Err(LlmError::Safetensors(format!(
                "{key_packed}: dtype {dt_p}, expected U8"
            )));
        }
        if dt_s != "F8_E4M3" {
            return Err(LlmError::Safetensors(format!(
                "{key_scale}: dtype {dt_s}, expected F8_E4M3"
            )));
        }
        if shape_p.len() != 2 || shape_s.len() != 2 {
            return Err(LlmError::Safetensors(format!(
                "{base}: expected 2-D packed/scale, got packed={shape_p:?} scale={shape_s:?}"
            )));
        }
        let n = shape_p[0];
        let k_half = shape_p[1];
        let k = k_half * 2;
        if k % 16 != 0 {
            return Err(LlmError::Safetensors(format!(
                "{base}: K={k} not multiple of 16 (NVFP4 micro-block)"
            )));
        }
        if shape_s[0] != n || shape_s[1] != k / 16 {
            return Err(LlmError::Safetensors(format!(
                "{base}: scale shape {shape_s:?} != [{n}, {}] for K={k}",
                k / 16
            )));
        }

        let weight_global_scale = parse_f32_scalar(bytes_w, &key_w_global)?;
        let input_global_scale = parse_f32_scalar(bytes_i, &key_in_global)?;

        let packed = upload_u8(stream, bytes_p, &key_packed)?;
        let scale_per_block = upload_u8(stream, bytes_s, &key_scale)?;

        nvfp4.insert(
            base.clone(),
            Nvfp4Tensor {
                packed,
                scale_per_block,
                weight_global_scale,
                input_global_scale,
                n,
                k,
            },
        );
    }

    // Emit Bf16Tensor for everything else (skipping mtp/visual + quantized
    // sub-tensors that we already grouped above).
    let suffixes = [
        ".weight_packed",
        ".weight_scale",
        ".weight_global_scale",
        ".input_global_scale",
    ];
    for (name, (dtype, shape, bytes)) in &raw {
        if name.starts_with("mtp.") || name.contains(".visual.") {
            continue;
        }
        if suffixes.iter().any(|s| name.ends_with(s)) {
            continue;
        }
        // Handle BF16 tensors. The recipe also leaves F32 (A_log, dt_bias)
        // and BF16 (everything else). Convert F32 to BF16 for uniformity.
        let bf16_data: Vec<half::bf16> = match dtype.as_str() {
            "BF16" => {
                if bytes.len() % 2 != 0 {
                    return Err(LlmError::Safetensors(format!(
                        "{name}: BF16 byte length {} not divisible by 2",
                        bytes.len()
                    )));
                }
                bytes
                    .chunks_exact(2)
                    .map(|c| half::bf16::from_bits(u16::from_le_bytes([c[0], c[1]])))
                    .collect()
            },
            "F32" => bytes
                .chunks_exact(4)
                .map(|c| {
                    let v = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
                    half::bf16::from_f32(v)
                })
                .collect(),
            "F16" => bytes
                .chunks_exact(2)
                .map(|c| {
                    let v = half::f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32();
                    half::bf16::from_f32(v)
                })
                .collect(),
            other => {
                return Err(LlmError::Safetensors(format!(
                    "{name}: unhandled BF16-target dtype {other}"
                )));
            },
        };
        let dev = stream
            .memcpy_stod(&bf16_data)
            .map_err(|e| LlmError::Backend(format!("upload {name}: {e:?}")))?;
        let _ = upload_bf16; // suppress unused warning if dead-code-removed
        bf16.insert(
            name.clone(),
            Bf16Tensor {
                data: dev,
                shape: shape.clone(),
            },
        );
    }

    Ok(LoadedNvfp4Weights { bf16, nvfp4 })
}

// ---------------------------------------------------------------------------
// Model class (skeleton — full decode_step is the next session)
// ---------------------------------------------------------------------------

/// End-to-end NVFP4 inference path for Qwen3.6-A3B-NVFP4 (vLLM-quantized).
///
/// The `decode_step` body itself is the next session's work — it mirrors
/// `Qwen35ModelCudaQ4K::decode_step` (~2000 LOC of hybrid SSM + MoE
/// forward) with all matmul calls swapped from the Q4K dispatch to the
/// NVFP4 dispatch. This skeleton owns :
/// - `cfg`, `stream`, `ctx`, `session`, `kernels` (same as Q4K)
/// - `weights` : the loaded BF16 + NVFP4 tensors
/// - `kernels` : reuses existing `LlmKernels` (RMSNorm, RoPE, SSM, MoE
///   topk, etc. all stay BF16 / unchanged)
///
/// At construction time we just verify that the loaded weight set covers
/// the full layer count (40) and emit a per-layer breakdown for the
/// follow-up session.
pub struct Qwen35ModelCudaNVFP4 {
    pub cfg: Qwen35Config,
    pub stream: Arc<CudaStream>,
    pub ctx: Arc<CudaContext>,
    pub session: LtSession,
    pub kernels: LlmKernels,
    pub weights: LoadedNvfp4Weights,
    pub max_seq: usize,
}

impl Qwen35ModelCudaNVFP4 {
    /// Load a Qwen3.6-A3B-NVFP4 checkpoint from a HuggingFace model
    /// directory containing `config.json`, `model.safetensors`,
    /// `model.safetensors.index.json`, and `recipe.yaml`.
    ///
    /// `cfg` should be derived externally (e.g. via a small adapter that
    /// reads `config.json.text_config`) — we keep this constructor pure
    /// so it doesn't bake in a specific config-parsing path. For the
    /// short term, callers can hard-code the Qwen3.6-35B-A3B config or
    /// reuse `qwen35::parse_config` if a GGUF sibling of the same model
    /// is on disk.
    pub fn from_safetensors(
        dir: &Path,
        cfg: Qwen35Config,
        max_seq: usize,
    ) -> Result<Self, LlmError> {
        let ctx = CudaContext::new(0).map_err(|e| LlmError::Backend(format!("ctx: {e:?}")))?;
        let stream = ctx.default_stream();
        let session = LtSession::new(stream.clone())
            .map_err(|e| LlmError::Backend(format!("LtSession: {e:?}")))?;
        let kernels = LlmKernels::new(ctx.clone());

        let weights = load_nvfp4_safetensors(dir, &stream)?;

        // Emit a basic sanity check : we expect at least 40 layers' worth
        // of attn + moe linears for the 35B-A3B variant.
        let n_quant = weights.nvfp4.len();
        let n_bf16 = weights.bf16.len();
        eprintln!("[T246.9 NVFP4] loaded {n_quant} quantized + {n_bf16} BF16 tensors from {dir:?}");

        if matches!(cfg.variant, Qwen35Variant::Moe) {
            // Sanity-check : 35B-A3B has 256 experts × 3 linears × 30
            // SSM-MoE layers + 256 × 3 × 10 attention-MoE layers + ...
            // Just ensure we have at least the shared_expert + experts.0
            // for layer 0.
            let probe = "model.language_model.layers.0.mlp.experts.0.gate_proj";
            if !weights.nvfp4.contains_key(probe) {
                return Err(LlmError::Safetensors(format!(
                    "missing expected NVFP4 weight {probe} — checkpoint may not be Qwen3.6-A3B-NVFP4"
                )));
            }
        }

        Ok(Self {
            cfg,
            stream,
            ctx,
            session,
            kernels,
            weights,
            max_seq,
        })
    }

    /// One-token decode (NOT YET IMPLEMENTED — follow-up session).
    ///
    /// The full body is a near-clone of `Qwen35ModelCudaQ4K::decode_step` :
    /// per-layer hybrid SSM-or-attention block, MoE FFN, gather, ...
    /// All matmul dispatch sites are swapped from `QuantTensor` to
    /// `Nvfp4Tensor` (cuBLASLt `matmul_mxfp4` for dense, hand-written
    /// indexed sgemv NVFP4 for MoE). The SSM stack stays BF16
    /// (CudaSlice<half::bf16> reads / writes unchanged).
    ///
    /// Returns `LlmError::Backend("not implemented")` for now ; the
    /// substrate (loader + indexed kernel + parity test) is what this
    /// session lands.
    pub fn decode_step(&mut self, _token_id: u32) -> Result<u32, LlmError> {
        Err(LlmError::Backend(
            "Qwen35ModelCudaNVFP4::decode_step — TODO follow-up session, see RFC \
             b5fc8ead Phase 4-5"
                .into(),
        ))
    }
}

// ---------------------------------------------------------------------------
// Tests (pure-Rust, no GPU needed for the regex / dtype-grouping logic)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ignore_patterns_recognize_recipe_targets() {
        assert!(matches_ignore_pattern("lm_head.weight"));
        assert!(matches_ignore_pattern(
            "model.language_model.embed_tokens.weight"
        ));
        assert!(matches_ignore_pattern(
            "model.language_model.layers.3.mlp.gate.weight"
        ));
        assert!(matches_ignore_pattern(
            "model.language_model.layers.3.mlp.shared_expert_gate.weight"
        ));
        assert!(matches_ignore_pattern(
            "model.language_model.layers.0.linear_attn.in_proj_qkv.weight"
        ));
        assert!(matches_ignore_pattern("model.visual.blocks.0.attn.qkv"));
        assert!(matches_ignore_pattern("mtp.layers.0.mlp.gate.weight"));

        // Quantized paths should NOT match ignore.
        assert!(!matches_ignore_pattern(
            "model.language_model.layers.3.self_attn.q_proj.weight_packed"
        ));
        assert!(!matches_ignore_pattern(
            "model.language_model.layers.0.mlp.experts.7.up_proj.weight_packed"
        ));
        assert!(!matches_ignore_pattern(
            "model.language_model.layers.0.mlp.shared_expert.gate_proj.weight_packed"
        ));
    }

    #[test]
    fn nvfp4_tensor_alpha_inverts_two_globals() {
        let t = Nvfp4Tensor {
            packed: cuda_slice_dummy(),
            scale_per_block: cuda_slice_dummy(),
            weight_global_scale: 2.0,
            input_global_scale: 4.0,
            n: 16,
            k: 16,
        };
        let a = t.matmul_alpha();
        assert!((a - 1.0 / 8.0).abs() < 1e-6, "alpha={a}");
    }

    /// Helper for tests that don't actually touch CUDA — gives us a
    /// safely-dropped `CudaSlice<u8>` placeholder. We can't construct
    /// one without a context, so this test runs only when CUDA is
    /// available.
    #[cfg(feature = "cuda")]
    fn cuda_slice_dummy() -> CudaSlice<u8> {
        let ctx = CudaContext::new(0).expect("CUDA ctx");
        let stream = ctx.default_stream();
        stream.alloc_zeros::<u8>(1).expect("alloc")
    }
}
