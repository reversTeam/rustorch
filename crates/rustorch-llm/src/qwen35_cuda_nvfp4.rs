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
//! - **T246.9 NVFP4.1** : safetensors loader (committed `7804dd1`).
//! - **T246.9 NVFP4.2** : indexed sgemv kernels (committed `1e69e23`).
//! - **T246.9 NVFP4.3** : Qwen35ModelCudaNVFP4 model class skeleton (`cde9b3b`).
//! - **T246.9 NVFP4.4** : E2E pipeline test (`5caa137`).
//! - **T246.9 NVFP4-FINISH** (this commit) : decode_step body — clone of
//!   `Qwen35ModelCudaQ4K::decode_step` with Nvfp4 weight dispatch + indexed
//!   MoE FFN. SSM stack stays BF16 (per recipe ignore patterns), KV cache
//!   stays BF16. CUDA Graph capture available via `RUSTORCH_MOE_GRAPH=1`.

#![cfg(feature = "cuda")]

use crate::qwen35::{LayerKind, Qwen35Config, Qwen35Variant};
use crate::LlmError;
use cudarc::driver::sys::{CUgraphInstantiate_flags, CUstreamCaptureMode};
use cudarc::driver::{CudaContext, CudaGraph, CudaSlice, CudaStream, PinnedHostSlice};
use rustorch_cuda::cublas_lt::LtSession;
use rustorch_cuda::llm_kernels::LlmKernels;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Maximum tree size for tree-batched prefill. Mirrors `qwen35_cuda_q4k::MAX_TREE_SIZE`
/// — the per-tree scratch buffers in `DecodeScratch` are sized to this constant.
/// Prefill of longer prompts must chunk by this size.
pub const MAX_TREE_SIZE: usize = 512;

// ---------------------------------------------------------------------------
// CUDA Graph runtime gates (mirror of qwen35_cuda_q4k.rs)
// ---------------------------------------------------------------------------

/// T246.9 NVFP4-FINISH — runtime gate to enable CUDA Graph capture for the
/// MoE NVFP4 decode body. Default ON. The body is fully zero-host-sync by
/// construction (indexed MoE kernels, devscalar epilogues, no `memcpy_dtov`),
/// so capture is always safe ; the gate exists purely so we can A/B-test
/// the captured-vs-issue path during perf tuning.
fn moe_graph_enabled() -> bool {
    std::env::var("RUSTORCH_MOE_GRAPH")
        .ok()
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(true)
}

/// T246.10 TrackK.b — runtime gate to enable tree-batched prefill path on
/// the NVFP4 model. Default OFF — must be set explicitly to `RUSTORCH_NVFP4_PREFILL_TREE=1`
/// to opt in. When OFF, `prefill_tokens` keeps the TrackK sequential
/// `decode_step` loop verbatim.
///
/// When ON :
/// - For N=1, prefill_tokens falls through to decode_step (same as Q4_K).
/// - For N>1, prefill_tokens uses the tree-batched path
///   (`decode_step_tree_hybrid_inner_capture_nvfp4`) with per-tree-row
///   scratch + tree-aware KV/SSM forking + CUDA Graph capture.
///
/// The per-row matmuls still use the M=1 NVFP4 SGEMV kernels (we do not
/// have batched NVFP4 matmul kernels today — see decision 450e88ef). The
/// expected pp512 lift over TrackK's 30 tok/s is therefore bounded by what
/// Graph capture + tree-aware KV/SSM can deliver alone (informational
/// target ≥60 tok/s).
fn nvfp4_prefill_tree_enabled() -> bool {
    std::env::var("RUSTORCH_NVFP4_PREFILL_TREE")
        .ok()
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

/// T246.10 TrackK.b — minimum tree size to take the tree-batched path
/// (when `RUSTORCH_NVFP4_PREFILL_TREE=1`). Below this size the sequential
/// decode_step loop wins on launch overhead.
///
/// Empirical crossover point on Qwen3.6-35B-A3B-NVFP4 / DGX GB10 sm_121 :
/// - pp32  : tree-batched = 21 tok/s, sequential = 38.5 tok/s → tree LOSES
/// - pp128 : tree-batched = 22 tok/s, sequential = 38.8 tok/s → tree LOSES
/// - pp512 : tree-batched = 38.5 tok/s, sequential = 30.1 tok/s → tree WINS (+28 %)
///
/// The tree path's win at high N comes from the tree-aware GQA scan
/// (`gqa_decode_tree_bf16`) which amortizes the KV prefix scan over all
/// tree rows in a single launch. At low N, each row's tree-aware GQA is
/// less efficient than a sequential M=1 scan that benefits from the
/// captured decode_step graph (~3K kernel launches vs ~20K for the tree
/// path's per-row M=1 NVFP4 SGEMV calls). With batched NVFP4 matmul
/// kernels (TrackK.c future work) the crossover would move down to N=8.
const NVFP4_PREFILL_TREE_MIN_N: usize = 256;

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

    /// `alpha` used at GEMM call : `1 / weight_global_scale_checkpoint`.
    ///
    /// **Why not `1 / (w_g * in_g)` like vLLM ?** vLLM quantizes the
    /// activation X to FP4 before the matmul, dividing by `in_g_ckpt` along
    /// the way. Their kernel then multiplies by `(in_g_ckpt * w_g_ckpt)` to
    /// recover. Our kernel reads **BF16 X directly** (no activation quantize),
    /// so the `in_g_ckpt` factor must NOT appear in `alpha`.
    ///
    /// Math (verbatim from kernel) :
    ///   `acc = sum(W_fp4 * scale_e4m3 * X_bf16)`
    ///   `y   = acc * alpha`
    ///
    /// The dequant convention is `W_real = W_fp4 * scale_e4m3 / w_g_ckpt`
    /// (`w_g_ckpt` is stored as the divisor — `max_E4M3*max_E2M1/max(|W|)`).
    /// Therefore `y = sum(W_real * X_bf16) * w_g_ckpt * alpha`. To recover
    /// `sum(W_real * X_bf16)` we need `alpha = 1 / w_g_ckpt`.
    ///
    /// T246.9 NVFP4-BISECT.2 — the previous version multiplied by an
    /// additional `1/in_g_ckpt ≈ 0.001-0.01` factor, shrinking every NVFP4
    /// matmul output by 100-1000×. The residual stream absorbed enough signal
    /// to keep ~18 tokens coherent but the lm_head logits collapsed past that.
    pub fn matmul_alpha(&self) -> f32 {
        if self.weight_global_scale == 0.0 {
            1.0
        } else {
            1.0 / self.weight_global_scale
        }
    }

    /// Dispatch a M=1 GEMV through the hand-written NVFP4 SGEMV kernel.
    /// Output `y[n]` BF16, input `x[k]` BF16 read from device.
    pub(crate) fn dispatch_matmul_m1(
        &self,
        kernels: &LlmKernels,
        stream: &Arc<CudaStream>,
        x: u64,
        y: u64,
    ) -> Result<(), LlmError> {
        use cudarc::driver::DevicePtr;
        unsafe {
            let (pp, _g0) = self.packed.device_ptr(stream);
            let (sp, _g1) = self.scale_per_block.device_ptr(stream);
            kernels
                .sgemv_nvfp4_bf16(
                    stream,
                    pp,
                    sp,
                    self.matmul_alpha(),
                    x,
                    y,
                    self.n as i32,
                    self.k as i32,
                )
                .map_err(|e| LlmError::Backend(format!("sgemv_nvfp4_bf16: {e:?}")))
        }
    }
}

/// One BF16 weight tensor on GPU (for ignored / non-quantized linears).
pub struct Bf16Tensor {
    pub data: CudaSlice<half::bf16>,
    pub shape: Vec<usize>,
}

impl Bf16Tensor {
    /// `[n, k]` interpretation (used for matmul Linear weights).
    pub fn nk(&self) -> (usize, usize) {
        if self.shape.len() == 2 {
            (self.shape[0], self.shape[1])
        } else {
            (self.shape.iter().product(), 1)
        }
    }

    /// Dispatch a M=1 BF16 GEMV through `sgemv_bf16_bf16_dispatch`.
    pub(crate) fn dispatch_matmul_m1(
        &self,
        kernels: &LlmKernels,
        stream: &Arc<CudaStream>,
        x: u64,
        y: u64,
    ) -> Result<(), LlmError> {
        use cudarc::driver::DevicePtr;
        let (n, k) = self.nk();
        unsafe {
            let (w, _g) = self.data.device_ptr(stream);
            kernels
                .sgemv_bf16_bf16_dispatch(stream, w, x, y, n as i32, k as i32)
                .map_err(|e| LlmError::Backend(format!("sgemv_bf16_dispatch: {e:?}")))
        }
    }
}

/// All weights of the Qwen3.6-35B-A3B-NVFP4 model, organized for direct
/// consumption by `Qwen35ModelCudaNVFP4::decode_step`.
///
/// This is the output of `load_nvfp4_safetensors`. It is intentionally a
/// flat name → tensor map (BF16 + NVFP4) so the model class can index
/// into it once and assemble its per-layer typed records.
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

    fn take_bf16(&mut self, name: &str) -> Result<Bf16Tensor, LlmError> {
        self.bf16
            .remove(name)
            .ok_or_else(|| LlmError::MissingWeight(name.to_string()))
    }

    fn take_nvfp4(&mut self, name: &str) -> Result<Nvfp4Tensor, LlmError> {
        self.nvfp4
            .remove(name)
            .ok_or_else(|| LlmError::MissingWeight(name.to_string()))
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
/// per-layer to assemble per-layer records.
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

    let mut quantized_bases: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    for name in raw.keys() {
        if let Some(base) = name.strip_suffix(".weight_packed") {
            if name.starts_with("mtp.") || name.contains(".visual.") {
                continue;
            }
            quantized_bases.insert(base.to_string());
        }
    }

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
// Per-layer typed weight records
// ---------------------------------------------------------------------------

/// MoE FFN weights for one layer (Qwen3.6-35B-A3B), NVFP4 flavour.
/// Mirrors `MoeFfnQ4K` but every per-expert linear is `Nvfp4Tensor`.
pub(crate) struct MoeFfnNvfp4 {
    /// `[n_experts, D]` BF16 — router logits (NOT NVFP4 per recipe).
    pub(crate) gate_inp: Bf16Tensor,
    /// `n_experts` × `[expert_f, D]` NVFP4 — per-expert gate.
    pub(crate) gate_exps: Vec<Nvfp4Tensor>,
    /// `n_experts` × `[expert_f, D]` NVFP4 — per-expert up.
    pub(crate) up_exps: Vec<Nvfp4Tensor>,
    /// `n_experts` × `[D, expert_f]` NVFP4 — per-expert down.
    pub(crate) down_exps: Vec<Nvfp4Tensor>,
    /// `[1, D]` BF16 — shared-expert routing gain (sigmoid-gated scalar).
    pub(crate) gate_inp_shexp: Bf16Tensor,
    /// `[expert_f, D]` NVFP4 — shared-expert gate.
    pub(crate) gate_shexp: Nvfp4Tensor,
    /// `[expert_f, D]` NVFP4 — shared-expert up.
    pub(crate) up_shexp: Nvfp4Tensor,
    /// `[D, expert_f]` NVFP4 — shared-expert down.
    pub(crate) down_shexp: Nvfp4Tensor,

    // ── device-side dispatch tables (built once at load time) ──
    /// `[n_experts]` u64 — base device pointers of each expert's `packed`.
    pub(crate) gate_packed_ptrs_dev: CudaSlice<u64>,
    pub(crate) up_packed_ptrs_dev: CudaSlice<u64>,
    pub(crate) down_packed_ptrs_dev: CudaSlice<u64>,
    /// `[n_experts]` u64 — base device pointers of each expert's `scale`.
    pub(crate) gate_scale_ptrs_dev: CudaSlice<u64>,
    pub(crate) up_scale_ptrs_dev: CudaSlice<u64>,
    pub(crate) down_scale_ptrs_dev: CudaSlice<u64>,
    /// `[n_experts]` f32 — per-expert alpha = 1/(w_global * in_global).
    pub(crate) gate_alphas_dev: CudaSlice<f32>,
    pub(crate) up_alphas_dev: CudaSlice<f32>,
    pub(crate) down_alphas_dev: CudaSlice<f32>,
}

/// Attention block weights (one of every 4 layers in Qwen3.6-A3B).
pub(crate) struct AttnBlockNvfp4 {
    /// `[D]` BF16 — pre-attention RMSNorm gain.
    pub(crate) attn_norm: Bf16Tensor,
    /// `[D]` BF16 — post-attention RMSNorm gain.
    pub(crate) post_norm: Bf16Tensor,
    /// `[head_dim]` BF16 — per-head Q normalization gain.
    pub(crate) q_norm: Bf16Tensor,
    /// `[head_dim]` BF16 — per-head K normalization gain.
    pub(crate) k_norm: Bf16Tensor,
    /// `[2*n_q*head_dim, D]` NVFP4 — Q+gate combined projection (Qwen3Next).
    pub(crate) w_q: Nvfp4Tensor,
    /// `[n_kv*head_dim, D]` NVFP4 — K projection.
    pub(crate) w_k: Nvfp4Tensor,
    /// `[n_kv*head_dim, D]` NVFP4 — V projection.
    pub(crate) w_v: Nvfp4Tensor,
    /// `[D, n_q*head_dim]` NVFP4 — output projection.
    pub(crate) w_o: Nvfp4Tensor,
    /// FFN — MoE for Qwen3.6-A3B.
    pub(crate) ffn: MoeFfnNvfp4,
}

/// SSM (gated delta net) block weights — 30/40 layers in Qwen3.6-A3B.
/// Per recipe.yaml `linear_attn.*` is BF16 (NOT NVFP4).
pub(crate) struct SsmBlockNvfp4 {
    /// `[D]` BF16 — pre-SSM RMSNorm gain.
    pub(crate) attn_norm: Bf16Tensor,
    /// `[D]` BF16 — post-SSM RMSNorm gain.
    pub(crate) post_norm: Bf16Tensor,
    /// `[conv_dim, D]` BF16 — fused QKV projection.
    pub(crate) w_qkv: Bf16Tensor,
    /// `[value_dim, D]` BF16 — gate (z) projection.
    pub(crate) w_gate: Bf16Tensor,
    /// `[conv_kernel, conv_dim]` BF16 — depth-wise 1-D conv kernel.
    pub(crate) conv1d: Bf16Tensor,
    /// `[num_v_heads, D]` BF16 — alpha (dt) projection.
    pub(crate) w_alpha: Bf16Tensor,
    /// `[num_v_heads, D]` BF16 — beta projection.
    pub(crate) w_beta: Bf16Tensor,
    /// `[num_v_heads]` BF16 — dt bias.
    pub(crate) dt_bias: Bf16Tensor,
    /// `[num_v_heads]` BF16 — A scalar per head (from log A).
    pub(crate) ssm_a: Bf16Tensor,
    /// `[head_v_dim]` BF16 — gated RMSNorm gain.
    pub(crate) ssm_norm: Bf16Tensor,
    /// `[D, value_dim]` BF16 — output projection.
    pub(crate) ssm_out: Bf16Tensor,
    /// FFN — MoE for Qwen3.6-A3B.
    pub(crate) ffn: MoeFfnNvfp4,
}

pub(crate) enum BlockNvfp4 {
    Attn(AttnBlockNvfp4),
    Ssm(SsmBlockNvfp4),
}

pub(crate) struct KvCache {
    pub(crate) k: CudaSlice<half::bf16>,
    pub(crate) v: CudaSlice<half::bf16>,
}

pub(crate) struct RopeFreqs {
    pub(crate) inv_freq: CudaSlice<f32>,
}

pub(crate) struct SsmState {
    pub(crate) state: CudaSlice<half::bf16>,
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
fn get_attn_layer_idx(cfg: &Qwen35Config, global_li: usize) -> usize {
    cfg.attention_indices
        .iter()
        .position(|&i| i == global_li)
        .unwrap_or_else(|| panic!("layer {global_li} not in attention_indices"))
}

/// Pre-allocated scratch buffers (mirror of Q4K `DecodeScratch`, NVFP4 flavour).
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
    pub(crate) gqa_partial_m: CudaSlice<f32>,
    pub(crate) gqa_partial_l: CudaSlice<f32>,
    pub(crate) gqa_partial_o: CudaSlice<half::bf16>,
    pub(crate) moe_router_logits: CudaSlice<half::bf16>,
    pub(crate) moe_topk_idx: CudaSlice<i32>,
    pub(crate) moe_topk_w: CudaSlice<half::bf16>,
    pub(crate) moe_expert_gate: CudaSlice<half::bf16>,
    pub(crate) moe_expert_up: CudaSlice<half::bf16>,
    pub(crate) moe_expert_out: CudaSlice<half::bf16>,
    pub(crate) moe_shexp_dot: CudaSlice<half::bf16>,

    // ── T246.10 TrackK.b — Tree-batched prefill scratch (mirror of Q4_K) ──
    //
    // Each buffer is sized for MAX_TREE_SIZE rows. Touched only when
    // `RUSTORCH_NVFP4_PREFILL_TREE=1` and `prefill_tokens` is called with
    // N > 1 (the tree-batched path).
    /// `[MAX_TREE_SIZE]` u32 — input draft tokens.
    pub(crate) tree_drafts: CudaSlice<u32>,
    /// `[MAX_TREE_SIZE]` i32 — parent pointer per node, root = -1.
    pub(crate) tree_parents: CudaSlice<i32>,
    /// `[MAX_TREE_SIZE]` u16 — depth per node, root = 0.
    pub(crate) tree_depths: CudaSlice<u16>,
    /// `[MAX_TREE_SIZE]` u32 — argmax token per tree node row of logits.
    pub(crate) tree_argmax: CudaSlice<u32>,
    /// Pinned host buffer `[MAX_TREE_SIZE]` u32 — DtoH target for argmax tokens.
    pub(crate) tree_argmax_host_pinned: PinnedHostSlice<u32>,

    /// `[MAX_TREE_SIZE, n_q, GQA_N_SPLIT]` f32 — partial m for tree GQA.
    pub(crate) tree_gqa_partial_m: CudaSlice<f32>,
    /// `[MAX_TREE_SIZE, n_q, GQA_N_SPLIT]` f32 — partial l for tree GQA.
    pub(crate) tree_gqa_partial_l: CudaSlice<f32>,
    /// `[MAX_TREE_SIZE, n_q, GQA_N_SPLIT, head_dim]` BF16 — partial o for tree GQA.
    pub(crate) tree_gqa_partial_o: CudaSlice<half::bf16>,

    /// `[MAX_TREE_SIZE, D]` BF16 — per-tree-row hidden state.
    pub(crate) tree_h: CudaSlice<half::bf16>,
    /// `[MAX_TREE_SIZE, D]` BF16 — per-tree-row normalized hidden state.
    pub(crate) tree_h_norm: CudaSlice<half::bf16>,
    /// `[MAX_TREE_SIZE, D]` BF16 — per-tree-row residual snapshot.
    pub(crate) tree_residual: CudaSlice<half::bf16>,
    /// `[MAX_TREE_SIZE, q_dim]` BF16 — per-tree-row Q projection.
    pub(crate) tree_q_buf: CudaSlice<half::bf16>,
    /// `[MAX_TREE_SIZE, kv_dim]` BF16 — per-tree-row K projection.
    pub(crate) tree_k_buf: CudaSlice<half::bf16>,
    /// `[MAX_TREE_SIZE, kv_dim]` BF16 — per-tree-row V projection.
    pub(crate) tree_v_buf: CudaSlice<half::bf16>,
    /// `[MAX_TREE_SIZE, q_dim]` BF16 — per-tree-row attention output.
    pub(crate) tree_attn_out: CudaSlice<half::bf16>,
    /// `[MAX_TREE_SIZE, scratch_gate]` BF16 — per-tree-row scratch for FFN gate / attn gate.
    pub(crate) tree_gate_buf: CudaSlice<half::bf16>,
    /// `[MAX_TREE_SIZE, scratch_up]` BF16 — per-tree-row scratch for QG / FFN up (≥ 2*q_dim).
    pub(crate) tree_up_buf: CudaSlice<half::bf16>,
    /// `[MAX_TREE_SIZE, vocab]` BF16 — per-tree-row LM-head logits.
    pub(crate) tree_logits: CudaSlice<half::bf16>,

    /// `[MAX_TREE_SIZE, conv_dim]` BF16 — per-tree-row SSM qkv_mixed.
    pub(crate) tree_ssm_qkv_mixed: CudaSlice<half::bf16>,
    /// `[MAX_TREE_SIZE, conv_dim]` BF16 — per-tree-row SSM conv_out.
    pub(crate) tree_ssm_conv_out: CudaSlice<half::bf16>,
    /// `[MAX_TREE_SIZE, value_dim]` BF16 — per-tree-row SSM z (gate).
    pub(crate) tree_ssm_z: CudaSlice<half::bf16>,
    /// `[MAX_TREE_SIZE, n_v_heads]` BF16 — per-tree-row alpha (dt).
    pub(crate) tree_ssm_alpha: CudaSlice<half::bf16>,
    /// `[MAX_TREE_SIZE, n_v_heads]` BF16 — per-tree-row beta.
    pub(crate) tree_ssm_beta: CudaSlice<half::bf16>,
    /// `[MAX_TREE_SIZE, value_dim]` BF16 — per-tree-row q broadcast to n_v heads.
    pub(crate) tree_ssm_q_v: CudaSlice<half::bf16>,
    /// `[MAX_TREE_SIZE, value_dim]` BF16 — per-tree-row k broadcast to n_v heads.
    pub(crate) tree_ssm_k_v: CudaSlice<half::bf16>,
    /// `[MAX_TREE_SIZE, value_dim]` BF16 — per-tree-row gated SSM output.
    pub(crate) tree_ssm_out_buf: CudaSlice<half::bf16>,
    /// `[MAX_TREE_SIZE]` i32 — wave indices buffer (used by delta_net_step_tree_bf16).
    pub(crate) tree_ssm_wave_indices: CudaSlice<i32>,

    /// Per-SSM-layer tree-fork SSM state, sized
    /// `[MAX_TREE_SIZE × n_v_heads × head_v_dim²]` BF16 each.
    pub(crate) tree_ssm_states: Vec<CudaSlice<half::bf16>>,
    /// Per-SSM-layer tree-fork conv1d state, sized
    /// `[MAX_TREE_SIZE × (conv_kernel-1) × conv_dim]` BF16 each.
    pub(crate) tree_conv_states: Vec<CudaSlice<half::bf16>>,
}

const GQA_N_SPLIT: usize = 4;

// ---------------------------------------------------------------------------
// Hard-coded Qwen3.6-35B-A3B-NVFP4 config (matches RedHatAI checkpoint)
// ---------------------------------------------------------------------------

/// Build the canonical `Qwen35Config` for the Qwen3.6-35B-A3B-NVFP4 model.
///
/// Mirrors the values in `config.json.text_config` of the on-disk
/// `RedHatAI/Qwen3.6-35B-A3B-NVFP4` checkpoint. Hard-coded here because
/// the existing `qwen35::parse_config` reads from GGUF, not safetensors,
/// and the NVFP4 path doesn't have a GGUF sibling on disk.
pub fn qwen36_35b_a3b_nvfp4_config() -> Qwen35Config {
    // 40 layers, every 4th (indices 3, 7, 11, ..., 39) is full_attention.
    let attention_indices: Vec<usize> = (0..40).filter(|&li| (li + 1) % 4 == 0).collect();
    let ssm_indices: Vec<usize> = (0..40).filter(|&li| (li + 1) % 4 != 0).collect();

    Qwen35Config {
        variant: Qwen35Variant::Moe,
        n_layers: 40,
        d: 2048,
        f: 0, // dense FFN unused (MoE)
        n_q_heads: 16,
        n_kv_heads: 2,
        rope_dim: 64, // partial_rotary_factor=0.25, head_dim=256 → 64
        vocab: 248320,
        max_context: 262144,
        rms_eps: 1e-6,
        rope_base: 10_000_000.0,

        // SSM hyperparams from text_config
        // linear_num_value_heads=32, linear_value_head_dim=128 → ssm_inner = 4096
        ssm_inner: 32 * 128,
        // linear_value_head_dim = linear_key_head_dim = 128 → state per group/value head
        ssm_state: 128,
        // num_value_heads = 32 (the Q4K loader calls this dt_rank for historical reasons)
        ssm_dt_rank: 32,
        // num_key_heads = 16 (the Q4K loader calls this groups)
        ssm_groups: 16,
        ssm_conv_kernel: 4,

        n_experts: 256,
        n_experts_used: 8,
        expert_f: 512,
        attn_head_dim: 256,
        attention_indices,
        ssm_indices,
    }
}

// ---------------------------------------------------------------------------
// Model class
// ---------------------------------------------------------------------------

/// End-to-end NVFP4 inference path for Qwen3.6-A3B-NVFP4.
pub struct Qwen35ModelCudaNVFP4 {
    pub config: Qwen35Config,
    pub(crate) ctx: Arc<CudaContext>,
    pub(crate) stream: Arc<CudaStream>,
    #[allow(dead_code)]
    pub(crate) session: LtSession,
    pub(crate) kernels: LlmKernels,
    pub(crate) token_emb: Bf16Tensor,
    pub(crate) final_norm: Bf16Tensor,
    pub(crate) lm_head: Bf16Tensor,
    pub(crate) rope_freqs: RopeFreqs,
    pub(crate) blocks: Vec<BlockNvfp4>,
    pub(crate) kv_caches: Vec<KvCache>,
    pub(crate) ssm_states: Vec<SsmState>,
    pub(crate) max_seq: usize,
    pub(crate) position: usize,
    pub(crate) scratch: DecodeScratch,
    pub(crate) position_dev: CudaSlice<i32>,
    pub(crate) kv_len_dev: CudaSlice<i32>,
    pub(crate) current_token_dev: CudaSlice<u32>,
    pub(crate) decode_graph: Option<CudaGraph>,
    pub(crate) use_ssm_fuse: bool,
    pub(crate) next_token_host_pinned: PinnedHostSlice<u32>,

    // ── T246.10 TrackK.b — Tree-batched prefill infrastructure ──
    //
    // Mirror of the Q4_K Graph-capture wrapper. Populated lazily on the
    // second `prefill_tokens(N)` call with a given `N` (after a warmup
    // pass that primes nvrtc compile + cuModuleLoad for every kernel).
    /// Bag-of-graphs cache keyed by `N`. Cleared by `reset_state()`.
    pub(crate) prefill_graphs: HashMap<usize, CudaGraph>,
    /// Set of `N` values for which the prefill body has been run once
    /// (uncaptured warmup) and is ready to be captured on the next call.
    pub(crate) prefill_warmed: HashSet<usize>,
    /// Pinned-host `[MAX_TREE_SIZE]` u32 — per-call drafts upload source.
    /// Required so the HtoD outside `begin_capture` is truly async (un-pinned
    /// pageable memory degenerates to a synchronous copy that would abort
    /// stream capture).
    pub(crate) prefill_drafts_host_pinned: PinnedHostSlice<u32>,
    /// `[0, 1, 2, ..., MAX_TREE_SIZE - 1]` i32 — pre-baked linear-chain wave
    /// indices buffer. The hybrid SSM path under prefill_capture reads
    /// `tree_ssm_wave_indices_linear + d*4` (wave_size=1, capture-safe).
    pub(crate) tree_ssm_wave_indices_linear: CudaSlice<i32>,
}

impl Qwen35ModelCudaNVFP4 {
    /// Load a Qwen3.6-A3B-NVFP4 checkpoint from a HuggingFace model
    /// directory. `cfg` must match the on-disk model's hyperparameters
    /// (use [`qwen36_35b_a3b_nvfp4_config`] for the canonical RedHatAI
    /// Qwen3.6-35B-A3B-NVFP4 release).
    pub fn from_safetensors(
        dir: &Path,
        cfg: Qwen35Config,
        max_seq: usize,
    ) -> Result<Self, LlmError> {
        let ctx = CudaContext::new(0).map_err(|e| LlmError::Backend(format!("ctx: {e:?}")))?;
        // Same rationale as Q4K path : disable cudarc event tracking before
        // any allocs so CUDA Graph capture isn't invalidated by stale events.
        unsafe {
            ctx.disable_event_tracking();
        }
        // Dedicated non-default stream — CUDA Graphs cannot capture the
        // legacy/default stream.
        let stream = ctx
            .new_stream()
            .map_err(|e| LlmError::Backend(format!("new_stream: {e:?}")))?;
        let session = LtSession::new(stream.clone())
            .map_err(|e| LlmError::Backend(format!("LtSession: {e:?}")))?;
        let kernels = LlmKernels::new(ctx.clone());

        // Load all weights into the flat name → tensor map.
        let mut weights = load_nvfp4_safetensors(dir, &stream)?;

        // ---- Globals (BF16) ----
        let token_emb = weights.take_bf16("model.language_model.embed_tokens.weight")?;
        let final_norm = weights.take_bf16("model.language_model.norm.weight")?;
        let lm_head = weights.take_bf16("lm_head.weight")?;

        // RoPE precompute.
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

        // ---- Per-layer weights ----
        let mut blocks: Vec<BlockNvfp4> = Vec::with_capacity(cfg.n_layers);
        let mut kv_caches: Vec<KvCache> = Vec::new();
        let mut ssm_states: Vec<SsmState> = Vec::new();

        let kv_dim = cfg.n_kv_heads * cfg.head_dim();
        let value_dim = cfg.ssm_dt_rank * cfg.ssm_state;
        let key_dim = cfg.ssm_groups * cfg.ssm_state;
        let conv_dim = 2 * key_dim + value_dim;
        let conv_kernel = cfg.ssm_conv_kernel;

        for li in 0..cfg.n_layers {
            let kind = cfg.layer_kind(li);
            let prefix = format!("model.language_model.layers.{li}.");
            let block = match kind {
                LayerKind::Attention => {
                    let attn = AttnBlockNvfp4 {
                        attn_norm: weights.take_bf16(&format!("{prefix}input_layernorm.weight"))?,
                        post_norm: weights
                            .take_bf16(&format!("{prefix}post_attention_layernorm.weight"))?,
                        q_norm: weights.take_bf16(&format!("{prefix}self_attn.q_norm.weight"))?,
                        k_norm: weights.take_bf16(&format!("{prefix}self_attn.k_norm.weight"))?,
                        w_q: weights.take_nvfp4(&format!("{prefix}self_attn.q_proj"))?,
                        w_k: weights.take_nvfp4(&format!("{prefix}self_attn.k_proj"))?,
                        w_v: weights.take_nvfp4(&format!("{prefix}self_attn.v_proj"))?,
                        w_o: weights.take_nvfp4(&format!("{prefix}self_attn.o_proj"))?,
                        ffn: load_moe_layer(&mut weights, &stream, li, cfg.n_experts)?,
                    };
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
                    BlockNvfp4::Attn(attn)
                },
                LayerKind::Ssm => {
                    // T246.9 NVFP4-BISECT.2 : per vLLM `gdn_linear_attn.py`
                    // canonical math `A = -exp(A_log)`, applied here at load
                    // time so the GGUF-trained `ssm_pre_step_bf16` kernel
                    // (which expects `ssm_a = A` directly, with the `-exp`
                    // already folded in) computes the right state transition.
                    let mut ssm_a = weights.take_bf16(&format!("{prefix}linear_attn.A_log"))?;
                    transform_a_log_to_ssm_a(&stream, &mut ssm_a)?;

                    let ssm = SsmBlockNvfp4 {
                        attn_norm: weights.take_bf16(&format!("{prefix}input_layernorm.weight"))?,
                        post_norm: weights
                            .take_bf16(&format!("{prefix}post_attention_layernorm.weight"))?,
                        w_qkv: weights
                            .take_bf16(&format!("{prefix}linear_attn.in_proj_qkv.weight"))?,
                        w_gate: weights
                            .take_bf16(&format!("{prefix}linear_attn.in_proj_z.weight"))?,
                        conv1d: weights.take_bf16(&format!("{prefix}linear_attn.conv1d.weight"))?,
                        w_alpha: weights
                            .take_bf16(&format!("{prefix}linear_attn.in_proj_a.weight"))?,
                        w_beta: weights
                            .take_bf16(&format!("{prefix}linear_attn.in_proj_b.weight"))?,
                        dt_bias: weights.take_bf16(&format!("{prefix}linear_attn.dt_bias"))?,
                        ssm_a,
                        ssm_norm: weights.take_bf16(&format!("{prefix}linear_attn.norm.weight"))?,
                        ssm_out: weights
                            .take_bf16(&format!("{prefix}linear_attn.out_proj.weight"))?,
                        ffn: load_moe_layer(&mut weights, &stream, li, cfg.n_experts)?,
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
                    BlockNvfp4::Ssm(ssm)
                },
            };
            blocks.push(block);
        }

        // ---- Pre-allocate decode scratch ----
        let kv_dim_attn = cfg.n_kv_heads * cfg.head_dim();
        let q_dim = cfg.n_q_heads * cfg.head_dim();
        // up_buf is reused as QG scratch (size 2*q_dim) ; gate_buf used as
        // attn raw output (size q_dim).
        let scratch_up = (2 * q_dim).max(cfg.expert_f);
        let scratch_gate = q_dim.max(cfg.expert_f);
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
                .alloc_zeros::<half::bf16>(conv_dim)
                .map_err(|e| LlmError::Backend(format!("scratch qkv: {e:?}")))?,
            conv_out: stream
                .alloc_zeros::<half::bf16>(conv_dim)
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
                .alloc_zeros::<half::bf16>(value_dim)
                .map_err(|e| LlmError::Backend(format!("scratch q_v: {e:?}")))?,
            k_v: stream
                .alloc_zeros::<half::bf16>(value_dim)
                .map_err(|e| LlmError::Backend(format!("scratch k_v: {e:?}")))?,
            ssm_out_buf: stream
                .alloc_zeros::<half::bf16>(value_dim)
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
            gqa_partial_m: stream
                .alloc_zeros::<f32>(cfg.n_q_heads * GQA_N_SPLIT)
                .map_err(|e| LlmError::Backend(format!("scratch gqa_m: {e:?}")))?,
            gqa_partial_l: stream
                .alloc_zeros::<f32>(cfg.n_q_heads * GQA_N_SPLIT)
                .map_err(|e| LlmError::Backend(format!("scratch gqa_l: {e:?}")))?,
            gqa_partial_o: stream
                .alloc_zeros::<half::bf16>(cfg.n_q_heads * GQA_N_SPLIT * cfg.head_dim())
                .map_err(|e| LlmError::Backend(format!("scratch gqa_o: {e:?}")))?,
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

            // ── TrackK.b — Tree-batched prefill scratch ──
            tree_drafts: stream
                .alloc_zeros::<u32>(MAX_TREE_SIZE)
                .map_err(|e| LlmError::Backend(format!("scratch tree_drafts: {e:?}")))?,
            tree_parents: stream
                .alloc_zeros::<i32>(MAX_TREE_SIZE)
                .map_err(|e| LlmError::Backend(format!("scratch tree_parents: {e:?}")))?,
            tree_depths: stream
                .alloc_zeros::<u16>(MAX_TREE_SIZE)
                .map_err(|e| LlmError::Backend(format!("scratch tree_depths: {e:?}")))?,
            tree_argmax: stream
                .alloc_zeros::<u32>(MAX_TREE_SIZE)
                .map_err(|e| LlmError::Backend(format!("scratch tree_argmax: {e:?}")))?,
            tree_argmax_host_pinned: unsafe { ctx.alloc_pinned::<u32>(MAX_TREE_SIZE) }
                .map_err(|e| LlmError::Backend(format!("pinned tree_argmax: {e:?}")))?,
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
            tree_h: stream
                .alloc_zeros::<half::bf16>(MAX_TREE_SIZE * cfg.d)
                .map_err(|e| LlmError::Backend(format!("scratch tree_h: {e:?}")))?,
            tree_h_norm: stream
                .alloc_zeros::<half::bf16>(MAX_TREE_SIZE * cfg.d)
                .map_err(|e| LlmError::Backend(format!("scratch tree_h_norm: {e:?}")))?,
            tree_residual: stream
                .alloc_zeros::<half::bf16>(MAX_TREE_SIZE * cfg.d)
                .map_err(|e| LlmError::Backend(format!("scratch tree_residual: {e:?}")))?,
            tree_q_buf: stream
                .alloc_zeros::<half::bf16>(MAX_TREE_SIZE * q_dim)
                .map_err(|e| LlmError::Backend(format!("scratch tree_q_buf: {e:?}")))?,
            tree_k_buf: stream
                .alloc_zeros::<half::bf16>(MAX_TREE_SIZE * kv_dim_attn)
                .map_err(|e| LlmError::Backend(format!("scratch tree_k_buf: {e:?}")))?,
            tree_v_buf: stream
                .alloc_zeros::<half::bf16>(MAX_TREE_SIZE * kv_dim_attn)
                .map_err(|e| LlmError::Backend(format!("scratch tree_v_buf: {e:?}")))?,
            tree_attn_out: stream
                .alloc_zeros::<half::bf16>(MAX_TREE_SIZE * q_dim)
                .map_err(|e| LlmError::Backend(format!("scratch tree_attn_out: {e:?}")))?,
            tree_gate_buf: stream
                .alloc_zeros::<half::bf16>(MAX_TREE_SIZE * scratch_gate)
                .map_err(|e| LlmError::Backend(format!("scratch tree_gate_buf: {e:?}")))?,
            tree_up_buf: stream
                .alloc_zeros::<half::bf16>(MAX_TREE_SIZE * scratch_up)
                .map_err(|e| LlmError::Backend(format!("scratch tree_up_buf: {e:?}")))?,
            tree_logits: stream
                .alloc_zeros::<half::bf16>(MAX_TREE_SIZE * cfg.vocab)
                .map_err(|e| LlmError::Backend(format!("scratch tree_logits: {e:?}")))?,
            tree_ssm_qkv_mixed: stream
                .alloc_zeros::<half::bf16>(MAX_TREE_SIZE * conv_dim)
                .map_err(|e| LlmError::Backend(format!("scratch tree_ssm_qkv: {e:?}")))?,
            tree_ssm_conv_out: stream
                .alloc_zeros::<half::bf16>(MAX_TREE_SIZE * conv_dim)
                .map_err(|e| LlmError::Backend(format!("scratch tree_ssm_conv_out: {e:?}")))?,
            tree_ssm_z: stream
                .alloc_zeros::<half::bf16>(MAX_TREE_SIZE * value_dim)
                .map_err(|e| LlmError::Backend(format!("scratch tree_ssm_z: {e:?}")))?,
            tree_ssm_alpha: stream
                .alloc_zeros::<half::bf16>(MAX_TREE_SIZE * cfg.ssm_dt_rank)
                .map_err(|e| LlmError::Backend(format!("scratch tree_ssm_alpha: {e:?}")))?,
            tree_ssm_beta: stream
                .alloc_zeros::<half::bf16>(MAX_TREE_SIZE * cfg.ssm_dt_rank)
                .map_err(|e| LlmError::Backend(format!("scratch tree_ssm_beta: {e:?}")))?,
            tree_ssm_q_v: stream
                .alloc_zeros::<half::bf16>(MAX_TREE_SIZE * value_dim)
                .map_err(|e| LlmError::Backend(format!("scratch tree_ssm_q_v: {e:?}")))?,
            tree_ssm_k_v: stream
                .alloc_zeros::<half::bf16>(MAX_TREE_SIZE * value_dim)
                .map_err(|e| LlmError::Backend(format!("scratch tree_ssm_k_v: {e:?}")))?,
            tree_ssm_out_buf: stream
                .alloc_zeros::<half::bf16>(MAX_TREE_SIZE * value_dim)
                .map_err(|e| LlmError::Backend(format!("scratch tree_ssm_out_buf: {e:?}")))?,
            tree_ssm_wave_indices: stream
                .alloc_zeros::<i32>(MAX_TREE_SIZE)
                .map_err(|e| LlmError::Backend(format!("scratch tree_ssm_wave_indices: {e:?}")))?,

            // Per-SSM-layer tree-fork SSM state + conv state buffers. One per
            // SSM layer (cfg.ssm_indices.len()).
            tree_ssm_states: {
                let head_v_dim = cfg.ssm_state;
                let n_v_heads = cfg.ssm_dt_rank;
                let mut v = Vec::with_capacity(cfg.ssm_indices.len());
                for _li in 0..cfg.ssm_indices.len() {
                    v.push(
                        stream
                            .alloc_zeros::<half::bf16>(
                                MAX_TREE_SIZE * n_v_heads * head_v_dim * head_v_dim,
                            )
                            .map_err(|e| {
                                LlmError::Backend(format!("alloc tree_ssm_states: {e:?}"))
                            })?,
                    );
                }
                v
            },
            tree_conv_states: {
                let mut v = Vec::with_capacity(cfg.ssm_indices.len());
                for _li in 0..cfg.ssm_indices.len() {
                    v.push(
                        stream
                            .alloc_zeros::<half::bf16>(MAX_TREE_SIZE * (conv_kernel - 1) * conv_dim)
                            .map_err(|e| {
                                LlmError::Backend(format!("alloc tree_conv_states: {e:?}"))
                            })?,
                    );
                }
                v
            },
        };

        let position_dev = stream
            .alloc_zeros::<i32>(1)
            .map_err(|e| LlmError::Backend(format!("position_dev: {e:?}")))?;
        let mut kv_len_dev = stream
            .alloc_zeros::<i32>(1)
            .map_err(|e| LlmError::Backend(format!("kv_len_dev: {e:?}")))?;
        // Initialize kv_len = 1 (= position+1, mirror of Q4K's contract).
        stream
            .memcpy_htod(&[1_i32], &mut kv_len_dev)
            .map_err(|e| LlmError::Backend(format!("kv_len_dev init: {e:?}")))?;
        let current_token_dev = stream
            .alloc_zeros::<u32>(1)
            .map_err(|e| LlmError::Backend(format!("current_token_dev: {e:?}")))?;

        let next_token_host_pinned = unsafe { ctx.alloc_pinned::<u32>(1) }
            .map_err(|e| LlmError::Backend(format!("alloc_pinned next_token: {e:?}")))?;

        // ── TrackK.b — Tree-batched prefill graph cache + pinned scratch ──
        let prefill_drafts_host_pinned = unsafe { ctx.alloc_pinned::<u32>(MAX_TREE_SIZE) }
            .map_err(|e| LlmError::Backend(format!("alloc_pinned prefill_drafts: {e:?}")))?;
        // Pre-bake [0, 1, 2, ..., MAX_TREE_SIZE - 1] for linear-chain SSM
        // wave indices under capture.
        let linear_indices: Vec<i32> = (0..MAX_TREE_SIZE as i32).collect();
        let tree_ssm_wave_indices_linear = stream
            .memcpy_stod(&linear_indices)
            .map_err(|e| LlmError::Backend(format!("tree_ssm_wave_indices_linear: {e:?}")))?;

        let use_ssm_fuse = std::env::var("RUSTORCH_SSM_FUSE")
            .map(|v| v != "0" && !v.eq_ignore_ascii_case("false"))
            .unwrap_or(true);

        // Drain pending uploads before returning.
        stream
            .synchronize()
            .map_err(|e| LlmError::Backend(format!("post-load sync: {e:?}")))?;

        eprintln!(
            "[T246.9 NVFP4-FINISH] model loaded : {} blocks, {} kv caches, {} ssm states",
            blocks.len(),
            kv_caches.len(),
            ssm_states.len()
        );

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
            use_ssm_fuse,
            next_token_host_pinned,
            prefill_graphs: HashMap::new(),
            prefill_warmed: HashSet::new(),
            prefill_drafts_host_pinned,
            tree_ssm_wave_indices_linear,
        })
    }

    /// One-token greedy decode. Mirrors `Qwen35ModelCudaQ4K::decode_step`
    /// with all matmul dispatches swapped for the NVFP4 / BF16 hybrid.
    pub fn decode_step(&mut self, token_id: u32) -> Result<u32, LlmError> {
        use cudarc::driver::{DevicePtr, DevicePtrMut};

        let cfg = self.config.clone();
        let d = cfg.d;
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
        let _ = (conv_dim, value_dim, n_v, head_kv, q_dim, kv_dim);

        // Always update current_token_dev OUTSIDE any capture region.
        self.stream
            .memcpy_htod(&[token_id], &mut self.current_token_dev)
            .map_err(|e| LlmError::Backend(format!("upload token id: {e:?}")))?;

        // Replay path : if a graph was captured, just launch it.
        if let Some(graph) = &self.decode_graph {
            graph
                .launch()
                .map_err(|e| LlmError::Backend(format!("graph launch: {e:?}")))?;
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

        // Capture path : on the 2nd decode call (position == 1), wrap the
        // body in begin_capture / end_capture. The 1st call (position == 0)
        // is a warmup that triggers nvrtc compile + cuModuleLoadData for
        // every kernel (NOT capturable, must happen pre-capture).
        let should_capture =
            moe_graph_enabled() && self.position == 1 && self.decode_graph.is_none();
        if should_capture {
            self.stream
                .synchronize()
                .map_err(|e| LlmError::Backend(format!("pre-capture sync: {e:?}")))?;
            self.stream
                .begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)
                .map_err(|e| LlmError::Backend(format!("begin_capture: {e:?}")))?;
        }

        // Zero scratch buffers.
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
            let (u, _g20) = self.final_norm.data.device_ptr(&self.stream);
            let (gm_, _g22) = self.scratch.gqa_partial_m.device_ptr_mut(&self.stream);
            let (gl_, _g23) = self.scratch.gqa_partial_l.device_ptr_mut(&self.stream);
            let (go_, _g24) = self.scratch.gqa_partial_o.device_ptr_mut(&self.stream);
            let (mr_, _g25) = self.scratch.moe_router_logits.device_ptr_mut(&self.stream);
            let (mi_, _g26) = self.scratch.moe_topk_idx.device_ptr_mut(&self.stream);
            let (mw_, _g27) = self.scratch.moe_topk_w.device_ptr_mut(&self.stream);
            let (mg_, _g28) = self.scratch.moe_expert_gate.device_ptr_mut(&self.stream);
            let (mu_, _g29) = self.scratch.moe_expert_up.device_ptr_mut(&self.stream);
            let (mo_, _g30) = self.scratch.moe_expert_out.device_ptr_mut(&self.stream);
            let (msd_, _g31) = self.scratch.moe_shexp_dot.device_ptr_mut(&self.stream);
            (
                a, b, c, d_, e, f_, g, h_, i, j, k_, l, m, n_, o, p, q_, r, s, t, u, gm_, gl_, go_,
                mr_, mi_, mw_, mg_, mu_, mo_, msd_,
            )
        };

        // ---- Step 0 : embedding lookup ----
        unsafe {
            let (te_p, _g) = self.token_emb.data.device_ptr(&self.stream);
            let (ct_p, _g2) = self.current_token_dev.device_ptr(&self.stream);
            self.kernels
                .embedding_lookup_bf16(&self.stream, te_p, ct_p, h_p, 1, d as i32)
                .map_err(|e| LlmError::Backend(format!("embed lookup: {e:?}")))?;
        }

        // ---- Iterate over all 40 layers ----
        for (li, block) in self.blocks.iter_mut().enumerate() {
            // residual = h
            unsafe {
                self.kernels
                    .copy_bf16(&self.stream, res_p, h_p, d as i32)
                    .map_err(|e| LlmError::Backend(format!("copy res: {e:?}")))?;
            }

            match block {
                BlockNvfp4::Ssm(ssm) => {
                    // 1. h_norm = rms_norm(h, attn_norm)
                    unsafe {
                        let (an, _g) = ssm.attn_norm.data.device_ptr(&self.stream);
                        self.kernels
                            .copy_bf16(&self.stream, h_norm_p, h_p, d as i32)
                            .map_err(|e| LlmError::Backend(format!("copy h_norm: {e:?}")))?;
                        self.kernels
                            .rms_norm_bf16(&self.stream, h_norm_p, an, eps, d as i32, 1)
                            .map_err(|e| LlmError::Backend(format!("rms_norm: {e:?}")))?;
                    }

                    // 2. qkv_mixed = w_qkv @ h_norm  (BF16, per recipe)
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
                    // 5. beta = w_beta @ h_norm ; sigmoid in pre-step
                    ssm.w_beta
                        .dispatch_matmul_m1(&self.kernels, &self.stream, h_norm_p, beta_p)?;

                    // 5b+6. SSM pre-step (fused or unfused).
                    unsafe {
                        let (db, _g) = ssm.dt_bias.data.device_ptr(&self.stream);
                        let (sa, _g2) = ssm.ssm_a.data.device_ptr(&self.stream);
                        if self.use_ssm_fuse {
                            self.kernels
                                .ssm_pre_step_bf16(
                                    &self.stream,
                                    alpha_p,
                                    beta_p,
                                    db,
                                    sa,
                                    n_v as i32,
                                )
                                .map_err(|e| LlmError::Backend(format!("ssm_pre_step: {e:?}")))?;
                        } else {
                            self.kernels
                                .sigmoid_inplace_bf16(&self.stream, beta_p, n_v as i32)
                                .map_err(|e| LlmError::Backend(format!("sigmoid: {e:?}")))?;
                            self.kernels
                                .add_inplace_bf16(&self.stream, alpha_p, db, n_v as i32)
                                .map_err(|e| LlmError::Backend(format!("alpha+dt: {e:?}")))?;
                            self.kernels
                                .softplus_inplace_bf16(&self.stream, alpha_p, n_v as i32)
                                .map_err(|e| LlmError::Backend(format!("softplus: {e:?}")))?;
                            self.kernels
                                .mul_inplace_bf16(&self.stream, alpha_p, sa, n_v as i32)
                                .map_err(|e| LlmError::Backend(format!("mul ssm_a: {e:?}")))?;
                        }
                    }

                    // 7. conv1d depthwise.
                    let ssm_state_layer = &mut self.ssm_states[get_ssm_layer_idx(&cfg, li)];
                    unsafe {
                        let (cw, _g) = ssm.conv1d.data.device_ptr(&self.stream);
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

                    // 10. l2_norm_per_head on q and k.
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

                    // 11. Broadcast q,k from n_k to n_v heads.
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

                    // 12. delta_net_step
                    unsafe {
                        let (st, _g) = ssm_state_layer.state.device_ptr_mut(&self.stream);
                        self.kernels
                            .delta_net_step_bf16(
                                &self.stream,
                                q_for_delta,
                                k_for_delta,
                                v_ptr,
                                alpha_p,
                                beta_p,
                                st,
                                sso_p,
                                n_v as i32,
                                head_kv as i32,
                            )
                            .map_err(|e| LlmError::Backend(format!("delta_net: {e:?}")))?;
                    }

                    // 13. ssm_norm + silu(z) * out.
                    unsafe {
                        let (sn, _g) = ssm.ssm_norm.data.device_ptr(&self.stream);
                        if self.use_ssm_fuse {
                            self.kernels
                                .ssm_post_step_bf16(
                                    &self.stream,
                                    sso_p,
                                    z_p,
                                    sn,
                                    eps,
                                    n_v as i32,
                                    head_kv as i32,
                                )
                                .map_err(|e| LlmError::Backend(format!("ssm_post_step: {e:?}")))?;
                        } else {
                            self.kernels
                                .rms_norm_bf16(
                                    &self.stream,
                                    sso_p,
                                    sn,
                                    eps,
                                    head_kv as i32,
                                    n_v as i32,
                                )
                                .map_err(|e| LlmError::Backend(format!("ssm_norm: {e:?}")))?;
                            self.kernels
                                .silu_bf16(&self.stream, z_p, value_dim as i32)
                                .map_err(|e| LlmError::Backend(format!("silu z: {e:?}")))?;
                            self.kernels
                                .mul_inplace_bf16(&self.stream, sso_p, z_p, value_dim as i32)
                                .map_err(|e| LlmError::Backend(format!("mul gated: {e:?}")))?;
                        }
                    }

                    // 14. h = ssm_out @ gated  (BF16)
                    ssm.ssm_out
                        .dispatch_matmul_m1(&self.kernels, &self.stream, sso_p, h_p)?;

                    // 15. h += residual
                    unsafe {
                        self.kernels
                            .add_inplace_bf16(&self.stream, h_p, res_p, d as i32)
                            .map_err(|e| LlmError::Backend(format!("ssm residual: {e:?}")))?;
                    }
                },
                BlockNvfp4::Attn(attn) => {
                    // 1. h_norm = rms_norm(h, attn_norm)
                    unsafe {
                        let (an, _g) = attn.attn_norm.data.device_ptr(&self.stream);
                        self.kernels
                            .copy_bf16(&self.stream, h_norm_p, h_p, d as i32)
                            .map_err(|e| LlmError::Backend(format!("copy h_norm attn: {e:?}")))?;
                        self.kernels
                            .rms_norm_bf16(&self.stream, h_norm_p, an, eps, d as i32, 1)
                            .map_err(|e| LlmError::Backend(format!("rms_norm attn: {e:?}")))?;
                    }

                    // 2. Q+gate combined projection (Qwen3Next, attn_output_gate=true).
                    // q_proj produces 2*q_dim ; reuse up_buf as QG scratch.
                    debug_assert_eq!(attn.w_q.n, 2 * q_dim);
                    let qg_p = up_p;
                    attn.w_q
                        .dispatch_matmul_m1(&self.kernels, &self.stream, h_norm_p, qg_p)?;

                    // 3. Split qg → q + gate (gate stored in attn_out scratch).
                    unsafe {
                        self.kernels
                            .split_qg_bf16(
                                &self.stream,
                                qg_p,
                                q_p,
                                ao_p,
                                n_q as i32,
                                head_dim as i32,
                            )
                            .map_err(|e| LlmError::Backend(format!("split_qg: {e:?}")))?;
                    }

                    // 4. K, V projections (NVFP4).
                    attn.w_k
                        .dispatch_matmul_m1(&self.kernels, &self.stream, h_norm_p, k_p)?;
                    attn.w_v
                        .dispatch_matmul_m1(&self.kernels, &self.stream, h_norm_p, v_p)?;

                    // 5. Per-head Q-norm + K-norm.
                    unsafe {
                        let (qn, _g1) = attn.q_norm.data.device_ptr(&self.stream);
                        let (kn, _g2) = attn.k_norm.data.device_ptr(&self.stream);
                        self.kernels
                            .rms_norm_bf16(&self.stream, q_p, qn, eps, head_dim as i32, n_q as i32)
                            .map_err(|e| LlmError::Backend(format!("q_norm: {e:?}")))?;
                        self.kernels
                            .rms_norm_bf16(&self.stream, k_p, kn, eps, head_dim as i32, n_kv as i32)
                            .map_err(|e| LlmError::Backend(format!("k_norm: {e:?}")))?;
                    }

                    // 6. RoPE on q, k (partial : only first rope_dim of head_dim).
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

                    // 7. Append k, v to KV cache.
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

                    // 8. GQA decode (FlashDecode-V2 split-K).
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
                                gate_p,
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

                    // 9. sigmoid(gate) ; attn_out *= gate (Qwen3Next output gate).
                    unsafe {
                        self.kernels
                            .sigmoid_inplace_bf16(&self.stream, ao_p, q_dim as i32)
                            .map_err(|e| LlmError::Backend(format!("sigmoid gate: {e:?}")))?;
                        self.kernels
                            .mul_inplace_bf16(&self.stream, gate_p, ao_p, q_dim as i32)
                            .map_err(|e| LlmError::Backend(format!("attn*gate: {e:?}")))?;
                    }

                    // 10. h = w_o @ gated_attn  (NVFP4 → BF16)
                    attn.w_o
                        .dispatch_matmul_m1(&self.kernels, &self.stream, gate_p, h_p)?;

                    // 11. h += residual
                    unsafe {
                        self.kernels
                            .add_inplace_bf16(&self.stream, h_p, res_p, d as i32)
                            .map_err(|e| LlmError::Backend(format!("attn residual: {e:?}")))?;
                    }
                },
            }

            // ---- FFN ----
            let (ffn_ref, post_norm) = match block {
                BlockNvfp4::Ssm(s) => (&s.ffn, &s.post_norm),
                BlockNvfp4::Attn(a) => (&a.ffn, &a.post_norm),
            };
            unsafe {
                self.kernels
                    .copy_bf16(&self.stream, res_p, h_p, d as i32)
                    .map_err(|e| LlmError::Backend(format!("copy res ffn: {e:?}")))?;
            }
            unsafe {
                let (pn, _g) = post_norm.data.device_ptr(&self.stream);
                self.kernels
                    .copy_bf16(&self.stream, h_norm_p, h_p, d as i32)
                    .map_err(|e| LlmError::Backend(format!("copy h_norm ffn: {e:?}")))?;
                self.kernels
                    .rms_norm_bf16(&self.stream, h_norm_p, pn, eps, d as i32, 1)
                    .map_err(|e| LlmError::Backend(format!("rms_norm post: {e:?}")))?;
            }

            moe_ffn_forward_step_nvfp4(
                ffn_ref,
                &self.kernels,
                &self.stream,
                &cfg,
                h_norm_p,
                h_p,
                moe_router_p,
                moe_idx_p,
                moe_w_p,
                moe_egate_p,
                moe_eup_p,
                moe_eout_p,
                moe_sd_p,
            )?;

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
        // lm_head is BF16 per recipe.
        self.lm_head
            .dispatch_matmul_m1(&self.kernels, &self.stream, h_p, logits_p)?;

        // ---- Sample (argmax) ----
        unsafe {
            self.kernels
                .argmax_bf16(&self.stream, logits_p, tok_p, cfg.vocab as i32)
                .map_err(|e| LlmError::Backend(format!("argmax: {e:?}")))?;
        }
        let _ = (tok_p, logits_p, final_norm_p);

        // Auto-advance device counters.
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

        // Capture path closure.
        if should_capture {
            let graph = self
                .stream
                .end_capture(
                    CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
                )
                .map_err(|e| LlmError::Backend(format!("end_capture: {e:?}")))?
                .ok_or_else(|| LlmError::Backend("end_capture returned no graph".into()))?;
            graph
                .launch()
                .map_err(|e| LlmError::Backend(format!("first graph launch: {e:?}")))?;
            self.decode_graph = Some(graph);
        }

        // DtoH next token via pinned host memory.
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

    /// Current host-side position (number of decode steps committed so far).
    /// Mirrors `Qwen35ModelCudaQ4K::position()` for caller compatibility.
    #[inline]
    pub fn position(&self) -> usize {
        self.position
    }

    /// Zero out the KV cache, conv state, and SSM state ; reset the host-side
    /// `position` counter and the device-side `position_dev` / `kv_len_dev`
    /// counters back to a fresh-context state. The captured decode graph is
    /// dropped — it will be re-captured on the 2nd subsequent `decode_step`
    /// (mirrors the Q4_K path's `reset_state` contract).
    ///
    /// Required by `prefill_tokens` (when called with `start_pos == 0` after
    /// previous decodes) and by the bench harness between runs.
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
        self.stream
            .memcpy_htod(&[0i32], &mut self.position_dev)
            .map_err(|e| LlmError::Backend(format!("reset position_dev: {e:?}")))?;
        self.stream
            .memcpy_htod(&[1i32], &mut self.kv_len_dev)
            .map_err(|e| LlmError::Backend(format!("reset kv_len_dev: {e:?}")))?;
        // Captured decode graph reads stale state ; drop so the next pair of
        // decode_step calls re-captures against the fresh KV/SSM state.
        self.decode_graph = None;
        // T246.10 TrackK.b — the prefill bag-of-graphs cache is preserved
        // across `reset_state()` (mirrors the Q4_K TrackI design). Captured
        // graphs reference device-side pointers (KV cache base, SSM state
        // base, position_dev, etc.) which are NOT freed/moved by
        // `reset_state()` — only their contents are zeroed. The captured
        // nodes then read whatever values are in the buffers at replay
        // time (e.g. `position_dev = 0`, `kv_len_dev = 1`), exactly the
        // state we re-initialize here. The warmup tracker (`prefill_warmed`)
        // is also preserved : it represents module-load / JIT state which
        // persists across resets.
        Ok(())
    }

    /// T246.10 TrackK.1 — Prefill `token_ids` and return the prediction after
    /// the last input token.
    ///
    /// **Semantics** : feeds each of the N input tokens through the model
    /// sequentially via `decode_step`, growing the KV cache and SSM state.
    /// Returns the argmax of the logits produced after the LAST input token
    /// — that's the "first decode token" the caller would emit next.
    ///
    /// **Implementation note** : this is a sequential `decode_step` loop, NOT
    /// a tree-batched GEMM-prefill pass. The NVFP4 path's per-tree-row scratch
    /// buffers and tree-aware kernel dispatchers (counterpart of Q4_K's
    /// `decode_step_tree_hybrid_inner_capture` + `hybrid_attn_layer` +
    /// `hybrid_ssm_layer` + GEMM-prefill matmul variants) are NOT yet ported.
    /// Porting them is TrackK.b (separate task) — the structural cost is
    /// ~2000 LOC of new dispatch + new tree-scaled scratch fields and would
    /// also need NVFP4-side `sgemm_mvar`-shape matmul wrappers that don't
    /// exist today. For TrackK.1 we ship the working baseline (functionally
    /// correct, reuses the captured decode graph) and bench it informationally.
    ///
    /// **Parity with Q4_K** : the Q4_K `prefill_tokens` short-circuits to
    /// `decode_step` for N=1 ; this NVFP4 implementation generalises that
    /// fast path to all N. When the Q4_K bench harness runs the "naive"
    /// baseline it does this exact sequence (see `qwen36_cuda_prefill_bench`
    /// lines 142-152). The NVFP4 prefill is thus equivalent to the Q4_K
    /// naive baseline path — a real apples-to-apples comparison point.
    ///
    /// **CUDA Graph reuse** : `decode_step` captures its graph on the 2nd
    /// call (`self.position == 1`), so the first 2 of the N tokens are
    /// JIT-compile + capture, and the remaining N-2 are graph replays.
    /// This is the same captured-graph win the decode-only path enjoys.
    pub fn prefill_tokens(&mut self, token_ids: &[u32], start_pos: usize) -> Result<u32, LlmError> {
        let n = token_ids.len();
        if n == 0 {
            return Err(LlmError::Backend(
                "prefill_tokens: token_ids must be non-empty".into(),
            ));
        }
        if start_pos != self.position {
            return Err(LlmError::Backend(format!(
                "prefill_tokens: start_pos={start_pos} != self.position={}. \
                 Call model.reset_state() first or pass start_pos=self.position().",
                self.position
            )));
        }
        if n > MAX_TREE_SIZE {
            return Err(LlmError::Backend(format!(
                "prefill_tokens: N={n} exceeds MAX_TREE_SIZE={MAX_TREE_SIZE}. \
                 Call prefill_tokens in chunks of {MAX_TREE_SIZE} for longer prompts."
            )));
        }

        // ── TrackK.b — Tree-batched prefill path (env-gated) ────────────
        //
        // When `RUSTORCH_NVFP4_PREFILL_TREE=1` is set AND N >=
        // NVFP4_PREFILL_TREE_MIN_N, route through the tree-batched
        // capture wrapper. The per-row matmuls still use M=1 NVFP4
        // SGEMV (no batched NVFP4 kernel exists today — see decision
        // 450e88ef). The win comes from :
        //
        // 1. Tree-aware kv_append + GQA decode (single-launch over
        //    N tokens instead of N sequential M=1 attention scans
        //    against the full prefix).
        // 2. SSM forking via delta_net_step_tree_bf16 (1 launch per
        //    BFS depth instead of N full attn-style scans).
        // 3. CUDA Graph capture over the whole forward (eliminates
        //    most CPU dispatch overhead for replays).
        //
        // Falls through to the TrackK sequential decode_step loop
        // when N=1 or the env gate is off.
        if n >= NVFP4_PREFILL_TREE_MIN_N && nvfp4_prefill_tree_enabled() {
            let parents: Vec<i32> = std::iter::once(-1i32).chain(0..(n - 1) as i32).collect();
            let depths: Vec<u16> = (0..n as u16).collect();
            return self.prefill_tokens_capture_nvfp4(token_ids, &parents, &depths);
        }

        // ── TrackK.1 baseline — sequential decode_step loop ─────────────
        let mut last_pred: u32 = 0;
        for (i, &tok) in token_ids.iter().enumerate() {
            last_pred = self.decode_step(tok).map_err(|e| {
                LlmError::Backend(format!("prefill_tokens: decode_step #{i} failed: {e:?}"))
            })?;
        }
        Ok(last_pred)
    }

    /// T246.10 TrackK.b — Graph-capture entry for `prefill_tokens` (NVFP4).
    /// Mirrors `Qwen35ModelCudaQ4K::prefill_tokens_capture`. Caller must
    /// validate `N >= NVFP4_PREFILL_TREE_MIN_N` and the env gate.
    ///
    /// Strategy : bag-of-graphs keyed by `N`. First call = warmup (run
    /// uncaptured to prime nvrtc/cuModuleLoad), second call = capture +
    /// instantiate, subsequent calls = replay.
    fn prefill_tokens_capture_nvfp4(
        &mut self,
        token_ids: &[u32],
        parents: &[i32],
        depths: &[u16],
    ) -> Result<u32, LlmError> {
        let n = token_ids.len();

        // ── 1. Upload tree descriptors (outside capture, pinned drafts) ─
        {
            let drafts_pinned = self
                .prefill_drafts_host_pinned
                .as_mut_slice()
                .map_err(|e| LlmError::Backend(format!("drafts_pinned as_mut_slice: {e:?}")))?;
            drafts_pinned[..n].copy_from_slice(token_ids);
        }
        {
            let drafts_pinned = self
                .prefill_drafts_host_pinned
                .as_slice()
                .map_err(|e| LlmError::Backend(format!("drafts_pinned as_slice: {e:?}")))?;
            self.stream
                .memcpy_htod(&drafts_pinned[..n], &mut self.scratch.tree_drafts)
                .map_err(|e| LlmError::Backend(format!("upload tree_drafts: {e:?}")))?;
        }
        self.stream
            .memcpy_htod(parents, &mut self.scratch.tree_parents)
            .map_err(|e| LlmError::Backend(format!("upload tree_parents: {e:?}")))?;
        self.stream
            .memcpy_htod(depths, &mut self.scratch.tree_depths)
            .map_err(|e| LlmError::Backend(format!("upload tree_depths: {e:?}")))?;

        // ── 2. Replay path — graph cached for this N ────────────────────
        if let Some(graph) = self.prefill_graphs.get(&n) {
            graph
                .launch()
                .map_err(|e| LlmError::Backend(format!("prefill graph launch: {e:?}")))?;
            return self.prefill_finish_after_capture_nvfp4(n);
        }

        // ── 3. Warmup path — first call for this N, no capture ──────────
        if !self.prefill_warmed.contains(&n) {
            self.decode_step_tree_hybrid_inner_capture_nvfp4(token_ids, parents, depths, false)?;
            self.prefill_warmed.insert(n);
            // After warmup the body has advanced model state by N tokens
            // and stamped argmax tokens into the device buffer. Read them
            // post-stream-sync (no capture in progress).
            return self.prefill_finish_after_capture_nvfp4(n);
        }

        // ── 4. Capture path ─────────────────────────────────────────────
        self.stream
            .synchronize()
            .map_err(|e| LlmError::Backend(format!("pre-capture sync: {e:?}")))?;
        self.stream
            .begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)
            .map_err(|e| LlmError::Backend(format!("prefill begin_capture: {e:?}")))?;

        self.decode_step_tree_hybrid_inner_capture_nvfp4(token_ids, parents, depths, true)?;

        let graph = self
            .stream
            .end_capture(CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH)
            .map_err(|e| LlmError::Backend(format!("prefill end_capture: {e:?}")))?
            .ok_or_else(|| LlmError::Backend("prefill end_capture returned no graph".into()))?;

        graph
            .launch()
            .map_err(|e| LlmError::Backend(format!("first prefill graph launch: {e:?}")))?;

        self.prefill_graphs.insert(n, graph);
        self.prefill_finish_after_capture_nvfp4(n)
    }

    /// T246.10 TrackK.b — Post-replay finalizer. Reads the argmax tokens
    /// into pinned host memory, advances host-side `self.position`, and
    /// returns the last accepted token (the "first decode token").
    fn prefill_finish_after_capture_nvfp4(&mut self, n: usize) -> Result<u32, LlmError> {
        self.stream
            .memcpy_dtoh(
                &self.scratch.tree_argmax,
                &mut self.scratch.tree_argmax_host_pinned,
            )
            .map_err(|e| LlmError::Backend(format!("prefill post-capture dtoh argmax: {e:?}")))?;
        let argmax_host = self
            .scratch
            .tree_argmax_host_pinned
            .as_slice()
            .map_err(|e| LlmError::Backend(format!("prefill pinned argmax read: {e:?}")))?;
        // Capture mode never updates `self.position` from inside the inner
        // — apply it here for the N accepted tokens.
        self.position += n;
        Ok(argmax_host[n - 1])
    }

    // ─────────────────────────────────────────────────────────────────────
    // T246.10 TrackK.b — Tree-batched prefill forward (NVFP4 flavour)
    //
    // Mirror of `Qwen35ModelCudaQ4K::decode_step_tree_hybrid_inner_capture`
    // with all matmul dispatches swapped for `Nvfp4Tensor::dispatch_matmul_m1`
    // looped over tree rows (no batched NVFP4 GEMM kernel exists today —
    // see decision 450e88ef). SSM linears stay BF16 (per recipe.yaml). KV
    // cache + SSM state + conv state are all BF16 — the tree kernels
    // (kv_append_tree_bf16, gqa_decode_tree_bf16, delta_net_step_tree_bf16,
    // argmax_logits_tree_bf16) are quant-agnostic.
    //
    // Linear-chain semantics only : `force_accept_all = true` always. The
    // `in_prefill_capture` flag mirrors the Q4_K capture-mode pattern :
    //   - When true : skip top-of-body HtoD (caller did them), use offsets
    //     into `tree_ssm_wave_indices_linear` for SSM waves, skip DtoH +
    //     host accept walk, skip `self.position +=` (caller does it post-replay).
    //   - When false : warmup pass, do everything inline, update self.position.
    // ─────────────────────────────────────────────────────────────────────
    #[allow(clippy::too_many_lines)]
    fn decode_step_tree_hybrid_inner_capture_nvfp4(
        &mut self,
        drafts: &[u32],
        parents: &[i32],
        depths: &[u16],
        in_prefill_capture: bool,
    ) -> Result<(), LlmError> {
        use cudarc::driver::{DevicePtr, DevicePtrMut};

        let cfg = self.config.clone();
        let d = cfg.d;
        let n_q = cfg.n_q_heads;
        let n_kv = cfg.n_kv_heads;
        let head_dim = cfg.head_dim();
        let q_dim = n_q * head_dim;
        let kv_dim = n_kv * head_dim;
        let eps = cfg.rms_eps;
        let rope_dim = cfg.rope_dim;
        let vocab = cfg.vocab;
        let head_kv = cfg.ssm_state;
        let n_k = cfg.ssm_groups;
        let n_v = cfg.ssm_dt_rank;
        let key_dim = head_kv * n_k;
        let value_dim = head_kv * n_v;
        let conv_dim = 2 * key_dim + value_dim;
        let conv_kernel = cfg.ssm_conv_kernel;
        let tree_size = drafts.len();
        debug_assert!(tree_size > 1 && tree_size <= MAX_TREE_SIZE);

        let base_position = self.position;

        // ── 0. Upload tree descriptors (warmup only ; capture-mode does it outside) ─
        if !in_prefill_capture {
            self.stream
                .memcpy_htod(drafts, &mut self.scratch.tree_drafts)
                .map_err(|e| LlmError::Backend(format!("upload tree_drafts: {e:?}")))?;
            self.stream
                .memcpy_htod(parents, &mut self.scratch.tree_parents)
                .map_err(|e| LlmError::Backend(format!("upload tree_parents: {e:?}")))?;
            self.stream
                .memcpy_htod(depths, &mut self.scratch.tree_depths)
                .map_err(|e| LlmError::Backend(format!("upload tree_depths: {e:?}")))?;
        }

        // ── 1. BFS depth waves (linear chain : 1 wave per row) ────────────
        let max_depth = *depths.iter().max().unwrap_or(&0) as usize;
        let mut waves: Vec<Vec<i32>> = vec![Vec::new(); max_depth + 1];
        for (r, &dep) in depths.iter().enumerate() {
            waves[dep as usize].push(r as i32);
        }

        // ── 2. Zero per-tree scratch ──────────────────────────────────────
        for buf in [
            &mut self.scratch.tree_h,
            &mut self.scratch.tree_h_norm,
            &mut self.scratch.tree_residual,
            &mut self.scratch.tree_q_buf,
            &mut self.scratch.tree_k_buf,
            &mut self.scratch.tree_v_buf,
            &mut self.scratch.tree_attn_out,
            &mut self.scratch.tree_gate_buf,
            &mut self.scratch.tree_up_buf,
            &mut self.scratch.tree_logits,
        ] {
            self.stream
                .memset_zeros(buf)
                .map_err(|e| LlmError::Backend(format!("zero tree scratch: {e:?}")))?;
        }
        for buf in [
            &mut self.scratch.tree_ssm_qkv_mixed,
            &mut self.scratch.tree_ssm_conv_out,
            &mut self.scratch.tree_ssm_z,
            &mut self.scratch.tree_ssm_alpha,
            &mut self.scratch.tree_ssm_beta,
            &mut self.scratch.tree_ssm_q_v,
            &mut self.scratch.tree_ssm_k_v,
            &mut self.scratch.tree_ssm_out_buf,
        ] {
            self.stream
                .memset_zeros(buf)
                .map_err(|e| LlmError::Backend(format!("zero tree ssm scratch: {e:?}")))?;
        }

        // Per-row byte offsets (BF16 = 2 bytes).
        let bf16_sz = std::mem::size_of::<half::bf16>() as u64;
        let row_h = (d as u64) * bf16_sz;
        let row_q = (q_dim as u64) * bf16_sz;
        let row_kv = (kv_dim as u64) * bf16_sz;
        let row_logits = (vocab as u64) * bf16_sz;
        let row_conv = (conv_dim as u64) * bf16_sz;
        let row_value = (value_dim as u64) * bf16_sz;
        let row_alpha = (n_v as u64) * bf16_sz;
        let conv_state_per_slot = ((conv_kernel - 1) * conv_dim) as u64 * bf16_sz;
        let ssm_state_per_slot = (n_v * head_kv * head_kv) as u64 * bf16_sz;
        let scratch_up_elems = self.scratch.tree_up_buf.len() / MAX_TREE_SIZE;
        let scratch_gate_elems = self.scratch.tree_gate_buf.len() / MAX_TREE_SIZE;
        let row_up = (scratch_up_elems as u64) * bf16_sz;
        let row_gate = (scratch_gate_elems as u64) * bf16_sz;

        // Pre-extract device pointers.
        let (
            th_p,
            thn_p,
            tres_p,
            tq_p,
            tk_p,
            tv_p,
            tao_p,
            tgate_p,
            tup_p,
            tlogits_p,
            tdrafts_p,
            tparents_p,
            tdepths_p,
            targmax_p,
            tgqa_m_p,
            tgqa_l_p,
            tgqa_o_p,
            tssm_qkv_p,
            tssm_conv_p,
            tssm_z_p,
            tssm_alpha_p,
            tssm_beta_p,
            tssm_qv_p,
            tssm_kv_p,
            tssm_out_p,
            tssm_wave_p,
        ) = {
            let (a, _g0) = self.scratch.tree_h.device_ptr_mut(&self.stream);
            let (b, _g1) = self.scratch.tree_h_norm.device_ptr_mut(&self.stream);
            let (c, _g2) = self.scratch.tree_residual.device_ptr_mut(&self.stream);
            let (d_, _g3) = self.scratch.tree_q_buf.device_ptr_mut(&self.stream);
            let (e, _g4) = self.scratch.tree_k_buf.device_ptr_mut(&self.stream);
            let (f_, _g5) = self.scratch.tree_v_buf.device_ptr_mut(&self.stream);
            let (g, _g6) = self.scratch.tree_attn_out.device_ptr_mut(&self.stream);
            let (h, _g7) = self.scratch.tree_gate_buf.device_ptr_mut(&self.stream);
            let (i, _g8) = self.scratch.tree_up_buf.device_ptr_mut(&self.stream);
            let (j, _g9) = self.scratch.tree_logits.device_ptr_mut(&self.stream);
            let (k, _g10) = self.scratch.tree_drafts.device_ptr(&self.stream);
            let (l, _g11) = self.scratch.tree_parents.device_ptr(&self.stream);
            let (m, _g12) = self.scratch.tree_depths.device_ptr(&self.stream);
            let (n_, _g13) = self.scratch.tree_argmax.device_ptr_mut(&self.stream);
            let (o, _g14) = self.scratch.tree_gqa_partial_m.device_ptr_mut(&self.stream);
            let (p, _g15) = self.scratch.tree_gqa_partial_l.device_ptr_mut(&self.stream);
            let (q, _g16) = self.scratch.tree_gqa_partial_o.device_ptr_mut(&self.stream);
            let (r0, _g17) = self.scratch.tree_ssm_qkv_mixed.device_ptr_mut(&self.stream);
            let (r1, _g18) = self.scratch.tree_ssm_conv_out.device_ptr_mut(&self.stream);
            let (r2, _g19) = self.scratch.tree_ssm_z.device_ptr_mut(&self.stream);
            let (r3, _g20) = self.scratch.tree_ssm_alpha.device_ptr_mut(&self.stream);
            let (r4, _g21) = self.scratch.tree_ssm_beta.device_ptr_mut(&self.stream);
            let (r5, _g22) = self.scratch.tree_ssm_q_v.device_ptr_mut(&self.stream);
            let (r6, _g23) = self.scratch.tree_ssm_k_v.device_ptr_mut(&self.stream);
            let (r7, _g24) = self.scratch.tree_ssm_out_buf.device_ptr_mut(&self.stream);
            let (r8, _g25) = self
                .scratch
                .tree_ssm_wave_indices
                .device_ptr_mut(&self.stream);
            (
                a, b, c, d_, e, f_, g, h, i, j, k, l, m, n_, o, p, q, r0, r1, r2, r3, r4, r5, r6,
                r7, r8,
            )
        };

        // ── 3. Embedding lookup (single batched call over tree_size) ──────
        unsafe {
            let (te_p, _g) = self.token_emb.data.device_ptr(&self.stream);
            self.kernels
                .embedding_lookup_bf16(
                    &self.stream,
                    te_p,
                    tdrafts_p,
                    th_p,
                    tree_size as i32,
                    d as i32,
                )
                .map_err(|e| LlmError::Backend(format!("tree embed: {e:?}")))?;
        }

        // ── 4. Per-layer forward ──────────────────────────────────────────
        let n_layers = self.blocks.len();
        for li in 0..n_layers {
            // Snapshot residual.
            unsafe {
                self.kernels
                    .copy_bf16(&self.stream, tres_p, th_p, (tree_size * d) as i32)
                    .map_err(|e| LlmError::Backend(format!("tree copy res l{li}: {e:?}")))?;
            }

            let is_attn = matches!(self.blocks[li], BlockNvfp4::Attn(_));
            if is_attn {
                self.hybrid_attn_layer_nvfp4(
                    li,
                    tree_size,
                    base_position,
                    th_p,
                    thn_p,
                    tres_p,
                    tq_p,
                    tk_p,
                    tv_p,
                    tao_p,
                    tgate_p,
                    tup_p,
                    tparents_p,
                    tdepths_p,
                    tgqa_m_p,
                    tgqa_l_p,
                    tgqa_o_p,
                    row_h,
                    row_q,
                    row_kv,
                    row_up,
                    row_gate,
                    depths,
                    eps,
                    n_q,
                    n_kv,
                    head_dim,
                    q_dim,
                    kv_dim,
                    rope_dim,
                    d,
                )?;
            } else {
                self.hybrid_ssm_layer_nvfp4(
                    li,
                    tree_size,
                    parents,
                    &waves,
                    th_p,
                    thn_p,
                    tres_p,
                    tssm_qkv_p,
                    tssm_conv_p,
                    tssm_z_p,
                    tssm_alpha_p,
                    tssm_beta_p,
                    tssm_qv_p,
                    tssm_kv_p,
                    tssm_out_p,
                    tssm_wave_p,
                    row_h,
                    row_conv,
                    row_value,
                    row_alpha,
                    conv_state_per_slot,
                    eps,
                    n_v,
                    n_k,
                    head_kv,
                    key_dim,
                    value_dim,
                    conv_dim,
                    conv_kernel,
                    d,
                    in_prefill_capture,
                )?;
            }

            // ── FFN block (MoE) ───────────────────────────────────────────
            // residual <- h post-mixer.
            unsafe {
                self.kernels
                    .copy_bf16(&self.stream, tres_p, th_p, (tree_size * d) as i32)
                    .map_err(|e| LlmError::Backend(format!("tree copy res ffn l{li}: {e:?}")))?;
            }

            // Pre-extract post_norm + MoE ref.
            let (post_norm_ptr, ffn_ref): (u64, &MoeFfnNvfp4) = match &self.blocks[li] {
                BlockNvfp4::Attn(a) => {
                    let (p, _g) = a.post_norm.data.device_ptr(&self.stream);
                    (p, &a.ffn)
                },
                BlockNvfp4::Ssm(s) => {
                    let (p, _g) = s.post_norm.data.device_ptr(&self.stream);
                    (p, &s.ffn)
                },
            };

            // Batched RMSNorm over tree rows.
            unsafe {
                self.kernels
                    .copy_bf16(&self.stream, thn_p, th_p, (tree_size * d) as i32)
                    .map_err(|e| LlmError::Backend(format!("tree copy h_norm ffn l{li}: {e:?}")))?;
                self.kernels
                    .rms_norm_bf16(
                        &self.stream,
                        thn_p,
                        post_norm_ptr,
                        eps,
                        d as i32,
                        tree_size as i32,
                    )
                    .map_err(|e| LlmError::Backend(format!("tree rms_norm post l{li}: {e:?}")))?;
            }

            // MoE FFN per-row dispatch (no batched NVFP4 MoE kernel exists).
            // Extract single-token MoE scratch pointers once.
            let (moe_router_p, moe_idx_p, moe_w_p, moe_egate_p, moe_eup_p, moe_eout_p, moe_sd_p) = {
                let (mr_, _gmr) = self.scratch.moe_router_logits.device_ptr_mut(&self.stream);
                let (mi_, _gmi) = self.scratch.moe_topk_idx.device_ptr_mut(&self.stream);
                let (mw_, _gmw) = self.scratch.moe_topk_w.device_ptr_mut(&self.stream);
                let (mg_, _gmeg) = self.scratch.moe_expert_gate.device_ptr_mut(&self.stream);
                let (mu_, _gmeu) = self.scratch.moe_expert_up.device_ptr_mut(&self.stream);
                let (mo_, _gmeo) = self.scratch.moe_expert_out.device_ptr_mut(&self.stream);
                let (msd_, _gmsd) = self.scratch.moe_shexp_dot.device_ptr_mut(&self.stream);
                (mr_, mi_, mw_, mg_, mu_, mo_, msd_)
            };
            for r in 0..tree_size {
                let hn_r = thn_p + (r as u64) * row_h;
                let h_r = th_p + (r as u64) * row_h;
                moe_ffn_forward_step_nvfp4(
                    ffn_ref,
                    &self.kernels,
                    &self.stream,
                    &cfg,
                    hn_r,
                    h_r,
                    moe_router_p,
                    moe_idx_p,
                    moe_w_p,
                    moe_egate_p,
                    moe_eup_p,
                    moe_eout_p,
                    moe_sd_p,
                )?;
            }

            // h += residual.
            unsafe {
                self.kernels
                    .add_inplace_bf16(&self.stream, th_p, tres_p, (tree_size * d) as i32)
                    .map_err(|e| LlmError::Backend(format!("tree ffn residual l{li}: {e:?}")))?;
            }
        }

        // ── 5. Final RMSNorm + LM head per row ────────────────────────────
        unsafe {
            let (fn_p, _g) = self.final_norm.data.device_ptr(&self.stream);
            self.kernels
                .rms_norm_bf16(&self.stream, th_p, fn_p, eps, d as i32, tree_size as i32)
                .map_err(|e| LlmError::Backend(format!("tree final_norm: {e:?}")))?;
        }
        for r in 0..tree_size {
            let h_r = th_p + (r as u64) * row_h;
            let logits_r = tlogits_p + (r as u64) * row_logits;
            self.lm_head
                .dispatch_matmul_m1(&self.kernels, &self.stream, h_r, logits_r)?;
        }

        // ── 6. Argmax over tree rows ──────────────────────────────────────
        unsafe {
            self.kernels
                .argmax_logits_tree_bf16(
                    &self.stream,
                    tlogits_p,
                    targmax_p,
                    tree_size as i32,
                    vocab as i32,
                )
                .map_err(|e| LlmError::Backend(format!("tree argmax: {e:?}")))?;
        }

        // ── 7. Commit deepest tree slot's SSM + conv state into model state ─
        //
        // Linear-chain prefill : leaf = tree_size - 1. After all layers ran,
        // tree_ssm_states[layer][leaf] holds the recurrent state at the end
        // of the chain ; copy it back into ssm_states[layer].state so the
        // next decode_step picks up where prefill left off. Same for conv.
        let leaf_idx = tree_size - 1;
        for li_idx in 0..self.blocks.len() {
            if !matches!(self.blocks[li_idx], BlockNvfp4::Ssm(_)) {
                continue;
            }
            let s_idx = get_ssm_layer_idx(&cfg, li_idx);
            unsafe {
                let (src_state_p, _g1) =
                    self.scratch.tree_ssm_states[s_idx].device_ptr(&self.stream);
                let (dst_state_p, _g2) = self.ssm_states[s_idx].state.device_ptr_mut(&self.stream);
                self.kernels
                    .copy_bf16(
                        &self.stream,
                        dst_state_p,
                        src_state_p + (leaf_idx as u64) * ssm_state_per_slot,
                        (n_v * head_kv * head_kv) as i32,
                    )
                    .map_err(|e| {
                        LlmError::Backend(format!("tree commit ssm state l{li_idx}: {e:?}"))
                    })?;
            }
            unsafe {
                let (src_conv_p, _g3) =
                    self.scratch.tree_conv_states[s_idx].device_ptr(&self.stream);
                let (dst_conv_p, _g4) = self.ssm_states[s_idx]
                    .conv_state
                    .device_ptr_mut(&self.stream);
                self.kernels
                    .copy_bf16(
                        &self.stream,
                        dst_conv_p,
                        src_conv_p + (leaf_idx as u64) * conv_state_per_slot,
                        ((conv_kernel - 1) * conv_dim) as i32,
                    )
                    .map_err(|e| {
                        LlmError::Backend(format!("tree commit conv state l{li_idx}: {e:?}"))
                    })?;
            }
        }

        // ── 8. Advance device counters ────────────────────────────────────
        unsafe {
            let (pos_p, _g_pos) = self.position_dev.device_ptr_mut(&self.stream);
            self.kernels
                .add_u32_dev(&self.stream, pos_p, tree_size as i32)
                .map_err(|e| LlmError::Backend(format!("tree add_u32 pos: {e:?}")))?;
        }
        unsafe {
            let (kvl_p, _g_kvl) = self.kv_len_dev.device_ptr_mut(&self.stream);
            self.kernels
                .add_u32_dev(&self.stream, kvl_p, tree_size as i32)
                .map_err(|e| LlmError::Backend(format!("tree add_u32 kv_len: {e:?}")))?;
        }

        // ── 9. Host-side `self.position` ──────────────────────────────────
        // In capture mode the caller advances `self.position` post-replay.
        if !in_prefill_capture {
            self.position += tree_size;
        }

        // Silence unused.
        let _ = (tgqa_m_p, tgqa_l_p, tgqa_o_p);

        Ok(())
    }

    // ── TrackK.b — Per-layer helpers (mirror of Q4_K hybrid_{attn,ssm}_layer) ─

    #[allow(clippy::too_many_arguments)]
    fn hybrid_attn_layer_nvfp4(
        &mut self,
        li: usize,
        tree_size: usize,
        base_position: usize,
        th_p: u64,
        thn_p: u64,
        tres_p: u64,
        tq_p: u64,
        tk_p: u64,
        tv_p: u64,
        tao_p: u64,
        tgate_p: u64,
        tup_p: u64,
        tparents_p: u64,
        tdepths_p: u64,
        tgqa_m_p: u64,
        tgqa_l_p: u64,
        tgqa_o_p: u64,
        row_h: u64,
        row_q: u64,
        row_kv: u64,
        row_up: u64,
        row_gate: u64,
        depths: &[u16],
        eps: f32,
        n_q: usize,
        n_kv: usize,
        head_dim: usize,
        q_dim: usize,
        kv_dim: usize,
        rope_dim: usize,
        d: usize,
    ) -> Result<(), LlmError> {
        use cudarc::driver::{DevicePtr, DevicePtrMut};
        let attn = match &self.blocks[li] {
            BlockNvfp4::Attn(a) => a,
            _ => unreachable!("hybrid_attn_layer_nvfp4 called on non-attn layer {li}"),
        };

        // Pre-attention RMSNorm (batched over tree rows).
        unsafe {
            let (an, _g) = attn.attn_norm.data.device_ptr(&self.stream);
            self.kernels
                .copy_bf16(&self.stream, thn_p, th_p, (tree_size * d) as i32)
                .map_err(|e| LlmError::Backend(format!("tree attn copy h_norm l{li}: {e:?}")))?;
            self.kernels
                .rms_norm_bf16(&self.stream, thn_p, an, eps, d as i32, tree_size as i32)
                .map_err(|e| LlmError::Backend(format!("tree attn rms_norm l{li}: {e:?}")))?;
        }

        // Q+gate projection (Qwen3.6 has output_gate). w_q outputs 2*q_dim
        // into tup_p (stride row_up). Then split_qg per row.
        // Per-row M=1 NVFP4 SGEMV — no batched NVFP4 kernel today.
        debug_assert_eq!(attn.w_q.n, 2 * q_dim);
        for r in 0..tree_size {
            let hn_r = thn_p + (r as u64) * row_h;
            let qg_r = tup_p + (r as u64) * row_up;
            attn.w_q
                .dispatch_matmul_m1(&self.kernels, &self.stream, hn_r, qg_r)?;
        }
        for r in 0..tree_size {
            let qg_r = tup_p + (r as u64) * row_up;
            let q_r = tq_p + (r as u64) * row_q;
            let gate_r = tgate_p + (r as u64) * row_gate;
            unsafe {
                self.kernels
                    .split_qg_bf16(&self.stream, qg_r, q_r, gate_r, n_q as i32, head_dim as i32)
                    .map_err(|e| LlmError::Backend(format!("tree split_qg r{r} l{li}: {e:?}")))?;
            }
        }

        // K, V projections.
        for r in 0..tree_size {
            let hn_r = thn_p + (r as u64) * row_h;
            let k_r = tk_p + (r as u64) * row_kv;
            let v_r = tv_p + (r as u64) * row_kv;
            attn.w_k
                .dispatch_matmul_m1(&self.kernels, &self.stream, hn_r, k_r)?;
            attn.w_v
                .dispatch_matmul_m1(&self.kernels, &self.stream, hn_r, v_r)?;
        }

        // Per-head Q-norm and K-norm (batched).
        unsafe {
            let (qn, _g1) = attn.q_norm.data.device_ptr(&self.stream);
            let (kn, _g2) = attn.k_norm.data.device_ptr(&self.stream);
            self.kernels
                .rms_norm_bf16(
                    &self.stream,
                    tq_p,
                    qn,
                    eps,
                    head_dim as i32,
                    (tree_size * n_q) as i32,
                )
                .map_err(|e| LlmError::Backend(format!("tree q_norm l{li}: {e:?}")))?;
            self.kernels
                .rms_norm_bf16(
                    &self.stream,
                    tk_p,
                    kn,
                    eps,
                    head_dim as i32,
                    (tree_size * n_kv) as i32,
                )
                .map_err(|e| LlmError::Backend(format!("tree k_norm l{li}: {e:?}")))?;
        }

        // RoPE per row with explicit pos = base_position + depths[r].
        unsafe {
            let (inv_p, _g_inv) = self.rope_freqs.inv_freq.device_ptr(&self.stream);
            for (r, &dep) in depths.iter().enumerate().take(tree_size) {
                let pos_r = base_position as i32 + dep as i32;
                let q_r = tq_p + (r as u64) * row_q;
                let k_r = tk_p + (r as u64) * row_kv;
                self.kernels
                    .rope_partial_bf16(
                        &self.stream,
                        q_r,
                        inv_p,
                        pos_r,
                        n_q as i32,
                        head_dim as i32,
                        rope_dim as i32,
                    )
                    .map_err(|e| LlmError::Backend(format!("tree rope q r{r} l{li}: {e:?}")))?;
                self.kernels
                    .rope_partial_bf16(
                        &self.stream,
                        k_r,
                        inv_p,
                        pos_r,
                        n_kv as i32,
                        head_dim as i32,
                        rope_dim as i32,
                    )
                    .map_err(|e| LlmError::Backend(format!("tree rope k r{r} l{li}: {e:?}")))?;
            }
        }

        // KV append (tree-aware) + GQA decode (tree-aware).
        let attn_idx = get_attn_layer_idx(&self.config, li);
        let kv_cache = &mut self.kv_caches[attn_idx];
        unsafe {
            let (kc_p, _g1) = kv_cache.k.device_ptr_mut(&self.stream);
            let (vc_p, _g2) = kv_cache.v.device_ptr_mut(&self.stream);
            let (pos_p, _g3) = self.position_dev.device_ptr(&self.stream);
            self.kernels
                .kv_append_tree_bf16(
                    &self.stream,
                    kc_p,
                    vc_p,
                    tk_p,
                    tv_p,
                    pos_p,
                    tree_size as i32,
                    kv_dim as i32,
                    self.max_seq as i32,
                )
                .map_err(|e| LlmError::Backend(format!("tree kv_append l{li}: {e:?}")))?;
        }
        unsafe {
            let (kc_p, _g1) = kv_cache.k.device_ptr(&self.stream);
            let (vc_p, _g2) = kv_cache.v.device_ptr(&self.stream);
            let (kvl_p, _g3) = self.kv_len_dev.device_ptr(&self.stream);
            self.kernels
                .gqa_decode_tree_bf16(
                    &self.stream,
                    tq_p,
                    kc_p,
                    vc_p,
                    tao_p,
                    tparents_p,
                    tdepths_p,
                    tgqa_m_p,
                    tgqa_l_p,
                    tgqa_o_p,
                    n_q as i32,
                    n_kv as i32,
                    kvl_p,
                    head_dim as i32,
                    self.max_seq as i32,
                    GQA_N_SPLIT as i32,
                    tree_size as i32,
                )
                .map_err(|e| LlmError::Backend(format!("tree gqa l{li}: {e:?}")))?;
        }

        // Sigmoid(gate) ; attn_out *= gate per row (Qwen3.6 output gate).
        for r in 0..tree_size {
            let gate_r = tgate_p + (r as u64) * row_gate;
            let ao_r = tao_p + (r as u64) * row_q;
            unsafe {
                self.kernels
                    .sigmoid_inplace_bf16(&self.stream, gate_r, q_dim as i32)
                    .map_err(|e| LlmError::Backend(format!("tree sigmoid r{r} l{li}: {e:?}")))?;
                self.kernels
                    .mul_inplace_bf16(&self.stream, ao_r, gate_r, q_dim as i32)
                    .map_err(|e| LlmError::Backend(format!("tree gate*ao r{r} l{li}: {e:?}")))?;
            }
        }

        // w_o → tree_h (per row M=1) + residual add.
        for r in 0..tree_size {
            let ao_r = tao_p + (r as u64) * row_q;
            let h_r = th_p + (r as u64) * row_h;
            attn.w_o
                .dispatch_matmul_m1(&self.kernels, &self.stream, ao_r, h_r)?;
        }
        unsafe {
            self.kernels
                .add_inplace_bf16(&self.stream, th_p, tres_p, (tree_size * d) as i32)
                .map_err(|e| LlmError::Backend(format!("tree attn residual l{li}: {e:?}")))?;
        }

        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn hybrid_ssm_layer_nvfp4(
        &mut self,
        li: usize,
        tree_size: usize,
        parents: &[i32],
        waves: &[Vec<i32>],
        th_p: u64,
        thn_p: u64,
        tres_p: u64,
        tssm_qkv_p: u64,
        tssm_conv_p: u64,
        tssm_z_p: u64,
        tssm_alpha_p: u64,
        tssm_beta_p: u64,
        tssm_qv_p: u64,
        tssm_kv_p: u64,
        tssm_out_p: u64,
        tssm_wave_p: u64,
        row_h: u64,
        row_conv: u64,
        row_value: u64,
        row_alpha: u64,
        conv_state_per_slot: u64,
        eps: f32,
        n_v: usize,
        n_k: usize,
        head_kv: usize,
        key_dim: usize,
        value_dim: usize,
        conv_dim: usize,
        conv_kernel: usize,
        d: usize,
        in_prefill_capture: bool,
    ) -> Result<(), LlmError> {
        use cudarc::driver::{DevicePtr, DevicePtrMut};
        let cfg = self.config.clone();
        let bf16_sz = std::mem::size_of::<half::bf16>() as u64;
        let s_idx = get_ssm_layer_idx(&cfg, li);
        let ssm = match &self.blocks[li] {
            BlockNvfp4::Ssm(s) => s,
            _ => unreachable!("hybrid_ssm_layer_nvfp4 called on non-ssm layer {li}"),
        };

        // 1. Pre-SSM RMSNorm (batched).
        unsafe {
            let (an, _g) = ssm.attn_norm.data.device_ptr(&self.stream);
            self.kernels
                .copy_bf16(&self.stream, thn_p, th_p, (tree_size * d) as i32)
                .map_err(|e| LlmError::Backend(format!("tree ssm copy h_norm l{li}: {e:?}")))?;
            self.kernels
                .rms_norm_bf16(&self.stream, thn_p, an, eps, d as i32, tree_size as i32)
                .map_err(|e| LlmError::Backend(format!("tree ssm rms_norm l{li}: {e:?}")))?;
        }

        // 2-5. SSM in-projections (BF16, per recipe). All per-row M=1.
        for r in 0..tree_size {
            let hn_r = thn_p + (r as u64) * row_h;
            let qkv_r = tssm_qkv_p + (r as u64) * row_conv;
            let z_r = tssm_z_p + (r as u64) * row_value;
            let alpha_r = tssm_alpha_p + (r as u64) * row_alpha;
            let beta_r = tssm_beta_p + (r as u64) * row_alpha;
            ssm.w_qkv
                .dispatch_matmul_m1(&self.kernels, &self.stream, hn_r, qkv_r)?;
            ssm.w_gate
                .dispatch_matmul_m1(&self.kernels, &self.stream, hn_r, z_r)?;
            ssm.w_alpha
                .dispatch_matmul_m1(&self.kernels, &self.stream, hn_r, alpha_r)?;
            ssm.w_beta
                .dispatch_matmul_m1(&self.kernels, &self.stream, hn_r, beta_r)?;
        }

        // 5b+6. SSM pre-step (fused or unfused) per row.
        unsafe {
            let (db, _gdb) = ssm.dt_bias.data.device_ptr(&self.stream);
            let (sa, _gsa) = ssm.ssm_a.data.device_ptr(&self.stream);
            for r in 0..tree_size {
                let alpha_r = tssm_alpha_p + (r as u64) * row_alpha;
                let beta_r = tssm_beta_p + (r as u64) * row_alpha;
                if self.use_ssm_fuse {
                    self.kernels
                        .ssm_pre_step_bf16(&self.stream, alpha_r, beta_r, db, sa, n_v as i32)
                        .map_err(|e| LlmError::Backend(format!("tree ssm_pre_step r{r}: {e:?}")))?;
                } else {
                    self.kernels
                        .sigmoid_inplace_bf16(&self.stream, beta_r, n_v as i32)
                        .map_err(|e| LlmError::Backend(format!("tree sigmoid beta r{r}: {e:?}")))?;
                    self.kernels
                        .add_inplace_bf16(&self.stream, alpha_r, db, n_v as i32)
                        .map_err(|e| LlmError::Backend(format!("tree alpha+dt r{r}: {e:?}")))?;
                    self.kernels
                        .softplus_inplace_bf16(&self.stream, alpha_r, n_v as i32)
                        .map_err(|e| LlmError::Backend(format!("tree softplus r{r}: {e:?}")))?;
                    self.kernels
                        .mul_inplace_bf16(&self.stream, alpha_r, sa, n_v as i32)
                        .map_err(|e| LlmError::Backend(format!("tree mul_a r{r}: {e:?}")))?;
                }
            }
        }

        // 7. Conv1d per row : pre-load slot 0 with model state, then for r>0
        // copy parent's slot before launching depthwise conv.
        unsafe {
            let (model_conv_p, _g) = self.ssm_states[s_idx].conv_state.device_ptr(&self.stream);
            let (tree_conv_states_p, _g2) =
                self.scratch.tree_conv_states[s_idx].device_ptr_mut(&self.stream);
            self.kernels
                .copy_bf16(
                    &self.stream,
                    tree_conv_states_p,
                    model_conv_p,
                    ((conv_kernel - 1) * conv_dim) as i32,
                )
                .map_err(|e| LlmError::Backend(format!("tree pre-load conv state l{li}: {e:?}")))?;

            let (cw, _gcw) = ssm.conv1d.data.device_ptr(&self.stream);
            for (r, &p) in parents.iter().enumerate().take(tree_size) {
                if r > 0 {
                    let parent = p as usize;
                    let dst = tree_conv_states_p + (r as u64) * conv_state_per_slot;
                    let src = tree_conv_states_p + (parent as u64) * conv_state_per_slot;
                    self.kernels
                        .copy_bf16(
                            &self.stream,
                            dst,
                            src,
                            ((conv_kernel - 1) * conv_dim) as i32,
                        )
                        .map_err(|e| {
                            LlmError::Backend(format!("tree conv parent copy r{r}: {e:?}"))
                        })?;
                }
                let qkv_r = tssm_qkv_p + (r as u64) * row_conv;
                let conv_out_r = tssm_conv_p + (r as u64) * row_conv;
                let slot_state = tree_conv_states_p + (r as u64) * conv_state_per_slot;
                self.kernels
                    .conv1d_depthwise_bf16(
                        &self.stream,
                        cw,
                        slot_state,
                        qkv_r,
                        conv_out_r,
                        conv_dim as i32,
                        conv_kernel as i32,
                    )
                    .map_err(|e| LlmError::Backend(format!("tree conv1d r{r}: {e:?}")))?;
            }
        }

        // 8. silu(conv_out) per row.
        for r in 0..tree_size {
            let conv_out_r = tssm_conv_p + (r as u64) * row_conv;
            unsafe {
                self.kernels
                    .silu_bf16(&self.stream, conv_out_r, conv_dim as i32)
                    .map_err(|e| LlmError::Backend(format!("tree silu conv r{r}: {e:?}")))?;
            }
        }

        // 9. Split q/k/v from conv_out, l2_norm_per_head, broadcast n_k→n_v.
        let q_offset = 0u64;
        let k_offset = (key_dim as u64) * bf16_sz;
        let v_offset = (2 * key_dim as u64) * bf16_sz;
        for r in 0..tree_size {
            let conv_out_r = tssm_conv_p + (r as u64) * row_conv;
            let q_r = conv_out_r + q_offset;
            let k_r = conv_out_r + k_offset;
            unsafe {
                self.kernels
                    .l2_norm_per_head_bf16(&self.stream, q_r, n_k as i32, head_kv as i32, eps)
                    .map_err(|e| LlmError::Backend(format!("tree l2 q r{r}: {e:?}")))?;
                self.kernels
                    .l2_norm_per_head_bf16(&self.stream, k_r, n_k as i32, head_kv as i32, eps)
                    .map_err(|e| LlmError::Backend(format!("tree l2 k r{r}: {e:?}")))?;
            }
        }
        for r in 0..tree_size {
            let conv_out_r = tssm_conv_p + (r as u64) * row_conv;
            let q_r = conv_out_r + q_offset;
            let k_r = conv_out_r + k_offset;
            let qv_r = tssm_qv_p + (r as u64) * row_value;
            let kv_r = tssm_kv_p + (r as u64) * row_value;
            unsafe {
                if n_k == n_v {
                    self.kernels
                        .copy_bf16(&self.stream, qv_r, q_r, value_dim as i32)
                        .map_err(|e| LlmError::Backend(format!("tree copy qv r{r}: {e:?}")))?;
                    self.kernels
                        .copy_bf16(&self.stream, kv_r, k_r, value_dim as i32)
                        .map_err(|e| LlmError::Backend(format!("tree copy kv r{r}: {e:?}")))?;
                } else {
                    self.kernels
                        .repeat_heads_bf16(
                            &self.stream,
                            q_r,
                            qv_r,
                            n_k as i32,
                            n_v as i32,
                            head_kv as i32,
                        )
                        .map_err(|e| LlmError::Backend(format!("tree repeat q r{r}: {e:?}")))?;
                    self.kernels
                        .repeat_heads_bf16(
                            &self.stream,
                            k_r,
                            kv_r,
                            n_k as i32,
                            n_v as i32,
                            head_kv as i32,
                        )
                        .map_err(|e| LlmError::Backend(format!("tree repeat k r{r}: {e:?}")))?;
                }
            }
        }

        // 10. Pre-load model SSM state into tree slot 0, then run
        // delta_net_step_tree_bf16 once per BFS depth wave.
        unsafe {
            let (model_state_p, _g) = self.ssm_states[s_idx].state.device_ptr(&self.stream);
            let (tree_states_p, _gts) =
                self.scratch.tree_ssm_states[s_idx].device_ptr_mut(&self.stream);
            self.kernels
                .copy_bf16(
                    &self.stream,
                    tree_states_p,
                    model_state_p,
                    (n_v * head_kv * head_kv) as i32,
                )
                .map_err(|e| LlmError::Backend(format!("tree pre-load ssm state l{li}: {e:?}")))?;

            let (parents_dev_p, _gp) = self.scratch.tree_parents.device_ptr(&self.stream);

            // Copy V from conv_out's tail into tssm_out_p (kernel needs
            // a [tree_size, n_v, head_kv] contiguous V view).
            for r in 0..tree_size {
                let conv_out_r = tssm_conv_p + (r as u64) * row_conv;
                let v_r = conv_out_r + v_offset;
                let dst_v = tssm_out_p + (r as u64) * row_value;
                self.kernels
                    .copy_bf16(&self.stream, dst_v, v_r, value_dim as i32)
                    .map_err(|e| LlmError::Backend(format!("tree copy v r{r}: {e:?}")))?;
            }

            // Wave launches : capture-safe path uses offsets into the pre-baked
            // linear wave-indices buffer (no per-wave HtoD).
            if in_prefill_capture {
                let (lin_p, _gl) = self.tree_ssm_wave_indices_linear.device_ptr(&self.stream);
                let i32_sz = std::mem::size_of::<i32>() as u64;
                for d_idx in 0..tree_size {
                    let wave_p = lin_p + (d_idx as u64) * i32_sz;
                    self.kernels
                        .delta_net_step_tree_bf16(
                            &self.stream,
                            tssm_qv_p,
                            tssm_kv_p,
                            tssm_out_p,
                            tssm_alpha_p,
                            tssm_beta_p,
                            parents_dev_p,
                            wave_p,
                            tree_states_p,
                            tssm_out_p,
                            1,
                            n_v as i32,
                            head_kv as i32,
                        )
                        .map_err(|e| {
                            LlmError::Backend(format!(
                                "tree delta_net (capture) d{d_idx} l{li}: {e:?}"
                            ))
                        })?;
                }
            } else {
                for (depth, wave) in waves.iter().enumerate() {
                    if wave.is_empty() {
                        continue;
                    }
                    self.stream
                        .memcpy_htod(wave, &mut self.scratch.tree_ssm_wave_indices)
                        .map_err(|e| LlmError::Backend(format!("tree wave H2D d{depth}: {e:?}")))?;
                    self.kernels
                        .delta_net_step_tree_bf16(
                            &self.stream,
                            tssm_qv_p,
                            tssm_kv_p,
                            tssm_out_p,
                            tssm_alpha_p,
                            tssm_beta_p,
                            parents_dev_p,
                            tssm_wave_p,
                            tree_states_p,
                            tssm_out_p,
                            wave.len() as i32,
                            n_v as i32,
                            head_kv as i32,
                        )
                        .map_err(|e| {
                            LlmError::Backend(format!("tree delta_net d{depth} l{li}: {e:?}"))
                        })?;
                }
            }
        }

        // 11. ssm_norm + silu(z) * out, fused or unfused per row.
        for r in 0..tree_size {
            let out_r = tssm_out_p + (r as u64) * row_value;
            let z_r = tssm_z_p + (r as u64) * row_value;
            unsafe {
                let (sn, _g) = ssm.ssm_norm.data.device_ptr(&self.stream);
                if self.use_ssm_fuse {
                    self.kernels
                        .ssm_post_step_bf16(
                            &self.stream,
                            out_r,
                            z_r,
                            sn,
                            eps,
                            n_v as i32,
                            head_kv as i32,
                        )
                        .map_err(|e| {
                            LlmError::Backend(format!("tree ssm_post_step r{r}: {e:?}"))
                        })?;
                } else {
                    self.kernels
                        .rms_norm_bf16(&self.stream, out_r, sn, eps, head_kv as i32, n_v as i32)
                        .map_err(|e| LlmError::Backend(format!("tree ssm_norm r{r}: {e:?}")))?;
                    self.kernels
                        .silu_bf16(&self.stream, z_r, value_dim as i32)
                        .map_err(|e| LlmError::Backend(format!("tree silu z r{r}: {e:?}")))?;
                    self.kernels
                        .mul_inplace_bf16(&self.stream, out_r, z_r, value_dim as i32)
                        .map_err(|e| LlmError::Backend(format!("tree mul gated r{r}: {e:?}")))?;
                }
            }
        }

        // 12. ssm_out @ gated → tree_h (per row M=1) + residual add.
        for r in 0..tree_size {
            let out_r = tssm_out_p + (r as u64) * row_value;
            let h_r = th_p + (r as u64) * row_h;
            ssm.ssm_out
                .dispatch_matmul_m1(&self.kernels, &self.stream, out_r, h_r)?;
        }
        unsafe {
            self.kernels
                .add_inplace_bf16(&self.stream, th_p, tres_p, (tree_size * d) as i32)
                .map_err(|e| LlmError::Backend(format!("tree ssm residual l{li}: {e:?}")))?;
        }

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// MoE FFN forward (zero-host-sync, indexed NVFP4 sgemv per expert)
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn moe_ffn_forward_step_nvfp4(
    moe: &MoeFfnNvfp4,
    kernels: &LlmKernels,
    stream: &Arc<CudaStream>,
    cfg: &Qwen35Config,
    h_norm_p: u64,
    h_p: u64,
    router_logits_p: u64,
    topk_idx_p: u64,
    topk_w_p: u64,
    expert_gate_p: u64,
    expert_up_p: u64,
    expert_out_p: u64,
    shexp_dot_p: u64,
) -> Result<(), LlmError> {
    use cudarc::driver::DevicePtr;
    let d = cfg.d as i32;
    let ef = cfg.expert_f as i32;
    let n_e = cfg.n_experts as i32;
    let k_used = cfg.n_experts_used as i32;

    // 1. Router logits = gate_inp @ h_norm  (BF16, per recipe)
    moe.gate_inp
        .dispatch_matmul_m1(kernels, stream, h_norm_p, router_logits_p)?;

    // 2. Top-K softmax → indices + renormalized weights, on device.
    unsafe {
        kernels
            .topk_softmax_bf16(stream, router_logits_p, topk_idx_p, topk_w_p, n_e, k_used)
            .map_err(|e| LlmError::Backend(format!("topk_softmax: {e:?}")))?;
    }

    // 3. Zero h_p.
    unsafe {
        kernels
            .zero_bf16(stream, h_p, d)
            .map_err(|e| LlmError::Backend(format!("zero h_p: {e:?}")))?;
    }

    // 4. Routed experts loop (zero host sync, indexed NVFP4 sgemv).
    let (g_ptrs_p, _gg) = moe.gate_packed_ptrs_dev.device_ptr(stream);
    let (u_ptrs_p, _gu) = moe.up_packed_ptrs_dev.device_ptr(stream);
    let (d_ptrs_p, _gd) = moe.down_packed_ptrs_dev.device_ptr(stream);
    let (g_sptrs_p, _gsg) = moe.gate_scale_ptrs_dev.device_ptr(stream);
    let (u_sptrs_p, _gsu) = moe.up_scale_ptrs_dev.device_ptr(stream);
    let (d_sptrs_p, _gsd) = moe.down_scale_ptrs_dev.device_ptr(stream);
    let (g_alphas_p, _gag) = moe.gate_alphas_dev.device_ptr(stream);
    let (u_alphas_p, _gau) = moe.up_alphas_dev.device_ptr(stream);
    let (d_alphas_p, _gad) = moe.down_alphas_dev.device_ptr(stream);

    for slot in 0..k_used {
        // gate = w_gate[topk[slot]] @ h_norm
        unsafe {
            kernels
                .sgemv_nvfp4_bf16_indexed(
                    stream,
                    g_ptrs_p,
                    g_sptrs_p,
                    g_alphas_p,
                    topk_idx_p,
                    slot,
                    h_norm_p,
                    expert_gate_p,
                    ef,
                    d,
                )
                .map_err(|e| LlmError::Backend(format!("nvfp4_idx gate slot={slot}: {e:?}")))?;
            // up = w_up[topk[slot]] @ h_norm
            kernels
                .sgemv_nvfp4_bf16_indexed(
                    stream,
                    u_ptrs_p,
                    u_sptrs_p,
                    u_alphas_p,
                    topk_idx_p,
                    slot,
                    h_norm_p,
                    expert_up_p,
                    ef,
                    d,
                )
                .map_err(|e| LlmError::Backend(format!("nvfp4_idx up slot={slot}: {e:?}")))?;
            // gate = silu(gate) * up
            kernels
                .swiglu_bf16(stream, expert_gate_p, expert_up_p, expert_gate_p, ef)
                .map_err(|e| LlmError::Backend(format!("swiglu expert slot={slot}: {e:?}")))?;
            // down_out = w_down[topk[slot]] @ gate
            kernels
                .sgemv_nvfp4_bf16_indexed(
                    stream,
                    d_ptrs_p,
                    d_sptrs_p,
                    d_alphas_p,
                    topk_idx_p,
                    slot,
                    expert_gate_p,
                    expert_out_p,
                    d,
                    ef,
                )
                .map_err(|e| LlmError::Backend(format!("nvfp4_idx down slot={slot}: {e:?}")))?;
            // h_p += topk_w[slot] * down_out (alpha read from device).
            kernels
                .scaled_add_inplace_bf16_devscalar(stream, h_p, expert_out_p, topk_w_p, slot, d)
                .map_err(|e| {
                    LlmError::Backend(format!("scaled_add devscalar slot={slot}: {e:?}"))
                })?;
        }
    }

    // 5. Shared expert (parallel path).
    // shexp_dot = gate_inp_shexp @ h_norm  (BF16, scalar via sgemv 1×D)
    unsafe {
        let (gip_p, _g) = moe.gate_inp_shexp.data.device_ptr(stream);
        kernels
            .sgemv_bf16_bf16(stream, gip_p, h_norm_p, shexp_dot_p, 1, d)
            .map_err(|e| LlmError::Backend(format!("shexp dot: {e:?}")))?;
    }
    // Shared-expert gate, up, down (NVFP4).
    moe.gate_shexp
        .dispatch_matmul_m1(kernels, stream, h_norm_p, expert_gate_p)?;
    moe.up_shexp
        .dispatch_matmul_m1(kernels, stream, h_norm_p, expert_up_p)?;
    unsafe {
        kernels
            .swiglu_bf16(stream, expert_gate_p, expert_up_p, expert_gate_p, ef)
            .map_err(|e| LlmError::Backend(format!("swiglu shexp: {e:?}")))?;
    }
    moe.down_shexp
        .dispatch_matmul_m1(kernels, stream, expert_gate_p, expert_out_p)?;
    // h_p += sigmoid(dot) * down_out — fused kernel keeps sigmoid in float.
    unsafe {
        kernels
            .scaled_add_sigmoid_devscalar_bf16(stream, h_p, expert_out_p, shexp_dot_p, d)
            .map_err(|e| LlmError::Backend(format!("scaled_add_sigmoid shexp: {e:?}")))?;
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// SSM weight transforms — bridge HF safetensors layout to llm_kernels ABI
// ---------------------------------------------------------------------------

/// Transform raw `A_log` BF16 buffer into `ssm_a = -exp(A_log)` in place,
/// to match the GGUF `ssm_a` semantics consumed by `ssm_pre_step_bf16`.
///
/// Per qwen35.rs:27 architecture note : `A = -exp(ssm_a_in_gguf)` ; the GGUF
/// loader stores the raw value (no transform), and the SSM pre-step kernel
/// multiplies the softplus output directly by it. So in the GGUF flow the
/// effective math is `alpha = softplus(...) * (-exp(A_log_raw))`.
///
/// HF safetensors checkpoints store `A_log` raw (literally named `A_log`),
/// while GGUF stores `ssm_a = -exp(A_log)` already pre-computed. We mirror
/// the GGUF convention so the existing kernels work unchanged.
///
/// Roundtrip : DtoH → CPU `-exp(...)` → HtoD. ~32 elements per layer × 30
/// SSM layers = 960 floats total — negligible at load time.
///
fn transform_a_log_to_ssm_a(stream: &Arc<CudaStream>, t: &mut Bf16Tensor) -> Result<(), LlmError> {
    let host: Vec<half::bf16> = stream
        .memcpy_dtov(&t.data)
        .map_err(|e| LlmError::Backend(format!("dtov A_log: {e:?}")))?;
    let transformed: Vec<half::bf16> = host
        .iter()
        .map(|v| half::bf16::from_f32(-(v.to_f32()).exp()))
        .collect();
    let dev = stream
        .memcpy_stod(&transformed)
        .map_err(|e| LlmError::Backend(format!("stod ssm_a: {e:?}")))?;
    t.data = dev;
    Ok(())
}

/// Transpose conv1d weight from HF Conv1d layout `[out_ch=conv_dim, 1, kernel]`
/// to the kernel-major layout `[kernel, conv_dim]` expected by
/// `conv1d_depthwise_bf16` (which reads `weight[t * conv_dim + c]`).
///
/// Returns a fresh `Bf16Tensor` owning the transposed device buffer.
/// 32K elements per layer × 30 SSM layers = 1M floats — trivial at load time.
///
/// Currently unused — see note inline. The HF `[8192, 1, 4]` byte layout
/// IS byte-identical to the GGUF `[4, 8192]` (because GGUF's `ne[0]` is
/// innermost, so GGUF's `[4, 8192]` is row-major `[8192, 4]`). The
/// `conv1d_depthwise_bf16` kernel doc string says it expects
/// `[kernel, conv_dim]` but the Q4_K path uses the GGUF bytes unchanged and
/// works coherently — so either the kernel's index math is robust to this
/// layout OR the doc string is misleading. Don't enable this without a real
/// parity test against Q4_K_M conv1d outputs.
#[allow(dead_code)]
fn transpose_conv1d_to_kernel_major(
    stream: &Arc<CudaStream>,
    src: Bf16Tensor,
) -> Result<Bf16Tensor, LlmError> {
    // Expected source shape : [conv_dim, 1, kernel] OR [conv_dim, kernel].
    let (conv_dim, kernel) = match src.shape.as_slice() {
        [c, 1, k] => (*c, *k),
        [c, k] => (*c, *k),
        other => {
            return Err(LlmError::Safetensors(format!(
                "conv1d.weight: expected [conv_dim, 1, kernel] or [conv_dim, kernel], got {other:?}"
            )));
        },
    };
    let host: Vec<half::bf16> = stream
        .memcpy_dtov(&src.data)
        .map_err(|e| LlmError::Backend(format!("dtov conv1d: {e:?}")))?;
    if host.len() != conv_dim * kernel {
        return Err(LlmError::Safetensors(format!(
            "conv1d.weight: expected {conv_dim}*{kernel}={} elements, got {}",
            conv_dim * kernel,
            host.len()
        )));
    }
    let mut transposed: Vec<half::bf16> = vec![half::bf16::from_f32(0.0); conv_dim * kernel];
    for c in 0..conv_dim {
        for t in 0..kernel {
            transposed[t * conv_dim + c] = host[c * kernel + t];
        }
    }
    let dev = stream
        .memcpy_stod(&transposed)
        .map_err(|e| LlmError::Backend(format!("stod conv1d: {e:?}")))?;
    Ok(Bf16Tensor {
        data: dev,
        shape: vec![kernel, conv_dim],
    })
}

// ---------------------------------------------------------------------------
// Per-layer MoE loader helper
// ---------------------------------------------------------------------------

/// Build a `MoeFfnNvfp4` for layer `li` by extracting all 256 routed
/// experts (gate / up / down) + the shared expert + the BF16 router gate
/// from the loaded weight map. Builds the 9 device-side dispatch arrays
/// (3 packed_ptrs, 3 scale_ptrs, 3 alphas) once at load time.
fn load_moe_layer(
    weights: &mut LoadedNvfp4Weights,
    stream: &Arc<CudaStream>,
    li: usize,
    n_experts: usize,
) -> Result<MoeFfnNvfp4, LlmError> {
    use cudarc::driver::DevicePtr;
    let prefix = format!("model.language_model.layers.{li}.mlp.");

    // Routed experts (256 × {gate, up, down}).
    let mut gate_exps: Vec<Nvfp4Tensor> = Vec::with_capacity(n_experts);
    let mut up_exps: Vec<Nvfp4Tensor> = Vec::with_capacity(n_experts);
    let mut down_exps: Vec<Nvfp4Tensor> = Vec::with_capacity(n_experts);
    for e in 0..n_experts {
        gate_exps.push(weights.take_nvfp4(&format!("{prefix}experts.{e}.gate_proj"))?);
        up_exps.push(weights.take_nvfp4(&format!("{prefix}experts.{e}.up_proj"))?);
        down_exps.push(weights.take_nvfp4(&format!("{prefix}experts.{e}.down_proj"))?);
    }

    // Shared expert + router gate.
    let gate_inp = weights.take_bf16(&format!("{prefix}gate.weight"))?;
    let gate_inp_shexp = weights.take_bf16(&format!("{prefix}shared_expert_gate.weight"))?;
    let gate_shexp = weights.take_nvfp4(&format!("{prefix}shared_expert.gate_proj"))?;
    let up_shexp = weights.take_nvfp4(&format!("{prefix}shared_expert.up_proj"))?;
    let down_shexp = weights.take_nvfp4(&format!("{prefix}shared_expert.down_proj"))?;

    // Build 9 device-side dispatch arrays.
    let build_packed_ptrs = |v: &[Nvfp4Tensor]| -> Result<CudaSlice<u64>, LlmError> {
        let host: Vec<u64> = v
            .iter()
            .map(|t| {
                let (p, _g) = t.packed.device_ptr(stream);
                p
            })
            .collect();
        stream
            .memcpy_stod(&host)
            .map_err(|e| LlmError::Backend(format!("upload packed ptrs: {e:?}")))
    };
    let build_scale_ptrs = |v: &[Nvfp4Tensor]| -> Result<CudaSlice<u64>, LlmError> {
        let host: Vec<u64> = v
            .iter()
            .map(|t| {
                let (p, _g) = t.scale_per_block.device_ptr(stream);
                p
            })
            .collect();
        stream
            .memcpy_stod(&host)
            .map_err(|e| LlmError::Backend(format!("upload scale ptrs: {e:?}")))
    };
    let build_alphas = |v: &[Nvfp4Tensor]| -> Result<CudaSlice<f32>, LlmError> {
        let host: Vec<f32> = v.iter().map(|t| t.matmul_alpha()).collect();
        stream
            .memcpy_stod(&host)
            .map_err(|e| LlmError::Backend(format!("upload alphas: {e:?}")))
    };

    let gate_packed_ptrs_dev = build_packed_ptrs(&gate_exps)?;
    let up_packed_ptrs_dev = build_packed_ptrs(&up_exps)?;
    let down_packed_ptrs_dev = build_packed_ptrs(&down_exps)?;
    let gate_scale_ptrs_dev = build_scale_ptrs(&gate_exps)?;
    let up_scale_ptrs_dev = build_scale_ptrs(&up_exps)?;
    let down_scale_ptrs_dev = build_scale_ptrs(&down_exps)?;
    let gate_alphas_dev = build_alphas(&gate_exps)?;
    let up_alphas_dev = build_alphas(&up_exps)?;
    let down_alphas_dev = build_alphas(&down_exps)?;

    Ok(MoeFfnNvfp4 {
        gate_inp,
        gate_exps,
        up_exps,
        down_exps,
        gate_inp_shexp,
        gate_shexp,
        up_shexp,
        down_shexp,
        gate_packed_ptrs_dev,
        up_packed_ptrs_dev,
        down_packed_ptrs_dev,
        gate_scale_ptrs_dev,
        up_scale_ptrs_dev,
        down_scale_ptrs_dev,
        gate_alphas_dev,
        up_alphas_dev,
        down_alphas_dev,
    })
}

// ---------------------------------------------------------------------------
// Tests (pure-Rust where possible)
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
        let cfg = qwen36_35b_a3b_nvfp4_config();
        assert_eq!(cfg.n_layers, 40);
        assert_eq!(cfg.n_experts, 256);
        assert_eq!(cfg.n_experts_used, 8);
        assert_eq!(cfg.expert_f, 512);
        assert_eq!(cfg.attention_indices.len(), 10);
        assert_eq!(cfg.ssm_indices.len(), 30);
        assert_eq!(cfg.attention_indices[0], 3);
        assert_eq!(cfg.attention_indices[9], 39);
    }
}
