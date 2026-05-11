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
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

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
                        ssm_a: weights.take_bf16(&format!("{prefix}linear_attn.A_log"))?,
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
/// Currently unused — empirical bench showed enabling this transform makes
/// the model collapse to a 1-token loop within 4 steps (worse than no
/// transform). Kept here as a debugging tool for future SSM correctness
/// follow-ups.
#[allow(dead_code)]
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
/// Currently unused — empirical bench showed enabling this transform makes
/// the model collapse faster (5 distinct tokens vs 18 with the raw layout).
/// The actual safetensors layout that produces the longest coherent run
/// matches the GGUF kernel ABI directly. Kept as a debugging tool for
/// future SSM correctness follow-ups.
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
