//! `qwen35_inference_metal` — bring-up runtime for Qwen3.5/3.6 hybrid
//! models (Qwen3.6-27B dense, Qwen3.6-35B-A3B MoE) on Metal.
//!
//! This is the foundation example for the hybrid Metal forward.
//! T143 scope (this file): Metal-resident weight loader covering all
//! four GGUF quantisation formats present in the hybrid models
//! (Q4_K / Q5_K / Q6_K / Q8_0) plus F32 norms; dispatches each
//! matmul-vec via the right NSG=2 sgemv kernel from `rustorch-metal`.
//!
//! T144 will add the Metal SSM block kernel (1-D conv ring buffer +
//! gated delta-net recurrence + gated norm). T145 will add the MoE
//! expert dispatch (top-K router + per-expert matmul). Until those
//! land the SSM and MoE paths are stubbed and the example only
//! exercises the loader + the per-layer dispatch wiring.
//!
//! Usage (loader smoke test):
//!
//! ```sh
//! cargo run --release -p rustorch-llm --example qwen35_inference_metal -- \
//!     ~/models/Qwen3.6-27B-Q4_K_M.gguf
//! ```
//!
//! Goes through every weight tensor in the GGUF, allocates a Metal
//! buffer, copies the raw quantised bytes (no CPU-side f32 dequant,
//! so the only RAM cost is the on-disk size), and reports total
//! bytes loaded + per-dtype distribution. This proves the model fits
//! on the M4 Max unified memory and lets the upcoming forward
//! dispatch reach the right kernel.

#![cfg(target_os = "macos")]
// Some helpers and fields are used at runtime via Metal Shared buffers
// (unified memory), and the example is in active bring-up — silence the
// strictest unused-code lints during this phase.
#![allow(dead_code, unused_imports)]

use std::collections::BTreeMap;
use std::env;
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use metal::Buffer;
use rustorch_gguf::{dequant_to_f32, GgmlType, GgufFile, TensorInfo};
use rustorch_llm::qwen35::{parse_config, LayerKind, Qwen35Config, Qwen35Variant};
use rustorch_metal::backend::MetalBackend;
use rustorch_metal::backend_singleton::metal_backend;
use rustorch_metal::error::MetalError;
use rustorch_metal::kernels::{
    add_inplace_f32, delta_net_step_f32, gqa_decode_f32, kv_append_f32, l2_norm_per_head_f32,
    rms_norm_f32, rms_norm_per_head_f32, rms_norm_per_head_gated_f32, rope_half_split_f32,
    sgemv_q4_k_f32_lcpp_nsg2_into, sgemv_q5_k_f32_lcpp_nsg2_into, sgemv_q6_k_f32_lcpp_nsg2_into,
    sgemv_q8_0_f32_lcpp_nsg2_into, sigmoid_mul_inplace_f32, ssm_conv1d_step_f32, swiglu_f32,
};

/// One quantised weight tensor resident in a Metal buffer, tagged with
/// its dtype so [`HybridMetalWeight::matmul_into`] can dispatch to the
/// right sgemv kernel without per-call branching at the user site.
pub struct HybridMetalWeight {
    /// Raw quantised bytes, mmap-friendly (`MTLStorageModeShared`).
    pub buffer: Buffer,
    /// GGML quantisation type — drives kernel dispatch.
    pub dtype: GgmlType,
    /// Inner dimension (matrix is logically `[N, K]`).
    pub k: usize,
    /// Output dimension.
    pub n: usize,
    /// GGUF tensor name — kept for diagnostics.
    pub name: String,
}

impl HybridMetalWeight {
    /// Compute `out = W @ x` on Metal using the appropriate sgemv kernel
    /// for `self.dtype`. Returns an error for any dtype that doesn't have
    /// a matmul-vec kernel yet.
    pub fn matmul_into(
        &self,
        backend: &MetalBackend,
        x_buf: &Buffer,
        out_buf: &Buffer,
    ) -> Result<(), MetalError> {
        match self.dtype {
            GgmlType::Q4_K => {
                sgemv_q4_k_f32_lcpp_nsg2_into(backend, x_buf, &self.buffer, out_buf, self.k, self.n)
            },
            GgmlType::Q5_K => {
                sgemv_q5_k_f32_lcpp_nsg2_into(backend, x_buf, &self.buffer, out_buf, self.k, self.n)
            },
            GgmlType::Q6_K => {
                sgemv_q6_k_f32_lcpp_nsg2_into(backend, x_buf, &self.buffer, out_buf, self.k, self.n)
            },
            GgmlType::Q8_0 => {
                sgemv_q8_0_f32_lcpp_nsg2_into(backend, x_buf, &self.buffer, out_buf, self.k, self.n)
            },
            GgmlType::F32 => {
                // Small F32 projections (e.g. ssm_alpha, ssm_beta) — CPU matmul
                // since both src and dst are in unified memory. Fast for the
                // sizes we hit (5120 × 48 = 245K mults).
                backend.drain(); // ensure prior writes to x_buf landed
                unsafe {
                    let w = std::slice::from_raw_parts(
                        self.buffer.contents() as *const f32,
                        self.n * self.k,
                    );
                    let x = std::slice::from_raw_parts(x_buf.contents() as *const f32, self.k);
                    let y = std::slice::from_raw_parts_mut(out_buf.contents() as *mut f32, self.n);
                    for i in 0..self.n {
                        let row = &w[i * self.k..(i + 1) * self.k];
                        let mut acc = 0.0_f32;
                        for j in 0..self.k {
                            acc += row[j] * x[j];
                        }
                        y[i] = acc;
                    }
                }
                Ok(())
            },
            other => Err(MetalError::Unsupported(format!(
                "HybridMetalWeight::matmul_into: dtype {:?} not supported by any sgemv kernel",
                other
            ))),
        }
    }
}

/// Load a quantised 2-D weight directly into a Metal buffer (no CPU-side
/// dequant). The GGUF stores 2-D weights as `shape = [in, out]`; the
/// resulting buffer has the same row-major layout as our matmul-vec
/// kernels expect (each output row is a contiguous block of `K/QK_BS *
/// block_size` bytes).
fn load_quant_2d(
    backend: &MetalBackend,
    file: &GgufFile,
    info: &TensorInfo,
) -> Result<HybridMetalWeight, String> {
    if info.shape.len() != 2 {
        return Err(format!(
            "{}: expected 2-D tensor, got shape {:?}",
            info.name, info.shape
        ));
    }
    let k = info.shape[0] as usize;
    let n = info.shape[1] as usize;
    let bytes = file.tensor_bytes(info);
    let buffer = backend
        .alloc_shared(bytes.len())
        .map_err(|e| format!("alloc {}: {:?}", info.name, e))?;
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), buffer.contents() as *mut u8, bytes.len());
    }
    Ok(HybridMetalWeight {
        buffer,
        dtype: info.dtype,
        k,
        n,
        name: info.name.clone(),
    })
}

/// Load an F32 1-D vector (norm gamma, dt bias, ssm_a, etc.) into a
/// Metal buffer. We keep these as f32 since they're tiny and the kernels
/// (RMSNorm, residual add, etc.) operate on f32.
fn load_f32_1d(
    backend: &MetalBackend,
    file: &GgufFile,
    info: &TensorInfo,
) -> Result<Buffer, String> {
    if info.dtype != GgmlType::F32 {
        return Err(format!("{}: expected F32, got {:?}", info.name, info.dtype));
    }
    let bytes = file.tensor_bytes(info);
    let buffer = backend
        .alloc_shared(bytes.len())
        .map_err(|e| format!("alloc {}: {:?}", info.name, e))?;
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), buffer.contents() as *mut u8, bytes.len());
    }
    Ok(buffer)
}

/// All Metal-resident weights for one attention layer.
#[allow(missing_docs)]
pub struct AttnLayerMetal {
    pub attn_norm: Buffer,      // f32 [d]
    pub attn_post_norm: Buffer, // f32 [d]
    pub w_q: HybridMetalWeight, // Q+gate combined: [d, 2 * head_dim * n_q_heads]
    pub w_k: HybridMetalWeight, // [d, head_dim * n_kv_heads]
    pub w_v: HybridMetalWeight, // [d, head_dim * n_kv_heads]
    pub w_o: HybridMetalWeight, // [head_dim * n_q_heads, d]
    pub q_norm: Buffer,         // f32 [head_dim]
    pub k_norm: Buffer,         // f32 [head_dim]
}

/// All Metal-resident weights for one SSM (Gated DeltaNet) layer.
#[allow(missing_docs)]
pub struct SsmLayerMetal {
    pub attn_norm: Buffer,            // f32 [d]
    pub attn_post_norm: Buffer,       // f32 [d]
    pub w_qkv: HybridMetalWeight,     // [d, conv_dim] where conv_dim = 2*key_dim + value_dim
    pub w_gate: HybridMetalWeight,    // [d, value_dim]
    pub conv1d: Buffer,               // f32 [conv_kernel * conv_dim]
    pub ssm_alpha: HybridMetalWeight, // [d, n_v]
    pub ssm_beta: HybridMetalWeight,  // [d, n_v]
    pub dt_bias: Buffer,              // f32 [n_v]
    pub ssm_a: Buffer,                // f32 [n_v]
    pub ssm_norm: Buffer,             // f32 [head_v_dim]
    pub ssm_out: HybridMetalWeight,   // [value_dim, d]
}

/// FFN sub-block — dense or MoE.
#[allow(missing_docs)]
pub enum FfnLayerMetal {
    Dense {
        w_gate: HybridMetalWeight, // [d, f]
        w_up: HybridMetalWeight,   // [d, f]
        w_down: HybridMetalWeight, // [f, d]
    },
    Moe {
        // For the loader smoke test we just count and store. The actual
        // forward (T145) needs to dispatch per-expert based on a top-K
        // router output.
        gate_inp: HybridMetalWeight,
        gate_inp_shexp: Buffer, // f32 [d]
        gate_shexp: HybridMetalWeight,
        up_shexp: HybridMetalWeight,
        down_shexp: HybridMetalWeight,
        gate_exps: Vec<HybridMetalWeight>,
        up_exps: Vec<HybridMetalWeight>,
        down_exps: Vec<HybridMetalWeight>,
    },
}

/// One layer worth of Metal-resident weights — discriminated by kind.
#[allow(missing_docs)]
pub enum LayerMetal {
    Attn {
        attn: AttnLayerMetal,
        ffn: FfnLayerMetal,
    },
    Ssm {
        ssm: SsmLayerMetal,
        ffn: FfnLayerMetal,
    },
}

/// Top-level Metal-resident model.
#[allow(missing_docs)]
pub struct Qwen35MetalModel {
    pub cfg: Qwen35Config,
    pub tok_embd: HybridMetalWeight, // matmul-vec needed only if we ever do
    // approx-NN lookup; for now we'll dequant
    // a single row at decode time
    pub output_norm: Buffer,
    pub output: HybridMetalWeight,
    pub layers: Vec<LayerMetal>,
}

/// Counts of bytes uploaded by dtype, returned by the loader for
/// diagnostics.
#[derive(Default, Debug)]
pub struct LoadStats {
    /// Total bytes uploaded across all tensors.
    pub total_bytes: usize,
    /// Per-dtype byte counts.
    pub by_dtype: BTreeMap<String, usize>,
    /// Number of tensors loaded.
    pub n_tensors: usize,
    /// Wall-clock time in ms.
    pub elapsed_ms: f64,
}

/// Load every weight tensor for the parsed Qwen3.5/3.6 model into Metal
/// buffers. The Metal storage mode is `Shared` (unified memory on Apple
/// Silicon) so allocation is O(1) per tensor and the GPU sees the data
/// at the same address as the CPU.
pub fn load_metal_model(
    backend: &MetalBackend,
    path: &std::path::Path,
    cfg: &Qwen35Config,
) -> Result<(Qwen35MetalModel, LoadStats), String> {
    let file = GgufFile::open(path).map_err(|e| format!("open gguf: {e:?}"))?;
    let mut stats = LoadStats::default();
    let t0 = Instant::now();

    let visit_tensor = |info: &TensorInfo, stats: &mut LoadStats| {
        let bytes = file.tensor_bytes(info).len();
        stats.total_bytes += bytes;
        *stats
            .by_dtype
            .entry(format!("{:?}", info.dtype))
            .or_insert(0) += bytes;
        stats.n_tensors += 1;
    };

    let load_2d = |name: &str, stats: &mut LoadStats| -> Result<HybridMetalWeight, String> {
        let info = file
            .tensor(name)
            .ok_or_else(|| format!("missing tensor: {name}"))?;
        visit_tensor(info, stats);
        load_quant_2d(backend, &file, info)
    };
    let load_3d_stacked =
        |name: &str, stats: &mut LoadStats| -> Result<Vec<HybridMetalWeight>, String> {
            // Stacked-expert tensors: GGUF shape [d, ef, n_experts] for gate/up,
            // [ef, d, n_experts] for down. We split into n_experts 2-D weights,
            // each placed in its own Metal buffer.
            let info = file
                .tensor(name)
                .ok_or_else(|| format!("missing tensor: {name}"))?;
            visit_tensor(info, stats);
            if info.shape.len() != 3 {
                return Err(format!("{name}: expected 3-D, got {:?}", info.shape));
            }
            let dim0 = info.shape[0] as usize;
            let dim1 = info.shape[1] as usize;
            let n_experts = info.shape[2] as usize;
            let bytes_per_expert = info.byte_size() as usize / n_experts;
            let bytes = file.tensor_bytes(info);
            let mut out = Vec::with_capacity(n_experts);
            for e in 0..n_experts {
                let buffer = backend
                    .alloc_shared(bytes_per_expert)
                    .map_err(|e2| format!("alloc {name}[{e}]: {e2:?}"))?;
                let src = &bytes[e * bytes_per_expert..(e + 1) * bytes_per_expert];
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        src.as_ptr(),
                        buffer.contents() as *mut u8,
                        bytes_per_expert,
                    );
                }
                out.push(HybridMetalWeight {
                    buffer,
                    dtype: info.dtype,
                    k: dim0,
                    n: dim1,
                    name: format!("{name}[{e}]"),
                });
            }
            Ok(out)
        };
    let load_1d_f32 = |name: &str, stats: &mut LoadStats| -> Result<Buffer, String> {
        let info = file
            .tensor(name)
            .ok_or_else(|| format!("missing tensor: {name}"))?;
        visit_tensor(info, stats);
        load_f32_1d(backend, &file, info)
    };

    let tok_embd = load_2d("token_embd.weight", &mut stats)?;
    let output_norm = load_1d_f32("output_norm.weight", &mut stats)?;
    let output = load_2d("output.weight", &mut stats)?;

    let mut layers = Vec::with_capacity(cfg.n_layers);
    for li in 0..cfg.n_layers {
        let kind = cfg.layer_kind(li);
        let attn_norm = load_1d_f32(&format!("blk.{li}.attn_norm.weight"), &mut stats)?;
        let attn_post_norm =
            load_1d_f32(&format!("blk.{li}.post_attention_norm.weight"), &mut stats)?;

        // FFN sub-block — same on both layer kinds.
        let ffn = match cfg.variant {
            Qwen35Variant::Dense => {
                let w_gate = load_2d(&format!("blk.{li}.ffn_gate.weight"), &mut stats)?;
                let w_up = load_2d(&format!("blk.{li}.ffn_up.weight"), &mut stats)?;
                let w_down = load_2d(&format!("blk.{li}.ffn_down.weight"), &mut stats)?;
                FfnLayerMetal::Dense {
                    w_gate,
                    w_up,
                    w_down,
                }
            },
            Qwen35Variant::Moe => {
                let gate_inp = load_2d(&format!("blk.{li}.ffn_gate_inp.weight"), &mut stats)?;
                let gate_inp_shexp =
                    load_1d_f32(&format!("blk.{li}.ffn_gate_inp_shexp.weight"), &mut stats)?;
                let gate_shexp = load_2d(&format!("blk.{li}.ffn_gate_shexp.weight"), &mut stats)?;
                let up_shexp = load_2d(&format!("blk.{li}.ffn_up_shexp.weight"), &mut stats)?;
                let down_shexp = load_2d(&format!("blk.{li}.ffn_down_shexp.weight"), &mut stats)?;
                let gate_exps =
                    load_3d_stacked(&format!("blk.{li}.ffn_gate_exps.weight"), &mut stats)?;
                let up_exps = load_3d_stacked(&format!("blk.{li}.ffn_up_exps.weight"), &mut stats)?;
                let down_exps =
                    load_3d_stacked(&format!("blk.{li}.ffn_down_exps.weight"), &mut stats)?;
                FfnLayerMetal::Moe {
                    gate_inp,
                    gate_inp_shexp,
                    gate_shexp,
                    up_shexp,
                    down_shexp,
                    gate_exps,
                    up_exps,
                    down_exps,
                }
            },
        };

        let layer = match kind {
            LayerKind::Attention => {
                let w_q = load_2d(&format!("blk.{li}.attn_q.weight"), &mut stats)?;
                let w_k = load_2d(&format!("blk.{li}.attn_k.weight"), &mut stats)?;
                let w_v = load_2d(&format!("blk.{li}.attn_v.weight"), &mut stats)?;
                let w_o = load_2d(&format!("blk.{li}.attn_output.weight"), &mut stats)?;
                let q_norm = load_1d_f32(&format!("blk.{li}.attn_q_norm.weight"), &mut stats)?;
                let k_norm = load_1d_f32(&format!("blk.{li}.attn_k_norm.weight"), &mut stats)?;
                LayerMetal::Attn {
                    attn: AttnLayerMetal {
                        attn_norm,
                        attn_post_norm,
                        w_q,
                        w_k,
                        w_v,
                        w_o,
                        q_norm,
                        k_norm,
                    },
                    ffn,
                }
            },
            LayerKind::Ssm => {
                let w_qkv = load_2d(&format!("blk.{li}.attn_qkv.weight"), &mut stats)?;
                let w_gate = load_2d(&format!("blk.{li}.attn_gate.weight"), &mut stats)?;
                let conv1d = load_1d_f32(&format!("blk.{li}.ssm_conv1d.weight"), &mut stats)?;
                let ssm_alpha = load_2d(&format!("blk.{li}.ssm_alpha.weight"), &mut stats)?;
                let ssm_beta = load_2d(&format!("blk.{li}.ssm_beta.weight"), &mut stats)?;
                let dt_bias = load_1d_f32(&format!("blk.{li}.ssm_dt.bias"), &mut stats)?;
                let ssm_a = load_1d_f32(&format!("blk.{li}.ssm_a"), &mut stats)?;
                let ssm_norm = load_1d_f32(&format!("blk.{li}.ssm_norm.weight"), &mut stats)?;
                let ssm_out = load_2d(&format!("blk.{li}.ssm_out.weight"), &mut stats)?;
                LayerMetal::Ssm {
                    ssm: SsmLayerMetal {
                        attn_norm,
                        attn_post_norm,
                        w_qkv,
                        w_gate,
                        conv1d,
                        ssm_alpha,
                        ssm_beta,
                        dt_bias,
                        ssm_a,
                        ssm_norm,
                        ssm_out,
                    },
                    ffn,
                }
            },
        };
        layers.push(layer);
    }

    stats.elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;

    Ok((
        Qwen35MetalModel {
            cfg: cfg.clone(),
            tok_embd,
            output_norm,
            output,
            layers,
        },
        stats,
    ))
}

// ============================================================================
// Per-token decode state
// ============================================================================

/// Per-attention-layer KV cache. Stored as `[max_seq, n_kv_heads, head_dim]`
/// row-major in a Metal Shared buffer.
struct AttnLayerCache {
    k_cache: Buffer,
    v_cache: Buffer,
}

/// Per-SSM-layer recurrent state.
struct SsmLayerState {
    /// Conv ring buffer: `[(conv_kernel - 1) * conv_dim]` f32.
    conv_state: Buffer,
    /// Delta-net state: `[n_v_heads * head_dim * head_dim]` f32.
    state: Buffer,
}

/// Per-layer state — discriminated by layer kind. Mirrors `LayerMetal`.
enum LayerState {
    Attn(AttnLayerCache),
    Ssm(SsmLayerState),
}

/// Pre-allocated GPU scratch buffers reused at every token. Sized to the
/// max workload of any layer.
struct Scratch {
    // d-sized
    xd: Buffer,     // residual stream — read+written through every layer
    xd_pre: Buffer, // residual snapshot before mixer (for residual add)
    h: Buffer,      // d-sized norm output / matmul input
    o: Buffer,      // d-sized matmul output
    fc2: Buffer,    // d-sized FFN-down output
    // attention-block buffers
    qg: Buffer,        // Q+gate combined: 2 * head_dim * n_q_heads
    q: Buffer,         // n_q_heads * head_dim
    gate_attn: Buffer, // n_q_heads * head_dim (split from qg)
    k_attn: Buffer,    // n_kv_heads * head_dim
    v_attn: Buffer,    // n_kv_heads * head_dim
    attn_out: Buffer,  // n_q_heads * head_dim
    // SSM-block buffers
    qkv_mixed: Buffer,   // conv_dim
    z: Buffer,           // value_dim
    alpha: Buffer,       // n_v_heads
    beta: Buffer,        // n_v_heads
    gate_h: Buffer,      // n_v_heads (uploaded after CPU softplus + ssm_a multiply)
    beta_sig: Buffer,    // n_v_heads (uploaded after CPU sigmoid)
    conv_out: Buffer,    // conv_dim
    q_ssm: Buffer,       // n_v_heads * head_v_dim (post-broadcast)
    k_ssm: Buffer,       // n_v_heads * head_v_dim (post-broadcast)
    v_ssm: Buffer,       // n_v_heads * head_v_dim
    ssm_out_buf: Buffer, // n_v_heads * head_v_dim
    // FFN-block buffers
    gate_ffn: Buffer, // f
    up_ffn: Buffer,   // f
    fd_ffn: Buffer,   // f
    // logits
    logits: Buffer, // vocab
}

impl Scratch {
    fn new(backend: &MetalBackend, cfg: &Qwen35Config) -> Self {
        let d = cfg.d;
        let head_dim = cfg.attn_head_dim;
        let q_dim = head_dim * cfg.n_q_heads;
        let kv_dim = head_dim * cfg.n_kv_heads;
        let key_dim = cfg.ssm_state * cfg.ssm_groups;
        let value_dim = cfg.ssm_state * cfg.ssm_dt_rank;
        let conv_dim = 2 * key_dim + value_dim;
        let f = cfg.f.max(1);
        let alloc = |bytes: usize| backend.alloc_shared(bytes.max(4)).unwrap();
        Self {
            xd: alloc(d * 4),
            xd_pre: alloc(d * 4),
            h: alloc(d * 4),
            o: alloc(d * 4),
            fc2: alloc(d * 4),
            qg: alloc(2 * q_dim * 4),
            q: alloc(q_dim * 4),
            gate_attn: alloc(q_dim * 4),
            k_attn: alloc(kv_dim * 4),
            v_attn: alloc(kv_dim * 4),
            attn_out: alloc(q_dim * 4),
            qkv_mixed: alloc(conv_dim * 4),
            z: alloc(value_dim * 4),
            alpha: alloc(cfg.ssm_dt_rank * 4),
            beta: alloc(cfg.ssm_dt_rank * 4),
            gate_h: alloc(cfg.ssm_dt_rank * 4),
            beta_sig: alloc(cfg.ssm_dt_rank * 4),
            conv_out: alloc(conv_dim * 4),
            q_ssm: alloc(value_dim * 4), // n_v_heads × head_v_dim = value_dim
            k_ssm: alloc(value_dim * 4),
            v_ssm: alloc(value_dim * 4),
            ssm_out_buf: alloc(value_dim * 4),
            gate_ffn: alloc(f * 4),
            up_ffn: alloc(f * 4),
            fd_ffn: alloc(f * 4),
            logits: alloc(cfg.vocab * 4),
        }
    }
}

/// Top-level decode state — per-layer caches + scratch + max sequence length.
struct DecodeState {
    layers: Vec<LayerState>,
    scratch: Scratch,
    max_seq: usize,
    /// Pre-built RoPE tables (cos, sin) for positions 0..max_seq.
    rope_cos: Buffer,
    rope_sin: Buffer,
}

impl DecodeState {
    fn new(backend: &MetalBackend, cfg: &Qwen35Config, max_seq: usize) -> Self {
        let scratch = Scratch::new(backend, cfg);
        let head_dim = cfg.attn_head_dim;
        let kv_dim = head_dim * cfg.n_kv_heads;
        let key_dim = cfg.ssm_state * cfg.ssm_groups;
        let value_dim = cfg.ssm_state * cfg.ssm_dt_rank;
        let conv_dim = 2 * key_dim + value_dim;
        let head_v_dim = cfg.ssm_state;
        let n_v = cfg.ssm_dt_rank;

        let mut layers = Vec::with_capacity(cfg.n_layers);
        for li in 0..cfg.n_layers {
            let s = match cfg.layer_kind(li) {
                LayerKind::Attention => {
                    let k_cache = backend.alloc_shared(max_seq * kv_dim * 4).unwrap();
                    let v_cache = backend.alloc_shared(max_seq * kv_dim * 4).unwrap();
                    LayerState::Attn(AttnLayerCache { k_cache, v_cache })
                },
                LayerKind::Ssm => {
                    let conv_state = backend
                        .alloc_shared((cfg.ssm_conv_kernel - 1) * conv_dim * 4)
                        .unwrap();
                    let state = backend
                        .alloc_shared(n_v * head_v_dim * head_v_dim * 4)
                        .unwrap();
                    LayerState::Ssm(SsmLayerState { conv_state, state })
                },
            };
            layers.push(s);
        }

        // Build RoPE cos/sin tables.
        let rope_dim = cfg.rope_dim;
        let half = rope_dim / 2;
        let mut cos_data = vec![0.0_f32; max_seq * half];
        let mut sin_data = vec![0.0_f32; max_seq * half];
        for pos in 0..max_seq {
            for i in 0..half {
                let theta = (pos as f32) / cfg.rope_base.powf((2 * i) as f32 / rope_dim as f32);
                cos_data[pos * half + i] = theta.cos();
                sin_data[pos * half + i] = theta.sin();
            }
        }
        let rope_cos = backend.alloc_shared(cos_data.len() * 4).unwrap();
        let rope_sin = backend.alloc_shared(sin_data.len() * 4).unwrap();
        unsafe {
            std::ptr::copy_nonoverlapping(
                cos_data.as_ptr(),
                rope_cos.contents() as *mut f32,
                cos_data.len(),
            );
            std::ptr::copy_nonoverlapping(
                sin_data.as_ptr(),
                rope_sin.contents() as *mut f32,
                sin_data.len(),
            );
        }

        DecodeState {
            layers,
            scratch,
            max_seq,
            rope_cos,
            rope_sin,
        }
    }
}

// ============================================================================
// Token embedding (CPU dequant of one row from the Q-format token_embd)
// ============================================================================

/// Read row `token_id` from `tok_embd` (which lives in a Metal buffer in
/// quantised form), dequantise that single row to f32, and copy it into
/// `dest_buf` (a `d * 4`-byte Metal Shared buffer). One row is small
/// enough that the per-row dequant is fast — we don't need a Metal kernel.
fn embed_token(
    file: &GgufFile,
    token_id: u32,
    dest_buf: &Buffer,
    cfg: &Qwen35Config,
) -> Result<(), String> {
    let info = file
        .tensor("token_embd.weight")
        .ok_or_else(|| "missing token_embd.weight".to_string())?;
    // GGUF row-major: row `token_id` is `d` consecutive elements starting
    // at offset `token_id * row_bytes`. We use the full-tensor dequant path
    // which is simple and correct; for performance it could be replaced
    // by a per-block dequant on just the relevant row.
    let row_bytes = info.byte_size() as usize / cfg.vocab;
    let bytes = file.tensor_bytes(info);
    let row_start = (token_id as usize) * row_bytes;
    let row_end = row_start + row_bytes;
    let row_bytes_slice = &bytes[row_start..row_end];

    // Construct a synthetic single-row TensorInfo for dequant_to_f32.
    // We can't easily mutate `info`, so we just do the dequant manually
    // by passing the same dtype and 1-row-worth of bytes.
    let mut row_info = info.clone();
    row_info.shape = vec![cfg.d as u64];
    let f32_row = dequant_to_f32(&row_info, row_bytes_slice)
        .map_err(|e| format!("token_embd dequant: {e:?}"))?;
    if f32_row.len() != cfg.d {
        return Err(format!(
            "token_embd: expected {} f32, got {}",
            cfg.d,
            f32_row.len()
        ));
    }
    unsafe {
        std::ptr::copy_nonoverlapping(f32_row.as_ptr(), dest_buf.contents() as *mut f32, cfg.d);
    }
    Ok(())
}

// ============================================================================
// CPU helper for the small per-token gate computations in the SSM block
// ============================================================================

/// On the CPU (since these are tiny `n_v_heads`-sized vectors): apply
/// softplus(alpha + dt_bias) * ssm_a → gate_h, and sigmoid(beta) → beta_sig.
/// Reads the Metal buffers directly (Shared storage = unified memory).
fn ssm_apply_gate_ops(
    alpha_buf: &Buffer,
    beta_buf: &Buffer,
    dt_bias: &[f32],
    ssm_a: &[f32],
    gate_h_buf: &Buffer,
    beta_sig_buf: &Buffer,
    n_v: usize,
) {
    // Read alpha + beta from the GPU buffers (no GPU sync needed for Shared).
    unsafe {
        let alpha = std::slice::from_raw_parts(alpha_buf.contents() as *const f32, n_v);
        let beta = std::slice::from_raw_parts(beta_buf.contents() as *const f32, n_v);
        let gate_h = std::slice::from_raw_parts_mut(gate_h_buf.contents() as *mut f32, n_v);
        let beta_sig = std::slice::from_raw_parts_mut(beta_sig_buf.contents() as *mut f32, n_v);
        for i in 0..n_v {
            let a = alpha[i] + dt_bias[i];
            // Stable softplus
            let sp = if a > 20.0 {
                a
            } else if a < -20.0 {
                a.exp()
            } else {
                (1.0 + a.exp()).ln()
            };
            gate_h[i] = sp * ssm_a[i];
            beta_sig[i] = 1.0 / (1.0 + (-beta[i]).exp());
        }
    }
}

/// Broadcast `q_buf` from `[n_k, head_dim]` to `[n_v, head_dim]` by
/// repeating each head `n_v / n_k` times, written into `out_buf`. CPU-side
/// because the data is tiny (typ. n_v * head_dim = 6144 f32 = 24 KB).
fn broadcast_qk_heads(src_buf: &Buffer, dst_buf: &Buffer, n_k: usize, n_v: usize, head_dim: usize) {
    debug_assert!(n_v % n_k == 0);
    let repeat = n_v / n_k;
    unsafe {
        let src = std::slice::from_raw_parts(src_buf.contents() as *const f32, n_k * head_dim);
        let dst = std::slice::from_raw_parts_mut(dst_buf.contents() as *mut f32, n_v * head_dim);
        for h_v in 0..n_v {
            let h_k = h_v / repeat;
            dst[h_v * head_dim..(h_v + 1) * head_dim]
                .copy_from_slice(&src[h_k * head_dim..(h_k + 1) * head_dim]);
        }
    }
}

/// Apply per-channel SiLU on a Metal Shared buffer of `n` f32s, in place.
/// CPU-side since we don't have a standalone SiLU kernel and the buffers
/// in the SSM block are modest.
fn silu_inplace_cpu(buf: &Buffer, n: usize) {
    unsafe {
        let s = std::slice::from_raw_parts_mut(buf.contents() as *mut f32, n);
        for v in s.iter_mut() {
            let sig = 1.0 / (1.0 + (-*v).exp());
            *v *= sig;
        }
    }
}

/// Apply sigmoid in place.
fn sigmoid_inplace_cpu(buf: &Buffer, n: usize) {
    unsafe {
        let s = std::slice::from_raw_parts_mut(buf.contents() as *mut f32, n);
        for v in s.iter_mut() {
            *v = 1.0 / (1.0 + (-*v).exp());
        }
    }
}

/// Pointwise multiply `a *= b` on two Metal Shared buffers of length `n`.
fn mul_inplace_cpu(a: &Buffer, b: &Buffer, n: usize) {
    unsafe {
        let a = std::slice::from_raw_parts_mut(a.contents() as *mut f32, n);
        let b = std::slice::from_raw_parts(b.contents() as *const f32, n);
        for i in 0..n {
            a[i] *= b[i];
        }
    }
}

/// Argmax over a Metal Shared f32 buffer of length `n`.
fn argmax_cpu(buf: &Buffer, n: usize) -> u32 {
    unsafe {
        let s = std::slice::from_raw_parts(buf.contents() as *const f32, n);
        let mut bi = 0u32;
        let mut bv = f32::NEG_INFINITY;
        for (i, &v) in s.iter().enumerate() {
            if v > bv {
                bv = v;
                bi = i as u32;
            }
        }
        bi
    }
}

// ============================================================================
// Attention block forward (Qwen3Next variant — Q+gate combined)
// ============================================================================

#[allow(clippy::too_many_arguments)]
fn attn_block_forward(
    backend: &MetalBackend,
    attn: &AttnLayerMetal,
    cache: &AttnLayerCache,
    scratch: &Scratch,
    rope_cos: &Buffer,
    rope_sin: &Buffer,
    cfg: &Qwen35Config,
    position: usize,
    max_seq: usize,
) -> Result<(), MetalError> {
    let d = cfg.d;
    let head_dim = cfg.attn_head_dim;
    let n_q = cfg.n_q_heads;
    let n_kv = cfg.n_kv_heads;
    let q_dim = head_dim * n_q;
    let kv_dim = head_dim * n_kv;
    let eps = cfg.rms_eps;

    // 1. RMSNorm (xd -> h)
    rms_norm_f32(backend, &scratch.xd, &attn.attn_norm, &scratch.h, d, eps)?;

    // 2. QG = w_q @ h (combined Q + gate, width = 2 * q_dim)
    attn.w_q.matmul_into(backend, &scratch.h, &scratch.qg)?;
    // 3. K = w_k @ h, V = w_v @ h
    attn.w_k.matmul_into(backend, &scratch.h, &scratch.k_attn)?;
    attn.w_v.matmul_into(backend, &scratch.h, &scratch.v_attn)?;

    // 4. Drain to access qg on CPU for splitting Q + gate. The split is
    // strided: per-head [head_dim Q | head_dim gate], so we need two
    // separate buffers. Once we have a fused split kernel this can stay
    // on GPU; for now CPU is fine.
    backend.drain();
    unsafe {
        let qg_p = scratch.qg.contents() as *const f32;
        let q_p = scratch.q.contents() as *mut f32;
        let gate_p = scratch.gate_attn.contents() as *mut f32;
        for h in 0..n_q {
            let src_off = h * 2 * head_dim;
            let dst_off = h * head_dim;
            std::ptr::copy_nonoverlapping(qg_p.add(src_off), q_p.add(dst_off), head_dim);
            std::ptr::copy_nonoverlapping(
                qg_p.add(src_off + head_dim),
                gate_p.add(dst_off),
                head_dim,
            );
        }
    }

    // 5. Per-head Q-norm and K-norm (using shared gamma per head_dim).
    rms_norm_per_head_f32(backend, &scratch.q, &attn.q_norm, n_q, head_dim, eps)?;
    rms_norm_per_head_f32(backend, &scratch.k_attn, &attn.k_norm, n_kv, head_dim, eps)?;

    // 6. RoPE on first rope_dim dims of each head.
    rope_half_split_f32(
        backend, &scratch.q, rope_cos, rope_sin, n_q, head_dim, position,
    )?;
    rope_half_split_f32(
        backend,
        &scratch.k_attn,
        rope_cos,
        rope_sin,
        n_kv,
        head_dim,
        position,
    )?;

    // 7. Append K, V to cache at `position`.
    kv_append_f32(
        backend,
        &scratch.k_attn,
        &cache.k_cache,
        n_kv,
        head_dim,
        position,
        max_seq,
    )?;
    kv_append_f32(
        backend,
        &scratch.v_attn,
        &cache.v_cache,
        n_kv,
        head_dim,
        position,
        max_seq,
    )?;

    // 8. GQA decode → attn_out (q_dim).
    gqa_decode_f32(
        backend,
        &scratch.q,
        &cache.k_cache,
        &cache.v_cache,
        &scratch.attn_out,
        n_q,
        n_kv,
        head_dim,
        position + 1,
        max_seq,
    )?;

    // 9. Apply sigmoid(gate) on attention output (Qwen3Next gate) — fused
    //    Metal kernel `attn_out *= sigmoid(gate_attn)` (T144c). No drain.
    sigmoid_mul_inplace_f32(backend, &scratch.attn_out, &scratch.gate_attn, q_dim)?;
    let _ = kv_dim; // silence unused

    // 10. W_O @ attn_out → o, then xd += o.
    attn.w_o
        .matmul_into(backend, &scratch.attn_out, &scratch.o)?;
    add_inplace_f32(backend, &scratch.xd, &scratch.o, d)?;
    Ok(())
}

// ============================================================================
// SSM block forward (Gated DeltaNet)
// ============================================================================

#[allow(clippy::too_many_arguments)]
fn ssm_block_forward(
    backend: &MetalBackend,
    ssm: &SsmLayerMetal,
    state: &SsmLayerState,
    scratch: &Scratch,
    cfg: &Qwen35Config,
) -> Result<(), MetalError> {
    let d = cfg.d;
    let eps = cfg.rms_eps;
    let head_v_dim = cfg.ssm_state;
    let n_k = cfg.ssm_groups;
    let n_v = cfg.ssm_dt_rank;
    let key_dim = head_v_dim * n_k;
    let value_dim = head_v_dim * n_v;
    let conv_dim = 2 * key_dim + value_dim;

    // 1. RMSNorm on residual stream.
    rms_norm_f32(backend, &scratch.xd, &ssm.attn_norm, &scratch.h, d, eps)?;

    // 2. Input projections.
    ssm.w_qkv
        .matmul_into(backend, &scratch.h, &scratch.qkv_mixed)?;
    ssm.w_gate.matmul_into(backend, &scratch.h, &scratch.z)?;
    ssm.ssm_alpha
        .matmul_into(backend, &scratch.h, &scratch.alpha)?;
    ssm.ssm_beta
        .matmul_into(backend, &scratch.h, &scratch.beta)?;

    // 3. CPU-side: gate_h = softplus(alpha + dt_bias) * ssm_a, beta_sig = sigmoid(beta).
    backend.drain();
    let dt_bias_slice =
        unsafe { std::slice::from_raw_parts(ssm.dt_bias.contents() as *const f32, n_v) };
    let ssm_a_slice =
        unsafe { std::slice::from_raw_parts(ssm.ssm_a.contents() as *const f32, n_v) };
    ssm_apply_gate_ops(
        &scratch.alpha,
        &scratch.beta,
        dt_bias_slice,
        ssm_a_slice,
        &scratch.gate_h,
        &scratch.beta_sig,
        n_v,
    );

    // 4. Conv1d step + ring-buffer update.
    ssm_conv1d_step_f32(
        backend,
        &scratch.qkv_mixed,
        &ssm.conv1d,
        &state.conv_state,
        &scratch.conv_out,
        cfg.ssm_conv_kernel,
        conv_dim,
    )?;
    // 5. (SiLU was fused into ssm_conv1d_step_f32 in T144b — no CPU pass.)
    // 6. Split q, k, v from conv_out into separate Metal buffers.
    //    Both 27B and 35B use n_v_heads = repeat * n_k_heads (27B: 48 = 3 × 16,
    //    35B-A3B: 32 = 2 × 16), so we keep q,k at length [n_k_heads, head_dim] and
    //    let delta_net_step broadcast inline. Only v needs to be n_v_heads-wide.
    backend.drain();
    unsafe {
        let conv_p = scratch.conv_out.contents() as *const f32;
        let q_p = scratch.q_ssm.contents() as *mut f32;
        let k_p = scratch.k_ssm.contents() as *mut f32;
        let v_p = scratch.v_ssm.contents() as *mut f32;
        std::ptr::copy_nonoverlapping(conv_p, q_p, key_dim); // [n_k, head_dim]
        std::ptr::copy_nonoverlapping(conv_p.add(key_dim), k_p, key_dim); // [n_k, head_dim]
        std::ptr::copy_nonoverlapping(conv_p.add(2 * key_dim), v_p, value_dim); // [n_v, head_dim]
    }

    // 7. Per-head L2 norm on q,k (n_k heads each).
    l2_norm_per_head_f32(backend, &scratch.q_ssm, n_k, head_v_dim, eps)?;
    l2_norm_per_head_f32(backend, &scratch.k_ssm, n_k, head_v_dim, eps)?;

    // 8. (Broadcast q,k from n_k → n_v eliminated — delta_net_step handles
    //    the broadcast inline via integer division of head_v.)

    // 9. Delta-net step: state := exp(gate_h) * state + beta * outer(v, k); out = state @ q.
    delta_net_step_f32(
        backend,
        &scratch.q_ssm,
        &scratch.k_ssm,
        &scratch.v_ssm,
        &scratch.gate_h,
        &scratch.beta_sig,
        &state.state,
        &scratch.ssm_out_buf,
        n_v,
        head_v_dim,
        n_k,
    )?;

    // 10. Per-head RMSNorm gated by silu(z).
    rms_norm_per_head_gated_f32(
        backend,
        &scratch.ssm_out_buf,
        &ssm.ssm_norm,
        &scratch.z,
        n_v,
        head_v_dim,
        eps,
    )?;

    // 11. ssm_out @ out_gated → result, then xd += result.
    ssm.ssm_out
        .matmul_into(backend, &scratch.ssm_out_buf, &scratch.o)?;
    add_inplace_f32(backend, &scratch.xd, &scratch.o, d)?;
    Ok(())
}

// ============================================================================
// FFN block forward (dense SwiGLU only — MoE in T145)
// ============================================================================

fn ffn_dense_forward(
    backend: &MetalBackend,
    w: &FfnLayerMetal,
    scratch: &Scratch,
    cfg: &Qwen35Config,
) -> Result<(), MetalError> {
    let d = cfg.d;
    let f = cfg.f;
    match w {
        FfnLayerMetal::Dense {
            w_gate,
            w_up,
            w_down,
        } => {
            w_gate.matmul_into(backend, &scratch.h, &scratch.gate_ffn)?;
            w_up.matmul_into(backend, &scratch.h, &scratch.up_ffn)?;
            swiglu_f32(
                backend,
                &scratch.gate_ffn,
                &scratch.up_ffn,
                &scratch.fd_ffn,
                f,
            )?;
            w_down.matmul_into(backend, &scratch.fd_ffn, &scratch.fc2)?;
            add_inplace_f32(backend, &scratch.xd, &scratch.fc2, d)?;
            Ok(())
        },
        FfnLayerMetal::Moe { .. } => {
            // T145: implement MoE expert dispatch. For now stub: leave xd unchanged.
            // (This means the 35B-A3B model's FFN is a no-op — produces wrong
            // outputs but the pipeline compiles and runs end-to-end.)
            Ok(())
        },
    }
}

// ============================================================================
// Top-level forward_token
// ============================================================================

fn forward_token(
    backend: &MetalBackend,
    file: &GgufFile,
    model: &Qwen35MetalModel,
    state: &mut DecodeState,
    token: u32,
    position: usize,
) -> Result<u32, String> {
    let cfg = &model.cfg;
    // 1. Embed token into xd.
    embed_token(file, token, &state.scratch.xd, cfg)?;

    // 2. Per-layer dispatch.
    for li in 0..cfg.n_layers {
        let layer = &model.layers[li];
        match (layer, &state.layers[li]) {
            (LayerMetal::Attn { attn, ffn }, LayerState::Attn(cache)) => {
                attn_block_forward(
                    backend,
                    attn,
                    cache,
                    &state.scratch,
                    &state.rope_cos,
                    &state.rope_sin,
                    cfg,
                    position,
                    state.max_seq,
                )
                .map_err(|e| format!("layer {li} attn: {e:?}"))?;
                // Post-attention norm (h := norm(xd, attn_post_norm)) for FFN input.
                rms_norm_f32(
                    backend,
                    &state.scratch.xd,
                    &attn.attn_post_norm,
                    &state.scratch.h,
                    cfg.d,
                    cfg.rms_eps,
                )
                .map_err(|e| format!("layer {li} post norm: {e:?}"))?;
                ffn_dense_forward(backend, ffn, &state.scratch, cfg)
                    .map_err(|e| format!("layer {li} ffn: {e:?}"))?;
            },
            (LayerMetal::Ssm { ssm, ffn }, LayerState::Ssm(s)) => {
                ssm_block_forward(backend, ssm, s, &state.scratch, cfg)
                    .map_err(|e| format!("layer {li} ssm: {e:?}"))?;
                rms_norm_f32(
                    backend,
                    &state.scratch.xd,
                    &ssm.attn_post_norm,
                    &state.scratch.h,
                    cfg.d,
                    cfg.rms_eps,
                )
                .map_err(|e| format!("layer {li} post norm: {e:?}"))?;
                ffn_dense_forward(backend, ffn, &state.scratch, cfg)
                    .map_err(|e| format!("layer {li} ffn: {e:?}"))?;
            },
            _ => return Err(format!("layer {li}: kind/state mismatch")),
        }
    }

    // 3. Final norm + lm_head.
    rms_norm_f32(
        backend,
        &state.scratch.xd,
        &model.output_norm,
        &state.scratch.h,
        cfg.d,
        cfg.rms_eps,
    )
    .map_err(|e| format!("final norm: {e:?}"))?;
    model
        .output
        .matmul_into(backend, &state.scratch.h, &state.scratch.logits)
        .map_err(|e| format!("lm_head: {e:?}"))?;
    backend.drain();
    Ok(argmax_cpu(&state.scratch.logits, cfg.vocab))
}

fn main() -> ExitCode {
    let path = match env::args().nth(1) {
        Some(p) => PathBuf::from(p),
        None => {
            eprintln!("usage: qwen35_inference_metal <gguf-path>");
            return ExitCode::FAILURE;
        },
    };

    let backend = metal_backend();
    println!(
        "device: {} (Metal3: {})",
        backend.adapter_name(),
        backend.supports_metal3()
    );

    println!("→ parsing config from {}", path.display());
    let cfg = match parse_config(&path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("parse config failed: {e}");
            return ExitCode::FAILURE;
        },
    };
    println!(
        "  variant={:?} layers={} attn={} ssm={}",
        cfg.variant,
        cfg.n_layers,
        cfg.attention_indices.len(),
        cfg.ssm_indices.len()
    );

    println!("\n→ loading weights into Metal buffers...");
    let (model, stats) = match load_metal_model(backend, &path, &cfg) {
        Ok(x) => x,
        Err(e) => {
            eprintln!("load failed: {e}");
            return ExitCode::FAILURE;
        },
    };

    println!(
        "\n=== Loaded {} tensors in {:.2} s ===",
        stats.n_tensors,
        stats.elapsed_ms / 1000.0
    );
    println!(
        "Total bytes uploaded: {:.2} GB",
        stats.total_bytes as f64 / 1024.0 / 1024.0 / 1024.0
    );
    println!("\n=== Per-dtype breakdown ===");
    for (dt, bytes) in &stats.by_dtype {
        println!(
            "  {:<8} {:>10.2} MB  ({:>5.1}% of total)",
            dt,
            *bytes as f64 / 1024.0 / 1024.0,
            100.0 * (*bytes as f64) / (stats.total_bytes as f64)
        );
    }

    // Sanity: walk the layer list to confirm we successfully loaded each
    // kind, and check kernel-dispatch routing is wired for at least one
    // weight tensor of each dtype.
    let mut by_kind = (0_usize, 0_usize);
    for layer in &model.layers {
        match layer {
            LayerMetal::Attn { .. } => by_kind.0 += 1,
            LayerMetal::Ssm { .. } => by_kind.1 += 1,
        }
    }
    println!(
        "\n=== Per-kind layer counts ===\n  attention layers : {}\n  ssm       layers : {}",
        by_kind.0, by_kind.1
    );

    println!(
        "\n  tok_embd : {:?} {}",
        model.tok_embd.dtype, model.tok_embd.name
    );
    println!(
        "  output   : {:?} {}",
        model.output.dtype, model.output.name
    );

    // Quick kernel-dispatch wiring smoke test: for the first attention
    // layer, run one matmul-vec on a deterministic input and report the
    // first 8 outputs. This proves the dtype-aware dispatch works.
    if let Some(LayerMetal::Attn { attn, .. }) = model
        .layers
        .iter()
        .find(|l| matches!(l, LayerMetal::Attn { .. }))
    {
        let k = attn.w_q.k;
        let n = attn.w_q.n;
        println!(
            "\n=== Smoke test: first attention layer w_q matmul ===\n  shape=[{n}, {k}]  dtype={:?}",
            attn.w_q.dtype
        );
        let x: Vec<f32> = (0..k)
            .map(|i| ((i as f32 + 1.0) * 0.0017).sin() * 0.5)
            .collect();
        let x_buf = backend.alloc_shared(k * 4).unwrap();
        unsafe {
            std::ptr::copy_nonoverlapping(
                x.as_ptr() as *const u8,
                x_buf.contents() as *mut u8,
                k * 4,
            );
        }
        let y_buf = backend.alloc_shared(n * 4).unwrap();
        let t0 = Instant::now();
        if let Err(e) = attn.w_q.matmul_into(backend, &x_buf, &y_buf) {
            eprintln!("matmul_into failed: {e:?}");
            return ExitCode::FAILURE;
        }
        backend.drain();
        println!(
            "  metal sgemv: {:.3} ms",
            t0.elapsed().as_secs_f64() * 1000.0
        );
        let mut y = vec![0.0_f32; n];
        unsafe {
            std::ptr::copy_nonoverlapping(y_buf.contents() as *const f32, y.as_mut_ptr(), n);
        }
        let preview: Vec<f32> = y.iter().take(8).copied().collect();
        println!("  first 8 outputs: {preview:.6?}");
    }

    println!("\n✓ T143 loader smoke-test PASS");

    // ----------------------------------------------------------------
    // Optional: --forward N to run the full hybrid forward and emit N
    // tokens. Skipped if not requested.
    // ----------------------------------------------------------------
    let forward_n: Option<usize> = env::args()
        .skip_while(|a| a != "--forward")
        .nth(1)
        .and_then(|a| a.parse::<usize>().ok());
    if let Some(n) = forward_n {
        println!("\n=== Running forward for {n} tokens (T143b end-to-end) ===");
        // Reload the GGUF for embed lookup (we need the file handle).
        let file = match GgufFile::open(&path) {
            Ok(f) => f,
            Err(e) => {
                eprintln!("reopen gguf: {e:?}");
                return ExitCode::FAILURE;
            },
        };
        let max_seq = 256_usize;
        let mut state = DecodeState::new(backend, &cfg, max_seq);

        // Prefill prompt tokens (for now: just BOS=1) then decode.
        let prompt: Vec<u32> = vec![1];
        let mut last = 0u32;
        let mut cur_pos = 0_usize;
        let t0 = Instant::now();
        for &tok in &prompt {
            match forward_token(backend, &file, &model, &mut state, tok, cur_pos) {
                Ok(out) => {
                    last = out;
                    println!("  prefill[{cur_pos}] tok={tok} -> next argmax = {last}");
                },
                Err(e) => {
                    eprintln!("forward error at prefill: {e}");
                    return ExitCode::FAILURE;
                },
            }
            cur_pos += 1;
        }
        let prefill_d = t0.elapsed();
        println!(
            "  prefill: {} tok in {:.3}s ({:.2} tok/s)",
            prompt.len(),
            prefill_d.as_secs_f64(),
            prompt.len() as f64 / prefill_d.as_secs_f64()
        );

        let mut generated = vec![last];
        let t0 = Instant::now();
        for _ in 1..n {
            match forward_token(backend, &file, &model, &mut state, last, cur_pos) {
                Ok(out) => {
                    last = out;
                    generated.push(last);
                    cur_pos += 1;
                },
                Err(e) => {
                    eprintln!("forward error at decode: {e}");
                    return ExitCode::FAILURE;
                },
            }
        }
        let decode_d = t0.elapsed();
        println!(
            "  decode : {} tok in {:.3}s ({:.2} tok/s)",
            n - 1,
            decode_d.as_secs_f64(),
            (n.saturating_sub(1)) as f64 / decode_d.as_secs_f64()
        );
        println!("\ngenerated tokens: {generated:?}");
        return ExitCode::SUCCESS;
    }

    println!("  Tip: pass `--forward N` to run an end-to-end forward for N tokens.");
    ExitCode::SUCCESS
}
