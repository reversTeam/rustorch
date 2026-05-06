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

use std::collections::{BTreeMap, HashMap};
use std::env;
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use metal::Buffer;
use rustorch_gguf::metadata::{MetaArray, MetaValue};
use rustorch_gguf::{dequant_to_f32, GgmlType, GgufFile, TensorInfo};
use rustorch_llm::qwen35::{describe_model, parse_config, LayerKind, Qwen35Config, Qwen35Variant};
use rustorch_metal::backend::MetalBackend;
use rustorch_metal::backend_singleton::metal_backend;
use rustorch_metal::error::MetalError;
use rustorch_metal::kernels::{
    add_inplace_f32, delta_net_step_f32, delta_net_step_with_l2_f32, gqa_decode_f32, kv_append_f32,
    l2_norm_per_head_f32, rms_norm_f32, rms_norm_per_head_f32, rms_norm_per_head_gated_f32,
    rope_half_split_f32, sgemm_q4_k_f32_simdgroup_matrix_64_into,
    sgemm_q4_k_f32_simdgroup_matrix_into, sgemv_f32_lcpp_simd_into, sgemv_q3_k_f32_lcpp_nsg1_into,
    sgemv_q3_k_f32_lcpp_nsg2_into, sgemv_q4_k_f32_lcpp_nsg2_into,
    sgemv_q4_k_gather_f32_lcpp_nsg2_into, sgemv_q5_k_f32_lcpp_nsg2_into,
    sgemv_q5_k_gather_f32_lcpp_nsg2_into, sgemv_q6_k_f32_lcpp_nsg2_into,
    sgemv_q6_k_gather_f32_lcpp_nsg2_into, sgemv_q8_0_f32_lcpp_nsg2_into, sigmoid_add_moe_f32,
    sigmoid_mul_inplace_f32, split_qg_per_head_f32, split_qkv_f32, ssm_apply_gate_f32,
    ssm_conv1d_step_f32, swiglu_f32, topk_softmax_norm_f32, weighted_add_inplace_f32,
    weighted_reduce_add_f32, zero_f32,
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
            GgmlType::Q3_K => {
                // T158 phase 1b — Q3_K dequant + sgemv (3.44 bpw, -24% DRAM
                // vs Q4_K). Validé numériquement vs CPU `dequant_q3_k`.
                // T160 phase 1 — bundle NSG=2 NR0=1 : 2 simdgroups par TG,
                // chaque simdgroup possède 1 row. Halve la dispatch count vs
                // NSG=1 et améliore la co-residency simdgroup sur Apple GPU.
                // Fallback NSG=1 si Metal3 absent ou K%256 != 0.
                // T160 — Q3_K NSG=2 NR0=1 ix-stripped fast-path. +20 % decode
                // mesuré sur Qwen3-14B Q3_K_M vs T158 NSG=1 (median 32.1 vs 26.8 t/s).
                // Fallback NSG=1 si Metal3 absent ou K%256 != 0. Le switch
                // RUSTORCH_Q3K_FORCE_NSG1=1 conserve le legacy NSG=1 pour
                // bisection / profiling / régression check.
                let force_nsg1 = std::env::var("RUSTORCH_Q3K_FORCE_NSG1").is_ok();
                if !force_nsg1 && self.k % 256 == 0 {
                    sgemv_q3_k_f32_lcpp_nsg2_into(
                        backend,
                        x_buf,
                        &self.buffer,
                        out_buf,
                        self.k,
                        self.n,
                    )
                    .or_else(|_| {
                        sgemv_q3_k_f32_lcpp_nsg1_into(
                            backend,
                            x_buf,
                            &self.buffer,
                            out_buf,
                            self.k,
                            self.n,
                        )
                    })
                } else {
                    sgemv_q3_k_f32_lcpp_nsg1_into(
                        backend,
                        x_buf,
                        &self.buffer,
                        out_buf,
                        self.k,
                        self.n,
                    )
                }
            },
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
                // T151 — Small F32 projections (e.g. ssm_alpha, ssm_beta) now
                // run on GPU via `sgemv_f32_lcpp_simd_into` (1 simdgroup per
                // output column). The previous path was CPU naive matmul +
                // `backend.drain()`, costing ~250 µs/call × 64 calls/token
                // (= 32 SSM layers × 2 F32 projections) = ~16 ms/token =
                // ~23% of the 27B decode budget. The new path keeps the
                // matmul on the same Metal command buffer, no host-side
                // synchronisation needed.
                sgemv_f32_lcpp_simd_into(backend, x_buf, &self.buffer, out_buf, self.k, self.n)
            },
            other => Err(MetalError::Unsupported(format!(
                "HybridMetalWeight::matmul_into: dtype {:?} not supported by any sgemv kernel",
                other
            ))),
        }
    }

    /// T162 — batched matmul `out[M, N] = x[M, K] @ W[N, K]^T` (M ≥ 1).
    ///
    /// Pour M = 1 : équivalent à `matmul_into` (sgemv path optimal).
    /// Pour M ≥ 8 : utilise les kernels SGEMM simdgroup_matrix avec dispatcher
    /// heuristique :
    /// - M ≥ 64 et N % 64 == 0 : `sgemm_q4_k_f32_simdgroup_matrix_64_into`
    ///   (tile 64×64 multi-warp, ×4 vs sgemv loop sur shapes 14B FFN)
    /// - M ≥ 8 et N % 8 == 0 : `sgemm_q4_k_f32_simdgroup_matrix_into`
    ///   (tile 8×8 single-warp, ×2 vs sgemv loop)
    ///
    /// Pour M ∈ [2, 7] : pas encore de path dédié, retourne erreur (le caller
    /// doit padder ou loopper sgemv).
    ///
    /// Pré-conditions : `x_batched` est `[M, K]` f32 row-major, `out_batched`
    /// est `[M, N]` f32 row-major. K doit être multiple de 256 (Q4_K) ou 8 (autres).
    ///
    /// Status : Q4_K supporté avec dispatcher complet. Autres quants : fallback
    /// vers sgemv-loop M fois (à venir, gain attendu mineur car decode reste M=1).
    pub fn matmul_batched_into(
        &self,
        backend: &MetalBackend,
        m: usize,
        x_batched: &Buffer,
        out_batched: &Buffer,
    ) -> Result<(), MetalError> {
        if m == 0 {
            return Ok(());
        }
        if m == 1 {
            return self.matmul_into(backend, x_batched, out_batched);
        }
        match self.dtype {
            GgmlType::Q4_K => {
                if m >= 64 && m % 64 == 0 && self.n % 64 == 0 && self.k % 256 == 0 {
                    sgemm_q4_k_f32_simdgroup_matrix_64_into(
                        backend,
                        x_batched,
                        &self.buffer,
                        out_batched,
                        m,
                        self.n,
                        self.k,
                    )
                } else if m >= 8 && m % 8 == 0 && self.n % 8 == 0 && self.k % 256 == 0 {
                    sgemm_q4_k_f32_simdgroup_matrix_into(
                        backend,
                        x_batched,
                        &self.buffer,
                        out_batched,
                        m,
                        self.n,
                        self.k,
                    )
                } else {
                    Err(MetalError::Unsupported(format!(
                        "HybridMetalWeight::matmul_batched_into: Q4_K SGEMM requires M >= 8, M%8 == 0, N%8 == 0, K%256 == 0 (got M={m}, N={}, K={})",
                        self.n, self.k
                    )))
                }
            },
            other => Err(MetalError::Unsupported(format!(
                "HybridMetalWeight::matmul_batched_into: dtype {:?} not yet supported for M > 1 batched path",
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

/// T152 — Bloc d'experts MoE stocké en UN SEUL buffer Metal Q4_K.
///
/// La forme GGUF est `[k=in_dim, n=out_dim, n_experts]` (les 3 axes dans
/// l'ordre fast→slow). Stockée contiguë en Metal, l'ordre des bytes est
/// `expert.row.col_super_block` (l'expert est l'axe le plus lent), ce qui
/// est ce que le kernel `sgemv_q4_k_gather_f32_lcpp_nsg2` attend.
///
/// Inspiration : MLX `SwitchLinear.weight` (forme `[E, N, K]` quantifiée
/// d'un bloc) ; `affine_gather_qmm_rhs` lit `weight + idx*stride_w`.
pub struct StackedQuantizedExperts {
    /// Tous les bytes des `n_experts` matrices Q-quantisées, contigus.
    pub buffer: Buffer,
    /// Type de quantization (Q4_K typiquement).
    pub dtype: GgmlType,
    /// Nombre d'experts.
    pub n_experts: usize,
    /// Dimension d'entrée par expert (K, p.ex. d=5120 pour gate/up,
    /// expert_f=1024 pour down).
    pub k: usize,
    /// Dimension de sortie par expert (N, p.ex. expert_f=1024 pour
    /// gate/up, d=5120 pour down).
    pub n: usize,
    /// Taille d'un expert en bytes — `n * (k/256) * 144` pour Q4_K.
    /// C'est aussi le `expert_stride_bytes` à passer au kernel gather.
    pub bytes_per_expert: usize,
    /// Nom du tensor GGUF.
    pub name: String,
}

/// FFN sub-block — dense or MoE.
#[allow(missing_docs)]
// T152 — Moe variant is much larger than Dense (3 stacked buffers + 4
// HybridMetalWeights + 1 router buffer vs 3 HybridMetalWeights). Clippy
// suggests boxing — but each layer holds only one variant and they're
// never moved post-load, so the size penalty is inert.
#[allow(clippy::large_enum_variant)]
pub enum FfnLayerMetal {
    Dense {
        w_gate: HybridMetalWeight, // [d, f]
        w_up: HybridMetalWeight,   // [d, f]
        w_down: HybridMetalWeight, // [f, d]
    },
    Moe {
        gate_inp: HybridMetalWeight,
        gate_inp_shexp: Buffer, // f32 [d]
        gate_shexp: HybridMetalWeight,
        up_shexp: HybridMetalWeight,
        down_shexp: HybridMetalWeight,
        // T152 — un seul buffer stacked par projection au lieu de
        // `Vec<HybridMetalWeight>` à n_experts entrées. Permet le path
        // gather sgemv (1 dispatch / projection / layer au lieu de
        // top_k = 8 dispatchs).
        gate_exps_stacked: StackedQuantizedExperts,
        up_exps_stacked: StackedQuantizedExperts,
        down_exps_stacked: StackedQuantizedExperts,
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
    let load_stacked_one_buffer =
        |name: &str, stats: &mut LoadStats| -> Result<StackedQuantizedExperts, String> {
            // T152 — Stacked expert tensors: GGUF shape `[k, n, n_experts]`
            // (les 3 axes dans l'ordre fast → slow). On les charge dans UN
            // SEUL Metal buffer contigu (l'expert est l'axe le plus lent),
            // ce qui est exactement ce que le kernel `sgemv_q4_k_gather_*`
            // attend pour offset = `expert_id * bytes_per_expert`.
            let info = file
                .tensor(name)
                .ok_or_else(|| format!("missing tensor: {name}"))?;
            visit_tensor(info, stats);
            if info.shape.len() != 3 {
                return Err(format!("{name}: expected 3-D, got {:?}", info.shape));
            }
            let k = info.shape[0] as usize;
            let n = info.shape[1] as usize;
            let n_experts = info.shape[2] as usize;
            let total_bytes = info.byte_size() as usize;
            let bytes_per_expert = total_bytes / n_experts;
            let bytes = file.tensor_bytes(info);
            let buffer = backend
                .alloc_shared(total_bytes)
                .map_err(|e| format!("alloc {name}: {e:?}"))?;
            unsafe {
                std::ptr::copy_nonoverlapping(
                    bytes.as_ptr(),
                    buffer.contents() as *mut u8,
                    total_bytes,
                );
            }
            Ok(StackedQuantizedExperts {
                buffer,
                dtype: info.dtype,
                n_experts,
                k,
                n,
                bytes_per_expert,
                name: name.to_string(),
            })
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
        let ffn_pre_norm_name = cfg.variant.ffn_pre_norm_name();
        let attn_post_norm =
            load_1d_f32(&format!("blk.{li}.{ffn_pre_norm_name}.weight"), &mut stats)?;

        // FFN sub-block — same on both layer kinds.
        let ffn = match cfg.variant {
            Qwen35Variant::Qwen3PureTransformer | Qwen35Variant::Dense => {
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
                let gate_exps_stacked =
                    load_stacked_one_buffer(&format!("blk.{li}.ffn_gate_exps.weight"), &mut stats)?;
                let up_exps_stacked =
                    load_stacked_one_buffer(&format!("blk.{li}.ffn_up_exps.weight"), &mut stats)?;
                let down_exps_stacked =
                    load_stacked_one_buffer(&format!("blk.{li}.ffn_down_exps.weight"), &mut stats)?;
                FfnLayerMetal::Moe {
                    gate_inp,
                    gate_inp_shexp,
                    gate_shexp,
                    up_shexp,
                    down_shexp,
                    gate_exps_stacked,
                    up_exps_stacked,
                    down_exps_stacked,
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
    // MoE-specific buffers (only used when variant == Moe)
    moe_logits: Buffer,      // n_experts (router output)
    moe_acc: Buffer,         // d (accumulated weighted expert outputs)
    moe_expert_out: Buffer,  // d (single-expert output, reused per expert)
    moe_gate: Buffer,        // expert_f
    moe_up: Buffer,          // expert_f
    moe_fd: Buffer,          // expert_f
    moe_shared_gate: Buffer, // 1 (scalar — shared expert gate)
    // T152 — gather buffers : `[n_used, ef]` pour gate/up/fd, `[n_used, d]`
    // pour down. Permettent au path MoE gather (1 dispatch / projection /
    // layer) de stocker les outputs des `n_used` experts en parallèle.
    moe_gate_gather: Buffer, // n_used * expert_f
    moe_up_gather: Buffer,   // n_used * expert_f
    moe_fd_gather: Buffer,   // n_used * expert_f
    moe_down_gather: Buffer, // n_used * d
    moe_topw_buf: Buffer,    // n_used (top-K weights f32, GPU-side for reduce)
    // T152.1b — buffer indices GPU-side : produit par `topk_softmax_norm_f32`
    // depuis les routing logits, consommé par les 3 sgemv gather. Élimine le
    // round-trip CPU et le drain qui le précédait.
    moe_indices_buf: Buffer, // n_used (u32)
    // T152.1 — scalar buffer pour `dot(gate_inp_shexp, h)` du shared expert.
    // Calculé via sgemv N=1 sur GPU, lu par sigmoid_add_moe_f32 sans drain.
    moe_dot_scalar: Buffer, // 1 (f32)
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
            // MoE — sized to whichever variant we're loading. n_experts and
            // expert_f are 0 for non-MoE variants; use .max(1) so the
            // buffer alloc still succeeds.
            moe_logits: alloc(cfg.n_experts.max(1) * 4),
            moe_acc: alloc(d * 4),
            moe_expert_out: alloc(d * 4),
            // T152 — gather buffers (sized for n_experts_used × {ef, d}).
            // Pour les variants non-MoE, n_experts_used=0 → max(1) pour
            // que l'alloc soit valide.
            moe_gate_gather: alloc(cfg.n_experts_used.max(1) * cfg.expert_f.max(1) * 4),
            moe_up_gather: alloc(cfg.n_experts_used.max(1) * cfg.expert_f.max(1) * 4),
            moe_fd_gather: alloc(cfg.n_experts_used.max(1) * cfg.expert_f.max(1) * 4),
            moe_down_gather: alloc(cfg.n_experts_used.max(1) * d * 4),
            moe_topw_buf: alloc(cfg.n_experts_used.max(1) * 4),
            moe_indices_buf: alloc(cfg.n_experts_used.max(1) * 4),
            moe_dot_scalar: alloc(4),
            moe_gate: alloc(cfg.expert_f.max(1) * 4),
            moe_up: alloc(cfg.expert_f.max(1) * 4),
            moe_fd: alloc(cfg.expert_f.max(1) * 4),
            moe_shared_gate: alloc(4),
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
                    // Conv ring buffer + delta-net state are recurrent state —
                    // they MUST start at zero, otherwise garbage in the buffers
                    // (Metal makes no zeroing guarantees on `new_buffer`) would
                    // taint every SSM layer from token 0 and produce gibberish.
                    let conv_bytes = (cfg.ssm_conv_kernel - 1) * conv_dim * 4;
                    let state_bytes = n_v * head_v_dim * head_v_dim * 4;
                    let conv_state = backend.alloc_shared(conv_bytes).unwrap();
                    let state = backend.alloc_shared(state_bytes).unwrap();
                    unsafe {
                        std::ptr::write_bytes(conv_state.contents() as *mut u8, 0, conv_bytes);
                        std::ptr::write_bytes(state.contents() as *mut u8, 0, state_bytes);
                    }
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

    // 2. Q (and gate, when applicable). Layout depends on variant:
    //   - qwen3 (pure transformer): w_q outputs [d → q_dim], no gate.
    //   - qwen35 / qwen35moe       : w_q outputs [d → 2 * q_dim], Q + gate combined.
    let has_q_gate = cfg.variant.has_q_gate();
    if has_q_gate {
        attn.w_q.matmul_into(backend, &scratch.h, &scratch.qg)?;
    } else {
        attn.w_q.matmul_into(backend, &scratch.h, &scratch.q)?;
    }
    // 3. K = w_k @ h, V = w_v @ h
    attn.w_k.matmul_into(backend, &scratch.h, &scratch.k_attn)?;
    attn.w_v.matmul_into(backend, &scratch.h, &scratch.v_attn)?;

    // 4. T146b — GPU split of QG into Q + gate per head (only for the
    // Qwen3Next-style variants). For qwen3 the matmul wrote directly into
    // scratch.q above and no split is needed.
    if has_q_gate {
        split_qg_per_head_f32(
            backend,
            &scratch.qg,
            &scratch.q,
            &scratch.gate_attn,
            n_q,
            head_dim,
        )?;
    }

    // 5. Per-head Q-norm and K-norm (using shared gamma per head_dim).
    rms_norm_per_head_f32(backend, &scratch.q, &attn.q_norm, n_q, head_dim, eps)?;
    rms_norm_per_head_f32(backend, &scratch.k_attn, &attn.k_norm, n_kv, head_dim, eps)?;

    // 6. RoPE on first rope_dim dims of each head.
    let rope_dim = cfg.rope_dim;
    rope_half_split_f32(
        backend, &scratch.q, rope_cos, rope_sin, n_q, head_dim, rope_dim, position,
    )?;
    rope_half_split_f32(
        backend,
        &scratch.k_attn,
        rope_cos,
        rope_sin,
        n_kv,
        head_dim,
        rope_dim,
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

    // 9. Apply sigmoid(gate) on attention output (Qwen3Next-only). Pure
    //    qwen3 has no per-head gate so this is skipped.
    if has_q_gate {
        sigmoid_mul_inplace_f32(backend, &scratch.attn_out, &scratch.gate_attn, q_dim)?;
    }
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
    dump_prefix: &str,
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
    dump_buf(
        backend,
        &scratch.h,
        d,
        &format!("{dump_prefix}/ssm/01_norm_h"),
    );

    // 2. Input projections.
    ssm.w_qkv
        .matmul_into(backend, &scratch.h, &scratch.qkv_mixed)?;
    dump_buf(
        backend,
        &scratch.qkv_mixed,
        conv_dim,
        &format!("{dump_prefix}/ssm/02_qkv_mixed"),
    );
    ssm.w_gate.matmul_into(backend, &scratch.h, &scratch.z)?;
    dump_buf(
        backend,
        &scratch.z,
        value_dim,
        &format!("{dump_prefix}/ssm/03_z"),
    );
    ssm.ssm_alpha
        .matmul_into(backend, &scratch.h, &scratch.alpha)?;
    dump_buf(
        backend,
        &scratch.alpha,
        n_v,
        &format!("{dump_prefix}/ssm/04_alpha"),
    );
    ssm.ssm_beta
        .matmul_into(backend, &scratch.h, &scratch.beta)?;
    dump_buf(
        backend,
        &scratch.beta,
        n_v,
        &format!("{dump_prefix}/ssm/05_beta"),
    );

    // 3. T146a — fused GPU kernel: gate_h = softplus(alpha + dt_bias) * ssm_a,
    //    beta_sig = sigmoid(beta). No drain needed.
    ssm_apply_gate_f32(
        backend,
        &scratch.alpha,
        &scratch.beta,
        &ssm.dt_bias,
        &ssm.ssm_a,
        &scratch.gate_h,
        &scratch.beta_sig,
        n_v,
    )?;
    dump_buf(
        backend,
        &scratch.gate_h,
        n_v,
        &format!("{dump_prefix}/ssm/06_gate_h"),
    );
    dump_buf(
        backend,
        &scratch.beta_sig,
        n_v,
        &format!("{dump_prefix}/ssm/07_beta_sig"),
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
    dump_buf(
        backend,
        &scratch.conv_out,
        conv_dim,
        &format!("{dump_prefix}/ssm/08_conv_out"),
    );
    // 5. (SiLU was fused into ssm_conv1d_step_f32 in T144b — no CPU pass.)
    // 6. T151 — GPU-side three-way split of conv_out → (q, k, v). Replaces
    //    the old CPU `drain + ptr::copy_nonoverlapping` path: 32 SSM layers
    //    × 1 drain/token were a major decode bottleneck. The kernel keeps
    //    everything in the same Metal command buffer so commit_async every
    //    5 layers can coalesce GPU work cleanly.
    //    Both 27B and 35B use n_v_heads = repeat * n_k_heads (27B: 48 = 3 × 16,
    //    35B-A3B: 32 = 2 × 16), so we keep q,k at length [n_k_heads, head_dim]
    //    (key_dim each) and let delta_net_step broadcast inline. Only v is
    //    n_v_heads-wide (value_dim).
    split_qkv_f32(
        backend,
        &scratch.conv_out,
        &scratch.q_ssm,
        &scratch.k_ssm,
        &scratch.v_ssm,
        key_dim,
        key_dim,
        value_dim,
    )?;
    dump_buf(
        backend,
        &scratch.q_ssm,
        key_dim,
        &format!("{dump_prefix}/ssm/09_q_split"),
    );
    dump_buf(
        backend,
        &scratch.k_ssm,
        key_dim,
        &format!("{dump_prefix}/ssm/10_k_split"),
    );
    dump_buf(
        backend,
        &scratch.v_ssm,
        value_dim,
        &format!("{dump_prefix}/ssm/11_v_split"),
    );

    // 7+9. T154-fast — Fuse L2 norm de q,k DANS delta_net_step.
    //
    // Avant : 2 dispatches l2_norm_per_head_f32 (q, k) + 1 dispatch
    // delta_net_step_f32 = 3 dispatches/SSM-layer.
    //
    // Maintenant : 1 dispatch delta_net_step_with_l2_f32 qui calcule inv_q
    // et inv_k via simd_sum en début, puis applique en scalaires multiplicatifs
    // sur les simd_sums internes (proj_r *= inv_k, delta_eff = delta * inv_k,
    // final out *= inv_q). State SSM identique numériquement.
    //
    // Switch RUSTORCH_SSM_FORCE_LEGACY_L2=1 pour bisection / régression check.
    let force_legacy_l2 = std::env::var("RUSTORCH_SSM_FORCE_LEGACY_L2").is_ok();
    if force_legacy_l2 {
        // Legacy path : 2 L2 dispatches + delta_net legacy.
        l2_norm_per_head_f32(backend, &scratch.q_ssm, n_k, head_v_dim, eps)?;
        l2_norm_per_head_f32(backend, &scratch.k_ssm, n_k, head_v_dim, eps)?;
        dump_buf(
            backend,
            &scratch.q_ssm,
            key_dim,
            &format!("{dump_prefix}/ssm/12_q_l2"),
        );
        dump_buf(
            backend,
            &scratch.k_ssm,
            key_dim,
            &format!("{dump_prefix}/ssm/13_k_l2"),
        );
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
    } else {
        // T154-fast : fused L2 + delta_net en 1 dispatch.
        delta_net_step_with_l2_f32(
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
            eps,
        )?;
    }
    dump_buf(
        backend,
        &scratch.ssm_out_buf,
        value_dim,
        &format!("{dump_prefix}/ssm/14_dnet_out"),
    );

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
    dump_buf(
        backend,
        &scratch.ssm_out_buf,
        value_dim,
        &format!("{dump_prefix}/ssm/15_gated_norm"),
    );

    // 11. ssm_out @ out_gated → result, then xd += result.
    ssm.ssm_out
        .matmul_into(backend, &scratch.ssm_out_buf, &scratch.o)?;
    dump_buf(backend, &scratch.o, d, &format!("{dump_prefix}/ssm/16_o"));
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
        FfnLayerMetal::Moe {
            gate_inp,
            gate_inp_shexp,
            gate_shexp,
            up_shexp,
            down_shexp,
            gate_exps_stacked,
            up_exps_stacked,
            down_exps_stacked,
        } => {
            let n_experts = cfg.n_experts;
            let n_used = cfg.n_experts_used;
            let ef = cfg.expert_f;

            // 1. Routing logits = gate_inp @ h  → [n_experts]
            gate_inp.matmul_into(backend, &scratch.h, &scratch.moe_logits)?;

            // 2. T152.1b — Top-K + softmax + renormalize 100% GPU.
            //    Avant : drain + CPU softmax + argsort + renormalize. Maintenant :
            //    1 dispatch d'un threadgroup unique 256 threads qui produit
            //    `moe_indices_buf [n_used] u32` + `moe_topw_buf [n_used] f32`.
            //    Économie : 1 drain × 16 MoE layers = 16 drains/token sur 35B-A3B.
            topk_softmax_norm_f32(
                backend,
                &scratch.moe_logits,
                &scratch.moe_indices_buf,
                &scratch.moe_topw_buf,
                n_experts,
                n_used,
            )?;

            // 3. T147a — zero the accumulator on GPU (no drain).
            zero_f32(backend, &scratch.moe_acc, d)?;

            // 4. T152 — gather sgemv path (1 dispatch / projection au lieu de
            //    n_used dispatchs). Inspired by MLX `affine_gather_qmm_rhs`.
            //    Dispatch dynamique selon dtype (Q4_K, Q5_K, Q6_K) —
            //    Qwen3.6-35B-A3B mixe les 3 selon les layers.
            //    T152.1b — indices viennent de `topk_softmax_norm_f32` (GPU
            //    buffer, plus de set_bytes CPU).
            let gather_dispatch = |stacked: &StackedQuantizedExperts,
                                   x: &Buffer,
                                   out: &Buffer,
                                   k: usize,
                                   n: usize,
                                   x_stride: usize|
             -> Result<(), MetalError> {
                match stacked.dtype {
                    GgmlType::Q4_K => sgemv_q4_k_gather_f32_lcpp_nsg2_into(
                        backend,
                        x,
                        &stacked.buffer,
                        &scratch.moe_indices_buf,
                        n_used,
                        out,
                        k,
                        n,
                        stacked.bytes_per_expert,
                        x_stride,
                    ),
                    GgmlType::Q5_K => sgemv_q5_k_gather_f32_lcpp_nsg2_into(
                        backend,
                        x,
                        &stacked.buffer,
                        &scratch.moe_indices_buf,
                        n_used,
                        out,
                        k,
                        n,
                        stacked.bytes_per_expert,
                        x_stride,
                    ),
                    GgmlType::Q6_K => sgemv_q6_k_gather_f32_lcpp_nsg2_into(
                        backend,
                        x,
                        &stacked.buffer,
                        &scratch.moe_indices_buf,
                        n_used,
                        out,
                        k,
                        n,
                        stacked.bytes_per_expert,
                        x_stride,
                    ),
                    other => Err(MetalError::Unsupported(format!(
                        "T152 gather: dtype {:?} not yet supported (only Q4_K/Q5_K/Q6_K)",
                        other
                    ))),
                }
            };

            // gate_proj : [n_used, ef] = stacked_gate[indices, :, :] @ h (broadcast)
            gather_dispatch(
                gate_exps_stacked,
                &scratch.h,
                &scratch.moe_gate_gather,
                d,
                ef,
                0, // x_stride_floats=0 → broadcast
            )
            .map_err(|e| MetalError::Unsupported(format!("moe gate gather: {e:?}")))?;

            // up_proj : [n_used, ef] = stacked_up[indices, :, :] @ h (broadcast)
            gather_dispatch(
                up_exps_stacked,
                &scratch.h,
                &scratch.moe_up_gather,
                d,
                ef,
                0,
            )
            .map_err(|e| MetalError::Unsupported(format!("moe up gather: {e:?}")))?;

            // swiglu : fd_gather[b, i] = silu(gate[b, i]) * up[b, i] sur n_used*ef
            // éléments traités comme un tableau 1D (kernel élément-wise pur).
            swiglu_f32(
                backend,
                &scratch.moe_gate_gather,
                &scratch.moe_up_gather,
                &scratch.moe_fd_gather,
                n_used * ef,
            )?;

            // down_proj : [n_used, d] = stacked_down[indices, :, :] @ fd_gather (per-row)
            gather_dispatch(
                down_exps_stacked,
                &scratch.moe_fd_gather,
                &scratch.moe_down_gather,
                ef,
                d,
                ef, // x_stride_floats=ef → per-row input
            )
            .map_err(|e| MetalError::Unsupported(format!("moe down gather: {e:?}")))?;

            // T152 — somme pondérée multi-row : moe_acc += sum_b top_w[b] * down_gather[b, :]
            weighted_reduce_add_f32(
                backend,
                &scratch.moe_down_gather,
                &scratch.moe_topw_buf,
                &scratch.moe_acc,
                n_used,
                d,
            )?;

            // 5. Shared expert: standard SwiGLU FFN with sigmoid gate scalar.
            //    shared_gate is a vector of size d (per-element gate, not scalar).
            //    llama.cpp uses ffn_gate_inp_shexp · h then sigmoid → scalar per token.
            //    But the GGUF stores ffn_gate_inp_shexp as [d] f32 — applied as a
            //    point-wise (NOT a dot product). Reading llama.cpp again:
            //    `shared_gate = build_lora_mm(ffn_gate_inp_shexp, cur)` with
            //    ffn_gate_inp_shexp of shape [d] would imply a 1×d matrix → output
            //    is a scalar per token. So a dot product.
            gate_shexp.matmul_into(backend, &scratch.h, &scratch.moe_gate)?;
            up_shexp.matmul_into(backend, &scratch.h, &scratch.moe_up)?;
            swiglu_f32(
                backend,
                &scratch.moe_gate,
                &scratch.moe_up,
                &scratch.moe_fd,
                ef,
            )?;
            down_shexp.matmul_into(backend, &scratch.moe_fd, &scratch.moe_expert_out)?;

            // T152.1 — Shared expert gating + final add ENTIÈREMENT GPU.
            // Avant : drain + CPU dot + CPU sigmoid + CPU add. Maintenant :
            // - sgemv N=1 calcule `dot(gate_inp_shexp, h)` dans moe_dot_scalar
            // - `sigmoid_add_moe_f32` lit le scalaire et applique
            //   `xd[i] += moe_acc[i] + sigmoid(scalar) * moe_expert_out[i]`
            // 1 dispatch GPU au lieu de 1 drain + 2 CPU loops sur d éléments.
            // Économie : 1 drain × 16 MoE layers = 16 drains/token sur 35B-A3B.
            sgemv_f32_lcpp_simd_into(
                backend,
                &scratch.h,
                gate_inp_shexp,
                &scratch.moe_dot_scalar,
                d,
                1,
            )?;
            sigmoid_add_moe_f32(
                backend,
                &scratch.moe_acc,
                &scratch.moe_expert_out,
                &scratch.moe_dot_scalar,
                &scratch.xd,
                d,
            )?;
            Ok(())
        },
    }
}

// ============================================================================
// Top-level forward_token
// ============================================================================

// ============================================================================
// T150 — Layer-by-layer bisection helper.
//
// Set `RUSTORCH_DUMP_LAYERS=1` to print mean/std/min/max/nan-count + first
// 8 floats of the residual stream after every step of the forward pass.
// Set `RUSTORCH_DUMP_DIR=/path` to ALSO write the raw f32-LE buffer to
// `<dir>/<label>.bin` for offline numerical diff against a reference run
// (CPU forward, llama.cpp eval-callback, …).
//
// Both flags force a `backend.drain()` at every dump point so any in-flight
// GPU work is visible before we read — perf is intentionally tanked when
// debug is on.
// ============================================================================
fn dump_mode() -> u8 {
    use std::sync::OnceLock;
    static MODE: OnceLock<u8> = OnceLock::new();
    *MODE.get_or_init(|| {
        if env::var("RUSTORCH_DUMP_DIR").is_ok() {
            2
        } else if env::var("RUSTORCH_DUMP_LAYERS").is_ok() {
            1
        } else {
            0
        }
    })
}

fn dump_buf(backend: &MetalBackend, buf: &Buffer, n: usize, label: &str) {
    let mode = dump_mode();
    if mode == 0 || n == 0 {
        return;
    }
    backend.drain();
    let mut v = vec![0.0_f32; n];
    unsafe {
        std::ptr::copy_nonoverlapping(buf.contents() as *const f32, v.as_mut_ptr(), n);
    }
    let mut sum = 0.0_f64;
    let mut sumsq = 0.0_f64;
    let mut mn = f32::INFINITY;
    let mut mx = f32::NEG_INFINITY;
    let mut nan_count = 0_usize;
    for &x in &v {
        if !x.is_finite() {
            nan_count += 1;
            continue;
        }
        let xd = x as f64;
        sum += xd;
        sumsq += xd * xd;
        if x < mn {
            mn = x;
        }
        if x > mx {
            mx = x;
        }
    }
    let valid = (n - nan_count).max(1) as f64;
    let mean = sum / valid;
    let var = (sumsq / valid - mean * mean).max(0.0);
    let std = var.sqrt();
    let preview: Vec<f32> = v.iter().take(8).copied().collect();
    eprintln!(
        "[dump] {label:<42} n={n:>7} mean={mean:>+11.3e} std={std:>9.3e} min={mn:>+9.3e} max={mx:>+9.3e} nan={nan_count} head={preview:.4?}"
    );
    if mode >= 2 {
        if let Ok(dir) = env::var("RUSTORCH_DUMP_DIR") {
            let _ = std::fs::create_dir_all(&dir);
            let safe = label
                .chars()
                .map(|c| match c {
                    '/' | ' ' | '\t' => '_',
                    other => other,
                })
                .collect::<String>();
            let path = std::path::PathBuf::from(dir).join(format!("{safe}.bin"));
            let bytes = unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, n * 4) };
            let _ = std::fs::write(&path, bytes);
        }
    }
}

/// T155 — Profiling instrumented forward pour identifier les hot kernels.
/// Quand `RUSTORCH_PROFILE=1` est set, drain le GPU après chaque phase
/// majeure (embed, attn, ssm, ffn dense, ffn moe, final, lm_head) et
/// accumule les timings. Imprimés à la fin du forward via une global mutex.
/// Coût : ~5-10 ms/token de drains additionnels — acceptable pour diag.
type ProfileMap = std::sync::Mutex<std::collections::BTreeMap<&'static str, (u64, f64)>>;
static PROFILE_ACCUM: std::sync::OnceLock<ProfileMap> = std::sync::OnceLock::new();

fn profile_map() -> &'static ProfileMap {
    PROFILE_ACCUM.get_or_init(|| std::sync::Mutex::new(std::collections::BTreeMap::new()))
}

fn profile_enabled() -> bool {
    use std::sync::OnceLock;
    static PROFILE: OnceLock<bool> = OnceLock::new();
    *PROFILE.get_or_init(|| env::var("RUSTORCH_PROFILE").is_ok())
}

fn profile_record(label: &'static str, dur: std::time::Duration) {
    let mut g = profile_map().lock().unwrap();
    let entry = g.entry(label).or_insert((0u64, 0.0));
    entry.0 += 1;
    entry.1 += dur.as_secs_f64();
}

/// Helper pour timer une phase et accumuler. Drain forcé après.
fn profile_drain_record(backend: &MetalBackend, label: &'static str, t0: std::time::Instant) {
    if profile_enabled() {
        backend.drain();
        profile_record(label, t0.elapsed());
    }
}

fn profile_print_summary() {
    let g = profile_map().lock().unwrap();
    if g.is_empty() {
        return;
    }
    let total: f64 = g.values().map(|(_, t)| *t).sum();
    eprintln!("\n=== RUSTORCH_PROFILE summary (cumulative across all forward calls) ===");
    eprintln!(
        "{:<24} {:>10} {:>14} {:>14} {:>9}",
        "phase", "calls", "total ms", "avg µs/call", "% total"
    );
    let mut sorted: Vec<_> = g.iter().collect();
    sorted.sort_by(|a, b| b.1 .1.partial_cmp(&a.1 .1).unwrap());
    for (label, (n, t)) in sorted {
        eprintln!(
            "{label:<24} {n:>10} {:>14.3} {:>14.2} {:>9.2}",
            t * 1000.0,
            (t * 1e6) / *n as f64,
            100.0 * t / total
        );
    }
    eprintln!(
        "{:<24} {:>10} {:>14.3} {:>14} {:>9.2}",
        "TOTAL",
        "—",
        total * 1000.0,
        "—",
        100.0
    );
}

fn forward_token(
    backend: &MetalBackend,
    file: &GgufFile,
    model: &Qwen35MetalModel,
    state: &mut DecodeState,
    token: u32,
    position: usize,
) -> Result<u32, String> {
    let cfg = &model.cfg;
    let t0 = std::time::Instant::now();
    // 1. Embed token into xd.
    embed_token(file, token, &state.scratch.xd, cfg)?;
    dump_buf(
        backend,
        &state.scratch.xd,
        cfg.d,
        &format!("p{position:03}/embed"),
    );
    profile_drain_record(backend, "embed", t0);

    // 2. Per-layer dispatch.
    for li in 0..cfg.n_layers {
        let layer = &model.layers[li];
        let kind_tag = match layer {
            LayerMetal::Attn { .. } => "attn",
            LayerMetal::Ssm { .. } => "ssm_",
        };
        let t_layer = std::time::Instant::now();
        let _ = t_layer; // silenced; per-block timings below are more useful
        let is_moe = matches!(
            model.layers[li],
            LayerMetal::Attn {
                ffn: FfnLayerMetal::Moe { .. },
                ..
            } | LayerMetal::Ssm {
                ffn: FfnLayerMetal::Moe { .. },
                ..
            }
        );
        match (layer, &state.layers[li]) {
            (LayerMetal::Attn { attn, ffn }, LayerState::Attn(cache)) => {
                let t = std::time::Instant::now();
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
                profile_drain_record(backend, "attn_block", t);
                dump_buf(
                    backend,
                    &state.scratch.xd,
                    cfg.d,
                    &format!("p{position:03}/L{li:02}_{kind_tag}_post_mixer"),
                );
                // Post-attention norm (h := norm(xd, attn_post_norm)) for FFN input.
                let t = std::time::Instant::now();
                rms_norm_f32(
                    backend,
                    &state.scratch.xd,
                    &attn.attn_post_norm,
                    &state.scratch.h,
                    cfg.d,
                    cfg.rms_eps,
                )
                .map_err(|e| format!("layer {li} post norm: {e:?}"))?;
                profile_drain_record(backend, "post_norm", t);
                dump_buf(
                    backend,
                    &state.scratch.h,
                    cfg.d,
                    &format!("p{position:03}/L{li:02}_{kind_tag}_post_norm"),
                );
                let t = std::time::Instant::now();
                ffn_dense_forward(backend, ffn, &state.scratch, cfg)
                    .map_err(|e| format!("layer {li} ffn: {e:?}"))?;
                profile_drain_record(backend, if is_moe { "ffn_moe" } else { "ffn_dense" }, t);
                dump_buf(
                    backend,
                    &state.scratch.xd,
                    cfg.d,
                    &format!("p{position:03}/L{li:02}_{kind_tag}_post_ffn"),
                );
            },
            (LayerMetal::Ssm { ssm, ffn }, LayerState::Ssm(s)) => {
                let prefix = format!("p{position:03}/L{li:02}");
                let t = std::time::Instant::now();
                ssm_block_forward(backend, ssm, s, &state.scratch, cfg, &prefix)
                    .map_err(|e| format!("layer {li} ssm: {e:?}"))?;
                profile_drain_record(backend, "ssm_block", t);
                dump_buf(
                    backend,
                    &state.scratch.xd,
                    cfg.d,
                    &format!("p{position:03}/L{li:02}_{kind_tag}_post_mixer"),
                );
                let t = std::time::Instant::now();
                rms_norm_f32(
                    backend,
                    &state.scratch.xd,
                    &ssm.attn_post_norm,
                    &state.scratch.h,
                    cfg.d,
                    cfg.rms_eps,
                )
                .map_err(|e| format!("layer {li} post norm: {e:?}"))?;
                profile_drain_record(backend, "post_norm", t);
                dump_buf(
                    backend,
                    &state.scratch.h,
                    cfg.d,
                    &format!("p{position:03}/L{li:02}_{kind_tag}_post_norm"),
                );
                let t = std::time::Instant::now();
                ffn_dense_forward(backend, ffn, &state.scratch, cfg)
                    .map_err(|e| format!("layer {li} ffn: {e:?}"))?;
                profile_drain_record(backend, if is_moe { "ffn_moe" } else { "ffn_dense" }, t);
                dump_buf(
                    backend,
                    &state.scratch.xd,
                    cfg.d,
                    &format!("p{position:03}/L{li:02}_{kind_tag}_post_ffn"),
                );
            },
            _ => return Err(format!("layer {li}: kind/state mismatch")),
        }
        // T146c — CPU↔GPU pipelining via mid-token commits. Sweet spot from
        // the 14B (T133) was every 5 layers, gives the GPU steady work while
        // CPU continues encoding the next segment. Disabled when dumping
        // (we drain at every hook anyway).
        if dump_mode() == 0 && (li + 1) % 5 == 0 && li + 1 < cfg.n_layers {
            backend.commit_async();
        }
    }

    // 3. Final norm + lm_head.
    let t = std::time::Instant::now();
    rms_norm_f32(
        backend,
        &state.scratch.xd,
        &model.output_norm,
        &state.scratch.h,
        cfg.d,
        cfg.rms_eps,
    )
    .map_err(|e| format!("final norm: {e:?}"))?;
    profile_drain_record(backend, "final_norm", t);
    dump_buf(
        backend,
        &state.scratch.h,
        cfg.d,
        &format!("p{position:03}/final_norm"),
    );
    let t = std::time::Instant::now();
    model
        .output
        .matmul_into(backend, &state.scratch.h, &state.scratch.logits)
        .map_err(|e| format!("lm_head: {e:?}"))?;
    backend.drain();
    if profile_enabled() {
        profile_record("lm_head", t.elapsed());
    }
    dump_buf(
        backend,
        &state.scratch.logits,
        cfg.vocab,
        &format!("p{position:03}/logits"),
    );
    Ok(argmax_cpu(&state.scratch.logits, cfg.vocab))
}

// ============================================================================
// Byte-level BPE tokenizer using vocab + merges from the GGUF metadata.
// Same algorithm GPT-2 / Qwen2 / Qwen3 / Qwen3.5 / Qwen3.6 use: each input
// byte is mapped to a printable Unicode char, then BPE merges are applied
// greedily by rank, then strings are looked up in the vocab.
// ============================================================================

/// Build the 256-byte → unicode codepoint map used by the GPT-2 / Qwen
/// byte-level pretokenizer. Bytes that are already printable map to
/// themselves; non-printable bytes get assigned codepoints starting at
/// 0x100. Reverse map is via a HashMap built once.
fn bytes_to_unicode_map() -> ([char; 256], HashMap<char, u8>) {
    let mut bs: Vec<u32> = Vec::with_capacity(256);
    for c in (b'!' as u32)..=(b'~' as u32) {
        bs.push(c);
    }
    for c in 0xA1..=0xAC {
        bs.push(c);
    }
    for c in 0xAE..=0xFF {
        bs.push(c);
    }
    let mut cs: Vec<u32> = bs.clone();
    let mut n = 0u32;
    for b in 0..256u32 {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(256 + n);
            n += 1;
        }
    }
    let mut byte_to_char = ['\0'; 256];
    let mut char_to_byte = HashMap::new();
    for (b, c) in bs.iter().zip(cs.iter()) {
        let ch = char::from_u32(*c).unwrap_or('?');
        byte_to_char[*b as usize] = ch;
        char_to_byte.insert(ch, *b as u8);
    }
    (byte_to_char, char_to_byte)
}

struct GgufTokenizer {
    /// Token-id → token string (byte-mapped).
    tokens: Vec<String>,
    /// Token string → id.
    vocab: HashMap<String, u32>,
    /// BPE merge pairs ranked by index (lower = higher priority).
    ranks: HashMap<(String, String), usize>,
    byte_to_char: [char; 256],
    char_to_byte: HashMap<char, u8>,
    eos_id: u32,
}

impl GgufTokenizer {
    fn from_gguf(file: &GgufFile) -> Result<Self, String> {
        let tokens_array = file
            .metadata()
            .get("tokenizer.ggml.tokens")
            .and_then(|v| v.as_array())
            .ok_or("missing tokenizer.ggml.tokens")?;
        let tokens: Vec<String> = match tokens_array {
            MetaArray::String(v) => v.clone(),
            _ => return Err("tokenizer.ggml.tokens is not a String array".into()),
        };
        let merges_array = file
            .metadata()
            .get("tokenizer.ggml.merges")
            .and_then(|v| v.as_array())
            .ok_or("missing tokenizer.ggml.merges")?;
        let merges: Vec<String> = match merges_array {
            MetaArray::String(v) => v.clone(),
            _ => return Err("tokenizer.ggml.merges is not a String array".into()),
        };
        let mut vocab = HashMap::with_capacity(tokens.len());
        for (i, t) in tokens.iter().enumerate() {
            vocab.insert(t.clone(), i as u32);
        }
        let mut ranks = HashMap::with_capacity(merges.len());
        for (i, m) in merges.iter().enumerate() {
            if let Some((a, b)) = m.split_once(' ') {
                ranks.insert((a.to_string(), b.to_string()), i);
            }
        }
        let (byte_to_char, char_to_byte) = bytes_to_unicode_map();
        // EOS may live under tokenizer.ggml.eos_token_id, eot_id, or im_end.
        let eos_id = file
            .metadata()
            .get("tokenizer.ggml.eos_token_id")
            .and_then(|v| match v {
                MetaValue::U32(x) => Some(*x),
                MetaValue::U64(x) => Some(*x as u32),
                MetaValue::I32(x) => Some(*x as u32),
                _ => None,
            })
            .unwrap_or(2);
        Ok(Self {
            tokens,
            vocab,
            ranks,
            byte_to_char,
            char_to_byte,
            eos_id,
        })
    }

    /// Resolve a special-token string like "<|im_start|>" to its single
    /// token id (without going through BPE).
    fn special_id(&self, s: &str) -> Option<u32> {
        self.vocab.get(s).copied()
    }

    /// Run BPE on a single chunk of input chars (byte-mapped). Greedy
    /// lowest-rank merge.
    fn bpe(&self, token: &str) -> Vec<String> {
        let mut word: Vec<String> = token.chars().map(|c| c.to_string()).collect();
        loop {
            let mut best_rank = usize::MAX;
            let mut best_idx = None;
            for i in 0..word.len().saturating_sub(1) {
                if let Some(&r) = self.ranks.get(&(word[i].clone(), word[i + 1].clone())) {
                    if r < best_rank {
                        best_rank = r;
                        best_idx = Some(i);
                    }
                }
            }
            match best_idx {
                Some(i) => {
                    let merged = format!("{}{}", word[i], word[i + 1]);
                    word.splice(i..=i + 1, std::iter::once(merged));
                },
                None => break,
            }
        }
        word
    }

    /// Pre-tokenize `text` into rough chunks (whitespace + word/non-word
    /// boundaries) similar to the GPT-2 regex but using a Rust-only,
    /// `\p{L}`-free approximation. Imperfect on Unicode-heavy input but
    /// adequate for English chat prompts.
    fn pre_tokenize(text: &str) -> Vec<String> {
        let chars: Vec<char> = text.chars().collect();
        let mut out: Vec<String> = Vec::new();
        let mut i = 0;
        while i < chars.len() {
            // Consume leading single space if next char is alphanumeric/punct.
            let mut chunk = String::new();
            if chars[i] == ' ' && i + 1 < chars.len() && !chars[i + 1].is_whitespace() {
                chunk.push(' ');
                i += 1;
            }
            if i >= chars.len() {
                out.push(chunk);
                break;
            }
            let c0 = chars[i];
            if c0.is_alphanumeric() {
                while i < chars.len() && chars[i].is_alphanumeric() {
                    chunk.push(chars[i]);
                    i += 1;
                }
            } else if c0.is_whitespace() {
                while i < chars.len() && chars[i].is_whitespace() {
                    chunk.push(chars[i]);
                    i += 1;
                }
            } else {
                // Punctuation / symbol: take one char at a time.
                chunk.push(c0);
                i += 1;
            }
            out.push(chunk);
        }
        out
    }

    /// Encode raw `text` to token ids. Special tokens like `<|im_start|>`
    /// are recognised verbatim if they're in the vocab.
    fn encode(&self, text: &str) -> Vec<u32> {
        let mut out = Vec::new();
        // Split on the special-token markers we care about. Anything between
        // them goes through BPE. This is a simple linear scan that's enough
        // for ChatML wrapping.
        let specials = [
            "<|im_start|>",
            "<|im_end|>",
            "<|endoftext|>",
            "<|im_sep|>",
            "<|object_ref_start|>",
            "<|object_ref_end|>",
        ];
        let mut rest = text;
        loop {
            // Find the earliest occurrence of any special token.
            let mut earliest: Option<(usize, &str)> = None;
            for s in &specials {
                if let Some(pos) = rest.find(s) {
                    if earliest.map_or(true, |(p, _)| pos < p) {
                        earliest = Some((pos, s));
                    }
                }
            }
            match earliest {
                Some((pos, s)) => {
                    if pos > 0 {
                        let pre = &rest[..pos];
                        out.extend(self.encode_segment(pre));
                    }
                    if let Some(id) = self.special_id(s) {
                        out.push(id);
                    } else {
                        out.extend(self.encode_segment(s));
                    }
                    rest = &rest[pos + s.len()..];
                },
                None => {
                    if !rest.is_empty() {
                        out.extend(self.encode_segment(rest));
                    }
                    break;
                },
            }
        }
        out
    }

    fn encode_segment(&self, segment: &str) -> Vec<u32> {
        let mut ids = Vec::new();
        for chunk in Self::pre_tokenize(segment) {
            let bytes = chunk.as_bytes();
            let mapped: String = bytes
                .iter()
                .map(|&b| self.byte_to_char[b as usize])
                .collect();
            for piece in self.bpe(&mapped) {
                if let Some(&id) = self.vocab.get(&piece) {
                    ids.push(id);
                } else {
                    // Fallback: emit each char as its own token id.
                    for ch in piece.chars() {
                        let s = ch.to_string();
                        if let Some(&id) = self.vocab.get(&s) {
                            ids.push(id);
                        }
                    }
                }
            }
        }
        ids
    }

    /// Decode token ids back to text, reversing the byte-level mapping.
    fn decode(&self, ids: &[u32]) -> String {
        let mut joined = String::new();
        for &id in ids {
            if let Some(t) = self.tokens.get(id as usize) {
                joined.push_str(t);
            }
        }
        let mut bytes: Vec<u8> = Vec::with_capacity(joined.len());
        for ch in joined.chars() {
            if let Some(&b) = self.char_to_byte.get(&ch) {
                bytes.push(b);
            } else {
                // Multi-byte char that's a special token surface form (e.g.
                // <|im_end|>): keep as UTF-8.
                let mut buf = [0u8; 4];
                let s = ch.encode_utf8(&mut buf);
                bytes.extend_from_slice(s.as_bytes());
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }

    fn chatml_prompt(&self, system: &str, user: &str) -> String {
        let mut s = String::new();
        if !system.is_empty() {
            s.push_str(&format!("<|im_start|>system\n{}<|im_end|>\n", system));
        }
        s.push_str(&format!("<|im_start|>user\n{}<|im_end|>\n", user));
        s.push_str("<|im_start|>assistant\n");
        s
    }
}

/// T162 — Bench complet de tous les matmuls Q4_K d'une layer + projection prefill.
///
/// Mesure le speedup `matmul_batched_into` (SGEMM phase 2/3-bis) vs
/// `matmul_into` × M (sgemv loop) sur les 7 matmuls d'une layer transformer
/// dense (Q/K/V/O attn + gate/up/down FFN), sur les bytes Q4_K RÉELS
/// du modèle chargé. Calcule ensuite la projection prefill end-to-end.
fn bench_batched_matmul_on_loaded_weights(backend: &MetalBackend, model: &Qwen35MetalModel) {
    // Sélection : 1ère couche Attn dense Q4_K (= Qwen3-14B style).
    let mut found: Option<(&AttnLayerMetal, &FfnLayerMetal)> = None;
    for layer in &model.layers {
        if let LayerMetal::Attn { attn, ffn } = layer {
            if matches!(ffn, FfnLayerMetal::Dense { .. }) {
                found = Some((attn, ffn));
                break;
            }
        }
    }
    let (attn, ffn) = match found {
        Some(x) => x,
        None => {
            eprintln!("bench-batched: pas de layer Attn+Dense (modèle non-dense ?)");
            return;
        },
    };
    let (w_gate, w_up, w_down) = match ffn {
        FfnLayerMetal::Dense {
            w_gate,
            w_up,
            w_down,
        } => (w_gate, w_up, w_down),
        _ => return,
    };

    let weights: Vec<(&str, &HybridMetalWeight)> = vec![
        ("Q proj    ", &attn.w_q),
        ("K proj    ", &attn.w_k),
        ("V proj    ", &attn.w_v),
        ("O proj    ", &attn.w_o),
        ("FFN gate  ", w_gate),
        ("FFN up    ", w_up),
        ("FFN down  ", w_down),
    ];

    let m_values = [1usize, 8, 24, 64, 128];

    for &m in &m_values {
        println!("\n=== T162 bench-batched layer 0 attn+ffn (M={m}) ===");
        let mut total_batched_ms = 0.0_f64;
        let mut total_loop_ms = 0.0_f64;

        for (label, w) in &weights {
            if w.dtype != GgmlType::Q4_K {
                println!(
                    "  {label} [K={}, N={}] dtype={:?} : skipped (non-Q4_K)",
                    w.k, w.n, w.dtype
                );
                continue;
            }
            let x_buf = backend.alloc_shared(m * w.k * 4).unwrap();
            let out_buf = backend.alloc_shared(m * w.n * 4).unwrap();
            unsafe {
                let x_ptr = x_buf.contents() as *mut f32;
                for i in 0..(m * w.k) {
                    *x_ptr.add(i) = ((i as f32 + 1.0) * 0.001).sin();
                }
            }

            let warmups = 3;
            let iters = 20;

            let batched_ms = {
                let mut errored = false;
                for _ in 0..warmups {
                    if w.matmul_batched_into(backend, m, &x_buf, &out_buf).is_err() {
                        errored = true;
                        break;
                    }
                }
                backend.drain();
                if errored {
                    f64::NAN
                } else {
                    let t0 = std::time::Instant::now();
                    for _ in 0..iters {
                        let _ = w.matmul_batched_into(backend, m, &x_buf, &out_buf);
                    }
                    backend.drain();
                    t0.elapsed().as_secs_f64() * 1000.0 / iters as f64
                }
            };

            for _ in 0..warmups {
                for _row in 0..m {
                    let _ = w.matmul_into(backend, &x_buf, &out_buf);
                }
            }
            backend.drain();
            let t0 = std::time::Instant::now();
            for _ in 0..iters {
                for _row in 0..m {
                    let _ = w.matmul_into(backend, &x_buf, &out_buf);
                }
            }
            backend.drain();
            let loop_ms = t0.elapsed().as_secs_f64() * 1000.0 / iters as f64;

            let speedup = if batched_ms.is_finite() && loop_ms > 0.0 {
                loop_ms / batched_ms
            } else {
                f64::NAN
            };
            println!(
                "  {label} [{:>5} → {:>5}] : batched={:7.3}ms  loop={:7.3}ms  ×{:5.2}",
                w.k, w.n, batched_ms, loop_ms, speedup
            );
            if batched_ms.is_finite() {
                total_batched_ms += batched_ms;
            } else {
                total_batched_ms += loop_ms;
            }
            total_loop_ms += loop_ms;
        }

        let total_speedup = total_loop_ms / total_batched_ms;
        let n_layers = model.cfg.n_layers;
        let full_loop_ms = total_loop_ms * n_layers as f64;
        let full_batched_ms = total_batched_ms * n_layers as f64;
        println!(
            "  TOTAL/layer (matmul only): batched={:7.3}ms  loop={:7.3}ms  ×{:5.2}",
            total_batched_ms, total_loop_ms, total_speedup
        );
        println!(
            "  × {} layers : batched={:.1}ms  loop={:.1}ms",
            n_layers, full_batched_ms, full_loop_ms
        );

        if total_speedup.is_finite() && total_speedup > 1.0 {
            // Approximation Amdahl : si matmul = 60% du forward time, gain
            // total = 1 / (0.4 + 0.6/×_matmul).
            let matmul_frac = 0.60_f64;
            let projected_speedup = 1.0 / (1.0 - matmul_frac + matmul_frac / total_speedup);
            println!(
                "  PROJECTED prefill speedup (matmul ≈ 60% forward) : ×{:5.2}",
                projected_speedup
            );
        }
    }
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
    println!("\n{}", describe_model(&cfg));

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

    // T162 — `--bench-batched` mode : benche `matmul_batched_into` sur une
    // vraie weight Q4_K du modèle chargé (FFN gate de la layer 0). Mesure
    // le speedup SGEMM (phase 2 / 3) vs sgemv-loop M fois sur les exact mêmes
    // bytes que ceux utilisés en inférence.
    if env::args().any(|a| a == "--bench-batched") {
        bench_batched_matmul_on_loaded_weights(backend, &model);
        return ExitCode::SUCCESS;
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
    // Optional inference flags:
    //   --forward N        : generate N tokens (prefill + decode).
    //   --prompt-ids 1,234 : comma-separated token ids to prefill with
    //                        (defaults to [BOS=1] when only --forward
    //                        is provided).
    //   --max-seq N        : KV cache time dimension (default 256).
    //   --eos-id N         : stop generation if any of these IDs is sampled
    //                        (comma-separated). Default empty.
    //   --stream           : print each generated token id as it is decoded.
    // ----------------------------------------------------------------
    let forward_n: Option<usize> = env::args()
        .skip_while(|a| a != "--forward")
        .nth(1)
        .and_then(|a| a.parse::<usize>().ok());
    let prompt_ids: Option<Vec<u32>> =
        env::args()
            .skip_while(|a| a != "--prompt-ids")
            .nth(1)
            .map(|s| {
                s.split(',')
                    .filter_map(|t| t.trim().parse::<u32>().ok())
                    .collect::<Vec<u32>>()
            });
    let max_seq: usize = env::args()
        .skip_while(|a| a != "--max-seq")
        .nth(1)
        .and_then(|a| a.parse::<usize>().ok())
        .unwrap_or(256);
    let eos_ids: Vec<u32> = env::args()
        .skip_while(|a| a != "--eos-id")
        .nth(1)
        .map(|s| {
            s.split(',')
                .filter_map(|t| t.trim().parse::<u32>().ok())
                .collect()
        })
        .unwrap_or_default();
    let stream = env::args().any(|a| a == "--stream");

    // High-level chat: --prompt "text" auto-tokenises through the GGUF's
    // embedded vocab+merges and wraps the prompt in the Qwen ChatML
    // template. Decoded answer is printed at the end.
    let prompt_text: Option<String> = env::args().skip_while(|a| a != "--prompt").nth(1);
    let system_text: String = env::args()
        .skip_while(|a| a != "--system")
        .nth(1)
        .unwrap_or_default();
    let no_chat_template = env::args().any(|a| a == "--no-chat-template");

    if let Some(prompt_str) = prompt_text {
        println!("\n=== Chat mode (Rust tokenizer + ChatML) ===");
        let file = match GgufFile::open(&path) {
            Ok(f) => f,
            Err(e) => {
                eprintln!("reopen gguf: {e:?}");
                return ExitCode::FAILURE;
            },
        };
        let tok = match GgufTokenizer::from_gguf(&file) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("tokenizer init: {e}");
                return ExitCode::FAILURE;
            },
        };
        let n = forward_n.unwrap_or(128);
        let wrapped = if no_chat_template {
            prompt_str.clone()
        } else {
            tok.chatml_prompt(&system_text, &prompt_str)
        };
        let prompt_ids = tok.encode(&wrapped);
        // Add im_end as an extra stop token alongside the EOS.
        let im_end_id = tok.special_id("<|im_end|>").unwrap_or(tok.eos_id);
        let stops: [u32; 2] = [tok.eos_id, im_end_id];
        println!(
            "  prompt ({} tok): {}{}",
            prompt_ids.len(),
            &wrapped.chars().take(120).collect::<String>(),
            if wrapped.chars().count() > 120 {
                "..."
            } else {
                ""
            }
        );
        let mut state = DecodeState::new(backend, &cfg, max_seq);
        let mut last = 0u32;
        let mut cur_pos = 0_usize;
        let t0 = Instant::now();
        for &t in &prompt_ids {
            match forward_token(backend, &file, &model, &mut state, t, cur_pos) {
                Ok(out) => last = out,
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
            prompt_ids.len(),
            prefill_d.as_secs_f64(),
            prompt_ids.len() as f64 / prefill_d.as_secs_f64()
        );

        let mut generated = vec![last];
        let t0 = Instant::now();
        let mut hit_stop = false;
        for _ in 1..n {
            if cur_pos + 1 >= max_seq {
                break;
            }
            match forward_token(backend, &file, &model, &mut state, last, cur_pos) {
                Ok(out) => {
                    last = out;
                    generated.push(last);
                    cur_pos += 1;
                    if stops.contains(&last) {
                        hit_stop = true;
                        break;
                    }
                },
                Err(e) => {
                    eprintln!("forward error at decode: {e}");
                    return ExitCode::FAILURE;
                },
            }
        }
        let decode_d = t0.elapsed();
        let n_decoded = generated.len().saturating_sub(1) as f64;
        println!(
            "  decode : {} tok in {:.3}s ({:.2} tok/s){}",
            n_decoded as usize,
            decode_d.as_secs_f64(),
            n_decoded / decode_d.as_secs_f64().max(1e-9),
            if hit_stop { " [STOP]" } else { "" }
        );
        // Strip a trailing stop token if present so the answer doesn't
        // include the marker text.
        let mut answer_ids = generated.clone();
        if let Some(&l) = answer_ids.last() {
            if stops.contains(&l) {
                answer_ids.pop();
            }
        }
        let answer = tok.decode(&answer_ids);
        println!("\n=== Answer ===");
        println!("{}", answer);
        println!("===");
        profile_print_summary();
        return ExitCode::SUCCESS;
    }

    if let Some(n) = forward_n {
        println!("\n=== Running forward for {n} tokens (T143b end-to-end) ===");
        let file = match GgufFile::open(&path) {
            Ok(f) => f,
            Err(e) => {
                eprintln!("reopen gguf: {e:?}");
                return ExitCode::FAILURE;
            },
        };
        let mut state = DecodeState::new(backend, &cfg, max_seq);

        // Prefill: take the user-supplied prompt ids if given, else fall back
        // to [BOS=1] which matches the legacy --forward smoke-test.
        let prompt: Vec<u32> = prompt_ids.clone().unwrap_or_else(|| vec![1]);
        if prompt.len() >= max_seq {
            eprintln!(
                "prompt has {} tokens but max_seq={max_seq}. Pass --max-seq with a larger value.",
                prompt.len()
            );
            return ExitCode::FAILURE;
        }
        if stream {
            print!("prefill ids: ");
            for &t in &prompt {
                print!("{t} ");
            }
            println!();
        }
        let mut last = 0u32;
        let mut cur_pos = 0_usize;
        let t0 = Instant::now();
        for &tok in &prompt {
            match forward_token(backend, &file, &model, &mut state, tok, cur_pos) {
                Ok(out) => last = out,
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
        if stream {
            println!("first generated id (after prefill): {last}");
        }
        let t0 = Instant::now();
        let mut hit_eos = false;
        for _ in 1..n {
            if cur_pos + 1 >= max_seq {
                eprintln!(
                    "reached max_seq={max_seq} during decode, stopping after {} tokens",
                    generated.len()
                );
                break;
            }
            match forward_token(backend, &file, &model, &mut state, last, cur_pos) {
                Ok(out) => {
                    last = out;
                    generated.push(last);
                    if stream {
                        println!("  decode[{cur_pos}] -> {last}");
                    }
                    cur_pos += 1;
                    if eos_ids.contains(&last) {
                        hit_eos = true;
                        break;
                    }
                },
                Err(e) => {
                    eprintln!("forward error at decode: {e}");
                    return ExitCode::FAILURE;
                },
            }
        }
        let decode_d = t0.elapsed();
        let n_decoded = generated.len().saturating_sub(1) as f64;
        println!(
            "  decode : {} tok in {:.3}s ({:.2} tok/s){}",
            n_decoded as usize,
            decode_d.as_secs_f64(),
            n_decoded / decode_d.as_secs_f64().max(1e-9),
            if hit_eos { " [EOS]" } else { "" }
        );
        println!("\ngenerated tokens: {generated:?}");
        profile_print_summary();
        return ExitCode::SUCCESS;
    }

    println!("  Tip: pass `--forward N` to run an end-to-end forward for N tokens.");
    println!("       Optional: --prompt-ids 1,2,3 (comma-sep ids) --max-seq N --eos-id N --stream");
    ExitCode::SUCCESS
}
