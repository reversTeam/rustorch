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
    add_inplace_batched_f32, add_inplace_f32, argmax_batched_f32, build_em_perm_f32_into,
    delta_net_persistent_scan_f32_into, delta_net_step_f32, delta_net_step_with_l2_f32,
    delta_net_step_with_l2_f32_with_offsets, delta_net_step_with_l2_f32_with_qkv_offsets,
    gather_pack_rows_f32, gqa_decode_batched_f32, gqa_decode_f32, gqa_decode_f32_nsg2,
    gqa_decode_f32_splitk, gqa_decode_f32_splitk_nsg2, gqa_decode_f32_splitk_nsg4,
    kv_append_batched_f32, kv_append_f32, l2_norm_per_head_f32, mul_mm_id_map0_into,
    mul_mm_id_q4_k_f32_into, mul_mm_id_q4_k_q4_k_swiglu_f32_into, mul_mm_id_q5_k_f32_into,
    rms_norm_batched_f32, rms_norm_f32, rms_norm_per_head_batched_f32, rms_norm_per_head_f32,
    rms_norm_per_head_gated_batched_f32, rms_norm_per_head_gated_f32,
    rms_norm_per_head_gated_f32_with_offsets, rope_half_split_f32,
    rope_half_split_partial_batched_f32, scatter_moe_acc_f32_into, sgemm_f32_simdgroup_matrix_into,
    sgemm_q3_k_f32_simdgroup_matrix_64_into, sgemm_q3_k_f32_simdgroup_matrix_into,
    sgemm_q4_k_f32_expert_major_8x64_half_into, sgemm_q4_k_f32_expert_major_8x8_into,
    sgemm_q4_k_f32_simdgroup_matrix_64_into, sgemm_q4_k_f32_simdgroup_matrix_into,
    sgemm_q5_k_f32_expert_major_8x64_half_into, sgemm_q5_k_f32_expert_major_8x8_into,
    sgemm_q6_k_f32_simdgroup_matrix_64_into, sgemm_q6_k_f32_simdgroup_matrix_into,
    sgemm_q8_0_f32_8x64_half_into, sgemm_q8_0_f32_simdgroup_matrix_into, sgemv_f32_lcpp_simd_into,
    sgemv_q3_k_f32_lcpp_nsg1_into, sgemv_q3_k_f32_lcpp_nsg2_into, sgemv_q4_k_f32_lcpp_nsg2_into,
    sgemv_q4_k_gather_f32_lcpp_nsg2_into, sgemv_q4_k_gather_per_token_f32_lcpp_nsg2_into,
    sgemv_q5_k_f32_lcpp_nsg2_into, sgemv_q5_k_gather_f32_lcpp_nsg2_into,
    sgemv_q6_k_f32_lcpp_nsg2_into, sgemv_q6_k_gather_f32_lcpp_nsg2_into,
    sgemv_q8_0_f32_lcpp_nsg2_into, sigmoid_add_moe_batched_f32, sigmoid_add_moe_dot_fused_f32,
    sigmoid_add_moe_f32, sigmoid_mul_inplace_batched_f32, sigmoid_mul_inplace_f32,
    split_qg_per_head_batched_f32, split_qg_per_head_f32, split_qkv_f32,
    ssm_apply_gate_batched_f32, ssm_apply_gate_f32, ssm_conv1d_step_f32,
    ssm_conv1d_step_f32_with_io_offsets, ssm_conv1d_step_f32_with_offset, swiglu_batched_f32,
    swiglu_f32, topk_softmax_norm_batched_f32, topk_softmax_norm_f32,
    topk_softmax_norm_parallel_f32, unpermute_rows_f32, weighted_add_inplace_f32,
    weighted_reduce_add_batched_f32, weighted_reduce_add_f32, weighted_scatter_add_f32, zero_f32,
};

/// T172 Day 5 — Lazy global AMX executor for Innovation 1 hybrid forward.
/// Created on first access (when RUSTORCH_AMX_ROUTING_ASYNC=1 triggers the
/// hybrid path). Spawns one CPU worker thread that lives for the process.
fn amx_executor(backend: &MetalBackend) -> &'static rustorch_metal::async_amx::AsyncAmxExecutor {
    use std::sync::OnceLock;
    static EXEC: OnceLock<rustorch_metal::async_amx::AsyncAmxExecutor> = OnceLock::new();
    EXEC.get_or_init(|| {
        let event = backend.device.new_shared_event();
        rustorch_metal::async_amx::AsyncAmxExecutor::with_event(event)
    })
}

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
                // T177 (rejeté) — wiring qmv_fast ici a donné 0% gain mesurable
                // sur 35B-A3B decode (matmul_into Q4_K = routing matmul ~50µs/call,
                // pas le hot path). Cf. note T168 dans kernels.rs:7901.
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
                // T151 — Small F32 projections on GPU via `sgemv_f32_lcpp_simd_into`.
                //
                // T183 (rejected end-to-end) — wrote `sgemv_f32_cached_x_into`
                // that caches x[K] in threadgroup memory. Microbench: ×2 speedup
                // (22µs → 11µs, parity byte-identical). End-to-end on natural
                // prompts: WASH (-1% to -2%). Likely cause: in production the
                // L2 cache already absorbs x reads, so our explicit TG-mem
                // caching is pure overhead. Lesson: microbench gains require
                // end-to-end confirmation. Kernel kept dead-code in kernels.rs.
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
            GgmlType::Q3_K => {
                // T162 phase 7 (8×8) + phase 7-bis (64×64 multi-warp).
                if m >= 64 && m % 64 == 0 && self.n % 64 == 0 && self.k % 256 == 0 {
                    sgemm_q3_k_f32_simdgroup_matrix_64_into(
                        backend,
                        x_batched,
                        &self.buffer,
                        out_batched,
                        m,
                        self.n,
                        self.k,
                    )
                } else if m >= 8 && m % 8 == 0 && self.n % 8 == 0 && self.k % 256 == 0 {
                    sgemm_q3_k_f32_simdgroup_matrix_into(
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
                        "HybridMetalWeight::matmul_batched_into: Q3_K SGEMM requires M >= 8, M%8 == 0, N%8 == 0, K%256 == 0 (got M={m}, N={}, K={})",
                        self.n, self.k
                    )))
                }
            },
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
            GgmlType::Q6_K => {
                // T162 phase 5 (8×8) + phase 5-bis (64×64 multi-warp).
                // Heuristique : M ≥ 64 et alignement 64 → multi-warp.
                if m >= 64 && m % 64 == 0 && self.n % 64 == 0 && self.k % 256 == 0 {
                    sgemm_q6_k_f32_simdgroup_matrix_64_into(
                        backend,
                        x_batched,
                        &self.buffer,
                        out_batched,
                        m,
                        self.n,
                        self.k,
                    )
                } else if m >= 8 && m % 8 == 0 && self.n % 8 == 0 && self.k % 256 == 0 {
                    sgemm_q6_k_f32_simdgroup_matrix_into(
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
                        "HybridMetalWeight::matmul_batched_into: Q6_K SGEMM requires M >= 8, M%8 == 0, N%8 == 0, K%256 == 0 (got M={m}, N={}, K={})",
                        self.n, self.k
                    )))
                }
            },
            GgmlType::F32 => {
                // T162 phase 9e — F32 SGEMM pour les petites projections SSM
                // (ssm_alpha, ssm_beta : K=d, N=n_v ~ 32-48).
                // Pré-conditions : M%8, N%8, K%8 (cf sgemm_f32_simdgroup_matrix).
                if m >= 8 && m % 8 == 0 && self.n % 8 == 0 && self.k % 8 == 0 {
                    sgemm_f32_simdgroup_matrix_into(
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
                        "HybridMetalWeight::matmul_batched_into: F32 SGEMM requires M%8, N%8, K%8 (got M={m}, N={}, K={})",
                        self.n, self.k
                    )))
                }
            },
            GgmlType::Q5_K => {
                // T175 P2 — Q5_K SGEMM batched via expert_major kernel en mode
                // "single-expert" : tile_expert_ids = [0, ...] et expert_stride
                // = 0. Le kernel calcule W = W_stacked + 0 × 0 = W_stacked
                // (matrice unique). Élimine le fallback drain×M qui coûtait
                // 128 × 250 µs = 32 ms / matmul sur prefill 35B-A3B.
                if m >= 8 && m % 8 == 0 && self.n % 8 == 0 && self.k % 256 == 0 {
                    let n_tiles = m / 8;
                    let zeros = vec![0u32; n_tiles];
                    let tile_buf = backend.alloc_shared(n_tiles * 4)?;
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            zeros.as_ptr(),
                            tile_buf.contents() as *mut u32,
                            n_tiles,
                        );
                    }
                    sgemm_q5_k_f32_expert_major_8x8_into(
                        backend,
                        x_batched,
                        &self.buffer,
                        &tile_buf,
                        out_batched,
                        m,
                        self.n,
                        self.k,
                        0, // expert_stride_bytes = 0 → single-expert mode
                    )
                } else {
                    Err(MetalError::Unsupported(format!(
                        "HybridMetalWeight::matmul_batched_into: Q5_K SGEMM requires M%8, N%8, K%256 (got M={m}, N={}, K={})",
                        self.n, self.k
                    )))
                }
            },
            GgmlType::Q8_0 => {
                // T175 — Q8_0 SGEMM batched. Critical for shared expert weights
                // and attn projections on Qwen3.6 35B-A3B (Q8_0 = 9% of model).
                // 8×64 multi-warp half MMA when N%64 (1.25× vs 8×8), fallback 8×8.
                if m >= 8 && m % 8 == 0 && self.n % 64 == 0 && self.k % 32 == 0 {
                    sgemm_q8_0_f32_8x64_half_into(
                        backend,
                        x_batched,
                        &self.buffer,
                        out_batched,
                        m,
                        self.n,
                        self.k,
                    )
                } else if m >= 8 && m % 8 == 0 && self.n % 8 == 0 && self.k % 32 == 0 {
                    sgemm_q8_0_f32_simdgroup_matrix_into(
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
                        "HybridMetalWeight::matmul_batched_into: Q8_0 SGEMM requires M%8, N%8, K%32 (got M={m}, N={}, K={})",
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

/// T180 — Quantize a f32 weight slice to Q8_0 format.
///
/// Q8_0 super-block: 32 weights → (fp16 scale, [int8; 32]) = 34 bytes.
/// Output layout: contiguous blocks, total size = `(n_weights / 32) * 34` bytes.
///
/// Used at load-time to convert F32 routing matrices (`ffn_gate_inp.weight`,
/// 2 MB/layer × 40 layers = 80 MB/token of F32 bandwidth) to Q8_0 (4× compression,
/// faster `sgemv_q8_0_f32_lcpp_nsg2` dispatch path).
///
/// Pre-conditions: `weights.len() % 32 == 0`.
fn quantize_f32_to_q8_0(weights: &[f32]) -> Vec<u8> {
    use half::f16;
    assert_eq!(
        weights.len() % 32,
        0,
        "Q8_0 requires multiple of 32 weights"
    );
    let n_blocks = weights.len() / 32;
    let mut out = vec![0u8; n_blocks * 34];
    for b in 0..n_blocks {
        let block = &weights[b * 32..(b + 1) * 32];
        let amax = block.iter().fold(0.0_f32, |acc, &x| acc.max(x.abs()));
        let d = amax / 127.0;
        let inv_d = if d != 0.0 { 1.0 / d } else { 0.0 };
        let off = b * 34;
        let d_h = f16::from_f32(d).to_le_bytes();
        out[off] = d_h[0];
        out[off + 1] = d_h[1];
        for i in 0..32 {
            let q = (block[i] * inv_d).round().clamp(-127.0, 127.0) as i8;
            out[off + 2 + i] = q as u8;
        }
    }
    out
}

/// T180 — Load a 2D tensor from GGUF, optionally re-quantizing F32 weights to Q8_0.
///
/// If `requantize_f32_to_q8_0` is true AND the source tensor is F32, the f32 bytes
/// are read, quantized to Q8_0 in-place, and uploaded as Q8_0. The returned
/// `HybridMetalWeight` has `dtype = Q8_0` so `matmul_into` automatically routes
/// to the Q8_0 sgemv kernel (4× less bandwidth, faster dispatch).
fn load_quant_2d_maybe_requantize(
    backend: &MetalBackend,
    file: &GgufFile,
    info: &TensorInfo,
    requantize_f32_to_q8_0: bool,
) -> Result<HybridMetalWeight, String> {
    if info.shape.len() != 2 {
        return Err(format!(
            "{}: expected 2-D tensor, got shape {:?}",
            info.name, info.shape
        ));
    }
    let k = info.shape[0] as usize;
    let n = info.shape[1] as usize;
    if requantize_f32_to_q8_0 && info.dtype == GgmlType::F32 {
        // Read F32 bytes, reinterpret as f32 slice, quantize to Q8_0.
        let bytes = file.tensor_bytes(info);
        let n_floats = bytes.len() / 4;
        if k * n != n_floats {
            return Err(format!(
                "{}: F32 byte count {} != k*n {}",
                info.name,
                n_floats,
                k * n
            ));
        }
        if k % 32 != 0 {
            return Err(format!(
                "{}: F32 K={k} not multiple of 32, cannot quantize to Q8_0",
                info.name
            ));
        }
        let weights: &[f32] =
            unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const f32, n_floats) };
        let q8_bytes = quantize_f32_to_q8_0(weights);
        let buffer = backend
            .alloc_shared(q8_bytes.len())
            .map_err(|e| format!("alloc {}: {:?}", info.name, e))?;
        unsafe {
            std::ptr::copy_nonoverlapping(
                q8_bytes.as_ptr(),
                buffer.contents() as *mut u8,
                q8_bytes.len(),
            );
        }
        return Ok(HybridMetalWeight {
            buffer,
            dtype: GgmlType::Q8_0,
            k,
            n,
            name: info.name.clone(),
        });
    }
    // Normal path: copy raw bytes.
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

    // T180 — env flag to opt-in re-quantize F32 routing matrices to Q8_0 at
    // load time. When set, `load_2d_routing` (used only for `ffn_gate_inp`)
    // converts F32 weights → Q8_0 (4× compression, faster dispatch path).
    let routing_quant_q8 = std::env::var("RUSTORCH_ROUTING_QUANT")
        .map(|v| v.eq_ignore_ascii_case("q8_0"))
        .unwrap_or(false);

    let load_2d = |name: &str, stats: &mut LoadStats| -> Result<HybridMetalWeight, String> {
        let info = file
            .tensor(name)
            .ok_or_else(|| format!("missing tensor: {name}"))?;
        visit_tensor(info, stats);
        load_quant_2d(backend, &file, info)
    };
    // T180 — variant that re-quantizes F32 → Q8_0 for routing matrices when
    // RUSTORCH_ROUTING_QUANT=q8_0. Currently unused after T180 rejection;
    // kept available for future re-test on models where routing is the actual
    // bottleneck (e.g. larger n_experts where bandwidth dominates).
    #[allow(dead_code)]
    let _load_2d_routing =
        |name: &str, stats: &mut LoadStats| -> Result<HybridMetalWeight, String> {
            let info = file
                .tensor(name)
                .ok_or_else(|| format!("missing tensor: {name}"))?;
            visit_tensor(info, stats);
            load_quant_2d_maybe_requantize(backend, &file, info, routing_quant_q8)
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
    // Some Qwen models (small variants like Qwen3.5-4B) tie the output to the
    // token embedding (no separate output.weight). Detect & reuse.
    let output = if file.tensor("output.weight").is_some() {
        load_2d("output.weight", &mut stats)?
    } else {
        eprintln!("  [tied embeddings detected — reusing token_embd as output]");
        load_2d("token_embd.weight", &mut stats)?
    };

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
                // T180 (rejected) — tried routing F32 → Q8_0 quant via load_2d_routing.
                // Result: wash -1% on 5 prompts × 3 runs (within ±2% noise).
                // Root cause: profile drain artifact made moe.routing look like 5.5ms
                // (23% of decode), but real per-call cost is ~1-2 µs in chained mode.
                // The F32 sgemv was already efficient; quant compression saved
                // bandwidth on a non-bottleneck. `load_2d_routing` kept available
                // for re-test on different models (e.g. larger n_experts).
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

/// T162 phase 9b — Maximum batch size for `attn_block_forward_batch` /
/// upcoming `forward_batch` (qwen35). 128 mirrors qwen3-14B's B_MAX.
const B_MAX_BATCH: usize = 128;

/// T162 phase 9b — Per-attn-block batched scratch buffers (B_MAX-sized).
/// Caller-allocated so phase 9d can reuse the same buffers across all attn
/// layers in the batched forward pass.
///
/// Buffers are sized for the worst-case batch (`B_MAX_BATCH × per-token`),
/// smaller B uses the prefix.
struct BatchScratchAttn {
    h: Buffer,         // [B_MAX, d]
    qg: Buffer,        // [B_MAX, 2 * q_dim]  (only used when has_q_gate)
    q: Buffer,         // [B_MAX, q_dim]
    gate_attn: Buffer, // [B_MAX, q_dim]
    k_attn: Buffer,    // [B_MAX, kv_dim]
    v_attn: Buffer,    // [B_MAX, kv_dim]
    attn_out: Buffer,  // [B_MAX, q_dim]
    o: Buffer,         // [B_MAX, d]
}

impl BatchScratchAttn {
    fn new(backend: &MetalBackend, cfg: &Qwen35Config) -> Self {
        let d = cfg.d;
        let head_dim = cfg.attn_head_dim;
        let q_dim = head_dim * cfg.n_q_heads;
        let kv_dim = head_dim * cfg.n_kv_heads;
        let b = B_MAX_BATCH;
        let alloc = |bytes: usize| backend.alloc_shared(bytes.max(4)).unwrap();
        Self {
            h: alloc(b * d * 4),
            qg: alloc(b * 2 * q_dim * 4),
            q: alloc(b * q_dim * 4),
            gate_attn: alloc(b * q_dim * 4),
            k_attn: alloc(b * kv_dim * 4),
            v_attn: alloc(b * kv_dim * 4),
            attn_out: alloc(b * q_dim * 4),
            o: alloc(b * d * 4),
        }
    }
}

/// T162 phase 9b — dispatcher batched matmul Q3_K/Q4_K/Q6_K → SGEMM
/// simdgroup_matrix (M=B). Mirrors `dispatch_batched_matmul` in
/// qwen_inference_metal.rs.
///
/// Pour M=1 fallback `matmul_into` (sgemv path optimal). Pour M ≥ 8 et
/// alignement N % 8, K % 256 → SGEMM tile 8×8 ou 64×64. Pour M ∈ [2,7]
/// fallback à un loop M× sgemv (CPU-side dispatch).
fn dispatch_batched_attn_matmul(
    backend: &MetalBackend,
    w: &HybridMetalWeight,
    m: usize,
    x_batched: &Buffer,
    out_batched: &Buffer,
) -> Result<(), MetalError> {
    if m == 0 {
        return Ok(());
    }
    if m == 1 {
        return w.matmul_into(backend, x_batched, out_batched);
    }
    // M ≥ 8 et alignement → SGEMM via matmul_batched_into si dtype supporté.
    // Fallback CPU-copy loop si matmul_batched_into renvoie Unsupported (e.g.
    // Q5_K, Q8_0 pas encore portés en SGEMM).
    if m >= 8 && m % 8 == 0 && w.n % 8 == 0 && w.k % 256 == 0 {
        match w.matmul_batched_into(backend, m, x_batched, out_batched) {
            Ok(()) => return Ok(()),
            Err(MetalError::Unsupported(_)) => { /* fallthrough to loop */ },
            Err(e) => return Err(e),
        }
    }
    // Sinon : loop sgemv M× (sub-buffers offset par token via CPU memcpy).
    // CRITICAL : drain BEFORE le memcpy CPU pour forcer la flush GPU→shared
    // memory de `x_batched` (sinon on lit des données stale écrites par le
    // dispatch précédent encore en flight). Pareil entre matmul_into et le
    // memcpy de retour.
    backend.drain();
    let x_per_token = w.k * 4;
    let out_per_token = w.n * 4;
    for i in 0..m {
        let x_view = backend.alloc_shared(x_per_token).unwrap();
        let out_view = backend.alloc_shared(out_per_token).unwrap();
        unsafe {
            let src = (x_batched.contents() as *const u8).add(i * x_per_token);
            std::ptr::copy_nonoverlapping(src, x_view.contents() as *mut u8, x_per_token);
        }
        w.matmul_into(backend, &x_view, &out_view)?;
        backend.drain();
        unsafe {
            let dst = (out_batched.contents() as *mut u8).add(i * out_per_token);
            std::ptr::copy_nonoverlapping(out_view.contents() as *const u8, dst, out_per_token);
        }
    }
    Ok(())
}

/// T162 phase 9b — Batched variant of `attn_block_forward` for B-token prefill.
///
/// Equivalent to calling `attn_block_forward` B times sequentially at positions
/// `pos_base..pos_base+B`, but uses batched primitives :
///   - `rms_norm_batched_f32` (1 dispatch vs B)
///   - SGEMM via `dispatch_batched_attn_matmul` (1 dispatch / projection vs B)
///   - `split_qg_per_head_batched_f32` (1 dispatch vs B, only if `has_q_gate`)
///   - `rms_norm_per_head_batched_f32` (1 dispatch vs B, for Q + K norms)
///   - `rope_half_split_partial_batched_f32` (1 dispatch vs B)
///   - `kv_append_batched_f32` (1 dispatch vs B)
///   - `gqa_decode_batched_f32` (1 dispatch vs B, causal mask multi-position)
///   - `sigmoid_mul_inplace_batched_f32` (1 dispatch vs B, only if `has_q_gate`)
///   - `add_inplace_batched_f32` for the final residual.
///
/// Le KV cache est mis à jour aux positions `pos_base + (0..B)` (causalité
/// préservée par `gqa_decode_batched_f32`).
///
/// Pré-conditions :
///   - `xd_batched` : `[B, d]` row-major (in/out — résidu cumulé sur place)
///   - `cache.k_cache`, `cache.v_cache` : `[max_seq, n_kv_heads, head_dim]`
///   - `1 ≤ b ≤ B_MAX_BATCH`
///
/// Status : non wiré dans `forward_token` — fonction standalone testable
/// via le futur `--bench-attn-batch` mode (phase 9d).
#[allow(dead_code, clippy::too_many_arguments)]
fn attn_block_forward_batch(
    backend: &MetalBackend,
    attn: &AttnLayerMetal,
    cache: &AttnLayerCache,
    xd_batched: &Buffer,
    rope_cos: &Buffer,
    rope_sin: &Buffer,
    scratch: &BatchScratchAttn,
    cfg: &Qwen35Config,
    pos_base: usize,
    b: usize,
    max_seq: usize,
) -> Result<(), MetalError> {
    if b == 0 || b > B_MAX_BATCH {
        return Err(MetalError::ShapeMismatch(format!(
            "attn_block_forward_batch: B must be in 1..={B_MAX_BATCH} (got {b})"
        )));
    }
    let d = cfg.d;
    let head_dim = cfg.attn_head_dim;
    let n_q = cfg.n_q_heads;
    let n_kv = cfg.n_kv_heads;
    let q_dim = head_dim * n_q;
    let eps = cfg.rms_eps;
    let has_q_gate = cfg.variant.has_q_gate();

    // 1. Batched RMSNorm (xd → h).
    rms_norm_batched_f32(backend, xd_batched, &attn.attn_norm, &scratch.h, d, b, eps)?;

    // 2. Q (and gate, when applicable). w_q outputs either q_dim or 2 * q_dim.
    if has_q_gate {
        dispatch_batched_attn_matmul(backend, &attn.w_q, b, &scratch.h, &scratch.qg)?;
    } else {
        dispatch_batched_attn_matmul(backend, &attn.w_q, b, &scratch.h, &scratch.q)?;
    }
    // 3. K, V batched matmul.
    dispatch_batched_attn_matmul(backend, &attn.w_k, b, &scratch.h, &scratch.k_attn)?;
    dispatch_batched_attn_matmul(backend, &attn.w_v, b, &scratch.h, &scratch.v_attn)?;

    // 4. Batched per-head split of QG → Q + gate (Qwen3Next variants only).
    if has_q_gate {
        split_qg_per_head_batched_f32(
            backend,
            &scratch.qg,
            &scratch.q,
            &scratch.gate_attn,
            n_q,
            head_dim,
            b,
        )?;
    }

    // 5. Batched per-head Q-norm and K-norm (shared gamma per head_dim).
    rms_norm_per_head_batched_f32(backend, &scratch.q, &attn.q_norm, n_q, head_dim, b, eps)?;
    rms_norm_per_head_batched_f32(
        backend,
        &scratch.k_attn,
        &attn.k_norm,
        n_kv,
        head_dim,
        b,
        eps,
    )?;

    // 6. Batched partial RoPE on first rope_dim dims of each head.
    let rope_dim = cfg.rope_dim;
    rope_half_split_partial_batched_f32(
        backend, &scratch.q, rope_cos, rope_sin, n_q, head_dim, rope_dim, pos_base, b,
    )?;
    rope_half_split_partial_batched_f32(
        backend,
        &scratch.k_attn,
        rope_cos,
        rope_sin,
        n_kv,
        head_dim,
        rope_dim,
        pos_base,
        b,
    )?;

    // 7. Batched KV cache append at positions pos_base..pos_base+B.
    kv_append_batched_f32(
        backend,
        &scratch.k_attn,
        &cache.k_cache,
        n_kv,
        head_dim,
        pos_base,
        b,
        max_seq,
    )?;
    kv_append_batched_f32(
        backend,
        &scratch.v_attn,
        &cache.v_cache,
        n_kv,
        head_dim,
        pos_base,
        b,
        max_seq,
    )?;

    // 8. Batched GQA decode with causal mask (each token i attends to KV[..pos_base+i+1]).
    gqa_decode_batched_f32(
        backend,
        &scratch.q,
        &cache.k_cache,
        &cache.v_cache,
        &scratch.attn_out,
        n_q,
        n_kv,
        head_dim,
        pos_base,
        b,
        max_seq,
    )?;

    // 9. Batched sigmoid(gate) on attention output (Qwen3Next-only).
    if has_q_gate {
        sigmoid_mul_inplace_batched_f32(backend, &scratch.attn_out, &scratch.gate_attn, q_dim, b)?;
    }

    // 10. W_O batched matmul → o, then xd += o (batched).
    dispatch_batched_attn_matmul(backend, &attn.w_o, b, &scratch.attn_out, &scratch.o)?;
    add_inplace_batched_f32(backend, xd_batched, &scratch.o, d, b)?;
    Ok(())
}

/// T162 phase 9c — Per-FFN-block batched scratch buffers (B_MAX-sized).
/// Used by `ffn_dense_forward_batch` for the dense FFN path (gate/up/down +
/// SwiGLU + residual). MoE FFN handled separately in phase 9f.
struct BatchScratchFfn {
    h_post: Buffer, // [B_MAX, d] — post-norm residual input
    gate: Buffer,   // [B_MAX, f]
    up: Buffer,     // [B_MAX, f]
    fd: Buffer,     // [B_MAX, f] — post-SwiGLU
    fc2: Buffer,    // [B_MAX, d] — post-down output
}

impl BatchScratchFfn {
    fn new(backend: &MetalBackend, cfg: &Qwen35Config) -> Self {
        let d = cfg.d;
        let f = cfg.f.max(1);
        let b = B_MAX_BATCH;
        let alloc = |bytes: usize| backend.alloc_shared(bytes.max(4)).unwrap();
        Self {
            h_post: alloc(b * d * 4),
            gate: alloc(b * f * 4),
            up: alloc(b * f * 4),
            fd: alloc(b * f * 4),
            fc2: alloc(b * d * 4),
        }
    }
}

/// T162 phase 9c — Batched dense FFN forward.
///
/// Equivalent to B sequential `ffn_dense_forward(FfnLayerMetal::Dense)` calls
/// but uses batched primitives (rms_norm_batched, dispatch_batched_attn_matmul,
/// swiglu_batched, add_inplace_batched).
///
/// Pré-conditions :
///   - `xd_batched` : `[B, d]` row-major (in/out — résidu cumulé sur place)
///   - `1 ≤ b ≤ B_MAX_BATCH`
///   - FFN doit être de la variante `Dense` (caller doit checker).
///
/// Status : non wiré — foundation pour phase 9d.
#[allow(dead_code, clippy::too_many_arguments)]
fn ffn_dense_forward_batch(
    backend: &MetalBackend,
    ffn_norm: &Buffer,
    w_gate: &HybridMetalWeight,
    w_up: &HybridMetalWeight,
    w_down: &HybridMetalWeight,
    xd_batched: &Buffer,
    scratch: &BatchScratchFfn,
    cfg: &Qwen35Config,
    b: usize,
) -> Result<(), MetalError> {
    if b == 0 || b > B_MAX_BATCH {
        return Err(MetalError::ShapeMismatch(format!(
            "ffn_dense_forward_batch: B must be in 1..={B_MAX_BATCH} (got {b})"
        )));
    }
    let d = cfg.d;
    let f = cfg.f;
    let eps = cfg.rms_eps;

    // 1. Batched FFN norm (xd → h_post).
    rms_norm_batched_f32(backend, xd_batched, ffn_norm, &scratch.h_post, d, b, eps)?;

    // 2. Batched gate + up matmul.
    dispatch_batched_attn_matmul(backend, w_gate, b, &scratch.h_post, &scratch.gate)?;
    dispatch_batched_attn_matmul(backend, w_up, b, &scratch.h_post, &scratch.up)?;

    // 3. Batched SwiGLU : fd[bi, i] = silu(gate[bi, i]) * up[bi, i].
    swiglu_batched_f32(backend, &scratch.gate, &scratch.up, &scratch.fd, f, b)?;

    // 4. Batched down + residual.
    dispatch_batched_attn_matmul(backend, w_down, b, &scratch.fd, &scratch.fc2)?;
    add_inplace_batched_f32(backend, xd_batched, &scratch.fc2, d, b)?;
    Ok(())
}

/// T162 phase 9e — Per-SSM-block batched scratch buffers (B_MAX-sized).
/// Sized to match the per-token Scratch fields but with B dimension prefixed.
/// `qkv_mixed`, `z`, `alpha`, `beta`, `gate_h`, `beta_sig` are populated by
/// the batched input-projection + apply_gate phase. The SSM scan (conv +
/// delta_net) reads these per-token (CPU memcpy slice into `state.scratch.*`).
/// `ssm_out_buf` collects the per-token scan outputs, then the batched ssm_out
/// projection writes `o`. Final residual add into `xd_batched` (caller).
struct BatchScratchSsm {
    h: Buffer,           // [B_MAX, d] post-ssm-norm
    qkv_mixed: Buffer,   // [B_MAX, conv_dim]
    z: Buffer,           // [B_MAX, value_dim]
    alpha: Buffer,       // [B_MAX, n_v]
    beta: Buffer,        // [B_MAX, n_v]
    gate_h: Buffer,      // [B_MAX, n_v]
    beta_sig: Buffer,    // [B_MAX, n_v]
    ssm_out_buf: Buffer, // [B_MAX, value_dim]
    o: Buffer,           // [B_MAX, d] post-ssm_out projection
    /// T200.3 — Batched conv1d output buffer for persistent scan path.
    /// Layout: [B_MAX, conv_dim]. Written per-step via conv1d_step with
    /// io_offsets, read by delta_net_persistent_scan_f32_into.
    conv_out_batched: Buffer,
}

impl BatchScratchSsm {
    fn new(backend: &MetalBackend, cfg: &Qwen35Config) -> Self {
        let d = cfg.d;
        let head_v_dim = cfg.ssm_state;
        let n_k = cfg.ssm_groups;
        let n_v = cfg.ssm_dt_rank;
        let key_dim = head_v_dim * n_k;
        let value_dim = head_v_dim * n_v;
        let conv_dim = 2 * key_dim + value_dim;
        let b = B_MAX_BATCH;
        let alloc = |bytes: usize| backend.alloc_shared(bytes.max(4)).unwrap();
        Self {
            h: alloc(b * d * 4),
            qkv_mixed: alloc(b * conv_dim * 4),
            z: alloc(b * value_dim * 4),
            alpha: alloc(b * n_v * 4),
            beta: alloc(b * n_v * 4),
            gate_h: alloc(b * n_v * 4),
            beta_sig: alloc(b * n_v * 4),
            ssm_out_buf: alloc(b * value_dim * 4),
            o: alloc(b * d * 4),
            conv_out_batched: alloc(b * conv_dim * 4),
        }
    }
}

/// T162 phase 9e — Batched SSM block forward.
///
/// Equivalent to B sequential `ssm_block_forward` calls but with batched
/// projections (input qkv/z/alpha/beta + apply_gate + output ssm_out). The
/// recurrent scan (conv1d_step + delta_net_step + per-head-norm) stays
/// sequential per-token because conv_state and ssm state are recurrent.
///
/// Pré-conditions :
///   - `xd_batched` : `[B, d]` row-major (in/out — résidu cumulé sur place)
///   - `state.scratch` : per-token Scratch utilisé pour la phase scan
///   - 1 ≤ b ≤ B_MAX_BATCH
#[allow(clippy::too_many_arguments)]
fn ssm_block_forward_batch(
    backend: &MetalBackend,
    ssm: &SsmLayerMetal,
    s: &SsmLayerState,
    xd_batched: &Buffer,
    batch_scratch: &BatchScratchSsm,
    scratch: &Scratch,
    cfg: &Qwen35Config,
    b: usize,
) -> Result<(), MetalError> {
    if b == 0 || b > B_MAX_BATCH {
        return Err(MetalError::ShapeMismatch(format!(
            "ssm_block_forward_batch: B must be in 1..={B_MAX_BATCH} (got {b})"
        )));
    }
    let d = cfg.d;
    let eps = cfg.rms_eps;
    let head_v_dim = cfg.ssm_state;
    let n_k = cfg.ssm_groups;
    let n_v = cfg.ssm_dt_rank;
    let key_dim = head_v_dim * n_k;
    let value_dim = head_v_dim * n_v;
    let conv_dim = 2 * key_dim + value_dim;

    // 1. Batched pre-mixer norm.
    let _t_norm = std::time::Instant::now();
    rms_norm_batched_f32(
        backend,
        xd_batched,
        &ssm.attn_norm,
        &batch_scratch.h,
        d,
        b,
        eps,
    )?;
    profile_drain_record(backend, "  fbs.norm", _t_norm);

    // 2. Batched 4 input projections (Q+K+V mixed, gate-z, alpha, beta).
    let _t_proj = std::time::Instant::now();
    dispatch_batched_attn_matmul(
        backend,
        &ssm.w_qkv,
        b,
        &batch_scratch.h,
        &batch_scratch.qkv_mixed,
    )?;
    dispatch_batched_attn_matmul(backend, &ssm.w_gate, b, &batch_scratch.h, &batch_scratch.z)?;
    dispatch_batched_attn_matmul(
        backend,
        &ssm.ssm_alpha,
        b,
        &batch_scratch.h,
        &batch_scratch.alpha,
    )?;
    dispatch_batched_attn_matmul(
        backend,
        &ssm.ssm_beta,
        b,
        &batch_scratch.h,
        &batch_scratch.beta,
    )?;
    profile_drain_record(backend, "  fbs.in_proj4", _t_proj);

    // 3. Batched ssm_apply_gate.
    let _t_gate = std::time::Instant::now();
    ssm_apply_gate_batched_f32(
        backend,
        &batch_scratch.alpha,
        &batch_scratch.beta,
        &ssm.dt_bias,
        &ssm.ssm_a,
        &batch_scratch.gate_h,
        &batch_scratch.beta_sig,
        n_v,
        b,
    )?;
    profile_drain_record(backend, "  fbs.apply_gate", _t_gate);

    // 4. Per-token scan (recurrent). T175 P0 — drain-free path : on bind les
    //    inputs `qkv_mixed` / `gate_h` / `beta_sig` / `z` directement à
    //    `batch_scratch.*` avec offset bi via les variantes `_with_offsets`,
    //    et on écrit l'output directement dans `batch_scratch.ssm_out_buf[bi]`.
    //    Élimine 4 memcpy CPU + 1 drain par token (scaling B × N_layers
    //    × 250 µs sur 35B-A3B).
    //
    //    Le scan reste séquentiel (s.state et s.conv_state sont read+write par
    //    iter), mais les hazards mémoire sont gérés automatiquement par Metal
    //    intra-CB ; pas besoin de drain pour les sérialiser.
    let conv_off_stride = conv_dim * 4;
    let value_off_stride = value_dim * 4;
    let n_v_off_stride = n_v * 4;
    let _t_scan = std::time::Instant::now();
    // T200.3 — Persistent scan path (opt-in via RUSTORCH_T200=1).
    // Replaces B per-timestep delta_net dispatches by 1 persistent_scan call.
    // Conv1d and gated_norm still per-step (their state/locality differ).
    // Net dispatches per layer: 2*B + 1 (vs 3*B before T200).
    let env_t200 = std::env::var("RUSTORCH_T200")
        .ok()
        .map(|v| v == "1")
        .unwrap_or(false);
    if env_t200 {
        // Phase A : per-step conv1d, write each output to batched buffer
        // [B, conv_dim]. Conv1d's recurrent state forces sequential dispatches.
        for bi in 0..b {
            ssm_conv1d_step_f32_with_io_offsets(
                backend,
                &batch_scratch.qkv_mixed,
                bi * conv_off_stride,
                &ssm.conv1d,
                &s.conv_state,
                &batch_scratch.conv_out_batched,
                bi * conv_off_stride,
                cfg.ssm_conv_kernel,
                conv_dim,
            )?;
        }
        // Phase B : single persistent_scan dispatch over all B timesteps.
        // State row stays in registers across timesteps; B-1 state DRAM
        // round-trips eliminated.
        delta_net_persistent_scan_f32_into(
            backend,
            &batch_scratch.conv_out_batched,
            &batch_scratch.gate_h,
            &batch_scratch.beta_sig,
            &s.state,
            &batch_scratch.ssm_out_buf,
            b,
            n_v,
            head_v_dim,
            n_k,
            conv_dim,
            eps,
        )?;
        // Phase C : T201 — batched gated RMS norm in 1 dispatch.
        // No recurrent state (per-step independent), so trivially parallel.
        // Replaces B per-step calls with 1 call per layer.
        rms_norm_per_head_gated_batched_f32(
            backend,
            &batch_scratch.ssm_out_buf,
            &ssm.ssm_norm,
            &batch_scratch.z,
            b,
            n_v,
            head_v_dim,
            eps,
        )?;
    } else {
        for bi in 0..b {
            // Conv1d step lit batch_scratch.qkv_mixed[bi*conv_dim..] directement.
            ssm_conv1d_step_f32_with_offset(
                backend,
                &batch_scratch.qkv_mixed,
                bi * conv_off_stride,
                &ssm.conv1d,
                &s.conv_state,
                &scratch.conv_out,
                cfg.ssm_conv_kernel,
                conv_dim,
            )?;
            // T199b — split_qkv_f32 ELIMINATED via Metal buffer offsets.
            delta_net_step_with_l2_f32_with_qkv_offsets(
                backend,
                &scratch.conv_out,
                0,
                &scratch.conv_out,
                key_dim * 4,
                &scratch.conv_out,
                2 * key_dim * 4,
                &batch_scratch.gate_h,
                bi * n_v_off_stride,
                &batch_scratch.beta_sig,
                bi * n_v_off_stride,
                &s.state,
                &batch_scratch.ssm_out_buf,
                bi * value_off_stride,
                n_v,
                head_v_dim,
                n_k,
                eps,
            )?;
        }
        // T201 — batched gated RMS norm AFTER the scan loop (no recurrent state).
        rms_norm_per_head_gated_batched_f32(
            backend,
            &batch_scratch.ssm_out_buf,
            &ssm.ssm_norm,
            &batch_scratch.z,
            b,
            n_v,
            head_v_dim,
            eps,
        )?;
    }
    profile_drain_record(backend, "  fbs.scan_loop", _t_scan);

    // 5. Batched output projection ssm_out_buf → o.
    let _t_out = std::time::Instant::now();
    dispatch_batched_attn_matmul(
        backend,
        &ssm.ssm_out,
        b,
        &batch_scratch.ssm_out_buf,
        &batch_scratch.o,
    )?;
    // 6. Batched residual add.
    add_inplace_batched_f32(backend, xd_batched, &batch_scratch.o, d, b)?;
    profile_drain_record(backend, "  fbs.out_residual", _t_out);
    Ok(())
}

/// T162 phase 9f — Per-MoE-block batched scratch buffers (B_MAX-sized).
/// Sized for the worst case 35B-A3B (B_MAX × n_used × max(d, ef) floats).
/// Allocates ~30MB total (acceptable, single allocation amortized across layers).
struct BatchScratchMoe {
    h_post: Buffer,       // [B_MAX, d] post-attn-norm input
    logits: Buffer,       // [B_MAX, n_experts]
    indices: Buffer,      // [B_MAX, n_used] u32
    topw: Buffer,         // [B_MAX, n_used] f32
    h_repl: Buffer,       // [B_MAX * n_used, d] — h replicated for gather x_stride=K
    gate_gather: Buffer,  // [B_MAX * n_used, ef]
    up_gather: Buffer,    // [B_MAX * n_used, ef]
    fd_gather: Buffer,    // [B_MAX * n_used, ef]
    down_gather: Buffer,  // [B_MAX * n_used, d]
    moe_acc: Buffer,      // [B_MAX, d] weighted sum of routed experts
    shexp_gate: Buffer,   // [B_MAX, ef] shared expert gate proj
    shexp_up: Buffer,     // [B_MAX, ef]
    shexp_fd: Buffer,     // [B_MAX, ef] post-SwiGLU
    shexp_out: Buffer,    // [B_MAX, d] shared expert down proj
    shexp_scalar: Buffer, // [B_MAX] gate_inp_shexp · h scalar gate

    // T163 phase 9f-quater — expert-major MoE pipeline buffers.
    // M_padded worst case = B_MAX*n_used + 7*n_experts (each expert pads ≤ 7 rows).
    em_src_indices: Buffer, // [M_padded] u32 = src token (b*n_used+k_slot) or sentinel
    em_gather_src: Buffer,  // [M_padded] u32 = b for gather (or sentinel)
    em_tile_expert_ids: Buffer, // [M_padded/8] u32 = expert per M-tile
    em_x_packed: Buffer,    // [M_padded, d] gathered+padded x for expert_major
    em_out_ef_gate: Buffer, // [M_padded, ef] expert_major output (gate)
    em_out_ef_up: Buffer,   // [M_padded, ef] expert_major output (up)
    em_fd: Buffer,          // [M_padded, ef] post-swiglu
    em_out_d: Buffer,       // [M_padded, d] expert_major down output
    em_m_padded: Buffer,    // T175 P0' — [1] u32, m_padded computed by GPU sort kernel

    // T197 — mul_mm_id pipeline buffers (alternative to expert-major above).
    // Tile M=32 (vs em 8) + indirected gather inside kernel = no x_packed staging.
    // M_max = B_MAX_BATCH (max tokens routed to one expert ≤ B since top-K returns
    // distinct experts per token).
    mmid_tpe: Buffer,      // [E] u32 — tokens per expert (output of map0)
    mmid_ids: Buffer,      // [E * M_max] u32 — sorted (b*n_used+slot) per expert
    mmid_pos: Buffer,      // [B * n_used] u32 — inverse permutation
    mmid_gate_out: Buffer, // [E, M_max, ef] f32 per-expert gate output
    mmid_up_out: Buffer,   // [E, M_max, ef] f32 per-expert up output
    mmid_silu: Buffer,     // [E, M_max, ef] f32 post-SwiGLU (or alias gate_out)
    mmid_down_out: Buffer, // [E, M_max, d] f32 per-expert down output
}

impl BatchScratchMoe {
    fn new(backend: &MetalBackend, cfg: &Qwen35Config) -> Self {
        let d = cfg.d;
        let ef = cfg.expert_f.max(1);
        let n_experts = cfg.n_experts.max(1);
        let n_used = cfg.n_experts_used.max(1);
        let b = B_MAX_BATCH;
        let alloc = |bytes: usize| backend.alloc_shared(bytes.max(4)).unwrap();
        Self {
            h_post: alloc(b * d * 4),
            logits: alloc(b * n_experts * 4),
            indices: alloc(b * n_used * 4),
            topw: alloc(b * n_used * 4),
            h_repl: alloc(b * n_used * d * 4),
            gate_gather: alloc(b * n_used * ef * 4),
            up_gather: alloc(b * n_used * ef * 4),
            fd_gather: alloc(b * n_used * ef * 4),
            down_gather: alloc(b * n_used * d * 4),
            moe_acc: alloc(b * d * 4),
            shexp_gate: alloc(b * ef * 4),
            shexp_up: alloc(b * ef * 4),
            shexp_fd: alloc(b * ef * 4),
            shexp_out: alloc(b * d * 4),
            shexp_scalar: alloc(b * 4),

            // M_padded worst case for expert_major pipeline.
            em_src_indices: alloc((b * n_used + 7 * n_experts) * 4),
            em_gather_src: alloc((b * n_used + 7 * n_experts) * 4),
            em_tile_expert_ids: alloc(((b * n_used + 7 * n_experts) / 8 + 1) * 4),
            em_x_packed: alloc((b * n_used + 7 * n_experts) * d * 4),
            em_out_ef_gate: alloc((b * n_used + 7 * n_experts) * ef * 4),
            em_out_ef_up: alloc((b * n_used + 7 * n_experts) * ef * 4),
            em_fd: alloc((b * n_used + 7 * n_experts) * ef * 4),
            em_out_d: alloc((b * n_used + 7 * n_experts) * d * 4),
            em_m_padded: alloc(4),

            // T197 — mul_mm_id buffers. M_max = B = B_MAX_BATCH (safe upper bound
            // documented in mul_mm_id_map0_into: top-K with distinct experts per
            // token guarantees tpe[e] ≤ B for all e).
            mmid_tpe: alloc(n_experts * 4),
            mmid_ids: alloc(n_experts * b * 4),
            mmid_pos: alloc(b * n_used * 4),
            mmid_gate_out: alloc(n_experts * b * ef * 4),
            mmid_up_out: alloc(n_experts * b * ef * 4),
            mmid_silu: alloc(n_experts * b * ef * 4),
            mmid_down_out: alloc(n_experts * b * d * 4),
        }
    }
}

/// T163 phase 9f-quater — sort tokens par expert + pad mult-8.
fn build_expert_major_perm(
    indices_cpu: &[u32],
    n_used: usize,
    n_experts: usize,
    src_indices_out: &mut Vec<u32>,
    gather_src_out: &mut Vec<u32>,
    tile_expert_ids_out: &mut Vec<u32>,
) -> usize {
    src_indices_out.clear();
    gather_src_out.clear();
    tile_expert_ids_out.clear();
    const SENTINEL: u32 = 0xFFFFFFFFu32;

    let mut buckets: Vec<Vec<u32>> = (0..n_experts).map(|_| Vec::new()).collect();
    for (i, &expert) in indices_cpu.iter().enumerate() {
        let e = expert as usize;
        if e < n_experts {
            buckets[e].push(i as u32);
        }
    }
    for (e, bucket) in buckets.iter().enumerate() {
        if bucket.is_empty() {
            continue;
        }
        for &idx in bucket {
            src_indices_out.push(idx);
            gather_src_out.push(idx / n_used as u32);
        }
        let padded = bucket.len().div_ceil(8) * 8;
        for _ in bucket.len()..padded {
            src_indices_out.push(SENTINEL);
            gather_src_out.push(SENTINEL);
        }
        for _ in 0..(padded / 8) {
            tile_expert_ids_out.push(e as u32);
        }
    }
    src_indices_out.len()
}

/// T162 phase 9f — gather dispatch helper (Q4_K/Q5_K/Q6_K stacked experts).
#[allow(clippy::too_many_arguments)]
fn dispatch_gather_sgemv(
    backend: &MetalBackend,
    stacked: &StackedQuantizedExperts,
    x: &Buffer,
    out: &Buffer,
    indices: &Buffer,
    b_eff: usize, // B * n_used (total expert evaluations)
    k: usize,
    n: usize,
    x_stride: usize,
) -> Result<(), MetalError> {
    match stacked.dtype {
        GgmlType::Q4_K => sgemv_q4_k_gather_f32_lcpp_nsg2_into(
            backend,
            x,
            &stacked.buffer,
            indices,
            b_eff,
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
            indices,
            b_eff,
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
            indices,
            b_eff,
            out,
            k,
            n,
            stacked.bytes_per_expert,
            x_stride,
        ),
        other => Err(MetalError::Unsupported(format!(
            "MoE gather: dtype {other:?} not supported"
        ))),
    }
}

/// T162 phase 9f — Batched MoE FFN forward.
///
/// Equivalent à B sequential `ffn_dense_forward(FfnLayerMetal::Moe { .. })`.
/// Approche : routing batched + replicate h pour gather B*n_used → 1 dispatch
/// pour gate/up/down (vs B per-token loops). Top-K, weighted_reduce, et
/// shared-gate scalar restent per-token (loops simples, GPU 1-dispatch each).
///
/// La projection shared-expert (gate/up/down + swiglu) est BATCHÉE via les
/// dispatch_batched_attn_matmul (déjà SGEMM tile 64×64 si M ≥ 64).
///
/// Status : foundation phase 9f-a. La per-token boucle pour topk / reduce /
/// sigmoid_add peut être remplacée par des kernels batched dans 9f-b.
#[allow(clippy::too_many_arguments)]
fn ffn_moe_forward_batch(
    backend: &MetalBackend,
    ffn_norm: &Buffer,
    gate_inp: &HybridMetalWeight,
    gate_inp_shexp: &Buffer,
    gate_shexp: &HybridMetalWeight,
    up_shexp: &HybridMetalWeight,
    down_shexp: &HybridMetalWeight,
    gate_exps_stacked: &StackedQuantizedExperts,
    up_exps_stacked: &StackedQuantizedExperts,
    down_exps_stacked: &StackedQuantizedExperts,
    xd_batched: &Buffer,
    scratch: &Scratch,
    batch_scratch: &BatchScratchMoe,
    cfg: &Qwen35Config,
    b: usize,
) -> Result<(), MetalError> {
    if b == 0 || b > B_MAX_BATCH {
        return Err(MetalError::ShapeMismatch(format!(
            "ffn_moe_forward_batch: B must be in 1..={B_MAX_BATCH} (got {b})"
        )));
    }
    let d = cfg.d;
    let ef = cfg.expert_f;
    let n_experts = cfg.n_experts;
    let n_used = cfg.n_experts_used;
    let eps = cfg.rms_eps;
    let b_eff = b * n_used; // total expert evaluations across all tokens

    // 1. Batched post-attn norm (xd → h_post, gamma=ffn_norm).
    let _t_rmsn = std::time::Instant::now();
    rms_norm_batched_f32(
        backend,
        xd_batched,
        ffn_norm,
        &batch_scratch.h_post,
        d,
        b,
        eps,
    )?;
    profile_drain_record(backend, "  fbm.pre_norm", _t_rmsn);

    // 2. Batched routing logits : gate_inp_batched @ h_post → logits [B, n_experts].
    let _t_logits = std::time::Instant::now();
    dispatch_batched_attn_matmul(
        backend,
        gate_inp,
        b,
        &batch_scratch.h_post,
        &batch_scratch.logits,
    )?;
    profile_drain_record(backend, "  fbm.logits", _t_logits);
    // T175 P1 — drain supprimé : `topk_softmax_norm_batched_f32` lit
    // `batch_scratch.logits` qui vient d'être écrit par le dispatch précédent.
    // Metal sérialise via memory hazards intra-CB ; le drain de 250 µs était
    // pur gaspillage. (Coût total éliminé : 30 layers × 250 µs = 7.5 ms/chunk.)

    // 3. Batched top-K + softmax + renormalize : 1 dispatch (B threadgroups)
    //    au lieu de B per-token loops + drains.
    let _t_topk = std::time::Instant::now();
    topk_softmax_norm_batched_f32(
        backend,
        &batch_scratch.logits,
        &batch_scratch.indices,
        &batch_scratch.topw,
        n_experts,
        n_used,
        b,
    )?;
    profile_drain_record(backend, "  fbm.topk", _t_topk);

    // 4-7. T163 phase 9f-quater + 9f-cinq : EXPERT-MAJOR pipeline étendu Q4_K + Q5_K.
    //      Dispatch SGEMM expert_major selon dtype par projection (gate/up/down).
    //
    // **DÉCOUVERTE CRITIQUE T163 phase 9f-six** : la stratégie expert_major avec
    // padding mult-8 par expert SE DÉGRADE QUAND n_experts >> B*n_used. Pour
    // 35B-A3B (n_experts=256, B*n_used=256 avec B=32), chaque expert reçoit en
    // moyenne 1 eval → padded à 8 rows = 87% de padding wasted. Le SGEMM fait
    // alors ~8× plus de compute que nécessaire, devenant plus lent que per-token.
    //
    // GUARD : expert_major rentable seulement quand B*n_used >> n_experts (i.e.,
    // experts répétés en moyenne). Heuristique : `b_eff >= n_experts`.
    let gate_em = matches!(gate_exps_stacked.dtype, GgmlType::Q4_K | GgmlType::Q5_K);
    let up_em = matches!(up_exps_stacked.dtype, GgmlType::Q4_K | GgmlType::Q5_K);
    let down_em = matches!(down_exps_stacked.dtype, GgmlType::Q4_K | GgmlType::Q5_K);
    // T169 (Voie D) — guard threshold relaxable via env. Default keeps the
    // T163 phase 9f-six behavior (b_eff >= n_experts). Setting RUSTORCH_EM_THRESHOLD
    // to a smaller value (e.g. 8) enables expert_major at smaller batches to
    // measure whether the BlockMMA gain offsets the padding waste at B<n_experts.
    let em_threshold: usize = std::env::var("RUSTORCH_EM_THRESHOLD")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(n_experts);
    let all_em = gate_em && up_em && down_em && b_eff >= em_threshold;

    let dispatch_em_sgemm = |stacked: &StackedQuantizedExperts,
                             a_buf: &Buffer,
                             c_buf: &Buffer,
                             m: usize,
                             n: usize,
                             k: usize|
     -> Result<(), MetalError> {
        match stacked.dtype {
            GgmlType::Q4_K => {
                // T175 Day 5 wire-up : préfère le tile 8×64 half MMA quand
                // N est aligné sur 64 (1.25× speedup vs 8×8 sur 35B-A3B).
                // Fallback 8×8 sinon (e.g. shared expert avec N petit).
                if n % 64 == 0 {
                    sgemm_q4_k_f32_expert_major_8x64_half_into(
                        backend,
                        a_buf,
                        &stacked.buffer,
                        &batch_scratch.em_tile_expert_ids,
                        c_buf,
                        m,
                        n,
                        k,
                        stacked.bytes_per_expert,
                    )
                } else {
                    sgemm_q4_k_f32_expert_major_8x8_into(
                        backend,
                        a_buf,
                        &stacked.buffer,
                        &batch_scratch.em_tile_expert_ids,
                        c_buf,
                        m,
                        n,
                        k,
                        stacked.bytes_per_expert,
                    )
                }
            },
            GgmlType::Q5_K => {
                // T175 — Same 8×64 wire-up as Q4_K. Q5_K down_proj on 35B-A3B
                // is ~33% of MoE compute → 1.25× speedup expected.
                if n % 64 == 0 {
                    sgemm_q5_k_f32_expert_major_8x64_half_into(
                        backend,
                        a_buf,
                        &stacked.buffer,
                        &batch_scratch.em_tile_expert_ids,
                        c_buf,
                        m,
                        n,
                        k,
                        stacked.bytes_per_expert,
                    )
                } else {
                    sgemm_q5_k_f32_expert_major_8x8_into(
                        backend,
                        a_buf,
                        &stacked.buffer,
                        &batch_scratch.em_tile_expert_ids,
                        c_buf,
                        m,
                        n,
                        k,
                        stacked.bytes_per_expert,
                    )
                }
            },
            other => Err(MetalError::Unsupported(format!(
                "expert_major SGEMM: dtype {other:?} not yet ported"
            ))),
        }
    };

    // T197.1b — mm_id pipeline DEFAULT ON when shapes match, opt-out via RUSTORCH_MMID=0.
    // Replaces the [drain + CPU sort + upload + gather_pack + em_*×3 + unpermute
    // + reduce] chain by [GPU map0 + mm_id_q4_k×2 + swiglu + mm_id_q5_k + scatter].
    // Eliminates: 1 CPU drain, 1 CPU sort+upload, 1 zero_f32, 1 gather_pack_rows,
    // 1 unpermute_rows, 1 weighted_reduce_add. Keeps: 6 GPU dispatches total.
    //
    // Measured (M4 Max, 35B-A3B Q4_K_M, B=128, 5 runs each):
    //   EM    : mean 108 t/s, range 82-131 (std 21)
    //   mm_id : mean 131 t/s, range 126-140 (std 5)  → +21% mean, ×4 stability
    let env_mmid = std::env::var("RUSTORCH_MMID")
        .ok()
        .map(|v| v != "0")
        .unwrap_or(true);
    let mmid_supported = env_mmid
        && all_em
        && matches!(gate_exps_stacked.dtype, GgmlType::Q4_K)
        && matches!(up_exps_stacked.dtype, GgmlType::Q4_K)
        && matches!(down_exps_stacked.dtype, GgmlType::Q5_K)
        && b % 32 == 0; // mm_id requires M_max % 32 == 0

    if mmid_supported {
        // T197.1c — m_max sized to expected max(tpe) with safety margin.
        //
        // Theoretical worst case (every token same expert): m_max = b.
        // Practical case with softmax routing on n_experts=256, n_used=8:
        //   mean(tpe) = b * n_used / n_experts ≈ 4 for B=128
        //   max(tpe) typically ≤ 10-15 with softmax routing
        //
        // m_max=b/4 (=32 for B=128) gives ~3-8× safety margin on max(tpe)
        // and reduces dispatched TGs by 4× compared to m_max=b. Measured
        // (M4 Max, 35B-A3B, B=128, 5 runs):
        //   m_max=128 : 131 t/s steady (mean)
        //   m_max=64  : 341 t/s (×2.6)
        //   m_max=32  : 352 t/s (×2.7)
        //
        // RUSTORCH_MMID_MMAX env override for tuning. Use larger value
        // (up to b) if observe wrong outputs (silent token drop on overflow).
        let m_max_env = std::env::var("RUSTORCH_MMID_MMAX")
            .ok()
            .and_then(|s| s.parse::<usize>().ok());
        let m_max = match m_max_env {
            Some(v) if v > 0 && v <= b && v % 32 == 0 => v,
            _ => {
                // Default: b/4 aligned to 32, with a floor of 32.
                ((b / 4).max(32) / 32) * 32
            },
        };

        // Stage 1 : GPU sort routing — no CPU drain, no upload.
        let _t_map0 = std::time::Instant::now();
        mul_mm_id_map0_into(
            backend,
            &batch_scratch.indices,
            &batch_scratch.mmid_tpe,
            &batch_scratch.mmid_ids,
            &batch_scratch.mmid_pos,
            b,
            n_used,
            n_experts,
            m_max,
        )?;
        profile_drain_record(backend, "  fbm.mmid_map0", _t_map0);

        // T198 — Fused gate + up + swiglu in single kernel (default ON, opt-out RUSTORCH_T198=0).
        // Eliminates intermediate gate_out/up_out DRAM round-trips.
        // Measured (35B-A3B B=128): 345 → 357 t/s prefill (+3.5%), parity bit-exact.
        let env_t198 = std::env::var("RUSTORCH_T198")
            .ok()
            .map(|v| v != "0")
            .unwrap_or(true);

        if env_t198 {
            let _t_fused = std::time::Instant::now();
            mul_mm_id_q4_k_q4_k_swiglu_f32_into(
                backend,
                &batch_scratch.h_post,
                &gate_exps_stacked.buffer,
                &up_exps_stacked.buffer,
                &batch_scratch.mmid_ids,
                &batch_scratch.mmid_tpe,
                &batch_scratch.mmid_silu,
                n_experts,
                m_max,
                ef,
                d,
                n_used,
            )?;
            profile_drain_record(backend, "  fbm.mmid_fused_gateupswiglu", _t_fused);
        } else {
            // Stage 2a : Q4_K mm_id gate.
            let _t_gate = std::time::Instant::now();
            mul_mm_id_q4_k_f32_into(
                backend,
                &batch_scratch.h_post,
                &gate_exps_stacked.buffer,
                &batch_scratch.mmid_ids,
                &batch_scratch.mmid_tpe,
                &batch_scratch.mmid_gate_out,
                n_experts,
                m_max,
                ef,
                d,
                n_used,
            )?;
            profile_drain_record(backend, "  fbm.mmid_gate", _t_gate);

            // Stage 2b : Q4_K mm_id up.
            let _t_up = std::time::Instant::now();
            mul_mm_id_q4_k_f32_into(
                backend,
                &batch_scratch.h_post,
                &up_exps_stacked.buffer,
                &batch_scratch.mmid_ids,
                &batch_scratch.mmid_tpe,
                &batch_scratch.mmid_up_out,
                n_experts,
                m_max,
                ef,
                d,
                n_used,
            )?;
            profile_drain_record(backend, "  fbm.mmid_up", _t_up);

            // Stage 3 : SwiGLU on flat [E*M_max*ef] buffer.
            let _t_silu = std::time::Instant::now();
            swiglu_f32(
                backend,
                &batch_scratch.mmid_gate_out,
                &batch_scratch.mmid_up_out,
                &batch_scratch.mmid_silu,
                n_experts * m_max * ef,
            )?;
            profile_drain_record(backend, "  fbm.mmid_swiglu", _t_silu);
        }

        // Stage 4 : Q5_K mm_id down.
        // Note : N=d, K=ef (the reverse of gate/up which had N=ef, K=d).
        let _t_down = std::time::Instant::now();
        mul_mm_id_q5_k_f32_into(
            backend,
            &batch_scratch.mmid_silu,
            &down_exps_stacked.buffer,
            &batch_scratch.mmid_ids,
            &batch_scratch.mmid_tpe,
            &batch_scratch.mmid_down_out,
            n_experts,
            m_max,
            d,
            ef,
            n_used,
        )?;
        profile_drain_record(backend, "  fbm.mmid_down", _t_down);

        // Stage 5 : weighted scatter back to moe_acc[B, d]. OVERWRITES (no zero needed).
        let _t_scatter = std::time::Instant::now();
        scatter_moe_acc_f32_into(
            backend,
            &batch_scratch.mmid_down_out,
            &batch_scratch.indices,
            &batch_scratch.topw,
            &batch_scratch.mmid_pos,
            &batch_scratch.moe_acc,
            b,
            n_used,
            m_max,
            d,
        )?;
        profile_drain_record(backend, "  fbm.mmid_scatter", _t_scatter);
    } else if all_em {
        // CPU sort indices.
        let _t_drain = std::time::Instant::now();
        backend.drain();
        profile_drain_record(backend, "  fbm.cpu_drain", _t_drain);
        let _t_sort = std::time::Instant::now();
        let mut src_indices_vec: Vec<u32> = Vec::with_capacity(b_eff + 7 * n_experts);
        let mut gather_src_vec: Vec<u32> = Vec::with_capacity(b_eff + 7 * n_experts);
        let mut tile_expert_ids_vec: Vec<u32> = Vec::with_capacity(b_eff / 8 + n_experts);
        let indices_cpu = unsafe {
            std::slice::from_raw_parts(batch_scratch.indices.contents() as *const u32, b * n_used)
        };
        let m_padded = build_expert_major_perm(
            indices_cpu,
            n_used,
            n_experts,
            &mut src_indices_vec,
            &mut gather_src_vec,
            &mut tile_expert_ids_vec,
        );

        // Upload to GPU.
        unsafe {
            std::ptr::copy_nonoverlapping(
                src_indices_vec.as_ptr(),
                batch_scratch.em_src_indices.contents() as *mut u32,
                m_padded,
            );
            std::ptr::copy_nonoverlapping(
                gather_src_vec.as_ptr(),
                batch_scratch.em_gather_src.contents() as *mut u32,
                m_padded,
            );
            std::ptr::copy_nonoverlapping(
                tile_expert_ids_vec.as_ptr(),
                batch_scratch.em_tile_expert_ids.contents() as *mut u32,
                tile_expert_ids_vec.len(),
            );
        }
        profile_drain_record(backend, "  fbm.cpu_sort", _t_sort);

        // GPU pipeline expert-major.
        let _t_zero = std::time::Instant::now();
        zero_f32(backend, &batch_scratch.moe_acc, b * d)?;
        profile_drain_record(backend, "  fbm.zero", _t_zero);
        let _t_pack = std::time::Instant::now();
        gather_pack_rows_f32(
            backend,
            &batch_scratch.h_post,
            &batch_scratch.em_gather_src,
            &batch_scratch.em_x_packed,
            d,
            m_padded,
        )?;
        profile_drain_record(backend, "  fbm.pack", _t_pack);
        let _t_gate = std::time::Instant::now();
        dispatch_em_sgemm(
            gate_exps_stacked,
            &batch_scratch.em_x_packed,
            &batch_scratch.em_out_ef_gate,
            m_padded,
            ef,
            d,
        )?;
        profile_drain_record(backend, "  fbm.em_gate", _t_gate);
        let _t_up = std::time::Instant::now();
        dispatch_em_sgemm(
            up_exps_stacked,
            &batch_scratch.em_x_packed,
            &batch_scratch.em_out_ef_up,
            m_padded,
            ef,
            d,
        )?;
        profile_drain_record(backend, "  fbm.em_up", _t_up);
        let _t_silu = std::time::Instant::now();
        swiglu_f32(
            backend,
            &batch_scratch.em_out_ef_gate,
            &batch_scratch.em_out_ef_up,
            &batch_scratch.em_fd,
            m_padded * ef,
        )?;
        profile_drain_record(backend, "  fbm.swiglu", _t_silu);
        let _t_down = std::time::Instant::now();
        dispatch_em_sgemm(
            down_exps_stacked,
            &batch_scratch.em_fd,
            &batch_scratch.em_out_d,
            m_padded,
            d,
            ef,
        )?;
        profile_drain_record(backend, "  fbm.em_down", _t_down);
        let _t_unp = std::time::Instant::now();
        unpermute_rows_f32(
            backend,
            &batch_scratch.em_out_d,
            &batch_scratch.em_src_indices,
            &batch_scratch.down_gather,
            d,
            m_padded,
        )?;
        profile_drain_record(backend, "  fbm.unpermute", _t_unp);
        let _t_red = std::time::Instant::now();
        weighted_reduce_add_batched_f32(
            backend,
            &batch_scratch.down_gather,
            &batch_scratch.topw,
            &batch_scratch.moe_acc,
            n_used,
            d,
            b,
        )?;
        profile_drain_record(backend, "  fbm.reduce", _t_red);
    } else {
        // Fallback : gather_per_token (T162 phase 9f-bis) pour les modèles
        // mixed-dtype (e.g., 35B-A3B Q4_K gate/up + Q5_K down).
        zero_f32(backend, &batch_scratch.moe_acc, b * d)?;
        sgemv_q4_k_gather_per_token_f32_lcpp_nsg2_into(
            backend,
            &batch_scratch.h_post,
            &gate_exps_stacked.buffer,
            &batch_scratch.indices,
            b_eff,
            n_used,
            &batch_scratch.gate_gather,
            d,
            ef,
            gate_exps_stacked.bytes_per_expert,
        )?;
        sgemv_q4_k_gather_per_token_f32_lcpp_nsg2_into(
            backend,
            &batch_scratch.h_post,
            &up_exps_stacked.buffer,
            &batch_scratch.indices,
            b_eff,
            n_used,
            &batch_scratch.up_gather,
            d,
            ef,
            up_exps_stacked.bytes_per_expert,
        )?;
        swiglu_f32(
            backend,
            &batch_scratch.gate_gather,
            &batch_scratch.up_gather,
            &batch_scratch.fd_gather,
            b_eff * ef,
        )?;
        dispatch_gather_sgemv(
            backend,
            down_exps_stacked,
            &batch_scratch.fd_gather,
            &batch_scratch.down_gather,
            &batch_scratch.indices,
            b_eff,
            ef,
            d,
            ef,
        )?;
        weighted_reduce_add_batched_f32(
            backend,
            &batch_scratch.down_gather,
            &batch_scratch.topw,
            &batch_scratch.moe_acc,
            n_used,
            d,
            b,
        )?;
    }

    // 9. Shared expert pipeline (BATCHED dense FFN).
    let _t_sh_gate = std::time::Instant::now();
    dispatch_batched_attn_matmul(
        backend,
        gate_shexp,
        b,
        &batch_scratch.h_post,
        &batch_scratch.shexp_gate,
    )?;
    profile_drain_record(backend, "  fbm.sh_gate", _t_sh_gate);
    let _t_sh_up = std::time::Instant::now();
    dispatch_batched_attn_matmul(
        backend,
        up_shexp,
        b,
        &batch_scratch.h_post,
        &batch_scratch.shexp_up,
    )?;
    profile_drain_record(backend, "  fbm.sh_up", _t_sh_up);
    swiglu_batched_f32(
        backend,
        &batch_scratch.shexp_gate,
        &batch_scratch.shexp_up,
        &batch_scratch.shexp_fd,
        ef,
        b,
    )?;
    let _t_sh_down = std::time::Instant::now();
    dispatch_batched_attn_matmul(
        backend,
        down_shexp,
        b,
        &batch_scratch.shexp_fd,
        &batch_scratch.shexp_out,
    )?;
    profile_drain_record(backend, "  fbm.sh_down", _t_sh_down);

    // 10. Batched fused gate-scalar + sigmoid_add_moe (final residual).
    let _t_sigmadd = std::time::Instant::now();
    sigmoid_add_moe_batched_f32(
        backend,
        &batch_scratch.moe_acc,
        &batch_scratch.shexp_out,
        gate_inp_shexp,
        &batch_scratch.h_post,
        xd_batched,
        d,
        b,
    )?;
    profile_drain_record(backend, "  fbm.sigmadd", _t_sigmadd);

    // Silence unused warning on scratch (kept in signature for future direct use).
    let _ = scratch;
    Ok(())
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
    //
    // T186 — `gqa_decode_f32_nsg2` (NSG=2) halves per-thread sequential work
    //   over kv_len. ×1.13 to ×2.21 microbench at short→long kv.
    // T190 v2 — `gqa_decode_f32_splitk` (FlashDecoding-style) splits kv_len
    //   into chunks dispatched as N TGs per Q head, then 1 TG reduces.
    //   Auto-falls back to NSG=2 for kv ≤ 1024. Wins at long kv:
    //     kv=2048  : 611 → 336 µs  (×1.85)
    //     kv=4096  : 1393 → 358 µs (×3.89 !)
    //   Parity max_rel_err 8.4e-5 (numerical noise from reduction order).
    // T194 — `gqa_decode_f32_splitk_nsg2` widens Phase 1 to 64 threads/TG (NSG=2)
    //   instead of 32. Doubles SM occupancy at long kv:
    //     kv=2048  : 333 → 171 µs  (×1.95 vs T190v2)
    //     kv=4096  : 345 → 194 µs  (×1.78 vs T190v2)
    //   Parity max_rel_err 2.4e-7 (essentially identical).
    // T195 — `gqa_decode_f32_splitk_nsg4` widens further to 128 threads/TG (4 SGs).
    //   ~×1.15-1.21 vs T194 NSG=2 at long kv. Same parity (rel_err 2.3e-7).
    // Activable via RUSTORCH_GQA_SPLITK_NSG4 (default ON), RUSTORCH_GQA_SPLITK_NSG2 (=0 to bisect to NSG=2).
    let use_gqa_splitk_nsg4 = std::env::var("RUSTORCH_GQA_SPLITK_NSG4")
        .map(|v| v != "0")
        .unwrap_or(true);
    let use_gqa_splitk_nsg2 = std::env::var("RUSTORCH_GQA_SPLITK_NSG2")
        .map(|v| v != "0")
        .unwrap_or(true);
    let use_gqa_splitk = std::env::var("RUSTORCH_GQA_SPLITK")
        .map(|v| v != "0")
        .unwrap_or(true);
    let use_gqa_nsg2 = std::env::var("RUSTORCH_GQA_NSG2")
        .map(|v| v != "0")
        .unwrap_or(true);
    if use_gqa_splitk_nsg4 {
        gqa_decode_f32_splitk_nsg4(
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
    } else if use_gqa_splitk_nsg2 {
        gqa_decode_f32_splitk_nsg2(
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
    } else if use_gqa_splitk {
        gqa_decode_f32_splitk(
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
    } else if use_gqa_nsg2 {
        gqa_decode_f32_nsg2(
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
    } else {
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
    }

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
    let _ssm_t0 = std::time::Instant::now();
    rms_norm_f32(backend, &scratch.xd, &ssm.attn_norm, &scratch.h, d, eps)?;
    profile_drain_record(backend, "  ssm.in_norm", _ssm_t0);
    let _ssm_t0_proj = std::time::Instant::now();
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
    profile_drain_record(backend, "  ssm.in_proj", _ssm_t0_proj);

    // 3. T146a — fused GPU kernel: gate_h = softplus(alpha + dt_bias) * ssm_a,
    //    beta_sig = sigmoid(beta). No drain needed.
    //
    // T178 (rejected) — tried to fuse this gate_apply + delta_net + gated_norm
    // into a single mega-kernel `ssm_block_mega_f32`. Result: −5% regression
    // because per-head TG (32 TGs) underutilizes M4 Max SMs vs original per-row
    // TG (4096 TGs). 5th occurrence of the chained-encoder fusion trap (cf. T86,
    // T164, T165, T177). Mega-kernel kept dead-code in kernels.rs for reference.
    let _ssm_t0_gate = std::time::Instant::now();
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
    profile_drain_record(backend, "  ssm.gate_apply", _ssm_t0_gate);

    // 4. Conv1d step + ring-buffer update.
    let _ssm_t0_conv = std::time::Instant::now();
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
    profile_drain_record(backend, "  ssm.conv1d", _ssm_t0_conv);
    let _ssm_t0_split = std::time::Instant::now();
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
    profile_drain_record(backend, "  ssm.split_qkv", _ssm_t0_split);
    let _ssm_t0_dnet = std::time::Instant::now();
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
    // Switch RUSTORCH_SSM_FORCE_LEGACY_L2=1 pour bisection / régression check.
    //
    // T178 mega-kernel (per-head TG, 32 TGs) rejected -5% (kept dead-code).
    // T179 V2 MLX-style dispatch (1024 TGs × 128 threads, state in registers)
    //   wash -1% (kept dead-code + parity test in kernels.rs).
    // 3 SSM optimization attempts → SSM is NOT the bottleneck.
    // Real hot path = MoE (50%) + LM head sgemv (25%).
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
    profile_drain_record(backend, "  ssm.delta_net", _ssm_t0_dnet);

    // 10. Per-head RMSNorm gated by silu(z).
    let _ssm_t0_gnorm = std::time::Instant::now();
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

    profile_drain_record(backend, "  ssm.gated_norm", _ssm_t0_gnorm);

    // 11. ssm_out @ out_gated → result, then xd += result.
    let _ssm_t0_out = std::time::Instant::now();
    ssm.ssm_out
        .matmul_into(backend, &scratch.ssm_out_buf, &scratch.o)?;
    dump_buf(backend, &scratch.o, d, &format!("{dump_prefix}/ssm/16_o"));
    add_inplace_f32(backend, &scratch.xd, &scratch.o, d)?;
    profile_drain_record(backend, "  ssm.out_proj", _ssm_t0_out);
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
            // T171 — Sync AMX (Apple Accelerate) offload for routing matmul.
            // Opt-in via RUSTORCH_AMX_ROUTING=1.
            //
            // **EXPERIMENTAL VALIDATION RESULT (negative for sync, validates
            // need for async)**:
            //   Standalone bench : AMX 9.4 µs/call vs GPU 50-70 µs (5-7× speedup).
            //   Sync end-to-end on 35B-A3B Qwen3.6 :
            //     baseline (no AMX)        : 39.48 t/s prefill, 41.80 t/s decode
            //     RUSTORCH_AMX_ROUTING=1   : 29.58 t/s prefill (-25%), 34.60 t/s (-17%)
            //   Quality : bit-identical greedy output.
            //
            // The drain cost (~250 µs × 40 layers = 10 ms/token) eats the AMX
            // savings (~1.6 ms compute saved). ASYNC pattern (MTLEvent + CPU
            // thread) is required to actually exploit AMX. See note 1ce4dd46
            // for the future async design and 3a0040ea for the full Innovation 1
            // RFC.
            //
            // Code kept as opt-in for validation point; env-gated so default
            // runs unaffected.
            let _moe_t0_routing = std::time::Instant::now();
            let use_amx_sync = std::env::var("RUSTORCH_AMX_ROUTING").is_ok()
                && matches!(gate_inp.dtype, GgmlType::F32);
            let use_amx_async = std::env::var("RUSTORCH_AMX_ROUTING_ASYNC").is_ok()
                && matches!(gate_inp.dtype, GgmlType::F32);
            if use_amx_async {
                // T172 Day 5 + T173 amortization — async hybrid GPU+CPU routing.
                // GPU produced h. Encode signal_event, submit AMX, do GPU work
                // INDEPENDENT of routing (shared expert path + zero(moe_acc) +
                // dot_scalar) in the gap, then encode_wait_for_event.
                // The GPU work between submit and wait amortizes the encoder-break
                // overhead, hiding the AMX 9 µs entirely behind ~80-150 µs of
                // shared-expert GPU compute.
                //
                // Day 5 was: submit + immediate wait → -10/-14% (overhead > savings).
                // T173 reorders to put shared expert + dot_scalar + zero(moe_acc)
                // BETWEEN submit and wait, expecting net positive gain.
                let exec = amx_executor(backend);
                let (wait_v, signal_v) = exec.next_event_pair();
                backend.encode_signal_event(exec.event(), wait_v);
                exec.submit_with_sync(
                    rustorch_metal::async_amx::AmxJob {
                        h_ptr: scratch.h.contents() as *const f32,
                        w_ptr: gate_inp.buffer.contents() as *const f32,
                        out_ptr: scratch.moe_logits.contents() as *mut f32,
                        k: gate_inp.k,
                        n: gate_inp.n,
                    },
                    wait_v,
                    signal_v,
                );

                // === GPU work in PARALLEL with CPU AMX (no read of moe_logits here) ===
                // 1. Zero accumulator early (otherwise done later at line ~2678)
                zero_f32(backend, &scratch.moe_acc, d)?;
                // 2. Shared expert path (independent of routing)
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
                // 3. Dot scalar for shared gate
                sgemv_f32_lcpp_simd_into(
                    backend,
                    &scratch.h,
                    gate_inp_shexp,
                    &scratch.moe_dot_scalar,
                    d,
                    1,
                )?;
                // === All this GPU work overlapped with CPU AMX routing ===

                backend.encode_wait_for_event(exec.event(), signal_v);
                // Sigmoid_add_moe needs both moe_acc (filled by gather chain
                // below) and moe_expert_out (already done above) — but it must
                // run AFTER the gather chain. Skip the redundant later
                // shared+dot calls by setting a flag.
            } else if use_amx_sync {
                // Drain GPU so h is fully written before CPU AMX reads it.
                backend.drain();
                let k = gate_inp.k;
                let n = gate_inp.n;
                unsafe {
                    let h_ptr = scratch.h.contents() as *const f32;
                    let w_ptr = gate_inp.buffer.contents() as *const f32;
                    let y_ptr = scratch.moe_logits.contents() as *mut f32;
                    rustorch_cpu::accelerate::cblas_sgemv(
                        rustorch_cpu::accelerate::CBLAS_ROW_MAJOR,
                        rustorch_cpu::accelerate::CBLAS_NO_TRANS,
                        n as i32, // m of A
                        k as i32, // n of A
                        1.0,
                        w_ptr,
                        k as i32, // lda
                        h_ptr,
                        1,
                        0.0,
                        y_ptr,
                        1,
                    );
                }
            } else {
                gate_inp.matmul_into(backend, &scratch.h, &scratch.moe_logits)?;
            }

            // 2. T152.1b — Top-K + softmax + renormalize 100% GPU.
            //    Avant : drain + CPU softmax + argsort + renormalize. Maintenant :
            //    1 dispatch d'un threadgroup unique 256 threads qui produit
            //    `moe_indices_buf [n_used] u32` + `moe_topw_buf [n_used] f32`.
            //    Économie : 1 drain × 16 MoE layers = 16 drains/token sur 35B-A3B.
            //
            // Note T165 (negative result) : tentative de fusion en 1 kernel
            // (routing_topk_softmax_norm_f32) régresse -50% car le matmul
            // perd son parallélisme massif (256 sgs //) en se contractant à
            // 1 TG (8 sgs serial). Voir gotcha pour le détail.
            //
            // T182 — `topk_softmax_norm_parallel_f32` réduit la boucle topk
            // sérielle (1 thread × K × N = 2048 ops) à une réduction
            // simdgroup-coopérative (K × 40 ops). Speedup ×9 (115µs → 12µs)
            // sur micro-bench M4 Max → ~4 ms/token gain attendu sur le décode.
            // Activable via RUSTORCH_TOPK_PARALLEL=1 (default). Set =0 pour
            // bisection / régression.
            let use_parallel_topk = std::env::var("RUSTORCH_TOPK_PARALLEL")
                .map(|v| v != "0")
                .unwrap_or(true);
            if use_parallel_topk {
                topk_softmax_norm_parallel_f32(
                    backend,
                    &scratch.moe_logits,
                    &scratch.moe_indices_buf,
                    &scratch.moe_topw_buf,
                    n_experts,
                    n_used,
                )?;
            } else {
                topk_softmax_norm_f32(
                    backend,
                    &scratch.moe_logits,
                    &scratch.moe_indices_buf,
                    &scratch.moe_topw_buf,
                    n_experts,
                    n_used,
                )?;
            }
            profile_drain_record(backend, "  moe.routing", _moe_t0_routing);

            // 3. T147a — zero the accumulator on GPU (no drain).
            // T173 : skip if async path already zeroed it during AMX overlap.
            // T193 (REJECTED, default OFF) : `weighted_reduce_store_f32`
            // overwrites moe_acc in 1 dispatch instead of (zero + accumulate).
            // Mesured WASH on 35B-A3B (-0.2%, within noise, 6 alternated runs).
            // Cause : zero_f32 is only 1.5 µs (already cheap), and the store
            // variant runs at the same kernel time as the accumulate. Encoder
            // sharing already amortizes the dispatch overhead. Kernel kept in
            // metal/kernels.rs and activable via RUSTORCH_REDUCE_STORE=1 for
            // future combination experiments.
            let use_reduce_store = std::env::var("RUSTORCH_REDUCE_STORE")
                .map(|v| v == "1")
                .unwrap_or(false);
            if !use_amx_async && !use_reduce_store {
                zero_f32(backend, &scratch.moe_acc, d)?;
            }

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
                    // T177 (rejeté, re-confirmation de T168) — wired
                    // sgemv_q4_k_gather_qmv_fast_into ici → 0% gain mesuré
                    // (42.0 vs 42.4 t/s sur 5 prompts naturels). MLX qmv_fast
                    // pattern n'apporte rien sur Q4_K dequant ALU-saturé.
                    // Cf. T168 dead-code doc dans kernels.rs:7901.
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
            let _moe_t0_gate = std::time::Instant::now();
            gather_dispatch(
                gate_exps_stacked,
                &scratch.h,
                &scratch.moe_gate_gather,
                d,
                ef,
                0, // x_stride_floats=0 → broadcast
            )
            .map_err(|e| MetalError::Unsupported(format!("moe gate gather: {e:?}")))?;
            profile_drain_record(backend, "  moe.gather_gate", _moe_t0_gate);

            // up_proj : [n_used, ef] = stacked_up[indices, :, :] @ h (broadcast)
            //
            // T191 — when up_exps is Q4_K, fuse the SwiGLU into the up_proj
            // kernel output. Saves 1 dispatch/layer × 40 = 40 dispatches/token,
            // ~200-400 µs decode wall-clock saved (CPU encoder overhead).
            // Activable via RUSTORCH_GATHER_SWIGLU_FUSED (default ON).
            // Microbench parity : byte-identique avec gather + swiglu_f32.
            let use_gather_swiglu_fused = std::env::var("RUSTORCH_GATHER_SWIGLU_FUSED")
                .map(|v| v != "0")
                .unwrap_or(true);
            let _moe_t0_up = std::time::Instant::now();
            if use_gather_swiglu_fused && up_exps_stacked.dtype == GgmlType::Q4_K {
                use rustorch_metal::kernels::sgemv_q4_k_gather_swiglu_f32_lcpp_nsg2_into;
                sgemv_q4_k_gather_swiglu_f32_lcpp_nsg2_into(
                    backend,
                    &scratch.h,
                    &up_exps_stacked.buffer,
                    &scratch.moe_indices_buf,
                    n_used,
                    &scratch.moe_fd_gather,
                    &scratch.moe_gate_gather,
                    d,
                    ef,
                    up_exps_stacked.bytes_per_expert,
                    0,
                )
                .map_err(|e| MetalError::Unsupported(format!("moe up+swiglu fused: {e:?}")))?;
                profile_drain_record(backend, "  moe.gather_up_swiglu", _moe_t0_up);
            } else {
                gather_dispatch(
                    up_exps_stacked,
                    &scratch.h,
                    &scratch.moe_up_gather,
                    d,
                    ef,
                    0,
                )
                .map_err(|e| MetalError::Unsupported(format!("moe up gather: {e:?}")))?;
                profile_drain_record(backend, "  moe.gather_up", _moe_t0_up);

                // swiglu : fd_gather[b, i] = silu(gate[b, i]) * up[b, i] sur n_used*ef
                // éléments traités comme un tableau 1D (kernel élément-wise pur).
                let _moe_t0_swiglu = std::time::Instant::now();
                swiglu_f32(
                    backend,
                    &scratch.moe_gate_gather,
                    &scratch.moe_up_gather,
                    &scratch.moe_fd_gather,
                    n_used * ef,
                )?;
                profile_drain_record(backend, "  moe.swiglu_top", _moe_t0_swiglu);
            }

            // down_proj : [n_used, d] = stacked_down[indices, :, :] @ fd_gather (per-row)
            let _moe_t0_down = std::time::Instant::now();
            gather_dispatch(
                down_exps_stacked,
                &scratch.moe_fd_gather,
                &scratch.moe_down_gather,
                ef,
                d,
                ef, // x_stride_floats=ef → per-row input
            )
            .map_err(|e| MetalError::Unsupported(format!("moe down gather: {e:?}")))?;
            profile_drain_record(backend, "  moe.gather_down", _moe_t0_down);

            // T152 — somme pondérée multi-row : moe_acc = sum_b top_w[b] * down_gather[b, :]
            // T193 — store-only variant (=) au lieu d'accumulate (+=) : élimine
            // le zero_f32(moe_acc) précédent. Saves 1 dispatch + 1 DRAM read/layer.
            let _moe_t0_reduce = std::time::Instant::now();
            if use_reduce_store && !use_amx_async {
                use rustorch_metal::kernels::weighted_reduce_store_f32;
                weighted_reduce_store_f32(
                    backend,
                    &scratch.moe_down_gather,
                    &scratch.moe_topw_buf,
                    &scratch.moe_acc,
                    n_used,
                    d,
                )?;
            } else {
                weighted_reduce_add_f32(
                    backend,
                    &scratch.moe_down_gather,
                    &scratch.moe_topw_buf,
                    &scratch.moe_acc,
                    n_used,
                    d,
                )?;
            }
            profile_drain_record(backend, "  moe.reduce_top", _moe_t0_reduce);

            // 5. Shared expert: standard SwiGLU FFN with sigmoid gate scalar.
            //    shared_gate is a vector of size d (per-element gate, not scalar).
            //    llama.cpp uses ffn_gate_inp_shexp · h then sigmoid → scalar per token.
            //    But the GGUF stores ffn_gate_inp_shexp as [d] f32 — applied as a
            //    point-wise (NOT a dot product). Reading llama.cpp again:
            //    `shared_gate = build_lora_mm(ffn_gate_inp_shexp, cur)` with
            //    ffn_gate_inp_shexp of shape [d] would imply a 1×d matrix → output
            //    is a scalar per token. So a dot product.
            // T173 : skip if async path already ran shared expert during AMX overlap.
            // T192 — fuse swiglu into up_shexp matmul output when up_shexp is Q4_K.
            // Eliminates 1 dispatch/layer × 40 = 40 dispatches/token. Activable
            // via RUSTORCH_SHEXP_SWIGLU_FUSED (default ON). Parity verified.
            let use_shexp_swiglu_fused = std::env::var("RUSTORCH_SHEXP_SWIGLU_FUSED")
                .map(|v| v != "0")
                .unwrap_or(true);
            let _moe_t0_shared = std::time::Instant::now();
            if !use_amx_async {
                gate_shexp.matmul_into(backend, &scratch.h, &scratch.moe_gate)?;
                if use_shexp_swiglu_fused && up_shexp.dtype == GgmlType::Q4_K {
                    use rustorch_metal::kernels::sgemv_q4_k_swiglu_f32_lcpp_nsg2_into;
                    sgemv_q4_k_swiglu_f32_lcpp_nsg2_into(
                        backend,
                        &scratch.h,
                        &up_shexp.buffer,
                        &scratch.moe_fd,
                        &scratch.moe_gate,
                        up_shexp.k,
                        up_shexp.n,
                    )?;
                } else {
                    up_shexp.matmul_into(backend, &scratch.h, &scratch.moe_up)?;
                    swiglu_f32(
                        backend,
                        &scratch.moe_gate,
                        &scratch.moe_up,
                        &scratch.moe_fd,
                        ef,
                    )?;
                }
                down_shexp.matmul_into(backend, &scratch.moe_fd, &scratch.moe_expert_out)?;
            }
            profile_drain_record(backend, "  moe.shared", _moe_t0_shared);

            // T152.1 — Shared expert gating + final add ENTIÈREMENT GPU.
            // Avant : drain + CPU dot + CPU sigmoid + CPU add. Maintenant :
            // - sgemv N=1 calcule `dot(gate_inp_shexp, h)` dans moe_dot_scalar
            // - `sigmoid_add_moe_f32` lit le scalaire et applique
            //   `xd[i] += moe_acc[i] + sigmoid(scalar) * moe_expert_out[i]`
            // 1 dispatch GPU au lieu de 1 drain + 2 CPU loops sur d éléments.
            // Économie : 1 drain × 16 MoE layers = 16 drains/token sur 35B-A3B.
            // T173 : skip dot_scalar if async path already computed it.
            // T188 — when AMX async is OFF (default), use the fused
            // `sigmoid_add_moe_dot_fused_f32` kernel which inlines the dot
            // computation into the same dispatch. Saves 1 dispatch per MoE
            // layer × 40 = 40 dispatches/token.
            // Microbench : 10.3 µs (2 dispatches) → 5.6 µs (fused) = ×1.84.
            // Activable via RUSTORCH_DOT_FUSED (default ON, set =0 for bisect).
            let _moe_t0_final = std::time::Instant::now();
            let use_dot_fused = std::env::var("RUSTORCH_DOT_FUSED")
                .map(|v| v != "0")
                .unwrap_or(true);
            if !use_amx_async && use_dot_fused {
                sigmoid_add_moe_dot_fused_f32(
                    backend,
                    &scratch.moe_acc,
                    &scratch.moe_expert_out,
                    gate_inp_shexp,
                    &scratch.h,
                    &scratch.xd,
                    d,
                )?;
            } else {
                if !use_amx_async {
                    sgemv_f32_lcpp_simd_into(
                        backend,
                        &scratch.h,
                        gate_inp_shexp,
                        &scratch.moe_dot_scalar,
                        d,
                        1,
                    )?;
                }
                sigmoid_add_moe_f32(
                    backend,
                    &scratch.moe_acc,
                    &scratch.moe_expert_out,
                    &scratch.moe_dot_scalar,
                    &scratch.xd,
                    d,
                )?;
            }
            profile_drain_record(backend, "  moe.final_add", _moe_t0_final);
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
        // the 14B (T133) was every 5 layers. T176 — make the period
        // configurable to A/B test on 35B-A3B (different layer count).
        // RUSTORCH_COMMIT_PERIOD=0 disables; default = 5 (legacy).
        let commit_period: usize = std::env::var("RUSTORCH_COMMIT_PERIOD")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(5);
        if dump_mode() == 0
            && commit_period > 0
            && (li + 1) % commit_period == 0
            && li + 1 < cfg.n_layers
        {
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

/// T162 phase 9d — Bundle of B_MAX-sized batched scratch buffers for
/// `forward_batch`. Combines xd residual stream + attn block scratch + FFN
/// dense scratch. SSM/MoE layers fall back to per-token sequential dispatch
/// using the regular `Scratch` (passed alongside).
struct BatchScratch {
    xd: Buffer, // [B_MAX, d] residual stream
    attn: BatchScratchAttn,
    ffn: BatchScratchFfn,
    ssm: BatchScratchSsm,
    moe: BatchScratchMoe,
    // T176 — Speculative decoding scratch : batched final norm + lm_head outputs.
    h_final_batched: Buffer, // [B_MAX, d] post-final-norm
    logits_batched: Buffer,  // [B_MAX, vocab]
    argmax_indices: Buffer,  // [B_MAX] u32 — GPU argmax output
}

impl BatchScratch {
    fn new(backend: &MetalBackend, cfg: &Qwen35Config) -> Self {
        let d = cfg.d;
        let vocab = cfg.vocab;
        Self {
            xd: backend.alloc_shared(B_MAX_BATCH * d * 4).unwrap(),
            attn: BatchScratchAttn::new(backend, cfg),
            ffn: BatchScratchFfn::new(backend, cfg),
            ssm: BatchScratchSsm::new(backend, cfg),
            moe: BatchScratchMoe::new(backend, cfg),
            h_final_batched: backend.alloc_shared(B_MAX_BATCH * d * 4).unwrap(),
            logits_batched: backend.alloc_shared(B_MAX_BATCH * vocab * 4).unwrap(),
            argmax_indices: backend.alloc_shared(B_MAX_BATCH * 4).unwrap(),
        }
    }
}

/// T162 phase 9d — Batched forward for B-token prefill on Qwen3.5/3.6 hybrid.
///
/// Per-layer dispatch :
///   - Attn + Dense FFN  : full batched (attn_block_forward_batch + ffn_dense_forward_batch)
///   - Attn + MoE FFN    : per-token loop (sequential forward_token-style block)
///   - SSM  + Dense FFN  : SSM scan per-token (state recurrent), FFN dense batched
///   - SSM  + MoE FFN    : per-token loop full layer
///
/// Pour 27B Dense (16 attn + 48 ssm) : ~50% des matmuls batchés (16×4 attn +
/// 64×3 FFN dense) — Attn projections + tous les FFN. SSM scan reste séquentiel.
///
/// Pour 35B-A3B MoE : pas de speedup en phase 9d (tous les FFN sont MoE,
/// donc per-token loop). 9f portera le MoE gather batched.
///
/// Final norm + lm_head : on calcule UNIQUEMENT pour le dernier token (B-1)
/// car en prefill on veut juste le next-token id à partir du prompt complet.
/// (Speculative decoding voudrait tous les B logits — phase ultérieure.)
///
/// Pré-conditions :
///   - 1 ≤ B ≤ B_MAX_BATCH
///   - `tokens.len() == B`
///   - `pos_base + B ≤ max_seq`
#[allow(clippy::too_many_arguments)]
fn forward_batch(
    backend: &MetalBackend,
    file: &GgufFile,
    model: &Qwen35MetalModel,
    state: &mut DecodeState,
    batch_scratch: &BatchScratch,
    tokens: &[u32],
    pos_base: usize,
) -> Result<u32, String> {
    let outs = forward_batch_argmax(
        backend,
        file,
        model,
        state,
        batch_scratch,
        tokens,
        pos_base,
        false,
    )?;
    Ok(*outs.last().unwrap())
}

/// T176 — Forward batch with per-token argmax option (speculative-friendly).
///
/// Same layer-by-layer body as `forward_batch`, but the final norm + lm_head
/// can be applied to either :
/// - just the LAST token (`all_argmax=false`, current prefill behavior)
/// - ALL B tokens (`all_argmax=true`, needed for speculative decoding to
///   verify each candidate's predicted next token).
///
/// Returns Vec of argmax tokens — length 1 if `all_argmax=false`, length B
/// if `all_argmax=true`.
#[allow(clippy::too_many_arguments)]
fn forward_batch_argmax(
    backend: &MetalBackend,
    file: &GgufFile,
    model: &Qwen35MetalModel,
    state: &mut DecodeState,
    batch_scratch: &BatchScratch,
    tokens: &[u32],
    pos_base: usize,
    all_argmax: bool,
) -> Result<Vec<u32>, String> {
    let b = tokens.len();
    if b == 0 || b > B_MAX_BATCH {
        return Err(format!(
            "forward_batch: B must be in 1..={B_MAX_BATCH} (got {b})"
        ));
    }
    let cfg = &model.cfg;
    let d = cfg.d;
    let max_seq = state.max_seq;

    // 1. Embed B tokens into xd_batched [B, d]. T175 P3 — write directly into
    //    batch_scratch.xd[bi*d..] via dequant + raw ptr, eliminating
    //    `backend.alloc_shared(d*4)` per token (was ~6 ms/chunk at B=128).
    {
        let info = file
            .tensor("token_embd.weight")
            .ok_or_else(|| "missing token_embd.weight".to_string())?;
        let row_bytes = info.byte_size() as usize / cfg.vocab;
        let bytes = file.tensor_bytes(info);
        let mut row_info = info.clone();
        row_info.shape = vec![cfg.d as u64];
        for (bi, &tok) in tokens.iter().enumerate() {
            let row_start = (tok as usize) * row_bytes;
            let row_end = row_start + row_bytes;
            let f32_row = dequant_to_f32(&row_info, &bytes[row_start..row_end])
                .map_err(|e| format!("token_embd dequant: {e:?}"))?;
            if f32_row.len() != cfg.d {
                return Err(format!(
                    "token_embd: expected {} f32, got {}",
                    cfg.d,
                    f32_row.len()
                ));
            }
            unsafe {
                let dst = (batch_scratch.xd.contents() as *mut f32).add(bi * d);
                std::ptr::copy_nonoverlapping(f32_row.as_ptr(), dst, d);
            }
        }
    }

    // 2. Per-layer dispatch.
    for li in 0..cfg.n_layers {
        let layer = &model.layers[li];
        match (layer, &state.layers[li]) {
            (
                LayerMetal::Attn {
                    attn,
                    ffn:
                        FfnLayerMetal::Dense {
                            w_gate,
                            w_up,
                            w_down,
                        },
                },
                LayerState::Attn(cache),
            ) => {
                // Full batched : Attn + FFN dense.
                let _t0_ab = std::time::Instant::now();
                attn_block_forward_batch(
                    backend,
                    attn,
                    cache,
                    &batch_scratch.xd,
                    &state.rope_cos,
                    &state.rope_sin,
                    &batch_scratch.attn,
                    cfg,
                    pos_base,
                    b,
                    max_seq,
                )
                .map_err(|e| format!("L{li} attn batched: {e:?}"))?;
                profile_drain_record(backend, "  fb.attn_block_batched", _t0_ab);
                let _t0_fb = std::time::Instant::now();
                ffn_dense_forward_batch(
                    backend,
                    &attn.attn_post_norm,
                    w_gate,
                    w_up,
                    w_down,
                    &batch_scratch.xd,
                    &batch_scratch.ffn,
                    cfg,
                    b,
                )
                .map_err(|e| format!("L{li} ffn batched: {e:?}"))?;
                profile_drain_record(backend, "  fb.ffn_dense_batched", _t0_fb);
            },
            (
                LayerMetal::Attn {
                    attn,
                    ffn:
                        FfnLayerMetal::Moe {
                            gate_inp,
                            gate_inp_shexp,
                            gate_shexp,
                            up_shexp,
                            down_shexp,
                            gate_exps_stacked,
                            up_exps_stacked,
                            down_exps_stacked,
                        },
                },
                LayerState::Attn(cache),
            ) => {
                // T170 — Attn batched + MoE FFN BATCHED (was per-token loop with
                // drain × B that ate 1.28s for B=128 on 35B-A3B prefill).
                // Replaces dead-code ffn_moe_forward_batch entry into the path.
                let _t0_ab = std::time::Instant::now();
                attn_block_forward_batch(
                    backend,
                    attn,
                    cache,
                    &batch_scratch.xd,
                    &state.rope_cos,
                    &state.rope_sin,
                    &batch_scratch.attn,
                    cfg,
                    pos_base,
                    b,
                    state.max_seq,
                )
                .map_err(|e| format!("L{li} attn batched: {e:?}"))?;
                profile_drain_record(backend, "  fb.attn_block_batched", _t0_ab);
                let _t0_fbm = std::time::Instant::now();
                ffn_moe_forward_batch(
                    backend,
                    &attn.attn_post_norm,
                    gate_inp,
                    gate_inp_shexp,
                    gate_shexp,
                    up_shexp,
                    down_shexp,
                    gate_exps_stacked,
                    up_exps_stacked,
                    down_exps_stacked,
                    &batch_scratch.xd,
                    &state.scratch,
                    &batch_scratch.moe,
                    cfg,
                    b,
                )
                .map_err(|e| format!("L{li} moe batched: {e:?}"))?;
                profile_drain_record(backend, "  fb.ffn_moe_batched", _t0_fbm);
            },
            (
                LayerMetal::Ssm {
                    ssm,
                    ffn:
                        FfnLayerMetal::Dense {
                            w_gate,
                            w_up,
                            w_down,
                        },
                },
                LayerState::Ssm(s),
            ) => {
                // T162 phase 9e — SSM block batched (projections + apply_gate
                // batched, scan séquentiel, output proj batched).
                let _t0_sb = std::time::Instant::now();
                ssm_block_forward_batch(
                    backend,
                    ssm,
                    s,
                    &batch_scratch.xd,
                    &batch_scratch.ssm,
                    &state.scratch,
                    cfg,
                    b,
                )
                .map_err(|e| format!("L{li} ssm batched: {e:?}"))?;
                profile_drain_record(backend, "  fb.ssm_block_batched", _t0_sb);
                // FFN dense batched (post-attn norm + gate/up/swiglu/down + residual).
                let _t0_fb = std::time::Instant::now();
                ffn_dense_forward_batch(
                    backend,
                    &ssm.attn_post_norm,
                    w_gate,
                    w_up,
                    w_down,
                    &batch_scratch.xd,
                    &batch_scratch.ffn,
                    cfg,
                    b,
                )
                .map_err(|e| format!("L{li} ssm-ffn batched: {e:?}"))?;
                profile_drain_record(backend, "  fb.ffn_dense_batched", _t0_fb);
            },
            (
                LayerMetal::Ssm {
                    ssm,
                    ffn:
                        FfnLayerMetal::Moe {
                            gate_inp,
                            gate_inp_shexp,
                            gate_shexp,
                            up_shexp,
                            down_shexp,
                            gate_exps_stacked,
                            up_exps_stacked,
                            down_exps_stacked,
                        },
                },
                LayerState::Ssm(s),
            ) => {
                // T170 — SSM batched + MoE FFN BATCHED. Identical fix to the
                // Attn+MoE branch above: removes per-token drain loop that ate
                // most of the prefill time on 35B-A3B (30 SSM+MoE layers × B
                // drains = 38400 drains for B=128).
                let _t0_sb = std::time::Instant::now();
                ssm_block_forward_batch(
                    backend,
                    ssm,
                    s,
                    &batch_scratch.xd,
                    &batch_scratch.ssm,
                    &state.scratch,
                    cfg,
                    b,
                )
                .map_err(|e| format!("L{li} ssm batched: {e:?}"))?;
                profile_drain_record(backend, "  fb.ssm_block_batched", _t0_sb);
                let _t0_fbm = std::time::Instant::now();
                ffn_moe_forward_batch(
                    backend,
                    &ssm.attn_post_norm,
                    gate_inp,
                    gate_inp_shexp,
                    gate_shexp,
                    up_shexp,
                    down_shexp,
                    gate_exps_stacked,
                    up_exps_stacked,
                    down_exps_stacked,
                    &batch_scratch.xd,
                    &state.scratch,
                    &batch_scratch.moe,
                    cfg,
                    b,
                )
                .map_err(|e| format!("L{li} ssm+moe batched: {e:?}"))?;
                profile_drain_record(backend, "  fb.ffn_moe_batched", _t0_fbm);
            },
            _ => return Err(format!("L{li}: kind/state mismatch")),
        }
    }

    // 3. Final norm + lm_head — branch sur all_argmax.
    if !all_argmax {
        // Path classique : seulement le dernier token (next-token pred prefill).
        backend.drain();
        unsafe {
            let src = (batch_scratch.xd.contents() as *const f32).add((b - 1) * d);
            let dst = state.scratch.xd.contents() as *mut f32;
            std::ptr::copy_nonoverlapping(src, dst, d);
        }
        rms_norm_f32(
            backend,
            &state.scratch.xd,
            &model.output_norm,
            &state.scratch.h,
            d,
            cfg.rms_eps,
        )
        .map_err(|e| format!("final norm: {e:?}"))?;
        model
            .output
            .matmul_into(backend, &state.scratch.h, &state.scratch.logits)
            .map_err(|e| format!("lm_head: {e:?}"))?;
        backend.drain();
        Ok(vec![argmax_cpu(&state.scratch.logits, cfg.vocab)])
    } else {
        // T176 — Speculative path : final norm + lm_head batched sur tous les B
        // tokens. Renvoie un argmax par row → Vec<u32> de taille B.
        rms_norm_batched_f32(
            backend,
            &batch_scratch.xd,
            &model.output_norm,
            &batch_scratch.h_final_batched,
            d,
            b,
            cfg.rms_eps,
        )
        .map_err(|e| format!("final_norm batched: {e:?}"))?;
        model
            .output
            .matmul_batched_into(
                backend,
                b,
                &batch_scratch.h_final_batched,
                &batch_scratch.logits_batched,
            )
            .map_err(|e| format!("lm_head batched: {e:?}"))?;
        // T176 — GPU argmax (1 dispatch B threadgroups vs CPU loop B × vocab=151936
        // reads ≈ 5 ms per spec round). Reads only B u32 indices CPU-side post-drain.
        argmax_batched_f32(
            backend,
            &batch_scratch.logits_batched,
            &batch_scratch.argmax_indices,
            b,
            cfg.vocab,
        )
        .map_err(|e| format!("argmax batched: {e:?}"))?;
        backend.drain();
        let mut outs = vec![0u32; b];
        unsafe {
            std::ptr::copy_nonoverlapping(
                batch_scratch.argmax_indices.contents() as *const u32,
                outs.as_mut_ptr(),
                b,
            );
        }
        Ok(outs)
    }
}

/// T162 phase 9d helper — single-layer per-token fallback for Attn+MoE / SSM+MoE.
/// Iterates B times, copying xd_batched[bi] ↔ scratch.xd around a regular
/// per-token block dispatch (attn_block_forward / ssm_block_forward + post-norm
/// + ffn_dense_forward which handles the MoE branch).
#[allow(clippy::too_many_arguments)]
fn forward_batch_per_token_layer(
    backend: &MetalBackend,
    model: &Qwen35MetalModel,
    state: &mut DecodeState,
    batch_scratch: &BatchScratch,
    li: usize,
    b: usize,
    pos_base: usize,
) -> Result<(), String> {
    let cfg = &model.cfg;
    let d = cfg.d;
    // CRITICAL : drain pour que les GPU writes des layers précédentes soient
    // visibles avant les CPU-memcpy lectures de xd_batched.
    backend.drain();
    for bi in 0..b {
        unsafe {
            let src = (batch_scratch.xd.contents() as *const f32).add(bi * d);
            let dst = state.scratch.xd.contents() as *mut f32;
            std::ptr::copy_nonoverlapping(src, dst, d);
        }
        match (&model.layers[li], &state.layers[li]) {
            (LayerMetal::Attn { attn, ffn }, LayerState::Attn(cache)) => {
                attn_block_forward(
                    backend,
                    attn,
                    cache,
                    &state.scratch,
                    &state.rope_cos,
                    &state.rope_sin,
                    cfg,
                    pos_base + bi,
                    state.max_seq,
                )
                .map_err(|e| format!("L{li}/{bi} attn: {e:?}"))?;
                rms_norm_f32(
                    backend,
                    &state.scratch.xd,
                    &attn.attn_post_norm,
                    &state.scratch.h,
                    d,
                    cfg.rms_eps,
                )
                .map_err(|e| format!("L{li}/{bi} post norm: {e:?}"))?;
                ffn_dense_forward(backend, ffn, &state.scratch, cfg)
                    .map_err(|e| format!("L{li}/{bi} ffn: {e:?}"))?;
            },
            (LayerMetal::Ssm { ssm, ffn }, LayerState::Ssm(s)) => {
                let prefix = format!("p{:03}/L{li:02}", pos_base + bi);
                ssm_block_forward(backend, ssm, s, &state.scratch, cfg, &prefix)
                    .map_err(|e| format!("L{li}/{bi} ssm: {e:?}"))?;
                rms_norm_f32(
                    backend,
                    &state.scratch.xd,
                    &ssm.attn_post_norm,
                    &state.scratch.h,
                    d,
                    cfg.rms_eps,
                )
                .map_err(|e| format!("L{li}/{bi} post norm: {e:?}"))?;
                ffn_dense_forward(backend, ffn, &state.scratch, cfg)
                    .map_err(|e| format!("L{li}/{bi} ffn: {e:?}"))?;
            },
            _ => return Err(format!("L{li}/{bi}: kind/state mismatch")),
        }
        backend.drain();
        unsafe {
            let src = state.scratch.xd.contents() as *const f32;
            let dst = (batch_scratch.xd.contents() as *mut f32).add(bi * d);
            std::ptr::copy_nonoverlapping(src, dst, d);
        }
    }
    Ok(())
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

/// T162 phase 9b — Parité `attn_block_forward_batch` vs B sequential
/// `attn_block_forward` calls. Sélectionne la 1ère layer Attn du modèle, alloue
/// 2 KV caches indépendants (séq vs batched), compare les résidus xd[i] post-
/// attention. Vise rel < 1e-3 (B doit reproduire exactement la séquence des B
/// forward_token avec le même état de cache).
fn parity_attn_batch_on_loaded_weights(
    backend: &MetalBackend,
    cfg: &Qwen35Config,
    model: &Qwen35MetalModel,
) -> Result<(), String> {
    // Sélection : 1ère layer Attn du modèle. FFN dense optionnel : si trouvé
    // on teste aussi ffn_dense_forward_batch (phase 9c) end-to-end ; sinon
    // (35B-A3B = MoE-only) on teste juste attn_block_forward_batch (phase 9b)
    // — le MoE batched arrive en phase 9f.
    let attn = model
        .layers
        .iter()
        .find_map(|l| match l {
            LayerMetal::Attn { attn, .. } => Some(attn),
            _ => None,
        })
        .ok_or("no Attn layer found")?;
    let ffn_dense: Option<(&HybridMetalWeight, &HybridMetalWeight, &HybridMetalWeight)> =
        model.layers.iter().find_map(|l| match l {
            LayerMetal::Attn {
                ffn:
                    FfnLayerMetal::Dense {
                        w_gate,
                        w_up,
                        w_down,
                    },
                ..
            } => Some((w_gate, w_up, w_down)),
            _ => None,
        });

    let d = cfg.d;
    let head_dim = cfg.attn_head_dim;
    let n_kv = cfg.n_kv_heads;
    let kv_dim = head_dim * n_kv;
    let max_seq = 64_usize;
    // T162 phase 9d : also test B=1 (forward_batch path called per chunk
    // when prompt has only 1 token).
    let b: usize = std::env::var("RUSTORCH_PARITY_B")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8);
    let pos_base = 0_usize;

    let phase_label = if ffn_dense.is_some() {
        "phase 9b+9c (attn + FFN dense)"
    } else {
        "phase 9b only (no Attn+Dense — MoE-only model)"
    };
    println!("=== T162 {phase_label} — Parity batched (B={b}) ===");
    println!(
        "  d={d} head_dim={head_dim} n_q={} n_kv={n_kv} rope_dim={}",
        cfg.n_q_heads, cfg.rope_dim
    );

    // Build deterministic xd_all [B, d].
    let mut xd_all = vec![0.0f32; b * d];
    for bi in 0..b {
        for i in 0..d {
            xd_all[bi * d + i] = ((i as f32 + 1.0 + bi as f32 * 7.0) * 0.0007).sin() * 0.4;
        }
    }

    // Pre-build RoPE tables for [0, max_seq).
    let half = cfg.rope_dim / 2;
    let mut cos_tab = vec![0.0_f32; max_seq * half];
    let mut sin_tab = vec![0.0_f32; max_seq * half];
    for pos in 0..max_seq {
        for i in 0..half {
            let theta = (pos as f32) / cfg.rope_base.powf((2 * i) as f32 / cfg.rope_dim as f32);
            cos_tab[pos * half + i] = theta.cos();
            sin_tab[pos * half + i] = theta.sin();
        }
    }
    let rope_cos = backend.alloc_shared(max_seq * half * 4).unwrap();
    let rope_sin = backend.alloc_shared(max_seq * half * 4).unwrap();
    unsafe {
        std::ptr::copy_nonoverlapping(
            cos_tab.as_ptr(),
            rope_cos.contents() as *mut f32,
            max_seq * half,
        );
        std::ptr::copy_nonoverlapping(
            sin_tab.as_ptr(),
            rope_sin.contents() as *mut f32,
            max_seq * half,
        );
    }

    // === Sequential reference path: B calls of attn_block_forward, each with
    //     its own pre-zeroed KV cache (just the same cache state evolving
    //     causally, since pos increases each call). ===
    let cache_seq_k = backend.alloc_shared(max_seq * kv_dim * 4).unwrap();
    let cache_seq_v = backend.alloc_shared(max_seq * kv_dim * 4).unwrap();
    unsafe {
        std::ptr::write_bytes(cache_seq_k.contents() as *mut u8, 0, max_seq * kv_dim * 4);
        std::ptr::write_bytes(cache_seq_v.contents() as *mut u8, 0, max_seq * kv_dim * 4);
    }
    let cache_seq = AttnLayerCache {
        k_cache: cache_seq_k,
        v_cache: cache_seq_v,
    };

    let scratch_seq = Scratch::new(backend, cfg);
    let mut xd_seq = vec![0.0_f32; b * d];
    for bi in 0..b {
        // Load xd[bi] into scratch_seq.xd.
        unsafe {
            std::ptr::copy_nonoverlapping(
                xd_all[bi * d..(bi + 1) * d].as_ptr(),
                scratch_seq.xd.contents() as *mut f32,
                d,
            );
        }
        attn_block_forward(
            backend,
            attn,
            &cache_seq,
            &scratch_seq,
            &rope_cos,
            &rope_sin,
            cfg,
            pos_base + bi,
            max_seq,
        )
        .map_err(|e| format!("seq[{bi}] forward: {e:?}"))?;
        // Post-attn norm + FFN dense (only if model has Attn+Dense layer).
        if let Some((w_gate, w_up, w_down)) = ffn_dense {
            rms_norm_f32(
                backend,
                &scratch_seq.xd,
                &attn.attn_post_norm,
                &scratch_seq.h,
                d,
                cfg.rms_eps,
            )
            .map_err(|e| format!("seq[{bi}] post-norm: {e:?}"))?;
            w_gate
                .matmul_into(backend, &scratch_seq.h, &scratch_seq.gate_ffn)
                .map_err(|e| format!("seq[{bi}] gate: {e:?}"))?;
            w_up.matmul_into(backend, &scratch_seq.h, &scratch_seq.up_ffn)
                .map_err(|e| format!("seq[{bi}] up: {e:?}"))?;
            swiglu_f32(
                backend,
                &scratch_seq.gate_ffn,
                &scratch_seq.up_ffn,
                &scratch_seq.fd_ffn,
                cfg.f,
            )
            .map_err(|e| format!("seq[{bi}] swiglu: {e:?}"))?;
            w_down
                .matmul_into(backend, &scratch_seq.fd_ffn, &scratch_seq.fc2)
                .map_err(|e| format!("seq[{bi}] down: {e:?}"))?;
            add_inplace_f32(backend, &scratch_seq.xd, &scratch_seq.fc2, d)
                .map_err(|e| format!("seq[{bi}] residual: {e:?}"))?;
        }
        backend.drain();
        // Read back xd post-FFN.
        unsafe {
            std::ptr::copy_nonoverlapping(
                scratch_seq.xd.contents() as *const f32,
                xd_seq[bi * d..(bi + 1) * d].as_mut_ptr(),
                d,
            );
        }
    }

    // === Batched path: 1 call of attn_block_forward_batch with own KV cache. ===
    let cache_bat_k = backend.alloc_shared(max_seq * kv_dim * 4).unwrap();
    let cache_bat_v = backend.alloc_shared(max_seq * kv_dim * 4).unwrap();
    unsafe {
        std::ptr::write_bytes(cache_bat_k.contents() as *mut u8, 0, max_seq * kv_dim * 4);
        std::ptr::write_bytes(cache_bat_v.contents() as *mut u8, 0, max_seq * kv_dim * 4);
    }
    let cache_bat = AttnLayerCache {
        k_cache: cache_bat_k,
        v_cache: cache_bat_v,
    };
    let xd_bat_buf = backend.alloc_shared(B_MAX_BATCH * d * 4).unwrap();
    unsafe {
        std::ptr::copy_nonoverlapping(xd_all.as_ptr(), xd_bat_buf.contents() as *mut f32, b * d);
    }
    let scratch_bat = BatchScratchAttn::new(backend, cfg);
    attn_block_forward_batch(
        backend,
        attn,
        &cache_bat,
        &xd_bat_buf,
        &rope_cos,
        &rope_sin,
        &scratch_bat,
        cfg,
        pos_base,
        b,
        max_seq,
    )
    .map_err(|e| format!("batched attn forward: {e:?}"))?;
    // Phase 9c — batched FFN dense (post-attn norm + gate/up/swiglu/down + residual).
    if let Some((w_gate, w_up, w_down)) = ffn_dense {
        let scratch_ffn = BatchScratchFfn::new(backend, cfg);
        ffn_dense_forward_batch(
            backend,
            &attn.attn_post_norm,
            w_gate,
            w_up,
            w_down,
            &xd_bat_buf,
            &scratch_ffn,
            cfg,
            b,
        )
        .map_err(|e| format!("batched ffn forward: {e:?}"))?;
    }
    backend.drain();

    let mut xd_bat = vec![0.0_f32; b * d];
    unsafe {
        std::ptr::copy_nonoverlapping(
            xd_bat_buf.contents() as *const f32,
            xd_bat.as_mut_ptr(),
            b * d,
        );
    }

    // Compare per-token outputs. Use max-magnitude denominator to avoid
    // inflating the rel diff in regions where seq value is near zero
    // (FP32 reduction-order noise dominates there).
    let mut max_rel = 0.0_f32;
    let mut max_abs = 0.0_f32;
    let mut max_idx = (0usize, 0usize);
    let mut sum_sq_diff = 0.0_f64;
    let mut sum_sq_seq = 0.0_f64;
    for bi in 0..b {
        for i in 0..d {
            let a = xd_seq[bi * d + i];
            let bv = xd_bat[bi * d + i];
            let abs = (a - bv).abs();
            let denom = a.abs().max(bv.abs()).max(1e-3);
            let rel = abs / denom;
            if rel > max_rel {
                max_rel = rel;
                max_abs = abs;
                max_idx = (bi, i);
            }
            sum_sq_diff += (abs as f64).powi(2);
            sum_sq_seq += (a as f64).powi(2);
        }
    }
    let rms_diff = (sum_sq_diff / (b * d) as f64).sqrt() as f32;
    let rms_seq = (sum_sq_seq / (b * d) as f64).sqrt() as f32;
    let global_rms_rel = rms_diff / rms_seq.max(1e-6);
    let (bi, i) = max_idx;
    println!(
        "  Max rel diff (max-denom): {:.3e} (abs {:.3e}) at [b={bi}, i={i}]: seq={} bat={}",
        max_rel,
        max_abs,
        xd_seq[bi * d + i],
        xd_bat[bi * d + i]
    );
    println!(
        "  Global RMS rel diff      : {:.3e} (rms_diff={:.3e}, rms_seq={:.3e})",
        global_rms_rel, rms_diff, rms_seq
    );
    // Tolerance : max-denom rel < 5e-2 (FP32 noise after RMSNorm + matmul +
    // softmax + reduce can accumulate to a few % per element) AND global
    // RMS rel < 5e-3 (the bulk of the residual must agree to 0.5 %).
    if max_rel > 5e-2 {
        return Err(format!("PARITY FAIL: max rel = {max_rel:.3e} (>5e-2)"));
    }
    if global_rms_rel > 5e-3 {
        return Err(format!(
            "PARITY FAIL: global RMS rel = {global_rms_rel:.3e} (>5e-3)"
        ));
    }
    println!("  ✓ PARITY PASS (max rel < 5e-2, RMS rel < 5e-3)");
    Ok(())
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
            if !matches!(w.dtype, GgmlType::Q3_K | GgmlType::Q4_K | GgmlType::Q6_K) {
                println!(
                    "  {label} [K={}, N={}] dtype={:?} : skipped (non-Q3/Q4/Q6_K)",
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

    // T162 phase 9b — `--parity-attn-batch` mode : valide attn_block_forward_batch
    // contre B sequential attn_block_forward sur la première layer Attn du modèle.
    // Compare le résidu xd[i] post-attention pour chaque token i ∈ [0, B).
    if env::args().any(|a| a == "--parity-attn-batch") {
        match parity_attn_batch_on_loaded_weights(backend, &cfg, &model) {
            Ok(()) => return ExitCode::SUCCESS,
            Err(e) => {
                eprintln!("parity-attn-batch failed: {e}");
                return ExitCode::FAILURE;
            },
        }
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
    // T162 phase 9d — `--prefill-batch B` : chunk size for batched prefill
    // (forward_batch). 0 / unset = legacy per-token forward_token.
    // Recommandé : 32 ou 64 pour bénéficier des SGEMM tile 64×64.
    let prefill_batch: usize = env::args()
        .skip_while(|a| a != "--prefill-batch")
        .nth(1)
        .and_then(|a| a.parse::<usize>().ok())
        .unwrap_or(0)
        .min(B_MAX_BATCH);
    // T176 — `--speculative N` : speculative decoding with N-gram lookahead.
    // 0 = off (default), 2..=8 enables with B = N candidates per round.
    let speculative_b: usize = env::args()
        .skip_while(|a| a != "--speculative")
        .nth(1)
        .and_then(|a| a.parse::<usize>().ok())
        .unwrap_or(0)
        .min(B_MAX_BATCH);

    // High-level chat: --prompt "text" auto-tokenises through the GGUF's
    // embedded vocab+merges and wraps the prompt in the Qwen ChatML
    // template. Decoded answer is printed at the end.
    let prompt_text: Option<String> = env::args().skip_while(|a| a != "--prompt").nth(1);
    let system_text: String = env::args()
        .skip_while(|a| a != "--system")
        .nth(1)
        .unwrap_or_default();
    let no_chat_template = env::args().any(|a| a == "--no-chat-template");
    let chat_mode = env::args().any(|a| a == "--chat");

    // T185 — Interactive chat mode with persistent KV cache across turns.
    // Each new user message is prefilled as a delta on top of the existing
    // KV state, so multi-turn conversations don't pay the full prefill cost
    // again. /exit, /clear, /stats commands. Streams tokens by default.
    if chat_mode {
        use std::io::{BufRead, Write};

        println!("\n=== Interactive chat mode (Qwen3.6-35B-A3B) ===");
        println!(
            "  max_seq={max_seq}, decode budget per turn={} tokens",
            forward_n.unwrap_or(2048)
        );
        if !system_text.is_empty() {
            println!("  system: {system_text}");
        }
        println!("  Commands: /exit  /clear  /stats  /system <text>");
        println!("  KV cache persists across turns (multi-turn conversation).");

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
        let im_end_id = tok.special_id("<|im_end|>").unwrap_or(tok.eos_id);
        let im_start_id = tok.special_id("<|im_start|>").unwrap_or(tok.eos_id);
        // Stop on EOS, <|im_end|> (turn end), or <|im_start|> (model started a
        // NEXT turn header — happens when the model "forgets" to emit im_end).
        let stops = [tok.eos_id, im_end_id, im_start_id];
        let max_decode = forward_n.unwrap_or(2048);

        let mut state = DecodeState::new(backend, &cfg, max_seq);
        let mut cur_pos = 0_usize;
        let mut last: u32 = 0;
        let mut turn = 0_usize;
        let mut current_system = system_text.clone();

        let stdin = std::io::stdin();
        let mut stdin_lock = stdin.lock();

        loop {
            print!("\n[user] ");
            std::io::stdout().flush().ok();
            let mut line = String::new();
            match stdin_lock.read_line(&mut line) {
                Ok(0) => break, // EOF (Ctrl-D)
                Ok(_) => {},
                Err(_) => break,
            }
            let input = line.trim();
            if input.is_empty() {
                continue;
            }
            // Slash commands.
            if let Some(rest) = input.strip_prefix("/system ") {
                current_system = rest.to_string();
                println!("(system prompt updated; will apply on /clear or next turn)");
                continue;
            }
            match input {
                "/exit" | "/quit" => {
                    println!("Bye.");
                    break;
                },
                "/clear" => {
                    state = DecodeState::new(backend, &cfg, max_seq);
                    cur_pos = 0;
                    turn = 0;
                    println!("(history cleared, KV cache reset)");
                    continue;
                },
                "/stats" => {
                    println!(
                        "  turn={turn}  kv_pos={cur_pos}  max_seq={max_seq}  remaining={}",
                        max_seq.saturating_sub(cur_pos)
                    );
                    continue;
                },
                _ => {},
            }

            // Build the chunk to prefill for this turn.
            // Turn 0: full ChatML wrapper (system + user + assistant header).
            // Turn ≥1: only the new user→assistant delta. The previous
            // assistant turn's <|im_end|> was already decoded into the KV
            // cache, so we just append a newline + new user block.
            let chunk_text = if turn == 0 {
                tok.chatml_prompt(&current_system, input)
            } else {
                format!(
                    "\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
                    input
                )
            };
            let chunk_ids = tok.encode(&chunk_text);

            // Capacity check: prefill + decode budget must fit in KV cache.
            let needed = cur_pos + chunk_ids.len() + 16; // 16 = small margin for first decode steps
            if needed >= max_seq {
                eprintln!(
                    "KV cache nearly full (pos={cur_pos}, +chunk={}, max_seq={max_seq}). Use /clear to reset history.",
                    chunk_ids.len()
                );
                continue;
            }

            // Prefill the chunk (per-token; chat mode is short bursts so
            // this is fine without batched prefill).
            let t_prefill = Instant::now();
            for &t in &chunk_ids {
                match forward_token(backend, &file, &model, &mut state, t, cur_pos) {
                    Ok(out) => last = out,
                    Err(e) => {
                        eprintln!("forward error at chat prefill: {e}");
                        return ExitCode::FAILURE;
                    },
                }
                cur_pos += 1;
            }
            let dt_prefill = t_prefill.elapsed();

            // Decode + stream.
            print!("[assistant] ");
            std::io::stdout().flush().ok();
            let mut generated: Vec<u32> = vec![last];
            // Helper : safely emit the new char-boundary-aligned suffix of
            // `cur_clean` versus `emitted`. Avoids slicing in the middle of
            // multi-byte UTF-8 chars (e.g. emoji 🇫🇷). Trailing partial
            // characters wait for the next iteration.
            let emit_safe = |cur_clean: &str, emitted: &mut String| {
                // T185-fix : the tokenizer emits U+FFFD (`�`) as placeholder
                // for incomplete multi-byte chars. When subsequent tokens
                // complete the char, the placeholder gets replaced by the
                // real char and earlier byte positions shift. To avoid
                // printing transient placeholders that get overwritten,
                // we trim trailing `�` runs from cur_clean before emitting.
                // Three bytes per `�` (U+FFFD encoded as EF BF BD).
                let mut effective_len = cur_clean.len();
                let fffd_bytes: [u8; 3] = [0xEF, 0xBF, 0xBD];
                while effective_len >= 3
                    && cur_clean.as_bytes()[effective_len - 3..effective_len] == fffd_bytes
                {
                    effective_len -= 3;
                }
                // Tokenizer decode can also change the byte-level structure
                // of earlier output as new tokens complete a multi-byte UTF-8
                // char. When that happens, `emitted` is no longer a prefix
                // of cur_clean; walk back to the largest common prefix on
                // a boundary.
                if !cur_clean.starts_with(emitted.as_str()) {
                    let mut common = emitted
                        .as_bytes()
                        .iter()
                        .zip(cur_clean.as_bytes().iter())
                        .take_while(|(a, b)| a == b)
                        .count();
                    while common > 0 && !cur_clean.is_char_boundary(common) {
                        common -= 1;
                    }
                    emitted.clear();
                    emitted.push_str(&cur_clean[..common]);
                }
                if effective_len <= emitted.len() {
                    return;
                }
                let mut safe_end = effective_len;
                while safe_end > emitted.len() && !cur_clean.is_char_boundary(safe_end) {
                    safe_end -= 1;
                }
                if safe_end > emitted.len() {
                    use std::io::Write;
                    print!("{}", &cur_clean[emitted.len()..safe_end]);
                    std::io::stdout().flush().ok();
                    emitted.clear();
                    emitted.push_str(&cur_clean[..safe_end]);
                }
            };
            // Emit the first generated token (predicted right after prefill).
            let initial = tok.decode(&generated);
            let initial_clean = initial
                .replace("<|im_end|>", "")
                .replace("<|im_start|>", "")
                .replace("<|endoftext|>", "");
            let mut emitted = String::new();
            emit_safe(&initial_clean, &mut emitted);

            let t_decode = Instant::now();
            let mut hit_stop = stops.contains(&last);
            let mut n_decoded = 1_usize;
            while n_decoded < max_decode && !hit_stop {
                if cur_pos + 1 >= max_seq {
                    println!("\n[hit max_seq during decode]");
                    break;
                }
                match forward_token(backend, &file, &model, &mut state, last, cur_pos) {
                    Ok(out) => {
                        last = out;
                        generated.push(last);
                        cur_pos += 1;
                        n_decoded += 1;
                        // Emit incremental delta (UTF-8 safe).
                        let cur = tok.decode(&generated);
                        let cur_clean = cur
                            .replace("<|im_end|>", "")
                            .replace("<|im_start|>", "")
                            .replace("<|endoftext|>", "");
                        emit_safe(&cur_clean, &mut emitted);
                        if stops.contains(&last) {
                            hit_stop = true;
                        }
                    },
                    Err(e) => {
                        eprintln!("\nforward error at chat decode: {e}");
                        return ExitCode::FAILURE;
                    },
                }
            }
            // Final flush in case the last partial char now has its full
            // bytes available.
            let final_cur = tok.decode(&generated);
            let final_clean = final_cur
                .replace("<|im_end|>", "")
                .replace("<|im_start|>", "")
                .replace("<|endoftext|>", "");
            emit_safe(&final_clean, &mut emitted);
            let dt_decode = t_decode.elapsed();
            println!(); // newline after streamed answer
            println!(
                "  [prefill {} tok in {:.2}s ({:.1} t/s) | decode {} tok in {:.2}s ({:.1} t/s){}]",
                chunk_ids.len(),
                dt_prefill.as_secs_f64(),
                chunk_ids.len() as f64 / dt_prefill.as_secs_f64().max(1e-9),
                n_decoded,
                dt_decode.as_secs_f64(),
                n_decoded as f64 / dt_decode.as_secs_f64().max(1e-9),
                if hit_stop { " stop" } else { "" }
            );
            turn += 1;
        }
        return ExitCode::SUCCESS;
    }

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
        let im_start_id = tok.special_id("<|im_start|>").unwrap_or(tok.eos_id);
        // T185-fix: also stop on <|im_start|> (next-turn header). Without
        // this, the model occasionally outputs `<|im_start|>user\n...` after
        // its answer and the "user\n..." text leaks past the special-token
        // strip into the displayed output.
        let stops: [u32; 3] = [tok.eos_id, im_end_id, im_start_id];
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
        // T162 phase 9d : si --prefill-batch B est set et B > 1, utilise
        // forward_batch en chunks. Sinon, legacy per-token forward_token.
        if prefill_batch > 1 {
            // Detect MoE-only model — phase 9d falls back to per-token loop on
            // MoE FFN layers, which adds memcpy/drain overhead. Warn user (9f
            // will bring batched MoE).
            let any_moe = model.layers.iter().any(|l| {
                matches!(
                    l,
                    LayerMetal::Attn {
                        ffn: FfnLayerMetal::Moe { .. },
                        ..
                    } | LayerMetal::Ssm {
                        ffn: FfnLayerMetal::Moe { .. },
                        ..
                    }
                )
            });
            let any_dense = model.layers.iter().any(|l| {
                matches!(
                    l,
                    LayerMetal::Attn {
                        ffn: FfnLayerMetal::Dense { .. },
                        ..
                    } | LayerMetal::Ssm {
                        ffn: FfnLayerMetal::Dense { .. },
                        ..
                    }
                )
            });
            // T196 — Inverse du défaut. Le commit `6496bbd` (T162 phase 9f-route)
            // avait ajouté `force_pertoken = any_moe && !any_dense` parce que le
            // batched MoE régressait à 23 t/s sur 35B-A3B B=16.
            // **T175 (commits `bc1dae2` Q8_0 SGEMM batched + `f1ef15d` SSM scan
            // drain elimination) a corrigé ce path** : le batched fait maintenant
            // 232 t/s à B=32, 305 t/s à B=64, **347 t/s à B=128** sur 35B-A3B
            // (Q4_K_M, M4 Max), soit ×8.2 vs le path per-token (42 t/s).
            //
            // Auto-route batched par défaut sur MoE-only post-T175. Opt-out via
            // `RUSTORCH_MOE_BATCHED=0` si une régression future émerge.
            let env_moe_batched = std::env::var("RUSTORCH_MOE_BATCHED").ok();
            let force_pertoken = (any_moe && !any_dense) && env_moe_batched.as_deref() == Some("0");
            if force_pertoken {
                println!(
                    "  NOTE : modèle MoE-only — opt-out per-token forcé via \
                     RUSTORCH_MOE_BATCHED=0 (slow ~42 t/s ; default batched ~347 t/s)."
                );
            } else {
                println!(
                    "  T162 phase 9d : prefill batched (B={prefill_batch}, B_MAX={B_MAX_BATCH})"
                );
                if any_moe && !any_dense {
                    println!(
                        "  T162 phase 9f-bis : MoE batched (gather_per_token kernel, sans memcpy)"
                    );
                } else if any_moe {
                    println!("  NOTE : hybride avec layers MoE — gain partiel sur layers Dense.");
                }
            }
            if force_pertoken {
                // MoE-only auto-route per-token forward.
                for &t in &prompt_ids {
                    match forward_token(backend, &file, &model, &mut state, t, cur_pos) {
                        Ok(out) => last = out,
                        Err(e) => {
                            eprintln!("forward error at prefill (per-token): {e}");
                            return ExitCode::FAILURE;
                        },
                    }
                    cur_pos += 1;
                }
            } else {
                let batch_scratch = BatchScratch::new(backend, &cfg);
                let mut idx = 0;
                while idx < prompt_ids.len() {
                    let remaining = prompt_ids.len() - idx;
                    let chunk_size = if remaining >= prefill_batch {
                        prefill_batch
                    } else if remaining >= 8 {
                        (remaining / 8) * 8
                    } else {
                        for &t in &prompt_ids[idx..] {
                            match forward_token(backend, &file, &model, &mut state, t, cur_pos) {
                                Ok(out) => last = out,
                                Err(e) => {
                                    eprintln!("forward error at prefill tail: {e}");
                                    return ExitCode::FAILURE;
                                },
                            }
                            cur_pos += 1;
                        }
                        break;
                    };
                    let chunk = &prompt_ids[idx..idx + chunk_size];
                    match forward_batch(
                        backend,
                        &file,
                        &model,
                        &mut state,
                        &batch_scratch,
                        chunk,
                        cur_pos,
                    ) {
                        Ok(out) => last = out,
                        Err(e) => {
                            eprintln!("forward_batch error at prefill: {e}");
                            return ExitCode::FAILURE;
                        },
                    }
                    cur_pos += chunk_size;
                    idx += chunk_size;
                }
            }
        } else {
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
        let mut spec_drafts = 0usize;
        let mut spec_accepted = 0usize;
        let mut spec_rounds = 0usize;
        // T184 — when --stream is set, print tokens as they decode by re-decoding
        // the whole generated[1..] sequence and emitting the new suffix. This
        // handles BPE merges + special tokens cleanly (no partial UTF-8).
        let mut stream_emitted = String::new();
        if stream {
            use std::io::Write;
            print!("\n=== Streaming ===\n");
            std::io::stdout().flush().ok();
            // First-decoded token (the one returned at end of prefill).
            stream_emitted = tok.decode(&generated);
            print!("{}", stream_emitted);
            std::io::stdout().flush().ok();
        }
        // Helper closure: print new tokens incrementally when streaming.
        // Strips ChatML special tokens and slices on UTF-8 char boundaries
        // so we don't split multi-byte chars (e.g. emoji) mid-byte.
        let emit_stream = |stream_emitted: &mut String, generated: &[u32]| {
            if !stream {
                return;
            }
            use std::io::Write;
            let mut cur = tok.decode(generated);
            for special in [
                "<|im_end|>",
                "<|im_start|>",
                "<|endoftext|>",
                "<|startoftext|>",
            ] {
                cur = cur.replace(special, "");
            }
            // T185-fix: trim trailing U+FFFD (`�`) placeholders so we don't
            // emit transient chars that will be replaced by real ones.
            let mut effective_len = cur.len();
            let fffd_bytes: [u8; 3] = [0xEF, 0xBF, 0xBD];
            while effective_len >= 3
                && cur.as_bytes()[effective_len - 3..effective_len] == fffd_bytes
            {
                effective_len -= 3;
            }
            // Defensive: when stream_emitted no longer matches cur prefix
            // (tokenizer's `�` placeholder got replaced by a real char as
            // more tokens arrived), walk back to the largest common boundary.
            if !cur.starts_with(stream_emitted.as_str()) {
                let mut common = stream_emitted
                    .as_bytes()
                    .iter()
                    .zip(cur.as_bytes().iter())
                    .take_while(|(a, b)| a == b)
                    .count();
                while common > 0 && !cur.is_char_boundary(common) {
                    common -= 1;
                }
                stream_emitted.clear();
                stream_emitted.push_str(&cur[..common]);
            }
            if effective_len <= stream_emitted.len() {
                return;
            }
            let mut safe_end = effective_len;
            while safe_end > stream_emitted.len() && !cur.is_char_boundary(safe_end) {
                safe_end -= 1;
            }
            if safe_end > stream_emitted.len() {
                print!("{}", &cur[stream_emitted.len()..safe_end]);
                std::io::stdout().flush().ok();
                stream_emitted.clear();
                stream_emitted.push_str(&cur[..safe_end]);
            }
        };
        if (2..=32).contains(&speculative_b) && speculative_b % 8 == 0 {
            // T176 — Speculative decoding with 2-gram lookahead cache (port T167).
            //
            // 1. Build cache (prev2, prev1) → continuation [c0, c1, ..., c_{K-1}]
            //    seeded from prompt_ids + first decode token.
            // 2. Each round : if cache hit, build candidates = [last] + cont (size B),
            //    forward_batch_argmax(all=true) → B argmax. Walk longest prefix
            //    where outs[i] == candidates[i+1]. Commit accepted + 1 bonus.
            // 3. Update cache from accepted sequence.
            //
            // Dynamic abort : if rolling acceptance < BREAK_EVEN, fall back to
            // forward_token for that round (never regress baseline).
            use std::collections::HashMap;
            let b_total = speculative_b;
            let k_draft = b_total - 1;
            // T176b — Dual-cache (3-gram primary, 2-gram fallback) for better
            // acceptance rate on natural text. 3-gram is more specific (lower
            // hit rate but higher precision when hit), 2-gram is the fallback
            // when 3-gram misses.
            let mut ngram3: HashMap<(u32, u32, u32), Vec<u32>> = HashMap::new();
            let mut ngram2: HashMap<(u32, u32), Vec<u32>> = HashMap::new();
            let update_caches = |ngram3: &mut HashMap<(u32, u32, u32), Vec<u32>>,
                                 ngram2: &mut HashMap<(u32, u32), Vec<u32>>,
                                 history: &[u32],
                                 k_draft: usize| {
                if history.len() < 3 {
                    return;
                }
                // 3-gram pass : need at least 3 history tokens before pos i+1.
                if history.len() >= 4 {
                    for i in 2..history.len() - 1 {
                        let key3 = (history[i - 2], history[i - 1], history[i]);
                        let end = (i + 1 + k_draft).min(history.len());
                        let cont: Vec<u32> = history[i + 1..end].to_vec();
                        if !cont.is_empty() {
                            ngram3.insert(key3, cont);
                        }
                    }
                }
                // 2-gram fallback pass.
                for i in 1..history.len() - 1 {
                    let key2 = (history[i - 1], history[i]);
                    let end = (i + 1 + k_draft).min(history.len());
                    let cont: Vec<u32> = history[i + 1..end].to_vec();
                    if !cont.is_empty() {
                        ngram2.insert(key2, cont);
                    }
                }
            };
            update_caches(&mut ngram3, &mut ngram2, &prompt_ids, k_draft);
            // Seed last + first generated token.
            let mut seed: Vec<u32> = Vec::with_capacity(prompt_ids.len() + 1);
            seed.extend_from_slice(&prompt_ids);
            seed.push(last);
            update_caches(&mut ngram3, &mut ngram2, &seed, k_draft);

            let mut prev_token: u32 = prompt_ids.last().copied().unwrap_or(last);
            let mut prev_token2: u32 = if prompt_ids.len() >= 2 {
                prompt_ids[prompt_ids.len() - 2]
            } else {
                prev_token
            };

            const ABORT_WINDOW: usize = 8;
            // T176b — Break-even depends on B : forward_batch B overhead ≈ alpha + beta*B
            // vs forward_token ≈ alpha + beta. To gain, accepted tokens > overhead.
            // Empirically : need ~1.5 accepted tokens minimum to break even at B=8,
            //              ~2 at B=16, ~3 at B=24, ~4 at B=32.
            // → BREAK_EVEN_PCT = 100 * (1.5 + 0.05*B) / k_draft for safety margin.
            let break_even_pct: f32 = 100.0 * (1.5 + 0.05 * b_total as f32) / k_draft as f32;
            let mut recent_accept: std::collections::VecDeque<u32> =
                std::collections::VecDeque::with_capacity(ABORT_WINDOW);

            let batch_scratch_dec = BatchScratch::new(backend, &cfg);

            while generated.len() < n {
                if cur_pos + b_total >= max_seq {
                    break;
                }
                // T176b — Try 3-gram first (more specific = higher acceptance),
                // fall back to 2-gram if miss.
                let key3 = (prev_token2, prev_token, last);
                let key2 = (prev_token, last);
                let cont3 = ngram3.get(&key3);
                let cont2 = ngram2.get(&key2);
                let chosen_cont = cont3.or(cont2);
                let cache_hit = chosen_cont.map_or(0, |v| v.len()) > 0;
                let window_acc = if recent_accept.is_empty() {
                    100.0
                } else {
                    let total: u32 = recent_accept.iter().sum();
                    100.0 * (total as f32) / (recent_accept.len() as f32 * (k_draft as f32))
                };
                let should_spec = cache_hit && window_acc >= break_even_pct;

                if !should_spec {
                    match forward_token(backend, &file, &model, &mut state, last, cur_pos) {
                        Ok(out) => {
                            prev_token2 = prev_token;
                            prev_token = last;
                            last = out;
                            generated.push(last);
                            cur_pos += 1;
                            emit_stream(&mut stream_emitted, &generated);
                            // T176 fix : update caches from generated history even
                            // on fallback steps.
                            if generated.len() >= 3 {
                                let hist_start = generated.len().saturating_sub(8);
                                update_caches(
                                    &mut ngram3,
                                    &mut ngram2,
                                    &generated[hist_start..],
                                    k_draft,
                                );
                            }
                            if stops.contains(&last) {
                                hit_stop = true;
                                break;
                            }
                        },
                        Err(e) => {
                            eprintln!("forward error at decode (spec fallback): {e}");
                            return ExitCode::FAILURE;
                        },
                    }
                    continue;
                }

                // Build candidates [last, c0, c1, ..., c_{K-1}].
                let mut candidates: Vec<u32> = Vec::with_capacity(b_total);
                candidates.push(last);
                if let Some(cont) = chosen_cont {
                    for &t in cont.iter().take(k_draft) {
                        candidates.push(t);
                    }
                }
                while candidates.len() < b_total {
                    candidates.push(last);
                }

                let outs = match forward_batch_argmax(
                    backend,
                    &file,
                    &model,
                    &mut state,
                    &batch_scratch_dec,
                    &candidates,
                    cur_pos,
                    true,
                ) {
                    Ok(v) => v,
                    Err(e) => {
                        eprintln!("forward error at decode (speculative): {e}");
                        return ExitCode::FAILURE;
                    },
                };
                spec_rounds += 1;
                spec_drafts += k_draft;

                // Walk longest prefix accepted.
                let mut accepted = 0usize;
                for i in 0..k_draft {
                    if outs[i] == candidates[i + 1] {
                        accepted += 1;
                    } else {
                        break;
                    }
                }
                spec_accepted += accepted;
                if recent_accept.len() == ABORT_WINDOW {
                    recent_accept.pop_front();
                }
                recent_accept.push_back(accepted as u32);

                // Commit accepted drafts.
                let mut local_stop = false;
                for i in 0..accepted {
                    let t = candidates[i + 1];
                    generated.push(t);
                    if generated.len() >= n {
                        break;
                    }
                    if stops.contains(&t) {
                        local_stop = true;
                        hit_stop = true;
                        break;
                    }
                }
                if local_stop || generated.len() >= n {
                    break;
                }
                // Bonus token.
                let bonus = outs[accepted];
                generated.push(bonus);
                emit_stream(&mut stream_emitted, &generated);
                if generated.len() >= 3 {
                    prev_token2 = generated[generated.len() - 3];
                    prev_token = generated[generated.len() - 2];
                } else if generated.len() >= 2 {
                    prev_token = generated[generated.len() - 2];
                }
                last = bonus;
                cur_pos += accepted + 1;
                if stops.contains(&last) {
                    hit_stop = true;
                    break;
                }

                // Update caches from recent committed sequence.
                let history_start = generated.len().saturating_sub(accepted + 4);
                let hist_window: Vec<u32> = if history_start >= 3 {
                    generated[history_start - 3..].to_vec()
                } else {
                    generated.clone()
                };
                update_caches(&mut ngram3, &mut ngram2, &hist_window, k_draft);
            }
        } else {
            for _ in 1..n {
                if cur_pos + 1 >= max_seq {
                    break;
                }
                match forward_token(backend, &file, &model, &mut state, last, cur_pos) {
                    Ok(out) => {
                        last = out;
                        generated.push(last);
                        cur_pos += 1;
                        emit_stream(&mut stream_emitted, &generated);
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
        }
        let decode_d = t0.elapsed();
        let n_decoded = generated.len().saturating_sub(1) as f64;
        let spec_info = if speculative_b > 0 && spec_rounds > 0 {
            let acc_pct = 100.0 * (spec_accepted as f32) / (spec_drafts.max(1) as f32);
            format!(
                " [spec={speculative_b} rounds={spec_rounds} acc={spec_accepted}/{spec_drafts} ({acc_pct:.0}%)]"
            )
        } else {
            String::new()
        };
        if stream {
            // Newline so the `decode :` stat starts on its own line, not
            // appended to the streamed text.
            println!();
        }
        println!(
            "  decode : {} tok in {:.3}s ({:.2} tok/s){}{}",
            n_decoded as usize,
            decode_d.as_secs_f64(),
            n_decoded / decode_d.as_secs_f64().max(1e-9),
            if hit_stop { " [STOP]" } else { "" },
            spec_info
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
        if stream {
            // Already printed incrementally — just close the streaming block
            // and avoid re-dumping the full answer. Print a newline + marker
            // for clarity.
            println!("\n=== End ===");
        } else {
            println!("\n=== Answer ===");
            println!("{}", answer);
            println!("===");
        }
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
