//! T246 — `Qwen35ModelCudaQ4K` : full e2e CUDA forward path for Qwen3.5/3.6
//! hybrid (SSM + Attention) Q4_K_M GGUF models.
//!
//! Architecture vs `LlamaModelCudaQ4K` (Qwen2.5) :
//! - Same Q4_K / Q5_K / Q6_K matmul weights stored as raw bytes on GPU.
//! - Hybrid : 16/64 layers are full attention, 48/64 are SSM (gated delta net).
//! - Each block holds a different set of weights depending on its kind.
//!
//! Performance target on Qwen3.6-27B Q4_K_M : >= 11.73 tok/s (llama.cpp tg128
//! parity) with coherent text output, then push past via batched M=N forward.
//!
//! Status (T246.1) : LOADER ONLY. decode_step still returns Err — implemented
//! incrementally in T246.2 (SSM block), T246.3 (attn block), T246.4 (FFN + sample),
//! T246.5 (prefill).

#![cfg(feature = "cuda")]

use crate::qwen35::{LayerKind, Qwen35Config, Qwen35Variant};
use crate::LlmError;
use cudarc::driver::sys::{CUgraphInstantiate_flags, CUstreamCaptureMode};

/// T246.5.7 — number of kv-len splits in the FlashDecode-V2 GQA kernel.
/// 4 gives a 4× boost in attention-block parallelism (32 heads × 4 splits =
/// 128 blocks/layer vs 32 with the serial online kernel) and is fixed at
/// capture time so the staging buffers can be sized once.
const GQA_N_SPLIT: usize = 4;

/// T246.7 P1.3 — maximum draft tree size accepted by `decode_step_tree`.
/// Sized to give headroom over the (W=5, L=5) Jacobi window which yields
/// at most 21 tree nodes (1 + W*(L-1)). 32 leaves room for (W=5, L=7) and
/// future tweaks without re-allocating the tree scratch buffers.
/// T246.10 A6 — bumped from 32 to 512 to enable batched prefill via the
/// existing tree-attention infrastructure with a linear-chain topology.
///
/// Memory impact at 512 (decision 25db464b):
/// - Pure-transformer per-tree scratch: ~250 MB (mostly `tree_logits` =
///   `MAX × vocab × 2B` ≈ 152 MB at vocab=151936).
/// - Hybrid (Qwen3.6 Dense/MoE) adds SSM tree-state buffers ~ MAX × 32 × 64
///   × 64 × 2B per layer ≈ 8.4 MB × 48 layers = ~400 MB at MAX=32 → ~6.4 GB
///   at MAX=512.
///
/// Acceptable on the GB10 256 GB unified pool. Linear-chain prefill
/// degenerates the SSM tree forking to one BFS depth wave per row (sequential
/// SSM is correct, slow per launch but at N=512 the dominant cost shifts to
/// the attention/FFN matmul fan-out, which is where the prefill speedup
/// lives).
pub const MAX_TREE_SIZE: usize = 512;

/// T246.8 A5 — number of K-direction chunks used by the split-K lm_head
/// kernel when enabled (`RUSTORCH_LM_HEAD_SPLIT=1`). For Qwen3.6-35B-A3B
/// the lm_head shape is (N=152064, K=2048) → blocks_per_row=8 → one
/// super-block per chunk. Smaller K's (or non-multiples) fall back to V2.
/// Sized as `_MAX` because the scratch buffer is pre-allocated for the
/// worst case at model load.
const LM_HEAD_K_CHUNKS_MAX: usize = 8;
use cudarc::driver::{CudaContext, CudaGraph, CudaSlice, CudaStream, PinnedHostSlice};
use rustorch_cuda::cublas_lt::LtSession;
use rustorch_cuda::llm_kernels::LlmKernels;
use rustorch_gguf::reader::GgufFile;
use rustorch_gguf::tensor::GgmlType;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

/// T246.8 A2.3 — runtime gate for the async (zero-host-sync) MoE FFN
/// path. Default OFF for safety until parity is checked. Set
/// `RUSTORCH_MOE_ASYNC=1` to enable. When enabled, the per-layer MoE
/// dispatch reads top-K indices/weights directly from device memory
/// (via the indexed kernels, T246.8 A2.1) and never blocks the stream.
fn moe_async_enabled() -> bool {
    std::env::var("RUSTORCH_MOE_ASYNC")
        .ok()
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

/// T246.8 A2.4 — runtime gate for re-enabling CUDA Graph capture for
/// the MoE variant. Only valid when `RUSTORCH_MOE_ASYNC=1` (otherwise
/// the per-layer host syncs invalidate capture). Default ON when
/// async is enabled, OFF otherwise.
fn moe_graph_enabled() -> bool {
    if !moe_async_enabled() {
        return false;
    }
    std::env::var("RUSTORCH_MOE_GRAPH")
        .ok()
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(true)
}

/// T246.8 A4 — runtime gate for the mul_mm_id mega-kernel MoE path.
/// Only valid when `RUSTORCH_MOE_ASYNC=1` (the mega-kernels are an
/// optimisation OF the async/zero-host-sync path). Default OFF — flip
/// to `RUSTORCH_MOE_MEGA=1` to fuse the K=8 per-expert SGEMV launches
/// of gate/up into single kernels, plus a fused routed-reduce epilogue.
fn moe_mega_enabled() -> bool {
    if !moe_async_enabled() {
        return false;
    }
    std::env::var("RUSTORCH_MOE_MEGA")
        .ok()
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

/// T246.10 A6b.3 — runtime gate for the batched GEMM prefill dispatch.
/// Default OFF for safety. When set to 1, prefill paths (force_accept_all
/// = true) with tree_size >= GEMM_PREFILL_MIN_M replace their per-row
/// dispatch_matmul_m1 loops on Q/K/V/O/gate/up/down/SSM-in/SSM-out with a
/// single dispatch_matmul_mvar call.
fn gemm_prefill_enabled() -> bool {
    std::env::var("RUSTORCH_GEMM_PREFILL")
        .ok()
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

/// Minimum M tile size below which the batched GEMM path is NOT used.
/// Decode (M=1) and small trees stay on the SGEMV path (A2/A4 wins).
/// At M >= 8 the M=8 / mvar path is bandwidth-positive (T245.4).
const GEMM_PREFILL_MIN_M: usize = 8;

/// T246.10 TrackE.2 — runtime gate for the batched MoE Group-GEMM
/// dispatch. Default OFF for safety. Only valid when both `RUSTORCH_MOE_MEGA`
/// (or async) AND `RUSTORCH_GEMM_PREFILL` are also set : we replace the
/// per-row `moe_ffn_forward_step_mega` loop with a single Group-GEMM
/// pass over all M tokens for the gate / up / down projections.
///
/// Requires :
///   - tree_size >= GROUP_GEMM_MIN_M (default = GEMM_PREFILL_MIN_M, 8)
///   - force_accept_all = true (prefill mode)
///
/// Decode (tree_size = 1) is NEVER routed through this path.
fn moe_group_gemm_enabled() -> bool {
    std::env::var("RUSTORCH_MOE_GROUP_GEMM")
        .ok()
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

/// Minimum M batch below which we keep the per-token MoE mega loop.
/// At M >= 8 the per-block W amortization across L2 cache becomes
/// significant and the launch-count reduction dominates.
const GROUP_GEMM_MIN_M: usize = 8;

/// T246.10 TrackE.3 / TrackE.4 — runtime gate for the sort-permutation
/// Group-GEMM path. Default OFF for safety. Only valid when
/// `RUSTORCH_MOE_GROUP_GEMM=1` is also set : we then build the compact
/// permutation arrays via `mm_ids_helper_bf16` and dispatch the gate /
/// up matmuls through the matching sorted kernel — Q4_K via
/// `mul_mm_id_gemm_q4_k_sorted_bf16` (TrackE.3) and Q5_K via
/// `mul_mm_id_gemm_q5_k_sorted_bf16` (TrackE.4) — so adjacent compact
/// blocks share L1/L2 weight tiles.
fn moe_group_gemm_sorted_enabled() -> bool {
    std::env::var("RUSTORCH_MOE_GROUP_GEMM_SORTED")
        .ok()
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

/// T246.10 TrackI — runtime gate for CUDA Graph capture on the prefill
/// path (`prefill_tokens` linear-chain forward). Default OFF for safety.
///
/// When set to `1` (or `true`), the second call to `prefill_tokens` with
/// a given N captures the full forward body into a `cudaGraph_t` keyed by
/// N. Every subsequent call with the same N replays the captured graph
/// with a single `cuGraphLaunch`, eliminating ~99 % of `cuLaunchKernel`
/// host overhead (PRE-FLIGHT measured ~1.23 M launches / pp512 →
/// ~10 s of host time, vs llama.cpp's ~2 627 launches via the same
/// graph-capture optimization).
///
/// Requires the same env preconditions as prefill GEMM batching :
/// `RUSTORCH_MOE_ASYNC=1`, `RUSTORCH_MOE_GRAPH=1`, `RUSTORCH_MOE_MEGA=1`,
/// `RUSTORCH_GEMM_PREFILL=1` (and recommended
/// `RUSTORCH_MOE_GROUP_GEMM=1` `RUSTORCH_MOE_GROUP_GEMM_SORTED=1`). The
/// body must contain ZERO host-syncs / `memcpy_dtov` / pageable HtoD
/// during replay — the wrapper handles tree-descriptor uploads and the
/// final argmax DtoH explicitly outside the capture region, and the SSM
/// per-wave HtoD chain is replaced by offsets into a pre-baked linear
/// wave-indices buffer (linear chains have wave[d] = [d]).
fn prefill_graph_enabled() -> bool {
    std::env::var("RUSTORCH_PREFILL_GRAPH")
        .ok()
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

/// T246.10 TrackG-lite — runtime gate for the BF16xBF16 mma.sync m16n8k16
/// tensor-core GEMM path. Default OFF for safety.
///
/// When set, the `QuantTensor::Bf16` arm of `dispatch_matmul_mvar` routes
/// M >= GEMM_BF16_MMA_MIN_M (16) to `gemm_bf16_bf16_mma` (mma.sync) instead
/// of the warp-shuffle `sgemm_bf16_bf16_mvar`. Below the threshold the
/// warp-shuffle path is kept (A3 negative result : mma at M=1 loses to
/// warp-shuffle — see note 8e23a850).
///
/// Targets the attention QKV/O matmuls and SSM in_proj/out_proj which
/// currently dominate per-pp512 GPU compute (sgemm_bf16_bf16_mvar = 27%
/// per TrackE.4 nsys).
fn gemm_bf16_mma_enabled() -> bool {
    std::env::var("RUSTORCH_GEMM_BF16_MMA")
        .ok()
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

/// T246.10 TrackF — env gate for the column-parallel delta_net_step_tree_bf16
/// variant. Default OFF preserves bit-exact parity with the baseline kernel ;
/// set RUSTORCH_DELTA_NET_OPT=1 to switch to the coalesced-state variant
/// (target : -50 % kernel time on the SSM block, +10 % pp512 wall-clock).
fn delta_net_opt_enabled() -> bool {
    std::env::var("RUSTORCH_DELTA_NET_OPT")
        .ok()
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

/// Minimum M below which the BF16 mma path is NOT used. mma.sync m16n8k16
/// needs M >= 16 to fill its 16-row tile ; smaller M wastes 87.5%+ of
/// tensor-core compute on broadcast (A3 negative result confirmed).
pub(crate) const GEMM_BF16_MMA_MIN_M: usize = 16;

/// T246.10 MMQ-WHOLESALE — env gate for the Q8_1-packed INT8-staged Q4_K
/// matmul path (kernels landed in 4206d2d). Default OFF preserves bit-exact
/// parity with TrackG-lite. Set `RUSTORCH_MMQ_WHOLESALE=1` to route the
/// batched shared-expert Q4_K matmuls through `quantize_mmq_q8_1_bf16_ds4` +
/// `mul_mat_q4_k_q8_1_mma` (Phase 6a — Option A : shared expert only).
fn mmq_wholesale_enabled() -> bool {
    std::env::var("RUSTORCH_MMQ_WHOLESALE")
        .ok()
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

/// Minimum M below which the MMQ-WHOLESALE path is NOT used. The kernel
/// instantiation is mmq_x=64, so M < 64 wastes tile compute. Below this
/// threshold we fall back to the existing per-token shared-expert loop.
pub(crate) const MMQ_WHOLESALE_MIN_M: i32 = 64;

/// One matmul weight on GPU. Qwen3.6 Q4_K_M uses a mix of Q4_K / Q5_K /
/// Q6_K for big matmuls and F32 (→ BF16) for small ones (ssm_alpha, ssm_beta
/// when n_v_heads is small : n=48 typical, K=hidden_size, ≈1 MB each).
pub(crate) enum QuantTensor {
    Bf16 {
        weights: CudaSlice<half::bf16>,
        n: usize,
        k: usize,
    },
    Q4K {
        bytes: CudaSlice<u8>,
        n: usize, // out dim (rows)
        k: usize, // in dim (cols)
    },
    Q5K {
        bytes: CudaSlice<u8>,
        n: usize,
        k: usize,
    },
    Q6K {
        bytes: CudaSlice<u8>,
        n: usize,
        k: usize,
    },
}

impl QuantTensor {
    /// Dispatch a M=1 GEMV through the appropriate kernel.
    ///
    /// `x_q8_staging` is a device pointer to a scratch buffer of at least
    /// `(K/32) * 36` bytes used by the Q4_K dp4a path (T246.5.5) for the
    /// on-the-fly Q8_1 quantization of the activation. Pass 0 to disable
    /// dp4a and force the float v2 fallback. The buffer is overwritten
    /// each call so the same one can be reused across all matmuls.
    pub(crate) fn dispatch_matmul_m1(
        &self,
        kernels: &LlmKernels,
        stream: &Arc<CudaStream>,
        x: u64,
        y: u64,
        x_q8_staging: u64,
    ) -> Result<(), LlmError> {
        unsafe {
            use cudarc::driver::DevicePtr;
            match self {
                QuantTensor::Bf16 { weights, n, k } => {
                    let (w, _g) = weights.device_ptr(stream);
                    // T246.8 A3 — dispatch routes V2 (mma.sync m16n8k16
                    // tensor-core SGEMV) for N >= 128 + K%16==0 ; V1 for
                    // small N or unsupported K.
                    kernels
                        .sgemv_bf16_bf16_dispatch(stream, w, x, y, *n as i32, *k as i32)
                        .map_err(|e| LlmError::Backend(format!("sgemv_bf16_dispatch: {e:?}")))
                },
                QuantTensor::Q4K { bytes, n, k } => {
                    let (w, _g) = bytes.device_ptr(stream);
                    if x_q8_staging != 0 {
                        // T246.5.5 — dp4a path: quantize x → Q8_1, then
                        // Q4_K × Q8_1 SGEMV using __dp4a (mirror of
                        // llama.cpp vec_dot_q4_K_q8_1_impl_vmmq).
                        kernels
                            .quantize_q8_1_bf16(stream, x, x_q8_staging, *k as i32)
                            .map_err(|e| LlmError::Backend(format!("quant q8_1: {e:?}")))?;
                        kernels
                            .sgemv_q4k_q8_1_dp4a_bf16(
                                stream,
                                w,
                                x_q8_staging,
                                y,
                                *n as i32,
                                *k as i32,
                            )
                            .map_err(|e| LlmError::Backend(format!("sgemv_q4k_dp4a: {e:?}")))
                    } else {
                        // T246.6.7 — V3 (4-row-per-block, per-thread scale,
                        // __launch_bounds__) targets 60-70% peak BW vs V2's
                        // ~38%. Parity-tested vs V2.
                        kernels
                            .sgemv_q4k_bf16_v3(stream, w, x, y, *n as i32, *k as i32)
                            .map_err(|e| LlmError::Backend(format!("sgemv_q4k_v3: {e:?}")))
                    }
                },
                QuantTensor::Q5K { bytes, n, k } => {
                    let (w, _g) = bytes.device_ptr(stream);
                    // T246.6.9 — Heuristic: V3 (multi-row) when N is large
                    // enough to saturate SMs ; V2 for small SSM α/β matmuls
                    // where V3's 4-row block underutilizes occupancy.
                    if *n >= 256 {
                        kernels
                            .sgemv_q5k_bf16_v3(stream, w, x, y, *n as i32, *k as i32)
                            .map_err(|e| LlmError::Backend(format!("sgemv_q5k_v3: {e:?}")))
                    } else {
                        kernels
                            .sgemv_q5k_bf16(stream, w, x, y, *n as i32, *k as i32)
                            .map_err(|e| LlmError::Backend(format!("sgemv_q5k: {e:?}")))
                    }
                },
                QuantTensor::Q6K { bytes, n, k } => {
                    let (w, _g) = bytes.device_ptr(stream);
                    kernels
                        // T246.6.8 — V4 measured equal to V2 (no win) on the
                        // LM head shape (N=152064 × K=5120). V4 kernel kept
                        // for future investigation. Sticking with V2.
                        .sgemv_q6k_bf16_v2(stream, w, x, y, *n as i32, *k as i32)
                        .map_err(|e| LlmError::Backend(format!("sgemv_q6k_v2: {e:?}")))
                },
            }
        }
    }

    /// Dispatch a M=8 batched GEMM through the appropriate kernel.
    /// Used by prefill (T246.5) and speculative decoding.
    #[allow(dead_code)]
    pub(crate) fn dispatch_matmul_m8(
        &self,
        kernels: &LlmKernels,
        stream: &Arc<CudaStream>,
        x: u64,
        y: u64,
    ) -> Result<(), LlmError> {
        unsafe {
            use cudarc::driver::DevicePtr;
            match self {
                QuantTensor::Bf16 { .. } => Err(LlmError::Backend(
                    "BF16 M=8 batched not yet implemented (T246.5)".into(),
                )),
                QuantTensor::Q4K { bytes, n, k } => {
                    let (w, _g) = bytes.device_ptr(stream);
                    kernels
                        .sgemm_q4k_bf16_m8(stream, w, x, y, *n as i32, *k as i32)
                        .map_err(|e| LlmError::Backend(format!("sgemm_q4k_m8: {e:?}")))
                },
                QuantTensor::Q5K { bytes, n, k } => {
                    let (w, _g) = bytes.device_ptr(stream);
                    kernels
                        .sgemm_q5k_bf16_m8(stream, w, x, y, *n as i32, *k as i32)
                        .map_err(|e| LlmError::Backend(format!("sgemm_q5k_m8: {e:?}")))
                },
                QuantTensor::Q6K { bytes, n, k } => {
                    let (w, _g) = bytes.device_ptr(stream);
                    kernels
                        .sgemm_q6k_bf16_m8(stream, w, x, y, *n as i32, *k as i32)
                        .map_err(|e| LlmError::Backend(format!("sgemm_q6k_m8: {e:?}")))
                },
            }
        }
    }

    /// T246.10 A6b.2 — Dispatch a M-variable batched GEMM through the
    /// appropriate kernel. Used by prefill (T246.10) when tree_size >= 8.
    ///
    /// Layout :
    ///   x : [M, K] BF16 row-major
    ///   y : [M, N] BF16 row-major
    ///
    /// Kernel grid is (N, ceil(M/8)) ; one super-block W read amortized
    /// across all M m-rows (W is read 1× per (row, mtile_idx) block).
    pub(crate) fn dispatch_matmul_mvar(
        &self,
        kernels: &LlmKernels,
        stream: &Arc<CudaStream>,
        m: usize,
        x: u64,
        y: u64,
    ) -> Result<(), LlmError> {
        unsafe {
            use cudarc::driver::DevicePtr;
            match self {
                QuantTensor::Bf16 { weights, n, k } => {
                    let (w, _g) = weights.device_ptr(stream);
                    // TrackG-lite : route M >= 16 to mma.sync m16n8k16 BF16
                    // tensor-core path. M < 16 stays on warp-shuffle (A3
                    // proved mma loses at small M — note 8e23a850).
                    if gemm_bf16_mma_enabled() && m >= GEMM_BF16_MMA_MIN_M {
                        kernels
                            .gemm_bf16_bf16_mma(stream, w, x, y, m as i32, *n as i32, *k as i32)
                            .map_err(|e| LlmError::Backend(format!("gemm_bf16_mma: {e:?}")))
                    } else {
                        kernels
                            .sgemm_bf16_bf16_mvar(stream, w, x, y, m as i32, *n as i32, *k as i32)
                            .map_err(|e| LlmError::Backend(format!("sgemm_bf16_mvar: {e:?}")))
                    }
                },
                QuantTensor::Q4K { bytes, n, k } => {
                    let (w, _g) = bytes.device_ptr(stream);
                    kernels
                        .sgemm_q4k_bf16_mvar(stream, w, x, y, m as i32, *n as i32, *k as i32)
                        .map_err(|e| LlmError::Backend(format!("sgemm_q4k_mvar: {e:?}")))
                },
                QuantTensor::Q5K { bytes, n, k } => {
                    let (w, _g) = bytes.device_ptr(stream);
                    kernels
                        .sgemm_q5k_bf16_mvar(stream, w, x, y, m as i32, *n as i32, *k as i32)
                        .map_err(|e| LlmError::Backend(format!("sgemm_q5k_mvar: {e:?}")))
                },
                QuantTensor::Q6K { bytes, n, k } => {
                    let (w, _g) = bytes.device_ptr(stream);
                    kernels
                        .sgemm_q6k_bf16_mvar(stream, w, x, y, m as i32, *n as i32, *k as i32)
                        .map_err(|e| LlmError::Backend(format!("sgemm_q6k_mvar: {e:?}")))
                },
            }
        }
    }

    pub(crate) fn shape(&self) -> (usize, usize) {
        match self {
            QuantTensor::Bf16 { n, k, .. } => (*n, *k),
            QuantTensor::Q4K { n, k, .. } => (*n, *k),
            QuantTensor::Q5K { n, k, .. } => (*n, *k),
            QuantTensor::Q6K { n, k, .. } => (*n, *k),
        }
    }

    /// Return this tensor's base device pointer as a u64 (for indexed
    /// dispatch tables).
    pub(crate) fn base_ptr(&self, stream: &Arc<CudaStream>) -> u64 {
        use cudarc::driver::DevicePtr;
        match self {
            QuantTensor::Bf16 { weights, .. } => {
                let (p, _g) = weights.device_ptr(stream);
                p
            },
            QuantTensor::Q4K { bytes, .. }
            | QuantTensor::Q5K { bytes, .. }
            | QuantTensor::Q6K { bytes, .. } => {
                let (p, _g) = bytes.device_ptr(stream);
                p
            },
        }
    }
}

/// T246.8 A2 — Dispatch an indexed M=1 SGEMV through the right kernel
/// for `kind`. Each call launches one indexed kernel that reads the
/// expert weight pointer from `expert_ptrs_p[topk_indices_p[slot]]` on
/// device. Mirrors `QuantTensor::dispatch_matmul_m1`'s dispatch logic.
///
/// `n` and `k` are the per-expert output/input dims (uniform across the
/// stacked-expert tensor).
///
/// `x_q8_staging` : pass the device pointer to a `(K/32)*36`-byte Q8_1
/// staging buffer to enable the dp4a path for Q4_K experts (much faster
/// than the float V3 fallback). Pass 0 to force the float V3 path. Caller
/// must have already filled the staging buffer with `quantize_q8_1_bf16(x)`
/// before calling this dispatcher (the buffer is shared across all k
/// indexed calls in one MoE layer iteration).
#[allow(clippy::too_many_arguments)]
pub(crate) fn dispatch_indexed_matmul_m1(
    kernels: &LlmKernels,
    stream: &Arc<CudaStream>,
    kind: ExpertQuantKind,
    expert_ptrs_p: u64,
    topk_indices_p: u64,
    slot: i32,
    x: u64,
    y: u64,
    n: i32,
    k: i32,
    x_q8_staging: u64,
) -> Result<(), LlmError> {
    unsafe {
        match kind {
            ExpertQuantKind::Q4K => {
                if x_q8_staging != 0 {
                    // Mirror `QuantTensor::dispatch_matmul_m1` Q4K dp4a path :
                    // quantize x → x_q8_staging just before the kernel.
                    kernels
                        .quantize_q8_1_bf16(stream, x, x_q8_staging, k)
                        .map_err(|e| LlmError::Backend(format!("q8_1 quant idx: {e:?}")))?;
                    kernels
                        .sgemv_q4k_q8_1_dp4a_bf16_indexed(
                            stream,
                            expert_ptrs_p,
                            topk_indices_p,
                            slot,
                            x_q8_staging,
                            y,
                            n,
                            k,
                        )
                        .map_err(|e| LlmError::Backend(format!("q4k_dp4a_idx: {e:?}")))
                } else {
                    kernels
                        .sgemv_q4k_bf16_v3_indexed(
                            stream,
                            expert_ptrs_p,
                            topk_indices_p,
                            slot,
                            x,
                            y,
                            n,
                            k,
                        )
                        .map_err(|e| LlmError::Backend(format!("q4k_v3_idx: {e:?}")))
                }
            },
            ExpertQuantKind::Q5K => kernels
                .sgemv_q5k_bf16_v3_indexed(stream, expert_ptrs_p, topk_indices_p, slot, x, y, n, k)
                .map_err(|e| LlmError::Backend(format!("q5k_v3_idx: {e:?}"))),
            ExpertQuantKind::Q6K => kernels
                .sgemv_q6k_bf16_v3_indexed(stream, expert_ptrs_p, topk_indices_p, slot, x, y, n, k)
                .map_err(|e| LlmError::Backend(format!("q6k_v3_idx: {e:?}"))),
            ExpertQuantKind::Bf16 => kernels
                .sgemv_bf16_bf16_indexed(stream, expert_ptrs_p, topk_indices_p, slot, x, y, n, k)
                .map_err(|e| LlmError::Backend(format!("bf16_idx: {e:?}"))),
        }
    }
}

/// Quant family of a stacked-expert tensor — all experts in a `Vec<QuantTensor>`
/// share the same source dtype (per GGUF tensor contract). Used by the indexed
/// MoE dispatch path (T246.8 A2) to pick the right kernel without
/// per-iteration host-side dispatch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ExpertQuantKind {
    Q4K,
    Q5K,
    Q6K,
    Bf16,
}

/// T246.8 A4 — mul_mm_id mega-kernel dispatcher : single launch covers
/// all `k_used` experts. Output `y` is `[k_used, n]` BF16, slot-major.
/// Replaces the K-iteration `dispatch_indexed_matmul_m1` loop on the
/// `RUSTORCH_MOE_MEGA=1` path.
#[allow(clippy::too_many_arguments)]
pub(crate) fn dispatch_indexed_mega(
    kernels: &LlmKernels,
    stream: &Arc<CudaStream>,
    kind: ExpertQuantKind,
    expert_ptrs_p: u64,
    topk_indices_p: u64,
    x: u64,
    y: u64,
    n: i32,
    k: i32,
    k_used: i32,
    x_q8_staging: u64,
) -> Result<(), LlmError> {
    unsafe {
        match kind {
            ExpertQuantKind::Q4K => {
                if x_q8_staging != 0 {
                    kernels
                        .quantize_q8_1_bf16(stream, x, x_q8_staging, k)
                        .map_err(|e| LlmError::Backend(format!("q8_1 quant mega: {e:?}")))?;
                    kernels
                        .mul_mm_id_q4_k_q8_1_dp4a_bf16(
                            stream,
                            expert_ptrs_p,
                            topk_indices_p,
                            x_q8_staging,
                            y,
                            n,
                            k,
                            k_used,
                        )
                        .map_err(|e| LlmError::Backend(format!("mul_mm_id q4k_dp4a: {e:?}")))
                } else {
                    kernels
                        .mul_mm_id_q4_k_bf16(
                            stream,
                            expert_ptrs_p,
                            topk_indices_p,
                            x,
                            y,
                            n,
                            k,
                            k_used,
                        )
                        .map_err(|e| LlmError::Backend(format!("mul_mm_id q4k: {e:?}")))
                }
            },
            ExpertQuantKind::Q5K => kernels
                .mul_mm_id_q5_k_bf16(stream, expert_ptrs_p, topk_indices_p, x, y, n, k, k_used)
                .map_err(|e| LlmError::Backend(format!("mul_mm_id q5k: {e:?}"))),
            ExpertQuantKind::Q6K => kernels
                .mul_mm_id_q6_k_bf16(stream, expert_ptrs_p, topk_indices_p, x, y, n, k, k_used)
                .map_err(|e| LlmError::Backend(format!("mul_mm_id q6k: {e:?}"))),
            ExpertQuantKind::Bf16 => kernels
                .mul_mm_id_bf16_bf16(stream, expert_ptrs_p, topk_indices_p, x, y, n, k, k_used)
                .map_err(|e| LlmError::Backend(format!("mul_mm_id bf16: {e:?}"))),
        }
    }
}

/// T246.10 TrackE.2 — Group-GEMM (M-variable) indexed dispatcher.
///
/// Mirror of `dispatch_indexed_mega` extended to M tokens. Replaces an
/// outer `for m in 0..M { dispatch_indexed_mega(m) }` loop with a single
/// 3D-grid launch.
///
/// `topk_indices_p` is `[M, k_used]` i32 row-major device pointer.
/// `x` is `[M, K]` BF16 row-major (or `[M, (K/32)*36]` u8 row-major for
/// the Q4_K dp4a path with `x_q8_staging != 0`).
/// `y` is `[M, k_used, N]` BF16 row-major (output).
#[allow(clippy::too_many_arguments)]
pub(crate) fn dispatch_indexed_group_gemm(
    kernels: &LlmKernels,
    stream: &Arc<CudaStream>,
    kind: ExpertQuantKind,
    expert_ptrs_p: u64,
    topk_indices_p: u64,
    x: u64,
    y: u64,
    m: i32,
    n: i32,
    k: i32,
    k_used: i32,
    x_q8_m_staging: u64, // [M, (K/32)*36] device, 0 disables dp4a
) -> Result<(), LlmError> {
    unsafe {
        match kind {
            ExpertQuantKind::Q4K => {
                if x_q8_m_staging != 0 {
                    // Per-token Q8_1 quantization : iterate M (cheap kernel,
                    // amortized by the large gate/up/down GEMM that follows).
                    let bf16_sz = std::mem::size_of::<half::bf16>() as u64;
                    let q8_per_tok = ((k as usize) / 32 * 36) as u64;
                    for tok in 0..m {
                        kernels
                            .quantize_q8_1_bf16(
                                stream,
                                x + (tok as u64) * (k as u64) * bf16_sz,
                                x_q8_m_staging + (tok as u64) * q8_per_tok,
                                k,
                            )
                            .map_err(|e| LlmError::Backend(format!("q8_1 quant group: {e:?}")))?;
                    }
                    kernels
                        .mul_mm_id_gemm_q4_k_q8_1_dp4a_bf16(
                            stream,
                            expert_ptrs_p,
                            topk_indices_p,
                            x_q8_m_staging,
                            y,
                            m,
                            n,
                            k,
                            k_used,
                        )
                        .map_err(|e| LlmError::Backend(format!("mul_mm_id_gemm q4k_dp4a: {e:?}")))
                } else {
                    kernels
                        .mul_mm_id_gemm_q4_k_bf16(
                            stream,
                            expert_ptrs_p,
                            topk_indices_p,
                            x,
                            y,
                            m,
                            n,
                            k,
                            k_used,
                        )
                        .map_err(|e| LlmError::Backend(format!("mul_mm_id_gemm q4k: {e:?}")))
                }
            },
            ExpertQuantKind::Q5K => kernels
                .mul_mm_id_gemm_q5_k_bf16(
                    stream,
                    expert_ptrs_p,
                    topk_indices_p,
                    x,
                    y,
                    m,
                    n,
                    k,
                    k_used,
                )
                .map_err(|e| LlmError::Backend(format!("mul_mm_id_gemm q5k: {e:?}"))),
            ExpertQuantKind::Q6K => kernels
                .mul_mm_id_gemm_q6_k_bf16(
                    stream,
                    expert_ptrs_p,
                    topk_indices_p,
                    x,
                    y,
                    m,
                    n,
                    k,
                    k_used,
                )
                .map_err(|e| LlmError::Backend(format!("mul_mm_id_gemm q6k: {e:?}"))),
            ExpertQuantKind::Bf16 => kernels
                .mul_mm_id_gemm_bf16_bf16(
                    stream,
                    expert_ptrs_p,
                    topk_indices_p,
                    x,
                    y,
                    m,
                    n,
                    k,
                    k_used,
                )
                .map_err(|e| LlmError::Backend(format!("mul_mm_id_gemm bf16: {e:?}"))),
        }
    }
}

/// T246.10 TrackE.3 / TrackE.4 — dispatch the appropriate sort-permutation
/// Group-GEMM kernel based on the expert quant kind. Currently supports
/// Q4_K (TrackE.3) and Q5_K (TrackE.4). For other quants the caller MUST
/// fall back to `dispatch_indexed_group_gemm` (the `sort_active` gate in
/// `moe_ffn_forward_step_group_gemm` enforces this).
#[allow(clippy::too_many_arguments)]
pub(crate) fn dispatch_sorted_group_gemm(
    kernels: &LlmKernels,
    stream: &Arc<CudaStream>,
    kind: ExpertQuantKind,
    expert_ptrs_p: u64,
    topk_indices_p: u64,
    ids_src1_p: u64,
    ids_dst_p: u64,
    x: u64,
    y: u64,
    m: i32,
    n: i32,
    k: i32,
    k_used: i32,
    label: &'static str,
) -> Result<(), LlmError> {
    unsafe {
        match kind {
            ExpertQuantKind::Q4K => kernels
                .mul_mm_id_gemm_q4_k_sorted_bf16(
                    stream,
                    expert_ptrs_p,
                    topk_indices_p,
                    ids_src1_p,
                    ids_dst_p,
                    x,
                    y,
                    m,
                    n,
                    k,
                    k_used,
                )
                .map_err(|e| LlmError::Backend(format!("sorted {label} q4k: {e:?}"))),
            ExpertQuantKind::Q5K => kernels
                .mul_mm_id_gemm_q5_k_sorted_bf16(
                    stream,
                    expert_ptrs_p,
                    topk_indices_p,
                    ids_src1_p,
                    ids_dst_p,
                    x,
                    y,
                    m,
                    n,
                    k,
                    k_used,
                )
                .map_err(|e| LlmError::Backend(format!("sorted {label} q5k: {e:?}"))),
            ExpertQuantKind::Q6K | ExpertQuantKind::Bf16 => Err(LlmError::Backend(format!(
                "dispatch_sorted_group_gemm: unsupported kind {kind:?} for {label} \
                 — only Q4_K and Q5_K have sorted kernels"
            ))),
        }
    }
}

/// MoE FFN weights for one layer (Qwen3.6-35B-A3B). Top-K routed experts +
/// parallel shared expert. T246.6.
pub(crate) struct MoeFfnQ4K {
    /// `[n_experts, D]` quantized — router logits.
    pub(crate) gate_inp: QuantTensor,
    /// `n_experts` × `[expert_f, D]` quantized — per-expert gate.
    pub(crate) gate_exps: Vec<QuantTensor>,
    /// `n_experts` × `[expert_f, D]` quantized — per-expert up.
    pub(crate) up_exps: Vec<QuantTensor>,
    /// `n_experts` × `[D, expert_f]` quantized — per-expert down.
    pub(crate) down_exps: Vec<QuantTensor>,
    /// `[D]` BF16 — shared-expert routing gain (sigmoid-gated scalar).
    pub(crate) gate_inp_shexp: CudaSlice<half::bf16>,
    /// `[expert_f, D]` quantized — shared-expert gate.
    pub(crate) gate_shexp: QuantTensor,
    /// `[expert_f, D]` quantized — shared-expert up.
    pub(crate) up_shexp: QuantTensor,
    /// `[D, expert_f]` quantized — shared-expert down.
    pub(crate) down_shexp: QuantTensor,

    // ── T246.8 A2 — device-side dispatch tables (built once at load time) ──
    /// `[n_experts]` u64 — base device pointers of each `gate_exps[e]`.
    pub(crate) gate_exp_ptrs_dev: CudaSlice<u64>,
    pub(crate) up_exp_ptrs_dev: CudaSlice<u64>,
    pub(crate) down_exp_ptrs_dev: CudaSlice<u64>,
    /// Quant family for each set (homogeneous within a stacked tensor).
    pub(crate) gate_exp_kind: ExpertQuantKind,
    pub(crate) up_exp_kind: ExpertQuantKind,
    pub(crate) down_exp_kind: ExpertQuantKind,
}

/// FFN variant — dense SwiGLU or top-K MoE.
pub(crate) enum FfnQ4K {
    Dense {
        gate: QuantTensor,
        up: QuantTensor,
        down: QuantTensor,
    },
    Moe(Box<MoeFfnQ4K>),
}

/// Attention block weights (one of every 4 layers in Qwen3.6).
pub(crate) struct AttnBlockQ4K {
    /// `[D]` BF16 — pre-attention RMSNorm gain.
    pub(crate) attn_norm: CudaSlice<half::bf16>,
    /// `[D]` BF16 — post-attention RMSNorm gain (HF naming `post_attention_norm`,
    /// or `ffn_norm` for qwen2 / qwen3 pure-transformer variants).
    pub(crate) post_norm: CudaSlice<half::bf16>,
    /// `[head_dim]` BF16 — per-head Q normalization gain (RMSNorm applied per head).
    /// `None` for qwen2 (no QK-norm).
    pub(crate) q_norm: Option<CudaSlice<half::bf16>>,
    /// `[head_dim]` BF16 — per-head K normalization gain.
    /// `None` for qwen2 (no QK-norm).
    pub(crate) k_norm: Option<CudaSlice<half::bf16>>,
    /// `[n_q_heads * head_dim, D]` quantized — Q projection.
    pub(crate) w_q: QuantTensor,
    /// `[n_kv_heads * head_dim, D]` quantized — K projection.
    pub(crate) w_k: QuantTensor,
    /// `[n_kv_heads * head_dim, D]` quantized — V projection.
    pub(crate) w_v: QuantTensor,
    /// `[D, n_q_heads * head_dim]` quantized — output projection.
    pub(crate) w_o: QuantTensor,
    /// `[q_dim]` BF16 — Q projection bias (qwen2 only).
    pub(crate) b_q: Option<CudaSlice<half::bf16>>,
    /// `[kv_dim]` BF16 — K projection bias (qwen2 only).
    pub(crate) b_k: Option<CudaSlice<half::bf16>>,
    /// `[kv_dim]` BF16 — V projection bias (qwen2 only).
    pub(crate) b_v: Option<CudaSlice<half::bf16>>,
    /// FFN — dense (Qwen3.6-27B) or MoE (Qwen3.6-35B-A3B).
    pub(crate) ffn: FfnQ4K,
}

/// SSM (gated delta net) block weights — 48/64 layers in Qwen3.6.
pub(crate) struct SsmBlockQ4K {
    /// `[D]` BF16 — pre-SSM RMSNorm gain.
    pub(crate) attn_norm: CudaSlice<half::bf16>,
    /// `[D]` BF16 — post-SSM RMSNorm gain.
    pub(crate) post_norm: CudaSlice<half::bf16>,
    /// `[conv_dim, D]` quantized — fused QKV projection (split after conv).
    pub(crate) w_qkv: QuantTensor,
    /// `[value_dim, D]` quantized — gate (z) projection.
    pub(crate) w_gate: QuantTensor,
    /// `[conv_kernel, conv_dim]` BF16 — depth-wise 1-D conv kernel.
    pub(crate) conv1d: CudaSlice<half::bf16>,
    /// `[num_v_heads, D]` quantized — alpha (dt) projection.
    pub(crate) w_alpha: QuantTensor,
    /// `[num_v_heads, D]` quantized — beta projection.
    pub(crate) w_beta: QuantTensor,
    /// `[num_v_heads]` BF16 — dt bias.
    pub(crate) dt_bias: CudaSlice<half::bf16>,
    /// `[num_v_heads]` BF16 — A scalar per head (from log A).
    pub(crate) ssm_a: CudaSlice<half::bf16>,
    /// `[head_v_dim]` BF16 — gated RMSNorm gain.
    pub(crate) ssm_norm: CudaSlice<half::bf16>,
    /// `[D, value_dim]` quantized — output projection.
    pub(crate) ssm_out: QuantTensor,
    /// FFN — dense or MoE (T246.6, Qwen3.6-35B-A3B).
    pub(crate) ffn: FfnQ4K,
}

/// Per-block weights : either attention or SSM.
pub(crate) enum BlockQ4K {
    Attn(AttnBlockQ4K),
    Ssm(SsmBlockQ4K),
}

/// Per-attention-layer KV cache (allocated up to `max_seq` tokens).
pub(crate) struct KvCache {
    /// `[max_seq, n_kv_heads * head_dim]` BF16.
    pub(crate) k: CudaSlice<half::bf16>,
    /// `[max_seq, n_kv_heads * head_dim]` BF16.
    pub(crate) v: CudaSlice<half::bf16>,
}

/// Precomputed RoPE inverse frequencies `[rope_dim / 2]` F32, on GPU.
pub(crate) struct RopeFreqs {
    pub(crate) inv_freq: CudaSlice<f32>,
}

/// Per-SSM-layer recurrent state.
pub(crate) struct SsmState {
    /// `[num_v_heads, head_v_dim, head_v_dim]` BF16 — outer-product state.
    pub(crate) state: CudaSlice<half::bf16>,
    /// `[(conv_kernel - 1), conv_dim]` BF16 — ring buffer for depth-wise conv.
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
#[allow(dead_code)]
fn get_attn_layer_idx(cfg: &Qwen35Config, global_li: usize) -> usize {
    cfg.attention_indices
        .iter()
        .position(|&i| i == global_li)
        .unwrap_or_else(|| panic!("layer {global_li} not in attention_indices"))
}

/// Pre-allocated scratch buffers for decode_step (hoisted out of the hot path).
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
    /// T246.5.5 — staging buffer for Q8_1 quantized activation (used by the
    /// dp4a Q4_K matmul path). Sized to accommodate the largest K seen in
    /// any matmul of the model: `(max_k / 32) * 36` bytes.
    pub(crate) x_q8_scratch: CudaSlice<u8>,
    /// T246.5.7 — FlashDecode split-K staging buffers.
    /// `[n_q_heads, N_SPLIT]` floats — partial max scores per (head, split).
    pub(crate) gqa_partial_m: CudaSlice<f32>,
    /// `[n_q_heads, N_SPLIT]` floats — partial l (sum of expf weights).
    pub(crate) gqa_partial_l: CudaSlice<f32>,
    /// `[n_q_heads, N_SPLIT, head_dim]` bf16 — partial output.
    pub(crate) gqa_partial_o: CudaSlice<half::bf16>,
    /// T246.6 — MoE staging.
    /// `[n_experts]` BF16 — router logits scratch.
    pub(crate) moe_router_logits: CudaSlice<half::bf16>,
    /// `[k]` i32 — top-K expert indices.
    pub(crate) moe_topk_idx: CudaSlice<i32>,
    /// `[k]` BF16 — renormalized top-K weights.
    pub(crate) moe_topk_w: CudaSlice<half::bf16>,
    /// `[expert_f]` BF16 — per-expert gate buffer.
    pub(crate) moe_expert_gate: CudaSlice<half::bf16>,
    /// `[expert_f]` BF16 — per-expert up buffer.
    pub(crate) moe_expert_up: CudaSlice<half::bf16>,
    /// `[d]` BF16 — per-expert output (down result, accumulated into h).
    pub(crate) moe_expert_out: CudaSlice<half::bf16>,
    /// `[1]` BF16 — shared-expert sigmoid dot product scratch.
    pub(crate) moe_shexp_dot: CudaSlice<half::bf16>,
    // ── T246.7 P1.3 — Lookahead Decoding tree-attention scratch ──
    /// `[MAX_TREE_SIZE]` u32 — input draft tokens for `decode_step_tree`.
    pub(crate) tree_drafts: CudaSlice<u32>,
    /// `[MAX_TREE_SIZE]` i32 — parent pointer per tree node, root = -1.
    pub(crate) tree_parents: CudaSlice<i32>,
    /// `[MAX_TREE_SIZE]` u16 — depth per tree node, root = 0. T246.10 A6
    /// widened u8 → u16 so linear-chain prefill with N up to 65k still fits
    /// the depth field (a chain of 512 tokens has max depth 511 which
    /// overflows u8).
    pub(crate) tree_depths: CudaSlice<u16>,
    /// `[MAX_TREE_SIZE]` u32 — argmax token per tree node row of logits.
    pub(crate) tree_argmax: CudaSlice<u32>,
    /// `[MAX_TREE_SIZE, n_q, n_split]` f32 — partial m for tree GQA.
    pub(crate) tree_gqa_partial_m: CudaSlice<f32>,
    /// `[MAX_TREE_SIZE, n_q, n_split]` f32 — partial l for tree GQA.
    pub(crate) tree_gqa_partial_l: CudaSlice<f32>,
    /// `[MAX_TREE_SIZE, n_q, n_split, head_dim]` BF16 — partial o for tree GQA.
    pub(crate) tree_gqa_partial_o: CudaSlice<half::bf16>,
    // ── T246.7 P1.4 — Lookahead Decoding multi-token forward scratch ──
    //
    // These buffers hold per-tree-row activations for `decode_step_tree`
    // when `tree_size > 1`. They are sized for `MAX_TREE_SIZE` rows so the
    // capacity is fixed at construction time.
    //
    // For `tree_size = 1` the existing per-token buffers (`h`, `q_buf`,
    // ...) are used for a strict drop-in to `decode_step` — these tree
    // buffers are touched only when `tree_size > 1`.
    /// `[MAX_TREE_SIZE, D]` BF16 — per-tree-row residual stream.
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
    /// `[MAX_TREE_SIZE, max(f, q_dim, 2*q_dim)]` BF16 — per-tree-row
    /// scratch for FFN gate / QG (re-used across the forward).
    pub(crate) tree_gate_buf: CudaSlice<half::bf16>,
    /// `[MAX_TREE_SIZE, max(f, 2*q_dim)]` BF16 — per-tree-row scratch
    /// for FFN up / QG (re-used across the forward).
    pub(crate) tree_up_buf: CudaSlice<half::bf16>,
    /// `[MAX_TREE_SIZE, vocab]` BF16 — per-tree-row LM-head logits.
    pub(crate) tree_logits: CudaSlice<half::bf16>,
    /// Host-pinned `[MAX_TREE_SIZE]` u32 — DtoH target for tree argmax tokens.
    pub(crate) tree_argmax_host_pinned: PinnedHostSlice<u32>,

    // ── T246.10 TrackE.2 — batched MoE Group-GEMM scratch ──
    //
    // Per-row MoE scratch sized for MAX_TREE_SIZE rows so the prefill /
    // tree-verify path can route ALL M tokens through one Group-GEMM
    // launch per gate / up / down. Only used when
    // `RUSTORCH_MOE_GROUP_GEMM=1` AND `tree_size >= GROUP_GEMM_MIN_M`.
    //
    // Memory budget (Qwen3.6-A3B : n_experts=128, k_used=8, ef=18944,
    // d=2048, MAX_TREE_SIZE=512) :
    //   moe_router_logits_M : 512×128×2  ≈ 128 KB
    //   moe_topk_idx_M      : 512×8×4    ≈ 16 KB
    //   moe_topk_w_M        : 512×8×2    ≈ 8 KB
    //   moe_expert_gate_M   : 512×8×ef×2 ≈ 155 MB
    //   moe_expert_up_M     : 512×8×ef×2 ≈ 155 MB
    //   moe_expert_out_M    : 512×8×d×2  ≈ 16 MB
    // Total : ~326 MB (one-time allocation, fits comfortably in GB10's 120 GB).
    // For Dense / non-MoE variants n_experts == 0 so the (.max(1)) sizing
    // reduces this to a few hundred bytes.
    /// `[MAX_TREE_SIZE, n_experts]` BF16 — per-token router logits.
    pub(crate) moe_router_logits_m: CudaSlice<half::bf16>,
    /// `[MAX_TREE_SIZE, k_used]` i32 — per-token top-K expert indices.
    pub(crate) moe_topk_idx_m: CudaSlice<i32>,
    /// `[MAX_TREE_SIZE, k_used]` BF16 — per-token renormalized weights.
    pub(crate) moe_topk_w_m: CudaSlice<half::bf16>,
    /// `[MAX_TREE_SIZE, k_used, expert_f]` BF16 — per-token gate output.
    pub(crate) moe_expert_gate_m: CudaSlice<half::bf16>,
    /// `[MAX_TREE_SIZE, k_used, expert_f]` BF16 — per-token up output.
    pub(crate) moe_expert_up_m: CudaSlice<half::bf16>,
    /// `[MAX_TREE_SIZE, k_used, d]` BF16 — per-token down output.
    pub(crate) moe_expert_out_m: CudaSlice<half::bf16>,
    /// `[MAX_TREE_SIZE, K/32 * 36]` u8 — per-token Q8_1 staging for dp4a.
    pub(crate) moe_x_q8_m: CudaSlice<u8>,

    // ── T246.10 MMQ-WHOLESALE — packed Q8_1 staging for the INT8-staged Q4_K
    // matmul path. Sized to hold the worst-case M × (K/128) × 144 B payload
    // (= K_max chosen as max(d, expert_f) since the shared expert runs both
    // gate/up at K=d AND down at K=expert_f). For Qwen3.6-A3B at
    // MAX_TREE_SIZE=512, max(d=5120, ef=18944)=18944 → 512 × 148 × 144 ≈
    // 10.9 MB. Zero-sized for non-MoE / decode-only variants.
    /// `[MAX_TREE_SIZE * (max(d, ef) / 128) * 144]` u8 — packed Q8_1 input
    /// for `quantize_mmq_q8_1_bf16_ds4` + `mul_mat_q4_k_q8_1_mma`.
    pub(crate) moe_x_q8_mmq_m: CudaSlice<u8>,

    // ── T246.10 TrackE.3 — sort-permutation Group-GEMM scratch ──
    //
    // Used only when `RUSTORCH_MOE_GROUP_GEMM_SORTED=1` is also set on top of
    // `RUSTORCH_MOE_GROUP_GEMM=1`. The `mm_ids_helper_bf16` kernel writes
    // permutation tables here once per layer, and the sorted Group-GEMM
    // consumes them. Total ≈ (M*k_used + n_experts + 1) × 4 B per layer
    // (≈ 16 KB for Qwen3.6 MAX_TREE_SIZE=512, k_used=8, n_experts=128).
    /// `[MAX_TREE_SIZE * k_used]` i32 — compact_idx → source token.
    pub(crate) moe_ids_src1_m: CudaSlice<i32>,
    /// `[MAX_TREE_SIZE * k_used]` i32 — compact_idx → flat dst (token*k_used+slot).
    pub(crate) moe_ids_dst_m: CudaSlice<i32>,
    /// `[n_experts + 1]` i32 — prefix sums per expert.
    pub(crate) moe_expert_bounds: CudaSlice<i32>,

    // ── T246.7 TrackC — SSM-hybrid Lookahead per-branch state forking ──
    //
    // For SSM-hybrid variants (Dense / MoE on Qwen3.6) every SSM layer
    // carries a recurrent state that must be forked per tree branch
    // during the verify pass, then the deepest accepted node's state is
    // copied back into the model's per-layer scratch on commit.
    //
    // Memory budget on Qwen3.6-27B (48 SSM layers, n_v=48, head_v=128) :
    //   - tree_ssm_states : 32 × 48 × 48 × 128 × 128 × 2 bytes ≈ 2.25 GB
    //   - tree_conv_states : 32 × 48 × (4-1) × 10240 × 2 bytes ≈ 92 MB
    // For pure-transformer variants these vectors are empty (zero cost).
    /// Per-SSM-layer tree-fork state buffer, sized
    /// `[MAX_TREE_SIZE × n_v_heads × head_v_dim²]` BF16 each.
    /// Empty for pure-transformer variants.
    pub(crate) tree_ssm_states: Vec<CudaSlice<half::bf16>>,
    /// Per-SSM-layer tree-fork conv1d state buffer, sized
    /// `[MAX_TREE_SIZE × (conv_kernel-1) × conv_dim]` BF16 each.
    /// Empty for pure-transformer variants.
    pub(crate) tree_conv_states: Vec<CudaSlice<half::bf16>>,
    /// Per-tree-row scratch for the SSM-hybrid forward.
    /// `[MAX_TREE_SIZE × conv_dim]` BF16 — qkv_mixed (post-input-projection).
    pub(crate) tree_ssm_qkv_mixed: CudaSlice<half::bf16>,
    /// `[MAX_TREE_SIZE × conv_dim]` BF16 — conv_out.
    pub(crate) tree_ssm_conv_out: CudaSlice<half::bf16>,
    /// `[MAX_TREE_SIZE × value_dim]` BF16 — z (gate).
    pub(crate) tree_ssm_z: CudaSlice<half::bf16>,
    /// `[MAX_TREE_SIZE × n_v_heads]` BF16 — alpha (dt).
    pub(crate) tree_ssm_alpha: CudaSlice<half::bf16>,
    /// `[MAX_TREE_SIZE × n_v_heads]` BF16 — beta.
    pub(crate) tree_ssm_beta: CudaSlice<half::bf16>,
    /// `[MAX_TREE_SIZE × value_dim]` BF16 — q broadcast to n_v heads.
    pub(crate) tree_ssm_q_v: CudaSlice<half::bf16>,
    /// `[MAX_TREE_SIZE × value_dim]` BF16 — k broadcast to n_v heads.
    pub(crate) tree_ssm_k_v: CudaSlice<half::bf16>,
    /// `[MAX_TREE_SIZE × value_dim]` BF16 — gated SSM output.
    pub(crate) tree_ssm_out_buf: CudaSlice<half::bf16>,
    /// `[MAX_TREE_SIZE]` i32 — depth-wave indices buffer used by the
    /// `delta_net_step_tree_bf16` launcher (one launch per BFS depth).
    pub(crate) tree_ssm_wave_indices: CudaSlice<i32>,

    // ── T246.8 A5 — lm_head split-K staging buffer ──
    /// `[LM_HEAD_K_CHUNKS, vocab]` FP32 — partial sums for the split-K
    /// Q6_K SGEMV used by `dispatch_lm_head` when `RUSTORCH_LM_HEAD_SPLIT=1`
    /// is set AND the lm_head is large enough (N >= 100k). Sized for the
    /// max K_CHUNKS=8 supported (the lm_head case on Qwen3.6-35B-A3B
    /// K=2048 → blocks_per_row=8 → K_CHUNKS=8). For Dense/non-MoE models
    /// where the lm_head is smaller this buffer is still allocated but
    /// unused.
    pub(crate) lm_head_split_partial: CudaSlice<f32>,
}

/// CUDA-resident Qwen3.5/3.6 hybrid model with Q4_K_M weights.
pub struct Qwen35ModelCudaQ4K {
    pub config: Qwen35Config,
    pub(crate) ctx: Arc<CudaContext>,
    pub(crate) stream: Arc<CudaStream>,
    #[allow(dead_code)]
    pub(crate) session: LtSession,
    pub(crate) kernels: LlmKernels,
    /// Token embedding `[V, D]` BF16 (dequantized at load — single read at start).
    pub(crate) token_emb: CudaSlice<half::bf16>,
    /// Final RMSNorm gain `[D]` BF16.
    pub(crate) final_norm: CudaSlice<half::bf16>,
    /// LM head `[V, D]` quantized (typically Q6_K).
    pub(crate) lm_head: QuantTensor,
    /// Precomputed RoPE inv_freq buffer.
    pub(crate) rope_freqs: RopeFreqs,
    /// Per-block weights, one per layer.
    pub(crate) blocks: Vec<BlockQ4K>,
    /// KV cache, one entry per attention layer (in original layer-index order).
    pub(crate) kv_caches: Vec<KvCache>,
    /// SSM state, one entry per SSM layer.
    pub(crate) ssm_states: Vec<SsmState>,
    /// Maximum sequence length supported by the pre-allocated KV cache.
    pub(crate) max_seq: usize,
    /// Current decode position (incremented after each forward_token).
    pub(crate) position: usize,
    /// Pre-allocated scratch buffers (one-time alloc).
    pub(crate) scratch: DecodeScratch,
    /// T246.5.3 — device-resident position counter, mirror of `position`.
    /// Used by `rope_partial_bf16_devcnt` so RoPE remains correct across
    /// CUDA Graph replays. Updated each step via tiny H2D (Phase B).
    pub(crate) position_dev: CudaSlice<i32>,
    /// T246.5.3 — device-resident kv_len counter (= position + 1).
    /// Used by `gqa_decode_online_bf16_devcnt`. Updated each step.
    pub(crate) kv_len_dev: CudaSlice<i32>,
    /// T246.5.3 — device-resident current-token-id buffer.
    /// Used by `embedding_lookup_bf16` so the per-step token id is read
    /// from device memory inside the captured graph.
    pub(crate) current_token_dev: CudaSlice<u32>,
    /// T246.5.3 — captured CUDA Graph of one full decode_step body.
    /// Initialized lazily on the 2nd decode call (after the 1st-call warmup
    /// triggers all kernel JIT compilations). Once set, every subsequent
    /// decode_step replays this graph in a single launch instead of issuing
    /// ~1908 individual cuLaunchKernel calls.
    pub(crate) decode_graph: Option<CudaGraph>,
    /// T246.5.5 — toggle for the dp4a Q4_K matmul path. Off by default:
    /// current dp4a kernel is W-bandwidth-bound on M=1 (same as the float
    /// v2 path) but adds a quantize-q8_1 kernel per matmul → ~1% slower
    /// than v2 today. Set `RUSTORCH_USE_DP4A_Q4K=1` at process start to
    /// enable for benchmarking / iterative kernel optimization.
    pub(crate) use_dp4a_q4k: bool,
    /// T246.8 A1 — toggle for the fused SSM mega-kernels. ON by default
    /// (set `RUSTORCH_SSM_FUSE=0` to fall back to the unfused chain for
    /// A/B benchmarking and parity testing). When ON, the SSM block
    /// pre-step (sigmoid+add+softplus+mul on alpha/beta) and post-step
    /// (rms_norm+silu+mul on out/z) chains are each replaced by a single
    /// fused launch (`ssm_pre_step_bf16` / `ssm_post_step_bf16`).
    pub(crate) use_ssm_fuse: bool,
    /// T246.5.6 — pinned host buffer (1× u32) for the per-step next-token
    /// DtoH. Replacing `memcpy_dtov` (Vec<u32> on pageable mem, which
    /// the driver implicitly synchronizes) with `memcpy_dtoh` to this
    /// pinned slice keeps the DtoH truly async and lets the driver DMA
    /// directly into host memory ; the only sync point becomes the
    /// `as_slice()` call which waits on a dedicated event instead of the
    /// full stream.
    pub(crate) next_token_host_pinned: PinnedHostSlice<u32>,
    /// T246.10 TrackI — Bag-of-graphs cache for `prefill_tokens` captured
    /// bodies, keyed by `N` (= linear-chain `tree_size`). Populated lazily
    /// on the second call with a given N (after a warmup pass that triggers
    /// nvrtc compilation for every kernel). Each subsequent call with the
    /// same N replays the captured graph in a single launch.
    ///
    /// Reset (cleared) by `reset_state()` so per-conversation state changes
    /// always force a re-capture.
    pub(crate) prefill_graphs: HashMap<usize, CudaGraph>,
    /// T246.10 TrackI — set of N values for which the prefill body has
    /// been run once (uncaptured warmup) and is ready to be captured on
    /// the next call. Mirrors llama.cpp's two-call warmup pattern in
    /// `ggml_backend_cuda_graph_compute` : the first call triggers JIT
    /// compilation of every kernel (nvrtc/cuModuleLoadData) which is not
    /// itself capturable ; only the second call enters `begin_capture`
    /// / `end_capture`.
    pub(crate) prefill_warmed: std::collections::HashSet<usize>,
    /// T246.10 TrackI — pinned-host descriptor buffers for the tree
    /// `drafts` / `parents` / `depths` HtoD uploads. We pre-bake the
    /// `parents = [-1, 0, 1, ..., N-2]` and `depths = [0, 1, ..., N-1]`
    /// values for the maximum N once (linear-chain semantics never
    /// change), and only `drafts` is rewritten per call. Pinned memory
    /// is required so the HtoD upload (which happens OUTSIDE the capture
    /// region but before each replay) does NOT implicitly synchronize the
    /// stream (`cuMemcpyHtoDAsync` on pageable host memory degenerates
    /// into a synchronous copy that aborts in-progress captures).
    pub(crate) prefill_drafts_host_pinned: PinnedHostSlice<u32>,
    /// T246.10 TrackI — linear-chain wave-indices buffer
    /// `[0, 1, 2, ..., MAX_TREE_SIZE - 1]`. Populated once at construction
    /// (constant for the lifetime of the model). The hybrid SSM path
    /// reads `tree_ssm_wave_indices_linear + d*4` with `wave_size=1` for
    /// depth `d` so each SSM wave kernel call needs zero HtoD setup
    /// before launch — capturable.
    pub(crate) tree_ssm_wave_indices_linear: CudaSlice<i32>,
}

impl Qwen35ModelCudaQ4K {
    /// Load Qwen3.5/3.6 hybrid GGUF (Q4_K_M typically) directly to GPU using
    /// raw Q4_K / Q5_K / Q6_K bytes for matmuls. Norms, biases, ssm_a,
    /// dt_bias, conv1d : F32 → BF16 dequant (small tensors).
    ///
    /// Multi-shard GGUF : caller must pre-merge shards (TODO T246.5+).
    pub fn from_gguf(path: &Path, max_seq: usize) -> Result<Self, LlmError> {
        let ctx =
            CudaContext::new(0).map_err(|e| LlmError::Backend(format!("CudaContext: {e:?}")))?;
        // T246.5.3 — Disable cudarc's automatic event tracking BEFORE allocating
        // anything. With multi-stream mode + event tracking on, every
        // device_ptr_mut() injects `stream.wait(event)` calls referencing events
        // recorded on the warmup pass. During CUDA Graph capture those waits
        // reference an event NOT belonging to the capture → the graph is
        // invalidated immediately (CUDA_ERROR_STREAM_CAPTURE_INVALIDATED on
        // the first op after begin_capture). All decode work runs on a single
        // stream so we don't need cross-stream safety here.
        // SAFETY: disable_event_tracking only affects slices created AFTER this
        // call. We immediately allocate everything below, so no pre-existing
        // tracked slices exist.
        unsafe {
            ctx.disable_event_tracking();
        }
        // T246.5.3 — dedicated non-default stream for CUDA Graph capture.
        // CUDA Graphs cannot capture the default/legacy stream — begin_capture
        // returns CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED on it.
        let stream = ctx
            .new_stream()
            .map_err(|e| LlmError::Backend(format!("new_stream: {e:?}")))?;
        let session = LtSession::new(stream.clone())
            .map_err(|e| LlmError::Backend(format!("LtSession: {e:?}")))?;
        let kernels = LlmKernels::new(ctx.clone());

        let cfg = crate::qwen35::parse_config(path)
            .map_err(|e| LlmError::Backend(format!("parse_config: {e:?}")))?;
        let file = GgufFile::open(path).map_err(|e| LlmError::Backend(format!("gguf: {e:?}")))?;

        // T246.6 — MoE (Qwen3.6-35B-A3B) is now supported via FfnQ4K::Moe.
        // Dense (Qwen3.6-27B) continues through the original path.

        // ---- Helpers ----
        let load_quant = |name: &str| -> Result<QuantTensor, LlmError> {
            let info = file
                .tensor(name)
                .ok_or_else(|| LlmError::MissingWeight(name.to_string()))?;
            let bytes = file.tensor_bytes(info);
            let n = info.shape[1] as usize;
            let k = info.shape[0] as usize;
            match info.dtype {
                GgmlType::Q4_K => {
                    let dev = stream
                        .memcpy_stod(bytes)
                        .map_err(|e| LlmError::Backend(format!("upload {name}: {e:?}")))?;
                    Ok(QuantTensor::Q4K { bytes: dev, n, k })
                },
                GgmlType::Q5_K => {
                    let dev = stream
                        .memcpy_stod(bytes)
                        .map_err(|e| LlmError::Backend(format!("upload {name}: {e:?}")))?;
                    Ok(QuantTensor::Q5K { bytes: dev, n, k })
                },
                GgmlType::Q6_K => {
                    let dev = stream
                        .memcpy_stod(bytes)
                        .map_err(|e| LlmError::Backend(format!("upload {name}: {e:?}")))?;
                    Ok(QuantTensor::Q6K { bytes: dev, n, k })
                },
                // F32 / F16 / Q8_0 fallback : dequant to BF16 at load time,
                // dispatch via sgemv_bf16_bf16. Q8_0 unblocks 35B-A3B
                // (T246.6) where attn_qkv / attn_output are Q8_0 not Q4_K
                // in the UD-Q4_K_M repack.
                GgmlType::F32 | GgmlType::F16 | GgmlType::Q8_0 => {
                    let f32_buf = rustorch_gguf::dequant::dequant_to_f32(info, bytes)
                        .map_err(|e| LlmError::Backend(format!("dequant {name}: {e:?}")))?;
                    let bf: Vec<half::bf16> =
                        f32_buf.iter().copied().map(half::bf16::from_f32).collect();
                    let dev = stream
                        .memcpy_stod(&bf)
                        .map_err(|e| LlmError::Backend(format!("upload {name}: {e:?}")))?;
                    Ok(QuantTensor::Bf16 { weights: dev, n, k })
                },
                other => Err(LlmError::Backend(format!(
                    "unsupported dtype {other:?} for {name} — only Q4_K/Q5_K/Q6_K/Q8_0/F32/F16 supported"
                ))),
            }
        };

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

        // T246.6 — Load a stacked-expert quantized tensor. GGUF stores
        // `[d_inner, rows, n_experts]` as one quantized blob. We slice and
        // upload each expert independently. Supports K-quants (Q4/5/6_K
        // preserved on-device) and Q8_0 / F32 / F16 (dequantized to BF16
        // on the host side). 35B-A3B is mixed Q4_K + Q5_K + Q8_0 in the
        // UD-Q4_K_M repack.
        let load_stacked_quant_experts = |name: &str,
                                          n_experts: usize,
                                          rows: usize,
                                          k: usize|
         -> Result<Vec<QuantTensor>, LlmError> {
            let info = file
                .tensor(name)
                .ok_or_else(|| LlmError::MissingWeight(name.to_string()))?;
            let bytes = file.tensor_bytes(info);
            let block_size = info.dtype.block_size();
            let type_size = info.dtype.type_size();
            let weights_per_expert = rows * k;
            if weights_per_expert % block_size != 0 {
                return Err(LlmError::Backend(format!(
                    "{name}: expert size {} not aligned on {} ({:?})",
                    weights_per_expert, block_size, info.dtype
                )));
            }
            let stride = weights_per_expert / block_size * type_size;
            if bytes.len() != n_experts * stride {
                return Err(LlmError::Backend(format!(
                    "{name}: expected {} bytes ({} × {}), got {}",
                    n_experts * stride,
                    n_experts,
                    stride,
                    bytes.len()
                )));
            }
            let mut out = Vec::with_capacity(n_experts);
            for e in 0..n_experts {
                let slice = &bytes[e * stride..(e + 1) * stride];
                let qt = match info.dtype {
                    GgmlType::Q4_K => {
                        let dev = stream.memcpy_stod(slice).map_err(|err| {
                            LlmError::Backend(format!("upload {name}#{e}: {err:?}"))
                        })?;
                        QuantTensor::Q4K {
                            bytes: dev,
                            n: rows,
                            k,
                        }
                    },
                    GgmlType::Q5_K => {
                        let dev = stream.memcpy_stod(slice).map_err(|err| {
                            LlmError::Backend(format!("upload {name}#{e}: {err:?}"))
                        })?;
                        QuantTensor::Q5K {
                            bytes: dev,
                            n: rows,
                            k,
                        }
                    },
                    GgmlType::Q6_K => {
                        let dev = stream.memcpy_stod(slice).map_err(|err| {
                            LlmError::Backend(format!("upload {name}#{e}: {err:?}"))
                        })?;
                        QuantTensor::Q6K {
                            bytes: dev,
                            n: rows,
                            k,
                        }
                    },
                    // Dequant path : produce per-expert BF16 device buffer.
                    GgmlType::Q8_0 | GgmlType::F32 | GgmlType::F16 => {
                        let synth = rustorch_gguf::tensor::TensorInfo {
                            name: format!("{name}#{e}"),
                            shape: vec![k as u64, rows as u64],
                            dtype: info.dtype,
                            offset: 0,
                        };
                        let f32_buf = rustorch_gguf::dequant::dequant_to_f32(&synth, slice)
                            .map_err(|err| {
                                LlmError::Backend(format!("dequant {name}#{e}: {err:?}"))
                            })?;
                        let bf: Vec<half::bf16> =
                            f32_buf.iter().copied().map(half::bf16::from_f32).collect();
                        let dev = stream.memcpy_stod(&bf).map_err(|err| {
                            LlmError::Backend(format!("upload {name}#{e}: {err:?}"))
                        })?;
                        QuantTensor::Bf16 {
                            weights: dev,
                            n: rows,
                            k,
                        }
                    },
                    other => {
                        return Err(LlmError::Backend(format!(
                            "{name}: unsupported expert dtype {other:?}"
                        )));
                    },
                };
                out.push(qt);
            }
            Ok(out)
        };

        // T246.6 — Load the FFN portion for layer `li`. Returns Dense for
        // Qwen3.6-27B (and Qwen3PureTransformer if it ever lands), Moe for
        // Qwen3.6-35B-A3B.
        let load_ffn = |li: usize| -> Result<FfnQ4K, LlmError> {
            let key = |s: &str| format!("blk.{li}.{s}");
            match cfg.variant {
                Qwen35Variant::Dense
                | Qwen35Variant::Qwen2PureTransformer
                | Qwen35Variant::Qwen3PureTransformer => Ok(FfnQ4K::Dense {
                    gate: load_quant(&key("ffn_gate.weight"))?,
                    up: load_quant(&key("ffn_up.weight"))?,
                    down: load_quant(&key("ffn_down.weight"))?,
                }),
                Qwen35Variant::Moe => {
                    let n_e = cfg.n_experts;
                    let ef = cfg.expert_f;
                    let d = cfg.d;
                    let gate_exps =
                        load_stacked_quant_experts(&key("ffn_gate_exps.weight"), n_e, ef, d)?;
                    let up_exps =
                        load_stacked_quant_experts(&key("ffn_up_exps.weight"), n_e, ef, d)?;
                    let down_exps =
                        load_stacked_quant_experts(&key("ffn_down_exps.weight"), n_e, d, ef)?;

                    // T246.8 A2 — build device-side weight pointer tables.
                    // Each `Vec<QuantTensor>` is homogeneous in dtype (single
                    // GGUF tensor, same source dtype). The pointer values are
                    // stable for the model's lifetime since the underlying
                    // CudaSlice<u8>/CudaSlice<bf16> buffers are owned by
                    // `MoeFfnQ4K` and never reallocated.
                    let kind_of = |v: &[QuantTensor]| -> Result<ExpertQuantKind, LlmError> {
                        if v.is_empty() {
                            return Err(LlmError::Backend("MoE expert vec is empty".into()));
                        }
                        let k = match &v[0] {
                            QuantTensor::Q4K { .. } => ExpertQuantKind::Q4K,
                            QuantTensor::Q5K { .. } => ExpertQuantKind::Q5K,
                            QuantTensor::Q6K { .. } => ExpertQuantKind::Q6K,
                            QuantTensor::Bf16 { .. } => ExpertQuantKind::Bf16,
                        };
                        for (i, t) in v.iter().enumerate().skip(1) {
                            let ki = match t {
                                QuantTensor::Q4K { .. } => ExpertQuantKind::Q4K,
                                QuantTensor::Q5K { .. } => ExpertQuantKind::Q5K,
                                QuantTensor::Q6K { .. } => ExpertQuantKind::Q6K,
                                QuantTensor::Bf16 { .. } => ExpertQuantKind::Bf16,
                            };
                            if ki != k {
                                return Err(LlmError::Backend(format!(
                                    "MoE expert vec is heterogeneous : expert 0 is {k:?} but expert {i} is {ki:?}"
                                )));
                            }
                        }
                        Ok(k)
                    };
                    let build_ptrs = |v: &[QuantTensor]| -> Result<CudaSlice<u64>, LlmError> {
                        let host: Vec<u64> = v.iter().map(|t| t.base_ptr(&stream)).collect();
                        stream
                            .memcpy_stod(&host)
                            .map_err(|e| LlmError::Backend(format!("upload expert ptrs: {e:?}")))
                    };
                    let gate_exp_kind = kind_of(&gate_exps)?;
                    let up_exp_kind = kind_of(&up_exps)?;
                    let down_exp_kind = kind_of(&down_exps)?;
                    let gate_exp_ptrs_dev = build_ptrs(&gate_exps)?;
                    let up_exp_ptrs_dev = build_ptrs(&up_exps)?;
                    let down_exp_ptrs_dev = build_ptrs(&down_exps)?;

                    let moe = MoeFfnQ4K {
                        gate_inp: load_quant(&key("ffn_gate_inp.weight"))?,
                        gate_exps,
                        up_exps,
                        down_exps,
                        gate_inp_shexp: load_bf16(&key("ffn_gate_inp_shexp.weight"))?,
                        gate_shexp: load_quant(&key("ffn_gate_shexp.weight"))?,
                        up_shexp: load_quant(&key("ffn_up_shexp.weight"))?,
                        down_shexp: load_quant(&key("ffn_down_shexp.weight"))?,
                        gate_exp_ptrs_dev,
                        up_exp_ptrs_dev,
                        down_exp_ptrs_dev,
                        gate_exp_kind,
                        up_exp_kind,
                        down_exp_kind,
                    };
                    Ok(FfnQ4K::Moe(Box::new(moe)))
                },
            }
        };

        // ---- Globals ----
        let token_emb = load_bf16("token_embd.weight")?;
        let final_norm = load_bf16("output_norm.weight")?;
        // Precompute RoPE inv_freq for the attention layers.
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
        let lm_head = if file.tensor("output.weight").is_some() {
            load_quant("output.weight")?
        } else {
            // Tied embeddings — for simplicity we don't yet share storage.
            // Allocate a separate Q4_K copy or keep BF16 fallback.
            return Err(LlmError::Backend(
                "tied embeddings not yet supported in Q4K path (Qwen3.6 has separate output.weight)".into(),
            ));
        };

        // ---- Per-layer weights ----
        let mut blocks: Vec<BlockQ4K> = Vec::with_capacity(cfg.n_layers);
        let mut kv_caches: Vec<KvCache> = Vec::new();
        let mut ssm_states: Vec<SsmState> = Vec::new();

        let kv_dim = cfg.n_kv_heads * cfg.head_dim();
        let value_dim = cfg.ssm_dt_rank * cfg.ssm_state;
        let key_dim = cfg.ssm_groups * cfg.ssm_state;
        let conv_dim = 2 * key_dim + value_dim;
        let conv_kernel = cfg.ssm_conv_kernel;

        for li in 0..cfg.n_layers {
            let kind = cfg.layer_kind(li);
            let key = |s: &str| format!("blk.{li}.{s}");
            let block = match kind {
                LayerKind::Attention => {
                    // P1.5 — Qwen2 has no QK-norm, has Q/K/V biases, uses
                    // `ffn_norm` as the FFN pre-norm. Qwen3 / Qwen3.5 / 3.6
                    // keep the legacy paths.
                    let post_norm_name = format!("{}.weight", cfg.variant.ffn_pre_norm_name());
                    let q_norm = if cfg.variant.attn_has_qk_norm() {
                        Some(load_bf16(&key("attn_q_norm.weight"))?)
                    } else {
                        None
                    };
                    let k_norm = if cfg.variant.attn_has_qk_norm() {
                        Some(load_bf16(&key("attn_k_norm.weight"))?)
                    } else {
                        None
                    };
                    let (b_q, b_k, b_v) = if cfg.variant.attn_has_qkv_bias() {
                        (
                            Some(load_bf16(&key("attn_q.bias"))?),
                            Some(load_bf16(&key("attn_k.bias"))?),
                            Some(load_bf16(&key("attn_v.bias"))?),
                        )
                    } else {
                        (None, None, None)
                    };
                    let attn = AttnBlockQ4K {
                        attn_norm: load_bf16(&key("attn_norm.weight"))?,
                        post_norm: load_bf16(&key(&post_norm_name))?,
                        q_norm,
                        k_norm,
                        w_q: load_quant(&key("attn_q.weight"))?,
                        w_k: load_quant(&key("attn_k.weight"))?,
                        w_v: load_quant(&key("attn_v.weight"))?,
                        w_o: load_quant(&key("attn_output.weight"))?,
                        b_q,
                        b_k,
                        b_v,
                        ffn: load_ffn(li)?,
                    };
                    // Allocate KV cache for this layer.
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
                    BlockQ4K::Attn(attn)
                },
                LayerKind::Ssm => {
                    let ssm = SsmBlockQ4K {
                        attn_norm: load_bf16(&key("attn_norm.weight"))?,
                        post_norm: load_bf16(&key("post_attention_norm.weight"))?,
                        w_qkv: load_quant(&key("attn_qkv.weight"))?,
                        w_gate: load_quant(&key("attn_gate.weight"))?,
                        conv1d: load_bf16(&key("ssm_conv1d.weight"))?,
                        w_alpha: load_quant(&key("ssm_alpha.weight"))?,
                        w_beta: load_quant(&key("ssm_beta.weight"))?,
                        dt_bias: load_bf16(&key("ssm_dt.bias"))?,
                        ssm_a: load_bf16(&key("ssm_a"))?,
                        ssm_norm: load_bf16(&key("ssm_norm.weight"))?,
                        ssm_out: load_quant(&key("ssm_out.weight"))?,
                        ffn: load_ffn(li)?,
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
                    BlockQ4K::Ssm(ssm)
                },
            };
            blocks.push(block);
        }

        // ---- Pre-allocate decode scratch buffers (one-time) ----
        let kv_dim_attn = cfg.n_kv_heads * cfg.head_dim();
        let q_dim = cfg.n_q_heads * cfg.head_dim();
        // up_buf is (re)used as QG buffer (size 2*q_dim) ; gate_buf used as
        // attn raw output (size q_dim). Both must fit f for FFN AND 2*q_dim for QG.
        let scratch_up = cfg.f.max(2 * q_dim);
        let scratch_gate = cfg.f.max(q_dim);
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
                .alloc_zeros::<half::bf16>(2 * key_dim + value_dim)
                .map_err(|e| LlmError::Backend(format!("scratch qkv: {e:?}")))?,
            conv_out: stream
                .alloc_zeros::<half::bf16>(2 * key_dim + value_dim)
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
                .alloc_zeros::<half::bf16>(cfg.ssm_dt_rank * cfg.ssm_state)
                .map_err(|e| LlmError::Backend(format!("scratch q_v: {e:?}")))?,
            k_v: stream
                .alloc_zeros::<half::bf16>(cfg.ssm_dt_rank * cfg.ssm_state)
                .map_err(|e| LlmError::Backend(format!("scratch k_v: {e:?}")))?,
            ssm_out_buf: stream
                .alloc_zeros::<half::bf16>(cfg.ssm_dt_rank * cfg.ssm_state)
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
            x_q8_scratch: {
                // Max K across all matmuls = cfg.f (down_proj K=F). Round up
                // to next multiple of 32 for safety.
                let max_k = cfg.f.max(cfg.d).max(q_dim).max(value_dim);
                let max_k_blocks = max_k.div_ceil(32);
                stream
                    .alloc_zeros::<u8>(max_k_blocks * 36)
                    .map_err(|e| LlmError::Backend(format!("scratch x_q8: {e:?}")))?
            },
            // T246.5.7 — FlashDecode split-K staging.
            gqa_partial_m: stream
                .alloc_zeros::<f32>(cfg.n_q_heads * GQA_N_SPLIT)
                .map_err(|e| LlmError::Backend(format!("scratch gqa_m: {e:?}")))?,
            gqa_partial_l: stream
                .alloc_zeros::<f32>(cfg.n_q_heads * GQA_N_SPLIT)
                .map_err(|e| LlmError::Backend(format!("scratch gqa_l: {e:?}")))?,
            gqa_partial_o: stream
                .alloc_zeros::<half::bf16>(cfg.n_q_heads * GQA_N_SPLIT * cfg.head_dim())
                .map_err(|e| LlmError::Backend(format!("scratch gqa_o: {e:?}")))?,
            // T246.6 — MoE staging (allocated even for Dense to keep struct
            // shape stable ; for Dense the buffers are tiny and unused).
            moe_router_logits: stream
                .alloc_zeros::<half::bf16>(cfg.n_experts.max(1))
                .map_err(|e| LlmError::Backend(format!("scratch moe_router: {e:?}")))?,
            moe_topk_idx: stream
                .alloc_zeros::<i32>(cfg.n_experts_used.max(1))
                .map_err(|e| LlmError::Backend(format!("scratch moe_idx: {e:?}")))?,
            moe_topk_w: stream
                .alloc_zeros::<half::bf16>(cfg.n_experts_used.max(1))
                .map_err(|e| LlmError::Backend(format!("scratch moe_w: {e:?}")))?,
            // T246.8 A4 — sized for the mega-kernel `[k_used, expert_f]`
            // and `[k_used, d]` outputs. Backwards-compatible : the legacy
            // per-slot path uses only the first slot's slice.
            moe_expert_gate: stream
                .alloc_zeros::<half::bf16>(cfg.expert_f.max(1) * cfg.n_experts_used.max(1))
                .map_err(|e| LlmError::Backend(format!("scratch moe_gate: {e:?}")))?,
            moe_expert_up: stream
                .alloc_zeros::<half::bf16>(cfg.expert_f.max(1) * cfg.n_experts_used.max(1))
                .map_err(|e| LlmError::Backend(format!("scratch moe_up: {e:?}")))?,
            moe_expert_out: stream
                .alloc_zeros::<half::bf16>(cfg.d * cfg.n_experts_used.max(1))
                .map_err(|e| LlmError::Backend(format!("scratch moe_out: {e:?}")))?,
            moe_shexp_dot: stream
                .alloc_zeros::<half::bf16>(1)
                .map_err(|e| LlmError::Backend(format!("scratch moe_shexp_dot: {e:?}")))?,
            // T246.7 P1.3 — Lookahead Decoding tree-attention scratch.
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
            // T246.7 P1.4 — multi-token forward scratch.
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
            tree_argmax_host_pinned: unsafe { ctx.alloc_pinned::<u32>(MAX_TREE_SIZE) }
                .map_err(|e| LlmError::Backend(format!("alloc_pinned tree_argmax: {e:?}")))?,

            // ── T246.10 TrackE.2 — Group-GEMM batched MoE staging ──
            moe_router_logits_m: stream
                .alloc_zeros::<half::bf16>(MAX_TREE_SIZE * cfg.n_experts.max(1))
                .map_err(|e| LlmError::Backend(format!("scratch moe_router_m: {e:?}")))?,
            moe_topk_idx_m: stream
                .alloc_zeros::<i32>(MAX_TREE_SIZE * cfg.n_experts_used.max(1))
                .map_err(|e| LlmError::Backend(format!("scratch moe_idx_m: {e:?}")))?,
            moe_topk_w_m: stream
                .alloc_zeros::<half::bf16>(MAX_TREE_SIZE * cfg.n_experts_used.max(1))
                .map_err(|e| LlmError::Backend(format!("scratch moe_w_m: {e:?}")))?,
            moe_expert_gate_m: stream
                .alloc_zeros::<half::bf16>(
                    MAX_TREE_SIZE * cfg.n_experts_used.max(1) * cfg.expert_f.max(1),
                )
                .map_err(|e| LlmError::Backend(format!("scratch moe_gate_m: {e:?}")))?,
            moe_expert_up_m: stream
                .alloc_zeros::<half::bf16>(
                    MAX_TREE_SIZE * cfg.n_experts_used.max(1) * cfg.expert_f.max(1),
                )
                .map_err(|e| LlmError::Backend(format!("scratch moe_up_m: {e:?}")))?,
            moe_expert_out_m: stream
                .alloc_zeros::<half::bf16>(MAX_TREE_SIZE * cfg.n_experts_used.max(1) * cfg.d)
                .map_err(|e| LlmError::Backend(format!("scratch moe_out_m: {e:?}")))?,
            moe_x_q8_m: {
                // Per-(m, slot) Q8_1 staging : (K/32)*36 bytes per pseudo-token.
                // The down projection is launched with M' = M*k_used and K = ef,
                // so the staging needs M*k_used * (ef/32)*36 bytes (worst case).
                // For Qwen3.6-A3B at MAX_TREE_SIZE=512, k_used=8, ef=18944 this
                // is ≈ 88 MB ; fits within our overall 326 MB MoE budget.
                let k_blocks_gate = (cfg.d + 31) / 32;
                let k_blocks_down = (cfg.expert_f.max(1) + 31) / 32;
                // Use the down-proj sizing (it dominates for M*k_used > M).
                let staging_bytes_per_pseudo_tok = k_blocks_down.max(k_blocks_gate) * 36;
                let total_pseudo_tok = MAX_TREE_SIZE * cfg.n_experts_used.max(1);
                stream
                    .alloc_zeros::<u8>(total_pseudo_tok * staging_bytes_per_pseudo_tok)
                    .map_err(|e| LlmError::Backend(format!("scratch moe_x_q8_m: {e:?}")))?
            },
            moe_x_q8_mmq_m: {
                // MMQ-WHOLESALE packed Q8_1 layout : 144 B per 128-elem K-block.
                // The shared-expert gate/up run at K=d, the down at K=ef ; size
                // for the max so the same scratch covers both. For Qwen3.6-A3B
                // (MAX_TREE_SIZE=512, max(d=5120, ef=18944)=18944) this is
                // 512 * (18944/128) * 144 ≈ 10.9 MB. Sized to ceil so K
                // multiples of 128 always fit.
                let k_max = cfg.d.max(cfg.expert_f.max(1));
                let k_blocks_128 = (k_max + 127) / 128;
                stream
                    .alloc_zeros::<u8>(MAX_TREE_SIZE * k_blocks_128 * 144)
                    .map_err(|e| LlmError::Backend(format!("scratch moe_x_q8_mmq_m: {e:?}")))?
            },

            // ── T246.10 TrackE.3 — sort-permutation Group-GEMM scratch ──
            moe_ids_src1_m: stream
                .alloc_zeros::<i32>(MAX_TREE_SIZE * cfg.n_experts_used.max(1))
                .map_err(|e| LlmError::Backend(format!("scratch moe_ids_src1_m: {e:?}")))?,
            moe_ids_dst_m: stream
                .alloc_zeros::<i32>(MAX_TREE_SIZE * cfg.n_experts_used.max(1))
                .map_err(|e| LlmError::Backend(format!("scratch moe_ids_dst_m: {e:?}")))?,
            moe_expert_bounds: stream
                .alloc_zeros::<i32>(cfg.n_experts.max(1) + 1)
                .map_err(|e| LlmError::Backend(format!("scratch moe_expert_bounds: {e:?}")))?,

            // ── T246.7 TrackC — SSM-hybrid Lookahead per-branch buffers ──
            // For SSM-hybrid models (Qwen3.6 Dense / MoE) we pre-allocate one
            // tree-state buffer per SSM layer plus per-tree-row scratch.
            // For pure-transformer variants `cfg.ssm_indices` is empty and
            // these allocations are zero-sized.
            tree_ssm_states: {
                let head_v_dim = cfg.ssm_state;
                let n_v_heads = cfg.ssm_dt_rank;
                let mut v = Vec::with_capacity(cfg.ssm_indices.len());
                for _ in 0..cfg.ssm_indices.len() {
                    let buf = stream
                        .alloc_zeros::<half::bf16>(
                            MAX_TREE_SIZE * n_v_heads * head_v_dim * head_v_dim,
                        )
                        .map_err(|e| LlmError::Backend(format!("scratch tree_ssm_state: {e:?}")))?;
                    v.push(buf);
                }
                v
            },
            tree_conv_states: {
                let mut v = Vec::with_capacity(cfg.ssm_indices.len());
                let conv_kernel_minus_1 = cfg.ssm_conv_kernel.saturating_sub(1);
                for _ in 0..cfg.ssm_indices.len() {
                    let buf = stream
                        .alloc_zeros::<half::bf16>(MAX_TREE_SIZE * conv_kernel_minus_1 * conv_dim)
                        .map_err(|e| {
                            LlmError::Backend(format!("scratch tree_conv_state: {e:?}"))
                        })?;
                    v.push(buf);
                }
                v
            },
            tree_ssm_qkv_mixed: stream
                .alloc_zeros::<half::bf16>(MAX_TREE_SIZE * conv_dim.max(1))
                .map_err(|e| LlmError::Backend(format!("scratch tree_ssm_qkv: {e:?}")))?,
            tree_ssm_conv_out: stream
                .alloc_zeros::<half::bf16>(MAX_TREE_SIZE * conv_dim.max(1))
                .map_err(|e| LlmError::Backend(format!("scratch tree_ssm_conv_out: {e:?}")))?,
            tree_ssm_z: stream
                .alloc_zeros::<half::bf16>(MAX_TREE_SIZE * value_dim.max(1))
                .map_err(|e| LlmError::Backend(format!("scratch tree_ssm_z: {e:?}")))?,
            tree_ssm_alpha: stream
                .alloc_zeros::<half::bf16>(MAX_TREE_SIZE * cfg.ssm_dt_rank.max(1))
                .map_err(|e| LlmError::Backend(format!("scratch tree_ssm_alpha: {e:?}")))?,
            tree_ssm_beta: stream
                .alloc_zeros::<half::bf16>(MAX_TREE_SIZE * cfg.ssm_dt_rank.max(1))
                .map_err(|e| LlmError::Backend(format!("scratch tree_ssm_beta: {e:?}")))?,
            tree_ssm_q_v: stream
                .alloc_zeros::<half::bf16>(MAX_TREE_SIZE * value_dim.max(1))
                .map_err(|e| LlmError::Backend(format!("scratch tree_ssm_q_v: {e:?}")))?,
            tree_ssm_k_v: stream
                .alloc_zeros::<half::bf16>(MAX_TREE_SIZE * value_dim.max(1))
                .map_err(|e| LlmError::Backend(format!("scratch tree_ssm_k_v: {e:?}")))?,
            tree_ssm_out_buf: stream
                .alloc_zeros::<half::bf16>(MAX_TREE_SIZE * value_dim.max(1))
                .map_err(|e| LlmError::Backend(format!("scratch tree_ssm_out_buf: {e:?}")))?,
            tree_ssm_wave_indices: stream
                .alloc_zeros::<i32>(MAX_TREE_SIZE)
                .map_err(|e| LlmError::Backend(format!("scratch tree_ssm_wave: {e:?}")))?,
            // T246.8 A5 — staging for split-K lm_head. Sized for max
            // K_CHUNKS=8 (lm_head K=2048 case). For Qwen3.6-35B-A3B
            // vocab=152064 → 8 * 152064 * 4 = ~4.9 MB. Allocated even when
            // not used (zero cost at runtime unless RUSTORCH_LM_HEAD_SPLIT=1).
            lm_head_split_partial: stream
                .alloc_zeros::<f32>(LM_HEAD_K_CHUNKS_MAX * cfg.vocab)
                .map_err(|e| LlmError::Backend(format!("scratch lm_head_split: {e:?}")))?,
        };

        // T246.5.3 — device-resident counters for CUDA Graph capture.
        let position_dev = stream
            .memcpy_stod(&[0i32])
            .map_err(|e| LlmError::Backend(format!("position_dev: {e:?}")))?;
        let kv_len_dev = stream
            .memcpy_stod(&[1i32])
            .map_err(|e| LlmError::Backend(format!("kv_len_dev: {e:?}")))?;
        let current_token_dev = stream
            .memcpy_stod(&[0u32])
            .map_err(|e| LlmError::Backend(format!("current_token_dev: {e:?}")))?;

        // T246.5.6 — 1× u32 pinned host buffer for async next-token DtoH
        // (allocated before the struct ctor to avoid moving `ctx` early).
        let next_token_host_pinned = unsafe { ctx.alloc_pinned::<u32>(1) }
            .map_err(|e| LlmError::Backend(format!("alloc_pinned next_token: {e:?}")))?;

        // T246.10 TrackI — pinned-host drafts buffer (rewritten each
        // prefill call), sized MAX_TREE_SIZE.
        let prefill_drafts_host_pinned = unsafe { ctx.alloc_pinned::<u32>(MAX_TREE_SIZE) }
            .map_err(|e| LlmError::Backend(format!("alloc_pinned prefill drafts: {e:?}")))?;

        // T246.10 TrackI — pre-baked linear wave-indices [0..MAX_TREE_SIZE]
        // so the SSM per-depth dispatch can use offsets without a per-wave
        // HtoD upload (which would invalidate capture).
        let wave_indices_host: Vec<i32> = (0..MAX_TREE_SIZE as i32).collect();
        let tree_ssm_wave_indices_linear = stream
            .memcpy_stod(&wave_indices_host)
            .map_err(|e| LlmError::Backend(format!("upload wave_indices_linear: {e:?}")))?;

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
            use_dp4a_q4k: std::env::var("RUSTORCH_USE_DP4A_Q4K")
                .map(|v| v != "0" && !v.is_empty())
                .unwrap_or(false),
            // T246.8 A1 — fused SSM kernels default ON ; opt-out via
            // RUSTORCH_SSM_FUSE=0 for A/B parity / bench.
            use_ssm_fuse: std::env::var("RUSTORCH_SSM_FUSE")
                .map(|v| v != "0" && !v.is_empty())
                .unwrap_or(true),
            next_token_host_pinned,
            prefill_graphs: HashMap::new(),
            prefill_warmed: std::collections::HashSet::new(),
            prefill_drafts_host_pinned,
            tree_ssm_wave_indices_linear,
        })
    }

    /// Reset per-conversation state (KV cache + SSM state) to start fresh.
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
        // T246.5.3 — also reset device-side counters.
        self.stream
            .memcpy_htod(&[0i32], &mut self.position_dev)
            .map_err(|e| LlmError::Backend(format!("reset position_dev: {e:?}")))?;
        self.stream
            .memcpy_htod(&[1i32], &mut self.kv_len_dev)
            .map_err(|e| LlmError::Backend(format!("reset kv_len_dev: {e:?}")))?;
        // T246.5.3 — captured graph is no longer valid for the new state.
        // It will be re-captured on the 2nd decode_step after this reset.
        self.decode_graph = None;
        // T246.10 TrackI — the prefill bag-of-graphs cache is preserved
        // across `reset_state()`. Captured graphs reference device-side
        // pointers (KV cache base, SSM state base, position_dev, etc.)
        // which are NOT freed/moved by `reset_state()` — only their
        // contents are zeroed via `memset_zeros`. The captured nodes
        // then read whatever values are in the buffers at replay time
        // (e.g. `position_dev = 0`, `kv_len_dev = 1`), which is exactly
        // the state we re-initialize here. So each replay starts from
        // a clean state without re-capturing. The warmup tracker
        // (`prefill_warmed`) is also preserved : it represents
        // module-load / JIT state which persists across resets.
        Ok(())
    }

    /// T246.8 A5 — Dispatch the lm_head matmul. By default uses the
    /// standard `QuantTensor::dispatch_matmul_m1` (V2 for Q6_K). When
    /// `RUSTORCH_LM_HEAD_SPLIT=1` AND the lm_head is large enough
    /// (`N >= 100_000`) AND stored as Q6_K AND `K` is divisible by
    /// `LM_HEAD_K_CHUNKS_MAX * 256`, the split-K kernel
    /// (`sgemv_q6k_bf16_split_k`) is used instead — targets the 3 ms/tok
    /// floor identified by A3 (note 8e23a850).
    ///
    /// Per-token env-var read is acceptable : called 1-32× per decode and
    /// `std::env::var` is ~150 ns each.
    #[inline]
    fn dispatch_lm_head(&mut self, h_p: u64, logits_p: u64, x_q8_p: u64) -> Result<(), LlmError> {
        // Hot-path : check env once per call (negligible — 150 ns × few/token).
        let split_on = std::env::var("RUSTORCH_LM_HEAD_SPLIT")
            .map(|v| v == "1")
            .unwrap_or(false);
        // Allow K_CHUNKS to be tuned via env (`RUSTORCH_LM_HEAD_SPLIT_K`,
        // default 8 — full split of the 8 super-blocks for lm_head K=2048).
        // Clamped to [1, LM_HEAD_K_CHUNKS_MAX].
        let k_chunks_env: usize = std::env::var("RUSTORCH_LM_HEAD_SPLIT_K")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(LM_HEAD_K_CHUNKS_MAX)
            .clamp(1, LM_HEAD_K_CHUNKS_MAX);
        // Decide split-K eligibility + extract weight pointer before any
        // &mut self borrow on scratch.
        let split_args: Option<(u64, i32, i32, i32)> = if split_on {
            match &self.lm_head {
                QuantTensor::Q6K { bytes, n, k } => {
                    let blocks_per_row = *k / 256;
                    if *n >= 100_000 && blocks_per_row % k_chunks_env == 0 && *k % 256 == 0 {
                        use cudarc::driver::DevicePtr;
                        // SAFETY : device_ptr returns a u64 valid for the
                        // lifetime of the guard ; guard dropped at end of
                        // this expression. Kernel reads only.
                        let (w, _g) = bytes.device_ptr(&self.stream);
                        Some((w, *n as i32, *k as i32, k_chunks_env as i32))
                    } else {
                        None
                    }
                },
                _ => None,
            }
        } else {
            None
        };
        if let Some((w, n, k, k_chunks)) = split_args {
            use cudarc::driver::DevicePtrMut;
            unsafe {
                let (partial_p, _g1) = self
                    .scratch
                    .lm_head_split_partial
                    .device_ptr_mut(&self.stream);
                self.kernels
                    .sgemv_q6k_bf16_split_k(
                        &self.stream,
                        w,
                        h_p,
                        partial_p,
                        logits_p,
                        n,
                        k,
                        k_chunks,
                    )
                    .map_err(|e| LlmError::Backend(format!("lm_head split_k: {e:?}")))?;
            }
            return Ok(());
        }
        // Default path (A4 baseline) — preserved bit-identical when
        // RUSTORCH_LM_HEAD_SPLIT=0 (or unset).
        self.lm_head
            .dispatch_matmul_m1(&self.kernels, &self.stream, h_p, logits_p, x_q8_p)
    }

    /// Decode one token. T246.2-4 implementation : full Qwen3.6 forward
    /// (hybrid SSM + Attention) at M=1, returning next token id.
    pub fn decode_step(&mut self, token_id: u32) -> Result<u32, LlmError> {
        use cudarc::driver::{DevicePtr, DevicePtrMut};

        let cfg = self.config.clone();
        let d = cfg.d;
        let f = cfg.f;
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

        // ---- Use pre-allocated scratch (no per-step alloc, T246.4.1) ----
        let _ = (conv_dim, value_dim, n_v, head_kv, q_dim, kv_dim, f);

        // T246.5.3 — Always update current_token_dev (1 × 4-byte H2D, OUTSIDE
        // any capture region so it executes immediately and remains mutable
        // between graph replays). position_dev and kv_len_dev are self-managing
        // via increment_u32_dev kernels at the end of the body — no need to
        // write them from host after init.
        self.stream
            .memcpy_htod(&[token_id], &mut self.current_token_dev)
            .map_err(|e| LlmError::Backend(format!("upload token id: {e:?}")))?;

        // T246.5.3 — Replay path: if a graph was captured on a previous step,
        // just launch it. Skips ~1908 cuLaunchKernel calls for one replay.
        if let Some(graph) = &self.decode_graph {
            graph
                .launch()
                .map_err(|e| LlmError::Backend(format!("graph launch: {e:?}")))?;
            // T246.5.6 — async DtoH into pinned host buffer ; sync only on
            // the dedicated event when reading the value back.
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

        // T246.5.3 — Capture path: on the 2nd decode call (position == 1),
        // wrap the body in begin_capture / end_capture so all kernels become
        // a single replayable CUDA Graph. The 1st call (position == 0) is a
        // warmup that triggers nvrtc compile + cuModuleLoadData for every
        // kernel — these are NOT capturable and must happen before begin_capture.
        // T246.6 — MoE variant USED to issue memcpy_dtov inside the decode
        // body to read top-K indices to host. That host sync is incompatible
        // with CUDA Graph capture, so capture was previously disabled
        // unconditionally for MoE.
        // T246.8 A2.4 — with `RUSTORCH_MOE_ASYNC=1` the new
        // `moe_ffn_forward_step_async` path eliminates all host syncs in
        // the MoE body (indexed dispatch + devscalar scaled_add + on-device
        // sigmoid). When also `RUSTORCH_MOE_GRAPH=1` (default ON when
        // ASYNC is on), allow capture for MoE too.
        let moe_in_use = matches!(cfg.variant, Qwen35Variant::Moe);
        let moe_capture_ok = !moe_in_use || moe_graph_enabled();
        let should_capture = moe_capture_ok && self.position == 1 && self.decode_graph.is_none();
        if should_capture {
            // T246.5.3 — Drain pending stream work before begin_capture. The
            // H2D upload of token_id above is cuMemcpyHtoDAsync on pageable
            // memory, which is *not* a captureable op. If it bleeds into the
            // capture region the capture is invalidated immediately on the
            // next op (CUDA_ERROR_STREAM_CAPTURE_INVALIDATED).
            self.stream
                .synchronize()
                .map_err(|e| LlmError::Backend(format!("pre-capture sync: {e:?}")))?;
            self.stream
                .begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)
                .map_err(|e| LlmError::Backend(format!("begin_capture: {e:?}")))?;
        }

        // T246.4.2 — zero scratch buffers at start of each step. Without this,
        // residual reads of stale buffers cause non-deterministic output across
        // runs even with the same input.
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
            x_q8_p,
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
            let (u, _g20) = self.final_norm.device_ptr(&self.stream);
            // T246.5.5 — only expose the Q8_1 staging ptr when the dp4a
            // path is enabled (env-gated). Passing 0 forces the float v2
            // fallback in dispatch_matmul_m1.
            let v = if self.use_dp4a_q4k {
                let (p_, _g21) = self.scratch.x_q8_scratch.device_ptr_mut(&self.stream);
                p_
            } else {
                0u64
            };
            // T246.5.7 — FlashDecode split-K staging ptrs.
            let (gm_, _g22) = self.scratch.gqa_partial_m.device_ptr_mut(&self.stream);
            let (gl_, _g23) = self.scratch.gqa_partial_l.device_ptr_mut(&self.stream);
            let (go_, _g24) = self.scratch.gqa_partial_o.device_ptr_mut(&self.stream);
            // T246.6 — MoE staging ptrs.
            let (mr_, _g25) = self.scratch.moe_router_logits.device_ptr_mut(&self.stream);
            let (mi_, _g26) = self.scratch.moe_topk_idx.device_ptr_mut(&self.stream);
            let (mw_, _g27) = self.scratch.moe_topk_w.device_ptr_mut(&self.stream);
            let (mg_, _g28) = self.scratch.moe_expert_gate.device_ptr_mut(&self.stream);
            let (mu_, _g29) = self.scratch.moe_expert_up.device_ptr_mut(&self.stream);
            let (mo_, _g30) = self.scratch.moe_expert_out.device_ptr_mut(&self.stream);
            let (msd_, _g31) = self.scratch.moe_shexp_dot.device_ptr_mut(&self.stream);
            (
                a, b, c, d_, e, f_, g, h_, i, j, k_, l, m, n_, o, p, q_, r, s, t, u, v, gm_, gl_,
                go_, mr_, mi_, mw_, mg_, mu_, mo_, msd_,
            )
        };

        // T246.5.3 — removed `self.stream.synchronize()` here. host syncs are
        // INVALID during CUDA Graph capture. The memset_zeros above are
        // ordered on the same stream as the kernels below, so they execute
        // in order without needing a host barrier.

        // T246.4.3 — DEBUG : if env var set, dump KV cache + SSM state values
        // BEFORE any computation, to verify alloc_zeros gave us actual zeros.
        if self.position == 0 && std::env::var("RUSTORCH_DEBUG_INIT").is_ok() {
            if let Some(kv) = self.kv_caches.first() {
                let k_bytes: Vec<half::bf16> = self
                    .stream
                    .memcpy_dtov(&kv.k)
                    .map_err(|e| LlmError::Backend(format!("dl k: {e:?}")))?;
                let nz = k_bytes
                    .iter()
                    .take(1024)
                    .filter(|&&v| v.to_f32() != 0.0)
                    .count();
                let sample: Vec<f32> = k_bytes.iter().take(8).map(|v| v.to_f32()).collect();
                eprintln!("[debug-init] kv0.k first 8 = {sample:?} ({nz}/1024 nonzero)");
            }
            if let Some(ss) = self.ssm_states.first() {
                let s_bytes: Vec<half::bf16> = self
                    .stream
                    .memcpy_dtov(&ss.state)
                    .map_err(|e| LlmError::Backend(format!("dl state: {e:?}")))?;
                let nz = s_bytes
                    .iter()
                    .take(1024)
                    .filter(|&&v| v.to_f32() != 0.0)
                    .count();
                let sample: Vec<f32> = s_bytes.iter().take(8).map(|v| v.to_f32()).collect();
                eprintln!("[debug-init] ssm0.state first 8 = {sample:?} ({nz}/1024 nonzero)");
            }
            // Also dump token_emb row for token_id=1 to verify deterministic load.
            let te_bytes: Vec<half::bf16> = self
                .stream
                .memcpy_dtov(&self.token_emb)
                .map_err(|e| LlmError::Backend(format!("dl te: {e:?}")))?;
            let row_off = (token_id as usize) * d;
            let sample: Vec<f32> = te_bytes
                .iter()
                .skip(row_off)
                .take(8)
                .map(|v| v.to_f32())
                .collect();
            eprintln!("[debug-init] token_emb[token_id={token_id}] first 8 = {sample:?}");
        }

        // ---- Step 0 : Load h from token_emb[current_token_dev, :] ----
        // T246.5.3 — use embedding_lookup_bf16 with token_id read from device.
        // This makes the lookup CUDA-Graph-capturable (was: host pointer
        // arithmetic on token_id).
        unsafe {
            let (te_p, _g) = self.token_emb.device_ptr(&self.stream);
            let (ct_p, _g2) = self.current_token_dev.device_ptr(&self.stream);
            self.kernels
                .embedding_lookup_bf16(&self.stream, te_p, ct_p, h_p, 1, d as i32)
                .map_err(|e| LlmError::Backend(format!("embed lookup: {e:?}")))?;
        }
        // T246.5.3 — removed sync here, same reason as above.

        // ---- Iterate over all 64 layers ----
        for (li, block) in self.blocks.iter_mut().enumerate() {
            // residual = h
            unsafe {
                self.kernels
                    .copy_bf16(&self.stream, res_p, h_p, d as i32)
                    .map_err(|e| LlmError::Backend(format!("copy res: {e:?}")))?;
            }

            match block {
                BlockQ4K::Ssm(ssm) => {
                    // 1. h_norm = rms_norm(h, attn_norm)
                    unsafe {
                        let (an, _g) = ssm.attn_norm.device_ptr(&self.stream);
                        // rms_norm_bf16 is in-place on x; copy h → h_norm first.
                        self.kernels
                            .copy_bf16(&self.stream, h_norm_p, h_p, d as i32)
                            .map_err(|e| LlmError::Backend(format!("copy h_norm: {e:?}")))?;
                        self.kernels
                            .rms_norm_bf16(&self.stream, h_norm_p, an, eps, d as i32, 1)
                            .map_err(|e| LlmError::Backend(format!("rms_norm: {e:?}")))?;
                    }

                    // 2. qkv_mixed = w_qkv @ h_norm
                    ssm.w_qkv.dispatch_matmul_m1(
                        &self.kernels,
                        &self.stream,
                        h_norm_p,
                        qkv_mixed_p,
                        x_q8_p,
                    )?;

                    // 3. z = w_gate @ h_norm
                    ssm.w_gate.dispatch_matmul_m1(
                        &self.kernels,
                        &self.stream,
                        h_norm_p,
                        z_p,
                        x_q8_p,
                    )?;

                    // 4. alpha = w_alpha @ h_norm
                    ssm.w_alpha.dispatch_matmul_m1(
                        &self.kernels,
                        &self.stream,
                        h_norm_p,
                        alpha_p,
                        x_q8_p,
                    )?;

                    // 5. beta_logit = w_beta @ h_norm ; beta = sigmoid
                    ssm.w_beta.dispatch_matmul_m1(
                        &self.kernels,
                        &self.stream,
                        h_norm_p,
                        beta_p,
                        x_q8_p,
                    )?;
                    // 5b+6. Fused SSM pre-step (T246.8 A1.1) :
                    //   sigmoid(beta) ; alpha += dt_bias ; softplus(alpha) ;
                    //   alpha *= ssm_a → gate_h. Replaces 4 launches with 1
                    //   when RUSTORCH_SSM_FUSE=1 (default), bit-exact w/ unfused.
                    unsafe {
                        let (db, _g) = ssm.dt_bias.device_ptr(&self.stream);
                        let (sa, _g2) = ssm.ssm_a.device_ptr(&self.stream);
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
                                .map_err(|e| LlmError::Backend(format!("alpha+dt_bias: {e:?}")))?;
                            self.kernels
                                .softplus_inplace_bf16(&self.stream, alpha_p, n_v as i32)
                                .map_err(|e| LlmError::Backend(format!("softplus: {e:?}")))?;
                            self.kernels
                                .mul_inplace_bf16(&self.stream, alpha_p, sa, n_v as i32)
                                .map_err(|e| LlmError::Backend(format!("mul ssm_a: {e:?}")))?;
                        }
                    }
                    // alpha is now gate_h (alpha_softplus * ssm_a).

                    // 7. conv_out = conv1d_depthwise(conv1d, conv_state, qkv_mixed)
                    let ssm_state_layer = &mut self.ssm_states[get_ssm_layer_idx(&cfg, li)];
                    unsafe {
                        let (cw, _g) = ssm.conv1d.device_ptr(&self.stream);
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

                    // 10. l2_norm_per_head on q (n_k heads of head_kv) and k
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

                    // 11. Broadcast q,k from n_k to n_v heads (factor n_v / n_k)
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

                    // 12. delta_net_step : state update + out
                    unsafe {
                        let (st, _g) = ssm_state_layer.state.device_ptr_mut(&self.stream);
                        self.kernels
                            .delta_net_step_bf16(
                                &self.stream,
                                q_for_delta,
                                k_for_delta,
                                v_ptr,
                                alpha_p, // gate_h
                                beta_p,
                                st,
                                sso_p,
                                n_v as i32,
                                head_kv as i32,
                            )
                            .map_err(|e| LlmError::Backend(format!("delta_net: {e:?}")))?;
                    }

                    // 13. ssm_norm per head + multiply by silu(z) :
                    //   Fused (T246.8 A1.2) : ssm_post_step_bf16 replaces
                    //   rms_norm_bf16 + silu_bf16 + mul_inplace_bf16 (3→1
                    //   launches) when RUSTORCH_SSM_FUSE=1 (default).
                    unsafe {
                        let (sn, _g) = ssm.ssm_norm.device_ptr(&self.stream);
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
                            // RMSNorm per head : we have rms_norm_bf16 with batch parameter.
                            // Use n_v batches, each of size head_kv, with same gamma.
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

                    // 14. h = ssm_out @ gated  (overwrites h)
                    ssm.ssm_out.dispatch_matmul_m1(
                        &self.kernels,
                        &self.stream,
                        sso_p,
                        h_p,
                        x_q8_p,
                    )?;

                    // 15. residual : h += residual
                    unsafe {
                        self.kernels
                            .add_inplace_bf16(&self.stream, h_p, res_p, d as i32)
                            .map_err(|e| LlmError::Backend(format!("ssm residual: {e:?}")))?;
                    }
                },
                BlockQ4K::Attn(attn) => {
                    // T246.3 — full attention block forward at M=1.
                    // P1.5 — Qwen2 path : no Q+gate, has Q/K/V biases, no QK-norm,
                    // no sigmoid output gate. Branch on cfg.variant flags.

                    // 1. h_norm = rms_norm(h, attn_norm)
                    unsafe {
                        let (an, _g) = attn.attn_norm.device_ptr(&self.stream);
                        self.kernels
                            .copy_bf16(&self.stream, h_norm_p, h_p, d as i32)
                            .map_err(|e| LlmError::Backend(format!("copy h_norm attn: {e:?}")))?;
                        self.kernels
                            .rms_norm_bf16(&self.stream, h_norm_p, an, eps, d as i32, 1)
                            .map_err(|e| LlmError::Backend(format!("rms_norm attn: {e:?}")))?;
                    }

                    // 2. Q projection. With Q+gate (Qwen3Next) it produces a
                    //    `2*q_dim` blob that we split. Without it (qwen2 / qwen3
                    //    pure transformer) it produces just `q_dim` directly.
                    if cfg.variant.attn_has_output_gate() {
                        let (qg_n, _qg_k) = attn.w_q.shape();
                        debug_assert_eq!(qg_n, 2 * q_dim);
                        // Reuse `up_buf` (size f=17408 ≥ 2*q_dim) as QG scratch.
                        let qg_p = up_p;
                        attn.w_q.dispatch_matmul_m1(
                            &self.kernels,
                            &self.stream,
                            h_norm_p,
                            qg_p,
                            x_q8_p,
                        )?;
                        // 3. Split qg into q (q_dim) and gate (q_dim, used as sigmoid gate).
                        //    Use q_buf and (reuse) attn_out as gate buffer.
                        unsafe {
                            self.kernels
                                .split_qg_bf16(
                                    &self.stream,
                                    qg_p,
                                    q_p,
                                    ao_p, // store gate here temporarily
                                    n_q as i32,
                                    head_dim as i32,
                                )
                                .map_err(|e| LlmError::Backend(format!("split_qg: {e:?}")))?;
                        }
                    } else {
                        attn.w_q.dispatch_matmul_m1(
                            &self.kernels,
                            &self.stream,
                            h_norm_p,
                            q_p,
                            x_q8_p,
                        )?;
                    }

                    // 4. K, V projections.
                    attn.w_k.dispatch_matmul_m1(
                        &self.kernels,
                        &self.stream,
                        h_norm_p,
                        k_p,
                        x_q8_p,
                    )?;
                    attn.w_v.dispatch_matmul_m1(
                        &self.kernels,
                        &self.stream,
                        h_norm_p,
                        v_p,
                        x_q8_p,
                    )?;

                    // 4b. Qwen2 — add QKV biases.
                    if cfg.variant.attn_has_qkv_bias() {
                        unsafe {
                            let (bq, _gbq) = attn
                                .b_q
                                .as_ref()
                                .expect("qwen2: b_q present")
                                .device_ptr(&self.stream);
                            let (bk, _gbk) = attn
                                .b_k
                                .as_ref()
                                .expect("qwen2: b_k present")
                                .device_ptr(&self.stream);
                            let (bv, _gbv) = attn
                                .b_v
                                .as_ref()
                                .expect("qwen2: b_v present")
                                .device_ptr(&self.stream);
                            self.kernels
                                .add_inplace_bf16(&self.stream, q_p, bq, q_dim as i32)
                                .map_err(|e| LlmError::Backend(format!("q_bias: {e:?}")))?;
                            self.kernels
                                .add_inplace_bf16(&self.stream, k_p, bk, kv_dim as i32)
                                .map_err(|e| LlmError::Backend(format!("k_bias: {e:?}")))?;
                            self.kernels
                                .add_inplace_bf16(&self.stream, v_p, bv, kv_dim as i32)
                                .map_err(|e| LlmError::Backend(format!("v_bias: {e:?}")))?;
                        }
                    }

                    // 5. Per-head Q-norm and K-norm (RMSNorm with shared gamma).
                    //    Skipped for qwen2 (no QK-norm).
                    if cfg.variant.attn_has_qk_norm() {
                        unsafe {
                            let (qn, _g1) = attn
                                .q_norm
                                .as_ref()
                                .expect("qk_norm variant: q_norm present")
                                .device_ptr(&self.stream);
                            let (kn, _g2) = attn
                                .k_norm
                                .as_ref()
                                .expect("qk_norm variant: k_norm present")
                                .device_ptr(&self.stream);
                            // rms_norm_bf16(x, gamma, eps, n, batch) where each batch is size n.
                            self.kernels
                                .rms_norm_bf16(
                                    &self.stream,
                                    q_p,
                                    qn,
                                    eps,
                                    head_dim as i32,
                                    n_q as i32,
                                )
                                .map_err(|e| LlmError::Backend(format!("q_norm: {e:?}")))?;
                            self.kernels
                                .rms_norm_bf16(
                                    &self.stream,
                                    k_p,
                                    kn,
                                    eps,
                                    head_dim as i32,
                                    n_kv as i32,
                                )
                                .map_err(|e| LlmError::Backend(format!("k_norm: {e:?}")))?;
                        }
                    }

                    // 6. RoPE on q (n_q heads) and k (n_kv heads).
                    // Qwen3.6 : rope_dim=64 < head_dim=256. Only first 64 dims rotate.
                    // T246.5.3 — devcnt variant reads `pos` from position_dev.
                    let position = self.position;
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

                    // 7. Append k, v to KV cache at slot `*position_dev`.
                    // T246.5.3 — single graph-capturable kernel handles both
                    // K and V, with slot index read from device pointer.
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

                    // 8. GQA decode — T246.5.7 FlashDecode-V2 split-K kernel
                    // (4× more SM occupancy than the serial online kernel by
                    // splitting kv_len work across GQA_N_SPLIT blocks/head).
                    let _ = position; // kept for the host-side KV append above
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
                                gate_p, // attn raw output (size f ≥ q_dim)
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

                    // 9. sigmoid(gate) ; attn_out *= gate
                    //    Qwen2 has no output gate, so just skip this step.
                    if cfg.variant.attn_has_output_gate() {
                        unsafe {
                            self.kernels
                                .sigmoid_inplace_bf16(&self.stream, ao_p, q_dim as i32)
                                .map_err(|e| LlmError::Backend(format!("sigmoid gate: {e:?}")))?;
                            self.kernels
                                .mul_inplace_bf16(&self.stream, gate_p, ao_p, q_dim as i32)
                                .map_err(|e| LlmError::Backend(format!("attn*gate: {e:?}")))?;
                        }
                    }

                    // 10. h = w_o @ gated_attn  (output dim d)
                    attn.w_o.dispatch_matmul_m1(
                        &self.kernels,
                        &self.stream,
                        gate_p,
                        h_p,
                        x_q8_p,
                    )?;

                    // 11. residual : h += residual
                    unsafe {
                        self.kernels
                            .add_inplace_bf16(&self.stream, h_p, res_p, d as i32)
                            .map_err(|e| LlmError::Backend(format!("attn residual: {e:?}")))?;
                    }
                },
            }

            // ---- FFN ----
            let (ffn_ref, post_norm) = match block {
                BlockQ4K::Ssm(s) => (&s.ffn, &s.post_norm),
                BlockQ4K::Attn(a) => (&a.ffn, &a.post_norm),
            };
            // residual = h (pre-FFN value, after attn/ssm + first residual)
            unsafe {
                self.kernels
                    .copy_bf16(&self.stream, res_p, h_p, d as i32)
                    .map_err(|e| LlmError::Backend(format!("copy res ffn: {e:?}")))?;
            }
            // h_norm = rms_norm(h, post_norm)
            unsafe {
                let (pn, _g) = post_norm.device_ptr(&self.stream);
                self.kernels
                    .copy_bf16(&self.stream, h_norm_p, h_p, d as i32)
                    .map_err(|e| LlmError::Backend(format!("copy h_norm ffn: {e:?}")))?;
                self.kernels
                    .rms_norm_bf16(&self.stream, h_norm_p, pn, eps, d as i32, 1)
                    .map_err(|e| LlmError::Backend(format!("rms_norm post: {e:?}")))?;
            }
            match ffn_ref {
                FfnQ4K::Dense { gate, up, down } => {
                    // gate = w_gate @ h_norm ; up = w_up @ h_norm ; gate = silu(gate) * up
                    gate.dispatch_matmul_m1(&self.kernels, &self.stream, h_norm_p, gate_p, x_q8_p)?;
                    up.dispatch_matmul_m1(&self.kernels, &self.stream, h_norm_p, up_p, x_q8_p)?;
                    unsafe {
                        self.kernels
                            .swiglu_bf16(&self.stream, gate_p, up_p, gate_p, f as i32)
                            .map_err(|e| LlmError::Backend(format!("swiglu: {e:?}")))?;
                    }
                    // h = w_down @ gate
                    down.dispatch_matmul_m1(&self.kernels, &self.stream, gate_p, h_p, x_q8_p)?;
                },
                FfnQ4K::Moe(moe) => {
                    if moe_mega_enabled() {
                        moe_ffn_forward_step_mega(
                            moe,
                            &self.kernels,
                            &self.stream,
                            &cfg,
                            h_norm_p,
                            h_p,
                            x_q8_p,
                            moe_router_p,
                            moe_idx_p,
                            moe_w_p,
                            moe_egate_p,
                            moe_eup_p,
                            moe_eout_p,
                            moe_sd_p,
                        )?;
                    } else if moe_async_enabled() {
                        moe_ffn_forward_step_async(
                            moe,
                            &self.kernels,
                            &self.stream,
                            &cfg,
                            h_norm_p,
                            h_p,
                            x_q8_p,
                            moe_router_p,
                            moe_idx_p,
                            moe_w_p,
                            moe_egate_p,
                            moe_eup_p,
                            moe_eout_p,
                            moe_sd_p,
                        )?;
                    } else {
                        moe_ffn_forward_step(
                            moe,
                            &self.kernels,
                            &self.stream,
                            &cfg,
                            h_norm_p,
                            h_p,
                            x_q8_p,
                            moe_router_p,
                            moe_idx_p,
                            &self.scratch.moe_topk_idx,
                            moe_w_p,
                            &self.scratch.moe_topk_w,
                            moe_egate_p,
                            moe_eup_p,
                            moe_eout_p,
                            moe_sd_p,
                            &self.scratch.moe_shexp_dot,
                        )?;
                    }
                },
            }
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
        // T246.8 A5 — split-K dispatch (env-gated via RUSTORCH_LM_HEAD_SPLIT=1).
        self.dispatch_lm_head(h_p, logits_p, x_q8_p)?;

        // ---- Sample (argmax for now) ----
        unsafe {
            self.kernels
                .argmax_bf16(&self.stream, logits_p, tok_p, cfg.vocab as i32)
                .map_err(|e| LlmError::Backend(format!("argmax: {e:?}")))?;
        }
        let _ = (tok_p, logits_p, final_norm_p); // silence unused warnings

        // T246.5.3 — Auto-advance device counters at the end of body. These
        // become part of the captured graph so each replay also advances the
        // counters, eliminating the need for host-side writes between replays.
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

        // T246.5.3 — Capture path closure: end_capture + first launch.
        if should_capture {
            let graph = self
                .stream
                .end_capture(
                    CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
                )
                .map_err(|e| LlmError::Backend(format!("end_capture: {e:?}")))?
                .ok_or_else(|| LlmError::Backend("end_capture returned no graph".into()))?;
            // Launch once now to actually execute the recorded body for this step.
            graph
                .launch()
                .map_err(|e| LlmError::Backend(format!("first graph launch: {e:?}")))?;
            self.decode_graph = Some(graph);
        }

        // Read back next token (host op, OUTSIDE any capture). T246.5.6 —
        // async DtoH to pinned host memory + event-based sync.
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

    /// T246.10 A6 — Process a prompt of `N` tokens at once and return the
    /// first decode token (i.e. the model's argmax after seeing all N
    /// inputs). Uses the existing tree-attention infrastructure with a
    /// linear-chain topology (`parents = [-1, 0, 1, .., N-2]`,
    /// `depths = [0, 1, .., N-1]`).
    ///
    /// **Constraints**
    /// - `1 ≤ token_ids.len() ≤ MAX_TREE_SIZE` (512 since T246.10 A6 bump).
    /// - `start_pos` must equal `self.position` AND device counter state
    ///   (`*position_dev`, `*kv_len_dev`). At the moment we only support
    ///   `start_pos == self.position` (incremental prefill from current
    ///   state). The model must NOT have any in-flight CUDA Graph capture
    ///   (decode-only graphs are invalidated and rebuilt on the next
    ///   `decode_step`).
    /// - Variant : works for `Qwen2PureTransformer`, `Qwen3PureTransformer`,
    ///   `Dense`, `Moe` — all four variants.
    ///
    /// **Semantics** : after this call the KV cache contains the N appended
    /// tokens, the SSM state advances by N (per-layer recurrence on the
    /// chain), `self.position += N`, and the returned u32 is the predicted
    /// next token (`argmax(logits[token_ids[N-1]])`) which is bit-equivalent
    /// (or 1 ULP BF16) to running `decode_step` N times in a loop.
    ///
    /// **CUDA Graph interaction** : tree forwards bypass graph capture by
    /// design (decision RFC D4 in note f0e68045). The decode-only graph
    /// captured by earlier `decode_step` calls is preserved : it reads
    /// device counters via pointer indirection so its validity is unchanged
    /// by the counter advance we perform here.
    pub fn prefill_tokens(&mut self, token_ids: &[u32], start_pos: usize) -> Result<u32, LlmError> {
        let n = token_ids.len();
        if n == 0 {
            return Err(LlmError::Backend(
                "prefill_tokens: token_ids must be non-empty".into(),
            ));
        }
        if n > MAX_TREE_SIZE {
            return Err(LlmError::Backend(format!(
                "prefill_tokens: N={n} exceeds MAX_TREE_SIZE={MAX_TREE_SIZE}. \
                 Call prefill_tokens in chunks of {MAX_TREE_SIZE} for longer prompts."
            )));
        }
        if start_pos != self.position {
            return Err(LlmError::Backend(format!(
                "prefill_tokens: start_pos={start_pos} != self.position={}. \
                 Call model.reset_state() first or pass start_pos=self.position().",
                self.position
            )));
        }

        // ── Single-token fast path ──────────────────────────────────────
        // For N=1, prefill is equivalent to a single decode_step. Use
        // decode_step directly (preserves the captured graph win).
        if n == 1 {
            return self.decode_step(token_ids[0]);
        }

        // ── Linear-chain tree descriptors ───────────────────────────────
        // parents = [-1, 0, 1, ..., n-2], depths = [0, 1, ..., n-1].
        let parents: Vec<i32> = std::iter::once(-1i32).chain((0..(n - 1) as i32)).collect();
        let depths: Vec<u16> = (0..n as u16).collect();

        // ── T246.10 TrackI — CUDA Graph capture / replay path ──────────
        //
        // When `RUSTORCH_PREFILL_GRAPH=1` and `n` is a previously-captured
        // (or will-be-captured) bucket size, we route through the bag-of-
        // graphs cache (`self.prefill_graphs`). The first call with a
        // given `n` runs the body uncaptured (warmup — JIT all kernels)
        // ; the second runs it again under `begin_capture` /
        // `end_capture`, instantiates a `cudaGraph_t`, caches it, and
        // launches it. Every subsequent call with the same `n` replays
        // the captured graph in a single `cuGraphLaunch` (≈ 5-10 µs CPU
        // dispatch vs ~10 s on the per-launch path per PRE-FLIGHT note
        // 1ac7de4a).
        //
        // Pre-conditions (mirrors the decode-side graph capture in
        // `decode_step` and the llama.cpp `TAG_MUL_MAT_ID_CUDA_GRAPHS`
        // compatibility check) :
        //
        // 1. The MoE body must be sync-free → `RUSTORCH_MOE_ASYNC=1`
        //    AND (`RUSTORCH_MOE_MEGA=1` OR `RUSTORCH_MOE_GROUP_GEMM=1`).
        // 2. The GEMM-prefill path is required so the matmul calls go
        //    through `dispatch_matmul_mvar` (no host-side shape
        //    branching inside the body) →  `RUSTORCH_GEMM_PREFILL=1`.
        // 3. n must be ≥ `GEMM_PREFILL_MIN_M` (otherwise the body
        //    branches into per-row dispatch, which is correct but not
        //    the design target of TrackI).
        //
        // When any condition fails (or the env gate is off) we fall
        // through to the legacy path below, preserving the
        // `RUSTORCH_PREFILL_GRAPH=0` baseline bit-exactly.
        let variant = self.config.variant;
        let moe_ok = !matches!(variant, Qwen35Variant::Moe)
            || (moe_async_enabled() && (moe_mega_enabled() || moe_group_gemm_enabled()));
        let graph_eligible =
            prefill_graph_enabled() && moe_ok && gemm_prefill_enabled() && n >= GEMM_PREFILL_MIN_M;

        if graph_eligible {
            return self.prefill_tokens_capture(token_ids, &parents, &depths);
        }

        // ── Dispatch by variant (legacy / RUSTORCH_PREFILL_GRAPH=0 path) ─
        let accepted = match variant {
            Qwen35Variant::Qwen2PureTransformer | Qwen35Variant::Qwen3PureTransformer => {
                self.decode_step_tree_pure_transformer_inner(token_ids, &parents, &depths, true)?
            },
            Qwen35Variant::Dense | Qwen35Variant::Moe => {
                self.decode_step_tree_hybrid_inner(token_ids, &parents, &depths, true)?
            },
        };

        if accepted.len() != n {
            return Err(LlmError::Backend(format!(
                "prefill_tokens: tree forward returned {} accepted tokens, \
                 expected {} (force_accept_all bug ?)",
                accepted.len(),
                n
            )));
        }
        // The token at position N-1 of accepted[] is `argmax(logits)` after
        // seeing all N input tokens — that's the "first decode token" we
        // return to the caller.
        Ok(accepted[n - 1])
    }

    /// T246.10 TrackI — Graph-capture entry for `prefill_tokens`. Caller
    /// must validate `n >= GEMM_PREFILL_MIN_M`, MoE async/mega/group-gemm
    /// invariants, etc. See `prefill_tokens` for the env-gate logic.
    ///
    /// Strategy : bag-of-graphs keyed by `n`. First call with a given
    /// `n` runs the body uncaptured (warmup — primes nvrtc compile +
    /// CUmodule load for every kernel). Second call runs under
    /// `begin_capture` / `end_capture`, instantiates the graph, caches
    /// it under `n`. Every subsequent call replays the cached graph in
    /// a single launch.
    fn prefill_tokens_capture(
        &mut self,
        token_ids: &[u32],
        parents: &[i32],
        depths: &[u16],
    ) -> Result<u32, LlmError> {
        use cudarc::driver::sys::{CUgraphInstantiate_flags, CUstreamCaptureMode};
        let n = token_ids.len();

        // ── 1. Upload tree descriptors (capture-safe, pinned source) ─────
        //
        // The captured graph reads `tree_drafts` / `tree_parents` /
        // `tree_depths` by pointer. We upload the per-call values into
        // those device buffers OUTSIDE the capture region — pageable
        // HtoD inside the region would degenerate into a synchronous
        // copy and abort the capture
        // (`CUDA_ERROR_STREAM_CAPTURE_INVALIDATED`). The source must
        // come from pinned host memory so the upload is truly async
        // and does not implicitly drain the stream.
        //
        // For `parents` / `depths` on a linear-chain prefill the values
        // are constant in `n` (parents = [-1, 0, .., n-2], depths =
        // [0, 1, .., n-1]) ; the caller passes them in. We accept the
        // small cost of one un-pinned upload per call for these two —
        // it happens BEFORE begin_capture and the explicit pre-capture
        // `synchronize()` below absorbs any implicit driver sync. Only
        // `drafts` is re-uploaded every call (the token content varies),
        // so pinning matters most for it.
        {
            let mut drafts_pinned = self
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
                .map_err(|e| LlmError::Backend(format!("upload tree_drafts (capture): {e:?}")))?;
        }
        self.stream
            .memcpy_htod(parents, &mut self.scratch.tree_parents)
            .map_err(|e| LlmError::Backend(format!("upload tree_parents (capture): {e:?}")))?;
        self.stream
            .memcpy_htod(depths, &mut self.scratch.tree_depths)
            .map_err(|e| LlmError::Backend(format!("upload tree_depths (capture): {e:?}")))?;

        // ── 2. Replay path — graph cached for this n ────────────────────
        let variant = self.config.variant;
        if let Some(graph) = self.prefill_graphs.get(&n) {
            graph
                .launch()
                .map_err(|e| LlmError::Backend(format!("prefill graph launch: {e:?}")))?;
            // After replay, the device-side `argmax` buffer has N entries
            // ; DtoH them to the pinned host buffer.
            return self.prefill_finish_after_capture(n);
        }

        // ── 3. Warmup path — first call for this n, no capture ──────────
        // Run the body uncaptured to ensure all kernels JIT-compile and
        // load before we attempt to capture. The body MUST run on the
        // same stream as the eventual capture (it does — `self.stream`).
        // Mirrors llama.cpp's `warmup_complete` two-call pattern.
        if !self.prefill_warmed.contains(&n) {
            // Standard (non-capture) prefill body. We've already
            // uploaded the descriptors above ; the inner re-uploads
            // them (cheap, ~12 µs) which is fine for the warmup pass.
            let accepted = match variant {
                Qwen35Variant::Qwen2PureTransformer | Qwen35Variant::Qwen3PureTransformer => {
                    self.decode_step_tree_pure_transformer_inner(token_ids, parents, depths, true)?
                },
                Qwen35Variant::Dense | Qwen35Variant::Moe => {
                    self.decode_step_tree_hybrid_inner(token_ids, parents, depths, true)?
                },
            };
            if accepted.len() != n {
                return Err(LlmError::Backend(format!(
                    "prefill_tokens (warmup): tree forward returned {} accepted tokens, \
                     expected {} (force_accept_all bug ?)",
                    accepted.len(),
                    n
                )));
            }
            // Mark warmup complete — the next call with this n will
            // go down the capture path.
            self.prefill_warmed.insert(n);
            return Ok(accepted[n - 1]);
        }

        // ── 4. Capture path — second call, instantiate graph ────────────
        // Drain any pending stream work : ensures the tree-descriptor HtoD
        // uploads above are fully retired before we transition the stream
        // into capture mode.
        self.stream
            .synchronize()
            .map_err(|e| LlmError::Backend(format!("pre-capture sync: {e:?}")))?;
        self.stream
            .begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)
            .map_err(|e| LlmError::Backend(format!("prefill begin_capture: {e:?}")))?;

        // Run the *capturable* body. The inner function knows to skip
        // the HtoD descriptors (already done), use linear wave-indices
        // offsets, skip the final DtoH + accept walk, and skip the
        // host-side `self.position +=` (we do it post-launch).
        let _accepted = match variant {
            Qwen35Variant::Qwen2PureTransformer | Qwen35Variant::Qwen3PureTransformer => self
                .decode_step_tree_pure_transformer_inner_capture(
                    token_ids, parents, depths, true, true,
                )?,
            Qwen35Variant::Dense | Qwen35Variant::Moe => {
                self.decode_step_tree_hybrid_inner_capture(token_ids, parents, depths, true, true)?
            },
        };

        let graph = self
            .stream
            .end_capture(CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH)
            .map_err(|e| LlmError::Backend(format!("prefill end_capture: {e:?}")))?
            .ok_or_else(|| LlmError::Backend("prefill end_capture returned no graph".into()))?;

        // Launch the captured graph once to materialize the recorded
        // work for this call (which is what the user actually asked
        // for — we promised to do the prefill, not just record it).
        graph
            .launch()
            .map_err(|e| LlmError::Backend(format!("first prefill graph launch: {e:?}")))?;

        self.prefill_graphs.insert(n, graph);
        self.prefill_finish_after_capture(n)
    }

    /// T246.10 TrackI — post-replay finalizer. Reads the argmax tokens
    /// into pinned host memory, advances the host-side `self.position`,
    /// and returns the last accepted token (the "first decode token").
    fn prefill_finish_after_capture(&mut self, n: usize) -> Result<u32, LlmError> {
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
        // Capture mode never updates `self.position` from inside the
        // inner — apply it here for the N accepted tokens.
        self.position += n;
        Ok(argmax_host[n - 1])
    }

    /// T246.7 P1.3d — tree-aware decode step for Lookahead Decoding (RFC
    /// f0e68045, design D1).
    ///
    /// Takes a draft tree encoded as a flat BFS-ordered array :
    /// - `drafts[r]`  = the token id at tree node r ; `drafts[0]` is the
    ///   "seed" (the most-recently-accepted token from the previous step).
    /// - `parents[r]` = parent node index in the tree, root (r=0) has -1.
    /// - `depths[r]`  = depth of node r (root depth = 0).
    ///
    /// All inputs must have the same length, ≤ `MAX_TREE_SIZE`. Returns the
    /// accepted token suffix `[1..=accept_len]` ; the seed (`drafts[0]`)
    /// itself is NOT returned (it was already in the model's KV state).
    ///
    /// **Phase 1 scope** : tree_size = 1 is a strict drop-in replacement for
    /// `decode_step(drafts[0])` — full kernel parity (including CUDA Graph
    /// replay) and bit-exact output. tree_size > 1 returns `LlmError::Backend`
    /// for now ; the multi-branch forward + acceptance walk lands in P1.4
    /// alongside the full Lookahead manager loop. The CUDA primitives needed
    /// for tree_size > 1 (`kv_append_tree_bf16`, `gqa_decode_tree_bf16`,
    /// `argmax_logits_tree_bf16`, `add_u32_dev`, `set_u32_dev`) are already
    /// in place ; only the per-tree-token forward orchestration is deferred.
    ///
    /// Per RFC design decisions :
    /// - D1 : NEW method, never modifies `decode_step`.
    /// - D2 : write-then-truncate KV (no scratch cache copy).
    /// - D4 : no graph capture for verify (the inner `decode_step` call
    ///   may use its own captured graph for tree_size=1 ; tree_size>1
    ///   would explicitly disable capture).
    pub fn decode_step_tree(
        &mut self,
        drafts: &[u32],
        parents: &[i32],
        depths: &[u16],
    ) -> Result<Vec<u32>, LlmError> {
        // ── Validation ───────────────────────────────────────────────────
        if drafts.is_empty() {
            return Err(LlmError::Backend(
                "decode_step_tree: drafts must be non-empty".into(),
            ));
        }
        if drafts.len() != parents.len() || drafts.len() != depths.len() {
            return Err(LlmError::Backend(format!(
                "decode_step_tree: length mismatch — drafts={}, parents={}, depths={}",
                drafts.len(),
                parents.len(),
                depths.len()
            )));
        }
        if drafts.len() > MAX_TREE_SIZE {
            return Err(LlmError::Backend(format!(
                "decode_step_tree: tree_size {} exceeds MAX_TREE_SIZE {}",
                drafts.len(),
                MAX_TREE_SIZE
            )));
        }
        if parents[0] != -1 {
            return Err(LlmError::Backend(format!(
                "decode_step_tree: root must have parent=-1, got {}",
                parents[0]
            )));
        }
        if depths[0] != 0 {
            return Err(LlmError::Backend(format!(
                "decode_step_tree: root must have depth=0, got {}",
                depths[0]
            )));
        }
        for (r, &p) in parents.iter().enumerate().skip(1) {
            if p < 0 || (p as usize) >= r {
                return Err(LlmError::Backend(format!(
                    "decode_step_tree: parents[{r}] = {p} must be in [0, {r})"
                )));
            }
            let expected_depth = depths[p as usize] + 1;
            if depths[r] != expected_depth {
                return Err(LlmError::Backend(format!(
                    "decode_step_tree: depths[{r}] = {} must equal depths[parent={}] + 1 = {}",
                    depths[r], p, expected_depth
                )));
            }
        }

        let tree_size = drafts.len();

        // ── tree_size = 1 : delegate to decode_step (bit-exact equivalence) ──
        // The root's own KV is at slot kv_len-1, fully covered by Phase A
        // of the existing M=1 attention path. The "tree" semantics
        // degenerate to a single forward pass on drafts[0], whose output
        // we return as the accepted token suffix of length 1.
        if tree_size == 1 {
            let tok = self.decode_step(drafts[0])?;
            return Ok(vec![tok]);
        }

        // ── tree_size > 1 : dispatch by variant ──────────────────────────
        // Pure-transformer  → decode_step_tree_pure_transformer (P1.4a).
        // Dense (SSM+attn)  → decode_step_tree_hybrid (T246.7 TrackC.3).
        // MoE  (SSM+attn)   → not yet supported (Phase 2 follow-up : the
        //                     MoE router introduces host-sync per layer
        //                     which interacts badly with the per-row
        //                     forward).
        match self.config.variant {
            Qwen35Variant::Qwen2PureTransformer | Qwen35Variant::Qwen3PureTransformer => {
                self.decode_step_tree_pure_transformer(drafts, parents, depths)
            },
            Qwen35Variant::Dense => self.decode_step_tree_hybrid(drafts, parents, depths),
            Qwen35Variant::Moe => Err(LlmError::Backend(format!(
                "decode_step_tree multi-token (tree_size={tree_size}): MoE+SSM \
                 hybrid Lookahead deferred to a later task — see note f0477041. \
                 Set RUSTORCH_LOOKAHEAD=0 to fall back to decode_step."
            ))),
        }
    }

    /// T246.7 TrackC.3 — multi-token tree forward for the SSM-hybrid Dense
    /// variant (Qwen3.6-27B). Mirrors `decode_step_tree_pure_transformer`
    /// but with two extra responsibilities :
    ///
    /// 1. **Per-branch SSM state forking** — for each SSM layer we
    ///    pre-load the model's current per-layer state into the root
    ///    slot of `tree_ssm_states[layer]` and `tree_conv_states[layer]`,
    ///    then run the per-row SSM forward where each tree node reads
    ///    from its parent's slot via `delta_net_step_tree_bf16` and a
    ///    parent-aware conv1d sequence.
    ///
    /// 2. **Commit on accept** — after the acceptance walk we copy the
    ///    deepest accepted node's SSM state and conv state back into the
    ///    model's per-layer scratch, dropping the discarded branches.
    ///
    /// Conv1d state forking uses the per-row sequential strategy (see
    /// the deferred-design note f0477041) : we copy parent → child slot
    /// in `tree_conv_states[layer]` then call the existing scalar
    /// `conv1d_depthwise_bf16` on slot `r`. For up to MAX_TREE_SIZE=32
    /// nodes this is 32 small kernel launches per SSM layer per call.
    /// `delta_net_step_tree_bf16` itself handles forking natively but
    /// requires one launch per BFS depth wave (read-after-write barrier
    /// across waves). On Qwen3.6-27B (48 SSM layers, max depth ~7)
    /// this is ~336 small kernel launches — driver-overhead-bound, not
    /// compute-bound.
    #[allow(clippy::too_many_lines)]
    fn decode_step_tree_hybrid(
        &mut self,
        drafts: &[u32],
        parents: &[i32],
        depths: &[u16],
    ) -> Result<Vec<u32>, LlmError> {
        self.decode_step_tree_hybrid_inner(drafts, parents, depths, false)
    }

    /// T246.10 A6 — shared implementation behind `decode_step_tree_hybrid`
    /// and `prefill_tokens` (hybrid path). When `force_accept_all` is true,
    /// skip the acceptance walk and force-accept every BFS-ordered tree node.
    /// This is the prefill mode (linear-chain trees committed verbatim).
    ///
    /// T246.10 TrackI — when `in_prefill_capture` is true (only valid with
    /// `force_accept_all`), the function executes the *capturable body
    /// only* :
    ///
    /// - Tree-descriptor HtoD uploads (drafts/parents/depths) are skipped
    ///   (the caller pre-uploaded them OUTSIDE the capture region into
    ///   the same scratch buffers).
    /// - Per-wave `memcpy_htod` chains in the SSM path are replaced by
    ///   pointer offsets into the pre-baked `tree_ssm_wave_indices_linear`
    ///   buffer (valid only for linear-chain prefill : wave[d] = [d]).
    /// - The final argmax DtoH + host accept walk are skipped : the
    ///   caller reads `tree_argmax_host_pinned` AFTER `end_capture`. The
    ///   function returns `Ok(vec![])`.
    /// - All other operations (kernel launches, device-side counter
    ///   advances, SSM/conv state commits) remain inside the captured
    ///   body and are replayed bit-identically on each replay.
    #[allow(clippy::too_many_lines)]
    fn decode_step_tree_hybrid_inner(
        &mut self,
        drafts: &[u32],
        parents: &[i32],
        depths: &[u16],
        force_accept_all: bool,
    ) -> Result<Vec<u32>, LlmError> {
        self.decode_step_tree_hybrid_inner_capture(drafts, parents, depths, force_accept_all, false)
    }

    /// T246.10 TrackI — full implementation with explicit capture-mode
    /// flag. See `decode_step_tree_hybrid_inner` for the contract.
    #[allow(clippy::too_many_lines, clippy::too_many_arguments)]
    fn decode_step_tree_hybrid_inner_capture(
        &mut self,
        drafts: &[u32],
        parents: &[i32],
        depths: &[u16],
        force_accept_all: bool,
        in_prefill_capture: bool,
    ) -> Result<Vec<u32>, LlmError> {
        debug_assert!(
            !in_prefill_capture || force_accept_all,
            "in_prefill_capture requires force_accept_all"
        );
        use cudarc::driver::{DevicePtr, DevicePtrMut};

        let cfg = self.config.clone();
        let d = cfg.d;
        let f = cfg.f;
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

        // T246.10 A6b.3 — GEMM prefill : when `force_accept_all` is true AND
        // tree_size >= GEMM_PREFILL_MIN_M AND RUSTORCH_GEMM_PREFILL=1, replace
        // per-row dispatch_matmul_m1 loops on Q/K/V/O + SSM in/out projections
        // with single dispatch_matmul_mvar calls (M=tree_size). The Dense FFN
        // and MoE FFN keep per-row dispatch (MoE expert routing is per-token).
        let use_gemm =
            force_accept_all && tree_size >= GEMM_PREFILL_MIN_M && gemm_prefill_enabled();

        let base_position = self.position;

        // ── 0. Upload tree descriptors ────────────────────────────────────
        // T246.10 TrackI — in capture mode the caller has already uploaded
        // these descriptors OUTSIDE the capture region (so the HtoD does
        // not abort the in-progress stream capture). The device buffers
        // are the same `scratch.tree_*` slots — the captured graph reads
        // them by pointer.
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

        // ── 1. Group tree nodes by BFS depth wave ─────────────────────────
        // Used by the SSM `delta_net_step_tree_bf16` launches : one launch
        // per depth so reads of parent slots see the prior-wave writes.
        let max_depth = *depths.iter().max().unwrap_or(&0) as usize;
        let mut waves: Vec<Vec<i32>> = vec![Vec::new(); max_depth + 1];
        for (r, &dep) in depths.iter().enumerate() {
            waves[dep as usize].push(r as i32);
        }

        // ── 2. Zero per-tree scratch (sized MAX_TREE_SIZE × per-token) ────
        // Mirrors decode_step_tree_pure_transformer.
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

        // Pre-extract device pointers (once per call ; the guards keep
        // the underlying CudaSlice alive for the duration of `unsafe`).
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
            x_q8_p,
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
            let q8 = if self.use_dp4a_q4k {
                let (s, _g26) = self.scratch.x_q8_scratch.device_ptr_mut(&self.stream);
                s
            } else {
                0u64
            };
            (
                a, b, c, d_, e, f_, g, h, i, j, k, l, m, n_, o, p, q, r0, r1, r2, r3, r4, r5, r6,
                r7, r8, q8,
            )
        };

        // ── 3. Embedding lookup ───────────────────────────────────────────
        unsafe {
            let (te_p, _g) = self.token_emb.device_ptr(&self.stream);
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
        // Iterate by index so we can borrow `self` for kernel calls AND
        // mutate `self.ssm_states` / `self.kv_caches` inside the loop.
        let n_layers = self.blocks.len();
        for li in 0..n_layers {
            // Snapshot residual and pre-norm hidden state per row.
            unsafe {
                self.kernels
                    .copy_bf16(&self.stream, tres_p, th_p, (tree_size * d) as i32)
                    .map_err(|e| LlmError::Backend(format!("hyb copy res l{li}: {e:?}")))?;
            }

            // Differ on layer kind. Borrow the block by raw index ; we
            // never alias mutably across this borrow.
            let is_attn = matches!(self.blocks[li], BlockQ4K::Attn(_));
            if is_attn {
                self.hybrid_attn_layer(
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
                    x_q8_p,
                    row_h,
                    row_q,
                    row_kv,
                    depths,
                    f,
                    eps,
                    n_q,
                    n_kv,
                    head_dim,
                    q_dim,
                    kv_dim,
                    rope_dim,
                    d,
                    use_gemm,
                )?;
            } else {
                self.hybrid_ssm_layer(
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
                    x_q8_p,
                    row_h,
                    row_conv,
                    row_value,
                    row_alpha,
                    conv_state_per_slot,
                    ssm_state_per_slot,
                    eps,
                    n_v,
                    n_k,
                    head_kv,
                    key_dim,
                    value_dim,
                    conv_dim,
                    conv_kernel,
                    d,
                    use_gemm,
                    in_prefill_capture,
                )?;
            }

            // ── FFN block ────────────────────────────────────────────────
            // residual <- h (post-mixer value)
            unsafe {
                self.kernels
                    .copy_bf16(&self.stream, tres_p, th_p, (tree_size * d) as i32)
                    .map_err(|e| LlmError::Backend(format!("hyb copy res ffn l{li}: {e:?}")))?;
            }

            // Pick the post_norm and ffn from the right block kind.
            let (post_norm_ptr, ffn_ref): (u64, &FfnQ4K) = match &self.blocks[li] {
                BlockQ4K::Attn(a) => {
                    let (p, _g) = a.post_norm.device_ptr(&self.stream);
                    (p, &a.ffn)
                },
                BlockQ4K::Ssm(s) => {
                    let (p, _g) = s.post_norm.device_ptr(&self.stream);
                    (p, &s.ffn)
                },
            };

            // h_norm = rms_norm(h, post_norm) — batched.
            unsafe {
                self.kernels
                    .copy_bf16(&self.stream, thn_p, th_p, (tree_size * d) as i32)
                    .map_err(|e| LlmError::Backend(format!("hyb copy h_norm ffn l{li}: {e:?}")))?;
                self.kernels
                    .rms_norm_bf16(
                        &self.stream,
                        thn_p,
                        post_norm_ptr,
                        eps,
                        d as i32,
                        tree_size as i32,
                    )
                    .map_err(|e| LlmError::Backend(format!("hyb rms_norm post l{li}: {e:?}")))?;
            }

            let row_ffn_gate = (self.scratch.tree_gate_buf.len() / MAX_TREE_SIZE) as u64 * bf16_sz;
            let row_ffn_up = (self.scratch.tree_up_buf.len() / MAX_TREE_SIZE) as u64 * bf16_sz;
            match ffn_ref {
                FfnQ4K::Dense { gate, up, down } => {
                    // Hybrid Dense FFN : enable GEMM only when scratch
                    // strides match the GEMM output strides (gate.N and up.N
                    // both == f BF16 elements). For hybrid models (Qwen3.6
                    // Dense), scratch_up = max(f, 2*q_dim). If 2*q_dim > f
                    // (typical for Qwen3.6 35B-A3B style) the stride
                    // mismatches and we fall back to per-row dispatch.
                    let gate_stride_ok = gate.shape().0 * bf16_sz as usize == row_ffn_gate as usize;
                    let up_stride_ok = up.shape().0 * bf16_sz as usize == row_ffn_up as usize;
                    if use_gemm && gate_stride_ok && up_stride_ok {
                        gate.dispatch_matmul_mvar(
                            &self.kernels,
                            &self.stream,
                            tree_size,
                            thn_p,
                            tgate_p,
                        )?;
                        up.dispatch_matmul_mvar(
                            &self.kernels,
                            &self.stream,
                            tree_size,
                            thn_p,
                            tup_p,
                        )?;
                        for r in 0..tree_size {
                            let gate_r = tgate_p + (r as u64) * row_ffn_gate;
                            let up_r = tup_p + (r as u64) * row_ffn_up;
                            unsafe {
                                self.kernels
                                    .swiglu_bf16(&self.stream, gate_r, up_r, gate_r, f as i32)
                                    .map_err(|e| {
                                        LlmError::Backend(format!("hyb swiglu r{r} l{li}: {e:?}"))
                                    })?;
                            }
                        }
                        down.dispatch_matmul_mvar(
                            &self.kernels,
                            &self.stream,
                            tree_size,
                            tgate_p,
                            th_p,
                        )?;
                    } else {
                        for r in 0..tree_size {
                            let hn_r = thn_p + (r as u64) * row_h;
                            let gate_r = tgate_p + (r as u64) * row_ffn_gate;
                            let up_r = tup_p + (r as u64) * row_ffn_up;
                            let h_r = th_p + (r as u64) * row_h;
                            gate.dispatch_matmul_m1(
                                &self.kernels,
                                &self.stream,
                                hn_r,
                                gate_r,
                                x_q8_p,
                            )?;
                            up.dispatch_matmul_m1(&self.kernels, &self.stream, hn_r, up_r, x_q8_p)?;
                            unsafe {
                                self.kernels
                                    .swiglu_bf16(&self.stream, gate_r, up_r, gate_r, f as i32)
                                    .map_err(|e| {
                                        LlmError::Backend(format!("hyb swiglu r{r} l{li}: {e:?}"))
                                    })?;
                            }
                            down.dispatch_matmul_m1(
                                &self.kernels,
                                &self.stream,
                                gate_r,
                                h_r,
                                x_q8_p,
                            )?;
                        }
                    }
                },
                FfnQ4K::Moe(moe) => {
                    // T246.10 A6 — MoE FFN per-row dispatch in the
                    // tree-forward path. The MoE scratch buffers
                    // (router/topk/expert) are single-token sized, so we
                    // call `moe_ffn_forward_step_*` once per BFS row,
                    // reading `h_norm[r]` and writing `h[r]`.
                    //
                    // Picks the async or mega variant based on env gates
                    // (both reduce to zero-host-sync per-token MoE ; mega
                    // also collapses the K=8 expert sgemv launches). The
                    // legacy SYNC path (host memcpy_dtov per layer per row)
                    // is NOT supported in prefill mode : at N=512 it would
                    // issue ~N × 64 layers × 2 syncs = 65 k host syncs,
                    // serializing the entire pipeline. The caller must set
                    // `RUSTORCH_MOE_ASYNC=1` (or `RUSTORCH_MOE_MEGA=1`).
                    if !moe_mega_enabled() && !moe_async_enabled() {
                        return Err(LlmError::Backend(
                            "decode_step_tree_hybrid_inner: MoE FFN at layer ".to_string()
                                + &li.to_string()
                                + " requires RUSTORCH_MOE_ASYNC=1 or \
                                   RUSTORCH_MOE_MEGA=1 for tree/prefill mode \
                                   (legacy sync path would serialize \
                                   ~N×64 layers host syncs).",
                        ));
                    }
                    use cudarc::driver::DevicePtrMut;
                    let (
                        moe_router_p,
                        moe_idx_p,
                        moe_w_p,
                        moe_egate_p,
                        moe_eup_p,
                        moe_eout_p,
                        moe_sd_p,
                    ) = {
                        let (mr_, _gmr) =
                            self.scratch.moe_router_logits.device_ptr_mut(&self.stream);
                        let (mi_, _gmi) = self.scratch.moe_topk_idx.device_ptr_mut(&self.stream);
                        let (mw_, _gmw) = self.scratch.moe_topk_w.device_ptr_mut(&self.stream);
                        let (mg_, _gmeg) =
                            self.scratch.moe_expert_gate.device_ptr_mut(&self.stream);
                        let (mu_, _gmeu) = self.scratch.moe_expert_up.device_ptr_mut(&self.stream);
                        let (mo_, _gmeo) = self.scratch.moe_expert_out.device_ptr_mut(&self.stream);
                        let (msd_, _gmsd) = self.scratch.moe_shexp_dot.device_ptr_mut(&self.stream);
                        (mr_, mi_, mw_, mg_, mu_, mo_, msd_)
                    };
                    // T246.10 TrackE.2 — Group-GEMM batched MoE forward.
                    // Only valid when force_accept_all (prefill mode) AND
                    // tree_size >= GROUP_GEMM_MIN_M AND RUSTORCH_MOE_GROUP_GEMM=1.
                    // Decode (tree_size=1) and small trees keep the per-row
                    // mega loop (A2/A4 wins are preserved).
                    let use_group_gemm = force_accept_all
                        && tree_size >= GROUP_GEMM_MIN_M
                        && moe_group_gemm_enabled()
                        && moe_mega_enabled();
                    if use_group_gemm {
                        let (
                            mrlm_p,
                            midxm_p,
                            mwm_p,
                            megm_p,
                            meum_p,
                            meom_p,
                            mxq8m_p,
                            ids_src1_m_p,
                            ids_dst_m_p,
                            expert_bounds_p,
                        ) = {
                            let (a, _g1) = self
                                .scratch
                                .moe_router_logits_m
                                .device_ptr_mut(&self.stream);
                            let (b, _g2) = self.scratch.moe_topk_idx_m.device_ptr_mut(&self.stream);
                            let (c, _g3) = self.scratch.moe_topk_w_m.device_ptr_mut(&self.stream);
                            let (d_, _g4) =
                                self.scratch.moe_expert_gate_m.device_ptr_mut(&self.stream);
                            let (e_, _g5) =
                                self.scratch.moe_expert_up_m.device_ptr_mut(&self.stream);
                            let (f_, _g6) =
                                self.scratch.moe_expert_out_m.device_ptr_mut(&self.stream);
                            let (g_, _g7) = self.scratch.moe_x_q8_m.device_ptr_mut(&self.stream);
                            let (h_, _g8) =
                                self.scratch.moe_ids_src1_m.device_ptr_mut(&self.stream);
                            let (i_, _g9) = self.scratch.moe_ids_dst_m.device_ptr_mut(&self.stream);
                            let (j_, _g10) =
                                self.scratch.moe_expert_bounds.device_ptr_mut(&self.stream);
                            (a, b, c, d_, e_, f_, g_, h_, i_, j_)
                        };
                        // TrackE.3 — env-gated sort-permutation Group-GEMM.
                        // Only valid on top of the unsorted Group-GEMM path.
                        let use_sorted = moe_group_gemm_sorted_enabled();
                        // MMQ-WHOLESALE.6a — packed Q8_1 staging pointer for
                        // the shared-expert MMQ path (0 = skip, fall back to
                        // per-token loop). Gated downstream by the env flag
                        // + minimum M + Q4_K weight kind.
                        let (mxq8mmq_p, _g11) =
                            self.scratch.moe_x_q8_mmq_m.device_ptr_mut(&self.stream);
                        moe_ffn_forward_step_group_gemm(
                            moe,
                            &self.kernels,
                            &self.stream,
                            &cfg,
                            tree_size as i32,
                            thn_p,
                            th_p,
                            mrlm_p,
                            midxm_p,
                            mwm_p,
                            megm_p,
                            meum_p,
                            meom_p,
                            mxq8m_p,
                            x_q8_p,
                            moe_sd_p,
                            moe_egate_p,
                            moe_eup_p,
                            moe_eout_p,
                            ids_src1_m_p,
                            ids_dst_m_p,
                            expert_bounds_p,
                            use_sorted,
                            mxq8mmq_p,
                        )?;
                    } else {
                        for r in 0..tree_size {
                            let hn_r = thn_p + (r as u64) * row_h;
                            let h_r = th_p + (r as u64) * row_h;
                            if moe_mega_enabled() {
                                moe_ffn_forward_step_mega(
                                    moe,
                                    &self.kernels,
                                    &self.stream,
                                    &cfg,
                                    hn_r,
                                    h_r,
                                    x_q8_p,
                                    moe_router_p,
                                    moe_idx_p,
                                    moe_w_p,
                                    moe_egate_p,
                                    moe_eup_p,
                                    moe_eout_p,
                                    moe_sd_p,
                                )?;
                            } else {
                                // moe_async_enabled() == true by the guard above.
                                moe_ffn_forward_step_async(
                                    moe,
                                    &self.kernels,
                                    &self.stream,
                                    &cfg,
                                    hn_r,
                                    h_r,
                                    x_q8_p,
                                    moe_router_p,
                                    moe_idx_p,
                                    moe_w_p,
                                    moe_egate_p,
                                    moe_eup_p,
                                    moe_eout_p,
                                    moe_sd_p,
                                )?;
                            }
                        }
                    }
                },
            }

            // h += residual (post-FFN).
            unsafe {
                self.kernels
                    .add_inplace_bf16(&self.stream, th_p, tres_p, (tree_size * d) as i32)
                    .map_err(|e| LlmError::Backend(format!("hyb ffn residual l{li}: {e:?}")))?;
            }
        }

        // ── 5. Final RMSNorm + LM head per row ────────────────────────────
        unsafe {
            let (fn_p, _g) = self.final_norm.device_ptr(&self.stream);
            self.kernels
                .rms_norm_bf16(&self.stream, th_p, fn_p, eps, d as i32, tree_size as i32)
                .map_err(|e| LlmError::Backend(format!("hyb final_norm: {e:?}")))?;
        }
        // T246.10 HOTFIX-A6b — batch lm_head through dispatch_matmul_mvar when
        // in GEMM prefill mode. lm_head is Q6_K on Qwen3.6 ; the per-row
        // SGEMV path was firing tree_size copies of sgemv_q6k_bf16_v2 = 1835 ms
        // per pp512 (19% of GPU time, PRE-FLIGHT note 1ac7de4a). The row stride
        // of tree_logits is vocab × bf16 (= N × bf16), matching the mvar layout
        // y[M, N] row-major. The lm_head K = hidden_size = d (multiple of 256
        // for Qwen3.6 : d = 2048). Falls back to per-row dispatch when K%256 != 0
        // or use_gemm is false (preserving bit-exact parity for decode and
        // legacy paths).
        let lm_k_ok = self.lm_head.shape().1 % 256 == 0;
        if use_gemm && lm_k_ok {
            self.lm_head.dispatch_matmul_mvar(
                &self.kernels,
                &self.stream,
                tree_size,
                th_p,
                tlogits_p,
            )?;
        } else {
            for r in 0..tree_size {
                let h_r = th_p + (r as u64) * row_h;
                let logits_r = tlogits_p + (r as u64) * row_logits;
                // T246.8 A5 — split-K dispatch (env-gated).
                self.dispatch_lm_head(h_r, logits_r, x_q8_p)?;
            }
        }

        // ── 6. Argmax + DtoH ──────────────────────────────────────────────
        unsafe {
            self.kernels
                .argmax_logits_tree_bf16(
                    &self.stream,
                    tlogits_p,
                    targmax_p,
                    tree_size as i32,
                    vocab as i32,
                )
                .map_err(|e| LlmError::Backend(format!("hyb argmax: {e:?}")))?;
        }

        // ── 6. DtoH the argmax tokens to pinned host buffer ───────────────
        //
        // T246.10 TrackI — in capture mode the DtoH + host accept walk
        // MUST happen AFTER `end_capture` (we can't read host-side
        // pinned memory from inside a capturing stream). The caller
        // (`prefill_tokens`) does this read post-replay. For linear-
        // chain prefill the accept walk degenerates to
        // `accepted_indices = [0..tree_size]` and the compact loop is
        // a no-op (each `src == i`), so we can shortcut the entire
        // host-side logic here.
        let (accepted_indices, accepted_tokens): (Vec<usize>, Vec<u32>) = if in_prefill_capture {
            // Linear chain : accept all, tokens are filled by caller
            // post-replay (an empty Vec is returned and the caller
            // ignores it — `prefill_tokens` reads the argmax directly).
            let idx: Vec<usize> = (0..tree_size).collect();
            (idx, Vec::new())
        } else {
            self.stream
                .memcpy_dtoh(
                    &self.scratch.tree_argmax,
                    &mut self.scratch.tree_argmax_host_pinned,
                )
                .map_err(|e| LlmError::Backend(format!("hyb dtoh argmax: {e:?}")))?;
            let argmax_host = self
                .scratch
                .tree_argmax_host_pinned
                .as_slice()
                .map_err(|e| LlmError::Backend(format!("hyb pinned argmax: {e:?}")))?;

            // ── 7. CPU acceptance walk (same as pure-transformer path) ────
            // T246.10 A6 — prefill mode (`force_accept_all = true`) bypasses
            // the acceptance walk and force-accepts every BFS-ordered node.
            // For a linear-chain prefill the BFS order is the chain itself,
            // so `accepted_indices[i] = i`, the compaction loop (step 8)
            // is a no-op (each `src == i`), and the counter advance writes
            // all N positions.
            if force_accept_all {
                let idx: Vec<usize> = (0..tree_size).collect();
                let tok: Vec<u32> = (0..tree_size).map(|r| argmax_host[r]).collect();
                (idx, tok)
            } else {
                let mut children: Vec<Vec<usize>> = vec![Vec::new(); tree_size];
                for (r, &p) in parents.iter().enumerate().skip(1) {
                    children[p as usize].push(r);
                }
                let mut a_indices: Vec<usize> = vec![0];
                let mut a_tokens: Vec<u32> = vec![argmax_host[0]];
                let mut cur = 0usize;
                loop {
                    let next_tok = argmax_host[cur];
                    let mut found: Option<usize> = None;
                    for &c in &children[cur] {
                        if drafts[c] == next_tok {
                            found = Some(c);
                            break;
                        }
                    }
                    match found {
                        Some(c) => {
                            a_indices.push(c);
                            a_tokens.push(argmax_host[c]);
                            cur = c;
                        },
                        None => break,
                    }
                }
                (a_indices, a_tokens)
            }
        };
        let accept_len = accepted_indices.len();
        debug_assert!(accept_len >= 1);

        // ── 8. Compact accepted KV slots (attention layers only) ──────────
        for li in 0..self.blocks.len() {
            if !matches!(self.blocks[li], BlockQ4K::Attn(_)) {
                continue;
            }
            let attn_idx = get_attn_layer_idx(&cfg, li);
            let kv_cache = &mut self.kv_caches[attn_idx];
            unsafe {
                let (kc_p, _g1) = kv_cache.k.device_ptr_mut(&self.stream);
                let (vc_p, _g2) = kv_cache.v.device_ptr_mut(&self.stream);
                for (i, &src) in accepted_indices.iter().enumerate().take(accept_len).skip(1) {
                    if src == i {
                        continue;
                    }
                    let src_off = (base_position as u64 + src as u64) * (kv_dim as u64);
                    let dst_off = (base_position as u64 + i as u64) * (kv_dim as u64);
                    self.kernels
                        .copy_bf16(
                            &self.stream,
                            kc_p + dst_off * bf16_sz,
                            kc_p + src_off * bf16_sz,
                            kv_dim as i32,
                        )
                        .map_err(|e| LlmError::Backend(format!("hyb compact K l{li}: {e:?}")))?;
                    self.kernels
                        .copy_bf16(
                            &self.stream,
                            vc_p + dst_off * bf16_sz,
                            vc_p + src_off * bf16_sz,
                            kv_dim as i32,
                        )
                        .map_err(|e| LlmError::Backend(format!("hyb compact V l{li}: {e:?}")))?;
                }
            }
        }

        // ── 9. Commit accepted SSM state — copy deepest accepted node's
        //       per-layer state and conv state back into the model's
        //       per-SSM-layer buffers. ────────────────────────────────────
        let leaf_idx = *accepted_indices.last().unwrap();
        for li in 0..self.blocks.len() {
            if !matches!(self.blocks[li], BlockQ4K::Ssm(_)) {
                continue;
            }
            let s_idx = get_ssm_layer_idx(&cfg, li);
            // Two scopes : the SsmState struct holds both `state` and
            // `conv_state` fields, but cudarc's device_ptr_mut takes a
            // &mut on the slice, and the borrow checker can't see that
            // the two field borrows are disjoint when going through
            // `self.ssm_states[s_idx]`. Splitting into two separate
            // scopes (releasing the first guard before acquiring the
            // second) sidesteps this.
            unsafe {
                let (src_state_p, _g1) =
                    self.scratch.tree_ssm_states[s_idx].device_ptr(&self.stream);
                let (dst_state_p, _g2) = self.ssm_states[s_idx].state.device_ptr_mut(&self.stream);
                let n_state_elems = (n_v * head_kv * head_kv) as i32;
                self.kernels
                    .copy_bf16(
                        &self.stream,
                        dst_state_p,
                        src_state_p + (leaf_idx as u64) * ssm_state_per_slot,
                        n_state_elems,
                    )
                    .map_err(|e| LlmError::Backend(format!("hyb commit ssm state l{li}: {e:?}")))?;
            }
            unsafe {
                let (src_conv_p, _g3) =
                    self.scratch.tree_conv_states[s_idx].device_ptr(&self.stream);
                let (dst_conv_p, _g4) = self.ssm_states[s_idx]
                    .conv_state
                    .device_ptr_mut(&self.stream);
                let n_conv_elems = ((conv_kernel - 1) * conv_dim) as i32;
                self.kernels
                    .copy_bf16(
                        &self.stream,
                        dst_conv_p,
                        src_conv_p + (leaf_idx as u64) * conv_state_per_slot,
                        n_conv_elems,
                    )
                    .map_err(|e| LlmError::Backend(format!("hyb commit conv l{li}: {e:?}")))?;
            }
        }

        // ── 10. Advance device counters ───────────────────────────────────
        unsafe {
            let (pos_p, _g_pos) = self.position_dev.device_ptr_mut(&self.stream);
            self.kernels
                .add_u32_dev(&self.stream, pos_p, accept_len as i32)
                .map_err(|e| LlmError::Backend(format!("hyb add_u32 pos: {e:?}")))?;
        }
        unsafe {
            let (kvl_p, _g_kvl) = self.kv_len_dev.device_ptr_mut(&self.stream);
            self.kernels
                .add_u32_dev(&self.stream, kvl_p, accept_len as i32)
                .map_err(|e| LlmError::Backend(format!("hyb add_u32 kv_len: {e:?}")))?;
        }
        // T246.10 TrackI — in capture mode the host-side `self.position`
        // bookkeeping happens in the caller (`prefill_tokens`), not here.
        // The device-side `position_dev` IS advanced inside the captured
        // graph above, so each replay correctly steps the device counter.
        if !in_prefill_capture {
            self.position += accept_len;
        }

        Ok(accepted_tokens)
    }

    // ── TrackC.3 — per-layer helpers for decode_step_tree_hybrid ──────────
    //
    // The two helpers below isolate the attention and SSM forward paths so
    // the dispatching `decode_step_tree_hybrid` body stays manageable.
    // They take a wide arg list because they share the per-call scratch
    // pointers extracted at the top of the dispatcher (avoids re-acquiring
    // device_ptr_mut inside hot loops, and keeps the borrow scope short).

    #[allow(clippy::too_many_arguments)]
    fn hybrid_attn_layer(
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
        x_q8_p: u64,
        row_h: u64,
        row_q: u64,
        row_kv: u64,
        depths: &[u16],
        _f: usize,
        eps: f32,
        n_q: usize,
        n_kv: usize,
        head_dim: usize,
        q_dim: usize,
        kv_dim: usize,
        rope_dim: usize,
        d: usize,
        use_gemm: bool,
    ) -> Result<(), LlmError> {
        use cudarc::driver::{DevicePtr, DevicePtrMut};
        let cfg = &self.config;
        let attn = match &self.blocks[li] {
            BlockQ4K::Attn(a) => a,
            _ => unreachable!("hybrid_attn_layer called on non-attn layer {li}"),
        };

        // Per-row strides for the wider scratch buffers (gate / up).
        let bf16_sz = std::mem::size_of::<half::bf16>() as u64;
        let row_gate = (self.scratch.tree_gate_buf.len() / MAX_TREE_SIZE) as u64 * bf16_sz;
        let row_up = (self.scratch.tree_up_buf.len() / MAX_TREE_SIZE) as u64 * bf16_sz;

        // RMSNorm.
        unsafe {
            let (an, _g) = attn.attn_norm.device_ptr(&self.stream);
            self.kernels
                .copy_bf16(&self.stream, thn_p, th_p, (tree_size * d) as i32)
                .map_err(|e| LlmError::Backend(format!("hyb attn copy h_norm l{li}: {e:?}")))?;
            self.kernels
                .rms_norm_bf16(&self.stream, thn_p, an, eps, d as i32, tree_size as i32)
                .map_err(|e| LlmError::Backend(format!("hyb attn rms_norm l{li}: {e:?}")))?;
        }

        // Q projection. With Q+gate (Qwen3 / Qwen3.5 / Qwen3.6), w_q
        // outputs 2*q_dim per row : we route it through tree_up_buf
        // (scratch row stride ≥ 2*q_dim guaranteed at constructor) then
        // split into q (→ tq_p) and gate (→ tree_gate_buf). For variants
        // without output gate (qwen2), w_q outputs q_dim directly into
        // tq_p.
        if cfg.variant.attn_has_output_gate() {
            if use_gemm {
                // T246.10 A6b.3 — one mvar call writes M=tree_size rows of
                // [q | gate] into tup_p (stride row_up = 2*q_dim = w_q.N).
                attn.w_q.dispatch_matmul_mvar(
                    &self.kernels,
                    &self.stream,
                    tree_size,
                    thn_p,
                    tup_p,
                )?;
            } else {
                for r in 0..tree_size {
                    let hn_r = thn_p + (r as u64) * row_h;
                    let qg_r = tup_p + (r as u64) * row_up;
                    attn.w_q
                        .dispatch_matmul_m1(&self.kernels, &self.stream, hn_r, qg_r, x_q8_p)?;
                }
            }
            // split_qg per row (output_gate variants only). Per-row launch
            // is fine — split_qg is a small element-wise op.
            for r in 0..tree_size {
                let qg_r = tup_p + (r as u64) * row_up;
                let q_r = tq_p + (r as u64) * row_q;
                let gate_r = tgate_p + (r as u64) * row_gate;
                unsafe {
                    self.kernels
                        .split_qg_bf16(
                            &self.stream,
                            qg_r,
                            q_r,
                            gate_r, // pre-sigmoid gate, q_dim per row
                            n_q as i32,
                            head_dim as i32,
                        )
                        .map_err(|e| {
                            LlmError::Backend(format!("hyb split_qg r{r} l{li}: {e:?}"))
                        })?;
                }
            }
        } else if use_gemm {
            attn.w_q
                .dispatch_matmul_mvar(&self.kernels, &self.stream, tree_size, thn_p, tq_p)?;
        } else {
            for r in 0..tree_size {
                let hn_r = thn_p + (r as u64) * row_h;
                let q_r = tq_p + (r as u64) * row_q;
                attn.w_q
                    .dispatch_matmul_m1(&self.kernels, &self.stream, hn_r, q_r, x_q8_p)?;
            }
        }

        // K, V projections.
        if use_gemm {
            attn.w_k
                .dispatch_matmul_mvar(&self.kernels, &self.stream, tree_size, thn_p, tk_p)?;
            attn.w_v
                .dispatch_matmul_mvar(&self.kernels, &self.stream, tree_size, thn_p, tv_p)?;
        } else {
            for r in 0..tree_size {
                let hn_r = thn_p + (r as u64) * row_h;
                let k_r = tk_p + (r as u64) * row_kv;
                let v_r = tv_p + (r as u64) * row_kv;
                attn.w_k
                    .dispatch_matmul_m1(&self.kernels, &self.stream, hn_r, k_r, x_q8_p)?;
                attn.w_v
                    .dispatch_matmul_m1(&self.kernels, &self.stream, hn_r, v_r, x_q8_p)?;
            }
        }

        // QKV bias add (qwen2 only — Qwen3.6 has no bias).
        if cfg.variant.attn_has_qkv_bias() {
            unsafe {
                let (bq, _gbq) = attn
                    .b_q
                    .as_ref()
                    .expect("qwen2: b_q present")
                    .device_ptr(&self.stream);
                let (bk, _gbk) = attn
                    .b_k
                    .as_ref()
                    .expect("qwen2: b_k present")
                    .device_ptr(&self.stream);
                let (bv, _gbv) = attn
                    .b_v
                    .as_ref()
                    .expect("qwen2: b_v present")
                    .device_ptr(&self.stream);
                for r in 0..tree_size {
                    let q_r = tq_p + (r as u64) * row_q;
                    let k_r = tk_p + (r as u64) * row_kv;
                    let v_r = tv_p + (r as u64) * row_kv;
                    self.kernels
                        .add_inplace_bf16(&self.stream, q_r, bq, q_dim as i32)
                        .map_err(|e| LlmError::Backend(format!("hyb q_bias r{r}: {e:?}")))?;
                    self.kernels
                        .add_inplace_bf16(&self.stream, k_r, bk, kv_dim as i32)
                        .map_err(|e| LlmError::Backend(format!("hyb k_bias r{r}: {e:?}")))?;
                    self.kernels
                        .add_inplace_bf16(&self.stream, v_r, bv, kv_dim as i32)
                        .map_err(|e| LlmError::Backend(format!("hyb v_bias r{r}: {e:?}")))?;
                }
            }
        }

        // Per-head Q-norm and K-norm.
        if cfg.variant.attn_has_qk_norm() {
            unsafe {
                let (qn, _g1) = attn
                    .q_norm
                    .as_ref()
                    .expect("qk_norm variant: q_norm present")
                    .device_ptr(&self.stream);
                let (kn, _g2) = attn
                    .k_norm
                    .as_ref()
                    .expect("qk_norm variant: k_norm present")
                    .device_ptr(&self.stream);
                self.kernels
                    .rms_norm_bf16(
                        &self.stream,
                        tq_p,
                        qn,
                        eps,
                        head_dim as i32,
                        (tree_size * n_q) as i32,
                    )
                    .map_err(|e| LlmError::Backend(format!("hyb q_norm l{li}: {e:?}")))?;
                self.kernels
                    .rms_norm_bf16(
                        &self.stream,
                        tk_p,
                        kn,
                        eps,
                        head_dim as i32,
                        (tree_size * n_kv) as i32,
                    )
                    .map_err(|e| LlmError::Backend(format!("hyb k_norm l{li}: {e:?}")))?;
            }
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
                    .map_err(|e| LlmError::Backend(format!("hyb rope q r{r} l{li}: {e:?}")))?;
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
                    .map_err(|e| LlmError::Backend(format!("hyb rope k r{r} l{li}: {e:?}")))?;
            }
        }

        // KV append + GQA decode (tree-aware). Attention output writes
        // into tree_attn_out (tao_p), q_dim per row.
        let attn_idx = get_attn_layer_idx(cfg, li);
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
                .map_err(|e| LlmError::Backend(format!("hyb kv_append l{li}: {e:?}")))?;
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
                .map_err(|e| LlmError::Backend(format!("hyb gqa l{li}: {e:?}")))?;
        }

        // Sigmoid gate × attn_out per row (Qwen3.6 has output gate).
        // Gate is in tree_gate_buf (q_dim per row, gate_r stride).
        if cfg.variant.attn_has_output_gate() {
            for r in 0..tree_size {
                let gate_r = tgate_p + (r as u64) * row_gate;
                let ao_r = tao_p + (r as u64) * row_q;
                unsafe {
                    self.kernels
                        .sigmoid_inplace_bf16(&self.stream, gate_r, q_dim as i32)
                        .map_err(|e| LlmError::Backend(format!("hyb sigmoid r{r} l{li}: {e:?}")))?;
                    self.kernels
                        .mul_inplace_bf16(&self.stream, ao_r, gate_r, q_dim as i32)
                        .map_err(|e| LlmError::Backend(format!("hyb gate*ao r{r} l{li}: {e:?}")))?;
                }
            }
        }

        // w_o → tree_h ; residual add.
        if use_gemm {
            attn.w_o
                .dispatch_matmul_mvar(&self.kernels, &self.stream, tree_size, tao_p, th_p)?;
        } else {
            for r in 0..tree_size {
                let ao_r = tao_p + (r as u64) * row_q;
                let h_r = th_p + (r as u64) * row_h;
                attn.w_o
                    .dispatch_matmul_m1(&self.kernels, &self.stream, ao_r, h_r, x_q8_p)?;
            }
        }
        unsafe {
            self.kernels
                .add_inplace_bf16(&self.stream, th_p, tres_p, (tree_size * d) as i32)
                .map_err(|e| LlmError::Backend(format!("hyb attn residual l{li}: {e:?}")))?;
        }

        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::too_many_arguments)]
    fn hybrid_ssm_layer(
        &mut self,
        li: usize,
        tree_size: usize,
        parents: &[i32],
        waves: &[Vec<i32>],
        th_p: u64,
        thn_p: u64,
        _tres_p: u64,
        tssm_qkv_p: u64,
        tssm_conv_p: u64,
        tssm_z_p: u64,
        tssm_alpha_p: u64,
        tssm_beta_p: u64,
        tssm_qv_p: u64,
        tssm_kv_p: u64,
        tssm_out_p: u64,
        tssm_wave_p: u64,
        x_q8_p: u64,
        row_h: u64,
        row_conv: u64,
        row_value: u64,
        row_alpha: u64,
        conv_state_per_slot: u64,
        _ssm_state_per_slot: u64,
        eps: f32,
        n_v: usize,
        n_k: usize,
        head_kv: usize,
        key_dim: usize,
        value_dim: usize,
        conv_dim: usize,
        conv_kernel: usize,
        d: usize,
        use_gemm: bool,
        in_prefill_capture: bool,
    ) -> Result<(), LlmError> {
        use cudarc::driver::{DevicePtr, DevicePtrMut};
        let cfg = &self.config;
        let bf16_sz = std::mem::size_of::<half::bf16>() as u64;
        let s_idx = get_ssm_layer_idx(cfg, li);
        let ssm = match &self.blocks[li] {
            BlockQ4K::Ssm(s) => s,
            _ => unreachable!("hybrid_ssm_layer called on non-ssm layer {li}"),
        };

        // 1. Pre-SSM RMSNorm — batched over tree rows.
        unsafe {
            let (an, _g) = ssm.attn_norm.device_ptr(&self.stream);
            self.kernels
                .copy_bf16(&self.stream, thn_p, th_p, (tree_size * d) as i32)
                .map_err(|e| LlmError::Backend(format!("hyb ssm copy h_norm l{li}: {e:?}")))?;
            self.kernels
                .rms_norm_bf16(&self.stream, thn_p, an, eps, d as i32, tree_size as i32)
                .map_err(|e| LlmError::Backend(format!("hyb ssm rms_norm l{li}: {e:?}")))?;
        }

        // 2-5. SSM in-projections : w_qkv, w_gate, w_alpha, w_beta. Strides
        // match (row_conv = conv_dim = w_qkv.N ; row_value = value_dim =
        // w_gate.N ; row_alpha = n_v = w_alpha.N = w_beta.N), safe to mvar.
        //
        // Note : w_alpha/w_beta require K % 256 == 0 for the Q5_K mvar
        // kernel. If hidden_size (= w_alpha.K) is not a multiple of 256, fall
        // back to the per-row path for these two.
        let alpha_k_ok = ssm.w_alpha.shape().1 % 256 == 0;
        let beta_k_ok = ssm.w_beta.shape().1 % 256 == 0;
        if use_gemm {
            ssm.w_qkv.dispatch_matmul_mvar(
                &self.kernels,
                &self.stream,
                tree_size,
                thn_p,
                tssm_qkv_p,
            )?;
            ssm.w_gate.dispatch_matmul_mvar(
                &self.kernels,
                &self.stream,
                tree_size,
                thn_p,
                tssm_z_p,
            )?;
            if alpha_k_ok {
                ssm.w_alpha.dispatch_matmul_mvar(
                    &self.kernels,
                    &self.stream,
                    tree_size,
                    thn_p,
                    tssm_alpha_p,
                )?;
            } else {
                for r in 0..tree_size {
                    let hn_r = thn_p + (r as u64) * row_h;
                    let alpha_r = tssm_alpha_p + (r as u64) * row_alpha;
                    ssm.w_alpha.dispatch_matmul_m1(
                        &self.kernels,
                        &self.stream,
                        hn_r,
                        alpha_r,
                        x_q8_p,
                    )?;
                }
            }
            if beta_k_ok {
                ssm.w_beta.dispatch_matmul_mvar(
                    &self.kernels,
                    &self.stream,
                    tree_size,
                    thn_p,
                    tssm_beta_p,
                )?;
            } else {
                for r in 0..tree_size {
                    let hn_r = thn_p + (r as u64) * row_h;
                    let beta_r = tssm_beta_p + (r as u64) * row_alpha;
                    ssm.w_beta.dispatch_matmul_m1(
                        &self.kernels,
                        &self.stream,
                        hn_r,
                        beta_r,
                        x_q8_p,
                    )?;
                }
            }
        } else {
            for r in 0..tree_size {
                let hn_r = thn_p + (r as u64) * row_h;
                let qkv_r = tssm_qkv_p + (r as u64) * row_conv;
                let z_r = tssm_z_p + (r as u64) * row_value;
                let alpha_r = tssm_alpha_p + (r as u64) * row_alpha;
                let beta_r = tssm_beta_p + (r as u64) * row_alpha;
                ssm.w_qkv
                    .dispatch_matmul_m1(&self.kernels, &self.stream, hn_r, qkv_r, x_q8_p)?;
                ssm.w_gate
                    .dispatch_matmul_m1(&self.kernels, &self.stream, hn_r, z_r, x_q8_p)?;
                ssm.w_alpha.dispatch_matmul_m1(
                    &self.kernels,
                    &self.stream,
                    hn_r,
                    alpha_r,
                    x_q8_p,
                )?;
                ssm.w_beta
                    .dispatch_matmul_m1(&self.kernels, &self.stream, hn_r, beta_r, x_q8_p)?;
            }
        }

        // 5b+6. Fused SSM pre-step (T246.8 A1.1) per row :
        //   sigmoid(beta) ; alpha += dt_bias ; softplus(alpha) ;
        //   alpha *= ssm_a → gate_h. Replaces 4 launches per row by 1
        //   when RUSTORCH_SSM_FUSE=1 (default).
        unsafe {
            let (db, _gdb) = ssm.dt_bias.device_ptr(&self.stream);
            let (sa, _gsa) = ssm.ssm_a.device_ptr(&self.stream);
            for r in 0..tree_size {
                let alpha_r = tssm_alpha_p + (r as u64) * row_alpha;
                let beta_r = tssm_beta_p + (r as u64) * row_alpha;
                if self.use_ssm_fuse {
                    self.kernels
                        .ssm_pre_step_bf16(&self.stream, alpha_r, beta_r, db, sa, n_v as i32)
                        .map_err(|e| LlmError::Backend(format!("hyb ssm_pre_step r{r}: {e:?}")))?;
                } else {
                    self.kernels
                        .sigmoid_inplace_bf16(&self.stream, beta_r, n_v as i32)
                        .map_err(|e| LlmError::Backend(format!("hyb sigmoid beta r{r}: {e:?}")))?;
                    self.kernels
                        .add_inplace_bf16(&self.stream, alpha_r, db, n_v as i32)
                        .map_err(|e| LlmError::Backend(format!("hyb alpha+dt r{r}: {e:?}")))?;
                    self.kernels
                        .softplus_inplace_bf16(&self.stream, alpha_r, n_v as i32)
                        .map_err(|e| LlmError::Backend(format!("hyb softplus r{r}: {e:?}")))?;
                    self.kernels
                        .mul_inplace_bf16(&self.stream, alpha_r, sa, n_v as i32)
                        .map_err(|e| LlmError::Backend(format!("hyb mul_a r{r}: {e:?}")))?;
                }
            }
        }

        // 7. Conv1d per tree row. Pre-load slot 0 with model's current
        //    conv state, then for each row r > 0 copy parent's conv slot
        //    into row r's slot before launching the (in-place) conv1d.
        unsafe {
            let (model_conv_p, _g) = self.ssm_states[s_idx].conv_state.device_ptr(&self.stream);
            let (tree_conv_states_p, _g2) =
                self.scratch.tree_conv_states[s_idx].device_ptr_mut(&self.stream);
            // Slot 0 ← model state.
            self.kernels
                .copy_bf16(
                    &self.stream,
                    tree_conv_states_p,
                    model_conv_p,
                    ((conv_kernel - 1) * conv_dim) as i32,
                )
                .map_err(|e| LlmError::Backend(format!("hyb pre-load conv state l{li}: {e:?}")))?;

            // Walk in BFS order (depth waves order doesn't matter for
            // conv since each row's call is sequential and independent
            // once parent's slot is committed). parents[r] < r so a
            // forward 0..tree_size pass respects dependency.
            let (cw, _gcw) = ssm.conv1d.device_ptr(&self.stream);
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
                            LlmError::Backend(format!("hyb conv parent copy r{r}: {e:?}"))
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
                    .map_err(|e| LlmError::Backend(format!("hyb conv1d r{r}: {e:?}")))?;
            }
        }

        // 8. silu(conv_out) per row.
        for r in 0..tree_size {
            let conv_out_r = tssm_conv_p + (r as u64) * row_conv;
            unsafe {
                self.kernels
                    .silu_bf16(&self.stream, conv_out_r, conv_dim as i32)
                    .map_err(|e| LlmError::Backend(format!("hyb silu conv r{r}: {e:?}")))?;
            }
        }

        // 9. Split q/k/v from conv_out, l2_norm_per_head, broadcast n_k→n_v.
        // Conv output layout per row : [2*key_dim (q,k) | value_dim (v)].
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
                    .map_err(|e| LlmError::Backend(format!("hyb l2 q r{r}: {e:?}")))?;
                self.kernels
                    .l2_norm_per_head_bf16(&self.stream, k_r, n_k as i32, head_kv as i32, eps)
                    .map_err(|e| LlmError::Backend(format!("hyb l2 k r{r}: {e:?}")))?;
            }
        }

        // Broadcast q,k from n_k → n_v heads per row into tree_ssm_q_v / k_v.
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
                        .map_err(|e| LlmError::Backend(format!("hyb copy qv r{r}: {e:?}")))?;
                    self.kernels
                        .copy_bf16(&self.stream, kv_r, k_r, value_dim as i32)
                        .map_err(|e| LlmError::Backend(format!("hyb copy kv r{r}: {e:?}")))?;
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
                        .map_err(|e| LlmError::Backend(format!("hyb repeat q r{r}: {e:?}")))?;
                    self.kernels
                        .repeat_heads_bf16(
                            &self.stream,
                            k_r,
                            kv_r,
                            n_k as i32,
                            n_v as i32,
                            head_kv as i32,
                        )
                        .map_err(|e| LlmError::Backend(format!("hyb repeat k r{r}: {e:?}")))?;
                }
            }
        }

        // 10. Pre-load model's current SSM state into tree slot 0, then
        //     launch delta_net_step_tree_bf16 once per BFS depth wave.
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
                .map_err(|e| LlmError::Backend(format!("hyb pre-load ssm state l{li}: {e:?}")))?;

            let (parents_dev_p, _gp) = self.scratch.tree_parents.device_ptr(&self.stream);
            // V slice : v lives at conv_out + v_offset for each row.
            // Build a contiguous-per-row layout : the kernel expects
            // [tree_size, n_v, head_kv] BF16. tssm_conv_p row stride is
            // conv_dim ; the v sub-region within a row is value_dim and
            // is already laid out as [n_v, head_kv]. So passing
            // tssm_conv_p + v_offset as base, with row stride = conv_dim
            // BF16 elements, would NOT give the kernel a contiguous
            // [tree_size, n_v, head_kv] view. We must repack v into a
            // dedicated `tree_v` buffer or accept the layout mismatch.
            //
            // Workaround : copy each row's v segment into tssm_out_buf
            // (used as V scratch — it has size value_dim per row,
            // matching exactly).
            for r in 0..tree_size {
                let conv_out_r = tssm_conv_p + (r as u64) * row_conv;
                let v_r = conv_out_r + v_offset;
                let dst_v = tssm_out_p + (r as u64) * row_value;
                self.kernels
                    .copy_bf16(&self.stream, dst_v, v_r, value_dim as i32)
                    .map_err(|e| LlmError::Backend(format!("hyb copy v r{r}: {e:?}")))?;
            }

            // Launch one wave per depth.
            //
            // T246.10 TrackI — when in capture mode (linear-chain prefill),
            // we avoid the per-wave `memcpy_htod` (which is capture-
            // incompatible) by passing offsets into the pre-baked linear
            // wave-indices buffer (`tree_ssm_wave_indices_linear` =
            // `[0, 1, ..., MAX_TREE_SIZE-1]`). For linear-chain prefill
            // every wave is a single element : wave[d] = [d], so we
            // launch with `wave_indices = linear_buf + d*4`,
            // `wave_size = 1`. For non-linear trees (Lookahead verify
            // path) we still upload per wave.
            // T246.10 TrackF — env-gated dispatch to the column-parallel
            // delta_net_step_tree_bf16_opt variant. Default OFF preserves
            // bit-exact parity with the baseline kernel.
            let use_dn_opt = delta_net_opt_enabled();
            if in_prefill_capture {
                // Linear-chain : one row per depth, depth = row index.
                let (lin_p, _gl) = self.tree_ssm_wave_indices_linear.device_ptr(&self.stream);
                let i32_sz = std::mem::size_of::<i32>() as u64;
                for d_idx in 0..tree_size {
                    let wave_p = lin_p + (d_idx as u64) * i32_sz;
                    let launch_res = if use_dn_opt {
                        self.kernels.delta_net_step_tree_bf16_opt(
                            &self.stream,
                            tssm_qv_p,
                            tssm_kv_p,
                            tssm_out_p,   // V
                            tssm_alpha_p, // gate
                            tssm_beta_p,  // beta
                            parents_dev_p,
                            wave_p,
                            tree_states_p,
                            tssm_out_p,
                            1, // wave_size
                            n_v as i32,
                            head_kv as i32,
                        )
                    } else {
                        self.kernels.delta_net_step_tree_bf16(
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
                    };
                    launch_res.map_err(|e| {
                        LlmError::Backend(format!(
                            "hyb delta_net_tree (capture) d{d_idx} l{li}: {e:?}"
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
                        .map_err(|e| LlmError::Backend(format!("hyb wave H2D d{depth}: {e:?}")))?;

                    // delta_net_step_tree_bf16(q, k, v, gate, beta, parents,
                    //   wave_indices, tree_states, out, wave_size, n_heads,
                    //   head_dim).
                    let launch_res = if use_dn_opt {
                        self.kernels.delta_net_step_tree_bf16_opt(
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
                    } else {
                        self.kernels.delta_net_step_tree_bf16(
                            &self.stream,
                            tssm_qv_p,
                            tssm_kv_p,
                            tssm_out_p,   // V (we copied it above)
                            tssm_alpha_p, // gate (alpha after softplus*ssm_a)
                            tssm_beta_p,  // beta
                            parents_dev_p,
                            tssm_wave_p,
                            tree_states_p,
                            // out destination : we re-use tssm_out_buf
                            // since it holds the input V which we no
                            // longer need after the kernel reads it
                            // (within the same launch).
                            tssm_out_p,
                            wave.len() as i32,
                            n_v as i32,
                            head_kv as i32,
                        )
                    };
                    launch_res.map_err(|e| {
                        LlmError::Backend(format!("hyb delta_net_tree d{depth} l{li}: {e:?}"))
                    })?;
                }
            }
        }

        // 11. ssm_norm per head + multiply by silu(z) per row :
        //   Fused (T246.8 A1.2) : ssm_post_step_bf16 replaces
        //   rms_norm_bf16 + silu_bf16 + mul_inplace_bf16 (3→1) per row
        //   when RUSTORCH_SSM_FUSE=1 (default).
        for r in 0..tree_size {
            let out_r = tssm_out_p + (r as u64) * row_value;
            let z_r = tssm_z_p + (r as u64) * row_value;
            unsafe {
                let (sn, _g) = ssm.ssm_norm.device_ptr(&self.stream);
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
                        .map_err(|e| LlmError::Backend(format!("hyb ssm_post_step r{r}: {e:?}")))?;
                } else {
                    self.kernels
                        .rms_norm_bf16(&self.stream, out_r, sn, eps, head_kv as i32, n_v as i32)
                        .map_err(|e| LlmError::Backend(format!("hyb ssm_norm r{r}: {e:?}")))?;
                    self.kernels
                        .silu_bf16(&self.stream, z_r, value_dim as i32)
                        .map_err(|e| LlmError::Backend(format!("hyb silu z r{r}: {e:?}")))?;
                    self.kernels
                        .mul_inplace_bf16(&self.stream, out_r, z_r, value_dim as i32)
                        .map_err(|e| LlmError::Backend(format!("hyb mul gated r{r}: {e:?}")))?;
                }
            }
        }

        // 12. ssm_out @ gated → tree_h ; residual is already in tres_p
        //     and will be added by the dispatcher's FFN-pre block (so we
        //     do NOT add residual here).
        let out_k_ok = ssm.ssm_out.shape().1 % 256 == 0;
        if use_gemm && out_k_ok {
            ssm.ssm_out.dispatch_matmul_mvar(
                &self.kernels,
                &self.stream,
                tree_size,
                tssm_out_p,
                th_p,
            )?;
        } else {
            for r in 0..tree_size {
                let out_r = tssm_out_p + (r as u64) * row_value;
                let h_r = th_p + (r as u64) * row_h;
                ssm.ssm_out
                    .dispatch_matmul_m1(&self.kernels, &self.stream, out_r, h_r, x_q8_p)?;
            }
        }
        // Residual : h += residual (= pre-mixer hidden state).
        unsafe {
            self.kernels
                .add_inplace_bf16(&self.stream, th_p, _tres_p, (tree_size * d) as i32)
                .map_err(|e| LlmError::Backend(format!("hyb ssm residual l{li}: {e:?}")))?;
        }

        Ok(())
    }

    /// T246.7 P1.4 — multi-token tree forward for the pure-transformer
    /// variant (Qwen3PureTransformer / qwen2 / qwen3 family).
    ///
    /// Implements the full forward pass for a `tree_size`-node draft tree :
    ///
    /// 1. Embedding lookup for all `tree_size` tokens (one batched call ;
    ///    `embedding_lookup_bf16` already supports `seq` argument).
    /// 2. Per attention layer (every layer is attention for pure
    ///    transformer) :
    ///    - a. RMSNorm + QKV + per-head Q/K-norm + RoPE — looped per row,
    ///      since these are HBM-bound and N small kernels pipeline well.
    ///      RoPE uses `rope_partial_bf16` (explicit pos = base + depth[r]).
    ///    - b. `kv_append_tree_bf16` — ONE call writes all tree positions.
    ///    - c. `gqa_decode_tree_bf16` — ONE call computes attention for all
    ///      tree rows with parents/depths-aware causal mask.
    ///    - d. w_o + residual add per row.
    ///    - e. RMSNorm + Dense FFN (gate/up/down + SwiGLU) per row + residual.
    /// 3. Final RMSNorm + LM head per row.
    /// 4. `argmax_logits_tree_bf16` — ONE call writes all tree_size
    ///    argmax tokens.
    /// 5. DtoH the argmax tokens → host buffer.
    /// 6. CPU-side acceptance walk (per design decision db18cf7a) :
    ///    starting from root, find the child whose token equals the
    ///    model's prediction at the parent row, accept, recurse.
    /// 7. Compact accepted KV slots into contiguous positions
    ///    `[pos_dev .. pos_dev + accept_len)` (BFS guarantees src ≥ dst,
    ///    so a forward in-place copy is safe).
    /// 8. Advance device counters via `add_u32_dev`.
    ///
    /// Per RFC design decision D4 : graph capture is NOT used for the
    /// verify pass (variable input shape). The single-token decode_step
    /// graph is preserved across calls.
    #[allow(clippy::too_many_lines)]
    fn decode_step_tree_pure_transformer(
        &mut self,
        drafts: &[u32],
        parents: &[i32],
        depths: &[u16],
    ) -> Result<Vec<u32>, LlmError> {
        self.decode_step_tree_pure_transformer_inner(drafts, parents, depths, false)
    }

    /// T246.10 A6 — shared implementation behind
    /// `decode_step_tree_pure_transformer` and `prefill_tokens`. When
    /// `force_accept_all = true`, the acceptance walk is skipped and every
    /// BFS-ordered tree node is force-accepted (prefill linear-chain mode).
    #[allow(clippy::too_many_lines)]
    fn decode_step_tree_pure_transformer_inner(
        &mut self,
        drafts: &[u32],
        parents: &[i32],
        depths: &[u16],
        force_accept_all: bool,
    ) -> Result<Vec<u32>, LlmError> {
        self.decode_step_tree_pure_transformer_inner_capture(
            drafts,
            parents,
            depths,
            force_accept_all,
            false,
        )
    }

    /// T246.10 TrackI — full pure-transformer prefill body with explicit
    /// capture-mode flag. See `decode_step_tree_hybrid_inner_capture` for
    /// the contract (skip-HtoD-at-top + skip-DtoH-at-bottom + no host
    /// position bookkeeping when capturing).
    #[allow(clippy::too_many_lines)]
    fn decode_step_tree_pure_transformer_inner_capture(
        &mut self,
        drafts: &[u32],
        parents: &[i32],
        depths: &[u16],
        force_accept_all: bool,
        in_prefill_capture: bool,
    ) -> Result<Vec<u32>, LlmError> {
        debug_assert!(
            !in_prefill_capture || force_accept_all,
            "in_prefill_capture requires force_accept_all"
        );
        use cudarc::driver::{DevicePtr, DevicePtrMut};

        let cfg = self.config.clone();
        let d = cfg.d;
        let f = cfg.f;
        let n_q = cfg.n_q_heads;
        let n_kv = cfg.n_kv_heads;
        let head_dim = cfg.head_dim();
        let q_dim = n_q * head_dim;
        let kv_dim = n_kv * head_dim;
        let eps = cfg.rms_eps;
        let rope_dim = cfg.rope_dim;
        let vocab = cfg.vocab;
        let tree_size = drafts.len();
        debug_assert!(tree_size > 1 && tree_size <= MAX_TREE_SIZE);

        // T246.10 A6b.3 — GEMM prefill : when `force_accept_all` is true AND
        // tree_size >= GEMM_PREFILL_MIN_M AND RUSTORCH_GEMM_PREFILL=1, replace
        // the per-row `dispatch_matmul_m1` loops for Q/K/V/O + gate/up/down
        // with a single batched `dispatch_matmul_mvar` call (M=tree_size).
        let use_gemm =
            force_accept_all && tree_size >= GEMM_PREFILL_MIN_M && gemm_prefill_enabled();

        // Snapshot the base position before this verify pass so we can
        // recover per-row absolute positions for RoPE (= base + depth[r]).
        let base_position = self.position;

        // ── 0. Upload tree descriptors (drafts / parents / depths) ────────
        // `tree_drafts` and friends are sized MAX_TREE_SIZE ; only the
        // first tree_size entries are read by the kernels.
        // T246.10 TrackI — capture mode caller pre-uploads OUTSIDE the
        // capture region. See decode_step_tree_hybrid_inner_capture.
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

        // ── 1. Zero per-tree scratch (sized MAX_TREE_SIZE × per-token) ────
        self.stream
            .memset_zeros(&mut self.scratch.tree_h)
            .map_err(|e| LlmError::Backend(format!("zero tree_h: {e:?}")))?;
        self.stream
            .memset_zeros(&mut self.scratch.tree_h_norm)
            .map_err(|e| LlmError::Backend(format!("zero tree_h_norm: {e:?}")))?;
        self.stream
            .memset_zeros(&mut self.scratch.tree_residual)
            .map_err(|e| LlmError::Backend(format!("zero tree_residual: {e:?}")))?;
        self.stream
            .memset_zeros(&mut self.scratch.tree_q_buf)
            .map_err(|e| LlmError::Backend(format!("zero tree_q_buf: {e:?}")))?;
        self.stream
            .memset_zeros(&mut self.scratch.tree_k_buf)
            .map_err(|e| LlmError::Backend(format!("zero tree_k_buf: {e:?}")))?;
        self.stream
            .memset_zeros(&mut self.scratch.tree_v_buf)
            .map_err(|e| LlmError::Backend(format!("zero tree_v_buf: {e:?}")))?;
        self.stream
            .memset_zeros(&mut self.scratch.tree_attn_out)
            .map_err(|e| LlmError::Backend(format!("zero tree_attn_out: {e:?}")))?;
        self.stream
            .memset_zeros(&mut self.scratch.tree_gate_buf)
            .map_err(|e| LlmError::Backend(format!("zero tree_gate_buf: {e:?}")))?;
        self.stream
            .memset_zeros(&mut self.scratch.tree_up_buf)
            .map_err(|e| LlmError::Backend(format!("zero tree_up_buf: {e:?}")))?;
        self.stream
            .memset_zeros(&mut self.scratch.tree_logits)
            .map_err(|e| LlmError::Backend(format!("zero tree_logits: {e:?}")))?;

        // Pre-extract device pointers & their gigantic guard locks (drop
        // at end of unsafe block). The `tree_*` buffers are sized
        // MAX_TREE_SIZE-rows ; row r lives at offset r*per_row.
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
            x_q8_p,
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
            let q8 = if self.use_dp4a_q4k {
                let (s, _g17) = self.scratch.x_q8_scratch.device_ptr_mut(&self.stream);
                s
            } else {
                0u64
            };
            (a, b, c, d_, e, f_, g, h, i, j, k, l, m, n_, o, p, q, q8)
        };

        // Per-row byte offsets (BF16 = 2 bytes).
        let bf16_sz = std::mem::size_of::<half::bf16>() as u64;
        let row_h = (d as u64) * bf16_sz;
        let row_q = (q_dim as u64) * bf16_sz;
        let row_kv = (kv_dim as u64) * bf16_sz;
        let row_logits = (vocab as u64) * bf16_sz;

        // ── 2. Embedding lookup for all tree_size tokens ──────────────────
        unsafe {
            let (te_p, _g) = self.token_emb.device_ptr(&self.stream);
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

        // ── 3. Per-layer forward (every layer is attention for pure
        //       transformer ; FFN is dense SwiGLU) ──────────────────────────
        for (li, block) in self.blocks.iter_mut().enumerate() {
            let attn = match block {
                BlockQ4K::Attn(a) => a,
                BlockQ4K::Ssm(_) => {
                    return Err(LlmError::Backend(format!(
                        "decode_step_tree_pure_transformer: layer {li} is SSM \
                         but variant={:?} should have only Attn layers \
                         (loader/config bug)",
                        self.config.variant
                    )));
                },
            };

            // residual <- h (per row)
            unsafe {
                self.kernels
                    .copy_bf16(&self.stream, tres_p, th_p, (tree_size * d) as i32)
                    .map_err(|e| LlmError::Backend(format!("tree copy res: {e:?}")))?;
            }

            // RMSNorm h_norm = rms_norm(h, attn_norm) — per row, vectorized
            // via the rms_norm_bf16 batch arg (one call for tree_size rows).
            unsafe {
                let (an, _g) = attn.attn_norm.device_ptr(&self.stream);
                self.kernels
                    .copy_bf16(&self.stream, thn_p, th_p, (tree_size * d) as i32)
                    .map_err(|e| LlmError::Backend(format!("tree copy h_norm: {e:?}")))?;
                self.kernels
                    .rms_norm_bf16(&self.stream, thn_p, an, eps, d as i32, tree_size as i32)
                    .map_err(|e| LlmError::Backend(format!("tree rms_norm attn: {e:?}")))?;
            }

            // Per-row Q, K, V projections.
            if use_gemm {
                // T246.10 A6b.3 — one mvar call per Q/K/V replaces tree_size
                // M=1 SGEMVs. Input thn_p[tree_size, d] row-major, outputs
                // tq_p[tree_size, q_dim], tk_p[tree_size, kv_dim],
                // tv_p[tree_size, kv_dim] all row-major.
                attn.w_q.dispatch_matmul_mvar(
                    &self.kernels,
                    &self.stream,
                    tree_size,
                    thn_p,
                    tq_p,
                )?;
                attn.w_k.dispatch_matmul_mvar(
                    &self.kernels,
                    &self.stream,
                    tree_size,
                    thn_p,
                    tk_p,
                )?;
                attn.w_v.dispatch_matmul_mvar(
                    &self.kernels,
                    &self.stream,
                    tree_size,
                    thn_p,
                    tv_p,
                )?;
            } else {
                for r in 0..tree_size {
                    let hn_r = thn_p + (r as u64) * row_h;
                    let q_r = tq_p + (r as u64) * row_q;
                    let k_r = tk_p + (r as u64) * row_kv;
                    let v_r = tv_p + (r as u64) * row_kv;
                    attn.w_q
                        .dispatch_matmul_m1(&self.kernels, &self.stream, hn_r, q_r, x_q8_p)?;
                    attn.w_k
                        .dispatch_matmul_m1(&self.kernels, &self.stream, hn_r, k_r, x_q8_p)?;
                    attn.w_v
                        .dispatch_matmul_m1(&self.kernels, &self.stream, hn_r, v_r, x_q8_p)?;
                }
            }

            // P1.5 — Qwen2 : add Q/K/V biases per row (broadcast over rows).
            if cfg.variant.attn_has_qkv_bias() {
                unsafe {
                    let (bq, _gbq) = attn
                        .b_q
                        .as_ref()
                        .expect("qwen2: b_q present")
                        .device_ptr(&self.stream);
                    let (bk, _gbk) = attn
                        .b_k
                        .as_ref()
                        .expect("qwen2: b_k present")
                        .device_ptr(&self.stream);
                    let (bv, _gbv) = attn
                        .b_v
                        .as_ref()
                        .expect("qwen2: b_v present")
                        .device_ptr(&self.stream);
                    for r in 0..tree_size {
                        let q_r = tq_p + (r as u64) * row_q;
                        let k_r = tk_p + (r as u64) * row_kv;
                        let v_r = tv_p + (r as u64) * row_kv;
                        self.kernels
                            .add_inplace_bf16(&self.stream, q_r, bq, q_dim as i32)
                            .map_err(|e| LlmError::Backend(format!("tree q_bias: {e:?}")))?;
                        self.kernels
                            .add_inplace_bf16(&self.stream, k_r, bk, kv_dim as i32)
                            .map_err(|e| LlmError::Backend(format!("tree k_bias: {e:?}")))?;
                        self.kernels
                            .add_inplace_bf16(&self.stream, v_r, bv, kv_dim as i32)
                            .map_err(|e| LlmError::Backend(format!("tree v_bias: {e:?}")))?;
                    }
                }
            }

            // Per-head Q-norm and K-norm — one call per row covering all
            // heads (rms_norm_bf16 with batch=n_q / n_kv).
            // Skipped for qwen2 (no QK-norm).
            if cfg.variant.attn_has_qk_norm() {
                unsafe {
                    let (qn, _g1) = attn
                        .q_norm
                        .as_ref()
                        .expect("qk_norm variant: q_norm present")
                        .device_ptr(&self.stream);
                    let (kn, _g2) = attn
                        .k_norm
                        .as_ref()
                        .expect("qk_norm variant: k_norm present")
                        .device_ptr(&self.stream);
                    self.kernels
                        .rms_norm_bf16(
                            &self.stream,
                            tq_p,
                            qn,
                            eps,
                            head_dim as i32,
                            (tree_size * n_q) as i32,
                        )
                        .map_err(|e| LlmError::Backend(format!("tree q_norm: {e:?}")))?;
                    self.kernels
                        .rms_norm_bf16(
                            &self.stream,
                            tk_p,
                            kn,
                            eps,
                            head_dim as i32,
                            (tree_size * n_kv) as i32,
                        )
                        .map_err(|e| LlmError::Backend(format!("tree k_norm: {e:?}")))?;
                }
            }

            // RoPE per row with explicit pos = base_position + depths[r].
            // base_position is the snapshot before this call ; that is the
            // root's position. Children at depth d use base_position + d.
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
                        .map_err(|e| LlmError::Backend(format!("tree rope q: {e:?}")))?;
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
                        .map_err(|e| LlmError::Backend(format!("tree rope k: {e:?}")))?;
                }
            }

            // Append all tree positions to KV cache in ONE kernel.
            // pos_dev currently points at the next free slot (= old pos +
            // 1 in the existing convention). kv_append_tree writes at
            // [pos_dev + r] for r in 0..tree_size.
            let attn_idx = get_attn_layer_idx(&cfg, li);
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
                    .map_err(|e| LlmError::Backend(format!("tree kv_append: {e:?}")))?;
            }

            // Tree-aware GQA decode in ONE kernel pair (partial + combine).
            unsafe {
                let (kc_p, _g1) = kv_cache.k.device_ptr(&self.stream);
                let (vc_p, _g2) = kv_cache.v.device_ptr(&self.stream);
                let (kv_len_p, _g3) = self.kv_len_dev.device_ptr(&self.stream);
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
                        kv_len_p,
                        head_dim as i32,
                        self.max_seq as i32,
                        GQA_N_SPLIT as i32,
                        tree_size as i32,
                    )
                    .map_err(|e| LlmError::Backend(format!("tree gqa_decode: {e:?}")))?;
            }

            // w_o per row : h_r = w_o @ attn_out_r.
            if use_gemm {
                attn.w_o.dispatch_matmul_mvar(
                    &self.kernels,
                    &self.stream,
                    tree_size,
                    tao_p,
                    th_p,
                )?;
            } else {
                for r in 0..tree_size {
                    let ao_r = tao_p + (r as u64) * row_q;
                    let h_r = th_p + (r as u64) * row_h;
                    attn.w_o
                        .dispatch_matmul_m1(&self.kernels, &self.stream, ao_r, h_r, x_q8_p)?;
                }
            }

            // h += residual (per row, single batched call).
            unsafe {
                self.kernels
                    .add_inplace_bf16(&self.stream, th_p, tres_p, (tree_size * d) as i32)
                    .map_err(|e| LlmError::Backend(format!("tree attn residual: {e:?}")))?;
            }

            // ── FFN block ───────────────────────────────────────────────
            // residual <- h (post-attention value)
            unsafe {
                self.kernels
                    .copy_bf16(&self.stream, tres_p, th_p, (tree_size * d) as i32)
                    .map_err(|e| LlmError::Backend(format!("tree copy res ffn: {e:?}")))?;
            }
            // h_norm = rms_norm(h, post_norm) — batched over tree_size rows.
            unsafe {
                let (pn, _g) = attn.post_norm.device_ptr(&self.stream);
                self.kernels
                    .copy_bf16(&self.stream, thn_p, th_p, (tree_size * d) as i32)
                    .map_err(|e| LlmError::Backend(format!("tree copy h_norm ffn: {e:?}")))?;
                self.kernels
                    .rms_norm_bf16(&self.stream, thn_p, pn, eps, d as i32, tree_size as i32)
                    .map_err(|e| LlmError::Backend(format!("tree rms_norm post: {e:?}")))?;
            }

            // Dense SwiGLU FFN per row : gate, up, swiglu, down.
            // Per-row buffer offsets in tree_gate_buf / tree_up_buf use the
            // FFN scratch row size (max(f, q_dim, 2*q_dim)).
            let row_ffn_gate = (self.scratch.tree_gate_buf.len() / MAX_TREE_SIZE) as u64 * bf16_sz;
            let row_ffn_up = (self.scratch.tree_up_buf.len() / MAX_TREE_SIZE) as u64 * bf16_sz;
            match &attn.ffn {
                FfnQ4K::Dense { gate, up, down } => {
                    // Stride guard : the mvar kernel writes y[m * N + col] for
                    // N = gate.shape().0 (= f). The per-row scratch is strided
                    // at scratch_gate = max(f, q_dim) and scratch_up =
                    // max(f, 2*q_dim) BF16 elements. The GEMM path is only
                    // bit-safe when the GEMM row stride (N) matches the scratch
                    // row stride. Falls back to per-row when mismatched.
                    let gate_stride_ok = gate.shape().0 * bf16_sz as usize == row_ffn_gate as usize;
                    let up_stride_ok = up.shape().0 * bf16_sz as usize == row_ffn_up as usize;
                    if use_gemm && gate_stride_ok && up_stride_ok {
                        gate.dispatch_matmul_mvar(
                            &self.kernels,
                            &self.stream,
                            tree_size,
                            thn_p,
                            tgate_p,
                        )?;
                        up.dispatch_matmul_mvar(
                            &self.kernels,
                            &self.stream,
                            tree_size,
                            thn_p,
                            tup_p,
                        )?;
                        // SwiGLU per row : still N small launches (or could be
                        // batched but per-row is fine, swiglu is fast).
                        for r in 0..tree_size {
                            let gate_r = tgate_p + (r as u64) * row_ffn_gate;
                            let up_r = tup_p + (r as u64) * row_ffn_up;
                            unsafe {
                                self.kernels
                                    .swiglu_bf16(&self.stream, gate_r, up_r, gate_r, f as i32)
                                    .map_err(|e| {
                                        LlmError::Backend(format!("tree swiglu r{r}: {e:?}"))
                                    })?;
                            }
                        }
                        down.dispatch_matmul_mvar(
                            &self.kernels,
                            &self.stream,
                            tree_size,
                            tgate_p,
                            th_p,
                        )?;
                    } else {
                        for r in 0..tree_size {
                            let hn_r = thn_p + (r as u64) * row_h;
                            let gate_r = tgate_p + (r as u64) * row_ffn_gate;
                            let up_r = tup_p + (r as u64) * row_ffn_up;
                            let h_r = th_p + (r as u64) * row_h;
                            gate.dispatch_matmul_m1(
                                &self.kernels,
                                &self.stream,
                                hn_r,
                                gate_r,
                                x_q8_p,
                            )?;
                            up.dispatch_matmul_m1(&self.kernels, &self.stream, hn_r, up_r, x_q8_p)?;
                            unsafe {
                                self.kernels
                                    .swiglu_bf16(&self.stream, gate_r, up_r, gate_r, f as i32)
                                    .map_err(|e| {
                                        LlmError::Backend(format!("tree swiglu r{r}: {e:?}"))
                                    })?;
                            }
                            down.dispatch_matmul_m1(
                                &self.kernels,
                                &self.stream,
                                gate_r,
                                h_r,
                                x_q8_p,
                            )?;
                        }
                    }
                },
                FfnQ4K::Moe(_) => {
                    return Err(LlmError::Backend(
                        "decode_step_tree_pure_transformer: MoE FFN not \
                         supported for pure-transformer variant"
                            .into(),
                    ));
                },
            }

            // h += residual (post-FFN).
            unsafe {
                self.kernels
                    .add_inplace_bf16(&self.stream, th_p, tres_p, (tree_size * d) as i32)
                    .map_err(|e| LlmError::Backend(format!("tree ffn residual: {e:?}")))?;
            }
        }

        // ── 4. Final RMSNorm + LM head per row ────────────────────────────
        unsafe {
            let (fn_p, _g) = self.final_norm.device_ptr(&self.stream);
            self.kernels
                .rms_norm_bf16(&self.stream, th_p, fn_p, eps, d as i32, tree_size as i32)
                .map_err(|e| LlmError::Backend(format!("tree final_norm: {e:?}")))?;
        }
        // T246.10 HOTFIX-A6b — batch lm_head through dispatch_matmul_mvar when
        // in GEMM prefill mode. See decode_step_tree_hybrid_inner for full
        // rationale. Falls back to the per-row split-K dispatch when use_gemm
        // is false or K%256 != 0 (preserves bit-exact parity for decode and
        // legacy paths).
        let lm_k_ok = self.lm_head.shape().1 % 256 == 0;
        if use_gemm && lm_k_ok {
            self.lm_head.dispatch_matmul_mvar(
                &self.kernels,
                &self.stream,
                tree_size,
                th_p,
                tlogits_p,
            )?;
        } else {
            for r in 0..tree_size {
                let h_r = th_p + (r as u64) * row_h;
                let logits_r = tlogits_p + (r as u64) * row_logits;
                // T246.8 A5 — split-K dispatch (env-gated).
                self.dispatch_lm_head(h_r, logits_r, x_q8_p)?;
            }
        }

        // ── 5. Argmax over all tree_size logits rows ──────────────────────
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

        // ── 6. DtoH the argmax tokens to pinned host buffer ───────────────
        //
        // T246.10 TrackI — in capture mode the DtoH + host accept walk
        // MUST happen AFTER `end_capture` (we can't read host-side
        // pinned memory from inside a capturing stream). The caller
        // (`prefill_tokens`) does this read post-replay. For linear-
        // chain prefill the accept walk degenerates to
        // `accepted_indices = [0..tree_size]` and the compact loop is
        // a no-op (each `src == i`).
        let (accepted_indices, accepted_tokens): (Vec<usize>, Vec<u32>) = if in_prefill_capture {
            let idx: Vec<usize> = (0..tree_size).collect();
            (idx, Vec::new())
        } else {
            self.stream
                .memcpy_dtoh(
                    &self.scratch.tree_argmax,
                    &mut self.scratch.tree_argmax_host_pinned,
                )
                .map_err(|e| LlmError::Backend(format!("dtoh tree_argmax: {e:?}")))?;
            let argmax_host = self
                .scratch
                .tree_argmax_host_pinned
                .as_slice()
                .map_err(|e| LlmError::Backend(format!("read pinned tree_argmax: {e:?}")))?;

            // ── 7. CPU acceptance walk ────────────────────────────────
            // For each tree node r in [0, tree_size), `argmax_host[r]` is
            // the model's prediction for the next token if it had been
            // fed the ancestor chain ending at r (root → ... → r).
            //
            // T246.10 A6 — prefill mode (`force_accept_all = true`)
            // bypasses the walk and force-accepts every BFS-ordered node.
            if force_accept_all {
                let idx: Vec<usize> = (0..tree_size).collect();
                let tok: Vec<u32> = (0..tree_size).map(|r| argmax_host[r]).collect();
                (idx, tok)
            } else {
                // Build a child-list once : children[parent] = Vec<child_idx>.
                let mut children: Vec<Vec<usize>> = vec![Vec::new(); tree_size];
                for (r, &p) in parents.iter().enumerate().skip(1) {
                    children[p as usize].push(r);
                }

                let mut a_indices: Vec<usize> = vec![0]; // root
                let mut a_tokens: Vec<u32> = vec![argmax_host[0]];
                let mut cur = 0usize;
                loop {
                    let next_tok = argmax_host[cur];
                    let mut found: Option<usize> = None;
                    for &c in &children[cur] {
                        if drafts[c] == next_tok {
                            found = Some(c);
                            break;
                        }
                    }
                    match found {
                        Some(c) => {
                            a_indices.push(c);
                            a_tokens.push(argmax_host[c]);
                            cur = c;
                        },
                        None => break,
                    }
                }
                (a_indices, a_tokens)
            }
        };
        let accept_len = accepted_indices.len();
        debug_assert!(accept_len >= 1);

        // ── 8. Compact accepted KV slots into [pos_dev .. pos_dev+accept_len)
        // For each i in 1..accept_len, the accepted node's K/V was written
        // at slot pos_dev + accepted_indices[i] (BFS index ≥ i). We need
        // it at slot pos_dev + i. Per BFS invariant src ≥ dst, so a
        // forward in-place copy (smallest dst first) is safe.
        for (li, block) in self.blocks.iter_mut().enumerate() {
            if !matches!(block, BlockQ4K::Attn(_)) {
                continue;
            }
            let attn_idx = get_attn_layer_idx(&cfg, li);
            let kv_cache = &mut self.kv_caches[attn_idx];
            unsafe {
                let (kc_p, _g1) = kv_cache.k.device_ptr_mut(&self.stream);
                let (vc_p, _g2) = kv_cache.v.device_ptr_mut(&self.stream);
                for (i, &src) in accepted_indices.iter().enumerate().take(accept_len).skip(1) {
                    if src == i {
                        continue;
                    }
                    let src_off = (base_position as u64 + src as u64) * (kv_dim as u64);
                    let dst_off = (base_position as u64 + i as u64) * (kv_dim as u64);
                    // copy_bf16(dst, src, n) — n is element count.
                    self.kernels
                        .copy_bf16(
                            &self.stream,
                            kc_p + dst_off * bf16_sz,
                            kc_p + src_off * bf16_sz,
                            kv_dim as i32,
                        )
                        .map_err(|e| LlmError::Backend(format!("tree compact K l{li}: {e:?}")))?;
                    self.kernels
                        .copy_bf16(
                            &self.stream,
                            vc_p + dst_off * bf16_sz,
                            vc_p + src_off * bf16_sz,
                            kv_dim as i32,
                        )
                        .map_err(|e| LlmError::Backend(format!("tree compact V l{li}: {e:?}")))?;
                }
            }
        }

        // ── 9. Advance device counters by accept_len ──────────────────────
        unsafe {
            let (pos_p, _g_pos) = self.position_dev.device_ptr_mut(&self.stream);
            self.kernels
                .add_u32_dev(&self.stream, pos_p, accept_len as i32)
                .map_err(|e| LlmError::Backend(format!("add_u32_dev pos: {e:?}")))?;
        }
        unsafe {
            let (kvl_p, _g_kvl) = self.kv_len_dev.device_ptr_mut(&self.stream);
            self.kernels
                .add_u32_dev(&self.stream, kvl_p, accept_len as i32)
                .map_err(|e| LlmError::Backend(format!("add_u32_dev kv_len: {e:?}")))?;
        }
        // T246.10 TrackI — host-side position bookkeeping moves to the
        // caller (`prefill_tokens`) in capture mode.
        if !in_prefill_capture {
            self.position += accept_len;
        }

        // The CUDA Graph (if any was captured for the single-token path) is
        // still valid : we did not modify any buffer it references.

        Ok(accepted_tokens)
    }
}

/// T246.6.5 — MoE FFN forward step (Qwen3.6-35B-A3B).
///
/// Computes one token's MoE output : top-K routed experts (each a Q4_K
/// SwiGLU FFN) + parallel shared expert. Writes the routed sum into `h_p`
/// (which is then accumulated with the residual by the caller).
///
/// Caller passes the typed slice references for the three buffers we
/// memcpy_dtov from (top-K indices, top-K weights, shared-expert dot
/// product) — Rust's borrow checker accepts these alongside the loop's
/// `&mut self.blocks` mutable borrow because they're disjoint fields of
/// `self.scratch`. The same buffers' raw u64 ptrs are passed for kernel
/// arguments (since `LlmKernels::*_bf16` APIs are u64-typed).
///
/// Semantics mirror `qwen35_cpu::ffn_moe_forward`.
#[allow(clippy::too_many_arguments)]
fn moe_ffn_forward_step(
    moe: &MoeFfnQ4K,
    kernels: &LlmKernels,
    stream: &Arc<CudaStream>,
    cfg: &Qwen35Config,
    h_norm_p: u64,
    h_p: u64,
    x_q8_p: u64,
    router_logits_p: u64,
    topk_idx_p: u64,
    topk_idx_slice: &CudaSlice<i32>,
    topk_w_p: u64,
    topk_w_slice: &CudaSlice<half::bf16>,
    expert_gate_p: u64,
    expert_up_p: u64,
    expert_out_p: u64,
    shexp_dot_p: u64,
    shexp_dot_slice: &CudaSlice<half::bf16>,
) -> Result<(), LlmError> {
    let d = cfg.d as i32;
    let ef = cfg.expert_f as i32;
    let n_e = cfg.n_experts as i32;
    let k = cfg.n_experts_used as i32;

    // ---- 1. Router logits ----
    moe.gate_inp
        .dispatch_matmul_m1(kernels, stream, h_norm_p, router_logits_p, x_q8_p)?;

    // ---- 2. Top-K softmax → indices + renormalized weights on device ----
    unsafe {
        kernels
            .topk_softmax_bf16(stream, router_logits_p, topk_idx_p, topk_w_p, n_e, k)
            .map_err(|e| LlmError::Backend(format!("topk_softmax: {e:?}")))?;
    }

    // ---- 3. Download top-K to host (one DtoH sync per layer) ----
    // Note : this sync is incompatible with CUDA Graph capture ; the model
    // ctor disables graph capture when variant == Moe (handled elsewhere
    // by the `position == 1 && decode_graph.is_none()` guard, since the
    // capture path branch is taken only after step 0 warmup).
    let topk_idx: Vec<i32> = stream
        .memcpy_dtov(topk_idx_slice)
        .map_err(|e| LlmError::Backend(format!("dtov topk_idx: {e:?}")))?;
    let topk_w_bf: Vec<half::bf16> = stream
        .memcpy_dtov(topk_w_slice)
        .map_err(|e| LlmError::Backend(format!("dtov topk_w: {e:?}")))?;

    // ---- 4. Zero h_p (start fresh accumulation for routed sum) ----
    // Use scaled_add with y=x=h_p, alpha=-1 → h_p -= h_p = 0 (aliasing OK
    // since each thread reads then writes its own index in one go).
    unsafe {
        kernels
            .scaled_add_inplace_bf16(stream, h_p, h_p, -1.0, d)
            .map_err(|e| LlmError::Backend(format!("zero h_p: {e:?}")))?;
    }

    // ---- 5. Routed experts loop ----
    for i in 0..k as usize {
        let e_idx = topk_idx[i] as usize;
        let w_e = topk_w_bf[i].to_f32();
        if e_idx >= moe.gate_exps.len() {
            return Err(LlmError::Backend(format!(
                "MoE expert idx {e_idx} out of range (n_experts={})",
                moe.gate_exps.len()
            )));
        }

        moe.gate_exps[e_idx].dispatch_matmul_m1(
            kernels,
            stream,
            h_norm_p,
            expert_gate_p,
            x_q8_p,
        )?;
        moe.up_exps[e_idx].dispatch_matmul_m1(kernels, stream, h_norm_p, expert_up_p, x_q8_p)?;
        unsafe {
            kernels
                .swiglu_bf16(stream, expert_gate_p, expert_up_p, expert_gate_p, ef)
                .map_err(|e| LlmError::Backend(format!("swiglu expert {e_idx}: {e:?}")))?;
        }
        moe.down_exps[e_idx].dispatch_matmul_m1(
            kernels,
            stream,
            expert_gate_p,
            expert_out_p,
            x_q8_p,
        )?;
        unsafe {
            kernels
                .scaled_add_inplace_bf16(stream, h_p, expert_out_p, w_e, d)
                .map_err(|e| LlmError::Backend(format!("scaled_add expert {e_idx}: {e:?}")))?;
        }
    }

    // ---- 6. Shared expert (parallel path) ----
    // shexp_dot = gate_inp_shexp · h_norm  (1×D BF16 matmul)
    unsafe {
        use cudarc::driver::DevicePtr;
        let (gip_p, _g) = moe.gate_inp_shexp.device_ptr(stream);
        kernels
            .sgemv_bf16_bf16(stream, gip_p, h_norm_p, shexp_dot_p, 1, d)
            .map_err(|e| LlmError::Backend(format!("shexp dot: {e:?}")))?;
    }
    let dot_bf: Vec<half::bf16> = stream
        .memcpy_dtov(shexp_dot_slice)
        .map_err(|e| LlmError::Backend(format!("dtov shexp_dot: {e:?}")))?;
    let shexp_w = {
        let v = dot_bf[0].to_f32();
        1.0_f32 / (1.0 + (-v).exp())
    };

    moe.gate_shexp
        .dispatch_matmul_m1(kernels, stream, h_norm_p, expert_gate_p, x_q8_p)?;
    moe.up_shexp
        .dispatch_matmul_m1(kernels, stream, h_norm_p, expert_up_p, x_q8_p)?;
    unsafe {
        kernels
            .swiglu_bf16(stream, expert_gate_p, expert_up_p, expert_gate_p, ef)
            .map_err(|e| LlmError::Backend(format!("swiglu shexp: {e:?}")))?;
    }
    moe.down_shexp
        .dispatch_matmul_m1(kernels, stream, expert_gate_p, expert_out_p, x_q8_p)?;
    unsafe {
        kernels
            .scaled_add_inplace_bf16(stream, h_p, expert_out_p, shexp_w, d)
            .map_err(|e| LlmError::Backend(format!("scaled_add shexp: {e:?}")))?;
    }

    Ok(())
}

/// T246.8 A2.3 — async (zero-host-sync) MoE FFN forward step. Same
/// semantics as `moe_ffn_forward_step` but every per-layer host sync is
/// eliminated :
///
///   - top-K indices and weights are NEVER read back to host : the
///     indexed SGEMV kernels (q4k/q5k/q6k v3 + bf16, T246.8 A2.1) read
///     `topk_indices[slot]` from device memory and use it to fetch the
///     per-expert weight pointer from a pre-cached device array.
///   - top-K weights are consumed by `scaled_add_inplace_bf16_devscalar`
///     (alpha read from device).
///   - shared-expert sigmoid is computed in-place on device via
///     `sigmoid_inplace_bf16` then consumed by the same devscalar
///     scaled_add primitive.
///
/// The Q4_K dp4a x_q8_1 staging is computed ONCE per MoE layer before
/// the per-expert loop (since `h_norm` is the same input for all
/// experts) — this matches what `dispatch_matmul_m1` does internally
/// per call but amortizes the quantize across the k=8 routed experts.
///
/// Net result : zero `memcpy_dtov` in the body. The full decode_step
/// is now CUDA-Graph-capturable for the MoE variant.
#[allow(clippy::too_many_arguments)]
fn moe_ffn_forward_step_async(
    moe: &MoeFfnQ4K,
    kernels: &LlmKernels,
    stream: &Arc<CudaStream>,
    cfg: &Qwen35Config,
    h_norm_p: u64,
    h_p: u64,
    x_q8_p: u64,
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
    let k = cfg.n_experts_used as i32;

    // ---- 1. Router logits ----
    moe.gate_inp
        .dispatch_matmul_m1(kernels, stream, h_norm_p, router_logits_p, x_q8_p)?;

    // ---- 2. Top-K softmax → indices + renormalized weights on device ----
    unsafe {
        kernels
            .topk_softmax_bf16(stream, router_logits_p, topk_idx_p, topk_w_p, n_e, k)
            .map_err(|e| LlmError::Backend(format!("topk_softmax: {e:?}")))?;
    }

    // ---- 3. Zero h_p (clean primitive ; replaces self-aliasing trick) ----
    unsafe {
        kernels
            .zero_bf16(stream, h_p, d)
            .map_err(|e| LlmError::Backend(format!("zero h_p: {e:?}")))?;
    }

    // ---- 4. Routed experts loop (zero host sync) ----
    // `dispatch_indexed_matmul_m1` internally quantizes its own input
    // when called for Q4K with x_q8_p != 0, mirroring the per-call
    // quantize that `QuantTensor::dispatch_matmul_m1` does. So we don't
    // need to manage the staging buffer outside the dispatcher.
    let (g_ptrs_p, _gg) = moe.gate_exp_ptrs_dev.device_ptr(stream);
    let (u_ptrs_p, _gu) = moe.up_exp_ptrs_dev.device_ptr(stream);
    let (d_ptrs_p, _gd) = moe.down_exp_ptrs_dev.device_ptr(stream);
    for slot in 0..k {
        // gate = w_gate[topk[slot]] @ h_norm
        dispatch_indexed_matmul_m1(
            kernels,
            stream,
            moe.gate_exp_kind,
            g_ptrs_p,
            topk_idx_p,
            slot,
            h_norm_p,
            expert_gate_p,
            ef,
            d,
            x_q8_p,
        )?;
        // up = w_up[topk[slot]] @ h_norm
        dispatch_indexed_matmul_m1(
            kernels,
            stream,
            moe.up_exp_kind,
            u_ptrs_p,
            topk_idx_p,
            slot,
            h_norm_p,
            expert_up_p,
            ef,
            d,
            x_q8_p,
        )?;
        // gate = silu(gate) * up
        unsafe {
            kernels
                .swiglu_bf16(stream, expert_gate_p, expert_up_p, expert_gate_p, ef)
                .map_err(|e| LlmError::Backend(format!("swiglu expert slot={slot}: {e:?}")))?;
        }
        // down_out = w_down[topk[slot]] @ gate
        dispatch_indexed_matmul_m1(
            kernels,
            stream,
            moe.down_exp_kind,
            d_ptrs_p,
            topk_idx_p,
            slot,
            expert_gate_p,
            expert_out_p,
            d,
            ef,
            x_q8_p,
        )?;
        // h_p += topk_w[slot] * down_out (alpha read from device)
        unsafe {
            kernels
                .scaled_add_inplace_bf16_devscalar(stream, h_p, expert_out_p, topk_w_p, slot, d)
                .map_err(|e| {
                    LlmError::Backend(format!("scaled_add devscalar slot={slot}: {e:?}"))
                })?;
        }
    }

    // ---- 5. Shared expert (parallel path) ----
    // shexp_dot = gate_inp_shexp · h_norm  (BF16 scalar) ; sigmoid fused
    // into the final scaled_add to keep the sigmoid value in float precision
    // (matching the sync path's host-side `1/(1+exp(-v))` computation
    // precision exactly — no intermediate bf16 rounding).
    unsafe {
        let (gip_p, _g) = moe.gate_inp_shexp.device_ptr(stream);
        kernels
            .sgemv_bf16_bf16(stream, gip_p, h_norm_p, shexp_dot_p, 1, d)
            .map_err(|e| LlmError::Backend(format!("shexp dot: {e:?}")))?;
    }

    moe.gate_shexp
        .dispatch_matmul_m1(kernels, stream, h_norm_p, expert_gate_p, x_q8_p)?;
    moe.up_shexp
        .dispatch_matmul_m1(kernels, stream, h_norm_p, expert_up_p, x_q8_p)?;
    unsafe {
        kernels
            .swiglu_bf16(stream, expert_gate_p, expert_up_p, expert_gate_p, ef)
            .map_err(|e| LlmError::Backend(format!("swiglu shexp: {e:?}")))?;
    }
    moe.down_shexp
        .dispatch_matmul_m1(kernels, stream, expert_gate_p, expert_out_p, x_q8_p)?;
    // h_p += sigmoid(dot) * down_out — fused kernel keeps sigmoid in float.
    unsafe {
        kernels
            .scaled_add_sigmoid_devscalar_bf16(stream, h_p, expert_out_p, shexp_dot_p, d)
            .map_err(|e| LlmError::Backend(format!("scaled_add_sigmoid shexp: {e:?}")))?;
    }

    Ok(())
}

/// T246.8 A4 — MoE FFN forward using the mul_mm_id mega-kernels.
///
/// Drop-in replacement for `moe_ffn_forward_step_async` when
/// `RUSTORCH_MOE_MEGA=1`. The K=8 per-expert SGEMV launches of
/// gate/up/down and their scaled_add epilogue are replaced by :
///   - mul_mm_id gate           : 1 launch, computes `[k_used, ef]`
///   - mul_mm_id up             : 1 launch, computes `[k_used, ef]`
///   - swiglu over k_used*ef    : 1 launch (already elementwise)
///   - per-slot down            : K iterations of indexed sgemv (each
///                                slot has its own gate output → cannot
///                                yet be collapsed without an x-stride
///                                variant of mul_mm_id)
///   - scaled_add_routed        : 1 launch, h += Σ topk_w[s]*down[s,:]
///
/// Net launch reduction per layer (Qwen3.6-A3B, k=8) :
///   before : 8 gate + 8 up + 8 swiglu + 8 down + 8 scaled_add = 40
///   after  : 1 + 1 + 1 + 8 + 1 = 12
/// Saved : ~28 launches × 64 MoE layers ≈ 1800 launches/token.
///
/// NOTE on parity : the per-slot SGEMV body inside the mega-kernel is
/// byte-for-byte identical to the indexed kernel (A3 lesson) so gate/up
/// outputs are BIT-EXACT vs the loop. The fused `scaled_add_routed_bf16`
/// reduces in fp32 then down-casts once — vs the per-slot path which
/// down-casts BF16 between each step → ~1 BF16 ULP drift per element,
/// equivalent to A2's drift-from-token-48+ behaviour.
#[allow(clippy::too_many_arguments)]
fn moe_ffn_forward_step_mega(
    moe: &MoeFfnQ4K,
    kernels: &LlmKernels,
    stream: &Arc<CudaStream>,
    cfg: &Qwen35Config,
    h_norm_p: u64,
    h_p: u64,
    x_q8_p: u64,
    router_logits_p: u64,
    topk_idx_p: u64,
    topk_w_p: u64,
    expert_gate_p: u64, // [k_used, ef]
    expert_up_p: u64,   // [k_used, ef]
    expert_out_p: u64,  // [k_used, d]
    shexp_dot_p: u64,
) -> Result<(), LlmError> {
    use cudarc::driver::DevicePtr;
    let d = cfg.d as i32;
    let ef = cfg.expert_f as i32;
    let n_e = cfg.n_experts as i32;
    let k = cfg.n_experts_used as i32;

    // ---- 1. Router logits ----
    moe.gate_inp
        .dispatch_matmul_m1(kernels, stream, h_norm_p, router_logits_p, x_q8_p)?;

    // ---- 2. Top-K softmax → indices + renormalized weights on device ----
    unsafe {
        kernels
            .topk_softmax_bf16(stream, router_logits_p, topk_idx_p, topk_w_p, n_e, k)
            .map_err(|e| LlmError::Backend(format!("topk_softmax mega: {e:?}")))?;
    }

    // ---- 3. Zero h_p ----
    unsafe {
        kernels
            .zero_bf16(stream, h_p, d)
            .map_err(|e| LlmError::Backend(format!("zero h_p mega: {e:?}")))?;
    }

    // ---- 4. Mega gate / up / swiglu / per-slot down / routed-add ----
    let (g_ptrs_p, _gg) = moe.gate_exp_ptrs_dev.device_ptr(stream);
    let (u_ptrs_p, _gu) = moe.up_exp_ptrs_dev.device_ptr(stream);
    let (d_ptrs_p, _gd) = moe.down_exp_ptrs_dev.device_ptr(stream);

    // gate[k_used, ef] = w_gate[topk[s]] @ h_norm  (single launch)
    dispatch_indexed_mega(
        kernels,
        stream,
        moe.gate_exp_kind,
        g_ptrs_p,
        topk_idx_p,
        h_norm_p,
        expert_gate_p,
        ef,
        d,
        k,
        x_q8_p,
    )?;
    // up[k_used, ef]   = w_up[topk[s]]   @ h_norm  (single launch)
    dispatch_indexed_mega(
        kernels,
        stream,
        moe.up_exp_kind,
        u_ptrs_p,
        topk_idx_p,
        h_norm_p,
        expert_up_p,
        ef,
        d,
        k,
        x_q8_p,
    )?;
    // swiglu over the entire k_used * ef element block (elementwise — same
    // kernel as the per-slot variant just with a larger n).
    unsafe {
        kernels
            .swiglu_bf16(stream, expert_gate_p, expert_up_p, expert_gate_p, ef * k)
            .map_err(|e| LlmError::Backend(format!("swiglu mega: {e:?}")))?;
    }
    // Per-slot down : each slot has its own gate output ; the current
    // `mul_mm_id_*` kernels take a single `x[K]` shared across all slots
    // so they can't be used here without a stride-x variant. Even with
    // the per-slot loop, gate/up/swiglu have collapsed (24 → 3 launches)
    // and the scaled_add routed-reduce will collapse 8 → 1.
    let elem_size = std::mem::size_of::<half::bf16>() as u64;
    for slot in 0..k {
        let x_slot_p = expert_gate_p + (slot as u64) * (ef as u64) * elem_size;
        let y_slot_p = expert_out_p + (slot as u64) * (d as u64) * elem_size;
        dispatch_indexed_matmul_m1(
            kernels,
            stream,
            moe.down_exp_kind,
            d_ptrs_p,
            topk_idx_p,
            slot,
            x_slot_p,
            y_slot_p,
            d,
            ef,
            x_q8_p,
        )?;
    }
    // h_p += Σ_{s} topk_w[s] * down[s, :]
    //
    // We have two epilogue options :
    //   a) `scaled_add_routed_bf16` (1 launch, sums all K in fp32 then
    //      down-casts once) — drifts ~1 ULP per element vs the per-step
    //      down-cast path, which compounds across 64 layers and produces
    //      different greedy tokens from token 1 onwards on Qwen3.6-A3B
    //      (real-world : ~12% perf win, but parity gate fails).
    //   b) K iterations of `scaled_add_inplace_bf16_devscalar` (legacy
    //      A2 path) — bit-exact with MEGA=0 epilogue but adds K=8 launches.
    //
    // We default to (b) for parity ; flip RUSTORCH_MOE_MEGA_ROUTED=1 to
    // opt into (a) for the extra ~3-4% win at the cost of token drift.
    let mega_routed = std::env::var("RUSTORCH_MOE_MEGA_ROUTED")
        .ok()
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    if mega_routed {
        unsafe {
            kernels
                .scaled_add_routed_bf16(stream, h_p, expert_out_p, topk_w_p, d, k)
                .map_err(|e| LlmError::Backend(format!("scaled_add_routed: {e:?}")))?;
        }
    } else {
        for slot in 0..k {
            let y_slot_p = expert_out_p + (slot as u64) * (d as u64) * elem_size;
            unsafe {
                kernels
                    .scaled_add_inplace_bf16_devscalar(stream, h_p, y_slot_p, topk_w_p, slot, d)
                    .map_err(|e| LlmError::Backend(format!("scaled_add slot={slot}: {e:?}")))?;
            }
        }
    }

    // ---- 5. Shared expert (parallel path, identical to async variant) ----
    unsafe {
        let (gip_p, _g) = moe.gate_inp_shexp.device_ptr(stream);
        kernels
            .sgemv_bf16_bf16(stream, gip_p, h_norm_p, shexp_dot_p, 1, d)
            .map_err(|e| LlmError::Backend(format!("shexp dot mega: {e:?}")))?;
    }
    moe.gate_shexp
        .dispatch_matmul_m1(kernels, stream, h_norm_p, expert_gate_p, x_q8_p)?;
    moe.up_shexp
        .dispatch_matmul_m1(kernels, stream, h_norm_p, expert_up_p, x_q8_p)?;
    unsafe {
        kernels
            .swiglu_bf16(stream, expert_gate_p, expert_up_p, expert_gate_p, ef)
            .map_err(|e| LlmError::Backend(format!("swiglu shexp mega: {e:?}")))?;
    }
    moe.down_shexp
        .dispatch_matmul_m1(kernels, stream, expert_gate_p, expert_out_p, x_q8_p)?;
    unsafe {
        kernels
            .scaled_add_sigmoid_devscalar_bf16(stream, h_p, expert_out_p, shexp_dot_p, d)
            .map_err(|e| LlmError::Backend(format!("scaled_add_sigmoid shexp mega: {e:?}")))?;
    }

    Ok(())
}

/// T246.10 TrackE.2 — batched MoE Group-GEMM forward for M tokens.
///
/// Replaces the per-row `moe_ffn_forward_step_mega` loop when
/// `RUSTORCH_MOE_GROUP_GEMM=1` AND `m_tokens >= GROUP_GEMM_MIN_M`. The
/// routed-experts gate/up/down get collapsed into 3 Group-GEMM launches
/// (1 per projection, regardless of M or k_used), and the routed-reduce
/// epilogue is a per-token loop of `scaled_add_routed_bf16`.
///
/// T246.10 TrackE.3 / TrackE.4 — when `use_sorted=true`
/// (env `RUSTORCH_MOE_GROUP_GEMM_SORTED=1`) AND the gate / up expert
/// weights are a SORTED-supported quant (Q4_K or Q5_K), build a
/// device-side compact-by-expert permutation via `mm_ids_helper_bf16` and
/// dispatch the gate / up matmuls through the matching sorted kernel
/// (`mul_mm_id_gemm_q4_k_sorted_bf16` / `mul_mm_id_gemm_q5_k_sorted_bf16`).
/// Adjacent compact slots share the same expert → L1/L2 weight tile reuse.
/// Output layout is identical to the unsorted path so the rest of the
/// pipeline (swiglu, scaled_add, down-proj, shared expert) is unchanged.
///
/// The shared expert keeps the per-token loop : it's a Dense FFN (one set
/// of weights applied uniformly), so going batched there is straightforward
/// follow-up work but not the hot path TrackE.2 targets.
///
/// Layout :
///   `h_norm_m_p`         : [M, d]              BF16 (input, post-norm)
///   `h_m_p`              : [M, d]              BF16 (output, accumulator)
///   `router_logits_m_p`  : [M, n_experts]      BF16 scratch
///   `topk_idx_m_p`       : [M, k_used]         i32 scratch
///   `topk_w_m_p`         : [M, k_used]         BF16 scratch
///   `expert_gate_m_p`    : [M, k_used, ef]     BF16 scratch
///   `expert_up_m_p`      : [M, k_used, ef]     BF16 scratch
///   `expert_out_m_p`     : [M, k_used, d]      BF16 scratch
///   `x_q8_m_p`           : [M, (max(d,ef)/32)*36] u8 scratch (dp4a, 0 to skip)
///   `shexp_dot_p`        : [1] BF16 (single-token shared expert, reused per row)
///   `expert_gate_p`      : [ef] BF16 (single-token shared expert scratch)
///   `expert_up_p`        : [ef] BF16 (single-token shared expert scratch)
///   `expert_out_p`       : [d]  BF16 (single-token shared expert scratch)
///   `ids_src1_m_p`       : [M*k_used] i32 scratch (TrackE.3, 0 to skip sort)
///   `ids_dst_m_p`        : [M*k_used] i32 scratch (TrackE.3)
///   `expert_bounds_p`    : [n_experts+1] i32 scratch (TrackE.3)
///   `use_sorted`         : TrackE.3 gate, requires Q4_K-BF16 gate/up
///   `x_q8_mmq_m_p`       : [M, (max(d,ef)/128)*144] u8 scratch (MMQ-WHOLESALE, 0 to skip)
#[allow(clippy::too_many_arguments)]
fn moe_ffn_forward_step_group_gemm(
    moe: &MoeFfnQ4K,
    kernels: &LlmKernels,
    stream: &Arc<CudaStream>,
    cfg: &Qwen35Config,
    m_tokens: i32,
    h_norm_m_p: u64,
    h_m_p: u64,
    router_logits_m_p: u64,
    topk_idx_m_p: u64,
    topk_w_m_p: u64,
    expert_gate_m_p: u64,
    expert_up_m_p: u64,
    expert_out_m_p: u64,
    x_q8_m_p: u64,
    // Single-token scratch for the shared expert per-row loop :
    x_q8_p: u64,
    shexp_dot_p: u64,
    expert_gate_p: u64,
    expert_up_p: u64,
    expert_out_p: u64,
    // TrackE.3 — sort-permutation scratch (0 / false to skip).
    ids_src1_m_p: u64,
    ids_dst_m_p: u64,
    expert_bounds_p: u64,
    use_sorted: bool,
    // MMQ-WHOLESALE.6a — packed Q8_1 staging for the shared-expert MMQ path
    // (0 to skip, falls back to per-token loop).
    x_q8_mmq_m_p: u64,
) -> Result<(), LlmError> {
    use cudarc::driver::DevicePtr;
    let d = cfg.d as i32;
    let ef = cfg.expert_f as i32;
    let n_e = cfg.n_experts as i32;
    let k = cfg.n_experts_used as i32;
    let bf16_sz = std::mem::size_of::<half::bf16>() as u64;

    // ---- 1. Router logits + topk_softmax (per-token loop) ----
    //
    // We keep the router as a per-token M=1 dispatch (not batched mvar) to
    // preserve byte-for-byte parity with the decode_step path (which also
    // calls dispatch_matmul_m1). Switching to dispatch_matmul_mvar here
    // would route through a different kernel (sgemm_bf16_bf16_mvar) and
    // produce numerically different — but kernel-level bit-exact — outputs.
    // The per-token loop is cheap : `gate_inp` is `[n_experts, d]` and only
    // M small sgemv launches. The MoE matmul batching (gate/up/down) is
    // where the launch reduction matters.
    for m in 0..m_tokens {
        let hn_row_p = h_norm_m_p + (m as u64) * (d as u64) * bf16_sz;
        let rl_row_p = router_logits_m_p + (m as u64) * (n_e as u64) * bf16_sz;
        moe.gate_inp
            .dispatch_matmul_m1(kernels, stream, hn_row_p, rl_row_p, x_q8_p)?;
        let idx_p = topk_idx_m_p + (m as u64) * (k as u64) * (std::mem::size_of::<i32>() as u64);
        let w_p = topk_w_m_p + (m as u64) * (k as u64) * bf16_sz;
        unsafe {
            kernels
                .topk_softmax_bf16(stream, rl_row_p, idx_p, w_p, n_e, k)
                .map_err(|e| LlmError::Backend(format!("topk_softmax m={m}: {e:?}")))?;
        }
    }

    // ---- 3. Zero h_m (one launch over M*d) ----
    unsafe {
        kernels
            .zero_bf16(stream, h_m_p, m_tokens * d)
            .map_err(|e| LlmError::Backend(format!("zero h_m group: {e:?}")))?;
    }

    // ---- 3b. T246.10 TrackE.3 / TrackE.4 — build sort permutation (once per layer) ----
    //
    // Only when `use_sorted=true` AND the gate/up expert kind is one of the
    // sorted-supported quants (Q4_K via TrackE.3, Q5_K via TrackE.4). The
    // sorted kernels use BF16 input directly — when active we BYPASS the
    // dp4a path on the gate / up matmuls (A3 showed dp4a vs BF16 is a wash
    // on this shape ; the win comes from cache reuse, not compute
    // throughput). The down-projection always uses the unsorted dispatch
    // (its input layout `[M*k_used, ef]` is already permuted, not
    // `[token, k_used]`).
    let is_sorted_kind =
        |k: ExpertQuantKind| -> bool { matches!(k, ExpertQuantKind::Q4K | ExpertQuantKind::Q5K) };
    let sort_active = use_sorted
        && ids_src1_m_p != 0
        && ids_dst_m_p != 0
        && expert_bounds_p != 0
        && is_sorted_kind(moe.gate_exp_kind)
        && is_sorted_kind(moe.up_exp_kind);
    if sort_active {
        unsafe {
            kernels
                .mm_ids_helper_bf16(
                    stream,
                    topk_idx_m_p,
                    ids_src1_m_p,
                    ids_dst_m_p,
                    expert_bounds_p,
                    m_tokens,
                    k,
                    n_e,
                )
                .map_err(|e| LlmError::Backend(format!("mm_ids_helper group: {e:?}")))?;
        }
    }

    // ---- 4. Gate (batched Group-GEMM call covering ALL M*k slots) ----
    //
    // Per-block parity test PASSES bit-exact at Qwen3.6-A3B shape, but the
    // integrated path may produce 1 BF16 ULP drift vs the per-row mega
    // baseline (task spec accepts 1 ULP). Bench reality-checks the win :
    // if the parity drift is significant enough to break decode quality
    // downstream, we can fall back to the per-token loop above.
    let (g_ptrs_p, _gg) = moe.gate_exp_ptrs_dev.device_ptr(stream);
    let (u_ptrs_p, _gu) = moe.up_exp_ptrs_dev.device_ptr(stream);
    let (d_ptrs_p, _gd) = moe.down_exp_ptrs_dev.device_ptr(stream);

    if sort_active {
        dispatch_sorted_group_gemm(
            kernels,
            stream,
            moe.gate_exp_kind,
            g_ptrs_p,
            topk_idx_m_p,
            ids_src1_m_p,
            ids_dst_m_p,
            h_norm_m_p,
            expert_gate_m_p,
            m_tokens,
            ef,
            d,
            k,
            "gate",
        )?;
    } else {
        dispatch_indexed_group_gemm(
            kernels,
            stream,
            moe.gate_exp_kind,
            g_ptrs_p,
            topk_idx_m_p,
            h_norm_m_p,
            expert_gate_m_p,
            m_tokens,
            ef,
            d,
            k,
            x_q8_m_p,
        )?;
    }

    // ---- 5. Up (batched Group-GEMM call) ----
    if sort_active {
        dispatch_sorted_group_gemm(
            kernels,
            stream,
            moe.up_exp_kind,
            u_ptrs_p,
            topk_idx_m_p,
            ids_src1_m_p,
            ids_dst_m_p,
            h_norm_m_p,
            expert_up_m_p,
            m_tokens,
            ef,
            d,
            k,
            "up",
        )?;
    } else {
        dispatch_indexed_group_gemm(
            kernels,
            stream,
            moe.up_exp_kind,
            u_ptrs_p,
            topk_idx_m_p,
            h_norm_m_p,
            expert_up_m_p,
            m_tokens,
            ef,
            d,
            k,
            x_q8_m_p,
        )?;
    }

    // ---- 6. SwiGLU over M * k_used * ef in one shot ----
    unsafe {
        kernels
            .swiglu_bf16(
                stream,
                expert_gate_m_p,
                expert_up_m_p,
                expert_gate_m_p,
                m_tokens * k * ef,
            )
            .map_err(|e| LlmError::Backend(format!("swiglu group: {e:?}")))?;
    }

    // ---- 7. Down (single Group-GEMM launch, M' = M*k_used flat, k_used' = 1) ----
    //
    // The down's input is [M, k_used, ef] (= flat [M*k_used, ef] BF16) and
    // output is [M, k_used, d] (= flat [M*k_used, d]). Flatten (m, slot) as
    // m' and let kernel read topk_indices[m' * 1 + 0] = topk_idx_m_p[m']
    // which is topk[m, slot] — exactly the right expert pick.
    //
    // T246.10 TrackE.4 — when `use_sorted=true` AND the down expert kind is
    // a sorted-supported quant (Q4_K or Q5_K), also build a compact-by-expert
    // permutation for the flat `[M*k_used, 1]` topk view and dispatch the
    // matching sorted kernel. The down projection is the Q5_K bottleneck on
    // Qwen3.6-A3B Q4_K_M (37 of 40 layers ship `down_exps` as Q5_K), so this
    // is where the bulk of TrackE.4's wall-clock win actually lands.
    let down_sort_active = use_sorted
        && ids_src1_m_p != 0
        && ids_dst_m_p != 0
        && expert_bounds_p != 0
        && is_sorted_kind(moe.down_exp_kind);
    if down_sort_active {
        // The down's effective topk view is [M*k_used, k_used'=1], so rebuild
        // the permutation with `n_tokens = M*k`, `k_used = 1`.
        let m_flat = m_tokens * k;
        unsafe {
            kernels
                .mm_ids_helper_bf16(
                    stream,
                    topk_idx_m_p,
                    ids_src1_m_p,
                    ids_dst_m_p,
                    expert_bounds_p,
                    m_flat,
                    1,
                    n_e,
                )
                .map_err(|e| LlmError::Backend(format!("mm_ids_helper down: {e:?}")))?;
        }
        dispatch_sorted_group_gemm(
            kernels,
            stream,
            moe.down_exp_kind,
            d_ptrs_p,
            topk_idx_m_p,
            ids_src1_m_p,
            ids_dst_m_p,
            expert_gate_m_p,
            expert_out_m_p,
            m_flat, // M' = M * k_used
            d,
            ef,
            1, // k_used' = 1
            "down",
        )?;
    } else {
        dispatch_indexed_group_gemm(
            kernels,
            stream,
            moe.down_exp_kind,
            d_ptrs_p,
            topk_idx_m_p,
            expert_gate_m_p,
            expert_out_m_p,
            m_tokens * k, // M' = M * k_used
            d,
            ef,
            1, // k_used' = 1
            x_q8_m_p,
        )?;
    }

    // ---- 8. Routed scaled-add epilogue (per-token, per-slot loop) ----
    //
    // CRITICAL : we use the per-slot `scaled_add_inplace_bf16_devscalar`
    // loop (M * k_used launches), NOT the fused `scaled_add_routed_bf16`
    // (M launches). The fused variant accumulates in fp32 then down-casts
    // once per element, while the per-slot loop down-casts to BF16 between
    // each slot — the 1-ULP-per-element drift compounds across 64 layers
    // and produces token-level divergence (see comment in mega kernel,
    // RUSTORCH_MOE_MEGA_ROUTED behavior). For bit-exact parity vs the
    // per-row mega path the per-slot loop is required.
    //
    // Cost : M * k_used = 8 * 8 = 64 launches/layer for N=8 prefill —
    // still ~50× fewer than the per-token mega-loop (4096 launches/layer).
    for m in 0..m_tokens {
        let h_row_p = h_m_p + (m as u64) * (d as u64) * bf16_sz;
        let tw_row_p = topk_w_m_p + (m as u64) * (k as u64) * bf16_sz;
        for slot in 0..k {
            let eo_slot_p = expert_out_m_p
                + (m as u64) * (k as u64) * (d as u64) * bf16_sz
                + (slot as u64) * (d as u64) * bf16_sz;
            unsafe {
                kernels
                    .scaled_add_inplace_bf16_devscalar(
                        stream, h_row_p, eo_slot_p, tw_row_p, slot, d,
                    )
                    .map_err(|e| {
                        LlmError::Backend(format!("scaled_add_inplace m={m} s={slot}: {e:?}"))
                    })?;
            }
        }
    }

    // ---- 9. Shared expert (Dense FFN) per-token loop ----
    //
    // The shared expert applies one fixed set of weights uniformly across
    // all M tokens. The hottest matmuls (gate_shexp, up_shexp, down_shexp)
    // are Dense — they can be batched via `dispatch_matmul_mvar` (sgemm-mvar).
    // The two NON-batchable parts are :
    //   - `gate_inp_shexp` : [1, d] → scalar per token, sized M (cheap).
    //   - `scaled_add_sigmoid_devscalar_bf16` epilogue : per-token devscalar.
    //
    // We keep the existing per-token sequence but call dispatch_matmul_mvar
    // for the gate/up/down where it helps.
    //
    // To avoid an additional [M, ef] / [M, d] scratch allocation just for
    // the shared expert (which would double our memory cost), we run the
    // shared expert per-token using the single-token scratch buffers
    // (`expert_gate_p`, `expert_up_p`, `expert_out_p`). The cost is M
    // launches × (gate + up + swiglu + down + scaled_add) = 5M launches.
    // For M=512 that's 2560 launches — vs the 4 launches gain on routed
    // experts being 16320 saved (from 16384 to 64). Net : huge win.

    // ── T246.10 MMQ-WHOLESALE.6a — batched shared-expert MMQ fast-path ──
    //
    // When `RUSTORCH_MMQ_WHOLESALE=1` AND `m_tokens >= MMQ_WHOLESALE_MIN_M`
    // AND the shared-expert weights are all Q4_K (the only kind the MMQ
    // kernel supports), replace the 3 per-token Q4_K SGEMVs (gate/up/down)
    // with a single quantize + 3 batched `mul_mat_q4_k_q8_1_mma` launches.
    //
    // Buffer reuse strategy (no extra allocation needed for the BF16 sides
    // because the routed-expert scratch is FREE after step 8 closes) :
    //   - `expert_gate_m_p[0 : M*ef]` → batched gate output [M, ef]
    //   - `expert_up_m_p[0 : M*ef]`   → batched up output  [M, ef]
    //   - `expert_out_m_p[0 : M*d]`   → batched down output [M, d]
    //   - `x_q8_mmq_m_p[0 : M*(K/128)*144]` → packed Q8_1 of h_norm_m
    //                                         (overwritten between gate/up
    //                                         and down because K differs).
    //
    // The two per-token small kernels (sigmoid `sgemv_bf16_bf16` and the
    // `scaled_add_sigmoid_devscalar_bf16` epilogue) stay in their own loop ;
    // they're cheap (M scalar reads). The big matmuls dominate cost.
    let mmq_shexp_active = mmq_wholesale_enabled()
        && m_tokens >= MMQ_WHOLESALE_MIN_M
        && x_q8_mmq_m_p != 0
        && matches!(&moe.gate_shexp, QuantTensor::Q4K { .. })
        && matches!(&moe.up_shexp, QuantTensor::Q4K { .. })
        && matches!(&moe.down_shexp, QuantTensor::Q4K { .. })
        && (d as usize) % 256 == 0
        && (ef as usize) % 256 == 0;

    if mmq_shexp_active {
        // Step 9a — batched gate matmul : [M, d] × W_gate [ef, d] → [M, ef].
        // 1. Quantize h_norm_m_p (BF16) → x_q8_mmq_m_p (packed Q8_1).
        // 2. mul_mat_q4_k_q8_1_mma over [M, ef] with K=d.
        unsafe {
            kernels
                .quantize_mmq_q8_1_bf16_ds4(stream, h_norm_m_p, x_q8_mmq_m_p, m_tokens, d)
                .map_err(|e| LlmError::Backend(format!("MMQ quantize shexp gate K={d}: {e:?}")))?;
        }
        if let QuantTensor::Q4K { bytes, .. } = &moe.gate_shexp {
            use cudarc::driver::DevicePtr;
            let (w_p, _g) = bytes.device_ptr(stream);
            unsafe {
                kernels
                    .mul_mat_q4_k_q8_1_mma(
                        stream,
                        w_p,
                        x_q8_mmq_m_p,
                        expert_gate_m_p,
                        m_tokens,
                        ef,
                        d,
                    )
                    .map_err(|e| LlmError::Backend(format!("MMQ shexp gate: {e:?}")))?;
            }
        }
        // Step 9c — batched up matmul (reuse the same Q8_1 staging — same K=d).
        if let QuantTensor::Q4K { bytes, .. } = &moe.up_shexp {
            use cudarc::driver::DevicePtr;
            let (w_p, _g) = bytes.device_ptr(stream);
            unsafe {
                kernels
                    .mul_mat_q4_k_q8_1_mma(
                        stream,
                        w_p,
                        x_q8_mmq_m_p,
                        expert_up_m_p,
                        m_tokens,
                        ef,
                        d,
                    )
                    .map_err(|e| LlmError::Backend(format!("MMQ shexp up: {e:?}")))?;
            }
        }
        // Step 9d — batched SwiGLU over [M, ef] in-place into expert_gate_m_p.
        unsafe {
            kernels
                .swiglu_bf16(
                    stream,
                    expert_gate_m_p,
                    expert_up_m_p,
                    expert_gate_m_p,
                    m_tokens * ef,
                )
                .map_err(|e| LlmError::Backend(format!("MMQ shexp swiglu: {e:?}")))?;
        }
        // Step 9e — batched down matmul : [M, ef] × W_down [d, ef] → [M, d].
        // Re-quantize the SwiGLU output (K=ef this time).
        unsafe {
            kernels
                .quantize_mmq_q8_1_bf16_ds4(stream, expert_gate_m_p, x_q8_mmq_m_p, m_tokens, ef)
                .map_err(|e| LlmError::Backend(format!("MMQ quantize shexp down K={ef}: {e:?}")))?;
        }
        if let QuantTensor::Q4K { bytes, .. } = &moe.down_shexp {
            use cudarc::driver::DevicePtr;
            let (w_p, _g) = bytes.device_ptr(stream);
            unsafe {
                kernels
                    .mul_mat_q4_k_q8_1_mma(
                        stream,
                        w_p,
                        x_q8_mmq_m_p,
                        expert_out_m_p,
                        m_tokens,
                        d,
                        ef,
                    )
                    .map_err(|e| LlmError::Backend(format!("MMQ shexp down: {e:?}")))?;
            }
        }
        // Step 9f — per-token sigmoid gate + sigmoid-scaled add (small kernels,
        // kept per-row to preserve bit-exact parity with the baseline epilogue).
        for m in 0..m_tokens {
            let h_norm_row_p = h_norm_m_p + (m as u64) * (d as u64) * bf16_sz;
            let h_row_p = h_m_p + (m as u64) * (d as u64) * bf16_sz;
            let eo_row_p = expert_out_m_p + (m as u64) * (d as u64) * bf16_sz;
            unsafe {
                let (gip_p, _g) = moe.gate_inp_shexp.device_ptr(stream);
                kernels
                    .sgemv_bf16_bf16(stream, gip_p, h_norm_row_p, shexp_dot_p, 1, d)
                    .map_err(|e| LlmError::Backend(format!("MMQ shexp dot m={m}: {e:?}")))?;
                kernels
                    .scaled_add_sigmoid_devscalar_bf16(stream, h_row_p, eo_row_p, shexp_dot_p, d)
                    .map_err(|e| LlmError::Backend(format!("MMQ shexp sigadd m={m}: {e:?}")))?;
            }
        }
    } else {
        // Baseline per-token shared-expert loop (unchanged).
        for m in 0..m_tokens {
            let h_norm_row_p = h_norm_m_p + (m as u64) * (d as u64) * bf16_sz;
            let h_row_p = h_m_p + (m as u64) * (d as u64) * bf16_sz;
            unsafe {
                let (gip_p, _g) = moe.gate_inp_shexp.device_ptr(stream);
                kernels
                    .sgemv_bf16_bf16(stream, gip_p, h_norm_row_p, shexp_dot_p, 1, d)
                    .map_err(|e| LlmError::Backend(format!("shexp dot group m={m}: {e:?}")))?;
            }
            moe.gate_shexp.dispatch_matmul_m1(
                kernels,
                stream,
                h_norm_row_p,
                expert_gate_p,
                x_q8_p,
            )?;
            moe.up_shexp
                .dispatch_matmul_m1(kernels, stream, h_norm_row_p, expert_up_p, x_q8_p)?;
            unsafe {
                kernels
                    .swiglu_bf16(stream, expert_gate_p, expert_up_p, expert_gate_p, ef)
                    .map_err(|e| LlmError::Backend(format!("swiglu shexp group m={m}: {e:?}")))?;
            }
            moe.down_shexp.dispatch_matmul_m1(
                kernels,
                stream,
                expert_gate_p,
                expert_out_p,
                x_q8_p,
            )?;
            unsafe {
                kernels
                    .scaled_add_sigmoid_devscalar_bf16(
                        stream,
                        h_row_p,
                        expert_out_p,
                        shexp_dot_p,
                        d,
                    )
                    .map_err(|e| {
                        LlmError::Backend(format!("scaled_add_sigmoid shexp group m={m}: {e:?}"))
                    })?;
            }
        }
    }

    Ok(())
}
