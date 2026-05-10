//! `qwen35_cpu` — CPU reference forward for Qwen3.5/3.6 hybrid models.
//!
//! This module is the **correctness reference** for the hybrid SSM +
//! Attention architecture used by `Qwen3.6-27B` (`qwen35`, dense FFN) and
//! `Qwen3.6-35B-A3B` (`qwen35moe`, MoE FFN). Everything here runs in
//! plain f32 on a single CPU thread — slow (~0.1-0.5 tok/s on M4 Max
//! depending on model), but architecturally complete and intentionally
//! simple so it can be diffed against the llama.cpp reference.
//!
//! The Metal-accelerated path lives in subsequent commits (T144+).
//!
//! ## Building blocks
//!
//! - [`Qwen35Weights`] — full-weight in-memory representation. Loaded by
//!   [`load_weights`] which dequantises every GGUF tensor (Q4_K / Q5_K /
//!   Q6_K / Q8_0 / F32) to f32 and partitions per-layer slots according
//!   to the layer kind (Attention vs SSM).
//! - [`Qwen35State`] — per-token persistent state. Holds the
//!   attention KV cache (one entry per attention layer) and the SSM
//!   recurrent state + 1-D conv ring buffer (one entry per SSM layer).
//! - [`forward_token`] — single-token forward step. Delegates to
//!   [`ssm_block_forward`] or [`attn_block_forward`] based on
//!   [`Qwen35Config::layer_kind`], then runs the (dense or MoE) FFN
//!   sub-block with a residual connection.
//!
//! ## Numerical conventions
//!
//! - Weight matrices are stored row-major as `[out_dim * in_dim]` f32.
//!   `gemv(W, x, y)` computes `y[i] = Σ_k W[i*K + k] * x[k]`.
//! - GGUF stores 2-D weights as `shape = [in, out]`, but our row-major
//!   representation transposes to `[out, in]` so the matmul-vec is the
//!   natural `for i in 0..N: y[i] = Σ x[k] * W[i, k]`.
//! - `silu(x) = x * sigmoid(x) = x / (1 + exp(-x))`.
//! - `rms_norm(x, gamma, eps)` returns `x_i / sqrt(mean(x²) + eps) * gamma_i`.
//! - `l2_norm(x, eps)` returns `x_i / sqrt(sum(x²) + eps)` (per-vector L2).
//!
//! ## Hybrid-block math (cross-check against llama.cpp)
//!
//! ### Gated Delta Net (the "SSM" block)
//!
//! The block is **not** classic Mamba-2 — it's a recurrent linear
//! attention with a delta-rule update (DeltaNet). Per token:
//!
//! ```text
//!   qkv_mixed   = wqkv  @ x                  (d → 2*key_dim + value_dim)
//!   z           = wqkv_gate @ x              (d → value_dim)
//!   beta        = sigmoid(ssm_beta @ x)      (d → num_v_heads)
//!   alpha       = ssm_alpha @ x              (d → num_v_heads)
//!   alpha_bias  = alpha + ssm_dt.bias
//!   alpha_sp    = softplus(alpha_bias)
//!   gate_h      = alpha_sp * ssm_a           (per-head gate, num_v_heads)
//!
//!   ; depth-wise causal 1-D convolution on the qkv stream, with running
//!   ; ring buffer of size (kernel-1) carried across tokens
//!   conv_input  = concat(conv_state, qkv_mixed)        ; [kernel, conv_dim]
//!   conv_out_c  = sum_k conv_kernel[k, c] * conv_input[k, c]   for each c
//!   conv_out    = silu(conv_out_c)
//!   conv_state := conv_input[1:]                                ; ring shift
//!
//!   ; split the post-conv stream into Q, K, V (per group / head)
//!   q_conv      = conv_out[..key_dim]            view as [head_k_dim, num_k_heads]
//!   k_conv      = conv_out[key_dim..2*key_dim]   view as [head_k_dim, num_k_heads]
//!   v_conv      = conv_out[2*key_dim..]          view as [head_v_dim, num_v_heads]
//!   q_conv      = l2_norm(q_conv, eps)           (per-head)
//!   k_conv      = l2_norm(k_conv, eps)           (per-head)
//!
//!   ; if num_k_heads != num_v_heads, broadcast Q/K to V's head count
//!   q_conv'     = repeat_to(num_v_heads, q_conv)
//!   k_conv'     = repeat_to(num_v_heads, k_conv)
//!
//!   ; delta-net recurrence — one outer-product update per head h
//!   for h in 0..num_v_heads:
//!     state[h]  = exp(gate_h[h]) * state[h] + beta[h] * outer(v_conv[h], k_conv'[h])
//!     out[h]    = state[h] @ q_conv'[h]                ; [head_v_dim]
//!
//!   ; gated norm + output projection
//!   out_normed  = rms_norm(out, ssm_norm)
//!   out_gated   = out_normed * silu(z)
//!   final       = ssm_out @ out_gated                  ; [d]
//! ```
//!
//! ### Attention (Qwen3Next-style)
//!
//! Q is concatenated with a per-head gate, the gate is sigmoid-applied
//! to the post-attention output before the W_O projection. Multi-axis
//! RoPE with sections is supported but for text-only sequences reduces
//! to a 1-D rotation on the first `rope_dim` dimensions of each head.
//!
//! ### FFN
//!
//! Dense variant: standard SwiGLU. MoE variant: top-K routed experts +
//! a parallel shared expert with sigmoid-gating (see
//! [`ffn_moe_forward`]).
//!
//! All this is verified against `tmp/llama.cpp/src/models/qwen35.cpp`
//! and `qwen35moe.cpp`.

// We intentionally use explicit `for i in 0..n` loops for math-heavy code
// that maps directly to the published reference. Iterator-based rewrites
// would obscure the `[h, r, c]` indexing semantics.
#![allow(clippy::needless_range_loop)]

use crate::qwen35::{LayerKind, Qwen35Config, Qwen35Variant};

use rustorch_gguf::{dequant_to_f32, GgufError, GgufFile, TensorInfo};
use std::path::Path;

/// Errors raised during weight loading or forward computation.
#[derive(Debug)]
pub enum Qwen35CpuError {
    /// GGUF reader / metadata parsing.
    Gguf(String),
    /// Dequantisation failed for some tensor.
    Dequant(String),
    /// A required tensor is missing from the GGUF.
    MissingTensor(String),
    /// Tensor shape does not match what the architecture expects.
    ShapeMismatch {
        /// Tensor name.
        name: String,
        /// Expected `[out, in]` shape.
        expected: Vec<usize>,
        /// Observed shape.
        got: Vec<usize>,
    },
}

impl std::fmt::Display for Qwen35CpuError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Qwen35CpuError::Gguf(s) => write!(f, "gguf: {s}"),
            Qwen35CpuError::Dequant(s) => write!(f, "dequant: {s}"),
            Qwen35CpuError::MissingTensor(s) => write!(f, "missing tensor: {s}"),
            Qwen35CpuError::ShapeMismatch {
                name,
                expected,
                got,
            } => write!(
                f,
                "shape mismatch on {name}: expected {expected:?}, got {got:?}"
            ),
        }
    }
}

impl std::error::Error for Qwen35CpuError {}

impl From<GgufError> for Qwen35CpuError {
    fn from(e: GgufError) -> Self {
        Qwen35CpuError::Gguf(format!("{e:?}"))
    }
}

/// One dense f32 weight tensor in caller-friendly row-major layout.
/// For a 2-D weight, `data` is `[rows * cols]` with `rows = out_dim`,
/// `cols = in_dim` — matmul-vec friendly.
#[derive(Debug)]
pub struct DenseWeight {
    /// Flattened f32 weight values.
    pub data: Vec<f32>,
    /// Logical shape, row-major. For a 2-D weight shape is `[out, in]`.
    pub shape: Vec<usize>,
}

impl DenseWeight {
    /// Number of output rows for a 2-D weight.
    pub fn rows(&self) -> usize {
        self.shape.first().copied().unwrap_or(0)
    }
    /// Number of input columns for a 2-D weight.
    pub fn cols(&self) -> usize {
        self.shape.get(1).copied().unwrap_or(0)
    }
}

/// Per-attention-layer weights.
#[derive(Debug)]
pub struct AttnLayerWeights {
    pub attn_norm: Vec<f32>,
    pub attn_post_norm: Vec<f32>,
    /// Combined Q + per-head gate projection: shape `[2 * head_dim * n_q_heads, d]`.
    /// llama.cpp's Qwen3Next uses a single `wq` of width `2 * head_dim * n_q_heads`
    /// where the first half is Q and the second half is the gate.
    pub w_q: DenseWeight,
    /// K projection: `[head_dim * n_kv_heads, d]`.
    pub w_k: DenseWeight,
    /// V projection: `[head_dim * n_kv_heads, d]`.
    pub w_v: DenseWeight,
    /// Output projection: `[d, head_dim * n_q_heads]`.
    pub w_o: DenseWeight,
    /// Per-head Q norm (rms): `[head_dim]`.
    pub q_norm: Vec<f32>,
    /// Per-head K norm (rms): `[head_dim]`.
    pub k_norm: Vec<f32>,
}

/// Per-SSM-layer weights (Gated Delta Net).
#[derive(Debug)]
pub struct SsmLayerWeights {
    pub attn_norm: Vec<f32>,
    pub attn_post_norm: Vec<f32>,
    /// `[conv_dim, d]` where `conv_dim = 2 * key_dim + value_dim`.
    pub w_qkv: DenseWeight,
    /// `[value_dim, d]` — z gate input projection.
    pub w_gate: DenseWeight,
    /// 1-D depth-wise conv kernel: `[conv_kernel, conv_dim]`. Stored
    /// row-major; conv_dim is the channel dim.
    pub conv1d: Vec<f32>,
    pub conv1d_kernel: usize,
    pub conv1d_channels: usize,
    /// `[num_v_heads, d]` — per-token alpha projection.
    pub ssm_alpha: DenseWeight,
    /// `[num_v_heads, d]` — per-token beta projection.
    pub ssm_beta: DenseWeight,
    /// `[num_v_heads]` — dt bias.
    pub dt_bias: Vec<f32>,
    /// `[num_v_heads]` — A_log scalar parameter (already negative-exp'd
    /// in some checkpoints; we keep it as stored and apply the
    /// llama.cpp ordering: gate = softplus(alpha+bias) * ssm_a).
    pub ssm_a: Vec<f32>,
    /// `[head_v_dim]` — gated rms-norm gamma over the read-out.
    pub ssm_norm: Vec<f32>,
    /// `[d, value_dim]` — output projection back to residual stream.
    pub ssm_out: DenseWeight,
}

/// Dense (qwen35) FFN weights — standard SwiGLU.
#[derive(Debug)]
pub struct DenseFfnWeights {
    pub w_gate: DenseWeight,
    pub w_up: DenseWeight,
    pub w_down: DenseWeight,
}

/// MoE (qwen35moe) FFN weights — top-K routed experts + shared expert.
#[derive(Debug)]
pub struct MoeFfnWeights {
    /// `[n_experts, d]` — routing logits projection.
    pub gate_inp: DenseWeight,
    /// `[n_experts][expert_f, d]` — gate per expert.
    pub gate_exps: Vec<DenseWeight>,
    /// `[n_experts][expert_f, d]` — up per expert.
    pub up_exps: Vec<DenseWeight>,
    /// `[n_experts][d, expert_f]` — down per expert.
    pub down_exps: Vec<DenseWeight>,
    /// `[d]` — shared expert routing weight (sigmoid-gated scalar per token).
    pub gate_inp_shexp: Vec<f32>,
    /// `[expert_f, d]` — shared expert gate.
    pub gate_shexp: DenseWeight,
    /// `[expert_f, d]` — shared expert up.
    pub up_shexp: DenseWeight,
    /// `[d, expert_f]` — shared expert down.
    pub down_shexp: DenseWeight,
}

/// Variant of FFN per layer.
#[derive(Debug)]
pub enum FfnWeights {
    /// Dense SwiGLU FFN.
    Dense(DenseFfnWeights),
    /// MoE FFN.
    Moe(MoeFfnWeights),
}

/// Per-layer weights — discriminated by the layer kind.
#[derive(Debug)]
pub enum LayerWeights {
    /// Attention layer (with FFN).
    Attn {
        /// Attention sub-block.
        attn: AttnLayerWeights,
        /// FFN sub-block.
        ffn: FfnWeights,
    },
    /// SSM (gated delta net) layer (with FFN).
    Ssm {
        /// Gated delta net sub-block.
        ssm: SsmLayerWeights,
        /// FFN sub-block.
        ffn: FfnWeights,
    },
}

/// Full model weights, fully dequantised to f32.
pub struct Qwen35Weights {
    /// Architecture config (parsed from GGUF metadata).
    pub cfg: Qwen35Config,
    /// `[vocab, d]` — token embeddings.
    pub tok_embd: Vec<f32>,
    /// `[d]` — final RMS norm gamma.
    pub output_norm: Vec<f32>,
    /// `[vocab, d]` — lm_head weights.
    pub output: Vec<f32>,
    /// Per-layer weights, indexed by block index.
    pub layers: Vec<LayerWeights>,
}

// ============================================================================
// Loading
// ============================================================================

fn dequant_named(file: &GgufFile, name: &str) -> Result<Vec<f32>, Qwen35CpuError> {
    let info: &TensorInfo = file
        .tensor(name)
        .ok_or_else(|| Qwen35CpuError::MissingTensor(name.to_string()))?;
    let bytes = file.tensor_bytes(info);
    dequant_to_f32(info, bytes).map_err(|e| Qwen35CpuError::Dequant(format!("{name}: {e:?}")))
}

/// Load + dequantise a 2-D weight stored in GGUF as `shape = [in, out]`
/// (the GGUF convention) and return it as row-major `[out, in]` f32.
/// Caller specifies the EXPECTED `[out, in]` shape so we can validate.
fn load_2d(file: &GgufFile, name: &str, expect: [usize; 2]) -> Result<DenseWeight, Qwen35CpuError> {
    let info = file
        .tensor(name)
        .ok_or_else(|| Qwen35CpuError::MissingTensor(name.to_string()))?;
    if info.shape.len() != 2 {
        return Err(Qwen35CpuError::ShapeMismatch {
            name: name.to_string(),
            expected: expect.to_vec(),
            got: info.shape.iter().map(|&x| x as usize).collect(),
        });
    }
    let in_dim = info.shape[0] as usize;
    let out_dim = info.shape[1] as usize;
    if in_dim != expect[1] || out_dim != expect[0] {
        return Err(Qwen35CpuError::ShapeMismatch {
            name: name.to_string(),
            expected: expect.to_vec(),
            got: vec![out_dim, in_dim],
        });
    }
    let bytes = file.tensor_bytes(info);
    let raw = dequant_to_f32(info, bytes)
        .map_err(|e| Qwen35CpuError::Dequant(format!("{name}: {e:?}")))?;
    // GGUF row-major data order for a [in, out] tensor is
    //   for o in 0..out: for i in 0..in: data[o*in + i]
    // ... but wait: ggml stores weights such that the matmul is
    // `y = W @ x` with `W: [out, in]`, BUT the wire layout iterates the
    // FAST axis first. Inspecting llama.cpp's matmul-vec kernels shows
    // each row `o` occupies a contiguous span of `in_dim` elements. Our
    // `dequant_to_f32` returns the data in this same order, so we can
    // reuse it as-is — it's already `[out, in]` row-major.
    Ok(DenseWeight {
        data: raw,
        shape: vec![out_dim, in_dim],
    })
}

/// Load a 1-D vector tensor (RMSNorm gammas, biases, etc.).
fn load_1d(file: &GgufFile, name: &str, expect_len: usize) -> Result<Vec<f32>, Qwen35CpuError> {
    let info = file
        .tensor(name)
        .ok_or_else(|| Qwen35CpuError::MissingTensor(name.to_string()))?;
    let total: u64 = info.shape.iter().product();
    if total as usize != expect_len {
        return Err(Qwen35CpuError::ShapeMismatch {
            name: name.to_string(),
            expected: vec![expect_len],
            got: info.shape.iter().map(|&x| x as usize).collect(),
        });
    }
    dequant_named(file, name)
}

/// Load weights for one attention layer.
fn load_attn_layer(
    file: &GgufFile,
    li: usize,
    cfg: &Qwen35Config,
) -> Result<AttnLayerWeights, Qwen35CpuError> {
    let d = cfg.d;
    let head_dim = cfg.attn_head_dim;
    let n_q = cfg.n_q_heads;
    let n_kv = cfg.n_kv_heads;

    let attn_norm = load_1d(file, &format!("blk.{li}.attn_norm.weight"), d)?;
    let attn_post_norm = load_1d(file, &format!("blk.{li}.post_attention_norm.weight"), d)?;
    // Q is the COMBINED Q+gate projection (Qwen3Next style): width = 2 * head_dim * n_q.
    let w_q = load_2d(
        file,
        &format!("blk.{li}.attn_q.weight"),
        [2 * head_dim * n_q, d],
    )?;
    let w_k = load_2d(
        file,
        &format!("blk.{li}.attn_k.weight"),
        [head_dim * n_kv, d],
    )?;
    let w_v = load_2d(
        file,
        &format!("blk.{li}.attn_v.weight"),
        [head_dim * n_kv, d],
    )?;
    // Output: head_dim * n_q -> d (note: n_q because attention output is n_q heads).
    let w_o = load_2d(
        file,
        &format!("blk.{li}.attn_output.weight"),
        [d, head_dim * n_q],
    )?;
    let q_norm = load_1d(file, &format!("blk.{li}.attn_q_norm.weight"), head_dim)?;
    let k_norm = load_1d(file, &format!("blk.{li}.attn_k_norm.weight"), head_dim)?;
    Ok(AttnLayerWeights {
        attn_norm,
        attn_post_norm,
        w_q,
        w_k,
        w_v,
        w_o,
        q_norm,
        k_norm,
    })
}

/// Load weights for one SSM/Gated-DeltaNet layer.
fn load_ssm_layer(
    file: &GgufFile,
    li: usize,
    cfg: &Qwen35Config,
) -> Result<SsmLayerWeights, Qwen35CpuError> {
    let d = cfg.d;
    let head_kv = cfg.ssm_state; // head_k_dim = head_v_dim = ssm_state
    let n_k = cfg.ssm_groups;
    let n_v = cfg.ssm_dt_rank;
    let key_dim = head_kv * n_k;
    let value_dim = head_kv * n_v;
    let conv_dim = 2 * key_dim + value_dim;

    let attn_norm = load_1d(file, &format!("blk.{li}.attn_norm.weight"), d)?;
    let attn_post_norm = load_1d(file, &format!("blk.{li}.post_attention_norm.weight"), d)?;
    let w_qkv = load_2d(file, &format!("blk.{li}.attn_qkv.weight"), [conv_dim, d])?;
    let w_gate = load_2d(file, &format!("blk.{li}.attn_gate.weight"), [value_dim, d])?;

    // 1-D conv kernel: GGUF shape is [conv_kernel, conv_channels]. We
    // store it as a flat [conv_kernel * conv_dim] f32 in row-major.
    let conv1d_name = format!("blk.{li}.ssm_conv1d.weight");
    let conv_info = file
        .tensor(&conv1d_name)
        .ok_or_else(|| Qwen35CpuError::MissingTensor(conv1d_name.clone()))?;
    if conv_info.shape.len() != 2
        || conv_info.shape[0] as usize != cfg.ssm_conv_kernel
        || conv_info.shape[1] as usize != conv_dim
    {
        return Err(Qwen35CpuError::ShapeMismatch {
            name: conv1d_name,
            expected: vec![cfg.ssm_conv_kernel, conv_dim],
            got: conv_info.shape.iter().map(|&x| x as usize).collect(),
        });
    }
    let conv1d = dequant_named(file, &conv1d_name)?;

    let ssm_alpha = load_2d(file, &format!("blk.{li}.ssm_alpha.weight"), [n_v, d])?;
    let ssm_beta = load_2d(file, &format!("blk.{li}.ssm_beta.weight"), [n_v, d])?;
    let dt_bias = load_1d(file, &format!("blk.{li}.ssm_dt.bias"), n_v)?;
    let ssm_a = load_1d(file, &format!("blk.{li}.ssm_a"), n_v)?;
    let ssm_norm = load_1d(file, &format!("blk.{li}.ssm_norm.weight"), head_kv)?;
    let ssm_out = load_2d(file, &format!("blk.{li}.ssm_out.weight"), [d, value_dim])?;

    Ok(SsmLayerWeights {
        attn_norm,
        attn_post_norm,
        w_qkv,
        w_gate,
        conv1d,
        conv1d_kernel: cfg.ssm_conv_kernel,
        conv1d_channels: conv_dim,
        ssm_alpha,
        ssm_beta,
        dt_bias,
        ssm_a,
        ssm_norm,
        ssm_out,
    })
}

/// Load FFN weights for one layer (dense or MoE).
fn load_ffn(file: &GgufFile, li: usize, cfg: &Qwen35Config) -> Result<FfnWeights, Qwen35CpuError> {
    match cfg.variant {
        Qwen35Variant::Qwen3PureTransformer | Qwen35Variant::Dense => {
            let f = cfg.f;
            let d = cfg.d;
            let w_gate = load_2d(file, &format!("blk.{li}.ffn_gate.weight"), [f, d])?;
            let w_up = load_2d(file, &format!("blk.{li}.ffn_up.weight"), [f, d])?;
            let w_down = load_2d(file, &format!("blk.{li}.ffn_down.weight"), [d, f])?;
            Ok(FfnWeights::Dense(DenseFfnWeights {
                w_gate,
                w_up,
                w_down,
            }))
        },
        Qwen35Variant::Moe => {
            let d = cfg.d;
            let ef = cfg.expert_f;
            let n_e = cfg.n_experts;
            // GGUF stores stacked-expert tensors as 3-D: shape [in, ef, n_experts]
            // for gate/up and [ef, d, n_experts] for down.
            // We dequantise the whole stacked tensor then split into per-expert
            // 2-D `DenseWeight` slices for ergonomic forward code.
            let gate_inp = load_2d(file, &format!("blk.{li}.ffn_gate_inp.weight"), [n_e, d])?;
            let gate_inp_shexp = load_1d(file, &format!("blk.{li}.ffn_gate_inp_shexp.weight"), d)?;
            let gate_shexp = load_2d(file, &format!("blk.{li}.ffn_gate_shexp.weight"), [ef, d])?;
            let up_shexp = load_2d(file, &format!("blk.{li}.ffn_up_shexp.weight"), [ef, d])?;
            let down_shexp = load_2d(file, &format!("blk.{li}.ffn_down_shexp.weight"), [d, ef])?;

            // Stacked-expert tensors. Naming: shape [in, ef, n_experts] for gate/up,
            // [ef, d, n_experts] for down. We split into n_experts DenseWeight blocks.
            let gate_exps =
                load_stacked_experts(file, &format!("blk.{li}.ffn_gate_exps.weight"), n_e, ef, d)?;
            let up_exps =
                load_stacked_experts(file, &format!("blk.{li}.ffn_up_exps.weight"), n_e, ef, d)?;
            let down_exps = load_stacked_experts_down(
                file,
                &format!("blk.{li}.ffn_down_exps.weight"),
                n_e,
                d,
                ef,
            )?;

            Ok(FfnWeights::Moe(MoeFfnWeights {
                gate_inp,
                gate_exps,
                up_exps,
                down_exps,
                gate_inp_shexp,
                gate_shexp,
                up_shexp,
                down_shexp,
            }))
        },
    }
}

/// Load a stacked-expert FFN gate or up tensor. GGUF shape `[d, ef,
/// n_experts]`; result is a Vec of n_experts 2-D weights of shape
/// `[ef, d]`.
fn load_stacked_experts(
    file: &GgufFile,
    name: &str,
    n_experts: usize,
    ef: usize,
    d: usize,
) -> Result<Vec<DenseWeight>, Qwen35CpuError> {
    let info = file
        .tensor(name)
        .ok_or_else(|| Qwen35CpuError::MissingTensor(name.to_string()))?;
    if info.shape.len() != 3
        || info.shape[0] as usize != d
        || info.shape[1] as usize != ef
        || info.shape[2] as usize != n_experts
    {
        return Err(Qwen35CpuError::ShapeMismatch {
            name: name.to_string(),
            expected: vec![d, ef, n_experts],
            got: info.shape.iter().map(|&x| x as usize).collect(),
        });
    }
    let raw = dequant_named(file, name)?;
    let stride = ef * d;
    let mut out = Vec::with_capacity(n_experts);
    for e in 0..n_experts {
        let slice = &raw[e * stride..(e + 1) * stride];
        out.push(DenseWeight {
            data: slice.to_vec(),
            shape: vec![ef, d],
        });
    }
    Ok(out)
}

/// Load stacked-expert ffn_down_exps. GGUF shape `[ef, d, n_experts]`;
/// per-expert weight is `[d, ef]`.
fn load_stacked_experts_down(
    file: &GgufFile,
    name: &str,
    n_experts: usize,
    d: usize,
    ef: usize,
) -> Result<Vec<DenseWeight>, Qwen35CpuError> {
    let info = file
        .tensor(name)
        .ok_or_else(|| Qwen35CpuError::MissingTensor(name.to_string()))?;
    if info.shape.len() != 3
        || info.shape[0] as usize != ef
        || info.shape[1] as usize != d
        || info.shape[2] as usize != n_experts
    {
        return Err(Qwen35CpuError::ShapeMismatch {
            name: name.to_string(),
            expected: vec![ef, d, n_experts],
            got: info.shape.iter().map(|&x| x as usize).collect(),
        });
    }
    let raw = dequant_named(file, name)?;
    let stride = d * ef;
    let mut out = Vec::with_capacity(n_experts);
    for e in 0..n_experts {
        let slice = &raw[e * stride..(e + 1) * stride];
        out.push(DenseWeight {
            data: slice.to_vec(),
            shape: vec![d, ef],
        });
    }
    Ok(out)
}

/// Load full Qwen3.5/3.6 model into f32 dense weights. Slow (~30s+ for
/// the 27B because we dequantise everything, allocating ~30 GB f32 in
/// total for the 27B and ~20 GB for the 35B-A3B).
pub fn load_weights(path: &Path, cfg: &Qwen35Config) -> Result<Qwen35Weights, Qwen35CpuError> {
    let file = GgufFile::open(path)?;

    let tok_embd = {
        let info = file
            .tensor("token_embd.weight")
            .ok_or_else(|| Qwen35CpuError::MissingTensor("token_embd.weight".to_string()))?;
        // GGUF token_embd is [d, vocab] so dequant gives [vocab, d] row-major.
        let _ = info;
        dequant_named(&file, "token_embd.weight")?
    };
    let output_norm = load_1d(&file, "output_norm.weight", cfg.d)?;
    let output = dequant_named(&file, "output.weight")?;

    let mut layers = Vec::with_capacity(cfg.n_layers);
    for li in 0..cfg.n_layers {
        let kind = cfg.layer_kind(li);
        let layer = match kind {
            LayerKind::Attention => LayerWeights::Attn {
                attn: load_attn_layer(&file, li, cfg)?,
                ffn: load_ffn(&file, li, cfg)?,
            },
            LayerKind::Ssm => LayerWeights::Ssm {
                ssm: load_ssm_layer(&file, li, cfg)?,
                ffn: load_ffn(&file, li, cfg)?,
            },
        };
        layers.push(layer);
    }

    Ok(Qwen35Weights {
        cfg: cfg.clone(),
        tok_embd,
        output_norm,
        output,
        layers,
    })
}

// ============================================================================
// CPU primitives
// ============================================================================

/// `y = W @ x` with `W: [N, K]` row-major, `x: [K]`, `y: [N]`. Plain
/// triple-loop f32 — slow but obviously correct.
pub fn gemv(w: &[f32], x: &[f32], y: &mut [f32], n: usize, k: usize) {
    debug_assert_eq!(w.len(), n * k);
    debug_assert_eq!(x.len(), k);
    debug_assert_eq!(y.len(), n);
    use rayon::prelude::*;
    // T242 perf : for m=1 (single-token autoregressive decode), the matmul
    // is memory-bound (limited by RAM bandwidth, not compute). Parallelizing
    // the output rows across cores spreads the read load across CPU cores
    // → ~15-18× speedup on ARM Grace (DGX Spark, 20 cores).
    //
    // Inner loop is a 4-wide unrolled dot product. LLVM auto-vectorizes
    // each chunk to NEON (ARM) or AVX (x86).
    //
    // Note : `gemm::gemm` is BETTER for m>1 (matrix-matrix) but WORSE for
    // m=1 because its sgemv dispatch path doesn't parallelize well across
    // output rows on memory-bound shapes. Tested empirically.
    if n * k < 65_536 {
        for i in 0..n {
            let row = &w[i * k..(i + 1) * k];
            let mut acc = 0.0_f32;
            for j in 0..k {
                acc += row[j] * x[j];
            }
            y[i] = acc;
        }
        return;
    }
    y.par_iter_mut().enumerate().for_each(|(i, y_i)| {
        let row = &w[i * k..(i + 1) * k];
        let mut a0 = 0.0f32;
        let mut a1 = 0.0f32;
        let mut a2 = 0.0f32;
        let mut a3 = 0.0f32;
        let chunks = k / 4;
        for c in 0..chunks {
            let off = c * 4;
            a0 += row[off] * x[off];
            a1 += row[off + 1] * x[off + 1];
            a2 += row[off + 2] * x[off + 2];
            a3 += row[off + 3] * x[off + 3];
        }
        let mut acc = a0 + a1 + a2 + a3;
        for j in (chunks * 4)..k {
            acc += row[j] * x[j];
        }
        *y_i = acc;
    });
}

/// In-place RMS norm with per-element gain: `x_i := x_i / sqrt(mean(x²)
/// + eps) * gamma_i`.
pub fn rms_norm(x: &mut [f32], gamma: &[f32], eps: f32) {
    debug_assert_eq!(x.len(), gamma.len());
    let n = x.len() as f32;
    let mut ss = 0.0_f32;
    for &v in x.iter() {
        ss += v * v;
    }
    let inv = 1.0 / (ss / n + eps).sqrt();
    for i in 0..x.len() {
        x[i] = x[i] * inv * gamma[i];
    }
}

/// Per-vector L2 normalisation: `x_i := x_i / sqrt(sum(x²) + eps)`.
/// Used inside the gated delta net for Q and K.
pub fn l2_norm(x: &mut [f32], eps: f32) {
    let mut ss = 0.0_f32;
    for &v in x.iter() {
        ss += v * v;
    }
    let inv = 1.0 / (ss + eps).sqrt();
    for v in x.iter_mut() {
        *v *= inv;
    }
}

/// SiLU activation: `silu(x) = x * sigmoid(x)`.
pub fn silu(x: &mut [f32]) {
    for v in x.iter_mut() {
        let s = 1.0 / (1.0 + (-*v).exp());
        *v *= s;
    }
}

/// Sigmoid activation in place.
pub fn sigmoid(x: &mut [f32]) {
    for v in x.iter_mut() {
        *v = 1.0 / (1.0 + (-*v).exp());
    }
}

/// Softplus activation: `softplus(x) = ln(1 + exp(x))` with stable
/// implementation for large positive values.
pub fn softplus(x: &mut [f32]) {
    for v in x.iter_mut() {
        // Numerical stability: for large x, log(1+exp(x)) ≈ x; for very
        // negative x, log(1+exp(x)) ≈ exp(x).
        if *v > 20.0 {
            // pass
        } else if *v < -20.0 {
            *v = (*v).exp();
        } else {
            *v = (1.0 + (*v).exp()).ln();
        }
    }
}

/// Argmax over a slice.
pub fn argmax(logits: &[f32]) -> u32 {
    let mut bi = 0u32;
    let mut bv = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > bv {
            bv = v;
            bi = i as u32;
        }
    }
    bi
}

// ============================================================================
// Per-token state
// ============================================================================

/// Attention KV cache for one attention layer.
#[derive(Debug)]
pub struct AttnLayerCache {
    /// Stacked K vectors over time: `[max_seq, n_kv_heads, head_dim]`.
    pub k: Vec<f32>,
    /// Stacked V vectors over time: `[max_seq, n_kv_heads, head_dim]`.
    pub v: Vec<f32>,
}

/// SSM recurrent state for one SSM layer.
#[derive(Debug)]
pub struct SsmLayerState {
    /// 1-D conv ring buffer: `[(conv_kernel - 1) * conv_channels]` f32.
    /// Stored row-major so we shift one row left each step.
    pub conv_state: Vec<f32>,
    /// Per-head delta-net state: `[num_v_heads * head_v_dim * head_v_dim]`.
    pub state: Vec<f32>,
}

/// Per-layer state — discriminated by layer kind.
#[derive(Debug)]
pub enum LayerState {
    /// Attention KV cache.
    Attn(AttnLayerCache),
    /// SSM recurrent state.
    Ssm(SsmLayerState),
}

/// Top-level model state: per-layer caches/states.
#[derive(Debug)]
pub struct Qwen35State {
    /// Per-block state, sized `n_layers`.
    pub layers: Vec<LayerState>,
    /// Maximum context length (size of attention KV cache time dim).
    pub max_seq: usize,
}

impl Qwen35State {
    /// Allocate fresh zero-state for the given config and max sequence length.
    pub fn new(cfg: &Qwen35Config, max_seq: usize) -> Self {
        let mut layers = Vec::with_capacity(cfg.n_layers);
        for li in 0..cfg.n_layers {
            let s = match cfg.layer_kind(li) {
                LayerKind::Attention => {
                    let kv_dim = cfg.attn_head_dim * cfg.n_kv_heads;
                    LayerState::Attn(AttnLayerCache {
                        k: vec![0.0_f32; max_seq * kv_dim],
                        v: vec![0.0_f32; max_seq * kv_dim],
                    })
                },
                LayerKind::Ssm => {
                    let conv_dim =
                        2 * cfg.ssm_state * cfg.ssm_groups + cfg.ssm_state * cfg.ssm_dt_rank;
                    let conv_state = vec![0.0_f32; (cfg.ssm_conv_kernel - 1) * conv_dim];
                    let head_v = cfg.ssm_state;
                    let n_v = cfg.ssm_dt_rank;
                    let state = vec![0.0_f32; n_v * head_v * head_v];
                    LayerState::Ssm(SsmLayerState { conv_state, state })
                },
            };
            layers.push(s);
        }
        Qwen35State { layers, max_seq }
    }
}

// ============================================================================
// Forward — minimal scaffold (per-block forward functions are stubs that
// will be filled in subsequent commits T141b/T141c). The structure here
// shows the wiring; running it as-is on a real model will currently
// produce wrong outputs because the SSM and attention forwards just
// return their input unchanged.
// ============================================================================

/// Single-token forward — returns the argmax of the lm_head logits at
/// the given decode position.
///
/// **Status:** scaffolded. The block forwards (`ssm_block_forward`,
/// `attn_block_forward`, `ffn_dense_forward`, `ffn_moe_forward`) are
/// stubs that simply pass the input through. They will be filled in
/// T141b (SSM block), T141c (attention block), T142 (MoE FFN) — at
/// which point this function produces the correct argmax.
pub fn forward_token(
    weights: &Qwen35Weights,
    state: &mut Qwen35State,
    token: u32,
    position: usize,
) -> u32 {
    let cfg = &weights.cfg;
    let d = cfg.d;
    // x = token embedding (copy)
    let mut x = weights.tok_embd[(token as usize) * d..(token as usize + 1) * d].to_vec();

    for li in 0..cfg.n_layers {
        let layer = &weights.layers[li];
        let mut residual = x.clone();
        let mut cur = x.clone();
        rms_norm(
            &mut cur,
            match layer {
                LayerWeights::Attn { attn, .. } => &attn.attn_norm,
                LayerWeights::Ssm { ssm, .. } => &ssm.attn_norm,
            },
            cfg.rms_eps,
        );
        // Mixer (attention or SSM) — stubbed.
        match (layer, &mut state.layers[li]) {
            (LayerWeights::Attn { attn, .. }, LayerState::Attn(cache)) => {
                attn_block_forward(attn, &mut cur, cache, cfg, position);
            },
            (LayerWeights::Ssm { ssm, .. }, LayerState::Ssm(s)) => {
                ssm_block_forward(ssm, &mut cur, s, cfg);
            },
            _ => unreachable!("layer/state kind mismatch"),
        }
        // Residual after mixer.
        for (a, b) in residual.iter_mut().zip(cur.iter()) {
            *a += *b;
        }
        // FFN with post-attention norm.
        let mut ffn_in = residual.clone();
        rms_norm(
            &mut ffn_in,
            match layer {
                LayerWeights::Attn { attn, .. } => &attn.attn_post_norm,
                LayerWeights::Ssm { ssm, .. } => &ssm.attn_post_norm,
            },
            cfg.rms_eps,
        );
        let ffn_out = match layer {
            LayerWeights::Attn { ffn, .. } | LayerWeights::Ssm { ffn, .. } => match ffn {
                FfnWeights::Dense(w) => ffn_dense_forward(w, &ffn_in),
                FfnWeights::Moe(w) => ffn_moe_forward(w, &ffn_in, cfg),
            },
        };
        // Final residual.
        x = residual;
        for (a, b) in x.iter_mut().zip(ffn_out.iter()) {
            *a += *b;
        }
    }

    // Final RMS norm + lm_head.
    rms_norm(&mut x, &weights.output_norm, cfg.rms_eps);
    let mut logits = vec![0.0_f32; cfg.vocab];
    gemv(&weights.output, &x, &mut logits, cfg.vocab, cfg.d);
    argmax(&logits)
}

/// Apply 1-D RoPE to the first `rope_dim` dims of each head, in place.
/// Uses the half-split convention (rotates pairs `(x_i, x_{i+rope_dim/2})`)
/// as in standard Qwen / Llama. `rope_base` is the theta.
pub fn rope_inplace(
    x: &mut [f32],
    n_heads: usize,
    head_dim: usize,
    rope_dim: usize,
    position: usize,
    rope_base: f32,
) {
    let half = rope_dim / 2;
    for h in 0..n_heads {
        let off = h * head_dim;
        for i in 0..half {
            let theta = (position as f32) / rope_base.powf((2 * i) as f32 / rope_dim as f32);
            let (s, c) = theta.sin_cos();
            let a = x[off + i];
            let b = x[off + i + half];
            x[off + i] = a * c - b * s;
            x[off + i + half] = a * s + b * c;
        }
    }
}

/// **Qwen3Next-style attention forward — single token decode step.**
///
/// `cur` enters as the post-attn-norm residual stream (`[d]`) and is
/// overwritten with the attention block output. The KV cache is appended
/// to in place at `position`.
///
/// Key differences from vanilla Qwen3 MHA:
///  1. `wq` outputs `2 * head_dim * n_q_heads` — first half is Q,
///     second half is a per-head gate.
///  2. The post-attention output is multiplied by `sigmoid(gate)`
///     before the W_O projection.
pub fn attn_block_forward(
    attn: &AttnLayerWeights,
    cur: &mut [f32],
    cache: &mut AttnLayerCache,
    cfg: &Qwen35Config,
    position: usize,
) {
    let d = cfg.d;
    let head_dim = cfg.attn_head_dim;
    let n_q = cfg.n_q_heads;
    let n_kv = cfg.n_kv_heads;
    let kv_dim = head_dim * n_kv;
    let q_dim = head_dim * n_q;
    let eps = cfg.rms_eps;

    let cur_in = cur.to_vec();

    // ---- 1. Q (combined Q + gate), K, V projections ----
    let mut qg = vec![0.0_f32; 2 * q_dim];
    gemv(&attn.w_q.data, &cur_in, &mut qg, 2 * q_dim, d);

    // Split: per llama.cpp, the layout is [head_dim, 2 * n_head] interleaved
    // so that Q occupies even-numbered halves of each per-head pair. Concretely,
    // for head h in 0..n_q:
    //   q_h = qg[h * 2 * head_dim .. h * 2 * head_dim + head_dim]
    //   gate_h = qg[h * 2 * head_dim + head_dim .. (h + 1) * 2 * head_dim]
    let mut q = vec![0.0_f32; q_dim];
    let mut gate = vec![0.0_f32; q_dim];
    for h in 0..n_q {
        let src_off = h * 2 * head_dim;
        let dst_off = h * head_dim;
        q[dst_off..dst_off + head_dim].copy_from_slice(&qg[src_off..src_off + head_dim]);
        gate[dst_off..dst_off + head_dim]
            .copy_from_slice(&qg[src_off + head_dim..src_off + 2 * head_dim]);
    }

    let mut k = vec![0.0_f32; kv_dim];
    gemv(&attn.w_k.data, &cur_in, &mut k, kv_dim, d);
    let mut v = vec![0.0_f32; kv_dim];
    gemv(&attn.w_v.data, &cur_in, &mut v, kv_dim, d);

    // ---- 2. Per-head Q-norm and K-norm (with shared gamma per head_dim) ----
    for h in 0..n_q {
        rms_norm(&mut q[h * head_dim..(h + 1) * head_dim], &attn.q_norm, eps);
    }
    for h in 0..n_kv {
        rms_norm(&mut k[h * head_dim..(h + 1) * head_dim], &attn.k_norm, eps);
    }

    // ---- 3. RoPE on Q and K (first rope_dim dims of each head) ----
    rope_inplace(&mut q, n_q, head_dim, cfg.rope_dim, position, cfg.rope_base);
    rope_inplace(
        &mut k,
        n_kv,
        head_dim,
        cfg.rope_dim,
        position,
        cfg.rope_base,
    );

    // ---- 4. Append K, V to cache at `position` ----
    debug_assert!(position < cache.k.len() / kv_dim);
    cache.k[position * kv_dim..(position + 1) * kv_dim].copy_from_slice(&k);
    cache.v[position * kv_dim..(position + 1) * kv_dim].copy_from_slice(&v);

    // ---- 5. GQA attention over positions 0..=position ----
    // For each q-head h_q in 0..n_q:
    //   k-head h_kv = h_q / (n_q / n_kv)
    //   scores[t] = q_h · K[t, h_kv] / sqrt(head_dim)   for t in 0..=position
    //   weights = softmax(scores)
    //   out_h = Σ weights[t] * V[t, h_kv]
    let kv_len = position + 1;
    let scale = 1.0_f32 / (head_dim as f32).sqrt();
    let gqa_factor = n_q / n_kv;
    let mut attn_out = vec![0.0_f32; q_dim];
    let mut scores = vec![0.0_f32; kv_len];
    for h_q in 0..n_q {
        let h_kv = h_q / gqa_factor;
        let q_h = &q[h_q * head_dim..(h_q + 1) * head_dim];
        // Compute scores
        for t in 0..kv_len {
            let kv_off = t * kv_dim + h_kv * head_dim;
            let mut s = 0.0_f32;
            for i in 0..head_dim {
                s += q_h[i] * cache.k[kv_off + i];
            }
            scores[t] = s * scale;
        }
        // Softmax
        let mut max_s = f32::NEG_INFINITY;
        for &s in scores.iter().take(kv_len) {
            if s > max_s {
                max_s = s;
            }
        }
        let mut sum_exp = 0.0_f32;
        for s in scores.iter_mut().take(kv_len) {
            *s = (*s - max_s).exp();
            sum_exp += *s;
        }
        let inv = 1.0 / sum_exp.max(1e-30);
        for s in scores.iter_mut().take(kv_len) {
            *s *= inv;
        }
        // Weighted sum of V
        let out_off = h_q * head_dim;
        for i in 0..head_dim {
            attn_out[out_off + i] = 0.0;
        }
        for t in 0..kv_len {
            let kv_off = t * kv_dim + h_kv * head_dim;
            let w_t = scores[t];
            for i in 0..head_dim {
                attn_out[out_off + i] += w_t * cache.v[kv_off + i];
            }
        }
    }

    // ---- 6. Apply per-head sigmoid gate ----
    sigmoid(&mut gate);
    for i in 0..q_dim {
        attn_out[i] *= gate[i];
    }

    // ---- 7. Output projection: w_o [d, q_dim] ----
    let mut out = vec![0.0_f32; d];
    gemv(&attn.w_o.data, &attn_out, &mut out, d, q_dim);
    cur.copy_from_slice(&out);
}

/// **Gated Delta Net forward — single token decode step.**
///
/// `cur` enters the block as the post-attn-norm residual stream of
/// shape `[d]` and is overwritten with the SSM mixer output (also
/// `[d]`). The recurrent state and conv ring buffer in `state` are
/// updated in place.
///
/// Faithful port of `llama_model_qwen35::graph::build_layer_attn_linear`
/// from llama.cpp. Decode-time, n_seqs=1, n_seq_tokens=1.
pub fn ssm_block_forward(
    ssm: &SsmLayerWeights,
    cur: &mut [f32],
    state: &mut SsmLayerState,
    cfg: &Qwen35Config,
) {
    let d = cfg.d;
    let head_kv = cfg.ssm_state; // head_k_dim = head_v_dim
    let n_k = cfg.ssm_groups;
    let n_v = cfg.ssm_dt_rank;
    let key_dim = head_kv * n_k;
    let value_dim = head_kv * n_v;
    let conv_dim = ssm.conv1d_channels;
    debug_assert_eq!(conv_dim, 2 * key_dim + value_dim);
    let head_v_dim = value_dim / n_v;
    debug_assert_eq!(head_v_dim, head_kv);
    let conv_kernel = ssm.conv1d_kernel;
    let eps = cfg.rms_eps;

    // ---- 1. Input projections from `cur` (the input shadow) ----
    let cur_in = cur.to_vec();

    let mut qkv_mixed = vec![0.0_f32; conv_dim];
    gemv(&ssm.w_qkv.data, &cur_in, &mut qkv_mixed, conv_dim, d);

    let mut z = vec![0.0_f32; value_dim];
    gemv(&ssm.w_gate.data, &cur_in, &mut z, value_dim, d);

    let mut beta = vec![0.0_f32; n_v];
    gemv(&ssm.ssm_beta.data, &cur_in, &mut beta, n_v, d);
    sigmoid(&mut beta);

    let mut alpha = vec![0.0_f32; n_v];
    gemv(&ssm.ssm_alpha.data, &cur_in, &mut alpha, n_v, d);
    for i in 0..n_v {
        alpha[i] += ssm.dt_bias[i];
    }
    softplus(&mut alpha);
    // gate_h = alpha_softplus * ssm_a (per-head scalar gate). llama.cpp
    // uses `gate = ggml_mul(alpha_softplus, ssm_a)` — pointwise.
    let mut gate_h = vec![0.0_f32; n_v];
    for i in 0..n_v {
        gate_h[i] = alpha[i] * ssm.ssm_a[i];
    }

    // ---- 2. 1-D causal conv with running ring buffer ----
    // conv_state holds the previous (kernel - 1) timesteps — flat row-major
    // `[(kernel - 1), conv_dim]`. We treat it as `[conv_dim][kernel - 1]`
    // for simpler per-channel convolution. We re-arrange logically: for
    // channel c, history is conv_state[t * conv_dim + c] for t in 0..kernel-1.
    let mut conv_out = vec![0.0_f32; conv_dim];
    let cs = &mut state.conv_state;
    debug_assert_eq!(cs.len(), (conv_kernel - 1) * conv_dim);

    // Build a length-`conv_kernel` window per channel: history rows + qkv_mixed.
    // The kernel weight is row-major `[conv_kernel, conv_dim]`, channel = inner.
    for c in 0..conv_dim {
        let mut acc = 0.0_f32;
        // Window positions 0..(kernel-1) = history; (kernel-1) = current input
        for t in 0..(conv_kernel - 1) {
            let v = cs[t * conv_dim + c];
            let w = ssm.conv1d[t * conv_dim + c];
            acc += v * w;
        }
        let v = qkv_mixed[c];
        let w = ssm.conv1d[(conv_kernel - 1) * conv_dim + c];
        acc += v * w;
        conv_out[c] = acc;
    }
    // Update conv ring buffer: shift left by one timestep, append qkv_mixed.
    if conv_kernel >= 2 {
        // Shift positions 1..(kernel-1) into 0..(kernel-2).
        for t in 0..(conv_kernel - 2) {
            for c in 0..conv_dim {
                cs[t * conv_dim + c] = cs[(t + 1) * conv_dim + c];
            }
        }
        // Place current qkv_mixed at last history slot.
        let last = conv_kernel - 2;
        for c in 0..conv_dim {
            cs[last * conv_dim + c] = qkv_mixed[c];
        }
    }
    // SiLU(conv_out)
    silu(&mut conv_out);

    // ---- 3. Split q/k/v from conv_out and per-head L2 normalise q,k ----
    // Layout: q is first key_dim, k is next key_dim, v is last value_dim.
    let mut q = conv_out[0..key_dim].to_vec();
    let mut k = conv_out[key_dim..2 * key_dim].to_vec();
    let v = conv_out[2 * key_dim..2 * key_dim + value_dim].to_vec();
    // Per-head L2 norm. Q/K have num_k_heads heads of dim head_kv.
    for h in 0..n_k {
        l2_norm(&mut q[h * head_kv..(h + 1) * head_kv], eps);
        l2_norm(&mut k[h * head_kv..(h + 1) * head_kv], eps);
    }

    // ---- 4. If num_k_heads != num_v_heads, broadcast q,k to num_v_heads ----
    // The repeat factor is num_v_heads / num_k_heads.
    let repeat = if n_k == n_v {
        1
    } else {
        debug_assert_eq!(n_v % n_k, 0);
        n_v / n_k
    };
    let q_v: Vec<f32> = if repeat == 1 {
        q.clone()
    } else {
        let mut out = vec![0.0_f32; n_v * head_kv];
        for h_v in 0..n_v {
            let h_k = h_v / repeat;
            out[h_v * head_kv..(h_v + 1) * head_kv]
                .copy_from_slice(&q[h_k * head_kv..(h_k + 1) * head_kv]);
        }
        out
    };
    let k_v: Vec<f32> = if repeat == 1 {
        k.clone()
    } else {
        let mut out = vec![0.0_f32; n_v * head_kv];
        for h_v in 0..n_v {
            let h_k = h_v / repeat;
            out[h_v * head_kv..(h_v + 1) * head_kv]
                .copy_from_slice(&k[h_k * head_kv..(h_k + 1) * head_kv]);
        }
        out
    };

    // ---- 5. Delta-net recurrence ----
    // For each head h in 0..n_v:
    //   state[h]  ← exp(gate_h[h]) * state[h] + beta[h] * outer(v[h], k[h])
    //   out[h]    ← state[h] @ q[h]
    // state[h] is a [head_v_dim, head_v_dim] matrix (because head_k_dim = head_v_dim).
    let head_dim = head_v_dim; // = head_kv
    let mut out_per_head = vec![0.0_f32; n_v * head_dim];
    for h in 0..n_v {
        let g = gate_h[h].exp();
        let b = beta[h];
        let s_off = h * head_dim * head_dim;
        let v_h = &v[h * head_dim..(h + 1) * head_dim];
        let k_h = &k_v[h * head_dim..(h + 1) * head_dim];
        let q_h = &q_v[h * head_dim..(h + 1) * head_dim];
        // Update state: row r → state[r, c] = g * state[r, c] + b * v[r] * k[c]
        // Then out[r] = sum_c state[r, c] * q[c]
        for r in 0..head_dim {
            let v_r = v_h[r];
            let mut acc = 0.0_f32;
            for c in 0..head_dim {
                let updated = g * state.state[s_off + r * head_dim + c] + b * v_r * k_h[c];
                state.state[s_off + r * head_dim + c] = updated;
                acc += updated * q_h[c];
            }
            out_per_head[h * head_dim + r] = acc;
        }
    }

    // ---- 6. Gated norm on the per-head output ----
    // norm_gated(out, ssm_norm, z): rms_norm(out, ssm_norm) * silu(z)
    // ssm_norm has shape [head_v_dim] — applied per-head.
    let mut out_normed = out_per_head;
    for h in 0..n_v {
        rms_norm(
            &mut out_normed[h * head_dim..(h + 1) * head_dim],
            &ssm.ssm_norm,
            eps,
        );
    }
    let mut z_silu = z;
    silu(&mut z_silu);
    for i in 0..n_v * head_dim {
        out_normed[i] *= z_silu[i];
    }

    // ---- 7. Final output projection: ssm_out [d, value_dim] ----
    let mut out_final = vec![0.0_f32; d];
    gemv(&ssm.ssm_out.data, &out_normed, &mut out_final, d, value_dim);
    cur.copy_from_slice(&out_final);
}

/// Dense SwiGLU FFN: `y = w_down @ (silu(w_gate @ x) * (w_up @ x))`.
pub fn ffn_dense_forward(w: &DenseFfnWeights, x: &[f32]) -> Vec<f32> {
    let f = w.w_gate.rows();
    let d = w.w_gate.cols();
    let mut gate = vec![0.0_f32; f];
    let mut up = vec![0.0_f32; f];
    gemv(&w.w_gate.data, x, &mut gate, f, d);
    gemv(&w.w_up.data, x, &mut up, f, d);
    silu(&mut gate);
    for i in 0..f {
        gate[i] *= up[i];
    }
    let mut out = vec![0.0_f32; d];
    gemv(&w.w_down.data, &gate, &mut out, d, f);
    out
}

/// MoE FFN forward (Qwen3.6-35B-A3B). Top-K routed experts + parallel
/// shared expert :
///
/// ```text
///   raw_scores = ffn_gate_inp @ h                     [n_experts]
///   probs      = softmax(raw_scores)
///   topk_e, topk_p = top-K(probs, K)
///   topk_p_n   = topk_p / sum(topk_p)                 (renormalize)
///
///   y_routed   = Σ_{i=0..K} topk_p_n[i] · expert_swiglu(topk_e[i], h)
///   shexp_w    = sigmoid(ffn_gate_inp_shexp · h)
///   y_shexp    = shexp_w · shared_swiglu(h)
///   y          = y_routed + y_shexp
/// ```
///
/// where `expert_swiglu(e, h) = ffn_down_exps[e] @ (silu(ffn_gate_exps[e] @ h)
/// * (ffn_up_exps[e] @ h))` and `shared_swiglu(h)` is the same form on the
/// `*_shexp` weights.
pub fn ffn_moe_forward(w: &MoeFfnWeights, x: &[f32], cfg: &Qwen35Config) -> Vec<f32> {
    let d = cfg.d;
    let ef = cfg.expert_f;
    let n_experts = cfg.n_experts;
    let k = cfg.n_experts_used;
    assert_eq!(x.len(), d, "x dim mismatch");
    assert_eq!(w.gate_inp.rows(), n_experts);
    assert_eq!(w.gate_inp.cols(), d);

    // ---- 1. Router : raw_scores = gate_inp @ x ; probs = softmax(raw) ----
    let mut raw_scores = vec![0.0_f32; n_experts];
    gemv(&w.gate_inp.data, x, &mut raw_scores, n_experts, d);
    // softmax with max-shift for numerical stability
    let max_r = raw_scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut probs = vec![0.0_f32; n_experts];
    let mut z = 0.0_f32;
    for e in 0..n_experts {
        probs[e] = (raw_scores[e] - max_r).exp();
        z += probs[e];
    }
    for p in &mut probs {
        *p /= z;
    }

    // ---- 2. Top-K + renormalize ----
    let mut idx_p: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
    // Stable partial sort: keep top-K by probability descending.
    idx_p.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    idx_p.truncate(k);
    let sum_topk: f32 = idx_p.iter().map(|(_, p)| *p).sum();
    let inv_sum = if sum_topk > 0.0 { 1.0 / sum_topk } else { 0.0 };
    for (_, p) in idx_p.iter_mut() {
        *p *= inv_sum;
    }

    // ---- 3. Routed experts : Σ_e w_e · expert_swiglu(e, x) ----
    let mut y_routed = vec![0.0_f32; d];
    let mut gate_buf = vec![0.0_f32; ef];
    let mut up_buf = vec![0.0_f32; ef];
    let mut hidden_buf = vec![0.0_f32; ef];
    let mut down_buf = vec![0.0_f32; d];
    for &(e, w_e) in &idx_p {
        // gate = silu(ffn_gate_exps[e] @ x)
        gemv(&w.gate_exps[e].data, x, &mut gate_buf, ef, d);
        silu(&mut gate_buf);
        // up = ffn_up_exps[e] @ x
        gemv(&w.up_exps[e].data, x, &mut up_buf, ef, d);
        // hidden = gate * up
        for i in 0..ef {
            hidden_buf[i] = gate_buf[i] * up_buf[i];
        }
        // out = ffn_down_exps[e] @ hidden
        for v in down_buf.iter_mut() {
            *v = 0.0;
        }
        gemv(&w.down_exps[e].data, &hidden_buf, &mut down_buf, d, ef);
        // accumulate w_e * out into y_routed
        for i in 0..d {
            y_routed[i] += w_e * down_buf[i];
        }
    }

    // ---- 4. Shared expert (parallel) ----
    // shexp_w = sigmoid(gate_inp_shexp · x)  (scalar)
    let dot: f32 = w
        .gate_inp_shexp
        .iter()
        .zip(x.iter())
        .map(|(a, b)| a * b)
        .sum();
    let shexp_w = 1.0 / (1.0 + (-dot).exp());
    // gate_shexp_out = silu(gate_shexp @ x)
    gemv(&w.gate_shexp.data, x, &mut gate_buf, ef, d);
    silu(&mut gate_buf);
    // up_shexp_out = up_shexp @ x
    gemv(&w.up_shexp.data, x, &mut up_buf, ef, d);
    for i in 0..ef {
        hidden_buf[i] = gate_buf[i] * up_buf[i];
    }
    let mut y_shexp = vec![0.0_f32; d];
    gemv(&w.down_shexp.data, &hidden_buf, &mut y_shexp, d, ef);
    for v in y_shexp.iter_mut() {
        *v *= shexp_w;
    }

    // ---- 5. y = y_routed + y_shexp ----
    for i in 0..d {
        y_routed[i] += y_shexp[i];
    }
    y_routed
}

#[cfg(test)]
mod moe_tests {
    use super::*;
    use crate::qwen35::Qwen35Variant;

    fn make_cfg(d: usize, ef: usize, n_e: usize, k: usize) -> Qwen35Config {
        Qwen35Config {
            variant: Qwen35Variant::Moe,
            n_layers: 1,
            d,
            f: 0,
            n_q_heads: 1,
            n_kv_heads: 1,
            rope_dim: 1,
            vocab: 1,
            max_context: 1,
            rms_eps: 1e-6,
            rope_base: 10000.0,
            ssm_inner: 1,
            ssm_state: 1,
            ssm_dt_rank: 1,
            ssm_groups: 1,
            ssm_conv_kernel: 1,
            n_experts: n_e,
            n_experts_used: k,
            expert_f: ef,
            attn_head_dim: 1,
            attention_indices: vec![],
            ssm_indices: vec![0],
        }
    }

    fn dense(rows: usize, cols: usize, fill: f32) -> DenseWeight {
        DenseWeight {
            data: vec![fill; rows * cols],
            shape: vec![rows, cols],
        }
    }

    /// Smoke test : MoE forward should return non-zero output for non-zero
    /// input + non-zero weights, and produce different routing for
    /// different inputs.
    #[test]
    fn ffn_moe_forward_basic_smoke() {
        let d = 4;
        let ef = 6;
        let n_e = 3;
        let k = 2;
        let cfg = make_cfg(d, ef, n_e, k);

        // Distinct gate_inp rows so different inputs route to different experts.
        let mut gate_inp = dense(n_e, d, 0.0);
        // expert 0 prefers x[0] high, expert 1 prefers x[1] high, expert 2 prefers x[2].
        // Sparse identity-ish: expert e prefers x[e] high.
        gate_inp.data[0] = 2.0;
        gate_inp.data[d + 1] = 2.0;
        gate_inp.data[2 * d + 2] = 2.0;

        let w = MoeFfnWeights {
            gate_inp,
            gate_exps: (0..n_e)
                .map(|e| dense(ef, d, 0.1 * (e + 1) as f32))
                .collect(),
            up_exps: (0..n_e)
                .map(|e| dense(ef, d, 0.05 * (e + 1) as f32))
                .collect(),
            down_exps: (0..n_e)
                .map(|e| dense(d, ef, 0.07 * (e + 1) as f32))
                .collect(),
            gate_inp_shexp: vec![0.0; d],
            gate_shexp: dense(ef, d, 0.1),
            up_shexp: dense(ef, d, 0.05),
            down_shexp: dense(d, ef, 0.07),
        };

        let x_a = vec![1.0_f32, 0.0, 0.0, 0.0];
        let x_b = vec![0.0_f32, 0.0, 1.0, 0.0];
        let y_a = ffn_moe_forward(&w, &x_a, &cfg);
        let y_b = ffn_moe_forward(&w, &x_b, &cfg);
        assert_eq!(y_a.len(), d);
        assert_eq!(y_b.len(), d);
        // Different inputs should produce different outputs (different
        // experts get selected → different effective FFN).
        let same: bool = y_a.iter().zip(&y_b).all(|(a, b)| (a - b).abs() < 1e-6);
        assert!(
            !same,
            "MoE forward should route differently for different inputs ; y_a={:?}, y_b={:?}",
            y_a, y_b
        );
        // Output should be non-zero for non-zero input + non-zero weights.
        assert!(
            y_a.iter().any(|v| v.abs() > 1e-6),
            "y_a should be non-zero, got {:?}",
            y_a
        );
    }
}
