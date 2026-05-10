//! Qwen3.5 / Qwen3.6 **hybrid SSM + Attention** architecture support.
//!
//! Targets two GGUF flavours:
//!
//! - `qwen35`     — dense FFN. Used by Qwen3.5-27B, Qwen3.6-27B.
//! - `qwen35moe`  — MoE FFN with shared expert. Used by Qwen3.6-35B-A3B.
//!
//! Both share the same hybrid layer pattern: most blocks are **Mamba-style
//! SSM blocks**, every 4th block (indices 3, 7, 11, …) is a standard
//! **multi-head attention block**. The FFN sub-block is shared (same input,
//! same output dim) but its internal computation differs between dense and
//! MoE variants.
//!
//! ## Per-block structure
//!
//! ### SSM block (Mamba-2 flavoured)
//!
//! ```text
//!   x_in -> attn_norm -> [residual = x_in]
//!         |
//!         | attn_qkv  : [d, 2*d_inner]    -> xz   (split into x and z)
//!         | attn_gate : [d, d_inner_g]    -> g
//!         |
//!         |   x = silu(conv1d(x))         (kernel=4, depth-wise)
//!         |   dt = softplus(alpha @ x + dt_bias)
//!         |   B  = beta  @ x              (per-token state inputs)
//!         |   A  = -exp(ssm_a)            (state transition diag)
//!         |   state := exp(A * dt) * state + dt * B * x
//!         |   y = state^T @ C              (read-out)
//!         |   y = ssm_norm(y) * silu(g)
//!         |   y = ssm_out @ y             ([d_inner_g, d])
//!         |
//!   x_pre_ffn = x_in + y
//!         |
//!   x_ffn = post_attention_norm(x_pre_ffn)
//!         | -> ffn_block (dense or MoE)
//!   x_out = x_pre_ffn + ffn(x_ffn)
//! ```
//!
//! ### Attention block (vanilla Qwen3-style)
//!
//! Same residual pattern; replaces the SSM mixer with standard MHA:
//!
//! ```text
//!   h = attn_norm(x_in)
//!   q = attn_q @ h ; k = attn_k @ h ; v = attn_v @ h
//!   q = attn_q_norm(q) ; k = attn_k_norm(k)
//!   q,k = rope(q, k, pos)
//!   y = mha(q, k, v)
//!   y = attn_output @ y
//!   x_pre_ffn = x_in + y
//!   x_out = x_pre_ffn + ffn(post_attention_norm(x_pre_ffn))
//! ```
//!
//! ### FFN sub-block — `qwen35` (dense)
//!
//! Standard SwiGLU: `ffn_down @ (silu(ffn_gate @ h) * (ffn_up @ h))`.
//!
//! ### FFN sub-block — `qwen35moe`
//!
//! Top-K MoE with a parallel shared expert:
//!
//! ```text
//!   y_routed = sum_{e in topk(softmax(ffn_gate_inp @ h))}
//!                w_e * (ffn_down_exps[e] @ (silu(ffn_gate_exps[e] @ h)
//!                                          * ffn_up_exps[e] @ h))
//!   y_shared = ffn_down_shexp @ (silu(ffn_gate_shexp @ h)
//!                              * ffn_up_shexp @ h)
//!             * sigmoid(ffn_gate_inp_shexp @ h)
//!   y = y_routed + y_shared
//! ```
//!
//! ## What this module provides today (foundation pass)
//!
//! - [`Qwen35Config`] — architecture-level parameters parsed from GGUF
//!   metadata. Detects dense vs MoE variant and exposes all hybrid hyperparams.
//! - [`LayerKind`] — discriminates SSM vs attention blocks.
//! - [`layer_kind_for_index`] — derives the per-index layer kind from the
//!   "every 4th is attention" pattern (3, 7, 11, …).
//! - [`describe_model`] — pretty-prints a parsed model summary, useful
//!   during the bring-up phase.
//!
//! Forward computation (CPU reference + Metal kernels) lands in subsequent
//! commits — see the implementation plan in the example
//! `examples/qwen35_inspect.rs`.

use std::collections::BTreeSet;
use std::path::Path;

use rustorch_gguf::{GgmlType, GgufFile};

/// Qwen architecture variant. Despite the module name (`qwen35`), this
/// also covers the legacy `qwen2` and `qwen3` (pure transformer)
/// families so the loader and forward pipeline can dispatch all four
/// Qwen flavours (Qwen2.5-7B / Qwen3-14B / Qwen3.6-27B / Qwen3.6-35B-A3B)
/// from a single binary.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Qwen35Variant {
    /// `qwen2` — legacy Qwen2 pure transformer (Qwen2 / Qwen2.5 family).
    /// Differs from `qwen3` by : (1) Q/K/V projections carry a bias
    /// vector ; (2) NO per-head Q-norm / K-norm ; (3) NO sigmoid output
    /// gate on attention. FFN pre-norm uses `ffn_norm` (same as qwen3),
    /// dense SwiGLU FFN.
    Qwen2PureTransformer,
    /// `qwen3` — pure transformer (no SSM), single-width `attn_q` (no
    /// Q+gate combine), dense FFN, `ffn_norm` for the FFN pre-norm.
    /// Used by Qwen3-14B and other Qwen3 family checkpoints.
    Qwen3PureTransformer,
    /// `qwen35` — hybrid SSM+attention, Qwen3Next attention (Q+gate
    /// combined, sigmoid gate on attn output), dense FFN,
    /// `post_attention_norm` for the FFN pre-norm. Used by Qwen3.5-27B,
    /// Qwen3.6-27B.
    Dense,
    /// `qwen35moe` — same as `Dense` but the FFN is MoE with a parallel
    /// shared expert. Used by Qwen3.6-35B-A3B.
    Moe,
}

impl Qwen35Variant {
    /// String key used in GGUF metadata under `<arch>.<param>` paths.
    pub fn arch_str(self) -> &'static str {
        match self {
            Qwen35Variant::Qwen2PureTransformer => "qwen2",
            Qwen35Variant::Qwen3PureTransformer => "qwen3",
            Qwen35Variant::Dense => "qwen35",
            Qwen35Variant::Moe => "qwen35moe",
        }
    }

    /// True if attention's `wq` carries Q+gate combined (Qwen3Next style).
    /// False for legacy Qwen2/Qwen3 where `wq` outputs only Q.
    pub fn has_q_gate(self) -> bool {
        matches!(self, Qwen35Variant::Dense | Qwen35Variant::Moe)
    }

    /// True if any SSM (gated delta net) layers are present. For pure
    /// transformer variants this is always false.
    pub fn has_ssm(self) -> bool {
        matches!(self, Qwen35Variant::Dense | Qwen35Variant::Moe)
    }

    /// Tensor name used for the FFN pre-norm. `qwen2` and `qwen3` call it
    /// `ffn_norm`; `qwen35*` calls it `post_attention_norm`.
    pub fn ffn_pre_norm_name(self) -> &'static str {
        match self {
            Qwen35Variant::Qwen2PureTransformer | Qwen35Variant::Qwen3PureTransformer => "ffn_norm",
            _ => "post_attention_norm",
        }
    }

    /// True if the attention projections (Q, K, V) carry a bias vector.
    /// True only for Qwen2 ; Qwen3 / Qwen3.5 / Qwen3.6 are all bias-free.
    pub fn attn_has_qkv_bias(self) -> bool {
        matches!(self, Qwen35Variant::Qwen2PureTransformer)
    }

    /// True if per-head Q-norm / K-norm RMSNorm is applied after the
    /// Q / K projections. False for Qwen2, true for everything else.
    pub fn attn_has_qk_norm(self) -> bool {
        !matches!(self, Qwen35Variant::Qwen2PureTransformer)
    }

    /// True if the attention output carries a sigmoid output gate
    /// (Qwen3Next style). True for Qwen3 / Qwen3.5 / Qwen3.6 (the gate
    /// is the second half of `attn_q` 's `2*q_dim` output). False for
    /// Qwen2 (no gate at all — `attn_q` width is exactly `q_dim`).
    pub fn attn_has_output_gate(self) -> bool {
        matches!(
            self,
            Qwen35Variant::Qwen3PureTransformer | Qwen35Variant::Dense | Qwen35Variant::Moe
        )
    }

    /// Pure-transformer variants — no SSM, every layer is attention.
    /// Used to pick the `decode_step_tree_pure_transformer` fast path
    /// (introduced in T246.7 P1.4a, extended to qwen2 in P1.5).
    pub fn is_pure_transformer(self) -> bool {
        matches!(
            self,
            Qwen35Variant::Qwen2PureTransformer | Qwen35Variant::Qwen3PureTransformer
        )
    }
}

/// Per-block role discriminator.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LayerKind {
    /// Standard multi-head attention with QK-norm + RoPE.
    Attention,
    /// Mamba-2 selective state-space block.
    Ssm,
}

/// Qwen3.5/3.6 architecture configuration.
///
/// All fields are derived from GGUF metadata (`<arch>.*` keys). The
/// `n_layers`, `d`, `vocab` etc. mirror the dense Qwen3 config; the SSM
/// fields and (optionally) MoE fields capture the hybrid bits.
#[derive(Clone, Debug)]
pub struct Qwen35Config {
    /// Variant — Dense or Moe.
    pub variant: Qwen35Variant,
    /// Total number of blocks (`block_count`). 64 for 27B, 40 for 35B-A3B.
    pub n_layers: usize,
    /// Residual stream dim (`embedding_length`). 5120 for 27B, 2048 for 35B-A3B.
    pub d: usize,
    /// FFN hidden dim (`feed_forward_length`). Only used in dense variant.
    pub f: usize,
    /// Number of attention query heads (`attention.head_count`).
    pub n_q_heads: usize,
    /// Number of K/V heads (`attention.head_count_kv`). GQA factor = n_q_heads / n_kv_heads.
    pub n_kv_heads: usize,
    /// RoPE rotated subspace dim (`rope.dimension_count`). Often head_dim/2 or full head_dim.
    pub rope_dim: usize,
    /// Vocabulary size.
    pub vocab: usize,
    /// Maximum context length supported by the trained checkpoint.
    pub max_context: usize,
    /// RMS norm epsilon.
    pub rms_eps: f32,
    /// RoPE base frequency (theta).
    pub rope_base: f32,

    // ---- SSM hyperparams ----
    /// `ssm.inner_size` — width of the SSM "wide" channel (d_inner).
    pub ssm_inner: usize,
    /// `ssm.state_size` — per-group SSM hidden state dim (d_state).
    pub ssm_state: usize,
    /// `ssm.time_step_rank` — low-rank dt projection dim.
    pub ssm_dt_rank: usize,
    /// `ssm.group_count` — number of SSM channel groups.
    pub ssm_groups: usize,
    /// `ssm.conv_kernel` — 1-D causal convolution kernel size (typ. 4).
    pub ssm_conv_kernel: usize,

    // ---- MoE hyperparams (only Variant::Moe) ----
    /// `expert_count` — total number of experts. 0 for dense variant.
    pub n_experts: usize,
    /// `expert_used_count` — top-K experts active per token (e.g. 8).
    pub n_experts_used: usize,
    /// `expert_feed_forward_length` — per-expert FFN hidden dim. 0 for dense.
    pub expert_f: usize,

    /// Per-head attention dim — not always `d / n_q_heads`. For Qwen3.6-27B
    /// the actual attention projection width comes from `attn_q.weight`
    /// shape: `attn_q` is [d, n_q_heads * attn_head_dim]. We read it from
    /// the first attention layer's tensor at parse time.
    pub attn_head_dim: usize,

    // ---- Layout-derived helpers ----
    /// Indices of attention layers, sorted ascending. Detected from the
    /// presence of `attn_q.weight` per block.
    pub attention_indices: Vec<usize>,
    /// Indices of SSM layers, sorted ascending. Complement of `attention_indices`.
    pub ssm_indices: Vec<usize>,
}

impl Qwen35Config {
    /// Returns the kind of layer at index `li`. O(log n) lookup.
    pub fn layer_kind(&self, li: usize) -> LayerKind {
        if self.attention_indices.binary_search(&li).is_ok() {
            LayerKind::Attention
        } else {
            LayerKind::Ssm
        }
    }

    /// Per-head attention dimension (only meaningful for attention
    /// layers). For Qwen3.6 27B (n_q_heads=24, d=5120) this is NOT
    /// `d / n_q_heads` — the projection has its own width that's read
    /// from the actual `attn_q.weight` tensor at parse time. For models
    /// without an explicit attention layer (theoretically only-SSM), this
    /// returns 0.
    pub fn head_dim(&self) -> usize {
        self.attn_head_dim
    }

    /// Per-group SSM head dim — the `d_inner` channels are split across
    /// `ssm_groups` groups. Each group carries an independent state matrix.
    /// Returns 0 when the architecture has no SSM block.
    pub fn ssm_group_dim(&self) -> usize {
        if self.ssm_groups == 0 {
            0
        } else {
            self.ssm_inner / self.ssm_groups
        }
    }

    /// Bytes used per layer for SSM state at decode time:
    /// `ssm_inner * ssm_state * 4`. Per-token persistent.
    pub fn ssm_state_bytes_per_layer(&self) -> usize {
        self.ssm_inner * self.ssm_state * 4
    }

    /// Bytes used per layer for the 1-D conv ring buffer:
    /// `(conv_kernel - 1) * 2 * ssm_inner * 4`. The factor `2` is for the
    /// `attn_qkv` doubled-input layout (the conv operates on the raw qkv
    /// projection, not on a single-d_inner stream).
    pub fn ssm_conv_state_bytes_per_layer(&self) -> usize {
        (self.ssm_conv_kernel - 1) * 2 * self.ssm_inner * 4
    }
}

fn meta_u32(file: &GgufFile, arch: &str, key: &str) -> Option<u32> {
    file.metadata()
        .get(&format!("{arch}.{key}"))
        .and_then(|v| v.as_u32())
}

fn meta_f32(file: &GgufFile, arch: &str, key: &str) -> Option<f32> {
    file.metadata()
        .get(&format!("{arch}.{key}"))
        .and_then(|v| v.as_f32())
}

/// Derive the per-index layer kind from the "every 4th block is attention"
/// rule. Used as a fallback when `attn_q` tensors are not present at parse
/// time. The rule is empirically observed in Qwen3.5-27B and Qwen3.6-{27B, 35B-A3B}:
/// indices `3, 7, 11, …` are attention; the rest are SSM.
pub fn layer_kind_for_index(li: usize) -> LayerKind {
    if (li + 1) % 4 == 0 {
        LayerKind::Attention
    } else {
        LayerKind::Ssm
    }
}

/// Errors produced while parsing a Qwen3.5/3.6 GGUF.
#[derive(Debug)]
pub enum Qwen35LoadError {
    /// I/O or parser-level failure surfaced from the GGUF reader.
    Gguf(String),
    /// The architecture string in metadata is not one of the supported variants.
    UnsupportedArch(String),
    /// A required GGUF metadata key is missing.
    MissingMeta(String),
    /// A required tensor for the parsed architecture is missing.
    MissingTensor(String),
}

impl std::fmt::Display for Qwen35LoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Qwen35LoadError::Gguf(s) => write!(f, "gguf read: {s}"),
            Qwen35LoadError::UnsupportedArch(s) => write!(f, "unsupported architecture: {s}"),
            Qwen35LoadError::MissingMeta(s) => write!(f, "missing metadata key: {s}"),
            Qwen35LoadError::MissingTensor(s) => write!(f, "missing tensor: {s}"),
        }
    }
}

impl std::error::Error for Qwen35LoadError {}

/// Parse a Qwen3.5/3.6 GGUF file's metadata + tensor inventory into a
/// [`Qwen35Config`]. Does NOT load the weight bytes — that comes during
/// the actual forward bring-up.
pub fn parse_config(path: &Path) -> Result<Qwen35Config, Qwen35LoadError> {
    let file = GgufFile::open(path).map_err(|e| Qwen35LoadError::Gguf(format!("{e:?}")))?;
    let arch_str = file
        .metadata()
        .get("general.architecture")
        .and_then(|v| v.as_str())
        .ok_or_else(|| Qwen35LoadError::MissingMeta("general.architecture".to_string()))?
        .to_string();
    let variant = match arch_str.as_str() {
        "qwen2" => Qwen35Variant::Qwen2PureTransformer,
        "qwen3" => Qwen35Variant::Qwen3PureTransformer,
        "qwen35" => Qwen35Variant::Dense,
        "qwen35moe" => Qwen35Variant::Moe,
        other => return Err(Qwen35LoadError::UnsupportedArch(other.to_string())),
    };
    let arch = variant.arch_str();
    let req_u32 = |key: &str| -> Result<usize, Qwen35LoadError> {
        meta_u32(&file, arch, key)
            .map(|x| x as usize)
            .ok_or_else(|| Qwen35LoadError::MissingMeta(format!("{arch}.{key}")))
    };

    let n_layers = req_u32("block_count")?;
    let d = req_u32("embedding_length")?;
    // feed_forward_length is present in pure transformer + dense variant;
    // MoE uses expert_feed_forward_length.
    let f = if variant != Qwen35Variant::Moe {
        req_u32("feed_forward_length")?
    } else {
        0
    };
    let n_q_heads = req_u32("attention.head_count")?;
    let n_kv_heads = req_u32("attention.head_count_kv")?;
    // qwen3 has no `rope.dimension_count` metadata key; fall back to full
    // head_dim (we'll resolve the correct value at attn_head_dim time).
    let rope_dim = meta_u32(&file, arch, "rope.dimension_count")
        .map(|x| x as usize)
        .unwrap_or(0);
    let max_context = req_u32("context_length")?;
    let rms_eps = meta_f32(&file, arch, "attention.layer_norm_rms_epsilon").ok_or_else(|| {
        Qwen35LoadError::MissingMeta(format!("{arch}.attention.layer_norm_rms_epsilon"))
    })?;
    let rope_base = meta_f32(&file, arch, "rope.freq_base").unwrap_or(10_000.0);

    // SSM hyperparams only present in the hybrid variants.
    let (ssm_inner, ssm_state, ssm_dt_rank, ssm_groups, ssm_conv_kernel) = if variant.has_ssm() {
        (
            req_u32("ssm.inner_size")?,
            req_u32("ssm.state_size")?,
            req_u32("ssm.time_step_rank")?,
            req_u32("ssm.group_count")?,
            req_u32("ssm.conv_kernel")?,
        )
    } else {
        (0, 0, 0, 0, 0)
    };

    let n_experts = if variant == Qwen35Variant::Moe {
        req_u32("expert_count")?
    } else {
        0
    };
    let n_experts_used = if variant == Qwen35Variant::Moe {
        req_u32("expert_used_count")?
    } else {
        0
    };
    let expert_f = if variant == Qwen35Variant::Moe {
        req_u32("expert_feed_forward_length")?
    } else {
        0
    };

    // Vocab — token_embd.weight shape is [d, vocab] in GGUF tile layout.
    let vocab = file
        .tensor("token_embd.weight")
        .map(|t| t.shape.iter().copied().max().unwrap_or(0) as usize)
        .ok_or_else(|| Qwen35LoadError::MissingTensor("token_embd.weight".to_string()))?;

    // Discover attention vs SSM by inspecting which blocks carry attn_q.
    let mut attn_set: BTreeSet<usize> = BTreeSet::new();
    let mut ssm_set: BTreeSet<usize> = BTreeSet::new();
    for t in file.tensors() {
        if let Some(rest) = t.name.strip_prefix("blk.") {
            if let Some(dot) = rest.find('.') {
                if let Ok(idx) = rest[..dot].parse::<usize>() {
                    let suffix = &rest[dot + 1..];
                    if suffix == "attn_q.weight" {
                        attn_set.insert(idx);
                    } else if suffix == "ssm_a" {
                        ssm_set.insert(idx);
                    }
                }
            }
        }
    }
    // Sanity: every layer must be either attn or ssm and not both.
    for li in 0..n_layers {
        let in_attn = attn_set.contains(&li);
        let in_ssm = ssm_set.contains(&li);
        if in_attn && in_ssm {
            return Err(Qwen35LoadError::Gguf(format!(
                "block {li} carries both attn_q.weight and ssm_a — unsupported"
            )));
        }
        if !in_attn && !in_ssm {
            // Fall back to the empirical "every 4th = attention" rule.
            match layer_kind_for_index(li) {
                LayerKind::Attention => {
                    attn_set.insert(li);
                },
                LayerKind::Ssm => {
                    ssm_set.insert(li);
                },
            }
        }
    }
    let attention_indices: Vec<usize> = attn_set.into_iter().collect();
    let ssm_indices: Vec<usize> = ssm_set.into_iter().collect();

    // Read `attn_head_dim` from the first attention layer's attn_q.weight
    // shape. GGUF stores 2D weights as [in, out].
    //   qwen3 (pure transformer): out = n_q_heads * head_dim
    //   qwen35 / qwen35moe       : out = n_q_heads * head_dim * 2 (Q+gate)
    let attn_head_dim = if let Some(&first_attn_li) = attention_indices.first() {
        let q_name = format!("blk.{first_attn_li}.attn_q.weight");
        match file.tensor(&q_name) {
            Some(t) => {
                let out_dim =
                    t.shape.get(1).copied().ok_or_else(|| {
                        Qwen35LoadError::Gguf(format!("{q_name}: shape too short"))
                    })? as usize;
                let factor = if variant.has_q_gate() { 2 } else { 1 };
                let denom = n_q_heads * factor;
                if out_dim % denom != 0 {
                    return Err(Qwen35LoadError::Gguf(format!(
                        "{q_name} out_dim {out_dim} not divisible by n_q_heads*{factor} = {denom}"
                    )));
                }
                out_dim / denom
            },
            None => {
                return Err(Qwen35LoadError::MissingTensor(q_name));
            },
        }
    } else {
        0
    };

    // qwen3 has no rope.dimension_count in metadata — default to full head_dim.
    let rope_dim = if rope_dim == 0 {
        attn_head_dim
    } else {
        rope_dim
    };

    Ok(Qwen35Config {
        variant,
        n_layers,
        d,
        f,
        n_q_heads,
        n_kv_heads,
        rope_dim,
        vocab,
        max_context,
        rms_eps,
        rope_base,
        ssm_inner,
        ssm_state,
        ssm_dt_rank,
        ssm_groups,
        ssm_conv_kernel,
        n_experts,
        n_experts_used,
        expert_f,
        attn_head_dim,
        attention_indices,
        ssm_indices,
    })
}

/// Pretty-print a parsed Qwen3.5/3.6 model summary. Useful as a smoke
/// test during bring-up.
pub fn describe_model(cfg: &Qwen35Config) -> String {
    let mut s = String::new();
    use std::fmt::Write;
    writeln!(s, "Qwen3.5/3.6 hybrid model — variant {:?}", cfg.variant).ok();
    writeln!(s, "  n_layers     = {}", cfg.n_layers).ok();
    writeln!(
        s,
        "    attention  = {} layers @ {:?}",
        cfg.attention_indices.len(),
        cfg.attention_indices
    )
    .ok();
    writeln!(
        s,
        "    ssm        = {} layers @ {:?}",
        cfg.ssm_indices.len(),
        &cfg.ssm_indices[..cfg.ssm_indices.len().min(8)]
    )
    .ok();
    writeln!(s, "  d            = {}", cfg.d).ok();
    writeln!(s, "  vocab        = {}", cfg.vocab).ok();
    writeln!(s, "  max_context  = {}", cfg.max_context).ok();
    writeln!(s, "  rms_eps      = {}", cfg.rms_eps).ok();
    writeln!(s, "  rope_base    = {}", cfg.rope_base).ok();
    writeln!(s, "  rope_dim     = {}", cfg.rope_dim).ok();
    writeln!(s, "  attention").ok();
    writeln!(s, "    n_q_heads  = {}", cfg.n_q_heads).ok();
    writeln!(s, "    n_kv_heads = {}", cfg.n_kv_heads).ok();
    writeln!(s, "    head_dim   = {}", cfg.head_dim()).ok();
    writeln!(s, "  ssm").ok();
    writeln!(s, "    inner      = {}", cfg.ssm_inner).ok();
    writeln!(s, "    state      = {}", cfg.ssm_state).ok();
    writeln!(s, "    dt_rank    = {}", cfg.ssm_dt_rank).ok();
    writeln!(s, "    groups     = {}", cfg.ssm_groups).ok();
    writeln!(s, "    group_dim  = {}", cfg.ssm_group_dim()).ok();
    writeln!(s, "    conv_kern  = {}", cfg.ssm_conv_kernel).ok();
    writeln!(
        s,
        "    state bytes/layer = {} ({:.1} MB)",
        cfg.ssm_state_bytes_per_layer(),
        cfg.ssm_state_bytes_per_layer() as f64 / 1024.0 / 1024.0
    )
    .ok();
    if cfg.variant == Qwen35Variant::Moe {
        writeln!(s, "  moe").ok();
        writeln!(s, "    n_experts        = {}", cfg.n_experts).ok();
        writeln!(s, "    n_experts_used   = {}", cfg.n_experts_used).ok();
        writeln!(s, "    expert_f         = {}", cfg.expert_f).ok();
    } else {
        writeln!(s, "  ffn (dense)").ok();
        writeln!(s, "    f                = {}", cfg.f).ok();
    }
    s
}

/// Map of expected GGUF tensor names per layer kind. Used for validation
/// and to drive the loader. The `Q4_K`/`Q6_K`/etc. dtype hints are not
/// enforced here — the loader honours whatever the GGUF says.
pub fn expected_tensor_names(li: usize, kind: LayerKind, variant: Qwen35Variant) -> Vec<String> {
    let mut v = Vec::new();
    let p = |s: &str| format!("blk.{li}.{s}");
    // Pre-attention norm — same name on all variants.
    v.push(p("attn_norm.weight"));
    // FFN pre-norm: qwen3 uses `ffn_norm`, qwen35* uses `post_attention_norm`.
    v.push(p(&format!("{}.weight", variant.ffn_pre_norm_name())));
    match kind {
        LayerKind::Attention => {
            v.extend([
                p("attn_q.weight"),
                p("attn_k.weight"),
                p("attn_v.weight"),
                p("attn_output.weight"),
            ]);
            if variant.attn_has_qk_norm() {
                v.push(p("attn_q_norm.weight"));
                v.push(p("attn_k_norm.weight"));
            }
            if variant.attn_has_qkv_bias() {
                v.push(p("attn_q.bias"));
                v.push(p("attn_k.bias"));
                v.push(p("attn_v.bias"));
            }
        },
        LayerKind::Ssm => {
            v.extend([
                p("attn_qkv.weight"),
                p("attn_gate.weight"),
                p("ssm_a"),
                p("ssm_alpha.weight"),
                p("ssm_beta.weight"),
                p("ssm_conv1d.weight"),
                p("ssm_dt.bias"),
                p("ssm_norm.weight"),
                p("ssm_out.weight"),
            ]);
        },
    }
    match variant {
        Qwen35Variant::Qwen2PureTransformer
        | Qwen35Variant::Qwen3PureTransformer
        | Qwen35Variant::Dense => {
            v.extend([
                p("ffn_gate.weight"),
                p("ffn_up.weight"),
                p("ffn_down.weight"),
            ]);
        },
        Qwen35Variant::Moe => {
            v.extend([
                p("ffn_gate_inp.weight"),
                p("ffn_gate_exps.weight"),
                p("ffn_up_exps.weight"),
                p("ffn_down_exps.weight"),
                p("ffn_gate_inp_shexp.weight"),
                p("ffn_gate_shexp.weight"),
                p("ffn_up_shexp.weight"),
                p("ffn_down_shexp.weight"),
            ]);
        },
    }
    v
}

/// Validate that every expected tensor for the parsed config exists in
/// the GGUF. Returns the list of missing tensor names (empty if all
/// expected tensors are present).
pub fn missing_tensors(path: &Path, cfg: &Qwen35Config) -> Result<Vec<String>, Qwen35LoadError> {
    let file = GgufFile::open(path).map_err(|e| Qwen35LoadError::Gguf(format!("{e:?}")))?;
    let mut missing = Vec::new();
    // Non-block tensors.
    for n in ["token_embd.weight", "output_norm.weight", "output.weight"] {
        if file.tensor(n).is_none() {
            missing.push(n.to_string());
        }
    }
    // Per-block tensors.
    for li in 0..cfg.n_layers {
        let kind = cfg.layer_kind(li);
        for name in expected_tensor_names(li, kind, cfg.variant) {
            if file.tensor(&name).is_none() {
                missing.push(name);
            }
        }
    }
    Ok(missing)
}

/// Lightweight tensor inventory entry — name + shape + dtype, used to
/// echo the model's structure without loading bytes.
#[derive(Clone, Debug)]
pub struct TensorRef {
    /// GGUF tensor name (e.g. `blk.0.ssm_a`).
    pub name: String,
    /// Logical shape as stored in the GGUF.
    pub shape: Vec<u64>,
    /// GGML quantisation type.
    pub dtype: GgmlType,
}

/// Return the inventory of every tensor expected by the parsed config,
/// resolved against the actual GGUF file. Tensors absent from the file are
/// reported in the second return value.
pub fn full_inventory(
    path: &Path,
    cfg: &Qwen35Config,
) -> Result<(Vec<TensorRef>, Vec<String>), Qwen35LoadError> {
    let file = GgufFile::open(path).map_err(|e| Qwen35LoadError::Gguf(format!("{e:?}")))?;
    let mut found = Vec::new();
    let mut missing = Vec::new();

    let mut visit = |name: &str| {
        if let Some(t) = file.tensor(name) {
            found.push(TensorRef {
                name: t.name.clone(),
                shape: t.shape.clone(),
                dtype: t.dtype,
            });
        } else {
            missing.push(name.to_string());
        }
    };

    for n in ["token_embd.weight", "output_norm.weight", "output.weight"] {
        visit(n);
    }
    for li in 0..cfg.n_layers {
        let kind = cfg.layer_kind(li);
        for name in expected_tensor_names(li, kind, cfg.variant) {
            visit(&name);
        }
    }
    Ok((found, missing))
}
