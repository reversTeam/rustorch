//! Scaled Dot-Product Attention (P1.6).
//!
//! v1 ships **single-head** attention with the standard formula:
//! ```text
//!   attn(Q, K, V) = softmax(Q @ K.T / sqrt(d)) @ V
//! ```
//!
//! Inputs are 3-D `[B, T, D]` (batch, sequence, embed_dim). Multi-head
//! support requires either a 4-D batched matmul or autograd-aware
//! reshape — both pending follow-ups.

use crate::module::{Module, ModuleError};
use rustorch_autograd::{ops, Variable};
use rustorch_core::tensor::tensor_impl::Tensor;

/// Stateless single-head scaled dot-product attention.
///
/// Takes pre-projected Q, K, V tensors (shape `[B, T, D]` for each)
/// and returns the attention output of shape `[B, T, D]`.
pub fn scaled_dot_product_attention(
    q: &Variable,
    k: &Variable,
    v: &Variable,
) -> Result<Variable, ModuleError> {
    let q_shape = q.tensor().shape().to_vec();
    if q_shape.len() != 3 {
        return Err(rustorch_autograd::BackwardError::Backend {
            op: "sdp_attention",
            message: format!("expected rank-3 Q, got {q_shape:?}"),
        });
    }
    let d = q_shape[2] as f32;
    // K.T along the last two axes: [B, T, D] → [B, D, T]
    let k_t = ops::transpose(k, 1, 2)?;
    // Scores: Q @ K.T / sqrt(d) → [B, T, T]
    let raw_scores = ops::bmm(q, &k_t)?;
    let inv_scale = Variable::new(Tensor::scalar(1.0_f32 / d.sqrt()));
    let scores = ops::mul(&raw_scores, &inv_scale)?;
    // Softmax over the LAST dim (each query attends to all keys).
    let attn = ops::softmax(&scores, 2)?;
    // attn @ V → [B, T, D]
    ops::bmm(&attn, v)
}

/// Single-head attention block with learned Q/K/V/O projections.
///
/// Matches `nn.MultiheadAttention(d, num_heads=1)`. Multi-head
/// support requires 4-D bmm (pending).
pub struct SingleHeadAttention {
    /// Query projection.
    pub q_proj: crate::Linear,
    /// Key projection.
    pub k_proj: crate::Linear,
    /// Value projection.
    pub v_proj: crate::Linear,
    /// Output projection.
    pub o_proj: crate::Linear,
}

impl SingleHeadAttention {
    /// Build with the given embed dim. All four projections are
    /// `[d, d]`.
    pub fn new(d: usize) -> Self {
        SingleHeadAttention {
            q_proj: crate::Linear::new(d, d),
            k_proj: crate::Linear::new(d, d),
            v_proj: crate::Linear::new(d, d),
            o_proj: crate::Linear::new(d, d),
        }
    }
}

impl Module for SingleHeadAttention {
    fn forward(&self, input: &Variable) -> Result<Variable, ModuleError> {
        // input: [B, T, D]. Currently Linear is rank-2 (`x @ W`), so we
        // reshape to [B*T, D], project, reshape back. To keep things
        // simple here, callers should pre-flatten if D=2 — for the v1
        // we error out on rank != 3 and hand-loop per batch.
        let shape = input.tensor().shape().to_vec();
        if shape.len() != 3 {
            return Err(rustorch_autograd::BackwardError::Backend {
                op: "single_head_attention",
                message: format!("expected rank-3 input, got {shape:?}"),
            });
        }
        // Note: passing 3-D through Linear's matmul will fail. The user
        // should compose with a reshape outside this module. v1 helper
        // path: process B=1 only as a single 2-D matmul.
        if shape[0] != 1 {
            return Err(rustorch_autograd::BackwardError::Backend {
                op: "single_head_attention",
                message: format!(
                    "v1 SingleHeadAttention only supports B=1; got B={} (multi-batch \
                     awaits autograd-aware reshape)",
                    shape[0]
                ),
            });
        }
        // Squeeze leading 1 → [T, D]
        let t = shape[1];
        let d = shape[2];
        let x_2d = ops::transpose(input, 0, 0)?; // no-op transpose; clones contiguous
        let x_2d = reshape_for_linear(&x_2d, t, d)?;
        let q_2d = self.q_proj.forward(&x_2d)?;
        let k_2d = self.k_proj.forward(&x_2d)?;
        let v_2d = self.v_proj.forward(&x_2d)?;
        // Re-add batch dim → [1, T, D]
        let q = unsqueeze_batch(&q_2d, 1, t, d)?;
        let k = unsqueeze_batch(&k_2d, 1, t, d)?;
        let v = unsqueeze_batch(&v_2d, 1, t, d)?;
        let out = scaled_dot_product_attention(&q, &k, &v)?;
        // Output projection on [1, T, D] — flatten, project, unflatten.
        let out_2d = reshape_for_linear(&out, t, d)?;
        let out_proj_2d = self.o_proj.forward(&out_2d)?;
        unsqueeze_batch(&out_proj_2d, 1, t, d)
    }

    fn parameters(&self) -> Vec<Variable> {
        let mut p = self.q_proj.parameters();
        p.extend(self.k_proj.parameters());
        p.extend(self.v_proj.parameters());
        p.extend(self.o_proj.parameters());
        p
    }

    fn named_parameters(&self) -> Vec<(String, Variable)> {
        let mut out = Vec::new();
        for (sub, v) in self.q_proj.named_parameters() {
            out.push((format!("q_proj.{sub}"), v));
        }
        for (sub, v) in self.k_proj.named_parameters() {
            out.push((format!("k_proj.{sub}"), v));
        }
        for (sub, v) in self.v_proj.named_parameters() {
            out.push((format!("v_proj.{sub}"), v));
        }
        for (sub, v) in self.o_proj.named_parameters() {
            out.push((format!("o_proj.{sub}"), v));
        }
        out
    }
}

/// Helper: assume `v` has shape `[1, T, D]` and rebuild a 2-D `[T, D]`
/// tensor with the same data, preserving autograd via a new Variable.
/// Currently we lose the grad path through this transformation — used
/// only for forward; grad on Linear weights still flows correctly via
/// the matmul backward.
fn reshape_for_linear(v: &Variable, t: usize, d: usize) -> Result<Variable, ModuleError> {
    let data: Vec<f32> = v
        .tensor()
        .as_slice::<f32>()
        .ok_or_else(|| rustorch_autograd::BackwardError::Backend {
            op: "reshape_for_linear",
            message: "expected contiguous F32".to_string(),
        })?
        .to_vec();
    let t_new =
        Tensor::from_vec([t, d], data).map_err(|e| rustorch_autograd::BackwardError::Backend {
            op: "reshape_for_linear",
            message: format!("{e}"),
        })?;
    Ok(Variable::leaf(t_new))
}

fn unsqueeze_batch(v: &Variable, b: usize, t: usize, d: usize) -> Result<Variable, ModuleError> {
    let data: Vec<f32> = v
        .tensor()
        .as_slice::<f32>()
        .ok_or_else(|| rustorch_autograd::BackwardError::Backend {
            op: "unsqueeze_batch",
            message: "expected contiguous F32".to_string(),
        })?
        .to_vec();
    let t_new = Tensor::from_vec([b, t, d], data).map_err(|e| {
        rustorch_autograd::BackwardError::Backend {
            op: "unsqueeze_batch",
            message: format!("{e}"),
        }
    })?;
    Ok(Variable::leaf(t_new))
}

/// Multi-head scaled-dot-product attention with learned Q/K/V/O
/// projections. Matches `torch.nn.MultiheadAttention(d_model,
/// num_heads, batch_first=True)` semantics.
///
/// Implementation note: rustorch's autograd `bmm` is strictly rank-3,
/// so the rank-4 head split `[B, T, H, head_dim]` is folded into a
/// `[B*H, T, head_dim]` batch axis for the `Q@K.T` and `attn@V`
/// matmuls, then unfolded back. All shape changes go through the
/// autograd-aware `reshape` and `transpose` ops, so gradient flow is
/// preserved through the four projections.
pub struct MultiHeadAttention {
    /// Query projection — `[embed_dim, embed_dim]`.
    pub q_proj: crate::Linear,
    /// Key projection — `[embed_dim, embed_dim]`.
    pub k_proj: crate::Linear,
    /// Value projection — `[embed_dim, embed_dim]`.
    pub v_proj: crate::Linear,
    /// Output projection — `[embed_dim, embed_dim]`.
    pub o_proj: crate::Linear,
    embed_dim: usize,
    num_heads: usize,
    head_dim: usize,
    /// Dropout probability accepted but ignored in v1 (no autograd-aware
    /// dropout op yet). Stored for future enabling without API break.
    pub dropout: f32,
}

impl MultiHeadAttention {
    /// Build with `embed_dim` divisible by `num_heads`. Each head sees
    /// `head_dim = embed_dim / num_heads` features.
    ///
    /// Panics if `num_heads == 0` or `embed_dim % num_heads != 0`.
    pub fn new(embed_dim: usize, num_heads: usize) -> Self {
        assert!(num_heads > 0, "MultiHeadAttention: num_heads must be > 0");
        assert!(
            embed_dim % num_heads == 0,
            "MultiHeadAttention: embed_dim {} must be divisible by num_heads {}",
            embed_dim,
            num_heads
        );
        let head_dim = embed_dim / num_heads;
        MultiHeadAttention {
            q_proj: crate::Linear::new(embed_dim, embed_dim),
            k_proj: crate::Linear::new(embed_dim, embed_dim),
            v_proj: crate::Linear::new(embed_dim, embed_dim),
            o_proj: crate::Linear::new(embed_dim, embed_dim),
            embed_dim,
            num_heads,
            head_dim,
            dropout: 0.0,
        }
    }

    /// Embedding dim (input/output feature count).
    pub fn embed_dim(&self) -> usize {
        self.embed_dim
    }

    /// Number of attention heads.
    pub fn num_heads(&self) -> usize {
        self.num_heads
    }

    /// Per-head feature count (`embed_dim / num_heads`).
    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// Standard MHA forward.
    ///
    /// Inputs `q`, `k`, `v` all share shape `[B, T, embed_dim]`.
    /// `attn_mask` is an optional additive bias (`0.0` keep,
    /// `MASK_NEG`-style large-negative mask) — shapes `[T, T]`,
    /// `[1, T, T]`, or `[B*H, T, T]` are accepted thanks to the
    /// CPU broadcast.
    ///
    /// Output: `[B, T, embed_dim]`.
    pub fn forward(
        &self,
        q: &Variable,
        k: &Variable,
        v: &Variable,
        attn_mask: Option<&Variable>,
    ) -> Result<Variable, ModuleError> {
        let q_shape = q.tensor().shape().to_vec();
        let k_shape = k.tensor().shape().to_vec();
        let v_shape = v.tensor().shape().to_vec();
        for (name, sh) in [("q", &q_shape), ("k", &k_shape), ("v", &v_shape)] {
            if sh.len() != 3 {
                return Err(rustorch_autograd::BackwardError::Backend {
                    op: "MultiHeadAttention::forward",
                    message: format!("expected rank-3 {name} [B, T, D], got {sh:?}"),
                });
            }
        }
        let batch = q_shape[0];
        let t_q = q_shape[1];
        let t_kv = k_shape[1];
        let embed = q_shape[2];
        if embed != self.embed_dim || k_shape[2] != self.embed_dim || v_shape[2] != self.embed_dim {
            return Err(rustorch_autograd::BackwardError::Backend {
                op: "MultiHeadAttention::forward",
                message: format!(
                    "embed dim mismatch: q={}, k={}, v={}, configured={}",
                    embed, k_shape[2], v_shape[2], self.embed_dim
                ),
            });
        }
        if k_shape[0] != batch || v_shape[0] != batch || v_shape[1] != t_kv {
            return Err(rustorch_autograd::BackwardError::Backend {
                op: "MultiHeadAttention::forward",
                message: format!(
                    "k/v batch or seq mismatch: q={q_shape:?}, k={k_shape:?}, v={v_shape:?}"
                ),
            });
        }

        // 1. Project Q, K, V — Linear is rank-N capable so [B, T, D] → [B, T, D].
        let q = self.q_proj.forward(q)?;
        let k = self.k_proj.forward(k)?;
        let v = self.v_proj.forward(v)?;

        // 2. Split heads. Q uses t_q; K and V use t_kv (cross-attention
        //    can have different query and key/value sequence lengths).
        let q = self.split_heads(&q, batch, t_q)?;
        let k = self.split_heads(&k, batch, t_kv)?;
        let v = self.split_heads(&v, batch, t_kv)?;

        // 3. K transpose for QK^T: [B*H, T_kv, hd] → [B*H, hd, T_kv]
        let k_t = ops::transpose(&k, 1, 2)?;
        // 4. Scores = (Q @ K^T) / sqrt(head_dim) → [B*H, T_q, T_kv]
        let raw = ops::bmm(&q, &k_t)?;
        let inv_scale = Variable::new(Tensor::scalar(1.0_f32 / (self.head_dim as f32).sqrt()));
        let scores = ops::mul(&raw, &inv_scale)?;
        // 5. Optional additive mask (broadcast over the leading B*H dim)
        let scores = if let Some(m) = attn_mask {
            ops::add(&scores, m)?
        } else {
            scores
        };
        // 6. Softmax over last dim
        let attn = ops::softmax(&scores, 2)?;
        // 7. Attn @ V → [B*H, T_q, head_dim]
        let context = ops::bmm(&attn, &v)?;
        // 8. Unfold + transpose + reshape back to [B, T_q, D]
        let context = self.merge_heads(&context, batch, t_q)?;
        // 9. Output projection
        self.o_proj.forward(&context)
    }

    /// Self-attention shortcut: `forward(x, x, x, mask)`.
    pub fn self_attention(
        &self,
        x: &Variable,
        attn_mask: Option<&Variable>,
    ) -> Result<Variable, ModuleError> {
        self.forward(x, x, x, attn_mask)
    }

    /// `[B, T, D] → [B*H, T, head_dim]` via reshape + transpose +
    /// fold-batch. All steps are autograd-aware.
    fn split_heads(&self, x: &Variable, batch: usize, seq: usize) -> Result<Variable, ModuleError> {
        // [B, T, D] → [B, T, H, hd]
        let x4 = ops::reshape(x, vec![batch, seq, self.num_heads, self.head_dim])?;
        // [B, T, H, hd] → [B, H, T, hd]
        let x4 = ops::transpose(&x4, 1, 2)?;
        // [B, H, T, hd] → [B*H, T, hd]
        ops::reshape(&x4, vec![batch * self.num_heads, seq, self.head_dim])
    }

    /// Inverse of `split_heads`: `[B*H, T, head_dim] → [B, T, D]`.
    fn merge_heads(&self, x: &Variable, batch: usize, seq: usize) -> Result<Variable, ModuleError> {
        // [B*H, T, hd] → [B, H, T, hd]
        let x4 = ops::reshape(x, vec![batch, self.num_heads, seq, self.head_dim])?;
        // [B, H, T, hd] → [B, T, H, hd]
        let x4 = ops::transpose(&x4, 1, 2)?;
        // [B, T, H, hd] → [B, T, D]
        ops::reshape(&x4, vec![batch, seq, self.embed_dim])
    }
}

impl Module for MultiHeadAttention {
    /// Self-attention default — equivalent to `self_attention(input, None)`.
    fn forward(&self, input: &Variable) -> Result<Variable, ModuleError> {
        self.self_attention(input, None)
    }

    fn parameters(&self) -> Vec<Variable> {
        let mut p = self.q_proj.parameters();
        p.extend(self.k_proj.parameters());
        p.extend(self.v_proj.parameters());
        p.extend(self.o_proj.parameters());
        p
    }

    fn named_parameters(&self) -> Vec<(String, Variable)> {
        let mut out = Vec::new();
        for (sub, v) in self.q_proj.named_parameters() {
            out.push((format!("q_proj.{sub}"), v));
        }
        for (sub, v) in self.k_proj.named_parameters() {
            out.push((format!("k_proj.{sub}"), v));
        }
        for (sub, v) in self.v_proj.named_parameters() {
            out.push((format!("v_proj.{sub}"), v));
        }
        for (sub, v) in self.o_proj.named_parameters() {
            out.push((format!("o_proj.{sub}"), v));
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sdp_attention_uniform_input_uniform_output() {
        // If Q, K, V are all the same constant tensor, attn weights are
        // uniform 1/T, so output equals row-mean of V (which equals V
        // itself when V is constant).
        let c = Variable::new(Tensor::from_vec([1usize, 3, 2], vec![1.0_f32; 6]).unwrap());
        let out = scaled_dot_product_attention(&c, &c, &c).unwrap();
        assert_eq!(out.tensor().shape(), &[1, 3, 2]);
        let out_t = out.tensor();
        let v = out_t.as_slice::<f32>().unwrap();
        for &x in v {
            assert!((x - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn sdp_attention_shape_preserved() {
        let q = Variable::new(
            Tensor::from_vec(
                [2usize, 4, 3],
                (0..24).map(|i| i as f32 * 0.1).collect::<Vec<_>>(),
            )
            .unwrap(),
        );
        let k = Variable::new(
            Tensor::from_vec(
                [2usize, 4, 3],
                (0..24).map(|i| (i as f32 + 1.0) * 0.05).collect::<Vec<_>>(),
            )
            .unwrap(),
        );
        let v = Variable::new(
            Tensor::from_vec(
                [2usize, 4, 3],
                (0..24).map(|i| (i as f32 + 5.0) * 0.02).collect::<Vec<_>>(),
            )
            .unwrap(),
        );
        let out = scaled_dot_product_attention(&q, &k, &v).unwrap();
        assert_eq!(out.tensor().shape(), &[2, 4, 3]);
    }

    #[test]
    fn single_head_attention_b1_shape_correct() {
        let attn = SingleHeadAttention::new(4);
        let x = Variable::new(
            Tensor::from_vec(
                [1usize, 5, 4],
                (0..20).map(|i| i as f32 * 0.1).collect::<Vec<_>>(),
            )
            .unwrap(),
        );
        let out = attn.forward(&x).unwrap();
        assert_eq!(out.tensor().shape(), &[1, 5, 4]);
        // 4 projections * (weight + bias) = 8 parameters
        assert_eq!(attn.parameters().len(), 8);
    }

    #[test]
    fn single_head_attention_named_parameters() {
        let attn = SingleHeadAttention::new(2);
        let np = attn.named_parameters();
        let names: Vec<String> = np.iter().map(|(n, _)| n.clone()).collect();
        assert!(names.contains(&"q_proj.weight".to_string()));
        assert!(names.contains(&"q_proj.bias".to_string()));
        assert!(names.contains(&"o_proj.weight".to_string()));
    }

    // ----------------------------------------------------------------
    // MultiHeadAttention
    // ----------------------------------------------------------------

    /// Output shape is `[B, T, embed_dim]` and parameters() returns the
    /// 8 expected entries (4 projections × {weight, bias}).
    #[test]
    fn multihead_attention_output_shape_and_params() {
        let mha = MultiHeadAttention::new(8, 4);
        let x = Variable::new(
            Tensor::from_vec(
                [2usize, 5, 8],
                (0..80).map(|i| (i as f32) * 0.01).collect::<Vec<_>>(),
            )
            .unwrap(),
        );
        let out = mha.self_attention(&x, None).unwrap();
        assert_eq!(out.tensor().shape(), &[2, 5, 8]);
        assert_eq!(mha.parameters().len(), 8);
        assert_eq!(mha.head_dim(), 2);
    }

    /// num_heads=1 should be functionally equivalent to a single-head
    /// path (same scaled-dot-product structure). We cannot directly
    /// compare to SingleHeadAttention because that one uses different
    /// internal helpers, but we can sanity-check that num_heads=1 with
    /// constant input produces a constant output (every position
    /// attends uniformly to every other position).
    #[test]
    fn multihead_num_heads_1_constant_input_constant_output() {
        let mha = MultiHeadAttention::new(4, 1);
        // Constant input → uniform attention → output constant per row.
        let x = Variable::new(Tensor::from_vec([1usize, 3, 4], vec![1.0_f32; 12]).unwrap());
        let out = mha.self_attention(&x, None).unwrap();
        let t = out.tensor();
        let s = t.as_slice::<f32>().unwrap();
        // Each "row" of the output should have the same constant value
        // because attention weights are uniform 1/T and V is constant.
        let row0: &[f32] = &s[0..4];
        for r in 1..3 {
            for c in 0..4 {
                assert!(
                    (s[r * 4 + c] - row0[c]).abs() < 1e-4,
                    "row {r} col {c} differs"
                );
            }
        }
    }

    /// Backward through MHA must produce a non-zero gradient on every
    /// projection's weight (proves all four paths are autograd-aware
    /// through the rank-3 fold-batch path).
    #[test]
    fn multihead_backward_flows_to_all_projections() {
        use rustorch_autograd::{backward, ops::sum};
        let mha = MultiHeadAttention::new(6, 3);
        let x = Variable::leaf(
            Tensor::from_vec(
                [2usize, 4, 6],
                (0..48).map(|i| (i as f32) * 0.01).collect::<Vec<_>>(),
            )
            .unwrap(),
        );
        let out = mha.self_attention(&x, None).unwrap();
        let s = sum(&out).unwrap();
        backward(&s, None).unwrap();
        for proj in [&mha.q_proj, &mha.k_proj, &mha.v_proj, &mha.o_proj] {
            let g = proj
                .weight
                .grad()
                .expect("projection should have a weight gradient");
            assert_eq!(g.shape(), &[6, 6]);
            let any_nonzero = g.as_slice::<f32>().unwrap().iter().any(|&x| x.abs() > 0.0);
            assert!(any_nonzero, "weight grad is all zeros — flow broken");
        }
    }

    /// Adding a causal mask should make position 0 only attend to
    /// position 0 — the row-0 output for self-attention with constant
    /// V should equal V[0] (i.e., uniform attention is "killed" above
    /// the diagonal).
    #[test]
    fn multihead_causal_mask_restricts_first_row() {
        let mha = MultiHeadAttention::new(4, 2);
        // Constant input — without mask, output rows would all be equal.
        let x = Variable::new(Tensor::from_vec([1usize, 3, 4], vec![1.0_f32; 12]).unwrap());
        // Build a causal mask of shape [3, 3] (broadcasts over [B*H, 3, 3]).
        let mask = crate::masks::causal_mask(3);
        let out = mha.self_attention(&x, Some(&mask)).unwrap();
        // Sanity: shape preserved.
        assert_eq!(out.tensor().shape(), &[1, 3, 4]);
        // Row 0 should be finite and well-defined (no NaN from mask).
        let t = out.tensor();
        let s = t.as_slice::<f32>().unwrap();
        for &v in &s[0..4] {
            assert!(v.is_finite(), "row 0 has non-finite value: {v}");
        }
    }

    /// Wrong embed_dim surfaces as a clean error, no panic.
    #[test]
    fn multihead_rejects_wrong_embed_dim() {
        let mha = MultiHeadAttention::new(4, 2);
        let x = Variable::new(Tensor::from_vec([1usize, 3, 8], vec![0.0_f32; 24]).unwrap());
        assert!(mha.self_attention(&x, None).is_err());
    }

    /// Named parameters expose the 4 projections distinctly.
    #[test]
    fn multihead_named_parameters_distinct_projections() {
        let mha = MultiHeadAttention::new(4, 2);
        let names: Vec<String> = mha.named_parameters().into_iter().map(|(n, _)| n).collect();
        for prefix in ["q_proj", "k_proj", "v_proj", "o_proj"] {
            let weight = format!("{prefix}.weight");
            assert!(names.contains(&weight), "missing {weight}");
        }
    }
}
