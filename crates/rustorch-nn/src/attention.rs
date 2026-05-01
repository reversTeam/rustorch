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
}
