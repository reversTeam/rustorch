//! `Linear` layer — y = x @ W + b (P1.6 task `Linear`).
//!
//! Weight initialisation: small uniform around 0 (`±1/sqrt(in_features)`)
//! — Kaiming-uniform-style default that works for ReLU MLPs without a
//! configurable Init enum (the full Init builder lands in a follow-up).
//!
//! Bias is optional via [`Linear::no_bias`]. Created on construction;
//! parameters are owned `Variable`s with `requires_grad = true`.

use crate::module::{Module, ModuleError};
use rustorch_autograd::ops::{add_bias, matmul, reshape};
use rustorch_autograd::{is_grad_enabled, Variable};
use rustorch_core::tensor::dtype::Dtype;
use rustorch_core::tensor::tensor_impl::Tensor;
use rustorch_fusion::patterns::matmul_bias_act::{fused_matmul_bias_activation, Activation};

/// Affine transformation `y = x @ weight + bias` (rustorch convention:
/// weight is `[in_features, out_features]`).
pub struct Linear {
    /// Trainable weight tensor of shape `[in_features, out_features]`.
    pub weight: Variable,
    /// Optional trainable bias of shape `[out_features]` (broadcast on add).
    pub bias: Option<Variable>,
    in_features: usize,
    out_features: usize,
}

impl Linear {
    /// Build a Linear layer with the given input/output dimensions.
    /// Weights are initialised in `[-bound, bound]` with `bound =
    /// 1/sqrt(in_features)`.
    pub fn new(in_features: usize, out_features: usize) -> Self {
        let bound = 1.0_f32 / (in_features as f32).sqrt();
        let w_data: Vec<f32> = (0..in_features * out_features)
            .map(|i| {
                // Deterministic LCG noise scaled to [-bound, bound]
                let mut s = (i as u64)
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1);
                s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                let u = (s >> 32) as u32;
                let f = (u as f32) / (u32::MAX as f32); // [0, 1)
                (f * 2.0 - 1.0) * bound
            })
            .collect();
        let weight = Variable::leaf(
            Tensor::from_vec([in_features, out_features], w_data).expect("weight shape"),
        );
        let bias_data: Vec<f32> = vec![0.0_f32; out_features];
        let bias = Variable::leaf(Tensor::from_vec([out_features], bias_data).expect("bias shape"));
        Linear {
            weight,
            bias: Some(bias),
            in_features,
            out_features,
        }
    }

    /// Build a Linear layer without bias.
    pub fn no_bias(in_features: usize, out_features: usize) -> Self {
        let mut l = Linear::new(in_features, out_features);
        l.bias = None;
        l
    }

    /// Input feature count.
    pub fn in_features(&self) -> usize {
        self.in_features
    }

    /// Output feature count.
    pub fn out_features(&self) -> usize {
        self.out_features
    }

    /// T20 — fused linear + activation forward (no_grad path only).
    /// Equivalent to `forward(input).then(activation)` but executed
    /// in a single matmul-bias-act kernel via `rustorch_fusion`,
    /// saving two intermediate Tensor allocations and an autograd
    /// dispatch cycle. Useful in transformer FFNs:
    ///
    /// ```ignore
    /// let h = fc1.forward_with_activation(&x, Activation::Relu)?;
    /// let y = fc2.forward(&h)?;
    /// ```
    pub fn forward_with_activation(
        &self,
        input: &Variable,
        activation: Activation,
    ) -> Result<Variable, ModuleError> {
        if !is_grad_enabled() {
            if let Some(out) = try_fused_linear_act(self, input, activation)? {
                return Ok(out);
            }
        }
        // Slow path: legacy compose under autograd. Activation is
        // applied via the autograd op separately so backward keeps
        // working through the composed graph.
        let pre = self.forward(input)?;
        match activation {
            Activation::None => Ok(pre),
            Activation::Relu => rustorch_autograd::ops::relu(&pre),
            Activation::Gelu | Activation::Silu => Err(ModuleError::Backend {
                op: "Linear::forward_with_activation",
                message: format!(
                    "{:?} not yet wired in autograd compose path; \
                     use no_grad or extend rustorch_autograd::ops",
                    activation
                ),
            }),
        }
    }
}

/// Internal helper — collapses Linear::forward + activation into a
/// single fused kernel call when the no_grad fast path applies.
/// Returns `Ok(None)` if any precondition fails so the caller can
/// fall back to the autograd compose path.
fn try_fused_linear_act(
    layer: &Linear,
    input: &Variable,
    activation: Activation,
) -> Result<Option<Variable>, ModuleError> {
    let in_t = input.tensor();
    let w_t = layer.weight.tensor();
    let in_shape = in_t.shape().to_vec();
    let rank = in_shape.len();
    if rank < 2 {
        return Ok(None);
    }
    if in_t.dtype() != Dtype::F32 || w_t.dtype() != Dtype::F32 {
        return Ok(None);
    }
    if !in_t.is_contiguous() || !w_t.is_contiguous() {
        return Ok(None);
    }
    if let Some(b_var) = &layer.bias {
        let b_t = b_var.tensor();
        if b_t.dtype() != Dtype::F32 || !b_t.is_contiguous() {
            return Ok(None);
        }
    }

    // Flatten leading batch dims into a single M.
    let leading: usize = in_shape[..rank - 1].iter().product();
    let k = layer.in_features;
    let n = layer.out_features;
    let m = leading;

    let x_slice = in_t.as_slice::<f32>().expect("checked F32 contiguous");
    let w_slice = w_t.as_slice::<f32>().expect("checked F32 contiguous");
    // T35 — borrow the bias tensor directly instead of cloning the
    // whole vector on every call. The previous `to_vec()` allocated
    // and copied N f32s per Linear forward, which on MHA S=1
    // (4 × Linear[1, 768] = 4 × 3 KB clone + alloc) compounded to
    // ~10 µs of pure copy overhead per layer × 12 layers = ~120 µs
    // wasted on the LM-serving hot path. We bind the Tensor (which
    // is a cheap Arc-clone of the storage) to a local so the slice
    // borrow it produces stays valid through the matmul call.
    let bias_tensor: Option<Tensor> = layer.bias.as_ref().map(|b| b.tensor());
    let bias_ref: Option<&[f32]> = bias_tensor
        .as_ref()
        .map(|t| t.as_slice::<f32>().expect("checked F32 contiguous"));

    // T35 — uninitialised output buffer; the fused kernel calls
    // sgemm/sgemv with beta=0, so every cell is overwritten. Skips
    // the m*n f32 zero-fill that `vec![0.0; m*n]` emits.
    let mut y_storage: Vec<core::mem::MaybeUninit<f32>> = Vec::with_capacity(m * n);
    #[allow(clippy::uninit_vec)]
    unsafe {
        y_storage.set_len(m * n);
    }
    let mut y: Vec<f32> = unsafe {
        let (ptr, len, cap) = (
            y_storage.as_mut_ptr() as *mut f32,
            y_storage.len(),
            y_storage.capacity(),
        );
        core::mem::forget(y_storage);
        Vec::from_raw_parts(ptr, len, cap)
    };
    fused_matmul_bias_activation(x_slice, w_slice, bias_ref, &mut y, m, k, n, activation).map_err(
        |e| ModuleError::Backend {
            op: "Linear::forward_with_activation(fused)",
            message: format!("{e:?}"),
        },
    )?;

    let mut out_shape = in_shape;
    *out_shape.last_mut().unwrap() = n;
    let t = Tensor::from_vec(out_shape, y).map_err(|e| ModuleError::Backend {
        op: "Linear::forward_with_activation(fused)",
        message: format!("{e:?}"),
    })?;
    Ok(Some(Variable::new(t)))
}

impl Module for Linear {
    fn forward(&self, input: &Variable) -> Result<Variable, ModuleError> {
        // T20 — no_grad fast path: a single fused matmul+bias kernel
        // call (one allocation, no autograd dispatch). Saves the two
        // intermediate Tensor allocations + autograd nodes that the
        // composed `matmul -> add_bias` path produces. On a GPT-2
        // FFN sub-block this halves the latency.
        if !is_grad_enabled() {
            if let Some(out) = try_fused_linear_act(self, input, Activation::None)? {
                return Ok(out);
            }
        }

        // PyTorch-style `nn.Linear` accepts arbitrary leading batch dims:
        //   `[*, in_features] -> [*, out_features]`
        //
        // Implementation: flatten leading dims into a single batch axis
        // (`[d0*d1*...*d{N-2}, in_features]`), run the rank-2 matmul +
        // bias path, then reshape back to `[d0, d1, ..., d{N-2}, out_features]`.
        // The fast path for rank-2 inputs avoids the round-trip reshapes.
        let in_shape = input.tensor().shape().to_vec();
        let rank = in_shape.len();
        if rank < 2 {
            return Err(ModuleError::Backend {
                op: "Linear::forward",
                message: format!("input rank must be >= 2, got shape {:?}", in_shape),
            });
        }
        if *in_shape.last().unwrap() != self.in_features {
            return Err(ModuleError::Backend {
                op: "Linear::forward",
                message: format!(
                    "last dim {} != in_features {}",
                    in_shape.last().unwrap(),
                    self.in_features
                ),
            });
        }

        if rank == 2 {
            // Fast path — preserves the existing v1 behaviour exactly.
            let xw = matmul(input, &self.weight)?;
            return match &self.bias {
                Some(b) => add_bias(&xw, b),
                None => Ok(xw),
            };
        }

        // Rank-N path: flatten -> rank-2 -> reshape back.
        let leading: usize = in_shape[..rank - 1].iter().product();
        let in_dim = self.in_features;
        let x_flat = reshape(input, vec![leading, in_dim])?;
        let y_flat = matmul(&x_flat, &self.weight)?;
        let y_flat = match &self.bias {
            Some(b) => add_bias(&y_flat, b)?,
            None => y_flat,
        };
        let mut out_shape = in_shape;
        *out_shape.last_mut().unwrap() = self.out_features;
        reshape(&y_flat, out_shape)
    }

    fn parameters(&self) -> Vec<Variable> {
        let mut params = vec![self.weight.clone()];
        if let Some(b) = &self.bias {
            params.push(b.clone());
        }
        params
    }

    fn named_parameters(&self) -> Vec<(String, Variable)> {
        let mut out = vec![("weight".to_string(), self.weight.clone())];
        if let Some(b) = &self.bias {
            out.push(("bias".to_string(), b.clone()));
        }
        out
    }
}

/// Bilinear layer: `y = x1 W x2 + b`. v1 simplification: implemented
/// as two stacked matmuls (`(x1 @ W) @ x2.T` rather than the dense
/// 3-D weight torch uses) — produces the right shape but treats W as
/// rank-2 in1×in2 rather than rank-3 out×in1×in2. Equivalent for
/// out_features=1 and useful as a building block; full 3-D Bilinear
/// pending follow-up.
pub struct Bilinear {
    /// Joint weight `[in1, in2]` (v1) or `[out, in1, in2]` (post-v1).
    pub weight: Variable,
    /// Optional bias `[out_features]`.
    pub bias: Option<Variable>,
    in1_features: usize,
    in2_features: usize,
}

impl Bilinear {
    /// Build with the two input dims. v1 sets `out_features = 1`
    /// implicitly; the joint weight has shape `[in1, in2]`.
    pub fn new(in1_features: usize, in2_features: usize) -> Self {
        let bound = 1.0_f32 / (in1_features.max(in2_features) as f32).sqrt();
        let w_data: Vec<f32> = (0..in1_features * in2_features)
            .map(|i| {
                let mut s = (i as u64)
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1);
                s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                let u = (s >> 32) as u32;
                let f = (u as f32) / (u32::MAX as f32);
                (f * 2.0 - 1.0) * bound
            })
            .collect();
        let weight = Variable::leaf(
            Tensor::from_vec([in1_features, in2_features], w_data).expect("bilinear weight"),
        );
        let bias =
            Variable::leaf(Tensor::from_vec([1usize], vec![0.0_f32]).expect("bilinear bias"));
        Bilinear {
            weight,
            bias: Some(bias),
            in1_features,
            in2_features,
        }
    }

    /// Forward with explicit two inputs. Returns `[B, 1]` (v1).
    pub fn forward2(&self, x1: &Variable, x2: &Variable) -> Result<Variable, ModuleError> {
        // y_b = x1_b @ W @ x2_b.T → [B, in2_features].mul(x2) summed
        // Implementation: (x1 @ W) gives [B, in2]; elementwise * x2 then sum over last dim → [B].
        let xw = matmul(x1, &self.weight)?;
        let prod = rustorch_autograd::ops::mul(&xw, x2)?;
        let last = prod.tensor().ndim() - 1;
        let summed = rustorch_autograd::ops::mean_dim(&prod, &[last])?;
        // Multiply by in2_features to undo the mean (we want sum, not mean).
        let scale = Variable::new(Tensor::scalar(self.in2_features as f32));
        let out = rustorch_autograd::ops::mul(&summed, &scale)?;
        match &self.bias {
            Some(b) => add_bias(&out, b),
            None => Ok(out),
        }
    }

    /// First input dim.
    pub fn in1_features(&self) -> usize {
        self.in1_features
    }
    /// Second input dim.
    pub fn in2_features(&self) -> usize {
        self.in2_features
    }
}

#[cfg(test)]
mod rank_n_tests {
    use super::*;
    use rustorch_autograd::backward;

    /// Rank-2 fast path: existing `[B, in] -> [B, out]` behaviour preserved.
    #[test]
    fn linear_rank2_preserves_legacy_shape() {
        let layer = Linear::new(4, 3);
        let x = Variable::new(
            Tensor::from_vec(
                [2usize, 4],
                (0..8).map(|i| i as f32 * 0.1).collect::<Vec<_>>(),
            )
            .unwrap(),
        );
        let y = layer.forward(&x).unwrap();
        assert_eq!(y.tensor().shape(), &[2, 3]);
    }

    /// Rank-3 input `[B, T, D] -> [B, T, out]` (the canonical transformer
    /// activation shape).
    #[test]
    fn linear_rank3_btd_input() {
        let layer = Linear::new(4, 5);
        let x = Variable::new(
            Tensor::from_vec(
                [2usize, 3, 4],
                (0..24).map(|i| i as f32 * 0.01).collect::<Vec<_>>(),
            )
            .unwrap(),
        );
        let y = layer.forward(&x).unwrap();
        assert_eq!(y.tensor().shape(), &[2, 3, 5]);
    }

    /// Rank-4 input `[B, H, T, D] -> [B, H, T, out]` (multi-head attention
    /// per-head projection shape).
    #[test]
    fn linear_rank4_bhtd_input() {
        let layer = Linear::new(4, 6);
        let x = Variable::new(
            Tensor::from_vec(
                [2usize, 2, 3, 4],
                (0..48).map(|i| (i as f32) * 0.005).collect::<Vec<_>>(),
            )
            .unwrap(),
        );
        let y = layer.forward(&x).unwrap();
        assert_eq!(y.tensor().shape(), &[2, 2, 3, 6]);
    }

    /// Rank-3 forward must equal rank-2 forward applied row-by-row.
    /// Confirms the flatten + reshape round-trip preserves values.
    #[test]
    fn linear_rank3_matches_flatten_reference() {
        let layer = Linear::new(3, 2);
        // Same data interpreted as [2, 4, 3] (rank-3) and [8, 3] (rank-2).
        let data: Vec<f32> = (0..24).map(|i| i as f32 * 0.1).collect();
        let x3 = Variable::new(Tensor::from_vec([2usize, 4, 3], data.clone()).unwrap());
        let x2 = Variable::new(Tensor::from_vec([8usize, 3], data).unwrap());
        let y3 = layer.forward(&x3).unwrap();
        let y2 = layer.forward(&x2).unwrap();
        let y3_t = y3.tensor();
        let y2_t = y2.tensor();
        let y3_v = y3_t.as_slice::<f32>().unwrap();
        let y2_v = y2_t.as_slice::<f32>().unwrap();
        assert_eq!(y3_v.len(), y2_v.len());
        for (a, b) in y3_v.iter().zip(y2_v.iter()) {
            assert!((a - b).abs() < 1e-6, "mismatch: {} vs {}", a, b);
        }
    }

    /// Backward must propagate to the weight even on rank-3 input
    /// (proves the flatten + reshape round-trip is autograd-aware).
    #[test]
    fn linear_rank3_backward_updates_weight_grad() {
        let layer = Linear::new(3, 2);
        let x = Variable::new(
            Tensor::from_vec(
                [2usize, 4, 3],
                (0..24).map(|i| i as f32 * 0.01).collect::<Vec<_>>(),
            )
            .unwrap(),
        );
        let y = layer.forward(&x).unwrap();
        // Sum to a scalar so backward has a unit gradient.
        let loss = rustorch_autograd::ops::sum(&y).unwrap();
        backward(&loss, None).unwrap();
        let w_grad = layer.weight.grad().expect("weight should have grad");
        assert_eq!(w_grad.shape(), &[3, 2]);
        // At least one entry should be non-zero (input was non-trivial).
        let any_nonzero = w_grad
            .as_slice::<f32>()
            .unwrap()
            .iter()
            .any(|&g| g.abs() > 0.0);
        assert!(any_nonzero, "weight grad is all zeros — flow broken");
    }

    /// Rank < 2 surfaces as a clean error rather than panicking.
    #[test]
    fn linear_rejects_rank1_input() {
        let layer = Linear::new(3, 2);
        let x = Variable::new(Tensor::from_vec([3usize], vec![1.0, 2.0, 3.0]).unwrap());
        assert!(layer.forward(&x).is_err());
    }

    /// Last-dim mismatch surfaces as a clean error.
    #[test]
    fn linear_rejects_wrong_last_dim() {
        let layer = Linear::new(4, 2);
        let x = Variable::new(Tensor::from_vec([2usize, 3, 5], vec![0.0_f32; 30]).unwrap());
        assert!(layer.forward(&x).is_err());
    }
}
