//! `Linear` layer — y = x @ W + b (P1.6 task `Linear`).
//!
//! Weight initialisation: small uniform around 0 (`±1/sqrt(in_features)`)
//! — Kaiming-uniform-style default that works for ReLU MLPs without a
//! configurable Init enum (the full Init builder lands in a follow-up).
//!
//! Bias is optional via [`Linear::no_bias`]. Created on construction;
//! parameters are owned `Variable`s with `requires_grad = true`.

use crate::module::{Module, ModuleError};
use rustorch_autograd::ops::{add_bias, matmul};
use rustorch_autograd::Variable;
use rustorch_core::tensor::tensor_impl::Tensor;

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
}

impl Module for Linear {
    fn forward(&self, input: &Variable) -> Result<Variable, ModuleError> {
        // x [B, in] @ W [in, out] = [B, out]; then optional + bias [out]
        // (broadcast across batch via the autograd-aware `add_bias` op).
        let xw = matmul(input, &self.weight)?;
        match &self.bias {
            Some(b) => add_bias(&xw, b),
            None => Ok(xw),
        }
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
