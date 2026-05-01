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
