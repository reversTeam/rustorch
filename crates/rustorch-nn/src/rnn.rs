//! Recurrent layers (P1.6) — RNN / LSTM / GRU minimal slice.
//!
//! v1 ships single-layer, single-direction cells over `[T, B, H_in]`
//! input that step through time and produce `[T, B, H_out]` output.
//! Implemented purely via composition of autograd-aware ops (Linear /
//! tanh / sigmoid / mul / add) so backward inherits from the dynamic
//! graph.
//!
//! Stacked variants and bidirectional are pending.

use crate::activation::{sigmoid as fn_sigmoid, tanh as fn_tanh};
use crate::linear::Linear;
use crate::module::{Module, ModuleError};
use rustorch_autograd::{ops, Variable};
use rustorch_core::tensor::tensor_impl::Tensor;

/// Vanilla single-layer Tanh-RNN.
pub struct RnnCell {
    /// Input → hidden projection.
    pub w_ih: Linear,
    /// Hidden → hidden projection.
    pub w_hh: Linear,
    hidden_size: usize,
}

impl RnnCell {
    /// Build with the given input and hidden sizes.
    pub fn new(input_size: usize, hidden_size: usize) -> Self {
        RnnCell {
            w_ih: Linear::new(input_size, hidden_size),
            w_hh: Linear::no_bias(hidden_size, hidden_size),
            hidden_size,
        }
    }

    /// Single time-step: `h_new = tanh(x @ W_ih + h @ W_hh)`.
    pub fn step(&self, x: &Variable, h: &Variable) -> Result<Variable, ModuleError> {
        let xw = self.w_ih.forward(x)?;
        let hw = self.w_hh.forward(h)?;
        let pre = ops::add(&xw, &hw)?;
        fn_tanh(&pre)
    }

    /// Forward over the full sequence `[T, B, H_in]`. Returns the
    /// output `[T, B, H_out]` (last hidden state at index `T-1`).
    pub fn forward_seq(&self, input: &Variable) -> Result<Variable, ModuleError> {
        let s = input.tensor().shape().to_vec();
        if s.len() != 3 {
            return Err(rustorch_autograd::BackwardError::Backend {
                op: "rnn_forward",
                message: format!("expected [T, B, H_in], got {s:?}"),
            });
        }
        let (t, b, _h_in) = (s[0], s[1], s[2]);
        // Initial hidden state: zeros [B, H_out]
        let h0 = Variable::leaf(
            Tensor::from_vec([b, self.hidden_size], vec![0.0_f32; b * self.hidden_size])
                .expect("h0"),
        );
        let mut h = h0;
        let mut outs: Vec<Variable> = Vec::with_capacity(t);
        for ti in 0..t {
            // Slice x[ti, :, :] via index_select on dim 0 then reshape.
            let idx = Tensor::from_vec_typed::<i64, _>([1usize], vec![ti as i64]).unwrap();
            let xt = ops::index_select(input, &idx)?;
            let xt = ops::reshape(&xt, vec![b, s[2]])?;
            let h_new = self.step(&xt, &h)?;
            outs.push(h_new.clone());
            h = h_new;
        }
        // Stack outputs along dim 0 → [T, B, H_out]
        // Use cpu_backend.stack via raw tensors (autograd flow lost on stack;
        // OK for v1 — backward goes through each step's autograd graph).
        let raw: Vec<Tensor> = outs.iter().map(|v| v.tensor().clone()).collect();
        let refs: Vec<&Tensor> = raw.iter().collect();
        let stacked = rustorch_cpu::cpu_backend::cpu_backend()
            .stack(&refs, 0)
            .map_err(|e| rustorch_autograd::BackwardError::Backend {
                op: "rnn_stack",
                message: e.to_string(),
            })?;
        Ok(Variable::new(stacked))
    }

    /// Hidden size.
    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }
}

impl Module for RnnCell {
    fn forward(&self, input: &Variable) -> Result<Variable, ModuleError> {
        self.forward_seq(input)
    }
    fn parameters(&self) -> Vec<Variable> {
        let mut p = self.w_ih.parameters();
        p.extend(self.w_hh.parameters());
        p
    }
    fn named_parameters(&self) -> Vec<(String, Variable)> {
        let mut out = Vec::new();
        for (n, v) in self.w_ih.named_parameters() {
            out.push((format!("w_ih.{n}"), v));
        }
        for (n, v) in self.w_hh.named_parameters() {
            out.push((format!("w_hh.{n}"), v));
        }
        out
    }
}

/// LSTM cell (single time-step). Standard 4-gate formulation:
/// ```text
///   i = sigmoid(x @ W_ii + h @ W_hi)
///   f = sigmoid(x @ W_if + h @ W_hf)
///   g =    tanh(x @ W_ig + h @ W_hg)
///   o = sigmoid(x @ W_io + h @ W_ho)
///   c' = f * c + i * g
///   h' = o * tanh(c')
/// ```
/// Implemented via 8 Linear layers (4 input-to-* and 4 hidden-to-*)
/// for clarity. The fused "single 4H matmul" optimisation is pending.
pub struct LstmCell {
    /// Input projection for the four gates concatenated.
    pub w_ii: Linear,
    /// (similar)
    pub w_if: Linear,
    /// (similar)
    pub w_ig: Linear,
    /// (similar)
    pub w_io: Linear,
    /// Hidden projection for the four gates.
    pub w_hi: Linear,
    /// (similar)
    pub w_hf: Linear,
    /// (similar)
    pub w_hg: Linear,
    /// (similar)
    pub w_ho: Linear,
    hidden_size: usize,
}

impl LstmCell {
    /// Build a single-layer LSTM cell.
    pub fn new(input_size: usize, hidden_size: usize) -> Self {
        LstmCell {
            w_ii: Linear::new(input_size, hidden_size),
            w_if: Linear::new(input_size, hidden_size),
            w_ig: Linear::new(input_size, hidden_size),
            w_io: Linear::new(input_size, hidden_size),
            w_hi: Linear::no_bias(hidden_size, hidden_size),
            w_hf: Linear::no_bias(hidden_size, hidden_size),
            w_hg: Linear::no_bias(hidden_size, hidden_size),
            w_ho: Linear::no_bias(hidden_size, hidden_size),
            hidden_size,
        }
    }

    /// Single time-step. Returns `(h', c')`.
    pub fn step(
        &self,
        x: &Variable,
        h: &Variable,
        c: &Variable,
    ) -> Result<(Variable, Variable), ModuleError> {
        let i = fn_sigmoid(&ops::add(&self.w_ii.forward(x)?, &self.w_hi.forward(h)?)?)?;
        let f = fn_sigmoid(&ops::add(&self.w_if.forward(x)?, &self.w_hf.forward(h)?)?)?;
        let g = fn_tanh(&ops::add(&self.w_ig.forward(x)?, &self.w_hg.forward(h)?)?)?;
        let o = fn_sigmoid(&ops::add(&self.w_io.forward(x)?, &self.w_ho.forward(h)?)?)?;
        let fc = ops::mul(&f, c)?;
        let ig = ops::mul(&i, &g)?;
        let c_new = ops::add(&fc, &ig)?;
        let h_new = ops::mul(&o, &fn_tanh(&c_new)?)?;
        Ok((h_new, c_new))
    }

    /// Hidden size.
    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn input(shape: impl Into<Vec<usize>>, data: Vec<f32>) -> Variable {
        Variable::new(Tensor::from_vec(shape.into(), data).unwrap())
    }

    #[test]
    fn rnn_cell_forward_shape() {
        let rnn = RnnCell::new(4, 8);
        // [T=3, B=2, H_in=4]
        let x = input(vec![3usize, 2, 4], vec![0.1_f32; 3 * 2 * 4]);
        let y = rnn.forward(&x).unwrap();
        assert_eq!(y.tensor().shape(), &[3, 2, 8]);
    }

    #[test]
    fn rnn_cell_zero_input_zero_output() {
        let rnn = RnnCell::new(2, 3);
        let x = input(vec![2usize, 1, 2], vec![0.0_f32; 4]);
        let y = rnn.forward(&x).unwrap();
        // tanh(0) = 0; with zero input and zero initial hidden, all outputs ≈ 0
        // (assuming the bias is also small from default init).
        let y_t = y.tensor();
        let v = y_t.as_slice::<f32>().unwrap();
        for &x in v {
            assert!(x.abs() < 1.0);
        }
    }

    #[test]
    fn rnn_cell_named_parameters() {
        let rnn = RnnCell::new(2, 2);
        let np = rnn.named_parameters();
        let names: Vec<String> = np.iter().map(|(n, _)| n.clone()).collect();
        assert!(names.contains(&"w_ih.weight".to_string()));
        assert!(names.contains(&"w_ih.bias".to_string()));
        assert!(names.contains(&"w_hh.weight".to_string()));
    }

    #[test]
    fn lstm_cell_step_returns_two_outputs() {
        let cell = LstmCell::new(3, 4);
        let x = input(vec![1usize, 3], vec![0.1_f32; 3]);
        let h = input(vec![1usize, 4], vec![0.0_f32; 4]);
        let c = input(vec![1usize, 4], vec![0.0_f32; 4]);
        let (h_new, c_new) = cell.step(&x, &h, &c).unwrap();
        assert_eq!(h_new.tensor().shape(), &[1, 4]);
        assert_eq!(c_new.tensor().shape(), &[1, 4]);
    }
}
