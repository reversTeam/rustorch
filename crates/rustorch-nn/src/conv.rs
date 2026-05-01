//! Convolution Modules (P1.6).
//!
//! v1: [`Conv2d`] with stride=1, configurable padding, optional bias.
//! Higher-rank conv (Conv1d/Conv3d) and configurable stride/dilation/
//! groups are pending.

use crate::init::{init_with_seed, FanMode, Init, Nonlinearity};
use crate::module::{Module, ModuleError};
use rustorch_autograd::{ops, Variable};
use rustorch_core::tensor::tensor_impl::Tensor;

/// 2-D convolution. Layout: input `[N, C_in, H, W]`,
/// weight `[C_out, C_in, kH, kW]`. Stride is fixed at 1 in v1.
pub struct Conv2d {
    /// Trainable convolution weight.
    pub weight: Variable,
    /// Optional trainable bias of shape `[C_out]`.
    pub bias: Option<Variable>,
    in_channels: usize,
    out_channels: usize,
    kernel_size: (usize, usize),
    padding: (usize, usize),
}

impl Conv2d {
    /// Build a square-kernel `Conv2d` with the given channels and
    /// kernel size. Default: padding=0, bias=true, Kaiming-uniform init
    /// (FanIn, ReLU nonlinearity).
    pub fn new(in_channels: usize, out_channels: usize, kernel_size: usize) -> Self {
        Self::with_padding(
            in_channels,
            out_channels,
            (kernel_size, kernel_size),
            (0, 0),
        )
    }

    /// Build with custom (kH, kW) and (padH, padW).
    pub fn with_padding(
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize),
        padding: (usize, usize),
    ) -> Self {
        let (kh, kw) = kernel_size;
        let weight_t = init_with_seed(
            Init::KaimingUniform {
                mode: FanMode::FanIn,
                nonlinearity: Nonlinearity::Relu,
            },
            &[out_channels, in_channels, kh, kw],
            0,
        );
        let bias_t =
            Tensor::from_vec([out_channels], vec![0.0_f32; out_channels]).expect("bias shape");
        Conv2d {
            weight: Variable::leaf(weight_t),
            bias: Some(Variable::leaf(bias_t)),
            in_channels,
            out_channels,
            kernel_size,
            padding,
        }
    }

    /// Disable the bias parameter.
    #[must_use]
    pub fn no_bias(mut self) -> Self {
        self.bias = None;
        self
    }

    /// Input-channel count.
    pub fn in_channels(&self) -> usize {
        self.in_channels
    }
    /// Output-channel count.
    pub fn out_channels(&self) -> usize {
        self.out_channels
    }
    /// Kernel size (kH, kW).
    pub fn kernel_size(&self) -> (usize, usize) {
        self.kernel_size
    }
}

/// 1-D convolution. Layout `[N, C_in, L]`. Implemented as a Conv2d on
/// `[N, C_in, 1, L]` with kernel `(1, kW)` for v1 — semantically
/// identical and reuses the existing Conv2d kernels.
pub struct Conv1d {
    /// Underlying Conv2d.
    inner: Conv2d,
    kernel_size: usize,
}

impl Conv1d {
    /// Build a Conv1d.
    pub fn new(in_channels: usize, out_channels: usize, kernel_size: usize) -> Self {
        let inner = Conv2d::with_padding(in_channels, out_channels, (1, kernel_size), (0, 0));
        Conv1d { inner, kernel_size }
    }

    /// Build with explicit padding.
    pub fn with_padding(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        padding: usize,
    ) -> Self {
        let inner = Conv2d::with_padding(in_channels, out_channels, (1, kernel_size), (0, padding));
        Conv1d { inner, kernel_size }
    }

    /// Disable bias.
    #[must_use]
    pub fn no_bias(mut self) -> Self {
        self.inner = self.inner.no_bias();
        self
    }

    /// Kernel size.
    pub fn kernel_size(&self) -> usize {
        self.kernel_size
    }
}

impl Module for Conv1d {
    fn forward(&self, input: &Variable) -> Result<Variable, ModuleError> {
        // Reshape [N, C, L] → [N, C, 1, L], conv, then reshape back.
        let s = input.tensor().shape().to_vec();
        if s.len() != 3 {
            return Err(rustorch_autograd::BackwardError::Backend {
                op: "conv1d",
                message: format!("expected rank-3 input [N, C, L], got {s:?}"),
            });
        }
        let (n, c, l) = (s[0], s[1], s[2]);
        let x_4d = ops::reshape(input, vec![n, c, 1, l])?;
        let y_4d = self.inner.forward(&x_4d)?;
        let s2 = y_4d.tensor().shape().to_vec();
        let (n2, c2, _, l2) = (s2[0], s2[1], s2[2], s2[3]);
        ops::reshape(&y_4d, vec![n2, c2, l2])
    }

    fn parameters(&self) -> Vec<Variable> {
        self.inner.parameters()
    }

    fn named_parameters(&self) -> Vec<(String, Variable)> {
        self.inner.named_parameters()
    }
}

/// 2-D max-pooling layer (stateless, no parameters).
pub struct MaxPool2d {
    kernel_size: (usize, usize),
    stride: (usize, usize),
}

impl MaxPool2d {
    /// Build a square-kernel max-pool with matching stride.
    pub fn new(kernel_size: usize) -> Self {
        MaxPool2d {
            kernel_size: (kernel_size, kernel_size),
            stride: (kernel_size, kernel_size),
        }
    }

    /// Build with explicit kernel and stride.
    pub fn with_stride(kernel_size: (usize, usize), stride: (usize, usize)) -> Self {
        MaxPool2d {
            kernel_size,
            stride,
        }
    }
}

impl Module for MaxPool2d {
    fn forward(&self, input: &Variable) -> Result<Variable, ModuleError> {
        ops::max_pool2d(input, self.kernel_size, self.stride)
    }
}

impl Module for Conv2d {
    fn forward(&self, input: &Variable) -> Result<Variable, ModuleError> {
        ops::conv2d(
            input,
            &self.weight,
            self.bias.as_ref(),
            self.padding.0,
            self.padding.1,
        )
    }

    fn parameters(&self) -> Vec<Variable> {
        let mut p = vec![self.weight.clone()];
        if let Some(b) = &self.bias {
            p.push(b.clone());
        }
        p
    }

    fn named_parameters(&self) -> Vec<(String, Variable)> {
        let mut out = vec![("weight".to_string(), self.weight.clone())];
        if let Some(b) = &self.bias {
            out.push(("bias".to_string(), b.clone()));
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustorch_autograd::{backward, ops};

    #[test]
    fn conv2d_shape_3x3_no_padding() {
        let conv = Conv2d::new(1, 2, 3);
        let x = Variable::new(
            Tensor::from_vec([1usize, 1, 5, 5], (0..25).map(|i| i as f32).collect()).unwrap(),
        );
        let y = conv.forward(&x).unwrap();
        assert_eq!(y.tensor().shape(), &[1, 2, 3, 3]); // (5-3+1) = 3
    }

    #[test]
    fn conv2d_padding_preserves_spatial_dims() {
        let conv = Conv2d::with_padding(3, 4, (3, 3), (1, 1));
        let x = Variable::new(
            Tensor::from_vec([2usize, 3, 8, 8], vec![0.5_f32; 2 * 3 * 8 * 8]).unwrap(),
        );
        let y = conv.forward(&x).unwrap();
        assert_eq!(y.tensor().shape(), &[2, 4, 8, 8]);
    }

    #[test]
    fn conv2d_no_bias_only_one_param() {
        let conv = Conv2d::new(2, 3, 1).no_bias();
        assert_eq!(conv.parameters().len(), 1);
        let np = conv.named_parameters();
        assert_eq!(np.len(), 1);
        assert_eq!(np[0].0, "weight");
    }

    #[test]
    fn conv2d_backward_grads_flow_to_weight_and_input() {
        let conv = Conv2d::new(1, 1, 3);
        let x = Variable::leaf(
            Tensor::from_vec([1usize, 1, 5, 5], (0..25).map(|i| i as f32 * 0.1).collect()).unwrap(),
        );
        let y = conv.forward(&x).unwrap();
        let s = ops::sum(&y).unwrap();
        backward(&s, None).unwrap();
        // x grad
        assert!(x
            .grad()
            .unwrap()
            .as_slice::<f32>()
            .unwrap()
            .iter()
            .any(|v| v.abs() > 1e-6));
        // weight grad
        let w_grad = conv.weight.grad().unwrap();
        assert_eq!(w_grad.shape(), &[1, 1, 3, 3]);
        assert!(w_grad
            .as_slice::<f32>()
            .unwrap()
            .iter()
            .any(|v| v.abs() > 1e-6));
        // bias grad
        let b_grad = conv.bias.as_ref().unwrap().grad().unwrap();
        assert_eq!(b_grad.shape(), &[1]);
    }

    #[test]
    fn conv2d_named_parameters() {
        let conv = Conv2d::new(3, 8, 3);
        let np = conv.named_parameters();
        let names: Vec<String> = np.iter().map(|(n, _)| n.clone()).collect();
        assert!(names.contains(&"weight".to_string()));
        assert!(names.contains(&"bias".to_string()));
    }

    #[test]
    fn maxpool2d_2x2_halves_spatial() {
        let pool = MaxPool2d::new(2);
        let x = Variable::new(
            Tensor::from_vec([1usize, 1, 4, 4], (0..16).map(|i| i as f32).collect()).unwrap(),
        );
        let y = pool.forward(&x).unwrap();
        assert_eq!(y.tensor().shape(), &[1, 1, 2, 2]);
    }

    #[test]
    fn maxpool2d_backward_scatters_to_argmax() {
        let pool = MaxPool2d::new(2);
        let x = Variable::leaf(
            Tensor::from_vec([1usize, 1, 2, 2], vec![1.0_f32, 5.0, 3.0, 2.0]).unwrap(),
        );
        let y = pool.forward(&x).unwrap();
        let s = ops::sum(&y).unwrap();
        backward(&s, None).unwrap();
        let g = x.grad().unwrap();
        // Grad should land only at the argmax position (index 1, value 5.0).
        assert_eq!(g.as_slice::<f32>().unwrap(), &[0.0_f32, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn conv2d_then_maxpool_chains_correctly() {
        // Conv2d(1→2, 3x3) → MaxPool2d(2x2). Output should be sensible.
        let conv = Conv2d::new(1, 2, 3);
        let pool = MaxPool2d::new(2);
        let x = Variable::new(Tensor::from_vec([1usize, 1, 6, 6], vec![1.0_f32; 36]).unwrap());
        let h = conv.forward(&x).unwrap();
        // After conv: [1, 2, 4, 4]
        assert_eq!(h.tensor().shape(), &[1, 2, 4, 4]);
        let y = pool.forward(&h).unwrap();
        // After pool: [1, 2, 2, 2]
        assert_eq!(y.tensor().shape(), &[1, 2, 2, 2]);
    }
}
