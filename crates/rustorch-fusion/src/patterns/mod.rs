//! Hand-fused kernel library.

pub mod elementwise;
pub mod matmul_bias_act;
pub mod norm_linear;

pub use elementwise::elementwise_chain;
pub use matmul_bias_act::{
    fused_matmul_bias_activation, naive_matmul_bias_activation, Activation, FusionError,
};
pub use norm_linear::{layernorm_linear, residual_add_scale_mul};
