//! # rustorch-wgpu
//!
//! Cross-platform GPU backend for rustorch via [wgpu] (WebGPU/Vulkan/
//! Metal/DX12). The same crate compiles for native targets (Linux,
//! macOS, Windows) and `wasm32-unknown-unknown` (browser WebGPU).
//!
//! v1 ships:
//! - [`WgpuBackend`] holding device/queue/limits + a pipeline cache.
//! - [`WgpuStorage`] wrapping `wgpu::Buffer` with refcount.
//! - Round-trip transfer helpers `to_gpu` / `to_cpu` between
//!   `rustorch_core::Tensor` and `WgpuStorage`.
//!
//! Kernels (element-wise, matmul, reductions, conv) ship in P2.2+.
//!
//! [wgpu]: https://wgpu.rs

#![cfg_attr(docsrs, feature(doc_cfg))]
#![warn(missing_docs)]
#![warn(rust_2018_idioms)]

pub mod argmax;
pub mod attention;
pub mod backend;
pub mod backend_impl;
pub mod backend_singleton;
pub mod broadcast;
pub mod cache;
pub mod capabilities;
pub mod conv;
pub mod elementwise;
pub mod error;
pub mod flash_attn;
pub mod fused;
pub mod fused_adamw;
pub mod layernorm;
pub mod mapped_upload;
pub mod matmul;
// `pooled::PooledBuffer` was removed in P3.Z Task A — its
// return-to-pool semantics are now expressed via the `on_drop`
// callback in `rustorch_core::tensor::storage::WgpuStorage`,
// captured by `crate::storage::WgpuStorage::allocate_pooled`.
pub mod preprocessor;
pub mod reduce;
pub mod registry;
pub mod runtime;
pub mod shaders;
pub mod softmax;
pub mod staging;
pub mod storage;
pub mod template;
pub mod transfer;
pub mod transpose;

pub use argmax::{argmax_rows, ArgKind};
pub use attention::{
    apply_causal_mask, attention_naive, attention_naive_causal, mul_scalar, multi_head_attention,
};
pub use backend::WgpuBackend;
pub use backend_singleton::{try_wgpu_backend, wgpu_backend};
pub use broadcast::{broadcast_shape, dispatch_binary_broadcast};
pub use capabilities::Capabilities;
pub use conv::{conv2d_forward, transpose_weight, Conv2dCfg};
pub use elementwise::{dispatch_binary, dispatch_unary};
pub use error::WgpuError;
pub use flash_attn::flash_attention;
pub use fused::{linear_gelu_fused, linear_relu_fused, FusedAct};
pub use layernorm::{layernorm_rows, layernorm_welford_rows, rmsnorm_rows};
pub use mapped_upload::upload_mapped_at_creation;
pub use matmul::{matmul, matmul_with_transposes};
pub use preprocessor::{expand_includes, expand_macros, PreprocessError};
pub use reduce::{reduce_rows, ReduceKind};
pub use registry::{KernelRegistry, OpId};
pub use runtime::{block_on, submit_and_wait};
pub use softmax::{log_softmax_axis, log_softmax_rows, softmax_axis, softmax_rows};
pub use staging::StagingRing;
pub use storage::WgpuStorage;
pub use template::{validate_all_kernels, validate_wgsl};
pub use transfer::{to_cpu, to_cpu_async, to_gpu};
pub use transpose::transpose2d;

/// Crate version reported at runtime.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smoke_version_present() {
        assert!(!VERSION.is_empty());
    }
}
