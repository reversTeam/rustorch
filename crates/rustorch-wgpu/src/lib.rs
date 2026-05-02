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
pub mod cache;
pub mod conv;
pub mod elementwise;
pub mod error;
pub mod flash_attn;
pub mod fused;
pub mod layernorm;
pub mod mapped_upload;
pub mod matmul;
pub mod pooled;
pub mod reduce;
pub mod shaders;
pub mod softmax;
pub mod staging;
pub mod storage;
pub mod transfer;
pub mod transpose;

pub use argmax::{argmax_rows, ArgKind};
pub use attention::{attention_naive, mul_scalar};
pub use backend::WgpuBackend;
pub use conv::{conv2d_forward, transpose_weight, Conv2dCfg};
pub use elementwise::{dispatch_binary, dispatch_unary};
pub use error::WgpuError;
pub use flash_attn::flash_attention;
pub use fused::{linear_gelu_fused, linear_relu_fused, FusedAct};
pub use layernorm::{layernorm_rows, rmsnorm_rows};
pub use mapped_upload::upload_mapped_at_creation;
pub use matmul::matmul;
pub use reduce::{reduce_rows, ReduceKind};
pub use softmax::{log_softmax_rows, softmax_rows};
pub use staging::StagingRing;
pub use storage::WgpuStorage;
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
