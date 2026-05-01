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

pub mod backend;
pub mod cache;
pub mod error;
pub mod storage;
pub mod transfer;

pub use backend::WgpuBackend;
pub use error::WgpuError;
pub use storage::WgpuStorage;
pub use transfer::{to_cpu, to_gpu};

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
