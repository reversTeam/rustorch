//! # rustorch
//!
//! Pure Rust port of PyTorch — Tensor library + autograd + nn modules + optimizers
//! + multi-backend (CPU, wgpu/WebGPU, CUDA optional) + WASM-first.
//!
//! This is the public umbrella crate users add to their Cargo.toml. It re-exports
//! the most useful symbols from the underlying modular crates so that user code
//! reads as if everything lived in one place — exactly like `torch` in Python.
//!
//! ```
//! // Example aligned with the v0.7.2 user docs:
//! //
//! //   use rustorch::prelude::*;
//! //   let dev = Device::cuda_if_available()?;
//! //   let x = Tensor::zeros::<f32>([2, 3], &dev)?;
//! //   let net = Sequential::new()
//! //       .add(Linear::new(784, 256).build(&dev)?)
//! //       .add(ReLU);
//! //
//! // The actual symbols become available as their host crates are filled in.
//! ```

#![cfg_attr(docsrs, feature(doc_cfg))]
#![warn(rust_2018_idioms)]

// Re-export sub-crates as modules so users can write `rustorch::nn::Linear`
// or `rustorch::optim::AdamW` once the implementations land.
pub use rustorch_autograd as autograd;
pub use rustorch_core as core;
pub use rustorch_cpu as cpu;
pub use rustorch_data as data;
pub use rustorch_nn as nn;
pub use rustorch_optim as optim;
pub use rustorch_serde as serde;

/// Curated re-exports — the everyday surface a user pulls in via `use rustorch::prelude::*;`.
///
/// Empty for now — populated as P1.x lands the actual `Tensor`, `Module`, etc.
pub mod prelude {}

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
