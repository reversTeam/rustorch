//! # rustorch-cpu — CPU backend for rustorch (P1.2).
//!
//! This crate provides:
//!
//! - [`error::BackendError`] — unified error type returned by every
//!   backend method.
//! - [`parallel::parallel_for`] — rayon-based work-splitter with a
//!   serial WASM fallback.
//! - [`allocator::CpuAllocator`] — re-export of the 64B-aligned
//!   storage allocator.
//! - [`simd::Simd`] — scalar SIMD shim (auto-vectorized; pulp/wide
//!   integration deferred).
//! - [`iterator`] — TensorIterator (broadcasting + map_unary /
//!   map_binary).
//! - [`backend::Backend`] — dispatch trait.
//! - [`cpu_backend::CpuBackend`] — the concrete CPU `Backend` impl.
//! - [`mock::MockBackend`] — test double recording every call.
//! - [`profile::Profiler`] — feature-gated profiler hooks.

#![cfg_attr(docsrs, feature(doc_cfg))]
#![warn(missing_docs)]
#![warn(rust_2018_idioms)]

#[cfg(target_os = "macos")]
pub mod accelerate;
pub mod allocator;
pub mod backend;
pub mod cpu_backend;
pub mod error;
pub mod iterator;
pub mod kernels;
pub mod mock;
pub mod parallel;
pub mod pinned;
pub mod profile;
pub mod quant;
pub mod simd;

pub use backend::Backend;
pub use cpu_backend::{cpu_backend, CpuBackend};
pub use error::BackendError;
pub use mock::{CallRecord, MockBackend};

/// Crate version reported at runtime.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smoke_version_present() {
        assert!(!VERSION.is_empty());
    }

    #[test]
    fn cpu_backend_singleton_works() {
        let b = cpu_backend();
        assert_eq!(b.name(), "cpu");
    }
}
