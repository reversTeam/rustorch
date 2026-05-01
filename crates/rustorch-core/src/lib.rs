//! # rustorch-core — Tensor / Storage / Layout / dtypes (P1.1)
//!
//! This crate is the zero-ML foundation for rustorch. It exposes:
//!
//! - [`Tensor`] — multi-dim array handle composing `Storage`, `Layout`,
//!   `VersionCounter`, and autograd metadata flags.
//! - [`tensor::Dtype`] — 8-variant element type tag matching
//!   safetensors and PyTorch's surface (F32, F64, F16, BF16, I64, I32,
//!   I8, Bool).
//! - [`tensor::Element`] — sealed trait bridging Rust scalar types to
//!   `Dtype`.
//! - [`tensor::Shape`] / [`tensor::Layout`] / [`tensor::Storage`] /
//!   [`tensor::VersionCounter`].
//!
//! The full design is documented in RFC-0002 (Tensor design) and
//! RFC-0004 (Backend trait). The P0.3 prototype's "Tensor over
//! `Vec<f32>`" has been replaced by this module-driven version.

#![cfg_attr(docsrs, feature(doc_cfg))]
#![warn(missing_docs)]
#![warn(rust_2018_idioms)]

mod format;
pub mod ops;
pub mod tensor;

pub use tensor::tensor_impl::{Tensor, TensorError};

/// Crate version reported at runtime.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

// --------------------------------------------------------------------------
// Legacy prototype error type — kept for the existing op-level Error variants
// (ShapeMismatch, Rank). These will fold into a unified `RusTorchError` enum
// in P1.5/P1.6.
// --------------------------------------------------------------------------

/// Errors produced by the rustorch-core ops.
///
/// `ShapeDataMismatch` is now produced via [`TensorError`] inside
/// constructors — kept here as a type alias-friendly variant for ops
/// transitioning incrementally.
#[derive(Debug, Clone, PartialEq)]
pub enum Error {
    /// Constructor was given a `data` of wrong length for the requested
    /// shape.
    ShapeDataMismatch {
        /// Expected length = product of the requested shape.
        expected: usize,
        /// Length of the `data` slice the user actually passed.
        got: usize,
    },
    /// Op was called with shapes that cannot be combined.
    ShapeMismatch {
        /// Name of the op for the user-visible error message.
        op: &'static str,
        /// Shape of the first operand.
        lhs: Vec<usize>,
        /// Shape of the second operand.
        rhs: Vec<usize>,
    },
    /// Op required N-D inputs but got something else.
    Rank {
        /// Name of the op.
        op: &'static str,
        /// Rank actually received.
        got: usize,
        /// Human-readable description of the expected rank.
        want: &'static str,
    },
    /// Wrap a `TensorError` (e.g. shape/data mismatch on construction).
    Tensor(TensorError),
}

impl From<TensorError> for Error {
    fn from(e: TensorError) -> Self {
        Error::Tensor(e)
    }
}

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Error::ShapeDataMismatch { expected, got } => {
                write!(
                    f,
                    "shape product != data len: expected {expected}, got {got}"
                )
            },
            Error::ShapeMismatch { op, lhs, rhs } => {
                write!(f, "{op}: incompatible shapes {lhs:?} vs {rhs:?}")
            },
            Error::Rank { op, got, want } => {
                write!(f, "{op}: rank mismatch (got {got}, want {want})")
            },
            Error::Tensor(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for Error {}

/// Convenience alias used throughout the crate.
pub type Result<T> = std::result::Result<T, Error>;

// --------------------------------------------------------------------------
// Smoke tests at the crate root
// --------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smoke_version_present() {
        assert!(!VERSION.is_empty());
    }

    #[test]
    fn re_export_tensor_api() {
        let t = Tensor::from_vec([2usize, 3], vec![1.0_f32; 6]).unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.len(), 6);
        assert_eq!(t.ndim(), 2);
    }

    #[test]
    fn error_from_tensor_error() {
        let inner = Tensor::from_vec([2usize, 3], vec![1.0_f32; 5]).unwrap_err();
        let e: Error = inner.into();
        // Display still works
        let _ = e.to_string();
    }
}
