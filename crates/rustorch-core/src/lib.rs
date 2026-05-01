//! # rustorch-core — prototype
//!
//! P0.3 prototype: a deliberately minimal `Tensor` over `Vec<f32>` with a
//! handful of ops (`add` with 1D broadcasting, `matmul` 2-D), a PyTorch-style
//! `Display` impl, and a `Result`-typed error path.
//!
//! This crate is the prototype that **validates the design** acted in
//! RFCs 0001–0006. The full Tensor (multi-dtype, multi-backend, autograd)
//! lands in P1.1+ and replaces this file in place.
//!
//! No autograd, no backend abstraction, no dtypes other than f32.
//! Total LOC budget for the prototype crate: ≤ 300.

#![cfg_attr(docsrs, feature(doc_cfg))]
#![warn(missing_docs)]
#![warn(rust_2018_idioms)]

mod format;
pub mod ops;
pub mod tensor;

use std::fmt;

/// Crate version reported at runtime.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

// --------------------------------------------------------------------------
// Error type — Result-based, no panics from the public API in the steady state.
// --------------------------------------------------------------------------

/// Errors produced by the rustorch-core prototype.
#[derive(Debug, Clone, PartialEq)]
pub enum Error {
    /// Constructor was given a `data` of wrong length for the requested shape.
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
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
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
        }
    }
}

impl std::error::Error for Error {}

/// Convenience alias used throughout the crate.
pub type Result<T> = std::result::Result<T, Error>;

// --------------------------------------------------------------------------
// Tensor — header + Vec<f32> body. Refcount via Vec's Clone-on-write semantics
// is intentionally NOT used here (full design uses Arc<Storage> in P1.1).
// --------------------------------------------------------------------------

/// A minimal f32 tensor for prototype validation.
///
/// In Phase 1 this becomes `Tensor` from RFC-0002 with full Storage / Layout /
/// Dtype / autograd metadata; the public surface (`zeros`, `from_vec`,
/// `shape`, `len`) is forward-compatible.
#[derive(Clone, PartialEq)]
pub struct Tensor {
    shape: Vec<usize>,
    data: Vec<f32>,
}

impl Tensor {
    /// Build a tensor from an explicit shape and a contiguous row-major
    /// `data` slice. Validates `data.len() == product(shape)`.
    pub fn from_vec(shape: impl Into<Vec<usize>>, data: Vec<f32>) -> Result<Tensor> {
        let shape = shape.into();
        let expected = numel(&shape);
        if data.len() != expected {
            return Err(Error::ShapeDataMismatch {
                expected,
                got: data.len(),
            });
        }
        Ok(Tensor { shape, data })
    }

    /// `Tensor::zeros([2, 3])` — all zeros, row-major contiguous.
    pub fn zeros(shape: impl Into<Vec<usize>>) -> Tensor {
        let shape = shape.into();
        let n = numel(&shape);
        Tensor {
            shape,
            data: vec![0.0; n],
        }
    }

    /// `Tensor::ones([2, 3])` — all ones.
    pub fn ones(shape: impl Into<Vec<usize>>) -> Tensor {
        let shape = shape.into();
        let n = numel(&shape);
        Tensor {
            shape,
            data: vec![1.0; n],
        }
    }

    /// Build a 0-dimensional (scalar) tensor.
    pub fn scalar(v: f32) -> Tensor {
        Tensor {
            shape: vec![],
            data: vec![v],
        }
    }

    /// Shape — read-only view into the dimensions.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Total number of elements (product of `shape`, with empty shape == 1
    /// for 0-d / scalar tensors).
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// `true` if `len() == 0` (only possible when some axis is 0).
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Flat row-major data slice.
    pub fn data(&self) -> &[f32] {
        &self.data
    }

    /// Number of dimensions.
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }
}

/// Compute the product of a shape, handling the empty (0-d / scalar) case
/// as length 1 — same as `numpy.prod([])`.
fn numel(shape: &[usize]) -> usize {
    if shape.is_empty() {
        1
    } else {
        shape.iter().product()
    }
}

// --------------------------------------------------------------------------
// Tests
// --------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smoke_version_present() {
        assert!(!VERSION.is_empty());
    }

    #[test]
    fn from_vec_ok() {
        let t = Tensor::from_vec([2usize, 3], vec![1.0; 6]).unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.len(), 6);
        assert_eq!(t.ndim(), 2);
    }

    #[test]
    fn zeros_and_ones() {
        let z = Tensor::zeros([2usize, 3]);
        assert_eq!(z.data(), &[0.0; 6]);
        let o = Tensor::ones([2usize, 3]);
        assert_eq!(o.data(), &[1.0; 6]);
    }

    #[test]
    fn scalar_is_0d_with_len_1() {
        let s = Tensor::scalar(2.71);
        assert_eq!(s.shape(), &[] as &[usize]);
        assert_eq!(s.len(), 1);
        assert_eq!(s.data(), &[2.71]);
    }

    #[test]
    fn empty_tensor() {
        let t = Tensor::zeros([0usize]);
        assert_eq!(t.shape(), &[0]);
        assert_eq!(t.len(), 0);
        assert!(t.is_empty());
    }

    #[test]
    fn shape_data_mismatch_returns_err() {
        let err = Tensor::from_vec([2usize, 3], vec![1.0; 5]).unwrap_err();
        assert_eq!(
            err,
            Error::ShapeDataMismatch {
                expected: 6,
                got: 5
            }
        );
        // Display also reasonable
        assert_eq!(
            err.to_string(),
            "shape product != data len: expected 6, got 5",
        );
    }
}
