//! Backend error type — shared across rustorch-cpu and downstream backends
//! (RFC-0004, P1.2 task `BackendError`).
//!
//! Every method on the [`Backend`](crate::backend::Backend) trait returns
//! `Result<T, BackendError>`. Variants are precise enough to drive
//! human-readable error messages that match PyTorch's wording style
//! ("Sizes of tensors must match...").

use core::fmt;
use rustorch_core::tensor::dtype::Dtype;
use rustorch_core::tensor::storage::StorageError;

/// Errors returned by any [`Backend`](crate::backend::Backend) method.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BackendError {
    /// Two operands have shapes that cannot be jointly used for `op`.
    ShapeMismatch {
        /// Op name (`"add"`, `"matmul"`, …).
        op: &'static str,
        /// Left operand shape.
        lhs: Vec<usize>,
        /// Right operand shape.
        rhs: Vec<usize>,
    },
    /// Two operands have incompatible dtypes.
    DtypeMismatch {
        /// Op name.
        op: &'static str,
        /// LHS dtype.
        lhs: Dtype,
        /// RHS dtype.
        rhs: Dtype,
    },
    /// Buffer allocation failed for `bytes` bytes.
    OutOfMemory {
        /// Bytes requested.
        bytes: usize,
    },
    /// Op exists in the trait but isn't implemented for this `device`.
    UnsupportedOp {
        /// Op name.
        op: &'static str,
        /// Device name (`"cpu"`, `"wgpu"`, `"cuda"`).
        device: &'static str,
    },
    /// IEEE-754 anomaly produced by a kernel that opts in to anomaly
    /// detection (NaN / Inf in `requires_grad` paths).
    NumericalError(String),
    /// Out-of-range index in `gather`, `index_select`, `slice`, ...
    IndexOutOfBounds {
        /// Op name.
        op: &'static str,
        /// Bad index value.
        index: i64,
        /// Allowed range size on the indexed axis.
        bound: usize,
    },
    /// In-place op was attempted on storage with multiple owners or on
    /// an alias of one of the operands.
    AliasingViolation {
        /// Op name.
        op: &'static str,
    },
}

impl BackendError {
    /// Cap shape display length so very large shapes don't overflow log
    /// lines. Returns the original shape unchanged if `<= 8` dims.
    fn fmt_shape(shape: &[usize]) -> String {
        if shape.len() <= 8 {
            format!("{shape:?}")
        } else {
            let head = &shape[..4];
            let tail = &shape[shape.len() - 4..];
            format!("[{head:?}…{tail:?} (rank {})]", shape.len())
        }
    }
}

impl fmt::Display for BackendError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BackendError::ShapeMismatch { op, lhs, rhs } => write!(
                f,
                "{op}: sizes of tensors must match: {} vs {}",
                Self::fmt_shape(lhs),
                Self::fmt_shape(rhs)
            ),
            BackendError::DtypeMismatch { op, lhs, rhs } => {
                write!(f, "{op}: dtype mismatch: {lhs} vs {rhs}")
            },
            BackendError::OutOfMemory { bytes } => {
                write!(f, "out of memory allocating {bytes} bytes")
            },
            BackendError::UnsupportedOp { op, device } => {
                write!(f, "{op} is not supported on device {device}")
            },
            BackendError::NumericalError(msg) => write!(f, "numerical error: {msg}"),
            BackendError::IndexOutOfBounds { op, index, bound } => {
                write!(f, "{op}: index {index} out of bounds (axis size {bound})")
            },
            BackendError::AliasingViolation { op } => {
                write!(f, "{op}: in-place op forbidden when storage is aliased")
            },
        }
    }
}

impl std::error::Error for BackendError {}

impl From<StorageError> for BackendError {
    fn from(e: StorageError) -> Self {
        match e {
            StorageError::OutOfMemory { byte_len } => BackendError::OutOfMemory { bytes: byte_len },
            StorageError::InvalidAlloc { byte_len, .. } => {
                BackendError::OutOfMemory { bytes: byte_len }
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shape_mismatch_display() {
        let e = BackendError::ShapeMismatch {
            op: "add",
            lhs: vec![2, 3],
            rhs: vec![3, 2],
        };
        let s = e.to_string();
        assert!(s.contains("add"));
        assert!(s.contains("[2, 3]"));
        assert!(s.contains("[3, 2]"));
        assert!(s.contains("must match"));
    }

    #[test]
    fn shape_mismatch_long_shape_truncated() {
        let big: Vec<usize> = (0..20).collect();
        let e = BackendError::ShapeMismatch {
            op: "add",
            lhs: big.clone(),
            rhs: big,
        };
        let s = e.to_string();
        // No panic, output bounded
        assert!(s.contains("rank 20"));
    }

    #[test]
    fn oom_display_includes_bytes() {
        let e = BackendError::OutOfMemory { bytes: 4096 };
        assert!(e.to_string().contains("4096"));
    }

    #[test]
    fn dtype_mismatch_display() {
        let e = BackendError::DtypeMismatch {
            op: "matmul",
            lhs: Dtype::F32,
            rhs: Dtype::I64,
        };
        let s = e.to_string();
        assert!(s.contains("matmul"));
        assert!(s.contains("f32"));
        assert!(s.contains("i64"));
    }

    #[test]
    fn unsupported_op_mentions_device() {
        let e = BackendError::UnsupportedOp {
            op: "qr",
            device: "wgpu",
        };
        assert!(e.to_string().contains("qr"));
        assert!(e.to_string().contains("wgpu"));
    }

    #[test]
    fn numerical_error_passes_message() {
        let e = BackendError::NumericalError("NaN in matmul".into());
        assert!(e.to_string().contains("NaN"));
    }

    #[test]
    fn index_out_of_bounds() {
        let e = BackendError::IndexOutOfBounds {
            op: "gather",
            index: -1,
            bound: 5,
        };
        let s = e.to_string();
        assert!(s.contains("gather"));
        assert!(s.contains("-1"));
        assert!(s.contains("5"));
    }

    #[test]
    fn aliasing_violation() {
        let e = BackendError::AliasingViolation { op: "add_" };
        assert!(e.to_string().contains("add_"));
        assert!(e.to_string().contains("aliased"));
    }

    #[test]
    fn from_storage_error_oom() {
        let s = StorageError::OutOfMemory { byte_len: 1234 };
        let b: BackendError = s.into();
        assert_eq!(b, BackendError::OutOfMemory { bytes: 1234 });
    }

    #[test]
    fn from_storage_error_invalid_alloc_maps_to_oom() {
        let s = StorageError::InvalidAlloc {
            byte_len: 9999,
            align: 64,
        };
        let b: BackendError = s.into();
        assert_eq!(b, BackendError::OutOfMemory { bytes: 9999 });
    }

    #[test]
    fn debug_format_smoke() {
        let _ = format!(
            "{:?}",
            BackendError::ShapeMismatch {
                op: "add",
                lhs: vec![1],
                rhs: vec![2]
            }
        );
    }
}
