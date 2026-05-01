//! `Backend` trait — the dispatch surface for tensor ops
//! (RFC-0004, P1.2 task `Backend trait definition`).
//!
//! v1 ships a *minimal* trait surface (~12 methods) covering the ops
//! that P1.3 (elementary), P1.4 (linalg/conv/reductions) and P1.6 (nn)
//! actually call. The trait is `dyn`-safe (`&dyn Backend`) and
//! `Send + Sync`; the registry returns a `&'static dyn Backend`
//! singleton per device.
//!
//! Adding a new method to this trait is a non-breaking change in the
//! Rust sense as long as the method comes with a default impl that
//! returns `BackendError::UnsupportedOp` — that lets out-of-tree GPU
//! backends roll out incrementally.

use crate::error::BackendError;
use rustorch_core::tensor::tensor_impl::Tensor;

/// Single dispatch surface for tensor ops on a given device.
pub trait Backend: Send + Sync {
    /// Stable name of the device — `"cpu"`, `"wgpu"`, `"cuda"`, …
    fn name(&self) -> &'static str;

    /// Element-wise add: `out = lhs + rhs` (broadcast).
    fn add(&self, lhs: &Tensor, rhs: &Tensor) -> Result<Tensor, BackendError>;

    /// Element-wise sub: `out = lhs - rhs` (broadcast).
    fn sub(&self, lhs: &Tensor, rhs: &Tensor) -> Result<Tensor, BackendError>;

    /// Element-wise mul: `out = lhs * rhs` (broadcast).
    fn mul(&self, lhs: &Tensor, rhs: &Tensor) -> Result<Tensor, BackendError>;

    /// Element-wise div: `out = lhs / rhs` (broadcast).
    fn div(&self, lhs: &Tensor, rhs: &Tensor) -> Result<Tensor, BackendError>;

    /// Element-wise unary negation.
    fn neg(&self, src: &Tensor) -> Result<Tensor, BackendError>;

    /// Matrix multiplication on rank-2 tensors.
    fn matmul(&self, lhs: &Tensor, rhs: &Tensor) -> Result<Tensor, BackendError>;

    /// Sum over all elements.
    fn sum(&self, _src: &Tensor) -> Result<Tensor, BackendError> {
        Err(BackendError::UnsupportedOp {
            op: "sum",
            device: self.name(),
        })
    }

    /// Mean over all elements.
    fn mean(&self, _src: &Tensor) -> Result<Tensor, BackendError> {
        Err(BackendError::UnsupportedOp {
            op: "mean",
            device: self.name(),
        })
    }

    /// ReLU activation.
    fn relu(&self, src: &Tensor) -> Result<Tensor, BackendError>;

    /// Equality (element-wise) → Bool tensor.
    fn eq(&self, lhs: &Tensor, rhs: &Tensor) -> Result<Tensor, BackendError>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::sync::atomic::AtomicUsize;

    /// A fixed-output stub that's never asked to compute anything beyond
    /// `name`. Used to verify the trait is dyn-safe and Send + Sync.
    struct StubBackend;

    impl Backend for StubBackend {
        fn name(&self) -> &'static str {
            "stub"
        }
        fn add(&self, _: &Tensor, _: &Tensor) -> Result<Tensor, BackendError> {
            Ok(Tensor::scalar(0.0))
        }
        fn sub(&self, _: &Tensor, _: &Tensor) -> Result<Tensor, BackendError> {
            Ok(Tensor::scalar(0.0))
        }
        fn mul(&self, _: &Tensor, _: &Tensor) -> Result<Tensor, BackendError> {
            Ok(Tensor::scalar(0.0))
        }
        fn div(&self, _: &Tensor, _: &Tensor) -> Result<Tensor, BackendError> {
            Ok(Tensor::scalar(0.0))
        }
        fn neg(&self, _: &Tensor) -> Result<Tensor, BackendError> {
            Ok(Tensor::scalar(0.0))
        }
        fn matmul(&self, _: &Tensor, _: &Tensor) -> Result<Tensor, BackendError> {
            Ok(Tensor::scalar(0.0))
        }
        fn relu(&self, _: &Tensor) -> Result<Tensor, BackendError> {
            Ok(Tensor::scalar(0.0))
        }
        fn eq(&self, _: &Tensor, _: &Tensor) -> Result<Tensor, BackendError> {
            Ok(Tensor::scalar(0.0))
        }
    }

    #[test]
    fn dyn_backend_is_object_safe() {
        let s: &dyn Backend = &StubBackend;
        assert_eq!(s.name(), "stub");
    }

    #[test]
    fn default_sum_returns_unsupported() {
        let s: &dyn Backend = &StubBackend;
        let t = Tensor::scalar(1.0);
        let err = s.sum(&t).unwrap_err();
        assert!(matches!(
            err,
            BackendError::UnsupportedOp {
                op: "sum",
                device: "stub"
            }
        ));
    }

    #[test]
    fn default_mean_returns_unsupported() {
        let s: &dyn Backend = &StubBackend;
        let t = Tensor::scalar(1.0);
        let err = s.mean(&t).unwrap_err();
        assert!(matches!(
            err,
            BackendError::UnsupportedOp {
                op: "mean",
                device: "stub"
            }
        ));
    }

    #[test]
    fn backend_send_sync_compiles() {
        // Compile-time check that any &'static dyn Backend can cross threads.
        fn assert_send<T: Send>() {}
        fn assert_sync<T: Sync>() {}
        assert_send::<&'static dyn Backend>();
        assert_sync::<&'static dyn Backend>();
    }

    #[test]
    fn shared_arc_dyn_backend_works() {
        // We expose backends via Arc<dyn Backend> in the registry — assert
        // that pattern compiles + behaves.
        use std::sync::Arc;
        let b: Arc<dyn Backend> = Arc::new(StubBackend);
        assert_eq!(b.name(), "stub");
        let _counter = AtomicUsize::new(0);
    }
}
