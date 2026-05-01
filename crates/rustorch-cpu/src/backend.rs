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

    // -------------------- P1.3 — extra elementary ops --------------------
    //
    // Each extra method has a default Unsupported impl. Backends opt in by
    // overriding; downstream ops degrade gracefully via `Result<_, BackendError>`.

    /// Element-wise absolute value.
    fn abs(&self, _src: &Tensor) -> Result<Tensor, BackendError> {
        Err(unsupported("abs", self.name()))
    }
    /// Element-wise square root.
    fn sqrt(&self, _src: &Tensor) -> Result<Tensor, BackendError> {
        Err(unsupported("sqrt", self.name()))
    }
    /// Element-wise exponential.
    fn exp(&self, _src: &Tensor) -> Result<Tensor, BackendError> {
        Err(unsupported("exp", self.name()))
    }
    /// Element-wise natural logarithm.
    fn log(&self, _src: &Tensor) -> Result<Tensor, BackendError> {
        Err(unsupported("log", self.name()))
    }
    /// Element-wise sine.
    fn sin(&self, _src: &Tensor) -> Result<Tensor, BackendError> {
        Err(unsupported("sin", self.name()))
    }
    /// Element-wise cosine.
    fn cos(&self, _src: &Tensor) -> Result<Tensor, BackendError> {
        Err(unsupported("cos", self.name()))
    }
    /// Element-wise tangent.
    fn tan(&self, _src: &Tensor) -> Result<Tensor, BackendError> {
        Err(unsupported("tan", self.name()))
    }
    /// Element-wise scalar pow: `out = src^exponent`.
    fn pow_scalar(&self, _src: &Tensor, _exponent: f64) -> Result<Tensor, BackendError> {
        Err(unsupported("pow_scalar", self.name()))
    }
    /// Element-wise tensor-tensor pow: `out = lhs^rhs` (broadcast).
    fn pow(&self, _lhs: &Tensor, _rhs: &Tensor) -> Result<Tensor, BackendError> {
        Err(unsupported("pow", self.name()))
    }
    /// 1 / sqrt(x).
    fn rsqrt(&self, _src: &Tensor) -> Result<Tensor, BackendError> {
        Err(unsupported("rsqrt", self.name()))
    }
    /// exp(x) - 1, accurate near 0.
    fn expm1(&self, _src: &Tensor) -> Result<Tensor, BackendError> {
        Err(unsupported("expm1", self.name()))
    }
    /// log(1 + x), accurate near 0.
    fn log1p(&self, _src: &Tensor) -> Result<Tensor, BackendError> {
        Err(unsupported("log1p", self.name()))
    }
    /// Base-2 logarithm.
    fn log2(&self, _src: &Tensor) -> Result<Tensor, BackendError> {
        Err(unsupported("log2", self.name()))
    }
    /// Base-10 logarithm.
    fn log10(&self, _src: &Tensor) -> Result<Tensor, BackendError> {
        Err(unsupported("log10", self.name()))
    }
    /// Element-wise asin (radians).
    fn asin(&self, _src: &Tensor) -> Result<Tensor, BackendError> {
        Err(unsupported("asin", self.name()))
    }
    /// Element-wise acos (radians).
    fn acos(&self, _src: &Tensor) -> Result<Tensor, BackendError> {
        Err(unsupported("acos", self.name()))
    }
    /// Element-wise atan (radians).
    fn atan(&self, _src: &Tensor) -> Result<Tensor, BackendError> {
        Err(unsupported("atan", self.name()))
    }
    /// 2-argument atan: `atan2(y, x)`.
    fn atan2(&self, _y: &Tensor, _x: &Tensor) -> Result<Tensor, BackendError> {
        Err(unsupported("atan2", self.name()))
    }
    /// Hyperbolic sine.
    fn sinh(&self, _src: &Tensor) -> Result<Tensor, BackendError> {
        Err(unsupported("sinh", self.name()))
    }
    /// Hyperbolic cosine.
    fn cosh(&self, _src: &Tensor) -> Result<Tensor, BackendError> {
        Err(unsupported("cosh", self.name()))
    }
    /// ELU: `x if x > 0 else alpha * (exp(x) - 1)`.
    fn elu(&self, _src: &Tensor, _alpha: f64) -> Result<Tensor, BackendError> {
        Err(unsupported("elu", self.name()))
    }
    /// Softplus: `(1/beta) * log(1 + exp(beta * x))`, numerically stable.
    fn softplus(&self, _src: &Tensor, _beta: f64) -> Result<Tensor, BackendError> {
        Err(unsupported("softplus", self.name()))
    }
    /// HardSwish: `x * relu6(x + 3) / 6`.
    fn hardswish(&self, _src: &Tensor) -> Result<Tensor, BackendError> {
        Err(unsupported("hardswish", self.name()))
    }
    /// HardTanh: clamp(x, min, max).
    fn hardtanh(&self, _src: &Tensor, _min: f64, _max: f64) -> Result<Tensor, BackendError> {
        Err(unsupported("hardtanh", self.name()))
    }
    /// HardSigmoid: `clamp((x + 3) / 6, 0, 1)`.
    fn hardsigmoid(&self, _src: &Tensor) -> Result<Tensor, BackendError> {
        Err(unsupported("hardsigmoid", self.name()))
    }
    /// Sigmoid: `1 / (1 + exp(-x))`.
    fn sigmoid(&self, _src: &Tensor) -> Result<Tensor, BackendError> {
        Err(unsupported("sigmoid", self.name()))
    }
    /// Tanh.
    fn tanh(&self, _src: &Tensor) -> Result<Tensor, BackendError> {
        Err(unsupported("tanh", self.name()))
    }
    /// GELU (exact, using erf).
    fn gelu(&self, _src: &Tensor) -> Result<Tensor, BackendError> {
        Err(unsupported("gelu", self.name()))
    }
    /// LeakyReLU: x if x > 0 else slope*x.
    fn leaky_relu(&self, _src: &Tensor, _slope: f64) -> Result<Tensor, BackendError> {
        Err(unsupported("leaky_relu", self.name()))
    }
    /// SiLU / Swish: x * sigmoid(x).
    fn silu(&self, _src: &Tensor) -> Result<Tensor, BackendError> {
        Err(unsupported("silu", self.name()))
    }

    /// Element-wise not-equal.
    fn ne(&self, _lhs: &Tensor, _rhs: &Tensor) -> Result<Tensor, BackendError> {
        Err(unsupported("ne", self.name()))
    }
    /// Element-wise less-than.
    fn lt(&self, _lhs: &Tensor, _rhs: &Tensor) -> Result<Tensor, BackendError> {
        Err(unsupported("lt", self.name()))
    }
    /// Element-wise less-than-or-equal.
    fn le(&self, _lhs: &Tensor, _rhs: &Tensor) -> Result<Tensor, BackendError> {
        Err(unsupported("le", self.name()))
    }
    /// Element-wise greater-than.
    fn gt(&self, _lhs: &Tensor, _rhs: &Tensor) -> Result<Tensor, BackendError> {
        Err(unsupported("gt", self.name()))
    }
    /// Element-wise greater-than-or-equal.
    fn ge(&self, _lhs: &Tensor, _rhs: &Tensor) -> Result<Tensor, BackendError> {
        Err(unsupported("ge", self.name()))
    }
    /// Bool tensor: which lanes are NaN?
    fn isnan(&self, _src: &Tensor) -> Result<Tensor, BackendError> {
        Err(unsupported("isnan", self.name()))
    }
    /// Bool tensor: which lanes are ±Inf?
    fn isinf(&self, _src: &Tensor) -> Result<Tensor, BackendError> {
        Err(unsupported("isinf", self.name()))
    }
    /// Bool tensor: which lanes are finite (not NaN/Inf)?
    fn isfinite(&self, _src: &Tensor) -> Result<Tensor, BackendError> {
        Err(unsupported("isfinite", self.name()))
    }

    /// Convert a tensor from its current dtype to `target` dtype.
    ///
    /// **Saturating** float→int conversions (NaN → 0, +Inf → MAX,
    /// -Inf → MIN), matching `torch.Tensor.to(dtype)` semantics. Bool
    /// conversion follows: any nonzero → true; true → 1; false → 0.
    fn cast(
        &self,
        _src: &Tensor,
        _target: rustorch_core::tensor::dtype::Dtype,
    ) -> Result<Tensor, BackendError> {
        Err(unsupported("cast", self.name()))
    }
}

#[allow(unused)]
fn unsupported(op: &'static str, device: &'static str) -> BackendError {
    BackendError::UnsupportedOp { op, device }
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
