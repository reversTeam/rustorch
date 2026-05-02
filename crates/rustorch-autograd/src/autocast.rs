//! Mixed-precision autocast — promote selected ops to a lower-precision
//! dtype within a scoped guard, while keeping numerically sensitive ops
//! (softmax, layernorm, loss reductions) in f32.
//!
//! The guard is **thread-local** so different threads (e.g. distributed
//! data-parallel ranks) may opt in independently.
//!
//! # Recipe
//!
//! ```ignore
//! use rustorch_autograd::autocast::{autocast, AutocastDtype};
//! let _g = autocast(AutocastDtype::BF16);    // RAII: dropped → off
//! let y = layer.forward(&x)?;                 // matmul/conv run in bf16
//! let loss = cross_entropy(&y, &target)?;     // promoted back to f32
//! ```
//!
//! # Promotion table
//!
//! | op family               | autocast dtype | rationale                       |
//! |-------------------------|----------------|---------------------------------|
//! | matmul, conv2d, linear  | bf16/f16       | dominant FLOPs, range-tolerant  |
//! | add, sub, mul, div      | bf16/f16       | element-wise, range-tolerant    |
//! | relu, gelu, sigmoid     | bf16/f16       | activation                      |
//! | softmax, log_softmax    | f32            | needs accurate exp/log          |
//! | layer_norm, rms_norm    | f32            | reduce + 1/sqrt(var+eps)        |
//! | cross_entropy, mse_loss | f32            | reduction order matters         |
//! | sum, mean, var          | f32            | accumulation matters            |
//!
//! Op authors call [`current_dtype`] with their op name on dispatch and
//! cast inputs accordingly; `None` means "no autocast active, leave
//! input dtype alone".
//!
//! Modeled after `torch.autocast`.

use rustorch_core::tensor::dtype::Dtype;
use std::cell::Cell;

/// Lower-precision dtype that autocast may promote ops *into*.
///
/// The two values reflect the practical landscape today:
/// - [`AutocastDtype::BF16`] — Brain float, same exponent range as f32,
///   no GradScaler needed. Recommended on Apple Silicon, Hopper, and
///   any backend with native bf16 GEMM.
/// - [`AutocastDtype::F16`] — IEEE binary16, ~5-bit exponent, requires
///   loss scaling for stable training. Mostly here for compatibility.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AutocastDtype {
    /// Brain-float 16.
    BF16,
    /// IEEE half-precision 16.
    F16,
}

impl AutocastDtype {
    /// Map back onto the matching [`Dtype`] variant.
    #[inline]
    pub fn as_dtype(self) -> Dtype {
        match self {
            AutocastDtype::BF16 => Dtype::BF16,
            AutocastDtype::F16 => Dtype::F16,
        }
    }
}

thread_local! {
    static AUTOCAST: Cell<Option<AutocastDtype>> = const { Cell::new(None) };
}

/// RAII guard returned by [`autocast`]. Restoring the previous state on
/// drop means autocast scopes nest correctly even if intermediate code
/// disables them.
#[must_use = "autocast() must be bound to a guard; bind to `_g` to keep it alive"]
pub struct AutocastGuard {
    previous: Option<AutocastDtype>,
}

impl Drop for AutocastGuard {
    fn drop(&mut self) {
        AUTOCAST.with(|c| c.set(self.previous));
    }
}

/// Enter an autocast scope. Subsequent ops on this thread that are
/// listed in the promotion table will be cast to `dtype` for compute.
///
/// ```ignore
/// {
///     let _g = autocast(AutocastDtype::BF16);
///     // ops here run in bf16 where promoted, f32 elsewhere
/// }
/// // back to f32
/// ```
pub fn autocast(dtype: AutocastDtype) -> AutocastGuard {
    let previous = AUTOCAST.with(|c| c.replace(Some(dtype)));
    AutocastGuard { previous }
}

/// Explicitly disable autocast within a scope. Mirrors `torch.autocast(enabled=False)`.
pub fn no_autocast() -> AutocastGuard {
    let previous = AUTOCAST.with(|c| c.replace(None));
    AutocastGuard { previous }
}

/// Op-name classification used by the promotion table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Policy {
    /// Promote inputs to the autocast dtype (e.g. matmul, conv).
    Promote,
    /// Force inputs to f32 even if the surrounding scope is bf16/f16
    /// (e.g. softmax, layernorm).
    KeepF32,
    /// No-op — let the op run in its native dtype.
    Passthrough,
}

fn policy_for(op_name: &str) -> Policy {
    match op_name {
        // FLOPs-dominated, range-tolerant
        "matmul" | "conv2d" | "conv1d" | "linear" | "bmm" | "addmm" => Policy::Promote,
        // Element-wise arithmetic
        "add" | "sub" | "mul" | "div" | "neg" => Policy::Promote,
        // Activations
        "relu" | "gelu" | "silu" | "sigmoid" | "tanh" => Policy::Promote,
        // Numerically sensitive — keep f32
        "softmax" | "log_softmax" | "layer_norm" | "rms_norm" | "batch_norm" => Policy::KeepF32,
        "cross_entropy" | "mse_loss" | "nll_loss" | "binary_cross_entropy" => Policy::KeepF32,
        "sum" | "mean" | "var" | "std" => Policy::KeepF32,
        // Anything else — don't touch.
        _ => Policy::Passthrough,
    }
}

/// Returns the dtype the op named `op_name` should run in given the
/// current autocast scope, or `None` if the op should keep its inputs'
/// native dtype.
///
/// Op-dispatch sites typically:
///
/// ```ignore
/// let dtype = autocast::current_dtype("matmul").unwrap_or_else(|| inputs[0].dtype());
/// let casts: Vec<Tensor> = inputs.iter().map(|t| t.to_dtype(dtype)).collect();
/// ```
pub fn current_dtype(op_name: &str) -> Option<Dtype> {
    let scope = AUTOCAST.with(|c| c.get())?;
    match policy_for(op_name) {
        Policy::Promote => Some(scope.as_dtype()),
        Policy::KeepF32 => Some(Dtype::F32),
        Policy::Passthrough => None,
    }
}

/// Returns `true` iff an autocast scope is currently active on this
/// thread. Cheap query for fast-paths that want to avoid the dispatch
/// table when nothing is enabled.
#[inline]
pub fn is_active() -> bool {
    AUTOCAST.with(|c| c.get().is_some())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_autocast_returns_none() {
        assert!(!is_active());
        assert_eq!(current_dtype("matmul"), None);
    }

    #[test]
    fn autocast_promotes_matmul_to_bf16() {
        let _g = autocast(AutocastDtype::BF16);
        assert!(is_active());
        assert_eq!(current_dtype("matmul"), Some(Dtype::BF16));
        assert_eq!(current_dtype("linear"), Some(Dtype::BF16));
        assert_eq!(current_dtype("relu"), Some(Dtype::BF16));
    }

    #[test]
    fn autocast_keeps_softmax_in_f32() {
        let _g = autocast(AutocastDtype::BF16);
        assert_eq!(current_dtype("softmax"), Some(Dtype::F32));
        assert_eq!(current_dtype("layer_norm"), Some(Dtype::F32));
        assert_eq!(current_dtype("cross_entropy"), Some(Dtype::F32));
    }

    #[test]
    fn unknown_ops_are_passthrough() {
        let _g = autocast(AutocastDtype::BF16);
        assert_eq!(current_dtype("some_random_op"), None);
    }

    #[test]
    fn nested_no_autocast_disables_then_restores() {
        let _g = autocast(AutocastDtype::BF16);
        assert!(is_active());
        {
            let _inner = no_autocast();
            assert!(!is_active());
            assert_eq!(current_dtype("matmul"), None);
        }
        assert!(is_active());
        assert_eq!(current_dtype("matmul"), Some(Dtype::BF16));
    }

    #[test]
    fn drop_restores_previous() {
        {
            let _g = autocast(AutocastDtype::BF16);
            assert_eq!(current_dtype("matmul"), Some(Dtype::BF16));
        }
        // After drop: back to no autocast.
        assert!(!is_active());
        assert_eq!(current_dtype("matmul"), None);
    }

    #[test]
    fn f16_dtype_routed_correctly() {
        let _g = autocast(AutocastDtype::F16);
        assert_eq!(current_dtype("matmul"), Some(Dtype::F16));
        assert_eq!(current_dtype("softmax"), Some(Dtype::F32));
    }
}
