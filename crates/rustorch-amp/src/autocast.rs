//! Autocast scope guard + op promotion list.
//!
//! Sets a thread-local "current mixed dtype" that the eager
//! dispatcher (or the kernel caller) consults to decide whether to
//! run the next op in bf16 / fp16 or keep it in f32.

use std::cell::Cell;

/// Mixed-precision dtype selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mixed {
    /// No autocast — every op runs in its native dtype.
    None,
    /// bf16 — wider exponent than fp16, no GradScaler needed (Ampere+
    /// recommended).
    Bf16,
    /// fp16 — narrower range, requires GradScaler for stable training
    /// (Volta/Turing legacy support).
    Fp16,
}

/// Op kinds that the promotion list distinguishes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OpKind {
    /// Matrix multiplication — promote to bf16/fp16.
    Matmul,
    /// Convolution — promote to bf16/fp16.
    Conv,
    /// Linear (fused matmul+bias) — promote.
    Linear,
    /// Embedding lookup — promote.
    Embedding,
    /// Softmax — keep in f32 for numerical stability.
    Softmax,
    /// LayerNorm — keep in f32.
    LayerNorm,
    /// RMSNorm — keep in f32.
    RmsNorm,
    /// BatchNorm — keep in f32.
    BatchNorm,
    /// Loss function (CE / MSE / etc) — keep in f32.
    Loss,
    /// Reduction (sum / mean / max) — keep in f32.
    Reduction,
    /// Element-wise op — pass through (use input dtype).
    Elementwise,
}

thread_local! {
    static CURRENT_MIXED: Cell<Mixed> = const { Cell::new(Mixed::None) };
    static FORCE_FP32_DEPTH: Cell<u32> = const { Cell::new(0) };
}

/// RAII guard installing a mixed-precision context. Drops back to the
/// previous setting (supports nesting).
pub struct AutocastGuard {
    prev: Mixed,
}

impl AutocastGuard {
    /// Enter an autocast scope.
    pub fn enter(mixed: Mixed) -> Self {
        let prev = CURRENT_MIXED.with(|c| {
            let old = c.get();
            c.set(mixed);
            old
        });
        Self { prev }
    }
}

impl Drop for AutocastGuard {
    fn drop(&mut self) {
        CURRENT_MIXED.with(|c| c.set(self.prev));
    }
}

/// Run a closure inside an autocast scope. The scope is exited when
/// the closure returns (Ok or Err).
pub fn autocast<R>(mixed: Mixed, f: impl FnOnce() -> R) -> R {
    let _g = AutocastGuard::enter(mixed);
    f()
}

/// Force f32 for an inner scope, bypassing the surrounding autocast.
pub fn autocast_force_fp32<R>(f: impl FnOnce() -> R) -> R {
    FORCE_FP32_DEPTH.with(|c| c.set(c.get() + 1));
    let result = f();
    FORCE_FP32_DEPTH.with(|c| c.set(c.get() - 1));
    result
}

/// Inspect the current effective mixed dtype, taking the
/// `force_fp32` override into account.
pub fn current_dtype() -> Mixed {
    if FORCE_FP32_DEPTH.with(|c| c.get()) > 0 {
        return Mixed::None;
    }
    CURRENT_MIXED.with(|c| c.get())
}

/// Decide whether `op` should run in the low-precision dtype under
/// the current autocast. Returns `Some(low_prec)` when promotion
/// applies, `None` to keep f32.
pub fn promote_for(op: OpKind) -> Option<Mixed> {
    match current_dtype() {
        Mixed::None => None,
        m @ (Mixed::Bf16 | Mixed::Fp16) => match op {
            OpKind::Matmul | OpKind::Conv | OpKind::Linear | OpKind::Embedding => Some(m),
            OpKind::Softmax
            | OpKind::LayerNorm
            | OpKind::RmsNorm
            | OpKind::BatchNorm
            | OpKind::Loss
            | OpKind::Reduction => None,
            OpKind::Elementwise => None, // pass-through (caller's dtype)
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_autocast_by_default() {
        assert_eq!(current_dtype(), Mixed::None);
    }

    #[test]
    fn autocast_scopes_are_thread_local() {
        autocast(Mixed::Bf16, || {
            assert_eq!(current_dtype(), Mixed::Bf16);
        });
        // Outside the scope: back to None.
        assert_eq!(current_dtype(), Mixed::None);
    }

    #[test]
    fn nested_autocast_inner_overrides_outer() {
        autocast(Mixed::Bf16, || {
            assert_eq!(current_dtype(), Mixed::Bf16);
            autocast(Mixed::Fp16, || {
                assert_eq!(current_dtype(), Mixed::Fp16);
            });
            // Restored.
            assert_eq!(current_dtype(), Mixed::Bf16);
        });
    }

    #[test]
    fn force_fp32_overrides_autocast() {
        autocast(Mixed::Bf16, || {
            assert_eq!(current_dtype(), Mixed::Bf16);
            autocast_force_fp32(|| {
                assert_eq!(current_dtype(), Mixed::None);
            });
            assert_eq!(current_dtype(), Mixed::Bf16);
        });
    }

    #[test]
    fn matmul_promoted_under_bf16_autocast() {
        autocast(Mixed::Bf16, || {
            assert_eq!(promote_for(OpKind::Matmul), Some(Mixed::Bf16));
            assert_eq!(promote_for(OpKind::Conv), Some(Mixed::Bf16));
        });
    }

    #[test]
    fn softmax_kept_in_f32_under_autocast() {
        autocast(Mixed::Bf16, || {
            assert_eq!(promote_for(OpKind::Softmax), None);
            assert_eq!(promote_for(OpKind::LayerNorm), None);
            assert_eq!(promote_for(OpKind::Loss), None);
        });
    }

    #[test]
    fn no_promotion_when_autocast_is_none() {
        assert_eq!(promote_for(OpKind::Matmul), None);
        assert_eq!(promote_for(OpKind::Softmax), None);
    }

    #[test]
    fn force_fp32_disables_promotion() {
        autocast(Mixed::Bf16, || {
            autocast_force_fp32(|| {
                assert_eq!(promote_for(OpKind::Matmul), None);
            });
        });
    }
}
