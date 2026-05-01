//! Thread-local "no-grad" guard.
//!
//! Per RFC-0003, gradient tracking is enabled by default. `no_grad`
//! flips a thread-local flag that ops check before recording a
//! `grad_fn` — analogous to `torch.no_grad()`.

use std::cell::Cell;

thread_local! {
    static GRAD_ENABLED: Cell<bool> = const { Cell::new(true) };
}

/// Returns `true` iff gradient tracking is currently enabled on this
/// thread.
#[inline]
pub fn is_grad_enabled() -> bool {
    GRAD_ENABLED.with(|f| f.get())
}

/// Run `f` with gradient tracking disabled. Restores the previous
/// state on return — even on panic — via the [`NoGradGuard`].
pub fn no_grad<F, R>(f: F) -> R
where
    F: FnOnce() -> R,
{
    let _guard = NoGradGuard::enter();
    f()
}

/// Run `f` with gradient tracking explicitly enabled — used to
/// re-enable grad tracking inside a `no_grad` scope.
pub fn with_grad<F, R>(f: F) -> R
where
    F: FnOnce() -> R,
{
    let _guard = WithGradGuard::enter();
    f()
}

/// RAII guard that disables gradient tracking on construction and
/// restores the previous state on drop.
pub struct NoGradGuard {
    previous: bool,
}

impl NoGradGuard {
    /// Enter a no-grad scope.
    pub fn enter() -> Self {
        let previous = GRAD_ENABLED.with(|f| {
            let p = f.get();
            f.set(false);
            p
        });
        NoGradGuard { previous }
    }
}

impl Drop for NoGradGuard {
    fn drop(&mut self) {
        GRAD_ENABLED.with(|f| f.set(self.previous));
    }
}

/// RAII guard that enables gradient tracking, restoring on drop.
pub struct WithGradGuard {
    previous: bool,
}

impl WithGradGuard {
    /// Enter a with-grad scope (re-enables grad inside no_grad).
    pub fn enter() -> Self {
        let previous = GRAD_ENABLED.with(|f| {
            let p = f.get();
            f.set(true);
            p
        });
        WithGradGuard { previous }
    }
}

impl Drop for WithGradGuard {
    fn drop(&mut self) {
        GRAD_ENABLED.with(|f| f.set(self.previous));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_enabled() {
        assert!(is_grad_enabled());
    }

    #[test]
    fn no_grad_disables_within_scope() {
        assert!(is_grad_enabled());
        no_grad(|| {
            assert!(!is_grad_enabled());
        });
        assert!(is_grad_enabled());
    }

    #[test]
    fn nested_no_grad_restores_correctly() {
        no_grad(|| {
            assert!(!is_grad_enabled());
            no_grad(|| {
                assert!(!is_grad_enabled());
            });
            assert!(!is_grad_enabled());
        });
        assert!(is_grad_enabled());
    }

    #[test]
    fn with_grad_inside_no_grad_re_enables() {
        no_grad(|| {
            assert!(!is_grad_enabled());
            with_grad(|| {
                assert!(is_grad_enabled());
            });
            assert!(!is_grad_enabled());
        });
    }

    #[test]
    fn no_grad_guard_restores_on_drop() {
        let _g = NoGradGuard::enter();
        assert!(!is_grad_enabled());
        drop(_g);
        assert!(is_grad_enabled());
    }
}
