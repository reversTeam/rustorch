//! Version counter for in-place mutation safety (RFC-0002 / RFC-0003,
//! P1.1 task `VersionCounter`).
//!
//! Every [`Tensor`](super::Tensor) carries a `VersionCounter` whose
//! integer is bumped on every in-place op (`add_`, `mul_`, `copy_`,
//! `fill_`, `zero_`, …). Autograd's `SavedVariable` snapshots the
//! version when a tensor is saved for backward; if the version has
//! advanced by the time `backward()` runs, we panic with a clear
//! `"in-place modification of saved tensor"` message — matching
//! PyTorch's behaviour.
//!
//! Views share the *same* counter as their base, so an in-place op
//! through a view also bumps the base's version.
//!
//! ```
//! use rustorch_core::tensor::version::VersionCounter;
//!
//! let v = VersionCounter::new();
//! assert_eq!(v.current(), 0);
//! v.bump();
//! assert_eq!(v.current(), 1);
//!
//! // Views share the counter (cheap Arc clone).
//! let view = v.clone();
//! view.bump();
//! assert_eq!(v.current(), 2);
//! ```

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// Thread-safe monotonic version counter shared between a tensor and
/// its views.
///
/// `Clone` is the cheap `Arc` bump and produces a counter that *shares*
/// the underlying atomic — bumping through any clone affects every
/// other clone.
///
/// The atomic uses `Relaxed` ordering: the counter is a write-after-
/// write monotonic value used only by the autograd engine to compare
/// snapshots taken under happens-before from a `SavedVariable`'s
/// creation site (which provides its own synchronisation).
#[derive(Debug, Clone, Default)]
pub struct VersionCounter(Arc<AtomicU64>);

impl VersionCounter {
    /// Build a new counter starting at version `0`.
    #[inline]
    pub fn new() -> Self {
        VersionCounter(Arc::new(AtomicU64::new(0)))
    }

    /// Bump the version by 1. Returns the *new* version. Atomic +1
    /// under `Relaxed` ordering.
    #[inline]
    pub fn bump(&self) -> u64 {
        // fetch_add returns the old value; we want the new one.
        self.0.fetch_add(1, Ordering::Relaxed) + 1
    }

    /// Read the current version.
    #[inline]
    pub fn current(&self) -> u64 {
        self.0.load(Ordering::Relaxed)
    }

    /// Snapshot the current version. Returned [`VersionSnapshot`] is
    /// `Copy` and `Eq` — used by autograd's `SavedVariable` to detect
    /// in-place modification.
    #[inline]
    pub fn snapshot(&self) -> VersionSnapshot {
        VersionSnapshot(self.current())
    }

    /// `true` iff `self` has been bumped since `snap` was taken.
    #[inline]
    pub fn has_changed_since(&self, snap: VersionSnapshot) -> bool {
        self.current() != snap.0
    }

    /// `true` iff this counter and `other` are the *same* `Arc`
    /// (shared with views) — useful for tests.
    #[inline]
    pub fn shares_with(&self, other: &VersionCounter) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl PartialEq for VersionCounter {
    /// Two counters compare equal iff they share the same underlying
    /// atomic *and* both currently read the same version. This treats
    /// independently-allocated counters with the same value as
    /// distinct, which matches the Tensor equality semantics.
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0) && self.current() == other.current()
    }
}

/// Snapshot of a version. `Copy + Eq + Hash`, used as the discriminant
/// in [`SavedVariable`](#) (autograd, P1.5).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VersionSnapshot(u64);

impl VersionSnapshot {
    /// The numeric version this snapshot was taken at.
    #[inline]
    pub fn version(self) -> u64 {
        self.0
    }
}

// --------------------------------------------------------------------------
// Tests
// --------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc as StdArc;
    use std::thread;

    #[test]
    fn new_starts_at_zero() {
        let v = VersionCounter::new();
        assert_eq!(v.current(), 0);
    }

    #[test]
    fn default_starts_at_zero() {
        let v = VersionCounter::default();
        assert_eq!(v.current(), 0);
    }

    #[test]
    fn bump_returns_new_version() {
        let v = VersionCounter::new();
        assert_eq!(v.bump(), 1);
        assert_eq!(v.bump(), 2);
        assert_eq!(v.bump(), 3);
        assert_eq!(v.current(), 3);
    }

    #[test]
    fn clone_shares_counter() {
        let v = VersionCounter::new();
        let view = v.clone();
        assert!(view.shares_with(&v));
        view.bump();
        assert_eq!(v.current(), 1);
        assert_eq!(view.current(), 1);
        v.bump();
        assert_eq!(view.current(), 2);
    }

    #[test]
    fn distinct_counters_are_independent() {
        let a = VersionCounter::new();
        let b = VersionCounter::new();
        assert!(!a.shares_with(&b));
        a.bump();
        assert_eq!(a.current(), 1);
        assert_eq!(b.current(), 0);
    }

    #[test]
    fn snapshot_detects_change() {
        let v = VersionCounter::new();
        let snap = v.snapshot();
        assert_eq!(snap.version(), 0);
        assert!(!v.has_changed_since(snap));
        v.bump();
        assert!(v.has_changed_since(snap));
    }

    #[test]
    fn snapshot_is_copy() {
        let v = VersionCounter::new();
        let s1 = v.snapshot();
        let s2 = s1; // Copy
        assert_eq!(s1, s2);
    }

    #[test]
    fn equality_requires_same_arc_and_value() {
        let a = VersionCounter::new();
        let a_view = a.clone();
        assert_eq!(a, a_view);
        a.bump();
        assert_eq!(a, a_view); // both bumped together
        let b = VersionCounter::new(); // distinct
        assert_ne!(a, b);
    }

    #[test]
    fn concurrent_bumps_are_atomic() {
        // 8 threads each bumping 10_000 times → final version == 80_000.
        let v = StdArc::new(VersionCounter::new());
        let mut handles = Vec::with_capacity(8);
        for _ in 0..8 {
            let v = v.clone();
            handles.push(thread::spawn(move || {
                for _ in 0..10_000 {
                    v.bump();
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
        assert_eq!(v.current(), 80_000);
    }

    #[test]
    fn monotonic_property_500_ops() {
        // Random LCG-driven bumps; current() must never decrease.
        let v = VersionCounter::new();
        let mut s: u64 = 42;
        let mut prev = v.current();
        for _ in 0..500 {
            s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            if s & 1 == 0 {
                v.bump();
            }
            let cur = v.current();
            assert!(cur >= prev);
            prev = cur;
        }
    }

    #[test]
    fn debug_format() {
        let v = VersionCounter::new();
        let _ = format!("{v:?}"); // smoke
    }
}
