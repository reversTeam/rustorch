//! Tensor lifetime analysis.
//!
//! A [`LifetimeTable`] captures, for each tensor in an execution
//! trace, the time-step at which it was *defined* (produced) and the
//! time-step of its *last use* downstream. The trace is a sequence
//! of "ops": op `t` defines tensor `D_t` and reads zero or more
//! existing tensors `R_t = {…}`.
//!
//! Two tensors with intervals `[d_a, l_a]` and `[d_b, l_b]` are said
//! to be **interval-disjoint** iff `l_a < d_b` or `l_b < d_a`. Such
//! tensors can share the same physical slot.

use std::collections::HashMap;

/// Opaque tensor identifier. The planner never inspects the value;
/// callers can use whatever monotonically-increasing IDs they like
/// (e.g. autograd node indices, hash-of-ptr, etc.).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TensorId(pub u64);

/// `[def, last_use]` interval (inclusive on both ends).
///
/// Invariant: `def <= last_use`. A tensor that is produced but never
/// read still has `last_use == def` (its slot can be reused on the
/// next time-step).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Interval {
    /// Time-step at which the tensor is produced.
    pub def: u32,
    /// Time-step of the tensor's last read. Equal to `def` for
    /// tensors that are produced and never consumed.
    pub last_use: u32,
    /// Tensor's payload size in bytes — passed straight to the
    /// allocator's slot-sizing logic.
    pub bytes: u64,
}

impl Interval {
    /// Number of time-steps the tensor is alive (inclusive).
    #[inline]
    pub fn duration(&self) -> u32 {
        self.last_use - self.def + 1
    }

    /// True iff this interval and `other` overlap on at least one
    /// time-step. Disjoint intervals can share a slot.
    #[inline]
    pub fn overlaps(&self, other: &Interval) -> bool {
        // Closed-interval intersection: [d_a, l_a] ∩ [d_b, l_b] != ∅
        self.def <= other.last_use && other.def <= self.last_use
    }
}

/// Mapping from tensor ID → its [`Interval`].
///
/// Built incrementally via [`LifetimeTable::record_def`] and
/// [`LifetimeTable::record_use`] as the trace is walked.
#[derive(Debug, Default, Clone)]
pub struct LifetimeTable {
    intervals: HashMap<TensorId, Interval>,
}

impl LifetimeTable {
    /// Empty table.
    pub fn new() -> Self {
        Self::default()
    }

    /// Mark `id` as produced at `step` with payload `bytes`.
    /// Initialises `last_use = step` so a tensor that is never read
    /// has duration 1.
    pub fn record_def(&mut self, id: TensorId, step: u32, bytes: u64) {
        self.intervals
            .entry(id)
            .and_modify(|iv| {
                // If a tensor is "re-defined" (which shouldn't happen
                // in a valid SSA trace), keep the earlier def — the
                // planner needs the largest envelope that bounds all
                // observed positions.
                iv.def = iv.def.min(step);
                iv.last_use = iv.last_use.max(step);
                iv.bytes = iv.bytes.max(bytes);
            })
            .or_insert(Interval {
                def: step,
                last_use: step,
                bytes,
            });
    }

    /// Extend `id`'s `last_use` to at least `step`. Tensors that are
    /// read before being defined raise — but in a valid graph that
    /// never happens.
    ///
    /// Returns `Err` if the tensor was never defined.
    pub fn record_use(&mut self, id: TensorId, step: u32) -> Result<(), LifetimeError> {
        match self.intervals.get_mut(&id) {
            Some(iv) => {
                iv.last_use = iv.last_use.max(step);
                Ok(())
            },
            None => Err(LifetimeError::UseBeforeDef(id, step)),
        }
    }

    /// Borrow the interval for `id`, if any.
    pub fn get(&self, id: TensorId) -> Option<&Interval> {
        self.intervals.get(&id)
    }

    /// Iterate over all `(id, interval)` pairs in insertion order is
    /// **not** guaranteed (it's a HashMap). Sort externally if needed.
    pub fn iter(&self) -> impl Iterator<Item = (&TensorId, &Interval)> {
        self.intervals.iter()
    }

    /// Number of tensors tracked.
    pub fn len(&self) -> usize {
        self.intervals.len()
    }

    /// Empty table?
    pub fn is_empty(&self) -> bool {
        self.intervals.is_empty()
    }

    /// Sum of `bytes` across every interval — i.e. the peak memory
    /// the naive "one buffer per tensor" allocator would need. Used
    /// as the denominator in savings reports.
    pub fn naive_total_bytes(&self) -> u64 {
        self.intervals.values().map(|iv| iv.bytes).sum()
    }
}

/// Errors raised by [`LifetimeTable`] operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LifetimeError {
    /// `record_use(id, step)` was called for an id that was never
    /// `record_def`'d. Indicates a malformed trace.
    UseBeforeDef(TensorId, u32),
}

impl core::fmt::Display for LifetimeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            LifetimeError::UseBeforeDef(id, step) => {
                write!(f, "tensor {:?} used at step {} before def", id, step)
            },
        }
    }
}

impl std::error::Error for LifetimeError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn duration_is_inclusive() {
        let iv = Interval {
            def: 3,
            last_use: 7,
            bytes: 0,
        };
        assert_eq!(iv.duration(), 5);
    }

    #[test]
    fn never_used_tensor_has_duration_one() {
        let iv = Interval {
            def: 4,
            last_use: 4,
            bytes: 16,
        };
        assert_eq!(iv.duration(), 1);
    }

    #[test]
    fn overlap_is_symmetric() {
        let a = Interval {
            def: 0,
            last_use: 5,
            bytes: 0,
        };
        let b = Interval {
            def: 4,
            last_use: 8,
            bytes: 0,
        };
        assert!(a.overlaps(&b));
        assert!(b.overlaps(&a));
    }

    #[test]
    fn disjoint_intervals_dont_overlap() {
        let a = Interval {
            def: 0,
            last_use: 3,
            bytes: 0,
        };
        let b = Interval {
            def: 4,
            last_use: 7,
            bytes: 0,
        };
        assert!(!a.overlaps(&b));
        assert!(!b.overlaps(&a));
    }

    #[test]
    fn touching_intervals_do_overlap() {
        // [0,3] and [3,5] both alive at step 3 → cannot share a slot.
        let a = Interval {
            def: 0,
            last_use: 3,
            bytes: 0,
        };
        let b = Interval {
            def: 3,
            last_use: 5,
            bytes: 0,
        };
        assert!(a.overlaps(&b));
    }

    #[test]
    fn record_def_then_use_extends_last_use() {
        let mut t = LifetimeTable::new();
        let a = TensorId(1);
        t.record_def(a, 2, 64);
        t.record_use(a, 5).unwrap();
        t.record_use(a, 4).unwrap(); // earlier — shouldn't shrink
        let iv = t.get(a).unwrap();
        assert_eq!(iv.def, 2);
        assert_eq!(iv.last_use, 5);
        assert_eq!(iv.bytes, 64);
    }

    #[test]
    fn record_use_before_def_errs() {
        let mut t = LifetimeTable::new();
        let a = TensorId(1);
        let err = t.record_use(a, 0).unwrap_err();
        assert_eq!(err, LifetimeError::UseBeforeDef(a, 0));
    }

    #[test]
    fn naive_total_bytes_sums_intervals() {
        let mut t = LifetimeTable::new();
        t.record_def(TensorId(1), 0, 64);
        t.record_def(TensorId(2), 1, 32);
        t.record_def(TensorId(3), 2, 16);
        assert_eq!(t.naive_total_bytes(), 64 + 32 + 16);
    }
}
