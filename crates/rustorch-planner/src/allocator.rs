//! Interval-disjoint allocator.
//!
//! Given a [`LifetimeTable`], pack tensors into the smallest set of
//! physical slots such that:
//! 1. No two tensors with overlapping intervals share a slot.
//! 2. Each slot's size = `max(bytes)` over the tensors assigned to it.
//!
//! Algorithm: **first-fit-decreasing** (FFD).
//!   1. Sort tensors by `(bytes desc, def asc)` so the largest, longest-
//!      lived tensors are placed first — they're the hardest to fit.
//!   2. For each tensor `t`, scan existing slots in order; place `t`
//!      in the first slot whose intervals are all disjoint from
//!      `t.interval`. If none fits, open a new slot.
//!
//! FFD is `O(n²)` worst-case but the constant is tiny and `n` (number
//! of intermediate tensors in a typical training step) is in the
//! hundreds. The optimal solution is NP-hard (interval graph
//! coloring with weights) — FFD is within a small constant factor
//! and is what Inductor uses in practice.

use crate::lifetime::{Interval, LifetimeTable, TensorId};

/// One physical slot — a contiguous byte range whose size is the max
/// of all tensors assigned to it. Tensors in the same slot have
/// pairwise-disjoint intervals.
#[derive(Debug, Clone)]
pub struct Slot {
    /// Slot's byte size = `max(bytes)` over `tenants`.
    pub bytes: u64,
    /// Tensors that share this slot, in def-order.
    pub tenants: Vec<TensorId>,
    intervals: Vec<Interval>,
}

impl Slot {
    fn new(id: TensorId, iv: Interval) -> Self {
        Slot {
            bytes: iv.bytes,
            tenants: vec![id],
            intervals: vec![iv],
        }
    }

    /// True iff `iv` does not overlap any of the slot's existing
    /// intervals — i.e. the slot can host this new tensor.
    fn fits(&self, iv: &Interval) -> bool {
        self.intervals.iter().all(|existing| !existing.overlaps(iv))
    }

    fn add(&mut self, id: TensorId, iv: Interval) {
        self.bytes = self.bytes.max(iv.bytes);
        self.tenants.push(id);
        self.intervals.push(iv);
    }
}

/// Final memory-plan output.
#[derive(Debug, Clone)]
pub struct PlanResult {
    /// All allocated slots, in creation order.
    pub slots: Vec<Slot>,
    /// Mapping from each tensor to the index of its assigned slot.
    pub assignment: Vec<(TensorId, usize)>,
}

impl PlanResult {
    /// Sum of every slot's `bytes`. The bound the runtime needs to
    /// allocate up-front.
    pub fn peak_bytes(&self) -> u64 {
        self.slots.iter().map(|s| s.bytes).sum()
    }

    /// Saved bytes vs. the naive "one buffer per tensor" baseline.
    /// Returns `(saved_bytes, ratio)` where `ratio = saved / naive`.
    pub fn savings_vs(&self, naive_total: u64) -> (u64, f64) {
        let peak = self.peak_bytes();
        let saved = naive_total.saturating_sub(peak);
        let ratio = if naive_total == 0 {
            0.0
        } else {
            saved as f64 / naive_total as f64
        };
        (saved, ratio)
    }

    /// Number of distinct slots.
    pub fn slot_count(&self) -> usize {
        self.slots.len()
    }

    /// Look up the slot index for a given tensor.
    pub fn slot_of(&self, id: TensorId) -> Option<usize> {
        self.assignment
            .iter()
            .find(|(tid, _)| *tid == id)
            .map(|(_, idx)| *idx)
    }
}

/// Run the planner on `table`. Tensors are scheduled in
/// first-fit-decreasing order (largest first; ties broken by earlier
/// `def`). Returns a [`PlanResult`] with the slot list and per-tensor
/// assignment.
pub fn plan(table: &LifetimeTable) -> PlanResult {
    // Collect (id, interval) and sort: bytes desc, then def asc.
    let mut order: Vec<(TensorId, Interval)> = table.iter().map(|(id, iv)| (*id, *iv)).collect();
    order.sort_by(|a, b| {
        b.1.bytes
            .cmp(&a.1.bytes)
            .then_with(|| a.1.def.cmp(&b.1.def))
            .then_with(|| a.0.cmp(&b.0))
    });

    let mut slots: Vec<Slot> = Vec::new();
    let mut assignment: Vec<(TensorId, usize)> = Vec::with_capacity(order.len());

    for (id, iv) in order {
        // First-fit: scan existing slots for a non-overlapping one.
        let mut placed_in: Option<usize> = None;
        for (idx, slot) in slots.iter_mut().enumerate() {
            if slot.fits(&iv) {
                slot.add(id, iv);
                placed_in = Some(idx);
                break;
            }
        }
        let idx = placed_in.unwrap_or_else(|| {
            slots.push(Slot::new(id, iv));
            slots.len() - 1
        });
        assignment.push((id, idx));
    }

    PlanResult { slots, assignment }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disjoint_tensors_share_a_slot() {
        let mut t = LifetimeTable::new();
        t.record_def(TensorId(1), 0, 64);
        t.record_use(TensorId(1), 1).unwrap();
        t.record_def(TensorId(2), 2, 64);
        t.record_use(TensorId(2), 3).unwrap();
        let p = plan(&t);
        assert_eq!(p.slot_count(), 1);
        assert_eq!(p.peak_bytes(), 64);
    }

    #[test]
    fn overlapping_tensors_get_distinct_slots() {
        let mut t = LifetimeTable::new();
        t.record_def(TensorId(1), 0, 64);
        t.record_def(TensorId(2), 1, 64);
        // Both alive at step 1 → must be in different slots.
        t.record_use(TensorId(1), 2).unwrap();
        t.record_use(TensorId(2), 2).unwrap();
        let p = plan(&t);
        assert_eq!(p.slot_count(), 2);
        assert_eq!(p.peak_bytes(), 128);
    }

    #[test]
    fn slot_size_is_max_of_tenants() {
        let mut t = LifetimeTable::new();
        // Three disjoint tensors of sizes 32, 64, 16 → all share one
        // slot of size 64 (the max).
        t.record_def(TensorId(1), 0, 32);
        t.record_use(TensorId(1), 1).unwrap();
        t.record_def(TensorId(2), 2, 64);
        t.record_use(TensorId(2), 3).unwrap();
        t.record_def(TensorId(3), 4, 16);
        let p = plan(&t);
        assert_eq!(p.slot_count(), 1);
        assert_eq!(p.peak_bytes(), 64);
    }

    #[test]
    fn first_fit_decreasing_packs_two_pairs() {
        // 4 tensors, two pairs of disjoint intervals:
        //   T1:[0..2] T2:[3..5]      → share slot
        //   T3:[0..2] T4:[3..5]      → share slot
        //   T1 vs T3 overlap, T2 vs T4 overlap → 2 distinct slots total
        let mut t = LifetimeTable::new();
        t.record_def(TensorId(1), 0, 100);
        t.record_use(TensorId(1), 2).unwrap();
        t.record_def(TensorId(2), 3, 50);
        t.record_use(TensorId(2), 5).unwrap();
        t.record_def(TensorId(3), 0, 100);
        t.record_use(TensorId(3), 2).unwrap();
        t.record_def(TensorId(4), 3, 50);
        t.record_use(TensorId(4), 5).unwrap();
        let p = plan(&t);
        assert_eq!(p.slot_count(), 2);
        // Two slots, each sized to its largest tenant (100 each).
        assert_eq!(p.peak_bytes(), 200);
    }

    #[test]
    fn savings_vs_naive_is_correct() {
        // 3 tensors, all disjoint, 100 bytes each → naive=300, plan=100.
        let mut t = LifetimeTable::new();
        for (i, def) in [(1, 0), (2, 2), (3, 4)] {
            t.record_def(TensorId(i), def, 100);
            t.record_use(TensorId(i), def + 1).unwrap();
        }
        let p = plan(&t);
        let (saved, ratio) = p.savings_vs(t.naive_total_bytes());
        assert_eq!(saved, 200);
        assert!((ratio - 2.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn empty_table_yields_empty_plan() {
        let t = LifetimeTable::new();
        let p = plan(&t);
        assert_eq!(p.slot_count(), 0);
        assert_eq!(p.peak_bytes(), 0);
    }

    #[test]
    fn slot_of_returns_correct_index() {
        let mut t = LifetimeTable::new();
        t.record_def(TensorId(7), 0, 32);
        t.record_use(TensorId(7), 1).unwrap();
        t.record_def(TensorId(8), 0, 64); // overlaps T7
        t.record_use(TensorId(8), 1).unwrap();
        let p = plan(&t);
        assert_eq!(p.slot_count(), 2);
        assert!(p.slot_of(TensorId(7)).is_some());
        assert!(p.slot_of(TensorId(8)).is_some());
        assert_ne!(p.slot_of(TensorId(7)), p.slot_of(TensorId(8)));
    }

    /// Realistic transformer block trace: matmul → add → softmax → matmul.
    /// Each intermediate is alive for only one step.
    #[test]
    fn transformer_block_savings() {
        let mut t = LifetimeTable::new();
        // step 0: x defined (input, kept alive throughout)
        t.record_def(TensorId(0), 0, 4096); // x
                                            // step 1: q = x @ Wq
        t.record_use(TensorId(0), 1).unwrap();
        t.record_def(TensorId(1), 1, 4096); // q
                                            // step 2: k = x @ Wk
        t.record_use(TensorId(0), 2).unwrap();
        t.record_def(TensorId(2), 2, 4096); // k
                                            // step 3: v = x @ Wv
        t.record_use(TensorId(0), 3).unwrap();
        t.record_def(TensorId(3), 3, 4096); // v
                                            // step 4: scores = q @ k.t() — kills q and k
        t.record_use(TensorId(1), 4).unwrap();
        t.record_use(TensorId(2), 4).unwrap();
        t.record_def(TensorId(4), 4, 8192); // scores [S, S]
                                            // step 5: probs = softmax(scores) — kills scores
        t.record_use(TensorId(4), 5).unwrap();
        t.record_def(TensorId(5), 5, 8192); // probs
                                            // step 6: out = probs @ v — kills probs, v, x
        t.record_use(TensorId(5), 6).unwrap();
        t.record_use(TensorId(3), 6).unwrap();
        t.record_def(TensorId(6), 6, 4096); // out
        let p = plan(&t);
        // Naive baseline.
        let naive = t.naive_total_bytes();
        // Real transformer savings: we expect significantly fewer than
        // 7 slots and significantly less than `naive` bytes peak.
        assert!(p.slot_count() < 7, "got {} slots", p.slot_count());
        assert!(
            p.peak_bytes() < naive,
            "got peak {} vs naive {}",
            p.peak_bytes(),
            naive
        );
        let (_, ratio) = p.savings_vs(naive);
        // Plan should save at least 30% on this fixture.
        assert!(ratio >= 0.30, "expected ≥ 30% savings, got {ratio:.2}");
    }
}
