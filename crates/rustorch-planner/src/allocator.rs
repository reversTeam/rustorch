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

/// Round `bytes` up to the next multiple of `align`. `align` is
/// assumed `>= 1` (callers should pass `Interval::align` which is
/// clamped at construction).
#[inline]
fn round_up(bytes: u64, align: u32) -> u64 {
    let a = u64::from(align.max(1));
    bytes.div_ceil(a) * a
}

/// One physical slot — a contiguous byte range whose size is the max
/// of all tensors assigned to it. Tensors in the same slot have
/// pairwise-disjoint intervals.
#[derive(Debug, Clone)]
pub struct Slot {
    /// Slot's byte size = `max(bytes)` over `tenants`, rounded up to
    /// `align`.
    pub bytes: u64,
    /// Slot's required alignment = `max(align)` over `tenants`. The
    /// runtime must place the slot at an offset that is a multiple
    /// of this value, which then automatically satisfies every
    /// tenant's individual alignment.
    pub align: u32,
    /// Tensors that share this slot, in def-order.
    pub tenants: Vec<TensorId>,
    intervals: Vec<Interval>,
}

impl Slot {
    fn new(id: TensorId, iv: Interval) -> Self {
        let align = iv.align.max(1);
        Slot {
            bytes: round_up(iv.bytes, align),
            align,
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
        self.align = self.align.max(iv.align.max(1));
        self.bytes = round_up(self.bytes.max(iv.bytes), self.align);
        self.tenants.push(id);
        self.intervals.push(iv);
    }

    /// Append an aliased tenant — used by the in-place layer to
    /// register an output that shares storage with an already-placed
    /// input. The slot's `bytes`/`align` are NOT recomputed because
    /// the merged interval was already accounted for during FFD.
    /// The intervals list is not extended either — `fits()` is never
    /// called after `plan_with_inplace` returns.
    pub(crate) fn push_aliased_tenant(&mut self, id: TensorId) {
        self.tenants.push(id);
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

/// Errors returned by the budget-aware planner entry-point.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PlanError {
    /// The first-fit-decreasing schedule needs more memory than the
    /// caller's budget. The runtime cannot satisfy the trace as-is;
    /// the caller should either raise the budget or invoke a more
    /// aggressive strategy (gradient checkpointing, micro-batching).
    BudgetExceeded {
        /// Bytes the planner needs (sum of slot sizes after rounding).
        required: u64,
        /// Budget the caller passed in.
        budget: u64,
    },
}

impl core::fmt::Display for PlanError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            PlanError::BudgetExceeded { required, budget } => {
                write!(
                    f,
                    "memory plan needs {required} bytes, budget is {budget} bytes"
                )
            },
        }
    }
}

impl std::error::Error for PlanError {}

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

/// Budget-aware variant of [`plan`]. Computes the same FFD schedule
/// and returns it on success; if the resulting `peak_bytes` exceeds
/// `budget` the function returns
/// [`PlanError::BudgetExceeded`] with the precise overshoot so the
/// caller can decide whether to raise the budget, fall back to a
/// recompute strategy, or fail loudly.
pub fn plan_with_budget(table: &LifetimeTable, budget: u64) -> Result<PlanResult, PlanError> {
    let result = plan(table);
    let required = result.peak_bytes();
    if required > budget {
        Err(PlanError::BudgetExceeded { required, budget })
    } else {
        Ok(result)
    }
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

    // --- Alignment-aware tests --------------------------------------

    #[test]
    fn slot_align_promotes_to_max_of_tenants() {
        // Two disjoint tensors share a slot; one needs align 64 (cache
        // line), the other needs only align 1. The slot's effective
        // alignment must be 64 so the runtime can place it at any
        // 64-aligned offset and still satisfy both tenants.
        let mut t = LifetimeTable::new();
        t.record_def_with_align(TensorId(1), 0, 100, 64);
        t.record_use(TensorId(1), 1).unwrap();
        t.record_def_with_align(TensorId(2), 2, 50, 1);
        t.record_use(TensorId(2), 3).unwrap();
        let p = plan(&t);
        assert_eq!(p.slot_count(), 1);
        let slot = &p.slots[0];
        assert_eq!(slot.align, 64);
        // Slot bytes must be rounded up to align: max(100, 50) → 100,
        // then up to next 64-multiple = 128.
        assert_eq!(slot.bytes, 128);
    }

    #[test]
    fn slot_bytes_round_up_to_align_16() {
        let mut t = LifetimeTable::new();
        // Single tensor, 100 bytes, align 16 → slot.bytes = 112.
        t.record_def_with_align(TensorId(1), 0, 100, 16);
        t.record_use(TensorId(1), 1).unwrap();
        let p = plan(&t);
        assert_eq!(p.slot_count(), 1);
        assert_eq!(p.slots[0].align, 16);
        assert_eq!(p.slots[0].bytes, 112);
        // peak_bytes reflects the rounded total.
        assert_eq!(p.peak_bytes(), 112);
    }

    #[test]
    fn slot_bytes_round_up_to_align_256() {
        let mut t = LifetimeTable::new();
        // GPU coalescing: 1000 bytes, align 256 → 1024 bytes.
        t.record_def_with_align(TensorId(1), 0, 1000, 256);
        t.record_use(TensorId(1), 1).unwrap();
        let p = plan(&t);
        assert_eq!(p.slots[0].align, 256);
        assert_eq!(p.slots[0].bytes, 1024);
    }

    #[test]
    fn align_zero_is_clamped_to_one() {
        // Defensive: bogus align=0 must not divide-by-zero or break
        // the planner. record_def_with_align clamps it at 1.
        let mut t = LifetimeTable::new();
        t.record_def_with_align(TensorId(1), 0, 100, 0);
        let iv = t.get(TensorId(1)).unwrap();
        assert_eq!(iv.align, 1);
        let p = plan(&t);
        assert_eq!(p.slots[0].align, 1);
        assert_eq!(p.slots[0].bytes, 100);
    }

    // --- Budget-aware planner tests ---------------------------------

    #[test]
    fn plan_with_budget_succeeds_when_within_budget() {
        let mut t = LifetimeTable::new();
        t.record_def(TensorId(1), 0, 64);
        t.record_use(TensorId(1), 1).unwrap();
        t.record_def(TensorId(2), 2, 64);
        t.record_use(TensorId(2), 3).unwrap();
        let p = plan_with_budget(&t, 128).unwrap();
        // Disjoint → 1 slot of 64 bytes ≤ 128 budget.
        assert_eq!(p.peak_bytes(), 64);
    }

    #[test]
    fn plan_with_budget_returns_budget_exceeded() {
        let mut t = LifetimeTable::new();
        // Two overlapping 100-byte tensors → 2 slots, peak 200.
        t.record_def(TensorId(1), 0, 100);
        t.record_def(TensorId(2), 0, 100);
        t.record_use(TensorId(1), 1).unwrap();
        t.record_use(TensorId(2), 1).unwrap();
        let err = plan_with_budget(&t, 150).unwrap_err();
        assert_eq!(
            err,
            PlanError::BudgetExceeded {
                required: 200,
                budget: 150,
            }
        );
    }

    #[test]
    fn plan_with_budget_zero_rejects_any_allocation() {
        let mut t = LifetimeTable::new();
        t.record_def(TensorId(1), 0, 1);
        let err = plan_with_budget(&t, 0).unwrap_err();
        assert!(matches!(
            err,
            PlanError::BudgetExceeded {
                required: 1,
                budget: 0
            }
        ));
    }

    #[test]
    fn plan_with_budget_empty_trace_succeeds_at_any_budget() {
        let t = LifetimeTable::new();
        let p = plan_with_budget(&t, 0).unwrap();
        assert_eq!(p.peak_bytes(), 0);
    }
}

/// Property-based tests: 1000 random schedules per invariant.
///
/// Acceptance: "for any random schedule, planner output peak <= naive
/// peak" — but we go further and verify the slot invariants the
/// allocator MUST preserve under every input.
#[cfg(test)]
mod proptests {
    use super::*;
    use crate::lifetime::TensorId;
    use proptest::prelude::*;

    /// Strategy: a vector of `(id, def_step, duration, bytes, align)`
    /// records; we feed them to LifetimeTable then plan() the result.
    /// `align` is restricted to powers of two in {1, 2, 4, 8, 16, 32, 64,
    /// 128, 256} to mimic real-world hardware alignments.
    fn schedule_strategy() -> impl Strategy<Value = Vec<(u64, u32, u32, u64, u32)>> {
        let align_strat = prop_oneof![
            Just(1u32),
            Just(2u32),
            Just(4u32),
            Just(8u32),
            Just(16u32),
            Just(32u32),
            Just(64u32),
            Just(128u32),
            Just(256u32),
        ];
        prop::collection::vec(
            (0u64..100, 0u32..100, 1u32..20, 1u64..4096, align_strat),
            1..50,
        )
    }

    fn build_table(records: &[(u64, u32, u32, u64, u32)]) -> LifetimeTable {
        let mut t = LifetimeTable::new();
        for (id, def, dur, bytes, align) in records {
            t.record_def_with_align(TensorId(*id), *def, *bytes, *align);
            // Extend last_use to def + dur to model an alive interval.
            let _ = t.record_use(TensorId(*id), def + dur);
        }
        t
    }

    proptest! {
        #![proptest_config(ProptestConfig {
            cases: 1000,
            .. ProptestConfig::default()
        })]

        /// peak_bytes() <= naive_total_bytes() (rounded for alignment)
        /// — the planner should NEVER use more memory than the naive
        /// "one buffer per tensor" baseline (modulo alignment padding).
        #[test]
        fn peak_does_not_exceed_naive(records in schedule_strategy()) {
            let t = build_table(&records);
            let p = plan(&t);
            // Naive upper bound: sum of `bytes` rounded up to each
            // tensor's own align (since each would get its own slot).
            let naive_aligned: u64 = t.iter()
                .map(|(_, iv)| {
                    let a = u64::from(iv.align.max(1));
                    iv.bytes.div_ceil(a) * a
                })
                .sum();
            prop_assert!(
                p.peak_bytes() <= naive_aligned,
                "peak {} > naive_aligned {}",
                p.peak_bytes(), naive_aligned
            );
        }

        /// Every slot's tenants have pairwise-disjoint intervals — the
        /// fundamental safety property of the allocator.
        #[test]
        fn slot_tenants_pairwise_disjoint(records in schedule_strategy()) {
            let t = build_table(&records);
            let p = plan(&t);
            for slot in &p.slots {
                let intervals: Vec<_> = slot.tenants.iter()
                    .map(|id| *t.get(*id).unwrap())
                    .collect();
                for (i, a) in intervals.iter().enumerate() {
                    for b in intervals.iter().skip(i + 1) {
                        prop_assert!(
                            !a.overlaps(b),
                            "slot has overlapping tenants: {:?} vs {:?}",
                            a, b
                        );
                    }
                }
            }
        }

        /// Every slot's `align` equals max(tenant.align). The runtime
        /// promise: place the slot at any `slot.align`-aligned offset
        /// and every tenant is happy.
        #[test]
        fn slot_align_is_max_of_tenants(records in schedule_strategy()) {
            let t = build_table(&records);
            let p = plan(&t);
            for slot in &p.slots {
                let max_align = slot.tenants.iter()
                    .map(|id| t.get(*id).unwrap().align.max(1))
                    .max()
                    .unwrap_or(1);
                prop_assert_eq!(slot.align, max_align);
            }
        }

        /// Every slot's `bytes` is a multiple of its `align`. Required
        /// for the runtime to chain slots back-to-back without
        /// re-padding between them.
        #[test]
        fn slot_bytes_aligned(records in schedule_strategy()) {
            let t = build_table(&records);
            let p = plan(&t);
            for slot in &p.slots {
                let a = u64::from(slot.align.max(1));
                prop_assert_eq!(slot.bytes % a, 0);
            }
        }

        /// Every tensor in the table is assigned to exactly one slot.
        #[test]
        fn every_tensor_assigned_once(records in schedule_strategy()) {
            let t = build_table(&records);
            let p = plan(&t);
            prop_assert_eq!(p.assignment.len(), t.len());
            // No duplicate assignment for the same tensor id.
            let mut seen: std::collections::HashSet<TensorId> = Default::default();
            for (id, _) in &p.assignment {
                prop_assert!(seen.insert(*id), "duplicate assignment for {:?}", id);
            }
        }
    }
}
