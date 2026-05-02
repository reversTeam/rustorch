//! Topological reordering heuristic — Phase 3 task `8c5cf77d`.
//!
//! Some topological orderings of an op DAG are dramatically better
//! for peak memory than others. Classic motivating example: a deep
//! left branch + a shallow right branch joining at the end. Schedule
//! the deep branch first and its intermediate stays alive across the
//! entire right branch; schedule the shallow branch first and the
//! same memory is reused.
//!
//! This module operates on a NEW abstraction layer above the
//! lifetime table: an [`OpDag`] of [`OpNode`]s. Each op declares
//! what it `reads` (input tensors), what it `writes` (output
//! tensor), and the output's `bytes`/`align`. The reorder pass
//! produces a topological schedule via a **Sethi-Ullman-style greedy
//! minimize-live-set** policy:
//!
//!   1. Start with the "ready set" = ops whose every read is already
//!      produced (or has no producer in the DAG — i.e. external
//!      inputs).
//!   2. Score each ready op as `bytes_freed - bytes_allocated` where
//!      `bytes_freed` is the sum of the bytes of reads whose
//!      *last consumer in the DAG* is this op (so they die here),
//!      and `bytes_allocated` is `writes.bytes`.
//!   3. Pick the highest-scoring op (ties broken by smallest `bytes`,
//!      then smallest `OpId` for determinism).
//!   4. Update bookkeeping (which tensors are now produced, which
//!      consumers are still pending), recompute ready set.
//!   5. Repeat until every op is scheduled.
//!
//! Convert the schedule to a [`LifetimeTable`] via
//! [`OpDag::lifetime_for`] and feed the regular FFD allocator.
//!
//! ## What this module does NOT do
//!
//! - **Cycle detection** beyond raising an error if no progress is
//!   possible (the dependency graph IS a DAG by contract; cycles are
//!   a caller bug).
//! - **OOM-recovery checkpointing hints** — flagged as a separate
//!   concern that needs the gradient-checkpointing layer (Phase 3
//!   plan `Gradient Checkpointing`, draft).

use crate::lifetime::{LifetimeTable, TensorId};
use std::collections::{BTreeSet, HashMap, HashSet};

/// Opaque op identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct OpId(pub u64);

/// One op in the DAG: it consumes `reads` and produces `writes`.
#[derive(Debug, Clone)]
pub struct OpNode {
    /// Op identifier.
    pub id: OpId,
    /// Tensors this op consumes (its inputs).
    pub reads: Vec<TensorId>,
    /// Tensor this op produces.
    pub writes: TensorId,
    /// `writes`'s payload size in bytes.
    pub bytes: u64,
    /// `writes`'s required alignment (default 1).
    pub align: u32,
}

/// A directed acyclic graph of ops keyed by [`OpId`]. The reorder
/// pass operates on this; the final allocator stays unchanged.
#[derive(Debug, Default, Clone)]
pub struct OpDag {
    nodes: HashMap<OpId, OpNode>,
    /// `tensor → producing op`. External inputs are in the trace but
    /// have no producer; they're considered already produced.
    producer_of: HashMap<TensorId, OpId>,
}

impl OpDag {
    /// Empty DAG.
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert an op. Panics if two ops claim to produce the same
    /// `writes` tensor (SSA violation).
    pub fn add_op(&mut self, op: OpNode) {
        if let Some(prev) = self.producer_of.insert(op.writes, op.id) {
            panic!(
                "tensor {:?} is produced by both {:?} and {:?} — SSA violation",
                op.writes, prev, op.id
            );
        }
        self.nodes.insert(op.id, op);
    }

    /// Iterate over all ops.
    pub fn iter(&self) -> impl Iterator<Item = &OpNode> {
        self.nodes.values()
    }

    /// Number of ops.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Empty DAG?
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Get an op by id.
    pub fn get(&self, id: OpId) -> Option<&OpNode> {
        self.nodes.get(&id)
    }

    /// Build a [`LifetimeTable`] by walking `schedule` in order.
    /// Each op's writes is recorded at its position; each read
    /// extends its tensor's `last_use` to the consuming op's
    /// position.
    ///
    /// External inputs (tensors that appear in some op's `reads` but
    /// have no producer in this DAG) are recorded at step 0 with
    /// zero bytes — they're typically pre-allocated by the caller and
    /// not the planner's concern. Pass real bytes via a dedicated
    /// extension API if needed.
    pub fn lifetime_for(&self, schedule: &[OpId]) -> LifetimeTable {
        let mut t = LifetimeTable::new();
        // Pre-register external inputs so record_use doesn't error.
        let mut external: HashSet<TensorId> = HashSet::new();
        for op in self.nodes.values() {
            for r in &op.reads {
                if !self.producer_of.contains_key(r) {
                    external.insert(*r);
                }
            }
        }
        for ext in &external {
            t.record_def(*ext, 0, 0);
        }
        for (step, id) in schedule.iter().enumerate() {
            let op = self.nodes.get(id).expect("schedule contains unknown op");
            let step = step as u32;
            t.record_def_with_align(op.writes, step, op.bytes, op.align);
            for r in &op.reads {
                let _ = t.record_use(*r, step);
            }
        }
        t
    }
}

/// Errors returned by [`reorder`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReorderError {
    /// The DAG has a dependency cycle (no op is ever ready when some
    /// remain). A real DAG cannot reach this state; it's a caller
    /// bug surfaced as a clean error rather than an infinite loop.
    Cycle {
        /// Ops left unscheduled when the algorithm got stuck.
        remaining: Vec<OpId>,
    },
}

impl core::fmt::Display for ReorderError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            ReorderError::Cycle { remaining } => write!(
                f,
                "dependency cycle detected; {} ops left unscheduled",
                remaining.len()
            ),
        }
    }
}

impl std::error::Error for ReorderError {}

/// Compute a topological schedule that locally minimises the live
/// set via a Sethi-Ullman-style greedy heuristic. Returns the
/// `Vec<OpId>` in the chosen execution order.
pub fn reorder(dag: &OpDag) -> Result<Vec<OpId>, ReorderError> {
    if dag.is_empty() {
        return Ok(Vec::new());
    }

    // For each tensor, count how many remaining consumers it has.
    let mut remaining_consumers: HashMap<TensorId, u32> = HashMap::new();
    for op in dag.nodes.values() {
        for r in &op.reads {
            *remaining_consumers.entry(*r).or_insert(0) += 1;
        }
    }

    // External inputs are considered already produced.
    let mut produced: HashSet<TensorId> = HashSet::new();
    for op in dag.nodes.values() {
        for r in &op.reads {
            if !dag.producer_of.contains_key(r) {
                produced.insert(*r);
            }
        }
    }

    // Build initial ready set.
    let mut unscheduled: HashSet<OpId> = dag.nodes.keys().copied().collect();
    let mut ready: BTreeSet<OpId> = BTreeSet::new();
    for op in dag.nodes.values() {
        if op.reads.iter().all(|r| produced.contains(r)) {
            ready.insert(op.id);
        }
    }

    let mut schedule: Vec<OpId> = Vec::with_capacity(dag.len());

    while !unscheduled.is_empty() {
        if ready.is_empty() {
            // No ready op AND unscheduled remains → cycle.
            let mut remaining: Vec<OpId> = unscheduled.iter().copied().collect();
            remaining.sort();
            return Err(ReorderError::Cycle { remaining });
        }

        // Score each ready op: bytes_freed - bytes_allocated.
        // bytes_freed = sum of reads whose remaining_consumers == 1 (this op
        // is the last consumer; the tensor dies after this op).
        let pick = ready
            .iter()
            .map(|id| {
                let op = &dag.nodes[id];
                let bytes_freed: i64 = op
                    .reads
                    .iter()
                    .filter(|r| remaining_consumers.get(*r).copied().unwrap_or(0) == 1)
                    .filter_map(|r| {
                        // External input has no producing op in the DAG; it
                        // doesn't count as freed (we don't know its size).
                        if dag.producer_of.contains_key(r) {
                            dag.nodes.get(&dag.producer_of[r]).map(|p| p.bytes as i64)
                        } else {
                            None
                        }
                    })
                    .sum();
                let bytes_alloc = op.bytes as i64;
                let score = bytes_freed - bytes_alloc;
                // Tuple is (score asc, Reverse(bytes) asc, Reverse(id) asc)
                // — max_by picks the largest, which means: highest
                // score, then smallest bytes, then smallest id.
                (score, std::cmp::Reverse(op.bytes), std::cmp::Reverse(op.id))
            })
            .max_by(|a, b| a.cmp(b))
            .map(|(_, _, id)| id.0)
            .expect("ready is non-empty");

        // Commit the pick.
        schedule.push(pick);
        ready.remove(&pick);
        unscheduled.remove(&pick);

        // Update tensor bookkeeping.
        let op = &dag.nodes[&pick];
        for r in &op.reads {
            if let Some(cnt) = remaining_consumers.get_mut(r) {
                *cnt = cnt.saturating_sub(1);
            }
        }
        produced.insert(op.writes);

        // Newly-ready ops: any unscheduled op whose every read is now
        // produced.
        for u in &unscheduled {
            if ready.contains(u) {
                continue;
            }
            let op_u = &dag.nodes[u];
            if op_u.reads.iter().all(|r| produced.contains(r)) {
                ready.insert(*u);
            }
        }
    }

    Ok(schedule)
}

/// Convenience: reorder + build lifetime table + plan via FFD.
pub fn plan_with_reorder(dag: &OpDag) -> Result<crate::allocator::PlanResult, ReorderError> {
    let schedule = reorder(dag)?;
    let table = dag.lifetime_for(&schedule);
    Ok(crate::allocator::plan(&table))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::allocator::plan;

    fn op(id: u64, reads: &[u64], writes: u64, bytes: u64) -> OpNode {
        OpNode {
            id: OpId(id),
            reads: reads.iter().copied().map(TensorId).collect(),
            writes: TensorId(writes),
            bytes,
            align: 1,
        }
    }

    #[test]
    fn empty_dag_yields_empty_schedule() {
        let dag = OpDag::new();
        assert_eq!(reorder(&dag).unwrap(), Vec::<OpId>::new());
    }

    #[test]
    fn single_op_dag() {
        let mut dag = OpDag::new();
        dag.add_op(op(1, &[], 100, 64));
        let s = reorder(&dag).unwrap();
        assert_eq!(s, vec![OpId(1)]);
    }

    #[test]
    fn linear_chain_is_topological() {
        // op1: → t1; op2: t1 → t2; op3: t2 → t3
        let mut dag = OpDag::new();
        dag.add_op(op(1, &[], 1, 16));
        dag.add_op(op(2, &[1], 2, 16));
        dag.add_op(op(3, &[2], 3, 16));
        let s = reorder(&dag).unwrap();
        assert_eq!(s, vec![OpId(1), OpId(2), OpId(3)]);
    }

    #[test]
    fn diamond_is_topological_and_unique_choice() {
        // op1 → t1 (used by op2 and op3)
        // op2: t1 → t2
        // op3: t1 → t3
        // op4: t2, t3 → t4
        let mut dag = OpDag::new();
        dag.add_op(op(1, &[], 1, 16));
        dag.add_op(op(2, &[1], 2, 8));
        dag.add_op(op(3, &[1], 3, 32));
        dag.add_op(op(4, &[2, 3], 4, 16));
        let s = reorder(&dag).unwrap();
        // op1 first, op4 last; op2 vs op3 in between.
        assert_eq!(s[0], OpId(1));
        assert_eq!(s[3], OpId(4));
        assert!(s.contains(&OpId(2)) && s.contains(&OpId(3)));
        // The smaller branch (op2 = 8 bytes) should run FIRST so its
        // small allocation is in flight while op3 (32 bytes) runs.
        // Actually our score = freed - alloc. At ready time, both
        // op2 and op3 have read t1; both would free t1 if the OTHER
        // hasn't run. Since both read t1 and t1 has 2 consumers,
        // bytes_freed for the FIRST chosen = 0 (other still consumes
        // t1); bytes_alloc for op2 = 8, for op3 = 32. So op2 has
        // score 0 - 8 = -8, op3 has score 0 - 32 = -32. op2 picked
        // first (higher score). ✓
        assert_eq!(s, vec![OpId(1), OpId(2), OpId(3), OpId(4)]);
    }

    #[test]
    fn schedule_respects_data_dependencies_no_cycle() {
        // op1 → t1; op2: t1 → t2; op3: t1, t2 → t3
        let mut dag = OpDag::new();
        dag.add_op(op(1, &[], 1, 16));
        dag.add_op(op(2, &[1], 2, 16));
        dag.add_op(op(3, &[1, 2], 3, 16));
        let s = reorder(&dag).unwrap();
        // op1 before op2 before op3.
        let pos = |id: OpId| s.iter().position(|x| *x == id).unwrap();
        assert!(pos(OpId(1)) < pos(OpId(2)));
        assert!(pos(OpId(2)) < pos(OpId(3)));
    }

    #[test]
    fn lifetime_for_chain_records_intervals() {
        let mut dag = OpDag::new();
        dag.add_op(op(1, &[], 1, 16));
        dag.add_op(op(2, &[1], 2, 32));
        dag.add_op(op(3, &[2], 3, 64));
        let s = reorder(&dag).unwrap();
        let t = dag.lifetime_for(&s);
        // 3 tensors registered.
        assert_eq!(t.len(), 3);
        // t1 born at step 0, last_use at step 1 (op2 reads it).
        let iv = t.get(TensorId(1)).unwrap();
        assert_eq!(iv.def, 0);
        assert_eq!(iv.last_use, 1);
        assert_eq!(iv.bytes, 16);
    }

    #[test]
    fn external_input_is_registered_with_zero_bytes() {
        // op1 reads external tensor 99 (no producer in the DAG).
        let mut dag = OpDag::new();
        dag.add_op(op(1, &[99], 1, 16));
        let s = reorder(&dag).unwrap();
        let t = dag.lifetime_for(&s);
        // 99 is registered (zero-byte) so record_use doesn't error.
        assert!(t.get(TensorId(99)).is_some());
        assert_eq!(t.get(TensorId(99)).unwrap().bytes, 0);
    }

    #[test]
    fn plan_with_reorder_produces_valid_plan() {
        // 5-op chain: peak should be 2 slots (alternating).
        let mut dag = OpDag::new();
        dag.add_op(op(1, &[], 1, 100));
        dag.add_op(op(2, &[1], 2, 100));
        dag.add_op(op(3, &[2], 3, 100));
        dag.add_op(op(4, &[3], 4, 100));
        dag.add_op(op(5, &[4], 5, 100));
        let p = plan_with_reorder(&dag).unwrap();
        assert!(p.peak_bytes() <= 200);
    }

    #[test]
    fn cycle_yields_clean_error() {
        // op1 reads t2; op2 reads t1 — circular.
        let mut dag = OpDag::new();
        dag.add_op(op(1, &[2], 1, 16));
        dag.add_op(op(2, &[1], 2, 16));
        let err = reorder(&dag).unwrap_err();
        match err {
            ReorderError::Cycle { remaining } => {
                assert_eq!(remaining.len(), 2);
            },
        }
    }

    #[test]
    fn deep_left_shallow_right_join_picks_shallow_first() {
        //   op1 → t1
        //   op2 (left): t1 → t2 (small)
        //   op3 (right1): t1 → t3 (large)
        //   op4 (right2): t3 → t4 (large)
        //   op5 (join): t2, t4 → t5
        // Score on ready set after op1: op2 & op3 both read t1 (2
        // consumers). Neither frees t1 yet. op2 alloc=8, op3 alloc=64.
        // → op2 picked first. Then op3 (now t1's last consumer →
        // freed). Then op4. Then op5.
        let mut dag = OpDag::new();
        dag.add_op(op(1, &[], 1, 16));
        dag.add_op(op(2, &[1], 2, 8)); // small
        dag.add_op(op(3, &[1], 3, 64)); // large
        dag.add_op(op(4, &[3], 4, 64));
        dag.add_op(op(5, &[2, 4], 5, 16));
        let s = reorder(&dag).unwrap();
        let pos = |id: OpId| s.iter().position(|x| *x == id).unwrap();
        // The greedy heuristic should run op2 between op1 and op3.
        assert!(pos(OpId(2)) < pos(OpId(3)));
    }

    #[test]
    fn reorder_does_not_increase_peak_vs_naive_topological() {
        // Build a small DAG and compare reorder() peak vs an
        // arbitrary topological order. reorder() should be ≤.
        let mut dag = OpDag::new();
        dag.add_op(op(1, &[], 1, 16));
        dag.add_op(op(2, &[1], 2, 8));
        dag.add_op(op(3, &[1], 3, 64));
        dag.add_op(op(4, &[3], 4, 64));
        dag.add_op(op(5, &[2, 4], 5, 16));
        let reordered = reorder(&dag).unwrap();
        let reordered_peak = plan(&dag.lifetime_for(&reordered)).peak_bytes();

        // Arbitrary topological order: 1, 3, 4, 2, 5.
        let arbitrary = vec![OpId(1), OpId(3), OpId(4), OpId(2), OpId(5)];
        let arbitrary_peak = plan(&dag.lifetime_for(&arbitrary)).peak_bytes();

        // The reorder pass should be no worse than this arbitrary
        // valid topological ordering.
        assert!(reordered_peak <= arbitrary_peak);
    }
}

/// Property tests: 1000 random DAGs verifying reorder safety + bound.
#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    /// Strategy: an N-op DAG where op `i` reads from a random subset
    /// of {0..i-1} (guaranteeing acyclicity). Each op writes a
    /// distinct tensor `i`. Sizes and aligns are random within sane
    /// ranges.
    fn dag_strategy() -> impl Strategy<Value = Vec<(Vec<u32>, u64, u32)>> {
        prop::collection::vec(
            (
                prop::collection::vec(0u32..30, 0..3),
                1u64..1024,
                prop_oneof![Just(1u32), Just(16u32), Just(64u32)],
            ),
            1..30,
        )
    }

    fn build_dag(records: &[(Vec<u32>, u64, u32)]) -> OpDag {
        let mut dag = OpDag::new();
        for (i, (reads, bytes, align)) in records.iter().enumerate() {
            // Filter reads to indices < i (keeps DAG acyclic) and dedup.
            let mut filtered: Vec<TensorId> = reads
                .iter()
                .filter(|&&r| (r as usize) < i)
                .map(|r| TensorId(*r as u64))
                .collect();
            filtered.sort();
            filtered.dedup();
            dag.add_op(OpNode {
                id: OpId(i as u64),
                reads: filtered,
                writes: TensorId(i as u64),
                bytes: *bytes,
                align: *align,
            });
        }
        dag
    }

    proptest! {
        #![proptest_config(ProptestConfig {
            cases: 1000,
            .. ProptestConfig::default()
        })]

        /// reorder() always succeeds on acyclic DAGs and returns a
        /// schedule of the right length.
        #[test]
        fn reorder_succeeds_and_lengths_match(records in dag_strategy()) {
            let dag = build_dag(&records);
            let s = reorder(&dag).unwrap();
            prop_assert_eq!(s.len(), dag.len());
            // Every op appears exactly once.
            let unique: std::collections::HashSet<OpId> = s.iter().copied().collect();
            prop_assert_eq!(unique.len(), s.len());
        }

        /// Every op's reads are produced before the op runs.
        #[test]
        fn schedule_respects_dependencies(records in dag_strategy()) {
            let dag = build_dag(&records);
            let s = reorder(&dag).unwrap();
            let pos: std::collections::HashMap<OpId, usize> = s.iter()
                .enumerate()
                .map(|(i, id)| (*id, i))
                .collect();
            for op in dag.iter() {
                let op_pos = pos[&op.id];
                for r in &op.reads {
                    if let Some(producer) = dag.producer_of.get(r) {
                        prop_assert!(
                            pos[producer] < op_pos,
                            "op {:?} reads {:?} produced by {:?} which runs later",
                            op.id, r, producer
                        );
                    }
                }
            }
        }

        /// The resulting plan's peak does not exceed the
        /// alignment-padded naive total — the same robustness
        /// guarantee FFD provides.
        #[test]
        fn reorder_plan_peak_bounded(records in dag_strategy()) {
            let dag = build_dag(&records);
            let p = plan_with_reorder(&dag).unwrap();
            let table = dag.lifetime_for(&reorder(&dag).unwrap());
            let naive_aligned: u64 = table.iter()
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

        /// reorder() is deterministic for the same DAG.
        #[test]
        fn reorder_is_deterministic(records in dag_strategy()) {
            let dag = build_dag(&records);
            let s1 = reorder(&dag).unwrap();
            let s2 = reorder(&dag).unwrap();
            prop_assert_eq!(s1, s2);
        }
    }
}
