//! In-place mutation detection — Phase 3 task `eef1b67b`.
//!
//! When a unary op like `relu`, `abs`, `neg`, `sigmoid`, `tanh`
//! transforms an input tensor whose remaining lifetime ends at the
//! same step, the runtime can write the output **into the input's
//! storage** instead of allocating a fresh buffer. The planner's job
//! here is to identify which (input → output) candidates are safe and
//! to rewrite the slot assignment so both tensors share one slot.
//!
//! ## Safety conditions (all must hold for a hint to be applied)
//!
//! 1. Both input and output exist in the [`LifetimeTable`].
//! 2. The input's `last_use` equals the in-place step — the input is
//!    *dying* at the op, so reusing its storage doesn't violate any
//!    reader downstream.
//! 3. The input is **not** marked as having a backward hook
//!    ([`InPlaceBlockers::has_hook`]). Hooks need the original value
//!    at backward time; mutating in place would silently corrupt the
//!    backward pass.
//! 4. The input is **not** a view of another tensor
//!    ([`InPlaceBlockers::is_view`]). Mutating a view aliases someone
//!    else's storage and is never safe at the planner stage.
//!
//! ## Bit-exactness
//!
//! The planner is **metadata-only** — it never inspects tensor
//! contents. Hence in-place rewriting cannot alter bit-exact
//! behaviour: NaN/Inf preservation, sign-of-zero, denormal handling
//! are all properties of the OP implementation, not the planner. By
//! construction, `plan_with_inplace` produces a `PlanResult` that
//! depends only on the trace's structural metadata.
//!
//! ## Design
//!
//! In-place rewriting is implemented as a **pre-pass** over the
//! lifetime table: each safe `(input, output)` pair is "merged" into
//! a single virtual interval `[input.def, output.last_use]` with the
//! `max` of bytes/align. The merged table is fed to [`plan`] and the
//! aliased outputs are stitched back into the resulting slots
//! afterward. This keeps the FFD allocator unchanged and isolates
//! the in-place logic in this module.

use crate::allocator::{plan, PlanResult};
use crate::lifetime::{Interval, LifetimeTable, TensorId};
use std::collections::{HashMap, HashSet};

/// A request: "consider mutating the storage of `input` to hold
/// `output` at `step`". The planner verifies safety; the runtime
/// performs the mutation only if the planner accepts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct InPlaceHint {
    /// Tensor whose storage will be reused.
    pub input: TensorId,
    /// Tensor that will be produced into the input's storage.
    pub output: TensorId,
    /// Time-step at which the unary op runs (the input dies and the
    /// output is born here).
    pub step: u32,
}

impl InPlaceHint {
    /// Convenience constructor.
    pub fn new(input: TensorId, output: TensorId, step: u32) -> Self {
        Self {
            input,
            output,
            step,
        }
    }
}

/// Per-tensor flags that block in-place mutation regardless of the
/// hint. Populated by the caller (autograd / tracer) which knows
/// which tensors carry hooks or view aliases.
#[derive(Debug, Clone, Default)]
pub struct InPlaceBlockers {
    has_hook: HashSet<TensorId>,
    is_view: HashSet<TensorId>,
}

impl InPlaceBlockers {
    /// Empty (every tensor is candidate).
    pub fn new() -> Self {
        Self::default()
    }

    /// Mark `id` as carrying a backward hook — never mutate in place.
    pub fn mark_hook(&mut self, id: TensorId) -> &mut Self {
        self.has_hook.insert(id);
        self
    }

    /// Mark `id` as a view of another tensor — never mutate in place.
    pub fn mark_view(&mut self, id: TensorId) -> &mut Self {
        self.is_view.insert(id);
        self
    }

    /// True iff `id` is hook-free.
    pub fn is_hook_free(&self, id: TensorId) -> bool {
        !self.has_hook.contains(&id)
    }

    /// True iff `id` is not a view.
    pub fn is_not_view(&self, id: TensorId) -> bool {
        !self.is_view.contains(&id)
    }
}

/// Outcome of evaluating a single [`InPlaceHint`] against a table +
/// blockers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InPlaceDecision {
    /// All safety conditions hold; the planner will alias the output
    /// onto the input's slot.
    Apply,
    /// Either `input` or `output` is not registered in the table.
    BlockedByMissingId,
    /// The input's `last_use` is not equal to the in-place `step` —
    /// some downstream reader still needs the original value.
    BlockedByLastUse,
    /// The input carries a backward hook.
    BlockedByHook,
    /// The input is a view alias of another tensor.
    BlockedByView,
}

/// Decide whether `hint` passes the four safety conditions. Pure
/// function — no side effects.
pub fn evaluate_hint(
    table: &LifetimeTable,
    hint: &InPlaceHint,
    blockers: &InPlaceBlockers,
) -> InPlaceDecision {
    let input = match table.get(hint.input) {
        Some(iv) => iv,
        None => return InPlaceDecision::BlockedByMissingId,
    };
    if table.get(hint.output).is_none() {
        return InPlaceDecision::BlockedByMissingId;
    }
    if input.last_use != hint.step {
        return InPlaceDecision::BlockedByLastUse;
    }
    if !blockers.is_hook_free(hint.input) {
        return InPlaceDecision::BlockedByHook;
    }
    if !blockers.is_not_view(hint.input) {
        return InPlaceDecision::BlockedByView;
    }
    InPlaceDecision::Apply
}

/// Run [`plan`] with in-place rewriting applied to every hint that
/// satisfies all safety conditions.
///
/// Hints whose safety check fails (e.g. blocked by a hook) are
/// silently dropped — the resulting plan falls back to fresh-buffer
/// allocation for those outputs. The function never panics on bad
/// hints: missing IDs, mis-ordered (input/output), conflicting hints
/// targeting the same input — all return a valid plan.
///
/// Conflicting hints (two outputs both wanting to alias the same
/// input) are resolved by keeping the **first** safe hint encountered
/// in iteration order; subsequent conflicts are dropped.
pub fn plan_with_inplace(
    table: &LifetimeTable,
    hints: &[InPlaceHint],
    blockers: &InPlaceBlockers,
) -> PlanResult {
    // Step 1: filter to safe hints, dropping conflicts (one input
    // can be aliased by at most one output).
    let mut claimed_inputs: HashSet<TensorId> = HashSet::new();
    let mut claimed_outputs: HashSet<TensorId> = HashSet::new();
    let safe: Vec<InPlaceHint> = hints
        .iter()
        .filter(|h| evaluate_hint(table, h, blockers) == InPlaceDecision::Apply)
        .filter(|h| {
            // Drop if input or output already participates in another
            // accepted hint (transitive in-place chains are out of
            // scope for the v1 planner).
            if claimed_inputs.contains(&h.input)
                || claimed_outputs.contains(&h.output)
                || claimed_inputs.contains(&h.output)
                || claimed_outputs.contains(&h.input)
            {
                return false;
            }
            claimed_inputs.insert(h.input);
            claimed_outputs.insert(h.output);
            true
        })
        .copied()
        .collect();

    // Step 2: build a rewritten table where each safe pair's input
    // interval is extended to cover the output, and the output is
    // dropped from the table.
    let aliased_outputs: HashSet<TensorId> = safe.iter().map(|h| h.output).collect();
    let merged: HashMap<TensorId, Interval> = safe
        .iter()
        .map(|h| {
            let input_iv = *table.get(h.input).expect("evaluated as Apply");
            let output_iv = *table.get(h.output).expect("evaluated as Apply");
            let merged_iv = Interval {
                def: input_iv.def,
                last_use: input_iv.last_use.max(output_iv.last_use),
                bytes: input_iv.bytes.max(output_iv.bytes),
                align: input_iv.align.max(output_iv.align),
            };
            (h.input, merged_iv)
        })
        .collect();

    let mut rewritten = LifetimeTable::new();
    for (id, iv) in table.iter() {
        if aliased_outputs.contains(id) {
            continue; // dropped — will be re-attached after FFD
        }
        let final_iv = merged.get(id).copied().unwrap_or(*iv);
        rewritten.record_def_with_align(*id, final_iv.def, final_iv.bytes, final_iv.align);
        let _ = rewritten.record_use(*id, final_iv.last_use);
    }

    // Step 3: FFD on the rewritten table.
    let mut result = plan(&rewritten);

    // Step 4: stitch each aliased output into its input's slot.
    for hint in &safe {
        let input_slot = result
            .slot_of(hint.input)
            .expect("input was preserved in rewritten table");
        result.assignment.push((hint.output, input_slot));
        result.slots[input_slot].push_aliased_tenant(hint.output);
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn def_use(t: &mut LifetimeTable, id: u64, def: u32, last_use: u32, bytes: u64) {
        let id = TensorId(id);
        t.record_def(id, def, bytes);
        t.record_use(id, last_use).unwrap();
    }

    #[test]
    fn relu_with_refcount_one_input_mutates_in_place() {
        // Trace: x defined at 0, last used at 1. y = relu(x) at step 1.
        let mut t = LifetimeTable::new();
        def_use(&mut t, 1, 0, 1, 100); // x dies at step 1
        def_use(&mut t, 2, 1, 2, 100); // y born at step 1, dies at 2
        let hint = InPlaceHint::new(TensorId(1), TensorId(2), 1);
        let p = plan_with_inplace(&t, &[hint], &InPlaceBlockers::new());
        assert_eq!(
            p.slot_of(TensorId(1)),
            p.slot_of(TensorId(2)),
            "x and y must share a slot"
        );
        assert_eq!(p.slot_count(), 1);
    }

    #[test]
    fn relu_with_refcount_two_input_allocates_new_buffer() {
        // x is read at step 1 AND step 2 — last_use is 2, in-place at
        // step 1 must be blocked.
        let mut t = LifetimeTable::new();
        let x = TensorId(1);
        let y = TensorId(2);
        t.record_def(x, 0, 100);
        t.record_use(x, 1).unwrap(); // first read (the relu candidate)
        t.record_use(x, 2).unwrap(); // second read — blocks in-place
        t.record_def(y, 1, 100);
        t.record_use(y, 2).unwrap();
        let hint = InPlaceHint::new(x, y, 1);
        assert_eq!(
            evaluate_hint(&t, &hint, &InPlaceBlockers::new()),
            InPlaceDecision::BlockedByLastUse
        );
        let p = plan_with_inplace(&t, &[hint], &InPlaceBlockers::new());
        assert_ne!(p.slot_of(x), p.slot_of(y), "x and y must NOT share");
        assert_eq!(p.slot_count(), 2);
    }

    #[test]
    fn tensor_with_backward_hook_is_never_mutated_in_place() {
        let mut t = LifetimeTable::new();
        let x = TensorId(1);
        let y = TensorId(2);
        def_use(&mut t, 1, 0, 1, 100);
        def_use(&mut t, 2, 1, 2, 100);
        let mut blockers = InPlaceBlockers::new();
        blockers.mark_hook(x);
        let hint = InPlaceHint::new(x, y, 1);
        assert_eq!(
            evaluate_hint(&t, &hint, &blockers),
            InPlaceDecision::BlockedByHook
        );
        let p = plan_with_inplace(&t, &[hint], &blockers);
        assert_ne!(p.slot_of(x), p.slot_of(y));
    }

    #[test]
    fn view_alias_is_rejected_at_planner_stage() {
        let mut t = LifetimeTable::new();
        def_use(&mut t, 1, 0, 1, 100);
        def_use(&mut t, 2, 1, 2, 100);
        let mut blockers = InPlaceBlockers::new();
        blockers.mark_view(TensorId(1));
        let hint = InPlaceHint::new(TensorId(1), TensorId(2), 1);
        assert_eq!(
            evaluate_hint(&t, &hint, &blockers),
            InPlaceDecision::BlockedByView
        );
    }

    #[test]
    fn missing_id_is_rejected_safely() {
        let mut t = LifetimeTable::new();
        def_use(&mut t, 1, 0, 1, 100);
        // output id never registered
        let hint = InPlaceHint::new(TensorId(1), TensorId(99), 1);
        assert_eq!(
            evaluate_hint(&t, &hint, &InPlaceBlockers::new()),
            InPlaceDecision::BlockedByMissingId
        );
        // The planner must not panic on bad hints.
        let p = plan_with_inplace(&t, &[hint], &InPlaceBlockers::new());
        assert_eq!(p.slot_count(), 1);
    }

    #[test]
    fn empty_hint_list_matches_plan() {
        let mut t = LifetimeTable::new();
        def_use(&mut t, 1, 0, 2, 100);
        def_use(&mut t, 2, 1, 3, 100);
        def_use(&mut t, 3, 2, 4, 100);
        let with_inplace = plan_with_inplace(&t, &[], &InPlaceBlockers::new());
        let without = plan(&t);
        // Empty hints means the two algorithms produce the same peak.
        assert_eq!(with_inplace.peak_bytes(), without.peak_bytes());
    }

    #[test]
    fn empty_table_is_no_op() {
        let t = LifetimeTable::new();
        let hint = InPlaceHint::new(TensorId(1), TensorId(2), 0);
        let p = plan_with_inplace(&t, &[hint], &InPlaceBlockers::new());
        assert_eq!(p.slot_count(), 0);
        assert_eq!(p.peak_bytes(), 0);
    }

    #[test]
    fn conflicting_hints_resolved_by_first_wins() {
        // Two outputs both want to alias x — only the first hint is
        // kept; the second is silently dropped.
        let mut t = LifetimeTable::new();
        def_use(&mut t, 1, 0, 1, 100); // x
        def_use(&mut t, 2, 1, 2, 100); // y
        def_use(&mut t, 3, 1, 2, 100); // z (also wants x's storage)
        let h1 = InPlaceHint::new(TensorId(1), TensorId(2), 1);
        let h2 = InPlaceHint::new(TensorId(1), TensorId(3), 1);
        let p = plan_with_inplace(&t, &[h1, h2], &InPlaceBlockers::new());
        // x and y share; z gets its own slot.
        assert_eq!(p.slot_of(TensorId(1)), p.slot_of(TensorId(2)));
        assert_ne!(p.slot_of(TensorId(1)), p.slot_of(TensorId(3)));
    }

    #[test]
    fn chained_inplace_hints_break_cycles() {
        // Chain: x -> y -> z. The planner only applies the first
        // (x → y); the second (y → z) is dropped because y is
        // already a claimed output.
        let mut t = LifetimeTable::new();
        def_use(&mut t, 1, 0, 1, 100); // x dies at 1
        def_use(&mut t, 2, 1, 2, 100); // y dies at 2
        def_use(&mut t, 3, 2, 3, 100); // z dies at 3
        let h1 = InPlaceHint::new(TensorId(1), TensorId(2), 1);
        let h2 = InPlaceHint::new(TensorId(2), TensorId(3), 2);
        let p = plan_with_inplace(&t, &[h1, h2], &InPlaceBlockers::new());
        assert_eq!(p.slot_of(TensorId(1)), p.slot_of(TensorId(2)));
        // z does NOT chain through y — separate slot.
        assert_ne!(p.slot_of(TensorId(2)), p.slot_of(TensorId(3)));
    }

    #[test]
    fn inplace_well_formed_chain_does_not_hurt_peak() {
        // x -> relu(x)=y -> relu(y)=z, where each output is born
        // exactly when the previous input dies. In-place is at least
        // neutral — never strictly worse than no-op. Real wins
        // depend on the byte-size distribution; FFD already does an
        // excellent job when sizes are uniform.
        let mut t = LifetimeTable::new();
        def_use(&mut t, 1, 0, 1, 100); // x dies at 1
        def_use(&mut t, 2, 1, 2, 100); // y dies at 2 (born when x dies)
        def_use(&mut t, 3, 2, 3, 100); // z dies at 3 (born when y dies)
        let with_inplace = plan_with_inplace(
            &t,
            &[InPlaceHint::new(TensorId(1), TensorId(2), 1)],
            &InPlaceBlockers::new(),
        );
        let without = plan(&t);
        assert_eq!(with_inplace.peak_bytes(), without.peak_bytes());
    }

    #[test]
    fn inplace_saves_a_slot_when_input_is_largest() {
        // x is alive [0,1] bytes 100 (the largest). y born at 1
        // bytes 10. Without in-place: 2 slots (100 + 10) = 110.
        // With in-place: merged [0, 2] bytes 100. Peak: 100.
        let mut t = LifetimeTable::new();
        def_use(&mut t, 1, 0, 1, 100);
        def_use(&mut t, 2, 1, 2, 10);
        let with_inplace = plan_with_inplace(
            &t,
            &[InPlaceHint::new(TensorId(1), TensorId(2), 1)],
            &InPlaceBlockers::new(),
        );
        let without = plan(&t);
        assert_eq!(without.peak_bytes(), 110);
        assert_eq!(with_inplace.peak_bytes(), 100);
    }

    #[test]
    fn interval_extension_can_increase_peak_in_pathological_cases() {
        // GOTCHA documented for future maintainers: in-place is NOT a
        // universal win. When the OUTPUT outlives the INPUT by a long
        // margin, the merged interval extends across other tensors'
        // intervals — preventing reuse that the un-merged plan would
        // have allowed.
        //
        // Trace:
        //   x:  [0, 1] bytes 10  ← will be in-placed into y
        //   y:  [0, 4] bytes 1   ← long-lived, but the hint.step is 1
        //   z:  [2, 3] bytes 9
        // Without in-place: x and z disjoint → share slot (size 10),
        //                   y in its own slot (size 1) → peak 11.
        // With in-place x→y: merged [0,4] size 10, z[2,3] overlaps →
        //                    distinct slot (size 9) → peak 19.
        //
        // The runtime is responsible for emitting hints only when
        // they're locally beneficial. The planner trusts the hints
        // structurally and applies them as requested.
        let mut t = LifetimeTable::new();
        def_use(&mut t, 1, 0, 1, 10); // x
        t.record_def(TensorId(2), 0, 1); // y born at 0 (NOT chain-shaped)
        t.record_use(TensorId(2), 4).unwrap();
        def_use(&mut t, 3, 2, 3, 9); // z
        let hint = InPlaceHint::new(TensorId(1), TensorId(2), 1);
        let with_inplace = plan_with_inplace(&t, &[hint], &InPlaceBlockers::new());
        let without = plan(&t);
        // This is the documented pathology — in-place hurts here.
        assert!(
            with_inplace.peak_bytes() >= without.peak_bytes(),
            "expected the documented pathology — got opposite"
        );
    }
}

/// Property tests: 1000 cases verifying the planner-level invariants
/// of in-place rewriting.
#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    /// Strategy: chain-shaped trace + per-link "wants in-place" flag.
    ///
    /// Each tensor `i+1` is born **exactly** when tensor `i` dies, so
    /// the generated `InPlaceHint`s are structurally well-formed
    /// (output.def == input.last_use). This is the contract the
    /// runtime is expected to honour when it emits hints.
    fn chain_strategy() -> impl Strategy<Value = (Vec<(u32, u64)>, Vec<bool>)> {
        let chain = prop::collection::vec((1u32..5, 1u64..1024), 2..20);
        let flags = prop::collection::vec(any::<bool>(), 0..20);
        (chain, flags)
    }

    /// Build a chain-shaped trace: tensor `i` is defined when tensor
    /// `i-1` dies (`output.def == input.last_use`). Hints alias every
    /// adjacent pair.
    fn build_chain(records: &[(u32, u64)]) -> (LifetimeTable, Vec<InPlaceHint>) {
        let mut t = LifetimeTable::new();
        let mut hints = Vec::new();
        let mut last_use = 0u32;
        for (i, (dies_after, bytes)) in records.iter().enumerate() {
            let id = TensorId(i as u64);
            let def = if i == 0 { 0 } else { last_use };
            t.record_def(id, def, *bytes);
            let last = def.saturating_add(*dies_after);
            let _ = t.record_use(id, last);
            if i > 0 {
                let prev = TensorId((i - 1) as u64);
                hints.push(InPlaceHint::new(prev, id, def));
            }
            last_use = last;
        }
        (t, hints)
    }

    proptest! {
        #![proptest_config(ProptestConfig {
            cases: 1000,
            .. ProptestConfig::default()
        })]

        /// `plan_with_inplace.peak_bytes()` is bounded by the
        /// alignment-padded naive baseline — same robustness guarantee
        /// as plain `plan`. (The stronger "in-place never increases
        /// peak vs plan" claim does NOT hold universally: merging two
        /// chained intervals can consume FFD slack that previously
        /// allowed small tensors to share the largest slot for free.
        /// See `interval_extension_can_increase_peak_in_pathological_cases`
        /// for a worked example.)
        #[test]
        fn peak_does_not_exceed_naive((records, _flags) in chain_strategy()) {
            let (t, hints) = build_chain(&records);
            let with_inplace = plan_with_inplace(&t, &hints, &InPlaceBlockers::new());
            let naive_aligned: u64 = t.iter()
                .map(|(_, iv)| {
                    let a = u64::from(iv.align.max(1));
                    iv.bytes.div_ceil(a) * a
                })
                .sum();
            prop_assert!(
                with_inplace.peak_bytes() <= naive_aligned,
                "peak {} > naive_aligned {}",
                with_inplace.peak_bytes(), naive_aligned
            );
        }

        /// Every tensor in the original table appears in the
        /// assignment of `plan_with_inplace`. Aliased outputs are
        /// stitched back into their input's slot — none lost, none
        /// duplicated by id.
        #[test]
        fn assignment_covers_every_tensor((records, _flags) in chain_strategy()) {
            let (t, hints) = build_chain(&records);
            let p = plan_with_inplace(&t, &hints, &InPlaceBlockers::new());
            let assigned: std::collections::HashSet<TensorId> =
                p.assignment.iter().map(|(id, _)| *id).collect();
            for (id, _) in t.iter() {
                prop_assert!(assigned.contains(id), "tensor {:?} missing from assignment", id);
            }
            prop_assert_eq!(assigned.len(), t.len());
        }

        /// Hooks block — if every input is hook-marked, peak is
        /// identical to the plain `plan`.
        #[test]
        fn all_hooks_means_no_inplace((records, _flags) in chain_strategy()) {
            let (t, hints) = build_chain(&records);
            let mut blockers = InPlaceBlockers::new();
            for (id, _) in t.iter() {
                blockers.mark_hook(*id);
            }
            let with_inplace = plan_with_inplace(&t, &hints, &blockers);
            let without = plan(&t);
            prop_assert_eq!(with_inplace.peak_bytes(), without.peak_bytes());
        }

        /// Aliased outputs share their input's slot — the
        /// fundamental contract of in-place rewriting.
        #[test]
        fn aliased_outputs_share_input_slot((records, _flags) in chain_strategy()) {
            let (t, hints) = build_chain(&records);
            let p = plan_with_inplace(&t, &hints, &InPlaceBlockers::new());
            // Re-evaluate which hints would have been applied.
            let mut claimed_inputs = std::collections::HashSet::new();
            let mut claimed_outputs = std::collections::HashSet::new();
            for hint in &hints {
                if evaluate_hint(&t, hint, &InPlaceBlockers::new()) != InPlaceDecision::Apply {
                    continue;
                }
                if claimed_inputs.contains(&hint.input)
                    || claimed_outputs.contains(&hint.output)
                    || claimed_inputs.contains(&hint.output)
                    || claimed_outputs.contains(&hint.input)
                {
                    continue;
                }
                claimed_inputs.insert(hint.input);
                claimed_outputs.insert(hint.output);
                prop_assert_eq!(
                    p.slot_of(hint.input), p.slot_of(hint.output),
                    "hint {:?} was applied but slots differ", hint
                );
            }
        }
    }
}
