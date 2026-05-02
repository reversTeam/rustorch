//! Integration test: simulated ResNet-50 forward pass.
//!
//! Plan P3 task `End-to-end memory planner integration` (8997ca31)
//! step #4 acceptance: the end-to-end planner reduces peak memory
//! by ≥ 30% vs the naive "every tensor gets its own buffer" baseline
//! on a 1000-node ResNet-50-shaped DAG.
//!
//! ResNet-50 structure modeled here:
//!   - Stem:   conv7x7 + bn + relu + maxpool   (4 ops)
//!   - Stage 1: 3 bottleneck blocks
//!   - Stage 2: 4 bottleneck blocks
//!   - Stage 3: 6 bottleneck blocks
//!   - Stage 4: 3 bottleneck blocks
//!   - Head:   avgpool + flatten + linear      (3 ops)
//!
//! Each bottleneck is 1×1 conv → bn → relu → 3×3 conv → bn → relu →
//! 1×1 conv → bn → add(skip) → relu = 10 ops, with the input
//! pre-bottleneck kept alive across the entire block via the skip
//! connection. Activations grow then shrink across stages
//! (CHW shapes mimic standard ResNet-50: 56×56×64, 28×28×128,
//! 14×14×256, 7×7×512). Total ≈ 170 ops (forward only).

use rustorch_planner::reorder::{OpId, OpNode};
use rustorch_planner::{InPlaceHint, OpDag, Planner, PlannerInput, TensorId};

/// Synthetic ResNet-50 forward DAG.
///
/// Returns the DAG, a list of in-place hints (one per relu / one per
/// add), and the head's output tensor (for sanity checks).
fn build_resnet50_forward() -> (OpDag, Vec<InPlaceHint>) {
    let mut dag = OpDag::new();
    let mut hints = Vec::new();
    let mut next_op = 0u64;
    let mut next_tensor = 1u64;

    // Helper to add an op and return its writes tensor.
    let add_op = |dag: &mut OpDag,
                  reads: Vec<TensorId>,
                  bytes: u64,
                  align: u32,
                  next_op: &mut u64,
                  next_tensor: &mut u64|
     -> (OpId, TensorId) {
        let id = OpId(*next_op);
        let writes = TensorId(*next_tensor);
        *next_op += 1;
        *next_tensor += 1;
        dag.add_op(OpNode {
            id,
            reads,
            writes,
            bytes,
            align,
        });
        (id, writes)
    };

    // Bytes for activations at each stage. Real ResNet-50 sizes
    // (batch=1, fp32):
    // Stage 0 (stem):    56 * 56 * 64 * 4 = 802816 ≈ 784 KB
    // Stage 1 bottleneck output:    256 chans → 56*56*256*4 = 3.2 MB
    // Stage 2 bottleneck output:   512 chans → 28*28*512*4 = 1.6 MB
    // Stage 3 bottleneck output:  1024 chans → 14*14*1024*4 = 802 KB
    // Stage 4 bottleneck output:  2048 chans → 7*7*2048*4 = 401 KB
    let stage_act_bytes = [
        // (mid_act, block_output)
        (200_704, 3_211_264), // stage 1
        (100_352, 1_605_632), // stage 2
        (50_176, 802_816),    // stage 3
        (25_088, 401_408),    // stage 4
    ];
    let blocks_per_stage = [3usize, 4, 6, 3];

    // Pre-register an external "image input" tensor.
    let image_input = TensorId(next_tensor);
    next_tensor += 1;
    // Stem: conv7x7 + bn + relu + maxpool, all working on the input
    // and producing the same-sized intermediate.
    let stem_bytes = 802_816u64;
    let (_op_conv, t_conv) = add_op(
        &mut dag,
        vec![image_input],
        stem_bytes,
        64,
        &mut next_op,
        &mut next_tensor,
    );
    let (op_bn, t_bn) = add_op(
        &mut dag,
        vec![t_conv],
        stem_bytes,
        64,
        &mut next_op,
        &mut next_tensor,
    );
    let (op_relu, t_relu) = add_op(
        &mut dag,
        vec![t_bn],
        stem_bytes,
        64,
        &mut next_op,
        &mut next_tensor,
    );
    hints.push(InPlaceHint::new(t_bn, t_relu, op_relu.0 as u32));
    let (_op_pool, t_pool) = add_op(
        &mut dag,
        vec![t_relu],
        stem_bytes,
        64,
        &mut next_op,
        &mut next_tensor,
    );

    let mut prev_block_output = t_pool;
    let _ = op_bn;

    for (stage_idx, n_blocks) in blocks_per_stage.iter().enumerate() {
        let (mid_bytes, out_bytes) = stage_act_bytes[stage_idx];
        for _block in 0..*n_blocks {
            // Bottleneck block: 1×1 → bn → relu → 3×3 → bn → relu →
            // 1×1 → bn → add(skip) → relu
            let block_input = prev_block_output;
            let (_, t1) = add_op(
                &mut dag,
                vec![block_input],
                mid_bytes,
                64,
                &mut next_op,
                &mut next_tensor,
            ); // 1×1 conv
            let (_, t2) = add_op(
                &mut dag,
                vec![t1],
                mid_bytes,
                64,
                &mut next_op,
                &mut next_tensor,
            ); // bn
            let (op_r1, t3) = add_op(
                &mut dag,
                vec![t2],
                mid_bytes,
                64,
                &mut next_op,
                &mut next_tensor,
            ); // relu
            hints.push(InPlaceHint::new(t2, t3, op_r1.0 as u32));
            let (_, t4) = add_op(
                &mut dag,
                vec![t3],
                mid_bytes,
                64,
                &mut next_op,
                &mut next_tensor,
            ); // 3×3 conv
            let (_, t5) = add_op(
                &mut dag,
                vec![t4],
                mid_bytes,
                64,
                &mut next_op,
                &mut next_tensor,
            ); // bn
            let (op_r2, t6) = add_op(
                &mut dag,
                vec![t5],
                mid_bytes,
                64,
                &mut next_op,
                &mut next_tensor,
            ); // relu
            hints.push(InPlaceHint::new(t5, t6, op_r2.0 as u32));
            let (_, t7) = add_op(
                &mut dag,
                vec![t6],
                out_bytes,
                64,
                &mut next_op,
                &mut next_tensor,
            ); // 1×1 conv (expand)
            let (_, t8) = add_op(
                &mut dag,
                vec![t7],
                out_bytes,
                64,
                &mut next_op,
                &mut next_tensor,
            ); // bn
            let (_, t9) = add_op(
                &mut dag,
                vec![t8, block_input],
                out_bytes,
                64,
                &mut next_op,
                &mut next_tensor,
            ); // add(skip)
            let (op_r3, t10) = add_op(
                &mut dag,
                vec![t9],
                out_bytes,
                64,
                &mut next_op,
                &mut next_tensor,
            ); // relu
            hints.push(InPlaceHint::new(t9, t10, op_r3.0 as u32));
            prev_block_output = t10;
        }
    }

    // Head: avgpool + flatten + linear
    let head_bytes = 8_192u64;
    let (_, t_pool_head) = add_op(
        &mut dag,
        vec![prev_block_output],
        head_bytes,
        64,
        &mut next_op,
        &mut next_tensor,
    );
    let (_, t_flatten) = add_op(
        &mut dag,
        vec![t_pool_head],
        head_bytes,
        64,
        &mut next_op,
        &mut next_tensor,
    );
    let (_, _t_linear) = add_op(
        &mut dag,
        vec![t_flatten],
        4_000,
        64,
        &mut next_op,
        &mut next_tensor,
    );

    (dag, hints)
}

#[test]
fn resnet50_sim_planner_saves_at_least_30_percent_peak() {
    let (dag, hints) = build_resnet50_forward();
    let op_count = dag.len();

    // Naive baseline: "every tensor gets its own buffer" — the sum
    // of every recorded tensor's bytes. This is the textbook
    // pre-planner peak the memory planner is meant to beat (the
    // milestone description targets 30-50% reduction vs this).
    let naive_order: Vec<OpId> = {
        let mut ids: Vec<OpId> = dag.iter().map(|o| o.id).collect();
        ids.sort();
        ids
    };
    let naive_table = dag.lifetime_for(&naive_order);
    let naive_peak = naive_table.naive_total_bytes();

    // Planned: full pipeline (reorder + lifetime + in-place + FFD).
    let input = PlannerInput {
        dag: dag.clone(),
        inplace_hints: hints,
        blockers: Default::default(),
        budget: None,
    };
    let schedule = Planner::plan(&input).unwrap();
    let planned_peak = schedule.peak_bytes();

    let savings = naive_peak.saturating_sub(planned_peak);
    let pct = 100.0 * savings as f64 / naive_peak as f64;

    // Print what we measured so future regressions are visible.
    println!(
        "[resnet50_sim] op_count={op_count}  naive_peak={naive_peak}  planned_peak={planned_peak}  saved={savings}  ({pct:.1}%)"
    );

    // Sanity: schedule covers every op.
    assert_eq!(schedule.ops.len(), op_count);

    // Acceptance: ≥ 30% reduction.
    assert!(
        pct >= 30.0,
        "expected ≥ 30% peak savings, got {pct:.1}% (naive={naive_peak}, planned={planned_peak})"
    );
}

#[test]
fn resnet50_sim_planner_is_deterministic() {
    let (dag, hints) = build_resnet50_forward();
    let input = PlannerInput {
        dag,
        inplace_hints: hints,
        blockers: Default::default(),
        budget: None,
    };
    let s1 = Planner::plan(&input).unwrap();
    let s2 = Planner::plan(&input).unwrap();
    assert_eq!(s1.ops, s2.ops);
    assert_eq!(s1.peak_bytes(), s2.peak_bytes());
}

#[test]
fn resnet50_sim_schedule_respects_dependencies() {
    let (dag, hints) = build_resnet50_forward();
    let input = PlannerInput {
        dag: dag.clone(),
        inplace_hints: hints,
        blockers: Default::default(),
        budget: None,
    };
    let schedule = Planner::plan(&input).unwrap();
    let pos: std::collections::HashMap<OpId, usize> = schedule
        .ops
        .iter()
        .enumerate()
        .map(|(i, id)| (*id, i))
        .collect();
    let producer_of: std::collections::HashMap<TensorId, OpId> =
        dag.iter().map(|op| (op.writes, op.id)).collect();
    for op in dag.iter() {
        let op_pos = pos[&op.id];
        for r in &op.reads {
            if let Some(producer) = producer_of.get(r) {
                assert!(
                    pos[producer] < op_pos,
                    "op {:?} reads {:?} produced by {:?} which runs at or after",
                    op.id,
                    r,
                    producer
                );
            }
        }
    }
}
