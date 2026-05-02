//! Benchmark: topological reordering on transformer-block + 500-op
//! synthetic DAGs.
//!
//! Plan P3 task `Topological reordering heuristic` step #5
//! acceptance: reorder reduces peak by ≥ 15% on a transformer block;
//! runs in < 10 ms for 500 ops.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rustorch_planner::reorder::{OpDag, OpId, OpNode};
use rustorch_planner::{plan, reorder, TensorId};

/// Build a single transformer-attention block as an OpDag:
///   1. Q = X @ Wq        (small)
///   2. K = X @ Wk        (small)
///   3. V = X @ Wv        (small)
///   4. Scores = Q @ Kᵀ   (BIG, the bottleneck)
///   5. Probs  = softmax(Scores)  (BIG)
///   6. Out    = Probs @ V        (small)
///   7. Residual = Out + X        (small)
///
/// `X`, `Wq`, `Wk`, `Wv` are external inputs (zero-byte placeholders in
/// the lifetime table). The reorder pass must NOT schedule the BIG
/// scores/probs allocations alongside Q/K/V — picking the small
/// branches first keeps live set tighter.
fn build_attention_block(big: u64, small: u64) -> OpDag {
    let mut dag = OpDag::new();
    // External: X = 0, Wq = 1, Wk = 2, Wv = 3
    dag.add_op(OpNode {
        id: OpId(10),
        reads: vec![TensorId(0), TensorId(1)],
        writes: TensorId(11),
        bytes: small,
        align: 1,
    }); // Q
    dag.add_op(OpNode {
        id: OpId(20),
        reads: vec![TensorId(0), TensorId(2)],
        writes: TensorId(12),
        bytes: small,
        align: 1,
    }); // K
    dag.add_op(OpNode {
        id: OpId(30),
        reads: vec![TensorId(0), TensorId(3)],
        writes: TensorId(13),
        bytes: small,
        align: 1,
    }); // V
    dag.add_op(OpNode {
        id: OpId(40),
        reads: vec![TensorId(11), TensorId(12)],
        writes: TensorId(14),
        bytes: big,
        align: 1,
    }); // Scores
    dag.add_op(OpNode {
        id: OpId(50),
        reads: vec![TensorId(14)],
        writes: TensorId(15),
        bytes: big,
        align: 1,
    }); // Probs
    dag.add_op(OpNode {
        id: OpId(60),
        reads: vec![TensorId(15), TensorId(13)],
        writes: TensorId(16),
        bytes: small,
        align: 1,
    }); // Out
    dag.add_op(OpNode {
        id: OpId(70),
        reads: vec![TensorId(16), TensorId(0)],
        writes: TensorId(17),
        bytes: small,
        align: 1,
    }); // Residual
    dag
}

fn report_savings(label: &str, dag: &OpDag) {
    let reordered = reorder(dag).unwrap();
    let reordered_peak = plan(&dag.lifetime_for(&reordered)).peak_bytes();
    // "Naive": insertion order from the DAG iter (no reordering).
    let naive_order: Vec<OpId> = {
        let mut ids: Vec<OpId> = dag.iter().map(|o| o.id).collect();
        ids.sort();
        ids
    };
    let naive_peak = plan(&dag.lifetime_for(&naive_order)).peak_bytes();
    let saved = naive_peak.saturating_sub(reordered_peak);
    let pct = if naive_peak == 0 {
        0.0
    } else {
        100.0 * saved as f64 / naive_peak as f64
    };
    eprintln!(
        "[{label}] peak: naive_topological={naive_peak}  reordered={reordered_peak}  saved={saved} ({pct:.1}%)"
    );
}

fn bench_attention_block(c: &mut Criterion) {
    // Big = 4 MB (attention scores [B,S,S] for moderate S). Small =
    // 64 KB (typical Q/K/V size).
    let dag = build_attention_block(4 * 1024 * 1024, 64 * 1024);
    report_savings("attention-block", &dag);
    c.bench_function("reorder_attention_block", |b| {
        b.iter(|| {
            let s = reorder(black_box(&dag)).unwrap();
            black_box(s);
        });
    });
}

/// Two-big-producers fixture: classic Sethi-Ullman case where order
/// matters. Two independent ops produce LARGE tensors; consume one
/// before allocating the other or you double the peak.
///
///   op1: → t1 (BIG, 4 MB)
///   op2: t1 → t2 (small, kills t1)
///   op3: → t3 (BIG, 4 MB)
///   op4: t2, t3 → t4 (kills both)
///
/// Best order: op1, op2, op3, op4 — peak ≈ big + small.
/// Worst order: op1, op3, op2, op4 — peak = 2*big.
fn build_two_big_producers() -> OpDag {
    let mut dag = OpDag::new();
    let big = 4 * 1024 * 1024;
    let small = 64 * 1024;
    dag.add_op(OpNode {
        id: OpId(1),
        reads: vec![],
        writes: TensorId(1),
        bytes: big,
        align: 1,
    });
    dag.add_op(OpNode {
        id: OpId(2),
        reads: vec![TensorId(1)],
        writes: TensorId(2),
        bytes: small,
        align: 1,
    });
    dag.add_op(OpNode {
        id: OpId(3),
        reads: vec![],
        writes: TensorId(3),
        bytes: big,
        align: 1,
    });
    dag.add_op(OpNode {
        id: OpId(4),
        reads: vec![TensorId(2), TensorId(3)],
        writes: TensorId(4),
        bytes: small,
        align: 1,
    });
    dag
}

fn bench_two_big_producers(c: &mut Criterion) {
    let dag = build_two_big_producers();
    // For this fixture, "naive_topological" = sort by OpId, which IS
    // the bad order [1, 2, 3, 4]. Wait — actually [1,2,3,4] is the
    // GOOD order: t1 dies at step 2 (before t3 is allocated at step
    // 3). Let me also report the explicitly-worst order.
    let reordered = reorder(&dag).unwrap();
    let reordered_peak = plan(&dag.lifetime_for(&reordered)).peak_bytes();
    let bad_order = vec![OpId(1), OpId(3), OpId(2), OpId(4)];
    let bad_peak = plan(&dag.lifetime_for(&bad_order)).peak_bytes();
    let saved = bad_peak.saturating_sub(reordered_peak);
    let pct = if bad_peak == 0 {
        0.0
    } else {
        100.0 * saved as f64 / bad_peak as f64
    };
    eprintln!(
        "[two-big-producers] peak: bad_topological={bad_peak}  reordered={reordered_peak}  saved={saved} ({pct:.1}%)"
    );
    c.bench_function("reorder_two_big_producers", |b| {
        b.iter(|| {
            let s = reorder(black_box(&dag)).unwrap();
            black_box(s);
        });
    });
}

/// Build a synthetic 500-op DAG: 100 chained attention blocks, each
/// 5 ops. Used to measure the reorder() runtime cost vs the < 10 ms
/// acceptance target.
fn build_500_op_dag() -> OpDag {
    let mut dag = OpDag::new();
    let mut next_op = 0u64;
    let mut next_tensor = 1000u64;
    let mut prev_tensor = TensorId(0); // external "input"
    for block in 0..100 {
        let scores = TensorId(next_tensor);
        next_tensor += 1;
        let probs = TensorId(next_tensor);
        next_tensor += 1;
        let v = TensorId(next_tensor);
        next_tensor += 1;
        let out = TensorId(next_tensor);
        next_tensor += 1;
        let resid = TensorId(next_tensor);
        next_tensor += 1;

        // op_a = compute scores from prev
        dag.add_op(OpNode {
            id: OpId(next_op),
            reads: vec![prev_tensor],
            writes: scores,
            bytes: 1024 * 1024,
            align: 64,
        });
        next_op += 1;
        // op_b = softmax
        dag.add_op(OpNode {
            id: OpId(next_op),
            reads: vec![scores],
            writes: probs,
            bytes: 1024 * 1024,
            align: 64,
        });
        next_op += 1;
        // op_c = V (small, parallel to scores/probs)
        dag.add_op(OpNode {
            id: OpId(next_op),
            reads: vec![prev_tensor],
            writes: v,
            bytes: 16 * 1024,
            align: 64,
        });
        next_op += 1;
        // op_d = out = probs @ v
        dag.add_op(OpNode {
            id: OpId(next_op),
            reads: vec![probs, v],
            writes: out,
            bytes: 16 * 1024,
            align: 64,
        });
        next_op += 1;
        // op_e = residual + norm
        dag.add_op(OpNode {
            id: OpId(next_op),
            reads: vec![out, prev_tensor],
            writes: resid,
            bytes: 16 * 1024,
            align: 64,
        });
        next_op += 1;

        prev_tensor = resid;
        let _ = block;
    }
    dag
}

fn bench_reorder_500_ops(c: &mut Criterion) {
    let dag = build_500_op_dag();
    eprintln!("[500-op-dag] op_count={}", dag.len());
    c.bench_function("reorder_500_ops", |b| {
        b.iter(|| {
            let s = reorder(black_box(&dag)).unwrap();
            black_box(s);
        });
    });
}

criterion_group!(
    benches,
    bench_attention_block,
    bench_two_big_producers,
    bench_reorder_500_ops
);
criterion_main!(benches);
