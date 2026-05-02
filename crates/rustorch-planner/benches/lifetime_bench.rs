//! Benchmark: lifetime analysis on a synthetic ResNet-50-shaped graph.
//!
//! Plan P3 (Memory Planning) task `Buffer lifetime analysis on autograd
//! graph` step #5 acceptance: 1000-node lifetime analysis completes in
//! < 5 ms median. The fixture below builds a ~1000-node trace that
//! mimics ResNet-50's structure: a long stem of conv → bn → relu →
//! pool blocks (each producing 4 buffers), with ~16 residual skip
//! connections that keep an early activation alive across several
//! later steps.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rustorch_planner::lifetime::{LifetimeTable, TensorId};

/// Build a ~`n_blocks * 4`-node trace with `n_skips` residual edges.
///
/// Each "block" is `conv → bn → relu → pool` producing 4 fresh
/// tensors; the conv output of block `k` is also read by block
/// `k + skip_distance` to model a residual connection. Skip distance
/// is chosen so that residual buffers stay alive across ~5 blocks
/// (the average ResNet-50 residual span).
fn build_resnet_like_trace(n_blocks: u32, n_skips: u32) -> LifetimeTable {
    let mut t = LifetimeTable::new();
    let bytes_per_buf: u64 = 4 * 64 * 56 * 56; // ~800 KB activation
    let mut step: u32 = 0;
    let mut block_conv_id: Vec<TensorId> = Vec::with_capacity(n_blocks as usize);

    for b in 0..n_blocks {
        // conv
        let conv = TensorId(u64::from(step));
        t.record_def(conv, step, bytes_per_buf);
        block_conv_id.push(conv);
        step += 1;

        // bn — reads conv
        let bn = TensorId(u64::from(step));
        t.record_use(conv, step).unwrap();
        t.record_def(bn, step, bytes_per_buf);
        step += 1;

        // relu — reads bn
        let relu = TensorId(u64::from(step));
        t.record_use(bn, step).unwrap();
        t.record_def(relu, step, bytes_per_buf);
        step += 1;

        // pool — reads relu
        let pool = TensorId(u64::from(step));
        t.record_use(relu, step).unwrap();
        t.record_def(pool, step, bytes_per_buf);
        step += 1;

        // residual: every `n_blocks / n_skips` blocks, extend the
        // conv of `b - skip_distance` to be alive at this step.
        if n_skips > 0 && b > 0 && b % (n_blocks / n_skips).max(1) == 0 {
            let skip_distance = 5.min(b);
            let earlier = block_conv_id[(b - skip_distance) as usize];
            t.record_use(earlier, step - 1).unwrap();
        }
    }
    t
}

fn bench_lifetime_1000_nodes(c: &mut Criterion) {
    // ~1000 nodes = 250 blocks × 4 tensors; 16 residual skips.
    c.bench_function("lifetime_resnet50_like_1000_nodes", |b| {
        b.iter(|| {
            let t = build_resnet_like_trace(black_box(250), black_box(16));
            black_box(t);
        });
    });
}

fn bench_lifetime_naive_total(c: &mut Criterion) {
    // Cost of `naive_total_bytes` on the same fixture — should be
    // negligible (single pass over the HashMap).
    let t = build_resnet_like_trace(250, 16);
    c.bench_function("lifetime_naive_total_bytes_1000_nodes", |b| {
        b.iter(|| black_box(t.naive_total_bytes()));
    });
}

criterion_group!(
    benches,
    bench_lifetime_1000_nodes,
    bench_lifetime_naive_total
);
criterion_main!(benches);
