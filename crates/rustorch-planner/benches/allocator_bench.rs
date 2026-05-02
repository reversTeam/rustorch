//! Benchmark: FFD allocator on a 1000-buffer ResNet-50-like trace.
//!
//! Plan P3 (Memory Planning) task `Union-find allocator` step #5
//! acceptance: allocation on 1000 buffers completes in < 2 ms median.
//! Re-uses the lifetime fixture's structure (250 conv→bn→relu→pool
//! blocks + residual skips) to keep the two benchmarks comparable.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rustorch_planner::lifetime::{LifetimeTable, TensorId};
use rustorch_planner::{plan, plan_with_budget};

fn build_resnet_like_trace(n_blocks: u32, n_skips: u32) -> LifetimeTable {
    let mut t = LifetimeTable::new();
    let bytes_per_buf: u64 = 4 * 64 * 56 * 56;
    let mut step: u32 = 0;
    let mut block_conv_id: Vec<TensorId> = Vec::with_capacity(n_blocks as usize);
    for b in 0..n_blocks {
        let conv = TensorId(u64::from(step));
        // Conv outputs are usually aligned to GPU coalescing boundary.
        t.record_def_with_align(conv, step, bytes_per_buf, 256);
        block_conv_id.push(conv);
        step += 1;
        let bn = TensorId(u64::from(step));
        t.record_use(conv, step).unwrap();
        t.record_def_with_align(bn, step, bytes_per_buf, 64);
        step += 1;
        let relu = TensorId(u64::from(step));
        t.record_use(bn, step).unwrap();
        t.record_def(relu, step, bytes_per_buf);
        step += 1;
        let pool = TensorId(u64::from(step));
        t.record_use(relu, step).unwrap();
        t.record_def_with_align(pool, step, bytes_per_buf, 16);
        step += 1;

        if n_skips > 0 && b > 0 && b % (n_blocks / n_skips).max(1) == 0 {
            let skip_distance = 5.min(b);
            let earlier = block_conv_id[(b - skip_distance) as usize];
            t.record_use(earlier, step - 1).unwrap();
        }
    }
    t
}

fn bench_plan_1000_buffers(c: &mut Criterion) {
    let table = build_resnet_like_trace(250, 16);
    c.bench_function("plan_resnet50_like_1000_buffers", |b| {
        b.iter(|| {
            let p = plan(black_box(&table));
            black_box(p);
        });
    });
}

fn bench_plan_with_budget_within(c: &mut Criterion) {
    let table = build_resnet_like_trace(250, 16);
    // Use the actual peak as budget so the budget check is satisfied.
    let peak = plan(&table).peak_bytes();
    c.bench_function("plan_with_budget_within_budget_1000_buffers", |b| {
        b.iter(|| {
            let p = plan_with_budget(black_box(&table), black_box(peak)).unwrap();
            black_box(p);
        });
    });
}

criterion_group!(
    benches,
    bench_plan_1000_buffers,
    bench_plan_with_budget_within
);
criterion_main!(benches);
