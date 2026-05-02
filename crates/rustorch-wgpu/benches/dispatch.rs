//! Dispatch-side latency benchmarks for the wgpu backend.
//!
//! Measures the **CPU-side** cost of:
//! - `BroadcastPlan` construction (stride math, no GPU work)
//! - `split_dispatch` (the 1D-to-2D dispatch helper)
//! - `KernelRegistry::source` (string format on a hot path)
//!
//! These are the parts of the dispatch loop that don't depend on a
//! GPU adapter, so they run reliably on every CI worker. Full GPU
//! kernel timing requires a hardware runner and lives in
//! examples/bench_*.rs (not gated).

#![cfg(feature = "gpu-tests")]

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rustorch_wgpu::{broadcast_shape, KernelRegistry, OpId};

fn bench_broadcast_shape(c: &mut Criterion) {
    c.bench_function("broadcast_shape_4d", |b| {
        b.iter(|| {
            let plan = broadcast_shape(black_box(&[1, 2, 1, 4]), black_box(&[3, 1, 5, 1])).unwrap();
            black_box(plan)
        });
    });
}

fn bench_kernel_registry_source(c: &mut Criterion) {
    let reg = KernelRegistry::new();
    c.bench_function("kernel_registry_source_add", |b| {
        b.iter(|| black_box(reg.source(OpId::Add)));
    });
}

criterion_group!(benches, bench_broadcast_shape, bench_kernel_registry_source);
criterion_main!(benches);
