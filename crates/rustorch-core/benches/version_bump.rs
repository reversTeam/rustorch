//! Benchmark: `VersionCounter::bump` overhead.
//!
//! P1.1 task `Version counter for in-place mutation safety` step #5 —
//! target < 5 ns per call (Relaxed atomic fetch_add).

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rustorch_core::tensor::version::VersionCounter;

fn bench_bump(c: &mut Criterion) {
    let v = VersionCounter::new();
    c.bench_function("version_counter_bump", |b| {
        b.iter(|| {
            black_box(v.bump());
        });
    });
}

fn bench_current(c: &mut Criterion) {
    let v = VersionCounter::new();
    c.bench_function("version_counter_current", |b| {
        b.iter(|| {
            black_box(v.current());
        });
    });
}

criterion_group!(benches, bench_bump, bench_current);
criterion_main!(benches);
