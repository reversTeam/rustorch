//! Benchmark: `Layout::contiguous` construction.
//!
//! P1.1 task `Layout struct` step #5 — target < 50 ns for 4-D shapes.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rustorch_core::tensor::dtype::Dtype;
use rustorch_core::tensor::layout::Layout;

fn bench_layout_contiguous_4d(c: &mut Criterion) {
    c.bench_function("layout_contiguous_4d", |b| {
        b.iter(|| {
            let l = Layout::contiguous(black_box([8usize, 32, 224, 224]), black_box(Dtype::F32));
            black_box(l);
        });
    });
}

fn bench_layout_contiguous_2d(c: &mut Criterion) {
    c.bench_function("layout_contiguous_2d", |b| {
        b.iter(|| {
            let l = Layout::contiguous(black_box([1024usize, 1024]), black_box(Dtype::F32));
            black_box(l);
        });
    });
}

criterion_group!(
    benches,
    bench_layout_contiguous_4d,
    bench_layout_contiguous_2d
);
criterion_main!(benches);
