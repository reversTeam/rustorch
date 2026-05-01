//! Benchmark: `Shape::broadcast_with` on 4-D shapes.
//!
//! P1.1 task `Shape newtype + broadcasting helpers` step #5 — target
//! < 100 ns per call on [N,C,H,W] vs [C,1,1].

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rustorch_core::tensor::shape::Shape;

fn bench_broadcast_4d(c: &mut Criterion) {
    let a = Shape::from([8usize, 32, 224, 224]);
    let b = Shape::from([32usize, 1, 1]);
    c.bench_function("shape_broadcast_4d_NCHW_vs_C11", |bencher| {
        bencher.iter(|| {
            let r = black_box(&a).broadcast_with(black_box(&b)).unwrap();
            black_box(r);
        });
    });
}

fn bench_broadcast_identical(c: &mut Criterion) {
    let a = Shape::from([8usize, 32, 224, 224]);
    c.bench_function("shape_broadcast_identical_4d", |bencher| {
        bencher.iter(|| {
            let r = black_box(&a).broadcast_with(black_box(&a)).unwrap();
            black_box(r);
        });
    });
}

criterion_group!(benches, bench_broadcast_4d, bench_broadcast_identical);
criterion_main!(benches);
