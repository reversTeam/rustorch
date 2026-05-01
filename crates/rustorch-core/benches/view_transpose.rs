//! Benchmark: `Tensor::transpose` on a [1024, 1024] tensor.
//!
//! P1.1 task `View operations` step #5 — target < 1 µs (metadata-only,
//! no data copy).

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rustorch_core::tensor::tensor_impl::Tensor;

fn bench_transpose_1024(c: &mut Criterion) {
    let t = Tensor::zeros([1024usize, 1024]);
    c.bench_function("view_transpose_1024x1024", |b| {
        b.iter(|| {
            let r = black_box(&t).transpose(0, 1).unwrap();
            black_box(r);
        });
    });
}

fn bench_permute_3d(c: &mut Criterion) {
    let t = Tensor::zeros([16usize, 32, 64]);
    c.bench_function("view_permute_3d_201", |b| {
        b.iter(|| {
            let r = black_box(&t).permute(black_box(&[2, 0, 1])).unwrap();
            black_box(r);
        });
    });
}

criterion_group!(benches, bench_transpose_1024, bench_permute_3d);
criterion_main!(benches);
