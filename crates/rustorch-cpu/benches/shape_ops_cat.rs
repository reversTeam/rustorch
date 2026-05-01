//! Bench: `Backend::cat` of 4× [1024, 1024] f32 tensors along dim 0.
//!
//! P1.3 task `Shape ops` step #5 — within 90 % of memcpy peak.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rustorch_core::tensor::tensor_impl::Tensor;
use rustorch_cpu::cpu_backend::cpu_backend;

fn bench_cat_4x_1024_f32(c: &mut Criterion) {
    let backend = cpu_backend();
    let make = || Tensor::from_vec([1024usize, 1024], vec![0.0_f32; 1024 * 1024]).unwrap();
    let a = make();
    let b = make();
    let cc = make();
    let d = make();
    c.bench_function("backend_cat_4x1024x1024_f32_dim0", |bencher| {
        bencher.iter(|| {
            let r = backend
                .cat(black_box(&[&a, &b, &cc, &d]), black_box(0))
                .unwrap();
            black_box(r);
        });
    });
}

fn bench_memcpy_4x_1024_f32_baseline(c: &mut Criterion) {
    let n = 4 * 1024 * 1024;
    let bytes = n * core::mem::size_of::<f32>();
    let src = vec![0u8; bytes];
    c.bench_function("memcpy_4x1024x1024_f32_baseline", |b| {
        b.iter(|| {
            let mut dst = vec![0u8; bytes];
            dst.copy_from_slice(&src);
            black_box(dst);
        });
    });
}

criterion_group!(
    benches,
    bench_cat_4x_1024_f32,
    bench_memcpy_4x_1024_f32_baseline
);
criterion_main!(benches);
