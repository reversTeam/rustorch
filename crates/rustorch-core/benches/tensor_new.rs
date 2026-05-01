//! Benchmark: `Tensor::new` / `Tensor::zeros` on shape [1024, 1024].
//!
//! P1.1 task `Tensor struct definition` step #5 — assert that our
//! Tensor allocation is within 5 % of `ndarray::Array2::zeros`.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rustorch_core::tensor::dtype::Dtype;
use rustorch_core::tensor::tensor_impl::Tensor;

fn bench_tensor_zeros_1024(c: &mut Criterion) {
    c.bench_function("tensor_zeros_1024x1024_f32", |b| {
        b.iter(|| {
            let t = Tensor::zeros_dtype(black_box([1024usize, 1024]), black_box(Dtype::F32));
            black_box(t);
        });
    });
}

#[cfg(feature = "ndarray")]
fn bench_ndarray_zeros_1024(c: &mut Criterion) {
    c.bench_function("ndarray_zeros_1024x1024_f32", |b| {
        b.iter(|| {
            let arr = ndarray::Array2::<f32>::zeros((black_box(1024), black_box(1024)));
            black_box(arr);
        });
    });
}

#[cfg(feature = "ndarray")]
criterion_group!(benches, bench_tensor_zeros_1024, bench_ndarray_zeros_1024);

#[cfg(not(feature = "ndarray"))]
criterion_group!(benches, bench_tensor_zeros_1024);

criterion_main!(benches);
