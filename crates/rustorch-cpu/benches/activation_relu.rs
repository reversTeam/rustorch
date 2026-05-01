//! Bench: `Backend::relu` on 1M f32.
//!
//! P1.3 task `Activations` step #5 — within 5 % of memory bandwidth.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rustorch_core::tensor::tensor_impl::Tensor;
use rustorch_cpu::cpu_backend::cpu_backend;

fn bench_relu_1m_f32(c: &mut Criterion) {
    let backend = cpu_backend();
    let data: Vec<f32> = (0..1_000_000).map(|i| (i as f32) - 500_000.0).collect();
    let t = Tensor::from_vec([data.len()], data).unwrap();
    c.bench_function("backend_relu_1M_f32", |b| {
        b.iter(|| {
            let r = backend.relu(black_box(&t)).unwrap();
            black_box(r);
        });
    });
}

fn bench_scalar_relu_1m_f32_baseline(c: &mut Criterion) {
    let data: Vec<f32> = (0..1_000_000).map(|i| (i as f32) - 500_000.0).collect();
    c.bench_function("scalar_relu_1M_f32_baseline", |b| {
        b.iter(|| {
            let mut out = Vec::with_capacity(data.len());
            for &x in &data {
                out.push(black_box(x).max(0.0));
            }
            black_box(out);
        });
    });
}

criterion_group!(
    benches,
    bench_relu_1m_f32,
    bench_scalar_relu_1m_f32_baseline
);
criterion_main!(benches);
