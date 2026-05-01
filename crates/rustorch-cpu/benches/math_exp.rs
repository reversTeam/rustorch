//! Bench: `Backend::exp` on 1M f32.
//!
//! P1.3 task `Math ops` step #5 — within 30 % of libm peak.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rustorch_core::tensor::tensor_impl::Tensor;
use rustorch_cpu::cpu_backend::cpu_backend;

fn bench_exp_1m_f32(c: &mut Criterion) {
    let backend = cpu_backend();
    let data: Vec<f32> = (0..1_000_000).map(|i| (i as f32) * 1e-6).collect();
    let t = Tensor::from_vec([data.len()], data).unwrap();
    c.bench_function("backend_exp_1M_f32", |b| {
        b.iter(|| {
            let r = backend.exp(black_box(&t)).unwrap();
            black_box(r);
        });
    });
}

fn bench_libm_exp_1m_f32_baseline(c: &mut Criterion) {
    let data: Vec<f32> = (0..1_000_000).map(|i| (i as f32) * 1e-6).collect();
    c.bench_function("libm_exp_1M_f32_baseline", |b| {
        b.iter(|| {
            let mut out = Vec::with_capacity(data.len());
            for &x in &data {
                out.push(black_box(x).exp());
            }
            black_box(out);
        });
    });
}

criterion_group!(benches, bench_exp_1m_f32, bench_libm_exp_1m_f32_baseline);
criterion_main!(benches);
