//! Benchmark: MinMax observer update over 1M elements.
//!
//! Plan P3 task `Int8 dtype, observers, and quantize/dequantize ops`
//! step #5 acceptance: observer update over 1M elements completes in
//! < 1 ms.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rustorch_quant::{quantize, MinMaxObserver, QParams};

fn fixture(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (i as f32 * 0.137).sin() * 4.0 - 1.0)
        .collect()
}

fn bench_minmax_1m(c: &mut Criterion) {
    let xs = fixture(1_000_000);
    c.bench_function("minmax_observer_update_1m", |b| {
        b.iter(|| {
            let mut o = MinMaxObserver::new();
            o.update(black_box(&xs));
            black_box(o);
        });
    });
}

fn bench_quantize_1m(c: &mut Criterion) {
    let xs = fixture(1_000_000);
    let qp = QParams::from_min_max(-5.0, 5.0).unwrap();
    let mut out = vec![0i8; xs.len()];
    c.bench_function("quantize_1m", |b| {
        b.iter(|| {
            quantize(black_box(&xs), &mut out, qp).unwrap();
            black_box(&out);
        });
    });
}

criterion_group!(benches, bench_minmax_1m, bench_quantize_1m);
criterion_main!(benches);
