//! Benchmark: online softmax over 4096 entries vs naive single-pass.
//!
//! Plan P3 task `Online softmax (Welford-style max+sum aggregation)`
//! step #5 acceptance: online softmax on 4096 entries is within 1.1×
//! of naive single-pass. The online variant is "single-pass"
//! conceptually (one fold over the input) just like naive's
//! exp+sum loop after the max — but it pays an extra exp() per
//! element when a new max appears.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rustorch_attention::{combine_tiles, online_softmax_full, OnlineSoftmaxState};

fn fixture(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (i as f32 * 0.137).sin() * 4.0 - 1.0)
        .collect()
}

fn naive(xs: &[f32]) -> (f32, f32) {
    let m = xs.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let l: f32 = xs.iter().map(|x| (x - m).exp()).sum();
    (m, l)
}

fn bench_online_4096(c: &mut Criterion) {
    let xs = fixture(4096);
    c.bench_function("online_softmax_4096", |b| {
        b.iter(|| {
            let s = online_softmax_full(black_box(&xs));
            black_box(s);
        });
    });
}

fn bench_naive_4096(c: &mut Criterion) {
    let xs = fixture(4096);
    c.bench_function("naive_softmax_4096", |b| {
        b.iter(|| {
            let (m, l) = naive(black_box(&xs));
            black_box((m, l));
        });
    });
}

fn bench_tiled_combine_4096_x_64(c: &mut Criterion) {
    // Mimic the Flash Attention kernel: 4096 entries split into 64-
    // wide tiles, each tile's state computed and combined.
    let xs = fixture(4096);
    c.bench_function("tiled_64_combine_4096", |b| {
        b.iter(|| {
            let s = xs
                .chunks(64)
                .map(online_softmax_full)
                .fold(OnlineSoftmaxState::EMPTY, combine_tiles);
            black_box(s);
        });
    });
}

criterion_group!(
    benches,
    bench_online_4096,
    bench_naive_4096,
    bench_tiled_combine_4096_x_64
);
criterion_main!(benches);
