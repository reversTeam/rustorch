//! Benchmark: Flash forward vs naive at N=2048 (D=64).
//!
//! Plan P3 task `Flash Attention forward (CPU)` step #5 acceptance:
//! Flash shows O(N) peak memory and ≥ 1.5× time speedup vs naive at
//! N=2048. The naive path materialises the full N×N score matrix
//! (16 MB at f32, N=2048) which dwarfs Flash's per-tile working set
//! of bc × dim ≈ 16 KB.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rustorch_attention::{flash_forward, naive_forward, AttentionShape};

fn fixture(n_elements: usize, seed: u32) -> Vec<f32> {
    let mut s = seed.wrapping_mul(2654435761);
    (0..n_elements)
        .map(|_| {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            (s as f32 / u32::MAX as f32 - 0.5) * 2.0
        })
        .collect()
}

fn buffers(shape: &AttentionShape) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let n = shape.buffer_len();
    (
        fixture(n, 0xC0FFEE),
        fixture(n, 0xBADBEEF),
        fixture(n, 0xCAFE),
        vec![0.0; n],
    )
}

fn bench_flash_n2048(c: &mut Criterion) {
    let shape = AttentionShape::new(1, 1, 2048, 64);
    let (q, k, v, mut out) = buffers(&shape);
    c.bench_function("flash_forward_b1_h1_n2048_d64", |b| {
        b.iter(|| {
            flash_forward(black_box(&shape), &q, &k, &v, &mut out).unwrap();
            black_box(&out);
        });
    });
}

fn bench_naive_n2048(c: &mut Criterion) {
    let shape = AttentionShape::new(1, 1, 2048, 64);
    let (q, k, v, mut out) = buffers(&shape);
    c.bench_function("naive_forward_b1_h1_n2048_d64", |b| {
        b.iter(|| {
            naive_forward(black_box(&shape), &q, &k, &v, &mut out).unwrap();
            black_box(&out);
        });
    });
}

fn bench_flash_n1024_b2_h4(c: &mut Criterion) {
    // Realistic transformer batch: 2 sequences, 4 heads, 1024 tokens,
    // D=64. Tests that the parallel rayon path scales.
    let shape = AttentionShape::new(2, 4, 1024, 64);
    let (q, k, v, mut out) = buffers(&shape);
    c.bench_function("flash_forward_b2_h4_n1024_d64", |b| {
        b.iter(|| {
            flash_forward(black_box(&shape), &q, &k, &v, &mut out).unwrap();
            black_box(&out);
        });
    });
}

criterion_group!(
    benches,
    bench_flash_n2048,
    bench_naive_n2048,
    bench_flash_n1024_b2_h4
);
criterion_main!(benches);
