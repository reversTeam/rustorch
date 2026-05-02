//! Benchmark: long-sequence Flash forward vs naive at N=8192.
//!
//! Plan P3 task `Mathematical equivalence + end-to-end validation`
//! step #4 acceptance: at N=8192 D=64, Flash uses < 100 MB peak vs
//! naive ~2 GB; Flash runs ≥ 2× faster.
//!
//! ## Why naive blows up
//! Naive attention materialises the full N×N score matrix: at
//! N=8192 that's 8192² = 67 108 864 f32 cells = **256 MB per (b, h)
//! pair** just for scores. With B=1 H=1 D=64 we also need Q/K/V/O
//! each N*D*4 = 2 MB. Total naive peak ≈ **258 MB**.
//!
//! Flash never materialises N×N — it streams over Bc-sized tiles.
//! Per parallel task: `bc * dim * 4` for scores buffer + dim float
//! state + dim float output accumulator ≈ 64*64*4 + 64*8 ≈ **~16 KB**.
//! With one rayon task per (b, h) plus per-thread temporaries, peak
//! stays comfortably under 100 MB even at N=8192 (Q/K/V/O input
//! buffers themselves are 2 MB each = ~10 MB total).

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

fn bench_flash_n8192(c: &mut Criterion) {
    // B=1 H=1 N=8192 D=64. Buffers ≈ 2 MB each.
    let shape = AttentionShape::new(1, 1, 8192, 64);
    let (q, k, v, mut out) = buffers(&shape);
    eprintln!(
        "[long-seq] flash buffers ≈ {} MB each",
        shape.buffer_len() * 4 / (1024 * 1024)
    );
    c.bench_function("flash_forward_b1_h1_n8192_d64", |b| {
        b.iter(|| {
            flash_forward(black_box(&shape), &q, &k, &v, &mut out).unwrap();
            black_box(&out);
        });
    });
}

fn bench_naive_n8192(c: &mut Criterion) {
    // Naive needs to materialise an 8192×8192 f32 score matrix
    // = 256 MB. Allocate carefully — this bench is the bound the
    // Flash variant beats.
    let shape = AttentionShape::new(1, 1, 8192, 64);
    let (q, k, v, mut out) = buffers(&shape);
    eprintln!("[long-seq] naive will allocate 256 MB scores per call");
    c.bench_function("naive_forward_b1_h1_n8192_d64", |b| {
        b.iter(|| {
            naive_forward(black_box(&shape), &q, &k, &v, &mut out).unwrap();
            black_box(&out);
        });
    });
}

criterion_group!(benches, bench_flash_n8192, bench_naive_n8192);
criterion_main!(benches);
