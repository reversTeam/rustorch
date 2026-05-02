//! Benchmark: fused matmul+bias+activation vs naive sequential.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rustorch_fusion::{fused_matmul_bias_activation, naive_matmul_bias_activation, Activation};

fn fixture(m: usize, k: usize, n: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let x: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.1).sin()).collect();
    let w: Vec<f32> = (0..k * n).map(|i| (i as f32 * 0.07).cos()).collect();
    let b: Vec<f32> = (0..n).map(|i| i as f32 * 0.05).collect();
    (x, w, b)
}

fn bench_fused_relu(c: &mut Criterion) {
    let (m, k, n) = (256, 256, 256);
    let (x, w, b) = fixture(m, k, n);
    let mut y = vec![0.0f32; m * n];
    c.bench_function("fused_matmul_bias_relu_256x256x256", |bb| {
        bb.iter(|| {
            fused_matmul_bias_activation(
                black_box(&x),
                black_box(&w),
                Some(&b),
                &mut y,
                m,
                k,
                n,
                Activation::Relu,
            )
            .unwrap();
            black_box(&y);
        });
    });
}

fn bench_naive_relu(c: &mut Criterion) {
    let (m, k, n) = (256, 256, 256);
    let (x, w, b) = fixture(m, k, n);
    let mut y = vec![0.0f32; m * n];
    c.bench_function("naive_matmul_bias_relu_256x256x256", |bb| {
        bb.iter(|| {
            naive_matmul_bias_activation(
                black_box(&x),
                black_box(&w),
                Some(&b),
                &mut y,
                m,
                k,
                n,
                Activation::Relu,
            )
            .unwrap();
            black_box(&y);
        });
    });
}

criterion_group!(benches, bench_fused_relu, bench_naive_relu);
criterion_main!(benches);
