//! Benchmark: QuantLinear inference vs naive f32 Linear-equivalent.
//!
//! Plan P3 task `Dynamic int8 quantization for Linear/Conv2d` step
//! #5 acceptance: QuantLinear inference 4× smaller weight memory;
//! ≥ 1.5× speedup vs f32. The 4× memory claim is structural
//! (i8 vs f32 = 1 vs 4 bytes per weight); the speedup depends on
//! the int8 GEMM speed (currently scalar — SIMD specialisations
//! land in arch-specific commits per task 8d4aa7de).

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rustorch_quant::{gemm_f32_reference, QParams, QuantLinear};

fn fixture_f32(n: usize, seed: u32) -> Vec<f32> {
    let mut s = seed.wrapping_mul(2654435761);
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            (s as f32 / u32::MAX as f32 - 0.5) * 2.0
        })
        .collect()
}

fn bench_quant_linear_512x512_b32(c: &mut Criterion) {
    // batch 32, in 512, out 512. Weight memory:
    //   f32: 512*512*4 = 1 MB
    //   i8 : 512*512*1 = 256 KB → 4× smaller (structural).
    let in_features = 512;
    let out_features = 512;
    let batch = 32;
    let weight_f32 = fixture_f32(in_features * out_features, 0xC0FFEE);
    let w_min = weight_f32.iter().copied().fold(f32::INFINITY, f32::min);
    let w_max = weight_f32.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let w_qp = QParams::from_min_max(w_min, w_max).unwrap();
    let layer = QuantLinear::from_f32_weights(&weight_f32, w_qp, None, in_features, out_features);
    let input = fixture_f32(batch * in_features, 0xBADBEEF);
    let i_qp = QParams::from_min_max(-1.0, 1.0).unwrap();
    let mut out = vec![0.0f32; batch * out_features];
    eprintln!(
        "[QuantLinear] weight i8 = {} KB vs f32 = {} KB (4× smaller)",
        layer.weight_q.len() / 1024,
        weight_f32.len() * 4 / 1024
    );
    c.bench_function("quant_linear_b32_in512_out512", |b| {
        b.iter(|| {
            layer
                .forward(black_box(&input), &mut out, batch, Some(i_qp))
                .unwrap();
            black_box(&out);
        });
    });
}

fn bench_f32_linear_512x512_b32(c: &mut Criterion) {
    let in_features = 512;
    let out_features = 512;
    let batch = 32;
    let weight = fixture_f32(in_features * out_features, 0xC0FFEE);
    let input = fixture_f32(batch * in_features, 0xBADBEEF);
    let mut out = vec![0.0f32; batch * out_features];
    c.bench_function("f32_linear_reference_b32_in512_out512", |b| {
        b.iter(|| {
            gemm_f32_reference(
                black_box(&input),
                black_box(&weight),
                &mut out,
                batch,
                in_features,
                out_features,
            );
            black_box(&out);
        });
    });
}

criterion_group!(
    benches,
    bench_quant_linear_512x512_b32,
    bench_f32_linear_512x512_b32
);
criterion_main!(benches);
