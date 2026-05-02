//! Benchmark: scalar int8 GEMM vs f32 reference.
//!
//! Plan P3 task `SIMD int8 kernels` step #5 acceptance: int8 gemm
//! ≥ 2× f32 at 2048×2048 ON AN AVX-VNNI HOST. The scalar fallback
//! shipped here will NOT beat f32 — that's the whole point of the
//! SIMD specialisations (vpdpbusd packs 4 int8 products per
//! instruction, sdot does 4 per cycle). What this bench documents:
//!
//! 1. The scalar baseline timing — every SIMD specialisation must
//!    beat THIS to be worth landing.
//! 2. The f32 reference timing — what int8 needs to beat by 2× to
//!    meet the acceptance.
//! 3. The expected speedup from SIMD = ~4× over scalar int8 (rough
//!    rule of thumb for vpdpbusd / sdot vs scalar i32 mul-add).
//!
//! Bench is at 256×256×256 (smaller than the 2048× target) to keep
//! the scalar variant tractable in CI; the AVX-VNNI/NEON paths will
//! re-run at 2048× when they land.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rustorch_quant::{gemm_f32_reference, gemm_int8_scalar, QParams};

fn fixture_f32(n: usize, seed: u32) -> Vec<f32> {
    let mut s = seed.wrapping_mul(2654435761);
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            (s as f32 / u32::MAX as f32 - 0.5) * 2.0
        })
        .collect()
}

fn fixture_i8(n: usize, seed: u32) -> Vec<i8> {
    let mut s = seed.wrapping_mul(2654435761);
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            (s as i32 % 256 - 128) as i8
        })
        .collect()
}

fn bench_int8_256(c: &mut Criterion) {
    let m = 256;
    let k = 256;
    let n = 256;
    let a = fixture_i8(m * k, 0xC0FFEE);
    let b = fixture_i8(k * n, 0xBADBEEF);
    let mut out = vec![0.0f32; m * n];
    let qp = QParams {
        scale: 0.01,
        zero_point: 0,
    };
    c.bench_function("int8_gemm_scalar_256x256x256", |b_| {
        b_.iter(|| {
            gemm_int8_scalar(black_box(&a), black_box(&b), &mut out, m, k, n, qp, qp).unwrap();
            black_box(&out);
        });
    });
}

fn bench_f32_256(c: &mut Criterion) {
    let m = 256;
    let k = 256;
    let n = 256;
    let a = fixture_f32(m * k, 0xC0FFEE);
    let b = fixture_f32(k * n, 0xBADBEEF);
    let mut out = vec![0.0f32; m * n];
    c.bench_function("f32_gemm_reference_256x256x256", |b_| {
        b_.iter(|| {
            gemm_f32_reference(black_box(&a), black_box(&b), &mut out, m, k, n);
            black_box(&out);
        });
    });
}

criterion_group!(benches, bench_int8_256, bench_f32_256);
criterion_main!(benches);
