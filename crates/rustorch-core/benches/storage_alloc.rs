//! Benchmark: `Storage::cpu_zeroed` on 16 MiB buffers.
//!
//! P1.1 task `Storage enum with Cpu variant + refcount` step #5 — assert
//! that our 64-byte aligned allocator is within 10 % of `Vec<u8>::new`
//! (the system allocator baseline).

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rustorch_core::tensor::storage::Storage;

const SIZE: usize = 16 * 1024 * 1024;

fn bench_cpu_zeroed_16mib(c: &mut Criterion) {
    c.bench_function("storage_cpu_zeroed_16mib", |b| {
        b.iter(|| {
            let s = Storage::cpu_zeroed(black_box(SIZE)).unwrap();
            black_box(s);
        });
    });
}

fn bench_vec_zeroed_16mib(c: &mut Criterion) {
    c.bench_function("vec_zeroed_16mib_baseline", |b| {
        b.iter(|| {
            let v: Vec<u8> = vec![0u8; black_box(SIZE)];
            black_box(v);
        });
    });
}

criterion_group!(benches, bench_cpu_zeroed_16mib, bench_vec_zeroed_16mib);
criterion_main!(benches);
