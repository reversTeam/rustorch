//! RusTorch benchmark harness — emits JSON to stdout.
//!
//! Each op is run once for warmup, then N times under wall-clock timing.
//! We report median + p99 (ns) so the comparison is robust to GC pauses,
//! thermal hiccups, etc.

use rustorch_attention::{flash_forward, naive_forward, AttentionShape};
use rustorch_core::Tensor;
use rustorch_cpu::backend::Backend;
use rustorch_cpu::cpu_backend::cpu_backend;
use serde_json::json;
use std::time::Instant;

const WARMUP: usize = 3;
const ITERS: usize = 20;

fn det(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed.wrapping_mul(6364136223846793005);
    (0..n)
        .map(|i| {
            s = s.wrapping_add(i as u64).wrapping_mul(0xff51afd7ed558ccd);
            let bits = (s >> 33) as u32;
            (bits as f32 / u32::MAX as f32 - 0.5) * 2.0
        })
        .collect()
}

fn time<F: FnMut()>(mut f: F) -> (u128, u128) {
    for _ in 0..WARMUP {
        f();
    }
    let mut samples = Vec::with_capacity(ITERS);
    for _ in 0..ITERS {
        let t0 = Instant::now();
        f();
        samples.push(t0.elapsed().as_nanos());
    }
    samples.sort();
    let median = samples[samples.len() / 2];
    let p99 = samples[(samples.len() * 99 / 100).min(samples.len() - 1)];
    (median, p99)
}

fn bench_matmul(m: usize, k: usize, n: usize) -> serde_json::Value {
    let a = Tensor::from_vec(vec![m, k], det(0xA1, m * k)).unwrap();
    let b = Tensor::from_vec(vec![k, n], det(0xB2, k * n)).unwrap();
    let (med, p99) = time(|| {
        let _c = cpu_backend().matmul(&a, &b).unwrap();
    });
    let flops = 2.0 * m as f64 * k as f64 * n as f64;
    let gflops = flops / (med as f64);
    json!({
        "op": "matmul",
        "shape": format!("[{},{}] x [{},{}]", m, k, k, n),
        "median_ns": med,
        "p99_ns": p99,
        "gflops": gflops,
    })
}

fn bench_softmax(rows: usize, cols: usize) -> serde_json::Value {
    let x = Tensor::from_vec(vec![rows, cols], det(0xCA, rows * cols)).unwrap();
    let (med, p99) = time(|| {
        let _y = rustorch_cpu::kernels::softmax::softmax(&x, 1).unwrap();
    });
    json!({
        "op": "softmax",
        "shape": format!("[{},{}]", rows, cols),
        "median_ns": med,
        "p99_ns": p99,
    })
}

fn bench_flash_attention(b: usize, h: usize, n: usize, d: usize) -> serde_json::Value {
    let shape = AttentionShape::new(b, h, n, d);
    let bl = shape.buffer_len();
    let q = det(0xCAFE, bl);
    let k = det(0xBEEF, bl);
    let v = det(0xF00D, bl);
    let mut out = vec![0.0_f32; bl];
    let (med, p99) = time(|| {
        flash_forward(&shape, &q, &k, &v, &mut out).unwrap();
    });
    json!({
        "op": "flash_attention_forward",
        "shape": format!("B={} H={} N={} D={}", b, h, n, d),
        "median_ns": med,
        "p99_ns": p99,
    })
}

fn bench_naive_attention(b: usize, h: usize, n: usize, d: usize) -> serde_json::Value {
    let shape = AttentionShape::new(b, h, n, d);
    let bl = shape.buffer_len();
    let q = det(0xCAFE, bl);
    let k = det(0xBEEF, bl);
    let v = det(0xF00D, bl);
    let mut out = vec![0.0_f32; bl];
    let (med, p99) = time(|| {
        naive_forward(&shape, &q, &k, &v, &mut out).unwrap();
    });
    json!({
        "op": "naive_attention_forward",
        "shape": format!("B={} H={} N={} D={}", b, h, n, d),
        "median_ns": med,
        "p99_ns": p99,
    })
}

fn bench_elementwise_add(n: usize) -> serde_json::Value {
    let a_data = det(0xA1, n);
    let b_data = det(0xB2, n);
    let b = Tensor::from_vec(vec![n], b_data).unwrap();
    // Each iter rebuilds `a` fresh so storage is unique (no aliased panic).
    let (med, p99) = time(|| {
        let mut a = Tensor::from_vec(vec![n], a_data.clone()).unwrap();
        a.add_(&b).unwrap();
    });
    json!({
        "op": "elementwise_add",
        "shape": format!("[{}]", n),
        "median_ns": med,
        "p99_ns": p99,
    })
}

fn main() {
    let mut results = Vec::new();

    // Matmul: small / medium / large
    for &(m, k, n) in &[
        (128, 128, 128),
        (256, 256, 256),
        (512, 512, 512),
        (1024, 1024, 1024),
    ] {
        results.push(bench_matmul(m, k, n));
    }

    // Softmax: typical LLM logits shapes
    results.push(bench_softmax(128, 32_000));
    results.push(bench_softmax(512, 50_257)); // GPT-2 vocab

    // Flash Attention (their flagship) vs naive
    for &(bsz, h, n, d) in &[(1, 1, 512, 64), (1, 4, 1024, 64), (2, 4, 2048, 64)] {
        results.push(bench_flash_attention(bsz, h, n, d));
        results.push(bench_naive_attention(bsz, h, n, d));
    }

    // Element-wise add
    for &n in &[1_000_usize, 100_000, 1_000_000] {
        results.push(bench_elementwise_add(n));
    }

    let out = json!({
        "framework": "rustorch",
        "version": rustorch_attention::VERSION,
        "warmup_iters": WARMUP,
        "timed_iters": ITERS,
        "results": results,
    });
    println!("{}", serde_json::to_string_pretty(&out).unwrap());
}
