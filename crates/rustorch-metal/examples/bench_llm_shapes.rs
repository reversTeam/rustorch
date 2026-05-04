//! Compare Metal `simdgroup_matrix<f32>` vs Apple Accelerate AMX
//! `cblas_sgemm` on the exact shapes used by Qwen3-14B FFN/attention
//! projections. Drives the decision of whether to wire Metal into the
//! `rustorch-llm` decode hot path (Phase 2 of the LLM perf push).
//!
//! All Metal timings are bracketed by `backend.drain()` so we measure
//! committed compute time, not just the dispatch enqueue.
//!
//! Run with:
//! ```sh
//! cargo run -p rustorch-metal --release --example bench_llm_shapes
//! ```

#![cfg(target_os = "macos")]
#![allow(missing_docs)]

use rustorch_fusion::patterns::matmul_bias_act::{fused_matmul_bias_activation, Activation};
use rustorch_metal::backend_singleton::metal_backend;
use rustorch_metal::kernels::{
    matmul_simdgroup_f32, matmul_simdgroup_f32_coarsened, matmul_simdgroup_f32_coarsened_wide,
    matmul_simdgroup_f32_multisg,
};
use std::time::Instant;

fn det_vec(n: usize, seed: f32) -> Vec<f32> {
    (0..n)
        .map(|i| ((i as f32 + 1.0) * seed * 0.001).sin())
        .collect()
}

fn time_metal(m: usize, k: usize, n: usize, n_iter: usize) -> std::time::Duration {
    let backend = metal_backend();
    let a_buf = backend.alloc_shared(m * k * 4).expect("a alloc");
    let b_buf = backend.alloc_shared(k * n * 4).expect("b alloc");
    unsafe {
        let pa = a_buf.contents() as *mut f32;
        for (i, val) in det_vec(m * k, 1.0).iter().enumerate() {
            *pa.add(i) = *val;
        }
        let pb = b_buf.contents() as *mut f32;
        for (i, val) in det_vec(k * n, 0.5).iter().enumerate() {
            *pb.add(i) = *val;
        }
    }

    // Pick the fastest variant for the shape.
    let dispatch = |a: &metal::Buffer, b: &metal::Buffer| -> metal::Buffer {
        if m % 16 == 0 && n % 256 == 0 {
            matmul_simdgroup_f32_coarsened_wide(backend, a, b, m, k, n).unwrap()
        } else if n % 256 == 0 {
            matmul_simdgroup_f32_coarsened(backend, a, b, m, k, n).unwrap()
        } else if n % 64 == 0 {
            matmul_simdgroup_f32_multisg(backend, a, b, m, k, n).unwrap()
        } else {
            matmul_simdgroup_f32(backend, a, b, m, k, n).unwrap()
        }
    };

    // Warmup (compile pipeline + page in buffers + measure dispatch only).
    let _ = dispatch(&a_buf, &b_buf);
    backend.drain();

    let t0 = Instant::now();
    for _ in 0..n_iter {
        let _ = dispatch(&a_buf, &b_buf);
    }
    // Single drain at the end → average dispatch + compute time.
    backend.drain();
    t0.elapsed()
}

fn time_cpu(m: usize, k: usize, n: usize, n_iter: usize) -> std::time::Duration {
    let a = det_vec(m * k, 1.0);
    let b = det_vec(k * n, 0.5);
    let mut c = vec![0.0_f32; m * n];

    // Warmup.
    fused_matmul_bias_activation(&a, &b, None, &mut c, m, k, n, Activation::None).unwrap();

    let t0 = Instant::now();
    for _ in 0..n_iter {
        fused_matmul_bias_activation(&a, &b, None, &mut c, m, k, n, Activation::None).unwrap();
    }
    t0.elapsed()
}

fn flops(m: usize, k: usize, n: usize) -> f64 {
    2.0 * m as f64 * k as f64 * n as f64
}

fn run(label: &str, m: usize, k: usize, n: usize, n_iter: usize) {
    let metal_d = time_metal(m, k, n, n_iter);
    let cpu_d = time_cpu(m, k, n, n_iter);
    let metal_per = metal_d / n_iter as u32;
    let cpu_per = cpu_d / n_iter as u32;
    let metal_gflops = flops(m, k, n) / metal_per.as_secs_f64() / 1e9;
    let cpu_gflops = flops(m, k, n) / cpu_per.as_secs_f64() / 1e9;
    let speedup = cpu_per.as_secs_f64() / metal_per.as_secs_f64();
    println!(
        "{label:<28} [{m:>4}×{k:>5}] @ [{k:>5}×{n:>6}]  Metal {:>8.3}ms ({:>5.0} GFLOPS)  CPU AMX {:>8.3}ms ({:>5.0} GFLOPS)  speedup {:>4.1}×",
        metal_per.as_secs_f64() * 1e3,
        metal_gflops,
        cpu_per.as_secs_f64() * 1e3,
        cpu_gflops,
        speedup,
    );
}

fn main() {
    let backend = metal_backend();
    println!(
        "=== rustorch-metal vs rustorch-fusion (CBLAS AMX) — Qwen3-14B FFN/attn shapes ===\n\
         Adapter: {} | Metal3: {}\n",
        backend.adapter_name(),
        backend.supports_metal3(),
    );
    println!(
        "{:<28} {:>20}  {:>34}  {:>34}  {:>10}",
        "shape (label)", "shape", "Metal", "CPU AMX (cblas_sgemm)", "speedup"
    );
    println!("{:-<140}", "");

    let n_iter = 30;

    // Decode autoregressive: M=1, but kernels need M%8==0, so pad to M=8
    // (8× more compute, but B is loaded once per matmul so the cost is
    // dominated by B reads — should still beat CPU).
    let d = 5120;
    let f = 17408;
    let kv_dim = 1024;

    // Fused QKV: [8, D] @ [D, D + 2*KV_DIM] = [8, 5120] @ [5120, 7168]
    run("qkv_proj (M=8 padded)", 8, d, d + 2 * kv_dim, n_iter);

    // Fused gate_up: [8, D] @ [D, 2*F] = [8, 5120] @ [5120, 34816]
    run("gate_up_proj (M=8 padded)", 8, d, 2 * f, n_iter);

    // Down: [8, F] @ [F, D] = [8, 17408] @ [17408, 5120]
    run("down_proj (M=8 padded)", 8, f, d, n_iter);

    // O proj: [8, D] @ [D, D]
    run("o_proj (M=8 padded)", 8, d, d, n_iter);

    // LM head: [8, D] @ [D, V] (V padded to multiple of 256 below)
    let vocab = 151_936; // already mult of 64 (151936 = 593 × 256)
    run("lm_head (M=8 padded)", 8, d, vocab, n_iter);
}
