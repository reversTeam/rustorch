//! Test whether Apple M4 Max has dual AMX coprocessors usable by
//! parallel `cblas_sgemm` calls. If two independent sgemms can run
//! truly in parallel on two threads, we can split `gate` || `up`
//! into separate threads and beat the fused single-call path on
//! decode (where M=1 — fusion's only win is the shared loads of `h`,
//! which is tiny vs the weight reads).
//!
//! Run with:
//! ```sh
//! cargo run -p rustorch-fusion --release --example bench_parallel_sgemm
//! ```

#![allow(missing_docs)]

use rustorch_fusion::patterns::matmul_bias_act::{fused_matmul_bias_activation, Activation};
use std::time::Instant;

fn det_vec(n: usize, seed: f32) -> Vec<f32> {
    (0..n)
        .map(|i| ((i as f32 + 1.0) * seed * 0.001).sin())
        .collect()
}

fn time<F: FnMut()>(mut f: F, n_iter: usize) -> std::time::Duration {
    // Warmup.
    f();
    let t0 = Instant::now();
    for _ in 0..n_iter {
        f();
    }
    t0.elapsed()
}

fn main() {
    // Qwen3-14B FFN gate/up shape: M=1 (decode), K=5120, N=17408.
    let m = 1;
    let k = 5120;
    let n = 17408;
    let n_iter = 200;

    let h = det_vec(m * k, 1.0);
    let w_gate = det_vec(k * n, 0.5);
    let w_up = det_vec(k * n, 0.7);
    let w_gate_up = det_vec(k * 2 * n, 0.6); // fused [k, 2n]
    let mut gate_out = vec![0.0_f32; m * n];
    let mut up_out = vec![0.0_f32; m * n];
    let mut gate_up_out = vec![0.0_f32; m * 2 * n];

    println!("=== Apple M4 Max — sgemv parallelization test ===");
    println!("Shape per sgemv: M={m}, K={k}, N={n}  (Qwen3-14B FFN gate/up)");
    println!("Iterations:     {n_iter}\n");

    // Variant A: ONE fused sgemv [M, K] @ [K, 2N] (current T66 path).
    let dur_fused = time(
        || {
            fused_matmul_bias_activation(
                &h,
                &w_gate_up,
                None,
                &mut gate_up_out,
                m,
                k,
                2 * n,
                Activation::None,
            )
            .unwrap();
        },
        n_iter,
    );

    // Variant B: TWO sgemvs sequentially on the same thread.
    let dur_seq2 = time(
        || {
            fused_matmul_bias_activation(
                &h,
                &w_gate,
                None,
                &mut gate_out,
                m,
                k,
                n,
                Activation::None,
            )
            .unwrap();
            fused_matmul_bias_activation(&h, &w_up, None, &mut up_out, m, k, n, Activation::None)
                .unwrap();
        },
        n_iter,
    );

    // Variant C: TWO sgemvs in PARALLEL via rayon::join.
    // Tests whether the M4 Max has dual AMX coprocessors that two
    // simultaneous cblas_sgemm calls can saturate independently.
    let dur_par2 = time(
        || {
            // We can't borrow gate_out/up_out mutably across the closure
            // boundary if we want true parallelism, so allocate fresh
            // outputs inside each closure (the alloc cost is dominated
            // by the sgemm itself for this shape).
            rayon::join(
                || {
                    let mut go = vec![0.0_f32; m * n];
                    fused_matmul_bias_activation(
                        &h,
                        &w_gate,
                        None,
                        &mut go,
                        m,
                        k,
                        n,
                        Activation::None,
                    )
                    .unwrap();
                },
                || {
                    let mut uo = vec![0.0_f32; m * n];
                    fused_matmul_bias_activation(
                        &h,
                        &w_up,
                        None,
                        &mut uo,
                        m,
                        k,
                        n,
                        Activation::None,
                    )
                    .unwrap();
                },
            );
        },
        n_iter,
    );

    let ms_fused = dur_fused.as_secs_f64() * 1000.0 / n_iter as f64;
    let ms_seq2 = dur_seq2.as_secs_f64() * 1000.0 / n_iter as f64;
    let ms_par2 = dur_par2.as_secs_f64() * 1000.0 / n_iter as f64;
    println!(
        "Variant A — 1 fused [M, 2N]:               {:>7.3} ms/call    {:>5.1}× of fused",
        ms_fused, 1.0
    );
    println!(
        "Variant B — 2 separate (sequential):      {:>7.3} ms/call    {:>5.2}× of fused",
        ms_seq2,
        ms_seq2 / ms_fused
    );
    println!(
        "Variant C — 2 separate (rayon::join):     {:>7.3} ms/call    {:>5.2}× of fused",
        ms_par2,
        ms_par2 / ms_fused
    );
    println!();
    println!(
        "Speedup C vs A (fused) : {:.2}× — does parallel beat fused?",
        ms_fused / ms_par2
    );
    println!(
        "Speedup C vs B (seq2)  : {:.2}× — does dual AMX exist?",
        ms_seq2 / ms_par2
    );
}
