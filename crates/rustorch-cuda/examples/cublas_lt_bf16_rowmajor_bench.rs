//! `cublas_lt_bf16_rowmajor_bench` — DESIGN-PIVOT validation bench.
//!
//! Compares throughput of :
//!   1. `LtSession::matmul_bf16_rowmajor` (cuBLASLt, vendor tensor-core kernel)
//!   2. `LlmKernels::gemm_bf16_bf16_mma` (hand-written mma.sync m16n8k16)
//!   3. `LlmKernels::sgemm_bf16_bf16_mvar` (warp-shuffle scalar baseline)
//!
//! On the prefill shapes that matter for Qwen3.6 (M ∈ {16, 64, 128, 512},
//! K=2048, N=4096), reports per-call ms, effective bandwidth (GB/s), and
//! TFLOPS.
//!
//! Run :
//!   PATH=/usr/local/cuda-13.0/bin:$PATH \
//!     LD_LIBRARY_PATH=/usr/local/cuda-13.0/targets/sbsa-linux/lib:$LD_LIBRARY_PATH \
//!     cargo run --release --features cuda -p rustorch-cuda \
//!       --example cublas_lt_bf16_rowmajor_bench

#[cfg(not(feature = "cuda"))]
fn main() {
    println!("[bench] cuda feature is OFF — nothing to bench.");
}

#[cfg(feature = "cuda")]
fn main() {
    use cudarc::driver::{CudaContext, DevicePtr, DevicePtrMut};
    use rustorch_cuda::cublas_lt::LtSession;
    use rustorch_cuda::llm_kernels::LlmKernels;
    use std::time::Instant;

    let ctx = CudaContext::new(0).expect("ctx");
    let stream = ctx.default_stream();
    let kernels = LlmKernels::new(ctx.clone());
    let mut session = LtSession::new(stream.clone()).expect("LtSession");

    // ───────────────────────────────────────────────────────────────
    // GB10 unified memory peak BW is 200 GB/s. Theoretical BF16 GEMM
    // arith intensity for [M,K]·[K,N]=[M,N] = 2·M·N·K FLOPs over
    // (M·K + K·N + M·N)·2 bytes — large K means bandwidth bound for
    // M small, compute bound for M large (transition near M ≈ 50 on
    // BF16 tensor cores).
    // ───────────────────────────────────────────────────────────────

    let shapes: &[(usize, usize, usize)] = &[
        // Attention dense W_q/W_k/W_v/W_o on Qwen3.6 hidden=2048.
        (16, 4096, 2048),
        (64, 4096, 2048),
        (128, 4096, 2048),
        (256, 4096, 2048),
        (512, 4096, 2048),
        // SSM in_proj / out_proj rough shape.
        (128, 2048, 2048),
        (512, 2048, 2048),
    ];

    println!(
        "shape (M, N, K)      | cublasLt ms |   mma ms |  mvar ms || cublasLt GB/s | mma GB/s | mvar GB/s || cublasLt TFLOPS | mma TFLOPS"
    );
    println!("--------------------+------------+----------+----------+---------------+----------+-----------+-----------------+-----------");

    for &(m, n, k) in shapes {
        let w: Vec<half::bf16> = (0..n * k)
            .map(|i| half::bf16::from_f32(((i % 17) as f32 - 8.0) * 0.001))
            .collect();
        let x: Vec<half::bf16> = (0..m * k)
            .map(|i| half::bf16::from_f32(((i % 13) as f32 - 6.0) * 0.001))
            .collect();
        let w_dev = stream.memcpy_stod(&w).expect("w");
        let x_dev = stream.memcpy_stod(&x).expect("x");
        let mut y_cublas = stream.alloc_zeros::<half::bf16>(m * n).expect("y_cublas");
        let mut y_mma = stream.alloc_zeros::<half::bf16>(m * n).expect("y_mma");
        let mut y_mvar = stream.alloc_zeros::<half::bf16>(m * n).expect("y_mvar");

        // Warm-up each path
        unsafe {
            let (wp, _g1) = w_dev.device_ptr(&stream);
            let (xp, _g2) = x_dev.device_ptr(&stream);
            let (yc, _g3) = y_cublas.device_ptr_mut(&stream);
            session
                .matmul_bf16_rowmajor(wp, xp, yc, m, n, k, 1.0, 0.0)
                .expect("warmup cublas");
            let (ym, _g4) = y_mma.device_ptr_mut(&stream);
            kernels
                .gemm_bf16_bf16_mma(&stream, wp, xp, ym, m as i32, n as i32, k as i32)
                .expect("warmup mma");
            let (yv, _g5) = y_mvar.device_ptr_mut(&stream);
            kernels
                .sgemm_bf16_bf16_mvar(&stream, wp, xp, yv, m as i32, n as i32, k as i32)
                .expect("warmup mvar");
        }
        stream.synchronize().expect("sync");

        let iters = 32usize;
        // cuBLASLt timing
        let t0 = Instant::now();
        unsafe {
            let (wp, _g1) = w_dev.device_ptr(&stream);
            let (xp, _g2) = x_dev.device_ptr(&stream);
            let (yc, _g3) = y_cublas.device_ptr_mut(&stream);
            for _ in 0..iters {
                session
                    .matmul_bf16_rowmajor(wp, xp, yc, m, n, k, 1.0, 0.0)
                    .expect("cublas");
            }
        }
        stream.synchronize().expect("sync");
        let cublas_ms = t0.elapsed().as_secs_f64() * 1000.0 / iters as f64;

        // mma timing
        let t0 = Instant::now();
        unsafe {
            let (wp, _g1) = w_dev.device_ptr(&stream);
            let (xp, _g2) = x_dev.device_ptr(&stream);
            let (ym, _g3) = y_mma.device_ptr_mut(&stream);
            for _ in 0..iters {
                kernels
                    .gemm_bf16_bf16_mma(&stream, wp, xp, ym, m as i32, n as i32, k as i32)
                    .expect("mma");
            }
        }
        stream.synchronize().expect("sync");
        let mma_ms = t0.elapsed().as_secs_f64() * 1000.0 / iters as f64;

        // mvar timing
        let t0 = Instant::now();
        unsafe {
            let (wp, _g1) = w_dev.device_ptr(&stream);
            let (xp, _g2) = x_dev.device_ptr(&stream);
            let (yv, _g3) = y_mvar.device_ptr_mut(&stream);
            for _ in 0..iters {
                kernels
                    .sgemm_bf16_bf16_mvar(&stream, wp, xp, yv, m as i32, n as i32, k as i32)
                    .expect("mvar");
            }
        }
        stream.synchronize().expect("sync");
        let mvar_ms = t0.elapsed().as_secs_f64() * 1000.0 / iters as f64;

        // GB / TFLOPS
        let bytes_total = (m * k + n * k + m * n) as f64 * 2.0; // BF16
        let flops = 2.0 * m as f64 * n as f64 * k as f64;
        let bw = |ms: f64| bytes_total / (ms * 1e-3) / 1e9;
        let tflops = |ms: f64| flops / (ms * 1e-3) / 1e12;

        println!(
            "M={m:4} N={n:5} K={k:5} | {cublas_ms:9.4}  | {mma_ms:7.4} | {mvar_ms:7.4} || {:9.1}    | {:7.1}  | {:7.1}   || {:9.2}      | {:7.2}",
            bw(cublas_ms),
            bw(mma_ms),
            bw(mvar_ms),
            tflops(cublas_ms),
            tflops(mma_ms),
        );
    }
    println!("\nGB10 unified peak : 200 GB/s. Target for cuBLASLt : >= 150 GB/s.");
}
