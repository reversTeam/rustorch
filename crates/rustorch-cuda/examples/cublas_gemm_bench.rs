//! `cublas_gemm_bench` — measure cuBLAS gemm throughput on the active device.
//!
//! Runs `cublasSgemm` over a sweep of square shapes, allocating device
//! buffers once and looping many gemm calls so the measurement isolates
//! compute (no H2D/D2H per iteration). Reports wall-time and TFLOPS.
//!
//! ## Usage
//!
//! ```bash
//! # On DGX (with CUDA toolkit on PATH):
//! cargo run --release -p rustorch-cuda --features cuda --example cublas_gemm_bench
//!
//! # Optional: pick the device index (default 0)
//! cargo run --release -p rustorch-cuda --features cuda --example cublas_gemm_bench -- 0
//! ```
//!
//! Without `--features cuda`, prints a notice and exits cleanly so CI
//! can build the example on any host without failing.

#[cfg(not(feature = "cuda"))]
fn main() {
    println!("[cublas_gemm_bench] cuda feature is OFF — nothing to bench.");
    println!("[cublas_gemm_bench] Re-run with: cargo run --release -p rustorch-cuda --features cuda --example cublas_gemm_bench");
}

/// Lightweight String-backed error so we can `?` through cudarc's
/// CublasError + DriverError without pulling in anyhow as a dev-dep.
#[cfg(feature = "cuda")]
#[derive(Debug)]
struct BenchError(String);

#[cfg(feature = "cuda")]
impl std::fmt::Display for BenchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

#[cfg(feature = "cuda")]
impl std::error::Error for BenchError {}

#[cfg(feature = "cuda")]
impl From<cudarc::driver::DriverError> for BenchError {
    fn from(e: cudarc::driver::DriverError) -> Self {
        Self(format!("driver: {e:?}"))
    }
}

#[cfg(feature = "cuda")]
impl From<cudarc::cublas::result::CublasError> for BenchError {
    fn from(e: cudarc::cublas::result::CublasError) -> Self {
        Self(format!("cublas: {e:?}"))
    }
}

#[cfg(feature = "cuda")]
fn main() -> Result<(), BenchError> {
    use cudarc::cublas::{sys, CudaBlas, Gemm, GemmConfig};
    use cudarc::driver::CudaContext;
    use std::time::Instant;

    let device_index: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);

    println!("[cublas_gemm_bench] initialising CUDA on device {device_index}");
    let ctx = CudaContext::new(device_index)?;
    let stream = ctx.default_stream();
    let blas = CudaBlas::new(stream.clone())?;

    // Enable TF32 TensorCore path. Without this, cublasSgemm stays on
    // the slow FP32 ALU path. CUBLAS_TF32_TENSOR_OP_MATH = 3.
    // SAFETY: handle is valid (just constructed), mode is in-range.
    unsafe {
        let status = cudarc::cublas::sys::cublasSetMathMode(
            *blas.handle(),
            cudarc::cublas::sys::cublasMath_t::CUBLAS_TF32_TENSOR_OP_MATH,
        );
        if status != cudarc::cublas::sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
            return Err(BenchError(format!(
                "cublasSetMathMode(TF32) failed: status={status:?}"
            )));
        }
    }

    println!("[cublas_gemm_bench] device ready, sweeping (TF32 + BF16 + FP8 E4M3)");
    println!();
    println!(
        "  {:>5} {:>8} {:>10} {:>10} {:>10}",
        "M=N=K", "iters", "TF32 TFL", "BF16 TFL", "FP8 TFL"
    );
    println!("  {}", "─".repeat(56));

    // Square shape sweep. Shapes chosen to span small (cache-resident-ish)
    // up to 4096² which is the typical "large dense gemm" sweet spot.
    let shapes = [128usize, 256, 512, 1024, 2048, 4096];

    for &dim in &shapes {
        let m = dim;
        let n = dim;
        let k = dim;
        let n_iters = if dim <= 512 {
            200
        } else if dim <= 2048 {
            50
        } else {
            10
        };

        // Allocate device buffers once. Filled with deterministic small
        // values so any later checksum is stable run-to-run.
        let a_host: Vec<f32> = (0..m * k).map(|i| ((i as f32) * 0.001).sin()).collect();
        let b_host: Vec<f32> = (0..k * n).map(|i| ((i as f32) * 0.001).cos()).collect();

        let a_dev = stream.memcpy_stod(&a_host)?;
        let b_dev = stream.memcpy_stod(&b_host)?;
        let mut c_dev = stream.alloc_zeros::<f32>(m * n)?;

        let cfg = GemmConfig::<f32> {
            transa: sys::cublasOperation_t::CUBLAS_OP_N,
            transb: sys::cublasOperation_t::CUBLAS_OP_N,
            m: n as i32,
            n: m as i32,
            k: k as i32,
            alpha: 1.0,
            lda: n as i32,
            ldb: k as i32,
            beta: 0.0,
            ldc: n as i32,
        };

        // Warm-up: 5 calls to amortize first-launch JIT / handle setup.
        for _ in 0..5 {
            // SAFETY: shapes match cfg, a/b/c are alloc'd to the right sizes.
            unsafe {
                blas.gemm(cfg, &b_dev, &a_dev, &mut c_dev)?;
            }
        }
        stream.synchronize()?;

        let t0 = Instant::now();
        for _ in 0..n_iters {
            unsafe {
                blas.gemm(cfg, &b_dev, &a_dev, &mut c_dev)?;
            }
        }
        stream.synchronize()?;
        let wall_ms_tf32 = t0.elapsed().as_secs_f64() * 1000.0;
        let ms_per_iter_tf32 = wall_ms_tf32 / n_iters as f64;
        // FLOPS for a sgemm: 2 · M · N · K (mul + add per element).
        let flops_per_iter = 2.0 * (m as f64) * (n as f64) * (k as f64);
        let tf32_tflops = flops_per_iter / (ms_per_iter_tf32 / 1000.0) / 1e12;

        // ───────── BF16 path: same sweep, fresh buffers in bf16 ─────────
        let a_bf16: Vec<half::bf16> = a_host.iter().map(|x| half::bf16::from_f32(*x)).collect();
        let b_bf16: Vec<half::bf16> = b_host.iter().map(|x| half::bf16::from_f32(*x)).collect();
        let a_dev_bf16 = stream.memcpy_stod(&a_bf16)?;
        let b_dev_bf16 = stream.memcpy_stod(&b_bf16)?;
        let mut c_dev_bf16 = stream.alloc_zeros::<half::bf16>(m * n)?;
        let cfg_bf16 = cudarc::cublas::GemmConfig::<half::bf16> {
            transa: cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
            transb: cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
            m: n as i32,
            n: m as i32,
            k: k as i32,
            alpha: half::bf16::from_f32(1.0),
            lda: n as i32,
            ldb: k as i32,
            beta: half::bf16::from_f32(0.0),
            ldc: n as i32,
        };
        for _ in 0..5 {
            unsafe {
                blas.gemm(cfg_bf16, &b_dev_bf16, &a_dev_bf16, &mut c_dev_bf16)?;
            }
        }
        stream.synchronize()?;
        let t0 = Instant::now();
        for _ in 0..n_iters {
            unsafe {
                blas.gemm(cfg_bf16, &b_dev_bf16, &a_dev_bf16, &mut c_dev_bf16)?;
            }
        }
        stream.synchronize()?;
        let wall_ms_bf16 = t0.elapsed().as_secs_f64() * 1000.0;
        let ms_per_iter_bf16 = wall_ms_bf16 / n_iters as f64;
        let bf16_tflops = flops_per_iter / (ms_per_iter_bf16 / 1000.0) / 1e12;
        let _ = (wall_ms_tf32, ms_per_iter_tf32, ms_per_iter_bf16);

        // ───────── FP8 (E4M3) via our cublas_lt::matmul_fp8 wrapper ─────────
        let a_fp8: Vec<u8> = (0..m * k)
            .map(|i| 0x10u8.wrapping_add((i as u8) % 0x60u8))
            .collect();
        let b_fp8: Vec<u8> = (0..k * n)
            .map(|i| 0x10u8.wrapping_add((i.wrapping_mul(7) as u8) % 0x60u8))
            .collect();
        let a_dev_fp8 = stream.memcpy_stod(&a_fp8)?;
        let b_dev_fp8 = stream.memcpy_stod(&b_fp8)?;
        let mut c_dev_fp8 = stream.alloc_zeros::<half::bf16>(m * n)?;
        let workspace_size = 32 * 1024 * 1024usize;
        let workspace_dev = stream.alloc_zeros::<u8>(workspace_size)?;

        let fp8_attempt = || -> Result<f64, BenchError> {
            use cudarc::driver::{DevicePtr, DevicePtrMut};
            // Warm-up.
            for _ in 0..5 {
                unsafe {
                    let (a_ptr, _r1) = a_dev_fp8.device_ptr(&stream);
                    let (b_ptr, _r2) = b_dev_fp8.device_ptr(&stream);
                    let (c_ptr, _r3) = c_dev_fp8.device_ptr_mut(&stream);
                    let (w_ptr, _r4) = workspace_dev.device_ptr(&stream);
                    rustorch_cuda::cublas_lt::matmul_fp8(
                        a_ptr,
                        b_ptr,
                        c_ptr,
                        m,
                        k,
                        n,
                        1.0,
                        0.0,
                        rustorch_cuda::cublas_lt::Fp8Kind::E4M3,
                        rustorch_cuda::cublas_lt::Fp8Output::Bf16,
                        w_ptr,
                        workspace_size,
                        stream.cu_stream() as u64,
                    )
                    .map_err(|e| BenchError(format!("matmul_fp8: {e}")))?;
                }
            }
            stream.synchronize()?;
            let t0 = Instant::now();
            for _ in 0..n_iters {
                unsafe {
                    let (a_ptr, _r1) = a_dev_fp8.device_ptr(&stream);
                    let (b_ptr, _r2) = b_dev_fp8.device_ptr(&stream);
                    let (c_ptr, _r3) = c_dev_fp8.device_ptr_mut(&stream);
                    let (w_ptr, _r4) = workspace_dev.device_ptr(&stream);
                    rustorch_cuda::cublas_lt::matmul_fp8(
                        a_ptr,
                        b_ptr,
                        c_ptr,
                        m,
                        k,
                        n,
                        1.0,
                        0.0,
                        rustorch_cuda::cublas_lt::Fp8Kind::E4M3,
                        rustorch_cuda::cublas_lt::Fp8Output::Bf16,
                        w_ptr,
                        workspace_size,
                        stream.cu_stream() as u64,
                    )
                    .map_err(|e| BenchError(format!("matmul_fp8: {e}")))?;
                }
            }
            stream.synchronize()?;
            Ok(t0.elapsed().as_secs_f64() * 1000.0)
        };
        let fp8_tflops = match fp8_attempt() {
            Ok(wall_ms_fp8) => {
                let ms_per_iter_fp8 = wall_ms_fp8 / n_iters as f64;
                flops_per_iter / (ms_per_iter_fp8 / 1000.0) / 1e12
            },
            Err(e) => {
                if dim == 128 {
                    eprintln!("[cublas_gemm_bench] FP8 path skipped — {e}");
                }
                f64::NAN
            },
        };

        println!(
            "  {:>5} {:>8} {:>10.2} {:>10.2} {:>10.2}",
            dim, n_iters, tf32_tflops, bf16_tflops, fp8_tflops
        );
    }

    println!();
    println!("[cublas_gemm_bench] done");
    Ok(())
}
