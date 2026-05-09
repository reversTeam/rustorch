//! `cusparselt_probe` — test isolé pour identifier la config cuSPARSELt
//! qui passe sur GB10 sans renvoyer code 7. Essaie systématiquement
//! plusieurs combinaisons (compute type, alignment, prune algo) sur des
//! shapes simples puis Qwen-réalistes.

#[cfg(not(feature = "cuda"))]
fn main() {
    println!("[cusparselt_probe] cuda feature is OFF");
}

#[cfg(feature = "cuda")]
#[derive(Debug)]
struct ProbeError(String);

#[cfg(feature = "cuda")]
impl std::fmt::Display for ProbeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

#[cfg(feature = "cuda")]
impl std::error::Error for ProbeError {}

#[cfg(feature = "cuda")]
impl From<cudarc::driver::DriverError> for ProbeError {
    fn from(e: cudarc::driver::DriverError) -> Self {
        Self(format!("driver: {e:?}"))
    }
}

#[cfg(feature = "cuda")]
impl From<rustorch_cuda::error::CudaError> for ProbeError {
    fn from(e: rustorch_cuda::error::CudaError) -> Self {
        Self(format!("cuda: {e:?}"))
    }
}

#[cfg(feature = "cuda")]
fn main() -> Result<(), ProbeError> {
    use cudarc::driver::CudaContext;
    use rustorch_cuda::cusparse_lt::SparseLtSession;

    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let session = SparseLtSession::new(stream.clone())?;
    println!("[cusparselt_probe] session OK");

    // Test grid : shape × alignment × pour identifier le sweet spot
    let shapes: &[(usize, usize, usize, &str)] = &[
        (256, 256, 256, "tiny"),
        (1024, 1024, 1024, "small-square"),
        (2048, 2048, 2048, "med-square"),
        (5120, 2048, 2048, "qwen-qkv-tiny"),
        (5120, 2048, 4096, "qwen-qkv-mid"),
        (5120, 2048, 8192, "qwen-qkv-real"),
        (2048, 2048, 2048, "qwen-attnout-tiny"),
        (2048, 2048, 8192, "qwen-attnout-real"),
        // Larger shapes that fail in qwen_block_bench
        (22016, 2048, 2048, "qwen-ffngu-tiny"),
        (22016, 2048, 4096, "qwen-ffngu-mid"),
        (22016, 2048, 8192, "qwen-ffngu-real"),
        (2048, 11008, 2048, "qwen-ffndn-tiny"),
        (2048, 11008, 4096, "qwen-ffndn-mid"),
        (2048, 11008, 8192, "qwen-ffndn-real"),
    ];

    // Allocate big enough buffers once (max from grid)
    let max_w_elems = shapes.iter().map(|(m, k, _, _)| m * k).max().unwrap();
    let max_act_elems = shapes.iter().map(|(_, k, n, _)| k * n).max().unwrap();
    let max_out_elems = shapes.iter().map(|(m, _, n, _)| m * n).max().unwrap();
    let weight_host = vec![half::bf16::from_f32(0.01); max_w_elems];
    let act_host = vec![half::bf16::from_f32(0.01); max_act_elems];
    let weight_dev = stream.memcpy_stod(&weight_host)?;
    let act_dev = stream.memcpy_stod(&act_host)?;
    let mut out_dev = stream.alloc_zeros::<half::bf16>(max_out_elems)?;

    println!();
    println!("Shape sweep (BF16, alignment=16, Compute_32F, Tile prune, default algo):");
    println!(
        "  {:<22} {:>6} {:>6} {:>6}    Result",
        "name", "m", "k", "n"
    );
    for (m, k, n, name) in shapes {
        use cudarc::driver::{DevicePtr, DevicePtrMut};
        let (w_p, _r) = unsafe { weight_dev.device_ptr(&stream) };
        let mut sp = match unsafe { session.prune_compress_bf16(w_p, *m, *k, *n) } {
            Ok(sp) => sp,
            Err(e) => {
                println!("  {name:<22} {m:>6} {k:>6} {n:>6}    PRUNE/COMPRESS FAIL: {e:?}");
                continue;
            },
        };
        let (b_p, _r) = unsafe { act_dev.device_ptr(&stream) };
        let (c_p, _r2) = unsafe { out_dev.device_ptr_mut(&stream) };
        let res = unsafe { session.matmul_bf16(&mut sp, b_p, c_p, 1.0, 0.0) };
        match res {
            Ok(_) => {
                stream.synchronize()?;
                // Time it
                let t0 = std::time::Instant::now();
                for _ in 0..20 {
                    let res2 = unsafe { session.matmul_bf16(&mut sp, b_p, c_p, 1.0, 0.0) };
                    if res2.is_err() {
                        println!("  {name:<22} {m:>6} {k:>6} {n:>6}    MATMUL DIED: {res2:?}");
                        break;
                    }
                }
                stream.synchronize()?;
                let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0 / 20.0;
                let tflops = 2.0 * (*m * *k * *n) as f64 / 1e12 / (elapsed_ms / 1000.0);
                println!(
                    "  {name:<22} {m:>6} {k:>6} {n:>6}    OK  {elapsed_ms:>6.3} ms  {tflops:>6.1} TFLOPS"
                );
            },
            Err(e) => {
                println!("  {name:<22} {m:>6} {k:>6} {n:>6}    MATMUL FAIL: {e:?}");
            },
        }
    }

    Ok(())
}
