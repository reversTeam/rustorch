//! `llm_kernels_test` — parité CPU vs CUDA pour les kernels LLM BF16.

#[cfg(not(feature = "cuda"))]
fn main() {
    println!("[llm_kernels_test] cuda feature is OFF");
}

#[cfg(feature = "cuda")]
#[derive(Debug)]
struct TestErr(String);

#[cfg(feature = "cuda")]
impl std::fmt::Display for TestErr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

#[cfg(feature = "cuda")]
impl std::error::Error for TestErr {}

#[cfg(feature = "cuda")]
impl From<cudarc::driver::DriverError> for TestErr {
    fn from(e: cudarc::driver::DriverError) -> Self {
        Self(format!("driver: {e:?}"))
    }
}

#[cfg(feature = "cuda")]
impl From<rustorch_cuda::error::CudaError> for TestErr {
    fn from(e: rustorch_cuda::error::CudaError) -> Self {
        Self(format!("cuda: {e:?}"))
    }
}

#[cfg(feature = "cuda")]
fn main() -> Result<(), TestErr> {
    use cudarc::driver::{CudaContext, DevicePtr, DevicePtrMut};
    use rustorch_cuda::llm_kernels::LlmKernels;

    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let kernels = LlmKernels::new(ctx.clone());

    // ───────── Test RMSNorm BF16 ─────────
    println!("=== rms_norm_bf16 ===");
    let n = 2048usize;
    let batch = 4usize;
    let eps = 1e-5f32;
    let x_host: Vec<half::bf16> = (0..n * batch)
        .map(|i| half::bf16::from_f32(((i % 100) as f32 - 50.0) * 0.01))
        .collect();
    let gamma_host: Vec<half::bf16> = (0..n)
        .map(|i| half::bf16::from_f32(1.0 + (i as f32) * 1e-4))
        .collect();

    // CPU reference
    let mut x_cpu = x_host.clone();
    for b in 0..batch {
        let off = b * n;
        let sum_sq: f32 = x_cpu[off..off + n]
            .iter()
            .map(|v| {
                let f = v.to_f32();
                f * f
            })
            .sum();
        let inv_rms = 1.0 / ((sum_sq / n as f32) + eps).sqrt();
        for i in 0..n {
            let v = x_cpu[off + i].to_f32() * inv_rms * gamma_host[i].to_f32();
            x_cpu[off + i] = half::bf16::from_f32(v);
        }
    }

    // GPU
    let mut x_dev = stream.memcpy_stod(&x_host)?;
    let gamma_dev = stream.memcpy_stod(&gamma_host)?;
    unsafe {
        let (x_p, _r1) = x_dev.device_ptr_mut(&stream);
        let (g_p, _r2) = gamma_dev.device_ptr(&stream);
        kernels.rms_norm_bf16(&stream, x_p, g_p, eps, n as i32, batch as i32)?;
    }
    stream.synchronize()?;
    let x_gpu_bytes = stream.memcpy_dtov(&x_dev)?;
    let x_gpu: &[half::bf16] = bytemuck::cast_slice(&x_gpu_bytes);

    let mut max_diff = 0f32;
    for i in 0..n * batch {
        let d = (x_cpu[i].to_f32() - x_gpu[i].to_f32()).abs();
        if d > max_diff {
            max_diff = d;
        }
    }
    println!("  max abs diff CPU vs GPU = {max_diff:.6}");
    println!("  PASS = {}", max_diff < 0.01);

    // ───────── Test SiLU BF16 ─────────
    println!();
    println!("=== silu_bf16 ===");
    let n_silu = 4096usize;
    let x_host: Vec<half::bf16> = (0..n_silu)
        .map(|i| half::bf16::from_f32(((i as i32 - 2048) as f32) * 0.01))
        .collect();
    let mut x_cpu = x_host.clone();
    for v in x_cpu.iter_mut() {
        let f = v.to_f32();
        *v = half::bf16::from_f32(f / (1.0 + (-f).exp()));
    }
    let mut x_dev = stream.memcpy_stod(&x_host)?;
    unsafe {
        let (x_p, _r1) = x_dev.device_ptr_mut(&stream);
        kernels.silu_bf16(&stream, x_p, n_silu as i32)?;
    }
    stream.synchronize()?;
    let x_gpu_bytes = stream.memcpy_dtov(&x_dev)?;
    let x_gpu: &[half::bf16] = bytemuck::cast_slice(&x_gpu_bytes);
    let mut max_diff = 0f32;
    for i in 0..n_silu {
        let d = (x_cpu[i].to_f32() - x_gpu[i].to_f32()).abs();
        if d > max_diff {
            max_diff = d;
        }
    }
    println!("  max abs diff CPU vs GPU = {max_diff:.6}");
    println!("  PASS = {}", max_diff < 0.01);

    // ───────── Test SwiGLU BF16 ─────────
    println!();
    println!("=== swiglu_bf16 ===");
    let n_sw = 11008usize;
    let gate_host: Vec<half::bf16> = (0..n_sw)
        .map(|i| half::bf16::from_f32(((i % 50) as f32) * 0.05))
        .collect();
    let up_host: Vec<half::bf16> = (0..n_sw)
        .map(|i| half::bf16::from_f32(0.5 + (i % 30) as f32 * 0.01))
        .collect();
    let mut out_cpu = vec![half::bf16::from_f32(0.0); n_sw];
    for i in 0..n_sw {
        let g = gate_host[i].to_f32();
        let u = up_host[i].to_f32();
        let s = g / (1.0 + (-g).exp());
        out_cpu[i] = half::bf16::from_f32(s * u);
    }
    let gate_dev = stream.memcpy_stod(&gate_host)?;
    let up_dev = stream.memcpy_stod(&up_host)?;
    let mut out_dev = stream.alloc_zeros::<half::bf16>(n_sw)?;
    unsafe {
        let (g_p, _r1) = gate_dev.device_ptr(&stream);
        let (u_p, _r2) = up_dev.device_ptr(&stream);
        let (o_p, _r3) = out_dev.device_ptr_mut(&stream);
        kernels.swiglu_bf16(&stream, g_p, u_p, o_p, n_sw as i32)?;
    }
    stream.synchronize()?;
    let out_gpu_bytes = stream.memcpy_dtov(&out_dev)?;
    let out_gpu: &[half::bf16] = bytemuck::cast_slice(&out_gpu_bytes);
    let mut max_diff = 0f32;
    for i in 0..n_sw {
        let d = (out_cpu[i].to_f32() - out_gpu[i].to_f32()).abs();
        if d > max_diff {
            max_diff = d;
        }
    }
    println!("  max abs diff CPU vs GPU = {max_diff:.6}");
    println!("  PASS = {}", max_diff < 0.01);

    // ───────── Test argmax BF16 ─────────
    println!();
    println!("=== argmax_bf16 ===");
    let n_amax = 152064usize; // Qwen vocab size approx
    let target_idx: u32 = 12345;
    let logits_host: Vec<half::bf16> = (0..n_amax)
        .map(|i| {
            if i as u32 == target_idx {
                half::bf16::from_f32(99.0)
            } else {
                half::bf16::from_f32((i % 100) as f32 * 0.01)
            }
        })
        .collect();
    let logits_dev = stream.memcpy_stod(&logits_host)?;
    let mut out_dev = stream.alloc_zeros::<u32>(1)?;
    unsafe {
        let (l_p, _r1) = logits_dev.device_ptr(&stream);
        let (o_p, _r2) = out_dev.device_ptr_mut(&stream);
        kernels.argmax_bf16(&stream, l_p, o_p, n_amax as i32)?;
    }
    stream.synchronize()?;
    let out = stream.memcpy_dtov(&out_dev)?;
    println!("  GPU argmax = {}, expected = {}", out[0], target_idx);
    println!("  PASS = {}", out[0] == target_idx);

    Ok(())
}
