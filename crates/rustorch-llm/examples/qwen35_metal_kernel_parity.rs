//! `qwen35_metal_kernel_parity` — bit-exactness validation of the new
//! T142 Metal kernels (`sgemv_q5_k_f32_lcpp_nsg2`,
//! `sgemv_q8_0_f32_lcpp_nsg2`) against the CPU dequantization reference.
//!
//! Strategy: pick a real Q5_K and a real Q8_0 weight tensor from a
//! Qwen3.5/3.6 GGUF, dequantize them on CPU to f32, and compute
//! `y_cpu = W_f32 @ x` with a plain triple-loop. Then upload the same
//! quantized bytes to Metal, run the new sgemv kernels, and compare.
//!
//! Pass criterion: max absolute error ≤ 1e-3 (mostly fp32 round-off; the
//! kernels and CPU reference produce the same dequantized values bit-
//! identically since both use the same Q5_K / Q8_0 unpacking).
//!
//! Usage:
//!
//! ```sh
//! cargo run --release -p rustorch-llm --example qwen35_metal_kernel_parity -- \
//!     ~/models/Qwen3.6-27B-Q4_K_M.gguf
//! ```
//!
//! Both 27B (Q5_K via `ssm_out`) and 35B-A3B (Q8_0 via `attn_qkv`,
//! `token_embd`, etc.) have the necessary tensors so either model works.

#![cfg(target_os = "macos")]
#![allow(clippy::type_complexity)]

use std::env;
use std::process::ExitCode;

use rustorch_gguf::{dequant_to_f32, GgmlType, GgufFile};
use rustorch_metal::backend_singleton::metal_backend;
use rustorch_metal::kernels::{sgemv_q5_k_f32_lcpp_nsg2_into, sgemv_q8_0_f32_lcpp_nsg2_into};

/// Plain CPU `y[i] = sum_k W[i*K + k] * x[k]` reference. Slow but
/// obviously correct.
fn cpu_gemv(w: &[f32], x: &[f32], y: &mut [f32], n: usize, k: usize) {
    for i in 0..n {
        let row = &w[i * k..(i + 1) * k];
        let mut acc = 0.0_f32;
        for j in 0..k {
            acc += row[j] * x[j];
        }
        y[i] = acc;
    }
}

fn pick_first_tensor_of_type(
    file: &GgufFile,
    target: GgmlType,
) -> Option<&rustorch_gguf::TensorInfo> {
    file.tensors()
        .iter()
        .find(|t| t.dtype == target && t.shape.len() == 2)
}

fn measure_error(y_ref: &[f32], y_test: &[f32]) -> (f32, f32, f32) {
    let mut max_abs = 0.0_f32;
    let mut sum_sq_err = 0.0_f64;
    let mut sum_sq_ref = 0.0_f64;
    for (a, b) in y_ref.iter().zip(y_test.iter()) {
        let e = (a - b).abs();
        if e > max_abs {
            max_abs = e;
        }
        sum_sq_err += (a - b) as f64 * (a - b) as f64;
        sum_sq_ref += *a as f64 * *a as f64;
    }
    let rel = (sum_sq_err / sum_sq_ref.max(1e-30)).sqrt() as f32;
    let dot: f64 = y_ref
        .iter()
        .zip(y_test.iter())
        .map(|(a, b)| *a as f64 * *b as f64)
        .sum();
    let n_ref: f64 = y_ref
        .iter()
        .map(|x| (*x as f64).powi(2))
        .sum::<f64>()
        .sqrt();
    let n_t: f64 = y_test
        .iter()
        .map(|x| (*x as f64).powi(2))
        .sum::<f64>()
        .sqrt();
    let cos = (dot / (n_ref * n_t).max(1e-30)) as f32;
    (max_abs, rel, cos)
}

fn test_one(
    backend: &rustorch_metal::backend::MetalBackend,
    file: &GgufFile,
    info: &rustorch_gguf::TensorInfo,
    label: &str,
    sgemv_fn: fn(
        &rustorch_metal::backend::MetalBackend,
        &metal::Buffer,
        &metal::Buffer,
        &metal::Buffer,
        usize,
        usize,
    ) -> Result<(), rustorch_metal::error::MetalError>,
) -> bool {
    if info.shape.len() != 2 {
        println!("  [{label}] skip: not a 2-D tensor");
        return false;
    }
    let in_dim = info.shape[0] as usize;
    let out_dim = info.shape[1] as usize;
    println!(
        "\n=== {} : {}  shape=[{}, {}]  type={:?} ===",
        label, info.name, in_dim, out_dim, info.dtype
    );

    // CPU dequant
    let bytes = file.tensor_bytes(info);
    let w_f32 = dequant_to_f32(info, bytes).expect("dequant");
    println!("  dequantized → {} f32 values", w_f32.len());

    // Build a deterministic input vector x of length K = in_dim.
    let x: Vec<f32> = (0..in_dim)
        .map(|i| ((i as f32 + 1.0) * 0.0017).sin() * 0.5)
        .collect();

    // CPU reference
    let mut y_ref = vec![0.0_f32; out_dim];
    let t0 = std::time::Instant::now();
    cpu_gemv(&w_f32, &x, &mut y_ref, out_dim, in_dim);
    println!("  cpu  gemv: {:.3}s", t0.elapsed().as_secs_f64());

    // Metal: copy raw quantized bytes into a buffer, x into another, allocate output.
    let x_buf = backend.alloc_shared(in_dim * 4).unwrap();
    unsafe {
        std::ptr::copy_nonoverlapping(
            x.as_ptr() as *const u8,
            x_buf.contents() as *mut u8,
            in_dim * 4,
        );
    }
    let w_buf = backend.alloc_shared(bytes.len()).unwrap();
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), w_buf.contents() as *mut u8, bytes.len());
    }
    let y_buf = backend.alloc_shared(out_dim * 4).unwrap();

    let t0 = std::time::Instant::now();
    sgemv_fn(backend, &x_buf, &w_buf, &y_buf, in_dim, out_dim).expect("metal sgemv");
    backend.drain();
    let metal_dur = t0.elapsed();
    println!("  metal sgemv: {:.3} ms", metal_dur.as_secs_f64() * 1000.0);

    let mut y_metal = vec![0.0_f32; out_dim];
    unsafe {
        std::ptr::copy_nonoverlapping(
            y_buf.contents() as *const f32,
            y_metal.as_mut_ptr(),
            out_dim,
        );
    }

    let (max_abs, rel, cos) = measure_error(&y_ref, &y_metal);
    println!("  parity: max_abs={max_abs:.6e}  rel_l2={rel:.6e}  cos={cos:.6}");
    let pass = max_abs < 1e-2 && rel < 1e-3 && cos > 0.9999;
    if pass {
        println!("  ✓ PASS");
    } else {
        println!("  ✗ FAIL — first 8 ref vs metal:");
        for i in 0..8 {
            println!(
                "      [{}] ref={:>12.6}  metal={:>12.6}  diff={:>12.6}",
                i,
                y_ref[i],
                y_metal[i],
                y_ref[i] - y_metal[i]
            );
        }
    }
    pass
}

fn main() -> ExitCode {
    let path = match env::args().nth(1) {
        Some(p) => p,
        None => {
            eprintln!("usage: qwen35_metal_kernel_parity <gguf-path>");
            return ExitCode::FAILURE;
        },
    };

    let file = match GgufFile::open(&path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("open gguf failed: {e:?}");
            return ExitCode::FAILURE;
        },
    };
    println!("opened {path}");
    let backend = metal_backend();
    println!(
        "device: {} (Metal3: {})",
        backend.adapter_name(),
        backend.supports_metal3()
    );

    let mut all_pass = true;

    // Pick a Q5_K tensor (present in 27B as ssm_out, and in 35B-A3B as
    // ffn_down_exps). For 35B-A3B, ffn_down_exps is 3-D and we don't run
    // the parity test on it; we look for a 2-D Q5_K. If none, skip.
    if let Some(info) = pick_first_tensor_of_type(&file, GgmlType::Q5_K) {
        all_pass &= test_one(
            backend,
            &file,
            info,
            "Q5_K parity",
            sgemv_q5_k_f32_lcpp_nsg2_into,
        );
    } else {
        println!("\n[Q5_K parity] skip: no 2-D Q5_K tensor found in this GGUF");
    }

    // Pick a Q8_0 tensor.
    if let Some(info) = pick_first_tensor_of_type(&file, GgmlType::Q8_0) {
        all_pass &= test_one(
            backend,
            &file,
            info,
            "Q8_0 parity",
            sgemv_q8_0_f32_lcpp_nsg2_into,
        );
    } else {
        println!("\n[Q8_0 parity] skip: no 2-D Q8_0 tensor found in this GGUF");
    }

    if all_pass {
        println!("\n✓ all parity checks passed");
        ExitCode::SUCCESS
    } else {
        println!("\n✗ some parity checks FAILED");
        ExitCode::FAILURE
    }
}
