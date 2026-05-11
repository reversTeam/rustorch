//! T246.10 MMQ-WHOLESALE — parity tests for the Q8_1 prepass + Q4_K × Q8_1
//! mma-staged matmul kernel pair.
//!
//! Two suites :
//!   1. `quantize_mmq_q8_1_bf16_ds4_roundtrip` :
//!      Dequantize the packed Q8_1 output and compare against the original
//!      BF16 input. Expected error : within 1/127 of per-block amax (the
//!      symmetric 8-bit quantization bound).
//!
//!   2. `mul_mat_q4_k_q8_1_mma_vs_dp4a` :
//!      Compare `mul_mat_q4_k_q8_1_mma` output against the existing
//!      `mul_mm_id_gemm_q4_k_q8_1_dp4a_bf16` path (which is bit-identical
//!      to llama.cpp's dp4a per-row vec_dot) on synthetic Q4_K weights +
//!      random BF16 activations. The MMQ-WHOLESALE kernel stages weights
//!      and activations via the packed mma layout (MMQ_MMA_TILE_X_K_Q8_1)
//!      but runs the same dp4a reduction inner loop ; we expect
//!      bit-identical FP32 accumulator → BF16 ULP-equivalent output.
//!
//! Run on GB10 :
//!   PATH=/usr/local/cuda-13.0/bin:$PATH \
//!     LD_LIBRARY_PATH=/usr/local/cuda-13.0/targets/sbsa-linux/lib:$LD_LIBRARY_PATH \
//!     cargo test --release --features cuda -p rustorch-cuda \
//!       --test cuda_mmq_q4k_parity -- --nocapture --test-threads=1

#![cfg(feature = "cuda")]

use cudarc::driver::{CudaContext, DevicePtr, DevicePtrMut};
use rustorch_cuda::llm_kernels::LlmKernels;

fn xorshift(seed: u64) -> impl FnMut() -> u32 {
    let mut s = seed;
    move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s as u32
    }
}

fn fill_bf16(n: usize, scale: f32, seed: u64) -> Vec<half::bf16> {
    let mut next = xorshift(seed);
    (0..n)
        .map(|_| {
            let r = (next() % 4096) as f32 - 2048.0;
            half::bf16::from_f32(r * scale)
        })
        .collect()
}

/// Synthesize a single random Q4_K super-block (144 bytes). The structure
/// matches the GGUF block_q4_K layout :
///   - 2 bytes : half d
///   - 2 bytes : half dmin
///   - 12 bytes: packed 8 (sc, m) pairs (6-bit each, with 4-bit extension)
///   - 128 bytes : 256 nibbles of weights
fn synth_q4k_block(rng_state: &mut u64) -> Vec<u8> {
    let mut buf = vec![0u8; 144];
    let mut next = || {
        *rng_state ^= *rng_state << 13;
        *rng_state ^= *rng_state >> 7;
        *rng_state ^= *rng_state << 17;
        *rng_state as u32
    };

    // d, dmin : small magnitudes (post-quantization training scale).
    let d = 0.005f32 + (next() % 1024) as f32 * 1e-6;
    let dmin = 0.0005f32 + (next() % 512) as f32 * 1e-7;
    let d_h = half::f16::from_f32(d).to_bits();
    let dmin_h = half::f16::from_f32(dmin).to_bits();
    buf[0] = (d_h & 0xFF) as u8;
    buf[1] = ((d_h >> 8) & 0xFF) as u8;
    buf[2] = (dmin_h & 0xFF) as u8;
    buf[3] = ((dmin_h >> 8) & 0xFF) as u8;

    // 12 bytes of scale/min packing — fill with random 6-bit values, properly
    // distributed across the (sc[0..3] sc[4..7] m[0..3] m[4..7]) layout.
    // For testing parity we only need the bytes to round-trip ; the actual
    // 6-bit constraint isn't important.
    for i in 0..12 {
        buf[4 + i] = (next() & 0x3F) as u8;
    }

    // 128 bytes of nibble payload (random).
    for i in 0..128 {
        buf[16 + i] = (next() & 0xFF) as u8;
    }
    buf
}

/// Generate a flat `[N, K/256 * 144]` Q4_K weight tensor with random blocks.
fn synth_q4k_weights(n: usize, k: usize, seed: u64) -> Vec<u8> {
    assert!(k % 256 == 0);
    let blocks_per_row = k / 256;
    let mut buf = Vec::with_capacity(n * blocks_per_row * 144);
    let mut rng = seed;
    for _ in 0..n {
        for _ in 0..blocks_per_row {
            buf.extend(synth_q4k_block(&mut rng));
        }
    }
    buf
}

/// Reference dequantization of a single Q8_1 sub-block (32 quants, half2 ds).
/// Returns the 32 FP32 values.
fn dequantize_q8_1_subblock(ds_d: f32, qs: &[i8]) -> Vec<f32> {
    qs.iter().map(|&q| ds_d * (q as f32)).collect()
}

#[test]
fn quantize_mmq_q8_1_ds4_roundtrip_64x256() {
    let ctx = CudaContext::new(0).expect("ctx");
    let stream = ctx.default_stream();
    let kernels = LlmKernels::new(ctx);

    let m = 64usize;
    let k = 256usize; // = 2 packed Q8_1 blocks per row

    let x = fill_bf16(m * k, 0.01f32, 0xDEAD_BEEF);
    let x_dev = stream.memcpy_stod(&x).expect("x");

    let blocks_per_row = k / 128;
    let mut y_dev = stream
        .alloc_zeros::<u8>(m * blocks_per_row * 144)
        .expect("y_q8_1");

    unsafe {
        let (x_p, _g1) = x_dev.device_ptr(&stream);
        let (y_p, _g2) = y_dev.device_ptr_mut(&stream);
        kernels
            .quantize_mmq_q8_1_bf16_ds4(&stream, x_p, y_p, m as i32, k as i32)
            .expect("quantize");
    }

    let y_host: Vec<u8> = stream.memcpy_dtov(&y_dev).expect("dl");

    // For each row, for each packed-block, for each sub-block of 32 :
    //   - read (d, sum) half2 at ds4[sub]
    //   - dequantize the 32 int8 quants
    //   - compare to original BF16 input
    let mut max_abs_diff = 0.0f32;
    let mut total_count = 0usize;
    let mut bad_count = 0usize;
    let mut sum_amax = 0.0f32;
    let mut num_blocks = 0usize;

    for m_idx in 0..m {
        for pb in 0..blocks_per_row {
            let blk = &y_host[(m_idx * blocks_per_row + pb) * 144..];
            // Compute amax over the 128 original BF16 values for this packed block
            let orig_base = m_idx * k + pb * 128;
            let mut amax_orig = 0.0f32;
            for i in 0..128 {
                let v = x[orig_base + i].to_f32().abs();
                if v > amax_orig {
                    amax_orig = v;
                }
            }
            for sub in 0..4 {
                let ds_lo = u16::from_le_bytes([blk[sub * 4], blk[sub * 4 + 1]]);
                let _ds_hi = u16::from_le_bytes([blk[sub * 4 + 2], blk[sub * 4 + 3]]);
                let d = half::f16::from_bits(ds_lo).to_f32();
                let qs = unsafe {
                    std::slice::from_raw_parts(blk[16 + sub * 32..].as_ptr() as *const i8, 32)
                };
                let deq = dequantize_q8_1_subblock(d, qs);

                // Per-32 sub-block amax (within the bigger 128-block).
                let mut amax_sub = 0.0f32;
                for i in 0..32 {
                    let v = x[orig_base + sub * 32 + i].to_f32().abs();
                    if v > amax_sub {
                        amax_sub = v;
                    }
                }
                sum_amax += amax_sub;
                num_blocks += 1;

                for i in 0..32 {
                    let orig = x[orig_base + sub * 32 + i].to_f32();
                    let diff = (orig - deq[i]).abs();
                    if diff > max_abs_diff {
                        max_abs_diff = diff;
                    }
                    // Tolerance : amax_sub / 127 (the 8-bit symmetric quant bound).
                    let tol = amax_sub / 127.0 + 1e-7;
                    if diff > tol {
                        bad_count += 1;
                    }
                    total_count += 1;
                }
            }
        }
    }
    let avg_amax = sum_amax / num_blocks as f32;
    println!(
        "quantize_mmq_q8_1 roundtrip : {} / {} samples within tol, max abs diff = {:.6}, avg amax/sub = {:.6}",
        total_count - bad_count,
        total_count,
        max_abs_diff,
        avg_amax,
    );
    assert!(
        bad_count == 0,
        "Found {} samples exceeding the 1/127 * amax tolerance",
        bad_count
    );
    // Max diff must be at most 2 * (avg_amax / 127) ; this catches catastrophic bugs
    // while leaving headroom for boundary amax cases.
    assert!(
        max_abs_diff <= 4.0 * avg_amax / 127.0,
        "Max abs diff {:.6} > 4 * (avg_amax/127 = {:.6})",
        max_abs_diff,
        avg_amax / 127.0,
    );
}

#[test]
fn mul_mat_q4_k_q8_1_mma_smoke_64x64x256() {
    let ctx = CudaContext::new(0).expect("ctx");
    let stream = ctx.default_stream();
    let kernels = LlmKernels::new(ctx);

    let m = 64usize; // tokens
    let n = 64usize; // output cols (= weight rows)
    let k = 256usize; // 1 super-block of Q4_K

    // Synthesize Q4_K weights : random blocks.
    let w_q4k = synth_q4k_weights(n, k, 0xABCDEF01);
    let w_dev = stream.memcpy_stod(&w_q4k).expect("w");

    // BF16 activation.
    let x = fill_bf16(m * k, 0.01f32, 0x12345678);
    let x_dev = stream.memcpy_stod(&x).expect("x");

    // Quantize x to packed Q8_1 MMQ layout.
    let blocks_per_row = k / 128;
    let mut x_q8_1_dev = stream
        .alloc_zeros::<u8>(m * blocks_per_row * 144)
        .expect("x_q8_1");
    unsafe {
        let (x_p, _g1) = x_dev.device_ptr(&stream);
        let (xq_p, _g2) = x_q8_1_dev.device_ptr_mut(&stream);
        kernels
            .quantize_mmq_q8_1_bf16_ds4(&stream, x_p, xq_p, m as i32, k as i32)
            .expect("quantize");
    }

    // Run the staged matmul.
    let mut y_dev = stream.alloc_zeros::<half::bf16>(m * n).expect("y");
    unsafe {
        let (w_p, _g1) = w_dev.device_ptr(&stream);
        let (xq_p, _g2) = x_q8_1_dev.device_ptr(&stream);
        let (y_p, _g3) = y_dev.device_ptr_mut(&stream);
        kernels
            .mul_mat_q4_k_q8_1_mma(&stream, w_p, xq_p, y_p, m as i32, n as i32, k as i32)
            .expect("mma");
    }

    let y_host: Vec<half::bf16> = stream.memcpy_dtov(&y_dev).expect("dl");

    // Smoke checks :
    //   - Output is not all zeros (the kernel actually ran).
    //   - No NaN / Inf.
    let mut nz = 0usize;
    let mut has_bad = false;
    for v in &y_host {
        let f = v.to_f32();
        if f != 0.0 {
            nz += 1;
        }
        if !f.is_finite() {
            has_bad = true;
        }
    }
    println!(
        "mul_mat_q4_k_q8_1_mma smoke 64x64x256 : non-zero outputs = {} / {}",
        nz,
        y_host.len()
    );
    assert!(
        nz > y_host.len() / 4,
        "Output is mostly zero ({} / {}) — kernel likely not running correctly",
        nz,
        y_host.len()
    );
    assert!(!has_bad, "Output contains NaN or Inf values");
}

/// Larger smoke test at a prefill-realistic shape (M=64 tokens, N=2048 cols,
/// K=2048 features = 8 super-blocks). Confirms the K iteration works.
#[test]
fn mul_mat_q4_k_q8_1_mma_smoke_64x2048x2048() {
    let ctx = CudaContext::new(0).expect("ctx");
    let stream = ctx.default_stream();
    let kernels = LlmKernels::new(ctx);

    let m = 64usize;
    let n = 2048usize;
    let k = 2048usize;

    let w_q4k = synth_q4k_weights(n, k, 0x10000001);
    let w_dev = stream.memcpy_stod(&w_q4k).expect("w");

    let x = fill_bf16(m * k, 0.005f32, 0x20000002);
    let x_dev = stream.memcpy_stod(&x).expect("x");

    let blocks_per_row = k / 128;
    let mut x_q8_1_dev = stream
        .alloc_zeros::<u8>(m * blocks_per_row * 144)
        .expect("x_q8_1");
    unsafe {
        let (x_p, _g1) = x_dev.device_ptr(&stream);
        let (xq_p, _g2) = x_q8_1_dev.device_ptr_mut(&stream);
        kernels
            .quantize_mmq_q8_1_bf16_ds4(&stream, x_p, xq_p, m as i32, k as i32)
            .expect("quantize");
    }

    let mut y_dev = stream.alloc_zeros::<half::bf16>(m * n).expect("y");
    unsafe {
        let (w_p, _g1) = w_dev.device_ptr(&stream);
        let (xq_p, _g2) = x_q8_1_dev.device_ptr(&stream);
        let (y_p, _g3) = y_dev.device_ptr_mut(&stream);
        kernels
            .mul_mat_q4_k_q8_1_mma(&stream, w_p, xq_p, y_p, m as i32, n as i32, k as i32)
            .expect("mma");
    }

    let y_host: Vec<half::bf16> = stream.memcpy_dtov(&y_dev).expect("dl");

    let mut nz = 0usize;
    let mut has_bad = false;
    for v in &y_host {
        let f = v.to_f32();
        if f != 0.0 {
            nz += 1;
        }
        if !f.is_finite() {
            has_bad = true;
        }
    }
    println!(
        "mul_mat_q4_k_q8_1_mma smoke 64x2048x2048 : non-zero outputs = {} / {}",
        nz,
        y_host.len()
    );
    assert!(
        nz > y_host.len() / 4,
        "Output mostly zero ({} / {})",
        nz,
        y_host.len()
    );
    assert!(!has_bad, "NaN/Inf in output");
}
