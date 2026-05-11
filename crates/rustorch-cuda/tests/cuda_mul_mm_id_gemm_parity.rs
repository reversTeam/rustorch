//! T246.10 TrackE.2 — Parity tests for `mul_mm_id_gemm_*` Group-GEMM
//! kernels vs the M-iteration `mul_mm_id_*` (A4) reference loop.
//!
//! Each Group-GEMM kernel collapses
//!   `for m in 0..M { mul_mm_id_*(x[m], topk[m], y[m]) }`
//! into a single launch with grid_z = M. The PER-(token, slot, row) BODY
//! is byte-for-byte identical to the M=1 kernel, so we assert BIT-EXACT
//! parity (no ULP slack).
//!
//! Test matrix : Q4_K (warp-shuffle + dp4a), Q5_K, Q6_K, BF16 ;
//! M ∈ {8, 32, 128, 512} ; k_used = 8 ; n_experts = 16 ; N ∈ {128, 4096} ;
//! K = 256 (synthetic) and K = 2048 (Qwen3.6 MoE FFN dim).
//!
//! Run on GB10 (DGX Spark) :
//!   PATH=/usr/local/cuda-13.0/bin:$PATH \
//!     LD_LIBRARY_PATH=/usr/local/cuda-13.0/targets/sbsa-linux/lib:$LD_LIBRARY_PATH \
//!     cargo test --release --features cuda -p rustorch-cuda \
//!       --test cuda_mul_mm_id_gemm_parity -- --ignored --nocapture --test-threads=1

#![cfg(feature = "cuda")]

use cudarc::driver::{CudaContext, DevicePtr, DevicePtrMut};
use rustorch_cuda::llm_kernels::LlmKernels;

fn rng_u8_blob(seed: u64, n: usize) -> Vec<u8> {
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    let mut out = vec![0u8; n];
    for b in out.iter_mut() {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        *b = (state & 0xFF) as u8;
    }
    out
}

fn bf16_vec(n: usize, seed: f32) -> Vec<half::bf16> {
    (0..n)
        .map(|i| half::bf16::from_f32(((i as f32 + 1.0) * 0.013 + seed).sin() * 0.4))
        .collect()
}

fn build_q4k_blob(seed: u64, n_rows: usize, k: usize) -> Vec<u8> {
    assert!(k % 256 == 0);
    let blocks_per_row = k / 256;
    let row_bytes = blocks_per_row * 144;
    let mut w = rng_u8_blob(seed, n_rows * row_bytes);
    for row in 0..n_rows {
        for blk in 0..blocks_per_row {
            let off = row * row_bytes + blk * 144;
            let d = half::f16::from_f32(0.05).to_le_bytes();
            let dmin = half::f16::from_f32(0.025).to_le_bytes();
            w[off] = d[0];
            w[off + 1] = d[1];
            w[off + 2] = dmin[0];
            w[off + 3] = dmin[1];
            for i in 0..12 {
                w[off + 4 + i] &= 0x3F;
            }
        }
    }
    w
}

fn build_q5k_blob(seed: u64, n_rows: usize, k: usize) -> Vec<u8> {
    assert!(k % 256 == 0);
    let blocks_per_row = k / 256;
    let row_bytes = blocks_per_row * 176;
    let mut w = rng_u8_blob(seed, n_rows * row_bytes);
    for row in 0..n_rows {
        for blk in 0..blocks_per_row {
            let off = row * row_bytes + blk * 176;
            let d = half::f16::from_f32(0.05).to_le_bytes();
            let dmin = half::f16::from_f32(0.025).to_le_bytes();
            w[off] = d[0];
            w[off + 1] = d[1];
            w[off + 2] = dmin[0];
            w[off + 3] = dmin[1];
            for i in 0..12 {
                w[off + 4 + i] &= 0x3F;
            }
        }
    }
    w
}

fn build_q6k_blob(seed: u64, n_rows: usize, k: usize) -> Vec<u8> {
    assert!(k % 256 == 0);
    let blocks_per_row = k / 256;
    let row_bytes = blocks_per_row * 210;
    let mut w = rng_u8_blob(seed, n_rows * row_bytes);
    for row in 0..n_rows {
        for blk in 0..blocks_per_row {
            let off = row * row_bytes + blk * 210;
            let d = half::f16::from_f32(0.05).to_le_bytes();
            w[off + 208] = d[0];
            w[off + 209] = d[1];
        }
    }
    w
}

fn build_bf16_blob(seed: u64, n_rows: usize, k: usize) -> Vec<half::bf16> {
    let n = n_rows * k;
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    (0..n)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let f = ((state & 0xFFFF) as i32 - 0x8000) as f32 / 0x8000_0000_u32 as f32 * 0.4;
            half::bf16::from_f32(f)
        })
        .collect()
}

/// Generic harness : build per-token (per-row of M) topk, run reference
/// M-iteration loop against `mul_mm_id_*` and Group-GEMM single launch
/// against `mul_mm_id_gemm_*`, assert bit-exact equality on the
/// [M, k_used, N] output.
#[allow(clippy::too_many_arguments)]
fn run_parity_q(
    label: &str,
    n_experts: usize,
    m_tokens: usize,
    n_rows: usize,
    k_dim: usize,
    k_used: usize,
    expert_blobs: Vec<Vec<u8>>,
    x_dev_init: impl FnOnce(&std::sync::Arc<cudarc::driver::CudaStream>) -> u64,
    x_per_token_bytes: usize, // stride in bytes between consecutive token rows in x
    mut ref_f: impl FnMut(u64, u64, u64, u64, i32, i32, i32),
    mut group_f: impl FnMut(u64, u64, u64, u64, i32, i32, i32, i32),
) {
    let ctx = CudaContext::new(0).expect("ctx");
    let stream = ctx.default_stream();

    // Upload weight blobs (one per expert) and collect device base pointers.
    let mut exp_dev: Vec<cudarc::driver::CudaSlice<u8>> = Vec::new();
    for w in &expert_blobs {
        exp_dev.push(stream.memcpy_stod(w).expect("upload expert"));
    }
    let mut ptrs: Vec<u64> = Vec::with_capacity(n_experts);
    for d in &exp_dev {
        let (p, _g) = d.device_ptr(&stream);
        ptrs.push(p);
    }
    let ptrs_dev = stream.memcpy_stod(&ptrs).expect("upload ptrs");

    // Activations : caller's choice (BF16 vs Q8_1 staging).
    let x_p = x_dev_init(&stream);

    // Per-token top-K : (m * 7919 + s * 23 + 1) % n_experts (deterministic
    // but non-trivial mapping, ensures we visit many distinct experts).
    let topk_all: Vec<i32> = (0..m_tokens)
        .flat_map(|m| {
            (0..k_used).map(move |s| {
                let v = (m.wrapping_mul(7919) + s.wrapping_mul(23) + 1) % n_experts;
                v as i32
            })
        })
        .collect();
    let topk_dev = stream.memcpy_stod(&topk_all).expect("upload topk");

    // Reference output : run M iterations, each calling the M=1 kernel
    // with a sliced topk of just k_used entries for that token.
    let mut y_ref = stream
        .alloc_zeros::<half::bf16>(m_tokens * k_used * n_rows)
        .expect("alloc y_ref");
    let bf16_sz = std::mem::size_of::<half::bf16>();
    {
        let (pp, _g) = ptrs_dev.device_ptr(&stream);
        for tok in 0..m_tokens {
            // topk slice for this token starts at &topk_dev[tok * k_used].
            let (tp_base, _g2) = topk_dev.device_ptr(&stream);
            let tp = tp_base + (tok * k_used * std::mem::size_of::<i32>()) as u64;
            // x slice for this token.
            let x_tok = x_p + (tok * x_per_token_bytes) as u64;
            // y slice for this token : [k_used, N] starts at tok * k_used * N.
            let (y_base, _g4) = y_ref.device_ptr_mut(&stream);
            let y_tok = y_base + (tok * k_used * n_rows * bf16_sz) as u64;
            ref_f(
                pp,
                tp,
                x_tok,
                y_tok,
                n_rows as i32,
                k_dim as i32,
                k_used as i32,
            );
        }
    }

    // Group-GEMM output : single launch.
    let mut y_grp = stream
        .alloc_zeros::<half::bf16>(m_tokens * k_used * n_rows)
        .expect("alloc y_grp");
    {
        let (pp, _g) = ptrs_dev.device_ptr(&stream);
        let (tp, _g2) = topk_dev.device_ptr(&stream);
        let (y_p, _g3) = y_grp.device_ptr_mut(&stream);
        group_f(
            pp,
            tp,
            x_p,
            y_p,
            m_tokens as i32,
            n_rows as i32,
            k_dim as i32,
            k_used as i32,
        );
    }

    let r_ref: Vec<half::bf16> = stream.memcpy_dtov(&y_ref).expect("dtov ref");
    let r_grp: Vec<half::bf16> = stream.memcpy_dtov(&y_grp).expect("dtov grp");

    let mut diffs = 0usize;
    for tok in 0..m_tokens {
        for slot in 0..k_used {
            for i in 0..n_rows {
                let idx = (tok * k_used + slot) * n_rows + i;
                let ar = r_ref[idx];
                let am = r_grp[idx];
                if ar.to_bits() != am.to_bits() {
                    if diffs < 8 {
                        eprintln!(
                            "{label} tok={tok} slot={slot} row={i} \
                             ref=0x{:04x}={} grp=0x{:04x}={} (e_idx={})",
                            ar.to_bits(),
                            ar.to_f32(),
                            am.to_bits(),
                            am.to_f32(),
                            topk_all[tok * k_used + slot],
                        );
                    }
                    diffs += 1;
                }
            }
        }
    }
    assert_eq!(
        diffs, 0,
        "{label} : {diffs} BF16 bit-mismatches (M={m_tokens}, N={n_rows}, K={k_dim}, k_used={k_used})",
    );
    eprintln!("PASS {label} : M={m_tokens} N={n_rows} K={k_dim} k_used={k_used}");
}

fn run_q4k_case(label: &str, m: usize, n_rows: usize, k_dim: usize, k_used: usize) {
    let n_experts = 16usize;
    let blobs: Vec<Vec<u8>> = (0..n_experts)
        .map(|e| build_q4k_blob(0xa2_b3_4d_e5 ^ (e as u64 * 17), n_rows, k_dim))
        .collect();

    let ctx = CudaContext::new(0).expect("ctx");
    let stream = ctx.default_stream();
    let kernels = LlmKernels::new(ctx);
    // Per-token x : [M, K] BF16.
    let mut x_all = Vec::<half::bf16>::with_capacity(m * k_dim);
    for tok in 0..m {
        let v = bf16_vec(k_dim, 0.011 + (tok as f32) * 0.073);
        x_all.extend_from_slice(&v);
    }
    let x_dev = stream.memcpy_stod(&x_all).expect("upload x");

    run_parity_q(
        label,
        n_experts,
        m,
        n_rows,
        k_dim,
        k_used,
        blobs,
        |s| {
            let (p, _g) = x_dev.device_ptr(s);
            p
        },
        k_dim * std::mem::size_of::<half::bf16>(),
        |pp, tp, xp, yp, n, k, k_used| unsafe {
            // Reference : call M=1 kernel with this single-token topk slice.
            kernels
                .mul_mm_id_q4_k_bf16(&stream, pp, tp, xp, yp, n, k, k_used)
                .expect("mul_mm_id_q4k");
        },
        |pp, tp, xp, yp, m, n, k, k_used| unsafe {
            kernels
                .mul_mm_id_gemm_q4_k_bf16(&stream, pp, tp, xp, yp, m, n, k, k_used)
                .expect("mul_mm_id_gemm_q4k");
        },
    );
}

fn run_q4k_dp4a_case(label: &str, m: usize, n_rows: usize, k_dim: usize, k_used: usize) {
    let n_experts = 16usize;
    let blobs: Vec<Vec<u8>> = (0..n_experts)
        .map(|e| build_q4k_blob(0xb1_77_e3_99 ^ (e as u64 * 23), n_rows, k_dim))
        .collect();

    let ctx = CudaContext::new(0).expect("ctx");
    let stream = ctx.default_stream();
    let kernels = LlmKernels::new(ctx);

    // Activations : [M, K] BF16, then quantize per-token to [M, K/32*36] Q8_1.
    let mut x_all = Vec::<half::bf16>::with_capacity(m * k_dim);
    for tok in 0..m {
        let v = bf16_vec(k_dim, 0.017 + (tok as f32) * 0.091);
        x_all.extend_from_slice(&v);
    }
    let x_bf_dev = stream.memcpy_stod(&x_all).expect("upload x_bf");
    let q8_per_tok = (k_dim / 32) * 36;
    let x_q8_dev = stream.alloc_zeros::<u8>(m * q8_per_tok).expect("alloc q8");

    // Per-token Q8_1 quantization. The existing helper takes a single
    // [K] input and writes [K/32*36] output ; iterate.
    unsafe {
        let (xbf, _g) = x_bf_dev.device_ptr(&stream);
        let (xq, _g2) = x_q8_dev.device_ptr(&stream);
        let bf16_sz = std::mem::size_of::<half::bf16>() as u64;
        for tok in 0..m {
            kernels
                .quantize_q8_1_bf16(
                    &stream,
                    xbf + (tok as u64) * (k_dim as u64) * bf16_sz,
                    xq + (tok as u64) * (q8_per_tok as u64),
                    k_dim as i32,
                )
                .expect("q8_1 per-tok");
        }
    }

    run_parity_q(
        label,
        n_experts,
        m,
        n_rows,
        k_dim,
        k_used,
        blobs,
        |s| {
            let (p, _g) = x_q8_dev.device_ptr(s);
            p
        },
        q8_per_tok,
        |pp, tp, xp, yp, n, k, k_used| unsafe {
            kernels
                .mul_mm_id_q4_k_q8_1_dp4a_bf16(&stream, pp, tp, xp, yp, n, k, k_used)
                .expect("mul_mm_id_q4k_dp4a");
        },
        |pp, tp, xp, yp, m, n, k, k_used| unsafe {
            kernels
                .mul_mm_id_gemm_q4_k_q8_1_dp4a_bf16(&stream, pp, tp, xp, yp, m, n, k, k_used)
                .expect("mul_mm_id_gemm_q4k_dp4a");
        },
    );
}

fn run_q5k_case(label: &str, m: usize, n_rows: usize, k_dim: usize, k_used: usize) {
    let n_experts = 16usize;
    let blobs: Vec<Vec<u8>> = (0..n_experts)
        .map(|e| build_q5k_blob(0xc3_44_91_aa ^ (e as u64 * 31), n_rows, k_dim))
        .collect();

    let ctx = CudaContext::new(0).expect("ctx");
    let stream = ctx.default_stream();
    let kernels = LlmKernels::new(ctx);
    let mut x_all = Vec::<half::bf16>::with_capacity(m * k_dim);
    for tok in 0..m {
        let v = bf16_vec(k_dim, 0.023 + (tok as f32) * 0.061);
        x_all.extend_from_slice(&v);
    }
    let x_dev = stream.memcpy_stod(&x_all).expect("upload x");

    run_parity_q(
        label,
        n_experts,
        m,
        n_rows,
        k_dim,
        k_used,
        blobs,
        |s| {
            let (p, _g) = x_dev.device_ptr(s);
            p
        },
        k_dim * std::mem::size_of::<half::bf16>(),
        |pp, tp, xp, yp, n, k, k_used| unsafe {
            kernels
                .mul_mm_id_q5_k_bf16(&stream, pp, tp, xp, yp, n, k, k_used)
                .expect("mul_mm_id_q5k");
        },
        |pp, tp, xp, yp, m, n, k, k_used| unsafe {
            kernels
                .mul_mm_id_gemm_q5_k_bf16(&stream, pp, tp, xp, yp, m, n, k, k_used)
                .expect("mul_mm_id_gemm_q5k");
        },
    );
}

fn run_q6k_case(label: &str, m: usize, n_rows: usize, k_dim: usize, k_used: usize) {
    let n_experts = 16usize;
    let blobs: Vec<Vec<u8>> = (0..n_experts)
        .map(|e| build_q6k_blob(0xd5_61_77_88 ^ (e as u64 * 41), n_rows, k_dim))
        .collect();

    let ctx = CudaContext::new(0).expect("ctx");
    let stream = ctx.default_stream();
    let kernels = LlmKernels::new(ctx);
    let mut x_all = Vec::<half::bf16>::with_capacity(m * k_dim);
    for tok in 0..m {
        let v = bf16_vec(k_dim, 0.029 + (tok as f32) * 0.053);
        x_all.extend_from_slice(&v);
    }
    let x_dev = stream.memcpy_stod(&x_all).expect("upload x");

    run_parity_q(
        label,
        n_experts,
        m,
        n_rows,
        k_dim,
        k_used,
        blobs,
        |s| {
            let (p, _g) = x_dev.device_ptr(s);
            p
        },
        k_dim * std::mem::size_of::<half::bf16>(),
        |pp, tp, xp, yp, n, k, k_used| unsafe {
            kernels
                .mul_mm_id_q6_k_bf16(&stream, pp, tp, xp, yp, n, k, k_used)
                .expect("mul_mm_id_q6k");
        },
        |pp, tp, xp, yp, m, n, k, k_used| unsafe {
            kernels
                .mul_mm_id_gemm_q6_k_bf16(&stream, pp, tp, xp, yp, m, n, k, k_used)
                .expect("mul_mm_id_gemm_q6k");
        },
    );
}

fn run_bf16_case(label: &str, m: usize, n_rows: usize, k_dim: usize, k_used: usize) {
    let n_experts = 16usize;
    let bf16_blobs: Vec<Vec<half::bf16>> = (0..n_experts)
        .map(|e| build_bf16_blob(0xe7_88_99_aa ^ (e as u64 * 47), n_rows, k_dim))
        .collect();

    let ctx = CudaContext::new(0).expect("ctx");
    let stream = ctx.default_stream();
    let kernels = LlmKernels::new(ctx);

    let mut exp_dev: Vec<cudarc::driver::CudaSlice<half::bf16>> = Vec::new();
    for w in &bf16_blobs {
        exp_dev.push(stream.memcpy_stod(w).expect("upload bf16 expert"));
    }
    let mut ptrs: Vec<u64> = Vec::with_capacity(n_experts);
    for d in &exp_dev {
        let (p, _g) = d.device_ptr(&stream);
        ptrs.push(p);
    }
    let ptrs_dev = stream.memcpy_stod(&ptrs).expect("upload ptrs");

    let mut x_all = Vec::<half::bf16>::with_capacity(m * k_dim);
    for tok in 0..m {
        let v = bf16_vec(k_dim, 0.037 + (tok as f32) * 0.041);
        x_all.extend_from_slice(&v);
    }
    let x_dev = stream.memcpy_stod(&x_all).expect("upload x");

    let topk_all: Vec<i32> = (0..m)
        .flat_map(|tok| {
            (0..k_used).map(move |s| {
                let v = (tok.wrapping_mul(7919) + s.wrapping_mul(23) + 1) % n_experts;
                v as i32
            })
        })
        .collect();
    let topk_dev = stream.memcpy_stod(&topk_all).expect("upload topk");

    let bf16_sz = std::mem::size_of::<half::bf16>();
    let mut y_ref = stream
        .alloc_zeros::<half::bf16>(m * k_used * n_rows)
        .expect("alloc y_ref");
    {
        let (pp, _g) = ptrs_dev.device_ptr(&stream);
        let (xp_base, _g3) = x_dev.device_ptr(&stream);
        for tok in 0..m {
            let (tp_base, _g2) = topk_dev.device_ptr(&stream);
            let tp = tp_base + (tok * k_used * std::mem::size_of::<i32>()) as u64;
            let xp = xp_base + (tok * k_dim * bf16_sz) as u64;
            let (y_base, _g4) = y_ref.device_ptr_mut(&stream);
            let yp = y_base + (tok * k_used * n_rows * bf16_sz) as u64;
            unsafe {
                kernels
                    .mul_mm_id_bf16_bf16(
                        &stream,
                        pp,
                        tp,
                        xp,
                        yp,
                        n_rows as i32,
                        k_dim as i32,
                        k_used as i32,
                    )
                    .expect("mul_mm_id_bf16");
            }
        }
    }

    let mut y_grp = stream
        .alloc_zeros::<half::bf16>(m * k_used * n_rows)
        .expect("alloc y_grp");
    {
        let (pp, _g) = ptrs_dev.device_ptr(&stream);
        let (tp, _g2) = topk_dev.device_ptr(&stream);
        let (xp, _g3) = x_dev.device_ptr(&stream);
        let (yp, _g4) = y_grp.device_ptr_mut(&stream);
        unsafe {
            kernels
                .mul_mm_id_gemm_bf16_bf16(
                    &stream,
                    pp,
                    tp,
                    xp,
                    yp,
                    m as i32,
                    n_rows as i32,
                    k_dim as i32,
                    k_used as i32,
                )
                .expect("mul_mm_id_gemm_bf16");
        }
    }

    let r_ref: Vec<half::bf16> = stream.memcpy_dtov(&y_ref).expect("dtov ref");
    let r_grp: Vec<half::bf16> = stream.memcpy_dtov(&y_grp).expect("dtov grp");
    let mut diffs = 0usize;
    for tok in 0..m {
        for slot in 0..k_used {
            for i in 0..n_rows {
                let idx = (tok * k_used + slot) * n_rows + i;
                let ar = r_ref[idx];
                let am = r_grp[idx];
                if ar.to_bits() != am.to_bits() {
                    if diffs < 4 {
                        eprintln!(
                            "{label} tok={tok} slot={slot} row={i} ref=0x{:04x} grp=0x{:04x}",
                            ar.to_bits(),
                            am.to_bits()
                        );
                    }
                    diffs += 1;
                }
            }
        }
    }
    assert_eq!(
        diffs, 0,
        "{label} : {diffs} BF16 bit-mismatches (M={m}, N={n_rows}, K={k_dim}, k_used={k_used})",
    );
    eprintln!("PASS {label} : M={m} N={n_rows} K={k_dim} k_used={k_used}");
}

// ── Q4_K warp-shuffle parity ─────────────────────────────────────────

#[test]
#[ignore = "requires CUDA GPU — run with --ignored on DGX"]
fn mul_mm_id_gemm_q4_k_parity_small() {
    run_q4k_case("q4k M=8 N=128 K=256", 8, 128, 256, 8);
    run_q4k_case("q4k M=32 N=128 K=256", 32, 128, 256, 8);
}

#[test]
#[ignore = "requires CUDA GPU — run with --ignored on DGX"]
fn mul_mm_id_gemm_q4_k_parity_medium() {
    run_q4k_case("q4k M=128 N=1024 K=2048", 128, 1024, 2048, 8);
}

#[test]
#[ignore = "requires CUDA GPU — run with --ignored on DGX"]
fn mul_mm_id_gemm_q4_k_parity_large() {
    // Approximate Qwen3.6 MoE FFN shape : expert_f=18944, K=2048.
    // Use N=4096 to keep test cost reasonable while still hitting multi-block grids.
    run_q4k_case("q4k M=512 N=4096 K=2048", 512, 4096, 2048, 8);
}

// ── Q4_K dp4a parity ─────────────────────────────────────────────────

#[test]
#[ignore = "requires CUDA GPU — run with --ignored on DGX"]
fn mul_mm_id_gemm_q4_k_dp4a_parity_small() {
    run_q4k_dp4a_case("q4k_dp4a M=8 N=128 K=256", 8, 128, 256, 8);
    run_q4k_dp4a_case("q4k_dp4a M=32 N=128 K=256", 32, 128, 256, 8);
}

#[test]
#[ignore = "requires CUDA GPU — run with --ignored on DGX"]
fn mul_mm_id_gemm_q4_k_dp4a_parity_large() {
    run_q4k_dp4a_case("q4k_dp4a M=128 N=4096 K=2048", 128, 4096, 2048, 8);
}

// ── Q5_K / Q6_K / BF16 parity ────────────────────────────────────────

#[test]
#[ignore = "requires CUDA GPU — run with --ignored on DGX"]
fn mul_mm_id_gemm_q5_k_parity() {
    run_q5k_case("q5k M=32 N=1024 K=256", 32, 1024, 256, 8);
    run_q5k_case("q5k M=128 N=1024 K=2048", 128, 1024, 2048, 8);
}

#[test]
#[ignore = "requires CUDA GPU — run with --ignored on DGX"]
fn mul_mm_id_gemm_q6_k_parity() {
    run_q6k_case("q6k M=32 N=1024 K=256", 32, 1024, 256, 8);
    run_q6k_case("q6k M=128 N=1024 K=2048", 128, 1024, 2048, 8);
}

#[test]
#[ignore = "requires CUDA GPU — run with --ignored on DGX"]
fn mul_mm_id_gemm_bf16_parity() {
    run_bf16_case("bf16 M=32 N=1024 K=256", 32, 1024, 256, 8);
    run_bf16_case("bf16 M=128 N=1024 K=2048", 128, 1024, 2048, 8);
}
