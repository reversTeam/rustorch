//! T246.8 A4 — Parity tests for `mul_mm_id_*` mega-kernels vs the
//! K-iteration `sgemv_*_indexed` reference loop.
//!
//! The mega-kernel collapses `for slot in 0..k_used { sgemv_*_indexed(slot) }`
//! into a single launch with a 2-D grid where `blockIdx.y` carries the
//! slot index. The PER-SLOT BODY is byte-for-byte identical to the
//! existing v3 indexed kernel — A3 demonstrated that any reduction-order
//! change produces token-level drift even when synthetic parity holds.
//! These tests assert BIT-EXACT equality.
//!
//! Test matrix : Q4_K (v3 + dp4a), Q5_K, Q6_K, BF16 ; K_used ∈ {1, 8} ;
//! N ∈ {128, 1024, 18944 (Qwen3.6 MoE expert_f)}.
//!
//! Run on GB10 (DGX Spark) :
//!   PATH=/usr/local/cuda-13.0/bin:$PATH \
//!     LD_LIBRARY_PATH=/usr/local/cuda-13.0/targets/sbsa-linux/lib:$LD_LIBRARY_PATH \
//!     cargo test --release --features cuda -p rustorch-cuda \
//!       --test cuda_mul_mm_id_q4_k_parity -- --ignored --nocapture --test-threads=1

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

/// Build a Q4_K row-major weights blob `[n_rows, K]` with proper d/dmin/scales.
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

/// Build a Q5_K row-major weights blob.
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

/// Build a Q6_K row-major weights blob.
fn build_q6k_blob(seed: u64, n_rows: usize, k: usize) -> Vec<u8> {
    assert!(k % 256 == 0);
    let blocks_per_row = k / 256;
    let row_bytes = blocks_per_row * 210;
    let mut w = rng_u8_blob(seed, n_rows * row_bytes);
    // Set d at offset 208 ; signed-char scales at 192..208 are random fine.
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

/// Build a BF16 row-major weights blob `[n_rows, K]`.
fn build_bf16_blob(seed: u64, n_rows: usize, k: usize) -> Vec<half::bf16> {
    let n = n_rows * k;
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    (0..n)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            // Map to a small-magnitude BF16 to keep accumulation well-behaved.
            let f = ((state & 0xFFFF) as i32 - 0x8000) as f32 / 0x8000_0000_u32 as f32 * 0.4;
            half::bf16::from_f32(f)
        })
        .collect()
}

/// Generic multi-expert harness : upload n_experts independent weight
/// blobs (size `expert_bytes`), then run reference per-slot kernel into
/// y_ref, mega-kernel into y_mega, and assert bit-exact equality.
fn run_parity<RefF, MegaF>(
    label: &str,
    n_experts: usize,
    n_rows: usize,
    k_dim: usize,
    k_used: usize,
    expert_blobs: Vec<Vec<u8>>,
    x_dev_init: impl FnOnce(&std::sync::Arc<cudarc::driver::CudaStream>) -> u64,
    mut ref_f: RefF,
    mut mega_f: MegaF,
) where
    RefF: FnMut(
        u64, // expert_ptrs (device)
        u64, // topk_indices (device)
        i32, // slot
        u64, // x device
        u64, // y device (single row, [N])
        i32, // n
        i32, // k
    ),
    MegaF: FnMut(
        u64, // expert_ptrs
        u64, // topk_indices
        u64, // x
        u64, // y device (k_used rows, [k_used, N])
        i32, // n
        i32, // k
        i32, // k_used
    ),
{
    use cudarc::driver::DevicePtr;
    let ctx = CudaContext::new(0).expect("ctx");
    let stream = ctx.default_stream();

    // Upload weight blobs and collect device base pointers.
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

    // Activation x_dev managed by caller (different layouts for q8_1 vs bf16).
    let x_p = x_dev_init(&stream);

    // Top-K indices : pick a non-trivial mapping (slot s → (s * 3 + 1) % n_experts).
    let topk: Vec<i32> = (0..k_used)
        .map(|s| ((s * 3 + 1) % n_experts) as i32)
        .collect();
    let topk_dev = stream.memcpy_stod(&topk).expect("upload topk");

    // Reference output : run K reference launches.
    let mut y_ref = stream
        .alloc_zeros::<half::bf16>(k_used * n_rows)
        .expect("alloc y_ref");
    {
        let (pp, _g) = ptrs_dev.device_ptr(&stream);
        let (tp, _g2) = topk_dev.device_ptr(&stream);
        for slot in 0..k_used {
            // Pass a y-pointer offset to slot * n_rows.
            let (y_base, _g4) = y_ref.device_ptr_mut(&stream);
            let y_p = y_base + (slot * n_rows * std::mem::size_of::<half::bf16>()) as u64;
            ref_f(pp, tp, slot as i32, x_p, y_p, n_rows as i32, k_dim as i32);
        }
    }

    // Mega output : single launch.
    let mut y_mega = stream
        .alloc_zeros::<half::bf16>(k_used * n_rows)
        .expect("alloc y_mega");
    {
        let (pp, _g) = ptrs_dev.device_ptr(&stream);
        let (tp, _g2) = topk_dev.device_ptr(&stream);
        let (y_p, _g3) = y_mega.device_ptr_mut(&stream);
        mega_f(pp, tp, x_p, y_p, n_rows as i32, k_dim as i32, k_used as i32);
    }

    let r_ref: Vec<half::bf16> = stream.memcpy_dtov(&y_ref).expect("dtov ref");
    let r_mega: Vec<half::bf16> = stream.memcpy_dtov(&y_mega).expect("dtov mega");

    let mut diffs = 0usize;
    for slot in 0..k_used {
        for i in 0..n_rows {
            let idx = slot * n_rows + i;
            let ar = r_ref[idx];
            let am = r_mega[idx];
            if ar.to_bits() != am.to_bits() {
                if diffs < 8 {
                    eprintln!(
                        "{label} slot={slot} row={i} ref=0x{:04x}={} mega=0x{:04x}={} (e_idx={})",
                        ar.to_bits(),
                        ar.to_f32(),
                        am.to_bits(),
                        am.to_f32(),
                        topk[slot],
                    );
                }
                diffs += 1;
            }
        }
    }
    assert_eq!(
        diffs, 0,
        "{label} : {diffs} BF16-bit-mismatches (n_rows={n_rows} k_dim={k_dim} k_used={k_used})",
    );
}

fn run_q4k_v3_case(label: &str, n_rows: usize, k_dim: usize, k_used: usize) {
    let n_experts = (k_used + 2).max(4);
    let row_bytes = (k_dim / 256) * 144;
    let blobs: Vec<Vec<u8>> = (0..n_experts)
        .map(|e| build_q4k_blob(0xa2_b3_4d_e5 ^ (e as u64 * 17), n_rows, k_dim))
        .collect();
    let _ = row_bytes;

    let ctx = CudaContext::new(0).expect("ctx");
    let stream = ctx.default_stream();
    let kernels = LlmKernels::new(ctx);
    let x_bf = bf16_vec(k_dim, 0.011);
    let x_dev = stream.memcpy_stod(&x_bf).expect("upload x");

    run_parity(
        label,
        n_experts,
        n_rows,
        k_dim,
        k_used,
        blobs,
        |s| {
            let (p, _g) = x_dev.device_ptr(s);
            p
        },
        |pp, tp, slot, xp, yp, n, k| unsafe {
            kernels
                .sgemv_q4k_bf16_v3_indexed(&stream, pp, tp, slot, xp, yp, n, k)
                .expect("v3_indexed");
        },
        |pp, tp, xp, yp, n, k, k_used| unsafe {
            kernels
                .mul_mm_id_q4_k_bf16(&stream, pp, tp, xp, yp, n, k, k_used)
                .expect("mul_mm_id_q4k");
        },
    );
}

fn run_q4k_dp4a_case(label: &str, n_rows: usize, k_dim: usize, k_used: usize) {
    let n_experts = (k_used + 2).max(4);
    let blobs: Vec<Vec<u8>> = (0..n_experts)
        .map(|e| build_q4k_blob(0xb1_77_e3_99 ^ (e as u64 * 23), n_rows, k_dim))
        .collect();

    let ctx = CudaContext::new(0).expect("ctx");
    let stream = ctx.default_stream();
    let kernels = LlmKernels::new(ctx);
    let x_bf = bf16_vec(k_dim, 0.017);
    let x_dev = stream.memcpy_stod(&x_bf).expect("upload x");
    // Quantize x to q8_1 staging.
    let x_q8_dev = stream
        .alloc_zeros::<u8>((k_dim / 32) * 36)
        .expect("alloc q8");
    unsafe {
        let (xp, _g) = x_dev.device_ptr(&stream);
        let (xq, _g2) = x_q8_dev.device_ptr(&stream);
        kernels
            .quantize_q8_1_bf16(&stream, xp, xq, k_dim as i32)
            .expect("q8_1");
    }

    run_parity(
        label,
        n_experts,
        n_rows,
        k_dim,
        k_used,
        blobs,
        |s| {
            let (p, _g) = x_q8_dev.device_ptr(s);
            p
        },
        |pp, tp, slot, xp, yp, n, k| unsafe {
            kernels
                .sgemv_q4k_q8_1_dp4a_bf16_indexed(&stream, pp, tp, slot, xp, yp, n, k)
                .expect("q4k_dp4a_indexed");
        },
        |pp, tp, xp, yp, n, k, k_used| unsafe {
            kernels
                .mul_mm_id_q4_k_q8_1_dp4a_bf16(&stream, pp, tp, xp, yp, n, k, k_used)
                .expect("mul_mm_id_q4k_dp4a");
        },
    );
}

fn run_q5k_case(label: &str, n_rows: usize, k_dim: usize, k_used: usize) {
    let n_experts = (k_used + 2).max(4);
    let blobs: Vec<Vec<u8>> = (0..n_experts)
        .map(|e| build_q5k_blob(0xc3_44_91_aa ^ (e as u64 * 31), n_rows, k_dim))
        .collect();

    let ctx = CudaContext::new(0).expect("ctx");
    let stream = ctx.default_stream();
    let kernels = LlmKernels::new(ctx);
    let x_bf = bf16_vec(k_dim, 0.023);
    let x_dev = stream.memcpy_stod(&x_bf).expect("upload x");

    run_parity(
        label,
        n_experts,
        n_rows,
        k_dim,
        k_used,
        blobs,
        |s| {
            let (p, _g) = x_dev.device_ptr(s);
            p
        },
        |pp, tp, slot, xp, yp, n, k| unsafe {
            kernels
                .sgemv_q5k_bf16_v3_indexed(&stream, pp, tp, slot, xp, yp, n, k)
                .expect("q5k_v3_indexed");
        },
        |pp, tp, xp, yp, n, k, k_used| unsafe {
            kernels
                .mul_mm_id_q5_k_bf16(&stream, pp, tp, xp, yp, n, k, k_used)
                .expect("mul_mm_id_q5k");
        },
    );
}

fn run_q6k_case(label: &str, n_rows: usize, k_dim: usize, k_used: usize) {
    let n_experts = (k_used + 2).max(4);
    let blobs: Vec<Vec<u8>> = (0..n_experts)
        .map(|e| build_q6k_blob(0xd5_61_77_88 ^ (e as u64 * 41), n_rows, k_dim))
        .collect();

    let ctx = CudaContext::new(0).expect("ctx");
    let stream = ctx.default_stream();
    let kernels = LlmKernels::new(ctx);
    let x_bf = bf16_vec(k_dim, 0.029);
    let x_dev = stream.memcpy_stod(&x_bf).expect("upload x");

    run_parity(
        label,
        n_experts,
        n_rows,
        k_dim,
        k_used,
        blobs,
        |s| {
            let (p, _g) = x_dev.device_ptr(s);
            p
        },
        |pp, tp, slot, xp, yp, n, k| unsafe {
            kernels
                .sgemv_q6k_bf16_v3_indexed(&stream, pp, tp, slot, xp, yp, n, k)
                .expect("q6k_v3_indexed");
        },
        |pp, tp, xp, yp, n, k, k_used| unsafe {
            kernels
                .mul_mm_id_q6_k_bf16(&stream, pp, tp, xp, yp, n, k, k_used)
                .expect("mul_mm_id_q6k");
        },
    );
}

fn run_bf16_case(label: &str, n_rows: usize, k_dim: usize, k_used: usize) {
    let n_experts = (k_used + 2).max(4);
    let bf16_blobs: Vec<Vec<half::bf16>> = (0..n_experts)
        .map(|e| build_bf16_blob(0xe7_88_99_aa ^ (e as u64 * 47), n_rows, k_dim))
        .collect();

    let ctx = CudaContext::new(0).expect("ctx");
    let stream = ctx.default_stream();
    let kernels = LlmKernels::new(ctx);

    // Upload BF16 expert blobs separately (different from u8 path).
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

    let x_bf = bf16_vec(k_dim, 0.037);
    let x_dev = stream.memcpy_stod(&x_bf).expect("upload x");

    let topk: Vec<i32> = (0..k_used)
        .map(|s| ((s * 3 + 1) % n_experts) as i32)
        .collect();
    let topk_dev = stream.memcpy_stod(&topk).expect("upload topk");

    let mut y_ref = stream
        .alloc_zeros::<half::bf16>(k_used * n_rows)
        .expect("alloc y_ref");
    {
        let (pp, _g) = ptrs_dev.device_ptr(&stream);
        let (tp, _g2) = topk_dev.device_ptr(&stream);
        let (xp, _g3) = x_dev.device_ptr(&stream);
        for slot in 0..k_used {
            let (y_base, _g4) = y_ref.device_ptr_mut(&stream);
            let y_p = y_base + (slot * n_rows * std::mem::size_of::<half::bf16>()) as u64;
            unsafe {
                kernels
                    .sgemv_bf16_bf16_indexed(
                        &stream,
                        pp,
                        tp,
                        slot as i32,
                        xp,
                        y_p,
                        n_rows as i32,
                        k_dim as i32,
                    )
                    .expect("bf16_indexed");
            }
        }
    }

    let mut y_mega = stream
        .alloc_zeros::<half::bf16>(k_used * n_rows)
        .expect("alloc y_mega");
    {
        let (pp, _g) = ptrs_dev.device_ptr(&stream);
        let (tp, _g2) = topk_dev.device_ptr(&stream);
        let (xp, _g3) = x_dev.device_ptr(&stream);
        let (y_p, _g4) = y_mega.device_ptr_mut(&stream);
        unsafe {
            kernels
                .mul_mm_id_bf16_bf16(
                    &stream,
                    pp,
                    tp,
                    xp,
                    y_p,
                    n_rows as i32,
                    k_dim as i32,
                    k_used as i32,
                )
                .expect("mul_mm_id_bf16");
        }
    }

    let r_ref: Vec<half::bf16> = stream.memcpy_dtov(&y_ref).expect("dtov ref");
    let r_mega: Vec<half::bf16> = stream.memcpy_dtov(&y_mega).expect("dtov mega");
    let mut diffs = 0usize;
    for slot in 0..k_used {
        for i in 0..n_rows {
            let idx = slot * n_rows + i;
            let ar = r_ref[idx];
            let am = r_mega[idx];
            if ar.to_bits() != am.to_bits() {
                if diffs < 4 {
                    eprintln!(
                        "{label} slot={slot} row={i} ref=0x{:04x} mega=0x{:04x}",
                        ar.to_bits(),
                        am.to_bits()
                    );
                }
                diffs += 1;
            }
        }
    }
    assert_eq!(
        diffs, 0,
        "{label} : {diffs} BF16 bit-mismatches (n_rows={n_rows}, K={k_dim}, k_used={k_used})",
    );
}

#[test]
#[ignore = "requires CUDA GPU — run with --ignored on DGX"]
fn mul_mm_id_q4_k_bf16_v3_parity_k1() {
    run_q4k_v3_case("q4k_v3 K=1 N=128 K_used=1", 128, 256, 1);
    run_q4k_v3_case("q4k_v3 K=1 N=1024 K_used=1", 1024, 256, 1);
}

#[test]
#[ignore = "requires CUDA GPU — run with --ignored on DGX"]
fn mul_mm_id_q4_k_bf16_v3_parity_k8() {
    run_q4k_v3_case("q4k_v3 K=8 N=128 K_used=8", 128, 256, 8);
    run_q4k_v3_case("q4k_v3 K=8 N=1024 K_used=8", 1024, 256, 8);
    // Qwen3.6-A3B expert_f size :
    run_q4k_v3_case("q4k_v3 K=8 N=18944 K_used=8", 18944, 2048, 8);
}

#[test]
#[ignore = "requires CUDA GPU — run with --ignored on DGX"]
fn mul_mm_id_q4_k_dp4a_parity_k8() {
    run_q4k_dp4a_case("q4k_dp4a K=8 N=128", 128, 256, 8);
    run_q4k_dp4a_case("q4k_dp4a K=8 N=1024", 1024, 256, 8);
    run_q4k_dp4a_case("q4k_dp4a K=8 N=18944 K=2048", 18944, 2048, 8);
}

#[test]
#[ignore = "requires CUDA GPU — run with --ignored on DGX"]
fn mul_mm_id_q5_k_parity_k8() {
    run_q5k_case("q5k K=8 N=128", 128, 256, 8);
    run_q5k_case("q5k K=8 N=1024", 1024, 256, 8);
}

#[test]
#[ignore = "requires CUDA GPU — run with --ignored on DGX"]
fn mul_mm_id_q6_k_parity_k8() {
    run_q6k_case("q6k K=8 N=128", 128, 256, 8);
    run_q6k_case("q6k K=8 N=1024", 1024, 256, 8);
}

#[test]
#[ignore = "requires CUDA GPU — run with --ignored on DGX"]
fn mul_mm_id_bf16_parity_k8() {
    run_bf16_case("bf16 K=8 N=128", 128, 256, 8);
    run_bf16_case("bf16 K=8 N=1024", 1024, 256, 8);
}

/// Routed-reduce parity : `scaled_add_routed_bf16` should equal
/// the K-iteration `scaled_add_inplace_bf16_devscalar` loop.
#[test]
#[ignore = "requires CUDA GPU — run with --ignored on DGX"]
fn scaled_add_routed_parity_k8() {
    let n = 1024usize;
    let k_used = 8usize;
    let ctx = CudaContext::new(0).expect("ctx");
    let stream = ctx.default_stream();
    let kernels = LlmKernels::new(ctx);

    let x_bf: Vec<half::bf16> = (0..(k_used * n))
        .map(|i| half::bf16::from_f32(((i as f32) * 0.013 + 0.7).sin() * 0.4))
        .collect();
    let alpha_bf: Vec<half::bf16> = (0..k_used)
        .map(|i| half::bf16::from_f32(((i as f32) * 0.21 + 0.13).sin() * 0.5))
        .collect();
    let y_init: Vec<half::bf16> = (0..n)
        .map(|i| half::bf16::from_f32(((i as f32) * 0.005 + 0.1).cos() * 0.4))
        .collect();

    let x_dev = stream.memcpy_stod(&x_bf).expect("upload x");
    let alpha_dev = stream.memcpy_stod(&alpha_bf).expect("upload alpha");
    let mut y_ref_dev = stream.memcpy_stod(&y_init).expect("upload y_ref");
    let mut y_mega_dev = stream.memcpy_stod(&y_init).expect("upload y_mega");

    // Reference : K iterations.
    {
        let (xp_base, _g) = x_dev.device_ptr(&stream);
        let (ap, _ga) = alpha_dev.device_ptr(&stream);
        for slot in 0..k_used {
            let xp = xp_base + (slot * n * std::mem::size_of::<half::bf16>()) as u64;
            let (yp, _gy) = y_ref_dev.device_ptr_mut(&stream);
            unsafe {
                kernels
                    .scaled_add_inplace_bf16_devscalar(&stream, yp, xp, ap, slot as i32, n as i32)
                    .expect("ref devscalar");
            }
        }
    }

    // Mega : single launch.
    {
        let (xp, _g) = x_dev.device_ptr(&stream);
        let (ap, _ga) = alpha_dev.device_ptr(&stream);
        let (yp, _gy) = y_mega_dev.device_ptr_mut(&stream);
        unsafe {
            kernels
                .scaled_add_routed_bf16(&stream, yp, xp, ap, n as i32, k_used as i32)
                .expect("mega routed");
        }
    }

    let r_ref: Vec<half::bf16> = stream.memcpy_dtov(&y_ref_dev).expect("dtov ref");
    let r_mega: Vec<half::bf16> = stream.memcpy_dtov(&y_mega_dev).expect("dtov mega");

    eprintln!(
        "first 4 ref vs mega : ref=[{:.5}, {:.5}, {:.5}, {:.5}] mega=[{:.5}, {:.5}, {:.5}, {:.5}]",
        r_ref[0].to_f32(),
        r_ref[1].to_f32(),
        r_ref[2].to_f32(),
        r_ref[3].to_f32(),
        r_mega[0].to_f32(),
        r_mega[1].to_f32(),
        r_mega[2].to_f32(),
        r_mega[3].to_f32()
    );
    // BF16 accumulation across 8 routed slots : reference does
    // (yi + a*xi) → bf16 → next slot, mega does sum-in-fp32 → bf16.
    // Difference is purely the per-step bf16 round, bounded by ~K * eps_bf16
    // ≈ 8 * 2^-7 = 0.0625 absolute or 1% relative for typical magnitudes.
    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;
    let mut bad = 0usize;
    for i in 0..n {
        let a = r_ref[i].to_f32();
        let b = r_mega[i].to_f32();
        let d = (a - b).abs();
        max_abs = max_abs.max(d);
        let denom = a.abs().max(b.abs()).max(1e-3);
        let rel = d / denom;
        max_rel = max_rel.max(rel);
        // Allow up to 5% relative — should be ~1-2% in practice.
        if d > 0.02 && rel > 0.05 {
            bad += 1;
        }
    }
    assert_eq!(
        bad, 0,
        "scaled_add_routed parity : {bad} elements > tol (max_abs={max_abs:.4}, max_rel={max_rel:.4})",
    );
    eprintln!("scaled_add_routed parity : max_abs={max_abs:.5}, max_rel={max_rel:.5}");
}
