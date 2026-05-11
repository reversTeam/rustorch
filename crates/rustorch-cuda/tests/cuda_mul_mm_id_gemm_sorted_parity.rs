//! T246.10 TrackE.3 — Parity tests for the sort-permutation Group-GEMM kernel
//! `mul_mm_id_gemm_q4_k_sorted_bf16` against the unsorted TrackE.2 reference
//! `mul_mm_id_gemm_q4_k_bf16`.
//!
//! Both kernels :
//!   - Read the same `[M, K]` BF16 activations
//!   - Read the same `[M, k_used]` int32 topk_indices
//!   - Write the same `[M, k_used, N]` BF16 output layout
//!   - Use byte-for-byte identical FP arithmetic (warp-shuffle Q4_K BF16 dot)
//!
//! Only the schedule changes : the sorted kernel iterates a compact
//! by-expert slot index, while the unsorted iterates `(token, slot)` in
//! original order. → BIT-EXACT parity is REQUIRED (no ULP slack).
//!
//! The helper kernel `mm_ids_helper_bf16` is implicitly exercised :
//! the sorted-Group-GEMM consumes its `ids_src1` / `ids_dst` outputs.
//!
//! Run on GB10 (DGX Spark) :
//!   PATH=/usr/local/cuda-13.0/bin:$PATH \
//!     LD_LIBRARY_PATH=/usr/local/cuda-13.0/targets/sbsa-linux/lib:$LD_LIBRARY_PATH \
//!     cargo test --release --features cuda -p rustorch-cuda \
//!       --test cuda_mul_mm_id_gemm_sorted_parity -- --ignored --nocapture --test-threads=1

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

/// Mirror of the topk distribution used by TrackE.2 parity tests :
/// `(m * 7919 + s * 23 + 1) % n_experts`. Deterministic, visits many experts.
fn make_topk(m_tokens: usize, k_used: usize, n_experts: usize) -> Vec<i32> {
    (0..m_tokens)
        .flat_map(|m| {
            (0..k_used).map(move |s| {
                let v = (m.wrapping_mul(7919) + s.wrapping_mul(23) + 1) % n_experts;
                v as i32
            })
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn run_sorted_parity_q4k(label: &str, m_tokens: usize, n_rows: usize, k_dim: usize, k_used: usize) {
    // We deliberately use a LARGE expert count (128 = Qwen3.6-A3B) to stress
    // mm_ids_helper's prefix-scan over many experts. Smaller cases too.
    let n_experts = 128usize;

    let ctx = CudaContext::new(0).expect("ctx");
    let stream = ctx.default_stream();
    let kernels = LlmKernels::new(ctx);

    // Build expert weight blobs and upload.
    let blobs: Vec<Vec<u8>> = (0..n_experts)
        .map(|e| build_q4k_blob(0xa2_b3_4d_e5 ^ (e as u64 * 17), n_rows, k_dim))
        .collect();
    let mut exp_dev: Vec<cudarc::driver::CudaSlice<u8>> = Vec::new();
    for w in &blobs {
        exp_dev.push(stream.memcpy_stod(w).expect("upload expert"));
    }
    let mut ptrs: Vec<u64> = Vec::with_capacity(n_experts);
    for d in &exp_dev {
        let (p, _g) = d.device_ptr(&stream);
        ptrs.push(p);
    }
    let ptrs_dev = stream.memcpy_stod(&ptrs).expect("upload ptrs");

    // Activations [M, K] BF16.
    let mut x_all = Vec::<half::bf16>::with_capacity(m_tokens * k_dim);
    for tok in 0..m_tokens {
        let v = bf16_vec(k_dim, 0.011 + (tok as f32) * 0.073);
        x_all.extend_from_slice(&v);
    }
    let x_dev = stream.memcpy_stod(&x_all).expect("upload x");

    // Topk indices [M, k_used] i32.
    let topk_all = make_topk(m_tokens, k_used, n_experts);
    let topk_dev = stream.memcpy_stod(&topk_all).expect("upload topk");

    // Permutation scratch arrays produced by mm_ids_helper.
    let n_slots = m_tokens * k_used;
    let ids_src1 = stream.alloc_zeros::<i32>(n_slots).expect("alloc ids_src1");
    let ids_dst = stream.alloc_zeros::<i32>(n_slots).expect("alloc ids_dst");
    let expert_bounds = stream
        .alloc_zeros::<i32>(n_experts + 1)
        .expect("alloc expert_bounds");

    // Build permutation on device.
    unsafe {
        let (tp, _g) = topk_dev.device_ptr(&stream);
        let (ip1, _g1) = ids_src1.device_ptr(&stream);
        let (ip2, _g2) = ids_dst.device_ptr(&stream);
        let (epb, _g3) = expert_bounds.device_ptr(&stream);
        kernels
            .mm_ids_helper_bf16(
                &stream,
                tp,
                ip1,
                ip2,
                epb,
                m_tokens as i32,
                k_used as i32,
                n_experts as i32,
            )
            .expect("mm_ids_helper_bf16");
    }

    // Verify expert_bounds : every expert's count should sum to n_slots.
    let bounds_host: Vec<i32> = stream.memcpy_dtov(&expert_bounds).expect("dtov bounds");
    assert_eq!(
        bounds_host[n_experts] as usize, n_slots,
        "{label} : expert_bounds[n_experts] = {} but n_slots = {n_slots}",
        bounds_host[n_experts]
    );

    // Verify permutation : every (token, slot) original pair appears exactly
    // once in ids_dst, and ids_src1[c] / ids_dst[c] are consistent.
    let src1_host: Vec<i32> = stream.memcpy_dtov(&ids_src1).expect("dtov src1");
    let dst_host: Vec<i32> = stream.memcpy_dtov(&ids_dst).expect("dtov dst");
    let mut seen = vec![false; n_slots];
    for c in 0..n_slots {
        let token = src1_host[c] as usize;
        let dst_lin = dst_host[c] as usize;
        let slot = dst_lin - token * k_used;
        assert!(
            token < m_tokens && slot < k_used && dst_lin == token * k_used + slot,
            "{label} : c={c} inconsistent src1={} dst={} (M={} k_used={})",
            src1_host[c],
            dst_host[c],
            m_tokens,
            k_used
        );
        assert!(
            !seen[dst_lin],
            "{label} : c={c} duplicate dst_lin={dst_lin}",
        );
        seen[dst_lin] = true;
    }
    for (i, &s) in seen.iter().enumerate() {
        assert!(s, "{label} : dst_lin={i} never seen in permutation");
    }
    // Verify sort : compact slots are in non-decreasing expert order.
    let mut prev_e: i32 = -1;
    for c in 0..n_slots {
        let token = src1_host[c] as usize;
        let dst_lin = dst_host[c] as usize;
        let slot = dst_lin - token * k_used;
        let e = topk_all[token * k_used + slot];
        assert!(
            e >= prev_e,
            "{label} : sort broken at c={c} (e={e} < prev_e={prev_e})"
        );
        prev_e = e;
    }

    // Reference output via the unsorted TrackE.2 kernel.
    let mut y_ref = stream
        .alloc_zeros::<half::bf16>(m_tokens * k_used * n_rows)
        .expect("alloc y_ref");
    unsafe {
        let (pp, _g) = ptrs_dev.device_ptr(&stream);
        let (tp, _g1) = topk_dev.device_ptr(&stream);
        let (xp, _g2) = x_dev.device_ptr(&stream);
        let (yp, _g3) = y_ref.device_ptr_mut(&stream);
        kernels
            .mul_mm_id_gemm_q4_k_bf16(
                &stream,
                pp,
                tp,
                xp,
                yp,
                m_tokens as i32,
                n_rows as i32,
                k_dim as i32,
                k_used as i32,
            )
            .expect("mul_mm_id_gemm_q4_k_bf16");
    }

    // Sorted output via TrackE.3 kernel.
    let mut y_sorted = stream
        .alloc_zeros::<half::bf16>(m_tokens * k_used * n_rows)
        .expect("alloc y_sorted");
    unsafe {
        let (pp, _g) = ptrs_dev.device_ptr(&stream);
        let (tp, _g1) = topk_dev.device_ptr(&stream);
        let (ip1, _g2) = ids_src1.device_ptr(&stream);
        let (ip2, _g3) = ids_dst.device_ptr(&stream);
        let (xp, _g4) = x_dev.device_ptr(&stream);
        let (yp, _g5) = y_sorted.device_ptr_mut(&stream);
        kernels
            .mul_mm_id_gemm_q4_k_sorted_bf16(
                &stream,
                pp,
                tp,
                ip1,
                ip2,
                xp,
                yp,
                m_tokens as i32,
                n_rows as i32,
                k_dim as i32,
                k_used as i32,
            )
            .expect("mul_mm_id_gemm_q4_k_sorted_bf16");
    }

    let r_ref: Vec<half::bf16> = stream.memcpy_dtov(&y_ref).expect("dtov ref");
    let r_sort: Vec<half::bf16> = stream.memcpy_dtov(&y_sorted).expect("dtov sort");

    let mut diffs = 0usize;
    for tok in 0..m_tokens {
        for slot in 0..k_used {
            for i in 0..n_rows {
                let idx = (tok * k_used + slot) * n_rows + i;
                let ar = r_ref[idx];
                let am = r_sort[idx];
                if ar.to_bits() != am.to_bits() {
                    if diffs < 8 {
                        eprintln!(
                            "{label} tok={tok} slot={slot} row={i} \
                             ref=0x{:04x}={} sort=0x{:04x}={} (e_idx={})",
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
    eprintln!(
        "PASS {label} : M={m_tokens} N={n_rows} K={k_dim} k_used={k_used} n_experts={n_experts}"
    );
}

#[test]
#[ignore = "requires CUDA GPU — run with --ignored on DGX"]
fn sorted_q4k_parity_small() {
    run_sorted_parity_q4k("sorted-q4k M=8 N=128 K=256", 8, 128, 256, 8);
    run_sorted_parity_q4k("sorted-q4k M=32 N=128 K=256", 32, 128, 256, 8);
}

#[test]
#[ignore = "requires CUDA GPU — run with --ignored on DGX"]
fn sorted_q4k_parity_medium() {
    run_sorted_parity_q4k("sorted-q4k M=128 N=1024 K=2048", 128, 1024, 2048, 8);
}

#[test]
#[ignore = "requires CUDA GPU — run with --ignored on DGX"]
fn sorted_q4k_parity_large() {
    // Approximate Qwen3.6-35B-A3B MoE FFN shape : 128 experts, ef=N=4096
    // (instead of 18944 to keep test cost reasonable), K=d=2048.
    run_sorted_parity_q4k("sorted-q4k M=512 N=4096 K=2048", 512, 4096, 2048, 8);
}

/// Exact Qwen3.6-35B-A3B MoE gate/up shape : M=8, N=ef=18944, K=d=2048, k_used=8,
/// n_experts=128 — the smallest prefill batch that triggers Group-GEMM dispatch.
#[test]
#[ignore = "requires CUDA GPU — run with --ignored on DGX"]
fn sorted_q4k_parity_qwen36_a3b_min() {
    run_sorted_parity_q4k("sorted-q4k Qwen3.6-A3B M=8", 8, 18944, 2048, 8);
}
