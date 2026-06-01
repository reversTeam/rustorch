//! Q4K-MOE-CUBLAS Phase 1 — validate Path A (dequant Q4_K → BF16 + cuBLASLt
//! per-expert SGEMM) against the current sorted Group-GEMM kernel.
//!
//! Representative Qwen3.6-35B-A3B MoE FFN shape :
//!   M=512 tokens, k_used=8, n_experts=128, K=2048, N=expert_f=768
//!
//! For each candidate path we measure the WHOLE per-MoE-matmul cost (one
//! gate/up/down step) :
//!   * sorted GroupGEMM : single `mul_mm_id_gemm_q4_k_sorted_bf16` launch
//!     reading the 128 experts' Q4_K weights once via the permutation table.
//!   * Path A           : (1) dequant Q4_K → BF16 scratch for ALL experts that
//!     have routed tokens, (2) per-expert cuBLASLt `matmul_bf16_rowmajor` over
//!     `[count_e, K] × [N, K]^T → [count_e, N]`. Output layout matches the
//!     sorted kernel (scattered back through `ids_dst`).
//!
//! The Path A timing INCLUDES :
//!   * one-time per-layer dequant of all needed experts
//!   * the gather/scatter pre and post permutation (we leave the input gather
//!     out for clarity — sorted kernel doesn't gather either, the cost is the
//!     128 cuBLASLt launches × M_e tile cost).
//!
//! Pass / fail criterion : Path A < 50 % of the sorted Group-GEMM time.
//!
//! Run :
//!   PATH=/usr/local/cuda-13.0/bin:$PATH \
//!   LD_LIBRARY_PATH=/usr/local/cuda-13.0/targets/sbsa-linux/lib:$LD_LIBRARY_PATH \
//!   cargo run --release --features cuda -p rustorch-cuda \
//!     --example q4k_moe_cublas_bench

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
    const Q4_K_BYTES: usize = 144;

    let ctx = CudaContext::new(0).expect("ctx");
    let stream = ctx.default_stream();
    let kernels = LlmKernels::new(ctx.clone());
    let mut session = LtSession::new(stream.clone()).expect("LtSession");

    // ───────────────────────────────────────────────────────────────
    // Representative Qwen3.6-35B-A3B MoE FFN shape.
    // ───────────────────────────────────────────────────────────────
    let m: usize = 512;
    let k: usize = 2048;
    let n: usize = 768;
    let n_experts: usize = 128;
    let k_used: usize = 8;
    let total_slots = m * k_used; // 4096

    println!("Q4K-MOE-CUBLAS Phase 1 standalone bench");
    println!("  M = {m}, K = {k}, N (expert_f) = {n}");
    println!("  n_experts = {n_experts}, k_used = {k_used}, total_slots = {total_slots}");
    println!();

    // ───────────────────────────────────────────────────────────────
    // Synthetic Q4_K weights : 128 expert weight tiles, each [N, K] = 128
    // rows × 2048 cols = 128 super-blocks per row × N rows.
    // ───────────────────────────────────────────────────────────────
    assert_eq!(k % 256, 0, "K must be multiple of 256 for Q4_K");
    let blocks_per_row = k / 256; // 8
    let bytes_per_expert = n * blocks_per_row * Q4_K_BYTES;
    let bytes_total = bytes_per_expert * n_experts;
    println!(
        "  bytes_per_expert = {bytes_per_expert} ({:.1} MB)",
        bytes_per_expert as f32 / 1e6
    );
    println!(
        "  bytes_total      = {bytes_total} ({:.1} MB)",
        bytes_total as f32 / 1e6
    );

    let w_q4k_host: Vec<u8> = (0..bytes_total)
        .map(|i| ((i * 31 + 17) % 256) as u8)
        .collect();
    let w_q4k_dev = stream.memcpy_stod(&w_q4k_host).expect("w");

    // Per-expert base pointers.
    let (w_base_p, _gp) = w_q4k_dev.device_ptr(&stream);
    let expert_ptrs_host: Vec<u64> = (0..n_experts)
        .map(|e| w_base_p + (e * bytes_per_expert) as u64)
        .collect();
    let expert_ptrs_dev = stream.memcpy_stod(&expert_ptrs_host).expect("expert_ptrs");

    // BF16 input activations [M, K].
    let x_host: Vec<half::bf16> = (0..m * k)
        .map(|i| half::bf16::from_f32(((i % 13) as f32 - 6.0) * 0.001))
        .collect();
    let x_dev = stream.memcpy_stod(&x_host).expect("x");

    // Balanced routing : token t picks experts (t + offset) % n_experts for
    // offset = 0..k_used. Identical to llama.cpp's balanced-routing test.
    let topk_indices_host: Vec<i32> = (0..total_slots)
        .map(|i| {
            let token = i / k_used;
            let slot = i % k_used;
            ((token * k_used + slot) % n_experts) as i32
        })
        .collect();
    let topk_indices_dev = stream.memcpy_stod(&topk_indices_host).expect("topk");

    // Scratch for ids_helper + sorted Group-GEMM.
    let mut ids_src1_dev = stream.alloc_zeros::<i32>(total_slots).expect("ids_src1");
    let mut ids_dst_dev = stream.alloc_zeros::<i32>(total_slots).expect("ids_dst");
    let mut expert_bounds_dev = stream
        .alloc_zeros::<i32>(n_experts + 1)
        .expect("expert_bounds");
    let mut y_sorted_dev = stream
        .alloc_zeros::<half::bf16>(total_slots * n)
        .expect("y_sorted");

    // ───────────────────────────────────────────────────────────────
    // Build sort permutation (mm_ids_helper_bf16).
    // ───────────────────────────────────────────────────────────────
    unsafe {
        let (tk_p, _g) = topk_indices_dev.device_ptr(&stream);
        let (is_p, _g) = ids_src1_dev.device_ptr_mut(&stream);
        let (id_p, _g) = ids_dst_dev.device_ptr_mut(&stream);
        let (eb_p, _g) = expert_bounds_dev.device_ptr_mut(&stream);
        kernels
            .mm_ids_helper_bf16(
                &stream,
                tk_p,
                is_p,
                id_p,
                eb_p,
                m as i32,
                k_used as i32,
                n_experts as i32,
            )
            .expect("mm_ids_helper");
    }
    stream.synchronize().expect("sync helper");

    let expert_bounds_host: Vec<i32> = stream.memcpy_dtov(&expert_bounds_dev).expect("eb_dtov");
    let mut active_experts = 0usize;
    let mut max_e = 0i32;
    let mut min_e = i32::MAX;
    for e in 0..n_experts {
        let cnt = expert_bounds_host[e + 1] - expert_bounds_host[e];
        if cnt > 0 {
            active_experts += 1;
            max_e = max_e.max(cnt);
            min_e = min_e.min(cnt);
        }
    }
    println!("  routed : active_experts={active_experts}, per-expert min={min_e}, max={max_e}");
    println!();

    // ───────────────────────────────────────────────────────────────
    // (A) Sorted Group-GEMM baseline.
    // ───────────────────────────────────────────────────────────────
    let warmup = 4usize;
    let iters = 32usize;

    unsafe {
        let (ep_p, _g) = expert_ptrs_dev.device_ptr(&stream);
        let (tk_p, _g) = topk_indices_dev.device_ptr(&stream);
        let (is_p, _g) = ids_src1_dev.device_ptr(&stream);
        let (id_p, _g) = ids_dst_dev.device_ptr(&stream);
        let (x_p, _g) = x_dev.device_ptr(&stream);
        let (y_p, _g) = y_sorted_dev.device_ptr_mut(&stream);
        for _ in 0..warmup {
            kernels
                .mul_mm_id_gemm_q4_k_sorted_bf16(
                    &stream,
                    ep_p,
                    tk_p,
                    is_p,
                    id_p,
                    x_p,
                    y_p,
                    m as i32,
                    n as i32,
                    k as i32,
                    k_used as i32,
                )
                .expect("sorted warmup");
        }
    }
    stream.synchronize().expect("sync sorted warmup");

    let t0 = Instant::now();
    unsafe {
        let (ep_p, _g) = expert_ptrs_dev.device_ptr(&stream);
        let (tk_p, _g) = topk_indices_dev.device_ptr(&stream);
        let (is_p, _g) = ids_src1_dev.device_ptr(&stream);
        let (id_p, _g) = ids_dst_dev.device_ptr(&stream);
        let (x_p, _g) = x_dev.device_ptr(&stream);
        let (y_p, _g) = y_sorted_dev.device_ptr_mut(&stream);
        for _ in 0..iters {
            kernels
                .mul_mm_id_gemm_q4_k_sorted_bf16(
                    &stream,
                    ep_p,
                    tk_p,
                    is_p,
                    id_p,
                    x_p,
                    y_p,
                    m as i32,
                    n as i32,
                    k as i32,
                    k_used as i32,
                )
                .expect("sorted iter");
        }
    }
    stream.synchronize().expect("sync sorted bench");
    let sorted_ms = t0.elapsed().as_secs_f32() * 1000.0 / iters as f32;

    println!("  (A) sorted Group-GEMM : {sorted_ms:.4} ms / call");

    // ───────────────────────────────────────────────────────────────
    // (B) Path A : dequant Q4_K → BF16 scratch (full 128 experts) +
    //     per-expert cuBLASLt matmul_bf16_rowmajor.
    //
    // Pre-arrange a CONTIGUOUS [active_experts, M_e_max, K] gathered
    // input buffer ? We skip that — measure the dequant + per-expert
    // GEMM call cost only, since the gather/scatter overhead is fully
    // attributable to Path A but secondary to validate viability.
    // ───────────────────────────────────────────────────────────────

    // BF16 scratch [n_experts, N, K] — worst case is all experts active.
    let dequant_n_blocks = (n_experts * n * blocks_per_row) as i64;
    let dequant_elements = (n_experts * n * k) as usize;
    let mut w_bf16_dev = stream
        .alloc_zeros::<half::bf16>(dequant_elements)
        .expect("w_bf16");
    println!(
        "  (B) Path A : dequant scratch = {:.1} MB",
        (dequant_elements * 2) as f32 / 1e6
    );

    // Per-expert output buffer : `[total_slots, N]` (output of all per-expert
    // GEMMs, concatenated). For balanced routing every expert gets ~32 tokens,
    // we provide a buffer of the same shape as `y_sorted_dev` for simplicity.
    let mut y_path_a_dev = stream
        .alloc_zeros::<half::bf16>(total_slots * n)
        .expect("y_path_a");

    // Pre-pack input : for each compact slot, we have the source token's [K]
    // row. The gather kernel doesn't exist — to time Path A *upper bound* we
    // pass the original `x_dev` (which means cuBLASLt sees a strided gather
    // we can't express via simple cuBLASLt API). To stay strictly comparable
    // we instead PRE-COPY each expert's relevant tokens into a CONTIGUOUS
    // staging buffer once, BEFORE the timed loop (this favors Path A — gives
    // it the most charitable measurement).
    let total_slots_padded = total_slots; // exact
    let mut x_gather_dev = stream
        .alloc_zeros::<half::bf16>(total_slots_padded * k)
        .expect("x_gather");

    // Build gather permutation : `ids_src1[c] * k` is the source row offset.
    let ids_src1_host: Vec<i32> = stream.memcpy_dtov(&ids_src1_dev).expect("ids_src1_dtov");
    {
        // CPU gather (one-time, not timed) — fill x_gather from x_host per ids_src1.
        let x_gather_host: Vec<half::bf16> = (0..total_slots)
            .flat_map(|c| {
                let token = ids_src1_host[c] as usize;
                let row_start = token * k;
                x_host[row_start..row_start + k].iter().copied()
            })
            .collect();
        x_gather_dev = stream.memcpy_stod(&x_gather_host).expect("x_gather upload");
    }

    // Warmup Path A.
    unsafe {
        let (wq_p, _g) = w_q4k_dev.device_ptr(&stream);
        let (wb_p, _g) = w_bf16_dev.device_ptr_mut(&stream);
        kernels
            .dequant_q4_k_to_bf16(&stream, wq_p, wb_p, dequant_n_blocks)
            .expect("dequant warmup");
    }
    stream.synchronize().expect("sync dequant warmup");

    // Per-expert cuBLASLt matmul : for each active expert e with count_e > 0,
    //   y_path_a[start_e : start_e + count_e, :] =
    //     x_gather[start_e : start_e + count_e, :] @ w_bf16[e, :, :].T
    // Each call shape : M=count_e, N=768, K=2048.
    for _ in 0..warmup {
        unsafe {
            let (wq_p, _g) = w_q4k_dev.device_ptr(&stream);
            let (wb_p, _g) = w_bf16_dev.device_ptr_mut(&stream);
            kernels
                .dequant_q4_k_to_bf16(&stream, wq_p, wb_p, dequant_n_blocks)
                .expect("dequant iter");
        }
        let (wb_p, _g) = unsafe { w_bf16_dev.device_ptr(&stream) };
        let (xg_p, _g) = unsafe { x_gather_dev.device_ptr(&stream) };
        let (yp_p, _g) = unsafe { y_path_a_dev.device_ptr_mut(&stream) };
        for e in 0..n_experts {
            let start = expert_bounds_host[e] as usize;
            let end = expert_bounds_host[e + 1] as usize;
            let count = end - start;
            if count == 0 {
                continue;
            }
            let w_e_p = wb_p + (e * n * k * 2) as u64; // *2 = sizeof(bf16)
            let x_e_p = xg_p + (start * k * 2) as u64;
            let y_e_p = yp_p + (start * n * 2) as u64;
            unsafe {
                session
                    .matmul_bf16_rowmajor(w_e_p, x_e_p, y_e_p, count, n, k, 1.0, 0.0)
                    .expect("path A warmup cuBLASLt");
            }
        }
    }
    stream.synchronize().expect("sync path A warmup");

    // Path A timed run.
    let t0 = Instant::now();
    for _ in 0..iters {
        unsafe {
            let (wq_p, _g) = w_q4k_dev.device_ptr(&stream);
            let (wb_p, _g) = w_bf16_dev.device_ptr_mut(&stream);
            kernels
                .dequant_q4_k_to_bf16(&stream, wq_p, wb_p, dequant_n_blocks)
                .expect("dequant timed");
        }
        let (wb_p, _g) = unsafe { w_bf16_dev.device_ptr(&stream) };
        let (xg_p, _g) = unsafe { x_gather_dev.device_ptr(&stream) };
        let (yp_p, _g) = unsafe { y_path_a_dev.device_ptr_mut(&stream) };
        for e in 0..n_experts {
            let start = expert_bounds_host[e] as usize;
            let end = expert_bounds_host[e + 1] as usize;
            let count = end - start;
            if count == 0 {
                continue;
            }
            let w_e_p = wb_p + (e * n * k * 2) as u64;
            let x_e_p = xg_p + (start * k * 2) as u64;
            let y_e_p = yp_p + (start * n * 2) as u64;
            unsafe {
                session
                    .matmul_bf16_rowmajor(w_e_p, x_e_p, y_e_p, count, n, k, 1.0, 0.0)
                    .expect("path A iter");
            }
        }
    }
    stream.synchronize().expect("sync path A bench");
    let path_a_ms = t0.elapsed().as_secs_f32() * 1000.0 / iters as f32;

    println!("  (B) dequant + cuBLASLt (Path A) : {path_a_ms:.4} ms / call");

    // ───────────────────────────────────────────────────────────────
    // (C) Standalone dequant-only timing (informational).
    // ───────────────────────────────────────────────────────────────
    let t0 = Instant::now();
    for _ in 0..iters {
        unsafe {
            let (wq_p, _g) = w_q4k_dev.device_ptr(&stream);
            let (wb_p, _g) = w_bf16_dev.device_ptr_mut(&stream);
            kernels
                .dequant_q4_k_to_bf16(&stream, wq_p, wb_p, dequant_n_blocks)
                .expect("dequant standalone");
        }
    }
    stream.synchronize().expect("sync dequant standalone");
    let dequant_only_ms = t0.elapsed().as_secs_f32() * 1000.0 / iters as f32;
    println!("    (dequant-only standalone : {dequant_only_ms:.4} ms / call)");

    // ───────────────────────────────────────────────────────────────
    // Report.
    // ───────────────────────────────────────────────────────────────
    println!();
    let speedup = sorted_ms / path_a_ms;
    let viable = path_a_ms < 0.5 * sorted_ms;
    println!("───────────────────────────────────────────────────────");
    println!("  Path A speedup over sorted Group-GEMM : {speedup:.2}×");
    if viable {
        println!("  Result : VIABLE (Path A < 50 % of baseline) — proceed Phase 2");
    } else {
        println!("  Result : NON-VIABLE (Path A >= 50 % of baseline) — deadend Phase 2/3");
        println!(
            "  Reason : dequant scratch BW dominates. Dequant alone = {:.1} % of baseline.",
            100.0 * dequant_only_ms / sorted_ms
        );
    }
}
