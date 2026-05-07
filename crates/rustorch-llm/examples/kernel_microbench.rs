//! T182 — Kernel micro-benchmark suite.
//!
//! This bench runs each suspect kernel from the Qwen3.6-35B-A3B decode path
//! 1000× in isolation on synthetic data of the actual shapes used in production.
//! The whole 1000-iter sequence is committed in a single command buffer; we
//! measure wall-clock around `drain()`. With 1000 iterations the per-CB setup
//! overhead is amortized and the average gives a fiable per-kernel cost.
//!
//! ## Why this approach
//!
//! Our `profile_drain_record` was producing drain-cumulative artifacts. Our
//! attempt at MTLCounterSampleBuffer hit Apple Silicon driver constraints
//! (T181). Micro-benchmarks in isolation work without any instrumentation —
//! same approach as `llama-bench` / `gguf-bench`.
//!
//! ## Run
//!
//! ```sh
//! cargo run --release --example kernel_microbench -p rustorch-llm
//! ```
//!
//! Output : table with kernel name, per-iter µs, total ms, theoretical
//! contribution to a forward pass at the layer counts of Qwen3.6-35B-A3B.

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("kernel_microbench only runs on macOS (Metal backend)");
}

#[cfg(target_os = "macos")]
fn main() {
    use half::f16;
    use rustorch_metal::backend_singleton::metal_backend;
    use rustorch_metal::kernels::{
        add_inplace_f32, argmax_batched_f32, delta_net_step_with_l2_f32, gqa_decode_f32,
        kv_append_f32, rms_norm_f32, rms_norm_per_head_gated_f32, rope_half_split_f32,
        sgemv_f32_cached_x_into, sgemv_f32_lcpp_simd_into, sgemv_q4_k_f32_lcpp_nsg2_into,
        sgemv_q4_k_gather_f32_lcpp_nsg2_into, sgemv_q5_k_f32_lcpp_nsg2_into,
        sgemv_q6_k_f32_lcpp_nsg2_into, sgemv_q8_0_f32_lcpp_nsg2_into, sigmoid_add_moe_f32,
        sigmoid_mul_inplace_f32, split_qkv_f32, ssm_apply_gate_f32, ssm_conv1d_step_f32,
        swiglu_f32, topk_softmax_norm_f32, topk_softmax_norm_parallel_f32, weighted_reduce_add_f32,
    };
    use std::time::Instant;

    const ITERS: usize = 1000;

    let backend = metal_backend();
    println!(
        "device: {} (Metal3: {})",
        backend.adapter_name(),
        backend.supports_metal3()
    );
    println!("\n=== kernel_microbench ({} iters per kernel) ===", ITERS);

    // Helper: synthetic Q4_K bytes (filled with non-zero pattern).
    fn fake_q4k_bytes(k: usize, n: usize) -> Vec<u8> {
        let blocks_per_row = k / 256;
        let row_bytes = blocks_per_row * 144;
        let mut bytes = vec![0u8; n * row_bytes];
        for (i, b) in bytes.iter_mut().enumerate() {
            // Non-zero deterministic pattern; doesn't matter for timing.
            *b = ((i * 7 + 1) & 0xFF) as u8;
        }
        bytes
    }

    fn fake_q5k_bytes(k: usize, n: usize) -> Vec<u8> {
        // Q5_K: 256 weights / 176 bytes per super-block.
        let blocks_per_row = k / 256;
        let row_bytes = blocks_per_row * 176;
        let mut bytes = vec![0u8; n * row_bytes];
        for (i, b) in bytes.iter_mut().enumerate() {
            *b = ((i * 11 + 1) & 0xFF) as u8;
        }
        bytes
    }

    fn fake_q6k_bytes(k: usize, n: usize) -> Vec<u8> {
        let blocks_per_row = k / 256;
        let row_bytes = blocks_per_row * 210;
        let mut bytes = vec![0u8; n * row_bytes];
        for (i, b) in bytes.iter_mut().enumerate() {
            *b = ((i * 13 + 1) & 0xFF) as u8;
        }
        bytes
    }

    fn fake_q8_0_bytes(k: usize, n: usize) -> Vec<u8> {
        // Q8_0: 32 weights / 34 bytes per block.
        let blocks_per_row = k / 32;
        let row_bytes = blocks_per_row * 34;
        let mut bytes = vec![0u8; n * row_bytes];
        // Fill scale (fp16=1.0) + int8 weights with small pattern.
        let one_h = f16::from_f32(0.1).to_le_bytes();
        for blk in 0..(n * blocks_per_row) {
            let off = blk * 34;
            bytes[off] = one_h[0];
            bytes[off + 1] = one_h[1];
            for j in 0..32 {
                bytes[off + 2 + j] = ((blk + j) as i8 & 0x7f) as u8;
            }
        }
        bytes
    }

    fn fake_f32(n: usize, seed: f32) -> Vec<f32> {
        (0..n)
            .map(|i| ((i as f32 + 1.0) * seed * 0.001).sin())
            .collect()
    }

    // Result accumulator
    let mut results: Vec<(String, f64, f64, String, usize)> = Vec::new(); // (label, total_ms, avg_us, shape, calls_per_token)

    // -------- Kernel 1: F32 routing matmul (gate_inp), K=2048 N=256 (CURRENT) --------
    let rt_k = 2048usize;
    let rt_n = 256usize;
    let rt_x = fake_f32(rt_k, 1.7);
    let rt_w = fake_f32(rt_k * rt_n, 2.3);
    let rt_x_buf = backend.alloc_shared(rt_k * 4).unwrap();
    let rt_w_buf = backend.alloc_shared(rt_k * rt_n * 4).unwrap();
    let rt_y_buf = backend.alloc_shared(rt_n * 4).unwrap();
    unsafe {
        std::ptr::copy_nonoverlapping(rt_x.as_ptr(), rt_x_buf.contents() as *mut f32, rt_k);
        std::ptr::copy_nonoverlapping(rt_w.as_ptr(), rt_w_buf.contents() as *mut f32, rt_k * rt_n);
    }
    {
        sgemv_f32_lcpp_simd_into(backend, &rt_x_buf, &rt_w_buf, &rt_y_buf, rt_k, rt_n).unwrap();
        backend.drain();
        let t0 = Instant::now();
        for _ in 0..ITERS {
            sgemv_f32_lcpp_simd_into(backend, &rt_x_buf, &rt_w_buf, &rt_y_buf, rt_k, rt_n).unwrap();
        }
        backend.drain();
        let dt = t0.elapsed();
        results.push((
            "sgemv_f32 (routing CURRENT)".to_string(),
            dt.as_secs_f64() * 1000.0,
            dt.as_secs_f64() * 1e6 / ITERS as f64,
            format!("K={rt_k} N={rt_n} F32"),
            40,
        ));
    }
    // -------- Kernel 1b: T183 F32 sgemv with cached x --------
    {
        sgemv_f32_cached_x_into(backend, &rt_x_buf, &rt_w_buf, &rt_y_buf, rt_k, rt_n).unwrap();
        backend.drain();
        let t0 = Instant::now();
        for _ in 0..ITERS {
            sgemv_f32_cached_x_into(backend, &rt_x_buf, &rt_w_buf, &rt_y_buf, rt_k, rt_n).unwrap();
        }
        backend.drain();
        let dt = t0.elapsed();
        results.push((
            "sgemv_f32 CACHED_X (T183)".to_string(),
            dt.as_secs_f64() * 1000.0,
            dt.as_secs_f64() * 1e6 / ITERS as f64,
            format!("K={rt_k} N={rt_n} F32"),
            40,
        ));
    }
    // -------- Parity check T183 vs current --------
    {
        let mut y_cur = vec![0.0_f32; rt_n];
        let mut y_t183 = vec![0.0_f32; rt_n];
        sgemv_f32_lcpp_simd_into(backend, &rt_x_buf, &rt_w_buf, &rt_y_buf, rt_k, rt_n).unwrap();
        backend.drain();
        unsafe {
            std::ptr::copy_nonoverlapping(
                rt_y_buf.contents() as *const f32,
                y_cur.as_mut_ptr(),
                rt_n,
            );
        }
        sgemv_f32_cached_x_into(backend, &rt_x_buf, &rt_w_buf, &rt_y_buf, rt_k, rt_n).unwrap();
        backend.drain();
        unsafe {
            std::ptr::copy_nonoverlapping(
                rt_y_buf.contents() as *const f32,
                y_t183.as_mut_ptr(),
                rt_n,
            );
        }
        let mut max_rel = 0.0_f32;
        for (a, b) in y_cur.iter().zip(y_t183.iter()) {
            let denom = a.abs().max(1e-4);
            let rel = (a - b).abs() / denom;
            if rel > max_rel {
                max_rel = rel;
            }
        }
        println!(
            "\nParity T183 sgemv_f32_cached_x: max_rel_err = {max_rel:.3e}  ({})",
            if max_rel < 1e-4 {
                "PASS ✓"
            } else {
                "FAIL ✗"
            }
        );
    }

    // -------- Kernel 2: Q4_K MoE gather sgemv, K=2048 N=512 b=8 --------
    // gate_proj/up_proj use the gather variant.
    {
        let k = 2048usize;
        let n = 512usize; // ef
        let n_used = 8usize;
        let n_experts = 256usize;
        let bytes_per_expert = (k / 256) * 144 * n;
        let total_bytes = n_experts * bytes_per_expert;
        let x = fake_f32(k, 3.1);
        let w = fake_q4k_bytes(k, n * n_experts);
        let indices: Vec<u32> = (0..n_used).map(|i| i as u32).collect();
        let x_buf = backend.alloc_shared(k * 4).unwrap();
        let w_buf = backend.alloc_shared(total_bytes).unwrap();
        let idx_buf = backend.alloc_shared(n_used * 4).unwrap();
        let y_buf = backend.alloc_shared(n_used * n * 4).unwrap();
        unsafe {
            std::ptr::copy_nonoverlapping(x.as_ptr(), x_buf.contents() as *mut f32, k);
            std::ptr::copy_nonoverlapping(
                w.as_ptr(),
                w_buf.contents() as *mut u8,
                w.len().min(total_bytes),
            );
            std::ptr::copy_nonoverlapping(indices.as_ptr(), idx_buf.contents() as *mut u32, n_used);
        }
        // Warmup
        sgemv_q4_k_gather_f32_lcpp_nsg2_into(
            backend,
            &x_buf,
            &w_buf,
            &idx_buf,
            n_used,
            &y_buf,
            k,
            n,
            bytes_per_expert,
            0,
        )
        .unwrap();
        backend.drain();
        let t0 = Instant::now();
        for _ in 0..ITERS {
            sgemv_q4_k_gather_f32_lcpp_nsg2_into(
                backend,
                &x_buf,
                &w_buf,
                &idx_buf,
                n_used,
                &y_buf,
                k,
                n,
                bytes_per_expert,
                0,
            )
            .unwrap();
        }
        backend.drain();
        let dt = t0.elapsed();
        results.push((
            "sgemv_q4_k_gather (MoE expert)".to_string(),
            dt.as_secs_f64() * 1000.0,
            dt.as_secs_f64() * 1e6 / ITERS as f64,
            format!("K={k} N={n} Q4_K b={n_used}"),
            120, // 40 layers × 3 projections (gate/up/down)
        ));
    }

    // -------- Kernel 3: Q5_K SSM w_qkv, K=2048 N=8192 --------
    {
        let k = 2048usize;
        let n = 8192usize;
        let x = fake_f32(k, 4.2);
        let w = fake_q5k_bytes(k, n);
        let x_buf = backend.alloc_shared(k * 4).unwrap();
        let w_buf = backend.alloc_shared(w.len()).unwrap();
        let y_buf = backend.alloc_shared(n * 4).unwrap();
        unsafe {
            std::ptr::copy_nonoverlapping(x.as_ptr(), x_buf.contents() as *mut f32, k);
            std::ptr::copy_nonoverlapping(w.as_ptr(), w_buf.contents() as *mut u8, w.len());
        }
        sgemv_q5_k_f32_lcpp_nsg2_into(backend, &x_buf, &w_buf, &y_buf, k, n).unwrap();
        backend.drain();
        let t0 = Instant::now();
        for _ in 0..ITERS {
            sgemv_q5_k_f32_lcpp_nsg2_into(backend, &x_buf, &w_buf, &y_buf, k, n).unwrap();
        }
        backend.drain();
        let dt = t0.elapsed();
        results.push((
            "sgemv_q5_k (SSM w_qkv)".to_string(),
            dt.as_secs_f64() * 1000.0,
            dt.as_secs_f64() * 1e6 / ITERS as f64,
            format!("K={k} N={n} Q5_K"),
            30, // 30 SSM layers
        ));
    }

    // -------- Kernel 4: Q5_K SSM w_gate, K=2048 N=4096 --------
    {
        let k = 2048usize;
        let n = 4096usize;
        let w = fake_q5k_bytes(k, n);
        let x_buf = backend.alloc_shared(k * 4).unwrap();
        let w_buf = backend.alloc_shared(w.len()).unwrap();
        let y_buf = backend.alloc_shared(n * 4).unwrap();
        unsafe {
            std::ptr::copy_nonoverlapping(w.as_ptr(), w_buf.contents() as *mut u8, w.len());
        }
        sgemv_q5_k_f32_lcpp_nsg2_into(backend, &x_buf, &w_buf, &y_buf, k, n).unwrap();
        backend.drain();
        let t0 = Instant::now();
        for _ in 0..ITERS {
            sgemv_q5_k_f32_lcpp_nsg2_into(backend, &x_buf, &w_buf, &y_buf, k, n).unwrap();
        }
        backend.drain();
        let dt = t0.elapsed();
        results.push((
            "sgemv_q5_k (SSM w_gate)".to_string(),
            dt.as_secs_f64() * 1000.0,
            dt.as_secs_f64() * 1e6 / ITERS as f64,
            format!("K={k} N={n} Q5_K"),
            30,
        ));
    }

    // -------- Kernel 5: Q4_K SSM out_proj K=4096 N=2048 (Q4_K standard sgemv) --------
    {
        let k = 4096usize;
        let n = 2048usize;
        let w = fake_q4k_bytes(k, n);
        let x_buf = backend.alloc_shared(k * 4).unwrap();
        let w_buf = backend.alloc_shared(w.len()).unwrap();
        let y_buf = backend.alloc_shared(n * 4).unwrap();
        unsafe {
            std::ptr::copy_nonoverlapping(w.as_ptr(), w_buf.contents() as *mut u8, w.len());
        }
        sgemv_q4_k_f32_lcpp_nsg2_into(backend, &x_buf, &w_buf, &y_buf, k, n).unwrap();
        backend.drain();
        let t0 = Instant::now();
        for _ in 0..ITERS {
            sgemv_q4_k_f32_lcpp_nsg2_into(backend, &x_buf, &w_buf, &y_buf, k, n).unwrap();
        }
        backend.drain();
        let dt = t0.elapsed();
        results.push((
            "sgemv_q4_k (SSM ssm_out)".to_string(),
            dt.as_secs_f64() * 1000.0,
            dt.as_secs_f64() * 1e6 / ITERS as f64,
            format!("K={k} N={n} Q4_K"),
            30,
        ));
    }

    // -------- Kernel 6: Q6_K LM head, K=2048 N=248320 --------
    {
        let k = 2048usize;
        let n = 248320usize;
        let w = fake_q6k_bytes(k, n);
        let x_buf = backend.alloc_shared(k * 4).unwrap();
        let w_buf = backend.alloc_shared(w.len()).unwrap();
        let y_buf = backend.alloc_shared(n * 4).unwrap();
        unsafe {
            std::ptr::copy_nonoverlapping(w.as_ptr(), w_buf.contents() as *mut u8, w.len());
        }
        sgemv_q6_k_f32_lcpp_nsg2_into(backend, &x_buf, &w_buf, &y_buf, k, n).unwrap();
        backend.drain();
        let t0 = Instant::now();
        for _ in 0..ITERS {
            sgemv_q6_k_f32_lcpp_nsg2_into(backend, &x_buf, &w_buf, &y_buf, k, n).unwrap();
        }
        backend.drain();
        let dt = t0.elapsed();
        results.push((
            "sgemv_q6_k (LM head)".to_string(),
            dt.as_secs_f64() * 1000.0,
            dt.as_secs_f64() * 1e6 / ITERS as f64,
            format!("K={k} N={n} Q6_K"),
            1,
        ));
    }

    // -------- Kernel 7: Q8_0 token embed sgemv (output Q8_0 path), K=2048 N=2048 --------
    {
        let k = 2048usize;
        let n = 2048usize;
        let w = fake_q8_0_bytes(k, n);
        let x_buf = backend.alloc_shared(k * 4).unwrap();
        let w_buf = backend.alloc_shared(w.len()).unwrap();
        let y_buf = backend.alloc_shared(n * 4).unwrap();
        unsafe {
            std::ptr::copy_nonoverlapping(w.as_ptr(), w_buf.contents() as *mut u8, w.len());
        }
        sgemv_q8_0_f32_lcpp_nsg2_into(backend, &x_buf, &w_buf, &y_buf, k, n).unwrap();
        backend.drain();
        let t0 = Instant::now();
        for _ in 0..ITERS {
            sgemv_q8_0_f32_lcpp_nsg2_into(backend, &x_buf, &w_buf, &y_buf, k, n).unwrap();
        }
        backend.drain();
        let dt = t0.elapsed();
        results.push((
            "sgemv_q8_0 (sample shape)".to_string(),
            dt.as_secs_f64() * 1000.0,
            dt.as_secs_f64() * 1e6 / ITERS as f64,
            format!("K={k} N={n} Q8_0"),
            10, // approx 10 attn layers worth
        ));
    }

    // -------- Kernel 8: rms_norm_f32, d=2048 --------
    {
        let d = 2048usize;
        let x = fake_f32(d, 5.7);
        let g = fake_f32(d, 1.1);
        let x_buf = backend.alloc_shared(d * 4).unwrap();
        let g_buf = backend.alloc_shared(d * 4).unwrap();
        let y_buf = backend.alloc_shared(d * 4).unwrap();
        unsafe {
            std::ptr::copy_nonoverlapping(x.as_ptr(), x_buf.contents() as *mut f32, d);
            std::ptr::copy_nonoverlapping(g.as_ptr(), g_buf.contents() as *mut f32, d);
        }
        rms_norm_f32(backend, &x_buf, &g_buf, &y_buf, d, 1e-6).unwrap();
        backend.drain();
        let t0 = Instant::now();
        for _ in 0..ITERS {
            rms_norm_f32(backend, &x_buf, &g_buf, &y_buf, d, 1e-6).unwrap();
        }
        backend.drain();
        let dt = t0.elapsed();
        results.push((
            "rms_norm_f32".to_string(),
            dt.as_secs_f64() * 1000.0,
            dt.as_secs_f64() * 1e6 / ITERS as f64,
            format!("d={d}"),
            80, // 40 layers × 2 norms (pre-attn/ssm + pre-ffn)
        ));
    }

    // -------- Kernel 9: ssm_apply_gate_f32, n_v=32 --------
    {
        let n_v = 32usize;
        let alpha = fake_f32(n_v, 1.3);
        let beta = fake_f32(n_v, 1.5);
        let dt_b = fake_f32(n_v, 1.7);
        let ssm_a = fake_f32(n_v, 1.9);
        let alpha_buf = backend.alloc_shared(n_v * 4).unwrap();
        let beta_buf = backend.alloc_shared(n_v * 4).unwrap();
        let dt_buf = backend.alloc_shared(n_v * 4).unwrap();
        let a_buf = backend.alloc_shared(n_v * 4).unwrap();
        let gh_buf = backend.alloc_shared(n_v * 4).unwrap();
        let bs_buf = backend.alloc_shared(n_v * 4).unwrap();
        unsafe {
            std::ptr::copy_nonoverlapping(alpha.as_ptr(), alpha_buf.contents() as *mut f32, n_v);
            std::ptr::copy_nonoverlapping(beta.as_ptr(), beta_buf.contents() as *mut f32, n_v);
            std::ptr::copy_nonoverlapping(dt_b.as_ptr(), dt_buf.contents() as *mut f32, n_v);
            std::ptr::copy_nonoverlapping(ssm_a.as_ptr(), a_buf.contents() as *mut f32, n_v);
        }
        ssm_apply_gate_f32(
            backend, &alpha_buf, &beta_buf, &dt_buf, &a_buf, &gh_buf, &bs_buf, n_v,
        )
        .unwrap();
        backend.drain();
        let t0 = Instant::now();
        for _ in 0..ITERS {
            ssm_apply_gate_f32(
                backend, &alpha_buf, &beta_buf, &dt_buf, &a_buf, &gh_buf, &bs_buf, n_v,
            )
            .unwrap();
        }
        backend.drain();
        let dt = t0.elapsed();
        results.push((
            "ssm_apply_gate_f32".to_string(),
            dt.as_secs_f64() * 1000.0,
            dt.as_secs_f64() * 1e6 / ITERS as f64,
            format!("n_v={n_v}"),
            30,
        ));
    }

    // -------- Kernel 10: delta_net_step_with_l2_f32, n_v=32, head_dim=128, n_k=16 --------
    {
        let n_v = 32usize;
        let head_dim = 128usize;
        let n_k = 16usize;
        let key_dim = n_k * head_dim;
        let value_dim = n_v * head_dim;
        let q = fake_f32(key_dim, 6.1);
        let k = fake_f32(key_dim, 6.3);
        let v = fake_f32(value_dim, 6.5);
        let gh = fake_f32(n_v, 1.3);
        let bs = fake_f32(n_v, 1.5);
        let state = fake_f32(n_v * head_dim * head_dim, 1.7);
        let q_buf = backend.alloc_shared(key_dim * 4).unwrap();
        let k_buf = backend.alloc_shared(key_dim * 4).unwrap();
        let v_buf = backend.alloc_shared(value_dim * 4).unwrap();
        let gh_buf = backend.alloc_shared(n_v * 4).unwrap();
        let bs_buf = backend.alloc_shared(n_v * 4).unwrap();
        let st_buf = backend.alloc_shared(state.len() * 4).unwrap();
        let out_buf = backend.alloc_shared(value_dim * 4).unwrap();
        unsafe {
            std::ptr::copy_nonoverlapping(q.as_ptr(), q_buf.contents() as *mut f32, key_dim);
            std::ptr::copy_nonoverlapping(k.as_ptr(), k_buf.contents() as *mut f32, key_dim);
            std::ptr::copy_nonoverlapping(v.as_ptr(), v_buf.contents() as *mut f32, value_dim);
            std::ptr::copy_nonoverlapping(gh.as_ptr(), gh_buf.contents() as *mut f32, n_v);
            std::ptr::copy_nonoverlapping(bs.as_ptr(), bs_buf.contents() as *mut f32, n_v);
            std::ptr::copy_nonoverlapping(
                state.as_ptr(),
                st_buf.contents() as *mut f32,
                state.len(),
            );
        }
        delta_net_step_with_l2_f32(
            backend, &q_buf, &k_buf, &v_buf, &gh_buf, &bs_buf, &st_buf, &out_buf, n_v, head_dim,
            n_k, 1e-6,
        )
        .unwrap();
        backend.drain();
        let t0 = Instant::now();
        for _ in 0..ITERS {
            delta_net_step_with_l2_f32(
                backend, &q_buf, &k_buf, &v_buf, &gh_buf, &bs_buf, &st_buf, &out_buf, n_v,
                head_dim, n_k, 1e-6,
            )
            .unwrap();
        }
        backend.drain();
        let dt = t0.elapsed();
        results.push((
            "delta_net_step_with_l2_f32".to_string(),
            dt.as_secs_f64() * 1000.0,
            dt.as_secs_f64() * 1e6 / ITERS as f64,
            format!("n_v={n_v} hd={head_dim} n_k={n_k}"),
            30,
        ));
    }

    // -------- Kernel 11: topk_softmax_norm_f32, n_experts=256, k=8 (CURRENT) --------
    let topk_n = 256usize;
    let topk_k = 8usize;
    let topk_logits = fake_f32(topk_n, 7.1);
    let topk_logits_buf = backend.alloc_shared(topk_n * 4).unwrap();
    let topk_idx_buf = backend.alloc_shared(topk_k * 4).unwrap();
    let topk_w_buf = backend.alloc_shared(topk_k * 4).unwrap();
    unsafe {
        std::ptr::copy_nonoverlapping(
            topk_logits.as_ptr(),
            topk_logits_buf.contents() as *mut f32,
            topk_n,
        );
    }
    {
        topk_softmax_norm_f32(
            backend,
            &topk_logits_buf,
            &topk_idx_buf,
            &topk_w_buf,
            topk_n,
            topk_k,
        )
        .unwrap();
        backend.drain();
        let t0 = Instant::now();
        for _ in 0..ITERS {
            topk_softmax_norm_f32(
                backend,
                &topk_logits_buf,
                &topk_idx_buf,
                &topk_w_buf,
                topk_n,
                topk_k,
            )
            .unwrap();
        }
        backend.drain();
        let dt = t0.elapsed();
        results.push((
            "topk_softmax_norm_f32 (CURRENT, serial)".to_string(),
            dt.as_secs_f64() * 1000.0,
            dt.as_secs_f64() * 1e6 / ITERS as f64,
            format!("n_experts={topk_n} k={topk_k}"),
            40,
        ));
    }

    // -------- Kernel 11b: topk_softmax_norm_parallel_f32, n_experts=256, k=8 (T182) --------
    {
        topk_softmax_norm_parallel_f32(
            backend,
            &topk_logits_buf,
            &topk_idx_buf,
            &topk_w_buf,
            topk_n,
            topk_k,
        )
        .unwrap();
        backend.drain();
        let t0 = Instant::now();
        for _ in 0..ITERS {
            topk_softmax_norm_parallel_f32(
                backend,
                &topk_logits_buf,
                &topk_idx_buf,
                &topk_w_buf,
                topk_n,
                topk_k,
            )
            .unwrap();
        }
        backend.drain();
        let dt = t0.elapsed();
        results.push((
            "topk_softmax_norm_PARALLEL_f32 (T182)".to_string(),
            dt.as_secs_f64() * 1000.0,
            dt.as_secs_f64() * 1e6 / ITERS as f64,
            format!("n_experts={topk_n} k={topk_k}"),
            40,
        ));
    }

    // -------- Parity check: parallel vs serial top-K must give same indices+weights --------
    {
        let mut idx_serial = vec![0u32; topk_k];
        let mut w_serial = vec![0.0_f32; topk_k];
        topk_softmax_norm_f32(
            backend,
            &topk_logits_buf,
            &topk_idx_buf,
            &topk_w_buf,
            topk_n,
            topk_k,
        )
        .unwrap();
        backend.drain();
        unsafe {
            std::ptr::copy_nonoverlapping(
                topk_idx_buf.contents() as *const u32,
                idx_serial.as_mut_ptr(),
                topk_k,
            );
            std::ptr::copy_nonoverlapping(
                topk_w_buf.contents() as *const f32,
                w_serial.as_mut_ptr(),
                topk_k,
            );
        }
        let mut idx_par = vec![0u32; topk_k];
        let mut w_par = vec![0.0_f32; topk_k];
        topk_softmax_norm_parallel_f32(
            backend,
            &topk_logits_buf,
            &topk_idx_buf,
            &topk_w_buf,
            topk_n,
            topk_k,
        )
        .unwrap();
        backend.drain();
        unsafe {
            std::ptr::copy_nonoverlapping(
                topk_idx_buf.contents() as *const u32,
                idx_par.as_mut_ptr(),
                topk_k,
            );
            std::ptr::copy_nonoverlapping(
                topk_w_buf.contents() as *const f32,
                w_par.as_mut_ptr(),
                topk_k,
            );
        }
        println!("\nParity check (T182 parallel vs serial topk):");
        println!("  serial   idx={idx_serial:?}");
        println!("           w  ={w_serial:?}");
        println!("  parallel idx={idx_par:?}");
        println!("           w  ={w_par:?}");
        let idx_match = idx_serial == idx_par;
        let w_match = idx_serial.iter().zip(idx_par.iter()).all(|(a, b)| a == b)
            && w_serial
                .iter()
                .zip(w_par.iter())
                .all(|(a, b)| (a - b).abs() < 1e-5);
        println!(
            "  parity: {}",
            if idx_match && w_match {
                "PASS ✓"
            } else {
                "FAIL ✗"
            }
        );
    }

    // -------- Kernel 12: swiglu_f32, f=512 (MoE expert FFN dim) --------
    {
        let f = 512usize;
        let g_buf = backend.alloc_shared(f * 4).unwrap();
        let u_buf = backend.alloc_shared(f * 4).unwrap();
        let y_buf = backend.alloc_shared(f * 4).unwrap();
        swiglu_f32(backend, &g_buf, &u_buf, &y_buf, f).unwrap();
        backend.drain();
        let t0 = Instant::now();
        for _ in 0..ITERS {
            swiglu_f32(backend, &g_buf, &u_buf, &y_buf, f).unwrap();
        }
        backend.drain();
        let dt = t0.elapsed();
        results.push((
            "swiglu_f32 (MoE expert)".to_string(),
            dt.as_secs_f64() * 1000.0,
            dt.as_secs_f64() * 1e6 / ITERS as f64,
            format!("f={f}"),
            320, // 40 layers × 8 experts (n_used)
        ));
    }

    // -------- Kernel 13: ssm_conv1d_step_f32, conv_dim=8192, kernel=4 --------
    {
        let conv_dim = 8192usize;
        let kernel = 4usize;
        let x_buf = backend.alloc_shared(conv_dim * 4).unwrap();
        let w_buf = backend.alloc_shared(conv_dim * kernel * 4).unwrap();
        let state_buf = backend.alloc_shared(conv_dim * kernel * 4).unwrap();
        let y_buf = backend.alloc_shared(conv_dim * 4).unwrap();
        ssm_conv1d_step_f32(
            backend, &x_buf, &w_buf, &state_buf, &y_buf, kernel, conv_dim,
        )
        .unwrap();
        backend.drain();
        let t0 = Instant::now();
        for _ in 0..ITERS {
            ssm_conv1d_step_f32(
                backend, &x_buf, &w_buf, &state_buf, &y_buf, kernel, conv_dim,
            )
            .unwrap();
        }
        backend.drain();
        let dt = t0.elapsed();
        results.push((
            "ssm_conv1d_step_f32".to_string(),
            dt.as_secs_f64() * 1000.0,
            dt.as_secs_f64() * 1e6 / ITERS as f64,
            format!("conv_dim={conv_dim} k={kernel}"),
            30,
        ));
    }

    // -------- Kernel 14: rms_norm_per_head_gated_f32, n=32, hd=128 --------
    {
        let n = 32usize;
        let hd = 128usize;
        let x_buf = backend.alloc_shared(n * hd * 4).unwrap();
        let g_buf = backend.alloc_shared(hd * 4).unwrap();
        let z_buf = backend.alloc_shared(n * hd * 4).unwrap();
        rms_norm_per_head_gated_f32(backend, &x_buf, &g_buf, &z_buf, n, hd, 1e-6).unwrap();
        backend.drain();
        let t0 = Instant::now();
        for _ in 0..ITERS {
            rms_norm_per_head_gated_f32(backend, &x_buf, &g_buf, &z_buf, n, hd, 1e-6).unwrap();
        }
        backend.drain();
        let dt = t0.elapsed();
        results.push((
            "rms_norm_per_head_gated_f32".to_string(),
            dt.as_secs_f64() * 1000.0,
            dt.as_secs_f64() * 1e6 / ITERS as f64,
            format!("n={n} hd={hd}"),
            30,
        ));
    }

    // -------- Kernel 15: split_qkv_f32, conv_dim=8192 split (2k+2k+4k) --------
    {
        let q_len = 2048usize;
        let k_len = 2048usize;
        let v_len = 4096usize;
        let total = q_len + k_len + v_len;
        let src_buf = backend.alloc_shared(total * 4).unwrap();
        let q_buf = backend.alloc_shared(q_len * 4).unwrap();
        let k_buf = backend.alloc_shared(k_len * 4).unwrap();
        let v_buf = backend.alloc_shared(v_len * 4).unwrap();
        split_qkv_f32(
            backend, &src_buf, &q_buf, &k_buf, &v_buf, q_len, k_len, v_len,
        )
        .unwrap();
        backend.drain();
        let t0 = Instant::now();
        for _ in 0..ITERS {
            split_qkv_f32(
                backend, &src_buf, &q_buf, &k_buf, &v_buf, q_len, k_len, v_len,
            )
            .unwrap();
        }
        backend.drain();
        let dt = t0.elapsed();
        results.push((
            "split_qkv_f32".to_string(),
            dt.as_secs_f64() * 1000.0,
            dt.as_secs_f64() * 1e6 / ITERS as f64,
            format!("q={q_len} k={k_len} v={v_len}"),
            30,
        ));
    }

    // -------- Kernel 16: sigmoid_add_moe_f32, d=2048 --------
    {
        let d = 2048usize;
        let acc = backend.alloc_shared(d * 4).unwrap();
        let sh = backend.alloc_shared(d * 4).unwrap();
        let sc = backend.alloc_shared(4).unwrap();
        let xd = backend.alloc_shared(d * 4).unwrap();
        sigmoid_add_moe_f32(backend, &acc, &sh, &sc, &xd, d).unwrap();
        backend.drain();
        let t0 = Instant::now();
        for _ in 0..ITERS {
            sigmoid_add_moe_f32(backend, &acc, &sh, &sc, &xd, d).unwrap();
        }
        backend.drain();
        let dt = t0.elapsed();
        results.push((
            "sigmoid_add_moe_f32".to_string(),
            dt.as_secs_f64() * 1000.0,
            dt.as_secs_f64() * 1e6 / ITERS as f64,
            format!("d={d}"),
            40,
        ));
    }

    // -------- Kernel 17: weighted_reduce_add_f32, b=8 d=2048 --------
    {
        let b = 8usize;
        let d = 2048usize;
        let src = backend.alloc_shared(b * d * 4).unwrap();
        let w = backend.alloc_shared(b * 4).unwrap();
        let acc = backend.alloc_shared(d * 4).unwrap();
        weighted_reduce_add_f32(backend, &src, &w, &acc, b, d).unwrap();
        backend.drain();
        let t0 = Instant::now();
        for _ in 0..ITERS {
            weighted_reduce_add_f32(backend, &src, &w, &acc, b, d).unwrap();
        }
        backend.drain();
        let dt = t0.elapsed();
        results.push((
            "weighted_reduce_add_f32".to_string(),
            dt.as_secs_f64() * 1000.0,
            dt.as_secs_f64() * 1e6 / ITERS as f64,
            format!("b={b} d={d}"),
            40,
        ));
    }

    // -------- Kernel 18: kv_append_f32, n_kv=2 head_dim=256 --------
    {
        let n_kv = 2usize;
        let hd = 256usize;
        let max_seq = 256usize;
        let src = backend.alloc_shared(n_kv * hd * 4).unwrap();
        let dst = backend.alloc_shared(n_kv * hd * max_seq * 4).unwrap();
        kv_append_f32(backend, &src, &dst, n_kv, hd, 0, max_seq).unwrap();
        backend.drain();
        let t0 = Instant::now();
        for pos in 0..ITERS {
            kv_append_f32(backend, &src, &dst, n_kv, hd, pos % max_seq, max_seq).unwrap();
        }
        backend.drain();
        let dt = t0.elapsed();
        results.push((
            "kv_append_f32".to_string(),
            dt.as_secs_f64() * 1000.0,
            dt.as_secs_f64() * 1e6 / ITERS as f64,
            format!("n_kv={n_kv} hd={hd}"),
            20, // 10 attn × 2 (k + v)
        ));
    }

    // -------- Kernel 19: rope_half_split_f32, n_heads=16 hd=256 rope_dim=64 --------
    {
        let n_heads = 16usize;
        let hd = 256usize;
        let rope_dim = 64usize;
        let x = backend.alloc_shared(n_heads * hd * 4).unwrap();
        let cos = backend.alloc_shared(rope_dim / 2 * 4).unwrap();
        let sin = backend.alloc_shared(rope_dim / 2 * 4).unwrap();
        rope_half_split_f32(backend, &x, &cos, &sin, n_heads, hd, rope_dim, 0).unwrap();
        backend.drain();
        let t0 = Instant::now();
        for pos in 0..ITERS {
            rope_half_split_f32(backend, &x, &cos, &sin, n_heads, hd, rope_dim, pos).unwrap();
        }
        backend.drain();
        let dt = t0.elapsed();
        results.push((
            "rope_half_split_f32".to_string(),
            dt.as_secs_f64() * 1000.0,
            dt.as_secs_f64() * 1e6 / ITERS as f64,
            format!("h={n_heads} hd={hd} rd={rope_dim}"),
            20, // 10 attn × 2 (q + k)
        ));
    }

    // -------- Kernel 20: gqa_decode_f32, n_q=16 n_kv=2 hd=256 kv_len=64 --------
    {
        let n_q = 16usize;
        let n_kv = 2usize;
        let hd = 256usize;
        let kv_len = 64usize;
        let max_seq = 256usize;
        let q = backend.alloc_shared(n_q * hd * 4).unwrap();
        let kc = backend.alloc_shared(n_kv * hd * max_seq * 4).unwrap();
        let vc = backend.alloc_shared(n_kv * hd * max_seq * 4).unwrap();
        let out = backend.alloc_shared(n_q * hd * 4).unwrap();
        gqa_decode_f32(backend, &q, &kc, &vc, &out, n_q, n_kv, hd, kv_len, max_seq).unwrap();
        backend.drain();
        let t0 = Instant::now();
        for _ in 0..ITERS {
            gqa_decode_f32(backend, &q, &kc, &vc, &out, n_q, n_kv, hd, kv_len, max_seq).unwrap();
        }
        backend.drain();
        let dt = t0.elapsed();
        results.push((
            "gqa_decode_f32".to_string(),
            dt.as_secs_f64() * 1000.0,
            dt.as_secs_f64() * 1e6 / ITERS as f64,
            format!("n_q={n_q} n_kv={n_kv} hd={hd} kv_len={kv_len}"),
            10,
        ));
    }

    // -------- Kernel 21: argmax_batched_f32, b=1 vocab=248320 --------
    {
        let vocab = 248320usize;
        let logits = backend.alloc_shared(vocab * 4).unwrap();
        let idx = backend.alloc_shared(4).unwrap();
        argmax_batched_f32(backend, &logits, &idx, 1, vocab).unwrap();
        backend.drain();
        let t0 = Instant::now();
        for _ in 0..ITERS {
            argmax_batched_f32(backend, &logits, &idx, 1, vocab).unwrap();
        }
        backend.drain();
        let dt = t0.elapsed();
        results.push((
            "argmax_batched_f32 (sample)".to_string(),
            dt.as_secs_f64() * 1000.0,
            dt.as_secs_f64() * 1e6 / ITERS as f64,
            format!("vocab={vocab}"),
            1,
        ));
    }

    // -------- Kernel 22: sigmoid_mul_inplace_f32, n=4096 (SSM gated) --------
    {
        let n = 4096usize;
        let x = backend.alloc_shared(n * 4).unwrap();
        let g = backend.alloc_shared(n * 4).unwrap();
        sigmoid_mul_inplace_f32(backend, &x, &g, n).unwrap();
        backend.drain();
        let t0 = Instant::now();
        for _ in 0..ITERS {
            sigmoid_mul_inplace_f32(backend, &x, &g, n).unwrap();
        }
        backend.drain();
        let dt = t0.elapsed();
        results.push((
            "sigmoid_mul_inplace_f32".to_string(),
            dt.as_secs_f64() * 1000.0,
            dt.as_secs_f64() * 1e6 / ITERS as f64,
            format!("n={n}"),
            40,
        ));
    }

    // -------- Kernel 23: add_inplace_f32, d=2048 (residual) --------
    {
        let d = 2048usize;
        let x = backend.alloc_shared(d * 4).unwrap();
        let y = backend.alloc_shared(d * 4).unwrap();
        add_inplace_f32(backend, &x, &y, d).unwrap();
        backend.drain();
        let t0 = Instant::now();
        for _ in 0..ITERS {
            add_inplace_f32(backend, &x, &y, d).unwrap();
        }
        backend.drain();
        let dt = t0.elapsed();
        results.push((
            "add_inplace_f32 (residual)".to_string(),
            dt.as_secs_f64() * 1000.0,
            dt.as_secs_f64() * 1e6 / ITERS as f64,
            format!("d={d}"),
            80, // 40 layers × 2 residuals
        ));
    }

    // -------- Print results --------
    println!();
    println!(
        "{:<35} {:>10} {:>14} {:>15} {:>26} {:>14}",
        "kernel", "iters", "total_ms", "avg µs/call", "shape", "calls/token → ms"
    );
    println!("{}", "-".repeat(120));
    let mut by_token: Vec<(String, f64, String, usize, f64)> = results
        .iter()
        .map(|(label, _total_ms, avg_us, shape, calls)| {
            let per_token_ms = (avg_us * *calls as f64) / 1000.0;
            (label.clone(), *avg_us, shape.clone(), *calls, per_token_ms)
        })
        .collect();
    by_token.sort_by(|a, b| b.4.partial_cmp(&a.4).unwrap());
    let total_ms_per_token: f64 = by_token.iter().map(|x| x.4).sum();
    for (label, avg_us, shape, _calls, per_token_ms) in &by_token {
        let pct = 100.0 * per_token_ms / total_ms_per_token;
        println!(
            "{:<35} {:>10} {:>14.3} {:>15.3} {:>26} {:>9.3} ({:>4.1}%)",
            label,
            ITERS,
            avg_us * ITERS as f64 / 1000.0,
            avg_us,
            shape,
            per_token_ms,
            pct
        );
    }
    println!("{}", "-".repeat(120));
    println!(
        "{:<35} {:>10} {:>14} {:>15} {:>26} {:>9.3}",
        "TOTAL (sum of measured kernels)", "", "", "", "", total_ms_per_token
    );
    println!(
        "(Decode wall-clock measured at 42 t/s = 23.8 ms/token. Unmeasured kernels = {:.1} ms/token.)",
        23.8 - total_ms_per_token
    );
}
