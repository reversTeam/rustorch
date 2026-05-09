//! `qwen_block_bench` — measure effective throughput on Qwen-shaped matmuls.
//!
//! Where the dense gemm bench reports synthetic peak TFLOPS at square
//! 4096²/16384² shapes, this one runs the **5 matmuls of one Qwen3.6
//! transformer block** at realistic dimensions and reports the projected
//! end-to-end token throughput. Useful to validate that our `LtSession`
//! integration scales to model-sized workloads.
//!
//! Per block (Qwen3.6-27B reference shapes — same hidden dim as the MoE
//! 35B-A3B variant, just dense FFN to keep the bench simple):
//!   - QKV proj : seq × hidden × (n_heads*head_dim + 2*n_kv*head_dim)
//!   - Attn out : seq × hidden × hidden
//!   - Up + Gate: 2 × seq × hidden × ffn_hidden  (parallel SwiGLU)
//!   - Down     : seq × ffn_hidden × hidden
//!
//! Default config (Qwen3.6-27B):
//!   hidden=2048, ffn=11008, n_heads=16, n_kv=2, head_dim=256, n_layers=40
//!   prompt seq=2048
//!
//! Reports per-precision (BF16 / FP8 / FP4 cached): per-block ms,
//! per-forward ms, effective prefill tok/s.

#[cfg(not(feature = "cuda"))]
fn main() {
    println!("[qwen_block_bench] cuda feature is OFF — nothing to bench.");
}

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
impl From<rustorch_cuda::error::CudaError> for BenchError {
    fn from(e: rustorch_cuda::error::CudaError) -> Self {
        Self(format!("cuda: {e:?}"))
    }
}

#[cfg(feature = "cuda")]
fn main() -> Result<(), BenchError> {
    use cudarc::driver::CudaContext;
    use rustorch_cuda::cublas_lt::LtSession;

    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let mut session = LtSession::new(stream.clone())?;
    println!(
        "[qwen_block_bench] cublasLt workspace: {} MiB (override via RUSTORCH_CUBLASLT_WORKSPACE_MB)",
        session.workspace_bytes / (1024 * 1024)
    );

    // Qwen3.6-27B-style dense block dims. Override via RUSTORCH_BENCH_MODEL=72b
    // pour Qwen-72B-like shapes (hidden=8192, ffn=29568) — beaucoup plus gros
    // matmuls qui devraient mieux utiliser les Tensor Cores GB10.
    let model = std::env::var("RUSTORCH_BENCH_MODEL").unwrap_or_else(|_| "27b".into());
    let (hidden, ffn, n_heads, n_kv, head_dim, n_layers) = match model.as_str() {
        "72b" => (8192usize, 29568usize, 64usize, 8usize, 128usize, 80usize),
        "qwen3-235b" | "235b" => (4096usize, 12288usize, 64usize, 4usize, 128usize, 94usize),
        _ => (2048usize, 11008usize, 16usize, 2usize, 256usize, 40usize),
    };
    println!("[qwen_block_bench] model={model}");
    // Override seq via RUSTORCH_BENCH_SEQ to test scaling (1024/2048/4096/8192).
    let seq = std::env::var("RUSTORCH_BENCH_SEQ")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(2048);

    let qkv_n = n_heads * head_dim + 2 * n_kv * head_dim; // 16*256 + 2*2*256 = 5120
    let attn_out_n = hidden;
    let ffn_up_n = ffn;
    let ffn_gate_n = ffn;
    let ffn_down_n = hidden;

    println!("[qwen_block_bench] Qwen3.6-27B dense reference shapes");
    println!("[qwen_block_bench]   hidden={hidden}  ffn={ffn}  n_heads={n_heads}  n_kv={n_kv}  head_dim={head_dim}");
    println!("[qwen_block_bench]   n_layers={n_layers}  prompt_seq={seq}");
    println!();

    // Per-block FLOPS:
    //   QKV       : 2 · seq · hidden · qkv_n
    //   Attn-out  : 2 · seq · hidden · hidden
    //   Up + Gate : 2 · 2 · seq · hidden · ffn
    //   Down      : 2 · seq · ffn · hidden
    let flops_qkv = 2.0 * seq as f64 * hidden as f64 * qkv_n as f64;
    let flops_attn_out = 2.0 * seq as f64 * hidden as f64 * hidden as f64;
    let flops_ffn_up = 2.0 * seq as f64 * hidden as f64 * ffn_up_n as f64;
    let flops_ffn_gate = 2.0 * seq as f64 * hidden as f64 * ffn_gate_n as f64;
    let flops_ffn_down = 2.0 * seq as f64 * ffn as f64 * ffn_down_n as f64;
    let flops_per_block =
        flops_qkv + flops_attn_out + flops_ffn_up + flops_ffn_gate + flops_ffn_down;
    let flops_per_forward = flops_per_block * n_layers as f64;
    println!(
        "[qwen_block_bench] FLOPS per block: {:.2} GFLOPS  ·  per forward: {:.2} TFLOPS",
        flops_per_block / 1e9,
        flops_per_forward / 1e12
    );
    println!();

    // Allocate inputs (same hidden activation [seq × hidden] for all blocks).
    let act_bf16 = vec![half::bf16::from_f32(0.01); seq * hidden];
    let act_dev_bf16 = stream.memcpy_stod(&act_bf16)?;

    // Weights — production trick (T240.8f): fuse FFN gate + up into a single
    // weight (hidden × 2·ffn), running them as ONE matmul instead of two.
    // Real Qwen / LLaMA / Mixtral all do this in inference.
    let ffn_gate_up_n = ffn_up_n + ffn_gate_n; // = 2 · ffn
    let w_qkv_bf16 = vec![half::bf16::from_f32(0.01); hidden * qkv_n];
    let w_attn_out_bf16 = vec![half::bf16::from_f32(0.01); hidden * attn_out_n];
    let w_ffn_gate_up_bf16 = vec![half::bf16::from_f32(0.01); hidden * ffn_gate_up_n];
    let w_ffn_down_bf16 = vec![half::bf16::from_f32(0.01); ffn * ffn_down_n];

    let w_qkv_dev = stream.memcpy_stod(&w_qkv_bf16)?;
    let w_attn_out_dev = stream.memcpy_stod(&w_attn_out_bf16)?;
    let w_ffn_gate_up_dev = stream.memcpy_stod(&w_ffn_gate_up_bf16)?;
    let w_ffn_down_dev = stream.memcpy_stod(&w_ffn_down_bf16)?;

    // Outputs (intermediate buffers).
    let mut out_qkv = stream.alloc_zeros::<half::bf16>(seq * qkv_n)?;
    let mut out_attn = stream.alloc_zeros::<half::bf16>(seq * attn_out_n)?;
    let mut out_gate_up = stream.alloc_zeros::<half::bf16>(seq * ffn_gate_up_n)?;
    let mut out_down = stream.alloc_zeros::<half::bf16>(seq * ffn_down_n)?;

    // ───────── BF16 path (T240.8f: fused gate+up) ─────────
    println!("[qwen_block_bench] BF16 path (TensorCore native, fused gate+up)");
    let bf16_per_block_ms = run_bf16_block(
        &mut session,
        &stream,
        &act_dev_bf16,
        &w_qkv_dev,
        &mut out_qkv,
        &w_attn_out_dev,
        &mut out_attn,
        &w_ffn_gate_up_dev,
        &mut out_gate_up,
        &w_ffn_down_dev,
        &mut out_down,
        seq,
        hidden,
        qkv_n,
        attn_out_n,
        ffn_gate_up_n,
        ffn_down_n,
        ffn,
        50,
    )?;
    let bf16_per_forward_ms = bf16_per_block_ms * n_layers as f64;
    let bf16_tok_s = (seq as f64 / bf16_per_forward_ms) * 1000.0;
    let bf16_tflops = flops_per_forward / 1e12 / (bf16_per_forward_ms / 1000.0);
    println!(
        "  per-block: {:.3} ms  |  per-forward (40 layers): {:.1} ms  |  prefill: {:.0} tok/s  |  effective {:.1} TFLOPS",
        bf16_per_block_ms, bf16_per_forward_ms, bf16_tok_s, bf16_tflops
    );

    // ───────── NVFP4 path (T240.8e) ─────────
    // GB10/sm_121 takes NVFP4 (VEC16 blocks, UE4M3 scales). Datacenter
    // Blackwell + Hopper want MXFP4 (VEC32 + UE8M0). UE4M3 byte 0x70
    // decodes to ≈1.0 → unscaled matmul (the byte values are dummies; the
    // bench measures kernel cost, not numerical output).
    use rustorch_cuda::cublas_lt::{Fp4ScaleMode, Fp8Output};
    let block = 16usize; // NVFP4 block size
    assert!(hidden % block == 0 && ffn % block == 0);

    // FP4 packed: 1 byte per 2 elements. Fused gate+up weight (T240.8f).
    let act_fp4: Vec<u8> = (0..seq * hidden / 2)
        .map(|i| (i as u8).wrapping_mul(11))
        .collect();
    let ffn_int_fp4: Vec<u8> = (0..seq * ffn / 2)
        .map(|i| (i as u8).wrapping_mul(13))
        .collect();
    let w_qkv_fp4: Vec<u8> = (0..hidden * qkv_n / 2)
        .map(|i| (i as u8).wrapping_mul(17))
        .collect();
    let w_attn_out_fp4: Vec<u8> = (0..hidden * attn_out_n / 2)
        .map(|i| (i as u8).wrapping_mul(19))
        .collect();
    // Fused gate+up: hidden × 2·ffn FP4 packed (1 byte = 2 elements)
    let w_ffn_gate_up_fp4: Vec<u8> = (0..hidden * ffn_gate_up_n / 2)
        .map(|i| (i as u8).wrapping_mul(23))
        .collect();
    let w_ffn_down_fp4: Vec<u8> = (0..ffn * ffn_down_n / 2)
        .map(|i| (i as u8).wrapping_mul(31))
        .collect();

    // Scales (UE4M3 byte 0x70 ≈ 1.0). M × (K/block_size) per FP4 matrix.
    let act_scale: Vec<u8> = vec![0x70u8; seq * (hidden / block)];
    let ffn_int_scale: Vec<u8> = vec![0x70u8; seq * (ffn / block)];
    let w_qkv_scale: Vec<u8> = vec![0x70u8; (hidden / block) * qkv_n];
    let w_attn_out_scale: Vec<u8> = vec![0x70u8; (hidden / block) * attn_out_n];
    let w_ffn_gate_up_scale: Vec<u8> = vec![0x70u8; (hidden / block) * ffn_gate_up_n];
    let w_ffn_down_scale: Vec<u8> = vec![0x70u8; (ffn / block) * ffn_down_n];

    let act_fp4_dev = stream.memcpy_stod(&act_fp4)?;
    let ffn_int_fp4_dev = stream.memcpy_stod(&ffn_int_fp4)?;
    let w_qkv_fp4_dev = stream.memcpy_stod(&w_qkv_fp4)?;
    let w_attn_out_fp4_dev = stream.memcpy_stod(&w_attn_out_fp4)?;
    let w_ffn_gate_up_fp4_dev = stream.memcpy_stod(&w_ffn_gate_up_fp4)?;
    let w_ffn_down_fp4_dev = stream.memcpy_stod(&w_ffn_down_fp4)?;

    let act_scale_dev = stream.memcpy_stod(&act_scale)?;
    let ffn_int_scale_dev = stream.memcpy_stod(&ffn_int_scale)?;
    let w_qkv_scale_dev = stream.memcpy_stod(&w_qkv_scale)?;
    let w_attn_out_scale_dev = stream.memcpy_stod(&w_attn_out_scale)?;
    let w_ffn_gate_up_scale_dev = stream.memcpy_stod(&w_ffn_gate_up_scale)?;
    let w_ffn_down_scale_dev = stream.memcpy_stod(&w_ffn_down_scale)?;

    println!();
    println!("[qwen_block_bench] NVFP4 path (Vec16Ue4m3, native on GB10, fused g+up)");

    let fp4_per_block_ms = run_fp4_block(
        &mut session,
        &stream,
        &act_fp4_dev,
        &act_scale_dev,
        &ffn_int_fp4_dev,
        &ffn_int_scale_dev,
        &w_qkv_fp4_dev,
        &w_qkv_scale_dev,
        &mut out_qkv,
        &w_attn_out_fp4_dev,
        &w_attn_out_scale_dev,
        &mut out_attn,
        &w_ffn_gate_up_fp4_dev,
        &w_ffn_gate_up_scale_dev,
        &mut out_gate_up,
        &w_ffn_down_fp4_dev,
        &w_ffn_down_scale_dev,
        &mut out_down,
        seq,
        hidden,
        qkv_n,
        attn_out_n,
        ffn_gate_up_n,
        ffn_down_n,
        ffn,
        Fp4ScaleMode::Vec16Ue4m3,
        Fp8Output::Bf16,
        50,
    )?;
    let fp4_per_forward_ms = fp4_per_block_ms * n_layers as f64;
    let fp4_tok_s = (seq as f64 / fp4_per_forward_ms) * 1000.0;
    let fp4_tflops = flops_per_forward / 1e12 / (fp4_per_forward_ms / 1000.0);
    println!(
        "  per-block: {:.3} ms  |  per-forward (40 layers): {:.1} ms  |  prefill: {:.0} tok/s  |  effective {:.1} TFLOPS",
        fp4_per_block_ms, fp4_per_forward_ms, fp4_tok_s, fp4_tflops
    );

    println!();
    println!(
        "[qwen_block_bench] BF16 → NVFP4 speedup: {:.2}× ({:.1} → {:.1} TFLOPS)",
        bf16_per_block_ms / fp4_per_block_ms,
        bf16_tflops,
        fp4_tflops
    );
    println!(
        "[qwen_block_bench] gap to GB10 FP4-dense ceiling 427 TFLOPS: {:.1}%",
        100.0 * fp4_tflops / 427.0
    );
    println!(
        "[qwen_block_bench] gap to advertised 1000 TOPS (FP4 sparse): {:.1}%",
        100.0 * fp4_tflops / 1000.0
    );

    // ───────── NVFP4 + CUDA Graph (T240.8k) ─────────
    // Capture les 4 matmuls FP4 en un CUDA Graph et replay N fois.
    // Élimine totalement l'overhead de kernel launch (~5-10 µs × 4 matmuls
    // × 50 iters = 1-2 ms qu'on peut récupérer). Sur les workloads FP4 où
    // chaque matmul prend ~1 ms, ce sont des % gros à grappiller.
    let skip_graph = std::env::var("RUSTORCH_SKIP_GRAPH")
        .map(|v| v == "1")
        .unwrap_or(false);
    if !skip_graph {
        println!();
        println!("[qwen_block_bench] NVFP4 + CUDA Graph (T240.8k)");
        match try_run_fp4_graph(
            &ctx,
            &act_fp4_dev,
            &act_scale_dev,
            &ffn_int_fp4_dev,
            &ffn_int_scale_dev,
            &w_qkv_fp4_dev,
            &w_qkv_scale_dev,
            &w_attn_out_fp4_dev,
            &w_attn_out_scale_dev,
            &w_ffn_gate_up_fp4_dev,
            &w_ffn_gate_up_scale_dev,
            &w_ffn_down_fp4_dev,
            &w_ffn_down_scale_dev,
            seq,
            hidden,
            qkv_n,
            attn_out_n,
            ffn_gate_up_n,
            ffn_down_n,
            ffn,
            Fp4ScaleMode::Vec16Ue4m3,
            Fp8Output::Bf16,
            50,
        ) {
            Ok(graph_per_block_ms) => {
                let g_per_forward = graph_per_block_ms * n_layers as f64;
                let g_tok_s = (seq as f64 / g_per_forward) * 1000.0;
                let g_tflops = flops_per_forward / 1e12 / (g_per_forward / 1000.0);
                println!(
                    "  per-block: {:.3} ms  |  per-forward (40 layers): {:.1} ms  |  prefill: {:.0} tok/s  |  effective {:.1} TFLOPS",
                    graph_per_block_ms, g_per_forward, g_tok_s, g_tflops
                );
                println!(
                    "[qwen_block_bench] FP4 → FP4+graph speedup: {:.2}× ({:.1} → {:.1} TFLOPS)",
                    fp4_per_block_ms / graph_per_block_ms,
                    fp4_tflops,
                    g_tflops
                );
                println!(
                    "[qwen_block_bench] gap to GB10 FP4-dense ceiling 427 TFLOPS: {:.1}%",
                    100.0 * g_tflops / 427.0
                );
                println!(
                    "[qwen_block_bench] gap to advertised 1000 TOPS: {:.1}%",
                    100.0 * g_tflops / 1000.0
                );
            },
            Err(e) => println!("  skipped: {}", e.0),
        }
    }

    // ───────── NVFP4 multi-stream (T240.8i) ─────────
    // Lancer N forwards concurrents sur N streams pour exploiter mieux les
    // SMs (workload = continuous batching prod). Throughput agrégé devrait
    // multiplier par ~N×0.6-0.8 selon l'overlap.
    let n_streams: usize = std::env::var("RUSTORCH_NSTREAMS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    if n_streams >= 2 {
        println!();
        println!("[qwen_block_bench] NVFP4 multi-stream ({n_streams} parallel forwards)");
        // Streams supplémentaires (le default stream est déjà compté)
        let mut streams: Vec<std::sync::Arc<cudarc::driver::CudaStream>> = vec![stream.clone()];
        for _ in 1..n_streams {
            streams.push(ctx.new_stream()?);
        }

        // Une session distincte par stream (les sessions cublasLt sont liées
        // à un stream pour leur workspace). Réutilise l'autotune-mxfp4 du
        // chemin single-stream.
        let mut sessions: Vec<rustorch_cuda::cublas_lt::LtSession> = Vec::with_capacity(n_streams);
        for s in &streams {
            sessions.push(rustorch_cuda::cublas_lt::LtSession::new(s.clone())?);
        }

        // Output buffers par stream pour éviter les conflits write-after-write
        let mut out_qkv_v: Vec<cudarc::driver::CudaSlice<half::bf16>> =
            Vec::with_capacity(n_streams);
        let mut out_attn_v: Vec<cudarc::driver::CudaSlice<half::bf16>> =
            Vec::with_capacity(n_streams);
        let mut out_gu_v: Vec<cudarc::driver::CudaSlice<half::bf16>> =
            Vec::with_capacity(n_streams);
        let mut out_dn_v: Vec<cudarc::driver::CudaSlice<half::bf16>> =
            Vec::with_capacity(n_streams);
        for s in &streams {
            out_qkv_v.push(s.alloc_zeros::<half::bf16>(seq * qkv_n)?);
            out_attn_v.push(s.alloc_zeros::<half::bf16>(seq * attn_out_n)?);
            out_gu_v.push(s.alloc_zeros::<half::bf16>(seq * ffn_gate_up_n)?);
            out_dn_v.push(s.alloc_zeros::<half::bf16>(seq * ffn_down_n)?);
        }

        // Warm-up : 3 itérations sur chaque stream (pour caching cublasLt)
        for _ in 0..3 {
            for i in 0..n_streams {
                let s = &streams[i];
                let session = &mut sessions[i];
                unsafe {
                    use cudarc::driver::{DevicePtr, DevicePtrMut};
                    {
                        let (a_p, _r1) = act_fp4_dev.device_ptr(s);
                        let (sa_p, _r2) = act_scale_dev.device_ptr(s);
                        let (b_p, _r3) = w_qkv_fp4_dev.device_ptr(s);
                        let (sb_p, _r4) = w_qkv_scale_dev.device_ptr(s);
                        let (c_p, _r5) = out_qkv_v[i].device_ptr_mut(s);
                        session.matmul_mxfp4(
                            a_p,
                            sa_p,
                            b_p,
                            sb_p,
                            c_p,
                            seq,
                            hidden,
                            qkv_n,
                            1.0,
                            0.0,
                            Fp8Output::Bf16,
                            Fp4ScaleMode::Vec16Ue4m3,
                        )?;
                    }
                    {
                        let (a_p, _r1) = act_fp4_dev.device_ptr(s);
                        let (sa_p, _r2) = act_scale_dev.device_ptr(s);
                        let (b_p, _r3) = w_attn_out_fp4_dev.device_ptr(s);
                        let (sb_p, _r4) = w_attn_out_scale_dev.device_ptr(s);
                        let (c_p, _r5) = out_attn_v[i].device_ptr_mut(s);
                        session.matmul_mxfp4(
                            a_p,
                            sa_p,
                            b_p,
                            sb_p,
                            c_p,
                            seq,
                            hidden,
                            attn_out_n,
                            1.0,
                            0.0,
                            Fp8Output::Bf16,
                            Fp4ScaleMode::Vec16Ue4m3,
                        )?;
                    }
                    {
                        let (a_p, _r1) = act_fp4_dev.device_ptr(s);
                        let (sa_p, _r2) = act_scale_dev.device_ptr(s);
                        let (b_p, _r3) = w_ffn_gate_up_fp4_dev.device_ptr(s);
                        let (sb_p, _r4) = w_ffn_gate_up_scale_dev.device_ptr(s);
                        let (c_p, _r5) = out_gu_v[i].device_ptr_mut(s);
                        session.matmul_mxfp4(
                            a_p,
                            sa_p,
                            b_p,
                            sb_p,
                            c_p,
                            seq,
                            hidden,
                            ffn_gate_up_n,
                            1.0,
                            0.0,
                            Fp8Output::Bf16,
                            Fp4ScaleMode::Vec16Ue4m3,
                        )?;
                    }
                    {
                        let (a_p, _r1) = ffn_int_fp4_dev.device_ptr(s);
                        let (sa_p, _r2) = ffn_int_scale_dev.device_ptr(s);
                        let (b_p, _r3) = w_ffn_down_fp4_dev.device_ptr(s);
                        let (sb_p, _r4) = w_ffn_down_scale_dev.device_ptr(s);
                        let (c_p, _r5) = out_dn_v[i].device_ptr_mut(s);
                        session.matmul_mxfp4(
                            a_p,
                            sa_p,
                            b_p,
                            sb_p,
                            c_p,
                            seq,
                            ffn,
                            ffn_down_n,
                            1.0,
                            0.0,
                            Fp8Output::Bf16,
                            Fp4ScaleMode::Vec16Ue4m3,
                        )?;
                    }
                }
            }
        }
        for s in &streams {
            s.synchronize()?;
        }

        // Bench timed : 50 itérations × n_streams en parallèle
        let iters = 50usize;
        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            for i in 0..n_streams {
                let s = &streams[i];
                let session = &mut sessions[i];
                unsafe {
                    use cudarc::driver::{DevicePtr, DevicePtrMut};
                    {
                        let (a_p, _r1) = act_fp4_dev.device_ptr(s);
                        let (sa_p, _r2) = act_scale_dev.device_ptr(s);
                        let (b_p, _r3) = w_qkv_fp4_dev.device_ptr(s);
                        let (sb_p, _r4) = w_qkv_scale_dev.device_ptr(s);
                        let (c_p, _r5) = out_qkv_v[i].device_ptr_mut(s);
                        session.matmul_mxfp4(
                            a_p,
                            sa_p,
                            b_p,
                            sb_p,
                            c_p,
                            seq,
                            hidden,
                            qkv_n,
                            1.0,
                            0.0,
                            Fp8Output::Bf16,
                            Fp4ScaleMode::Vec16Ue4m3,
                        )?;
                    }
                    {
                        let (a_p, _r1) = act_fp4_dev.device_ptr(s);
                        let (sa_p, _r2) = act_scale_dev.device_ptr(s);
                        let (b_p, _r3) = w_attn_out_fp4_dev.device_ptr(s);
                        let (sb_p, _r4) = w_attn_out_scale_dev.device_ptr(s);
                        let (c_p, _r5) = out_attn_v[i].device_ptr_mut(s);
                        session.matmul_mxfp4(
                            a_p,
                            sa_p,
                            b_p,
                            sb_p,
                            c_p,
                            seq,
                            hidden,
                            attn_out_n,
                            1.0,
                            0.0,
                            Fp8Output::Bf16,
                            Fp4ScaleMode::Vec16Ue4m3,
                        )?;
                    }
                    {
                        let (a_p, _r1) = act_fp4_dev.device_ptr(s);
                        let (sa_p, _r2) = act_scale_dev.device_ptr(s);
                        let (b_p, _r3) = w_ffn_gate_up_fp4_dev.device_ptr(s);
                        let (sb_p, _r4) = w_ffn_gate_up_scale_dev.device_ptr(s);
                        let (c_p, _r5) = out_gu_v[i].device_ptr_mut(s);
                        session.matmul_mxfp4(
                            a_p,
                            sa_p,
                            b_p,
                            sb_p,
                            c_p,
                            seq,
                            hidden,
                            ffn_gate_up_n,
                            1.0,
                            0.0,
                            Fp8Output::Bf16,
                            Fp4ScaleMode::Vec16Ue4m3,
                        )?;
                    }
                    {
                        let (a_p, _r1) = ffn_int_fp4_dev.device_ptr(s);
                        let (sa_p, _r2) = ffn_int_scale_dev.device_ptr(s);
                        let (b_p, _r3) = w_ffn_down_fp4_dev.device_ptr(s);
                        let (sb_p, _r4) = w_ffn_down_scale_dev.device_ptr(s);
                        let (c_p, _r5) = out_dn_v[i].device_ptr_mut(s);
                        session.matmul_mxfp4(
                            a_p,
                            sa_p,
                            b_p,
                            sb_p,
                            c_p,
                            seq,
                            ffn,
                            ffn_down_n,
                            1.0,
                            0.0,
                            Fp8Output::Bf16,
                            Fp4ScaleMode::Vec16Ue4m3,
                        )?;
                    }
                }
            }
        }
        for s in &streams {
            s.synchronize()?;
        }
        let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;

        // Throughput agrégé : iters × n_streams forwards complets
        let total_forwards = iters * n_streams;
        let per_block_ms = elapsed_ms / iters as f64; // wall time pour 1 itération de N forwards
        let per_forward_ms = per_block_ms * n_layers as f64 / n_streams as f64;
        let agg_tok_s = (seq as f64 * total_forwards as f64 / elapsed_ms) * 1000.0;
        let agg_tflops = flops_per_forward * n_streams as f64
            / 1e12
            / ((per_block_ms * n_layers as f64) / 1000.0);
        println!(
            "  per-block (wall, N parallel): {:.3} ms  |  agg prefill: {:.0} tok/s  |  effective {:.1} TFLOPS",
            per_block_ms, agg_tok_s, agg_tflops
        );
        println!(
            "[qwen_block_bench] single → multi-stream {}× speedup: {:.2}× ({:.1} → {:.1} TFLOPS)",
            n_streams,
            agg_tflops / fp4_tflops,
            fp4_tflops,
            agg_tflops
        );
        println!(
            "[qwen_block_bench] gap to advertised 1000 TOPS: {:.1}%",
            100.0 * agg_tflops / 1000.0
        );
    }

    // ───────── NVFP4 + 2:4 sparsity path (T240.8h) ─────────
    // FP4 inputs avec poids 2:4 + activation dense FP4. cuSPARSELt 0.9
    // supporte CUDA_R_4F_E2M1 + scales VEC32_UE4M3. Note bloc=32 vs
    // VEC16 du chemin cublasLt dense.
    let skip_sparse_fp4 = std::env::var("RUSTORCH_SKIP_SPARSE_FP4")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    if !skip_sparse_fp4 {
        println!();
        println!("[qwen_block_bench] NVFP4 + 2:4 sparsity path (cuSPARSELt FP4)");
        // Pre-allocate FP4 act/weight scales en VEC32 (bloc 32 pour cuSPARSELt FP4)
        let block32 = 32usize;
        let act_scale32: Vec<u8> = vec![0x70u8; seq * (hidden / block32)];
        let w_qkv_scale32: Vec<u8> = vec![0x70u8; (hidden / block32) * qkv_n];
        let w_attn_out_scale32: Vec<u8> = vec![0x70u8; (hidden / block32) * attn_out_n];
        let w_ffn_gate_up_scale32: Vec<u8> = vec![0x70u8; (hidden / block32) * ffn_gate_up_n];
        let w_ffn_down_scale32: Vec<u8> = vec![0x70u8; (ffn / block32) * ffn_down_n];
        let ffn_int_scale32: Vec<u8> = vec![0x70u8; seq * (ffn / block32)];

        let act_scale32_dev = stream.memcpy_stod(&act_scale32)?;
        let w_qkv_scale32_dev = stream.memcpy_stod(&w_qkv_scale32)?;
        let w_attn_out_scale32_dev = stream.memcpy_stod(&w_attn_out_scale32)?;
        let w_ffn_gate_up_scale32_dev = stream.memcpy_stod(&w_ffn_gate_up_scale32)?;
        let w_ffn_down_scale32_dev = stream.memcpy_stod(&w_ffn_down_scale32)?;
        let ffn_int_scale32_dev = stream.memcpy_stod(&ffn_int_scale32)?;

        match try_run_sparse_fp4_block(
            &stream,
            &act_fp4_dev,
            &act_scale32_dev,
            &ffn_int_fp4_dev,
            &ffn_int_scale32_dev,
            &w_qkv_fp4_dev,
            &w_qkv_scale32_dev,
            &mut out_qkv,
            &w_attn_out_fp4_dev,
            &w_attn_out_scale32_dev,
            &mut out_attn,
            &w_ffn_gate_up_fp4_dev,
            &w_ffn_gate_up_scale32_dev,
            &mut out_gate_up,
            &w_ffn_down_fp4_dev,
            &w_ffn_down_scale32_dev,
            &mut out_down,
            seq,
            hidden,
            qkv_n,
            attn_out_n,
            ffn_gate_up_n,
            ffn_down_n,
            ffn,
            50,
        ) {
            Ok(sp_fp4_per_block_ms) => {
                let sp_fp4_per_forward_ms = sp_fp4_per_block_ms * n_layers as f64;
                let sp_fp4_tok_s = (seq as f64 / sp_fp4_per_forward_ms) * 1000.0;
                let sp_fp4_tflops = flops_per_forward / 1e12 / (sp_fp4_per_forward_ms / 1000.0);
                println!(
                    "  per-block: {:.3} ms  |  per-forward (40 layers): {:.1} ms  |  prefill: {:.0} tok/s  |  effective {:.1} TFLOPS",
                    sp_fp4_per_block_ms, sp_fp4_per_forward_ms, sp_fp4_tok_s, sp_fp4_tflops
                );
                println!(
                    "[qwen_block_bench] NVFP4 → NVFP4+2:4 speedup: {:.2}× ({:.1} → {:.1} TFLOPS)",
                    fp4_per_block_ms / sp_fp4_per_block_ms,
                    fp4_tflops,
                    sp_fp4_tflops
                );
                println!(
                    "[qwen_block_bench] gap to advertised 1000 TOPS (FP4 sparse): {:.1}%",
                    100.0 * sp_fp4_tflops / 1000.0
                );
            },
            Err(e) => {
                println!("  skipped: {}", e.0);
            },
        }
    }

    // ───────── BF16 + 2:4 sparsity path (T240.8c) ─────────
    // cuSPARSELt 2:4 structured-sparse · dense matmul. Operates on BF16
    // weights pruned to 2:4 + compressed; activation stays dense BF16.
    // Community-measured gain on GB10: ~1.79× vs dense BF16 at large M.
    // Skip if RUSTORCH_SKIP_SPARSE=1 (e.g. host without cuSPARSELt installed).
    let skip_sparse = std::env::var("RUSTORCH_SKIP_SPARSE")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    if !skip_sparse {
        println!();
        println!("[qwen_block_bench] BF16 + 2:4 sparsity path (cuSPARSELt)");
        match try_run_sparse_block(
            &stream,
            &act_dev_bf16,
            &w_qkv_dev,
            &mut out_qkv,
            &w_attn_out_dev,
            &mut out_attn,
            &w_ffn_gate_up_dev,
            &mut out_gate_up,
            &w_ffn_down_dev,
            &mut out_down,
            seq,
            hidden,
            qkv_n,
            attn_out_n,
            ffn_gate_up_n,
            ffn_down_n,
            ffn,
            50,
        ) {
            Ok(sparse_per_block_ms) => {
                let sparse_per_forward_ms = sparse_per_block_ms * n_layers as f64;
                let sparse_tok_s = (seq as f64 / sparse_per_forward_ms) * 1000.0;
                let sparse_tflops = flops_per_forward / 1e12 / (sparse_per_forward_ms / 1000.0);
                println!(
                    "  per-block: {:.3} ms  |  per-forward (40 layers): {:.1} ms  |  prefill: {:.0} tok/s  |  effective {:.1} TFLOPS",
                    sparse_per_block_ms, sparse_per_forward_ms, sparse_tok_s, sparse_tflops
                );
                println!(
                    "[qwen_block_bench] BF16 → BF16+2:4 speedup: {:.2}× ({:.1} → {:.1} TFLOPS)",
                    bf16_per_block_ms / sparse_per_block_ms,
                    bf16_tflops,
                    sparse_tflops
                );
                println!(
                    "[qwen_block_bench] gap to advertised 1000 TOPS: {:.1}%",
                    100.0 * sparse_tflops / 1000.0
                );
            },
            Err(e) => {
                println!("  skipped: {}", e.0);
            },
        }
    }

    Ok(())
}

#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn try_run_fp4_graph(
    ctx: &std::sync::Arc<cudarc::driver::CudaContext>,
    act_fp4: &cudarc::driver::CudaSlice<u8>,
    act_scale: &cudarc::driver::CudaSlice<u8>,
    ffn_int_fp4: &cudarc::driver::CudaSlice<u8>,
    ffn_int_scale: &cudarc::driver::CudaSlice<u8>,
    w_qkv: &cudarc::driver::CudaSlice<u8>,
    w_qkv_scale: &cudarc::driver::CudaSlice<u8>,
    w_attn_out: &cudarc::driver::CudaSlice<u8>,
    w_attn_out_scale: &cudarc::driver::CudaSlice<u8>,
    w_ffn_gate_up: &cudarc::driver::CudaSlice<u8>,
    w_ffn_gate_up_scale: &cudarc::driver::CudaSlice<u8>,
    w_ffn_down: &cudarc::driver::CudaSlice<u8>,
    w_ffn_down_scale: &cudarc::driver::CudaSlice<u8>,
    seq: usize,
    hidden: usize,
    qkv_n: usize,
    attn_out_n: usize,
    ffn_gate_up_n: usize,
    ffn_down_n: usize,
    ffn: usize,
    scale_mode: rustorch_cuda::cublas_lt::Fp4ScaleMode,
    out_dtype: rustorch_cuda::cublas_lt::Fp8Output,
    iters: usize,
) -> Result<f64, BenchError> {
    use cudarc::driver::sys::{CUgraphInstantiate_flags, CUstreamCaptureMode};
    use cudarc::driver::{DevicePtr, DevicePtrMut};
    use rustorch_cuda::cublas_lt::LtSession;
    use std::time::Instant;

    // Stream dédié pour la capture (le default stream peut ne pas l'accepter).
    let cap_stream = ctx.new_stream()?;
    let mut session = LtSession::new(cap_stream.clone())?;

    // Output buffers locaux (pour ne pas interférer avec les paths déjà mesurés).
    let mut out_qkv = cap_stream.alloc_zeros::<half::bf16>(seq * qkv_n)?;
    let mut out_attn = cap_stream.alloc_zeros::<half::bf16>(seq * attn_out_n)?;
    let mut out_gu = cap_stream.alloc_zeros::<half::bf16>(seq * ffn_gate_up_n)?;
    let mut out_dn = cap_stream.alloc_zeros::<half::bf16>(seq * ffn_down_n)?;

    // Warm-up : construit le cache cublasLt avant la capture (sinon le
    // premier call alloue/builde des descripteurs qui ne sont pas
    // capturables proprement).
    for _ in 0..3 {
        unsafe {
            {
                let (a_p, _r1) = act_fp4.device_ptr(&cap_stream);
                let (sa_p, _r2) = act_scale.device_ptr(&cap_stream);
                let (b_p, _r3) = w_qkv.device_ptr(&cap_stream);
                let (sb_p, _r4) = w_qkv_scale.device_ptr(&cap_stream);
                let (c_p, _r5) = out_qkv.device_ptr_mut(&cap_stream);
                session.matmul_mxfp4(
                    a_p, sa_p, b_p, sb_p, c_p, seq, hidden, qkv_n, 1.0, 0.0, out_dtype, scale_mode,
                )?;
            }
            {
                let (a_p, _r1) = act_fp4.device_ptr(&cap_stream);
                let (sa_p, _r2) = act_scale.device_ptr(&cap_stream);
                let (b_p, _r3) = w_attn_out.device_ptr(&cap_stream);
                let (sb_p, _r4) = w_attn_out_scale.device_ptr(&cap_stream);
                let (c_p, _r5) = out_attn.device_ptr_mut(&cap_stream);
                session.matmul_mxfp4(
                    a_p, sa_p, b_p, sb_p, c_p, seq, hidden, attn_out_n, 1.0, 0.0, out_dtype,
                    scale_mode,
                )?;
            }
            {
                let (a_p, _r1) = act_fp4.device_ptr(&cap_stream);
                let (sa_p, _r2) = act_scale.device_ptr(&cap_stream);
                let (b_p, _r3) = w_ffn_gate_up.device_ptr(&cap_stream);
                let (sb_p, _r4) = w_ffn_gate_up_scale.device_ptr(&cap_stream);
                let (c_p, _r5) = out_gu.device_ptr_mut(&cap_stream);
                session.matmul_mxfp4(
                    a_p,
                    sa_p,
                    b_p,
                    sb_p,
                    c_p,
                    seq,
                    hidden,
                    ffn_gate_up_n,
                    1.0,
                    0.0,
                    out_dtype,
                    scale_mode,
                )?;
            }
            {
                let (a_p, _r1) = ffn_int_fp4.device_ptr(&cap_stream);
                let (sa_p, _r2) = ffn_int_scale.device_ptr(&cap_stream);
                let (b_p, _r3) = w_ffn_down.device_ptr(&cap_stream);
                let (sb_p, _r4) = w_ffn_down_scale.device_ptr(&cap_stream);
                let (c_p, _r5) = out_dn.device_ptr_mut(&cap_stream);
                session.matmul_mxfp4(
                    a_p, sa_p, b_p, sb_p, c_p, seq, ffn, ffn_down_n, 1.0, 0.0, out_dtype,
                    scale_mode,
                )?;
            }
        }
    }
    cap_stream.synchronize()?;

    // Capture de UN bloc (4 matmuls). On replay ce bloc 50 fois.
    cap_stream
        .begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)
        .map_err(|e| BenchError(format!("begin_capture: {e:?}")))?;
    unsafe {
        {
            let (a_p, _r1) = act_fp4.device_ptr(&cap_stream);
            let (sa_p, _r2) = act_scale.device_ptr(&cap_stream);
            let (b_p, _r3) = w_qkv.device_ptr(&cap_stream);
            let (sb_p, _r4) = w_qkv_scale.device_ptr(&cap_stream);
            let (c_p, _r5) = out_qkv.device_ptr_mut(&cap_stream);
            session.matmul_mxfp4(
                a_p, sa_p, b_p, sb_p, c_p, seq, hidden, qkv_n, 1.0, 0.0, out_dtype, scale_mode,
            )?;
        }
        {
            let (a_p, _r1) = act_fp4.device_ptr(&cap_stream);
            let (sa_p, _r2) = act_scale.device_ptr(&cap_stream);
            let (b_p, _r3) = w_attn_out.device_ptr(&cap_stream);
            let (sb_p, _r4) = w_attn_out_scale.device_ptr(&cap_stream);
            let (c_p, _r5) = out_attn.device_ptr_mut(&cap_stream);
            session.matmul_mxfp4(
                a_p, sa_p, b_p, sb_p, c_p, seq, hidden, attn_out_n, 1.0, 0.0, out_dtype, scale_mode,
            )?;
        }
        {
            let (a_p, _r1) = act_fp4.device_ptr(&cap_stream);
            let (sa_p, _r2) = act_scale.device_ptr(&cap_stream);
            let (b_p, _r3) = w_ffn_gate_up.device_ptr(&cap_stream);
            let (sb_p, _r4) = w_ffn_gate_up_scale.device_ptr(&cap_stream);
            let (c_p, _r5) = out_gu.device_ptr_mut(&cap_stream);
            session.matmul_mxfp4(
                a_p,
                sa_p,
                b_p,
                sb_p,
                c_p,
                seq,
                hidden,
                ffn_gate_up_n,
                1.0,
                0.0,
                out_dtype,
                scale_mode,
            )?;
        }
        {
            let (a_p, _r1) = ffn_int_fp4.device_ptr(&cap_stream);
            let (sa_p, _r2) = ffn_int_scale.device_ptr(&cap_stream);
            let (b_p, _r3) = w_ffn_down.device_ptr(&cap_stream);
            let (sb_p, _r4) = w_ffn_down_scale.device_ptr(&cap_stream);
            let (c_p, _r5) = out_dn.device_ptr_mut(&cap_stream);
            session.matmul_mxfp4(
                a_p, sa_p, b_p, sb_p, c_p, seq, ffn, ffn_down_n, 1.0, 0.0, out_dtype, scale_mode,
            )?;
        }
    }
    let graph = cap_stream
        .end_capture(CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH)
        .map_err(|e| BenchError(format!("end_capture: {e:?}")))?
        .ok_or_else(|| BenchError("graph empty".into()))?;

    // Replay timed
    let t0 = Instant::now();
    for _ in 0..iters {
        graph
            .launch()
            .map_err(|e| BenchError(format!("graph_launch: {e:?}")))?;
    }
    cap_stream.synchronize()?;
    let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
    Ok(elapsed_ms / iters as f64)
}

#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn try_run_sparse_fp4_block(
    stream: &std::sync::Arc<cudarc::driver::CudaStream>,
    act_fp4: &cudarc::driver::CudaSlice<u8>,
    act_scale: &cudarc::driver::CudaSlice<u8>,
    ffn_int_fp4: &cudarc::driver::CudaSlice<u8>,
    ffn_int_scale: &cudarc::driver::CudaSlice<u8>,
    w_qkv: &cudarc::driver::CudaSlice<u8>,
    w_qkv_scale: &cudarc::driver::CudaSlice<u8>,
    out_qkv: &mut cudarc::driver::CudaSlice<half::bf16>,
    w_attn_out: &cudarc::driver::CudaSlice<u8>,
    w_attn_out_scale: &cudarc::driver::CudaSlice<u8>,
    out_attn: &mut cudarc::driver::CudaSlice<half::bf16>,
    w_ffn_gate_up: &cudarc::driver::CudaSlice<u8>,
    w_ffn_gate_up_scale: &cudarc::driver::CudaSlice<u8>,
    out_gate_up: &mut cudarc::driver::CudaSlice<half::bf16>,
    w_ffn_down: &cudarc::driver::CudaSlice<u8>,
    w_ffn_down_scale: &cudarc::driver::CudaSlice<u8>,
    out_down: &mut cudarc::driver::CudaSlice<half::bf16>,
    seq: usize,
    hidden: usize,
    qkv_n: usize,
    attn_out_n: usize,
    ffn_gate_up_n: usize,
    ffn_down_n: usize,
    ffn: usize,
    iters: usize,
) -> Result<f64, BenchError> {
    use cudarc::driver::{DevicePtr, DevicePtrMut};
    use rustorch_cuda::cusparse_lt::SparseLtSession;
    use std::time::Instant;

    // Workaround GB10 0.9.1.1 — une session par poids
    let ses_qkv = SparseLtSession::new(stream.clone())?;
    let ses_attn = SparseLtSession::new(stream.clone())?;
    let ses_gu = SparseLtSession::new(stream.clone())?;
    let ses_dn = SparseLtSession::new(stream.clone())?;

    println!("  prune+compress FP4 weights to 2:4 (4 sessions) ...");
    let (sa_qkv, _r) = unsafe { w_qkv_scale.device_ptr(stream) };
    let (sa_attn, _r) = unsafe { w_attn_out_scale.device_ptr(stream) };
    let (sa_gu, _r) = unsafe { w_ffn_gate_up_scale.device_ptr(stream) };
    let (sa_dn, _r) = unsafe { w_ffn_down_scale.device_ptr(stream) };
    let (sb_act, _r) = unsafe { act_scale.device_ptr(stream) };
    let (sb_int, _r) = unsafe { ffn_int_scale.device_ptr(stream) };

    let (w_qkv_p, _r) = unsafe { w_qkv.device_ptr(stream) };
    let mut sp_qkv =
        unsafe { ses_qkv.prune_compress_fp4(w_qkv_p, sa_qkv, sb_act, qkv_n, hidden, seq) }?;
    let (w_attn_p, _r) = unsafe { w_attn_out.device_ptr(stream) };
    let mut sp_attn =
        unsafe { ses_attn.prune_compress_fp4(w_attn_p, sa_attn, sb_act, attn_out_n, hidden, seq) }?;
    let (w_gu_p, _r) = unsafe { w_ffn_gate_up.device_ptr(stream) };
    let mut sp_gu =
        unsafe { ses_gu.prune_compress_fp4(w_gu_p, sa_gu, sb_act, ffn_gate_up_n, hidden, seq) }?;
    let (w_dn_p, _r) = unsafe { w_ffn_down.device_ptr(stream) };
    let mut sp_dn =
        unsafe { ses_dn.prune_compress_fp4(w_dn_p, sa_dn, sb_int, ffn_down_n, ffn, seq) }?;
    println!("  weights ready, running 4 sparse-FP4 matmuls per block");

    // Warm-up.
    for _ in 0..3 {
        unsafe {
            {
                let (b_p, _r) = act_fp4.device_ptr(stream);
                let (c_p, _r2) = out_qkv.device_ptr_mut(stream);
                ses_qkv.matmul_bf16(&mut sp_qkv, b_p, c_p, 1.0, 0.0)?;
            }
            {
                let (b_p, _r) = act_fp4.device_ptr(stream);
                let (c_p, _r2) = out_attn.device_ptr_mut(stream);
                ses_attn.matmul_bf16(&mut sp_attn, b_p, c_p, 1.0, 0.0)?;
            }
            {
                let (b_p, _r) = act_fp4.device_ptr(stream);
                let (c_p, _r2) = out_gate_up.device_ptr_mut(stream);
                ses_gu.matmul_bf16(&mut sp_gu, b_p, c_p, 1.0, 0.0)?;
            }
            {
                let (b_p, _r) = ffn_int_fp4.device_ptr(stream);
                let (c_p, _r2) = out_down.device_ptr_mut(stream);
                ses_dn.matmul_bf16(&mut sp_dn, b_p, c_p, 1.0, 0.0)?;
            }
        }
    }
    stream.synchronize()?;

    let t0 = Instant::now();
    for _ in 0..iters {
        unsafe {
            {
                let (b_p, _r) = act_fp4.device_ptr(stream);
                let (c_p, _r2) = out_qkv.device_ptr_mut(stream);
                ses_qkv.matmul_bf16(&mut sp_qkv, b_p, c_p, 1.0, 0.0)?;
            }
            {
                let (b_p, _r) = act_fp4.device_ptr(stream);
                let (c_p, _r2) = out_attn.device_ptr_mut(stream);
                ses_attn.matmul_bf16(&mut sp_attn, b_p, c_p, 1.0, 0.0)?;
            }
            {
                let (b_p, _r) = act_fp4.device_ptr(stream);
                let (c_p, _r2) = out_gate_up.device_ptr_mut(stream);
                ses_gu.matmul_bf16(&mut sp_gu, b_p, c_p, 1.0, 0.0)?;
            }
            {
                let (b_p, _r) = ffn_int_fp4.device_ptr(stream);
                let (c_p, _r2) = out_down.device_ptr_mut(stream);
                ses_dn.matmul_bf16(&mut sp_dn, b_p, c_p, 1.0, 0.0)?;
            }
        }
    }
    stream.synchronize()?;
    let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
    Ok(elapsed_ms / iters as f64)
}

#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn try_run_sparse_block(
    stream: &std::sync::Arc<cudarc::driver::CudaStream>,
    act: &cudarc::driver::CudaSlice<half::bf16>,
    w_qkv: &cudarc::driver::CudaSlice<half::bf16>,
    out_qkv: &mut cudarc::driver::CudaSlice<half::bf16>,
    w_attn_out: &cudarc::driver::CudaSlice<half::bf16>,
    out_attn: &mut cudarc::driver::CudaSlice<half::bf16>,
    w_ffn_gate_up: &cudarc::driver::CudaSlice<half::bf16>,
    out_gate_up: &mut cudarc::driver::CudaSlice<half::bf16>,
    w_ffn_down: &cudarc::driver::CudaSlice<half::bf16>,
    out_down: &mut cudarc::driver::CudaSlice<half::bf16>,
    seq: usize,
    hidden: usize,
    qkv_n: usize,
    attn_out_n: usize,
    ffn_gate_up_n: usize,
    ffn_down_n: usize,
    ffn: usize,
    iters: usize,
) -> Result<f64, BenchError> {
    use cudarc::driver::{DevicePtr, DevicePtrMut};
    use rustorch_cuda::cusparse_lt::SparseLtSession;
    use std::time::Instant;

    // T240.8c WORKAROUND : cuSPARSELt 0.9.1.1 sur GB10 corrompt l'état
    // interne quand on partage la même Session entre N SparseWeight.
    // Workaround : une Session par poids (4 ici).
    let sparse_session_qkv = SparseLtSession::new(stream.clone())?;
    let sparse_session_attn = SparseLtSession::new(stream.clone())?;
    let sparse_session_gu = SparseLtSession::new(stream.clone())?;
    let sparse_session_dn = SparseLtSession::new(stream.clone())?;

    println!("  pruning + compressing weights to 2:4 (4 sessions) ...");
    let (w_qkv_p, _r) = unsafe { w_qkv.device_ptr(stream) };
    let mut sparse_qkv =
        unsafe { sparse_session_qkv.prune_compress_bf16(w_qkv_p, qkv_n, hidden, seq) }?;
    let (w_attn_p, _r) = unsafe { w_attn_out.device_ptr(stream) };
    let mut sparse_attn =
        unsafe { sparse_session_attn.prune_compress_bf16(w_attn_p, attn_out_n, hidden, seq) }?;
    let (w_gu_p, _r) = unsafe { w_ffn_gate_up.device_ptr(stream) };
    let mut sparse_gu =
        unsafe { sparse_session_gu.prune_compress_bf16(w_gu_p, ffn_gate_up_n, hidden, seq) }?;
    let (w_dn_p, _r) = unsafe { w_ffn_down.device_ptr(stream) };
    let mut sparse_dn =
        unsafe { sparse_session_dn.prune_compress_bf16(w_dn_p, ffn_down_n, ffn, seq) }?;
    println!("  weights ready, running 4 sparse matmuls per block (4 sessions, workaround GB10 0.9.1.1 bug)");
    // Test mode "ITER1" : combien de répétitions tient avant fail
    if std::env::var("RUSTORCH_SPARSE_ITER")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .is_some()
    {
        let n: usize = std::env::var("RUSTORCH_SPARSE_ITER")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1);
        println!("  [iter-test] running {n} iterations, sync each, report when fails");
        for i in 0..n {
            unsafe {
                let (b_p, _r) = act.device_ptr(stream);
                let (c_p, _r2) = out_qkv.device_ptr_mut(stream);
                let r = sparse_session_qkv.matmul_bf16(&mut sparse_qkv, b_p, c_p, 1.0, 0.0);
                stream.synchronize().ok();
                if r.is_err() {
                    println!("    iter {i}: qkv FAIL {r:?}");
                    return Ok(0.001);
                }
            }
            unsafe {
                let (b_p, _r) = act.device_ptr(stream);
                let (c_p, _r2) = out_attn.device_ptr_mut(stream);
                let r = sparse_session_attn.matmul_bf16(&mut sparse_attn, b_p, c_p, 1.0, 0.0);
                stream.synchronize().ok();
                if r.is_err() {
                    println!("    iter {i}: attn FAIL {r:?}");
                    return Ok(0.001);
                }
            }
            unsafe {
                let (b_p, _r) = act.device_ptr(stream);
                let (c_p, _r2) = out_gate_up.device_ptr_mut(stream);
                let r = sparse_session_gu.matmul_bf16(&mut sparse_gu, b_p, c_p, 1.0, 0.0);
                stream.synchronize().ok();
                if r.is_err() {
                    println!("    iter {i}: gu FAIL {r:?}");
                    return Ok(0.001);
                }
            }
            unsafe {
                let (b_p, _r3) = out_gate_up.device_ptr(stream);
                let (c_p, _r2) = out_down.device_ptr_mut(stream);
                let r = sparse_session_dn.matmul_bf16(&mut sparse_dn, b_p, c_p, 1.0, 0.0);
                stream.synchronize().ok();
                if r.is_err() {
                    println!("    iter {i}: dn FAIL {r:?}");
                    return Ok(0.001);
                }
            }
        }
        println!("    {n} iterations OK");
        return Ok(0.001);
    }
    // Diagnostic : sync après chacune pour identifier laquelle échoue
    if std::env::var("RUSTORCH_SPARSE_DEBUG")
        .map(|v| v == "1")
        .unwrap_or(false)
    {
        unsafe {
            let (b_p, _r) = act.device_ptr(stream);
            let (c_p, _r2) = out_qkv.device_ptr_mut(stream);
            let r = sparse_session_qkv.matmul_bf16(&mut sparse_qkv, b_p, c_p, 1.0, 0.0);
            stream.synchronize().ok();
            println!("    [debug] qkv sparse matmul: {r:?}");
        }
        unsafe {
            let (b_p, _r) = act.device_ptr(stream);
            let (c_p, _r2) = out_attn.device_ptr_mut(stream);
            let r = sparse_session_attn.matmul_bf16(&mut sparse_attn, b_p, c_p, 1.0, 0.0);
            stream.synchronize().ok();
            println!("    [debug] attn sparse matmul: {r:?}");
        }
        unsafe {
            let (b_p, _r) = act.device_ptr(stream);
            let (c_p, _r2) = out_gate_up.device_ptr_mut(stream);
            let r = sparse_session_gu.matmul_bf16(&mut sparse_gu, b_p, c_p, 1.0, 0.0);
            stream.synchronize().ok();
            println!("    [debug] gate_up sparse matmul: {r:?}");
        }
        unsafe {
            let (b_p, _r) = out_gate_up.device_ptr(stream);
            let (c_p, _r2) = out_down.device_ptr_mut(stream);
            let r = sparse_session_dn.matmul_bf16(&mut sparse_dn, b_p, c_p, 1.0, 0.0);
            stream.synchronize().ok();
            println!("    [debug] down sparse matmul: {r:?}");
        }
        return Ok(0.001);
    }
    if std::env::var("RUSTORCH_SPARSE_AUTOTUNE")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
    {
        println!("  autotuning sparse plans (this can take minutes per shape) ...");
        unsafe {
            let (b_p, _r) = act.device_ptr(stream);
            let (c_p, _r2) = out_qkv.device_ptr_mut(stream);
            sparse_session_qkv.autotune(&mut sparse_qkv, b_p, c_p, 1.0, 0.0)?;
        }
        unsafe {
            let (b_p, _r) = act.device_ptr(stream);
            let (c_p, _r2) = out_attn.device_ptr_mut(stream);
            sparse_session_attn.autotune(&mut sparse_attn, b_p, c_p, 1.0, 0.0)?;
        }
        unsafe {
            let (b_p, _r) = act.device_ptr(stream);
            let (c_p, _r2) = out_gate_up.device_ptr_mut(stream);
            sparse_session_gu.autotune(&mut sparse_gu, b_p, c_p, 1.0, 0.0)?;
        }
        unsafe {
            let (b_p, _r3) = out_gate_up.device_ptr(stream);
            let (c_p, _r2) = out_down.device_ptr_mut(stream);
            sparse_session_dn.autotune(&mut sparse_dn, b_p, c_p, 1.0, 0.0)?;
        }
        println!("  autotune done");
    }

    // Warm-up.
    for _ in 0..3 {
        unsafe {
            {
                let (b_p, _r) = act.device_ptr(stream);
                let (c_p, _r2) = out_qkv.device_ptr_mut(stream);
                sparse_session_qkv.matmul_bf16(&mut sparse_qkv, b_p, c_p, 1.0, 0.0)?;
            }
            {
                let (b_p, _r) = act.device_ptr(stream);
                let (c_p, _r2) = out_attn.device_ptr_mut(stream);
                sparse_session_attn.matmul_bf16(&mut sparse_attn, b_p, c_p, 1.0, 0.0)?;
            }
            {
                let (b_p, _r) = act.device_ptr(stream);
                let (c_p, _r2) = out_gate_up.device_ptr_mut(stream);
                sparse_session_gu.matmul_bf16(&mut sparse_gu, b_p, c_p, 1.0, 0.0)?;
            }
            // FFN-down takes the (k=ffn, n=seq) slice of out_gate_up
            {
                let (b_p, _r3) = out_gate_up.device_ptr(stream);
                let (c_p, _r2) = out_down.device_ptr_mut(stream);
                sparse_session_dn.matmul_bf16(&mut sparse_dn, b_p, c_p, 1.0, 0.0)?;
            }
        }
    }
    stream.synchronize()?;

    let t0 = Instant::now();
    for _ in 0..iters {
        unsafe {
            {
                let (b_p, _r) = act.device_ptr(stream);
                let (c_p, _r2) = out_qkv.device_ptr_mut(stream);
                sparse_session_qkv.matmul_bf16(&mut sparse_qkv, b_p, c_p, 1.0, 0.0)?;
            }
            {
                let (b_p, _r) = act.device_ptr(stream);
                let (c_p, _r2) = out_attn.device_ptr_mut(stream);
                sparse_session_attn.matmul_bf16(&mut sparse_attn, b_p, c_p, 1.0, 0.0)?;
            }
            {
                let (b_p, _r) = act.device_ptr(stream);
                let (c_p, _r2) = out_gate_up.device_ptr_mut(stream);
                sparse_session_gu.matmul_bf16(&mut sparse_gu, b_p, c_p, 1.0, 0.0)?;
            }
            {
                let (b_p, _r3) = out_gate_up.device_ptr(stream);
                let (c_p, _r2) = out_down.device_ptr_mut(stream);
                sparse_session_dn.matmul_bf16(&mut sparse_dn, b_p, c_p, 1.0, 0.0)?;
            }
        }
    }
    stream.synchronize()?;
    let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
    Ok(elapsed_ms / iters as f64)
}

#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn run_bf16_block(
    session: &mut rustorch_cuda::cublas_lt::LtSession,
    stream: &std::sync::Arc<cudarc::driver::CudaStream>,
    act: &cudarc::driver::CudaSlice<half::bf16>,
    w_qkv: &cudarc::driver::CudaSlice<half::bf16>,
    out_qkv: &mut cudarc::driver::CudaSlice<half::bf16>,
    w_attn_out: &cudarc::driver::CudaSlice<half::bf16>,
    out_attn: &mut cudarc::driver::CudaSlice<half::bf16>,
    w_ffn_gate_up: &cudarc::driver::CudaSlice<half::bf16>, // T240.8f fused
    out_gate_up: &mut cudarc::driver::CudaSlice<half::bf16>,
    w_ffn_down: &cudarc::driver::CudaSlice<half::bf16>,
    out_down: &mut cudarc::driver::CudaSlice<half::bf16>,
    seq: usize,
    hidden: usize,
    qkv_n: usize,
    attn_out_n: usize,
    ffn_gate_up_n: usize, // = 2·ffn
    ffn_down_n: usize,
    ffn: usize,
    iters: usize,
) -> Result<f64, BenchError> {
    use cudarc::driver::{DevicePtr, DevicePtrMut};
    use std::time::Instant;

    // T240.8f — 4 matmuls per block (was 5): fused FFN gate+up via single
    // (hidden × 2·ffn) matmul. Real Qwen / LLaMA / Mixtral all do this.
    // Out_gate_up is (seq × 2·ffn) column-major — first ffn columns = "up"
    // partition (used as input to FFN-down), next ffn columns = "gate"
    // (would feed SwiGLU activation in real model).

    // T240.8b — opt-in multi-algo autotune.
    let autotune = std::env::var("RUSTORCH_CUBLASLT_AUTOTUNE")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("on") || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    if autotune {
        let n_candidates: u32 = std::env::var("RUSTORCH_CUBLASLT_AUTOTUNE_N")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(8);
        let n_passes: u32 = 5;
        unsafe {
            let (a_p, _r1) = act.device_ptr(stream);
            let (b_qkv, _r2) = w_qkv.device_ptr(stream);
            let (c_qkv, _r3) = out_qkv.device_ptr_mut(stream);
            let ms_qkv = session.autotune_bf16(
                a_p,
                b_qkv,
                c_qkv,
                seq,
                hidden,
                qkv_n,
                n_candidates,
                n_passes,
            )?;
            println!(
                "[autotune]   QKV proj   m={seq} k={hidden} n={qkv_n}        best={ms_qkv:.3} ms"
            );
        }
        unsafe {
            let (a_p, _r1) = act.device_ptr(stream);
            let (b_p, _r2) = w_attn_out.device_ptr(stream);
            let (c_p, _r3) = out_attn.device_ptr_mut(stream);
            let ms = session.autotune_bf16(
                a_p,
                b_p,
                c_p,
                seq,
                hidden,
                attn_out_n,
                n_candidates,
                n_passes,
            )?;
            println!(
                "[autotune]   Attn out   m={seq} k={hidden} n={attn_out_n}        best={ms:.3} ms"
            );
        }
        unsafe {
            let (a_p, _r1) = act.device_ptr(stream);
            let (b_p, _r2) = w_ffn_gate_up.device_ptr(stream);
            let (c_p, _r3) = out_gate_up.device_ptr_mut(stream);
            let ms = session.autotune_bf16(
                a_p,
                b_p,
                c_p,
                seq,
                hidden,
                ffn_gate_up_n,
                n_candidates,
                n_passes,
            )?;
            println!(
                "[autotune]   FFN g+up   m={seq} k={hidden} n={ffn_gate_up_n}    best={ms:.3} ms"
            );
        }
        unsafe {
            let (a_p, _r1) = out_gate_up.device_ptr(stream);
            let (b_p, _r2) = w_ffn_down.device_ptr(stream);
            let (c_p, _r3) = out_down.device_ptr_mut(stream);
            let ms = session.autotune_bf16(
                a_p,
                b_p,
                c_p,
                seq,
                ffn,
                ffn_down_n,
                n_candidates,
                n_passes,
            )?;
            println!(
                "[autotune]   FFN down   m={seq} k={ffn} n={ffn_down_n}        best={ms:.3} ms"
            );
        }
    }

    // Warm-up.
    for _ in 0..3 {
        unsafe {
            // QKV proj
            {
                let (a_p, _r1) = act.device_ptr(stream);
                let (b_p, _r2) = w_qkv.device_ptr(stream);
                let (c_p, _r3) = out_qkv.device_ptr_mut(stream);
                session.matmul_bf16(a_p, b_p, c_p, seq, hidden, qkv_n, 1.0, 0.0)?;
            }
            // Attn out
            {
                let (a_p, _r1) = act.device_ptr(stream);
                let (b_p, _r2) = w_attn_out.device_ptr(stream);
                let (c_p, _r3) = out_attn.device_ptr_mut(stream);
                session.matmul_bf16(a_p, b_p, c_p, seq, hidden, attn_out_n, 1.0, 0.0)?;
            }
            // FFN gate+up fused (T240.8f)
            {
                let (a_p, _r1) = act.device_ptr(stream);
                let (b_p, _r2) = w_ffn_gate_up.device_ptr(stream);
                let (c_p, _r3) = out_gate_up.device_ptr_mut(stream);
                session.matmul_bf16(a_p, b_p, c_p, seq, hidden, ffn_gate_up_n, 1.0, 0.0)?;
            }
            // FFN down — uses first ffn columns of out_gate_up (the "up" partition)
            {
                let (a_p, _r1) = out_gate_up.device_ptr(stream);
                let (b_p, _r2) = w_ffn_down.device_ptr(stream);
                let (c_p, _r3) = out_down.device_ptr_mut(stream);
                session.matmul_bf16(a_p, b_p, c_p, seq, ffn, ffn_down_n, 1.0, 0.0)?;
            }
        }
    }
    stream.synchronize()?;

    let t0 = Instant::now();
    for _ in 0..iters {
        unsafe {
            {
                let (a_p, _r1) = act.device_ptr(stream);
                let (b_p, _r2) = w_qkv.device_ptr(stream);
                let (c_p, _r3) = out_qkv.device_ptr_mut(stream);
                session.matmul_bf16(a_p, b_p, c_p, seq, hidden, qkv_n, 1.0, 0.0)?;
            }
            {
                let (a_p, _r1) = act.device_ptr(stream);
                let (b_p, _r2) = w_attn_out.device_ptr(stream);
                let (c_p, _r3) = out_attn.device_ptr_mut(stream);
                session.matmul_bf16(a_p, b_p, c_p, seq, hidden, attn_out_n, 1.0, 0.0)?;
            }
            {
                let (a_p, _r1) = act.device_ptr(stream);
                let (b_p, _r2) = w_ffn_gate_up.device_ptr(stream);
                let (c_p, _r3) = out_gate_up.device_ptr_mut(stream);
                session.matmul_bf16(a_p, b_p, c_p, seq, hidden, ffn_gate_up_n, 1.0, 0.0)?;
            }
            {
                let (a_p, _r1) = out_gate_up.device_ptr(stream);
                let (b_p, _r2) = w_ffn_down.device_ptr(stream);
                let (c_p, _r3) = out_down.device_ptr_mut(stream);
                session.matmul_bf16(a_p, b_p, c_p, seq, ffn, ffn_down_n, 1.0, 0.0)?;
            }
        }
    }
    stream.synchronize()?;
    let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
    Ok(elapsed_ms / iters as f64)
}

#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn run_fp4_block(
    session: &mut rustorch_cuda::cublas_lt::LtSession,
    stream: &std::sync::Arc<cudarc::driver::CudaStream>,
    act_fp4: &cudarc::driver::CudaSlice<u8>,
    act_scale: &cudarc::driver::CudaSlice<u8>,
    ffn_int_fp4: &cudarc::driver::CudaSlice<u8>,
    ffn_int_scale: &cudarc::driver::CudaSlice<u8>,
    w_qkv: &cudarc::driver::CudaSlice<u8>,
    w_qkv_scale: &cudarc::driver::CudaSlice<u8>,
    out_qkv: &mut cudarc::driver::CudaSlice<half::bf16>,
    w_attn_out: &cudarc::driver::CudaSlice<u8>,
    w_attn_out_scale: &cudarc::driver::CudaSlice<u8>,
    out_attn: &mut cudarc::driver::CudaSlice<half::bf16>,
    w_ffn_gate_up: &cudarc::driver::CudaSlice<u8>, // T240.8f fused
    w_ffn_gate_up_scale: &cudarc::driver::CudaSlice<u8>,
    out_gate_up: &mut cudarc::driver::CudaSlice<half::bf16>,
    w_ffn_down: &cudarc::driver::CudaSlice<u8>,
    w_ffn_down_scale: &cudarc::driver::CudaSlice<u8>,
    out_down: &mut cudarc::driver::CudaSlice<half::bf16>,
    seq: usize,
    hidden: usize,
    qkv_n: usize,
    attn_out_n: usize,
    ffn_gate_up_n: usize,
    ffn_down_n: usize,
    ffn: usize,
    scale_mode: rustorch_cuda::cublas_lt::Fp4ScaleMode,
    out_dtype: rustorch_cuda::cublas_lt::Fp8Output,
    iters: usize,
) -> Result<f64, BenchError> {
    use cudarc::driver::{DevicePtr, DevicePtrMut};
    use std::time::Instant;

    // T240.8f — 4 matmuls per block (was 5): fused FFN gate+up.

    // T240.8g — opt-in NVFP4 autotune (set RUSTORCH_CUBLASLT_AUTOTUNE=1).
    let autotune = std::env::var("RUSTORCH_CUBLASLT_AUTOTUNE")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("on") || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    if autotune {
        let n_candidates: u32 = std::env::var("RUSTORCH_CUBLASLT_AUTOTUNE_N")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(8);
        let n_passes: u32 = 5;
        unsafe {
            let (a_p, _r1) = act_fp4.device_ptr(stream);
            let (sa_p, _r2) = act_scale.device_ptr(stream);
            let (b_p, _r3) = w_qkv.device_ptr(stream);
            let (sb_p, _r4) = w_qkv_scale.device_ptr(stream);
            let (c_p, _r5) = out_qkv.device_ptr_mut(stream);
            let ms = session.autotune_mxfp4(
                a_p,
                sa_p,
                b_p,
                sb_p,
                c_p,
                seq,
                hidden,
                qkv_n,
                out_dtype,
                scale_mode,
                n_candidates,
                n_passes,
            )?;
            println!(
                "[autotune-fp4] QKV proj   m={seq} k={hidden} n={qkv_n}        best={ms:.3} ms"
            );
        }
        unsafe {
            let (a_p, _r1) = act_fp4.device_ptr(stream);
            let (sa_p, _r2) = act_scale.device_ptr(stream);
            let (b_p, _r3) = w_attn_out.device_ptr(stream);
            let (sb_p, _r4) = w_attn_out_scale.device_ptr(stream);
            let (c_p, _r5) = out_attn.device_ptr_mut(stream);
            let ms = session.autotune_mxfp4(
                a_p,
                sa_p,
                b_p,
                sb_p,
                c_p,
                seq,
                hidden,
                attn_out_n,
                out_dtype,
                scale_mode,
                n_candidates,
                n_passes,
            )?;
            println!("[autotune-fp4] Attn out   m={seq} k={hidden} n={attn_out_n}        best={ms:.3} ms");
        }
        unsafe {
            let (a_p, _r1) = act_fp4.device_ptr(stream);
            let (sa_p, _r2) = act_scale.device_ptr(stream);
            let (b_p, _r3) = w_ffn_gate_up.device_ptr(stream);
            let (sb_p, _r4) = w_ffn_gate_up_scale.device_ptr(stream);
            let (c_p, _r5) = out_gate_up.device_ptr_mut(stream);
            let ms = session.autotune_mxfp4(
                a_p,
                sa_p,
                b_p,
                sb_p,
                c_p,
                seq,
                hidden,
                ffn_gate_up_n,
                out_dtype,
                scale_mode,
                n_candidates,
                n_passes,
            )?;
            println!(
                "[autotune-fp4] FFN g+up   m={seq} k={hidden} n={ffn_gate_up_n}    best={ms:.3} ms"
            );
        }
        unsafe {
            let (a_p, _r1) = ffn_int_fp4.device_ptr(stream);
            let (sa_p, _r2) = ffn_int_scale.device_ptr(stream);
            let (b_p, _r3) = w_ffn_down.device_ptr(stream);
            let (sb_p, _r4) = w_ffn_down_scale.device_ptr(stream);
            let (c_p, _r5) = out_down.device_ptr_mut(stream);
            let ms = session.autotune_mxfp4(
                a_p,
                sa_p,
                b_p,
                sb_p,
                c_p,
                seq,
                ffn,
                ffn_down_n,
                out_dtype,
                scale_mode,
                n_candidates,
                n_passes,
            )?;
            println!(
                "[autotune-fp4] FFN down   m={seq} k={ffn} n={ffn_down_n}        best={ms:.3} ms"
            );
        }
    }

    // Warm-up.
    for _ in 0..3 {
        unsafe {
            // QKV proj
            {
                let (a_p, _r1) = act_fp4.device_ptr(stream);
                let (sa_p, _r2) = act_scale.device_ptr(stream);
                let (b_p, _r3) = w_qkv.device_ptr(stream);
                let (sb_p, _r4) = w_qkv_scale.device_ptr(stream);
                let (c_p, _r5) = out_qkv.device_ptr_mut(stream);
                session.matmul_mxfp4(
                    a_p, sa_p, b_p, sb_p, c_p, seq, hidden, qkv_n, 1.0, 0.0, out_dtype, scale_mode,
                )?;
            }
            // Attn out
            {
                let (a_p, _r1) = act_fp4.device_ptr(stream);
                let (sa_p, _r2) = act_scale.device_ptr(stream);
                let (b_p, _r3) = w_attn_out.device_ptr(stream);
                let (sb_p, _r4) = w_attn_out_scale.device_ptr(stream);
                let (c_p, _r5) = out_attn.device_ptr_mut(stream);
                session.matmul_mxfp4(
                    a_p, sa_p, b_p, sb_p, c_p, seq, hidden, attn_out_n, 1.0, 0.0, out_dtype,
                    scale_mode,
                )?;
            }
            // FFN gate+up fused
            {
                let (a_p, _r1) = act_fp4.device_ptr(stream);
                let (sa_p, _r2) = act_scale.device_ptr(stream);
                let (b_p, _r3) = w_ffn_gate_up.device_ptr(stream);
                let (sb_p, _r4) = w_ffn_gate_up_scale.device_ptr(stream);
                let (c_p, _r5) = out_gate_up.device_ptr_mut(stream);
                session.matmul_mxfp4(
                    a_p,
                    sa_p,
                    b_p,
                    sb_p,
                    c_p,
                    seq,
                    hidden,
                    ffn_gate_up_n,
                    1.0,
                    0.0,
                    out_dtype,
                    scale_mode,
                )?;
            }
            // FFN down (uses pre-quantized FFN intermediate as input)
            {
                let (a_p, _r1) = ffn_int_fp4.device_ptr(stream);
                let (sa_p, _r2) = ffn_int_scale.device_ptr(stream);
                let (b_p, _r3) = w_ffn_down.device_ptr(stream);
                let (sb_p, _r4) = w_ffn_down_scale.device_ptr(stream);
                let (c_p, _r5) = out_down.device_ptr_mut(stream);
                session.matmul_mxfp4(
                    a_p, sa_p, b_p, sb_p, c_p, seq, ffn, ffn_down_n, 1.0, 0.0, out_dtype,
                    scale_mode,
                )?;
            }
        }
    }
    stream.synchronize()?;

    let t0 = Instant::now();
    for _ in 0..iters {
        unsafe {
            {
                let (a_p, _r1) = act_fp4.device_ptr(stream);
                let (sa_p, _r2) = act_scale.device_ptr(stream);
                let (b_p, _r3) = w_qkv.device_ptr(stream);
                let (sb_p, _r4) = w_qkv_scale.device_ptr(stream);
                let (c_p, _r5) = out_qkv.device_ptr_mut(stream);
                session.matmul_mxfp4(
                    a_p, sa_p, b_p, sb_p, c_p, seq, hidden, qkv_n, 1.0, 0.0, out_dtype, scale_mode,
                )?;
            }
            {
                let (a_p, _r1) = act_fp4.device_ptr(stream);
                let (sa_p, _r2) = act_scale.device_ptr(stream);
                let (b_p, _r3) = w_attn_out.device_ptr(stream);
                let (sb_p, _r4) = w_attn_out_scale.device_ptr(stream);
                let (c_p, _r5) = out_attn.device_ptr_mut(stream);
                session.matmul_mxfp4(
                    a_p, sa_p, b_p, sb_p, c_p, seq, hidden, attn_out_n, 1.0, 0.0, out_dtype,
                    scale_mode,
                )?;
            }
            {
                let (a_p, _r1) = act_fp4.device_ptr(stream);
                let (sa_p, _r2) = act_scale.device_ptr(stream);
                let (b_p, _r3) = w_ffn_gate_up.device_ptr(stream);
                let (sb_p, _r4) = w_ffn_gate_up_scale.device_ptr(stream);
                let (c_p, _r5) = out_gate_up.device_ptr_mut(stream);
                session.matmul_mxfp4(
                    a_p,
                    sa_p,
                    b_p,
                    sb_p,
                    c_p,
                    seq,
                    hidden,
                    ffn_gate_up_n,
                    1.0,
                    0.0,
                    out_dtype,
                    scale_mode,
                )?;
            }
            {
                let (a_p, _r1) = ffn_int_fp4.device_ptr(stream);
                let (sa_p, _r2) = ffn_int_scale.device_ptr(stream);
                let (b_p, _r3) = w_ffn_down.device_ptr(stream);
                let (sb_p, _r4) = w_ffn_down_scale.device_ptr(stream);
                let (c_p, _r5) = out_down.device_ptr_mut(stream);
                session.matmul_mxfp4(
                    a_p, sa_p, b_p, sb_p, c_p, seq, ffn, ffn_down_n, 1.0, 0.0, out_dtype,
                    scale_mode,
                )?;
            }
        }
    }
    stream.synchronize()?;
    let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
    Ok(elapsed_ms / iters as f64)
}
