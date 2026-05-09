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

    // Qwen3.6-27B-style dense block dims (same hidden as 35B-A3B MoE).
    let hidden = 2048usize;
    let ffn = 11008usize;
    let n_heads = 16usize;
    let n_kv = 2usize;
    let head_dim = 256usize;
    let n_layers = 40usize;
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

    let sparse_session = SparseLtSession::new(stream.clone())?;

    // Build SparseWeight for each Qwen weight (one-time, like model loading).
    // cuSPARSELt convention: sparse-A. We model the matmul as
    //   C = W_sparse · X    (weight on the left, activation on the right)
    // where W has shape (n × hidden) and X has shape (hidden × seq).
    // For QKV: m=qkv_n, k=hidden, n=seq.
    println!("  pruning + compressing weights to 2:4 ...");
    let (w_qkv_p, _r) = unsafe { w_qkv.device_ptr(stream) };
    let mut sparse_qkv =
        unsafe { sparse_session.prune_compress_bf16(w_qkv_p, qkv_n, hidden, seq) }?;
    let (w_attn_p, _r) = unsafe { w_attn_out.device_ptr(stream) };
    let mut sparse_attn =
        unsafe { sparse_session.prune_compress_bf16(w_attn_p, attn_out_n, hidden, seq) }?;
    let (w_gu_p, _r) = unsafe { w_ffn_gate_up.device_ptr(stream) };
    let mut sparse_gu =
        unsafe { sparse_session.prune_compress_bf16(w_gu_p, ffn_gate_up_n, hidden, seq) }?;
    let (w_dn_p, _r) = unsafe { w_ffn_down.device_ptr(stream) };
    let mut sparse_dn =
        unsafe { sparse_session.prune_compress_bf16(w_dn_p, ffn_down_n, ffn, seq) }?;
    println!("  weights ready, running 4 sparse matmuls per block");

    // Warm-up.
    for _ in 0..3 {
        unsafe {
            let (b_p, _r) = act.device_ptr(stream);
            let (c_p, _r2) = out_qkv.device_ptr_mut(stream);
            sparse_session.matmul_bf16(&mut sparse_qkv, b_p, c_p, 1.0, 0.0)?;
            let (c_p, _r2) = out_attn.device_ptr_mut(stream);
            sparse_session.matmul_bf16(&mut sparse_attn, b_p, c_p, 1.0, 0.0)?;
            let (c_p, _r2) = out_gate_up.device_ptr_mut(stream);
            sparse_session.matmul_bf16(&mut sparse_gu, b_p, c_p, 1.0, 0.0)?;
            // FFN-down takes the (k=ffn, n=seq) slice of out_gate_up
            let (b_p, _r3) = out_gate_up.device_ptr(stream);
            let (c_p, _r2) = out_down.device_ptr_mut(stream);
            sparse_session.matmul_bf16(&mut sparse_dn, b_p, c_p, 1.0, 0.0)?;
        }
    }
    stream.synchronize()?;

    let t0 = Instant::now();
    for _ in 0..iters {
        unsafe {
            let (b_p, _r) = act.device_ptr(stream);
            let (c_p, _r2) = out_qkv.device_ptr_mut(stream);
            sparse_session.matmul_bf16(&mut sparse_qkv, b_p, c_p, 1.0, 0.0)?;
            let (c_p, _r2) = out_attn.device_ptr_mut(stream);
            sparse_session.matmul_bf16(&mut sparse_attn, b_p, c_p, 1.0, 0.0)?;
            let (c_p, _r2) = out_gate_up.device_ptr_mut(stream);
            sparse_session.matmul_bf16(&mut sparse_gu, b_p, c_p, 1.0, 0.0)?;
            let (b_p, _r3) = out_gate_up.device_ptr(stream);
            let (c_p, _r2) = out_down.device_ptr_mut(stream);
            sparse_session.matmul_bf16(&mut sparse_dn, b_p, c_p, 1.0, 0.0)?;
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
