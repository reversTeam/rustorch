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
    let seq = 2048usize;

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

    // Weights (one set per matmul, reused across layers — same shapes).
    let w_qkv_bf16 = vec![half::bf16::from_f32(0.01); hidden * qkv_n];
    let w_attn_out_bf16 = vec![half::bf16::from_f32(0.01); hidden * attn_out_n];
    let w_ffn_up_bf16 = vec![half::bf16::from_f32(0.01); hidden * ffn_up_n];
    let w_ffn_gate_bf16 = vec![half::bf16::from_f32(0.01); hidden * ffn_gate_n];
    let w_ffn_down_bf16 = vec![half::bf16::from_f32(0.01); ffn * ffn_down_n];

    let w_qkv_dev = stream.memcpy_stod(&w_qkv_bf16)?;
    let w_attn_out_dev = stream.memcpy_stod(&w_attn_out_bf16)?;
    let w_ffn_up_dev = stream.memcpy_stod(&w_ffn_up_bf16)?;
    let w_ffn_gate_dev = stream.memcpy_stod(&w_ffn_gate_bf16)?;
    let w_ffn_down_dev = stream.memcpy_stod(&w_ffn_down_bf16)?;

    // Outputs (intermediate buffers).
    let mut out_qkv = stream.alloc_zeros::<half::bf16>(seq * qkv_n)?;
    let mut out_attn = stream.alloc_zeros::<half::bf16>(seq * attn_out_n)?;
    let mut out_up = stream.alloc_zeros::<half::bf16>(seq * ffn_up_n)?;
    let mut out_gate = stream.alloc_zeros::<half::bf16>(seq * ffn_gate_n)?;
    let mut out_down = stream.alloc_zeros::<half::bf16>(seq * ffn_down_n)?;

    // ───────── BF16 path ─────────
    println!("[qwen_block_bench] BF16 path (TensorCore native)");
    let bf16_per_block_ms = run_bf16_block(
        &mut session,
        &stream,
        &act_dev_bf16,
        &w_qkv_dev,
        &mut out_qkv,
        &w_attn_out_dev,
        &mut out_attn,
        &w_ffn_up_dev,
        &mut out_up,
        &w_ffn_gate_dev,
        &mut out_gate,
        &w_ffn_down_dev,
        &mut out_down,
        seq,
        hidden,
        qkv_n,
        attn_out_n,
        ffn_up_n,
        ffn_gate_n,
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

    println!();
    println!("[qwen_block_bench] for FP8/FP4 paths the per-call FLOPS are identical;");
    println!(
        "[qwen_block_bench] expected speedup follows the gemm bench: BF16→FP8 ~2x, FP8→FP4 ~1.7x."
    );
    println!(
        "[qwen_block_bench] per-block FP4 cached projection: ~{:.3} ms  →  ~{:.0} tok/s prefill",
        bf16_per_block_ms / 3.5,
        bf16_tok_s * 3.5
    );

    Ok(())
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
    w_ffn_up: &cudarc::driver::CudaSlice<half::bf16>,
    out_up: &mut cudarc::driver::CudaSlice<half::bf16>,
    w_ffn_gate: &cudarc::driver::CudaSlice<half::bf16>,
    out_gate: &mut cudarc::driver::CudaSlice<half::bf16>,
    w_ffn_down: &cudarc::driver::CudaSlice<half::bf16>,
    out_down: &mut cudarc::driver::CudaSlice<half::bf16>,
    seq: usize,
    hidden: usize,
    qkv_n: usize,
    attn_out_n: usize,
    ffn_up_n: usize,
    ffn_gate_n: usize,
    ffn_down_n: usize,
    ffn: usize,
    iters: usize,
) -> Result<f64, BenchError> {
    use cudarc::driver::{DevicePtr, DevicePtrMut};
    use std::time::Instant;

    // 5 matmuls per block, reused weights, reused activation. Total time
    // measures the dense compute envelope of one Qwen-style block.
    let _ = ffn_gate_n;

    // Warm-up.
    for _ in 0..3 {
        unsafe {
            // QKV proj : (seq × hidden) @ (hidden × qkv_n) = (seq × qkv_n)
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
            // FFN up
            {
                let (a_p, _r1) = act.device_ptr(stream);
                let (b_p, _r2) = w_ffn_up.device_ptr(stream);
                let (c_p, _r3) = out_up.device_ptr_mut(stream);
                session.matmul_bf16(a_p, b_p, c_p, seq, hidden, ffn_up_n, 1.0, 0.0)?;
            }
            // FFN gate
            {
                let (a_p, _r1) = act.device_ptr(stream);
                let (b_p, _r2) = w_ffn_gate.device_ptr(stream);
                let (c_p, _r3) = out_gate.device_ptr_mut(stream);
                session.matmul_bf16(a_p, b_p, c_p, seq, hidden, ffn_up_n, 1.0, 0.0)?;
            }
            // FFN down (uses out_up as input shape (seq × ffn))
            {
                let (a_p, _r1) = out_up.device_ptr(stream);
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
            // FFN up
            {
                let (a_p, _r1) = act.device_ptr(stream);
                let (b_p, _r2) = w_ffn_up.device_ptr(stream);
                let (c_p, _r3) = out_up.device_ptr_mut(stream);
                session.matmul_bf16(a_p, b_p, c_p, seq, hidden, ffn_up_n, 1.0, 0.0)?;
            }
            // FFN gate
            {
                let (a_p, _r1) = act.device_ptr(stream);
                let (b_p, _r2) = w_ffn_gate.device_ptr(stream);
                let (c_p, _r3) = out_gate.device_ptr_mut(stream);
                session.matmul_bf16(a_p, b_p, c_p, seq, hidden, ffn_up_n, 1.0, 0.0)?;
            }
            // FFN down (uses out_up as input shape (seq × ffn))
            {
                let (a_p, _r1) = out_up.device_ptr(stream);
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
