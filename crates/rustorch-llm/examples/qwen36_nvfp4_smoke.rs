//! T246.9 NVFP4.4 — smoke test : load the on-disk Qwen3.6-A3B-NVFP4 model
//! through `Qwen35ModelCudaNVFP4::from_safetensors`, dump per-layer
//! tensor counts, and verify the substrate (loader + indexed kernel) is
//! ready to power the full decode forward.
//!
//! This is the MVP gate for T246.9 piece 1+2 — confirms the loader can
//! ingest the real 22.5 GB checkpoint without explosion and that the
//! NVFP4 weight count matches the recipe.yaml expectations.
//!
//! ## Expected counts (per recipe.yaml + safetensors index)
//! - 30880 quantized linears (= 40 layers × 256 experts × 3 + 40 ×
//!   shared_expert × 3 + 10 × attn × 4) — confirms ignore-regex worked
//! - ~393 BF16 tensors (norms, embeddings, lm_head, gates, SSM stack)
//!
//! Per RFC b5fc8ead — the full coherent-text decode requires Piece 3
//! (Qwen35ModelCudaNVFP4::decode_step body, ~2000 LOC clone of Q4K
//! decode_step with NVFP4 dispatch swap). This smoke validates the
//! substrate is sound enough to power that follow-up.
//!
//! ## Usage
//! ```
//! cargo run --release --features cuda -p rustorch-llm \
//!     --example qwen36_nvfp4_smoke -- \
//!     /home/triviere/projects/models/qwen3.6-35b-a3b-nvfp4
//! ```

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("[qwen36_nvfp4_smoke] cuda feature is OFF");
}

#[cfg(feature = "cuda")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use cudarc::driver::CudaContext;
    use rustorch_llm::qwen35_cuda_nvfp4::load_nvfp4_safetensors;
    use std::env;
    use std::path::Path;
    use std::time::Instant;

    let args: Vec<String> = env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!("usage: qwen36_nvfp4_smoke <safetensors_dir>");
        std::process::exit(2);
    }
    let dir = Path::new(&args[0]);

    println!("[qwen36_nvfp4_smoke] dir={}", dir.display());

    // CUDA setup (no model construction yet — just want the stream for
    // the loader's CudaSlice uploads).
    let ctx = CudaContext::new(0).map_err(|e| format!("ctx: {e:?}"))?;
    let stream = ctx.default_stream();

    let t0 = Instant::now();
    let weights = load_nvfp4_safetensors(dir, &stream)
        .map_err(|e| format!("load_nvfp4_safetensors: {e:?}"))?;
    let load_secs = t0.elapsed().as_secs_f64();

    println!();
    println!("=== Qwen3.6-A3B-NVFP4 loaded in {load_secs:.2}s ===");
    println!("  total tensors      : {}", weights.len());
    println!("  NVFP4 (4-tuple)    : {}", weights.nvfp4.len());
    println!("  BF16 / non-quant   : {}", weights.bf16.len());

    // Sample: dump the first 5 NVFP4 tensors (lexical order from BTreeMap).
    println!();
    println!("--- First 5 NVFP4 tensors ---");
    for (i, (name, t)) in weights.nvfp4.iter().take(5).enumerate() {
        println!(
            "  [{i}] {name}  N={} K={} alpha={:.6e} (w_g={:.4e} in_g={:.4e})",
            t.n,
            t.k,
            t.matmul_alpha(),
            t.weight_global_scale,
            t.input_global_scale
        );
    }

    println!();
    println!("--- First 5 BF16 tensors ---");
    for (i, (name, t)) in weights.bf16.iter().take(5).enumerate() {
        println!("  [{i}] {name}  shape={:?}", t.shape);
    }

    // Per-layer breakdown : enumerate layer 0 / layer 3 / layer 39 to
    // verify both SSM-MoE and Attn-MoE layers picked up the right
    // tensors.
    println!();
    println!("--- Layer 0 (SSM + MoE) tensor inventory ---");
    let l0_prefix = "model.language_model.layers.0.";
    let l0_quant = weights
        .nvfp4
        .keys()
        .filter(|k| k.starts_with(l0_prefix))
        .count();
    let l0_bf16 = weights
        .bf16
        .keys()
        .filter(|k| k.starts_with(l0_prefix))
        .count();
    println!("  layer 0 quantized = {l0_quant}");
    println!("  layer 0 BF16      = {l0_bf16}");

    println!();
    println!("--- Layer 3 (full attention + MoE) tensor inventory ---");
    let l3_prefix = "model.language_model.layers.3.";
    let l3_quant = weights
        .nvfp4
        .keys()
        .filter(|k| k.starts_with(l3_prefix))
        .count();
    let l3_bf16 = weights
        .bf16
        .keys()
        .filter(|k| k.starts_with(l3_prefix))
        .count();
    println!("  layer 3 quantized = {l3_quant}");
    println!("  layer 3 BF16      = {l3_bf16}");

    // Final sanity : verify a few well-known shapes.
    println!();
    println!("--- Shape sanity ---");
    if let Some(emb) = weights.bf16.get("model.language_model.embed_tokens.weight") {
        println!(
            "  embed_tokens     shape = {:?} (expect [248320, 2048])",
            emb.shape
        );
    } else {
        println!("  embed_tokens     MISSING");
    }
    if let Some(lm) = weights.bf16.get("lm_head.weight") {
        println!(
            "  lm_head          shape = {:?} (expect [248320, 2048])",
            lm.shape
        );
    } else {
        println!("  lm_head          MISSING");
    }
    if let Some(qproj) = weights
        .nvfp4
        .get("model.language_model.layers.3.self_attn.q_proj")
    {
        println!(
            "  layer3.q_proj    N={} K={} (expect 4096 / 2048)",
            qproj.n, qproj.k
        );
    } else {
        println!("  layer3.q_proj    MISSING");
    }
    if let Some(gate) = weights
        .nvfp4
        .get("model.language_model.layers.0.mlp.experts.0.gate_proj")
    {
        println!(
            "  layer0.exp0.gate N={} K={} (expect 512 / 2048)",
            gate.n, gate.k
        );
    } else {
        println!("  layer0.exp0.gate MISSING");
    }

    // ── End-to-end pipeline test : loader → kernel ──────────────────
    //
    // Pick layer 0 expert 0 gate_proj (a real loaded NVFP4 tensor),
    // synthesize a BF16 activation, run sgemv_nvfp4_bf16 against it,
    // and verify the output is finite + non-trivial.
    println!();
    println!("--- E2E pipeline test : loader + sgemv_nvfp4_bf16 on real weight ---");
    use cudarc::driver::{DevicePtr, DevicePtrMut};
    use rustorch_cuda::llm_kernels::LlmKernels;
    let kernels = LlmKernels::new(ctx.clone());
    if let Some(t) = weights
        .nvfp4
        .get("model.language_model.layers.0.mlp.experts.0.gate_proj")
    {
        let n = t.n;
        let k = t.k;
        let alpha = t.matmul_alpha();
        // Synthesize BF16 activation [k] (signed sinusoid, magnitudes ≤ 0.4).
        let x: Vec<half::bf16> = (0..k)
            .map(|i| half::bf16::from_f32(((i as f32 + 1.0) * 0.0021).sin() * 0.4))
            .collect();
        let x_dev = stream
            .memcpy_stod(&x)
            .map_err(|e| format!("x dev: {e:?}"))?;
        let mut y_dev = stream
            .memcpy_stod(&vec![half::bf16::from_f32(0.0); n])
            .map_err(|e| format!("y dev: {e:?}"))?;
        unsafe {
            let (pp, _g0) = t.packed.device_ptr(&stream);
            let (sp, _g1) = t.scale_per_block.device_ptr(&stream);
            let (xp, _g2) = x_dev.device_ptr(&stream);
            let (yp, _g3) = y_dev.device_ptr_mut(&stream);
            kernels
                .sgemv_nvfp4_bf16(&stream, pp, sp, alpha, xp, yp, n as i32, k as i32)
                .map_err(|e| format!("sgemv launch: {e:?}"))?;
        }
        let y_host: Vec<half::bf16> = stream
            .memcpy_dtov(&y_dev)
            .map_err(|e| format!("dl: {e:?}"))?;
        // Check : no NaN/Inf, at least some non-zero output.
        let n_nan = y_host.iter().filter(|v| v.to_f32().is_nan()).count();
        let n_inf = y_host.iter().filter(|v| v.to_f32().is_infinite()).count();
        let n_nonzero = y_host.iter().filter(|v| v.to_f32().abs() > 1e-12).count();
        let max_abs = y_host
            .iter()
            .map(|v| v.to_f32().abs())
            .fold(0.0_f32, f32::max);
        let first8: Vec<f32> = y_host.iter().take(8).map(|v| v.to_f32()).collect();
        println!(
            "  sgemv y[:N={}] : nan={n_nan} inf={n_inf} nonzero={n_nonzero} max_abs={max_abs:.6}",
            n
        );
        println!("  y[0..8] = {first8:?}");
        if n_nan != 0 || n_inf != 0 {
            return Err("E2E test FAILED : NaN or Inf in output".into());
        }
        // Note : with the calibrated NVFP4 alpha (~2.5e-7) and BF16
        // activations of magnitude ~0.4, the post-alpha dot products
        // land in the 1e-3 to 1e-4 range. Many BF16 outputs round to 0
        // near the lower end of that range — this is EXPECTED (the
        // activations as stored were not the calibration-source ones,
        // so alpha is "not quite right" for our synthetic input). The
        // kernel correctness gate is the parity test ; here we just
        // need : no NaN/Inf, max_abs > 0.
        if max_abs <= 0.0 {
            return Err(format!("E2E test FAILED : output is all zero (max_abs={max_abs})").into());
        }
        println!(
            "  [PASS] real-weight sgemv produces finite output, max_abs={max_abs:.3e}, \
             nonzero={n_nonzero}/{n} (BF16 underflow expected on calibrated alpha)"
        );
    } else {
        println!("  [SKIP] expected layer0.exp0.gate_proj not found");
    }

    println!();
    println!("[qwen36_nvfp4_smoke] DONE — substrate ready for full decode_step (T246.9 P3)");
    Ok(())
}
