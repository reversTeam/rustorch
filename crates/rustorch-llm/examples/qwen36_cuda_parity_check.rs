//! T246 — parity check : run ONE decode step on both CPU and CUDA,
//! compare the predicted next token ids. Validates that our CUDA
//! Qwen3.6 forward produces semantically aligned output with the
//! reference CPU forward (modulo BF16 quantization noise).
//!
//! Usage : `qwen36_cuda_parity_check <gguf_path> [start_token]`

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("[qwen36_cuda_parity_check] cuda feature is OFF");
}

#[cfg(feature = "cuda")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use rustorch_llm::qwen35::parse_config;
    use rustorch_llm::qwen35_cpu::{forward_token, load_weights, Qwen35State};
    use rustorch_llm::qwen35_cuda_q4k::Qwen35ModelCudaQ4K;
    use std::env;
    use std::path::Path;
    use std::time::Instant;

    let args: Vec<String> = env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!("usage: qwen36_cuda_parity_check <gguf_path> [start_token=1]");
        std::process::exit(2);
    }
    let gguf_path = Path::new(&args[0]);
    let start_token: u32 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(1);

    println!("[parity_check] gguf={}", gguf_path.display());
    println!("[parity_check] start_token={start_token}");

    // ---- Load CUDA model ----
    println!("\n[1/3] Loading CUDA model...");
    let t0 = Instant::now();
    let mut cuda_model =
        Qwen35ModelCudaQ4K::from_gguf(gguf_path, 16).map_err(|e| format!("from_gguf: {e:?}"))?;
    println!("  loaded in {:.2}s", t0.elapsed().as_secs_f64());

    // ---- Run CUDA decode_step ----
    println!("\n[2/3] Running CUDA decode_step...");
    let t0 = Instant::now();
    let cuda_next = cuda_model
        .decode_step(start_token)
        .map_err(|e| format!("cuda decode: {e:?}"))?;
    let cuda_secs = t0.elapsed().as_secs_f64();
    println!(
        "  CUDA: {start_token} → {cuda_next}  ({:.3}s = {:.2} tok/s)",
        cuda_secs,
        1.0 / cuda_secs
    );

    // ---- Load CPU model ----
    println!("\n[3/3] Loading CPU reference model (slow, dequantizes all weights to f32)...");
    let cfg = parse_config(gguf_path)?;
    let t0 = Instant::now();
    let cpu_weights = load_weights(gguf_path, &cfg)?;
    let cpu_load_secs = t0.elapsed().as_secs_f64();
    println!("  loaded in {cpu_load_secs:.2}s");
    let mut cpu_state = Qwen35State::new(&cfg, 16);

    // Run CPU forward_token (returns the next token id directly via argmax).
    let t0 = Instant::now();
    let cpu_next = forward_token(&cpu_weights, &mut cpu_state, start_token, 0);
    let cpu_secs = t0.elapsed().as_secs_f64();
    println!(
        "  CPU : {start_token} → {cpu_next}  ({:.2}s = {:.3} tok/s)",
        cpu_secs,
        1.0 / cpu_secs
    );

    // ---- Compare ----
    println!();
    println!("=== PARITY RESULT ===");
    println!("  CUDA next : {cuda_next}");
    println!("  CPU  next : {cpu_next}");
    if cuda_next == cpu_next {
        println!("  ✅ MATCH — CUDA forward produces the same argmax token as CPU reference.");
    } else {
        println!("  ❌ MISMATCH — CUDA produces a different argmax than CPU.");
        println!("     This indicates a numerical bug in the CUDA forward path.");
        println!("     Differences are EXPECTED at small scale due to BF16 vs F32 precision,");
        println!("     but argmax mismatch on a fresh state usually points to layout / shape /");
        println!("     scaling bugs that need investigation.");
    }

    Ok(())
}
