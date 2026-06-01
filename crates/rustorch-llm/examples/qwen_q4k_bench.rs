//! `qwen_q4k_bench` — bench end-to-end Qwen2.5-7B Q4_K_M en utilisant
//! sgemv_q4k_bf16_v2 + sgemv_q6k_bf16 directement (pas de CPU dequant).
//!
//! Usage : `qwen_q4k_bench <gguf_path>`

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("[qwen_q4k_bench] cuda feature is OFF");
}

#[cfg(feature = "cuda")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use rustorch_llm::cuda_backend_q4k::LlamaModelCudaQ4K;
    use std::env;
    use std::path::Path;
    use std::time::Instant;

    let args: Vec<String> = env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!("usage: qwen_q4k_bench <gguf_path> [n_iters]");
        std::process::exit(2);
    }
    let gguf_path = Path::new(&args[0]);
    let n_iters: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(30);

    println!("[qwen_q4k_bench] gguf={}", gguf_path.display());
    println!("[qwen_q4k_bench] n_iters={n_iters}");

    let t0 = Instant::now();
    let mut model = LlamaModelCudaQ4K::from_gguf(gguf_path, 256).map_err(|e| format!("{e:?}"))?;
    println!(
        "[qwen_q4k_bench] loaded in {:.2}s — {} layers, hidden={}, ffn={}",
        t0.elapsed().as_secs_f64(),
        model.config.num_hidden_layers,
        model.config.hidden_size,
        model.config.intermediate_size
    );

    let elapsed_ms = model
        .bench_decode_matmul_only(n_iters)
        .map_err(|e| format!("{e:?}"))?;
    let tok_s = 1000.0 / elapsed_ms;

    println!();
    println!("=== Qwen2.5-7B Q4_K_M end-to-end (Q4K + Q6K direct, matmul-only) ===");
    println!("  per-token : {elapsed_ms:.2} ms = {tok_s:.2} tok/s");
    println!();
    println!("  Comparaison sur DGX Spark GB10 :");
    println!("    rustorch (this)            : {tok_s:.2} tok/s");
    println!("    rustorch BF16 (existant)   : 11.20 tok/s");
    println!("    llama.cpp Q4_K_M           : 47.15 tok/s");
    let r = tok_s / 47.15;
    println!(
        "    ratio rustorch/llama.cpp   : {r:.2}× ({})",
        if r >= 1.0 { "FASTER ✓" } else { "slower" }
    );

    Ok(())
}
