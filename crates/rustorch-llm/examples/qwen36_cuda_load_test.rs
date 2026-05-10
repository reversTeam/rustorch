//! T246.1 — load test : verify Qwen35ModelCudaQ4K::from_gguf can load
//! a real Qwen3.6-27B Q4_K_M GGUF without errors.
//!
//! Usage : `qwen36_cuda_load_test <gguf_path>`

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("[qwen36_cuda_load_test] cuda feature is OFF");
}

#[cfg(feature = "cuda")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use rustorch_llm::qwen35_cuda_q4k::Qwen35ModelCudaQ4K;
    use std::env;
    use std::path::Path;
    use std::time::Instant;

    let args: Vec<String> = env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!("usage: qwen36_cuda_load_test <gguf_path>");
        std::process::exit(2);
    }
    let gguf_path = Path::new(&args[0]);
    let max_seq: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(2048);

    println!("[qwen36_cuda_load_test] gguf={}", gguf_path.display());
    println!("[qwen36_cuda_load_test] max_seq={max_seq}");

    let t0 = Instant::now();
    let model = Qwen35ModelCudaQ4K::from_gguf(gguf_path, max_seq)
        .map_err(|e| format!("from_gguf: {e:?}"))?;
    let load_secs = t0.elapsed().as_secs_f64();

    println!();
    println!("=== Qwen35ModelCudaQ4K loaded in {load_secs:.2}s ===");
    println!("  variant            : {:?}", model.config.variant);
    println!("  n_layers           : {}", model.config.n_layers);
    println!(
        "  attention_indices  : {} layers",
        model.config.attention_indices.len()
    );
    println!(
        "  ssm_indices        : {} layers",
        model.config.ssm_indices.len()
    );
    println!("  hidden d           : {}", model.config.d);
    println!("  ffn   f            : {}", model.config.f);
    println!("  vocab              : {}", model.config.vocab);
    println!("  n_q_heads          : {}", model.config.n_q_heads);
    println!("  n_kv_heads         : {}", model.config.n_kv_heads);
    println!("  head_dim           : {}", model.config.head_dim());
    println!("  ssm_groups         : {}", model.config.ssm_groups);
    println!("  ssm_state          : {}", model.config.ssm_state);
    println!("  ssm_dt_rank        : {}", model.config.ssm_dt_rank);
    println!("  ssm_conv_kernel    : {}", model.config.ssm_conv_kernel);

    Ok(())
}
