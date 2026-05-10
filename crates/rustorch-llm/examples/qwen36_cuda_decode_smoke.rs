//! T246 — smoke test : load Qwen3.6-27B Q4_K_M, run decode_step a few times,
//! print the output token ids. Validates end-to-end forward executes without
//! errors. Numerical correctness vs CPU reference is a separate concern.
//!
//! Usage : `qwen36_cuda_decode_smoke <gguf_path> [n_steps] [start_token]`

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("[qwen36_cuda_decode_smoke] cuda feature is OFF");
}

#[cfg(feature = "cuda")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use rustorch_llm::qwen35_cuda_q4k::Qwen35ModelCudaQ4K;
    use std::env;
    use std::path::Path;
    use std::time::Instant;

    let args: Vec<String> = env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!("usage: qwen36_cuda_decode_smoke <gguf_path> [n_steps=10] [start_token=1]");
        std::process::exit(2);
    }
    let gguf_path = Path::new(&args[0]);
    let n_steps: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(10);
    let start_token: u32 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(1);

    println!("[qwen36_cuda_decode_smoke] gguf={}", gguf_path.display());
    println!("[qwen36_cuda_decode_smoke] n_steps={n_steps} start_token={start_token}");

    let t0 = Instant::now();
    let mut model = Qwen35ModelCudaQ4K::from_gguf(gguf_path, n_steps + 8)
        .map_err(|e| format!("from_gguf: {e:?}"))?;
    let load_secs = t0.elapsed().as_secs_f64();
    println!("[qwen36_cuda_decode_smoke] loaded in {load_secs:.2}s");

    let mut tokens = vec![start_token];
    let mut cur = start_token;

    let t0 = Instant::now();
    for step in 0..n_steps {
        let next = match model.decode_step(cur) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("[qwen36_cuda_decode_smoke] decode_step failed at step {step}: {e:?}");
                std::process::exit(1);
            },
        };
        tokens.push(next);
        cur = next;
    }
    let elapsed = t0.elapsed().as_secs_f64();
    let tok_s = n_steps as f64 / elapsed;

    println!();
    println!("=== Qwen3.6-27B Q4_K_M decode smoke ===");
    println!("  total tokens : {}", tokens.len());
    println!("  decode time  : {elapsed:.3}s");
    println!("  speed        : {tok_s:.2} tok/s");
    println!("  llama.cpp tg128 reference : 11.73 tok/s");
    println!("  ratio        : {:.2}×", tok_s / 11.73);
    println!("  token ids    : {tokens:?}");

    Ok(())
}
