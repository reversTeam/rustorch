//! T246 regression test : run decode_step twice in the same process with the
//! same input, assert outputs are identical. Catches non-determinism bugs
//! (uninitialized memory reads, missed __syncthreads, race conditions).
//!
//! Usage : `qwen36_cuda_determinism_test <gguf_path>`

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("[determinism] cuda feature is OFF");
}

#[cfg(feature = "cuda")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use rustorch_llm::qwen35_cuda_q4k::Qwen35ModelCudaQ4K;
    use std::env;
    use std::path::Path;

    let args: Vec<String> = env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!("usage: qwen36_cuda_determinism_test <gguf_path>");
        std::process::exit(2);
    }
    let gguf_path = Path::new(&args[0]);
    let n_steps: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(5);
    let start_token: u32 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(1);

    println!("[determinism] gguf={}", gguf_path.display());
    println!("[determinism] n_steps={n_steps} start_token={start_token}");

    // ---- Run A : load + decode N tokens ----
    println!("\n[A] First decode pass...");
    let mut model_a = Qwen35ModelCudaQ4K::from_gguf(gguf_path, n_steps + 8)
        .map_err(|e| format!("from_gguf A: {e:?}"))?;
    let mut tokens_a = vec![start_token];
    let mut cur = start_token;
    for _ in 0..n_steps {
        cur = model_a
            .decode_step(cur)
            .map_err(|e| format!("decode A: {e:?}"))?;
        tokens_a.push(cur);
    }
    println!("  A tokens: {tokens_a:?}");
    drop(model_a);

    // ---- Run B : reload + decode N tokens (same input) ----
    println!("\n[B] Second decode pass (fresh model load)...");
    let mut model_b = Qwen35ModelCudaQ4K::from_gguf(gguf_path, n_steps + 8)
        .map_err(|e| format!("from_gguf B: {e:?}"))?;
    let mut tokens_b = vec![start_token];
    let mut cur = start_token;
    for _ in 0..n_steps {
        cur = model_b
            .decode_step(cur)
            .map_err(|e| format!("decode B: {e:?}"))?;
        tokens_b.push(cur);
    }
    println!("  B tokens: {tokens_b:?}");

    // ---- Compare ----
    println!();
    println!("=== DETERMINISM RESULT ===");
    if tokens_a == tokens_b {
        println!("  ✅ PASS — same input → same output across two passes.");
        Ok(())
    } else {
        println!("  ❌ FAIL — output diverges between identical passes.");
        let first_diff = tokens_a
            .iter()
            .zip(tokens_b.iter())
            .position(|(a, b)| a != b);
        println!(
            "  First mismatch at step {:?} : A={:?} vs B={:?}",
            first_diff, tokens_a, tokens_b
        );
        std::process::exit(1)
    }
}
