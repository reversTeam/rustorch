//! T246.7 P1.3d — smoke test for `Qwen35ModelCudaQ4K::decode_step_tree`.
//!
//! Verifies that `decode_step_tree(drafts=[seed], parents=[-1], depths=[0])`
//! called N times in a loop produces the SAME first N tokens as
//! `decode_step(seed)` called N times. This is the tree_size=1 degenerate
//! case ; it must be a strict drop-in replacement for `decode_step`.
//!
//! Usage : `qwen36_cuda_decode_step_tree_smoke <gguf_path> [n_steps] [seed_token]`

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("[smoke] cuda feature is OFF");
}

#[cfg(feature = "cuda")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use rustorch_llm::qwen35_cuda_q4k::Qwen35ModelCudaQ4K;
    use std::env;
    use std::path::Path;
    use std::time::Instant;

    let args: Vec<String> = env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!("usage: qwen36_cuda_decode_step_tree_smoke <gguf_path> [n_steps] [seed]");
        std::process::exit(2);
    }
    let gguf_path = Path::new(&args[0]);
    let n_steps: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(32);
    let seed: u32 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(1);

    println!("[smoke] gguf={}", gguf_path.display());
    println!("[smoke] n_steps={n_steps} seed={seed}");

    // ── Path A : decode_step baseline ────────────────────────────────────
    println!("\n[A] decode_step baseline...");
    let mut model_a = Qwen35ModelCudaQ4K::from_gguf(gguf_path, n_steps + 16)
        .map_err(|e| format!("from_gguf A: {e:?}"))?;
    let mut tokens_a: Vec<u32> = Vec::with_capacity(n_steps);
    let mut cur = seed;
    let t0 = Instant::now();
    for _ in 0..n_steps {
        cur = model_a
            .decode_step(cur)
            .map_err(|e| format!("decode_step A: {e:?}"))?;
        tokens_a.push(cur);
    }
    let dt_a = t0.elapsed();
    println!(
        "  A : {} tokens in {:.3}s = {:.3} tok/s",
        tokens_a.len(),
        dt_a.as_secs_f32(),
        tokens_a.len() as f32 / dt_a.as_secs_f32()
    );
    println!("  A tokens: {tokens_a:?}");
    drop(model_a);

    // ── Path B : decode_step_tree(tree_size=1) ───────────────────────────
    println!("\n[B] decode_step_tree(tree_size=1)...");
    let mut model_b = Qwen35ModelCudaQ4K::from_gguf(gguf_path, n_steps + 16)
        .map_err(|e| format!("from_gguf B: {e:?}"))?;
    let mut tokens_b: Vec<u32> = Vec::with_capacity(n_steps);
    let mut cur = seed;
    let t0 = Instant::now();
    for _ in 0..n_steps {
        let drafts = [cur];
        let parents = [-1i32];
        let depths = [0u16];
        let accepted = model_b
            .decode_step_tree(&drafts, &parents, &depths)
            .map_err(|e| format!("decode_step_tree B: {e:?}"))?;
        if accepted.len() != 1 {
            return Err(format!(
                "expected accepted.len()=1 for tree_size=1, got {}",
                accepted.len()
            )
            .into());
        }
        cur = accepted[0];
        tokens_b.push(cur);
    }
    let dt_b = t0.elapsed();
    println!(
        "  B : {} tokens in {:.3}s = {:.3} tok/s",
        tokens_b.len(),
        dt_b.as_secs_f32(),
        tokens_b.len() as f32 / dt_b.as_secs_f32()
    );
    println!("  B tokens: {tokens_b:?}");

    // ── Compare ──────────────────────────────────────────────────────────
    if tokens_a == tokens_b {
        println!(
            "\n[smoke] PASS — decode_step_tree(tree_size=1) ≡ decode_step ({n_steps} tokens match)"
        );
        let ratio = dt_b.as_secs_f32() / dt_a.as_secs_f32();
        println!(
            "[smoke] microbench : tree_size=1 vs decode_step ratio = {ratio:.3}× ({:+.1}%)",
            (ratio - 1.0) * 100.0
        );
        if (ratio - 1.0).abs() > 0.05 {
            eprintln!(
                "[smoke] WARN : tree path is more than 5% off baseline ({:+.1}%) — \
                 acceptable for now (overhead is the input validation + Vec alloc per call) \
                 but worth profiling.",
                (ratio - 1.0) * 100.0
            );
        }
        Ok(())
    } else {
        eprintln!("\n[smoke] FAIL — tokens diverge");
        for (i, (a, b)) in tokens_a.iter().zip(tokens_b.iter()).enumerate() {
            if a != b {
                eprintln!("  step {i}: decode_step={a} ≠ decode_step_tree={b}");
            }
        }
        Err("decode_step_tree(tree_size=1) does not match decode_step".into())
    }
}
