//! T246.7 TrackC.3 — smoke test for `decode_step_tree_hybrid` on Qwen3.6.
//!
//! Generates N tokens via `decode_step` (baseline). Then on a fresh model
//! generates N tokens via `decode_step_tree(drafts=[t0, t1], parents=[-1, 0],
//! depths=[0, 1])` where t1 is a wrong-guess token : the verify pass should
//! accept the root token (t0 = baseline next token) AND the model's own
//! prediction for the depth-1 position. The first accepted token MUST equal
//! the baseline's first token (since acceptance walk's root = argmax(root)).
//!
//! For tree_size=2 with intentionally-wrong child draft, accept_len should
//! be exactly 1 (root only) — the wrong child won't match the model's
//! argmax. This isolates that the hybrid forward produces a correct
//! root-position output even when SSM-state forking is exercised on the
//! depth-1 branch (which gets discarded).
//!
//! Usage : `qwen36_hybrid_tree_smoke <gguf_path> [n_steps] [seed_token]`

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("[hybrid-smoke] cuda feature is OFF");
}

#[cfg(feature = "cuda")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use rustorch_llm::qwen35_cuda_q4k::Qwen35ModelCudaQ4K;
    use std::env;
    use std::path::Path;
    use std::time::Instant;

    let args: Vec<String> = env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!("usage: qwen36_hybrid_tree_smoke <gguf_path> [n_steps] [seed]");
        std::process::exit(2);
    }
    let gguf_path = Path::new(&args[0]);
    let n_steps: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(8);
    let seed: u32 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(1);

    println!("[hybrid-smoke] gguf={}", gguf_path.display());
    println!("[hybrid-smoke] n_steps={n_steps} seed={seed}");

    // ── Path A : decode_step baseline ────────────────────────────────────
    println!("\n[A] baseline decode_step...");
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

    // ── Path B : decode_step_tree(tree_size=2) per step ──────────────────
    // Use intentionally-wrong child to force accept_len=1, root-only.
    println!("\n[B] hybrid decode_step_tree(tree_size=2)...");
    let mut model_b = Qwen35ModelCudaQ4K::from_gguf(gguf_path, n_steps + 16)
        .map_err(|e| format!("from_gguf B: {e:?}"))?;
    let mut tokens_b: Vec<u32> = Vec::with_capacity(n_steps);
    let mut cur = seed;
    let t0 = Instant::now();
    for step in 0..n_steps {
        // Tree : root = cur, child = an arbitrary "wrong" draft (=0).
        // The model's argmax for the root row is the prediction we want.
        // The child row's prediction is unused unless drafts[1] happens
        // to match argmax_host[0] — vanishingly unlikely with draft=0.
        let drafts = [cur, 0u32];
        let parents = [-1i32, 0];
        let depths = [0u8, 1];
        let accepted = model_b
            .decode_step_tree(&drafts, &parents, &depths)
            .map_err(|e| format!("decode_step_tree B step {step}: {e:?}"))?;
        if accepted.is_empty() {
            return Err(format!("step {step}: empty accepted").into());
        }
        cur = accepted[0]; // root's argmax
        tokens_b.push(cur);
        if accepted.len() > 1 {
            println!(
                "  step {step}: acceptance walked deeper than expected ({} accepted)",
                accepted.len()
            );
        }
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
            "\n[hybrid-smoke] PASS — hybrid decode_step_tree(tree_size=2) root \
             matches baseline ({n_steps} tokens)"
        );
        Ok(())
    } else {
        eprintln!("\n[hybrid-smoke] FAIL — tokens diverge");
        for (i, (a, b)) in tokens_a.iter().zip(tokens_b.iter()).enumerate() {
            if a != b {
                eprintln!("  step {i}: baseline={a} vs hybrid={b}");
            }
        }
        Err("hybrid decode_step_tree root output does not match baseline".into())
    }
}
