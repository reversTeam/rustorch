//! T246.10 A6 — batched prefill bench for `Qwen35ModelCudaQ4K::prefill_tokens`.
//!
//! Measures wall-clock prefill throughput at multiple prompt lengths and
//! compares to the naive baseline (looping `decode_step` N times). Outputs
//! a table with tok/s, speedup factor, and a comparison line to the
//! llama.cpp pp512 reference (2264 tok/s on the same hardware).
//!
//! Usage : `qwen36_cuda_prefill_bench <gguf_path> [n_default=128]`
//!
//! For each N in `{32, 128, 256, 512, n_default}` (deduped, only ≤ 512) :
//! - Run prefill 3× wall-clock avg
//! - Run naive baseline (decode_step loop) 1× wall-clock
//! - Report : prefill tok/s, naive tok/s, speedup, vs-llama.cpp
//!
//! Production env recommended :
//!   `RUSTORCH_MOE_ASYNC=1 RUSTORCH_MOE_GRAPH=1 RUSTORCH_MOE_MEGA=1`.

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("[qwen36_cuda_prefill_bench] cuda feature is OFF");
}

#[cfg(feature = "cuda")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use rustorch_llm::qwen35_cuda_q4k::{Qwen35ModelCudaQ4K, MAX_TREE_SIZE};
    use std::env;
    use std::path::Path;
    use std::time::Instant;

    let args: Vec<String> = env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!(
            "usage: qwen36_cuda_prefill_bench <gguf_path> [n_default=128]\n\n\
             Recommended env :\n  \
             RUSTORCH_MOE_ASYNC=1 RUSTORCH_MOE_GRAPH=1 RUSTORCH_MOE_MEGA=1"
        );
        std::process::exit(2);
    }
    let gguf_path = Path::new(&args[0]);
    let n_default: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(128);

    // Pick the set of N values to bench. Dedup + sort + cap to MAX_TREE_SIZE.
    let mut ns: Vec<usize> = vec![32, 128, 256, 512, n_default]
        .into_iter()
        .filter(|n| *n >= 1 && *n <= MAX_TREE_SIZE)
        .collect();
    ns.sort();
    ns.dedup();
    let n_max = *ns.iter().max().unwrap_or(&n_default);

    println!("[prefill_bench] gguf={}", gguf_path.display());
    println!("[prefill_bench] N values = {ns:?}");
    println!("[prefill_bench] MAX_TREE_SIZE = {MAX_TREE_SIZE}");
    println!(
        "[prefill_bench] env : RUSTORCH_MOE_ASYNC={:?} RUSTORCH_MOE_GRAPH={:?} \
         RUSTORCH_MOE_MEGA={:?}",
        env::var("RUSTORCH_MOE_ASYNC").ok(),
        env::var("RUSTORCH_MOE_GRAPH").ok(),
        env::var("RUSTORCH_MOE_MEGA").ok()
    );

    // Load the model once with enough KV slots for the largest N + 8 decode.
    let t_load = Instant::now();
    let mut model = Qwen35ModelCudaQ4K::from_gguf(gguf_path, n_max + 16)
        .map_err(|e| format!("from_gguf: {e:?}"))?;
    println!(
        "[prefill_bench] loaded in {:.2}s",
        t_load.elapsed().as_secs_f64()
    );

    // Fixed deterministic prompt (BOS=1 + a chain of IDs). Real prompts
    // would be tokenized from text ; for bench we just need consistent
    // valid token IDs that exercise the embedding + decode pipeline.
    let make_prompt = |n: usize| -> Vec<u32> {
        let mut v = Vec::with_capacity(n);
        v.push(1u32); // BOS
        for i in 1..n {
            // Cheap deterministic mix : keeps IDs in a valid vocab range.
            v.push(((i * 31 + 7) % 10_000) as u32 + 100);
        }
        v
    };

    let llama_ref_pp512: f64 = 2264.0;
    println!("\n=== Qwen3.6 CUDA prefill bench ===");
    println!(
        "{:>6} | {:>14} | {:>14} | {:>10} | {:>14}",
        "N", "prefill tok/s", "naive tok/s", "speedup", "vs llama.cpp"
    );
    println!("{}", "-".repeat(70));

    for &n in &ns {
        let prompt = make_prompt(n);

        // ── A. prefill_tokens, 3-run avg, drop the slowest run ──────────
        let mut runs_ms: Vec<f64> = Vec::with_capacity(3);
        let mut first_tok: Option<u32> = None;
        for run in 0..3 {
            // Reset model state so each run starts fresh from position 0.
            model
                .reset_state()
                .map_err(|e| format!("reset N={n} run={run}: {e:?}"))?;
            let t0 = Instant::now();
            let tok = model
                .prefill_tokens(&prompt, 0)
                .map_err(|e| format!("prefill N={n} run={run}: {e:?}"))?;
            let dt_ms = t0.elapsed().as_secs_f64() * 1000.0;
            runs_ms.push(dt_ms);
            if first_tok.is_none() {
                first_tok = Some(tok);
            } else if first_tok != Some(tok) {
                eprintln!(
                    "[WARN] N={n}: prefill output drift across runs ({:?} vs {tok}) \
                     — non-determinism alarm",
                    first_tok
                );
            }
        }
        let mean_ms_prefill = runs_ms.iter().sum::<f64>() / runs_ms.len() as f64;
        let prefill_tok_s = (n as f64) / (mean_ms_prefill / 1000.0);

        // ── B. Naive baseline : decode_step loop, single run ────────────
        model
            .reset_state()
            .map_err(|e| format!("reset for naive N={n}: {e:?}"))?;
        let t0 = Instant::now();
        let mut cur = prompt[0];
        let _ = model
            .decode_step(cur)
            .map_err(|e| format!("naive step 0 N={n}: {e:?}"))?;
        for &t in &prompt[1..] {
            cur = model
                .decode_step(t)
                .map_err(|e| format!("naive step N={n}: {e:?}"))?;
        }
        let _ = cur;
        let dt_naive_s = t0.elapsed().as_secs_f64();
        let naive_tok_s = (n as f64) / dt_naive_s;

        let speedup = prefill_tok_s / naive_tok_s;
        let vs_llama = prefill_tok_s / llama_ref_pp512;

        println!(
            "{:>6} | {:>14.1} | {:>14.1} | {:>9.2}× | {:>13.2}×",
            n, prefill_tok_s, naive_tok_s, speedup, vs_llama
        );
    }
    println!("{}", "-".repeat(70));
    println!("llama.cpp pp512 reference (same hardware) : {llama_ref_pp512:.0} tok/s");
    println!("Target floors : pp32 ≥ 200, pp128 ≥ 500, pp512 ≥ 1000 tok/s (per A6 task brief)");

    Ok(())
}
