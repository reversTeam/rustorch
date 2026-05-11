//! T246.9 NVFP4-FINISH — end-to-end NVFP4 decode smoke test.
//!
//! Loads `Qwen35ModelCudaNVFP4` from a HuggingFace `nvfp4-pack-quantized`
//! safetensors directory, runs `decode_step` for N tokens starting from
//! `seed_token`, prints the generated token IDs, and reports wall-clock
//! tok/s. The first call always JIT-compiles all kernels (slow) ; the
//! reported tok/s averages over the post-warmup tokens.
//!
//! ## Usage
//! ```
//! cargo run --release --features cuda -p rustorch-llm \
//!     --example qwen36_nvfp4_smoke -- \
//!     /home/triviere/projects/models/qwen3.6-35b-a3b-nvfp4 [N=32] [seed=1]
//! ```
//!
//! Env vars :
//!   - `RUSTORCH_MOE_GRAPH=0` : disable CUDA Graph capture (debug A/B)
//!   - `RUSTORCH_SSM_FUSE=0`  : disable fused SSM pre/post-step kernels
//!   - `RUSTORCH_NVFP4_LOAD_ONLY=1` : load + inventory + pipeline test only
//!   - `RUSTORCH_NVFP4_PREFILL_BENCH=1` : after the decode smoke, run a
//!     3-run wall-clock prefill bench at N ∈ {32, 128, 512} via the new
//!     `prefill_tokens` API (TrackK.2). Reports tok/s table.
//!
//! Per RFC b5fc8ead — coherence success = generated tokens look like
//! plausible text (no NaN, no all-zeros, no obvious repetition).

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("[qwen36_nvfp4_smoke] cuda feature is OFF");
}

#[cfg(feature = "cuda")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use cudarc::driver::CudaContext;
    use rustorch_llm::qwen35_cuda_nvfp4::{
        load_nvfp4_safetensors, qwen36_35b_a3b_nvfp4_config, Qwen35ModelCudaNVFP4,
    };
    use std::env;
    use std::path::Path;
    use std::time::Instant;

    let args: Vec<String> = env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!("usage: qwen36_nvfp4_smoke <safetensors_dir> [N=32] [seed=1]");
        std::process::exit(2);
    }
    let dir = Path::new(&args[0]);
    let n_tokens: usize = args.get(1).map(|s| s.parse().unwrap_or(32)).unwrap_or(32);
    let seed_arg: u32 = args.get(2).map(|s| s.parse().unwrap_or(1)).unwrap_or(1);

    // Optional multi-token prefill via `RUSTORCH_NVFP4_PROMPT_IDS=1,2,3,...`.
    // Useful for visual-coherence smoke tests where a single seed token does
    // not have enough context to predict English text (both Q4_K_M and
    // NVFP4 output multilingual subwords from a 1-token seed).
    let prompt_ids: Vec<u32> = match std::env::var("RUSTORCH_NVFP4_PROMPT_IDS") {
        Ok(s) if !s.is_empty() => s
            .split(',')
            .filter_map(|t| t.trim().parse::<u32>().ok())
            .collect(),
        _ => vec![seed_arg],
    };
    assert!(!prompt_ids.is_empty(), "prompt empty");

    println!("[qwen36_nvfp4_smoke] dir={}", dir.display());
    println!(
        "[qwen36_nvfp4_smoke] N={n_tokens} prompt_len={} prompt={:?}",
        prompt_ids.len(),
        prompt_ids
    );

    // --- Optional load-only fast path (legacy substrate validation) -----
    if std::env::var("RUSTORCH_NVFP4_LOAD_ONLY").ok().as_deref() == Some("1") {
        let ctx = CudaContext::new(0).map_err(|e| format!("ctx: {e:?}"))?;
        let stream = ctx.default_stream();
        let t0 = Instant::now();
        let weights = load_nvfp4_safetensors(dir, &stream)
            .map_err(|e| format!("load_nvfp4_safetensors: {e:?}"))?;
        let load_secs = t0.elapsed().as_secs_f64();
        println!("[load-only] {} tensors in {load_secs:.2}s", weights.len());
        println!("[load-only]   NVFP4 = {}", weights.nvfp4.len());
        println!("[load-only]   BF16  = {}", weights.bf16.len());
        return Ok(());
    }

    // --- Full model load ------------------------------------------------
    let cfg = qwen36_35b_a3b_nvfp4_config();
    let max_seq = 1024;
    println!(
        "[qwen36_nvfp4_smoke] loading model (40 layers, 256 experts, vocab={})...",
        cfg.vocab
    );
    let t_load = Instant::now();
    let mut model = Qwen35ModelCudaNVFP4::from_safetensors(dir, cfg, max_seq)
        .map_err(|e| format!("from_safetensors: {e:?}"))?;
    let load_secs = t_load.elapsed().as_secs_f64();
    println!("[qwen36_nvfp4_smoke] model loaded in {load_secs:.2}s");

    // --- Decode N tokens ------------------------------------------------
    let mut tokens: Vec<u32> = Vec::with_capacity(n_tokens + prompt_ids.len());

    // Prefill : push every prompt token through decode_step to grow the KV
    // cache + SSM state. Only the predicted next-token after the LAST
    // prompt token is kept as the first auto-regressive output.
    println!(
        "[qwen36_nvfp4_smoke] prefilling {} prompt token(s)...",
        prompt_ids.len()
    );
    let mut last_pred: u32 = 0;
    for (i, &pid) in prompt_ids.iter().enumerate() {
        tokens.push(pid);
        last_pred = model
            .decode_step(pid)
            .map_err(|e| format!("decode_step prefill #{i}: {e:?}"))?;
    }
    // First auto-regressive token = prediction after the last prompt token.
    tokens.push(last_pred);
    let mut current = last_pred;

    println!("[qwen36_nvfp4_smoke] decoding {n_tokens} tokens auto-regressively from {current}...");
    let mut warmup_secs = 0.0_f64;
    let t_decode = Instant::now();
    // We already produced 1 auto-regressive token in prefill ; decode N-1 more.
    let n_to_decode = n_tokens.saturating_sub(1);
    for i in 0..n_to_decode {
        let t_step = Instant::now();
        let next = model
            .decode_step(current)
            .map_err(|e| format!("decode_step #{i}: {e:?}"))?;
        let step_secs = t_step.elapsed().as_secs_f64();
        if i == 0 {
            warmup_secs = step_secs;
        }
        tokens.push(next);
        current = next;
        // Print every 4 tokens so we have visibility without spam.
        if i < 8 || i % 4 == 3 {
            println!("  [step {i:>3}] {step_secs:.3}s -> {next}");
        }
    }
    let total_secs = t_decode.elapsed().as_secs_f64();
    let post_warmup_secs = (total_secs - warmup_secs).max(1e-9);
    let post_warmup_tok = (n_to_decode.saturating_sub(1)).max(1) as f64;
    let tok_per_sec = post_warmup_tok / post_warmup_secs;

    println!();
    println!("=== Generated tokens (N={}) ===", tokens.len());
    println!("{:?}", tokens);
    println!();
    println!(
        "=== tok/s (post-warmup) : {tok_per_sec:.2} ({:.0} ms total, {:.0} ms warmup) ===",
        total_secs * 1000.0,
        warmup_secs * 1000.0
    );
    println!("=== Compare : Q4_K_M baseline 32.60 tok/s, llama.cpp NVFP4 ref ~63 tok/s ===");

    // --- Coherence sanity gates ----------------------------------------
    let n_distinct: std::collections::HashSet<u32> = tokens.iter().copied().collect();
    let max_run = max_consecutive_repeat(&tokens);
    println!();
    println!(
        "=== Coherence : {} distinct ids, longest run={max_run} ===",
        n_distinct.len()
    );

    let in_vocab = tokens.iter().all(|&t| t < 248320);
    if !in_vocab {
        return Err("FAIL : at least one token out of vocab range".into());
    }
    if n_distinct.len() < 4 {
        eprintln!("WARN : <4 distinct tokens — likely incoherent (loop)");
    }
    if max_run > n_tokens / 2 {
        eprintln!("WARN : single-token run > N/2 — likely incoherent");
    }

    // --- T246.10 TrackK.2 — optional prefill bench --------------------
    //
    // Activated by `RUSTORCH_NVFP4_PREFILL_BENCH=1`. Mirrors the structure
    // of `qwen36_cuda_prefill_bench` (TrackI bench harness for Q4_K) :
    // a deterministic token-id prompt of length N is fed through the new
    // `prefill_tokens` API and wall-clock tok/s is averaged over 3 runs
    // (post 2 warmup runs to amortise the captured-graph compile / first
    // launch). The "first decode token" returned by each run is compared
    // across runs as a non-determinism gate.
    if std::env::var("RUSTORCH_NVFP4_PREFILL_BENCH")
        .ok()
        .as_deref()
        == Some("1")
    {
        println!();
        println!("=== T246.10 TrackK.2 — NVFP4 prefill bench ===");
        let ns: Vec<usize> = vec![32, 128, 512];
        println!(
            "{:>6} | {:>15} | {:>14} | {:>14}",
            "N", "prefill tok/s", "mean ms", "first_tok"
        );
        println!("{}", "-".repeat(60));
        for &n in &ns {
            // Deterministic prompt : BOS=1, then a chain matching the
            // Q4_K bench harness (`qwen36_cuda_prefill_bench.rs`) so the
            // numbers are roughly comparable. Wraps to vocab=248320.
            let mut prompt: Vec<u32> = Vec::with_capacity(n);
            prompt.push(1u32);
            for i in 1..n {
                prompt.push(((i * 31 + 7) % 10_000) as u32 + 100);
            }

            let total_runs = 5;
            let warmup_runs = 2;
            let mut runs_ms: Vec<f64> = Vec::with_capacity(total_runs - warmup_runs);
            let mut first_tok: Option<u32> = None;
            for run in 0..total_runs {
                model
                    .reset_state()
                    .map_err(|e| format!("reset N={n} run={run}: {e:?}"))?;
                let t0 = Instant::now();
                let tok = model
                    .prefill_tokens(&prompt, 0)
                    .map_err(|e| format!("prefill N={n} run={run}: {e:?}"))?;
                let dt_ms = t0.elapsed().as_secs_f64() * 1000.0;
                if run >= warmup_runs {
                    runs_ms.push(dt_ms);
                }
                if first_tok.is_none() {
                    first_tok = Some(tok);
                } else if first_tok != Some(tok) {
                    eprintln!(
                        "[WARN] N={n}: prefill output drift across runs ({:?} vs {tok})",
                        first_tok
                    );
                }
            }
            let mean_ms = runs_ms.iter().sum::<f64>() / (runs_ms.len() as f64);
            let prefill_tok_s = (n as f64) / (mean_ms / 1000.0);
            println!(
                "{:>6} | {:>15.2} | {:>14.1} | {:>14}",
                n,
                prefill_tok_s,
                mean_ms,
                first_tok.unwrap_or(0)
            );
        }
        println!("{}", "-".repeat(60));
        println!("(NVFP4 prefill = sequential decode_step loop — TrackK.1 baseline ;");
        println!(" tree-batched GEMM-prefill port is TrackK.b future work.)");
    }

    println!();
    println!("[qwen36_nvfp4_smoke] DONE — coherent if token IDs above look sensible");
    Ok(())
}

#[cfg(feature = "cuda")]
fn max_consecutive_repeat(toks: &[u32]) -> usize {
    let mut best = 0usize;
    let mut cur = 0usize;
    let mut last: Option<u32> = None;
    for &t in toks {
        if Some(t) == last {
            cur += 1;
        } else {
            cur = 1;
        }
        last = Some(t);
        if cur > best {
            best = cur;
        }
    }
    best
}
