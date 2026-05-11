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
    let seed: u32 = args.get(2).map(|s| s.parse().unwrap_or(1)).unwrap_or(1);

    println!("[qwen36_nvfp4_smoke] dir={}", dir.display());
    println!("[qwen36_nvfp4_smoke] N={n_tokens} seed={seed}");

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
    let mut tokens: Vec<u32> = Vec::with_capacity(n_tokens + 1);
    tokens.push(seed);
    let mut current = seed;

    println!("[qwen36_nvfp4_smoke] decoding {n_tokens} tokens from seed={seed}...");
    let mut warmup_secs = 0.0_f64;
    let t_decode = Instant::now();
    for i in 0..n_tokens {
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
    let post_warmup_tok = (n_tokens - 1).max(1) as f64;
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
