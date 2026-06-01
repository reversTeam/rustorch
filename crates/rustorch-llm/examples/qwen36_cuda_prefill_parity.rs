//! T246.10 A6 — parity test : `prefill_tokens(prompt)` must produce the
//! same first decode token as running `decode_step(prompt[i])` in a loop.
//!
//! For N ∈ {1, 8, 32, 128} and a fixed deterministic prompt :
//! 1. Path A : `let mut last = ?; for t in prompt { last = decode_step(t); }` →
//!    last is the model's argmax-after-N-tokens prediction.
//! 2. Path B : `let last = prefill_tokens(prompt, 0)?` → same prediction.
//! 3. Compare. Bit-identical (or 1 BF16 ULP per task acceptance criterion).
//!
//! Usage : `qwen36_cuda_prefill_parity <gguf_path>`

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("[prefill_parity] cuda feature is OFF");
}

#[cfg(feature = "cuda")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use rustorch_llm::qwen35_cuda_q4k::Qwen35ModelCudaQ4K;
    use std::env;
    use std::path::Path;

    let args: Vec<String> = env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!("usage: qwen36_cuda_prefill_parity <gguf_path>");
        std::process::exit(2);
    }
    let gguf_path = Path::new(&args[0]);

    println!("[prefill_parity] gguf={}", gguf_path.display());

    let ns: Vec<usize> = vec![1, 8, 32, 128];
    let n_max = *ns.iter().max().unwrap();

    let mut model = Qwen35ModelCudaQ4K::from_gguf(gguf_path, n_max + 16)
        .map_err(|e| format!("from_gguf: {e:?}"))?;

    // Deterministic prompt — same seed as bench so we can cross-reference.
    let make_prompt = |n: usize| -> Vec<u32> {
        let mut v = Vec::with_capacity(n);
        v.push(1u32);
        for i in 1..n {
            v.push(((i * 31 + 7) % 10_000) as u32 + 100);
        }
        v
    };

    let mut all_pass = true;
    for &n in &ns {
        let prompt = make_prompt(n);

        // ── Path A : decode_step loop ────────────────────────────────────
        model
            .reset_state()
            .map_err(|e| format!("reset A N={n}: {e:?}"))?;
        let mut last_a: u32 = 0;
        for &t in &prompt {
            last_a = model
                .decode_step(t)
                .map_err(|e| format!("decode A N={n}: {e:?}"))?;
        }

        // ── Path B : prefill_tokens(prompt) ──────────────────────────────
        model
            .reset_state()
            .map_err(|e| format!("reset B N={n}: {e:?}"))?;
        let last_b = model
            .prefill_tokens(&prompt, 0)
            .map_err(|e| format!("prefill B N={n}: {e:?}"))?;

        let pass = last_a == last_b;
        all_pass &= pass;
        println!(
            "N={:>4} : path A (decode_step×N) = {:>7}  | path B (prefill_tokens) = {:>7}  | {}",
            n,
            last_a,
            last_b,
            if pass {
                "PASS (bit-identical)"
            } else {
                "FAIL — divergence"
            }
        );

        if !pass {
            eprintln!(
                "  ⚠️  Parity FAIL at N={n} : decode_step loop and prefill_tokens \
                 produce different first decode tokens. \
                 This indicates either a kernel-numerics drift > 1 BF16 ULP or a \
                 logic bug in the tree forward (RoPE pos, KV append slot, \
                 SSM state forking, MoE per-row dispatch, etc)."
            );
        }
    }

    if all_pass {
        println!("\n[prefill_parity] PASS — all N ∈ {ns:?} match bit-identically.");
        Ok(())
    } else {
        Err("prefill_parity : at least one N diverged".into())
    }
}
