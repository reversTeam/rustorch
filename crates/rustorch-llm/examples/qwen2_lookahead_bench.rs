//! T246.7 P1.4b — Lookahead Decoding bench harness for Qwen pure-transformer
//! models (target : Qwen2.5-7B-Instruct Q4_K_M).
//!
//! Runs two decode loops back-to-back over the SAME model state and
//! compares outputs + wall-clock :
//!
//! - Baseline (`RUSTORCH_LOOKAHEAD=0` or unset) : N calls to
//!   `decode_step(token)`, recording the token sequence and decode time.
//! - Lookahead (`RUSTORCH_LOOKAHEAD=1`) : `LookaheadManager` builds a
//!   `DraftTree` each step ; on `tree_size > 1` the verify pass calls
//!   `decode_step_tree(drafts, parents, depths)` and records 1..L
//!   accepted tokens per step.
//!
//! Greedy-argmax acceptance theorem : the accepted continuation MUST be
//! bit-identical to the baseline up to the first divergence, so the
//! first N tokens of both runs are checked for parity.
//!
//! Usage : `qwen2_lookahead_bench <gguf_path> [n_steps=64] [seed=1] [W=5] [L=5]`
//!
//! Env vars :
//! - `RUSTORCH_LOOKAHEAD` : 0 (default) → baseline only, 1 → both.
//! - `RUSTORCH_LOOKAHEAD_W`, `RUSTORCH_LOOKAHEAD_L` : window dims (also
//!   accepted as positional CLI args).
//!
//! P1.4 KNOWN GAP : `Qwen35ModelCudaQ4K::from_gguf` does not yet parse
//! the `qwen2` GGUF arch (Qwen2.5 family) — see decision 64cd3ccd. Until
//! that lands, this bench can only be exercised against
//! `Qwen35Variant::Qwen3PureTransformer` checkpoints (none available on
//! DGX today). The wiring is complete and the algorithm is in place ;
//! pulling the actual numbers requires the Qwen2 loader to land first.

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("[lookahead-bench] cuda feature is OFF");
}

#[cfg(feature = "cuda")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use rustorch_llm::lookahead::LookaheadManager;
    use rustorch_llm::qwen35_cuda_q4k::Qwen35ModelCudaQ4K;
    use std::env;
    use std::path::Path;
    use std::time::Instant;

    let args: Vec<String> = env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!("usage: qwen2_lookahead_bench <gguf_path> [n_steps=64] [seed=1] [W=5] [L=5]");
        std::process::exit(2);
    }
    let gguf_path = Path::new(&args[0]);
    let n_steps: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(64);
    let seed: u32 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(1);
    let w: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(5);
    let l: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(5);

    let lookahead_on = std::env::var("RUSTORCH_LOOKAHEAD")
        .ok()
        .map(|v| v != "0" && !v.is_empty())
        .unwrap_or(false);

    println!("[lookahead-bench] gguf={}", gguf_path.display());
    println!(
        "[lookahead-bench] n_steps={n_steps} seed={seed} W={w} L={l} lookahead={}",
        if lookahead_on { "ON" } else { "OFF" }
    );

    // ── Path A : baseline decode_step ────────────────────────────────────
    println!("\n[A] baseline decode_step ...");
    // Allow extra KV slots so we don't tip over max_seq when Lookahead
    // explores a deeper tree.
    let max_seq_pad = n_steps.saturating_add(w * l + 16);
    let mut model_a = Qwen35ModelCudaQ4K::from_gguf(gguf_path, max_seq_pad)
        .map_err(|e| format!("from_gguf A: {e:?}"))?;
    let mut tokens_a: Vec<u32> = Vec::with_capacity(n_steps);
    let mut cur = seed;
    let t0 = Instant::now();
    for step in 0..n_steps {
        cur = model_a
            .decode_step(cur)
            .map_err(|e| format!("decode_step A step {step}: {e:?}"))?;
        tokens_a.push(cur);
    }
    let dt_a = t0.elapsed();
    let tok_s_a = tokens_a.len() as f64 / dt_a.as_secs_f64();
    println!(
        "  A : {} tokens in {:.3}s = {:.3} tok/s",
        tokens_a.len(),
        dt_a.as_secs_f64(),
        tok_s_a
    );
    println!(
        "  A tokens (first 16) : {:?}",
        &tokens_a[..tokens_a.len().min(16)]
    );
    drop(model_a);

    if !lookahead_on {
        println!("\n[lookahead-bench] RUSTORCH_LOOKAHEAD not set — baseline-only run complete.");
        return Ok(());
    }

    // ── Path B : Lookahead decode_step_tree loop ──────────────────────────
    println!("\n[B] Lookahead (W={w}, L={l}) ...");
    let mut model_b = Qwen35ModelCudaQ4K::from_gguf(gguf_path, max_seq_pad)
        .map_err(|e| format!("from_gguf B: {e:?}"))?;

    let mut mgr = LookaheadManager::new(w, l);
    let mut tokens_b: Vec<u32> = Vec::with_capacity(n_steps);
    let mut cur = seed;

    let t0 = Instant::now();
    let mut total_drafts: u64 = 0;
    let mut total_accepted_drafts: u64 = 0;
    let mut steps_taken: u64 = 0;
    while tokens_b.len() < n_steps {
        let tree = mgr.build_drafts(cur);
        let tree_size = tree.tokens.len();
        if tree_size == 1 {
            // Cold-start or empty pool : standard decode_step.
            let next = model_b
                .decode_step(cur)
                .map_err(|e| format!("decode_step B (tree=1): {e:?}"))?;
            tokens_b.push(next);
            // Feed Lookahead manager the validated token (root + 1 generated).
            mgr.record_acceptance(&[], 0, &[cur, next]);
            cur = next;
        } else {
            // tree_size > 1 : Lookahead verify pass.
            let drafts: &[u32] = &tree.tokens;
            let accepted = model_b
                .decode_step_tree(drafts, &tree.parents, &tree.depths)
                .map_err(|e| format!("decode_step_tree B (tree={tree_size}): {e:?}"))?;
            if accepted.is_empty() {
                return Err("decode_step_tree returned empty accepted vec".into());
            }
            // Push accepted tokens up to n_steps.
            let take = accepted.len().min(n_steps - tokens_b.len());
            tokens_b.extend(accepted.iter().take(take));
            // Update Lookahead state. `generated` per the contract = root +
            // accepted continuation. drafts (excluding root) is what was
            // submitted ; accepted_count is the number of accepted draft
            // tokens. The first accepted_count entries of `accepted` (after
            // the root) align with the accepted draft path.
            let drafts_excl_root: &[u32] = &drafts[1..];
            // The "accepted draft count" : we accepted accept_len tokens but
            // the first one is the model's own prediction at the root (not
            // a draft). So accepted_drafts = accept_len - 1.
            let accepted_drafts = accepted.len().saturating_sub(1);
            total_drafts += drafts_excl_root.len() as u64;
            total_accepted_drafts += accepted_drafts as u64;
            // generated for the manager : [root_token, then the accepted tokens].
            let mut generated = Vec::with_capacity(1 + accepted.len());
            generated.push(cur);
            generated.extend(accepted.iter().copied());
            mgr.record_acceptance(drafts_excl_root, accepted_drafts, &generated);
            cur = *accepted.last().unwrap();
        }
        steps_taken += 1;
    }
    let dt_b = t0.elapsed();
    let tok_s_b = tokens_b.len() as f64 / dt_b.as_secs_f64();
    let stats = mgr.stats();
    println!(
        "  B : {} tokens in {:.3}s = {:.3} tok/s ({} verify steps)",
        tokens_b.len(),
        dt_b.as_secs_f64(),
        tok_s_b,
        steps_taken
    );
    println!(
        "  B tokens (first 16) : {:?}",
        &tokens_b[..tokens_b.len().min(16)]
    );
    println!(
        "  Lookahead stats : draft_acceptance_rate={:.3} pool_hit_rate={:.3} \
         total_drafts={} accepted_drafts={} pool_size={}",
        if total_drafts > 0 {
            total_accepted_drafts as f64 / total_drafts as f64
        } else {
            0.0
        },
        stats.pool_hit_rate(),
        total_drafts,
        total_accepted_drafts,
        mgr.pool().len(),
    );
    drop(model_b);

    // ── Parity check ─────────────────────────────────────────────────────
    let n_check = tokens_a.len().min(tokens_b.len());
    let mut diverge: Option<usize> = None;
    for i in 0..n_check {
        if tokens_a[i] != tokens_b[i] {
            diverge = Some(i);
            break;
        }
    }
    println!("\n[parity] checking first {n_check} tokens ...");
    match diverge {
        None => {
            println!("[parity] PASS — first {n_check} tokens bit-identical");
        },
        Some(i) => {
            eprintln!(
                "[parity] FAIL at index {i} : baseline={} lookahead={}",
                tokens_a[i], tokens_b[i]
            );
            eprintln!(
                "  baseline next 5  : {:?}",
                &tokens_a[i..(i + 5).min(tokens_a.len())]
            );
            eprintln!(
                "  lookahead next 5 : {:?}",
                &tokens_b[i..(i + 5).min(tokens_b.len())]
            );
            return Err(format!("parity check failed at token {i}").into());
        },
    }

    // ── Bench summary ────────────────────────────────────────────────────
    let speedup = tok_s_b / tok_s_a;
    println!("\n=== Lookahead Decoding bench summary ===");
    println!("  baseline   : {tok_s_a:.3} tok/s");
    println!("  lookahead  : {tok_s_b:.3} tok/s ({steps_taken} verify steps)");
    println!(
        "  speedup    : {speedup:.3}× ({:+.1}%)",
        (speedup - 1.0) * 100.0
    );
    println!(
        "  W={w} L={l} draft_acceptance_rate={:.3} pool_hit_rate={:.3}",
        if total_drafts > 0 {
            total_accepted_drafts as f64 / total_drafts as f64
        } else {
            0.0
        },
        stats.pool_hit_rate()
    );
    if speedup >= 1.4 {
        println!("[lookahead-bench] PASS : speedup ≥ 1.4× target");
    } else {
        println!(
            "[lookahead-bench] WARN : speedup {speedup:.3}× below 1.4× target — \
             expected for cold-start or low-pool-hit traffic ; investigate \
             (W,L) sweep / longer runs."
        );
    }

    Ok(())
}
