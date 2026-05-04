//! `qwen_inference` — load a real GGUF Qwen3 model and generate tokens.
//!
//! Usage:
//!
//!   cargo run --release -p rustorch-llm --example qwen_inference -- \
//!     ~/models/Qwen3-14B-Claude-4.5-Opus-Distill.q4_k_m.gguf
//!
//! Optional flags:
//!   --max-new <N>       : number of tokens to generate (default 16)
//!   --max-seq <N>       : KV-cache + RoPE table size (default 256)
//!   --prompt-ids 1,2,3  : decode from the given raw token ids (default [1])
//!
//! This example dequantizes the entire GGUF file to f32 in RAM —
//! expect ~8x of the file size as peak memory (a 8.4 GiB Q4_K_M model
//! becomes ~30 GiB of f32 weights). On-the-fly dequant is a follow-up.
//!
//! No tokenizer wrapper here yet — we feed raw token IDs and report
//! the raw token IDs out. The user can pipe through the GGUF
//! tokenizer (a follow-up step).

use std::env;
use std::process::ExitCode;
use std::time::Instant;

use rustorch_llm::{GgufWeights, LlamaConfig, LlamaModel};
use rustorch_nn::sampling::SamplingConfig;

fn humanize_bytes(b: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
    let mut v = b as f64;
    let mut u = 0;
    while v >= 1024.0 && u < UNITS.len() - 1 {
        v /= 1024.0;
        u += 1;
    }
    format!("{:.2} {}", v, UNITS[u])
}

fn main() -> ExitCode {
    let mut args: Vec<String> = env::args().skip(1).collect();
    let mut max_new: usize = 16;
    let mut max_seq: usize = 256;
    let mut prompt_ids: Vec<u32> = vec![1];

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--max-new" => {
                if let Some(v) = args.get(i + 1) {
                    max_new = v.parse().unwrap_or(max_new);
                    args.remove(i + 1);
                    args.remove(i);
                    continue;
                }
            },
            "--max-seq" => {
                if let Some(v) = args.get(i + 1) {
                    max_seq = v.parse().unwrap_or(max_seq);
                    args.remove(i + 1);
                    args.remove(i);
                    continue;
                }
            },
            "--prompt-ids" => {
                if let Some(v) = args.get(i + 1) {
                    prompt_ids = v
                        .split(',')
                        .filter_map(|s| s.trim().parse::<u32>().ok())
                        .collect();
                    if prompt_ids.is_empty() {
                        prompt_ids = vec![1];
                    }
                    args.remove(i + 1);
                    args.remove(i);
                    continue;
                }
            },
            _ => {},
        }
        i += 1;
    }

    if args.is_empty() {
        eprintln!(
            "usage: qwen_inference <file.gguf> [--max-new N] [--max-seq N] [--prompt-ids 1,2,3]"
        );
        return ExitCode::from(2);
    }
    let path = &args[0];

    println!("→ opening {}", path);
    let t0 = Instant::now();
    let (cfg, weights) = match GgufWeights::open(path) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("error opening + dequantizing: {e}");
            return ExitCode::from(1);
        },
    };
    let load_secs = t0.elapsed().as_secs_f64();

    print_config(&cfg, weights.tied_lm_head, load_secs);

    println!("\n→ building model (transpose + RoPE table) ...");
    let t1 = Instant::now();
    let model = match LlamaModel::from_gguf(cfg, weights, max_seq) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("error building model: {e}");
            return ExitCode::from(1);
        },
    };
    let build_secs = t1.elapsed().as_secs_f64();
    println!(
        "  built in {build_secs:.2}s — running {max_new} decode steps from prompt {prompt_ids:?}"
    );

    // Diagnostic: print the top-10 logits for the first decode step
    // so we can sanity-check the output distribution.
    let top = model.debug_top_logits(&prompt_ids, max_seq, 10);
    println!("\n══════ TOP-10 LOGITS (after prefill) ══════");
    for (i, (id, l)) in top.iter().enumerate() {
        println!("  rank {:>2}: token_id={:>6}  logit={:>10.4}", i + 1, id, l);
    }
    let l_max = top.first().map(|x| x.1).unwrap_or(0.0);
    let l_min = top.last().map(|x| x.1).unwrap_or(0.0);
    println!("  spread top1-top10 = {:.4}", l_max - l_min);

    let sampling = SamplingConfig::greedy();
    let t2 = Instant::now();
    let new_tokens = model.generate(&prompt_ids, &sampling, max_new, max_seq);
    let gen_secs = t2.elapsed().as_secs_f64();

    println!("\n══════ GENERATED TOKENS ══════");
    println!(
        "{:?}  ({} tokens in {:.2}s = {:.2} tok/s)",
        new_tokens,
        new_tokens.len(),
        gen_secs,
        new_tokens.len() as f64 / gen_secs
    );

    ExitCode::SUCCESS
}

fn print_config(cfg: &LlamaConfig, tied: bool, load_secs: f64) {
    let kv_dim = cfg.n_kv_heads() * cfg.head_dim();
    let est_f32_bytes = (cfg.vocab_size * cfg.hidden_size  // embed
        + cfg.vocab_size * cfg.hidden_size                  // lm_head (or tied)
        + cfg.hidden_size                                   // final_norm
        + cfg.num_hidden_layers
            * (cfg.hidden_size                              // attn_norm
                + cfg.hidden_size * cfg.hidden_size         // q
                + 2 * kv_dim * cfg.hidden_size              // k, v
                + cfg.hidden_size * cfg.hidden_size         // o
                + cfg.hidden_size                           // ffn_norm
                + 2 * cfg.intermediate_size * cfg.hidden_size // gate, up
                + cfg.intermediate_size * cfg.hidden_size)) // down
        * 4;

    println!("\n══════ MODEL CONFIG ══════");
    println!(
        "  hidden_size        : {} (head_dim={}, n_heads={}, n_kv={})",
        cfg.hidden_size,
        cfg.head_dim(),
        cfg.num_attention_heads,
        cfg.n_kv_heads()
    );
    println!("  num_hidden_layers  : {}", cfg.num_hidden_layers);
    println!("  intermediate_size  : {}", cfg.intermediate_size);
    println!("  vocab_size         : {}", cfg.vocab_size);
    println!(
        "  context_length     : {}  (rope_theta={})",
        cfg.max_position_embeddings, cfg.rope_theta
    );
    println!(
        "  rms_norm_eps       : {:e}    tie_word_embeddings={}",
        cfg.rms_norm_eps, tied
    );
    println!(
        "  estimated f32 RAM  : {} (load+dequant took {:.2}s)",
        humanize_bytes(est_f32_bytes as u64),
        load_secs
    );
}
