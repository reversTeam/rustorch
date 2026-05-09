//! `qwen35_cpu_inference` — bench CPU réel pour Qwen3.5/3.6 hybrid
//! (transformer + Gated DeltaNet SSM). Charge un GGUF, encode un prompt
//! avec le tokenizer, run forward_token en boucle, décode l'output.
//!
//! Usage :
//!   qwen35_cpu_inference <gguf_path> <tokenizer.json>
//!     [--prompt "text"] [--max-new N] [--max-seq N]

use rustorch_llm::qwen35::{describe_model, parse_config, Qwen35Variant};
use rustorch_llm::qwen35_cpu::{forward_token, load_weights, Qwen35State};
use rustorch_tokenizer::BpeTokenizer;
use std::env;
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

fn main() -> ExitCode {
    let mut args: Vec<String> = env::args().skip(1).collect();
    if args.len() < 2 {
        eprintln!(
            "usage: qwen35_cpu_inference <gguf> <tokenizer.json> \
             [--prompt \"text\"] [--max-new N] [--max-seq N]"
        );
        return ExitCode::from(2);
    }
    let gguf_path = PathBuf::from(args.remove(0));
    let tok_path = PathBuf::from(args.remove(0));
    let mut prompt_text = "Hello, who are you?".to_string();
    let mut max_new: usize = 16;
    let mut max_seq: usize = 256;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--prompt" => {
                if let Some(v) = args.get(i + 1) {
                    prompt_text = v.clone();
                    i += 2;
                    continue;
                }
            },
            "--max-new" => {
                if let Some(v) = args.get(i + 1) {
                    max_new = v.parse().unwrap_or(max_new);
                    i += 2;
                    continue;
                }
            },
            "--max-seq" => {
                if let Some(v) = args.get(i + 1) {
                    max_seq = v.parse().unwrap_or(max_seq);
                    i += 2;
                    continue;
                }
            },
            _ => {
                i += 1;
            },
        }
    }
    println!("[qwen35_cpu_inference] gguf={}", gguf_path.display());
    println!("[qwen35_cpu_inference] tokenizer={}", tok_path.display());
    println!("[qwen35_cpu_inference] prompt=\"{prompt_text}\" max_new={max_new} max_seq={max_seq}");

    // 1. Parse GGUF metadata into Qwen35Config.
    let t0 = Instant::now();
    let cfg = match parse_config(&gguf_path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("error parsing config: {e:?}");
            return ExitCode::from(1);
        },
    };
    println!(
        "[qwen35_cpu_inference] config parsed in {:.2}s — variant={:?}",
        t0.elapsed().as_secs_f64(),
        cfg.variant
    );
    println!();
    println!("{}", describe_model(&cfg));
    println!();

    // 2. Load tensors (full weight materialization to f32).
    let t1 = Instant::now();
    let weights = match load_weights(&gguf_path, &cfg) {
        Ok(w) => w,
        Err(e) => {
            eprintln!("error loading weights: {e:?}");
            return ExitCode::from(1);
        },
    };
    println!(
        "[qwen35_cpu_inference] weights loaded in {:.2}s",
        t1.elapsed().as_secs_f64()
    );

    // 3. Load tokenizer & encode prompt.
    let tokenizer = match BpeTokenizer::from_file(&tok_path) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("error loading tokenizer: {e}");
            return ExitCode::from(1);
        },
    };
    let prompt_ids = match tokenizer.encode(&prompt_text, false) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("error encoding prompt: {e}");
            return ExitCode::from(1);
        },
    };
    println!(
        "[qwen35_cpu_inference] prompt → {} tokens : {prompt_ids:?}",
        prompt_ids.len()
    );

    if matches!(cfg.variant, Qwen35Variant::Moe) {
        eprintln!("⚠ NOTE : MoE FFN forward is currently a STUB (zeros). 35B-A3B output will be incoherent until T142 lands.");
    }

    // 4. Prefill : run forward_token on each prompt token.
    let mut state = Qwen35State::new(&cfg, max_seq);
    let t_prefill = Instant::now();
    let mut last_id: u32 = prompt_ids[0];
    for (pos, &tok) in prompt_ids.iter().enumerate() {
        last_id = forward_token(&weights, &mut state, tok, pos);
    }
    let prefill_secs = t_prefill.elapsed().as_secs_f64();
    println!(
        "[qwen35_cpu_inference] prefill {} tokens in {:.2}s ({:.2} tok/s)",
        prompt_ids.len(),
        prefill_secs,
        prompt_ids.len() as f64 / prefill_secs
    );

    // 5. Decode : autoregressive generation.
    let mut out_ids = prompt_ids.clone();
    let t_decode = Instant::now();
    for step in 0..max_new {
        let pos = prompt_ids.len() + step;
        last_id = forward_token(&weights, &mut state, last_id, pos);
        out_ids.push(last_id);
    }
    let decode_secs = t_decode.elapsed().as_secs_f64();
    let tok_s = max_new as f64 / decode_secs;
    println!(
        "[qwen35_cpu_inference] decode {max_new} tokens in {decode_secs:.2}s = {tok_s:.2} tok/s"
    );

    // 6. Show generated text.
    let new_text = match tokenizer.decode(&out_ids[prompt_ids.len()..], true) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("error decoding: {e}");
            return ExitCode::from(1);
        },
    };
    let full_text = match tokenizer.decode(&out_ids, true) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("error decoding full: {e}");
            return ExitCode::from(1);
        },
    };

    println!();
    println!("══════════ GENERATED TEXT ══════════");
    println!("{full_text}");
    println!("══════════════════════════════════");
    println!("New tokens : {new_text:?}");
    println!("Token IDs  : {:?}", &out_ids[prompt_ids.len()..]);
    println!();
    println!("Backend : CPU (forward_token)   throughput : {tok_s:.2} tok/s");

    ExitCode::SUCCESS
}
