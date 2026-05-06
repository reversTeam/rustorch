//! Minimal Qwen3 inference via candle-rs Metal backend — used as a
//! perf baseline to compare against `rustorch-llm`.
//!
//! No `hf_hub`, no `tokenizer.json`: prompt is supplied as comma-
//! separated token IDs (use `llama-tokenize` to convert text first).
//!
//! Usage:
//! ```sh
//! cargo run -p rustorch-bench-candle --release --bin qwen3_candle -- \
//!     --model /Users/triviere/models/Qwen3-14B-Claude-4.5-Opus-Distill.q4_k_m.gguf \
//!     --prompt-ids 12522,5193,264,882,11,1052,572,264,2613,25105,879 \
//!     --n 50
//! ```
//!
//! ## What we measure
//!
//! - Prompt processing throughput (prefill: K tokens in parallel)
//! - Decode throughput (N - 1 tokens, autoregressive)
//!
//! These are the same numbers `rustorch-llm/examples/qwen_inference`
//! reports for an apples-to-apples comparison.

use anyhow::Result;
use candle_core::quantized::gguf_file;
use candle_core::Tensor;
use candle_transformers::generation::{LogitsProcessor, Sampling};
use candle_transformers::models::quantized_qwen3::ModelWeights as Qwen3;
use clap::Parser;
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to the Qwen3 Q4_K_M GGUF file.
    #[arg(long)]
    model: String,

    /// Prompt as comma-separated token IDs (already tokenised — bypass
    /// the need for a `tokenizer.json` next to the model).
    #[arg(long)]
    prompt_ids: String,

    /// Number of NEW tokens to generate (greedy/top-1).
    #[arg(long, default_value_t = 50)]
    n: usize,

    /// Force CPU even if Metal is available (default: Metal on macOS).
    #[arg(long, default_value_t = false)]
    cpu: bool,
}

fn pick_device(cpu: bool) -> Result<candle_core::Device> {
    if cpu {
        Ok(candle_core::Device::Cpu)
    } else if candle_core::utils::metal_is_available() {
        Ok(candle_core::Device::new_metal(0)?)
    } else {
        Ok(candle_core::Device::Cpu)
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    let prompt: Vec<u32> = args
        .prompt_ids
        .split(',')
        .map(|s| s.trim().parse::<u32>().unwrap())
        .collect();

    let device = pick_device(args.cpu)?;
    println!("device  : {:?}", device);
    println!("model   : {}", args.model);
    println!("prompt  : {} tokens : {:?}", prompt.len(), prompt);
    println!();

    // Load GGUF.
    let mut file = std::fs::File::open(&args.model)?;
    let t0 = Instant::now();
    let gguf = gguf_file::Content::read(&mut file).map_err(|e| anyhow::anyhow!("{e:?}"))?;
    let mut total_size = 0usize;
    for (_, t) in gguf.tensor_infos.iter() {
        let n = t.shape.elem_count();
        total_size += n * t.ggml_dtype.type_size() / t.ggml_dtype.block_size();
    }
    println!(
        "loaded {} tensors ({} MB on disk) in {:.2}s",
        gguf.tensor_infos.len(),
        total_size / 1_000_000,
        t0.elapsed().as_secs_f32()
    );

    let t1 = Instant::now();
    let mut model = Qwen3::from_gguf(gguf, &mut file, &device)?;
    println!("model built in {:.2}s", t1.elapsed().as_secs_f32());
    println!();

    // Prefill: forward the whole prompt at once.
    let mut all_tokens = prompt.clone();
    let mut logits_processor = LogitsProcessor::from_sampling(0, Sampling::ArgMax);

    let t_prefill = Instant::now();
    let input = Tensor::new(prompt.as_slice(), &device)?.unsqueeze(0)?;
    let logits = model.forward(&input, 0)?;
    let logits = logits.squeeze(0)?;
    let mut next_token = logits_processor.sample(&logits)?;
    let prefill_d = t_prefill.elapsed();
    all_tokens.push(next_token);

    // Decode: one token at a time, autoregressive.
    let t_decode = Instant::now();
    for index in 0..(args.n - 1) {
        let input = Tensor::new(&[next_token], &device)?.unsqueeze(0)?;
        let logits = model.forward(&input, prompt.len() + index)?;
        let logits = logits.squeeze(0)?;
        next_token = logits_processor.sample(&logits)?;
        all_tokens.push(next_token);
    }
    let decode_d = t_decode.elapsed();

    let prefill_tps = prompt.len() as f64 / prefill_d.as_secs_f64();
    let decode_tps = (args.n - 1) as f64 / decode_d.as_secs_f64();
    println!(
        "prefill : {:>4} tokens in {:.3}s => {:>6.2} tok/s",
        prompt.len(),
        prefill_d.as_secs_f64(),
        prefill_tps,
    );
    println!(
        "decode  : {:>4} tokens in {:.3}s => {:>6.2} tok/s",
        args.n - 1,
        decode_d.as_secs_f64(),
        decode_tps,
    );
    println!();
    println!("generated token IDs: {:?}", &all_tokens[prompt.len()..]);
    Ok(())
}
