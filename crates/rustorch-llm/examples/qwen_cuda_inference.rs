//! `qwen_cuda_inference` — T241.6c verification : load real Qwen GGUF
//! + tokenizer, run BF16 then NVFP4 forward on CUDA, decode token stream
//! into text and verify the output is coherent (not garbage).
//!
//! Usage :
//!   qwen_cuda_inference <gguf_path> <tokenizer_json_path> [--prompt "text"]
//!     [--max-new N] [--max-seq N] [--fp4]

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("[qwen_cuda_inference] cuda feature is OFF");
}

#[cfg(feature = "cuda")]
#[derive(Debug)]
struct InferErr(String);
#[cfg(feature = "cuda")]
impl std::fmt::Display for InferErr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}
#[cfg(feature = "cuda")]
impl std::error::Error for InferErr {}
#[cfg(feature = "cuda")]
impl From<rustorch_llm::LlmError> for InferErr {
    fn from(e: rustorch_llm::LlmError) -> Self {
        Self(e.to_string())
    }
}

#[cfg(feature = "cuda")]
fn argmax(xs: &[f32]) -> u32 {
    let mut best = 0u32;
    let mut bv = f32::NEG_INFINITY;
    for (i, &v) in xs.iter().enumerate() {
        if v > bv {
            bv = v;
            best = i as u32;
        }
    }
    best
}

#[cfg(feature = "cuda")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use rustorch_llm::{GgufWeights, LlamaModel, LlamaModelCuda};
    use rustorch_tokenizer::BpeTokenizer;
    use std::env;
    use std::time::Instant;

    let args: Vec<String> = env::args().skip(1).collect();
    if args.len() < 2 {
        eprintln!(
            "usage: qwen_cuda_inference <gguf_path> <tokenizer_json> \
             [--prompt \"text\"] [--max-new N] [--max-seq N] [--fp4]"
        );
        std::process::exit(2);
    }
    let gguf_path = args[0].clone();
    let tok_path = args[1].clone();
    let mut prompt_text = "Hello, who are you?".to_string();
    let mut max_new = 32usize;
    let mut max_seq = 256usize;
    let mut fp4 = false;
    let mut i = 2;
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
            "--fp4" => {
                fp4 = true;
                i += 1;
                continue;
            },
            _ => {
                i += 1;
            },
        }
    }

    println!("[qwen_cuda_inference] gguf={gguf_path}");
    println!("[qwen_cuda_inference] tokenizer={tok_path}");
    println!("[qwen_cuda_inference] prompt=\"{prompt_text}\"");
    println!("[qwen_cuda_inference] max_new={max_new} max_seq={max_seq} fp4={fp4}");

    // 1. Load tokenizer
    let tokenizer =
        BpeTokenizer::from_file(&tok_path).map_err(|e| InferErr(format!("tokenizer load: {e}")))?;
    let prompt_ids = tokenizer
        .encode(&prompt_text, false)
        .map_err(|e| InferErr(format!("encode: {e}")))?;
    println!(
        "[qwen_cuda_inference] prompt encoded → {} tokens",
        prompt_ids.len()
    );
    println!("                       ids={prompt_ids:?}");

    // 2. Load GGUF on CPU
    let t0 = Instant::now();
    let (cfg, weights) =
        GgufWeights::open(&gguf_path).map_err(|e| InferErr(format!("gguf open: {e}")))?;
    println!(
        "[qwen_cuda_inference] gguf loaded in {:.2}s — d={} layers={} vocab={}",
        t0.elapsed().as_secs_f64(),
        cfg.hidden_size,
        cfg.num_hidden_layers,
        cfg.vocab_size
    );
    let t1 = Instant::now();
    let cpu = LlamaModel::from_gguf(cfg.clone(), weights, max_seq)
        .map_err(|e| InferErr(format!("from_gguf: {e}")))?;
    println!(
        "[qwen_cuda_inference] CPU model built in {:.2}s",
        t1.elapsed().as_secs_f64()
    );

    // 3. Upload to CUDA
    let t2 = Instant::now();
    let mut cuda = LlamaModelCuda::from_cpu(cpu, max_seq)
        .map_err(|e| InferErr(format!("from_cpu CUDA: {e}")))?;
    println!(
        "[qwen_cuda_inference] uploaded to GPU in {:.2}s",
        t2.elapsed().as_secs_f64()
    );

    if fp4 {
        let t3 = Instant::now();
        cuda.enable_fp4()
            .map_err(|e| InferErr(format!("enable_fp4: {e}")))?;
        println!(
            "[qwen_cuda_inference] FP4 quantize done in {:.2}s",
            t3.elapsed().as_secs_f64()
        );
    }

    // 4. Prefill + decode
    cuda.reset_kv();
    let mut output_ids: Vec<u32> = prompt_ids.clone();
    let t_decode = Instant::now();
    // Prefill
    for &tok in &prompt_ids {
        if fp4 {
            cuda.decode_step_fp4(tok)
                .map_err(|e| InferErr(format!("decode_step_fp4 prefill: {e}")))?;
        } else {
            cuda.decode_step(tok)
                .map_err(|e| InferErr(format!("decode_step prefill: {e}")))?;
        }
    }
    // Decode max_new tokens greedy
    let mut last = *prompt_ids.last().expect("non-empty prompt");
    for _ in 0..max_new {
        let logits = cuda
            .last_logits()
            .map_err(|e| InferErr(format!("last_logits: {e}")))?;
        let next = argmax(&logits);
        output_ids.push(next);
        last = next;
        if fp4 {
            cuda.decode_step_fp4(last)
                .map_err(|e| InferErr(format!("decode_step_fp4: {e}")))?;
        } else {
            cuda.decode_step(last)
                .map_err(|e| InferErr(format!("decode_step: {e}")))?;
        }
    }
    let elapsed = t_decode.elapsed().as_secs_f64();
    let tok_s = max_new as f64 / elapsed;
    println!();
    println!(
        "[qwen_cuda_inference] generated {max_new} tokens in {elapsed:.2}s = {tok_s:.2} tok/s",
    );

    // 5. Decode + display
    let new_only = &output_ids[prompt_ids.len()..];
    let text = tokenizer
        .decode(&output_ids, true)
        .map_err(|e| InferErr(format!("decode: {e}")))?;
    let new_text = tokenizer
        .decode(new_only, true)
        .map_err(|e| InferErr(format!("decode new: {e}")))?;

    println!();
    println!("══════════════ GENERATED TEXT ══════════════");
    println!("{text}");
    println!("════════════════════════════════════════════");
    println!("New tokens only : {new_text:?}");
    println!("Token IDs       : {:?}", new_only);
    println!();
    println!(
        "Backend : {} ({} tok/s)",
        if fp4 { "CUDA NVFP4" } else { "CUDA BF16" },
        tok_s
    );

    Ok(())
}
