//! `qwen_synth_bench` — bench end-to-end Qwen forward CUDA sur DGX Spark.
//!
//! Mesure tok/s decode pour différentes shapes (Qwen-3.6-27B / 35B / 72B)
//! sans avoir besoin du modèle GGUF réel : les poids sont initialisés à
//! zéro, ce qui ne donne pas de logits sensés mais exerce TOUT le
//! pipeline CUDA (40-80 layers × attention + FFN matmuls + custom kernels)
//! avec les vraies shapes Qwen, donc le tok/s mesuré est représentatif.
//!
//! Comparable à llama.cpp pp8192 / tg128 sur DGX Spark.

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("[qwen_synth_bench] cuda feature is OFF");
}

#[cfg(feature = "cuda")]
#[derive(Debug)]
struct BenchErr(String);

#[cfg(feature = "cuda")]
impl std::fmt::Display for BenchErr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

#[cfg(feature = "cuda")]
impl std::error::Error for BenchErr {}

#[cfg(feature = "cuda")]
impl From<rustorch_llm::LlmError> for BenchErr {
    fn from(e: rustorch_llm::LlmError) -> Self {
        Self(format!("{e}"))
    }
}

#[cfg(feature = "cuda")]
fn main() -> Result<(), BenchErr> {
    use rustorch_llm::{LlamaConfig, LlamaModelCuda};
    use std::time::Instant;

    let model = std::env::var("RUSTORCH_BENCH_MODEL").unwrap_or_else(|_| "27b".into());

    // Qwen-3.6 reference shapes
    let cfg = match model.as_str() {
        "27b" | "35b" => LlamaConfig {
            hidden_size: 2048,
            num_attention_heads: 16,
            num_key_value_heads: Some(2),
            intermediate_size: 11008,
            num_hidden_layers: 40,
            vocab_size: 152064,
            max_position_embeddings: 8192,
            rms_norm_eps: 1e-6,
            rope_theta: 1_000_000.0,
            tie_word_embeddings: false,
        },
        "72b" => LlamaConfig {
            hidden_size: 8192,
            num_attention_heads: 64,
            num_key_value_heads: Some(8),
            intermediate_size: 29568,
            num_hidden_layers: 80,
            vocab_size: 152064,
            max_position_embeddings: 8192,
            rms_norm_eps: 1e-6,
            rope_theta: 1_000_000.0,
            tie_word_embeddings: false,
        },
        other => {
            return Err(BenchErr(format!(
                "unknown model `{other}`, use 27b|35b|72b"
            )))
        },
    };

    let max_seq: usize = std::env::var("RUSTORCH_BENCH_SEQ")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(512);
    let n_decode: usize = std::env::var("RUSTORCH_BENCH_DECODE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(64);
    let warmup: usize = 4;

    println!("[qwen_synth_bench] model={model}");
    println!(
        "[qwen_synth_bench]   hidden={}  ffn={}  n_heads={}  n_kv={}  head_dim={}",
        cfg.hidden_size,
        cfg.intermediate_size,
        cfg.num_attention_heads,
        cfg.n_kv_heads(),
        cfg.head_dim()
    );
    println!(
        "[qwen_synth_bench]   n_layers={}  vocab={}  max_seq={}  decode_steps={}",
        cfg.num_hidden_layers, cfg.vocab_size, max_seq, n_decode
    );

    println!("[qwen_synth_bench] allocating dummy weights on GPU...");
    let t_alloc = Instant::now();
    let mut cuda = LlamaModelCuda::from_dummy(cfg.clone(), max_seq)?;
    println!(
        "[qwen_synth_bench]   alloc done in {:.2}s",
        t_alloc.elapsed().as_secs_f64()
    );

    // Warmup
    println!("[qwen_synth_bench] warmup ({warmup} decode steps)...");
    cuda.reset_kv();
    for i in 0..warmup {
        let _ = cuda.decode_step(i as u32)?;
    }
    cuda.reset_kv();

    // Decode bench
    println!("[qwen_synth_bench] decode timing ({n_decode} steps)...");
    let t0 = Instant::now();
    for i in 0..n_decode {
        let _ = cuda.decode_step((i % cfg.vocab_size) as u32)?;
    }
    let elapsed = t0.elapsed().as_secs_f64();
    let per_step_ms = elapsed * 1000.0 / n_decode as f64;
    let tok_s = n_decode as f64 / elapsed;

    println!();
    println!("=================== Qwen-{model} BF16 CUDA decode (rustorch) ===================");
    println!("  per-step      : {per_step_ms:.2} ms");
    println!("  tokens/s      : {tok_s:.2}");
    println!("  total elapsed : {elapsed:.2} s ({n_decode} steps)");
    println!("================================================================");
    println!();
    println!("Reference (llama.cpp DGX Spark, Qwen-3.6-27B BF16) :");
    println!("  prefill pp8192 : 893 tok/s");
    println!("  decode  tg128  : 4.54 tok/s  ← our cible (BF16)");
    println!();

    Ok(())
}
