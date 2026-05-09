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

    // Reference shapes — keys "27b"/"35b-a3b" are LEGACY proxies (Qwen2-style).
    // For real Qwen-3.6 architecture, use "qwen3.6-27b" or "qwen3.6-35b-a3b".
    //
    // CAVEAT for Qwen-3.6 :
    //   - Hybrid architecture : 75% layers are "linear_attention"
    //     (Mamba/SSM-style), only 25% are "full_attention" (transformer).
    //     Our bench models the FULL transformer block compute for ALL n_layers,
    //     which OVERESTIMATES the real cost (linear attention is cheaper).
    //   - 35B-A3B is MoE (256 experts, 8 active). We compute the dense
    //     equivalent (8 active experts × 512 = 4096 ffn) per layer,
    //     ignoring router overhead.
    //   - head_dim = 256 in Qwen-3.6 (vs 128 in Qwen-2). We adjust hidden
    //     to keep n_heads × head_dim = hidden when needed.
    let cfg = match model.as_str() {
        "27b" => LlamaConfig {
            // Legacy Qwen-2.5-27B dense (kept for backward bench comparison)
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
        "35b-a3b" | "35b" => LlamaConfig {
            // Legacy Qwen-2.5-style 35B-A3B proxy (NOT Qwen-3.6 hybrid)
            hidden_size: 2048,
            num_attention_heads: 32,
            num_key_value_heads: Some(4),
            intermediate_size: 4608,
            num_hidden_layers: 40,
            vocab_size: 248320,
            max_position_embeddings: 8192,
            rms_norm_eps: 1e-6,
            rope_theta: 1_000_000.0,
            tie_word_embeddings: false,
        },
        "qwen3.6-27b" => LlamaConfig {
            // Qwen-3.6-27B (real architecture, dense equivalent for bench).
            // Real config : hidden=5120, n_heads=24, n_kv=4, head_dim=256.
            // We adjust hidden to 6144 (= 24×256) so head_dim() stays 256.
            // ffn = 17408 (real) ; n_layers = 64 (real, includes 75%
            // linear_attention which is OVERESTIMATED here as full attn).
            hidden_size: 6144,
            num_attention_heads: 24,
            num_key_value_heads: Some(4),
            intermediate_size: 17408,
            num_hidden_layers: 64,
            vocab_size: 248320,
            max_position_embeddings: 8192,
            rms_norm_eps: 1e-6,
            rope_theta: 1_000_000.0,
            tie_word_embeddings: false,
        },
        "qwen3.6-35b-a3b" => LlamaConfig {
            // Qwen-3.6-35B-A3B MoE (256 experts, 8 active per token).
            // Real : hidden=2048, n_heads=16, head_dim=256 → adjusted hidden
            // to 4096 (= 16×256). ffn dense-equivalent = 8 × 512 = 4096.
            // n_layers=40 (real, 75% linear_attention overestimated as full).
            hidden_size: 4096,
            num_attention_heads: 16,
            num_key_value_heads: Some(2),
            intermediate_size: 4096, // = 8 active experts × 512
            num_hidden_layers: 40,
            vocab_size: 248320,
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
                "unknown model `{other}`, use 27b|35b-a3b|qwen3.6-27b|qwen3.6-35b-a3b|72b"
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

    let fp4_only_mode = std::env::var("RUSTORCH_BENCH_FP4_ONLY")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);

    println!(
        "[qwen_synth_bench] allocating dummy weights on GPU{}...",
        if fp4_only_mode {
            " (FP4-only mode, skip BF16)"
        } else {
            ""
        }
    );
    let t_alloc = Instant::now();
    let mut cuda = if fp4_only_mode {
        LlamaModelCuda::from_dummy_fp4_only(cfg.clone(), max_seq)?
    } else {
        LlamaModelCuda::from_dummy(cfg.clone(), max_seq)?
    };
    println!(
        "[qwen_synth_bench]   alloc done in {:.2}s",
        t_alloc.elapsed().as_secs_f64()
    );

    // Warmup BF16 (skip si fp4_only mode car les stubs invalident le path BF16)
    if !fp4_only_mode {
        println!("[qwen_synth_bench] warmup ({warmup} decode steps)...");
        cuda.reset_kv();
        for i in 0..warmup {
            let _ = cuda.decode_step(i as u32)?;
        }
        cuda.reset_kv();
    }

    // Decode bench BF16 (skip si fp4_only)
    let (per_step_bf16, tok_s_bf16) = if fp4_only_mode {
        (-1.0, -1.0) // sentinel
    } else {
        println!("[qwen_synth_bench] decode timing BF16 ({n_decode} steps)...");
        let t0 = Instant::now();
        for i in 0..n_decode {
            let _ = cuda.decode_step((i % cfg.vocab_size) as u32)?;
        }
        let elapsed = t0.elapsed().as_secs_f64();
        (
            elapsed * 1000.0 / n_decode as f64,
            n_decode as f64 / elapsed,
        )
    };

    // Optional FP4 path
    let do_fp4 = std::env::var("RUSTORCH_BENCH_FP4")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(true);
    let fp4_result = if do_fp4 {
        println!("[qwen_synth_bench] enabling FP4 weights (T241.5)...");
        let t_fp4_alloc = Instant::now();
        cuda.enable_fp4()?;
        println!(
            "  fp4 alloc done in {:.2}s",
            t_fp4_alloc.elapsed().as_secs_f64()
        );
        cuda.reset_kv();
        // Warmup FP4
        for i in 0..warmup {
            let _ = cuda.decode_step_fp4(i as u32)?;
        }
        cuda.reset_kv();
        println!("[qwen_synth_bench] decode timing NVFP4 ({n_decode} steps)...");
        let t0 = Instant::now();
        for i in 0..n_decode {
            let _ = cuda.decode_step_fp4((i % cfg.vocab_size) as u32)?;
        }
        let elapsed = t0.elapsed().as_secs_f64();
        let per_step = elapsed * 1000.0 / n_decode as f64;
        let tok_s = n_decode as f64 / elapsed;
        Some((per_step, tok_s, elapsed))
    } else {
        None
    };

    println!();
    println!("=========== Qwen-{model} CUDA decode (rustorch) ===========");
    if !fp4_only_mode {
        println!("  BF16  per-step : {per_step_bf16:>7.2} ms   tokens/s : {tok_s_bf16:>7.2}");
    } else {
        println!("  BF16            : skipped (fp4_only mode)");
    }
    if let Some((p, t, _e)) = fp4_result {
        if !fp4_only_mode && tok_s_bf16 > 0.0 {
            let actual_speedup = t / tok_s_bf16;
            println!(
                "  NVFP4 per-step : {p:>7.2} ms   tokens/s : {t:>7.2}   ({actual_speedup:.2}× BF16)"
            );
        } else {
            println!("  NVFP4 per-step : {p:>7.2} ms   tokens/s : {t:>7.2}");
        }
    }
    println!("===========================================================");
    println!();
    println!("Reference llama.cpp DGX Spark, Qwen-3.6-27B :");
    println!("  prefill BF16 pp8192 : 893 tok/s");
    println!("  decode  BF16 tg128  : 4.54 tok/s");
    println!("  decode  Q4_K_M tg128: 11.85 tok/s");
    println!();

    Ok(())
}
