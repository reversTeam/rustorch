//! `qwen_parity_test` — T241.6b lite : compare CPU forward vs CUDA
//! forward sur un small LlamaModel avec random weights identiques.
//!
//! Test : si les deux paths produisent les mêmes top-1 tokens (greedy),
//! le pipeline CUDA est numériquement validé contre le CPU reference.
//!
//! Le test utilise un small modèle (d=128, layers=2, vocab=256) pour
//! que le BF16 quantization noise reste petit. Avec seed fixe, les
//! weights random sont reproductibles entre les deux models.

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("[qwen_parity_test] cuda feature is OFF");
}

#[cfg(feature = "cuda")]
#[derive(Debug)]
struct PErr(String);

#[cfg(feature = "cuda")]
impl std::fmt::Display for PErr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

#[cfg(feature = "cuda")]
impl std::error::Error for PErr {}

#[cfg(feature = "cuda")]
impl From<rustorch_llm::LlmError> for PErr {
    fn from(e: rustorch_llm::LlmError) -> Self {
        Self(format!("{e}"))
    }
}

#[cfg(feature = "cuda")]
fn main() -> Result<(), PErr> {
    use rustorch_llm::{LlamaConfig, LlamaModel, LlamaModelCuda};
    use rustorch_nn::sampling::SamplingConfig;

    // Small model — kept tiny so BF16 noise impact is bounded.
    let cfg = LlamaConfig {
        hidden_size: 128,
        num_attention_heads: 4,
        num_key_value_heads: Some(2),
        intermediate_size: 256,
        num_hidden_layers: 2,
        vocab_size: 256,
        max_position_embeddings: 64,
        rms_norm_eps: 1e-6,
        rope_theta: 10_000.0,
        tie_word_embeddings: false,
    };
    let max_seq = 32usize;
    let seed = 42u64;
    let prompt = vec![7u32];
    let max_new = 4usize;

    println!(
        "[qwen_parity_test] config: d={} layers={} vocab={} head_dim={}",
        cfg.hidden_size,
        cfg.num_hidden_layers,
        cfg.vocab_size,
        cfg.head_dim()
    );
    println!("[qwen_parity_test] prompt={prompt:?} max_new={max_new} seed={seed}");

    // ===== CPU path =====
    println!("[qwen_parity_test] running CPU forward...");
    let model_cpu = LlamaModel::from_random(cfg.clone(), max_seq, seed);
    let cpu_tokens = model_cpu.generate(&prompt, &SamplingConfig::greedy(), max_new, max_seq);
    println!("  CPU tokens : {cpu_tokens:?}");

    // ===== CUDA path BF16 =====
    println!("[qwen_parity_test] running CUDA BF16 forward...");
    let model_for_cuda = LlamaModel::from_random(cfg.clone(), max_seq, seed);
    let mut cuda = LlamaModelCuda::from_cpu(model_for_cuda, max_seq)?;
    cuda.reset_kv();
    let mut cuda_tokens: Vec<u32> = prompt.clone();
    // Prefill : feed each prompt token (greedy argmax) — same prompt → idem
    for &tok in &prompt {
        let _ = cuda.decode_step(tok)?;
    }
    // Decode max_new tokens greedy
    let mut last = *prompt.last().unwrap();
    for _ in 0..max_new {
        let next = cuda.decode_step(last)?;
        cuda_tokens.push(next);
        last = next;
    }
    println!("  CUDA BF16 tokens : {cuda_tokens:?}");

    // ===== CUDA path NVFP4 =====
    println!("[qwen_parity_test] running CUDA NVFP4 forward...");
    cuda.enable_fp4()?;
    cuda.reset_kv();
    let mut fp4_tokens: Vec<u32> = prompt.clone();
    for &tok in &prompt {
        let _ = cuda.decode_step_fp4(tok)?;
    }
    let mut last = *prompt.last().unwrap();
    for _ in 0..max_new {
        let next = cuda.decode_step_fp4(last)?;
        fp4_tokens.push(next);
        last = next;
    }
    println!("  CUDA NVFP4 tokens: {fp4_tokens:?}");

    // ===== Compare =====
    let cpu_match = cpu_tokens == cuda_tokens;
    let fp4_match = cpu_tokens == fp4_tokens;
    let bf16_top1_match = cpu_tokens
        .iter()
        .skip(prompt.len())
        .zip(cuda_tokens.iter().skip(prompt.len()))
        .take_while(|(a, b)| a == b)
        .count();
    let fp4_top1_match = cpu_tokens
        .iter()
        .skip(prompt.len())
        .zip(fp4_tokens.iter().skip(prompt.len()))
        .take_while(|(a, b)| a == b)
        .count();

    println!();
    println!("=================== Parity Results ===================");
    println!(
        "  CPU vs CUDA-BF16  : full_match={cpu_match} prefix_match={bf16_top1_match}/{}",
        max_new
    );
    println!(
        "  CPU vs CUDA-NVFP4 : full_match={fp4_match} prefix_match={fp4_top1_match}/{}",
        max_new
    );
    println!("======================================================");
    println!();

    if !cpu_match {
        println!("⚠ BF16 path diverge — investigate kernel parity (RMSNorm, RoPE, GQA, ...)");
    }
    if !fp4_match {
        println!("ℹ FP4 path diverge attendu (quantization noise BF16→FP4)");
    }

    Ok(())
}
