//! `qwen_parity_test` — T241.6b lite : compare CPU forward vs CUDA
//! forward sur un small LlamaModel avec random weights identiques.
//!
//! Test : si les deux paths produisent les mêmes top-1 tokens (greedy),
//! le pipeline CUDA est numériquement validé contre le CPU reference.
//!
//! Le test utilise un small modèle (d=128, layers=2, vocab=256) pour
//! que le BF16 quantization noise reste petit. Avec seed fixe, les
//! weights random sont reproductibles entre les deux models.
//!
//! ## Diagnostics
//!
//! En plus du greedy match, on dump aussi pour chaque step :
//!   - top-5 (token_id, logit) côté CPU
//!   - top-5 (token_id, logit) côté CUDA-BF16
//!   - cosine similarity entre les deux distributions complètes
//!
//! Cela permet de localiser EXACTEMENT à quel step la divergence
//! apparaît (utile pour bisect un bug kernel — RMSNorm overflow,
//! RoPE convention, GQA softmax, etc.).

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
fn topk(logits: &[f32], k: usize) -> Vec<(u32, f32)> {
    let mut idx: Vec<u32> = (0..logits.len() as u32).collect();
    idx.sort_by(|&a, &b| {
        logits[b as usize]
            .partial_cmp(&logits[a as usize])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    idx.into_iter()
        .take(k)
        .map(|i| (i, logits[i as usize]))
        .collect()
}

#[cfg(feature = "cuda")]
fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot = a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>();
    let na = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na * nb)
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
fn fmt_topk(top: &[(u32, f32)]) -> String {
    top.iter()
        .map(|(t, l)| format!("({t}, {l:+.3})"))
        .collect::<Vec<_>>()
        .join(" ")
}

#[cfg(feature = "cuda")]
fn main() -> Result<(), PErr> {
    use rustorch_llm::{LlamaConfig, LlamaModel, LlamaModelCuda};

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

    // ===== CPU path : full step-by-step logits =====
    println!("[qwen_parity_test] running CPU forward (step-by-step logits)...");
    let model_cpu = LlamaModel::from_random(cfg.clone(), max_seq, seed);
    let cpu_logits = model_cpu.forward_step_logits(&prompt, max_new, max_seq);
    println!(
        "  CPU collected {} step distributions (vocab={})",
        cpu_logits.len(),
        cpu_logits[0].len()
    );

    // ===== CUDA path BF16 : prefill + decode, capture logits at each step =====
    println!("[qwen_parity_test] running CUDA BF16 forward (capture logits)...");
    let model_for_cuda = LlamaModel::from_random(cfg.clone(), max_seq, seed);
    let mut cuda = LlamaModelCuda::from_cpu(model_for_cuda, max_seq)?;
    cuda.reset_kv();
    let mut cuda_logits: Vec<Vec<f32>> = Vec::with_capacity(prompt.len() + max_new);
    // Prefill : feed each prompt token, capture logits per step.
    for &tok in &prompt {
        let _ = cuda.decode_step(tok)?;
        cuda_logits.push(cuda.last_logits()?);
    }
    // Decode max_new tokens greedy from CUDA's own argmax.
    let mut last = argmax(cuda_logits.last().unwrap());
    for _ in 0..max_new {
        let _ = cuda.decode_step(last)?;
        let logits = cuda.last_logits()?;
        last = argmax(&logits);
        cuda_logits.push(logits);
    }

    // ===== Compare per-step =====
    println!();
    println!("============== Per-step parity report ==============");
    println!(
        "{:>4}  {:>14}  {:>16}  {}",
        "step", "cos(cpu,cuda)", "argmax cpu/cuda", "top-5 sample"
    );
    let mut all_match = true;
    for (i, (cpu_l, cuda_l)) in cpu_logits.iter().zip(cuda_logits.iter()).enumerate() {
        let cos = cosine(cpu_l, cuda_l);
        let cpu_top = topk(cpu_l, 5);
        let cuda_top = topk(cuda_l, 5);
        let cpu_argmax = cpu_top[0].0;
        let cuda_argmax = cuda_top[0].0;
        let match_flag = if cpu_argmax == cuda_argmax {
            "✓"
        } else {
            "✗"
        };
        if cpu_argmax != cuda_argmax {
            all_match = false;
        }
        println!(
            "{i:>4}  {cos:>14.6}  {cpu_argmax:>3}/{cuda_argmax:<3} {match_flag}    cpu={}",
            fmt_topk(&cpu_top)
        );
        println!(
            "                                       cuda={}",
            fmt_topk(&cuda_top)
        );
    }
    println!();

    // ===== Greedy token sequences =====
    let cpu_tokens: Vec<u32> = cpu_logits.iter().map(|l| argmax(l)).collect();
    let cuda_tokens: Vec<u32> = cuda_logits.iter().map(|l| argmax(l)).collect();
    println!("CPU  argmax sequence : {cpu_tokens:?}");
    println!("CUDA argmax sequence : {cuda_tokens:?}");

    let bf16_top1_match = cpu_tokens
        .iter()
        .zip(cuda_tokens.iter())
        .take_while(|(a, b)| a == b)
        .count();

    println!();
    println!("================ Final Result ================");
    println!(
        "  CPU vs CUDA-BF16 : full_match={all_match} prefix_match={bf16_top1_match}/{}",
        cpu_tokens.len()
    );
    println!("==============================================");

    Ok(())
}
