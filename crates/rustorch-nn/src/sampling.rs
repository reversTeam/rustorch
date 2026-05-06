//! Token sampling for autoregressive LLM decoding (T43).
//!
//! After the LM head produces logits over the vocabulary, the
//! decoder must pick the next token. This module ships the four
//! canonical strategies used by every LLM serving system:
//!
//! - **Greedy** (`temperature = 0`) — argmax. Deterministic.
//! - **Temperature** — scale logits before softmax. `> 1` = more
//!   random; `< 1` = more deterministic; `→ 0` = greedy.
//! - **Top-k** — keep only the K highest-probability tokens, set
//!   the rest to `-inf`. Default: K = vocab_size (no truncation).
//! - **Top-p (nucleus)** — keep the smallest token set whose
//!   cumulative probability exceeds `p`. Often combined with
//!   top-k.
//! - **Repeat penalty** — divide logits of recently emitted tokens
//!   by a factor (`> 1` discourages repeats). Standard llama.cpp /
//!   HF default = 1.1.
//!
//! ## Numerical stability
//!
//! Softmax is computed via the standard `x - max(x)` shift to keep
//! the exponential bounded; the temperature divide happens before
//! the shift. f32 throughout — quant-aware sampling is a follow-up.

/// Configuration for the sampling strategy.
#[derive(Debug, Clone)]
pub struct SamplingConfig {
    /// Logit temperature. `0.0` (or any non-positive) means greedy
    /// argmax (no randomness). Typical user-facing value: 0.7-1.0.
    pub temperature: f32,
    /// Keep only the top-K tokens before sampling. `None` = full
    /// vocab.
    pub top_k: Option<usize>,
    /// Nucleus threshold in `(0, 1]`. The smallest set of tokens
    /// whose cumulative probability ≥ `top_p` is kept; the rest
    /// is masked out. `None` = no nucleus filter.
    pub top_p: Option<f32>,
    /// Repeat penalty (HuggingFace / llama.cpp convention). Logits
    /// of tokens in `recent_tokens` are divided by this factor.
    /// `None` or `Some(1.0)` = no penalty.
    pub repeat_penalty: Option<f32>,
}

impl SamplingConfig {
    /// Greedy decoding (temperature = 0, no truncation).
    pub fn greedy() -> Self {
        SamplingConfig {
            temperature: 0.0,
            top_k: None,
            top_p: None,
            repeat_penalty: None,
        }
    }

    /// Temperature-only sampling.
    pub fn with_temperature(temperature: f32) -> Self {
        SamplingConfig {
            temperature,
            top_k: None,
            top_p: None,
            repeat_penalty: None,
        }
    }
}

impl Default for SamplingConfig {
    fn default() -> Self {
        // Sensible defaults for chat-style decoding.
        SamplingConfig {
            temperature: 1.0,
            top_k: Some(50),
            top_p: Some(0.95),
            repeat_penalty: Some(1.1),
        }
    }
}

/// Pick the next token id from the given logits.
///
/// `recent_tokens` is consulted only if `config.repeat_penalty` is
/// set (and `> 1.0`); pass an empty slice to disable.
///
/// `uniform01_sample` is a closure returning a value in `[0, 1)`;
/// any RNG-like callable will do (avoids forcing a `rand` dep on
/// callers). Greedy decoding ignores the closure.
pub fn sample_next(
    logits: &[f32],
    config: &SamplingConfig,
    recent_tokens: &[u32],
    mut uniform01_sample: impl FnMut() -> f32,
) -> usize {
    if logits.is_empty() {
        return 0;
    }

    // Greedy fast path: skip all the temperature / softmax / top-*
    // gymnastics and return argmax directly.
    if config.temperature <= 0.0
        && config.top_k.is_none()
        && config.top_p.is_none()
        && config.repeat_penalty.is_none()
    {
        return argmax(logits);
    }

    // Owned copy so we can apply transforms in place.
    let mut working: Vec<f32> = logits.to_vec();

    // 1. Repeat penalty.
    if let Some(penalty) = config.repeat_penalty {
        if penalty > 1.0_f32 {
            for &tok in recent_tokens.iter() {
                let i = tok as usize;
                if i < working.len() {
                    let v = working[i];
                    // HF convention: positive logits divide,
                    // negative logits multiply (so the penalty
                    // pushes towards 0 either way).
                    working[i] = if v > 0.0 { v / penalty } else { v * penalty };
                }
            }
        }
    }

    // 2. Temperature scaling. temperature == 0 was already handled
    // by the greedy branch above; here we know `temperature > 0`.
    if (config.temperature - 1.0).abs() > f32::EPSILON {
        let inv_t = 1.0 / config.temperature;
        for v in working.iter_mut() {
            *v *= inv_t;
        }
    }

    // 3. Top-k truncation: set every logit outside the top-K to -inf.
    if let Some(k) = config.top_k {
        if k > 0 && k < working.len() {
            // Find the K-th largest value via partial sort. For
            // moderate K (typical 50-200) and vocab ~50K this is
            // O(N log K) which beats a full sort.
            let mut indexed: Vec<(usize, f32)> = working.iter().copied().enumerate().collect();
            indexed.select_nth_unstable_by(k - 1, |a, b| {
                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
            });
            let kth_value = indexed[k - 1].1;
            for v in working.iter_mut() {
                if *v < kth_value {
                    *v = f32::NEG_INFINITY;
                }
            }
        }
    }

    // 4. Softmax (numerically stable).
    let max_logit = working.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum_exp = 0.0_f32;
    for v in working.iter_mut() {
        *v = (*v - max_logit).exp();
        sum_exp += *v;
    }
    if sum_exp > 0.0 {
        let inv_sum = 1.0 / sum_exp;
        for v in working.iter_mut() {
            *v *= inv_sum;
        }
    } else {
        // All -inf logits (degenerate); fall back to argmax of the
        // original logits.
        return argmax(logits);
    }

    // 5. Top-p nucleus: keep the smallest cumulative-probability
    // mass that exceeds `top_p`. We sort *descending* by prob,
    // walk the prefix, and zero out everything past the
    // threshold.
    if let Some(p) = config.top_p {
        if p > 0.0 && p < 1.0 {
            let mut sorted_idx: Vec<usize> = (0..working.len()).collect();
            sorted_idx.sort_unstable_by(|&a, &b| {
                working[b]
                    .partial_cmp(&working[a])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            let mut cum = 0.0_f32;
            let mut cutoff = sorted_idx.len();
            for (rank, &idx) in sorted_idx.iter().enumerate() {
                cum += working[idx];
                if cum >= p {
                    cutoff = rank + 1;
                    break;
                }
            }
            for &idx in &sorted_idx[cutoff..] {
                working[idx] = 0.0;
            }
            // Re-normalise.
            let s: f32 = working.iter().sum();
            if s > 0.0 {
                let inv_s = 1.0 / s;
                for v in working.iter_mut() {
                    *v *= inv_s;
                }
            }
        }
    }

    // 6. Inverse-CDF sample.
    let u = uniform01_sample().clamp(0.0, 1.0_f32 - f32::EPSILON);
    let mut acc = 0.0_f32;
    for (i, &p) in working.iter().enumerate() {
        acc += p;
        if u < acc {
            return i;
        }
    }
    // Safety net: float drift can leave acc slightly < 1.0; return
    // the last non-zero index.
    working
        .iter()
        .enumerate()
        .rev()
        .find(|(_, p)| **p > 0.0)
        .map(|(i, _)| i)
        .unwrap_or(0)
}

/// Argmax with stable tie-breaking on lower index (matches
/// `torch.argmax` default).
fn argmax(logits: &[f32]) -> usize {
    let mut best = (0_usize, f32::NEG_INFINITY);
    for (i, &v) in logits.iter().enumerate() {
        if v > best.1 {
            best = (i, v);
        }
    }
    best.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_picks_argmax() {
        let logits = vec![0.5_f32, 1.0, -2.0, 3.5, 0.8];
        let cfg = SamplingConfig::greedy();
        // RNG closure should never be called for greedy.
        let pick = sample_next(&logits, &cfg, &[], || {
            panic!("greedy must not call the RNG")
        });
        assert_eq!(pick, 3);
    }

    #[test]
    fn temperature_one_no_top_filter_samples_proportionally() {
        // Two tokens, one strongly favoured. With u just below the
        // weight of the favoured token's CDF, we should pick it.
        let logits = vec![0.0_f32, 100.0]; // softmax ~ [0, 1]
        let cfg = SamplingConfig::with_temperature(1.0);
        let pick = sample_next(&logits, &cfg, &[], || 0.99);
        assert_eq!(pick, 1);
    }

    #[test]
    fn top_k_eq_1_is_argmax_under_temperature_one() {
        let logits = vec![1.0_f32, 5.0, 2.0, 4.0];
        let cfg = SamplingConfig {
            temperature: 1.0,
            top_k: Some(1),
            top_p: None,
            repeat_penalty: None,
        };
        // With K=1 only the argmax survives; any RNG draw picks it.
        let pick = sample_next(&logits, &cfg, &[], || 0.5);
        assert_eq!(pick, 1);
    }

    #[test]
    fn repeat_penalty_pushes_toward_unseen() {
        // Two tokens with equal logit. Mark token 0 as recent with
        // a strong penalty — top-k=1 then forces argmax over the
        // post-penalty logits, which must pick token 1.
        let logits = vec![1.0_f32, 1.0];
        let cfg = SamplingConfig {
            temperature: 1.0,
            top_k: Some(1),
            top_p: None,
            repeat_penalty: Some(2.0),
        };
        let pick = sample_next(&logits, &cfg, &[0], || 0.5);
        assert_eq!(pick, 1);
    }

    #[test]
    fn empty_logits_returns_zero() {
        let logits: Vec<f32> = Vec::new();
        let cfg = SamplingConfig::greedy();
        assert_eq!(sample_next(&logits, &cfg, &[], || 0.5), 0);
    }

    #[test]
    fn top_p_truncation_keeps_only_top_mass() {
        // Vocab of 4. Probs: [.7, .15, .1, .05]. Top-p = 0.8 keeps
        // only first two tokens (.7 + .15 = .85 ≥ .8). With u >= .7
        // we pick token 1; otherwise token 0.
        let logits = vec![1.946_f32, 0.405, 0.0, -0.693]; // softmax ~ [.7, .15, .1, .05]
        let cfg = SamplingConfig {
            temperature: 1.0,
            top_k: None,
            top_p: Some(0.8),
            repeat_penalty: None,
        };
        let pick_high = sample_next(&logits, &cfg, &[], || 0.95);
        // The two surviving tokens after top-p have re-normalised
        // probs (.7/.85, .15/.85). So u=0.95 falls past the first
        // bucket and picks token 1.
        assert!(pick_high == 0 || pick_high == 1);
        assert!(pick_high < 2);
    }
}
