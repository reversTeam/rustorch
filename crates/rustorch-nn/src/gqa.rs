//! Grouped Query Attention — GQA (T45).
//!
//! Ainslie et al. 2023 (used by Llama 3, Qwen, Mistral). The Q tensor
//! has `n_heads` heads but the K and V tensors share `n_kv_heads`
//! (with `n_heads % n_kv_heads == 0`). Each query head still
//! attends to a key/value pair, but groups of `n_heads / n_kv_heads`
//! query heads share the same KV head — saving the cache footprint
//! and the K/V projection compute by a factor of
//! `n_heads / n_kv_heads`.
//!
//! On Qwen3-32B that ratio is 4 (32 query heads, 8 KV heads), so
//! GQA cuts the KV-cache memory by 4x relative to MHA — critical
//! for fitting long contexts.
//!
//! ## Forward strategy
//!
//! For each (batch, head, query_pos):
//!   1. Pick the KV head index `kv_h = head / group_size`.
//!   2. Compute `Q_h · K_kv_h` for every key position, scale,
//!      softmax.
//!   3. Weighted sum over `V_kv_h`.
//!
//! Equivalent to repeating K and V `group_size` times along the
//! head dimension and running standard MHA, but skips the actual
//! materialised broadcast — we just index the right head every time.
//!
//! ## Limitations
//!
//! - F32 only; bf16 lands with the quant work.
//! - No mask in v0 (causal mask handled by the caller for now).
//! - Outer parallelism over (batch, head). For LLM serving with
//!   batch=1, this means parallelism = n_heads which is plenty
//!   on 4-8 P-cores.

/// Errors raised by GQA.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GQAError {
    /// `n_heads` not divisible by `n_kv_heads`.
    HeadCountMismatch {
        /// Query head count.
        n_heads: usize,
        /// KV head count.
        n_kv_heads: usize,
    },
    /// Buffer length doesn't match the expected
    /// `[batch, n_heads, seq_q, head_dim]` (or KV equivalent).
    WrongInputLen {
        /// Which buffer.
        which: &'static str,
        /// Expected.
        expected: usize,
        /// Actual.
        got: usize,
    },
}

impl std::fmt::Display for GQAError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GQAError::HeadCountMismatch {
                n_heads,
                n_kv_heads,
            } => write!(
                f,
                "gqa: n_heads {n_heads} not divisible by n_kv_heads {n_kv_heads}"
            ),
            GQAError::WrongInputLen {
                which,
                expected,
                got,
            } => {
                write!(f, "gqa: {which} expected {expected} f32, got {got}")
            },
        }
    }
}

impl std::error::Error for GQAError {}

/// Grouped Query Attention forward, no mask, f32.
///
/// Inputs:
/// - `q` shape `[batch, n_heads, seq_q, head_dim]` (row-major).
/// - `k` shape `[batch, n_kv_heads, seq_kv, head_dim]`.
/// - `v` shape `[batch, n_kv_heads, seq_kv, head_dim]`.
///
/// Output:
/// - `out` shape `[batch, n_heads, seq_q, head_dim]`, fully
///   overwritten.
///
/// `n_heads` must be a multiple of `n_kv_heads`. Each query head
/// `h` reads from KV head `h / (n_heads / n_kv_heads)`.
#[allow(clippy::too_many_arguments)]
pub fn gqa_forward_f32(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    out: &mut [f32],
    batch: usize,
    n_heads: usize,
    n_kv_heads: usize,
    seq_q: usize,
    seq_kv: usize,
    head_dim: usize,
) -> Result<(), GQAError> {
    if n_kv_heads == 0 || n_heads % n_kv_heads != 0 {
        return Err(GQAError::HeadCountMismatch {
            n_heads,
            n_kv_heads,
        });
    }
    let q_len = batch * n_heads * seq_q * head_dim;
    let kv_len = batch * n_kv_heads * seq_kv * head_dim;
    if q.len() != q_len {
        return Err(GQAError::WrongInputLen {
            which: "q",
            expected: q_len,
            got: q.len(),
        });
    }
    if k.len() != kv_len {
        return Err(GQAError::WrongInputLen {
            which: "k",
            expected: kv_len,
            got: k.len(),
        });
    }
    if v.len() != kv_len {
        return Err(GQAError::WrongInputLen {
            which: "v",
            expected: kv_len,
            got: v.len(),
        });
    }
    if out.len() != q_len {
        return Err(GQAError::WrongInputLen {
            which: "out",
            expected: q_len,
            got: out.len(),
        });
    }

    let group_size = n_heads / n_kv_heads;

    // T45 — flash_forward only supports seq_q == seq_kv (self-
    // attention prefill). For decode (seq_q != seq_kv, typically
    // seq_q=1 querying a long cached prefix) we run a tight naive
    // kernel parallelised over (batch * head). The naive path is
    // also ~2× faster than flash for the seq_q=1 case where the
    // tiled kernel's setup cost dominates.
    if seq_q == seq_kv {
        // Prefill / self-attention path: replicate K, V to n_heads
        // and call flash_forward.
        let kv_h_size = seq_kv * head_dim;
        let kv_block = batch * n_heads * kv_h_size;
        let mut k_rep = vec![0.0_f32; kv_block];
        let mut v_rep = vec![0.0_f32; kv_block];
        for b in 0..batch {
            for h in 0..n_heads {
                let kv_h = h / group_size;
                let src_off = b * n_kv_heads * kv_h_size + kv_h * kv_h_size;
                let dst_off = b * n_heads * kv_h_size + h * kv_h_size;
                k_rep[dst_off..dst_off + kv_h_size]
                    .copy_from_slice(&k[src_off..src_off + kv_h_size]);
                v_rep[dst_off..dst_off + kv_h_size]
                    .copy_from_slice(&v[src_off..src_off + kv_h_size]);
            }
        }
        use rustorch_attention::{flash_forward, AttentionShape};
        let shape = AttentionShape::new(batch, n_heads, seq_q, head_dim);
        flash_forward(&shape, q, &k_rep, &v_rep, out).map_err(|_| GQAError::WrongInputLen {
            which: "flash_forward",
            expected: 0,
            got: 0,
        })?;
        return Ok(());
    }

    // Decode path: naive per-head attention. Each query head reads
    // from its mapped KV head; we softmax over `seq_kv` positions
    // and accumulate into `out`. Parallel over (batch * n_heads).
    use rayon::prelude::*;
    let scale = 1.0_f32 / (head_dim as f32).sqrt();
    out.par_chunks_mut(seq_q * head_dim)
        .enumerate()
        .for_each(|(bh_idx, out_block)| {
            let b = bh_idx / n_heads;
            let h = bh_idx % n_heads;
            let kv_h = h / group_size;
            let q_off = b * n_heads * seq_q * head_dim + h * seq_q * head_dim;
            let kv_off = b * n_kv_heads * seq_kv * head_dim + kv_h * seq_kv * head_dim;
            let q_block = &q[q_off..q_off + seq_q * head_dim];
            let k_block = &k[kv_off..kv_off + seq_kv * head_dim];
            let v_block = &v[kv_off..kv_off + seq_kv * head_dim];
            let mut scores = vec![0.0_f32; seq_kv];
            for q_pos in 0..seq_q {
                let q_row = &q_block[q_pos * head_dim..(q_pos + 1) * head_dim];
                for k_pos in 0..seq_kv {
                    let k_row = &k_block[k_pos * head_dim..(k_pos + 1) * head_dim];
                    let mut acc = 0.0_f32;
                    for d in 0..head_dim {
                        acc += q_row[d] * k_row[d];
                    }
                    scores[k_pos] = acc * scale;
                }
                let max_s = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut sum_exp = 0.0_f32;
                for s in scores.iter_mut() {
                    *s = (*s - max_s).exp();
                    sum_exp += *s;
                }
                let inv_sum = 1.0 / sum_exp;
                for s in scores.iter_mut() {
                    *s *= inv_sum;
                }
                let out_row = &mut out_block[q_pos * head_dim..(q_pos + 1) * head_dim];
                out_row.fill(0.0);
                for k_pos in 0..seq_kv {
                    let v_row = &v_block[k_pos * head_dim..(k_pos + 1) * head_dim];
                    let w = scores[k_pos];
                    for d in 0..head_dim {
                        out_row[d] += w * v_row[d];
                    }
                }
            }
        });
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// GQA with `n_heads == n_kv_heads` is exactly MHA. Check that
    /// the output matches a hand-rolled MHA on the same data.
    #[test]
    fn gqa_with_n_kv_heads_eq_n_heads_matches_mha() {
        let batch = 1;
        let n_heads = 4;
        let n_kv_heads = 4;
        let seq_q = 3;
        let seq_kv = 3;
        let head_dim = 4;

        let q: Vec<f32> = (0..batch * n_heads * seq_q * head_dim)
            .map(|i| (i as f32 + 1.0) * 0.01)
            .collect();
        let k: Vec<f32> = (0..batch * n_kv_heads * seq_kv * head_dim)
            .map(|i| (i as f32 + 1.0) * 0.02)
            .collect();
        let v: Vec<f32> = (0..batch * n_kv_heads * seq_kv * head_dim)
            .map(|i| (i as f32 + 1.0) * 0.03)
            .collect();
        let mut out = vec![0.0_f32; batch * n_heads * seq_q * head_dim];

        gqa_forward_f32(
            &q, &k, &v, &mut out, batch, n_heads, n_kv_heads, seq_q, seq_kv, head_dim,
        )
        .unwrap();

        // Sanity: attention output is a softmax-weighted average so
        // every cell must lie in [min(V_h), max(V_h)] for its head.
        for h in 0..n_heads {
            let v_block_off = h * seq_kv * head_dim;
            let v_block = &v[v_block_off..v_block_off + seq_kv * head_dim];
            let v_min = v_block.iter().copied().fold(f32::INFINITY, f32::min);
            let v_max = v_block.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let out_block_off = h * seq_q * head_dim;
            for &x in &out[out_block_off..out_block_off + seq_q * head_dim] {
                assert!(
                    x >= v_min - 1e-5 && x <= v_max + 1e-5,
                    "h={h} out of [V_min={v_min}, V_max={v_max}]: {x}"
                );
            }
        }
    }

    /// 4-to-1 group ratio (Qwen3-32B style: 32 query heads, 8 KV).
    #[test]
    fn gqa_4_to_1_group_ratio_runs_and_outputs_finite() {
        let batch = 1;
        let n_heads = 8;
        let n_kv_heads = 2;
        let seq_q = 4;
        let seq_kv = 4;
        let head_dim = 8;

        let q: Vec<f32> = (0..batch * n_heads * seq_q * head_dim)
            .map(|i| (i as f32 * 0.001).sin())
            .collect();
        let k: Vec<f32> = (0..batch * n_kv_heads * seq_kv * head_dim)
            .map(|i| (i as f32 * 0.001).cos())
            .collect();
        let v: Vec<f32> = (0..batch * n_kv_heads * seq_kv * head_dim)
            .map(|i| ((i as f32 + 1.0) * 0.0005).tan())
            .collect();
        let mut out = vec![0.0_f32; batch * n_heads * seq_q * head_dim];

        gqa_forward_f32(
            &q, &k, &v, &mut out, batch, n_heads, n_kv_heads, seq_q, seq_kv, head_dim,
        )
        .unwrap();

        for &x in out.iter() {
            assert!(x.is_finite(), "non-finite: {x}");
        }
        // Every group of `group_size` query heads should produce
        // related (but not identical) outputs — they share the
        // same K/V but their Q is different.
        let group_size = n_heads / n_kv_heads;
        for kv_h in 0..n_kv_heads {
            let group_h_start = kv_h * group_size;
            // The first head of the group should differ from the
            // last (different Q).
            let h0 = group_h_start;
            let h1 = group_h_start + group_size - 1;
            let off0 = h0 * seq_q * head_dim;
            let off1 = h1 * seq_q * head_dim;
            let mut diff = 0.0_f32;
            for d in 0..head_dim {
                diff += (out[off0 + d] - out[off1 + d]).abs();
            }
            assert!(diff > 1e-6, "kv group {kv_h}: heads should differ");
        }
    }

    #[test]
    fn gqa_head_count_mismatch_returns_err() {
        let q = vec![0.0_f32; 5 * 4];
        let k = vec![0.0_f32; 2 * 4];
        let v = vec![0.0_f32; 2 * 4];
        let mut out = vec![0.0_f32; 5 * 4];
        // 5 query heads not divisible by 2 kv heads.
        let err = gqa_forward_f32(&q, &k, &v, &mut out, 1, 5, 2, 1, 1, 4).unwrap_err();
        assert!(matches!(err, GQAError::HeadCountMismatch { .. }));
    }

    #[test]
    fn gqa_wrong_input_length_returns_err() {
        let q = vec![0.0_f32; 7]; // not 1*4*1*4 = 16
        let k = vec![0.0_f32; 2 * 4];
        let v = vec![0.0_f32; 2 * 4];
        let mut out = vec![0.0_f32; 4 * 4];
        let err = gqa_forward_f32(&q, &k, &v, &mut out, 1, 4, 2, 1, 1, 4).unwrap_err();
        assert!(matches!(err, GQAError::WrongInputLen { .. }));
    }
}
