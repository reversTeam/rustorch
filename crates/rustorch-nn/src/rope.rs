//! Rotary Position Embeddings — RoPE (T42).
//!
//! Su et al. 2021. The position-dependent transformation applied to
//! Q and K projections in modern LLMs (Llama, Qwen, Mistral, Phi,
//! GPT-NeoX, etc.) — replaces the explicit additive position
//! embedding of GPT-2-style models.
//!
//! ## Math
//!
//! For each token at position `p`, head_dim `D` (must be even),
//! and dim pair `(2k, 2k+1)`:
//!
//! ```text
//!   theta_k = 1 / (base ^ (2k / D))
//!   angle   = p * theta_k
//!   x[p, 2k]   = x[p, 2k]   * cos(angle) - x[p, 2k+1] * sin(angle)
//!   x[p, 2k+1] = x[p, 2k]   * sin(angle) + x[p, 2k+1] * cos(angle)
//! ```
//!
//! `base` is 10000 by default (Llama-style); Qwen uses 10000 as well
//! for short contexts and dynamic scaling via NTK-aware extensions
//! for longer contexts (deferred to a follow-up).
//!
//! ## Implementation
//!
//! Pre-computes `cos[max_seq, D/2]` and `sin[max_seq, D/2]` tables
//! at construction. Forward applies rotation in-place to a
//! `[batch, n_heads, seq, head_dim]` buffer at a given position
//! offset (so we can apply RoPE to a single new token starting at
//! `current_len` during autoregressive decode without recomputing
//! over the full prefix).

/// Rotary Position Embedding helper.
pub struct RoPE {
    /// Per-head feature dim. Must be even.
    pub head_dim: usize,
    /// Maximum sequence length cached in the cos/sin tables.
    pub max_seq: usize,
    /// `[max_seq * head_dim/2]` row-major cos table.
    cos: Vec<f32>,
    /// `[max_seq * head_dim/2]` row-major sin table.
    sin: Vec<f32>,
}

impl RoPE {
    /// Expose the precomputed `cos[max_seq * head_dim/2]` table for
    /// upload into a GPU buffer (needed by `rustorch-metal`'s RoPE
    /// kernel — avoids re-running the trig at every forward pass).
    pub fn cos_table(&self) -> &[f32] {
        &self.cos
    }
    /// Companion to [`RoPE::cos_table`].
    pub fn sin_table(&self) -> &[f32] {
        &self.sin
    }
}

impl RoPE {
    /// Build the cos/sin tables for `head_dim` and `max_seq` using
    /// the given `base` (typical: `10000.0`).
    ///
    /// Panics if `head_dim` is odd.
    pub fn new(head_dim: usize, max_seq: usize, base: f32) -> Self {
        assert!(
            head_dim % 2 == 0,
            "RoPE: head_dim must be even, got {}",
            head_dim
        );
        let half = head_dim / 2;
        let mut cos = Vec::with_capacity(max_seq * half);
        let mut sin = Vec::with_capacity(max_seq * half);
        // theta_k = 1 / (base ^ (2k / head_dim)) for k in 0..D/2
        let inv_thetas: Vec<f32> = (0..half)
            .map(|k| 1.0_f32 / base.powf((2 * k) as f32 / head_dim as f32))
            .collect();
        for p in 0..max_seq {
            for &inv_theta in inv_thetas.iter() {
                let angle = p as f32 * inv_theta;
                cos.push(angle.cos());
                sin.push(angle.sin());
            }
        }
        RoPE {
            head_dim,
            max_seq,
            cos,
            sin,
        }
    }

    /// Apply RoPE in place to `x[batch, n_heads, seq, head_dim]`,
    /// where positions start at `position_offset`. For prefill on a
    /// fresh prompt, `position_offset = 0`. For autoregressive
    /// decode of a single new token, `position_offset = current_len`
    /// and `seq = 1`.
    pub fn apply_inplace(
        &self,
        x: &mut [f32],
        batch: usize,
        n_heads: usize,
        seq: usize,
        position_offset: usize,
    ) -> Result<(), RoPEError> {
        let d = self.head_dim;
        let half = d / 2;
        let expected_len = batch * n_heads * seq * d;
        if x.len() != expected_len {
            return Err(RoPEError::WrongInputLen {
                expected: expected_len,
                got: x.len(),
            });
        }
        if position_offset + seq > self.max_seq {
            return Err(RoPEError::PositionOverflow {
                max_seq: self.max_seq,
                requested: position_offset + seq,
            });
        }
        // For each (b, h, s) row of length d, rotate dim pairs
        // (2k, 2k+1) by angle = (position_offset + s) * theta_k.
        for b in 0..batch {
            for h in 0..n_heads {
                for s in 0..seq {
                    let p = position_offset + s;
                    let row_off = b * n_heads * seq * d + h * seq * d + s * d;
                    let table_off = p * half;
                    for k in 0..half {
                        let i0 = row_off + 2 * k;
                        let i1 = row_off + 2 * k + 1;
                        let c = self.cos[table_off + k];
                        let si = self.sin[table_off + k];
                        let x0 = x[i0];
                        let x1 = x[i1];
                        x[i0] = x0 * c - x1 * si;
                        x[i1] = x0 * si + x1 * c;
                    }
                }
            }
        }
        Ok(())
    }

    /// Apply RoPE in place using the HuggingFace `apply_rotary_pos_emb`
    /// half-split convention (used by Llama / Qwen / Mistral / Phi when
    /// the model is loaded from HF or GGUF in HF format).
    ///
    /// Unlike [`apply_inplace`] which pairs dimensions `(2k, 2k+1)`
    /// (interleaved / GPT-NeoX original RoFormer convention), this
    /// pairs dim `k` with dim `k + D/2` (split-half). The two
    /// formulations are mathematically equivalent rotations but use
    /// a different memory layout — and HF weights are baked for the
    /// half-split layout, so applying interleaved RoPE on HF weights
    /// scrambles Q/K and breaks attention completely (residual stream
    /// loses input dependence by layer 1).
    ///
    /// Formula:
    /// ```text
    ///   x'[k]       = x[k]       * cos(angle) - x[k + D/2] * sin(angle)
    ///   x'[k + D/2] = x[k + D/2] * cos(angle) + x[k]       * sin(angle)
    /// ```
    /// for k in 0..D/2.
    pub fn apply_inplace_half_split(
        &self,
        x: &mut [f32],
        batch: usize,
        n_heads: usize,
        seq: usize,
        position_offset: usize,
    ) -> Result<(), RoPEError> {
        let d = self.head_dim;
        let half = d / 2;
        let expected_len = batch * n_heads * seq * d;
        if x.len() != expected_len {
            return Err(RoPEError::WrongInputLen {
                expected: expected_len,
                got: x.len(),
            });
        }
        if position_offset + seq > self.max_seq {
            return Err(RoPEError::PositionOverflow {
                max_seq: self.max_seq,
                requested: position_offset + seq,
            });
        }
        for b in 0..batch {
            for h in 0..n_heads {
                for s in 0..seq {
                    let p = position_offset + s;
                    let row_off = b * n_heads * seq * d + h * seq * d + s * d;
                    let table_off = p * half;
                    for k in 0..half {
                        let i0 = row_off + k;
                        let i1 = row_off + k + half;
                        let c = self.cos[table_off + k];
                        let si = self.sin[table_off + k];
                        let x0 = x[i0];
                        let x1 = x[i1];
                        x[i0] = x0 * c - x1 * si;
                        x[i1] = x1 * c + x0 * si;
                    }
                }
            }
        }
        Ok(())
    }
}

/// Errors raised by RoPE.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RoPEError {
    /// Buffer length doesn't match `batch * n_heads * seq * head_dim`.
    WrongInputLen {
        /// Expected length.
        expected: usize,
        /// Actual length.
        got: usize,
    },
    /// `position_offset + seq` exceeds the max_seq the tables were sized for.
    PositionOverflow {
        /// Configured max_seq.
        max_seq: usize,
        /// Requested final position.
        requested: usize,
    },
}

impl std::fmt::Display for RoPEError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RoPEError::WrongInputLen { expected, got } => {
                write!(f, "rope: expected buffer of {expected} f32, got {got}")
            },
            RoPEError::PositionOverflow { max_seq, requested } => {
                write!(f, "rope: position {requested} exceeds max_seq {max_seq}")
            },
        }
    }
}

impl std::error::Error for RoPEError {}

#[cfg(test)]
mod tests {
    use super::*;

    /// At position 0, RoPE is the identity (cos=1, sin=0).
    #[test]
    fn rope_position_0_is_identity() {
        let rope = RoPE::new(4, 8, 10000.0);
        let mut x = vec![1.0_f32, 2.0, 3.0, 4.0]; // [B=1, H=1, S=1, D=4]
        let original = x.clone();
        rope.apply_inplace(&mut x, 1, 1, 1, 0).unwrap();
        for (a, b) in x.iter().zip(original.iter()) {
            assert!((a - b).abs() < 1e-6, "got {a} expected {b}");
        }
    }

    /// Rotation preserves the L2 norm per pair (2k, 2k+1).
    #[test]
    fn rope_preserves_pair_norms() {
        let rope = RoPE::new(8, 16, 10000.0);
        let mut x: Vec<f32> = (0..8).map(|i| (i + 1) as f32 * 0.1).collect();
        let original = x.clone();
        rope.apply_inplace(&mut x, 1, 1, 1, 5).unwrap();
        // Each pair (2k, 2k+1) must keep |x|^2 invariant.
        for k in 0..4 {
            let before = original[2 * k].powi(2) + original[2 * k + 1].powi(2);
            let after = x[2 * k].powi(2) + x[2 * k + 1].powi(2);
            assert!(
                (before - after).abs() < 1e-5,
                "pair {k}: before {before} after {after}"
            );
        }
    }

    #[test]
    fn rope_overflow_returns_err() {
        let rope = RoPE::new(4, 4, 10000.0);
        let mut x = vec![0.0_f32; 8];
        let err = rope.apply_inplace(&mut x, 1, 1, 2, 3).unwrap_err();
        assert!(matches!(err, RoPEError::PositionOverflow { .. }));
    }

    #[test]
    fn rope_wrong_input_length() {
        let rope = RoPE::new(4, 4, 10000.0);
        let mut bad = vec![0.0_f32; 7]; // not 1*1*2*4 = 8
        let err = rope.apply_inplace(&mut bad, 1, 1, 2, 0).unwrap_err();
        assert!(matches!(err, RoPEError::WrongInputLen { .. }));
    }

    /// Multiple positions: applying RoPE to seq=N at offset=0 must
    /// match applying it row by row at offset=p.
    #[test]
    fn rope_batched_matches_row_by_row() {
        let rope = RoPE::new(4, 16, 10000.0);
        let seq = 5;
        let original: Vec<f32> = (0..seq * 4).map(|i| (i as f32 + 1.0) * 0.1).collect();
        // Batched apply.
        let mut batched = original.clone();
        rope.apply_inplace(&mut batched, 1, 1, seq, 0).unwrap();
        // Row-by-row apply: for s in 0..seq, apply at offset s with seq=1.
        let mut row_by_row = original.clone();
        for s in 0..seq {
            let row_slice = &mut row_by_row[s * 4..(s + 1) * 4];
            rope.apply_inplace(row_slice, 1, 1, 1, s).unwrap();
        }
        for (a, b) in batched.iter().zip(row_by_row.iter()) {
            assert!((a - b).abs() < 1e-5, "mismatch: {a} vs {b}");
        }
    }
}
