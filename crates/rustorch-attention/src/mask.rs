//! Attention masks — Phase 3 task `abdfefda` (causal + padding only).
//!
//! Two mask flavours, applied INSIDE the score-tile loop so the
//! masked entries become `-inf` BEFORE the online softmax sees them
//! (preserving numerical stability — no NaN cancellation, no
//! finite-output-from-Inf-input artefacts).
//!
//! ## Causal mask
//! For each query position `i`, attention can only see keys at
//! position `j <= i`. Geometrically: the score matrix has its
//! upper triangle (j > i) set to `-inf`.
//!
//! ## Padding mask
//! Per-key boolean vector of length `seq`. `true` = keep, `false` =
//! mask out (set its column in the score matrix to `-inf`). Useful
//! for variable-length sequences batched together.
//!
//! ## Dropout (deferred)
//! Dropout requires seedable, replayable RNG state shared between
//! forward and backward. That belongs to a separate runtime layer
//! and is intentionally NOT shipped in this task — see the related
//! decision `Masking task scope` recorded in the project graph.

/// Attention mask flavour.
#[derive(Debug, Clone)]
pub enum Mask<'a> {
    /// No masking — every Q-K pair contributes.
    None,
    /// Causal: upper triangle of the score matrix is masked
    /// (j > i → score = -inf). Standard for autoregressive LMs.
    Causal,
    /// Per-key keep flags of length `seq`. `false` entries get masked.
    Padding(&'a [bool]),
}

impl<'a> Mask<'a> {
    /// True iff `(qi, kj)` should be EXCLUDED from attention.
    /// Pure helper used by the kernel.
    #[inline]
    pub fn is_masked(&self, qi: usize, kj: usize) -> bool {
        match self {
            Mask::None => false,
            Mask::Causal => kj > qi,
            Mask::Padding(keep) => kj < keep.len() && !keep[kj],
        }
    }

    /// Validate the mask against a sequence length. `Causal` and
    /// `None` are always valid; `Padding(keep)` requires
    /// `keep.len() == seq`.
    pub fn validate(&self, seq: usize) -> Result<(), MaskError> {
        match self {
            Mask::None | Mask::Causal => Ok(()),
            Mask::Padding(keep) => {
                if keep.len() == seq {
                    Ok(())
                } else {
                    Err(MaskError::PaddingLengthMismatch {
                        expected: seq,
                        got: keep.len(),
                    })
                }
            },
        }
    }
}

/// Errors raised by mask validation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MaskError {
    /// Padding mask length doesn't match the sequence length.
    PaddingLengthMismatch {
        /// Expected length (= shape.seq).
        expected: usize,
        /// Actual `keep.len()`.
        got: usize,
    },
}

impl core::fmt::Display for MaskError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            MaskError::PaddingLengthMismatch { expected, got } => {
                write!(f, "padding mask len {got}, expected {expected}")
            },
        }
    }
}

impl std::error::Error for MaskError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_mask_never_masks() {
        let m = Mask::None;
        for qi in 0..8 {
            for kj in 0..8 {
                assert!(!m.is_masked(qi, kj));
            }
        }
    }

    #[test]
    fn causal_masks_upper_triangle() {
        let m = Mask::Causal;
        // Lower triangle + diagonal → not masked.
        assert!(!m.is_masked(2, 0));
        assert!(!m.is_masked(2, 2));
        // Upper triangle → masked.
        assert!(m.is_masked(2, 3));
        assert!(m.is_masked(0, 1));
    }

    #[test]
    fn padding_masks_only_false_positions() {
        let keep = [true, false, true, true, false];
        let m = Mask::Padding(&keep);
        assert!(!m.is_masked(0, 0));
        assert!(m.is_masked(0, 1));
        assert!(!m.is_masked(0, 2));
        assert!(!m.is_masked(0, 3));
        assert!(m.is_masked(0, 4));
    }

    #[test]
    fn padding_validate_length_match() {
        let keep = [true; 5];
        let m = Mask::Padding(&keep);
        assert!(m.validate(5).is_ok());
        assert_eq!(
            m.validate(6).unwrap_err(),
            MaskError::PaddingLengthMismatch {
                expected: 6,
                got: 5
            }
        );
    }

    #[test]
    fn empty_padding_treated_as_no_keep() {
        let keep: [bool; 0] = [];
        let m = Mask::Padding(&keep);
        // Out-of-range kj is treated as not-in-mask (no panic).
        assert!(!m.is_masked(0, 0));
    }

    #[test]
    fn causal_at_diagonal_is_visible() {
        let m = Mask::Causal;
        for i in 0..16 {
            assert!(!m.is_masked(i, i));
        }
    }
}
