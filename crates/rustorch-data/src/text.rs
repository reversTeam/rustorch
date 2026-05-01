//! Text augmentations / tokenizers (P1.9 minimal slice).
//!
//! v1 ships:
//! - [`Pad`] — right-pad a token sequence to a fixed length.
//! - [`Truncate`] — clip a token sequence to a maximum length.
//! - [`WhitespaceTokenizer`] — split on whitespace; vocabulary built
//!   from a corpus, OOV tokens map to `<unk>`.
//!
//! BPE / WordPiece / SentencePiece (HuggingFace-compatible) require
//! merge tables and are pending follow-ups.

use std::collections::HashMap;

/// Right-pad a sequence with `pad_id` until length `max_len`.
/// If the input is already ≥ `max_len`, returns it unchanged
/// (use [`Truncate`] to clip).
pub struct Pad {
    /// Target sequence length.
    pub max_len: usize,
    /// Padding token id.
    pub pad_id: i64,
}

impl Pad {
    /// Build with the given length and pad id (default 0).
    pub fn new(max_len: usize) -> Self {
        Pad { max_len, pad_id: 0 }
    }

    /// Override the padding token id.
    #[must_use]
    pub fn pad_id(mut self, id: i64) -> Self {
        self.pad_id = id;
        self
    }

    /// Apply: returns a new vector of length `max_len`.
    pub fn apply(&self, tokens: &[i64]) -> Vec<i64> {
        let mut out = tokens.to_vec();
        while out.len() < self.max_len {
            out.push(self.pad_id);
        }
        out
    }
}

/// Clip a sequence to `max_len`. No padding.
pub struct Truncate {
    /// Maximum length.
    pub max_len: usize,
}

impl Truncate {
    /// Build with the given maximum length.
    pub fn new(max_len: usize) -> Self {
        Truncate { max_len }
    }

    /// Apply: returns a slice of at most `max_len` tokens.
    pub fn apply<'a>(&self, tokens: &'a [i64]) -> &'a [i64] {
        if tokens.len() > self.max_len {
            &tokens[..self.max_len]
        } else {
            tokens
        }
    }
}

/// Simple whitespace tokenizer with a fixed vocabulary.
///
/// Vocabulary is keyed by (lower-cased) word; OOV maps to `<unk>` (id 0).
/// `<pad>` is reserved at id 0; `<unk>` at id 1.
pub struct WhitespaceTokenizer {
    word_to_id: HashMap<String, i64>,
}

impl WhitespaceTokenizer {
    /// Build with the given vocabulary list (order = ids starting at 2).
    pub fn new(vocab: Vec<String>) -> Self {
        let mut map: HashMap<String, i64> = HashMap::new();
        map.insert("<pad>".into(), 0);
        map.insert("<unk>".into(), 1);
        for (i, w) in vocab.into_iter().enumerate() {
            map.entry(w.to_lowercase()).or_insert((i + 2) as i64);
        }
        WhitespaceTokenizer { word_to_id: map }
    }

    /// Build a vocabulary by scanning a corpus; collects unique words.
    pub fn from_corpus(corpus: &[&str]) -> Self {
        let mut seen: HashMap<String, ()> = HashMap::new();
        for line in corpus {
            for w in line.split_whitespace() {
                seen.entry(w.to_lowercase()).or_insert(());
            }
        }
        let vocab: Vec<String> = seen.into_keys().collect();
        Self::new(vocab)
    }

    /// Encode text → ids. OOV → `<unk>` (id 1).
    pub fn encode(&self, text: &str) -> Vec<i64> {
        text.split_whitespace()
            .map(|w| self.word_to_id.get(&w.to_lowercase()).copied().unwrap_or(1))
            .collect()
    }

    /// Vocabulary size (including `<pad>` and `<unk>`).
    pub fn vocab_size(&self) -> usize {
        self.word_to_id.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pad_extends_to_max_len() {
        let pad = Pad::new(8);
        let v = pad.apply(&[1_i64, 2, 3]);
        assert_eq!(v, vec![1_i64, 2, 3, 0, 0, 0, 0, 0]);
    }

    #[test]
    fn pad_no_op_when_already_at_max() {
        let pad = Pad::new(3);
        let v = pad.apply(&[1_i64, 2, 3]);
        assert_eq!(v, vec![1_i64, 2, 3]);
    }

    #[test]
    fn truncate_clips_excess() {
        let trunc = Truncate::new(3);
        let v = trunc.apply(&[1_i64, 2, 3, 4, 5]);
        assert_eq!(v, &[1_i64, 2, 3]);
    }

    #[test]
    fn whitespace_tokenizer_round_trip() {
        let tok = WhitespaceTokenizer::from_corpus(&["the cat sat on the mat"]);
        let ids = tok.encode("the cat");
        assert_eq!(ids.len(), 2);
        // Both tokens should be in vocab → not <unk> (1).
        assert!(ids.iter().all(|&id| id != 1));
    }

    #[test]
    fn whitespace_tokenizer_oov_maps_to_unk() {
        let tok = WhitespaceTokenizer::new(vec!["hello".into()]);
        let ids = tok.encode("hello world");
        assert_eq!(ids[0], 2); // hello
        assert_eq!(ids[1], 1); // world → <unk>
    }
}
