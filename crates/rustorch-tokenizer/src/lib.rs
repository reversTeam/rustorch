//! BPE / SentencePiece tokenizer for the RusTorch LLM-runner (T55).
//!
//! Thin wrapper over the HuggingFace [`tokenizers`] crate that
//! exposes only the operations the autoregressive decode loop
//! needs: `encode(text) → Vec<u32>` and `decode(ids) → String`.
//!
//! ## Why wrap, not re-implement
//!
//! The `tokenizers` crate is the canonical Rust BPE — it loads
//! every modern LLM's `tokenizer.json` (Llama, Qwen, Mistral,
//! GPT-2, GPT-NeoX, Phi, …) directly from the HuggingFace Hub
//! format. Re-implementing BPE from scratch would mean
//! reproducing 5+ years of HF correctness fixes for marginal
//! perf benefit; the wrapper instead exposes a small, stable
//! API and lets the underlying `tokenizers` crate do the heavy
//! lifting.
//!
//! ## Usage
//!
//! ```ignore
//! use rustorch_tokenizer::BpeTokenizer;
//!
//! let tok = BpeTokenizer::from_file("models/qwen3-32b/tokenizer.json")?;
//! let ids = tok.encode("Hello, world!")?;
//! let text = tok.decode(&ids, /*skip_special=*/true)?;
//! ```

use std::path::Path;
use tokenizers::tokenizer::{Tokenizer, TokenizerImpl};

/// Errors raised by the tokenizer.
#[derive(Debug)]
pub enum TokenizerError {
    /// Loading the `tokenizer.json` file failed.
    Load(String),
    /// Encoding the input string failed.
    Encode(String),
    /// Decoding token ids back to text failed.
    Decode(String),
}

impl std::fmt::Display for TokenizerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TokenizerError::Load(s) => write!(f, "tokenizer load: {s}"),
            TokenizerError::Encode(s) => write!(f, "tokenizer encode: {s}"),
            TokenizerError::Decode(s) => write!(f, "tokenizer decode: {s}"),
        }
    }
}

impl std::error::Error for TokenizerError {}

/// BPE / Unigram / WordPiece tokenizer. Loads any HuggingFace
/// `tokenizer.json` and exposes the encode/decode primitives the
/// LLM-runner needs.
pub struct BpeTokenizer {
    inner: Tokenizer,
}

impl BpeTokenizer {
    /// Load a tokenizer from a HuggingFace `tokenizer.json` file.
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, TokenizerError> {
        let inner = Tokenizer::from_file(path).map_err(|e| TokenizerError::Load(format!("{e}")))?;
        Ok(BpeTokenizer { inner })
    }

    /// Load a tokenizer from in-memory JSON bytes (e.g. embedded
    /// at compile time or fetched from network).
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, TokenizerError> {
        let inner =
            Tokenizer::from_bytes(bytes).map_err(|e| TokenizerError::Load(format!("{e}")))?;
        Ok(BpeTokenizer { inner })
    }

    /// Encode a string to a sequence of token ids. By default the
    /// tokenizer's configured pre-tokeniser, post-tokeniser, and
    /// any added special tokens (e.g. BOS) are applied.
    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>, TokenizerError> {
        let enc = self
            .inner
            .encode(text, add_special_tokens)
            .map_err(|e| TokenizerError::Encode(format!("{e}")))?;
        Ok(enc.get_ids().to_vec())
    }

    /// Decode token ids back to a UTF-8 string. `skip_special_tokens`
    /// removes BOS / EOS / pad tokens from the output.
    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String, TokenizerError> {
        self.inner
            .decode(ids, skip_special_tokens)
            .map_err(|e| TokenizerError::Decode(format!("{e}")))
    }

    /// Vocabulary size (number of valid token ids).
    pub fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }

    /// Borrow the underlying HuggingFace `Tokenizer` for advanced
    /// use cases (custom batching, padding, truncation strategies).
    pub fn inner(&self) -> &Tokenizer {
        &self.inner
    }
}

// `TokenizerImpl` is re-exported as an opaque alias in case downstream
// code wants to talk to the HF type directly. This avoids forcing every
// caller to add `tokenizers` to their own deps when they only need the
// type name.
#[allow(dead_code)]
type _HfTokenizer = TokenizerImpl<
    tokenizers::ModelWrapper,
    tokenizers::NormalizerWrapper,
    tokenizers::PreTokenizerWrapper,
    tokenizers::PostProcessorWrapper,
    tokenizers::DecoderWrapper,
>;

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal in-memory BPE tokenizer for unit tests
    /// (no on-disk fixture required). This is the smallest valid
    /// `tokenizer.json` that the HF crate accepts: a flat WordLevel
    /// model with three tokens.
    fn tiny_word_level_json() -> String {
        // Build the JSON via concat to avoid raw-string escape pain
        // around the WordPiece "##" prefix (decoders/WordPiece needs
        // it even though the actual subwords are absent here).
        let prefix = "##".to_string();
        format!(
            r#"{{
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": [],
            "normalizer": null,
            "pre_tokenizer": {{"type":"Whitespace"}},
            "post_processor": null,
            "decoder": {{"type":"WordPiece","prefix":"{}","cleanup":true}},
            "model": {{
                "type": "WordLevel",
                "vocab": {{"hello": 0, "world": 1, "[UNK]": 2}},
                "unk_token": "[UNK]"
            }}
        }}"#,
            prefix
        )
    }

    #[test]
    fn encode_decode_round_trip() {
        let json = tiny_word_level_json();
        let tok = BpeTokenizer::from_bytes(json.as_bytes()).unwrap();
        let ids = tok.encode("hello world", false).unwrap();
        assert_eq!(ids, vec![0, 1]);
        let text = tok.decode(&ids, false).unwrap();
        assert_eq!(text, "hello world");
    }

    #[test]
    fn vocab_size_includes_added_tokens() {
        let json = tiny_word_level_json();
        let tok = BpeTokenizer::from_bytes(json.as_bytes()).unwrap();
        assert_eq!(tok.vocab_size(), 3);
    }

    #[test]
    fn unknown_word_maps_to_unk() {
        let json = tiny_word_level_json();
        let tok = BpeTokenizer::from_bytes(json.as_bytes()).unwrap();
        let ids = tok.encode("foo", false).unwrap();
        assert_eq!(ids, vec![2]); // [UNK]
    }
}
