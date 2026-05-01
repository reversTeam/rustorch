//! # rustorch-serde
//!
//! safetensors reader + writer for rustorch. Compatible with HuggingFace's
//! safetensors format (header JSON + binary blocks). Used by `Module::save`
//! and `Module::load`. Also exposes `state_dict()` / `load_state_dict()`.
//!
//! Concrete impl lands in P1.8 (Sérialisation safetensors).

#![cfg_attr(docsrs, feature(doc_cfg))]
#![warn(rust_2018_idioms)]

/// Crate version reported at runtime.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smoke_version_present() {
        assert!(!VERSION.is_empty());
    }
}
