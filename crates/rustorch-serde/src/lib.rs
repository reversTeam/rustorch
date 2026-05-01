//! # rustorch-serde
//!
//! safetensors reader + writer for rustorch. Compatible with HuggingFace's
//! safetensors format (header JSON + binary blocks). Used by `Module::save`
//! and `Module::load`. Also exposes `state_dict()` / `load_state_dict()`
//! glue.

#![cfg_attr(docsrs, feature(doc_cfg))]
#![warn(missing_docs)]
#![warn(rust_2018_idioms)]

pub mod safetensors;

pub use safetensors::{read_from, read_path, write_path, write_to, SafetensorsError};

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
