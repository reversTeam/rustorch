//! # rustorch-derive
//!
//! Procedural macros for rustorch. Currently exposes `#[derive(Module)]`
//! which generates `parameters()`, `buffers()`, `modules()`, `train()` for
//! struct types whose fields hold parameters / buffers / submodules.
//!
//! Concrete impl lands in P1.6 (Module system).

#![warn(rust_2018_idioms)]

use proc_macro::TokenStream;

/// Placeholder `#[derive(Module)]` — full implementation in P1.6.
///
/// For now this expands to nothing so that downstream crates can already
/// reference the derive name without a compile error.
#[proc_macro_derive(Module, attributes(parameter, buffer, module))]
pub fn derive_module(_input: TokenStream) -> TokenStream {
    TokenStream::new()
}
