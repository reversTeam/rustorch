//! Tensor module — RFC-0002 design (Phase 1, P1.1).
//!
//! This module is being assembled incrementally during P1.1. The final
//! shape (per RFC-0002) is:
//!
//! ```text
//! tensor/
//! ├── mod.rs         — re-exports + Tensor struct (top-level)
//! ├── dtype.rs       — Dtype enum + Element trait     ← landed
//! ├── shape.rs       — Shape newtype + broadcasting   ← upcoming
//! ├── layout.rs      — Layout (shape/strides/dtype)   ← upcoming
//! ├── storage.rs     — Storage enum + Arc<Inner>      ← upcoming
//! ├── version.rs     — VersionCounter                 ← upcoming
//! ├── view.rs        — view/reshape/transpose/permute ← upcoming
//! └── convert.rs     — to(device), to(dtype), …       ← upcoming
//! ```
//!
//! While this module is under construction, the legacy P0.3 `Tensor`
//! struct still lives in `crate::lib` and is the public face of
//! `rustorch_core`. The submodules below ship in parallel and are
//! integrated piece-by-piece.

pub mod dtype;
pub mod layout;
pub mod shape;
pub mod storage;
pub mod version;

pub use dtype::{Dtype, Element};
pub use layout::{Layout, LayoutError};
pub use shape::{BroadcastError, Shape};
pub use storage::{Storage, StorageError};
pub use version::{VersionCounter, VersionSnapshot};
