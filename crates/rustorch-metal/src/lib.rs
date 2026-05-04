//! `rustorch-metal` — Apple Metal direct backend.
//!
//! P3.Z Task J — bypass `wgpu`/`MoltenVK` for the canonical Apple
//! perf path. Direct dispatch via the [`metal`] Rust bindings (same
//! Foundation framework wgpu-hal uses internally) lets us target the
//! Apple Silicon tensor units (`simdgroup_matrix<bfloat,8,8>`) and
//! unified memory (`MTLStorageModeShared`) directly — neither is
//! reachable via Metal-via-`wgpu`.
//!
//! ## Why a direct backend?
//!
//! On M4 Max, Linear 1024² + MSE + AdamW measured against PyTorch:
//!
//! | Stack | ms/step | Why |
//! |---|---:|---|
//! | rustorch-wgpu (Task A complete) | 2.35 | scalar f32 matmul, no SUBGROUP_MATRIX on Metal-via-wgpu |
//! | PyTorch MPS | 0.89 | uses `simdgroup_matrix` via MPSMatrixMultiplication |
//! | rustorch-metal target | **0.4-0.5** | direct `simdgroup_matrix<bfloat,8,8>` + unified memory + commitAndContinue |
//!
//! Goal is **1.8-2.2× FASTER than PyTorch MPS** on the standard
//! training-step bench. The plan acceptance gate for Task J is
//! `< 0.5 ms/step Linear 1024² M4 Max`.
//!
//! ## Status
//!
//! Scaffolding phase — a minimal `MetalBackend` struct that owns a
//! `MTLDevice` and `MTLCommandQueue`, plus a smoke-test add kernel
//! to validate the dispatch pipeline. Subsequent commits will add:
//! - Storage::Metal payload integration (Task A wired up)
//! - `simdgroup_matrix<bfloat,8,8>` matmul kernel
//! - Native flash_attention (port of philipturner/metal-flash-attention)
//! - All forward + backward kernels
//! - FusedAdamW.metal MultiTensorApply<4, 512>
//! - Metal heap caching allocator (3 pools, PyTorch MPS pattern)
//! - commitAndContinue submit batching
//! - Mixed precision (bf16 M3+, f16 + GradScaler M1/M2)

#![warn(missing_docs)]

#[cfg(target_os = "macos")]
pub mod backend;

#[cfg(target_os = "macos")]
pub mod backend_singleton;

#[cfg(target_os = "macos")]
pub mod kernels;

/// Stub re-export so non-macOS targets can build code that
/// references `rustorch_metal::error::MetalError` (the error type)
/// even when no concrete backend is available.
pub mod error;

#[cfg(not(target_os = "macos"))]
pub use error::MetalError;
