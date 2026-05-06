//! # rustorch-amp
//!
//! Mixed-precision primitives for rustorch — `bf16` / `fp16`
//! conversion, an `Autocast` scope guard with op promotion list, a
//! `GradScaler` for fp16 loss scaling, and scalar bf16/fp16 matmul
//! kernels with f32 accumulator.
//!
//! ## What this crate ships
//! - [`dtypes`] — `f32 ↔ bf16` and `f32 ↔ fp16` conversions via the
//!   workspace `half` crate; preserves NaN/Inf/-0; round-to-nearest-even.
//! - [`autocast`] — thread-local `Autocast::Bf16 | Fp16 | None` guard
//!   + `OpKind` promotion list (matmul/conv → low-prec, softmax/norm/loss → f32)
//! - [`grad_scaler`] — `GradScaler` state machine matching PyTorch's
//!   API (`init_scale=65536`, `growth_factor=2.0`, `backoff_factor=0.5`,
//!   `growth_interval=2000`)
//! - [`bf16_kernels`] — scalar bf16/fp16 matmul with f32 accumulator
//!
//! ## What's deferred
//! - `Module::to_bf16() / to_fp16()` — needs `rustorch-nn` integration
//!   (recursing over Module trait); separate task
//! - End-to-end mixed-precision TRAINING — needs autograd custom-Function
//!   API + GradScaler hooks into the optimizer step (see Gradient
//!   Checkpointing's autograd-integration task)
//! - SIMD specialisations (AVX-512 BF16 / NEON) — arch-specific commits

#![cfg_attr(docsrs, feature(doc_cfg))]
#![warn(missing_docs)]
#![warn(rust_2018_idioms)]

#[cfg(target_os = "macos")]
pub mod accelerate;
pub mod autocast;
pub mod bf16_kernels;
pub mod dtypes;
pub mod grad_scaler;
pub mod model_cast;

pub use autocast::{autocast, autocast_force_fp32, current_dtype, AutocastGuard, Mixed, OpKind};
pub use bf16_kernels::{matmul_bf16_with_f32_accum, matmul_fp16_with_f32_accum};
pub use dtypes::{f32_to_bf16, f32_to_fp16};
pub use grad_scaler::{GradScaler, ScalerStep};
pub use model_cast::{
    f32_param_bytes, low_prec_param_bytes, parameters_from_bf16, parameters_from_fp16,
    parameters_to_bf16, parameters_to_fp16,
};

/// Crate version reported at runtime.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
