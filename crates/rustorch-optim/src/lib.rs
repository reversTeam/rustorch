//! # rustorch-optim
//!
//! `Optimizer` trait + concrete impls: SGD, Adam, AdamW, Lion, Adafactor,
//! RMSprop, Adagrad, Adamax, NAdam, RAdam, LBFGS, Adadelta. LR schedulers
//! (StepLR, CosineAnnealingLR, WarmupCosine, OneCycleLR, Plateau, ...).
//!
//! Concrete impl lands in P1.7 (Optimizers).

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
