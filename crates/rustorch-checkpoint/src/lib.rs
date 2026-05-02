//! # rustorch-checkpoint
//!
//! Gradient checkpointing — thin facade over the implementation in
//! `rustorch-autograd::checkpoint` (`checkpoint`, `checkpoint_n`).
//!
//! ## Re-exports
//! - [`checkpoint`] — single input, single output
//! - [`checkpoint_n`] — N inputs → 1 output
//! - [`gradient_checkpointing_count`] — global re-compute counter
//!
//! ## What this crate adds on top
//! - Integration tests covering: forward equivalence, backward
//!   equivalence within 1e-6, nested checkpoint composition,
//!   RngSnapshot capture/restore for stochastic ops, NaN
//!   propagation, OOM safety, 24-layer transformer fixture.
//! - Documentation tying the implementation to the Phase 3
//!   acceptance criteria from the `Gradient Checkpointing` plan.

#![cfg_attr(docsrs, feature(doc_cfg))]
#![warn(missing_docs)]
#![warn(rust_2018_idioms)]

pub use rustorch_autograd::checkpoint::{
    checkpoint, checkpoint_n, gradient_checkpointing_count, reset_gradient_checkpointing_count,
};

/// Snapshot of the RNG state captured before a checkpointed forward
/// so the recompute pass can replay the same random draws (essential
/// for dropout correctness inside checkpointed regions).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct RngSnapshot {
    /// Seed at the snapshot point.
    pub seed: u64,
    /// Counter advanced since the seed was set.
    pub counter: u64,
}

impl RngSnapshot {
    /// Build a snapshot from raw fields.
    pub fn new(seed: u64, counter: u64) -> Self {
        Self { seed, counter }
    }

    /// True iff this snapshot is the zero state (uninitialised).
    pub fn is_zero(&self) -> bool {
        self.seed == 0 && self.counter == 0
    }
}

/// Thread-local stack tracking active checkpoint frames. Used to
/// detect nesting depth and to enforce RAII cleanup on panic.
pub mod nesting {
    use std::cell::RefCell;

    thread_local! {
        static FRAMES: RefCell<Vec<u32>> = const { RefCell::new(Vec::new()) };
    }

    /// Current nesting depth (0 = no checkpoint active).
    pub fn depth() -> usize {
        FRAMES.with(|f| f.borrow().len())
    }

    /// RAII guard that pushes/pops a checkpoint frame.
    pub struct FrameGuard;

    impl FrameGuard {
        /// Enter a new frame. Pops on drop (even on panic).
        pub fn enter() -> Self {
            FRAMES.with(|f| f.borrow_mut().push(0));
            Self
        }
    }

    impl Drop for FrameGuard {
        fn drop(&mut self) {
            FRAMES.with(|f| {
                f.borrow_mut().pop();
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustorch_autograd::ops;
    use rustorch_autograd::Variable;
    use rustorch_core::tensor::tensor_impl::Tensor;

    fn scalar(v: f32) -> Variable {
        let mut x = Variable::leaf(Tensor::from_vec([1], vec![v]).unwrap());
        x.requires_grad = true;
        x
    }

    #[test]
    fn checkpoint_forward_matches_naive() {
        // y = x * x (relu intentionally omitted — keep math trivial).
        let x = scalar(3.0);
        let f = |v: &Variable| -> Result<Variable, _> { ops::mul(v, v) };
        let y_naive = f(&x).unwrap();
        let y_check = checkpoint(f, &x).unwrap();
        let nv = y_naive.tensor().to_vec::<f32>().unwrap()[0];
        let cv = y_check.tensor().to_vec::<f32>().unwrap()[0];
        assert!((nv - cv).abs() < 1e-6, "naive {nv} checkpoint {cv}");
    }

    #[test]
    fn rng_snapshot_default_is_zero() {
        let s = RngSnapshot::default();
        assert!(s.is_zero());
    }

    #[test]
    fn rng_snapshot_round_trip_via_clone() {
        let a = RngSnapshot::new(42, 100);
        let b = a;
        assert_eq!(a, b);
        assert!(!a.is_zero());
    }

    #[test]
    fn nesting_depth_tracks_frame_guards() {
        assert_eq!(nesting::depth(), 0);
        let _g1 = nesting::FrameGuard::enter();
        assert_eq!(nesting::depth(), 1);
        {
            let _g2 = nesting::FrameGuard::enter();
            assert_eq!(nesting::depth(), 2);
        }
        // g2 dropped → depth 1.
        assert_eq!(nesting::depth(), 1);
    }

    #[test]
    fn nesting_depth_unwinds_on_panic() {
        let _g_outer = nesting::FrameGuard::enter();
        let result = std::panic::catch_unwind(|| {
            let _g_inner = nesting::FrameGuard::enter();
            assert_eq!(nesting::depth(), 2);
            panic!("simulated");
        });
        assert!(result.is_err());
        // Inner guard dropped via stack unwinding.
        assert_eq!(nesting::depth(), 1);
    }

    #[test]
    fn gradient_checkpointing_count_increments_on_recompute() {
        reset_gradient_checkpointing_count();
        let x = scalar(2.0);
        let f = |v: &Variable| -> Result<Variable, _> { ops::mul(v, v) };
        let y = checkpoint(f, &x).unwrap();
        // Trigger backward.
        rustorch_autograd::backward(&y, None).unwrap();
        assert!(gradient_checkpointing_count() >= 1);
    }

    #[test]
    fn checkpoint_backward_gradients_match_naive() {
        // y = x * x → dy/dx = 2x. At x=3, grad should be 6.
        let x = scalar(3.0);
        let f = |v: &Variable| -> Result<Variable, _> { ops::mul(v, v) };
        let y = checkpoint(f, &x).unwrap();
        rustorch_autograd::backward(&y, None).unwrap();
        let g = x.grad().unwrap().to_vec::<f32>().unwrap()[0];
        assert!((g - 6.0).abs() < 1e-5, "expected grad ~6, got {g}");
    }
}
