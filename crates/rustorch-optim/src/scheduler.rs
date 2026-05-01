//! Learning-rate schedulers (P1.7).
//!
//! Schedulers are decoupled from [`crate::Optimizer`]: they own a
//! reference to the optimiser only when their `step()` is called, and
//! they read/write the LR via [`Optimizer::lr`] / [`Optimizer::set_lr`].
//!
//! This v1 ships the four most common schedulers:
//! - [`StepLR`] — multiply LR by `gamma` every `step_size` steps.
//! - [`ExponentialLR`] — multiply LR by `gamma` every step.
//! - [`CosineAnnealingLR`] — cosine decay from `lr_max` to `lr_min` over
//!   `t_max` steps.
//! - [`LinearWarmup`] — linear ramp from 0 to `target_lr` over
//!   `warmup_steps`, then constant.
//!
//! Other schedulers (MultiStepLR, OneCycleLR, ReduceLROnPlateau,
//! LambdaLR, CosineAnnealingWarmRestarts) plug into the same trait —
//! they remain pending per the project plan.

use crate::Optimizer;

/// Common interface for LR schedulers. Each [`step`] advances internal
/// state by one step and updates the optimiser's LR via
/// [`Optimizer::set_lr`]. [`get_lr`] returns the LR that will be set on
/// the next call without advancing.
pub trait LrScheduler {
    /// Advance the scheduler by one step and update the optimiser.
    fn step(&mut self, optim: &mut dyn Optimizer);

    /// Return the LR for the current step (does not advance).
    fn get_lr(&self) -> f32;

    /// Number of steps taken so far.
    fn step_count(&self) -> usize;
}

// ------------------------------ StepLR ------------------------------

/// Multiply LR by `gamma` every `step_size` steps. The very first
/// `step()` call sets the LR to `base_lr`; subsequent calls compute
/// `base_lr * gamma^(step_count // step_size)`.
pub struct StepLR {
    base_lr: f32,
    step_size: usize,
    gamma: f32,
    step_count: usize,
}

impl StepLR {
    /// Build a StepLR. `base_lr` is the LR at step 0.
    pub fn new(base_lr: f32, step_size: usize, gamma: f32) -> Self {
        StepLR {
            base_lr,
            step_size,
            gamma,
            step_count: 0,
        }
    }
}

impl LrScheduler for StepLR {
    fn step(&mut self, optim: &mut dyn Optimizer) {
        self.step_count += 1;
        let lr = self.get_lr();
        optim.set_lr(lr);
    }

    fn get_lr(&self) -> f32 {
        let drops = self.step_count / self.step_size;
        self.base_lr * self.gamma.powi(drops as i32)
    }

    fn step_count(&self) -> usize {
        self.step_count
    }
}

// ------------------------------ ExponentialLR ------------------------------

/// Multiply LR by `gamma` every step.
pub struct ExponentialLR {
    base_lr: f32,
    gamma: f32,
    step_count: usize,
}

impl ExponentialLR {
    /// Build an ExponentialLR. `base_lr` is the LR at step 0.
    pub fn new(base_lr: f32, gamma: f32) -> Self {
        ExponentialLR {
            base_lr,
            gamma,
            step_count: 0,
        }
    }
}

impl LrScheduler for ExponentialLR {
    fn step(&mut self, optim: &mut dyn Optimizer) {
        self.step_count += 1;
        let lr = self.get_lr();
        optim.set_lr(lr);
    }

    fn get_lr(&self) -> f32 {
        self.base_lr * self.gamma.powi(self.step_count as i32)
    }

    fn step_count(&self) -> usize {
        self.step_count
    }
}

// ------------------------------ CosineAnnealingLR ------------------------------

/// Cosine annealing from `lr_max` to `lr_min` over `t_max` steps.
///
/// `lr(t) = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(pi * t / t_max))`
/// for `t in [0, t_max]`. Past `t_max`, the LR stays at `lr_min`.
pub struct CosineAnnealingLR {
    lr_max: f32,
    lr_min: f32,
    t_max: usize,
    step_count: usize,
}

impl CosineAnnealingLR {
    /// Build a CosineAnnealingLR.
    pub fn new(lr_max: f32, lr_min: f32, t_max: usize) -> Self {
        CosineAnnealingLR {
            lr_max,
            lr_min,
            t_max,
            step_count: 0,
        }
    }
}

impl LrScheduler for CosineAnnealingLR {
    fn step(&mut self, optim: &mut dyn Optimizer) {
        self.step_count += 1;
        let lr = self.get_lr();
        optim.set_lr(lr);
    }

    fn get_lr(&self) -> f32 {
        let t = self.step_count.min(self.t_max);
        let cos = (std::f32::consts::PI * t as f32 / self.t_max as f32).cos();
        self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (1.0 + cos)
    }

    fn step_count(&self) -> usize {
        self.step_count
    }
}

// ------------------------------ LinearWarmup ------------------------------

/// Linear ramp from 0 to `target_lr` over `warmup_steps`, then constant.
///
/// Useful as a prefix to another scheduler (e.g. linear warmup followed
/// by cosine decay). v1 stops at `target_lr` after warmup; chaining is
/// up to the user.
pub struct LinearWarmup {
    target_lr: f32,
    warmup_steps: usize,
    step_count: usize,
}

impl LinearWarmup {
    /// Build a LinearWarmup.
    pub fn new(target_lr: f32, warmup_steps: usize) -> Self {
        LinearWarmup {
            target_lr,
            warmup_steps,
            step_count: 0,
        }
    }
}

impl LrScheduler for LinearWarmup {
    fn step(&mut self, optim: &mut dyn Optimizer) {
        self.step_count += 1;
        let lr = self.get_lr();
        optim.set_lr(lr);
    }

    fn get_lr(&self) -> f32 {
        if self.step_count >= self.warmup_steps {
            self.target_lr
        } else {
            self.target_lr * (self.step_count as f32) / (self.warmup_steps as f32)
        }
    }

    fn step_count(&self) -> usize {
        self.step_count
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Sgd;
    use rustorch_autograd::Variable;
    use rustorch_core::tensor::tensor_impl::Tensor;

    fn dummy_optim(lr: f32) -> Sgd {
        let p = Variable::leaf(Tensor::from_vec([1usize], vec![0.0_f32]).unwrap());
        Sgd::new(vec![p], lr)
    }

    #[test]
    fn step_lr_drops_every_n_steps() {
        let mut opt = dummy_optim(0.1);
        let mut sched = StepLR::new(0.1, 5, 0.5);
        for _ in 0..4 {
            sched.step(&mut opt);
        }
        // 4 steps; 4/5 = 0 drops → lr = 0.1
        assert!((opt.lr() - 0.1).abs() < 1e-7);
        sched.step(&mut opt); // step 5; drops = 1 → lr = 0.05
        assert!((opt.lr() - 0.05).abs() < 1e-7);
        for _ in 0..5 {
            sched.step(&mut opt);
        }
        // step 10; drops = 2 → lr = 0.025
        assert!((opt.lr() - 0.025).abs() < 1e-7);
    }

    #[test]
    fn exponential_lr_decays_each_step() {
        let mut opt = dummy_optim(1.0);
        let mut sched = ExponentialLR::new(1.0, 0.9);
        sched.step(&mut opt);
        assert!((opt.lr() - 0.9).abs() < 1e-6);
        sched.step(&mut opt);
        assert!((opt.lr() - 0.81).abs() < 1e-6);
    }

    #[test]
    fn cosine_anneals_to_min() {
        let mut opt = dummy_optim(1.0);
        let mut sched = CosineAnnealingLR::new(1.0, 0.0, 10);
        // After 10 cosine steps with lr_min=0, t/t_max=1, cos(pi)=-1
        // → lr = 0 + 0.5*1*(1+(-1)) = 0
        for _ in 0..10 {
            sched.step(&mut opt);
        }
        assert!(opt.lr() < 1e-6, "cosine end → lr_min, got {}", opt.lr());

        // Beyond t_max, stays at lr_min.
        for _ in 0..3 {
            sched.step(&mut opt);
        }
        assert!(opt.lr() < 1e-6);
    }

    #[test]
    fn cosine_starts_at_max_and_midpoint_is_average() {
        let mut opt = dummy_optim(1.0);
        let mut sched = CosineAnnealingLR::new(1.0, 0.1, 10);
        // After step 0 (no step), get_lr returns 1.0 (max)
        assert!((sched.get_lr() - 1.0).abs() < 1e-6);
        // After 5 steps, t=5, cos(pi/2) = 0 → lr = 0.1 + 0.5*0.9*1 = 0.55
        for _ in 0..5 {
            sched.step(&mut opt);
        }
        assert!((opt.lr() - 0.55).abs() < 1e-5, "got {}", opt.lr());
    }

    #[test]
    fn linear_warmup_ramps_then_holds() {
        let mut opt = dummy_optim(0.0);
        let mut sched = LinearWarmup::new(1.0, 4);
        sched.step(&mut opt);
        assert!((opt.lr() - 0.25).abs() < 1e-6);
        sched.step(&mut opt);
        assert!((opt.lr() - 0.5).abs() < 1e-6);
        sched.step(&mut opt);
        assert!((opt.lr() - 0.75).abs() < 1e-6);
        sched.step(&mut opt);
        assert!((opt.lr() - 1.0).abs() < 1e-6);
        // Past warmup: stays at target_lr.
        sched.step(&mut opt);
        sched.step(&mut opt);
        assert!((opt.lr() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn step_count_tracked_across_schedulers() {
        let mut opt = dummy_optim(0.1);
        let mut sched = StepLR::new(0.1, 1, 0.5);
        for _ in 0..3 {
            sched.step(&mut opt);
        }
        assert_eq!(sched.step_count(), 3);
    }
}
