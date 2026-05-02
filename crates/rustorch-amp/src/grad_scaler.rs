//! GradScaler — fp16 loss scaling state machine matching PyTorch's
//! `torch.cuda.amp.GradScaler` API.
//!
//! Why scaling: fp16 has range ~6e-8 to 65504. Loss gradients smaller
//! than 6e-8 underflow to zero. Multiplying the loss by a large scale
//! before backward shifts gradients into the representable range;
//! after backward, gradients are unscaled before the optimizer step.
//!
//! Why not bf16: bf16 has the SAME exponent range as f32 (~1e-38 to
//! 3e38), so underflow is a non-issue. GradScaler is fp16-only.

/// Outcome of `GradScaler::step` indicating whether the optimizer
/// step was applied or skipped due to inf/nan in gradients.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScalerStep {
    /// Step applied — gradients were finite, optimizer state advanced.
    Applied,
    /// Step skipped — inf/nan detected; scale was halved via
    /// `backoff_factor`.
    Skipped,
}

/// fp16 loss-scaling state. Behaviour mirrors PyTorch:
/// - `init_scale = 65536` (2^16)
/// - `growth_factor = 2.0` (double after a clean window)
/// - `backoff_factor = 0.5` (halve on overflow detection)
/// - `growth_interval = 2000` (steps between growth attempts)
/// - Scale is clamped to `[1.0, 2^24]`.
#[derive(Debug, Clone)]
pub struct GradScaler {
    scale: f32,
    growth_factor: f32,
    backoff_factor: f32,
    growth_interval: u32,
    growth_tracker: u32,
}

impl GradScaler {
    /// Build a GradScaler with PyTorch defaults.
    pub fn new() -> Self {
        Self {
            scale: 65536.0,
            growth_factor: 2.0,
            backoff_factor: 0.5,
            growth_interval: 2000,
            growth_tracker: 0,
        }
    }

    /// Build a GradScaler with custom parameters.
    pub fn with_params(
        init_scale: f32,
        growth_factor: f32,
        backoff_factor: f32,
        growth_interval: u32,
    ) -> Self {
        Self {
            scale: init_scale,
            growth_factor,
            backoff_factor,
            growth_interval,
            growth_tracker: 0,
        }
    }

    /// Current scale value.
    pub fn current_scale(&self) -> f32 {
        self.scale
    }

    /// Current growth tracker (consecutive clean steps).
    pub fn growth_tracker(&self) -> u32 {
        self.growth_tracker
    }

    /// Multiply a loss value by the current scale. The autograd
    /// backward of the scaled loss yields scaled gradients — the
    /// caller (or `unscale`) divides by `scale` afterward.
    pub fn scale_loss(&self, loss: f32) -> f32 {
        loss * self.scale
    }

    /// Inspect a slice of gradients for inf / nan and decide whether
    /// to apply the optimizer step. On success unscales the
    /// gradients in place; on failure leaves them untouched and
    /// applies the backoff factor.
    pub fn step(&mut self, grads: &mut [f32]) -> ScalerStep {
        let any_bad = grads.iter().any(|g| !g.is_finite());
        if any_bad {
            self.scale = (self.scale * self.backoff_factor).max(1.0);
            self.growth_tracker = 0;
            ScalerStep::Skipped
        } else {
            // Unscale grads in place so the optimizer sees
            // fp32-equivalent gradients.
            let inv_scale = 1.0 / self.scale;
            for g in grads.iter_mut() {
                *g *= inv_scale;
            }
            self.growth_tracker += 1;
            ScalerStep::Applied
        }
    }

    /// Apply the growth schedule: if `growth_interval` consecutive
    /// clean steps have elapsed, multiply the scale by
    /// `growth_factor`. Clamp to `[1, 2^24]`.
    pub fn update(&mut self) {
        if self.growth_tracker >= self.growth_interval {
            self.scale = (self.scale * self.growth_factor).clamp(1.0, (1u32 << 24) as f32);
            self.growth_tracker = 0;
        }
    }
}

impl Default for GradScaler {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_pytorch() {
        let s = GradScaler::new();
        assert_eq!(s.current_scale(), 65536.0);
        assert_eq!(s.growth_tracker(), 0);
    }

    #[test]
    fn scale_loss_multiplies_by_scale() {
        let s = GradScaler::new();
        assert_eq!(s.scale_loss(1.0), 65536.0);
        assert_eq!(s.scale_loss(0.5), 32768.0);
    }

    #[test]
    fn step_applied_unscales_gradients_in_place() {
        let mut s = GradScaler::new();
        let mut grads = vec![65536.0f32, -65536.0, 32768.0];
        let outcome = s.step(&mut grads);
        assert_eq!(outcome, ScalerStep::Applied);
        assert_eq!(grads, vec![1.0f32, -1.0, 0.5]);
        assert_eq!(s.growth_tracker(), 1);
    }

    #[test]
    fn step_skipped_when_grad_is_inf() {
        let mut s = GradScaler::new();
        let original_scale = s.current_scale();
        let mut grads = vec![1.0f32, f32::INFINITY, 1.0];
        let outcome = s.step(&mut grads);
        assert_eq!(outcome, ScalerStep::Skipped);
        // Scale halved.
        assert_eq!(s.current_scale(), original_scale * 0.5);
        assert_eq!(s.growth_tracker(), 0);
    }

    #[test]
    fn step_skipped_when_grad_is_nan() {
        let mut s = GradScaler::new();
        let mut grads = vec![1.0f32, f32::NAN, 1.0];
        assert_eq!(s.step(&mut grads), ScalerStep::Skipped);
    }

    #[test]
    fn scale_clamped_at_lower_bound() {
        let mut s = GradScaler::with_params(2.0, 2.0, 0.5, 2000);
        let mut bad_grads = vec![f32::NAN];
        s.step(&mut bad_grads); // 2 → 1
        s.step(&mut bad_grads); // would be 0.5 → clamped to 1
        assert_eq!(s.current_scale(), 1.0);
    }

    #[test]
    fn growth_doubles_scale_after_interval() {
        let mut s = GradScaler::with_params(1.0, 2.0, 0.5, 3);
        for _ in 0..3 {
            s.step(&mut [0.5f32]);
        }
        assert_eq!(s.growth_tracker(), 3);
        s.update();
        assert_eq!(s.current_scale(), 2.0);
        assert_eq!(s.growth_tracker(), 0);
    }

    #[test]
    fn growth_clamped_at_upper_bound() {
        let mut s = GradScaler::with_params((1u32 << 23) as f32, 2.0, 0.5, 1);
        s.step(&mut [0.5f32]);
        s.update();
        // Would double to 2^24 — exactly the cap.
        assert_eq!(s.current_scale(), (1u32 << 24) as f32);
    }

    #[test]
    fn growth_does_not_trigger_before_interval() {
        let mut s = GradScaler::with_params(1.0, 2.0, 0.5, 5);
        s.step(&mut [0.5f32]);
        s.step(&mut [0.5f32]);
        s.update();
        assert_eq!(s.current_scale(), 1.0); // not grown yet
    }
}
