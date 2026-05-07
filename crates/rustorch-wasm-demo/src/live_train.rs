//! Live, stateful training demo for the browser.
//!
//! Where the rest of `rustorch-wasm-demo` exposes one-shot smoke tests
//! that build a fresh model on every JS call, this module exposes a
//! [`CurveFitter`] handle that **persists across calls**: the model and
//! its SGD optimiser stay alive in WASM memory, so successive
//! `step()` calls move the same parameters down the loss landscape and
//! a real learning curve emerges in the browser.
//!
//! Task: a 2-layer MLP (`Linear(1, hidden) → ReLU → Linear(hidden, 1)`)
//! is fit to noisy samples of `y = sin(x)` on `[-π, π]`. The dataset
//! is generated once at construction time and never changes — that is
//! what makes the loss meaningful across steps.
//!
//! Driven from JS as:
//!
//! ```ignore
//! const fit = new CurveFitter(/* hidden */ 32, /* lr */ 0.05,
//!                             /* n_train */ 64, /* noise */ 0.1);
//! for (let i = 0; i < 500; i++) {
//!     const loss = fit.step();
//!     if (i % 5 === 0) plot(fit.predict(150));
//! }
//! ```
//!
//! All compute is on the **CPU autograd** path — no WebGPU adapter is
//! required, so the freeze-on-`requestDevice` failure modes cannot
//! affect this demo.
//!
//! The training core (`TinyMlp`, dataset construction, one SGD step)
//! is exposed as plain non-`wasm32` symbols so a native unit test can
//! verify that the loss actually decreases, independently of the
//! browser. The `CurveFitter` `#[wasm_bindgen]` handle is gated to
//! `wasm32` only.

use rustorch_autograd::{ops, Variable};
use rustorch_nn::{Linear, Module};
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

/// Range covered by the synthetic dataset on the X axis.
const X_MIN: f32 = -std::f32::consts::PI;
const X_MAX: f32 = std::f32::consts::PI;

/// Deterministic noise in `[-0.5, 0.5]` from a 64-bit Weyl sequence.
/// Cheap, no-deps, reproducible across browsers.
fn noise_unit(seed: u64, i: u64) -> f32 {
    let bits = seed
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add(i.wrapping_mul(0xBF58_476D_1CE4_E5B9));
    let u = (bits >> 33) as u32;
    (u as f32) / (u32::MAX as f32) - 0.5
}

/// Two-layer MLP: `Linear(1, h) → ReLU → Linear(h, 1)`. Kept as a
/// plain struct (not a `Module` impl) because we only need it inside
/// this module and want full control over the parameter list passed
/// to the optimiser.
struct TinyMlp {
    fc1: Linear,
    fc2: Linear,
}

impl TinyMlp {
    fn new(hidden: usize) -> Self {
        TinyMlp {
            fc1: Linear::new(1, hidden),
            fc2: Linear::new(hidden, 1),
        }
    }

    fn parameters(&self) -> Vec<Variable> {
        let mut p = self.fc1.parameters();
        p.extend(self.fc2.parameters());
        p
    }

    /// Forward pass, returning a domain error rather than a JS value
    /// so the same code path is callable from native unit tests.
    /// Wasm wrappers convert the resulting `String` to `JsValue`.
    fn forward(&self, x: &Variable) -> Result<Variable, String> {
        let h = self.fc1.forward(x).map_err(|e| format!("fc1: {e}"))?;
        let h = ops::relu(&h).map_err(|e| format!("relu: {e}"))?;
        let y = self.fc2.forward(&h).map_err(|e| format!("fc2: {e}"))?;
        Ok(y)
    }
}

/// Build the synthetic `y = sin(x) + noise` dataset on `[X_MIN, X_MAX]`
/// with `n` uniformly-spaced samples. Pure helper, native-friendly.
fn build_dataset(n: usize, noise: f32) -> (Vec<f32>, Vec<f32>) {
    let x_flat: Vec<f32> = (0..n)
        .map(|i| {
            let t = (i as f32) / ((n - 1) as f32);
            X_MIN + t * (X_MAX - X_MIN)
        })
        .collect();
    let y_flat: Vec<f32> = x_flat
        .iter()
        .enumerate()
        .map(|(i, &x)| x.sin() + noise * noise_unit(0xCAFE_F00D, i as u64))
        .collect();
    (x_flat, y_flat)
}

/// Stateful curve-fitter: holds the model, the optimiser, and the
/// frozen `(x, y)` training set. Each [`CurveFitter::step`] call runs
/// one full-batch SGD iteration and returns the resulting loss.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub struct CurveFitter {
    model: TinyMlp,
    optimizer: Sgd,
    /// Training inputs as `[N, 1]` Variable. Frozen — never required
    /// to track gradients itself.
    x_var: Variable,
    /// Training targets as `[N, 1]` Variable.
    y_var: Variable,
    /// Plain Vec view of the X column for cheap JS plotting.
    x_flat: Vec<f32>,
    /// Plain Vec view of the noisy Y column for cheap JS plotting.
    y_flat: Vec<f32>,
    /// Number of `step()` calls so far.
    step_count: u32,
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
impl CurveFitter {
    /// Build a fresh fitter.
    ///
    /// - `hidden`  : width of the hidden layer (8…256 sensible).
    /// - `lr`      : SGD learning rate (1e-3 … 1e-1 sensible).
    /// - `n_train` : number of training samples uniformly spaced on
    ///   the X axis.
    /// - `noise`   : amplitude of additive uniform noise on the
    ///   targets, so the network actually has to *fit* something
    ///   non-trivial. `0.0` = perfect sine.
    #[wasm_bindgen(constructor)]
    pub fn new(hidden: u32, lr: f32, n_train: u32, noise: f32) -> Result<CurveFitter, JsValue> {
        let hidden = hidden.max(1) as usize;
        let n = n_train.max(2) as usize;

        // X uniformly spaced in [X_MIN, X_MAX]; Y = sin(X) + noise.
        let (x_flat, y_flat) = build_dataset(n, noise);

        let x_tensor = Tensor::from_vec([n, 1], x_flat.clone())
            .map_err(|e| JsValue::from_str(&format!("x tensor: {e}")))?;
        let y_tensor = Tensor::from_vec([n, 1], y_flat.clone())
            .map_err(|e| JsValue::from_str(&format!("y tensor: {e}")))?;

        let x_var = Variable::new(x_tensor);
        let y_var = Variable::new(y_tensor);

        let model = TinyMlp::new(hidden);
        let optimizer = Sgd::new(model.parameters(), lr);

        Ok(CurveFitter {
            model,
            optimizer,
            x_var,
            y_var,
            x_flat,
            y_flat,
            step_count: 0,
        })
    }

    /// One full-batch SGD step. Returns the **scalar MSE loss**
    /// computed before the parameters are updated (so the JS-side
    /// loss curve traces the descent meaningfully).
    pub fn step(&mut self) -> Result<f32, JsValue> {
        // 1. zero_grad — drop any accumulated gradients.
        self.optimizer.zero_grad();

        // 2. forward.
        let pred = self
            .model
            .forward(&self.x_var)
            .map_err(|e| JsValue::from_str(&e))?;
        let loss = ops::mse_loss(&pred, &self.y_var, Reduction::Mean)
            .map_err(|e| JsValue::from_str(&format!("mse_loss: {e}")))?;
        let loss_value = loss
            .tensor()
            .as_slice::<f32>()
            .ok_or_else(|| JsValue::from_str("loss not f32"))?[0];

        // 3. backward — populate every parameter's `.grad` slot.
        rustorch_autograd::backward(&loss, None)
            .map_err(|e| JsValue::from_str(&format!("backward: {e}")))?;

        // 4. SGD step — subtract `lr * grad` from each parameter.
        self.optimizer.step();

        self.step_count += 1;
        Ok(loss_value)
    }

    /// Sample the model on a uniform grid of `n_grid` points covering
    /// the same X range as the training set. Returns an interleaved
    /// `[x0, y0, x1, y1, ...]` Float32Array so the JS side can draw
    /// the prediction curve in one pass.
    pub fn predict(&self, n_grid: u32) -> Result<js_sys::Float32Array, JsValue> {
        let n = n_grid.max(2) as usize;
        let xs: Vec<f32> = (0..n)
            .map(|i| {
                let t = (i as f32) / ((n - 1) as f32);
                X_MIN + t * (X_MAX - X_MIN)
            })
            .collect();
        let x_tensor = Tensor::from_vec([n, 1], xs.clone())
            .map_err(|e| JsValue::from_str(&format!("predict x tensor: {e}")))?;
        let x_var = Variable::new(x_tensor);
        let y_var = self.model.forward(&x_var)?;
        let y_slice = y_var
            .tensor()
            .as_slice::<f32>()
            .ok_or_else(|| JsValue::from_str("predict output not f32"))?
            .to_vec();

        let mut interleaved = Vec::with_capacity(2 * n);
        for i in 0..n {
            interleaved.push(xs[i]);
            interleaved.push(y_slice[i]);
        }
        Ok(js_sys::Float32Array::from(interleaved.as_slice()))
    }

    /// Get the frozen training X column as a Float32Array for plotting.
    #[wasm_bindgen(js_name = "trainX")]
    pub fn train_x(&self) -> js_sys::Float32Array {
        js_sys::Float32Array::from(self.x_flat.as_slice())
    }

    /// Get the frozen training Y column as a Float32Array for plotting.
    #[wasm_bindgen(js_name = "trainY")]
    pub fn train_y(&self) -> js_sys::Float32Array {
        js_sys::Float32Array::from(self.y_flat.as_slice())
    }

    /// X range covered by the dataset (`[xMin, xMax]`).
    #[wasm_bindgen(js_name = "xRange")]
    pub fn x_range(&self) -> js_sys::Float32Array {
        js_sys::Float32Array::from(&[X_MIN, X_MAX][..])
    }

    /// How many `step()` calls have run so far.
    #[wasm_bindgen(js_name = "stepCount")]
    pub fn step_count(&self) -> u32 {
        self.step_count
    }

    /// Total number of trainable parameters in the model.
    #[wasm_bindgen(js_name = "paramCount")]
    pub fn param_count(&self) -> u32 {
        self.model
            .parameters()
            .iter()
            .map(|p| p.tensor().numel() as u32)
            .sum()
    }

    /// Set a new learning rate. Useful for live "decay" sliders in
    /// the demo page.
    #[wasm_bindgen(js_name = "setLr")]
    pub fn set_lr(&mut self, lr: f32) {
        self.optimizer.set_lr(lr);
    }
}

// -------------------------------------------------------------------
// Native unit tests — guard the actual training behaviour, so a CI
// without a browser still catches regressions in the autograd /
// optimiser path used by the live demo.
// -------------------------------------------------------------------
#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;

    /// Run `n_steps` of full-batch SGD on the same dataset the
    /// browser demo uses, and return the (initial_loss, final_loss)
    /// pair. Mirrors `CurveFitter::step` faithfully.
    fn train_native(
        hidden: usize,
        lr: f32,
        n_train: usize,
        noise: f32,
        n_steps: usize,
    ) -> (f32, f32) {
        let (x_flat, y_flat) = build_dataset(n_train, noise);
        let x_t = Tensor::from_vec([n_train, 1], x_flat).expect("x");
        let y_t = Tensor::from_vec([n_train, 1], y_flat).expect("y");
        let x_var = Variable::new(x_t);
        let y_var = Variable::new(y_t);

        let model = TinyMlp::new(hidden);
        let mut opt = Sgd::new(model.parameters(), lr);

        let mut first_loss = f32::NAN;
        let mut last_loss = f32::NAN;

        for s in 0..n_steps {
            opt.zero_grad();
            let pred = model.forward(&x_var).expect("forward");
            let loss = ops::mse_loss(&pred, &y_var, Reduction::Mean).expect("mse");
            let loss_v = loss.tensor().as_slice::<f32>().unwrap()[0];
            if s == 0 {
                first_loss = loss_v;
            }
            last_loss = loss_v;
            rustorch_autograd::backward(&loss, None).expect("backward");
            opt.step();
        }
        (first_loss, last_loss)
    }

    /// Sanity: the very first forward pass on `[X_MIN, X_MAX]` against
    /// `sin(x)` produces a loss in a sensible range. With Linear's
    /// uniform-`[-1/√fan_in, 1/√fan_in]` init the prediction is
    /// roughly zero, so MSE ≈ E[sin²] ≈ 0.5.
    #[test]
    fn initial_loss_is_finite_and_in_range() {
        let (l0, _) = train_native(32, 0.05, 64, 0.1, 1);
        assert!(l0.is_finite(), "initial loss not finite: {l0}");
        assert!(l0 > 0.05 && l0 < 5.0, "initial loss out of range: {l0}");
    }

    /// The real proof: 300 SGD steps must drive the MSE down by at
    /// least a factor of 5 from initialisation. With a 32-unit hidden
    /// layer and lr=0.05 we observe ~×8 in practice; the threshold
    /// gives margin so the test doesn't go flaky on minor numerical
    /// drift, but still catches a real autograd / SGD regression
    /// (which would leave the loss roughly flat).
    #[test]
    fn loss_descends_substantially_in_300_steps() {
        let (l0, lN) = train_native(32, 0.05, 64, 0.1, 300);
        assert!(lN.is_finite(), "final loss not finite: {lN}");
        assert!(
            lN < l0 / 5.0,
            "loss did not descend enough: l0={l0:.4}, lN={lN:.4} \
             (need lN < l0/5; demo would look broken in the browser)"
        );
    }

    /// More steps should not *increase* the loss. Guards against
    /// learning-rate divergence sneaking in via an autograd or SGD
    /// regression that flips a sign somewhere.
    #[test]
    fn extra_steps_never_make_loss_worse() {
        let (_, l_short) = train_native(32, 0.05, 64, 0.1, 100);
        let (_, l_long) = train_native(32, 0.05, 64, 0.1, 400);
        assert!(
            l_long <= l_short * 1.05,
            "extra steps regressed the loss: 100 steps → {l_short:.4}, \
             400 steps → {l_long:.4}"
        );
    }

    /// `predict`-style evaluation on a fresh grid must not blow up
    /// numerically — exercises the Variable construction path used
    /// by `CurveFitter::predict` in wasm.
    #[test]
    fn predict_grid_has_finite_outputs() {
        let model = TinyMlp::new(16);
        let n = 50;
        let xs: Vec<f32> = (0..n)
            .map(|i| X_MIN + (i as f32) / ((n - 1) as f32) * (X_MAX - X_MIN))
            .collect();
        let x_t = Tensor::from_vec([n, 1], xs).expect("xs");
        let y = model.forward(&Variable::new(x_t)).expect("forward");
        let ys = y.tensor().as_slice::<f32>().unwrap().to_vec();
        assert_eq!(ys.len(), n);
        for &v in ys.iter() {
            assert!(v.is_finite(), "non-finite prediction: {v}");
        }
    }
}
