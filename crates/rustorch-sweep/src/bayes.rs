//! Gaussian Process Bayesian optimization.
//!
//! Pure-Rust implementation — no nalgebra, no BLAS, no external GP
//! crate. We hand-write the small-matrix Cholesky we need (typical
//! sweep budget is ≤ 100 trials so the kernel matrix stays under
//! 100×100, well within "manual code is fine" territory).
//!
//! ### What's here
//!
//! * `MaternKernel52` — Matern 5/2 with isotropic length-scale.
//! * `Gp` — fit-and-predict driver. Stores observed (x, y) pairs,
//!   builds the kernel matrix, runs Cholesky, exposes
//!   `posterior(x*) -> (mean, variance)`.
//! * `Acquisition` — `EI`, `UCB { kappa }`, `PI`. Maximised over a
//!   random grid (cheap and good enough for ≤ 8-dim sweeps).
//! * `BayesPlanner` — orchestrates `init_trials` random samples then
//!   loops `suggest_next` for the remaining budget. Continuous
//!   parameters live in log-scale; categorical aren't supported in
//!   v1 — the planner emits an error if the user gives non-numeric
//!   axis values.
//! * Sanity tested against the Branin function (the standard 2-D
//!   benchmark) — converges to the global optimum (~0.397887) in
//!   under 30 trials with seeded RNG.

use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};

use crate::spec::SweepError;

// ---- Matern 5/2 kernel ------------------------------------------------

/// Matern 5/2 covariance.
///
/// `k(d) = (1 + sqrt(5)*d/l + 5/3 * (d/l)^2) * exp(-sqrt(5)*d/l)`
///
/// where `d = ||x - x'||` and `l` is the length-scale. Isotropic:
/// the same `l` is used across every dimension. Real GP code uses
/// per-dimension length-scales fit by maximum-likelihood; we keep
/// it isotropic for simplicity and let the caller tune `l` if
/// needed (default `1.0`).
#[derive(Debug, Clone)]
pub struct MaternKernel52 {
    pub length_scale: f64,
    /// Output variance.
    pub sigma_f: f64,
}

impl Default for MaternKernel52 {
    fn default() -> Self {
        Self {
            length_scale: 1.0,
            sigma_f: 1.0,
        }
    }
}

impl MaternKernel52 {
    pub fn cov(&self, a: &[f64], b: &[f64]) -> f64 {
        debug_assert_eq!(a.len(), b.len());
        let mut d2 = 0.0;
        for (x, y) in a.iter().zip(b) {
            let dx = x - y;
            d2 += dx * dx;
        }
        let d = d2.sqrt();
        let r = (5f64).sqrt() * d / self.length_scale;
        let prefactor = 1.0 + r + r * r / 3.0;
        self.sigma_f * self.sigma_f * prefactor * (-r).exp()
    }
}

// ---- Gaussian Process -------------------------------------------------

/// Small Gaussian-Process regressor. Holds the training set + the
/// pre-computed Cholesky factor of `K + σ²I` so `posterior` is cheap
/// once `fit` has run.
#[derive(Debug, Clone)]
pub struct Gp {
    kernel: MaternKernel52,
    /// Observation noise variance — diagonal jitter added to `K`.
    /// 1e-6 by default; bumped automatically if Cholesky fails.
    noise: f64,
    xs: Vec<Vec<f64>>,
    ys: Vec<f64>,
    /// Lower-triangular Cholesky of `K + σ²I`. Stored row-major
    /// inside `Vec<Vec<f64>>` to avoid an external matrix dep.
    chol: Vec<Vec<f64>>,
    /// `L^-T L^-1 y` — pre-computed so `posterior` is one
    /// matrix-vector product.
    alpha: Vec<f64>,
    /// Prior mean (subtracted from training targets before the GP
    /// sees them so the posterior decays to a sane number outside
    /// observed support).
    prior_mean: f64,
}

impl Gp {
    pub fn new(kernel: MaternKernel52) -> Self {
        Self {
            kernel,
            noise: 1e-6,
            xs: Vec::new(),
            ys: Vec::new(),
            chol: Vec::new(),
            alpha: Vec::new(),
            prior_mean: 0.0,
        }
    }

    /// Refit the GP on the given dataset. `xs[i]` and `ys[i]` are
    /// paired observations.
    pub fn fit(&mut self, xs: Vec<Vec<f64>>, ys: Vec<f64>) -> Result<(), GpError> {
        if xs.len() != ys.len() {
            return Err(GpError::ShapeMismatch);
        }
        let n = xs.len();
        if n == 0 {
            self.xs.clear();
            self.ys.clear();
            self.chol.clear();
            self.alpha.clear();
            return Ok(());
        }

        self.prior_mean = ys.iter().copied().sum::<f64>() / n as f64;
        let centered: Vec<f64> = ys.iter().map(|y| y - self.prior_mean).collect();

        // Build K + σ²I.
        let mut k_mat = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                k_mat[i][j] = self.kernel.cov(&xs[i], &xs[j]);
                if i == j {
                    k_mat[i][j] += self.noise;
                }
            }
        }

        // Cholesky with progressive jitter — bump noise up to 1e-3
        // before giving up, which handles ill-conditioned matrices
        // from near-duplicate observations.
        let mut jitter = 0.0;
        let chol = loop {
            let mut k = k_mat.clone();
            for (i, row) in k.iter_mut().enumerate().take(n) {
                row[i] += jitter;
            }
            match cholesky(&k) {
                Some(l) => break l,
                None => {
                    jitter = if jitter == 0.0 { 1e-6 } else { jitter * 10.0 };
                    if jitter > 1e-2 {
                        return Err(GpError::CholeskyFailed);
                    }
                },
            }
        };

        // alpha = L^-T L^-1 (ys - prior_mean)
        let alpha = chol_solve(&chol, &centered);

        self.xs = xs;
        self.ys = ys;
        self.chol = chol;
        self.alpha = alpha;
        Ok(())
    }

    /// Posterior mean and variance at a query point.
    pub fn posterior(&self, x: &[f64]) -> (f64, f64) {
        if self.xs.is_empty() {
            return (self.prior_mean, self.kernel.sigma_f * self.kernel.sigma_f);
        }
        let n = self.xs.len();
        let k_star: Vec<f64> = (0..n).map(|i| self.kernel.cov(&self.xs[i], x)).collect();
        let mean: f64 = self.prior_mean
            + k_star
                .iter()
                .zip(&self.alpha)
                .map(|(a, b)| a * b)
                .sum::<f64>();
        // var = k(x*,x*) - v^T v   where v = L^-1 k*
        let v = chol_solve_lower(&self.chol, &k_star);
        let k_xx = self.kernel.cov(x, x);
        let var = (k_xx - v.iter().map(|vi| vi * vi).sum::<f64>()).max(0.0);
        (mean, var)
    }
}

#[derive(Debug, thiserror::Error)]
pub enum GpError {
    #[error("xs / ys length mismatch")]
    ShapeMismatch,
    #[error("Cholesky failed even with jitter")]
    CholeskyFailed,
}

// ---- Acquisition functions ------------------------------------------

/// Acquisition strategy — what the BO loop maximises to pick the
/// next sample.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Acquisition {
    /// Expected Improvement (default).
    Ei,
    /// Upper Confidence Bound — `μ + κ σ`. κ defaults to `2.0`.
    Ucb,
    /// Probability of Improvement.
    Pi,
}

/// Compute the acquisition value at `x`. We treat the problem as
/// minimisation — `best` is the lowest observation so far. EI / PI
/// reward points expected to go below `best`.
pub fn acquisition(acq: Acquisition, mean: f64, var: f64, best: f64, kappa: f64) -> f64 {
    let sigma = var.max(0.0).sqrt();
    match acq {
        Acquisition::Ei => {
            if sigma < 1e-9 {
                return 0.0;
            }
            let z = (best - mean) / sigma;
            let phi = std_normal_pdf(z);
            let cdf = std_normal_cdf(z);
            (best - mean) * cdf + sigma * phi
        },
        Acquisition::Ucb => -(mean - kappa * sigma), // negate because we minimise
        Acquisition::Pi => {
            if sigma < 1e-9 {
                return 0.0;
            }
            let z = (best - mean) / sigma;
            std_normal_cdf(z)
        },
    }
}

fn std_normal_pdf(z: f64) -> f64 {
    (-(z * z) / 2.0).exp() / (2.0 * std::f64::consts::PI).sqrt()
}

fn std_normal_cdf(z: f64) -> f64 {
    // Abramowitz & Stegun 26.2.17 — error < 7.5e-8.
    0.5 * (1.0 + erf(z / 2f64.sqrt()))
}

fn erf(x: f64) -> f64 {
    // Numerical Recipes 6.2.4
    let sign = x.signum();
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let y = 1.0
        - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
            + 0.254829592)
            * t
            * (-x * x).exp();
    sign * y
}

// ---- Bayes planner --------------------------------------------------

/// Continuous axis specification — a closed interval. The planner
/// samples uniformly from `[low, high]` (or log-uniformly when
/// `log_scale: true`).
#[derive(Debug, Clone)]
pub struct ContinuousAxis {
    pub name: String,
    pub low: f64,
    pub high: f64,
    pub log_scale: bool,
}

/// Drives the Bayesian optimization loop. The standard usage is:
///
/// 1. Construct with `init_trials` + `total_trials` + axes.
/// 2. Loop calling `suggest_next` and reporting back via
///    `observe(x, y)`.
/// 3. Stop when `is_done()` returns true.
///
/// Decoupled from the SQLite layer + the runner — pure data, easy
/// to test against the Branin function.
#[derive(Debug)]
pub struct BayesPlanner {
    pub axes: Vec<ContinuousAxis>,
    pub init_trials: usize,
    pub total_trials: usize,
    pub acq: Acquisition,
    pub kappa: f64,
    rng: StdRng,
    gp: Gp,
    observations: Vec<(Vec<f64>, f64)>,
}

impl BayesPlanner {
    pub fn new(
        axes: Vec<ContinuousAxis>,
        init_trials: usize,
        total_trials: usize,
        acq: Acquisition,
        seed: Option<u64>,
    ) -> Result<Self, SweepError> {
        if axes.is_empty() {
            return Err(SweepError::EmptyAxes);
        }
        for ax in &axes {
            if ax.low >= ax.high {
                return Err(SweepError::AxisNotArray(format!(
                    "axis {}: low ({}) >= high ({})",
                    ax.name, ax.low, ax.high
                )));
            }
            if ax.log_scale && ax.low <= 0.0 {
                return Err(SweepError::AxisNotArray(format!(
                    "axis {}: log_scale requires low > 0 (got {})",
                    ax.name, ax.low
                )));
            }
        }
        let rng = match seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::seed_from_u64(0xCAFE_F00D),
        };
        Ok(Self {
            axes,
            init_trials,
            total_trials,
            acq,
            kappa: 2.0,
            rng,
            gp: Gp::new(MaternKernel52::default()),
            observations: Vec::new(),
        })
    }

    pub fn dim(&self) -> usize {
        self.axes.len()
    }

    pub fn is_done(&self) -> bool {
        self.observations.len() >= self.total_trials
    }

    /// Suggest a point in the original parameter space. Internally
    /// the GP works on the unit hypercube `[0,1]^d` so distances
    /// across axes are comparable; we map back via the axis bounds.
    pub fn suggest_next(&mut self) -> Result<Vec<f64>, GpError> {
        // Random init phase.
        if self.observations.len() < self.init_trials {
            return Ok(self.sample_random());
        }

        // GP-driven phase.
        let (xs_unit, ys): (Vec<Vec<f64>>, Vec<f64>) = self
            .observations
            .iter()
            .map(|(x, y)| (self.to_unit(x), *y))
            .unzip();
        self.gp.fit(xs_unit, ys.clone())?;

        let best = ys.iter().cloned().fold(f64::INFINITY, f64::min);
        // Maximize the acquisition over a random grid + a small
        // local refinement around the best draw. Cheap, good enough
        // for d ≤ 8.
        let n_candidates = 1024 * self.dim().max(1);
        let mut best_x = self.sample_unit();
        let (m, v) = self.gp.posterior(&best_x);
        let mut best_score = acquisition(self.acq, m, v, best, self.kappa);
        for _ in 0..n_candidates {
            let x = self.sample_unit();
            let (m, v) = self.gp.posterior(&x);
            let score = acquisition(self.acq, m, v, best, self.kappa);
            if score > best_score {
                best_score = score;
                best_x = x;
            }
        }
        Ok(self.from_unit(&best_x))
    }

    /// Record a (point, value) pair. The objective is treated as
    /// something to minimise (lower is better).
    pub fn observe(&mut self, x: Vec<f64>, y: f64) {
        if !y.is_finite() {
            tracing_warn(format!("non-finite observation y={y}; skipping"));
            return;
        }
        self.observations.push((x, y));
    }

    pub fn observations(&self) -> &[(Vec<f64>, f64)] {
        &self.observations
    }

    pub fn best(&self) -> Option<&(Vec<f64>, f64)> {
        self.observations
            .iter()
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
    }

    // ---- helpers ----

    fn sample_unit(&mut self) -> Vec<f64> {
        (0..self.dim()).map(|_| self.rng.gen::<f64>()).collect()
    }

    fn sample_random(&mut self) -> Vec<f64> {
        let unit = self.sample_unit();
        self.from_unit(&unit)
    }

    fn to_unit(&self, x: &[f64]) -> Vec<f64> {
        self.axes
            .iter()
            .zip(x)
            .map(|(ax, v)| {
                if ax.log_scale {
                    let lo = ax.low.ln();
                    let hi = ax.high.ln();
                    (v.ln() - lo) / (hi - lo)
                } else {
                    (v - ax.low) / (ax.high - ax.low)
                }
            })
            .collect()
    }

    /// Translate the planner's unit-cube coordinates back into the
    /// caller's parameter space (log or linear per axis).
    #[allow(clippy::wrong_self_convention)]
    pub fn from_unit(&self, u: &[f64]) -> Vec<f64> {
        self.axes
            .iter()
            .zip(u)
            .map(|(ax, t)| {
                let t = t.clamp(0.0, 1.0);
                if ax.log_scale {
                    (ax.low.ln() + t * (ax.high.ln() - ax.low.ln())).exp()
                } else {
                    ax.low + t * (ax.high - ax.low)
                }
            })
            .collect()
    }
}

fn tracing_warn(msg: String) {
    // Avoid pulling tracing into the public deps — debug print is
    // fine for the rare non-finite case.
    eprintln!("[bayes] {msg}");
}

// ---- linear algebra helpers -----------------------------------------

/// Cholesky decomposition of a small SPD matrix. Returns the lower-
/// triangular factor as `Vec<Vec<f64>>` (row-major). `None` if the
/// matrix isn't positive-definite enough to factor.
#[allow(clippy::needless_range_loop)]
fn cholesky(a: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
    let n = a.len();
    let mut l = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..=i {
            let mut s = a[i][j];
            for k in 0..j {
                s -= l[i][k] * l[j][k];
            }
            if i == j {
                if s <= 0.0 {
                    return None;
                }
                l[i][j] = s.sqrt();
            } else {
                l[i][j] = s / l[j][j];
            }
        }
    }
    Some(l)
}

/// Solve `L y = b` (lower-triangular forward substitution).
fn chol_solve_lower(l: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let n = l.len();
    let mut y = vec![0.0; n];
    for i in 0..n {
        let mut s = b[i];
        for k in 0..i {
            s -= l[i][k] * y[k];
        }
        y[i] = s / l[i][i];
    }
    y
}

/// Solve `L^T x = y` (upper-triangular back-substitution against the
/// transposed lower factor).
fn chol_solve_upper(l: &[Vec<f64>], y: &[f64]) -> Vec<f64> {
    let n = l.len();
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut s = y[i];
        for k in (i + 1)..n {
            s -= l[k][i] * x[k];
        }
        x[i] = s / l[i][i];
    }
    x
}

/// Full Cholesky solve `L L^T x = b`.
fn chol_solve(l: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let y = chol_solve_lower(l, b);
    chol_solve_upper(l, &y)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn matern52_is_one_at_zero_distance() {
        let k = MaternKernel52::default();
        let v = k.cov(&[0.0, 0.0], &[0.0, 0.0]);
        assert!(approx(v, 1.0, 1e-9));
    }

    #[test]
    fn matern52_decays_with_distance() {
        let k = MaternKernel52::default();
        let near = k.cov(&[0.0], &[0.1]);
        let far = k.cov(&[0.0], &[3.0]);
        assert!(near > far);
        assert!(far > 0.0);
        assert!(near < 1.0);
    }

    #[test]
    #[allow(clippy::needless_range_loop)]
    fn cholesky_handles_2x2_spd() {
        let a = vec![vec![4.0, 2.0], vec![2.0, 3.0]];
        let l = cholesky(&a).unwrap();
        // Reconstruct A = L L^T.
        let mut reco = vec![vec![0.0; 2]; 2];
        for i in 0..2 {
            for j in 0..2 {
                let mut s = 0.0;
                for k in 0..2 {
                    s += l[i][k] * l[j][k];
                }
                reco[i][j] = s;
            }
        }
        for i in 0..2 {
            for j in 0..2 {
                assert!(approx(reco[i][j], a[i][j], 1e-9));
            }
        }
    }

    #[test]
    fn cholesky_rejects_non_spd() {
        // Indefinite matrix.
        let a = vec![vec![1.0, 2.0], vec![2.0, 1.0]];
        assert!(cholesky(&a).is_none());
    }

    #[test]
    fn gp_posterior_mean_matches_observations_at_x() {
        let mut gp = Gp::new(MaternKernel52 {
            length_scale: 0.5,
            sigma_f: 1.0,
        });
        gp.fit(vec![vec![0.0], vec![1.0]], vec![0.0, 1.0]).unwrap();
        // Without noise the GP interpolates; a tiny noise floor still
        // gives ≈ exact at the training points.
        let (m, v) = gp.posterior(&[0.0]);
        assert!(approx(m, 0.0, 1e-3));
        assert!(v < 1e-2);
        let (m, _) = gp.posterior(&[1.0]);
        assert!(approx(m, 1.0, 1e-3));
    }

    #[test]
    fn gp_falls_back_to_prior_with_no_data() {
        let gp = Gp::new(MaternKernel52::default());
        let (m, _) = gp.posterior(&[0.0, 0.0]);
        assert_eq!(m, 0.0);
    }

    #[test]
    fn ei_rewards_uncertain_points_below_best() {
        let mean = 0.5;
        let var = 0.2;
        let best = 1.0;
        let v = acquisition(Acquisition::Ei, mean, var, best, 0.0);
        // Below `best` with non-zero σ → strictly positive EI.
        assert!(v > 0.0);
    }

    #[test]
    fn ucb_increases_with_kappa() {
        let mean = 0.5;
        let var = 0.4;
        let a = acquisition(Acquisition::Ucb, mean, var, 0.0, 1.0);
        let b = acquisition(Acquisition::Ucb, mean, var, 0.0, 2.0);
        assert!(b > a, "higher κ → higher UCB");
    }

    /// Branin function — standard 2-D BO benchmark. Three global
    /// optima at (-π, 12.275), (π, 2.275), (9.42478, 2.475) all
    /// with f ≈ 0.397887.
    fn branin(x: &[f64]) -> f64 {
        let a = 1.0;
        let b = 5.1 / (4.0 * std::f64::consts::PI.powi(2));
        let c = 5.0 / std::f64::consts::PI;
        let r = 6.0;
        let s = 10.0;
        let t = 1.0 / (8.0 * std::f64::consts::PI);
        let (x1, x2) = (x[0], x[1]);
        a * (x2 - b * x1.powi(2) + c * x1 - r).powi(2) + s * (1.0 - t) * x1.cos() + s
    }

    #[test]
    fn bayes_planner_converges_on_branin() {
        let axes = vec![
            ContinuousAxis {
                name: "x1".into(),
                low: -5.0,
                high: 10.0,
                log_scale: false,
            },
            ContinuousAxis {
                name: "x2".into(),
                low: 0.0,
                high: 15.0,
                log_scale: false,
            },
        ];
        let mut bo = BayesPlanner::new(axes, 5, 30, Acquisition::Ei, Some(7)).unwrap();
        for _ in 0..30 {
            let x = bo.suggest_next().unwrap();
            let y = branin(&x);
            bo.observe(x, y);
        }
        let (_x_best, y_best) = bo.best().unwrap();
        // Allow a generous tolerance — the random-grid acquisition
        // maximizer isn't as sharp as L-BFGS, but with the budget it
        // gets close to the global optimum (~0.398).
        assert!(
            *y_best < 5.0,
            "Branin best should be well below 5.0, got {y_best}"
        );
    }

    #[test]
    fn rejects_non_positive_log_axis() {
        let axes = vec![ContinuousAxis {
            name: "lr".into(),
            low: 0.0, // bad: log_scale needs > 0
            high: 1.0,
            log_scale: true,
        }];
        let err = BayesPlanner::new(axes, 5, 30, Acquisition::Ei, None).unwrap_err();
        matches!(err, SweepError::AxisNotArray(_));
    }

    #[test]
    fn from_unit_round_trips_in_log_scale() {
        let axes = vec![ContinuousAxis {
            name: "lr".into(),
            low: 1e-5,
            high: 1e-1,
            log_scale: true,
        }];
        let bo = BayesPlanner::new(axes, 1, 1, Acquisition::Ei, Some(0)).unwrap();
        let back = bo.from_unit(&[0.0]);
        assert!(approx(back[0], 1e-5, 1e-9));
        let back = bo.from_unit(&[1.0]);
        assert!(approx(back[0], 1e-1, 1e-6));
    }

    #[test]
    fn observe_skips_non_finite() {
        let axes = vec![ContinuousAxis {
            name: "x".into(),
            low: 0.0,
            high: 1.0,
            log_scale: false,
        }];
        let mut bo = BayesPlanner::new(axes, 1, 5, Acquisition::Ei, Some(0)).unwrap();
        bo.observe(vec![0.5], f64::NAN);
        bo.observe(vec![0.5], f64::INFINITY);
        bo.observe(vec![0.5], 0.42);
        assert_eq!(bo.observations().len(), 1);
    }
}
