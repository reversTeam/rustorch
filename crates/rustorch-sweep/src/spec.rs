//! Sweep specification + planner.
//!
//! A `SweepSpec` is a strategy + an ordered list of (key, values)
//! axes. `plan(base_cfg)` resolves it into a `Vec<Trial>` where each
//! trial is the base cfg deep-merged with one combination.
//!
//! Why pure data? The console backend persists the spec in SQLite
//! (`sweeps` table) and the runner consumes the `Trial` list. Both
//! sides need to deserialize the same shape; keeping the planner
//! free of I/O lets us test it under cargo + reuse it from the CLI
//! `rustorch sweep` command.

use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};

/// Crate-wide error type. Distinct from `serde_json::Error` so the
/// caller can branch on intent (bad axis vs. bad json).
#[derive(Debug, thiserror::Error)]
pub enum SweepError {
    /// `add(key, values)` was called with a non-array `values` blob.
    #[error("axis `{0}` must be a JSON array")]
    AxisNotArray(String),
    /// No axes added — every strategy needs at least one.
    #[error("sweep must have at least one axis")]
    EmptyAxes,
    /// `base_cfg` was something other than a JSON object.
    #[error("base cfg must be a JSON object")]
    BaseCfgNotObject,
    /// Wrap json errors when (de)serializing.
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),
}

/// Sugar for handlers that return `Result<T, SweepError>`.
pub type SweepResult<T> = Result<T, SweepError>;

/// One concrete trial in a sweep — the merged config + the axis
/// values that produced it.
#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
pub struct Trial {
    /// ULID-as-string. Stable across the trial's lifetime so the
    /// runner can reuse it as the `run_id` if it wants.
    pub id: String,
    /// Base cfg deep-merged with the per-axis assignments.
    pub cfg: serde_json::Value,
    /// What each axis was set to for this trial — useful for the
    /// Experiments view to re-derive the column projections.
    pub axes: serde_json::Value,
    /// For ASHA: the resource budget (epochs / steps) the trial
    /// should run for at this rung. `None` for grid/random.
    pub rung_budget: Option<u32>,
}

/// Pluggable strategies. `Grid` is the default because it's the
/// only one that produces a finite, deterministic plan.
#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub enum Strategy {
    /// Full Cartesian product over the axes.
    Grid,
    /// Random sampling. `trials` controls how many to draw.
    Random(RandomConfig),
    /// Asynchronous Successive Halving. Returns rung-budget'd trials
    /// in waves; the orchestrator decides which survive each rung.
    Asha(AshaConfig),
    /// Gaussian-process Bayesian optimization. Stub today — falls
    /// back to Random with a warning so callers can ship.
    Bayes(BayesConfig),
}

/// Knobs for `Strategy::Random`.
#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
pub struct RandomConfig {
    /// How many trials to draw. Capped at the Cartesian-product size
    /// to avoid duplicate combinations.
    pub trials: usize,
    /// PRNG seed for reproducibility. `None` = random per-call.
    pub seed: Option<u64>,
}

/// Knobs for `Strategy::Asha`.
#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
pub struct AshaConfig {
    /// Reduction factor — keep top `1/eta` of trials at each rung.
    /// Default 3 from the original paper.
    pub eta: u32,
    /// Number of brackets (Hyperband-style diversity). 1 = pure ASHA.
    pub brackets: u32,
    /// Smallest resource (epochs / steps) the first rung uses.
    pub min_resource: u32,
    /// Maximum resource a surviving trial can run for.
    pub max_resource: u32,
}

/// Knobs for `Strategy::Bayes`. The planner needs continuous axes
/// (low/high pairs); discrete axes from `add(name, [v1,v2,...])`
/// are interpreted as `[min(values), max(values)]` and rounded back
/// to the nearest discrete value at trial-time.
#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
pub struct BayesConfig {
    /// How many random trials to draw before the GP kicks in.
    pub init_trials: usize,
    /// Total budget — the planner returns at most this many trials.
    pub trials: usize,
    /// Acquisition function: `"ei"` | `"ucb"` | `"pi"`.
    pub acquisition: String,
    pub seed: Option<u64>,
}

/// One axis (`name`, list of values).
#[derive(Debug, Clone)]
struct Axis {
    name: String,
    values: Vec<serde_json::Value>,
}

/// Full sweep description. Build via `grid()` / `random()` / `asha()`
/// / `bayes()` factory + `add()` chain calls.
#[derive(Debug, Clone)]
pub struct SweepSpec {
    strategy: Strategy,
    axes: Vec<Axis>,
}

impl SweepSpec {
    /// Builder for `Strategy::Grid`.
    pub fn grid() -> SweepBuilder {
        SweepBuilder::new(Strategy::Grid)
    }

    /// Builder for `Strategy::Random`.
    pub fn random(trials: usize, seed: Option<u64>) -> SweepBuilder {
        SweepBuilder::new(Strategy::Random(RandomConfig { trials, seed }))
    }

    /// Builder for `Strategy::Asha`.
    pub fn asha(cfg: AshaConfig) -> SweepBuilder {
        SweepBuilder::new(Strategy::Asha(cfg))
    }

    /// Builder for `Strategy::Bayes` (falls back to random for now).
    pub fn bayes(cfg: BayesConfig) -> SweepBuilder {
        SweepBuilder::new(Strategy::Bayes(cfg))
    }

    /// What strategy this sweep runs.
    pub fn strategy(&self) -> &Strategy {
        &self.strategy
    }

    /// Total number of distinct combinations the axes can produce —
    /// useful for sizing the random / bayes trial cap.
    pub fn cartesian_size(&self) -> usize {
        self.axes.iter().map(|a| a.values.len()).product()
    }

    /// Resolve the sweep into a flat list of trials.
    pub fn plan(&self, base_cfg: &serde_json::Value) -> SweepResult<Vec<Trial>> {
        if !base_cfg.is_object() {
            return Err(SweepError::BaseCfgNotObject);
        }
        if self.axes.is_empty() {
            return Err(SweepError::EmptyAxes);
        }

        match &self.strategy {
            Strategy::Grid => self.plan_grid(base_cfg),
            Strategy::Random(cfg) => self.plan_random(base_cfg, cfg),
            Strategy::Asha(cfg) => self.plan_asha(base_cfg, cfg),
            Strategy::Bayes(cfg) => {
                // Stub — fall back to Random. The Console UI flags
                // this so users know they're getting random sampling
                // until the GP lands.
                let r = RandomConfig {
                    trials: cfg.trials,
                    seed: cfg.seed,
                };
                self.plan_random(base_cfg, &r)
            },
        }
    }

    fn plan_grid(&self, base_cfg: &serde_json::Value) -> SweepResult<Vec<Trial>> {
        let combos = cartesian(&self.axes);
        Ok(combos
            .into_iter()
            .map(|combo| make_trial(base_cfg, &self.axes, &combo, None))
            .collect())
    }

    fn plan_random(
        &self,
        base_cfg: &serde_json::Value,
        cfg: &RandomConfig,
    ) -> SweepResult<Vec<Trial>> {
        let max = self.cartesian_size();
        let take = cfg.trials.min(max);
        let mut rng = match cfg.seed {
            Some(s) => StdRng::seed_from_u64(s),
            // Cheap entropy source — UNIX time in nanos. Good enough
            // for "random ordering of sweep trials"; sensitive callers
            // should pass their own seed.
            None => {
                let t = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_nanos() as u64)
                    .unwrap_or(0);
                StdRng::seed_from_u64(t)
            },
        };
        let mut combos = cartesian(&self.axes);
        combos.shuffle(&mut rng);
        Ok(combos
            .into_iter()
            .take(take)
            .map(|combo| make_trial(base_cfg, &self.axes, &combo, None))
            .collect())
    }

    fn plan_asha(&self, base_cfg: &serde_json::Value, cfg: &AshaConfig) -> SweepResult<Vec<Trial>> {
        if cfg.eta < 2 {
            return Err(SweepError::AxisNotArray(format!(
                "asha.eta must be ≥ 2, got {}",
                cfg.eta
            )));
        }
        if cfg.min_resource == 0 || cfg.max_resource < cfg.min_resource {
            return Err(SweepError::AxisNotArray(format!(
                "asha resources invalid: min={} max={}",
                cfg.min_resource, cfg.max_resource
            )));
        }

        // Compute the rung budget schedule: r_i = min * eta^i, capped
        // at max. We emit every Cartesian combo at the first rung;
        // the orchestrator prunes to the top 1/eta and re-emits at
        // the next rung.
        let combos = cartesian(&self.axes);
        let mut budgets = Vec::new();
        let mut b = cfg.min_resource;
        while b <= cfg.max_resource {
            budgets.push(b);
            let next = b.saturating_mul(cfg.eta);
            if next == b {
                break;
            }
            b = next;
        }
        // First rung carries every combo; deeper rungs are emitted by
        // the orchestrator after pruning. We surface them all here so
        // the planner output is self-describing.
        let mut out = Vec::with_capacity(combos.len() * budgets.len() / cfg.eta as usize + 1);
        let mut survivors = combos.clone();
        for (rung_idx, budget) in budgets.iter().enumerate() {
            for combo in &survivors {
                out.push(make_trial(base_cfg, &self.axes, combo, Some(*budget)));
            }
            // Prune to the top 1/eta heuristically (planner doesn't
            // know real metrics — it just trims by index for the
            // schedule preview).
            let keep = survivors.len() / cfg.eta as usize;
            if keep == 0 || rung_idx + 1 == budgets.len() {
                break;
            }
            survivors.truncate(keep);
        }
        Ok(out)
    }
}

/// Fluent builder. Use `.add(name, values)` for each axis, then
/// `.build()`.
pub struct SweepBuilder {
    strategy: Strategy,
    axes: Vec<Axis>,
}

impl SweepBuilder {
    pub(crate) fn new(strategy: Strategy) -> Self {
        Self {
            strategy,
            axes: Vec::new(),
        }
    }

    /// Append an axis. `values` must be a JSON array.
    pub fn add(mut self, name: &str, values: &serde_json::Value) -> Result<Self, SweepError> {
        let arr = values
            .as_array()
            .ok_or_else(|| SweepError::AxisNotArray(name.to_string()))?
            .clone();
        self.axes.push(Axis {
            name: name.to_string(),
            values: arr,
        });
        Ok(self)
    }

    /// Finalize the spec.
    pub fn build(self) -> SweepSpec {
        SweepSpec {
            strategy: self.strategy,
            axes: self.axes,
        }
    }
}

// ---- helpers --------------------------------------------------------

fn cartesian(axes: &[Axis]) -> Vec<Vec<serde_json::Value>> {
    let mut out = vec![vec![]];
    for a in axes {
        let mut next = Vec::with_capacity(out.len() * a.values.len());
        for prefix in &out {
            for v in &a.values {
                let mut row = prefix.clone();
                row.push(v.clone());
                next.push(row);
            }
        }
        out = next;
    }
    out
}

fn make_trial(
    base: &serde_json::Value,
    axes: &[Axis],
    combo: &[serde_json::Value],
    rung_budget: Option<u32>,
) -> Trial {
    let mut cfg = base.clone();
    let mut axis_map = serde_json::Map::new();
    if let Some(obj) = cfg.as_object_mut() {
        for (a, v) in axes.iter().zip(combo) {
            obj.insert(a.name.clone(), v.clone());
            axis_map.insert(a.name.clone(), v.clone());
        }
    }
    Trial {
        id: ulid::Ulid::new().to_string(),
        cfg,
        axes: serde_json::Value::Object(axis_map),
        rung_budget,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn grid_cartesian_size() {
        let s = SweepSpec::grid()
            .add("lr", &json!([1e-4, 3e-4, 1e-3]))
            .unwrap()
            .add("batch", &json!([64, 128]))
            .unwrap()
            .build();
        assert_eq!(s.cartesian_size(), 6);
        let trials = s.plan(&json!({})).unwrap();
        assert_eq!(trials.len(), 6);
        // Every trial cfg has both axes set.
        for t in &trials {
            assert!(t.cfg.get("lr").is_some());
            assert!(t.cfg.get("batch").is_some());
        }
    }

    #[test]
    fn grid_merges_with_base_cfg() {
        let base = json!({"optim": "adamw", "lr": 0.0});
        let s = SweepSpec::grid()
            .add("lr", &json!([1e-4, 1e-3]))
            .unwrap()
            .build();
        let trials = s.plan(&base).unwrap();
        assert_eq!(trials.len(), 2);
        assert_eq!(trials[0].cfg["optim"], "adamw");
        // Axis values overwrite base.
        let lrs: Vec<f64> = trials
            .iter()
            .map(|t| t.cfg["lr"].as_f64().unwrap())
            .collect();
        assert!(lrs.contains(&1e-4));
        assert!(lrs.contains(&1e-3));
    }

    #[test]
    fn random_is_reproducible_with_seed() {
        let s1 = SweepSpec::random(3, Some(42))
            .add("lr", &json!([1e-4, 3e-4, 1e-3, 1e-2]))
            .unwrap()
            .add("batch", &json!([32, 64]))
            .unwrap()
            .build();
        let s2 = SweepSpec::random(3, Some(42))
            .add("lr", &json!([1e-4, 3e-4, 1e-3, 1e-2]))
            .unwrap()
            .add("batch", &json!([32, 64]))
            .unwrap()
            .build();
        let t1: Vec<_> = s1
            .plan(&json!({}))
            .unwrap()
            .into_iter()
            .map(|t| t.axes)
            .collect();
        let t2: Vec<_> = s2
            .plan(&json!({}))
            .unwrap()
            .into_iter()
            .map(|t| t.axes)
            .collect();
        assert_eq!(t1, t2, "same seed → same trials");
    }

    #[test]
    fn random_caps_at_cartesian_size() {
        let s = SweepSpec::random(100, Some(0))
            .add("lr", &json!([1e-4, 1e-3]))
            .unwrap()
            .build();
        let trials = s.plan(&json!({})).unwrap();
        assert_eq!(trials.len(), 2, "should not exceed full Cartesian product");
    }

    #[test]
    fn asha_emits_increasing_rung_budgets() {
        let s = SweepSpec::asha(AshaConfig {
            eta: 3,
            brackets: 1,
            min_resource: 1,
            max_resource: 27,
        })
        .add(
            "lr",
            &json!([1e-4, 3e-4, 1e-3, 1e-2, 1e-1, 1.0, 10., 100., 1000.]),
        )
        .unwrap()
        .build();
        let trials = s.plan(&json!({})).unwrap();
        // Distinct budgets seen.
        let mut budgets: Vec<u32> = trials.iter().filter_map(|t| t.rung_budget).collect();
        budgets.sort();
        budgets.dedup();
        // 9 combos at eta=3 produce 3 rungs (last has 1 survivor →
        // pruning to 0 stops the schedule before rung 4).
        assert_eq!(budgets, vec![1, 3, 9]);
    }

    #[test]
    fn empty_axes_errors() {
        let s = SweepSpec::grid().build();
        let err = s.plan(&json!({})).unwrap_err();
        assert!(matches!(err, SweepError::EmptyAxes));
    }

    #[test]
    fn non_array_axis_errors() {
        let err = SweepSpec::grid().add("lr", &json!("not an array"));
        assert!(matches!(err, Err(SweepError::AxisNotArray(_))));
    }

    #[test]
    fn non_object_base_cfg_errors() {
        let s = SweepSpec::grid().add("lr", &json!([1e-3])).unwrap().build();
        let err = s.plan(&json!("string")).unwrap_err();
        assert!(matches!(err, SweepError::BaseCfgNotObject));
    }

    #[test]
    fn bayes_falls_back_to_random_today() {
        let s = SweepSpec::bayes(BayesConfig {
            init_trials: 5,
            trials: 3,
            acquisition: "ei".into(),
            seed: Some(7),
        })
        .add("lr", &json!([1e-4, 3e-4, 1e-3]))
        .unwrap()
        .build();
        let trials = s.plan(&json!({})).unwrap();
        // Random fallback caps at trials × cartesian.
        assert!(trials.len() <= 3);
    }
}
