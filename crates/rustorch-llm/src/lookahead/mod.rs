//! Lookahead Decoding (CPU side) — T246.7 P1.2.
//!
//! Implements the host-side scaffolding for Lookahead Decoding
//! (Fu et al. 2024, arxiv 2402.02057). The two building blocks live
//! in submodules:
//!
//! - [`ngram_pool::NGramPool`] — rolling 2→1 trigram cache fed by
//!   the verified continuations.
//! - [`jacobi::JacobiWindow`] — `W × L` Jacobi-iteration guess buffer
//!   that emits the per-step [`jacobi::DraftTree`].
//!
//! [`LookaheadManager`] composes them and exposes a single entry
//! point for the GPU decode loop:
//!
//! ```ignore
//! use rustorch_llm::lookahead::LookaheadManager;
//!
//! let mut mgr = LookaheadManager::new(5, 5);
//! // pretend the model already validated `cur` and the previous
//! // token was `prev`:
//! let tree = mgr.build_drafts(cur);
//! // ... GPU verify pass returns the accepted suffix `ks`,
//! //     plus the full set of generated tokens `gen` (drafts that
//! //     were validated):
//! mgr.record_acceptance(&drafts_in_order, ks.len(), &gen);
//! ```
//!
//! ## Opt-in via environment
//!
//! [`LookaheadManager::from_env`] reads:
//!
//! - `RUSTORCH_LOOKAHEAD=1` — master enable. Returns `None` otherwise.
//! - `RUSTORCH_LOOKAHEAD_W` — window width (default `5`).
//! - `RUSTORCH_LOOKAHEAD_L` — window depth (default `5`).
//!
//! ## Hot-path safety
//!
//! No panics. Invalid env values fall back to defaults with a debug
//! warning (gated on `RUSTORCH_LOOKAHEAD_DEBUG`). All operations are
//! `O(W * L)` in the worst case; `build_drafts` runs in well under
//! 5 µs on a single CPU core (see
//! `examples/lookahead_pool_microbench.rs`).

pub mod jacobi;
pub mod ngram_pool;

pub use jacobi::{DraftTree, JacobiWindow};
pub use ngram_pool::{NGramPool, PoolStats, DEFAULT_CAPACITY, L_MAX, MAX_PER_PREFIX};

/// Aggregate stats for an entire lookahead session.
#[derive(Debug, Clone, Copy, Default)]
pub struct LookaheadStats {
    /// Number of `build_drafts` calls.
    pub total_calls: u64,
    /// Total number of draft tokens emitted (root excluded).
    pub total_drafts: u64,
    /// Total number of draft tokens that the GPU verify pass
    /// accepted (root excluded). Always `<= total_drafts`.
    pub total_accepted: u64,
    /// Number of distinct prefixes evicted from the n-gram pool.
    pub pool_evictions: u64,
    /// Total queries to the pool.
    pub pool_total_queries: u64,
    /// Total pool hits.
    pub pool_hit_count: u64,
}

impl LookaheadStats {
    /// Acceptance rate = `total_accepted / total_drafts`. Returns
    /// `0.0` when no drafts have been emitted yet.
    pub fn acceptance_rate(&self) -> f64 {
        if self.total_drafts == 0 {
            0.0
        } else {
            self.total_accepted as f64 / self.total_drafts as f64
        }
    }

    /// Pool hit rate = `pool_hit_count / pool_total_queries`. Returns
    /// `0.0` when no queries have been issued yet.
    pub fn pool_hit_rate(&self) -> f64 {
        if self.pool_total_queries == 0 {
            0.0
        } else {
            self.pool_hit_count as f64 / self.pool_total_queries as f64
        }
    }
}

/// Top-level Lookahead Decoding manager. Owns the n-gram pool and
/// the Jacobi window and tracks aggregate statistics.
pub struct LookaheadManager {
    pool: NGramPool,
    window: JacobiWindow,
    stats: LookaheadStats,
    /// The most-recently seen "previous" token. Tracked here so
    /// [`Self::build_drafts`] can supply a 2-token prefix to the
    /// pool without burdening the caller.
    prev_token: Option<u32>,
}

impl LookaheadManager {
    /// Construct a manager with the default n-gram pool capacity and
    /// the supplied Jacobi window dimensions.
    pub fn new(w: usize, l: usize) -> Self {
        Self {
            pool: NGramPool::new(),
            window: JacobiWindow::new(w, l),
            stats: LookaheadStats::default(),
            prev_token: None,
        }
    }

    /// Construct from a custom pool and window — useful for tests
    /// and for callers that want non-default pool capacity.
    pub fn with_components(pool: NGramPool, window: JacobiWindow) -> Self {
        Self {
            pool,
            window,
            stats: LookaheadStats::default(),
            prev_token: None,
        }
    }

    /// Construct from environment variables. Returns `None` when
    /// `RUSTORCH_LOOKAHEAD` is unset, empty, or `0`.
    ///
    /// `RUSTORCH_LOOKAHEAD_W` (default `5`) and
    /// `RUSTORCH_LOOKAHEAD_L` (default `5`) override the window
    /// dimensions; non-numeric or zero values silently fall back to
    /// defaults (a warning is printed when `RUSTORCH_LOOKAHEAD_DEBUG`
    /// is set).
    pub fn from_env() -> Option<Self> {
        let enabled = std::env::var("RUSTORCH_LOOKAHEAD")
            .ok()
            .map(|v| v != "0" && !v.is_empty())
            .unwrap_or(false);
        if !enabled {
            return None;
        }
        let w = parse_dim_env("RUSTORCH_LOOKAHEAD_W", 5);
        let l = parse_dim_env("RUSTORCH_LOOKAHEAD_L", 5);
        Some(Self::new(w, l))
    }

    /// Window width (W).
    pub fn w(&self) -> usize {
        self.window.w()
    }

    /// Window depth (L).
    pub fn l(&self) -> usize {
        self.window.l()
    }

    /// Borrow the underlying n-gram pool (read-only).
    pub fn pool(&self) -> &NGramPool {
        &self.pool
    }

    /// Borrow the Jacobi window state (read-only).
    pub fn window(&self) -> &JacobiWindow {
        &self.window
    }

    /// Build the next [`DraftTree`] for the GPU verify pass.
    pub fn build_drafts(&mut self, last_token: u32) -> DraftTree {
        let tree = self
            .window
            .build_drafts(last_token, self.prev_token, &mut self.pool);
        self.stats.total_calls += 1;
        // Drafts are everything but the root.
        self.stats.total_drafts += (tree.len().saturating_sub(1)) as u64;
        // Surface pool stats into the aggregate snapshot.
        let ps = self.pool.stats();
        self.stats.pool_total_queries = ps.total_queries;
        self.stats.pool_hit_count = ps.hit_count;
        self.stats.pool_evictions = ps.evictions;
        tree
    }

    /// Record the outcome of a verify pass.
    ///
    /// - `drafts` — the draft tokens that were submitted (root
    ///   excluded). Stored only via the `generated` updates below;
    ///   passed in for symmetry with future training-flavoured
    ///   schedulers.
    /// - `accepted_count` — number of draft tokens the GPU
    ///   accepted; bumps the running acceptance counter.
    /// - `generated` — the full continuation that the verify pass
    ///   produced (root + accepted prefix + speculatively-sampled
    ///   tail token). Used to (a) seed the next Jacobi window via
    ///   [`JacobiWindow::update`] and (b) feed every freshly-observed
    ///   trigram into the n-gram pool.
    pub fn record_acceptance(&mut self, _drafts: &[u32], accepted_count: usize, generated: &[u32]) {
        self.stats.total_accepted += accepted_count as u64;
        // Insert every trigram observed in the actual generated
        // continuation. This is the mechanism that makes the n-gram
        // hit rate grow over generation.
        self.pool.insert(generated);

        // Refresh prev_token so the next build_drafts can query the
        // pool with a 2-token prefix.
        if generated.len() >= 2 {
            self.prev_token = Some(generated[generated.len() - 2]);
        } else if let Some(&t) = generated.last() {
            self.prev_token = Some(t);
        }

        // Advance the Jacobi window. Per RFC the validated continuation
        // is everything past the root in `generated`.
        let accepted_payload: &[u32] = if generated.is_empty() {
            &[]
        } else {
            &generated[1.min(generated.len())..]
        };
        self.window.update(accepted_payload, &[]);
    }

    /// Snapshot of the running statistics. Pool counters are
    /// refreshed on every [`Self::build_drafts`] call.
    pub fn stats(&self) -> &LookaheadStats {
        &self.stats
    }

    /// Reset all statistics counters (pool contents and window state
    /// are preserved).
    pub fn reset_stats(&mut self) {
        self.stats = LookaheadStats::default();
        self.pool.reset_stats();
    }
}

fn parse_dim_env(var: &str, default: usize) -> usize {
    match std::env::var(var) {
        Ok(s) => match s.parse::<usize>() {
            Ok(v) if v >= 1 => v,
            _ => {
                if std::env::var("RUSTORCH_LOOKAHEAD_DEBUG").is_ok() {
                    eprintln!("[lookahead] {var}={s:?} invalid, falling back to default {default}");
                }
                default
            },
        },
        Err(_) => default,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manager_build_then_record_updates_pool_and_window() {
        let mut mgr = LookaheadManager::new(3, 3);
        // Step 1: cold start.
        let tree = mgr.build_drafts(10);
        assert_eq!(tree.len(), 1); // singleton root
        assert_eq!(mgr.stats().total_calls, 1);
        assert_eq!(mgr.stats().total_drafts, 0);

        // Pretend GPU returned 3 tokens of continuation.
        let drafts: [u32; 0] = [];
        mgr.record_acceptance(&drafts, 0, &[10, 20, 30, 40]);

        // Step 2: window seeded; pool now contains trigrams.
        // Pool has (10,20)->30, (20,30)->40.
        let tree = mgr.build_drafts(40);
        // Window seeded ⇒ depth-1 children from columns + pool query
        // for (prev=30, cur=40). No pool hit on (30,40), so children
        // come solely from Jacobi columns.
        tree.validate().unwrap();
        assert!(tree.len() >= 2);
        assert!(mgr.pool().len() >= 2);
    }

    #[test]
    fn manager_records_acceptance_count_in_stats() {
        let mut mgr = LookaheadManager::new(2, 3);
        mgr.record_acceptance(&[1, 2], 2, &[5, 1, 2]);
        assert_eq!(mgr.stats().total_accepted, 2);
        // 2 trigrams from [5,1,2]? len=3 → 1 trigram.
        assert_eq!(mgr.pool().len(), 1);
    }

    #[test]
    fn manager_pool_hit_rate_grows_with_repeats() {
        let mut mgr = LookaheadManager::new(4, 4);
        // Warm pool with a repeating pattern.
        mgr.record_acceptance(&[], 0, &[1, 2, 3, 1, 2, 3, 1, 2, 3]);
        // Now the prefix (1,2) has continuation 3 cached. Build a
        // tree with last_token=2 and the manager will set prev_token
        // from the recorded generated, so let's simulate that
        // explicitly:
        mgr.prev_token = Some(1);
        let tree = mgr.build_drafts(2);
        tree.validate().unwrap();
        // Pool should have hit at least once.
        assert!(mgr.stats().pool_hit_count >= 1);
        assert!(mgr.stats().pool_hit_rate() > 0.0);
    }

    #[test]
    fn from_env_returns_none_when_disabled() {
        // Tests touch global env — we restore on exit. Cargo runs
        // tests in parallel by default, so these env var names are
        // unique to this test.
        let saved = std::env::var("RUSTORCH_LOOKAHEAD").ok();
        std::env::remove_var("RUSTORCH_LOOKAHEAD");
        assert!(LookaheadManager::from_env().is_none());
        std::env::set_var("RUSTORCH_LOOKAHEAD", "0");
        assert!(LookaheadManager::from_env().is_none());
        match saved {
            Some(v) => std::env::set_var("RUSTORCH_LOOKAHEAD", v),
            None => std::env::remove_var("RUSTORCH_LOOKAHEAD"),
        }
    }

    #[test]
    fn parse_dim_env_falls_back_on_garbage() {
        std::env::set_var("FAKE_DIM_VAR_T2467", "not-a-number");
        assert_eq!(parse_dim_env("FAKE_DIM_VAR_T2467", 5), 5);
        std::env::set_var("FAKE_DIM_VAR_T2467", "0");
        assert_eq!(parse_dim_env("FAKE_DIM_VAR_T2467", 5), 5);
        std::env::set_var("FAKE_DIM_VAR_T2467", "7");
        assert_eq!(parse_dim_env("FAKE_DIM_VAR_T2467", 5), 7);
        std::env::remove_var("FAKE_DIM_VAR_T2467");
    }

    #[test]
    fn acceptance_rate_division_safe() {
        let s = LookaheadStats::default();
        assert_eq!(s.acceptance_rate(), 0.0);
        assert_eq!(s.pool_hit_rate(), 0.0);
        let s2 = LookaheadStats {
            total_drafts: 10,
            total_accepted: 3,
            pool_total_queries: 4,
            pool_hit_count: 1,
            ..Default::default()
        };
        assert!((s2.acceptance_rate() - 0.3).abs() < 1e-9);
        assert!((s2.pool_hit_rate() - 0.25).abs() < 1e-9);
    }
}
