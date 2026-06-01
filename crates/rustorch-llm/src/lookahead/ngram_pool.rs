//! N-gram pool for Lookahead Decoding (T246.7 P1.2).
//!
//! Maintains a rolling cache of n-grams (currently `n = 3`, keyed by a
//! 2-token prefix → continuation token) extracted from the
//! generation history. The cache feeds the
//! [`crate::lookahead::JacobiWindow`] guess generator: when a recent
//! prefix matches the last two generated tokens, the stored
//! continuations become high-confidence draft branches that the GPU
//! can verify in a single forward pass.
//!
//! ## Storage
//!
//! `HashMap<(u32, u32), VecDeque<[u32; L_MAX]>>` keyed by the 2-token
//! prefix; the `VecDeque` stores up to `MAX_PER_PREFIX` continuations
//! in insertion order (newest at the back). A global LRU bound
//! (`capacity`, default 2000 distinct prefixes) prevents pathological
//! growth on long sessions.
//!
//! ## Hot-path budget
//!
//! `query` is O(1) hash lookup + a small `VecDeque` walk; `insert` is
//! O(1) amortized. Microbench gate (`lookahead_pool_microbench`)
//! enforces sub-µs query latency so that the per-step pool overhead
//! stays well under 5 µs (verify pass is ~50 ms).

use std::collections::{HashMap, VecDeque};

/// Maximum length of a stored continuation. We currently use n=3
/// trigrams (prefix length 2 + continuation length 1), so `L_MAX = 1`
/// suffices for the on-disk payload — but we keep a constant so
/// callers can extend to longer continuations without churning the
/// type signature.
pub const L_MAX: usize = 1;

/// Default per-prefix continuation buffer capacity. Older
/// continuations are evicted FIFO when this is exceeded — recency is
/// preferred because long-range generation patterns drift.
pub const MAX_PER_PREFIX: usize = 8;

/// Default cap on the number of distinct 2-token prefixes the pool
/// holds. Eviction is FIFO on the recency-ordered key list (see
/// [`NGramPool::recency`]). 2000 prefixes × ~80 bytes ≈ 160 KB,
/// negligible vs. the 11 GB Q4_K_M model.
pub const DEFAULT_CAPACITY: usize = 2000;

/// Per-pool statistics, useful for tuning W/L and validating that the
/// n-gram cache is actually warming up over generation.
#[derive(Debug, Clone, Copy, Default)]
pub struct PoolStats {
    /// Total number of [`NGramPool::query`] calls issued.
    pub total_queries: u64,
    /// Number of queries that returned at least one continuation.
    pub hit_count: u64,
    /// Number of distinct prefixes evicted by capacity pressure.
    pub evictions: u64,
}

impl PoolStats {
    /// Hit rate in `[0.0, 1.0]`. Returns `0.0` if no queries have run
    /// yet (never `NaN`).
    pub fn hit_rate(&self) -> f64 {
        if self.total_queries == 0 {
            0.0
        } else {
            self.hit_count as f64 / self.total_queries as f64
        }
    }
}

/// Rolling n-gram cache indexed by a 2-token prefix.
///
/// See module docs for the storage layout and complexity guarantees.
pub struct NGramPool {
    /// Hash table keyed by `(prev_prev, prev)` → continuation buffer.
    table: HashMap<(u32, u32), VecDeque<[u32; L_MAX]>>,
    /// LRU recency queue of prefixes (oldest at the front). When the
    /// table outgrows `capacity`, the front prefix is evicted.
    recency: VecDeque<(u32, u32)>,
    /// Maximum number of distinct prefixes to retain.
    capacity: usize,
    /// Mutable stats surfaced via [`Self::stats`].
    stats: PoolStats,
}

impl NGramPool {
    /// Create a pool with the default capacity ([`DEFAULT_CAPACITY`]).
    pub fn new() -> Self {
        Self::with_capacity(DEFAULT_CAPACITY)
    }

    /// Create a pool with an explicit capacity (number of distinct
    /// 2-token prefixes). Set to `0` to disable the cache entirely.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            table: HashMap::with_capacity(capacity.max(1)),
            recency: VecDeque::with_capacity(capacity.max(1)),
            capacity,
            stats: PoolStats::default(),
        }
    }

    /// Configured prefix capacity.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Number of distinct prefixes currently held.
    pub fn len(&self) -> usize {
        self.table.len()
    }

    /// `true` iff no prefixes are stored.
    pub fn is_empty(&self) -> bool {
        self.table.is_empty()
    }

    /// Read-only access to the running statistics.
    pub fn stats(&self) -> &PoolStats {
        &self.stats
    }

    /// Reset the running statistics counters. Pool contents are not
    /// cleared.
    pub fn reset_stats(&mut self) {
        self.stats = PoolStats::default();
    }

    /// Insert every trigram extractable from `tokens` into the pool.
    ///
    /// For input `[a, b, c, d]` we record the trigrams `(a, b) → c`
    /// and `(b, c) → d`. Sequences shorter than 3 tokens are no-ops.
    /// Duplicate insertions are kept (the `VecDeque` reflects
    /// recency); newest continuations sit at the back.
    pub fn insert(&mut self, tokens: &[u32]) {
        if self.capacity == 0 || tokens.len() < 3 {
            return;
        }
        for window in tokens.windows(3) {
            let key = (window[0], window[1]);
            let cont: [u32; L_MAX] = [window[2]];
            self.insert_one(key, cont);
        }
    }

    /// Insert a single trigram by `(prefix_a, prefix_b)` →
    /// continuation. Useful for callers that want fine-grained
    /// control (e.g. injecting curated patterns).
    pub fn insert_trigram(&mut self, prefix: (u32, u32), continuation: u32) {
        if self.capacity == 0 {
            return;
        }
        self.insert_one(prefix, [continuation]);
    }

    fn insert_one(&mut self, key: (u32, u32), cont: [u32; L_MAX]) {
        let entry = self.table.entry(key);
        let is_new = matches!(entry, std::collections::hash_map::Entry::Vacant(_));
        let buf = entry.or_insert_with(|| VecDeque::with_capacity(MAX_PER_PREFIX));
        if buf.len() == MAX_PER_PREFIX {
            buf.pop_front();
        }
        buf.push_back(cont);

        if is_new {
            self.recency.push_back(key);
            self.evict_if_needed();
        } else {
            // Bump recency: remove old position then re-push at the
            // back. The recency queue is small (==len()) so the
            // linear scan is fine for the capacities we target.
            if let Some(pos) = self.recency.iter().position(|k| *k == key) {
                self.recency.remove(pos);
            }
            self.recency.push_back(key);
        }
    }

    fn evict_if_needed(&mut self) {
        while self.table.len() > self.capacity {
            if let Some(victim) = self.recency.pop_front() {
                self.table.remove(&victim);
                self.stats.evictions += 1;
            } else {
                break;
            }
        }
    }

    /// Query the pool for continuations matching `prefix` (last two
    /// tokens). Returns the stored continuations in newest-first
    /// order so callers can prioritise recency. The returned slice is
    /// borrowed from the pool — copy out before the next `insert`
    /// call.
    ///
    /// `prefix` must contain at least 2 tokens; shorter prefixes
    /// always return an empty `Vec`.
    pub fn query(&mut self, prefix: &[u32]) -> Vec<[u32; L_MAX]> {
        self.stats.total_queries += 1;
        if prefix.len() < 2 {
            return Vec::new();
        }
        let n = prefix.len();
        let key = (prefix[n - 2], prefix[n - 1]);
        match self.table.get(&key) {
            None => Vec::new(),
            Some(buf) => {
                self.stats.hit_count += 1;
                // Return newest first.
                buf.iter().rev().copied().collect()
            },
        }
    }

    /// Non-mutating query variant — does NOT update statistics.
    /// Useful for inspection / debugging; production callers should
    /// prefer [`Self::query`] so hit-rate metrics stay accurate.
    pub fn peek(&self, prefix: &[u32]) -> Vec<[u32; L_MAX]> {
        if prefix.len() < 2 {
            return Vec::new();
        }
        let n = prefix.len();
        let key = (prefix[n - 2], prefix[n - 1]);
        match self.table.get(&key) {
            None => Vec::new(),
            Some(buf) => buf.iter().rev().copied().collect(),
        }
    }
}

impl Default for NGramPool {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for NGramPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NGramPool")
            .field("len", &self.table.len())
            .field("capacity", &self.capacity)
            .field("stats", &self.stats)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_then_query_exact_prefix() {
        let mut pool = NGramPool::new();
        // Trigrams: (1,2)->3, (2,3)->4, (3,4)->5
        pool.insert(&[1, 2, 3, 4, 5]);

        let conts = pool.query(&[1, 2]);
        assert_eq!(conts, vec![[3]]);

        let conts = pool.query(&[2, 3]);
        assert_eq!(conts, vec![[4]]);

        let conts = pool.query(&[3, 4]);
        assert_eq!(conts, vec![[5]]);
    }

    #[test]
    fn query_miss_returns_empty() {
        let mut pool = NGramPool::new();
        pool.insert(&[1, 2, 3]);
        let conts = pool.query(&[10, 20]);
        assert!(conts.is_empty());
        assert_eq!(pool.stats().total_queries, 1);
        assert_eq!(pool.stats().hit_count, 0);
    }

    #[test]
    fn query_uses_last_two_tokens_of_prefix() {
        let mut pool = NGramPool::new();
        pool.insert(&[7, 8, 9]);
        // A 3-token prefix should still match using the last 2.
        let conts = pool.query(&[1, 7, 8]);
        assert_eq!(conts, vec![[9]]);
    }

    #[test]
    fn short_input_is_noop() {
        let mut pool = NGramPool::new();
        pool.insert(&[1, 2]); // len 2 — no trigrams.
        assert_eq!(pool.len(), 0);
        let conts = pool.query(&[1]); // prefix < 2 — empty.
        assert!(conts.is_empty());
    }

    #[test]
    fn duplicate_insertions_accumulate_with_recency() {
        let mut pool = NGramPool::new();
        // (1,2)->3 then (1,2)->4 — both stored, newest first on query.
        pool.insert(&[1, 2, 3]);
        pool.insert(&[1, 2, 4]);
        let conts = pool.query(&[1, 2]);
        assert_eq!(conts, vec![[4], [3]]);
    }

    #[test]
    fn per_prefix_buffer_caps_at_max() {
        let mut pool = NGramPool::new();
        for k in 0u32..(MAX_PER_PREFIX as u32 + 4) {
            pool.insert_trigram((1, 2), k);
        }
        let conts = pool.query(&[1, 2]);
        assert_eq!(conts.len(), MAX_PER_PREFIX);
        // Newest first: last inserted was MAX_PER_PREFIX+3.
        assert_eq!(conts[0], [(MAX_PER_PREFIX as u32 + 3)]);
    }

    #[test]
    fn capacity_overflow_evicts_lru_prefix() {
        let mut pool = NGramPool::with_capacity(2);
        pool.insert_trigram((1, 1), 100); // prefix (1,1)
        pool.insert_trigram((2, 2), 200); // prefix (2,2)
        pool.insert_trigram((3, 3), 300); // prefix (3,3) — evicts (1,1)

        assert_eq!(pool.len(), 2);
        assert!(pool.peek(&[1, 1]).is_empty()); // evicted
        assert_eq!(pool.peek(&[2, 2]), vec![[200]]);
        assert_eq!(pool.peek(&[3, 3]), vec![[300]]);
        assert_eq!(pool.stats().evictions, 1);
    }

    #[test]
    fn capacity_zero_disables_pool() {
        let mut pool = NGramPool::with_capacity(0);
        pool.insert(&[1, 2, 3, 4]);
        assert_eq!(pool.len(), 0);
        let conts = pool.query(&[1, 2]);
        assert!(conts.is_empty());
    }

    #[test]
    fn touched_prefix_gets_recency_bump() {
        let mut pool = NGramPool::with_capacity(2);
        pool.insert_trigram((1, 1), 100);
        pool.insert_trigram((2, 2), 200);
        // Re-insert (1,1) — it should now be the youngest and (2,2)
        // becomes the eviction victim.
        pool.insert_trigram((1, 1), 101);
        pool.insert_trigram((3, 3), 300);

        assert_eq!(pool.len(), 2);
        assert!(pool.peek(&[2, 2]).is_empty()); // (2,2) was evicted
        assert!(!pool.peek(&[1, 1]).is_empty());
        assert!(!pool.peek(&[3, 3]).is_empty());
    }

    #[test]
    fn stats_track_queries_and_hits() {
        let mut pool = NGramPool::new();
        pool.insert(&[1, 2, 3]);
        let _ = pool.query(&[1, 2]); // hit
        let _ = pool.query(&[9, 9]); // miss
        let _ = pool.query(&[1, 2]); // hit
        let s = pool.stats();
        assert_eq!(s.total_queries, 3);
        assert_eq!(s.hit_count, 2);
        assert!((s.hit_rate() - 2.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn reset_stats_clears_counters_only() {
        let mut pool = NGramPool::new();
        pool.insert(&[1, 2, 3]);
        let _ = pool.query(&[1, 2]);
        pool.reset_stats();
        assert_eq!(pool.stats().total_queries, 0);
        assert_eq!(pool.stats().hit_count, 0);
        // Pool contents preserved.
        assert_eq!(pool.peek(&[1, 2]), vec![[3]]);
    }
}
