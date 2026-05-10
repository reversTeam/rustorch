//! Jacobi window guess generator for Lookahead Decoding (T246.7 P1.2).
//!
//! Implements the W-by-L Jacobi iteration window described in
//! Fu et al. (2024, arxiv 2402.02057, §3.2). The window holds `W`
//! parallel guess sequences, each of length `L`, that the next decode
//! step verifies in a single forward pass. Whenever the cached
//! [`crate::lookahead::NGramPool`] produces a hit on the most recent
//! 2-token prefix, additional branches are grafted onto the draft
//! tree so that one verify pass can exit with up to several accepted
//! tokens.
//!
//! # Tree encoding
//!
//! The output [`DraftTree`] is a parent-pointer array stored in BFS
//! order:
//!
//! - `tokens[0]` is the root (the *current* token already validated
//!   on the GPU side).
//! - `parents[0] = -1`. For every other `i`, `parents[i] < i` because
//!   nodes are emitted level-by-level.
//! - `depths[0] = 0`. Each child's depth is `parents`'s depth + 1.
//!
//! BFS ordering is required by the downstream tree-attention kernel:
//! when it walks tokens 0..N writing K/V cache slots, every node's
//! parent has already been written.
//!
//! Worst-case node count: `1 + W * (L - 1)` (root + W chains of
//! length `L - 1`). For the canonical `(W=5, L=5)` configuration this
//! is 21 nodes — well under the 32 node ceiling embedded in the
//! verify kernels.

use super::ngram_pool::NGramPool;

/// Parent-pointer draft tree consumed by the GPU verify pass.
#[derive(Debug, Clone, Default)]
pub struct DraftTree {
    /// Token IDs in BFS order. Index 0 is the root (the most recently
    /// validated token).
    pub tokens: Vec<u32>,
    /// `parents[i]` is the index of node `i`'s parent, or `-1` for
    /// the root. Always `< i` for non-root nodes (BFS invariant).
    pub parents: Vec<i32>,
    /// Depth of each node from the root (root depth = 0).
    pub depths: Vec<u8>,
}

impl DraftTree {
    /// Number of nodes in the tree (root included).
    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    /// `true` iff the tree is empty (should never happen — the root
    /// is always present after [`JacobiWindow::build_drafts`]).
    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    /// Maximum depth observed (root depth = 0).
    pub fn max_depth(&self) -> u8 {
        self.depths.iter().copied().max().unwrap_or(0)
    }

    /// Returns the path of token IDs from the root down to `node_idx`,
    /// in root-to-leaf order. Useful for extracting the accepted
    /// continuation after the verify pass.
    pub fn path_to(&self, node_idx: usize) -> Vec<u32> {
        if node_idx >= self.tokens.len() {
            return Vec::new();
        }
        let mut path = Vec::with_capacity(self.depths[node_idx] as usize + 1);
        let mut cur = node_idx as i32;
        while cur >= 0 {
            path.push(self.tokens[cur as usize]);
            cur = self.parents[cur as usize];
        }
        path.reverse();
        path
    }

    /// Validate the BFS / parent-pointer invariants. Returns
    /// `Err(reason)` on the first violation. Used in tests and as a
    /// debug-build sanity check.
    pub fn validate(&self) -> Result<(), String> {
        let n = self.tokens.len();
        if n != self.parents.len() || n != self.depths.len() {
            return Err(format!(
                "length mismatch: tokens={} parents={} depths={}",
                n,
                self.parents.len(),
                self.depths.len()
            ));
        }
        if n == 0 {
            return Err("tree is empty".to_string());
        }
        if self.parents[0] != -1 {
            return Err(format!("root parent must be -1 (got {})", self.parents[0]));
        }
        if self.depths[0] != 0 {
            return Err(format!("root depth must be 0 (got {})", self.depths[0]));
        }
        for i in 1..n {
            let p = self.parents[i];
            if p < 0 || (p as usize) >= i {
                return Err(format!(
                    "node {i}: parent {p} must be in [0, {i}) (BFS order)"
                ));
            }
            let expected = self.depths[p as usize] + 1;
            if self.depths[i] != expected {
                return Err(format!(
                    "node {i}: depth {} != parent_depth+1 ({})",
                    self.depths[i], expected
                ));
            }
        }
        Ok(())
    }
}

/// Jacobi iteration window state.
///
/// Maintains `W` parallel guess sequences of length `L`. On each
/// decode step `build_drafts` assembles a [`DraftTree`] mixing the
/// current Jacobi guesses with n-gram pool hits, and `update`
/// advances the window after the GPU verify pass returns the accepted
/// continuation.
pub struct JacobiWindow {
    /// Number of parallel guess sequences (window width).
    w: usize,
    /// Length of each guess sequence (window depth).
    l: usize,
    /// `W` rolling guess sequences. `guesses[j].len() == L` once the
    /// window has been seeded; before that it may be shorter.
    guesses: Vec<Vec<u32>>,
    /// `true` until [`Self::update`] has consumed the first verified
    /// continuation. Used to know whether the window is meaningful or
    /// still in cold-start mode.
    seeded: bool,
}

impl JacobiWindow {
    /// Construct a window with the given dimensions.
    ///
    /// Both `w` and `l` must be `>= 1`. Typical values per the RFC are
    /// `(W=5, L=5)`, giving a 21-node draft tree.
    pub fn new(w: usize, l: usize) -> Self {
        assert!(w >= 1 && l >= 1, "JacobiWindow requires w >= 1 and l >= 1");
        Self {
            w,
            l,
            guesses: vec![Vec::with_capacity(l); w],
            seeded: false,
        }
    }

    /// Window width `W`.
    pub fn w(&self) -> usize {
        self.w
    }

    /// Window depth `L`.
    pub fn l(&self) -> usize {
        self.l
    }

    /// Whether the window has been seeded by at least one
    /// [`Self::update`] call.
    pub fn is_seeded(&self) -> bool {
        self.seeded
    }

    /// Maximum number of nodes [`Self::build_drafts`] can emit:
    /// `1 + W * (L - 1)` (root + one chain of length `L-1` per
    /// window column, possibly augmented by n-gram branches up to
    /// the same total).
    pub fn max_nodes(&self) -> usize {
        1 + self.w * self.l.saturating_sub(1)
    }

    /// Reset the window to cold-start state. Subsequent
    /// [`Self::build_drafts`] calls will produce a degenerate (W=1)
    /// tree until [`Self::update`] reseeds the guesses.
    pub fn reset(&mut self) {
        for g in &mut self.guesses {
            g.clear();
        }
        self.seeded = false;
    }

    /// Build a [`DraftTree`] for the next verify pass.
    ///
    /// The root token is `last_token` (the most recently validated
    /// token from the previous decode step). Children are sourced
    /// from:
    ///
    /// 1. The Jacobi window — each of the `W` guess columns
    ///    contributes a chain of length `L-1` starting at the root.
    /// 2. The n-gram pool — when the prefix `(prev, last_token)`
    ///    matches a stored continuation, it is grafted as an extra
    ///    child of the root (truncated to fit `max_nodes`).
    ///
    /// On the very first call (window not seeded yet, pool likely
    /// empty) the tree degenerates to a single-node root, signalling
    /// the caller to fall back to the standard `decode_step`.
    pub fn build_drafts(
        &mut self,
        last_token: u32,
        prev_token: Option<u32>,
        pool: &mut NGramPool,
    ) -> DraftTree {
        let cap = self.max_nodes();
        let mut tree = DraftTree {
            tokens: Vec::with_capacity(cap),
            parents: Vec::with_capacity(cap),
            depths: Vec::with_capacity(cap),
        };
        // Root node — index 0.
        tree.tokens.push(last_token);
        tree.parents.push(-1);
        tree.depths.push(0);

        // 1. n-gram pool branches grafted at the root. We query with
        //    the 2-token prefix (prev_token, last_token) when
        //    available; otherwise the pool can't help.
        let pool_branches: Vec<u32> = if let Some(prev) = prev_token {
            let conts = pool.query(&[prev, last_token]);
            // Each cont is [u32; L_MAX]; flatten + dedupe to avoid
            // wasting tree slots on duplicate first-tokens.
            let mut firsts: Vec<u32> = Vec::with_capacity(conts.len());
            for c in conts {
                let t = c[0];
                if !firsts.contains(&t) {
                    firsts.push(t);
                }
            }
            firsts
        } else {
            Vec::new()
        };

        // 2. Jacobi guess columns: at most W chains of depth L-1.
        //    Each column j contributes one node per remaining depth.
        //    Chains are appended level-by-level so the BFS invariant
        //    holds.
        if !self.seeded {
            // Cold start: only the root is meaningful. Pool branches
            // (if any) still get grafted — they represent priors from
            // an earlier session if pool was warmed externally.
            self.append_root_children(&mut tree, &pool_branches);
            return tree;
        }

        // Hot path: build BFS tree of width up to W (Jacobi columns
        // first, then any extra n-gram first-tokens that weren't
        // already a Jacobi column head).
        let mut child_first_tokens: Vec<u32> = Vec::with_capacity(self.w);
        for col in self.guesses.iter().take(self.w) {
            if let Some(&t) = col.first() {
                child_first_tokens.push(t);
            }
        }
        for &t in pool_branches.iter() {
            if child_first_tokens.contains(&t) {
                continue;
            }
            if child_first_tokens.len() >= self.w {
                break;
            }
            child_first_tokens.push(t);
        }

        // Emit depth-1 children.
        let depth1_start = tree.tokens.len();
        for &t in &child_first_tokens {
            if tree.tokens.len() >= cap {
                break;
            }
            tree.tokens.push(t);
            tree.parents.push(0);
            tree.depths.push(1);
        }

        // Emit deeper levels along the Jacobi columns. Pool branches
        // contribute only the depth-1 token (we don't speculatively
        // chain past the cached trigram — that would risk validating
        // arbitrary noise).
        let mut prev_level_indices: Vec<usize> = (depth1_start..tree.tokens.len()).collect();

        for d in 2..self.l {
            let mut cur_level_indices: Vec<usize> = Vec::new();
            for (col_idx, parent_idx) in prev_level_indices.iter().copied().enumerate().take(self.w)
            {
                if tree.tokens.len() >= cap {
                    break;
                }
                let col = &self.guesses[col_idx];
                if let Some(&t) = col.get(d - 1) {
                    let new_idx = tree.tokens.len();
                    tree.tokens.push(t);
                    tree.parents.push(parent_idx as i32);
                    tree.depths.push(d as u8);
                    cur_level_indices.push(new_idx);
                }
            }
            if cur_level_indices.is_empty() {
                break;
            }
            prev_level_indices = cur_level_indices;
        }

        tree
    }

    fn append_root_children(&self, tree: &mut DraftTree, children: &[u32]) {
        let cap = self.max_nodes();
        for &t in children {
            if tree.tokens.len() >= cap {
                break;
            }
            tree.tokens.push(t);
            tree.parents.push(0);
            tree.depths.push(1);
        }
    }

    /// Update the window after a verify pass.
    ///
    /// `accepted` is the path of token IDs the GPU accepted, **not
    /// including** the root that was already validated upstream
    /// (i.e. the verify pass returned `accepted.len()` new tokens).
    /// `_validated_logits` is reserved for future Jacobi-style
    /// resampling — Phase 1 ignores it and refills the tail of each
    /// column from the accepted continuation.
    ///
    /// Concretely, after `update`:
    /// - Each of the `W` columns holds a sequence of length `L`.
    /// - The first column is the accepted continuation (padded with
    ///   the last accepted token if `accepted.len() < L`).
    /// - The remaining `W-1` columns are seeded with shifts of the
    ///   accepted continuation and `0` padding for the tail.
    pub fn update(&mut self, accepted: &[u32], _validated_logits: &[u32]) {
        if accepted.is_empty() {
            // No tokens accepted — keep state, mark seeded so
            // subsequent build_drafts uses the existing window.
            self.seeded = self.seeded || !self.guesses[0].is_empty();
            return;
        }
        let pad = *accepted.last().expect("non-empty");

        // Column 0: the accepted prefix, right-padded with the last
        // accepted token to length L.
        let col0 = build_col(accepted, pad, self.l);
        self.guesses[0] = col0;

        // Columns 1..W: rolling shifts of the accepted prefix.
        // Column j starts at offset j (clamped) into the accepted
        // tokens; this matches the standard Jacobi rolling-window
        // initialisation. Tails are pad-filled.
        for (j, col) in self.guesses.iter_mut().enumerate().take(self.w).skip(1) {
            let start = j.min(accepted.len().saturating_sub(1));
            let shifted: Vec<u32> = accepted[start..].to_vec();
            *col = build_col(&shifted, pad, self.l);
        }
        self.seeded = true;
    }
}

fn build_col(seed: &[u32], pad: u32, l: usize) -> Vec<u32> {
    let mut col = Vec::with_capacity(l);
    for i in 0..l {
        col.push(seed.get(i).copied().unwrap_or(pad));
    }
    col
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[should_panic(expected = "JacobiWindow requires w >= 1")]
    fn new_rejects_zero_dimensions() {
        let _ = JacobiWindow::new(0, 5);
    }

    #[test]
    fn build_with_empty_pool_unseeded_is_singleton() {
        let mut w = JacobiWindow::new(5, 5);
        let mut pool = NGramPool::new();
        let tree = w.build_drafts(42, None, &mut pool);
        assert_eq!(tree.len(), 1);
        assert_eq!(tree.tokens[0], 42);
        assert_eq!(tree.parents[0], -1);
        assert_eq!(tree.depths[0], 0);
        tree.validate().unwrap();
    }

    #[test]
    fn build_with_pool_hit_unseeded_grafts_branches() {
        let mut w = JacobiWindow::new(5, 5);
        let mut pool = NGramPool::new();
        // (10, 20) -> 30 and (10, 20) -> 31
        pool.insert_trigram((10, 20), 30);
        pool.insert_trigram((10, 20), 31);

        let tree = w.build_drafts(20, Some(10), &mut pool);
        // Root + 2 grafts (depth 1) — pool returned [31, 30] newest first.
        assert_eq!(tree.len(), 3);
        assert_eq!(tree.tokens[0], 20);
        assert_eq!(tree.depths, vec![0, 1, 1]);
        assert_eq!(tree.parents, vec![-1, 0, 0]);
        // Newest first.
        assert_eq!(&tree.tokens[1..], &[31u32, 30u32]);
        tree.validate().unwrap();
    }

    #[test]
    fn build_seeded_emits_full_jacobi_chains_bfs() {
        let mut w = JacobiWindow::new(2, 3); // small for assertion clarity
        let mut pool = NGramPool::new();

        // Seed the window: pretend last verify returned [100, 101, 102]
        w.update(&[100, 101, 102], &[]);

        // After update, columns are length L=3:
        //   col0 = [100, 101, 102]
        //   col1 = [101, 102, 102]
        let tree = w.build_drafts(99, None, &mut pool);

        // BFS traversal:
        //   depth 0: [99]                    -> idx 0
        //   depth 1: [100 (col0), 101(col1)] -> idx 1, 2
        //   depth 2: [101 (col0), 102(col1)] -> idx 3, 4
        // L=3 so we emit depth 1 and depth 2 (depth 1..L means d in 2..3 -> only d=2).
        assert_eq!(tree.tokens, vec![99, 100, 101, 101, 102]);
        assert_eq!(tree.parents, vec![-1, 0, 0, 1, 2]);
        assert_eq!(tree.depths, vec![0, 1, 1, 2, 2]);
        tree.validate().unwrap();
        assert!(tree.len() <= w.max_nodes());
    }

    #[test]
    fn build_max_nodes_invariant_holds() {
        let mut w = JacobiWindow::new(5, 5);
        let mut pool = NGramPool::new();
        // Pollute pool with many continuations.
        for k in 0u32..50 {
            pool.insert_trigram((1, 2), k);
        }
        w.update(&[10, 11, 12, 13, 14], &[]);

        let tree = w.build_drafts(2, Some(1), &mut pool);
        tree.validate().unwrap();
        assert!(
            tree.len() <= w.max_nodes(),
            "len={} cap={}",
            tree.len(),
            w.max_nodes()
        );
        // First child layer at most W wide.
        let depth1_count = tree.depths.iter().filter(|&&d| d == 1).count();
        assert!(depth1_count <= w.w());
    }

    #[test]
    fn update_full_acceptance_advances_columns() {
        let mut w = JacobiWindow::new(3, 4);
        w.update(&[5, 6, 7, 8], &[]);
        assert!(w.is_seeded());

        // Column 0 should be the accepted prefix.
        assert_eq!(w.guesses[0], vec![5, 6, 7, 8]);
        // Column 1 should be shifted by 1.
        assert_eq!(w.guesses[1], vec![6, 7, 8, 8]); // padded with last
                                                    // Column 2 should be shifted by 2.
        assert_eq!(w.guesses[2], vec![7, 8, 8, 8]);
    }

    #[test]
    fn update_partial_acceptance_pads_tail() {
        let mut w = JacobiWindow::new(2, 5);
        w.update(&[42, 43], &[]); // only 2 tokens accepted; L=5
        assert_eq!(w.guesses[0], vec![42, 43, 43, 43, 43]);
        assert_eq!(w.guesses[1], vec![43, 43, 43, 43, 43]);
    }

    #[test]
    fn update_empty_acceptance_preserves_state() {
        let mut w = JacobiWindow::new(2, 3);
        w.update(&[10, 11, 12], &[]);
        let snapshot = w.guesses.clone();
        w.update(&[], &[]);
        assert_eq!(w.guesses, snapshot);
        assert!(w.is_seeded());
    }

    #[test]
    fn reset_returns_to_cold_start() {
        let mut w = JacobiWindow::new(2, 3);
        w.update(&[10, 11, 12], &[]);
        assert!(w.is_seeded());
        w.reset();
        assert!(!w.is_seeded());
        for g in &w.guesses {
            assert!(g.is_empty());
        }
    }

    #[test]
    fn path_to_returns_root_to_leaf() {
        let mut w = JacobiWindow::new(2, 3);
        let mut pool = NGramPool::new();
        w.update(&[100, 101, 102], &[]);
        let tree = w.build_drafts(99, None, &mut pool);
        // Leaf at depth 2, col 0 path: 99 -> 100 -> 101.
        let path = tree.path_to(3);
        assert_eq!(path, vec![99, 100, 101]);
    }

    #[test]
    fn validate_catches_corruption() {
        let mut tree = DraftTree {
            tokens: vec![1, 2],
            parents: vec![-1, 5], // bogus parent
            depths: vec![0, 1],
        };
        assert!(tree.validate().is_err());
        tree.parents[1] = 0;
        tree.depths[1] = 7; // wrong depth
        assert!(tree.validate().is_err());
        tree.depths[1] = 1;
        tree.validate().unwrap();
    }
}
