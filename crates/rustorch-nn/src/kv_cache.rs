//! Key-Value cache for autoregressive transformer decoding (T41).
//!
//! Without a KV-cache, generating N tokens requires N forward passes
//! that each recompute attention over the entire prefix — O(N²) per
//! layer in compute and bandwidth. The KV-cache stores the K/V
//! projections of past tokens once, then appends only the new
//! token's K/V on each step, reducing per-token cost to O(N) per
//! layer (the new query attends to all cached K/V).
//!
//! ## Layout
//!
//! Each layer caches two `[batch, n_heads, max_seq, head_dim]` f32
//! buffers (K and V), pre-allocated to the maximum sequence length
//! the user expects. `current_len` tracks how many positions are
//! valid; the kernel reads only `[..current_len]` slices.
//!
//! ## Usage pattern
//!
//! ```ignore
//! let mut cache = KVCache::new(num_layers=12, batch=1,
//!                              n_heads=12, head_dim=64,
//!                              max_seq=1024);
//! for token_step in 0..max_new_tokens {
//!     for (layer_idx, layer) in layers.iter().enumerate() {
//!         let (q, k_new, v_new) = project_qkv(layer, &x);
//!         cache.append(layer_idx, &k_new, &v_new);
//!         // Attention uses cache.k_view(layer_idx) and v_view…
//!     }
//!     cache.advance();
//! }
//! ```
//!
//! ## v0 limitations
//!
//! - f32 only (Qwen INT4 / bf16 caches land with the quant work).
//! - Pre-allocates max_seq up front; future work may switch to a
//!   ring buffer + sliding window for very long contexts.
//! - Caller must ensure `append` is called once per layer per token
//!   step before `advance()`.

/// Per-layer K and V caches for a transformer block.
///
/// Stored as flat `Vec<f32>` (not Tensor) because the cache buffer
/// must be in-place mutable across token steps; the Tensor type's
/// `Arc<Storage>` model copy-on-writes on aliased mutation, which
/// would defeat the cache. Callers wrap the slice in a
/// `[batch, n_heads, current_len, head_dim]` view at attention time
/// (see `flash_forward`'s `&[f32]` API).
pub struct KVLayerCache {
    /// `[batch, n_heads, max_seq, head_dim]` row-major f32 buffer
    /// for K. Only positions `[..current_len]` are valid.
    k: Vec<f32>,
    /// Same shape as `k`, holding V.
    v: Vec<f32>,
    batch: usize,
    n_heads: usize,
    max_seq: usize,
    head_dim: usize,
}

impl KVLayerCache {
    fn new(batch: usize, n_heads: usize, max_seq: usize, head_dim: usize) -> Self {
        let n = batch * n_heads * max_seq * head_dim;
        KVLayerCache {
            k: vec![0.0_f32; n],
            v: vec![0.0_f32; n],
            batch,
            n_heads,
            max_seq,
            head_dim,
        }
    }

    /// Write the K and V projections for `seq_added` new positions
    /// at offset `current_len`. Both `new_k` and `new_v` must be
    /// `[batch, n_heads, seq_added, head_dim]` row-major slices.
    fn append(
        &mut self,
        current_len: usize,
        seq_added: usize,
        new_k: &[f32],
        new_v: &[f32],
    ) -> Result<(), KVCacheError> {
        if current_len + seq_added > self.max_seq {
            return Err(KVCacheError::Overflow {
                requested: current_len + seq_added,
                max: self.max_seq,
            });
        }
        let bhsd = self.batch * self.n_heads * seq_added * self.head_dim;
        if new_k.len() != bhsd || new_v.len() != bhsd {
            return Err(KVCacheError::WrongInputLen {
                expected: bhsd,
                got_k: new_k.len(),
                got_v: new_v.len(),
            });
        }
        let hd = self.head_dim;
        let s_max = self.max_seq;
        let s_new = seq_added;
        for b in 0..self.batch {
            for h in 0..self.n_heads {
                let src_off = b * self.n_heads * s_new * hd + h * s_new * hd;
                let dst_off = b * self.n_heads * s_max * hd + h * s_max * hd + current_len * hd;
                self.k[dst_off..dst_off + s_new * hd]
                    .copy_from_slice(&new_k[src_off..src_off + s_new * hd]);
                self.v[dst_off..dst_off + s_new * hd]
                    .copy_from_slice(&new_v[src_off..src_off + s_new * hd]);
            }
        }
        Ok(())
    }
}

/// All-layer KV cache, owning one [`KVLayerCache`] per transformer
/// layer. Tracks the current sequence length.
pub struct KVCache {
    layers: Vec<KVLayerCache>,
    /// Number of valid positions in each layer's cache. Bumped by
    /// [`KVCache::advance`] after every layer has been appended to
    /// for the current token.
    current_len: usize,
    /// Maximum sequence length we pre-allocated for.
    max_seq: usize,
}

impl KVCache {
    /// Build a fresh cache with `num_layers` empty layer slots.
    pub fn new(
        num_layers: usize,
        batch: usize,
        n_heads: usize,
        head_dim: usize,
        max_seq: usize,
    ) -> Self {
        let layers = (0..num_layers)
            .map(|_| KVLayerCache::new(batch, n_heads, max_seq, head_dim))
            .collect();
        KVCache {
            layers,
            current_len: 0,
            max_seq,
        }
    }

    /// Number of layers cached.
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Maximum sequence length the cache was sized for.
    pub fn max_seq(&self) -> usize {
        self.max_seq
    }

    /// Number of valid token positions cached. Reads return slices
    /// of this length along the sequence axis.
    pub fn current_len(&self) -> usize {
        self.current_len
    }

    /// Reset to length zero (cache contents become stale but the
    /// allocation is preserved).
    pub fn clear(&mut self) {
        self.current_len = 0;
    }

    /// Append `seq_added` new (K, V) rows to layer `layer_idx`.
    /// Caller is expected to call this for every layer (in order)
    /// before [`advance`] increments the position.
    pub fn append(
        &mut self,
        layer_idx: usize,
        seq_added: usize,
        new_k: &[f32],
        new_v: &[f32],
    ) -> Result<(), KVCacheError> {
        let num_layers = self.layers.len();
        let current_len = self.current_len;
        let layer = self
            .layers
            .get_mut(layer_idx)
            .ok_or(KVCacheError::LayerOutOfRange {
                layer: layer_idx,
                num_layers,
            })?;
        layer.append(current_len, seq_added, new_k, new_v)
    }

    /// Advance the position counter by `seq_added`. Call once per
    /// token step (after every layer's `append`).
    pub fn advance(&mut self, seq_added: usize) -> Result<(), KVCacheError> {
        let new_len = self.current_len + seq_added;
        if new_len > self.max_seq {
            return Err(KVCacheError::Overflow {
                requested: new_len,
                max: self.max_seq,
            });
        }
        self.current_len = new_len;
        Ok(())
    }

    /// Borrow the K cache buffer for `layer_idx`. The returned
    /// slice is the full `[batch, n_heads, max_seq, head_dim]`
    /// f32 buffer; only the first `current_len()` positions along
    /// the seq axis are valid.
    pub fn k_buffer(&self, layer_idx: usize) -> Option<&[f32]> {
        Some(self.layers.get(layer_idx)?.k.as_slice())
    }

    /// Borrow the V cache buffer for `layer_idx`. Same layout as
    /// `k_buffer`.
    pub fn v_buffer(&self, layer_idx: usize) -> Option<&[f32]> {
        Some(self.layers.get(layer_idx)?.v.as_slice())
    }

    /// Per-axis dims of the cache as `(batch, n_heads, max_seq,
    /// head_dim)`. Convenience for callers building flash_attention
    /// shape descriptors.
    pub fn dims(&self) -> Option<(usize, usize, usize, usize)> {
        let layer = self.layers.first()?;
        Some((layer.batch, layer.n_heads, layer.max_seq, layer.head_dim))
    }
}

/// Errors raised by the KV cache.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KVCacheError {
    /// Caller asked to append beyond `max_seq`.
    Overflow {
        /// Requested new total length.
        requested: usize,
        /// Maximum the cache was sized for.
        max: usize,
    },
    /// Append's input slice has the wrong length.
    WrongInputLen {
        /// Expected `batch * n_heads * seq_added * head_dim`.
        expected: usize,
        /// Actual K input length.
        got_k: usize,
        /// Actual V input length.
        got_v: usize,
    },
    /// Layer index out of bounds.
    LayerOutOfRange {
        /// Requested.
        layer: usize,
        /// Number of layers in the cache.
        num_layers: usize,
    },
    /// Internal: the cache tensor was aliased and could not acquire
    /// unique mutable access. Should not happen in v0 usage.
    Aliased,
}

impl std::fmt::Display for KVCacheError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            KVCacheError::Overflow { requested, max } => {
                write!(
                    f,
                    "kv_cache overflow: requested {requested} positions, max {max}"
                )
            },
            KVCacheError::WrongInputLen {
                expected,
                got_k,
                got_v,
            } => write!(
                f,
                "kv_cache append: expected {expected} f32, got K={got_k} V={got_v}"
            ),
            KVCacheError::LayerOutOfRange { layer, num_layers } => {
                write!(f, "kv_cache layer {layer} out of range (have {num_layers})")
            },
            KVCacheError::Aliased => write!(f, "kv_cache: tensor was aliased"),
        }
    }
}

impl std::error::Error for KVCacheError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kv_cache_append_and_read_back() {
        let mut cache = KVCache::new(2, 1, 2, 4, 8);
        assert_eq!(cache.num_layers(), 2);
        assert_eq!(cache.max_seq(), 8);
        assert_eq!(cache.current_len(), 0);

        // Append 3 tokens to layer 0.
        let bhsd = 2 * 3 * 4;
        let new_k: Vec<f32> = (0..bhsd).map(|i| i as f32 * 0.1).collect();
        let new_v: Vec<f32> = (0..bhsd).map(|i| i as f32 * 0.01).collect();
        cache.append(0, 3, &new_k, &new_v).unwrap();
        cache.append(1, 3, &new_k, &new_v).unwrap();
        cache.advance(3).unwrap();

        assert_eq!(cache.current_len(), 3);

        // Layer 0 K buffer should hold our values at offset 0.
        let k_buf = cache.k_buffer(0).unwrap();
        // For (b=0, h=0, s in 0..3, hd in 0..4): offset = h*max_seq*hd + s*hd + d
        //   h=0, s=0, d=0..4 -> indices [0, 1, 2, 3]
        //   h=0, s=1, d=0..4 -> indices [4, 5, 6, 7]
        //   h=0, s=2, d=0..4 -> indices [8, 9, 10, 11]
        //   h=1, s=0, d=0..4 -> indices [32, 33, 34, 35] (max_seq=8 * hd=4)
        for s in 0..3 {
            for d in 0..4 {
                let cache_idx = s * 4 + d;
                let new_idx = s * 4 + d;
                assert!(
                    (k_buf[cache_idx] - new_k[new_idx]).abs() < 1e-6,
                    "h=0 s={s} d={d}"
                );
            }
        }
    }

    #[test]
    fn kv_cache_overflow_returns_err() {
        let mut cache = KVCache::new(1, 1, 1, 4, 4);
        let n = 5 * 4;
        let buf = vec![0.0_f32; n];
        let err = cache.append(0, 5, &buf, &buf).unwrap_err();
        assert!(matches!(err, KVCacheError::Overflow { .. }));
    }

    #[test]
    fn kv_cache_wrong_input_len() {
        let mut cache = KVCache::new(1, 1, 1, 4, 8);
        let bad = vec![0.0_f32; 3]; // not 1*1*1*4 = 4
        let err = cache.append(0, 1, &bad, &bad).unwrap_err();
        assert!(matches!(err, KVCacheError::WrongInputLen { .. }));
    }

    #[test]
    fn kv_cache_layer_out_of_range() {
        let mut cache = KVCache::new(2, 1, 1, 4, 8);
        let buf = vec![0.0_f32; 4];
        let err = cache.append(2, 1, &buf, &buf).unwrap_err();
        assert!(matches!(err, KVCacheError::LayerOutOfRange { .. }));
    }

    #[test]
    fn kv_cache_clear_resets_current_len() {
        let mut cache = KVCache::new(1, 1, 1, 4, 8);
        let buf = vec![0.0_f32; 4];
        cache.append(0, 1, &buf, &buf).unwrap();
        cache.advance(1).unwrap();
        assert_eq!(cache.current_len(), 1);
        cache.clear();
        assert_eq!(cache.current_len(), 0);
    }
}
