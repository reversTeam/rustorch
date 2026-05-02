//! Pipeline cache + buffer pool.
//!
//! v1 ships:
//! - [`PipelineCache`] — `HashMap<key, Arc<wgpu::ComputePipeline>>`,
//!   keyed by `(op_name, dtype-tag, shape-signature)`. Exposes
//!   `get_or_insert_with` so callers compile WGSL lazily.
//! - [`BufferPool`] — recycles `wgpu::Buffer`s by power-of-two size
//!   bucket. Backed by a `Mutex` so it is safe to share across
//!   threads. v1 uses it as a hint-only pool: callers can opt in
//!   via `acquire`/`release`; `WgpuStorage::allocate` does not yet
//!   route through it.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

/// Key identifying a compiled compute pipeline.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PipelineKey {
    /// Logical op name (e.g. "add", "mul").
    pub op: &'static str,
    /// Dtype tag (e.g. "f32", "f64") for shader specialization.
    pub dtype: &'static str,
    /// Optional shape-signature variant ("scalar", "broadcast", etc.).
    pub variant: &'static str,
}

/// Compiled-pipeline cache. Cheap to clone (Arc-wrapped Mutex inside).
#[derive(Default, Clone)]
pub struct PipelineCache {
    inner: Arc<Mutex<HashMap<PipelineKey, Arc<wgpu::ComputePipeline>>>>,
}

impl PipelineCache {
    /// Look up a pipeline; if absent, build it with the provided
    /// closure and store it.
    pub fn get_or_insert_with<F>(&self, key: PipelineKey, f: F) -> Arc<wgpu::ComputePipeline>
    where
        F: FnOnce() -> wgpu::ComputePipeline,
    {
        let mut guard = self.inner.lock().expect("cache lock poisoned");
        guard.entry(key).or_insert_with(|| Arc::new(f())).clone()
    }

    /// Number of currently-cached pipelines.
    pub fn len(&self) -> usize {
        self.inner.lock().expect("cache lock").len()
    }

    /// True iff the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Aggregate counters reported by a [`BufferPool`]. All fields are
/// monotonically increasing except `bytes_pooled` which tracks the
/// current resident bytes inside the pool's bins.
#[derive(Debug, Default, Clone, Copy)]
pub struct PoolMetricsSnapshot {
    /// Total times a caller asked for a buffer (acquire + acquire_pooled).
    pub acquires: u64,
    /// Of those, how many were served by recycling a pool entry.
    pub hits: u64,
    /// Of those, how many forced a fresh `device.create_buffer`.
    pub misses: u64,
    /// Total times a buffer was returned to the pool (manual or via Drop).
    pub releases: u64,
    /// Total times a buffer was dropped because the pool was at capacity
    /// (or larger than the per-bucket cap if configured).
    pub evictions: u64,
    /// Cumulative bytes ever allocated freshly from the device. Useful
    /// for spotting allocation churn.
    pub bytes_allocated: u64,
    /// Current bytes resident in the pool's bins.
    pub bytes_pooled: u64,
}

#[derive(Debug, Default)]
struct PoolMetrics {
    acquires: AtomicU64,
    hits: AtomicU64,
    misses: AtomicU64,
    releases: AtomicU64,
    evictions: AtomicU64,
    bytes_allocated: AtomicU64,
    bytes_pooled: AtomicU64,
}

impl PoolMetrics {
    fn snapshot(&self) -> PoolMetricsSnapshot {
        PoolMetricsSnapshot {
            acquires: self.acquires.load(Ordering::Relaxed),
            hits: self.hits.load(Ordering::Relaxed),
            misses: self.misses.load(Ordering::Relaxed),
            releases: self.releases.load(Ordering::Relaxed),
            evictions: self.evictions.load(Ordering::Relaxed),
            bytes_allocated: self.bytes_allocated.load(Ordering::Relaxed),
            bytes_pooled: self.bytes_pooled.load(Ordering::Relaxed),
        }
    }
}

/// Configurable pool policy.
#[derive(Debug, Clone, Copy)]
pub struct PoolPolicy {
    /// Max bytes the pool will keep across all bins. Releases that
    /// would push past this limit drop the buffer instead of pooling
    /// it (counted in `evictions`). `u64::MAX` ≈ unbounded.
    pub max_bytes: u64,
    /// Max entries kept per bucket. Useful to avoid pathological
    /// hoarding when one workload allocates and frees many tensors of
    /// the same size in a row. `usize::MAX` ≈ unbounded.
    pub max_per_bucket: usize,
}

impl Default for PoolPolicy {
    fn default() -> Self {
        // 1 GiB total cap, 32 buffers per bucket — generous defaults
        // that work for laptop-sized models without hoarding forever.
        // Honours the `WGPU_POOL_MAX` env var (interpreted as bytes)
        // so users can tune at process-start without recompiling.
        let max_bytes = std::env::var("WGPU_POOL_MAX")
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(1 << 30);
        PoolPolicy {
            max_bytes,
            max_per_bucket: 32,
        }
    }
}

/// Inner state of a [`BufferPool`]. Held behind an `Arc` so that
/// [`crate::pooled::PooledBuffer`] can hold a `Weak<PoolInner>` and
/// detect pool teardown without keeping it alive.
pub struct PoolInner {
    /// Size-bucketed bins of recycled buffers.
    pub(crate) bins: Mutex<HashMap<u64, Vec<wgpu::Buffer>>>,
    /// Live pool policy. Held behind a Mutex so it can be retuned at
    /// runtime; reads on the hot path are short.
    policy: Mutex<PoolPolicy>,
    metrics: PoolMetrics,
}

impl PoolInner {
    /// Allocate a fresh, empty pool inner with default policy.
    fn new() -> Self {
        PoolInner {
            bins: Mutex::new(HashMap::new()),
            policy: Mutex::new(PoolPolicy::default()),
            metrics: PoolMetrics::default(),
        }
    }

    /// Push a buffer back into the pool, applying the eviction policy.
    /// Called by [`crate::pooled::PooledBuffer::drop`].
    pub(crate) fn try_release(&self, buf: wgpu::Buffer, bucket: u64) {
        let policy = *self.policy.lock().expect("policy lock");
        let cur_bytes = self.metrics.bytes_pooled.load(Ordering::Relaxed);
        let mut guard = self.bins.lock().expect("pool lock");
        let bin = guard.entry(bucket).or_default();
        // Reject if either the per-bucket cap or the global byte cap
        // would be exceeded.
        if bin.len() >= policy.max_per_bucket || cur_bytes.saturating_add(bucket) > policy.max_bytes
        {
            self.metrics.evictions.fetch_add(1, Ordering::Relaxed);
            // `buf` falls out of scope and is freed by wgpu.
            return;
        }
        bin.push(buf);
        self.metrics
            .bytes_pooled
            .fetch_add(bucket, Ordering::Relaxed);
        self.metrics.releases.fetch_add(1, Ordering::Relaxed);
    }
}

/// Buffer pool — recycles GPU buffers by size bucket.
///
/// Two access patterns:
/// 1. **Manual** via [`BufferPool::acquire`] / [`BufferPool::release`]
///    — caller owns the buffer's lifecycle.
/// 2. **Auto** via [`BufferPool::acquire_pooled`], which returns a
///    [`crate::pooled::PooledBuffer`] that returns itself to the pool
///    when its last reference is dropped. This is the path
///    [`crate::WgpuStorage::allocate`] uses by default.
#[derive(Clone)]
pub struct BufferPool {
    inner: Arc<PoolInner>,
}

impl Default for BufferPool {
    fn default() -> Self {
        BufferPool {
            inner: Arc::new(PoolInner::new()),
        }
    }
}

impl BufferPool {
    /// Take a buffer that holds at least `size` bytes. The bucket size
    /// is rounded up to the next power of two so that fragmented
    /// allocations recycle effectively.
    ///
    /// **Manual lifecycle**: caller is responsible for calling
    /// [`BufferPool::release`] when done. Prefer
    /// [`BufferPool::acquire_pooled`] for the typical case.
    pub fn acquire(
        &self,
        device: &wgpu::Device,
        size: u64,
        usage: wgpu::BufferUsages,
    ) -> wgpu::Buffer {
        let bucket = size.next_power_of_two().max(64);
        self.inner.metrics.acquires.fetch_add(1, Ordering::Relaxed);
        let mut guard = self.inner.bins.lock().expect("pool lock");
        if let Some(bin) = guard.get_mut(&bucket) {
            if let Some(buf) = bin.pop() {
                self.inner.metrics.hits.fetch_add(1, Ordering::Relaxed);
                self.inner
                    .metrics
                    .bytes_pooled
                    .fetch_sub(bucket, Ordering::Relaxed);
                return buf;
            }
        }
        drop(guard);
        self.inner.metrics.misses.fetch_add(1, Ordering::Relaxed);
        self.inner
            .metrics
            .bytes_allocated
            .fetch_add(bucket, Ordering::Relaxed);
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rustorch-wgpu pooled buffer"),
            size: bucket,
            usage,
            mapped_at_creation: false,
        })
    }

    /// Return a buffer to the pool for reuse. Honors the configured
    /// [`PoolPolicy`]: drops the buffer (and counts it as an eviction)
    /// if either the per-bucket cap or the global byte cap would be
    /// exceeded.
    pub fn release(&self, size: u64, buffer: wgpu::Buffer) {
        let bucket = size.next_power_of_two().max(64);
        self.inner.try_release(buffer, bucket);
    }

    /// Replace the live pool policy. Affects subsequent releases only.
    pub fn set_policy(&self, policy: PoolPolicy) {
        *self.inner.policy.lock().expect("policy lock") = policy;
    }

    /// Snapshot the current metrics. Cheap (atomic loads, no locks).
    pub fn metrics(&self) -> PoolMetricsSnapshot {
        self.inner.metrics.snapshot()
    }

    /// Drop **all** buffers currently held in the pool. Each evicted
    /// buffer is counted in `metrics.evictions`. Used by the
    /// evict-and-retry path of [`BufferPool::try_acquire`] when the
    /// device is OOM. Returns the number of buffers evicted.
    pub fn evict_all(&self) -> usize {
        let mut guard = self.inner.bins.lock().expect("pool lock");
        let mut evicted = 0_usize;
        for (bucket, bin) in guard.iter_mut() {
            evicted += bin.len();
            self.inner
                .metrics
                .bytes_pooled
                .fetch_sub((*bucket) * bin.len() as u64, Ordering::Relaxed);
            self.inner
                .metrics
                .evictions
                .fetch_add(bin.len() as u64, Ordering::Relaxed);
            bin.clear();
        }
        evicted
    }

    /// Acquire a buffer with explicit OOM handling. On failure: drop
    /// every pooled buffer (releasing GPU memory back to the driver)
    /// and retry once. If that still fails, returns
    /// [`crate::error::WgpuError::OutOfMemory`] with a metric snapshot
    /// suitable for telemetry.
    pub fn try_acquire(
        &self,
        device: &wgpu::Device,
        size: u64,
        usage: wgpu::BufferUsages,
    ) -> Result<wgpu::Buffer, crate::error::WgpuError> {
        // First attempt: normal acquire under a captured error scope so
        // an OOM raises a wgpu validation error we can intercept rather
        // than going to the device's uncaptured error handler (which by
        // default panics in `RustyError`).
        device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        let buf = self.acquire(device, size, usage);
        let err = pollster::block_on(device.pop_error_scope());
        if err.is_none() {
            return Ok(buf);
        }
        // First attempt OOM'd. Drop the buffer the device returned (it
        // may be in an invalid state per wgpu's contract), then evict
        // and retry once.
        drop(buf);
        let evicted = self.evict_all();
        // If we evicted nothing, retrying won't free anything new.
        if evicted == 0 {
            let snap = self.metrics();
            return Err(crate::error::WgpuError::OutOfMemory {
                requested: size,
                snapshot: format!(
                    "pool empty (bytes_alloc={}, hits={}, misses={})",
                    snap.bytes_allocated, snap.hits, snap.misses
                ),
            });
        }
        device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        let buf2 = self.acquire(device, size, usage);
        let err2 = pollster::block_on(device.pop_error_scope());
        if err2.is_none() {
            return Ok(buf2);
        }
        let snap = self.metrics();
        Err(crate::error::WgpuError::OutOfMemory {
            requested: size,
            snapshot: format!(
                "after evicting {} buffers: bytes_alloc={}, evictions={}, misses={}",
                evicted, snap.bytes_allocated, snap.evictions, snap.misses
            ),
        })
    }

    /// Acquire a buffer wrapped in a [`crate::pooled::PooledBuffer`]
    /// that will return itself to this pool when dropped.
    ///
    /// The bucket is sized as the next power of two of `size`,
    /// clamped to at least 64 bytes (wgpu's minimum useful buffer).
    pub fn acquire_pooled(
        &self,
        device: &wgpu::Device,
        size: u64,
        usage: wgpu::BufferUsages,
    ) -> crate::pooled::PooledBuffer {
        let bucket = size.next_power_of_two().max(64);
        let buffer = self.acquire(device, bucket, usage);
        crate::pooled::PooledBuffer::from_pool(buffer, bucket, Arc::downgrade(&self.inner))
    }

    /// How many buffers are pooled in total.
    pub fn len(&self) -> usize {
        self.inner
            .bins
            .lock()
            .expect("pool lock")
            .values()
            .map(|v| v.len())
            .sum()
    }

    /// Empty pool?
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Number of buffers currently sitting in the bin for `size`'s
    /// bucket. Useful for tests asserting recycling actually happens.
    pub fn bucket_len(&self, size: u64) -> usize {
        let bucket = size.next_power_of_two().max(64);
        self.inner
            .bins
            .lock()
            .expect("pool lock")
            .get(&bucket)
            .map(Vec::len)
            .unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pipeline_key_hashable() {
        let k1 = PipelineKey {
            op: "add",
            dtype: "f32",
            variant: "scalar",
        };
        let k2 = k1.clone();
        let mut m: HashMap<PipelineKey, u32> = HashMap::new();
        m.insert(k1, 1);
        assert_eq!(m.get(&k2), Some(&1));
    }

    #[test]
    fn pipeline_cache_default_empty() {
        let cache = PipelineCache::default();
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn buffer_pool_default_empty() {
        let pool = BufferPool::default();
        assert!(pool.is_empty());
    }
}
