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

/// Buffer pool — recycles GPU buffers by size bucket.
///
/// v1 is intentionally minimal: callers can `acquire(size)` a buffer
/// (creating a fresh one if no recycled fit), use it, then `release`
/// it back. WgpuStorage doesn't yet route through this — it's wired
/// for future caching of intermediate matmul/conv buffers.
#[derive(Default, Clone)]
pub struct BufferPool {
    inner: Arc<Mutex<HashMap<u64, Vec<wgpu::Buffer>>>>,
}

impl BufferPool {
    /// Take a buffer that holds at least `size` bytes. The bucket size
    /// is rounded up to the next power of two so that fragmented
    /// allocations recycle effectively.
    pub fn acquire(
        &self,
        device: &wgpu::Device,
        size: u64,
        usage: wgpu::BufferUsages,
    ) -> wgpu::Buffer {
        let bucket = size.next_power_of_two().max(64);
        let mut guard = self.inner.lock().expect("pool lock");
        if let Some(bin) = guard.get_mut(&bucket) {
            if let Some(buf) = bin.pop() {
                return buf;
            }
        }
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rustorch-wgpu pooled buffer"),
            size: bucket,
            usage,
            mapped_at_creation: false,
        })
    }

    /// Return a buffer to the pool for reuse.
    pub fn release(&self, size: u64, buffer: wgpu::Buffer) {
        let bucket = size.next_power_of_two().max(64);
        let mut guard = self.inner.lock().expect("pool lock");
        guard.entry(bucket).or_default().push(buffer);
    }

    /// How many buffers are pooled in total.
    pub fn len(&self) -> usize {
        self.inner
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
