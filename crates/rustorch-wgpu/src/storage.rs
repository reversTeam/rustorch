//! `WgpuStorage` — refcounted handle to a `wgpu::Buffer` that returns
//! itself to a [`BufferPool`] when the last reference is dropped.
//!
//! See [`crate::pooled`] for the drop-to-pool wrapper.

use crate::cache::BufferPool;
use crate::error::WgpuError;
use crate::pooled::PooledBuffer;
use rustorch_core::tensor::dtype::Dtype;
use std::sync::Arc;

/// A refcounted GPU buffer + dtype. The buffer is shared via `Arc<PooledBuffer>`
/// so views/slices can clone cheaply, AND the underlying wgpu::Buffer is
/// returned to the source pool when the last clone is dropped.
#[derive(Clone)]
pub struct WgpuStorage {
    /// Underlying wgpu buffer wrapped in a pool-aware drop hook.
    pub buffer: Arc<PooledBuffer>,
    /// Element dtype (kept here because wgpu::Buffer is just bytes).
    pub dtype: Dtype,
    /// Number of elements (`bytes / dtype.byte_size()`).
    pub numel: usize,
}

impl WgpuStorage {
    /// Allocate a fresh GPU buffer of the requested `numel` elements
    /// of `dtype`. Buffer is created with STORAGE | COPY_SRC | COPY_DST
    /// usage so it can participate in compute shaders + transfers.
    ///
    /// **No pool**: this path bypasses the [`BufferPool`] (the
    /// allocation goes straight to the device, and will Drop normally
    /// instead of being recycled). Prefer
    /// [`WgpuStorage::allocate_pooled`] when a pool is available.
    pub fn allocate(device: &wgpu::Device, numel: usize, dtype: Dtype) -> Result<Self, WgpuError> {
        let n_bytes = numel * dtype.byte_size();
        let size = (n_bytes as u64).max(4);
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rustorch-wgpu storage"),
            size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Ok(WgpuStorage {
            buffer: Arc::new(PooledBuffer::standalone(buffer)),
            dtype,
            numel,
        })
    }

    /// Allocate via `pool` so that, when this storage's last clone is
    /// dropped, the underlying buffer is recycled into the pool's
    /// matching bucket.
    pub fn allocate_pooled(
        device: &wgpu::Device,
        pool: &BufferPool,
        numel: usize,
        dtype: Dtype,
    ) -> Result<Self, WgpuError> {
        let n_bytes = numel * dtype.byte_size();
        let size = (n_bytes as u64).max(4);
        let pooled = pool.acquire_pooled(
            device,
            size,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        );
        Ok(WgpuStorage {
            buffer: Arc::new(pooled),
            dtype,
            numel,
        })
    }

    /// Total byte size of the underlying buffer (`numel * dtype.byte_size()`,
    /// possibly padded to wgpu minimum).
    pub fn byte_size(&self) -> usize {
        self.numel * self.dtype.byte_size()
    }
}

#[cfg(all(test, feature = "gpu-tests"))]
mod gpu_pool_tests {
    use super::*;
    use crate::backend::WgpuBackend;

    #[test]
    fn pooled_storage_returns_to_pool_on_last_drop() {
        let backend = WgpuBackend::new_blocking().expect("init wgpu");
        let pool = BufferPool::default();
        assert!(pool.is_empty());

        // Round numel so the bucket is deterministic.
        let storage = WgpuStorage::allocate_pooled(&backend.device, &pool, 64, Dtype::F32).unwrap();
        // Pool stays empty while the storage is alive.
        assert!(pool.is_empty());
        let clone = storage.clone();
        assert!(pool.is_empty(), "clone should not return to pool yet");
        drop(clone);
        assert!(
            pool.is_empty(),
            "second-to-last reference should not return either"
        );
        drop(storage);
        // After last reference drops, exactly one buffer should be in
        // the matching bucket (256-byte → 256, the next pow2 of 64*4).
        assert_eq!(pool.len(), 1, "buffer should have returned to pool");
        // And acquiring a same-sized storage should now reuse that buffer.
        let storage2 =
            WgpuStorage::allocate_pooled(&backend.device, &pool, 64, Dtype::F32).unwrap();
        assert_eq!(pool.len(), 0, "fresh allocation should pop from pool");
        drop(storage2);
        assert_eq!(pool.len(), 1, "drop again returns to pool");
    }

    #[test]
    fn standalone_storage_does_not_touch_pool() {
        let backend = WgpuBackend::new_blocking().expect("init wgpu");
        let pool = BufferPool::default();
        // allocate() bypasses the pool intentionally.
        let storage = WgpuStorage::allocate(&backend.device, 64, Dtype::F32).unwrap();
        drop(storage);
        assert!(pool.is_empty());
    }

    #[test]
    fn pool_policy_evicts_when_bucket_full() {
        use crate::cache::PoolPolicy;
        let backend = WgpuBackend::new_blocking().expect("init wgpu");
        let pool = BufferPool::default();
        // Cap each bucket at 1 entry — second drop must evict, not pool.
        pool.set_policy(PoolPolicy {
            max_bytes: u64::MAX,
            max_per_bucket: 1,
        });
        let s1 = WgpuStorage::allocate_pooled(&backend.device, &pool, 64, Dtype::F32).unwrap();
        let s2 = WgpuStorage::allocate_pooled(&backend.device, &pool, 64, Dtype::F32).unwrap();
        drop(s1);
        drop(s2);
        // Bucket cap is 1, so the second drop is evicted (not pooled).
        assert_eq!(pool.len(), 1);
        let m = pool.metrics();
        assert_eq!(m.evictions, 1);
        assert_eq!(m.acquires, 2);
        assert_eq!(m.misses, 2);
    }

    #[test]
    fn pool_metrics_count_hits_after_recycling() {
        let backend = WgpuBackend::new_blocking().expect("init wgpu");
        let pool = BufferPool::default();

        let s1 = WgpuStorage::allocate_pooled(&backend.device, &pool, 64, Dtype::F32).unwrap();
        drop(s1);
        let m_after_first = pool.metrics();
        assert_eq!(m_after_first.misses, 1);
        assert_eq!(m_after_first.hits, 0);

        // Same-bucket allocation should hit.
        let s2 = WgpuStorage::allocate_pooled(&backend.device, &pool, 64, Dtype::F32).unwrap();
        let m_after_second = pool.metrics();
        assert_eq!(m_after_second.hits, 1);
        assert_eq!(m_after_second.misses, 1, "no extra device alloc");
        drop(s2);
    }
}
