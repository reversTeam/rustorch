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
    fn evict_all_clears_pool_and_decrements_pooled_bytes() {
        let backend = WgpuBackend::new_blocking().expect("init wgpu");
        let pool = BufferPool::default();
        let s = WgpuStorage::allocate_pooled(&backend.device, &pool, 64, Dtype::F32).unwrap();
        drop(s);
        assert_eq!(pool.len(), 1);
        let evicted = pool.evict_all();
        assert_eq!(evicted, 1);
        assert!(pool.is_empty());
        let m = pool.metrics();
        assert_eq!(m.bytes_pooled, 0);
        assert!(m.evictions >= 1);
    }

    #[test]
    fn try_acquire_succeeds_for_normal_allocation() {
        let backend = WgpuBackend::new_blocking().expect("init wgpu");
        let pool = BufferPool::default();
        // 4 KiB tensor — well within any GPU's budget.
        let buf = pool
            .try_acquire(
                &backend.device,
                4096,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            )
            .expect("normal allocation must succeed");
        assert!(buf.size() >= 4096);
    }

    #[test]
    fn pooled_storage_send_and_sync() {
        // Compile-time assertion: the Storage type can cross thread
        // boundaries. If WgpuStorage stops being Send + Sync, this
        // test fails to compile.
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<WgpuStorage>();
    }

    #[test]
    fn pooled_storage_zero_numel_drop_is_safe() {
        // numel = 0 → buffer size clamps to 4 (wgpu min).  Drop
        // should not panic; pool counts the slot exactly once.
        let backend = WgpuBackend::new_blocking().expect("init");
        let pool = BufferPool::default();
        let s = WgpuStorage::allocate_pooled(&backend.device, &pool, 0, Dtype::F32).unwrap();
        drop(s);
        // Zero-sized allocation still hits the pool once with the
        // 4-byte minimum bucket.
        assert_eq!(pool.len(), 1);
    }

    #[test]
    fn pooled_storage_threaded_clone_is_safe() {
        // Spawn N threads, each clones the same storage, increments,
        // and drops. After all threads, the storage's refcount is back
        // to one and dropping it returns the buffer to the pool.
        use std::sync::Arc as StdArc;
        use std::thread;

        let backend = WgpuBackend::new_blocking().expect("init");
        let pool = BufferPool::default();
        let storage = WgpuStorage::allocate_pooled(&backend.device, &pool, 64, Dtype::F32).unwrap();
        let storage = StdArc::new(storage);

        let handles: Vec<_> = (0..8)
            .map(|_| {
                let s = storage.clone();
                thread::spawn(move || {
                    let _local = (*s).clone();
                    // _local drops here — but the original Arc-wrapped
                    // copy is still alive in the parent.
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
        // Outer Arc still alive — buffer not returned to pool yet.
        assert!(pool.is_empty());
        // try_unwrap returns Err if there are other strong refs;
        // here we know we're the unique holder so unwrap to the inner
        // value via map_err to side-step the missing Debug impl on
        // WgpuStorage.
        let inner = StdArc::try_unwrap(storage).map_err(|_| ()).unwrap();
        drop(inner);
        // Now the underlying PooledBuffer is dropped → pool gains 1.
        assert_eq!(pool.len(), 1);
    }

    #[test]
    fn pool_max_bytes_zero_disables_pooling() {
        use crate::cache::PoolPolicy;
        let backend = WgpuBackend::new_blocking().expect("init");
        let pool = BufferPool::default();
        // Cap to zero → every release must evict.
        pool.set_policy(PoolPolicy {
            max_bytes: 0,
            max_per_bucket: 64,
        });
        let s = WgpuStorage::allocate_pooled(&backend.device, &pool, 64, Dtype::F32).unwrap();
        drop(s);
        // Pool stays empty because max_bytes=0.
        assert!(pool.is_empty());
        let m = pool.metrics();
        assert!(m.evictions >= 1);
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
