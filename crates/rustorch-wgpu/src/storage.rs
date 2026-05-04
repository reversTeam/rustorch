//! `WgpuStorage` — refcounted handle to a `wgpu::Buffer` shared between
//! kernel dispatch (this crate) and the `rustorch-core` Tensor's
//! `Storage::Wgpu(...)` variant (Task A — P3.Z Storage Option A).
//!
//! Architecture (post-Task A):
//!
//! ```text
//!   rustorch_wgpu::WgpuStorage {
//!       pub buffer: rustorch_core::tensor::storage::WgpuStorage,  // shared, Arc-clonable
//!       pub dtype:  Dtype,
//!       pub numel:  usize,
//!   }
//!   ↑ kernel-side wrapper carrying dtype + numel metadata
//!
//!   rustorch_core::tensor::storage::WgpuStorage  ← Arc<WgpuStorageInner>
//!     - holds the actual wgpu::Buffer (ManuallyDrop)
//!     - has a Drop hook (Box<dyn FnOnce>) for pool return-on-drop
//!     - Derefs to &wgpu::Buffer so existing call sites that did
//!       `storage.buffer.as_entire_binding()` keep working unchanged.
//! ```
//!
//! When a Tensor's `Storage::Wgpu(handle)` and a kernel-side
//! `WgpuStorage::buffer` reference the SAME core handle (Arc clone),
//! the buffer survives across op boundaries with no host↔device round
//! trip — that's the perf win of Storage Option A.

use crate::cache::BufferPool;
use crate::error::WgpuError;
use rustorch_core::tensor::dtype::Dtype;
use rustorch_core::tensor::storage::WgpuStorage as CoreWgpuStorage;

/// A refcounted GPU buffer + dtype + element count. The buffer is held
/// inside a `core::WgpuStorage` so it can be cloned cheaply into a
/// Tensor's `Storage::Wgpu(...)` variant — both then share the same
/// `Arc<WgpuStorageInner>` and the buffer is freed (or returned to
/// pool via the on_drop hook) when the LAST clone drops.
#[derive(Clone)]
pub struct WgpuStorage {
    /// The shared GPU buffer. `core::WgpuStorage` Derefs to
    /// `&wgpu::Buffer`, so existing kernel call sites that wrote
    /// `storage.buffer.as_entire_binding()` keep working.
    pub buffer: CoreWgpuStorage,
    /// Element dtype (the wgpu buffer is just bytes; we track dtype
    /// here for kernel selection / parity checks).
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
            buffer: CoreWgpuStorage::standalone(buffer, n_bytes),
            dtype,
            numel,
        })
    }

    /// Allocate via `pool` so that, when this storage's last clone is
    /// dropped, the underlying buffer is recycled into the pool's
    /// matching bucket. The bucket key is `size.next_power_of_two().max(64)`
    /// (matches [`BufferPool::acquire`]).
    pub fn allocate_pooled(
        device: &wgpu::Device,
        pool: &BufferPool,
        numel: usize,
        dtype: Dtype,
    ) -> Result<Self, WgpuError> {
        let n_bytes = numel * dtype.byte_size();
        let size = (n_bytes as u64).max(4);
        let bucket = size.next_power_of_two().max(64);
        let buffer = pool.acquire(
            device,
            bucket,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        );
        // Capture a Weak<PoolInner> in the on_drop closure so the
        // buffer is returned to the pool when the last clone of the
        // core::WgpuStorage drops. If the pool has been torn down by
        // then, `Weak::upgrade` returns None and the buffer is freed
        // normally.
        let pool_weak = pool.weak_inner();
        let core = CoreWgpuStorage::with_pool_return(buffer, n_bytes, move |buf| {
            if let Some(inner) = pool_weak.upgrade() {
                inner.try_release(buf, bucket);
            }
            // else: pool was dropped, `buf` falls out and frees normally.
        });
        Ok(WgpuStorage {
            buffer: core,
            dtype,
            numel,
        })
    }

    /// Total byte size of the underlying buffer (`numel * dtype.byte_size()`,
    /// possibly padded to wgpu minimum).
    pub fn byte_size(&self) -> usize {
        self.numel * self.dtype.byte_size()
    }

    /// Build a `WgpuStorage` from an existing `core::WgpuStorage` that
    /// was extracted from a Tensor's `Storage::Wgpu(...)`. Used by the
    /// no-round-trip path in `transfer::to_gpu` and `backend_impl.rs`
    /// to skip the host upload when the input tensor is already on
    /// the GPU.
    pub fn from_core(core: CoreWgpuStorage, dtype: Dtype, numel: usize) -> Self {
        WgpuStorage {
            buffer: core,
            dtype,
            numel,
        }
    }

    /// Borrow the inner `core::WgpuStorage` so it can be cloned into a
    /// Tensor's `Storage::Wgpu(...)` variant.
    #[inline]
    pub fn core_handle(&self) -> &CoreWgpuStorage {
        &self.buffer
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
}
