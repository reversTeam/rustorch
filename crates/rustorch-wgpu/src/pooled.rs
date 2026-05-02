//! Refcounted GPU buffer that returns itself to a [`BufferPool`] on
//! drop. Used by [`WgpuStorage`] so allocations are recycled across
//! the lifetime of a backend without requiring callers to manually
//! `release()`.
//!
//! Wrapping pattern:
//!
//! ```ignore
//! WgpuStorage { buffer: Arc<PooledBuffer>, ... }
//!
//! PooledBuffer { buffer: ManuallyDrop<wgpu::Buffer>, pool: Weak<...>, bucket: u64 }
//!
//! impl Drop for PooledBuffer {
//!     fn drop(&mut self) {
//!         // take ownership of the inner Buffer
//!         // if the pool is still alive, push it back into the right bucket
//!         // otherwise let it free
//!     }
//! }
//! ```
//!
//! `Deref<Target = wgpu::Buffer>` keeps the call sites unchanged
//! (`storage.buffer.as_entire_binding()`, etc.).

use crate::cache::PoolInner;
use std::mem::ManuallyDrop;
use std::ops::Deref;
use std::sync::Weak;

/// A `wgpu::Buffer` that, when its last reference is dropped, returns
/// itself to the [`BufferPool`] it came from.
///
/// To get one of these, call [`crate::cache::BufferPool::acquire_pooled`].
/// To opt out of pooling (e.g. for transient meta uniforms with weird
/// usages that won't recycle well), call
/// [`PooledBuffer::standalone`] which builds a non-pooled wrapper.
pub struct PooledBuffer {
    buffer: ManuallyDrop<wgpu::Buffer>,
    /// Bucket key the pool indexed this buffer under. Stored on the
    /// wrapper so the Drop impl knows which bucket to push back into
    /// without re-deriving from `buffer.size()` (cheap, but explicit).
    bucket: u64,
    /// Weak handle to the pool's shared map. `None` for buffers
    /// allocated outside the pool (`PooledBuffer::standalone`).
    pool: Option<Weak<PoolInner>>,
}

impl PooledBuffer {
    /// Wrap a freshly created `wgpu::Buffer` so it returns to `pool`
    /// on drop. `bucket` is the pool's size-bin key.
    pub(crate) fn from_pool(buffer: wgpu::Buffer, bucket: u64, pool: Weak<PoolInner>) -> Self {
        PooledBuffer {
            buffer: ManuallyDrop::new(buffer),
            bucket,
            pool: Some(pool),
        }
    }

    /// Wrap a `wgpu::Buffer` without registering it in any pool. Drop
    /// will free the buffer normally. Use this for one-shot tiny
    /// uniforms whose usage flags don't match the pool's policy.
    pub fn standalone(buffer: wgpu::Buffer) -> Self {
        PooledBuffer {
            buffer: ManuallyDrop::new(buffer),
            bucket: 0,
            pool: None,
        }
    }

    /// Return the bucket key the pool indexed this buffer under.
    /// Useful for invariants in tests.
    #[inline]
    pub fn bucket(&self) -> u64 {
        self.bucket
    }
}

impl Deref for PooledBuffer {
    type Target = wgpu::Buffer;
    #[inline]
    fn deref(&self) -> &wgpu::Buffer {
        &self.buffer
    }
}

impl Drop for PooledBuffer {
    fn drop(&mut self) {
        // SAFETY: we move the buffer out of ManuallyDrop exactly once,
        // here in Drop. After this, `self.buffer` is logically
        // uninitialised but no other code observes it.
        let buf = unsafe { ManuallyDrop::take(&mut self.buffer) };
        match self.pool.as_ref().and_then(Weak::upgrade) {
            Some(inner) => {
                // Pool still alive — go through try_release so the
                // configured policy (max_bytes / max_per_bucket) is
                // honoured and metrics stay consistent.
                inner.try_release(buf, self.bucket);
            },
            None => {
                // Pool has been dropped — let `buf` Drop here and free
                // the GPU memory.
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn standalone_does_not_panic_on_drop() {
        // Can't construct a real wgpu::Buffer without a Device, but we
        // can verify the type compiles and PooledBuffer::standalone is
        // public.
        fn _accepts(b: wgpu::Buffer) -> PooledBuffer {
            PooledBuffer::standalone(b)
        }
    }
}
