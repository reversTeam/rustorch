//! Storage — refcounted byte buffer (RFC-0002 / RFC-0004, P1.1 task `Storage`).
//!
//! `Storage` is the **owner of buffer memory** that backs one or many
//! [`Tensor`](super::Tensor) views. Multiple views (e.g. `t.transpose()`,
//! `t.slice()`) share the same `Storage` via `Arc<StorageInner>` —
//! cloning is a single atomic refcount bump.
//!
//! v1 ships only the `Cpu` variant; the enum is `#[non_exhaustive]` so
//! `Cuda` / `Wgpu` / `Metal` can be added in later phases without
//! breaking the public match-on-storage discipline.
//!
//! ```
//! use rustorch_core::tensor::storage::Storage;
//! use rustorch_core::tensor::dtype::Dtype;
//!
//! // Allocate 1024 zero-initialised f32 elements on CPU (4096 bytes,
//! // aligned to 64 bytes).
//! let s = Storage::cpu_zeroed(1024 * 4).unwrap();
//! assert_eq!(s.byte_len(), 4096);
//! assert!(s.is_unique());      // no other clones exist yet
//! assert_eq!(s.strong_count(), 1);
//! let _view = s.clone();
//! assert!(!s.is_unique());     // a view exists now
//! assert_eq!(s.strong_count(), 2);
//! # let _ = Dtype::F32;
//! ```

use core::fmt;
use core::ptr::NonNull;
use core::slice;
use std::alloc::{self, Layout as AllocLayout};
use std::sync::Arc;

/// Cache-line alignment used for every CPU allocation. Matches the
/// alignment our SIMD codepaths assume (AVX2 / NEON 128-bit) so that
/// every buffer is safely loadable as a 32-byte aligned vector — the
/// 64-byte choice gives us one cache-line of slack for AVX-512.
pub const CPU_ALIGN: usize = 64;

/// Refcounted, type-erased buffer that backs a tensor.
///
/// `#[non_exhaustive]` so adding `Cuda(...)` / `Wgpu(...)` later is
/// non-breaking for downstream code that uses pattern matching with a
/// catch-all arm.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum Storage {
    /// Aligned heap buffer on the host (CPU).
    Cpu(CpuStorage),
}

impl Storage {
    /// Allocate a zero-initialised CPU buffer of `byte_len` bytes,
    /// aligned to [`CPU_ALIGN`].
    ///
    /// `byte_len == 0` returns a sentinel storage backed by
    /// `NonNull::dangling()` — never dereference its pointer when
    /// `byte_len() == 0`.
    pub fn cpu_zeroed(byte_len: usize) -> Result<Storage, StorageError> {
        Ok(Storage::Cpu(CpuStorage::zeroed(byte_len)?))
    }

    /// Allocate an *uninitialised* CPU buffer of `byte_len` bytes,
    /// aligned to [`CPU_ALIGN`]. Caller is responsible for filling
    /// every byte before reading.
    ///
    /// # Safety
    ///
    /// The returned bytes have unspecified values; reading them is
    /// undefined behaviour until they are explicitly written. Most
    /// callers should use [`Storage::cpu_zeroed`] instead.
    pub unsafe fn cpu_uninit(byte_len: usize) -> Result<Storage, StorageError> {
        Ok(Storage::Cpu(CpuStorage::uninit(byte_len)?))
    }

    /// Build a CPU storage from a `Vec<u8>`. Convenience constructor
    /// used by tests and by the safetensors loader. Reallocates if the
    /// vec's pointer is not already 64-byte aligned.
    pub fn cpu_from_bytes(bytes: Vec<u8>) -> Result<Storage, StorageError> {
        Ok(Storage::Cpu(CpuStorage::from_bytes(bytes)?))
    }

    /// Total bytes in the buffer.
    #[inline]
    pub fn byte_len(&self) -> usize {
        match self {
            Storage::Cpu(s) => s.byte_len,
        }
    }

    /// `true` iff this `Storage` is the unique owner (no clones exist).
    /// Used by in-place ops to take a fast path that avoids COW.
    #[inline]
    pub fn is_unique(&self) -> bool {
        self.strong_count() == 1
    }

    /// Number of strong references to the underlying buffer (Arc count).
    #[inline]
    pub fn strong_count(&self) -> usize {
        match self {
            Storage::Cpu(s) => Arc::strong_count(&s.inner),
        }
    }

    /// Read-only access to the raw bytes. Returns `&[]` when
    /// `byte_len() == 0`.
    pub fn as_bytes(&self) -> &[u8] {
        match self {
            Storage::Cpu(s) => s.as_bytes(),
        }
    }

    /// Get a mutable slice if and only if `is_unique()` returns `true`.
    /// Returns `None` if any clone exists (caller must `to_owned` /
    /// COW).
    pub fn as_bytes_mut(&mut self) -> Option<&mut [u8]> {
        match self {
            Storage::Cpu(s) => s.as_bytes_mut(),
        }
    }

    /// Reinterpret the bytes as a slice of `T`.
    ///
    /// # Safety
    ///
    /// `T` must match the dtype the buffer was allocated for, and the
    /// byte length must be divisible by `size_of::<T>()`.
    pub unsafe fn as_slice<T: Copy + 'static>(&self) -> &[T] {
        let bytes = self.as_bytes();
        let n = bytes.len() / core::mem::size_of::<T>();
        // SAFETY: caller asserted that the bytes are a valid `[T]`.
        slice::from_raw_parts(bytes.as_ptr() as *const T, n)
    }
}

// --------------------------------------------------------------------------
// CpuStorage — Arc<CpuStorageInner>
// --------------------------------------------------------------------------

/// CPU buffer with `Arc`-cheap clones.
#[derive(Debug, Clone)]
pub struct CpuStorage {
    inner: Arc<CpuStorageInner>,
    /// Cached for fast access — same as `inner.byte_len`.
    byte_len: usize,
}

impl CpuStorage {
    fn zeroed(byte_len: usize) -> Result<CpuStorage, StorageError> {
        if byte_len == 0 {
            return Ok(empty_cpu());
        }
        let layout = alloc_layout(byte_len)?;
        // SAFETY: layout is non-zero because byte_len > 0.
        let ptr = unsafe { alloc::alloc_zeroed(layout) };
        let ptr = NonNull::new(ptr).ok_or(StorageError::OutOfMemory { byte_len })?;
        Ok(CpuStorage::wrap(ptr, layout, byte_len))
    }

    unsafe fn uninit(byte_len: usize) -> Result<CpuStorage, StorageError> {
        if byte_len == 0 {
            return Ok(empty_cpu());
        }
        let layout = alloc_layout(byte_len)?;
        // SAFETY: layout is non-zero because byte_len > 0.
        let ptr = unsafe { alloc::alloc(layout) };
        let ptr = NonNull::new(ptr).ok_or(StorageError::OutOfMemory { byte_len })?;
        Ok(CpuStorage::wrap(ptr, layout, byte_len))
    }

    fn from_bytes(bytes: Vec<u8>) -> Result<CpuStorage, StorageError> {
        let n = bytes.len();
        if n == 0 {
            return Ok(empty_cpu());
        }
        // Always reallocate to guarantee CPU_ALIGN alignment — Vec<u8>
        // gives 1-byte alignment.
        // SAFETY: we initialise the full byte_len from `bytes`.
        let mut s = unsafe { CpuStorage::uninit(n)? };
        // Replace inner via Arc::get_mut: only one owner exists.
        {
            let inner = Arc::get_mut(&mut s.inner)
                .expect("freshly allocated CpuStorage has unique Arc owner");
            // SAFETY: inner.ptr points to byte_len uninitialised bytes;
            // we copy `n == byte_len` bytes from `bytes`.
            unsafe {
                core::ptr::copy_nonoverlapping(bytes.as_ptr(), inner.ptr.as_ptr(), n);
            }
        }
        Ok(s)
    }

    fn wrap(ptr: NonNull<u8>, layout: AllocLayout, byte_len: usize) -> CpuStorage {
        CpuStorage {
            inner: Arc::new(CpuStorageInner {
                ptr,
                layout,
                byte_len,
            }),
            byte_len,
        }
    }

    fn as_bytes(&self) -> &[u8] {
        if self.byte_len == 0 {
            return &[];
        }
        // SAFETY: inner.ptr points to byte_len initialised bytes (by
        // construction of every CpuStorage::* constructor).
        unsafe { slice::from_raw_parts(self.inner.ptr.as_ptr(), self.byte_len) }
    }

    fn as_bytes_mut(&mut self) -> Option<&mut [u8]> {
        if self.byte_len == 0 {
            return Some(&mut []);
        }
        let inner = Arc::get_mut(&mut self.inner)?;
        // SAFETY: unique Arc owner via get_mut, byte_len bytes init.
        Some(unsafe { slice::from_raw_parts_mut(inner.ptr.as_ptr(), inner.byte_len) })
    }
}

/// The single owner of an aligned heap allocation. Drops via the
/// stored `AllocLayout`.
pub(crate) struct CpuStorageInner {
    pub(crate) ptr: NonNull<u8>,
    pub(crate) layout: AllocLayout,
    pub(crate) byte_len: usize,
}

// SAFETY: `CpuStorageInner` does not provide thread-local handles;
// the buffer is plain bytes. Send and Sync mirror Arc<CpuStorageInner>.
unsafe impl Send for CpuStorageInner {}
unsafe impl Sync for CpuStorageInner {}

impl fmt::Debug for CpuStorageInner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CpuStorageInner")
            .field("ptr", &self.ptr.as_ptr())
            .field("byte_len", &self.byte_len)
            .field("align", &self.layout.align())
            .finish()
    }
}

impl Drop for CpuStorageInner {
    fn drop(&mut self) {
        if self.byte_len > 0 {
            // SAFETY: ptr/layout are the ones returned from `alloc_*`
            // with the same `AllocLayout`.
            unsafe { alloc::dealloc(self.ptr.as_ptr(), self.layout) };
        }
    }
}

// --------------------------------------------------------------------------
// Helpers
// --------------------------------------------------------------------------

fn alloc_layout(byte_len: usize) -> Result<AllocLayout, StorageError> {
    AllocLayout::from_size_align(byte_len, CPU_ALIGN).map_err(|_| StorageError::InvalidAlloc {
        byte_len,
        align: CPU_ALIGN,
    })
}

fn empty_cpu() -> CpuStorage {
    // Sentinel pointer for empty buffers; never dereferenced because
    // `byte_len == 0` is checked everywhere before raw access.
    let ptr = NonNull::<u8>::dangling();
    CpuStorage {
        inner: Arc::new(CpuStorageInner {
            ptr,
            // Use a 1-byte layout — never deallocated (byte_len == 0
            // skips the dealloc branch in Drop).
            layout: AllocLayout::from_size_align(1, 1).expect("trivial layout"),
            byte_len: 0,
        }),
        byte_len: 0,
    }
}

// --------------------------------------------------------------------------
// Errors
// --------------------------------------------------------------------------

/// Errors returned by [`Storage`] constructors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StorageError {
    /// `byte_len` exceeded `isize::MAX` or alignment is invalid.
    InvalidAlloc {
        /// Bytes requested.
        byte_len: usize,
        /// Alignment requested (always [`CPU_ALIGN`] in v1).
        align: usize,
    },
    /// The system allocator returned null for the requested size.
    OutOfMemory {
        /// Bytes requested.
        byte_len: usize,
    },
}

impl fmt::Display for StorageError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StorageError::InvalidAlloc { byte_len, align } => {
                write!(
                    f,
                    "invalid alloc layout: byte_len={byte_len}, align={align} \
                     (must satisfy byte_len <= isize::MAX and align is power-of-two)"
                )
            },
            StorageError::OutOfMemory { byte_len } => {
                write!(f, "out of memory allocating {byte_len} bytes")
            },
        }
    }
}

impl std::error::Error for StorageError {}

// --------------------------------------------------------------------------
// Tests
// --------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_zeroed_alignment_is_64() {
        let s = Storage::cpu_zeroed(1024).unwrap();
        let bytes = s.as_bytes();
        let addr = bytes.as_ptr() as usize;
        assert_eq!(addr % CPU_ALIGN, 0, "alignment violation: addr={addr:x}");
        assert_eq!(bytes.len(), 1024);
        // Zeroed
        assert!(bytes.iter().all(|&b| b == 0));
    }

    #[test]
    fn cpu_zeroed_byte_len_zero_yields_empty_sentinel() {
        let s = Storage::cpu_zeroed(0).unwrap();
        assert_eq!(s.byte_len(), 0);
        assert!(s.as_bytes().is_empty());
    }

    #[test]
    fn cpu_uninit_returns_writable_buffer() {
        // SAFETY: we write every byte before reading.
        let mut s = unsafe { Storage::cpu_uninit(128) }.unwrap();
        let bytes = s.as_bytes_mut().unwrap();
        for (i, b) in bytes.iter_mut().enumerate() {
            *b = (i & 0xff) as u8;
        }
        let view = s.as_bytes();
        for (i, &b) in view.iter().enumerate() {
            assert_eq!(b, (i & 0xff) as u8);
        }
    }

    #[test]
    fn cpu_from_bytes_round_trip() {
        let original: Vec<u8> = (0..200).map(|i| (i & 0xff) as u8).collect();
        let s = Storage::cpu_from_bytes(original.clone()).unwrap();
        assert_eq!(s.byte_len(), 200);
        assert_eq!(s.as_bytes(), original.as_slice());
        let addr = s.as_bytes().as_ptr() as usize;
        assert_eq!(addr % CPU_ALIGN, 0, "from_bytes must re-align");
    }

    #[test]
    fn clone_shares_buffer_ptr() {
        let s = Storage::cpu_zeroed(64).unwrap();
        let view = s.clone();
        assert_eq!(s.as_bytes().as_ptr(), view.as_bytes().as_ptr());
        assert_eq!(s.strong_count(), 2);
        assert_eq!(view.strong_count(), 2);
    }

    #[test]
    fn drop_decrements_refcount() {
        let s = Storage::cpu_zeroed(64).unwrap();
        assert_eq!(s.strong_count(), 1);
        let view = s.clone();
        assert_eq!(s.strong_count(), 2);
        drop(view);
        assert_eq!(s.strong_count(), 1);
    }

    #[test]
    fn is_unique_flips_with_clones() {
        let s = Storage::cpu_zeroed(64).unwrap();
        assert!(s.is_unique());
        let view = s.clone();
        assert!(!s.is_unique());
        assert!(!view.is_unique());
        drop(view);
        assert!(s.is_unique());
    }

    #[test]
    fn as_bytes_mut_returns_none_when_shared() {
        let mut s = Storage::cpu_zeroed(32).unwrap();
        // Unique → Some
        assert!(s.as_bytes_mut().is_some());
        let _view = s.clone();
        // Shared → None (caller must COW)
        assert!(s.as_bytes_mut().is_none());
    }

    #[test]
    fn as_bytes_mut_empty_buffer_is_some() {
        let mut s = Storage::cpu_zeroed(0).unwrap();
        assert!(s.as_bytes_mut().is_some());
        assert!(s.as_bytes_mut().unwrap().is_empty());
    }

    #[test]
    fn slice_reinterpret_as_f32() {
        let bytes: Vec<u8> = bytemuck::cast_slice(&[1.0_f32, 2.0, 3.0, 4.0]).to_vec();
        let s = Storage::cpu_from_bytes(bytes).unwrap();
        // SAFETY: we know the buffer is a packed [f32; 4].
        let slice: &[f32] = unsafe { s.as_slice::<f32>() };
        assert_eq!(slice, &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn stress_alloc_free_no_leak() {
        // Quick stress: 10k alloc/free cycles. Validates Drop runs and
        // strong_count returns to 1 on each iteration.
        for _ in 0..10_000 {
            let s = Storage::cpu_zeroed(4096).unwrap();
            assert_eq!(s.strong_count(), 1);
        }
    }

    #[test]
    fn send_sync_smoke() {
        // Compile-time check that Storage is Send + Sync.
        fn assert_send<T: Send>() {}
        fn assert_sync<T: Sync>() {}
        assert_send::<Storage>();
        assert_sync::<Storage>();
    }

    #[test]
    fn out_of_memory_for_huge_alloc_returns_error() {
        // Size deliberately above what any system can satisfy: this
        // should either be rejected at the layout-construction step
        // (InvalidAlloc) or by the allocator (OutOfMemory). Either way
        // we expect an Err, not a panic.
        let huge = isize::MAX as usize / 2 + 1;
        let res = Storage::cpu_zeroed(huge);
        assert!(res.is_err());
    }

    #[test]
    fn invalid_alloc_for_isize_max_returns_error() {
        // Layout::from_size_align rejects byte_len > isize::MAX.
        let bad = (isize::MAX as usize).wrapping_add(1);
        let res = Storage::cpu_zeroed(bad);
        match res {
            Err(StorageError::InvalidAlloc { byte_len, align }) => {
                assert_eq!(byte_len, bad);
                assert_eq!(align, CPU_ALIGN);
            },
            other => panic!("expected InvalidAlloc, got {other:?}"),
        }
    }

    #[test]
    fn debug_format_does_not_panic() {
        let s = Storage::cpu_zeroed(16).unwrap();
        let _ = format!("{s:?}");
    }

    #[test]
    fn storage_error_display() {
        let e = StorageError::OutOfMemory { byte_len: 1024 };
        assert!(e.to_string().contains("1024"));
        let e = StorageError::InvalidAlloc {
            byte_len: 4,
            align: 64,
        };
        assert!(e.to_string().contains("byte_len=4"));
        assert!(e.to_string().contains("align=64"));
    }
}
