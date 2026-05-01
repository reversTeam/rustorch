//! Aligned CPU allocator (P1.2 task `Aligned allocator`).
//!
//! The actual aligned allocator lives in `rustorch-core::tensor::storage`
//! (since `Storage` is the type that owns buffers). This module re-
//! exports a small typed shim so callers in `rustorch-cpu` can name a
//! `CpuAllocator` for their own purposes (e.g. arenas in future
//! kernel work).
//!
//! ```
//! use rustorch_cpu::allocator::{CpuAllocator, CPU_ALIGN};
//!
//! let s = CpuAllocator::zeroed(1024).unwrap();
//! assert_eq!(s.byte_len(), 1024);
//! assert_eq!((s.as_bytes().as_ptr() as usize) % CPU_ALIGN, 0);
//! ```

pub use rustorch_core::tensor::storage::CPU_ALIGN;
use rustorch_core::tensor::storage::{Storage, StorageError};

/// Cache-line-aligned CPU buffer allocator. Thin wrapper over the
/// [`Storage`](rustorch_core::tensor::storage::Storage) constructors;
/// kept here so that future kernel-local arenas can hang off the same
/// abstraction.
pub struct CpuAllocator;

impl CpuAllocator {
    /// Allocate a zero-initialised aligned buffer of `byte_len` bytes.
    pub fn zeroed(byte_len: usize) -> Result<Storage, StorageError> {
        Storage::cpu_zeroed(byte_len)
    }

    /// Allocate an *uninitialised* aligned buffer.
    ///
    /// # Safety
    ///
    /// Caller is responsible for filling every byte before reading.
    pub unsafe fn uninit(byte_len: usize) -> Result<Storage, StorageError> {
        // SAFETY: forward the contract directly.
        unsafe { Storage::cpu_uninit(byte_len) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zeroed_returns_aligned_buffer() {
        let s = CpuAllocator::zeroed(2048).unwrap();
        assert_eq!(s.byte_len(), 2048);
        assert_eq!((s.as_bytes().as_ptr() as usize) % CPU_ALIGN, 0);
        assert!(s.as_bytes().iter().all(|&b| b == 0));
    }

    #[test]
    fn uninit_returns_writable_aligned_buffer() {
        // SAFETY: we write every byte before reading.
        let mut s = unsafe { CpuAllocator::uninit(64) }.unwrap();
        let bytes = s.as_bytes_mut().unwrap();
        for (i, b) in bytes.iter_mut().enumerate() {
            *b = (i & 0xff) as u8;
        }
        assert_eq!((s.as_bytes().as_ptr() as usize) % CPU_ALIGN, 0);
        for (i, &b) in s.as_bytes().iter().enumerate() {
            assert_eq!(b, (i & 0xff) as u8);
        }
    }

    #[test]
    fn zero_byte_alloc_is_safe() {
        let s = CpuAllocator::zeroed(0).unwrap();
        assert_eq!(s.byte_len(), 0);
        assert!(s.as_bytes().is_empty());
    }

    #[test]
    fn cpu_align_is_64() {
        assert_eq!(CPU_ALIGN, 64);
    }
}
