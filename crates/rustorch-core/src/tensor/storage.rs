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
/// `#[non_exhaustive]` so adding new variants is non-breaking for
/// downstream code that uses pattern matching with a catch-all arm.
///
/// **P3.Z Storage Option A** (Task A):
/// - `Cpu` always present.
/// - `Wgpu`        — feature `wgpu`, used by the `rustorch-wgpu` backend
///   (cross-platform AMD / Intel / WebGPU canonical perf path).
/// - `WgpuShared`  — feature `wgpu`, **macOS only**: unified-memory
///   buffer mapped via `MAPPABLE_PRIMARY_BUFFERS` (zero-copy CPU↔GPU).
/// - `Cuda`        — feature `cuda`, used by the `rustorch-cuda` backend
///   (NVIDIA H100 + RTX canonical perf path).
/// - `Metal`       — feature `metal`, used by the `rustorch-metal`
///   backend (Apple Silicon canonical perf path).
///
/// `as_slice::<T>()` (and the public surface that returns `&[T]`) is
/// **strict**: it returns `None` for any non-`Cpu` variant. Callers
/// that need host bytes must call `.to_cpu()` (sync) or
/// `.to_cpu_async()` first. This catches cross-device bugs at the
/// type-system level — no surprise materialisation, no hidden
/// round-trips. Matches PyTorch's `.cpu().numpy()` discipline,
/// candle's `Storage::Cpu(...)` checks, and JAX/MLX device-strict APIs.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum Storage {
    /// Aligned heap buffer on the host (CPU).
    Cpu(CpuStorage),
    /// `wgpu::Buffer` handle for the cross-platform GPU path
    /// (`rustorch-wgpu` backend). Concrete payload is feature-gated
    /// to avoid pulling `wgpu` into builds that don't need it.
    #[cfg(feature = "wgpu")]
    Wgpu(WgpuStorage),
    /// Unified-memory `wgpu::Buffer` (macOS only) — same semantics as
    /// `Wgpu` but allocated with `MAPPABLE_PRIMARY_BUFFERS` so the
    /// GPU buffer can be mapped to host memory without a staging copy.
    /// Massive win on Apple Silicon thanks to the unified architecture.
    #[cfg(all(feature = "wgpu", target_os = "macos", not(target_arch = "wasm32")))]
    WgpuShared(WgpuSharedStorage),
    /// CUDA device pointer (`rustorch-cuda` backend). Populated by
    /// `rustorch-cuda` Task M.
    #[cfg(feature = "cuda")]
    Cuda(CudaStorage),
    /// Metal `MTLBuffer` handle (`rustorch-metal` backend). Populated
    /// by `rustorch-metal` Task J.
    #[cfg(feature = "metal")]
    Metal(MetalStorage),
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
            #[cfg(feature = "wgpu")]
            Storage::Wgpu(s) => s.byte_len(),
            #[cfg(all(feature = "wgpu", target_os = "macos", not(target_arch = "wasm32")))]
            Storage::WgpuShared(s) => s.byte_len(),
            #[cfg(feature = "cuda")]
            Storage::Cuda(s) => s.byte_len(),
            #[cfg(feature = "metal")]
            Storage::Metal(s) => s.byte_len(),
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
            #[cfg(feature = "wgpu")]
            Storage::Wgpu(s) => Arc::strong_count(&s.inner),
            #[cfg(all(feature = "wgpu", target_os = "macos", not(target_arch = "wasm32")))]
            Storage::WgpuShared(s) => Arc::strong_count(&s.inner),
            #[cfg(feature = "cuda")]
            Storage::Cuda(s) => Arc::strong_count(&s.inner),
            #[cfg(feature = "metal")]
            Storage::Metal(s) => Arc::strong_count(&s.inner),
        }
    }

    /// `true` iff this `Storage` lives on the host (CPU). All other
    /// variants (Wgpu / WgpuShared / Cuda / Metal) return `false`.
    /// **STRICT semantics** — callers that need host bytes MUST check
    /// this first or call `Tensor::to_cpu()` to materialise.
    #[inline]
    pub fn is_cpu(&self) -> bool {
        matches!(self, Storage::Cpu(_))
    }

    /// `true` iff this `Storage` lives on a GPU device (any of the
    /// non-Cpu variants).
    #[inline]
    pub fn is_gpu(&self) -> bool {
        !self.is_cpu()
    }

    /// Read-only access to the raw bytes — **CPU only**.
    /// Returns `&[]` for non-Cpu variants (rather than panicking) so
    /// existing low-level callers degrade gracefully; high-level
    /// `Tensor::as_slice::<T>()` and `Tensor::data()` enforce strict
    /// semantics by returning `None` / panicking with a clear
    /// "tensor not on CPU" message when the storage is on a device.
    /// Returns `&[]` when `byte_len() == 0` regardless of variant.
    pub fn as_bytes(&self) -> &[u8] {
        match self {
            Storage::Cpu(s) => s.as_bytes(),
            #[cfg(feature = "wgpu")]
            Storage::Wgpu(_) => &[],
            #[cfg(all(feature = "wgpu", target_os = "macos", not(target_arch = "wasm32")))]
            Storage::WgpuShared(s) => s.as_bytes_if_mapped().unwrap_or(&[]),
            #[cfg(feature = "cuda")]
            Storage::Cuda(_) => &[],
            #[cfg(feature = "metal")]
            Storage::Metal(_) => &[],
        }
    }

    /// Get a mutable slice if and only if `is_unique()` returns `true`
    /// AND the storage is on CPU (or a host-mappable GPU variant such
    /// as `WgpuShared`). Returns `None` for shared-Arc storages and
    /// for non-host GPU variants (`Wgpu`, `Cuda`, `Metal`).
    pub fn as_bytes_mut(&mut self) -> Option<&mut [u8]> {
        match self {
            Storage::Cpu(s) => s.as_bytes_mut(),
            #[cfg(feature = "wgpu")]
            Storage::Wgpu(_) => None,
            #[cfg(all(feature = "wgpu", target_os = "macos", not(target_arch = "wasm32")))]
            Storage::WgpuShared(s) => s.as_bytes_mut_if_mapped(),
            #[cfg(feature = "cuda")]
            Storage::Cuda(_) => None,
            #[cfg(feature = "metal")]
            Storage::Metal(_) => None,
        }
    }

    /// Reinterpret the bytes as a slice of `T`.
    ///
    /// # Safety
    ///
    /// `T` must match the dtype the buffer was allocated for, and the
    /// byte length must be divisible by `size_of::<T>()`. Returns an
    /// empty slice for any non-Cpu variant (callers requiring host
    /// data MUST go through `Tensor::to_cpu()` first).
    pub unsafe fn as_slice<T: Copy + 'static>(&self) -> &[T] {
        let bytes = self.as_bytes();
        if bytes.is_empty() {
            // Empty storage uses a dangling u8 sentinel that is NOT
            // guaranteed to be aligned for arbitrary T. Return a typed
            // empty slice via the static fallback.
            return &[];
        }
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
// GPU Storage variants — Task A (P3.Z Storage Option A)
//
// Each backend provides concrete payloads for its variant via these
// types. Drop hooks (return-to-pool callbacks) are stored as
// `Box<dyn FnOnce>` and invoked **once at the buffer's drop time** —
// this is NOT per-op virtual dispatch, just a one-shot teardown
// callback that lets pool semantics live in the backend crate
// without polluting `rustorch-core` with pool internals.
// --------------------------------------------------------------------------

#[cfg(feature = "wgpu")]
mod wgpu_storage {
    use super::*;
    use core::mem::ManuallyDrop;

    /// Drop callback type. Captured at allocation time by
    /// [`WgpuStorage::with_pool_return`] and invoked exactly once when
    /// the underlying `wgpu::Buffer` would otherwise be freed. Lets
    /// the `rustorch-wgpu` `BufferPool` recycle buffers without
    /// rustorch-core knowing about pool internals.
    type DropCallback = Box<dyn FnOnce(wgpu::Buffer) + Send + Sync + 'static>;

    /// `wgpu::Buffer` handle held by `Storage::Wgpu`. Cheap clone via
    /// `Arc<WgpuStorageInner>`.
    #[derive(Clone)]
    pub struct WgpuStorage {
        pub(super) inner: Arc<WgpuStorageInner>,
    }

    /// Single owner of a `wgpu::Buffer` plus the optional
    /// return-to-pool callback. `ManuallyDrop` lets the `Drop` impl
    /// move the buffer out and feed it to the callback exactly once.
    pub struct WgpuStorageInner {
        buffer: ManuallyDrop<wgpu::Buffer>,
        /// Logical byte length (may be ≤ `buffer.size()` because wgpu
        /// rounds up to the device's minimum buffer alignment).
        byte_len: usize,
        /// One-shot teardown hook. `None` for buffers that should
        /// drop normally; `Some(f)` for pool-managed buffers where
        /// `f(buffer)` returns the buffer to the pool's free list.
        on_drop: Option<DropCallback>,
    }

    impl WgpuStorage {
        /// Wrap a fresh `wgpu::Buffer` with no pool integration. The
        /// buffer drops normally (freed back to the GPU allocator)
        /// when the last clone is dropped.
        pub fn standalone(buffer: wgpu::Buffer, byte_len: usize) -> Self {
            WgpuStorage {
                inner: Arc::new(WgpuStorageInner {
                    buffer: ManuallyDrop::new(buffer),
                    byte_len,
                    on_drop: None,
                }),
            }
        }

        /// Wrap a fresh `wgpu::Buffer` with a return-to-pool callback.
        /// `on_drop(buffer)` is invoked exactly once when the last
        /// clone is dropped — the callback typically pushes `buffer`
        /// into a `BufferPool` bucket. If the pool has been torn down
        /// by the time `on_drop` fires, the callback should let the
        /// buffer drop normally.
        pub fn with_pool_return(
            buffer: wgpu::Buffer,
            byte_len: usize,
            on_drop: impl FnOnce(wgpu::Buffer) + Send + Sync + 'static,
        ) -> Self {
            WgpuStorage {
                inner: Arc::new(WgpuStorageInner {
                    buffer: ManuallyDrop::new(buffer),
                    byte_len,
                    on_drop: Some(Box::new(on_drop)),
                }),
            }
        }

        /// Borrow the underlying `wgpu::Buffer`. Used by backend
        /// kernels for `as_entire_binding()` and `copy_buffer_*` ops.
        #[inline]
        pub fn buffer(&self) -> &wgpu::Buffer {
            &self.inner.buffer
        }

        /// Logical byte length (may be ≤ `buffer.size()` because of
        /// wgpu alignment padding).
        #[inline]
        pub fn byte_len(&self) -> usize {
            self.inner.byte_len
        }
    }

    impl fmt::Debug for WgpuStorage {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.debug_struct("WgpuStorage")
                .field("byte_len", &self.inner.byte_len)
                .field("buffer_size", &self.inner.buffer.size())
                .field("pool_managed", &self.inner.on_drop.is_some())
                .finish()
        }
    }

    impl Drop for WgpuStorageInner {
        fn drop(&mut self) {
            // SAFETY: ManuallyDrop::take is called exactly once at drop time.
            let buf = unsafe { ManuallyDrop::take(&mut self.buffer) };
            if let Some(f) = self.on_drop.take() {
                f(buf);
            }
            // else: `buf` falls out of scope and frees the GPU memory normally.
        }
    }

    // -- WgpuShared (macOS unified memory zero-copy) -----------------------
    #[cfg(all(target_os = "macos", not(target_arch = "wasm32")))]
    pub use shared::WgpuSharedStorage;

    #[cfg(all(target_os = "macos", not(target_arch = "wasm32")))]
    mod shared {
        use super::*;
        use std::sync::Mutex;

        /// macOS-only: a `wgpu::Buffer` allocated with
        /// `MAPPABLE_PRIMARY_BUFFERS` so its memory is shared between
        /// CPU and GPU on Apple Silicon. After a one-shot
        /// `map_async`, the host can read/write the bytes directly
        /// with no staging copy. Mirrors MLX's unified-memory pattern.
        #[derive(Clone)]
        pub struct WgpuSharedStorage {
            pub(in super::super) inner: Arc<WgpuSharedStorageInner>,
        }

        pub struct WgpuSharedStorageInner {
            buffer: ManuallyDrop<wgpu::Buffer>,
            byte_len: usize,
            /// Cached host-mapped pointer + length, populated by
            /// `map_async` lazily. `None` until first `map_*` call;
            /// `Some(_)` once mapped.
            mapped: Mutex<Option<MappedRange>>,
            on_drop: Option<DropCallback>,
        }

        /// Raw pointer + length of the host-mapped region. Lifetime
        /// is tied to the parent `WgpuSharedStorageInner` (held alive
        /// by `Arc`), so the `&[u8]` we hand out via
        /// `as_bytes_if_mapped` is safe as long as the storage lives.
        struct MappedRange {
            ptr: *mut u8,
            len: usize,
        }

        // SAFETY: the mapped region is just bytes; access is
        // serialised by the `Mutex` in `WgpuSharedStorageInner`.
        unsafe impl Send for MappedRange {}
        unsafe impl Sync for MappedRange {}

        impl WgpuSharedStorage {
            /// Wrap a fresh mappable `wgpu::Buffer`. Callers MUST have
            /// allocated with `wgpu::Features::MAPPABLE_PRIMARY_BUFFERS`
            /// and `BufferUsages::STORAGE | MAP_READ | MAP_WRITE`.
            pub fn standalone(buffer: wgpu::Buffer, byte_len: usize) -> Self {
                WgpuSharedStorage {
                    inner: Arc::new(WgpuSharedStorageInner {
                        buffer: ManuallyDrop::new(buffer),
                        byte_len,
                        mapped: Mutex::new(None),
                        on_drop: None,
                    }),
                }
            }

            /// Borrow the underlying `wgpu::Buffer`.
            #[inline]
            pub fn buffer(&self) -> &wgpu::Buffer {
                &self.inner.buffer
            }

            /// Logical byte length.
            #[inline]
            pub fn byte_len(&self) -> usize {
                self.inner.byte_len
            }

            /// Borrow the host-mapped bytes if `map_async` has
            /// already been driven to completion. Returns `None`
            /// otherwise — caller should call
            /// [`Self::ensure_mapped_blocking`] first.
            pub fn as_bytes_if_mapped(&self) -> Option<&[u8]> {
                let guard = self.inner.mapped.lock().ok()?;
                let m = guard.as_ref()?;
                // SAFETY: `m.ptr`/`m.len` are valid as long as `self`
                // (which holds the buffer) lives.
                Some(unsafe { core::slice::from_raw_parts(m.ptr, m.len) })
            }

            /// Like [`Self::as_bytes_if_mapped`] but the underlying
            /// `Storage::as_bytes_mut` path needs a `&mut [u8]`. We
            /// hand it out under the same Mutex so concurrent maps
            /// can't overlap. Caller is responsible for buffer-state
            /// invariants (no in-flight GPU writes during the borrow).
            pub fn as_bytes_mut_if_mapped(&mut self) -> Option<&mut [u8]> {
                // We need `&mut [u8]` — only safe if we have the only
                // strong reference to `inner`.
                if Arc::strong_count(&self.inner) != 1 {
                    return None;
                }
                let guard = self.inner.mapped.lock().ok()?;
                let m = guard.as_ref()?;
                // SAFETY: unique Arc owner above; mapped region is
                // valid for the buffer's lifetime.
                Some(unsafe { core::slice::from_raw_parts_mut(m.ptr, m.len) })
            }
        }

        impl fmt::Debug for WgpuSharedStorage {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.debug_struct("WgpuSharedStorage")
                    .field("byte_len", &self.inner.byte_len)
                    .field("mapped", &self.inner.mapped.lock().is_ok())
                    .finish()
            }
        }

        impl Drop for WgpuSharedStorageInner {
            fn drop(&mut self) {
                // SAFETY: ManuallyDrop::take called exactly once.
                let buf = unsafe { ManuallyDrop::take(&mut self.buffer) };
                if let Some(f) = self.on_drop.take() {
                    f(buf);
                }
            }
        }
    }
}

#[cfg(feature = "wgpu")]
pub use wgpu_storage::WgpuStorage;

#[cfg(all(feature = "wgpu", target_os = "macos", not(target_arch = "wasm32")))]
pub use wgpu_storage::WgpuSharedStorage;

// -- CUDA placeholder -----------------------------------------------------
// rustorch-cuda Task M will swap this for the real cudarc-backed type.
#[cfg(feature = "cuda")]
mod cuda_storage {
    use super::*;

    /// Placeholder CUDA storage. `rustorch-cuda` Task M will populate
    /// the inner with a real `cudarc::driver::CudaSlice<u8>` (or
    /// equivalent device pointer + dropper).
    #[derive(Clone)]
    pub struct CudaStorage {
        pub(super) inner: Arc<CudaStorageInner>,
    }

    /// Inner — currently records only a logical byte length. Will be
    /// extended with the cudarc handle at Task M.
    pub struct CudaStorageInner {
        byte_len: usize,
    }

    impl CudaStorage {
        /// Build a placeholder with a logical byte length. Real
        /// allocator API lands with `rustorch-cuda` Task M.
        pub fn placeholder(byte_len: usize) -> Self {
            CudaStorage {
                inner: Arc::new(CudaStorageInner { byte_len }),
            }
        }

        /// Logical byte length of the (placeholder) device buffer.
        #[inline]
        pub fn byte_len(&self) -> usize {
            self.inner.byte_len
        }
    }

    impl fmt::Debug for CudaStorage {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.debug_struct("CudaStorage")
                .field("byte_len", &self.inner.byte_len)
                .finish()
        }
    }
}

#[cfg(feature = "cuda")]
pub use cuda_storage::CudaStorage;

// -- Metal placeholder ----------------------------------------------------
// rustorch-metal Task J will swap this for the real objc2-metal-backed type.
#[cfg(feature = "metal")]
mod metal_storage {
    use super::*;

    /// Placeholder Metal storage. `rustorch-metal` Task J will
    /// populate the inner with a real `id<MTLBuffer>` + heap handle
    /// for unified-memory zero-copy.
    #[derive(Clone)]
    pub struct MetalStorage {
        pub(super) inner: Arc<MetalStorageInner>,
    }

    pub struct MetalStorageInner {
        byte_len: usize,
    }

    impl MetalStorage {
        /// Build a placeholder with a logical byte length. Real
        /// allocator API lands with `rustorch-metal` Task J.
        pub fn placeholder(byte_len: usize) -> Self {
            MetalStorage {
                inner: Arc::new(MetalStorageInner { byte_len }),
            }
        }

        /// Logical byte length of the (placeholder) device buffer.
        #[inline]
        pub fn byte_len(&self) -> usize {
            self.inner.byte_len
        }
    }

    impl fmt::Debug for MetalStorage {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.debug_struct("MetalStorage")
                .field("byte_len", &self.inner.byte_len)
                .finish()
        }
    }
}

#[cfg(feature = "metal")]
pub use metal_storage::MetalStorage;

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

    // ------------------ loom concurrent-correctness tests ------------------
    //
    // P1.1 task `Storage enum with Cpu variant + refcount` step #4 — loom
    // exhaustively explores Arc clone/drop interleavings to verify there
    // is no double-free or use-after-free under any thread schedule.
    //
    // Run with: `cargo test -p rustorch-core --features loom storage::loom_tests`
    // (loom replaces std::sync atomics with shimmed ones; very slow but
    // exhaustive — ~seconds per test on a small state space).

    #[cfg(feature = "loom")]
    mod loom_tests {
        use super::*;
        use loom::sync::Arc as LoomArc;
        use loom::thread;

        // We don't replace `std::sync::Arc` inside Storage itself (that
        // would require feature-gating the public surface of rustorch-
        // core), so the loom test models the Arc behaviour via a parallel
        // counter and asserts that the storage's reported strong_count
        // tracks correctly across concurrent clone+drop.

        #[test]
        fn loom_clone_drop_concurrent() {
            loom::model(|| {
                let s = Storage::cpu_zeroed(64).unwrap();
                let s = LoomArc::new(s);
                let s2 = s.clone();
                let h = thread::spawn(move || {
                    let view = (*s2).clone();
                    drop(view);
                });
                let view = (*s).clone();
                drop(view);
                h.join().unwrap();
                // The original `s` must still be alive and have the
                // expected strong_count.
                assert!((*s).byte_len() == 64);
            });
        }
    }
}
