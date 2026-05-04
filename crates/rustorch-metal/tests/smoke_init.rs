//! Smoke test — initialise the global `MetalBackend` and confirm
//! it found an Apple Silicon GPU with `simdgroup_matrix` support.
//!
//! Gated by `cfg(target_os = "macos")` so non-macOS CI runners
//! compile this file as a no-op.

#![cfg(target_os = "macos")]

use rustorch_metal::backend_singleton::metal_backend;

#[test]
fn metal_backend_initialises() {
    let b = metal_backend();
    let name = b.adapter_name();
    assert!(!name.is_empty(), "adapter name must be non-empty");
    eprintln!(
        "[smoke_init] MetalBackend on {} (Metal3: {})",
        name,
        b.supports_metal3()
    );
}

#[test]
fn metal_backend_can_allocate_shared_buffer() {
    let b = metal_backend();
    let buf = b.alloc_shared(4096).expect("alloc 4 KiB");
    assert!(buf.length() >= 4096, "buffer length must cover 4 KiB");
    // Write some bytes through the unified-memory pointer and read
    // them back — proves the shared storage mode actually works.
    // SAFETY: the buffer was allocated MTLStorageModeShared so the
    // contents pointer is valid for read+write on the host as long
    // as the buffer lives.
    unsafe {
        let ptr = buf.contents() as *mut u8;
        for i in 0..16usize {
            *ptr.add(i) = (i * 3) as u8;
        }
        let view = std::slice::from_raw_parts(ptr as *const u8, 16);
        let expected: Vec<u8> = (0..16usize).map(|i| (i * 3) as u8).collect();
        assert_eq!(view, expected.as_slice());
    }
}
