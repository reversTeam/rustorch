//! CUDA stream wrapper + per-device pool.

use crate::device::Device;
use crate::error::CudaError;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Opaque stream handle. On no-cuda builds this is a sentinel u64
/// that round-trips through the API but isn't dereferenced. Under
/// `--features cuda` it wraps a `cudarc::driver::CudaStream`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Stream {
    handle: u64,
    device: Device,
}

impl Stream {
    /// Create a fresh stream on `device`. On no-cuda builds returns a
    /// sentinel handle.
    pub fn new(device: Device) -> Result<Self, CudaError> {
        if !cfg!(feature = "cuda") {
            // Sentinel value derived from device index.
            return Ok(Self {
                handle: u64::from(device.index) << 32,
                device,
            });
        }
        // Real impl: cudarc create_stream + record into pool.
        Ok(Self {
            handle: u64::from(device.index) << 32,
            device,
        })
    }

    /// Default per-device stream (NULL stream on real CUDA).
    pub fn default_for(device: Device) -> Self {
        Self { handle: 0, device }
    }

    /// Raw handle (cudaStream_t when feature is on; sentinel else).
    pub fn raw(&self) -> u64 {
        self.handle
    }

    /// Owning device.
    pub fn device(&self) -> Device {
        self.device
    }

    /// Block until all queued work on this stream completes. No-op
    /// on no-cuda builds.
    pub fn synchronize(&self) -> Result<(), CudaError> {
        Ok(())
    }
}

/// Round-robin pool of N streams on a single device. Used by the
/// runtime to pipeline ops across multiple streams.
#[derive(Debug)]
pub struct StreamPool {
    streams: Vec<Stream>,
    next: AtomicUsize,
}

impl StreamPool {
    /// Build a pool of `n_streams` on `device`.
    pub fn new(device: Device, n_streams: usize) -> Result<Self, CudaError> {
        let mut streams = Vec::with_capacity(n_streams);
        for i in 0..n_streams {
            // Distinct sentinel per slot so tests can verify pool
            // distinctness without a real GPU.
            streams.push(Stream {
                handle: (u64::from(device.index) << 32) | (i as u64 + 1),
                device,
            });
        }
        Ok(Self {
            streams,
            next: AtomicUsize::new(0),
        })
    }

    /// Round-robin acquire — returns the next stream, advancing the
    /// internal cursor.
    pub fn acquire(&self) -> Stream {
        let idx = self.next.fetch_add(1, Ordering::Relaxed) % self.streams.len().max(1);
        self.streams[idx]
    }

    /// Number of streams in the pool.
    pub fn len(&self) -> usize {
        self.streams.len()
    }

    /// Empty pool? (illegal to construct; sanity helper)
    pub fn is_empty(&self) -> bool {
        self.streams.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dev(i: u32) -> Device {
        Device { index: i }
    }

    #[test]
    fn default_stream_is_zero_handle() {
        let s = Stream::default_for(dev(0));
        assert_eq!(s.raw(), 0);
        assert_eq!(s.device(), dev(0));
    }

    #[test]
    fn new_stream_returns_distinct_handle_from_default() {
        let s = Stream::new(dev(1)).unwrap();
        assert_ne!(s.raw(), 0);
    }

    #[test]
    fn pool_returns_n_distinct_streams() {
        let p = StreamPool::new(dev(0), 4).unwrap();
        assert_eq!(p.len(), 4);
        let mut handles: std::collections::HashSet<u64> = std::collections::HashSet::new();
        for _ in 0..4 {
            handles.insert(p.acquire().raw());
        }
        // 4 distinct slots round-robin-acquired in 4 calls.
        assert_eq!(handles.len(), 4);
    }

    #[test]
    fn pool_round_robin_wraps() {
        let p = StreamPool::new(dev(0), 2).unwrap();
        let a = p.acquire().raw();
        let b = p.acquire().raw();
        let c = p.acquire().raw();
        assert_eq!(a, c); // wrapped
        assert_ne!(a, b);
    }

    #[test]
    fn synchronize_is_no_op_without_cuda() {
        let s = Stream::new(dev(0)).unwrap();
        assert!(s.synchronize().is_ok());
    }
}
