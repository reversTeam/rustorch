//! Error types for the wgpu backend.

use rustorch_core::tensor::dtype::Dtype;

/// All errors raised by `rustorch-wgpu`.
#[derive(Debug, thiserror::Error)]
pub enum WgpuError {
    /// No suitable wgpu adapter could be found on this machine.
    #[error("no wgpu adapter available (no GPU? unsupported backend?)")]
    NoAdapter,

    /// Adapter device-creation failed.
    #[error("device request failed: {0}")]
    DeviceRequest(String),

    /// A buffer-mapping operation failed (read-back, etc.).
    #[error("buffer map failed: {0}")]
    MapFailure(String),

    /// Dtype is not yet supported on the wgpu path.
    #[error("dtype {0:?} not yet supported by wgpu backend")]
    UnsupportedDtype(Dtype),

    /// Generic shape error during a wgpu operation.
    #[error("shape mismatch: {0}")]
    ShapeMismatch(String),

    /// GPU is out of memory and even after evicting the pool the
    /// allocation could not be satisfied.
    ///
    /// The string carries a one-line snapshot of the pool's metrics at
    /// the moment of failure (bytes pooled, bytes allocated cumulative)
    /// so users can wire it into their telemetry without depending on
    /// the [`crate::cache::PoolMetricsSnapshot`] type directly.
    #[error("GPU out of memory: requested {requested} bytes, {snapshot}")]
    OutOfMemory {
        /// Bytes requested by the failing allocation.
        requested: u64,
        /// One-line snapshot of pool metrics at failure time.
        snapshot: String,
    },
}
