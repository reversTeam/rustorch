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

    /// WGSL shader failed to compile. The diagnostic carries the
    /// driver / naga error message verbatim so callers can surface it
    /// to logs / users without losing source-line info.
    #[error("WGSL shader compile error in `{label}`: {message}")]
    ShaderCompile {
        /// Human-readable kernel label (e.g. `"matmul"`, `"softmax"`).
        label: String,
        /// Raw error text from the validator / driver.
        message: String,
    },

    /// Caller asked for a zero-byte allocation. wgpu requires every
    /// buffer to have non-zero size; rustorch surfaces this as a
    /// distinct variant rather than silently padding to 4 bytes so
    /// the upstream bug is visible.
    #[error("zero-size allocation request ({context})")]
    ZeroSize {
        /// Where the zero-size request originated (op name / label).
        context: String,
    },

    /// The wgpu device entered the lost state — typically because the
    /// driver crashed or a long-running compute hit a watchdog timer.
    /// Recovery requires re-creating the [`crate::WgpuBackend`] from
    /// scratch.
    #[error("wgpu device lost: {reason}")]
    DeviceLost {
        /// Reason reported by the driver, if any.
        reason: String,
    },

    /// A wgpu validation error was captured (uncaptured-error scope).
    /// Means the runtime spotted a bind-group / shader / encoder
    /// problem after the dispatch was already submitted; usually a
    /// rustorch bug, surface to the user.
    #[error("wgpu validation error: {0}")]
    Validation(String),
}
