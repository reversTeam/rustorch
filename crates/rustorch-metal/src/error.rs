//! Metal backend errors.

use thiserror::Error;

/// Errors surfaced by the Metal backend.
#[derive(Debug, Error)]
pub enum MetalError {
    /// No Metal device available (running outside macOS, or in a
    /// container without GPU access).
    #[error("no Metal device available: {0}")]
    NoDevice(String),

    /// Failed to compile a `.metal` shader source.
    #[error("shader compile failed: {0}")]
    ShaderCompile(String),

    /// Failed to create a compute pipeline state from a function.
    #[error("pipeline state creation failed: {0}")]
    PipelineState(String),

    /// Allocation failure (out of GPU memory).
    #[error("Metal allocation failed: {requested} bytes ({reason})")]
    AllocFailed {
        /// Bytes requested.
        requested: usize,
        /// Driver-reported reason.
        reason: String,
    },

    /// Shape / dtype mismatch surfaced from a kernel call.
    #[error("Metal shape mismatch: {0}")]
    ShapeMismatch(String),

    /// Catch-all for unsupported operations during scaffolding.
    #[error("unsupported operation: {0}")]
    Unsupported(String),
}
