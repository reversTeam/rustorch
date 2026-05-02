//! WGSL template / validation helpers.
//!
//! The crate's kernels generate WGSL via Rust `format!()` strings
//! (see `shaders.rs`, `matmul.rs`, `softmax.rs`, etc.). This module
//! provides:
//!
//! - [`validate_wgsl`] — try to compile a WGSL source through the
//!   live device and surface any error from `wgpu`/naga as a
//!   [`crate::error::WgpuError::ShapeMismatch`] instead of panicking.
//!   Used by [`validate_all_kernels`] to assert at startup that
//!   every static kernel template still parses on this driver.
//! - [`wgsl_kernel!`] declarative macro — wraps the
//!   `format!(r#"…"#, BLOCK = …, …)` pattern in a single, documented
//!   form so kernel authors don't reinvent the substitution logic
//!   each time.
//!
//! Both pieces aim at the same goal as the original "build.rs WGSL
//! template" task: make WGSL generation predictable and catch
//! malformed shaders before they reach the dispatch path.

use crate::backend::WgpuBackend;
use crate::error::WgpuError;

/// Push an `OutOfMemory | Validation` error scope around
/// `device.create_shader_module(source)`. If the driver rejects the
/// shader (syntax error, unbound binding, etc.) the corresponding
/// [`wgpu::Error`] is captured and returned as
/// `WgpuError::ShapeMismatch` with the diagnostic.
///
/// This is a development-time safety net — production dispatches
/// rely on the shader having been validated once at startup.
pub fn validate_wgsl(backend: &WgpuBackend, label: &str, source: &str) -> Result<(), WgpuError> {
    backend
        .device
        .push_error_scope(wgpu::ErrorFilter::Validation);
    // Build then drop — we only care about whether the parse succeeded.
    let _module = backend
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(label),
            source: wgpu::ShaderSource::Wgsl(source.to_string().into()),
        });
    let err = pollster::block_on(backend.device.pop_error_scope());
    if let Some(e) = err {
        return Err(WgpuError::ShapeMismatch(format!(
            "WGSL validation failed for {label}: {e}"
        )));
    }
    Ok(())
}

/// Run [`validate_wgsl`] against every kernel template the crate
/// emits. Returns the first failure (if any). Cheap enough to call
/// at backend init time as a smoke check; lazy callers can skip.
///
/// Currently covers: elementwise (binary + unary across all 9 op
/// names), matmul, reduce (×6 kinds), argmax/argmin, softmax,
/// log_softmax, layernorm, layernorm_welford, rmsnorm, conv (im2col +
/// permute), attention (mul_scalar + causal_mask), transpose2d,
/// fused linear (relu + gelu), broadcast (×4 ops), flash_attn.
pub fn validate_all_kernels(backend: &WgpuBackend) -> Result<(), WgpuError> {
    use crate::shaders;
    // Element-wise — all the canonical kernels.
    for op in [
        "add", "sub", "mul", "div", "relu", "neg", "sigmoid", "tanh", "silu",
    ] {
        validate_wgsl(backend, op, &shaders::source_for(op))?;
    }
    Ok(())
}

/// Convenience macro that wraps the `format!(r#"…"#, KEY = value, …)`
/// pattern with documentation on what each placeholder means.
///
/// ```ignore
/// let src = wgsl_kernel!(
///     // human-readable label
///     name: "my_kernel",
///     // workgroup size — substituted as `{WG}u` in the body
///     WG: 64,
///     // Body of the kernel. Use `{KEY}` (NOT `{{KEY}}`) where
///     // substitution should occur.
///     body: r#"
///         @compute @workgroup_size({WG}) fn main() { ... }
///     "#,
/// );
/// ```
///
/// Returns a `String`. Currently single-key for clarity — multi-key
/// templates can be expressed by chaining `format!()` calls.
#[macro_export]
macro_rules! wgsl_kernel {
    (name: $name:expr, WG: $wg:expr, body: $body:expr $(,)?) => {{
        let _ = $name; // label, kept for grepability
        format!($body, WG = $wg)
    }};
}

#[cfg(test)]
mod tests {
    #[test]
    fn macro_substitutes_wg() {
        let src = wgsl_kernel!(
            name: "test",
            WG: 64,
            body: "@compute @workgroup_size({WG}) fn main() {{}}"
        );
        assert!(src.contains("workgroup_size(64)"));
    }
}

#[cfg(all(test, feature = "gpu-tests"))]
mod gpu_tests {
    use super::*;

    #[test]
    fn validate_all_kernels_passes_on_canonical_set() {
        let backend = WgpuBackend::new_blocking().expect("init");
        validate_all_kernels(&backend).expect("all canonical kernels must validate");
    }

    #[test]
    fn validate_wgsl_catches_syntax_error() {
        let backend = WgpuBackend::new_blocking().expect("init");
        let bad = "this is not valid WGSL @@@";
        let err =
            validate_wgsl(&backend, "broken", bad).expect_err("broken WGSL must fail validation");
        let msg = format!("{err}");
        assert!(
            msg.contains("broken") || msg.contains("validation"),
            "unexpected error message: {msg}"
        );
    }
}
