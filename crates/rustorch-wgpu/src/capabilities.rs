//! Adapter capability probe.
//!
//! Captures the subset of `wgpu::Features` and `wgpu::Limits` that
//! kernel code actually inspects (e.g. SHADER_F16, max workgroup
//! size). Each kernel author can `if backend.capabilities().has_f16()
//! { … } else { … }` instead of touching `device.features().contains(...)`
//! directly — this keeps the per-kernel feature surface small and
//! traceable.
//!
//! Boolean fields are conservative: any `false` from the probe means
//! "treat as unavailable" even on platforms that technically support
//! it but where wgpu does not expose the feature flag. This avoids
//! over-eager dispatch when the runtime can't fall back.

use rustorch_core::tensor::dtype::Dtype;

/// Adapter / device feature snapshot.
#[derive(Debug, Clone)]
pub struct Capabilities {
    /// Snapshot of [`wgpu::Limits`] that came back from the device.
    pub limits: wgpu::Limits,
    /// Snapshot of [`wgpu::Features`] actually granted (subset of
    /// requested).
    pub features: wgpu::Features,
    /// `SHADER_F16` extension active — required for native f16
    /// arithmetic in WGSL. v1 kernels do all f16 math via f32 cast
    /// because no major adapter exposed this in stable wgpu 22.
    pub shader_f16: bool,
    /// `TIMESTAMP_QUERY` extension active — enables `wgpu::QuerySet`
    /// for kernel timing without a host round-trip. Used by future
    /// profiler code; off by default in v1.
    pub timestamp_query: bool,
    /// `SUBGROUP` extension active — enables `subgroupAdd`,
    /// `subgroupMin` and friends in WGSL. Reduces shared-memory
    /// pressure for tree reductions.
    pub subgroup: bool,
}

impl Capabilities {
    /// Probe the adapter + device for the features rustorch cares
    /// about. Returns a [`Capabilities`] snapshot.
    pub fn probe(adapter: &wgpu::Adapter, device: &wgpu::Device) -> Self {
        let features = device.features();
        let limits = device.limits();
        // wgpu 22 exposes optional feature flags via `Features::*`.
        // We probe by name so the snapshot stays stable as wgpu adds
        // more flags.
        let shader_f16 = features.contains(wgpu::Features::SHADER_F16);
        let timestamp_query = features.contains(wgpu::Features::TIMESTAMP_QUERY);
        // SUBGROUP / SHADER_PRIMITIVE_INDEX names changed across wgpu
        // versions; we test for the most common.
        let subgroup = features
            .iter_names()
            .any(|(name, _)| name.eq_ignore_ascii_case("SUBGROUP"));
        let _ = adapter; // adapter info is informational; kept for forward-compat
        Capabilities {
            limits,
            features,
            shader_f16,
            timestamp_query,
            subgroup,
        }
    }

    /// True iff the adapter claims native f16 support (matches
    /// `Features::SHADER_F16`).
    #[inline]
    pub fn has_f16(&self) -> bool {
        self.shader_f16
    }

    /// True iff the GPU can natively run a kernel of this dtype
    /// (i.e. without host-side cast). f32 / i32 always; bf16 / f16
    /// only when SHADER_F16 is granted.
    pub fn supports_dtype(&self, dtype: Dtype) -> bool {
        match dtype {
            Dtype::F32 | Dtype::I32 | Dtype::I64 | Dtype::I8 | Dtype::Bool => true,
            Dtype::F16 | Dtype::BF16 => self.shader_f16,
            // F64 has no path on any major wgpu backend.
            Dtype::F64 => false,
        }
    }

    /// Maximum bytes the device will accept for a single buffer.
    /// Caller should clamp before dispatch.
    #[inline]
    pub fn max_buffer_size(&self) -> u64 {
        self.limits.max_buffer_size
    }

    /// Maximum threads per workgroup in the X dimension.
    #[inline]
    pub fn max_workgroup_size_x(&self) -> u32 {
        self.limits.max_compute_workgroup_size_x
    }

    /// Maximum total threads per workgroup (`x * y * z`).
    #[inline]
    pub fn max_workgroup_invocations(&self) -> u32 {
        self.limits.max_compute_invocations_per_workgroup
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn supports_dtype_table_f32_yes_f64_no() {
        let caps = Capabilities {
            limits: wgpu::Limits::downlevel_defaults(),
            features: wgpu::Features::empty(),
            shader_f16: false,
            timestamp_query: false,
            subgroup: false,
        };
        assert!(caps.supports_dtype(Dtype::F32));
        assert!(caps.supports_dtype(Dtype::I32));
        assert!(!caps.supports_dtype(Dtype::F64));
        assert!(!caps.supports_dtype(Dtype::F16)); // shader_f16=false
        assert!(!caps.supports_dtype(Dtype::BF16));
    }

    #[test]
    fn supports_dtype_with_f16_enabled() {
        let caps = Capabilities {
            limits: wgpu::Limits::downlevel_defaults(),
            features: wgpu::Features::empty(),
            shader_f16: true,
            timestamp_query: false,
            subgroup: false,
        };
        assert!(caps.supports_dtype(Dtype::F16));
        assert!(caps.supports_dtype(Dtype::BF16));
    }
}

#[cfg(all(test, feature = "gpu-tests"))]
mod gpu_tests {
    use super::*;
    use crate::backend::WgpuBackend;

    #[test]
    fn capabilities_probe_returns_finite_limits() {
        let backend = WgpuBackend::new_blocking().expect("init");
        let caps = backend.capabilities();
        // Sanity checks: any sane GPU exposes ≥ 64 threads in WG.x.
        assert!(caps.max_workgroup_size_x() >= 64);
        // f32 always available.
        assert!(caps.supports_dtype(Dtype::F32));
    }
}
