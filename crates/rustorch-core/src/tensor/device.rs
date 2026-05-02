//! Device descriptor.
//!
//! Tags a [`Tensor`](super::tensor_impl::Tensor) with the
//! computational device it lives on. v1 is informational only — the
//! actual GPU storage is held by `rustorch_wgpu::WgpuStorage`, not
//! by the core `Tensor`. Future revisions can add a dispatch table
//! keyed by `Device` once a unified storage enum lands.
//!
//! Currently used:
//! - As a public marker on `to_device` calls (planned).
//! - By tooling (state_dict / serializers) to record where the
//!   tensor was last materialised.

/// A computational device.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Device {
    /// CPU — the default. All `rustorch_core::Tensor` instances live
    /// here at construction time.
    #[default]
    Cpu,
    /// GPU via `wgpu` (Vulkan / Metal / DX12 / WebGPU). Tensors
    /// become "Wgpu" once they have been uploaded via
    /// `rustorch_wgpu::to_gpu`. The actual buffer is held by
    /// `WgpuStorage`; the core `Tensor` keeps a CPU shadow until the
    /// caller drops it.
    Wgpu,
}

impl core::fmt::Display for Device {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Device::Cpu => write!(f, "cpu"),
            Device::Wgpu => write!(f, "wgpu"),
        }
    }
}

impl Device {
    /// Lower-case identifier — matches the PyTorch convention.
    /// `"cpu"` / `"wgpu"`.
    pub fn name(self) -> &'static str {
        match self {
            Device::Cpu => "cpu",
            Device::Wgpu => "wgpu",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_cpu() {
        assert_eq!(Device::default(), Device::Cpu);
    }

    #[test]
    fn display_matches_name() {
        assert_eq!(format!("{}", Device::Cpu), "cpu");
        assert_eq!(format!("{}", Device::Wgpu), "wgpu");
    }

    #[test]
    fn ord_eq_works() {
        let mut s = std::collections::HashSet::new();
        s.insert(Device::Cpu);
        s.insert(Device::Wgpu);
        assert_eq!(s.len(), 2);
        s.insert(Device::Cpu);
        assert_eq!(s.len(), 2);
    }
}
