//! Device descriptor.
//!
//! Tags a [`Tensor`](super::tensor_impl::Tensor) with the
//! computational device it lives on. P3.Z Task A wires this up to
//! the [`Storage`](super::storage::Storage) variant — `Cpu` ↔
//! `Storage::Cpu`, `Wgpu` ↔ `Storage::Wgpu(...)`, etc. The autograd
//! dispatcher routes ops by inspecting `Tensor::device()`.

/// A computational device.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Device {
    /// CPU — the default. All `rustorch_core::Tensor` instances live
    /// here at construction time.
    #[default]
    Cpu,
    /// Cross-platform GPU via `wgpu` (Vulkan / DX12 / WebGPU; also
    /// Metal-via-wgpu, but with a perf ceiling vs `Metal` direct).
    /// Canonical perf path on AMD RDNA3+, Intel Arc/Xe2, and the
    /// browser (WebGPU).
    Wgpu,
    /// **Apple Silicon Metal direct** (P3.Z Task J). Bypasses
    /// MoltenVK / Metal-via-wgpu to dispatch via the `metal` Rust
    /// bindings. Unlocks `simdgroup_matrix<bfloat,8,8>` (Apple
    /// tensor units) and unified memory — the canonical perf path
    /// to **beat PyTorch MPS** on Apple Silicon.
    Metal,
    /// **NVIDIA CUDA direct** (P3.Z Task M). Bypasses wgpu/Vulkan to
    /// dispatch via cudarc + cuBLASLt + cuDNN — the canonical perf
    /// path on NVIDIA datacenter (H100/B200) and consumer (RTX).
    Cuda,
}

impl core::fmt::Display for Device {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Device::Cpu => write!(f, "cpu"),
            Device::Wgpu => write!(f, "wgpu"),
            Device::Metal => write!(f, "metal"),
            Device::Cuda => write!(f, "cuda"),
        }
    }
}

impl Device {
    /// Lower-case identifier — matches the PyTorch convention.
    /// `"cpu"` / `"wgpu"` / `"metal"` / `"cuda"`.
    pub fn name(self) -> &'static str {
        match self {
            Device::Cpu => "cpu",
            Device::Wgpu => "wgpu",
            Device::Metal => "metal",
            Device::Cuda => "cuda",
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
        assert_eq!(format!("{}", Device::Metal), "metal");
        assert_eq!(format!("{}", Device::Cuda), "cuda");
    }

    #[test]
    fn ord_eq_works() {
        let mut s = std::collections::HashSet::new();
        s.insert(Device::Cpu);
        s.insert(Device::Wgpu);
        s.insert(Device::Metal);
        s.insert(Device::Cuda);
        assert_eq!(s.len(), 4);
        s.insert(Device::Cpu);
        assert_eq!(s.len(), 4);
    }
}
