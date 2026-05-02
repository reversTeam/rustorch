//! CUDA device discovery + properties.

use crate::error::CudaError;

/// One CUDA-capable device.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Device {
    /// Driver-assigned device index (0 .. num_devices).
    pub index: u32,
}

/// Capability + memory metadata for a device.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceProperties {
    /// Compute capability major (e.g. 8 for sm_80).
    pub cc_major: u32,
    /// Compute capability minor.
    pub cc_minor: u32,
    /// Total device global memory in bytes.
    pub total_mem_bytes: u64,
    /// Streaming-multiprocessor count.
    pub sm_count: u32,
}

impl Device {
    /// Enumerate all CUDA-capable devices visible to this process.
    /// Returns an empty Vec on no-cuda builds.
    #[cfg(not(feature = "cuda"))]
    pub fn all() -> Vec<Self> {
        Vec::new()
    }

    /// Enumerate via `cuDeviceGetCount`.
    #[cfg(feature = "cuda")]
    pub fn all() -> Vec<Self> {
        // Real impl routes via cudarc::driver::Device::all(); stubbed
        // until exercised on a real GPU.
        Vec::new()
    }

    /// Read this device's `DeviceProperties`. Returns
    /// `NoDeviceFound` on no-cuda builds.
    pub fn properties(&self) -> Result<DeviceProperties, CudaError> {
        if !cfg!(feature = "cuda") {
            return Err(CudaError::NoDeviceFound);
        }
        // Real impl: cuDeviceGetAttribute for each field.
        Err(CudaError::NoDeviceFound)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_returns_empty_without_cuda_feature() {
        let devs = Device::all();
        if !cfg!(feature = "cuda") {
            assert!(devs.is_empty());
        }
    }

    #[test]
    fn properties_returns_no_device_without_feature() {
        let d = Device { index: 0 };
        if !cfg!(feature = "cuda") {
            assert_eq!(d.properties().unwrap_err(), CudaError::NoDeviceFound);
        }
    }

    #[test]
    fn device_index_is_hashable() {
        // Used by handle / pipeline caches keyed on device.
        let mut set = std::collections::HashSet::new();
        set.insert(Device { index: 0 });
        set.insert(Device { index: 1 });
        set.insert(Device { index: 0 });
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn properties_struct_is_copy() {
        let p = DeviceProperties {
            cc_major: 8,
            cc_minor: 0,
            total_mem_bytes: 24 * 1024 * 1024 * 1024,
            sm_count: 108,
        };
        let q = p; // Copy
        assert_eq!(p, q);
    }
}
