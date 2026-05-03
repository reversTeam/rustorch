//! Device-aware backend dispatcher (P3.Y plan, Phase B1).
//!
//! Picks the right `&'static dyn Backend` at op time based on the
//! [`Device`](rustorch_core::tensor::device::Device) of a Variable's
//! tensor — so the autograd op layer can call `pick_backend(...).matmul(...)`
//! and have it route to either `cpu_backend()` or `wgpu_backend()` without
//! threading the backend through every call site.
//!
//! Without the `wgpu` feature, `Device::Wgpu` panics with a clear error
//! pointing the user at the `rustorch-autograd/wgpu` feature flag. With
//! the feature, dispatch routes to `rustorch_wgpu::wgpu_backend()`.
//!
//! Mixed-device ops are explicitly rejected (no auto-promotion) so
//! debugging is straightforward and aligns with PyTorch RFC 0026.

use crate::backward::BackwardError;
use crate::variable::Variable;
use rustorch_core::tensor::device::Device;
use rustorch_cpu::backend::Backend;
use rustorch_cpu::cpu_backend::cpu_backend;

/// Pick the `&'static dyn Backend` for a given device.
///
/// CPU dispatch is always available. Wgpu dispatch requires the
/// `rustorch-autograd/wgpu` feature (off by default).
pub(crate) fn pick_backend(device: Device) -> &'static dyn Backend {
    match device {
        Device::Cpu => cpu_backend(),
        #[cfg(feature = "wgpu")]
        Device::Wgpu => rustorch_wgpu::wgpu_backend(),
        #[cfg(not(feature = "wgpu"))]
        Device::Wgpu => panic!(
            "rustorch-autograd: dispatched to Device::Wgpu but the `wgpu` feature is OFF. \
             Build with `--features wgpu` (or enable the umbrella `rustorch/wgpu` feature) \
             to enable GPU dispatch through `rustorch-wgpu`."
        ),
    }
}

/// Verify two operands live on the same device, returning that device
/// on success or [`BackwardError::DeviceMismatch`] otherwise.
///
/// Used by every binary op forward to gate dispatch and surface a clear
/// error before any kernel runs.
pub(crate) fn require_same_device_2(
    op: &'static str,
    lhs: &Variable,
    rhs: &Variable,
) -> Result<Device, BackwardError> {
    let l = lhs.device();
    let r = rhs.device();
    if l != r {
        return Err(BackwardError::DeviceMismatch { op, lhs: l, rhs: r });
    }
    Ok(l)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pick_cpu_returns_singleton() {
        let a = pick_backend(Device::Cpu);
        let b = pick_backend(Device::Cpu);
        assert!(std::ptr::eq(a, b));
        assert_eq!(a.name(), "cpu");
    }

    #[test]
    fn require_same_device_accepts_matching() {
        let a = Variable::new(rustorch_core::tensor::tensor_impl::Tensor::scalar(1.0));
        let b = Variable::new(rustorch_core::tensor::tensor_impl::Tensor::scalar(2.0));
        let device = require_same_device_2("test", &a, &b).unwrap();
        assert_eq!(device, Device::Cpu);
    }

    #[test]
    fn require_same_device_rejects_mismatch() {
        // Build two Variables on different devices using `Tensor::with_device`
        // — note this only sets the device tag, the data still lives in
        // the CPU shadow. That's fine for testing the dispatcher's
        // device-mismatch gate, which acts before any kernel call.
        let cpu_t = rustorch_core::tensor::tensor_impl::Tensor::scalar(1.0);
        let wgpu_t =
            rustorch_core::tensor::tensor_impl::Tensor::scalar(2.0).with_device(Device::Wgpu);
        let a = Variable::new(cpu_t);
        let b = Variable::new(wgpu_t);
        let err = require_same_device_2("test", &a, &b).unwrap_err();
        match err {
            BackwardError::DeviceMismatch { op, lhs, rhs } => {
                assert_eq!(op, "test");
                assert_eq!(lhs, Device::Cpu);
                assert_eq!(rhs, Device::Wgpu);
            },
            other => panic!("expected DeviceMismatch, got {other:?}"),
        }
    }
}
