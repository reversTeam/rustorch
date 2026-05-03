//! `Module` trait — the common API for every nn building block
//! (P1.6 task `Module trait`).
//!
//! v1 ships the minimum surface needed to assemble + train a Linear+
//! ReLU MLP:
//! - `forward(input) -> Result<output>` for the actual computation
//! - `parameters()` returning every trainable [`Variable`] inside the
//!   module (recursively into children)
//! - `train()` / `eval()` toggling — drives Dropout and BatchNorm in
//!   later iterations.
//!
//! Helper [`ModuleError`] wraps autograd errors uniformly.

use rustorch_autograd::{BackwardError, Variable};
use rustorch_core::tensor::device::Device;
use rustorch_core::tensor::dtype::Dtype;

/// Errors returned by [`Module::forward`].
pub type ModuleError = BackwardError;

/// Trainable parameter newtype — alias for [`Variable`] in v1.
/// Distinguishes parameters from arbitrary tensors at the type level
/// for documentation and future static enforcement.
pub type Parameter = Variable;

/// Non-trainable buffer newtype — alias for [`Variable`] in v1.
/// Used by Modules that need to track running statistics (e.g.
/// BatchNorm) without making them learnable parameters.
pub type Buffer = Variable;

/// Common trait for every neural-network module.
pub trait Module: Send + Sync {
    /// Compute the forward pass.
    fn forward(&self, input: &Variable) -> Result<Variable, ModuleError>;

    /// Trainable parameters owned by this module + every child module.
    /// Order is stable across calls (used by optimisers to map slot →
    /// gradient).
    fn parameters(&self) -> Vec<Variable> {
        Vec::new()
    }

    /// Trainable parameters paired with their local names. Default impl
    /// derives names from index (`"0"`, `"1"`, …); composite modules
    /// override to expose meaningful names like `"weight"` / `"bias"` /
    /// child paths (e.g. `"0.weight"` for a Sequential).
    ///
    /// Used by [`crate::state_dict::state_dict`] to build a flat
    /// HashMap keyed by dotted path.
    fn named_parameters(&self) -> Vec<(String, Variable)> {
        self.parameters()
            .into_iter()
            .enumerate()
            .map(|(i, v)| (i.to_string(), v))
            .collect()
    }

    /// Switch to training mode (Dropout active, BatchNorm updates running
    /// stats). Default is no-op for stateless modules.
    fn train(&mut self) {}

    /// Switch to eval mode.
    fn eval(&mut self) {}

    /// Cast every parameter to the requested dtype **in place**.
    ///
    /// Walks `parameters()` and rebuilds each Variable's stored
    /// [`rustorch_core::Tensor`] via [`Tensor::to_dtype`]. Buffers
    /// (e.g. BatchNorm running stats) are *not* touched by default;
    /// composite modules with non-trainable Variables should override.
    ///
    /// Used by mixed-precision recipes:
    /// ```ignore
    /// model.to_dtype(Dtype::BF16);
    /// // forward / backward now run in bf16; the optimiser may keep
    /// // f32 master weights via a separate copy if desired.
    /// ```
    fn to_dtype(&mut self, target: Dtype) {
        for p in self.parameters() {
            let cur = p.tensor();
            if cur.dtype() != target {
                p.set_data(cur.to_dtype(target));
            }
        }
    }

    /// Convenience: cast every parameter to bf16 in place.
    #[inline]
    fn to_bf16(&mut self) {
        self.to_dtype(Dtype::BF16);
    }

    /// Convenience: cast every parameter to f16 in place.
    #[inline]
    fn to_f16(&mut self) {
        self.to_dtype(Dtype::F16);
    }

    /// Convenience: cast every parameter back to f32 in place.
    #[inline]
    fn to_f32(&mut self) {
        self.to_dtype(Dtype::F32);
    }

    /// Move every parameter to the requested device **in place**.
    ///
    /// P3.Y plan, Phase E — walks `parameters()` and re-tags each
    /// Variable's stored Tensor with the new device. With Storage
    /// Option B, the actual data stays in the CPU shadow; the device
    /// tag drives autograd's `pick_backend(...)` dispatch so the
    /// forward / backward / optimiser ops route to the right backend
    /// at op time.
    ///
    /// Composite modules (Sequential, MLP, …) inherit this default
    /// because their `parameters()` walks recursively into children;
    /// no per-module override needed.
    ///
    /// ```ignore
    /// let mut model: MyMlp = MyMlp::new();
    /// model.to_device(Device::Wgpu);
    /// // forward / backward / optimiser.step now run on Wgpu.
    /// ```
    fn to_device(&mut self, device: Device) {
        for p in self.parameters() {
            let cur = p.tensor();
            if cur.device() != device {
                let new_t = cur.clone().with_device(device);
                p.set_data(new_t);
            }
        }
    }
}

#[cfg(test)]
mod cast_tests {
    use super::*;
    use crate::Linear;

    #[test]
    fn linear_to_bf16_round_trip() {
        let mut layer = Linear::new(4, 3);
        // All params start in f32.
        for p in layer.parameters() {
            assert_eq!(p.tensor().dtype(), Dtype::F32);
        }
        layer.to_bf16();
        for p in layer.parameters() {
            assert_eq!(p.tensor().dtype(), Dtype::BF16);
        }
        // Round-trip back to f32.
        layer.to_f32();
        for p in layer.parameters() {
            assert_eq!(p.tensor().dtype(), Dtype::F32);
        }
    }

    #[test]
    fn sequential_to_device_propagates_recursively() {
        // P3.Y plan, Phase E step d1a387c5 — Sequential's default
        // `to_device` impl walks `parameters()` which recurses into all
        // child modules. So `model.to_device(Wgpu)` tags every Linear
        // (and any future child module) without per-module override.
        use crate::Sequential;
        let mut net: Sequential = Sequential::new()
            .add(Linear::new(8, 4))
            .add(Linear::new(4, 2));
        // Sanity: 4 params (2 linear × {weight, bias}) all on Cpu.
        assert_eq!(net.parameters().len(), 4);
        for p in net.parameters() {
            assert_eq!(p.tensor().device(), Device::Cpu);
        }
        net.to_device(Device::Wgpu);
        for p in net.parameters() {
            assert_eq!(p.tensor().device(), Device::Wgpu);
        }
    }

    #[test]
    fn linear_to_device_propagates_to_all_params() {
        // P3.Y plan, Phase E — `to_device` should walk parameters() and
        // tag each one. Verifies the default impl on the Module trait
        // works for a real module (Linear) without any per-module override.
        let mut layer = Linear::new(4, 3);
        for p in layer.parameters() {
            assert_eq!(p.tensor().device(), Device::Cpu);
        }
        layer.to_device(Device::Wgpu);
        for p in layer.parameters() {
            assert_eq!(p.tensor().device(), Device::Wgpu);
        }
        // Round-trip back to Cpu.
        layer.to_device(Device::Cpu);
        for p in layer.parameters() {
            assert_eq!(p.tensor().device(), Device::Cpu);
        }
    }

    #[test]
    fn to_dtype_is_noop_when_already_target() {
        let mut layer = Linear::new(2, 2);
        // Snapshot the storage pointer-equivalent (we use the data
        // values themselves) before and after a same-dtype cast.
        let before: Vec<Vec<f32>> = layer
            .parameters()
            .iter()
            .map(|p| p.tensor().as_slice::<f32>().unwrap().to_vec())
            .collect();
        layer.to_dtype(Dtype::F32);
        let after: Vec<Vec<f32>> = layer
            .parameters()
            .iter()
            .map(|p| p.tensor().as_slice::<f32>().unwrap().to_vec())
            .collect();
        assert_eq!(before, after);
    }
}
