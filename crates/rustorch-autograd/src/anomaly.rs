//! Anomaly mode for autograd — NaN/Inf detection (P1.5).
//!
//! Mirrors `torch.autograd.set_detect_anomaly(true)`. When enabled,
//! tensors produced by autograd ops are scanned for NaN/Inf values;
//! the first occurrence panics with the originating op name.
//!
//! Cost when **off** is zero (the flag check compiles to a single
//! load + branch). Cost when **on** is `O(numel)` per op — only use
//! during debugging.

use rustorch_core::tensor::dtype::Dtype;
use rustorch_core::tensor::tensor_impl::Tensor;
use std::cell::Cell;

thread_local! {
    static DETECT_ANOMALY: Cell<bool> = const { Cell::new(false) };
}

/// Toggle anomaly detection on the current thread.
pub fn set_detect_anomaly(enabled: bool) {
    DETECT_ANOMALY.with(|f| f.set(enabled));
}

/// True iff anomaly detection is currently enabled on this thread.
#[inline]
pub fn is_detect_anomaly() -> bool {
    DETECT_ANOMALY.with(|f| f.get())
}

/// RAII guard: enable anomaly detection within a scope, restore the
/// previous state on drop.
pub struct DetectAnomalyGuard {
    prev: bool,
}

impl DetectAnomalyGuard {
    /// Enter anomaly mode.
    pub fn new() -> Self {
        let prev = is_detect_anomaly();
        set_detect_anomaly(true);
        DetectAnomalyGuard { prev }
    }
}

impl Default for DetectAnomalyGuard {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for DetectAnomalyGuard {
    fn drop(&mut self) {
        set_detect_anomaly(self.prev);
    }
}

/// Scan a tensor for NaN / Inf. If anomaly mode is OFF, returns
/// immediately with `None`. If ON and a bad value is found, returns
/// the first index encountered as `Some(idx)`.
///
/// Use [`check_or_panic`] to wrap this with a panic for the typical
/// debug workflow.
pub fn first_bad_value(t: &Tensor) -> Option<usize> {
    if !is_detect_anomaly() {
        return None;
    }
    match t.dtype() {
        Dtype::F32 => t
            .as_slice::<f32>()
            .and_then(|s| s.iter().position(|v| !v.is_finite())),
        Dtype::F64 => t
            .as_slice::<f64>()
            .and_then(|s| s.iter().position(|v| !v.is_finite())),
        // Integers and bool can't be NaN/Inf.
        _ => None,
    }
}

/// Scan a tensor; if anomaly mode is on AND a NaN/Inf is found,
/// panic with a message naming the op.
pub fn check_or_panic(op: &str, t: &Tensor) {
    if let Some(idx) = first_bad_value(t) {
        let kind = match t.dtype() {
            Dtype::F32 => {
                let v = t.as_slice::<f32>().unwrap()[idx];
                if v.is_nan() {
                    "NaN"
                } else {
                    "Inf"
                }
            },
            Dtype::F64 => {
                let v = t.as_slice::<f64>().unwrap()[idx];
                if v.is_nan() {
                    "NaN"
                } else {
                    "Inf"
                }
            },
            _ => "non-finite",
        };
        panic!(
            "autograd anomaly: {kind} produced by op '{op}' at flat index {idx} \
             (shape {:?}, dtype {:?})",
            t.shape(),
            t.dtype()
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustorch_core::tensor::tensor_impl::Tensor;

    #[test]
    fn anomaly_off_by_default() {
        // Each test runs in its own thread (rust default) so this is fresh.
        assert!(!is_detect_anomaly());
    }

    #[test]
    fn detect_anomaly_guard_restores_on_drop() {
        assert!(!is_detect_anomaly());
        {
            let _g = DetectAnomalyGuard::new();
            assert!(is_detect_anomaly());
        }
        assert!(!is_detect_anomaly());
    }

    #[test]
    fn nested_guards_restore_correctly() {
        assert!(!is_detect_anomaly());
        {
            let _g1 = DetectAnomalyGuard::new();
            assert!(is_detect_anomaly());
            {
                let _g2 = DetectAnomalyGuard::new();
                assert!(is_detect_anomaly());
            }
            // Outer guard still active.
            assert!(is_detect_anomaly());
        }
        assert!(!is_detect_anomaly());
    }

    #[test]
    fn first_bad_value_off_returns_none() {
        // Off by default — should never scan.
        let t = Tensor::from_vec([3], vec![f32::NAN, 1.0, 2.0]).unwrap();
        assert!(first_bad_value(&t).is_none());
    }

    #[test]
    fn first_bad_value_on_returns_index() {
        let _g = DetectAnomalyGuard::new();
        let t = Tensor::from_vec([4], vec![1.0_f32, 2.0, f32::INFINITY, 4.0]).unwrap();
        assert_eq!(first_bad_value(&t), Some(2));
    }

    #[test]
    fn first_bad_value_clean_tensor_none() {
        let _g = DetectAnomalyGuard::new();
        let t = Tensor::from_vec([3], vec![1.0_f32, 2.0, 3.0]).unwrap();
        assert!(first_bad_value(&t).is_none());
    }

    #[test]
    fn integer_tensor_never_anomaly() {
        let _g = DetectAnomalyGuard::new();
        let t = Tensor::from_vec_typed::<i64, _>([3], vec![1_i64, 2, 3]).unwrap();
        assert!(first_bad_value(&t).is_none());
    }

    #[test]
    #[should_panic(expected = "anomaly")]
    fn check_or_panic_panics_on_nan() {
        let _g = DetectAnomalyGuard::new();
        let t = Tensor::from_vec([2], vec![f32::NAN, 1.0]).unwrap();
        check_or_panic("test_op", &t);
    }

    #[test]
    fn check_or_panic_silent_when_off() {
        // No guard — anomaly is off.
        let t = Tensor::from_vec([2], vec![f32::NAN, 1.0]).unwrap();
        check_or_panic("test_op", &t); // Should NOT panic.
    }
}
