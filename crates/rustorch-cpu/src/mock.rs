//! `MockBackend` — test double for verifying dispatch + call ordering
//! (P1.2 task `MockBackend`).
//!
//! Used by autograd tests to assert that backward calls a specific
//! kernel sequence without performing real computation. Records every
//! invocation into a `Vec<CallRecord>` accessible via
//! [`MockBackend::calls`].

use crate::backend::Backend;
use crate::error::BackendError;
use rustorch_core::tensor::tensor_impl::Tensor;
use std::sync::Mutex;

/// Single recorded backend invocation.
#[derive(Debug, Clone, PartialEq)]
pub struct CallRecord {
    /// Op name (`"add"`, `"matmul"`, ...).
    pub op: &'static str,
    /// Shapes of the input tensors (in order).
    pub input_shapes: Vec<Vec<usize>>,
}

/// Recordable backend — every method appends a CallRecord and returns
/// a canned tensor (default: a scalar 0).
pub struct MockBackend {
    calls: Mutex<Vec<CallRecord>>,
    /// Canned reply used by every method except `name`. Defaults to a
    /// scalar 0.0.
    canned: Mutex<Tensor>,
}

impl Default for MockBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl MockBackend {
    /// Build a fresh `MockBackend` with an empty call log and the
    /// default canned response (scalar `0.0`).
    pub fn new() -> Self {
        MockBackend {
            calls: Mutex::new(Vec::new()),
            canned: Mutex::new(Tensor::scalar(0.0)),
        }
    }

    /// Replace the canned response returned by every op method.
    pub fn set_canned(&self, t: Tensor) {
        *self.canned.lock().unwrap() = t;
    }

    /// Snapshot of the recorded call log.
    pub fn calls(&self) -> Vec<CallRecord> {
        self.calls.lock().unwrap().clone()
    }

    /// Number of recorded calls. Most-frequent assertion in tests.
    pub fn call_count(&self, op: &str) -> usize {
        self.calls
            .lock()
            .unwrap()
            .iter()
            .filter(|r| r.op == op)
            .count()
    }

    fn record(&self, op: &'static str, inputs: &[&Tensor]) {
        self.calls.lock().unwrap().push(CallRecord {
            op,
            input_shapes: inputs.iter().map(|t| t.shape().to_vec()).collect(),
        });
    }
}

impl Backend for MockBackend {
    fn name(&self) -> &'static str {
        "mock"
    }

    fn add(&self, lhs: &Tensor, rhs: &Tensor) -> Result<Tensor, BackendError> {
        self.record("add", &[lhs, rhs]);
        Ok(self.canned.lock().unwrap().clone())
    }

    fn sub(&self, lhs: &Tensor, rhs: &Tensor) -> Result<Tensor, BackendError> {
        self.record("sub", &[lhs, rhs]);
        Ok(self.canned.lock().unwrap().clone())
    }

    fn mul(&self, lhs: &Tensor, rhs: &Tensor) -> Result<Tensor, BackendError> {
        self.record("mul", &[lhs, rhs]);
        Ok(self.canned.lock().unwrap().clone())
    }

    fn div(&self, lhs: &Tensor, rhs: &Tensor) -> Result<Tensor, BackendError> {
        self.record("div", &[lhs, rhs]);
        Ok(self.canned.lock().unwrap().clone())
    }

    fn neg(&self, src: &Tensor) -> Result<Tensor, BackendError> {
        self.record("neg", &[src]);
        Ok(self.canned.lock().unwrap().clone())
    }

    fn matmul(&self, lhs: &Tensor, rhs: &Tensor) -> Result<Tensor, BackendError> {
        self.record("matmul", &[lhs, rhs]);
        Ok(self.canned.lock().unwrap().clone())
    }

    fn relu(&self, src: &Tensor) -> Result<Tensor, BackendError> {
        self.record("relu", &[src]);
        Ok(self.canned.lock().unwrap().clone())
    }

    fn eq(&self, lhs: &Tensor, rhs: &Tensor) -> Result<Tensor, BackendError> {
        self.record("eq", &[lhs, rhs]);
        Ok(self.canned.lock().unwrap().clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn name_is_mock() {
        let b = MockBackend::new();
        assert_eq!(b.name(), "mock");
    }

    #[test]
    fn records_calls_in_order() {
        let b = MockBackend::new();
        let t = Tensor::scalar(1.0);
        let _ = b.add(&t, &t);
        let _ = b.mul(&t, &t);
        let _ = b.add(&t, &t);
        let calls = b.calls();
        assert_eq!(calls.len(), 3);
        assert_eq!(calls[0].op, "add");
        assert_eq!(calls[1].op, "mul");
        assert_eq!(calls[2].op, "add");
    }

    #[test]
    fn records_input_shapes() {
        let b = MockBackend::new();
        let a = Tensor::from_vec([2usize, 3], vec![1.0_f32; 6]).unwrap();
        let bb = Tensor::from_vec([3usize], vec![1.0_f32; 3]).unwrap();
        let _ = b.add(&a, &bb);
        let calls = b.calls();
        assert_eq!(calls[0].input_shapes, vec![vec![2, 3], vec![3]]);
    }

    #[test]
    fn call_count_filters_by_op() {
        let b = MockBackend::new();
        let t = Tensor::scalar(0.0);
        let _ = b.add(&t, &t);
        let _ = b.add(&t, &t);
        let _ = b.mul(&t, &t);
        assert_eq!(b.call_count("add"), 2);
        assert_eq!(b.call_count("mul"), 1);
        assert_eq!(b.call_count("sub"), 0);
    }

    #[test]
    fn canned_response_returned() {
        let b = MockBackend::new();
        let canned = Tensor::from_vec([2usize], vec![7.0_f32, 8.0]).unwrap();
        b.set_canned(canned.clone());
        let t = Tensor::scalar(0.0);
        let r = b.matmul(&t, &t).unwrap();
        assert_eq!(r.as_slice::<f32>().unwrap(), &[7.0, 8.0]);
    }

    #[test]
    fn dyn_dispatch() {
        let b: Box<dyn Backend> = Box::new(MockBackend::new());
        let t = Tensor::scalar(1.0);
        let _ = b.relu(&t);
        // No introspection through dyn; just ensure dispatch compiles.
    }

    #[test]
    fn empty_call_log_initially() {
        let b = MockBackend::new();
        assert!(b.calls().is_empty());
        assert_eq!(b.call_count("add"), 0);
    }

    #[test]
    fn records_unary_input_shape() {
        let b = MockBackend::new();
        let t = Tensor::from_vec([5usize], vec![1.0_f32; 5]).unwrap();
        let _ = b.relu(&t);
        let _ = b.neg(&t);
        let calls = b.calls();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].op, "relu");
        assert_eq!(calls[0].input_shapes, vec![vec![5]]);
        assert_eq!(calls[1].op, "neg");
    }
}
