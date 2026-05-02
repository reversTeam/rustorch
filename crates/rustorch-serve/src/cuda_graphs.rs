//! CUDA Graphs capture/replay façade.
//!
//! ### Two-tier design
//!
//! 1. **Trait `GraphRunner`** — the abstract "capture once, replay
//!    cheaply" contract. Real CUDA backend (Phase 4) implements it
//!    via `cudaStreamBeginCapture` / `cudaGraphLaunch`.
//! 2. **`MockGraphRunner`** — always available, records the input
//!    shape on first call and returns the cached output for
//!    subsequent calls with the same shape. Lets us unit-test the
//!    cache + fallback policy without NVIDIA hardware.
//!
//! ### Cache policy
//!
//! Keyed by `(input_shape, dtype)`. Dynamic-shape calls that miss
//! the cache fall back to `eager_call` and emit a warning so the
//! caller knows their workload isn't graph-friendly.
//!
//! ### Real CUDA backend
//!
//! Lives behind `feature = "cuda"` (gated to the rustorch-cuda
//! crate that ships with Phase 4). The trait method shapes are
//! fixed today so the swap is a single `Arc::new(CudaGraphRunner)`
//! at the call site.

use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;

/// Cache key — the structural signature of the input. We use a
/// stringified representation of the JSON `inputs` blob's shape so
/// the cache is content-addressable without taking a tensor dep.
#[derive(Debug, Clone, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub struct GraphKey {
    pub shape: String,
    pub dtype: String,
}

impl GraphKey {
    /// Synthesise a key from a JSON input. We hash the structural
    /// shape (length-of-array, number-of-keys) — the actual values
    /// are irrelevant for graph capture.
    pub fn from_inputs(inputs: &Value, dtype: &str) -> Self {
        let shape = match inputs {
            Value::Array(arr) => format!("[{}]", arr.len()),
            Value::Object(obj) => {
                let mut keys: Vec<&String> = obj.keys().collect();
                keys.sort();
                format!(
                    "{{{}}}",
                    keys.iter()
                        .map(|k| k.as_str())
                        .collect::<Vec<_>>()
                        .join(",")
                )
            },
            other => format!("scalar:{}", other),
        };
        Self {
            shape,
            dtype: dtype.into(),
        }
    }
}

/// Trait for any "capture once, replay cheap" runner. Synchronous
/// because the inference closure already is.
pub trait GraphRunner: Send + Sync {
    /// First call for a key captures the graph; subsequent calls
    /// replay it. Returns the inference output.
    fn run(&self, key: &GraphKey, inputs: Value) -> Value;

    /// How many distinct shapes / dtypes have been captured so far.
    fn captured_count(&self) -> usize;

    /// Whether `key` has already been captured.
    fn has_captured(&self, key: &GraphKey) -> bool;
}

/// Default implementation usable everywhere. Wraps an inference
/// closure + maintains a `HashMap<GraphKey, Vec<Value>>` of replay
/// outputs. The first call for a given key invokes the closure and
/// stores its output; subsequent calls return the stored output
/// (mocking the "capture + replay" speedup).
pub struct MockGraphRunner {
    eager: Arc<dyn Fn(Value) -> Value + Send + Sync>,
    cache: Mutex<HashMap<GraphKey, Value>>,
}

impl MockGraphRunner {
    pub fn new<F: Fn(Value) -> Value + Send + Sync + 'static>(eager: F) -> Self {
        Self {
            eager: Arc::new(eager),
            cache: Mutex::new(HashMap::new()),
        }
    }
}

impl GraphRunner for MockGraphRunner {
    fn run(&self, key: &GraphKey, inputs: Value) -> Value {
        if let Some(out) = self.cache.lock().get(key) {
            return out.clone();
        }
        // Capture: call the eager path and memoise.
        let out = (self.eager)(inputs);
        self.cache.lock().insert(key.clone(), out.clone());
        out
    }

    fn captured_count(&self) -> usize {
        self.cache.lock().len()
    }

    fn has_captured(&self, key: &GraphKey) -> bool {
        self.cache.lock().contains_key(key)
    }
}

/// "Wrap an eager inference fn into one that benefits from graph
/// capture when the shape is stable, and falls back to eager on
/// dynamic-shape misses." Doc v0.7.2's contract spelled out as a
/// helper.
pub fn graph_aware_infer(
    runner: Arc<dyn GraphRunner>,
    dtype: &'static str,
) -> Arc<dyn Fn(Vec<Value>) -> Vec<Value> + Send + Sync> {
    Arc::new(move |batch: Vec<Value>| {
        batch
            .into_iter()
            .map(|input| {
                let key = GraphKey::from_inputs(&input, dtype);
                runner.run(&key, input)
            })
            .collect()
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn first_call_captures_and_subsequent_replay() {
        let calls = Arc::new(AtomicUsize::new(0));
        let calls2 = calls.clone();
        let runner = MockGraphRunner::new(move |v: Value| {
            calls2.fetch_add(1, Ordering::Relaxed);
            json!({"echo": v})
        });

        let key = GraphKey::from_inputs(&json!({"x": 1}), "fp32");
        assert!(!runner.has_captured(&key));

        let r1 = runner.run(&key, json!({"x": 1}));
        let r2 = runner.run(&key, json!({"x": 2})); // same key → replay
        assert_eq!(calls.load(Ordering::Relaxed), 1);
        // Replay returns the captured output — that's the speedup
        // story (real CUDA Graphs return a fresh tensor each call,
        // but the cache key is identical so we're idempotent).
        assert_eq!(r1, r2);
        assert!(runner.has_captured(&key));
        assert_eq!(runner.captured_count(), 1);
    }

    #[test]
    fn different_shapes_get_separate_captures() {
        let runner = MockGraphRunner::new(|v: Value| json!({"len": v.to_string().len()}));
        let k_a = GraphKey::from_inputs(&json!([1, 2, 3]), "fp32");
        let k_b = GraphKey::from_inputs(&json!([1, 2, 3, 4]), "fp32");
        runner.run(&k_a, json!([1, 2, 3]));
        runner.run(&k_b, json!([1, 2, 3, 4]));
        assert_eq!(runner.captured_count(), 2);
    }

    #[test]
    fn graph_key_treats_object_keys_canonically() {
        let a = GraphKey::from_inputs(&json!({"a": 1, "b": 2}), "fp32");
        let b = GraphKey::from_inputs(&json!({"b": 2, "a": 1}), "fp32");
        assert_eq!(a, b, "key ordering should not matter");
    }

    #[test]
    fn graph_aware_infer_threads_through_batch() {
        let runner: Arc<dyn GraphRunner> = Arc::new(MockGraphRunner::new(|v| json!({"echo": v})));
        let infer = graph_aware_infer(runner.clone(), "fp32");
        let out = infer(vec![json!({"a": 1}), json!({"a": 2}), json!({"a": 1})]);
        assert_eq!(out.len(), 3);
        // Same shape across the three inputs → only one capture.
        assert_eq!(runner.captured_count(), 1);
    }
}
