//! Chrome Trace Format emitter.
//!
//! The Trace Event Format is documented at
//! https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU
//! — the relevant subset for our needs is the "Complete Event" type
//! (`ph: "X"`) with name, category, timestamp, duration, pid/tid.

use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Instant;

/// One "complete" event in the trace. `ts` and `dur` are in
/// microseconds since the profiler started — that's the unit
/// `chrome://tracing` expects.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceEvent {
    /// Event name, shown on the timeline.
    pub name: String,
    /// Optional category (filterable in the UI).
    pub cat: String,
    /// Always `"X"` for complete events.
    pub ph: &'static str,
    /// Process id — 1 for the rustorch process.
    pub pid: u32,
    /// Thread id — host OS thread (best-effort identifier).
    pub tid: u64,
    /// Microseconds since profiler start.
    pub ts: u64,
    /// Duration in microseconds.
    pub dur: u64,
    /// Free-form args — useful for shapes / GPU ids / loss values.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub args: Option<serde_json::Value>,
}

/// Wraps a shared event buffer + the wall-clock zero point. Cloning
/// is cheap — it grabs another `Arc` to the same buffer.
#[derive(Clone)]
pub struct Profiler {
    inner: Arc<ProfilerInner>,
}

struct ProfilerInner {
    start: Instant,
    events: Mutex<Vec<TraceEvent>>,
}

impl Default for Profiler {
    fn default() -> Self {
        Self::new()
    }
}

impl Profiler {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(ProfilerInner {
                start: Instant::now(),
                events: Mutex::new(Vec::new()),
            }),
        }
    }

    /// Start a scoped span. The returned `ScopeGuard` records the
    /// duration when dropped.
    pub fn scope(&self, name: impl Into<String>) -> ScopeGuard {
        ScopeGuard {
            profiler: self.clone(),
            name: name.into(),
            cat: "default".into(),
            args: None,
            started: Instant::now(),
        }
    }

    /// Like `scope`, with an explicit category.
    pub fn scope_cat(&self, name: impl Into<String>, cat: impl Into<String>) -> ScopeGuard {
        ScopeGuard {
            profiler: self.clone(),
            name: name.into(),
            cat: cat.into(),
            args: None,
            started: Instant::now(),
        }
    }

    /// Append a pre-formed event to the buffer (used by the sampling
    /// profiler + tests).
    pub fn submit(&self, ev: TraceEvent) {
        self.inner.events.lock().push(ev);
    }

    /// Snapshot every recorded event so far.
    pub fn snapshot(&self) -> Vec<TraceEvent> {
        self.inner.events.lock().clone()
    }

    /// Microseconds elapsed since profiler start.
    pub fn now_us(&self) -> u64 {
        self.inner.start.elapsed().as_micros() as u64
    }

    /// Serialize the trace as Chrome-Trace JSON (the wrapper format
    /// expected by `chrome://tracing`).
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        let events = self.snapshot();
        let payload = serde_json::json!({
            "traceEvents": events,
            "displayTimeUnit": "ms",
        });
        serde_json::to_string(&payload)
    }

    /// Write the trace JSON to a file. Creates parents.
    pub fn write_json<P: AsRef<std::path::Path>>(&self, path: P) -> std::io::Result<()> {
        let s = self.to_json().map_err(std::io::Error::other)?;
        if let Some(parent) = path.as_ref().parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(path, s)
    }
}

/// Drop-on-scope-exit guard that records a span on the parent
/// `Profiler`. Created via `Profiler::scope`.
pub struct ScopeGuard {
    profiler: Profiler,
    name: String,
    cat: String,
    args: Option<serde_json::Value>,
    started: Instant,
}

impl ScopeGuard {
    /// Attach a JSON args blob to the event when it eventually
    /// records.
    pub fn with_args(mut self, args: serde_json::Value) -> Self {
        self.args = Some(args);
        self
    }
}

impl Drop for ScopeGuard {
    fn drop(&mut self) {
        let dur = self.started.elapsed().as_micros() as u64;
        let ts_at_start = self
            .started
            .saturating_duration_since(self.profiler.inner.start)
            .as_micros() as u64;
        self.profiler.submit(TraceEvent {
            name: std::mem::take(&mut self.name),
            cat: std::mem::take(&mut self.cat),
            ph: "X",
            pid: 1,
            tid: thread_id(),
            ts: ts_at_start,
            dur,
            args: self.args.take(),
        });
    }
}

/// Best-effort host thread id. We don't use `std::thread::current()`
/// because `ThreadId` doesn't expose its bits stably.
fn thread_id() -> u64 {
    // OS-specific syscalls would be more correct; for now we use
    // the address of a TLS variable as a stable per-thread cookie.
    thread_local! { static MARKER: u8 = const { 0 }; }
    MARKER.with(|m| m as *const u8 as u64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scope_records_event_on_drop() {
        let p = Profiler::new();
        {
            let _g = p.scope("forward");
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        let snap = p.snapshot();
        assert_eq!(snap.len(), 1);
        assert_eq!(snap[0].name, "forward");
        assert_eq!(snap[0].ph, "X");
        assert!(snap[0].dur > 0);
    }

    #[test]
    fn json_format_loads_as_chrome_trace() {
        let p = Profiler::new();
        {
            let _g = p.scope("conv");
        }
        let s = p.to_json().unwrap();
        let v: serde_json::Value = serde_json::from_str(&s).unwrap();
        assert!(v["traceEvents"].is_array());
        assert_eq!(v["displayTimeUnit"], "ms");
        assert_eq!(v["traceEvents"][0]["name"], "conv");
    }

    #[test]
    fn nested_scopes_record_independently() {
        let p = Profiler::new();
        {
            let _outer = p.scope("step");
            {
                let _inner = p.scope("forward");
            }
            {
                let _inner = p.scope("backward");
            }
        }
        let snap = p.snapshot();
        // Inner scopes drop first.
        assert_eq!(snap.len(), 3);
        let names: Vec<&str> = snap.iter().map(|e| e.name.as_str()).collect();
        assert_eq!(names, ["forward", "backward", "step"]);
    }

    #[test]
    fn args_are_attached_when_provided() {
        let p = Profiler::new();
        {
            let _g = p
                .scope("matmul")
                .with_args(serde_json::json!({"shape": [256, 256]}));
        }
        let snap = p.snapshot();
        assert_eq!(snap[0].args.as_ref().unwrap()["shape"][0], 256);
    }

    #[test]
    fn write_json_round_trips_through_disk() {
        let tmp = std::env::temp_dir().join(format!(
            "rustorch-prof-{}.json",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let p = Profiler::new();
        {
            let _g = p.scope("step0");
        }
        p.write_json(&tmp).unwrap();
        let s = std::fs::read_to_string(&tmp).unwrap();
        assert!(s.contains("step0"));
        let _ = std::fs::remove_file(&tmp);
    }
}
