//! `Sink` trait + the in-tree sinks (Stdout JSON, File, In-memory,
//! Console hook).
//!
//! W&B, MLflow and TensorBoard adapters live in this same module
//! behind the `http` feature (they need an HTTP client). v1 ships
//! the local sinks unconditionally so a training script with
//! `--log-format json` works without any external service.

use chrono::{DateTime, Utc};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::fs::OpenOptions;
use std::io::Write as _;
use std::path::PathBuf;
use std::sync::Arc;

/// Severity hint for `Event` payloads. Doesn't affect routing — every
/// sink sees every level — but useful for downstream filters.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum LogLevel {
    Trace,
    Debug,
    Info,
    Warn,
    Error,
}

/// One scalar sample point. Mirrors the Console DB shape so the
/// runner can stream straight in.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Metric {
    pub name: String,
    pub value: f64,
    pub step: i64,
    pub ts: DateTime<Utc>,
}

/// Wire-level event. The `Sink::submit` method is the only entry
/// point — multiple variants exist so sinks can route differently
/// (W&B has a separate API for histograms vs scalars).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum LogEvent {
    /// Single scalar at `step`.
    Metric(Metric),
    /// Structured event (`name` + arbitrary JSON payload).
    Event {
        name: String,
        level: LogLevel,
        payload: serde_json::Value,
        ts: DateTime<Utc>,
    },
    /// Bucketed histogram. Buckets are pre-computed by the caller.
    Histogram {
        name: String,
        buckets: Vec<f64>,
        counts: Vec<u64>,
        step: i64,
        ts: DateTime<Utc>,
    },
    /// Image artifact reference. The path is relative to the run's
    /// artifact directory; the actual bytes don't go through the
    /// sink (they sit on disk).
    Image {
        name: String,
        path: String,
        step: i64,
        ts: DateTime<Utc>,
    },
    /// Embedding tensor reference (for projector views).
    Embedding {
        name: String,
        path: String,
        labels_path: Option<String>,
        step: i64,
        ts: DateTime<Utc>,
    },
    /// Hyperparameters — emitted at most once per run, alongside the
    /// final metrics for ranking.
    Hparams {
        params: serde_json::Value,
        final_metrics: serde_json::Value,
        ts: DateTime<Utc>,
    },
}

impl LogEvent {
    /// Returns the event's logical timestamp.
    pub fn ts(&self) -> DateTime<Utc> {
        match self {
            Self::Metric(m) => m.ts,
            Self::Event { ts, .. } => *ts,
            Self::Histogram { ts, .. } => *ts,
            Self::Image { ts, .. } => *ts,
            Self::Embedding { ts, .. } => *ts,
            Self::Hparams { ts, .. } => *ts,
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum SinkError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("serialize: {0}")]
    Serde(#[from] serde_json::Error),
    #[error("sink rejected event: {0}")]
    Rejected(String),
}

/// One destination for `LogEvent`s. Implementations must be
/// `Send + Sync` so the logger can fan out from any thread.
pub trait Sink: Send + Sync {
    /// Forward an event. Returning `Err` does NOT stop the logger —
    /// the caller logs at `warn!` and moves on.
    fn submit(&self, ev: &LogEvent) -> Result<(), SinkError>;

    /// Best-effort flush. Default is a no-op.
    fn flush(&self) -> Result<(), SinkError> {
        Ok(())
    }

    /// Sink name for diagnostics.
    fn name(&self) -> &'static str;
}

// ---- StdoutJsonSink -------------------------------------------------

/// One JSON line per event on stdout. Cheapest possible sink — used
/// when `--log-format json` is set on the CLI.
#[derive(Debug, Default)]
pub struct StdoutJsonSink {
    /// `true` to print pretty-formatted JSON (multiline). Defaults
    /// to compact (one line per event), which is what tools expect.
    pub pretty: bool,
}

impl Sink for StdoutJsonSink {
    fn submit(&self, ev: &LogEvent) -> Result<(), SinkError> {
        let s = if self.pretty {
            serde_json::to_string_pretty(ev)?
        } else {
            serde_json::to_string(ev)?
        };
        println!("{s}");
        Ok(())
    }
    fn name(&self) -> &'static str {
        "stdout_json"
    }
}

// ---- FileSink ------------------------------------------------------

/// Append-only NDJSON file. Each event is one line. Useful for
/// post-mortem inspection without a network sink.
pub struct FileSink {
    path: PathBuf,
    handle: Mutex<std::fs::File>,
}

impl FileSink {
    pub fn create(path: impl Into<PathBuf>) -> Result<Self, SinkError> {
        let path = path.into();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let f = OpenOptions::new().create(true).append(true).open(&path)?;
        Ok(Self {
            path,
            handle: Mutex::new(f),
        })
    }

    pub fn path(&self) -> &PathBuf {
        &self.path
    }
}

impl Sink for FileSink {
    fn submit(&self, ev: &LogEvent) -> Result<(), SinkError> {
        let s = serde_json::to_string(ev)?;
        let mut f = self.handle.lock();
        writeln!(f, "{s}")?;
        Ok(())
    }
    fn flush(&self) -> Result<(), SinkError> {
        self.handle.lock().flush()?;
        Ok(())
    }
    fn name(&self) -> &'static str {
        "file"
    }
}

// ---- MemorySink ----------------------------------------------------

/// In-memory ring used for tests + the Console SSE bridge. Cloning
/// gives shared access to the same underlying buffer.
#[derive(Debug, Default, Clone)]
pub struct MemorySink {
    inner: Arc<Mutex<Vec<LogEvent>>>,
}

impl MemorySink {
    pub fn snapshot(&self) -> Vec<LogEvent> {
        self.inner.lock().clone()
    }

    pub fn len(&self) -> usize {
        self.inner.lock().len()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.lock().is_empty()
    }

    pub fn clear(&self) {
        self.inner.lock().clear();
    }
}

impl Sink for MemorySink {
    fn submit(&self, ev: &LogEvent) -> Result<(), SinkError> {
        self.inner.lock().push(ev.clone());
        Ok(())
    }
    fn name(&self) -> &'static str {
        "memory"
    }
}

// ---- ConsoleHook --------------------------------------------------

/// Sink that calls a user-supplied closure for each event. Used by
/// the Console backend to push events into the SSE hub without
/// taking a runtime dep on the broadcast machinery here.
pub struct ConsoleHook {
    cb: Box<dyn Fn(&LogEvent) + Send + Sync>,
}

impl ConsoleHook {
    pub fn new<F: Fn(&LogEvent) + Send + Sync + 'static>(cb: F) -> Self {
        Self { cb: Box::new(cb) }
    }
}

impl Sink for ConsoleHook {
    fn submit(&self, ev: &LogEvent) -> Result<(), SinkError> {
        (self.cb)(ev);
        Ok(())
    }
    fn name(&self) -> &'static str {
        "console_hook"
    }
}

// ---- Logger --------------------------------------------------------

/// Multi-sink logger. Cheap to clone — internals live behind `Arc`.
#[derive(Clone)]
pub struct Logger {
    sinks: Arc<Vec<Arc<dyn Sink>>>,
}

impl Logger {
    pub fn builder() -> LoggerBuilder {
        LoggerBuilder::default()
    }

    pub fn submit(&self, ev: LogEvent) {
        for sink in self.sinks.iter() {
            if let Err(e) = sink.submit(&ev) {
                tracing::warn!(sink = sink.name(), error = %e, "log sink failed");
            }
        }
    }

    /// Convenience — construct a `LogEvent::Metric` and submit.
    /// NaN values are silently dropped (logged once at warn).
    pub fn metric(&self, name: impl Into<String>, value: f64, step: i64) {
        if !value.is_finite() {
            tracing::warn!(value, "log_metric: dropping non-finite value");
            return;
        }
        self.submit(LogEvent::Metric(Metric {
            name: name.into(),
            value,
            step,
            ts: Utc::now(),
        }));
    }

    pub fn event(&self, name: impl Into<String>, payload: serde_json::Value) {
        self.event_at(name, LogLevel::Info, payload);
    }

    pub fn event_at(&self, name: impl Into<String>, level: LogLevel, payload: serde_json::Value) {
        self.submit(LogEvent::Event {
            name: name.into(),
            level,
            payload,
            ts: Utc::now(),
        });
    }

    pub fn histogram(
        &self,
        name: impl Into<String>,
        buckets: Vec<f64>,
        counts: Vec<u64>,
        step: i64,
    ) {
        self.submit(LogEvent::Histogram {
            name: name.into(),
            buckets,
            counts,
            step,
            ts: Utc::now(),
        });
    }

    pub fn image(&self, name: impl Into<String>, path: impl Into<String>, step: i64) {
        self.submit(LogEvent::Image {
            name: name.into(),
            path: path.into(),
            step,
            ts: Utc::now(),
        });
    }

    pub fn hparams(&self, params: serde_json::Value, final_metrics: serde_json::Value) {
        self.submit(LogEvent::Hparams {
            params,
            final_metrics,
            ts: Utc::now(),
        });
    }

    /// Flush every sink. Best-effort.
    pub fn flush(&self) {
        for s in self.sinks.iter() {
            let _ = s.flush();
        }
    }
}

/// Fluent builder.
#[derive(Default)]
pub struct LoggerBuilder {
    sinks: Vec<Arc<dyn Sink>>,
}

impl LoggerBuilder {
    pub fn with_sink<S: Sink + 'static>(mut self, sink: S) -> Self {
        self.sinks.push(Arc::new(sink));
        self
    }

    pub fn build(self) -> Logger {
        Logger {
            sinks: Arc::new(self.sinks),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn memory_sink_collects_metrics() {
        let mem = MemorySink::default();
        let logger = Logger::builder().with_sink(mem.clone()).build();
        logger.metric("loss", 0.5, 0);
        logger.metric("loss", 0.4, 1);
        let snap = mem.snapshot();
        assert_eq!(snap.len(), 2);
        match &snap[0] {
            LogEvent::Metric(m) => assert_eq!(m.value, 0.5),
            _ => panic!("expected metric"),
        }
    }

    #[test]
    fn nan_metric_is_dropped() {
        let mem = MemorySink::default();
        let logger = Logger::builder().with_sink(mem.clone()).build();
        logger.metric("loss", f64::NAN, 0);
        logger.metric("loss", f64::INFINITY, 1);
        logger.metric("loss", 0.42, 2);
        assert_eq!(mem.len(), 1);
    }

    #[test]
    fn multi_sink_fans_out() {
        let a = MemorySink::default();
        let b = MemorySink::default();
        let logger = Logger::builder()
            .with_sink(a.clone())
            .with_sink(b.clone())
            .build();
        logger.metric("acc", 0.91, 100);
        assert_eq!(a.len(), 1);
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn event_payload_roundtrips() {
        let mem = MemorySink::default();
        let logger = Logger::builder().with_sink(mem.clone()).build();
        logger.event_at(
            "checkpoint_saved",
            LogLevel::Info,
            serde_json::json!({"step": 100, "path": "x.safetensors"}),
        );
        let snap = mem.snapshot();
        match &snap[0] {
            LogEvent::Event {
                name,
                level,
                payload,
                ..
            } => {
                assert_eq!(name, "checkpoint_saved");
                assert_eq!(*level, LogLevel::Info);
                assert_eq!(payload["step"], 100);
            },
            _ => panic!("expected event"),
        }
    }

    #[test]
    fn console_hook_invokes_callback() {
        let counter = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let c2 = counter.clone();
        let logger = Logger::builder()
            .with_sink(ConsoleHook::new(move |_ev| {
                c2.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }))
            .build();
        for _ in 0..5 {
            logger.metric("x", 1.0, 0);
        }
        assert_eq!(counter.load(std::sync::atomic::Ordering::Relaxed), 5);
    }

    #[test]
    fn file_sink_appends_ndjson() {
        let tmp = std::env::temp_dir().join(format!("rustorch-log-{}.ndjson", ulid_now()));
        let sink = FileSink::create(&tmp).unwrap();
        let logger = Logger::builder().with_sink(sink).build();
        logger.metric("loss", 0.1, 0);
        logger.metric("loss", 0.05, 1);
        logger.flush();
        let s = std::fs::read_to_string(&tmp).unwrap();
        assert_eq!(s.lines().count(), 2);
        // Each line is valid JSON and has a `kind` field.
        for line in s.lines() {
            let v: serde_json::Value = serde_json::from_str(line).unwrap();
            assert_eq!(v["kind"], "metric");
        }
        let _ = std::fs::remove_file(&tmp);
    }

    fn ulid_now() -> String {
        use std::time::{SystemTime, UNIX_EPOCH};
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
            .to_string()
    }

    #[test]
    fn high_throughput_does_not_drop() {
        // Reproduce the doc target — 1000 metric calls/s sustained
        // through the synchronous path. We don't actually clock here
        // (CI variance) but assert nothing is lost.
        let mem = MemorySink::default();
        let logger = Logger::builder().with_sink(mem.clone()).build();
        for i in 0..1000 {
            logger.metric("loss", 1.0 / (i + 1) as f64, i);
        }
        assert_eq!(mem.len(), 1000);
    }
}
