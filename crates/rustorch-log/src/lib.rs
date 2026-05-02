//! Multi-sink logger for runs. Aligned with doc v0.7.2 §
//! `/Recipes/Logging & metrics`.
//!
//! ### API
//!
//! ```ignore
//! use rustorch_log::{Logger, StdoutJsonSink, MemorySink};
//!
//! let mem = MemorySink::default();
//! let logger = Logger::builder()
//!     .with_sink(StdoutJsonSink::default())
//!     .with_sink(mem.clone())
//!     .build();
//!
//! logger.metric("loss", 0.42, 100);
//! logger.event("checkpoint_saved", serde_json::json!({"path": "ckpt.safetensors"}));
//! ```
//!
//! ### Design
//!
//! * `LogEvent` is the wire-level enum — Metric, Event, Histogram,
//!   Image (path-only for now), Embedding (path-only), Hparams.
//! * `Sink` is `Send + Sync` — sinks live behind `Arc` so the logger
//!   can fan out to N destinations without locking.
//! * Failures from one sink are logged at `warn!` and never propagate
//!   to the caller — losing W&B doesn't crash a training step.
//! * `MemorySink` keeps everything in memory so the Console backend
//!   can echo logs into the SSE topics without round-tripping through
//!   gRPC.

mod sink;

pub use sink::{
    ConsoleHook, FileSink, LogEvent, LogLevel, Logger, LoggerBuilder, MemorySink, Metric, Sink,
    SinkError, StdoutJsonSink,
};
