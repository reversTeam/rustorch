//! Chrome Trace Format (CTF) profiler. Aligned with doc v0.7.2 §
//! `/Recipes/Profiling`. Emits JSON loadable by `chrome://tracing`,
//! Perfetto, or `speedscope`.
//!
//! ### Usage
//!
//! ```rust
//! use rustorch_profile::Profiler;
//!
//! let mut p = Profiler::new();
//! {
//!     let _g = p.scope("forward");
//!     // ... real work ...
//! }
//! {
//!     let _g = p.scope("backward");
//!     // ... real work ...
//! }
//! let json = p.to_json().unwrap();
//! assert!(json.contains("\"name\":\"forward\""));
//! ```
//!
//! ### Format
//!
//! Chrome Trace Event Format with `ph: "X"` (complete events). Each
//! event records `{name, cat, pid, tid, ts (µs), dur (µs), args}`.
//! Pid is fixed to 1 (the rustorch process); tid uses the OS thread
//! id so cross-thread profiles render correctly.
//!
//! ### Sampling profiler
//!
//! `SamplingProfiler::start(period)` spawns a background thread that
//! captures the call stack of every `tid` it knows about every
//! `period`. Today the captured frames are stub names (the host
//! process's PC); a real implementation would unwind via
//! `backtrace`. The infrastructure (sample buffer, JSON emitter,
//! configurable period) is in place so the unwinder is the only
//! follow-up.

mod chrome;
mod sampling;

pub use chrome::{Profiler, ScopeGuard, TraceEvent};
pub use sampling::{Sample, SamplingProfiler, SamplingStats};
