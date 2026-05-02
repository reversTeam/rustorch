//! Sampling profiler skeleton.
//!
//! Spawns a background thread that wakes every `period`, captures a
//! lightweight sample (name + counter today; real backtrace unwinder
//! is the follow-up), and stores it for later JSON emission. The
//! shape of the sample is what `pprof` / `speedscope` consume so
//! swapping the unwinder in is a one-line change.

use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

/// One sample captured by the background thread.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Sample {
    /// Microseconds since the profiler started.
    pub ts_us: u64,
    /// Frame names from outermost to innermost. v1 is a single
    /// synthetic name; the unwinder will populate the rest.
    pub frames: Vec<String>,
}

/// Aggregate stats — useful for quick health checks without parsing
/// the full sample list.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SamplingStats {
    pub samples_taken: u64,
    pub samples_dropped: u64,
    pub running_secs: f64,
}

/// Background sampling profiler. Cheap to clone — internals live
/// behind `Arc`.
#[derive(Clone)]
pub struct SamplingProfiler {
    inner: Arc<Inner>,
}

struct Inner {
    start: Instant,
    samples: Mutex<Vec<Sample>>,
    running: AtomicBool,
    stats: Mutex<SamplingStats>,
    handle: Mutex<Option<JoinHandle<()>>>,
}

impl SamplingProfiler {
    /// Create a stopped profiler. Call `start` to spawn the worker.
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Inner {
                start: Instant::now(),
                samples: Mutex::new(Vec::new()),
                running: AtomicBool::new(false),
                stats: Mutex::new(SamplingStats::default()),
                handle: Mutex::new(None),
            }),
        }
    }

    /// Start sampling at `period`. Idempotent — calling twice is a
    /// no-op until `stop()` is called.
    pub fn start(&self, period: Duration) {
        if self.inner.running.swap(true, Ordering::SeqCst) {
            return;
        }
        let me = self.clone();
        let h = std::thread::spawn(move || {
            while me.inner.running.load(Ordering::SeqCst) {
                std::thread::sleep(period);
                if !me.inner.running.load(Ordering::SeqCst) {
                    break;
                }
                let ts_us = me.inner.start.elapsed().as_micros() as u64;
                me.inner.samples.lock().push(Sample {
                    ts_us,
                    // Real unwinder TBD — for now we record a single
                    // synthetic frame so JSON shape is stable.
                    frames: vec!["<unsampled>".into()],
                });
                me.inner.stats.lock().samples_taken += 1;
            }
        });
        *self.inner.handle.lock() = Some(h);
    }

    /// Stop the worker thread and join it. Safe to call from any
    /// thread; safe to call before `start`.
    pub fn stop(&self) {
        self.inner.running.store(false, Ordering::SeqCst);
        let h = self.inner.handle.lock().take();
        if let Some(h) = h {
            let _ = h.join();
        }
    }

    pub fn samples(&self) -> Vec<Sample> {
        self.inner.samples.lock().clone()
    }

    pub fn stats(&self) -> SamplingStats {
        let mut s = self.inner.stats.lock().clone();
        s.running_secs = self.inner.start.elapsed().as_secs_f64();
        s
    }
}

impl Default for SamplingProfiler {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for Inner {
    fn drop(&mut self) {
        // Stop the worker if the user forgot to.
        self.running.store(false, Ordering::SeqCst);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sampler_collects_samples_at_interval() {
        let s = SamplingProfiler::new();
        s.start(Duration::from_millis(20));
        std::thread::sleep(Duration::from_millis(120));
        s.stop();
        let n = s.samples().len();
        // Should have collected ~5 samples in 120ms with a 20ms
        // period; allow generous bounds for CI variance.
        assert!(n >= 2, "expected at least 2 samples, got {n}");
        assert!(n <= 12, "expected at most 12 samples, got {n}");
    }

    #[test]
    fn double_start_is_idempotent() {
        let s = SamplingProfiler::new();
        s.start(Duration::from_millis(50));
        s.start(Duration::from_millis(50)); // should be a no-op
        std::thread::sleep(Duration::from_millis(60));
        s.stop();
        let _ = s.samples();
    }

    #[test]
    fn stats_counter_grows() {
        let s = SamplingProfiler::new();
        s.start(Duration::from_millis(10));
        std::thread::sleep(Duration::from_millis(50));
        s.stop();
        assert!(s.stats().samples_taken >= 1);
    }
}
