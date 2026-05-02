//! CUDA event wrapper for cross-stream sync + timing.

use crate::error::CudaError;
use crate::stream::Stream;

/// Status of an `Event` query.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EventStatus {
    /// Recorded work has completed.
    Ready,
    /// Recorded work still in flight.
    NotReady,
}

/// CUDA event handle. On no-cuda builds: tracks recorded-yes/no via
/// a bool; query() returns Ready iff record() was called.
#[derive(Debug, Clone, Copy)]
pub struct Event {
    recorded: bool,
    /// Synthetic timestamp captured at record() — used by
    /// `elapsed_ms` when the cuda feature is off so tests can
    /// exercise the API end-to-end.
    timestamp_ns: u128,
}

impl Event {
    /// Build a new (un-recorded) event.
    pub fn new() -> Self {
        Self {
            recorded: false,
            timestamp_ns: 0,
        }
    }

    /// Record the event onto a stream. Without `--features cuda` this
    /// just stores a bool + a host clock timestamp so `elapsed_ms`
    /// has something meaningful to return.
    pub fn record(&mut self, _stream: &Stream) -> Result<(), CudaError> {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        self.recorded = true;
        self.timestamp_ns = now;
        Ok(())
    }

    /// Block until the recorded work completes. No-op on no-cuda
    /// builds.
    pub fn synchronize(&self) -> Result<(), CudaError> {
        Ok(())
    }

    /// Query status — Ready iff record() has been called.
    pub fn query(&self) -> EventStatus {
        if self.recorded {
            EventStatus::Ready
        } else {
            EventStatus::NotReady
        }
    }

    /// Elapsed milliseconds between this event and `other` (must both
    /// have been recorded). Mimics `cudaEventElapsedTime`.
    pub fn elapsed_ms(&self, other: &Event) -> Result<f32, CudaError> {
        if !self.recorded || !other.recorded {
            return Err(CudaError::Unsupported {
                msg: "elapsed_ms requires both events to be recorded".into(),
            });
        }
        let diff_ns = other.timestamp_ns.abs_diff(self.timestamp_ns);
        Ok((diff_ns as f64 / 1_000_000.0) as f32)
    }
}

impl Default for Event {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::Device;

    fn s() -> Stream {
        Stream::new(Device { index: 0 }).unwrap()
    }

    #[test]
    fn new_event_is_not_ready() {
        let e = Event::new();
        assert_eq!(e.query(), EventStatus::NotReady);
    }

    #[test]
    fn record_then_query_returns_ready() {
        let mut e = Event::new();
        e.record(&s()).unwrap();
        assert_eq!(e.query(), EventStatus::Ready);
    }

    #[test]
    fn synchronize_is_safe_on_unrecorded_event() {
        // No panic on the no-cuda fallback path.
        let e = Event::new();
        assert!(e.synchronize().is_ok());
    }

    #[test]
    fn elapsed_ms_requires_both_recorded() {
        let e1 = Event::new();
        let mut e2 = Event::new();
        e2.record(&s()).unwrap();
        assert!(e1.elapsed_ms(&e2).is_err());
    }

    #[test]
    fn elapsed_ms_finite_for_two_recorded_events() {
        let mut e1 = Event::new();
        let mut e2 = Event::new();
        e1.record(&s()).unwrap();
        std::thread::sleep(std::time::Duration::from_micros(100));
        e2.record(&s()).unwrap();
        let dt = e1.elapsed_ms(&e2).unwrap();
        assert!(dt >= 0.0 && dt.is_finite());
    }
}
