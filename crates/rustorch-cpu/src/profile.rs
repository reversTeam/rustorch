//! Lightweight profiler hooks (P1.2 task `Profiler hooks`).
//!
//! v1 ships a minimal API for timing op invocations and emitting a
//! Chrome trace JSON. The full Phase-2.5 profiler (sampling mode,
//! catched dataloader/allreduce/checkpoint events, NVTX ranges) lives
//! in P2.4.
//!
//! Off by default behind the `profile` feature flag — when disabled,
//! all `Profiler::*` calls are no-ops and compile to nothing.
//!
//! ```
//! # #[cfg(feature = "profile")]
//! # {
//! use rustorch_cpu::profile::Profiler;
//!
//! let _g = Profiler::scope("matmul", &[]);
//! // ... op runs here ...
//! # }
//! ```

#[cfg(feature = "profile")]
mod imp {
    use std::cell::RefCell;
    use std::time::{SystemTime, UNIX_EPOCH};

    /// One recorded event in Chrome-trace format.
    #[derive(Debug, Clone)]
    pub struct Event {
        /// Op name.
        pub name: &'static str,
        /// Microseconds since epoch.
        pub ts_us: u128,
        /// Event duration in microseconds.
        pub dur_us: u128,
        /// Free-form tags.
        pub args: Vec<(&'static str, String)>,
    }

    thread_local! {
        static EVENTS: RefCell<Vec<Event>> = const { RefCell::new(Vec::new()) };
    }

    /// Profiler scope guard — records a `complete event` when dropped.
    pub struct Profiler {
        name: &'static str,
        start: u128,
        args: Vec<(&'static str, String)>,
    }

    impl Profiler {
        /// Open a new scope with `name` and optional key=value tags.
        #[must_use = "the returned guard must outlive the operation it profiles"]
        pub fn scope(name: &'static str, args: &[(&'static str, &str)]) -> Self {
            Profiler {
                name,
                start: now_us(),
                args: args.iter().map(|(k, v)| (*k, (*v).to_string())).collect(),
            }
        }
    }

    impl Drop for Profiler {
        fn drop(&mut self) {
            let dur = now_us().saturating_sub(self.start);
            let ev = Event {
                name: self.name,
                ts_us: self.start,
                dur_us: dur,
                args: core::mem::take(&mut self.args),
            };
            EVENTS.with(|e| e.borrow_mut().push(ev));
        }
    }

    /// Snapshot the recorded events on the current thread.
    pub fn snapshot() -> Vec<Event> {
        EVENTS.with(|e| e.borrow().clone())
    }

    /// Clear the per-thread event ring.
    pub fn clear() {
        EVENTS.with(|e| e.borrow_mut().clear());
    }

    /// Number of recorded events on the current thread.
    pub fn count() -> usize {
        EVENTS.with(|e| e.borrow().len())
    }

    /// Render the per-thread event ring as a Chrome trace JSON string.
    /// Format: `{"traceEvents": [...]}` per the Chrome tracing spec.
    pub fn flush_chrome_trace() -> String {
        let evs = snapshot();
        let mut s = String::from("{\"traceEvents\":[");
        for (i, e) in evs.iter().enumerate() {
            if i > 0 {
                s.push(',');
            }
            s.push_str(&format!(
                r#"{{"name":"{}","ph":"X","ts":{},"dur":{},"pid":0,"tid":0"#,
                escape(e.name),
                e.ts_us,
                e.dur_us,
            ));
            if !e.args.is_empty() {
                s.push_str(",\"args\":{");
                for (j, (k, v)) in e.args.iter().enumerate() {
                    if j > 0 {
                        s.push(',');
                    }
                    s.push_str(&format!("\"{}\":\"{}\"", escape(k), escape(v)));
                }
                s.push('}');
            }
            s.push('}');
        }
        s.push_str("]}");
        s
    }

    fn now_us() -> u128 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_micros())
            .unwrap_or(0)
    }

    fn escape(s: &str) -> String {
        s.replace('\\', "\\\\").replace('"', "\\\"")
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn scope_records_event() {
            clear();
            {
                let _g = Profiler::scope("test_op", &[]);
                std::thread::sleep(std::time::Duration::from_micros(10));
            }
            assert_eq!(count(), 1);
            let evs = snapshot();
            assert_eq!(evs[0].name, "test_op");
            assert!(evs[0].dur_us >= 1);
        }

        #[test]
        fn empty_flush() {
            clear();
            let s = flush_chrome_trace();
            assert_eq!(s, r#"{"traceEvents":[]}"#);
        }

        #[test]
        fn flush_contains_event_name() {
            clear();
            {
                let _g = Profiler::scope("my_op", &[]);
            }
            let s = flush_chrome_trace();
            assert!(s.contains("\"name\":\"my_op\""));
            assert!(s.contains("\"ph\":\"X\""));
        }

        #[test]
        fn flush_with_args() {
            clear();
            {
                let _g = Profiler::scope("op", &[("dtype", "f32")]);
            }
            let s = flush_chrome_trace();
            assert!(s.contains("\"dtype\":\"f32\""));
        }

        #[test]
        fn count_grows() {
            clear();
            for _ in 0..5 {
                let _g = Profiler::scope("a", &[]);
            }
            assert_eq!(count(), 5);
        }

        #[test]
        fn escape_handles_quotes() {
            clear();
            {
                let _g = Profiler::scope("bad\"name", &[]);
            }
            let s = flush_chrome_trace();
            assert!(s.contains(r#""name":"bad\"name""#));
        }
    }
}

#[cfg(not(feature = "profile"))]
mod imp {
    /// No-op profiler scope when the `profile` feature is disabled.
    pub struct Profiler;

    impl Profiler {
        /// Returns a no-op guard.
        #[must_use = "even no-op profiler guard should be bound to maintain symmetry"]
        pub fn scope(_name: &'static str, _args: &[(&'static str, &str)]) -> Self {
            Profiler
        }
    }

    /// No events recorded when feature off.
    pub fn count() -> usize {
        0
    }

    /// Empty trace JSON when feature off.
    pub fn flush_chrome_trace() -> String {
        r#"{"traceEvents":[]}"#.into()
    }

    /// Clears nothing.
    pub fn clear() {}

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn no_op_count_is_zero() {
            assert_eq!(count(), 0);
        }

        #[test]
        fn no_op_flush_is_empty_array() {
            let s = flush_chrome_trace();
            assert_eq!(s, r#"{"traceEvents":[]}"#);
        }

        #[test]
        fn no_op_scope_compiles() {
            let _g = Profiler::scope("smoke", &[]);
        }
    }
}

pub use imp::{clear, count, flush_chrome_trace, Profiler};
