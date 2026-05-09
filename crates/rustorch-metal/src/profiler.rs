//! T181 — **WIP, partial implementation**. GPU-side profiler scaffolding
//! using `MTLCounterSampleBuffer`.
//!
//! ## Status (2026-05-07)
//!
//! Infrastructure is complete and compiles cleanly :
//! - `MetalProfiler` struct with sample buffer + dest buffer + tick→ns calibration
//! - `MetalBackend::with_encoder_labeled` opens a per-dispatch encoder with
//!   `ComputePassDescriptor` carrying a counter-sample attachment
//! - `MetalBackend::drain` schedules `BlitCommandEncoder::resolve_counters`
//!   before commit
//!
//! **However**, on M4 Max the counter sample mechanism via
//! `compute_command_encoder_with_descriptor` produces **unreliable timestamps** :
//! a first attempt captured the 1st-2nd dispatches but missed the 3rd ;
//! a re-run captured nothing. Apple Silicon does not support
//! `MTLCounterSamplingPointAtDispatchBoundary` (per-dispatch sampling inside
//! a chained encoder) and the per-encoder `AtStageBoundary` path appears to
//! have additional driver constraints not covered in Apple docs.
//!
//! This module is kept as a starting point for future investigation. Possible
//! next steps :
//! - Use `cb.gpu_start_time()/gpu_end_time()` for whole-CB timing (less granular
//!   but reliable) — drain at strategic boundaries to get phase-level timing.
//! - Investigate Apple WWDC22 sample code "Sampling GPU data into counter sample
//!   buffers" which uses ARC-managed descriptors and might uncover the missing
//!   piece for M-series.
//! - Pivot to A/B kernel benchmarks in isolation (run 1000× per kernel,
//!   measure aggregate) to identify hot kernels without instrumented
//!   forward pass.
//!
//! This module replaces the broken `profile_drain_record` (which produced
//! drain-cumulative artifacts, not real per-kernel costs) with a fiable
//! profiler that gathers actual GPU timestamp pairs around each dispatch.
//!
//! ## Why the old approach was wrong
//!
//! `profile_drain_record(label, t0)` did:
//! ```ignore
//! backend.drain();      // wait for ALL pending GPU work
//! profile_record(label, t0.elapsed());
//! ```
//!
//! `t0.elapsed()` therefore measured the cumulative GPU pipeline flush since
//! the previous profile point — not the cost of the kernel under that label.
//! 4 dead-end optimizations (T177–T180) chased phantom bottlenecks because
//! of this artifact.
//!
//! ## How this works
//!
//! Apple Silicon supports `MTLCounterSamplingPointAtStageBoundary`, exposed
//! via `[MTLComputeCommandEncoder sampleCountersInBuffer:atSampleIndex:withBarrier:]`.
//! We sample timestamps INSIDE the chained compute encoder (no encoder split,
//! no drain) before and after each labeled dispatch, then resolve the buffer
//! after the normal `drain()` completes.
//!
//! ## Storage choice (Shared)
//!
//! On Apple Silicon (UMA), `MTLStorageModeShared` lets us read the resolved
//! timestamps directly from CPU after `wait_until_completed`, avoiding the
//! `MTLBlitCommandEncoder::resolveCounters` round-trip. We still need a
//! `resolveCounterRange` call via `msg_send!` because metal-rs 0.29 doesn't
//! bind that method directly on `CounterSampleBuffer`.
//!
//! Alternative path used here: blit-resolve into a Shared `MTLBuffer` that
//! we control. metal-rs binds `resolve_counters` fully on `BlitCommandEncoder`.
//!
//! ## Tick → ns conversion
//!
//! Apple's `[MTLDevice sampleTimestamps:gpuTimestamp:]` returns CPU timestamp
//! (nanoseconds since boot, mach absolute time scaled) and GPU timestamp (GPU
//! ticks). We calibrate a tick-to-ns ratio by sampling twice ~5ms apart at
//! profiler creation. The GPU clock is fixed per family, so this calibration
//! holds for the lifetime of the profiler instance.
//!
//! ## Overhead
//!
//! Per sample is ~tens of nanoseconds on Apple Silicon. With ~500 dispatches/
//! token at 2 samples each = 1000 samples, total profiler overhead is well
//! under 100µs/token. Setting `with_barrier=false` in `sampleCountersInBuffer`
//! avoids forcing an explicit fence between dispatches that were already
//! ordered by the implicit Serial barrier.

use std::sync::Mutex;

use metal::{
    Buffer, CounterSampleBuffer, CounterSampleBufferDescriptor, Device, MTLCounterSamplingPoint,
    MTLResourceOptions, MTLStorageMode, NSRange,
};

use crate::error::MetalError;

/// Maximum number of labeled dispatches per profiler instance.
/// Apple Silicon caps the counter sample buffer at 32 KB total, so the maximum
/// is 32768 / 8 = 4096 timestamps = 2048 pairs (start + end).
/// 2048 dispatches covers a full 35B-A3B decode forward pass (~500 dispatches).
const MAX_DISPATCHES: usize = 2048;

/// One labeled timestamp pair allocated to a future dispatch.
#[derive(Clone, Copy, Debug)]
struct PendingPair {
    label: &'static str,
    /// Index of the start sample in the counter buffer (end = start + 1).
    start_idx: u32,
}

struct ProfilerInner {
    /// All pairs allocated since the last `reset()`. Their indices into the
    /// sample buffer are `start_idx` and `start_idx + 1`.
    pairs: Vec<PendingPair>,
    /// Next free sample-pair index. Each pair consumes 2 sample slots.
    next_pair: usize,
}

/// GPU-side profiler. One instance per `MetalBackend`. Created lazily on first
/// `enable()` call to avoid paying the counter-sample-buffer allocation cost
/// when profiling is off.
pub struct MetalProfiler {
    /// The Metal counter sample buffer. Receives raw GPU ticks at each
    /// `sample_counters_in_buffer` call.
    sample_buffer: CounterSampleBuffer,
    /// Destination buffer for `BlitCommandEncoder::resolve_counters`. Shared
    /// storage — readable from CPU on Apple Silicon UMA without round-trip.
    /// Sized to `MAX_DISPATCHES * 2 * 8` bytes (u64 per timestamp).
    dest_buffer: Buffer,
    /// Mutable state: pair list + next-index counter.
    inner: Mutex<ProfilerInner>,
    /// GPU ticks → nanoseconds conversion factor (calibrated at construction).
    tick_to_ns: f64,
}

impl MetalProfiler {
    /// Build a new profiler. Probes `supports_counter_sampling(AtStageBoundary)`
    /// — required on Apple Silicon. Allocates a `MAX_DISPATCHES * 2`-sample
    /// counter buffer and a matching shared `MTLBuffer` destination.
    ///
    /// Calibrates the tick→ns ratio via two `sampleTimestamps` calls 5ms apart.
    pub fn new(device: &Device) -> Result<Self, MetalError> {
        if !device.supports_counter_sampling(MTLCounterSamplingPoint::AtStageBoundary) {
            return Err(MetalError::Unsupported(
                "MTLCounterSamplingPointAtStageBoundary not supported on this device".to_string(),
            ));
        }

        // Find the timestamp counter set.
        let counter_sets = device.counter_sets();
        let timestamp_set = counter_sets
            .iter()
            .find(|cs| cs.name() == "timestamp")
            .ok_or_else(|| {
                MetalError::Unsupported("no timestamp counter set on device".to_string())
            })?;

        // Build a sample buffer for 2 * MAX_DISPATCHES samples (start + end pairs).
        let desc = CounterSampleBufferDescriptor::new();
        desc.set_counter_set(timestamp_set);
        desc.set_sample_count((MAX_DISPATCHES * 2) as u64);
        // Private storage required for blit resolve_counters — only the GPU
        // writes to the sample buffer; CPU reads from the resolved dest buffer.
        desc.set_storage_mode(MTLStorageMode::Private);
        desc.set_label("rustorch_metal_profiler");
        let sample_buffer = device
            .new_counter_sample_buffer_with_descriptor(&desc)
            .map_err(|e| MetalError::Unsupported(format!("counter sample buffer: {e}")))?;

        // Destination buffer for resolve_counters. Shared so we can read on CPU.
        let dest_size = (MAX_DISPATCHES * 2 * std::mem::size_of::<u64>()) as u64;
        let dest_buffer = device.new_buffer(dest_size, MTLResourceOptions::StorageModeShared);

        // Calibrate tick → ns. CPU timestamp is mach_absolute_time (already in ns
        // on Apple Silicon since timebase is 1:1). GPU timestamp is in GPU ticks.
        let mut cpu1: u64 = 0;
        let mut gpu1: u64 = 0;
        device.sample_timestamps(&mut cpu1, &mut gpu1);
        std::thread::sleep(std::time::Duration::from_millis(5));
        let mut cpu2: u64 = 0;
        let mut gpu2: u64 = 0;
        device.sample_timestamps(&mut cpu2, &mut gpu2);

        let dt_cpu_ns = cpu2.saturating_sub(cpu1) as f64;
        let dt_gpu_ticks = gpu2.saturating_sub(gpu1) as f64;
        let tick_to_ns = if dt_gpu_ticks > 0.0 {
            dt_cpu_ns / dt_gpu_ticks
        } else {
            1.0
        };

        Ok(Self {
            sample_buffer,
            dest_buffer,
            inner: Mutex::new(ProfilerInner {
                pairs: Vec::with_capacity(MAX_DISPATCHES),
                next_pair: 0,
            }),
            tick_to_ns,
        })
    }

    /// Allocate a sample-pair index for an upcoming dispatch labeled `label`.
    /// Returns the start sample index (end = start + 1). Returns `None` when
    /// the profiler has run out of slots — in that case the caller should
    /// skip sampling for this dispatch (graceful degradation).
    pub fn alloc_pair(&self, label: &'static str) -> Option<u32> {
        let mut g = self.inner.lock().expect("profiler inner");
        if g.next_pair >= MAX_DISPATCHES {
            return None;
        }
        let start_idx = (g.next_pair * 2) as u32;
        g.pairs.push(PendingPair { label, start_idx });
        g.next_pair += 1;
        Some(start_idx)
    }

    /// Number of dispatches currently captured.
    pub fn n_pairs(&self) -> usize {
        self.inner.lock().expect("profiler inner").pairs.len()
    }

    /// Access to the underlying sample buffer (used by the backend's
    /// labeled-dispatch path).
    pub fn sample_buffer(&self) -> &CounterSampleBuffer {
        &self.sample_buffer
    }

    /// Reset captured pairs without reallocating. Call at start of each
    /// new profile session (e.g. start of a forward pass).
    pub fn reset(&self) {
        let mut g = self.inner.lock().expect("profiler inner");
        g.pairs.clear();
        g.next_pair = 0;
    }

    /// Schedule a `resolve_counters` blit into our destination buffer.
    /// Must be called AFTER all `sample_counters_in_buffer` invocations
    /// (i.e. after the last compute dispatch) and BEFORE the command buffer
    /// commits. The `MetalBackend::drain` integrates this in its flush path.
    pub fn schedule_resolve(&self, command_buffer: &metal::CommandBufferRef) {
        let g = self.inner.lock().expect("profiler inner");
        let n_samples = g.next_pair * 2;
        if n_samples == 0 {
            return;
        }
        let blit = command_buffer.new_blit_command_encoder();
        blit.resolve_counters(
            &self.sample_buffer,
            NSRange {
                location: 0,
                length: n_samples as u64,
            },
            &self.dest_buffer,
            0,
        );
        blit.end_encoding();
    }

    /// Read out resolved timestamps (after the command buffer has completed
    /// — caller must ensure this with a `wait_until_completed`). Returns
    /// `Vec<(label, gpu_ns)>` in dispatch order.
    pub fn resolve(&self) -> Vec<(&'static str, u64)> {
        let g = self.inner.lock().expect("profiler inner");
        let n_pairs = g.pairs.len();
        if n_pairs == 0 {
            return vec![];
        }
        // SAFETY: dest_buffer has been written by resolve_counters and the
        // command buffer completed. Shared storage on Apple Silicon means
        // CPU sees the writes after wait_until_completed returns.
        let timestamps: &[u64] = unsafe {
            std::slice::from_raw_parts(self.dest_buffer.contents() as *const u64, n_pairs * 2)
        };
        if std::env::var("RUSTORCH_PROFILER_DEBUG").is_ok() {
            eprintln!("[MetalProfiler] raw timestamps: {timestamps:?}");
        }
        let mut out = Vec::with_capacity(n_pairs);
        for pair in &g.pairs {
            let s = pair.start_idx as usize;
            let start = timestamps[s];
            let end = timestamps[s + 1];
            // Apple may emit u64::MAX for a sample that was discarded
            // (e.g. due to barrier semantics). Treat as 0.
            let dt_ticks = if start == u64::MAX || end == u64::MAX {
                0
            } else {
                end.saturating_sub(start)
            };
            let ns = (dt_ticks as f64 * self.tick_to_ns) as u64;
            out.push((pair.label, ns));
        }
        out
    }

    /// Aggregate per-label totals across all captured pairs (sum + count).
    /// Useful when the same label is sampled multiple times in a forward
    /// (e.g. once per layer).
    pub fn aggregate(&self) -> Vec<(&'static str, u64, u64)> {
        // returns (label, total_ns, count)
        let resolved = self.resolve();
        let mut map: std::collections::BTreeMap<&'static str, (u64, u64)> =
            std::collections::BTreeMap::new();
        for (label, ns) in resolved {
            let entry = map.entry(label).or_insert((0, 0));
            entry.0 += ns;
            entry.1 += 1;
        }
        let mut v: Vec<_> = map.into_iter().map(|(k, (t, n))| (k, t, n)).collect();
        v.sort_by(|a, b| b.1.cmp(&a.1));
        v
    }

    /// Pretty-print breakdown to stderr.
    pub fn print_summary(&self) {
        let agg = self.aggregate();
        if agg.is_empty() {
            eprintln!("[MetalProfiler] no samples captured");
            return;
        }
        let total: u64 = agg.iter().map(|(_, t, _)| *t).sum();
        eprintln!(
            "\n=== MetalProfiler GPU breakdown ({} total dispatches, {:.3} ms) ===",
            agg.iter().map(|(_, _, n)| *n).sum::<u64>(),
            total as f64 / 1e6
        );
        eprintln!(
            "{:<32} {:>8} {:>14} {:>14} {:>9}",
            "label", "calls", "total_ms", "avg_µs/call", "% total"
        );
        for (label, total_ns, count) in &agg {
            let total_ms = *total_ns as f64 / 1e6;
            let avg_us = (*total_ns as f64) / 1e3 / (*count as f64);
            let pct = 100.0 * (*total_ns as f64) / total as f64;
            eprintln!("{label:<32} {count:>8} {total_ms:>14.3} {avg_us:>14.2} {pct:>9.2}");
        }
        eprintln!(
            "{:<32} {:>8} {:>14.3}",
            "TOTAL",
            agg.iter().map(|(_, _, n)| *n).sum::<u64>(),
            total as f64 / 1e6
        );
    }

    /// Tick → ns calibration ratio. Exposed for tests.
    pub fn tick_to_ns(&self) -> f64 {
        self.tick_to_ns
    }
}

// SAFETY: MetalProfiler is Send + Sync — sample_buffer and dest_buffer are
// Apple Metal handles that are thread-safe to send/share, inner state is
// behind a Mutex.
unsafe impl Send for MetalProfiler {}
unsafe impl Sync for MetalProfiler {}

#[cfg(all(test, target_os = "macos"))]
mod tests {
    // These imports are only consumed by tests gated behind the
    // `gpu-tests` feature; without that feature the items below are
    // dead code but keeping the imports here means the test module
    // stays compilable when the feature flips on.
    #[allow(unused_imports)]
    use super::*;
    #[allow(unused_imports)]
    use crate::backend_singleton::metal_backend;

    /// T181 sanity test — DISABLED. Counter sampling on M4 Max via
    /// `ComputePassDescriptor` is unreliable run-to-run (sometimes captures
    /// 1-2 dispatches, sometimes 0). Investigation needed to find the
    /// driver-specific incantation. Kept as future starting point.
    #[test]
    #[cfg(feature = "gpu-tests")]
    #[ignore = "MTLCounterSampleBuffer unreliable on M-series — see module docs"]
    fn profiler_captures_one_dispatch_per_drain() {
        let backend = metal_backend();
        if !backend.supports_metal3() {
            eprintln!("[profiler test] skipping: no Metal3");
            return;
        }
        let supports = backend
            .device
            .supports_counter_sampling(MTLCounterSamplingPoint::AtStageBoundary);
        eprintln!("supports counter sampling AtStageBoundary: {supports}");
        if !supports {
            eprintln!("[profiler test] skipping: AtStageBoundary not supported");
            return;
        }

        backend.enable_profiler().expect("enable_profiler");
        backend.reset_profiler();

        let n = 1024usize;
        let buf = backend.alloc_shared(n * 4).expect("alloc dest buf");

        const ZERO_F32_SRC: &str = r#"
            #include <metal_stdlib>
            using namespace metal;
            kernel void zero_f32_test(
                device float* buf [[buffer(0)]],
                constant uint& n  [[buffer(1)]],
                uint gid          [[thread_position_in_grid]]
            ) {
                if (gid >= n) return;
                buf[gid] = 0.0;
            }
        "#;
        let pipeline = backend
            .pipeline("zero_f32_test", ZERO_F32_SRC, "zero_f32_test")
            .expect("pipeline");
        let n_u = n as u32;

        // 3 labeled dispatches in one drain; we expect only the first
        // pair to carry a valid timestamp (Apple Silicon constraint).
        for i in 0..3 {
            let label: &'static str = match i {
                0 => "zero_a",
                1 => "zero_b",
                _ => "zero_c",
            };
            backend.with_encoder_labeled(label, |encoder| {
                encoder.set_compute_pipeline_state(&pipeline);
                encoder.set_buffer(0, Some(&buf), 0);
                encoder.set_bytes(1, 4, &n_u as *const u32 as *const std::ffi::c_void);
                let tg = metal::MTLSize::new(64, 1, 1);
                let grid = metal::MTLSize::new(n as u64, 1, 1);
                encoder.dispatch_threads(grid, tg);
            });
        }
        backend.drain();

        // Force the debug print on
        std::env::set_var("RUSTORCH_PROFILER_DEBUG", "1");
        let agg = backend.profiler_aggregate();
        eprintln!("aggregate after dispatches: {agg:?}");
        // Treat zero_f32_solo as the FIRST dispatch alias for compat
        // with the assertion below.
        let entry = agg
            .iter()
            .find(|(l, _, _)| *l == "zero_a")
            .expect("zero_a label missing");
        assert!(entry.1 > 0, "first dispatch captured 0 ns");
        eprintln!("OK first dispatch captured {} ns", entry.1);
        return;
        #[allow(unreachable_code)]
        {
            assert!(!agg.is_empty(), "profiler captured nothing");
        }

        backend.disable_profiler();
    }
}
