//! T172 — `AsyncAmxExecutor` (Innovation 1, Days 2-3)
//!
//! Concurrent CPU AMX (Apple Accelerate) executor for offloading small F32
//! matmul ops in parallel with GPU compute. Designed to enable hybrid
//! GPU+CPU forward where the CPU AMX channel runs independently of the GPU
//! Metal channel — both use Apple Silicon unified memory.
//!
//! ## Two API levels
//!
//! 1. **`submit(AmxJob)`** (Day 2) : pure CPU concurrency, no GPU sync.
//!    Caller must ensure input coherency (e.g. `backend.drain()` first)
//!    and wait for completion before reading outputs.
//!
//! 2. **`submit_with_sync(AmxJob, wait_value, signal_value)`** (Day 3) :
//!    integrates `metal::SharedEvent` for fine-grained GPU↔CPU sync.
//!    Worker thread waits on `event.signaled_value() >= wait_value`
//!    before running AMX, then `set_signaled_value(signal_value)` after.
//!    Caller pairs this with `command_buffer.encode_signal_event(..., wait_value)`
//!    before and `encode_wait_for_event(..., signal_value)` after.
//!
//! ## Architecture
//!
//! - One worker thread spawned at executor construction.
//! - Jobs submitted via lock-free `mpsc::Sender<AmxJob>`.
//! - Each job carries raw pointers to input/output buffers (caller
//!   guarantees lifetime + coherency) plus shape (k, n) and a completion
//!   counter.
//! - Caller can `wait_all()` to block until all submitted jobs have run.
//!
//! ## Bench reference
//!
//! - Standalone AMX `cblas_sgemv` (d=2048, n=256, F32) : **9.43 µs/call**
//! - GPU sgemv same shape (per profile) : ~50-70 µs/call
//! - Async pattern PoC (40 layers): **2.52× speedup vs sync**
//!
//! Notes : `1ce4dd46` (AMX bench), `14e2a530` (PoC + Day 1-5 plan).

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc;
use std::sync::Arc;
use std::thread::{self, JoinHandle};

use metal::SharedEvent;

/// Wraps `metal::SharedEvent` with `Send + Sync` markers so it can be moved
/// into the worker thread. SharedEvent's whole purpose is GPU↔CPU sync, so
/// it's logically thread-safe (Apple-documented). The metal-rs binding
/// doesn't add the markers automatically (foreign_obj_type macro is
/// conservative).
struct SharedEventHandle(SharedEvent);

// SAFETY: Apple's MTLSharedEvent is documented as thread-safe for
// signaledValue/setSignaledValue. The Objective-C runtime handles the
// underlying refcount.
unsafe impl Send for SharedEventHandle {}
unsafe impl Sync for SharedEventHandle {}

impl SharedEventHandle {
    fn signaled_value(&self) -> u64 {
        self.0.signaled_value()
    }
    fn set_signaled_value(&self, v: u64) {
        self.0.set_signaled_value(v);
    }
    fn inner(&self) -> &SharedEvent {
        &self.0
    }
}

/// A single AMX matmul job: `out[n] = w[n, k] @ h[k]` (row-major, F32).
///
/// Pointers are raw to avoid Metal `Buffer` Send constraints. Caller
/// guarantees that the underlying buffers outlive the job and that the
/// memory is coherent at submit time (no concurrent GPU writes pending).
pub struct AmxJob {
    /// Pointer to input vector `h[k]` (F32).
    pub h_ptr: *const f32,
    /// Pointer to weight matrix `w[n, k]` row-major (F32).
    pub w_ptr: *const f32,
    /// Pointer to output `out[n]` (F32). Will be fully overwritten.
    pub out_ptr: *mut f32,
    /// Inner dim (length of `h` / cols of `w`).
    pub k: usize,
    /// Output dim (rows of `w` / length of `out`).
    pub n: usize,
}

// SAFETY: AmxJob carries raw pointers; the caller is responsible for
// ensuring no concurrent access during the executor's lifetime of the job.
// We mark it Send so it can cross the channel into the worker thread.
unsafe impl Send for AmxJob {}

/// Concurrent AMX executor. Spawns one CPU worker thread that polls a
/// channel for jobs and runs `cblas_sgemv` on each. Optionally synced to a
/// `metal::SharedEvent` for fine-grained GPU↔CPU coordination.
pub struct AsyncAmxExecutor {
    job_tx: Option<mpsc::Sender<Message>>,
    submitted: Arc<AtomicU64>,
    completed: Arc<AtomicU64>,
    worker: Option<JoinHandle<()>>,
    /// Optional shared event for GPU↔CPU sync. When `None`, the worker
    /// only does CPU concurrency (Day 2 path).
    sync_event: Option<Arc<SharedEventHandle>>,
    /// Monotonic counter for generating event values. Caller should use
    /// `next_event_value()` to allocate fresh values for each
    /// signal/wait pair.
    next_value: AtomicU64,
}

/// A job paired with optional GPU sync values. If `wait_value > 0`, the
/// worker spins on `event.signaled_value() >= wait_value` before running
/// AMX. After AMX completes, if `signal_value > 0`, the worker calls
/// `event.set_signaled_value(signal_value)` to unblock GPU consumers.
struct SyncedJob {
    job: AmxJob,
    /// Worker waits for `event.signaled_value() >= wait_value` (0 = no wait).
    wait_value: u64,
    /// Worker calls `event.set_signaled_value(signal_value)` post-AMX (0 = no signal).
    signal_value: u64,
}

// SAFETY: same as AmxJob — caller manages buffer lifetime.
unsafe impl Send for SyncedJob {}

/// Internal message between submitter and worker.
enum Message {
    Job(SyncedJob),
    Shutdown,
}

impl AsyncAmxExecutor {
    /// Create a new executor with one worker thread, no GPU sync (Day 2 mode).
    pub fn new() -> Self {
        Self::new_inner(None)
    }

    /// Create a new executor with a shared `metal::SharedEvent` for GPU sync
    /// (Day 3 mode). Use `submit_with_sync` to coordinate with GPU command
    /// buffers via `encode_signal_event` / `encode_wait_for_event`.
    pub fn with_event(event: SharedEvent) -> Self {
        Self::new_inner(Some(Arc::new(SharedEventHandle(event))))
    }

    fn new_inner(sync_event: Option<Arc<SharedEventHandle>>) -> Self {
        let (tx, rx) = mpsc::channel::<Message>();
        let submitted = Arc::new(AtomicU64::new(0));
        let completed = Arc::new(AtomicU64::new(0));
        let completed_for_worker = Arc::clone(&completed);
        let event_for_worker = sync_event.as_ref().map(Arc::clone);

        let worker = thread::Builder::new()
            .name("rustorch-amx-worker".to_string())
            .spawn(move || worker_loop(rx, completed_for_worker, event_for_worker))
            .expect("spawn AMX worker thread");

        Self {
            job_tx: Some(tx),
            submitted,
            completed,
            worker: Some(worker),
            sync_event,
            next_value: AtomicU64::new(1),
        }
    }

    /// Submit a job to be processed asynchronously, no GPU sync.
    /// Returns immediately. Use `wait_all()` to block until all submitted
    /// jobs have completed.
    pub fn submit(&self, job: AmxJob) {
        self.submit_inner(SyncedJob {
            job,
            wait_value: 0,
            signal_value: 0,
        });
    }

    /// Submit a job with GPU sync. Worker waits for
    /// `event.signaled_value() >= wait_value` before running AMX, then
    /// calls `event.set_signaled_value(signal_value)` after completion.
    ///
    /// Caller pairs this with on the GPU command buffer:
    /// 1. `cmd.encode_signal_event(&exec.event(), wait_value)` after the
    ///    GPU dispatch that produces the AMX input
    /// 2. `cmd.encode_wait_for_event(&exec.event(), signal_value)` before
    ///    the GPU dispatch that consumes the AMX output
    ///
    /// Requires the executor was created with `with_event(...)`.
    pub fn submit_with_sync(&self, job: AmxJob, wait_value: u64, signal_value: u64) {
        debug_assert!(
            self.sync_event.is_some(),
            "submit_with_sync requires AsyncAmxExecutor::with_event(...)"
        );
        self.submit_inner(SyncedJob {
            job,
            wait_value,
            signal_value,
        });
    }

    fn submit_inner(&self, synced: SyncedJob) {
        self.submitted.fetch_add(1, Ordering::Release);
        if let Some(tx) = &self.job_tx {
            // Channel send is infallible while the worker is alive.
            let _ = tx.send(Message::Job(synced));
        }
    }

    /// Allocate a fresh, monotonically increasing value for use as a
    /// signal/wait token. Returns `(wait_value, signal_value)` pair where
    /// `signal_value = wait_value + 1`.
    pub fn next_event_pair(&self) -> (u64, u64) {
        let v = self.next_value.fetch_add(2, Ordering::AcqRel);
        (v, v + 1)
    }

    /// Returns the shared event so callers can encode signal/wait on
    /// their command buffers. Panics if executor has no event.
    pub fn event(&self) -> &SharedEvent {
        self.sync_event
            .as_ref()
            .expect("AsyncAmxExecutor was created without an event")
            .inner()
    }

    /// Block until all submitted jobs have been processed.
    /// Spin-waits on the `completed` counter — adequate for the typical
    /// pattern where jobs take 1-50 µs.
    pub fn wait_all(&self) {
        let target = self.submitted.load(Ordering::Acquire);
        while self.completed.load(Ordering::Acquire) < target {
            std::hint::spin_loop();
        }
    }

    /// How many jobs have been submitted in total.
    pub fn n_submitted(&self) -> u64 {
        self.submitted.load(Ordering::Acquire)
    }

    /// How many jobs have been completed in total.
    pub fn n_completed(&self) -> u64 {
        self.completed.load(Ordering::Acquire)
    }
}

impl Drop for AsyncAmxExecutor {
    fn drop(&mut self) {
        // Signal shutdown and join the worker.
        if let Some(tx) = self.job_tx.take() {
            let _ = tx.send(Message::Shutdown);
        }
        if let Some(handle) = self.worker.take() {
            let _ = handle.join();
        }
    }
}

impl Default for AsyncAmxExecutor {
    fn default() -> Self {
        Self::new()
    }
}

fn worker_loop(
    rx: mpsc::Receiver<Message>,
    completed: Arc<AtomicU64>,
    sync_event: Option<Arc<SharedEventHandle>>,
) {
    while let Ok(msg) = rx.recv() {
        match msg {
            Message::Job(synced) => {
                // If we have a sync event and a non-zero wait value, spin
                // until the GPU has signaled that the input is ready.
                if synced.wait_value > 0 {
                    if let Some(ev) = &sync_event {
                        while ev.signaled_value() < synced.wait_value {
                            std::hint::spin_loop();
                        }
                    }
                }

                run_amx_sgemv(&synced.job);

                // Signal completion to the GPU side if requested.
                if synced.signal_value > 0 {
                    if let Some(ev) = &sync_event {
                        ev.set_signaled_value(synced.signal_value);
                    }
                }

                completed.fetch_add(1, Ordering::Release);
            },
            Message::Shutdown => break,
        }
    }
}

/// Run a single AMX sgemv via Apple Accelerate. `out[n] = w[n, k] @ h[k]`
/// with row-major `w`.
fn run_amx_sgemv(job: &AmxJob) {
    use rustorch_cpu::accelerate::{cblas_sgemv, CBLAS_NO_TRANS, CBLAS_ROW_MAJOR};
    // SAFETY: pointers are valid for the duration of the job per caller
    // contract (see AmxJob doc). cblas_sgemv reads K elements from h_ptr,
    // K*N from w_ptr, writes N to out_ptr.
    unsafe {
        cblas_sgemv(
            CBLAS_ROW_MAJOR,
            CBLAS_NO_TRANS,
            job.n as i32,
            job.k as i32,
            1.0,
            job.w_ptr,
            job.k as i32,
            job.h_ptr,
            1,
            0.0,
            job.out_ptr,
            1,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Naive CPU reference for sgemv : `out[i] = sum_k w[i, k] * h[k]`.
    fn naive_sgemv(h: &[f32], w: &[f32], n: usize, k: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; n];
        for i in 0..n {
            let mut s = 0.0f32;
            for j in 0..k {
                s += w[i * k + j] * h[j];
            }
            out[i] = s;
        }
        out
    }

    #[test]
    fn submit_and_wait_single_job_matches_naive() {
        let k = 256;
        let n = 64;
        let h: Vec<f32> = (0..k).map(|i| ((i as f32) * 0.01).sin()).collect();
        let w: Vec<f32> = (0..n * k).map(|i| ((i as f32) * 0.005).cos()).collect();
        let mut out = vec![0.0f32; n];
        let expected = naive_sgemv(&h, &w, n, k);

        let exec = AsyncAmxExecutor::new();
        let job = AmxJob {
            h_ptr: h.as_ptr(),
            w_ptr: w.as_ptr(),
            out_ptr: out.as_mut_ptr(),
            k,
            n,
        };
        exec.submit(job);
        exec.wait_all();

        for i in 0..n {
            let denom = expected[i].abs().max(1e-6);
            let rel = (expected[i] - out[i]).abs() / denom;
            assert!(
                rel < 1e-4,
                "row {i}: expected {} got {} (rel {:.3e})",
                expected[i],
                out[i],
                rel
            );
        }

        assert_eq!(exec.n_submitted(), 1);
        assert_eq!(exec.n_completed(), 1);
    }

    #[test]
    fn submit_many_jobs_all_complete() {
        let k = 128;
        let n = 32;
        let h: Vec<f32> = (0..k).map(|i| ((i as f32) * 0.01).sin()).collect();
        let w: Vec<f32> = (0..n * k).map(|i| ((i as f32) * 0.005).cos()).collect();
        let n_jobs = 50;
        let mut outs: Vec<Vec<f32>> = (0..n_jobs).map(|_| vec![0.0f32; n]).collect();

        let exec = AsyncAmxExecutor::new();
        for out in outs.iter_mut() {
            let job = AmxJob {
                h_ptr: h.as_ptr(),
                w_ptr: w.as_ptr(),
                out_ptr: out.as_mut_ptr(),
                k,
                n,
            };
            exec.submit(job);
        }
        exec.wait_all();

        let expected = naive_sgemv(&h, &w, n, k);
        for (j, out) in outs.iter().enumerate() {
            for i in 0..n {
                let denom = expected[i].abs().max(1e-6);
                let rel = (expected[i] - out[i]).abs() / denom;
                assert!(
                    rel < 1e-4,
                    "job {j}, row {i}: expected {} got {} (rel {:.3e})",
                    expected[i],
                    out[i],
                    rel
                );
            }
        }
        assert_eq!(exec.n_submitted(), n_jobs as u64);
        assert_eq!(exec.n_completed(), n_jobs as u64);
    }

    /// T172 Day 3 — verify that submit_with_sync correctly waits for the
    /// shared event before running AMX, and signals after completion.
    /// Simulates GPU by directly setting/checking the event value from
    /// the test thread.
    #[test]
    fn submit_with_sync_waits_and_signals() {
        let backend = crate::backend_singleton::metal_backend();
        let event = backend.device.new_shared_event();
        let exec = AsyncAmxExecutor::with_event(event);

        let k = 64;
        let n = 32;
        let h: Vec<f32> = (0..k).map(|i| ((i as f32) * 0.01).sin()).collect();
        let w: Vec<f32> = (0..n * k).map(|i| ((i as f32) * 0.005).cos()).collect();
        let mut out = vec![0.0f32; n];

        // Allocate a fresh event-value pair.
        let (wait_v, signal_v) = exec.next_event_pair();
        assert_eq!(signal_v, wait_v + 1);

        // Submit before signaling — worker should spin on event.
        exec.submit_with_sync(
            AmxJob {
                h_ptr: h.as_ptr(),
                w_ptr: w.as_ptr(),
                out_ptr: out.as_mut_ptr(),
                k,
                n,
            },
            wait_v,
            signal_v,
        );

        // Worker should still be waiting (no AMX done yet).
        std::thread::sleep(std::time::Duration::from_millis(2));
        assert_eq!(exec.n_completed(), 0, "worker should be waiting on event");

        // "GPU" signals that input is ready — simulated by setting event value.
        exec.event().set_signaled_value(wait_v);

        // Worker now runs AMX. Wait for completion.
        exec.wait_all();
        assert_eq!(exec.n_completed(), 1);

        // Verify output (parity vs naive).
        let expected = naive_sgemv(&h, &w, n, k);
        for i in 0..n {
            let denom = expected[i].abs().max(1e-6);
            let rel = (expected[i] - out[i]).abs() / denom;
            assert!(rel < 1e-4, "row {i}: rel {rel:.3e}");
        }

        // Verify the worker signaled the completion value.
        assert!(
            exec.event().signaled_value() >= signal_v,
            "expected event signaled to {} got {}",
            signal_v,
            exec.event().signaled_value()
        );
    }

    /// Sanity: dropping the executor cleanly terminates the worker thread
    /// without panic.
    #[test]
    fn drop_cleans_up_worker() {
        let exec = AsyncAmxExecutor::new();
        // Submit a few jobs to ensure the worker is active.
        let h = [1.0f32; 16];
        let w = [1.0f32; 16 * 4];
        let mut out = [0.0f32; 4];
        exec.submit(AmxJob {
            h_ptr: h.as_ptr(),
            w_ptr: w.as_ptr(),
            out_ptr: out.as_mut_ptr(),
            k: 16,
            n: 4,
        });
        exec.wait_all();
        // Implicit drop here — worker should join cleanly.
    }
}
