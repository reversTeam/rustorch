//! T172 — `AsyncAmxExecutor` (Innovation 1, Day 2)
//!
//! Concurrent CPU AMX (Apple Accelerate) executor for offloading small F32
//! matmul ops in parallel with GPU compute. Designed to enable hybrid
//! GPU+CPU forward where the CPU AMX channel runs independently of the GPU
//! Metal channel — both use Apple Silicon unified memory.
//!
//! ## Day 2 scope (this module)
//!
//! Pure CPU concurrency. NO Metal sync yet. Caller is responsible for
//! ensuring input buffers are coherent (e.g. via `backend.drain()` before
//! submit) and for waiting on completion before reading outputs.
//!
//! Day 3 will add `metal::SharedEvent` integration so the worker thread
//! waits on GPU events and signals back, removing the need for explicit
//! drains in the integration code.
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
/// channel for jobs and runs `cblas_sgemv` on each.
pub struct AsyncAmxExecutor {
    job_tx: Option<mpsc::Sender<Message>>,
    submitted: Arc<AtomicU64>,
    completed: Arc<AtomicU64>,
    worker: Option<JoinHandle<()>>,
}

/// Internal message between submitter and worker.
enum Message {
    Job(AmxJob),
    Shutdown,
}

impl AsyncAmxExecutor {
    /// Create a new executor with one worker thread.
    pub fn new() -> Self {
        let (tx, rx) = mpsc::channel::<Message>();
        let submitted = Arc::new(AtomicU64::new(0));
        let completed = Arc::new(AtomicU64::new(0));
        let completed_for_worker = Arc::clone(&completed);

        let worker = thread::Builder::new()
            .name("rustorch-amx-worker".to_string())
            .spawn(move || worker_loop(rx, completed_for_worker))
            .expect("spawn AMX worker thread");

        Self {
            job_tx: Some(tx),
            submitted,
            completed,
            worker: Some(worker),
        }
    }

    /// Submit a job to be processed asynchronously by the worker.
    /// Returns immediately. Use `wait_all()` to block until all submitted
    /// jobs have completed.
    pub fn submit(&self, job: AmxJob) {
        self.submitted.fetch_add(1, Ordering::Release);
        if let Some(tx) = &self.job_tx {
            // Channel send is infallible while the worker is alive.
            let _ = tx.send(Message::Job(job));
        }
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

fn worker_loop(rx: mpsc::Receiver<Message>, completed: Arc<AtomicU64>) {
    while let Ok(msg) = rx.recv() {
        match msg {
            Message::Job(job) => {
                run_amx_sgemv(&job);
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
