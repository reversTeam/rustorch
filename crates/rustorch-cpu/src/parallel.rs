//! `parallel_for` wrapper over rayon with a WASM serial fallback
//! (RFC-0006, P1.2 task `parallel_for`).
//!
//! On native targets, work is split across rayon threads via
//! [`rayon::iter::IntoParallelIterator`]. On `wasm32-*` (where rayon is
//! unavailable in the workspace) the same API runs serially.
//!
//! ```
//! use rustorch_cpu::parallel::parallel_for;
//!
//! let mut buf = vec![0_i64; 1024];
//! parallel_for(0..buf.len(), 64, |chunk| {
//!     for i in chunk { /* per-element work — example: */ let _ = i; }
//! });
//! ```

use core::ops::Range;

/// Run `f` over disjoint chunks of `range`, each chunk handed to a
/// worker thread (or executed serially on WASM).
///
/// `chunk_hint` is the desired number of elements per chunk; the
/// implementation may round to a power of two or coalesce small
/// chunks if `chunk_hint < 8`.
///
/// Returns immediately when `range` is empty.
#[cfg(not(target_arch = "wasm32"))]
pub fn parallel_for<F>(range: Range<usize>, chunk_hint: usize, f: F)
where
    F: Fn(Range<usize>) + Sync + Send,
{
    use rayon::prelude::*;
    let total = range.end.saturating_sub(range.start);
    if total == 0 {
        return;
    }
    let chunk = effective_chunk_size(total, chunk_hint);
    let n_chunks = total.div_ceil(chunk);
    (0..n_chunks).into_par_iter().for_each(|i| {
        let start = range.start + i * chunk;
        let end = (start + chunk).min(range.end);
        f(start..end);
    });
}

/// Serial fallback used on `target_arch="wasm32"` builds.
#[cfg(target_arch = "wasm32")]
pub fn parallel_for<F>(range: Range<usize>, chunk_hint: usize, f: F)
where
    F: Fn(Range<usize>),
{
    let total = range.end.saturating_sub(range.start);
    if total == 0 {
        return;
    }
    let chunk = effective_chunk_size(total, chunk_hint);
    let n_chunks = total.div_ceil(chunk);
    for i in 0..n_chunks {
        let start = range.start + i * chunk;
        let end = (start + chunk).min(range.end);
        f(start..end);
    }
}

/// Returns the number of worker threads `parallel_for` will use.
#[cfg(not(target_arch = "wasm32"))]
pub fn num_threads() -> usize {
    rayon::current_num_threads()
}

/// On WASM there is no thread pool — always 1.
#[cfg(target_arch = "wasm32")]
pub fn num_threads() -> usize {
    1
}

/// Pick a chunk size given the total work and the user's hint. Floors
/// at 8 (sub-cache-line work isn't worth dispatching) and never
/// exceeds total. With a `0` hint, falls back to `total / num_threads`.
fn effective_chunk_size(total: usize, hint: usize) -> usize {
    let n = num_threads().max(1);
    let baseline = total.div_ceil(n);
    let raw = if hint == 0 { baseline } else { hint };
    raw.clamp(8.min(total).max(1), total.max(1))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn empty_range_is_no_op() {
        let counter = AtomicUsize::new(0);
        parallel_for(0..0, 16, |_chunk| {
            counter.fetch_add(1, Ordering::Relaxed);
        });
        assert_eq!(counter.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn single_thread_sum_matches_serial() {
        // Sum over a deterministic closure — no FP, so result is exact.
        let n = 10_000usize;
        let total = AtomicUsize::new(0);
        parallel_for(0..n, 64, |chunk| {
            let mut local = 0usize;
            for i in chunk {
                local += i;
            }
            total.fetch_add(local, Ordering::Relaxed);
        });
        let expected: usize = (0..n).sum();
        assert_eq!(total.load(Ordering::Relaxed), expected);
    }

    #[test]
    fn touches_each_index_exactly_once() {
        let n = 4096;
        let touched: Vec<AtomicUsize> = (0..n).map(|_| AtomicUsize::new(0)).collect();
        parallel_for(0..n, 32, |chunk| {
            for i in chunk {
                touched[i].fetch_add(1, Ordering::Relaxed);
            }
        });
        for (i, t) in touched.iter().enumerate() {
            assert_eq!(t.load(Ordering::Relaxed), 1, "index {i} touched != once");
        }
    }

    #[test]
    fn chunk_hint_zero_uses_baseline() {
        let n = 256;
        let chunks = std::sync::Mutex::new(Vec::<Range<usize>>::new());
        parallel_for(0..n, 0, |chunk| {
            chunks.lock().unwrap().push(chunk);
        });
        let mut got: Vec<_> = chunks.lock().unwrap().clone();
        got.sort_by_key(|r| r.start);
        // Coverage is complete and disjoint.
        let mut next = 0;
        for r in &got {
            assert_eq!(r.start, next);
            next = r.end;
        }
        assert_eq!(next, n);
    }

    #[test]
    fn parallel_result_matches_serial_for_addition() {
        let n = 100_000usize;
        let mut data = vec![0i64; n];
        for (i, x) in data.iter_mut().enumerate() {
            *x = i as i64;
        }
        // Parallel sum
        let par_total = AtomicUsize::new(0);
        parallel_for(0..n, 1024, |chunk| {
            let mut s: i64 = 0;
            for i in chunk {
                s += data[i];
            }
            par_total.fetch_add(s as usize, Ordering::Relaxed);
        });
        let serial: i64 = data.iter().sum();
        assert_eq!(par_total.load(Ordering::Relaxed) as i64, serial);
    }

    #[test]
    fn num_threads_is_at_least_one() {
        assert!(num_threads() >= 1);
    }

    #[test]
    fn chunk_size_clamped() {
        // Internal helper test: hint 0 with 1 thread → baseline = total.
        // We can't directly observe num_threads here cleanly, just sanity
        // that effective_chunk_size returns something reasonable.
        assert!(effective_chunk_size(1, 0) >= 1);
        assert!(effective_chunk_size(100, 5) <= 100);
    }
}
