//! Lookahead pool / draft-tree microbench (T246.7 P1.2).
//!
//! Regression guard for the CPU-side lookahead overhead: builds an
//! [`NGramPool`], inserts 10k tokens of synthetic generation, then
//! measures the per-call cost of [`NGramPool::query`] (1k random
//! prefixes) and the end-to-end
//! [`LookaheadManager::build_drafts`] cost (10k synthetic decode
//! steps). Reports hit rate, ns/op and the worst-case microsecond
//! cost.
//!
//! Run with:
//!
//! ```bash
//! cargo run --release -p rustorch-llm --example lookahead_pool_microbench
//! ```
//!
//! Gate (per RFC): `build_drafts` overhead must remain under 5 µs
//! on a single CPU core (verify pass on GPU is ~50 ms; we want pool
//! overhead to be noise).

use std::time::Instant;

use rustorch_llm::lookahead::{LookaheadManager, NGramPool};

const STREAM_LEN: usize = 10_000;
const NUM_QUERIES: usize = 1_000;
const VOCAB: u32 = 256; // small vocab boosts hit rate, mimics Zipfian token reuse

/// Linear-congruential PRNG (no_std-friendly, deterministic).
struct Lcg(u64);
impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next_u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        (self.0 >> 33) as u32
    }
    fn next_in(&mut self, max: u32) -> u32 {
        self.next_u32() % max
    }
}

fn synth_stream(len: usize, seed: u64) -> Vec<u32> {
    let mut rng = Lcg::new(seed);
    let mut out = Vec::with_capacity(len);
    for _ in 0..len {
        out.push(rng.next_in(VOCAB));
    }
    out
}

fn bench_pool() -> (f64, f64, f64) {
    let stream = synth_stream(STREAM_LEN, 0xDEAD_BEEF);
    let mut pool = NGramPool::new();

    // --- Insert phase: 10k tokens.
    let t0 = Instant::now();
    pool.insert(&stream);
    let insert_ns_per_token = t0.elapsed().as_nanos() as f64 / STREAM_LEN as f64;

    // --- Query phase: 1k random prefixes (drawn from the same vocab
    //     so we get realistic hit-rate behaviour).
    let mut rng = Lcg::new(0xCAFE_F00D);
    let prefixes: Vec<[u32; 2]> = (0..NUM_QUERIES)
        .map(|_| [rng.next_in(VOCAB), rng.next_in(VOCAB)])
        .collect();

    let t0 = Instant::now();
    let mut total_hits = 0usize;
    for p in &prefixes {
        let conts = pool.query(p);
        if !conts.is_empty() {
            total_hits += 1;
        }
    }
    let query_ns_per_op = t0.elapsed().as_nanos() as f64 / NUM_QUERIES as f64;
    let hit_rate = total_hits as f64 / NUM_QUERIES as f64;
    (insert_ns_per_token, query_ns_per_op, hit_rate)
}

fn bench_build_drafts() -> (f64, f64, f64) {
    let stream = synth_stream(STREAM_LEN, 0xFEED_FACE);
    let mut mgr = LookaheadManager::new(5, 5);

    // Warm pool with the first 1000 tokens (insertion budget).
    mgr.record_acceptance(&[], 0, &stream[..1000]);

    let t0 = Instant::now();
    let mut worst_ns = 0u128;
    for tok in stream.iter().copied().take(STREAM_LEN) {
        let t1 = Instant::now();
        let _tree = mgr.build_drafts(tok);
        let dt = t1.elapsed().as_nanos();
        if dt > worst_ns {
            worst_ns = dt;
        }
    }
    let total_ns = t0.elapsed().as_nanos() as f64;
    let avg_ns = total_ns / STREAM_LEN as f64;
    let stats = mgr.stats();
    (avg_ns, worst_ns as f64, stats.pool_hit_rate())
}

fn main() {
    println!("Lookahead pool microbench — T246.7 P1.2");
    println!("STREAM_LEN={STREAM_LEN}, NUM_QUERIES={NUM_QUERIES}, VOCAB={VOCAB}");
    println!();

    // 3-run avg for the pool ops.
    let mut sum_insert: f64 = 0.0;
    let mut sum_query: f64 = 0.0;
    let mut sum_hit: f64 = 0.0;
    let mut sum_build: f64 = 0.0;
    let mut sum_worst: f64 = 0.0;
    let mut sum_build_hit: f64 = 0.0;
    const RUNS: usize = 3;
    for _ in 0..RUNS {
        let (ins, qry, hr) = bench_pool();
        sum_insert += ins;
        sum_query += qry;
        sum_hit += hr;

        let (bd, worst, bhr) = bench_build_drafts();
        sum_build += bd;
        sum_worst = sum_worst.max(worst);
        sum_build_hit += bhr;
    }
    let avg_insert = sum_insert / RUNS as f64;
    let avg_query = sum_query / RUNS as f64;
    let avg_hit = sum_hit / RUNS as f64;
    let avg_build = sum_build / RUNS as f64;
    let avg_build_hit = sum_build_hit / RUNS as f64;

    println!("--- NGramPool ops (3-run avg) ---");
    println!("  insert : {avg_insert:.1} ns / token");
    println!(
        "  query  : {avg_query:.1} ns / op    (hit rate: {:.1}%)",
        avg_hit * 100.0
    );
    println!();
    println!("--- LookaheadManager.build_drafts (3-run avg over {STREAM_LEN} steps) ---");
    println!(
        "  avg : {avg_build:.1} ns / call ({:.2} µs)",
        avg_build / 1000.0
    );
    println!(
        "  worst (any single call across runs) : {sum_worst:.0} ns ({:.2} µs)",
        sum_worst / 1000.0
    );
    println!(
        "  pool hit rate over the run : {:.1}%",
        avg_build_hit * 100.0
    );
    println!();

    // Gate: build_drafts must average under 5 µs.
    let gate_ns: f64 = 5_000.0;
    if avg_build > gate_ns {
        eprintln!(
            "FAIL — build_drafts avg {avg_build:.0} ns exceeds the 5 µs gate (gate={gate_ns} ns)"
        );
        std::process::exit(1);
    }
    println!("PASS — build_drafts overhead under the 5 µs gate.");
}
