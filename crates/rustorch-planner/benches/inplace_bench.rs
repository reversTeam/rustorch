//! Benchmark: in-place rewriting on a 50-layer MLP fixture.
//!
//! Plan P3 task `In-place mutation detection` step #5: measure peak
//! memory savings of `plan_with_inplace` vs `plan` on a realistic
//! 50-layer trace. Two fixtures are exercised:
//!
//! 1. **Uniform MLP** — 50 chained `linear → relu` ops, every layer
//!    same hidden dim. Demonstrates the FFD-saturation case: in-place
//!    is NEUTRAL because FFD already packs the alternating pattern
//!    optimally.
//!
//! 2. **Varied MLP** — 50 layers with growing-then-shrinking hidden
//!    dims (autoencoder shape). Demonstrates the case where in-place
//!    wins by allowing big intermediates to be subsumed.
//!
//! Both benches print the savings ratio so the project tracks the
//! actual numbers over time.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rustorch_planner::lifetime::{LifetimeTable, TensorId};
use rustorch_planner::{plan, plan_with_inplace, InPlaceBlockers, InPlaceHint};

/// Build a chained `linear → relu` trace for `n_layers` with the
/// given per-layer activation sizes. Hints alias `linear_i` → `relu_i`
/// at the step where `linear_i` dies.
fn build_mlp(layer_bytes: &[u64]) -> (LifetimeTable, Vec<InPlaceHint>) {
    let mut t = LifetimeTable::new();
    let mut hints = Vec::new();
    let mut step = 0u32;
    let mut tensor_id = 0u64;
    let mut prev_relu: Option<TensorId> = None;
    for (layer_idx, bytes) in layer_bytes.iter().enumerate() {
        // linear_i — produced from prev_relu (consumes it)
        let linear = TensorId(tensor_id);
        tensor_id += 1;
        if let Some(prev) = prev_relu {
            t.record_use(prev, step).unwrap(); // consumes prev_relu
        }
        t.record_def(linear, step, *bytes);
        let linear_dies_at = step + 1;
        t.record_use(linear, linear_dies_at).unwrap();
        step = linear_dies_at;

        // relu_i — produced from linear_i (consumes it). This is the
        // in-place candidate.
        let relu = TensorId(tensor_id);
        tensor_id += 1;
        t.record_def(relu, step, *bytes);
        if layer_idx < layer_bytes.len() - 1 {
            // not the last layer — relu will be consumed by next linear
            let relu_dies_at = step + 1;
            t.record_use(relu, relu_dies_at).unwrap();
        }
        // Hint: linear → relu at step where linear dies.
        hints.push(InPlaceHint::new(linear, relu, step));
        prev_relu = Some(relu);

        // Advance step past relu's use (next linear born here).
        step += 1;
    }
    (t, hints)
}

fn report_savings(label: &str, t: &LifetimeTable, hints: &[InPlaceHint]) {
    let with_inplace = plan_with_inplace(t, hints, &InPlaceBlockers::new()).peak_bytes();
    let without = plan(t).peak_bytes();
    let saved = without.saturating_sub(with_inplace);
    let pct = if without == 0 {
        0.0
    } else {
        100.0 * saved as f64 / without as f64
    };
    eprintln!(
        "[{label}] peak: without={without}  with_inplace={with_inplace}  saved={saved} ({pct:.1}%)"
    );
}

fn bench_uniform_mlp(c: &mut Criterion) {
    // 50 layers, every layer 4096 bytes (a typical hidden activation).
    let layer_bytes: Vec<u64> = vec![4096; 50];
    let (t, hints) = build_mlp(&layer_bytes);
    report_savings("uniform-mlp-50", &t, &hints);
    c.bench_function("plan_with_inplace_uniform_mlp_50", |b| {
        b.iter(|| {
            let p = plan_with_inplace(black_box(&t), black_box(&hints), &InPlaceBlockers::new());
            black_box(p);
        });
    });
}

fn bench_big_intermediate_mlp(c: &mut Criterion) {
    // 50-layer MLP with ONE big intermediate (e.g. an attention scores
    // matrix in a transformer block). Without in-place, BOTH the L
    // and R slots inherit the big size (FFD can't avoid putting
    // linear_25 in one slot and relu_25 in the other); with in-place
    // the big tensor only inhabits one slot — significant savings.
    let mut layer_bytes: Vec<u64> = vec![4096; 50];
    layer_bytes[25] = 4 * 1024 * 1024; // 4 MB attention-scores-shaped
    let (t, hints) = build_mlp(&layer_bytes);
    report_savings("big-intermediate-mlp-50", &t, &hints);
    c.bench_function("plan_with_inplace_big_intermediate_mlp_50", |b| {
        b.iter(|| {
            let p = plan_with_inplace(black_box(&t), black_box(&hints), &InPlaceBlockers::new());
            black_box(p);
        });
    });
}

fn bench_autoencoder_mlp(c: &mut Criterion) {
    // 50-layer autoencoder shape: dims grow then shrink. Symmetric
    // layouts let FFD saturate without help — savings are typically
    // 0% in this case (documented as a planner-design property).
    let layer_bytes: Vec<u64> = (0..50)
        .map(|i| {
            let d = if i < 25 { i + 1 } else { 50 - i } as u64;
            d * d * 16
        })
        .collect();
    let (t, hints) = build_mlp(&layer_bytes);
    report_savings("autoencoder-mlp-50", &t, &hints);
    c.bench_function("plan_with_inplace_autoencoder_mlp_50", |b| {
        b.iter(|| {
            let p = plan_with_inplace(black_box(&t), black_box(&hints), &InPlaceBlockers::new());
            black_box(p);
        });
    });
}

criterion_group!(
    benches,
    bench_uniform_mlp,
    bench_big_intermediate_mlp,
    bench_autoencoder_mlp
);
criterion_main!(benches);
