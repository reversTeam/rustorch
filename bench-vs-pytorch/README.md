# bench-vs-pytorch — Performance comparison harness

Obrain-grade benchmark suite that pits RusTorch CPU kernels against PyTorch
on the **same machine** (Apple M4 Max). Used to drive plan
**P3.X — RusTorch Performance Sprint** (`7c83aecb-411c-43c6-a199-89a720da934b`).

## Why this exists

PyTorch is the de-facto baseline truth for ML perf. Without a same-machine
comparison, all RusTorch performance claims are unfounded. This crate fixes
that by:

- Running both stacks on the **same data, same shapes, same machine**
- Using **criterion** (statistical) on the Rust side, `time.perf_counter_ns` on Python
- Forcing **`target-cpu=native`** so LLVM auto-vectorizes for NEON
- **Pinning QoS** via `taskpolicy -c utility` to reduce scheduler noise
- **Auto-archiving dated baselines** in `baselines/` (committed)
- Tracking **delta** between baselines via `compare.py --before X --after Y`

## Quick start

```bash
./bench.sh                          # full run (~5 min on M4 Max)
./bench.sh --label post-T2          # tag the date with a milestone
./bench.sh --skip-pytorch           # only re-run Rust side (faster iteration)
```

Output goes to:
- `baselines/rustorch-YYYY-MM-DD[-LABEL].json`
- `baselines/pytorch-1thread-YYYY-MM-DD[-LABEL].json`
- `baselines/pytorch-12thread-YYYY-MM-DD[-LABEL].json`

Plus a console summary table with cache-tier (L1d/L2/DRAM), throughput
(GFLOPS or Gelem/s), and PyTorch ratios.

## Comparing two baselines

```bash
python3 compare.py \
  --before baselines/rustorch-2026-05-03-baseline-T0.json \
  --after  baselines/rustorch-2026-05-10-post-T2.json \
  --pytorch1 baselines/pytorch-1thread-2026-05-10-post-T2.json
```

Shows the per-op delta and whether the optimization moves the needle relative
to PyTorch.

## What's benchmarked

| Crate | Op | Why |
|---|---|---|
| `rustorch-core` | matmul f32 | The bottleneck. 64³ → 1024³, spans L1d→L2. |
| `rustorch-amp` | matmul bf16 with f32 accumulator | AMP CPU path (currently a regression on M4 — no bf16 NEON). |
| `rustorch-fusion` | fused matmul+bias+ReLU vs naive | Validates that fusion saves the bias/activation pass. |
| `rustorch-cpu/kernels` | softmax row-wise | Hot path in LLM logits + transformer attn. |
| `rustorch-attention` | flash forward vs naive | Their flagship. Validates 2× → 17× speedup at long sequences. |
| `rustorch-core` | tensor in-place add | Memory-bandwidth-bound test for SIMD elementwise. |

## Cache-tier classification

For the M4 Max P-core:
- **L1d**: 192 KB → working set fits per-thread, compute-bound
- **L2**: 16 MB shared → L1d miss but stays on-chip
- **DRAM**: above 16 MB → memory-bandwidth-bound (peak ~120 GB/s)

The `compare.py` table prints the tier so a perf shift can be attributed to
the right cause (compute optimization vs memory layout vs prefetching).

## Files

```
bench-vs-pytorch/
├── Cargo.toml                  # standalone crate (workspace=[])
├── .cargo/config.toml          # rustflags = -C target-cpu=native -C target-feature=+neon,+fp16
├── benches/
│   └── compare.rs              # criterion harness (matmul/bf16/fused/softmax/flash/add)
├── src/main.rs                 # legacy hand-rolled timer (kept for cross-check)
├── pytorch_bench.py            # PyTorch harness (1-thread + N-thread)
├── parse_criterion.py          # criterion estimates.json → flat schema
├── compare.py                  # snapshot + delta comparison
├── bench.sh                    # one-command runner
└── baselines/
    ├── CHANGELOG.md            # human-readable run log
    ├── rustorch-*.json         # archived RusTorch baselines
    └── pytorch-*.json          # archived PyTorch baselines
```

## Pattern reference

The harness follows the methodology validated on **obrain-substrate** (T1-T17 sprint):

- Note `724497e3` — baseline as a versioned artifact (not a screenshot)
- Note `b4d1b671` — bit-exact parity tests are non-negotiable
- Note `8a58a738` — for memory-bound work, count cache reads not arithmetic
- Note `ff6b24a1` — algorithmic transforms (HashMap → Vec) before SIMD
- Note `d5a9c653` — closure tables with delta vs Neo4j as the gold standard

## Reproducibility caveats

- Thermal throttling on long bench runs (≥ 3 min) on M4 Max: top P-cores
  throttle 5-15% under sustained scalar load. Witnessed: flash @ N=2048
  measured 99 ms in a hot-cache standalone run, then 543 ms when run as
  the last bench in a 5-shape sweep. **Mitigation**: `taskpolicy -c utility`
  + run benchmarks pre-bench in cool state, or pin to a single P-core via
  `--cores 1`.

- PyTorch on macOS uses Accelerate framework via vDSP/cBLAS — heavily
  vectorized + multithreaded by default. Forcing `torch.set_num_threads(1)`
  is the closest fair comparison to RusTorch's single-threaded scalar path,
  but Accelerate may still issue NEON instructions inside cBLAS routines
  even on 1 thread.
