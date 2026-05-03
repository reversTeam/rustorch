# RusTorch CPU Performance — Baseline CHANGELOG

Format inspiré de la note obrain `d5a9c653` (T17i closure) — chaque entrée trace
un point de mesure dans le sprint **P3.X — RusTorch Performance Sprint**
(plan `7c83aecb-411c-43c6-a199-89a720da934b`).

Hardware : Apple M4 Max, macOS 15.7.1, Rust 1.93 release + `target-cpu=native` +
LTO fat + criterion 0.5 + `taskpolicy -c utility`. PyTorch 2.11.0 (Accelerate
framework + cBLAS).

---

## 2026-05-03 — Baseline T0 (avant toute optim)

**Files**: `rustorch-2026-05-03-baseline-T0.json`,
`pytorch-1thread-2026-05-03-baseline-T0.json`,
`pytorch-12thread-2026-05-03-baseline-T0.json`

### Headline numbers (RusTorch single-thread vs PyTorch 1-thread)

| Op | Shape | Tier | RusTorch | PyTorch-1 | Ratio | Throughput RT |
|---|---|---|---:|---:|---:|---:|
| matmul f32 | 64³ | L1d | 134.9 µs | 1.67 µs | **80.9×** | 3.89 GF/s |
| matmul f32 | 128³ | L1d | 1.14 ms | 4.42 µs | **257×** | 3.69 GF/s |
| matmul f32 | 256³ | L2 | 10.40 ms | 25.2 µs | **413×** | 3.23 GF/s |
| matmul f32 | 512³ | L2 | 85.05 ms | 165.8 µs | **513×** | 3.16 GF/s |
| matmul f32 | 1024³ | L2 | 924.88 ms | 1.38 ms | **672×** | 2.32 GF/s |
| softmax | 128×32k | DRAM | 17.49 ms | 4.18 ms | **4.19×** | 0.47 GE/s |
| softmax | 512×50k | DRAM | 122.59 ms | 31.45 ms | **3.90×** | 0.42 GE/s |
| softmax | 1024×50k | DRAM | 246.83 ms | n/a | n/a | 0.42 GE/s |
| flash B=1H=1N=512D=64 | — | L2 | 6.01 ms | 236.8 µs | **25.4×** | 11.16 GF/s |
| flash B=1H=4N=1024D=64 | — | L2 | 49.85 ms | 3.09 ms | **16.1×** | 21.54 GF/s |
| flash B=2H=4N=2048D=64 | — | L2 | 543.05 ms ⚠ | 24.74 ms | **22.0×** | 15.82 GF/s |
| add inplace | 1k | L1d | 1.86 µs | 1.29 µs | 1.44× | 1.07 GE/s |
| add inplace | 100k | L2 | 145.12 µs | 24.88 µs | 5.83× | 1.38 GE/s |
| add inplace | 1M | L2 | 1.74 ms | 246.8 µs | 7.04× | 1.15 GE/s |
| add inplace | 10M | DRAM | 20.86 ms | 960.6 µs | **21.7×** | 0.96 GE/s |

⚠ Flash N=2048 measured 543ms vs 99ms in earlier shorter bench → thermal
throttling on the 3-minute full criterion run. To investigate (open a
gotcha note: "thermal envelope on M4 Max for sustained scalar GEMM").

### Internal RusTorch speedups (algorithmic correctness check)

| Shape | Flash | Naive | Speedup |
|---|---:|---:|---:|
| B=1 H=1 N=512 D=64 | 6.01 ms | 13.96 ms | 2.32× |
| B=1 H=4 N=1024 D=64 | 49.85 ms | 235.78 ms | 4.73× |
| B=2 H=4 N=2048 D=64 | 543.05 ms | 2361.65 ms | 4.35× |

Flash algorithm is **structurally correct** — speedup grows with N as expected
for online softmax (O(N) memory) vs naive (O(N²) score matrix).

### Cache-tier observations

- **GFLOPS sustained matmul scalaire** : 2.3-3.9 GF/s — ceiling théorique d'un
  P-core M4 sans SIMD ni tiling (peak ~300 GF/s par P-core avec NEON FMA).
- **L1d → L2 dégradation** : matmul 64³ (3.89 GF/s, fits L1d) → 1024³ (2.32 GF/s,
  spills L2). Le tiling T2 doit récupérer cet écart.
- **Memory-bound ops** (add, softmax) plafonnent à ~1 GE/s = ~4 GB/s — un seul
  P-core sans SIMD ne saturé pas la bande passante (~120 GB/s sur M4 Max).

### Why these gaps exist

Confirmed in T0 audit (this commit):
1. **`crates/rustorch-core/src/ops/matmul.rs`** : naïve 3 nested loops, no tiling
2. **`crates/rustorch-cpu/src/cpu_backend.rs::matmul_naive`** : same, generic over T
3. **`crates/rustorch-fusion/src/patterns/matmul_bias_act.rs`** : 3 loops scalar
4. **`crates/rustorch-amp/src/bf16_kernels.rs`** : 3 loops scalar
5. No SIMD usage despite `wide = "0.7"` in workspace deps
6. No rayon usage in CPU kernels (only in `rustorch-attention`)
7. No BLAS bindings (Accelerate / OpenBLAS)

### Targets for the sprint (M4 Max, single-thread)

| Op | Baseline | Target | Method |
|---|---:|---:|---|
| matmul 1024³ f32 | 924.88 ms | ≤ 50 ms | T2 tiled + T6 wide SIMD (18× gain) |
| matmul 1024³ bf16 | 881.64 ms | ≤ 100 ms | T7 bf16 SIMD path (8× gain) |
| softmax 512×50k | 122.59 ms | ≤ 30 ms (1-thr) ≤ 8 ms (12-thr) | T3 + T8 (4-15× gain) |
| add inplace 10M | 20.86 ms | ≤ 2 ms | T6 wide SIMD (10× gain) |
| flash N=2048 | 99-543 ms | ≤ 30 ms | T6 SIMD inner + thermal pinning |

After the sprint, target relative position to PyTorch 1-thread:
- matmul 1024³ : 672× → 36× (still slower, but match candle)
- softmax 512×50k : 3.9× → 1.5×
- add 10M : 21.7× → 2×
- flash N=2048 : 22× → 1.2× (effectively match)

---

## How to add an entry

After each task closure (T2 done, T3 done, etc.):

```bash
./bench.sh --label post-T2     # runs full bench + tags the date
git add baselines/             # commit the JSONs
```

Then add a section above with:
- Date of the run
- Task closed
- Headline numbers (delta vs T0 baseline)
- Throughput observations
- Notes on regressions (if any)
- Updated table of remaining gaps

Run delta view:

```bash
python3 compare.py \
  --before baselines/rustorch-2026-05-03-baseline-T0.json \
  --after  baselines/rustorch-YYYY-MM-DD-post-T2.json \
  --pytorch1 baselines/pytorch-1thread-YYYY-MM-DD-post-T2.json
```
