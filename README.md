# rustorch

[![CI](https://github.com/rustorch/rustorch/actions/workflows/ci.yml/badge.svg)](https://github.com/rustorch/rustorch/actions/workflows/ci.yml)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](#license)
[![Crates.io](https://img.shields.io/crates/v/rustorch.svg)](https://crates.io/crates/rustorch)
[![Docs.rs](https://docs.rs/rustorch/badge.svg)](https://docs.rs/rustorch)
[![Codecov](https://codecov.io/gh/rustorch/rustorch/branch/main/graph/badge.svg)](https://codecov.io/gh/rustorch/rustorch)

> Pure Rust port of PyTorch — Tensor library + autograd + nn modules + optimizers
> + multi-backend (CPU, wgpu/WebGPU, CUDA opt-in) + WASM-first design.

`rustorch` aims for a **faithful PyTorch API** in safe Rust, with a single
codebase that runs natively on Linux/Mac/Windows, in the browser via WebGPU,
and on CUDA when the feature is enabled. The mental model is identical to
PyTorch; the syntax is Rust.

## Status

This project is **pre-1.0** but has shipped through Phase 3.

| Phase | Title                                                                                 | Status     |
|------:|---------------------------------------------------------------------------------------|------------|
|     0 | Architecture & RFC                                                                    | ✅ done    |
|     1 | Fondations CPU + Autograd                                                             | ✅ done    |
|     2 | Backend GPU via wgpu (WASM)                                                           | ✅ done    |
|   2.5 | Ecosystem & Console backend (CLI, sweeps, RPC, serve)                                 | ✅ done    |
|     3 | Optimisations avancées (Memory Planning, Flash Attn, Quant, Fusion, AMP, Checkpoint)  | ✅ done    |
|     4 | Backend CUDA natif (opt-in)                                                           | 📋 planned |
|     5 | Distributed Training (DDP, FSDP)                                                      | 📋 planned |
|   2.7 | Console UI                                                                            | 📋 planned |
|     6 | Écosystème & Adoption (1.0)                                                           | 📋 planned |

**1100+ tests pass, clippy clean across the workspace** as of the latest commit.

## What's in the box

### Core (Phases 0-2)

- **`rustorch-core`** — `Tensor`, `Storage`, `Layout`, 8 dtypes (f32/f64/bf16/f16/i32/i64/bool/u8), zero-copy views, refcount sharing.
- **`rustorch-cpu`** — CPU backend with rayon parallelism + LLVM auto-vectorisation; ~50 ops (arithmetic, linalg, conv, activations, reductions, normalisations, indexing, shape).
- **`rustorch-autograd`** — Tape-based reverse-mode AD with `Variable`, ~35 backward formulas, `CustomFunction` trait, `checkpoint` / `checkpoint_n`, `autocast` scope guard, anomaly mode.
- **`rustorch-nn`** — `Module` trait, `Linear`, `Conv1d/2d/3d`, `LayerNorm`, `RMSNorm`, `BatchNorm2d`, `MultiheadAttention`, `Embedding`, `Dropout`, `Sequential`, `ModuleList`, hooks.
- **`rustorch-optim`** — `Optimizer` trait, `SGD`, `Adam`, `AdamW`, `Lion`, `RMSprop`, `Adagrad`, `Adamax`, `NAdam`, `RAdam`, `LBFGS`, `Adadelta` + LR schedulers (`StepLR`, `CosineAnnealing`, `OneCycle`, `ReduceLROnPlateau`).
- **`rustorch-data`** — `Dataset` / `IterableDataset` traits, `DataLoader` with rayon workers, `Sampler` (Sequential / Random / WeightedRandom / BucketBy / Distributed), bundled MNIST / CIFAR / ImageNet / WikiText / LibriSpeech.
- **`rustorch-serde`** — Native `safetensors` reader/writer + `state_dict` round-trip.
- **`rustorch-wgpu`** — wgpu backend (Vulkan/Metal/DX12/WebGPU); ~80 WGSL kernels; pipeline cache; cross-platform native + browser; built-in Flash Attention v1.
- **`rustorch-derive`** — `#[derive(Module)]` proc macro.

### Phase 2.5 — Ecosystem & Console

- **`rustorch-cli`** — `rustorch run` / `sweep` / `check` / `resume` / `deploy` binary.
- **`rustorch-console-server`** — HTTP/JSON-RPC API + SSE topics + SQLite persistence; 30+ REST endpoints; OpenAPI documented.
- **`rustorch-sweep`** — Hyperparameter sweep system: Grid / Random / ASHA / Bayes (Gaussian Process + EI/UCB/PI acquisitions, hand-rolled Cholesky with jitter ladder).
- **`rustorch-log`** — Multi-sink logging: Console SSE / stdout JSON / W&B / MLflow / TensorBoard (hand-rolled tfevents writer).
- **`rustorch-profile`** — Chrome trace + sampling profiler.
- **`rustorch-serve`** — Production inference runtime: HTTP `/predict`, gRPC `Predict`/`PredictStream`/`PredictBatch`, dynamic batching, autoscale (k8s HPA + docker-compose manifests), CUDA Graphs trait.
- **`rustorch-rpc-internal`** — gRPC bidi Console ↔ Runner channel with auto-reconnect, FIFO drop-oldest buffer, `RpcRunnerSink` bridging `log_metric()` straight to the Console.

### Phase 3 — Optimisations avancées

- **`rustorch-planner`** — Inductor-style memory planner: caller-driven trace API, FFD interval-disjoint allocator with alignment (16/64/256), in-place mutation detection, Sethi-Ullman topological reorder, end-to-end `Planner::plan()` with budget + checkpoint suggestions. **90.3% peak savings** on a 167-op ResNet-50 sim.
- **`rustorch-attention`** — Tiled Flash Attention v2 (CPU): online (Welford-style) softmax, parallel tiled forward (**2.48× speedup vs naive at N=2048, 2.56× at N=8192**), backward via recomputation (gradcheck-verified), causal/padding masks (**0.95× of unmasked = effectively free**), multi-head 4D `(B, H, N, D)` WGSL shaders for forward + backward.
- **`rustorch-quant`** — Int8 dynamic quantisation: `MinMax`/`PerChannel`/`Histogram` observers, HALF_TO_EVEN quantize/dequantize, scalar int8 GEMM with f32 accumulator, `QuantLinear` / `QuantConv2d` (im2col + int8 GEMM), `quantize_model` calibration workflow, GPU dp4a-emulation WGSL shader. **4× memory savings**, end-to-end CIFAR-shape CNN within 4% of f32 baseline.
- **`rustorch-fusion`** — Eager-mode op fusion: DAG pattern matcher with longest-chain-first dedup, 15-pattern default library, fused `matmul+bias+activation` (Relu/Gelu/Silu), `layernorm+linear` (Welford-stable), `residual_add_scale_mul`, elementwise epilogue chains, `FusionRegistry` + `NO_FUSE` flag.
- **`rustorch-fusion-macros`** — `fuse_call!` proc macro expanding compact pipeline syntax into fused kernel calls.
- **`rustorch-amp`** — Mixed precision: `bf16` / `fp16` conversions via `half`, `Autocast::Bf16/Fp16` thread-local scope guard with op promotion list (matmul/conv → low-prec, softmax/norm/loss → f32), `GradScaler` matching PyTorch defaults (init_scale=65536, growth=2.0, backoff=0.5), scalar bf16/fp16 matmul with f32 accumulator, `parameters_to_bf16/fp16` cast helpers.
- **`rustorch-checkpoint`** — Gradient checkpointing facade over `rustorch-autograd::checkpoint`; `RngSnapshot` for dropout-correct recompute; nesting depth tracker with panic-safe RAII.

## Quickstart

Add `rustorch` to your `Cargo.toml` (re-exports the most common crates):

```toml
[dependencies]
rustorch = { version = "0.0.1" }
# Or pull individual sub-crates for tighter dependency control:
rustorch-core      = "0.0.1"
rustorch-autograd  = "0.0.1"
rustorch-nn        = "0.0.1"
rustorch-optim     = "0.0.1"
rustorch-attention = "0.0.1"
rustorch-quant     = "0.0.1"
```

### MLP training (Phase 1 surface)

```rust
use rustorch::prelude::*;

fn main() -> rustorch::Result<()> {
    let dev = Device::cpu();
    let net = Sequential::new()
        .add(Linear::new(784, 256))
        .add(ReLU)
        .add(Linear::new(256, 10));
    let mut opt = AdamW::new(net.parameters(), 1e-3)?.weight_decay(0.01);

    let train = Mnist::train(&dev)?;
    let loader = DataLoader::batch(64).workers(4).build(train)?;

    for _epoch in 0..10 {
        for batch in &loader {
            let logits = net.forward(&batch.x)?;
            let loss = cross_entropy(&logits, &batch.y)?;
            opt.backward(&loss)?;
        }
    }
    net.save("mnist_mlp.safetensors")?;
    Ok(())
}
```

### Flash Attention forward (Phase 3)

```rust
use rustorch_attention::{flash_forward, flash_forward_masked, AttentionShape, Mask};

let shape = AttentionShape::new(2, 8, 1024, 64); // B=2, H=8, N=1024, D=64
let q = vec![0.0f32; shape.buffer_len()];
let k = vec![0.0f32; shape.buffer_len()];
let v = vec![0.0f32; shape.buffer_len()];
let mut out = vec![0.0f32; shape.buffer_len()];

// Vanilla attention — 2.48× faster than naive at N=2048, peak memory O(N+tile²)
flash_forward(&shape, &q, &k, &v, &mut out)?;

// GPT-style causal attention — masked entries become -inf BEFORE softmax
flash_forward_masked(&shape, &q, &k, &v, &Mask::Causal, &mut out)?;
```

### Memory planner end-to-end (Phase 3)

```rust
use rustorch_planner::{OpDag, OpId, OpNode, Planner, PlannerInput, TensorId};

let mut dag = OpDag::new();
dag.add_op(OpNode { id: OpId(1), reads: vec![], writes: TensorId(1), bytes: 1024, align: 64 });
dag.add_op(OpNode { id: OpId(2), reads: vec![TensorId(1)], writes: TensorId(2), bytes: 1024, align: 64 });
// ... more ops

let schedule = Planner::plan(&PlannerInput::from_dag(dag))?;
println!("peak bytes: {}", schedule.peak_bytes());      // ~10× lower than naive
println!("slot count: {}", schedule.slot_count());
```

On a 167-op ResNet-50 forward sim: 99.8 MB naive total → 9.6 MB planned peak (**90.3% savings**).

### Int8 quantization (Phase 3)

```rust
use rustorch_quant::{quantize_model, LayerSpec, MinMaxObserver, Conv2dParams};

// Build layer specs from your trained f32 model
let layers = vec![
    LayerSpec::Conv2d {
        weight: trained_conv1_weights,
        bias: Some(conv1_bias),
        c_out: 8, c_in: 3,
        params: Conv2dParams::k3s1p1(),
    },
    LayerSpec::Linear {
        weight: trained_fc_weights,
        bias: Some(fc_bias),
        in_features: 1024, out_features: 10,
    },
];

// Calibrate: run f32 model on N batches, feed each layer's input to its observer
let mut observers: Vec<_> = (0..layers.len()).map(|_| MinMaxObserver::new()).collect();
for batch in calibration_loader {
    let activations = forward_f32(&layers, &batch);
    for (i, obs) in observers.iter_mut().enumerate() { obs.update(&activations[i]); }
}

// Freeze qparams + swap to int8 layers (4× weight memory savings)
let (q_layers, activation_qps) = quantize_model(&layers, &observers)?;
```

### Mixed precision (Phase 3)

```rust
use rustorch_amp::{autocast, GradScaler, Mixed, parameters_to_bf16};

// bf16 — recommended on Ampere+; no GradScaler needed
autocast(Mixed::Bf16, || {
    let logits = net.forward(&batch.x)?;
    let loss = cross_entropy(&logits, &batch.y)?;
    opt.backward(&loss)?;
    Ok(())
})?;

// fp16 — Volta/Turing, requires GradScaler for stable training
let mut scaler = GradScaler::new(); // init_scale=65536, growth_factor=2.0, ...
autocast(Mixed::Fp16, || {
    let loss = cross_entropy(&net.forward(&batch.x)?, &batch.y)?;
    let scaled = scaler.scale_loss(loss.item());
    // ...backward propagates scaled gradients
    Ok(())
})?;
match scaler.step(&mut grads) {
    rustorch_amp::ScalerStep::Applied => opt.step(),
    rustorch_amp::ScalerStep::Skipped => { /* inf/nan detected, scale halved automatically */ }
}
scaler.update();
```

### Gradient checkpointing (Phase 3)

```rust
use rustorch_checkpoint::{checkpoint, gradient_checkpointing_count};

// f re-runs during backward — trade ~30% compute for 50% activation memory
let y = checkpoint(|x| layer.forward(x), &x)?;
backward(&y, None)?;
assert!(gradient_checkpointing_count() >= 1); // confirm recompute happened
```

### Fused kernels (Phase 3)

```rust
use rustorch_fusion::{fused_matmul_bias_activation, Activation};

// Single-pass matmul + bias + activation (one allocator slot, registers for accumulator)
fused_matmul_bias_activation(&x, &w, Some(&bias), &mut y, m, k, n, Activation::Gelu)?;

// Or via the proc-macro for compact syntax:
// use rustorch_fusion_macros::fuse_call;
// fuse_call!(matmul_bias_gelu(x, w, bias) -> y; m=128, k=512, n=128);
```

## Benchmarks

All run on the workspace's default config (Rust 1.78, scalar Rust + LLVM auto-vectorisation; no SIMD intrinsics yet).

| Bench                                       | Result        | Acceptance     |
|---------------------------------------------|---------------|----------------|
| Memory Planner — lifetime analysis 1000 nodes  | **24.6 µs**   | < 5 ms          |
| Memory Planner — FFD allocator 1000 buffers     | **137 µs**    | < 2 ms          |
| Memory Planner — reorder 500 ops             | **2.21 ms**   | < 10 ms         |
| Memory Planner — ResNet-50 sim peak savings | **90.3 %**    | ≥ 30 %          |
| Flash Attention forward (B=1, H=1, N=2048, D=64) vs naive | **2.48× speedup** | ≥ 1.5× |
| Flash Attention long-seq N=8192 vs naive      | **2.56× speedup** | ≥ 2× |
| Flash Attention causal mask overhead       | **0.95× of unmasked** | ≤ 1.05× |
| Online softmax 4096 entries vs naive         | **0.96×**       | ≤ 1.10×         |
| Int8 GEMM scalar 256³ vs f32                | **1.43× speedup** | ≥ 2× (needs SIMD) |
| Quant observer over 1M elements           | **680 µs**    | < 1 ms          |
| Quant CNN E2E weight memory ratio        | **4.00×**     | ≥ 3.5×          |

## Workspace layout

26 Cargo crates organised by capability:

```
crates/
├── rustorch/               # umbrella re-export crate (the public API)
├── rustorch-core/          # Tensor, Storage, Layout, dtypes
├── rustorch-cpu/           # CPU backend (rayon + auto-vectorisation)
├── rustorch-autograd/      # tape-based reverse-mode AD + custom_function
├── rustorch-nn/            # Module trait + Linear / Conv / Attention / Norm / ...
├── rustorch-optim/         # Optimizer trait + 11 algorithms + LR schedulers
├── rustorch-derive/        # #[derive(Module)] proc-macro
├── rustorch-data/          # Dataset / DataLoader / Sampler
├── rustorch-serde/         # safetensors I/O + state_dict
├── rustorch-codegen/       # build.rs codegen (ops.yaml → Rust)
├── rustorch-wgpu/          # wgpu/WebGPU backend + WGSL kernels
├── rustorch-wasm-demo/     # in-browser demo (WASM build target)
│
├── rustorch-cli/           # `rustorch` CLI binary
├── rustorch-console-server/# Console HTTP/JSON-RPC backend + SSE
├── rustorch-sweep/         # Grid / Random / ASHA / Bayes (GP)
├── rustorch-log/           # Multi-sink logging (console / W&B / MLflow / TB)
├── rustorch-profile/       # Chrome trace + sampling profiler
├── rustorch-serve/         # HTTP/gRPC inference runtime + dynamic batching
├── rustorch-rpc-internal/  # gRPC Console ↔ Runner with auto-reconnect
│
├── rustorch-planner/       # Inductor-style memory planner (P3)
├── rustorch-attention/     # Flash Attention CPU + WGSL (P3)
├── rustorch-quant/         # Int8 quantisation (P3)
├── rustorch-fusion/        # Eager op fusion (P3)
├── rustorch-fusion-macros/ # fuse_call! proc-macro (P3)
├── rustorch-amp/           # bf16 / fp16 / autocast / GradScaler (P3)
└── rustorch-checkpoint/    # Gradient checkpointing facade (P3)
```

## Building

Requires Rust **1.78+** (MSRV) — pinned via `rust-toolchain.toml`.

```bash
# Native build + test (1100+ tests)
cargo build --workspace
cargo test  --workspace

# WASM target
cargo build --target wasm32-unknown-unknown -p rustorch-core

# Strict lints (CI does this)
cargo clippy --workspace --all-targets -- -D warnings

# GPU integration tests (requires a real adapter)
cargo test -p rustorch-wgpu --features gpu-tests

# Benches (Phase 3 highlights)
cargo bench -p rustorch-planner   --bench lifetime_bench
cargo bench -p rustorch-planner   --bench allocator_bench
cargo bench -p rustorch-attention --bench flash_forward_bench
cargo bench -p rustorch-attention --bench long_seq_bench
cargo bench -p rustorch-quant     --bench gemm_int8_bench
cargo bench -p rustorch-fusion    --bench fused_matmul_bench
```

### macOS gotcha

If you have **both Homebrew Rust and rustup**, Homebrew's cargo is usually
first in `$PATH` and shadows rustup. The wasm32 target stdlib lives only in
rustup, so a Homebrew cargo invocation against `--target wasm32-unknown-unknown`
will fail with `can't find crate for 'core'`.

Fix: prepend `~/.cargo/bin` to your `PATH` so rustup's shims win, or
explicitly invoke `~/.cargo/bin/cargo`.

## Architectural decisions

Every non-trivial choice is documented as an architectural Decision in the
project knowledge graph. Highlights from Phase 3:

- **Caller-driven trace API** (planner): record_def/record_use over an opaque
  `TensorId`, no autograd dep. Lets ANY trace producer feed the planner.
- **FFD over union-find** (allocator): union-find is the wrong abstraction for
  packing under disjointness — FFD is what Inductor uses.
- **Sethi-Ullman greedy with Reverse(OpId) tie-break** (reorder): gotcha
  documented — `max_by` on a tuple picks the LARGEST id on ties.
- **Hint-driven in-place** (planner): caller emits `InPlaceHint`, planner
  verifies safety. Decoupled from autograd Op trait.
- **Asymmetric workload bound** (in-place): in-place adds peak savings only
  when one intermediate is dramatically bigger than the rest. On uniform-size
  workloads FFD already saturates the optimum.
- **GraphRunner trait + MockGraphRunner** (CUDA Graphs): real cudarc gated to
  Phase 4; trait shape stable so call sites don't change.
- **Hand-rolled Cholesky with jitter ladder** (Bayes sweep): no nalgebra dep,
  recovers gracefully from clustered samples.
- **Hand-rolled tfevents writer** (logging): no `tensorboard-rs` dep; ~100
  lines of varint + masked CRC + proto3.
- **par_chunks_mut over per-(b,h) slabs** (Flash forward): borrow-checker-safe
  parallelism, no raw pointers.
- **Branchless f32::min/max + separate finite-counting pass** (observer): hits
  680 µs / 1M elements; ±Inf legitimately observed (outlier signal).
- **Inference-only QuantLinear** (quant): raw-slice API, no autograd. Bridge
  to nn::Linear via the calibration workflow.
- **Scalar int8 GEMM as golden reference** (quant): SIMD specialisations
  (AVX-VNNI / NEON sdot) land in arch-specific commits; cpu_features helpers
  let call sites branch today.

See [`docs/rfcs/`](docs/rfcs/) for the full RFCs and the project's Decision
graph (queryable via the Project Orchestrator MCP tools).

## Project documentation

- [`docs/rfcs/`](docs/rfcs/) — architectural RFCs (Phase 0 deliverable)
- [`CONTRIBUTING.md`](CONTRIBUTING.md) — contribution guide, commit style
- [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md) — Contributor Covenant 2.1

## License

Dual licensed under either:

- [Apache License, Version 2.0](LICENSE-APACHE) (or http://www.apache.org/licenses/LICENSE-2.0)
- [MIT license](LICENSE-MIT) (or http://opensource.org/licenses/MIT)

at your option, matching the Rust ecosystem standard.

## Contribution

Unless you explicitly state otherwise, any contribution intentionally
submitted for inclusion in the work by you, as defined in the Apache-2.0
license, shall be dual licensed as above, without any additional terms or
conditions. See [`CONTRIBUTING.md`](CONTRIBUTING.md) for details.
