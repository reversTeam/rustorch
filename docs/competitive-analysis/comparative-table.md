# Comparative table — rustorch vs burn vs candle vs tch-rs

This table consolidates the per-project deep-dives into a single
side-by-side reference. Cells are concrete values, not "similar" or
"unknown" — when a property is genuinely missing or N/A, that is
stated explicitly.

Sources:
- [`burn.md`](./burn.md) — burn 0.18
- [`candle.md`](./candle.md) — candle 0.8.x
- [`tch-rs.md`](./tch-rs.md) — tch-rs 0.18 (libtorch 2.5+)
- rustorch RFCs 0001–0006 (planned)

## 1. Mission & category

| Criterion | rustorch (planned) | burn | candle | tch-rs |
|-----------|--------------------|------|--------|--------|
| Category | Pure-Rust port | Pure-Rust framework | Pure-Rust framework | FFI wrapper over libtorch |
| Primary mission | PyTorch-faithful + WASM-first | Multi-backend training | Inference + serverless | "PyTorch from Rust, today" |
| Pure Rust toolchain | yes | yes | yes | **no** (needs libtorch C++) |
| WASM target | first-class (RFC-0006) | works, not headline | works (CPU+SIMD only) | **no** |
| `no_std` target | planned (Phase late) | **no** | **no** | **no** |

## 2. Tensor & shape model

| Criterion | rustorch | burn | candle | tch-rs |
|-----------|----------|------|--------|--------|
| Shape encoding | runtime (PyTorch-faithful) | static (`const D: usize`) | runtime | runtime (libtorch dispatch) |
| Storage abstraction | `Arc<StorageInner>` enum + Layout | per-kind associated type | `Arc<RwLock<Storage>>` enum | C++ `at::Tensor*` handle |
| In-place safety | version counter (atomic) | borrow checker + `Tensor::clone()` | `RwLock` lock acquisition | libtorch's autograd version |
| Memory alignment | uniform 64 B | backend-specific | system `Vec<T>` alignment | libtorch's allocator |
| `Tensor` vs `Variable` split | unified (PyTorch model) | unified | **split** (`Var` for params) | unified |

## 3. Backend architecture

| Criterion | rustorch | burn | candle | tch-rs |
|-----------|----------|------|--------|--------|
| Dispatch model | `trait Backend` (parametric) | `trait Backend` (parametric) | `enum Device` (closed set) | libtorch dispatcher |
| In-tree backends | CPU, wgpu, CUDA (later) | NdArray, Wgpu, Cuda, Tch, Candle | Cpu, Cuda, Metal | libtorch (cpu/cuda/mps) |
| Out-of-tree backends | open (trait-based) | open (trait-based) | **closed** (must edit core) | n/a (libtorch only) |
| Metal support | wgpu via Metal HAL | wgpu via Metal HAL | **first-party Metal** (mature) | via libtorch MPS |
| WebGPU support | first-party (RFC-0006) | yes (`burn-wgpu`) | community fork only | **no** |

## 4. Autograd

| Criterion | rustorch | burn | candle | tch-rs |
|-----------|----------|------|--------|--------|
| Implementation | thread-local tape | wrap-backend (`Autodiff<B>`) re-trace | inline `Op` graph on tensor | libtorch tape (C++) |
| Higher-order grads | `create_graph: true` flag | nest `Autodiff<Autodiff<B>>` | nested manual re-trace | libtorch native |
| `requires_grad` granularity | per-tensor flag | per-graph (Autodiff wrapper) | per-Var only | per-tensor flag |
| `no_grad()` scope | thread-local guard | use plain backend `B` | use plain `Tensor` | `no_grad_guard()` |
| Custom autograd Function | `CustomFunction` trait (P1.5) | `Backward` trait | `CustomOp` trait (limited) | libtorch C++ extensions |
| Anomaly mode (NaN/Inf detect) | planned (P1.5) | no | no | yes (libtorch) |

## 5. Op coverage (% of canonical PyTorch surface)

| Criterion | rustorch v1.0 target | burn | candle | tch-rs |
|-----------|----------------------|------|--------|--------|
| Elementwise + activations | 100% | ~95% | ~90% | 100% (libtorch) |
| Linalg (matmul/qr/svd/cholesky/einsum) | 100% (P1.4) | ~80% | ~50% (no qr/svd/cholesky/einsum) | 100% (libtorch) |
| Conv 1D/2D/3D + transpose | 100% (P1.4) | ~95% | ~90% | 100% (libtorch) |
| Normalizations (layer/batch/group/rms) | 100% (P1.4) | ~95% | ~95% | 100% (libtorch) |
| Loss functions incl. CTC | 100% (P1.6) | ~85% (no CTC) | ~70% (no CTC) | 100% (libtorch) |
| Sparse tensors | partial | partial | minimal | 100% (libtorch) |

## 6. Module / nn API

| Criterion | rustorch | burn | candle | tch-rs |
|-----------|----------|------|--------|--------|
| `#[derive(Module)]` macro | yes (P1.6) | yes | **no** (manual `VarBuilder`) | yes (`tch::nn::Path`) |
| Auto `parameters()` collection | yes | yes | manual via `VarMap` | yes |
| Native RNN/LSTM/GRU layers | yes (P1.6) | yes | cells only, no full layers | yes (libtorch) |
| Transformer / MultiheadAttention | yes (P1.6) | yes | yes (in `candle-nn`) | yes (libtorch) |

## 7. Training infrastructure

| Criterion | rustorch | burn | candle | tch-rs |
|-----------|----------|------|--------|--------|
| Optimizer suite | SGD, Adam, AdamW, Lion, Adafactor, etc. (P1.7) | SGD, Adam, AdamW | SGD, AdamW only | full (libtorch) |
| LR schedulers | Cosine, WarmupCosine, OneCycle, Plateau, etc. (P1.7) | Cosine, Linear | Cosine, Linear | full (libtorch) |
| Mixed precision (bf16/fp16) | autocast scope + GradScaler | manual cast | manual cast (no autocast) | libtorch native |
| Gradient checkpointing | planned | yes | no | yes |
| DataLoader (workers, pin_memory) | rich (P1.8) — bucket sampler, prefetch | basic | minimal | full (libtorch) |
| Augmentations library | vision/audio/text (P1.9) | partial | minimal | via libtorch |

## 8. Distributed

| Criterion | rustorch | burn | candle | tch-rs |
|-----------|----------|------|--------|--------|
| ProcessGroup abstraction | yes (Phase 5) | no (delegates to Tch backend) | no | yes (libtorch) |
| DDP | yes (Phase 5) | no | no | yes (libtorch) |
| FSDP | yes (Phase 5) | no | no | yes (libtorch) |
| NCCL backend | yes (Phase 5) | via Tch backend | no | yes (libtorch) |
| Elastic launcher | yes (`rustorch run --gpus N`) | no | no | torchrun (Python) |

## 9. Bindings & interop

| Criterion | rustorch | burn | candle | tch-rs |
|-----------|----------|------|--------|--------|
| Python bindings (PyO3) | first-party planned | community only | community only | not the use case |
| ONNX import | planned | yes (`burn-import`) | partial | via libtorch |
| ONNX export | planned (`rustorch-onnx`) | rudimentary | no | via libtorch |
| safetensors | yes (P1.8) | yes | yes | via Python ecosystem |
| HuggingFace model zoo compat | planned (HF compat plan) | manual port | **largest curated zoo** in Rust | via Python interop |

## 10. WASM / browser

| Criterion | rustorch | burn | candle | tch-rs |
|-----------|----------|------|--------|--------|
| `wasm32-unknown-unknown` target | first-class CI gate | works | works | **no** |
| WebGPU backend | first-party (RFC-0006) | yes (`burn-wgpu`) | community fork only | **no** |
| Bundle-size budget | hard `<10 MB` CI gate | none | implicit (~6 MB observed) | n/a |
| Browser demos shipped | planned (ResNet-18, GPT-2) | a few | many (Llama, Whisper, BERT, YOLO) | n/a |

## 11. Performance

| Criterion | rustorch v1.0 target | burn | candle | tch-rs |
|-----------|----------------------|------|--------|--------|
| Single-GPU train (vs PyTorch) | ±10% on canonical models | ~85-95% of PyTorch | n/a (inference focus) | bit-exact |
| Single-GPU inference (Llama-7B fp16) | match candle | unmeasured public | ≈ PyTorch (~5% slower) | bit-exact |
| WASM CPU inference | targeted via SIMD128 + WebGPU | works, not optimized | reference for the ecosystem | n/a |
| FlashAttention v2 | planned (custom plan) | partial | yes (separate kernel) | yes (libtorch) |

## 12. Maturity & community

| Criterion | rustorch | burn | candle | tch-rs |
|-----------|----------|------|--------|--------|
| GitHub stars (Q1 2026) | 0 (greenfield) | ~10 k | ~17 k | ~5 k |
| First commit | 2026-04 | 2022 | mid-2023 | 2020 |
| Funded development | not yet | yes (Tracel AI) | yes (Hugging Face) | volunteer (Mazare) |
| Release cadence | TBD | monthly minor | weekly | every 2-4 months |
| Production users | none yet | 4+ public | HF text-generation-inference + many | several ML serving cos |
| Contributors | 1 | 100+ | 280+ | 50+ |

## 13. Codebase scale

| Criterion | rustorch v1.0 target | burn | candle | tch-rs |
|-----------|----------------------|------|--------|--------|
| Core LoC (tensor + autograd) | ~30 k | ~35 k (tensor 25k + autodiff 10k) | ~14 k (tensor only) | ~13 k (auto-generated wrappers) |
| Total workspace LoC | ~80–100 k | ~100 k | ~80 k | ~25 k (relies on libtorch's >2 M) |
| Number of crates | 10 (planned) | 20+ | 5 core + many model crates | 3 (`tch`, `torch-sys`, `tch-tensorboard`) |
| Build dependencies | rust-only | rust-only | rust-only + optional CUDA toolkit | **libtorch (>500 MB to >2 GB)** |

## 14. Build & deployment

| Criterion | rustorch | burn | candle | tch-rs |
|-----------|----------|------|--------|--------|
| `cargo install <crate>` works | yes | yes | yes | **no** (needs `LIBTORCH=...`) |
| Cross-compilation | rustls-only, no OpenSSL | yes | yes | **painful** (needs cross-built libtorch) |
| Docker image size (CPU) | targeted ≤200 MB | ~150 MB | ~120 MB | ≥600 MB |
| Mobile (iOS/Android) | yes (Phase late) | works embedded | possible | no |

## 15. License & ecosystem fit

| Criterion | rustorch | burn | candle | tch-rs |
|-----------|----------|------|--------|--------|
| License | MIT OR Apache-2.0 | MIT OR Apache-2.0 | MIT OR Apache-2.0 | MIT OR Apache-2.0 |
| Trademark concerns | none planned | none | "candle" name is permissive | depends on libtorch (BSD) |
| Companion projects | rustorch-py, rustorch-onnx, rustorch-mobile | burn-import, burn-train | candle-transformers, candle-wasm-examples | torch-sys, tch-tensorboard |

## Summary scoring

For rustorch's mission (PyTorch-faithful + WASM-first + training capable):

| Project | matches mission? |
|---------|------------------|
| **rustorch** | by definition (greenfield, designed for it) |
| burn | partial — pure Rust + multi-backend, but static shapes diverge from PyTorch ergonomics; WASM not headline; no first-party DDP |
| candle | partial — pure Rust + WASM works, but `Var`/`Tensor` split breaks PyTorch fidelity; no autocast/DDP/FSDP; CPU-only WASM |
| tch-rs | **no** — disqualified by libtorch dependency (no WASM, no `no_std`, no pure-Rust toolchain) |

The combination of (a) PyTorch-faithful runtime shapes + per-tensor
`requires_grad`, (b) thread-local tape autograd with `create_graph`,
(c) WebGPU + WASM as a first-class CI gate with `<10 MB` budget, and
(d) first-party DDP/FSDP from day 1 — is **not jointly satisfied by any
existing project**. The closest combinations:

- burn satisfies (a partially), (b partially), (c partially), (d no).
- candle satisfies (a partially), (b no), (c partially), (d no).
- tch-rs satisfies (a fully) but fails (c) and (d-via-Rust-toolchain).

This gap is what `decision.md` resolves.
