---
id: 0001
title: Architecture overview & big decisions
status: Draft
date: 2026-05-01
authors:
  - rustorch team <rfc@rustorch.dev>
supersedes: []
superseded-by: []
---

# RFC-0001: Architecture overview & big decisions

## Summary

This RFC commits to six structural decisions for rustorch:
**(1) runtime-opaque tensor shapes**,
**(2) trait-based backend abstraction with monomorphization**,
**(3) tape-based autograd**,
**(4) `rayon`-based parallelism + sync core, `tokio` only at I/O edges**,
**(5) WASM-as-first-class target**,
**(6) `build.rs` codegen from a single `ops.yaml`**.

Together they define the workspace-level shape of the project.
All subsequent RFCs (0002 — Tensor; 0003 — Autograd; 0004 — Backend trait;
0005 — Codegen; 0006 — WASM) refine sub-areas but inherit these six
choices unchanged unless they explicitly supersede them.

## Motivation

rustorch reimplements the user-facing surface of PyTorch in safe Rust.
The hard part is not any single op; it is the **shape of the project as
a whole**: what abstractions must exist before we write the first matmul.
Get that wrong and we either rebuild the foundations later (cf. burn
v0.x rewrites) or end up with a moving API that nobody can integrate
against.

Concrete pain points that motivated this RFC:

- **PyTorch's TensorImpl.h** is 1500+ lines of C++ that cannot be ported
  literally to safe Rust. Some pattern-level decisions must be made.
- **burn 0.18** uses static type-level shapes (`Tensor<B, 4>`) which
  forces every op signature to thread a const generic; this is a great
  type-safety story but a poor PyTorch-faithful one.
- **candle** uses a single `Device` enum + `Storage` enum, which avoids
  the trait-object overhead but inflates pattern-matching in every op.
- The **WASM story** is fragile: rayon, fork, mmap, file I/O, and
  multi-threading don't all work in the browser. The ergonomics of the
  Rust language alone don't enforce a clean separation between
  WASM-safe and native-only code; we must.
- Rust's `build.rs` and `proc_macro` together **do** support codegen
  but you have to commit to one early — a `lib.rs` written by hand for
  six months can't be retrofitted into one generated from YAML.

If we don't make these decisions up front, the cost of changing them
later compounds with each line of `unsafe` we ship.

## Constraints

- **LoC budget.** We aim for ≤200k lines of hand-written Rust at v1.0,
  not counting generated code. Decisions that drag the LoC count past
  300k are rejected.
- **Performance.** Within ~30% of PyTorch eager on standard workloads
  (ResNet-50, GPT-2 small) by Phase 4. WASM/wgpu within ~3× of native
  CPU-baseline by Phase 2.
- **Ergonomics.** PyTorch-faithful API at the call site:
  `Tensor::zeros::<f32>([2, 3], &dev)?`, `net.forward(&x)?`,
  `opt.backward(&loss)?`. No exposing trait objects, no bouncing through
  an enum dispatch the user has to match on.
- **Compatibility.** MSRV pinned at **1.78** (per workspace `Cargo.toml`),
  edition 2021. Three host triples in CI (Linux, macOS, Windows) plus
  `wasm32-unknown-unknown`. CUDA is opt-in (`features = ["cuda"]`).
- **Build time.** `cargo check --workspace` ≤10s on a warm cache for an
  M-series Mac. `cargo build --workspace --release` ≤90s cold.
- **WASM/`no_std` scope.** Phase 0–1: `rustorch-core` and the public
  `rustorch` umbrella crate must build for `wasm32-unknown-unknown`
  with default features. `no_std` is a Phase 6 stretch goal restricted
  to the Tensor + Linear + activation subset.

## Alternatives Considered

The six big decisions interlock; we discuss each in its own subsection.

### Decision 1 — Tensor shape model

#### Alternative 1A: Static type-level shapes (`Tensor<B, T, R>`)

**Approach.** Encode rank in a const generic `R: usize`; track shape
elements via type-level integers (`typenum`-style) or const generics.
burn 0.18 follows this path.

**Pros.**
- Compile-time shape checks for many bugs.
- Erases `assert_eq!(shape, ...)` at runtime.
- Excellent IDE auto-complete.

**Cons.**
- Op signatures balloon: `matmul<B, T, const M: usize, const N: usize, ...>`.
- Dynamic-shape ops (data-dependent reshape, gather) require runtime
  fallback paths anyway, undermining the static guarantee.
- Diverges from PyTorch ergonomics (the explicit goal).
- Compile time inflates 2–3× from monomorphization explosion.

**Cost.** ~+30% LoC across the public API.

#### Alternative 1B: Runtime-opaque shapes (PyTorch-style)

**Approach.** `Tensor` holds `Vec<usize>` for shape, validated at the
op call site. Mismatches return `Err(ShapeMismatch { ... })`.

**Pros.**
- Matches PyTorch 1-to-1 — the RFC freeze in user docs v0.7.2 already
  assumes this.
- Compile-time cost is minimal.
- Dynamic-shape ops are a natural fit.

**Cons.**
- Shape bugs surface at runtime, not compile time.
- Marginal extra runtime check per op (negligible vs the kernel cost).

**Cost.** ~+0% LoC; this is essentially the baseline.

#### Alternative 1C: Hybrid (opaque + thin static helpers)

**Approach.** Tensor is opaque but we provide `Tensor::view_as<S: Shape>()`
helpers that return a typed view for the rare hot path that wants it.

**Pros.**
- Keeps the PyTorch-faithful default.
- Power users opt into static checks.

**Cons.**
- Two parallel APIs to document.
- Easy to introduce inconsistencies (an op accepting opaque only).

**Cost.** ~+5% LoC for the helpers.

### Decision 2 — Backend abstraction

#### Alternative 2A: `trait Backend` with associated types

**Approach.** `trait Backend { type Storage: Send+Sync; ... }`. Concrete
backends implement the trait. Tensor is parametric: `Tensor<B: Backend>`.
burn-style.

**Pros.**
- Each backend has its own `Storage` type — safe by construction
  (cannot mix CPU and CUDA tensors in a single op).
- Monomorphization specializes ops per backend; no virtual call cost.

**Cons.**
- Monomorphization inflates binary size.
- Cross-backend `Tensor::to(&device)` is awkward (different concrete types).

**Cost.** Medium.

#### Alternative 2B: `enum Storage { Cpu(...), Cuda(...), Wgpu(...) }`

**Approach.** `Storage` is a single enum with one variant per backend.
candle-style.

**Pros.**
- Concrete `Tensor` type — `Tensor::to(&dev)?` returns `Tensor`.
- Smaller binary.
- Easier dynamic dispatch when needed.

**Cons.**
- Every op contains a `match storage { ... }`, inflating LoC and giving
  the compiler less freedom.
- Adding a new backend requires touching the enum (harder for downstream
  forks).

**Cost.** Low.

#### Alternative 2C: `DispatchKey`-style runtime registry (PyTorch's path)

**Approach.** A `DispatchKey` value identifies the op×backend kernel;
ops look up the kernel via a hash map at runtime.

**Pros.**
- Most flexible — third parties register kernels with no rebuild.
- Mirrors how PyTorch C++ does it.

**Cons.**
- `unsafe` everywhere.
- Hash-lookup cost per op.
- Hostile to Rust's type system.

**Cost.** High in implementation, lower in LoC, hostile to Rust idioms.

### Decision 3 — Autograd model

#### Alternative 3A: Tape-based, thread-local recording

**Approach.** A thread-local `Tape` records `Arc<dyn Node>` boxes during
the forward pass. `backward()` traverses the tape in reverse topological
order. PyTorch C++ uses this.

**Pros.**
- Matches PyTorch's mental model exactly.
- Higher-order grads land naturally via `create_graph: true`.
- No control-flow restrictions on user code.

**Cons.**
- Allocates `Arc<dyn Node>` per non-trivial op.
- Thread-local means each thread sees its own tape (acceptable; user docs note this).

**Cost.** Reference design; baseline LoC.

#### Alternative 3B: Trait-based static graph (à la JAX `jit`)

**Approach.** Each op returns a strongly-typed expression node; the user
calls `compile()` to run the graph.

**Pros.**
- Whole-graph optimization opportunities.
- No Box+dyn.

**Cons.**
- Eager mode dies — every op return type changes.
- Diverges from PyTorch ergonomics.
- Phase-0 kill switch.

**Cost.** Prohibitive for a PyTorch-faithful project.

#### Alternative 3C: Re-trace on every backward

**Approach.** Don't store a tape; re-execute the forward symbolically
when `backward()` is called.

**Pros.**
- Zero runtime cost in pure forward / inference.
- Memory cheap.

**Cons.**
- Forward must be fully re-runnable (no in-place state, no I/O).
- 2× compute on every backward (not 1× extra — total is 3× a forward
  count: trace, recompute, backprop).

**Cost.** Conceptually clean but the 2× compute is unacceptable for
training-grade workloads.

### Decision 4 — Async / threading

#### Alternative 4A: `rayon` for compute, `tokio` for I/O edges

**Approach.** Compute kernels parallelize via `rayon::par_iter` /
`rayon::scope`. Async I/O (HTTP for dataset fetch, console SSE, gRPC)
uses `tokio`. The two never mix in the same future.

**Pros.**
- rayon is the de-facto Rust compute parallelism.
- tokio is the de-facto Rust async runtime.
- Clear boundary: compute is sync; async is the outer layer.

**Cons.**
- Two parallelism libraries instead of one.
- WASM has rayon disabled (no threads in browser).

**Cost.** Reference; baseline LoC.

#### Alternative 4B: Pure `std::thread` for compute, no async

**Approach.** Implement our own thread pool atop `std::thread`. No
`tokio`, no `rayon`. Async is wrapped via `std::sync::mpsc`.

**Pros.**
- Zero extra deps.
- Maximum control.

**Cons.**
- We rebuild rayon's work-stealing scheduler badly.
- Async APIs (Console SSE, gRPC) become very awkward.

**Cost.** ~+5k LoC of thread-pool code we then maintain forever.

#### Alternative 4C: `tokio` for everything, `tokio::task::spawn_blocking` for compute

**Approach.** Single runtime everywhere; compute kernels live inside
`spawn_blocking`.

**Pros.**
- One runtime.
- Easier futures composition.

**Cons.**
- Compute parallelism via tokio is much weaker than rayon's work-stealing.
- `spawn_blocking` is meant for blocking syscalls, not CPU-heavy fan-out.

**Cost.** Minor LoC, significant perf cost on compute.

### Decision 5 — WASM-first vs native-first

#### Alternative 5A: WASM as first-class CI target

**Approach.** `wasm32-unknown-unknown` is in the CI matrix from day 1
(see `.github/workflows/wasm.yml`). The `rustorch-core` and
`rustorch-autograd` crates are written so they compile cleanly on
wasm32 by default. Native-only code (rayon, mmap) is `#[cfg]`-gated.

**Pros.**
- The wgpu/WebGPU backend (Phase 2) is a real differentiator.
- Forces clean separation between platform-portable and platform-bound code.
- Catches regressions immediately rather than at the Phase 2 effort.

**Cons.**
- Some convenience features (e.g., file mmap of safetensors) need an
  alternative slow-path in WASM.
- Slightly higher cognitive overhead per op author.

**Cost.** Reference; baseline LoC; +1 CI job.

#### Alternative 5B: Native-first, WASM as a Phase 2 surprise

**Approach.** Build natively first; add WASM support when wgpu is ready.

**Pros.**
- Slightly fewer `cfg`s in early code.

**Cons.**
- Inevitable rewrites when WASM-incompatible APIs (e.g., `std::time::Instant`
  in some configs) leak into core.
- Risk of Phase 2 sliding because the surface to fix is wider.

**Cost.** Lower upfront LoC; significantly higher rework risk.

#### Alternative 5C: Skip WASM entirely

**Approach.** Native targets only.

**Pros.**
- Simplest implementation.

**Cons.**
- Forfeits the strongest differentiator vs burn/candle.
- Forfeits browser inference (a real edge use case).

**Cost.** N/A; rejected by Phase-0 product position.

### Decision 6 — Codegen vs hand-coded ops

#### Alternative 6A: Single `ops.yaml` → `build.rs` → Rust source

**Approach.** Each op (≈300 by v1.0) is described once in a YAML schema:
name, signature, backward formula, dispatch table per backend, doc.
A `build.rs` in `rustorch-codegen` parses the YAML and emits Rust
boilerplate (function signatures, autograd `Node` impls, dispatch
shims, eventually Python bindings).

**Pros.**
- Add an op once → it shows up in 6 places automatically.
- Keeps backends in sync.
- Enables auto-generated Python bindings later (P0.4).

**Cons.**
- `build.rs` codegen is opaque to "find references" tooling unless we
  emit `#[doc = "source:ops.yaml line N"]` (we will).
- Schema evolution requires care.

**Cost.** Up-front cost of the schema + build.rs (~1k LoC); pays back
after ~50 ops.

#### Alternative 6B: All ops hand-written

**Approach.** Each op is a hand-written function with a hand-written
backward `Node` impl per backend.

**Pros.**
- Easiest to start.
- No build-time machinery.

**Cons.**
- 300 ops × 6 places to touch = 1800 manual sync points.
- Inevitable drift.

**Cost.** Lower upfront, much higher long-term.

#### Alternative 6C: `proc_macro`-based DSL inside Rust

**Approach.** `op!{ matmul(a, b) -> c { ... } backward { ... } }` macro.

**Pros.**
- Lives inside Rust source, IDE-friendly.

**Cons.**
- Proc macros run at compile time, not reachable for tooling that
  inspects YAML; we'd lose the cross-language sharing benefit.
- Per-op compilation cost.

**Cost.** Medium up-front, fewer cross-cutting wins.

## Decision

For each axis we adopt the following:

| # | Axis | Decision |
|--:|------|----------|
| 1 | Tensor shape model | **1B — runtime-opaque shapes (PyTorch-faithful)** |
| 2 | Backend abstraction | **2A — `trait Backend` with associated types + monomorphization** |
| 3 | Autograd model | **3A — tape-based, thread-local recording** |
| 4 | Async/threading | **4A — rayon (compute) + tokio (I/O), strict separation** |
| 5 | WASM | **5A — first-class CI target from day 1** |
| 6 | Codegen | **6A — single `ops.yaml` consumed by `rustorch-codegen` build.rs** |

These are concretely already reflected in the workspace bootstrap
delivered by P0.2 (commit `23cca27a`):

- `rustorch-core` defines `Tensor` opaquely (Decision 1).
- `rustorch-cpu` is a separate crate so the `Backend` trait can have an
  associated `Storage` (Decision 2).
- `rustorch-autograd` is its own crate with thread-local tape semantics
  (Decision 3).
- `rustorch-cpu` brings `rayon` only on non-wasm targets (Decision 4
  + 5).
- `rustorch-codegen` is reserved as a build-time helper crate (Decision 6).

## Rationale

The unifying theme is **"do what PyTorch does where ergonomics matters,
do what Rust does where safety matters"**.

- **Shapes opaque** because the user docs v0.7.2 already commit to the
  PyTorch-faithful API. Static shapes are a wonderful research project,
  but they are a different product.
- **Trait-based backends** because Rust's monomorphization yields
  PyTorch-comparable performance without virtual calls, and the
  type-level isolation prevents accidental cross-device tensor mixing
  at compile time. The "awkward `to()` across types" objection is
  mitigated by the umbrella `rustorch` crate exposing a `Tensor` alias
  for the default `CpuBackend`, with explicit conversions otherwise.
- **Tape autograd** because higher-order grads need it, and re-trace's
  2× compute is unacceptable for training. The `Arc<dyn Node>` cost is
  dominated by kernel cost on any non-trivial op.
- **rayon + tokio split** because they each excel at one thing and
  fighting that means rebuilding badly. The strict separation is also
  WASM-friendly: rayon disappears on wasm32 cleanly.
- **WASM first-class** because (a) the wgpu/WebGPU backend is the
  product's strongest differentiator, (b) the cleanup cost compounds
  if we wait, (c) "WASM works because we built it that way" is a
  marketing line; "WASM works because we got lucky" is fragile.
- **YAML codegen** because the public API is large (~300 ops) and
  needs to stay in sync across 4–5 backends. Hand-syncing 300 × 5 =
  1500 sites is a guaranteed regression source.

## How does PyTorch do it?

- `aten/src/ATen/core/TensorImpl.h:42` — runtime-opaque shape (`SmallVector<int64_t>`).
- `aten/src/ATen/core/dispatch/DispatchKey.h:15` — DispatchKey enum
  driving the runtime registry (Decision 2C; we reject this approach
  but the file is the canonical reference).
- `torch/csrc/autograd/engine.cpp:120-260` — tape execution loop,
  thread-local `Variable::AutogradMeta`. Direct ancestor of our
  Decision 3A.
- `torch/csrc/autograd/python_function.cpp` — the `torch.autograd.Function`
  custom op extension surface. Mirrored by our `CustomFunction` trait
  in `rustorch-autograd`.
- `torch/csrc/jit/codegen/...` — does *not* drive the eager path;
  irrelevant here, but cited because it's a common confusion.
- `aten/src/ATen/native/native_functions.yaml` — single source of truth
  for native op signatures and codegen. Direct ancestor of our
  Decision 6A.

## How does burn / candle do it?

- **burn** ([`tracel-ai/burn` v0.18](https://github.com/tracel-ai/burn))
  - Decision 1: static (rank-typed) — diverges from rustorch.
  - Decision 2: trait `Backend` with associated `FloatTensorPrimitive` — same.
  - Decision 3: tape (`burn-autodiff` crate) — same.
  - Decision 4: hardware-agnostic kernels via burn-tensor's `Tensor` ops — finer-grained than rayon vs tokio.
  - Decision 5: WASM works but isn't a top priority.
  - Decision 6: macros in the `burn-tensor` crate, not YAML.

- **candle** ([`huggingface/candle`](https://github.com/huggingface/candle))
  - Decision 1: opaque shapes — same as rustorch.
  - Decision 2: enum-based `Storage` (Cpu/Cuda/Metal) — diverges; we pick trait.
  - Decision 3: minimal autograd (gradient-only when needed) — diverges; we want full PyTorch parity.
  - Decision 4: pure compute on owned threads, no rayon — diverges.
  - Decision 5: WASM works but supplemental.
  - Decision 6: hand-coded ops — diverges; we pick codegen.

The take-away: rustorch sits **between** burn and candle. We adopt
candle's opaque shapes (PyTorch fidelity), burn's trait-Backend
(performance), tape autograd from both, and a more deliberate WASM
+ codegen story than either.

## Migration plan

This is the foundational RFC of a new project — there is no prior
internal design to migrate from. Future RFCs that wish to revise any
of the six decisions must:

1. Be filed as a new RFC explicitly listing this file under
   `superseded-by`.
2. Address every "Pros" of the alternative being introduced.
3. Provide a migration path for downstream code that depends on the
   superseded decision.

## Open questions

- [ ] Decision 1 — Should we ship a `view_as_static<S>()` helper for
      power users, or strictly forbid static-shape parallel APIs?
- [ ] Decision 2 — How do we expose `to(&device)` ergonomically across
      different concrete `Tensor<B>` types? `impl<B: Backend> From<...>` chain?
      Or an `AnyTensor` enum at the public API boundary?
- [ ] Decision 3 — Should `create_graph` propagate transitively (PyTorch's
      behavior) or require explicit opt-in at each level?
- [ ] Decision 4 — Do we expose a synchronous `block_on` for tokio at the
      public API edge, or require user code to bring its own runtime?
- [ ] Decision 5 — Is `wasm32-wasi` in scope, or only `wasm32-unknown-unknown`?
      (WASI has `mmap`, etc., easing some pains; but it's a different
      target ecosystem.)
- [ ] Decision 6 — Which of the 6 places (op signature, dispatch, autograd
      Node, Python binding, doc, ONNX export) does codegen handle vs
      manual? Need to scope the YAML schema in RFC-0005.
- [ ] Should `f8` (8-bit floats — Phase 4 quantization) be a first-class
      `Dtype` variant or a wrapper around `u8`?
- [ ] Are there cases where we want a `dyn Backend` trait object after
      all (e.g., a user-facing `Device` enum that holds a `Box<dyn Backend>`)?

## References

- PyTorch source @ commit `01abe1d` (2026-04 vendored copy in `pytorch/`).
- burn 0.18 release notes — https://github.com/tracel-ai/burn/releases/tag/v0.18.0
- candle 0.x — https://github.com/huggingface/candle
- "Lessons from PyTorch's dispatcher" — Sasank Chilamkurthy, 2022 talk.
- "Why JAX is the wrong abstraction for production ML" — internal
  pre-RFC discussion notes (link to be added once notes are public).

## Decision matrix

| Aspect | 1B opaque | 2A trait | 3A tape | 4A rayon+tokio | 5A WASM-1st | 6A codegen |
|--------|-----------|----------|---------|----------------|-------------|------------|
| LoC at v1.0 | baseline | medium | baseline | baseline | +cfg cost | -100s of dup |
| Compile time | baseline | +monomorph | baseline | baseline | +1 CI job | +build.rs cost |
| Runtime perf | baseline | best | baseline | best | n/a | best |
| Ergonomics (PyTorch-faithful) | best | good | best | best | n/a | excellent |
| Implementation cost | baseline | baseline | baseline | baseline | medium | medium |
| Risk if wrong | low | medium | medium | low | medium | medium |
| Decision | **chosen** | **chosen** | **chosen** | **chosen** | **chosen** | **chosen** |
