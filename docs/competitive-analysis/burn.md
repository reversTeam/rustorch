# Competitive analysis — `burn`

| Field | Value |
|-------|-------|
| Repository | [tracel-ai/burn](https://github.com/tracel-ai/burn) |
| Maintainer | Tracel AI (commercial — funded team) |
| Reference version | 0.18 (most recent stable at the time of writing) |
| License | MIT OR Apache-2.0 |
| Mission | Multi-backend deep-learning framework in pure Rust |
| Status as competitor | **Closest direct competitor** to rustorch |

## Tensor model

burn uses **statically-typed shapes** via const generics:

```rust
pub struct Tensor<B: Backend, const D: usize, K = Float>
where K: TensorKind<B>;

let x: Tensor<CpuBackend, 2, Float> = /* ... */;
let y = x.matmul(w);  // rank checked at compile time
```

- The rank `D` is a const generic; the element kind `K` is a marker type
  (`Float`, `Int`, `Bool`).
- The actual storage layer (`B::FloatTensorPrimitive<D>`) is the backend's
  associated type — flat memory, opaque to user code.
- Shape *values* (not just rank) are **runtime-tracked** inside the primitive;
  burn does not encode dimensions at the type level beyond rank.

**Mapping to rustorch:** Diverges from RFC-0001 Decision 1 (we adopt
runtime-opaque shapes for PyTorch fidelity). burn's choice is excellent
for static guarantees but undermines the "feels-like-PyTorch" goal —
every op signature carries a `<const D: usize>`, every dim-changing op
needs an explicit constant.

Reference: `crates/burn-tensor/src/tensor/mod.rs`.

## Backend trait

burn's `trait Backend` carries multiple associated primitive types
(roughly one per element kind):

```rust
pub trait Backend: Clone + Default + Sized + Send + Sync + 'static {
    type Device: ...;
    type FloatTensorPrimitive<const D: usize>: ...;
    type IntTensorPrimitive<const D: usize>: ...;
    type BoolTensorPrimitive<const D: usize>: ...;
    type FullPrecisionBackend: Backend<...>;
    /* ~150 op methods */
}
```

- Backends implemented in-tree: `Candle` (delegates to candle), `NdArray`
  (CPU via ndarray), `Tch` (libtorch wrapper), `Wgpu` (multi-platform GPU),
  `Cuda`, `LibTorch` (alias).
- 150+ op methods on the trait; sub-traits (e.g., `ActivationOps`)
  group some.

**Mapping to rustorch:** Same direction as RFC-0004 Decision A (`trait Backend`
with associated `Storage`). The differences:
- burn carries multiple `*TensorPrimitive` types (one per kind); we collapse
  into a single `Storage` enum at the user-facing boundary while keeping
  per-backend types internally.
- burn already has a sub-trait split (e.g., `LinalgOps`, `ActivationOps`);
  we adopt this pattern progressively when the unified trait crosses ~100
  methods (per RFC-0004 open question).

Reference: `crates/burn-tensor/src/tensor/backend/base.rs`.

## Autograd

`burn-autodiff` wraps any `Backend B` and produces an
`Autodiff<B>` backend. Internally:

- A **graph** is built per-backward — burn re-traces tensor operations
  symbolically when `.backward()` is called.
- Each op is required to declare a backward via the `Backward` trait,
  conceptually similar to PyTorch's `torch.autograd.Function`.

**Mapping to rustorch:** Different from RFC-0003 Decision A (we use a
thread-local **tape**). burn's "wrap a backend" model has the advantage
of compile-time isolation between forward-only and forward+backward
contexts; the disadvantage is a less natural higher-order grad story.

burn does support higher-order grads via the same `Autodiff<Autodiff<B>>`
nesting, which is elegant. We trade that elegance for PyTorch-faithful
ergonomics.

Reference: `crates/burn-autodiff/src/lib.rs`.

## Module system + derive

burn provides `#[derive(Module)]` which auto-generates parameter
collection, device-move, save/load. Comparable to what we plan in P1.6.

```rust
#[derive(Module, Debug)]
pub struct Mlp<B: Backend> {
    fc1: Linear<B>,
    fc2: Linear<B>,
}
```

The derive recurses into fields that implement `Module`. Same model as
RFC-0002's planned `#[derive(Module)]`.

## Supported ops (top 50)

burn supports the canonical PyTorch surface for tensors (`add`, `mul`,
`matmul`, `bmm`, `conv2d`, `relu`, `gelu`, `softmax`, `layer_norm`,
`batch_norm`, `embedding`, `cross_entropy`, `mse`, `dropout`, `attention`,
`transformer*`, ...). Coverage is roughly **80-85% of common training
needs** vs PyTorch.

Notable gaps observed at the time of writing:
- Sparse tensor support is partial.
- ONNX import is more developed than ONNX export.
- Some PyTorch-specific extension points (custom autograd Functions
  taking arbitrary Python state) cannot be expressed because the trait
  surface is rust-typed.

## WASM support

burn compiles to `wasm32-unknown-unknown`. Their `wgpu` backend works
through `WebGPU` in the browser. However, WASM is **not** their headline
target — examples exist but the bundle-size discipline isn't strict.
A typical burn-wgpu example clocks in around 6-8 MB after `wasm-opt -Oz`.

**Mapping to rustorch:** rustorch elevates WASM to first-class with
hard CI gates from Phase 0 (RFC-0006). We expect to land smaller bundles
and a more maintained browser-demo pipeline as a result, though burn
already provides a working baseline.

## CUDA

`burn-cuda` exists. Implementation is direct CUDA bindings + cuBLAS /
cuDNN dispatch where applicable. Maturity is on par with the wgpu
backend (~Phase 4 effort for us).

## Distributed

No native DDP / FSDP at the time of writing. The `Tch` backend can
delegate to libtorch's distributed primitives, but there's no first-party
`burn-distributed` crate.

**Mapping to rustorch:** Phase 5 ProcessGroup / DDP / FSDP plans are
parity-with-PyTorch from day 1 once they land. This is a small but
real differentiator.

## Mobile / embedded

`no_std` is **not** supported by burn. Mobile (iOS/Android) targets work
when a Rust crate is embedded into a host app, but no first-party
mobile demo with a tight binary-size budget.

## Python bindings

burn does not ship Python bindings as a first-party crate. Some community
projects exist; none are maintained at the level of `tch-rs`'s libtorch
parity.

## ONNX

burn has `burn-import` for ONNX import. Export is more rudimentary.

## Maturity

- GitHub stars (Q1 2026): ~10k.
- Production users: at least 4 listed on the README (some in stealth);
  Tracel AI dogfoods burn.
- Funded development since 2023 — the project is sustainable and
  cadence is steady (monthly minor releases).
- 100+ contributors.

## Code size

`tokei` on the burn workspace:

- `crates/burn-tensor`: ~25k LoC
- `crates/burn-autodiff`: ~10k LoC
- `crates/burn-train`: ~5k LoC
- All tracked crates: ~100k LoC

Big project, comparable in scale to what rustorch is committing to.

## Limitations vs rustorch goals

1. **Static-shape API** — pulls user code away from PyTorch ergonomics.
2. **No tape-based autograd** — re-trace model is elegant but limits
   instrumentation tooling (no flat tape to dump for debugging).
3. **WASM not first-class** — works but not headline target.
4. **No first-party Python bindings** — limits adoption funnel.
5. **No first-party distributed (DDP/FSDP)** — the `Tch` backend delegates
   to libtorch but there's no native Rust path.

## Could we contribute instead of competing?

**Yes, with caveats.** Concrete avenues:
- File RFCs in `tracel-ai/burn` proposing a runtime-shape track parallel
  to the static-shape one. This would meet PyTorch-faithful users halfway.
- Contribute WASM hardening: a CI matrix entry, bundle-size budgets, a
  ResNet-18 / GPT-2 small demo meeting <10MB.
- Contribute a tape-based autodiff variant alongside the existing
  re-trace model.

**However**, the architectural divergence on shape model and autograd is
significant enough that contributing creates a fork-shaped patchset
rather than a clean PR series. The pragmatic call is to build rustorch
with explicit interop with burn (load each other's safetensors, share
ops.yaml schema where possible).

## Quick PR / feasibility scan

Open issues and discussions reviewed: ~20 issues touching Tensor model
or autograd; the maintainers are active and responsive. A focused PR
adding (e.g.) WASM CI tightening would be welcome and merged in 2-3
weeks; an architectural PR (parallel autograd) would face a much higher
review bar.

## Verdict

burn is the most credible Rust competitor to rustorch and a reference
worth studying continuously. The decision to build rustorch rather than
contribute is justified by:

1. PyTorch-faithful ergonomics (RFC-0001 Decision 1) — impossible to
   bolt onto burn's static-shape baseline without breaking their users.
2. Tape-based autograd (RFC-0003) — a different mental model.
3. WASM-first discipline (RFC-0006) — burn is good but not first-class.
4. Faithful Python bindings (Phase 6) — a clear gap in burn's roadmap.

We will treat burn as **prior art to study, not avoid**. Where we adopt
the same answer (e.g., `trait Backend` shape, `#[derive(Module)]`), we
say so explicitly in the relevant RFC.
