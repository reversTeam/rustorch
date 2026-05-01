---
id: 0004
title: Backend trait — interface, dispatch, monomorphization
status: Draft
date: 2026-05-01
authors:
  - rustorch team <rfc@rustorch.dev>
supersedes: []
superseded-by: []
---

# RFC-0004: Backend trait — interface, dispatch, monomorphization

## Summary

Concrete computation lives behind a single `trait Backend` with associated
types for `Storage`, `Stream`, and `Event`. Each backend (`CpuBackend`,
`WgpuBackend`, optional `CudaBackend`) implements the trait. The user-facing
`Tensor` is **opaque** (per RFC-0002 Decision C); internally it dispatches
through `Storage`'s enum to the appropriate backend method via a generated
match. This RFC locks the trait surface, error model, stream/event semantics,
and the rule for adding a new backend without breaking downstream code.

## Motivation

By the end of Phase 1 we will have ~100 ops × 3 backends = 300 implementations.
The trait surface is the contract that makes those 300 implementations
type-check together. Deferring this decision means:

- Every kernel author re-invents argument shapes (Tensor vs Storage vs raw bytes).
- Adding a backend requires editing 100 dispatch sites.
- Async backends (CUDA streams, WebGPU queues) have nowhere natural to live.
- Cross-backend tensor transfer (`x.to(&dev)`) becomes ad-hoc.

PyTorch's dispatcher took years to converge. We adopt the trait shape that
the Rust ecosystem (burn, wgpu, candle) has already validated.

## Constraints

- **Object-safety** is *not* required — backends are concrete in the dispatch
  match, never behind `dyn Backend`.
- **`Send + Sync + 'static`** for the backend type itself (so it can sit in
  `static` registries and cross threads).
- **Associated `Storage`** must be `Send + Sync + Clone` (cheap clone — header
  only or `Arc`).
- **Errors**: every fallible method returns `Result<T, BackendError>`.
- **Async support**: backends with native async (CUDA streams, WGPU queue)
  expose `async fn submit` paths; the CPU backend's `Stream` is `()`.
- **Method count cap**: ≤120 methods on the trait. Beyond that we group via
  sub-traits (e.g., `LinalgBackend`).
- **Adding a new backend**: must compile with no edits to `rustorch-core` or
  `rustorch-autograd`. Only `Storage` enum (in `rustorch-core`) gains a variant
  + match arms — these are additive when the enum is `#[non_exhaustive]`.

## Alternatives Considered

### Alternative A — Single `trait Backend` with associated types (chosen)

**Approach.**

```rust
pub trait Backend: Sized + Send + Sync + 'static {
    type Storage: Send + Sync + Clone;
    type Stream:  Send + Sync;
    type Event:   Send + Sync;

    fn name() -> &'static str;
    fn devices() -> Vec<Device<Self>>;
    fn stream(device: &Device<Self>) -> Self::Stream;

    // ~100 op methods follow:
    fn matmul(&self, a: &Self::Storage, b: &Self::Storage) -> Result<Self::Storage>;
    fn add(&self, a: &Self::Storage, b: &Self::Storage) -> Result<Self::Storage>;
    /* … */
}
```

**Pros.**
- Strong typing per backend — each backend's `Storage` is its own type.
- Monomorphization picks the right method body at the call site (no virtual
  dispatch).
- Adding a new method with a default body is a non-breaking change.

**Cons.**
- Trait surface is large (~100 methods).
- Each method signature uses the same `&self` pattern — boilerplate.

### Alternative B — Multiple sub-traits (`LinalgBackend`, `ConvBackend`, ...)

**Approach.** Split the 100-method trait into `LinalgBackend` (matmul, bmm,
addmm, ...), `ActivationBackend` (relu, gelu, ...), `ReduceBackend` (sum,
mean, ...), etc. Backend types implement each.

**Pros.**
- Smaller individual trait surfaces.
- Optional backends (e.g., `BatchNormBackend` is opt-in for the CPU backend
  but mandatory for CUDA).

**Cons.**
- 6–10 traits to keep in sync.
- Generic functions need long bounds: `fn foo<B: LinalgBackend + ActivationBackend>`.
- Trait-method lookup via sub-trait can confuse rustc on large generics.

**Cost.** Modest LoC win in exchange for verbose generic bounds.

### Alternative C — `DispatchKey` runtime registry (PyTorch's path)

**Approach.** Each (op, backend) pair is registered at startup into a global
hash map keyed by `(OpId, BackendId)`. The runtime looks up the kernel.

**Pros.**
- Third-party kernels register at runtime with no rebuild.
- Mirrors PyTorch C++.

**Cons.**
- Hash-table cost per op call.
- `unsafe` everywhere because the type-erased function pointers.
- Hostile to Rust's type system (Boxed `dyn Fn(&[...]) -> [...]`).

**Cost.** Implementation cost high; ergonomic cost very high.

### Alternative D — Free functions per op + cfg-gated impl modules

**Approach.** Each op is a free function with a `match Storage { ... }` body
that calls into a backend-specific module (`crate::cpu::matmul_f32`, etc.).
No trait at all.

**Pros.**
- Simplest surface — no trait dance.

**Cons.**
- Adding a backend means editing every op's match.
- No way to express a "default body" (e.g., `cross_entropy` = `log_softmax`
  + `nll` for any backend).
- Generic algorithms (autograd) lose their handle on the backend type.

**Cost.** Lowest upfront LoC; highest long-term sync cost.

## Decision

We adopt **Alternative A: single `trait Backend` with associated types**.
The concrete shape:

```rust
// crates/rustorch-core/src/backend/mod.rs   (lands with P1.1 / P1.2)

pub trait Backend: Sized + Send + Sync + 'static {
    type Storage: Send + Sync + Clone + std::fmt::Debug;
    type Stream:  Send + Sync;
    type Event:   Send + Sync;

    /// Identifying name, used in `Display` and error messages.
    fn name() -> &'static str;

    /// Enumerate devices. For CPU = vec![0]; for CUDA = vec![0..N].
    fn devices() -> Vec<Device<Self>>;

    /// Get a stream — the unit of async work submission.
    fn stream(device: &Device<Self>) -> Self::Stream;

    /// Synchronize a stream; CPU backend is a no-op.
    fn sync(stream: &Self::Stream) -> Result<(), BackendError>;

    /// Allocate a Storage of the given byte size, aligned.
    fn alloc(device: &Device<Self>, size_bytes: usize, align: usize)
        -> Result<Self::Storage, BackendError>;

    // Op methods follow. Grouped by category for readability.
    // Element-wise:
    fn add(&self, a: &Self::Storage, b: &Self::Storage)
        -> Result<Self::Storage, BackendError>;
    fn mul(&self, a: &Self::Storage, b: &Self::Storage)
        -> Result<Self::Storage, BackendError>;
    /* ... */

    // Linalg:
    fn matmul(&self, a: &Self::Storage, b: &Self::Storage)
        -> Result<Self::Storage, BackendError>;
    /* ... */

    // Reductions, conv, norm, etc.
}

#[derive(thiserror::Error, Debug)]
pub enum BackendError {
    #[error("shape mismatch: {op}: lhs={lhs:?}, rhs={rhs:?}")]
    ShapeMismatch { op: &'static str, lhs: Vec<usize>, rhs: Vec<usize> },
    #[error("dtype mismatch: {op}: lhs={lhs:?}, rhs={rhs:?}")]
    DtypeMismatch { op: &'static str, lhs: Dtype, rhs: Dtype },
    #[error("device mismatch: {op}: lhs={lhs:?}, rhs={rhs:?}")]
    DeviceMismatch { op: &'static str, lhs: String, rhs: String },
    #[error("out of memory: {device}: requested {bytes} bytes")]
    OutOfMemory { device: String, bytes: usize },
    #[error("unsupported: {0}")]
    Unsupported(&'static str),
    #[error("numerical: {0}")]
    Numerical(&'static str),
    #[error("backend-specific: {0}")]
    Backend(Box<dyn std::error::Error + Send + Sync>),
}

pub struct Device<B: Backend> {
    pub id: usize,
    pub stream: B::Stream,
}
```

The opaque user-facing `Tensor` (RFC-0002) carries an internal
`Storage` enum:

```rust
// rustorch-core
#[non_exhaustive]
pub enum Storage {
    Cpu(<CpuBackend as Backend>::Storage),
    #[cfg(feature = "cuda")] Cuda(<CudaBackend as Backend>::Storage),
    Wgpu(<WgpuBackend as Backend>::Storage),
}
```

Op dispatch is a **single match per op** in `rustorch-core` — generated by
RFC-0005 codegen from `ops.yaml`. Ops with identical signatures across
backends are one-liner matches that dispatch to `B::matmul(a, b)`.

For the `Test` / `Mock` backend used in tests, the trait can be implemented
in <100 LoC with placeholder methods that panic on call — useful for
testing autograd traversal without numerical correctness.

## Rationale

- **Associated types** preserve type-level isolation: a CUDA storage can
  never appear in a CPU op signature.
- **Storage-as-enum at the user layer** keeps the `Tensor` type concrete (per
  RFC-0002), while the backend-agnostic algorithms (autograd, codegen) can
  still operate on `B: Backend`.
- **`#[non_exhaustive]`** on `Storage` makes adding a new backend additive.
- **Errors via `BackendError` enum** — stable across backends so the
  user-facing `Result<Tensor>` doesn't leak backend-specific error types.
- **Stream + Event** because CUDA and WGPU expose them; the CPU backend's
  unit-typed equivalents make the abstraction free for synchronous code.

## How does PyTorch do it?

- `aten/src/ATen/core/dispatch/Dispatcher.cpp:120-300` — runtime dispatcher
  with hash-keyed `(OpId, DispatchKey)` lookup. We deliberately reject this
  approach in favor of monomorphization (perf + Rust-idiomatic).
- `aten/src/ATen/native/native_functions.yaml` — single source of truth for
  op signatures + dispatch table. We adopt this pattern in RFC-0005 with
  `ops.yaml`.
- `c10/util/intrusive_ptr.h` — PyTorch's intrusive refcount for storages.
  Rust's `Arc` covers this case at the cost of one extra indirection per
  storage (acceptable).

## How does burn / candle do it?

- **burn** (`burn-tensor` crate) uses a similar `trait Backend` with
  associated `Tensor` primitive types (`FloatTensorPrimitive`,
  `IntTensorPrimitive`). Slightly more granular than ours; our merged
  `Storage` keeps the trait surface smaller.
- **candle** uses an enum-only approach without a trait. Adding a new
  backend means editing every op. We prefer the trait so default-method
  implementations exist (e.g., `cross_entropy` = `log_softmax + nll` is
  shared across backends).
- **wgpu** itself uses `trait` + concrete impls (`Vulkan`, `Metal`,
  `Dx12`, `WebGPU`). We mirror its pattern at the crate-organization level
  (one crate per backend).

## Migration plan

Foundational. Future RFCs that wish to revise the trait surface (e.g.,
adding async generic methods once Rust async traits stabilize fully) must
provide a deprecation window where both signatures coexist.

## Open questions

- [ ] Should `Backend` be a **zero-sized type** convention (e.g.,
      `impl Backend for CpuBackend { ... }` with `CpuBackend` being a unit
      struct) and ops are static methods? Or instance methods on `&self`
      so backends can carry config? Current tilt: instance methods.
- [ ] Should we expose a **`DeviceLocation`** abstraction over physical
      device IDs (e.g., for unified memory on Apple Silicon)? Probably
      Phase 4.
- [ ] How do we handle **fallback ops**: if a backend doesn't implement
      a method, should the trait have a default that returns
      `Err(Unsupported)`, or should the absence be a compile error?
      Current decision: opt-in defaults via `default fn`, with explicit
      `Unsupported` for unimplemented ops.
- [ ] **Cross-backend transfer** (`x.to(&dev)`) goes through `to_cpu` +
      `from_cpu` round-trip by default. Should we expose direct GPU↔GPU
      transfer (`cudaMemcpyPeer`, `wgpu::Queue::copy_buffer_to_buffer`)
      via a `BackendTransfer` sub-trait? Phase 4 question.
- [ ] **Backend feature negotiation**: at runtime, a wgpu device may not
      support f64. How do we gracefully fail and surface a useful error?
      Current decision: backend's `alloc` returns `Err(Unsupported)` if
      the dtype is unsupported on the device.
- [ ] **`Layout` traversal**: does the trait take `Storage` + `Layout` or
      a wrapper `(&Storage, &Layout)` pair? Decided: pair, to avoid
      duplicating shape info on the storage.

## References

- PyTorch dispatcher — `aten/src/ATen/core/dispatch/`
- burn `trait Backend` — `crates/burn-tensor/src/tensor/backend/base.rs`
- candle `Storage` enum — `candle-core/src/storage.rs`
- wgpu portable abstraction — https://wgpu.rs

## Decision matrix

| Aspect | A trait+assoc (chosen) | B sub-traits | C dispatcher | D free fns |
|--------|:----------------------:|:------------:|:------------:|:----------:|
| Add a new backend | additive | additive | additive | edit 100 sites |
| Trait surface size | ~100 methods | smaller each | n/a | none |
| Ergonomics in user code | best | verbose generics | n/a | best |
| Perf | best (mono) | best | hash lookup | best |
| Default method bodies | yes | yes | no | no |
| Type safety | best | best | poor | poor |
| Decision | **chosen** | rejected | rejected | rejected |
