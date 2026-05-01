---
id: 0002
title: Tensor design — Storage, Layout, dtype, refcount
status: Draft
date: 2026-05-01
authors:
  - rustorch team <rfc@rustorch.dev>
supersedes: []
superseded-by: []
---

# RFC-0002: Tensor design — Storage, Layout, dtype, refcount

## Summary

The `Tensor` type is a header (shape + strides + offset + dtype + device + autograd
metadata) over a refcounted `Storage` blob. Shape is **runtime-opaque** (per
RFC-0001 Decision 1). Storage uses `Arc` for shared ownership. The public API
follows the v0.7.2 user-docs freeze:
`Tensor::zeros::<f32>([2, 3], &dev)?`, `x.to(&dev)?`, `x.detach()`,
`x.requires_grad_(true)`, all `Result`-typed.

## Motivation

The Tensor surface is the most-used type in the entire crate; thousands of
call sites will touch it. Locking down its **public layout, error type, view
semantics, and version-counter invariants** before P1.1 starts coding is the
difference between "Phase 1 ships in 4 months" and "Phase 1 ships and we
rewrite three months later".

Concrete pain points motivating this RFC:

- PyTorch's `TensorImpl` carries 30+ fields (`storage_`, `sizes_`, `strides_`,
  `numel_`, `dtype_`, `device_`, `is_contiguous_`, `version_counter_`,
  `autograd_meta_`, `named_tensor_meta_`, ...). Picking what to inline vs box
  is performance-critical.
- Rust's borrow checker does not love refcounted aliasing tracking — we need
  a discipline that the compiler can verify, not just hope works.
- View vs copy semantics must be **predictable** so users know when a write
  invalidates someone else's grad.
- The dtype enum interacts with serialization (safetensors header), with the
  HuggingFace dtype IDs, with WGSL kernels (no f64), with WASM (no SIMD f64).
  Pick wrong once and every kernel author pays.

## Constraints

- **Public API frozen by user-docs v0.7.2.** No deviation: signatures must
  type-check verbatim in user code lifted from those docs.
- **`Tensor: Send + Sync + Clone`.** Cheap clone (header-only).
- **No `unsafe` in user-facing API.** Internal `unsafe` is fine where bounded.
- **`#[no_std]` compatible** for the Tensor + Layout subset (Phase 6 stretch).
- **`Tensor: Sized`** — no `?Sized` slop in trait objects.
- **Result-typed.** Every fallible op returns `Result<T, RusTorchError>`.
- **Strict aliasing.** Writes through one view must increment a version counter
  visible to other views (see §"Mutation safety" below).

## Alternatives Considered

### Alternative A — Tensor is parametric over Backend (`Tensor<B: Backend>`)

**Approach.** burn-style. Each backend has its own concrete `Tensor` type,
specialized via monomorphization.

**Pros.**
- Cross-device misuse is a compile error.
- Maximum perf (no virtual dispatch).

**Cons.**
- Public API balloons: every function takes `<B: Backend>`.
- Cross-device `to(&device)` returns a different concrete type — awkward.

**Cost.** ~+30% of public-API LoC.

### Alternative B — Tensor is opaque, Storage is `Arc<dyn StorageVTable>`

**Approach.** candle-style enum (`Cpu`/`Cuda`/`Wgpu`) replaced by `Arc<dyn>`.

**Pros.**
- One concrete `Tensor`. Cross-device transfer is `tensor.to(&dev)` returning
  the same type.
- Minimum monomorphization.

**Cons.**
- Virtual dispatch on every op.
- Lifetime / variance gets tricky with `dyn Trait`.

**Cost.** ~baseline LoC; ~10–15% perf hit on tight kernels.

### Alternative C — Hybrid: opaque `Tensor` whose internal Storage is an enum

**Approach.** `Tensor` has a single concrete type. Internal `Storage` is an
enum `Cpu(...) | Wgpu(...) | Cuda(...)`. Op dispatch is a `match` inside
`Tensor::matmul` that calls the appropriate kernel.

**Pros.**
- Concrete `Tensor` type — best for PyTorch-faithful ergonomics.
- No virtual dispatch — the `match` monomorphizes per branch.
- Adding a new backend updates one enum + one match per op.

**Cons.**
- Every op has a `match` ladder.
- Adding a backend is technically a breaking change to the enum (mitigated
  by `#[non_exhaustive]`).

**Cost.** ~+5% LoC vs Alternative B; near-zero perf cost.

## Decision

We adopt **Alternative C: opaque `Tensor`, internal `Storage` enum**.

The concrete shape locked by this RFC:

```rust
// crates/rustorch-core/src/tensor.rs

#[derive(Clone)]
pub struct Tensor {
    inner: Arc<TensorInner>,
}

struct TensorInner {
    storage: Arc<Storage>,         // refcounted blob, see below
    layout:  Layout,                // shape + strides + offset + contiguous flag
    grad_fn: Option<Arc<dyn Node>>, // None for leafs / detached tensors
    grad:    AtomicGradSlot,        // Option<Tensor>, atomic-swappable
    version: AtomicU64,             // bumped on in-place mutation
    requires_grad: bool,            // immutable after construction (call .requires_grad_())
}

#[non_exhaustive]
pub enum Storage {
    Cpu(CpuStorage),
    #[cfg(feature = "cuda")]  Cuda(CudaStorage),
    Wgpu(WgpuStorage),
}

pub struct Layout {
    shape:       SmallVec<[usize; 8]>,  // most tensors are <=8d
    strides:     SmallVec<[isize; 8]>,  // signed for negative strides (flip)
    offset:      usize,                  // in elements, not bytes
    dtype:       Dtype,
    contiguous:  bool,                   // memo of .is_contiguous()
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Dtype {
    F32, F64, Bf16, F16, I32, I64, U8, Bool,
}
```

(Reference impl in P1.1; signatures locked here.)

## Rationale

- **`Arc<TensorInner>`, header is `Arc`-d.** `Tensor::clone()` is one atomic
  increment, the same as Python's `Tensor` reference. No data copied.
- **`Storage` is its own `Arc`** so a view (`x.permute()`) shares storage but
  has its own header (different layout, different grad_fn). PyTorch does the
  same.
- **`SmallVec<[_; 8]>`** because >99% of real tensors are at most 8d
  (transformer attention is 5d, image data is 4d). Spilling to heap for
  exotic 9d cases is fine and rare.
- **Version counter** lives on `TensorInner` so views share storage but each
  view tracks its own write history — required for autograd's `SavedVariable`
  (see RFC-0003).
- **`Dtype` is `#[repr(u8)]`** so safetensors and serializers can pack it
  trivially.
- **`AtomicGradSlot`** so concurrent backward (multi-stream future Phase 5)
  can accumulate without a global lock.

## How does PyTorch do it?

- `aten/src/ATen/core/TensorImpl.h:42-160` — `TensorImpl` struct: storage,
  sizes_and_strides, dtype, autograd_meta_. The C++ inheritance hierarchy
  (Tensor → TensorBase → c10::IValue) maps naturally to our `Tensor` →
  `Arc<TensorInner>` indirection.
- `c10/util/SmallVector.h` — PyTorch's own `SmallVector` for shapes/strides.
  We use the well-tested `smallvec` crate.
- `torch/csrc/autograd/variable.cpp:38-90` — `Variable::AutogradMeta` =
  the `(grad_fn, grad, requires_grad, version_counter)` quartet. Same fields
  inlined into `TensorInner` rather than separated.
- `aten/src/ATen/core/TensorOptions.h` — dtype + device + layout + memory_format
  options. We collapse to `(Dtype, Device)` and put memory_format in the layout.

## How does burn / candle do it?

- **burn 0.18** — `BurnTensor<B, T, D>` is parametric on `Backend`, element
  type, and rank. We diverge: we want PyTorch ergonomics over compile-time
  shape safety.
- **candle** — `Tensor` is a concrete type; internally an `enum Storage`
  (`Cpu(CpuStorage) | Cuda(CudaStorage) | Metal(MetalStorage)`). We adopt
  this exact pattern. Candle does not track autograd in the Tensor header;
  we do (per RFC-0003 tape model).
- **dfdx** — type-level shapes (`Tensor<Rank2<3, 4>, f32, Cpu>`). Strongest
  static guarantees, biggest divergence from PyTorch.

## Migration plan

This is a foundational RFC. RFC-0001 already adopted the global decisions
that make this concretely buildable. Future RFCs that wish to revise the
Tensor layout (e.g., add a `name_dim_meta_` field for named dims, à la
PyTorch experimental) must:

1. File a new RFC explicitly listing this as superseded.
2. Provide a migration shim so existing `Tensor` users compile with at most a
   `cargo fix`.

## Open questions

- [ ] Do we support **negative strides** (PyTorch's `.flip()` returns a strided
      view with negative stride)? Current decision: yes (`strides: isize`).
      But `numpy` allows it and it complicates SIMD code paths. Confirm in P1.1.
- [ ] **Unified `Device`**: should `Device` be `enum Device { Cpu, Cuda(usize),
      Wgpu(WgpuDevice) }` or `Arc<dyn DeviceImpl>`? Current default: enum.
- [ ] **Aliased writes** vs `version_counter`: do we allow the user to silence
      version-mismatch panics in autograd via `tensor.with_no_version_check(|t| ...)`?
      Risk vs ergonomics tradeoff.
- [ ] Should `Tensor: Hash`? `==` on tensors compares values element-wise (PyTorch
      semantics) — `Hash` would have to be element-wise too, which is expensive.
      Recommend: do not derive `Hash`, document.
- [ ] Should we support **complex dtypes** (`Complex64`, `Complex128`)? Phase 6
      stretch; not in v1.0.
- [ ] **NamedTensor** (PyTorch's experimental named-dim feature): out of scope
      for v1.0. Note in docs.
- [ ] **Quantized dtypes** (i4, qint8): RFC-0002.1 (TBD) or part of Phase 3
      Quantization?

## References

- PyTorch `TensorImpl` source — `aten/src/ATen/core/TensorImpl.h` (commit `01abe1d`).
- candle `Tensor` source — https://github.com/huggingface/candle/blob/main/candle-core/src/tensor.rs
- burn `Tensor` source — https://github.com/tracel-ai/burn/tree/main/crates/burn-tensor
- safetensors dtype mapping — https://huggingface.co/docs/safetensors/index
- IEEE 754 binary16 / bfloat16 — https://en.wikipedia.org/wiki/Bfloat16_floating-point_format

## Decision matrix

| Aspect | Alt A (parametric `<B>`) | Alt B (`dyn Storage`) | Alt C (enum Storage) (chosen) |
|--------|:------------------------:|:---------------------:|:-----------------------------:|
| API ergonomics | bad (generics everywhere) | good | **best (PyTorch-faithful)** |
| Cross-device `.to(&dev)` | painful | trivial | **trivial** |
| Perf (vs theoretical max) | best | -10-15% | **-1-2%** |
| LoC | +30% | baseline | **+5%** |
| Add new backend | trait impl | trait impl | enum + match arms |
| Compile-time cross-device safety | yes | no | no |
| Decision | rejected | rejected | **chosen** |
