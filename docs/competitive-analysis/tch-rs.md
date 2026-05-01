# Competitive analysis — `tch-rs`

| Field | Value |
|-------|-------|
| Repository | [LaurentMazare/tch-rs](https://github.com/LaurentMazare/tch-rs) |
| Maintainer | Laurent Mazare (also primary developer of `candle`) |
| Reference version | 0.18.x (tracking libtorch 2.5+) |
| License | MIT OR Apache-2.0 |
| Mission | Idiomatic Rust bindings to **libtorch** (the C++ side of PyTorch) |
| Status as competitor | **Not a competitor** — different category (FFI wrapper, not a port) |

## Executive summary

`tch-rs` is **not** a Rust port of PyTorch. It is a thin Rust layer over
the libtorch C++ ABI. Every `tch::Tensor` is a Rust handle holding a
C++ `at::Tensor*`, and every op call traverses an `extern "C"` boundary
into libtorch's dispatcher.

This makes `tch-rs`:

- A **non-competitor** for rustorch's mission (pure Rust + WASM).
- A **valid alternative** for users who want PyTorch behavior bit-for-bit
  in a Rust process today, accepting the libtorch dependency.

The remainder of this document explains why and identifies who should
pick `tch-rs`.

## What it is — architecturally

```rust
// from tch::wrappers::tensor
pub struct Tensor {
    c_tensor: *mut C_tensor,   // owned C++ at::Tensor handle
}

// every op is an FFI call
impl Tensor {
    pub fn matmul(&self, other: &Tensor) -> Tensor {
        let c_tensor = unsafe_torch_err!(atg_matmul(self.c_tensor, other.c_tensor));
        Tensor { c_tensor }
    }
}
```

- The shim layer is auto-generated from PyTorch's
  `Declarations.yaml` — the same source of truth that `torch::*` C++ ops
  use. This guarantees ~99% PyTorch op coverage automatically and keeps
  the binding in lockstep with each libtorch release.
- Tensor data lives **inside libtorch** — Rust never owns the bytes. A
  `tch::Tensor` is a smart pointer; cloning bumps a libtorch refcount.
- Autograd, dispatcher, allocator, profiler — **all libtorch**. `tch-rs`
  does not reimplement any of this.

References:
- `src/wrappers/tensor_generated.rs` (auto-generated, ~10k LoC)
- `src/wrappers/tensor.rs` (hand-written ergonomics)
- `torch-sys/src/c_generated.rs` (FFI bindings)

## Build complexity — the libtorch dependency

This is the single biggest practical cost of `tch-rs`:

1. **libtorch must be present at build time.**
   - Set `LIBTORCH=/path/to/libtorch` or `LIBTORCH_USE_PYTORCH=1` (uses
     `torch` from the active Python env).
   - Without it, `cargo build` fails immediately with a `build.rs`
     panic.
2. **libtorch is a >2 GB dependency** (CUDA flavor; CPU-only is ~500 MB).
3. **Glibc / libstdc++ ABI lock-in** — pre-built libtorch wheels target
   specific Linux distros; mixing toolchains causes runtime crashes
   (the infamous `_GLIBCXX_USE_CXX11_ABI` flag).
4. **No WASM** — libtorch does not target wasm32. `tch-rs` cannot
   build for the browser.
5. **No `no_std`** — libtorch needs a full POSIX runtime.
6. **Cross-compilation is painful** — must cross-compile libtorch first,
   then point `tch-rs` at the cross-built artifacts.

**Mapping to rustorch:** None of these constraints apply to a pure-Rust
port. `cargo install rustorch` will work everywhere `cargo install`
does.

## API ergonomics

The `tch-rs` API is **deliberately PyTorch-shaped** — that is the entire
selling point. Where it diverges, it is to be more idiomatic Rust:

```rust
let device = Device::cuda_if_available();
let xs = Tensor::randn(&[3, 4], (Kind::Float, device));
let ys = xs.matmul(&xs.tr());
let z = ys.softmax(-1, Kind::Float);
println!("{}", z);   // tensor formatted like Python
```

Notable Rust adaptations:
- `Kind` enum for dtype (vs `torch.dtype`).
- Builder-style optional args on calls that take many in Python.
- `&Tensor` borrows for input ops; ops returning new tensors return
  by value.
- Error handling via `Result<Tensor, TchError>` on the `_*_err` variants
  (most ops also have an unwrapping form that panics on libtorch error).
- `tch::nn` module mimics `torch.nn` (Linear, Conv2d, etc.) with a
  `Path`-based parameter registry comparable to candle's `VarBuilder`.

For someone fluent in PyTorch, `tch-rs` is the **shortest-Rust-distance**
between "I have a PyTorch model" and "it runs in a Rust process". burn,
candle, and rustorch all require an architectural translation;
`tch-rs` does not.

## Performance

- **Identical to PyTorch C++ runtime** — bit-exact. No measurable
  overhead from the FFI boundary on tensor sizes >1k elements; small-
  tensor hot paths (sub-millisecond ops) show 1-3% overhead from the
  Rust→C++ call.
- All CUDA / cuDNN / NCCL acceleration is available transparently.
- All PyTorch features land automatically when libtorch ships them
  (FlashAttention 2, scaled_dot_product_attention, torch.compile if
  opted in via the C++ API).

This is the **highest-perf option** in the Rust ML ecosystem today,
provided the libtorch constraints are acceptable.

## Use cases — when `tch-rs` is the right choice

**Pick `tch-rs` if:**

1. **Migrating an existing PyTorch service to Rust** for safety/perf
   reasons but the model graph itself is already debugged in Python.
   Drop in `tch-rs`, port the surrounding service code, ship.

2. **Need bit-for-bit PyTorch parity right now** — research reproduction,
   regression-test environments, scientific work where PyTorch is the
   reference oracle.

3. **Have libtorch already deployed** in a Linux/CUDA fleet and want
   to add Rust-authored services that share the same binary
   environment (avoids divergent dep stacks).

**Pick rustorch / burn / candle instead if:**

- WASM / browser / serverless deployment.
- `no_std` / embedded / mobile targets.
- Pure-Rust toolchain is a hard requirement (audit, supply-chain, no
  C++ deps).
- Want the model code itself to be Rust-native (not just the surrounding
  glue).
- Want a single `cargo build` to produce a binary with no system
  libtorch dependency.

## Maturity

- GitHub stars (Q1 2026): ~5k.
- Releases tracked against libtorch versions (every 2-4 months).
- Same maintainer as `candle` — Laurent Mazare. Fast issue triage,
  conservative PR merge cadence.
- Production users: several ML serving companies use it to wrap PyTorch
  models in Rust HTTP/gRPC services.

## Code size

`tokei` on `tch-rs`:

- `src/wrappers/tensor_generated.rs`: ~10k LoC (auto-generated)
- `src/wrappers/tensor.rs`: ~3k LoC
- `torch-sys`: ~5k LoC (FFI declarations)
- Total: ~25k LoC — small because libtorch does the actual work.

Compare:
- burn: ~100k LoC for what it owns.
- candle: ~80k LoC.
- rustorch v1.0 target: ~80-100k LoC.
- libtorch (the C++ side of PyTorch that `tch-rs` wraps): **>2M LoC**.

`tch-rs` has the smallest codebase by a large margin precisely because
it does not own the implementation.

## Limitations vs rustorch goals

1. **Requires libtorch** — disqualifies WASM, no_std, mobile,
   audit-conscious environments.
2. **No control over the autograd model** — bound to libtorch's tape;
   no opportunity to experiment with a faithful tape design or
   re-trace alternative.
3. **No control over kernel scheduling** — opaque dispatcher,
   profiling tied to libtorch's hooks.
4. **Glibc / libstdc++ ABI exposure** — runtime crashes possible if
   environment changes.
5. **Tracks libtorch release cadence** — feature lag of 1-2 weeks
   typical, blocking on libtorch shipping new ops.

## Could we contribute?

`tch-rs` is feature-frozen by design — it auto-generates from
`Declarations.yaml`, so 99% of "missing op" issues are resolved by
bumping the libtorch version. The hand-written ergonomic layer accepts
PRs but the surface is small.

For rustorch, `tch-rs` is **not a contribution target**. The two
projects share zero implementation and have orthogonal missions.

## Verdict

`tch-rs` is **excluded from rustorch's competitive frame**. It is in a
different category — an FFI wrapper around a different runtime, not a
Rust port of PyTorch. Different mission, different constraints,
different users.

We mention it in this analysis because users sometimes group "Rust ML"
projects together; the practical reality is:

- `tch-rs` = "PyTorch from Rust, today, with libtorch" — wrapper.
- `burn` / `candle` / `rustorch` = "ML in Rust, native" — implementations.

These two groups are complementary, not competing.

For rustorch's go/pivot decision in `decision.md`, `tch-rs` is **not a
factor** — it does not address the WASM, no_std, or pure-Rust
requirements that motivate rustorch's existence.
