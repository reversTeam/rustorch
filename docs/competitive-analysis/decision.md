# Decision — go / pivot for rustorch

| Field | Value |
|-------|-------|
| Author | Théotime Rivière (`theotimeriviere@gmail.com`) |
| Date | 2026-05-01 |
| Status | **Decided — proceed with rustorch as a standalone project** |
| Inputs | [`burn.md`](./burn.md), [`candle.md`](./candle.md), [`tch-rs.md`](./tch-rs.md), [`comparative-table.md`](./comparative-table.md), RFCs 0001–0006 |

## TL;DR

**Build rustorch.** Do not pivot to contributing to burn or candle.

The combination of constraints below is **not jointly satisfied** by any
existing Rust ML project, and the architectural divergences are too
deep to bridge through PRs:

1. PyTorch-faithful runtime-shape `Tensor` API with per-tensor
   `requires_grad` (RFC-0001 Decision 1, RFC-0002).
2. Thread-local tape autograd with `create_graph: true` for
   higher-order grads and a flat tape exposable to debug tooling
   (RFC-0003).
3. WebGPU as the WASM-first GPU backend with a hard `<10 MB` bundle
   budget enforced in CI (RFC-0006).
4. First-party DDP + FSDP + ProcessGroup from day 1 (Phase 5 plans).
5. First-party Python bindings (PyO3) shipped on PyPI as wheels for
   manylinux + Apple Silicon + Windows (Phase 6 plan).

## How the alternatives score against these five constraints

| Constraint | rustorch | burn | candle | tch-rs |
|-----------|----------|------|--------|--------|
| 1. PyTorch-faithful runtime shapes + per-tensor `requires_grad` | by design | runtime shapes ✓, but const-D leaks into every signature ✗ | runtime shapes ✓, but `Var`/`Tensor` split breaks fidelity ✗ | full fidelity (delegates to libtorch) |
| 2. Thread-local tape with `create_graph` and debug-dumpable tape | by design | wrap-backend re-trace, no flat tape | inline `Op` graph, no flat tape | libtorch tape (opaque from Rust) |
| 3. WebGPU first-class + `<10 MB` CI budget | by design | WebGPU works, no budget | community WebGPU only, ~6 MB observed | no WASM |
| 4. First-party DDP + FSDP from day 1 | by design (Phase 5) | none in core | none | yes (libtorch) |
| 5. First-party PyO3 bindings + PyPI wheels | by design (Phase 6) | community only | community only | not the use case |

No row is fully green for any alternative. The closest is burn (2/5
partial), but its static-shape baseline cannot be retrofitted without
breaking every existing burn user.

## Why contributing instead is not the right call

For each of burn and candle, the analysis docs articulate concrete
contribution paths. The blockers in summary:

### burn

- The runtime-shape direction would require shipping a parallel
  `Tensor<B, K>` (no `D`) alongside the existing `Tensor<B, D, K>`,
  effectively forking the public API. Tracel AI's commercial users are
  invested in static shapes; this PR would not be merged.
- A tape-based autodiff alongside `Autodiff<B>` re-trace doubles the
  surface. The maintainers have not signaled appetite for this.
- WASM hardening (CI, bundle budget, demo) is a smaller, mergeable
  contribution — but it solves only constraint 3, leaves 1, 2, 4, 5
  untouched.

### candle

- The `Var` / `Tensor` split is foundational to candle's "zero-overhead
  inference" pitch; making `requires_grad` per-tensor would
  invalidate that property. Laurent has stated openness in past issues
  but the change is not in scope for the project.
- The inline-`Op`-on-tensor autograd is the project's signature design
  — replacing it with a tape would be rejected.
- Linalg gap-filling (qr/svd/cholesky/einsum) is mergeable but small.
- WebGPU first-party backend is mergeable in principle but the
  community fork already sits unmerged, suggesting prioritization
  blockers.

### tch-rs

- Categorical mismatch — `tch-rs` is an FFI wrapper, not a Rust port.
  Contributing to it does not produce a pure-Rust + WASM artifact.

## Differentiation — why rustorch is worth building (≥ 3 USPs)

These are the unique selling points that justify the cost of a
greenfield project rather than incremental contributions:

1. **PyTorch-faithful Rust API at the source level.**
   ```rust
   let x = Tensor::from_vec(vec![1.0, 2.0], [2], &dev)?.requires_grad_(true);
   let y = (&x * &x).sum()?;
   y.backward()?;
   let dy_dx = x.grad()?;     // 2*x = [2.0, 4.0]
   ```
   No `<const D: usize>` infections. No `Var` vs `Tensor` split. The
   ported PyTorch tutorials read identically save for the `?` operator
   and `Tensor::` constructor.

2. **WebGPU + WASM `<10 MB` budget as a CI gate, not an afterthought.**
   - First-party `wasm32-unknown-unknown` build matrix entry from
     P0.2 onwards.
   - Bundle-size budget enforced per release.
   - Browser demos (ResNet-18, GPT-2 small) shipped at v1.0.
   - This is the largest single differentiator vs candle, which is
     CPU+SIMD only in the browser today.

3. **First-party distributed (DDP + FSDP + ProcessGroup) and PyO3
   bindings.**
   - Phase 5 lands DDP and FSDP without delegating to libtorch.
   - Phase 6 lands `pip install rustorch` with NumPy zero-copy and
     DLPack PyTorch interop.
   - Closes the two largest gaps in the Rust ML training story.

4. **(Bonus) Tape exposable to tooling.**
   The flat thread-local tape is a tooling primitive — chrome-trace
   export, anomaly mode with stack-on-NaN, profiler hooks attached
   to op events. burn's wrap-backend model and candle's inline-Op
   model both lose this affordance.

## Risks and mitigations

| risk | mitigation |
|------|------------|
| Architecture diverges from PyTorch in subtle ways under user pressure | RFCs land before code (Phase 0); each RFC has a "How does PyTorch do it?" section with code references |
| WASM bundle budget violated late in Phase 1 | Hard CI gate from P0.2; budget tracked weekly |
| Op coverage delays v1.0 | ops.yaml + codegen pipeline (RFC-0005) makes adding ops mechanical |
| Performance gap vs libtorch / cuBLAS | Backend trait makes the CUDA path delegate to cuBLAS/cuDNN via `cudarc`; gap should be 5-15% on canonical models |
| Maintainer bus-factor of one | Explicit roadmap visible to the community, RFC discussions on GitHub Discussions, contributor-friendly issue triage |
| burn / candle ship the differentiating features first | Continuous monitoring; if (say) burn shipped a runtime-shape track + DDP, re-evaluate at the next Phase boundary |

## What we adopt from prior art (non-pivot interop)

We are not pretending to invent the wheel. From the analysis docs, the
following decisions explicitly mirror burn or candle:

- `trait Backend` with associated `Storage` (RFC-0004 Decision A) —
  same direction as burn.
- `#[derive(Module)]` proc macro with `#[parameter]` / `#[buffer]`
  field annotations (P1.6) — same shape as burn's derive.
- Sub-trait grouping (`LinalgOps`, `ActivationOps`) once the unified
  trait crosses ~100 methods — burn's lesson.
- safetensors as the native checkpoint format (P1.8) — candle's lesson.
- Metal backend as a tier-1 target via wgpu's Metal HAL — candle's
  lesson on Apple Silicon importance.
- HuggingFace model zoo compatibility layer (separate plan) —
  candle's flywheel.

The RFCs we have already landed (0002–0006) cite both burn and candle
where the decisions are aligned, and where they diverge.

## Decision

**Continue rustorch as a greenfield project.**

The five-constraint envelope is unique to rustorch's mission. The
architectural divergences with burn (shape model, autograd) and candle
(`Var`/`Tensor` split, autograd, WebGPU absence) are too deep for
PR-shaped contributions. The differentiation is real and durable — at
least until either project decides to compete on the same envelope,
which the analysis suggests is unlikely in the next 12 months.

The roadmap stands as written in the existing 53 plans (P0 → P3).
Phase 0 closes with this decision; Phase 1 (P1.1 — Tensor core)
becomes the next active workstream.

## Sign-off

| Role | Name | Date | Notes |
|------|------|------|-------|
| Author | Théotime Rivière | 2026-05-01 | Initial decision to continue |
| Reviewer 1 | _to be filled_ | _pending_ | Stakeholder review pending |
| Reviewer 2 | _to be filled_ | _pending_ | Stakeholder review pending |

The decision is in effect from the date above. Reviewer feedback will
be captured in an addendum below if it changes any constraint or
mitigation; otherwise this document is final.

## Addendum — reviewer feedback

_(none yet — pending external review)_
