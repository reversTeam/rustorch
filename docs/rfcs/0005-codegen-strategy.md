---
id: 0005
title: Codegen strategy — `build.rs` from YAML
status: Draft
date: 2026-05-01
authors:
  - rustorch team <rfc@rustorch.dev>
supersedes: []
superseded-by: []
---

# RFC-0005: Codegen strategy — `build.rs` from YAML

## Summary

The ~300 ops shipped by v1.0 are described **once** in a YAML schema
(`schemas/ops.yaml`). A `build.rs` in `rustorch-codegen` parses the YAML
and emits Rust source for: function signatures, dispatch shims (the match
on `Storage` per RFC-0004), autograd `Node` impls (per RFC-0003), and
optional Python-binding stubs. Hand-written kernels live next to the
generated code and are referenced by name from the YAML.

## Motivation

We will have ~300 user-facing ops × 5–6 backends × 2 (forward + backward) ×
several language bindings = thousands of small files that must agree about
op names, signatures, and dtype tables. Without codegen, every op addition
is a multi-place edit and the inevitable drift becomes the #1 source of
silent regressions.

Concrete pain points:

- PyTorch has `aten/src/ATen/native/native_functions.yaml` (~1100 ops)
  consumed by ~10 codegen passes. Removing this file from PyTorch is
  unthinkable; they manage 100,000+ generated lines.
- burn currently maintains op signatures by hand; their `Backend` trait
  is now ~150 methods and their PRs frequently fix sync drift.
- candle hand-codes ops; no Python binding generation; adding a new dtype
  is a long PR.
- Without codegen we cannot generate Python bindings (Phase 6 P0.4) without
  a second source of truth.

## Constraints

- **`build.rs` only** — no proc macros for op definitions. Reason:
  proc macros are opaque to `find references` tooling and to grep, while
  generated source files in `OUT_DIR` are at least findable via
  `cargo expand` and the diagnostic emits filenames + line numbers.
- **Incremental rebuild on YAML change only.** `cargo build` should not
  re-codegen if the YAML hasn't changed (`println!("cargo:rerun-if-changed=...");`).
- **No runtime cost.** All generated code is monomorphized at compile.
- **Helpful error messages.** Malformed YAML must produce a diagnostic
  with the file:line of the offending entry, not a stack trace.
- **Schema evolution.** Adding fields to the YAML is a non-breaking change.
  Removing or renaming fields requires a `schema_version: 2` bump and a
  migration script.
- **Tooling-friendly.** Generated `.rs` files contain `#[doc = "auto-generated
  from ops.yaml line N — DO NOT EDIT BY HAND"]` headers so `cargo doc`
  surfaces the source.

## Alternatives Considered

### Alternative A — `ops.yaml` consumed by `build.rs` (chosen)

**Approach.**
- One YAML file: `schemas/ops.yaml` (authoritative).
- `rustorch-codegen` crate (already created in P0.2) houses parser + emitter.
- Each consuming crate (`rustorch-core`, `rustorch-cpu`, `rustorch-autograd`,
  `rustorch-nn`, future `rustorch-py`) has a small `build.rs` that calls into
  `rustorch-codegen` to emit just the artifacts it needs into `$OUT_DIR/`.
- The crate's `lib.rs` does `include!(concat!(env!("OUT_DIR"), "/ops_generated.rs"));`.

**Pros.**
- One source of truth.
- Adding an op = 1 YAML entry → propagates to ~6 places.
- `cargo expand` shows generated code if needed.
- IDE shows generated symbols via `rust-analyzer`'s OUT_DIR awareness.

**Cons.**
- `build.rs` codegen is initially opaque: a typo in YAML → cryptic compile
  error in generated code. Mitigated by the parser emitting
  `compile_error!("ops.yaml line 42: …")` proactively.

**Cost.** ~1000 LoC of parser + emitter; pays back after ~50 ops.

### Alternative B — `proc_macro!` DSL inside Rust source

**Approach.** `op!{ matmul(a: T, b: T) -> T { … } backward { … } }` macro.
Lives in a `rustorch-derive`-style crate and is invoked at op definition site.

**Pros.**
- No separate file format.
- IDE-friendly (proc macro + the host file).

**Cons.**
- Per-op compile cost on the macro expansion (slow when there are 300).
- Cannot share with non-Rust consumers (Python binding generation, ONNX
  export): they would have to re-parse Rust source.
- Macros can't easily emit cross-crate symbols (e.g., autograd Node impl
  from a CPU op definition site).

**Cost.** Comparable in implementation; loses cross-language benefit.

### Alternative C — Hand-write everything; no codegen

**Approach.** ops.yaml is replaced by hand-written op modules per backend.

**Pros.**
- Simplest mental model.

**Cons.**
- 300 × 5 sync points → guaranteed drift.
- Adding a Python binding doubles the surface manually.

**Cost.** Lowest upfront, highest long-term.

### Alternative D — TOML + `serde`-only parser

**Approach.** Same as A but TOML instead of YAML.

**Pros.**
- TOML's spec is shorter; serde_toml is leaner than serde_yaml.

**Cons.**
- TOML doesn't represent inheritance / templating well; we'd want to
  share dtype lists across ops.
- Multiline strings are awkward.

**Cost.** Marginal gain; YAML's familiarity (PyTorch precedent) wins.

## Decision

We adopt **Alternative A: `ops.yaml` + `build.rs`**.

```yaml
# schemas/ops.yaml (excerpt — full schema in RFC follow-up)

schema_version: 1
ops:
  matmul:
    summary: "Matrix multiplication of two 2-D tensors."
    signature:
      inputs:  [{ name: a, type: Tensor }, { name: b, type: Tensor }]
      output:  Tensor
    dtypes: [F32, F64, F16, Bf16]
    backward:
      formula: "grad_a = grad @ b^T ; grad_b = a^T @ grad"
      saves:   [a, b]
    backends:
      cpu:  { kernel: rustorch_cpu::kernels::matmul::matmul }
      wgpu: { kernel: rustorch_wgpu::kernels::matmul::matmul }
      cuda: { kernel: "rustorch_cuda::kernels::matmul::matmul",
              feature_gate: "cuda" }
    docs:
      pytorch_ref: "torch.matmul"
      see_also:    [bmm, addmm, dot]
      tags:        [linalg, blas]

  add:
    summary: "Elementwise addition with broadcasting."
    signature:
      inputs:  [{ name: a, type: Tensor }, { name: b, type: Tensor }]
      output:  Tensor
    dtypes: [F32, F64, F16, Bf16, I32, I64, U8, Bool]
    backward:
      formula: "grad_a = grad ; grad_b = grad (broadcast-aware)"
      saves:   []
    backends:
      cpu:  { kernel: rustorch_cpu::kernels::elementwise::add }
      wgpu: { kernel: rustorch_wgpu::kernels::elementwise::add }
    docs:
      pytorch_ref: "torch.add"
      tags:        [pointwise]
```

`rustorch-codegen` exposes:

```rust
// crates/rustorch-codegen/src/lib.rs

pub struct OpSchema { /* parsed YAML */ }

pub fn parse(yaml_path: &Path) -> Result<OpSchema, CodegenError>;

pub fn emit_dispatch_shims(schema: &OpSchema, out: &Path) -> Result<()>;
pub fn emit_autograd_nodes(schema: &OpSchema, out: &Path) -> Result<()>;
pub fn emit_op_signatures(schema: &OpSchema, out: &Path) -> Result<()>;
pub fn emit_python_stubs(schema: &OpSchema, out: &Path) -> Result<()>;  // Phase 6
```

Each consuming crate's `build.rs`:

```rust
// crates/rustorch-core/build.rs
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let schema = rustorch_codegen::parse("../../schemas/ops.yaml".into())?;
    let out = std::env::var("OUT_DIR")?;
    rustorch_codegen::emit_op_signatures(&schema, format!("{out}/op_signatures.rs").as_ref())?;
    rustorch_codegen::emit_dispatch_shims(&schema, format!("{out}/dispatch.rs").as_ref())?;
    println!("cargo:rerun-if-changed=../../schemas/ops.yaml");
    Ok(())
}
```

Hand-written kernels live in `rustorch-cpu/src/kernels/`, `rustorch-wgpu/src/kernels/`,
etc. The YAML's `backends.cpu.kernel` is a Rust path that resolves to a function
of the appropriate signature; the codegen-emitted dispatch shim calls it.

## Rationale

- **YAML over TOML/JSON** — readable for humans, allows multi-line strings
  for docstrings, has anchors for sharing dtype lists.
- **`build.rs` over proc macros** — generated `.rs` files in `OUT_DIR` are
  inspectable; macros are not. Faster compile per-op (parse YAML once vs
  expand macro per op).
- **`rustorch-codegen` as a crate** because both `build.rs` (compile-time)
  and possibly external tooling (`cargo run -p rustorch-codegen --bin
  doc-gen`) want to consume the schema.
- **Schema versioning** because we'll add fields. Bumping `schema_version`
  must be paired with `migrations/{N → N+1}.py` even though we're a Rust
  project — Python is fine for write-once migrations.
- **Diagnostic-quality errors** so a typo doesn't produce a 200-line rustc
  error from generated code; instead `compile_error!("ops.yaml:42: unknown
  dtype 'F8'")` at the YAML site.

## How does PyTorch do it?

- `aten/src/ATen/native/native_functions.yaml` — 1100 op entries; the
  authoritative source. We model `ops.yaml` after it.
- `tools/codegen/gen.py` — Python generator that emits headers, type stubs,
  Python bindings, dispatcher registrations. We rewrite the equivalent in
  Rust to keep the codegen toolchain in-language.
- `tools/codegen/api/native.py` — function signature emission. Direct
  inspiration for `emit_op_signatures`.

## How does burn / candle do it?

- **burn** uses Rust macros for op declaration (`burn-tensor` crate). Per-op
  compile cost; no cross-language sharing. We diverge.
- **candle** hand-codes ops. Adding a backend is a manual sweep. We diverge.
- **dfdx** uses const generics + macros; type-level shape information
  is part of the op declaration. We diverge per RFC-0001.

## Migration plan

The first authoritative `ops.yaml` lands together with P1.1 (Tensor core).
Until then, P0.x crates contain placeholder modules. Once codegen exists:

1. Move existing hand-written op signatures into the YAML.
2. Replace `pub fn matmul(...)` definitions with `include!`-d generated code.
3. Hand-written kernel functions stay; YAML's `backends.cpu.kernel` field
   simply points at them.
4. Acceptance: `cargo expand -p rustorch-core | grep "fn matmul"` shows
   one signature, generated; the body delegates to `rustorch-cpu::kernels::matmul`.

## Open questions

- [ ] **Backward formulas in YAML or in Rust?** PyTorch's
      `derivatives.yaml` keeps backward formulas as expressions parsed by
      the codegen. Pro: same source of truth. Con: building a small
      expression DSL in YAML is a project of its own. Current decision:
      backward formulas live in Rust as named functions referenced from
      YAML via `backward.kernel: rustorch_autograd::derivatives::matmul_backward`.
- [ ] Should the schema include **performance hints** (e.g., "this op
      benefits from CUDA Graphs", "this op has a fused-bias variant")?
      Pro: lets codegen emit specialized paths. Con: schema bloat.
      Current decision: no, for now.
- [ ] **One YAML or split?** PyTorch keeps signatures + dispatch + backward
      in one `native_functions.yaml`; recent versions split into smaller
      files. Current decision: one file until ~150 ops, then revisit.
- [ ] **Escaping in docstrings.** YAML's quirky escaping for `|` blocks
      and indentation can bite. Mitigation: codegen runs the content
      through `serde_yaml::Value` which normalizes.
- [ ] **Python codegen** — emit type stubs for `mypy` consumption (Phase 6
      PyO3 task). Schema has the data; emitter is a Phase-6 deliverable.
- [ ] **ONNX op mapping** — out of scope here; Phase 6 ONNX export plan
      will reference but not extend `ops.yaml`.
- [ ] Should the YAML schema enforce **alphabetical order** of op names?
      Diff-friendly + fewer merge conflicts. Current tilt: yes.

## References

- PyTorch `native_functions.yaml` — `aten/src/ATen/native/native_functions.yaml`
- PyTorch codegen — `tools/codegen/gen.py`
- serde_yaml docs — https://docs.rs/serde_yaml
- "Code generation in Rust: a case study from PyTorch" — internal talk notes (TBD)

## Decision matrix

| Aspect | A YAML+build.rs (chosen) | B proc_macro DSL | C hand-coded | D TOML+serde |
|--------|:------------------------:|:----------------:|:------------:|:------------:|
| Source of truth | one YAML | per-op site | distributed | one TOML |
| Cross-language sharing | yes | no | no | yes |
| Per-op compile cost | low | medium | low | low |
| IDE-friendliness | good | best | best | good |
| Schema evolution | versioned | ad-hoc | n/a | versioned |
| Decision | **chosen** | rejected | rejected | rejected |
