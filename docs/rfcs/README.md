# rustorch RFCs

Architectural RFCs (Request For Comments) for **rustorch**.
Each RFC lives in this directory as `NNNN-slug.md`, with `NNNN` a
zero-padded 4-digit ID in `0001..0099`.

> **New RFC?** Copy [`template.md`](template.md), allocate the next
> available ID, fill in the front-matter, and open a pull request.

## Process

```text
draft   →   proposed   →   accepted   →   implemented
                                       ↘ superseded
                                       ↘ withdrawn
```

- **Draft** — author is still iterating; not yet up for review.
- **Proposed** — open for review on a PR.
- **Accepted** — review concluded, decision binding for downstream code.
- **Superseded** — replaced by a newer RFC; cross-link via front-matter.
- **Withdrawn** — author retracted before acceptance.

CI (`scripts/lint-rfc.sh`) validates:

- File name matches `NNNN-slug.md`
- YAML front-matter has all required keys (`id`, `title`, `status`, `date`, `authors`)
- All mandatory sections present (Summary, Motivation, Constraints,
  Alternatives, Decision, Rationale, *How does PyTorch / burn / candle do it*,
  Migration plan, Open questions, References, Decision matrix)
- `id` is unique across the directory
- `markdownlint --strict` passes

## Index

| ID | Title | Status | Authors |
|----|-------|:------:|---------|
| [0001](0001-architecture.md) | Architecture overview & big decisions | ![Draft](https://img.shields.io/badge/-Draft-yellow) | rustorch team |
| [0002](0002-tensor-design.md) | Tensor design — Storage, Layout, dtype, refcount | ![Draft](https://img.shields.io/badge/-Draft-yellow) | rustorch team |
| [0003](0003-autograd-model.md) | Autograd model — tape, Node, backward execution | ![Draft](https://img.shields.io/badge/-Draft-yellow) | rustorch team |
| [0004](0004-backend-trait.md) | Backend trait — interface, dispatch, monomorphization | ![Draft](https://img.shields.io/badge/-Draft-yellow) | rustorch team |
| [0005](0005-codegen-strategy.md) | Codegen strategy — `build.rs` from YAML | ![Draft](https://img.shields.io/badge/-Draft-yellow) | rustorch team |
| [0006](0006-wasm-strategy.md) | WASM strategy — `wasm32`, wgpu, threads | ![Draft](https://img.shields.io/badge/-Draft-yellow) | rustorch team |

> Status badges:
> `Draft` = author iterating, `Proposed` = under review,
> `Accepted` = binding, `Superseded` = replaced, `Withdrawn` = retracted.

## Conventions

- Decisions documented here are **binding for downstream code** once
  Accepted. Code that contradicts an Accepted RFC must either change the
  code, or open a superseding RFC first.
- RFCs **must cite** the equivalent design in PyTorch, burn, and candle
  whenever possible. Pinned commit links preferred over branch links.
- "How does PyTorch do it?" sections **link to file:line** in
  upstream pytorch source — use a clone at `pytorch/` (not in workspace
  members; see top-level `Cargo.toml`'s `exclude`).
- Each RFC must list ≥3 seriously-considered alternatives with
  pros/cons, and ≥5 open questions.
- Numerical, performance, or memory claims must be either cited or
  explicitly tagged as estimates.
