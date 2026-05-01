---
id: NNNN
title: Short title in sentence case
status: Draft  # Draft | Proposed | Accepted | Superseded | Withdrawn
date: YYYY-MM-DD
authors:
  - Author Name <email@example.com>
supersedes: []
superseded-by: []
---

# RFC-NNNN: Short title in sentence case

## Summary

One paragraph (3-5 sentences) explaining what this RFC proposes and why
it matters. A reader who only reads this section should know the gist.

## Motivation

What problem are we solving? What goes wrong if we don't do this? What
opportunity are we capturing? Cite at least one concrete pain point that
motivated the RFC, and prefer named real cases over hypotheticals.

## Constraints

The boundary conditions every alternative must satisfy. Includes:

- **Lines of code** — order of magnitude budget
- **Performance** — wall-clock or memory targets when relevant
- **Ergonomics** — what user code must look like
- **Compatibility** — Rust MSRV, target triples, downstream API stability
- **Build time** — how much can we afford?
- **WASM / `no_std`** — what subset must compile to wasm32, what to no_std?

## Alternatives Considered

For each alternative (minimum **three**), include:

### Alternative N: Name

**Approach.** One paragraph describing the design.

**Pros.**
- Bullet 1
- Bullet 2

**Cons.**
- Bullet 1
- Bullet 2

**Cost.** Rough LOC / time-to-implement estimate.

## Decision

Which alternative we picked. State it in one sentence so a skim-reader
can find it instantly. Then expand on what this concretely means in code.

## Rationale

Why we chose Decision over the alternatives. Cite the constraints that
discriminated, the trade-offs we accepted, and any precedent.

## How does PyTorch do it?

Reference the relevant PyTorch source files at line level. Example:

> `aten/src/ATen/core/Tensor.h:42-58` — defines `c10::TensorImpl` with
> shape, strides, dtype, storage. The shape is runtime-opaque, mirroring
> our Decision in this RFC.

## How does burn / candle do it?

Side-by-side notes on how the closest Rust competitors handle the same
problem. Useful both as sanity check and as a list of trade-offs we are
implicitly accepting or rejecting. Provide URL or commit-pinned link.

## Migration plan

If this RFC supersedes prior internal designs, describe the migration path:
the deprecation window, automated rewrites, backward-compatibility shim.

For Phase-0 RFCs, "no prior design — clean slate" is acceptable.

## Open questions

A list of ≥5 items still being debated, marked clearly so reviewers can
see what is **not** locked in. Convention:

- [ ] Question phrased as a yes/no or branching choice.
- [ ] Question describing a research / measurement still owed.

## References

- Spec or paper URL
- Bug tracker link
- Slack / Discord / GitHub Discussion thread

## Decision matrix

| Aspect | Alt 1 | Alt 2 | Alt 3 (chosen) |
|--------|-------|-------|----------------|
| LOC | low | medium | low |
| Perf | medium | high | medium |
| WASM | yes | partial | yes |
| MSRV impact | none | nightly | none |
| Migration cost | n/a | high | low |

(Replace columns and rows as appropriate. The chosen column should be
visually marked, e.g. bold heading or a "(chosen)" suffix.)

---

<!-- markdownlint-disable -->
**Lint policy.** This file is checked with `markdownlint --strict
--config docs/rfcs/.markdownlint.json`. Front-matter is parsed as YAML
by the lint script in `scripts/lint-rfc.sh` (see RFC template task).
