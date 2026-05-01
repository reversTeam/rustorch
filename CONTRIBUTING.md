# Contributing to rustorch

Thank you for taking the time to contribute! This document covers the
development workflow, commit conventions, and PR checklist.

## Getting started

```bash
git clone https://github.com/rustorch/rustorch
cd rustorch
cargo build --workspace
cargo test --workspace
```

The workspace requires Rust **1.78+** (MSRV) — pinned via `rust-toolchain.toml`.
If you don't have rustup, install it from <https://rustup.rs>.

## Commit message conventions

We follow [Conventional Commits 1.0.0](https://www.conventionalcommits.org/).
The commit-msg git hook (installed via `scripts/install-hooks.sh`) validates
this automatically.

### Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Types

- `feat`: new feature
- `fix`: bug fix
- `refactor`: code restructuring without behavior change
- `perf`: performance improvement
- `test`: tests only
- `docs`: documentation only
- `build`: build system, CI, dependencies
- `chore`: misc maintenance
- `revert`: revert previous commit

### Scope examples

`tensor`, `autograd`, `nn`, `optim`, `cpu`, `wgpu`, `cuda`, `data`, `serde`,
`derive`, `cli`, `console`, `serve`, `docs`, `rfc`.

### Examples

```
feat(autograd): implement create_graph for higher-order gradients

Closes #42.
```

```
fix(cpu): handle empty matmul shape without panicking

The im2col path was unwrapping a 0-sized batch and triggering an
out-of-bounds index. Now returns an empty tensor of the expected shape.

Closes #87.
```

## Pull request checklist

Before opening a PR, please ensure:

- [ ] Code compiles: `cargo build --workspace`
- [ ] Tests pass: `cargo test --workspace`
- [ ] Formatted: `cargo fmt --all`
- [ ] No clippy warnings: `cargo clippy --workspace --all-targets -- -D warnings`
- [ ] Coverage is not regressed (CI computes the delta automatically)
- [ ] Docs updated for public API changes (`cargo doc --workspace --no-deps`)
- [ ] CHANGELOG.md entry under **Unreleased** if user-facing
- [ ] PR description references the relevant plan/task in the project orchestrator

## Style

- **Result-based error handling** — no panics in user-facing code; use `thiserror` for error enums.
- **No unwrap/expect in production paths** — only in tests, examples, build scripts.
- **Doc comments on every public item** — `#![warn(missing_docs)]` is enforced in `rustorch-core`.
- **Property tests** for invariants when reasonable (`proptest`).
- **Numerical parity tests** vs PyTorch where applicable, tolerance documented per op.

## RFCs

Architectural changes go through the RFC process. See `docs/rfcs/` for the
template and existing RFCs.

## Reporting issues

- **Bugs / questions** — open a GitHub issue.
- **Security disclosures** — email security@rustorch.dev (do **not** open a public issue).

## License

By contributing, you agree your contributions will be dual-licensed under
MIT OR Apache-2.0, matching the project license.
