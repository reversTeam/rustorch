# rustorch

[![CI](https://github.com/rustorch/rustorch/actions/workflows/ci.yml/badge.svg)](https://github.com/rustorch/rustorch/actions/workflows/ci.yml)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](#license)
[![Crates.io](https://img.shields.io/crates/v/rustorch.svg)](https://crates.io/crates/rustorch)
[![Docs.rs](https://docs.rs/rustorch/badge.svg)](https://docs.rs/rustorch)
[![Codecov](https://codecov.io/gh/rustorch/rustorch/branch/main/graph/badge.svg)](https://codecov.io/gh/rustorch/rustorch)

> Pure Rust port of PyTorch — Tensor library + autograd + nn modules + optimizers
> + multi-backend (CPU, wgpu/WebGPU, CUDA optional) + WASM-first design.

`rustorch` aims for a **faithful PyTorch API** in safe Rust, with a single
codebase that runs natively on Linux/Mac/Windows, in the browser via WebGPU,
and on CUDA when the feature is enabled. The mental model is identical to
PyTorch; the syntax is Rust.

## Status

This project is **pre-1.0** and currently in Phase 0 (Architecture & RFC).
The user-facing API is documented at [v0.7.2 docs](https://rustorch.dev/docs)
and is being progressively implemented per the roadmap below.

| Phase | Title | ETA | Status |
|------:|-------|-----|--------|
| 0 | Architecture & RFC | 2026-05 | in progress |
| 1 | Fondations CPU + Autograd | 2026-08 | planned |
| 2 | Backend GPU via wgpu (WASM) | 2026-11 | planned |
| 3 | Optimisations avancées (Flash Attn, AMP, ckpt) | 2027-01 | planned |
| 4 | Backend CUDA natif (opt-in) | 2027-04 | planned |
| 5 | Distributed (DDP) | 2027-06 | planned |
| 2.5 | Ecosystem & Console backend | 2027-08 | planned |
| 2.7 | Console UI | 2027-09 | planned |
| 6 | Écosystème & Adoption (1.0) | 2027-10 | planned |

## Quickstart (forward-looking — implementation in progress)

Add `rustorch` to your `Cargo.toml`:

```toml
[dependencies]
rustorch = { version = "0.0.1", features = ["bf16", "safetensors"] }
```

Then a small training loop will look like:

```rust
use rustorch::prelude::*;

fn main() -> rustorch::Result<()> {
    let dev = Device::cuda_if_available()?;
    let net = Sequential::new()
        .add(Linear::new(784, 256).build(&dev)?)
        .add(ReLU)
        .add(Linear::new(256, 10).build(&dev)?);
    let mut opt = AdamW::new(net.parameters(), 1e-3)?.weight_decay(0.01);

    let train = Mnist::train(&dev)?;
    let loader = DataLoader::batch(64).workers(4).pin_memory(true).build(train)?;

    for epoch in 0..10 {
        for batch in &loader {
            let logits = net.forward(&batch.x)?;
            let loss = cross_entropy(&logits, &batch.y)?;
            opt.backward(&loss)?;
        }
    }

    net.save("mnist_mlp.safetensors")?;
    Ok(())
}
```

## Workspace layout

The repository is a Cargo workspace of 10 crates:

```
crates/
├── rustorch/          # umbrella re-export crate (the public API)
├── rustorch-core/     # Tensor, Storage, Layout, dtypes
├── rustorch-cpu/      # CPU backend (rayon + SIMD)
├── rustorch-autograd/ # tape-based reverse-mode autodiff
├── rustorch-nn/       # Module + nn modules (Linear, Conv, MultiheadAttention, ...)
├── rustorch-optim/    # Optimizer + LR schedulers (SGD, Adam, AdamW, Lion, ...)
├── rustorch-derive/   # proc macros (#[derive(Module)])
├── rustorch-data/     # Dataset + DataLoader + Sampler
├── rustorch-serde/    # safetensors reader/writer + state_dict
└── rustorch-codegen/  # build.rs codegen helpers (ops.yaml -> Rust)
```

## Building

Requires Rust **1.78+** (MSRV) — pinned via `rust-toolchain.toml`.

```bash
# Native build + test
cargo build --workspace
cargo test --workspace

# WASM target
cargo build --target wasm32-unknown-unknown -p rustorch-core

# No-default-features (minimal)
cargo build --workspace --no-default-features
```

### macOS gotcha

If you have **both Homebrew Rust and rustup**, Homebrew's cargo is usually
first in `$PATH` and shadows rustup. The wasm32 target stdlib lives only in
rustup, so a Homebrew cargo invocation against `--target wasm32-unknown-unknown`
will fail with `can't find crate for 'core'`.

Fix: prepend `~/.cargo/bin` to your `PATH` so rustup's shims win, or
explicitly invoke `~/.cargo/bin/cargo`.

## Project documentation

- [`docs/rfcs/`](docs/rfcs/) — architectural RFCs (Phase 0 deliverable)
- [`CONTRIBUTING.md`](CONTRIBUTING.md) — contribution guide, commit style
- [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md) — Contributor Covenant 2.1

## License

Dual licensed under either:

- [Apache License, Version 2.0](LICENSE-APACHE) (or http://www.apache.org/licenses/LICENSE-2.0)
- [MIT license](LICENSE-MIT) (or http://opensource.org/licenses/MIT)

at your option, matching the Rust ecosystem standard.

## Contribution

Unless you explicitly state otherwise, any contribution intentionally
submitted for inclusion in the work by you, as defined in the Apache-2.0
license, shall be dual licensed as above, without any additional terms or
conditions. See [`CONTRIBUTING.md`](CONTRIBUTING.md) for details.
