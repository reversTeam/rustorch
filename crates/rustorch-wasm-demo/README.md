# rustorch-wasm-demo

Browser demo gallery for [rustorch](../../). Compiles the rustorch
public API down to `wasm32-unknown-unknown` and exposes five interactive
demos covering every layer of the framework — from raw GPU kernels to
autograd-aware training.

## Demo gallery

Open `web/index.html` to explore:

| Demo | Path | Crates exercised |
|------|------|------------------|
| **GPU smoke test** — `add → relu → softmax` | WebGPU | `rustorch-wgpu` |
| **GPU transformer block** — Q/K/V matmul + attention + LayerNorm | WebGPU | `rustorch-wgpu` |
| **CPU transformer encoder** — autograd-aware Linear+MHA+RMSNorm+causal mask+positional encoding | CPU | `rustorch-nn`, `rustorch-autograd`, `rustorch-cpu` |
| **CPU CrossAttentionPool** — Perceiver / Q-Former / BLIP-2 style pooling | CPU | `rustorch-nn` |
| **Training step** — forward + backward + clip_grad_norm on a 2-layer MLP | CPU | `rustorch-nn`, `rustorch-autograd`, `rustorch-optim` |

The CPU demos work in any browser with WASM support; the GPU demos
require Chrome 120+ with WebGPU enabled.

There are also two stand-alone pages:

- `web/gpt.html` — single GPT-style block on raw WGPU
- `web/resnet.html` — single ResNet-style block on raw WGPU

## Build

```bash
# One-time: install wasm-pack if you don't have it.
cargo install wasm-pack

# Build the WASM bundle (drops into `pkg/` next to this README).
wasm-pack build crates/rustorch-wasm-demo --target web
```

> **macOS gotcha**: if you have both Homebrew Rust and `rustup`,
> prepend `~/.cargo/bin` to your `PATH` so the `rustup` shims take
> precedence — only the `rustup` toolchain has the standard library
> for `wasm32-unknown-unknown`.

## Serve

```bash
cd crates/rustorch-wasm-demo
python3 -m http.server 8080
# Open http://localhost:8080/web/ in Chrome 120+
```

## Native tests

The wasm entry-points wrap pure-Rust building blocks that are exercised
by **native** unit tests (no WebGPU required):

```bash
cargo test -p rustorch-wasm-demo
```

This validates the `TinyEncoder` shape, parameter count, and causal-mask
NaN-safety without leaving Cargo.

## Browser compatibility

| Browser           | WebGPU                | Tested? |
| ----------------- | --------------------- | ------- |
| Chrome 120+       | enabled by default    | ✅       |
| Edge 120+         | enabled by default    | ✅       |
| Firefox Nightly   | behind `dom.webgpu`   | ⚠       |
| Safari TP 18+     | behind menu flag      | ⚠       |

## CI

The workspace's `.github/workflows/wasm.yml` builds this crate on
every PR to catch wasm32 regressions early.

## Public WASM entry-points

| JS function | Rust source | Description |
|-------------|-------------|-------------|
| `run_demo(b, k)` | `lib.rs` | GPU smoke test (add/relu/softmax) |
| `run_attention_block(s, d)` | `lib.rs` | GPU transformer block with timing breakdown |
| `run_encoder_block(b, t, d, num_heads, causal)` | `transformer_block.rs` | CPU autograd-aware encoder block |
| `run_cross_attention_pool(b, t, d, heads, queries)` | `transformer_block.rs` | CPU CrossAttentionPool |
| `run_training_step(batch, in_dim, hidden, out_dim, max_norm)` | `transformer_block.rs` | CPU forward + backward + clip_grad_norm |
