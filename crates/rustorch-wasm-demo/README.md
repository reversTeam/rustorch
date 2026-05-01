# rustorch-wasm-demo

Browser WebGPU demo for [rustorch](../../). Compiles the `rustorch-wgpu`
backend down to `wasm32-unknown-unknown` and exposes a `run_demo()` JS
function that performs `add → relu → softmax` on the browser's WebGPU
adapter.

## Build

```bash
# One-time: install wasm-pack if you don't have it.
cargo install wasm-pack

# Build the WASM bundle (drops into `pkg/` next to this README).
wasm-pack build crates/rustorch-wasm-demo --target web
```

## Run

```bash
# Serve the static `web/` directory + the freshly built `pkg/`.
# Any static-file server works — `python3 -m http.server` is the
# minimum.
cd crates/rustorch-wasm-demo
python3 -m http.server 8080

# Then open Chrome 120+ at http://localhost:8080/web/ and click
# "Run forward pass". You should see 16 softmax probabilities
# summed to 1 per row.
```

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
