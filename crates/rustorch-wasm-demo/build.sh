#!/usr/bin/env bash
# Build the WASM bundle for the browser demo.
#
# Output: crates/rustorch-wasm-demo/pkg/  (auto-loaded by web/index.html)
#
# Requirements:
#   - rustup target wasm32-unknown-unknown installed
#   - wasm-pack:  cargo install wasm-pack
#   - On macOS, the rustup-managed cargo on PATH (not Homebrew's).
#     Easiest:  PATH="$HOME/.cargo/bin:$PATH" ./build.sh

set -euo pipefail

cd "$(dirname "$0")"
echo "==> Building rustorch-wasm-demo for the browser…"
echo "    (this pulls in rustorch-wgpu compiled to wasm32-unknown-unknown)"
echo

# `--target web` produces an ES-module-friendly bundle that index.html
# can `import` directly from `pkg/rustorch_wasm_demo.js`.
wasm-pack build . --target web --out-dir pkg

echo
echo "==> Done. Now serve and open in Chrome 120+:"
echo "      cd $(pwd)"
echo "      python3 -m http.server 8080"
echo "      open http://localhost:8080/web/"
