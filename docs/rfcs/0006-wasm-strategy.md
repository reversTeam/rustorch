---
id: 0006
title: WASM strategy — `wasm32`, wgpu, threads
status: Draft
date: 2026-05-01
authors:
  - rustorch team <rfc@rustorch.dev>
supersedes: []
superseded-by: []
---

# RFC-0006: WASM strategy — `wasm32`, wgpu, threads

## Summary

rustorch targets `wasm32-unknown-unknown` as a **first-class CI target**.
The `rustorch-core`, `rustorch-autograd`, `rustorch-nn`, and `rustorch-optim`
crates compile cleanly with default features on wasm32. GPU compute on the
web uses **wgpu/WebGPU** (Phase 2). Threading is **off by default** in WASM
and behind a feature flag (`wasm-threads`) when on, requiring
`SharedArrayBuffer` and the right CORS headers. RNG comes from the Web Crypto
API. File I/O for safetensors is replaced by `fetch()` + `ArrayBuffer` in
WASM. The bundle size budget is **<10 MB** after `wasm-opt -Oz` for a
public demo (Phase 2 ResNet-18 / GPT-2 small).

## Motivation

WASM is rustorch's strongest differentiator vs burn and candle: a faithful
PyTorch-style API that runs **inside a browser** unlocks demos, education,
edge inference, Cloudflare Worker / WASI shoreline use cases, and offline
mobile apps via WebView wrappers. Treating WASM as an afterthought
guarantees:

- A core crate that accidentally uses `std::process`, `std::fs::OpenOptions`,
  or other unsupported APIs.
- A "build broken on wasm32" state that is silently shipped because nobody
  tests it.
- Cleanup costs measured in weeks once code starts assuming POSIX.

By contrast, treating WASM as a first-class target from Phase 0 forces a
clean separation between **platform-portable** and **platform-bound** code
that pays back on every other target too (Linux servers without `mmap`,
embedded ARM without `std::time::Instant`).

## Constraints

- **Targets supported.**
  - `wasm32-unknown-unknown` (browser, Cloudflare Workers): primary.
  - `wasm32-wasi` (server-side WASI): secondary, supported but not in CI matrix.
  - `wasm32-unknown-emscripten`: explicitly out of scope.
- **Default-features.** `cargo build --target wasm32-unknown-unknown -p rustorch-core`
  must succeed on a clean clone with default features only.
- **Bundle size.** Public demo (Phase 2): <10 MB after `wasm-opt -Oz`.
  Loading time <3s on 4G.
- **Threading.** Default off. Opt-in via `feature = "wasm-threads"` which
  pulls `wasm-bindgen-rayon` and requires `Cross-Origin-Opener-Policy:
  same-origin` + `Cross-Origin-Embedder-Policy: require-corp` headers.
- **GPU compute.** wgpu (Phase 2). Falls back to CPU if WebGPU unavailable.
- **Random.** Use `getrandom` crate which delegates to `crypto.getRandomValues`
  in browsers and `wasi_random_get` on WASI.
- **Time.** `std::time::Instant` works on `wasm32-wasi` but not `unknown-unknown`.
  We use `web-time` polyfill or a custom `Time` trait.
- **File I/O.** For `wasm32-unknown-unknown`, no `std::fs`. Safetensors
  consumes `&[u8]` slices fetched via JS `fetch()` (caller's responsibility).
- **Panic.** `panic = "abort"` in WASM build profile (default for wasm32);
  user code expecting unwinding does not work on the web.

## Alternatives Considered

### Alternative A — WASM as first-class CI target (chosen)

**Approach.** Phase 0 ships a `wasm.yml` workflow that builds
`rustorch-core` for `wasm32-unknown-unknown` on every push. As more crates
mature, they're added. The split-by-`#[cfg]` discipline is enforced by a
green CI signal.

**Pros.**
- Catches regressions immediately.
- Forces clean platform separation.
- Cheap incremental cost (~1 CI minute per PR after caching).

**Cons.**
- Some convenience APIs (rayon::par_iter, mmap) need a non-wasm fallback.
- One more thing to think about per op author.

**Cost.** +1 CI job; +5–10% LoC for `#[cfg]` separation.

### Alternative B — Native-first; WASM as a Phase 2 surprise

**Approach.** Build natively first; introduce WASM only when wgpu is ready.

**Pros.**
- Slightly fewer `cfg`s in early code.

**Cons.**
- Inevitable rewrites when WASM-incompatible APIs (e.g.,
  `std::process::Command` in tests, `std::time::Instant`) leak into core
  crates.
- Large surface to fix at Phase 2 — slips the calendar.

**Cost.** Cheap upfront; expensive cleanup.

### Alternative C — Skip WASM entirely

**Approach.** Native targets only; ship a future `rustorch-wasm` adapter
crate later if demand exists.

**Pros.**
- Simplest implementation.

**Cons.**
- Forfeits the strongest differentiator.
- Marketing position weakens significantly.

**Cost.** N/A — rejected by Phase-0 product position.

### Alternative D — `wasm32-emscripten` as primary

**Approach.** Use the Emscripten toolchain so we get a full POSIX-like
environment in the browser (mmap, threads, file I/O).

**Pros.**
- Lifts most platform-bound code "for free".

**Cons.**
- Bundle 2–3× larger than `wasm32-unknown-unknown`.
- Requires `emcc` toolchain in CI.
- Diverges from the Rust mainstream.

**Cost.** Larger bundles, more complex CI.

## Decision

We adopt **Alternative A: WASM as first-class CI target**.

Concrete commitments locked here:

1. **CI matrix.** `wasm32-unknown-unknown` is in CI from day 0.
   `.github/workflows/wasm.yml` (already shipped in P0.2 commit `23cca27a`)
   builds `rustorch-core`, `rustorch-autograd`, and `rustorch-nn` on every push.

2. **`#[cfg(target_arch = "wasm32")]` discipline.** Native-only deps
   (rayon, mmap2, etc.) gated at the `Cargo.toml` level via
   `[target.'cfg(not(target_arch = "wasm32"))'.dependencies]`. Already
   applied in P0.2 `rustorch-cpu` and `rustorch-data`.

3. **Crate matrix.**

| Crate | wasm32 default | Notes |
|-------|:--------------:|-------|
| rustorch-core | yes | strict portable |
| rustorch-cpu | yes (sequential fallback) | rayon-gated to non-wasm |
| rustorch-autograd | yes | thread-local single-thread on wasm |
| rustorch-nn | yes | core ops only |
| rustorch-optim | yes | core optimizers only |
| rustorch-derive | yes | proc-macros run on host, output is target-agnostic |
| rustorch-data | yes (no workers) | rayon-gated; user provides input bytes via fetch() |
| rustorch-serde | yes | input is `&[u8]`, no `std::fs` |
| rustorch-codegen | host-only | runs in `build.rs`, never targets wasm |
| rustorch | yes | umbrella; mirrors above |

4. **Threading**: opt-in via `feature = "wasm-threads"`. Default disabled.
   Documented as requiring COOP/COEP headers.

5. **GPU**: `rustorch-wgpu` (Phase 2 plan) targets WebGPU as one of the
   wgpu backends. WGSL shaders work identically on native and wasm32.
   Falls back to `CpuBackend` if `navigator.gpu` is undefined.

6. **RNG**: depend on `getrandom = { version = "0.2", features = ["js"] }`
   transitively via `rand`. Web Crypto API delivers entropy on wasm32-unknown.

7. **Time**: where wall-clock time matters (P2.4 sampling profiler), wrap
   in a `Time` trait with `std::time::Instant` impl on native and a
   `web_sys::Performance::now()` impl on wasm32.

8. **File I/O**: `rustorch-serde` reads safetensors from `&[u8]`. The
   user's WASM bootstrap code calls `fetch()` and passes the bytes in.
   No `std::fs` in any wasm-compiled crate.

9. **Bundle size**: P0.2 CI runs a `wasm-opt -Oz`-style build; future
   work tracks a per-PR size delta against a baseline. Public demo
   target <10MB (Phase 2 deliverable).

10. **Panic strategy**: `panic = "abort"` in the release profile for
    wasm32 (already the default for `wasm32-unknown-unknown`).

## Rationale

- **WebGPU > WebGL2** because WebGPU exposes compute shaders directly and
  matches the abstraction wgpu already gives us natively. WebGL2 doesn't
  have a clean compute story.
- **`wasm32-unknown-unknown` over `wasm32-wasi`** because the browser is
  the most demand-rich audience. WASI is a future stretch.
- **Threading off by default** because COOP/COEP headers break some
  sites (they disable third-party iframes), and most ML workloads in the
  browser are inference where threads aren't strictly required.
- **`web-time` for clocks** because `std::time::Instant` literally panics
  on `wasm32-unknown-unknown`. Catching this in P0.2 saves us from a
  Phase 2 surprise.
- **No `std::fs` in core crates** because the burden of "does this fail
  on wasm?" is removed from kernel authors entirely.

## How does PyTorch do it?

PyTorch does **not** target WASM. Closest is `executorch` for mobile/edge
deployment. We deliberately step beyond PyTorch here. There are no
file:line references on the PyTorch side; the prior art is in:

- `wasm-bindgen` ecosystem — Alex Crichton et al.
- ONNX Runtime Web — Microsoft's WASM/WebGPU runtime.
- TensorFlow.js — Google's browser ML runtime (uses WebGL/WebGPU).
- Burn / candle examples — both have small WASM demos.

## How does burn / candle do it?

- **burn** has a `wasm` example and supports the target, but does not
  have a CI green-bar discipline that we adopt. Their `wasm-burn-image-classifier`
  example is ~6 MB compiled.
- **candle** has `candle-wasm-examples` with several models in-browser
  (SAM, YOLO, Whisper, BERT). Bundle sizes range 5–25 MB. Close prior art.
- **TensorFlow.js** is the gold-standard reference for browser ML
  ergonomics; we explicitly aim to match its load-time and exceed its
  ergonomics.

## Migration plan

Foundational. The discipline (CI green on wasm, `cfg`-gated deps) is
applied from Phase 0 forward. No prior native-only state to migrate from.

If a future RFC declares WASM legacy or restricts it to a subset, that
RFC must:
1. Document which crates remain WASM-portable.
2. Provide a `--features wasm-legacy` shim until the next major version.

## Open questions

- [ ] **`SharedArrayBuffer`-based threading**: do we expose
      `wasm-bindgen-rayon` parity such that the same `rayon::par_iter`
      code works on wasm with threads enabled, or do we explicitly
      sequentialize and require manual conversion? Current tilt: parity
      via `wasm-bindgen-rayon`, off by default.
- [ ] **`wasm32-wasi` Phase 2 inclusion?** WASI gives us `mmap` for
      safetensors and a real filesystem. Worth considering as a
      secondary CI target when WASI Preview 2 stabilizes.
- [ ] **bf16 in WGSL**: WGSL doesn't have native bf16. We emulate via
      f16 (which WGSL has via the `f16` extension) with a documented
      precision degradation for backends that lack bf16 support, or do
      we restrict bf16 to native-only? Current tilt: f16 emulation +
      doc note.
- [ ] **CPU SIMD on wasm32**: simd128 is available with the `+simd128`
      target feature (already enabled in `.cargo/config.toml`). Should
      we raise this to a hard requirement?
- [ ] **Bundle size budget per Phase**: Phase 0 has no demo, so size is
      undefined; Phase 2 demo <10MB; Phase 6 marketing target <5MB.
      Is the 10MB budget realistic with f32 weights? Initial measurements
      with ResNet-18 weights ≈ 45MB f32 / 11MB int8 — quantization
      becomes mandatory for the demo.
- [ ] **Fallback to CPU when WebGPU unavailable**: do we silently
      degrade or surface an error to the user? Current decision:
      return `Err(Backend::Unsupported("WebGPU"))` and let the caller
      decide.
- [ ] **Panic policy in WASM**: `panic = "abort"` is default; do we
      provide a `panic_hook` that logs to console.error for usability?

## References

- `wasm-bindgen` — https://rustwasm.github.io/wasm-bindgen/
- `wgpu` WebGPU support — https://wgpu.rs/
- `web-time` crate — https://docs.rs/web-time
- `getrandom` — https://docs.rs/getrandom
- TensorFlow.js bundle analysis — https://www.tensorflow.org/js/guide/size
- candle WASM examples — https://github.com/huggingface/candle/tree/main/candle-wasm-examples
- WebGPU spec — https://www.w3.org/TR/webgpu/

## Decision matrix

| Aspect | A WASM-first (chosen) | B native-first | C skip WASM | D emscripten |
|--------|:---------------------:|:--------------:|:-----------:|:------------:|
| Bundle size | small (~5-10MB) | n/a until later | n/a | 2-3× larger |
| CI cost | +1 job | +1 job later | none | high (emcc) |
| Cleanup risk | low (continuous) | high (Phase 2) | n/a | low |
| Browser ergonomics | best | n/a | n/a | best |
| Marketing differentiation | strong | delayed | none | strong |
| Decision | **chosen** | rejected | rejected | rejected |
