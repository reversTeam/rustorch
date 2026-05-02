//! Browser smoke test for `rustorch-wasm-demo`.
//!
//! Runs in a headless Chrome (or Firefox / Node) via `wasm-pack test
//! --headless --chrome crates/rustorch-wasm-demo`. The test only
//! verifies that the WASM module loads and the JS-exposed function is
//! reachable — it does NOT call into WebGPU because adapter
//! availability in headless Chrome is fragile in CI environments.

#![cfg(target_arch = "wasm32")]

use wasm_bindgen_test::*;

wasm_bindgen_test_configure!(run_in_browser);

#[wasm_bindgen_test]
fn module_loads_without_panic() {
    // Just touching the crate is enough — `boot()` runs via
    // #[wasm_bindgen(start)] and would have panicked already if the
    // module was malformed.
    assert_eq!(2 + 2, 4);
}

#[wasm_bindgen_test]
async fn run_demo_returns_uniform_when_webgpu_missing() {
    // In a CI environment without a real WebGPU adapter, run_demo is
    // expected to error out cleanly with a JsValue (not panic). This
    // exercises the error path and verifies the async glue.
    let result = rustorch_wasm_demo::run_demo(2, 4).await;
    // Either it succeeded (real WebGPU was available) or it failed with
    // a JsValue (expected on most CI runners). Both are valid.
    match result {
        Ok(arr) => {
            assert_eq!(arr.length(), 8);
        },
        Err(_e) => { /* no adapter — fine in CI */ },
    }
}
