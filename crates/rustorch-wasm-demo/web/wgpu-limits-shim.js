// Compatibility shim — wgpu 22 (still pinned in this workspace) sends
// `maxInterStageShaderComponents` in the `requestDevice` descriptor.
// Chrome 130+ removed that limit from the WebGPU spec and now rejects
// any non-undefined value with `OperationError`.
//
// We patch BOTH `GPUAdapter.prototype.requestDevice` AND, defensively,
// `navigator.gpu.requestAdapter` so the patch survives even if the
// adapter handle was captured before page-script execution finished.
//
// Drop this file (and the <script> tags that load it) when wgpu is
// bumped to a version that no longer emits the field — currently
// wgpu 23+ (the field was removed in October 2024).
//
// MUST be loaded BEFORE any wasm module that uses WebGPU, because
// the wasm-bindgen generated `__wbg_requestDevice_*` thunk does
// `arg0.requestDevice(arg1)` — which goes through the prototype
// chain at *call* time, so as long as our prototype patch is in
// place by then we win.
//
// Set `window.__rustorchShimDebug = true` before this script loads
// (e.g. via a small inline script tag) to see verbose interception
// logs in the JS console.
(function () {
  const STRIPPED = ["maxInterStageShaderComponents"];
  const TAG = "[rustorch shim]";
  // Always log on load so the developer can confirm the shim ran.
  // Quiet logs (per-call interception) are gated behind the debug flag.
  const DEBUG = typeof window !== "undefined" && !!window.__rustorchShimDebug;
  const debug = (...a) => { if (DEBUG) console.log(TAG, ...a); };

  function sanitize(descriptor) {
    if (!descriptor || typeof descriptor !== "object") return descriptor;
    const limits = descriptor.requiredLimits;
    if (!limits || typeof limits !== "object") return descriptor;
    let stripped = 0;
    for (const k of STRIPPED) {
      if (k in limits) {
        try {
          delete limits[k];
          stripped++;
        } catch (e) {
          console.warn(TAG, "delete failed for", k, e);
        }
      }
    }
    if (stripped > 0) {
      console.info(TAG, "stripped", stripped, "legacy limit(s) from requiredLimits:",
        STRIPPED.join(", "));
    }
    return descriptor;
  }

  // Layer 1 — patch GPUAdapter.prototype.requestDevice. This is the
  // canonical interception point. wasm-bindgen's `arg0.requestDevice(arg1)`
  // goes through the prototype chain so this catches every adapter
  // instance, even ones obtained before the patch ran.
  if (typeof GPUAdapter !== "undefined" && GPUAdapter.prototype) {
    const proto = GPUAdapter.prototype;
    const orig = proto.requestDevice;
    if (orig && !orig.__rustorchPatched) {
      const patched = function (descriptor) {
        debug("intercepted requestDevice", descriptor);
        return orig.call(this, sanitize(descriptor));
      };
      patched.__rustorchPatched = true;
      proto.requestDevice = patched;
      console.info(TAG, "loaded — GPUAdapter.prototype.requestDevice patched");
    } else if (orig && orig.__rustorchPatched) {
      debug("GPUAdapter.prototype.requestDevice already patched — skipping");
    }
  } else {
    console.warn(TAG, "GPUAdapter not available — WebGPU disabled in this browser");
  }

  // Layer 2 — wrap navigator.gpu.requestAdapter so even any future
  // override of GPUAdapter.prototype is bypassed by re-patching the
  // returned adapter instance directly. Belt and braces.
  if (typeof navigator !== "undefined" && navigator.gpu) {
    const origReqAdapter = navigator.gpu.requestAdapter.bind(navigator.gpu);
    navigator.gpu.requestAdapter = function (...args) {
      return origReqAdapter(...args).then((adapter) => {
        if (!adapter) return adapter;
        // Replace the per-instance method as a final safety net.
        const inst = adapter.requestDevice.bind(adapter);
        adapter.requestDevice = function (descriptor) {
          debug("instance-level intercept");
          return inst(sanitize(descriptor));
        };
        return adapter;
      });
    };
    debug("navigator.gpu.requestAdapter wrapped");
  }
})();
