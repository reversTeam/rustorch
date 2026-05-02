//! Single ResNet basic-block forward pass on the GPU.
//!
//! ```text
//!     y = relu(conv2(relu(conv1(x))) + x)        # if c_in == c_out
//!     y = relu(conv2(relu(conv1(x))) + skip(x))  # 1×1 conv on skip otherwise
//! ```
//!
//! v1 ships **untrained random weights** so the demo page can prove
//! the full Conv2d → ReLU → Conv2d → residual chain runs in the
//! browser without depending on a multi-MB ImageNet checkpoint. The
//! follow-up wires this module to [`crate::model_loader`] so users
//! can drop in real ResNet-18 weights from HuggingFace.
//!
//! On the JS side the page hands a `[1, C_in, H, W]` F32 array; the
//! Rust function dispatches the block and returns the output array
//! plus its shape.

use rustorch_core::tensor::tensor_impl::Tensor;
use rustorch_wgpu::{
    conv2d_forward, dispatch_binary, dispatch_unary, to_cpu, to_gpu, transpose_weight, Conv2dCfg,
    WgpuBackend, WgpuError, WgpuStorage,
};

/// Generate `n` deterministic f32 values in `[-0.05, 0.05]` so the
/// random "weights" stay well-behaved through Conv → ReLU.
fn det_weights(n: usize, seed: u64) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let mut x = seed
                .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                .wrapping_add(i as u64);
            x ^= x >> 33;
            x = x.wrapping_mul(0xff51_afd7_ed55_8ccd);
            x ^= x >> 33;
            let unit = ((x >> 33) as u32 as f32) / (u32::MAX as f32);
            (unit - 0.5) * 0.1
        })
        .collect()
}

/// Run a basic ResNet block on `[1, c_in, h, w]` F32 input. Output
/// shape is `[1, c_out, h, w]` (same spatial dims because pad=1 on
/// the 3×3 convs).
pub fn run_resnet_block(
    backend: &WgpuBackend,
    input_chw: &[f32],
    c_in: usize,
    c_out: usize,
    h: usize,
    w: usize,
) -> Result<Vec<f32>, WgpuError> {
    if input_chw.len() != c_in * h * w {
        return Err(WgpuError::ShapeMismatch(format!(
            "resnet_block: input len {} != c_in*h*w {}",
            input_chw.len(),
            c_in * h * w
        )));
    }
    let kh = 3;
    let kw = 3;
    let cfg = Conv2dCfg {
        kh,
        kw,
        sh: 1,
        sw: 1,
        ph: 1,
        pw: 1,
        ..Default::default()
    };

    // Upload input.
    let tx = Tensor::from_vec([1, c_in, h, w], input_chw.to_vec())
        .map_err(|e| WgpuError::ShapeMismatch(format!("{e}")))?;
    let gx = to_gpu(backend, &tx)?;

    // First conv: c_in → c_out.
    let w1 = det_weights(c_out * c_in * kh * kw, 0xA1);
    let w1_t = transpose_weight(&w1, c_out, c_in, kh, kw);
    let tw1 = Tensor::from_vec([c_in * kh * kw, c_out], w1_t)
        .map_err(|e| WgpuError::ShapeMismatch(format!("{e}")))?;
    let gw1 = to_gpu(backend, &tw1)?;
    let (g_conv1, _, _) = conv2d_forward(backend, &gx, &gw1, 1, c_in, h, w, c_out, cfg)?;
    let g_relu1 = dispatch_unary(backend, "relu", &g_conv1)?;

    // Second conv: c_out → c_out.
    let w2 = det_weights(c_out * c_out * kh * kw, 0xB2);
    let w2_t = transpose_weight(&w2, c_out, c_out, kh, kw);
    let tw2 = Tensor::from_vec([c_out * kh * kw, c_out], w2_t)
        .map_err(|e| WgpuError::ShapeMismatch(format!("{e}")))?;
    let gw2 = to_gpu(backend, &tw2)?;
    let (g_conv2, _, _) = conv2d_forward(backend, &g_relu1, &gw2, 1, c_out, h, w, c_out, cfg)?;

    // Residual: if c_in == c_out, add the input directly; otherwise
    // run a 1×1 conv on the skip path to adapt the channel count.
    let g_skip = if c_in == c_out {
        gx
    } else {
        let cfg_1x1 = Conv2dCfg {
            kh: 1,
            kw: 1,
            sh: 1,
            sw: 1,
            ph: 0,
            pw: 0,
            ..Default::default()
        };
        let ws = det_weights(c_out * c_in, 0xC3);
        let ws_t = transpose_weight(&ws, c_out, c_in, 1, 1);
        let tws = Tensor::from_vec([c_in, c_out], ws_t)
            .map_err(|e| WgpuError::ShapeMismatch(format!("{e}")))?;
        let gws = to_gpu(backend, &tws)?;
        conv2d_forward(backend, &gx, &gws, 1, c_in, h, w, c_out, cfg_1x1)?.0
    };

    // Add then activate.
    let g_sum = dispatch_binary(backend, "add", &g_conv2, &g_skip)?;
    let g_out = dispatch_unary(backend, "relu", &g_sum)?;

    let out_t: WgpuStorage = g_out;
    let cpu = to_cpu(backend, &out_t, vec![1, c_out, h, w])?;
    Ok(cpu
        .as_slice::<f32>()
        .ok_or_else(|| WgpuError::ShapeMismatch("out not f32".into()))?
        .to_vec())
}

#[cfg(target_arch = "wasm32")]
mod web {
    use super::*;
    use wasm_bindgen::prelude::*;

    /// JS-callable: run one ResNet basic block on the given image.
    /// Returns the output as a Float32Array. Same-channels variant
    /// (c_in == c_out) for now; the demo page caps at small h × w
    /// so the round-trip stays fast.
    #[wasm_bindgen]
    pub async fn run_resnet_block_js(
        input_chw: &[f32],
        c_in: u32,
        c_out: u32,
        h: u32,
        w: u32,
    ) -> Result<js_sys::Float32Array, JsValue> {
        let backend = WgpuBackend::new()
            .await
            .map_err(|e| JsValue::from_str(&format!("backend init: {e}")))?;
        let out = run_resnet_block(
            &backend,
            input_chw,
            c_in as usize,
            c_out as usize,
            h as usize,
            w as usize,
        )
        .map_err(|e| JsValue::from_str(&format!("resnet_block: {e}")))?;
        Ok(js_sys::Float32Array::from(out.as_slice()))
    }
}

#[cfg(target_arch = "wasm32")]
pub use web::*;

#[cfg(all(test, not(target_arch = "wasm32"), feature = "gpu-tests"))]
mod gpu_tests {
    use super::*;

    #[test]
    fn resnet_block_runs_same_channels() {
        let backend = WgpuBackend::new_blocking().expect("init");
        let (c_in, h, w) = (4, 8, 8);
        let input: Vec<f32> = (0..c_in * h * w).map(|i| (i as f32) * 0.01).collect();
        let out = run_resnet_block(&backend, &input, c_in, c_in, h, w).unwrap();
        assert_eq!(out.len(), c_in * h * w);
        // After ReLU the result must be ≥ 0.
        for &v in &out {
            assert!(v >= 0.0, "negative value after final ReLU: {v}");
        }
    }

    #[test]
    fn resnet_block_runs_different_channels() {
        // c_in=3 → c_out=8 — 1×1 conv on the skip path kicks in.
        let backend = WgpuBackend::new_blocking().expect("init");
        let (c_in, c_out, h, w) = (3, 8, 6, 6);
        let input: Vec<f32> = (0..c_in * h * w).map(|i| (i as f32) * 0.02).collect();
        let out = run_resnet_block(&backend, &input, c_in, c_out, h, w).unwrap();
        assert_eq!(out.len(), c_out * h * w);
        for &v in &out {
            assert!(v >= 0.0);
        }
    }
}
