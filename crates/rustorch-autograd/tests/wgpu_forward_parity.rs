//! P3.Y plan, Phase B step 9cfcd9eb (Parity test forward CPU vs Wgpu).
//!
//! Validates the full dispatch chain (A1 Variable::device + A2
//! wgpu_backend singleton + A3 Backend trait + A4 impl Backend for
//! WgpuBackend + B1 forward dispatch in 33 ops + C1 backward dispatch)
//! by running a small Linear+ReLU computation on CPU and on Wgpu and
//! comparing cosine similarity ≥ 0.999.
//!
//! Gated by:
//! - `cfg(feature = "wgpu")` — autograd builds without the wgpu dep when
//!   the feature is off, so the whole test file is `cfg`-disabled then.
//! - `cfg_attr(not(feature = "gpu-tests"), ignore)` per individual test —
//!   needs a GPU adapter; CI without one runs the tests as `ignored`.
//!
//! Run with:
//! ```sh
//! cargo test -p rustorch-autograd --features wgpu,gpu-tests --test wgpu_forward_parity
//! ```

#![cfg(feature = "wgpu")]

use rustorch_autograd::{ops, Variable};
use rustorch_core::tensor::device::Device;
use rustorch_core::tensor::tensor_impl::Tensor;

/// Cosine similarity between two flat F32 buffers.
///
/// Used as the parity metric — strict element-wise equality is too
/// fragile because GPU FMA reordering produces tiny float drift, but
/// the geometry of the activation vectors must match the CPU result.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "cosine_similarity: length mismatch");
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for (&x, &y) in a.iter().zip(b.iter()) {
        let xf = x as f64;
        let yf = y as f64;
        dot += xf * yf;
        na += xf * xf;
        nb += yf * yf;
    }
    if na == 0.0 || nb == 0.0 {
        return 1.0;
    }
    (dot / (na.sqrt() * nb.sqrt())) as f32
}

/// Build a tensor of shape `shape` filled with deterministic values.
/// Seeds via a multiplier so different rows have varied magnitudes.
fn det_tensor(shape: &[usize], seed: f32) -> Tensor {
    let n: usize = shape.iter().product();
    let data: Vec<f32> = (0..n)
        .map(|i| ((i as f32 + 1.0) * seed * 0.01).sin())
        .collect();
    Tensor::from_vec(shape.to_vec(), data).expect("det_tensor build")
}

/// Move a Tensor onto a target device by tagging the device. With the
/// transitional Storage Option B, the actual data still lives in the
/// CPU shadow; the Wgpu kernels round-trip through `to_gpu`/`to_cpu`
/// every op call. Future Storage refactor (Option A) will land a
/// proper canonical GPU storage.
fn to_dev(t: Tensor, device: Device) -> Tensor {
    t.with_device(device)
}

/// Wrap a tensor as a non-grad Variable on the given device. Used for
/// inputs and weights that don't require grad in this parity test.
fn var(t: Tensor, device: Device) -> Variable {
    Variable::new(to_dev(t, device))
}

#[test]
#[cfg_attr(not(feature = "gpu-tests"), ignore = "needs a GPU adapter")]
fn add_parity_cpu_vs_wgpu() {
    let a = det_tensor(&[4, 8], 1.0);
    let b = det_tensor(&[4, 8], 2.0);

    let r_cpu = ops::add(&var(a.clone(), Device::Cpu), &var(b.clone(), Device::Cpu)).unwrap();
    let r_gpu = ops::add(&var(a, Device::Wgpu), &var(b, Device::Wgpu)).unwrap();

    let cs = cosine_similarity(
        r_cpu.tensor().as_slice::<f32>().unwrap(),
        r_gpu.tensor().as_slice::<f32>().unwrap(),
    );
    assert!(cs > 0.9999, "add cosine sim too low: {cs}");
}

#[test]
#[cfg_attr(not(feature = "gpu-tests"), ignore = "needs a GPU adapter")]
fn matmul_parity_cpu_vs_wgpu() {
    let a = det_tensor(&[8, 16], 1.0);
    let b = det_tensor(&[16, 4], 2.0);

    let r_cpu = ops::matmul(&var(a.clone(), Device::Cpu), &var(b.clone(), Device::Cpu)).unwrap();
    let r_gpu = ops::matmul(&var(a, Device::Wgpu), &var(b, Device::Wgpu)).unwrap();

    let cs = cosine_similarity(
        r_cpu.tensor().as_slice::<f32>().unwrap(),
        r_gpu.tensor().as_slice::<f32>().unwrap(),
    );
    assert!(cs > 0.999, "matmul cosine sim too low: {cs}");
}

#[test]
#[cfg_attr(not(feature = "gpu-tests"), ignore = "needs a GPU adapter")]
fn linear_relu_parity_cpu_vs_wgpu() {
    // A two-Linear MLP forward (without bias for now — bias requires the
    // add_bias impl on WgpuBackend, validated separately below):
    //   x  [B=4, In=16] @ W1 [16, 32] = [4, 32] -> relu -> @ W2 [32, 8] = [4, 8]
    let x = det_tensor(&[4, 16], 1.0);
    let w1 = det_tensor(&[16, 32], 0.5);
    let w2 = det_tensor(&[32, 8], 0.7);

    fn forward(x: Tensor, w1: Tensor, w2: Tensor, device: Device) -> Tensor {
        let xv = var(x, device);
        let w1v = var(w1, device);
        let w2v = var(w2, device);
        let h = ops::matmul(&xv, &w1v).unwrap();
        let h = ops::relu(&h).unwrap();
        let y = ops::matmul(&h, &w2v).unwrap();
        y.tensor().clone()
    }

    let r_cpu = forward(x.clone(), w1.clone(), w2.clone(), Device::Cpu);
    let r_gpu = forward(x, w1, w2, Device::Wgpu);

    let cs = cosine_similarity(
        r_cpu.as_slice::<f32>().unwrap(),
        r_gpu.as_slice::<f32>().unwrap(),
    );
    assert!(cs > 0.999, "linear+relu cosine sim too low: {cs}");
}

#[test]
#[cfg_attr(not(feature = "gpu-tests"), ignore = "needs a GPU adapter")]
fn linear_bias_relu_parity_cpu_vs_wgpu() {
    // Linear with bias:
    //   y = relu(x @ W + b) using ops::add_bias
    let x = det_tensor(&[4, 16], 1.0);
    let w = det_tensor(&[16, 8], 0.5);
    let b = det_tensor(&[8], 0.3);

    fn forward(x: Tensor, w: Tensor, b: Tensor, device: Device) -> Tensor {
        let xv = var(x, device);
        let wv = var(w, device);
        let bv = var(b, device);
        let h = ops::matmul(&xv, &wv).unwrap();
        let h = ops::add_bias(&h, &bv).unwrap();
        let y = ops::relu(&h).unwrap();
        y.tensor().clone()
    }

    let r_cpu = forward(x.clone(), w.clone(), b.clone(), Device::Cpu);
    let r_gpu = forward(x, w, b, Device::Wgpu);

    let cs = cosine_similarity(
        r_cpu.as_slice::<f32>().unwrap(),
        r_gpu.as_slice::<f32>().unwrap(),
    );
    assert!(cs > 0.999, "linear+bias+relu cosine sim too low: {cs}");
}

#[test]
#[cfg_attr(not(feature = "gpu-tests"), ignore = "needs a GPU adapter")]
fn backward_parity_matmul_sigmoid_matmul_sum() {
    // Backward parity (P3.Y plan, Phase C step 922e577c):
    // Compute y = sum(matmul(sigmoid(matmul(x, w1)), w2)) on CPU and Wgpu,
    // run backward(), and compare x.grad cosine similarity ≥ 0.999.
    //
    // Validates the FULL forward+backward dispatch chain through:
    //   - MatMulBackward.apply (uses matmul × 2 — works via WgpuBackend::matmul)
    //   - SigmoidBackward.apply (uses sub + mul + mul — works via WgpuBackend ops)
    //   - SumBackward.apply (pure CPU shape op — works via Storage Option B
    //     CPU shadow data)
    //
    // No new WGSL kernel needed: the backward composes from existing
    // forward ops on Wgpu. This proves C1 (Node device field + dispatch)
    // is correct end-to-end.
    use rustorch_autograd::backward;

    let x_data = det_tensor(&[4, 16], 1.0);
    let w1_data = det_tensor(&[16, 8], 0.5);
    let w2_data = det_tensor(&[8, 1], 0.7);

    fn forward_and_backward(
        x_data: Tensor,
        w1_data: Tensor,
        w2_data: Tensor,
        device: Device,
    ) -> Tensor {
        // x requires_grad so we can read its gradient back.
        let x = Variable::leaf(to_dev(x_data, device)).requires_grad(true);
        let w1 = Variable::leaf(to_dev(w1_data, device));
        let w2 = Variable::leaf(to_dev(w2_data, device));

        let h = ops::matmul(&x, &w1).unwrap();
        let h = ops::sigmoid(&h).unwrap();
        let y = ops::matmul(&h, &w2).unwrap();
        let loss = ops::sum(&y).unwrap();

        backward(&loss, None).unwrap();

        x.grad().expect("x.grad after backward")
    }

    let g_cpu = forward_and_backward(
        x_data.clone(),
        w1_data.clone(),
        w2_data.clone(),
        Device::Cpu,
    );
    let g_gpu = forward_and_backward(x_data, w1_data, w2_data, Device::Wgpu);

    assert_eq!(g_cpu.shape(), g_gpu.shape());
    let cs = cosine_similarity(
        g_cpu.as_slice::<f32>().unwrap(),
        g_gpu.as_slice::<f32>().unwrap(),
    );
    assert!(
        cs > 0.999,
        "x.grad cosine sim too low: {cs} (cpu={:?}, gpu={:?})",
        g_cpu.as_slice::<f32>().unwrap(),
        g_gpu.as_slice::<f32>().unwrap()
    );
}

#[test]
#[cfg_attr(not(feature = "gpu-tests"), ignore = "needs a GPU adapter")]
fn backward_parity_add_mul_sum_uses_unbroadcast_to() {
    // Backward parity for Add+Mul backward — exercises `unbroadcast_to`
    // (used by AddBackward/MulBackward when forward broadcast happens
    // along the bias axis). Confirms WgpuBackend::sum_dim + reshape +
    // unbroadcast_to compose correctly to match CPU gradients.
    //
    // Chain: (x + b) * w summed; backward through MulBackward + AddBackward.
    // x [4, 8], b [8] (broadcast), w [4, 8] same as x.
    use rustorch_autograd::backward;

    let x_data = det_tensor(&[4, 8], 1.0);
    let b_data = det_tensor(&[8], 0.5);
    let w_data = det_tensor(&[4, 8], 0.7);

    fn forward_and_backward(
        x_data: Tensor,
        b_data: Tensor,
        w_data: Tensor,
        device: Device,
    ) -> (Tensor, Tensor) {
        let x = Variable::leaf(to_dev(x_data, device)).requires_grad(true);
        let b = Variable::leaf(to_dev(b_data, device)).requires_grad(true);
        let w = Variable::leaf(to_dev(w_data, device));

        // y = (x + b) * w  [4, 8]
        let xb = ops::add(&x, &b).unwrap(); // x.shape == b.shape (broadcast happens via the trait if shapes differed)
        let y = ops::mul(&xb, &w).unwrap();
        let loss = ops::sum(&y).unwrap();

        backward(&loss, None).unwrap();

        (
            x.grad().expect("x.grad after backward"),
            b.grad().expect("b.grad after backward"),
        )
    }

    let (gx_cpu, gb_cpu) =
        forward_and_backward(x_data.clone(), b_data.clone(), w_data.clone(), Device::Cpu);
    let (gx_gpu, gb_gpu) = forward_and_backward(x_data, b_data, w_data, Device::Wgpu);

    assert_eq!(gx_cpu.shape(), gx_gpu.shape());
    assert_eq!(gb_cpu.shape(), gb_gpu.shape());
    let cs_x = cosine_similarity(
        gx_cpu.as_slice::<f32>().unwrap(),
        gx_gpu.as_slice::<f32>().unwrap(),
    );
    let cs_b = cosine_similarity(
        gb_cpu.as_slice::<f32>().unwrap(),
        gb_gpu.as_slice::<f32>().unwrap(),
    );
    assert!(cs_x > 0.999, "x.grad cosine sim too low: {cs_x}");
    assert!(cs_b > 0.999, "b.grad cosine sim too low: {cs_b}");
}

#[test]
fn device_mismatch_returns_clear_error() {
    // This test runs without a GPU because it only exercises the
    // dispatch gate, which trips before any kernel call. Validates
    // that mixing CPU and Wgpu Variables returns a DeviceMismatch
    // error, not a silent CPU-promotion or panic.
    let a = Variable::new(det_tensor(&[4, 8], 1.0)); // CPU
    let b = Variable::new(det_tensor(&[4, 8], 2.0).with_device(Device::Wgpu));

    let err = ops::add(&a, &b).unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("device mismatch"),
        "expected device mismatch error, got: {msg}"
    );
}
