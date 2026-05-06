//! MNIST training on the wgpu backend (P3.Y plan, Phase F step f5dbd791).
//!
//! Trains a tiny MLP on `synthetic_mnist` with everything dispatched
//! through `WgpuBackend` (Metal on macOS, Vulkan on Linux, DX12 on
//! Windows, WebGPU on wasm32). Demonstrates the end-to-end API:
//! - `model.to_device(Device::Wgpu)` moves params to the GPU device tag.
//! - `ops::*` forward calls dispatch via `Variable::device()`.
//! - `backward()` walks `Node::apply()` bodies that route through
//!   `pick_backend(self.device)`.
//! - `AdamW::step()` updates params via `write_param_data` which
//!   preserves the device tag.
//!
//! Run with:
//! ```sh
//! cargo run -p rustorch --release --example mnist_wgpu --features wgpu
//! ```
//!
//! With Storage Option B (transitional), every op pays a host↔device
//! round-trip, so this example demonstrates **correctness** rather than
//! peak GPU throughput. Storage Option A migration (Tensor enum
//! `Cpu | Wgpu`) is the perf follow-up.

#![allow(missing_docs)]

use rustorch::nn::module::Module;
use rustorch::nn::Linear;
use rustorch_autograd::{backward, ops, Variable};
use rustorch_core::tensor::device::Device;
use rustorch_core::tensor::tensor_impl::Tensor;
use rustorch_cpu::backend::Reduction;
use rustorch_data::bundled::synthetic_mnist;
use rustorch_optim::{AdamW, Optimizer};

/// Tiny 2-layer MLP: 784 → hidden → 10 with bias + ReLU between.
struct MnistMlp {
    fc1: Linear,
    fc2: Linear,
}

impl MnistMlp {
    fn new(hidden: usize) -> Self {
        Self {
            fc1: Linear::new(784, hidden),
            fc2: Linear::new(hidden, 10),
        }
    }

    fn forward(&self, x: &Variable) -> Variable {
        let h = self.fc1.forward(x).expect("fc1 forward");
        let h = ops::relu(&h).expect("relu");
        self.fc2.forward(&h).expect("fc2 forward")
    }
}

impl Module for MnistMlp {
    fn forward(&self, input: &Variable) -> Result<Variable, rustorch::nn::module::ModuleError> {
        Ok(MnistMlp::forward(self, input))
    }

    fn parameters(&self) -> Vec<Variable> {
        let mut ps = self.fc1.parameters();
        ps.extend(self.fc2.parameters());
        ps
    }
}

fn flatten_images(images_chw: &Tensor) -> Tensor {
    // [N, 1, 28, 28] → [N, 784], reusing the shadow buffer.
    let shape = images_chw.shape();
    let n = shape[0];
    let data = images_chw.as_slice::<f32>().expect("f32 mnist images");
    Tensor::from_vec([n, 784], data.to_vec()).expect("flatten")
}

/// Convert i64 class labels [N] → one-hot `[N, 10]` f32 — used for the
/// MSE-style loss path when `cross_entropy` falls back through CPU.
fn one_hot(labels: &Tensor, n_classes: usize) -> Tensor {
    let n = labels.numel();
    let labels = labels.as_slice::<i64>().expect("i64 labels");
    let mut data = vec![0.0_f32; n * n_classes];
    for (i, &l) in labels.iter().enumerate() {
        let l = l as usize;
        data[i * n_classes + l] = 1.0;
    }
    Tensor::from_vec([n, n_classes], data).expect("one_hot")
}

fn main() {
    let device = Device::Wgpu;
    println!("=== MNIST training on {device} ===");

    // 1) Tiny dataset (synthetic) — N=512 train, N=128 test. Real MNIST
    // would plug in `rustorch_data::mnist::Mnist::new(...)` (when shipped).
    let train = synthetic_mnist(512, 0xDEAD_BEEF);
    let test = synthetic_mnist(128, 0xCAFE_BABE);

    // 2) Build model and move to Wgpu — Module::to_device walks
    // parameters() and re-tags.
    let mut model = MnistMlp::new(64);
    model.to_device(device);
    println!(
        "model: 2-layer MLP (784 → 64 → 10), {} params, on {device}",
        model.parameters().len()
    );

    // 3) Materialize tensors on Wgpu (data tag).
    let xs = flatten_images(train.inputs()).with_device(device);
    let ys = one_hot(train.targets(), 10).with_device(device);
    let xs_test = flatten_images(test.inputs()).with_device(device);
    let ys_test = one_hot(test.targets(), 10).with_device(device);

    let xs_var = Variable::new(xs);
    let ys_var = Variable::new(ys);

    // 4) Optimiser — AdamW with 0.005 lr, 50 steps full-batch.
    let mut opt = AdamW::new(model.parameters(), 0.005);

    println!("training for 50 full-batch steps…");
    let mut prev_loss = f32::INFINITY;
    for step in 0..50 {
        opt.zero_grad();
        let logits = MnistMlp::forward(&model, &xs_var);
        // MSE on one-hot targets — proxies cross-entropy for this demo
        // without depending on the cross_entropy op chain.
        let loss = ops::mse_loss(&logits, &ys_var, Reduction::Mean).expect("mse");
        let loss_val = loss.tensor().as_slice::<f32>().unwrap()[0];
        if step % 10 == 0 || step == 49 {
            println!("  step {step:>2}: loss = {loss_val:.4}");
        }
        backward(&loss, None).expect("backward");
        opt.step();
        prev_loss = loss_val;
    }
    println!("final training loss: {prev_loss:.4}");

    // 5) Evaluate on the held-out test set — argmax of logits vs labels.
    let test_logits = MnistMlp::forward(&model, &Variable::new(xs_test));
    let logits = test_logits.tensor();
    let logits_data = logits.as_slice::<f32>().expect("f32 logits");
    let targets_oh = ys_test.as_slice::<f32>().unwrap();
    let n = logits.shape()[0];
    let n_classes = logits.shape()[1];
    let mut correct = 0usize;
    for i in 0..n {
        let row = &logits_data[i * n_classes..(i + 1) * n_classes];
        let true_row = &targets_oh[i * n_classes..(i + 1) * n_classes];
        let pred = row
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(idx, _)| idx)
            .unwrap();
        let truth = true_row
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(idx, _)| idx)
            .unwrap();
        if pred == truth {
            correct += 1;
        }
    }
    let acc = (correct as f32 / n as f32) * 100.0;
    println!("test accuracy: {acc:.1}% ({correct} / {n})");
    println!("=== done ===");
}
