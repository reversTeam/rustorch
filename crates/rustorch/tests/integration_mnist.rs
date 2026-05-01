//! Integration test: MNIST-style MLP end-to-end with state_dict
//! save/load roundtrip and DataLoader.

use rustorch_autograd::{backward, Variable};
use rustorch_cpu::backend::Reduction;
use rustorch_data::{synthetic_mnist, DataLoader, RandomSampler, TensorDataset};
use rustorch_nn::{
    load_state_dict, state_dict, Criterion, CrossEntropyLoss, Linear, Module, Relu, Sequential,
};
use rustorch_optim::{Adam, Optimizer};

#[test]
fn mnist_mlp_full_pipeline() {
    let n_train = 256;
    let batch_size = 32;
    let lr = 1e-3_f32;

    let train_ds = synthetic_mnist(n_train, 0xC0FF_EEAA);

    let model = Sequential::new()
        .add(Linear::new(28 * 28, 128))
        .add(Relu)
        .add(Linear::new(128, 64))
        .add(Relu)
        .add(Linear::new(64, 10));

    let mut opt = Adam::new(model.parameters(), lr);

    // 1 epoch is enough to verify end-to-end.
    let sampler = RandomSampler::with_seed(n_train, 42);
    let train_ds_clone =
        TensorDataset::new(train_ds.inputs().clone(), train_ds.targets().clone()).unwrap();
    let mut loader = DataLoader::new(train_ds_clone, sampler, batch_size).drop_last(true);

    let mut total_steps = 0;
    let mut last_loss = f32::INFINITY;
    for batch in loader.iter_epoch() {
        let (x_4d, y) = batch.unwrap();
        // Flatten [B, 1, 28, 28] → [B, 784]
        let x_flat = rustorch_core::Tensor::from_vec(
            [batch_size, 28 * 28],
            x_4d.as_slice::<f32>().unwrap().to_vec(),
        )
        .unwrap();
        let xv = Variable::new(x_flat);
        let yv = Variable::new(y);

        let logits = model.forward(&xv).unwrap();
        let loss = CrossEntropyLoss::new(Reduction::Mean)
            .forward(&logits, &yv)
            .unwrap();
        let loss_val = loss.tensor().as_slice::<f32>().unwrap()[0];
        assert!(loss_val.is_finite());
        last_loss = loss_val;
        total_steps += 1;

        opt.zero_grad();
        backward(&loss, None).unwrap();
        opt.step();
    }
    assert!(total_steps > 0);
    assert!(last_loss < 100.0, "final loss insane: {last_loss}");

    // state_dict round-trip via in-memory safetensors.
    let sd = state_dict(&model);
    let mut buf = Vec::<u8>::new();
    rustorch_serde::write_to(&mut buf, &sd).unwrap();
    let mut reader = std::io::Cursor::new(buf);
    let sd_back = rustorch_serde::read_from(&mut reader).unwrap();
    assert_eq!(sd.len(), sd_back.len());

    // Build a fresh model and load the weights.
    let fresh = Sequential::new()
        .add(Linear::new(28 * 28, 128))
        .add(Relu)
        .add(Linear::new(128, 64))
        .add(Relu)
        .add(Linear::new(64, 10));
    let report = load_state_dict(&fresh, &sd_back, true).unwrap();
    assert!(report.missing.is_empty());
    assert!(report.unexpected.is_empty());

    // After load, fresh's state_dict should equal the original.
    let sd_fresh = state_dict(&fresh);
    for (k, v) in &sd {
        let v2 = &sd_fresh[k];
        assert_eq!(
            v.as_slice::<f32>().unwrap(),
            v2.as_slice::<f32>().unwrap(),
            "mismatch for key {k}"
        );
    }
}
