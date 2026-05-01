//! Integration tests for P1.4 reductions / softmax / loss kernels.

use rustorch_core::tensor::tensor_impl::Tensor;
use rustorch_cpu::backend::Reduction;
use rustorch_cpu::cpu_backend::cpu_backend;
use rustorch_cpu::error::BackendError;

// ---------------------- Reductions ----------------------

#[test]
fn sum_dim_keepdim() {
    let backend = cpu_backend();
    let t = Tensor::from_vec([2usize, 3], vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let s = backend.sum_dim(&t, &[1], true).unwrap();
    assert_eq!(s.shape(), &[2, 1]);
    assert_eq!(s.as_slice::<f32>().unwrap(), &[6.0_f32, 15.0]);
    let s_drop = backend.sum_dim(&t, &[1], false).unwrap();
    assert_eq!(s_drop.shape(), &[2]);
    assert_eq!(s_drop.as_slice::<f32>().unwrap(), &[6.0_f32, 15.0]);
}

#[test]
fn sum_dim_multiple_axes() {
    let backend = cpu_backend();
    let t = Tensor::from_vec(
        [2usize, 2, 2],
        vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
    )
    .unwrap();
    let s = backend.sum_dim(&t, &[1, 2], false).unwrap();
    assert_eq!(s.shape(), &[2]);
    assert_eq!(s.as_slice::<f32>().unwrap(), &[10.0_f32, 26.0]);
}

#[test]
fn mean_dim() {
    let backend = cpu_backend();
    let t = Tensor::from_vec(
        [2usize, 4],
        vec![2.0_f32, 4.0, 6.0, 8.0, 1.0, 1.0, 1.0, 1.0],
    )
    .unwrap();
    let m = backend.mean_dim(&t, &[1], false).unwrap();
    assert_eq!(m.as_slice::<f32>().unwrap(), &[5.0_f32, 1.0]);
}

#[test]
fn max_dim_argmax() {
    let backend = cpu_backend();
    let t = Tensor::from_vec([2usize, 3], vec![1.0_f32, 5.0, 2.0, 9.0, 0.5, 3.0]).unwrap();
    let m = backend.max_dim(&t, 1, false).unwrap();
    assert_eq!(m.as_slice::<f32>().unwrap(), &[5.0_f32, 9.0]);
    let am = backend.argmax(&t, 1, false).unwrap();
    assert_eq!(am.as_slice::<i64>().unwrap(), &[1_i64, 0]);
}

#[test]
fn min_dim_argmin() {
    let backend = cpu_backend();
    let t = Tensor::from_vec([2usize, 3], vec![1.0_f32, 5.0, 2.0, 9.0, 0.5, 3.0]).unwrap();
    let m = backend.min_dim(&t, 1, false).unwrap();
    assert_eq!(m.as_slice::<f32>().unwrap(), &[1.0_f32, 0.5]);
    let am = backend.argmin(&t, 1, false).unwrap();
    assert_eq!(am.as_slice::<i64>().unwrap(), &[0_i64, 1]);
}

#[test]
fn var_dim_welford_stable_extreme_values() {
    // [1e10, 1e10+1, 1e10+2] in F64 — Welford handles this; naive
    // (E[x²] - E[x]²) loses all precision.
    let backend = cpu_backend();
    let t = Tensor::from_vec_typed::<f64, _>([1usize, 3], vec![1e10_f64, 1e10 + 1.0, 1e10 + 2.0])
        .unwrap();
    let v = backend.var_dim(&t, 1, false, false).unwrap();
    let val = v.as_slice::<f64>().unwrap()[0];
    // Population variance of [a, a+1, a+2] is 2/3 ≈ 0.6667
    assert!((val - 0.6667).abs() < 0.01, "got {val}");
}

#[test]
fn var_dim_basic_f32() {
    // [1, 2, 3] population variance = 2/3
    let backend = cpu_backend();
    let t = Tensor::from_vec([1usize, 3], vec![1.0_f32, 2.0, 3.0]).unwrap();
    let v = backend.var_dim(&t, 1, false, false).unwrap();
    let val = v.as_slice::<f32>().unwrap()[0];
    assert!((val - 0.6667).abs() < 1e-3, "got {val}");
}

#[test]
fn full_max_min_prod_all_any() {
    let backend = cpu_backend();
    let t = Tensor::from_vec([4usize], vec![1.0_f32, 2.0, 3.0, 4.0]).unwrap();
    assert_eq!(backend.max(&t).unwrap().as_slice::<f32>().unwrap(), &[4.0]);
    assert_eq!(backend.min(&t).unwrap().as_slice::<f32>().unwrap(), &[1.0]);
    assert_eq!(
        backend.prod(&t).unwrap().as_slice::<f32>().unwrap(),
        &[24.0]
    );
    let bools = Tensor::from_vec_typed::<bool, _>([3usize], vec![true, true, true]).unwrap();
    assert_eq!(
        backend.all(&bools).unwrap().as_slice::<bool>().unwrap(),
        &[true]
    );
    assert_eq!(
        backend.any(&bools).unwrap().as_slice::<bool>().unwrap(),
        &[true]
    );
    let mixed = Tensor::from_vec_typed::<bool, _>([3usize], vec![true, false, true]).unwrap();
    assert_eq!(
        backend.all(&mixed).unwrap().as_slice::<bool>().unwrap(),
        &[false]
    );
    assert_eq!(
        backend.any(&mixed).unwrap().as_slice::<bool>().unwrap(),
        &[true]
    );
}

#[test]
fn cumsum_cumprod() {
    let backend = cpu_backend();
    let t = Tensor::from_vec([4usize], vec![1.0_f32, 2.0, 3.0, 4.0]).unwrap();
    let cs = backend.cumsum(&t, 0).unwrap();
    assert_eq!(cs.as_slice::<f32>().unwrap(), &[1.0, 3.0, 6.0, 10.0]);
    let cp = backend.cumprod(&t, 0).unwrap();
    assert_eq!(cp.as_slice::<f32>().unwrap(), &[1.0, 2.0, 6.0, 24.0]);
}

#[test]
fn reduction_dim_out_of_range() {
    let backend = cpu_backend();
    let t = Tensor::from_vec([3usize], vec![1.0_f32; 3]).unwrap();
    assert!(matches!(
        backend.sum_dim(&t, &[5], false),
        Err(BackendError::IndexOutOfBounds { .. })
    ));
}

#[test]
fn sum_dim_empty_tensor_zero_dim_returns_zero() {
    let backend = cpu_backend();
    let t = Tensor::zeros([0usize, 3]);
    let s = backend.sum_dim(&t, &[0], false).unwrap();
    // Reducing the 0-axis of an empty tensor → shape [3] of zeros.
    assert_eq!(s.shape(), &[3]);
    assert_eq!(s.as_slice::<f32>().unwrap(), &[0.0_f32, 0.0, 0.0]);
}

// ---------------------- Softmax / log_softmax ----------------------

#[test]
fn softmax_sums_to_1() {
    let backend = cpu_backend();
    let t = Tensor::from_vec([1usize, 3], vec![1.0_f32, 2.0, 3.0]).unwrap();
    let s = backend.softmax(&t, 1).unwrap();
    let sum: f32 = s.as_slice::<f32>().unwrap().iter().sum();
    assert!((sum - 1.0).abs() < 1e-6);
}

#[test]
fn softmax_translation_invariant() {
    // softmax(x + c) == softmax(x)
    let backend = cpu_backend();
    let t = Tensor::from_vec([1usize, 3], vec![1.0_f32, 2.0, 3.0]).unwrap();
    let t_plus = Tensor::from_vec([1usize, 3], vec![1001.0_f32, 1002.0, 1003.0]).unwrap();
    let a = backend.softmax(&t, 1).unwrap();
    let b = backend.softmax(&t_plus, 1).unwrap();
    let av = a.as_slice::<f32>().unwrap();
    let bv = b.as_slice::<f32>().unwrap();
    for (x, y) in av.iter().zip(bv.iter()) {
        assert!((x - y).abs() < 1e-6);
    }
}

#[test]
fn softmax_stable_on_extreme_inputs() {
    // [0, 1000] should saturate to [0, 1] without NaN.
    let backend = cpu_backend();
    let t = Tensor::from_vec([1usize, 2], vec![0.0_f32, 1000.0]).unwrap();
    let s = backend.softmax(&t, 1).unwrap();
    let v = s.as_slice::<f32>().unwrap();
    assert!(v[0] >= 0.0 && v[0] < 1e-6);
    assert!((v[1] - 1.0).abs() < 1e-6);
}

#[test]
fn log_softmax_consistent_with_softmax() {
    // log(softmax(x)) == log_softmax(x)
    let backend = cpu_backend();
    let t = Tensor::from_vec([1usize, 4], vec![1.0_f32, -2.0, 0.5, 3.0]).unwrap();
    let s = backend.softmax(&t, 1).unwrap();
    let ls = backend.log_softmax(&t, 1).unwrap();
    let sv = s.as_slice::<f32>().unwrap();
    let lsv = ls.as_slice::<f32>().unwrap();
    for (a, b) in sv.iter().zip(lsv.iter()) {
        assert!(
            (a.ln() - b).abs() < 1e-5,
            "softmax.ln() != log_softmax: {} vs {}",
            a.ln(),
            b
        );
    }
}

#[test]
fn softmax_dim_out_of_range() {
    let backend = cpu_backend();
    let t = Tensor::from_vec([3usize], vec![1.0_f32; 3]).unwrap();
    assert!(matches!(
        backend.softmax(&t, 5),
        Err(BackendError::IndexOutOfBounds { .. })
    ));
}

// ---------------------- Loss functions ----------------------

#[test]
fn mse_loss_zeros_vs_ones() {
    let backend = cpu_backend();
    let z = Tensor::from_vec([3usize], vec![0.0_f32; 3]).unwrap();
    let o = Tensor::from_vec([3usize], vec![1.0_f32; 3]).unwrap();
    let m = backend.mse_loss(&z, &o, Reduction::Mean).unwrap();
    assert_eq!(m.as_slice::<f32>().unwrap(), &[1.0_f32]);
    let s = backend.mse_loss(&z, &o, Reduction::Sum).unwrap();
    assert_eq!(s.as_slice::<f32>().unwrap(), &[3.0_f32]);
    let n = backend.mse_loss(&z, &o, Reduction::None).unwrap();
    assert_eq!(n.as_slice::<f32>().unwrap(), &[1.0_f32, 1.0, 1.0]);
}

#[test]
fn nll_loss_picks_target_log_prob() {
    let backend = cpu_backend();
    // log_probs [2, 3] (manual log_softmax of [[1,2,3],[4,5,6]] like values)
    let lp = Tensor::from_vec([2usize, 3], vec![-2.0_f32, -1.0, -0.5, -1.5, -0.3, -2.5]).unwrap();
    let target = Tensor::from_vec_typed::<i64, _>([2usize], vec![2_i64, 0]).unwrap();
    // expected -log_probs = -(-0.5) and -(-1.5) → 0.5, 1.5; mean = 1.0
    let l = backend.nll_loss(&lp, &target, Reduction::Mean).unwrap();
    assert!((l.as_slice::<f32>().unwrap()[0] - 1.0).abs() < 1e-6);
}

#[test]
fn cross_entropy_matches_log_softmax_then_nll() {
    let backend = cpu_backend();
    let logits = Tensor::from_vec([2usize, 3], vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let target = Tensor::from_vec_typed::<i64, _>([2usize], vec![2_i64, 0]).unwrap();
    let ce = backend
        .cross_entropy(&logits, &target, Reduction::Mean)
        .unwrap();
    // Manual computation
    let ls = backend.log_softmax(&logits, 1).unwrap();
    let manual = backend.nll_loss(&ls, &target, Reduction::Mean).unwrap();
    let a = ce.as_slice::<f32>().unwrap()[0];
    let b = manual.as_slice::<f32>().unwrap()[0];
    assert!((a - b).abs() < 1e-6);
}

#[test]
fn cross_entropy_target_dtype_must_be_i64() {
    let backend = cpu_backend();
    let logits = Tensor::from_vec([1usize, 3], vec![1.0_f32, 2.0, 3.0]).unwrap();
    let target = Tensor::from_vec_typed::<i32, _>([1usize], vec![0_i32]).unwrap();
    assert!(matches!(
        backend.cross_entropy(&logits, &target, Reduction::Mean),
        Err(BackendError::DtypeMismatch { .. })
    ));
}

#[test]
fn nll_loss_out_of_bounds_target() {
    let backend = cpu_backend();
    let lp = Tensor::from_vec([1usize, 3], vec![-1.0_f32, -2.0, -3.0]).unwrap();
    let target = Tensor::from_vec_typed::<i64, _>([1usize], vec![5_i64]).unwrap();
    assert!(matches!(
        backend.nll_loss(&lp, &target, Reduction::Mean),
        Err(BackendError::IndexOutOfBounds { .. })
    ));
}

#[test]
fn bce_with_logits_stable_on_extremes() {
    // bce_with_logits([1000, -1000], [1, 0]) ≈ 0 + 0 ≈ 0 (perfect predictions).
    let backend = cpu_backend();
    let x = Tensor::from_vec([2usize], vec![1000.0_f32, -1000.0]).unwrap();
    let t = Tensor::from_vec([2usize], vec![1.0_f32, 0.0]).unwrap();
    let l = backend.bce_with_logits(&x, &t, Reduction::Mean).unwrap();
    let v = l.as_slice::<f32>().unwrap()[0];
    assert!(v.is_finite() && v.abs() < 1e-3, "expected ~0, got {v}");
}

#[test]
fn cross_entropy_full_path_smoke() {
    // Random-ish 4x10 logits + 4 targets. Just ensure the loss is finite.
    let backend = cpu_backend();
    let logits: Vec<f32> = (0..40).map(|i| (i as f32) * 0.1).collect();
    let l = Tensor::from_vec([4usize, 10], logits).unwrap();
    let target = Tensor::from_vec_typed::<i64, _>([4usize], vec![0_i64, 1, 2, 3]).unwrap();
    let ce = backend.cross_entropy(&l, &target, Reduction::Mean).unwrap();
    let v = ce.as_slice::<f32>().unwrap()[0];
    assert!(v.is_finite() && v > 0.0);
}
