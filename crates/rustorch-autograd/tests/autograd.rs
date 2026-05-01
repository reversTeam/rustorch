//! Autograd integration tests (P1.5).
//!
//! Validates the backward formulas via:
//! - hand-computed analytic checks (e.g. d/dx (x*x).sum() == 2x)
//! - finite-difference gradient checks
//! - chain-rule properties

use rustorch_autograd::ops::{
    abs, add, bmm, cross_entropy, div, exp, leaky_relu, log, log_softmax, matmul, mean, mse_loss,
    mul, neg, pow_scalar, relu, sigmoid, silu, softmax, sqrt, sub, sum, tanh,
};
use rustorch_autograd::{backward, no_grad, with_grad, Variable};
use rustorch_core::tensor::tensor_impl::Tensor;
use rustorch_cpu::backend::Reduction;

// -------------------- core formulas --------------------

#[test]
fn add_backward_distributes_grad_to_both_inputs() {
    let x = Variable::leaf(Tensor::from_vec([3], vec![1.0_f32, 2.0, 3.0]).unwrap());
    let y = Variable::leaf(Tensor::from_vec([3], vec![10.0_f32, 20.0, 30.0]).unwrap());
    let s = sum(&add(&x, &y).unwrap()).unwrap();
    backward(&s, None).unwrap();
    let dx = x.grad().unwrap();
    let dy = y.grad().unwrap();
    assert_eq!(dx.as_slice::<f32>().unwrap(), &[1.0_f32, 1.0, 1.0]);
    assert_eq!(dy.as_slice::<f32>().unwrap(), &[1.0_f32, 1.0, 1.0]);
}

#[test]
fn x_squared_sum_gradient_equals_2x() {
    // (x*x).sum() backward → dx = 2x
    let x = Variable::leaf(Tensor::from_vec([3], vec![3.0_f32, 4.0, 5.0]).unwrap());
    let xsq = mul(&x, &x).unwrap();
    let s = sum(&xsq).unwrap();
    backward(&s, None).unwrap();
    let dx = x.grad().unwrap();
    assert_eq!(dx.as_slice::<f32>().unwrap(), &[6.0_f32, 8.0, 10.0]);
}

#[test]
fn neg_backward_negates_grad() {
    let x = Variable::leaf(Tensor::from_vec([3], vec![1.0_f32, 2.0, 3.0]).unwrap());
    let s = sum(&neg(&x).unwrap()).unwrap();
    backward(&s, None).unwrap();
    let dx = x.grad().unwrap();
    assert_eq!(dx.as_slice::<f32>().unwrap(), &[-1.0_f32, -1.0, -1.0]);
}

#[test]
fn sub_backward_correct_signs() {
    let x = Variable::leaf(Tensor::from_vec([2], vec![5.0_f32, 7.0]).unwrap());
    let y = Variable::leaf(Tensor::from_vec([2], vec![2.0_f32, 3.0]).unwrap());
    let s = sum(&sub(&x, &y).unwrap()).unwrap();
    backward(&s, None).unwrap();
    assert_eq!(
        x.grad().unwrap().as_slice::<f32>().unwrap(),
        &[1.0_f32, 1.0]
    );
    assert_eq!(
        y.grad().unwrap().as_slice::<f32>().unwrap(),
        &[-1.0_f32, -1.0]
    );
}

#[test]
fn matmul_backward_2d() {
    // A [2,3], B [3,2]; loss = sum(A @ B)
    // dA = grad @ B.T (where grad = ones [2,2])
    // dB = A.T @ grad
    let a = Variable::leaf(
        Tensor::from_vec([2usize, 3], vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
    );
    let b = Variable::leaf(
        Tensor::from_vec([3usize, 2], vec![7.0_f32, 8.0, 9.0, 10.0, 11.0, 12.0]).unwrap(),
    );
    let c = matmul(&a, &b).unwrap();
    let s = sum(&c).unwrap();
    backward(&s, None).unwrap();

    // dA = ones[2,2] @ B.T   where B.T = [[7,9,11],[8,10,12]]
    //    = ones row · B.T = column sums of B (each row dup)
    //    = [[15,19,23],[15,19,23]]
    let da = a.grad().unwrap();
    assert_eq!(da.shape(), &[2, 3]);
    assert_eq!(
        da.as_slice::<f32>().unwrap(),
        &[15.0_f32, 19.0, 23.0, 15.0, 19.0, 23.0]
    );

    // dB = A.T @ ones[2,2]
    //    A.T = [[1,4],[2,5],[3,6]]
    //    A.T @ ones[2,2] = [[5,5],[7,7],[9,9]]
    let db = b.grad().unwrap();
    assert_eq!(db.shape(), &[3, 2]);
    assert_eq!(
        db.as_slice::<f32>().unwrap(),
        &[5.0_f32, 5.0, 7.0, 7.0, 9.0, 9.0]
    );
}

#[test]
fn relu_backward_zeroes_negative_path() {
    let x = Variable::leaf(Tensor::from_vec([4], vec![-1.0_f32, 0.0, 1.0, 2.0]).unwrap());
    let s = sum(&relu(&x).unwrap()).unwrap();
    backward(&s, None).unwrap();
    // d/dx relu(x) = 1 if x > 0 else 0
    assert_eq!(
        x.grad().unwrap().as_slice::<f32>().unwrap(),
        &[0.0_f32, 0.0, 1.0, 1.0]
    );
}

#[test]
fn sigmoid_backward_at_zero_is_quarter() {
    // sigmoid(0) = 0.5; d/dx sigmoid(0) = 0.5 * 0.5 = 0.25
    let x = Variable::leaf(Tensor::from_vec([1], vec![0.0_f32]).unwrap());
    let s = sum(&sigmoid(&x).unwrap()).unwrap();
    backward(&s, None).unwrap();
    let g = x.grad().unwrap().as_slice::<f32>().unwrap()[0];
    assert!((g - 0.25).abs() < 1e-6);
}

#[test]
fn tanh_backward_at_zero_is_one() {
    // tanh(0) = 0; d/dx tanh(0) = 1 - 0² = 1
    let x = Variable::leaf(Tensor::from_vec([1], vec![0.0_f32]).unwrap());
    let s = sum(&tanh(&x).unwrap()).unwrap();
    backward(&s, None).unwrap();
    let g = x.grad().unwrap().as_slice::<f32>().unwrap()[0];
    assert!((g - 1.0).abs() < 1e-6);
}

#[test]
fn mean_backward_distributes_one_over_n() {
    let x = Variable::leaf(Tensor::from_vec([4], vec![1.0_f32, 2.0, 3.0, 4.0]).unwrap());
    let m = mean(&x).unwrap();
    backward(&m, None).unwrap();
    // d/dx mean(x) = 1/n on every element; n = 4 → [0.25; 4]
    assert_eq!(x.grad().unwrap().as_slice::<f32>().unwrap(), &[0.25_f32; 4]);
}

#[test]
fn chain_rule_addmul() {
    // y = (x + a) * b ; dx = b
    let x = Variable::leaf(Tensor::from_vec([3], vec![1.0_f32, 2.0, 3.0]).unwrap());
    let a = Variable::new(Tensor::from_vec([3], vec![1.0_f32, 1.0, 1.0]).unwrap());
    let b = Variable::new(Tensor::from_vec([3], vec![5.0_f32, 6.0, 7.0]).unwrap());
    let s = sum(&mul(&add(&x, &a).unwrap(), &b).unwrap()).unwrap();
    backward(&s, None).unwrap();
    assert_eq!(
        x.grad().unwrap().as_slice::<f32>().unwrap(),
        &[5.0_f32, 6.0, 7.0]
    );
}

#[test]
fn multi_path_grad_accumulates() {
    // y = x + x → dy/dx = 2 (gradient should accumulate via two paths
    // converging on the same leaf).
    let x = Variable::leaf(Tensor::from_vec([3], vec![1.0_f32, 2.0, 3.0]).unwrap());
    let s = sum(&add(&x, &x).unwrap()).unwrap();
    backward(&s, None).unwrap();
    assert_eq!(
        x.grad().unwrap().as_slice::<f32>().unwrap(),
        &[2.0_f32, 2.0, 2.0]
    );
}

#[test]
fn requires_grad_false_path_is_silent() {
    // When neither input requires grad, no grad_fn is created and
    // backward errs out cleanly.
    let x = Variable::new(Tensor::from_vec([3], vec![1.0_f32; 3]).unwrap());
    let y = Variable::new(Tensor::from_vec([3], vec![2.0_f32; 3]).unwrap());
    let s = sum(&add(&x, &y).unwrap()).unwrap();
    assert!(!s.requires_grad);
    assert!(matches!(
        backward(&s, None),
        Err(rustorch_autograd::BackwardError::NoGradFn)
    ));
}

// -------------------- no_grad / with_grad --------------------

#[test]
fn no_grad_skips_grad_fn_creation() {
    let x = Variable::leaf(Tensor::from_vec([2], vec![1.0_f32, 2.0]).unwrap());
    let y_with = mul(&x, &x).unwrap();
    assert!(y_with.requires_grad, "outside no_grad: requires_grad set");

    let y_without = no_grad(|| mul(&x, &x).unwrap());
    assert!(
        !y_without.requires_grad,
        "inside no_grad: requires_grad cleared"
    );
}

#[test]
fn with_grad_inside_no_grad_re_enables() {
    let x = Variable::leaf(Tensor::from_vec([2], vec![1.0_f32, 2.0]).unwrap());
    let y = no_grad(|| with_grad(|| mul(&x, &x).unwrap()));
    assert!(y.requires_grad);
}

// -------------------- detach --------------------

#[test]
fn detach_breaks_graph() {
    let x = Variable::leaf(Tensor::from_vec([2], vec![1.0_f32, 2.0]).unwrap());
    let y = mul(&x, &x).unwrap().detach();
    assert!(!y.requires_grad);
    let s = sum(&mul(&y, &y).unwrap()).unwrap(); // no path back to x via y
    assert!(matches!(
        backward(&s, None),
        Err(rustorch_autograd::BackwardError::NoGradFn)
    ));
}

// -------------------- cross_entropy --------------------

#[test]
fn cross_entropy_backward_softmax_minus_one_hot() {
    // For input x [N, C], target t, reduction Mean:
    //   dL/dx = (softmax(x) - one_hot(t)) / N
    let logits = Variable::leaf(
        Tensor::from_vec([2usize, 3], vec![1.0_f32, 2.0, 3.0, 1.0, 1.0, 1.0]).unwrap(),
    );
    let target = Variable::new(Tensor::from_vec_typed::<i64, _>([2usize], vec![2_i64, 0]).unwrap());
    let loss = cross_entropy(&logits, &target, Reduction::Mean).unwrap();
    backward(&loss, None).unwrap();
    let g = logits.grad().unwrap();
    // softmax([1,2,3]) ≈ [0.0900, 0.2447, 0.6652]; minus one-hot @ idx 2 = [0.0900, 0.2447, -0.3348]
    // softmax([1,1,1]) = [1/3, 1/3, 1/3]; minus one-hot @ idx 0 = [-0.6667, 0.3333, 0.3333]
    // Then divide by N=2.
    let v = g.as_slice::<f32>().unwrap();
    // Sample 0 (target=2)
    assert!((v[0] - 0.0900 / 2.0).abs() < 1e-3);
    assert!((v[1] - 0.2447 / 2.0).abs() < 1e-3);
    assert!((v[2] + 0.3348 / 2.0).abs() < 1e-3);
    // Sample 1 (target=0)
    assert!((v[3] + 0.6667 / 2.0).abs() < 1e-3);
    assert!((v[4] - 0.3333 / 2.0).abs() < 1e-3);
    assert!((v[5] - 0.3333 / 2.0).abs() < 1e-3);
}

// -------------------- mse_loss --------------------

#[test]
fn mse_backward_2_diff_over_n() {
    // d/dx mse(x, y, mean) = 2(x - y) / n
    let x = Variable::leaf(Tensor::from_vec([4], vec![3.0_f32, 4.0, 5.0, 6.0]).unwrap());
    let y = Variable::new(Tensor::from_vec([4], vec![1.0_f32, 1.0, 1.0, 1.0]).unwrap());
    let l = mse_loss(&x, &y, Reduction::Mean).unwrap();
    backward(&l, None).unwrap();
    let g = x.grad().unwrap();
    // diff = [2, 3, 4, 5]; 2*diff/n = 2*[2,3,4,5]/4 = [1, 1.5, 2, 2.5]
    assert_eq!(g.as_slice::<f32>().unwrap(), &[1.0_f32, 1.5, 2.0, 2.5]);
}

// -------------------- finite-difference gradcheck --------------------

#[test]
fn gradcheck_x_squared_sum() {
    // Numerical d sum(x²)/dx_i ≈ 2*x_i; analytical 2x.
    let xs = vec![1.0_f32, -0.5, 0.7, 2.5];
    let h = 1e-3_f32;
    for (i, &xi) in xs.iter().enumerate() {
        let mut x_plus = xs.clone();
        let mut x_minus = xs.clone();
        x_plus[i] = xi + h;
        x_minus[i] = xi - h;
        let f_plus: f32 = x_plus.iter().map(|v| v * v).sum();
        let f_minus: f32 = x_minus.iter().map(|v| v * v).sum();
        let numerical = (f_plus - f_minus) / (2.0 * h);
        // Analytical
        let var = Variable::leaf(Tensor::from_vec([4], xs.clone()).unwrap());
        let s = sum(&mul(&var, &var).unwrap()).unwrap();
        backward(&s, None).unwrap();
        let analytical = var.grad().unwrap().as_slice::<f32>().unwrap()[i];
        assert!(
            (numerical - analytical).abs() < 1e-2,
            "gradcheck mismatch at i={}: num={}, ana={}",
            i,
            numerical,
            analytical
        );
    }
}

// -------------------- MNIST-like end-to-end micro-test --------------------

#[test]
fn mlp_one_step_descends_loss() {
    // Tiny 2-layer MLP: 4 → 3 → 2 with ReLU, classify with cross_entropy.
    // After one SGD step, loss must strictly decrease.
    let lr = 0.01_f32;

    // Init weights (small random-ish values).
    let w1 = Variable::leaf(
        Tensor::from_vec(
            [4usize, 3],
            vec![
                0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.9, -1.0, 1.1, -1.2,
            ],
        )
        .unwrap(),
    );
    let w2 = Variable::leaf(
        Tensor::from_vec([3usize, 2], vec![0.1, -0.1, 0.2, -0.2, 0.3, -0.3]).unwrap(),
    );

    let x = Variable::new(Tensor::from_vec([1usize, 4], vec![0.5, -0.5, 0.25, -0.25]).unwrap());
    let target = Variable::new(Tensor::from_vec_typed::<i64, _>([1usize], vec![1_i64]).unwrap());

    // Forward + initial loss.
    let h1 = relu(&matmul(&x, &w1).unwrap()).unwrap();
    let logits = matmul(&h1, &w2).unwrap();
    let loss = cross_entropy(&logits, &target, Reduction::Mean).unwrap();
    let initial_loss = loss.tensor().as_slice::<f32>().unwrap()[0];

    // Backward.
    backward(&loss, None).unwrap();

    // SGD step on w1, w2 (manual: w -= lr * w.grad).
    let g1 = w1.grad().unwrap();
    let g2 = w2.grad().unwrap();
    let w1_t = w1.tensor();
    let w1_data: Vec<f32> = w1_t
        .as_slice::<f32>()
        .unwrap()
        .iter()
        .zip(g1.as_slice::<f32>().unwrap())
        .map(|(w, g)| w - lr * g)
        .collect();
    let w2_t = w2.tensor();
    let w2_data: Vec<f32> = w2_t
        .as_slice::<f32>()
        .unwrap()
        .iter()
        .zip(g2.as_slice::<f32>().unwrap())
        .map(|(w, g)| w - lr * g)
        .collect();
    let w1_new = Variable::leaf(Tensor::from_vec([4usize, 3], w1_data).unwrap());
    let w2_new = Variable::leaf(Tensor::from_vec([3usize, 2], w2_data).unwrap());

    // Forward + new loss.
    let h1n = relu(&matmul(&x, &w1_new).unwrap()).unwrap();
    let logits_n = matmul(&h1n, &w2_new).unwrap();
    let loss_n = cross_entropy(&logits_n, &target, Reduction::Mean).unwrap();
    let new_loss = loss_n.tensor().as_slice::<f32>().unwrap()[0];

    assert!(
        new_loss < initial_loss,
        "after one SGD step, loss should decrease: {} -> {}",
        initial_loss,
        new_loss
    );
}

// -------------------- new activations: silu, leaky_relu, softmax, log_softmax --------------------

#[test]
fn silu_backward_at_zero_is_half() {
    // silu(0) = 0; d/dx silu(x) at x=0 = sigmoid(0) * (1 + 0*(1-sigmoid(0))) = 0.5
    let x = Variable::leaf(Tensor::from_vec([1usize], vec![0.0_f32]).unwrap());
    let y = silu(&x).unwrap();
    backward(&y, None).unwrap();
    let dx = x.grad().unwrap();
    let v = dx.as_slice::<f32>().unwrap()[0];
    assert!((v - 0.5).abs() < 1e-5, "silu'(0) = 0.5, got {v}");
}

#[test]
fn silu_finite_difference() {
    // d/dx silu sum at x = [1, -1] checked vs central finite difference.
    let x_v = vec![1.0_f32, -1.0];
    let x = Variable::leaf(Tensor::from_vec([2usize], x_v.clone()).unwrap());
    let s = sum(&silu(&x).unwrap()).unwrap();
    backward(&s, None).unwrap();
    let dx = x.grad().unwrap();
    let dx_v = dx.as_slice::<f32>().unwrap();
    let h = 1e-3_f32;
    for i in 0..2 {
        let mut xp = x_v.clone();
        let mut xm = x_v.clone();
        xp[i] += h;
        xm[i] -= h;
        let xp_v = Variable::new(Tensor::from_vec([2usize], xp).unwrap());
        let xm_v = Variable::new(Tensor::from_vec([2usize], xm).unwrap());
        let sp: f32 = silu(&xp_v)
            .unwrap()
            .tensor()
            .as_slice::<f32>()
            .unwrap()
            .iter()
            .sum();
        let sm: f32 = silu(&xm_v)
            .unwrap()
            .tensor()
            .as_slice::<f32>()
            .unwrap()
            .iter()
            .sum();
        let fd = (sp - sm) / (2.0 * h);
        assert!(
            (dx_v[i] - fd).abs() < 1e-2,
            "silu'@{} = analytic {}, finite-diff {}",
            x_v[i],
            dx_v[i],
            fd
        );
    }
}

#[test]
fn leaky_relu_backward_mask() {
    // d/dx leaky_relu(x, 0.1) at x = [1, -2, 0.5, -0.1] = [1, 0.1, 1, 0.1]
    let x = Variable::leaf(Tensor::from_vec([4usize], vec![1.0_f32, -2.0, 0.5, -0.1]).unwrap());
    let s = sum(&leaky_relu(&x, 0.1).unwrap()).unwrap();
    backward(&s, None).unwrap();
    let dx = x.grad().unwrap();
    let dx_v = dx.as_slice::<f32>().unwrap();
    for (got, expected) in dx_v.iter().zip(&[1.0_f32, 0.1, 1.0, 0.1]) {
        assert!(
            (got - expected).abs() < 1e-6,
            "got {got}, expected {expected}"
        );
    }
}

#[test]
fn softmax_finite_difference_dim_minus_one() {
    // d/dx softmax(x).sum() = 0 (softmax sums to 1 → derivative of constant
    // sum is zero). Strictly: J^T 1 = y - y = 0 because y * (1 - sum(y)) = 0.
    let x_v = vec![1.0_f32, 2.0, 3.0];
    let x = Variable::leaf(Tensor::from_vec([3usize], x_v).unwrap());
    let s = sum(&softmax(&x, 0).unwrap()).unwrap();
    backward(&s, None).unwrap();
    let dx = x.grad().unwrap();
    let dx_v = dx.as_slice::<f32>().unwrap();
    for &v in dx_v {
        assert!(v.abs() < 1e-5, "softmax sum gradient ≈ 0, got {v}");
    }
}

#[test]
fn log_softmax_finite_difference() {
    // log_softmax sum gradient: d/dx_i (sum_j log_softmax_j) = N/N - softmax_i*N (when summed)
    // For dim=0 with N classes: dx = ones - softmax * N
    let x_v = vec![0.0_f32, 1.0, 2.0];
    let x = Variable::leaf(Tensor::from_vec([3usize], x_v.clone()).unwrap());
    let s = sum(&log_softmax(&x, 0).unwrap()).unwrap();
    backward(&s, None).unwrap();
    let dx = x.grad().unwrap();
    let dx_v = dx.as_slice::<f32>().unwrap();

    // Reference via finite difference
    let h = 1e-3_f32;
    for i in 0..3 {
        let mut xp = x_v.clone();
        let mut xm = x_v.clone();
        xp[i] += h;
        xm[i] -= h;
        let xp_v = Variable::new(Tensor::from_vec([3usize], xp).unwrap());
        let xm_v = Variable::new(Tensor::from_vec([3usize], xm).unwrap());
        let sp: f32 = log_softmax(&xp_v, 0)
            .unwrap()
            .tensor()
            .as_slice::<f32>()
            .unwrap()
            .iter()
            .sum();
        let sm: f32 = log_softmax(&xm_v, 0)
            .unwrap()
            .tensor()
            .as_slice::<f32>()
            .unwrap()
            .iter()
            .sum();
        let fd = (sp - sm) / (2.0 * h);
        assert!(
            (dx_v[i] - fd).abs() < 1e-2,
            "log_softmax_sum'@{} = analytic {}, fd {}",
            i,
            dx_v[i],
            fd
        );
    }
}

// -------------------- math ops: div, exp, log, sqrt, abs, pow_scalar --------------------

#[test]
fn div_backward_quotient_rule() {
    // y = x / k (k constant) → dy/dx = 1/k
    let x = Variable::leaf(Tensor::from_vec([3], vec![3.0_f32, 6.0, 9.0]).unwrap());
    let k = Variable::new(Tensor::from_vec([3], vec![3.0_f32, 3.0, 3.0]).unwrap());
    let s = sum(&div(&x, &k).unwrap()).unwrap();
    backward(&s, None).unwrap();
    let dx = x.grad().unwrap();
    let dx_v = dx.as_slice::<f32>().unwrap();
    for &v in dx_v {
        assert!((v - 1.0 / 3.0).abs() < 1e-5);
    }
}

#[test]
fn exp_backward_equals_output() {
    // d/dx exp(x).sum() = exp(x); sum kicks in 1·exp grad on each lane
    let x = Variable::leaf(Tensor::from_vec([3], vec![0.0_f32, 1.0, 2.0]).unwrap());
    let s = sum(&exp(&x).unwrap()).unwrap();
    backward(&s, None).unwrap();
    let dx = x.grad().unwrap();
    let dx_v = dx.as_slice::<f32>().unwrap();
    let expected = [1.0_f32, std::f32::consts::E, std::f32::consts::E.powi(2)];
    for (got, e) in dx_v.iter().zip(expected.iter()) {
        assert!((got - e).abs() < 1e-3, "exp'@x = {got}, expected {e}");
    }
}

#[test]
fn log_backward_is_reciprocal() {
    let x = Variable::leaf(Tensor::from_vec([3], vec![1.0_f32, 2.0, 4.0]).unwrap());
    let s = sum(&log(&x).unwrap()).unwrap();
    backward(&s, None).unwrap();
    let dx = x.grad().unwrap();
    let dx_v = dx.as_slice::<f32>().unwrap();
    for (got, x_v) in dx_v.iter().zip(&[1.0_f32, 2.0, 4.0]) {
        let expected = 1.0 / x_v;
        assert!(
            (got - expected).abs() < 1e-5,
            "log'@{x_v} = {got}, expected {expected}"
        );
    }
}

#[test]
fn sqrt_backward_half_over_sqrt_x() {
    // d/dx sqrt(x).sum() = 1/(2*sqrt(x))
    let x = Variable::leaf(Tensor::from_vec([3], vec![1.0_f32, 4.0, 9.0]).unwrap());
    let s = sum(&sqrt(&x).unwrap()).unwrap();
    backward(&s, None).unwrap();
    let dx = x.grad().unwrap();
    let dx_v = dx.as_slice::<f32>().unwrap();
    for (got, x_v) in dx_v.iter().zip(&[1.0_f32, 4.0, 9.0]) {
        let expected = 0.5_f32 / x_v.sqrt();
        assert!(
            (got - expected).abs() < 1e-5,
            "sqrt'@{x_v} = {got}, expected {expected}"
        );
    }
}

#[test]
fn abs_backward_sign_function() {
    let x = Variable::leaf(Tensor::from_vec([4], vec![3.0_f32, -2.0, 0.0, 5.0]).unwrap());
    let s = sum(&abs(&x).unwrap()).unwrap();
    backward(&s, None).unwrap();
    let dx = x.grad().unwrap();
    assert_eq!(dx.as_slice::<f32>().unwrap(), &[1.0_f32, -1.0, 0.0, 1.0]);
}

#[test]
fn pow_scalar_backward_x_to_3() {
    // y = x^3 → dy/dx = 3*x²
    let x = Variable::leaf(Tensor::from_vec([3], vec![1.0_f32, 2.0, 3.0]).unwrap());
    let s = sum(&pow_scalar(&x, 3.0).unwrap()).unwrap();
    backward(&s, None).unwrap();
    let dx = x.grad().unwrap();
    let dx_v = dx.as_slice::<f32>().unwrap();
    for (got, x_v) in dx_v.iter().zip(&[1.0_f32, 2.0, 3.0]) {
        let expected = 3.0 * x_v * x_v;
        assert!(
            (got - expected).abs() < 1e-4,
            "x³'@{x_v} = {got}, expected {expected}"
        );
    }
}

// -------------------- bmm (batched matmul) --------------------

#[test]
fn bmm_forward_matches_per_batch_matmul() {
    // B=2, M=2, K=3, N=2
    let a = Variable::new(
        Tensor::from_vec(
            [2usize, 2, 3],
            vec![
                1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
        )
        .unwrap(),
    );
    let b = Variable::new(
        Tensor::from_vec(
            [2usize, 3, 2],
            vec![
                1.0_f32, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0,
            ],
        )
        .unwrap(),
    );
    let c = bmm(&a, &b).unwrap();
    assert_eq!(c.tensor().shape(), &[2, 2, 2]);
    // batch 0: A[0]= [[1,2,3],[4,5,6]] @ [[1,0],[0,1],[1,0]] = [[1+3, 2], [4+6, 5]] = [[4,2],[10,5]]
    // batch 1: A[1]= [[7,8,9],[10,11,12]] @ [[0,1],[1,0],[0,1]] = [[8, 7+9],[11, 10+12]] = [[8,16],[11,22]]
    let expected = vec![4.0_f32, 2.0, 10.0, 5.0, 8.0, 16.0, 11.0, 22.0];
    assert_eq!(c.tensor().as_slice::<f32>().unwrap(), expected.as_slice());
}

#[test]
fn bmm_backward_propagates_to_both_inputs() {
    // Simple bmm with sum loss
    let a = Variable::leaf(Tensor::from_vec([1usize, 2, 2], vec![1.0_f32, 2.0, 3.0, 4.0]).unwrap());
    let b = Variable::leaf(Tensor::from_vec([1usize, 2, 2], vec![1.0_f32, 2.0, 3.0, 4.0]).unwrap());
    let c = bmm(&a, &b).unwrap();
    let s = sum(&c).unwrap();
    backward(&s, None).unwrap();
    // d/dA sum(A @ B) = grad @ B.T = ones[2,2] @ [[1,3],[2,4]] = [[3,7],[3,7]]
    let da = a.grad().unwrap();
    assert_eq!(da.shape(), &[1, 2, 2]);
    assert_eq!(da.as_slice::<f32>().unwrap(), &[3.0_f32, 7.0, 3.0, 7.0]);
    // d/dB sum(A @ B) = A.T @ grad = [[1,3],[2,4]] @ ones[2,2] = [[4,4],[6,6]]
    let db = b.grad().unwrap();
    assert_eq!(db.shape(), &[1, 2, 2]);
    assert_eq!(db.as_slice::<f32>().unwrap(), &[4.0_f32, 4.0, 6.0, 6.0]);
}
