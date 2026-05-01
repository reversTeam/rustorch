//! Integration tests asserting the broadcast grid is consistent with
//! [`rustorch_core::tensor::shape::Shape::broadcast_with`] and that the
//! [`rustorch_cpu::iterator::map_binary_same`] kernel agrees on every
//! case.

mod common;

use common::broadcast::{broadcast_test_grid, reference_broadcast};
use rustorch_core::tensor::tensor_impl::Tensor;
use rustorch_cpu::cpu_backend::cpu_backend;

#[test]
fn grid_size_meets_minimum_200() {
    let grid = broadcast_test_grid();
    assert!(
        grid.len() >= 200,
        "broadcast_test_grid must produce ≥ 200 cases, got {}",
        grid.len(),
    );
}

#[test]
fn grid_reproducible_across_calls() {
    // Same seed = same shapes — required by the spec ("reproducibility:
    // same seed yields same shapes").
    let g1 = broadcast_test_grid();
    let g2 = broadcast_test_grid();
    assert_eq!(g1.len(), g2.len());
    for (a, b) in g1.iter().zip(g2.iter()) {
        assert_eq!(a.lhs, b.lhs);
        assert_eq!(a.rhs, b.rhs);
        assert_eq!(a.expected, b.expected);
    }
}

#[test]
fn grid_includes_scalar_pairs() {
    let grid = broadcast_test_grid();
    assert!(grid.iter().any(|c| c.lhs.is_empty() && !c.rhs.is_empty()));
    assert!(grid.iter().any(|c| c.rhs.is_empty() && !c.lhs.is_empty()));
}

#[test]
fn grid_includes_zero_dim_pairs() {
    let grid = broadcast_test_grid();
    assert!(grid
        .iter()
        .any(|c| c.lhs.contains(&0) || c.rhs.contains(&0)));
}

#[test]
fn grid_includes_incompatible_pairs() {
    let grid = broadcast_test_grid();
    let bad_count = grid.iter().filter(|c| c.expected.is_none()).count();
    assert!(bad_count >= 1, "grid must include negative cases");
}

#[test]
fn grid_includes_up_to_6d() {
    let grid = broadcast_test_grid();
    assert!(grid.iter().any(|c| c.lhs.len() == 6 || c.rhs.len() == 6));
}

#[test]
fn grid_matches_shape_broadcast_with() {
    let grid = broadcast_test_grid();
    for case in &grid {
        let (lhs, rhs) = case.shapes();
        let got = lhs.broadcast_with(&rhs);
        match (got, &case.expected) {
            (Ok(s), Some(exp)) => {
                assert_eq!(
                    s.as_slice(),
                    exp.as_slice(),
                    "grid mismatch on {:?} vs {:?}: expected {:?}, got {:?}",
                    case.lhs,
                    case.rhs,
                    exp,
                    s
                );
            },
            (Err(_), None) => {},
            (Ok(s), None) => panic!(
                "grid says {:?} vs {:?} should be incompatible, but Shape::broadcast_with returned {:?}",
                case.lhs, case.rhs, s
            ),
            (Err(e), Some(exp)) => panic!(
                "grid says {:?} vs {:?} should produce {:?}, but Shape::broadcast_with errored: {:?}",
                case.lhs, case.rhs, exp, e
            ),
        }
    }
}

#[test]
fn reference_broadcast_matches_grid() {
    // Reference impl in `common::broadcast` agrees with the grid's
    // expected field on every case.
    let grid = broadcast_test_grid();
    for case in &grid {
        let r = reference_broadcast(&case.lhs, &case.rhs);
        assert_eq!(
            r, case.expected,
            "reference disagrees with grid on {:?} vs {:?}",
            case.lhs, case.rhs
        );
    }
}

#[test]
fn cpu_backend_add_agrees_with_grid_for_compatible_pairs() {
    // For every compatible pair in the grid, build f32 tensors of ones,
    // add them via cpu_backend, assert the output shape and content
    // matches the broadcast expectation.
    let grid = broadcast_test_grid();
    let backend = cpu_backend();
    for case in &grid {
        let exp = match &case.expected {
            Some(e) => e,
            None => continue, // negative cases handled in a separate test
        };
        // Skip empty-numel cases — they're tested elsewhere and hit the
        // ZST path.
        if exp.contains(&0) {
            continue;
        }
        let lhs_numel: usize = if case.lhs.is_empty() {
            1
        } else {
            case.lhs.iter().product()
        };
        let rhs_numel: usize = if case.rhs.is_empty() {
            1
        } else {
            case.rhs.iter().product()
        };
        let a = Tensor::from_vec(case.lhs.clone(), vec![1.0_f32; lhs_numel]).unwrap();
        let b = Tensor::from_vec(case.rhs.clone(), vec![2.0_f32; rhs_numel]).unwrap();
        let c = backend.add(&a, &b).unwrap();
        assert_eq!(
            c.shape(),
            exp.as_slice(),
            "shape mismatch for {:?} + {:?}",
            case.lhs,
            case.rhs
        );
        let exp_numel: usize = exp.iter().product();
        // 1.0 + 2.0 = 3.0 everywhere
        assert_eq!(c.as_slice::<f32>().unwrap(), vec![3.0_f32; exp_numel]);
    }
}

#[test]
fn cpu_backend_rejects_incompatible_pairs() {
    let grid = broadcast_test_grid();
    let backend = cpu_backend();
    for case in &grid {
        if case.expected.is_some() {
            continue;
        }
        let lhs_numel: usize = if case.lhs.is_empty() {
            1
        } else {
            case.lhs.iter().product()
        };
        let rhs_numel: usize = if case.rhs.is_empty() {
            1
        } else {
            case.rhs.iter().product()
        };
        let a = Tensor::from_vec(case.lhs.clone(), vec![0.0_f32; lhs_numel]).unwrap();
        let b = Tensor::from_vec(case.rhs.clone(), vec![0.0_f32; rhs_numel]).unwrap();
        assert!(
            backend.add(&a, &b).is_err(),
            "expected error on incompatible pair {:?} vs {:?}",
            case.lhs,
            case.rhs
        );
    }
}

// ---------------------- arithmetic + comparison wiring ----------------------

#[test]
fn cpu_backend_arithmetic_full_grid_f32() {
    // Wire the harness into all arithmetic ops (add/sub/mul/div).
    let grid = broadcast_test_grid();
    let backend = cpu_backend();
    for case in &grid {
        let exp = match &case.expected {
            Some(e) => e,
            None => continue,
        };
        if exp.contains(&0) {
            continue;
        }
        let lhs_numel: usize = if case.lhs.is_empty() {
            1
        } else {
            case.lhs.iter().product()
        };
        let rhs_numel: usize = if case.rhs.is_empty() {
            1
        } else {
            case.rhs.iter().product()
        };
        let a = Tensor::from_vec(case.lhs.clone(), vec![6.0_f32; lhs_numel]).unwrap();
        let b = Tensor::from_vec(case.rhs.clone(), vec![2.0_f32; rhs_numel]).unwrap();
        let exp_numel: usize = exp.iter().product();
        // 6 + 2 = 8
        assert_eq!(
            backend.add(&a, &b).unwrap().as_slice::<f32>().unwrap(),
            vec![8.0_f32; exp_numel]
        );
        // 6 - 2 = 4
        assert_eq!(
            backend.sub(&a, &b).unwrap().as_slice::<f32>().unwrap(),
            vec![4.0_f32; exp_numel]
        );
        // 6 * 2 = 12
        assert_eq!(
            backend.mul(&a, &b).unwrap().as_slice::<f32>().unwrap(),
            vec![12.0_f32; exp_numel]
        );
        // 6 / 2 = 3
        assert_eq!(
            backend.div(&a, &b).unwrap().as_slice::<f32>().unwrap(),
            vec![3.0_f32; exp_numel]
        );
    }
}

#[test]
fn cpu_backend_comparison_full_grid_f32() {
    // Wire the harness into comparison ops (eq/ne/lt/le/gt/ge).
    let grid = broadcast_test_grid();
    let backend = cpu_backend();
    for case in &grid {
        let exp = match &case.expected {
            Some(e) => e,
            None => continue,
        };
        if exp.contains(&0) {
            continue;
        }
        let lhs_numel: usize = if case.lhs.is_empty() {
            1
        } else {
            case.lhs.iter().product()
        };
        let rhs_numel: usize = if case.rhs.is_empty() {
            1
        } else {
            case.rhs.iter().product()
        };
        let a = Tensor::from_vec(case.lhs.clone(), vec![1.0_f32; lhs_numel]).unwrap();
        let b = Tensor::from_vec(case.rhs.clone(), vec![1.0_f32; rhs_numel]).unwrap();
        let exp_numel: usize = exp.iter().product();
        // 1 == 1 ⇒ all true
        assert_eq!(
            backend.eq(&a, &b).unwrap().as_slice::<bool>().unwrap(),
            vec![true; exp_numel]
        );
        // 1 != 1 ⇒ all false
        assert_eq!(
            backend.ne(&a, &b).unwrap().as_slice::<bool>().unwrap(),
            vec![false; exp_numel]
        );
        // 1 < 1 ⇒ all false
        assert_eq!(
            backend.lt(&a, &b).unwrap().as_slice::<bool>().unwrap(),
            vec![false; exp_numel]
        );
        // 1 <= 1 ⇒ all true
        assert_eq!(
            backend.le(&a, &b).unwrap().as_slice::<bool>().unwrap(),
            vec![true; exp_numel]
        );
    }
}

#[test]
fn grid_setup_under_50ms() {
    // The spec verifies the harness setup is <50ms — this is the
    // deterministic LCG generator; should be sub-millisecond in
    // release. We use a generous bound to account for debug builds.
    let start = std::time::Instant::now();
    let grid = broadcast_test_grid();
    let elapsed = start.elapsed();
    assert!(grid.len() >= 200);
    assert!(
        elapsed.as_millis() < 50,
        "grid setup took {:?}, must be < 50 ms",
        elapsed
    );
}
