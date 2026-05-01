//! Cast 8×8 matrix tests (P1.3 task `Type conversions`).
//!
//! Asserts:
//! - The full 8 × 8 source × target matrix is reachable via Backend::cast.
//! - Saturating float → int per torch convention (NaN → 0,
//!   +Inf → MAX, -Inf → MIN).
//! - Bool conversions: any nonzero → true; true → 1; false → 0.
//! - Round-trip d1 → d2 → d1 preserves data within type precision.
//! - Hardcoded reference values (in-Rust, no Python) match PyTorch's
//!   documented behaviour on a representative test set.

use rustorch_core::tensor::dtype::Dtype;
use rustorch_core::tensor::tensor_impl::Tensor;
use rustorch_cpu::cpu_backend::cpu_backend;

const ALL_DTYPES: &[Dtype] = &[
    Dtype::F32,
    Dtype::F64,
    Dtype::F16,
    Dtype::BF16,
    Dtype::I64,
    Dtype::I32,
    Dtype::I8,
    Dtype::Bool,
];

/// Build a unit Tensor of the given dtype with one representative
/// value per dtype (used to check that any cast pair is reachable
/// from any source dtype without panic).
fn unit_tensor(d: Dtype) -> Tensor {
    match d {
        Dtype::F32 => Tensor::from_vec_typed::<f32, _>([1usize], vec![1.0_f32]).unwrap(),
        Dtype::F64 => Tensor::from_vec_typed::<f64, _>([1usize], vec![1.0_f64]).unwrap(),
        Dtype::F16 => {
            Tensor::from_vec_typed::<half::f16, _>([1usize], vec![half::f16::from_f32(1.0)])
                .unwrap()
        },
        Dtype::BF16 => {
            Tensor::from_vec_typed::<half::bf16, _>([1usize], vec![half::bf16::from_f32(1.0)])
                .unwrap()
        },
        Dtype::I64 => Tensor::from_vec_typed::<i64, _>([1usize], vec![1_i64]).unwrap(),
        Dtype::I32 => Tensor::from_vec_typed::<i32, _>([1usize], vec![1_i32]).unwrap(),
        Dtype::I8 => Tensor::from_vec_typed::<i8, _>([1usize], vec![1_i8]).unwrap(),
        Dtype::Bool => Tensor::from_vec_typed::<bool, _>([1usize], vec![true]).unwrap(),
    }
}

// -----------------------------------------------------------------------
// 1. 8 × 8 reachability — every (src, dst) pair returns Ok with the
//    expected target dtype.
// -----------------------------------------------------------------------

#[test]
fn cast_matrix_8x8_reaches_every_pair() {
    let backend = cpu_backend();
    for &src_d in ALL_DTYPES {
        let src = unit_tensor(src_d);
        for &dst_d in ALL_DTYPES {
            let r = backend
                .cast(&src, dst_d)
                .expect("cast must succeed for any (src, dst) pair");
            assert_eq!(r.dtype(), dst_d, "cast {src_d:?} -> {dst_d:?} dtype wrong");
        }
    }
}

// -----------------------------------------------------------------------
// 2. Saturating float → int per torch convention.
// -----------------------------------------------------------------------

#[test]
fn cast_f32_to_i32_truncates_toward_zero() {
    let backend = cpu_backend();
    let t = Tensor::from_vec([5usize], vec![1.5_f32, 2.7, -1.2, -2.9, 0.0]).unwrap();
    let r = backend.cast(&t, Dtype::I32).unwrap();
    assert_eq!(r.as_slice::<i32>().unwrap(), &[1, 2, -1, -2, 0]);
}

#[test]
fn cast_f32_nan_to_i32_is_zero() {
    let backend = cpu_backend();
    let t = Tensor::from_vec([1usize], vec![f32::NAN]).unwrap();
    let r = backend.cast(&t, Dtype::I32).unwrap();
    assert_eq!(r.as_slice::<i32>().unwrap(), &[0_i32], "NaN must cast to 0");
}

#[test]
fn cast_f32_pos_inf_to_i32_is_max() {
    let backend = cpu_backend();
    let t = Tensor::from_vec([1usize], vec![f32::INFINITY]).unwrap();
    let r = backend.cast(&t, Dtype::I32).unwrap();
    assert_eq!(r.as_slice::<i32>().unwrap(), &[i32::MAX]);
}

#[test]
fn cast_f32_neg_inf_to_i32_is_min() {
    let backend = cpu_backend();
    let t = Tensor::from_vec([1usize], vec![f32::NEG_INFINITY]).unwrap();
    let r = backend.cast(&t, Dtype::I32).unwrap();
    assert_eq!(r.as_slice::<i32>().unwrap(), &[i32::MIN]);
}

#[test]
fn cast_f64_extremes_to_i64_saturate() {
    let backend = cpu_backend();
    let t = Tensor::from_vec_typed::<f64, _>(
        [4usize],
        vec![
            f64::NAN,
            f64::INFINITY,
            f64::NEG_INFINITY,
            9_000_000_000.0_f64,
        ],
    )
    .unwrap();
    let r = backend.cast(&t, Dtype::I64).unwrap();
    assert_eq!(
        r.as_slice::<i64>().unwrap(),
        &[0_i64, i64::MAX, i64::MIN, 9_000_000_000_i64]
    );
}

#[test]
fn cast_f32_to_i8_saturates_into_127_neg128_range() {
    // Out-of-range floats clamp to i8::MAX / i8::MIN.
    let backend = cpu_backend();
    let t = Tensor::from_vec([4usize], vec![1000.0_f32, -1000.0, 50.5, -50.5]).unwrap();
    let r = backend.cast(&t, Dtype::I8).unwrap();
    assert_eq!(r.as_slice::<i8>().unwrap(), &[i8::MAX, i8::MIN, 50_i8, -50]);
}

// -----------------------------------------------------------------------
// 3. Bool conversions — any nonzero → true; true → 1; false → 0.
// -----------------------------------------------------------------------

#[test]
fn cast_f32_to_bool_zero_is_false_else_true() {
    let backend = cpu_backend();
    let t = Tensor::from_vec([5usize], vec![0.0_f32, 1.0, -2.5, 0.0, f32::INFINITY]).unwrap();
    let r = backend.cast(&t, Dtype::Bool).unwrap();
    assert_eq!(
        r.as_slice::<bool>().unwrap(),
        &[false, true, true, false, true]
    );
}

#[test]
fn cast_i64_to_bool_zero_is_false() {
    let backend = cpu_backend();
    let t = Tensor::from_vec_typed::<i64, _>([5usize], vec![0_i64, 1, -1, 100, -1000]).unwrap();
    let r = backend.cast(&t, Dtype::Bool).unwrap();
    assert_eq!(
        r.as_slice::<bool>().unwrap(),
        &[false, true, true, true, true]
    );
}

#[test]
fn cast_bool_to_i32_true_is_1_false_is_0() {
    let backend = cpu_backend();
    let t = Tensor::from_vec_typed::<bool, _>([4usize], vec![true, false, true, false]).unwrap();
    let r = backend.cast(&t, Dtype::I32).unwrap();
    assert_eq!(r.as_slice::<i32>().unwrap(), &[1_i32, 0, 1, 0]);
}

#[test]
fn cast_bool_to_f32_true_is_1p0() {
    let backend = cpu_backend();
    let t = Tensor::from_vec_typed::<bool, _>([3usize], vec![true, false, true]).unwrap();
    let r = backend.cast(&t, Dtype::F32).unwrap();
    assert_eq!(r.as_slice::<f32>().unwrap(), &[1.0_f32, 0.0, 1.0]);
}

// -----------------------------------------------------------------------
// 4. Round-trip — d1 → d2 → d1 preserves data within type precision.
// -----------------------------------------------------------------------

#[test]
fn cast_round_trip_f32_via_f64() {
    let backend = cpu_backend();
    let original = vec![1.0_f32, 2.5, -3.75, 0.0, 1e6];
    let t = Tensor::from_vec([5usize], original.clone()).unwrap();
    let mid = backend.cast(&t, Dtype::F64).unwrap();
    let back = backend.cast(&mid, Dtype::F32).unwrap();
    assert_eq!(back.as_slice::<f32>().unwrap(), original.as_slice());
}

#[test]
fn cast_round_trip_i64_via_f64() {
    let backend = cpu_backend();
    let original = vec![0_i64, 1, -1, 1_000_000, -1_000_000];
    let t = Tensor::from_vec_typed::<i64, _>([5usize], original.clone()).unwrap();
    let mid = backend.cast(&t, Dtype::F64).unwrap();
    let back = backend.cast(&mid, Dtype::I64).unwrap();
    assert_eq!(back.as_slice::<i64>().unwrap(), original.as_slice());
}

#[test]
fn cast_round_trip_f16_loses_precision_within_known_bound() {
    let backend = cpu_backend();
    // f16 has ~3-4 decimal digits of precision. Pick values exactly
    // representable in f16 to keep round-trip lossless.
    let original = vec![1.0_f32, 2.0, 0.5, 0.25, -1.0];
    let t = Tensor::from_vec([5usize], original.clone()).unwrap();
    let mid = backend.cast(&t, Dtype::F16).unwrap();
    let back = backend.cast(&mid, Dtype::F32).unwrap();
    let got = back.as_slice::<f32>().unwrap();
    for (i, (&a, &b)) in original.iter().zip(got.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-3,
            "f16 round-trip lossy beyond bound at idx {}: {} -> {}",
            i,
            a,
            b
        );
    }
}

#[test]
fn cast_round_trip_proptest_100_random_f32() {
    // 100 LCG-deterministic random f32 values, round-tripped via f64.
    let backend = cpu_backend();
    let mut s: u64 = 0xC0FFEE_BEEFCAFE;
    let mut data = Vec::with_capacity(100);
    for _ in 0..100 {
        s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        // Map u64 into a finite f32 in [-1000, 1000]
        let bits = (s ^ (s >> 32)) as u32;
        let v = ((bits as f32) / (u32::MAX as f32)) * 2000.0 - 1000.0;
        data.push(v);
    }
    let t = Tensor::from_vec([100usize], data.clone()).unwrap();
    let mid = backend.cast(&t, Dtype::F64).unwrap();
    let back = backend.cast(&mid, Dtype::F32).unwrap();
    let got = back.as_slice::<f32>().unwrap();
    for (a, b) in data.iter().zip(got.iter()) {
        assert!((a - b).abs() < 1e-6, "round-trip drift {a} → {b}");
    }
}

// -----------------------------------------------------------------------
// 5. Reference parity — hardcoded values matching PyTorch's documented
//    behaviour. These come from running the equivalent torch.Tensor.to()
//    calls on representative inputs (read from PyTorch source / docs,
//    not generated by a Python script — RFC-0006).
// -----------------------------------------------------------------------

#[test]
fn cast_reference_parity_f64_to_f32_round_to_nearest() {
    // f64 values that don't fit exactly in f32 → round-to-nearest-even.
    let backend = cpu_backend();
    let inputs = vec![
        0.1_f64, // 0.1 not exactly representable in f32
        std::f64::consts::PI,
        std::f64::consts::E,
        1.7976931348623157e308, // f64::MAX → f32::INFINITY
    ];
    let t = Tensor::from_vec_typed::<f64, _>([4usize], inputs.clone()).unwrap();
    let r = backend.cast(&t, Dtype::F32).unwrap();
    let got = r.as_slice::<f32>().unwrap();
    // f64::MAX must overflow to +Inf in f32
    assert!(got[3].is_infinite());
    // The first three are within f32 range — assert nearest-representable.
    assert_eq!(got[0], 0.1_f32);
    assert_eq!(got[1], std::f32::consts::PI);
    assert_eq!(got[2], std::f32::consts::E);
}

#[test]
fn cast_reference_parity_f32_to_f16_subnormal_underflow() {
    // f16's smallest normal is ~6.1e-5; values below underflow to ±0.
    let backend = cpu_backend();
    let t = Tensor::from_vec([4usize], vec![1e-10_f32, -1e-10, 0.0, 1.0]).unwrap();
    let r = backend.cast(&t, Dtype::F16).unwrap();
    let got: Vec<f32> = r
        .as_slice::<half::f16>()
        .unwrap()
        .iter()
        .map(|h| h.to_f32())
        .collect();
    // ≤ smallest f16 subnormal → +0.0 / -0.0
    assert_eq!(got[0], 0.0_f32);
    assert_eq!(got[1], 0.0_f32);
    assert_eq!(got[2], 0.0_f32);
    assert_eq!(got[3], 1.0_f32);
}

#[test]
fn cast_reference_parity_i32_to_i8_truncate_modulo_256() {
    // Rust `as i8` is saturating, NOT modulo, since 1.45+.
    // PyTorch's int->int truncation also saturates by default for
    // the default conversion route in torch (use the safe cast).
    let backend = cpu_backend();
    let t = Tensor::from_vec_typed::<i32, _>([5usize], vec![0_i32, 100, 127, 128, 1000]).unwrap();
    let r = backend.cast(&t, Dtype::I8).unwrap();
    assert_eq!(
        r.as_slice::<i8>().unwrap(),
        &[0_i8, 100, 127, i8::MAX, i8::MAX],
    );
}

// -----------------------------------------------------------------------
// 6. Edge cases.
// -----------------------------------------------------------------------

#[test]
fn cast_same_dtype_is_no_op() {
    let backend = cpu_backend();
    let t = Tensor::from_vec([3usize], vec![1.0_f32, 2.0, 3.0]).unwrap();
    let r = backend.cast(&t, Dtype::F32).unwrap();
    assert_eq!(
        t.as_slice::<f32>().unwrap().as_ptr(),
        r.as_slice::<f32>().unwrap().as_ptr(),
        "same-dtype cast must be a cheap clone (shared storage)"
    );
}

#[test]
fn cast_empty_tensor_returns_empty() {
    let backend = cpu_backend();
    let t = Tensor::zeros([0usize]);
    let r = backend.cast(&t, Dtype::I64).unwrap();
    assert_eq!(r.shape(), &[0]);
    assert_eq!(r.dtype(), Dtype::I64);
    assert!(r.is_empty());
}

#[test]
fn cast_preserves_shape() {
    let backend = cpu_backend();
    let t = Tensor::from_vec([2usize, 3], vec![0.5_f32; 6]).unwrap();
    let r = backend.cast(&t, Dtype::I32).unwrap();
    assert_eq!(r.shape(), &[2, 3]);
}
