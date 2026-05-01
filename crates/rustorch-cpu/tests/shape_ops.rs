//! Shape ops kernel tests (P1.3 task `Shape ops`).
//!
//! Covers cat / stack / split / chunk / repeat / flip / roll with the
//! spec's edge cases:
//! - empty cat list returns error
//! - NaN preserved through cat/stack
//! - shape mismatch (different non-cat dims) returns ShapeMismatch
//! - dtype mismatch in cat returns DtypeMismatch
//! - split then cat returns original tensor (round-trip property)

use rustorch_core::tensor::tensor_impl::Tensor;
use rustorch_cpu::cpu_backend::cpu_backend;
use rustorch_cpu::error::BackendError;

// ------------------------- cat -------------------------

#[test]
fn cat_two_2x3_along_dim_0_yields_4x3() {
    let backend = cpu_backend();
    let a = Tensor::from_vec([2usize, 3], vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let b = Tensor::from_vec([2usize, 3], vec![7.0_f32, 8.0, 9.0, 10.0, 11.0, 12.0]).unwrap();
    let c = backend.cat(&[&a, &b], 0).unwrap();
    assert_eq!(c.shape(), &[4, 3]);
    assert_eq!(
        c.as_slice::<f32>().unwrap(),
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
    );
}

#[test]
fn cat_along_dim_1() {
    let backend = cpu_backend();
    let a = Tensor::from_vec([2usize, 2], vec![1.0_f32, 2.0, 3.0, 4.0]).unwrap();
    let b = Tensor::from_vec([2usize, 1], vec![5.0_f32, 6.0]).unwrap();
    let c = backend.cat(&[&a, &b], 1).unwrap();
    assert_eq!(c.shape(), &[2, 3]);
    assert_eq!(
        c.as_slice::<f32>().unwrap(),
        &[1.0, 2.0, 5.0, 3.0, 4.0, 6.0]
    );
}

#[test]
fn cat_preserves_nan() {
    let backend = cpu_backend();
    let a = Tensor::from_vec([2usize], vec![f32::NAN, 1.0]).unwrap();
    let b = Tensor::from_vec([1usize], vec![2.0_f32]).unwrap();
    let c = backend.cat(&[&a, &b], 0).unwrap();
    let v = c.as_slice::<f32>().unwrap();
    assert!(v[0].is_nan());
    assert_eq!(v[1], 1.0);
    assert_eq!(v[2], 2.0);
}

#[test]
fn cat_empty_list_is_err() {
    let backend = cpu_backend();
    assert!(matches!(
        backend.cat(&[], 0),
        Err(BackendError::ShapeMismatch { .. })
    ));
}

#[test]
fn cat_shape_mismatch_off_dim_returns_err() {
    let backend = cpu_backend();
    let a = Tensor::from_vec([2usize, 3], vec![1.0_f32; 6]).unwrap();
    let b = Tensor::from_vec([3usize, 4], vec![1.0_f32; 12]).unwrap();
    assert!(matches!(
        backend.cat(&[&a, &b], 0),
        Err(BackendError::ShapeMismatch { .. })
    ));
}

#[test]
fn cat_dtype_mismatch_returns_err() {
    let backend = cpu_backend();
    let a = Tensor::from_vec([2usize], vec![1.0_f32, 2.0]).unwrap();
    let b = Tensor::from_vec_typed::<i64, _>([2usize], vec![1_i64, 2]).unwrap();
    assert!(matches!(
        backend.cat(&[&a, &b], 0),
        Err(BackendError::DtypeMismatch { .. })
    ));
}

#[test]
fn cat_dim_out_of_range_returns_err() {
    let backend = cpu_backend();
    let a = Tensor::from_vec([3usize], vec![1.0_f32; 3]).unwrap();
    assert!(matches!(
        backend.cat(&[&a, &a], 5),
        Err(BackendError::IndexOutOfBounds { .. })
    ));
}

// ------------------------- stack -------------------------

#[test]
fn stack_three_along_dim_1_introduces_new_dim() {
    let backend = cpu_backend();
    let a = Tensor::from_vec([2usize], vec![1.0_f32, 2.0]).unwrap();
    let b = Tensor::from_vec([2usize], vec![3.0_f32, 4.0]).unwrap();
    let c = Tensor::from_vec([2usize], vec![5.0_f32, 6.0]).unwrap();
    let s = backend.stack(&[&a, &b, &c], 1).unwrap();
    assert_eq!(s.shape(), &[2, 3]);
    // stack along dim 1: result[i, j] = inputs[j][i]
    // → [[1,3,5], [2,4,6]]
    assert_eq!(
        s.as_slice::<f32>().unwrap(),
        &[1.0, 3.0, 5.0, 2.0, 4.0, 6.0]
    );
}

#[test]
fn stack_at_dim_0_is_unsqueeze_then_cat() {
    let backend = cpu_backend();
    let a = Tensor::from_vec([3usize], vec![1.0_f32, 2.0, 3.0]).unwrap();
    let b = Tensor::from_vec([3usize], vec![4.0_f32, 5.0, 6.0]).unwrap();
    let s = backend.stack(&[&a, &b], 0).unwrap();
    assert_eq!(s.shape(), &[2, 3]);
    assert_eq!(
        s.as_slice::<f32>().unwrap(),
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    );
}

#[test]
fn stack_shape_mismatch_returns_err() {
    let backend = cpu_backend();
    let a = Tensor::from_vec([3usize], vec![1.0_f32; 3]).unwrap();
    let b = Tensor::from_vec([4usize], vec![1.0_f32; 4]).unwrap();
    assert!(matches!(
        backend.stack(&[&a, &b], 0),
        Err(BackendError::ShapeMismatch { .. })
    ));
}

// ------------------------- split / chunk -------------------------

#[test]
fn split_returns_pieces() {
    let backend = cpu_backend();
    let t = Tensor::from_vec([6usize], vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let pieces = backend.split(&t, 2, 0).unwrap();
    assert_eq!(pieces.len(), 3);
    assert_eq!(pieces[0].as_slice::<f32>().unwrap(), &[1.0, 2.0]);
    assert_eq!(pieces[1].as_slice::<f32>().unwrap(), &[3.0, 4.0]);
    assert_eq!(pieces[2].as_slice::<f32>().unwrap(), &[5.0, 6.0]);
}

#[test]
fn split_uneven_last_piece_smaller() {
    let backend = cpu_backend();
    let t = Tensor::from_vec([5usize], vec![1.0_f32, 2.0, 3.0, 4.0, 5.0]).unwrap();
    let pieces = backend.split(&t, 2, 0).unwrap();
    assert_eq!(pieces.len(), 3);
    assert_eq!(pieces[2].as_slice::<f32>().unwrap(), &[5.0]);
}

#[test]
fn split_size_zero_returns_err() {
    let backend = cpu_backend();
    let t = Tensor::from_vec([3usize], vec![1.0_f32; 3]).unwrap();
    assert!(matches!(
        backend.split(&t, 0, 0),
        Err(BackendError::ShapeMismatch { .. })
    ));
}

#[test]
fn split_then_cat_round_trip() {
    // Property: split(t, k, dim) ; cat(pieces, dim) == t (within ordering).
    let backend = cpu_backend();
    let t = Tensor::from_vec([2usize, 6], (1..=12).map(|i| i as f32).collect()).unwrap();
    let pieces = backend.split(&t, 3, 1).unwrap();
    let refs: Vec<&Tensor> = pieces.iter().collect();
    let recombined = backend.cat(&refs, 1).unwrap();
    assert_eq!(recombined.shape(), t.shape());
    assert_eq!(
        recombined.as_slice::<f32>().unwrap(),
        t.as_slice::<f32>().unwrap()
    );
}

#[test]
fn chunk_into_n_pieces() {
    let backend = cpu_backend();
    let t = Tensor::from_vec([6usize], vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let pieces = backend.chunk(&t, 3, 0).unwrap();
    assert_eq!(pieces.len(), 3);
    assert_eq!(pieces[0].as_slice::<f32>().unwrap(), &[1.0, 2.0]);
    assert_eq!(pieces[1].as_slice::<f32>().unwrap(), &[3.0, 4.0]);
    assert_eq!(pieces[2].as_slice::<f32>().unwrap(), &[5.0, 6.0]);
}

#[test]
fn chunk_with_uneven_division() {
    let backend = cpu_backend();
    let t = Tensor::from_vec([7usize], (1..=7).map(|i| i as f32).collect()).unwrap();
    let pieces = backend.chunk(&t, 3, 0).unwrap();
    // ceil(7/3) = 3 → pieces of [3, 3, 1]
    assert_eq!(pieces[0].shape(), &[3]);
    assert_eq!(pieces[1].shape(), &[3]);
    assert_eq!(pieces[2].shape(), &[1]);
}

#[test]
fn chunk_zero_chunks_returns_err() {
    let backend = cpu_backend();
    let t = Tensor::from_vec([3usize], vec![1.0_f32; 3]).unwrap();
    assert!(matches!(
        backend.chunk(&t, 0, 0),
        Err(BackendError::ShapeMismatch { .. })
    ));
}

// ------------------------- repeat -------------------------

#[test]
fn repeat_1d() {
    let backend = cpu_backend();
    let t = Tensor::from_vec([3usize], vec![1.0_f32, 2.0, 3.0]).unwrap();
    let r = backend.repeat(&t, &[3]).unwrap();
    assert_eq!(r.shape(), &[9]);
    assert_eq!(
        r.as_slice::<f32>().unwrap(),
        &[1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0]
    );
}

#[test]
fn repeat_2d() {
    let backend = cpu_backend();
    let t = Tensor::from_vec([2usize, 2], vec![1.0_f32, 2.0, 3.0, 4.0]).unwrap();
    let r = backend.repeat(&t, &[2, 1]).unwrap();
    assert_eq!(r.shape(), &[4, 2]);
    assert_eq!(
        r.as_slice::<f32>().unwrap(),
        &[1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0]
    );
}

#[test]
fn repeat_with_factor_one_is_identity() {
    let backend = cpu_backend();
    let t = Tensor::from_vec([3usize], vec![1.0_f32, 2.0, 3.0]).unwrap();
    let r = backend.repeat(&t, &[1]).unwrap();
    assert_eq!(r.as_slice::<f32>().unwrap(), t.as_slice::<f32>().unwrap());
}

#[test]
fn repeat_rank_mismatch_returns_err() {
    let backend = cpu_backend();
    let t = Tensor::from_vec([3usize], vec![1.0_f32; 3]).unwrap();
    assert!(matches!(
        backend.repeat(&t, &[2, 2]),
        Err(BackendError::ShapeMismatch { .. })
    ));
}

// ------------------------- flip -------------------------

#[test]
fn flip_1d_reverses() {
    let backend = cpu_backend();
    let t = Tensor::from_vec([4usize], vec![1.0_f32, 2.0, 3.0, 4.0]).unwrap();
    let r = backend.flip(&t, &[0]).unwrap();
    assert_eq!(r.as_slice::<f32>().unwrap(), &[4.0, 3.0, 2.0, 1.0]);
}

#[test]
fn flip_2d_along_both_dims() {
    let backend = cpu_backend();
    let t = Tensor::from_vec([2usize, 2], vec![1.0_f32, 2.0, 3.0, 4.0]).unwrap();
    let r = backend.flip(&t, &[0, 1]).unwrap();
    assert_eq!(r.as_slice::<f32>().unwrap(), &[4.0, 3.0, 2.0, 1.0]);
}

#[test]
fn flip_involution_property() {
    // flip(flip(t, dims), dims) == t
    let backend = cpu_backend();
    let t = Tensor::from_vec([2usize, 3], vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let f1 = backend.flip(&t, &[0]).unwrap();
    let f2 = backend.flip(&f1, &[0]).unwrap();
    assert_eq!(f2.as_slice::<f32>().unwrap(), t.as_slice::<f32>().unwrap());
}

#[test]
fn flip_dim_out_of_range() {
    let backend = cpu_backend();
    let t = Tensor::from_vec([3usize], vec![1.0_f32; 3]).unwrap();
    assert!(matches!(
        backend.flip(&t, &[5]),
        Err(BackendError::IndexOutOfBounds { .. })
    ));
}

// ------------------------- roll -------------------------

#[test]
fn roll_1d_positive_shift() {
    let backend = cpu_backend();
    let t = Tensor::from_vec([5usize], vec![1.0_f32, 2.0, 3.0, 4.0, 5.0]).unwrap();
    let r = backend.roll(&t, 2, 0).unwrap();
    assert_eq!(r.as_slice::<f32>().unwrap(), &[4.0, 5.0, 1.0, 2.0, 3.0]);
}

#[test]
fn roll_1d_negative_shift() {
    let backend = cpu_backend();
    let t = Tensor::from_vec([5usize], vec![1.0_f32, 2.0, 3.0, 4.0, 5.0]).unwrap();
    let r = backend.roll(&t, -1, 0).unwrap();
    assert_eq!(r.as_slice::<f32>().unwrap(), &[2.0, 3.0, 4.0, 5.0, 1.0]);
}

#[test]
fn roll_zero_shift_is_identity() {
    let backend = cpu_backend();
    let t = Tensor::from_vec([3usize], vec![1.0_f32, 2.0, 3.0]).unwrap();
    let r = backend.roll(&t, 0, 0).unwrap();
    assert_eq!(r.as_slice::<f32>().unwrap(), t.as_slice::<f32>().unwrap());
}

#[test]
fn roll_full_period_is_identity() {
    let backend = cpu_backend();
    let t = Tensor::from_vec([4usize], vec![1.0_f32, 2.0, 3.0, 4.0]).unwrap();
    let r = backend.roll(&t, 4, 0).unwrap();
    assert_eq!(r.as_slice::<f32>().unwrap(), t.as_slice::<f32>().unwrap());
}
