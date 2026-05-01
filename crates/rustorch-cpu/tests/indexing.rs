//! Indexing kernel tests (P1.3 task `Indexing`).
//!
//! Covers gather/scatter/scatter_add/index_select/masked_select/
//! masked_fill/where/nonzero with the spec's edge cases:
//! - empty index returns empty output
//! - NaN preserved in masked_select
//! - shape mismatch returns BackendError
//! - dtype mismatch (idx must be I64) returns BackendError
//! - out-of-bounds index returns IndexOutOfBounds
//! - identity property: `gather(t, dim, arange(t.size(dim))) == t`

use rustorch_core::tensor::tensor_impl::Tensor;
use rustorch_cpu::cpu_backend::cpu_backend;
use rustorch_cpu::error::BackendError;

// ---------------------------- gather ----------------------------

#[test]
fn gather_2d_along_dim_1() {
    let backend = cpu_backend();
    // src = [[1,2,3],[4,5,6]]; idx = [[0,2],[1,1]] along dim=1
    // → [[1,3],[5,5]]
    let src = Tensor::from_vec([2usize, 3], vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let idx = Tensor::from_vec_typed::<i64, _>([2usize, 2], vec![0, 2, 1, 1]).unwrap();
    let out = backend.gather(&src, 1, &idx).unwrap();
    assert_eq!(out.shape(), &[2, 2]);
    assert_eq!(out.as_slice::<f32>().unwrap(), &[1.0, 3.0, 5.0, 5.0]);
}

#[test]
fn gather_identity_with_arange_idx() {
    // gather(t, dim, arange(size(dim))) == t (identity property).
    let backend = cpu_backend();
    let src = Tensor::from_vec([2usize, 3], vec![10.0_f32, 20.0, 30.0, 40.0, 50.0, 60.0]).unwrap();
    // For each row, idx is [0, 1, 2]
    let idx = Tensor::from_vec_typed::<i64, _>([2usize, 3], vec![0, 1, 2, 0, 1, 2]).unwrap();
    let out = backend.gather(&src, 1, &idx).unwrap();
    assert_eq!(
        out.as_slice::<f32>().unwrap(),
        src.as_slice::<f32>().unwrap()
    );
}

#[test]
fn gather_idx_dtype_must_be_i64() {
    let backend = cpu_backend();
    let src = Tensor::from_vec([3usize], vec![1.0_f32; 3]).unwrap();
    let idx = Tensor::from_vec_typed::<i32, _>([3usize], vec![0, 1, 2]).unwrap();
    assert!(matches!(
        backend.gather(&src, 0, &idx),
        Err(BackendError::DtypeMismatch { .. })
    ));
}

#[test]
fn gather_out_of_bounds_returns_err() {
    let backend = cpu_backend();
    let src = Tensor::from_vec([3usize], vec![1.0_f32; 3]).unwrap();
    let idx = Tensor::from_vec_typed::<i64, _>([1usize], vec![5_i64]).unwrap();
    assert!(matches!(
        backend.gather(&src, 0, &idx),
        Err(BackendError::IndexOutOfBounds { .. })
    ));
    let idx_neg = Tensor::from_vec_typed::<i64, _>([1usize], vec![-1_i64]).unwrap();
    assert!(matches!(
        backend.gather(&src, 0, &idx_neg),
        Err(BackendError::IndexOutOfBounds { .. })
    ));
}

#[test]
fn gather_dim_out_of_range() {
    let backend = cpu_backend();
    let src = Tensor::from_vec([3usize], vec![1.0_f32; 3]).unwrap();
    let idx = Tensor::from_vec_typed::<i64, _>([3usize], vec![0_i64, 1, 2]).unwrap();
    assert!(matches!(
        backend.gather(&src, 5, &idx),
        Err(BackendError::IndexOutOfBounds { .. })
    ));
}

#[test]
fn gather_rank_mismatch() {
    let backend = cpu_backend();
    let src = Tensor::from_vec([3usize], vec![1.0_f32; 3]).unwrap();
    let idx = Tensor::from_vec_typed::<i64, _>([2usize, 2], vec![0_i64, 1, 2, 0]).unwrap();
    assert!(matches!(
        backend.gather(&src, 0, &idx),
        Err(BackendError::ShapeMismatch { .. })
    ));
}

// ---------------------------- scatter / scatter_add ----------------------------

#[test]
fn scatter_overwrites() {
    // dst = zeros [2, 3]; idx = [[0,1,2],[2,1,0]] along dim=1; src = ones-like idx
    // scatter writes 1.0 at every (row, idx[row, col]) — covers every cell.
    let backend = cpu_backend();
    let dst = Tensor::zeros([2usize, 3]);
    let idx = Tensor::from_vec_typed::<i64, _>([2usize, 3], vec![0_i64, 1, 2, 2, 1, 0]).unwrap();
    let src = Tensor::from_vec([2usize, 3], vec![1.0_f32; 6]).unwrap();
    let out = backend.scatter(&dst, 1, &idx, &src).unwrap();
    assert_eq!(out.as_slice::<f32>().unwrap(), &[1.0_f32; 6]);
}

#[test]
fn scatter_add_accumulates_duplicates() {
    // dst = zeros[3]; idx = [0, 0, 1]; src = [1, 2, 3]
    // scatter_add: dst[0] += 1; dst[0] += 2; dst[1] += 3 → [3, 3, 0]
    let backend = cpu_backend();
    let dst = Tensor::zeros([3usize]);
    let idx = Tensor::from_vec_typed::<i64, _>([3usize], vec![0_i64, 0, 1]).unwrap();
    let src = Tensor::from_vec([3usize], vec![1.0_f32, 2.0, 3.0]).unwrap();
    let out = backend.scatter_add(&dst, 0, &idx, &src).unwrap();
    assert_eq!(out.as_slice::<f32>().unwrap(), &[3.0, 3.0, 0.0]);
}

#[test]
fn scatter_dtype_mismatch() {
    let backend = cpu_backend();
    let dst = Tensor::zeros([3usize]);
    let idx = Tensor::from_vec_typed::<i64, _>([3usize], vec![0_i64, 1, 2]).unwrap();
    let src = Tensor::from_vec_typed::<i64, _>([3usize], vec![1_i64, 2, 3]).unwrap();
    assert!(matches!(
        backend.scatter(&dst, 0, &idx, &src),
        Err(BackendError::DtypeMismatch { .. })
    ));
}

#[test]
fn scatter_out_of_bounds_index() {
    let backend = cpu_backend();
    let dst = Tensor::zeros([3usize]);
    let idx = Tensor::from_vec_typed::<i64, _>([3usize], vec![0_i64, 1, 5]).unwrap();
    let src = Tensor::from_vec([3usize], vec![1.0_f32; 3]).unwrap();
    assert!(matches!(
        backend.scatter(&dst, 0, &idx, &src),
        Err(BackendError::IndexOutOfBounds { .. })
    ));
}

// ---------------------------- index_select ----------------------------

#[test]
fn index_select_picks_rows() {
    // src [3,2] = [[10,20],[30,40],[50,60]]; indices = [0, 2]
    // → [[10,20],[50,60]]
    let backend = cpu_backend();
    let src = Tensor::from_vec([3usize, 2], vec![10.0_f32, 20.0, 30.0, 40.0, 50.0, 60.0]).unwrap();
    let indices = Tensor::from_vec_typed::<i64, _>([2usize], vec![0_i64, 2]).unwrap();
    let out = backend.index_select(&src, 0, &indices).unwrap();
    assert_eq!(out.shape(), &[2, 2]);
    assert_eq!(out.as_slice::<f32>().unwrap(), &[10.0, 20.0, 50.0, 60.0]);
}

#[test]
fn index_select_empty_indices_returns_empty_along_dim() {
    let backend = cpu_backend();
    let src = Tensor::from_vec([3usize], vec![1.0_f32, 2.0, 3.0]).unwrap();
    let indices = Tensor::from_vec_typed::<i64, _>([0usize], vec![]).unwrap();
    let out = backend.index_select(&src, 0, &indices).unwrap();
    assert_eq!(out.shape(), &[0]);
    assert!(out.is_empty());
}

#[test]
fn index_select_indices_must_be_1d() {
    let backend = cpu_backend();
    let src = Tensor::from_vec([3usize], vec![1.0_f32; 3]).unwrap();
    let indices = Tensor::from_vec_typed::<i64, _>([1usize, 2], vec![0_i64, 1]).unwrap();
    assert!(matches!(
        backend.index_select(&src, 0, &indices),
        Err(BackendError::ShapeMismatch { .. })
    ));
}

// ---------------------------- masked_select ----------------------------

#[test]
fn masked_select_returns_flat() {
    let backend = cpu_backend();
    let src = Tensor::from_vec([2usize, 3], vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let mask =
        Tensor::from_vec_typed::<bool, _>([2usize, 3], vec![true, false, true, false, true, false])
            .unwrap();
    let out = backend.masked_select(&src, &mask).unwrap();
    assert_eq!(out.shape(), &[3]);
    assert_eq!(out.as_slice::<f32>().unwrap(), &[1.0, 3.0, 5.0]);
}

#[test]
fn masked_select_preserves_nan() {
    let backend = cpu_backend();
    let src = Tensor::from_vec([3usize], vec![1.0_f32, f32::NAN, 3.0]).unwrap();
    let mask = Tensor::from_vec_typed::<bool, _>([3usize], vec![true, true, true]).unwrap();
    let out = backend.masked_select(&src, &mask).unwrap();
    let v = out.as_slice::<f32>().unwrap();
    assert!(v[1].is_nan());
    assert_eq!(v[0], 1.0);
    assert_eq!(v[2], 3.0);
}

#[test]
fn masked_select_dtype_mismatch_on_mask() {
    let backend = cpu_backend();
    let src = Tensor::from_vec([3usize], vec![1.0_f32; 3]).unwrap();
    let mask = Tensor::from_vec([3usize], vec![1.0_f32; 3]).unwrap();
    assert!(matches!(
        backend.masked_select(&src, &mask),
        Err(BackendError::DtypeMismatch { .. })
    ));
}

// ---------------------------- masked_fill ----------------------------

#[test]
fn masked_fill_writes_value() {
    let backend = cpu_backend();
    let src = Tensor::from_vec([4usize], vec![1.0_f32, 2.0, 3.0, 4.0]).unwrap();
    let mask = Tensor::from_vec_typed::<bool, _>([4usize], vec![false, true, false, true]).unwrap();
    let out = backend.masked_fill(&src, &mask, -1.0).unwrap();
    assert_eq!(out.as_slice::<f32>().unwrap(), &[1.0, -1.0, 3.0, -1.0]);
}

// ---------------------------- where ----------------------------

#[test]
fn where_selects_per_element() {
    let backend = cpu_backend();
    let cond = Tensor::from_vec_typed::<bool, _>([4usize], vec![true, false, true, false]).unwrap();
    let x = Tensor::from_vec([4usize], vec![1.0_f32, 2.0, 3.0, 4.0]).unwrap();
    let y = Tensor::from_vec([4usize], vec![10.0_f32, 20.0, 30.0, 40.0]).unwrap();
    let out = backend.r#where(&cond, &x, &y).unwrap();
    assert_eq!(out.as_slice::<f32>().unwrap(), &[1.0, 20.0, 3.0, 40.0]);
}

#[test]
fn where_dtype_mismatch_xy() {
    let backend = cpu_backend();
    let cond = Tensor::from_vec_typed::<bool, _>([3usize], vec![true; 3]).unwrap();
    let x = Tensor::from_vec([3usize], vec![1.0_f32; 3]).unwrap();
    let y = Tensor::from_vec_typed::<i64, _>([3usize], vec![1_i64; 3]).unwrap();
    assert!(matches!(
        backend.r#where(&cond, &x, &y),
        Err(BackendError::DtypeMismatch { .. })
    ));
}

// ---------------------------- nonzero ----------------------------

#[test]
fn nonzero_1d() {
    let backend = cpu_backend();
    let src = Tensor::from_vec([5usize], vec![1.0_f32, 0.0, 2.0, 0.0, -3.0]).unwrap();
    let out = backend.nonzero(&src).unwrap();
    assert_eq!(out.shape(), &[3, 1]);
    assert_eq!(out.as_slice::<i64>().unwrap(), &[0_i64, 2, 4]);
}

#[test]
fn nonzero_2d() {
    let backend = cpu_backend();
    let src = Tensor::from_vec([2usize, 3], vec![0.0_f32, 1.0, 0.0, 2.0, 0.0, 3.0]).unwrap();
    let out = backend.nonzero(&src).unwrap();
    assert_eq!(out.shape(), &[3, 2]);
    // Coordinates of nonzero: (0,1), (1,0), (1,2)
    assert_eq!(out.as_slice::<i64>().unwrap(), &[0_i64, 1, 1, 0, 1, 2]);
}

#[test]
fn nonzero_bool() {
    let backend = cpu_backend();
    let src = Tensor::from_vec_typed::<bool, _>([4usize], vec![true, false, true, false]).unwrap();
    let out = backend.nonzero(&src).unwrap();
    assert_eq!(out.shape(), &[2, 1]);
    assert_eq!(out.as_slice::<i64>().unwrap(), &[0_i64, 2]);
}

#[test]
fn nonzero_all_zeros_returns_empty() {
    let backend = cpu_backend();
    let src = Tensor::zeros([3usize]);
    let out = backend.nonzero(&src).unwrap();
    assert_eq!(out.shape(), &[0, 1]);
    assert!(out.is_empty());
}
