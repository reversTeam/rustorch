//! Element-wise addition with limited broadcasting (scalar + N-D, row + 2-D).

use crate::{Error, Result, Tensor};

/// `a + b`. Broadcast rules in this prototype:
/// 1. Same shape → element-wise sum.
/// 2. Scalar (shape `[]`) + N-D → fill scalar across.
/// 3. Row vector `[N]` + 2-D `[M, N]` → repeat the row M times.
///
/// Anything else returns `Err(Error::ShapeMismatch { op: "add", .. })`.
/// IEEE-754 semantics — NaN, ±Inf propagate, no special handling.
pub fn add(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    let (sa, sb) = (a.shape(), b.shape());
    let (da, db) = (a.data(), b.data());

    // Case 1: identical shapes
    if sa == sb {
        let out: Vec<f32> = da.iter().zip(db.iter()).map(|(x, y)| x + y).collect();
        return Tensor::from_vec(sa.to_vec(), out).map_err(Into::into);
    }

    // Case 2a: a is scalar
    if sa.is_empty() {
        let s = da[0];
        let out: Vec<f32> = db.iter().map(|y| y + s).collect();
        return Tensor::from_vec(sb.to_vec(), out).map_err(Into::into);
    }
    // Case 2b: b is scalar
    if sb.is_empty() {
        let s = db[0];
        let out: Vec<f32> = da.iter().map(|x| x + s).collect();
        return Tensor::from_vec(sa.to_vec(), out).map_err(Into::into);
    }

    // Case 3a: row vector [N] + matrix [M, N]
    if sa.len() == 1 && sb.len() == 2 && sa[0] == sb[1] {
        let n = sa[0];
        let m = sb[0];
        let mut out = Vec::with_capacity(m * n);
        for i in 0..m {
            for j in 0..n {
                out.push(db[i * n + j] + da[j]);
            }
        }
        return Tensor::from_vec(vec![m, n], out).map_err(Into::into);
    }
    // Case 3b: matrix [M, N] + row vector [N]
    if sb.len() == 1 && sa.len() == 2 && sb[0] == sa[1] {
        let n = sb[0];
        let m = sa[0];
        let mut out = Vec::with_capacity(m * n);
        for i in 0..m {
            for j in 0..n {
                out.push(da[i * n + j] + db[j]);
            }
        }
        return Tensor::from_vec(vec![m, n], out).map_err(Into::into);
    }

    Err(Error::ShapeMismatch {
        op: "add",
        lhs: sa.to_vec(),
        rhs: sb.to_vec(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_shape() {
        let a = Tensor::from_vec([2usize], vec![1.0, 2.0]).unwrap();
        let b = Tensor::from_vec([2usize], vec![3.0, 4.0]).unwrap();
        assert_eq!(add(&a, &b).unwrap().data(), &[4.0, 6.0]);
    }

    #[test]
    fn scalar_broadcast() {
        let a = Tensor::scalar(10.0);
        let b = Tensor::from_vec([2usize, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        assert_eq!(add(&a, &b).unwrap().data(), &[11.0, 12.0, 13.0, 14.0]);
    }

    #[test]
    fn row_broadcast() {
        // shape [3] + shape [2, 3]
        let row = Tensor::from_vec([3usize], vec![10.0, 20.0, 30.0]).unwrap();
        let mat = Tensor::from_vec([2usize, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let out = add(&row, &mat).unwrap();
        assert_eq!(out.shape(), &[2, 3]);
        assert_eq!(out.data(), &[11.0, 22.0, 33.0, 14.0, 25.0, 36.0]);
        // and the symmetric form
        let out2 = add(&mat, &row).unwrap();
        assert_eq!(out2.data(), out.data());
    }

    #[test]
    fn shape_mismatch_err() {
        let a = Tensor::from_vec([2usize, 3], vec![0.0; 6]).unwrap();
        let b = Tensor::from_vec([3usize, 2], vec![0.0; 6]).unwrap();
        match add(&a, &b) {
            Err(Error::ShapeMismatch { op, lhs, rhs }) => {
                assert_eq!(op, "add");
                assert_eq!(lhs, vec![2, 3]);
                assert_eq!(rhs, vec![3, 2]);
            },
            _ => panic!("expected ShapeMismatch"),
        }
    }

    #[test]
    fn empty_plus_empty() {
        let a = Tensor::zeros([0usize]);
        let b = Tensor::zeros([0usize]);
        let s = add(&a, &b).unwrap();
        assert_eq!(s.shape(), &[0]);
        assert!(s.is_empty());
    }

    #[test]
    fn nan_propagates() {
        let a = Tensor::from_vec([1usize], vec![f32::NAN]).unwrap();
        let b = Tensor::from_vec([1usize], vec![1.0]).unwrap();
        assert!(add(&a, &b).unwrap().data()[0].is_nan());
    }

    #[test]
    fn inf_minus_inf_is_nan() {
        let a = Tensor::from_vec([1usize], vec![f32::INFINITY]).unwrap();
        let b = Tensor::from_vec([1usize], vec![f32::NEG_INFINITY]).unwrap();
        assert!(add(&a, &b).unwrap().data()[0].is_nan());
    }
}
