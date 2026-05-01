//! Naïve 2-D matrix multiplication — three nested loops, no optimization.

use crate::{Error, Result, Tensor};

/// `(M, K) × (K, N) → (M, N)`. Two 2-D tensors only.
///
/// Returns `Err(Error::Rank { ... })` for non-2-D inputs and
/// `Err(Error::ShapeMismatch { ... })` when the inner dimensions disagree.
pub fn matmul(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    if a.ndim() != 2 {
        return Err(Error::Rank {
            op: "matmul",
            got: a.ndim(),
            want: "2-D",
        });
    }
    if b.ndim() != 2 {
        return Err(Error::Rank {
            op: "matmul",
            got: b.ndim(),
            want: "2-D",
        });
    }
    let (m, k1) = (a.shape()[0], a.shape()[1]);
    let (k2, n) = (b.shape()[0], b.shape()[1]);
    if k1 != k2 {
        return Err(Error::ShapeMismatch {
            op: "matmul",
            lhs: a.shape().to_vec(),
            rhs: b.shape().to_vec(),
        });
    }
    let (da, db) = (a.data(), b.data());
    let mut out = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for k in 0..k1 {
                sum += da[i * k1 + k] * db[k * n + j];
            }
            out[i * n + j] = sum;
        }
    }
    Tensor::from_vec(vec![m, n], out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_2x3_3x2() {
        // [[1, 2, 3], [4, 5, 6]] @ [[7, 8], [9, 10], [11, 12]]
        let a = Tensor::from_vec([2usize, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let b = Tensor::from_vec([3usize, 2], vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]).unwrap();
        let c = matmul(&a, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        // [[ 1*7+2*9+3*11, 1*8+2*10+3*12],
        //  [ 4*7+5*9+6*11, 4*8+5*10+6*12]]
        // = [[ 58, 64], [139, 154]]
        assert_eq!(c.data(), &[58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    fn identity_matmul_is_id() {
        // 3x3 I times an arbitrary 3x2 matrix returns the matrix.
        let id = Tensor::from_vec(
            [3usize, 3],
            vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        )
        .unwrap();
        let x = Tensor::from_vec([3usize, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        assert_eq!(matmul(&id, &x).unwrap().data(), x.data());
    }

    #[test]
    fn shape_mismatch_returns_err() {
        let a = Tensor::from_vec([2usize, 3], vec![0.0; 6]).unwrap();
        let b = Tensor::from_vec([4usize, 2], vec![0.0; 8]).unwrap();
        match matmul(&a, &b) {
            Err(Error::ShapeMismatch { op, lhs, rhs }) => {
                assert_eq!(op, "matmul");
                assert_eq!(lhs, vec![2, 3]);
                assert_eq!(rhs, vec![4, 2]);
            },
            _ => panic!("expected ShapeMismatch"),
        }
    }

    #[test]
    fn rank_mismatch_returns_err() {
        let a = Tensor::scalar(1.0);
        let b = Tensor::from_vec([2usize, 2], vec![0.0; 4]).unwrap();
        match matmul(&a, &b) {
            Err(Error::Rank { op, got, want }) => {
                assert_eq!(op, "matmul");
                assert_eq!(got, 0);
                assert_eq!(want, "2-D");
            },
            _ => panic!("expected Rank"),
        }
    }

    #[test]
    fn empty_dim_yields_empty() {
        // (0, 3) × (3, 2) is a valid empty result of shape (0, 2)
        let a = Tensor::from_vec([0usize, 3], vec![]).unwrap();
        let b = Tensor::from_vec([3usize, 2], vec![0.0; 6]).unwrap();
        let c = matmul(&a, &b).unwrap();
        assert_eq!(c.shape(), &[0, 2]);
        assert!(c.is_empty());
    }

    #[test]
    fn nan_propagates() {
        let a = Tensor::from_vec([1usize, 1], vec![f32::NAN]).unwrap();
        let b = Tensor::from_vec([1usize, 1], vec![1.0]).unwrap();
        assert!(matmul(&a, &b).unwrap().data()[0].is_nan());
    }
}
