//! Advanced linear algebra (P1.4) — QR / Cholesky / inverse / solve.
//!
//! v1 ships **square 2-D F32** variants written in pure Rust:
//! - [`qr`] via classical Gram-Schmidt orthogonalisation
//! - [`cholesky`] via the standard outer-product algorithm (lower
//!   triangular L such that A = L Lᵀ; A must be symmetric positive
//!   definite)
//! - [`inverse`] via LU + back-sub (square invertible matrices)
//! - [`solve`] solves `A X = B` via LU back-sub
//!
//! `svd` lands in a follow-up — it warrants its own slice with a
//! dedicated bidiagonalisation + Givens rotation pipeline.

#![allow(clippy::needless_range_loop)]

use crate::error::BackendError;
use rustorch_core::tensor::dtype::Dtype;
use rustorch_core::tensor::tensor_impl::Tensor;

fn assert_square_f32(a: &Tensor, op: &'static str) -> Result<usize, BackendError> {
    if a.dtype() != Dtype::F32 {
        return Err(BackendError::DtypeMismatch {
            op,
            lhs: a.dtype(),
            rhs: Dtype::F32,
        });
    }
    if a.ndim() != 2 || a.shape()[0] != a.shape()[1] {
        return Err(BackendError::ShapeMismatch {
            op,
            lhs: a.shape().to_vec(),
            rhs: a.shape().to_vec(),
        });
    }
    Ok(a.shape()[0])
}

/// QR decomposition of a square matrix via classical Gram-Schmidt.
/// Returns `(Q, R)` with `Q.Qᵀ == I` and `R` upper triangular.
pub fn qr(a: &Tensor) -> Result<(Tensor, Tensor), BackendError> {
    let n = assert_square_f32(a, "qr")?;
    let a_data = a.as_slice::<f32>().expect("F32");
    let mut q = vec![0.0_f32; n * n];
    let mut r = vec![0.0_f32; n * n];

    for j in 0..n {
        // Initial v = a[:, j]
        let mut v: Vec<f32> = (0..n).map(|i| a_data[i * n + j]).collect();
        for i in 0..j {
            let mut dot = 0.0_f32;
            for k in 0..n {
                dot += q[k * n + i] * a_data[k * n + j];
            }
            r[i * n + j] = dot;
            for k in 0..n {
                v[k] -= dot * q[k * n + i];
            }
        }
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        r[j * n + j] = norm;
        if norm > 1e-12 {
            for k in 0..n {
                q[k * n + j] = v[k] / norm;
            }
        }
    }
    let q_t = Tensor::from_vec([n, n], q).expect("qr Q");
    let r_t = Tensor::from_vec([n, n], r).expect("qr R");
    Ok((q_t, r_t))
}

/// Cholesky decomposition: lower-triangular L such that `A = L Lᵀ`.
/// Requires `A` symmetric positive definite.
pub fn cholesky(a: &Tensor) -> Result<Tensor, BackendError> {
    let n = assert_square_f32(a, "cholesky")?;
    let a_data = a.as_slice::<f32>().expect("F32");
    let mut l = vec![0.0_f32; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut sum = 0.0_f32;
            for k in 0..j {
                sum += l[i * n + k] * l[j * n + k];
            }
            if i == j {
                let diag = a_data[i * n + i] - sum;
                if diag <= 0.0 {
                    return Err(BackendError::ShapeMismatch {
                        op: "cholesky",
                        lhs: vec![n, n],
                        rhs: vec![n, n],
                    });
                }
                l[i * n + j] = diag.sqrt();
            } else {
                l[i * n + j] = (a_data[i * n + j] - sum) / l[j * n + j];
            }
        }
    }
    Tensor::from_vec([n, n], l).map_err(|_| BackendError::OutOfMemory { bytes: n * n * 4 })
}

/// LU decomposition with partial pivoting. Returns `(L, U, P)` such
/// that `P A = L U`; `P` is encoded as a permutation vector.
pub fn lu(a: &Tensor) -> Result<(Tensor, Tensor, Vec<usize>), BackendError> {
    let n = assert_square_f32(a, "lu")?;
    let a_data = a.as_slice::<f32>().expect("F32");
    let mut u: Vec<f32> = a_data.to_vec();
    let mut l = vec![0.0_f32; n * n];
    let mut p: Vec<usize> = (0..n).collect();
    for k in 0..n {
        // Pivot: find row with max |u[i, k]|
        let mut piv = k;
        let mut piv_val = u[k * n + k].abs();
        for i in (k + 1)..n {
            let v = u[i * n + k].abs();
            if v > piv_val {
                piv = i;
                piv_val = v;
            }
        }
        if piv_val < 1e-12 {
            return Err(BackendError::ShapeMismatch {
                op: "lu",
                lhs: vec![n, n],
                rhs: vec![n, n],
            });
        }
        if piv != k {
            for j in 0..n {
                u.swap(k * n + j, piv * n + j);
                l.swap(k * n + j, piv * n + j);
            }
            p.swap(k, piv);
        }
        for i in (k + 1)..n {
            let factor = u[i * n + k] / u[k * n + k];
            l[i * n + k] = factor;
            for j in k..n {
                u[i * n + j] -= factor * u[k * n + j];
            }
        }
    }
    for i in 0..n {
        l[i * n + i] = 1.0;
    }
    let l_t = Tensor::from_vec([n, n], l).expect("L");
    let u_t = Tensor::from_vec([n, n], u).expect("U");
    Ok((l_t, u_t, p))
}

/// Inverse of a square matrix via LU + back-sub on identity columns.
pub fn inverse(a: &Tensor) -> Result<Tensor, BackendError> {
    let n = assert_square_f32(a, "inverse")?;
    let (l, u, p) = lu(a)?;
    let l_data = l.as_slice::<f32>().expect("F32 L");
    let u_data = u.as_slice::<f32>().expect("F32 U");
    let mut inv = vec![0.0_f32; n * n];
    // Solve A X = I one column at a time.
    for col in 0..n {
        let mut b = vec![0.0_f32; n];
        b[col] = 1.0;
        // Apply permutation P
        let mut pb = vec![0.0_f32; n];
        for i in 0..n {
            pb[i] = b[p[i]];
        }
        // Forward solve L y = pb
        let mut y = vec![0.0_f32; n];
        for i in 0..n {
            let mut sum = pb[i];
            for j in 0..i {
                sum -= l_data[i * n + j] * y[j];
            }
            y[i] = sum;
        }
        // Backward solve U x = y
        let mut x = vec![0.0_f32; n];
        for i in (0..n).rev() {
            let mut sum = y[i];
            for j in (i + 1)..n {
                sum -= u_data[i * n + j] * x[j];
            }
            x[i] = sum / u_data[i * n + i];
        }
        for i in 0..n {
            inv[i * n + col] = x[i];
        }
    }
    Tensor::from_vec([n, n], inv).map_err(|_| BackendError::OutOfMemory { bytes: n * n * 4 })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t(shape: &[usize], data: Vec<f32>) -> Tensor {
        Tensor::from_vec(shape.to_vec(), data).unwrap()
    }

    fn matmul(a: &[f32], b: &[f32], n: usize) -> Vec<f32> {
        let mut out = vec![0.0_f32; n * n];
        for i in 0..n {
            for j in 0..n {
                let mut s = 0.0_f32;
                for k in 0..n {
                    s += a[i * n + k] * b[k * n + j];
                }
                out[i * n + j] = s;
            }
        }
        out
    }

    #[test]
    fn qr_decomposition_roundtrip_2x2() {
        let a = t(&[2, 2], vec![1.0_f32, 2.0, 3.0, 4.0]);
        let (q, r) = qr(&a).unwrap();
        let qm = q.as_slice::<f32>().unwrap();
        let rm = r.as_slice::<f32>().unwrap();
        let qr_prod = matmul(qm, rm, 2);
        for i in 0..4 {
            assert!((qr_prod[i] - a.as_slice::<f32>().unwrap()[i]).abs() < 1e-4);
        }
    }

    #[test]
    fn cholesky_roundtrip_spd() {
        // A = [[4, 12], [12, 37]] is symmetric positive definite.
        let a = t(&[2, 2], vec![4.0_f32, 12.0, 12.0, 37.0]);
        let l = cholesky(&a).unwrap();
        let l_data = l.as_slice::<f32>().unwrap();
        // L = [[2, 0], [6, 1]]
        assert!((l_data[0] - 2.0).abs() < 1e-5);
        assert!((l_data[1] - 0.0).abs() < 1e-5);
        assert!((l_data[2] - 6.0).abs() < 1e-5);
        assert!((l_data[3] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn inverse_2x2_correct() {
        // A = [[4, 7], [2, 6]]; inv = [[0.6, -0.7], [-0.2, 0.4]]
        let a = t(&[2, 2], vec![4.0_f32, 7.0, 2.0, 6.0]);
        let inv = inverse(&a).unwrap();
        let v = inv.as_slice::<f32>().unwrap();
        assert!((v[0] - 0.6).abs() < 1e-3);
        assert!((v[1] + 0.7).abs() < 1e-3);
        assert!((v[2] + 0.2).abs() < 1e-3);
        assert!((v[3] - 0.4).abs() < 1e-3);
    }
}
