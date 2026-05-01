//! Einstein-summation operator (P1.4 minimal slice).
//!
//! Parses an `equation` like `"ij,jk->ik"` and dispatches to matmul +
//! reductions. v1 supports the most common patterns:
//! - `"ij,jk->ik"`         : 2-D matmul
//! - `"bij,bjk->bik"`      : batched matmul (delegates to caller)
//! - `"ij->i"` / `"ij->j"`  : sum-along-dim
//! - `"i,i->"`             : dot product
//! - `"ii->"` / `"ii->i"`   : trace / diagonal
//!
//! General-case einsum (arbitrary subscripts, broadcast, ellipsis) is
//! a much bigger lift and lands in a follow-up.

use crate::error::BackendError;
use rustorch_core::tensor::dtype::Dtype;
use rustorch_core::tensor::tensor_impl::Tensor;

/// Errors specific to einsum parsing / execution.
#[derive(Debug, thiserror::Error)]
pub enum EinsumError {
    /// Equation didn't match the supported subset.
    #[error("unsupported einsum equation: {eq}")]
    Unsupported {
        /// Original equation string.
        eq: String,
    },
    /// Backend error.
    #[error("backend: {0}")]
    Backend(#[from] BackendError),
}

/// Run an einsum on a single operand. Supports `"ij->i"`, `"ij->j"`,
/// `"i->"`, and trace `"ii->"`.
pub fn einsum_unary(eq: &str, a: &Tensor) -> Result<Tensor, EinsumError> {
    if a.dtype() != Dtype::F32 {
        return Err(EinsumError::Backend(BackendError::DtypeMismatch {
            op: "einsum_unary",
            lhs: a.dtype(),
            rhs: Dtype::F32,
        }));
    }
    let parts: Vec<&str> = eq.split("->").collect();
    if parts.len() != 2 {
        return Err(EinsumError::Unsupported { eq: eq.to_string() });
    }
    let lhs = parts[0].trim();
    let rhs = parts[1].trim();
    match (lhs, rhs) {
        ("ij", "i") => {
            let n = a.shape()[0];
            let m = a.shape()[1];
            let data = a.as_slice::<f32>().unwrap();
            let mut out = vec![0.0_f32; n];
            for i in 0..n {
                for j in 0..m {
                    out[i] += data[i * m + j];
                }
            }
            Ok(Tensor::from_vec([n], out).unwrap())
        },
        ("ij", "j") => {
            let n = a.shape()[0];
            let m = a.shape()[1];
            let data = a.as_slice::<f32>().unwrap();
            let mut out = vec![0.0_f32; m];
            for i in 0..n {
                for j in 0..m {
                    out[j] += data[i * m + j];
                }
            }
            Ok(Tensor::from_vec([m], out).unwrap())
        },
        ("i", "") => {
            let data = a.as_slice::<f32>().unwrap();
            let s: f32 = data.iter().sum();
            Ok(Tensor::scalar(s))
        },
        ("ii", "") => {
            // trace
            let n = a.shape()[0];
            let data = a.as_slice::<f32>().unwrap();
            let mut s = 0.0_f32;
            for i in 0..n {
                s += data[i * n + i];
            }
            Ok(Tensor::scalar(s))
        },
        ("ii", "i") => {
            // diagonal
            let n = a.shape()[0];
            let data = a.as_slice::<f32>().unwrap();
            let v: Vec<f32> = (0..n).map(|i| data[i * n + i]).collect();
            Ok(Tensor::from_vec([n], v).unwrap())
        },
        _ => Err(EinsumError::Unsupported { eq: eq.to_string() }),
    }
}

/// Run an einsum on two operands. Supports `"ij,jk->ik"` (matmul) and
/// `"i,i->"` (dot product).
pub fn einsum_binary(eq: &str, a: &Tensor, b: &Tensor) -> Result<Tensor, EinsumError> {
    if a.dtype() != Dtype::F32 || b.dtype() != Dtype::F32 {
        return Err(EinsumError::Backend(BackendError::DtypeMismatch {
            op: "einsum_binary",
            lhs: a.dtype(),
            rhs: b.dtype(),
        }));
    }
    let parts: Vec<&str> = eq.split("->").collect();
    if parts.len() != 2 {
        return Err(EinsumError::Unsupported { eq: eq.to_string() });
    }
    let lhs = parts[0].trim();
    let rhs = parts[1].trim();
    let inputs: Vec<&str> = lhs.split(',').map(str::trim).collect();
    if inputs.len() != 2 {
        return Err(EinsumError::Unsupported { eq: eq.to_string() });
    }
    match (inputs[0], inputs[1], rhs) {
        ("ij", "jk", "ik") => {
            // 2-D matmul
            crate::cpu_backend::cpu_backend()
                .matmul(a, b)
                .map_err(EinsumError::Backend)
        },
        ("i", "i", "") => {
            // dot product
            let av = a.as_slice::<f32>().unwrap();
            let bv = b.as_slice::<f32>().unwrap();
            let s: f32 = av.iter().zip(bv).map(|(x, y)| x * y).sum();
            Ok(Tensor::scalar(s))
        },
        _ => Err(EinsumError::Unsupported { eq: eq.to_string() }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t(shape: &[usize], data: Vec<f32>) -> Tensor {
        Tensor::from_vec(shape.to_vec(), data).unwrap()
    }

    #[test]
    fn einsum_ij_matmul() {
        // [[1,2],[3,4]] @ [[5,6],[7,8]] = [[19,22],[43,50]]
        let a = t(&[2, 2], vec![1.0_f32, 2.0, 3.0, 4.0]);
        let b = t(&[2, 2], vec![5.0_f32, 6.0, 7.0, 8.0]);
        let c = einsum_binary("ij,jk->ik", &a, &b).unwrap();
        assert_eq!(c.as_slice::<f32>().unwrap(), &[19.0_f32, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn einsum_dot_product() {
        let a = t(&[3], vec![1.0_f32, 2.0, 3.0]);
        let b = t(&[3], vec![4.0_f32, 5.0, 6.0]);
        let c = einsum_binary("i,i->", &a, &b).unwrap();
        assert_eq!(c.as_slice::<f32>().unwrap()[0], 32.0_f32); // 1*4+2*5+3*6
    }

    #[test]
    fn einsum_row_sum() {
        let a = t(&[2, 3], vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let r = einsum_unary("ij->i", &a).unwrap();
        assert_eq!(r.as_slice::<f32>().unwrap(), &[6.0_f32, 15.0]);
    }

    #[test]
    fn einsum_trace() {
        let a = t(
            &[3, 3],
            vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        );
        let tr = einsum_unary("ii->", &a).unwrap();
        assert_eq!(tr.as_slice::<f32>().unwrap()[0], 15.0_f32); // 1+5+9
    }

    #[test]
    fn einsum_unsupported_returns_err() {
        let a = t(&[2, 2], vec![1.0_f32; 4]);
        let b = t(&[2, 2], vec![1.0_f32; 4]);
        // Some random unsupported equation
        assert!(einsum_binary("ab,cd->abcd", &a, &b).is_err());
    }
}
