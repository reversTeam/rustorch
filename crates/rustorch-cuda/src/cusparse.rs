//! cuSPARSE bindings — sparse-dense gemm + CSR/COO conversions.
//!
//! Without `--features cuda`, ops fall back to scalar CSR-dense gemm
//! so tests can exercise the math without GPU.

#![allow(clippy::needless_range_loop)]

use crate::error::CudaError;

/// Compressed Sparse Row (CSR) matrix.
#[derive(Debug, Clone)]
pub struct CsrMatrix {
    /// Row count.
    pub rows: usize,
    /// Column count.
    pub cols: usize,
    /// Non-zero values (length = nnz).
    pub values: Vec<f32>,
    /// Column index per non-zero (length = nnz).
    pub col_indices: Vec<usize>,
    /// Row pointer (length = rows + 1; row i has nnz[row_ptr[i]..row_ptr[i+1]]).
    pub row_ptr: Vec<usize>,
}

impl CsrMatrix {
    /// Number of non-zero entries.
    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    /// Empty matrix shape (rows × cols), no non-zeros.
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            values: Vec::new(),
            col_indices: Vec::new(),
            row_ptr: vec![0; rows + 1],
        }
    }
}

/// COO triplets: parallel `(row[i], col[i], value[i])` arrays.
#[derive(Debug, Clone)]
pub struct CooTriplets {
    /// Row indices.
    pub row: Vec<usize>,
    /// Column indices.
    pub col: Vec<usize>,
    /// Non-zero values.
    pub value: Vec<f32>,
    /// Total rows.
    pub rows: usize,
    /// Total cols.
    pub cols: usize,
}

/// Convert COO triplets to CSR. Sorts by (row, col) implicitly via
/// the row_ptr accumulation — caller's COO need not be pre-sorted.
pub fn coo_to_csr(coo: &CooTriplets) -> Result<CsrMatrix, CudaError> {
    if coo.row.len() != coo.col.len() || coo.row.len() != coo.value.len() {
        return Err(CudaError::Unsupported {
            msg: "COO row/col/value length mismatch".into(),
        });
    }
    let nnz = coo.value.len();
    let mut row_ptr = vec![0usize; coo.rows + 1];
    for &r in &coo.row {
        if r >= coo.rows {
            return Err(CudaError::Unsupported {
                msg: format!("COO row {r} out of range (rows={})", coo.rows),
            });
        }
        row_ptr[r + 1] += 1;
    }
    for i in 1..=coo.rows {
        row_ptr[i] += row_ptr[i - 1];
    }
    let mut values = vec![0.0f32; nnz];
    let mut col_indices = vec![0usize; nnz];
    let mut cursor = row_ptr.clone();
    for i in 0..nnz {
        let r = coo.row[i];
        let dst = cursor[r];
        values[dst] = coo.value[i];
        col_indices[dst] = coo.col[i];
        cursor[r] += 1;
    }
    Ok(CsrMatrix {
        rows: coo.rows,
        cols: coo.cols,
        values,
        col_indices,
        row_ptr,
    })
}

/// Convert CSR back to COO triplets. Used by tests for round-trip
/// validation.
pub fn csr_to_coo(csr: &CsrMatrix) -> CooTriplets {
    let nnz = csr.nnz();
    let mut row = Vec::with_capacity(nnz);
    let mut col = Vec::with_capacity(nnz);
    let mut value = Vec::with_capacity(nnz);
    for r in 0..csr.rows {
        for j in csr.row_ptr[r]..csr.row_ptr[r + 1] {
            row.push(r);
            col.push(csr.col_indices[j]);
            value.push(csr.values[j]);
        }
    }
    CooTriplets {
        row,
        col,
        value,
        rows: csr.rows,
        cols: csr.cols,
    }
}

/// Sparse-dense matrix multiplication: `c = csr @ b`.
///
/// `b` is `[csr.cols, n]` row-major; `c` is `[csr.rows, n]`.
pub fn spmm_csr_dense(
    csr: &CsrMatrix,
    b: &[f32],
    c: &mut [f32],
    n: usize,
) -> Result<(), CudaError> {
    if b.len() != csr.cols * n {
        return Err(CudaError::Unsupported {
            msg: format!("b expected {}, got {}", csr.cols * n, b.len()),
        });
    }
    if c.len() != csr.rows * n {
        return Err(CudaError::Unsupported {
            msg: format!("c expected {}, got {}", csr.rows * n, c.len()),
        });
    }
    if csr.nnz() == 0 {
        for v in c.iter_mut() {
            *v = 0.0;
        }
        return Ok(());
    }
    for r in 0..csr.rows {
        for col in 0..n {
            let mut acc = 0.0f32;
            for j in csr.row_ptr[r]..csr.row_ptr[r + 1] {
                let k = csr.col_indices[j];
                acc += csr.values[j] * b[k * n + col];
            }
            c[r * n + col] = acc;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small_csr() -> CsrMatrix {
        // 2×3 matrix:
        // [1, 0, 2]
        // [0, 3, 0]
        CsrMatrix {
            rows: 2,
            cols: 3,
            values: vec![1.0, 2.0, 3.0],
            col_indices: vec![0, 2, 1],
            row_ptr: vec![0, 2, 3],
        }
    }

    #[test]
    fn nnz_counts_values() {
        let csr = small_csr();
        assert_eq!(csr.nnz(), 3);
    }

    #[test]
    fn zeros_csr_has_zero_nnz() {
        let csr = CsrMatrix::zeros(4, 5);
        assert_eq!(csr.nnz(), 0);
        assert_eq!(csr.row_ptr.len(), 5);
    }

    #[test]
    fn coo_to_csr_round_trip_preserves_values() {
        let csr = small_csr();
        let coo = csr_to_coo(&csr);
        let csr2 = coo_to_csr(&coo).unwrap();
        assert_eq!(csr.values, csr2.values);
        assert_eq!(csr.col_indices, csr2.col_indices);
        assert_eq!(csr.row_ptr, csr2.row_ptr);
    }

    #[test]
    fn coo_out_of_range_row_returns_error() {
        let coo = CooTriplets {
            row: vec![10],
            col: vec![0],
            value: vec![1.0],
            rows: 2,
            cols: 2,
        };
        assert!(coo_to_csr(&coo).is_err());
    }

    #[test]
    fn spmm_csr_dense_correct_for_2x3_x_3x1() {
        let csr = small_csr();
        // b is 3×1: [4, 5, 6]
        let b = vec![4.0f32, 5.0, 6.0];
        let mut c = vec![0.0f32; 2];
        spmm_csr_dense(&csr, &b, &mut c, 1).unwrap();
        // row 0: 1*4 + 2*6 = 16
        // row 1: 3*5 = 15
        assert_eq!(c, vec![16.0, 15.0]);
    }

    #[test]
    fn spmm_empty_csr_yields_zero_output() {
        let csr = CsrMatrix::zeros(2, 3);
        let b = vec![1.0f32; 3];
        let mut c = vec![0.5f32; 2];
        spmm_csr_dense(&csr, &b, &mut c, 1).unwrap();
        assert_eq!(c, vec![0.0, 0.0]);
    }
}
