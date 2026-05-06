//! Minimal FFI bindings to Apple's `Accelerate.framework` BLAS surface.
//!
//! On Apple Silicon (M1+) the `cblas_sgemm` symbol from Accelerate
//! routes f32 matmul through the AMX matrix coprocessor — typically
//! 5-10× faster than pure-Rust `gemm`-rs on the same hardware. PyTorch
//! CPU on macOS uses the same path (since 2.3+), so we only beat them
//! by trimming dispatch overhead elsewhere; without Accelerate we lose
//! the BLAS race outright.
//!
//! Bindings are kept tiny (just `cblas_sgemm` + the constants we need)
//! to avoid pulling in a heavyweight crate. Links the framework via
//! `#[link(name = "Accelerate", kind = "framework")]`; no `accelerate-src`
//! / `cblas-sys` build dep.
//!
//! ## Why a hand-rolled binding
//!
//! - `accelerate-src` is empty (just `#[link]`); we'd still need to
//!   declare the externs ourselves.
//! - `cblas-sys` exposes the full ~200-symbol cblas surface, much of
//!   which we don't need; bringing it in just for `sgemm` is overkill.
//! - The binding is stable Apple ABI — has been the same since
//!   macOS 10.4. No churn risk.

use std::os::raw::{c_float, c_int};

/// CBLAS layout constants (verbatim from `<cblas.h>` Apple SDK).
pub const CBLAS_ROW_MAJOR: c_int = 101;
/// `op(A) = A`, no transpose.
pub const CBLAS_NO_TRANS: c_int = 111;
/// `op(A) = A^T`.
pub const CBLAS_TRANS: c_int = 112;

#[link(name = "Accelerate", kind = "framework")]
extern "C" {
    /// Single-precision GEMM: `C := alpha · op(A) · op(B) + beta · C`.
    ///
    /// All matrices are interpreted in **row-major** when `order =
    /// CBLAS_ROW_MAJOR`. `lda` / `ldb` / `ldc` are leading dimensions
    /// in elements (not bytes).
    ///
    /// Apple's Accelerate `cblas_sgemm` dispatches to AMX tile units
    /// transparently on Apple Silicon for shapes large enough to
    /// amortise the AMX setup cost (typically m·n·k ≳ 10⁵).
    #[allow(clippy::too_many_arguments)]
    pub fn cblas_sgemm(
        order: c_int,
        transa: c_int,
        transb: c_int,
        m: c_int,
        n: c_int,
        k: c_int,
        alpha: c_float,
        a: *const c_float,
        lda: c_int,
        b: *const c_float,
        ldb: c_int,
        beta: c_float,
        c: *mut c_float,
        ldc: c_int,
    );

    /// Single-precision matrix-vector product: `y := alpha · op(A) · x + beta · y`.
    ///
    /// Critical for the M=1 (or N=1) matmul case in LLM single-token
    /// decode (LM head projection on a single query row), where
    /// `cblas_sgemm` falls back to a sub-optimal generic dispatch on
    /// Apple Silicon. `cblas_sgemv` instead drives a tuned bandwidth-
    /// bound kernel that saturates DRAM throughput for the
    /// matrix-vector pattern.
    #[allow(clippy::too_many_arguments)]
    pub fn cblas_sgemv(
        order: c_int,
        trans: c_int,
        m: c_int,
        n: c_int,
        alpha: c_float,
        a: *const c_float,
        lda: c_int,
        x: *const c_float,
        incx: c_int,
        beta: c_float,
        y: *mut c_float,
        incy: c_int,
    );
}

/// Safe Rust wrapper for the row-major `C = A @ B` case (no transpose,
/// alpha=1, beta=0). Caller must guarantee that the slices match the
/// promised dimensions (m·k for A, k·n for B, m·n for C).
///
/// # Safety
/// - `a.len() >= m * k`, `b.len() >= k * n`, `c.len() >= m * n`.
/// - The C buffer is fully overwritten (beta=0), so its prior contents
///   don't need to be initialised.
pub unsafe fn sgemm_row_major(m: usize, k: usize, n: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    debug_assert!(a.len() >= m * k, "a too small: {} < {}", a.len(), m * k);
    debug_assert!(b.len() >= k * n, "b too small: {} < {}", b.len(), k * n);
    debug_assert!(c.len() >= m * n, "c too small: {} < {}", c.len(), m * n);
    // T33 — dispatch M=1 to sgemv. cblas_sgemm has a generic dispatch
    // path on M=1 that bottlenecks at ~2 GB/s (40 ms for an LM head
    // [1, 768] @ [768, 50257]); cblas_sgemv saturates DRAM
    // bandwidth (~80 GB/s) for the matrix-vector pattern, dropping
    // the same op to ~2 ms.
    if m == 1 {
        sgemv_row_major_m1(k, n, a, b, c);
        return;
    }
    cblas_sgemm(
        CBLAS_ROW_MAJOR,
        CBLAS_NO_TRANS,
        CBLAS_NO_TRANS,
        m as c_int,
        n as c_int,
        k as c_int,
        1.0,
        a.as_ptr(),
        k as c_int, // lda = K (row-major A is m×k, stride between rows)
        b.as_ptr(),
        n as c_int, // ldb = N (row-major B is k×n, stride between rows)
        0.0,
        c.as_mut_ptr(),
        n as c_int, // ldc = N
    );
}

/// Safe Rust wrapper for the row-major M=1 matrix-vector product
/// `c[1, n] = a[1, k] @ b[k, n]`. Internally drives `cblas_sgemv`
/// with the B matrix transposed: when `op(A) = A^T`, the output is
/// `y[n] = A^T[n, k] @ x[k]` — which matches our row-major
/// `b @ a^T` interpretation. Faster than `sgemm` with M=1 because
/// the gemv kernel is bandwidth-tuned for the matrix-vector case.
///
/// # Safety
/// - `a.len() >= k`, `b.len() >= k * n`, `c.len() >= n`.
/// - The C buffer is fully overwritten (beta=0).
pub unsafe fn sgemv_row_major_m1(k: usize, n: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    debug_assert!(a.len() >= k, "a too small: {} < {}", a.len(), k);
    debug_assert!(b.len() >= k * n, "b too small: {} < {}", b.len(), k * n);
    debug_assert!(c.len() >= n, "c too small: {} < {}", c.len(), n);
    // We want y[n] = b^T @ a where b is row-major [k, n] (so b^T is
    // [n, k]). cblas_sgemv with trans=CBLAS_TRANS treats the input
    // matrix as transposed: y = alpha * A^T @ x + beta * y.
    // Pass b as the matrix (laid out row-major as [k, n], lda=n,
    // m_dim=k, n_dim=n) with trans=CBLAS_TRANS to compute
    // y[n] = b[k,n]^T @ a[k] = a · b.
    cblas_sgemv(
        CBLAS_ROW_MAJOR,
        CBLAS_TRANS,
        k as c_int, // m_dim of source matrix = K rows of B
        n as c_int, // n_dim of source matrix = N cols of B
        1.0,
        b.as_ptr(),
        n as c_int, // lda = N (row-major B has stride N between rows)
        a.as_ptr(),
        1, // incx = 1 (a is contiguous f32)
        0.0,
        c.as_mut_ptr(),
        1, // incy = 1
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Naive reference matmul for parity checking.
    fn naive(m: usize, k: usize, n: usize, a: &[f32], b: &[f32]) -> Vec<f32> {
        let mut c = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0f32;
                for kk in 0..k {
                    acc += a[i * k + kk] * b[kk * n + j];
                }
                c[i * n + j] = acc;
            }
        }
        c
    }

    #[test]
    fn sgemm_row_major_64x64x64_matches_naive() {
        let m = 64;
        let k = 64;
        let n = 64;
        let a: Vec<f32> = (0..m * k)
            .map(|i| ((i as f32 + 1.0) * 0.001).sin())
            .collect();
        let b: Vec<f32> = (0..k * n)
            .map(|i| ((i as f32 + 1.0) * 0.0005).cos())
            .collect();
        let expected = naive(m, k, n, &a, &b);
        let mut got = vec![0.0f32; m * n];
        // SAFETY: buffers sized m*k, k*n, m*n.
        unsafe {
            sgemm_row_major(m, k, n, &a, &b, &mut got);
        }
        for i in 0..m * n {
            let rel = (got[i] - expected[i]).abs() / expected[i].abs().max(1e-6);
            assert!(
                rel < 1e-4,
                "i={i}: got {} vs expected {} rel_err {}",
                got[i],
                expected[i],
                rel
            );
        }
    }

    /// T33 — sgemv path correctness on the LM head shape.
    #[test]
    fn sgemm_row_major_m1_dispatches_sgemv_correctly() {
        let m = 1;
        let k = 32;
        let n = 17;
        let a: Vec<f32> = (0..k).map(|i| ((i as f32 + 1.0) * 0.01).sin()).collect();
        let b: Vec<f32> = (0..k * n)
            .map(|i| ((i as f32 + 1.0) * 0.005).cos())
            .collect();
        let expected = naive(m, k, n, &a, &b);
        let mut got = vec![0.0f32; m * n];
        // SAFETY: matches signature.
        unsafe {
            sgemm_row_major(m, k, n, &a, &b, &mut got);
        }
        for i in 0..n {
            let rel = (got[i] - expected[i]).abs() / expected[i].abs().max(1e-6);
            assert!(
                rel < 1e-4,
                "i={i}: got {} vs expected {} rel_err {}",
                got[i],
                expected[i],
                rel
            );
        }
    }
}
