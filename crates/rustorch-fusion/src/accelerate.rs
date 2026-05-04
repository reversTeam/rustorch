//! Minimal in-crate FFI bindings to Apple's `Accelerate.framework`
//! `cblas_sgemm`. Used by [`patterns::matmul_bias_act`] to drive the
//! matmul portion of the fused kernel through the AMX matrix
//! coprocessor on Apple Silicon.
//!
//! This is a deliberate ~50 LOC duplication of `rustorch-cpu::accelerate`:
//! `rustorch-fusion` is a leaf crate that operates on raw `&[f32]`
//! slices and intentionally does not depend on `rustorch-cpu` (which
//! pulls Tensor + Backend + iterator machinery). Keeping the binding
//! local preserves the leaf-crate invariant; the binding is stable
//! Apple ABI (unchanged since macOS 10.4) so the duplication has no
//! drift cost.
//!
//! Compiled only on `target_os = "macos"`; on other platforms the
//! caller falls back to the `gemm` crate.

use std::os::raw::{c_float, c_int};

const CBLAS_ROW_MAJOR: c_int = 101;
const CBLAS_NO_TRANS: c_int = 111;

#[link(name = "Accelerate", kind = "framework")]
extern "C" {
    #[allow(clippy::too_many_arguments)]
    fn cblas_sgemm(
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
}

/// Row-major `C := A @ B` (alpha=1, beta=0, no transpose).
///
/// # Safety
/// - `a.len() >= m * k`, `b.len() >= k * n`, `c.len() >= m * n`.
/// - The C buffer is fully overwritten (beta=0).
pub unsafe fn sgemm_row_major(m: usize, k: usize, n: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    debug_assert!(a.len() >= m * k, "a too small: {} < {}", a.len(), m * k);
    debug_assert!(b.len() >= k * n, "b too small: {} < {}", b.len(), k * n);
    debug_assert!(c.len() >= m * n, "c too small: {} < {}", c.len(), m * n);
    cblas_sgemm(
        CBLAS_ROW_MAJOR,
        CBLAS_NO_TRANS,
        CBLAS_NO_TRANS,
        m as c_int,
        n as c_int,
        k as c_int,
        1.0,
        a.as_ptr(),
        k as c_int,
        b.as_ptr(),
        n as c_int,
        0.0,
        c.as_mut_ptr(),
        n as c_int,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let (m, k, n) = (64, 64, 64);
        let a: Vec<f32> = (0..m * k)
            .map(|i| ((i as f32 + 1.0) * 0.001).sin())
            .collect();
        let b: Vec<f32> = (0..k * n)
            .map(|i| ((i as f32 + 1.0) * 0.0005).cos())
            .collect();
        let expected = naive(m, k, n, &a, &b);
        let mut got = vec![0.0f32; m * n];
        unsafe { sgemm_row_major(m, k, n, &a, &b, &mut got) };
        for i in 0..m * n {
            let rel = (got[i] - expected[i]).abs() / expected[i].abs().max(1e-6);
            assert!(
                rel < 1e-4,
                "i={i}: got {} vs expected {} rel {rel}",
                got[i],
                expected[i]
            );
        }
    }
}
