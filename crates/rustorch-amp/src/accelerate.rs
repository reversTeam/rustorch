//! Minimal in-crate FFI to Apple `Accelerate.framework` `cblas_sgemm`.
//! Used by [`crate::bf16_kernels`] for the up-cast → AMX sgemm path.
//! See `rustorch-cpu::accelerate` for the design rationale.

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

/// Row-major `c := a @ b` (alpha=1, beta=0, no transpose).
///
/// # Safety
/// `a.len() ≥ m·k`, `b.len() ≥ k·n`, `c.len() ≥ m·n`.
pub unsafe fn sgemm_row_major(m: usize, k: usize, n: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    debug_assert!(a.len() >= m * k);
    debug_assert!(b.len() >= k * n);
    debug_assert!(c.len() >= m * n);
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
