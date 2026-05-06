//! Minimal in-crate FFI bindings to Apple's `Accelerate.framework`
//! `cblas_sgemm`. Used by [`crate::cpu_forward`] to drive Flash
//! Attention's score (`Q @ K^T`) and value-accumulator (`P @ V`)
//! sgemms through Apple AMX on Apple Silicon.
//!
//! This is a deliberate ~50 LOC duplication of the same module in
//! `rustorch-cpu` and `rustorch-fusion`: each crate stays a leaf so
//! the dependency graph remains a DAG. The binding is stable Apple
//! ABI (unchanged since macOS 10.4) — no drift cost.

use std::os::raw::{c_float, c_int};

/// CBLAS row-major layout enum value.
pub const CBLAS_ROW_MAJOR: c_int = 101;
/// `op(A) = A`.
pub const CBLAS_NO_TRANS: c_int = 111;
/// `op(A) = A^T`.
pub const CBLAS_TRANS: c_int = 112;

#[link(name = "Accelerate", kind = "framework")]
extern "C" {
    /// `C := alpha · op(A) · op(B) + beta · C` (single precision).
    /// Routes through Apple AMX on M1/M2/M3/M4 for shapes large
    /// enough to amortise tile setup (`m·n·k ≳ 10^5`).
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
}
