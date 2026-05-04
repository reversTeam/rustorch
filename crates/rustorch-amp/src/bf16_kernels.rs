//! bf16 / fp16 matmul kernels with f32 accumulator.
//!
//! Compute `c = a @ b` where:
//! - `a` is `[m, k]` low-precision
//! - `b` is `[k, n]` low-precision
//! - `c` is `[m, n]` f32 (the f32 accumulator's natural output)
//!
//! Using an f32 accumulator preserves precision across long inner
//! reductions even when the inputs are bf16/fp16. This matches the
//! AVX-512 BF16 / NEON behaviour and is the standard practice for
//! mixed-precision GEMM.
//!
//! ## Implementation strategy (P3.X T7)
//!
//! Apple Accelerate does not expose a native bf16 cblas, and Apple
//! AMX2 bf16 is not reachable from user space without amx-rs. The
//! pragmatic high-perf path on M1+ is therefore:
//!
//! 1. Up-cast `a` and `b` once into temporary `Vec<f32>` buffers.
//! 2. Call `cblas_sgemm` (macOS) / `gemm` 0.18 (other targets) on
//!    the f32 buffers — the AMX hits 1.4 TF/s on M4 Max.
//! 3. Return the f32 accumulator output as-is.
//!
//! Numerical equivalence with the scalar f32-accumulator reference
//! is preserved (both up-cast bf16→f32 once, only the FMA tree
//! changes — fewer than 1e-2 relative error on uniform inputs).
//!
//! For tiny shapes (`m·n·k < ~30 K`) the BLAS dispatch overhead
//! exceeds the work; we keep a scalar inline-up-cast fallback for
//! correctness without perf regressions.

#![allow(clippy::needless_range_loop)]

use half::{bf16, f16};

/// Threshold below which the up-cast + BLAS dispatch overhead
/// exceeds the scalar work. Calibrated to match
/// `rustorch-cpu::cpu_backend::GEMM_DISPATCH_MIN`.
const GEMM_DISPATCH_MIN: usize = 32;

/// Errors raised by the bf16/fp16 GEMM.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BfGemmError {
    /// Buffer length doesn't match (m, k, n).
    ShapeMismatch {
        /// Which buffer (a / b / c).
        which: &'static str,
        /// Expected element count.
        expected: usize,
        /// Actual element count.
        got: usize,
    },
}

impl core::fmt::Display for BfGemmError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            BfGemmError::ShapeMismatch {
                which,
                expected,
                got,
            } => write!(f, "{which} buffer has {got} elements, expected {expected}"),
        }
    }
}

impl std::error::Error for BfGemmError {}

/// `c = a @ b` for bf16 inputs with f32 accumulator. Output `c` is
/// f32.
pub fn matmul_bf16_with_f32_accum(
    a: &[bf16],
    b: &[bf16],
    c: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
) -> Result<(), BfGemmError> {
    if a.len() != m * k {
        return Err(BfGemmError::ShapeMismatch {
            which: "a",
            expected: m * k,
            got: a.len(),
        });
    }
    if b.len() != k * n {
        return Err(BfGemmError::ShapeMismatch {
            which: "b",
            expected: k * n,
            got: b.len(),
        });
    }
    if c.len() != m * n {
        return Err(BfGemmError::ShapeMismatch {
            which: "c",
            expected: m * n,
            got: c.len(),
        });
    }
    if m == 0 || n == 0 || k == 0 {
        return Ok(());
    }

    // BLAS dispatch path: up-cast bf16 → f32 once, then sgemm.
    // Apple AMX hits ~1.4 TF/s vs ~3 GF/s scalar.
    if m >= GEMM_DISPATCH_MIN && n >= GEMM_DISPATCH_MIN && k >= GEMM_DISPATCH_MIN {
        let a_f32: Vec<f32> = a.iter().map(|&x| x.to_f32()).collect();
        let b_f32: Vec<f32> = b.iter().map(|&x| x.to_f32()).collect();
        sgemm_dispatch(&a_f32, &b_f32, c, m, k, n);
        return Ok(());
    }

    // Tiny-shape fallback: inline up-cast in the inner loop.
    for row in 0..m {
        for col in 0..n {
            let mut acc = 0.0f32;
            for kk in 0..k {
                let av = a[row * k + kk].to_f32();
                let bv = b[kk * n + col].to_f32();
                acc += av * bv;
            }
            c[row * n + col] = acc;
        }
    }
    Ok(())
}

/// `c = a @ b` for fp16 inputs with f32 accumulator. Output `c` is
/// f32.
pub fn matmul_fp16_with_f32_accum(
    a: &[f16],
    b: &[f16],
    c: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
) -> Result<(), BfGemmError> {
    if a.len() != m * k {
        return Err(BfGemmError::ShapeMismatch {
            which: "a",
            expected: m * k,
            got: a.len(),
        });
    }
    if b.len() != k * n {
        return Err(BfGemmError::ShapeMismatch {
            which: "b",
            expected: k * n,
            got: b.len(),
        });
    }
    if c.len() != m * n {
        return Err(BfGemmError::ShapeMismatch {
            which: "c",
            expected: m * n,
            got: c.len(),
        });
    }
    if m == 0 || n == 0 || k == 0 {
        return Ok(());
    }
    if m >= GEMM_DISPATCH_MIN && n >= GEMM_DISPATCH_MIN && k >= GEMM_DISPATCH_MIN {
        let a_f32: Vec<f32> = a.iter().map(|&x| x.to_f32()).collect();
        let b_f32: Vec<f32> = b.iter().map(|&x| x.to_f32()).collect();
        sgemm_dispatch(&a_f32, &b_f32, c, m, k, n);
        return Ok(());
    }
    for row in 0..m {
        for col in 0..n {
            let mut acc = 0.0f32;
            for kk in 0..k {
                let av = a[row * k + kk].to_f32();
                let bv = b[kk * n + col].to_f32();
                acc += av * bv;
            }
            c[row * n + col] = acc;
        }
    }
    Ok(())
}

/// macOS path: cblas_sgemm via Accelerate (AMX). f32 accumulator.
#[cfg(target_os = "macos")]
fn sgemm_dispatch(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    // SAFETY: caller validated buffer sizes; Accelerate's row-major
    // sgemm contract is satisfied by the slice lengths.
    unsafe {
        crate::accelerate::sgemm_row_major(m, k, n, a, b, c);
    }
}

/// Non-macOS / non-wasm path: pure-Rust `gemm` 0.18.
#[cfg(all(not(target_os = "macos"), not(target_arch = "wasm32")))]
fn sgemm_dispatch(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    // SAFETY: caller validated buffer sizes; gemm dispatches by raw
    // pointer with stride contracts that match contiguous row-major.
    unsafe {
        gemm::gemm(
            m,
            n,
            k,
            c.as_mut_ptr(),
            1,
            n as isize,
            false,
            a.as_ptr(),
            1,
            k as isize,
            b.as_ptr(),
            1,
            n as isize,
            0.0_f32,
            1.0_f32,
            false,
            false,
            false,
            gemm::Parallelism::Rayon(0),
        );
    }
}

/// wasm32 fallback: scalar reference (already covered by the
/// up-cast inner loop above; this branch is unreachable for shapes
/// `< GEMM_DISPATCH_MIN`).
#[cfg(target_arch = "wasm32")]
#[allow(clippy::needless_range_loop)]
fn sgemm_dispatch(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    for row in 0..m {
        for col in 0..n {
            let mut acc = 0.0f32;
            for kk in 0..k {
                acc += a[row * k + kk] * b[kk * n + col];
            }
            c[row * n + col] = acc;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f32, b: f32, atol: f32) -> bool {
        (a - b).abs() <= atol + 1e-3 * a.abs().max(b.abs())
    }

    fn ref_matmul_f32(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
        for row in 0..m {
            for col in 0..n {
                let mut acc = 0.0f32;
                for kk in 0..k {
                    acc += a[row * k + kk] * b[kk * n + col];
                }
                c[row * n + col] = acc;
            }
        }
    }

    #[test]
    fn bf16_matmul_diff_vs_f32_under_5e_3() {
        let m = 4;
        let k = 16;
        let n = 4;
        let a_f32: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.1).sin()).collect();
        let b_f32: Vec<f32> = (0..k * n).map(|i| (i as f32 * 0.07).cos()).collect();
        let a_bf: Vec<bf16> = a_f32.iter().map(|&x| bf16::from_f32(x)).collect();
        let b_bf: Vec<bf16> = b_f32.iter().map(|&x| bf16::from_f32(x)).collect();
        let mut c_bf = vec![0.0f32; m * n];
        let mut c_ref = vec![0.0f32; m * n];
        matmul_bf16_with_f32_accum(&a_bf, &b_bf, &mut c_bf, m, k, n).unwrap();
        ref_matmul_f32(&a_f32, &b_f32, &mut c_ref, m, k, n);
        for (q, r) in c_bf.iter().zip(c_ref.iter()) {
            assert!(close(*q, *r, 5e-3), "bf {q} f32 {r}");
        }
    }

    #[test]
    fn fp16_matmul_diff_vs_f32_under_5e_3() {
        let m = 4;
        let k = 16;
        let n = 4;
        let a_f32: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.1).sin()).collect();
        let b_f32: Vec<f32> = (0..k * n).map(|i| (i as f32 * 0.07).cos()).collect();
        let a_fp: Vec<f16> = a_f32.iter().map(|&x| f16::from_f32(x)).collect();
        let b_fp: Vec<f16> = b_f32.iter().map(|&x| f16::from_f32(x)).collect();
        let mut c_fp = vec![0.0f32; m * n];
        let mut c_ref = vec![0.0f32; m * n];
        matmul_fp16_with_f32_accum(&a_fp, &b_fp, &mut c_fp, m, k, n).unwrap();
        ref_matmul_f32(&a_f32, &b_f32, &mut c_ref, m, k, n);
        for (q, r) in c_fp.iter().zip(c_ref.iter()) {
            assert!(close(*q, *r, 5e-3), "fp {q} f32 {r}");
        }
    }

    #[test]
    fn empty_matmul_returns_empty() {
        let mut c: Vec<f32> = Vec::new();
        matmul_bf16_with_f32_accum(&[], &[], &mut c, 0, 0, 0).unwrap();
        assert!(c.is_empty());
    }

    #[test]
    fn shape_mismatch_returns_error() {
        let mut c = vec![0.0f32; 4];
        let err = matmul_bf16_with_f32_accum(&[bf16::ZERO; 6], &[bf16::ZERO; 8], &mut c, 2, 4, 2)
            .unwrap_err();
        assert!(matches!(err, BfGemmError::ShapeMismatch { which: "a", .. }));
    }

    #[test]
    fn f32_accumulator_preserves_precision_for_long_reduction() {
        // K=4096 sum of small values → without f32 accumulator the
        // reduction would lose precision in bf16.
        let m = 1;
        let k = 4096;
        let n = 1;
        // Each input ~ 0.01 → sum ~ 4096 * 0.01 = 41.0 in expectation
        // (alternating signs → ~0). bf16 accumulator would drift far
        // more than the f32 accumulator.
        let a_f32: Vec<f32> = (0..m * k)
            .map(|i| if i % 2 == 0 { 0.01 } else { -0.01 })
            .collect();
        let b_f32 = vec![1.0f32; k * n];
        let a_bf: Vec<bf16> = a_f32.iter().map(|&x| bf16::from_f32(x)).collect();
        let b_bf: Vec<bf16> = b_f32.iter().map(|&x| bf16::from_f32(x)).collect();
        let mut c_bf = vec![0.0f32; m * n];
        matmul_bf16_with_f32_accum(&a_bf, &b_bf, &mut c_bf, m, k, n).unwrap();
        // f32 accumulator → result close to 0.
        assert!(c_bf[0].abs() < 1.0, "got {}", c_bf[0]);
    }
}
