//! Fused matmul+bias+activation kernels.
//!
//! Computes `y = activation(x @ w + b)` via a fast two-stage pipeline:
//!
//! 1. **Matmul stage**: dispatched through the SIMD/AMX-vectorised
//!    [`crate::accelerate::sgemm_row_major`] (macOS, AMX) or the
//!    `gemm` crate (other platforms, NEON+AVX-512). Below
//!    [`GEMM_DISPATCH_MIN`] = 32 we use a scalar fused inner kernel
//!    (single pass over the output, accumulator in register, bias and
//!    activation folded in) — this beats the BLAS dispatch cost for
//!    tiny shapes that fit in L1d.
//!
//! 2. **Epilogue stage** (only for the BLAS path): a single SIMD-friendly
//!    pass over `y[m * n]` applies bias broadcast (per-column) and the
//!    activation. One read/write per element, LLVM auto-vectorises the
//!    inner loop.
//!
//! ### Why two stages instead of one
//!
//! Before P3.X T2.5 this kernel ran a hand-rolled scalar 3-loop matmul
//! that ignored both `gemm` and `cblas_sgemm`. Measured 911 ms on
//! 1024³ vs 1.62 ms for the dispatched matmul alone — a 562× regression
//! that silently invalidated all `Linear+ReLU` layers (cf. gotcha note
//! `b345ef4b`). The fix routes the matmul through the same dispatch as
//! `CpuBackend::matmul`; the bias+activation epilogue stays in this
//! crate (it is 100% bandwidth-bound and a separate pass is fine on
//! M×N output that already fits in L2 after the GEMM).
//!
//! For L1d-resident shapes (m, k, n all < 32) the scalar single-pass
//! fused kernel is preserved as it remains the fastest option there.

use std::cmp::Ordering;

/// Threshold below which the BLAS dispatch overhead exceeds the work.
/// Calibrated empirically on Apple M4 Max; matches the value used by
/// `rustorch-cpu::cpu_backend::GEMM_DISPATCH_MIN`.
const GEMM_DISPATCH_MIN: usize = 32;

/// Activation choices for the fused epilogue.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Activation {
    /// Identity (no activation).
    None,
    /// max(0, x).
    Relu,
    /// 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³))) — tanh
    /// approximation, common for transformer FFNs.
    Gelu,
    /// x * sigmoid(x).
    Silu,
}

impl Activation {
    /// Apply the activation to a single scalar.
    #[inline]
    pub fn apply(self, x: f32) -> f32 {
        match self {
            Activation::None => x,
            Activation::Relu => x.max(0.0),
            Activation::Gelu => {
                let inner = (2.0_f32 / std::f32::consts::PI).sqrt() * (x + 0.044715 * x * x * x);
                0.5 * x * (1.0 + inner.tanh())
            },
            Activation::Silu => x * sigmoid(x),
        }
    }
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Errors raised by the fused kernel.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FusionError {
    /// Buffer length doesn't match `(m, k, n)`.
    ShapeMismatch {
        /// Which buffer.
        which: &'static str,
        /// Expected element count.
        expected: usize,
        /// Actual count.
        got: usize,
    },
}

impl core::fmt::Display for FusionError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            FusionError::ShapeMismatch {
                which,
                expected,
                got,
            } => write!(f, "{which} buffer has {got} elements, expected {expected}"),
        }
    }
}

impl std::error::Error for FusionError {}

/// Fused `y = activation(x @ w + b)` in a single pass.
///
/// - `x` is `[m, k]` row-major
/// - `w` is `[k, n]` row-major
/// - `b` is `[n]` (broadcast over batch m)
/// - `y` is `[m, n]` row-major
#[allow(clippy::too_many_arguments)]
#[allow(clippy::needless_range_loop)]
pub fn fused_matmul_bias_activation(
    x: &[f32],
    w: &[f32],
    b: Option<&[f32]>,
    y: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    activation: Activation,
) -> Result<(), FusionError> {
    if x.len() != m * k {
        return Err(FusionError::ShapeMismatch {
            which: "x",
            expected: m * k,
            got: x.len(),
        });
    }
    if w.len() != k * n {
        return Err(FusionError::ShapeMismatch {
            which: "w",
            expected: k * n,
            got: w.len(),
        });
    }
    if y.len() != m * n {
        return Err(FusionError::ShapeMismatch {
            which: "y",
            expected: m * n,
            got: y.len(),
        });
    }
    if let Some(bias) = b {
        if bias.len() != n {
            return Err(FusionError::ShapeMismatch {
                which: "b",
                expected: n,
                got: bias.len(),
            });
        }
    }

    if m == 0 || n == 0 {
        return Ok(());
    }

    // Fast path: BLAS-vectorised matmul + tight epilogue. Used when
    // total work (m·n·k FLOPs) is large enough to amortise the BLAS
    // dispatch overhead. T33: previously gated on
    //   `m >= 32 && n >= 32 && k >= 32`, which fell through to the
    // scalar 3-loop on M=1 (LM head single-token decode):
    //   M=1, K=768, N=50257 = 38 M FLOPs — at ~1 GFLOPS scalar that
    // takes 38 ms instead of the ~1.5 ms cBLAS sgemv hits, a ~25×
    // single-token decode regression. Now we dispatch on FLOP count
    // alone (>= 1 M FLOPs ≈ 64³, the original break-even point) so
    // matrix-vector / vector-matrix patterns (LM head, attention
    // proj on S=1) reach the BLAS path.
    const GEMM_DISPATCH_MIN_FLOPS: usize = 1_000_000;
    if m * n * k >= GEMM_DISPATCH_MIN_FLOPS {
        matmul_dispatch(x, w, y, m, k, n);
        apply_bias_activation_epilogue(y, b, m, n, activation);
        return Ok(());
    }
    // Tiny shapes (m·n·k < 1 M) where dispatch overhead exceeds work.
    // The all-dims < 32 cases are also covered here.
    let _ = GEMM_DISPATCH_MIN;

    // Slow path (tiny shapes fit in L1d): single-pass scalar fused
    // kernel. Beats the BLAS dispatch cost for L1d-resident
    // workloads (m·n·k < ~30 K). LLVM auto-vectorises the inner-K
    // accumulator loop on `target-cpu=native`.
    for row in 0..m {
        for col in 0..n {
            let mut acc = b.map(|bb| bb[col]).unwrap_or(0.0);
            for kk in 0..k {
                acc += x[row * k + kk] * w[kk * n + col];
            }
            y[row * n + col] = activation.apply(acc);
        }
    }
    Ok(())
}

/// Dispatch the `[m, k] @ [k, n] -> [m, n]` matmul through the best
/// available BLAS path. macOS hits Apple AMX via Accelerate; other
/// non-wasm targets use the `gemm` crate (NEON / AVX-512). wasm32
/// falls back to the scalar nested-loop kernel.
#[cfg(all(target_os = "macos", not(target_arch = "wasm32")))]
fn matmul_dispatch(x: &[f32], w: &[f32], y: &mut [f32], m: usize, k: usize, n: usize) {
    // SAFETY: caller validated x.len() == m*k, w.len() == k*n, y.len() == m*n.
    unsafe {
        crate::accelerate::sgemm_row_major(m, k, n, x, w, y);
    }
}

#[cfg(all(not(target_os = "macos"), not(target_arch = "wasm32")))]
fn matmul_dispatch(x: &[f32], w: &[f32], y: &mut [f32], m: usize, k: usize, n: usize) {
    // Row-major [m, k] @ [k, n] -> [m, n]:
    //   strides for the gemm crate (in elements):
    //     - x: rs=k, cs=1
    //     - w: rs=n, cs=1
    //     - y: rs=n, cs=1
    // SAFETY: buffers are exactly m*k, k*n, m*n long in f32 and live for
    // the duration of the call. The gemm crate is `unsafe fn` because it
    // works through raw pointers, not because of additional invariants.
    unsafe {
        gemm::gemm(
            m,
            n,
            k,
            y.as_mut_ptr(),
            1,
            n as isize,
            false,
            x.as_ptr(),
            1,
            k as isize,
            w.as_ptr(),
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

#[cfg(target_arch = "wasm32")]
#[allow(clippy::needless_range_loop)]
fn matmul_dispatch(x: &[f32], w: &[f32], y: &mut [f32], m: usize, k: usize, n: usize) {
    // No SIMD intrinsics on wasm32; LLVM still auto-vectorises with
    // simd128 enabled, but we avoid pulling in `gemm` (no wasm support).
    for row in 0..m {
        for col in 0..n {
            let mut acc = 0.0f32;
            for kk in 0..k {
                acc += x[row * k + kk] * w[kk * n + col];
            }
            y[row * n + col] = acc;
        }
    }
}

/// Apply bias broadcast (per-column) and activation in a single
/// SIMD-friendly pass over the output buffer.
///
/// One read + one write per element. LLVM auto-vectorises the inner
/// column loop on `target-cpu=native` for `Activation::None` and
/// `Activation::Relu`; transcendental activations (Gelu, Silu) are
/// scalar-per-lane today (T8-new pulp pass will replace `.exp()` and
/// `.tanh()` with vectorised approximations).
#[inline]
fn apply_bias_activation_epilogue(
    y: &mut [f32],
    b: Option<&[f32]>,
    m: usize,
    n: usize,
    activation: Activation,
) {
    match (b, activation) {
        (None, Activation::None) => { /* matmul output is final */ },
        (None, act) => {
            for row in 0..m {
                let row_slice = &mut y[row * n..row * n + n];
                for cell in row_slice.iter_mut() {
                    *cell = act.apply(*cell);
                }
            }
        },
        (Some(bias), Activation::None) => {
            for row in 0..m {
                let row_slice = &mut y[row * n..row * n + n];
                for (cell, bb) in row_slice.iter_mut().zip(bias.iter()) {
                    *cell += *bb;
                }
            }
        },
        (Some(bias), Activation::Relu) => {
            // The hottest fused path in transformer FFNs. Auto-
            // vectorised `add + max(0)` on NEON/AVX2.
            for row in 0..m {
                let row_slice = &mut y[row * n..row * n + n];
                for (cell, bb) in row_slice.iter_mut().zip(bias.iter()) {
                    let v = *cell + *bb;
                    *cell = v.max(0.0);
                }
            }
        },
        (Some(bias), act) => {
            for row in 0..m {
                let row_slice = &mut y[row * n..row * n + n];
                for (cell, bb) in row_slice.iter_mut().zip(bias.iter()) {
                    *cell = act.apply(*cell + *bb);
                }
            }
        },
    }
}

/// Naive reference (matmul → bias add → activation as 3 separate
/// passes). Used by tests + benches as the non-fused baseline.
#[allow(clippy::too_many_arguments)]
#[allow(clippy::needless_range_loop)]
pub fn naive_matmul_bias_activation(
    x: &[f32],
    w: &[f32],
    b: Option<&[f32]>,
    y: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    activation: Activation,
) -> Result<(), FusionError> {
    // matmul into y
    for row in 0..m {
        for col in 0..n {
            let mut acc = 0.0f32;
            for kk in 0..k {
                acc += x[row * k + kk] * w[kk * n + col];
            }
            y[row * n + col] = acc;
        }
    }
    // bias add (separate pass)
    if let Some(bias) = b {
        for row in 0..m {
            for col in 0..n {
                y[row * n + col] += bias[col];
            }
        }
    }
    // activation (separate pass)
    for cell in y.iter_mut() {
        *cell = activation.apply(*cell);
    }
    let _ = (x, w, k, Ordering::Equal); // silence unused warnings on reorderings
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f32, b: f32, atol: f32) -> bool {
        (a - b).abs() <= atol + 1e-5 * a.abs().max(b.abs())
    }

    fn fixture(m: usize, k: usize, n: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let x: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.1).sin()).collect();
        let w: Vec<f32> = (0..k * n).map(|i| (i as f32 * 0.07).cos()).collect();
        let b: Vec<f32> = (0..n).map(|i| i as f32 * 0.05).collect();
        (x, w, b)
    }

    #[test]
    fn fused_matches_naive_relu() {
        let (m, k, n) = (4, 8, 6);
        let (x, w, b) = fixture(m, k, n);
        let mut y_f = vec![0.0f32; m * n];
        let mut y_n = vec![0.0f32; m * n];
        fused_matmul_bias_activation(&x, &w, Some(&b), &mut y_f, m, k, n, Activation::Relu)
            .unwrap();
        naive_matmul_bias_activation(&x, &w, Some(&b), &mut y_n, m, k, n, Activation::Relu)
            .unwrap();
        for (a, b) in y_f.iter().zip(y_n.iter()) {
            assert!(close(*a, *b, 1e-5), "fused {a} naive {b}");
        }
    }

    #[test]
    fn fused_matches_naive_gelu() {
        let (m, k, n) = (3, 16, 4);
        let (x, w, b) = fixture(m, k, n);
        let mut y_f = vec![0.0f32; m * n];
        let mut y_n = vec![0.0f32; m * n];
        fused_matmul_bias_activation(&x, &w, Some(&b), &mut y_f, m, k, n, Activation::Gelu)
            .unwrap();
        naive_matmul_bias_activation(&x, &w, Some(&b), &mut y_n, m, k, n, Activation::Gelu)
            .unwrap();
        for (a, b) in y_f.iter().zip(y_n.iter()) {
            assert!(close(*a, *b, 1e-5), "fused {a} naive {b}");
        }
    }

    #[test]
    fn fused_matches_naive_silu() {
        let (m, k, n) = (2, 4, 3);
        let (x, w, b) = fixture(m, k, n);
        let mut y_f = vec![0.0f32; m * n];
        let mut y_n = vec![0.0f32; m * n];
        fused_matmul_bias_activation(&x, &w, Some(&b), &mut y_f, m, k, n, Activation::Silu)
            .unwrap();
        naive_matmul_bias_activation(&x, &w, Some(&b), &mut y_n, m, k, n, Activation::Silu)
            .unwrap();
        for (a, b) in y_f.iter().zip(y_n.iter()) {
            assert!(close(*a, *b, 1e-5));
        }
    }

    #[test]
    fn no_bias_is_supported() {
        let (m, k, n) = (2, 4, 2);
        let (x, w, _) = fixture(m, k, n);
        let mut y_f = vec![0.0f32; m * n];
        fused_matmul_bias_activation(&x, &w, None, &mut y_f, m, k, n, Activation::None).unwrap();
        // Spot-check: y should not be all zeros.
        assert!(y_f.iter().any(|&v| v != 0.0));
    }

    #[test]
    fn shape_mismatch_returns_error() {
        let mut y = vec![0.0f32; 4];
        let err = fused_matmul_bias_activation(
            &[1.0; 6],
            &[1.0; 8],
            None,
            &mut y,
            2,
            4,
            2,
            Activation::None,
        )
        .unwrap_err();
        assert!(matches!(err, FusionError::ShapeMismatch { which: "x", .. }));
    }

    #[test]
    fn empty_input_returns_empty_output() {
        let mut y: Vec<f32> = Vec::new();
        fused_matmul_bias_activation(&[], &[], None, &mut y, 0, 0, 0, Activation::Relu).unwrap();
        assert!(y.is_empty());
    }

    #[test]
    fn nan_input_propagates_through_non_relu_activation() {
        // Relu uses f32::max which MASKS NaN to 0.0 — use None
        // (identity) for genuine NaN propagation. Shape: m=2, k=2, n=2.
        let mut x = vec![0.0f32; 4];
        x[0] = f32::NAN;
        let w = vec![1.0f32; 4];
        let mut y = vec![0.0f32; 4];
        fused_matmul_bias_activation(&x, &w, None, &mut y, 2, 2, 2, Activation::None).unwrap();
        // Row 0 cells touch x[0] (NaN) → should be NaN.
        assert!(y[0].is_nan());
        assert!(y[1].is_nan());
    }

    /// xorshift64* — fixed seed, deterministic, used to seed parity
    /// tests so that any future tolerance-violation reproduces.
    fn xorshift_fill(buf: &mut [f32], seed: u64) {
        let mut s = seed;
        for cell in buf.iter_mut() {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            // Map the low 24 bits into [-1, 1) — keeps GEMM accumulators
            // in a tame range so naive scalar and Accelerate AMX agree
            // to ~1e-4 (the upper bound documented in
            // accelerate::tests::sgemm_row_major_64x64x64_matches_naive).
            let bits = (s as u32) & 0x00FF_FFFF;
            *cell = (bits as f32) / 8_388_608.0_f32 - 1.0;
        }
    }

    /// Force the BLAS fast path (m, n, k ≥ 32) and assert it stays
    /// within tolerance of the naive scalar 3-loop reference.
    /// Regression test for P3.X T2.5 — without the dispatch fix, the
    /// "fused" kernel was 562× slower on 1024³ but still numerically
    /// correct, so we also need a perf-independent parity guarantee.
    #[test]
    fn fast_path_relu_matches_naive_64x64x64() {
        let (m, k, n) = (64, 64, 64);
        let mut x = vec![0.0f32; m * k];
        let mut w = vec![0.0f32; k * n];
        let mut bias = vec![0.0f32; n];
        xorshift_fill(&mut x, 0xDEADBEEF);
        xorshift_fill(&mut w, 0xCAFEBABE);
        xorshift_fill(&mut bias, 0x12345678);

        let mut y_fast = vec![0.0f32; m * n];
        let mut y_naive = vec![0.0f32; m * n];
        fused_matmul_bias_activation(&x, &w, Some(&bias), &mut y_fast, m, k, n, Activation::Relu)
            .unwrap();
        naive_matmul_bias_activation(&x, &w, Some(&bias), &mut y_naive, m, k, n, Activation::Relu)
            .unwrap();
        // Tolerance 1e-4 matches the cblas_sgemm parity test in
        // crate::accelerate::tests. AMX uses a different summation
        // tree than the scalar reference so bit-exact is not
        // achievable, but 1e-4 relative is well within float
        // semantics for f32 GEMM.
        for i in 0..m * n {
            let diff = (y_fast[i] - y_naive[i]).abs();
            let rel = diff / y_naive[i].abs().max(1e-6);
            assert!(
                diff <= 1e-4 || rel <= 1e-4,
                "i={i}: fast={} naive={} diff={diff} rel={rel}",
                y_fast[i],
                y_naive[i]
            );
        }
    }

    #[test]
    fn fast_path_no_activation_matches_naive_128x256x96() {
        let (m, k, n) = (128, 256, 96);
        let mut x = vec![0.0f32; m * k];
        let mut w = vec![0.0f32; k * n];
        xorshift_fill(&mut x, 0x1111_2222_3333_4444);
        xorshift_fill(&mut w, 0x5555_6666_7777_8888);

        let mut y_fast = vec![0.0f32; m * n];
        let mut y_naive = vec![0.0f32; m * n];
        fused_matmul_bias_activation(&x, &w, None, &mut y_fast, m, k, n, Activation::None).unwrap();
        naive_matmul_bias_activation(&x, &w, None, &mut y_naive, m, k, n, Activation::None)
            .unwrap();
        for i in 0..m * n {
            let diff = (y_fast[i] - y_naive[i]).abs();
            let rel = diff / y_naive[i].abs().max(1e-6);
            assert!(
                diff <= 1e-3 || rel <= 1e-3,
                "i={i}: fast={} naive={} diff={diff} rel={rel}",
                y_fast[i],
                y_naive[i]
            );
        }
    }

    #[test]
    fn boundary_m_eq_dispatch_min_uses_fast_path() {
        // Just above and just below the dispatch threshold: both
        // should produce numerically equivalent results so callers
        // never see a behaviour shift around m=32.
        for &(m, k, n) in &[(31, 31, 31), (32, 32, 32), (33, 33, 33)] {
            let mut x = vec![0.0f32; m * k];
            let mut w = vec![0.0f32; k * n];
            xorshift_fill(&mut x, 0xAAAA_BBBB_CCCC_DDDD ^ m as u64);
            xorshift_fill(&mut w, 0xEEEE_FFFF_0000_1111 ^ n as u64);
            let mut y_a = vec![0.0f32; m * n];
            let mut y_b = vec![0.0f32; m * n];
            fused_matmul_bias_activation(&x, &w, None, &mut y_a, m, k, n, Activation::Relu)
                .unwrap();
            naive_matmul_bias_activation(&x, &w, None, &mut y_b, m, k, n, Activation::Relu)
                .unwrap();
            for i in 0..m * n {
                let diff = (y_a[i] - y_b[i]).abs();
                let rel = diff / y_b[i].abs().max(1e-6);
                assert!(
                    diff <= 1e-4 || rel <= 1e-4,
                    "shape {m}x{k}x{n} i={i}: fast={} naive={} diff={diff} rel={rel}",
                    y_a[i],
                    y_b[i]
                );
            }
        }
    }

    #[test]
    fn relu_masks_nan_documented() {
        // Document for future maintainers: f32::max(NaN, 0) = 0,
        // so Relu silently sanitises NaN. Use Gelu/Silu/None if you
        // want NaN to flow through.
        let x = vec![f32::NAN, 0.0];
        let w = vec![1.0f32; 2];
        let mut y = vec![0.0f32; 1];
        fused_matmul_bias_activation(&x, &w, None, &mut y, 1, 2, 1, Activation::Relu).unwrap();
        assert_eq!(y[0], 0.0);
    }
}
