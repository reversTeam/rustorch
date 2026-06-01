//! Pulp-dispatched SIMD primitives for the Flash Attention inner
//! softmax loops (P3.X T8-new).
//!
//! ## Why a separate module
//!
//! The tile-level `Q @ K^T` and `P @ V` matmuls inside
//! [`crate::cpu_forward::flash_forward_bh_blas`] already hit Apple AMX
//! peak via `cblas_sgemm`. What's left between them is per-tile
//! softmax housekeeping that LLVM cannot reliably auto-vectorise:
//!
//! - a row-wise **max-reduce** over `bc` scores,
//! - a per-row **rescale** of the output accumulator by `exp(m_prev - m_new)`,
//! - a per-row **exp + sum** over `bc` scores,
//! - a final per-row **multiply-by-`1/l`** write-back over `dim`.
//!
//! Each of these is a simple `n`-element memory-bound pass that
//! [`pulp`]'s `Arch::dispatch` lowers to NEON `f32x4` on Apple Silicon
//! and AVX2 `f32x8` (or AVX-512 `f32x16`) on x86.
//!
//! ## Why not auto-vec
//!
//! The scalar `for r in 0..bc_used { … }` loops compile to clean fadd
//! / fmul on M-series, but the surrounding control flow (boundary
//! checks, alpha = 1.0 short-circuit, exp call) prevents LLVM from
//! widening the body. Dispatching through `pulp` forces the lane
//! width explicitly and keeps the compiler from re-rolling.
//!
//! ## Why not vDSP
//!
//! These rows are tiny (`bc = 64`) — the per-call FFI overhead of
//! `vvexpf` / `vDSP_vsmul` (~250 ns each) would dominate. Pulp stays
//! inlined.
//!
//! The `exp` call is intentionally kept **scalar inside the loop**:
//! pulp 0.22 does not expose a portable lane-wise `exp` and rolling
//! our own polynomial approximation would dilate the diff. The
//! subtract / sum / max / rescale ops around it are the ones that
//! actually move bytes, so SIMD'ing them is where the win lives.

use pulp::{Arch, Simd, WithSimd};

/// Compute `max(row[0..n])` using SIMD lanes. Returns `f32::NEG_INFINITY`
/// when `n == 0` (matching the scalar code path).
#[inline]
pub(crate) fn row_max(arch: Arch, row: &[f32]) -> f32 {
    if row.is_empty() {
        return f32::NEG_INFINITY;
    }
    struct Op<'a>(&'a [f32]);
    impl<'a> WithSimd for Op<'a> {
        type Output = f32;
        #[inline(always)]
        fn with_simd<S: Simd>(self, simd: S) -> f32 {
            let (head, tail) = S::as_simd_f32s(self.0);
            let mut acc_v = simd.splat_f32s(f32::NEG_INFINITY);
            for &v in head {
                acc_v = simd.max_f32s(acc_v, v);
            }
            let mut acc = simd.reduce_max_f32s(acc_v);
            for &x in tail {
                if x > acc {
                    acc = x;
                }
            }
            acc
        }
    }
    arch.dispatch(Op(row))
}

/// Compute `slice[i] *= scalar` for all `i`. Used to rescale the
/// running output accumulator `O` by `alpha = exp(m_prev - m_new)`
/// when a new row maximum arrives.
#[inline]
pub(crate) fn scale_in_place(arch: Arch, slice: &mut [f32], scalar: f32) {
    struct Op<'a> {
        slice: &'a mut [f32],
        scalar: f32,
    }
    impl<'a> WithSimd for Op<'a> {
        type Output = ();
        #[inline(always)]
        fn with_simd<S: Simd>(self, simd: S) {
            let scalar_v = simd.splat_f32s(self.scalar);
            let (head, tail) = S::as_mut_simd_f32s(self.slice);
            for v in head {
                *v = simd.mul_f32s(*v, scalar_v);
            }
            for v in tail {
                *v *= self.scalar;
            }
        }
    }
    arch.dispatch(Op { slice, scalar });
}

/// Compute `dst[i] = exp(src[i] - m_new)` and return `sum(dst)`.
///
/// We keep this routine **monomorphic and scalar** — `pulp` 0.22 does
/// not provide a lane-wise `exp` and the compiler's per-lane libm call
/// is already heavily optimised on M-series via libsystem_m's
/// vectorised `expf`. We expose it through this module anyway so the
/// inner softmax pass has a single SIMD-flavoured call surface
/// (matching `row_max` / `scale_in_place` / `mul_scalar`), and so a
/// future polynomial-approx pulp pass can swap implementations
/// behind the same signature without touching callers.
///
/// `dst` and `src` MUST have the same length.
#[inline]
pub(crate) fn exp_minus_max_and_sum(_arch: Arch, dst: &mut [f32], src: &[f32], m_new: f32) -> f32 {
    debug_assert_eq!(dst.len(), src.len());
    let mut tile_l = 0.0f32;
    for (d, &s) in dst.iter_mut().zip(src.iter()) {
        let e = (s - m_new).exp();
        *d = e;
        tile_l += e;
    }
    tile_l
}

/// Compute `dst[i] = src[i] * scalar`. Used for the final
/// write-back normalise: `out[i] = o_buf[i] * (1/l)`.
#[inline]
pub(crate) fn mul_scalar(arch: Arch, dst: &mut [f32], src: &[f32], scalar: f32) {
    debug_assert_eq!(dst.len(), src.len());
    struct Op<'a> {
        dst: &'a mut [f32],
        src: &'a [f32],
        scalar: f32,
    }
    impl<'a> WithSimd for Op<'a> {
        type Output = ();
        #[inline(always)]
        fn with_simd<S: Simd>(self, simd: S) {
            let scalar_v = simd.splat_f32s(self.scalar);
            let (src_head, src_tail) = S::as_simd_f32s(self.src);
            let (dst_head, dst_tail) = S::as_mut_simd_f32s(self.dst);
            for (sv, dv) in src_head.iter().zip(dst_head.iter_mut()) {
                *dv = simd.mul_f32s(*sv, scalar_v);
            }
            for (s, d) in src_tail.iter().zip(dst_tail.iter_mut()) {
                *d = *s * self.scalar;
            }
        }
    }
    arch.dispatch(Op { dst, src, scalar });
}

/// Zero `slice[..]`. Equivalent to `slice.fill(0.0)` but routed
/// through SIMD stores so it composes cleanly with the rest of the
/// inner-loop dispatch.
#[inline]
pub(crate) fn fill_zero(arch: Arch, slice: &mut [f32]) {
    struct Op<'a>(&'a mut [f32]);
    impl<'a> WithSimd for Op<'a> {
        type Output = ();
        #[inline(always)]
        fn with_simd<S: Simd>(self, simd: S) {
            let zero_v = simd.splat_f32s(0.0f32);
            let (head, tail) = S::as_mut_simd_f32s(self.0);
            for v in head {
                *v = zero_v;
            }
            for v in tail {
                *v = 0.0;
            }
        }
    }
    arch.dispatch(Op(slice));
}

#[cfg(test)]
mod tests {
    use super::*;

    fn arch() -> Arch {
        Arch::new()
    }

    #[test]
    fn row_max_basic() {
        let v = [1.0f32, 3.5, -2.0, 7.25, 4.0, 7.25];
        assert_eq!(row_max(arch(), &v), 7.25);
    }

    #[test]
    fn row_max_empty_is_neg_inf() {
        assert_eq!(row_max(arch(), &[]), f32::NEG_INFINITY);
    }

    #[test]
    fn row_max_long_slice_matches_scalar() {
        let n = 1024usize;
        let v: Vec<f32> = (0..n).map(|i| (i as f32 * 0.37).sin()).collect();
        let want = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let got = row_max(arch(), &v);
        assert!((got - want).abs() < 1e-6, "got={got} want={want}");
    }

    #[test]
    fn scale_in_place_basic() {
        let mut v = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        scale_in_place(arch(), &mut v, 0.5);
        assert_eq!(v, vec![0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]);
    }

    #[test]
    fn exp_minus_max_and_sum_basic() {
        let src = [0.0f32, 1.0, 2.0];
        let mut dst = [0.0f32; 3];
        let m = 2.0f32;
        let sum = exp_minus_max_and_sum(arch(), &mut dst, &src, m);
        let want: Vec<f32> = src.iter().map(|x| (x - m).exp()).collect();
        let want_sum: f32 = want.iter().sum();
        for (a, b) in dst.iter().zip(want.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
        assert!((sum - want_sum).abs() < 1e-6);
    }

    #[test]
    fn mul_scalar_basic() {
        let src = [2.0f32, 4.0, 6.0, 8.0, 10.0];
        let mut dst = [0.0_f32; 5];
        mul_scalar(arch(), &mut dst, &src, 0.25);
        assert_eq!(dst, [0.5, 1.0, 1.5, 2.0, 2.5]);
    }

    #[test]
    fn fill_zero_basic() {
        let mut v = vec![1.0_f32; 17];
        fill_zero(arch(), &mut v);
        assert!(v.iter().all(|&x| x == 0.0));
    }
}
