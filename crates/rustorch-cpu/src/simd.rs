//! SIMD abstraction (P1.2 task `SIMD abstraction`).
//!
//! v1 ships a **scalar** SIMD shim: a generic `Simd<T, N>` newtype with
//! lane-by-lane add/mul/fma. We rely on rustc's auto-vectorizer (with
//! `target-cpu=native` from `.cargo/config.toml`) to lower these into
//! AVX2/AVX-512/NEON instructions on supported targets.
//!
//! Pulling in `pulp` or `wide` is deliberately deferred — the workspace
//! has no such dep yet, and the elementary kernels in P1.3 don't need
//! the explicit lane-extract API. We can swap in a real SIMD crate
//! behind this same `Simd<T, N>` surface in a later iteration without
//! breaking callers.
//!
//! ```
//! use rustorch_cpu::simd::Simd;
//!
//! let a = Simd::<f32, 8>::splat(1.0);
//! let b = Simd::<f32, 8>::splat(2.0);
//! let c = a + b;
//! assert_eq!(c.to_array(), [3.0_f32; 8]);
//! ```

use core::ops::{Add, Mul, Sub};

/// Width-N lanes of `T`. v1 stores as `[T; N]` and relies on the
/// compiler's auto-vectorizer; the trait surface mirrors what a real
/// portable SIMD impl will offer.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Simd<T: Copy, const N: usize>([T; N]);

impl<T: Copy, const N: usize> Simd<T, N> {
    /// Broadcast a scalar to every lane.
    #[inline]
    pub fn splat(v: T) -> Self {
        Simd([v; N])
    }

    /// Build from an array of lanes.
    #[inline]
    pub fn from_array(a: [T; N]) -> Self {
        Simd(a)
    }

    /// Read out the lanes as a fixed-size array (Copy).
    #[inline]
    pub fn to_array(self) -> [T; N] {
        self.0
    }

    /// Number of lanes. Equal to the const generic.
    #[inline]
    pub const fn lanes() -> usize {
        N
    }
}

impl<T, const N: usize> Add for Simd<T, N>
where
    T: Copy + Add<Output = T>,
{
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        let mut out = self.0;
        // Help the auto-vectorizer recognise this as a fixed-size loop.
        for (o, (a, b)) in out.iter_mut().zip(self.0.iter().zip(rhs.0.iter())) {
            *o = *a + *b;
        }
        Simd(out)
    }
}

impl<T, const N: usize> Sub for Simd<T, N>
where
    T: Copy + Sub<Output = T>,
{
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        let mut out = self.0;
        for (o, (a, b)) in out.iter_mut().zip(self.0.iter().zip(rhs.0.iter())) {
            *o = *a - *b;
        }
        Simd(out)
    }
}

impl<T, const N: usize> Mul for Simd<T, N>
where
    T: Copy + Mul<Output = T>,
{
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        let mut out = self.0;
        for (o, (a, b)) in out.iter_mut().zip(self.0.iter().zip(rhs.0.iter())) {
            *o = *a * *b;
        }
        Simd(out)
    }
}

/// Fused multiply-add: `a * b + c`.
///
/// On `f32`/`f64` this collapses to a hardware FMA instruction when
/// the compile target supports it (e.g. `target-feature=+fma`).
#[inline]
pub fn fma_f32<const N: usize>(a: Simd<f32, N>, b: Simd<f32, N>, c: Simd<f32, N>) -> Simd<f32, N> {
    let mut out = c.0;
    for (i, o) in out.iter_mut().enumerate() {
        *o = a.0[i].mul_add(b.0[i], c.0[i]);
    }
    Simd(out)
}

/// FMA for `f64`.
#[inline]
pub fn fma_f64<const N: usize>(a: Simd<f64, N>, b: Simd<f64, N>, c: Simd<f64, N>) -> Simd<f64, N> {
    let mut out = c.0;
    for (i, o) in out.iter_mut().enumerate() {
        *o = a.0[i].mul_add(b.0[i], c.0[i]);
    }
    Simd(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splat_and_to_array() {
        let s = Simd::<f32, 4>::splat(2.5);
        assert_eq!(s.to_array(), [2.5_f32; 4]);
    }

    #[test]
    fn lanes_const() {
        assert_eq!(Simd::<f32, 8>::lanes(), 8);
    }

    #[test]
    fn add_lanewise() {
        let a = Simd::<f32, 4>::from_array([1.0, 2.0, 3.0, 4.0]);
        let b = Simd::<f32, 4>::from_array([5.0, 6.0, 7.0, 8.0]);
        assert_eq!((a + b).to_array(), [6.0, 8.0, 10.0, 12.0]);
    }

    #[test]
    fn sub_lanewise() {
        let a = Simd::<f32, 4>::from_array([10.0, 20.0, 30.0, 40.0]);
        let b = Simd::<f32, 4>::from_array([1.0, 2.0, 3.0, 4.0]);
        assert_eq!((a - b).to_array(), [9.0, 18.0, 27.0, 36.0]);
    }

    #[test]
    fn mul_lanewise() {
        let a = Simd::<f32, 4>::from_array([2.0, 3.0, 4.0, 5.0]);
        let b = Simd::<f32, 4>::from_array([5.0, 6.0, 7.0, 8.0]);
        assert_eq!((a * b).to_array(), [10.0, 18.0, 28.0, 40.0]);
    }

    #[test]
    fn fma_matches_madd() {
        let a = Simd::<f32, 4>::from_array([1.0, 2.0, 3.0, 4.0]);
        let b = Simd::<f32, 4>::splat(10.0);
        let c = Simd::<f32, 4>::splat(1.0);
        // 1*10+1, 2*10+1, 3*10+1, 4*10+1
        assert_eq!(fma_f32(a, b, c).to_array(), [11.0, 21.0, 31.0, 41.0]);
    }

    #[test]
    fn fma_f64_smoke() {
        let a = Simd::<f64, 2>::from_array([1.5, 2.5]);
        let b = Simd::<f64, 2>::splat(2.0);
        let c = Simd::<f64, 2>::splat(0.0);
        assert_eq!(fma_f64(a, b, c).to_array(), [3.0, 5.0]);
    }

    #[test]
    fn nan_propagates() {
        let a = Simd::<f32, 4>::from_array([f32::NAN, 1.0, 2.0, 3.0]);
        let b = Simd::<f32, 4>::splat(1.0);
        let r = (a + b).to_array();
        assert!(r[0].is_nan());
        assert_eq!(r[1], 2.0);
    }

    #[test]
    fn inf_preserved_through_fma() {
        let a = Simd::<f32, 2>::from_array([f32::INFINITY, -f32::INFINITY]);
        let b = Simd::<f32, 2>::splat(1.0);
        let c = Simd::<f32, 2>::splat(0.0);
        let r = fma_f32(a, b, c).to_array();
        assert_eq!(r[0], f32::INFINITY);
        assert_eq!(r[1], f32::NEG_INFINITY);
    }

    #[test]
    fn copy_and_eq() {
        let a = Simd::<i32, 4>::from_array([1, 2, 3, 4]);
        let b = a;
        assert_eq!(a, b);
    }

    #[test]
    fn debug_format() {
        let _ = format!("{:?}", Simd::<f32, 2>::splat(1.0));
    }

    #[test]
    fn add_against_scalar_loop_property() {
        let a = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = [9.0_f32, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0];
        let v = Simd::<f32, 8>::from_array(a) + Simd::<f32, 8>::from_array(b);
        let mut expected = [0_f32; 8];
        for (i, e) in expected.iter_mut().enumerate() {
            *e = a[i] + b[i];
        }
        assert_eq!(v.to_array(), expected);
    }
}
