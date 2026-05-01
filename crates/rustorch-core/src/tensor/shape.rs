//! Shape newtype + NumPy-style broadcasting (RFC-0002, P1.1 task `Shape`).
//!
//! [`Shape`] is the canonical container of dimensions for a [`Tensor`].
//! It is a thin newtype around `Vec<usize>` (we use a `Vec` rather than
//! a `SmallVec` in v1 to keep `rustorch-core` dependency-light; the
//! switch to a `SmallVec<[usize; 6]>` is a transparent change behind
//! [`Shape::as_slice`] when benchmarks demand it).
//!
//! # Broadcasting
//!
//! Implements NumPy / PyTorch broadcasting rules:
//!
//! 1. Right-align the two shapes.
//! 2. Walking from the trailing dim leftward, dimensions are
//!    *compatible* when they are equal, or one of them is `1`, or one
//!    is missing (treated as `1`).
//! 3. The resulting shape takes the maximum of the two compatible dims
//!    on each axis.
//!
//! ```
//! use rustorch_core::tensor::shape::Shape;
//!
//! let a = Shape::from([5, 1, 4]);
//! let b = Shape::from([3, 1]);
//! let out = a.broadcast_with(&b).unwrap();
//! assert_eq!(out.as_slice(), &[5, 3, 4]);
//! ```
//!
//! [`Tensor`]: super::Tensor

use core::fmt;

/// Container of tensor dimensions.
///
/// `Shape` is `Clone + Eq + Hash`, suitable as a `HashMap` key (e.g.
/// kernel cache) and as an op-graph identifier. Empty shape means
/// scalar (`numel() == 1`).
#[derive(Debug, Default, Clone, PartialEq, Eq, Hash)]
pub struct Shape(Vec<usize>);

/// Error returned by [`Shape::broadcast_with`] when two shapes cannot
/// be made compatible per NumPy rules.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BroadcastError {
    /// Left operand shape (kept for diagnostic display).
    pub lhs: Vec<usize>,
    /// Right operand shape.
    pub rhs: Vec<usize>,
    /// Index of the axis (right-aligned) where the mismatch occurred.
    pub axis_from_right: usize,
}

impl fmt::Display for BroadcastError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "incompatible shapes for broadcast: {:?} vs {:?} \
             (axis -{} from right)",
            self.lhs,
            self.rhs,
            self.axis_from_right + 1
        )
    }
}

impl std::error::Error for BroadcastError {}

impl Shape {
    /// Build a shape from any iterator of `usize`. The most ergonomic
    /// constructors are the `From<...>` impls — see those.
    #[inline]
    pub fn from_iter_dims<I: IntoIterator<Item = usize>>(iter: I) -> Self {
        Shape(iter.into_iter().collect())
    }

    /// Number of dimensions (rank). Scalar = 0.
    #[inline]
    pub fn ndim(&self) -> usize {
        self.0.len()
    }

    /// Returns `true` iff `ndim() == 0` (scalar).
    #[inline]
    pub fn is_scalar(&self) -> bool {
        self.0.is_empty()
    }

    /// Total number of elements.
    ///
    /// Scalar shape returns `1`; any axis equal to `0` makes the whole
    /// shape return `0`. Matches `torch.numel()`.
    #[inline]
    pub fn numel(&self) -> usize {
        if self.0.is_empty() {
            1
        } else {
            self.0.iter().product()
        }
    }

    /// Read-only access to the dim slice.
    #[inline]
    pub fn as_slice(&self) -> &[usize] {
        &self.0
    }

    /// Consume the shape and return its underlying `Vec<usize>`.
    #[inline]
    pub fn into_vec(self) -> Vec<usize> {
        self.0
    }

    /// NumPy-style broadcasting.
    ///
    /// Returns `Err(BroadcastError)` if the shapes are incompatible. On
    /// success, the resulting shape is `max(lhs[i], rhs[i])` per axis,
    /// after right-alignment.
    ///
    /// ```
    /// # use rustorch_core::tensor::shape::Shape;
    /// // Trailing-dim broadcast
    /// let a = Shape::from([2, 3, 4]);
    /// let b = Shape::from([4]);
    /// assert_eq!(a.broadcast_with(&b).unwrap().as_slice(), &[2, 3, 4]);
    ///
    /// // Two-way broadcast on different axes
    /// let a = Shape::from([5, 1, 4]);
    /// let b = Shape::from([3, 1]);
    /// assert_eq!(a.broadcast_with(&b).unwrap().as_slice(), &[5, 3, 4]);
    ///
    /// // Scalar broadcasts everywhere
    /// let s = Shape::default();
    /// assert_eq!(s.broadcast_with(&Shape::from([3])).unwrap().as_slice(), &[3]);
    ///
    /// // Incompatible
    /// assert!(Shape::from([3]).broadcast_with(&Shape::from([4])).is_err());
    /// ```
    pub fn broadcast_with(&self, other: &Shape) -> Result<Shape, BroadcastError> {
        let lhs = self.as_slice();
        let rhs = other.as_slice();
        let n = lhs.len().max(rhs.len());
        let mut out = Vec::with_capacity(n);

        for i in 0..n {
            // Right-aligned access: dim i (from right) on lhs / rhs, default to 1 if missing.
            let l = lhs.len().checked_sub(1 + i).map(|j| lhs[j]).unwrap_or(1);
            let r = rhs.len().checked_sub(1 + i).map(|j| rhs[j]).unwrap_or(1);
            let dim = match (l, r) {
                (a, b) if a == b => a,
                (1, b) => b,
                (a, 1) => a,
                _ => {
                    return Err(BroadcastError {
                        lhs: lhs.to_vec(),
                        rhs: rhs.to_vec(),
                        axis_from_right: i,
                    });
                },
            };
            out.push(dim);
        }
        out.reverse();
        Ok(Shape(out))
    }

    /// Compute the strides (in *elements*, not bytes) such that
    /// expanding `self` to `target` is implemented by setting the
    /// stride of broadcasted axes to `0`. Returns the strides aligned
    /// to `target`'s rank.
    ///
    /// Used by [`Layout`](super::layout::Layout)/`expand_to`.
    ///
    /// `Err(BroadcastError)` if `self` is not broadcast-compatible with
    /// `target` (every dim must already be `1` or already equal to
    /// the corresponding `target` dim).
    ///
    /// ```
    /// # use rustorch_core::tensor::shape::Shape;
    /// let s = Shape::from([3, 1]);
    /// let target = Shape::from([2, 3, 4]);
    /// assert_eq!(s.expand_strides_to(&target).unwrap(), &[0, 1, 0]);
    /// ```
    pub fn expand_strides_to(&self, target: &Shape) -> Result<Vec<isize>, BroadcastError> {
        let lhs = self.as_slice();
        let rhs = target.as_slice();
        if lhs.len() > rhs.len() {
            return Err(BroadcastError {
                lhs: lhs.to_vec(),
                rhs: rhs.to_vec(),
                axis_from_right: rhs.len(),
            });
        }
        // First, compute the source's element strides under contiguity.
        let mut elem_strides = vec![0isize; lhs.len()];
        if !lhs.is_empty() {
            let mut acc: isize = 1;
            for i in (0..lhs.len()).rev() {
                elem_strides[i] = acc;
                acc *= lhs[i] as isize;
            }
        }
        let mut out = vec![0isize; rhs.len()];
        let offset = rhs.len() - lhs.len();
        for (i, (&src, &dst)) in lhs.iter().zip(&rhs[offset..]).enumerate() {
            if src == dst {
                out[offset + i] = elem_strides[i];
            } else if src == 1 {
                out[offset + i] = 0;
            } else {
                return Err(BroadcastError {
                    lhs: lhs.to_vec(),
                    rhs: rhs.to_vec(),
                    axis_from_right: lhs.len() - 1 - i,
                });
            }
        }
        // Leading axes that don't exist in `lhs` are pure broadcast → stride 0.
        // (Already initialised to 0; nothing to do.)
        Ok(out)
    }
}

// --------------------------------------------------------------------------
// Conversions — ergonomic constructors
// --------------------------------------------------------------------------

impl<const N: usize> From<[usize; N]> for Shape {
    fn from(arr: [usize; N]) -> Shape {
        Shape(arr.to_vec())
    }
}

impl From<&[usize]> for Shape {
    fn from(s: &[usize]) -> Shape {
        Shape(s.to_vec())
    }
}

impl From<Vec<usize>> for Shape {
    fn from(v: Vec<usize>) -> Shape {
        Shape(v)
    }
}

impl<const N: usize> From<&[usize; N]> for Shape {
    fn from(arr: &[usize; N]) -> Shape {
        Shape(arr.to_vec())
    }
}

impl core::ops::Deref for Shape {
    type Target = [usize];
    #[inline]
    fn deref(&self) -> &[usize] {
        &self.0
    }
}

impl AsRef<[usize]> for Shape {
    #[inline]
    fn as_ref(&self) -> &[usize] {
        &self.0
    }
}

impl fmt::Display for Shape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", &self.0)
    }
}

// --------------------------------------------------------------------------
// Tests
// --------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ctors_array_slice_vec() {
        let a = Shape::from([2, 3, 4]);
        let b = Shape::from(&[2usize, 3, 4][..]);
        let c = Shape::from(vec![2usize, 3, 4]);
        assert_eq!(a, b);
        assert_eq!(b, c);
        assert_eq!(a.as_slice(), &[2, 3, 4]);
    }

    #[test]
    fn deref_and_as_ref() {
        let s = Shape::from([2, 3, 4]);
        // Deref to slice
        assert_eq!(s.len(), 3);
        assert_eq!(s.first(), Some(&2));
        assert_eq!(s.last(), Some(&4));
        // AsRef
        let slc: &[usize] = s.as_ref();
        assert_eq!(slc, &[2, 3, 4]);
    }

    #[test]
    fn scalar_shape_is_default_with_numel_one() {
        let s = Shape::default();
        assert!(s.is_scalar());
        assert_eq!(s.ndim(), 0);
        assert_eq!(s.numel(), 1);
        assert_eq!(s.as_slice(), &[] as &[usize]);
    }

    #[test]
    fn empty_dim_zeros_numel() {
        let s = Shape::from([0, 3]);
        assert_eq!(s.numel(), 0);
        assert!(!s.is_scalar());
    }

    #[test]
    fn ndim_and_numel() {
        assert_eq!(Shape::from([2, 3, 4]).ndim(), 3);
        assert_eq!(Shape::from([2, 3, 4]).numel(), 24);
        assert_eq!(Shape::from([7]).ndim(), 1);
        assert_eq!(Shape::from([7]).numel(), 7);
    }

    #[test]
    fn into_vec_round_trip() {
        let s = Shape::from([2, 3]);
        assert_eq!(s.into_vec(), vec![2, 3]);
    }

    #[test]
    fn display_format() {
        assert_eq!(Shape::from([2, 3, 4]).to_string(), "[2, 3, 4]");
        assert_eq!(Shape::default().to_string(), "[]");
    }

    #[test]
    fn broadcast_trailing_dim_only() {
        let a = Shape::from([2, 3, 4]);
        let b = Shape::from([4]);
        assert_eq!(a.broadcast_with(&b).unwrap().as_slice(), &[2, 3, 4]);
        assert_eq!(b.broadcast_with(&a).unwrap().as_slice(), &[2, 3, 4]);
    }

    #[test]
    fn broadcast_two_way_5_1_4_with_3_1() {
        let a = Shape::from([5, 1, 4]);
        let b = Shape::from([3, 1]);
        assert_eq!(a.broadcast_with(&b).unwrap().as_slice(), &[5, 3, 4]);
        assert_eq!(b.broadcast_with(&a).unwrap().as_slice(), &[5, 3, 4]);
    }

    #[test]
    fn broadcast_scalar_against_anything() {
        let s = Shape::default();
        assert_eq!(
            s.broadcast_with(&Shape::from([3])).unwrap().as_slice(),
            &[3]
        );
        assert_eq!(
            Shape::from([2, 5]).broadcast_with(&s).unwrap().as_slice(),
            &[2, 5]
        );
        assert!(s.broadcast_with(&s).unwrap().is_scalar());
    }

    #[test]
    fn broadcast_empty_dims() {
        // Empty tensor [0] broadcasts with [1] → [0]
        let a = Shape::from([0]);
        let b = Shape::from([1]);
        assert_eq!(a.broadcast_with(&b).unwrap().as_slice(), &[0]);
    }

    #[test]
    fn broadcast_incompatible() {
        let a = Shape::from([3]);
        let b = Shape::from([4]);
        let err = a.broadcast_with(&b).unwrap_err();
        assert_eq!(err.axis_from_right, 0);
        assert_eq!(err.lhs, vec![3]);
        assert_eq!(err.rhs, vec![4]);
        // Display should mention both sides.
        let s = err.to_string();
        assert!(s.contains("[3]"));
        assert!(s.contains("[4]"));
    }

    #[test]
    fn broadcast_incompatible_inner_axis() {
        let a = Shape::from([2, 3, 4]);
        let b = Shape::from([5, 4]);
        let err = a.broadcast_with(&b).unwrap_err();
        assert_eq!(err.axis_from_right, 1);
    }

    #[test]
    fn broadcast_commutative_on_random_pairs() {
        // Cross-check commutativity (a.broadcast(b) on success ⇒ b.broadcast(a) succeeds with same result)
        let pairs: &[(&[usize], &[usize])] = &[
            (&[3], &[3]),
            (&[1, 3], &[3]),
            (&[2, 1, 4], &[3, 1]),
            (&[5], &[1]),
            (&[2, 3, 4], &[2, 3, 4]),
            (&[7, 1, 1, 1], &[1, 5, 1, 9]),
        ];
        for (la, lb) in pairs {
            let a = Shape::from(*la);
            let b = Shape::from(*lb);
            let ab = a.broadcast_with(&b).unwrap();
            let ba = b.broadcast_with(&a).unwrap();
            assert_eq!(ab, ba, "non-commutative on ({la:?}, {lb:?})");
        }
    }

    #[test]
    fn broadcast_max_per_axis_sample_500() {
        // Deterministic LCG to walk a fixed but large pair space.
        // This stands in for the proptest fixture without pulling in proptest.
        let mut s: u64 = 0xDEADBEEF;
        let next = |s: &mut u64| {
            *s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            *s
        };
        for _ in 0..500 {
            // Up to 4 dims each, sizes in {1..=5}.
            let na = (next(&mut s) % 4 + 1) as usize;
            let nb = (next(&mut s) % 4 + 1) as usize;
            let mut a = Vec::with_capacity(na);
            let mut b = Vec::with_capacity(nb);
            for _ in 0..na {
                a.push(((next(&mut s) % 5) + 1) as usize);
            }
            for _ in 0..nb {
                b.push(((next(&mut s) % 5) + 1) as usize);
            }
            // Reference broadcast (NumPy rules) — independent re-impl.
            let n = na.max(nb);
            let mut expected = Vec::with_capacity(n);
            let mut compatible = true;
            for i in 0..n {
                let l = a.len().checked_sub(1 + i).map(|j| a[j]).unwrap_or(1);
                let r = b.len().checked_sub(1 + i).map(|j| b[j]).unwrap_or(1);
                if l == r {
                    expected.push(l);
                } else if l == 1 {
                    expected.push(r);
                } else if r == 1 {
                    expected.push(l);
                } else {
                    compatible = false;
                    break;
                }
            }
            expected.reverse();
            let lhs = Shape::from(a);
            let rhs = Shape::from(b);
            let got = lhs.broadcast_with(&rhs);
            if compatible {
                assert_eq!(got.unwrap().into_vec(), expected);
            } else {
                assert!(got.is_err());
            }
        }
    }

    #[test]
    fn expand_strides_basic() {
        // [3, 1] expanded to [2, 3, 4] → strides [0, 1, 0]
        let s = Shape::from([3, 1]);
        let target = Shape::from([2, 3, 4]);
        assert_eq!(s.expand_strides_to(&target).unwrap(), &[0, 1, 0]);
    }

    #[test]
    fn expand_strides_no_op() {
        let s = Shape::from([2, 3, 4]);
        let target = Shape::from([2, 3, 4]);
        // Contiguous strides for [2,3,4]: [12, 4, 1]
        assert_eq!(s.expand_strides_to(&target).unwrap(), &[12, 4, 1]);
    }

    #[test]
    fn expand_strides_scalar_to_anything() {
        let s = Shape::default();
        let target = Shape::from([2, 3, 4]);
        assert_eq!(s.expand_strides_to(&target).unwrap(), &[0, 0, 0]);
    }

    #[test]
    fn expand_strides_dim_mismatch_is_err() {
        // [3] cannot expand to [2] — neither equal nor 1.
        let s = Shape::from([3]);
        let target = Shape::from([2]);
        assert!(s.expand_strides_to(&target).is_err());
    }

    #[test]
    fn expand_strides_target_smaller_is_err() {
        let s = Shape::from([2, 3, 4]);
        let target = Shape::from([3, 4]);
        assert!(s.expand_strides_to(&target).is_err());
    }

    #[test]
    fn deep_shapes_work() {
        // Large rank shouldn't allocate-poison anything.
        let s: Shape = (1..=10).collect::<Vec<_>>().into();
        assert_eq!(s.ndim(), 10);
        assert_eq!(s.numel(), (1..=10).product::<usize>());
    }

    // ---------------------- proptest property tests ----------------------
    //
    // P1.1 task `Shape newtype + broadcasting helpers` step #4 — proptest
    // cross-check against an independent NumPy-rules reimplementation.

    use proptest::prelude::*;

    /// Strategy: dim shapes with rank 0..=4, sizes 1..=5.
    fn arb_shape() -> impl Strategy<Value = Vec<usize>> {
        proptest::collection::vec(1usize..=5, 0..=4)
    }

    /// Reference broadcasting (independent of `Shape::broadcast_with`).
    fn reference_broadcast(a: &[usize], b: &[usize]) -> Result<Vec<usize>, ()> {
        let n = a.len().max(b.len());
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let l = a.len().checked_sub(1 + i).map(|j| a[j]).unwrap_or(1);
            let r = b.len().checked_sub(1 + i).map(|j| b[j]).unwrap_or(1);
            match (l, r) {
                (a, b) if a == b => out.push(a),
                (1, b) => out.push(b),
                (a, 1) => out.push(a),
                _ => return Err(()),
            }
        }
        out.reverse();
        Ok(out)
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(500))]

        #[test]
        fn proptest_broadcast_matches_reference(a in arb_shape(), b in arb_shape()) {
            let lhs = Shape::from(a.clone());
            let rhs = Shape::from(b.clone());
            let got = lhs.broadcast_with(&rhs);
            let expected = reference_broadcast(&a, &b);
            match (got, expected) {
                (Ok(s), Ok(v)) => prop_assert_eq!(s.into_vec(), v),
                (Err(_), Err(())) => {},
                (Ok(s), Err(())) => prop_assert!(false, "ours OK ({:?}) but reference says incompatible", s),
                (Err(e), Ok(v)) => prop_assert!(false, "ours errored ({:?}) but reference produced {:?}", e, v),
            }
        }

        #[test]
        fn proptest_broadcast_commutative(a in arb_shape(), b in arb_shape()) {
            let lhs = Shape::from(a);
            let rhs = Shape::from(b);
            // a.broadcast(b) and b.broadcast(a) must agree (Ok or Err).
            match (lhs.broadcast_with(&rhs), rhs.broadcast_with(&lhs)) {
                (Ok(p), Ok(q)) => prop_assert_eq!(p, q),
                (Err(_), Err(_)) => {},
                _ => prop_assert!(false, "broadcast not commutative"),
            }
        }

        #[test]
        fn proptest_numel_is_product(a in arb_shape()) {
            let s = Shape::from(a.clone());
            let expected: usize = if a.is_empty() { 1 } else { a.iter().product() };
            prop_assert_eq!(s.numel(), expected);
        }

        #[test]
        fn proptest_self_broadcast_is_self(a in arb_shape()) {
            let s = Shape::from(a.clone());
            prop_assert_eq!(s.broadcast_with(&s).unwrap(), Shape::from(a));
        }
    }
}
