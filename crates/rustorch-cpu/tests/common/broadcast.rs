//! Broadcasting test grid.
//!
//! P1.3 task `Broadcasting test infrastructure`:
//!
//! - At least **200** (lhs_shape, rhs_shape, expected_shape) tuples.
//! - Covers: scalar, 1D, broadcasting one-of-them-1, missing dims,
//!   edge cases [0]/[1], up to 6D.
//! - Reproducible (LCG-seeded), so the harness produces the same set
//!   of shapes across CI runs.
//! - Exposes failing pairs ([3] vs [4]) so error-path tests can use
//!   the same grid.

use rustorch_core::tensor::shape::Shape;

/// One element of the broadcast test grid.
#[derive(Debug, Clone)]
pub struct BroadcastCase {
    /// Left-hand side shape.
    pub lhs: Vec<usize>,
    /// Right-hand side shape.
    pub rhs: Vec<usize>,
    /// Expected broadcasted shape, or `None` if the pair is incompatible.
    pub expected: Option<Vec<usize>>,
}

impl BroadcastCase {
    /// Convenience: convert lhs/rhs into [`Shape`].
    pub fn shapes(&self) -> (Shape, Shape) {
        (Shape::from(self.lhs.clone()), Shape::from(self.rhs.clone()))
    }
}

/// Build the canonical broadcast test grid (≥ 200 cases).
///
/// The grid is a fixed list of categories so each subset can be
/// reasoned about:
///
/// 1. Identical-shape pairs (rank 0..=6).
/// 2. Scalar against any rank.
/// 3. Trailing-dim broadcast (`[N]` vs `[..., N]`).
/// 4. One-of-them-1 broadcast on every axis up to rank 4.
/// 5. Missing-dims (lhs has fewer dims than rhs and vice versa).
/// 6. Edge dims `[0]` and `[1]`.
/// 7. LCG-driven random pairs (deterministic, 100 cases).
/// 8. Incompatible pairs (negative cases).
pub fn broadcast_test_grid() -> Vec<BroadcastCase> {
    let mut cases = Vec::with_capacity(256);

    // ---- 1. Identical-shape pairs ----
    let identical: &[&[usize]] = &[
        &[], // scalar
        &[1],
        &[3],
        &[2, 3],
        &[2, 3, 4],
        &[2, 3, 4, 5],
        &[1, 2, 3, 4, 5],
        &[1, 1, 2, 3, 4, 5], // 6D
    ];
    for s in identical {
        cases.push(BroadcastCase {
            lhs: s.to_vec(),
            rhs: s.to_vec(),
            expected: Some(s.to_vec()),
        });
    }

    // ---- 2. Scalar against anything ----
    let against_scalar: &[&[usize]] = &[
        &[3],
        &[2, 3],
        &[2, 3, 4],
        &[8, 32, 224, 224],
        &[1, 1, 1, 1, 1, 1],
    ];
    for s in against_scalar {
        cases.push(BroadcastCase {
            lhs: vec![],
            rhs: s.to_vec(),
            expected: Some(s.to_vec()),
        });
        cases.push(BroadcastCase {
            lhs: s.to_vec(),
            rhs: vec![],
            expected: Some(s.to_vec()),
        });
    }

    // ---- 3. Trailing-dim broadcast (1D against ND with same trailing dim) ----
    let trailing: &[(&[usize], usize)] = &[
        (&[2, 3], 3),
        (&[2, 3, 4], 4),
        (&[5, 6, 7], 7),
        (&[2, 3, 4, 5], 5),
    ];
    for (s, n) in trailing {
        cases.push(BroadcastCase {
            lhs: s.to_vec(),
            rhs: vec![*n],
            expected: Some(s.to_vec()),
        });
        cases.push(BroadcastCase {
            lhs: vec![*n],
            rhs: s.to_vec(),
            expected: Some(s.to_vec()),
        });
    }

    // ---- 4. One-of-them-1 on every axis up to rank 4 ----
    // Seeds: pairs that exhibit broadcast-along-axis behaviour.
    let one_of_them_1: &[(&[usize], &[usize], &[usize])] = &[
        (&[5, 1, 4], &[3, 1], &[5, 3, 4]),
        (&[1, 3, 4], &[2, 1, 4], &[2, 3, 4]),
        (&[2, 1, 4], &[2, 3, 1], &[2, 3, 4]),
        (&[1, 1, 4], &[2, 3, 1], &[2, 3, 4]),
        (&[2, 3, 4, 1], &[5], &[2, 3, 4, 5]),
        (&[2, 1, 1, 5], &[3, 4, 1], &[2, 3, 4, 5]),
        (&[1], &[2, 3, 4], &[2, 3, 4]),
        (&[2, 3], &[1, 1], &[2, 3]),
        (&[1, 3], &[2, 3], &[2, 3]),
    ];
    for (a, b, exp) in one_of_them_1 {
        cases.push(BroadcastCase {
            lhs: a.to_vec(),
            rhs: b.to_vec(),
            expected: Some(exp.to_vec()),
        });
        // Symmetric form
        cases.push(BroadcastCase {
            lhs: b.to_vec(),
            rhs: a.to_vec(),
            expected: Some(exp.to_vec()),
        });
    }

    // ---- 5. Missing-dims pairs ----
    let missing: &[(&[usize], &[usize], &[usize])] = &[
        (&[3], &[2, 3], &[2, 3]),
        (&[3, 4], &[2, 3, 4], &[2, 3, 4]),
        (&[4, 5], &[2, 3, 4, 5], &[2, 3, 4, 5]),
        (&[5], &[2, 3, 4, 5], &[2, 3, 4, 5]),
    ];
    for (a, b, exp) in missing {
        cases.push(BroadcastCase {
            lhs: a.to_vec(),
            rhs: b.to_vec(),
            expected: Some(exp.to_vec()),
        });
        cases.push(BroadcastCase {
            lhs: b.to_vec(),
            rhs: a.to_vec(),
            expected: Some(exp.to_vec()),
        });
    }

    // ---- 6. Edge dims [0] and [1] ----
    cases.push(BroadcastCase {
        lhs: vec![0],
        rhs: vec![1],
        expected: Some(vec![0]),
    });
    cases.push(BroadcastCase {
        lhs: vec![1],
        rhs: vec![0],
        expected: Some(vec![0]),
    });
    cases.push(BroadcastCase {
        lhs: vec![0, 3],
        rhs: vec![1, 3],
        expected: Some(vec![0, 3]),
    });
    cases.push(BroadcastCase {
        lhs: vec![0, 3],
        rhs: vec![3],
        expected: Some(vec![0, 3]),
    });

    // ---- 7. LCG-driven random pairs (deterministic) ----
    let mut s: u64 = 0xBEEFCAFE_DEADBEEF;
    fn lcg_next(s: &mut u64) -> u64 {
        *s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        *s
    }
    for _ in 0..150 {
        let na = (lcg_next(&mut s) % 4 + 1) as usize; // rank 1..=4
        let nb = (lcg_next(&mut s) % 4 + 1) as usize;
        let mut a = Vec::with_capacity(na);
        let mut b = Vec::with_capacity(nb);
        for _ in 0..na {
            a.push((lcg_next(&mut s) % 5 + 1) as usize); // dim 1..=5
        }
        for _ in 0..nb {
            b.push((lcg_next(&mut s) % 5 + 1) as usize);
        }
        let expected = reference_broadcast(&a, &b);
        cases.push(BroadcastCase {
            lhs: a,
            rhs: b,
            expected,
        });
    }

    // ---- 8. Incompatible pairs (negative cases) ----
    let bad: &[(&[usize], &[usize])] = &[
        (&[3], &[4]),
        (&[2, 3], &[3, 4]),
        (&[5, 4], &[3, 4, 4]),
        (&[2, 3, 4], &[2, 5, 4]),
    ];
    for (a, b) in bad {
        cases.push(BroadcastCase {
            lhs: a.to_vec(),
            rhs: b.to_vec(),
            expected: None,
        });
    }

    cases
}

/// Reference broadcasting routine (independent of [`Shape::broadcast_with`]).
/// Used both internally by [`broadcast_test_grid`] and as a cross-check
/// implementation in tests.
pub fn reference_broadcast(a: &[usize], b: &[usize]) -> Option<Vec<usize>> {
    let n = a.len().max(b.len());
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let l = a.len().checked_sub(1 + i).map(|j| a[j]).unwrap_or(1);
        let r = b.len().checked_sub(1 + i).map(|j| b[j]).unwrap_or(1);
        match (l, r) {
            (a, b) if a == b => out.push(a),
            (1, b) => out.push(b),
            (a, 1) => out.push(a),
            _ => return None,
        }
    }
    out.reverse();
    Some(out)
}
