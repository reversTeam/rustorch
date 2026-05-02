//! Flash Attention backward (CPU) — Phase 3 task `7645c3b0`.
//!
//! Computes `(dQ, dK, dV)` from a saved forward pass (Q, K, V, the
//! softmax statistics m and l, and the upstream gradient dO).
//!
//! ## Math
//!
//! Standard scaled-dot-product attention forward:
//!
//! ```text
//!   S = Q @ K^T * scale          shape [N, N]
//!   P = softmax_row(S)           shape [N, N]
//!   O = P @ V                    shape [N, D]
//! ```
//!
//! Backward derivation (chain rule):
//!
//! ```text
//!   dV = P^T @ dO
//!   dP = dO @ V^T
//!   dS = dsoftmax(P, dP)         (per-row Jacobian)
//!   dQ = dS @ K * scale
//!   dK = dS^T @ Q * scale
//! ```
//!
//! `dsoftmax`'s closed form for each row:
//! `dS_ij = P_ij * (dP_ij - sum_k P_ik * dP_ik)`
//!
//! ## What this task ships
//!
//! Forward-saved `(m, l)` are NOT needed by THIS scalar backward
//! (we recompute P from Q/K). Saving them is an optimisation for
//! the GPU recomputation path; the CPU scalar version recomputes
//! P from scratch which is simpler and avoids any saved-tensor
//! state mismatch.

#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_many_arguments)]

use crate::cpu_forward::{AttentionError, AttentionShape};

/// Compute (dQ, dK, dV) from saved Q/K/V + upstream gradient dO.
///
/// All buffers are `[B, H, N, D]` row-major except the output dQ/dK/dV
/// which match Q/K/V shapes respectively. dO is `[B, H, N, D]`.
pub fn flash_backward(
    shape: &AttentionShape,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    do_: &[f32],
    dq: &mut [f32],
    dk: &mut [f32],
    dv: &mut [f32],
) -> Result<(), AttentionError> {
    let n = shape.seq;
    let d = shape.dim;
    let expected = shape.buffer_len();
    for (which, buf) in [("q", q), ("k", k), ("v", v), ("do", do_)] {
        if buf.len() != expected {
            return Err(AttentionError::BufferShapeMismatch {
                which,
                expected,
                got: buf.len(),
            });
        }
    }
    if dq.len() != expected || dk.len() != expected || dv.len() != expected {
        return Err(AttentionError::OutputShapeMismatch {
            expected,
            got: dq.len().max(dk.len()).max(dv.len()),
        });
    }
    if shape.seq == 0 {
        return Ok(());
    }
    let scale = 1.0 / (d as f32).sqrt();
    // Zero outputs.
    for x in dq.iter_mut() {
        *x = 0.0;
    }
    for x in dk.iter_mut() {
        *x = 0.0;
    }
    for x in dv.iter_mut() {
        *x = 0.0;
    }

    let s_h = n * d;
    let s_b = shape.heads * s_h;
    for b in 0..shape.batch {
        for h in 0..shape.heads {
            let base = b * s_b + h * s_h;
            // Recompute P[i, j] for this (b, h).
            let mut p = vec![0.0f32; n * n];
            for i in 0..n {
                let mut row_max = f32::NEG_INFINITY;
                let q_i = &q[base + i * d..base + i * d + d];
                for j in 0..n {
                    let k_j = &k[base + j * d..base + j * d + d];
                    let mut s = 0.0f32;
                    for dd in 0..d {
                        s += q_i[dd] * k_j[dd];
                    }
                    let scaled = s * scale;
                    p[i * n + j] = scaled;
                    if scaled > row_max {
                        row_max = scaled;
                    }
                }
                let mut row_sum = 0.0f32;
                for j in 0..n {
                    let e = (p[i * n + j] - row_max).exp();
                    p[i * n + j] = e;
                    row_sum += e;
                }
                if row_sum > 0.0 && row_sum.is_finite() {
                    for j in 0..n {
                        p[i * n + j] /= row_sum;
                    }
                }
            }
            // dV = P^T @ dO  → dV[j, dd] += sum_i P[i, j] * dO[i, dd]
            for j in 0..n {
                let dv_j = &mut dv[base + j * d..base + j * d + d];
                for i in 0..n {
                    let pij = p[i * n + j];
                    let do_i = &do_[base + i * d..base + i * d + d];
                    for dd in 0..d {
                        dv_j[dd] += pij * do_i[dd];
                    }
                }
            }
            // dP = dO @ V^T  → dP[i, j] = sum_dd dO[i, dd] * V[j, dd]
            let mut dp = vec![0.0f32; n * n];
            for i in 0..n {
                let do_i = &do_[base + i * d..base + i * d + d];
                for j in 0..n {
                    let v_j = &v[base + j * d..base + j * d + d];
                    let mut acc = 0.0f32;
                    for dd in 0..d {
                        acc += do_i[dd] * v_j[dd];
                    }
                    dp[i * n + j] = acc;
                }
            }
            // dS = dsoftmax(P, dP)  per row
            //   row_dot = sum_j P[i, j] * dP[i, j]
            //   dS[i, j] = P[i, j] * (dP[i, j] - row_dot)
            let mut ds = vec![0.0f32; n * n];
            for i in 0..n {
                let mut row_dot = 0.0f32;
                for j in 0..n {
                    row_dot += p[i * n + j] * dp[i * n + j];
                }
                for j in 0..n {
                    ds[i * n + j] = p[i * n + j] * (dp[i * n + j] - row_dot);
                }
            }
            // dQ = dS @ K * scale  → dQ[i, dd] = scale * sum_j dS[i, j] * K[j, dd]
            for i in 0..n {
                let dq_i = &mut dq[base + i * d..base + i * d + d];
                for j in 0..n {
                    let k_j = &k[base + j * d..base + j * d + d];
                    let dsij = ds[i * n + j];
                    for dd in 0..d {
                        dq_i[dd] += scale * dsij * k_j[dd];
                    }
                }
            }
            // dK = dS^T @ Q * scale → dK[j, dd] = scale * sum_i dS[i, j] * Q[i, dd]
            for j in 0..n {
                let dk_j = &mut dk[base + j * d..base + j * d + d];
                for i in 0..n {
                    let q_i = &q[base + i * d..base + i * d + d];
                    let dsij = ds[i * n + j];
                    for dd in 0..d {
                        dk_j[dd] += scale * dsij * q_i[dd];
                    }
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu_forward::naive_forward;

    fn deterministic(n: usize, seed: u32) -> Vec<f32> {
        let mut s = seed.wrapping_mul(2654435761);
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                (s as f32 / u32::MAX as f32 - 0.5) * 2.0
            })
            .collect()
    }

    /// Numerical gradient via finite differences. Used to verify
    /// backward correctness on small shapes.
    fn finite_diff_grad(
        shape: &AttentionShape,
        q: &mut [f32],
        k: &[f32],
        v: &[f32],
        do_: &[f32],
        idx: usize,
        eps: f32,
    ) -> f32 {
        let n = shape.buffer_len();
        let mut o_plus = vec![0.0f32; n];
        let mut o_minus = vec![0.0f32; n];
        let saved = q[idx];
        q[idx] = saved + eps;
        naive_forward(shape, q, k, v, &mut o_plus).unwrap();
        q[idx] = saved - eps;
        naive_forward(shape, q, k, v, &mut o_minus).unwrap();
        q[idx] = saved;
        // dL/dQ_idx = sum_k dO_k * dO_k/dQ_idx ≈ dO · (o_plus - o_minus) / (2 eps)
        let mut g = 0.0f32;
        for k_ in 0..n {
            g += do_[k_] * (o_plus[k_] - o_minus[k_]) / (2.0 * eps);
        }
        g
    }

    #[test]
    fn backward_dq_matches_finite_difference() {
        let shape = AttentionShape::new(1, 1, 4, 4);
        let mut q = deterministic(shape.buffer_len(), 1);
        let k = deterministic(shape.buffer_len(), 2);
        let v = deterministic(shape.buffer_len(), 3);
        let do_ = deterministic(shape.buffer_len(), 4);
        let mut dq = vec![0.0f32; shape.buffer_len()];
        let mut dk = vec![0.0f32; shape.buffer_len()];
        let mut dv = vec![0.0f32; shape.buffer_len()];
        flash_backward(&shape, &q, &k, &v, &do_, &mut dq, &mut dk, &mut dv).unwrap();
        // Spot-check 3 dQ entries via finite diff.
        for &idx in &[0, 5, 10] {
            let analytic = dq[idx];
            let numeric = finite_diff_grad(&shape, &mut q, &k, &v, &do_, idx, 1e-3);
            assert!(
                (analytic - numeric).abs() < 5e-3,
                "dq[{idx}] analytic {analytic} numeric {numeric}"
            );
        }
    }

    #[test]
    fn backward_empty_seq_is_no_op() {
        let shape = AttentionShape::new(1, 1, 0, 4);
        let mut dq: Vec<f32> = Vec::new();
        let mut dk: Vec<f32> = Vec::new();
        let mut dv: Vec<f32> = Vec::new();
        flash_backward(&shape, &[], &[], &[], &[], &mut dq, &mut dk, &mut dv).unwrap();
        assert!(dq.is_empty());
    }

    #[test]
    fn backward_shape_mismatch_returns_error() {
        let shape = AttentionShape::new(1, 1, 4, 4);
        let q = vec![0.0f32; 10]; // wrong: should be 16
        let k = vec![0.0f32; 16];
        let v = vec![0.0f32; 16];
        let do_ = vec![0.0f32; 16];
        let mut dq = vec![0.0f32; 16];
        let mut dk = vec![0.0f32; 16];
        let mut dv = vec![0.0f32; 16];
        let err = flash_backward(&shape, &q, &k, &v, &do_, &mut dq, &mut dk, &mut dv).unwrap_err();
        assert!(matches!(
            err,
            AttentionError::BufferShapeMismatch { which: "q", .. }
        ));
    }

    #[test]
    fn backward_zero_grad_yields_zero_outputs() {
        let shape = AttentionShape::new(1, 1, 3, 4);
        let q = deterministic(shape.buffer_len(), 1);
        let k = deterministic(shape.buffer_len(), 2);
        let v = deterministic(shape.buffer_len(), 3);
        let do_ = vec![0.0f32; shape.buffer_len()];
        let mut dq = vec![0.0f32; shape.buffer_len()];
        let mut dk = vec![0.0f32; shape.buffer_len()];
        let mut dv = vec![0.0f32; shape.buffer_len()];
        flash_backward(&shape, &q, &k, &v, &do_, &mut dq, &mut dk, &mut dv).unwrap();
        for x in &dq {
            assert_eq!(*x, 0.0);
        }
        for x in &dk {
            assert_eq!(*x, 0.0);
        }
        for x in &dv {
            assert_eq!(*x, 0.0);
        }
    }
}
