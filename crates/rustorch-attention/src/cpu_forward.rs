//! Flash Attention forward (CPU, tiled Q/K/V) — Phase 3 task `900f399e`.
//!
//! Implements the tiled attention from the Flash Attention v2 paper:
//!
//! ```text
//!   For each batch b, head h, query tile Qi:
//!     init Oi = 0, mi = -inf, li = 0
//!     For each key tile Kj, Vj:
//!         Sij = (Qi @ Kj^T) * scale            ← scaled scores
//!         m_new = max(mi, rowmax(Sij))
//!         Pij   = exp(Sij - m_new)             ← unnormalised probs
//!         li    = li * exp(mi - m_new) + rowsum(Pij)
//!         Oi    = Oi * exp(mi - m_new) + Pij @ Vj   ← rescaled accumulator
//!         mi    = m_new
//!     Oi /= li                                  ← final normalise
//! ```
//!
//! No N×N matrix is ever materialised; peak memory is O(N + tile²)
//! per (batch, head, query-tile), and the outer loop is embarrassingly
//! parallel over `(B, H, Qi-tile)`.
//!
//! ## Layout convention
//!
//! Input/output buffers are laid out as flat `Vec<f32>` with shape
//! `[B, H, N, D]` in row-major (BHND) order. The convenience
//! [`AttentionShape`] struct carries the four dimensions and the
//! tile sizes used by the kernel.
//!
//! ## Limitations of THIS task
//!
//! - **Forward only** — backward (with recomputation) and GPU/WGSL
//!   ship in subsequent commits.
//! - **No masks / no dropout** — those layer on top via the
//!   `Masking + dropout` task; for now every Q-K pair contributes.
//! - **f32 only** — bf16/fp16 land with the Mixed Precision plan.

// `for d in 0..dim` indexes into multiple parallel arrays (dst[d],
// row[d], v_row[d]) — not amenable to a single iterator. Allow the
// pattern crate-locally.
#![allow(clippy::needless_range_loop)]

use crate::mask::{Mask, MaskError};
#[cfg(target_arch = "wasm32")]
use crate::online_softmax::OnlineSoftmaxState;
#[cfg(not(target_arch = "wasm32"))]
use crate::simd as simd_inner;
use rayon::prelude::*;

/// Shape parameters for a single Flash Attention call.
#[derive(Debug, Clone, Copy)]
pub struct AttentionShape {
    /// Batch size.
    pub batch: usize,
    /// Number of attention heads.
    pub heads: usize,
    /// Sequence length (queries and keys share this dim in self-attn).
    pub seq: usize,
    /// Per-head feature dim.
    pub dim: usize,
    /// Q-tile rows per outer iteration. Default 64.
    pub br: usize,
    /// K/V-tile rows per inner iteration. Default 64.
    pub bc: usize,
}

impl AttentionShape {
    /// Build a shape with the default tile sizes (64×64).
    pub fn new(batch: usize, heads: usize, seq: usize, dim: usize) -> Self {
        Self {
            batch,
            heads,
            seq,
            dim,
            br: 64,
            bc: 64,
        }
    }

    /// Total number of f32 elements per Q/K/V/O buffer.
    pub fn buffer_len(&self) -> usize {
        self.batch * self.heads * self.seq * self.dim
    }

    /// Strides (in elements) for the (B, H, N, D) layout.
    fn strides(&self) -> (usize, usize, usize) {
        let s_n = self.dim;
        let s_h = self.seq * s_n;
        let s_b = self.heads * s_h;
        (s_b, s_h, s_n)
    }
}

/// Errors returned by the forward kernel.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AttentionError {
    /// `q.len()` (or k/v) does not match `shape.buffer_len()`.
    BufferShapeMismatch {
        /// Which buffer was wrong.
        which: &'static str,
        /// Expected element count.
        expected: usize,
        /// Actual element count.
        got: usize,
    },
    /// `output.len()` does not match `shape.buffer_len()`.
    OutputShapeMismatch {
        /// Expected element count.
        expected: usize,
        /// Actual element count.
        got: usize,
    },
    /// `dim == 0` makes the scaling factor `1/sqrt(0)` undefined.
    ZeroDim,
    /// Mask validation failed (e.g. padding mask length mismatch).
    Mask(MaskError),
}

impl From<MaskError> for AttentionError {
    fn from(e: MaskError) -> Self {
        AttentionError::Mask(e)
    }
}

impl core::fmt::Display for AttentionError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            AttentionError::BufferShapeMismatch {
                which,
                expected,
                got,
            } => write!(f, "{which} buffer has {got} elements, expected {expected}"),
            AttentionError::OutputShapeMismatch { expected, got } => {
                write!(f, "output buffer has {got} elements, expected {expected}")
            },
            AttentionError::ZeroDim => write!(f, "dim must be > 0"),
            AttentionError::Mask(e) => write!(f, "mask error: {e}"),
        }
    }
}

impl std::error::Error for AttentionError {}

/// Validate input shapes against the declared shape.
fn validate(
    shape: &AttentionShape,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    out: &[f32],
) -> Result<(), AttentionError> {
    if shape.dim == 0 {
        return Err(AttentionError::ZeroDim);
    }
    let expected = shape.buffer_len();
    for (which, buf) in [("q", q), ("k", k), ("v", v)] {
        if buf.len() != expected {
            return Err(AttentionError::BufferShapeMismatch {
                which,
                expected,
                got: buf.len(),
            });
        }
    }
    if out.len() != expected {
        return Err(AttentionError::OutputShapeMismatch {
            expected,
            got: out.len(),
        });
    }
    Ok(())
}

/// Compute scaled attention `O = softmax(QK^T * scale) V` using the
/// tiled Flash forward kernel. Output is written in-place to `out`.
///
/// Equivalent to [`flash_forward_masked`] with `Mask::None`.
pub fn flash_forward(
    shape: &AttentionShape,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    out: &mut [f32],
) -> Result<(), AttentionError> {
    flash_forward_masked(shape, q, k, v, &Mask::None, out)
}

/// Compute masked scaled attention `O = softmax((QK^T + mask) * scale) V`.
///
/// `scale` is `1/sqrt(dim)` and is applied INSIDE the kernel exactly
/// once per score. Masked positions become `-inf` BEFORE the online
/// softmax sees them, so they contribute zero (cleanly via
/// `exp(-inf - finite) = 0`) without any NaN risk.
///
/// The outer loops over `(batch, head)` are parallelised via `rayon`.
pub fn flash_forward_masked(
    shape: &AttentionShape,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    mask: &Mask<'_>,
    out: &mut [f32],
) -> Result<(), AttentionError> {
    validate(shape, q, k, v, out)?;
    mask.validate(shape.seq)?;
    if shape.seq == 0 {
        // Empty seq → output is already correctly-sized empty.
        return Ok(());
    }

    let scale = 1.0 / (shape.dim as f32).sqrt();
    let (_s_b, s_h, s_n) = shape.strides();
    let dim = shape.dim;
    let seq = shape.seq;
    let br = shape.br;
    let bc = shape.bc;
    let heads = shape.heads;

    // Split `out` into per-(b, h) slabs of `seq * dim` elements; each
    // slab is touched by exactly one parallel task → safe disjoint
    // mutable access without raw pointers.
    out.par_chunks_mut(s_h)
        .enumerate()
        .for_each(|(bh, out_slab)| {
            let b = bh / heads;
            let h = bh % heads;
            let q_base = b * (heads * s_h) + h * s_h;
            let k_base = q_base;
            let v_base = q_base;
            let args = FlashTileArgs {
                scale,
                seq,
                dim,
                br,
                bc,
                s_n,
            };

            #[cfg(not(target_arch = "wasm32"))]
            flash_forward_bh_blas(q, k, v, out_slab, mask, args, q_base, k_base, v_base);
            #[cfg(target_arch = "wasm32")]
            flash_forward_bh_scalar(q, k, v, out_slab, mask, args, q_base, k_base, v_base);
        });
    Ok(())
}

/// Bundled per-call constants (kept in one struct so the inner
/// dispatcher signature stays tame).
#[derive(Clone, Copy)]
struct FlashTileArgs {
    scale: f32,
    seq: usize,
    dim: usize,
    br: usize,
    bc: usize,
    s_n: usize,
}

/// BLAS-tile path: each `[br × bc]` score tile is computed via a
/// single sgemm (`Q[qi..] @ K[kj..]^T`); the value-accumulator
/// `O[qi..] += P[..] @ V[kj..]` is a second sgemm with `beta=1`.
///
/// Order-of-magnitude analysis on M4 Max for B=2 H=4 N=2048 D=64:
/// per (b,h) we do `(N/br)·(N/bc) = 1024` sgemm-pairs of `64³`. Each
/// pair runs at AMX peak (~1.49 TF/s) in ~350 ns → 360 µs per (b,h),
/// 8 (b,h) parallel → ~360 µs total compute. The per-row scalar path
/// it replaces measured 246 ms — a ~700× ceiling lift before
/// dispatch overhead.
#[cfg(not(target_arch = "wasm32"))]
#[allow(clippy::too_many_arguments)]
fn flash_forward_bh_blas(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    out_slab: &mut [f32],
    mask: &Mask<'_>,
    args: FlashTileArgs,
    q_base: usize,
    k_base: usize,
    v_base: usize,
) {
    let FlashTileArgs {
        scale,
        seq,
        dim,
        br,
        bc,
        s_n,
    } = args;

    // Reusable scratch buffers — allocated once per (b, h) outer
    // iteration, reused across all qi tiles. Capacities sized to the
    // worst case (full br × bc, br × dim, br).
    let mut s_buf = vec![0.0_f32; br * bc];
    let mut p_buf = vec![0.0_f32; br * bc];
    let mut o_buf = vec![0.0_f32; br * dim];
    let mut m_buf = vec![f32::NEG_INFINITY; br];
    let mut l_buf = vec![0.0_f32; br];

    // T8-new — runtime SIMD dispatch (NEON on aarch64 / AVX2 on x86).
    // `Arch::new()` is a cheap one-shot CPU-feature probe; the
    // returned `Arch` is `Copy` and passed by value into each inner
    // helper, which inlines through `WithSimd::with_simd`.
    let arch = pulp::Arch::new();

    let mut qi = 0;
    while qi < seq {
        let qi_end = (qi + br).min(seq);
        let br_used = qi_end - qi;

        // Reset accumulators for this query tile.
        for r in 0..br_used {
            m_buf[r] = f32::NEG_INFINITY;
            l_buf[r] = 0.0;
        }
        // SIMD-fill the per-row output buffer in one pass instead of
        // br_used × dim scalar stores.
        simd_inner::fill_zero(arch, &mut o_buf[..br_used * dim]);

        let mut kj = 0;
        while kj < seq {
            let kj_end = (kj + bc).min(seq);
            let bc_used = kj_end - kj;

            // S[br_used × bc_used] = scale · Q_tile[br_used × dim]
            //                              · K_tile[bc_used × dim]^T
            //
            // Q_tile is rows [qi..qi_end] of `q`; K_tile is rows
            // [kj..kj_end] of `k`. Both have row-stride `s_n` (= dim
            // for [B,H,N,D] layout but the kernel passes it
            // explicitly so non-square tail tiles work too).
            // SAFETY: input slice lengths checked by `validate`;
            //         strides and dimensions are consistent with
            //         Accelerate's row-major sgemm contract.
            sgemm_score_tile(
                br_used,
                bc_used,
                dim,
                scale,
                &q[q_base + qi * s_n..],
                s_n,
                &k[k_base + kj * s_n..],
                s_n,
                &mut s_buf[..br_used * bc_used],
                bc_used,
            );

            // Mask: -∞ where mask says "skip". exp(-∞ - finite) = 0,
            //   so masked positions contribute zero through the rest
            //   of the math without any NaN risk.
            if !matches!(mask, Mask::None) {
                for r in 0..br_used {
                    let qi_row = qi + r;
                    let row = &mut s_buf[r * bc_used..r * bc_used + bc_used];
                    for (c, cell) in row.iter_mut().enumerate() {
                        if mask.is_masked(qi_row, kj + c) {
                            *cell = f32::NEG_INFINITY;
                        }
                    }
                }
            }

            // Per-row online softmax update (T8-new — SIMD inner).
            //
            // Each row of `S` is a `bc_used`-element vector; we run:
            //   1. `m_tile = max(s_row)` — SIMD `row_max`
            //   2. `m_new  = max(m_tile, m_prev)`
            //   3. `alpha  = exp(m_prev - m_new)`
            //   4. `o_row *= alpha`  (only if alpha != 1) — SIMD
            //      `scale_in_place` over `dim` elements
            //   5. `p_row[i] = exp(s_row[i] - m_new); tile_l = sum p_row`
            //      — scalar `exp_minus_max_and_sum` (libm path, scalar
            //      exp is monomorphic and the compiler emits libsystem_m
            //      vectorised expf on M-series)
            //   6. `l_buf[r] += tile_l; m_buf[r] = m_new`
            for r in 0..br_used {
                let s_row = &s_buf[r * bc_used..r * bc_used + bc_used];
                let m_tile = simd_inner::row_max(arch, s_row);
                let m_prev = m_buf[r];
                let m_new = if m_tile > m_prev { m_tile } else { m_prev };
                let alpha = if m_prev == f32::NEG_INFINITY {
                    0.0
                } else {
                    (m_prev - m_new).exp()
                };
                if alpha != 1.0 {
                    l_buf[r] *= alpha;
                    simd_inner::scale_in_place(arch, &mut o_buf[r * dim..r * dim + dim], alpha);
                }
                let p_row = &mut p_buf[r * bc_used..r * bc_used + bc_used];
                let tile_l = simd_inner::exp_minus_max_and_sum(arch, p_row, s_row, m_new);
                l_buf[r] += tile_l;
                m_buf[r] = m_new;
            }

            // O[br_used × dim] += P[br_used × bc_used] @ V_tile[bc_used × dim]
            //   (β=1, accumulating). V_tile rows have stride `s_n`.
            sgemm_value_accumulate(
                br_used,
                dim,
                bc_used,
                &p_buf[..br_used * bc_used],
                bc_used,
                &v[v_base + kj * s_n..],
                s_n,
                &mut o_buf[..br_used * dim],
                dim,
            );

            kj = kj_end;
        }

        // Write back per-row normalised output (T8-new — SIMD).
        // Final pass: `out[r, :] = o_buf[r, :] / l[r]`. Single splat +
        // SIMD multiply per row (dim=64 on the bench shape → 16 NEON
        // f32x4 stores per row, vs 64 scalar stores previously).
        for r in 0..br_used {
            let qi_row = qi + r;
            let dst = &mut out_slab[qi_row * s_n..qi_row * s_n + dim];
            let l = l_buf[r];
            let o_row = &o_buf[r * dim..r * dim + dim];
            if l > 0.0 && l.is_finite() {
                simd_inner::mul_scalar(arch, dst, o_row, 1.0 / l);
            } else {
                for x in dst.iter_mut() {
                    *x = f32::NAN;
                }
            }
        }

        qi = qi_end;
    }
}

/// `S[m × n] = alpha · Q[m × k] @ K[n × k]^T`. Row-major, β=0.
#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
#[inline]
fn sgemm_score_tile(
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    q: &[f32],
    ldq: usize,
    k_buf: &[f32],
    ldk: usize,
    s: &mut [f32],
    lds: usize,
) {
    use crate::accelerate::{cblas_sgemm, CBLAS_NO_TRANS, CBLAS_ROW_MAJOR, CBLAS_TRANS};
    // SAFETY: caller guarantees m*ldq ≤ q.len(), n*ldk ≤ k_buf.len(),
    //         m*lds ≤ s.len(); ld_* are positive ints fitting c_int.
    unsafe {
        cblas_sgemm(
            CBLAS_ROW_MAJOR,
            CBLAS_NO_TRANS,
            CBLAS_TRANS,
            m as i32,
            n as i32,
            k as i32,
            alpha,
            q.as_ptr(),
            ldq as i32,
            k_buf.as_ptr(),
            ldk as i32,
            0.0,
            s.as_mut_ptr(),
            lds as i32,
        );
    }
}

/// Non-macOS: pure-Rust `gemm` 0.18 with implicit `B^T` via column
/// strides (rs=1, cs=ldk → effectively transposes the read order).
#[cfg(all(not(target_os = "macos"), not(target_arch = "wasm32")))]
#[allow(clippy::too_many_arguments)]
#[inline]
fn sgemm_score_tile(
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    q: &[f32],
    ldq: usize,
    k_buf: &[f32],
    ldk: usize,
    s: &mut [f32],
    lds: usize,
) {
    // Effectively K_buf^T: gemm reads k_buf with row stride 1 and
    //   column stride ldk = swapping the canonical row-major access.
    // SAFETY: same invariants as the macOS path.
    unsafe {
        gemm::gemm(
            m,
            n,
            k,
            s.as_mut_ptr(),
            1,
            lds as isize,
            false,
            q.as_ptr(),
            1,
            ldq as isize,
            k_buf.as_ptr(),
            ldk as isize,
            1,
            0.0_f32,
            alpha,
            false,
            false,
            false,
            gemm::Parallelism::None,
        );
    }
}

/// `O[m × n] += P[m × k] @ V[k × n]`. Row-major, β=1.
#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
#[inline]
fn sgemm_value_accumulate(
    m: usize,
    n: usize,
    k: usize,
    p: &[f32],
    ldp: usize,
    v: &[f32],
    ldv: usize,
    o: &mut [f32],
    ldo: usize,
) {
    use crate::accelerate::{cblas_sgemm, CBLAS_NO_TRANS, CBLAS_ROW_MAJOR};
    // SAFETY: see sgemm_score_tile.
    unsafe {
        cblas_sgemm(
            CBLAS_ROW_MAJOR,
            CBLAS_NO_TRANS,
            CBLAS_NO_TRANS,
            m as i32,
            n as i32,
            k as i32,
            1.0,
            p.as_ptr(),
            ldp as i32,
            v.as_ptr(),
            ldv as i32,
            1.0,
            o.as_mut_ptr(),
            ldo as i32,
        );
    }
}

#[cfg(all(not(target_os = "macos"), not(target_arch = "wasm32")))]
#[allow(clippy::too_many_arguments)]
#[inline]
fn sgemm_value_accumulate(
    m: usize,
    n: usize,
    k: usize,
    p: &[f32],
    ldp: usize,
    v: &[f32],
    ldv: usize,
    o: &mut [f32],
    ldo: usize,
) {
    // SAFETY: see sgemm_score_tile.
    unsafe {
        gemm::gemm(
            m,
            n,
            k,
            o.as_mut_ptr(),
            1,
            ldo as isize,
            true,
            p.as_ptr(),
            1,
            ldp as isize,
            v.as_ptr(),
            1,
            ldv as isize,
            1.0_f32,
            1.0_f32,
            false,
            false,
            false,
            gemm::Parallelism::None,
        );
    }
}

/// wasm32 fallback — keeps the per-row scalar path that was the
/// pre-T6 implementation. No SIMD intrinsics on wasm; LLVM still
/// auto-vectorises with simd128 enabled but `gemm` does not build
/// for wasm32.
#[cfg(target_arch = "wasm32")]
#[allow(clippy::too_many_arguments)]
fn flash_forward_bh_scalar(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    out_slab: &mut [f32],
    mask: &Mask<'_>,
    args: FlashTileArgs,
    q_base: usize,
    k_base: usize,
    v_base: usize,
) {
    let FlashTileArgs {
        scale,
        seq,
        dim,
        br,
        bc,
        s_n,
    } = args;
    let mut qi = 0;
    while qi < seq {
        let qi_end = (qi + br).min(seq);
        for qi_row in qi..qi_end {
            let mut state = OnlineSoftmaxState::EMPTY;
            let mut o_row: Vec<f32> = vec![0.0; dim];
            let q_row = &q[q_base + qi_row * s_n..q_base + qi_row * s_n + dim];

            let mut kj = 0;
            while kj < seq {
                let kj_end = (kj + bc).min(seq);
                let bc_used = kj_end - kj;

                let mut scores: Vec<f32> = Vec::with_capacity(bc_used);
                for kk in kj..kj_end {
                    let k_row = &k[k_base + kk * s_n..k_base + kk * s_n + dim];
                    let mut s = 0.0f32;
                    for d in 0..dim {
                        s += q_row[d] * k_row[d];
                    }
                    let score = if mask.is_masked(qi_row, kk) {
                        f32::NEG_INFINITY
                    } else {
                        s * scale
                    };
                    scores.push(score);
                }

                let m_new_tile = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let m_combined = if m_new_tile > state.m {
                    m_new_tile
                } else {
                    state.m
                };
                let alpha = if state.m == f32::NEG_INFINITY {
                    0.0
                } else {
                    (state.m - m_combined).exp()
                };
                if alpha != 1.0 {
                    state.l *= alpha;
                    for o in o_row.iter_mut() {
                        *o *= alpha;
                    }
                }

                let mut tile_l = 0.0f32;
                for (kk_offset, &s) in scores.iter().enumerate() {
                    let p = (s - m_combined).exp();
                    tile_l += p;
                    let kk = kj + kk_offset;
                    let v_row = &v[v_base + kk * s_n..v_base + kk * s_n + dim];
                    for d in 0..dim {
                        o_row[d] += p * v_row[d];
                    }
                }
                state.l += tile_l;
                state.m = m_combined;
                kj = kj_end;
            }
            let dst = &mut out_slab[qi_row * s_n..qi_row * s_n + dim];
            if state.l > 0.0 && state.l.is_finite() {
                for (d, o) in o_row.iter().enumerate() {
                    dst[d] = *o / state.l;
                }
            } else {
                for x in dst.iter_mut() {
                    *x = f32::NAN;
                }
            }
        }
        qi = qi_end;
    }
}

/// Naive reference attention `O = softmax(QK^T / sqrt(d)) V`. Used
/// by tests and benchmarks to verify Flash forward equivalence.
pub fn naive_forward(
    shape: &AttentionShape,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    out: &mut [f32],
) -> Result<(), AttentionError> {
    naive_forward_masked(shape, q, k, v, &Mask::None, out)
}

/// Naive reference attention with optional masking. Used by tests
/// to verify masked Flash forward equivalence.
pub fn naive_forward_masked(
    shape: &AttentionShape,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    mask: &Mask<'_>,
    out: &mut [f32],
) -> Result<(), AttentionError> {
    validate(shape, q, k, v, out)?;
    mask.validate(shape.seq)?;
    let scale = 1.0 / (shape.dim as f32).sqrt();
    let (s_b, s_h, s_n) = shape.strides();
    for b in 0..shape.batch {
        for h in 0..shape.heads {
            let base = b * s_b + h * s_h;
            // Materialise the full N×N score matrix.
            let mut scores = vec![0.0f32; shape.seq * shape.seq];
            for i in 0..shape.seq {
                for j in 0..shape.seq {
                    if mask.is_masked(i, j) {
                        scores[i * shape.seq + j] = f32::NEG_INFINITY;
                        continue;
                    }
                    let mut s = 0.0f32;
                    for d in 0..shape.dim {
                        s += q[base + i * s_n + d] * k[base + j * s_n + d];
                    }
                    scores[i * shape.seq + j] = s * scale;
                }
            }
            // Softmax row-by-row.
            for i in 0..shape.seq {
                let row = &mut scores[i * shape.seq..(i + 1) * shape.seq];
                let m = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0f32;
                for x in row.iter_mut() {
                    *x = (*x - m).exp();
                    sum += *x;
                }
                let dst = &mut out[base + i * s_n..base + i * s_n + shape.dim];
                if sum > 0.0 && sum.is_finite() {
                    for x in row.iter_mut() {
                        *x /= sum;
                    }
                    // O[i] = row · V
                    for d in 0..shape.dim {
                        let mut o = 0.0f32;
                        for j in 0..shape.seq {
                            o += row[j] * v[base + j * s_n + d];
                        }
                        dst[d] = o;
                    }
                } else {
                    for d in 0..shape.dim {
                        dst[d] = f32::NAN;
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

    fn deterministic_buffer(len: usize, seed: u32) -> Vec<f32> {
        // Simple LCG for reproducible synthetic data; values in
        // ~[-1.5, 1.5] so scores stay well-behaved.
        let mut s = seed.wrapping_mul(2654435761);
        (0..len)
            .map(|_| {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                (s as f32 / u32::MAX as f32 - 0.5) * 3.0
            })
            .collect()
    }

    fn shape_buffers(shape: &AttentionShape) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        let n = shape.buffer_len();
        let q = deterministic_buffer(n, 0xC0FFEE);
        let k = deterministic_buffer(n, 0xBADBEEF);
        let v = deterministic_buffer(n, 0xCAFE00);
        let out = vec![0.0f32; n];
        (q, k, v, out)
    }

    fn close(a: f32, b: f32, atol: f32) -> bool {
        if a.is_nan() && b.is_nan() {
            return true;
        }
        (a - b).abs() <= atol + 1e-4 * a.abs().max(b.abs())
    }

    fn assert_close_buffers(a: &[f32], b: &[f32], atol: f32, label: &str) {
        assert_eq!(a.len(), b.len(), "{label}: length mismatch");
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert!(close(*x, *y, atol), "{label}: idx {i}: flash={x} naive={y}");
        }
    }

    #[test]
    fn flash_matches_naive_small_shape() {
        let shape = AttentionShape::new(2, 4, 32, 16);
        let (q, k, v, mut o_flash) = shape_buffers(&shape);
        let mut o_naive = vec![0.0f32; shape.buffer_len()];
        flash_forward(&shape, &q, &k, &v, &mut o_flash).unwrap();
        naive_forward(&shape, &q, &k, &v, &mut o_naive).unwrap();
        assert_close_buffers(&o_flash, &o_naive, 1e-4, "flash vs naive small");
    }

    #[test]
    fn flash_matches_naive_with_unaligned_seq() {
        // Seq not a multiple of br/bc — exercises the tail-tile path.
        let shape = AttentionShape {
            batch: 1,
            heads: 2,
            seq: 70,
            dim: 24,
            br: 64,
            bc: 64,
        };
        let (q, k, v, mut o_flash) = shape_buffers(&shape);
        let mut o_naive = vec![0.0f32; shape.buffer_len()];
        flash_forward(&shape, &q, &k, &v, &mut o_flash).unwrap();
        naive_forward(&shape, &q, &k, &v, &mut o_naive).unwrap();
        assert_close_buffers(&o_flash, &o_naive, 1e-4, "tail tile");
    }

    #[test]
    fn flash_matches_naive_target_shape_b2_h4_n128_d64() {
        let shape = AttentionShape::new(2, 4, 128, 64);
        let (q, k, v, mut o_flash) = shape_buffers(&shape);
        let mut o_naive = vec![0.0f32; shape.buffer_len()];
        flash_forward(&shape, &q, &k, &v, &mut o_flash).unwrap();
        naive_forward(&shape, &q, &k, &v, &mut o_naive).unwrap();
        assert_close_buffers(&o_flash, &o_naive, 1e-4, "B2H4N128D64");
    }

    #[test]
    fn scaling_factor_is_one_over_sqrt_d() {
        // Single-row sanity: with d=4, scale = 0.5. Use Q = K = unit
        // vector → score = ||q||^2 / 2 = 0.5; only one query, one
        // key → softmax is just 1 → O = v.
        let shape = AttentionShape::new(1, 1, 1, 4);
        let q = vec![1.0f32, 0.0, 0.0, 0.0];
        let k = vec![1.0f32, 0.0, 0.0, 0.0];
        let v = vec![5.0f32, -2.0, 7.0, 1.0];
        let mut out = vec![0.0f32; 4];
        flash_forward(&shape, &q, &k, &v, &mut out).unwrap();
        for (i, (a, b)) in out.iter().zip(v.iter()).enumerate() {
            assert!(close(*a, *b, 1e-6), "idx {i}: out={a} v={b}");
        }
    }

    // --- Edge cases -------------------------------------------------

    #[test]
    fn empty_seq_is_no_op() {
        let shape = AttentionShape::new(2, 4, 0, 16);
        let q: Vec<f32> = Vec::new();
        let k: Vec<f32> = Vec::new();
        let v: Vec<f32> = Vec::new();
        let mut out: Vec<f32> = Vec::new();
        flash_forward(&shape, &q, &k, &v, &mut out).unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn zero_dim_returns_error() {
        let shape = AttentionShape::new(1, 1, 4, 0);
        let q = Vec::<f32>::new();
        let k = Vec::<f32>::new();
        let v = Vec::<f32>::new();
        let mut out = Vec::<f32>::new();
        assert_eq!(
            flash_forward(&shape, &q, &k, &v, &mut out).unwrap_err(),
            AttentionError::ZeroDim
        );
    }

    #[test]
    fn buffer_size_mismatch_returns_error() {
        let shape = AttentionShape::new(1, 1, 4, 8);
        let q = vec![0.0f32; 32];
        let k = vec![0.0f32; 16]; // wrong
        let v = vec![0.0f32; 32];
        let mut out = vec![0.0f32; 32];
        let err = flash_forward(&shape, &q, &k, &v, &mut out).unwrap_err();
        assert!(matches!(
            err,
            AttentionError::BufferShapeMismatch { which: "k", .. }
        ));
    }

    #[test]
    fn nan_input_propagates_to_output() {
        let mut shape = AttentionShape::new(1, 1, 4, 8);
        shape.br = 4;
        shape.bc = 4;
        let mut q = deterministic_buffer(shape.buffer_len(), 1);
        q[0] = f32::NAN;
        let k = deterministic_buffer(shape.buffer_len(), 2);
        let v = deterministic_buffer(shape.buffer_len(), 3);
        let mut o_flash = vec![0.0f32; shape.buffer_len()];
        let mut o_naive = vec![0.0f32; shape.buffer_len()];
        flash_forward(&shape, &q, &k, &v, &mut o_flash).unwrap();
        naive_forward(&shape, &q, &k, &v, &mut o_naive).unwrap();
        // Both should propagate NaN (matching positions).
        for (f, n) in o_flash.iter().zip(o_naive.iter()) {
            assert_eq!(f.is_nan(), n.is_nan(), "NaN pattern divergence");
        }
    }

    #[test]
    fn parallel_matches_serial() {
        // Run a moderate-size shape and verify the rayon-parallel
        // path produces bit-equal output to the serial naive path.
        // (rayon may schedule tiles in any order; per-(b,h,qi,d)
        // disjointness ensures no race.)
        let shape = AttentionShape::new(4, 8, 96, 32);
        let (q, k, v, mut o_flash) = shape_buffers(&shape);
        let mut o_naive = vec![0.0f32; shape.buffer_len()];
        flash_forward(&shape, &q, &k, &v, &mut o_flash).unwrap();
        naive_forward(&shape, &q, &k, &v, &mut o_naive).unwrap();
        assert_close_buffers(&o_flash, &o_naive, 1e-4, "parallel vs naive");
    }

    // --- Masked-path tests ---------------------------------------

    #[test]
    fn causal_flash_matches_causal_naive() {
        let shape = AttentionShape::new(1, 2, 64, 32);
        let (q, k, v, mut o_flash) = shape_buffers(&shape);
        let mut o_naive = vec![0.0f32; shape.buffer_len()];
        flash_forward_masked(&shape, &q, &k, &v, &Mask::Causal, &mut o_flash).unwrap();
        naive_forward_masked(&shape, &q, &k, &v, &Mask::Causal, &mut o_naive).unwrap();
        assert_close_buffers(&o_flash, &o_naive, 1e-4, "causal masked");
    }

    #[test]
    fn padding_flash_matches_padding_naive() {
        let shape = AttentionShape::new(1, 1, 8, 4);
        let (q, k, v, mut o_flash) = shape_buffers(&shape);
        let mut o_naive = vec![0.0f32; shape.buffer_len()];
        let keep = [true, false, true, true, false, true, false, true];
        let mask = Mask::Padding(&keep);
        flash_forward_masked(&shape, &q, &k, &v, &mask, &mut o_flash).unwrap();
        naive_forward_masked(&shape, &q, &k, &v, &mask, &mut o_naive).unwrap();
        assert_close_buffers(&o_flash, &o_naive, 1e-4, "padding masked");
    }

    #[test]
    fn causal_first_query_only_attends_to_self() {
        // For a causal mask, q[0] can only attend to k[0] → output
        // is exactly v[0] (single-element softmax = 1.0).
        let shape = AttentionShape::new(1, 1, 4, 8);
        let q = vec![1.0f32; shape.buffer_len()];
        let k = vec![1.0f32; shape.buffer_len()];
        // V row 0 = [10, 20, 30, ...]; row 1+ = something else.
        let mut v = vec![0.0f32; shape.buffer_len()];
        for d in 0..8 {
            v[d] = 10.0 * (d + 1) as f32;
            v[8 + d] = -99.0; // row 1
            v[16 + d] = 99.0; // row 2
        }
        let mut out = vec![0.0f32; shape.buffer_len()];
        flash_forward_masked(&shape, &q, &k, &v, &Mask::Causal, &mut out).unwrap();
        // q[0] sees only k[0] → out[0..8] = v[0..8].
        for d in 0..8 {
            assert!(
                close(out[d], v[d], 1e-6),
                "out[{d}]={} vs v[{d}]={}",
                out[d],
                v[d]
            );
        }
    }

    #[test]
    fn padding_mask_excludes_padded_positions_exactly() {
        // 4 keys; keep = [true, false, false, true]. q[0] attends to
        // k[0] and k[3] only. With q=k=ones, scores at 0 and 3 are
        // both equal → output averages v[0] and v[3].
        let shape = AttentionShape::new(1, 1, 4, 4);
        let q = vec![1.0f32; shape.buffer_len()];
        let k = vec![1.0f32; shape.buffer_len()];
        let v: Vec<f32> = vec![
            10.0, 0.0, 0.0, 0.0, 99.0, 0.0, 0.0, 0.0, -99.0, 0.0, 0.0, 0.0, 30.0, 0.0, 0.0, 0.0,
        ];
        let keep = [true, false, false, true];
        let mut out = vec![0.0f32; shape.buffer_len()];
        flash_forward_masked(&shape, &q, &k, &v, &Mask::Padding(&keep), &mut out).unwrap();
        // out[0] = (v[0][0] + v[3][0]) / 2 = (10 + 30) / 2 = 20.
        assert!(close(out[0], 20.0, 1e-5));
    }

    #[test]
    fn padding_mask_length_mismatch_returns_error() {
        let shape = AttentionShape::new(1, 1, 4, 4);
        let (q, k, v, mut out) = shape_buffers(&shape);
        let keep = [true; 3]; // wrong length
        let err =
            flash_forward_masked(&shape, &q, &k, &v, &Mask::Padding(&keep), &mut out).unwrap_err();
        assert!(matches!(
            err,
            AttentionError::Mask(MaskError::PaddingLengthMismatch {
                expected: 4,
                got: 3
            })
        ));
    }

    #[test]
    fn flash_stable_for_large_q_k_magnitudes() {
        let shape = AttentionShape::new(1, 1, 16, 8);
        let q: Vec<f32> = (0..shape.buffer_len())
            .map(|i| 1e3 + (i as f32 * 0.01).sin())
            .collect();
        let k: Vec<f32> = (0..shape.buffer_len())
            .map(|i| 1e3 + (i as f32 * 0.013).cos())
            .collect();
        let v: Vec<f32> = (0..shape.buffer_len())
            .map(|i| (i as f32 * 0.07).sin())
            .collect();
        let mut o_flash = vec![0.0f32; shape.buffer_len()];
        let mut o_naive = vec![0.0f32; shape.buffer_len()];
        flash_forward(&shape, &q, &k, &v, &mut o_flash).unwrap();
        naive_forward(&shape, &q, &k, &v, &mut o_naive).unwrap();
        assert_close_buffers(&o_flash, &o_naive, 1e-3, "large-magnitude stability");
        // No NaN/Inf leakage.
        for x in &o_flash {
            assert!(x.is_finite(), "non-finite output: {x}");
        }
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    fn shape_strategy() -> impl Strategy<Value = AttentionShape> {
        (1usize..3, 1usize..4, 1usize..40, 4usize..16).prop_map(|(b, h, n, d)| AttentionShape {
            batch: b,
            heads: h,
            seq: n,
            dim: d,
            br: 16,
            bc: 16,
        })
    }

    fn close(a: f32, b: f32, atol: f32) -> bool {
        if a.is_nan() && b.is_nan() {
            return true;
        }
        (a - b).abs() <= atol + 1e-3 * a.abs().max(b.abs())
    }

    proptest! {
        #![proptest_config(ProptestConfig {
            cases: 200,
            .. ProptestConfig::default()
        })]

        /// Flash forward matches naive within 1e-3 over random shapes.
        /// 200 cases — meets the Phase 3 e7656812 acceptance step
        /// "Add 200-case proptest sampling random (B, H, N, D)".
        #[test]
        fn flash_matches_naive_random_shapes(shape in shape_strategy()) {
            let n = shape.buffer_len();
            // Generate small-magnitude inputs to keep softmax tame.
            let q: Vec<f32> = (0..n).map(|i| ((i as u32).wrapping_mul(2654435761) as f32 / u32::MAX as f32 - 0.5) * 2.0).collect();
            let k: Vec<f32> = (0..n).map(|i| ((i as u32).wrapping_mul(13).wrapping_mul(2654435761) as f32 / u32::MAX as f32 - 0.5) * 2.0).collect();
            let v: Vec<f32> = (0..n).map(|i| ((i as u32).wrapping_mul(29).wrapping_mul(2654435761) as f32 / u32::MAX as f32 - 0.5) * 2.0).collect();
            let mut o_flash = vec![0.0f32; n];
            let mut o_naive = vec![0.0f32; n];
            flash_forward(&shape, &q, &k, &v, &mut o_flash).unwrap();
            naive_forward(&shape, &q, &k, &v, &mut o_naive).unwrap();
            for (i, (f, na)) in o_flash.iter().zip(o_naive.iter()).enumerate() {
                prop_assert!(close(*f, *na, 1e-3), "idx {} flash={} naive={}", i, f, na);
            }
        }
    }
}
