//! Direct sgemv on Q4_K-quantised weights — no f32 dequantisation
//! to RAM. The dominant cost in `rustorch-llm` decode is reading
//! the FFN/attention weight matrices (50 MB Q4_K vs 712 MB f32 per
//! `gate_up`-style projection on Qwen3-14B). Reading the Q4_K bytes
//! directly cuts DRAM bandwidth by ~14×. The compute itself stays
//! in registers: each 256-weight super-block is dequantised on the
//! fly into a stack buffer, then dot-producted with the matching
//! slice of the input vector.
//!
//! ## Layout
//!
//! GGUF stores `ffn_gate.weight` etc. as `ne = [K, N]` (innermost K
//! varies fastest), which in numpy / row-major terms means `[N, K]`
//! — N "rows" of K weights, where each row holds `K / 256` Q4_K
//! super-blocks contiguously. A sgemv `y[n_idx] = sum_k x[k] *
//! W[n_idx, k]` walks the 144-byte super-blocks of row `n_idx`.

use crate::dequant::{dequant_q4_k, Q4_K_BYTES, QK_K};
use crate::DequantError;

/// Direct sgemv on Q4_K weights.
///
/// `x`: input activation `[K]` (f32).
/// `w_bytes`: Q4_K row-major weights `[N, K]` — `n_idx`-th row at
/// offset `n_idx * (K / 256) * 144`.
/// `y`: output `[N]` (f32). Existing contents are overwritten.
///
/// `K` must be a multiple of 256 (Q4_K super-block size).
pub fn sgemv_q4_k(
    x: &[f32],
    w_bytes: &[u8],
    y: &mut [f32],
    k: usize,
    n: usize,
) -> Result<(), DequantError> {
    if k % QK_K != 0 {
        return Err(DequantError::OutputSize {
            expected: (k / QK_K) * QK_K,
            got: k,
        });
    }
    let blocks_per_row = k / QK_K;
    let bytes_per_row = blocks_per_row * Q4_K_BYTES;
    let needed = n * bytes_per_row;
    if w_bytes.len() < needed {
        return Err(DequantError::BufferTooSmall {
            needed,
            have: w_bytes.len(),
        });
    }
    if x.len() != k {
        return Err(DequantError::OutputSize {
            expected: k,
            got: x.len(),
        });
    }
    if y.len() != n {
        return Err(DequantError::OutputSize {
            expected: n,
            got: y.len(),
        });
    }
    // Stack buffer for one super-block of dequantised f32 values.
    // Lives in registers / L1 — never spills to RAM.
    let mut block_buf = [0.0_f32; QK_K];
    for n_idx in 0..n {
        let mut acc = 0.0_f32;
        let row_off = n_idx * bytes_per_row;
        for blk in 0..blocks_per_row {
            let blk_off = row_off + blk * Q4_K_BYTES;
            dequant_q4_k(&w_bytes[blk_off..blk_off + Q4_K_BYTES], &mut block_buf)?;
            // Dot product over 256 weights — the compiler auto-
            // vectorises this with target-cpu=native on Apple
            // Silicon (NEON FMA). Manual NEON intrinsics in a
            // follow-up commit.
            let x_slice = &x[blk * QK_K..(blk + 1) * QK_K];
            for i in 0..QK_K {
                acc += x_slice[i] * block_buf[i];
            }
        }
        y[n_idx] = acc;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dequant::dequant_to_f32;
    use crate::tensor::{GgmlType, TensorInfo};
    use half::f16;

    fn build_one_block_per_row(n_rows: usize, k: usize) -> Vec<u8> {
        // n_rows × (k / 256) Q4_K blocks.
        // Each block: random-ish but deterministic. Encode with
        // d=1, dmin=0, scales=[i+1; 8], mins=[0; 8], qs[i] = i*7 mod 256.
        let blocks_per_row = k / QK_K;
        let total_bytes = n_rows * blocks_per_row * Q4_K_BYTES;
        let mut buf = vec![0u8; total_bytes];
        let dh = f16::from_f32(1.0).to_le_bytes();
        let dminh = f16::from_f32(0.0).to_le_bytes();
        for n_idx in 0..n_rows {
            for blk in 0..blocks_per_row {
                let off = (n_idx * blocks_per_row + blk) * Q4_K_BYTES;
                buf[off] = dh[0];
                buf[off + 1] = dh[1];
                buf[off + 2] = dminh[0];
                buf[off + 3] = dminh[1];
                // scales 6-bit per sub-block (8 sub-blocks)
                // pack so sc[i] = i + 1, m[i] = 0. Layout matches
                // unpack_q4_k_sc_m: scales[0..4]=sc[0..4], scales[4..8]=m[0..4],
                // scales[8..12]=high bits of sc[4..8] and m[4..8].
                for i in 0..4 {
                    buf[off + 4 + i] = (i as u8 + 1) & 0x3F;
                    buf[off + 4 + i + 4] = 0;
                    buf[off + 4 + 8 + i] = (i as u8 + 5) & 0x0F;
                }
                // qs: 256 4-bit values packed into 128 bytes.
                // Pair sub-blocks (j, j+1) share 32 bytes: low nibble
                // = sub-block j weight, high nibble = sub-block j+1
                // weight. Set all weights to 8 (mid-range nibble).
                for i in 0..128 {
                    buf[off + 16 + i] = 0x88;
                }
            }
        }
        buf
    }

    fn naive_sgemv_via_dequant(
        x: &[f32],
        w_bytes: &[u8],
        k: usize,
        n: usize,
    ) -> Result<Vec<f32>, DequantError> {
        // Dequant the entire matrix to f32 then do a plain sgemv.
        let info = TensorInfo {
            name: "test".to_string(),
            dtype: GgmlType::Q4_K,
            shape: vec![k as u64, n as u64],
            offset: 0,
        };
        let w_f32 = dequant_to_f32(&info, w_bytes)?;
        // w_f32 is row-major [N, K] (since GGUF ne[0] = fastest = K).
        let mut y = vec![0.0_f32; n];
        for n_idx in 0..n {
            let mut acc = 0.0_f32;
            for k_idx in 0..k {
                acc += x[k_idx] * w_f32[n_idx * k + k_idx];
            }
            y[n_idx] = acc;
        }
        Ok(y)
    }

    #[test]
    fn sgemv_q4k_matches_dequant_then_sgemv() {
        let n = 16;
        let k = 256; // exactly one super-block per row
        let w_bytes = build_one_block_per_row(n, k);
        let x: Vec<f32> = (0..k).map(|i| ((i as f32 + 1.0) * 0.001).sin()).collect();

        let y_ref = naive_sgemv_via_dequant(&x, &w_bytes, k, n).unwrap();
        let mut y_got = vec![0.0_f32; n];
        sgemv_q4_k(&x, &w_bytes, &mut y_got, k, n).unwrap();

        for i in 0..n {
            assert!(
                (y_ref[i] - y_got[i]).abs() < 1e-3,
                "row {i}: ref={} got={}",
                y_ref[i],
                y_got[i]
            );
        }
    }

    #[test]
    fn sgemv_q4k_multi_block_per_row() {
        let n = 4;
        let k = 1024; // 4 blocks per row
        let w_bytes = build_one_block_per_row(n, k);
        let x: Vec<f32> = (0..k).map(|i| ((i as f32 + 1.0) * 0.001).sin()).collect();

        let y_ref = naive_sgemv_via_dequant(&x, &w_bytes, k, n).unwrap();
        let mut y_got = vec![0.0_f32; n];
        sgemv_q4_k(&x, &w_bytes, &mut y_got, k, n).unwrap();

        for i in 0..n {
            assert!(
                (y_ref[i] - y_got[i]).abs() < 1e-3,
                "row {i}: ref={} got={}",
                y_ref[i],
                y_got[i]
            );
        }
    }
}
