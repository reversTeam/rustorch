//! Block-wise dequantization to f32.
//!
//! We mirror the reference layouts from `ggml-quants.h` / `ggml-quants.c`
//! in `llama.cpp` for the formats that appear in **Q4_K_M** quantized
//! Qwen / Llama checkpoints, namely:
//!
//! - `F32`, `F16`, `BF16` — passthrough/widen.
//! - `Q8_0`        — 32-wide, single f16 scale per block.
//! - `Q4_K`        — 256-wide super-block with 8 sub-blocks, 6-bit scales/mins.
//! - `Q6_K`        — 256-wide super-block, 6-bit weights, 8-bit scales.
//!
//! All other GGML types return [`DequantError::Unsupported`].
//!
//! ### Block layouts (see `ggml-quants.h`)
//!
//! ```text
//! struct block_q8_0 {     // 34 bytes for 32 weights
//!     ggml_fp16_t d;          //   2 B  scale
//!     int8_t      qs[32];     //  32 B  signed 8-bit weights
//! };
//!
//! struct block_q4_K {     // 144 bytes for 256 weights
//!     ggml_fp16_t d;          //   2 B  super-block scale for scales
//!     ggml_fp16_t dmin;       //   2 B  super-block scale for mins
//!     uint8_t     scales[12]; //  12 B  6-bit packed (scale, min) for 8 sub-blocks
//!     uint8_t     qs[128];    // 128 B  4-bit weights, 32 per sub-block
//! };
//!
//! struct block_q6_K {     // 210 bytes for 256 weights
//!     uint8_t     ql[128];    // 128 B  lower 4 bits of each weight
//!     uint8_t     qh[64];     //  64 B  upper 2 bits of each weight
//!     int8_t      scales[16]; //  16 B  per-16 sub-block 8-bit scales
//!     ggml_fp16_t d;          //   2 B  super-block scale
//! };
//! ```

use half::{bf16, f16};
use thiserror::Error;

use crate::tensor::{GgmlType, TensorInfo};

#[derive(Error, Debug)]
pub enum DequantError {
    #[error("unsupported dtype for dequantization: {0:?}")]
    Unsupported(GgmlType),

    #[error("buffer too small: needed {needed} bytes, have {have}")]
    BufferTooSmall { needed: usize, have: usize },

    #[error("output slice has wrong length: expected {expected}, got {got}")]
    OutputSize { expected: usize, got: usize },
}

/// Block-byte sizes — keep in sync with `GgmlType::type_size`.
const Q8_0_BS: usize = 32;
const Q8_0_BYTES: usize = 2 + 32;

/// Q4_K / Q6_K super-block size — 256 weights per block.
pub const QK_K: usize = 256;
/// Wire-format size of one Q4_K super-block (144 bytes: 2 d + 2 dmin + 12 scales + 128 nibbles).
pub const Q4_K_BYTES: usize = 2 + 2 + 12 + 128;
/// Wire-format size of one Q6_K super-block (210 bytes: 128 ql + 64 qh + 16 i8 scales + 2 d_f16).
pub const Q6_K_BYTES: usize = 128 + 64 + 16 + 2;

/// Total number of f32 elements produced for a tensor of given shape.
pub fn num_elements(t: &TensorInfo) -> usize {
    t.n_elements() as usize
}

/// Dequantize a complete tensor blob into a fresh `Vec<f32>`.
pub fn dequant_to_f32(t: &TensorInfo, src: &[u8]) -> Result<Vec<f32>, DequantError> {
    let n = num_elements(t);
    let mut out = vec![0f32; n];
    dequantize_block_chunk(t.dtype, src, &mut out)?;
    Ok(out)
}

/// Dequantize a contiguous chunk of blocks into `dst`. `dst.len()` must
/// equal the number of weights represented by `src` (i.e. blocks × bs).
pub fn dequantize_block_chunk(
    dtype: GgmlType,
    src: &[u8],
    dst: &mut [f32],
) -> Result<(), DequantError> {
    match dtype {
        GgmlType::F32 => dequant_f32(src, dst),
        GgmlType::F16 => dequant_f16(src, dst),
        GgmlType::BF16 => dequant_bf16(src, dst),
        GgmlType::Q8_0 => dequant_q8_0(src, dst),
        GgmlType::Q4_K => dequant_q4_k(src, dst),
        GgmlType::Q6_K => dequant_q6_k(src, dst),
        other => Err(DequantError::Unsupported(other)),
    }
}

// =============================================================================
// Dense formats
// =============================================================================

fn dequant_f32(src: &[u8], dst: &mut [f32]) -> Result<(), DequantError> {
    let needed = dst.len() * 4;
    if src.len() < needed {
        return Err(DequantError::BufferTooSmall {
            needed,
            have: src.len(),
        });
    }
    for (i, c) in src[..needed].chunks_exact(4).enumerate() {
        dst[i] = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
    }
    Ok(())
}

fn dequant_f16(src: &[u8], dst: &mut [f32]) -> Result<(), DequantError> {
    let needed = dst.len() * 2;
    if src.len() < needed {
        return Err(DequantError::BufferTooSmall {
            needed,
            have: src.len(),
        });
    }
    for (i, c) in src[..needed].chunks_exact(2).enumerate() {
        let h = f16::from_le_bytes([c[0], c[1]]);
        dst[i] = h.to_f32();
    }
    Ok(())
}

fn dequant_bf16(src: &[u8], dst: &mut [f32]) -> Result<(), DequantError> {
    let needed = dst.len() * 2;
    if src.len() < needed {
        return Err(DequantError::BufferTooSmall {
            needed,
            have: src.len(),
        });
    }
    for (i, c) in src[..needed].chunks_exact(2).enumerate() {
        let h = bf16::from_le_bytes([c[0], c[1]]);
        dst[i] = h.to_f32();
    }
    Ok(())
}

// =============================================================================
// Q8_0  —  32 weights / 34 bytes
// =============================================================================

fn dequant_q8_0(src: &[u8], dst: &mut [f32]) -> Result<(), DequantError> {
    if dst.len() % Q8_0_BS != 0 {
        return Err(DequantError::OutputSize {
            expected: dst.len() / Q8_0_BS * Q8_0_BS,
            got: dst.len(),
        });
    }
    let nb = dst.len() / Q8_0_BS;
    let needed = nb * Q8_0_BYTES;
    if src.len() < needed {
        return Err(DequantError::BufferTooSmall {
            needed,
            have: src.len(),
        });
    }

    for b in 0..nb {
        let off = b * Q8_0_BYTES;
        let d = f16::from_le_bytes([src[off], src[off + 1]]).to_f32();
        let qs = &src[off + 2..off + 2 + 32];
        let out = &mut dst[b * Q8_0_BS..(b + 1) * Q8_0_BS];
        for i in 0..32 {
            out[i] = (qs[i] as i8 as f32) * d;
        }
    }
    Ok(())
}

// =============================================================================
// Q4_K  —  256 weights / 144 bytes
//
// Reference: `dequantize_row_q4_K` in ggml-quants.c.
// Each super-block has 8 sub-blocks × 32 weights. For each sub-block i in 0..8
// we have a 6-bit scale `sc[i]` and a 6-bit min `m[i]`. The dequantized
// value of weight q (4-bit) is:
//        d * sc[i] * q  -  dmin * m[i]
// The 6-bit (sc, m) pairs are packed into 12 bytes via:
//   for i in 0..4:
//       sc[i]   = scales[i] & 0x3F
//       m[i]    = scales[i+4] & 0x3F
//       sc[i+4] = (scales[i+8] & 0x0F) | ((scales[i]   >> 6) << 4)
//       m[i+4]  = (scales[i+8] >>   4) | ((scales[i+4] >> 6) << 4)
// =============================================================================

#[inline]
fn unpack_q4_k_sc_m(scales: &[u8; 12]) -> ([u8; 8], [u8; 8]) {
    // sc_out and m_out are the 6-bit (scale, min) pairs for the 8 sub-blocks.
    let mut sc = [0u8; 8];
    let mut m = [0u8; 8];
    for i in 0..4 {
        sc[i] = scales[i] & 0x3F;
        m[i] = scales[i + 4] & 0x3F;
        sc[i + 4] = (scales[i + 8] & 0x0F) | ((scales[i] >> 6) << 4);
        m[i + 4] = (scales[i + 8] >> 4) | ((scales[i + 4] >> 6) << 4);
    }
    (sc, m)
}

/// Dequantise one Q4_K super-block (256 weights from 144 bytes) into
/// `dst`. Public so the direct-Q4_K matmul path
/// (`crate::sgemv_q4k`) can call it block-by-block.
pub fn dequant_q4_k(src: &[u8], dst: &mut [f32]) -> Result<(), DequantError> {
    if dst.len() % QK_K != 0 {
        return Err(DequantError::OutputSize {
            expected: dst.len() / QK_K * QK_K,
            got: dst.len(),
        });
    }
    let nb = dst.len() / QK_K;
    let needed = nb * Q4_K_BYTES;
    if src.len() < needed {
        return Err(DequantError::BufferTooSmall {
            needed,
            have: src.len(),
        });
    }

    for b in 0..nb {
        let off = b * Q4_K_BYTES;
        let d = f16::from_le_bytes([src[off], src[off + 1]]).to_f32();
        let dmin = f16::from_le_bytes([src[off + 2], src[off + 3]]).to_f32();

        let mut scales12 = [0u8; 12];
        scales12.copy_from_slice(&src[off + 4..off + 16]);
        let (sc, m) = unpack_q4_k_sc_m(&scales12);

        let qs = &src[off + 16..off + 16 + 128];
        let out = &mut dst[b * QK_K..(b + 1) * QK_K];

        // 8 sub-blocks of 32 weights. Pairs (j, j+1) of sub-blocks share
        // 32 nibble-packed bytes: low nibble = sub-block j, high nibble = j+1.
        for j_pair in 0..4 {
            let j0 = 2 * j_pair;
            let j1 = 2 * j_pair + 1;
            let nibbles = &qs[j_pair * 32..(j_pair + 1) * 32];

            let scale0 = d * sc[j0] as f32;
            let min0 = dmin * m[j0] as f32;
            let scale1 = d * sc[j1] as f32;
            let min1 = dmin * m[j1] as f32;

            // Lower nibble → sub-block j0.
            for k in 0..32 {
                let q = (nibbles[k] & 0x0F) as f32;
                out[j0 * 32 + k] = scale0 * q - min0;
            }
            // Upper nibble → sub-block j1.
            for k in 0..32 {
                let q = (nibbles[k] >> 4) as f32;
                out[j1 * 32 + k] = scale1 * q - min1;
            }
        }
    }
    Ok(())
}

// =============================================================================
// Q6_K  —  256 weights / 210 bytes
//
// Layout: ql[128] (low 4 bits) + qh[64] (high 2 bits) + scales[16] i8 + d (f16).
// 256 weights are split into 16 sub-blocks of 16 weights each.
// For sub-block i in 0..16:
//     scale_i = d * scales[i]
//     for each of the 16 weights `q`:
//         q6 = (low_4_bits | (high_2_bits << 4)) - 32   // signed 6-bit
//         out = scale_i * q6
//
// The packing of 256 6-bit values into ql+qh follows ggml-quants.c
// `dequantize_row_q6_K`:
//
//   for n in (0..256).step_by(128):  # two 128-weight halves
//       ql_half = &ql[n/2 ..]      # 64 bytes
//       qh_half = &qh[n/4 ..]      # 32 bytes
//       sc      = &scales[n/16 ..] # 8 entries (sub-blocks 0..8 then 8..16)
//       for l in 0..32:
//           q[l +  0] = (ql_half[l   ] & 0x0F) | ((qh_half[l] >> 0) & 3) << 4
//           q[l + 32] = (ql_half[l+32] & 0x0F) | ((qh_half[l] >> 2) & 3) << 4
//           q[l + 64] = (ql_half[l   ] >>  4 ) | ((qh_half[l] >> 4) & 3) << 4
//           q[l + 96] = (ql_half[l+32] >>  4 ) | ((qh_half[l] >> 6) & 3) << 4
//       # then subtract 32 and multiply by per-sub-block scale.
// =============================================================================

fn dequant_q6_k(src: &[u8], dst: &mut [f32]) -> Result<(), DequantError> {
    if dst.len() % QK_K != 0 {
        return Err(DequantError::OutputSize {
            expected: dst.len() / QK_K * QK_K,
            got: dst.len(),
        });
    }
    let nb = dst.len() / QK_K;
    let needed = nb * Q6_K_BYTES;
    if src.len() < needed {
        return Err(DequantError::BufferTooSmall {
            needed,
            have: src.len(),
        });
    }

    for b in 0..nb {
        let off = b * Q6_K_BYTES;
        let ql = &src[off..off + 128];
        let qh = &src[off + 128..off + 128 + 64];
        let scales_i8 = &src[off + 192..off + 192 + 16];
        let d = f16::from_le_bytes([src[off + 208], src[off + 209]]).to_f32();

        let out = &mut dst[b * QK_K..(b + 1) * QK_K];

        // Two halves of 128 weights each.
        for half in 0..2 {
            let ql_h = &ql[half * 64..half * 64 + 64];
            let qh_h = &qh[half * 32..half * 32 + 32];
            let sc_h = &scales_i8[half * 8..half * 8 + 8];
            let out_h = &mut out[half * 128..half * 128 + 128];

            // Scales come in groups of 16-element sub-blocks. Within this
            // 128-weight half there are 8 sub-blocks (sc_h[0..8]) and the
            // 128 weights map to sub-blocks like:
            //   weight idx  : 0..16   16..32  32..48  48..64
            //                 64..80  80..96  96..112 112..128
            //   sub-block i :   0       1       2       3
            //                   4       5       6       7
            for l in 0..32 {
                let qhh = qh_h[l];
                let q1 = ((ql_h[l] & 0x0F) as i32) | ((qhh & 0x03) as i32) << 4;
                let q2 = ((ql_h[l + 32] & 0x0F) as i32) | (((qhh >> 2) & 0x03) as i32) << 4;
                let q3 = ((ql_h[l] >> 4) as i32) | (((qhh >> 4) & 0x03) as i32) << 4;
                let q4 = ((ql_h[l + 32] >> 4) as i32) | (((qhh >> 6) & 0x03) as i32) << 4;

                // CRITICAL: Q6_K scales are stored as int8_t (signed).
                // Reading as u8 then casting `as i32` zero-extends, which
                // gives 0..255 instead of -128..127. Cast through `i8`
                // first to sign-extend correctly. Without this, ~50% of
                // the dequantized values have flipped magnitude (the
                // ones whose scale byte has bit 7 set), and matmuls on
                // Q6_K weights (e.g. attn_v in K_M quantization) produce
                // garbage that subtly amplifies through the residual
                // stream and collapses the model output to a single
                // token by layer ~3.
                let s1 = d * (sc_h[l / 16] as i8) as i32 as f32;
                let s2 = d * (sc_h[2 + l / 16] as i8) as i32 as f32;
                let s3 = d * (sc_h[4 + l / 16] as i8) as i32 as f32;
                let s4 = d * (sc_h[6 + l / 16] as i8) as i32 as f32;

                out_h[l] = s1 * (q1 - 32) as f32;
                out_h[l + 32] = s2 * (q2 - 32) as f32;
                out_h[l + 64] = s3 * (q3 - 32) as f32;
                out_h[l + 96] = s4 * (q4 - 32) as f32;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dequant_f32_passthrough() {
        let vals = [1.5_f32, -2.25, 7.0, 0.125];
        let mut bytes = Vec::new();
        for v in vals {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        let mut out = [0f32; 4];
        dequant_f32(&bytes, &mut out).unwrap();
        assert_eq!(out, vals);
    }

    #[test]
    fn dequant_q8_0_block_known_values() {
        // d=0.5, weights = -64,-32,...,62,63 step ~unused, we'll just use ramp.
        let mut buf = vec![0u8; Q8_0_BYTES];
        let d = 0.5_f32;
        let dh = f16::from_f32(d).to_le_bytes();
        buf[0] = dh[0];
        buf[1] = dh[1];
        for i in 0..32 {
            // values are i8 in [-128, 127]
            buf[2 + i] = ((i as i8) - 16) as u8;
        }

        let mut out = [0f32; 32];
        dequant_q8_0(&buf, &mut out).unwrap();
        for i in 0..32 {
            let want = ((i as i8 - 16) as f32) * d;
            assert!((out[i] - want).abs() < 1e-3);
        }
    }

    #[test]
    fn dequant_q4_k_zero_block_is_zero() {
        // Construct a Q4_K block with d=0 dmin=0 and any data — dequantizes to 0.
        let mut buf = vec![0u8; Q4_K_BYTES];
        // Some non-zero scale bytes to ensure we still test the unpacking path.
        for i in 4..16 {
            buf[i] = 0x33;
        }
        for i in 16..16 + 128 {
            buf[i] = 0xAB;
        }
        let mut out = vec![0f32; QK_K];
        dequant_q4_k(&buf, &mut out).unwrap();
        for v in &out {
            assert_eq!(*v, 0.0);
        }
    }

    #[test]
    fn dequant_q4_k_sc_m_unpack() {
        // Hand-pack an example: sc[0..4] = [1,2,3,4], m[0..4] = [5,6,7,8],
        // sc[4..8] = [9,10,11,12], m[4..8] = [13,14,15,16].
        // 6-bit values fit in [0,63].
        let sc_in = [1u8, 2, 3, 4, 9, 10, 11, 12];
        let m_in = [5u8, 6, 7, 8, 13, 14, 15, 16];

        let mut packed = [0u8; 12];
        for i in 0..4 {
            packed[i] = sc_in[i] | ((sc_in[i + 4] >> 4) << 6);
            packed[i + 4] = m_in[i] | ((m_in[i + 4] >> 4) << 6);
            packed[i + 8] = (sc_in[i + 4] & 0x0F) | ((m_in[i + 4] & 0x0F) << 4);
        }

        let (sc, m) = unpack_q4_k_sc_m(&packed);
        assert_eq!(sc, sc_in);
        assert_eq!(m, m_in);
    }

    #[test]
    fn dequant_q6_k_zero_block_is_minus_32_times_zero() {
        // d=0 → all weights dequantize to 0.
        let buf = vec![0u8; Q6_K_BYTES];
        let mut out = vec![0f32; QK_K];
        dequant_q6_k(&buf, &mut out).unwrap();
        for v in &out {
            assert_eq!(*v, 0.0);
        }
    }

    #[test]
    fn dequant_q6_k_negative_scales_sign_extend() {
        // REGRESSION TEST for the sign-extension bug in Q6_K dequant.
        //
        // Q6_K scales are stored as int8_t (signed), but they live in a
        // `&[u8]` buffer. Casting `u8 as i32` zero-extends, giving 0..255
        // instead of -128..127. The fix is to cast through `i8` first.
        //
        // This test triggers the bug by setting all scales to -1 (= 0xFF
        // in u8) and all weights to q6=33 (so q6-32 = 1). The expected
        // dequantized value is `d * scale * (q6 - 32) = 1.0 * (-1) * 1
        // = -1.0` for every weight.
        //
        // With the bug (zero-extend): output would be `1.0 * 255 * 1 =
        // +255.0` — wildly wrong both in magnitude (×255) and sign.
        //
        // Without this fix, real Qwen3 / Llama Q4_K_M GGUFs (which use
        // Q6_K for half the FFN/attention V weights) collapse to a single
        // garbage token by layer ~3 due to ~50% of Q6_K weights having a
        // negative scale byte that gets misread as a large positive.
        let mut buf = vec![0u8; Q6_K_BYTES];
        // ql: low nibble = 1, high nibble = 1 → both q6 nibbles contribute 1.
        for v in buf.iter_mut().take(128) {
            *v = 0x11;
        }
        // qh: each byte = 0b10_10_10_10 = 0xAA → all four 2-bit fields = 2.
        // Combined with low=1 high=2: q6 = 1 | (2 << 4) = 33.
        for v in buf.iter_mut().take(192).skip(128) {
            *v = 0xAA;
        }
        // scales: all = -1 (i8) = 0xFF (u8).
        for v in buf.iter_mut().take(208).skip(192) {
            *v = 0xFF;
        }
        // d = f16(1.0).
        let dh = f16::from_f32(1.0).to_le_bytes();
        buf[208] = dh[0];
        buf[209] = dh[1];

        let mut out = vec![0f32; QK_K];
        dequant_q6_k(&buf, &mut out).unwrap();

        for (i, v) in out.iter().enumerate() {
            assert!(
                (v - (-1.0_f32)).abs() < 1e-3,
                "weight {i}: expected -1.0 (signed scale), got {v} \
                 — Q6_K scales are int8_t and must be sign-extended",
            );
        }
    }

    #[test]
    fn dequant_q6_k_constant_scale_centered() {
        // Build a block with d=1, all scales=1, all ql/qh such that q6=32 → output=0.
        // For q6=(low|high<<4)-32 = 32 we need low=0 high=2 → ql=0 qh = 2 (in low 2 bits).
        // qh shape: each byte covers 4 weights via shifts 0,2,4,6.
        // Set all qh bytes = 0b10_10_10_10 = 0xAA → all four 2-bit fields = 2.
        let mut buf = vec![0u8; Q6_K_BYTES];
        for v in buf.iter_mut().take(192).skip(128) {
            *v = 0xAA;
        }
        // scales i8 = 1.
        for i in 192..208 {
            buf[i] = 1;
        }
        let dh = f16::from_f32(1.0).to_le_bytes();
        buf[208] = dh[0];
        buf[209] = dh[1];

        let mut out = vec![0f32; QK_K];
        dequant_q6_k(&buf, &mut out).unwrap();
        for v in &out {
            assert_eq!(*v, 0.0, "expected centered zero, got {v}");
        }
    }
}
