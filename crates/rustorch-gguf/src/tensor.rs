//! GGML tensor type enumeration and per-tensor descriptor.

use crate::reader::GgufError;

/// GGML tensor data type enum, mirroring `ggml_type` in `llama.cpp`.
///
/// Block sizes (`block_size`) and per-block byte size (`type_size`) come
/// directly from `ggml.c` and are kept in sync there. We only enumerate
/// the formats we have parsers for; the rest are recognized but their
/// `type_size`/`block_size` are still correct for offset computation.
///
/// We deliberately keep the GGML canonical SCREAMING_SNAKE names
/// (Q4_K, IQ2_XXS…) — silence the lint to match upstream identifiers.
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum GgmlType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    Q8_K = 15,
    IQ2_XXS = 16,
    IQ2_XS = 17,
    IQ3_XXS = 18,
    IQ1_S = 19,
    IQ4_NL = 20,
    IQ3_S = 21,
    IQ2_S = 22,
    IQ4_XS = 23,
    I8 = 24,
    I16 = 25,
    I32 = 26,
    I64 = 27,
    F64 = 28,
    IQ1_M = 29,
    BF16 = 30,
}

impl GgmlType {
    /// Construct from raw `u32` tag, validating it is a known type.
    pub fn from_u32(v: u32) -> Result<Self, GgufError> {
        let t = match v {
            0 => Self::F32,
            1 => Self::F16,
            2 => Self::Q4_0,
            3 => Self::Q4_1,
            6 => Self::Q5_0,
            7 => Self::Q5_1,
            8 => Self::Q8_0,
            9 => Self::Q8_1,
            10 => Self::Q2_K,
            11 => Self::Q3_K,
            12 => Self::Q4_K,
            13 => Self::Q5_K,
            14 => Self::Q6_K,
            15 => Self::Q8_K,
            16 => Self::IQ2_XXS,
            17 => Self::IQ2_XS,
            18 => Self::IQ3_XXS,
            19 => Self::IQ1_S,
            20 => Self::IQ4_NL,
            21 => Self::IQ3_S,
            22 => Self::IQ2_S,
            23 => Self::IQ4_XS,
            24 => Self::I8,
            25 => Self::I16,
            26 => Self::I32,
            27 => Self::I64,
            28 => Self::F64,
            29 => Self::IQ1_M,
            30 => Self::BF16,
            other => return Err(GgufError::UnknownDtype(other)),
        };
        Ok(t)
    }

    /// Number of values represented by a single quantization block (1 for
    /// non-quantized formats).
    pub fn block_size(self) -> usize {
        match self {
            Self::F32 | Self::F16 | Self::BF16 | Self::F64 => 1,
            Self::I8 | Self::I16 | Self::I32 | Self::I64 => 1,
            Self::Q4_0 | Self::Q4_1 | Self::Q5_0 | Self::Q5_1 => 32,
            Self::Q8_0 | Self::Q8_1 => 32,
            // K-quants pack 256 values per super-block.
            Self::Q2_K
            | Self::Q3_K
            | Self::Q4_K
            | Self::Q5_K
            | Self::Q6_K
            | Self::Q8_K
            | Self::IQ2_XXS
            | Self::IQ2_XS
            | Self::IQ3_XXS
            | Self::IQ1_S
            | Self::IQ4_NL
            | Self::IQ3_S
            | Self::IQ2_S
            | Self::IQ4_XS
            | Self::IQ1_M => 256,
        }
    }

    /// Number of bytes occupied by a single quantization block.
    pub fn type_size(self) -> usize {
        match self {
            Self::F32 | Self::I32 => 4,
            Self::F16 | Self::BF16 | Self::I16 => 2,
            Self::F64 | Self::I64 => 8,
            Self::I8 => 1,
            // Q4_0 / Q4_1: 32 4-bit weights + 1 (or 2) f16 scale(s).
            Self::Q4_0 => 2 + 16,     // f16 d + 16 bytes of nibbles
            Self::Q4_1 => 2 + 2 + 16, // d + m + nibbles
            Self::Q5_0 => 2 + 4 + 16, // d + qh(u32) + qs
            Self::Q5_1 => 2 + 2 + 4 + 16,
            Self::Q8_0 => 2 + 32, // f16 scale + 32 i8
            Self::Q8_1 => 4 + 32, // (f16 d, f16 s) packed = 4 bytes + 32 i8
            // K-quants block sizes (GGML_TYPE_TRAITS).
            // Q4_K: 144 bytes per 256 weights = (256/2) qs + 12 scales + 4 (d,dmin f16x2)
            Self::Q4_K => 2 + 2 + 12 + 128,
            // Q6_K: 210 bytes / 256 weights = ql[128] + qh[64] + scales[16] + d (f16)
            Self::Q6_K => 128 + 64 + 16 + 2,
            Self::Q2_K => 2 + 2 + 16 + 64, // d, dmin f16 + scales + qs
            Self::Q3_K => 32 + 64 + 12 + 2, // hmask + qs + scales + d
            Self::Q5_K => 2 + 2 + 12 + 32 + 128, // d, dmin + scales + qh + qs
            Self::Q8_K => 4 + 256 + 32,    // d (f32) + qs + bsums i16x16
            Self::IQ2_XXS => 2 + 64,
            Self::IQ2_XS => 2 + 64 + 16,
            Self::IQ3_XXS => 2 + 64 + 32,
            Self::IQ1_S => 2 + 32 + 8,
            Self::IQ4_NL => 2 + 16,
            Self::IQ3_S => 2 + 64 + 32 + 8 + 16,
            Self::IQ2_S => 2 + 64 + 16 + 8,
            Self::IQ4_XS => 2 + 2 + 64 + 128,
            Self::IQ1_M => 32 + 16 + 8,
        }
    }

    /// Byte size of a tensor of `n_elements` values quantized with this dtype.
    pub fn byte_size_for(self, n_elements: u64) -> u64 {
        let bs = self.block_size() as u64;
        let ts = self.type_size() as u64;
        debug_assert!(n_elements % bs == 0, "n_elements must align on block_size");
        (n_elements / bs) * ts
    }

    /// True if dequantization to f32 is implemented in [`crate::dequant`].
    pub fn is_dequant_supported(self) -> bool {
        matches!(
            self,
            Self::F32 | Self::F16 | Self::BF16 | Self::Q8_0 | Self::Q4_K | Self::Q6_K
        )
    }
}

/// Per-tensor descriptor parsed from the GGUF tensor index.
#[derive(Debug, Clone)]
pub struct TensorInfo {
    /// Tensor name (e.g. `"blk.0.attn_q.weight"`).
    pub name: String,
    /// Shape in row-major order — innermost dim last.
    /// For a `[out, in]` linear weight, GGUF stores `[in, out]`.
    pub shape: Vec<u64>,
    /// GGML data type.
    pub dtype: GgmlType,
    /// Offset *inside the tensor data section* (not absolute file offset).
    /// Add [`crate::GgufFile::data_offset`] to get the absolute byte offset.
    pub offset: u64,
}

impl TensorInfo {
    /// Total number of scalar elements (product of `shape`).
    pub fn n_elements(&self) -> u64 {
        self.shape.iter().product()
    }

    /// Byte size on disk, given the dtype's block size.
    pub fn byte_size(&self) -> u64 {
        self.dtype.byte_size_for(self.n_elements())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_types_roundtrip() {
        for tag in [0u32, 1, 8, 12, 14, 30] {
            let t = GgmlType::from_u32(tag).unwrap();
            assert_eq!(t as u32, tag);
        }
    }

    #[test]
    fn unknown_type_errors() {
        assert!(GgmlType::from_u32(999).is_err());
    }

    #[test]
    fn block_sizes_match_known_constants() {
        // K-quants are always 256 weights per block.
        assert_eq!(GgmlType::Q4_K.block_size(), 256);
        assert_eq!(GgmlType::Q6_K.block_size(), 256);
        // Legacy quants are 32-wide.
        assert_eq!(GgmlType::Q8_0.block_size(), 32);
        // Dense formats.
        assert_eq!(GgmlType::F32.block_size(), 1);
        assert_eq!(GgmlType::F16.block_size(), 1);
    }

    #[test]
    fn q4_k_byte_size_for_4096_x_4096() {
        // 4096 * 4096 = 16_777_216 weights. Q4_K is 144 bytes / 256 weights.
        let n = 4096u64 * 4096;
        let b = GgmlType::Q4_K.byte_size_for(n);
        assert_eq!(b, n / 256 * 144);
    }
}
