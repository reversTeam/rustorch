//! GGUF file parser — header, metadata KV, and tensor index.
//!
//! The reader takes a buffered byte slice (typically the result of an
//! `mmap` or a `Vec<u8>` from `fs::read`) and parses it lazily into a
//! [`GgufFile`]. Tensor *data* itself is not copied — only the index is
//! materialized; raw block bytes are exposed via
//! [`GgufFile::tensor_bytes`].
//!
//! This implementation targets GGUF v2 and v3 (the only two variants
//! produced by recent `llama.cpp` quantizers). v1 is not supported.

use std::fs;
use std::path::Path;
use thiserror::Error;

use crate::metadata::{MetaArray, MetaValue, MetaValueType, Metadata};
use crate::tensor::{GgmlType, TensorInfo};
use crate::{GGUF_DEFAULT_ALIGNMENT, GGUF_MAGIC};

#[derive(Error, Debug)]
pub enum GgufError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),

    #[error("not a GGUF file: bad magic 0x{0:08x}")]
    BadMagic(u32),

    #[error("unsupported GGUF version: {0} (supported: 2, 3)")]
    UnsupportedVersion(u32),

    #[error("unknown metadata value type tag: {0}")]
    UnknownMetaType(u32),

    #[error("unknown ggml dtype: {0}")]
    UnknownDtype(u32),

    #[error("truncated read: needed {needed} bytes, have {have}")]
    Truncated { needed: usize, have: usize },

    #[error("invalid utf-8 in {field}")]
    BadUtf8 { field: &'static str },

    #[error(
        "tensor offset out of bounds: tensor='{name}' offset={offset} data_section={data_len}"
    )]
    TensorOffsetOob {
        name: String,
        offset: u64,
        data_len: u64,
    },
}

/// Internal cursor over the file bytes — advances strictly forward.
struct Cursor<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    #[inline]
    fn need(&self, n: usize) -> Result<(), GgufError> {
        if self.pos + n > self.buf.len() {
            Err(GgufError::Truncated {
                needed: n,
                have: self.buf.len().saturating_sub(self.pos),
            })
        } else {
            Ok(())
        }
    }

    #[inline]
    fn read_u8(&mut self) -> Result<u8, GgufError> {
        self.need(1)?;
        let v = self.buf[self.pos];
        self.pos += 1;
        Ok(v)
    }

    #[inline]
    fn read_i8(&mut self) -> Result<i8, GgufError> {
        Ok(self.read_u8()? as i8)
    }

    #[inline]
    fn read_u16(&mut self) -> Result<u16, GgufError> {
        self.need(2)?;
        let v = u16::from_le_bytes([self.buf[self.pos], self.buf[self.pos + 1]]);
        self.pos += 2;
        Ok(v)
    }

    #[inline]
    fn read_i16(&mut self) -> Result<i16, GgufError> {
        Ok(self.read_u16()? as i16)
    }

    #[inline]
    fn read_u32(&mut self) -> Result<u32, GgufError> {
        self.need(4)?;
        let mut a = [0u8; 4];
        a.copy_from_slice(&self.buf[self.pos..self.pos + 4]);
        self.pos += 4;
        Ok(u32::from_le_bytes(a))
    }

    #[inline]
    fn read_i32(&mut self) -> Result<i32, GgufError> {
        Ok(self.read_u32()? as i32)
    }

    #[inline]
    fn read_u64(&mut self) -> Result<u64, GgufError> {
        self.need(8)?;
        let mut a = [0u8; 8];
        a.copy_from_slice(&self.buf[self.pos..self.pos + 8]);
        self.pos += 8;
        Ok(u64::from_le_bytes(a))
    }

    #[inline]
    fn read_i64(&mut self) -> Result<i64, GgufError> {
        Ok(self.read_u64()? as i64)
    }

    #[inline]
    fn read_f32(&mut self) -> Result<f32, GgufError> {
        Ok(f32::from_bits(self.read_u32()?))
    }

    #[inline]
    fn read_f64(&mut self) -> Result<f64, GgufError> {
        Ok(f64::from_bits(self.read_u64()?))
    }

    #[inline]
    fn read_bool(&mut self) -> Result<bool, GgufError> {
        Ok(self.read_u8()? != 0)
    }

    /// GGUF strings are `u64 length` + UTF-8 bytes (no NUL terminator).
    fn read_string(&mut self, field: &'static str) -> Result<String, GgufError> {
        let len = self.read_u64()? as usize;
        self.need(len)?;
        let s = std::str::from_utf8(&self.buf[self.pos..self.pos + len])
            .map_err(|_| GgufError::BadUtf8 { field })?
            .to_string();
        self.pos += len;
        Ok(s)
    }

    fn read_value(&mut self, vt: MetaValueType) -> Result<MetaValue, GgufError> {
        Ok(match vt {
            MetaValueType::U8 => MetaValue::U8(self.read_u8()?),
            MetaValueType::I8 => MetaValue::I8(self.read_i8()?),
            MetaValueType::U16 => MetaValue::U16(self.read_u16()?),
            MetaValueType::I16 => MetaValue::I16(self.read_i16()?),
            MetaValueType::U32 => MetaValue::U32(self.read_u32()?),
            MetaValueType::I32 => MetaValue::I32(self.read_i32()?),
            MetaValueType::F32 => MetaValue::F32(self.read_f32()?),
            MetaValueType::Bool => MetaValue::Bool(self.read_bool()?),
            MetaValueType::String => MetaValue::String(self.read_string("meta.string")?),
            MetaValueType::U64 => MetaValue::U64(self.read_u64()?),
            MetaValueType::I64 => MetaValue::I64(self.read_i64()?),
            MetaValueType::F64 => MetaValue::F64(self.read_f64()?),
            MetaValueType::Array => MetaValue::Array(self.read_array()?),
        })
    }

    fn read_array(&mut self) -> Result<MetaArray, GgufError> {
        let elem_type_tag = self.read_u32()?;
        let elem_type = MetaValueType::from_u32(elem_type_tag)
            .ok_or(GgufError::UnknownMetaType(elem_type_tag))?;
        let n = self.read_u64()? as usize;
        Ok(match elem_type {
            MetaValueType::U8 => {
                let mut v = vec![0u8; n];
                self.need(n)?;
                v.copy_from_slice(&self.buf[self.pos..self.pos + n]);
                self.pos += n;
                MetaArray::U8(v)
            },
            MetaValueType::I8 => {
                let mut v = Vec::with_capacity(n);
                for _ in 0..n {
                    v.push(self.read_i8()?);
                }
                MetaArray::I8(v)
            },
            MetaValueType::U16 => {
                let mut v = Vec::with_capacity(n);
                for _ in 0..n {
                    v.push(self.read_u16()?);
                }
                MetaArray::U16(v)
            },
            MetaValueType::I16 => {
                let mut v = Vec::with_capacity(n);
                for _ in 0..n {
                    v.push(self.read_i16()?);
                }
                MetaArray::I16(v)
            },
            MetaValueType::U32 => {
                let mut v = Vec::with_capacity(n);
                for _ in 0..n {
                    v.push(self.read_u32()?);
                }
                MetaArray::U32(v)
            },
            MetaValueType::I32 => {
                let mut v = Vec::with_capacity(n);
                for _ in 0..n {
                    v.push(self.read_i32()?);
                }
                MetaArray::I32(v)
            },
            MetaValueType::F32 => {
                let mut v = Vec::with_capacity(n);
                for _ in 0..n {
                    v.push(self.read_f32()?);
                }
                MetaArray::F32(v)
            },
            MetaValueType::Bool => {
                let mut v = Vec::with_capacity(n);
                for _ in 0..n {
                    v.push(self.read_bool()?);
                }
                MetaArray::Bool(v)
            },
            MetaValueType::String => {
                let mut v = Vec::with_capacity(n);
                for _ in 0..n {
                    v.push(self.read_string("meta.array.string")?);
                }
                MetaArray::String(v)
            },
            MetaValueType::U64 => {
                let mut v = Vec::with_capacity(n);
                for _ in 0..n {
                    v.push(self.read_u64()?);
                }
                MetaArray::U64(v)
            },
            MetaValueType::I64 => {
                let mut v = Vec::with_capacity(n);
                for _ in 0..n {
                    v.push(self.read_i64()?);
                }
                MetaArray::I64(v)
            },
            MetaValueType::F64 => {
                let mut v = Vec::with_capacity(n);
                for _ in 0..n {
                    v.push(self.read_f64()?);
                }
                MetaArray::F64(v)
            },
            MetaValueType::Array => {
                // Nested arrays are technically allowed by the spec but never
                // produced in practice. We refuse them rather than recurse.
                return Err(GgufError::UnknownMetaType(9));
            },
        })
    }
}

/// A parsed GGUF file held in memory, with a pre-built tensor index and
/// strongly-typed metadata.
///
/// The `bytes` field is the raw file contents — keeping it around lets us
/// service [`GgufFile::tensor_bytes`] without re-reading from disk.
pub struct GgufFile {
    pub(crate) bytes: Vec<u8>,
    version: u32,
    metadata: Metadata,
    tensors: Vec<TensorInfo>,
    /// Absolute file offset where the (aligned) tensor data section starts.
    data_offset: u64,
    alignment: u64,
}

impl GgufFile {
    /// Read and parse a GGUF file from disk.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, GgufError> {
        let bytes = fs::read(path.as_ref())?;
        Self::from_bytes(bytes)
    }

    /// Parse from an owned byte buffer (use this with `mmap` slices).
    pub fn from_bytes(bytes: Vec<u8>) -> Result<Self, GgufError> {
        let mut cur = Cursor::new(&bytes);

        // ---- Header ----
        let magic = cur.read_u32()?;
        if magic != GGUF_MAGIC {
            return Err(GgufError::BadMagic(magic));
        }
        let version = cur.read_u32()?;
        if version != 2 && version != 3 {
            return Err(GgufError::UnsupportedVersion(version));
        }
        let tensor_count = cur.read_u64()?;
        let metadata_count = cur.read_u64()?;

        // ---- Metadata KV ----
        let mut metadata = Metadata::default();
        for _ in 0..metadata_count {
            let key = cur.read_string("meta.key")?;
            let vt_tag = cur.read_u32()?;
            let vt = MetaValueType::from_u32(vt_tag).ok_or(GgufError::UnknownMetaType(vt_tag))?;
            let val = cur.read_value(vt)?;
            metadata.kv.insert(key, val);
        }

        // ---- Tensor index ----
        let mut tensors: Vec<TensorInfo> = Vec::with_capacity(tensor_count as usize);
        for _ in 0..tensor_count {
            let name = cur.read_string("tensor.name")?;
            let n_dims = cur.read_u32()? as usize;
            let mut shape = Vec::with_capacity(n_dims);
            for _ in 0..n_dims {
                shape.push(cur.read_u64()?);
            }
            let dtype_tag = cur.read_u32()?;
            let dtype = GgmlType::from_u32(dtype_tag)?;
            let offset = cur.read_u64()?;
            tensors.push(TensorInfo {
                name,
                shape,
                dtype,
                offset,
            });
        }

        // ---- Compute aligned data offset ----
        let alignment = metadata
            .get_u64("general.alignment")
            .unwrap_or(GGUF_DEFAULT_ALIGNMENT);
        let header_end = cur.pos as u64;
        let data_offset = header_end.div_ceil(alignment) * alignment;

        // ---- Validate tensor offsets are within the file ----
        let data_len = bytes.len() as u64 - data_offset;
        for t in &tensors {
            if t.offset + t.byte_size() > data_len {
                return Err(GgufError::TensorOffsetOob {
                    name: t.name.clone(),
                    offset: t.offset,
                    data_len,
                });
            }
        }

        Ok(Self {
            bytes,
            version,
            metadata,
            tensors,
            data_offset,
            alignment,
        })
    }

    pub fn version(&self) -> u32 {
        self.version
    }

    pub fn alignment(&self) -> u64 {
        self.alignment
    }

    pub fn metadata(&self) -> &Metadata {
        &self.metadata
    }

    pub fn tensors(&self) -> &[TensorInfo] {
        &self.tensors
    }

    /// Find a tensor by exact name. O(N) — for repeated lookups, build
    /// your own `HashMap<&str, &TensorInfo>`.
    pub fn tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.iter().find(|t| t.name == name)
    }

    /// Absolute file offset where the (aligned) tensor data section starts.
    pub fn data_offset(&self) -> u64 {
        self.data_offset
    }

    /// Returns the raw block bytes for a tensor (no dequantization).
    /// Slice length = `tensor.byte_size()`.
    pub fn tensor_bytes(&self, t: &TensorInfo) -> &[u8] {
        let start = (self.data_offset + t.offset) as usize;
        let end = start + t.byte_size() as usize;
        &self.bytes[start..end]
    }

    /// Total file size, as a sanity-check helper.
    pub fn file_size(&self) -> u64 {
        self.bytes.len() as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds a tiny in-memory GGUF file with one F32 tensor, then
    /// round-trips it through the parser.
    fn build_minimal_gguf_with_one_tensor() -> Vec<u8> {
        let mut buf: Vec<u8> = Vec::new();

        // Magic + version + counts.
        buf.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        buf.extend_from_slice(&3u32.to_le_bytes()); // version
        buf.extend_from_slice(&1u64.to_le_bytes()); // tensor_count
        buf.extend_from_slice(&2u64.to_le_bytes()); // metadata_count

        // KV[0]: ("general.architecture": String("toy")).
        let key = b"general.architecture";
        buf.extend_from_slice(&(key.len() as u64).to_le_bytes());
        buf.extend_from_slice(key);
        buf.extend_from_slice(&8u32.to_le_bytes()); // type=String
        let val = b"toy";
        buf.extend_from_slice(&(val.len() as u64).to_le_bytes());
        buf.extend_from_slice(val);

        // KV[1]: ("toy.block_count": U32(7)).
        let key = b"toy.block_count";
        buf.extend_from_slice(&(key.len() as u64).to_le_bytes());
        buf.extend_from_slice(key);
        buf.extend_from_slice(&4u32.to_le_bytes()); // type=U32
        buf.extend_from_slice(&7u32.to_le_bytes());

        // Tensor[0]: name="t.0", shape=[4], dtype=F32, offset=0.
        let tname = b"t.0";
        buf.extend_from_slice(&(tname.len() as u64).to_le_bytes());
        buf.extend_from_slice(tname);
        buf.extend_from_slice(&1u32.to_le_bytes()); // n_dims
        buf.extend_from_slice(&4u64.to_le_bytes()); // dim 0
        buf.extend_from_slice(&0u32.to_le_bytes()); // dtype = F32
        buf.extend_from_slice(&0u64.to_le_bytes()); // offset

        // Pad to alignment (32 by default).
        while buf.len() % 32 != 0 {
            buf.push(0);
        }
        // Tensor data: [1.0, 2.0, 3.0, 4.0].
        for x in [1.0_f32, 2.0, 3.0, 4.0] {
            buf.extend_from_slice(&x.to_le_bytes());
        }

        buf
    }

    #[test]
    fn parse_minimal_roundtrip() {
        let buf = build_minimal_gguf_with_one_tensor();
        let f = GgufFile::from_bytes(buf).expect("parse ok");
        assert_eq!(f.version(), 3);
        assert_eq!(f.metadata().architecture(), Some("toy"));
        assert_eq!(
            f.metadata()
                .get_arch("block_count")
                .and_then(|v| v.as_u32()),
            Some(7)
        );
        assert_eq!(f.tensors().len(), 1);
        let t = &f.tensors()[0];
        assert_eq!(t.name, "t.0");
        assert_eq!(t.shape, vec![4]);
        assert_eq!(t.dtype, GgmlType::F32);
        assert_eq!(t.byte_size(), 16);

        // Verify the raw bytes round-trip through `tensor_bytes`.
        let raw = f.tensor_bytes(t);
        let mut got = [0f32; 4];
        for (i, c) in raw.chunks_exact(4).enumerate() {
            got[i] = f32::from_le_bytes(c.try_into().unwrap());
        }
        assert_eq!(got, [1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn rejects_bad_magic() {
        let mut buf = build_minimal_gguf_with_one_tensor();
        buf[0] = 0xff;
        assert!(matches!(
            GgufFile::from_bytes(buf),
            Err(GgufError::BadMagic(_))
        ));
    }

    #[test]
    fn rejects_unknown_version() {
        let mut buf = build_minimal_gguf_with_one_tensor();
        buf[4..8].copy_from_slice(&99u32.to_le_bytes());
        assert!(matches!(
            GgufFile::from_bytes(buf),
            Err(GgufError::UnsupportedVersion(99))
        ));
    }

    #[test]
    fn detects_truncation() {
        let buf = build_minimal_gguf_with_one_tensor();
        let truncated = buf[..16].to_vec();
        assert!(matches!(
            GgufFile::from_bytes(truncated),
            Err(GgufError::Truncated { .. })
        ));
    }
}
