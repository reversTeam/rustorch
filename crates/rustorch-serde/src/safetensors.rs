//! safetensors reader + writer (P1.8).
//!
//! File format (HuggingFace safetensors):
//! ```text
//!   [0..8]                : u64 little-endian header_size N
//!   [8..8+N]              : JSON header { name: { dtype, shape, data_offsets: [start, end] }, ... }
//!   [8+N..]               : binary blob — each tensor's bytes packed contiguously
//! ```
//!
//! Both reader and writer are **pure Rust**. No mmap (yet); the reader
//! loads bytes into owned tensors. mmap support is a follow-up.
//!
//! Supported dtypes: F32, F64, F16, BF16, I64, I32, I8, Bool — the full
//! set rustorch's `Dtype` enum exposes. Each dtype maps to a string per
//! the safetensors spec ("F32", "F64", "F16", "BF16", "I64", "I32",
//! "I8", "BOOL").

use rustorch_core::tensor::dtype::Dtype;
use rustorch_core::tensor::tensor_impl::Tensor;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::io::{Read, Write};
use std::path::Path;

/// Errors during safetensors I/O.
#[derive(Debug, thiserror::Error)]
pub enum SafetensorsError {
    /// Underlying I/O failed.
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    /// JSON header parse / serialise error.
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
    /// Header references a dtype the runtime doesn't support.
    #[error("unsupported dtype: {0}")]
    UnsupportedDtype(String),
    /// File header is malformed.
    #[error("invalid header: {0}")]
    InvalidHeader(String),
    /// On read, a tensor's stated byte length doesn't match `numel * dtype.byte_size()`.
    #[error("size mismatch for tensor {name}: header says {expected} bytes, got {got}")]
    SizeMismatch {
        /// Tensor name.
        name: String,
        /// Bytes claimed by header.
        expected: usize,
        /// Actual data bytes.
        got: usize,
    },
}

// --- dtype ↔ string ---

fn dtype_to_str(d: Dtype) -> &'static str {
    match d {
        Dtype::F32 => "F32",
        Dtype::F64 => "F64",
        Dtype::F16 => "F16",
        Dtype::BF16 => "BF16",
        Dtype::I64 => "I64",
        Dtype::I32 => "I32",
        Dtype::I8 => "I8",
        Dtype::Bool => "BOOL",
    }
}

fn dtype_from_str(s: &str) -> Result<Dtype, SafetensorsError> {
    match s {
        "F32" => Ok(Dtype::F32),
        "F64" => Ok(Dtype::F64),
        "F16" => Ok(Dtype::F16),
        "BF16" => Ok(Dtype::BF16),
        "I64" => Ok(Dtype::I64),
        "I32" => Ok(Dtype::I32),
        "I8" => Ok(Dtype::I8),
        "BOOL" => Ok(Dtype::Bool),
        other => Err(SafetensorsError::UnsupportedDtype(other.to_string())),
    }
}

// --- header schema ---

#[derive(Debug, Serialize, Deserialize)]
struct HeaderEntry {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [usize; 2],
}

// ------------------------------ writer ------------------------------

/// Serialise a name→tensor map to a safetensors file at `path`.
pub fn write_path<P: AsRef<Path>>(
    path: P,
    tensors: &BTreeMap<String, Tensor>,
) -> Result<(), SafetensorsError> {
    let mut f = std::fs::File::create(path)?;
    write_to(&mut f, tensors)
}

/// Serialise a name→tensor map to any [`Write`] sink.
pub fn write_to<W: Write>(
    sink: &mut W,
    tensors: &BTreeMap<String, Tensor>,
) -> Result<(), SafetensorsError> {
    // Compute layout: walk tensors in BTreeMap order, build header
    // entries with data_offsets [start, end] referencing the binary
    // block (offsets are relative to the start of the binary section).
    let mut header: BTreeMap<String, HeaderEntry> = BTreeMap::new();
    let mut cursor = 0_usize;
    for (name, t) in tensors {
        let nbytes = t.numel() * t.dtype().byte_size();
        header.insert(
            name.clone(),
            HeaderEntry {
                dtype: dtype_to_str(t.dtype()).to_string(),
                shape: t.shape().to_vec(),
                data_offsets: [cursor, cursor + nbytes],
            },
        );
        cursor += nbytes;
    }

    // Encode header as JSON.
    let header_json = serde_json::to_string(&header)?;
    let header_bytes = header_json.into_bytes();
    let header_len = header_bytes.len() as u64;

    // Write [u64 LE header_size][header bytes][binary block].
    sink.write_all(&header_len.to_le_bytes())?;
    sink.write_all(&header_bytes)?;
    for t in tensors.values() {
        let bytes = t.storage().as_bytes();
        let nbytes = t.numel() * t.dtype().byte_size();
        // Defensive: if storage has more bytes than numel*dtype (e.g.
        // shared sub-view), only emit the first `nbytes`.
        sink.write_all(&bytes[..nbytes])?;
    }
    Ok(())
}

// ------------------------------ reader ------------------------------

/// Load a safetensors file from `path` into a name→tensor map.
pub fn read_path<P: AsRef<Path>>(path: P) -> Result<BTreeMap<String, Tensor>, SafetensorsError> {
    let mut f = std::fs::File::open(path)?;
    read_from(&mut f)
}

/// Load a safetensors file from any [`Read`] source.
pub fn read_from<R: Read>(src: &mut R) -> Result<BTreeMap<String, Tensor>, SafetensorsError> {
    // Read header_size.
    let mut size_buf = [0_u8; 8];
    src.read_exact(&mut size_buf)?;
    let header_size = u64::from_le_bytes(size_buf) as usize;
    if header_size > 100_000_000 {
        return Err(SafetensorsError::InvalidHeader(format!(
            "header_size {header_size} exceeds 100MB sanity limit"
        )));
    }
    // Read header JSON.
    let mut header_bytes = vec![0_u8; header_size];
    src.read_exact(&mut header_bytes)?;
    let header: BTreeMap<String, HeaderEntry> = serde_json::from_slice(&header_bytes)?;

    // Read binary block.
    let mut binary = Vec::new();
    src.read_to_end(&mut binary)?;

    // Slice the block per entry.
    let mut out: BTreeMap<String, Tensor> = BTreeMap::new();
    for (name, entry) in header {
        let dtype = dtype_from_str(&entry.dtype)?;
        let [start, end] = entry.data_offsets;
        if end > binary.len() || start > end {
            return Err(SafetensorsError::InvalidHeader(format!(
                "tensor {name}: offsets [{start}, {end}] outside binary block of len {}",
                binary.len()
            )));
        }
        let slice = &binary[start..end];
        let numel: usize = entry.shape.iter().product();
        let expected_bytes = numel * dtype.byte_size();
        if slice.len() != expected_bytes {
            return Err(SafetensorsError::SizeMismatch {
                name,
                expected: expected_bytes,
                got: slice.len(),
            });
        }
        // Build a Tensor from bytes by typed dispatch.
        let t = build_tensor_from_bytes(&entry.shape, dtype, slice)?;
        out.insert(name, t);
    }
    Ok(out)
}

fn build_tensor_from_bytes(
    shape: &[usize],
    dtype: Dtype,
    bytes: &[u8],
) -> Result<Tensor, SafetensorsError> {
    let shape_v = shape.to_vec();
    match dtype {
        Dtype::F32 => {
            let v: Vec<f32> = bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            Tensor::from_vec(shape_v, v)
                .map_err(|e| SafetensorsError::InvalidHeader(format!("tensor build f32: {e}")))
        },
        Dtype::F64 => {
            let v: Vec<f64> = bytes
                .chunks_exact(8)
                .map(|c| f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
                .collect();
            Tensor::from_vec_typed::<f64, _>(shape_v, v)
                .map_err(|e| SafetensorsError::InvalidHeader(format!("tensor build f64: {e}")))
        },
        Dtype::I64 => {
            let v: Vec<i64> = bytes
                .chunks_exact(8)
                .map(|c| i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
                .collect();
            Tensor::from_vec_typed::<i64, _>(shape_v, v)
                .map_err(|e| SafetensorsError::InvalidHeader(format!("tensor build i64: {e}")))
        },
        Dtype::I32 => {
            let v: Vec<i32> = bytes
                .chunks_exact(4)
                .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            Tensor::from_vec_typed::<i32, _>(shape_v, v)
                .map_err(|e| SafetensorsError::InvalidHeader(format!("tensor build i32: {e}")))
        },
        Dtype::I8 => {
            let v: Vec<i8> = bytes.iter().map(|&b| b as i8).collect();
            Tensor::from_vec_typed::<i8, _>(shape_v, v)
                .map_err(|e| SafetensorsError::InvalidHeader(format!("tensor build i8: {e}")))
        },
        Dtype::Bool => {
            let v: Vec<bool> = bytes.iter().map(|&b| b != 0).collect();
            Tensor::from_vec_typed::<bool, _>(shape_v, v)
                .map_err(|e| SafetensorsError::InvalidHeader(format!("tensor build bool: {e}")))
        },
        Dtype::F16 | Dtype::BF16 => {
            // Half-precision builds need explicit half::f16 / bf16
            // construction; defer to a follow-up slice.
            Err(SafetensorsError::UnsupportedDtype(
                dtype_to_str(dtype).to_string(),
            ))
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t_f32(shape: &[usize], data: Vec<f32>) -> Tensor {
        Tensor::from_vec(shape.to_vec(), data).unwrap()
    }

    fn t_i64(shape: &[usize], data: Vec<i64>) -> Tensor {
        Tensor::from_vec_typed::<i64, _>(shape.to_vec(), data).unwrap()
    }

    #[test]
    fn round_trip_single_f32_tensor() {
        let mut map: BTreeMap<String, Tensor> = BTreeMap::new();
        map.insert(
            "weight".to_string(),
            t_f32(&[2, 3], vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]),
        );

        let mut buf = Vec::new();
        write_to(&mut buf, &map).unwrap();

        let mut reader = std::io::Cursor::new(buf);
        let out = read_from(&mut reader).unwrap();

        assert_eq!(out.len(), 1);
        let got = &out["weight"];
        assert_eq!(got.shape(), &[2, 3]);
        assert_eq!(got.dtype(), Dtype::F32);
        assert_eq!(
            got.as_slice::<f32>().unwrap(),
            &[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]
        );
    }

    #[test]
    fn round_trip_mixed_dtypes() {
        let mut map: BTreeMap<String, Tensor> = BTreeMap::new();
        map.insert(
            "fc.weight".to_string(),
            t_f32(&[3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        );
        map.insert("fc.bias".to_string(), t_f32(&[2], vec![0.5, -0.5]));
        map.insert("targets".to_string(), t_i64(&[4], vec![0, 1, 2, 3]));

        let mut buf = Vec::new();
        write_to(&mut buf, &map).unwrap();
        let mut reader = std::io::Cursor::new(buf);
        let out = read_from(&mut reader).unwrap();

        assert_eq!(out.len(), 3);
        assert_eq!(out["fc.weight"].shape(), &[3, 2]);
        assert_eq!(out["fc.bias"].shape(), &[2]);
        assert_eq!(out["targets"].shape(), &[4]);
        assert_eq!(out["targets"].as_slice::<i64>().unwrap(), &[0_i64, 1, 2, 3]);
    }

    #[test]
    fn empty_map_round_trips() {
        let map: BTreeMap<String, Tensor> = BTreeMap::new();
        let mut buf = Vec::new();
        write_to(&mut buf, &map).unwrap();
        let mut reader = std::io::Cursor::new(buf);
        let out = read_from(&mut reader).unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn round_trip_preserves_nan() {
        let mut map: BTreeMap<String, Tensor> = BTreeMap::new();
        map.insert(
            "with_nan".to_string(),
            t_f32(&[3], vec![1.0_f32, f32::NAN, 3.0]),
        );
        let mut buf = Vec::new();
        write_to(&mut buf, &map).unwrap();
        let mut reader = std::io::Cursor::new(buf);
        let out = read_from(&mut reader).unwrap();
        let v = out["with_nan"].as_slice::<f32>().unwrap();
        assert_eq!(v[0], 1.0);
        assert!(v[1].is_nan());
        assert_eq!(v[2], 3.0);
    }

    #[test]
    fn invalid_header_size_rejected() {
        // u64 LE header_size exceeding sanity limit.
        let bytes: Vec<u8> = (200_000_000_u64).to_le_bytes().to_vec();
        let mut reader = std::io::Cursor::new(bytes);
        assert!(matches!(
            read_from(&mut reader),
            Err(SafetensorsError::InvalidHeader(_))
        ));
    }

    #[test]
    fn header_offsets_consistent_with_block() {
        let mut map: BTreeMap<String, Tensor> = BTreeMap::new();
        map.insert(
            "a".to_string(),
            t_f32(&[2], vec![1.0_f32, 2.0]), // 8 bytes
        );
        map.insert(
            "b".to_string(),
            t_f32(&[3], vec![3.0_f32, 4.0, 5.0]), // 12 bytes
        );
        let mut buf = Vec::new();
        write_to(&mut buf, &map).unwrap();
        let mut reader = std::io::Cursor::new(buf);
        let out = read_from(&mut reader).unwrap();
        assert_eq!(out["a"].as_slice::<f32>().unwrap(), &[1.0_f32, 2.0]);
        assert_eq!(out["b"].as_slice::<f32>().unwrap(), &[3.0_f32, 4.0, 5.0]);
    }
}
