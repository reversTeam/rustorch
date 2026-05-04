//! # rustorch-gguf
//!
//! Pure-Rust parser for the **GGUF** (GGML Universal File) format used by
//! `llama.cpp`, plus dequantization kernels for the K-quant family
//! (Q4_K, Q6_K, Q8_0). Designed to load real Qwen / Llama checkpoints
//! produced by quantization pipelines such as Q4_K_M / Q4_K_XL.
//!
//! ## Layers
//!
//! 1. [`reader::GgufFile`] — memory-mapped (read-only) view over a GGUF file
//!    that parses the header, metadata and tensor index without copying
//!    tensor data.
//! 2. [`metadata::MetaValue`] — strongly-typed metadata value enum with
//!    convenient accessors (`as_u32`, `as_str`, `as_array_u32`...).
//! 3. [`tensor::TensorInfo`] — per-tensor descriptor (name, shape, dtype,
//!    file offset).
//! 4. [`dequant`] — block-wise dequantization (Q4_K, Q6_K, Q8_0, F16,
//!    BF16, F32).
//!
//! ## Quick start
//!
//! ```no_run
//! use rustorch_gguf::GgufFile;
//!
//! let f = GgufFile::open("Qwen3.5-9B.Q4_K_M.gguf").unwrap();
//! println!("version  = {}", f.version());
//! println!("arch     = {}", f.metadata().get("general.architecture")
//!                              .and_then(|v| v.as_str()).unwrap_or("?"));
//! for t in f.tensors() {
//!     println!("{:>40}  shape={:?}  dtype={:?}", t.name, t.shape, t.dtype);
//! }
//! ```

#![allow(clippy::needless_range_loop)]

pub mod dequant;
pub mod metadata;
pub mod reader;
/// Direct sgemv on Q4_K-quantised weights — bypasses the f32
/// dequantisation cache miss for the LLM decode hot path.
pub mod sgemv_q4k;
pub mod tensor;

pub use dequant::{dequant_to_f32, dequantize_block_chunk, num_elements, DequantError};
pub use metadata::{MetaArray, MetaValue, MetaValueType};
pub use reader::{GgufError, GgufFile};
pub use sgemv_q4k::sgemv_q4_k;
pub use tensor::{GgmlType, TensorInfo};

/// GGUF magic bytes (`b"GGUF"` little-endian as a u32 = `0x46554747`).
pub const GGUF_MAGIC: u32 = 0x46554747;

/// Default tensor data alignment used when the file does not specify
/// `general.alignment` in its metadata.
pub const GGUF_DEFAULT_ALIGNMENT: u64 = 32;
