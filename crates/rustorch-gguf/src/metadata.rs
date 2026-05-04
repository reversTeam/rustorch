//! GGUF metadata KV value enum.
//!
//! GGUF metadata is a list of `(key, value)` pairs where the value is one
//! of 13 typed scalars or a homogeneous typed array. We model this with
//! an `enum MetaValue` and provide ergonomic accessors so callers can
//! write
//!
//! ```ignore
//! let n_layers = meta.get_u32("qwen35.block_count").unwrap();
//! ```
//!
//! without manual matching.

use std::collections::BTreeMap;

/// Tag values for GGUF metadata typed entries (matches `gguf_metadata_value_type`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum MetaValueType {
    U8 = 0,
    I8 = 1,
    U16 = 2,
    I16 = 3,
    U32 = 4,
    I32 = 5,
    F32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    U64 = 10,
    I64 = 11,
    F64 = 12,
}

impl MetaValueType {
    pub fn from_u32(v: u32) -> Option<Self> {
        Some(match v {
            0 => Self::U8,
            1 => Self::I8,
            2 => Self::U16,
            3 => Self::I16,
            4 => Self::U32,
            5 => Self::I32,
            6 => Self::F32,
            7 => Self::Bool,
            8 => Self::String,
            9 => Self::Array,
            10 => Self::U64,
            11 => Self::I64,
            12 => Self::F64,
            _ => return None,
        })
    }
}

/// A typed homogeneous array of values (variant of [`MetaValue::Array`]).
///
/// We keep the per-element typing — except for `String` arrays — as raw
/// numeric vectors to avoid the overhead of `Vec<MetaValue>` wrapping.
#[derive(Debug, Clone, PartialEq)]
pub enum MetaArray {
    U8(Vec<u8>),
    I8(Vec<i8>),
    U16(Vec<u16>),
    I16(Vec<i16>),
    U32(Vec<u32>),
    I32(Vec<i32>),
    F32(Vec<f32>),
    Bool(Vec<bool>),
    String(Vec<String>),
    U64(Vec<u64>),
    I64(Vec<i64>),
    F64(Vec<f64>),
}

impl MetaArray {
    pub fn len(&self) -> usize {
        match self {
            Self::U8(v) => v.len(),
            Self::I8(v) => v.len(),
            Self::U16(v) => v.len(),
            Self::I16(v) => v.len(),
            Self::U32(v) => v.len(),
            Self::I32(v) => v.len(),
            Self::F32(v) => v.len(),
            Self::Bool(v) => v.len(),
            Self::String(v) => v.len(),
            Self::U64(v) => v.len(),
            Self::I64(v) => v.len(),
            Self::F64(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// A typed metadata value.
#[derive(Debug, Clone, PartialEq)]
pub enum MetaValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    String(String),
    Array(MetaArray),
    U64(u64),
    I64(i64),
    F64(f64),
}

impl MetaValue {
    /// Best-effort cast to `u32` for any integer type.
    pub fn as_u32(&self) -> Option<u32> {
        Some(match self {
            Self::U8(v) => *v as u32,
            Self::U16(v) => *v as u32,
            Self::U32(v) => *v,
            Self::U64(v) => *v as u32,
            Self::I8(v) => *v as u32,
            Self::I16(v) => *v as u32,
            Self::I32(v) => *v as u32,
            Self::I64(v) => *v as u32,
            _ => return None,
        })
    }

    pub fn as_u64(&self) -> Option<u64> {
        Some(match self {
            Self::U8(v) => *v as u64,
            Self::U16(v) => *v as u64,
            Self::U32(v) => *v as u64,
            Self::U64(v) => *v,
            Self::I8(v) => *v as u64,
            Self::I16(v) => *v as u64,
            Self::I32(v) => *v as u64,
            Self::I64(v) => *v as u64,
            _ => return None,
        })
    }

    pub fn as_f32(&self) -> Option<f32> {
        Some(match self {
            Self::F32(v) => *v,
            Self::F64(v) => *v as f32,
            _ => return None,
        })
    }

    pub fn as_bool(&self) -> Option<bool> {
        if let Self::Bool(b) = self {
            Some(*b)
        } else {
            None
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        if let Self::String(s) = self {
            Some(s)
        } else {
            None
        }
    }

    pub fn as_array(&self) -> Option<&MetaArray> {
        if let Self::Array(a) = self {
            Some(a)
        } else {
            None
        }
    }
}

/// Parsed metadata KV map.
#[derive(Debug, Default, Clone)]
pub struct Metadata {
    pub kv: BTreeMap<String, MetaValue>,
}

impl Metadata {
    pub fn get(&self, key: &str) -> Option<&MetaValue> {
        self.kv.get(key)
    }

    pub fn get_u32(&self, key: &str) -> Option<u32> {
        self.get(key)?.as_u32()
    }

    pub fn get_u64(&self, key: &str) -> Option<u64> {
        self.get(key)?.as_u64()
    }

    pub fn get_f32(&self, key: &str) -> Option<f32> {
        self.get(key)?.as_f32()
    }

    pub fn get_str(&self, key: &str) -> Option<&str> {
        self.get(key)?.as_str()
    }

    pub fn get_array(&self, key: &str) -> Option<&MetaArray> {
        self.get(key)?.as_array()
    }

    /// Returns the architecture string (`general.architecture`).
    pub fn architecture(&self) -> Option<&str> {
        self.get_str("general.architecture")
    }

    /// Returns the architecture name with `.` prefix appended (e.g. `"qwen35"` → `"qwen35."`)
    /// for use in keying per-architecture config entries (`{arch}.block_count`).
    pub fn arch_prefix(&self) -> Option<String> {
        self.architecture().map(|a| format!("{}.", a))
    }

    /// Convenience helper: `meta.get_arch("block_count")` looks up
    /// `"<arch>.block_count"`.
    pub fn get_arch(&self, suffix: &str) -> Option<&MetaValue> {
        let p = self.arch_prefix()?;
        let key = format!("{}{}", p, suffix);
        self.get(&key)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn meta_value_type_from_u32() {
        assert_eq!(MetaValueType::from_u32(4), Some(MetaValueType::U32));
        assert_eq!(MetaValueType::from_u32(8), Some(MetaValueType::String));
        assert_eq!(MetaValueType::from_u32(9), Some(MetaValueType::Array));
        assert_eq!(MetaValueType::from_u32(99), None);
    }

    #[test]
    fn arch_prefix_helper() {
        let mut m = Metadata::default();
        m.kv.insert(
            "general.architecture".into(),
            MetaValue::String("qwen35".into()),
        );
        m.kv.insert("qwen35.block_count".into(), MetaValue::U32(48));

        assert_eq!(m.architecture(), Some("qwen35"));
        assert_eq!(m.arch_prefix(), Some("qwen35.".to_string()));
        assert_eq!(m.get_arch("block_count").and_then(|v| v.as_u32()), Some(48));
    }

    #[test]
    fn as_u32_widening() {
        assert_eq!(MetaValue::U8(7).as_u32(), Some(7));
        assert_eq!(MetaValue::U64(99).as_u32(), Some(99));
        // Strings don't widen.
        assert!(MetaValue::String("hi".into()).as_u32().is_none());
    }
}
