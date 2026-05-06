//! `gguf_inspect` — dump the header, metadata and tensor index of a GGUF file.
//!
//! Usage:
//!   cargo run --release --example gguf_inspect -- ~/models/Qwen3.5-9B.Q4_K_M.gguf
//!
//! Optional flags:
//!   --dtype-stats        : count of tensors per dtype
//!   --grep <pattern>     : only print tensors whose name contains the pattern
//!   --no-tensors         : skip the tensor listing (header + metadata only)

use std::collections::BTreeMap;
use std::env;
use std::process::ExitCode;

use rustorch_gguf::{GgufFile, MetaArray, MetaValue};

fn humanize_bytes(b: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
    let mut v = b as f64;
    let mut u = 0;
    while v >= 1024.0 && u < UNITS.len() - 1 {
        v /= 1024.0;
        u += 1;
    }
    format!("{:.2} {}", v, UNITS[u])
}

fn fmt_meta(v: &MetaValue) -> String {
    match v {
        MetaValue::String(s) => {
            if s.len() > 80 {
                format!("\"{}…\" ({} chars)", &s[..80], s.len())
            } else {
                format!("\"{}\"", s)
            }
        },
        MetaValue::Array(a) => match a {
            MetaArray::U8(v) => format!("[u8;{}]", v.len()),
            MetaArray::I8(v) => format!("[i8;{}]", v.len()),
            MetaArray::U16(v) => format!("[u16;{}]", v.len()),
            MetaArray::I16(v) => format!("[i16;{}]", v.len()),
            MetaArray::U32(v) => {
                if v.len() <= 8 {
                    format!("{:?}", v)
                } else {
                    format!("[u32;{}]  first8={:?}", v.len(), &v[..8])
                }
            },
            MetaArray::I32(v) => {
                if v.len() <= 8 {
                    format!("{:?}", v)
                } else {
                    format!("[i32;{}]  first8={:?}", v.len(), &v[..8])
                }
            },
            MetaArray::F32(v) => {
                if v.len() <= 8 {
                    format!("{:?}", v)
                } else {
                    format!("[f32;{}]  first8={:?}", v.len(), &v[..8])
                }
            },
            MetaArray::Bool(v) => format!("[bool;{}]", v.len()),
            MetaArray::String(v) => format!("[str;{}]", v.len()),
            MetaArray::U64(v) => format!("[u64;{}]", v.len()),
            MetaArray::I64(v) => format!("[i64;{}]", v.len()),
            MetaArray::F64(v) => format!("[f64;{}]", v.len()),
        },
        MetaValue::U8(v) => v.to_string(),
        MetaValue::I8(v) => v.to_string(),
        MetaValue::U16(v) => v.to_string(),
        MetaValue::I16(v) => v.to_string(),
        MetaValue::U32(v) => v.to_string(),
        MetaValue::I32(v) => v.to_string(),
        MetaValue::U64(v) => v.to_string(),
        MetaValue::I64(v) => v.to_string(),
        MetaValue::F32(v) => format!("{:.6}", v),
        MetaValue::F64(v) => format!("{:.6}", v),
        MetaValue::Bool(b) => b.to_string(),
    }
}

fn main() -> ExitCode {
    let mut args = env::args().skip(1).collect::<Vec<_>>();

    let mut dtype_stats = false;
    let mut no_tensors = false;
    let mut grep: Option<String> = None;
    args.retain(|a| {
        if a == "--dtype-stats" {
            dtype_stats = true;
            return false;
        }
        if a == "--no-tensors" {
            no_tensors = true;
            return false;
        }
        if let Some(rest) = a.strip_prefix("--grep=") {
            grep = Some(rest.to_string());
            return false;
        }
        true
    });

    if args.is_empty() {
        eprintln!("usage: gguf_inspect <file.gguf> [--dtype-stats] [--no-tensors] [--grep=<pat>]");
        return ExitCode::from(2);
    }
    let path = &args[0];

    println!("→ opening {}", path);
    let f = match GgufFile::open(path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::from(1);
        },
    };

    println!("\n══════ HEADER ══════");
    println!("file_size       : {}", humanize_bytes(f.file_size()));
    println!("version         : {}", f.version());
    println!("alignment       : {}", f.alignment());
    println!("data_offset     : 0x{:x}", f.data_offset());
    println!("n_tensors       : {}", f.tensors().len());
    println!("n_metadata_kv   : {}", f.metadata().kv.len());

    println!("\n══════ METADATA ══════");
    for (k, v) in &f.metadata().kv {
        println!("  {:<48}  {}", k, fmt_meta(v));
    }

    if dtype_stats {
        println!("\n══════ DTYPE STATS ══════");
        let mut by_dtype: BTreeMap<String, (usize, u64)> = BTreeMap::new();
        for t in f.tensors() {
            let key = format!("{:?}", t.dtype);
            let entry = by_dtype.entry(key).or_insert((0, 0));
            entry.0 += 1;
            entry.1 += t.byte_size();
        }
        let mut total_bytes = 0u64;
        for (k, (n, bytes)) in &by_dtype {
            println!(
                "  {:<10}  {:>5} tensors  {:>12}",
                k,
                n,
                humanize_bytes(*bytes)
            );
            total_bytes += bytes;
        }
        println!(
            "  {:<10}  {:>5}          {:>12}",
            "TOTAL",
            f.tensors().len(),
            humanize_bytes(total_bytes)
        );
    }

    if !no_tensors {
        println!("\n══════ TENSORS ══════");
        let pat = grep.as_deref();
        let mut shown = 0usize;
        for t in f.tensors() {
            if let Some(p) = pat {
                if !t.name.contains(p) {
                    continue;
                }
            }
            println!(
                "  {:<48}  shape={:<24}  dtype={:?}  bytes={}",
                t.name,
                format!("{:?}", t.shape),
                t.dtype,
                humanize_bytes(t.byte_size())
            );
            shown += 1;
        }
        if pat.is_some() {
            println!("\n  → {} tensor(s) matched pattern", shown);
        }
    }

    ExitCode::SUCCESS
}
