//! `qwen35_inspect` — load a Qwen3.5 / Qwen3.6 GGUF (dense or MoE
//! hybrid) and print the parsed architecture summary plus the per-layer
//! tensor inventory. This is the bring-up smoke test for the hybrid arch
//! support — it ensures we correctly identify the layer types, parse all
//! SSM hyperparameters, and resolve every tensor the future forward pass
//! will need.
//!
//! Usage:
//!
//! ```sh
//! cargo run --release -p rustorch-llm --example qwen35_inspect -- \
//!     ~/models/Qwen3.6-27B-Q4_K_M.gguf
//! ```

use std::env;
use std::path::PathBuf;
use std::process::ExitCode;

use rustorch_llm::qwen35::{
    describe_model, expected_tensor_names, full_inventory, missing_tensors, parse_config, LayerKind,
};

fn main() -> ExitCode {
    let path = match env::args().nth(1) {
        Some(p) => PathBuf::from(p),
        None => {
            eprintln!("usage: qwen35_inspect <gguf-path>");
            return ExitCode::FAILURE;
        },
    };

    println!("→ parsing config from {}", path.display());
    let cfg = match parse_config(&path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("parse failed: {e}");
            return ExitCode::FAILURE;
        },
    };
    println!("\n{}", describe_model(&cfg));

    // Per-layer tensor count breakdown.
    let attn_kinds_first_8: Vec<usize> = cfg.attention_indices.iter().take(8).copied().collect();
    let ssm_kinds_first_8: Vec<usize> = cfg.ssm_indices.iter().take(8).copied().collect();
    println!(
        "Attention layers (first 8 of {}): {:?}",
        cfg.attention_indices.len(),
        attn_kinds_first_8
    );
    println!(
        "SSM layers       (first 8 of {}): {:?}",
        cfg.ssm_indices.len(),
        ssm_kinds_first_8
    );

    // Show what tensors each layer kind expects.
    if let Some(&attn_li) = cfg.attention_indices.first() {
        println!("\nExpected tensors for attention layer {}:", attn_li);
        for n in expected_tensor_names(attn_li, LayerKind::Attention, cfg.variant) {
            println!("  {}", n);
        }
    }
    if let Some(&ssm_li) = cfg.ssm_indices.first() {
        println!("\nExpected tensors for SSM layer {}:", ssm_li);
        for n in expected_tensor_names(ssm_li, LayerKind::Ssm, cfg.variant) {
            println!("  {}", n);
        }
    }

    // Resolve every expected tensor against the actual GGUF.
    println!("\n→ resolving expected tensors against the GGUF...");
    let (found, missing) = match full_inventory(&path, &cfg) {
        Ok(x) => x,
        Err(e) => {
            eprintln!("inventory failed: {e}");
            return ExitCode::FAILURE;
        },
    };
    println!("  found   : {} tensors", found.len());
    println!("  missing : {} tensors", missing.len());
    if !missing.is_empty() {
        for m in missing.iter().take(20) {
            println!("    {m}");
        }
        if missing.len() > 20 {
            println!("    ... and {} more", missing.len() - 20);
        }
    }

    // Verify with the lightweight `missing_tensors` helper too.
    let missing2 = match missing_tensors(&path, &cfg) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("missing_tensors failed: {e}");
            return ExitCode::FAILURE;
        },
    };
    if missing2.len() != missing.len() {
        eprintln!(
            "missing_tensors disagreement: full_inventory={}, missing_tensors={}",
            missing.len(),
            missing2.len()
        );
    }

    // Bytes per layer estimate
    println!("\n=== Memory budget ===");
    let ssm_state = cfg.ssm_state_bytes_per_layer();
    let conv_state = cfg.ssm_conv_state_bytes_per_layer();
    println!(
        "  per SSM layer: state={:>7.1} KB  conv={:>5.1} KB",
        ssm_state as f64 / 1024.0,
        conv_state as f64 / 1024.0
    );
    let total_ssm = cfg.ssm_indices.len() * (ssm_state + conv_state);
    println!(
        "  SSM total state across {} SSM layers: {:.1} MB",
        cfg.ssm_indices.len(),
        total_ssm as f64 / 1024.0 / 1024.0
    );

    // Print tensor-type distribution
    use std::collections::BTreeMap;
    let mut by_dtype: BTreeMap<String, (usize, u64)> = BTreeMap::new();
    for tr in &found {
        let key = format!("{:?}", tr.dtype);
        let bytes = tr.shape.iter().product::<u64>();
        let entry = by_dtype.entry(key).or_default();
        entry.0 += 1;
        entry.1 += bytes;
    }
    println!("\n=== Tensor dtype distribution (found tensors) ===");
    for (dt, (count, n_elem)) in by_dtype {
        println!(
            "  {:<10} count={:>4}   total elements={}",
            dt, count, n_elem
        );
    }

    println!("\n✓ inspect OK");
    ExitCode::SUCCESS
}
