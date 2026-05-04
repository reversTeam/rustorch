//! Quick check: dump the first few q_norm/k_norm gamma values from
//! a GGUF file to see if they are non-trivial.

use std::env;

use rustorch_gguf::{dequant_to_f32, GgufFile};

fn main() {
    let path = env::args().nth(1).expect("usage: inspect_qnorm <gguf>");
    let f = GgufFile::open(&path).unwrap();
    for layer in [0usize, 1, 2, 3, 39] {
        for which in &[
            "attn_q_norm.weight",
            "attn_k_norm.weight",
            "attn_norm.weight",
            "ffn_norm.weight",
            "ffn_gate.weight",
            "ffn_up.weight",
            "ffn_down.weight",
        ] {
            let key = format!("blk.{layer}.{which}");
            let Some(t) = f.tensor(&key) else { continue };
            let bytes = f.tensor_bytes(t);
            let v = dequant_to_f32(t, bytes).unwrap();
            let n = v.len().min(8);
            let mean = v.iter().sum::<f32>() / v.len() as f32;
            let std = {
                let m = mean;
                (v.iter().map(|x| (x - m).powi(2)).sum::<f32>() / v.len() as f32).sqrt()
            };
            println!(
                "{key}  shape={:?}  mean={:.4} std={:.4}  first8={:?}",
                t.shape,
                mean,
                std,
                &v[..n]
            );
        }
    }
}
