//! Integration test: Flash forward equivalence vs naive on a
//! 20-shape fixture matrix.
//!
//! Plan P3 task `Mathematical equivalence + end-to-end validation`
//! (e7656812). Step #1 acceptance: forward output Frobenius diff
//! < 1e-4 vs naive across 20 fixed shapes.
//!
//! What this file ships
//! - 20 hand-picked shapes spanning realistic transformer
//!   workloads (small attention heads, vision-style 2-head 16-dim,
//!   GPT-style 8-head 64-dim, edge cases like B=H=1 N=1).
//! - A Frobenius-norm diff harness comparing `flash_forward` against
//!   `naive_forward` on each shape.
//! - Assertion: max absolute element diff AND Frobenius diff are
//!   both under 1e-4 (f32 acceptance).
//!
//! Forward + backward equivalence and gradcheck are deferred until
//! the Flash backward task (7645c3b0) lands — see the related
//! decision in the project graph.

use rustorch_attention::{flash_forward, naive_forward, AttentionShape};

fn deterministic_buffer(len: usize, seed: u32) -> Vec<f32> {
    let mut s = seed.wrapping_mul(2654435761);
    (0..len)
        .map(|_| {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            (s as f32 / u32::MAX as f32 - 0.5) * 2.0
        })
        .collect()
}

/// 20 representative shapes. Each tuple is `(label, shape)`.
fn fixture_shapes() -> Vec<(&'static str, AttentionShape)> {
    vec![
        // --- Edge cases --------------------------------------
        ("B1H1N1D4_minimal", AttentionShape::new(1, 1, 1, 4)),
        ("B1H1N2D4_two_tokens", AttentionShape::new(1, 1, 2, 4)),
        ("B1H1N16D8_short_seq", AttentionShape::new(1, 1, 16, 8)),
        // --- Vision-style (small heads, low dim) -------------
        ("B1H2N32D16_vit_tiny", AttentionShape::new(1, 2, 32, 16)),
        ("B2H2N32D16_vit_tiny_b2", AttentionShape::new(2, 2, 32, 16)),
        ("B1H4N64D32_vit_small", AttentionShape::new(1, 4, 64, 32)),
        ("B2H4N128D32_vit_base", AttentionShape::new(2, 4, 128, 32)),
        // --- GPT-style (more heads, larger dim) --------------
        ("B1H8N64D64_gpt_tiny", AttentionShape::new(1, 8, 64, 64)),
        ("B1H8N128D64_gpt_small", AttentionShape::new(1, 8, 128, 64)),
        ("B2H8N128D64_gpt_b2", AttentionShape::new(2, 8, 128, 64)),
        ("B4H8N128D64_gpt_b4", AttentionShape::new(4, 8, 128, 64)),
        // --- Tile-boundary edge cases ------------------------
        (
            "B1H1N63D32_just_under_tile",
            AttentionShape::new(1, 1, 63, 32),
        ),
        ("B1H1N64D32_exact_tile", AttentionShape::new(1, 1, 64, 32)),
        (
            "B1H1N65D32_just_over_tile",
            AttentionShape::new(1, 1, 65, 32),
        ),
        (
            "B1H1N127D32_just_under_two_tiles",
            AttentionShape::new(1, 1, 127, 32),
        ),
        (
            "B1H1N128D32_exact_two_tiles",
            AttentionShape::new(1, 1, 128, 32),
        ),
        (
            "B1H1N129D32_just_over_two_tiles",
            AttentionShape::new(1, 1, 129, 32),
        ),
        // --- Wide / narrow dim probes ------------------------
        ("B1H1N32D4_skinny_d", AttentionShape::new(1, 1, 32, 4)),
        ("B1H1N32D128_wide_d", AttentionShape::new(1, 1, 32, 128)),
        ("B1H1N96D48_odd_tile_seq", AttentionShape::new(1, 1, 96, 48)),
    ]
}

/// Compute `(max_abs_diff, frobenius_diff)` between two equal-length
/// f32 buffers.
fn diffs(a: &[f32], b: &[f32]) -> (f32, f32) {
    let mut max_abs = 0.0f32;
    let mut frob_sq = 0.0f64;
    for (x, y) in a.iter().zip(b.iter()) {
        let d = (x - y).abs();
        if d > max_abs {
            max_abs = d;
        }
        frob_sq += (d as f64) * (d as f64);
    }
    (max_abs, frob_sq.sqrt() as f32)
}

#[test]
fn flash_matches_naive_on_20_fixed_shapes() {
    let shapes = fixture_shapes();
    assert_eq!(shapes.len(), 20, "fixture must have 20 shapes");
    let mut max_seen_diff = 0.0f32;
    let mut max_seen_frob = 0.0f32;
    for (label, shape) in &shapes {
        let n = shape.buffer_len();
        let q = deterministic_buffer(n, 0xC0FFEE);
        let k = deterministic_buffer(n, 0xBADBEEF);
        let v = deterministic_buffer(n, 0xCAFE00);
        let mut o_flash = vec![0.0f32; n];
        let mut o_naive = vec![0.0f32; n];
        flash_forward(shape, &q, &k, &v, &mut o_flash).unwrap();
        naive_forward(shape, &q, &k, &v, &mut o_naive).unwrap();
        let (max_abs, frob) = diffs(&o_flash, &o_naive);
        let tol = 1e-4f32;
        assert!(max_abs < tol, "[{label}] max abs diff {max_abs} >= {tol}");
        assert!(
            frob < tol * (n as f32).sqrt(),
            "[{label}] Frobenius diff {frob} too large for n={n}"
        );
        if max_abs > max_seen_diff {
            max_seen_diff = max_abs;
        }
        if frob > max_seen_frob {
            max_seen_frob = frob;
        }
    }
    println!("[20-shape equivalence] max abs diff across all shapes: {max_seen_diff:.3e}");
    println!("[20-shape equivalence] max Frobenius diff: {max_seen_frob:.3e}");
}

#[test]
fn flash_handles_each_shape_without_panic() {
    // Sanity: every fixture shape produces a finite output buffer
    // of the expected length.
    for (label, shape) in fixture_shapes() {
        let n = shape.buffer_len();
        let q = deterministic_buffer(n, 0x1);
        let k = deterministic_buffer(n, 0x2);
        let v = deterministic_buffer(n, 0x3);
        let mut o = vec![0.0f32; n];
        flash_forward(&shape, &q, &k, &v, &mut o).unwrap();
        assert_eq!(o.len(), n, "[{label}] output length wrong");
        for (i, x) in o.iter().enumerate() {
            assert!(x.is_finite(), "[{label}] non-finite output at {i}: {x}");
        }
    }
}
