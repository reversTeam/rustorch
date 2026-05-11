//! T246.7 P1.3 — Parity tests for the Lookahead Decoding tree kernels.
//!
//! Each tree kernel must, when invoked with a degenerate single-token tree
//! (tree_size=1, root with parent=-1), produce a bit-equivalent output to
//! the corresponding scalar (M=1) kernel that already drives `decode_step`.
//!
//! Running on GB10 (DGX Spark) :
//!   PATH=/usr/local/cuda-13.0/bin:$PATH \
//!     cargo test --release --features cuda -p rustorch-cuda \
//!       --test cuda_tree_parity -- --nocapture --test-threads=1

#![cfg(feature = "cuda")]

use cudarc::driver::{CudaContext, DevicePtr, DevicePtrMut};
use rustorch_cuda::llm_kernels::LlmKernels;
use std::sync::Arc;

// ─────────────────────────────────────────────────────────────────────────
// P1.3a — kv_append_tree_bf16(tree_size=1) ≡ kv_append_bf16_devcnt
// ─────────────────────────────────────────────────────────────────────────

#[test]
fn kv_append_tree_bf16_tree_size_1_matches_scalar() {
    let kv_dim = 256usize;
    let max_seq = 8i32;
    let pos_init: i32 = 3;

    let k_in: Vec<half::bf16> = (0..kv_dim)
        .map(|i| half::bf16::from_f32(((i as f32) * 0.013).sin() * 0.4))
        .collect();
    let v_in: Vec<half::bf16> = (0..kv_dim)
        .map(|i| half::bf16::from_f32(((i as f32) * 0.017).cos() * 0.3))
        .collect();

    let ctx = CudaContext::new(0).expect("ctx");
    let stream = ctx.default_stream();
    let kernels = LlmKernels::new(ctx);

    // Two cache buffers : one written via the scalar kernel, one via the
    // tree kernel. Both filled from same input ; must end bit-identical.
    let total = (max_seq as usize) * kv_dim;
    let mut k_cache_a = stream.alloc_zeros::<half::bf16>(total).expect("kca");
    let mut v_cache_a = stream.alloc_zeros::<half::bf16>(total).expect("vca");
    let mut k_cache_b = stream.alloc_zeros::<half::bf16>(total).expect("kcb");
    let mut v_cache_b = stream.alloc_zeros::<half::bf16>(total).expect("vcb");
    let k_in_dev = stream.memcpy_stod(&k_in).expect("k_in");
    let v_in_dev = stream.memcpy_stod(&v_in).expect("v_in");
    let pos_dev = stream.memcpy_stod(&[pos_init]).expect("pos");

    unsafe {
        let (kca_p, _g1) = k_cache_a.device_ptr_mut(&stream);
        let (vca_p, _g2) = v_cache_a.device_ptr_mut(&stream);
        let (k_p, _g3) = k_in_dev.device_ptr(&stream);
        let (v_p, _g4) = v_in_dev.device_ptr(&stream);
        let (pos_p, _g5) = pos_dev.device_ptr(&stream);
        kernels
            .kv_append_bf16_devcnt(&stream, kca_p, vca_p, k_p, v_p, pos_p, kv_dim as i32)
            .unwrap();
    }
    unsafe {
        let (kcb_p, _g1) = k_cache_b.device_ptr_mut(&stream);
        let (vcb_p, _g2) = v_cache_b.device_ptr_mut(&stream);
        let (k_p, _g3) = k_in_dev.device_ptr(&stream);
        let (v_p, _g4) = v_in_dev.device_ptr(&stream);
        let (pos_p, _g5) = pos_dev.device_ptr(&stream);
        kernels
            .kv_append_tree_bf16(
                &stream,
                kcb_p,
                vcb_p,
                k_p,
                v_p,
                pos_p,
                1,
                kv_dim as i32,
                max_seq,
            )
            .unwrap();
    }

    let ka: Vec<half::bf16> = stream.memcpy_dtov(&k_cache_a).expect("dl ka");
    let va: Vec<half::bf16> = stream.memcpy_dtov(&v_cache_a).expect("dl va");
    let kb: Vec<half::bf16> = stream.memcpy_dtov(&k_cache_b).expect("dl kb");
    let vb: Vec<half::bf16> = stream.memcpy_dtov(&v_cache_b).expect("dl vb");

    assert_eq!(
        ka, kb,
        "kv_append_tree(tree_size=1) ≠ kv_append_devcnt for K"
    );
    assert_eq!(
        va, vb,
        "kv_append_tree(tree_size=1) ≠ kv_append_devcnt for V"
    );
}

#[test]
fn kv_append_tree_bf16_writes_consecutive_rows() {
    // tree_size=4 must write rows pos, pos+1, pos+2, pos+3 of the cache.
    let kv_dim = 64usize;
    let max_seq = 16i32;
    let pos_init: i32 = 5;
    let tree_size = 4i32;

    let k_in: Vec<half::bf16> = (0..(tree_size as usize) * kv_dim)
        .map(|i| half::bf16::from_f32((i as f32) * 0.01))
        .collect();
    let v_in: Vec<half::bf16> = (0..(tree_size as usize) * kv_dim)
        .map(|i| half::bf16::from_f32((i as f32) * 0.02))
        .collect();

    let ctx = CudaContext::new(0).expect("ctx");
    let stream = ctx.default_stream();
    let kernels = LlmKernels::new(ctx);
    let mut k_cache = stream
        .alloc_zeros::<half::bf16>((max_seq as usize) * kv_dim)
        .expect("kc");
    let mut v_cache = stream
        .alloc_zeros::<half::bf16>((max_seq as usize) * kv_dim)
        .expect("vc");
    let k_in_dev = stream.memcpy_stod(&k_in).expect("k_in");
    let v_in_dev = stream.memcpy_stod(&v_in).expect("v_in");
    let pos_dev = stream.memcpy_stod(&[pos_init]).expect("pos");

    unsafe {
        let (kc_p, _g1) = k_cache.device_ptr_mut(&stream);
        let (vc_p, _g2) = v_cache.device_ptr_mut(&stream);
        let (k_p, _g3) = k_in_dev.device_ptr(&stream);
        let (v_p, _g4) = v_in_dev.device_ptr(&stream);
        let (pos_p, _g5) = pos_dev.device_ptr(&stream);
        kernels
            .kv_append_tree_bf16(
                &stream,
                kc_p,
                vc_p,
                k_p,
                v_p,
                pos_p,
                tree_size,
                kv_dim as i32,
                max_seq,
            )
            .unwrap();
    }
    let kh: Vec<half::bf16> = stream.memcpy_dtov(&k_cache).expect("dl k");
    let vh: Vec<half::bf16> = stream.memcpy_dtov(&v_cache).expect("dl v");

    // Rows [0..pos_init) and [pos_init+tree_size..max_seq) must be zero.
    for r in 0..(max_seq as usize) {
        for j in 0..kv_dim {
            let idx = r * kv_dim + j;
            if (r as i32) < pos_init || (r as i32) >= pos_init + tree_size {
                assert_eq!(
                    kh[idx].to_f32(),
                    0.0,
                    "K[{r},{j}] should be untouched (zero)"
                );
                assert_eq!(vh[idx].to_f32(), 0.0, "V[{r},{j}] should be untouched");
            } else {
                let row_in_tree = (r as i32) - pos_init;
                let in_idx = (row_in_tree as usize) * kv_dim + j;
                assert_eq!(kh[idx], k_in[in_idx], "K[{r},{j}] mismatch");
                assert_eq!(vh[idx], v_in[in_idx], "V[{r},{j}] mismatch");
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────
// P1.3b — gqa_decode_tree_bf16(tree_size=1) ≡ gqa_decode_split_bf16
// ─────────────────────────────────────────────────────────────────────────

#[test]
fn gqa_decode_tree_bf16_tree_size_1_matches_split() {
    let n_q: usize = 8;
    let n_kv: usize = 2;
    let head_dim: usize = 64;
    let max_seq: i32 = 32;
    let n_split: i32 = 4;
    let kv_len: i32 = 17;

    let q_host: Vec<half::bf16> = (0..n_q * head_dim)
        .map(|i| half::bf16::from_f32(((i as f32) * 0.07).sin() * 0.4))
        .collect();
    let kv_total = n_kv * (max_seq as usize) * head_dim;
    let k_host: Vec<half::bf16> = (0..kv_total)
        .map(|i| half::bf16::from_f32(((i as f32) * 0.011).cos() * 0.3))
        .collect();
    let v_host: Vec<half::bf16> = (0..kv_total)
        .map(|i| half::bf16::from_f32(((i as f32) * 0.017).sin() * 0.25))
        .collect();

    let ctx = CudaContext::new(0).expect("ctx");
    let stream = ctx.default_stream();
    let kernels = LlmKernels::new(ctx);

    let q_dev = stream.memcpy_stod(&q_host).expect("q");
    let k_dev = stream.memcpy_stod(&k_host).expect("k");
    let v_dev = stream.memcpy_stod(&v_host).expect("v");
    let kv_len_dev = stream.memcpy_stod(&[kv_len]).expect("kv_len_dev");

    let mut out_split = stream
        .alloc_zeros::<half::bf16>(n_q * head_dim)
        .expect("os");
    let mut pm_split = stream
        .alloc_zeros::<f32>(n_q * (n_split as usize))
        .expect("pm");
    let mut pl_split = stream
        .alloc_zeros::<f32>(n_q * (n_split as usize))
        .expect("pl");
    let mut po_split = stream
        .alloc_zeros::<half::bf16>(n_q * (n_split as usize) * head_dim)
        .expect("po");

    unsafe {
        let (q_p, _g1) = q_dev.device_ptr(&stream);
        let (k_p, _g2) = k_dev.device_ptr(&stream);
        let (v_p, _g3) = v_dev.device_ptr(&stream);
        let (o_p, _g4) = out_split.device_ptr_mut(&stream);
        let (pm_p, _g5) = pm_split.device_ptr_mut(&stream);
        let (pl_p, _g6) = pl_split.device_ptr_mut(&stream);
        let (po_p, _g7) = po_split.device_ptr_mut(&stream);
        let (kld_p, _g8) = kv_len_dev.device_ptr(&stream);
        kernels
            .gqa_decode_split_bf16(
                &stream,
                q_p,
                k_p,
                v_p,
                o_p,
                pm_p,
                pl_p,
                po_p,
                n_q as i32,
                n_kv as i32,
                kld_p,
                head_dim as i32,
                max_seq,
                n_split,
            )
            .unwrap();
    }
    let y_split: Vec<half::bf16> = stream.memcpy_dtov(&out_split).expect("dl split");

    // Tree variant : tree_size=1, parent=-1, depth=0.
    let parents = stream.memcpy_stod(&[-1i32]).expect("par");
    let depths = stream.memcpy_stod(&[0u16]).expect("dep");

    let mut out_tree = stream
        .alloc_zeros::<half::bf16>(n_q * head_dim)
        .expect("ot");
    let mut pm_tree = stream
        .alloc_zeros::<f32>(1 * n_q * (n_split as usize))
        .expect("pmt");
    let mut pl_tree = stream
        .alloc_zeros::<f32>(1 * n_q * (n_split as usize))
        .expect("plt");
    let mut po_tree = stream
        .alloc_zeros::<half::bf16>(1 * n_q * (n_split as usize) * head_dim)
        .expect("pot");

    unsafe {
        let (q_p, _g1) = q_dev.device_ptr(&stream);
        let (k_p, _g2) = k_dev.device_ptr(&stream);
        let (v_p, _g3) = v_dev.device_ptr(&stream);
        let (o_p, _g4) = out_tree.device_ptr_mut(&stream);
        let (par_p, _g5) = parents.device_ptr(&stream);
        let (dep_p, _g6) = depths.device_ptr(&stream);
        let (pm_p, _g7) = pm_tree.device_ptr_mut(&stream);
        let (pl_p, _g8) = pl_tree.device_ptr_mut(&stream);
        let (po_p, _g9) = po_tree.device_ptr_mut(&stream);
        let (kld_p, _g10) = kv_len_dev.device_ptr(&stream);
        kernels
            .gqa_decode_tree_bf16(
                &stream,
                q_p,
                k_p,
                v_p,
                o_p,
                par_p,
                dep_p,
                pm_p,
                pl_p,
                po_p,
                n_q as i32,
                n_kv as i32,
                kld_p,
                head_dim as i32,
                max_seq,
                n_split,
                1,
            )
            .unwrap();
    }
    let y_tree: Vec<half::bf16> = stream.memcpy_dtov(&out_tree).expect("dl tree");

    // For tree_size=1, parents=[-1], depth=0 : the root's KV is at slot
    // kv_len-1 ∈ [0, kv_len), which Phase A already attends to. Phase B
    // is then empty (the chain has only the root, which is skipped).
    // → Bit-exact parity with `gqa_decode_split_bf16`.
    assert_eq!(y_tree, y_split, "tree(size=1) ≠ split (bit-exact expected)");
}

// ─────────────────────────────────────────────────────────────────────────
// P1.3c — argmax_logits_tree_bf16(tree_size=1) ≡ argmax_bf16
// ─────────────────────────────────────────────────────────────────────────

#[test]
fn argmax_logits_tree_bf16_tree_size_1_matches_scalar() {
    let vocab: usize = 4096;
    let logits: Vec<half::bf16> = (0..vocab)
        .map(|i| half::bf16::from_f32(((i as f32) * 0.0031).sin() * 2.0))
        .collect();

    let ctx = CudaContext::new(0).expect("ctx");
    let stream = ctx.default_stream();
    let kernels = LlmKernels::new(ctx);

    let logits_dev = stream.memcpy_stod(&logits).expect("logits");
    let mut tok_a = stream.alloc_zeros::<u32>(1).expect("tok_a");
    let mut tok_b = stream.alloc_zeros::<u32>(1).expect("tok_b");

    unsafe {
        let (l_p, _g) = logits_dev.device_ptr(&stream);
        let (a_p, _g2) = tok_a.device_ptr_mut(&stream);
        kernels
            .argmax_bf16(&stream, l_p, a_p, vocab as i32)
            .unwrap();
    }
    unsafe {
        let (l_p, _g) = logits_dev.device_ptr(&stream);
        let (b_p, _g2) = tok_b.device_ptr_mut(&stream);
        kernels
            .argmax_logits_tree_bf16(&stream, l_p, b_p, 1, vocab as i32)
            .unwrap();
    }

    let a: Vec<u32> = stream.memcpy_dtov(&tok_a).expect("dl a");
    let b: Vec<u32> = stream.memcpy_dtov(&tok_b).expect("dl b");
    assert_eq!(a, b, "argmax_logits_tree(size=1) ≠ argmax_bf16");
}

#[test]
fn argmax_logits_tree_bf16_per_row_argmax() {
    let tree_size: usize = 4;
    let vocab: usize = 1024;
    // Build logits where row r has its peak at index r * 17 + 1.
    let mut logits = vec![half::bf16::from_f32(0.0); tree_size * vocab];
    for r in 0..tree_size {
        for i in 0..vocab {
            logits[r * vocab + i] = half::bf16::from_f32(((i as f32) * 0.001).sin() * 0.1);
        }
        logits[r * vocab + r * 17 + 1] = half::bf16::from_f32(99.0);
    }

    let ctx = CudaContext::new(0).expect("ctx");
    let stream = ctx.default_stream();
    let kernels = LlmKernels::new(ctx);

    let logits_dev = stream.memcpy_stod(&logits).expect("logits");
    let mut tok = stream.alloc_zeros::<u32>(tree_size).expect("tok");
    unsafe {
        let (l_p, _g) = logits_dev.device_ptr(&stream);
        let (t_p, _g2) = tok.device_ptr_mut(&stream);
        kernels
            .argmax_logits_tree_bf16(&stream, l_p, t_p, tree_size as i32, vocab as i32)
            .unwrap();
    }
    let toks: Vec<u32> = stream.memcpy_dtov(&tok).expect("dl");
    for r in 0..tree_size {
        assert_eq!(toks[r], (r * 17 + 1) as u32, "row {r} argmax mismatch");
    }
}

// ─────────────────────────────────────────────────────────────────────────
// P1.3c — counter helpers
// ─────────────────────────────────────────────────────────────────────────

#[test]
fn add_u32_dev_increments_by_value() {
    let ctx = CudaContext::new(0).expect("ctx");
    let stream = ctx.default_stream();
    let kernels = LlmKernels::new(ctx);

    let mut counter = stream.memcpy_stod(&[10i32]).expect("counter");
    unsafe {
        let (p, _g) = counter.device_ptr_mut(&stream);
        kernels.add_u32_dev(&stream, p, 5).unwrap();
        kernels.add_u32_dev(&stream, p, 7).unwrap();
    }
    let v: Vec<i32> = stream.memcpy_dtov(&counter).expect("dl");
    assert_eq!(v[0], 22, "expected 10 + 5 + 7 = 22");
}

#[test]
fn set_u32_dev_writes_value() {
    let ctx = CudaContext::new(0).expect("ctx");
    let stream = ctx.default_stream();
    let kernels = LlmKernels::new(ctx);

    let mut buf = stream.memcpy_stod(&[999i32]).expect("buf");
    unsafe {
        let (p, _g) = buf.device_ptr_mut(&stream);
        kernels.set_u32_dev(&stream, p, 42).unwrap();
    }
    let v: Vec<i32> = stream.memcpy_dtov(&buf).expect("dl");
    assert_eq!(v[0], 42);
}

// Avoid unused warning on the import on non-cuda builds.
#[allow(dead_code)]
fn _arc_unused() -> Arc<()> {
    Arc::new(())
}
