//! Cross-backend reproducibility harness.
//!
//! On each backend (Metal / Vulkan / DX12 / WebGPU) this integration
//! test runs a deterministic chain of GPU ops on fixed seeded inputs
//! and compares the final result against a hand-rolled CPU reference.
//! Tolerances are per-op-class because:
//! - matmul accumulation order varies between tiled GPU and sequential
//!   CPU (relative error grows with K),
//! - softmax / log_softmax / layernorm involve `exp`/`log`/`sqrt` which
//!   round slightly differently across vendors.
//!
//! The same fixtures run on every CI worker, so any backend that
//! diverges past the configured tolerance fails the build at the
//! comparator step.
//!
//! Run with: `cargo test -p rustorch-wgpu --features gpu-tests --test parity`.

#![cfg(feature = "gpu-tests")]

use rustorch_core::tensor::tensor_impl::Tensor;
use rustorch_wgpu::{
    attention_naive, layernorm_rows, log_softmax_rows, matmul, reduce_rows, rmsnorm_rows,
    softmax_rows, to_cpu, to_gpu, transpose2d, ReduceKind, WgpuBackend,
};

// ----- Deterministic input generation ------------------------------------

/// Hash-based deterministic noise — avoids pulling in `rand` for fixtures.
/// `seed` is mixed with `i` so fixtures stay stable across runs.
fn det_f32(seed: u64, i: usize, lo: f32, hi: f32) -> f32 {
    let mut x = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(i as u64);
    x ^= x >> 33;
    x = x.wrapping_mul(0xff51afd7ed558ccd);
    x ^= x >> 33;
    let bits = (x & 0xFFFF_FFFF) as u32;
    let unit = (bits as f32) / (u32::MAX as f32);
    lo + (hi - lo) * unit
}

fn make(seed: u64, n: usize, lo: f32, hi: f32) -> Vec<f32> {
    (0..n).map(|i| det_f32(seed, i, lo, hi)).collect()
}

// ----- CPU references ----------------------------------------------------

fn cpu_matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0.0_f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut s = 0.0_f32;
            for kk in 0..k {
                s += a[i * k + kk] * b[kk * n + j];
            }
            out[i * n + j] = s;
        }
    }
    out
}

fn cpu_softmax(x: &[f32], b: usize, k: usize) -> Vec<f32> {
    let mut out = vec![0.0_f32; b * k];
    for r in 0..b {
        let row = &x[r * k..(r + 1) * k];
        let m = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = row.iter().map(|x| (x - m).exp()).collect();
        let z: f32 = exps.iter().sum();
        for (j, e) in exps.iter().enumerate() {
            out[r * k + j] = e / z;
        }
    }
    out
}

fn cpu_log_softmax(x: &[f32], b: usize, k: usize) -> Vec<f32> {
    let mut out = vec![0.0_f32; b * k];
    for r in 0..b {
        let row = &x[r * k..(r + 1) * k];
        let m = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let z: f32 = row.iter().map(|x| (x - m).exp()).sum();
        for (j, &v) in row.iter().enumerate() {
            out[r * k + j] = (v - m) - z.ln();
        }
    }
    out
}

fn cpu_layernorm(x: &[f32], gamma: &[f32], beta: &[f32], b: usize, k: usize, eps: f32) -> Vec<f32> {
    let mut out = vec![0.0_f32; b * k];
    for r in 0..b {
        let row = &x[r * k..(r + 1) * k];
        let mean = row.iter().sum::<f32>() / k as f32;
        let var = row.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / k as f32;
        let inv = 1.0 / (var + eps).sqrt();
        for j in 0..k {
            out[r * k + j] = (row[j] - mean) * inv * gamma[j] + beta[j];
        }
    }
    out
}

fn cpu_rmsnorm(x: &[f32], gamma: &[f32], b: usize, k: usize, eps: f32) -> Vec<f32> {
    let mut out = vec![0.0_f32; b * k];
    for r in 0..b {
        let row = &x[r * k..(r + 1) * k];
        let ms = row.iter().map(|v| v * v).sum::<f32>() / k as f32;
        let inv = 1.0 / (ms + eps).sqrt();
        for j in 0..k {
            out[r * k + j] = row[j] * inv * gamma[j];
        }
    }
    out
}

fn cpu_attention(q: &[f32], k: &[f32], v: &[f32], s: usize, d: usize) -> Vec<f32> {
    // scores = Q @ K^T / sqrt(d)
    let mut scores = vec![0.0_f32; s * s];
    for i in 0..s {
        for j in 0..s {
            let mut sum = 0.0_f32;
            for kk in 0..d {
                sum += q[i * d + kk] * k[j * d + kk];
            }
            scores[i * s + j] = sum / (d as f32).sqrt();
        }
    }
    let attn = cpu_softmax(&scores, s, s);
    // out = attn @ V
    let mut out = vec![0.0_f32; s * d];
    for i in 0..s {
        for j in 0..d {
            let mut sum = 0.0_f32;
            for kk in 0..s {
                sum += attn[i * s + kk] * v[kk * d + j];
            }
            out[i * d + j] = sum;
        }
    }
    out
}

// ----- Comparator --------------------------------------------------------

#[derive(Debug, Clone, Copy)]
enum Tol {
    /// Absolute tolerance — fine for shape-preserving element-wise ops.
    Abs(f32),
    /// Relative tolerance, scaled by max(|expected|, 1e-3) — needed for
    /// matmul / attention where absolute magnitudes can grow large.
    Rel(f32),
}

fn assert_close(actual: &[f32], expected: &[f32], tol: Tol, op: &str) {
    assert_eq!(actual.len(), expected.len(), "{op}: shape mismatch");
    let mut max_err: f32 = 0.0;
    let mut max_rel: f32 = 0.0;
    for (a, e) in actual.iter().zip(expected) {
        let err = (a - e).abs();
        let rel = err / e.abs().max(1e-3);
        max_err = max_err.max(err);
        max_rel = max_rel.max(rel);
        // `<=` so that `Tol::Abs(0.0)` allows bit-exact matches (e.g.
        // transpose2d, which permutes bits without ever touching FPUs).
        let pass = match tol {
            Tol::Abs(t) => err <= t,
            Tol::Rel(t) => rel <= t,
        };
        assert!(
            pass,
            "{op} divergence: actual={a} expected={e} err={err} rel={rel} (tol {tol:?})"
        );
    }
    eprintln!("  {op}: max_abs_err = {max_err:.3e}, max_rel_err = {max_rel:.3e}");
}

// ----- The actual fixtures -----------------------------------------------

#[test]
fn parity_matmul_64x64() {
    let backend = WgpuBackend::new_blocking().expect("init wgpu");
    let (m, k, n) = (64, 64, 64);
    let a = make(0xA1, m * k, -1.0, 1.0);
    let b = make(0xB2, k * n, -1.0, 1.0);
    let ta = Tensor::from_vec([m, k], a.clone()).unwrap();
    let tb = Tensor::from_vec([k, n], b.clone()).unwrap();
    let ga = to_gpu(&backend, &ta).unwrap();
    let gb = to_gpu(&backend, &tb).unwrap();
    let gc = matmul(&backend, &ga, &gb, m, k, n).unwrap();
    let c = to_cpu(&backend, &gc, vec![m, n]).unwrap();
    let expected = cpu_matmul(&a, &b, m, k, n);
    assert_close(
        c.as_slice::<f32>().unwrap(),
        &expected,
        Tol::Rel(1e-3),
        "matmul[64x64x64]",
    );
}

#[test]
fn parity_softmax_8x128() {
    let backend = WgpuBackend::new_blocking().expect("init wgpu");
    let (b, k) = (8, 128);
    let x = make(0xA50F7, b * k, -3.0, 3.0);
    let tx = Tensor::from_vec([b, k], x.clone()).unwrap();
    let gx = to_gpu(&backend, &tx).unwrap();
    let gy = softmax_rows(&backend, &gx, b, k).unwrap();
    let y = to_cpu(&backend, &gy, vec![b, k]).unwrap();
    assert_close(
        y.as_slice::<f32>().unwrap(),
        &cpu_softmax(&x, b, k),
        Tol::Abs(1e-5),
        "softmax[8x128]",
    );
}

#[test]
fn parity_log_softmax_8x128() {
    let backend = WgpuBackend::new_blocking().expect("init wgpu");
    let (b, k) = (8, 128);
    let x = make(0x10650, b * k, -3.0, 3.0);
    let tx = Tensor::from_vec([b, k], x.clone()).unwrap();
    let gx = to_gpu(&backend, &tx).unwrap();
    let gy = log_softmax_rows(&backend, &gx, b, k).unwrap();
    let y = to_cpu(&backend, &gy, vec![b, k]).unwrap();
    assert_close(
        y.as_slice::<f32>().unwrap(),
        &cpu_log_softmax(&x, b, k),
        Tol::Abs(1e-5),
        "log_softmax[8x128]",
    );
}

#[test]
fn parity_layernorm_4x64() {
    let backend = WgpuBackend::new_blocking().expect("init wgpu");
    let (b, k) = (4, 64);
    let x = make(0xCAFE, b * k, -2.0, 2.0);
    let gamma = make(0x6A33A, k, 0.5, 1.5);
    let beta = make(0xBE5A, k, -0.1, 0.1);
    let tx = Tensor::from_vec([b, k], x.clone()).unwrap();
    let tg = Tensor::from_vec([k], gamma.clone()).unwrap();
    let tb = Tensor::from_vec([k], beta.clone()).unwrap();
    let gx = to_gpu(&backend, &tx).unwrap();
    let gg = to_gpu(&backend, &tg).unwrap();
    let gbb = to_gpu(&backend, &tb).unwrap();
    let gy = layernorm_rows(&backend, &gx, &gg, &gbb, b, k, 1e-5).unwrap();
    let y = to_cpu(&backend, &gy, vec![b, k]).unwrap();
    assert_close(
        y.as_slice::<f32>().unwrap(),
        &cpu_layernorm(&x, &gamma, &beta, b, k, 1e-5),
        Tol::Abs(1e-3),
        "layernorm[4x64]",
    );
}

#[test]
fn parity_rmsnorm_4x64() {
    let backend = WgpuBackend::new_blocking().expect("init wgpu");
    let (b, k) = (4, 64);
    let x = make(0xCA50, b * k, -2.0, 2.0);
    let gamma = make(0x6A33, k, 0.5, 1.5);
    let tx = Tensor::from_vec([b, k], x.clone()).unwrap();
    let tg = Tensor::from_vec([k], gamma.clone()).unwrap();
    let gx = to_gpu(&backend, &tx).unwrap();
    let gg = to_gpu(&backend, &tg).unwrap();
    let gy = rmsnorm_rows(&backend, &gx, &gg, b, k, 1e-5).unwrap();
    let y = to_cpu(&backend, &gy, vec![b, k]).unwrap();
    assert_close(
        y.as_slice::<f32>().unwrap(),
        &cpu_rmsnorm(&x, &gamma, b, k, 1e-5),
        Tol::Abs(1e-3),
        "rmsnorm[4x64]",
    );
}

#[test]
fn parity_attention_seq16_d8() {
    let backend = WgpuBackend::new_blocking().expect("init wgpu");
    let (s, d) = (16, 8);
    let q = make(0x4, s * d, -0.5, 0.5);
    let k = make(0x5, s * d, -0.5, 0.5);
    let v = make(0x6, s * d, -0.5, 0.5);
    let tq = Tensor::from_vec([s, d], q.clone()).unwrap();
    let tk = Tensor::from_vec([s, d], k.clone()).unwrap();
    let tv = Tensor::from_vec([s, d], v.clone()).unwrap();
    let gq = to_gpu(&backend, &tq).unwrap();
    let gk = to_gpu(&backend, &tk).unwrap();
    let gv = to_gpu(&backend, &tv).unwrap();
    let go = attention_naive(&backend, &gq, &gk, &gv, s, d).unwrap();
    let o = to_cpu(&backend, &go, vec![s, d]).unwrap();
    assert_close(
        o.as_slice::<f32>().unwrap(),
        &cpu_attention(&q, &k, &v, s, d),
        Tol::Rel(1e-3),
        "attention[16x8]",
    );
}

#[test]
fn parity_reduce_sum_4x64() {
    let backend = WgpuBackend::new_blocking().expect("init wgpu");
    let (b, k) = (4, 64);
    let x = make(0x7E5C, b * k, -1.0, 1.0);
    let tx = Tensor::from_vec([b, k], x.clone()).unwrap();
    let gx = to_gpu(&backend, &tx).unwrap();
    let gs = reduce_rows(&backend, &gx, b, k, ReduceKind::Sum).unwrap();
    let s = to_cpu(&backend, &gs, vec![b]).unwrap();
    let mut expected = vec![0.0_f32; b];
    for r in 0..b {
        expected[r] = x[r * k..(r + 1) * k].iter().sum::<f32>();
    }
    assert_close(
        s.as_slice::<f32>().unwrap(),
        &expected,
        Tol::Rel(1e-3),
        "reduce_sum[4x64]",
    );
}

#[test]
fn parity_transpose_5x7() {
    let backend = WgpuBackend::new_blocking().expect("init wgpu");
    let (m, n) = (5, 7);
    let x = make(0xCA0A, m * n, -1.0, 1.0);
    let tx = Tensor::from_vec([m, n], x.clone()).unwrap();
    let gx = to_gpu(&backend, &tx).unwrap();
    let gt = transpose2d(&backend, &gx, m, n).unwrap();
    let t = to_cpu(&backend, &gt, vec![n, m]).unwrap();
    let mut expected = vec![0.0_f32; m * n];
    for i in 0..m {
        for j in 0..n {
            expected[j * m + i] = x[i * n + j];
        }
    }
    assert_close(
        t.as_slice::<f32>().unwrap(),
        &expected,
        Tol::Abs(0.0),
        "transpose2d[5x7]",
    );
}

/// End-to-end pipeline: input → matmul → softmax → matmul again. This
/// stresses the cumulative numerical drift across multiple kernels —
/// any backend that diverges past the chained tolerance fails here.
#[test]
fn parity_pipeline_matmul_softmax_matmul() {
    let backend = WgpuBackend::new_blocking().expect("init wgpu");
    let (m, k, n) = (16, 32, 16);
    let a = make(0xE2E0, m * k, -0.5, 0.5);
    let b = make(0xE2E1, k * n, -0.5, 0.5);
    let v = make(0xE2E2, n * 4, -0.5, 0.5);

    let ta = Tensor::from_vec([m, k], a.clone()).unwrap();
    let tb = Tensor::from_vec([k, n], b.clone()).unwrap();
    let tv = Tensor::from_vec([n, 4], v.clone()).unwrap();
    let ga = to_gpu(&backend, &ta).unwrap();
    let gb = to_gpu(&backend, &tb).unwrap();
    let gv = to_gpu(&backend, &tv).unwrap();
    let g_logits = matmul(&backend, &ga, &gb, m, k, n).unwrap();
    let g_probs = softmax_rows(&backend, &g_logits, m, n).unwrap();
    let g_out = matmul(&backend, &g_probs, &gv, m, n, 4).unwrap();
    let out = to_cpu(&backend, &g_out, vec![m, 4]).unwrap();

    // CPU reference
    let logits = cpu_matmul(&a, &b, m, k, n);
    let probs = cpu_softmax(&logits, m, n);
    let expected = cpu_matmul(&probs, &v, m, n, 4);

    assert_close(
        out.as_slice::<f32>().unwrap(),
        &expected,
        Tol::Rel(1e-3),
        "pipeline[matmul→softmax→matmul]",
    );
}
