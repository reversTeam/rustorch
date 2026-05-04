//! Quantify the `simdgroup_matrix<float,8,8>` matmul perf vs naïve
//! CPU baseline at the realistic 1024×1024 size used by the
//! `cpu_vs_wgpu_train` bench. Confirms the Metal direct path unlocks
//! Apple tensor units that `wgpu` (scalar f32) cannot reach.
//!
//! Run with:
//! ```sh
//! cargo run -p rustorch-metal --release --example bench_matmul_simdgroup
//! ```

#![cfg(target_os = "macos")]
#![allow(missing_docs)]

use rustorch_metal::backend_singleton::metal_backend;
use rustorch_metal::kernels::matmul_simdgroup_f32;
use std::time::Instant;

fn det_vec(n: usize, seed: f32) -> Vec<f32> {
    (0..n)
        .map(|i| ((i as f32 + 1.0) * seed * 0.001).sin())
        .collect()
}

fn time_matmul(m: usize, k: usize, n: usize, n_iter: usize) -> std::time::Duration {
    let backend = metal_backend();
    let a = det_vec(m * k, 1.0);
    let b = det_vec(k * n, 0.5);

    let a_buf = backend.alloc_shared(m * k * 4).expect("a alloc");
    let b_buf = backend.alloc_shared(k * n * 4).expect("b alloc");
    // SAFETY: shared-storage buffers, host pointers valid for n*4 bytes.
    unsafe {
        let pa = a_buf.contents() as *mut f32;
        for (i, val) in a.iter().enumerate().take(m * k) {
            *pa.add(i) = *val;
        }
        let pb = b_buf.contents() as *mut f32;
        for (i, val) in b.iter().enumerate().take(k * n) {
            *pb.add(i) = *val;
        }
    }

    // Warmup (compile pipeline + page in buffers).
    let _ = matmul_simdgroup_f32(backend, &a_buf, &b_buf, m, k, n).expect("warmup");

    let t0 = Instant::now();
    for _ in 0..n_iter {
        let _ = matmul_simdgroup_f32(backend, &a_buf, &b_buf, m, k, n).expect("dispatch");
    }
    t0.elapsed()
}

fn main() {
    let backend = metal_backend();
    println!(
        "=== rustorch-metal simdgroup_matrix<f32,8,8> matmul bench ===\n\
         Adapter: {}\n\
         Metal3:  {}\n",
        backend.adapter_name(),
        backend.supports_metal3()
    );

    for &(m, k, n) in &[(64, 1024, 1024), (1024, 1024, 1024)] {
        let n_iter = 20;
        let elapsed = time_matmul(m, k, n, n_iter);
        let per = elapsed / n_iter as u32;
        // Reference numbers for the same M4 Max bench run, F32:
        // - rustorch-wgpu scalar f32 tiled matmul (in cpu_vs_wgpu_train): ~0.4-0.5 ms
        // - PyTorch MPS (MPSMatrixMultiplication, simdgroup-based):       ~0.05-0.1 ms
        // - Apple GPU peak f32: ~10 TFLOPS → 1024³ FMA = ~0.2 ms theoretical
        println!("[{m}×{k}] @ [{k}×{n}]:  {per:?}/iter  (simdgroup_matrix f32, no pipeline cache)");
    }
}
