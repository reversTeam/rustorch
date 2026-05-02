//! CPU vs GPU elementwise add benchmark.

use rustorch_core::tensor::tensor_impl::Tensor;
use rustorch_wgpu::{dispatch_binary, to_cpu, to_gpu, WgpuBackend};
use std::time::Instant;

fn cpu_add(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b).map(|(x, y)| x + y).collect()
}

fn bench_one(backend: &WgpuBackend, n: usize, repeat: usize) {
    let a: Vec<f32> = (0..n).map(|i| (i as f32) * 0.001).collect();
    let b: Vec<f32> = (0..n).map(|i| (i as f32) * 0.002).collect();

    let t0 = Instant::now();
    for _ in 0..repeat {
        let _ = std::hint::black_box(cpu_add(&a, &b));
    }
    let cpu_per = t0.elapsed() / repeat as u32;

    let ta = Tensor::from_vec([n], a).unwrap();
    let tb = Tensor::from_vec([n], b).unwrap();
    let ga = to_gpu(backend, &ta).unwrap();
    let gb = to_gpu(backend, &tb).unwrap();
    let _ = dispatch_binary(backend, "add", &ga, &gb).unwrap(); // warm-up

    // Compute-only (no readback) — simulates fused op chain on the GPU.
    let t0 = Instant::now();
    for _ in 0..repeat {
        let _ = dispatch_binary(backend, "add", &ga, &gb).unwrap();
    }
    let gpu_compute_per = t0.elapsed() / repeat as u32;

    // Compute + readback — simulates "do one op then come back to host".
    let t0 = Instant::now();
    for _ in 0..repeat {
        let gc = dispatch_binary(backend, "add", &ga, &gb).unwrap();
        let _ = to_cpu(backend, &gc, vec![n]).unwrap();
    }
    let gpu_round_trip_per = t0.elapsed() / repeat as u32;

    println!(
        "  N={:>9}  CPU {:>7.3} ms  | GPU compute {:>7.3} ms ({:>5.2}x)  | GPU+readback {:>7.3} ms ({:>5.2}x)",
        n,
        cpu_per.as_secs_f64() * 1000.0,
        gpu_compute_per.as_secs_f64() * 1000.0,
        cpu_per.as_secs_f64() / gpu_compute_per.as_secs_f64(),
        gpu_round_trip_per.as_secs_f64() * 1000.0,
        cpu_per.as_secs_f64() / gpu_round_trip_per.as_secs_f64()
    );
}

fn main() {
    let backend = WgpuBackend::new_blocking().expect("init wgpu");
    println!("Elementwise add: a + b");
    println!("---------------------------------------------------");
    // NOTE: 4M+ elements with workgroup_size=64 → 65536 groups, just above
    // the 65535 hard limit on dispatch_workgroups per dim. A 2D dispatch
    // (gid.x + gid.y * stride) would lift this; tracked for follow-up.
    for &n in &[1_024, 16_384, 262_144, 1_048_576, 2_097_152] {
        bench_one(&backend, n, 10);
    }
}
