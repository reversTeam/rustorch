//! Criterion benchmark — RusTorch vs PyTorch comparison harness.
//!
//! Compiled with `target-cpu=native` (see `.cargo/config.toml`) so LLVM
//! auto-vectorizes for M4 NEON. Outputs criterion JSON to `target/criterion/`,
//! plus a flat `rustorch_results.json` for cross-tool comparison.
//!
//! Run via: `cargo bench --bench compare -- --save-baseline rustorch`
//! or in CI: `cargo bench --bench compare -- --output-format bencher`

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rustorch_amp::bf16_kernels::matmul_bf16_with_f32_accum;
use rustorch_attention::{flash_forward, naive_forward, AttentionShape};
use rustorch_core::Tensor;
use rustorch_cpu::backend::Backend;
use rustorch_cpu::cpu_backend::cpu_backend;
use rustorch_fusion::patterns::matmul_bias_act::{
    fused_matmul_bias_activation, naive_matmul_bias_activation, Activation,
};
use std::hint::black_box as hint_black_box;

/// Deterministic [-1, 1] f32 generator without `rand` dep.
fn det(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed.wrapping_mul(6364136223846793005);
    (0..n)
        .map(|i| {
            s = s.wrapping_add(i as u64).wrapping_mul(0xff51afd7ed558ccd);
            let bits = (s >> 33) as u32;
            (bits as f32 / u32::MAX as f32 - 0.5) * 2.0
        })
        .collect()
}

/// f32 matmul via the **CpuBackend trait** — the actual user-facing path.
/// Goes through `Backend::matmul` which dispatches to `matmul_naive::<f32>`.
fn bench_matmul_f32(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul_f32");
    // Shapes chosen to span L1d (192 KB), L2 (16 MB), and DRAM tiers on M4 Max.
    // Working set = (m*k + k*n + m*n) * 4 bytes
    //   64³  =>  48 KB  (fits L1d entirely)
    //   128³ => 192 KB  (L1d boundary)
    //   256³ => 768 KB  (L2)
    //   512³ =>  3 MB   (L2)
    //   1024³ => 12 MB  (still L2 boundary)
    //   2048³ => 48 MB  (DRAM)
    for &(m, k, n) in &[
        (64, 64, 64),
        (128, 128, 128),
        (256, 256, 256),
        (512, 512, 512),
        (1024, 1024, 1024),
    ] {
        let a = Tensor::from_vec(vec![m, k], det(0xA1, m * k)).unwrap();
        let b = Tensor::from_vec(vec![k, n], det(0xB2, k * n)).unwrap();
        // 2*M*K*N FLOPs per call.
        group.throughput(Throughput::Elements((2 * m * k * n) as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}x{}x{}", m, k, n)),
            &(m, k, n),
            |bench, _| {
                bench.iter(|| {
                    let c = cpu_backend().matmul(black_box(&a), black_box(&b)).unwrap();
                    hint_black_box(c)
                });
            },
        );
    }
    group.finish();
}

/// BF16 matmul with f32 accumulator — the AMP path. Inputs/outputs are bf16
/// but accumulation happens in f32 (matches GPU bf16 tensor cores).
fn bench_matmul_bf16(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul_bf16_acc_f32");
    for &(m, k, n) in &[(256, 256, 256), (512, 512, 512), (1024, 1024, 1024)] {
        let a_f32 = det(0xA1, m * k);
        let b_f32 = det(0xB2, k * n);
        // Convert to bf16 inputs.
        let a_bf16: Vec<half::bf16> = a_f32.iter().map(|&v| half::bf16::from_f32(v)).collect();
        let b_bf16: Vec<half::bf16> = b_f32.iter().map(|&v| half::bf16::from_f32(v)).collect();
        let mut out = vec![0.0_f32; m * n]; // f32 accumulator output
        group.throughput(Throughput::Elements((2 * m * k * n) as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}x{}x{}", m, k, n)),
            &(m, k, n),
            |bench, _| {
                bench.iter(|| {
                    matmul_bf16_with_f32_accum(
                        black_box(&a_bf16),
                        black_box(&b_bf16),
                        black_box(&mut out),
                        m,
                        k,
                        n,
                    )
                    .unwrap();
                    hint_black_box(&out);
                });
            },
        );
    }
    group.finish();
}

/// Their flagship `fused matmul+bias+ReLU` from `rustorch-fusion`.
/// Compares fused vs unfused (3-pass naive) at the same shapes.
fn bench_fused_matmul_bias_relu(c: &mut Criterion) {
    let mut group = c.benchmark_group("fused_mbr_vs_naive");
    for &(m, k, n) in &[(256, 256, 256), (512, 512, 512), (1024, 1024, 1024)] {
        let x = det(0xA1, m * k);
        let w = det(0xB2, k * n);
        let b = det(0xCC, n);
        let mut y = vec![0.0_f32; m * n];

        group.throughput(Throughput::Elements((2 * m * k * n) as u64));
        group.bench_function(
            BenchmarkId::new("fused", format!("{}x{}x{}", m, k, n)),
            |bb| {
                bb.iter(|| {
                    fused_matmul_bias_activation(
                        black_box(&x),
                        black_box(&w),
                        black_box(Some(&b)),
                        &mut y,
                        m,
                        k,
                        n,
                        Activation::Relu,
                    )
                    .unwrap();
                    hint_black_box(&y);
                });
            },
        );
        group.bench_function(
            BenchmarkId::new("naive", format!("{}x{}x{}", m, k, n)),
            |bb| {
                bb.iter(|| {
                    naive_matmul_bias_activation(
                        black_box(&x),
                        black_box(&w),
                        black_box(Some(&b)),
                        &mut y,
                        m,
                        k,
                        n,
                        Activation::Relu,
                    )
                    .unwrap();
                    hint_black_box(&y);
                });
            },
        );
    }
    group.finish();
}

/// Numerically stable softmax via `rustorch_cpu::kernels::softmax::softmax`.
fn bench_softmax(c: &mut Criterion) {
    let mut group = c.benchmark_group("softmax_lastdim");
    for &(rows, cols) in &[(128, 32_000), (512, 50_257), (1024, 50_257)] {
        let x = Tensor::from_vec(vec![rows, cols], det(0xCA, rows * cols)).unwrap();
        group.throughput(Throughput::Elements((rows * cols) as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}x{}", rows, cols)),
            &(rows, cols),
            |bb, _| {
                bb.iter(|| {
                    let y = rustorch_cpu::kernels::softmax::softmax(black_box(&x), 1).unwrap();
                    hint_black_box(y);
                });
            },
        );
    }
    group.finish();
}

/// Their **flagship**: tiled Flash Attention forward (CPU, rayon-parallel
/// over batch×head). N varies — the speedup vs naive is most visible at
/// long sequences.
fn bench_flash_attention(c: &mut Criterion) {
    let mut group = c.benchmark_group("flash_attn_vs_naive");
    for &(b, h, n, d) in &[(1, 1, 512, 64), (1, 4, 1024, 64), (2, 4, 2048, 64)] {
        let shape = AttentionShape::new(b, h, n, d);
        let bl = shape.buffer_len();
        let q = det(0xCAFE, bl);
        let k = det(0xBEEF, bl);
        let v = det(0xF00D, bl);
        let mut out = vec![0.0_f32; bl];

        // Approximate FLOPs: 2 matmuls of [N,D]x[D,N] + 2*[N,N]x[N,D].
        let flops = (2 * b * h * n * n * d * 2) as u64;
        group.throughput(Throughput::Elements(flops));

        let label = format!("B{}H{}N{}D{}", b, h, n, d);
        group.bench_function(BenchmarkId::new("flash", &label), |bb| {
            bb.iter(|| {
                flash_forward(black_box(&shape), &q, &k, &v, &mut out).unwrap();
                hint_black_box(&out);
            });
        });
        group.bench_function(BenchmarkId::new("naive", &label), |bb| {
            bb.iter(|| {
                naive_forward(black_box(&shape), &q, &k, &v, &mut out).unwrap();
                hint_black_box(&out);
            });
        });
    }
    group.finish();
}

/// Element-wise `add_` via the in-place tensor API.
fn bench_elementwise_add(c: &mut Criterion) {
    let mut group = c.benchmark_group("elementwise_add_inplace");
    // Span L1d → DRAM cache tiers (4 KB → 40 MB)
    for &n in &[1_000_usize, 10_000, 100_000, 1_000_000, 10_000_000] {
        let a_data = det(0xA1, n);
        let b_data = det(0xB2, n);
        let b = Tensor::from_vec(vec![n], b_data).unwrap();
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |bb, _| {
            bb.iter(|| {
                let mut a = Tensor::from_vec(vec![n], a_data.clone()).unwrap();
                a.add_(black_box(&b)).unwrap();
                hint_black_box(a)
            });
        });
    }
    group.finish();
}

/// T21 — full-scale GPT-2 small block forward (B=1 S=512 D=768
/// H=12 F=3072). Verifies the T18+T19+T20 wins hold on the real
/// production shape, not just the small dev shape.
fn bench_transformer_block_gpt2_small(c: &mut Criterion) {
    use rustorch_autograd::{no_grad, ops, Variable};
    use rustorch_fusion::patterns::matmul_bias_act::Activation;
    use rustorch_nn::{LayerNorm, Linear, Module, MultiHeadAttention};

    let batch = 1_usize;
    let seq = 512_usize;
    let d_model = 768_usize;
    let n_heads = 12_usize;
    let d_ff = 3072_usize;

    let ln1 = LayerNorm::new(d_model);
    let mha = MultiHeadAttention::new(d_model, n_heads);
    let ln2 = LayerNorm::new(d_model);
    let fc1 = Linear::new(d_model, d_ff);
    let fc2 = Linear::new(d_ff, d_model);

    let x_data = det(0xC1A0, batch * seq * d_model);
    let x_tensor = Tensor::from_vec(vec![batch, seq, d_model], x_data).unwrap();

    let mut group = c.benchmark_group("transformer_block_forward_gpt2_small");
    group.sample_size(20);
    group.warm_up_time(std::time::Duration::from_millis(800));
    group.measurement_time(std::time::Duration::from_secs(5));
    group.bench_function(
        BenchmarkId::from_parameter(format!(
            "B{}S{}D{}H{}F{}",
            batch, seq, d_model, n_heads, d_ff
        )),
        |bb| {
            bb.iter(|| {
                no_grad(|| {
                    let x = Variable::new(x_tensor.clone());
                    let h = ln1.forward(&x).unwrap();
                    let h = mha.forward(&h, &h, &h, None).unwrap();
                    let x = ops::add(&x, &h).unwrap();
                    let h = ln2.forward(&x).unwrap();
                    let h = fc1.forward_with_activation(&h, Activation::Relu).unwrap();
                    let h = fc2.forward(&h).unwrap();
                    let out = ops::add(&x, &h).unwrap();
                    hint_black_box(out)
                })
            });
        },
    );
    group.finish();
}

/// Per-stage breakdown of the GPT-2 transformer block — reveals
/// where the 8× end-to-end gap vs PyTorch concentrates.
fn bench_transformer_block_breakdown(c: &mut Criterion) {
    use rustorch_autograd::{no_grad, ops, Variable};
    use rustorch_fusion::patterns::matmul_bias_act::Activation;
    use rustorch_nn::{LayerNorm, Linear, Module, MultiHeadAttention};

    let batch = 2_usize;
    let seq = 128_usize;
    let d_model = 256_usize;
    let n_heads = 4_usize;
    let d_ff = 1024_usize;

    let ln1 = LayerNorm::new(d_model);
    let mha = MultiHeadAttention::new(d_model, n_heads);
    let ln2 = LayerNorm::new(d_model);
    let fc1 = Linear::new(d_model, d_ff);
    let fc2 = Linear::new(d_ff, d_model);

    let x_data = det(0xC1A0, batch * seq * d_model);
    let x_tensor = Tensor::from_vec(vec![batch, seq, d_model], x_data).unwrap();

    let mut group = c.benchmark_group("transformer_block_stage");
    group.sample_size(20);
    group.warm_up_time(std::time::Duration::from_millis(500));
    group.measurement_time(std::time::Duration::from_secs(3));

    group.bench_function("layernorm", |bb| {
        bb.iter(|| {
            no_grad(|| {
                let x = Variable::new(x_tensor.clone());
                hint_black_box(ln1.forward(&x).unwrap())
            })
        });
    });
    group.bench_function("mha_self", |bb| {
        bb.iter(|| {
            no_grad(|| {
                let x = Variable::new(x_tensor.clone());
                hint_black_box(mha.forward(&x, &x, &x, None).unwrap())
            })
        });
    });
    group.bench_function("fc1_relu_fc2", |bb| {
        bb.iter(|| {
            no_grad(|| {
                let x = Variable::new(x_tensor.clone());
                let h = fc1.forward_with_activation(&x, Activation::Relu).unwrap();
                hint_black_box(fc2.forward(&h).unwrap())
            })
        });
    });
    group.bench_function("residual_add", |bb| {
        let y_tensor = x_tensor.clone();
        bb.iter(|| {
            no_grad(|| {
                let x = Variable::new(x_tensor.clone());
                let y = Variable::new(y_tensor.clone());
                hint_black_box(ops::add(&x, &y).unwrap())
            })
        });
    });
    let _ = (&ln2,); // keep ln2 alive for the linker
    group.finish();
}

/// GPT-2-style transformer block forward (no_grad inference).
///
/// Layout:
///   x = ln1(x)
///   x = x + mha(x, x, x)          // attention sub-block (self-attention)
///   x = ln2(x)
///   x = x + fc2(relu(fc1(x)))     // FFN sub-block
///
/// Shape (matches GPT-2-small reduced for fast iteration):
///   batch=2, seq=128, d_model=256, n_heads=4, d_ff=1024
///
/// **Note**: ReLU is used instead of GELU (PyTorch reference also
/// uses ReLU for fair comparison). RusTorch lacks `ops::gelu` in
/// the autograd path as of this commit — TODO for a future task.
///
/// This is the metric that decides whether the perf sprint actually
/// translates to faster transformer inference vs PyTorch — micro
/// kernel wins are pointless if the end-to-end block is slower.
fn bench_transformer_block(c: &mut Criterion) {
    use rustorch_autograd::{no_grad, ops, Variable};
    use rustorch_fusion::patterns::matmul_bias_act::Activation;
    use rustorch_nn::{LayerNorm, Linear, Module, MultiHeadAttention};

    // Two shape regimes:
    //   "small"  — fast iteration: B=2 S=128 D=256 H=4  F=1024
    //   "gpt2"   — real GPT-2-small block: B=1 S=512 D=768 H=12 F=3072
    // The bench function below runs the small variant; a separate
    // function `bench_transformer_block_gpt2_small` covers the
    // real-scale variant (T21). This split keeps the small variant
    // fast for iterative dev.
    let batch = 2_usize;
    let seq = 128_usize;
    let d_model = 256_usize;
    let n_heads = 4_usize;
    let d_ff = 1024_usize;

    let ln1 = LayerNorm::new(d_model);
    let mha = MultiHeadAttention::new(d_model, n_heads);
    let ln2 = LayerNorm::new(d_model);
    let fc1 = Linear::new(d_model, d_ff);
    let fc2 = Linear::new(d_ff, d_model);

    // Pre-build the input outside the loop — we measure forward,
    // not allocation.
    let x_data = det(0xC1A0, batch * seq * d_model);
    let x_tensor = Tensor::from_vec(vec![batch, seq, d_model], x_data).unwrap();

    let mut group = c.benchmark_group("transformer_block_forward");
    group.sample_size(20);
    group.warm_up_time(std::time::Duration::from_millis(800));
    group.measurement_time(std::time::Duration::from_secs(5));
    group.bench_function(
        BenchmarkId::from_parameter(format!(
            "B{}S{}D{}H{}F{}",
            batch, seq, d_model, n_heads, d_ff
        )),
        |bb| {
            bb.iter(|| {
                no_grad(|| {
                    let x = Variable::new(x_tensor.clone());
                    // Attention sub-block: ln1 -> mha(self) -> residual.
                    let h = ln1.forward(&x).unwrap();
                    let h = mha.forward(&h, &h, &h, None).unwrap();
                    let x = ops::add(&x, &h).unwrap();
                    // FFN sub-block: ln2 -> fc1+relu fused -> fc2 -> residual.
                    // T20: fc1 + relu collapses to a single fused
                    // matmul+bias+act kernel call (one allocation).
                    let h = ln2.forward(&x).unwrap();
                    let h = fc1.forward_with_activation(&h, Activation::Relu).unwrap();
                    let h = fc2.forward(&h).unwrap();
                    let out = ops::add(&x, &h).unwrap();
                    hint_black_box(out)
                })
            });
        },
    );
    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(50)
        .warm_up_time(std::time::Duration::from_millis(500))
        .measurement_time(std::time::Duration::from_secs(3));
    targets =
        bench_matmul_f32,
        bench_matmul_bf16,
        bench_fused_matmul_bias_relu,
        bench_softmax,
        bench_flash_attention,
        bench_elementwise_add,
        bench_transformer_block,
        bench_transformer_block_gpt2_small,
        bench_transformer_block_breakdown,
}
criterion_main!(benches);
