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
///
/// **T23 fairness fix**: prior versions cloned `a_data` and built a
/// fresh Tensor inside the iter loop, which on n=1M ≈ 4 MB charged
/// the measurement with a DRAM-bandwidth-bound clone (~50 µs) plus
/// allocator + Tensor::from_vec setup. PyTorch's bench loop does
/// `a = make(...)` ONCE outside the timer and then `a.add_(b)`
/// in-place, so we matched that — `iter_batched` prepares a fresh
/// `a` per iteration but only the `add_` call is timed.
fn bench_elementwise_add(c: &mut Criterion) {
    let mut group = c.benchmark_group("elementwise_add_inplace");
    // Span L1d → DRAM cache tiers (4 KB → 40 MB)
    for &n in &[1_000_usize, 10_000, 100_000, 1_000_000, 10_000_000] {
        let a_data = det(0xA1, n);
        let b_data = det(0xB2, n);
        let b = Tensor::from_vec(vec![n], b_data).unwrap();
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |bb, _| {
            bb.iter_batched(
                || Tensor::from_vec(vec![n], a_data.clone()).unwrap(),
                |mut a| {
                    a.add_(black_box(&b)).unwrap();
                    hint_black_box(a)
                },
                criterion::BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

/// T32 — LM head isolated bench: a `Linear(D=768 → V=50257)`
/// matmul on `[128, 768]` rows. This is the final logit
/// projection in any LLM forward, dominating ~10 G FLOPs and
/// often the biggest single op. Critical for autoregressive
/// decode where the full vocab is materialised every token.
fn bench_lm_head(c: &mut Criterion) {
    use rustorch_autograd::{no_grad, Variable};
    use rustorch_nn::{Linear, Module};

    let lm_head = Linear::new(768, 50257);
    let x_data = det(0xC1A6, 128 * 768);
    let x_t = Tensor::from_vec(vec![128usize, 768], x_data).unwrap();
    let mut group = c.benchmark_group("lm_head");
    group.sample_size(20);
    group.warm_up_time(std::time::Duration::from_millis(500));
    group.measurement_time(std::time::Duration::from_secs(3));
    group.bench_function("linear_128x768x50257", |bb| {
        bb.iter(|| {
            no_grad(|| {
                let x = Variable::new(x_t.clone());
                hint_black_box(lm_head.forward(&x).unwrap())
            })
        });
    });
    group.finish();
}

/// T44 — RoPE bench (Rotary Position Embeddings). On every LLM
/// forward this is applied to Q and K just before attention; for
/// long contexts the throughput matters. Validates that our T42
/// implementation matches or beats PyTorch's equivalent.
fn bench_rope(c: &mut Criterion) {
    use rustorch_nn::rope::RoPE;

    let head_dim = 64_usize;
    let n_heads = 12_usize;
    let seq = 512_usize;
    let batch = 1_usize;
    let n = batch * n_heads * seq * head_dim;

    let rope = RoPE::new(head_dim, 1024, 10000.0);
    let mut x = det(0xC0DE, n);

    let mut group = c.benchmark_group("rope_apply");
    group.sample_size(50);
    group.warm_up_time(std::time::Duration::from_millis(300));
    group.measurement_time(std::time::Duration::from_secs(2));
    group.bench_function(
        BenchmarkId::from_parameter(format!("B{}H{}S{}D{}", batch, n_heads, seq, head_dim)),
        |bb| {
            bb.iter(|| {
                rope.apply_inplace(&mut x, batch, n_heads, seq, 0).unwrap();
                hint_black_box(&x);
            });
        },
    );
    group.finish();
}

/// T44 — sampling bench (top-p + top-k + temperature). One sample
/// per token at decode time; on a 50K-vocab logits this loop runs
/// for every generated token, so it must not become a bottleneck.
fn bench_sampling(c: &mut Criterion) {
    use rustorch_nn::sampling::{sample_next, SamplingConfig};

    let vocab_size = 50257_usize;
    let logits = det(0xCA, vocab_size);
    let cfg = SamplingConfig::default(); // T=1 top_k=50 top_p=0.95

    // Deterministic LCG for reproducibility.
    let mut s: u64 = 1;
    let mut next_u = move || {
        s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        ((s >> 32) as f32) / (u32::MAX as f32)
    };

    let mut group = c.benchmark_group("sampling");
    group.sample_size(50);
    group.bench_function(
        BenchmarkId::from_parameter(format!("vocab{}_topk50_topp95", vocab_size)),
        |bb| {
            bb.iter(|| {
                let pick = sample_next(&logits, &cfg, &[], &mut next_u);
                hint_black_box(pick);
            });
        },
    );
    group.finish();
}

/// T36 — micro bench: raw `fused_matmul_bias_activation` call
/// vs the wrapper Linear S=1 path. Isolates the overhead of the
/// Tensor + Variable wrapping vs the actual sgemv work.
fn bench_linear_S1_micro(c: &mut Criterion) {
    use rustorch_autograd::{no_grad, Variable};
    use rustorch_fusion::patterns::matmul_bias_act::{fused_matmul_bias_activation, Activation};
    use rustorch_nn::{Linear, Module};

    let m = 1_usize;
    let k = 768_usize;
    let n = 768_usize;
    let x = det(0xA1, m * k);
    let w = det(0xB2, k * n);
    let b = det(0xCC, n);
    let mut y = vec![0.0_f32; m * n];

    let mut group = c.benchmark_group("linear_S1_micro");
    group.sample_size(50);
    group.warm_up_time(std::time::Duration::from_millis(300));
    group.measurement_time(std::time::Duration::from_secs(3));

    // Baseline: pure raw kernel call (no Tensor, no Variable).
    group.bench_function("fused_kernel_raw", |bb| {
        bb.iter(|| {
            fused_matmul_bias_activation(&x, &w, Some(&b), &mut y, m, k, n, Activation::None)
                .unwrap();
            hint_black_box(&y);
        });
    });

    // Through the Linear module forward (Tensor + Variable wrap).
    let fc = Linear::new(k, n);
    let x_t = Tensor::from_vec(vec![m, k], x.clone()).unwrap();
    group.bench_function("linear_forward_full", |bb| {
        bb.iter(|| {
            no_grad(|| {
                let xv = Variable::new(x_t.clone());
                hint_black_box(fc.forward(&xv).unwrap())
            })
        });
    });

    group.finish();
}

/// T33 — per-stage breakdown of the S=1 single-token decode to
/// pinpoint where the 29.5× gap vs PyTorch comes from. Same
/// methodology as T17 did for the S=128 block.
fn bench_single_token_breakdown(c: &mut Criterion) {
    use rustorch_autograd::{no_grad, ops, Variable};
    use rustorch_fusion::patterns::matmul_bias_act::Activation;
    use rustorch_nn::{Embedding, LayerNorm, Linear, Module, MultiHeadAttention};

    let d_model = 768_usize;
    let n_heads = 12_usize;
    let d_ff = 3072_usize;
    let vocab_size = 50257_usize;

    let token_emb = Embedding::with_seed(vocab_size, d_model, 0xC1A4);
    let ln = LayerNorm::new(d_model);
    let mha = MultiHeadAttention::new(d_model, n_heads);
    let fc1 = Linear::new(d_model, d_ff);
    let fc2 = Linear::new(d_ff, d_model);
    let lm_head = Linear::new(d_model, vocab_size);

    let ids_t = Tensor::from_vec_typed::<i64, _>([1_usize], vec![42_i64]).unwrap();
    let x_t = Tensor::from_vec(vec![1usize, 1, d_model], vec![0.1_f32; d_model]).unwrap();

    let mut group = c.benchmark_group("single_token_stage");
    group.sample_size(20);
    group.warm_up_time(std::time::Duration::from_millis(300));
    group.measurement_time(std::time::Duration::from_secs(2));

    group.bench_function("variable_wrap", |bb| {
        bb.iter(|| no_grad(|| hint_black_box(Variable::new(x_t.clone()))));
    });

    group.bench_function("embedding_lookup_S1", |bb| {
        bb.iter(|| no_grad(|| hint_black_box(token_emb.forward_indices(&ids_t).unwrap())));
    });

    group.bench_function("layernorm_S1", |bb| {
        bb.iter(|| {
            no_grad(|| {
                let x = Variable::new(x_t.clone());
                hint_black_box(ln.forward(&x).unwrap())
            })
        });
    });

    group.bench_function("mha_self_S1", |bb| {
        bb.iter(|| {
            no_grad(|| {
                let x = Variable::new(x_t.clone());
                hint_black_box(mha.forward(&x, &x, &x, None).unwrap())
            })
        });
    });

    group.bench_function("ffn_S1", |bb| {
        bb.iter(|| {
            no_grad(|| {
                let x = Variable::new(x_t.clone());
                let h = fc1.forward_with_activation(&x, Activation::Relu).unwrap();
                hint_black_box(fc2.forward(&h).unwrap())
            })
        });
    });

    group.bench_function("residual_add_S1", |bb| {
        let y_t = x_t.clone();
        bb.iter(|| {
            no_grad(|| {
                let x = Variable::new(x_t.clone());
                let y = Variable::new(y_t.clone());
                hint_black_box(ops::add(&x, &y).unwrap())
            })
        });
    });

    group.bench_function("lm_head_S1", |bb| {
        let small_t = Tensor::from_vec(vec![1usize, d_model], vec![0.1_f32; d_model]).unwrap();
        bb.iter(|| {
            no_grad(|| {
                let x = Variable::new(small_t.clone());
                hint_black_box(lm_head.forward(&x).unwrap())
            })
        });
    });

    group.finish();
}

/// T32 — single-token forward (S=1) pass through the GPT-2-small
/// stack. Models the cold-start latency of generating the FIRST
/// token of an autoregressive decode. Without KV-cache this also
/// represents the per-token cost of every subsequent decode step
/// (worst case naive).
fn bench_gpt2_single_token_decode(c: &mut Criterion) {
    use rustorch_autograd::{no_grad, ops, Variable};
    use rustorch_fusion::patterns::matmul_bias_act::Activation;
    use rustorch_nn::{Embedding, LayerNorm, Linear, Module, MultiHeadAttention};

    let batch = 1_usize;
    let seq = 1_usize;
    let d_model = 768_usize;
    let n_heads = 12_usize;
    let d_ff = 3072_usize;
    let num_layers = 12_usize;
    let vocab_size = 50257_usize;

    let token_emb = Embedding::with_seed(vocab_size, d_model, 0xC1A4);
    let pos_emb = Embedding::with_seed(seq.max(1), d_model, 0xC1A5);
    let mut layers: Vec<(LayerNorm, MultiHeadAttention, LayerNorm, Linear, Linear)> =
        Vec::with_capacity(num_layers);
    for _ in 0..num_layers {
        layers.push((
            LayerNorm::new(d_model),
            MultiHeadAttention::new(d_model, n_heads),
            LayerNorm::new(d_model),
            Linear::new(d_model, d_ff),
            Linear::new(d_ff, d_model),
        ));
    }
    let final_ln = LayerNorm::new(d_model);
    let lm_head = Linear::new(d_model, vocab_size);

    let ids: Vec<i64> = vec![42_i64];
    let ids_t = Tensor::from_vec_typed::<i64, _>([1_usize], ids).unwrap();
    let pos_t = Tensor::from_vec_typed::<i64, _>([1_usize], vec![0_i64]).unwrap();

    let mut group = c.benchmark_group("gpt2_single_token_decode");
    group.sample_size(20);
    group.warm_up_time(std::time::Duration::from_millis(500));
    group.measurement_time(std::time::Duration::from_secs(4));
    group.bench_function("L12_S1_D768", |bb| {
        bb.iter(|| {
            no_grad(|| {
                let tok = token_emb.forward_indices(&ids_t).unwrap();
                let pos = pos_emb.forward_indices(&pos_t).unwrap();
                let mut x = ops::add(&tok, &pos).unwrap();
                x = ops::reshape(&x, vec![batch, seq, d_model]).unwrap();
                for (ln1, mha, ln2, fc1, fc2) in layers.iter() {
                    let h = ln1.forward(&x).unwrap();
                    let h = mha.forward(&h, &h, &h, None).unwrap();
                    x = ops::add(&x, &h).unwrap();
                    let h = ln2.forward(&x).unwrap();
                    let h = fc1.forward_with_activation(&h, Activation::Relu).unwrap();
                    let h = fc2.forward(&h).unwrap();
                    x = ops::add(&x, &h).unwrap();
                }
                let h = final_ln.forward(&x).unwrap();
                hint_black_box(lm_head.forward(&h).unwrap())
            })
        });
    });
    group.finish();
}

/// T30 — full GPT-2-small forward stack: 12 transformer blocks
/// chained, with input-side embedding + final LayerNorm + LM head.
/// This is what a real LLM does on a single forward pass.
fn bench_gpt2_full_stack(c: &mut Criterion) {
    use rustorch_autograd::{no_grad, ops, Variable};
    use rustorch_fusion::patterns::matmul_bias_act::Activation;
    use rustorch_nn::{Embedding, LayerNorm, Linear, Module, MultiHeadAttention};

    // GPT-2 small reduced to fit a single bench iter in <100 ms.
    // Real GPT-2 small = 12 layers × {LN, MHA(D=768,H=12), LN, FFN(F=3072)} +
    //                    Embedding(50257, 768) + final LN + LM head(768→50257).
    let batch = 1_usize;
    let seq = 128_usize; // shorter than 512 for tractable iter time
    let d_model = 768_usize;
    let n_heads = 12_usize;
    let d_ff = 3072_usize;
    let num_layers = 12_usize;
    let vocab_size = 50257_usize;

    // Build 12 layers' worth of modules.
    let token_emb = Embedding::with_seed(vocab_size, d_model, 0xC1A4);
    let pos_emb = Embedding::with_seed(seq, d_model, 0xC1A5);

    let mut layers: Vec<(LayerNorm, MultiHeadAttention, LayerNorm, Linear, Linear)> =
        Vec::with_capacity(num_layers);
    for _ in 0..num_layers {
        layers.push((
            LayerNorm::new(d_model),
            MultiHeadAttention::new(d_model, n_heads),
            LayerNorm::new(d_model),
            Linear::new(d_model, d_ff),
            Linear::new(d_ff, d_model),
        ));
    }
    let final_ln = LayerNorm::new(d_model);
    let lm_head = Linear::new(d_model, vocab_size);

    // Pre-build deterministic input token ids.
    let mut ids: Vec<i64> = Vec::with_capacity(seq);
    let mut s: u64 = 1;
    for _ in 0..seq {
        s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        ids.push(((s >> 32) as i64).rem_euclid(vocab_size as i64));
    }
    let ids_t = Tensor::from_vec_typed::<i64, _>([batch, seq], ids).unwrap();

    let pos_ids: Vec<i64> = (0..seq as i64).collect();
    let pos_t = Tensor::from_vec_typed::<i64, _>([seq], pos_ids).unwrap();

    let mut group = c.benchmark_group("gpt2_full_stack_forward");
    group.sample_size(10);
    group.warm_up_time(std::time::Duration::from_millis(800));
    group.measurement_time(std::time::Duration::from_secs(8));
    group.bench_function(
        BenchmarkId::from_parameter(format!(
            "L{}B{}S{}D{}H{}F{}V{}",
            num_layers, batch, seq, d_model, n_heads, d_ff, vocab_size
        )),
        |bb| {
            bb.iter(|| {
                no_grad(|| {
                    // 1. Token embedding lookup. forward_indices wants
                    //    a 1-D index tensor so we squeeze the batch dim.
                    let flat_ids = Tensor::from_vec_typed::<i64, _>(
                        [batch * seq],
                        ids_t.as_slice::<i64>().unwrap().to_vec(),
                    )
                    .unwrap();
                    let tok_emb = token_emb.forward_indices(&flat_ids).unwrap();
                    let pos = pos_emb.forward_indices(&pos_t).unwrap();
                    // tok_emb is [batch*seq, D]; pos is [seq, D] —
                    // for batch=1 they broadcast cleanly.
                    let mut x = ops::add(&tok_emb, &pos).unwrap();
                    // Reshape to [B, S, D] for MHA.
                    x = ops::reshape(&x, vec![batch, seq, d_model]).unwrap();

                    // 2. 12 transformer blocks.
                    for (ln1, mha, ln2, fc1, fc2) in layers.iter() {
                        let h = ln1.forward(&x).unwrap();
                        let h = mha.forward(&h, &h, &h, None).unwrap();
                        x = ops::add(&x, &h).unwrap();
                        let h = ln2.forward(&x).unwrap();
                        let h = fc1.forward_with_activation(&h, Activation::Relu).unwrap();
                        let h = fc2.forward(&h).unwrap();
                        x = ops::add(&x, &h).unwrap();
                    }

                    // 3. Final layer norm + LM head -> logits over
                    //    vocabulary.
                    let h = final_ln.forward(&x).unwrap();
                    let logits = lm_head.forward(&h).unwrap();
                    hint_black_box(logits)
                })
            });
        },
    );
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

/// T25 — coverage expansion: bench every primary transformer op
/// in isolation. Each op is paired with the PyTorch equivalent in
/// pytorch_bench.py so we can spot behind-PT shapes that the
/// composite block bench may hide.
fn bench_transformer_ops(c: &mut Criterion) {
    use rustorch_autograd::{no_grad, ops, Variable};
    use rustorch_nn::{Embedding, LayerNorm, Linear, Module, RMSNorm};

    let mut group = c.benchmark_group("transformer_ops");
    group.sample_size(20);
    group.warm_up_time(std::time::Duration::from_millis(500));
    group.measurement_time(std::time::Duration::from_secs(3));

    // LayerNorm @ GPT-2 small shape: [1, 512, 768]
    {
        let ln = LayerNorm::new(768);
        let x_data = det(0xC1A0, 1 * 512 * 768);
        let x_t = Tensor::from_vec(vec![1usize, 512, 768], x_data).unwrap();
        group.bench_function("layernorm_B1S512D768", |bb| {
            bb.iter(|| {
                no_grad(|| {
                    let x = Variable::new(x_t.clone());
                    hint_black_box(ln.forward(&x).unwrap())
                })
            });
        });
    }

    // RMSNorm @ Llama-7B shape: [1, 1024, 4096]
    {
        let rms = RMSNorm::new(4096);
        let x_data = det(0xC1A1, 1 * 1024 * 4096);
        let x_t = Tensor::from_vec(vec![1usize, 1024, 4096], x_data).unwrap();
        group.bench_function("rmsnorm_B1S1024D4096", |bb| {
            bb.iter(|| {
                no_grad(|| {
                    let x = Variable::new(x_t.clone());
                    hint_black_box(rms.forward(&x).unwrap())
                })
            });
        });
    }

    // Linear @ GPT-2 small projection shape: [512, 768] -> [512, 768]
    {
        let fc = Linear::new(768, 768);
        let x_data = det(0xC1A2, 512 * 768);
        let x_t = Tensor::from_vec(vec![512usize, 768], x_data).unwrap();
        group.bench_function("linear_512x768x768", |bb| {
            bb.iter(|| {
                no_grad(|| {
                    let x = Variable::new(x_t.clone());
                    hint_black_box(fc.forward(&x).unwrap())
                })
            });
        });
    }

    // ReLU activation @ FFN intermediate shape: [1, 512, 3072]
    {
        let x_data = det(0xC1A3, 1 * 512 * 3072);
        let x_t = Tensor::from_vec(vec![1usize, 512, 3072], x_data).unwrap();
        group.bench_function("relu_B1S512F3072", |bb| {
            bb.iter(|| {
                no_grad(|| {
                    let x = Variable::new(x_t.clone());
                    hint_black_box(ops::relu(&x).unwrap())
                })
            });
        });
    }

    // Embedding lookup @ GPT-2 vocab: vocab=50257, embed=768, batch=128
    {
        let emb = Embedding::with_seed(50257, 768, 0xC1A4);
        let mut idx_data: Vec<i64> = Vec::with_capacity(128);
        let mut s: u64 = 1;
        for _ in 0..128 {
            s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            idx_data.push(((s >> 32) as i64).rem_euclid(50257));
        }
        let idx_t = Tensor::from_vec_typed::<i64, _>([128usize], idx_data).unwrap();
        group.bench_function("embedding_lookup_b128_v50k_d768", |bb| {
            bb.iter(|| no_grad(|| hint_black_box(emb.forward_indices(&idx_t).unwrap())));
        });
    }

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
        bench_transformer_ops,
        bench_gpt2_full_stack,
        bench_lm_head,
        bench_gpt2_single_token_decode,
        bench_single_token_breakdown,
        bench_linear_S1_micro,
        bench_rope,
        bench_sampling,
}
criterion_main!(benches);
