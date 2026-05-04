//! `qwen_inference_metal` — end-to-end Qwen3 decode using our
//! T71/T72 Metal Q4_K / Q6_K matmul kernels for the heavy
//! projections, plus the existing CPU rustorch-nn primitives for
//! everything else (RMSNorm, RoPE, GQA, SwiGLU). The point is to
//! measure end-to-end tok/s with our **own** kernels (no candle,
//! no llama.cpp, no ggml) on the way to beating MLX.
//!
//! Status: MVP. Each Metal sgemv is dispatched + drained synchronously
//! per call to keep the CPU-side scratch logic simple. A follow-up
//! commit will batch the GPU commands across a layer and skip the
//! drain so we don't pay 6 sync points per layer × 40 layers.
//!
//! Usage:
//!   cargo run --release -p rustorch-llm --example qwen_inference_metal -- \
//!       --model ~/models/Qwen3-14B-Claude-4.5-Opus-Distill.q4_k_m.gguf \
//!       --prompt-ids 12522,5193,264,882,11,1052,572,264,2613,25105,879 \
//!       --n 50

#![cfg(target_os = "macos")]
#![allow(dead_code)] // some scratch buffers are kept around for the
                     // CPU fallback path that's swapped in/out across
                     // commits as we port more ops to Metal kernels.

use std::env;
use std::process::ExitCode;
use std::time::Instant;

use rustorch_gguf::{GgmlType, GgufFile};
use rustorch_metal::backend::MetalBackend;
use rustorch_metal::backend_singleton::metal_backend;
use rustorch_metal::kernels::{
    add_inplace_f32, rms_norm_f32, sgemv_q4_k_f32_into, sgemv_q6_k_f32_into, swiglu_f32,
};

use metal::Buffer;

use rustorch_nn::gqa::gqa_forward_f32;
use rustorch_nn::kv_cache::KVCache;
use rustorch_nn::rope::RoPE;

/// One projection's worth of weights, kept GPU-resident.
struct MetalWeight {
    buffer: Buffer,
    k: usize,
    n: usize,
    dtype: GgmlType,
}

impl MetalWeight {
    fn matmul_into(&self, backend: &MetalBackend, x_buf: &Buffer, out_buf: &Buffer) {
        match self.dtype {
            GgmlType::Q4_K => {
                sgemv_q4_k_f32_into(backend, x_buf, &self.buffer, out_buf, self.k, self.n).unwrap()
            },
            GgmlType::Q6_K => {
                sgemv_q6_k_f32_into(backend, x_buf, &self.buffer, out_buf, self.k, self.n).unwrap()
            },
            _ => panic!("unsupported dtype: {:?}", self.dtype),
        }
    }
}

struct LayerWeights {
    attn_norm: Vec<f32>,   // [d] CPU copy (for QK norm fallback)
    attn_norm_buf: Buffer, // GPU copy
    w_q: MetalWeight,
    w_k: MetalWeight,
    w_v: MetalWeight,
    w_o: MetalWeight,
    ffn_norm: Vec<f32>,
    ffn_norm_buf: Buffer,
    w_gate: MetalWeight,
    w_up: MetalWeight,
    w_down: MetalWeight,
    attn_q_norm: Option<Vec<f32>>,
    attn_k_norm: Option<Vec<f32>>,
}

struct ModelMetal {
    cfg: ModelCfg,
    layers: Vec<LayerWeights>,
    token_emb: Vec<f32>, // [V, D] — kept f32 for cheap lookup
    final_norm: Vec<f32>,
    final_norm_buf: Buffer,
    lm_head: MetalWeight, // Q6_K usually
    rope: RoPE,
}

#[derive(Clone, Copy)]
struct ModelCfg {
    d: usize,
    f: usize,
    n_layers: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    vocab: usize,
    rms_eps: f32,
    rope_theta: f32,
    max_seq: usize,
}

fn alloc_metal_from_bytes(backend: &MetalBackend, bytes: &[u8]) -> Buffer {
    let buf = backend.alloc_shared(bytes.len()).unwrap();
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), buf.contents() as *mut u8, bytes.len());
    }
    buf
}

fn load_f32(file: &GgufFile, name: &str) -> Vec<f32> {
    let t = file
        .tensor(name)
        .unwrap_or_else(|| panic!("missing tensor: {name}"));
    let bytes = file.tensor_bytes(t);
    rustorch_gguf::dequant_to_f32(t, bytes).unwrap()
}

fn load_metal_weight(backend: &MetalBackend, file: &GgufFile, name: &str) -> MetalWeight {
    let t = file
        .tensor(name)
        .unwrap_or_else(|| panic!("missing tensor: {name}"));
    let bytes = file.tensor_bytes(t);
    // GGUF shape ne=[K, N] = numpy [N, K] row-major.
    // Q4_K / Q6_K bytes are already laid out as N rows of K weights
    // contiguously, which matches our kernel's expected layout.
    let k = t.shape[0] as usize;
    let n = t.shape[1] as usize;
    MetalWeight {
        buffer: alloc_metal_from_bytes(backend, bytes),
        k,
        n,
        dtype: t.dtype,
    }
}

fn rms_norm(x: &mut [f32], gamma: &[f32], eps: f32) {
    let d = x.len();
    let inv_d = 1.0_f32 / d as f32;
    let sq = x.iter().map(|v| v * v).sum::<f32>();
    let inv_rms = 1.0 / (sq * inv_d + eps).sqrt();
    for i in 0..d {
        x[i] = x[i] * inv_rms * gamma[i];
    }
}

fn rms_norm_per_head(x: &mut [f32], gamma: &[f32], n_heads: usize, head_dim: usize, eps: f32) {
    for h in 0..n_heads {
        let head = &mut x[h * head_dim..(h + 1) * head_dim];
        rms_norm(head, gamma, eps);
    }
}

fn argmax(logits: &[f32]) -> u32 {
    let (i, _) =
        logits
            .iter()
            .enumerate()
            .fold((0usize, f32::NEG_INFINITY), |(bi, bv), (i, &v)| {
                if v > bv {
                    (i, v)
                } else {
                    (bi, bv)
                }
            });
    i as u32
}

fn forward_token(
    backend: &MetalBackend,
    model: &ModelMetal,
    token_id: u32,
    position: usize,
    cache: &mut KVCache,
    scratch: &mut Scratch,
) -> u32 {
    let cfg = model.cfg;
    let d = cfg.d;
    let f = cfg.f;
    let kv_dim = cfg.n_kv_heads * cfg.head_dim;
    let n_heads = cfg.n_heads;
    let n_kv = cfg.n_kv_heads;
    let head_dim = cfg.head_dim;

    // Embed (CPU lookup, written into x_buf as residual stream).
    let off = (token_id as usize) * d;
    scratch.x.copy_from_slice(&model.token_emb[off..off + d]);
    unsafe {
        std::ptr::copy_nonoverlapping(scratch.x.as_ptr(), scratch.xd_buf.contents() as *mut f32, d);
    }

    for (li, layer) in model.layers.iter().enumerate() {
        // 1. RMSNorm GPU (no sync — chained to next matmul).
        rms_norm_f32(
            backend,
            &scratch.xd_buf,
            &layer.attn_norm_buf,
            &scratch.h_buf,
            d,
            cfg.rms_eps,
        )
        .unwrap();

        // 2. Q / K / V via Metal (3 separate sgemv).
        layer
            .w_q
            .matmul_into(backend, &scratch.h_buf, &scratch.q_buf);
        layer
            .w_k
            .matmul_into(backend, &scratch.h_buf, &scratch.k_buf);
        layer
            .w_v
            .matmul_into(backend, &scratch.h_buf, &scratch.v_buf);
        // We need Q/K/V on CPU for RoPE+GQA, so drain here is unavoidable
        // until we port RoPE+GQA to Metal too.
        backend.drain();
        unsafe {
            std::ptr::copy_nonoverlapping(
                scratch.q_buf.contents() as *const f32,
                scratch.q.as_mut_ptr(),
                d,
            );
            std::ptr::copy_nonoverlapping(
                scratch.k_buf.contents() as *const f32,
                scratch.k.as_mut_ptr(),
                kv_dim,
            );
            std::ptr::copy_nonoverlapping(
                scratch.v_buf.contents() as *const f32,
                scratch.v.as_mut_ptr(),
                kv_dim,
            );
        }

        // 3. QK-norm (CPU, optional).
        if let Some(qn) = layer.attn_q_norm.as_deref() {
            rms_norm_per_head(&mut scratch.q, qn, n_heads, head_dim, cfg.rms_eps);
        }
        if let Some(kn) = layer.attn_k_norm.as_deref() {
            rms_norm_per_head(&mut scratch.k, kn, n_kv, head_dim, cfg.rms_eps);
        }

        // 4. RoPE (CPU).
        model
            .rope
            .apply_inplace_half_split(&mut scratch.q, 1, n_heads, 1, position)
            .unwrap();
        model
            .rope
            .apply_inplace_half_split(&mut scratch.k, 1, n_kv, 1, position)
            .unwrap();

        // 5. KV cache append + GQA (CPU).
        cache.append(li, 1, &scratch.k, &scratch.v).unwrap();
        let kv_len = position + 1;
        let max_seq = cache.max_seq();
        let k_full = cache.k_buffer(li).unwrap();
        let v_full = cache.v_buffer(li).unwrap();
        let trim_len = n_kv * kv_len * head_dim;
        for kvh in 0..n_kv {
            let src = kvh * max_seq * head_dim;
            let dst = kvh * kv_len * head_dim;
            scratch.k_trim[dst..dst + kv_len * head_dim]
                .copy_from_slice(&k_full[src..src + kv_len * head_dim]);
            scratch.v_trim[dst..dst + kv_len * head_dim]
                .copy_from_slice(&v_full[src..src + kv_len * head_dim]);
        }
        gqa_forward_f32(
            &scratch.q,
            &scratch.k_trim[..trim_len],
            &scratch.v_trim[..trim_len],
            &mut scratch.attn_out,
            1,
            n_heads,
            n_kv,
            1,
            kv_len,
            head_dim,
        )
        .unwrap();

        // 6+7+8+9: O proj → residual_add → RMSNorm → gate/up → SwiGLU →
        // down → residual_add. ALL GPU, chained without drain in between.
        // The residual stream lives in xd_buf; CPU only sees it again at
        // the next iteration's top of layer (which we don't actually need
        // — the next iteration's first op is RMSNorm which reads xd_buf).
        // Net: 1 drain per layer (after Q/K/V for RoPE/GQA) instead of 4.
        unsafe {
            std::ptr::copy_nonoverlapping(
                scratch.attn_out.as_ptr(),
                scratch.h_buf.contents() as *mut f32,
                d,
            );
        }
        layer
            .w_o
            .matmul_into(backend, &scratch.h_buf, &scratch.o_buf);
        add_inplace_f32(backend, &scratch.xd_buf, &scratch.o_buf, d).unwrap();
        rms_norm_f32(
            backend,
            &scratch.xd_buf,
            &layer.ffn_norm_buf,
            &scratch.h_buf,
            d,
            cfg.rms_eps,
        )
        .unwrap();
        layer
            .w_gate
            .matmul_into(backend, &scratch.h_buf, &scratch.gate_buf);
        layer
            .w_up
            .matmul_into(backend, &scratch.h_buf, &scratch.up_buf);
        swiglu_f32(
            backend,
            &scratch.gate_buf,
            &scratch.up_buf,
            &scratch.fd_buf,
            f,
        )
        .unwrap();
        layer
            .w_down
            .matmul_into(backend, &scratch.fd_buf, &scratch.fc2_buf);
        add_inplace_f32(backend, &scratch.xd_buf, &scratch.fc2_buf, d).unwrap();
    }

    // Final RMSNorm GPU + LM head GPU. Residual stream is in xd_buf;
    // we run RMSNorm into h_buf, then lm_head into logits_buf.
    rms_norm_f32(
        backend,
        &scratch.xd_buf,
        &model.final_norm_buf,
        &scratch.h_buf,
        cfg.d,
        cfg.rms_eps,
    )
    .unwrap();
    model
        .lm_head
        .matmul_into(backend, &scratch.h_buf, &scratch.logits_buf);
    backend.drain();
    let mut logits = vec![0.0_f32; cfg.vocab];
    unsafe {
        std::ptr::copy_nonoverlapping(
            scratch.logits_buf.contents() as *const f32,
            logits.as_mut_ptr(),
            cfg.vocab,
        );
    }
    argmax(&logits)
}

struct Scratch {
    x: Vec<f32>,
    h: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    attn_out: Vec<f32>,
    o_out: Vec<f32>,
    gate_out: Vec<f32>,
    up_out: Vec<f32>,
    fc2_out: Vec<f32>,
    k_trim: Vec<f32>,
    v_trim: Vec<f32>,
    // Persistent GPU buffers — pre-allocated once, reused every layer
    // every token. Eliminates the per-call MTLBuffer alloc cost
    // (~10-50µs each) which was 6 allocs × 40 layers = 240/token.
    xd_buf: Buffer,     // d-sized residual stream (lives across all layers)
    h_buf: Buffer,      // d-sized norm output / matmul input
    fd_buf: Buffer,     // f-sized input (swiglu output → down)
    q_buf: Buffer,      // d-sized Q output
    k_buf: Buffer,      // kv_dim-sized K output
    v_buf: Buffer,      // kv_dim-sized V output
    o_buf: Buffer,      // d-sized O proj output
    gate_buf: Buffer,   // f-sized gate output
    up_buf: Buffer,     // f-sized up output
    fc2_buf: Buffer,    // d-sized down output
    logits_buf: Buffer, // vocab-sized lm_head output
}

impl Scratch {
    fn new(backend: &MetalBackend, cfg: &ModelCfg) -> Self {
        let d = cfg.d;
        let kv_dim = cfg.n_kv_heads * cfg.head_dim;
        let f = cfg.f;
        let max_kv = cfg.n_kv_heads * cfg.max_seq * cfg.head_dim;
        Scratch {
            x: vec![0.0; d],
            h: vec![0.0; d],
            q: vec![0.0; d],
            k: vec![0.0; kv_dim],
            v: vec![0.0; kv_dim],
            attn_out: vec![0.0; d],
            o_out: vec![0.0; d],
            gate_out: vec![0.0; f],
            up_out: vec![0.0; f],
            fc2_out: vec![0.0; d],
            k_trim: vec![0.0; max_kv],
            v_trim: vec![0.0; max_kv],
            xd_buf: backend.alloc_shared(d * 4).unwrap(),
            h_buf: backend.alloc_shared(d * 4).unwrap(),
            fd_buf: backend.alloc_shared(f * 4).unwrap(),
            q_buf: backend.alloc_shared(d * 4).unwrap(),
            k_buf: backend.alloc_shared(kv_dim * 4).unwrap(),
            v_buf: backend.alloc_shared(kv_dim * 4).unwrap(),
            o_buf: backend.alloc_shared(d * 4).unwrap(),
            gate_buf: backend.alloc_shared(f * 4).unwrap(),
            up_buf: backend.alloc_shared(f * 4).unwrap(),
            fc2_buf: backend.alloc_shared(d * 4).unwrap(),
            logits_buf: backend.alloc_shared(cfg.vocab * 4).unwrap(),
        }
    }
}

fn load_model(backend: &MetalBackend, path: &str, max_seq: usize) -> ModelMetal {
    println!("→ opening {path}");
    let file = GgufFile::open(path).expect("open gguf");
    let arch = file
        .metadata()
        .get("general.architecture")
        .and_then(|v| v.as_str())
        .unwrap_or("?");
    let p = |k: &str| {
        file.metadata()
            .get(&format!("{arch}.{k}"))
            .and_then(|v| v.as_u32())
            .unwrap() as usize
    };
    let pf = |k: &str| {
        file.metadata()
            .get(&format!("{arch}.{k}"))
            .and_then(|v| v.as_f32())
            .unwrap()
    };
    let cfg = ModelCfg {
        d: p("embedding_length"),
        f: p("feed_forward_length"),
        n_layers: p("block_count"),
        n_heads: p("attention.head_count"),
        n_kv_heads: p("attention.head_count_kv"),
        head_dim: p("attention.key_length"),
        vocab: file.tensor("token_embd.weight").unwrap().shape[1] as usize,
        rms_eps: pf("attention.layer_norm_rms_epsilon"),
        rope_theta: pf("rope.freq_base"),
        max_seq,
    };
    println!(
        "  arch={arch} d={} f={} layers={} heads={}/{} head_dim={} vocab={}",
        cfg.d, cfg.f, cfg.n_layers, cfg.n_heads, cfg.n_kv_heads, cfg.head_dim, cfg.vocab
    );

    let token_emb = load_f32(&file, "token_embd.weight");
    let final_norm = load_f32(&file, "output_norm.weight");
    let final_norm_buf = alloc_metal_from_bytes(backend, bytemuck_cast(&final_norm));
    let lm_head_name = if file.tensor("output.weight").is_some() {
        "output.weight"
    } else {
        "token_embd.weight" // tied
    };
    let lm_head = load_metal_weight(backend, &file, lm_head_name);

    let t_load = Instant::now();
    let mut layers = Vec::with_capacity(cfg.n_layers);
    for i in 0..cfg.n_layers {
        let attn_norm = load_f32(&file, &format!("blk.{i}.attn_norm.weight"));
        let ffn_norm = load_f32(&file, &format!("blk.{i}.ffn_norm.weight"));
        let attn_norm_buf = alloc_metal_from_bytes(backend, bytemuck_cast(&attn_norm));
        let ffn_norm_buf = alloc_metal_from_bytes(backend, bytemuck_cast(&ffn_norm));
        let layer = LayerWeights {
            attn_norm,
            attn_norm_buf,
            w_q: load_metal_weight(backend, &file, &format!("blk.{i}.attn_q.weight")),
            w_k: load_metal_weight(backend, &file, &format!("blk.{i}.attn_k.weight")),
            w_v: load_metal_weight(backend, &file, &format!("blk.{i}.attn_v.weight")),
            w_o: load_metal_weight(backend, &file, &format!("blk.{i}.attn_output.weight")),
            ffn_norm,
            ffn_norm_buf,
            w_gate: load_metal_weight(backend, &file, &format!("blk.{i}.ffn_gate.weight")),
            w_up: load_metal_weight(backend, &file, &format!("blk.{i}.ffn_up.weight")),
            w_down: load_metal_weight(backend, &file, &format!("blk.{i}.ffn_down.weight")),
            attn_q_norm: file
                .tensor(&format!("blk.{i}.attn_q_norm.weight"))
                .map(|t| rustorch_gguf::dequant_to_f32(t, file.tensor_bytes(t)).unwrap()),
            attn_k_norm: file
                .tensor(&format!("blk.{i}.attn_k_norm.weight"))
                .map(|t| rustorch_gguf::dequant_to_f32(t, file.tensor_bytes(t)).unwrap()),
        };
        layers.push(layer);
    }
    println!(
        "  loaded {} layers into Metal buffers in {:.2}s",
        cfg.n_layers,
        t_load.elapsed().as_secs_f32()
    );

    let rope = RoPE::new(cfg.head_dim, cfg.max_seq, cfg.rope_theta);
    ModelMetal {
        cfg,
        layers,
        token_emb,
        final_norm,
        final_norm_buf,
        lm_head,
        rope,
    }
}

/// Reinterpret a `&[f32]` as a `&[u8]` for upload into a shared MTLBuffer.
fn bytemuck_cast(v: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

fn main() -> ExitCode {
    let mut args: Vec<String> = env::args().skip(1).collect();
    let mut model_path: Option<String> = None;
    let mut prompt_ids: Vec<u32> = vec![1];
    let mut n: usize = 50;
    let mut max_seq: usize = 256;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--model" => {
                model_path = Some(args[i + 1].clone());
                args.remove(i + 1);
                args.remove(i);
                continue;
            },
            "--prompt-ids" => {
                prompt_ids = args[i + 1]
                    .split(',')
                    .filter_map(|s| s.trim().parse::<u32>().ok())
                    .collect();
                args.remove(i + 1);
                args.remove(i);
                continue;
            },
            "--n" => {
                n = args[i + 1].parse().unwrap_or(n);
                args.remove(i + 1);
                args.remove(i);
                continue;
            },
            "--max-seq" => {
                max_seq = args[i + 1].parse().unwrap_or(max_seq);
                args.remove(i + 1);
                args.remove(i);
                continue;
            },
            _ => {},
        }
        i += 1;
    }
    let path = model_path.expect("--model required");
    let _ = SamplingPlaceholder; // keep workspace happy
    let _ = LlamaConfigUnused;

    let backend = metal_backend();
    println!(
        "device: {} (Metal3: {})",
        backend.adapter_name(),
        backend.supports_metal3()
    );

    let model = load_model(backend, &path, max_seq);
    let mut cache = KVCache::new(
        model.cfg.n_layers,
        1,
        model.cfg.n_kv_heads,
        model.cfg.head_dim,
        max_seq,
    );
    let mut scratch = Scratch::new(backend, &model.cfg);

    println!("\n→ prefill {} tokens", prompt_ids.len());
    let t_pre = Instant::now();
    let mut last = 0u32;
    for (pos, &tok) in prompt_ids.iter().enumerate() {
        last = forward_token(backend, &model, tok, pos, &mut cache, &mut scratch);
        cache.advance(1).unwrap();
    }
    let prefill_d = t_pre.elapsed();
    println!(
        "  prefill: {} tokens in {:.3}s = {:.2} tok/s",
        prompt_ids.len(),
        prefill_d.as_secs_f64(),
        prompt_ids.len() as f64 / prefill_d.as_secs_f64()
    );

    let mut generated = vec![last];
    let t_dec = Instant::now();
    for _ in 1..n {
        let pos = cache.current_len();
        last = forward_token(backend, &model, last, pos, &mut cache, &mut scratch);
        cache.advance(1).unwrap();
        generated.push(last);
    }
    let decode_d = t_dec.elapsed();
    println!(
        "  decode : {} tokens in {:.3}s = {:.2} tok/s",
        n - 1,
        decode_d.as_secs_f64(),
        (n - 1) as f64 / decode_d.as_secs_f64()
    );
    println!("\ngenerated: {:?}", generated);
    ExitCode::SUCCESS
}

// Keep workspace deps happy.
#[allow(dead_code)]
struct SamplingPlaceholder;
#[allow(dead_code)]
struct LlamaConfigUnused;
