#!/usr/bin/env python3
"""PyTorch benchmark harness — emits the same JSON shape as the Rust harness.

Runs on CPU (and MPS for matmul comparison) on Apple Silicon. PyTorch on
macOS uses the Accelerate framework for BLAS via vDSP/cBLAS, so matmul
will be heavily SIMD/multithreaded.

We force single-thread first to put RusTorch (single-thread naive) on
fair footing, then run multi-threaded too so we see the full PyTorch
speed.
"""

import json
import statistics
import sys
import time

import torch
import torch.nn.functional as F

WARMUP = 3
ITERS = 20


def time_fn(fn):
    for _ in range(WARMUP):
        fn()
    samples = []
    for _ in range(ITERS):
        t0 = time.perf_counter_ns()
        fn()
        samples.append(time.perf_counter_ns() - t0)
    samples.sort()
    return samples[len(samples) // 2], samples[min(len(samples) * 99 // 100, len(samples) - 1)]


def make(seed, *shape):
    g = torch.Generator().manual_seed(seed)
    return torch.empty(shape).uniform_(-1.0, 1.0, generator=g)


def bench_matmul(m, k, n):
    a = make(0xA1, m, k)
    b = make(0xB2, k, n)
    med, p99 = time_fn(lambda: torch.matmul(a, b))
    flops = 2.0 * m * k * n
    return {
        "op": "matmul",
        "shape": f"[{m},{k}] x [{k},{n}]",
        "median_ns": med,
        "p99_ns": p99,
        "gflops": flops / med,
    }


def bench_softmax(rows, cols):
    x = make(0xCA, rows, cols)
    med, p99 = time_fn(lambda: F.softmax(x, dim=1))
    return {
        "op": "softmax",
        "shape": f"[{rows},{cols}]",
        "median_ns": med,
        "p99_ns": p99,
    }


def bench_attention_sdpa(b, h, n, d):
    """torch.nn.functional.scaled_dot_product_attention — uses Flash Attention v2 on supported HW.

    On CPU PyTorch dispatches to its mem-efficient backend (or naive).
    """
    q = make(0xCAFE, b, h, n, d)
    k = make(0xBEEF, b, h, n, d)
    v = make(0xF00D, b, h, n, d)
    med, p99 = time_fn(lambda: F.scaled_dot_product_attention(q, k, v))
    return {
        "op": "sdpa",
        "shape": f"B={b} H={h} N={n} D={d}",
        "median_ns": med,
        "p99_ns": p99,
    }


def bench_attention_naive(b, h, n, d):
    """Manual naive attention: (Q @ K^T / sqrt(d)).softmax @ V."""
    import math
    q = make(0xCAFE, b, h, n, d)
    k = make(0xBEEF, b, h, n, d)
    v = make(0xF00D, b, h, n, d)
    scale = 1.0 / math.sqrt(d)

    def step():
        s = torch.matmul(q, k.transpose(-2, -1)) * scale
        a = torch.softmax(s, dim=-1)
        return torch.matmul(a, v)

    med, p99 = time_fn(step)
    return {
        "op": "naive_attention_forward",
        "shape": f"B={b} H={h} N={n} D={d}",
        "median_ns": med,
        "p99_ns": p99,
    }


def bench_layernorm(batch, seq, d):
    ln = torch.nn.LayerNorm(d)
    for p in ln.parameters():
        p.requires_grad_(False)
    g = torch.Generator().manual_seed(0xC1A0)
    x = torch.empty(batch, seq, d).uniform_(-1.0, 1.0, generator=g)

    def step():
        with torch.no_grad():
            return ln(x)

    med, p99 = time_fn(step)
    return {"op": "layernorm", "shape": f"B={batch} S={seq} D={d}", "median_ns": med, "p99_ns": p99}


def bench_rmsnorm(batch, seq, d):
    # PyTorch >= 2.4 has nn.RMSNorm; otherwise emulate with the canonical
    # formula for parity.
    rms = getattr(torch.nn, "RMSNorm", None)
    if rms is not None:
        layer = rms(d)
        for p in layer.parameters():
            p.requires_grad_(False)
    else:
        layer = None
    g = torch.Generator().manual_seed(0xC1A1)
    x = torch.empty(batch, seq, d).uniform_(-1.0, 1.0, generator=g)
    if layer is not None:
        def step():
            with torch.no_grad():
                return layer(x)
    else:
        gamma = torch.ones(d)
        def step():
            with torch.no_grad():
                rms_x = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
                return x * rms_x * gamma
    med, p99 = time_fn(step)
    return {"op": "rmsnorm", "shape": f"B={batch} S={seq} D={d}", "median_ns": med, "p99_ns": p99}


def bench_linear(rows, in_dim, out_dim):
    fc = torch.nn.Linear(in_dim, out_dim)
    for p in fc.parameters():
        p.requires_grad_(False)
    g = torch.Generator().manual_seed(0xC1A2)
    x = torch.empty(rows, in_dim).uniform_(-1.0, 1.0, generator=g)

    def step():
        with torch.no_grad():
            return fc(x)

    med, p99 = time_fn(step)
    return {"op": "linear", "shape": f"[{rows},{in_dim}]->[{rows},{out_dim}]", "median_ns": med, "p99_ns": p99}


def bench_relu_isolated(batch, seq, dim):
    g = torch.Generator().manual_seed(0xC1A3)
    x = torch.empty(batch, seq, dim).uniform_(-1.0, 1.0, generator=g)

    def step():
        with torch.no_grad():
            return torch.relu(x)

    med, p99 = time_fn(step)
    return {"op": "relu", "shape": f"B={batch} S={seq} F={dim}", "median_ns": med, "p99_ns": p99}


def bench_embedding(num_embeds, embed_dim, batch):
    emb = torch.nn.Embedding(num_embeds, embed_dim)
    for p in emb.parameters():
        p.requires_grad_(False)
    g = torch.Generator().manual_seed(0xC1A4)
    idx = torch.randint(0, num_embeds, (batch,), generator=g)

    def step():
        with torch.no_grad():
            return emb(idx)

    med, p99 = time_fn(step)
    return {"op": "embedding", "shape": f"vocab={num_embeds} dim={embed_dim} b={batch}", "median_ns": med, "p99_ns": p99}


def bench_gqa(batch, n_heads, n_kv_heads, seq_q, seq_kv, head_dim):
    """Grouped Query Attention. Q has `n_heads` heads but K/V share
    `n_kv_heads` (Qwen / Llama 3 architecture). PyTorch's
    nn.MultiheadAttention doesn't expose GQA directly, so we
    implement the canonical ATen pattern manually:
        repeat K, V `group_size` times along the head dim, then run
        scaled_dot_product_attention.
    """
    g = torch.Generator().manual_seed(0xC0DE)
    q = torch.empty(batch, n_heads, seq_q, head_dim).uniform_(-1.0, 1.0, generator=g)
    k = torch.empty(batch, n_kv_heads, seq_kv, head_dim).uniform_(-1.0, 1.0, generator=g)
    v = torch.empty(batch, n_kv_heads, seq_kv, head_dim).uniform_(-1.0, 1.0, generator=g)
    group_size = n_heads // n_kv_heads

    def step():
        with torch.no_grad():
            # Standard GQA: repeat K, V to match query head count.
            k_rep = k.repeat_interleave(group_size, dim=1)
            v_rep = v.repeat_interleave(group_size, dim=1)
            return torch.nn.functional.scaled_dot_product_attention(q, k_rep, v_rep)

    med, p99 = time_fn(step)
    return {
        "op": "gqa_forward",
        "shape": f"B={batch} H={n_heads} KV={n_kv_heads} S={seq_q} D={head_dim}",
        "median_ns": med,
        "p99_ns": p99,
    }


def bench_rope(batch, n_heads, seq, head_dim):
    """RoPE — Rotary Position Embeddings. Applied to Q and K every
    transformer layer of every LLM (Llama / Qwen / Mistral / Phi).
    """
    x = torch.empty(batch, n_heads, seq, head_dim).normal_()
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2).float() / head_dim))
    pos = torch.arange(seq).float()
    sinusoid = torch.einsum("i,j->ij", pos, inv_freq)  # [seq, hd/2]
    cos = sinusoid.cos()  # [seq, hd/2]
    sin = sinusoid.sin()

    def step():
        with torch.no_grad():
            # Standard RoPE: rotate (2k, 2k+1) pair
            x_even = x[..., 0::2]
            x_odd = x[..., 1::2]
            rotated_even = x_even * cos - x_odd * sin
            rotated_odd = x_even * sin + x_odd * cos
            # Interleave the rotated pairs back into [..., head_dim]
            stacked = torch.stack([rotated_even, rotated_odd], dim=-1)
            return stacked.flatten(-2)

    med, p99 = time_fn(step)
    return {
        "op": "rope_apply",
        "shape": f"B={batch} H={n_heads} S={seq} D={head_dim}",
        "median_ns": med,
        "p99_ns": p99,
    }


def bench_sampling(vocab_size, top_k, top_p):
    """Token sampling: temperature + top_k + top_p. One call per
    decoded token, so the per-call cost matters."""
    logits = torch.empty(vocab_size).uniform_(-1.0, 1.0)

    def step():
        with torch.no_grad():
            # PyTorch standard sampling pipeline.
            x = logits.clone()
            # top-k
            if top_k is not None and top_k < vocab_size:
                topk_vals, _ = torch.topk(x, top_k)
                kth = topk_vals[-1]
                x[x < kth] = float("-inf")
            # temperature 1, softmax
            probs = torch.softmax(x, dim=-1)
            # top-p
            if top_p is not None and top_p < 1.0:
                sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                cum = sorted_probs.cumsum(dim=-1)
                cutoff = (cum < top_p).sum().item() + 1
                mask = torch.zeros_like(probs)
                mask[sorted_idx[:cutoff]] = 1
                probs = probs * mask
                probs = probs / probs.sum()
            # Sample.
            return torch.multinomial(probs, 1)

    med, p99 = time_fn(step)
    return {
        "op": "sampling",
        "shape": f"vocab={vocab_size} top_k={top_k} top_p={top_p}",
        "median_ns": med,
        "p99_ns": p99,
    }


def bench_lm_head(rows, d_model, vocab_size):
    """LM head matmul — Linear(d_model -> vocab_size) on `rows` token rows.

    Models the final logit projection in any LLM forward; ~10 G FLOPs
    on 128×768×50257 dominates the per-iter cost.
    """
    fc = torch.nn.Linear(d_model, vocab_size)
    for p in fc.parameters():
        p.requires_grad_(False)
    g = torch.Generator().manual_seed(0xC1A6)
    x = torch.empty(rows, d_model).uniform_(-1.0, 1.0, generator=g)

    def step():
        with torch.no_grad():
            return fc(x)

    med, p99 = time_fn(step)
    return {"op": "lm_head", "shape": f"[{rows},{d_model}]->[{rows},{vocab_size}]", "median_ns": med, "p99_ns": p99}


def bench_gpt2_single_token_decode(num_layers, d_model, n_heads, d_ff, vocab_size):
    """Single-token decode latency — the per-token cost of an
    autoregressive generation step (without KV-cache, which neither
    framework's basic nn.MultiheadAttention exposes natively here).
    """
    seq = 1
    batch = 1
    tok_emb = torch.nn.Embedding(vocab_size, d_model)
    pos_emb = torch.nn.Embedding(max(seq, 1), d_model)
    layers = []
    for _ in range(num_layers):
        layers.append(
            (
                torch.nn.LayerNorm(d_model),
                torch.nn.MultiheadAttention(d_model, n_heads, batch_first=True),
                torch.nn.LayerNorm(d_model),
                torch.nn.Linear(d_model, d_ff),
                torch.nn.Linear(d_ff, d_model),
            )
        )
    final_ln = torch.nn.LayerNorm(d_model)
    lm_head = torch.nn.Linear(d_model, vocab_size)
    for m in [tok_emb, pos_emb, final_ln, lm_head]:
        for p in m.parameters():
            p.requires_grad_(False)
    for tup in layers:
        for sub in tup:
            for p in sub.parameters():
                p.requires_grad_(False)

    ids = torch.tensor([[42]], dtype=torch.long)
    pos_ids = torch.zeros(1, dtype=torch.long)

    def step():
        with torch.no_grad():
            x = tok_emb(ids) + pos_emb(pos_ids)
            for ln1, mha, ln2, fc1, fc2 in layers:
                h = ln1(x)
                h, _ = mha(h, h, h, need_weights=False)
                x = x + h
                h = ln2(x)
                h = fc1(h)
                h = torch.relu(h)
                h = fc2(h)
                x = x + h
            h = final_ln(x)
            return lm_head(h)

    med, p99 = time_fn(step)
    return {
        "op": "gpt2_single_token_decode",
        "shape": f"L={num_layers} S=1 D={d_model} V={vocab_size}",
        "median_ns": med,
        "p99_ns": p99,
    }


def bench_gpt2_full_stack(num_layers, batch, seq, d_model, n_heads, d_ff, vocab_size):
    """Full GPT-2-small forward stack — embedding + N blocks + LN + LM head.

    Shape parity with bench_gpt2_full_stack on the Rust side. Uses
    torch.nn modules directly (LayerNorm, MultiheadAttention, Linear,
    Embedding) under torch.no_grad. ReLU instead of GELU for parity.
    """
    tok_emb = torch.nn.Embedding(vocab_size, d_model)
    pos_emb = torch.nn.Embedding(seq, d_model)
    layers = []
    for _ in range(num_layers):
        layers.append(
            (
                torch.nn.LayerNorm(d_model),
                torch.nn.MultiheadAttention(d_model, n_heads, batch_first=True),
                torch.nn.LayerNorm(d_model),
                torch.nn.Linear(d_model, d_ff),
                torch.nn.Linear(d_ff, d_model),
            )
        )
    final_ln = torch.nn.LayerNorm(d_model)
    lm_head = torch.nn.Linear(d_model, vocab_size)
    for m in [tok_emb, pos_emb, final_ln, lm_head]:
        for p in m.parameters():
            p.requires_grad_(False)
    for tup in layers:
        for sub in tup:
            for p in sub.parameters():
                p.requires_grad_(False)

    g = torch.Generator().manual_seed(0xC1A4)
    ids = torch.randint(0, vocab_size, (batch, seq), generator=g)
    pos_ids = torch.arange(seq)

    def step():
        with torch.no_grad():
            x = tok_emb(ids) + pos_emb(pos_ids)
            for ln1, mha, ln2, fc1, fc2 in layers:
                h = ln1(x)
                h, _ = mha(h, h, h, need_weights=False)
                x = x + h
                h = ln2(x)
                h = fc1(h)
                h = torch.relu(h)
                h = fc2(h)
                x = x + h
            h = final_ln(x)
            return lm_head(h)

    med, p99 = time_fn(step)
    return {
        "op": "gpt2_full_stack_forward",
        "shape": f"L={num_layers} B={batch} S={seq} D={d_model} H={n_heads} F={d_ff} V={vocab_size}",
        "median_ns": med,
        "p99_ns": p99,
    }


def bench_transformer_block(batch, seq, d_model, n_heads, d_ff):
    """GPT-2-style transformer block forward (no_grad inference).

    Layout matches the Rust bench `bench_transformer_block`:
        x = ln1(x)
        x = x + mha(x, x, x)         # self-attention
        x = ln2(x)
        x = x + fc2(relu(fc1(x)))    # FFN

    ReLU is used (not GELU) for parity with the RusTorch bench until
    `ops::gelu` lands in autograd.
    """
    ln1 = torch.nn.LayerNorm(d_model)
    # MHA: PyTorch's MultiheadAttention defaults to (seq, batch, dim) ordering;
    # we use batch_first=True to match the RusTorch [batch, seq, dim] layout.
    mha = torch.nn.MultiheadAttention(d_model, n_heads, batch_first=True)
    ln2 = torch.nn.LayerNorm(d_model)
    fc1 = torch.nn.Linear(d_model, d_ff)
    fc2 = torch.nn.Linear(d_ff, d_model)
    # Switch every parameter to requires_grad=False so PyTorch
    # short-circuits autograd construction (matches Rust no_grad).
    for m in [ln1, mha, ln2, fc1, fc2]:
        for p in m.parameters():
            p.requires_grad_(False)
    g = torch.Generator().manual_seed(0xC1A0)
    x = torch.empty(batch, seq, d_model).uniform_(-1.0, 1.0, generator=g)

    def step():
        with torch.no_grad():
            h = ln1(x)
            h, _ = mha(h, h, h, need_weights=False)
            y = x + h
            h = ln2(y)
            h = fc1(h)
            h = torch.relu(h)
            h = fc2(h)
            out = y + h
            return out

    med, p99 = time_fn(step)
    return {
        "op": "transformer_block_forward",
        "shape": f"B={batch} S={seq} D={d_model} H={n_heads} F={d_ff}",
        "median_ns": med,
        "p99_ns": p99,
    }


def bench_elementwise_add(n):
    a = make(0xA1, n)
    b = make(0xB2, n)
    med, p99 = time_fn(lambda: a.add_(b))
    return {
        "op": "elementwise_add",
        "shape": f"[{n}]",
        "median_ns": med,
        "p99_ns": p99,
    }


def run(num_threads, device_label="cpu"):
    torch.set_num_threads(num_threads)
    results = []
    for shape in [(64, 64, 64), (128, 128, 128), (256, 256, 256), (512, 512, 512), (1024, 1024, 1024)]:
        results.append(bench_matmul(*shape))
    results.append(bench_softmax(128, 32_000))
    results.append(bench_softmax(512, 50_257))
    for shape in [(1, 1, 512, 64), (1, 4, 1024, 64), (2, 4, 2048, 64)]:
        results.append(bench_attention_sdpa(*shape))
        results.append(bench_attention_naive(*shape))
    for n in [1_000, 10_000, 100_000, 1_000_000, 10_000_000]:
        results.append(bench_elementwise_add(n))
    # End-to-end transformer block forward (T17): the metric that
    # decides whether the perf sprint translates to faster transformer
    # inference vs PyTorch.
    results.append(bench_transformer_block(2, 128, 256, 4, 1024))
    # T21 — full-scale GPT-2 small block (real production shape).
    results.append(bench_transformer_block(1, 512, 768, 12, 3072))
    # T25 — coverage expansion: isolated transformer ops.
    results.append(bench_layernorm(1, 512, 768))
    results.append(bench_rmsnorm(1, 1024, 4096))
    results.append(bench_linear(512, 768, 768))
    results.append(bench_relu_isolated(1, 512, 3072))
    results.append(bench_embedding(50257, 768, 128))
    # T30 — GPT-2-small full stack (12 layers, embedding + final LN + LM head).
    results.append(bench_gpt2_full_stack(12, 1, 128, 768, 12, 3072, 50257))
    # T32 — LM head isolated (the dominant final matmul) and
    # single-token decode (per-token cost of LLM serving).
    results.append(bench_lm_head(128, 768, 50257))
    results.append(bench_gpt2_single_token_decode(12, 768, 12, 3072, 50257))
    return {
        "framework": "pytorch",
        "device": device_label,
        "num_threads": num_threads,
        "version": torch.__version__,
        "warmup_iters": WARMUP,
        "timed_iters": ITERS,
        "results": results,
    }


if __name__ == "__main__":
    threads = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    print(json.dumps(run(threads), indent=2))
