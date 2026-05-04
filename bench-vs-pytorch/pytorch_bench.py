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
