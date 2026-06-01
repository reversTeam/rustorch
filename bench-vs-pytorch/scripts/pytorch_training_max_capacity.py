#!/usr/bin/env python3
"""T246.12 MAX-CAP-TRAIN.2 — PyTorch CUDA training bench at MAX-CAPACITY.

Mirrors `crates/rustorch-llm/examples/training_bench_max_capacity_cuda.rs`
shape-for-shape so the rustorch vs PyTorch comparison is apples-to-apples.

Default workload (Option C, single-block, max shapes) :
    batch=32 seq=2048 hidden=4096 ffn=14336 layers=1 steps=20
Tokens / step = 65 536. ~46 TFLOP fwd GEMM, ~115 TFLOP fwd+bwd per step.

Reports the same RESULT_JSON line shape as the rustorch bench so a single
parser can ingest both.

Run :
    python3 pytorch_training_max_capacity.py [--compile] [--mode max-autotune]
"""

from __future__ import annotations

import argparse
import json
import statistics
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


class FFNBlock(nn.Module):
    """RMSNorm → gate proj → up proj → SwiGLU → down proj → residual.

    Identical math to the rustorch ``training_bench_max_capacity_cuda``
    bench (FFN only — no attention, since rustorch has no autograd-wired
    GQA backward at the time of writing).
    """

    def __init__(self, hidden: int, ffn: int):
        super().__init__()
        self.norm = nn.RMSNorm(hidden, eps=1e-6)
        self.w_gate = nn.Linear(hidden, ffn, bias=False)
        self.w_up = nn.Linear(hidden, ffn, bias=False)
        self.w_down = nn.Linear(ffn, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = self.norm(x)
        gate = self.w_gate(n)
        up = self.w_up(n)
        act = F.silu(gate) * up
        h = self.w_down(act)
        return h + x


class StackedFFN(nn.Module):
    def __init__(self, hidden: int, ffn: int, layers: int):
        super().__init__()
        self.blocks = nn.ModuleList([FFNBlock(hidden, ffn) for _ in range(layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x)
        return x


def init_det(t: torch.Tensor, seed: float) -> None:
    """Deterministic init mirroring the rustorch ``det_tensor`` helper."""
    n = t.numel()
    idx = torch.arange(1, n + 1, dtype=torch.float32, device=t.device)
    vals = ((idx * seed * 0.0001).sin() * 0.1).reshape(t.shape)
    with torch.no_grad():
        t.copy_(vals.to(t.dtype))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compile", action="store_true",
                        help="enable torch.compile (PyTorch best path)")
    parser.add_argument("--mode", choices=["default", "reduce-overhead", "max-autotune"],
                        default="max-autotune",
                        help="torch.compile mode (only if --compile)")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--seq", type=int, default=2048)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--ffn", type=int, default=14336)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    parser.add_argument("--warmup", type=int, default=3)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available — this bench requires a CUDA GPU.")

    device = torch.device("cuda:0")
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    tokens = args.batch * args.seq

    print("=== PyTorch CUDA MAX-CAPACITY training bench (T246.12 MAX-CAP-TRAIN.2) ===")
    print(
        f"shape : batch={args.batch} seq_len={args.seq} tokens/step={tokens} "
        f"hidden={args.hidden} ffn={args.ffn} layers={args.layers} steps={args.steps}"
    )
    print(f"dtype : {args.dtype} activations / matmul")
    print(f"compile : {'YES (' + args.mode + ')' if args.compile else 'NO (eager)'}")
    print(
        "device  : "
        f"{torch.cuda.get_device_name(0)} "
        f"sm={torch.cuda.get_device_capability(0)} "
        f"cuda={torch.version.cuda} torch={torch.__version__}"
    )
    print()

    # ---- Inputs / target / model (BF16 on-device) ----
    x = torch.empty(tokens, args.hidden, device=device, dtype=dtype)
    init_det(x, 1.0)
    target = torch.full(
        (tokens, args.hidden), 0.5, device=device, dtype=dtype, requires_grad=False
    )
    model = StackedFFN(args.hidden, args.ffn, args.layers).to(device=device, dtype=dtype)
    for li, blk in enumerate(model.blocks):
        s = li * 0.13 + 0.7
        init_det(blk.norm.weight, s)
        init_det(blk.w_gate.weight, s + 0.5)
        init_det(blk.w_up.weight, s + 0.3)
        init_det(blk.w_down.weight, s + 0.2)

    if args.compile:
        model = torch.compile(model, mode=args.mode)

    optim = torch.optim.AdamW(model.parameters(), lr=1e-4)

    def step_fn() -> torch.Tensor:
        optim.zero_grad(set_to_none=True)
        y = model(x)
        diff = y - target
        loss = (diff * diff).sum()
        loss.backward()
        optim.step()
        return loss

    # ---- Warm-up ----
    print(f"warm-up : {args.warmup} step(s)...")
    torch.cuda.synchronize()
    twarm = time.perf_counter()
    for _ in range(args.warmup):
        _ = step_fn()
    torch.cuda.synchronize()
    print(f"warm-up done : {time.perf_counter() - twarm:.3f}s")
    print()

    # ---- Measured loop (per-step timing for distribution) ----
    torch.cuda.reset_peak_memory_stats(device)
    step_times: list[float] = []
    last_loss = 0.0
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for s in range(args.steps):
        t_step = time.perf_counter()
        loss = step_fn()
        torch.cuda.synchronize()
        dt = time.perf_counter() - t_step
        step_times.append(dt)
        if s == 0 or s == args.steps - 1 or s % 5 == 0:
            last_loss = loss.item()
            print(f"  step {s:>3} loss={last_loss:.5e} step_time={dt:.3f}s")
    torch.cuda.synchronize()
    total_s = time.perf_counter() - t0

    # ---- FLOPS ----
    # Match rustorch accounting :
    #   fwd GEMMs per layer per step = 3 GEMMs each 2·M·K·N flops
    #     gate / up : 2 · tokens · hidden · ffn
    #     down      : 2 · tokens · ffn · hidden
    #   fwd_gemm_flops = 2 · tokens · hidden · ffn · 3
    #   total_flops_per_step ≈ 3× fwd (counting fwd+bwd GEMM doublet)
    fwd_gemm_flops = tokens * args.hidden * args.ffn * 2 * 3
    total_flops_per_step = fwd_gemm_flops * 3 * args.layers
    total_flops = total_flops_per_step * args.steps
    tflops = total_flops / total_s / 1e12
    pct_peak = tflops / 250.0 * 100.0

    tokens_total = tokens * args.steps
    tok_per_s = tokens_total / total_s
    ms_per_step = total_s * 1000.0 / args.steps
    peak_mem_gb = torch.cuda.max_memory_allocated(device) / 1e9
    p50 = statistics.median(step_times) * 1000.0
    p95 = (sorted(step_times)[int(len(step_times) * 0.95)]
           if len(step_times) > 1 else step_times[0]) * 1000.0

    print()
    print("---- results ----")
    print(f"  total wall-clock : {total_s:.4f}s")
    print(f"  per-step (mean)  : {ms_per_step:.2f} ms")
    print(f"  per-step (p50)   : {p50:.2f} ms")
    print(f"  per-step (p95)   : {p95:.2f} ms")
    print(f"  tokens/sec       : {tok_per_s:.1f}")
    print(f"  TFLOPS achieved  : {tflops:.2f}")
    print(f"  % of 250 TF peak : {pct_peak:.2f}%")
    print(f"  GPU peak mem     : {peak_mem_gb:.2f} GB")
    print(f"  final loss       : {last_loss:.5e}")
    print()

    payload = {
        "path": "pytorch_cuda_maxcap" + ("_compile-" + args.mode if args.compile else "_eager"),
        "steps": args.steps,
        "tokens_per_step": tokens,
        "layers": args.layers,
        "hidden": args.hidden,
        "ffn": args.ffn,
        "wall_clock_s": round(total_s, 4),
        "ms_per_step": round(ms_per_step, 4),
        "tokens_per_sec": round(tok_per_s, 2),
        "tflops": round(tflops, 4),
        "pct_peak": round(pct_peak, 4),
        "p50_ms": round(p50, 4),
        "p95_ms": round(p95, 4),
        "peak_mem_gb": round(peak_mem_gb, 4),
        "dtype": args.dtype,
        "device": torch.cuda.get_device_name(0),
        "sm": list(torch.cuda.get_device_capability(0)),
        "torch": torch.__version__,
    }
    print(f"RESULT_JSON {json.dumps(payload)}")
    print("=== done ===")


if __name__ == "__main__":
    main()
