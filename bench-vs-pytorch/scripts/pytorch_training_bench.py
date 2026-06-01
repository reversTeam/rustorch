#!/usr/bin/env python3
"""T246.11 TRAINING-BENCH.2 — PyTorch CUDA equivalent of the rustorch
training_bench_cuda example.

Same workload as ``crates/rustorch-llm/examples/training_bench_cuda.rs`` :
synthetic transformer-style block (RMSNorm + linear + SwiGLU + linear +
residual), BF16 activations, F32 master weights, AdamW. Reports
wall-clock and tokens/sec, matching the rustorch ``RESULT_JSON ...`` line
for direct comparison.

Run with:
    python3 pytorch_training_bench.py [--compile]

The optional ``--compile`` flag enables ``torch.compile`` (default :
inductor) — PyTorch's best path on modern GPUs. We default to it OFF
so the comparison is closer to ``rustorch.matmul_bf16_cuda`` (single
cuBLASLt matmul per call, no fusion / no compile-time graph) ; pass
``--compile`` to see the compiled-PyTorch ceiling.
"""

from __future__ import annotations

import argparse
import json
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


class BenchBlock(nn.Module):
    """RMSNorm → gate proj → up proj → SwiGLU → down proj → residual.

    Matches the rustorch ``training_bench_cuda.rs`` layout precisely.
    BF16 activations, F32 master weights (PyTorch's autocast semantics).
    """

    def __init__(self, hidden: int, ffn: int):
        super().__init__()
        # Use ``nn.Linear`` so we mirror PyTorch's standard wiring.
        self.norm = nn.RMSNorm(hidden, eps=1e-6)
        self.w_gate = nn.Linear(hidden, ffn, bias=False)
        self.w_up = nn.Linear(hidden, ffn, bias=False)
        self.w_down = nn.Linear(ffn, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = self.norm(x)
        gate = self.w_gate(n)
        up = self.w_up(n)
        # SwiGLU = silu(gate) * up — same as rustorch swiglu_cuda.
        act = F.silu(gate) * up
        h = self.w_down(act)
        return h + x


def init_det(t: torch.Tensor, seed: float) -> None:
    """Deterministic init mirroring the rustorch ``det_tensor`` helper."""
    n = t.numel()
    idx = torch.arange(1, n + 1, dtype=torch.float32, device=t.device)
    vals = ((idx * seed * 0.0001).sin() * 0.1).reshape(t.shape)
    with torch.no_grad():
        t.copy_(vals.to(t.dtype))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compile", action="store_true", help="enable torch.compile")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--seq", type=int, default=512)
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--ffn", type=int, default=4096)
    parser.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    parser.add_argument("--warmup", type=int, default=3)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available — this bench requires a CUDA GPU.")

    device = torch.device("cuda:0")
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32

    print("=== PyTorch CUDA training bench (T246.11 TRAINING-BENCH.2) ===")
    m = args.batch * args.seq
    print(
        f"shape : batch={args.batch} seq_len={args.seq} tokens/step={m} "
        f"hidden={args.hidden} ffn={args.ffn} steps={args.steps}"
    )
    print(f"dtype : {args.dtype} activations / matmul, F32 master weights")
    print(f"compile : {'YES (torch.compile)' if args.compile else 'NO (eager)'}")
    print(f"device  : {torch.cuda.get_device_name(0)} sm={torch.cuda.get_device_capability(0)}")
    print()

    # ---- Inputs / weights ----
    x = torch.empty(m, args.hidden, device=device, dtype=dtype)
    init_det(x, 1.0)
    target = torch.full(
        (m, args.hidden), 0.5, device=device, dtype=dtype, requires_grad=False
    )

    model = BenchBlock(args.hidden, args.ffn).to(device=device, dtype=dtype)
    # Deterministic init seeds matching the rustorch bench.
    init_det(model.norm.weight, 0.7)
    init_det(model.w_gate.weight, 0.5)
    init_det(model.w_up.weight, 0.3)
    init_det(model.w_down.weight, 0.2)

    if args.compile:
        model = torch.compile(model)

    optim = torch.optim.AdamW(model.parameters(), lr=1e-4)

    def step(x_in: torch.Tensor) -> torch.Tensor:
        optim.zero_grad(set_to_none=True)
        y = model(x_in)
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
        _ = step(x)
    torch.cuda.synchronize()
    print(f"warm-up done : {time.perf_counter() - twarm:.3f}s")
    print()

    # ---- Measured loop ----
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    last_loss = 0.0
    for s in range(args.steps):
        loss = step(x)
        if s == 0 or s == args.steps - 1:
            torch.cuda.synchronize()
            last_loss = loss.item()
            print(f"  step {s} loss={last_loss:.5e}")
    torch.cuda.synchronize()
    total_s = time.perf_counter() - t0

    tokens = m * args.steps
    tok_per_s = tokens / total_s
    ms_per_step = total_s * 1000.0 / args.steps
    peak_mem_gb = torch.cuda.max_memory_allocated(device) / 1e9

    print()
    print("---- results ----")
    print(f"  total wall-clock : {total_s:.4f}s")
    print(f"  per-step         : {ms_per_step:.2f} ms")
    print(f"  tokens/sec       : {tok_per_s:.1f}")
    print(f"  GPU peak mem     : {peak_mem_gb:.2f} GB")
    print(f"  final loss       : {last_loss:.5e}")
    print()

    payload = {
        "path": "pytorch_cuda" + ("_compile" if args.compile else "_eager"),
        "steps": args.steps,
        "tokens_per_step": m,
        "wall_clock_s": round(total_s, 4),
        "ms_per_step": round(ms_per_step, 4),
        "tokens_per_sec": round(tok_per_s, 2),
        "peak_mem_gb": round(peak_mem_gb, 4),
        "dtype": args.dtype,
        "device": torch.cuda.get_device_name(0),
        "sm": list(torch.cuda.get_device_capability(0)),
    }
    print(f"RESULT_JSON {json.dumps(payload)}")
    print("=== done ===")


if __name__ == "__main__":
    main()
