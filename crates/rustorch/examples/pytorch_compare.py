"""PyTorch comparison for the rustorch cpu_vs_wgpu_train example.

Mirror of `crates/rustorch/examples/cpu_vs_wgpu_train.rs` running on
PyTorch CPU + MPS so we can compare on the same machine with the same
workload. Generates the data quoted in the README perf section.

Run from a directory other than this repo's root (so Python doesn't
shadow the installed `torch` package with `pytorch/` source):

    cd /tmp && python3 /path/to/rustorch/crates/rustorch/examples/pytorch_compare.py

Workload: B=64, M=1024, N=1024, n_steps=5 (after 1 warmup step).
"""

from __future__ import annotations

import os
import time
import torch
from torch import optim


def time_train_steps(device: str, b: int, m: int, n: int, n_steps: int) -> float:
    """Run `n_steps` of (forward + MSE + backward + AdamW.step) and time it.

    Mirrors rustorch's `time_train_steps` in `cpu_vs_wgpu_train.rs`:
    deterministic init via sin-based seeding, AdamW lr=0.001, MSE mean.
    """
    dev = torch.device(device)

    def det(shape, seed: float) -> torch.Tensor:
        flat = torch.arange(
            1,
            1 + int(torch.tensor(shape).prod()),
            dtype=torch.float32,
        )
        return (flat * seed * 0.001).sin().reshape(shape).to(dev)

    xs = det((b, m), 1.0)
    ys = det((b, n), 0.5)
    w = torch.nn.Parameter(det((m, n), 0.1))
    bias = torch.nn.Parameter(det((n,), 0.05))
    opt = optim.AdamW([w, bias], lr=0.001)

    # 1 warmup step (matches rustorch's pre-roll)
    for _ in range(1):
        opt.zero_grad()
        pred = xs @ w + bias
        loss = ((pred - ys) ** 2).mean()
        loss.backward()
        opt.step()
    if device == "mps":
        torch.mps.synchronize()

    t0 = time.perf_counter()
    for _ in range(n_steps):
        opt.zero_grad()
        pred = xs @ w + bias
        loss = ((pred - ys) ** 2).mean()
        loss.backward()
        opt.step()
    if device == "mps":
        torch.mps.synchronize()
    return time.perf_counter() - t0


def main() -> None:
    b, m, n, n_steps = 64, 1024, 1024, 5
    print(
        f"=== PyTorch training-step bench — Linear [B={b}, M={m}] @ "
        f"[M={m}, N={n}], MSE+AdamW, {n_steps} steps ==="
    )
    print(
        f"PyTorch: {torch.__version__}, "
        f"MPS available: {torch.backends.mps.is_available()}"
    )

    cpu_time = time_train_steps("cpu", b, m, n, n_steps)
    cpu_per_ms = cpu_time * 1000 / n_steps
    print(
        f"PyTorch CPU  : {n_steps} steps in {cpu_time*1000:.1f} ms, "
        f"{cpu_per_ms:.2f} ms/step"
    )

    if torch.backends.mps.is_available():
        mps_time = time_train_steps("mps", b, m, n, n_steps)
        mps_per_ms = mps_time * 1000 / n_steps
        print(
            f"PyTorch MPS  : {n_steps} steps in {mps_time*1000:.1f} ms, "
            f"{mps_per_ms:.2f} ms/step"
        )

    print()
    print(
        "Cross-reference (run "
        "`cargo run -p rustorch --release --example cpu_vs_wgpu_train --features wgpu`):"
    )
    print("  rustorch CPU  : ~12.8 ms/step  (~7× slower than PyTorch CPU)")
    print(
        "  rustorch Wgpu : ~22.3 ms/step  "
        "(Storage Option B host↔device round-trip dominates)"
    )


if __name__ == "__main__":
    # Ensure we're not running from a directory that shadows the installed
    # torch package (e.g. running from rustorch/ root would import the
    # local pytorch/ source dir instead).
    cwd = os.path.abspath(os.getcwd())
    if cwd.endswith("rustorch") or "rustorch/" in cwd + "/":
        # Heuristic: re-run from /tmp if started from inside the repo.
        print(
            f"⚠️  cwd is {cwd}; PyTorch may be shadowed by the local "
            "pytorch/ source. Re-run from /tmp or a sibling directory."
        )
    main()
