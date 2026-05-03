#!/usr/bin/env python3
"""Parse criterion estimates.json files into a flat dict matching the
schema of the PyTorch harness so `compare.py` can ingest both."""

import json
import re
from pathlib import Path

ROOT = Path("target/criterion")

# Mapping criterion bench id (path-derived) -> (op_label, shape_label) used
# to align against PyTorch results.
def parse_path(p: Path):
    parts = p.parts[len(ROOT.parts):-2]
    # parts is like ('matmul_f32', '256x256x256') or ('flash_attn_vs_naive', 'flash', 'B1H1N512D64')
    if len(parts) == 2:
        group, shape = parts
        return group, shape
    elif len(parts) == 3:
        return f"{parts[0]}/{parts[1]}", parts[2]
    return None


results = []
for f in ROOT.glob("**/base/estimates.json"):
    pp = parse_path(f)
    if pp is None:
        continue
    op, shape = pp
    with open(f) as fp:
        est = json.load(fp)
    median = int(est["median"]["point_estimate"])
    upper = int(est["median"]["confidence_interval"]["upper_bound"])
    results.append({"op": op, "shape": shape, "median_ns": median, "p99_ns": upper})

# Stable order
results.sort(key=lambda r: (r["op"], r["shape"]))

out = {
    "framework": "rustorch",
    "harness": "criterion",
    "warmup_iters": "criterion-default",
    "timed_iters": "criterion-default (50 samples)",
    "build": "release + target-cpu=native",
    "results": results,
}
print(json.dumps(out, indent=2))
