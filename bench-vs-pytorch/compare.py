#!/usr/bin/env python3
"""Obrain-grade comparison tool.

Two modes:

1. Snapshot mode — `compare.py --rustorch X --pytorch1 Y --pytorch12 Z`
   Prints table: RusTorch vs PyTorch with cache-tier classification + Gelem/s.

2. Delta mode — `compare.py --before A --after B [--pytorch1 P1 [--pytorch12 P12]]`
   Compares two RusTorch baselines. Shows speedup/regression per op.
   Optionally re-projects against PyTorch.

Both modes emit a Markdown table to stdout.
"""

import argparse
import json
import sys
from pathlib import Path


# ----- Working set classification (M4 Max P-core) -------------------------
# L1d = 192 KB, L2 = 16 MB, DRAM beyond
M4_L1D_BYTES = 192 * 1024
M4_L2_BYTES = 16 * 1024 * 1024


def shape_size_bytes(op, shape):
    """Approximate the per-call working set in bytes (input + output)."""
    if op.startswith("matmul") or op.startswith("fused_mbr"):
        # "256x256x256"
        m, k, n = (int(x) for x in shape.split("x"))
        return (m * k + k * n + m * n) * 4  # f32
    if op.startswith("softmax"):
        rows, cols = (int(x) for x in shape.split("x"))
        return rows * cols * 4 * 2  # input + output
    if op.startswith("flash_attn") or op == "sdpa":
        # "B1H4N1024D64"
        try:
            b = int(shape.split("B")[1].split("H")[0])
            h = int(shape.split("H")[1].split("N")[0])
            n = int(shape.split("N")[1].split("D")[0])
            d = int(shape.split("D")[1])
            return b * h * n * d * 4 * 4  # Q+K+V+O all [B,H,N,D]
        except Exception:
            return 0
    if op.startswith("elementwise"):
        return int(shape) * 4 * 2  # in + out
    if op.startswith("naive_attention"):
        try:
            b = int(shape.split("B")[1].split("H")[0])
            h = int(shape.split("H")[1].split("N")[0])
            n = int(shape.split("N")[1].split("D")[0])
            d = int(shape.split("D")[1])
            return b * h * n * d * 4 * 4
        except Exception:
            return 0
    return 0


def cache_tier(bytes_):
    if bytes_ == 0:
        return "?"
    if bytes_ <= M4_L1D_BYTES:
        return "L1d"
    if bytes_ <= M4_L2_BYTES:
        return "L2"
    return "DRAM"


# ----- FLOPS estimator ---------------------------------------------------
def gflops(op, shape, ns):
    """Rough GFLOPS estimate for the op. Returns None if op is memory-bound."""
    if ns <= 0:
        return None
    if op.startswith("matmul") or op.startswith("fused_mbr") or op.startswith("matmul_bf16"):
        m, k, n = (int(x) for x in shape.split("x"))
        return (2.0 * m * k * n) / ns
    if op.startswith("flash_attn") or op == "sdpa" or op.startswith("naive_attention"):
        try:
            b = int(shape.split("B")[1].split("H")[0])
            h = int(shape.split("H")[1].split("N")[0])
            n = int(shape.split("N")[1].split("D")[0])
            d = int(shape.split("D")[1])
            return (4.0 * b * h * n * n * d) / ns  # 2 matmuls of NxD with NxN
        except Exception:
            return None
    return None


def gelems(op, shape, ns):
    """Throughput in Gelem/s (memory-bound metric)."""
    bytes_ = shape_size_bytes(op, shape)
    if bytes_ == 0 or ns <= 0:
        return None
    return (bytes_ / 4.0) / ns  # f32 elements


def fmt_ms(ns):
    if ns < 1_000:
        return f"{ns:>8.0f} ns"
    if ns < 1_000_000:
        return f"{ns/1000:>8.2f} µs"
    return f"{ns/1e6:>8.2f} ms"


def fmt_perf(op, shape, ns):
    g = gflops(op, shape, ns)
    e = gelems(op, shape, ns)
    if g is not None:
        return f"{g:>6.2f} GF/s"
    if e is not None:
        return f"{e:>5.2f} GE/s"
    return "—"


def fmt_ratio(rust_ns, pt_ns):
    if pt_ns == 0:
        return "n/a"
    r = rust_ns / pt_ns
    if r >= 100:
        return f"{r:>6.0f}×"
    if r >= 10:
        return f"{r:>6.1f}×"
    return f"{r:>6.2f}×"


# ----- PyTorch shape normalisation --------------------------------------
def normalize_pt_shape(op, shape):
    if op == "matmul":
        s = shape.replace(" ", "").replace("[", "").replace("]", "")
        a, b = s.split("x")
        m, _ = a.split(",")
        _, n = b.split(",")
        # use a-side k since k1==k2
        k1 = a.split(",")[1]
        return ("matmul_f32", f"{m}x{k1}x{n}")
    if op == "softmax":
        s = shape.replace("[", "").replace("]", "").replace(",", "x")
        return ("softmax_lastdim", s)
    if op == "sdpa":
        s = shape.replace(" ", "").replace("=", "")
        return ("flash_attn_vs_naive/flash", s)
    if op == "naive_attention_forward":
        s = shape.replace(" ", "").replace("=", "")
        return ("flash_attn_vs_naive/naive", s)
    if op == "elementwise_add":
        s = shape.replace("[", "").replace("]", "")
        return ("elementwise_add_inplace", s)
    return (None, None)


def index_pytorch(pt_data):
    out = {}
    for r in pt_data["results"]:
        k = normalize_pt_shape(r["op"], r["shape"])
        if k[0]:
            out[k] = r
    return out


# ----- Snapshot mode -----------------------------------------------------
def cmd_snapshot(args):
    rust = json.load(open(args.rustorch))
    pt1 = json.load(open(args.pytorch1)) if args.pytorch1 else None
    pt12 = json.load(open(args.pytorch12)) if args.pytorch12 else None

    pt1_idx = index_pytorch(pt1) if pt1 else {}
    pt12_idx = index_pytorch(pt12) if pt12 else {}

    print(f"\n{'='*140}")
    print(f"# Snapshot — RusTorch ({Path(args.rustorch).stem}) vs PyTorch (M4 Max)")
    print(f"{'='*140}")
    print(f"{'OP':<30} {'SHAPE':<22} {'TIER':<5} {'TIME':>10} {'PERF':>12} {'PT-1thr':>10} {'PT-12thr':>10} {'vs PT-1':>9} {'vs PT-12':>9}")
    print(f"{'-'*140}")

    last_grp = None
    for r in rust["results"]:
        op, shape, ns = r["op"], r["shape"], r["median_ns"]
        bytes_ = shape_size_bytes(op, shape)
        tier = cache_tier(bytes_)
        perf = fmt_perf(op, shape, ns)

        pt1_r = pt1_idx.get((op, shape))
        pt12_r = pt12_idx.get((op, shape))
        pt1_str = fmt_ms(pt1_r["median_ns"]) if pt1_r else "—"
        pt12_str = fmt_ms(pt12_r["median_ns"]) if pt12_r else "—"
        r1 = fmt_ratio(ns, pt1_r["median_ns"]) if pt1_r else "—"
        r12 = fmt_ratio(ns, pt12_r["median_ns"]) if pt12_r else "—"

        grp = op.split("/")[0]
        if grp != last_grp:
            if last_grp is not None:
                print(f"{'-'*140}")
            last_grp = grp

        print(f"{op:<30} {shape:<22} {tier:<5} {fmt_ms(ns):>10} {perf:>12} {pt1_str:>10} {pt12_str:>10} {r1:>9} {r12:>9}")

    # Internal speedups summary
    print(f"\n{'='*140}")
    print("# INTERNAL — Flash vs naive (RusTorch's flagship algorithmic optimization)")
    print(f"{'-'*140}")
    flash = {r["shape"]: r for r in rust["results"] if r["op"] == "flash_attn_vs_naive/flash"}
    naive = {r["shape"]: r for r in rust["results"] if r["op"] == "flash_attn_vs_naive/naive"}
    for shape, f in sorted(flash.items()):
        n = naive.get(shape)
        if not n:
            continue
        speedup = n["median_ns"] / f["median_ns"]
        print(f"  {shape:<22}  flash={fmt_ms(f['median_ns']):>10}  naive={fmt_ms(n['median_ns']):>10}  speedup={speedup:>5.2f}×")


# ----- Delta mode --------------------------------------------------------
def cmd_delta(args):
    before = json.load(open(args.before))
    after = json.load(open(args.after))
    pt1 = json.load(open(args.pytorch1)) if args.pytorch1 else None
    pt1_idx = index_pytorch(pt1) if pt1 else {}

    before_idx = {(r["op"], r["shape"]): r for r in before["results"]}

    print(f"\n{'='*140}")
    print(f"# Delta — {Path(args.before).stem} ⇒ {Path(args.after).stem}")
    print(f"{'='*140}")
    print(f"{'OP':<30} {'SHAPE':<22} {'TIER':<5} {'BEFORE':>10} {'AFTER':>10} {'DELTA':>10} {'GAIN':>8} {'vs PT-1':>9}")
    print(f"{'-'*140}")

    last_grp = None
    total_before = 0
    total_after = 0
    for r in after["results"]:
        op, shape, ns = r["op"], r["shape"], r["median_ns"]
        b = before_idx.get((op, shape))
        if b is None:
            continue
        bns = b["median_ns"]
        delta = ns - bns
        gain = bns / ns if ns > 0 else 0
        gain_str = f"{gain:>5.2f}×" if gain >= 1 else f"-{1/gain:>5.2f}×"
        delta_str = fmt_ms(delta) if delta >= 0 else f"-{fmt_ms(-delta).strip()}"
        bytes_ = shape_size_bytes(op, shape)
        tier = cache_tier(bytes_)

        pt1_r = pt1_idx.get((op, shape))
        pt1_ratio = fmt_ratio(ns, pt1_r["median_ns"]) if pt1_r else "—"

        grp = op.split("/")[0]
        if grp != last_grp:
            if last_grp is not None:
                print(f"{'-'*140}")
            last_grp = grp

        print(f"{op:<30} {shape:<22} {tier:<5} {fmt_ms(bns):>10} {fmt_ms(ns):>10} {delta_str:>10} {gain_str:>8} {pt1_ratio:>9}")
        total_before += bns
        total_after += ns

    if total_before > 0:
        cumul_gain = total_before / total_after
        print(f"\nCumulative time: {fmt_ms(total_before).strip()} → {fmt_ms(total_after).strip()}  ⇒  {cumul_gain:.2f}× overall speedup")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=False)

    ap.add_argument("--rustorch", help="RusTorch JSON (snapshot mode)")
    ap.add_argument("--pytorch1", help="PyTorch 1-thread JSON")
    ap.add_argument("--pytorch12", help="PyTorch 12-thread JSON")
    ap.add_argument("--before", help="Before RusTorch JSON (delta mode)")
    ap.add_argument("--after", help="After RusTorch JSON (delta mode)")

    args = ap.parse_args()

    if args.before and args.after:
        cmd_delta(args)
    elif args.rustorch:
        cmd_snapshot(args)
    else:
        ap.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
