#!/usr/bin/env python3
"""V2 comparison: criterion-derived RusTorch numbers vs PyTorch."""
import json

with open("rustorch_criterion.json") as f:
    rust = json.load(f)
with open("pytorch_1thread.json") as f:
    pt1 = json.load(f)
with open("pytorch_12thread.json") as f:
    pt12 = json.load(f)


def fmt_ms(ns):
    if ns < 1_000:
        return f"{ns:>8.0f} ns"
    if ns < 1_000_000:
        return f"{ns/1000:>8.2f} µs"
    return f"{ns/1e6:>8.2f} ms"


def fmt_ratio(rust_ns, pt_ns):
    if pt_ns == 0:
        return "n/a"
    r = rust_ns / pt_ns
    if r >= 1:
        return f"{r:>7.0f}×" if r >= 100 else f"{r:>7.1f}×"
    return f"{r:>7.2f}×"


# Build PyTorch lookup. PyTorch has shape "[m,k] x [k,n]" but RusTorch criterion
# uses "MxKxN". Need normalization.
def normalize_pt_shape(op, shape):
    if op == "matmul":
        # "[128,128] x [128,128]" -> "128x128x128"
        s = shape.replace(" ", "").replace("[", "").replace("]", "")
        a, b = s.split("x")
        m, k1 = a.split(",")
        k2, n = b.split(",")
        return ("matmul_f32", f"{m}x{k1}x{n}")
    if op == "softmax":
        # "[128,32000]" -> "128x32000"
        s = shape.replace("[", "").replace("]", "").replace(",", "x")
        return ("softmax_lastdim", s)
    if op == "sdpa":
        # "B=1 H=1 N=512 D=64" -> "B1H1N512D64"
        s = shape.replace(" ", "").replace("=", "")
        return ("flash_attn_vs_naive/flash", s)
    if op == "naive_attention_forward":
        s = shape.replace(" ", "").replace("=", "")
        return ("flash_attn_vs_naive/naive", s)
    if op == "elementwise_add":
        s = shape.replace("[", "").replace("]", "")
        return ("elementwise_add_inplace", s)
    return (None, None)


pt1_idx, pt12_idx = {}, {}
for r in pt1["results"]:
    k = normalize_pt_shape(r["op"], r["shape"])
    if k[0]:
        pt1_idx[k] = r
for r in pt12["results"]:
    k = normalize_pt_shape(r["op"], r["shape"])
    if k[0]:
        pt12_idx[k] = r


print(f"\n{'='*120}")
print(f"M4 Max — RusTorch (criterion + target-cpu=native + black_box) vs PyTorch {pt1['version']}")
print(f"{'='*120}")
print(f"{'OP':<35} {'SHAPE':<25} {'RUSTORCH':>12} {'PT-1thr':>12} {'PT-12thr':>12} {'vs PT-1':>10} {'vs PT-12':>10}")
print(f"{'-'*120}")

last_group = None
for r in rust["results"]:
    op, shape = r["op"], r["shape"]
    rust_ns = r["median_ns"]
    pt1_r = pt1_idx.get((op, shape))
    pt12_r = pt12_idx.get((op, shape))
    p1 = pt1_r["median_ns"] if pt1_r else None
    p12 = pt12_r["median_ns"] if pt12_r else None

    grp = op.split("/")[0]
    if grp != last_group:
        if last_group is not None:
            print(f"{'-'*120}")
        last_group = grp

    pt1_str = fmt_ms(p1) if p1 else "—"
    pt12_str = fmt_ms(p12) if p12 else "—"
    r1 = fmt_ratio(rust_ns, p1) if p1 else "—"
    r12 = fmt_ratio(rust_ns, p12) if p12 else "—"
    print(f"{op:<35} {shape:<25} {fmt_ms(rust_ns):>12} {pt1_str:>12} {pt12_str:>12} {r1:>10} {r12:>10}")

print(f"\n{'='*120}")
print("KEY FINDINGS")
print(f"{'='*120}")

# Internal speedups
print("\n[INTERNAL] Flash Attention vs naive (RusTorch's flagship algorithmic optimization):")
flash = {r["shape"]: r for r in rust["results"] if r["op"] == "flash_attn_vs_naive/flash"}
naive = {r["shape"]: r for r in rust["results"] if r["op"] == "flash_attn_vs_naive/naive"}
for shape, f in flash.items():
    n = naive.get(shape)
    if not n:
        continue
    speedup = n["median_ns"] / f["median_ns"]
    print(f"  {shape:<20}  flash={fmt_ms(f['median_ns'])}  vs  naive={fmt_ms(n['median_ns'])}  =>  {speedup:.2f}× speedup")

# Matmul scaling
print("\n[INTERNAL] Matmul scaling (3 nested loops, no SIMD/blocking):")
mm = {r["shape"]: r for r in rust["results"] if r["op"] == "matmul_f32"}
for shape, r in sorted(mm.items()):
    print(f"  {shape:<20}  median={fmt_ms(r['median_ns'])}")

# Fusion overhead
print("\n[INTERNAL] Fused matmul+bias+ReLU vs unfused (3-pass naive):")
fused = {r["shape"]: r for r in rust["results"] if r["op"] == "fused_mbr_vs_naive/fused"}
unfused = {r["shape"]: r for r in rust["results"] if r["op"] == "fused_mbr_vs_naive/naive"}
for shape, f in fused.items():
    u = unfused.get(shape)
    if u:
        gain = (u["median_ns"] - f["median_ns"]) / u["median_ns"] * 100
        print(f"  {shape:<20}  fused={fmt_ms(f['median_ns'])}  unfused={fmt_ms(u['median_ns'])}  saving={gain:+.1f}%  (matmul-dominated)")

# BF16 vs F32
print("\n[INTERNAL] BF16 (AMP) matmul vs F32 matmul on M4 CPU (no native bf16 SIMD):")
bf16 = {r["shape"]: r for r in rust["results"] if r["op"] == "matmul_bf16_acc_f32"}
f32 = {r["shape"]: r for r in rust["results"] if r["op"] == "matmul_f32"}
for shape, b in sorted(bf16.items()):
    f_r = f32.get(shape)
    if f_r:
        ratio = b["median_ns"] / f_r["median_ns"]
        verdict = "REGRESSION" if ratio > 1.05 else "neutral" if ratio < 1.05 else ("WIN" if ratio < 0.95 else "neutral")
        print(f"  {shape:<20}  bf16={fmt_ms(b['median_ns'])}  f32={fmt_ms(f_r['median_ns'])}  bf16/f32={ratio:.2f}×  {verdict}")
