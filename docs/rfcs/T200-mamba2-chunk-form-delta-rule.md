# T200 — Mamba2 chunk-form parallel scan for delta-rule SSM

**Status** : Draft (T200.1 audit, T200.2-3 pending)
**Author** : Performance team
**Date** : 2026-05-07

## Problem

`ssm_block_forward_batch` runs a **sequential B-step scan** over each SSM
layer. Profile shows `fbs.scan_loop = 2398 µs/call × 30 SSM × 4 chunks
= 288 ms = 22% of total prefill on Qwen3.6-35B-A3B (B=128, prompt 430 tok)`.

The scan is forced sequential because each timestep updates an SSM state
matrix that the next timestep reads. Eliminating this serialization is
the largest remaining single-pass lever on the model.

Reference : Dao & Gu, "Transformers are SSMs" (arXiv 2405.21060) §3.4
"Structured State Space Duality" — shows linear-attention-like recurrences
admit chunk-parallel execution.

## Current recurrence (delta-rule with L2-normed q, k)

For each timestep `t ∈ [0, B)`, head `head_v ∈ [0, n_v)`, row `r ∈ [0, head_dim)` :

```
γ_t = exp(gate_h_t[head_v])                 (scalar per (t, head_v))
inv_q_t = 1 / |q_t[head_k(head_v)]|         (scalar per (t, head_k))
inv_k_t = 1 / |k_t[head_k(head_v)]|

# Step 1 : decay state
S'_t[r, c] = γ_t · S_{t-1}[r, c]                     ∀ c

# Step 2 : projection (q^T S k pattern)
proj_r = (Σ_c S'_t[r, c] · k_t[c]) · inv_k_t

# Step 3 : delta-rule update (rank-1 outer product)
δ_r = β_t[head_v] · (v_t[head_v, r] - proj_r) · inv_k_t
S_t[r, c] = S'_t[r, c] + δ_r · k_t[c]                ∀ c

# Step 4 : output (linear attention readout)
out_t[head_v, r] = (Σ_c S_t[r, c] · q_t[c]) · inv_q_t · (1/√head_dim)
```

State shape per `head_v` : `[head_dim, head_dim]` = 128 × 128 = 64 KB f32.
Total state per layer (n_v=16 heads) : 1 MB.

## Math reformulation

### Substitution to absorb scalings

Let `k̃_t = k_t · inv_k_t` (L2-normed key) and `q̃_t = q_t · inv_q_t · (1/√d)`.
Define `α_t = β_t · inv_k_t`. Then :

```
proj_r        = <S'_t[r, :], k_t> · inv_k_t  =  <S'_t[r, :], k̃_t / inv_k_t> · inv_k_t
              =  <S'_t[r, :], k̃_t>                (the inv_k cancels because S' was scaled by γ but k̃ is normed)

WAIT — proj_r = γ_t · <S_{t-1}[r,:], k̃_t / inv_k_t> · inv_k_t  =  γ_t · <S_{t-1}[r,:], k̃_t>
```

Hmm, the scalings don't fully cancel. Let me redo.

Setting `K_t = k_t · inv_k_t`, `Q_t = q_t · inv_q_t / √d`, `α_t = β_t · inv_k_t² · k_t`
(absorbing inv_k_t twice into α and rescaling) :

After absorbtion (proof in `delta_net_step_with_l2_f32` shader comment), the
recurrence simplifies to :

```
S_t = γ_t · S_{t-1} + (β_t · v_t - β_t · γ_t · S_{t-1} K_t) · k_t^T   (vector eq)
out_t = S_t · Q_t                                                       (vector eq)
```

This has the form of a **linear attention** with a state-dependent input
(the `S_{t-1} K_t` term creates the delta-rule contraction). It's NOT a
pure Mamba1 SSM (where input doesn't depend on state).

### Chunk-form decomposition

Divide T=B timesteps into chunks of size C (try C=16 or C=32).
For chunk indexed `g ∈ [0, B/C)`, denote `t = g·C + i` with `i ∈ [0, C)`.

**Key observation** : within a chunk, if we treat `S_{t-1} K_t` as a
"feedback" that depends on previous timesteps, we can split into :

```
S_t = γ_t·γ_{t-1}·...·γ_{gC} · S_{gC-1}                       (carry from prior chunk)
    + Σ_{j=gC..t} (γ_t·γ_{t-1}·...·γ_{j+1}) · ΔS_j_intra       (intra-chunk contributions)
```

where `ΔS_j_intra` requires knowing `S_{j-1}` for the projection step.
This is the difficult part — the delta_r involves `S_{j-1}`.

**Trick (Mamba2 §3.4)** : the rank-1 outer-product structure means we can
reformulate as a **causal masked attention** :

```
out_g[i, r] = Q_t · S_{gC-1} · k̃_t                          (carry term — easy)
            + Σ_{j=0..i} attention_weight(i, j) · v_{gC+j}    (intra-chunk attention)
```

The intra-chunk contribution becomes a **causal attention over C tokens**
with attention weights involving `Π γ` (cumulative product) and
`<Q_i, k̃_j>`.

Detailed math : the `proj_r` term creates a recursive dependency that can
be unrolled within a chunk via dense matrix product, similar to how
FlashAttention computes attention without materializing full N² matrix.

### Specific to OUR delta-rule

The shader computes `proj_r` AND `δ_r · k[c]` per `(r, c)` — this is a
rank-1 outer product where the rank-1 vector depends on `S_{t-1}`.
The "true" Mamba2 SSD says : if the recurrence has form
`S_t = A_t · S_{t-1} + B_t @ x_t`, with `A_t` scalar, `B_t` vector, `x_t`
vector, then chunk-form gives :

```
S_chunk_end = (Π γ) · S_chunk_start + Σ (Π_residual γ) · (B_j @ x_j)
```

For our case `A_t = γ_t · I` (scalar diagonal), `B_t = β_t · I`, `x_t = (v_t - S_{t-1} K_t) k̃_t^T`.

Because `x_t` depends on `S_{t-1}`, we CANNOT apply chunk-form directly.
**This is the core difficulty.**

### Proposed approach — partial chunk-form (intra-chunk sequential)

Plan B : keep recurrence intra-chunk (small loop of C iterations), but
**eliminate kernel dispatch overhead** by doing all C iterations within
ONE kernel.

Currently the scan loop dispatches 3 kernels × B = 384 dispatches per
layer. With chunk-size C=128 (whole batch in 1 kernel), we'd have
3 dispatches per layer = 128× reduction in dispatch overhead.

**Within one persistent kernel** :
- Each TG handles 1 (head_v, row) pair
- The TG loops over t = 0..B sequentially
- Reads q_t, k_t, v_t, γ_t, β_t for current step from device memory
- Updates state in registers/threadgroup memory
- Writes out_t per timestep

State per TG : `head_dim` floats (the row) = 128 floats = 512 bytes.
Easy fit in registers.

Total dispatch reduction : ~15k per prefill saved.

This is **NOT true parallel scan** — within a chunk, work is still
sequential. But avoids dispatch + barrier overhead.

## Estimated gains

| Approach | Speedup on scan_loop | Global prefill gain | Effort | Risk |
|---|---|---|---|---|
| **Full Mamba2 chunk-form (true parallel)** | ~5× (288 ms → 60 ms) | +12-15% | 5-7 jours | high (math complex, parity numerical) |
| **Persistent kernel (sequential intra)** | ~2× (288 ms → 144 ms) | +6-8% | 2-3 jours | medium (state in TG, memory budget) |

Recommendation : start with **Plan B (persistent kernel)** for faster
shipping. Migrate to Plan A (true chunk-form) later if the math works
out cleanly.

## Plan B kernel design

```rust
pub fn delta_net_persistent_scan_f32_into(
    backend: &MetalBackend,
    qkv_combined_buf: &Buffer,         // [B, conv_dim] — already split via offsets
    qkv_combined_stride: usize,        // conv_dim * 4 bytes
    gate_h_batched_buf: &Buffer,       // [B, n_v]
    beta_sig_batched_buf: &Buffer,     // [B, n_v]
    state_buf: &Buffer,                // [n_v, head_dim, head_dim] read+write
    out_buf: &Buffer,                  // [B, n_v, head_dim]
    b: usize,                           // sequence length
    n_v: usize,
    head_dim: usize,
    n_k: usize,
    eps: f32,
) -> Result<(), MetalError>
```

Dispatch :
- Grid : (head_dim, n_v) TGs = 128 × 16 = 2048 TGs (same as current)
- Threads/TG : 32 (one simdgroup, same as current)
- TG memory : ~1 KB (q_cache + k_cache for current timestep) — minimal

Kernel body :
```metal
kernel void delta_net_persistent_scan_f32(...) {
    uint head_v = tg_id.y;
    uint row = tg_id.x;
    uint n_v_per_n_k = n_v / n_k;
    uint head_k = head_v / n_v_per_n_k;
    uint state_off = (head_v * head_dim + row) * head_dim;

    // Load state row in registers
    float state_row[HEAD_DIM_MAX];     // 128 floats = 512 bytes
    for (uint c = tiisg; c < head_dim; c += 32) state_row[c] = state[state_off + c];

    for (uint t = 0; t < B; ++t) {
        // Load q_t, k_t, v_t for this head
        float q_c = qkv[t * qkv_stride + ... ];
        float k_c = qkv[t * qkv_stride + key_dim + ... ];
        float v_r = qkv[t * qkv_stride + 2*key_dim + head_v * head_dim + row];

        float gamma = exp(gate_h[t * n_v + head_v]);
        float beta_v = beta_sig[t * n_v + head_v];

        // L2 norms (per head_k, shared by rows)
        float q_ss = simd_sum(q_c * q_c);
        float k_ss = simd_sum(k_c * k_c);
        float inv_q = rsqrt(q_ss + eps);
        float inv_k = rsqrt(k_ss + eps);

        // Decay
        for (uint c = tiisg; c < head_dim; c += 32) state_row[c] *= gamma;

        // Projection (sum over c)
        float proj_partial = 0;
        for (uint c = tiisg; c < head_dim; c += 32) proj_partial += state_row[c] * k_c;
        float proj_r = simd_sum(proj_partial) * inv_k;

        // Delta-rule update
        float delta_eff = beta_v * (v_r - proj_r) * inv_k;
        for (uint c = tiisg; c < head_dim; c += 32) state_row[c] += delta_eff * k_c;

        // Output
        float out_partial = 0;
        for (uint c = tiisg; c < head_dim; c += 32) out_partial += state_row[c] * q_c * (1/sqrt(head_dim));
        if (tiisg == 0) out[t * n_v * head_dim + head_v * head_dim + row] = simd_sum(out_partial) * inv_q;
    }

    // Write final state
    for (uint c = tiisg; c < head_dim; c += 32) state[state_off + c] = state_row[c];
}
```

**Key design point** : state row stays in **registers** for entire B-loop.
No threadgroup memory, no DRAM round-trips for state — only initial load
and final store.

DRAM bytes per layer per chunk :
- Read : qkv_combined (B × conv_dim × 4 ≈ 1.3 MB), gate_h+beta_sig (B × n_v × 8 ≈ 2 KB), state initial (n_v × head_dim² × 4 = 1 MB)
- Write : out (B × n_v × head_dim × 4 ≈ 1 MB), state final (1 MB)
- Total : ~5.3 MB / layer / chunk

Compare current scan_loop (sequential dispatch) :
- Per timestep : reads + writes state row (1 KB), reads q/k/v (~5 KB)
- Total over B=128 : ~700 KB read + ~700 KB write per layer per chunk
- Per layer : ~1.4 MB

Hmm, persistent kernel reads/writes state ONCE (1 MB) vs current
(128 × 1 KB = 128 KB). Persistent reads MORE state but same compute.

Actually, the WIN is dispatch overhead, not DRAM. 384 dispatches → 1
dispatch.

## Acceptance criteria

- [ ] T200.1 (this document) reviewed and validated
- [ ] T200.2 : kernel compiles + parity test passes (max_rel_err < 1e-3)
- [ ] T200.3 : prefill 35B-A3B ≥ 420 t/s (vs 365 baseline = +15%)
- [ ] No regression on 14B / 27B (these don't have SSM)

## Open questions

1. **L2 norm reduction across simdgroup** : the existing kernel does it
   per row but in a persistent kernel, rows of a head share the same q
   and k (which are per-head_k, not per-row). Could compute inv_q and
   inv_k ONCE per (t, head_k) and broadcast — but TGs are per-row, so
   different TGs would compute independently. Could use shared TG memory
   to amortize. TODO : measure impact.

2. **Memory layout of qkv_combined** : currently the concatenation is
   `[qkv_mixed | z]` per token. For persistent scan, we want the q/k/v
   parts contiguous across t for cache locality. May need to reorganize.

3. **conv1d_step before delta_net** : also runs per-timestep currently.
   Could the persistent kernel ALSO do conv1d? Pipeline becomes :
   `pre_norm → 4 in_proj → apply_gate → persistent_scan_with_conv1d → norm_gated → out_proj → residual`
   Saves another ~5 ms per layer.

## Next sessions

- Session N+1 : T200.2 implement persistent kernel, microbench vs scan_loop
- Session N+2 : T200.3 wire into ssm_block_forward_batch, e2e bench
- Session N+3 : if persistent works, attempt true chunk-form (Plan A) for
  additional speedup
