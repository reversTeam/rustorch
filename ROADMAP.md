# Rustorch Roadmap

Phase-by-phase delivery plan. Status reflects merged code; in-flight branches
are tracked in the Project Orchestrator's Plan graph.

## Phase 0 — Foundation

- ✅ Tensor core (storage, dtype, device tag, view ops, layout)
- ✅ Autograd engine (Tape, Variable, Node, backward())
- ✅ Backend trait (CpuBackend reference impl)

## Phase 1 — CPU completeness

- ✅ Element-wise + linear-algebra ops
- ✅ Reductions, softmax, loss functions
- ✅ Conv2d / BatchNorm / MaxPool / Embedding (ResNet path)
- ✅ Transformer building blocks (Linear rank-N, MHA, RMSNorm, SwiGLU,
  PositionalEncoding, l2_normalize)

## Phase 2 — Performance

- ✅ Gemm crate integration (faer-rs) for f32/f64 matmul
- ✅ SIMD lanes (AVX2/AVX-VNNI/NEON sdot) where applicable
- ✅ Pool / cache / staging infra for the Wgpu backend

## Phase 3 — Wgpu (cross-platform GPU)

- ✅ Standalone `WgpuBackend` with ~18 forward kernels (Phase 3.0)
- ✅ Wgpu autograd wiring (Phase 3.Y, this plan):
  - Phase 0 audit + decisions documented
  - Phase A plumbing (Variable::device, wgpu_backend singleton,
    Backend trait extension, `impl Backend for WgpuBackend`)
  - Phase B forward GPU end-to-end (33-op dispatch, parity tests)
  - Phase C backward GPU dispatch (Node device field, backward routing
    via `pick_backend(self.device)`, parity tests forward+backward)
  - Phase D device-aware optimisers (11 optims + clip_grad_norm_)
  - Phase E `Module::to_device()` for CPU↔GPU param transfer
  - Phase F examples (`mnist_wgpu`, `cpu_vs_wgpu_train`) + this docs update
- 📋 Phase 3.5 (perf follow-up): native WGSL backward kernels
  (relu_grad, sigmoid_grad, softmax_grad fused, layernorm_grad,
  scatter_add atomics, attention_grad). Currently CPU-fallback
  via round-trip — correct but sub-optimal.
- 📋 Phase 3.6 (perf follow-up): Storage Option A — `enum Storage
  { Cpu(Vec<f32>), Wgpu(WgpuStorage) }`. Removes the per-op
  host↔device round-trip and unlocks the wgpu speedup ceiling.

**Same-machine perf reference** (Apple M4 Max, Linear 1024×1024 + MSE + AdamW,
ms/step lower is better):

| Stack | ms/step | Notes |
|---|---:|---|
| PyTorch MPS | 0.89 | gold standard target |
| PyTorch CPU | 1.75 | MKL/Accelerate + multithreading |
| rustorch CPU | 12.81 | gemm-rs + rayon, single-thread bench |
| rustorch Wgpu | 22.30 | Storage Option B round-trip dominates |

Closing the rustorch Wgpu → PyTorch MPS gap is the explicit goal of
Phases 3.5 + 3.6.

## Phase 4 — CUDA

- 🚧 NVIDIA-specific backend via `rustorch-cuda` crate (cuBLAS, cuDNN,
  NCCL primitives are scaffolded; full `impl Backend for CudaBackend`
  pending).
- 📋 Drop-in once Storage Option A lands — same dispatch chain reuses
  the existing autograd / nn / optim layers.

## Phase 5 — Distributed training

- 📋 NCCL ProcessGroup (already scaffolded) wired into `optim` and
  `nn` for data-parallel training.
- 📋 FSDP (sharded params, sharded grads, sharded optimizer state).
- 📋 Pipeline parallelism via the `runner` RPC infra.

## Out of scope (today)

- Mixed precision GPU (autocast on CPU only)
- Sparse tensors
- Quantisation training (PTQ inference shipped in Phase 1.4)

## Contributing

See `docs/rfcs/` for active architectural RFCs. The Project Orchestrator
MCP plans (`388788ac…` for P3.Y) capture the granular task / step graph
with decisions and parity-test commits.
