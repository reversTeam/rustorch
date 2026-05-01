# Competitive analysis — `candle`

| Field | Value |
|-------|-------|
| Repository | [huggingface/candle](https://github.com/huggingface/candle) |
| Maintainer | Hugging Face (commercial — funded team, led by Laurent Mazare) |
| Reference version | 0.8.x (most recent stable at the time of writing) |
| License | MIT OR Apache-2.0 |
| Mission | Minimalist ML framework in pure Rust focused on **inference + serverless** |
| Status as competitor | **Closest mission-adjacent peer** — overlaps WASM/inference, diverges on training |

## Tensor model

candle uses **runtime-tracked shapes** (no const generics) — much closer to
PyTorch ergonomics than burn:

```rust
pub struct Tensor(Arc<Tensor_>);

struct Tensor_ {
    id: TensorId,
    storage: Arc<RwLock<Storage>>,
    layout: Layout,
    op: BackpropOp,        // None for leafs, Some(Op) for results
    is_variable: bool,
    dtype: DType,
    device: Device,
}
```

- `Tensor` is a thin handle — cheap to clone (`Arc` bump).
- `Storage` is wrapped in `Arc<RwLock<...>>` to allow shared backing
  buffers across views *and* support in-place mutation when the lock is
  held exclusively.
- `Layout` carries shape, strides, start_offset; contiguity is recomputed
  on demand (no cached flag).
- The backprop link (`op: BackpropOp`) is embedded directly in the tensor
  rather than recorded on a separate tape — see Autograd section.

**Mapping to rustorch:** Aligns with RFC-0001 Decision 1 (runtime shapes)
and RFC-0002's `Tensor { storage, layout, autograd, device }`. The biggest
divergence is the `RwLock` around storage: candle pays a synchronization
cost on every read; rustorch's `Arc<StorageInner>` + version counter
(P1.1 — VersionCounter) achieves the same in-place safety without lock
overhead.

References:
- `candle-core/src/tensor.rs` (Tensor + Tensor_ struct)
- `candle-core/src/storage.rs` (Storage enum)
- `candle-core/src/layout.rs` (Layout)

## Storage abstraction

`Storage` is a flat enum:

```rust
pub enum Storage {
    Cpu(CpuStorage),
    Cuda(CudaStorage),
    Metal(MetalStorage),
}
```

- No `Wgpu` variant in core 0.8 — wgpu support is experimental and lives
  out-of-tree (community fork).
- `CpuStorage` is itself an enum dispatching on dtype:
  `U8(Vec<u8>) | U32(Vec<u32>) | I64(Vec<i64>) | BF16(...) | F16(...) | F32(...) | F64(...)`.
  Each variant owns its `Vec<T>` directly — no aligned-allocator; allocations
  use the system allocator via `Vec`.
- Promotion to `Arc<RwLock<Storage>>` happens at the `Tensor_` level.

**Mapping to rustorch:** Same broad shape as RFC-0004 (`Storage` enum at
the Tensor boundary). Differences:

- candle's `CpuStorage` carries dtype as an enum variant (sum-type over
  dtypes). rustorch keeps a single byte buffer + a separate `Dtype` field
  in `Layout` (RFC-0002 Decision 2). The candle approach has stronger
  type guarantees but forces `match` arms in every kernel; ours
  centralizes dispatch in the codegen pipeline (RFC-0005).
- candle does not enforce 64B alignment — they rely on `Vec<T>`'s natural
  alignment. SIMD codepaths handle misalignment per-call. We aim for
  uniform 64B alignment to simplify kernel writing.

References:
- `candle-core/src/cpu_backend/mod.rs::CpuStorage`
- `candle-core/src/cuda_backend/mod.rs::CudaStorage`
- `candle-core/src/metal_backend/mod.rs::MetalStorage`

## Device enum vs trait Backend trade-off

candle deliberately picks a **Device enum** (concrete dispatch) over a
`trait Backend` (parametric dispatch):

```rust
pub enum Device {
    Cpu,
    Cuda(CudaDevice),
    Metal(MetalDevice),
}
```

- Every op pattern-matches on `Device` and calls into the appropriate
  backend module (`cpu_backend::matmul`, `cuda_backend::matmul`, ...).
- Adding a new backend means: new variant + new module + edit ~150 match
  arms across the workspace.
- Pros: simplicity, predictable dispatch, no generics infecting user code.
- Cons: closed set — a third party cannot ship an out-of-tree backend
  without forking candle.

**Mapping to rustorch:** RFC-0004 Decision A picks `trait Backend`
(parametric, open-set). The reasoning:

- rustorch's WASM-first commitment means we expect community backends
  (wgpu variants, future WebNN) — closed-enum dispatch would make
  contributions painful.
- The 150-method trait surface is mitigated by sub-traits and the codegen
  pipeline (RFC-0005). burn already proves this scales.

candle's choice is internally consistent for an **inference-first**
project where backends are vendor-curated; for a training-and-research
project we want the trait flexibility.

Reference: `candle-core/src/device.rs`.

## Autograd — re-trace, **not** tape

candle's autodiff is the most architecturally distinct piece. There is
**no thread-local tape**. Instead, every op that produces a tensor stores
a `BackpropOp` on the resulting tensor:

```rust
pub struct BackpropOp(Option<Op>);

pub enum Op {
    Binary(Tensor, Tensor, BinaryOp),
    Unary(Tensor, UnaryOp),
    Matmul(Tensor, Tensor),
    Reshape(Tensor),
    /* ~50 variants */
    CustomOp(Box<dyn CustomOp>),
}
```

When `tensor.backward()` is called, candle walks the `Op` graph backwards
starting from the output tensor, accumulating gradients into a
`HashMap<TensorId, Tensor>`:

```rust
// candle-core/src/backprop.rs (simplified)
pub fn backward(&self) -> Result<GradStore> {
    let sorted_nodes = self.sorted_nodes()?;  // topological sort
    let mut grads = GradStore::new();
    grads.insert(self.id(), Tensor::ones_like(self)?);
    for node in sorted_nodes {
        match node.op() {
            Op::Binary(lhs, rhs, BinaryOp::Mul) => {
                let grad = grads.remove(&node.id())?;
                grads.or_insert(lhs.id(), || grad.broadcast_mul(rhs))?;
                grads.or_insert(rhs.id(), || grad.broadcast_mul(lhs))?;
            }
            /* ... ~50 op variants ... */
        }
    }
    Ok(grads)
}
```

**Re-trace** in candle's vocabulary means: backward re-walks the graph
constructed during forward by following `BackpropOp` edges; it does not
re-execute the forward pass. (This terminology occasionally confuses
PyTorch users — there is no tape replay; it is graph traversal.)

### Concrete example — quadratic loss

```rust
let x = Var::from_tensor(&Tensor::new(&[3f32], &Device::Cpu)?)?;
let y = (&x * &x)?.sum_all()?;        // y = x²
let grads = y.backward()?;            // walks Op::Binary back
let dy_dx = grads.get(&x).unwrap();   // 2*x = 6
```

- During forward, `&x * &x` produces a tensor with
  `op = Op::Binary(x, x, Mul)`.
- `y.backward()` triggers `sorted_nodes()` + the match-on-Op loop above.
- No tape, no `requires_grad` flag on every tensor — only `Var` (the
  candle equivalent of a leaf parameter) is marked as gradient-bearing.

### `Var` vs `Tensor`

`Var` is a separate type wrapping `Tensor` with the `is_variable: true`
flag set. Only `Var`s receive gradients in the resulting `GradStore`.
This is significantly more restrictive than PyTorch's
`requires_grad=true` on any tensor.

**Mapping to rustorch:** RFC-0003 picks a thread-local tape. The trade-offs:

| dimension                           | candle (re-trace) | rustorch (tape)        |
|-------------------------------------|-------------------|------------------------|
| memory: forward keeps grad graph?   | yes (BackpropOp on every result) | yes (Node on tape) |
| higher-order grads                  | nested re-traces (works, but verbose) | `create_graph: true` flag (PyTorch-faithful) |
| `no_grad()` scope                   | not natively supported (must use plain `Tensor` rather than `Var`) | thread-local guard |
| inference-only path overhead        | none (no `Var`s = no `Op`s recorded) | tiny (skip tape on disabled flag) |
| in-place mutation safety            | RwLock + recompute | version counter snapshot |

candle's design wins on **inference-only** scenarios (zero grad-tracking
overhead by construction) and loses on **training ergonomics** (the
`Var`/`Tensor` split is non-PyTorch and clutters user code).

References:
- `candle-core/src/op.rs::Op` (op enum)
- `candle-core/src/backprop.rs::backward` (graph walk)
- `candle-core/src/variable.rs::Var` (leaf wrapper)

## Module system + parameter management

candle has **no `#[derive(Module)]`**. Instead, it offers a builder
pattern around `VarBuilder`:

```rust
pub trait Module {
    fn forward(&self, xs: &Tensor) -> Result<Tensor>;
}

// Construct with explicit naming
let vs = VarBuilder::from_safetensors(&["model.safetensors"], DType::F32, &dev)?;
let fc1 = candle_nn::linear(784, 256, vs.pp("fc1"))?;
let fc2 = candle_nn::linear(256, 10, vs.pp("fc2"))?;
```

- `VarBuilder` traverses a parameter tree by string prefix (`vs.pp("fc1")`
  → looks for `fc1.weight`, `fc1.bias` in the safetensors file).
- The user is responsible for assembling layers manually; there is no
  auto-derived `parameters()` collector.
- Saving requires another `VarMap` round-trip — there is no
  `module.save("path.safetensors")` shortcut.

**Mapping to rustorch:** RFC-0002 + P1.6 plan a proc-macro
`#[derive(Module)]` that auto-collects `#[parameter]` and `#[buffer]`
fields. This is closer to burn (and PyTorch's `nn.Module` introspection)
and removes the manual prefix juggling candle requires.

References:
- `candle-nn/src/var_builder.rs::VarBuilder`
- `candle-nn/src/linear.rs::linear` (constructor pattern)

## Supported ops

The candle-core op surface is roughly **150 ops**, weighted heavily
toward **inference**:

- All elementwise (add, sub, mul, div, pow, neg, abs, sqrt, exp, log,
  trig, activations).
- Reductions (sum, mean, max, min, argmax, var, std).
- Linalg (matmul, broadcast_matmul, conv1d/2d, conv_transpose1d/2d,
  pooling, layer_norm, batch_norm, group_norm, rms_norm, softmax, gelu,
  silu, attention via separate `flash-attn` kernel).
- Indexing (gather, scatter, index_select, narrow, slice).
- Embedding, embedding_bag.

Notable gaps relative to PyTorch:
- **No QR / SVD / Cholesky / matrix inverse / eigendecomposition**
  in core. Some linalg ops are deferred to user-side via direct
  cuSOLVER FFI.
- **No CTC loss** in core (an issue for ASR training).
- **No Einstein summation parser** — manual unsqueeze/permute required.
- **Limited custom autograd Functions** — possible via `CustomOp` trait
  but the surface is small (forward + backward only, no save_for_backward
  in the PyTorch sense).
- **No `nn.RNN/LSTM/GRU` natively** — community modules exist but core
  ships only `lstm`/`gru` cells, not full unrolled layers.

Coverage estimate: **70% of inference needs** met out-of-the-box; **45%
of training needs** met (loss functions, optimizer integration are
thin).

References:
- `candle-core/src/op.rs`
- `candle-nn/src/lib.rs`

## Backends supported

| backend | crate | status |
|---------|-------|--------|
| CPU (scalar + SIMD) | `candle-core` | stable |
| CUDA (cuBLAS + custom kernels via cudarc) | `candle-core/cuda_backend` | stable |
| Metal (Apple Silicon) | `candle-core/metal_backend` | stable, well-maintained |
| wgpu / WebGPU | community fork (`candle-wgpu`) | experimental, **not** in main |
| ROCm | not officially supported | n/a |

The Metal backend is notably **more mature** than burn's — Apple Silicon
is a first-class citizen for HF inference deployments.

**Mapping to rustorch:** RFC-0006 prioritizes wgpu/WebGPU as the cross-
platform GPU baseline. candle's lack of a first-party wgpu backend is a
real differentiator for rustorch's WASM-first goal.

## WASM support — measured

candle ships first-party `candle-wasm-examples` with several working
demos:

| demo | model | bundle size after wasm-opt -Oz |
|------|-------|--------------------------------|
| Llama 2 (tiny) | TinyLlama-1.1B q4_0 quantized | ~6 MB code + ~600 MB weights |
| Whisper (tiny.en) | encoder + decoder | ~4 MB code + ~75 MB weights |
| BERT-mini | embeddings classification | ~3 MB code + ~17 MB weights |
| YOLO v8n | detection | ~4 MB code + ~12 MB weights |

Notes:
- All examples use the **CPU** path (WASM SIMD via `wasm32-simd128`),
  **not** WebGPU. Performance is workable for small models but not
  competitive with native.
- No first-party WebGPU example in mainline. Community efforts exist
  but bit-rot is observed.
- Bundle sizes do **not** strictly enforce `<10MB` budget — Llama
  example is borderline.

**Mapping to rustorch:** RFC-0006 commits to **WebGPU + WASM-SIMD
fallback** with a hard `<10 MB` code-size budget enforced in CI. This is
the largest single differentiator vs candle.

References:
- `candle-wasm-examples/llama2-c/`
- `candle-wasm-examples/whisper/`
- `candle-wasm-examples/bert/`

## Inference vs training capability

| dimension | inference | training |
|-----------|-----------|----------|
| speed (single-GPU) | ≈ PyTorch (~5% slower on Llama-7B fp16) | unmeasured at scale |
| memory efficiency  | excellent (rms_norm fused, kv-cache support) | adequate (no grad checkpointing in core) |
| optimizer support  | n/a | SGD, AdamW, **no** Lion/Adafactor in core |
| LR schedulers      | n/a | Cosine, Linear; **no** WarmupCosine, OneCycle |
| mixed precision    | bf16/fp16 forward OK | autocast scope **not** supported, no GradScaler |
| distributed        | none | none in core |
| checkpointing      | safetensors load/save | adequate |
| dataloader         | n/a | minimal — example-level pipelines, no `DataLoader<D>` |

**Verdict:** candle is a **production-grade inference framework** with
training as a possibility, not a focus. The HuggingFace org dogfoods it
for serving, not for training new architectures.

## Famous models supported

candle ships official examples for:
- **LLM:** Llama 2/3, Mistral, Mixtral 8x7B, Phi-2/3, Falcon, GPT-2,
  Gemma, Qwen, Yi.
- **Vision:** ResNet, ViT, EfficientNet, YOLO v8, Segment Anything (SAM),
  DINOv2.
- **Multimodal:** CLIP, SigLIP, Stable Diffusion (1.5, XL, Turbo,
  SD3 ControlNet), BLIP.
- **Speech:** Whisper (all sizes), MetaVoice, Bark.
- **Specialized:** RWKV v5, Mamba, T5, BERT, all-MiniLM (sentence
  embeddings), Marian (translation).

This is the **largest curated zoo** in the Rust ecosystem and a
significant adoption advantage for HuggingFace's flywheel. burn ships
fewer official model examples; rustorch will start far behind on this
axis until the HuggingFace compatibility layer (separate plan) lands.

## Maturity

- GitHub stars (Q1 2026): ~17k (more than burn).
- Production users: HuggingFace itself (text-generation-inference uses
  candle for serverless endpoints), several startups in the model-serving
  space.
- Funded development since mid-2023 — primary developer Laurent Mazare
  is full-time on it.
- 280+ contributors, weekly releases.
- Issue triage is fast (median first response < 24 h).

## Code size

`tokei` on the candle workspace at 0.8.x:

- `candle-core`: ~14k LoC (vs burn-tensor 25k)
- `candle-nn`: ~6k LoC
- `candle-transformers`: ~25k LoC (model definitions)
- `candle-examples` + `candle-wasm-examples`: ~30k LoC
- All tracked crates: ~80k LoC (similar order to burn)

The core (`candle-core` + `candle-nn`) is **~10× smaller than PyTorch's
ATen+autograd** (which is candle's headline marketing line) and roughly
half the size of burn's tensor+autodiff layers. Most of candle's bulk
sits in `candle-transformers` — model implementations, not framework code.

## Limitations vs rustorch goals

1. **No first-party WebGPU backend.** WASM examples use CPU+SIMD only;
   rustorch's RFC-0006 commits to WebGPU as the WASM baseline.
2. **`Var` / `Tensor` split is non-PyTorch.** rustorch's `requires_grad`
   on any tensor is more faithful (RFC-0003).
3. **No `#[derive(Module)]`.** Manual `VarBuilder.pp(...)` prefix
   plumbing is verbose; rustorch's planned proc macro removes it.
4. **No autocast / GradScaler.** Mixed-precision training is a manual
   cast dance; rustorch P-Mixed-Precision plan provides PyTorch-faithful
   `autocast(Mixed::Bf16, || { ... })`.
5. **Linalg gaps.** No QR, SVD, Cholesky, einsum parser, CTC loss,
   eigendecomposition. rustorch P1.4 includes these as v1 requirements.
6. **No native distributed.** No DDP, FSDP, ProcessGroup. rustorch
   Phase 5 commits to first-party DDP+FSDP.
7. **DataLoader is minimal.** No bucketing samplers, no rich augmentation
   library; rustorch P1.8 + P1.9 land both.
8. **`RwLock<Storage>`** synchronization overhead per op (small but
   measurable on small-tensor hot paths). rustorch's version-counter
   approach achieves the same safety guarantee without locks.

## Could we contribute instead of competing?

**Partially.** Concrete contribution avenues to candle:

- A first-party WebGPU backend (`candle-wgpu`) hardened from the existing
  community fork. Would be welcomed — Laurent has stated openness in
  past issues.
- A WASM `<10MB` budget CI matrix entry.
- Linalg gap-filling (QR, SVD, Cholesky) — small focused PRs.

**However**, the architectural divergence on:
- the `Var`/`Tensor` split (autograd model),
- the missing tape (debug tooling story),
- the `Device`-enum-vs-`trait Backend` direction (extension model),

means contributing falls short of rustorch's mission. We can borrow
candle's safetensors loader, model zoo strategy, and Metal backend
implementation lessons; we cannot bend candle's autograd model into
rustorch's shape without a fork.

## Quick PR / feasibility scan

Open issues reviewed (sample of ~25 most recent):
- ~40% are model-implementation requests (new HF model X support).
- ~30% are bug reports (often fixed within days).
- ~20% are feature requests (WebGPU, FSDP, etc.) — these tend to sit.
- ~10% are ecosystem questions (Python bindings, ONNX import, etc.).

A focused PR adding (say) a missing linalg op is mergeable in a week.
An architectural PR (parallel autograd implementation) would face a
high review bar — Laurent is opinionated about keeping core minimal.

## Verdict

candle is the **most credible Rust inference framework**, with a clearer
adoption trajectory than burn (more stars, HF flywheel). For rustorch's
mission, candle is:

1. **Prior art to study** for safetensors interop, Metal backend
   patterns, and the model zoo curation strategy.
2. **A peer competitor** for WASM mindshare — rustorch must beat candle
   on bundle size + WebGPU performance to differentiate.
3. **Not a contribution target** — the autograd architecture and
   `Var`/`Tensor` split are too divergent from RFC-0003 to bridge.

We will treat candle as **the inference baseline to match-or-beat** at
v1.0 release. The training story (autograd, distributed, mixed precision,
optimizer suite) is where rustorch's value sits — candle deliberately
chose not to compete there.
