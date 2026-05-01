---
id: 0003
title: Autograd model — tape, Node, backward execution
status: Draft
date: 2026-05-01
authors:
  - rustorch team <rfc@rustorch.dev>
supersedes: []
superseded-by: []
---

# RFC-0003: Autograd model — tape, Node, backward execution

## Summary

Reverse-mode automatic differentiation in rustorch is **tape-based and
thread-local** (per RFC-0001 Decision 3). Each forward op that involves at
least one `requires_grad=true` tensor pushes an `Arc<dyn Node>` onto the
active thread's `Tape`. `Tensor::backward()` walks the tape in reverse
topological order, calling each `Node::apply` with the upstream gradient
and accumulating leaf gradients via an `AccumulateGrad` sink. Higher-order
gradients are supported via `backward(create_graph: true)` and the explicit
`grad(outputs, inputs, create_graph)` API.

## Motivation

The autograd engine is the single most-load-bearing subsystem after the
tensor type. Get it wrong and:

- **Higher-order grads break** (Hessian, MAML, GAN gradient penalty cannot run).
- **Custom ops require touching internals** — every researcher who needs an
  exotic gradient writes a fork.
- **Memory blows up** — naïve tape design holds full activations forever
  (PyTorch's "graph already freed" error exists for a reason).
- **Thread safety surprises** — a tape that's accidentally global causes
  spooky action between threads.
- **Numeric anomaly debugging is impossible** without a way to point a finger
  at *which* op produced the first NaN.

PyTorch's autograd engine took ~5 person-years to mature. We deliberately
adopt its design 1-to-1 to avoid relearning the same lessons.

## Constraints

- **Thread-local tape.** Each user-spawned thread has its own tape. Tensors
  cross threads; tapes don't.
- **Eager.** No batched / async backward — `backward()` blocks until done.
  Phase 5 distributed training adds a comm overlap path on top of this.
- **No global state.** All autograd globals live in `thread_local!{ }` so
  unit tests can run in parallel without contaminating each other.
- **Send + Sync** for `Node` impls — `Arc<dyn Node>` must cross threads
  cleanly even though the active tape doesn't.
- **`create_graph: bool`** on `Tensor::backward` and on `autograd::grad`.
  When `true`, the backward op is itself recorded so it can be differentiated.
- **Anomaly mode** behind a thread-local flag: when on, NaN/Inf in any
  produced gradient panics with the file:line of the forward op that
  introduced it.

## Alternatives Considered

### Alternative A — Tape-based, thread-local (chosen)

**Approach.** Each forward op that involves a tensor with `requires_grad=true`
pushes a `Node` onto a thread-local `Tape`. `backward()` traverses in reverse.

**Pros.**
- 1-to-1 mental model with PyTorch.
- Higher-order grads via re-recording.
- Custom ops via the `CustomFunction` trait — additive.

**Cons.**
- Allocation cost per op (`Arc<dyn Node>` heap-box).
- Memory cost = Σ(saved_for_backward).

**Cost.** Reference baseline.

### Alternative B — Trait-based static graph (JAX `jit` style)

**Approach.** Each op constructs a strongly-typed expression node;
`compile()` stages everything into a frozen graph; `grad(fn)` differentiates
the graph algebraically.

**Pros.**
- No allocations during forward.
- Whole-graph optimization opportunities.

**Cons.**
- Eager mode dies; user can't print intermediate tensors.
- Diverges from PyTorch ergonomics — RFC-0001 Decision 1 already rejects this.

**Cost.** Prohibitive for a PyTorch-faithful product.

### Alternative C — Re-trace on each backward

**Approach.** Forward emits no tape; backward re-runs the forward symbolically
with `Var<f32>` placeholders.

**Pros.**
- Zero memory cost during forward.
- No `Arc<dyn Node>`.

**Cons.**
- Forward must be deterministic and re-runnable (no I/O, no in-place state).
- 2× compute per backward call.

**Cost.** Memory savings small relative to activation memory dominated by
gradient checkpointing (Phase 3 separate feature). Compute cost too high.

### Alternative D — Per-tensor `grad_fn` cycle (PyTorch C++ historical)

**Approach.** Each `Tensor` holds its own `grad_fn`; `backward()` walks
backward edges from the output through `grad_fn` pointers. No central tape.

**Pros.**
- No "tape" concept — distributed graph.
- Slightly more cache-friendly walk (locality).

**Cons.**
- Hard to inspect / debug (no flat list).
- Anomaly mode complex (no obvious traversal order).
- Cycle detection needed for in-place ops on shared storage.

**Cost.** Roughly equivalent to A; marginally harder to instrument.

## Decision

We adopt **Alternative A: tape-based, thread-local recording**. Concrete
shape locked here:

```rust
// crates/rustorch-autograd/src/lib.rs

/// Reverse-mode autograd Node. Implementations live in derivatives/* per RFC-0005.
pub trait Node: Send + Sync + 'static {
    fn apply(&self, grad_outputs: &[Tensor]) -> Vec<Tensor>;
    fn next_edges(&self) -> &[Edge];
    fn name(&self) -> &'static str { std::any::type_name::<Self>() }
}

#[derive(Clone)]
pub struct Edge {
    pub node: Arc<dyn Node>,
    pub output_index: usize,
}

/// One entry per forward-op that produced a grad-requiring tensor.
struct TapeEntry {
    node: Arc<dyn Node>,
    output_count: usize,
}

thread_local! {
    static TAPE_STACK: RefCell<Vec<Vec<TapeEntry>>> = RefCell::new(vec![Vec::new()]);
    static GRAD_ENABLED: Cell<bool> = Cell::new(true);
    static DETECT_ANOMALY: Cell<bool> = Cell::new(false);
}

/// User-facing entry points
impl Tensor {
    pub fn backward(&self) -> Result<()> { /* ... */ }
    pub fn backward_with(&self, grad: Tensor, create_graph: bool, retain_graph: bool) -> Result<()> { /* ... */ }
    pub fn detach(&self) -> Tensor { /* clone with grad_fn = None */ }
    pub fn requires_grad_(self, b: bool) -> Tensor { /* … */ }
}

pub fn no_grad<R>(f: impl FnOnce() -> R) -> R { /* GRAD_ENABLED.set(false) for the scope */ }
pub fn with_grad<R>(f: impl FnOnce() -> R) -> R { /* GRAD_ENABLED.set(true) for the scope */ }
pub fn grad(outputs: &[&Tensor], inputs: &[&Tensor], create_graph: bool) -> Result<Vec<Tensor>>;
pub fn set_detect_anomaly(b: bool);
```

The leaf accumulator is a built-in `AccumulateGrad` Node that lands
gradient values into the leaf tensor's atomic `grad` slot.

## Rationale

- **Thread-local** because each test runs in isolation, and concurrent
  forward passes (data parallel, multi-stream) must not interfere.
- **`Arc<dyn Node>`** because Nodes vary in size and shape (matmul saves
  two tensors, ReLU saves a mask, conv2d saves the entire input). A single
  monomorphized graph would be hostile to incremental compile.
- **Tape stack** (vec of vecs) so `with_grad`/`no_grad` and nested forward
  passes (used internally by gradient checkpointing) push/pop cleanly
  rather than leak each other's entries.
- **`create_graph` re-records the backward** because higher-order grads
  must run through the same machinery. We don't have a "second tape";
  the first tape's nodes themselves register their backward op on the
  active tape when called.
- **`AccumulateGrad`** as a single explicit sink Node lets us implement
  multi-path accumulation (z = x + x → x.grad = 2) for free: both edges
  point at the same Node which sums incoming gradients.

## How does PyTorch do it?

- `torch/csrc/autograd/engine.cpp:120-265` — `Engine::execute_with_graph_task`
  is the exact analog of our backward traversal. We adopt the dependency-count
  approach (each op counts incoming edges, decrements on grad arrival,
  fires when zero).
- `torch/csrc/autograd/python_function.cpp:200-380` — `THPFunction` is
  PyTorch's `torch.autograd.Function` extension. Our `CustomFunction` trait
  mirrors it: a user implements `forward(ctx, inputs)` and
  `backward(ctx, grad_outputs)`.
- `torch/csrc/autograd/saved_variable.h` — `SavedVariable` with a `version`
  snapshot taken at save-time, panicking at unpack time on mismatch. Our
  type of the same name has the same invariants.
- `torch/csrc/autograd/anomaly_mode.cpp` — sets a thread-local that ops
  consult on emit; we replicate this with `DETECT_ANOMALY` thread-local.
- `aten/src/ATen/native/native_functions.yaml` — declares the backward
  formula per op. We emit the equivalent via RFC-0005 codegen from
  `derivatives.yaml`.

## How does burn / candle do it?

- **burn-autodiff** uses a tape but keyed by the `Backend` trait (since
  burn tensors are parametric). Same conceptual model as ours, more
  generic-heavy.
- **candle** has a minimal autograd: each `Tensor` stores an `Op` enum
  describing how it was produced; `backward()` walks the op tree. Works for
  inference + simple training, doesn't scale to 35+ derivatives + custom
  functions cleanly. We diverge: full PyTorch parity is the goal.
- **dfdx** records type-level information into the gradient pipeline —
  again rejected by RFC-0001 Decision 1.

## Migration plan

Foundational RFC. Future RFCs that revise the tape model (e.g., switching
to a graph-tracked model for deferred execution à la PyTorch 2.x's
TorchDynamo) must explicitly supersede this and provide a migration
shim that re-implements the existing user-facing surface (`backward`,
`no_grad`, `grad`, `CustomFunction`) atop the new model.

## Open questions

- [ ] Should `backward(create_graph=true)` be the *default* like JAX, or
      opt-in like PyTorch (`create_graph=false`)? Current default: opt-in,
      matching PyTorch.
- [ ] Should `retain_graph` be a separate flag from `create_graph`?
      PyTorch has both. Current decision: yes, both.
- [ ] **Buffers** (non-trainable state, like BatchNorm running stats):
      should mutating them require a no_grad scope? Current decision: yes,
      via `Buffer` newtype that owns a `no_grad` impl block.
- [ ] **Multi-output ops** (e.g., `topk` returns (values, indices)):
      what does `indices.backward()` do? Current decision: indices have
      `requires_grad = false` permanently, even if input did.
- [ ] Should `Tape` support **forward-mode** AD as a future extension?
      Probably yes — JAX-style `vjp`/`jvp` decoration on the same Node trait.
- [ ] **`set_grad_enabled`** as a public function vs only `no_grad`/`with_grad`?
      Current decision: only the scoped API. No global toggles.
- [ ] **`save_for_backward`** vs `save_for_backward_inputs` (a PyTorch
      historical mistake): we expose only the former.

## References

- PyTorch autograd engine — `torch/csrc/autograd/engine.cpp`
- PyTorch `SavedVariable` — `torch/csrc/autograd/saved_variable.h`
- "Automatic Differentiation in Machine Learning: a Survey" — Baydin et al.,
  JMLR 2017.
- "PyTorch Autograd Internals" — Edward Z. Yang's talks (2020).

## Decision matrix

| Aspect | A tape (chosen) | B static graph | C re-trace | D per-tensor cycle |
|--------|:--------------:|:--------------:|:----------:|:------------------:|
| Eager mode | yes | no | yes | yes |
| Higher-order grads | yes | yes | yes | hard |
| Memory at fwd | medium (Σ saved) | low | very low | medium |
| Compute at backward | 1× | 1× | 3× | 1× |
| Custom ops | trivial | hard | hard | trivial |
| Anomaly mode | trivial | hard | trivial | hard |
| PyTorch fidelity | 1-to-1 | none | partial | 1-to-1 |
| Decision | **chosen** | rejected | rejected | rejected |
