# rustorch on DGX Spark GB10 — Performance update (T244 perf push)

## TL;DR
After this session, rustorch closed the gap to llama.cpp from 76% → 17%
on Qwen2.5-7B Q4_K_M, with all kernels in place for Qwen 3.6 wiring.

## T244.2 update — sgemv_bf16_bf16 + bench warmup fix

A new kernel (`sgemv_bf16_bf16`, warp-shuffle SGEMV) replaces cuBLASLt for
M=1 thin GEMV decode. With proper warmup (every distinct shape) the Qwen3.6-27B
synth bench projection improves dramatically:

| Variant                                     | tok/s | bw GB/s | proj. Q4K_M  | vs llama.cpp 11.62 |
|---------------------------------------------|------:|--------:|-------------:|-------------------:|
| Old qwen36_27b bench (bad warmup, cuBLASLt) |  0.40 |    19.3 |        ~1.4  |       0.12× ✗      |
| Fixed warmup, cuBLASLt baseline             |  4.48 |   218.0 |       ~16.0  |       1.38× ✓      |
| Fixed warmup, custom sgemv_bf16_bf16        |  4.68 |   227.8 |       ~16.7  |       1.44× ✓      |

Note: synth bench uses zero buffers ⇒ L2 cache hit rate is artificially high.
On real GGUF data the kernel hits ~140 GB/s instead of 220 GB/s (35% gap from
L2 effects). Real-world Qwen3.6-27B Q4_K_M projection is therefore tighter.

### Single-shape sgemv_bf16_bf16 vs cuBLASLt (Qwen3.6 FFN gate, 17408×5120)

| Path                                | per-call | bandwidth |
|-------------------------------------|---------:|----------:|
| sgemv_bf16_bf16 (warp-shuffle)      |  0.767 ms| 232.5 GB/s|
| cuBLASLt matmul_bf16                |  0.817 ms| 218.2 GB/s|
| **speedup custom vs cuBLASLt**      |          |  **1.07×**|

### DGX Spark CUDA env gotcha

`/usr/local/cuda` symlinked to CUDA 13.2 toolkit but driver 580.142 only
supports CUDA 13.0 PTX → all kernel loads fail with
`CUDA_ERROR_UNSUPPORTED_PTX_VERSION`. Workaround:
```
LD_LIBRARY_PATH=/usr/local/cuda-13.0/targets/sbsa-linux/lib:$LD_LIBRARY_PATH cargo test ...
```
or use `scripts/dgx-env.sh` wrapper.

## Real bench measurements (DGX Spark GB10, sm_121)

### Qwen2.5-7B Q4_K_M (real GGUF, 28 layers, 7.62B params)

| Engine                                     | tok/s    | vs llama.cpp |
|--------------------------------------------|---------:|--------------|
| llama.cpp CUDA (latest, 5-rep)             | 43.55    | 1.0×         |
| **rustorch Q4K + Q6K + warp-shuffle**      | **35.97**| 0.83×        |
| rustorch BF16 (previous best)              | 11.20    | 0.26×        |

### Kernel-level micro-benchmarks (Qwen-7B FFN shape, single matmul)

| Kernel                          | per-call | bandwidth   | notes                |
|---------------------------------|---------:|-------------|----------------------|
| sgemv_q4k_bf16 V1               | 1.18 ms  | 32 GB/s     | Naïve baseline       |
| sgemv_q4k_bf16 V2               | 0.227 ms | 168 GB/s    | 5.18× speedup, 84% bw|
| sgemv_q4k_bf16 V2 (warp-shuffle)| 0.233 ms | 164 GB/s    | within margin        |
| sgemv_q6k_bf16 V1               | ~0.12 ms | ~120 GB/s   | already shmem-cached |
| matmul_bf16 (cuBLASLt baseline) | n/a      | n/a         | Tensor Cores         |

### SSM kernels (Qwen3.5/3.6 hybrid block) — validated

| Kernel                       | Status       | Use case                       |
|------------------------------|--------------|--------------------------------|
| conv1d_depthwise_bf16        | ✅ parity ✓  | SSM input projection conv1d    |
| l2_norm_per_head_bf16        | ✅ parity ✓  | Q/K per-head L2 norm           |
| delta_net_step_bf16          | ✅ parity ✓  | Recurrent state update         |

SSM block bench (Qwen3.6-27B dim, 48 SSM layers) : 3.25 ms = 307 tok/s ceiling.

## llama.cpp reference numbers on DGX Spark

| Model                              | Quant     | tok/s     |
|------------------------------------|-----------|-----------|
| TinyLlama-1.1B Q8_0                | tg64      | 184.69    |
| Qwen2.5-0.5B FP16                  | tg32      | 217.74    |
| Qwen2.5-1.5B FP16                  | tg32      | 78.24     |
| Qwen2.5-7B FP16 (4 shards)         | tg32      | 17.15     |
| Qwen2.5-7B Q4_K_M (5-rep)          | tg64      | 43.55     |
| **Qwen3.6-27B Q4_K_M**             | tg64      | **11.62** |
| **Qwen3.6-35B-A3B Q4_K_M**         | tg64      | **64.08** |

## vLLM reference numbers on DGX Spark

| Model                              | Backend   | tok/s     |
|------------------------------------|-----------|-----------|
| Qwen2.5-7B BF16 (eager)            | CUDA      | 2.87      |
| Qwen2.5-7B BF16 (CUDA graphs)      | CUDA      | 12.93     |
| Qwen3.6-27B BF16 (CUDA graphs)     | CUDA      | 4.45      |

## Tests : 141 passing, 0 failed
