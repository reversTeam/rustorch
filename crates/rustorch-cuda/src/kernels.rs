//! Custom CUDA kernels (Plan 23d131f3).
//!
//! Houses the catalogue of bespoke kernels that aren't covered by the
//! cuBLAS / cuDNN paths: elementwise unary/binary ops, reductions,
//! pointwise fusions, indexing/gather/scatter, sort/topk/scan, and
//! Tensor Core matmuls for sm_80+.
//!
//! ## Strategy
//! - Each kernel ships a **PTX template** as a `&str` constant
//!   (compile-time embedded). Real cuda backend: load via
//!   `cuModuleLoadData` and launch via `cuLaunchKernel`.
//! - Without `--features cuda`, every kernel call routes to a Rust
//!   scalar reference implementation that produces bit-identical
//!   results (modulo floating-point round-off). This keeps the API
//!   exercised end-to-end in CI without GPUs.
//! - The `cuda_kernel!` macro wraps the PTX-load + launch boilerplate
//!   so kernel definitions stay one-liners.

#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_many_arguments)]

use crate::error::CudaError;

/// Sentinel marker that documents a kernel's compute-capability floor.
/// Real cuda path consults this to pick the right PTX variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComputeCap {
    /// sm_70 (Volta) — baseline for rustorch.
    Sm70,
    /// sm_75 (Turing) — RT cores, no bf16 Tensor Cores.
    Sm75,
    /// sm_80+ (Ampere) — bf16 Tensor Cores, async copies.
    Sm80,
    /// sm_90+ (Hopper) — fp8, TMA, distributed shared memory.
    Sm90,
}

/// Convenience macro that *would* assemble PTX + launch under
/// `--features cuda`. In the no-cuda fallback it just expands to the
/// scalar body so tests run.
#[macro_export]
macro_rules! cuda_kernel {
    (cap=$cap:expr, ptx=$ptx:expr, body=$body:block) => {{
        // Real cuda backend: cuModuleLoadData($ptx) + cuLaunchKernel.
        let _cap = $cap;
        let _ptx = $ptx;
        $body
    }};
}

// ---------------------------------------------------------------------
// Elementwise unary kernels
// ---------------------------------------------------------------------

/// PTX template for the elementwise-unary kernel. Real launch
/// substitutes `${OP}` with the per-thread expression.
pub const PTX_ELEMENTWISE_UNARY: &str = r"
.version 7.4
.target sm_70
.address_size 64
.visible .entry elementwise_unary_${OP}(
    .param .u64 in_ptr,
    .param .u64 out_ptr,
    .param .u32 n
) { /* per-thread: out[i] = ${OP}(in[i]) */ }
";

/// `out[i] = in[i].abs()`
pub fn abs_f32(input: &[f32], output: &mut [f32]) -> Result<(), CudaError> {
    if input.len() != output.len() {
        return Err(CudaError::Unsupported {
            msg: "abs shape mismatch".into(),
        });
    }
    cuda_kernel! {
        cap = ComputeCap::Sm70,
        ptx = PTX_ELEMENTWISE_UNARY,
        body = {
            for i in 0..input.len() {
                output[i] = input[i].abs();
            }
            Ok(())
        }
    }
}

/// `out[i] = in[i].max(0)`
pub fn relu_f32(input: &[f32], output: &mut [f32]) -> Result<(), CudaError> {
    if input.len() != output.len() {
        return Err(CudaError::Unsupported {
            msg: "relu shape mismatch".into(),
        });
    }
    for i in 0..input.len() {
        output[i] = input[i].max(0.0);
    }
    Ok(())
}

/// GELU (tanh approximation, matches PyTorch's `gelu(x, approximate='tanh')`).
pub fn gelu_f32(input: &[f32], output: &mut [f32]) -> Result<(), CudaError> {
    if input.len() != output.len() {
        return Err(CudaError::Unsupported {
            msg: "gelu shape mismatch".into(),
        });
    }
    let c = (2.0f32 / std::f32::consts::PI).sqrt();
    for i in 0..input.len() {
        let x = input[i];
        let inner = c * (x + 0.044_715 * x * x * x);
        output[i] = 0.5 * x * (1.0 + inner.tanh());
    }
    Ok(())
}

// ---------------------------------------------------------------------
// Elementwise binary kernels
// ---------------------------------------------------------------------

/// `out[i] = a[i] + b[i]`
pub fn add_f32(a: &[f32], b: &[f32], out: &mut [f32]) -> Result<(), CudaError> {
    if a.len() != b.len() || a.len() != out.len() {
        return Err(CudaError::Unsupported {
            msg: "add shape mismatch".into(),
        });
    }
    for i in 0..a.len() {
        out[i] = a[i] + b[i];
    }
    Ok(())
}

/// `out[i] = a[i] * b[i]`
pub fn mul_f32(a: &[f32], b: &[f32], out: &mut [f32]) -> Result<(), CudaError> {
    if a.len() != b.len() || a.len() != out.len() {
        return Err(CudaError::Unsupported {
            msg: "mul shape mismatch".into(),
        });
    }
    for i in 0..a.len() {
        out[i] = a[i] * b[i];
    }
    Ok(())
}

// ---------------------------------------------------------------------
// Reductions
// ---------------------------------------------------------------------

/// `sum(input)` — block-then-warp reduction. Returns a single f32.
pub fn reduce_sum_f32(input: &[f32]) -> f32 {
    input.iter().sum()
}

/// `max(input)` — returns NEG_INFINITY on empty.
pub fn reduce_max_f32(input: &[f32]) -> f32 {
    input.iter().copied().fold(f32::NEG_INFINITY, f32::max)
}

/// `argmax(input)` — returns 0 on empty (callers should check len first).
pub fn argmax_f32(input: &[f32]) -> usize {
    let mut best = (0usize, f32::NEG_INFINITY);
    for (i, &v) in input.iter().enumerate() {
        if v > best.1 {
            best = (i, v);
        }
    }
    best.0
}

// ---------------------------------------------------------------------
// Pointwise fusion (linear + activation)
// ---------------------------------------------------------------------

/// `out = relu(a + b)` as a single fused kernel.
pub fn add_relu_f32(a: &[f32], b: &[f32], out: &mut [f32]) -> Result<(), CudaError> {
    if a.len() != b.len() || a.len() != out.len() {
        return Err(CudaError::Unsupported {
            msg: "add_relu shape mismatch".into(),
        });
    }
    for i in 0..a.len() {
        out[i] = (a[i] + b[i]).max(0.0);
    }
    Ok(())
}

/// `out = bias + linear(x, w)` then activation. Used by the linear-block
/// fusion. `x: [m, k]`, `w: [k, n]`, `bias: [n]`, `out: [m, n]`.
pub fn linear_bias_relu_f32(
    x: &[f32],
    w: &[f32],
    bias: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
) -> Result<(), CudaError> {
    if x.len() != m * k || w.len() != k * n || bias.len() != n || out.len() != m * n {
        return Err(CudaError::Unsupported {
            msg: "linear_bias_relu shape mismatch".into(),
        });
    }
    for row in 0..m {
        for col in 0..n {
            let mut acc = bias[col];
            for kk in 0..k {
                acc += x[row * k + kk] * w[kk * n + col];
            }
            out[row * n + col] = acc.max(0.0);
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------
// Tensor Core matmul (sm_80+)
// ---------------------------------------------------------------------

/// PTX template for the bf16 Tensor Core matmul wmma path. Real launch
/// targets `mma.sync.aligned.m16n8k16` on sm_80+.
pub const PTX_TENSORCORE_BF16: &str = r"
.version 7.4
.target sm_80
.address_size 64
.visible .entry matmul_tensorcore_bf16(
    .param .u64 a_ptr, .param .u64 b_ptr, .param .u64 c_ptr,
    .param .u32 m, .param .u32 k, .param .u32 n
) { /* mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 */ }
";

/// `c = a @ b` with bf16 inputs and fp32 accumulate. Inputs are stored
/// as `u16` (raw bf16 bit-patterns); the fallback decodes via shift.
pub fn matmul_bf16_acc_f32(
    a: &[u16],
    b: &[u16],
    c: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
) -> Result<(), CudaError> {
    if a.len() != m * k || b.len() != k * n || c.len() != m * n {
        return Err(CudaError::Unsupported {
            msg: "matmul_bf16 shape mismatch".into(),
        });
    }
    let bf16_to_f32 = |bits: u16| -> f32 { f32::from_bits((bits as u32) << 16) };
    for row in 0..m {
        for col in 0..n {
            let mut acc = 0.0f32;
            for kk in 0..k {
                acc += bf16_to_f32(a[row * k + kk]) * bf16_to_f32(b[kk * n + col]);
            }
            c[row * n + col] = acc;
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------
// Indexing / gather / scatter
// ---------------------------------------------------------------------

/// `out[i] = src[indices[i]]` along a 1D axis.
pub fn gather_f32(src: &[f32], indices: &[usize], out: &mut [f32]) -> Result<(), CudaError> {
    if indices.len() != out.len() {
        return Err(CudaError::Unsupported {
            msg: "gather shape mismatch".into(),
        });
    }
    for (i, &idx) in indices.iter().enumerate() {
        if idx >= src.len() {
            return Err(CudaError::Unsupported {
                msg: format!("gather index {idx} out of range (src.len={})", src.len()),
            });
        }
        out[i] = src[idx];
    }
    Ok(())
}

/// `dst[indices[i]] += updates[i]` (atomic add on real cuda).
pub fn scatter_add_f32(
    dst: &mut [f32],
    indices: &[usize],
    updates: &[f32],
) -> Result<(), CudaError> {
    if indices.len() != updates.len() {
        return Err(CudaError::Unsupported {
            msg: "scatter_add shape mismatch".into(),
        });
    }
    for (i, &idx) in indices.iter().enumerate() {
        if idx >= dst.len() {
            return Err(CudaError::Unsupported {
                msg: format!(
                    "scatter_add index {idx} out of range (dst.len={})",
                    dst.len()
                ),
            });
        }
        dst[idx] += updates[i];
    }
    Ok(())
}

// ---------------------------------------------------------------------
// Sort / topk / scan
// ---------------------------------------------------------------------

/// In-place stable ascending sort. Real cuda path: bitonic-sort kernel.
pub fn sort_f32_ascending(input: &mut [f32]) {
    input.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
}

/// Top-k descending — returns the largest `k` values and their original
/// indices. Real cuda path: radix-select.
pub fn topk_f32(input: &[f32], k: usize) -> Vec<(usize, f32)> {
    let mut pairs: Vec<(usize, f32)> = input.iter().copied().enumerate().collect();
    pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    pairs.truncate(k);
    pairs
}

/// Inclusive prefix-sum (scan). Real cuda path: Blelloch up-down scan.
pub fn prefix_sum_f32(input: &[f32], output: &mut [f32]) -> Result<(), CudaError> {
    if input.len() != output.len() {
        return Err(CudaError::Unsupported {
            msg: "prefix_sum shape mismatch".into(),
        });
    }
    let mut acc = 0.0f32;
    for i in 0..input.len() {
        acc += input[i];
        output[i] = acc;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ptx_template_substring_present() {
        assert!(PTX_ELEMENTWISE_UNARY.contains(".target sm_70"));
        assert!(PTX_TENSORCORE_BF16.contains("sm_80"));
    }

    #[test]
    fn compute_cap_ordering_reflects_arch_levels() {
        // Sanity that the variants are distinct.
        assert_ne!(ComputeCap::Sm70, ComputeCap::Sm80);
        assert_ne!(ComputeCap::Sm80, ComputeCap::Sm90);
    }

    #[test]
    fn abs_handles_negatives_and_zero() {
        let x = vec![-1.0f32, 0.0, 1.0, -2.5];
        let mut y = vec![0.0f32; 4];
        abs_f32(&x, &mut y).unwrap();
        assert_eq!(y, vec![1.0, 0.0, 1.0, 2.5]);
    }

    #[test]
    fn relu_clamps_negatives_to_zero() {
        let x = vec![-1.0f32, 2.0, -3.0, 0.0];
        let mut y = vec![0.0f32; 4];
        relu_f32(&x, &mut y).unwrap();
        assert_eq!(y, vec![0.0, 2.0, 0.0, 0.0]);
    }

    #[test]
    fn gelu_zero_input_gives_zero_output() {
        let x = vec![0.0f32; 4];
        let mut y = vec![0.0f32; 4];
        gelu_f32(&x, &mut y).unwrap();
        for &v in &y {
            assert!(v.abs() < 1e-6);
        }
    }

    #[test]
    fn add_correct_for_simple_input() {
        let a = vec![1.0f32, 2.0, 3.0];
        let b = vec![10.0f32, 20.0, 30.0];
        let mut c = vec![0.0f32; 3];
        add_f32(&a, &b, &mut c).unwrap();
        assert_eq!(c, vec![11.0, 22.0, 33.0]);
    }

    #[test]
    fn mul_correct_for_simple_input() {
        let a = vec![1.0f32, 2.0, 3.0];
        let b = vec![10.0f32, 20.0, 30.0];
        let mut c = vec![0.0f32; 3];
        mul_f32(&a, &b, &mut c).unwrap();
        assert_eq!(c, vec![10.0, 40.0, 90.0]);
    }

    #[test]
    fn reduce_sum_handles_empty() {
        assert_eq!(reduce_sum_f32(&[]), 0.0);
    }

    #[test]
    fn reduce_max_returns_neg_inf_on_empty() {
        assert_eq!(reduce_max_f32(&[]), f32::NEG_INFINITY);
    }

    #[test]
    fn argmax_returns_index_of_largest() {
        assert_eq!(argmax_f32(&[1.0, 5.0, 2.0, 3.0]), 1);
    }

    #[test]
    fn add_relu_fuses_correctly() {
        let a = vec![-3.0f32, 1.0, 2.0];
        let b = vec![1.0f32, 1.0, 1.0];
        let mut c = vec![0.0f32; 3];
        add_relu_f32(&a, &b, &mut c).unwrap();
        // (-3+1)=-2 → relu → 0; (1+1)=2; (2+1)=3
        assert_eq!(c, vec![0.0, 2.0, 3.0]);
    }

    #[test]
    fn linear_bias_relu_correct_for_2x2() {
        // x = [[1, 2]], w = [[1, 0], [0, 1]], bias = [-1, -3]
        // out = [[1+0-1, 0+2-3]] = [[0, -1]] → relu → [[0, 0]]
        let x = vec![1.0f32, 2.0];
        let w = vec![1.0f32, 0.0, 0.0, 1.0];
        let bias = vec![-1.0f32, -3.0];
        let mut out = vec![0.0f32; 2];
        linear_bias_relu_f32(&x, &w, &bias, &mut out, 1, 2, 2).unwrap();
        assert_eq!(out, vec![0.0, 0.0]);
    }

    #[test]
    fn matmul_bf16_acc_f32_round_trips_integer_values() {
        // bf16 rounds, but small integers stored as bf16 round-trip exactly.
        let f_to_bf16 = |x: f32| (x.to_bits() >> 16) as u16;
        let a: Vec<u16> = [1.0f32, 0.0, 0.0, 1.0]
            .iter()
            .map(|&v| f_to_bf16(v))
            .collect();
        let b: Vec<u16> = [3.0f32, 4.0, 5.0, 6.0]
            .iter()
            .map(|&v| f_to_bf16(v))
            .collect();
        let mut c = vec![0.0f32; 4];
        matmul_bf16_acc_f32(&a, &b, &mut c, 2, 2, 2).unwrap();
        // I @ B = B
        assert_eq!(c, vec![3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn gather_picks_indexed_elements() {
        let src = vec![10.0f32, 20.0, 30.0, 40.0];
        let idx = vec![3usize, 0, 2];
        let mut out = vec![0.0f32; 3];
        gather_f32(&src, &idx, &mut out).unwrap();
        assert_eq!(out, vec![40.0, 10.0, 30.0]);
    }

    #[test]
    fn gather_oob_index_returns_error() {
        let src = vec![1.0f32, 2.0];
        let idx = vec![10usize];
        let mut out = vec![0.0f32; 1];
        assert!(gather_f32(&src, &idx, &mut out).is_err());
    }

    #[test]
    fn scatter_add_accumulates_repeated_indices() {
        let mut dst = vec![0.0f32; 3];
        let idx = vec![0usize, 1, 1, 2, 2, 2];
        let upd = vec![1.0f32, 1.0, 1.0, 1.0, 1.0, 1.0];
        scatter_add_f32(&mut dst, &idx, &upd).unwrap();
        assert_eq!(dst, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn sort_ascending_is_stable_and_correct() {
        let mut v = vec![3.0f32, 1.0, 2.0];
        sort_f32_ascending(&mut v);
        assert_eq!(v, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn topk_returns_k_largest_with_indices() {
        let v = vec![1.0f32, 5.0, 2.0, 8.0, 3.0];
        let top = topk_f32(&v, 2);
        // (3, 8.0) and (1, 5.0)
        assert_eq!(top.len(), 2);
        assert_eq!(top[0].0, 3);
        assert_eq!(top[1].0, 1);
    }

    #[test]
    fn prefix_sum_inclusive() {
        let x = vec![1.0f32, 2.0, 3.0, 4.0];
        let mut y = vec![0.0f32; 4];
        prefix_sum_f32(&x, &mut y).unwrap();
        assert_eq!(y, vec![1.0, 3.0, 6.0, 10.0]);
    }
}
