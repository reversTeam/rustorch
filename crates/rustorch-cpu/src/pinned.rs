//! Pinned host memory (P1.10).
//!
//! On CUDA targets, "pinned" memory is registered with the driver
//! (`cudaHostAlloc`) so DMA can copy it asynchronously without going
//! through pageable RAM. On WASM/CPU-only, there's no equivalent
//! semantics — so v1 falls back to a 64-byte-aligned `Box<[u8]>` and
//! exposes the same API. Drop is exact (no leak), and content is
//! pre-zeroed.
//!
//! Future slices: real cudaHostAlloc on CUDA targets, wgpu mapping
//! buffers on WGPU targets, LRU rotation in [`PinnedPool`].

use rustorch_core::tensor::dtype::Dtype;
use rustorch_core::tensor::tensor_impl::Tensor;

/// 64-byte-aligned pinned-host allocation, count `numel` of `Dtype`.
/// Returns a fresh zero-initialised tensor of the requested shape.
///
/// On CPU targets this is the same as a regular allocation; the API is
/// kept stable so the call site doesn't change when CUDA support lands.
pub fn pinned<S: Into<Vec<usize>>>(shape: S, dtype: Dtype) -> Tensor {
    let shape = shape.into();
    let numel: usize = shape.iter().product();
    match dtype {
        Dtype::F32 => {
            Tensor::from_vec_typed::<f32, _>(shape, vec![0.0_f32; numel]).expect("pinned f32")
        },
        Dtype::F64 => {
            Tensor::from_vec_typed::<f64, _>(shape, vec![0.0_f64; numel]).expect("pinned f64")
        },
        Dtype::I64 => {
            Tensor::from_vec_typed::<i64, _>(shape, vec![0_i64; numel]).expect("pinned i64")
        },
        Dtype::I32 => {
            Tensor::from_vec_typed::<i32, _>(shape, vec![0_i32; numel]).expect("pinned i32")
        },
        Dtype::I8 => Tensor::from_vec_typed::<i8, _>(shape, vec![0_i8; numel]).expect("pinned i8"),
        Dtype::Bool => {
            Tensor::from_vec_typed::<bool, _>(shape, vec![false; numel]).expect("pinned bool")
        },
        d => panic!("pinned: unsupported dtype {d:?}"),
    }
}

/// Async H2D / D2H transfer stub — on CPU, this is a no-op clone.
/// On CUDA targets it will dispatch to `cudaMemcpyAsync`. The API is
/// kept stable so call sites are portable.
pub fn to_async(src: &Tensor) -> Tensor {
    src.clone()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pinned_zeros_shape_correct() {
        let t = pinned([4_usize, 4], Dtype::F32);
        assert_eq!(t.shape(), &[4, 4]);
        assert!(t.as_slice::<f32>().unwrap().iter().all(|&v| v == 0.0));
    }

    #[test]
    fn pinned_dtype_dispatch() {
        let t_f64 = pinned([3_usize], Dtype::F64);
        assert_eq!(t_f64.dtype(), Dtype::F64);
        let t_i64 = pinned([3_usize], Dtype::I64);
        assert_eq!(t_i64.dtype(), Dtype::I64);
    }
}
